# SoundML Benchmarks

The io rows live in `soundml-io/bench/` (`bench_io.ml`, `soundml_io.thumper`)
with the same budgets and blessing procedure; everything below applies to
both suites.

Thumper benchmark suite for SoundML, with a librosa cross-reference runner.
Benches are performance regression tests: each case is judged against the
committed `soundml.thumper` / `soundml_io.thumper` baselines, and the suite-level budgets fail the
build on a confirmed regression (wall time beyond 5%, allocations beyond 1%).

## Suites

- `window` — `Window.make` for every window spec at n = 2048, one row per
  spec, mirroring the librosa rows of `bench_soundml.py` one to one.
- `pipeline` — the cost of the `Pipeline` abstraction: `Pipeline.run` of a
  three-stage stateless toy chain (gain, bias, rectify) over a
  one-million-sample float32 tensor, next to the hand-written sequence of the
  same three Nx calls. The two rows must stay within a few percent of each
  other; the baseline pins both so either drifting fails the build.
- `stft` — `Stft.power_spectrum` offline over 30 s of mono audio at
  22.05 kHz, fft 2048 and hop 512, one row per dtype (float32, float64),
  mirroring the librosa rows one to one.
- `mel` — `mel_spectrogram` of the same signal through a 128-band filterbank,
  one row per dtype, mirroring the librosa rows one to one.
- `resample` — every face of `Resample` on one-second mono clips: `apply`
  across presets, rate pairs and dtypes; the GEMM surface next to the
  executor; the streaming kernel across chunk sizes (the 1024 row keeps the
  per-chunk dispatch overhead honest); the resample-then-STFT stage next to
  the same computation hand-written; the identity-rate passthrough; and
  `Config.create`, priced separately: a phase-rich plan designs one long
  prototype, so creation costs several times the one-second conversion it
  configures (4-13x) — build once and reuse is the documented
  contract. The Python twin has the fair-fight rows: librosa `soxr_hq` (the
  spec the default preset is designed to) and torchaudio at its published
  `kaiser_best` triple and its defaults.

  The measured position, both sides on the maintainer machine in one
  session (arm64, min-of-N over six interleaved alternations,
  python-soxr 1.1.0 driving the maintained libsoxr fork): at float32
  SoundML converts faster at both clip lengths — 0.44x `soxr_hq` on
  one-second 44.1 → 48 kHz and 0.42x on 30-second, 0.52x and 0.61x at
  44.1 → 16 kHz. At float64 the near-unity pair is ahead as well (0.89x
  one-second, 0.83x 30-second), and 44.1 → 16 kHz runs 1.22x at both
  lengths. The phase-rich plans run as one banded matrix product per
  fixed group of phase cycles, through the platform BLAS, in the
  kernel's own precision; the remaining plans execute their sharp stage
  by the same overlap-save FFT convolution libsoxr uses, through nx's
  transforms, with the spectrum shaping between the forward and inverse
  transforms (periodic tiling with its Hermitian mirror, the alias fold,
  the filter product) as one fused pass per transform line, outputs
  written in place into the preallocated result, and stacked transform
  lines spread across cores in proportion to per-line transform work.
  The residual float64 gap on the wide ratio is arithmetic: the matrix
  product spends 1 + M/T times an output's taps on its window and runs
  at the platform's double-precision GEMM rate, where libsoxr splits the
  same conversion into stages of far fewer multiplies per output. The
  float64 rows are not an equal-precision comparison either: soxr HQ
  runs single-precision internally regardless of I/O dtype, while
  SoundML float64 is a genuine double-precision path, which is why the
  float32 and float64 columns separate here and barely move for soxr.
  Streaming through the matrix product improves with chunk size on the
  near-unity pair (the 16 384-chunk row is the fastest of the three);
  the baseline records it. SoundML runs faster than torchaudio at its published
  `kaiser_best` settings and behind torchaudio's faster, lower-spec
  defaults. None of the references carries the bit-exact
  partition-invariance law that is this resampler's defining contract,
  and none exposes an incremental kernel with exact rational latency
  accounting.

- `io` — `Soundml_io` decode and encode over a deterministic corpus the
  suite generates with soundml-io's own writer (`soundml-io/bench/io_corpus.ml`; the C
  ceiling harness and the Python cross-reference read the identical bytes).
  Rows: whole-file `read` per format family at 1 s mono, 30 s stereo and
  30 s mono; the `Reader.read ?out` form (open, one full-file chunk into a
  lent destination, close — the C ceiling's own shape); `info`; the
  many-small-files ingest loop in both the allocating form (librosa's shape)
  and the reused-destination form (the C ceiling's shape); chunked writes;
  and the fused `read ~sample_rate` next to its offline decomposition
  (native read plus `Resample.apply`, config built per call on both sides).

  The measured position (all min-of-N, warm cache, reference host — Apple
  M4 Pro, libsndfile 1.2.2, soundfile 0.14.0, librosa 0.11.0 — over the
  shared corpus; every multiplier is same-session, the parties alternating
  round by round, and a range spans the sessions it was recorded in.
  Re-derive the ceilings with `dev/run_io_ceiling.sh` before porting any
  multiplier to another machine):

  - **Small reads (1 s files)**: per-file wall 0.95–1.07x the C ceiling on
    every family, both sessions (budget 1.25x) — 2.2–3.3x faster than
    `librosa.load(sr=None, mono=False)` on WAV PCM, 1.7x on wav-float32,
    1.4–1.5x on FLAC, 1.1x on Ogg.
  - **Bulk reads (30 s files)**: at the C ceiling on FLAC (1.01x), Ogg
    (1.00x) and the PCM mono cells (0.98–1.01x). Two cells sit above it,
    both for reasons outside the decode itself.

    The stereo WAV cells pay the planar-materialization pass — `data` is
    C-contiguous `[channels; frames]` by contract, and splitting
    libsndfile's interleaved delivery costs 0.14–0.19 ms per 30 s stereo
    file (arm64 `ld2` lane loads feeding non-temporal paired stores to the
    two channel cursors, so the staging block survives the pass) — landing
    1.10x (pcm24) to 1.58x (float32, where the baseline is pure memcpy)
    over the interleaved-destination C ceiling. Neither Python library
    pays that pass: `soundfile.read` hands back the interleaved
    `(frames, channels)` block libsndfile wrote and `librosa.load` a
    strided `.T` view of it, no copy. So the wav-float32 stereo cell reads
    1.41–1.50x the same-session `soundfile.read` minimum (0.52–0.53 ms
    against 0.35–0.37 ms, minima over ten and eight interleaved
    alternations), and the whole distance is the layout pass. Held against
    the delivered layout instead, the ordering reverses: materializing the
    view (`np.ascontiguousarray`, the `librosa_contig` rows of the Python
    twin) costs 1.16–1.17 ms on the same files, 2.2x our figure.

    The wav-float32 30 s mono cell reads 1.17–1.21x its allocating C
    ceiling (`read_alloc`: 0.167–0.171 ms against 0.141–0.143 ms) while
    the same decode into a lent destination (`reader_out`) is 1.04x. The
    difference is the result tensor's first touch: a fresh 5.3 MB bigarray
    is a fresh mapping and the kernel zero-fills its pages under the
    decoder's `read(2)` (~0.025 ms), where a C or CPython caller gets the
    previous buffer's warm pages back from the allocator. It is visible
    only here — every other 30 s cell decodes slowly enough to absorb it.
    `soundfile.read` allocates too and pays part of the same cost
    (1.12–1.19x that ceiling), so the two land close: the cell measures
    1.5–6.5% above the same-session `soundfile.read` minimum. The margin
    is small enough to sit inside this host's run-to-run spread, but it
    keeps its sign across sessions — this is the one bulk cell where the
    allocating form does not reach python-soundfile.
  - **Many small files (1000 x 1 s, total wall)**: the reused-destination
    loop runs 1.00–1.06x the C many-files ceiling (budget 1.15x, both
    sessions); the allocating loop runs 1.07–1.28x (OCaml GC churn on
    88 KB tensors). Its librosa multiple is librosa-side session-sensitive:
    the recording session measured 2.06x (wav-pcm16), 1.37x (wav-float32),
    1.35x (FLAC); an independent verification session on the same host
    (fresh corpus, adjacent min-of-12 runs) measured 1.85x / 1.30x / 1.33x
    — librosa's loop itself moved 70.2–76.7 ms between sessions, so the
    wav-pcm16 cell dips below its 1.9x floor when librosa runs fast
    (f32/FLAC hold their 1.3x / 1.1x floors in both). The form the C gate
    attaches to — the reused-destination loop — passes its budget in every
    session.
  - **Header probes**: `info` at 1.00–1.14x the C open ceiling (budget
    1.2x) on every family, both sessions.
  - **Writes**: FLAC 0.95–1.07x and Ogg 0.99–1.00x the C chunked-write
    ceiling across both sessions (encoder-bound; parity is the claim,
    budget 1.1x). The WAV cells are disk-noise-dominated on this host (the
    reference volume ran at 98% capacity in both sessions): write minima
    swing 2–3x between adjacent runs of the identical binary on every
    party — ours, the C ceiling and python-soundfile alike (e.g. our
    float32 cell measured 4.2 and 7.6 ms in back-to-back min-of-12 runs;
    the C pcm16 ceiling moved 8.3 → 13.4 ms between sessions) — so no
    absolute WAV-write figure is portable, and a ratio counts only when
    both parties are measured in one session, alternating; ratios taken
    from adjacent runs instead scatter 0.67–1.79x around the no-clipping
    C ceiling with no consistent direction. Three structural facts hold
    regardless: `SFC_SET_CLIPPING` — the pinned saturation policy,
    byte-compared against python-soundfile by the goldens — costs +27% on
    the float-to-PCM conversion (measured directly with a C probe);
    FLAC/Ogg writes sit at parity with python-soundfile; and the stereo
    cells carry the mirror of the read
    side's materialization pass, since libsndfile takes interleaved input
    and `data` is planar — 0.20–0.25 ms per 30 s stereo file, measured in
    C as the whole distance between a staged planar write and the
    interleaved-source ceiling. That pass is where the WAV float32 cell's
    distance from python-soundfile lives (python's input is already
    interleaved): the cell runs 1.05x the same-session interleaved-source
    C ceiling in both sessions — the encode itself is at the ceiling — and
    1.11–1.16x the same-session `soundfile.write` minimum. It is the one
    write cell that does not reach python-soundfile, and no arrangement of
    the staging closes it while the input stays planar.
  - **Fused resampled read**: `read ~sample_rate` within 1.004x (44.1→16 k)
    and 1.045x (44.1→48 k) of reading natively and applying
    `Resample.apply` afterwards (budget 1.05x, min-of-N; the thumper
    median-of-batches statistic reads 1.07x/1.03x — per-call kernel-state
    allocation shows up in means), while never materializing the
    native-rate signal. The fused path feeds the kernel multi-hundred-
    kiloframe blocks: at the L2-resident 32768-frame staging the FFT-
    executed plans run 1.5x their offline decomposition (few transform
    lines per step), and the block escalation recovers all of it.

  The rows are regression-ratcheted like every other group; the verdicts
  above are measurement evidence for this host, recorded next to the
  numbers they derive from (`dev/profile/gate_io.ml` re-measures every
  gate cell as CSV; `dev/run_io_ceiling.sh` re-derives the C side;
  `soundml-io/bench/bench_soundml_io.py` re-runs the Python side).

`dev/soxr_reference.py` is not a benchmark: it is the dev-time SoXR oracle
that regenerates the committed quality-harness vectors under
`soundml/test/resample/vectors/` (its header documents the pinned invocation). It is
SoXR's only appearance in the repository.

## Running the benchmarks

### SoundML (OCaml)

```bash
dune build @bench
```

runs the suite under `--quick` against the committed baseline and diffs the
proposed `soundml.thumper.corrected` (promote it with `dune promote` when a
confirmed improvement ratchets the baseline). For a full-precision run and the
report only:

```bash
dune exec soundml/bench/bench_soundml.exe
```

### Librosa (Python)

```bash
uv run --python 3.12 --with 'numpy<2.3' --with 'librosa==0.11.0' \
  python soundml/bench/bench_soundml.py
```

Add `--with torch --with torchaudio` for the torchaudio resample rows; they
are skipped (with a note) when torch is not installed.

(or plain `python soundml/bench/bench_soundml.py` in any environment with `numpy` and
`librosa` — the versions CI installs are in `.github/workflows/test.yml`;
librosa's numba dependency needs Python <= 3.12 and numpy < 2.3). The
window-generation rows are the seed; STFT and mel rows land together with
their OCaml counterparts.

## The baseline

`soundml.thumper` stores measurement evidence keyed by machine
(`host:` in the file header): verdicts only ever compare runs from the same
machine, so the committed section is specific to the maintainer machine that
recorded it. On any other machine the suite reports NEW rows (exit code 0)
and proposes a section for that machine as `soundml.thumper.corrected`.

To re-record the baseline for your machine:

```bash
dune exec soundml/bench/bench_soundml.exe -- bless
mv soundml/bench/soundml.thumper.corrected soundml/bench/soundml.thumper
```

(`bless` refuses to record on a loaded host; close the noisy neighbours or
pass `--force`.) Commit the result only when the machine is meant to be a
reference for CI or for review.
