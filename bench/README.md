# SoundML Benchmarks

Thumper benchmark suite for SoundML, with a librosa cross-reference runner.
Benches are performance regression tests: each case is judged against the
committed `soundml.thumper` baseline, and the suite-level budgets fail the
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
  `Config.create`, priced separately: the OLS plans design far smaller
  banks, so creation costs a fraction of the one-second conversion it
  configures (0.1-0.6x) — build once and reuse is still the documented
  contract. The Python twin has the fair-fight rows: librosa `soxr_hq` (the
  spec the default preset is designed to) and torchaudio at its published
  `kaiser_best` triple and its defaults.

  The measured position, both sides on the maintainer machine in one
  session (arm64, min-of-N, python-soxr 1.1.0 driving the maintained
  libsoxr fork): at equal spec on 30-second clips at float32, `soxr_hq`
  converts 2.0-3.0x faster (44.1 ↔ 48 kHz about 3.0x/2.5x, 44.1 → 16 kHz
  2.9x, 48 → 8 kHz 2.0x — previously 3.5-4.7x). The planner now executes
  its sharp stages by the same overlap-save FFT convolution libsoxr uses,
  through nx's transforms; the residual gap is the FFT execution rate
  (libsoxr's hand-vectorized single-precision butterflies against nx's
  double-interior transforms), an upstream lever, not an architectural
  one. The float64 rows are not an equal-precision comparison: soxr HQ
  runs single-precision internally regardless of I/O dtype, while SoundML
  float64 is a genuine double-precision path — and the FFT executor's
  double interior is native there, so float64 now runs within a few
  percent of float32 (2.5-3.2x from soxr's single-precision engine).
  On one-second clips every FFT-executed pair pays its block-transform
  dispatch at float32: 1.3-2x slower than the retired all-FIR executor
  there, worst on the 48 → 8 kHz `Fast and the near-unity pairs; the
  float64 rows pay it too under the thumper batch protocol (flat to
  1.4x, the `Best rows already 1.2-1.3x faster), while single-shot
  profile runs measure one-second float64 at or above the retired
  kernel — the crossover to the FFT win sits between one and thirty
  seconds. The streaming surface pays a second, related
  price at float32: at chunks of a few thousand samples the near-unity
  and /F-last FFT-executed pairs stream 1.2-1.6x slower than the pre-FFT
  kernel (float64 streaming improves everywhere, and sub-64-sample
  chunks — dispatch-dominated on both executors — do not regress),
  because a step stacks only the few blocks its chunk completes where
  the offline call stacks hundreds of transform lines. Streaming
  throughput is also not monotonic in chunk size (the committed
  16 384-chunk row runs slower than the 4 096 one on the near-unity
  pair): a few-line stacked-transform inefficiency in the upstream
  transforms, a known upstream item, not a SoundML-side lever. The
  baseline records all of these as the documented price of the plan.
  SoundML runs 1.2-3x faster than
  torchaudio at its published `kaiser_best` settings and behind
  torchaudio's faster, lower-spec defaults. None of the references
  carries the bit-exact partition-invariance law that is this resampler's
  defining contract, and none exposes an incremental kernel with exact
  rational latency accounting.

- `io` — `Soundml_io` decode and encode over a deterministic corpus the
  suite generates with soundml-io's own writer (`bench/io_corpus.ml`; the C
  ceiling harness and the Python cross-reference read the identical bytes).
  Rows: whole-file `read` per format family at 1 s mono, 30 s stereo and
  30 s mono; the `Reader.read ?out` form (open, one full-file chunk into a
  lent destination, close — the C ceiling's own shape); `info`; the
  many-small-files ingest loop in both the allocating form (librosa's shape)
  and the reused-destination form (the C ceiling's shape); chunked writes;
  and the fused `read ~sample_rate` next to its offline decomposition
  (native read plus `Resample.apply`, config built per call on both sides).

  The measured position (all min-of-N, warm cache, quiet reference host —
  Apple M4 Pro, libsndfile 1.2.2, soundfile 0.14.0, librosa 0.11.0 — over
  the shared corpus in one session; re-derive the ceilings with
  `bench/run_io_ceiling.sh` before porting any multiplier to another
  machine):

  - **Small reads (1 s files)**: per-file wall 0.95–1.07x the C ceiling on
    every family, both sessions (budget 1.25x) — 2.2–3.3x faster than
    `librosa.load(sr=None, mono=False)` on WAV PCM, 1.7x on wav-float32,
    1.4–1.5x on FLAC, 1.1x on Ogg.
  - **Bulk reads (30 s files)**: at the C ceiling on FLAC (1.02x), Ogg
    (1.01x) and every mono cell (0.95–1.10x). The stereo WAV cells pay the
    planar-materialization pass — `data` is C-contiguous
    `[channels; frames]` by contract, and transposing libsndfile's
    interleaved delivery costs ~0.2–0.3 ms per 30 s stereo file (~53 GB/s
    effective; NEON `ld2` after the channel-2 specialization in the stubs)
    — landing 1.19x (pcm16) to 1.73x (float32, where the baseline is pure
    memcpy) over the interleaved-destination C ceiling. The comparison that
    holds the layout fixed: librosa's channel-first result is a *strided
    view* of the interleaved buffer; materializing it
    (`np.ascontiguousarray`) costs python 4.26 ms (pcm16) / 1.03 ms
    (float32) against our 1.24 ms / 0.51 ms — 2.0–3.3x in our favor on the
    same delivered layout.
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
    stable WAV-write multiplier exists here; measured ratios scatter
    0.67–1.79x around the no-clipping C ceiling with no consistent
    direction. Two structural facts hold regardless: `SFC_SET_CLIPPING` —
    the pinned saturation policy, byte-compared against python-soundfile
    by the goldens — costs +27% on the float-to-PCM conversion (measured
    directly with a C probe), and FLAC/Ogg writes sit at parity with
    python-soundfile.
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
  numbers they derive from (`bench/profile/gate_io.ml` re-measures every
  gate cell as CSV; `bench/run_io_ceiling.sh` re-derives the C side;
  `bench/bench_soundml_io.py` re-runs the Python side).

`bench/soxr_reference.py` is not a benchmark: it is the dev-time SoXR oracle
that regenerates the committed quality-harness vectors under
`test/vectors/resample/` (its header documents the pinned invocation). It is
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
dune exec bench/bench_soundml.exe
```

### Librosa (Python)

```bash
uv run --python 3.12 --with 'numpy<2.3' --with 'librosa==0.11.0' \
  python bench/bench_soundml.py
```

Add `--with torch --with torchaudio` for the torchaudio resample rows; they
are skipped (with a note) when torch is not installed.

(or plain `python bench/bench_soundml.py` in any environment with `numpy` and
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
dune exec bench/bench_soundml.exe -- bless
mv bench/soundml.thumper.corrected bench/soundml.thumper
```

(`bless` refuses to record on a loaded host; close the noisy neighbours or
pass `--force`.) Commit the result only when the machine is meant to be a
reference for CI or for review.
