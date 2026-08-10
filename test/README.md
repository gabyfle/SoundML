# About SoundML Testing

[librosa](https://librosa.org/) 0.11 is the reference implementation for the
algorithms inside SoundML: every numerical module is tested for parity
against golden vectors generated from it (librosa delegates window
computation to scipy, so those vectors are equally scipy's).

## Golden vectors

The golden vectors live beside the suite that replays them —
`soundml/test/<suite>/vectors/*.json` and `soundml-io/test/vectors/*.json` —
and are
**committed to the repository**: running the test suite requires no Python.
Each file records the exact reference versions it was generated with, and
each case carries its generator parameters, the expected shape, and the
expected float64 values (JSON floats round-trip float64 exactly).

The OCaml side of the harness is the `tutils` library in `test/support/`:
`Tutils.Golden` loads the files, and `Tutils.check_close` asserts
elementwise parity with per-dtype tolerances — float64 near-exact
(`1e-12` relative), float32 at `1e-6` relative. New modules (mel, STFT,
...) plug into the same pattern: add a generator class to
`dev/generate_vectors.py`, commit the vectors it writes, and load them through
`Tutils.Golden` from a Windtrap suite.

Current suites:

- `windows` — every `Soundml.Window` type, periodic and symmetric, at odd
  and even lengths including the 1/2/3-point edge cases, plus
  `scipy.signal.check_COLA` truths for `Window.cola` (`cola.json`).
- `stft` — `librosa.stft` magnitude and power spectra over a deterministic
  LCG signal (reproduced bit-exactly on the OCaml side), covering
  `(fft_size, hop)` combos x centered/left alignment x odd/even lengths x
  float64/float32, plus `librosa.fft_frequencies` and
  `librosa.frames_to_time` truths for `Stft.frequencies` and `Stft.times`
  (`coordinates.json`).
- `istft` — the synthesis side of `Soundml.Stft`: `librosa.istft` over
  deterministic synthetic spectra (two LCG streams, one per complex
  component, with the imaginary DC and Nyquist bins zeroed so the inverse
  real transform discards nothing) across `(fft_size, hop, win_length)`
  combos x all three alignments x frame counts x float64/float32, plus an
  explicit output length above, below and at the natural one and one a hop
  and a sample long, which leaves whole frames unread (`lengths.json`), and
  `librosa.griffinlim` at `init=None` — its only deterministic initial
  phase — across iteration counts and momenta including the momentum-free
  classic algorithm. The spectra are inconsistent on purpose, so the cases
  measure the least-squares solution rather than a round trip; the round
  trip, the padding and normalisation grids, the frames a short length
  never reads and the Griffin-Lim convergence gates are properties, checked
  without an oracle.
  The `right` alignment has no librosa counterpart and is generated as its
  `center=False` synthesis with the `fft_size - 1` left-extension positions
  dropped.
- `db` — the `Soundml.Convert` decibel conversions (`power_to_db`,
  `amplitude_to_db`) over synthetic seeded inputs: fixed references, the
  `amin` floor, `top_db` clamping (global across a rank-two input),
  negative entries, both element dtypes. Each case stores its input in the
  params, flattened in C order.
- `mel` — `librosa.filters.mel` weight matrices (Slaney and HTK scales x
  slaney/no-op norms over several `(n_mels, fft_size)` geometries,
  including a nonzero `fmin` and an `fmax` pinned at Nyquist), plus
  `librosa.feature.melspectrogram` and `librosa.feature.mfcc` end-to-end
  over the LCG signal — `n_mfcc` 13 and 20, lifter 0 and 22, both element
  dtypes, and a decaying-envelope case that drives the log-mel through
  librosa's `top_db` clamp and `amin` floor.
- `features_spectral` — the flat spectral-shape features
  (`spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`,
  `spectral_flatness`) against `librosa.feature.*` over |LCG| magnitude
  spectrograms reproduced bit-exactly (several geometries, a batched
  rank-three input, a custom non-uniform frequency grid, both element
  dtypes) and end-to-end from the LCG signal through
  `Stft.power_spectrum ~power:1.`.
- `features_energy` — the flat energy features: `librosa.feature.rms` over
  audio (constant-zero centered padding) and over synthetic magnitude
  spectrograms (the `S=` path, halved DC/Nyquist bins), and
  `librosa.feature.zero_crossing_rate` (edge-copy centered padding) with
  explicit thresholds 0 and 0.5 beside the 1e-10 default — the librosa
  defaults and small `(frame_length, hop)` geometries including an odd
  frame length and `hop > frame_length`, a stereo case, both element
  dtypes.
- `features_onset` — `librosa.feature.spectral_contrast` over LCG magnitude
  spectrograms (default and large quantiles pinning numpy's half-to-even
  band sizing, linear and logarithmic differences, a silence tail driving
  the `amin` floor and the global `top_db` clamp) and
  `librosa.onset.onset_strength` end-to-end on its default log-power-mel
  chain (lags 1-3, centered and left alignment, degenerate short signals,
  the >80 dB envelope), both element dtypes.
- `io` — decode parity for `Soundml_io` against python-soundfile (the io
  vector files record the soundfile and bundled-libsndfile versions beside
  the base stack) over the committed fixtures in `test/io/corpus/`, which
  the same generator class writes: float64 decodes stored planar
  `[channels; frames]`, sample-exact for lossless cells — the float32
  expectation is the correctly-rounded float32 cast, an equality the
  generator asserts against python-soundfile's own float32 decode for every
  fixture — and Ogg/Vorbis at the measured cross-stack noise ceiling; plus
  the write-clipping golden (SFC_SET_CLIPPING saturation, not wraparound).
  The generator also constructs the malformed-input corpus under
  `test/io/corpus/malformed/`, seeds and offsets pinned in its MANIFEST.
- `resample` — the one deliberate exception to librosa bit-parity:
  `Soundml.Resample` targets soxr HQ's published specification with its own
  bits, so its harness asserts measured decibel thresholds, never vector
  parity. The committed file is SoXR oracle *measurements* (the -3 dB
  passband edge the harness tracks within 1 %, plus recorded
  SFDR/THD+N/sweep calibration columns), generated by
  `bench/soxr_reference.py` — SoXR's only appearance in the repository —
  rather than by `generate_vectors.py`.

## Regenerating the vectors

Rerun `dev/generate_vectors.py` only to regenerate the goldens, from an
environment with the pinned reference versions:

```sh
cd dev
python3 -m venv .venv
.venv/bin/pip install librosa==0.11.0 soundfile
.venv/bin/python generate_vectors.py
```

The SoXR quality oracle (`soundml/test/resample/vectors/soxr_reference.json`)
is regenerated separately by `dev/soxr_reference.py` (needs `soxr`).

The script refuses to run against any librosa other than 0.11.x and stamps
the exact `python`/`numpy`/`scipy`/`librosa` versions into every file it
writes.

## Audio fixtures

The committed io corpus (`soundml-io/test/corpus/`, malformed variants
pinned by its `MANIFEST`) is written by the io generator class in
`dev/generate_vectors.py`.
