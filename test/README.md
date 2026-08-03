# About SoundML Testing

[librosa](https://librosa.org/) 0.11 is the reference implementation for the
algorithms inside SoundML: every numerical module is tested for parity
against golden vectors generated from it (librosa delegates window
computation to scipy, so those vectors are equally scipy's).

## Golden vectors

The golden vectors live under `test/vectors/<suite>/*.json` and are
**committed to the repository**: running the test suite requires no Python.
Each file records the exact reference versions it was generated with, and
each case carries its generator parameters, the expected shape, and the
expected float64 values (JSON floats round-trip float64 exactly).

The OCaml side of the harness is the `tutils` library in `test/support/`:
`Tutils.Golden` loads the files, and `Tutils.check_close` asserts
elementwise parity with per-dtype tolerances — float64 near-exact
(`1e-12` relative), float32 at `1e-6` relative. New modules (mel, STFT,
...) plug into the same pattern: add a generator class to
`generate_vectors.py`, commit the vectors it writes, and load them through
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
- `db` — the `Soundml.Convert` decibel conversions (`power_to_db`,
  `amplitude_to_db`) over synthetic seeded inputs: fixed references, the
  `amin` floor, `top_db` clamping (global across a rank-two input),
  negative entries, both element dtypes. Each case stores its input in the
  params, flattened in C order.

## Regenerating the vectors

Rerun `generate_vectors.py` only to regenerate the goldens, from an
environment with the pinned reference versions:

```sh
cd test
python3 -m venv .venv
.venv/bin/pip install librosa==0.11.0
.venv/bin/python generate_vectors.py
```

The script refuses to run against any librosa other than 0.11.x and stamps
the exact `python`/`numpy`/`scipy`/`librosa` versions into every file it
writes.

## Audio fixtures

`generate_audio.sh` synthesizes small audio files with FFmpeg for the I/O
tests that need real files on disk.
