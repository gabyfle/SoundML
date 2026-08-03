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
