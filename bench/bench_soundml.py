"""Librosa cross-reference benchmarks for SoundML.

The Python twin of ``bench_soundml.ml``: every SoundML case with a librosa (or
scipy) reference implementation gets a row here under the same group/name
scheme, so the two tables read side by side. Window generation seeds the
suite; STFT and mel rows land together with their OCaml counterparts.

Run it from the repository root:

    python bench/bench_soundml.py
"""

from __future__ import annotations

import time
from typing import Any, Callable, List, Tuple

import librosa
import numpy as np

N_WINDOW = 2048

# The SoundML window specs and their librosa/scipy spellings. Parametrised
# specs use representative parameters; keep them in sync with the OCaml suite.
WINDOW_SPECS: List[Tuple[str, Any]] = [
    ("hann", "hann"),
    ("hamming", "hamming"),
    ("blackman", "blackman"),
    ("blackman_harris", "blackmanharris"),
    ("nuttall", "nuttall"),
    ("bartlett", "bartlett"),
    ("kaiser 8.6", ("kaiser", 8.6)),
    ("gaussian 256", ("gaussian", 256.0)),
    ("tukey 0.5", ("tukey", 0.5)),
    ("flat_top", "flattop"),
    ("rectangular", "boxcar"),
]


def bench(name: str, fn: Callable[[], Any]) -> Tuple[str, Callable[[], Any]]:
    return (name, fn)


def build_benchmarks() -> List[Tuple[str, Callable[[], Any]]]:
    benches: List[Tuple[str, Callable[[], Any]]] = []

    for name, spec in WINDOW_SPECS:
        def make(spec: Any) -> Callable[[], Any]:
            def run() -> None:
                w = librosa.filters.get_window(spec, N_WINDOW, fftbins=True)
                float(w.sum())

            return run

        benches.append(bench(f"window/{name} {N_WINDOW} (librosa)", make(spec)))

    return benches


def measure(
    fn: Callable[[], Any],
    *,
    warmup: int = 3,
    batches: int = 7,
    min_batch_seconds: float = 0.02,
) -> float:
    """Median wall-clock seconds per call, over ``batches`` timed batches.

    The batch size is calibrated by doubling until one batch takes at least
    ``min_batch_seconds``, so per-call durations stay resolvable for
    microsecond-scale cases.
    """
    for _ in range(warmup):
        fn()
    calls = 1
    while True:
        start = time.perf_counter()
        for _ in range(calls):
            fn()
        elapsed = time.perf_counter() - start
        if elapsed >= min_batch_seconds:
            break
        calls *= 2
    samples = []
    for _ in range(batches):
        start = time.perf_counter()
        for _ in range(calls):
            fn()
        samples.append((time.perf_counter() - start) / calls)
    return float(np.median(samples))


def format_seconds(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f}ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f}us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f}ms"
    return f"{seconds:.2f}s"


def main() -> None:
    benchmarks = build_benchmarks()
    width = max(len(name) for name, _ in benchmarks)
    print(f"{'Name':<{width}}  Wall/Run")
    print("-" * (width + 10))
    for name, fn in benchmarks:
        wall = measure(fn)
        print(f"{name:<{width}}  {format_seconds(wall):>8}")


if __name__ == "__main__":
    main()
