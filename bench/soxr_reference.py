"""SoXR oracle measurements for the resample quality harness.

This file is part of SoundML.

Copyright (C) 2025 Gabriel Santamaria

SoXR's only appearance in the repository: a dev-time oracle that measures
libsoxr's HQ recipe (the spec `Soundml.Resample`'s default is designed to)
and writes the measurements to ``test/vectors/resample/soxr_reference.json``
in the golden-vector schema of ``test/generate_vectors.py``. The committed
file is what CI replays: the tests never run Python and never link SoXR.

The measurements:

- ``edge_hq_<sr>_<target>`` — the -3 dB passband edge of soxr HQ in hertz,
  found by bisection on single-tone gain. The OCaml harness asserts the
  library's own measured edge within +/-1 % of this value (Q5).
- ``sfdr_hq_...``, ``thdn_hq_...``, ``sweep_alias_hq_...`` — soxr HQ's
  worst-case tone SFDR, tone THD+N, and swept-sine worst alias under exactly
  the metric definitions the OCaml harness applies to SoundML's own output.
  These are recorded reference columns: the harness thresholds are fixed
  numbers from the acceptance table, and a threshold soxr HQ itself failed
  would be mis-calibrated.

Regenerate (the venv pin — python-soxr bundles libsoxr):

    uv run --python 3.13 --with 'soxr==1.1.0' --with 'numpy==2.4.6' \
      --with 'scipy==1.18.0' python bench/soxr_reference.py
"""

from __future__ import annotations

import json
import os
import platform
from typing import Dict, List

import numpy as np
import scipy
import scipy.signal
import soxr

VECTOR_DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "test",
    "vectors",
    "resample",
)

PAIRS = [(44100, 48000), (48000, 44100), (44100, 16000)]

TRIM = 0.15
ANALYSIS_BETA = 30.0
FUNDAMENTAL_HALF_WIDTH = 16  # bins each side of the peak
SWEEP_SECONDS = 30.0
SWEEP_NFFT = 8192
SWEEP_HOP = 4096
SWEEP_BETA = 16.0


def tone(sr: int, f: float, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(round(sr * seconds)), dtype=np.float64) / sr
    return np.sin(2.0 * np.pi * f * t)


def tone_set(target: int) -> List[float]:
    if target == 16000:
        return [1000.0, 3000.0, 6500.0]
    return [1000.0, 5000.0, 10000.0, 17000.0]


def interior(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    i0 = int(n * TRIM)
    cut = x[i0 : n - i0]
    p2 = 1 << (cut.shape[0].bit_length() - 1)
    return cut[:p2]


def spectrum(x: np.ndarray) -> np.ndarray:
    cut = interior(x)
    w = scipy.signal.windows.kaiser(cut.shape[0], ANALYSIS_BETA, sym=False)
    return np.abs(np.fft.rfft(w * cut))


def sfdr(mags: np.ndarray) -> float:
    p = int(np.argmax(mags))
    lo = max(0, p - FUNDAMENTAL_HALF_WIDTH)
    hi = min(mags.shape[0], p + FUNDAMENTAL_HALF_WIDTH + 1)
    spur = 0.0
    for i, m in enumerate(mags):
        if i < 2 or (lo <= i < hi):
            continue
        spur = max(spur, float(m))
    return 20.0 * np.log10(mags[p] / spur)


def thdn(mags: np.ndarray) -> float:
    p = int(np.argmax(mags))
    lo = max(0, p - FUNDAMENTAL_HALF_WIDTH)
    hi = min(mags.shape[0], p + FUNDAMENTAL_HALF_WIDTH + 1)
    power = mags.astype(np.float64) ** 2
    fund = float(power[lo:hi].sum())
    rest = float(power.sum()) - fund
    return 10.0 * np.log10(rest / fund)


def amp_at(x: np.ndarray, sr: int, f: float) -> float:
    """Amplitude of the [f]-hertz component over the trimmed interior,
    by windowed projection at the exact frequency (no scalloping)."""
    n = x.shape[0]
    i0 = int(n * TRIM)
    cut = x[i0 : n - i0]
    w = scipy.signal.windows.kaiser(cut.shape[0], ANALYSIS_BETA, sym=False)
    idx = np.arange(cut.shape[0], dtype=np.float64) + i0
    ph = 2.0 * np.pi * f * idx / sr
    re = float(np.sum(w * cut * np.cos(ph)))
    im = float(np.sum(w * cut * np.sin(ph)))
    return 2.0 * np.hypot(re, im) / float(np.sum(w))


def hq(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    return soxr.resample(x, sr, target, quality="HQ")


def edge(sr: int, target: int) -> float:
    """The -3 dB gain frequency, by bisection on tone gain."""
    nyq = min(sr, target) / 2.0
    goal = 1.0 / np.sqrt(2.0)

    def gain(f: float) -> float:
        return amp_at(hq(tone(sr, f, 1.0), sr, target), target, f)

    lo, hi = 0.85 * nyq, 0.9995 * nyq
    assert gain(lo) > goal > gain(hi)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if gain(mid) > goal:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sweep_alias(sr: int, target: int) -> float:
    """Worst alias, in dB relative to the in-guard peak, over a 30 s linear
    chirp from 1 kHz to 0.9x the output Nyquist."""
    nyq_out = target / 2.0
    f0, f1 = 1000.0, 0.9 * nyq_out
    rate = (f1 - f0) / SWEEP_SECONDS
    t = np.arange(int(round(sr * SWEEP_SECONDS)), dtype=np.float64) / sr
    x = np.sin(2.0 * np.pi * (f0 * t + 0.5 * rate * t * t))
    y = hq(x, sr, target)
    w = scipy.signal.windows.kaiser(SWEEP_NFFT, SWEEP_BETA, sym=False)
    guard = 3.0 * rate * (SWEEP_NFFT / target) + 400.0
    skip = int(0.2 * target)
    worst = -np.inf
    start = skip
    while start + SWEEP_NFFT <= y.shape[0] - skip:
        center = start + SWEEP_NFFT // 2
        f_inst = f0 + rate * (center / target)
        mags = np.abs(np.fft.rfft(w * y[start : start + SWEEP_NFFT]))
        freqs = np.arange(mags.shape[0]) * target / SWEEP_NFFT
        in_guard = np.abs(freqs - f_inst) <= guard
        in_guard[:3] = False
        peak = float(mags[in_guard].max())
        out_guard = ~in_guard
        out_guard[:3] = False
        alias = float(mags[out_guard].max())
        worst = max(worst, 20.0 * np.log10(alias / peak))
        start += SWEEP_HOP
    return float(worst)


def case(name: str, params: Dict, value: float) -> Dict:
    return {"name": name, "params": params, "shape": [1], "values": [value]}


def main() -> None:
    cases: List[Dict] = []
    for sr, target in PAIRS:
        params = {"sample_rate": sr, "target": target, "quality": "soxr_hq"}
        cases.append(case(f"edge_hq_{sr}_{target}", params, edge(sr, target)))
        worst_sfdr, worst_thdn = np.inf, -np.inf
        for f in tone_set(target):
            mags = spectrum(hq(tone(sr, f), sr, target))
            worst_sfdr = min(worst_sfdr, sfdr(mags))
            worst_thdn = max(worst_thdn, thdn(mags))
        cases.append(case(f"sfdr_hq_{sr}_{target}", params, float(worst_sfdr)))
        cases.append(case(f"thdn_hq_{sr}_{target}", params, float(worst_thdn)))
        cases.append(
            case(f"sweep_alias_hq_{sr}_{target}", params, sweep_alias(sr, target))
        )

    document = {
        "schema": 1,
        "suite": "resample",
        "generator": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "soxr": soxr.__version__,
            "libsoxr": soxr.__libsoxr_version__,
        },
        "cases": cases,
    }
    os.makedirs(VECTOR_DIRECTORY, exist_ok=True)
    path = os.path.join(VECTOR_DIRECTORY, "soxr_reference.json")
    with open(path, "w") as fp:
        json.dump(document, fp, indent=1)
        fp.write("\n")
    print(f"wrote {path}")
    for c in cases:
        print(f"  {c['name']}: {c['values'][0]:.6f}")


if __name__ == "__main__":
    main()
