"""
This file is part of SoundML.

Copyright (C) 2025 Gabriel Santamaria

This script generates the golden test vectors committed under test/vectors/.
The reference implementation is librosa 0.11 (which delegates window
computation to scipy.signal); every generated file records the exact
versions it was produced with.

The vectors are committed to the repository: CI never runs this script and
never needs Python. Rerun it only to regenerate the goldens, from an
environment with the pinned reference versions:

    python3 -m venv .venv
    .venv/bin/pip install librosa==0.11.0
    .venv/bin/python generate_vectors.py

Each suite writes small JSON files under test/vectors/<suite>/. A file is

    {
      "schema": 1,
      "suite": "<suite name>",
      "generator": {"python": ..., "numpy": ..., "scipy": ..., "librosa": ...},
      "cases": [
        {"name": ..., "params": {...}, "shape": [...], "values": [...]},
        ...
      ]
    }

with "values" the expected float64 result flattened in C order ("shape"
gives its shape back). JSON floats round-trip float64 exactly. The OCaml
side of the harness is test/support/tutils.ml; new suites (mel, stft, ...)
add a generator class here and load their files through the same module.
"""

import json
import os
import platform

import librosa
import numpy as np
import scipy
import scipy.signal

VECTOR_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors")

GENERATOR_VERSIONS = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "librosa": librosa.__version__,
}

assert librosa.__version__.startswith(
    "0.11."
), f"librosa 0.11 is the pinned reference, found {librosa.__version__}"


def write_suite(suite: str, filename: str, cases: list):
    """Write one golden file for `suite`, pretty-printed with the values of
    each case kept on a single line."""
    directory = os.path.join(VECTOR_DIRECTORY, suite)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{filename}.json")
    rendered = []
    for case in cases:
        fields = [f'"{key}": {json.dumps(value)}' for key, value in case.items()]
        rendered.append("  {" + ", ".join(fields) + "}")
    body = ",\n".join(rendered)
    generator = json.dumps(GENERATOR_VERSIONS)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f'{{\n"schema": 1,\n"suite": "{suite}",\n'
            f'"generator": {generator},\n'
            f'"cases": [\n{body}\n]\n}}\n'
        )
    print(f"wrote {path} ({len(cases)} cases)")


class WindowVectorGenerator:
    """Golden vectors for Soundml.Window.

    Covers every window family x {periodic, symmetric} x lengths (odd and
    even, including the n = 1, 2, 3 edge cases), cross-checking that
    librosa.filters.get_window and scipy.signal.get_window agree exactly
    before emitting, plus scipy.signal.check_COLA truths for Window.cola.
    """

    SUITE = "windows"

    LENGTHS = [1, 2, 3, 8, 15, 16, 17, 64, 128]

    # family -> list of (case key, scipy window spec, extra JSON params)
    FAMILIES = {
        "hann": [("hann", "hann", {})],
        "hamming": [("hamming", "hamming", {})],
        "blackman": [("blackman", "blackman", {})],
        "blackman_harris": [("blackman_harris", "blackmanharris", {})],
        "nuttall": [("nuttall", "nuttall", {})],
        "bartlett": [("bartlett", "bartlett", {})],
        "flat_top": [("flat_top", "flattop", {})],
        "rectangular": [("rectangular", "boxcar", {})],
        "kaiser": [
            (f"kaiser_b{beta:g}", ("kaiser", beta), {"beta": beta})
            for beta in (5.0, 8.6, 14.0)
        ],
        "gaussian": [
            (f"gaussian_s{std:g}", ("gaussian", std), {"std": std})
            for std in (2.5, 7.0)
        ],
        "tukey": [
            (f"tukey_a{taper:g}", ("tukey", taper), {"taper": taper})
            for taper in (0.0, 0.25, 0.5, 1.0)
        ],
    }

    # (window family, spec index into FAMILIES[family], length, hop)
    COLA_CASES = [
        ("hann", 0, 1024, 512),
        ("hann", 0, 1024, 256),
        ("hann", 0, 1024, 341),
        ("hann", 0, 1023, 341),
        ("hamming", 0, 1024, 512),
        ("blackman", 0, 1024, 512),
        ("blackman", 0, 1024, 256),
        ("blackman_harris", 0, 1024, 512),
        ("blackman_harris", 0, 1024, 256),
        ("nuttall", 0, 1024, 256),
        ("flat_top", 0, 1024, 512),
        ("flat_top", 0, 1024, 128),
        ("bartlett", 0, 1024, 512),
        ("rectangular", 0, 1024, 1024),
        ("rectangular", 0, 1024, 512),
        ("rectangular", 0, 100, 30),
        ("kaiser", 1, 1024, 512),
        ("gaussian", 0, 1024, 256),
        ("tukey", 2, 1024, 512),
    ]

    def generate(self):
        for family, specs in self.FAMILIES.items():
            cases = []
            for key, window, extra in specs:
                for periodic in (True, False):
                    mode = "periodic" if periodic else "symmetric"
                    for n in self.LENGTHS:
                        expected = scipy.signal.get_window(window, n, fftbins=periodic)
                        cross = librosa.filters.get_window(window, n, fftbins=periodic)
                        assert np.array_equal(expected, cross), (window, n, periodic)
                        expected = np.asarray(expected, dtype=np.float64)
                        params = {
                            "window": family,
                            **extra,
                            "periodic": periodic,
                            "n": n,
                        }
                        cases.append(
                            {
                                "name": f"{key}_{mode}_{n}",
                                "params": params,
                                "shape": list(expected.shape),
                                "values": expected.tolist(),
                            }
                        )
            write_suite(self.SUITE, family, cases)
        self.generate_cola()

    def generate_cola(self):
        cases = []
        for family, index, length, hop in self.COLA_CASES:
            key, window, extra = self.FAMILIES[family][index]
            analysis = scipy.signal.get_window(window, length, fftbins=True)
            expected = bool(
                scipy.signal.check_COLA(analysis, length, length - hop, tol=1e-10)
            )
            cases.append(
                {
                    "name": f"{key}_{length}_{hop}",
                    "params": {"window": family, **extra, "length": length, "hop": hop},
                    "expected": expected,
                }
            )
        write_suite(self.SUITE, "cola", cases)


class DbConversionsVectorGenerator:
    """Golden vectors for Soundml.Convert's decibel conversions.

    power_to_db and amplitude_to_db against librosa, over synthetic seeded
    inputs: fixed references, the amin floor (values below it), top_db
    clamping (including a rank-two input whose threshold is global across
    rows), negative entries (powers are floored signed, amplitudes are
    magnitudes), and both element dtypes. Each case stores its input in the
    params (flattened in C order, exactly representable in the case dtype)
    and the librosa output in the values.
    """

    SUITE = "db"

    SEED = 0x5EED

    DTYPES = [np.float32, np.float64]

    def specs(self, rng):
        def spread(low: float, high: float, size):
            return np.exp(rng.uniform(np.log(low), np.log(high), size=size))

        return [
            # values well below the 1e-10 floor: the amin clamp must hit
            ("power_basic", "power_to_db", 1.0, 1e-10, None, spread(1e-14, 1e3, 64)),
            ("power_reference", "power_to_db", 2.5, 1e-8, None, spread(1e-12, 1e2, 64)),
            # librosa floors real powers signed: negative entries pin that
            (
                "power_negative",
                "power_to_db",
                1.0,
                1e-10,
                None,
                spread(1e-6, 1e2, 32) * rng.choice([-1.0, 1.0], 32),
            ),
            ("power_top80", "power_to_db", 1.0, 1e-10, 80.0, spread(1e-12, 1e2, 64)),
            ("power_top30", "power_to_db", 1.0, 1e-10, 30.0, spread(1e-12, 1e2, 64)),
            # the clamp threshold is global across both rows
            (
                "power_matrix_top20",
                "power_to_db",
                1.0,
                1e-10,
                20.0,
                spread(1e-9, 1e3, (2, 32)),
            ),
            # values below the 1e-5 amplitude floor, and negative ones
            (
                "amplitude_basic",
                "amplitude_to_db",
                1.0,
                1e-5,
                None,
                spread(1e-9, 1e2, 64) * rng.choice([-1.0, 1.0], 64),
            ),
            ("amplitude_reference", "amplitude_to_db", 0.5, 2e-5, None, spread(1e-7, 1e2, 64)),
            ("amplitude_top80", "amplitude_to_db", 1.0, 1e-5, 80.0, spread(1e-9, 1e2, 64)),
        ]

    def generate(self):
        functions = {
            "power_to_db": librosa.power_to_db,
            "amplitude_to_db": librosa.amplitude_to_db,
        }
        suites = {name: [] for name in functions}
        rng = np.random.default_rng(self.SEED)
        for name, function, reference, amin, top_db, values in self.specs(rng):
            for dtype in self.DTYPES:
                x = np.ascontiguousarray(values, dtype=dtype)
                y = functions[function](x, ref=reference, amin=amin, top_db=top_db)
                assert y.dtype == x.dtype, (name, x.dtype, y.dtype)
                suites[function].append(
                    {
                        "name": f"{name}_{np.dtype(dtype).name}",
                        "params": {
                            "function": function,
                            "dtype": np.dtype(dtype).name,
                            "reference": reference,
                            "amin": amin,
                            "top_db": top_db,
                            "input": x.astype(np.float64).flatten().tolist(),
                        },
                        "shape": list(y.shape),
                        "values": y.astype(np.float64).flatten().tolist(),
                    }
                )
        for function, cases in suites.items():
            write_suite(self.SUITE, function, cases)


GENERATORS = [WindowVectorGenerator, DbConversionsVectorGenerator]


if __name__ == "__main__":
    for generator in GENERATORS:
        generator().generate()
