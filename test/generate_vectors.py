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


class StftVectorGenerator:
    """Golden vectors for Soundml.Stft.

    Magnitude and power spectra from librosa.stft over a deterministic LCG
    signal (reproducible bit-exactly in OCaml through integer arithmetic),
    covering (fft_size, hop) combos x alignment (centered = center=True with
    reflect padding, left = center=False) x odd/even signal lengths x both
    float dtypes. The float32 cases quantize the signal to float32 and
    compute in float64, so the reference isolates input rounding from
    implementation precision. Also emits librosa.fft_frequencies and
    librosa.frames_to_time truths for Stft.frequencies and Stft.times, and one
    complex-valued case (real and imaginary parts) pinning the FFT sign
    convention, which magnitude-only cases cannot see.
    """

    SUITE = "stft"

    SR = 22050

    # (case key, fft_size, hop, win_length or None)
    COMBOS = [
        ("fft16_hop4", 16, 4, None),
        ("fft32_hop7", 32, 7, None),
        ("fft64_hop16", 64, 16, None),
        ("fft32_hop8_win20", 32, 8, 20),
    ]

    LENGTHS = [127, 128]

    COMPLEX_LENGTH = 37

    @staticmethod
    def lcg_signal(n, seed=20250803):
        """A length-n float64 signal from a 31-bit LCG: every operation is
        integer arithmetic plus one exact float64 division, so OCaml
        reproduces the values bit-for-bit."""
        values = []
        state = seed
        for _ in range(n):
            state = (1103515245 * state + 12345) % (1 << 31)
            values.append(state / float(1 << 30) - 1.0)
        return np.asarray(values, dtype=np.float64)

    def spectra_cases(self, fft_size, hop, win_length):
        cases = []
        wl = win_length if win_length is not None else fft_size
        for length in self.LENGTHS:
            base = self.lcg_signal(length)
            for dtype, y in (
                ("float64", base),
                ("float32", base.astype(np.float32).astype(np.float64)),
            ):
                for alignment, center in (("centered", True), ("left", False)):
                    spectrum = librosa.stft(
                        y,
                        n_fft=fft_size,
                        hop_length=hop,
                        win_length=wl,
                        window="hann",
                        center=center,
                        pad_mode="reflect",
                        dtype=np.complex128,
                    )
                    for kind, expected in (
                        ("magnitude", np.abs(spectrum)),
                        ("power", np.abs(spectrum) ** 2),
                    ):
                        params = {
                            "fft_size": fft_size,
                            "hop": hop,
                            "win_length": wl,
                            "alignment": alignment,
                            "length": length,
                            "dtype": dtype,
                            "kind": kind,
                        }
                        cases.append(
                            {
                                "name": (
                                    f"{kind}_{alignment}_{dtype}_n{length}"
                                ),
                                "params": params,
                                "shape": list(expected.shape),
                                "values": expected.flatten().tolist(),
                            }
                        )
        return cases

    def complex_cases(self, fft_size, hop):
        """Real and imaginary parts of one centered transform: a global
        conjugation or FFT sign-convention error passes every magnitude
        case, so one complex-valued golden pins the convention."""
        y = self.lcg_signal(self.COMPLEX_LENGTH)
        spectrum = librosa.stft(
            y,
            n_fft=fft_size,
            hop_length=hop,
            win_length=fft_size,
            window="hann",
            center=True,
            pad_mode="reflect",
            dtype=np.complex128,
        )
        cases = []
        for kind, expected in (
            ("real", np.real(spectrum)),
            ("imag", np.imag(spectrum)),
        ):
            params = {
                "fft_size": fft_size,
                "hop": hop,
                "win_length": fft_size,
                "alignment": "centered",
                "length": self.COMPLEX_LENGTH,
                "dtype": "float64",
                "kind": kind,
            }
            cases.append(
                {
                    "name": f"{kind}_centered_float64_n{self.COMPLEX_LENGTH}",
                    "params": params,
                    "shape": list(expected.shape),
                    "values": expected.flatten().tolist(),
                }
            )
        return cases

    def coordinate_cases(self):
        cases = []
        for key, fft_size, hop, _ in self.COMBOS:
            frequencies = librosa.fft_frequencies(sr=self.SR, n_fft=fft_size)
            cases.append(
                {
                    "name": f"frequencies_{key}",
                    "params": {
                        "fft_size": fft_size,
                        "hop": hop,
                        "sample_rate": self.SR,
                        "kind": "frequencies",
                    },
                    "shape": list(frequencies.shape),
                    "values": frequencies.tolist(),
                }
            )
            for alignment, center in (("centered", True), ("left", False)):
                for length in self.LENGTHS:
                    if center:
                        count = 1 + (length + 2 * (fft_size // 2) - fft_size) // hop
                    else:
                        count = 1 + (length - fft_size) // hop
                    times = librosa.frames_to_time(
                        np.arange(count), sr=self.SR, hop_length=hop
                    )
                    cases.append(
                        {
                            "name": f"times_{key}_{alignment}_n{length}",
                            "params": {
                                "fft_size": fft_size,
                                "hop": hop,
                                "sample_rate": self.SR,
                                "alignment": alignment,
                                "length": length,
                                "kind": "times",
                            },
                            "shape": list(times.shape),
                            "values": times.tolist(),
                        }
                    )
        return cases

    def generate(self):
        for key, fft_size, hop, win_length in self.COMBOS:
            write_suite(
                self.SUITE, key, self.spectra_cases(fft_size, hop, win_length)
            )
        write_suite(self.SUITE, "complex_fft16_hop4", self.complex_cases(16, 4))
        write_suite(self.SUITE, "coordinates", self.coordinate_cases())



class MelVectorGenerator:
    """Golden vectors for Soundml.Mel and the flat mel features.

    Three files:

    - filterbank: librosa.filters.mel weight matrices in float64, covering
      the Slaney and HTK scales x the slaney and no-op norms across several
      (n_mels, fft_size) geometries, including a nonzero fmin and an fmax
      pinned exactly at Nyquist.
    - mel_spectrogram: librosa.feature.melspectrogram end-to-end over the
      deterministic LCG signal (reproduced bit-exactly in OCaml), with
      pad_mode, htk and norm always passed explicitly and the filterbank
      computed in float64 (dtype=np.float64), so the reference carries no
      float32 quantisation of its own.
    - mfcc: librosa.feature.mfcc end-to-end (n_mfcc 13 and 20, lifter 0 and
      22, both scales), including a decaying-envelope case whose dynamic
      range exceeds 80 dB so the top_db clamp and the amin floor inside
      librosa's log-mel are both exercised.

    float32 cases quantize the signal to float32 and compute the reference
    in float64, exactly like the stft suite: the reference isolates input
    rounding from implementation precision.
    """

    SUITE = "mel"

    SEED = 20260803

    # (case key, sample_rate, fft_size, n_mels, fmin, fmax, [(scale, norm)])
    FILTERBANKS = [
        ("fft512_mel40", 22050, 512, 40, 0.0, None, [("slaney", "slaney")]),
        ("fft256_mel32", 22050, 256, 32, 0.0, None, [("htk", "none")]),
        (
            "fft128_mel13_bounds",
            16000,
            128,
            13,
            300.0,
            8000.0,
            [
                ("slaney", "slaney"),
                ("slaney", "none"),
                ("htk", "slaney"),
                ("htk", "none"),
            ],
        ),
        (
            "fft64_mel8_narrow",
            8000,
            64,
            8,
            100.0,
            4000.0,
            [
                ("slaney", "slaney"),
                ("slaney", "none"),
                ("htk", "slaney"),
                ("htk", "none"),
            ],
        ),
    ]

    @staticmethod
    def signal(n, envelope=False, seed=SEED):
        """The 31-bit LCG signal of the stft suite under this suite's seed,
        optionally shaped by an exp(-12 i / n) envelope whose ~104 dB power
        decay drives librosa's log-mel through both its top_db clamp and its
        amin floor."""
        base = StftVectorGenerator.lcg_signal(n, seed=seed)
        if not envelope:
            return base
        return base * np.exp(-12.0 * np.arange(n, dtype=np.float64) / n)

    @staticmethod
    def mel_kwargs(n_mels, fmin, fmax, scale, norm):
        return {
            "n_mels": n_mels,
            "fmin": fmin,
            "fmax": fmax,
            "htk": scale == "htk",
            "norm": None if norm == "none" else norm,
            "dtype": np.float64,
        }

    def filterbank_cases(self):
        cases = []
        for key, sr, fft_size, n_mels, fmin, fmax, combos in self.FILTERBANKS:
            for scale, norm in combos:
                weights = librosa.filters.mel(
                    sr=sr,
                    n_fft=fft_size,
                    **self.mel_kwargs(n_mels, fmin, fmax, scale, norm),
                )
                assert weights.shape == (n_mels, 1 + fft_size // 2)
                assert weights.dtype == np.float64
                cases.append(
                    {
                        "name": f"{key}_{scale}_{norm}",
                        "params": {
                            "sample_rate": sr,
                            "fft_size": fft_size,
                            "n_mels": n_mels,
                            "f_min": fmin,
                            "f_max": fmax,
                            "scale": scale,
                            "norm": norm,
                        },
                        "shape": list(weights.shape),
                        "values": weights.flatten().tolist(),
                    }
                )
        return cases

    # (case key, sample_rate, fft_size, hop, length, n_mels, fmin, fmax,
    #  scale, norm, power, alignment, dtypes)
    SPECTROGRAMS = [
        ("base", 22050, 512, 128, 1000, 40, 0.0, None, "slaney", "slaney",
         2.0, "centered", ["float64", "float32"]),
        ("magnitude", 22050, 512, 128, 1000, 40, 0.0, None, "slaney",
         "slaney", 1.0, "centered", ["float64"]),
        ("htk_none", 22050, 512, 128, 1000, 40, 0.0, None, "htk", "none",
         2.0, "centered", ["float64", "float32"]),
        ("left", 22050, 256, 64, 512, 32, 0.0, None, "slaney", "slaney",
         2.0, "left", ["float64"]),
        ("bounds", 16000, 128, 32, 400, 13, 300.0, 8000.0, "slaney",
         "slaney", 2.0, "centered", ["float64", "float32"]),
    ]

    def spectrogram_cases(self):
        cases = []
        for (key, sr, fft_size, hop, length, n_mels, fmin, fmax, scale,
             norm, power, alignment, dtypes) in self.SPECTROGRAMS:
            base = self.signal(length)
            for dtype in dtypes:
                y = (
                    base
                    if dtype == "float64"
                    else base.astype(np.float32).astype(np.float64)
                )
                expected = librosa.feature.melspectrogram(
                    y=y,
                    sr=sr,
                    n_fft=fft_size,
                    hop_length=hop,
                    win_length=fft_size,
                    window="hann",
                    center=alignment == "centered",
                    pad_mode="reflect",
                    power=power,
                    **self.mel_kwargs(n_mels, fmin, fmax, scale, norm),
                )
                cases.append(
                    {
                        "name": f"{key}_{dtype}",
                        "params": {
                            "sample_rate": sr,
                            "fft_size": fft_size,
                            "hop": hop,
                            "length": length,
                            "n_mels": n_mels,
                            "f_min": fmin,
                            "f_max": fmax,
                            "scale": scale,
                            "norm": norm,
                            "power": power,
                            "alignment": alignment,
                            "dtype": dtype,
                            "envelope": False,
                        },
                        "shape": list(expected.shape),
                        "values": expected.flatten().tolist(),
                    }
                )
        return cases

    # (case key, n_mfcc, lifter, scale, envelope, dtypes) over the base
    # geometry, plus one bounded-filterbank geometry appended below.
    MFCCS = [
        ("n20_lifter0", 20, 0.0, "slaney", False, ["float64", "float32"]),
        ("n13_lifter22", 13, 22.0, "slaney", False, ["float64", "float32"]),
        ("n13_htk", 13, 0.0, "htk", False, ["float64"]),
        ("n20_lifter22", 20, 22.0, "slaney", False, ["float64"]),
        ("clamped", 20, 0.0, "slaney", True, ["float64", "float32"]),
    ]

    def mfcc_cases(self):
        geometries = {
            "base": (22050, 512, 128, 1000, 40, 0.0, None),
            "bounds": (16000, 128, 32, 400, 13, 300.0, 8000.0),
        }
        specs = [("base",) + spec for spec in self.MFCCS]
        specs.append(("bounds", "n13_bounds", 13, 0.0, "slaney", False,
                      ["float64"]))
        cases = []
        for geometry, key, n_mfcc, lifter, scale, envelope, dtypes in specs:
            sr, fft_size, hop, length, n_mels, fmin, fmax = geometries[
                geometry
            ]
            base = self.signal(length, envelope=envelope)
            for dtype in dtypes:
                y = (
                    base
                    if dtype == "float64"
                    else base.astype(np.float32).astype(np.float64)
                )
                kwargs = self.mel_kwargs(n_mels, fmin, fmax, scale, "slaney")
                del kwargs["norm"]  # mfcc's own norm is the DCT's
                expected = librosa.feature.mfcc(
                    y=y,
                    sr=sr,
                    n_mfcc=n_mfcc,
                    dct_type=2,
                    norm="ortho",
                    lifter=lifter,
                    n_fft=fft_size,
                    hop_length=hop,
                    win_length=fft_size,
                    window="hann",
                    center=True,
                    pad_mode="reflect",
                    power=2.0,
                    **kwargs,
                )
                cases.append(
                    {
                        "name": f"{key}_{dtype}",
                        "params": {
                            "sample_rate": sr,
                            "fft_size": fft_size,
                            "hop": hop,
                            "length": length,
                            "n_mels": n_mels,
                            "f_min": fmin,
                            "f_max": fmax,
                            "scale": scale,
                            "norm": "slaney",
                            "n_mfcc": n_mfcc,
                            "lifter": lifter,
                            "alignment": "centered",
                            "dtype": dtype,
                            "envelope": envelope,
                        },
                        "shape": list(expected.shape),
                        "values": expected.flatten().tolist(),
                    }
                )
        return cases

    def generate(self):
        write_suite(self.SUITE, "filterbank", self.filterbank_cases())
        write_suite(self.SUITE, "mel_spectrogram", self.spectrogram_cases())
        write_suite(self.SUITE, "mfcc", self.mfcc_cases())


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



GENERATORS = [
    WindowVectorGenerator,
    StftVectorGenerator,
    MelVectorGenerator,
    DbConversionsVectorGenerator,
]

if __name__ == "__main__":
    for generator in GENERATORS:
        generator().generate()
