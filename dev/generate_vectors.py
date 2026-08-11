"""
This file is part of SoundML.

Copyright (C) 2025 Gabriel Santamaria

This script generates the golden test vectors committed beside each test suite (see SUITE_DIRECTORIES).
The reference implementation is librosa 0.11 (which delegates window
computation to scipy.signal); every generated file records the exact
versions it was produced with.

The vectors are committed to the repository: CI never runs this script and
never needs Python. Rerun it only to regenerate the goldens, from an
environment with the pinned reference versions:

    python3 -m venv .venv
    .venv/bin/pip install librosa==0.11.0
    .venv/bin/python generate_vectors.py

Each suite writes small JSON files under the suite's vectors/ directory. A file is

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

import io
import json
import os
import platform
import struct

import librosa
import numpy as np
import scipy
import scipy.signal
import soundfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Each suite's vectors are co-located with the test suite that replays them.
SUITE_DIRECTORIES = {
    "windows": "soundml/test/window/vectors",
    "stft": "soundml/test/stft/vectors",
    "istft": "soundml/test/istft/vectors",
    "mel": "soundml/test/mel/vectors",
    "db": "soundml/test/db/vectors",
    "features_spectral": "soundml/test/spectral/vectors",
    "features_energy": "soundml/test/energy/vectors",
    "features_onset": "soundml/test/onset/vectors",
    "cqt": "soundml/test/cqt/vectors",
    "chroma": "soundml/test/chroma/vectors",
    "hpss": "soundml/test/hpss/vectors",
    "resample": "soundml/test/resample/vectors",
    "io": "soundml-io/test/vectors",
}

GENERATOR_VERSIONS = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "librosa": librosa.__version__,
}

assert librosa.__version__.startswith(
    "0.11."
), f"librosa 0.11 is the pinned reference, found {librosa.__version__}"


def write_suite(suite: str, filename: str, cases: list, versions: dict = None):
    """Write one golden file for `suite`, pretty-printed with the values of
    each case kept on a single line."""
    directory = os.path.join(REPO_ROOT, SUITE_DIRECTORIES[suite])
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{filename}.json")
    rendered = []
    for case in cases:
        fields = [f'"{key}": {json.dumps(value)}' for key, value in case.items()]
        rendered.append("  {" + ", ".join(fields) + "}")
    body = ",\n".join(rendered)
    generator = json.dumps(GENERATOR_VERSIONS if versions is None else versions)
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



class IstftVectorGenerator:
    """Golden vectors for the synthesis side of Soundml.Stft.

    Three kinds of file:

    - inverse_<cell>: librosa.istft over deterministic synthetic spectra,
      covering (fft_size, hop, win_length) combos x alignment x frame counts
      x both float dtypes. The spectra come from two LCG streams (one per
      complex component, reproduced bit-exactly in OCaml through integer
      arithmetic), with the imaginary parts of the DC and Nyquist bins zeroed
      so the inverse real FFT discards nothing and the vectors stay portable
      across FFT implementations. They are deliberately inconsistent -- no
      signal has exactly this transform -- so the cases exercise the
      least-squares synthesis, not merely a round trip.

    - lengths: the same, with an explicit output length above, below and at
      the natural one, and one under a single frame span, pinning how many
      frames a shortened request reads.

    - griffinlim_<cell>: librosa.griffinlim at init=None (the deterministic
      all-ones phase; librosa's default is a random init, which no committed
      vector could pin), across iteration counts and momenta including the
      momentum-free classic algorithm, over an LCG magnitude spectrogram that
      is likewise inconsistent.

    Alignments map to librosa as elsewhere in the suite: centered is
    center=True, left is center=False, and right -- which has no librosa
    counterpart -- is the center=False synthesis with the fft_size - 1
    positions of its left extension dropped. pad_mode is always passed
    explicitly, and the float32 cases quantize the input to float32 and
    compute the reference in float64, so the reference isolates input
    rounding from implementation precision.
    """

    SUITE = "istft"

    # (case key, fft_size, hop, win_length)
    COMBOS = [
        ("fft16_hop4", 16, 4, 16),
        ("fft32_hop7", 32, 7, 32),
        ("fft32_hop8_win20", 32, 8, 20),
        ("fft64_hop16", 64, 16, 64),
        ("fft64_hop17_win40", 64, 17, 40),
        ("fft31_hop5", 31, 5, 31),
    ]

    # The analysis geometry of the benchmarks, at one frame count.
    BIG_COMBOS = [
        ("fft2048_hop512", 2048, 512, 2048),
        ("fft2048_hop500_win1200", 2048, 500, 1200),
    ]

    FRAME_COUNTS = [5, 12]

    ALIGNMENTS = ["centered", "left", "right"]

    # (case key, fft_size, hop, win_length, alignment, frames). The third cell
    # is the first two's geometry with a window shorter than the transform,
    # which the iteration re-centres inside the frame on every re-analysis.
    GRIFFINLIM_COMBOS = [
        ("fft64_hop16", 64, 16, 64, "centered", 8),
        ("fft512_hop128", 512, 128, 512, "left", 6),
        ("fft64_hop16_win40", 64, 16, 40, "centered", 8),
    ]

    # (n_iter, momentum): momentum 0 is the classic alternating projection,
    # 0.99 librosa's accelerated default.
    GRIFFINLIM_SCHEDULE = [(1, 0.0), (1, 0.99), (2, 0.5), (8, 0.0), (8, 0.99),
                           (32, 0.0), (32, 0.99)]

    @staticmethod
    def lcg(n, seed):
        """A length-n float64 stream from a 31-bit LCG: every operation is
        integer arithmetic plus one exact float64 division, so OCaml
        reproduces the values bit-for-bit."""
        values = []
        state = seed
        for _ in range(n):
            state = (1103515245 * state + 12345) % (1 << 31)
            values.append(state / float(1 << 30) - 1.0)
        return np.asarray(values, dtype=np.float64)

    @classmethod
    def spectrum(cls, fft_size, frames):
        """A deterministic [bins, frames] spectrum: one LCG stream per
        component, with imag(DC) and imag(Nyquist) zeroed -- the inverse real
        FFT ignores them, so leaving them nonzero would make the vectors
        depend on that detail."""
        bins = fft_size // 2 + 1
        re = cls.lcg(bins * frames, 20250803).reshape(bins, frames)
        im = cls.lcg(bins * frames, 20250804).reshape(bins, frames)
        im[0, :] = 0.0
        if fft_size % 2 == 0:
            im[bins - 1, :] = 0.0
        return re + 1j * im

    @staticmethod
    def widths(fft_size, alignment):
        if alignment == "centered":
            return fft_size // 2, fft_size // 2
        if alignment == "left":
            return 0, 0
        return fft_size - 1, 0

    @classmethod
    def inverse(cls, spectrum, fft_size, hop, win_length, alignment, length=None):
        """The reference synthesis at our geometry. librosa covers centered
        and left directly; right is left with the left extension trimmed,
        which is exactly what our synthesis does with it."""
        center = alignment == "centered"
        if alignment == "right":
            assert length is None
            reference = librosa.istft(
                spectrum,
                hop_length=hop,
                win_length=win_length,
                n_fft=fft_size,
                window="hann",
                center=False,
                dtype=np.float64,
            )
            return reference[fft_size - 1:]
        return librosa.istft(
            spectrum,
            hop_length=hop,
            win_length=win_length,
            n_fft=fft_size,
            window="hann",
            center=center,
            dtype=np.float64,
            length=length,
        )

    def inverse_cases(self, fft_size, hop, win_length, alignments, frame_counts,
                      dtypes):
        cases = []
        for frames in frame_counts:
            base = self.spectrum(fft_size, frames)
            for dtype in dtypes:
                spectrum = base
                if dtype == "float32":
                    spectrum = base.astype(np.complex64).astype(np.complex128)
                for alignment in alignments:
                    expected = self.inverse(
                        spectrum, fft_size, hop, win_length, alignment
                    )
                    cases.append(
                        {
                            "name": f"inverse_{alignment}_{dtype}_f{frames}",
                            "params": {
                                "fft_size": fft_size,
                                "hop": hop,
                                "win_length": win_length,
                                "alignment": alignment,
                                "frames": frames,
                                "dtype": dtype,
                            },
                            "shape": list(expected.shape),
                            "values": expected.tolist(),
                        }
                    )
        return cases

    def length_cases(self):
        """An explicit output length, above, below and at the natural one, and
        one a hop and a sample long: the frames a shortened request may skip
        are the ones that start past it, and a longer one is zero-filled. The
        reference inverts ceil((length + left + right) / hop) frames and no
        more, so only the fourth length leaves any of the spectrum unread. The
        natural length puts (frames - 1) * hop + fft_size padded positions in
        that numerator, and shortening it by three still leaves
        ceil((fft_size - 3) / hop) >= 1 frames on top of frames - 1, so the
        first three all read the whole spectrum."""
        cases = []
        for fft_size, hop, win_length, alignment in (
            (32, 8, 20, "centered"),
            (64, 17, 40, "left"),
        ):
            frames = 9
            spectrum = self.spectrum(fft_size, frames)
            left, right = self.widths(fft_size, alignment)
            natural = (frames - 1) * hop + fft_size - left - right
            for length in (natural, natural - 3, natural + 7, hop + 1):
                expected = self.inverse(
                    spectrum, fft_size, hop, win_length, alignment, length=length
                )
                cases.append(
                    {
                        "name": (
                            f"length{length}_{alignment}_fft{fft_size}"
                            f"_hop{hop}_win{win_length}"
                        ),
                        "params": {
                            "fft_size": fft_size,
                            "hop": hop,
                            "win_length": win_length,
                            "alignment": alignment,
                            "frames": frames,
                            "length": length,
                            "dtype": "float64",
                        },
                        "shape": list(expected.shape),
                        "values": expected.tolist(),
                    }
                )
        return cases

    def griffinlim_cases(self, fft_size, hop, win_length, alignment, frames):
        """One LCG stream shifted into [0, 2): a magnitude spectrogram no
        signal produces, so every iteration does real work."""
        bins = fft_size // 2 + 1
        base = self.lcg(bins * frames, 20250803).reshape(bins, frames) + 1.0
        cases = []
        for dtype in ("float64", "float32"):
            magnitudes = base
            if dtype == "float32":
                magnitudes = base.astype(np.float32).astype(np.float64)
            for n_iter, momentum in self.GRIFFINLIM_SCHEDULE:
                expected = librosa.griffinlim(
                    magnitudes,
                    n_iter=n_iter,
                    hop_length=hop,
                    win_length=win_length,
                    n_fft=fft_size,
                    window="hann",
                    center=alignment == "centered",
                    dtype=np.float64,
                    pad_mode="constant",
                    momentum=momentum,
                    init=None,
                )
                cases.append(
                    {
                        "name": (
                            f"griffinlim_{alignment}_{dtype}"
                            f"_n{n_iter}_m{momentum:g}"
                        ),
                        "params": {
                            "fft_size": fft_size,
                            "hop": hop,
                            "win_length": win_length,
                            "alignment": alignment,
                            "frames": frames,
                            "n_iter": n_iter,
                            "momentum": momentum,
                            "dtype": dtype,
                        },
                        "shape": list(expected.shape),
                        "values": expected.tolist(),
                    }
                )
        return cases

    def generate(self):
        for key, fft_size, hop, win_length in self.COMBOS:
            write_suite(
                self.SUITE,
                f"inverse_{key}",
                self.inverse_cases(
                    fft_size,
                    hop,
                    win_length,
                    self.ALIGNMENTS,
                    self.FRAME_COUNTS,
                    ("float64", "float32"),
                ),
            )
        for key, fft_size, hop, win_length in self.BIG_COMBOS:
            cases = self.inverse_cases(
                fft_size, hop, win_length, ["centered"], [5], ("float64", "float32")
            )
            cases += self.inverse_cases(
                fft_size, hop, win_length, ["left", "right"], [5], ("float64",)
            )
            write_suite(self.SUITE, f"inverse_{key}", cases)
        write_suite(self.SUITE, "lengths", self.length_cases())
        for key, fft_size, hop, win_length, alignment, frames in (
            self.GRIFFINLIM_COMBOS
        ):
            write_suite(
                self.SUITE,
                f"griffinlim_{key}",
                self.griffinlim_cases(
                    fft_size, hop, win_length, alignment, frames
                ),
            )


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



class SpectralFeaturesVectorGenerator:
    """Golden vectors for the flat spectral-shape features
    (spectral_centroid, spectral_bandwidth, spectral_rolloff,
    spectral_flatness), one file per feature.

    Direct-spectrogram cases feed librosa.feature.* a magnitude tensor
    reproduced bit-exactly on the OCaml side: the 31-bit LCG of the stft
    suite under this suite's seed, absolute value, C-order reshape. They
    cover several (bins, frames, sample_rate) geometries, a batched
    rank-three input, a custom non-uniform frequency grid (exact
    triangular-number arithmetic) and both element dtypes; float32 cases
    quantize the magnitudes to float32 and compute the reference in
    float64, exactly like the stft and mel suites.

    End-to-end cases run librosa.feature.* from the LCG signal with
    center=True and pad_mode="reflect" passed explicitly (the OCaml side
    computes the magnitude spectrogram with Stft.power_spectrum ~power:1.).
    Rolloff has no float32 end-to-end case: its output is a discrete bin
    choice, and float32 spectrogram rounding could flip the threshold bin
    against a float64 reference.
    """

    SUITE = "features_spectral"

    SEED = 20261024

    SR = 22050

    # end-to-end geometry
    LENGTH = 400

    FFT_SIZE = 128

    HOP = 32

    @classmethod
    def magnitudes(cls, shape, dtype):
        n = int(np.prod(shape))
        base = np.abs(StftVectorGenerator.lcg_signal(n, seed=cls.SEED))
        if dtype == "float32":
            base = base.astype(np.float32).astype(np.float64)
        return base.reshape(shape)

    @staticmethod
    def triangular_freqs(bins):
        """An increasing non-uniform grid in exact integer arithmetic:
        10 * (k + 1) * (k + 2) / 2 for bin k."""
        return np.array([10.0 * (k + 1) * (k + 2) / 2 for k in range(bins)])

    def spectrogram_case(self, function, name, shape, sr, dtype, *,
                         triangular=False, squared=False, **feature_kwargs):
        s = self.magnitudes(shape, dtype)
        if squared:
            s = s**2
        kwargs = dict(feature_kwargs)
        params = {
            "function": function.__name__,
            "source": "spectrogram",
            "shape_s": list(shape),
            "dtype": dtype,
            **{k: v for k, v in feature_kwargs.items()},
        }
        if function is not librosa.feature.spectral_flatness:
            params["sample_rate"] = sr
            kwargs["sr"] = sr
            if triangular:
                kwargs["freq"] = self.triangular_freqs(shape[-2])
        params["freqs"] = "triangular" if triangular else "fft"
        params["squared"] = squared
        expected = function(S=s, **kwargs)
        return {
            "name": name,
            "params": params,
            "shape": list(expected.shape),
            "values": expected.flatten().tolist(),
        }

    def signal_case(self, function, name, dtype, **feature_kwargs):
        base = StftVectorGenerator.lcg_signal(self.LENGTH, seed=self.SEED)
        y = (
            base
            if dtype == "float64"
            else base.astype(np.float32).astype(np.float64)
        )
        kwargs = dict(feature_kwargs)
        params = {
            "function": function.__name__,
            "source": "signal",
            "length": self.LENGTH,
            "fft_size": self.FFT_SIZE,
            "hop": self.HOP,
            "dtype": dtype,
            "freqs": "fft",
            "squared": False,
            **{k: v for k, v in feature_kwargs.items()},
        }
        if function is not librosa.feature.spectral_flatness:
            params["sample_rate"] = self.SR
            kwargs["sr"] = self.SR
        expected = function(
            y=y,
            n_fft=self.FFT_SIZE,
            hop_length=self.HOP,
            win_length=self.FFT_SIZE,
            window="hann",
            center=True,
            pad_mode="reflect",
            **kwargs,
        )
        return {
            "name": name,
            "params": params,
            "shape": list(expected.shape),
            "values": expected.flatten().tolist(),
        }

    def centroid_cases(self):
        f = librosa.feature.spectral_centroid
        return [
            self.spectrogram_case(f, "s9x12_float64", (9, 12), 22050, "float64"),
            self.spectrogram_case(f, "s9x12_float32", (9, 12), 22050, "float32"),
            self.spectrogram_case(
                f, "s17x7_sr8000_float64", (17, 7), 8000, "float64"
            ),
            self.spectrogram_case(
                f, "batch2x9x5_float64", (2, 9, 5), 22050, "float64"
            ),
            self.spectrogram_case(
                f, "s9x12_triangular_float64", (9, 12), 22050, "float64",
                triangular=True,
            ),
            self.signal_case(f, "signal_float64", "float64"),
            self.signal_case(f, "signal_float32", "float32"),
        ]

    def bandwidth_cases(self):
        f = librosa.feature.spectral_bandwidth
        return [
            self.spectrogram_case(
                f, "s9x12_p2_float64", (9, 12), 22050, "float64", p=2.0
            ),
            self.spectrogram_case(
                f, "s9x12_p2_float32", (9, 12), 22050, "float32", p=2.0
            ),
            self.spectrogram_case(
                f, "s9x12_p3_float64", (9, 12), 22050, "float64", p=3.0
            ),
            self.spectrogram_case(
                f, "s17x7_sr8000_p1_float64", (17, 7), 8000, "float64", p=1.0
            ),
            self.spectrogram_case(
                f, "batch2x9x5_p2_float64", (2, 9, 5), 22050, "float64", p=2.0
            ),
            self.spectrogram_case(
                f, "s9x12_triangular_p2_float64", (9, 12), 22050, "float64",
                triangular=True, p=2.0,
            ),
            self.signal_case(f, "signal_p2_float64", "float64", p=2.0),
            self.signal_case(f, "signal_p2_float32", "float32", p=2.0),
        ]

    def rolloff_cases(self):
        f = librosa.feature.spectral_rolloff
        return [
            self.spectrogram_case(
                f, "s9x12_r85_float64", (9, 12), 22050, "float64",
                roll_percent=0.85,
            ),
            self.spectrogram_case(
                f, "s9x12_r85_float32", (9, 12), 22050, "float32",
                roll_percent=0.85,
            ),
            self.spectrogram_case(
                f, "s9x12_r01_float64", (9, 12), 22050, "float64",
                roll_percent=0.01,
            ),
            self.spectrogram_case(
                f, "s9x12_r50_float64", (9, 12), 22050, "float64",
                roll_percent=0.5,
            ),
            self.spectrogram_case(
                f, "s9x12_r99_float64", (9, 12), 22050, "float64",
                roll_percent=0.99,
            ),
            self.spectrogram_case(
                f, "s17x7_sr8000_r85_float64", (17, 7), 8000, "float64",
                roll_percent=0.85,
            ),
            self.spectrogram_case(
                f, "batch2x9x5_r85_float64", (2, 9, 5), 22050, "float64",
                roll_percent=0.85,
            ),
            self.spectrogram_case(
                f, "s9x12_triangular_r85_float64", (9, 12), 22050, "float64",
                triangular=True, roll_percent=0.85,
            ),
            self.signal_case(f, "signal_r85_float64", "float64",
                             roll_percent=0.85),
            self.signal_case(f, "signal_r99_float64", "float64",
                             roll_percent=0.99),
        ]

    def flatness_cases(self):
        f = librosa.feature.spectral_flatness
        return [
            self.spectrogram_case(
                f, "s9x12_float64", (9, 12), 22050, "float64",
                amin=1e-10, power=2.0,
            ),
            self.spectrogram_case(
                f, "s9x12_float32", (9, 12), 22050, "float32",
                amin=1e-10, power=2.0,
            ),
            self.spectrogram_case(
                f, "s9x12_power1_squared_float64", (9, 12), 22050, "float64",
                squared=True, amin=1e-10, power=1.0,
            ),
            self.spectrogram_case(
                f, "s9x12_amin1e3_float64", (9, 12), 22050, "float64",
                amin=1e-3, power=2.0,
            ),
            self.spectrogram_case(
                f, "s17x7_float64", (17, 7), 22050, "float64",
                amin=1e-10, power=2.0,
            ),
            self.spectrogram_case(
                f, "batch2x9x5_float64", (2, 9, 5), 22050, "float64",
                amin=1e-10, power=2.0,
            ),
            self.signal_case(f, "signal_float64", "float64",
                             amin=1e-10, power=2.0),
            self.signal_case(f, "signal_float32", "float32",
                             amin=1e-10, power=2.0),
        ]

    def generate(self):
        write_suite(self.SUITE, "spectral_centroid", self.centroid_cases())
        write_suite(self.SUITE, "spectral_bandwidth", self.bandwidth_cases())
        write_suite(self.SUITE, "spectral_rolloff", self.rolloff_cases())
        write_suite(self.SUITE, "spectral_flatness", self.flatness_cases())


class EnergyFeaturesVectorGenerator:
    """Golden vectors for the flat energy features (Soundml.rms,
    Soundml.rms_of_spectrogram, Soundml.zero_crossing_rate).

    Three files, all over the deterministic LCG stream of the stft suite
    under this suite's seed (reproduced bit-exactly in OCaml):

    - rms: librosa.feature.rms over audio, with center and pad_mode passed
      explicitly at their librosa 0.11 defaults (constant-zero centered
      padding) and the reference computed in float64 (dtype=np.float64),
      covering the true defaults, small (frame_length, hop) geometries
      including an odd frame length and hop > frame_length, frame_length 1,
      and a stereo case (C-order reshape of the stream).
    - rms_spectrogram: librosa.feature.rms on the S= path over synthetic
      magnitude spectrograms (|LCG values| reshaped in C order), pinning the
      halved DC bin, the even-length Nyquist halving and its absence for an
      odd frame length.
    - zero_crossing_rate: librosa.feature.zero_crossing_rate (edge-copy
      centered padding, hard-wired there), with explicit thresholds 0 and
      0.5 beside the 1e-10 default.

    float32 cases quantize the input to float32 and compute the reference in
    float64, exactly like the stft suite: the reference isolates input
    rounding from implementation precision.
    """

    SUITE = "features_energy"

    SEED = 20260812

    def signal(self, n):
        return StftVectorGenerator.lcg_signal(n, seed=self.SEED)

    @staticmethod
    def quantize(y, dtype):
        return y if dtype == "float64" else y.astype(np.float32).astype(np.float64)

    # (case key, frame_length, hop, length, channels, dtypes)
    RMS_CASES = [
        ("defaults", 2048, 512, 3000, 1, ["float64", "float32"]),
        ("fl16_hop4", 16, 4, 100, 1, ["float64", "float32"]),
        ("fl15_hop7_odd", 15, 7, 64, 1, ["float64"]),
        ("fl8_hop11_gap", 8, 11, 60, 1, ["float64"]),
        ("fl1_hop3", 1, 3, 20, 1, ["float64"]),
        ("fl16_hop4_stereo", 16, 4, 80, 2, ["float64"]),
    ]

    def rms_cases(self):
        cases = []
        for key, frame_length, hop, length, channels, dtypes in self.RMS_CASES:
            base = self.signal(channels * length)
            if channels > 1:
                base = base.reshape(channels, length)
            for dtype in dtypes:
                y = self.quantize(base, dtype)
                expected = librosa.feature.rms(
                    y=y,
                    frame_length=frame_length,
                    hop_length=hop,
                    center=True,
                    pad_mode="constant",
                    dtype=np.float64,
                )
                cases.append(
                    {
                        "name": f"{key}_{dtype}",
                        "params": {
                            "frame_length": frame_length,
                            "hop": hop,
                            "length": length,
                            "channels": channels,
                            "dtype": dtype,
                        },
                        "shape": list(expected.shape),
                        "values": expected.flatten().tolist(),
                    }
                )
        return cases

    # (case key, frame_length, frames, channels, dtypes)
    RMS_SPECTROGRAM_CASES = [
        ("fl16", 16, 12, 1, ["float64", "float32"]),
        ("fl15_odd", 15, 10, 1, ["float64"]),
        ("defaults", 2048, 4, 1, ["float64"]),
        ("fl16_stereo", 16, 5, 2, ["float64"]),
    ]

    def rms_spectrogram_cases(self):
        cases = []
        for key, frame_length, frames, channels, dtypes in (
            self.RMS_SPECTROGRAM_CASES
        ):
            bins = frame_length // 2 + 1
            base = np.abs(self.signal(channels * bins * frames))
            shape = ([channels] if channels > 1 else []) + [bins, frames]
            base = base.reshape(shape)
            for dtype in dtypes:
                s = self.quantize(base, dtype)
                expected = librosa.feature.rms(
                    S=s, frame_length=frame_length, dtype=np.float64
                )
                cases.append(
                    {
                        "name": f"{key}_{dtype}",
                        "params": {
                            "frame_length": frame_length,
                            "bins": bins,
                            "frames": frames,
                            "channels": channels,
                            "dtype": dtype,
                        },
                        "shape": list(expected.shape),
                        "values": expected.flatten().tolist(),
                    }
                )
        return cases

    # (case key, frame_length, hop, threshold or None, length, channels,
    #  dtypes)
    ZCR_CASES = [
        ("defaults", 2048, 512, None, 3000, 1, ["float64", "float32"]),
        ("fl16_hop4", 16, 4, None, 100, 1, ["float64", "float32"]),
        ("fl15_hop7_odd", 15, 7, None, 64, 1, ["float64"]),
        ("fl8_hop11_gap", 8, 11, None, 60, 1, ["float64"]),
        ("fl16_hop8_thr0", 16, 8, 0.0, 100, 1, ["float64"]),
        ("fl16_hop8_thr05", 16, 8, 0.5, 100, 1, ["float64", "float32"]),
        ("fl16_hop4_stereo", 16, 4, None, 80, 2, ["float64"]),
    ]

    def zcr_cases(self):
        cases = []
        for key, frame_length, hop, threshold, length, channels, dtypes in (
            self.ZCR_CASES
        ):
            base = self.signal(channels * length)
            if channels > 1:
                base = base.reshape(channels, length)
            for dtype in dtypes:
                y = self.quantize(base, dtype)
                kwargs = {} if threshold is None else {"threshold": threshold}
                expected = librosa.feature.zero_crossing_rate(
                    y,
                    frame_length=frame_length,
                    hop_length=hop,
                    center=True,
                    **kwargs,
                )
                cases.append(
                    {
                        "name": f"{key}_{dtype}",
                        "params": {
                            "frame_length": frame_length,
                            "hop": hop,
                            "threshold": threshold,
                            "length": length,
                            "channels": channels,
                            "dtype": dtype,
                        },
                        "shape": list(expected.shape),
                        "values": expected.flatten().tolist(),
                    }
                )
        return cases

    def generate(self):
        write_suite(self.SUITE, "rms", self.rms_cases())
        write_suite(
            self.SUITE, "rms_spectrogram", self.rms_spectrogram_cases()
        )
        write_suite(self.SUITE, "zero_crossing_rate", self.zcr_cases())



class OnsetFeaturesVectorGenerator:
    """Golden vectors for the spectral-contrast and onset-strength features.

    Two files:

    - spectral_contrast: librosa.feature.spectral_contrast over magnitude
      spectrograms of the deterministic LCG signal (reproduced bit-exactly in
      OCaml), covering the librosa defaults, the linear difference, large
      quantiles that exercise numpy's rint half-to-even band sizing and its
      slice clamping, a left-aligned geometry, a small-FFT geometry, and a
      silence tail whose all-zero frames drive the amin floor and the global
      top_db clamp inside the logarithmic difference (verified to bind).
      pad_mode is always passed explicitly ("reflect", the Stft.Config
      default).

    - onset_strength: librosa.onset.onset_strength end-to-end on its default
      chain (the log-power mel spectrogram at power_to_db defaults): lags
      1-3, centered and left-aligned analysis, degenerate short signals
      (fewer frames than the lag or the centered shift), and the
      decaying-envelope case whose >80 dB range drives the log-mel through
      the top_db clamp. The mel parameters are always passed explicitly with
      dtype=np.float64, forwarded through **kwargs to melspectrogram (the
      mel-suite convention: the reference carries no float32 quantisation of
      its own). Centered cases go through the y= path; left-aligned cases go
      through the S= path with power_to_db applied at its defaults, because
      onset_strength's own center flag only controls the compensation shift,
      never the feature's framing.

    float32 cases quantize the signal to float32 and compute the reference in
    float64, exactly like the stft and mel suites.
    """

    SUITE = "features_onset"

    SEED = 20260802

    @staticmethod
    def signal(n, envelope=False, silence_tail=False, seed=SEED):
        """The 31-bit LCG signal of the stft suite under this suite's seed,
        optionally shaped by the mel suite's exp(-12 i / n) envelope or
        zeroed over its second half."""
        base = StftVectorGenerator.lcg_signal(n, seed=seed)
        if envelope:
            base = base * np.exp(-12.0 * np.arange(n, dtype=np.float64) / n)
        if silence_tail:
            base = base.copy()
            base[n // 2 :] = 0.0
        return base

    # (case key, sample_rate, fft_size, hop, length, n_bands, f_min,
    #  quantile, linear, alignment, silence_tail, dtypes)
    CONTRASTS = [
        ("defaults", 22050, 512, 128, 1000, 6, 200.0, 0.02, False,
         "centered", False, ["float64", "float32"]),
        ("linear", 22050, 512, 128, 1000, 6, 200.0, 0.02, True, "centered",
         False, ["float64", "float32"]),
        # 0.5 * count lands exactly on .5 for odd band sizes: rint half-even
        ("quantile_half", 22050, 512, 128, 1000, 5, 300.0, 0.5, False,
         "centered", False, ["float64"]),
        ("quantile_31", 22050, 512, 128, 1000, 3, 200.0, 0.31, True,
         "centered", False, ["float64"]),
        ("left", 22050, 256, 64, 600, 4, 200.0, 0.02, False, "left", False,
         ["float64"]),
        ("silence_tail", 22050, 512, 128, 1000, 6, 200.0, 0.02, False,
         "centered", True, ["float64", "float32"]),
        ("small_fft", 8000, 128, 32, 400, 4, 150.0, 0.1, False, "centered",
         False, ["float64"]),
    ]

    def contrast_cases(self):
        cases = []
        for (key, sr, fft_size, hop, length, n_bands, f_min, quantile,
             linear, alignment, silence_tail, dtypes) in self.CONTRASTS:
            base = self.signal(length, silence_tail=silence_tail)
            for dtype in dtypes:
                y = (
                    base
                    if dtype == "float64"
                    else base.astype(np.float32).astype(np.float64)
                )
                spectrum = np.abs(
                    librosa.stft(
                        y,
                        n_fft=fft_size,
                        hop_length=hop,
                        win_length=fft_size,
                        window="hann",
                        center=alignment == "centered",
                        pad_mode="reflect",
                        dtype=np.complex128,
                    )
                )
                expected = librosa.feature.spectral_contrast(
                    S=spectrum,
                    sr=sr,
                    n_fft=fft_size,
                    fmin=f_min,
                    n_bands=n_bands,
                    quantile=quantile,
                    linear=linear,
                )
                assert expected.shape == (n_bands + 1, spectrum.shape[-1])
                cases.append(
                    {
                        "name": f"{key}_{dtype}",
                        "params": {
                            "sample_rate": sr,
                            "fft_size": fft_size,
                            "hop": hop,
                            "length": length,
                            "n_bands": n_bands,
                            "f_min": f_min,
                            "quantile": quantile,
                            "linear": linear,
                            "alignment": alignment,
                            "silence_tail": silence_tail,
                            "envelope": False,
                            "dtype": dtype,
                        },
                        "shape": list(expected.shape),
                        "values": expected.flatten().tolist(),
                    }
                )
        return cases

    # (case key, sample_rate, fft_size, hop, length, n_mels, lag, alignment,
    #  envelope, dtypes)
    ONSETS = [
        ("base", 22050, 512, 128, 1000, 40, 1, "centered", False,
         ["float64", "float32"]),
        ("lag2", 22050, 512, 128, 1000, 40, 2, "centered", False,
         ["float64"]),
        ("lag3_small", 8000, 128, 32, 400, 13, 3, "centered", False,
         ["float64"]),
        ("left", 8000, 128, 32, 400, 13, 1, "left", False, ["float64"]),
        # 6 frames, lag 2, shift 2: a partially zero prefix
        ("short_partial", 8000, 64, 16, 80, 6, 2, "centered", False,
         ["float64"]),
        # 6 frames, lag 7: the envelope degenerates to zeros
        ("short_degenerate", 8000, 64, 16, 80, 6, 7, "centered", False,
         ["float64"]),
        ("envelope", 22050, 512, 128, 1000, 40, 1, "centered", True,
         ["float64", "float32"]),
    ]

    def onset_cases(self):
        cases = []
        for (key, sr, fft_size, hop, length, n_mels, lag, alignment,
             envelope, dtypes) in self.ONSETS:
            base = self.signal(length, envelope=envelope)
            for dtype in dtypes:
                y = (
                    base
                    if dtype == "float64"
                    else base.astype(np.float32).astype(np.float64)
                )
                mel_kwargs = MelVectorGenerator.mel_kwargs(
                    n_mels, 0.0, 0.5 * sr, "slaney", "slaney"
                )
                if alignment == "centered":
                    expected = librosa.onset.onset_strength(
                        y=y,
                        sr=sr,
                        n_fft=fft_size,
                        hop_length=hop,
                        lag=lag,
                        center=True,
                        pad_mode="reflect",
                        **mel_kwargs,
                    )
                else:
                    log_mel = librosa.power_to_db(
                        librosa.feature.melspectrogram(
                            y=y,
                            sr=sr,
                            n_fft=fft_size,
                            hop_length=hop,
                            win_length=fft_size,
                            window="hann",
                            center=False,
                            power=2.0,
                            **mel_kwargs,
                        )
                    )
                    expected = librosa.onset.onset_strength(
                        S=log_mel, lag=lag, center=False
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
                            "lag": lag,
                            "alignment": alignment,
                            "envelope": envelope,
                            "silence_tail": False,
                            "dtype": dtype,
                        },
                        "shape": list(expected.shape),
                        "values": expected.flatten().tolist(),
                    }
                )
        return cases

    def generate(self):
        write_suite(self.SUITE, "spectral_contrast", self.contrast_cases())
        write_suite(self.SUITE, "onset_strength", self.onset_cases())


class CqtVectorGenerator:
    """Golden vectors for Soundml.Cqt.

    Three files:

    - cqt_tight: librosa.vqt magnitudes over settings whose octave recursion
      never resamples (odd hops, and single-octave configurations), on the
      deterministic LCG signal of the stft suite. Nothing but double-precision
      arithmetic separates the two implementations there, so these cases carry
      the strict end of the parity statement.
    - cqt_wide: librosa.vqt magnitudes over the settings that do resample —
      every octave step and the early decimation — on a harmonic test signal
      with three note onsets, a transient and a noise floor. SoundML decimates
      with its own resampler, so these cases pin the architecture at the
      tolerance that substitution costs.
    - cqt_support: the frequency ladder (librosa.cqt_frequencies), the
      fractional filter lengths and main-lobe cutoffs
      (librosa.filters.wavelet_lengths) across windows, librosa's own
      early-downsampling schedule, and the Nyquist admissibility boundary.

    Every case pins tuning=0.0 (librosa estimates it from the signal by
    default), sparsity=0 (librosa discards 1% of each filter's mass by
    default) and pad_mode="constant"; float32 cases quantize the signal to
    float32 and compute the reference in float64, exactly like the stft suite.
    """

    SUITE = "cqt"

    SEED = 20260803

    SR = 22050

    C1 = 32.70319566257483

    C4 = 261.6255653005986

    # Passed to every librosa.vqt call: the settings a golden must pin for the
    # comparison to mean anything.
    COMMON = dict(
        tuning=0.0,
        sparsity=0,
        norm=1,
        window="hann",
        filter_scale=1,
        pad_mode="constant",
        res_type="soxr_hq",
        dtype=np.complex128,
    )

    @staticmethod
    def lcg(n, seed=SEED):
        return StftVectorGenerator.lcg_signal(n, seed=seed)

    # (fundamental in hertz, onset as a fraction of the signal)
    NOTES = ((65.406, 0.0), (130.813, 0.23), (246.942, 0.55))

    @classmethod
    def harmonic(cls, n, sr):
        """Three decaying harmonic notes over a transient and a noise floor:
        a broadband signal with real octave structure, so that every octave of
        the recursion carries energy. Onsets are placed as fractions of the
        signal so that every length exercises all three notes."""
        t = np.arange(n) / sr
        y = np.zeros(n)
        for f0, fraction in cls.NOTES:
            onset = fraction * n / sr
            env = np.exp(-3.0 * np.maximum(t - onset, 0)) * (t >= onset)
            for h in range(1, 25):
                if f0 * h < 0.45 * sr:
                    y += (0.7**h) * env * np.sin(2 * np.pi * f0 * h * (t - onset))
        y[n // 4] += 3.0
        return 0.2 * y + 0.002 * cls.lcg(n)

    # (case key, signal, length, sample_rate, n_bins, bins_per_octave, fmin,
    #  hop, gamma, scale, dtypes)
    TIGHT = [
        ("t1_cqt84_hop511", "lcg", 8192, SR, 84, 12, C1, 511, 0,
         True, ["float64", "float32"]),
        ("t2_cqt36_c4_hop375", "lcg", 4096, SR, 36, 12, C4, 375, 0,
         True, ["float64"]),
        ("t3_vqt84_erb_hop511", "lcg", 8192, SR, 84, 12, C1, 511, None,
         True, ["float64"]),
        ("t4_cqt12_f1500_hop512", "lcg", 4096, SR, 12, 12, 1500.0, 512, 0,
         True, ["float64"]),
        ("t5_cqt24_bpo24_f2000_hop256", "lcg", 4096, SR, 24, 24, 2000.0, 256, 0,
         True, ["float64"]),
    ]

    WIDE = [
        ("w1_cqt84_hop512", "harmonic", 8192, SR, 84, 12, C1, 512, 0,
         True, ["float64", "float32"]),
        ("w2_cqt48_early8", "harmonic", 8192, SR, 48, 12, C1, 512, 0,
         True, ["float64"]),
        ("w3_cqt252_bpo36", "harmonic", 4096, SR, 252, 36, C1, 512, 0,
         True, ["float64", "float32"]),
        ("w4_vqt84_gamma20", "harmonic", 4096, SR, 84, 12, C1, 512, 20.0,
         True, ["float64"]),
        ("w5_vqt84_erb", "harmonic", 4096, SR, 84, 12, C1, 512, None,
         True, ["float64"]),
        ("w6_cqt90_hop256", "harmonic", 4096, SR, 90, 12, C1, 256, 0,
         True, ["float64"]),
        ("w7_cqt84_fmin55", "harmonic", 4096, SR, 84, 12, 55.0, 512, 0,
         True, ["float64"]),
        ("w8_cqt60_hop128", "harmonic", 4096, SR, 60, 12, C1, 128, 0,
         True, ["float64"]),
        ("w9_cqt84_sr44100", "harmonic", 8192, 44100, 84, 12, C1, 512, 0,
         True, ["float64"]),
        ("w10_cqt84_noscale", "harmonic", 8192, SR, 84, 12, C1, 512, 0,
         False, ["float64"]),
        ("w11_cqt48_early8_noscale", "harmonic", 8192, SR, 48, 12, C1, 512, 0,
         False, ["float64"]),
    ]

    # The short-signal edge: fewer samples than one octave of the recursion
    # comfortably frames, on the bit-exact LCG signal.
    SHORT = [
        ("s1_cqt84_n1024", "lcg", 1024, SR, 84, 12, C1, 512, 0,
         True, ["float64"]),
        ("s2_cqt84_n4096", "lcg", 4096, SR, 84, 12, C1, 512, 0,
         True, ["float64"]),
    ]

    @classmethod
    def signal(cls, kind, n, sr):
        return cls.lcg(n) if kind == "lcg" else cls.harmonic(n, sr)

    @classmethod
    def gamma_key(cls, gamma):
        return "erb" if gamma is None else repr(float(gamma))

    def transform_cases(self, rows):
        cases = []
        for (key, kind, length, sr, n_bins, bpo, fmin, hop, gamma, scale,
             dtypes) in rows:
            base = self.signal(kind, length, sr)
            for dtype in dtypes:
                y = (
                    base
                    if dtype == "float64"
                    else base.astype(np.float32).astype(np.float64)
                )
                expected = np.abs(
                    librosa.vqt(
                        y,
                        sr=sr,
                        hop_length=hop,
                        fmin=fmin,
                        n_bins=n_bins,
                        bins_per_octave=bpo,
                        gamma=gamma,
                        scale=scale,
                        **self.COMMON,
                    )
                )
                assert expected.shape[0] == n_bins
                cases.append(
                    {
                        "name": f"{key}_{dtype}",
                        "params": {
                            "signal": kind,
                            "length": length,
                            "sample_rate": sr,
                            "n_bins": n_bins,
                            "bins_per_octave": bpo,
                            "fmin": fmin,
                            "hop": hop,
                            "gamma": self.gamma_key(gamma),
                            "tuning": 0.0,
                            "scale": scale,
                            "dtype": dtype,
                        },
                        "shape": list(expected.shape),
                        "values": expected.flatten().tolist(),
                    }
                )
        return cases

    # (case key, n_bins, bins_per_octave, fmin, tuning)
    LADDERS = [
        ("ladder84_bpo12_c1", 84, 12, C1, 0.0),
        ("ladder252_bpo36_c1", 252, 36, C1, 0.0),
        ("ladder7_bpo12_c1", 7, 12, C1, 0.0),
        ("ladder1_bpo12_c1", 1, 12, C1, 0.0),
        ("ladder60_bpo24_a1", 60, 24, 55.0, 0.0),
        ("ladder84_bpo12_tuned", 84, 12, C1, 0.22),
        ("ladder36_bpo12_tuned_neg", 36, 12, C4, -0.35),
    ]

    # (case key, window spec, n_bins, bins_per_octave, fmin, gamma, scale,
    #  sample_rate)
    SUPPORTS = [
        ("lengths84_hann", "hann", 84, 12, C1, 0, 1.0, SR),
        ("lengths84_bkh", "blackmanharris", 84, 12, C1, 0, 1.0, SR),
        ("lengths84_kaiser86", ("kaiser", 8.6), 84, 12, C1, 0, 1.0, SR),
        ("lengths252_hann_bpo36", "hann", 252, 36, C1, 0, 1.0, SR),
        ("lengths84_erb", "hann", 84, 12, C1, None, 1.0, SR),
        ("lengths84_gamma20", "hann", 84, 12, C1, 20.0, 1.0, SR),
        ("lengths84_scale05", "hann", 84, 12, C1, 0, 0.5, SR),
        ("lengths84_sr44100", "hann", 84, 12, C1, 0, 1.0, 44100),
    ]

    @classmethod
    def ladder(cls, n_bins, bpo, fmin, tuning=0.0):
        """The centre frequencies librosa.vqt itself analyses at: the tuned
        fmin walked octave by octave."""
        return librosa.interval_frequencies(
            n_bins=n_bins,
            fmin=fmin * 2.0 ** (tuning / bpo),
            intervals="equal",
            bins_per_octave=bpo,
            sort=True,
        )

    @classmethod
    def alphas(cls, freqs, bpo):
        if len(freqs) == 1:
            return getattr(librosa.core.constantq, "__et_relative_bw")(bpo)
        return librosa.filters._relative_bandwidth(freqs=freqs)

    def ladder_cases(self):
        cases = []
        cls_sr = self.SR
        for key, n_bins, bpo, fmin, tuning in self.LADDERS:
            expected = librosa.cqt_frequencies(
                n_bins=n_bins, fmin=fmin, bins_per_octave=bpo, tuning=tuning
            )
            cases.append(
                {
                    "name": key,
                    "params": {
                        "kind": "frequencies",
                        "n_bins": n_bins,
                        "bins_per_octave": bpo,
                        "fmin": fmin,
                        "tuning": tuning,
                        "sample_rate": cls_sr,
                    },
                    "shape": [n_bins],
                    "values": expected.tolist(),
                }
            )
        return cases

    def length_cases(self):
        cases = []
        for (key, window, n_bins, bpo, fmin, gamma, filter_scale,
             sr) in self.SUPPORTS:
            freqs = self.ladder(n_bins, bpo, fmin)
            lengths, cutoff = librosa.filters.wavelet_lengths(
                freqs=freqs,
                sr=sr,
                window=window,
                filter_scale=filter_scale,
                gamma=gamma,
                alpha=self.alphas(freqs, bpo),
            )
            params = {
                "kind": "filter_lengths",
                "n_bins": n_bins,
                "bins_per_octave": bpo,
                "fmin": fmin,
                "tuning": 0.0,
                "gamma": self.gamma_key(gamma),
                "filter_scale": filter_scale,
                "window": window if isinstance(window, str) else window[0],
                "beta": 0.0 if isinstance(window, str) else window[1],
                "sample_rate": sr,
            }
            cases.append(
                {
                    "name": key,
                    "params": params,
                    "shape": [n_bins],
                    "values": lengths.tolist(),
                }
            )
            cases.append(
                {
                    "name": f"{key}_cutoff",
                    "params": dict(params, kind="cutoff"),
                    "shape": [1],
                    "values": [float(cutoff)],
                }
            )
        return cases

    def early_cases(self):
        """librosa's own early-downsampling schedule: how many factors of two
        the signal is decimated by before the first octave."""
        count = getattr(librosa.core.constantq, "__early_downsample_count")
        cases = []
        for hop in (128, 256, 512):
            for n_bins in (12, 24, 36, 48, 60, 72, 84):
                freqs = self.ladder(n_bins, 12, self.C1)
                _, cutoff = librosa.filters.wavelet_lengths(
                    freqs=freqs,
                    sr=self.SR,
                    window="hann",
                    filter_scale=1,
                    gamma=0,
                    alpha=self.alphas(freqs, 12),
                )
                n_octaves = int(np.ceil(n_bins / 12))
                early = count(self.SR / 2.0, cutoff, hop, n_octaves)
                cases.append(
                    {
                        "name": f"early_n{n_bins}_hop{hop}",
                        "params": {
                            "kind": "early",
                            "n_bins": n_bins,
                            "bins_per_octave": 12,
                            "fmin": self.C1,
                            "tuning": 0.0,
                            "hop": hop,
                            "sample_rate": self.SR,
                        },
                        "shape": [1],
                        "values": [float(early)],
                    }
                )
        return cases

    def nyquist_cases(self):
        """The admissibility boundary: the largest n_bins whose filters stay
        under Nyquist, and the first that does not."""
        cases = []
        for n_bins, accepted in ((100, True), (101, True), (102, False)):
            freqs = self.ladder(n_bins, 12, self.C1)
            _, cutoff = librosa.filters.wavelet_lengths(
                freqs=freqs,
                sr=self.SR,
                window="hann",
                filter_scale=1,
                gamma=0,
                alpha=self.alphas(freqs, 12),
            )
            assert (cutoff <= self.SR / 2.0) == accepted, (n_bins, cutoff)
            cases.append(
                {
                    "name": f"nyquist_n{n_bins}",
                    "params": {
                        "kind": "nyquist",
                        "n_bins": n_bins,
                        "bins_per_octave": 12,
                        "fmin": self.C1,
                        "tuning": 0.0,
                        "sample_rate": self.SR,
                    },
                    "shape": [1],
                    "values": [1.0 if accepted else 0.0],
                }
            )
        return cases

    def generate(self):
        write_suite(self.SUITE, "cqt_tight", self.transform_cases(self.TIGHT))
        write_suite(
            self.SUITE,
            "cqt_wide",
            self.transform_cases(self.WIDE) + self.transform_cases(self.SHORT),
        )
        write_suite(
            self.SUITE,
            "cqt_support",
            self.ladder_cases()
            + self.length_cases()
            + self.early_cases()
            + self.nyquist_cases(),
        )


class ChromaVectorGenerator:
    """Golden vectors for Soundml.Chroma and the flat chroma features.

    Three files:

    - chroma_fb: librosa.filters.chroma weight matrices in float64 (librosa
      builds them in float32 by default, so every call passes
      dtype=np.float64), across n_chroma including an odd one, both octave
      envelopes, both base rotations and nonzero tunings; plus
      librosa.filters.cq_to_chroma assignment matrices, which are exactly 0/1.
    - chroma_stft: librosa.feature.chroma_stft end-to-end over the CQT suite's
      harmonic signal, covering the three normalisations this library offers.
    - chroma_cqt: librosa.feature.chroma_cqt over a constant-Q transform
      computed with sparsity=0 and passed in as C, since the y= path would
      carry librosa's default sparsified filter basis; one case uses an odd hop
      so that neither side resamples.

    float32 cases quantize the signal to float32 and compute the reference in
    float64, exactly like the stft suite.
    """

    SUITE = "chroma"

    SR = CqtVectorGenerator.SR

    C1 = CqtVectorGenerator.C1

    # (case key, sample_rate, fft_size, n_chroma, tuning, ctroct, octwidth,
    #  base_c)
    FILTERBANKS = [
        ("fb64_c12", SR, 64, 12, 0.0, 5.0, 2.0, True),
        ("fb64_c12_flat", SR, 64, 12, 0.0, 5.0, None, True),
        ("fb64_c24", SR, 64, 24, 0.0, 5.0, 2.0, True),
        ("fb128_c12", SR, 128, 12, 0.0, 5.0, 2.0, True),
        ("fb128_c12_basea", SR, 128, 12, 0.0, 5.0, 2.0, False),
        ("fb128_c12_tuned", SR, 128, 12, 0.22, 5.0, 2.0, True),
        ("fb128_c13_odd", SR, 128, 13, 0.0, 5.0, 2.0, True),
        ("fb128_c24_tuned_neg", SR, 128, 24, -0.35, 4.0, 1.5, True),
        ("fb128_c12_sr44100", 44100, 128, 12, 0.0, 5.0, 2.0, True),
    ]

    # (case key, n_bins, bins_per_octave, n_chroma, fmin)
    PROJECTIONS = [
        ("cq84_bpo12_c12", 84, 12, 12, C1),
        ("cq252_bpo36_c12", 252, 36, 12, C1),
        ("cq120_bpo24_c24", 120, 24, 24, C1),
        ("cq120_bpo24_c12", 120, 24, 12, C1),
        ("cq60_bpo12_c12_a1", 60, 12, 12, 55.0),
        ("cq84_bpo12_c4", 84, 12, 4, C1),
    ]

    def filterbank_cases(self):
        cases = []
        for (key, sr, fft_size, n_chroma, tuning, ctroct, octwidth,
             base_c) in self.FILTERBANKS:
            weights = librosa.filters.chroma(
                sr=sr,
                n_fft=fft_size,
                n_chroma=n_chroma,
                tuning=tuning,
                ctroct=ctroct,
                octwidth=octwidth,
                base_c=base_c,
                dtype=np.float64,
            )
            assert weights.shape == (n_chroma, 1 + fft_size // 2)
            cases.append(
                {
                    "name": key,
                    "params": {
                        "kind": "filterbank",
                        "sample_rate": sr,
                        "fft_size": fft_size,
                        "n_chroma": n_chroma,
                        "tuning": tuning,
                        "ctroct": ctroct,
                        "octwidth": octwidth,
                        "base_c": base_c,
                    },
                    "shape": list(weights.shape),
                    "values": weights.flatten().tolist(),
                }
            )
        return cases

    def projection_cases(self):
        cases = []
        for key, n_bins, bpo, n_chroma, fmin in self.PROJECTIONS:
            weights = librosa.filters.cq_to_chroma(
                n_bins,
                bins_per_octave=bpo,
                n_chroma=n_chroma,
                fmin=fmin,
                base_c=True,
                dtype=np.float64,
            )
            assert set(np.unique(weights)) <= {0.0, 1.0}
            cases.append(
                {
                    "name": key,
                    "params": {
                        "kind": "cqt_projection",
                        "sample_rate": self.SR,
                        "n_bins": n_bins,
                        "bins_per_octave": bpo,
                        "n_chroma": n_chroma,
                        "fmin": fmin,
                        "tuning": 0.0,
                        "hop": 512,
                    },
                    "shape": list(weights.shape),
                    "values": weights.flatten().tolist(),
                }
            )
        return cases

    NORMS = {"inf": np.inf, "l2": 2, "l1": 1, "none": None}

    # (case key, length, fft_size, hop, n_chroma, norm, power, dtype)
    SPECTRA = [
        ("stft_n2048_c12_inf", 8192, 2048, 512, 12, "inf", 2.0, "float64"),
        ("stft_n4096_c12_l2", 8192, 4096, 1024, 12, "l2", 2.0, "float64"),
        ("stft_n2048_c24_none", 8192, 2048, 512, 24, "none", 2.0, "float64"),
        ("stft_n2048_c12_l1_mag", 8192, 2048, 512, 12, "l1", 1.0, "float64"),
        ("stft_n2048_c12_inf", 8192, 2048, 512, 12, "inf", 2.0, "float32"),
    ]

    def spectrum_cases(self):
        cases = []
        for (key, length, fft_size, hop, n_chroma, norm, power,
             dtype) in self.SPECTRA:
            base = CqtVectorGenerator.harmonic(length, self.SR)
            y = (
                base
                if dtype == "float64"
                else base.astype(np.float32).astype(np.float64)
            )
            spectrum = (
                np.abs(
                    librosa.stft(
                        y,
                        n_fft=fft_size,
                        hop_length=hop,
                        window="hann",
                        center=True,
                        pad_mode="constant",
                        dtype=np.complex128,
                    )
                )
                ** power
            )
            expected = librosa.feature.chroma_stft(
                S=spectrum,
                sr=self.SR,
                n_fft=fft_size,
                tuning=0.0,
                n_chroma=n_chroma,
                norm=self.NORMS[norm],
                dtype=np.float64,
            )
            cases.append(
                {
                    "name": f"{key}_{dtype}",
                    "params": {
                        "kind": "chroma_stft",
                        "signal": "harmonic",
                        "length": length,
                        "sample_rate": self.SR,
                        "fft_size": fft_size,
                        "hop": hop,
                        "n_chroma": n_chroma,
                        "tuning": 0.0,
                        "norm": norm,
                        "power": power,
                        "dtype": dtype,
                    },
                    "shape": list(expected.shape),
                    "values": expected.flatten().tolist(),
                }
            )
        return cases

    # (case key, length, n_bins, bins_per_octave, fmin, hop, n_chroma, norm,
    #  dtype)
    CONSTANT_Q = [
        ("cqt84_hop512_inf", 8192, 84, 12, C1, 512, 12, "inf", "float64"),
        ("cqt252_bpo36_hop512_l2", 4096, 252, 36, C1, 512, 12, "l2", "float64"),
        ("cqt120_bpo24_c24_none", 8192, 120, 24, C1, 512, 24, "none", "float64"),
        ("cqt84_hop511_inf", 8192, 84, 12, C1, 511, 12, "inf", "float64"),
        ("cqt84_hop512_inf", 8192, 84, 12, C1, 512, 12, "inf", "float32"),
    ]

    def constant_q_cases(self):
        cases = []
        for (key, length, n_bins, bpo, fmin, hop, n_chroma, norm,
             dtype) in self.CONSTANT_Q:
            base = CqtVectorGenerator.harmonic(length, self.SR)
            y = (
                base
                if dtype == "float64"
                else base.astype(np.float32).astype(np.float64)
            )
            transform = np.abs(
                librosa.vqt(
                    y,
                    sr=self.SR,
                    hop_length=hop,
                    fmin=fmin,
                    n_bins=n_bins,
                    bins_per_octave=bpo,
                    gamma=0,
                    scale=True,
                    **CqtVectorGenerator.COMMON,
                )
            )
            expected = librosa.feature.chroma_cqt(
                C=transform,
                sr=self.SR,
                hop_length=hop,
                fmin=fmin,
                n_chroma=n_chroma,
                n_octaves=int(np.ceil(n_bins / bpo)),
                bins_per_octave=bpo,
                norm=self.NORMS[norm],
                threshold=0.0,
            )
            cases.append(
                {
                    "name": f"{key}_{dtype}",
                    "params": {
                        "kind": "chroma_cqt",
                        "signal": "harmonic",
                        "length": length,
                        "sample_rate": self.SR,
                        "n_bins": n_bins,
                        "bins_per_octave": bpo,
                        "fmin": fmin,
                        "hop": hop,
                        "n_chroma": n_chroma,
                        "gamma": "0.0",
                        "tuning": 0.0,
                        "scale": True,
                        "norm": norm,
                        "dtype": dtype,
                    },
                    "shape": list(expected.shape),
                    "values": expected.flatten().tolist(),
                }
            )
        return cases

    def generate(self):
        write_suite(
            self.SUITE,
            "chroma_fb",
            self.filterbank_cases() + self.projection_cases(),
        )
        write_suite(self.SUITE, "chroma_stft", self.spectrum_cases())
        write_suite(self.SUITE, "chroma_cqt", self.constant_q_cases())


class HpssVectorGenerator:
    """Golden vectors for the harmonic/percussive separation features.

    Four files:

    - hpss: librosa.decompose.hpss over a deterministic magnitude
      spectrogram — harmonic ridges every seventh bin, percussive columns
      every fifth frame, a 31-bit LCG noise floor and a silent top band,
      rebuilt bit-exactly in OCaml — across the kernel x power x margin x
      dtype grid, one case per component. The silent band drives the
      denormal branch of the mask (both enhanced values below the smallest
      positive normal), asserted to bind. One small-kernel cell keeps the
      interior of the sliding window under test at a size the boundary does
      not reach, and one batched cell pins the leading-axis mapping.

    - hpss_masks: the mask pair itself over the same spectrogram, including
      both margins at infinite power, where the values are exactly 0 and 1,
      and a wide-silence row at finite power and a margin above one, where
      the denormal branch of the mask is 0 rather than an even split.

    - hpss_boundary: frame counts at and just below the kernel size, where
      every window overhangs the frame axis, plus a short pair kernel.

    - hpss_effects: librosa.effects.hpss / harmonic / percussive on a short
      signal, with the analysis geometry passed explicitly (constant padding,
      the pad mode the OCaml configuration names) and the synthesis length
      pinned to the input length.

    Every emitted median stays inside k <= 2n + 1 on its axis, the region
    where scipy.ndimage.median_filter agrees with the reflection the library
    documents; the generator asserts both the bound and the agreement per
    case and refuses to emit otherwise.

    The spectrogram-domain float32 cases carry a genuine float32 reference,
    not the float64 reference of the stft and mel suites: separation selects
    values and multiplies them, with no accumulation whose order could
    differ, so a float32-in/float32-out reference is exactly reproducible.
    The effects cases have the STFT round trip beneath them and follow the
    usual convention instead — the signal is quantized to float32 and the
    reference computed in float64.
    """

    SUITE = "hpss"

    SEED = 20260811

    SIGNAL_SEED = 20260812

    BINS = 17

    FRAMES = 17

    SILENT_BINS = 3

    # a band wide enough that the frequency median of the default kernel is
    # zero inside it, which is what drives the denormal branch of the mask
    SILENT_WIDE = 9

    @staticmethod
    def spectrogram(bins, frames, seed, silent_bins):
        """A deterministic magnitude spectrogram: the 31-bit LCG of the stft
        suite folded to non-negative values, plus a constant ridge on every
        seventh bin (sustained partials, horizontal), a constant column on
        every fifth frame (transients, vertical), and a silent band of
        `silent_bins` at the top. Every value is one LCG draw and at most two
        additions of exactly representable constants, in this order, so the
        OCaml side rebuilds the same doubles bit for bit."""
        noise = np.abs(StftVectorGenerator.lcg_signal(bins * frames, seed=seed))
        s = np.empty((bins, frames), dtype=np.float64)
        for b in range(bins):
            for t in range(frames):
                v = noise[b * frames + t]
                v = v + (3.0 if b % 7 == 3 else 0.0)
                v = v + (2.0 if t % 5 == 2 else 0.0)
                s[b, t] = 0.0 if b >= bins - silent_bins else v
        return s

    @staticmethod
    def refl(i, n):
        p = 2 * n
        j = i % p
        return j if j < n else p - 1 - j

    @classmethod
    def median_model(cls, s, k, axis):
        """The library's documented filter: rank k // 2 of the window
        [i - k // 2, i + k - 1 - k // 2], indices reflected
        half-sample-symmetrically with period 2n."""
        x = s if axis == -1 else np.swapaxes(s, -1, -2)
        n = x.shape[-1]
        gather = np.array(
            [[cls.refl(i - k // 2 + t, n) for t in range(k)] for i in range(n)]
        )
        out = np.sort(x[..., gather], axis=-1)[..., k // 2]
        return out if axis == -1 else np.swapaxes(out, -1, -2)

    @classmethod
    def assert_clean(cls, s, kernel_size):
        """Refuse to emit a case whose kernel leaves the region where the
        reference median filter and the documented reflection agree."""
        k_h, k_p = kernel_size
        bins, frames = s.shape[-2], s.shape[-1]
        assert k_h <= 2 * frames + 1, f"k_h={k_h} exceeds 2 * {frames} + 1"
        assert k_p <= 2 * bins + 1, f"k_p={k_p} exceeds 2 * {bins} + 1"
        harm_shape = [1] * s.ndim
        harm_shape[-1] = k_h
        perc_shape = [1] * s.ndim
        perc_shape[-2] = k_p
        assert np.array_equal(
            scipy.ndimage.median_filter(s, size=harm_shape, mode="reflect"),
            cls.median_model(s, k_h, -1),
        ), f"time median disagrees with the model at k={k_h}, n={frames}"
        assert np.array_equal(
            scipy.ndimage.median_filter(s, size=perc_shape, mode="reflect"),
            cls.median_model(s, k_p, -2),
        ), f"freq median disagrees with the model at k={k_p}, n={bins}"

    @staticmethod
    def key(kernel_size, power, margin, dtype):
        k_h, k_p = kernel_size
        m_h, m_p = margin
        return (
            f"k{k_h}x{k_p}_p{'inf' if np.isinf(power) else f'{power:g}'}"
            f"_m{m_h:g}x{m_p:g}_{dtype}"
        )

    @classmethod
    def cell(
        cls, s, kernel_size, power, margin, dtype, silent_bins=None,
        extra=None, mask=False, prefix="",
    ):
        """One (kernel, power, margin, dtype) cell as two cases, the harmonic
        component (or mask) and the percussive one."""
        cls.assert_clean(s, kernel_size)
        source = s if dtype == "float64" else s.astype(np.float32)
        harm, perc = librosa.decompose.hpss(
            source, kernel_size=list(kernel_size), power=power,
            margin=list(margin), mask=mask,
        )
        if not mask:
            assert harm.dtype == source.dtype and perc.dtype == source.dtype
        cases = []
        for component, expected in (("harmonic", harm), ("percussive", perc)):
            params = {
                "bins": int(s.shape[-2]),
                "frames": int(s.shape[-1]),
                "planes": int(s.size // (s.shape[-2] * s.shape[-1])),
                "seed": cls.SEED,
                "silent_bins": cls.SILENT_BINS
                if silent_bins is None
                else silent_bins,
                "kernel_h": kernel_size[0],
                "kernel_p": kernel_size[1],
                "power": "inf" if np.isinf(power) else f"{power:g}",
                "margin_h": float(margin[0]),
                "margin_p": float(margin[1]),
                "dtype": dtype,
                "component": component,
            }
            if extra:
                params.update(extra)
            cases.append(
                {
                    "name": f"{prefix}"
                    f"{cls.key(kernel_size, power, margin, dtype)}_{component}",
                    "params": params,
                    "shape": list(expected.shape),
                    "values": expected.astype(np.float64).flatten().tolist(),
                }
            )
        return cases

    KERNELS = [(31, 31), (17, 31), (32, 32)]

    POWERS = [1.0, 2.0, np.inf]

    MARGINS = [(1.0, 1.0), (1.0, 3.0)]

    DTYPES = ["float64", "float32"]

    @staticmethod
    def assert_denormal(s, kernel_size, margin=(1.0, 1.0)):
        """The denormal branch of the mask must be reachable: the pointwise
        maximum that rescales a mask — (harm, margin_h * perc) for the
        harmonic one, (perc, margin_p * harm) for the percussive one — below
        the smallest positive normal somewhere. Returns where it binds for
        both masks at once."""
        harm = scipy.ndimage.median_filter(
            s, size=[1, kernel_size[0]], mode="reflect"
        )
        perc = scipy.ndimage.median_filter(
            s, size=[kernel_size[1], 1], mode="reflect"
        )
        bad = (
            np.minimum(
                np.maximum(harm, perc * margin[0]),
                np.maximum(perc, harm * margin[1]),
            )
            < np.finfo(np.float64).tiny
        )
        assert (
            bad.any()
        ), "the silent band no longer reaches the denormal branch"
        return bad

    def component_cases(self):
        s = self.spectrogram(self.BINS, self.FRAMES, self.SEED, self.SILENT_BINS)
        cases = []
        for kernel_size in self.KERNELS:
            for power in self.POWERS:
                for margin in self.MARGINS:
                    for dtype in self.DTYPES:
                        cases += self.cell(s, kernel_size, power, margin, dtype)
        # a kernel small enough that the window is interior over most of the
        # spectrogram, which the 31-frame kernels never are at this size, and
        # short enough for the silent band to zero both medians
        self.assert_denormal(s, (3, 5))
        for dtype in self.DTYPES:
            cases += self.cell(s, (3, 5), 2.0, (1.0, 1.0), dtype)
        # the denormal branch under the default kernel needs a wider silence
        wide = self.spectrogram(
            self.BINS, self.FRAMES, self.SEED, self.SILENT_WIDE
        )
        self.assert_denormal(wide, (31, 31))
        for power in (2.0, np.inf):
            for margin in self.MARGINS:
                cases += self.cell(
                    wide, (31, 31), power, margin, "float64",
                    silent_bins=self.SILENT_WIDE, prefix="silent_",
                )
        # leading axes map independently: two planes, the second halved
        batched = np.stack([s, s * 0.5])
        cases += self.cell(
            batched, (31, 31), 2.0, (1.0, 1.0), "float64",
            extra={"batched": True}, prefix="batched_",
        )
        for case in cases:
            case["params"].setdefault("batched", False)
        return cases

    # (kernel_size, power, margin, dtype, silent bins). The last row widens
    # the silence until the default kernel zeroes both medians inside it: at
    # a margin above one the pair is no longer a partition, so the undefined
    # quotient there is 0 rather than an even split of 0.5, and at finite
    # power the mask is the emitted value itself rather than a factor
    # multiplying a silent bin.
    MASK_CELLS = [
        ((31, 31), 2.0, (1.0, 1.0), "float64", SILENT_BINS),
        ((31, 31), 2.0, (1.0, 1.0), "float32", SILENT_BINS),
        ((31, 31), 1.0, (1.0, 3.0), "float64", SILENT_BINS),
        ((31, 31), np.inf, (1.0, 1.0), "float64", SILENT_BINS),
        ((31, 31), np.inf, (1.0, 3.0), "float32", SILENT_BINS),
        ((17, 31), np.inf, (1.0, 1.0), "float64", SILENT_BINS),
        ((17, 31), np.inf, (1.0, 3.0), "float64", SILENT_BINS),
        ((32, 32), np.inf, (1.0, 1.0), "float32", SILENT_BINS),
        ((32, 32), 2.0, (1.0, 3.0), "float32", SILENT_BINS),
        ((3, 5), 1.0, (1.0, 1.0), "float64", SILENT_BINS),
        ((31, 31), 2.0, (1.0, 3.0), "float64", SILENT_WIDE),
    ]

    def mask_cases(self):
        cases = []
        for kernel_size, power, margin, dtype, silent_bins in self.MASK_CELLS:
            s = self.spectrogram(
                self.BINS, self.FRAMES, self.SEED, silent_bins
            )
            wide = silent_bins != self.SILENT_BINS
            bad = self.assert_denormal(s, kernel_size, margin) if wide else None
            emitted = self.cell(
                s, kernel_size, power, margin, dtype, mask=True,
                silent_bins=silent_bins, prefix="silent_" if wide else "",
            )
            if wide:
                for case in emitted:
                    values = np.asarray(case["values"]).reshape(case["shape"])
                    assert (
                        values[bad] == 0.0
                    ).all(), "the denormal branch off the equal margins is 0"
            cases += emitted
        for case in cases:
            case["params"]["batched"] = False
            if case["params"]["power"] == "inf":
                values = np.asarray(case["values"])
                assert np.isin(values, [0.0, 1.0]).all(), "hard mask is not 0/1"
        return cases

    # (frames, kernel_size): the kernel at, above and around the frame count,
    # every window overhanging the frame axis
    BOUNDARY_CELLS = [
        (16, (31, 31)),
        (16, (32, 32)),
        (17, (31, 31)),
        (17, (32, 32)),
        (9, (17, 31)),
    ]

    def boundary_cases(self):
        cases = []
        for frames, kernel_size in self.BOUNDARY_CELLS:
            s = self.spectrogram(
                self.BINS, frames, self.SEED, self.SILENT_BINS
            )
            prefix = f"n{frames}_"
            for dtype in self.DTYPES:
                cases += self.cell(
                    s, kernel_size, 2.0, (1.0, 1.0), dtype, prefix=prefix
                )
            cases += self.cell(
                s, kernel_size, np.inf, (1.0, 1.0), "float64", prefix=prefix
            )
        for case in cases:
            case["params"]["batched"] = False
        return cases

    EFFECTS_FFT = 512

    EFFECTS_HOP = 128

    EFFECTS_LENGTH = 3072

    # (case key, face, kernel_size, power, margin, dtype)
    EFFECTS_CELLS = [
        ("defaults", "hpss", (31, 31), 2.0, (1.0, 1.0), "float64"),
        ("defaults", "hpss", (31, 31), 2.0, (1.0, 1.0), "float32"),
        ("hard", "harmonic", (17, 31), np.inf, (1.0, 1.0), "float64"),
        ("margin", "percussive", (31, 31), 2.0, (1.0, 3.0), "float64"),
    ]

    def effects_cases(self):
        base = StftVectorGenerator.lcg_signal(
            self.EFFECTS_LENGTH, seed=self.SIGNAL_SEED
        )
        analysis = dict(
            n_fft=self.EFFECTS_FFT,
            hop_length=self.EFFECTS_HOP,
            win_length=self.EFFECTS_FFT,
            center=True,
            pad_mode="constant",
        )
        cases = []
        for key, face, kernel_size, power, margin, dtype in self.EFFECTS_CELLS:
            y = (
                base
                if dtype == "float64"
                else base.astype(np.float32).astype(np.float64)
            )
            spectrum = librosa.stft(y, **analysis)
            self.assert_clean(np.abs(spectrum), kernel_size)
            shared = dict(
                kernel_size=list(kernel_size), power=power, margin=list(margin)
            )
            if face == "hpss":
                outputs = list(librosa.effects.hpss(y, **analysis, **shared))
                components = ["harmonic", "percussive"]
            elif face == "harmonic":
                outputs = [librosa.effects.harmonic(y, **analysis, **shared)]
                components = ["harmonic"]
            else:
                outputs = [librosa.effects.percussive(y, **analysis, **shared)]
                components = ["percussive"]
            for component, expected in zip(components, outputs):
                assert expected.shape == (self.EFFECTS_LENGTH,)
                cases.append(
                    {
                        "name": f"{key}_{face}_{component}_{dtype}",
                        "params": {
                            "fft_size": self.EFFECTS_FFT,
                            "hop": self.EFFECTS_HOP,
                            "length": self.EFFECTS_LENGTH,
                            "seed": self.SIGNAL_SEED,
                            "kernel_h": kernel_size[0],
                            "kernel_p": kernel_size[1],
                            "power": "inf" if np.isinf(power) else f"{power:g}",
                            "margin_h": float(margin[0]),
                            "margin_p": float(margin[1]),
                            "dtype": dtype,
                            "face": face,
                            "component": component,
                        },
                        "shape": list(expected.shape),
                        "values": expected.astype(np.float64).tolist(),
                    }
                )
        return cases

    def generate(self):
        write_suite(self.SUITE, "hpss", self.component_cases())
        write_suite(self.SUITE, "hpss_masks", self.mask_cases())
        write_suite(self.SUITE, "hpss_boundary", self.boundary_cases())
        write_suite(self.SUITE, "hpss_effects", self.effects_cases())


class IoVectorGenerator:
    """Golden fixtures and decode-parity vectors for Soundml_io.

    Writes three artifact sets, all committed:

      - soundml-io/test/corpus/: small deterministic audio fixtures (seeded
        chirp+noise in the bench-corpus recipe), written with
        python-soundfile across the tested container/encoding matrix;
      - soundml-io/test/corpus/malformed/: the malformed-input corpus — empty and
        garbage files, truncations at pinned offsets, header liars — with
        every seed and offset recorded in MANIFEST;
      - soundml-io/test/vectors/: python-soundfile float64 decodes of the corpus
        fixtures, stored planar [channels; frames] (JSON floats round-trip
        float64 exactly), plus the write-clipping golden. The float32
        decode is asserted here to equal the correctly-rounded float32 cast
        of the float64 decode for every fixture, so the OCaml harness
        derives its float32 expectation from the same file; a break in that
        law fails regeneration, not the replay.

    The reference is python-soundfile (its bundled libsndfile); the io
    vector files record both versions beside the base stack.
    """

    SUITE = "io"

    CORPUS_DIRECTORY = os.path.join(REPO_ROOT, "soundml-io", "test", "corpus")

    # (name, soundfile format, subtype, extension, rate, channels, frames, seed)
    PARITY_CASES = [
        ("wav_pcm16_22050_mono", "WAV", "PCM_16", "wav", 22050, 1, 2205, 101),
        ("wav_pcm16_44100_stereo", "WAV", "PCM_16", "wav", 44100, 2, 2205, 102),
        ("wav_pcm24_22050_stereo", "WAV", "PCM_24", "wav", 22050, 2, 2205, 103),
        ("wav_pcm32_22050_mono", "WAV", "PCM_32", "wav", 22050, 1, 2205, 104),
        ("wav_float32_22050_stereo", "WAV", "FLOAT", "wav", 22050, 2, 2205, 105),
        ("wav_float64_22050_mono", "WAV", "DOUBLE", "wav", 22050, 1, 2205, 106),
        ("aiff_pcm16_22050_mono", "AIFF", "PCM_16", "aiff", 22050, 1, 2205, 107),
        ("caf_pcm16_22050_stereo", "CAF", "PCM_16", "caf", 22050, 2, 2205, 108),
        ("flac_pcm16_22050_mono", "FLAC", "PCM_16", "flac", 22050, 1, 2205, 109),
        ("flac_pcm24_22050_stereo", "FLAC", "PCM_24", "flac", 22050, 2, 2205, 110),
        ("ogg_vorbis_22050_mono", "OGG", "VORBIS", "ogg", 22050, 1, 6615, 111),
        ("ogg_vorbis_44100_stereo", "OGG", "VORBIS", "ogg", 44100, 2, 6615, 112),
    ]

    MALFORMED_BASE_SEED = 4242
    GARBAGE_SEED = 97531
    TRUNCATION_FRACTION = 0.6
    MID_HEADER_OFFSET = 20

    def versions(self):
        return {
            **GENERATOR_VERSIONS,
            "soundfile": soundfile.__version__,
            "libsndfile": soundfile.__libsndfile_version__,
        }

    def signal(self, frames, rate, channels, seed):
        """The bench-corpus recipe: per-channel chirp + lowpassed noise,
        globally scaled to 0.60 peak. Deterministic in (frames, rate,
        channels, seed)."""
        rng = np.random.default_rng(seed)
        t = np.arange(frames, dtype=np.float64) / rate
        dur = frames / rate
        out = np.empty((frames, channels), dtype=np.float64)
        for c in range(channels):
            f0, f1 = 220.0 * (c + 1), 2000.0
            sweep = 0.45 * np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * dur) * t * t))
            noise = scipy.signal.lfilter([0.15], [1.0, -0.85], rng.standard_normal(frames))
            noise *= 0.15 / np.abs(noise).max()
            out[:, c] = sweep + noise
        out *= 0.60 / np.abs(out).max()
        return out

    def decode_case(self, name, path, relative, params):
        """Decode `path` with python-soundfile in both dtypes, assert the
        float32 = float32-cast-of-float64 law, and return the float64 case
        (values planar [channels; frames], C order)."""
        data64, rate = soundfile.read(path, dtype="float64", always_2d=True)
        data32, rate32 = soundfile.read(path, dtype="float32", always_2d=True)
        assert rate == rate32 and data64.shape == data32.shape, name
        assert np.array_equal(data32, data64.astype(np.float32)), name
        planar = np.ascontiguousarray(data64.T)
        return {
            "name": name,
            "params": {"file": relative, "sample_rate": rate, **params},
            "shape": list(planar.shape),
            "values": planar.flatten().tolist(),
        }

    def parity_cases(self):
        os.makedirs(self.CORPUS_DIRECTORY, exist_ok=True)
        cases = []
        for name, fmt, subtype, ext, rate, channels, frames, seed in self.PARITY_CASES:
            filename = f"{name}.{ext}"
            path = os.path.join(self.CORPUS_DIRECTORY, filename)
            soundfile.write(
                path,
                self.signal(frames, rate, channels, seed),
                rate,
                format=fmt,
                subtype=subtype,
            )
            params = {
                "container": fmt.lower(),
                "encoding": subtype.lower(),
                "seed": seed,
            }
            cases.append(self.decode_case(name, path, f"corpus/{filename}", params))
        return cases

    def clipping_cases(self):
        """The write-clipping golden: a ±1.5 full-scale ramp on the exact
        k/64 grid, encoded to WAV/PCM_16 by python-soundfile (which sets
        SFC_SET_CLIPPING, as Soundml_io.write does), decoded back in
        float64. Out-of-range samples saturate at full scale."""
        ramp = np.arange(-96, 97, dtype=np.float64) / 64.0
        buffer = io.BytesIO()
        soundfile.write(buffer, ramp, 22050, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        decoded, rate = soundfile.read(buffer, dtype="float64", always_2d=True)
        assert rate == 22050 and decoded.shape == (193, 1)
        assert decoded[0, 0] == -1.0 and decoded[-1, 0] == 32767.0 / 32768.0
        planar = np.ascontiguousarray(decoded.T)
        return [
            {
                "name": "pcm16_clip_ramp",
                "params": {
                    "container": "wav",
                    "encoding": "pcm_16",
                    "sample_rate": 22050,
                    "lo": -96,
                    "hi": 96,
                    "denominator": 64,
                },
                "shape": list(planar.shape),
                "values": planar.flatten().tolist(),
            }
        ]

    def render(self, fmt, subtype):
        """One second of the malformed-base signal (mono, 22050 Hz) rendered
        to `fmt` bytes in memory."""
        buffer = io.BytesIO()
        soundfile.write(
            buffer,
            self.signal(22050, 22050, 1, self.MALFORMED_BASE_SEED),
            22050,
            format=fmt,
            subtype=subtype,
        )
        return buffer.getvalue()

    @staticmethod
    def header_end(label, data):
        """Byte offset one past the container header: the canonical 44-byte
        RIFF header for WAV, the fLaC magic + STREAMINFO block for FLAC, the
        first Ogg page for OGG."""
        if label == "wav":
            assert data[36:40] == b"data", "non-canonical WAV header"
            return 44
        if label == "flac":
            assert data[:4] == b"fLaC"
            return 8 + 34
        second = data.find(b"OggS", 4)
        assert second > 0
        return second

    def malformed(self):
        directory = os.path.join(self.CORPUS_DIRECTORY, "malformed")
        os.makedirs(directory, exist_ok=True)
        manifest = [
            "# Malformed-input corpus, generated by generate_vectors.py",
            "# (IoVectorGenerator.malformed). Deterministic: base fixture is",
            f"# 1 s mono 22050 Hz of the corpus recipe, seed {self.MALFORMED_BASE_SEED};",
            f"# truncations cut at {self.TRUNCATION_FRACTION:.0%} of byte length,",
            f"# mid-header at offset {self.MID_HEADER_OFFSET}, and one byte past",
            "# the container header. Every file must produce a typed error or",
            "# valid data — never a crash, never zero-fill.",
            "",
        ]

        def emit(filename, data, description):
            with open(os.path.join(directory, filename), "wb") as f:
                f.write(data)
            manifest.append(f"{filename}\t{len(data)} bytes\t{description}")

        emit("empty.wav", b"", "zero bytes")
        rng = np.random.default_rng(self.GARBAGE_SEED)
        emit(
            "garbage.wav",
            rng.integers(0, 256, size=8192, dtype=np.uint8).tobytes(),
            f"8192 bytes of PRNG output, seed {self.GARBAGE_SEED}",
        )

        bases = {
            "wav": self.render("WAV", "PCM_16"),
            "flac": self.render("FLAC", "PCM_16"),
            "ogg": self.render("OGG", "VORBIS"),
        }
        for label, data in bases.items():
            cut = int(len(data) * self.TRUNCATION_FRACTION)
            emit(
                f"trunc.{label}",
                data[:cut],
                f"1 s base cut at byte {cut} of {len(data)}",
            )
            emit(
                f"trunc_hdr20.{label}",
                data[: self.MID_HEADER_OFFSET],
                f"1 s base cut mid-header at byte {self.MID_HEADER_OFFSET}",
            )
            past = self.header_end(label, data) + 1
            emit(
                f"trunc_hdr_past.{label}",
                data[:past],
                f"1 s base cut one byte past the header, at byte {past}",
            )

        wav = bytearray(bases["wav"])
        riff_size, data_size = len(wav) - 8, len(wav) - 44

        def patched_wav(fields):
            out = bytearray(wav)
            for offset, spec, value in fields:
                struct.pack_into(spec, out, offset, value)
            return bytes(out)

        emit(
            "liar_datasize.wav",
            patched_wav(
                [(4, "<I", min(0xFFFFFFFF, riff_size * 10)), (40, "<I", data_size * 10)]
            ),
            f"RIFF and data sizes inflated 10x ({data_size} -> {data_size * 10})",
        )
        emit(
            "liar_int32.wav",
            patched_wav([(4, "<I", 0xFFFFFFF7), (40, "<I", 0x7FFFFFFF)]),
            "data size 0x7FFFFFFF (INT32_MAX), RIFF size to match",
        )
        emit(
            "liar_channels0.wav",
            patched_wav([(22, "<H", 0)]),
            "fmt chunk claims 0 channels",
        )
        emit(
            "liar_channels65535.wav",
            patched_wav([(22, "<H", 0xFFFF)]),
            "fmt chunk claims 65535 channels",
        )
        emit(
            "liar_rate.wav",
            patched_wav([(24, "<I", 0x7FFFFFFF)]),
            "fmt chunk claims a 2147483647 Hz sample rate (INT32_MAX)",
        )

        flac = bytearray(bases["flac"])
        # STREAMINFO: the big-endian 64-bit word at bytes 18..26 packs
        # rate(20) | channels(3) | bits(5) | total_samples(36).
        (word,) = struct.unpack_from(">Q", flac, 18)
        total = word & ((1 << 36) - 1)
        assert total == 22050, total

        def patched_flac(new_total):
            out = bytearray(flac)
            struct.pack_into(">Q", out, 18, (word & ~((1 << 36) - 1)) | new_total)
            return bytes(out)

        emit(
            "liar_frames.flac",
            patched_flac(total * 1000),
            f"STREAMINFO total samples inflated 1000x ({total} -> {total * 1000})",
        )
        emit(
            "liar_int64.flac",
            patched_flac((1 << 36) - 1),
            "STREAMINFO total samples at the 36-bit maximum (68719476735)",
        )

        with open(os.path.join(directory, "MANIFEST"), "w", encoding="utf-8") as f:
            f.write("\n".join(manifest) + "\n")
        print(f"wrote {directory} ({len(manifest) - 8} files)")

    def generate(self):
        write_suite(self.SUITE, "decode", self.parity_cases(), self.versions())
        write_suite(self.SUITE, "clipping", self.clipping_cases(), self.versions())
        self.malformed()


GENERATORS = [
    WindowVectorGenerator,
    StftVectorGenerator,
    IstftVectorGenerator,
    MelVectorGenerator,
    DbConversionsVectorGenerator,
    SpectralFeaturesVectorGenerator,
    EnergyFeaturesVectorGenerator,
    OnsetFeaturesVectorGenerator,
    CqtVectorGenerator,
    ChromaVectorGenerator,
    HpssVectorGenerator,
    IoVectorGenerator,
]

if __name__ == "__main__":
    for generator in GENERATORS:
        generator().generate()
