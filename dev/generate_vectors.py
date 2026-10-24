"""
This file is part of SoundML.

Copyright (C) 2025 Gabriel Santamaria

This script generates the golden test vectors committed beside each test suite (see SUITE_DIRECTORIES).
The reference implementation is librosa 1.0 (which delegates window
computation to scipy.signal); every generated file records the exact
versions it was produced with.

The vectors are committed to the repository: CI never runs this script and
never needs Python. Rerun it only to regenerate the goldens, from the
repository root, in an ephemeral environment with the pinned reference
versions (CPython 3.15, the newest release every dependency, numba included,
ships wheels for):

    uv run --python 3.15 --with 'librosa==1.0.0' --with 'numpy==2.5.4' \
      --with 'scipy==1.18.1' --with 'soundfile==0.14.0' \
      python dev/generate_vectors.py

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
import warnings

import librosa
import numpy as np
import scipy
import scipy.interpolate
import scipy.signal
import scipy.stats
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
    "pvoc": "soundml/test/pvoc/vectors",
    "delta": "soundml/test/delta/vectors",
    "pcen": "soundml/test/pcen/vectors",
    "trim": "soundml/test/trim/vectors",
    "filters": "soundml/test/filter/vectors",
    "defaults": "soundml/test/defaults/vectors",
    "synthesis": "soundml/test/synthesis/vectors",
    "f0": "soundml/test/f0/vectors",
    "tuning": "soundml/test/tuning/vectors",
    "onset_detect": "soundml/test/onset_detect/vectors",
    "rhythm": "soundml/test/rhythm/vectors",
    "io": "soundml-io/test/vectors",
}

GENERATOR_VERSIONS = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "librosa": librosa.__version__,
}

assert librosa.__version__.startswith(
    "1.0."
), f"librosa 1.0 is the pinned reference, found {librosa.__version__}"


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
    rows, and rank-three inputs whose threshold is per channel: librosa 1.0's
    multichannel dB reduces over the last two axes, asserted to differ from
    the whole-tensor clamp), negative entries (powers are floored signed,
    amplitudes are magnitudes), and both element dtypes. Each case stores its
    input in the params (flattened in C order, exactly representable in the
    case dtype) and the librosa output in the values.
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
            # [channels; bins; frames]: the threshold is per channel, and the
            # second channel sits 40 dB under the first so the two differ
            (
                "power_channels_top20",
                "power_to_db",
                1.0,
                1e-10,
                20.0,
                spread(1e-9, 1e3, (2, 2, 16)) * np.array([[[1.0]], [[1e-4]]]),
            ),
            (
                "amplitude_channels_top40",
                "amplitude_to_db",
                1.0,
                1e-5,
                40.0,
                spread(1e-7, 1e2, (2, 2, 16)) * np.array([[[1.0]], [[1e-2]]]),
            ),
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
                if x.ndim > 2:
                    whole = functions[function](
                        x, ref=reference, amin=amin, top_db=top_db, axes=None
                    )
                    assert not np.array_equal(y, whole), name
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
      explicitly at their librosa 1.0 defaults (constant-zero centered
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
      pad_mode is always passed explicitly ("reflect", which the OCaml side
      names as ~pad:`Reflect).

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
      tolerance that substitution costs. One ladder reaches past librosa 1.0's
      decimation guard (asserted to bind), which keeps the rate between two
      octaves whenever the next one would top out above a fifth of it.
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

    # The decimation guard: from 98 bins up (C1, 12 per octave, 22050 Hz, no
    # early decimation) the second octave tops out above sr / 5, so librosa
    # 1.0 analyses it at the full rate instead of halving first.
    GUARD = [
        ("w12_cqt100_guard", "harmonic", 8192, SR, 100, 12, C1, 512, 0,
         True, ["float64"]),
    ]

    @classmethod
    def signal(cls, kind, n, sr):
        return cls.lcg(n) if kind == "lcg" else cls.harmonic(n, sr)

    def assert_guard(self):
        """Refuse a GUARD row the guard does not bind on: an even hop, no
        early decimation, and the top bin of the second octave above a fifth
        of the rate."""
        count = getattr(librosa.core.constantq, "__early_downsample_count")
        for (key, _, _, sr, n_bins, bpo, fmin, hop, gamma, _,
             _) in self.GUARD:
            freqs = self.ladder(n_bins, bpo, fmin)
            _, cutoff = librosa.filters.wavelet_lengths(
                freqs=freqs,
                sr=sr,
                window="hann",
                filter_scale=1,
                gamma=gamma,
                alpha=self.alphas(freqs, bpo),
            )
            early = count(sr / 2.0, cutoff, hop, int(np.ceil(n_bins / bpo)))
            assert early == 0 and hop % 2 == 0, key
            assert freqs[-bpo - 1] > sr / 5.0, key

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
        self.assert_guard()
        write_suite(
            self.SUITE,
            "cqt_wide",
            self.transform_cases(self.WIDE)
            + self.transform_cases(self.SHORT)
            + self.transform_cases(self.GUARD),
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


class PvocVectorGenerator:
    """Golden vectors for Soundml.Effects.

    Four kinds of file, all over deterministic 31-bit LCG signals reproduced
    bit-for-bit in OCaml through the same integer arithmetic:

    - stretch: librosa.effects.time_stretch over the geometry axis (fft size,
      hop, including non-divisible and odd combinations), the rate axis
      (compressing, expanding, unity and a non-round rate), two signal
      lengths, both float dtypes, short signals, and one batched pair. The
      analysis is librosa's default zero padding, which is this library's
      `Constant 0.` default as well. librosa 1.0's
      vocoder interpolates magnitudes linearly between analysis frames and
      extrapolates them from the last two past the final one, and its phase
      accumulates the raw inter-frame difference, which is zero from the
      final frame on. A single analysis frame leaves that interpolation
      undefined (librosa returns NaN), so no case analyses fewer than two
      frames, and every emitted value is asserted finite.

    - pitchstretch: the stretch stage of the pitch cases alone, at the exact
      float64 quotient the rational ratio produces, so the composition's rate
      is pinned independently of the resampler beneath it.

    - pitch: the whole librosa.effects.pitch_shift composition -- that same
      stretch, then librosa.resample at soxr_hq between the two terms of the
      ratio, then a crop to the input length. SoundML resamples with its own
      polyphase bank, so these cases carry the resampler substitution and are
      compared at a tolerance the suite justifies against soxr's own tier
      spread.

    - locked: the identity phase-locked variant, from the replica below --
      this script is the oracle for those cases, and the file records it as
      such. It is librosa 1.0's vocoder with the phases rewritten: peaks are
      picked on the analysis frame the output frame floors to, regions split
      halfway between consecutive peaks, and the locked phases are what the
      next frame accumulates from.

    float32 cases quantize the input to float32 and compute the reference in
    float64, so the vectors isolate input rounding from the accumulator's
    precision -- librosa propagates float32 phases for float32 audio, whose
    own spread against its float64 path is four to six orders above what
    separates the two implementations.
    """

    SUITE = "pvoc"

    # (case key, fft_size, hop)
    GEOMETRY = [
        ("fft64_hop16", 64, 16),
        ("fft32_hop7", 32, 7),
        ("fft256_hop64", 256, 64),
        ("fft2048_hop512", 2048, 512),
        ("fft2048_hop500", 2048, 500),
        ("fft31_hop5", 31, 5),
    ]

    RATES = [0.5, 0.75, 1.0, 1.25, 1.37, 2.0]

    RATE_GEOMETRY = [("fft256_hop64", 256, 64), ("fft2048_hop512", 2048, 512)]

    LENGTHS = [128, 1000]

    SHORT = 127

    # (case key, numerator, denominator) -- the frequency ratio L / M, which
    # stretches by M / L and then converts from L to M.
    PITCH_RATIOS = [
        ("octave_up", 2, 1),
        ("octave_down", 1, 2),
        ("major_third_up", 3524, 2797),
        ("fifth_up", 2655, 1772),
        ("tritone_down", 2378, 3363),
    ]

    # (case key, fft_size, hop, rate, n)
    LOCKED_CELLS = [
        ("fft256_hop64_r1p37", 256, 64, 1.37, 1000),
        ("fft2048_hop512_r0p5", 2048, 512, 0.5, 1000),
        ("fft64_hop16_r2", 64, 16, 2.0, 127),
    ]

    @staticmethod
    def lcg(n, seed=20250803):
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
    def signal(cls, n, channels=1):
        stream = cls.lcg(n * channels)
        return stream if channels == 1 else stream.reshape(channels, n)

    @staticmethod
    def quantized(y, dtype):
        return y.astype(np.float32).astype(np.float64) if dtype == "float32" else y

    @staticmethod
    def tag(rate):
        return ("r" + f"{rate:g}").replace(".", "p")

    @staticmethod
    def frames(n, hop):
        """The centered analysis frame count of an n-sample signal."""
        return 1 + n // hop

    @staticmethod
    def stretch(y, fft_size, hop, rate):
        expected = librosa.effects.time_stretch(y, rate=rate, n_fft=fft_size,
                                                hop_length=hop)
        assert np.isfinite(expected).all(), (fft_size, hop, rate, y.shape)
        return expected

    @classmethod
    def case(cls, name, params, expected):
        return {"name": name, "params": params,
                "shape": list(expected.shape), "values": expected.ravel().tolist()}

    def stretch_cases(self):
        """Every cell whose analysis has at least two frames: the 2048-point
        geometries over the 127- and 128-sample signals analyse one, where
        librosa's interpolation has nothing to interpolate."""
        cases = []
        for key, fft_size, hop in self.GEOMETRY:
            if self.frames(self.SHORT, hop) < 2:
                continue
            y = self.signal(self.SHORT)
            expected = self.stretch(y, fft_size, hop, 1.37)
            cases.append(self.case(
                f"geometry_{key}_r1p37_float64",
                {"fft_size": fft_size, "hop": hop, "rate": 1.37,
                 "n": self.SHORT, "channels": 1, "dtype": "float64"},
                expected))
        for rate in self.RATES:
            for key, fft_size, hop in self.RATE_GEOMETRY:
                for n in self.LENGTHS:
                    if self.frames(n, hop) < 2:
                        continue
                    for dtype in ("float32", "float64"):
                        y = self.quantized(self.signal(n), dtype)
                        expected = self.stretch(y, fft_size, hop, rate)
                        cases.append(self.case(
                            f"rate_{key}_{self.tag(rate)}_n{n}_{dtype}",
                            {"fft_size": fft_size, "hop": hop, "rate": rate,
                             "n": n, "channels": 1, "dtype": dtype},
                            expected))
        for rate in self.RATES:
            dtypes = ("float32", "float64") if rate in (0.5, 2.0) else ("float64",)
            for dtype in dtypes:
                y = self.quantized(self.signal(self.SHORT), dtype)
                expected = self.stretch(y, 64, 16, rate)
                cases.append(self.case(
                    f"short_fft64_hop16_{self.tag(rate)}_{dtype}",
                    {"fft_size": 64, "hop": 16, "rate": rate,
                     "n": self.SHORT, "channels": 1, "dtype": dtype},
                    expected))
        y = self.signal(1000, channels=2)
        cases.append(self.case(
            "batched_fft256_hop64_r1p37_float64",
            {"fft_size": 256, "hop": 64, "rate": 1.37, "n": 1000,
             "channels": 2, "dtype": "float64"},
            self.stretch(y, 256, 64, 1.37)))
        return cases

    def pitch_cells(self):
        cells = []
        for key, num, den in self.PITCH_RATIOS:
            for dtype in ("float32", "float64"):
                cells.append((key, num, den, "fft2048_hop512", 2048, 512, 1000, dtype))
        cells.append(("major_third_up", 3524, 2797, "fft256_hop64", 256, 64, 1000,
                      "float64"))
        return cells

    def pitch_stretch_cases(self):
        cases = []
        for key, num, den, gkey, fft_size, hop, n, dtype in self.pitch_cells():
            y = self.quantized(self.signal(n), dtype)
            expected = self.stretch(y, fft_size, hop, den / num)
            cases.append(self.case(
                f"pitchstretch_{key}_{gkey}_n{n}_{dtype}",
                {"fft_size": fft_size, "hop": hop, "num": num, "den": den,
                 "rate": den / num, "n": n, "channels": 1, "dtype": dtype},
                expected))
        return cases

    def pitch_cases(self):
        cases = []
        for key, num, den, gkey, fft_size, hop, n, dtype in self.pitch_cells():
            y = self.quantized(self.signal(n), dtype)
            stretched = self.stretch(y, fft_size, hop, den / num)
            shifted = librosa.resample(stretched, orig_sr=num, target_sr=den,
                                       res_type="soxr_hq")
            expected = librosa.util.fix_length(shifted, size=n)
            cases.append(self.case(
                f"pitch_{key}_{gkey}_n{n}_{dtype}",
                {"fft_size": fft_size, "hop": hop, "num": num, "den": den,
                 "n": n, "channels": 1, "dtype": dtype},
                expected))
        return cases

    @staticmethod
    def peaks_of(magnitudes):
        """The bins whose magnitude strictly exceeds every one of the four
        neighbours they have."""
        n = len(magnitudes)
        padded = np.pad(magnitudes, 2, mode="constant")
        core = padded[2:2 + n]
        return np.flatnonzero(
            (core > padded[0:n]) & (core > padded[1:1 + n])
            & (core > padded[3:3 + n]) & (core > padded[4:4 + n])
        )

    @classmethod
    def owners(cls, peaks, bins):
        """The peak each bin belongs to: regions split halfway between
        consecutive peaks, a bin equidistant from both going to the upper
        one."""
        if len(peaks) == 0:
            return None
        owner = np.zeros(bins, dtype=np.int64)
        bounds = ([0] + [(peaks[i] + peaks[i + 1] + 1) // 2
                         for i in range(len(peaks) - 1)] + [bins])
        for i, peak in enumerate(peaks):
            owner[bounds[i]:bounds[i + 1]] = peaks[i]
        return owner

    @classmethod
    def vocode_locked(cls, spectrum, rate):
        """The identity phase-locked phase vocoder: librosa 1.0's vocoder
        with each frame's phases rewritten, before they are emitted and
        before the next frame accumulates from them, so that the bins around
        a spectral peak carry the peak's accumulated phase offset by the
        phase difference they have in the analysis frame. The magnitudes are
        librosa's own interpolation, extrapolating past the final frame, and
        the phase advances by the raw difference to the next analysis frame,
        clamped to the final one."""
        frames = spectrum.shape[-1]
        assert frames >= 2, frames
        steps = np.arange(0, frames, rate, dtype=np.float64)
        bins = spectrum.shape[0]
        out = np.zeros((bins, len(steps)), dtype=np.complex128)
        magnitudes = scipy.interpolate.interp1d(
            np.arange(frames),
            np.abs(spectrum),
            kind="linear",
            axis=-1,
            fill_value="extrapolate",
            assume_sorted=True,
        )(steps)
        angles = np.angle(spectrum)
        accumulator = angles[:, 0].copy()
        for t, step in enumerate(steps):
            i = int(step)
            owner = cls.owners(cls.peaks_of(np.abs(spectrum[:, i])), bins)
            if owner is not None:
                reference = angles[:, i]
                locked = accumulator[owner] + (reference - reference[owner])
                accumulator = locked
            out[:, t] = librosa.util.phasor(accumulator, mag=magnitudes[:, t])
            following = angles[:, min(i + 1, frames - 1)]
            accumulator = accumulator + (following - angles[:, i])
        return out

    def locked_cases(self):
        cases = []
        for key, fft_size, hop, rate, n in self.LOCKED_CELLS:
            y = self.signal(n)
            spectrum = librosa.stft(y, n_fft=fft_size, hop_length=hop,
                                    dtype=np.complex128)
            stretched = self.vocode_locked(spectrum, rate)
            expected = librosa.istft(stretched, hop_length=hop, n_fft=fft_size,
                                     dtype=np.float64,
                                     length=int(round(n / rate)))
            assert np.isfinite(expected).all(), key
            cases.append(self.case(
                f"locked_{key}_n{n}_float64",
                {"fft_size": fft_size, "hop": hop, "rate": rate, "n": n,
                 "channels": 1, "dtype": "float64"},
                expected))
        return cases

    def generate(self):
        write_suite(self.SUITE, "stretch", self.stretch_cases())
        write_suite(self.SUITE, "pitchstretch", self.pitch_stretch_cases())
        write_suite(self.SUITE, "pitch", self.pitch_cases())
        write_suite(self.SUITE, "locked", self.locked_cases(),
                    dict(GENERATOR_VERSIONS, oracle="dev/generate_vectors.py"))


class DeltaVectorGenerator:
    """Golden vectors for Soundml.delta: librosa.feature.delta over the LCG
    stream under this suite's seed, reshaped in C order to [frames],
    [features; frames] and [channels; features; frames].

    The grid crosses widths 3, 5, 9 and 11 with orders 1 to 3 (and order 4
    at width 9), both borders librosa's delta passes to savgol_filter that
    the library implements ("interp", the default, and "nearest"), and frame
    counts at the edges of each border: exactly width frames (the fewest
    "interp" accepts), one more, and longer sequences; "nearest" also takes
    one and two frames. float32 cases quantize the input to float32 and
    compute the reference in float64, like the energy suite.
    """

    SUITE = "delta"

    SEED = 20261023

    # (width, order)
    FILTERS = [(3, 1), (5, 1), (5, 2), (5, 3), (9, 1), (9, 2), (9, 4), (11, 2)]

    def signal(self, n):
        return StftVectorGenerator.lcg_signal(n, seed=self.SEED)

    @staticmethod
    def quantize(y, dtype):
        return y if dtype == "float64" else y.astype(np.float32).astype(np.float64)

    def frame_counts(self, width, mode):
        counts = [width, width + 1, 40]
        if mode == "nearest":
            counts = [1, 2, width - 1] + counts
        return sorted(set(counts))

    def cases(self):
        cases = []
        for width, order in self.FILTERS:
            for mode in ("interp", "nearest"):
                for frames in self.frame_counts(width, mode):
                    for lead in ([], [3], [2, 3]):
                        shape = lead + [frames]
                        base = self.signal(int(np.prod(shape))).reshape(shape)
                        dtypes = ["float64", "float32"] if lead == [3] else ["float64"]
                        for dtype in dtypes:
                            x = self.quantize(base, dtype)
                            expected = librosa.feature.delta(
                                x, width=width, order=order, axis=-1, mode=mode)
                            assert expected.dtype == np.float64
                            assert np.isfinite(expected).all()
                            name = (f"w{width}_o{order}_{mode}_"
                                    f"{'x'.join(map(str, shape))}_{dtype}")
                            cases.append({
                                "name": name,
                                "params": {"width": width, "order": order,
                                           "mode": mode, "input_shape": shape,
                                           "dtype": dtype},
                                "shape": list(expected.shape),
                                "values": expected.ravel().tolist()})
        return cases

    def generate(self):
        write_suite(self.SUITE, "delta", self.cases())


class PcenVectorGenerator:
    """Golden vectors for Soundml.pcen: librosa.pcen over synthetic magnitude
    spectrograms, |LCG values| under this suite's seed scaled by 2 ** 15 and
    reshaped in C order to [frames], [bands; frames] and
    [channels; bands; frames].

    Cases cover librosa's defaults (22.05 kHz, hop 512), a 16 kHz hop-160
    speech geometry with a short time constant, explicit b (including 1, no
    smoothing), every compression branch (power 0, bias 0, the general
    case), a non-default gain and eps, and max_size 3 and 4 (odd and even
    windows of the band maximum, reflected at the edges; the batched cases
    pass max_axis=-2, the band axis, which librosa needs above rank 2).
    float32 cases quantize the input to float32 and compute the reference in
    float64.
    """

    SUITE = "pcen"

    SEED = 20261024

    # (case key, input shape, pcen keyword arguments, dtypes)
    CASES = [
        ("defaults", [16, 40], {}, ["float64", "float32"]),
        ("defaults_1d", [40], {}, ["float64"]),
        ("defaults_batched", [2, 16, 40], {}, ["float64"]),
        ("speech16k", [16, 40], {"sr": 16000, "hop_length": 160,
                                 "time_constant": 0.06}, ["float64"]),
        ("b0.1", [16, 40], {"b": 0.1}, ["float64"]),
        ("b1", [16, 40], {"b": 1.0}, ["float64"]),
        ("power0", [16, 40], {"power": 0.0}, ["float64"]),
        ("bias0", [16, 40], {"bias": 0.0}, ["float64"]),
        ("power1_gain0.5", [16, 40], {"power": 1.0, "gain": 0.5}, ["float64"]),
        ("eps1e-3", [16, 40], {"eps": 1e-3}, ["float64"]),
        ("max3", [16, 40], {"max_size": 3}, ["float64", "float32"]),
        ("max4", [16, 40], {"max_size": 4}, ["float64"]),
        ("max4_batched", [2, 16, 40], {"max_size": 4, "max_axis": -2},
         ["float64"]),
        ("max9_wider_than_half", [6, 40], {"max_size": 9}, ["float64"]),
    ]

    def signal(self, n):
        return StftVectorGenerator.lcg_signal(n, seed=self.SEED)

    @staticmethod
    def quantize(y, dtype):
        return y if dtype == "float64" else y.astype(np.float32).astype(np.float64)

    def cases(self):
        cases = []
        for key, shape, kwargs, dtypes in self.CASES:
            base = np.abs(self.signal(int(np.prod(shape)))).reshape(shape) * 2.0**15
            for dtype in dtypes:
                S = self.quantize(base, dtype)
                expected = librosa.pcen(S, **kwargs)
                assert expected.dtype == np.float64
                assert np.isfinite(expected).all(), key
                params = {"input_shape": shape, "dtype": dtype}
                params.update({{"sr": "sample_rate", "hop_length": "hop"}.get(k, k): v
                               for k, v in kwargs.items() if k != "max_axis"})
                cases.append({"name": f"{key}_{dtype}", "params": params,
                              "shape": list(expected.shape),
                              "values": expected.ravel().tolist()})
        return cases

    def generate(self):
        write_suite(self.SUITE, "pcen", self.cases())


class TrimVectorGenerator:
    """Golden vectors for Soundml.Effects.trim and split:
    librosa.effects.trim / split over piecewise signals, each a run of
    segments [length, scale] cut from one LCG stream under this suite's seed
    (scale 0 is silence), mono or [channels; samples] in C order.

    Cases cover the defaults, a smaller geometry, lower thresholds, fixed
    references, a stereo signal whose channels are loud at different times
    (librosa's aggregate=np.max), batches of rank three and four whose clips
    each take their own maximum as the reference, a signal of zeros (kept whole under
    ref=np.max, the floors cancelling), one shorter than a frame and one
    entirely below a fixed reference. The fixed-reference cases also pin
    Effects.split_stage, which takes only those: among them an odd frame
    length, a stereo signal at both dtypes, zeros made non-silent by a small
    reference (the 1e-5 floor sits 20 dB above its threshold), and a hop
    over half the frame whose last frame alone is loud, the empty interval
    at the end of the signal. Every frame's decibels are asserted to lie at
    least 0.1 dB from the threshold, so neither side's rounding can move a
    decision; float32 cases quantize the input first.
    """

    SUITE = "trim"

    SEED = 20261025

    QUIET_SPEECH = [[3000, 0.0], [5000, 0.5], [4000, 1e-4], [3000, 0.3],
                    [2500, 0.0]]

    # (case key, segments, channels, kwargs, dtypes)
    CASES = [
        ("defaults", QUIET_SPEECH, 1, {}, ["float64", "float32"]),
        ("fl512_hop128", QUIET_SPEECH, 1,
         {"frame_length": 512, "hop_length": 128}, ["float64"]),
        ("top_db20", [[2000, 0.01], [3000, 0.6], [3000, 0.02], [2000, 0.5]], 1,
         {"top_db": 20}, ["float64"]),
        ("ref1", QUIET_SPEECH, 1, {"ref": 1.0}, ["float64"]),
        ("ref0.1_top_db30", QUIET_SPEECH, 1, {"ref": 0.1, "top_db": 30},
         ["float64"]),
        ("stereo", [[4000, 0.0], [3000, 0.4], [5000, 0.0], [2000, 0.2],
                    [4000, 0.0], [3000, 0.5]], 2, {}, ["float64"]),
        ("zeros", [[4000, 0.0]], 1, {}, ["float64"]),
        ("short", [[300, 0.5]], 1, {}, ["float64"]),
        ("all_quiet_ref1", [[6000, 1e-5]], 1, {"ref": 1.0}, ["float64"]),
        ("fl511_hop128_ref1", QUIET_SPEECH, 1,
         {"frame_length": 511, "hop_length": 128, "ref": 1.0},
         ["float64", "float32"]),
        ("tail_hop_over_half", [[3996, 0.0], [100, 0.5]], 1,
         {"frame_length": 256, "hop_length": 512, "ref": 1.0}, ["float64"]),
        ("zeros_ref1e-3", [[4000, 0.0]], 1, {"ref": 1e-3}, ["float64"]),
        ("stereo_ref0.5_top_db40", [[4000, 0.0], [3000, 0.4], [5000, 0.0],
                                    [2000, 0.2], [4000, 0.0], [3000, 0.5]], 2,
         {"ref": 0.5, "top_db": 40}, ["float64", "float32"]),
    ]

    # Batches of rank three or more, each row its own segments: librosa 1.0's
    # amplitude_to_db takes ref=np.max over the last two axes, so every
    # [channels; samples] clip is measured against its own maximum, and a clip
    # quiet next to another (-74 dB here) still has non-silent frames. The
    # second case puts each row in a clip of its own.
    BATCH_ROWS = [
        [[3000, 0.0], [5000, 0.5], [10000, 0.0]],
        [[3000, 0.0], [5000, 0.3], [10000, 0.0]],
        [[12000, 0.0], [4000, 1e-4], [2000, 0.0]],
        [[12000, 0.0], [4000, 5e-5], [2000, 0.0]],
    ]

    # (case key, rows, leading shape, kwargs, dtypes)
    BATCH_CASES = [
        ("batch2x2", BATCH_ROWS, [2, 2], {}, ["float64", "float32"]),
        ("batch2x2x1", BATCH_ROWS, [2, 2, 1], {}, ["float64"]),
    ]

    def batch_signal(self, rows, lead):
        n = sum(length for length, _ in rows[0])
        stream = StftVectorGenerator.lcg_signal(len(rows) * n, seed=self.SEED)
        scales = np.concatenate(
            [np.concatenate([np.full(length, scale) for length, scale in row])
             for row in rows])
        return (stream * scales).reshape(*lead, n)

    def signal(self, segments, channels):
        n = sum(length for length, _ in segments)
        stream = StftVectorGenerator.lcg_signal(channels * n, seed=self.SEED)
        scales = np.concatenate([np.full(length, scale) for length, scale in segments])
        y = stream * np.tile(scales, channels)
        return y.reshape(channels, n) if channels > 1 else y

    @staticmethod
    def quantize(y, dtype):
        return y if dtype == "float64" else y.astype(np.float32).astype(np.float64)

    def margin(self, y, kwargs):
        frame_length = kwargs.get("frame_length", 2048)
        hop_length = kwargs.get("hop_length", 512)
        top_db = kwargs.get("top_db", 60)
        mse = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
        db = librosa.amplitude_to_db(mse[..., 0, :], ref=kwargs.get("ref", np.max),
                                     top_db=None)
        if db.ndim > 1:
            db = db.max(axis=tuple(range(db.ndim - 1)))
        return float(np.min(np.abs(db + top_db)))

    def cases(self):
        cases = []
        for key, segments, channels, kwargs, dtypes in self.CASES:
            for dtype in dtypes:
                y = self.quantize(self.signal(segments, channels), dtype)
                assert self.margin(y, kwargs) > 0.1, (key, self.margin(y, kwargs))
                _, index = librosa.effects.trim(y, **kwargs)
                intervals = librosa.effects.split(y, **kwargs)
                params = {"segments": segments, "channels": channels, "dtype": dtype}
                params.update({{"hop_length": "hop", "ref": "reference"}.get(k, k): v
                               for k, v in kwargs.items()})
                cases.append({"name": f"trim_{key}_{dtype}", "params": params,
                              "shape": [2],
                              "values": [float(v) for v in index]})
                cases.append({"name": f"split_{key}_{dtype}", "params": params,
                              "shape": list(intervals.shape),
                              "values": [float(v) for v in intervals.ravel()]})
        for key, rows, lead, kwargs, dtypes in self.BATCH_CASES:
            for dtype in dtypes:
                y = self.quantize(self.batch_signal(rows, lead), dtype)
                assert self.margin(y, kwargs) > 0.1, (key, self.margin(y, kwargs))
                _, index = librosa.effects.trim(y, **kwargs)
                intervals = librosa.effects.split(y, **kwargs)
                # One reference for the whole batch would decide otherwise.
                whole = float(librosa.feature.rms(y=y).max())
                assert not np.array_equal(
                    librosa.effects.split(y, ref=whole), intervals), key
                params = {"row_segments": rows, "lead": lead, "dtype": dtype}
                params.update(kwargs)
                cases.append({"name": f"trim_{key}_{dtype}", "params": params,
                              "shape": [2],
                              "values": [float(v) for v in index]})
                cases.append({"name": f"split_{key}_{dtype}", "params": params,
                              "shape": list(intervals.shape),
                              "values": [float(v) for v in intervals.ravel()]})
        return cases

    def generate(self):
        write_suite(self.SUITE, "trim", self.cases())


class FilterVectorGenerator:
    """Golden vectors for Soundml.Filter: scipy.signal's butter (transfer
    function and second-order sections), lfilter_zi, sosfilt_zi, lfilter,
    sosfilt, filtfilt and sosfiltfilt, and librosa.effects.preemphasis and
    deemphasis, over the LCG stream under this suite's seed reshaped in C
    order to [samples], [channels; samples] and [2; 3; samples].

    Every filtering case records the coefficients it filters with (and its
    initial state), so the filtering vectors pin the kernels independently of
    the design. States are recorded in the library's layout, the time axis
    last: a cascade's [..., n_sections, 2] where scipy has
    (n_sections, ..., 2), the generator moving the axis both ways. A case
    with "output": "zf" holds the final state of the case of the same name
    with "output": "y"; scipy returns a state only when given one, so the
    cases started from rest pass an explicit zero state.

    The band designs keep their edges away from the pole-pairing tie of
    zpk2sos (edges symmetric about a quarter of the sample rate give mirror
    poles equally close to the unit circle, whose pairing then turns on the
    last bit of the platform's complex square root); the generator asserts
    the distance. The transfer-function filtering cases use designs of order
    four or less, whose arithmetic the ill-conditioning of high-order
    polynomials does not amplify. float32 cases quantize the input to float32
    and compute the reference in float64. librosa's de-emphasis without a
    state returns the state of its uncorrected recursion, which the generator
    asserts; the vector records the state that continues the corrected
    output, coef times its last sample.
    """

    SUITE = "filters"

    SEED = 20261109

    # (key, band type, cutoffs in Hz, sample rate)
    BANDS = [
        ("low1000", "lowpass", [1000.0], 16000),
        ("low30", "lowpass", [30.0], 44100),
        ("low0.3", "lowpass", [0.3], 2),
        ("high300", "highpass", [300.0], 16000),
        ("high5000", "highpass", [5000.0], 44100),
        ("band300_3400", "bandpass", [300.0, 3400.0], 16000),
        ("band1000_1200", "bandpass", [1000.0, 1200.0], 16000),
        ("band50_15000", "bandpass", [50.0, 15000.0], 44100),
        ("stop2000_2500", "bandstop", [2000.0, 2500.0], 16000),
        ("stop100_6000", "bandstop", [100.0, 6000.0], 22050),
        ("stop49_51", "bandstop", [49.0, 51.0], 1000),
    ]

    BA_ORDERS = [0, 1, 2, 3, 4, 5, 8]

    SOS_ORDERS = [0, 1, 2, 3, 4, 5, 8, 10, 12]

    def signal(self, n):
        return StftVectorGenerator.lcg_signal(n, seed=self.SEED)

    def state(self, shape, seed_offset):
        """A deterministic state of `shape`, from a second LCG stream."""
        n = int(np.prod(shape))
        return StftVectorGenerator.lcg_signal(n, seed=self.SEED + seed_offset).reshape(shape)

    @staticmethod
    def quantize(y, dtype):
        return y if dtype == "float64" else y.astype(np.float32).astype(np.float64)

    def input(self, shape, dtype):
        return self.quantize(self.signal(int(np.prod(shape))).reshape(shape), dtype)

    @staticmethod
    def butter(order, kind, cutoffs, sr, output):
        wn = cutoffs[0] if len(cutoffs) == 1 else cutoffs
        return scipy.signal.butter(order, wn, btype=kind, fs=sr, output=output)

    def check_pairing(self, order, kind, cutoffs, sr):
        """Keeps the band designs away from zpk2sos's pairing ties: the band
        is not symmetric about sr / 4, and the poles' distances to the unit
        circle, which decide the order of the sections, stay well apart."""
        if len(cutoffs) == 2:
            assert abs(cutoffs[0] + cutoffs[1] - sr / 2) > 1e-6 * sr, (kind, cutoffs, sr)
        _, p, _ = self.butter(order, kind, cutoffs, sr, "zpk")
        upper = p[p.imag >= 0]
        if len(upper) > 1:
            d = np.sort(np.abs(1 - np.abs(upper)))
            assert np.min(np.diff(d)) > 1e-9, (order, kind, cutoffs, np.min(np.diff(d)))

    def butter_cases(self):
        cases = []
        for output, orders in (("ba", self.BA_ORDERS), ("sos", self.SOS_ORDERS)):
            for key, kind, cutoffs, sr in self.BANDS:
                for order in orders:
                    if output == "sos":
                        self.check_pairing(order, kind, cutoffs, sr)
                        values = self.butter(order, kind, cutoffs, sr, "sos")
                    else:
                        b, a = self.butter(order, kind, cutoffs, sr, "ba")
                        assert b.shape == a.shape
                        values = np.stack([b, a])
                    assert np.isfinite(values).all()
                    cases.append({
                        "name": f"{output}_{key}_order{order}",
                        "params": {"order": order, "kind": kind, "cutoffs": cutoffs,
                                   "sample_rate": sr, "output": output},
                        "shape": list(values.shape),
                        "values": values.ravel().tolist()})
        return cases

    # Transfer functions beyond butter's: b longer than a, a longer than b, a
    # leading coefficient to normalise by, a gain, FIR filters.
    HAND_BA = [
        ("b_longer", [0.5, 0.25, -0.1], [1.0, -0.3]),
        ("a_longer", [0.2], [1.0, -1.2, 0.5, -0.1]),
        ("a0_2.5", [0.3, -0.2, 0.1], [2.5, -0.5, 0.25]),
        ("gain", [0.3], [2.0]),
        ("fir3", [0.25, 0.5, 0.25], [1.0]),
        ("preemph", [1.0, -0.97], [1.0]),
    ]

    def fir(self, taps):
        return (0.1 * self.state([taps], 7)).tolist(), [1.0]

    def zi_cases(self):
        cases = []
        designs = []
        for order in range(1, 9):
            for key, kind, cutoffs, sr in (self.BANDS[0], self.BANDS[3], self.BANDS[5], self.BANDS[8]):
                b, a = self.butter(order, kind, cutoffs, sr, "ba")
                designs.append((f"{key}_order{order}", b.tolist(), a.tolist()))
        b, a = self.butter(3, "lowpass", [1000.0], 16000, "ba")
        designs.append(("low1000_order3_a_scaled", b.tolist(), (2.5 * a).tolist()))
        designs += self.HAND_BA
        designs.append(("fir140", *self.fir(140)))
        for key, b, a in designs:
            zi = scipy.signal.lfilter_zi(b, a)
            cases.append({"name": f"lfilter_zi_{key}",
                          "params": {"function": "lfilter_zi", "b": b, "a": a},
                          "shape": list(zi.shape), "values": zi.tolist()})
        for order in range(1, 13):
            for key, kind, cutoffs, sr in (self.BANDS[0], self.BANDS[3], self.BANDS[5], self.BANDS[8]):
                self.check_pairing(order, kind, cutoffs, sr)
                sos = self.butter(order, kind, cutoffs, sr, "sos")
                zi = scipy.signal.sosfilt_zi(sos)
                cases.append({"name": f"sosfilt_zi_{key}_order{order}",
                              "params": {"function": "sosfilt_zi",
                                         "sos": sos.ravel().tolist(),
                                         "n_sections": int(sos.shape[0])},
                              "shape": list(zi.shape), "values": zi.ravel().tolist()})
        return cases

    def lfilter_designs(self):
        designs = []
        for order, key, kind, cutoffs, sr in ((1, "low1000", "lowpass", [1000.0], 16000),
                                              (2, "low1000", "lowpass", [1000.0], 16000),
                                              (4, "low1000", "lowpass", [1000.0], 16000),
                                              (3, "high300", "highpass", [300.0], 16000),
                                              (2, "band300_3400", "bandpass", [300.0, 3400.0], 16000),
                                              (3, "stop2000_2500", "bandstop", [2000.0, 2500.0], 16000)):
            b, a = self.butter(order, kind, cutoffs, sr, "ba")
            designs.append((f"{key}_order{order}", b.tolist(), a.tolist()))
        designs += self.HAND_BA
        designs.append(("fir16", *self.fir(16)))
        return designs

    def sos_designs(self):
        designs = []
        for order, key, kind, cutoffs, sr in ((6, "high50", "highpass", [50.0], 16000),
                                              (4, "band1000_1200", "bandpass", [1000.0, 1200.0], 16000),
                                              (8, "low2000", "lowpass", [2000.0], 16000),
                                              (5, "stop100_6500", "bandstop", [100.0, 6500.0], 16000),
                                              (10, "low100", "lowpass", [100.0], 16000),
                                              (1, "low1000", "lowpass", [1000.0], 16000)):
            self.check_pairing(order, kind, cutoffs, sr)
            sos = self.butter(order, kind, cutoffs, sr, "sos")
            designs.append((f"{key}_order{order}", sos))
        return designs

    # (key, input shape, initial state, dtypes): the state is "none" (rest),
    # "full" (an LCG state of the full shape), "broadcast" (one state with a
    # leading axis of size 1, for every row) or "steady" (the steady state
    # scaled by each row's first sample)
    LAYOUTS = [
        ("1d_none", [200], "none", ["float64", "float32"]),
        ("1d_full", [200], "full", ["float64"]),
        ("1d_steady", [200], "steady", ["float64"]),
        ("2ch_broadcast", [2, 120], "broadcast", ["float64", "float32"]),
        ("2ch_steady", [2, 120], "steady", ["float64"]),
        ("batch_full", [2, 3, 40], "full", ["float64"]),
    ]

    # the designs whose float32 layouts are emitted
    FLOAT32_DESIGNS = ["low1000_order2", "band300_3400_order2", "a0_2.5", "fir3",
                       "high50_order6", "stop100_6500_order5"]

    def lfilter_cases(self):
        cases = []
        for key, b, a in self.lfilter_designs():
            order = max(len(a), len(b)) - 1
            for layout, shape, init, dtypes in self.LAYOUTS:
                for dtype in dtypes:
                    if dtype == "float32" and key not in self.FLOAT32_DESIGNS:
                        continue
                    x = self.input(shape, dtype)
                    lead = shape[:-1]
                    if init == "none":
                        zi = None
                        zi_ref = np.zeros(lead + [order])
                    elif init == "full":
                        zi = self.state(lead + [order], 1)
                        zi_ref = zi
                    elif init == "broadcast":
                        zi = self.state([1] * len(lead) + [order], 2)
                        zi_ref = zi
                    else:
                        zi = scipy.signal.lfilter_zi(b, a) * x[..., :1]
                        zi_ref = zi
                    y, zf = scipy.signal.lfilter(b, a, x, zi=zi_ref)
                    assert y.dtype == np.float64 and np.isfinite(y).all()
                    params = {"b": b, "a": a, "input_shape": shape, "dtype": dtype}
                    if zi is not None:
                        params["zi"] = np.asarray(zi).ravel().tolist()
                        params["zi_shape"] = list(np.shape(zi))
                    name = f"{key}_{layout}_{dtype}"
                    cases.append({"name": name, "params": {**params, "output": "y"},
                                  "shape": list(y.shape), "values": y.ravel().tolist()})
                    zf = np.broadcast_to(zf, lead + [order])
                    cases.append({"name": name, "params": {**params, "output": "zf"},
                                  "shape": list(zf.shape), "values": zf.ravel().tolist()})
        return cases

    def sosfilt_cases(self):
        cases = []
        for key, sos in self.sos_designs():
            ns = sos.shape[0]
            for layout, shape, init, dtypes in self.LAYOUTS:
                for dtype in dtypes:
                    if dtype == "float32" and key not in self.FLOAT32_DESIGNS:
                        continue
                    x = self.input(shape, dtype)
                    lead = shape[:-1]
                    if init == "none":
                        zi = None
                        zi_ref = np.zeros(lead + [ns, 2])
                    elif init == "full":
                        zi = self.state(lead + [ns, 2], 3)
                        zi_ref = zi
                    elif init == "broadcast":
                        zi = self.state([1] * len(lead) + [ns, 2], 4)
                        zi_ref = np.broadcast_to(zi, lead + [ns, 2])
                    else:
                        zi = scipy.signal.sosfilt_zi(sos) * x[..., :1, None]
                        zi_ref = zi
                    y, zf = scipy.signal.sosfilt(sos, x, zi=np.moveaxis(zi_ref, -2, 0))
                    zf = np.moveaxis(zf, 0, -2)
                    assert y.dtype == np.float64 and np.isfinite(y).all()
                    params = {"sos": sos.ravel().tolist(), "n_sections": int(ns),
                              "input_shape": shape, "dtype": dtype}
                    if zi is not None:
                        params["zi"] = np.asarray(zi).ravel().tolist()
                        params["zi_shape"] = list(np.shape(zi))
                    name = f"{key}_{layout}_{dtype}"
                    cases.append({"name": name, "params": {**params, "output": "y"},
                                  "shape": list(y.shape), "values": y.ravel().tolist()})
                    cases.append({"name": name, "params": {**params, "output": "zf"},
                                  "shape": list(zf.shape), "values": zf.ravel().tolist()})
        return cases

    def filtfilt_cases(self):
        cases = []
        low4 = self.butter(4, "lowpass", [1000.0], 16000, "ba")
        band2 = self.butter(2, "bandpass", [300.0, 3400.0], 16000, "ba")
        high3 = self.butter(3, "highpass", [300.0], 16000, "ba")
        a0 = ([0.3, -0.2, 0.1], [2.5, -0.5, 0.25])
        sos5 = self.butter(5, "lowpass", [1000.0], 16000, "sos")
        sos_band4 = self.butter(4, "bandpass", [300.0, 3400.0], 16000, "sos")
        sos_high6 = self.butter(6, "highpass", [300.0], 16000, "sos")
        # (key, design, input shape, pad, padlen, dtype)
        grid = [
            ("low4", low4, [300], "odd", None, "float64"),
            ("low4", low4, [300], "odd", None, "float32"),
            ("low4_min_length", low4, [16], "odd", None, "float64"),
            ("low4_min_length_plus_1", low4, [17], "odd", None, "float64"),
            ("low4_padlen_n_minus_1", low4, [40], "odd", 39, "float64"),
            ("low4_padlen0", low4, [300], "odd", 0, "float64"),
            ("low4_even", low4, [300], "even", None, "float64"),
            ("low4_constant", low4, [300], "constant", None, "float64"),
            ("low4_2ch", low4, [2, 200], "odd", None, "float64"),
            ("band2", band2, [300], "odd", None, "float64"),
            ("high3_batch", high3, [2, 3, 60], "odd", None, "float64"),
            ("a0_2.5", a0, [300], "odd", None, "float64"),
            ("sos_low5", sos5, [300], "odd", None, "float64"),
            ("sos_low5", sos5, [300], "odd", None, "float32"),
            ("sos_low5_min_length", sos5, [19], "odd", None, "float64"),
            ("sos_low5_min_length_plus_1", sos5, [20], "odd", None, "float64"),
            ("sos_low5_padlen_n_minus_1", sos5, [40], "odd", 39, "float64"),
            ("sos_low5_padlen0", sos5, [300], "odd", 0, "float64"),
            ("sos_low5_even", sos5, [300], "even", None, "float64"),
            ("sos_low5_constant", sos5, [300], "constant", None, "float64"),
            ("sos_band4_2ch", sos_band4, [2, 200], "odd", None, "float64"),
            ("sos_high6_batch", sos_high6, [2, 3, 60], "odd", None, "float64"),
        ]
        for key, design, shape, pad, padlen, dtype in grid:
            x = self.input(shape, dtype)
            if isinstance(design, np.ndarray):
                y = scipy.signal.sosfiltfilt(design, x, padtype=pad, padlen=padlen)
                params = {"function": "sosfiltfilt", "sos": design.ravel().tolist(),
                          "n_sections": int(design.shape[0])}
            else:
                b, a = design
                y = scipy.signal.filtfilt(b, a, x, padtype=pad, padlen=padlen)
                params = {"function": "filtfilt", "b": list(map(float, b)),
                          "a": list(map(float, a))}
            assert y.dtype == np.float64 and np.isfinite(y).all()
            params.update({"pad": pad, "input_shape": shape, "dtype": dtype})
            if padlen is not None:
                params["padlen"] = padlen
            cases.append({"name": f"{key}_{dtype}", "params": params,
                          "shape": list(y.shape), "values": y.ravel().tolist()})
        # the minimum lengths are the defaults' boundary: one sample fewer raises
        for design, n in ((low4, 15), (sos5, 18)):
            try:
                if isinstance(design, np.ndarray):
                    scipy.signal.sosfiltfilt(design, self.signal(n))
                else:
                    scipy.signal.filtfilt(*design, self.signal(n))
            except ValueError:
                pass
            else:
                raise AssertionError(f"padlen boundary moved: {n} samples accepted")
        return cases

    # (key, function, coef or None for the default, zi: None, "row" for one
    # state per row or a float for one state broadcast over every row, input
    # shape, dtypes)
    EMPHASIS = [
        ("pre_default_1d", "preemphasis", None, None, [300], ["float64", "float32"]),
        ("pre_default_2ch", "preemphasis", None, None, [2, 200], ["float64"]),
        ("pre_default_batch", "preemphasis", None, None, [2, 3, 60], ["float64"]),
        ("pre_default_two_samples", "preemphasis", None, None, [2], ["float64"]),
        ("pre_coef0.5", "preemphasis", 0.5, None, [300], ["float64"]),
        ("pre_coef0.999", "preemphasis", 0.999, None, [300], ["float64"]),
        ("pre_coef0_zi0", "preemphasis", 0.0, 0.0, [300], ["float64"]),
        ("pre_zi_scalar", "preemphasis", None, 0.3, [2, 200], ["float64", "float32"]),
        ("pre_zi_row", "preemphasis", None, "row", [2, 200], ["float64"]),
        ("de_default_1d", "deemphasis", None, None, [300], ["float64", "float32"]),
        ("de_default_2ch", "deemphasis", None, None, [2, 200], ["float64"]),
        ("de_default_batch", "deemphasis", None, None, [2, 3, 60], ["float64"]),
        ("de_default_two_samples", "deemphasis", None, None, [2], ["float64"]),
        ("de_coef0.5", "deemphasis", 0.5, None, [300], ["float64"]),
        ("de_coef0.999", "deemphasis", 0.999, None, [300], ["float64"]),
        ("de_coef0_zi0", "deemphasis", 0.0, 0.0, [300], ["float64"]),
        ("de_zi_scalar", "deemphasis", None, 0.3, [2, 200], ["float64", "float32"]),
        ("de_zi_row", "deemphasis", None, "row", [2, 200], ["float64"]),
    ]

    def emphasis_cases(self):
        cases = []
        for key, function, coef, zi, shape, dtypes in self.EMPHASIS:
            for dtype in dtypes:
                x = self.input(shape, dtype)
                lead = shape[:-1]
                kwargs = {} if coef is None else {"coef": coef}
                params = {"function": function, "input_shape": shape, "dtype": dtype}
                if coef is not None:
                    params["coef"] = coef
                if zi is None:
                    zi_ours = None
                elif zi == "row":
                    zi_ours = self.state(lead + [1], 5)
                else:
                    zi_ours = np.full([1], zi)
                if zi_ours is not None:
                    params["zi"] = zi_ours.ravel().tolist()
                    params["zi_shape"] = list(zi_ours.shape)
                    # librosa hands zi to lfilter, which wants every dimension
                    kwargs["zi"] = np.broadcast_to(zi_ours, lead + [1]).copy()
                fn = getattr(librosa.effects, function)
                y, zf = fn(x, return_zf=True, **kwargs)
                assert y.dtype == np.float64 and np.isfinite(y).all()
                if function == "deemphasis" and zi is None:
                    c = 0.97 if coef is None else coef
                    # librosa's own state is its uncorrected recursion's
                    _, uncorrected = scipy.signal.lfilter(
                        [1.0], [1.0, -c], x, zi=np.zeros(lead + [1]))
                    assert np.array_equal(zf, uncorrected), key
                    zf = c * y[..., -1:]
                name = f"{key}_{dtype}"
                cases.append({"name": name, "params": {**params, "output": "y"},
                              "shape": list(y.shape), "values": y.ravel().tolist()})
                cases.append({"name": name, "params": {**params, "output": "zf"},
                              "shape": list(zf.shape), "values": zf.ravel().tolist()})
        # the round trip of the defaults
        for shape in ([300], [2, 200]):
            x = self.input(shape, "float64")
            y = librosa.effects.deemphasis(librosa.effects.preemphasis(x))
            cases.append({"name": f"round_trip_{'x'.join(map(str, shape))}",
                          "params": {"function": "round_trip", "input_shape": shape,
                                     "dtype": "float64", "output": "y"},
                          "shape": list(y.shape), "values": y.ravel().tolist()})
        return cases

    def generate(self):
        write_suite(self.SUITE, "butter", self.butter_cases())
        write_suite(self.SUITE, "zi", self.zi_cases())
        write_suite(self.SUITE, "lfilter", self.lfilter_cases())
        write_suite(self.SUITE, "sosfilt", self.sosfilt_cases())
        write_suite(self.SUITE, "filtfilt", self.filtfilt_cases())
        write_suite(self.SUITE, "emphasis", self.emphasis_cases())


class DefaultsVectorGenerator:
    """Golden vectors for the library's defaults against librosa's: every
    call leaves pad_mode, center and window at librosa 1.0's defaults
    (constant zero padding, centered frames, the Hann window), over the LCG
    stream under this suite's seed.

    Three files:

    - stft: librosa.stft given n_fft alone -- magnitudes at two lengths and
      both dtypes, a signal shorter than half the transform (which reflection
      could only fill by reflecting repeatedly), the transform with every
      argument at its default, and the real and imaginary parts of one
      complex case.
    - features: librosa.feature.melspectrogram (at n_fft 512 and with every
      analysis argument at its default), mfcc, chroma_stft, spectral_contrast,
      spectral_centroid and librosa.onset.onset_strength, plus rms and
      zero_crossing_rate at their defaults. The feature functions fix
      hop_length at 512 whatever n_fft is, so the n_fft 512 cases pass 128 --
      what librosa.stft and Stft.Config both default to there; the mel and
      chroma filterbanks are computed in float64, chroma pins tuning=0.0
      (librosa estimates it from the signal), and rms asks for a float64
      result.
    - synthesis: librosa.griffinlim at init=None (its only deterministic
      start), librosa.effects.time_stretch and librosa.effects.hpss given
      n_fft alone.

    Every case that accepts pad_mode is asserted to move by more than a
    thousandth of its peak under pad_mode="reflect", so each one pins the
    default padding rather than merely passing through it; time_stretch
    cannot take the argument (it forwards its keywords to istft too) and
    zero_crossing_rate has none. float32 cases quantize the signal to
    float32 and compute the reference in float64.
    """

    SUITE = "defaults"

    SEED = 20261026

    SR = 22050

    def signal(self, n, dtype="float64"):
        y = StftVectorGenerator.lcg_signal(n, seed=self.SEED)
        return y if dtype == "float64" else y.astype(np.float32).astype(np.float64)

    @staticmethod
    def case(name, params, compute, sensitive=True):
        """One case: compute(**extra) is the librosa call at its defaults;
        when sensitive, the same call under pad_mode="reflect" must differ."""
        expected = np.asarray(compute(), dtype=np.float64)
        assert np.isfinite(expected).all(), name
        if sensitive:
            moved = np.max(np.abs(np.asarray(compute(pad_mode="reflect")) - expected))
            assert moved > 1e-3 * np.max(np.abs(expected)), (name, moved)
        return {"name": name, "params": params, "shape": list(expected.shape),
                "values": expected.ravel().tolist()}

    def stft_cases(self):
        cases = []
        # (fft_size, length, dtype, keyword arguments librosa.stft is given)
        for fft_size, n, dtype, kwargs in [
            (64, 127, "float64", {"n_fft": 64}),
            (64, 128, "float64", {"n_fft": 64}),
            (64, 127, "float32", {"n_fft": 64}),
            (64, 20, "float64", {"n_fft": 64}),
            (2048, 3000, "float64", {}),
        ]:
            y = self.signal(n, dtype)
            cases.append(self.case(
                f"magnitude_fft{fft_size}_n{n}_{dtype}",
                {"kind": "magnitude", "fft_size": fft_size, "length": n,
                 "dtype": dtype},
                lambda y=y, kwargs=kwargs, **extra: np.abs(librosa.stft(
                    y, dtype=np.complex128, **kwargs, **extra))))
        y = self.signal(37)
        for kind, part in (("real", np.real), ("imag", np.imag)):
            cases.append(self.case(
                f"{kind}_fft16_n37_float64",
                {"kind": kind, "fft_size": 16, "length": 37, "dtype": "float64"},
                lambda part=part, **extra: part(librosa.stft(
                    y, n_fft=16, dtype=np.complex128, **extra))))
        return cases

    def feature_cases(self):
        sr = self.SR
        y = self.signal(1000)
        y32 = self.signal(1000, "float32")
        long = self.signal(3000)
        geometry = {"n_fft": 512, "hop_length": 128}
        mel = {"n_mels": 40, "dtype": np.float64}

        def params(kind, n, dtype="float64", **extra):
            return {"kind": kind, "length": n, "dtype": dtype, **extra}

        return [
            self.case("melspectrogram_fft512_mel40_float64",
                      params("melspectrogram", 1000, fft_size=512, hop=128,
                             n_mels=40),
                      lambda **extra: librosa.feature.melspectrogram(
                          y=y, sr=sr, **geometry, **mel, **extra)),
            self.case("melspectrogram_fft512_mel40_float32",
                      params("melspectrogram", 1000, "float32", fft_size=512,
                             hop=128, n_mels=40),
                      lambda **extra: librosa.feature.melspectrogram(
                          y=y32, sr=sr, **geometry, **mel, **extra)),
            self.case("melspectrogram_defaults",
                      params("melspectrogram", 3000, fft_size=2048, hop=512,
                             n_mels=128),
                      lambda **extra: librosa.feature.melspectrogram(
                          y=long, sr=sr, dtype=np.float64, **extra)),
            self.case("mfcc_fft512_mel40",
                      params("mfcc", 1000, fft_size=512, hop=128, n_mels=40),
                      lambda **extra: librosa.feature.mfcc(
                          y=y, sr=sr, **geometry, **mel, **extra)),
            self.case("chroma_stft_fft512",
                      params("chroma_stft", 1000, fft_size=512, hop=128),
                      lambda **extra: librosa.feature.chroma_stft(
                          y=y, sr=sr, **geometry, tuning=0.0, dtype=np.float64,
                          **extra)),
            self.case("spectral_contrast_fft512",
                      params("spectral_contrast", 1000, fft_size=512, hop=128),
                      lambda **extra: librosa.feature.spectral_contrast(
                          y=y, sr=sr, **geometry, **extra)),
            self.case("spectral_centroid_fft512",
                      params("spectral_centroid", 1000, fft_size=512, hop=128),
                      lambda **extra: librosa.feature.spectral_centroid(
                          y=y, sr=sr, **geometry, **extra)),
            self.case("onset_strength_fft512_mel40",
                      params("onset_strength", 1000, fft_size=512, hop=128,
                             n_mels=40),
                      lambda **extra: librosa.onset.onset_strength(
                          y=y, sr=sr, **geometry, **mel, **extra)),
            self.case("onset_strength_defaults",
                      params("onset_strength", 3000, fft_size=2048, hop=512,
                             n_mels=128),
                      lambda **extra: librosa.onset.onset_strength(
                          y=long, sr=sr, dtype=np.float64, **extra)),
            self.case("rms_defaults", params("rms", 3000),
                      lambda **extra: librosa.feature.rms(
                          y=long, dtype=np.float64, **extra)),
            self.case("zero_crossing_rate_defaults",
                      params("zero_crossing_rate", 3000),
                      lambda: librosa.feature.zero_crossing_rate(long),
                      sensitive=False),
        ]

    def synthesis_cases(self):
        bins, frames = 33, 20
        magnitudes = np.abs(self.signal(bins * frames)).reshape(bins, frames) + 0.5
        y = self.signal(1000)
        return [
            self.case("griffinlim_fft64",
                      {"kind": "griffinlim", "fft_size": 64, "bins": bins,
                       "frames": frames, "dtype": "float64"},
                      lambda **extra: librosa.griffinlim(
                          magnitudes, init=None, **extra)),
            self.case("time_stretch_fft256_r1p25",
                      {"kind": "time_stretch", "fft_size": 256, "rate": 1.25,
                       "length": 1000, "dtype": "float64"},
                      lambda: librosa.effects.time_stretch(y, rate=1.25,
                                                           n_fft=256),
                      sensitive=False),
            self.case("hpss_harmonic_fft256",
                      {"kind": "harmonic", "fft_size": 256, "length": 1000,
                       "dtype": "float64"},
                      lambda **extra: librosa.effects.hpss(
                          y, n_fft=256, **extra)[0]),
            self.case("hpss_percussive_fft256",
                      {"kind": "percussive", "fft_size": 256, "length": 1000,
                       "dtype": "float64"},
                      lambda **extra: librosa.effects.hpss(
                          y, n_fft=256, **extra)[1]),
        ]

    def generate(self):
        write_suite(self.SUITE, "stft", self.stft_cases())
        write_suite(self.SUITE, "features", self.feature_cases())
        write_suite(self.SUITE, "synthesis", self.synthesis_cases())


class SynthesisVectorGenerator:
    """Golden vectors for Soundml.tone, chirp, clicks, mu_compress and
    mu_expand: librosa.tone, librosa.chirp (over scipy.signal.chirp),
    librosa.clicks and librosa.mu_compress / mu_expand. Parameters are
    recorded under the library's names (sample_rate, phase, hop,
    click_frequency); an absent phase is the default -pi/2.

    Four files:

    - tone: frequencies from 55 Hz to just under Nyquist and a negative one,
      the default and explicit phases, lengths given directly or as
      durations (truncated toward zero, and a length winning over a
      duration), and the last 1024 samples ("tail") of a tone of 2 ** 22
      samples, whose arguments reach 5e5 radians.
    - chirp: exponential and linear sweeps, ascending and descending, the
      exponential sweep's equal-frequency branch, a linear sweep from 0 Hz,
      a phase whose degree round trip through librosa and scipy moves its
      bits (asserted), and lengths for which librosa returns one sample
      fewer than asked. Every length case stores scipy.signal.chirp over all
      the samples asked for on librosa's sweep time (1 / sr) * length,
      asserted equal to librosa's output on the samples librosa returns
      ("reference_length" records how many that is when it falls short).
      The second-long 20 Hz to 20 kHz sweep keeps its last 2048 samples
      ("tail"), where its phase, and so its sensitivity to the C library's
      power, is largest.
    - clicks: the default click placed at times, frames and sample indices,
      overlapping clicks in unsorted order with a duplicate, clicks cut by
      the end of the track and positions past it dropped, a click ending
      exactly at the length, a shorter, lower click at 16 kHz, and a
      two-channel click cut from the LCG stream. The float32 cases are
      librosa's own float32 track; the float64 cases replay librosa's
      placement loop on a float64 track (librosa has none), the replay
      asserted bit-identical to librosa when run on a float32 track. A
      float32 case hands librosa the custom click quantized to float32, the
      click the library's float32 track adds. Times are (k + 0.5) / sr, half
      a sample from the truncation's ties.
    - mu: the continuous compression and expansion over the LCG stream plus
      the exact points -1, -0, 0, 0.5 and 1, quantized codes at mu 255, 15,
      100.5 (a non-integer mu: 101 levels), 1 (two levels) and 254 (an odd
      level count, with a level edge at exactly 0, which sample 0 meets), the
      expansion of every code at mu 255 and 15, and quantized round trips.
      Inputs are stored in the params. Every quantized input is kept only
      when its compressed value lies at least 1e-9 from every level edge or
      exactly on one by exact arithmetic (the points -1, 0 and 1), so no
      rounding of the logarithm can move a code.

    float32 cases quantize their input (where there is one) to float32 and
    compute the reference in float64.
    """

    SUITE = "synthesis"

    SEED = 20261103

    def lcg(self, n, seed_offset=0):
        return StftVectorGenerator.lcg_signal(n, seed=self.SEED + seed_offset)

    @staticmethod
    def quantize(y, dtype):
        y = np.asarray(y, dtype=np.float64)
        return y if dtype == "float64" else y.astype(np.float32).astype(np.float64)

    # (name, frequency, sample_rate, length, duration, phase, dtypes, tail)
    TONES = [
        ("a440_22k", 440.0, 22050, 2048, None, None, ["float64", "float32"], None),
        ("1k_44k_duration", 1000.0, 44100, None, 0.05, None, ["float64"], None),
        ("997.3_48k_phase0", 997.3, 48000, 4800, None, 0.0,
         ["float64", "float32"], None),
        ("near_nyquist_phase_pi3", 11013.975, 22050, 1000, None, np.pi / 3,
         ["float64"], None),
        ("55_16k_duration", 55.0, 16000, None, 0.3, None, ["float64"], None),
        ("length_wins", 440.0, 22050, 7, 1.0, None, ["float64"], None),
        ("negative_frequency", -440.0, 22050, 64, None, None, ["float64"], None),
        ("long_tail", 440.0, 22050, 1 << 22, None, None, ["float64"], 1024),
    ]

    def tone_cases(self):
        cases = []
        for name, f, sr, length, duration, phase, dtypes, tail in self.TONES:
            kwargs = {"sr": sr}
            if length is not None:
                kwargs["length"] = length
            if duration is not None:
                kwargs["duration"] = duration
            if phase is not None:
                kwargs["phi"] = phase
            y = librosa.tone(f, **kwargs)
            assert y.dtype == np.float64
            expected_len = length if length is not None else int(duration * sr)
            assert len(y) == expected_len, name
            if tail is not None:
                y = y[-tail:]
            for dtype in dtypes:
                params = {"frequency": f, "sample_rate": sr, "dtype": dtype}
                if length is not None:
                    params["length"] = length
                if duration is not None:
                    params["duration"] = duration
                if phase is not None:
                    params["phase"] = phase
                if tail is not None:
                    params["tail"] = tail
                cases.append({"name": f"{name}_{dtype}", "params": params,
                              "shape": list(y.shape), "values": y.tolist()})
        return cases

    # (name, fmin, fmax, sample_rate, length, duration, linear, phase, dtypes)
    CHIRPS = [
        ("exp_up_22k", 110.0, 7040.0, 22050, 4096, None, False, None,
         ["float64", "float32"]),
        ("exp_down_22k_duration", 7040.0, 110.0, 22050, None, 0.2, False, None,
         ["float64"]),
        ("lin_up_22k", 110.0, 7040.0, 22050, 4096, None, True, None,
         ["float64", "float32"]),
        ("lin_down_44k_duration_phase", 8000.0, 50.0, 44100, None, 0.1, True,
         0.5, ["float64"]),
        ("exp_equal", 440.0, 440.0, 22050, 1024, None, False, None, ["float64"]),
        ("lin_from_zero_16k", 0.0, 1000.0, 16000, 1600, None, True, None,
         ["float64"]),
        ("exp_wide_48k", 20.0, 20000.0, 48000, 48000, None, False, None,
         ["float64"]),
        ("exp_short_48k", 110.0, 880.0, 48000, 49, None, False, None,
         ["float64"]),
        ("lin_short_48k", 110.0, 880.0, 48000, 7, None, True, None,
         ["float64"]),
        ("exp_357_22k", 110.0, 880.0, 22050, 357, None, False, None,
         ["float64"]),
        ("lin_357_22k_phase0.7", 110.0, 880.0, 22050, 357, None, True, 0.7,
         ["float64"]),
    ]

    CHIRP_TAILS = {"exp_wide_48k": 2048}

    def chirp_cases(self):
        cases = []
        for (name, fmin, fmax, sr, length, duration, linear, phase,
             dtypes) in self.CHIRPS:
            kwargs = {"fmin": fmin, "fmax": fmax, "sr": sr, "linear": linear}
            if length is not None:
                kwargs["length"] = length
            if duration is not None:
                kwargs["duration"] = duration
            if phase is not None:
                kwargs["phi"] = phase
            y = librosa.chirp(**kwargs)
            assert y.dtype == np.float64
            phi = -np.pi * 0.5 if phase is None else phase
            if phase == 0.7:
                # the degree round trip moves this phase's bits
                assert np.deg2rad(phi / np.pi * 180) != phi
            reference_length = None
            if length is not None:
                # all the samples asked for, on librosa's sweep time
                t1 = (1.0 / sr) * length
                full = scipy.signal.chirp(
                    np.arange(length) / sr, fmin, t1, fmax,
                    method="linear" if linear else "logarithmic",
                    phi=phi / np.pi * 180)
                assert np.array_equal(full[:len(y)], y), name
                if len(y) != length:
                    assert len(y) == length - 1, name
                    reference_length = len(y)
                y = full
            else:
                assert len(y) == int(duration * sr), name
            for dtype in dtypes:
                params = {"fmin": fmin, "fmax": fmax, "sample_rate": sr,
                          "linear": linear, "dtype": dtype}
                if length is not None:
                    params["length"] = length
                if duration is not None:
                    params["duration"] = duration
                if phase is not None:
                    params["phase"] = phase
                if reference_length is not None:
                    params["reference_length"] = reference_length
                cases.append({"name": f"{name}_{dtype}", "params": params,
                              "shape": list(y.shape), "values": y.tolist()})
        for case in cases:
            tail = self.CHIRP_TAILS.get(case["name"].rsplit("_", 1)[0])
            if tail is not None:
                case["params"]["tail"] = tail
                case["values"] = case["values"][-tail:]
                case["shape"] = [tail]
            # the short and 357-sample cases are the ones librosa cuts short
            meant = "short" in case["name"] or "357" in case["name"]
            assert meant == ("reference_length" in case["params"]), case["name"]
        return cases

    @staticmethod
    def place(click, positions, length, dtype):
        """librosa.clicks' placement loop on a track of `dtype`."""
        shape = list(click.shape)
        shape[-1] = length
        track = np.zeros(shape, dtype=dtype)
        for start in positions:
            end = start + click.shape[-1]
            if end >= length:
                track[..., start:] += click[..., : length - start]
            else:
                track[..., start:end] += click
        return track

    @staticmethod
    def default_click(sr, click_freq, click_duration):
        """librosa.clicks' default click, as its source computes it."""
        angular_freq = 2 * np.pi * click_freq / float(sr)
        click = np.logspace(0, -10, num=int(sr * click_duration), base=2.0)
        click *= np.sin(angular_freq * np.arange(len(click)))
        return click

    def custom_click(self, shape):
        return self.lcg(int(np.prod(shape)), seed_offset=1).reshape(shape)

    def clicks_cases(self):
        sr = 22050

        def times(samples, rate=sr):
            return [(k + 0.5) / rate for k in samples]

        both = ["float64", "float32"]
        # (name, positions {"times"|"frames"|"samples": [...]}, extra params,
        #  dtypes)
        specs = [
            ("times_default", {"times": times([0, 2300, 4410])}, {},
             ["float64"]),
            ("frames_default", {"frames": [0, 2, 5]}, {}, ["float32"]),
            ("samples_overlap_unsorted",
             {"samples": [1102, 0, 441, 1102]}, {}, both),
            ("frames_cut_and_dropped", {"frames": [0, 3, 5, 9]},
             {"length": 3000}, both),
            ("times_end_at_length", {"times": times([0])}, {"length": 2205},
             ["float32"]),
            ("frames_880hz_16k", {"frames": [0, 2, 5]},
             {"sample_rate": 16000, "hop": 160, "click_frequency": 880.0,
              "click_duration": 0.05}, both),
            ("custom_stereo", {"samples": [0, 16, 40]},
             {"length": 100, "click_shape": [2, 64]}, both),
            ("custom_stereo_overlap_end", {"samples": [60, 0, 30, 60]},
             {"length": 100, "click_shape": [2, 64]}, both),
        ]
        cases = []
        for name, where, extra, dtypes in specs:
            rate = extra.get("sample_rate", sr)
            hop = extra.get("hop", 512)
            kwargs = {"sr": rate, "hop_length": hop}
            if "click_frequency" in extra:
                kwargs["click_freq"] = extra["click_frequency"]
            if "click_duration" in extra:
                kwargs["click_duration"] = extra["click_duration"]
            if "length" in extra:
                kwargs["length"] = extra["length"]
            # positions as librosa computes them
            if "times" in where:
                kwargs["times"] = np.asarray(where["times"])
                positions = (kwargs["times"] * rate).astype(int)
            elif "frames" in where:
                kwargs["frames"] = np.asarray(where["frames"])
                positions = kwargs["frames"] * hop
            else:
                # librosa takes sample indices as frames of a one-sample hop
                kwargs["frames"] = np.asarray(where["samples"])
                kwargs["hop_length"] = 1
                positions = kwargs["frames"]
            if "times" in where:
                # every time sits half a sample past its sample
                frac = kwargs["times"] * rate - positions
                assert (np.abs(frac - 0.5) < 1e-6).all(), name
            for dtype in dtypes:
                if "click_shape" in extra:
                    # the library adds a click of the track's own dtype
                    click = self.custom_click(extra["click_shape"])
                    if dtype == "float32":
                        click = click.astype(np.float32)
                    y32 = librosa.clicks(click=click, **kwargs)
                else:
                    click = self.default_click(
                        rate, kwargs.get("click_freq", 1000.0),
                        kwargs.get("click_duration", 0.1))
                    y32 = librosa.clicks(**kwargs)
                assert y32.dtype == np.float32
                if "length" in kwargs:
                    length = kwargs["length"]
                    kept = positions[positions < length]
                else:
                    length = positions.max() + click.shape[-1]
                    kept = positions
                replay32 = self.place(click, kept, length, np.float32)
                assert np.array_equal(
                    replay32.view(np.uint32), y32.view(np.uint32)), name
                if dtype == "float32":
                    expected = y32.astype(np.float64)
                else:
                    expected = self.place(click, kept, length, np.float64)
                params = {"sample_rate": rate, "dtype": dtype}
                params.update(where)
                params.update({k: v for k, v in extra.items()
                               if k != "sample_rate"})
                if "hop" not in params and "frames" in where:
                    params["hop"] = hop
                cases.append({"name": f"{name}_{dtype}", "params": params,
                              "shape": list(expected.shape),
                              "values": expected.ravel().tolist()})
        return cases

    # the least distance from a compressed value to a level edge
    MARGIN = 1e-9

    @staticmethod
    def compressed(x, mu):
        return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)

    def keeps_margin(self, x, mu):
        """Whether the code of each x is safe from any rounding of the
        logarithm: its compressed value lies MARGIN or more from every level
        edge, or x is one of the exact points -1, 0, 1."""
        edges = np.linspace(-1, 1, num=int(1 + mu), endpoint=True)
        c = self.compressed(x, mu)
        gap = np.min(np.abs(c[:, None] - edges[None, :]), axis=1)
        exact = np.isin(x, [-1.0, 0.0, 1.0])
        return (gap >= self.MARGIN) | exact

    def mu_inputs(self, n, seed_offset, dtype, mu=None):
        x = self.quantize(self.lcg(n, seed_offset=seed_offset), dtype)
        specials = self.quantize([-1.0, -0.5, -0.0, 0.0, 0.5, 1.0], dtype)
        x = np.concatenate([x, specials])
        if mu is not None:
            x = x[self.keeps_margin(x, mu)]
        return x

    def mu_case(self, name, op, mu, quantize, x, dtype, shape=None, compute=None):
        x = np.asarray(x, dtype=np.float64)
        if shape is not None:
            x = x.reshape(shape)
        if compute is None:
            if op == "compress":
                y = librosa.mu_compress(x, mu=mu, quantize=quantize)
            else:
                y = librosa.mu_expand(x, mu=mu, quantize=quantize)
        else:
            y = compute(x)
        y = np.asarray(y)
        assert y.shape == x.shape
        if op == "compress" and quantize:
            assert y.dtype == np.int64
        y = y.astype(np.float64)
        assert np.isfinite(y).all(), name
        return {"name": f"{name}_{dtype}",
                "params": {"op": op, "mu": mu, "quantize": quantize,
                           "dtype": dtype, "input_shape": list(x.shape),
                           "input": x.ravel().tolist()},
                "shape": list(y.shape), "values": y.ravel().tolist()}

    def mu_cases(self):
        cases = []
        # continuous compression
        for mu, dtypes, shape in [(255.0, ["float64", "float32"], [2, 259]),
                                  (15.0, ["float64"], None)]:
            for dtype in dtypes:
                x = self.mu_inputs(512, 2, dtype)
                cases.append(self.mu_case(f"compress_mu{mu:g}", "compress", mu,
                                          False, x, dtype, shape=shape))
        # quantized codes
        for mu, dtypes in [(255.0, ["float64", "float32"]), (15.0, ["float64"]),
                           (100.5, ["float64"]), (1.0, ["float64"]),
                           (254.0, ["float64"])]:
            for dtype in dtypes:
                x = self.mu_inputs(512, 3, dtype, mu=mu)
                assert len(x) > 400, (mu, len(x))
                case = self.mu_case(f"codes_mu{mu:g}", "compress", mu, True, x,
                                    dtype)
                values = np.asarray(case["values"])
                offset = int(mu + 1) // 2
                assert values.min() == -offset
                assert values.max() == int(1 + mu) - 1 - offset
                if mu == 254.0:
                    # the level edge at exactly 0: sample 0 sits on it
                    assert np.linspace(-1, 1, num=255)[127] == 0.0
                    assert values[np.asarray(case["params"]["input"]) == 0.0][0] == 0
                cases.append(case)
        # continuous expansion
        for dtype in ["float64", "float32"]:
            x = self.mu_inputs(512, 4, dtype)
            cases.append(self.mu_case("expand_mu255", "expand", 255.0, False, x,
                                      dtype))
        # every code, expanded
        for mu, top, dtypes in [(255.0, 128, ["float64", "float32"]),
                                (15.0, 8, ["float64"])]:
            codes = np.arange(-top, top + 1, dtype=np.float64)
            for dtype in dtypes:
                cases.append(self.mu_case(f"expand_codes_mu{mu:g}", "expand", mu,
                                          True, codes, dtype))
        # quantized round trips
        for mu in [255.0, 15.0]:
            x = self.mu_inputs(512, 5, "float64", mu=mu)
            cases.append(self.mu_case(
                f"roundtrip_mu{mu:g}", "roundtrip", mu, True, x, "float64",
                compute=lambda v, mu=mu: librosa.mu_expand(
                    librosa.mu_compress(v, mu=mu), mu=mu)))
        return cases

    def generate(self):
        write_suite(self.SUITE, "tone", self.tone_cases())
        write_suite(self.SUITE, "chirp", self.chirp_cases())
        write_suite(self.SUITE, "clicks", self.clicks_cases())
        write_suite(self.SUITE, "mu", self.mu_cases())
class F0VectorGenerator:
    """Golden vectors for Soundml.yin and Soundml.pyin: librosa.yin and
    librosa.pyin over synthetic signals, each channel a run of segments
    [length, recipe] evaluated on the segment's own time t = i / sr:

    - ["silence"];
    - ["tone", f, amp, phase]: amp * sin(2 pi f t + phase);
    - ["harmonic", f, k]: the sum over h = 1..k of
      (0.6 / h) * sin(2 pi (h f) t + 0.1 h);
    - ["vibrato", fc, depth, rate]:
      0.5 * sin(2 pi fc t + (depth / rate) * sin(2 pi rate t));
    - ["chirp", f0, f1]: 0.5 * sin(2 pi (f0 t + (0.5 k) t t)), linear from
      f0 to f1 over the segment, k = (f1 - f0) / (length / sr);
    - ["noise", scale, seed]: scale times the LCG stream under that seed;
    - ["offset", c, f, amp]: c + amp * sin(2 pi f t + 0.3).

    A case's "lead" is its leading shape and "channels" lists one segment run
    per channel in C order. Geometries: librosa's defaults (22.05 kHz, C2 to
    C7, 2048-sample frames), a 16 kHz speech geometry, an uncentered odd
    frame length at 44.1 kHz, 8 kHz with edge and reflect padding (and a hop
    longer than the frame), signals shorter than a frame, a [2; 3] batch,
    and pyin's own parameters, fill modes and Viterbi pruning thresholds.
    The constant-0.5 padding has no librosa argument: its reference is
    librosa on the signal padded by hand, uncentered.

    librosa computes the difference function through an FFT
    autocorrelation and its decisions are discrete: the troughs, the
    absolute threshold, the global minimum, the pyin thresholds, the
    parabola branch, the pitch-bin rounding and the Viterbi maxima. Every
    case asserts that each decision of every frame whose cumulative mean
    normalized difference is not identically zero lies at least MARGIN from
    its tie on librosa's own difference function, and that multiplying the
    input by 1 + 2 ** -40 times an LCG stream leaves every yin estimate
    within 1e-9 relative and every decoded pyin state unchanged. The
    sinusoids are rebuilt on the OCaml side with libm, within ulps of
    numpy's; the margins cover that as well. float32 cases quantize the
    input to float32 and compute the reference in float64. pyin cases stack
    f0 (NaN where unvoiced, unless the case fills), the voiced flag as 0/1
    and the voiced probability on a new leading axis.
    """

    SUITE = "f0"

    SEED = 20261103

    C2 = 65.40639132514966  # librosa.note_to_hz("C2")

    C7 = 2093.004522404789  # librosa.note_to_hz("C7")

    MARGIN = 1e-9

    DEFAULT = {"sample_rate": 22050, "fmin": C2, "fmax": C7}

    SPEECH = {"sample_rate": 16000, "fmin": 50.0, "fmax": 800.0,
              "frame_length": 1024, "hop": 160}

    ODD = {"sample_rate": 44100, "fmin": 100.0, "fmax": 1000.0,
           "frame_length": 2047, "hop": 300, "center": False}

    EDGE = {"sample_rate": 8000, "fmin": 80.0, "fmax": 400.0,
            "frame_length": 512, "hop": 128, "pad": "edge"}

    REFLECT = dict(EDGE, pad="reflect")

    REFLECT_LONG_HOP = dict(EDGE, pad="reflect", hop=700)

    VARIANT = {"n_thresholds": 50, "beta_parameters": [1.5, 12.0],
               "boltzmann_parameter": 1.5, "resolution": 0.05,
               "max_transition_rate": 20.0, "switch_prob": 0.05,
               "no_trough_prob": 0.05, "fill": 0.0, "fmin": 70.0,
               "fmax": 1100.0, "hop": 256}

    N = 11025

    @classmethod
    def mono(cls, *segments):
        return {"lead": [], "channels": [list(segments)]}

    @classmethod
    def whole(cls, recipe, n):
        return cls.mono([n, recipe])

    @classmethod
    def signals(cls, n):
        half = n // 2
        quiet = n * 4 // 11
        return {
            "tone97": cls.whole(["tone", 97.3, 0.5, 0.3], n),
            "tone311": cls.whole(["tone", 311.7, 0.5, 0.3], n),
            "tone1234": cls.whole(["tone", 1234.5, 0.4, 1.1], n),
            "harmonic147": cls.whole(["harmonic", 146.83, 6], n),
            "vibrato250": cls.whole(["vibrato", 250.0, 8.0, 5.0], n),
            "chirp120_380": cls.whole(["chirp", 120.0, 380.0], n),
            "noise": cls.whole(["noise", 0.3, cls.SEED], n),
            "silence": cls.whole(["silence"], n),
            "tone_to_silence": cls.mono([half, ["tone", 203.3, 0.5, 0.3]],
                                        [n - half, ["silence"]]),
            "silence_to_tone": cls.mono([quiet, ["silence"]],
                                        [n - quiet, ["tone", 523.0, 0.5, 0.0]]),
            "offset_tone": cls.whole(["offset", 0.2, 173.9, 0.4], n),
        }

    @classmethod
    def batch(cls, n):
        s = cls.signals(n)
        names = ["harmonic147", "tone_to_silence", "noise", "vibrato250",
                 "silence", "chirp120_380"]
        return {"lead": [2, 3],
                "channels": [s[k]["channels"][0] for k in names]}

    @classmethod
    def cases_table(cls):
        """(key, geometry, {signal name: signal}, functions, dtypes)."""
        d = cls.signals(cls.N)
        s16 = cls.signals(8000)
        s44 = cls.signals(22050)
        s8 = cls.signals(4000)
        both, yin, pyin = ("yin", "pyin"), ("yin",), ("pyin",)
        f64, f32 = ("float64",), ("float32",)

        def pick(table, *names):
            return {k: table[k] for k in names}

        voices = ("harmonic147", "tone_to_silence", "noise", "vibrato250")
        return [
            ("default", cls.DEFAULT, d, both, f64),
            ("threshold0.05", dict(cls.DEFAULT, trough_threshold=0.05),
             pick(d, "harmonic147", "chirp120_380"), yin, f64),
            ("threshold0.25", dict(cls.DEFAULT, trough_threshold=0.25),
             pick(d, "harmonic147", "chirp120_380"), yin, f64),
            ("speech16k", cls.SPEECH,
             pick(s16, "chirp120_380", "vibrato250", "tone_to_silence"), both,
             f64),
            ("odd44k_uncentered", cls.ODD,
             {"tone150": cls.whole(["tone", 150.0, 0.5, 0.3], 22050),
              "harmonic147": s44["harmonic147"],
              "chirp120_380": s44["chirp120_380"]}, both, f64),
            ("edge8k", cls.EDGE, pick(s8, "offset_tone", "tone_to_silence"),
             both, f64),
            ("reflect8k", cls.REFLECT,
             pick(s8, "offset_tone", "tone_to_silence"), both, f64),
            ("reflect8k_hop700", cls.REFLECT_LONG_HOP,
             pick(s8, "offset_tone", "tone_to_silence"), yin, f64),
            ("constant0.5", dict(cls.DEFAULT, pad_value=0.5),
             pick(d, "harmonic147", "tone_to_silence"), both, f64),
            ("short", cls.DEFAULT,
             {"harmonic147_n300": cls.whole(["harmonic", 146.83, 6], 300),
              "harmonic147_n1025": cls.whole(["harmonic", 146.83, 6], 1025)},
             both, f64),
            ("short_reflect", dict(cls.DEFAULT, pad="reflect"),
             {"harmonic147_n1": cls.whole(["harmonic", 146.83, 6], 1),
              "harmonic147_n700": cls.whole(["harmonic", 146.83, 6], 700)},
             both, f64),
            ("batch", cls.DEFAULT, {"batch2x3": cls.batch(cls.N)}, both,
             ("float64", "float32")),
            ("default", cls.DEFAULT,
             pick(d, "harmonic147", "chirp120_380", "tone_to_silence"), both,
             f32),
            ("decoded", dict(cls.DEFAULT, fill="decoded"),
             pick(d, "tone_to_silence", "silence", "noise", "vibrato250"),
             pyin, f64),
            ("variant", dict(cls.DEFAULT, **cls.VARIANT), pick(d, *voices),
             pyin, f64),
            ("exact", dict(cls.DEFAULT, transition_min_prob=0.0),
             pick(d, *voices), pyin, f64),
            ("variant_prune1e-3",
             dict(cls.DEFAULT, **cls.VARIANT, transition_min_prob=1e-3),
             pick(d, *voices), pyin, f64),
        ]

    # -- signals ---------------------------------------------------------------

    @staticmethod
    def segment(recipe, n, sr):
        kind = recipe[0]
        t = np.arange(n) / sr
        two_pi = 2 * np.pi
        if kind == "silence":
            return np.zeros(n)
        if kind == "tone":
            f, amp, phase = recipe[1:]
            return amp * np.sin(two_pi * f * t + phase)
        if kind == "harmonic":
            f, partials = recipe[1:]
            y = 0
            for h in range(1, partials + 1):
                y = y + (0.6 / h) * np.sin(two_pi * (h * f) * t + 0.1 * h)
            return y
        if kind == "vibrato":
            fc, depth, rate = recipe[1:]
            return 0.5 * np.sin(two_pi * fc * t
                                + (depth / rate) * np.sin(two_pi * rate * t))
        if kind == "chirp":
            f0, f1 = recipe[1:]
            k = (f1 - f0) / (n / sr)
            return 0.5 * np.sin(two_pi * (f0 * t + (0.5 * k) * t * t))
        if kind == "noise":
            scale, seed = recipe[1:]
            return scale * StftVectorGenerator.lcg_signal(n, seed=seed)
        if kind == "offset":
            c, f, amp = recipe[1:]
            return c + amp * np.sin(two_pi * f * t + 0.3)
        raise ValueError(kind)

    @classmethod
    def signal(cls, spec, sr):
        rows = [np.concatenate([cls.segment(r, n, sr) for n, r in segments])
                for segments in spec["channels"]]
        y = np.stack(rows)
        return y.reshape(spec["lead"] + [y.shape[-1]]) if spec["lead"] else y[0]

    @staticmethod
    def quantize(y, dtype):
        return y if dtype == "float64" else y.astype(np.float32).astype(np.float64)

    # -- the reference ---------------------------------------------------------

    @staticmethod
    def librosa_args(g, function):
        frame_length = g.get("frame_length", 2048)
        args = {"fmin": g["fmin"], "fmax": g["fmax"], "sr": g["sample_rate"],
                "frame_length": frame_length,
                "hop_length": g.get("hop", frame_length // 4),
                "center": g.get("center", True),
                "pad_mode": g.get("pad", "constant")}
        if function == "yin":
            args["trough_threshold"] = g.get("trough_threshold", 0.1)
            return args
        for key in ("n_thresholds", "boltzmann_parameter", "resolution",
                    "max_transition_rate", "switch_prob", "no_trough_prob"):
            if key in g:
                args[key] = g[key]
        if "beta_parameters" in g:
            args["beta_parameters"] = tuple(g["beta_parameters"])
        fill = g.get("fill", "nan")
        args["fill_na"] = (np.nan if fill == "nan"
                           else None if fill == "decoded" else fill)
        tmp = g.get("transition_min_prob", 1e-4)
        args["transition_min_prob"] = None if tmp == 0.0 else tmp
        return args

    @staticmethod
    def by_hand(y, g, args):
        """The constant-v padding librosa has no argument for: the signal
        padded by hand, then analysed uncentered."""
        if "pad_value" not in g:
            return y, args
        border = args["frame_length"] // 2
        y = np.pad(y, [(0, 0)] * (y.ndim - 1) + [(border, border)],
                   mode="constant", constant_values=g["pad_value"])
        return y, dict(args, center=False)

    @classmethod
    def run(cls, function, y, g):
        y, args = cls.by_hand(y, g, cls.librosa_args(g, function))
        if function == "yin":
            return librosa.yin(y, **args)
        f0, voiced, probability = librosa.pyin(y, **args)
        return np.stack([f0, voiced.astype(np.float64), probability])

    # -- decision margins on librosa's own difference function ----------------

    @classmethod
    def cmnd(cls, y, g):
        y, args = cls.by_hand(y, g, cls.librosa_args(g, "yin"))
        W, hop, sr = args["frame_length"], args["hop_length"], args["sr"]
        if args["center"]:
            y = np.pad(y, [(0, 0)] * (y.ndim - 1) + [(W // 2, W // 2)],
                       mode=args["pad_mode"])
        frames = librosa.util.frame(y, frame_length=W, hop_length=hop)
        min_period = int(np.floor(sr / g["fmax"]))
        max_period = min(int(np.ceil(sr / g["fmin"])), W - 1)
        from librosa.core import pitch
        c = pitch._cumulative_mean_normalized_difference(
            frames, min_period, max_period)
        rows = np.moveaxis(c, -2, -1).reshape(-1, c.shape[-2])
        return rows, min_period

    @staticmethod
    def troughs(x):
        t = np.zeros(len(x), bool)
        t[1:-1] = (x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:])
        t[-1] = x[-1] < x[-2]
        t[0] = x[0] < x[1]
        return t

    @staticmethod
    def parabola(x, i):
        """The parabola branch's distance from its tie, and the shift."""
        if i == 0 or i == len(x) - 1:
            return np.inf, 0.0
        a = x[i + 1] + x[i - 1] - 2 * x[i]
        b = (x[i + 1] - x[i - 1]) / 2
        return abs(abs(b) - abs(a)), (0.0 if abs(b) >= abs(a) else -b / a)

    @classmethod
    def margin(cls, y, g, function):
        rows, min_period = cls.cmnd(y, g)
        sr = g["sample_rate"]
        worst = np.inf
        for x in rows:
            if not np.any(x):
                continue  # digital silence: exact on every side
            diffs = np.abs(np.diff(x))
            t = cls.troughs(x)
            if function == "yin":
                thr = g.get("trough_threshold", 0.1)
                below = np.flatnonzero(t & (x < thr))
                if len(below):
                    pick = below[0]
                    upto = pick + 1
                    m = abs(x[pick] - thr)
                else:
                    order = np.sort(x)
                    pick = int(np.argmin(x))
                    upto = len(x)
                    m = order[1] - order[0]
                for j in range(upto):
                    if x[j] < thr + 1e-3:
                        m = min(m, abs(x[j] - thr))
                        if j > 0:
                            m = min(m, diffs[j - 1])
                        if j < len(x) - 1:
                            m = min(m, diffs[j])
                m = min(m, cls.parabola(x, pick)[0])
            else:
                n_thresholds = g.get("n_thresholds", 100)
                thresholds = np.linspace(0, 1, n_thresholds + 1)[1:]
                nbps = int(np.ceil(1.0 / g.get("resolution", 0.1)))
                n_bins = int(np.floor(12 * nbps * np.log2(g["fmax"] / g["fmin"]))) + 1
                m = diffs.min()
                index = np.flatnonzero(t)
                if len(index):
                    h = x[index]
                    m = min(m, np.min(np.abs(h[:, None] - thresholds[None, :])))
                    if len(h) > 1:
                        order = np.sort(h)
                        m = min(m, order[1] - order[0])
                    for i in index:
                        gap, shift = cls.parabola(x, i)
                        b = 12 * nbps * np.log2((sr / ((min_period + i) + shift))
                                                / g["fmin"])
                        m = min(m, gap, abs(b - np.floor(b) - 0.5),
                                abs(b - (n_bins - 0.5)))
            worst = min(worst, m)
        return worst

    def stable(self, function, y, g, expected):
        wiggle = StftVectorGenerator.lcg_signal(y.size, seed=self.SEED + 7)
        again = self.run(function,
                         y * (1 + 2.0 ** -40 * wiggle.reshape(y.shape)), g)
        if function == "yin":
            return float(np.max(np.abs(again - expected) / expected)) < 1e-9
        return np.array_equal(again[:2], expected[:2], equal_nan=True)

    def cases(self):
        out = {"yin": [], "pyin": []}
        for key, geometry, signals, functions, dtypes in self.cases_table():
            for name, spec in signals.items():
                base = self.signal(spec, geometry["sample_rate"])
                for dtype in dtypes:
                    y = self.quantize(base, dtype)
                    for function in functions:
                        m = self.margin(y, geometry, function)
                        assert m > self.MARGIN, (key, name, function, dtype, m)
                        expected = self.run(function, y, geometry)
                        assert self.stable(function, y, geometry, expected), (
                            key, name, function, dtype)
                        params = dict(geometry)
                        params.update(spec)
                        params["dtype"] = dtype
                        out[function].append({
                            "name": f"{function}_{key}_{name}_{dtype}",
                            "params": params,
                            "shape": list(expected.shape),
                            "values": expected.ravel().tolist()})
        return out

    def generate(self):
        out = self.cases()
        write_suite(self.SUITE, "yin", out["yin"])
        write_suite(self.SUITE, "pyin", out["pyin"])
class TuningVectorGenerator:
    """Golden vectors for Soundml.piptrack, pitch_tuning, estimate_tuning and
    estimate_tuning_of_spectrogram (librosa.piptrack, pitch_tuning and
    estimate_tuning).

    Three files:

    - piptrack: librosa.piptrack(S=...) over |LCG| spectrograms under this
      suite's seed, reshaped in C order (reproduced bit-exactly on the OCaml
      side), and over STFT magnitude and power spectrograms of tones. The
      LCG cases cover the defaults, thresholds 0 and 0.5, a scalar ref (which
      librosa does not scale by the threshold), the band clamped to [0,
      Nyquist), a negative fmin (clamped), an empty band, a batched rank-4
      input, an odd n_fft passed explicitly (whose top bin can peak), the
      edges of a 16 kHz, 512-point grid (bin 5 at exactly fmin, bin 128 at
      fmax excluded), band edges where rfftfreq's k * (1 / (n * (1 / sr)))
      and k * sr / n fall on opposite sides (at 12345 Hz, and the Nyquist
      bin of a 30-point grid at 22050 Hz, which rfftfreq puts in the band),
      quantised plateaus (exact ties), two bins (the Nyquist bin excluded,
      and the top bin of a 3-point grid) and one frame. Results are stored
      sparse: one row [C-order flat index, pitch, magnitude] per peak, shape
      [peaks; 3].
    - pitch_tuning: librosa.pitch_tuning over frequency sets stored in the
      params: semitone ladders at mid-bin detunings, the 0.001 and 0.03
      resolutions (34 bins), 36, 24 and 7 bins per octave, both wraps at
      half a bin, a majority, an exact count tie (the lowest bin wins),
      non-positive entries, frequencies below 27.5 Hz, the empty set and
      log-uniform sets at several resolutions.
    - estimate_tuning: librosa.estimate_tuning(y=...) over tones at known
      detunings (A4 at 440 * 2 ** (c / 1200) for several c), a harmonic
      note, a four-tone mixture, noise, a tone in noise, the resolution,
      bins_per_octave, band, threshold and ref options, a stereo signal
      (pooled into one estimate), a 16 kHz geometry and silence; and
      estimate_tuning(S=...) over the power spectrogram, chroma_stft's
      default path. The partials' exact frequencies are stored in the params.

    float32 cases quantize the input (the spectrogram, or the audio) to
    float32 and compute the reference in float64. Every decision the STFT
    cases make is asserted to sit away from its tie: the threshold, local
    maximum and shift comparisons by at least 1e-7 of the frame maximum, the
    residuals by 1e-7 from every histogram edge, and the support and the
    estimate unchanged when the spectrogram is perturbed by 1e-12 relative
    noise or when the detections within 1e-9 of the median are all kept or
    all dropped -- many orders of magnitude above the rounding by which two
    FFTs or two libms differ.
    """

    SUITE = "tuning"

    SEED = 20261107

    SR = 22050

    # 0.5 * 0.7 ** h, the harmonic note's partial amplitudes
    HARMONIC_AMPS = [0.5 * 0.7**h for h in range(8)]

    @staticmethod
    def cents(f, c):
        return f * 2.0 ** (c / 1200.0)

    def lcg(self, n):
        return StftVectorGenerator.lcg_signal(n, seed=self.SEED)

    @staticmethod
    def quantize(y, dtype):
        return y if dtype == "float64" else y.astype(np.float32).astype(np.float64)

    def audio(self, spec):
        """spec: {"n", "sample_rate", "channels": [{"partials": [[f, a], ...],
        "noise": level}, ...]}; the partials are summed in order, each
        a * sin(2 * pi * f * t) with t = arange(n) / sample_rate, then the
        noise, level times the LCG stream."""
        n, sr = spec["n"], spec["sample_rate"]
        t = np.arange(n) / sr
        rows = []
        for channel in spec["channels"]:
            y = np.zeros(n)
            for f, a in channel["partials"]:
                y = y + a * np.sin(2 * np.pi * f * t)
            if channel["noise"] != 0.0:
                y = y + channel["noise"] * self.lcg(n)
            rows.append(y)
        return rows[0] if len(rows) == 1 else np.stack(rows)

    def tones(self, partials, n=None, sample_rate=None, noise=0.0):
        return {"n": self.SR if n is None else n,
                "sample_rate": self.SR if sample_rate is None else sample_rate,
                "channels": [{"partials": [[float(f), float(a)] for f, a in partials],
                              "noise": noise}]}

    @staticmethod
    def stft_magnitude(y, fft_size, hop):
        return np.abs(librosa.stft(y, n_fft=fft_size, hop_length=hop, window="hann",
                                   center=True, pad_mode="constant",
                                   dtype=np.complex128))

    # ---- robustness of the discrete decisions --------------------------------

    @staticmethod
    def decision_margin(S, threshold, ref):
        """The smallest margin, relative to its frame's maximum, of the
        threshold comparisons, the local-maximum comparisons between bins
        above the threshold, and the |b| >= |a| shift test at each peak."""
        frames = np.moveaxis(S, -2, -1).reshape(-1, S.shape[-2])
        worst = np.inf
        for col in frames:
            mx = col.max()
            if mx == 0.0:
                continue
            rv = threshold * mx if ref is None else abs(ref)
            worst = min(worst, float(np.min(np.abs(col - rv)) / mx))
            above = col > rv
            both = above[1:] & above[:-1]
            if both.any():
                worst = min(worst, float(np.min(np.abs(np.diff(col))[both]) / mx))
            z = np.where(above, col, 0.0)
            peak = np.zeros(len(col), dtype=bool)
            peak[1:-1] = (z[1:-1] > z[:-2]) & (z[1:-1] >= z[2:])
            for k in np.nonzero(peak)[0]:
                a = (col[k + 1] + col[k - 1]) - 2.0 * col[k]
                b = (col[k + 1] - col[k - 1]) / 2.0
                worst = min(worst, float((abs(a) - abs(b)) / mx))
        return worst

    @staticmethod
    def residuals(freqs, bins_per_octave):
        r = np.mod(bins_per_octave * librosa.hz_to_octs(freqs), 1.0)
        r[r >= 0.5] -= 1.0
        return r

    def edge_margin(self, freqs, resolution, bins_per_octave):
        freqs = np.asarray(freqs, dtype=np.float64)
        freqs = freqs[freqs > 0]
        if freqs.size == 0:
            return np.inf
        r = self.residuals(freqs, bins_per_octave)
        edges = np.linspace(-0.5, 0.5, int(np.ceil(1.0 / resolution)) + 1)
        return float(np.min(np.abs(r[:, None] - edges[None, :])))

    def assert_robust_piptrack(self, key, S, sr, fft_size, kwargs):
        margin = self.decision_margin(S, kwargs.get("threshold", 0.1), kwargs.get("ref"))
        assert margin > 1e-7, (key, margin)
        P, _ = librosa.piptrack(S=S, sr=sr, n_fft=fft_size, **kwargs)
        rng = np.random.default_rng(self.SEED)
        for _ in range(4):
            Sp = S * (1.0 + 1e-12 * rng.uniform(-1.0, 1.0, S.shape))
            Pp, _ = librosa.piptrack(S=Sp, sr=sr, n_fft=fft_size, **kwargs)
            assert np.array_equal(P > 0, Pp > 0), key
        return margin

    def assert_robust_estimate(self, key, S, sr, fft_size, resolution, bins_per_octave,
                               kwargs):
        """The estimate of S, asserted stable; returns (estimate, margins)."""
        def estimate(S):
            return librosa.estimate_tuning(S=S, sr=sr, n_fft=fft_size,
                                           resolution=resolution,
                                           bins_per_octave=bins_per_octave, **kwargs)

        expected = estimate(S)
        margin = self.assert_robust_piptrack(key, S, sr, fft_size, kwargs)
        P, M = librosa.piptrack(S=S, sr=sr, n_fft=fft_size, **kwargs)
        mask = P > 0
        edge = self.edge_margin(P[mask], resolution, bins_per_octave)
        assert edge > 1e-7, (key, edge)
        if mask.any():
            thr = np.median(M[mask])
            near = mask & (np.abs(M - thr) <= 1e-9 * thr)
            chosen = mask & (M >= thr)
            pt = lambda f: librosa.pitch_tuning(f, resolution=resolution,
                                                bins_per_octave=bins_per_octave)
            assert pt(P[chosen | near]) == expected, key
            if (chosen & ~near).any():
                assert pt(P[chosen & ~near]) == expected, key
        rng = np.random.default_rng(self.SEED + 1)
        for _ in range(4):
            Sp = S * (1.0 + 1e-12 * rng.uniform(-1.0, 1.0, S.shape))
            assert estimate(Sp) == expected, key
        return expected, margin, edge

    # ---- piptrack ------------------------------------------------------------

    # (case key, source, shape, sample rate, n_fft passed or None, kwargs, dtypes)
    PIPTRACK_LCG = [
        ("lcg_129x12_defaults", "lcg", [129, 12], 22050, None, {},
         ["float64", "float32"]),
        ("lcg_129x12_threshold0", "lcg", [129, 12], 22050, None,
         {"threshold": 0.0}, ["float64"]),
        ("lcg_129x12_threshold0.5", "lcg", [129, 12], 22050, None,
         {"threshold": 0.5}, ["float64", "float32"]),
        ("lcg_129x12_ref0.3_threshold0.9", "lcg", [129, 12], 22050, None,
         {"ref": 0.3, "threshold": 0.9}, ["float64"]),
        ("lcg_129x12_fmin0_fmax1e9", "lcg", [129, 12], 22050, None,
         {"fmin": 0.0, "fmax": 1e9}, ["float64"]),
        ("lcg_129x12_fmin-5_fmax2000", "lcg", [129, 12], 22050, None,
         {"fmin": -5.0, "fmax": 2000.0}, ["float64"]),
        ("lcg_129x12_empty_band", "lcg", [129, 12], 22050, None,
         {"fmin": 2000.0, "fmax": 1000.0}, ["float64"]),
        ("lcg_2x3x65x6_batched", "lcg", [2, 3, 65, 6], 8000, None, {},
         ["float64", "float32"]),
        ("lcg_128x12_fft255_fmax1e9", "lcg", [128, 12], 22050, 255,
         {"fmax": 1e9}, ["float64"]),
        ("lcg_257x8_sr16000_fmin156.25", "lcg", [257, 8], 16000, 512,
         {"fmin": 156.25}, ["float64"]),
        ("lcg_33x30_sr12345_rfftfreq_edges", "lcg", [33, 30], 12345, None,
         {"fmin": 192.890625, "fmax": 1928.90625}, ["float64"]),
        ("lcg_16x40_fft30_nyquist_in_band", "lcg", [16, 40], 22050, None,
         {"fmin": 0.0, "fmax": 1e9}, ["float64"]),
        ("plateau_65x9_threshold0.2", "plateau", [65, 9], 8000, None,
         {"threshold": 0.2}, ["float64"]),
        ("lcg_2x5_two_bins_nyquist", "lcg", [2, 5], 8000, None,
         {"fmin": 0.0, "fmax": 1e9}, ["float64"]),
        ("lcg_2x5_two_bins_fft3", "lcg", [2, 5], 8000, 3,
         {"fmin": 0.0, "fmax": 1e9}, ["float64"]),
        ("lcg_33x1_one_frame", "lcg", [33, 1], 8000, None, {"fmin": 0.0},
         ["float64"]),
    ]

    def lcg_spectrogram(self, source, shape):
        S = np.abs(self.lcg(int(np.prod(shape)))).reshape(shape)
        return np.floor(S * 4.0) / 4.0 if source == "plateau" else S

    @staticmethod
    def sparse(P, M):
        flat = np.nonzero(P.ravel() > 0)[0]
        assert np.array_equal(flat, np.nonzero(M.ravel() > 0)[0])
        rows = np.stack([flat.astype(np.float64), P.ravel()[flat], M.ravel()[flat]],
                        axis=1)
        return [len(flat), 3], rows.ravel().tolist()

    @staticmethod
    def kwargs_params(kwargs):
        return {{"ref": "reference"}.get(k, k): v for k, v in kwargs.items()}

    def piptrack_cases(self):
        cases = []
        for key, source, shape, sr, n_fft, kwargs, dtypes in self.PIPTRACK_LCG:
            for dtype in dtypes:
                S = self.quantize(self.lcg_spectrogram(source, shape), dtype)
                P, M = librosa.piptrack(S=S, sr=sr, n_fft=n_fft or 2048, **kwargs)
                if n_fft is not None:
                    assert n_fft // 2 + 1 == shape[-2], key
                assert P.dtype == np.float64
                # these cases pin rfftfreq's arithmetic: the band edges fall
                # between it and k * sr / n_fft, on bins that do peak
                if key == "lcg_33x30_sr12345_rfftfreq_edges":
                    assert 1 * (12345 / 64) == kwargs["fmin"]
                    assert 10 * (12345 / 64) == kwargs["fmax"]
                    assert not P[1].any() and P[10].any(), key
                    P0, _ = librosa.piptrack(S=S, sr=sr, fmin=0.0, fmax=1e9)
                    assert P0[1].any() and P0[10].any(), key
                if key == "lcg_16x40_fft30_nyquist_in_band":
                    assert P[15].any(), key
                shape_out, values = self.sparse(P, M)
                params = {"source": source, "input_shape": shape, "sample_rate": sr,
                          "dtype": dtype}
                if n_fft is not None:
                    params["fft_size"] = n_fft
                params.update(self.kwargs_params(kwargs))
                cases.append({"name": f"{key}_{dtype}", "params": params,
                              "shape": shape_out, "values": values})
        a4 = 440.0
        mixture4 = [(self.cents(a4, 17.0), 0.5), (self.cents(554.365, 17.0), 0.4),
                    (self.cents(659.255, -31.0), 0.2), (self.cents(196.0, 40.0), 0.1)]
        harmonic = [(self.cents(196.0 * (h + 1), 17.0), amp)
                    for h, amp in enumerate(self.HARMONIC_AMPS)]
        stft_cases = [
            ("tones_mixture4", self.tones(mixture4), 2048, 512, 1, {},
             ["float64", "float32"]),
            ("tones_harmonic_power", self.tones(harmonic), 2048, 512, 2, {},
             ["float64"]),
            ("tones_16k_fft512", self.tones([(4000.0, 0.5), (self.cents(a4, 27.0), 0.3)],
                                            n=16000, sample_rate=16000),
             512, 128, 1, {}, ["float64"]),
            ("tones_mixture4_ref20_fmin300", self.tones(mixture4, noise=0.02), 2048,
             512, 1, {"ref": 20.0, "fmin": 300.0}, ["float64"]),
        ]
        for key, spec, fft_size, hop, power, kwargs, dtypes in stft_cases:
            sr = spec["sample_rate"]
            S64 = self.stft_magnitude(self.audio(spec), fft_size, hop) ** power
            for dtype in dtypes:
                S = self.quantize(S64, dtype)
                margin = self.assert_robust_piptrack(key, S, sr, fft_size, kwargs)
                P, M = librosa.piptrack(S=S, sr=sr, n_fft=fft_size, **kwargs)
                shape_out, values = self.sparse(P, M)
                params = {"source": "stft", "signal": spec, "fft_size": fft_size,
                          "hop": hop, "power": power, "sample_rate": sr,
                          "dtype": dtype, "margin": margin}
                params.update(self.kwargs_params(kwargs))
                cases.append({"name": f"{key}_{dtype}", "params": params,
                              "shape": shape_out, "values": values})
        return cases

    # ---- pitch_tuning --------------------------------------------------------

    def pitch_tuning_cases(self):
        def ladder(f0, cents, count, bpo=12):
            return [f0 * 2.0 ** (i / bpo + cents / (100.0 * bpo)) for i in range(count)]

        lcg = (self.lcg(400) + 1.0) / 2.0  # [0, 1)
        spread = 27.5 * 2.0 ** (lcg * 11.0 - 2.0)

        def loguniform(resolution, bpo):
            """The log-uniform set without the frequencies whose residual lies
            within 1e-6 of a histogram edge."""
            r = self.residuals(spread, bpo)
            edges = np.linspace(-0.5, 0.5, int(np.ceil(1.0 / resolution)) + 1)
            far = np.min(np.abs(r[:, None] - edges[None, :]), axis=1) > 1e-6
            return spread[far].tolist()

        # (case key, frequencies, resolution, bins_per_octave, dtypes)
        cases_spec = [
            ("ladder_+25.5c", ladder(55.0, 25.5, 24), 0.01, 12, ["float64", "float32"]),
            ("ladder_-13.55c_res0.001", ladder(55.0, -13.55, 24), 0.001, 12, ["float64"]),
            ("ladder_+16.18c_res0.03", ladder(110.0, 100.0 * (22.5 / 34.0 - 0.5), 24),
             0.03, 12, ["float64"]),
            ("ladder_bpo36", ladder(55.0, 31.5, 72, bpo=36), 0.01, 36, ["float64"]),
            ("ladder_bpo24", ladder(55.0, -27.5, 48, bpo=24), 0.01, 24, ["float64"]),
            ("ladder_bpo7", ladder(55.0, 8.5, 14, bpo=7), 0.01, 7, ["float64"]),
            ("ladder_+49.5c", ladder(55.0, 49.5, 12), 0.01, 12, ["float64"]),
            ("ladder_-49.5c", ladder(55.0, -49.5, 12), 0.01, 12, ["float64"]),
            ("majority", ladder(220.0, 12.5, 7) + ladder(98.0, -30.5, 5), 0.01, 12,
             ["float64"]),
            ("count_tie", ladder(220.0, 30.5, 4) + ladder(98.0, -20.5, 4), 0.01, 12,
             ["float64"]),
            ("non_positive_skipped", [0.0, -440.0] + ladder(110.0, 7.5, 5) + [0.0, -3.0],
             0.01, 12, ["float64"]),
            ("below_27.5Hz", ladder(3.4375, -36.5, 30), 0.01, 12, ["float64"]),
            ("all_zeros", [0.0, 0.0, 0.0], 0.01, 12, ["float64"]),
            ("loguniform_res0.01", loguniform(0.01, 12), 0.01, 12, ["float64", "float32"]),
            ("loguniform_res0.001_bpo36", loguniform(0.001, 36), 0.001, 36, ["float64"]),
            ("loguniform_res0.03_bpo24", loguniform(0.03, 24), 0.03, 24, ["float64"]),
            ("loguniform_res0.3_bpo7", loguniform(0.3, 7), 0.3, 7, ["float64"]),
        ]
        cases = []
        for key, freqs, resolution, bpo, dtypes in cases_spec:
            for dtype in dtypes:
                f = self.quantize(np.asarray(freqs, dtype=np.float64), dtype)
                margin = self.edge_margin(f, resolution, bpo)
                assert margin > 1e-7, (key, margin)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    expected = librosa.pitch_tuning(f, resolution=resolution,
                                                    bins_per_octave=bpo)
                params = {"frequencies": f.tolist(), "resolution": resolution,
                          "bins_per_octave": bpo, "dtype": dtype}
                cases.append({"name": f"{key}_{dtype}", "params": params,
                              "shape": [1], "values": [float(expected)]})
        return cases

    # ---- estimate_tuning -----------------------------------------------------

    def estimate_tuning_cases(self):
        finite_or_none = lambda v: v if v is not None and np.isfinite(v) else None
        a4 = 440.0
        mixture4 = [(self.cents(a4, 17.0), 0.5), (self.cents(554.365, 17.0), 0.4),
                    (self.cents(659.255, -31.0), 0.2), (self.cents(196.0, 40.0), 0.1)]
        harmonic = [(self.cents(196.0 * (h + 1), 17.0), amp)
                    for h, amp in enumerate(self.HARMONIC_AMPS)]
        tone = lambda c, **kw: self.tones([(self.cents(a4, c), 0.5)], **kw)
        stereo = {"n": self.SR, "sample_rate": self.SR,
                  "channels": [{"partials": [[self.cents(a4, 23.0), 0.5]], "noise": 0.0},
                               {"partials": [[self.cents(329.628, -14.0), 0.3]],
                                "noise": 0.0}]}
        # (case key, signal, fft_size, hop, power (spectrogram face) or None
        #  (audio face), resolution, bins_per_octave, kwargs, dtypes)
        cases_spec = []
        for c in [-40.3, -23.0, -7.5, 0.0, 12.5, 31.7, 45.0]:
            cases_spec.append((f"a4_{c:+g}c", tone(c), 2048, 512, None, 0.01, 12, {},
                               ["float64", "float32"] if c in (-23.0, 31.7)
                               else ["float64"]))
        cases_spec += [
            ("harmonic_g3_+17c", self.tones(harmonic), 2048, 512, None, 0.01, 12, {},
             ["float64", "float32"]),
            ("mixture4", self.tones(mixture4), 2048, 512, None, 0.01, 12, {},
             ["float64"]),
            ("noise", self.tones([], noise=0.3), 2048, 512, None, 0.01, 12, {},
             ["float64", "float32"]),
            ("a4_-18.5c_in_noise", self.tones([(self.cents(a4, -18.5), 0.5)], noise=0.02),
             2048, 512, None, 0.01, 12, {}, ["float64"]),
            ("a4_+21c_res0.001", tone(21.0), 2048, 512, None, 0.001, 12, {},
             ["float64"]),
            ("a4_+21c_res0.03", tone(21.0), 2048, 512, None, 0.03, 12, {}, ["float64"]),
            ("a4_+21c_bpo36", tone(21.0), 2048, 512, None, 0.01, 36, {}, ["float64"]),
            ("mixture4_threshold0.5_fmin300_fmax1000", self.tones(mixture4), 2048, 512,
             None, 0.01, 12, {"threshold": 0.5, "fmin": 300.0, "fmax": 1000.0},
             ["float64"]),
            ("a4_-12c_in_noise_ref5", self.tones([(self.cents(a4, -12.0), 0.5)],
                                                 noise=0.02),
             2048, 512, None, 0.01, 12, {"ref": 5.0}, ["float64"]),
            ("stereo_pooled", stereo, 2048, 512, None, 0.01, 12, {}, ["float64"]),
            ("a4_+8.5c_fft1024_hop256", tone(8.5), 1024, 256, None, 0.01, 12, {},
             ["float64"]),
            ("sr16000_fft512", self.tones([(4000.0, 0.5), (self.cents(a4, 27.0), 0.3)],
                                          n=16000, sample_rate=16000),
             512, 128, None, 0.01, 12, {}, ["float64"]),
            ("zeros", self.tones([], n=4096), 2048, 512, None, 0.01, 12, {},
             ["float64"]),
            ("power_harmonic_g3_+17c", self.tones(harmonic), 2048, 512, 2, 0.01, 12, {},
             ["float64", "float32"]),
            ("power_mixture4_bpo12", self.tones(mixture4), 2048, 512, 2, 0.01, 12, {},
             ["float64"]),
        ]
        cases = []
        for key, spec, fft_size, hop, power, resolution, bpo, kwargs, dtypes in cases_spec:
            sr = spec["sample_rate"]
            for dtype in dtypes:
                # the audio face quantizes the audio, the spectrogram face the
                # spectrogram
                if power is None:
                    y = self.quantize(self.audio(spec), dtype)
                    S = self.stft_magnitude(y, fft_size, hop)
                else:
                    y = self.audio(spec)
                    S = self.quantize(self.stft_magnitude(y, fft_size, hop) ** power,
                                      dtype)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if power is None:
                        expected = librosa.estimate_tuning(
                            y=y, sr=sr, n_fft=fft_size, hop_length=hop,
                            resolution=resolution, bins_per_octave=bpo, **kwargs)
                    else:
                        expected = librosa.estimate_tuning(
                            S=S, sr=sr, resolution=resolution,
                            bins_per_octave=bpo, **kwargs)
                    if np.any(S > 0):
                        robust, margin, edge = self.assert_robust_estimate(
                            key, S, sr, fft_size, resolution, bpo, kwargs)
                        assert robust == expected, key
                    else:
                        margin, edge = None, None
                params = {"signal": spec, "fft_size": fft_size, "hop": hop,
                          "face": "audio" if power is None else "spectrogram",
                          "resolution": resolution, "bins_per_octave": bpo,
                          "sample_rate": sr, "dtype": dtype,
                          "margin": finite_or_none(margin),
                          "edge_margin": finite_or_none(edge)}
                if power is not None:
                    params["power"] = power
                params.update(self.kwargs_params(kwargs))
                cases.append({"name": f"{key}_{dtype}", "params": params,
                              "shape": [1], "values": [float(expected)]})
                print(f"  {key}_{dtype}: {expected!r} margin={margin} edge={edge}")
        return cases

    def generate(self):
        write_suite(self.SUITE, "piptrack", self.piptrack_cases())
        write_suite(self.SUITE, "pitch_tuning", self.pitch_tuning_cases())
        write_suite(self.SUITE, "estimate_tuning", self.estimate_tuning_cases())
class OnsetDetectVectorGenerator:
    """Golden vectors for Soundml.peak_pick (and so peak_pick_stage),
    onset_detect, onset_detect_of_envelope and onset_backtrack. Every
    expected value is a frame index, stored as a float, and the tests compare
    the lists exactly. Four files:

    - peak_pick: librosa.util.peak_pick, by the greedy method and both
      dynamic-programming ones, over hand-made signals (plateaus, all-equal
      signals, a peak at the first and at the last frame, spikes exactly wait
      and wait + 1 frames apart, negative values, NaN and infinities) and over
      LCG signals under this suite's seed (|v|, its cube, |v| rounded down to
      quarters -- full of exact ties -- and v itself) under ten parameter
      sets: librosa's defaults at 22050 Hz / hop 512 and 16000 Hz / hop 160,
      its docstring's, Böck et al.'s online and offline settings, windows
      holding the current frame alone, windows wider than the signal and a
      wait longer than them. SoundML reproduces the reference's float64
      arithmetic term for term, so ties need no margin here.

    - onset_backtrack: librosa.onset.onset_backtrack over hand-made, step
      (ties) and continuous energies, with sorted, unsorted, duplicated and
      out-of-range events, a NaN energy, and energies too short to hold an
      interior frame.

    - onset_detect_envelope: librosa.onset.onset_detect(onset_envelope=...)
      over stored envelopes -- the onset_strength envelope of the mixture
      below at both geometries, an LCG envelope at five (sample_rate, hop)
      geometries, one of which derives windows of no frame -- with and
      without normalisation, explicit windows, the dynamic-programming
      methods, backtracking on the envelope and on a stored RMS curve, and
      the early exits (constant, zero, non-finite, one-frame and empty
      envelopes). Exact arithmetic again.

    - onset_detect: onset_detect end to end from two seconds of audio
      rebuilt in OCaml from the recipes in `audio` (clicks, decaying tone
      bursts, noise bursts and a mixture of the three over an LCG floor), at
      22050 Hz / n_fft 2048 / hop 512 / 128 mels -- librosa's own analysis --
      and 16000 Hz / n_fft 512 / hop 160 / 64 mels, with the defaults and,
      on the mixture, a larger delta, a longer wait, no normalisation,
      backtracking on the envelope and on the RMS curve, and dp_count. The
      envelope is onset_strength with pad_mode="constant" (librosa 1.0's
      default) and the mel parameters passed explicitly at dtype=np.float64;
      at 22050 Hz the default cases are also asserted to equal librosa's own
      onset_detect(y=...), whose mel basis is float32. Here the two sides'
      envelopes differ by rounding, so every frame's picking decision is
      asserted to lie at least PICK_FLOOR (of the normalised scale, or of the
      envelope's maximum when not normalising) from flipping, and every
      comparison backtracking relies on at least ENVELOPE_TIE_FLOOR or
      RMS_TIE_FLOOR of the curve's maximum from a tie, unless it ties two
      exact zeros that are structural: the envelope's constructed prefix, or
      frames at which every log-mel band falls clearly.

    float32 cases quantize the input to float32 and take the reference on its
    float64 image.
    """

    SUITE = "onset_detect"

    SEED = 20261103

    @staticmethod
    def quantize(x, dtype):
        x = np.asarray(x, dtype=np.float64)
        return x if dtype == "float64" else x.astype(np.float32).astype(np.float64)

    @staticmethod
    def case(name, params, frames):
        frames = np.asarray(frames)
        assert frames.ndim == 1, name
        return {"name": name, "params": params, "shape": [len(frames)],
                "values": [float(v) for v in frames]}

    def lcg(self, n, offset=0):
        return StftVectorGenerator.lcg_signal(n, seed=self.SEED + offset)

    @staticmethod
    def windows(pre_max, post_max, pre_avg, post_avg, delta, wait):
        return {"pre_max": pre_max, "post_max": post_max, "pre_avg": pre_avg,
                "post_avg": post_avg, "delta": delta, "wait": wait}

    # {1 Peak picking}

    METHODS = ["greedy", "dp_count", "dp_value"]

    # (key, pre_max, post_max, pre_avg, post_avg, delta, wait)
    WINDOWS = [
        ("unit", 1, 1, 1, 1, 0.0, 0),
        ("librosa_22050_512", 1, 1, 4, 5, 0.07, 1),
        ("librosa_16000_160", 3, 1, 10, 11, 0.07, 3),
        ("docstring", 3, 3, 3, 5, 0.5, 10),
        ("self_only", 0, 1, 0, 1, 0.0, 0),
        ("lookahead", 2, 4, 5, 2, 0.05, 0),
        ("long_wait", 10, 10, 10, 10, 0.0, 20),
        ("boeck_online", 3, 1, 10, 1, 0.1, 3),
        ("boeck_offline", 3, 4, 10, 2, 0.1, 0),
        ("wider_than_signal", 500, 500, 500, 500, 0.0, 0),
    ]

    SPIKES = [1.0 if i in (0, 3, 7, 10, 14, 17) else 0.0 for i in range(20)]

    # (key, signal, (pre_max, post_max, pre_avg, post_avg, delta, wait))
    HAND = [
        ("plateaus", [0, 1, 1, 0, 1, 1, 1, 0], (2, 3, 1, 1, 0.0, 0)),
        ("plateaus_wait1", [0, 1, 1, 0, 1, 1, 1, 0], (2, 3, 1, 1, 0.0, 1)),
        ("equal", [0.5] * 8, (1, 1, 1, 1, 0.0, 0)),
        ("equal_wait1", [0.5] * 8, (1, 1, 1, 1, 0.0, 1)),
        ("equal_wait2", [0.5] * 7, (1, 1, 1, 1, 0.0, 2)),
        ("first_frame", [5, 1, 1, 1], (1, 1, 1, 1, 0.0, 0)),
        ("last_frame", [0, 0.25, 0.5, 0.75, 1, 1.5, 2.5], (2, 2, 3, 3, 0.0, 0)),
        ("spikes_wait3", SPIKES, (1, 1, 1, 1, 0.0, 3)),
        ("spikes_wait4", SPIKES, (1, 1, 1, 1, 0.0, 4)),
        ("lookahead_wait2", [3, 0, 2, 0, 1, 0, 5, 0], (1, 2, 1, 1, 0.0, 2)),
        ("negative", [-3, -1, -2, -5, -4, -0.5, -6], (1, 1, 1, 1, 0.0, 0)),
        ("one_frame", [1.0], (1, 1, 1, 1, 0.0, 0)),
        ("one_frame_delta", [1.0], (1, 1, 1, 1, 0.1, 0)),
        ("wait_past_end", [1, 0, 2, 0, 3, 0, 4], (0, 1, 0, 1, 0.0, 100)),
        ("nan", [0, 1, np.nan, 0, 2, 0, 3, 1, 0, 1, 0], (1, 1, 2, 2, 0.0, 0)),
        ("nan_first", [np.nan, 0, 1, 0, 2, 0, 1, 0], (1, 1, 1, 1, 0.0, 0)),
        ("nan_lookahead", [0, 2, 0, 1, np.nan, 3, 0, 1, 0], (1, 3, 1, 1, 0.0, 0)),
        ("inf", [0, np.inf, 0, 1, 0, 2, 0], (1, 1, 1, 1, 0.0, 0)),
        ("inf_both", [0, np.inf, 0, -np.inf, 1, 0, 2, 0, 1], (1, 1, 2, 2, 0.0, 0)),
        ("neg_inf", [-np.inf, 0, 1, -np.inf, 2, 0], (1, 1, 1, 1, 0.0, 1)),
    ]

    PEAK_LENGTH = 160

    def peak_forms(self):
        """The LCG signals, rebuilt in OCaml by the same IEEE operations."""
        v = self.lcg(self.PEAK_LENGTH, 1)
        a = np.abs(v)
        return {"abs": a, "cube": (a * a) * a,
                "quarters": np.floor(4.0 * a) / 4.0, "signed": v}

    def peak_cases(self):
        cases = []
        differs = False
        for key, signal, w in self.HAND:
            x = np.asarray(signal, dtype=np.float64)
            p = self.windows(*w)
            greedy = librosa.util.peak_pick(x, **p)
            for method in self.METHODS:
                expected = librosa.util.peak_pick(x, method=method, **p)
                differs = differs or not np.array_equal(expected, greedy)
                cases.append(self.case(
                    f"{key}_{method}",
                    {"input": x.tolist(), **p, "method": method,
                     "dtype": "float64"},
                    expected))
        for form, x in self.peak_forms().items():
            methods = self.METHODS if form in ("abs", "quarters") else ["greedy"]
            dtypes = ["float64", "float32"] if form in ("abs", "cube") else ["float64"]
            for key, *w in self.WINDOWS:
                p = self.windows(*w)
                for dtype in dtypes:
                    xq = self.quantize(x, dtype)
                    greedy = librosa.util.peak_pick(xq, **p)
                    if key == "self_only":
                        assert len(greedy) == len(x), form
                    for method in methods:
                        expected = librosa.util.peak_pick(xq, method=method, **p)
                        differs = differs or not np.array_equal(expected, greedy)
                        cases.append(self.case(
                            f"{form}_{key}_{method}_{dtype}",
                            {"form": form, "length": len(x), **p,
                             "method": method, "dtype": dtype},
                            expected))
        assert differs, "no dynamic-programming case differs from greedy"
        return cases

    # {1 Backtracking}

    def backtrack_cases(self):
        cases = []

        def add(key, energy, events, dtype="float64"):
            e = self.quantize(energy, dtype)
            ev = np.asarray(events, dtype=int)
            expected = librosa.onset.onset_backtrack(ev, e)
            cases.append(self.case(
                key, {"energy": e.tolist(), "events": ev.tolist(),
                      "dtype": dtype}, expected))

        valleys = [5, 3, 4, 2, 2, 6, 1, 0, 0, 3, 7]
        add("valleys", valleys, list(range(11)) + [12])
        add("valleys_unsorted", valleys, [10, 2, 5])
        add("valleys_duplicates", valleys, [7, 2, 7, 4, 4, 0])
        add("flat_valley", [3, 1, 1, 1, 2, 0, 0, 4], list(range(8)))
        steps = np.floor(4.0 * np.abs(self.lcg(50, 2)))
        add("steps", steps, list(range(0, 50, 3)))
        add("steps_unsorted_past_end", steps, [49, 3, 60, 17, 17, 0, 33, 51])
        smooth = np.abs(self.lcg(60, 3))
        events = [int(70 * abs(v)) for v in self.lcg(15, 4)]
        add("continuous", smooth, events)
        add("continuous_float32", smooth, events, "float32")
        add("nan", [3, np.nan, 1, 2, 0, np.nan, 4, 1, 2], list(range(10)))
        add("length2", [1.0, 0.0], [0, 1, 5])
        add("length1", [1.0], [0, 3])
        add("length0", [], [0, 2])
        return cases

    # {1 Audio}

    SECONDS = 2

    # (sample_rate, fft_size, hop, n_mels)
    GEOMETRIES = [(22050, 2048, 512, 128), (16000, 512, 160, 64)]

    KINDS = ["clicks", "tones", "noise", "mix"]

    @staticmethod
    def fade(n, k):
        """Unit gain with a raised-cosine fade over the last k samples."""
        g = np.ones(n)
        k = min(k, n)
        g[n - k:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(1, k + 1) / k))
        return g

    @staticmethod
    def clicks(n, sr, amps, period, first):
        """Unit impulses of the amplitudes amps in turn, every period seconds
        from first."""
        y = np.zeros(n)
        p, step, k = int(first * sr), int(period * sr), 0
        while p < n:
            y[p] += amps[k % len(amps)]
            p += step
            k += 1
        return y

    def bursts(self, n, sr, kind, starts, dur, amp=0.5):
        """Bursts of dur seconds at the starts (seconds): an instant attack,
        an exp(-t / (dur / 3)) decay and a 20 ms raised-cosine fade-out over
        a sine at 220 * 2 ** ((k % 4) / 4) Hz for the k-th tone, or over the
        LCG stream of offset 1 for noise."""
        y = np.zeros(n)
        noise = self.lcg(n, 1)
        t = np.arange(n) / sr
        for k, s in enumerate(starts):
            a = int(s * sr)
            b = min(n, a + int(dur * sr))
            tt = t[a:b] - t[a]
            env = np.exp(-tt / (dur / 3)) * self.fade(b - a, int(0.02 * sr))
            if kind == "tone":
                src = np.sin(2 * np.pi * (220.0 * 2 ** ((k % 4) / 4)) * tt)
            else:
                src = noise[a:b]
            y[a:b] += (amp * env) * src
        return y

    def audio(self, kind, sr):
        """Two seconds of the signal kind at sr over a 1e-3 LCG floor."""
        n = self.SECONDS * sr
        floor = 1e-3 * self.lcg(n)
        if kind == "clicks":
            return floor + self.clicks(n, sr, (0.8, 0.25), 0.25, 0.1)
        if kind == "tones":
            return floor + self.bursts(n, sr, "tone", [0.1, 0.45, 0.8, 1.2, 1.55], 0.25)
        if kind == "noise":
            return floor + self.bursts(n, sr, "noise", [0.12, 0.5, 0.9, 1.33, 1.7], 0.15)
        assert kind == "mix"
        return (floor + self.bursts(n, sr, "tone", [0.15, 0.62, 1.1, 1.55], 0.35)
                + 0.6 * self.bursts(n, sr, "noise", [0.4, 0.95, 1.32], 0.1)
                + self.clicks(n, sr, (0.3, 0.1), 0.5, 0.27))

    @staticmethod
    def envelope(y, sr, fft_size, hop, n_mels):
        return librosa.onset.onset_strength(
            y=y, sr=sr, n_fft=fft_size, hop_length=hop, center=True,
            pad_mode="constant",
            **MelVectorGenerator.mel_kwargs(n_mels, 0.0, 0.5 * sr, "slaney", "slaney"))

    @staticmethod
    def rms(y, fft_size, hop):
        return librosa.feature.rms(y=y, frame_length=fft_size, hop_length=hop,
                                   center=True, pad_mode="constant",
                                   dtype=np.float64)[0]

    @staticmethod
    def normalise(e):
        e = e - e.min()
        return e / (e.max() + np.finfo(np.float64).tiny)

    @staticmethod
    def defaults(sr, hop):
        """onset_detect's window defaults, as it derives them."""
        return {"pre_max": int(0.03 * sr // hop), "post_max": 1,
                "pre_avg": int(0.10 * sr // hop),
                "post_avg": int(0.10 * sr // hop + 1),
                "wait": int(0.03 * sr // hop), "delta": 0.07}

    @staticmethod
    def pick_margin(z, pre_max, post_max, pre_avg, post_avg, delta, wait):
        """The smallest distance, over every frame, from flipping the frame's
        candidate test (the wait rule is integer arithmetic): a passing frame
        by the smaller of its two gaps, a failing one by its larger failing
        gap. A tie of the maximum that the threshold does not settle has no
        margin."""
        n = len(z)
        worst = np.inf
        for i in range(n):
            lo, hi = max(0, i - pre_max), min(n, i + post_max)
            others = [z[j] for j in range(lo, hi) if j != i]
            top = z[i] - max(others) if others else np.inf
            lo, hi = max(0, i - pre_avg), min(n, i + post_avg)
            threshold = z[i] - (np.mean(z[lo:hi]) + delta)
            if top >= 0 and threshold >= 0:
                m = min(top, threshold)
            else:
                m = max(-top if top < 0 else 0.0, -threshold if threshold < 0 else 0.0)
            worst = min(worst, m)
        return worst

    def structural_zeros(self, y, sr, fft_size, hop, n_mels):
        """The envelope frames that are exactly zero on both sides whatever
        their rounding: the constructed prefix (lag + n_fft // (2 hop)
        frames), and the frames at which every log-mel band falls by more
        than 0.01 dB, a hundred times the float32 analysis's error."""
        shift = fft_size // (2 * hop)
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=fft_size, hop_length=hop, center=True,
            pad_mode="constant",
            **MelVectorGenerator.mel_kwargs(n_mels, 0.0, 0.5 * sr, "slaney", "slaney"))
        rise = np.max(np.diff(librosa.power_to_db(S), axis=-1), axis=0)
        return {t for t in range(S.shape[-1])
                if t < 1 + shift or rise[t - 1 - shift] < -1e-2}

    @staticmethod
    def backtrack_margin(e, events, structural):
        """The smallest distance from a tie of the comparisons that decide
        where each event backtracks to -- whether the minimum it reaches is
        one, and whether each later frame up to the event is not -- except a
        tie of two structural exact zeros."""
        m = len(e)

        def is_minimum(i):
            return e[i] <= e[i - 1] and e[i] < e[i + 1]

        worst = np.inf
        for k in events:
            last = min(k, m - 2)
            j = max([0] + [i for i in range(1, last + 1) if is_minimum(i)])
            for i in range(max(j, 1), last + 1):
                for a, b in ((i, i - 1), (i, i + 1)):
                    if e[a] == 0.0 and e[b] == 0.0 and {a, b} <= structural:
                        continue
                    worst = min(worst, abs(e[a] - e[b]))
        return worst

    # The margins, as fractions of the curve's maximum (the normalised scale
    # when normalising). SoundML's envelope of these signals agrees with
    # librosa's within 3e-15 of its maximum from float64 audio and 3e-7 from
    # float32 audio, and its RMS curve within 3e-15 (soundml/test/onset_detect,
    # "agreement"). The smallest margins measured are 1.2e-3 for a pick (the
    # mixture at 22050 Hz), 1.5e-4 for an envelope comparison and 1.7e-6 for
    # an RMS one, each floor below leaving more than two orders of magnitude
    # over the agreement it must cover.
    PICK_FLOOR = 1e-3

    ENVELOPE_TIE_FLOOR = 1e-4

    RMS_TIE_FLOOR = 1e-6

    # (variant key, onset_detect options, normalize, backtrack)
    AUDIO_VARIANTS = [
        ("delta0.2", {"delta": 0.2}, True, "none"),
        ("wait8", {"wait": 8}, True, "none"),
        ("raw_delta1", {"delta": 1.0}, False, "none"),
        ("backtrack_envelope", {}, True, "envelope"),
        ("backtrack_rms", {}, True, "energy"),
        ("dp_count_wait4", {"method": "dp_count", "wait": 4}, True, "none"),
    ]

    @staticmethod
    def detect(env, sr, hop, options, normalize, backtrack, energy=None):
        return librosa.onset.onset_detect(
            onset_envelope=env, sr=sr, hop_length=hop, normalize=normalize,
            backtrack=backtrack != "none",
            energy=energy if backtrack == "energy" else None, **options)

    def audio_cases(self):
        cases = []
        for sr, fft_size, hop, n_mels in self.GEOMETRIES:
            for kind in self.KINDS:
                y = self.audio(kind, sr)
                for dtype in ["float64", "float32"]:
                    yq = self.quantize(y, dtype)
                    env = self.envelope(yq, sr, fft_size, hop, n_mels)
                    rms = self.rms(yq, fft_size, hop)
                    structural = self.structural_zeros(yq, sr, fft_size, hop, n_mels)
                    variants = [("defaults", {}, True, "none")]
                    if kind == "mix":
                        variants += [v for v in self.AUDIO_VARIANTS
                                     if v[3] != "energy" or dtype == "float64"]
                    for variant, options, normalize, backtrack in variants:
                        onsets = self.detect(env, sr, hop, options, normalize,
                                             backtrack, rms)
                        assert len(onsets) > 0, (sr, kind, dtype, variant)
                        z = self.normalise(env) if normalize else env
                        scale = 1.0 if normalize else float(np.max(env))
                        p = self.defaults(sr, hop)
                        p.update({k: v for k, v in options.items() if k != "method"})
                        margin = self.pick_margin(z, **p) / scale
                        assert margin >= self.PICK_FLOOR, (sr, kind, dtype, variant, margin)
                        if backtrack != "none":
                            picked = librosa.util.peak_pick(z, **p)
                            if backtrack == "envelope":
                                curve, floor, at = z, self.ENVELOPE_TIE_FLOOR, structural
                            else:
                                curve, floor, at = rms, self.RMS_TIE_FLOOR, set()
                            tie = self.backtrack_margin(curve, picked, at) / np.max(curve)
                            assert tie >= floor, (sr, kind, dtype, variant, tie)
                        if variant == "defaults" and fft_size == 2048:
                            own = librosa.onset.onset_detect(y=yq, sr=sr, hop_length=hop)
                            assert np.array_equal(own, onsets), (sr, kind, dtype)
                        params = {"signal": kind, "sample_rate": sr,
                                  "fft_size": fft_size, "hop": hop,
                                  "n_mels": n_mels, "normalize": normalize,
                                  "backtrack": backtrack, **options,
                                  "dtype": dtype}
                        cases.append(self.case(
                            f"{kind}_{sr}_{variant}_{dtype}", params, onsets))
        return cases

    # {1 Envelopes}

    # (variant key, onset_detect options, normalize, backtrack)
    ENVELOPE_VARIANTS = [
        ("defaults", {}, True, "none"),
        ("delta0.2", {"delta": 0.2}, True, "none"),
        ("wait8", {"wait": 8}, True, "none"),
        ("raw_delta1", {"delta": 1.0}, False, "none"),
        ("backtrack_envelope", {}, True, "envelope"),
        ("raw_backtrack_envelope", {"delta": 1.0}, False, "envelope"),
        ("backtrack_rms", {}, True, "energy"),
        ("dp_count_wait4", {"method": "dp_count", "wait": 4}, True, "none"),
        ("dp_value_wait4", {"method": "dp_value", "wait": 4}, True, "none"),
        ("docstring", {"pre_max": 3, "post_max": 3, "pre_avg": 3,
                       "post_avg": 5, "delta": 0.5, "wait": 10}, True, "none"),
    ]

    # (sample_rate, hop): the window defaults of each, the last all empty
    ENVELOPE_RATES = [(22050, 512), (16000, 160), (44100, 441), (8000, 128),
                      (8000, 2400)]

    ENVELOPE_LENGTH = 120

    def envelope_cases(self):
        cases = []

        def add(name, env, sr, hop, options, normalize, backtrack, dtype,
                energy=None, stored=True, extra=None):
            e = self.quantize(env, dtype)
            onsets = self.detect(e, sr, hop, options, normalize, backtrack, energy)
            params = dict(extra or {})
            if stored:
                params["envelope"] = e.tolist()
            if backtrack == "energy":
                params["energy"] = np.asarray(energy).tolist()
            params.update({"sample_rate": sr, "hop": hop, "normalize": normalize,
                           "backtrack": backtrack, **options, "dtype": dtype})
            cases.append(self.case(name, params, onsets))
            return onsets

        for sr, fft_size, hop, n_mels in self.GEOMETRIES:
            y = self.audio("mix", sr)
            env = self.envelope(y, sr, fft_size, hop, n_mels)
            rms = self.rms(y, fft_size, hop)
            for variant, options, normalize, backtrack in self.ENVELOPE_VARIANTS:
                dtypes = (["float64", "float32"]
                          if variant in ("defaults", "backtrack_envelope", "dp_count_wait4")
                          else ["float64"])
                for dtype in dtypes:
                    add(f"mix_{sr}_{variant}_{dtype}", env, sr, hop, options,
                        normalize, backtrack, dtype, energy=rms)
        a = np.abs(self.lcg(self.ENVELOPE_LENGTH, 5))
        spiky = ((a * a) * a) * 10.0
        for sr, hop in self.ENVELOPE_RATES:
            for variant, options, normalize, backtrack in [
                    ("defaults", {}, True, "none"),
                    ("raw_delta0.5", {"delta": 0.5}, False, "none"),
                    ("backtrack_envelope", {}, True, "envelope")]:
                if hop == 2400 and backtrack != "none":
                    # no onset to backtrack: librosa raises there
                    continue
                for dtype in ["float64", "float32"]:
                    onsets = add(f"spiky_{sr}_{hop}_{variant}_{dtype}", spiky, sr,
                                 hop, options, normalize, backtrack, dtype,
                                 stored=False,
                                 extra={"form": "spiky",
                                        "length": self.ENVELOPE_LENGTH})
                    if hop == 2400:
                        assert len(onsets) == 0
        # (key, envelope, normalize, options): the early exits and the
        # shortest envelopes, at 22050 Hz / hop 512
        for key, env, normalize, options in [
                ("constant", [2.5] * 50, True, {}),
                ("constant_raw", [2.5] * 50, False, {}),
                ("constant_raw_delta0", [2.5] * 50, False, {"delta": 0.0}),
                ("zeros_raw_delta0", [0.0] * 50, False, {"delta": 0.0}),
                ("one_frame", [1.0], True, {}),
                ("one_frame_raw_delta0", [1.0], False, {"delta": 0.0}),
                ("two_frames", [0.25, 1.0], True, {}),
                ("negative_raw_delta0", [-3.0, -1.0, -2.0, -5.0, -4.0, -0.5, -6.0],
                 False, {"delta": 0.0}),
                ("nan", [0.0, 1.0, np.nan, 0.5, 2.0, 0.0], True, {}),
                ("nan_raw", [0.0, 1.0, np.nan, 0.5, 2.0, 0.0], False, {}),
                ("inf", [0.0, 1.0, np.inf, 0.5, 2.0, 0.0], True, {}),
                ("neg_inf_raw", [0.0, 1.0, -np.inf, 0.5, 2.0, 0.0], False, {}),
                ("empty_raw", [], False, {})]:
            onsets = add(key, env, 22050, 512, options, normalize, "none", "float64")
            if key in ("constant", "constant_raw", "zeros_raw_delta0", "one_frame",
                       "nan", "nan_raw", "inf", "neg_inf_raw", "empty_raw"):
                assert len(onsets) == 0, key
        return cases

    def generate(self):
        write_suite(self.SUITE, "peak_pick", self.peak_cases())
        write_suite(self.SUITE, "onset_backtrack", self.backtrack_cases())
        write_suite(self.SUITE, "onset_detect_envelope", self.envelope_cases())
        write_suite(self.SUITE, "onset_detect", self.audio_cases())
class RhythmVectorGenerator:
    """Golden vectors for the tempo and beat features: librosa.feature.tempogram,
    fourier_tempogram and tempo, librosa.beat.beat_track and plp, and
    librosa.tempo_frequencies / fourier_tempo_frequencies.

    Inputs are onset envelopes given directly -- formula envelopes (a floor of
    |LCG values| under this suite's seed with spikes added at given frames,
    the fourth power of |LCG values|, zeros, negated spikes) and the
    envelopes librosa.onset.onset_strength computes from click tracks -- and
    the click tracks themselves, whose envelope the OCaml side computes with
    Soundml.onset_strength. A click is a 220-sample burst of the LCG stream
    under a ramp, scale (1 - i / 220), at integer sample positions stored in
    the case; a track may add LCG noise. Scenarios: steady clicks at 90, 120,
    128 and 150 BPM, 110 BPM in noise, a change from 100 to 140 BPM, a swung
    rhythm, noise alone and a 16 kHz track at a hop of 256. The click
    envelopes take onset_strength's mean over mel bands with every analysis
    argument explicit (n_fft 2048, constant zero padding, 128 Slaney mel
    bands in float64); librosa's beat_track and plp take the median when
    given audio, which the library does not offer, so the references pass
    onset_envelope= throughout. envelopes.json records each click envelope,
    which the envelope-given cases read back.

    Discrete decisions are asserted away from ties: every tempo's two best
    scores (every frame's under aggregate=None), the first-beat, last-beat
    and trim thresholds and every predecessor choice along each beat path,
    every plp frame's peak and the sign of its real part. Predecessors may
    tie exactly where both scores extend one cumulative score through the
    same penalty, which is the same bits on either side; the beats are also
    asserted unmoved under relative perturbations of the envelope (1e-9, 1e-7
    for audio cases, 1e-3 for float32 audio) that keep its zeros. A
    pure-python replica of the beat tracker reproduces librosa exactly on
    every case (asserted) and measures those margins; it also asserts that
    the first_spike case pins librosa's exclusion of the envelope's first
    value, and the bpm 123 case its half-to-even rounding of half a beat.
    The variant cases named in TEMPO_PINS and BEAT_PINS are asserted to
    change the default result. The tempograms are stored at a few frames
    each (both ramps and the interior) or whole, as the "columns" parameter
    lists. float32 cases quantize the envelope (or the audio) to float32 and
    compute the reference in float64; plp's "unit" case is a numpy
    reimplementation of the method with unit-modulus kernels, not librosa.
    """

    SUITE = "rhythm"

    SEED = 20261107

    SR = 22050

    HOP = 512

    CLICK_LEN = 220

    # the variants asserted to change the default result
    TEMPO_PINS = {"start60", "start200", "std2", "ac4", "max100", "median"}

    BEAT_PINS = {"notrim", "tight400", "tight680", "start80"}

    def lcg(self, n, seed):
        return StftVectorGenerator.lcg_signal(n, seed=seed)

    @staticmethod
    def quantize(y, dtype):
        return y if dtype == "float64" else y.astype(np.float32).astype(np.float64)

    # {1 Inputs}

    def envelope(self, spec):
        """A formula envelope: zeros, a floor of |LCG| or the fourth power of
        |LCG|, spikes added one at a time in order, optionally negated."""
        n = spec["n"]
        if spec["kind"] == "zeros":
            o = np.zeros(n)
        elif spec["kind"] == "abs":
            o = spec["floor"] * np.abs(self.lcg(n, spec["seed"]))
        elif spec["kind"] == "pow4":
            a = np.abs(self.lcg(n, spec["seed"]))
            s = a * a
            o = s * s
        else:
            raise ValueError(spec["kind"])
        for frame, height in spec.get("spikes", []):
            o[frame] += height
        if spec.get("negate", False):
            o = -o
        return o

    def spikes(self, first, period, n, height=1.0):
        return [[f, height] for f in range(first, n, period)]

    def audio(self, spec):
        n, length = spec["length"], self.CLICK_LEN
        burst = self.lcg(length, spec["seed"])
        ramp = 1.0 - np.arange(length) / length
        y = np.zeros(n)
        for p in spec["positions"]:
            m = min(length, n - p)
            y[p:p + m] += spec["scale"] * ramp[:m] * burst[:m]
        if spec["noise"] > 0:
            y = y + spec["noise"] * self.lcg(n, spec["seed"] + 1)
        return y

    @staticmethod
    def steady(bpm, sr, n, offset=0.1):
        period = 60.0 / bpm
        out, t = [], offset
        while int(round(t * sr)) < n:
            out.append(int(round(t * sr)))
            t += period
        return out

    def scenarios(self):
        sr = self.SR
        n = 12 * sr
        out = {}

        def track(key, positions, seed, noise=0.0, sample_rate=sr, length=n,
                  hop=self.HOP):
            out[key] = {"sample_rate": sample_rate, "hop": hop,
                        "length": length, "positions": positions,
                        "seed": seed, "scale": 0.8, "noise": noise}

        for i, bpm in enumerate([90.0, 120.0, 128.0, 150.0]):
            track(f"clicks{bpm:g}", self.steady(bpm, sr, n), self.SEED + i)
        track("clicks110_noise", self.steady(110.0, sr, n), self.SEED + 10,
              noise=0.05)
        first = self.steady(100.0, sr, n // 2)
        t0 = first[-1] / sr + 60.0 / 140.0
        second = [int(round((t0 + k * 60.0 / 140.0) * sr)) for k in range(100)]
        track("change100to140", first + [p for p in second if p < n],
              self.SEED + 11)
        swing, t = [], 0.1
        while int(round(t * sr)) < n:
            swing.append(int(round(t * sr)))
            late = int(round((t + 1.0 / 3.0) * sr))
            if late < n:
                swing.append(late)
            t += 0.5
        track("swing120", swing, self.SEED + 12)
        track("noise", [], self.SEED + 13, noise=0.3)
        sr16 = 16000
        track("clicks100_16k", self.steady(100.0, sr16, 10 * sr16),
              self.SEED + 14, sample_rate=sr16, length=10 * sr16, hop=256)
        return out

    def onset(self, y, sr, hop):
        return librosa.onset.onset_strength(
            y=y, sr=sr, n_fft=2048, hop_length=hop, center=True,
            pad_mode="constant",
            **MelVectorGenerator.mel_kwargs(128, 0.0, 0.5 * sr, "slaney",
                                            "slaney"))

    def formula_envelopes(self):
        s = self.SEED
        spiky = {"kind": "abs", "n": 600, "floor": 0.05, "seed": s + 20,
                 "spikes": self.spikes(3, 21, 600)}
        return {
            "spiky_p21": spiky,
            "spiky_p25": {"kind": "abs", "n": 600, "floor": 0.05,
                          "seed": s + 22, "spikes": self.spikes(7, 25, 600)},
            "spiky_p10": {"kind": "abs", "n": 300, "floor": 0.05,
                          "seed": s + 28, "spikes": self.spikes(4, 10, 300)},
            "pow4": {"kind": "pow4", "n": 600, "seed": s + 21},
            "first_spike": {"kind": "abs", "n": 400, "floor": 0.02,
                            "seed": s + 23,
                            "spikes": [[0, 12.0]]
                            + self.spikes(150, 23, 400, 0.4)},
            "zeros": {"kind": "zeros", "n": 300},
            "negative": dict(spiky, negate=True),
            "short2": {"kind": "abs", "n": 2, "floor": 0.5, "seed": s + 24},
            "short3": {"kind": "abs", "n": 3, "floor": 0.5, "seed": s + 25},
            "short1": {"kind": "abs", "n": 1, "floor": 0.5, "seed": s + 26},
            "small": {"kind": "abs", "n": 40, "floor": 0.1, "seed": s + 27,
                      "spikes": self.spikes(2, 5, 40)},
        }

    def prepare(self):
        self.tracks = self.scenarios()
        self.formulas = self.formula_envelopes()
        self.click_envelopes = {}
        self.audio_cache = {}
        for key, spec in self.tracks.items():
            y = self.audio(spec)
            self.audio_cache[key] = y
            self.click_envelopes[key] = self.onset(y, spec["sample_rate"],
                                                   spec["hop"])

    def source(self, key, dtype, origin):
        """The input of a case and the parameters that rebuild it: (envelope,
        sample_rate, hop, params). origin is "formula", "envelope" (librosa's
        click envelope, read back from envelopes.json) or "audio" (the click
        track, whose envelope each side computes)."""
        if origin == "formula":
            o = self.quantize(self.envelope(self.formulas[key]), dtype)
            return o, self.SR, self.HOP, {"envelope": self.formulas[key]}
        spec = self.tracks[key]
        if origin == "envelope":
            o = self.quantize(self.click_envelopes[key], dtype)
        else:
            y = self.quantize(self.audio_cache[key], dtype)
            o = self.onset(y, spec["sample_rate"], spec["hop"])
        return o, spec["sample_rate"], spec["hop"], {"clicks": key,
                                                     "origin": origin}

    def stack(self, keys, dtype):
        rows = [self.source(k, dtype, "formula")[0] for k in keys]
        return np.stack(rows), {"envelopes": [self.formulas[k] for k in keys]}

    # {1 Decisions}

    @staticmethod
    def top_gap(scores):
        """The gap between the two best finite scores (inf when only one)."""
        finite = np.sort(scores[np.isfinite(scores)])[::-1]
        return np.inf if finite.size < 2 else finite[0] - finite[1]

    def tempo_scores(self, o, sr, hop, start_bpm=120.0, std_bpm=1.0,
                     ac_size=8.0, max_tempo=320.0, aggregate=np.mean,
                     prior=None):
        win = int(int(ac_size * sr) // hop)
        tg = librosa.feature.tempogram(onset_envelope=o, sr=sr,
                                       hop_length=hop, win_length=win)
        if aggregate is not None:
            tg = aggregate(tg, axis=-1, keepdims=True)
        bpms = librosa.tempo_frequencies(win, hop_length=hop, sr=sr)
        if prior is None:
            logprior = -0.5 * ((np.log2(bpms) - np.log2(start_bpm))
                               / std_bpm) ** 2
        else:
            logprior = prior.logpdf(bpms)
        if max_tempo is not None:
            logprior[:int(np.argmax(bpms < max_tempo))] = -np.inf
        return np.log1p(1e6 * tg) + logprior[..., :, None]

    def check_tempo(self, name, scores):
        """Every column's two best scores at least 1e-6 apart."""
        flat = scores.reshape(-1, scores.shape[-2], scores.shape[-1])
        gap = min(self.top_gap(flat[r, :, t]) for r in range(flat.shape[0])
                  for t in range(flat.shape[2]))
        assert gap >= 1e-6, (name, gap)
        return gap

    @staticmethod
    def replica(o, sr, hop, bpm, tightness=100.0, trim=True,
                include_first=False, half_away=False):
        """librosa's beat tracker in python, with the margins of every
        decision the result depends on. include_first and half_away switch
        off the two quirks the goldens pin."""
        n = len(o)
        fpb = float(np.round(float(sr) / hop * 60.0 / bpm))
        on = o / (o.std(ddof=1) + np.finfo(o.dtype).tiny)
        window = np.exp(-0.5 * (np.arange(-fpb, fpb + 1) * 32.0 / fpb) ** 2)
        k_len = len(window)
        ls = np.zeros(n)
        upper = (lambda i: min(i + k_len // 2 + 1, k_len)) if include_first \
            else (lambda i: min(i + k_len // 2, k_len))
        for i in range(n):
            acc = 0.0
            for k in range(max(0, i + k_len // 2 - n + 1), upper(i)):
                acc += window[k] * on[i + k_len // 2 - k]
            ls[i] = acc
        tight = float(np.float32(tightness))
        half = int(np.floor(fpb / 2 + 0.5)) if half_away else int(np.rint(fpb / 2))
        thresh = 0.01 * ls.max()
        back = np.zeros(n, dtype=int)
        cum = np.zeros(n)
        margin = np.full(n, np.inf)
        first = True
        first_margin = np.inf
        for i in range(n):
            cands = []
            for loc in range(i - half, int(i - 2 * fpb - 1), -1):
                if loc < 0:
                    break
                cands.append((cum[loc] - tight * (np.log(i - loc)
                                                  - np.log(fpb)) ** 2, loc))
            best, loc_b = -np.inf, -1
            for score, loc in cands:
                if score > best:
                    best, loc_b = score, loc
            rest = [s for s, loc in cands if loc != loc_b and np.isfinite(s)]
            if rest:
                margin[i] = (best - max(rest)) / max(1.0, abs(best))
            cum[i] = ls[i] + best if loc_b >= 0 else ls[i]
            if first:
                first_margin = min(first_margin,
                                   abs(ls[i] - thresh) / max(thresh, 1e-300))
            if first and ls[i] < thresh:
                back[i] = -1
            else:
                back[i] = loc_b
                first = False
        is_max = np.zeros(n, dtype=bool)
        is_max[1:-1] = (cum[1:-1] > cum[:-2]) & (cum[1:-1] >= cum[2:])
        is_max[-1] = cum[-1] > cum[-2]
        values = np.sort(cum[is_max])
        c = len(values)
        median = (values[(c - 1) // 2] + values[c // 2]) / 2.0 if c else 0.0
        threshold = 0.5 * median
        tail = n - 1
        for i in range(n - 1, -1, -1):
            if is_max[i] and cum[i] >= threshold:
                tail = i
                break
        last_margin = (np.min(np.abs(cum[is_max] - threshold))
                       / max(abs(threshold), 1e-300)) if c else np.inf
        path = []
        i = tail
        while i >= 0:
            path.append(i)
            i = back[i]
        path_margin = min(margin[i] for i in path)
        beats = np.zeros(n, dtype=bool)
        beats[path] = True
        a = ls[beats]
        smooth = np.convolve(a, np.hanning(5))[2:n + 2]
        th = 0.5 * np.sqrt(np.mean(smooth ** 2)) if trim else 0.0
        keep = beats.copy()
        i = 0
        while i < n and ls[i] <= th:
            keep[i] = False
            i += 1
        lead = i
        i = n - 1
        while i >= 0 and ls[i] <= th:
            keep[i] = False
            i -= 1
        trail = i
        rel = lambda v: abs(v - th) / max(abs(th), 1e-300)
        trim_frames = list(range(0, min(lead + 1, n))) + list(range(max(trail, 0), n))
        trim_margin = min(rel(ls[j]) for j in trim_frames) if th != 0.0 else np.inf
        return np.flatnonzero(keep), {
            "path": path_margin, "first": first_margin, "last": last_margin,
            "trim": trim_margin}

    def check_beats(self, name, o, sr, hop, kwargs, scale=1e-9):
        tempo, beats = librosa.beat.beat_track(onset_envelope=o, sr=sr,
                                               hop_length=hop, **kwargs)
        if not o.any():
            return tempo, beats, {}
        bpm = kwargs.get("bpm")
        tempo_margin = np.inf
        if bpm is None:
            tempo_margin = self.check_tempo(name, self.tempo_scores(
                o, sr, hop, start_bpm=kwargs.get("start_bpm", 120.0),
                prior=kwargs.get("prior")))
            bpm = float(np.atleast_1d(tempo)[0])
        mine, margins = self.replica(o, sr, hop, bpm,
                                     tightness=kwargs.get("tightness", 100.0),
                                     trim=kwargs.get("trim", True))
        assert np.array_equal(mine, beats), (name, mine, beats)
        margins["tempo"] = tempo_margin
        # Predecessors tie exactly where both scores extend one cumulative
        # score through the same penalty, the local scores between them
        # vanishing below its last place: such a tie is the same bits on
        # both sides and breaks the same way. Every other decision keeps a
        # relative margin of 1e-6, and the beats survive perturbations of
        # the envelope far above the two sides' disagreement, which keep its
        # zeros.
        assert margins["path"] == 0.0 or margins["path"] >= 1e-6, (
            name, margins["path"])
        for key in ("first", "last", "trim"):
            assert margins[key] >= 1e-6, (name, key, margins[key])
        call = dict(kwargs, bpm=bpm)
        for seed in range(3):
            r = self.lcg(len(o), self.SEED + 90 + seed)
            _, moved = librosa.beat.beat_track(
                onset_envelope=o * (1.0 + scale * r), sr=sr, hop_length=hop,
                **call)
            assert np.array_equal(moved, beats), (name, seed)
        return tempo, beats, margins

    def check_plp(self, name, o, sr, hop, win=384, tempo_min=30.0,
                  tempo_max=300.0, prior=None):
        ft = librosa.feature.fourier_tempogram(onset_envelope=o, sr=sr,
                                               hop_length=hop, win_length=win)
        f = librosa.fourier_tempo_frequencies(sr=sr, hop_length=hop,
                                              win_length=win)
        if tempo_min is not None:
            ft[..., f < tempo_min, :] = 0
        if tempo_max is not None:
            ft[..., f > tempo_max, :] = 0
        mag = np.log1p(1e6 * np.abs(ft))
        if prior is not None:
            mag = mag + prior.logpdf(f)[:, None]
        flat = mag.reshape(-1, mag.shape[-2], mag.shape[-1])
        flat_ft = ft.reshape(flat.shape)
        gap, sign = np.inf, np.inf
        for r in range(flat.shape[0]):
            for t in range(flat.shape[2]):
                col = flat[r, :, t]
                if not np.any(flat_ft[r, :, t]):
                    continue
                gap = min(gap, self.top_gap(col))
                k = int(np.argmax(col))
                z = flat_ft[r, k, t]
                sign = min(sign, abs(z.real) / abs(z))
        assert gap >= 1e-6 and sign >= 1e-6, (name, gap, sign)
        return gap, sign

    @staticmethod
    def plp_unit(o, sr, hop, win=384, tempo_min=30.0, tempo_max=300.0):
        """plp with every frame divided by its own peak modulus: kernels of
        unit amplitude, Grosche and Mueller's method (not librosa)."""
        ft = librosa.feature.fourier_tempogram(onset_envelope=o, sr=sr,
                                               hop_length=hop, win_length=win)
        f = librosa.fourier_tempo_frequencies(sr=sr, hop_length=hop,
                                              win_length=win)
        ft[..., f < tempo_min, :] = 0
        ft[..., f > tempo_max, :] = 0
        mag = np.log1p(1e6 * np.abs(ft))
        ft[mag < mag.max(axis=-2, keepdims=True)] = 0
        ft /= librosa.util.tiny(ft) ** 0.5 + np.abs(ft).max(axis=-2,
                                                            keepdims=True)
        pulse = librosa.istft(ft, hop_length=1, n_fft=win, length=o.shape[-1])
        return librosa.util.normalize(np.clip(pulse, 0, None), axis=-1)

    # {1 Files}

    def frequency_cases(self):
        cases = []
        for n, sr, hop in [(4, 22050, 512), (344, 22050, 512),
                           (384, 22050, 512), (256, 16000, 256),
                           (100, 44100, 1024)]:
            values = librosa.tempo_frequencies(n, hop_length=hop, sr=sr)
            assert np.isinf(values[0])
            cases.append({"name": f"tempo_n{n}_sr{sr}_hop{hop}",
                          "params": {"kind": "tempo", "n": n,
                                     "sample_rate": sr, "hop": hop},
                          "shape": [n - 1], "values": values[1:].tolist()})
        for win, sr, hop in [(6, 22050, 512), (384, 22050, 512),
                             (383, 22050, 512), (256, 16000, 256),
                             (1, 22050, 512)]:
            values = librosa.fourier_tempo_frequencies(sr=sr, hop_length=hop,
                                                       win_length=win)
            cases.append({"name": f"fourier_win{win}_sr{sr}_hop{hop}",
                          "params": {"kind": "fourier", "n": win,
                                     "sample_rate": sr, "hop": hop},
                          "shape": [len(values)], "values": values.tolist()})
        return cases

    @staticmethod
    def columns(frames, win):
        p = win // 2
        picks = [0, p - 1, p, frames // 2, frames - p, frames - 1]
        return sorted({c for c in picks if 0 <= c < frames})

    def tempogram_cases(self):
        cases = []
        # (case key, input key, origin, win, center, norm, dtypes, all columns)
        specs = [
            ("spiky_p21", "spiky_p21", "formula", 384, True, "inf",
             ["float64", "float32"], False),
            ("pow4_win383", "pow4", "formula", 383, True, "inf", ["float64"],
             False),
            ("small_win8", "small", "formula", 8, True, "inf",
             ["float64", "float32"], True),
            ("small_win7", "small", "formula", 7, True, "inf", ["float64"],
             True),
            ("small_win1", "small", "formula", 1, True, "inf", ["float64"],
             True),
            ("small_win40", "small", "formula", 40, True, "inf", ["float64"],
             True),
            ("short3_win8", "short3", "formula", 8, True, "inf", ["float64"],
             True),
            ("short1_win8", "short1", "formula", 8, True, "inf", ["float64"],
             True),
            ("small_left_win8", "small", "formula", 8, False, "inf",
             ["float64"], True),
            ("small_left_win40", "small", "formula", 40, False, "inf",
             ["float64"], True),
            ("small_none", "small", "formula", 16, True, "none", ["float64"],
             True),
            ("small_l2", "small", "formula", 16, True, "2", ["float64"], True),
            ("small_l1", "small", "formula", 16, True, "1", ["float64"], True),
            ("small_l3", "small", "formula", 16, True, "3", ["float64"], True),
            ("clicks120_envelope", "clicks120", "envelope", 384, True, "inf",
             ["float64"], False),
            ("clicks120_audio", "clicks120", "audio", 384, True, "inf",
             ["float64"], False),
            ("clicks100_16k_audio", "clicks100_16k", "audio", 64, True, "inf",
             ["float64"], False),
        ]
        norms = {"inf": np.inf, "none": None, "1": 1.0, "2": 2.0, "3": 3.0}
        for key, src, origin, win, center, norm, dtypes, full in specs:
            for dtype in dtypes:
                o, sr, hop, params = self.source(src, dtype, origin)
                tg = librosa.feature.tempogram(onset_envelope=o, sr=sr,
                                               hop_length=hop, win_length=win,
                                               center=center,
                                               norm=norms[norm])
                frames = tg.shape[-1]
                cols = list(range(frames)) if full else self.columns(frames,
                                                                     win)
                expected = tg[..., cols]
                params.update({"win_length": win,
                               "alignment": "centered" if center else "left",
                               "norm": norm, "columns": cols, "dtype": dtype,
                               "sample_rate": sr, "hop": hop})
                cases.append({"name": f"{key}_{dtype}", "params": params,
                              "shape": list(expected.shape),
                              "values": expected.ravel().tolist()})
        # a batch of two envelopes, [2; 600]
        o, params = self.stack(["spiky_p21", "pow4"], "float64")
        tg = librosa.feature.tempogram(onset_envelope=o, win_length=64)
        cols = self.columns(tg.shape[-1], 64)
        params.update({"win_length": 64, "alignment": "centered",
                       "norm": "inf", "columns": cols, "dtype": "float64"})
        cases.append({"name": "batch_float64", "params": params,
                      "shape": list(tg[..., cols].shape),
                      "values": tg[..., cols].ravel().tolist()})
        return cases

    def fourier_cases(self):
        cases = []
        specs = [
            ("spiky_p21", "spiky_p21", 384, True, ["float64", "float32"]),
            ("small_win15", "small", 15, True, ["float64"]),
            ("small_left_win16", "small", 16, False, ["float64"]),
            ("short3_win8", "short3", 8, True, ["float64"]),
        ]
        for key, src, win, center, dtypes in specs:
            for dtype in dtypes:
                o, sr, hop, params = self.source(src, dtype, "formula")
                ft = librosa.feature.fourier_tempogram(
                    onset_envelope=o, sr=sr, hop_length=hop, win_length=win,
                    center=center)
                frames = ft.shape[-1]
                cols = (self.columns(frames, win) if win > 64
                        else list(range(frames)))
                z = ft[..., cols]
                expected = np.stack([z.real, z.imag], axis=-1)
                params.update({"win_length": win,
                               "alignment": "centered" if center else "left",
                               "columns": cols, "dtype": dtype})
                cases.append({"name": f"{key}_{dtype}", "params": params,
                              "shape": list(expected.shape),
                              "values": expected.ravel().tolist()})
        return cases

    def tempo_cases(self):
        cases = []
        uniform = scipy.stats.uniform(30, 300)
        lognorm = scipy.stats.lognorm(s=1.0, loc=np.log(120.0), scale=120.0)
        priors = {"uniform": uniform, "lognorm": lognorm}
        # (case key, input key, origin, kwargs, dtypes)
        specs = [
            ("clicks90", "clicks90", "envelope", {}, ["float64"]),
            ("clicks120", "clicks120", "envelope", {}, ["float64", "float32"]),
            ("clicks128", "clicks128", "envelope", {}, ["float64"]),
            ("clicks150", "clicks150", "envelope", {}, ["float64"]),
            ("clicks110_noise", "clicks110_noise", "envelope", {},
             ["float64"]),
            ("change100to140", "change100to140", "envelope", {}, ["float64"]),
            ("swing120", "swing120", "envelope", {}, ["float64"]),
            ("noise", "noise", "envelope", {}, ["float64"]),
            ("clicks100_16k", "clicks100_16k", "envelope", {}, ["float64"]),
            ("clicks90_audio", "clicks90", "audio", {}, ["float64"]),
            ("clicks150_audio", "clicks150", "audio", {}, ["float64",
                                                            "float32"]),
            ("change_audio", "change100to140", "audio", {}, ["float64"]),
            ("spiky_p21", "spiky_p21", "formula", {}, ["float64", "float32"]),
            ("start60", "spiky_p25", "formula", {"start_bpm": 60.0},
             ["float64"]),
            ("start200", "noise", "envelope", {"start_bpm": 200.0},
             ["float64"]),
            ("std0.5", "clicks90", "envelope", {"std_bpm": 0.5}, ["float64"]),
            ("std2", "clicks120", "envelope", {"std_bpm": 2.0}, ["float64"]),
            ("ac4", "noise", "envelope", {"ac_size": 4.0}, ["float64"]),
            ("max100", "clicks120", "envelope", {"max_tempo": 100.0},
             ["float64"]),
            ("max1", "clicks120", "envelope", {"max_tempo": 1.0},
             ["float64"]),
            ("max_none", "clicks120", "envelope", {"max_tempo": None},
             ["float64"]),
            ("frames_change", "change100to140", "envelope",
             {"aggregate": None}, ["float64"]),
            ("median", "pow4", "formula", {"aggregate": np.median},
             ["float64"]),
            ("prior_uniform", "clicks150", "envelope", {"prior": "uniform"},
             ["float64"]),
            ("prior_lognorm", "clicks90", "envelope", {"prior": "lognorm"},
             ["float64"]),
        ]
        for key, src, origin, kwargs, dtypes in specs:
            for dtype in dtypes:
                o, sr, hop, params = self.source(src, dtype, origin)
                call = dict(kwargs)
                if "prior" in call:
                    call["prior"] = priors[call["prior"]]
                expected = librosa.feature.tempo(onset_envelope=o, sr=sr,
                                                 hop_length=hop, **call)
                if key in self.TEMPO_PINS:
                    # the parameter decides this case
                    assert not np.array_equal(expected, librosa.feature.tempo(
                        onset_envelope=o, sr=sr, hop_length=hop)), key
                self.check_tempo(f"tempo {key}", self.tempo_scores(
                    o, sr, hop, **call))
                params.update({"dtype": dtype, "sample_rate": sr, "hop": hop})
                for k, v in kwargs.items():
                    if k == "aggregate":
                        params["aggregate"] = "none" if v is None else "median"
                    elif k == "max_tempo" and v is None:
                        params["max_tempo"] = "none"
                    else:
                        params[k] = v
                cases.append({"name": f"{key}_{dtype}", "params": params,
                              "shape": list(expected.shape),
                              "values": expected.ravel().tolist()})
        o, params = self.stack(["spiky_p21", "spiky_p25"], "float64")
        expected = librosa.feature.tempo(onset_envelope=o, sr=self.SR)
        self.check_tempo("tempo batch", self.tempo_scores(o, self.SR,
                                                          self.HOP))
        params.update({"dtype": "float64", "sample_rate": self.SR,
                       "hop": self.HOP})
        cases.append({"name": "batch_float64", "params": params,
                      "shape": list(expected.shape),
                      "values": expected.ravel().tolist()})
        return cases

    def beat_cases(self):
        cases = []
        uniform = scipy.stats.uniform(30, 300)
        specs = [
            ("clicks90", "clicks90", "envelope", {}, ["float64", "float32"]),
            ("clicks120", "clicks120", "envelope", {}, ["float64"]),
            ("clicks128", "clicks128", "envelope", {}, ["float64"]),
            ("clicks150", "clicks150", "envelope", {}, ["float64"]),
            ("clicks110_noise", "clicks110_noise", "envelope", {},
             ["float64"]),
            ("change100to140", "change100to140", "envelope", {}, ["float64"]),
            ("swing120", "swing120", "envelope", {}, ["float64"]),
            ("noise", "noise", "envelope", {}, ["float64"]),
            ("clicks100_16k", "clicks100_16k", "envelope", {}, ["float64"]),
            ("clicks90_audio", "clicks90", "audio", {}, ["float64",
                                                          "float32"]),
            ("clicks120_audio", "clicks120", "audio", {}, ["float64"]),
            ("change_audio", "change100to140", "audio", {}, ["float64"]),
            ("swing_audio", "swing120", "audio", {}, ["float64"]),
            ("clicks100_16k_audio", "clicks100_16k", "audio", {},
             ["float64"]),
            ("spiky_p21", "spiky_p21", "formula", {}, ["float64", "float32"]),
            ("first_spike", "first_spike", "formula", {}, ["float64",
                                                           "float32"]),
            ("zeros", "zeros", "formula", {}, ["float64"]),
            ("negative", "negative", "formula", {}, ["float64"]),
            ("short2", "short2", "formula", {"bpm": 120.0}, ["float64"]),
            ("bpm118", "clicks120", "envelope", {"bpm": 118.0}, ["float64"]),
            ("bpm123", "spiky_p10", "formula",
             {"bpm": 123.0, "tightness": 1.0}, ["float64"]),
            ("tight400", "clicks110_noise", "envelope", {"tightness": 400.0},
             ["float64"]),
            ("tight680", "change100to140", "envelope", {"tightness": 680.0},
             ["float64"]),
            ("tight_odd", "swing120", "envelope", {"tightness": 123.456},
             ["float64"]),
            ("notrim", "clicks150", "envelope", {"trim": False},
             ["float64"]),
            ("start80", "clicks150", "envelope", {"start_bpm": 80.0},
             ["float64"]),
            ("prior_uniform", "clicks90", "envelope", {"prior": "uniform"},
             ["float64"]),
        ]
        for key, src, origin, kwargs, dtypes in specs:
            for dtype in dtypes:
                o, sr, hop, params = self.source(src, dtype, origin)
                call = dict(kwargs)
                if "prior" in call:
                    call["prior"] = uniform
                # audio cases cover the onset envelope's own disagreement
                tempo, beats, margins = self.check_beats(
                    f"beat {key}_{dtype}", o, sr, hop, call,
                    scale=1e-7 if origin == "audio" else 1e-9)
                if origin == "audio" and dtype == "float32":
                    # The library's float32 envelope differs from the
                    # reference by up to 1e-4 of the decibel scale, its zeros
                    # kept: the beats must not move under a relative
                    # perturbation of 1e-3.
                    self.check_beats(f"beat {key}_{dtype} perturbed", o, sr,
                                     hop, call, scale=1e-3)
                if key in self.BEAT_PINS:
                    _, plain = librosa.beat.beat_track(onset_envelope=o,
                                                       sr=sr, hop_length=hop)
                    assert not np.array_equal(plain, beats), key
                if key == "first_spike":
                    bpm = float(np.atleast_1d(tempo)[0])
                    other, _ = self.replica(o, sr, hop, bpm, include_first=True)
                    assert not np.array_equal(other, beats), key
                if key == "bpm123":
                    other, _ = self.replica(o, sr, hop, 123.0, half_away=True)
                    assert not np.array_equal(other, beats), key
                params.update({"dtype": dtype, "sample_rate": sr, "hop": hop})
                params.update(kwargs)
                cases.append({"name": f"{key}_{dtype}", "params": params,
                              "shape": [len(beats)],
                              "values": [float(b) for b in beats]})
                cases.append({"name": f"tempo_{key}_{dtype}",
                              "params": params, "shape": [1],
                              "values": [float(np.atleast_1d(tempo)[0])]})
        return cases

    def plp_cases(self):
        cases = []
        uniform = scipy.stats.uniform(30, 300)
        lognorm = scipy.stats.lognorm(s=1.0, loc=np.log(120.0), scale=120.0)
        priors = {"uniform": uniform, "lognorm": lognorm}
        specs = [
            ("clicks120", "clicks120", "envelope", {}, ["float64"]),
            ("change100to140", "change100to140", "envelope", {}, ["float64"]),
            ("swing120", "swing120", "envelope", {}, ["float64"]),
            ("clicks120_audio", "clicks120", "audio", {}, ["float64"]),
            ("spiky_p21", "spiky_p21", "formula", {}, ["float64", "float32"]),
            ("range60_180", "clicks90", "envelope",
             {"tempo_min": 60.0, "tempo_max": 180.0}, ["float64"]),
            ("unbounded", "spiky_p25", "formula",
             {"tempo_min": None, "tempo_max": None}, ["float64"]),
            ("win256", "clicks128", "envelope", {"win_length": 256},
             ["float64"]),
            ("win383", "spiky_p25", "formula", {"win_length": 383},
             ["float64"]),
            ("prior_uniform", "clicks150", "envelope", {"prior": "uniform"},
             ["float64"]),
            ("prior_lognorm", "clicks90", "envelope", {"prior": "lognorm"},
             ["float64"]),
        ]
        for key, src, origin, kwargs, dtypes in specs:
            for dtype in dtypes:
                o, sr, hop, params = self.source(src, dtype, origin)
                call = dict(kwargs)
                if "prior" in call:
                    call["prior"] = priors[call["prior"]]
                expected = librosa.beat.plp(onset_envelope=o, sr=sr,
                                            hop_length=hop, **call)
                self.check_plp(f"plp {key}", o, sr, hop,
                               win=call.get("win_length", 384),
                               tempo_min=call.get("tempo_min", 30.0),
                               tempo_max=call.get("tempo_max", 300.0),
                               prior=call.get("prior"))
                params.update({"dtype": dtype, "sample_rate": sr, "hop": hop,
                               "kernels": "reference"})
                for k, v in kwargs.items():
                    params[k] = "none" if v is None else v
                cases.append({"name": f"{key}_{dtype}", "params": params,
                              "shape": list(expected.shape),
                              "values": expected.ravel().tolist()})
        o, params = self.stack(["spiky_p21", "spiky_p25"], "float64")
        expected = librosa.beat.plp(onset_envelope=o, sr=self.SR)
        self.check_plp("plp batch", o, self.SR, self.HOP)
        params.update({"dtype": "float64", "sample_rate": self.SR,
                       "hop": self.HOP, "kernels": "reference"})
        cases.append({"name": "batch_float64", "params": params,
                      "shape": list(expected.shape),
                      "values": expected.ravel().tolist()})
        o, sr, hop, params = self.source("change100to140", "float64",
                                         "envelope")
        expected = self.plp_unit(o, sr, hop)
        params.update({"dtype": "float64", "sample_rate": sr, "hop": hop,
                       "kernels": "unit"})
        cases.append({"name": "unit_change100to140_float64", "params": params,
                      "shape": list(expected.shape),
                      "values": expected.ravel().tolist()})
        return cases

    def envelope_cases(self):
        return [{"name": key, "params": self.tracks[key],
                 "shape": [len(o)], "values": o.tolist()}
                for key, o in self.click_envelopes.items()]

    def generate(self):
        assert np.array_equal(np.hanning(5), [0.0, 0.5, 1.0, 0.5, 0.0])
        self.prepare()
        write_suite(self.SUITE, "envelopes", self.envelope_cases())
        write_suite(self.SUITE, "frequencies", self.frequency_cases())
        write_suite(self.SUITE, "tempogram", self.tempogram_cases())
        write_suite(self.SUITE, "fourier_tempogram", self.fourier_cases())
        write_suite(self.SUITE, "tempo", self.tempo_cases())
        write_suite(self.SUITE, "beat_track", self.beat_cases())
        write_suite(self.SUITE, "plp", self.plp_cases())


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
    PvocVectorGenerator,
    IoVectorGenerator,
    DeltaVectorGenerator,
    PcenVectorGenerator,
    TrimVectorGenerator,
    FilterVectorGenerator,
    DefaultsVectorGenerator,
    SynthesisVectorGenerator,
    F0VectorGenerator,
    TuningVectorGenerator,
    OnsetDetectVectorGenerator,
    RhythmVectorGenerator,
]

if __name__ == "__main__":
    for generator in GENERATORS:
        generator().generate()
