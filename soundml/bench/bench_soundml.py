"""Librosa cross-reference benchmarks for SoundML.

The Python twin of ``bench_soundml.ml``: every SoundML case with a librosa (or
scipy) reference implementation gets a row here under the same group/name
scheme, so the two tables read side by side. Window generation seeds the
suite; STFT, synthesis and mel rows land together with their OCaml
counterparts.

Run it from the repository root:

    python bench/bench_soundml.py
"""

from __future__ import annotations

import time
from typing import Any, Callable, List, Tuple

import librosa
import numpy as np

N_WINDOW = 2048

# The stft/mel cases: 30 s of mono audio at librosa's default rate, analysed
# at fft 2048 and hop 512 with 128 mel bands; keep in sync with the OCaml
# suite.
SAMPLE_RATE = 22050
N_AUDIO = 30 * SAMPLE_RATE
N_FFT = 2048
HOP = 512
N_MELS = 128
N_GRIFFINLIM_ITER = 32

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


# The published librosa kaiser_best triple, as documented by
# torchaudio.functional.resample.
KAISER_BEST = {
    "lowpass_filter_width": 64,
    "rolloff": 0.9475937167399596,
    "resampling_method": "sinc_interp_kaiser",
    "beta": 14.769656459379492,
}

RESAMPLE_PAIRS = [("44k1-48k", 44100, 48000), ("44k1-16k", 44100, 16000)]


def resample_benchmarks() -> List[Tuple[str, Callable[[], Any]]]:
    """The fair fight for ``resample/apply high``: librosa's soxr_hq (the
    spec SoundML's default is designed to) and torchaudio at both its
    defaults (quality documented as incomparable) and its published librosa
    kaiser_best triple. torchaudio rows appear only when torch is
    installed."""
    benches: List[Tuple[str, Callable[[], Any]]] = []
    rng = np.random.default_rng(42)

    for pair, sr, target in RESAMPLE_PAIRS:
        clip = rng.standard_normal(sr).astype(np.float32)

        def make_librosa(clip: np.ndarray, sr: int, target: int):
            def run() -> None:
                librosa.resample(
                    clip, orig_sr=sr, target_sr=target, res_type="soxr_hq"
                )

            return run

        benches.append(
            bench(
                f"resample/apply high {pair} f32 (librosa soxr_hq)",
                make_librosa(clip, sr, target),
            )
        )

    try:
        import torch
        import torchaudio.functional as taf
    except ImportError:
        print("torchaudio not installed: skipping its resample rows")
        return benches

    for pair, sr, target in RESAMPLE_PAIRS:
        clip = torch.from_numpy(rng.standard_normal(sr).astype(np.float32))

        def make_ta(clip: "torch.Tensor", sr: int, target: int, kwargs: dict):
            def run() -> None:
                taf.resample(clip, sr, target, **kwargs)

            return run

        benches.append(
            bench(
                f"resample/apply high {pair} f32 (torchaudio kaiser_best)",
                make_ta(clip, sr, target, KAISER_BEST),
            )
        )
        benches.append(
            bench(
                f"resample/apply high {pair} f32 (torchaudio defaults)",
                make_ta(clip, sr, target, {}),
            )
        )

    return benches


# The constant-Q cases: the same 30 s of mono audio through the default
# seven-octave ladder, the erb variable-Q ladder and the 252-bin chromagram.
# Every call pins tuning=0.0 and sparsity=0 — librosa estimates tuning from the
# signal by default and discards 1% of each filter's mass — so the two suites
# compute the same transform. keep in sync with the OCaml suite.
CQT_FMIN = 32.70319566257483

CQT_COMMON = dict(
    sr=SAMPLE_RATE,
    hop_length=HOP,
    fmin=CQT_FMIN,
    tuning=0.0,
    sparsity=0,
    norm=1,
    window="hann",
    filter_scale=1,
    scale=True,
    pad_mode="constant",
    res_type="soxr_hq",
)


def cqt_benchmarks() -> List[Tuple[str, Callable[[], Any]]]:
    """librosa.vqt at gamma=0 and gamma=None, and the chroma_cqt composition.
    librosa.feature.chroma_cqt exposes no sparsity control, so its twin times
    the C= composition the OCaml row computes."""
    rng = np.random.default_rng(42)
    y32 = rng.random(N_AUDIO, dtype=np.float32) * 2.0 - 1.0
    y64 = y32.astype(np.float64)

    def transform(y: np.ndarray, n_bins: int, bins_per_octave: int, gamma: Any):
        dtype = np.complex64 if y.dtype == np.float32 else np.complex128

        def run() -> None:
            np.abs(
                librosa.vqt(
                    y,
                    n_bins=n_bins,
                    bins_per_octave=bins_per_octave,
                    gamma=gamma,
                    dtype=dtype,
                    **CQT_COMMON,
                )
            )

        return run

    def chroma(y: np.ndarray, n_bins: int, bins_per_octave: int):
        dtype = np.complex64 if y.dtype == np.float32 else np.complex128

        def run() -> None:
            transform = np.abs(
                librosa.vqt(
                    y,
                    n_bins=n_bins,
                    bins_per_octave=bins_per_octave,
                    gamma=0,
                    dtype=dtype,
                    **CQT_COMMON,
                )
            )
            librosa.feature.chroma_cqt(
                C=transform,
                sr=SAMPLE_RATE,
                hop_length=HOP,
                fmin=CQT_FMIN,
                n_chroma=12,
                n_octaves=n_bins // bins_per_octave,
                bins_per_octave=bins_per_octave,
                threshold=0.0,
            )

        return run

    benches: List[Tuple[str, Callable[[], Any]]] = []
    for tag, y in (("f32", y32), ("f64", y64)):
        benches.append(
            bench(f"cqt/cqt 30s {tag} 84/12 hop{HOP} (librosa)",
                  transform(y, 84, 12, 0))
        )
    for tag, y in (("f32", y32), ("f64", y64)):
        benches.append(
            bench(f"cqt/vqt 30s {tag} 84/12 gamma-erb hop{HOP} (librosa)",
                  transform(y, 84, 12, None))
        )
    for tag, y in (("f32", y32), ("f64", y64)):
        benches.append(
            bench(f"cqt/chroma_cqt 30s {tag} n252 bpo36 hop{HOP} (librosa)",
                  chroma(y, 252, 36))
        )
    return benches


def build_benchmarks() -> List[Tuple[str, Callable[[], Any]]]:
    benches: List[Tuple[str, Callable[[], Any]]] = []

    for name, spec in WINDOW_SPECS:
        def make(spec: Any) -> Callable[[], Any]:
            def run() -> None:
                w = librosa.filters.get_window(spec, N_WINDOW, fftbins=True)
                float(w.sum())

            return run

        benches.append(bench(f"window/{name} {N_WINDOW} (librosa)", make(spec)))

    rng = np.random.default_rng(42)
    y32 = rng.random(N_AUDIO, dtype=np.float32) * 2.0 - 1.0
    y64 = y32.astype(np.float64)

    def power_spectrum(y: np.ndarray) -> Callable[[], Any]:
        def run() -> None:
            s = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
            np.square(np.abs(s))

        return run

    def mel_spectrogram(y: np.ndarray) -> Callable[[], Any]:
        def run() -> None:
            librosa.feature.melspectrogram(
                y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP, power=2.0
            )

        return run

    def hpss_spectrogram(s: np.ndarray, power: float) -> Callable[[], Any]:
        def run() -> None:
            librosa.decompose.hpss(s, kernel_size=31, power=power)

        return run

    def hpss_signal(y: np.ndarray) -> Callable[[], Any]:
        def run() -> None:
            librosa.effects.hpss(y, n_fft=N_FFT, hop_length=HOP, kernel_size=31)

        return run

    def istft(s: np.ndarray) -> Callable[[], Any]:
        def run() -> None:
            librosa.istft(s, n_fft=N_FFT, hop_length=HOP)

        return run

    def griffinlim(s: np.ndarray) -> Callable[[], Any]:
        def run() -> None:
            librosa.griffinlim(
                s,
                n_iter=N_GRIFFINLIM_ITER,
                n_fft=N_FFT,
                hop_length=HOP,
                momentum=0.99,
                init=None,
            )

        return run

    for tag, y in (("f32", y32), ("f64", y64)):
        benches.append(
            bench(
                f"stft/power_spectrum 30s {tag} fft{N_FFT} hop{HOP} (librosa)",
                power_spectrum(y),
            )
        )
    # The synthesis rows: the spectrum is built once, outside the timed call,
    # exactly as the OCaml rows precompute theirs.
    for tag, y in (("f32", y32), ("f64", y64)):
        spectrum = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
        benches.append(
            bench(
                f"istft/invert 30s {tag} fft{N_FFT} hop{HOP} (librosa)",
                istft(spectrum),
            )
        )
    for tag, y in (("f32", y32), ("f64", y64)):
        magnitudes = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
        benches.append(
            bench(
                f"griffinlim/griffin_lim 30s n{N_GRIFFINLIM_ITER} {tag}"
                f" fft{N_FFT} hop{HOP} (librosa)",
                griffinlim(magnitudes),
            )
        )
    for tag, y in (("f32", y32), ("f64", y64)):
        benches.append(
            bench(
                f"mel/mel_spectrogram 30s {tag} fft{N_FFT} hop{HOP} (librosa)",
                mel_spectrogram(y),
            )
        )

    # The separation rows: the power spectrogram is built once, outside the
    # timed call, exactly as the OCaml rows precompute theirs.
    for tag, y in (("f32", y32), ("f64", y64)):
        spectrum = np.square(
            np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
        ).astype(y.dtype)
        benches.append(
            bench(
                f"hpss/hpss 30s {tag} k31 p2 fft{N_FFT} hop{HOP} (librosa)",
                hpss_spectrogram(spectrum, 2.0),
            )
        )
        if tag == "f32":
            benches.append(
                bench(
                    f"hpss/hpss 30s {tag} k31 pinf fft{N_FFT} hop{HOP}"
                    " (librosa)",
                    hpss_spectrogram(spectrum, np.inf),
                )
            )
    benches.append(
        bench(
            f"hpss/hpss signal 30s f32 fft{N_FFT} hop{HOP} (librosa)",
            hpss_signal(y32),
        )
    )
    def time_stretch(y: np.ndarray, rate: float) -> Callable[[], Any]:
        def run() -> None:
            librosa.effects.time_stretch(
                y, rate=rate, n_fft=N_FFT, hop_length=HOP
            )

        return run

    def pitch_shift(y: np.ndarray) -> Callable[[], Any]:
        def run() -> None:
            librosa.effects.pitch_shift(
                y, sr=SAMPLE_RATE, n_steps=4, n_fft=N_FFT, hop_length=HOP
            )

        return run

    for rate in (2.0, 0.5):
        for tag, y in (("f32", y32), ("f64", y64)):
            benches.append(
                bench(
                    f"pvoc/time_stretch 30s r{rate} {tag} fft{N_FFT} hop{HOP}"
                    " (librosa)",
                    time_stretch(y, rate),
                )
            )
    for tag, y in (("f32", y32), ("f64", y64)):
        benches.append(
            bench(
                f"pvoc/pitch_shift 30s +4st {tag} fft{N_FFT} hop{HOP} (librosa)",
                pitch_shift(y),
            )
        )

    benches.extend(cqt_benchmarks())
    benches.extend(resample_benchmarks())
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
