import pytest
import librosa
import numpy as np
import os

AUDIO_FILE_PATH = os.path.join(os.getcwd(), "bench/stft/wav_stereo_44100hz_1s.wav")

STFT_CONFIGURATIONS = [
    {"n_fft": 2048, "win_length": 2048, "hop_length": 512, "window_type": "hann"}
]

y_f32, sr_f32 = librosa.load(AUDIO_FILE_PATH, sr=None, mono=True, dtype=np.float32)
y_f64, sr_f64 = librosa.load(AUDIO_FILE_PATH, sr=None, mono=True, dtype=np.float64)
SIGNAL_LENGTH = len(y_f32)

@pytest.mark.parametrize("config", STFT_CONFIGURATIONS)
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_librosa_stft(benchmark, config, precision):
    if precision == "float32":
        audio_data = y_f32
    elif precision == "float64":
        audio_data = y_f64
    else:
        raise ValueError("Invalid precision")

    result = benchmark(
        librosa.stft,
        y=audio_data,
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        win_length=config["win_length"],
        window=config["window_type"],
        center=False
    )
    assert result is not None
