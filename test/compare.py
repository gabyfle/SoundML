import numpy as np
import matplotlib.pyplot as plt
import pydub
import time


def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_wav(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y


start = time.time()

filename = "test/sin_1k.wav"
fs, signal = read(filename)

end = time.time()
print("Temps lecture : ", end - start, "s")

if len(signal.shape) > 1:
    signal = signal[:, 0]

n = len(signal)

print("Taille du signal:", len(signal))

if signal.dtype == np.int16:
    signal = signal / 32768.0
elif signal.dtype == np.int32:
    signal = signal / 2147483648.0

start = time.time()

fft_signal = np.fft.fft(signal)
frequencies = np.fft.fftfreq(n, 1 / fs)

end = time.time()

print("Temps d'exécution : ", end - start, "s")
