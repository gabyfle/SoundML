import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

filename = "test/sin_1k.wav"
fs, signal = wavfile.read(filename)

if len(signal.shape) > 1:
    signal = signal[:, 0]

n = len(signal)

print(len(signal))
print(signal)

if signal.dtype == np.int16:
    signal = signal / 32768.0
elif signal.dtype == np.int32:
    signal = signal / 2147483648.0

fft_signal = np.fft.fft(signal)
print(n)
print(fs)
frequencies = np.fft.fftfreq(n, 1 / fs)

plt.figure()
plt.plot(frequencies[: n // 2], np.abs(fft_signal)[: n // 2])
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.title("Transformée de Fourier du signal")
plt.grid()
plt.show()
