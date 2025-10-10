import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.io import wavfile
import sounddevice as sd

alpha = 0.8
Fs = 22050
R = int(0.8 * Fs)
# R = 8 # for impulse response comparison for single and multi echo filter
a = 1
b = np.zeros(R+1)
b[0] = 1
b[-1] = alpha

x = np.zeros(2000)
x[100] = 1.0  # an impulse to see echo
y = sp.signal.lfilter(b, a, x)

# 3c start

# Impulse response
# n = np.arange(len(y))
# plt.figure()
# plt.stem(n, y)
# plt.title("Impulse response")
# plt.xlabel("n")
# plt.grid()

# Frequency response
# w, H = sp.signal.freqz(b, a, worN=4096)
# # f = w/(2*np.pi)*Fs
# f_norm = w / (2 * np.pi)
# plt.figure()
# plt.plot(f_norm, np.abs(H))
# plt.grid()
# plt.title("Frequency response")
# plt.show()

# 3c end

# 3d start

Fs, x_wav = wavfile.read('/Users/johnnynogo/NTNU/_DigSig/Assignments/Assignment5/piano.wav')

# to avoid distortion
if x_wav.ndim == 2:
    x_wav = x_wav.mean(axis=1)

if np.issubdtype(x_wav.dtype, np.integer):
    x_wav = x_wav.astype(np.float32) / np.iinfo(np.int16).max
else:
    x_wav = x_wav.astype(np.float32)

b = np.zeros(R+1, dtype=np.float32)
b[0] = 1.0
b[-1] = alpha
a = np.array([1.0], dtype=np.float32)
x_filtered = sp.signal.lfilter(b, a, x_wav).astype(np.float32)

peak = np.max(np.abs(x_filtered))
if np.isfinite(peak) and peak > 1.0:
    x_filtered /= peak

# sd.play(x_wav, Fs)
# sd.wait()

# sd.stop()
# # x_filtered = sp.signal.lfilter(b, a, x_wav)
# sd.play(x_filtered, Fs)
# sd.wait()

# 3d end

# 3e/f start
N = 6

be = np.zeros(N*R+1, dtype=np.float32)
be[0] = 1
be[-1] = -alpha**N
ae = np.zeros(R+1, dtype=np.float32)
ae[0] = 1
ae[-1] = -alpha

xe = np.zeros(2000)
xe[100] = 1.0  # an impulse to see echo
ye = sp.signal.lfilter(be, ae, xe)

# Impulse response
ne = np.arange(len(ye))
plt.figure()
plt.stem(ne, ye)
plt.title("Impulse response")
plt.xlabel("n")
plt.grid()

# Frequency response
we, He = sp.signal.freqz(be, ae, worN=4096)
# f = w/(2*np.pi)*Fs
f_norme = we / (2 * np.pi)
plt.figure()
plt.plot(f_norme, np.abs(He))
plt.grid()
plt.title("Frequency response")
plt.show()

# 3e/f end

# 3g start

x_multi_filtered = sp.signal.lfilter(be, ae, x_wav).astype(np.float32)

peake = np.max(np.abs(x_multi_filtered))
if np.isfinite(peake) and peake > 1.0:
    x_multi_filtered /= peake

sd.play(x_multi_filtered, Fs)
sd.wait()

# 3g end