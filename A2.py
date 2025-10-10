import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

f = 0.5
t = 1

Fs1 = 4000
Fs2 = 1500

N1 = Fs1 * t
N2 = Fs2 * t
n1 = np.arange(N1)
n2 = np.arange(N2)

xa = np.cos(2000 * np.pi * n1)

# sd.play(myarray, fs)
sd.play(xa, Fs1)
