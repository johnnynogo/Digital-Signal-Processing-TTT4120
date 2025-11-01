import numpy as np
import matplotlib.pyplot as plt

L = 100 # Change this one on B
x = np.zeros(L)
f1 = 7/40
f2 = 9/40

for n in range(L):
    x[n] = np.sin(2*np.pi*f1*n) + np.sin(2*np.pi*f2*n)

L_DFT = 128 # Change this one in C

X = np.fft.fft(x, L_DFT)
f = np.linspace(0, 1, L_DFT)

plt.plot(f, np.abs(X))
plt.title(f"Magnitude spectrum |X(f)| for segment length {L}")
plt.xlabel("f")
plt.ylabel("[X(f)|")
plt.grid()
plt.show()
