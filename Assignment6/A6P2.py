import numpy as np
import matplotlib.pyplot as plt

# P2A
Nx=28
n = np.arange(Nx)
x = 0.9**n

Nh=9
h = np.ones(Nh)
# print(h)

y = np.convolve(x, h)
# print(y)
# print(len(y)) # Prints 36

x_stem = np.arange(len(y))
# plt.stem(x_stem, y, label="y[n]")
plt.title("y[n] = x[n] * h[n]")
plt.xlabel("n")
plt.xlabel("y[n]")
plt.legend()
plt.grid()
# plt.show()
# P2A

# P2B
Ny = (Nx + Nh - 1) * 2 # Changing this variable for (B)
y_ifftn = np.arange(Ny)
H = np.fft.fft(h, Ny)
X = np.fft.fft(x, Ny)

Y = H * X
# print(Y)

y_ifft = np.fft.ifft(Y)
plt.stem(y_ifftn, y_ifft)
plt.title("y[n] via. freq. domain (IFFT)")
plt.xlabel("n")
plt.ylabel("y[n]")
plt.xlim(0, 35)
plt.grid()
plt.show()
# P2B