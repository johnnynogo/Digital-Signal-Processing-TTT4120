import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.io import loadmat

mat = loadmat('/Users/johnnynogo/NTNU/_DigSig/Assignments/Assignment5/signals.mat')

print("Keys in the .mat file:", mat.keys())

x = mat['x'].squeeze()
y = mat['y'].squeeze()

# 2a start
print("2a: ----------------------")
plt.figure()
plt.plot(x, label='x[n]')
plt.legend()
plt.grid()

plt.figure()
plt.plot(y, label='y[n]')
plt.legend()
plt.grid()

plt.show()
# 2a end

# 2b start
print("2b: ----------------------")
r_yx = sp.signal.correlate(y, x)
print(r_yx)

plt.plot(r_yx, label=r"$r_{yx}$")
plt.title(r"$r_{yx}$")
plt.legend()
plt.grid()
plt.show()

print(r"Length $r_{yx}$:" + f"{len(r_yx)}")
print(fr"Length $x[n] + y[n]$: {len(x) + len(y)}")
# 2b end

# 2c start
print("2c: ----------------------")
print(r_yx)

x_flipped = np.flip(x)
r_yx_func = np.convolve(y, x_flipped)
plt.figure()
plt.title(r"2c: $r_{yx} function$")
plt.stem(r_yx_func)
plt.grid()
plt.show()

# 2c end