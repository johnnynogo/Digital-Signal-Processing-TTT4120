import numpy as np
import matplotlib.pyplot as plt

# P1A
# x_white_binary = np.random.choice([-1, 1], 100)
# x_white_gaussian = np.random.randn(100)
# x_white_uniform = np.random.uniform(-np.sqrt(3), np.sqrt(3), 100)

# plt.figure()
# plt.subplot(3,1,1)
# plt.stem(x_white_binary)
# plt.title("White Binary Noise")
# plt.grid()

# plt.subplot(3,1,2)
# plt.stem(x_white_gaussian)
# plt.title("White Gaussian Noise")
# plt.grid()

# plt.subplot(3,1,3)
# plt.stem(x_white_uniform)
# plt.title("White Uniform Noise")
# plt.grid()

# plt.tight_layout()
# plt.show()

x_white_binary = np.random.choice([-1, 1], 20000)
x_white_gaussian = np.random.randn(20000)
x_white_uniform = np.random.uniform(-np.sqrt(3), np.sqrt(3), 20000)

print(f"Mean value of White Binary Noise: {np.mean(x_white_binary)}")
print(f"Mean value of White Gaussian Noise: {np.mean(x_white_gaussian)}")
print(f"Mean value of White Uniform Noise: {np.mean(x_white_uniform)}")

autocorr_binary = np.correlate(x_white_binary, x_white_binary, mode='full')
autocorr_gaussian = np.correlate(x_white_gaussian, x_white_gaussian, mode='full')
autocorr_uniform = np.correlate(x_white_uniform, x_white_uniform, mode='full')

mid = len(autocorr_binary) // 2
interval = np.arange(-10, 11, 1)

rxx_binary_slice = autocorr_binary[mid-10:mid+11].astype(float)
rxx_gaussian_slice = autocorr_gaussian[mid-10:mid+11]
rxx_uniform_slice = autocorr_uniform[mid-10:mid+11]

rxx_binary_slice /= rxx_binary_slice[10]
rxx_gaussian_slice /= rxx_gaussian_slice[10]
rxx_uniform_slice /= rxx_uniform_slice[10]

plt.figure()
plt.stem(interval, rxx_binary_slice, label='Binary', use_line_collection=True)
plt.title("Autocorrelations")
plt.xlabel("Lag l")
plt.ylabel("Normalized rxx[m]")
plt.grid(True)
plt.legend()

plt.figure()
plt.stem(interval, rxx_gaussian_slice, label='Gaussian', use_line_collection=True)
plt.title("Autocorrelations")
plt.xlabel("Lag l")
plt.ylabel("Normalized rxx[m]")
plt.grid(True)
plt.legend()

plt.figure()
plt.stem(interval, rxx_uniform_slice, label='Uniform', use_line_collection=True)
plt.title("Autocorrelations")
plt.xlabel("Lag l")
plt.ylabel("Normalized rxx[m]")
plt.grid(True)
plt.legend()
plt.show()
