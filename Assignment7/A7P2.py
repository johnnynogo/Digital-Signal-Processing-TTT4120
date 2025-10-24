import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# 2C
N=20000
sigma_w = 3/4
a = 0.5
w = np.random.randn(N)
b = [1]
a = [1, 0.5]
x = sp.signal.lfilter(b, a, w)

mean_est = np.mean(x)
power_est = np.mean(x**2)
print(f"Mean estimator: {mean_est}")
print(f"Power estimator: {power_est}")

lag = np.arange(-10, 11)
autocorr_est_calc = np.correlate(x, x, mode='full') / N
mid = len(autocorr_est_calc) // 2
autocorr_est = autocorr_est_calc[mid-10:mid+11]

l = np.arange(-10, 11)
autocorr_theoretical = (-0.5)**np.abs(l)

power_est = np.abs(np.fft.fftshift(np.fft.fft(autocorr_est, 512)))
power_est = power_est / np.max(power_est)

omega = np.linspace(-np.pi, np.pi, 512)
power_theoretical = 3 / (5 + 4 * np.cos(omega))
power_theoretical /= np.max(power_theoretical)


plt.subplot(2, 1, 1)
plt.stem(l, autocorr_est, linefmt='C0-', markerfmt='C0o', basefmt=' ')
plt.plot(l, autocorr_theoretical, 'r--', label='Theoretical')
plt.title('Autocorrelation Function rₓₓ(m)')
plt.xlabel('Lag m')
plt.ylabel('rₓₓ(m)')
plt.legend(['Theoretical', 'Estimated'])
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(omega, power_theoretical, 'r--', label='Theoretical PSD')
plt.plot(omega, power_est, 'b', label='Estimated PSD')
plt.title('Power Spectral Density Rₓₓ(ω)')
plt.xlabel('Frequency ω [rad/sample]')
plt.ylabel('Normalized magnitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
# 2C

# 2D
K_values = [10, 100]
plt.plot(omega / (2 * np.pi), power_theoretical / np.max(power_theoretical), 'k--', label='Theoretical PSD')

freqs, Pxx_periodogram = sp.signal.welch(x, nperseg=len(x), noverlap=0)
plt.semilogy(freqs, Pxx_periodogram / np.max(Pxx_periodogram), 'gray', alpha=0.5, label='Periodogram')

for K in K_values:
    nperseg = len(x) // K
    freqs, Pxx_bartlett = sp.signal.welch(x, nperseg=nperseg, noverlap=0)
    plt.semilogy(freqs, Pxx_bartlett / np.max(Pxx_bartlett), label=f'Bartlett K={K}')

plt.title('Bartlett PSD Estimates vs. Theoretical PSD')
plt.xlabel('Normalized Frequency')
plt.ylabel('Normalized Power')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# 2D

# 2E
simulations = 5
Ke = 75

plt.figure()
for i in range(simulations):
    we = np.sqrt(sigma_w) * np.random.randn(N)
    x = sp.signal.lfilter(b, a, we)
    freqs, psd = sp.signal.welch(x, nperseg=N//K, noverlap=0)
    psd /= np.max(psd)
    plt.plot(freqs, psd, alpha=0.6, label=f"simulation {i+1}")

plt.plot(omega / (2 * np.pi), power_theoretical / np.max(power_theoretical), 'k--', label='Theoretical PSD')
plt.title(f"Bartlett PSD Estimates {simulations} simulations (K={K})")
plt.xlabel("Normalized f")
plt.ylabel("Normalized Power")
plt.legend()
plt.grid()

plt.show()
# 2E