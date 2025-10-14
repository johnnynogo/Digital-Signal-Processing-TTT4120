import numpy as np
import matplotlib.pyplot as plt

# P1A START
f = np.linspace(0, 1, 1000)

def X(f):
    return (1 - (0.9*np.exp(-1j*2*np.pi*f))**28) / (1- 0.9*np.exp(-1j*2*np.pi*f))

# plt.plot(f, np.abs(X(f)), label="|X(f)|")
# plt.title("|X(f)|")
# plt.grid()

# plt.show()
# P1A END

# P1B/D START

Nx = 28
n = np.arange(Nx)
x = 0.9**n
fft_values = np.fft.fft(x)

print(fft_values)
print(len(fft_values))
print()


N = [Nx//4, Nx//2, Nx, 2*Nx]

for i in N:
    Xk = np.fft.fft(x, n=i)
    print(f"For Nx = {i}: {Xk}")
    print()

    # k = f*i
    k = np.arange(i) / i

    plt.stem(k, Xk, label=f"for Nx = {i}")
    plt.plot(f, np.abs(X(f)), label=f"X(f) for f = {f}")
    plt.title(f"for Nx = {i}")
    plt.xlabel("k")
    plt.ylabel("")
    plt.grid()

    plt.show()

# P1B/D END