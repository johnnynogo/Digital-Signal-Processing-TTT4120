import numpy as np
import matplotlib.pyplot as plt

a = -0.9 # values: 0.5, 0.9, -0.9
n = np.arange(0, 51, 1)
l = np.arange(-50, 51, 1)
# n = np.linspace(0, 50, 1000)
# l = np.linspace(-50, 50, 1000)
f = np.linspace(-0.5, 0.5, 1000)

# def x(n):
#     return a**n

x = []
for i in n:
    x.append(a**i)
# print(x)

# def rxx(l):
#     return (a**abs(l)) / (1-a**2)
rxx = []
for i in l:
    rxx.append((a**abs(i)) / (1-a**2))

def Sxx(f):
    return 1 / (1 + a**2 - 2*a*np.cos(2*np.pi*f))

plt.figure()
plt.stem(n, x, label=fr"a={a}")
plt.title("x[n]")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.legend()
plt.grid()

plt.figure()
# plt.plot(l, rxx(l), label=fr"$a = {a}$")
plt.stem(l, rxx, label=fr"a={a}")
plt.title(r"$r_{xx}$(l)")
plt.xlabel("l")
plt.ylabel(r"$r_{xx}$(l)")
plt.legend()
plt.grid()

plt.figure()
plt.plot(f, Sxx(f), label=fr"$a = {a}$")
plt.title(r"$S_{xx}$(f)")
plt.xlabel("f")
plt.ylabel(r"$S_{xx}$(f)")
plt.legend()
plt.grid()

plt.show()
