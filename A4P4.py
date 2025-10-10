import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# ==========
# PROBLEM 4A
# ==========

Ax = 0.25
Ay = 0.25
fx = 0.04
fy = 0.10
L = 500
N = 2048
n = np.linspace(0, L-1, L)
e = np.random.normal(size=L)

def d(n):
    return Ax * np.cos(2*np.pi*fx*n) + Ay * np.cos(2*np.pi*fy*n)

def g(n):
    return d(n) + e

fig1, (ax1, ax2) = plt.subplots(2)
ax1.set_title("Plot of sequences d[n] and g[n]")
ax1.plot(n, d(n))
ax1.grid()
ax2.plot(n, g(n))
ax2.grid()

D = np.fft.rfft(d(n), N)
G = np.fft.rfft(g(n), N)
freqs = np.fft.rfftfreq(N, d=1)

fig2, (ax3, ax4) = plt.subplots(2)
ax3.set_title("Plot of magnitude spectra [D(f)] and [G(f)]")
ax3.plot(freqs, abs(D))
ax3.grid()
ax4.plot(freqs, abs(G))
ax4.grid()
plt.show()

# =======================
# PLOT OF POLES AND ZEROS
# =======================
b = [1, 0, -1]
a_x = [1, -2*0.99*np.cos(2*np.pi*fx), 0.99*0.99]
a_y = [1, -2*0.99*np.cos(2*np.pi*fy), 0.99*0.99]

zeros = np.roots(b)
poles_x = np.roots(a_x)
poles_y = np.roots(a_y)
theta = np.linspace(0, 2*np.pi)

fig, ax = plt.subplots()
ax.plot(np.cos(theta), np.sin(theta), '--k')
ax.scatter(np.real(poles_x), np.imag(poles_x), marker='x', label='poles_x')
ax.scatter(np.real(poles_y), np.imag(poles_y), marker='x', label='poles_y')
ax.scatter(np.real(zeros), np.imag(zeros), marker='o', label='zeros')
ax.set_title("Imaginary plane plot")
ax.set_ylabel("Im{}")
ax.set_xlabel("Re{}")
ax.hlines(0, -1, 1)
ax.vlines(0, -1, 1)
ax.legend(loc='upper right')
ax.grid()

# ===========================
# PLOTS OF MAGNITUDE RESPONSE
# ===========================
w_x, h_x = sp.signal.freqz(b, a_x)
w_y, h_y = sp.signal.freqz(b, a_y)

fig, ax1 = plt.subplots()
ax1.plot(w_x/(2*np.pi), abs(h_x), label=r'H_x(f)')
ax1.plot(w_y/(2*np.pi), abs(h_y), label=r'H_y(f)')
ax1.vlines(0.04, 0, 100)
ax1.set_title("magnitude response")
ax1.legend()
ax1.grid()
plt.show()

# ===================================
# FILTERING NOISE CONTAMINATED SIGNAL
# ===================================
qx = sp.signal.lfilter(b, a_x, g(n))
qy = sp.signal.lfilter(b, a_y, g(n))

fig, (fax1, fax2) = plt.subplots(2)
fax1.set_title("outputs from filters: qx and qy")
fax1.plot(n, qx)
fax1.grid()

fax2.plot(n, qy)
fax2.grid()

QX = np.fft.rfft(qx, N)
QY = np.fft.rfft(qy, N)
freqs1 = np.fft.rfftfreq(N, d=1)

fig, (ax3, ax4) = plt.subplots(2)
ax3.set_title("Plot of magnitude spectra [Qx(f)] and [Qy(f)]")
ax3.plot(freqs1, abs(QX))
ax3.grid()
ax4.plot(freqs1, abs(QY))
ax4.grid()
plt.show()


# =================
# COMBINING FILTERS
# =================
h = h_x + h_y
w = w_x

num = np.polyadd(np.polymul(b, a_y), np.polymul(b, a_x))
den = np.polymul(a_x, a_y)

zeros = np.roots(num)
poles = np.roots(den)
print(f"Poles: {poles}")
print(f"Zeros: {zeros}")

fig, ax1 = plt.subplots()
ax1.plot(w/(2*np.pi), abs(h))
ax1.set_title("amplitude response combined filter")
ax1.grid()
plt.show()

fig, ax2 = plt.subplots()
ax2.set_title("pole-zero plot of combined filter")
ax2.set_xlabel("Re{}")
ax2.set_ylabel("Im{}")
ax2.plot(np.cos(theta), np.sin(theta), '--k')
ax2.scatter(np.real(zeros), np.imag(zeros), marker = 'o')
ax2.scatter(np.real(poles), np.imag(poles), marker = 'x')
ax2.grid()
plt.show()

q = qx+qy
fig, ax3 = plt.subplots()
ax3.set_title("output of combined filter")
ax3.plot(n, q)
plt.show()


Q = QX + QY
fig, ax4 = plt.subplots()
ax4.set_title("magnitude of combined filter")
ax4.plot(freqs, abs(Q))
plt.show()
