import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

alpha = 0.9
K = 1
# H(z) = B(z)/A(z)
b_u = [1.9, -1.9]
a_u = [2, -1.8]
b_l = [0.1, 0.1]
a_l = [2, -1.8]

# scipy.freqz calulates freq. response of digital filter given coefficients
# returns w (discrete sample freqs. from [0,pi])
# and h (freq.resp. as complext numbers, evaluated with freqs. from w)
w_u, h_u = sp.signal.freqz(b_u, a_u)
w_l, h_l = sp.signal.freqz(b_l, a_l)
w = w_u
h = h_u + h_l
# print(f"w = {w}")
# print(f"h = {h}")

fig, (ax1, ax2, ax3) = plt.subplots(3)

ax1.set_title("Magnitude response of upper&lower branch and entire filter")
ax1.plot(w_u, abs(h_u))
ax1.grid()

ax2.plot(w_l, abs(h_l))
ax2.grid()

ax3.plot(w, abs(h))
ax3.set_ylim(0, 1.1)
ax3.grid()

plt.savefig("magnitude_responses.png")
plt.show()
