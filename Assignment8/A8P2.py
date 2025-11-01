import numpy as np
from scipy.linalg import toeplitz, solve_toeplitz
import matplotlib.pyplot as plt

# 2C
r = np.array([1.25, -0.5, 0, 0])

def yule_walker(r, p):
    a = solve_toeplitz((r[:p], r[:p]), -r[1:p+1])
    sigma2 = r[0] + np.dot(a, r[1:p+1])
    return a, sigma2

for p in (1, 2, 3):
    a, sigma2 = yule_walker(r, p)
    print(f"AR({p}) coeffs a_1..a_{p} = {a},   sigma_f^2 = {sigma2}")
# 2C

# 2D
# a, sigma2 = yule_walker(r, 3)
# z = np.linspace(-10, 10, 1000)

# A = []
# sub_list = []
# for i in range(len(a)):
#     sub_list.append(1 + np.sum(a[i] * z**(-np.arange(1, i+1))))
#     A.append(sub_list)
#     sub_list = []

# PSD = []
# for j in range(len(a)):
#     sub_list.append(sigma2[i] / np.abs(A[i])**2)
#     PSD.append(sub_list)
#     sub_list = []

# for k in range(len(PSD)):
#     plt.figure()
#     plt.plot(z, PSD[k])
#     plt.grid()

# plt.show()

def A_of_omega(omega, a):
    k = np.arange(1, len(a)+1)[:, None]
    return 1 + np.sum(a[:, None] * np.exp(-1j * k * omega[None, :]), axis=0)

def psd_ar(omega, a, sigma2):
    A = A_of_omega(omega, a)
    return sigma2 / np.abs(A)**2

omega = np.linspace(-np.pi, np.pi, 1024)

plt.figure(figsize=(8,5))
for p in (1, 2, 3):
    a, sigma2 = yule_walker(r, p)
    P = psd_ar(omega, a, sigma2)
    plt.plot(omega/np.pi, P, label=f'AR({p})  a={np.round(a,4)}  σ_f²={sigma2:.4f}')

plt.plot(omega/np.pi, 1.25 - np.cos(omega), label=f"MA(1)")
plt.title('AR model PSDs from Yule–Walker')
plt.xlabel('Normalized frequency (×π rad/sample)')
plt.ylabel('Power')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# 2D
