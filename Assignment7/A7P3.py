import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

N = 20000
sigma_w2 = 3/4
a = 0.5
K_values = [20, 40, 100]
num_means = 200
a_filter=[1, 0.5]
b_filter=[1]

w = np.sqrt(sigma_w2) * np.random.randn(N)
x = sp.signal.lfilter(b_filter, a_filter, w)

plt.figure()

for i, K in enumerate(K_values):
    x_segment = x[:num_means * K].reshape(num_means, K)
    mean_estimates = np.mean(x_segment, axis=1)
    mean_of_means = np.mean(mean_estimates)
    var_of_means = np.var(mean_estimates)

    plt.subplot(3, 1, i + 1)
    plt.hist(mean_estimates, bins=20, color='blue', edgecolor='black')
    plt.xlim([-0.5, 0.5])
    plt.ylim([0, 40])
    plt.title(f'Histogram of Mean Estimates (K = {K})')
    plt.xlabel('Mean Estimate')
    plt.ylabel('Count')

    plt.text(0.05, 35, f'Mean = {mean_of_means:.4f}\nVar = {var_of_means:.5f}')

plt.tight_layout()
plt.show()
