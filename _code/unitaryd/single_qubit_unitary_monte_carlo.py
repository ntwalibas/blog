import numpy as np

from scipy.stats import unitary_group as ug

# Limit the number of decimal digits to 2
np.set_printoptions(precision = 2, suppress = True)

def monte_carlo_average(M, n_samples):
    R = np.zeros(M.shape)
    for _ in range(n_samples):
        U = ug.rvs(2)
        R = R + (U @ M @ U.conj().T)
    return (1 / n_samples) * R

if __name__ == "__main__":
    S = np.matrix([
        [1, 0],
        [0, 1j]
    ])
    print(monte_carlo_average(S, 50_000))
