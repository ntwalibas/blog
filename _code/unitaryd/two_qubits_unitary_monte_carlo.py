import numpy as np

from scipy.stats import unitary_group as ug

# Limit the number of decimal digits to 2
np.set_printoptions(precision = 2, suppress = True)

def monte_carlo_average(M, n_samples):
    R = np.zeros(M.shape)
    for _ in range(n_samples):
        U_i = ug.rvs(2)
        for _ in range(n_samples):
            U_j = ug.rvs(2)
            R = R + (np.kron(U_i, U_j) @ M @ np.kron(U_i, U_j).conj().T)
    return (1 / n_samples**2) * R

if __name__ == "__main__":
    CNOT = np.matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ])
    # We know the resulting matrix is real
    # so we make sure to take only the real parts of each entry
    print(monte_carlo_average(CNOT, 1_000).real)
