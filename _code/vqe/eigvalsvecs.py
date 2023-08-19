import numpy as np
import numpy.linalg as la

if __name__ == "__main__":
    H = np.matrix([
        [1,  0,  0,  0],
        [0,  1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0, -1]
    ])

    eigvals, eigvecs = la.eig(H)
    print(eigvals)
    print(eigvecs)