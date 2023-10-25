import numpy as np

from collections import deque

# Limit the number of decimal digits to 2
np.set_printoptions(precision = 2, suppress = True)

def matrix_in_list(element, list):
    for list_element in list:
        if np.allclose(list_element, element):
            return True
    return False

class Pauli:
    @staticmethod
    def generators():
        X = np.matrix([
            [0, 1],
            [1, 0]
        ])
        Y = np.matrix([
            [ 0, -1j],
            [1j,   0]
        ])
        Z = np.matrix([
            [1,  0],
            [0, -1]
        ])

        return [X, Y, Z]

    @staticmethod
    def group():
        group = []
        queue = deque()
        queue.append(
            np.matrix([
                [1, 0],
                [0, 1]
            ])
        )

        while queue:
            x = queue.popleft()
            if matrix_in_list(x, group):
                continue

            group.append(x)
            for generator in Pauli.generators():
                queue.append(x @ generator)
    
        return group

def unitary_design_average(M, t_design):
    R = np.zeros(M.shape)
    for U in t_design:
        R = R + (U @ M @ U.conj().T)
    return (1 / len(t_design)) * R

if __name__ == "__main__":
    S = np.matrix([
        [1,  0],
        [0, 1j]
    ])
    print(unitary_design_average(S, Pauli.group()))
