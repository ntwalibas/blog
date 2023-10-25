import numpy as np
import scipy.linalg as la

from collections import deque

def matrix_in_list(element, list):
    for list_element in list:
        if np.allclose(list_element, element):
            return True
    return False

def matrix_is_normalizer(x):
    pauli_group = Pauli.group()
    for pauli in pauli_group:
        p = x @ pauli @ x.conj().H
        if matrix_in_list(p, pauli_group):
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

class Clifford:
    @staticmethod
    def generators():
        H = (1/np.sqrt(2)) * np.matrix([
            [1,  1],
            [1, -1]
        ], dtype = complex)

        S = np.matrix([
            [1,  0],
            [0, 1j]
        ], dtype = complex)

        return [H, S]

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
            global_phase = 1 / np.emath.sqrt(la.det(x))
            x = x * global_phase

            # We need to account for a matrix with a negated phase
            if matrix_in_list(x, group) or matrix_in_list(-x, group):
                continue

            if matrix_is_normalizer(x):
                group.append(x)
                for clifford in Clifford.generators():
                    queue.append(x @ clifford)
    
        return group

if __name__ == "__main__":
    from pprint import pprint
    pprint(len(Clifford.group()))
