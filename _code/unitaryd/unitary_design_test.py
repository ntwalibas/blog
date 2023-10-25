import numpy as np
import scipy.linalg as la
import itertools as it

from collections import deque
from typing import List, Tuple

class Permutation:
    def __init__(self, n_qubits: int):
        if n_qubits < 1:
            raise ValueError("The number of qubits must be at least 1")
        self.n_qubits = n_qubits
    
    def get_basis_set(self):
        single_qubit_basis = ['0', '1']
        new_basis = single_qubit_basis
        temp_basis = []
        for _ in range(self.n_qubits - 1):
            for r in it.product(new_basis, single_qubit_basis):
                temp_basis.append(r[0] + r[1])
            new_basis = temp_basis
            temp_basis = []
        return new_basis
    
    def get_permutation_list(self, order: List) -> List[Tuple]:
        if len(order) != self.n_qubits:
            raise ValueError("The order list must have the same length as the number of qubits to permute")

        if len(set(order)) != self.n_qubits:
            raise ValueError("The order list cannot contain duplicate items")
        
        permutations = []
        basis = self.get_basis_set()
        
        for basis_element in basis:
            new_element = list(basis_element)
            old_element = list(basis_element)
            for old_index, new_index in enumerate(order):
                if new_index < 1 or new_index > self.n_qubits:
                    raise ValueError(f"Item {new_index} at index {old_index} in order list is out of bounds")
                new_element[old_index] = old_element[new_index - 1]
            permutations.append((basis_element, ''.join(new_element)))
            
        return permutations

    def get_permutation_matrix(self, order: List) -> np.ndarray:
        permutations = self.get_permutation_list(order)
        dim = len(permutations)
        permutation_matrix = np.zeros((dim, dim))
        
        for perm in permutations:
            x = int(perm[0], 2)
            y = int(perm[1], 2)
            permutation_matrix[x][y] = 1
        
        return permutation_matrix

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

def is_unitary_1_design(group):
    R = np.asmatrix(np.zeros((4,4)))

    for element in group:
        element_dagger = element.H
        R = R + (np.kron(element, element_dagger))
    
    F = (1 / len(group)) * R
    P_21 = Permutation(2).get_permutation_matrix([2, 1])
    return np.allclose(F, P_21 / 2)

if __name__ == "__main__":
    print(is_unitary_1_design(Pauli.group()))
