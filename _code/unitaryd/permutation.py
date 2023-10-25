import itertools as it
import numpy as np

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
    
if __name__ == "__main__":
    permutation = Permutation(2)
    print(permutation.get_permutation_matrix([2, 1]))