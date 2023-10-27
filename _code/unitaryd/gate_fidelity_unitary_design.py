import numpy as np
import pennylane as qml
import scipy.linalg as la

from collections import deque

# Make sure results are reproducible
np.random.seed(1)

def matrix_in_list(element, list):
    for list_element in list:
        if np.allclose(list_element, element):
            return True
    return False

def matrix_is_normalizer(x):
    pauli_group = Pauli.group()
    for pauli in pauli_group:
        p = x @ pauli @ x.conj().T
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
        group = Pauli.generators()
        group.append(
            np.matrix([
                [1, 0],
                [0, 1]
            ])
        )
    
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

def PauliX_e(angle, wire):
    # We add a randomly generated angle to simulate
    # stochastic calibration noise
    qml.RX(np.pi + angle, wires = wire)

def swap_test(state_prep_unitary, calibration_error_angle):
    n_shots = 50_000
    dev = qml.device(
        "default.qubit",
        wires = 3,
        shots = n_shots
    )

    @qml.qnode(dev)
    def swap_test_circuit():
        # Prepare the state X_e|psi> on qubit 1
        qml.QubitUnitary(state_prep_unitary, wires = 1)
        PauliX_e(calibration_error_angle, 1)

        # Prepare the state X|psi> on qubit 2
        qml.QubitUnitary(state_prep_unitary, wires = 2)
        qml.PauliX(wires = 2)

        # Perform the SWAP test
        qml.Hadamard(wires = 0)
        qml.CSWAP(wires = [0, 1, 2])
        qml.Hadamard(wires = 0)

        # Collect counts on qubit 0
        return qml.counts(qml.PauliZ(0))

    dist = swap_test_circuit()
    one_state_count = dist[-1] if -1 in dist else 0
    return 1 - (2 / n_shots) * one_state_count

def unitary_design_average(f, calibration_error_angle, unitaries):
    return np.mean([f(unitary, calibration_error_angle) for unitary in unitaries])

if __name__ == "__main__":
    calibration_error_angles = [0, np.pi / 2, np.pi]
    group = Pauli.group()
    for calibration_error_angle in calibration_error_angles:
        print(f"Fidelity at angle error {calibration_error_angle} =",
            unitary_design_average(
                swap_test,
                np.random.normal(0, calibration_error_angle, 1)[0],
                group
            )
        )
