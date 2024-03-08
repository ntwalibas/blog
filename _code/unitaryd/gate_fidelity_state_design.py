import numpy as np
import pennylane as qml


def zero(wire):
    # This non-circuit prepares the |0> state
    pass

def one(wire):
    # This circuit prepares the |1> state
    qml.PauliX(wires = wire)

def plus(wire):
    # This circuit prepares the |+> state
    qml.Hadamard(wires = wire)

def minus(wire):
    # This circuit prepares the |-> state
    qml.PauliX(wires = wire)
    qml.Hadamard(wires = wire)

def plus_i(wire):
    # This circuit prepares the |+i> state
    qml.Hadamard(wires = wire)
    qml.S(wires = wire)

def minus_i(wire):
    # This circuit prepares the |-i> state
    qml.PauliX(wires = wire)
    qml.Hadamard(wires = wire)
    qml.S(wires = wire)

def PauliX_e(angle, wire):
    qml.RX(np.pi + angle, wires = wire)

def swap_test(state_prep_gate, calibration_error_angle):
    n_shots = 50_000
    dev = qml.device(
        "default.qubit",
        wires = 3,
        shots = n_shots
    )

    @qml.qnode(dev)
    def swap_test_circuit():
        # Prepare the state X_e|psi> on qubit 1
        state_prep_gate(1)
        PauliX_e(calibration_error_angle, 1)
        
        # Prepare the state X|psi> on qubit 2
        state_prep_gate(2)
        qml.PauliX(wires = 2)

        # Perform the SWAP test
        qml.Hadamard(wires = 0)
        qml.CSWAP(wires = [0, 1, 2])
        qml.Hadamard(wires = 0)

        return qml.counts(qml.PauliZ(0))

    dist = swap_test_circuit()
    one_state_count = dist[-1] if -1 in dist else 0
    return 1 - (2 / n_shots) * one_state_count

def state_design_average(f, calibration_error_angle, states):
    return np.mean([f(state, calibration_error_angle) for state in states])

if __name__ == "__main__":
    """
    Ideally the calibration error angles will be random.
    We use deterministic angles of increasing value to demonstrate
    that the average fidelity decreases as the error angle increases.
    """
    calibration_error_angles = [0, np.pi/2, np.pi]
    for calibration_error_angle in calibration_error_angles:
        print(f"Fidelity at angle error {calibration_error_angle} =",
            state_design_average(
                swap_test,
                calibration_error_angle,
                [zero, one, plus, minus, plus_i, minus_i]
            )
        )
