import numpy as np
import pennylane as qml

def zero_ev(wire):
    # This non-circuit prepares the |0> state
    pass

def one_ev(wire):
    # This circuit prepares the |1> state
    qml.PauliX(wires = wire)

def plus_ev(wire):
    # This circuit prepares the |+> state
    qml.Hadamard(wires = wire)

def minus_ev(wire):
    # This circuit prepares the |-> state
    qml.PauliX(wires = wire)
    qml.Hadamard(wires = wire)

def plus_i_ev(wire):
    # This circuit prepares the |+i> state
    qml.Hadamard(wires = wire)
    qml.S(wires = wire)

def minus_i_ev(wire):
    # This circuit prepares the |-i> state
    qml.PauliX(wires = wire)
    qml.Hadamard(wires = wire)
    qml.S(wires = wire)

def swap_test(state_prep_gate):
    n_shots = 50_000
    dev = qml.device(
        "default.qubit",
        wires = 3,
        shots = n_shots
    )

    @qml.qnode(dev)
    def swap_test_circuit():
        # Prepare the state |psi> on qubit 1
        state_prep_gate(1)
        
        # Prepare the state X|psi> on qubit 2
        state_prep_gate(2)
        qml.PauliX(wires = 2)

        # Perform the SWAP test
        qml.Hadamard(wires = 0)
        qml.CSWAP(wires = [0, 1, 2])
        qml.Hadamard(wires = 0)

        # Making sure to collect statistics of qubit 0
        return qml.counts(qml.PauliZ(0))

    dist = swap_test_circuit()
    one_state_count = dist[-1] if -1 in dist else 0
    return 1 - (2 / n_shots) * one_state_count

def state_design_average(f, states):
    return np.mean([f(state) for state in states])

if __name__ == "__main__":
    print(state_design_average(
        swap_test,
        [zero_ev, one_ev, plus_ev, minus_ev, plus_i_ev, minus_i_ev]
    ))
