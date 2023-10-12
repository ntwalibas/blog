import pennylane as qml

from scipy.stats import unitary_group as ug

def swap_test(state_prep_unitary):
    n_shots = 50_000
    dev = qml.device(
        "default.qubit",
        wires = 3,
        shots = n_shots
    )

    @qml.qnode(dev)
    def swap_test_circuit():
        # Prepare the state |psi> on qubit 1
        qml.QubitUnitary(state_prep_unitary, wires = 1)

        # Prepare the state X|psi> on qubit 2
        qml.QubitUnitary(state_prep_unitary, wires = 2)
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

def monte_carlo_average(f, sample_size):
    total = 0

    for _ in range(sample_size):
        # ug.rvs will sample a random 2x2 matrix from the unitary group
        total += f(ug.rvs(2))

    return total / sample_size

if __name__ == "__main__":
    print(monte_carlo_average(swap_test, 5_000))
