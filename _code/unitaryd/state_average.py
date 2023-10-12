import pennylane as qml

from scipy.stats import unitary_group as ug

def swap_test(random_unitary):
    n_shots = 50_000
    dev = qml.device(
        "default.qubit",
        wires = 3,
        shots = n_shots
    )

    @qml.qnode(dev)
    def swap_test_circuit():
        qml.QubitUnitary(random_unitary, wires = 1)
        qml.QubitUnitary(random_unitary, wires = 2)
        qml.PauliX(wires = 2)
        qml.Hadamard(wires = 0)
        qml.CSWAP(wires = [0, 1, 2])
        qml.Hadamard(wires = 0)
        return qml.counts(qml.PauliZ(0))

    dist = swap_test_circuit()
    one_state_count = dist[-1] if -1 in dist else 0
    return 1 - (2 / n_shots) * one_state_count

def monte_carlo_average(f, sample_size):
    total = 0

    for _ in range(sample_size):
        random_unitary = ug.rvs(2)
        total += f(random_unitary)

    return total / sample_size

if __name__ == "__main__":
    print(monte_carlo_average(swap_test, 5_000))
