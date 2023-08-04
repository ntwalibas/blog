import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt

dev = qml.device(
    "default.qubit",
    wires = 1,
    shots = 10000
)

def one_qubit():
    qml.Hadamard(wires = 0)
    qml.PauliX(wires = 0)
    qml.Hadamard(wires = 0)
    return qml.counts(qml.PauliX(0))

def two_qubits_1(theta_1, theta_2, theta_3, phi_1, phi_2, phi_3):
    qml.RY(theta_1, wires = 1)
    qml.PhaseShift(phi_1, wires = 1)

    qml.CRY(theta_2, wires = [1, 0])
    qml.PhaseShift(phi_3 - phi_1, wires = 0)

    qml.CRY(theta_3, wires = [0, 1])
    
    qml.PauliX(wires = 0)
    qml.PhaseShift(phi_2 - phi_3, wires = 0)
    qml.PauliX(wires = 0)

    return qml.counts(qml.PauliZ(0) @ qml.PauliZ(1))

def two_qubits_2(theta_1, theta_2, theta_3, phi_1, phi_2, phi_3):
    qml.RY(theta_1, wires = 1)
    qml.PhaseShift(phi_1, wires = 1)

    qml.CRY(theta_2, wires = [1, 0])

    qml.ControlledPhaseShift(phi_3 - phi_1, wires = [0, 1])

    qml.CRY(theta_3, wires = [0, 1])

    qml.PauliX(wires = 1)
    qml.ControlledPhaseShift(phi_2 - phi_3, wires = [1, 0])
    qml.PauliX(wires = 1)

    return qml.counts(qml.PauliZ(0) @ qml.PauliZ(1))

if __name__ == "__main__":
    angles = np.random.normal(0, np.pi, 6)

    compiled_circuit = qml.compile(
        basis_set = ["CNOT", "RY", "RZ"],
        num_passes = 5
    )(two_qubits_2)
    qnode = qml.QNode(compiled_circuit, dev)

    qml.draw_mpl(qnode, decimals = 1, style = "sketch")(*angles)
    plt.show()
