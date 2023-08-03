import pennylane as qml

dev = qml.device(
    "default.qubit",
    wires = 2,
    shots = 100000
)

@qml.qnode(dev)
def xz_expval():
    qml.Hadamard(wires = 0)
    qml.PauliX(wires = 1)
    return qml.expval(qml.PauliX(0) @ qml.PauliZ(1))

@qml.qnode(dev)
def zi_expval():
    qml.Hadamard(wires = 0)
    qml.PauliX(wires = 1)
    return qml.expval(qml.PauliZ(1))

def h_expval():
    return xz_expval() + zi_expval()

if __name__ == "__main__":
    print(h_expval())
