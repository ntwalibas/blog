import pennylane as qml

dev = qml.device(
    "default.qubit",
    wires = 1,
    shots = 100000
)

@qml.qnode(dev)
def expval():
    qml.PauliX(wires = 0)
    qml.Hadamard(wires = 0)
    return qml.expval(qml.PauliX(0))

if __name__ == "__main__":
    print(expval())
