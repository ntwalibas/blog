import pennylane as qml
from pennylane import numpy as np

dev = qml.device(
    "default.qubit",
    wires = 1,
    shots = 10000
)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires = 0)
    qml.S(wires = 0)
    return qml.counts(qml.PauliY(0))


if __name__ == "__main__":
    results = circuit()
    print(results)
