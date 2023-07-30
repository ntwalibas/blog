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

@qml.qnode(dev)
def custom_circuit():
    # Prepare the state
    qml.Hadamard(wires = 0)
    qml.S(wires = 0)
    
    # Perform a change of basis
    qml.adjoint(qml.S(wires = 0))
    qml.Hadamard(wires = 0)

    # Measure in standard basis
    return qml.counts(qml.PauliZ(0))

if __name__ == "__main__":
    results = custom_circuit()
    print(results)
