import pennylane as qml
from pennylane import numpy as np

dev = qml.device(
    "default.qubit",
    wires = 2,
    shots = 10000
)

@qml.qnode(dev)
def circuit():
    qml.PauliX(wires = 0)
    qml.Hadamard(wires = 0)
    qml.PauliX(wires = 1)
    return qml.counts(qml.PauliX(0)), qml.counts(qml.PauliZ(1))
    return qml.counts()
    
    # return qml.expval(qml.PauliX(0))
    # return qml.expval(qml.PauliZ(1))

@qml.qnode(dev)
def circuit2():
    qml.PauliX(wires = 0)
    qml.PauliX(wires = 1)
    qml.Hadamard(wires = 1)
    return qml.counts(qml.PauliZ(0) @ qml.PauliX(1))

@qml.qnode(dev)
def circuit2p():
    qml.PauliX(wires = 0)
    qml.PauliX(wires = 1)
    qml.Hadamard(wires = 1)
    qml.Hadamard(wires = 1)
    return qml.counts(qml.PauliZ(0) @ qml.PauliZ(1))

@qml.qnode(dev)
def circuit3():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.counts(qml.PauliZ(0)), qml.counts(qml.PauliZ(1))

@qml.qnode(dev)
def circuit4():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.counts(qml.PauliZ(0) @ qml.PauliZ(1))

@qml.qnode(dev)
def circuit5():
    qml.PauliX(wires = 0)
    qml.Hadamard(wires = 0)
    return qml.counts(qml.PauliX(0) @ qml.PauliZ(1))

@qml.qnode(dev)
def circuit6():
    qml.PauliX(wires = 0)
    # qml.Hadamard(wires = 0)
    return qml.counts(qml.PauliZ(0) @ qml.PauliZ(1))

if __name__ == "__main__":
    results = circuit6()
    print(results)
