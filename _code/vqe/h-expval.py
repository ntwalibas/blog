import pennylane as qml
from pennylane import numpy as np

# We fix the seed to make results reproducible
np.random.seed(1)

dev = qml.device(
    "default.qubit",
    wires = 1,
    # We request the exact expectation value by not setting shots
    shots = None
)

@qml.qnode(dev)
def x_expval(y):
    qml.RY(y, wires = 0)
    return qml.expval(qml.PauliX(0))

@qml.qnode(dev)
def z_expval(y):
    qml.RY(y, wires = 0)
    return qml.expval(qml.PauliZ(0))

def h_expval(y):
    return (1/np.sqrt(2)) * (x_expval(y) + z_expval(y))

@qml.qnode(dev)
def hadamard_expval(y):
    qml.RY(y, wires = 0)
    return qml.expval(qml.Hadamard(0))

if __name__ == "__main__":
    custom_expval = h_expval(np.pi)
    builtin_expval = hadamard_expval(np.pi)
    print(custom_expval)
    print(builtin_expval)
    print(custom_expval == builtin_expval)
