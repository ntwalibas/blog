import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

dev = qml.device(
    "default.qubit",
    wires = 1,
    shots = 100000
)

@qml.qnode(dev)
def hadamard_cost(theta):
    qml.RY(theta[0], wires = 0)
    qml.RX(theta[1], wires = 0)
    qml.RY(theta[2], wires = 0)
    return qml.expval(qml.Hadamard(0))

def hadamard_vqe(cost, theta, maxiter):
    optimizer = qml.SPSAOptimizer(maxiter = maxiter)
    energy = cost(theta)
    history = [energy]

    for iter in range(maxiter):
        theta, energy = optimizer.step_and_cost(
            cost,
            theta
        )

        # Print the optimizer progress every 10 steps
        if iter % 10 == 0:
            print(f"Step = {iter},  Energy = {history[-1]:.8f}")
        
        # Save the full energy optimization history
        history.append(energy)
    
    return energy, history

if __name__ == "__main__":
    # Initialize theta from the normal distribution with mean 0 and variance np.pi
    init_theta = np.random.normal(0, np.pi, 3)
    
    # We try 100 iterations
    maxiter = 151

    # Run VQE
    energy, history = hadamard_vqe(hadamard_cost, init_theta, maxiter)

    # Print the final energy
    print(energy)

    # Plot the optimization history
    plt.figure(figsize=(10, 6))
    plt.plot(range(maxiter + 1), history, "go", ls = "dashed", label = "Energy")
    plt.xlabel("Optimization step", fontsize=13)
    plt.ylabel("Energy", fontsize=13)
    plt.show()
