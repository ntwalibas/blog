import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt

dev = qml.device(
    "default.qubit",
    wires = 2,
    shots = 100000
)

def ansatz(params):
    qml.RY(params[0], wires = 1)
    qml.PhaseShift(params[3], wires = 1)
    qml.CRY(params[1], wires = [1, 0])
    qml.ControlledPhaseShift(params[5] - params[3], wires = [0, 1])
    qml.CRY(params[2], wires = [0, 1])
    qml.PauliX(wires = 1)
    qml.ControlledPhaseShift(params[4] - params[5], wires = [1, 0])
    qml.PauliX(wires = 1)

@qml.qnode(dev)
def xz_cost(params):
    ansatz(params)
    return qml.expval(qml.PauliX(0) @ qml.PauliZ(1))

@qml.qnode(dev)
def iz_cost(params):
    ansatz(params)
    return qml.expval(qml.PauliZ(1))

@qml.qnode(dev)
def xz_iz_cost(params):
    ansatz(params)
    h = qml.PauliX(0) @ qml.PauliZ(1) + qml.PauliZ(1)
    return qml.expval(h)

def vqe(cost, params, maxiter):
    optimizer = qml.SPSAOptimizer(maxiter = maxiter)
    energy = cost(params)
    history = [energy]

    for iter in range(maxiter):
        params, energy = optimizer.step_and_cost(
            cost,
            params
        )

        # Print the optimizer progress every 10 steps
        if iter % 20 == 0:
            print(f"Step = {iter},  Energy = {history[-1]:.8f}")
        
        # Save the full energy optimization history
        history.append(energy)
    
    return energy, history

if __name__ == "__main__":
    # Initialize params from the normal distribution with mean 0 and variance np.pi
    init_params = np.random.normal(0, np.pi, 6)
    
    # We try 151 iterations
    maxiter = 401

    # Run VQE
    # xz_energy, xz_history = vqe(xz_cost, init_params, maxiter)
    # iz_energy, iz_history = vqe(iz_cost, init_params, maxiter)
    energy, history = vqe(xz_iz_cost, init_params, maxiter)

    # Print the final energy
    # print(xz_energy)
    # print(iz_energy)
    # print(xz_energy + iz_energy)
    print(energy)

    # Plot the optimization history
    plt.figure(figsize=(10, 6))
    plt.plot(range(maxiter + 1), history, "go", ls = "dashed", label = "Energy")
    plt.xlabel("Optimization step", fontsize=13)
    plt.ylabel("Energy", fontsize=13)
    plt.show()
