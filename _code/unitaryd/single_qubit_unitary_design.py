import numpy as np

class Pauli:
    @staticmethod
    def generators():
        X = np.matrix([
            [0, 1],
            [1, 0]
        ])
        Y = np.matrix([
            [ 0, -1j],
            [1j,   0]
        ])
        Z = np.matrix([
            [1,  0],
            [0, -1]
        ])

        return [X, Y, Z]

    @staticmethod
    def group():
        group = Pauli.generators()
        group.append(
            np.matrix([
                [1, 0],
                [0, 1]
            ])
        )
    
        return group

def unitary_design_average(M, t_design):
    R = np.zeros(M.shape)
    for U in t_design:
        R = R + (U @ M @ U.conj().T)
    return (1 / len(t_design)) * R

if __name__ == "__main__":
    S = np.matrix([
        [1,  0],
        [0, 1j]
    ])
    print(unitary_design_average(S, Pauli.group()))
