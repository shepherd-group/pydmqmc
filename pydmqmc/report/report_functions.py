import numpy as np

from numpy.typing import NDArray as Array


def trace(matrix: Array):
    return np.trace(matrix)

def energy(matrix: Array, hamiltonian: Array):
    return np.trace(hamiltonian @ matrix)

def von_neumann(matrix: Array):
    return -(matrix @ np.log(matrix)).trace()

# divide these by trace to get expectation