"""Functions for calculating observables."""

import numpy as np

from numpy.typing import NDArray as Array


def trace(matrix: Array):
    """Calculate the trace of a given matrix."""
    return np.trace(matrix)


def energy_numerator(matrix: Array, hamiltonian: Array):
    """Numerator of the energy estimator."""
    return np.trace(hamiltonian @ matrix)


def energy_expectation(matrix: Array, hamiltonian: Array):
    """Return the expectation value of the energy estimator."""
    return energy_numerator(matrix, hamiltonian) / trace(matrix)


def von_neumann_numerator(matrix: Array):
    """Numerator of the von Neumann estimator."""
    return -(matrix @ np.log(matrix)).trace()


def von_neumann_expectation(matrix: Array):
    """Return the expectation value of the von Neumann estimator."""
    return von_neumann_numerator(matrix) / trace(matrix)
