"""Functions for calculating observables."""

import numpy as np

from numpy.typing import NDArray as Array


def total_particles(matrix: Array):
    """Return the total number of particles (aka walkers) in a matrix."""
    return np.abs(matrix).sum()


def n_occupied_states(matrix: Array):
    """Return the number of occupied states in a given matrix."""
    return np.count_nonzero(matrix)


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
    # Tell NumPy to throw errors on "divide by zero" and "invalid value" issues with log
    with np.errstate(divide="raise", invalid="raise"):
        return -(matrix @ np.log(matrix)).trace()


def von_neumann_expectation(matrix: Array):
    """Return the expectation value of the von Neumann estimator."""
    return von_neumann_numerator(matrix) / trace(matrix)
