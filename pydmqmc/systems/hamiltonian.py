#!/usr/bin/env python

from .system import System

from numpy import zeros
from numpy.typing import NDArray as Array


class MatrixHamiltonian(System):
    """System for a HANDE-created Hamiltonian matrix.

    Use this class for systems defined by a Hamiltonian matrix
    output by HANDE.

    Parameters
    ----------
    matrix_file
        Filename for the HANDE Hamiltonian.
    iscomplex
        Whether or not the Hamiltonian is complex(?).

    Warnings
    --------
    Support for complex Hamiltonians is not yet implemented.
    Setting `iscomplex = True` will raise `NotImplementedError`.
    """

    def __init__(
            self,
            matrix_file: str,
            iscomplex: bool = False,
            **kwargs,
            ) -> None:

        System.__init__(self, **kwargs)

        self.matrix_file = matrix_file
        self.iscomplex = iscomplex

        self._hamiltonian = self._read_matrix()

    @property
    def hamiltonian(self):
        """Loaded Hamiltonian matrix."""
        return self._hamiltonian

    def _read_matrix(self) -> Array:
        """Load matrix from a HANDE file into a NumPy array."""
        if self.iscomplex:
            raise NotImplementedError(
                'Reading complete HANDE Hamiltonians is not currently '
                'implemented please send patches!'
            )

        ndets = 0
        elements = {}

        with open(self.matrix_file, 'rt') as stream:
            for line in stream:
                i, j, hij = line.split()

                i = int(i)
                j = int(j)
                hij = float(hij)

                ndets = max(i, j, ndets)

                elements[i, j] = hij

        ham = zeros((ndets, ndets), dtype=float)

        for (i, j), hij in elements.items():
            ham[i - 1, j - 1] = hij
            ham[j - 1, i - 1] = hij

        return ham
