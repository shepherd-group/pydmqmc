#!/usr/bin/env python

from .system import System

from numpy import zeros
from numpy.typing import NDArray as Array


class MatrixHamiltonian(System):
    """System for a HANDE-created Hamiltonian matrix.

    Use this class for systems defined by a triangular Hamiltonian matrix
    output by HANDE. The Hamiltonian will be stored as a 2D NumPy array.
    The reference energy is assumed to be element `[0,0]` of the matrix.

    Parameters
    ----------
    matrix_file
        Filename for the Hamiltonian.
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
        self._ndet = self._hamiltonian.shape[0]
        self._engref = self._hamiltonian[0, 0]

    @property
    def hamiltonian(self):
        """Loaded 2D Hamiltonian matrix."""
        return self._hamiltonian

    @property
    def ndeterminants(self):
        """Size of the determinant space."""
        return self._ndet

    @property
    def ref_energy(self):
        """Reference energy state."""
        return self._engref

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
