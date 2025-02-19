#!/usr/bin/env python

from .system import System

from numpy import zeros
from numpy.typing import NDArray as Array


class MatrixHamiltonian(System):
    r''' TODO: Write class docstring here.
    '''
    def __init__(
            self,
            matrix_file: str,
            iscomplex: bool = False,
            **kwargs,
        ) -> None:
        r''' TODO: Write __init__ docstring here.
        '''
        System.__init__(self, **kwargs)

        self.matrix_file = matrix_file
        self.iscomplex = iscomplex

        self.read_matrix()

        return

    def read_matrix(self) -> None:
        r''' TODO: Write read_matrix docstring here.
        '''
        self.hamiltonian = self.read_hande_hamil(
            self.matrix_file,
            self.iscomplex,
        )

        return

    @staticmethod
    def read_hande_hamil(matrix_file: str, iscomplex: bool = False) -> Array:
        r''' TODO: Write read_hande_hamilt docstring here.
        '''
        if iscomplex:
            raise NotImplementedError(
                'Reading complete HANDE Hamiltonians is not currently '
                'implemented please send patches!'
            )

        ndets = 0
        elements = {}

        with open(matrix_file, 'rt') as stream:
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
