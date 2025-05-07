"""Functions & System class for reading & using HANDE-created Hamiltonians."""
from .system import System

import numpy as np

from typing import Dict
from numpy.typing import NDArray as Array


class MatrixHamiltonian(System):
    """
    System defined by a Hamiltonian matrix written to file.

    Use this class for systems defined by a triangular Hamiltonian matrix
    output by HANDE. The Hamiltonian will be stored as a 2D NumPy array.
    The reference energy is assumed to be element `[0,0]` of the matrix.

    Parameters
    ----------
    input_file : str
        Filename for the Hamiltonian.
    is_complex : bool, default False
        Whether or not the Hamiltonian is complex.(???)
    shift : float, default 0.0
        A shift to apply to the diagonal elements of the Hamiltonian.
    use_ip : bool, default False
        Whether or not to use the interaction picture. If specified,
        the non-interacting Hamiltonian will be available through the
        `noninteracting_hamiltonian` attribute.

    Warnings
    --------
    Support for complex Hamiltonians is not yet implemented.
    Setting `is_complex = True` will raise `NotImplementedError`.

    Notes
    -----
    The `noninteracting_hamiltonian` will be `None`
    unless `use_ip` is specified when calling `initialize()`.
    """

    def __init__(
            self,
            input_file: str,
            is_complex: bool = False,
            shift: float = 0.0,
            use_ip: bool = False,
            ) -> None:

        super().__init__(input_file=input_file,
                         is_complex=is_complex)

        self._read_matrix()  # sets self._raw_hamil
        self._ndets = self._H.shape[0]
        self._ref_eng = self._H[0, 0]

    def _read_matrix(self) -> None:
        """Load matrix from a HANDE file into a NumPy array."""
        if self._is_complex:
            raise NotImplementedError(
                'Reading complete HANDE Hamiltonians is not currently '
                'implemented please send patches!'
            )

        ndets = 0
        elements = {}

        with open(self._input_file, 'rt') as stream:
            for line in stream:
                i, j, hij = line.split()

                i = int(i)
                j = int(j)
                hij = float(hij)

                ndets = max(i, j, ndets)

                elements[i, j] = hij

        ham = np.zeros((ndets, ndets), dtype=float)

        for (i, j), hij in elements.items():
            ham[i - 1, j - 1] = hij
            ham[j - 1, i - 1] = hij

        self._H = ham

