"""Functions & System class for reading & using HANDE-created Hamiltonians."""
from .system import System

import numpy as np

from typing import Dict
from numpy.typing import NDArray as Array


def read_matrix(matrix_filename: str, is_complex: bool = False) -> Array:
    """Load matrix from a HANDE file into a NumPy array."""
    if is_complex:
        raise NotImplementedError(
            'Reading complete HANDE Hamiltonians is not currently '
            'implemented please send patches!'
        )

    ndets = 0
    elements = {}

    with open(matrix_filename, 'rt') as stream:
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

    return ham


class MatrixHamiltonian(System):
    """
    System defined by a HANDE-created Hamiltonian matrix.

    Use this class for systems defined by a triangular Hamiltonian matrix
    output by HANDE. The Hamiltonian will be stored as a 2D NumPy array.
    The reference energy is assumed to be element `[0,0]` of the matrix.

    Parameters
    ----------
    input_file
        Filename for the Hamiltonian.
    is_complex
        Whether or not the Hamiltonian is complex.(???)
    shift
        A shift to apply to the diagonal elements of the Hamiltonian.
    use_ip
        Whether or not to use the interaction picture. If specified,
        the non-interacting Hamiltonian will be available through the
        `noninteracting_hamiltonian` attribute.

    Attributes
    ----------
    input_file
    reference_energy
    is_complex
    hamiltonian
    noninteracting_hamiltonian
    unshifted_hamiltonian
    raw_hamiltonian
    ndeterminants
    sort_map

    Warnings
    --------
    Support for complex Hamiltonians is not yet implemented.
    Setting `is_complex = True` will raise `NotImplementedError`.

    Notes
    -----
    The `noninteracting_hamiltonian` will be `None`
    unless `use_ip` is specified when calling `initialize()`.
    """

    @property
    def hamiltonian(self) -> Array | None:
        """Hamiltonian shifted by Hartree-Fock energy & any provided shift."""
        return self._shifted_hamil

    @property
    def noninteracting_hamiltonian(self) -> Array | None:
        """Non-interacting Hamiltonian."""
        return self._non_interacting

    @property
    def unshifted_hamiltonian(self) -> Array | None:
        """Sorted, unshifted Hamiltonian matrix."""
        return self._sorted_hamil

    @property
    def raw_hamiltonian(self) -> Array:
        """Unsorted, unshifted Hamiltonian matrix."""
        return self._raw_hamil

    @property
    def ndeterminants(self) -> int:
        """Size of the determinant space."""
        return self._ndet

    @property
    def sort_map(self) -> Dict[int, int] | None:
        """Maps original index of raw diagonals & their sorted position."""
        return self._sort_map

    def __init__(
            self,
            input_file: str,
            is_complex: bool = False,
            shift: int = 0,
            use_ip: bool = False,
            **kwargs,
            ) -> None:

        super().__init__(input_file=input_file,
                         is_complex=is_complex,
                         **kwargs)

        self._raw_hamil = read_matrix(self._input_file, self._is_complex)
        self._ndet = self._raw_hamil.shape[0]
        self._ref_eng = self._raw_hamil[0, 0]

        # The following are set by self._shift()
        # though self._non_interacting will remain None if use_ip is False.
        self._sorted_hamil = None
        self._sort_map = None
        self._shifted_hamil = None
        self._non_interacting = None
        self._shift(shift, use_ip)

    def _sort_on_diagonals(self) -> None:
        """
        Sort Hamiltonian based on ascending order of diagonal elements.

        Rearrange the Hamiltonian to be ascending on its diagonal elements.
        Store the sorted Hamiltonian array, array of sorted diagonals,
        and the dictionary mapping the diagonal's original index
        to its sorted position.
        """
        diags = np.diag(self._raw_hamil)
        sorted_index = np.argsort(diags)
        index_map = {int(ii): i for i, ii in enumerate(sorted_index)}

        sorted_hamil = np.zeros_like(self._raw_hamil)

        for i in range(self.ndeterminants):
            for j in range(self.ndeterminants):
                ii = index_map[i]
                jj = index_map[j]
                sorted_hamil[ii, jj] = self._raw_hamil[i, j]

        self._sorted_hamil = sorted_hamil
        self._sort_map = index_map

    def _shift(self,
               shift: int,
               use_ip: bool
               ) -> None:
        """
        Initialize & store relevant matrices for analytical QMC.

        Parameters
        ----------
        shift
            A shift to apply to the diagonal elements of the Hamiltonian.
        use_ip
            Whether or not to use the interaction picture. If specified,
            the non-interacting Hamiltonian will be available through the
            `noninteracting_hamiltonian` attribute.
        """
        self._sort_on_diagonals()  # sets self._sorted_hamil

        II = np.eye(self.ndeterminants)
        H = self._sorted_hamil - self.ref_energy * II - shift * II

        if use_ip:
            self._non_interacting = np.diag(np.diag(H))

        self._shifted_hamil = H
