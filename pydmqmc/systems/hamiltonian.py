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
    matrix_file
        Filename for the Hamiltonian.
    iscomplex
        Whether or not the Hamiltonian is complex.
    shift
        A shift to apply to the diagonal elements of the Hamiltonian.
    use_ip
        Whether or not to use the interaction picture. If specified,
        the non-interacting Hamiltonian will be available through the
        `noninteracting_hamiltonian` attribute.

    Attributes
    ----------
    matrix_filename
    is_complex
    hamiltonian
    noninteracting_hamiltonian
    unshifted_hamiltonian
    raw_hamiltonian
    ndeterminants
    ref_energy
    sort_map

    Warnings
    --------
    Support for complex Hamiltonians is not yet implemented.
    Setting `iscomplex = True` will raise `NotImplementedError`.

    Notes
    -----
    The `noninteracting_hamiltonian` will be `None`
    unless `use_ip` is specified when calling `initialize()`.
    """

    @property
    def matrix_filename(self) -> str:
        """Filename for loaded Hamiltonian."""
        return self._matrix_file

    @property
    def is_complex(self) -> bool:
        """Whether or not the Hamiltonain is complex."""
        return self._iscomplex

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
    def ref_energy(self) -> float:
        """Reference energy state."""
        return float(self._ref_eng)  # convert from np.float64

    @property
    def sort_map(self) -> Dict[int, int] | None:
        """Maps original index of raw diagonals & their sorted position."""
        return self._sort_map

    def __init__(
            self,
            matrix_file: str,
            iscomplex: bool = False,
            shift: int = 0,
            use_ip: bool = False,
            **kwargs,
            ) -> None:

        System.__init__(self, **kwargs)

        self._matrix_file = matrix_file
        self._iscomplex = iscomplex

        self._raw_hamil = read_matrix(self._matrix_file, self._iscomplex)
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
