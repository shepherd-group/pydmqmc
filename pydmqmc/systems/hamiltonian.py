"""System-derived class for reading & using HANDE-created Hamiltonians."""
from .system import System

import numpy as np
from numpy.typing import ArrayLike
import warnings

class MatrixHamiltonian(System):
    r"""
    System defined by a Hamiltonian matrix written to file.

    Use this class for systems defined by a triangular Hamiltonian matrix
    output by HANDE. The Hamiltonian will be stored as a 2D NumPy array.
    The reference energy is assumed to be element `[0,0]` of the matrix.

    Parameters
    ----------
    input_file : str
        Filename for the Hamiltonian.
    is_complex : bool, default False
        Whether or not the Hamiltonian is complex.
    n_orbitals : int, optional
        Number of orbitals in the system. Required if using the
        `generate_determinant_bitarrays` or `get_virtual_orbitals`
        methods.
    n_electrons, n_alpha, n_beta : int, optional
        Number of total, alpha, and beta electrons. All three or
        any two may be supplied, as 
        :math:`\textt{n_electrons} = \textt{n_alpha} + \textt{n_beta}`.
        The `generate_determinant_bitarrays` method requires `n_alpha`
        and `n_beta`; `n_electrons` is not required but can be used to
        infer `n_alpha` or `n_beta`.
    orbital_pg_symmetry : array_like
        Set of orbital point group symmetries. Required if using the
        `generate_determinant_bitarrays` or `get_virtual_orbitals`
        methods.

    Note
    ----
    The `get_bitarray_integers` and `generate_excitation_matrix` methods
    both depend on `generate_determinant_bitarrays` and will call this method
    if it has not previously been called. These methods are therefore
    subject to the same parameter requirements as noted above.

    Warnings
    --------
    Support for complex Hamiltonians is not yet implemented.
    Setting `is_complex = True` will raise `NotImplementedError`.
    """

    def __init__(
            self,
            input_file: str,
            is_complex: bool = False,
            n_orbitals: int | None = None,
            n_electrons: int | None = None,
            n_alpha: int | None = None,
            n_beta: int | None = None,
            orbital_pg_symmetry: ArrayLike | None = None,
            eigenvalues: ArrayLike | None = None,
            ) -> None:

        super().__init__(input_file=input_file,
                         is_complex=is_complex)

        self._read_matrix()  # set self._H
        self._ndets = self._H.shape[0]
        self._ref_eng = self._H[0, 0]

        self._norb = n_orbitals

        if n_electrons and n_alpha and n_beta:
            if n_alpha + n_beta != n_electrons:
                raise RuntimeError("Supplied total number of electrons "
                                   f"{n_electrons} is greater than the sum "
                                   f"of supplied alpha ({n_alpha}) and beta "
                                   f"({n_beta}) electrons.")
            self._nel = n_electrons
            self._na = n_alpha
            self._nb = n_beta
        elif n_electrons is not None and n_alpha is not None:
            self._nel = n_electrons
            self._na = n_alpha
            self._nb = n_electrons - n_alpha
        elif n_electrons is not None and n_beta is not None:
            self._nel = n_electrons
            self._na = n_electrons - n_beta
            self._nb = n_beta
        elif n_alpha is not None and n_beta is not None:
            self._nel = n_alpha + n_beta
            self._na = n_alpha
            self._nb = n_beta
        # check for insufficient info
        elif n_electrons is not None and n_alpha is None and n_beta is None:
            warnings.warn("Unable to set n_alpha and n_beta with current "
                          "information.")
            self._nel = n_electrons
        elif n_alpha is not None and n_electrons is None and n_beta is None:
            warnings.warn("Unable to set n_electrons and n_beta with current "
                          "information.")
            self._na = n_alpha
        elif n_beta is not None and n_electrons is None and n_alpha is None:
            warnings.warn("Unable to set n_electrons and n_alpha with current "
                          "information.")
            self._nb = n_beta

        # super.__init__() sets self._orbsym and self._eig to None
        if orbital_pg_symmetry is not None:
            if isinstance(orbital_pg_symmetry, np.ndarray):
                self._orbsym = orbital_pg_symmetry
            else:
                self._orbsym = np.array(orbital_pg_symmetry)

        if eigenvalues is not None:
            if isinstance(eigenvalues, np.ndarray):
                self._eig = eigenvalues
            else:
                self._eig = np.array(eigenvalues)

        super()._set_derived_quants()

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
