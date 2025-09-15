"""Base class for the `systems` submodule."""

from .. import utils

import numpy as np
from sympy.utilities.iterables import multiset_permutations as gen_perm_set

from numpy.typing import ArrayLike, NDArray as Array


class System:
    """
    Base class for defining quantum systems.

    Parameters
    ----------
    input_file : str
        Name of the integral file that defines the system.
    is_complex : bool, default True
        Whether or not the integral is complex;
        controls the integral index symmetry.
    """

    def __init__(
            self,
            input_file: str,
            is_complex: bool = False,
            **kwargs,
            ) -> None:

        self._input_file = input_file
        self._is_complex = is_complex

        # These attributes may be set via different means,
        # e.g. initialization or loading from file.
        # No mechanism for setting them in the base class
        # exists, but they are set to None as a fallback
        self._norb = None
        self._nel = None
        self._na = None
        self._nb = None
        self._orbsym = None
        self._H = None
        self._ndets = None
        self._ref_eng = None
        self._eig = None

        self._sym = None  # can we avoid this for MatrixHam?

        # These can be set with self._set_derived_quants
        # once the above attributes have been set.
        self._maxsym = None
        self._pg_mask = None
        self._orbs = None
        self._ms = None

        # CK: If we want System to include the reference state functionality
        # that's currently in the Integral class, then we can also migrate
        # psingle and pdouble attributes & associated calculation
        # to this class.

        # These can be set using the methods defined in this base class.
        self._bitarrays = None
        self._nex_mat = None

        return

    def _set_derived_quants(self):
        """
        Set attributes that can be derived from other attributes.

        This is intended to be used by child class __init__ in order to
        enforce consistency accross children.
        """
        if self._orbsym is not None:
            self._maxsym = int(2**np.ceil(np.log(np.max(self._orbsym)+1)
                               / np.log(2)))
            self._pg_mask = self._maxsym - 1

        if self._norb is not None:
            self._orbs = np.arange(self._norb)
            self._ms = np.array(
                        [(i+1) % 2 - i % 2 for i in self._orbs]
                       )

    @property
    def input_file(self) -> str:
        """Filename for loaded Hamiltonian."""
        return self._input_file

    @property
    def is_complex(self) -> bool:
        """Whether or not the Hamiltonain is complex."""
        return self._is_complex

    @property
    def n_orbitals(self) -> int | None:
        """Number of spin orbitals."""
        return self._norb

    @property
    def n_electrons(self) -> int | None:
        """Total number of electrons."""
        return self._nel

    @property
    def n_alpha(self) -> int | None:
        """Number of alpha electrons."""
        return self._na

    @property
    def n_beta(self) -> int | None:
        """Number of beta electrons."""
        return self._nb

    @property
    def orbital_pg_symmetry(self) -> Array | None:
        """Orbital point-group symmetries."""
        return self._orbsym

    @property
    def hamiltonian(self) -> Array | None:
        """The system's Hamiltonian."""
        return self._H

    @property
    def n_determinants(self) -> Array | None:
        """Size of the Hilbert space."""
        return self._ndets

    @property
    def ref_energy(self) -> float | None:
        """Reference Hartree-Fock energy."""
        return self._ref_eng

    @property
    def eigenvalues(self) -> Array | None:
        """System's single-particle eigenvalues."""
        return self._eig

    @property
    def max_symmetry(self) -> int | None:
        """Maximum point-group symmetry contained by the system."""
        return self._maxsym

    @property
    def pg_mask(self) -> int | None:
        """Mask used for point-group operations."""
        return self._pg_mask

    @property
    def orbitals(self) -> Array | None:
        """All orbital indexes."""
        return self._orbs

    @property
    def spin_polarizations(self) -> Array | None:
        """All possible spin polarization of the system(?)."""
        return self._ms

    @property
    def bitarrays(self) -> Array | None:
        """Array of bitarrays in the Hilbert space."""
        return self._bitarrays

    @property
    def excitation_matrix(self) -> Array | None:
        """An `n_determinants`-square matrix of excitations between i and j."""
        return self._nex_mat

    def zero_hamiltonian(self) -> None:
        """
        Subtract the Hartree-Fock energy from the Hamiltonian.

        This will overwrite the existing Hamiltonian with the
        shifted version.
        """
        if self._H is not None:
            self._H -= self._ref_eng * np.eye(self._ndets)
        else:
            raise RuntimeError(
                "The Hamiltonian is currently `None` and cannot be shifted.")

    def generate_determinant_bitarrays(self) -> None:
        """
        Generate all determinants as bitarrays.

        This function sets the `n_determinants` and `bitarrays` members.
        All determinints spanned by the system within the point-group symmetry
        will be generated.

        A reference bitarry is first generated separately for each of
        the spin channels. Next, all unqiue combinations of 1's & 0's are
        generated for that reference. Finally, all the unique
        concatenations of the alpha & beta bitarrays are iterated over,
        checking that the point group symmetry of the concatenated bitarray
        falls within the system's possible point groups.
        If the concatenated bitarray has an allowed point group symmetry,
        that determinant is stored.

        Warnings
        --------
        This function will only have an effect the first time it is run.
        If members `n_determinants` and `bitarrays` are not `None`, this
        function will return without making any changes.

        Notes
        -----
        A `bitarray` is shorthand for an array of 1's and 0's.
        More traditionally referred to as "bitstrings,"
        these are used to represent Slater determinants.
        1's represent an occupied orbital and 0 an unoccupied orbital.

        This function was originally called `generate_bit_arrays`
        and supported the `use_symmetry_block` boolean,
        which controls whether to use the symmetry reduced point
        group Hamiltonian (`True`) or whether the entire Hamiltonian
        containing all those point-group symmetries spanned by the system will
        be generated (`False`). Since in practice this parameter was always
        `True` and this function does not control Hamiltonian generation,
        this parameter has been removed from the function call and hardcoded
        to `True` within the function body.
        """
        if self._bitarrays is not None:  # do nothing
            return

        # Check for required components.
        # TODO decide if sym is hard required
        req_quants = [self._norb, self._na, self._nb,
                      self._orbsym, self._pg_mask]
        missing = [i is None for i in req_quants]
        if True in missing:
            raise RuntimeError("Method generate_determinant_bitarrays "
                               "requires the following be defined: "
                               "n_orbitals, n_alpha, n_beta, "
                               "orbital_pg_symmetry.")

        aba = np.zeros(int(self._norb/2), dtype=int)
        aba[:self._na] = 1
        alpha_bas = list(gen_perm_set(aba))[::-1]

        bba = np.zeros(int(self._norb/2), dtype=int)
        bba[:self._nb] = 1
        beta_bas = list(gen_perm_set(bba))[::-1]

        bas = []
        for bba in beta_bas:
            bind = 2*np.nonzero(bba)[0] + 1
            boccsym = self._orbsym[bind]
            bsym = utils.orb_sym(boccsym, self._pg_mask)

            for aba in alpha_bas:
                aind = 2*np.nonzero(aba)[0]
                aoccsym = self._orbsym[aind]
                asym = utils.orb_sym(aoccsym, self._pg_mask)

                sym = utils.cross_prod_sym(bsym, asym, self._pg_mask)
                ba = np.zeros(self._norb, dtype=int)
                ba[np.arange(0, self._norb, 2)] = aba
                ba[np.arange(1, self._norb, 2)] = bba

                use_symmetry_block = True
                if sym == self._sym:
                    bas.append(ba)
                elif not use_symmetry_block:
                    bas.append(ba)

        self._bitarrays = np.array(bas)

    def get_bitarray_integers(self) -> Array:
        """Integer representations of system bitarrays."""
        # Generate bitarrays if not already set.
        self.generate_determinant_bitarrays()

        bitints = [np.exp2(self._orbs[ba == 1]).sum()
                   for ba in self._bitarrays]
        return np.array(bitints).astype(np.int64)

    def generate_excitation_matrix(self) -> None:
        """
        Generate the matrix of excitations.

        Sets the `excitation_matrix` member. Each matrix index i, j represents
        the transition between bitarrays i and j. This function will also
        generate the system determinants if they have not already
        been generated, thereby setting `n_determinants` and `bitarrays`.

        Warnings
        --------
        This function will only have an effect the first time it is run.
        If the `excitation_matrix` member is not `None`, this
        function will return without making any changes.
        """
        if self._nex_mat is not None:
            return

        # Generate bitarrays if not already set.
        self.generate_determinant_bitarrays()
        self._nex_mat = np.zeros((self._ndets, self._ndets), dtype=np.int64)
        for i, b1 in enumerate(self._bitarrays):
            for j, b2 in enumerate(self._bitarrays[i+1:]):
                j += i + 1
                nex = utils.get_nex(b1, b2)
                self._nex_mat[i, j] = nex
                self._nex_mat[j, i] = nex

    def get_virtual_orbitals(self,
                             occ: ArrayLike
                             ) -> list[Array]:
        """
        Given an occupied orbital array, get information on virtual orbitals.

        Parameters
        ----------
        occ : array_like
            The occupied orbitals for the current determinant.

        Returns
        -------
        unocc : Array
            The unoccupied orbital indexes in the current determinant.
        virt_ms : Array
            The corresponding spins of `unocc`.
        virt_sym : Array
            The corresponding symmetries of `unocc`.
        nvirt : Array
            The number of unoccupied orbitals in each
            spin-symmetry as indexed by spin and symmetry.

        Warnings
        --------
        There is a always an empty array corresponding to a spin of
        :math:`ms = 0`.
        """
        # TODO this could probably use input checking on np.sum(occ)
        # versus the number of electrons in the system.
        # Should we also check np.max(occ)?

        # Check for required components
        req_quants = [self._orbs, self._orbsym, self._ms, self._maxsym]
        missing = [i is None for i in req_quants]
        if True in missing:
            raise RuntimeError("Method get_virtual_orbitals "
                               "requires the following be defined: "
                               "orbitals, orbital_pg_symmetry, "
                               "spin_polarizations, max_symmetry.")

        unocc = self._orbs[np.isin(self._orbs, occ, invert=True)]
        virt_ms = self._ms[unocc]
        virt_sym = self._orbsym[unocc]

        nvirt = np.zeros((3, self._maxsym))
        for ms, sym in zip(virt_ms, virt_sym):
            nvirt[ms, sym] += 1

        return unocc, virt_ms, virt_sym, nvirt
