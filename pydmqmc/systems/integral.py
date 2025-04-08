"""
Functions & System class for using HANDE-created integrals (FCIDUMP files).

Notes
-----
TODO I pulled `parallel_hamiltonian` out of this class for now
for ease of development. All Numba-related functions should not
be forgotton.
"""

import pydmqmc.utils as utils
from .system import System

import numpy as np
import json

from numpy.typing import NDArray as Array


def generate_ijab_symmetries_array(i: int, j: int,
                                   a: int, b: int,
                                   eight_fold: bool = True,
                                   rhf: bool = True) -> Array:
    """
    Generate an array of valid symmetry permutations of the orbital indicies.

    Assume physics notation.

    Parameters
    ----------
    i, j, a, b
        Orbital indexes.
    eight_fold
        Whether or not the system is 8-fold spatially symmetric.
    rhf
        Use restricted Hartree--Fock; indicates spin symmetry.

    Returns
    -------
    array
        All valid i, j, a, b permutations
    """
    # (???) are these error checks accurate?
    # Some error checking should be in place if this will be public.
    # It makes sense to have this function available outside of a system
    # though it may make more sense in the utils submodule.
    if a > i:
        raise ValueError(f"Index a {a} cannot excede i {i}.")
    if b > j:
        raise ValueError(f"Index b {b} cannot excede j {j}.")
    
    if rhf:
        i, j, a, b = i+i, j+j, a+a, b+b
    uhf = not rhf
    nspat = int(4 - 3*uhf)
    nspin = int(4 - 3*uhf + 4*eight_fold - 3*eight_fold*uhf)

    SS = [
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ]

    FF = [
        [i, j, a, b],
        [a, b, i, j],
        [j, i, b, a],
        [b, a, j, i],
    ]

    EF = [
        [i, j, a, b],
        [j, i, b, a],
        [a, b, i, j],
        [b, a, j, i],
        [a, j, i, b],
        [b, i, j, a],
        [i, b, a, j],
        [j, a, b, i],
    ]

    if eight_fold:
        P = np.repeat(EF, nspat, axis=0)
    else:
        P = np.repeat(FF, nspat, axis=0)

    if rhf:
        P += np.tile(SS, (nspin, 1))

    return P


class Integral(System):
    """
    System defined by a HANDE-created FCIDUMP integral file.

    Parameters
    ----------
    integral_file
        Name of the integral file that defines the system.
    is_complex
        Whether or not the integral is complex;
        controls the integral index symmetry.
    reference
        Specify the occupied spin orbitals for the system's reference
        determinant. If `None`, assume the lowest energy orbitals are
        occupied. If orbital energies are not `integral_file` we assume
        the first `nel` orbitals are occupied.
    symmetry
        Override the point-group symmetry of the system integrals
        in `integral_file`. If `None`, defaults to the value of `ISYM`
        from the `integral_file`.
    orbital_eigenvalues
        Calculate the orbital eigenvalues for the reference state.
    determinants
        Control whether we generate all determinants spanned by the system
        within the point-group symmetry.
    hamiltonian
        Generate the Hamiltonian for the system.
    excitation_matrix
        Control whether we generate the matrix of excitations
        between i and j indexing the matrix, where i and j represent the
        bitarrays at i and j.
    bitarray_integers
        Control whether the integer representation of the bitarrays
        are generated.

    Attributes
    ----------
    input_file
    is_complex
    unrestriced_HF
    h0e
    h1e
    h2e
    eigenvalues
    n_orbitals
    n_virtual
    n_electrons
    n_alpha
    n_beta
    spin_polarization
    orbital_pg_symmetry
    ground_state_pg
    max_symmetry
    pg_mask
    symmetry
    ref_determinant
    orbitals

    Contains:
        self.bitarrays: An array of the bitarray's in the hilbert space
        self.ndets: The total number of determinants in the hilbert space
        self.hii: The diagonal elements of the system Hamiltonian
        self.H: The system hamiltonian generated with the integrals
        self.Href: Reference Hartree-Fock state.
        self.psingle: Probability of generating a single excitation.
        self.pdouble: Probability of generating a double excitation.
        self.nex_mat: An ndets X ndets matrix of excitations between i and j
        self.bitints: Integer representations of the bitarrays.
    """

    @property
    def unrestrictd_HF(self) -> bool:
        """Does the system use unrestricted Hartree-Fock."""
        return self._uhf

    @property
    def h0e(self) -> float:
        """Core Hamiltonian."""
        return self._h0e

    @property
    def h1e(self) -> Array:
        """The one-particle integrals."""
        return self._h1e

    @property
    def h2e(self) -> Array:
        """The two-particle integrals."""
        return self._h2e

    @property
    def eigenvalues(self) -> Array:
        """System's single-particle eigenvalues."""
        return self._eig

    @property
    def n_orbitals(self) -> int:
        """Number of spin orbitals."""
        return self._norb

    @property
    def n_virtual(self) -> int:
        """Number of virtual orbitals."""
        return self._nvirt

    @property
    def n_electrons(self) -> int:
        """Total number of electrons."""
        return self._nel

    @property
    def n_alpha(self) -> int:
        """Number of alpha electrons."""
        return self._na

    @property
    def n_beta(self) -> int:
        """Number of beta electrons."""
        return self._nb

    @property
    def spin_polarization(self) -> int:
        """Spin polarization of the system."""
        return self._ms2

    @property
    def orbital_pg_symmetry(self) -> Array:
        """Orbital point-group symmetries."""
        return self._orbsym

    @property
    def ground_state_pg(self) -> int:
        """Ground state point-group of the system."""
        return self._isym

    @property
    def max_symmetry(self) -> int:
        """Maximum point-group symmetry contained by the system."""
        return self._maxsym

    @property
    def pg_mask(self) -> int:
        """Mask used for point-group operations."""
        return self._pg_mask

    @property
    def symmetry(self) -> int:
        """Point-group symmetry of the system."""
        return self._sym

    @property
    def ref_determinant(self) -> Array:
        """Bitarray for the reference occupation."""
        return self._ref_det

    @property
    def orbitals(self) -> Array:
        """All orbital indexes."""
        return self._orbs

    def __init__(self,
                 integral_file: str = None,
                 is_complex: bool = False,
                 reference: Array | None = None,
                 symmetry: int | None = None,
                 orbital_eigenvalues: bool = False,
                 determinants: bool = False,
                 hamiltonian: bool = False,
                 excitation_matrix: bool = False,
                 bitarray_integers: bool = False,
                 **kwargs
                 ) -> None:

        super().__init__(input_file = integral_file,
                        is_complex = is_complex,
                        **kwargs)

        self._uhf = False
        self._case_h0e = None
        self._case_h1e = None
        self._case_h2e = None
        self._case_eig = None
        self._h0e = 0.0
        self._h1e = None
        self._h2e = None
        self._eig = None
        self._norb = 0
        self._nvirt = 0
        self._nel = 0
        self._na = 0
        self._nb = 0
        self._orbsym = None
        self._ms = None  # does this need a getter?
        self._ms2 = 0
        self._isym = 0

        self._read_integral_file()

        self._maxsym = int(2**np.ceil(np.log(np.max(self._orbsym)+1)
                                      / np.log(2)))
        self._pg_mask = self._maxsym - 1

        self._set_reference(reference)
        self._set_symmetry(symmetry)
        self._symmetry_check()

        self._orbs = np.arange(self._norb)
        self._ref_det = np.zeros(self._norb, dtype=int)
        self._ref_det[self._ref] = 1
        # self._psingle, self._pdouble = calculate_psingle_pdouble(
        #                                     self._orbs,
        #                                     self._ms,
        #                                     self._orbsym,
        #                                     self._maxsym,
        #                                     self._pg_mask,
        #                                     self._ref_det)
        self._ref_eng = 0.5*(np.diag(self._h1e)[self._ref]).sum()
        self._ref_eng += 0.5*self._eig[self._ref].sum() + self._h0e

        if orbital_eigenvalues:  # can only be run once
            self._generate_orbital_eigenvalues()

        self._ndets = None
        self._bitarrays = None
        if determinants or hamiltonian or excitation_matrix \
           or bitarray_integers:
            self.generate_determinants()

        self._hii = None
        self._H = None
        if hamiltonian:
            self.generate_hamiltonian()

        self._nex_mat = None
        # TODO: are these two mutually exclusive?
        if excitation_matrix:
            self.generate_excitation_matrix()
        if bitarray_integers:
            self.generate_bitarray_integers()

    def _read_integral_file(self) -> None:
        """Read in an FCIDUMP file."""
        with open(self.input_file, 'r') as open_int_file:
            footer = False
            for line in open_int_file:
                line = line.replace('\n', '')
                if not footer:
                    line = line.upper()
                    if line[-1] != ',':
                        line = line + ','
                    if 'UHF' in line and 'TRUE' in line:
                        self._uhf = True

                if footer:
                    ls = line.split()
                    eri = float(ls[0])

                    i, a, j, b = [int(d)-1 for d in ls[1:]]
                    self._integral_case(i, j, a, b)

                    if self._case_h2e:
                        for i, j, a, b in self._permute_ijab(i, j, a, b):
                            self._h2e[i, j, a, b] = eri

                    elif self._case_h1e:
                        for i, j, a, b in self._permute_ijab(i, j, a, b):
                            self._h1e[i, a] = eri

                    elif self._case_eig:
                        self._eig[2*i:2*i+2] = eri

                    elif self._case_h0e:
                        self._h0e = eri

                elif '/' in line or 'END' in line:
                    footer = True
                    self._nb = int((self._nel - self._ms2) / 2)
                    self._na = self._nel - self._nb
                    self._norb -= int(self._uhf * (self._norb/2))
                    self._ms = np.array(
                        [(i+1) % 2 - i % 2 for i in range(self._norb)]
                        )
                    self._nvirt = self._norb - self._nel
                    self._alloc_arrays()

                elif 'ORBSYM' in line:
                    self._orbsym = line.split('=')[-1].split(',')[:-1]
                    self._orbsym = np.array(self._orbsym).astype(int) - 1
                    self._orbsym = np.repeat(self._orbsym, 2 - self._uhf)

                else:
                    ls = line.split(',')
                    for ld in ls:
                        if 'NORB' in ld:
                            self._norb = int(2*self._ld_strip(ld))
                        if 'NELEC' in ld:
                            self._nel = self._ld_strip(ld)
                        if 'MS2' in ld:
                            self._ms2 = self._ld_strip(ld)
                        if 'ISYM' in ld:
                            self._isym = self._ld_strip(ld) - 1

    def _integral_case(self,
                       i: int,
                       j: int,
                       a: int,
                       b: int
                       ) -> None:
        self._case_h2e = np.sign(b) != -1
        self._case_h1e = np.sign(a) != -1 and not self._case_h2e
        self._case_eig = np.sign(i) != -1 and not self._case_h1e
        self._case_h0e = np.sign(i) == -1

    def _permute_ijab(self,
                      i: int,
                      j: int,
                      a: int,
                      b: int
                      ) -> Array:
        return generate_ijab_symmetries_array(
                i, j, a, b,
                eight_fold=(not self._is_complex),
                rhf=(not self._uhf))

    def _alloc_arrays(self) -> None:
        """
        Generate the h2e, h1e and eig integral arrays.

        Size of the arrays depends on the total number of spin orbitals.
        """
        self._h2e = np.zeros((self._norb, self._norb, self._norb, self._norb))
        self._h1e = np.zeros((self._norb, self._norb))
        self._eig = np.zeros(self._norb)

    def _ld_strip(self, line_data: str) -> int:
        return int(line_data.split('=')[-1])

    def _set_reference(self, reference: None | Array) -> None:
        if reference is not None:
            self._ref = np.array(reference)
        elif np.sum(self._eig) != 0.0:
            self._ref = np.argsort(self._eig)[:self._nel]
        else:
            self._ref = np.arange(self._nel)

    def _set_symmetry(self, symmetry: None | int) -> None:
        if symmetry is not None:
            self._sym = symmetry
            if self._sym not in self._orbsym:
                error_msg = ' The provided symmetry is not within \n'
                error_msg += ' the symmetries spanned by the system! \n'
                error_msg += ' provided symmetry: %s \n' % self._sym
                error_msg += ' Symmetries spanned by system: %s' % self._orbsym
                raise ValueError(error_msg)
        else:
            self._sym = self._isym

    def _symmetry_check(self) -> None:
        aba_chk = self._orbsym[self._ref[self._ref % 2 == 0]]
        bba_chk = self._orbsym[self._ref[self._ref % 2 != 0]]
        bsym_chk = utils.orb_sym(bba_chk, self._pg_mask)
        asym_chk = utils.orb_sym(aba_chk, self._pg_mask)
        sym_chk = utils.cross_prod_pg_sym(bsym_chk, asym_chk, self._pg_mask)
        if sym_chk != self._sym:
            raise ValueError(
                "Reference determinant is not "
                "within the the symmetry of the system!"
                )

    def _generate_orbital_eigenvalues(self):
        r"""
        Modify eig integral arrays with orbital occupation energies(???).

        Math from Szabo and Ostlund:
        e_a = <a|h|a> + \sum_{b != a}^{N} <ab|ab> - <ab|ba>
        Note that, b != a only applies when a is occupied
        and b is always occupied
        """
        for a in range(self._norb):
            self._eig[a] = self._h1e[a, a]
            for b in self._ref[self._ref != a]:
                self._eig[a] += self._h2e[a, b, a, b]
                self._eig[a] -= self._h2e[a, b, b, a]

    # def generate_determinants(self):
    #     if self._ndets is not None:
    #         raise RuntimeError("Determinants have already been generated!")

    #     self._ndets, self._bitarrays = generate_bit_arrays(
    #                                     self._norb,
    #                                     self._na,
    #                                     self._nb,
    #                                     self._orbsym,
    #                                     self._pg_mask,
    #                                     self._sym)

    # def generate_hamiltonian(self):
    #     self._hii = np.array([get_hij(b,b,self) for b in self._bitarrays])
    #     esortind = np.argsort(self._hii)
    #     self._hii = self._hii[esortind]
    #     self._bitarrays = self._bitarrays[esortind]
    #     self._H = np.diag(self._hii)
    #     for i, b1 in enumerate(self._bitarrays):
    #         for j, b2 in enumerate(self._bitarrays[i+1:]):
    #             j += i + 1
    #             hij = get_hij(b1,b2,self)
    #             self._H[i,j], self._H[j,i] = hij, hij

    # def generate_excitation_matrix(self):
    #     self._nex_mat = np.zeros((self._ndets, self._ndets), dtype=np.int64)
    #     for i, b1 in enumerate(self._bitarrays):
    #         for j, b2 in enumerate(self._bitarrays[i+1:]):
    #             j += i + 1
    #             nex = get_nex(b1,b2)
    #             self._nex_mat[i,j] = nex
    #             self._nex_mat[j,i] = nex

    def generate_bitarray_integers(self):
        bitints = [np.exp2(self._orbs[ba==1]).sum() for ba in self._bitarrays]
        self.bitints = np.array(bitints).astype(np.int64)

    def dumpeigs(self, float_fmt=' % 24.16E', int_fmt='%3i'):
        fmt = float_fmt + f' {int_fmt} {int_fmt} {int_fmt} {int_fmt}'
        inds = np.arange(0, self._norb, 2 - self._uhf)
        for i in inds:
            iout = int(i/(2 - self._uhf)) + 1
            out_tuple = (self._eig[i], iout, 0, 0, 0)
            print(fmt % out_tuple)

    # def report(self) -> None:
    #     """Print information about the system."""
    #     print(' ---- System information ----')
    #     print(
    #         json.dumps(
    #             {
    #                 'int_file'  : self._input_file,
    #                 'UHF'       : self._uhf,
    #                 'Norb'      : self._norb,
    #                 'Nvirt'     : self._nvirt,
    #                 'Nel'       : self._nel,
    #                 'Na'        : self._na,
    #                 'Nb'        : self._nb,
    #                 'MS2'       : self._ms2,
    #                 'ISYM'      : self._isym,
    #                 'maxsym'    : self._maxsym,
    #                 'symmetry'  : self._sym,
    #                 'pg_mask'   : self._pg_mask,
    #                 'Href'      : self.Href,
    #                 'reference' : '%s' % self._ref,
    #                 'ref_det'   : '%s' % self._ref_det,
    #                 'p_single'  : self.psingle,
    #                 'p_double'  : self.pdouble,
    #             },
    #         indent=4, ensure_ascii=True)
    #     )
    #     print()
    #     print('  '+'#'*6+' Basis set table start. '+'#'*6)
    #     print(' '+'-'*50)
    #     print(' {:>8} {:>10} {:>6} {:>22}'\
    #             .format('index','Symmetry','ms','<i|f|i>'))
    #     print(' '+'-'*50)
    #     outstr = ' {:>8} {:>10} {:>6} {:> 22.12E}'
    #     for i in range(self._norb):
    #         print(outstr.format(i,self._orbsym[i],self._ms[i],self._eig[i]))
    #     print(' '+'-'*50)
    #     print('  '+'#'*6+'  Basis set table end.  '+'#'*6)
    #     self.print_symmetry_table()

    # def print_symmetry_table(self) -> None:
    #     """
    #     Write out the symmetry table for the system.
        
    #     Write out the general pg symmetry table from all combinations
    #     of point groups first. ( I am forgetful okay :P )
    #     Then write out the resulting point group table from the cross
    #     product of the orbital point groups.
    #     """
    #     print('\n')
    #     print( ' Symmetry cross product (xp) table using:')
    #     print(f'    pg_mask: {self._pg_mask}')
    #     print( '    row = pg symmetry 1')
    #     print( '    col = pg symmetry 2')
    #     print( '    xp[row,col]:')
    #     print()

    #     xp = np.zeros((self._maxsym, self._maxsym), dtype=np.int64)
    #     for isym in range(0, self._maxsym):
    #         for jsym in range(0, self._maxsym):
    #             xp_pg = bitarray_pg(isym, jsym, self._pg_mask)
    #             xp[isym,jsym] = xp_pg

    #     header = ' sym |'
    #     pg_rows = [f'  {isym:>2} |' for isym in range(0, self._maxsym)]
    #     for isym in range(0, self._maxsym):
    #         header += f' {isym:>2}'
    #         for jsym in range(0, self._maxsym):
    #             pg_rows[isym] += f' {xp[isym,jsym]:>2}'

    #     print(header)
    #     print(' ' + '-'*int(6 + 3 * (self._maxsym)))
    #     for isym in range(0, self._maxsym):
    #         print(pg_rows[isym])

    #     print()
    #     print( ' Symmetry cross product table from system orbitals:')
    #     print(f'    pg_mask: {self._pg_mask}')
    #     print( '    row = orbital 1')
    #     print( '    col = orbital 2')
    #     print( '    xp[row,col]:')
    #     print()

    #     xp = np.zeros((self._norb, self._norb), dtype=np.int64)
    #     for iorb in range(0, self._norb):
    #         isym = self._orbsym[iorb]
    #         for jorb in range(0, self._norb):
    #             jsym = self._orbsym[jorb]
    #             xp_pg = bitarray_pg(isym, jsym, self._pg_mask)
    #             xp[iorb,jorb] = xp_pg

    #     header = ' orb |'
    #     pg_rows = [f'  {iorb:>2} |' for iorb in range(0, self._norb)]
    #     for iorb in range(0, self._norb):
    #         header += f' {iorb:>2}'
    #         for jorb in range(0, self._norb):
    #             pg_rows[iorb] += f' {xp[iorb,jorb]:>2}'

    #     print(header)
    #     print(' ' + '-'*int(6 + 3 * (self._norb)))
    #     for irow in range(0, self._norb):
    #         print(pg_rows[irow])
