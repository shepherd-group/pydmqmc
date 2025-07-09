"""
System-derived class for using HANDE-created integrals (FCIDUMP files).

Notes
-----
TODO I pulled `parallel_hamiltonian` out of this class for now
for ease of development.
"""

from .. import utils
from .system import System

import numpy as np
import json

from numpy.typing import NDArray as Array
from numpy.typing import ArrayLike


class Integral(System):
    r"""
    System defined by a set of integral files.

    Parameters
    ----------
    input_file : str
        Name of the integral file that defines the system.
    is_complex : bool, default True
        Whether or not the integral is complex;
        controls the integral index symmetry.

    Other Parameters
    ----------------
    reference : array_like, optional
        Specify the occupied spin orbitals for the system's reference
        determinant. If `None`, assume the lowest energy orbitals are
        occupied. If orbital energies are not `input_file` we assume
        the first `nel` orbitals are occupied.
    symmetry : int, optional
        Override the point-group symmetry of the system integrals
        in `input_file`. If `None`, defaults to the value of `ISYM`
        from the `input_file`.
    orbital_eigenvalues : bool, default False
        Calculate the orbital eigenvalues for the reference state.
        Useful if the `input_file` does not have energies for
        `N 0 0 0` states where `N` is a positive integer.

    Notes
    -----
    Enabling `orbital_eigenvalues` using the following math
    from Szabo and Ostlund [1]_:

    .. math:: e_a = <a|h|a> + \sum_{b \neq a}^{N} <ab|ab> - <ab|ba>

    Note that, :math:`b \neq a` only applies when :math`a` is occupied
    and :math:`b` is always occupied.

    References
    ----------
    .. [1] Attila Szabo and Neil S. Ostlund, "Modern Quantum Chemistry:
        Introduction to Advanced Electronic Structure Theory," Dover Books
        on Chemistry, 1996
    """

    def __init__(self,
                 input_file: str,
                 is_complex: bool = False,
                 reference: ArrayLike | None = None,
                 symmetry: int | None = None,
                 orbital_eigenvalues: bool = False,
                 ) -> None:

        super().__init__(input_file=input_file,
                         is_complex=is_complex)

        # set attributes not in parent to a starting value
        self._uhf = False
        self._case_h0e = None
        self._case_h1e = None
        self._case_h2e = None
        self._case_eig = None
        self._h0e = None
        self._h1e = None
        self._h2e = None
        self._nvirt = None
        self._orbsym = None
        self._ms2 = None
        self._isym = None

        self._read_integral_file()

        super()._set_derived_quants()

        self._set_reference(reference)
        self._set_symmetry(symmetry)
        self._symmetry_check()

        self._ref_det = np.zeros(self._norb, dtype=int)
        self._ref_det[self._ref] = 1
        self._ref_eng = 0.5*(np.diag(self._h1e)[self._ref]).sum()
        self._ref_eng += 0.5*self._eig[self._ref].sum() + self._h0e
        self._calculate_psingle_pdouble()

        if orbital_eigenvalues:  # can only be run once
            self._generate_orbital_eigenvalues()

        # Each of these are set by "generate" methods.
        # They can only be set if None, meaning they
        # can each only be set once.
        self._ndets = None
        self._bitarrays = None
        self._H = None
        self._nex_mat = None

    @property
    def unrestricted_HF(self) -> bool:
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
    def n_virtual(self) -> int:
        """Number of virtual orbitals."""
        return self._nvirt

    @property
    def spin_polarization(self) -> int:
        """Spin polarization of the system."""
        return self._ms2

    @property
    def ground_state_pg(self) -> int:
        """Ground state point-group of the system."""
        return self._isym

    @property
    def symmetry(self) -> int:
        """Point-group symmetry of the system."""
        return self._sym

    @property
    def ref_determinant(self) -> Array:
        """Bitarray for the reference occupation."""
        return self._ref_det

    @property
    def prob_single(self) -> float:
        """Probability of generating a single excitation."""
        return self._psingle

    @property
    def prob_double(self) -> float:
        """Probability of generating a double excitation."""
        return self._pdouble

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
        return utils.generate_ijab_symmetries_array(
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
        sym_chk = utils.cross_prod_sym(bsym_chk, asym_chk, self._pg_mask)
        if sym_chk != self._sym:
            raise ValueError(
                "Reference determinant is not "
                "within the the symmetry of the system!"
                )

    def _generate_orbital_eigenvalues(self) -> None:
        r"""
        Modify eig integral arrays with orbital occupation energies(???).

        Notes
        -----
        Math from Szabo and Ostlund[1]_:

        .. math:: e_a = <a|h|a> + \sum_{b \neq a}^{N} <ab|ab> - <ab|ba>

        Note that, :math:`b \neq a` only applies when :math`a` is occupied
        and :math:`b` is always occupied.

        References
        ----------
        .. [1] Attila Szabo and Neil S. Ostlund, "Modern Quantum Chemistry:
           Introduction to Advanced Electronic Structure Theory," Dover Books
           on Chemistry, 1996
        """
        for a in range(self._norb):
            self._eig[a] = self._h1e[a, a]
            for b in self._ref[self._ref != a]:
                self._eig[a] += self._h2e[a, b, a, b]
                self._eig[a] -= self._h2e[a, b, b, a]

    def _calculate_psingle_pdouble(self) -> None:
        """
        Calculate the single and double excitation probabilities.

        The reference state is assumed to be a good candidate for the
        number of single and double excitations possible for a given system.
        This function simply counts the number of excitations
        which are allowed by symmetry and spin conservation for each electron.
        It does not account for factors like the value of Hamiltonian element
        between those excitations, etc.
        """
        occ = self._orbs[self._ref_det == 1]
        unocc, virt_ms, virt_sym, nvirt = self.get_virtual_orbitals(occ)

        occ_ms = self._ms[occ]
        occ_sym = self._orbsym[occ]
        nsingle = np.sum(nvirt[(occ_ms, occ_sym)])

        ndouble = 0
        for i, (ims, isym) in enumerate(zip(occ_ms, occ_sym)):
            for jms, jsym in zip(occ_ms[i+1:], occ_sym[i+1:]):
                for syma in np.arange(self._maxsym):
                    symb = utils.cross_prod_sym(isym, jsym, self._pg_mask)
                    symb = utils.cross_prod_sym(syma, symb, self._pg_mask)
                    if syma == symb and ims == jms:
                        ndouble += nvirt[ims, syma]*(nvirt[jms, symb]-1)/2
                    elif syma == symb:
                        ndouble += nvirt[ims, syma]*nvirt[jms, symb]
                    elif syma < symb:
                        ndouble += nvirt[ims, syma]*nvirt[jms, symb]
                        if ims != jms:
                            ndouble += nvirt[jms, syma]*nvirt[ims, symb]

        self._psingle = nsingle/(nsingle + ndouble)
        self._pdouble = ndouble/(nsingle + ndouble)

    def generate_determinant_bitarrays(self):
        super().generate_determinant_bitarrays()
        self._ndets = self._bitarrays.shape[0]

    def generate_hamiltonian(self) -> None:
        """
        Generate the Hamiltonian for the system.

        Sets the `hamiltonian` member.
        It will also generate the system determinants if they have not already
        been generated, thereby setting `n_determinants` and `bitarrays`.

        Warnings
        --------
        This function will only have an effect the first time it is run.
        If the `hamiltonian` member is not `None`, this
        function will return without making any changes.
        """
        if self._H is not None:
            return

        # Generate bitarrays if not already set.
        self.generate_determinant_bitarrays()

        hii = np.array([self._get_hij(b, b)
                        for b in self._bitarrays])
        esortind = np.argsort(hii)
        hii = hii[esortind]
        self._bitarrays = self._bitarrays[esortind]
        self._H = np.diag(hii)
        for i, b1 in enumerate(self._bitarrays):
            for j, b2 in enumerate(self._bitarrays[i+1:]):
                j += i + 1
                hij = self._get_hij(b1, b2)
                self._H[i, j], self._H[j, i] = hij, hij

    def _get_hij(self,
                 b1: Array,
                 b2: Array,
                 tol: float = 1E-16) -> float:
        nex, abrs, perms = utils.get_ex_info(b1, b2, self.n_electrons)
        if nex == 0:
            E = utils.sc0(b1, self)
        elif nex == 1:
            a, _, r, _ = abrs
            E = utils.sc1(b1, a, r, perms, self)
        elif nex == 2:
            a, b, r, s = abrs
            E = utils.sc2(a, b, r, s, perms, self)
        else:
            E = 0.0
        E *= int(abs(E) > tol)
        return E

    def random_bitarray_symspace(self) -> Array:
        """
        Generate a random determinant from the full space of all determinants.

        Returns
        -------
        Array
            The bitarray of the determinant
        """
        occa = np.random.choice(int(self._norb/2), self._na, replace=False)
        syma = utils.orb_sym(self._orbsym[2*occa], self._pg_mask)
        occb = np.random.choice(int(self._norb/2), self._nb, replace=False)
        symb = utils.orb_sym(self._orbsym[2*occb+1], self._pg_mask)

        if not (utils.cross_prod_sym(symb, syma, self._pg_mask)
                == self.symmetry):
            return self.random_bitarray_symspace()  # this is recursive??

        ba = np.zeros(self._norb, dtype=int)
        ba[2*occa] = 1
        ba[2*occb+1] = 1

        return ba

    def generate_renorm_excitation(self, ba: ArrayLike) -> list:
        """
        Generate an excitation from the given bitarray.

        Parameters
        ----------
        ba : array_like
            Bitarray for generating the excitation.
        """
        occ = self._orbs[ba == 1]
        occ_ms, occ_sym = self._ms[occ], self._orbsym[occ]
        unocc, virt_ms, virt_sym, nvirt = self.get_virtual_orbitals(occ)

        if np.random.random() < self._psingle:
            nsources = np.count_nonzero(nvirt[(occ_ms, occ_sym)] > 0)
            if nsources > 0:
                nex = 1
                while True:
                    i = np.random.choice(occ)
                    ims, isym = self._ms[i], self._orbsym[i]
                    if nvirt[ims, isym] > 0:
                        break
                allowed = (ims == virt_ms) & (isym == virt_sym)
                a = np.random.choice(unocc[allowed])
                pgen = self._psingle/(nsources*nvirt[ims, isym])
                ba2, perms = utils.get_single_perm(ba, i, a, self._nel)
                hij = utils.sc1(ba, i, a, perms, self)
            else:
                pgen, nex, hij, ba2 = 1.0, None, 0.0, None
        else:
            allowed_excit = False
            i, j = np.random.choice(occ, 2, replace=False)
            ijsym = utils.utils.conj_sym(
                        utils.cross_prod_sym(self._orbsym[i],
                                             self._orbsym[j],
                                             self._pg_mask),
                        self)
            ijms = self._ms[i] + self._ms[j]
            pgen_ij = 2.0/(self._nel*(self._nel-1))

            if ijms == -2:
                for syma in range(self._maxsym):
                    symb = utils.conj_sym(
                        utils.cross_prod_sym(syma, ijsym, self._pg_mask),
                        self)
                    bool1 = nvirt[-1, syma] > 0
                    bool2 = nvirt[-1, symb] > 1
                    bool3 = nvirt[-1, symb] == 1 and syma != symb
                    if bool1 and (bool2 or bool3):
                        allowed_excit = True
                        break
                fac = 2
                shift = 1
            elif ijms == 0:
                for syma in range(self._maxsym):
                    symb = utils.conj_sym(
                        utils.cross_prod_sym(syma, ijsym, self._pg_mask),
                        self)
                    bool1 = nvirt[-1, syma] > 0 and nvirt[1, symb] > 0
                    bool2 = nvirt[1, syma] > 0 and nvirt[-1, symb] > 0
                    if bool1 or bool2:
                        allowed_excit = True
                        break
                fac = 1
                shift = 0
            elif ijms == 2:
                for syma in range(self._maxsym):
                    symb = utils.conj_sym(
                        utils.cross_prod_sym(syma, ijsym, self._pg_mask),
                        self)
                    bool1 = nvirt[1, syma] > 0
                    bool2 = nvirt[1, symb] > 1
                    bool3 = nvirt[1, symb] == 1 and syma != symb
                    if bool1 and (bool2 or bool3):
                        allowed_excit = True
                        break
                fac = 2
                shift = 0

            if allowed_excit:
                nex = 2
                while True:
                    a = np.random.choice(unocc[unocc % fac == shift])
                    imsb = ijms - self._ms[a]
                    isymb = utils.conj_sym(
                        utils.cross_prod_sym(ijsym,
                                             self._orbsym[a],
                                             self._pg_mask),
                        self)
                    bool1 = nvirt[imsb, isymb] > 1
                    bool2 = nvirt[imsb, isymb] == 1
                    bool3 = isymb != self._orbsym[a] or ijms == 0
                    if bool1 or (bool2 and bool3):
                        allowed = (imsb == virt_ms) & (isymb == virt_sym)
                        b = np.random.choice(unocc[allowed])
                        if b != a:
                            break
                if a > b:
                    a, b = b, a

                imsa = self._ms[a]
                imsb = self._ms[b]
                if ijms == -2:
                    n_aij = int(self._norb/2) - self._nb
                    for syma in range(self._maxsym):
                        symb = utils.conj_sym(
                            utils.cross_prod_sym(syma, ijsym, self._pg_mask),
                            self)
                        bool1 = nvirt[-1, symb] == 0
                        bool2 = syma == symb and nvirt[-1, symb] == 1
                        if bool1 or bool2:
                            n_aij -= nvirt[-1, syma]
                    if self._orbsym[a] == self._orbsym[b]:
                        p_aijb = 1/(nvirt[imsa, self._orbsym[a]]-1)
                        p_bija = 1/(nvirt[imsb, self._orbsym[b]]-1)
                    else:
                        p_aijb = 1/(nvirt[imsa, self._orbsym[a]])
                        p_bija = 1/(nvirt[imsb, self._orbsym[b]])
                elif ijms == 0:
                    n_aij = self._norb - self._nel
                    for syma in range(self._maxsym):
                        symb = utils.conj_sym(
                            utils.cross_prod_sym(syma, ijsym, self._pg_mask),
                            self)
                        bool1 = nvirt[-1, symb] == 0
                        bool2 = nvirt[1, symb] == 0
                        if bool1:
                            n_aij -= nvirt[1, syma]
                        if bool2:
                            n_aij -= nvirt[-1, syma]
                    p_aijb = 1/(nvirt[imsa, self._orbsym[a]])
                    p_bija = 1/(nvirt[imsb, self._orbsym[b]])
                elif ijms == 2:
                    n_aij = int(self._norb/2) - self._na
                    for syma in range(self._maxsym):
                        symb = utils.conj_sym(
                            utils.cross_prod_sym(syma, ijsym, self._pg_mask),
                            self)
                        bool1 = nvirt[1, symb] == 0
                        bool2 = syma == symb and nvirt[1, symb] == 1
                        if bool1 or bool2:
                            n_aij -= nvirt[1, syma]
                    if self._orbsym[a] == self._orbsym[b]:
                        p_aijb = 1/(nvirt[imsa, self._orbsym[a]]-1)
                        p_bija = 1/(nvirt[imsb, self._orbsym[b]]-1)
                    else:
                        p_aijb = 1/(nvirt[imsa, self._orbsym[a]])
                        p_bija = 1/(nvirt[imsb, self._orbsym[b]])

                pgen = self._pdouble*pgen_ij*(1/n_aij)*(p_bija+p_aijb)
                ba2, perms = utils.get_double_perm(ba, i, j, a, b, self._nel)
                hij = utils.sc2(i, j, a, b, perms, self)
            else:
                pgen, nex, hij, ba2 = 1.0, None, 0.0, None

        return pgen, hij, nex, ba2

    def print_report(self) -> None:
        """Print information about the system."""
        print(' ---- System information ----')
        print(
            json.dumps(
                {
                    'int_file'  : self._input_file,
                    'UHF'       : self._uhf,
                    'Norb'      : self._norb,
                    'Nvirt'     : self._nvirt,
                    'Nel'       : self._nel,
                    'Na'        : self._na,
                    'Nb'        : self._nb,
                    'MS2'       : self._ms2,
                    'ISYM'      : self._isym,
                    'maxsym'    : self._maxsym,
                    'symmetry'  : self._sym,
                    'pg_mask'   : self._pg_mask,
                    'Href'      : self._ref_eng,
                    'reference' : '%s' % self._ref,
                    'ref_det'   : '%s' % self._ref_det,
                    'p_single'  : self._psingle,
                    'p_double'  : self._pdouble,
                },
                indent=4, ensure_ascii=True)
        )
        print()
        print('  '+'#'*6+' Basis set table start. '+'#'*6)
        print(' '+'-'*50)
        print(' {:>8} {:>10} {:>6} {:>22}'.format(
            'index', 'Symmetry', 'ms', '<i|f|i>'))
        print(' '+'-'*50)
        outstr = ' {:>8} {:>10} {:>6} {:> 22.12E}'
        for i in range(self._norb):
            print(outstr.format(i,
                                self._orbsym[i],
                                self._ms[i],
                                self._eig[i]))
        print(' '+'-'*50)
        print('  '+'#'*6+'  Ba(sis set table end.  '+'#'*6)
        self.print_symmetry_table()
        print('\nEigenvalues')
        self.print_eigenvalues()

    def print_symmetry_table(self) -> None:
        """
        Write out the symmetry table for the system.

        Write out the general pg symmetry table from all combinations
        of point groups first.
        Then write out the resulting point group table from the cross
        product of the orbital point groups.
        """
        print('\n')
        print(' Symmetry cross product (xp) table using:')
        print('    pg_mask: {}'.format(self._pg_mask))
        print('    row = pg symmetry 1')
        print('    col = pg symmetry 2')
        print('    xp[row,col]:')
        print()

        xp = np.zeros((self._maxsym, self._maxsym), dtype=np.int64)
        for isym in range(0, self._maxsym):
            for jsym in range(0, self._maxsym):
                xp_pg = utils.bitarray_pg(isym, jsym, self._pg_mask)
                xp[isym, jsym] = xp_pg

        header = ' sym |'
        pg_rows = [f'  {isym:>2} |' for isym in range(0, self._maxsym)]
        for isym in range(0, self._maxsym):
            header += f' {isym:>2}'
            for jsym in range(0, self._maxsym):
                pg_rows[isym] += f' {xp[isym, jsym]:>2}'

        print(header)
        print(' ' + '-'*int(6 + 3 * (self._maxsym)))
        for isym in range(0, self._maxsym):
            print(pg_rows[isym])

        print()
        print(' Symmetry cross product table from system orbitals:')
        print('    pg_mask: {}'.format(self._pg_mask))
        print('    row = orbital 1')
        print('    col = orbital 2')
        print('    xp[row,col]:')
        print()

        xp = np.zeros((self._norb, self._norb), dtype=np.int64)
        for iorb in range(0, self._norb):
            isym = self._orbsym[iorb]
            for jorb in range(0, self._norb):
                jsym = self._orbsym[jorb]
                xp_pg = utils.bitarray_pg(isym, jsym, self._pg_mask)
                xp[iorb, jorb] = xp_pg

        header = ' orb |'
        pg_rows = [f'  {iorb:>2} |' for iorb in range(0, self._norb)]
        for iorb in range(0, self._norb):
            header += f' {iorb:>2}'
            for jorb in range(0, self._norb):
                pg_rows[iorb] += f' {xp[iorb, jorb]:>2}'

        print(header)
        print(' ' + '-'*int(6 + 3 * (self._norb)))
        for irow in range(0, self._norb):
            print(pg_rows[irow])

    def print_eigenvalues(self,
                          float_fmt: str = ' % 24.16E',
                          int_fmt: str = '%3i') -> None:
        """
        Print eigenvalues.

        Parameters
        ----------
        float_fmt : str
            Format string for floats.
        int_fmt : str
            Format string for integers.
        """
        fmt = float_fmt + f' {int_fmt} {int_fmt} {int_fmt} {int_fmt}'
        inds = np.arange(0, self._norb, 2 - self._uhf)
        for i in inds:
            iout = int(i/(2 - self._uhf)) + 1
            out_tuple = (self._eig[i], iout, 0, 0, 0)
            print(fmt % out_tuple)
