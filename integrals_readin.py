#!/usr/bin/env python

import json
import numpy as np
from utilities import get_hij
from utilities import orb_sym as csym
from utilities import generate_bit_arrays
from utilities import cross_prod_pg_sym as xpsym
from excitations import calculate_psingle_pdouble

def ijab_32fold(i,j,a,b):
    r'''
    Generates and returns an array of the 32-fold symmetries for integrals
    indexed by the orbitals. Where we have 8-fold spatial symmetry and
    4-fold spin symmetry, hence 8x4 = 32-fold.
    We assume physicist notation.

    In:
        i,j,a,b: The orbital indexes
    Out:
        perms: The permutation of all indexes based on the 32-fold symmetry.
    '''
    i,j,a,b = 2*i, 2*j, 2*a, 2*b
    perms = [
        [   i,     j,      a,      b    ],
        [   i+1,   j+1,    a+1,    b+1  ],
        [   i,     j+1,    a,      b+1  ],
        [   i+1,   j,      a+1,    b    ],
        [   j,     i,      b,      a    ],
        [   j+1,   i+1,    b+1,    a+1  ],
        [   j,     i+1,    b,      a+1  ],
        [   j+1,   i,      b+1,    a    ],
        [   a,     b,      i,      j    ],
        [   a+1,   b+1,    i+1,    j+1  ],
        [   a,     b+1,    i,      j+1  ],
        [   a+1,   b,      i+1,    j    ],
        [   b,     a,      j,      i    ],
        [   b+1,   a+1,    j+1,    i+1  ],
        [   b,     a+1,    j,      i+1  ],
        [   b+1,   a,      j+1,    i    ],
        [   a,     j,      i,      b    ],
        [   a+1,   j+1,    i+1,    b+1  ],
        [   a,     j+1,    i,      b+1  ],
        [   a+1,   j,      i+1,    b    ],
        [   b,     i,      j,      a    ],
        [   b+1,   i+1,    j+1,    a+1  ],
        [   b,     i+1,    j,      a+1  ],
        [   b+1,   i,      j+1,    a    ],
        [   i,     b,      a,      j    ],
        [   i+1,   b+1,    a+1,    j+1  ],
        [   i,     b+1,    a,      j+1  ],
        [   i+1,   b,      a+1,    j    ],
        [   j,     a,      b,      i    ],
        [   j+1,   a+1,    b+1,    i+1  ],
        [   j,     a+1,    b,      i+1  ],
        [   j+1,   a,      b+1,    i    ],
    ]
    return perms

def ijab_16fold(i,j,a,b):
    r'''
    Generates and returns an array of the 16-fold symmetries for integrals
    indexed by complex orbitals. Where we have 4-fold spatial symmetry and
    4-fold spin symmetry, hence 4x4 = 16-fold.
    We assume physicist notation.

    In:
        i,j,a,b: The orbital indexes
    Out:
        perms: The permutation of all indexes based on the 32-fold symmetry.
    '''
    i,j,a,b = 2*i, 2*j, 2*a, 2*b
    perms = [
        [   i,      j,      a,      b   ],
        [   i+1,    j+1,    a+1,    b+1 ],
        [   i,      j+1,    a,      b+1 ],
        [   i+1,    j,      a+1,    b   ],
        [   a,      b,      i,      j   ],
        [   a+1,    b+1,    i+1,    j+1 ],
        [   a,      b+1,    i,      j+1 ],
        [   a+1,    b,      i+1,    j   ],
        [   j,      i,      b,      a   ],
        [   j+1,    i+1,    b+1,    a+1 ],
        [   j,      i+1,    b,      a+1 ],
        [   j+1,    i,      b+1,    a   ],
        [   b,      a,      j,      i   ],
        [   b+1,    a+1,    j+1,    i+1 ],
        [   b,      a+1,    j,      i+1 ],
        [   b+1,    a,      j+1,    i   ],
    ]
    return perms

def alloc_arrays(norb):
    r'''
    Generates the h2e, h1e and h0e integral arrays given the
    total number of spin orbitals "norb".

    In:
        norb: The total number of spin orbitals
    Out:
        h2e: The two-particle integrals array
        h1e: The one-particle integrals array
        eig: The single particle eigenvalues array
    '''
    h2e = np.zeros((norb,norb,norb,norb))
    h1e = np.zeros((norb,norb))
    eig = np.zeros(norb)
    return h2e, h1e, eig

class integral_system:

    info = \
    r'''
    class `integral_system`

    Read in an integral dump file and return the assorted information
    associated with it as a class object.

    -- Example --
    test_file = 'systems/STRICT-EIGENVALUES-STO3G-STR-H6.FCIDUMP'
    integral_data = integral_system(
                                int_file = test_file,
                                verbose = True,
                                hamiltonian = True,
                            )

    In:
        integral_file (required): The name of the integral file being read in
        comp (defaul=False): A boolean flag which controls the integral
            index symmetry.
        verbose (default=False): Print out the system information when 
            we are done reading it in.
        reference (default=None): Specifies the occupied spin orbitals for
            the systems reference determinant. The default "None" assumes the
            lowest energy orbitals are occupied. If we don't have orbital
            energies we assume the first nel orbitals are occupied.
        eigenvalues (default=False): Calculate the orbital eigenvalues
            for the reference state.
        symmetry (default=self.isym): The point-group symmetry of the
            system integrals we are reading in.
        determinants (default=False): A boolean to control whether we 
            generate all those determinants spanned by the system
            within the point-group symmetry
        hamiltonian (default=False): A boolean to control whether we generate
            the Hamiltonian for the system
    Contains:
        self.info: The string of this information
        self.int_file: The file were the integrals are read from
        self.complex_int: A boolean which controls if we have complex orbtials
        self.uhf: A boolean which controls if the system is unrestricted
        self.h2e: The two-particle integrals
        self.h1e: The one-particle integrals
        self.h0e: The core Hamiltonian
        self.eig: The systems single particle eigenvalues
        self.norb: The number of spin orbitals
        self.nvirt: The number of virtual orbitals in the system
        self.nel: The total number of electrons
        self.na: The number of alpha electrons
        self.nb: The number of beta electrons
        self.orbsym: The orbital point-group (pg) symmetries
        self.ms2: The spin polarization of the system
        self.isym: The ground state point-group of the system
        self.symmetry: The point-group symmetry of the system
        self.maxsym: The maximum point-group symmetry contained by the system
        self.pg_mask: The point-group mask of the system used for pg operations
        self.bitarrays: An array of the bitarray's in the hilbert space
        self.ndets: The total number of determinants in the hilbert space
        self.hii: The diagonal elements of the system Hamiltonian
        self.H: The system hamiltonian generated with the integrals
        self.Href: The reference Fock state
        self.reference_det: The reference bitarray for the reference occupation
        self.psingle: The probability of generating a single excitation
        self.pdouble: The probability of generating a double excitation
        self.orbitals: An array of all orbital indexes
    '''

    def __init__(
                self,
                int_file = None,
                comp = False,
                verbose = False,
                reference = None,
                eigenvalues = False,
                symmetry = None,
                determinants = False,
                hamiltonian = False,
            ):

        try:
            open(int_file, 'r').close()
        except (FileNotFoundError, TypeError) as error:
            print(' Integral file: \n\n %s \n\n does not exists!' % int_file)
            print('\n Please review the information below:')
            print(self.info)
            exit(1)

        self.int_file = int_file
        self.complex_int = comp
        self.uhf = False
        self.isym = 0
        self.h0e = 0.0

        with open(int_file, 'r') as open_int_file:
            footer = False
            for line in open_int_file:
                line = line.replace('\n','')
                if not footer:
                    if line[-1] != ',':
                        line = line + ','
                    if 'UHF' in line and ('true' in line or 'TRUE' in line):
                        self.uhf = True

                if footer:
                    ls = line.split()
                    eri = float(ls[0])
                    i, a, j, b = [int(d)-1 for d in ls[1:]]
                    ijab = self.permute_ijab(i,j,a,b)
                    self.integral_case(i,j,a,b)
                    if self.case_h2e:
                        for i, j, a, b in ijab:
                            self.h2e[i,j,a,b] = eri
                    elif self.case_h1e:
                        for i, j, a, b in ijab:
                            self.h1e[i,a] = eri
                    elif self.case_eig:
                        self.eig[2*i:2*i+2] = eri
                    elif self.case_h0e:
                        self.h0e = eri
                elif '/' in line or 'END' in line:
                    footer = True
                    self.nb = int((self.nel - self.ms2) / 2)
                    self.na = self.nel - self.nb
                    self.norb -= int(self.uhf  * (self.norb/2))
                    self.ms = np.array([(i+1)%2-i%2 for i in range(self.norb)])
                    self.h2e, self.h1e, self.eig = alloc_arrays(self.norb)
                    self.nvirt = self.norb - self.nel
                elif 'ORBSYM' in line:
                    self.orbsym = line.split('=')[-1].split(',')[:-1]
                    self.orbsym = np.array(self.orbsym).astype(int) - 1
                    self.orbsym = np.repeat(self.orbsym,2)
                else:
                    ls = line.split(',')
                    for ld in ls:
                        if 'NORB' in ld:
                            self.norb = int(2*self.ld_strip(ld))
                        if 'NELEC' in ld:
                            self.nel = self.ld_strip(ld)
                        if 'MS2' in ld:
                            self.ms2 = self.ld_strip(ld)
                        if 'ISYM' in ld:
                            self.isym = self.ld_strip(ld) - 1

        self.maxsym = int(2**np.ceil(np.log(np.max(self.orbsym)+1)/np.log(2)))
        self.pg_mask = self.maxsym - 1

        if symmetry is not None:
            self.symmetry = symmetry
            if self.symmetry not in self.orbsym:
                error_msg  = ' The provided symmetry is not within \n'
                error_msg += ' the symmetries spanned by the system! \n'
                error_msg += ' provided symmetry: %s \n' % self.symmetry
                error_msg += ' Symmetries spanned by system: %s' % self.orbsym
                raise ValueError(error_msg)
        else:
            self.symmetry = self.isym

        if reference is not None:
            self.reference = np.array(reference)
        elif np.sum(self.eig) != 0.0:
            self.reference = np.argsort(self.eig)[:self.nel]
        else:
            self.reference = np.arange(self.nel)

        aba_chk = self.orbsym[self.reference[self.reference % 2 == 0]]
        bba_chk = self.orbsym[self.reference[self.reference % 2 != 0]]
        asym_chk = csym(aba_chk, self.pg_mask)
        bsym_chk = csym(bba_chk, self.pg_mask)
        sym_chk = xpsym(bsym_chk, asym_chk, self.pg_mask)
        if sym_chk != self.symmetry:
            error_msg  = ' Reference determinant is not within the \n'
            error_msg += ' the symmetry of the system!'
            raise ValueError(error_msg)

        if eigenvalues:
            self.generate_orbital_eigenvalues()

        self.orbs = np.arange(self.norb)
        self.reference_det = np.zeros(self.norb, dtype=int)
        self.reference_det[self.reference] = 1
        self.psingle, self.pdouble = calculate_psingle_pdouble(self.orbs,
                                        self.ms,self.orbsym,self.maxsym,
                                        self.pg_mask,self.reference_det)

        self.Href  = 0.5*(np.diag(self.h1e)[self.reference]).sum()
        self.Href += 0.5*self.eig[self.reference].sum() + self.h0e

        if determinants or hamiltonian:
            self.generate_determinants()
        else:
            self.ndets, self.bitarrays = None, None

        if hamiltonian:
            self.generate_hamiltonian()
        else:
            self.hii, self.H = None, None

        if verbose:
            self.report()

    def report(self):
        print(' ---- System information ----')
        print(
            json.dumps(
                {
                    'int_file'  : self.int_file,
                    'UHF'       : self.uhf,
                    'Norb'      : self.norb,
                    'Nvirt'      : self.nvirt,
                    'Nel'       : self.nel,
                    'Na'        : self.na,
                    'Nb'        : self.nb,
                    'MS2'       : self.ms2,
                    'ISYM'      : self.isym,
                    'maxsym'    : self.maxsym,
                    'symmetry'  : self.symmetry,
                    'pg_mask'   : self.pg_mask,
                    'Href'      : self.Href,
                    'reference' : '%s' % self.reference,
                    'ref_det'   : '%s' % self.reference_det,
                    'p_single'  : self.psingle,
                    'p_double'  : self.pdouble,
                },
            indent=4, ensure_ascii=True)
        )
        print()
        print('  '+'\/'*6+' Basis set table start. '+'\/'*6)
        print(' '+'-'*50)
        print(' {:>8} {:>10} {:>6} {:>22}'\
                .format('index','Symmetry','ms','<i|f|i>'))
        print(' '+'-'*50)
        outstr = ' {:>8} {:>10} {:>6} {:> 22.12E}'
        for i in range(self.norb):
            print(outstr.format(i,self.orbsym[i],self.ms[i],self.eig[i]))
        print(' '+'-'*50)
        print('  '+'/\\'*6+'  Basis set table end.  '+'/\\'*6)

    def permute_ijab(self,i,j,a,b):
        if self.complex_int:
            return ijab_16fold(i,j,a,b)
        else:
            return ijab_32fold(i,j,a,b)

    def integral_case(self,i,j,a,b):
        self.case_h2e = np.sign(b) != -1
        self.case_h1e = np.sign(a) != -1 and not self.case_h2e
        self.case_eig = np.sign(i) != -1 and not self.case_h1e
        self.case_h0e = np.sign(i) == -1

    def ld_strip(self,line_data):
        return int(line_data.split('=')[-1])

    def generate_orbital_eigenvalues(self):
        '''
        Math from Szabo and Ostlund:
        e_a = <a|h|a> + \sum_{b != a}^{N} <ab|ab> - <ab|ba>
        Note that, b != a only applies when a is occupied
        and b is always occupied
        '''
        for a in range(self.norb):
            self.eig[a] = self.h1e[a,a]
            for b in self.reference[self.reference != a]:
                self.eig[a] += self.h2e[a,b,a,b]
                self.eig[a] -= self.h2e[a,b,b,a]

    def generate_determinants(self):
        self.ndets, self.bitarrays = generate_bit_arrays(self.norb, self.na,
                                        self.nb, self.orbsym, self.pg_mask,
                                        self.symmetry)

    def generate_hamiltonian(self):
        self.hii = np.array([get_hij(b,b,self) for b in self.bitarrays])
        esortind = np.argsort(self.hii)
        self.hii = self.hii[esortind]
        self.bitarrays = self.bitarrays[esortind]
        self.H = np.diag(self.hii)
        for i, b1 in enumerate(self.bitarrays):
            for j, b2 in enumerate(self.bitarrays[i+1:]):
                j += i + 1
                hij = get_hij(b1,b2,self)
                self.H[i,j], self.H[j,i] = hij, hij

    def dumpeigs(self, float_fmt=' % 24.16E', int_fmt='%3i'):
        fmt = float_fmt + f' {int_fmt} {int_fmt} {int_fmt} {int_fmt}'
        inds = np.arange(0,self.norb,2)
        for i in inds:
            iout = int(i/2)+1
            out_tuple = (self.eig[i], iout, 0, 0, 0)
            print(fmt % out_tuple)

if __name__ == '__main__':
    test_file = 'systems/STRICT-STO3G-STR-H4.FCIDUMP'
    sys = integral_system(
                        int_file = test_file,
                        verbose = True,
                        eigenvalues = True,
                        reference = [0,1,4,5],
                        hamiltonian = True,
                    )

