#!/usr/bin/env python

import sys
import time
import numpy as np

from scipy.optimize import newton
from integrals_readin import integral_system
from utilities import orb_sym, cross_prod_pg_sym, sc0
from sympy.utilities.iterables import multiset_permutations as MSP


class SumOverStates:
    def __init__(
                self,
                fcidump_file,
                betas,
                ftype=np.longdouble,
            ):

        self.fcidump_file = fcidump_file
        self.ftype = ftype
        self.N = len(betas)
        self.betas = betas

        self.initialize_pydmqmc()
        self.generate_eigenspectrum(
                self.pydmqmc.norb,
                self.pydmqmc.na,
                self.pydmqmc.nb,
                self.pydmqmc.orbsym,
                self.pydmqmc.pg_mask,
                self.pydmqmc.symmetry,
            )

        FD = self.calculate_all_quantities(self.ei, self.Sei)
        self.Cv, self.E, self.S = FD

    def initialize_pydmqmc(self):
        self.pydmqmc = integral_system(
                int_file=self.fcidump_file,
                verbose=False,
                determinants=False,
                eigenvalues=True,
            )
        self.eigs = self.pydmqmc.eig.copy()

    def generate_eigenspectrum(self, norb, na, nb, orbsym, pg_mask, symmetry):
        self.Hii, self.H00 = [], 0.0
        self.ei, self.e0 = [], 0.0

        aba = np.zeros(norb//2, dtype=int)
        aba[:na] = 1
        aba_store = aba.copy()

        bba = np.zeros(norb//2, dtype=int)
        bba[:nb] = 1
        beta_iterator = MSP(bba)

        aloc = np.arange(0, norb, 2)
        bloc = np.arange(1, norb, 2)

        ncal = 0
        for bba in beta_iterator:
            bind = 2*np.nonzero(bba)[0] + 1
            boccsym = orbsym[bind]
            bsym = orb_sym(boccsym, pg_mask)

            alpha_iterator = MSP(aba_store)
            for aba in alpha_iterator:
                aind = 2*np.nonzero(aba)[0]
                aoccsym = orbsym[aind]
                asym = orb_sym(aoccsym, pg_mask)

                sym = cross_prod_pg_sym(bsym, asym, pg_mask)
                if sym == symmetry:

                    ncal += 1
                    ba = np.zeros(norb, dtype=int)
                    ba[aloc] = aba
                    ba[bloc] = bba

                    hii = sc0(ba, self.pydmqmc)
                    self.Hii.append(hii)
                    if hii < self.H00:
                        self.H00 = hii

                    ei = self.eigs[ba == 1].sum() + self.pydmqmc.h0e
                    self.ei.append(ei)
                    if ei < self.e0:
                        self.e0 = ei

        # Create a shifted array for better numerical stability in the
        # calculation of the energy, specific heat capacity, and entropy
        self.Hii = np.array(self.Hii).astype(self.ftype)
        self.SHii = self.Hii - self.H00
        self.ei = np.array(self.ei).astype(self.ftype)
        self.Sei = self.ei - self.e0
        self.ndets = self.Hii.shape[0]
        print(f' Final size of Hilbert space: {self.ndets:<22}', flush=True)

    def calculate_all_quantities(self, eigenvalues, shifted_eigenvalues):
        ''' This routine seeks to minimize the computational overhead,
        in both CPU time and memory, for calculating the energy, entropy and
        specific heat capacities.

        This is achieved by only calculating quantities as needed, and a
        single time only. Rather than repeatedly for each call to calculate
        a given quantity.

        Furthermore, where possible, we employ the numpy dot product routine
        which will implicitly parallelize and/or use efficient math libraries
        like BLAS when this is possible. Praise NumPy!

        A single test with BeH2 (84992 determinants) showed that the
        compute time reduction was:
            Time using separate routines: 1.011224 (min)
            Time using simultaneous routines: 0.667151 (min)
        which is about a 34% reduction in runtime, pretty neat!

        During the same test, we also ensured the quantities were unchanged
        from the separate routines.
        '''
        Cv = np.zeros(self.N, dtype=self.ftype)
        energy = np.zeros(self.N, dtype=self.ftype)
        entropy = np.zeros(self.N, dtype=self.ftype)

        #print(
        #        f'{"Beta":>12} '
        #        f'{"Energy":>26} '
        #        f'{"Entropy":>26} '
        #        f'{"Cv":>26} '
        #    , flush=True)
        for i, b in enumerate(self.betas):
            z = np.exp(-b*shifted_eigenvalues)
            x = z*shifted_eigenvalues
            Z = z.sum()
            X = x.sum()
            Xprime = np.dot(-shifted_eigenvalues, x)
            Zprime = -X

            gradient = (Xprime*Z - X*Zprime)/(Z**2.0)
            Cv[i] = -b*b*gradient

            z /= Z
            energy[i] = np.dot(z, eigenvalues)

            S = -np.dot(z, np.log(z))
            entropy[i] = S

            #print(
            #        f'{b:>12.4f} '
            #        f'{energy[i]:> 26.16f} '
            #        f'{entropy[i]:>26.16f} '
            #        f'{Cv[i]:>26.16f} '
            #    , flush=True)

        return Cv, energy, entropy
