#!/usr/bin/env python

import numpy as np

from typing import Any
from scipy.optimize import newton
from integrals_readin import integral_system
from utilities import orb_sym, cross_prod_pg_sym, sc0
from sympy.utilities.iterables import multiset_permutations as MSP


class SumOverStates:
    def __init__(
                self,
                beta: Any = None,
                fcidump: str = None,
                dtype: Any = np.longdouble,
                grandcanonical: bool = False,
                hilbertspaceonly: bool = False,
                allspin: bool = False,
                allsector: bool = False,
                eigenvalues: bool = False,
                pregenerated_spectrum: bool = None,
                pregenerated_particle_number: int = None,
            ) -> None:

        if beta is None:
            raise ValueError('Please provide an array of beta to treat!')
        if fcidump is None and pregenerated_spectrum is None:
            raise ValueError('Please provide an FCIDUMP system file!')

        self.beta = beta
        self.fcidump = fcidump
        self.dtype = dtype
        self.grandcanonical = grandcanonical
        self.hilbertspaceonly = hilbertspaceonly
        self.allspin = allspin
        self.allsector = allsector
        self.eigenvalues = eigenvalues

        if pregenerated_spectrum is not None:
            if pregenerated_particle_number is None:
                raise ValueError('Please provide the particle number!')
            self.load_pregenerated_spectrum(pregenerated_spectrum,
                                            pregenerated_particle_number)
        else:
            # Load the system class
            self.initialize_system()

            # Get an estimate of the Hilbert space size
            self.estimate_hilbert_space_size()

            if self.hilbertspaceonly:
                return

            # Generate the Hilbert space
            self.generate_determinants()

        # Calculate the thermodynamic quantities for states in Hilbert space.
        self.calculate_thermodynamic_quantities()

    def load_pregenerated_spectrum(self, pregenerated, particles) -> None:
        self.nel = particles
        self.Himin = self.dtype(pregenerated.Ei.min())

        self.Ni = np.array(pregenerated.Ni).astype(self.dtype)
        self.Hi = np.array(pregenerated.Ei).astype(self.dtype)
        self.SHi = np.array(pregenerated.Ei).astype(self.dtype)
        self.SHi -= self.Himin

        intN = self.Ni.copy().astype(int)

        if intN.min() <= particles - 1:
            self.Hinmin = self.Hi[intN == (particles - 1)].min()
        else:
            self.Hinmin = 0.0

        if intN.max() >= particles + 1:
            self.Hipmin = self.Hi[intN == (particles + 1)].min()
        else:
            self.Hipmin = 0.0

        self.ndets = self.Hi.shape[0]
        print(f' Size of the loaded Hilbert space: {self.ndets:<22}',
              flush=True)

    def initialize_system(self):
        self.system = integral_system(
                int_file=self.fcidump,
                verbose=False,
                determinants=False,
                eigenvalues=self.eigenvalues,
            )
        self.eigs = self.system.eig.copy()
        self.norb = self.system.norb
        self.na = self.system.na
        self.nb = self.system.nb
        self.nel = self.system.nel
        self.orbsym = self.system.orbsym
        self.pg_mask = self.system.pg_mask
        self.symmetry = self.system.symmetry

    def estimate_hilbert_space_size(self):
        ndets = 0
        ma = mb = self.norb//2
        for na in range(0, ma+1):
            for nb in range(0, mb+1):
                nconserve = na + nb == self.nel
                msconserve = na == self.na and nb == self.nb

                # If doing GC, make sure N is conserved!
                if not self.grandcanonical and not nconserve:
                    continue

                # If doing a single spin, check here!
                if not self.allspin and not msconserve:
                    continue

                adets = np.math.factorial(ma)
                adets /= np.math.factorial(na)
                adets /= np.math.factorial(ma - na)

                bdets = np.math.factorial(mb)
                bdets /= np.math.factorial(nb)
                bdets /= np.math.factorial(mb - nb)

                ndets += adets*bdets

        print(' Estimate of Hilbert space: '
              f'{ndets:<24.12E} ({int(ndets)})', flush=True)

    def generate_determinants(self):
        self.Ni = []
        self.Hi = []
        self.SHi = []
        self.Himin = 0.0
        self.Hinmin = 0.0
        self.Hipmin = 0.0

        aloc = np.arange(0, self.norb, 2)
        bloc = np.arange(1, self.norb, 2)

        for na in range(0, self.norb//2+1):
            for nb in range(0, self.norb//2+1):
                nconserve = na + nb == self.nel
                msconserve = na == self.na and nb == self.nb

                # If doing GC, make sure N is conserved!
                if not self.grandcanonical and not nconserve:
                    continue

                # If doing a single spin, check here!
                if not self.allspin and not msconserve:
                    continue

                aba = np.zeros(self.norb//2, dtype=int)
                aba[:na] = 1
                aba_store = aba.copy()

                bba = np.zeros(self.norb//2, dtype=int)
                bba[:nb] = 1

                for bba in MSP(bba):
                    bind = 2*np.nonzero(bba)[0] + 1
                    boccsym = self.orbsym[bind]
                    bsym = orb_sym(boccsym, self.pg_mask)

                    for aba in MSP(aba_store):
                        aind = 2*np.nonzero(aba)[0]
                        aoccsym = self.orbsym[aind]
                        asym = orb_sym(aoccsym, self.pg_mask)

                        # Check if doing a single symmetry sector here!
                        if self.allsector:
                            sym = self.symmetry
                        else:
                            sym = cross_prod_pg_sym(bsym, asym, self.pg_mask)

                        if sym == self.symmetry:
                            ba = np.zeros(self.norb, dtype=int)
                            ba[aloc] = aba
                            ba[bloc] = bba

                            Hi = sc0(ba, self.system)

                            self.Ni.append(na + nb)
                            self.Hi.append(Hi)
                            self.SHi.append(Hi)

                            if Hi < self.Himin:
                                self.Himin = Hi
                            if na + nb == self.nel - 1 and Hi < self.Hinmin:
                                self.Hinmin = Hi
                            if na + nb == self.nel + 1 and Hi < self.Hipmin:
                                self.Hipmin = Hi

        self.Ni = np.array(self.Ni).astype(self.dtype)
        self.Hi = np.array(self.Hi).astype(self.dtype)
        self.SHi = np.array(self.SHi).astype(self.dtype)
        self.SHi -= self.dtype(self.Himin)

        self.ndets = self.Hi.shape[0]
        print(f' Size of the hilbert space: {self.ndets:<22}', flush=True)

    def calculate_thermodynamic_quantities(self):
        self.mu0 = self._zero_array()
        self.O0 = self._zero_array()
        self.U0 = self._zero_array()
        self.S0 = self._zero_array()
        self.Cv0 = self._zero_array()

        for index, beta in enumerate(self.beta):
            if (index)%(self.beta.shape[0]//100) == 0:
                print(f' {beta:>8.4f}')
            mu0 = self.calculate_mu0(beta)
            self.mu0[index] = mu0

            P0i = np.exp(-beta*(self.SHi - mu0*self.Ni))
            P0i /= P0i.sum()

            U0 = np.dot(P0i, self.Hi - mu0*self.Ni) + mu0*self.nel
            self.U0[index] = U0

            S0 = -np.log(np.power(P0i, P0i)).sum()
            self.S0[index] = S0

            O0 = U0 - (1.0/beta)*S0 - mu0*self.nel
            self.O0[index] = O0

            Cv0 = self.calculate_specific_heat0(beta, mu0)
            self.Cv0[index] = Cv0

    def calculate_specific_heat0(self, beta, mu0):
        if self.grandcanonical:
            Cv0 = 0.0
        else:
            # U(T) = f(T)/g(T)
            # U'(T) = (f'(T)g(T) - (1/T^2)*f(T)^2)/(g(T)^2)
            # U'(T) = f'(T)/g(T) - (beta f(T)/g(T))^2
            Pi = np.exp(-beta*self.SHi)
            g = np.einsum('i->', Pi)
            f = np.einsum('i,i->', self.SHi, Pi)
            df = beta*beta*np.einsum('i,i,i->', self.SHi, self.SHi, Pi)
            Cv0 = df/g - (beta*f/g)**2.0
        return Cv0

    @staticmethod
    def _nel(mu, beta, Ei, Ni):
        Pi = np.exp(-beta*(Ei - mu*Ni))
        N = np.dot(Pi, Ni)/Pi.sum()
        return N

    @staticmethod
    def _delta_nel(mu, beta, nel, _fnel, Ei, Ni):
        return _fnel(mu, beta, Ei, Ni) - nel

    def calculate_mu0(self, beta):
        if self.grandcanonical:
            mu0 = newton(
                    self._delta_nel,
                    0.5*(self.Hipmin - self.Hinmin),
                    args=(beta, self.nel, self._nel, self.SHi, self.Ni,),
                )
        else:
            mu0 = 0.0
        return mu0

    def _numerical_specific_heat(self, beta: Any) -> None:
        Cv0 = np.gradient(self.U0)/np.gradient(np.divide(1.0, self.beta))

        decimal = int(abs(np.log10(min(
                abs(self.beta[1] - self.beta[0]),
                abs(beta[1] - beta[0]),
            )))) + 2

        mask = np.isin(self.beta.round(decimal), beta.round(decimal))

        self.beta = self.beta[mask]
        self.mu0 = self.mu0[mask]
        self.O0 = self.O0[mask]
        self.U0 = self.U0[mask]
        self.S0 = self.S0[mask]
        self.Cv0 = Cv0[mask]

    def _zero_array(self):
        return np.zeros(self.beta.shape[0], dtype=self.dtype)
