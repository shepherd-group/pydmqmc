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
                only_canonical=False,
                skip_calculation=False,
                all_spin=False,
                eigenvalues=False,
            ):

        self.fcidump_file = fcidump_file
        self.ftype = ftype
        self.N = len(betas)
        self.betas = betas
        self.only_canonical = only_canonical
        self.skip_calculation = skip_calculation
        self.all_spin = all_spin
        self.eigenvalues = eigenvalues

        self.initialize_system()
        self.estimate_hilbert_space_size()

        if self.skip_calculation:
            return

        self.generate_determinants(
                self.system.norb,
                self.system.na,
                self.system.nb,
                self.system.orbsym,
                self.system.pg_mask,
                self.system.symmetry,
            )

        self.calculate_thermodynamic_quantities()

    def initialize_system(self):
        self.system = integral_system(
                int_file=self.fcidump_file,
                verbose=False,
                determinants=False,
                eigenvalues=self.eigenvalues,
            )
        self.eigs = self.system.eig.copy()

    def estimate_hilbert_space_size(self):
        ndets = 0
        ma = mb = self.system.norb//2
        #for na in range(0, min(self.system.nel+1, self.system.norb//2+1)):
        #    for nb in range(0, min(self.system.nel+1, self.system.norb//2+1)):
        for na in range(0, self.system.norb//2+1):
            for nb in range(0, self.system.norb//2+1):
                if self.only_canonical and na + nb != self.system.nel:
                    continue
                if not self.all_spin:
                    if self.only_canonical and na != self.system.na:
                        continue
                    if self.only_canonical and nb != self.system.nb:
                        continue

                adets = np.math.factorial(ma)
                adets /= np.math.factorial(na)
                adets /= np.math.factorial(ma - na)

                bdets = np.math.factorial(mb)
                bdets /= np.math.factorial(nb)
                bdets /= np.math.factorial(mb - nb)

                ndets += adets*bdets

        print(f' Estimate of Hilbert space: {ndets:<24.12E}', flush=True)

    def generate_determinants(self, norb, na, nb, orbsym, pg_mask, symmetry):
        self.E0i, self.E1i, self.Ni, self.Nconserve = [], [], [], []
        self.E0min, self.E0nmin, self.E0pmin = 0.0, 0.0, 0.0

        aloc = np.arange(0, norb, 2)
        bloc = np.arange(1, norb, 2)

        self.nel = na + nb
        refnel = na + nb
        refna = na
        refnb = nb

        #for na in range(0, min(refnel+1, norb//2+1)):
        #    for nb in range(0, min(refnel+1, norb//2+1)):
        for na in range(0, norb//2+1):
            for nb in range(0, norb//2+1):
                if self.only_canonical and na + nb != self.nel:
                    continue
                if not self.all_spin:
                    if self.only_canonical and na != refna:
                        continue
                    if self.only_canonical and nb != refnb:
                        continue

                aba = np.zeros(norb//2, dtype=int)
                aba[:na] = 1
                aba_store = aba.copy()

                bba = np.zeros(norb//2, dtype=int)
                bba[:nb] = 1
                beta_iterator = MSP(bba)

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
                        if sym == symmetry or not self.only_canonical:
                            if na + nb == refnel:
                                self.Nconserve.append(1)
                            else:
                                self.Nconserve.append(0)

                            ba = np.zeros(norb, dtype=int)
                            ba[aloc] = aba
                            ba[bloc] = bba

                            E0i = self.eigs[ba == 1].sum() + self.system.h0e
                            self.E0i.append(E0i)
                            if E0i < self.E0min:
                                self.E0min = E0i
                            if na + nb == refnel - 1 and E0i < self.E0nmin:
                                self.E0nmin = E0i
                            if na + nb == refnel + 1 and E0i < self.E0pmin:
                                self.E0pmin = E0i

                            Hii = sc0(ba, self.system)
                            E1i = Hii - E0i
                            self.E1i.append(E1i)

                            self.Ni.append(na + nb)

        self.E0i = np.array(self.E0i).astype(self.ftype)
        self.E1i = np.array(self.E1i).astype(self.ftype)
        self.Ni = np.array(self.Ni).astype(self.ftype)
        self.Nconserve = np.array(self.Nconserve).astype(bool)

        if self.only_canonical:
            self.E0i = self.E0i[self.Nconserve]
            self.E1i = self.E1i[self.Nconserve]
            self.Ni = self.Ni[self.Nconserve]

        self.SE0i = self.E0i - self.E0min
        self.SE1i = self.E1i - self.E0min

        self.ndets = self.E0i.shape[0]
        print(f' Size of the hilbert space: {self.ndets:<22}', flush=True)

    def calculate_thermodynamic_quantities(self):
        self.mu0 = self._zero_array()
        self.O0 = self._zero_array()
        self.U0 = self._zero_array()
        self.S0 = self._zero_array()
        self.Cv0 = self._zero_array()
        self.mu1 = self._zero_array()
        self.O1 = self._zero_array()
        self.U1 = self._zero_array()
        self.S1 = self._zero_array()
        self.Cv1 = self._zero_array()

        for index, beta in enumerate(self.betas):
            print(beta)
            mu0 = self.calculate_mu0(beta)
            self.mu0[index] = mu0

            P0i = np.exp(-beta*(self.SE0i - mu0*self.Ni))

            mu1 = self.calculate_mu1(beta, P0i)
            self.mu1[index] = mu1

            P1 = np.dot(-beta*self.E1i + beta*mu1*self.Ni, P0i)

            P0 = P0i.sum()
            P0i /= P0

            U0 = np.dot(P0i, self.E0i - mu0*self.Ni) + mu0*self.nel
            self.U0[index] = U0

            S0 = -np.log(np.power(P0i, P0i)).sum()
            self.S0[index] = S0

            O0 = U0 - (1.0/beta)*S0 - mu0*self.nel
            self.O0[index] = O0

            O1 = np.dot(self.E1i, P0i) - mu1*self.nel
            self.O1[index] = O1

            U1 = O1 + mu1*self.nel + beta*O1*(U0 - mu0*self.nel)
            U1 -= beta*np.einsum('i,i,i->', P0i,
                                 self.E0i - mu0*self.Ni,
                                 self.E1i - mu1*self.Ni)
            self.U1[index] = U1

            S1 = -beta*(O1 - U1 + mu1*self.nel)
            self.S1[index] = S1

            Cv0 = self.calculate_specific_heat0(beta, mu0)
            self.Cv0[index] = Cv0

            Cv1 = self.calculate_specific_heat1(beta, mu0, mu1)
            self.Cv1[index] = Cv1

    def calculate_specific_heat0(self, beta, mu0):
        if not self.only_canonical:
            return 0.0
        # Warning, only true for CE where mu is fixed
        ai = mu0*self.Ni - self.E0i
        ei = np.exp(beta*ai)
        d1 = np.einsum('i->', ei)
        n1 = np.einsum('i,i,i->', self.E0i, ai, ei)
        n2 = np.einsum('i,i->', self.E0i, ei)
        n3 = np.einsum('i,i->', ai, ei)
        return -beta*beta*(n1/d1 - (n2*n3)/(d1*d1))

    def calculate_specific_heat1(self, beta, mu0, mu1):
        if not self.only_canonical:
            return 0.0
        # Warning, only true for CE where mu is fixed
        ai = mu0*self.Ni - self.E0i
        ei = np.exp(beta*ai)
        d1 = np.einsum('i->', ei)
        n3 = np.einsum('i,i->', ai, ei)
        def _expect(obs):
            return np.einsum('i,i->', obs, ei)/d1
        def _delta_expect(obs):
            return -beta*beta*(np.einsum('i,i,i->', obs, ai, ei)/d1
                               - (np.einsum('i,i->', obs, ei)*n3)/(d1*d1))
        dE1 = _delta_expect(self.E1i)
        dbeta_E0E1 = beta*beta*_expect(self.E0i*self.E1i)
        dbeta_E0E1 -= beta*_delta_expect(self.E0i*self.E1i)
        dbeta_E0_E1 = -beta*beta*_expect(self.E0i)*_expect(self.E1i)
        dbeta_E0_E1 += beta*_delta_expect(self.E0i)*_expect(self.E1i)
        dbeta_E0_E1 += beta*_expect(self.E0i)*_delta_expect(self.E1i)
        return dE1 + dbeta_E0E1 + dbeta_E0_E1

    def calculate_mu0(self, beta):
        if self.only_canonical:
            return 0.0
        def _f_mu(mu, beta):
            hi = np.exp(-beta*(self.SE0i - mu*self.Ni))
            g = np.dot(hi, self.Ni)
            h = np.einsum('i->', hi)
            return g/h - self.nel
        def _df_mu(mu, beta):
            # f(u) = g(u)/h(u)
            # f'(u) = (g'(u)h(u) - g(u)h'(u))/h(u)^2
            # h'(u) = beta*g(u)
            hi = np.exp(-beta*(self.SE0i - mu*self.Ni))
            g = np.dot(hi, self.Ni)
            h = np.einsum('i->', hi)
            dg = beta*np.einsum('i,i,i->', self.Ni, self.Ni, hi)
            dh = beta*g
            return (h*dg - g*dh)/(h**2.0)
        return newton(_f_mu, 0.5*(self.E0pmin - self.E0nmin),
                      fprime=_df_mu, args=(beta,))

    def calculate_mu1(self, beta, P0i):
        if self.only_canonical:
            return 0.0
        def _f_mu(mu, beta, P0i):
            P1i = P0i*(-beta*self.SE1i + beta*mu*self.Ni)
            P1i /= P1i.sum()
            return np.dot(self.Ni, P1i) - self.nel
        def _f_mu_prime(mu, beta, P0i):
            hi = P0i*(-beta*self.SE1i + beta*mu*self.Ni)
            g = np.dot(self.Ni, hi)
            h = hi.sum()
            dg = np.dot(P0i, beta*self.Ni*self.Ni)
            dh = np.dot(P0i, beta*self.Ni)
            return (h*dg - g*dh)/h**2.0
        return newton(_f_mu, 0.0, fprime=_f_mu_prime, args=(beta, P0i))
        #else:
        #    mu1 = np.dot(P0i, self.SE1i*(self.Ni - self.nel))
        #    mu1 /= np.dot(P0i, self.Ni*(self.Ni - self.nel))
        #    return mu1

    def _zero_array(self):
        return np.zeros(self.N, dtype=self.ftype)
