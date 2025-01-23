#!/usr/bin/env python

import numpy as np

from typing import Any
from pandas import DataFrame
from scipy.optimize import newton
from integrals_readin import integral_system
from utilities import orb_sym, cross_prod_pg_sym, sc0


class Analytical:
    def __init__(
                self,
                fcidump: str,
                betas: Any,
                ftype: Any = np.longdouble,
                eigenvalues: bool = False,
            ) -> None:

        self.fcidump = fcidump
        self.betas = betas
        self.Nbeta = len(betas)
        self.ftype = ftype
        self.calculate_eigenvalues = eigenvalues

        self.readsystem()
        self.calculate_thermodynamic_quantities()

        self.data = DataFrame({
                'Beta': betas,
                'mu(0)': self.mu0,
                'Omega(0)': self.Omega0,
                'Omega(1)': self.Omega1,
            })

    def readsystem(self) -> None:
        self.readin = integral_system(
                int_file=self.fcidump,
                verbose=False,
                determinants=False,
                eigenvalues=self.calculate_eigenvalues,
            )

        self.ba0 = self.readin.reference_det
        self.nel = self.readin.nel
        self.h0 = self.readin.h0e
        self.ei = self.readin.eig.copy()
        self.h1 = self.readin.h1e
        self.h2 = self.readin.h2e
        self.homo = self.ei[np.argsort(self.ei)][self.nel-1]
        self.lumo = self.ei[np.argsort(self.ei)][self.nel]

    def calculate_mu0(self, beta: float) -> float:
        def _expi(beta, mu, ei) -> Any:
            return np.exp(beta*(ei - mu))
        def _dN(mu, beta, ei) -> float:
            # dN = \sum_i n_i - N
            # n_i = [1 + e^{beta(e_i - u)}]^-1
            expi = _expi(beta, mu, ei)
            ni = np.power(1.0 + expi, -1.0)
            return ni.sum() - self.nel
        def _ddN(mu, beta, ei) -> float:
            # d/dmu dN = \sum_i d/dmu n_i
            # d/dmu n_i = beta [1 + e^{beta(e_i - u)}]^-2 e^{beta(e_i - u)}
            expi = _expi(beta, mu, ei)
            ni2 = np.power(1.0 + expi, -2.0)
            return beta*np.dot(ni2, expi)
        def _dddN(mu, beta, ei) -> float:
            # d/dmu d/dmu dN = \sum_i d/dmu d/dmu n_i
            # d/dmu d/dmu n_i =
            #   2 beta^2 [1 + e^{beta(e_i - u)}]^-3 [e^{beta(e_i - u)}]^2
            #   - beta^2 [1 + e^{beta(e_i - u)}]^-2 e^{beta(e_i - u)}
            # d/dmu d/dmu n_i =
            #   beta^2 [2 n_i^3 e^{beta(e_i - u)}^2 - n_i^2 e^{beta(e_i - u)}]
            beta2 = beta*beta
            expi = _expi(beta, mu, ei)
            expi2 = np.power(expi, 2.0)
            ni3 = np.power(1.0 + expi, -3.0)
            ni2 = np.power(1.0 + expi, -2.0)
            return beta2*(2.0*ni3*expi2 - ni2*expi).sum()
        mu0 = newton(
                _dN,
                0.0,
                fprime=_ddN,
                fprime2=_dddN,
                args=(beta, self.ei),
                tol=1E-9,
                maxiter=1000,
            )
        return mu0

    def calculate_thermodynamic_quantities(self):
        self.mu0 = self._zero_array()
        self.Omega0 = self._zero_array()
        self.Omega1 = self._zero_array()
        self.U0 = self._zero_array()
        self.U1 = self._zero_array()
        self.S0 = self._zero_array()

        for index, beta in enumerate(self.betas):
            if index % (self.Nbeta//100) == 0:
                print(f'{beta:>9.5}')
            mu0 = self.calculate_mu0(beta)
            self.mu0[index] = mu0

            expi = np.exp(beta*(self.ei - mu0))
            ni = np.power(1.0 + expi, -1.0)

            # Energy equations taken from Eq. 54-57 in:
            # http://dx.doi.org/10.1016/j.chemphys.2016.08.001
            # E^{(0)} = \sum_p \epsilon_p \bar{n}_p
            # \bar{n}_p = [1 + e^{\beta*(\epsilon_p - \mu)}]^{-1}
            U0 = np.dot(self.ei, ni) + self.h0
            self.U0[index] = U0
            # E^{(1)}_\Omega = \sum_p v_{pp} \bar{n}_p
            #              + 1/2 \sum_{pq} <pq||pq> \bar{n}_p \bar{n}_q
            # \bar{v}_{pq} = v_{pq} + \sum_r <pr||qr> \bar{n}_r
            # v_{pq} = -\sum_r <pr||qr> n_r
            # n_r = { 1, r \in occ
            #       { 0, r \in unocc
            vpq = -(np.einsum('prqr,r->pq', self.h2, self.ba0)
                    - np.einsum('prrq,r->pq', self.h2, self.ba0))
            U1_Omega = np.einsum('pp,p->', vpq, ni)
            U1_Omega += 0.5*np.einsum('pqpq,p,q->', self.h2, ni, ni)
            U1_Omega -= 0.5*np.einsum('pqqp,p,q->', self.h2, ni, ni)
            # E^{(1)}_A =
            #     -\sum_p \epsilon_p \bar{v}_pp (\delta \bar{n}_p)/(\delta \mu)
            # \delta \bar{n}_p / \delta \mu = \beta \bar{n}_p (1 - \bar{n}_p)
            bar_vpq = vpq.copy()
            bar_vpq += np.einsum('prqr,r->pq', self.h2, ni)
            bar_vpq -= np.einsum('prrq,r->pq', self.h2, ni)
            delta_ni = beta*ni*(1-ni)
            U1_A = -np.einsum('p,pp,p->', self.ei, bar_vpq, delta_ni)
            self.U1[index] = U1_Omega + U1_A

            # From Eq. 4 in: https://arxiv.org/pdf/1810.03653.pdf
            # Omega0 = E_nuc + 1/\beta \sum_i ln n_i
            Omega0 = self.h0 + -(1.0/beta)*np.log(1.0 + expi).sum()
            # But in the code from:
            # https://github.com/awhite862/ft_mp2_test/blob/f35bd1ab66d0889cfed72811c384477f245d305a/ft_mp2_simple.py
            # We need an extra term:
            # Omega0 += emm
            # emm = ei - mu0
            Omega0 += (self.ei - mu0).sum()
            self.Omega0[index] = Omega0

            # Omega1 = \sum_i (f_ii - e_i)n_i - 1/2 \sum_ij <ij||ij>n_i n_j
            fock = self.h1.copy()
            fock += np.einsum('pjqj,j->pq', self.h2, ni)
            fock -= np.einsum('pjjq,j->pq', self.h2, ni)
            fii = np.einsum('ii->i', fock)
            Omega1 = np.einsum('i,i->', fii - self.ei, ni)
            Omega1 -= 0.5*np.einsum('ijij,i,j->', self.h2, ni, ni)
            Omega1 += 0.5*np.einsum('ijji,i,j->', self.h2, ni, ni)
            self.Omega1[index] = Omega1

    def _zero_array(self):
        return np.zeros(self.Nbeta, dtype=self.ftype)
