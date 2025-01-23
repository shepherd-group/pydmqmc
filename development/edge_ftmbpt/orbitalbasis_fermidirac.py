#!/usr/bin/env python

import numpy as np

from scipy.optimize import newton
from integrals_readin import integral_system
from utilities import orb_sym, cross_prod_pg_sym, sc0


class OrbitalBasis:
    def __init__(
                self,
                fcidump_file,
                betas,
                ftype=np.longdouble,
                eigenvalues=False,
            ):

        self.fcidump_file = fcidump_file
        self.ftype = ftype
        self.N = len(betas)
        self.betas = betas

        self.initialize_system(eigenvalues)
        self.generate_thermodynamic_quantities()

    def initialize_system(self, eigenvalues):
        self.system = integral_system(
                int_file=self.fcidump_file,
                verbose=False,
                determinants=False,
                eigenvalues=eigenvalues,
            )
        self.nel = self.system.nel
        self.h0 = self.system.h0e
        self.ei = self.system.eig.copy()
        self.h1 = self.system.h1e
        self.h2 = self.system.h2e
        self.homo = self.ei[np.argsort(self.ei)][self.nel-1]
        self.lumo = self.ei[np.argsort(self.ei)][self.nel]

    def generate_thermodynamic_quantities(self):
        self.mu0 = self._zero_array()
        self.omega0 = self._zero_array()
        self.U0 = self._zero_array()
        self.S0 = self._zero_array()
        self.mu1 = self._zero_array()
        self.omega1 = self._zero_array()
        self.U1 = self._zero_array()
        self.S1 = self._zero_array()

        for index, beta in enumerate(self.betas):
            mu0 = self.get_mu0(beta)
            self.mu0[index] = mu0

            self.set_occupations(beta, mu0)

            omega0 = self.h0 + (1.0/beta)*self.occupation_log(beta, mu0).sum()
            self.omega0[index] = omega0

            U0 = self.h0 + np.dot(self.ei, self.fpn)
            self.U0[index] = U0

            S0 = beta*(U0 - mu0*self.nel - omega0)
            self.S0[index] = S0

            self.set_ftfock(beta, mu0)

            mu1 = np.einsum('pp,p,p->', self.ftfock, self.fpn, self.fpp)
            mu1 /= np.einsum('p,p->', self.fpn, self.fpp)
            self.mu1[index] = mu1

            omega1 = np.einsum('pp,p->', self.ftfock, self.fpn)
            omega1 -= 0.5*np.einsum('pqpq,p,q->', self.h2, self.fpn, self.fpn)
            omega1 += 0.5*np.einsum('pqqp,p,q->', self.h2, self.fpn, self.fpn)
            omega1 -= mu1*self.nel
            self.omega1[index] = omega1

            self.set_shifted_ftfock(mu1)
            U1 = omega1 + mu1*self.nel
            U1 -= beta*np.einsum('pp,p,p,p->', self.sftfock,
                                 self.ei, self.fpn, self.fpp)
            self.U1[index] = U1

            S1 = beta*(U1 - mu1*self.nel - omega1)
            self.S1[index] = S1

    def set_occupations(self, beta, mu0):
        self.fpn = np.divide(1.0, 1.0 + np.exp(beta*(self.ei - mu0)))
        self.fpp = 1.0 - self.fpn

    def get_mu0(self, beta):
        def _f_mu(mu, beta):
            self.set_occupations(beta, mu)
            return self.fpn.sum() - self.nel
        def _df_mu(mu, beta):
            self.set_occupations(beta, mu)
            return beta*np.dot(self.fpp, self.fpn)
        def _ddf_mu(mu, beta):
            self.set_occupations(beta, mu)
            df = beta*self.fpp*self.fpn
            return (2*beta*df*self.fpn - beta*df).sum()
        return newton(_f_mu, 0.5*(self.homo+self.lumo),
                      fprime=_df_mu, fprime2=_ddf_mu, args=(beta,))

    def occupation_log(self, beta, mu0):
        # Algebraic manipulation to numerically stabilize the ln(f_i^+)
        term1 = beta*(self.ei - mu0)
        term2 = -np.log(1.0 + np.exp(beta*(self.ei - mu0)))
        return term1 + term2

    def set_ftfock(self, beta, mu0):
        self.set_occupations(beta, mu0)
        self.ftfock = self.h1.copy() - np.diag(self.ei)
        self.ftfock += np.einsum('prqr,r->pq', self.h2, self.fpn)
        self.ftfock -= np.einsum('prrq,r->pq', self.h2, self.fpn)

    def set_shifted_ftfock(self, mu1):
        self.sftfock = self.ftfock.copy()
        self.sftfock -= mu1*np.eye(self.ftfock.shape[0])

    def _zero_array(self):
        return np.zeros(self.N, dtype=self.ftype)
