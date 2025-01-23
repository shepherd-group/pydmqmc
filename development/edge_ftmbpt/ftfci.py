#!/usr/bin/env python

import numpy as np
import pandas as pd


class ftFCI:
    ''' A simple class for generating and storing ft-FCI data.
    '''
    def __init__(
                self,
                hande_file,
                betas,
                ftype=np.float64,
                fci=None,
                entropy=False,
                skip=False,
            ):
        self.hande_file = hande_file
        self.ftype = ftype
        self.N = len(betas)
        self.betas = betas
        self.temp = betas.copy()
        self.temp[0] = 1E-8
        self.temp = 1/self.temp

        if skip:
            return

        if fci is None:
            self.read_fci()
        else:
            self.fci = fci
        self.sum_over_states()
        self.calculate_specific_heat()
        if entropy:
            self.calculate_entropy()

    def read_fci(self):
        eigenvalues = []
        use_line = False
        with open(self.hande_file, 'r') as stream:
            for line in stream:
                line_data = line.split()
                if use_line and len(line_data) == 0:
                    break
                elif use_line:
                    eigenvalues.append(float(line_data[-1]))
                elif 'State     Energy' in line:
                    use_line = True
        self.fci = np.array(eigenvalues, dtype=self.ftype)
        self.Sfci = self.fci.copy() - self.fci[0]

    def sum_over_states_for_difference(self, betas):
        self.dftfci = np.zeros(len(betas), dtype=self.ftype)
        for i, b in enumerate(betas):
            z = np.exp(-b*self.Sfci)
            z /= z.sum()
            self.dftfci[i] = np.sum(z*self.fci)

    def sum_over_states(self):
        self.ftfci = np.zeros(self.N, dtype=self.ftype)
        for i, b in enumerate(self.betas):
            z = np.exp(-b*self.Sfci)
            z /= z.sum()
            self.ftfci[i] = np.sum(z*self.fci)

    def calculate_specific_heat(self):
        r''' Find the analytical specific heat Cv(b),
        given that:
            E(b) = X(b)/Z(b)
        where
            X(b) = \sum_i E_i exp(-b E_i);
            Z(b) = \sum_i exp(-b E_i).
        Then differentiating with respect to b we have:
            E'(b) = [ X'(b)Z(b) - X(b)Z'(b) ] / Z(b)^2,
        where
            X'(b) = \sum_i -E_i^2 exp(-b E_i);
            Z'(b) = \sum_i -E_i exp(-b E_i).
        Finally, the definition of Cv(b) is:
            Cv(b) = - b^2 E'(b) = - (1/[kb T^2]) E'(b)
        where
            kb = 1 Ha;
            b = 1/T.
        '''
        gradient = np.zeros(self.N, dtype=self.ftype)
        for i, b in enumerate(self.betas):
            z = np.exp(-b * self.Sfci)
            x = self.Sfci*z
            Z = z.sum()
            X = x.sum()
            Xprime = (-self.Sfci*x).sum()
            Zprime = -X
            gradient[i] = (Xprime*Z - X*Zprime)/(Z**2.0)
        self.gradient = gradient
        self.Cv = -self.betas*self.betas*gradient

    def calculate_specific_heat_for_difference(self, betas):
        dgradient = np.zeros(len(betas), dtype=self.ftype)
        for i, b in enumerate(betas):
            z = np.exp(-b * self.Sfci)
            x = self.Sfci*z
            Z = z.sum()
            X = x.sum()
            Xprime = (-self.Sfci*x).sum()
            Zprime = -X
            dgradient[i] = (Xprime*Z - X*Zprime)/(Z**2.0)
        self.dgradient = dgradient
        self.dCv = -self.betas*self.betas*dgradient

    def calculate_entropy(self):
        ''' Calculate the analytical entropy.
        For more information, see: 10.1007/s00214-014-1487-4
            S = - k_B Tr[W ln(W)]
            W = exp(-b H)/Z
            Z = Tr[exp(-b H)]
            k_B = 1.0 Ha
        '''
        self.entropy = np.zeros(self.N, dtype=self.ftype)
        for i, b in enumerate(self.betas):
            z = np.exp(-b*self.Sfci)
            z /= z.sum()
            S = -(z*np.log(z)).sum()
            self.entropy[i] = S
