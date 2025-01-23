#!/usr/bin/env python

import numpy as np
import pandas as pd

from manybody_fermidirac02 import SumOverStates


def get_data(system: str):

    fcidump_dict = {
            'Be': 'EIGENVALUES-Be-aug-cc-pVDZ.FCIDUMP',
            'BeH2': 'EIGENVALUES-BeH2-Be-cc-pVDZ-H-DZ.FCIDUMP',
            'CO': 'EIGENVALUES-CO-STO-3G.FCIDUMP',
            'HCN': 'EIGENVALUES-HCN-STO-3G.FCIDUMP',
            'LiF': 'EIGENVALUES-LiF-STO-3G.FCIDUMP',
            'N2': 'EIGENVALUES-N2-STO-3G.FCIDUMP',
            'equilibrium-H4': 'EIGENVALUES-equilibrium-H4-cc-pVDZ.FCIDUMP',
            'equilibrium-H8': 'EIGENVALUES-equilibrium-H8-STO-3G.FCIDUMP',
            'stretched-H8': 'EIGENVALUES-stretched-H8-STO-3G.FCIDUMP',
            'CH4': 'EIGENVALUES-CH4-cc-pVDZ.FCIDUMP',
            'H2O': 'EIGENVALUES-H2O-cc-pVDZ.FCIDUMP',
        }

    # Generate exact data
    gcembpt = SumOverStates(
            f'../dmqmc_data_pip1/{fcidump_dict[system]}',
            #np.arange(0.0, 25.01, 0.01),
            np.arange(0.0005, 25.0005, 0.0005),
            ftype=np.longdouble,
            only_canonical=False,
        )

    gcembpt.Cv0 = np.gradient(gcembpt.U0)
    gcembpt.Cv0 /= np.gradient(np.divide(1.0, gcembpt.betas))
    gcembpt.Cv1 = np.gradient(gcembpt.U1)
    gcembpt.Cv1 /= np.gradient(np.divide(1.0, gcembpt.betas))

    mask = np.isin(gcembpt.betas.round(5),
                   np.arange(0.01, 25.01, 0.01).round(5))

    gcembpt.betas = gcembpt.betas[mask]
    gcembpt.mu0 = gcembpt.mu0[mask]
    gcembpt.O0 = gcembpt.O0[mask]
    gcembpt.U0 = gcembpt.U0[mask]
    gcembpt.S0 = gcembpt.S0[mask]
    gcembpt.Cv0 = gcembpt.Cv0[mask]
    gcembpt.mu1 = gcembpt.mu1[mask]
    gcembpt.O1 = gcembpt.O1[mask]
    gcembpt.U1 = gcembpt.U1[mask]
    gcembpt.S1 = gcembpt.S1[mask]
    gcembpt.Cv1 = gcembpt.Cv1[mask]

    return gcembpt


def main():

    calcinfo = [
            #'Be',
            #'BeH2',
            #'CO',
            #'HCN',
            #'LiF',
            #'N2',
            #'equilibrium-H4',
            #'equilibrium-H8',
            'stretched-H8',
        ]

    for system in calcinfo:

        print()
        print('-'*80)
        print(f'Running system: {system}')
        print('-'*80)
        print()

        ftmb = get_data(
                system=system,
            )

        df = pd.DataFrame({
                'Beta': ftmb.betas,
                'T': np.divide(1.0, ftmb.betas),
                'U(0)': ftmb.U0,
                'U(1)': ftmb.U1,
                'U(0+1)': ftmb.U0 + ftmb.U1,
                'Cv(0)': ftmb.Cv0,
                'Cv(1)': ftmb.Cv1,
                'Cv(0+1)': ftmb.Cv0 + ftmb.Cv1,
                'S(0)': ftmb.S0,
                'S(1)': ftmb.S1,
                'S(0+1)': ftmb.S0 + ftmb.S1,
                'O(0)': ftmb.O0,
                'O(1)': ftmb.O1,
                'O(0+1)': ftmb.O0 + ftmb.O1,
                'mu0': ftmb.mu0,
                'mu1': ftmb.mu1,
            })

        csv = f'{system}-Many-Body-Fermi-Dirac-Data-GCE.csv'

        print()
        print('\n -- Saving final data frame --')
        print(df.iloc[::100].to_string(float_format='%10.6f'))
        print(df.to_string(float_format='%10.6f'))
        print(f'\n -- Saving to: {csv} --')
        df.round(12).to_csv(csv, index=False)

        print(f'\n -- Done with system: {system} --')
        print()
        print()


if __name__ == '__main__':
    main()
