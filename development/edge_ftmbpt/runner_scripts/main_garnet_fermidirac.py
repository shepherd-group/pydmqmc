#!/usr/bin/env python

import numpy as np
import pandas as pd

from garnet import Analytical


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

    # Generate data
    fd = Analytical(
            fcidump=f'../dmqmc_data_pip1/{fcidump_dict[system]}',
            #betas=np.arange(0.0, 25.01, 0.01),
            #betas=np.arange(1.0, 1.01, 0.01),
            betas=np.arange(0.0005, 25.0005, 0.0005),
            #betas=np.arange(0.005, 25.005, 0.005),
            ftype=np.longdouble,
        )

    fd.Cv0 = np.gradient(fd.U0)/np.gradient(np.divide(1.0, fd.betas))
    fd.Cv1 = np.gradient(fd.U1)/np.gradient(np.divide(1.0, fd.betas))

    finalbetas = np.arange(0.01, 25.01, 0.01)

    mask = np.isin(fd.betas.round(6), finalbetas.round(6))

    fd.betas = fd.betas[mask]
    fd.mu0 = fd.mu0[mask]
    fd.Omega0 = fd.Omega0[mask]
    fd.Omega1 = fd.Omega1[mask]
    fd.U0 = fd.U0[mask]
    fd.U1 = fd.U1[mask]
    fd.S0 = fd.S0[mask]
    #fd.S1 = fd.S1[mask]
    fd.Cv0 = fd.Cv0[mask]
    fd.Cv1 = fd.Cv1[mask]
    #fd.Cv1 = fd.Cv1[mask]

    return fd


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

        fd = get_data(
                system=system,
            )

        df = pd.DataFrame({
                'Beta': fd.betas,
                'T': np.divide(1.0, fd.betas),
                'U(0)': fd.U0,
                'U(1)': fd.U1,
                'U(0+1)': fd.U0 + fd.U1,
                'Cv(0)': fd.Cv0,
                'Cv(1)': fd.Cv1,
                'Cv(0+1)': fd.Cv0 + fd.Cv1,
                #'S': fd.S0 + fd.S1,
                'O(0)': fd.Omega0,
                'O(1)': fd.Omega1,
                'O(0+1)': fd.Omega0 + fd.Omega1,
                'mu0': fd.mu0,
                #'mu1': fd.mu1,
            })

        csv = f'{system}-Garnet-OrbitalBasis-GCE-Data.csv'

        print()
        print('\n -- Saving final data frame --')
        dfstr = df.copy()
        dfstr['dU'] = dfstr['U(0+1)'] - fd.readin.Href
        print(dfstr.iloc[::100].to_string(float_format='%10.6f'))
        #print(df.iloc[::100].to_string(float_format='%10.6f'))
        dfref = pd.read_csv(csv.replace('Garnet-', ''))
        dfref['dU(0+1)'] = dfref['U(0+1)'] - fd.readin.Href
        print(dfref.iloc[::100].to_string(float_format='%10.6f'))
        print(f' RHF: {fd.readin.Href:> 16.12f}')
        print(f' E_{{core}}: {fd.h0:> 16.12f}')
        print(f'\n -- Saving to: {csv} --')
        df.round(12).to_csv(csv, index=False)

        print(f'\n -- Done with system: {system} --')
        print()
        print()

        break


if __name__ == '__main__':
    main()
