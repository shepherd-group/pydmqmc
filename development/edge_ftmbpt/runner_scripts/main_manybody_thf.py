#!/usr/bin/env python

import numpy as np
import pandas as pd

from manybody_thf import SumOverStates


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
    thf = SumOverStates(
            #beta=np.arange(0.01, 25.01, 0.01),
            beta=np.arange(0.0005, 25.0005, 0.0005),
            fcidump=f'../dmqmc_data_pip1/{fcidump_dict[system]}',
            dtype=np.longdouble,
            grandcanonical=False,
            #grandcanonical=True,
            hilbertspaceonly=False,
            #hilbertspaceonly=True,
            #allspin=False,
            allspin=True,
            #allsector=False,
            allsector=True,
            eigenvalues=False,
        )

    if thf.hilbertspaceonly:
        return thf

    thf.Cv0 = np.gradient(thf.U0)/np.gradient(np.divide(1.0, thf.beta))

    finalbetas = np.arange(0.01, 25.01, 0.01)

    mask = np.isin(thf.beta.round(6), finalbetas.round(6))

    thf.beta = thf.beta[mask]
    thf.mu0 = thf.mu0[mask]
    thf.O0 = thf.O0[mask]
    thf.U0 = thf.U0[mask]
    thf.S0 = thf.S0[mask]
    thf.Cv0 = thf.Cv0[mask]

    return thf


def main():

    #runbenchmark()
    #return

    calcinfo = [
            'Be',
            'BeH2',
            'CO',
            'HCN',
            'LiF',
            'N2',
            'equilibrium-H4',
            'equilibrium-H8',
            #'stretched-H8',
        ]

    for system in calcinfo:

        print()
        print('-'*80)
        print(f'Running system: {system}')
        print('-'*80)
        print()

        thf = get_data(
                system=system,
            )

        if thf.hilbertspaceonly:
            continue
        else:
            df = pd.DataFrame({
                    'Beta': thf.beta,
                    'T': np.divide(1.0, thf.beta),
                    'E': thf.U0,
                    'Cv': thf.Cv0,
                    'S': thf.S0,
                    'O': thf.O0,
                    'mu0': thf.mu0,
                })

            #csv = f'{system}-ManyBody-GCE-THF-Data.csv'
            csv = f'{system}-ManyBody-CE-THF-AllSpin-AllSym-Data.csv'

            print()
            print('\n -- Saving final data frame --')
            #print(df.iloc[::100].to_string(float_format='%10.6f'))
            print(df.iloc[::100].to_string(float_format='%10.6f'))
            print(f'\n -- Saving to: {csv} --')
            df.round(12).to_csv(csv, index=False)

            print(f'\n -- Done with system: {system} --')
            print()
            print()


if __name__ == '__main__':
    main()
