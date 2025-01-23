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

    # Generate size of space estimate
    thf = SumOverStates(
            beta=np.arange(1.0, 2.0, 1.0),
            fcidump=f'../dmqmc_data_pip1/{fcidump_dict[system]}',
            dtype=np.longdouble,
            grandcanonical=False,
            #grandcanonical=True,
            #hilbertspaceonly=False,
            hilbertspaceonly=True,
            #allspin=False,
            allspin=True,
            #allsector=False,
            allsector=True,
            eigenvalues=False,
        )

    return thf


def main():

    calcinfo = [
            'Be',
            'BeH2',
            'CO',
            'HCN',
            'LiF',
            'N2',
            'equilibrium-H4',
            'equilibrium-H8',
            'stretched-H8',
        ]

    for system in calcinfo:

        print()
        print('-'*80)
        print(f'Running system: {system}')
        print('-'*80)

        thf = get_data(
                system=system,
            )

        continue


if __name__ == '__main__':
    main()
