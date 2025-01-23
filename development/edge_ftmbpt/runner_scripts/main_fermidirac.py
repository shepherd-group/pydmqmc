#!/usr/bin/env python

import numpy as np
import pandas as pd

#from manybody_fermidirac import SumOverStates
from manybody_fermidirac02 import SumOverStates
from orbitalbasis_fermidirac import OrbitalBasis


def runbenchmark():
    path = '../fermi-dirac-testing'
    H00 = -98.57075759

    so = pd.read_csv(f'{path}/sohirata_gce_fh_mpN.csv')
    so = so.drop(columns='T')
    betas = so['Beta'].values.flatten()

    # Do Orbital basis version of FD
    ftfd = OrbitalBasis(
            f'{path}/FH_nosym/nosym_fh_sto3g_trimmed.fcidump',
            betas,
            ftype=np.longdouble,
            eigenvalues=True,
        )

    print(so.round(4))
    print()

    so['mu(0-WZV-OB)'] = ftfd.mu0
    so['Omega(0-WZV-OB)'] = ftfd.omega0
    so['U(0-WZV-OB)'] = ftfd.U0
    so['S(0-WZV-OB)'] = ftfd.S0
    so['mu(1-WZV-OB)'] = ftfd.mu1
    so['Omega(1-WZV-OB)'] = ftfd.omega1
    so['U(1-WZV-OB)'] = ftfd.U1
    so['S(1-WZV-OB)'] = ftfd.S1

    print(so.round(5))
    print()

    for k in ['Omega', 'U', 'mu', 'S']:
        so[f'{k}(0+1)'] = so[f'{k}(0)'] + so[f'{k}(1)']
        so[f'{k}(0+1-WZV-OB)'] = so[f'{k}(0-WZV-OB)'] + so[f'{k}(1-WZV-OB)']
        so[f'd{k}(WZV-So)'] = so[f'{k}(0+1-WZV-OB)'] - so[f'{k}(0+1)']
        if k == 'U':
            so[f'd{k}(So-HF)'] = so[f'{k}(0+1)'] - H00
        else:
            so[f'd{k}(So-HF)'] = np.nan

        print(so.loc[:, ['Beta', f'{k}(0+1)', f'{k}(0+1-WZV-OB)',
                         f'd{k}(WZV-So)', f'd{k}(So-HF)']].round(4))
        if k == 'U':
            print(f'HF: {H00:> 16.12f}')
        print()

    return

    # Do Many body version of FD
    so = pd.read_csv(f'{path}/sohirata_gce_fh_mpN.csv')
    so = so.drop(columns='T')
    betas = so['Beta'].values.flatten()

    ftmb = SumOverStates(
            f'{path}/FH_nosym/nosym_fh_sto3g_trimmed.fcidump',
            betas,
            ftype=np.longdouble,
            eigenvalues=True,
        )

    print(so.round(4))
    print()

    so['Omega(0-WZV-MB)'] = ftmb.O0
    so['U(0-WZV-MB)'] = ftmb.U0
    so['mu(0-WZV-MB)'] = ftmb.mu0
    so['S(0-WZV-MB)'] = ftmb.S0
    so['Omega(1-WZV-MB)'] = ftmb.O1
    so['U(1-WZV-MB)'] = ftmb.U1
    so['mu(1-WZV-MB)'] = ftmb.mu1
    so['S(1-WZV-MB)'] = ftmb.S1

    print(so.round(4))
    print()

    for k in ['Omega', 'U', 'mu', 'S']:
        so[f'{k}(0+1)'] = so[f'{k}(0)'] + so[f'{k}(1)']
        so[f'{k}(0+1-WZV-MB)'] = so[f'{k}(0-WZV-MB)'] + so[f'{k}(1-WZV-MB)']
        so[f'd{k}(WZV-So)'] = so[f'{k}(0+1-WZV-MB)'] - so[f'{k}(0+1)']
        if k == 'U':
            so[f'd{k}(So-HF)'] = so[f'{k}(0+1)'] - H00
        else:
            so[f'd{k}(So-HF)'] = np.nan

        print(so.loc[:, ['Beta', f'{k}(0+1)', f'{k}(0+1-WZV-MB)',
                         f'd{k}(WZV-So)', f'd{k}(So-HF)']].round(4))
        if k == 'U':
            print(f'HF: {H00:> 16.12f}')
        print()


    # Do the canonical ensemble many body basis
    so = pd.read_csv(f'{path}/sohirata_ce_fh_mpN_more_data.csv')
    so = so.drop(columns='T')
    betas = so['Beta'].values.flatten()

    ftmb = SumOverStates(
            f'{path}/FH_nosym/nosym_fh_sto3g_trimmed.fcidump',
            betas,
            ftype=np.longdouble,
            only_canonical=True,
            all_spin=True,
            eigenvalues=True,
        )

    print(so.round(4))
    print()

    so['U(0-WZV-MB)'] = ftmb.U0
    so['U(1-WZV-MB)'] = ftmb.U1
    so['S(0-WZV-MB)'] = ftmb.S0
    so['S(1-WZV-MB)'] = ftmb.S1
    so['F(0-WZV-MB)'] = ftmb.O0
    so['F(1-WZV-MB)'] = ftmb.O1

    print(so.round(4))
    print()

    for k in ['U', 'S', 'F']:
        so[f'{k}(0+1)'] = so[f'{k}(0)'] + so[f'{k}(1)']
        so[f'{k}(0+1-WZV-MB)'] = so[f'{k}(0-WZV-MB)'] + so[f'{k}(1-WZV-MB)']
        so[f'd{k}(WZV-So)'] = so[f'{k}(0+1-WZV-MB)'] - so[f'{k}(0+1)']
        if k == 'U':
            so[f'd{k}(So-HF)'] = so[f'{k}(0+1)'] - H00
        else:
            so[f'd{k}(So-HF)'] = np.nan

        print(so.loc[:, ['Beta', f'{k}(0+1)', f'{k}(0+1-WZV-MB)',
                         f'd{k}(WZV-So)', f'd{k}(So-HF)']].round(4))
        if k == 'U':
            print(f'HF: {H00:> 16.12f}')
        print()

    return


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
    ftmb = SumOverStates(
            f'../dmqmc_data_pip1/{fcidump_dict[system]}',
            np.arange(0.0, 25.01, 0.01),
            ftype=np.longdouble,
            only_canonical=True,
        )

    return ftmb


def main():

    runbenchmark()
    return

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
        print()

        ftmb = get_data(
                system=system,
            )

        df = pd.DataFrame({
                'Beta': ftmb.betas,
                'T': np.divide(1.0, ftmb.betas),
                'E': ftmb.U0 + ftmb.U1,
                'Cv': ftmb.Cv0 + ftmb.Cv1,
                'S': ftmb.S0 + ftmb.S1,
                'O': ftmb.O0 + ftmb.O1,
                'mu0': ftmb.mu0,
                'mu1': ftmb.mu1,
            })

        csv = f'{system}-ManyBody-CE-Data.csv'

        print()
        print('\n -- Saving final data frame --')
        print(df.iloc[::100].to_string(float_format='%10.6f'))
        print(f'\n -- Saving to: {csv} --')
        df.round(12).to_csv(csv, index=False)

        print(f'\n -- Done with system: {system} --')
        print()
        print()


if __name__ == '__main__':
    main()
