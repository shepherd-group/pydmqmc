#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from time import time
from typing import Any, List
from local_utils import (
    getdata,
    State,
    Calculation,
    njittdm,
    njittdm_cont_orbs,
    njittdm_spin_orbs,
)
from matplotlib.backends.backend_pdf import PdfPages

mpl.rc('text', usetex=True)
mpl.rc('savefig', dpi=100)
mpl.rc('hatch', lw=0.25)
mpl.rc('lines', lw=2, markersize=6)
mpl.rc('legend', fontsize=8, numpoints=1)
mpl.rc(('axes', 'xtick', 'ytick'), labelsize=8)
mpl.rc('figure', dpi=200, figsize=(3.37, 3.37*(np.sqrt(5)-1)/2))
mpl.rc(
    'font',
    **{
        'family': 'serif',
        'sans-serif': 'Computer Modern Roman',
        'size': 8,
    }
)


def get_reference(calc: Calculation, hamil: bool = False) -> list[int]:

    psi2 = calc.sym.Cj*calc.sym.Cj

    isort = np.argsort(psi2, kind='stable')

    psi = calc.sym.Cj[isort][::-1]
    det = calc.sym.Xj[isort][::-1]

    if hamil:
        hii = np.diag(calc.sym.ham)[isort][::-1]

    orbs = np.arange(det[0].shape[0]) + 1

    Cmax = abs(psi[0])

    refs = []
    hiis = []

    for j, (Cj, Xj) in enumerate(zip(psi, det)):

        if abs(abs(Cj) - Cmax) > 1E-12:
            break

        occ = orbs[Xj == 1]

        refs.append(occ)

        detstr = 'det = {'+ ', '.join(occ.astype(str)) + '},'

        if hamil:
            hiis.append(hii[j])
            print(f'{Cj:> 22.12f} {detstr} {hii[j]: .12f}')
        else:
            print(f'{Cj:> 22.12f} {detstr}')

    if hamil:
        return refs, hiis
    else:
        return refs


def constructor(mats: dict) -> dict:
    # Not general at all :)
    for key in mats:
        M = np.zeros((8, 8), dtype=float)

        M1, M2 = [np.array(M) for M in mats[key]]

        a1, b1 = M1.shape
        a2, b2 = M2.shape

        if key == 'MUX':
            M[:a1,8-b1:] = M1[:,:]
            M[8-a2:,:b2] = M2[:,:]
        else:
            M[:a1,:b1] = M1[:,:]
            M[8-a2:,8-b2:] = M2[:,:]

        mats[key] = M

    return mats


def matrix_print(label: str, mat: np.array) -> None:
    print(label)
    for i, row in enumerate(mat):
        hstr = '      '
        rstr = f' {i+1:<5}'
        for j, v in enumerate(row):
            hstr += f' {j+1:>8}'
            rstr += f' {v:> 8.4f}'
        if i == 0:
            print(hstr)
        print(rstr)
    return


def main() -> None:

    np.set_printoptions(
        linewidth=360,
        formatter={
            'float': lambda x: f'{0:10}' if x == 0.0 else f'{x: 10.6f}'
        },
    )

    gs = Calculation(
        matrix_file='tdm.mat',
        matrix_constructor=constructor,
        state_object=State(
            0,
            getdata('FCI.out'),
            (0, '01ISYM.DET'),
            16,
            hamil='01ISYM.HAM',
        ),
    )

    print('M_mux')
    print(gs.dxmo)
    print('M_muy')
    print(gs.dymo)
    print('M_muz')
    print(gs.dzmo)

    print('-- Ground state reference --')
    gsref, gshii = get_reference(gs, hamil=True)

    for idet in range(1, 22):
        ex = Calculation(
            matrix_file='tdm.mat',
            matrix_constructor=constructor,
            state_object=State(
                idet,
                getdata('FCI.out'),
                (0, '01ISYM.DET'),
                16,
                hamil='01ISYM.HAM',
            ),
        )

        print('-- Excited reference --')
        exref, exhii = get_reference(ex, hamil=True)

        print(f'    Reference energy: {gs.sym.Ei:> 16.12f}')
        print(f'       Reference RHF: {gshii[0]:> 16.12f}')
        print(f'        Reference Ec: {gs.sym.Ei - gshii[0]:> 16.12f}')
        print(f'      Excited energy: {ex.sym.Ei:> 16.12f}')
        print(f'         Excited RHF: {exhii[0]:> 16.12f}')
        print(f'          Excited Ec: {ex.sym.Ei - exhii[0]:> 16.12f}')
        print(f'          Energy gap: {ex.sym.Ei - gs.sym.Ei:> 16.12f}')

        tdm = np.zeros((8, 8), dtype=float)
        conts = np.zeros((8, 8), dtype=bool)
        tdm, conts = njittdm_cont_orbs(
            gs.sym.Xj,
            gs.sym.Cj,
            ex.sym.Xj,
            ex.sym.Cj,
            tdm,
            conts,
        )

        mux = np.einsum('ij,ij->', tdm, gs.dxmo)
        muy = np.einsum('ij,ij->', tdm, gs.dymo)
        muz = np.einsum('ij,ij->', tdm, gs.dzmo)

        print('Found allowed non-zero transition!')
        print(f'          Excitation: E1(A1) -> E{idet+1}(A1)')
        print(f'    Reference energy: {gs.sym.Ei:> 16.12f}')
        print(f'      Excited energy: {ex.sym.Ei:> 16.12f}')
        print(f'          Energy gap: {ex.sym.Ei - gs.sym.Ei:> 16.12f}')
        print(f'                 mux: {mux:> 16.12f}')
        print(f'                 muy: {muy:> 16.12f}')
        print(f'                 muz: {muz:> 16.12f}')
        print('                 tdm:')
        print(tdm)
        #print(tdm.trace())
        #stdm = np.zeros((16, 16), dtype=float)
        #sconts = np.zeros((16, 16), dtype=float)
        #stdm, sconts = njittdm_spin_orbs(
        #    gs.sym.Xj,
        #    gs.sym.Cj,
        #    ex.sym.Xj,
        #    ex.sym.Cj,
        #    stdm,
        #    sconts,
        #)
        #print(' -- spin version --')
        #print('                 tdm:')
        #print(stdm)
        #matrix_print('', stdm)
        #for i in range(16):
        #    for j in range(16):
        #        if sconts[i,j]:
        #            ius = (i)//2
        #            jus = (j)//2
        #            print(
        #                f'{i+1:3} '
        #                f'{j+1:3} '
        #                f'{stdm[i,j]: .12f} '
        #                f'{ius+1:3} '
        #                f'{jus+1:3} '
        #                f'{gs.dxmo[ius,jus]: .12f} '
        #                f'{gs.dymo[ius,jus]: .12f} '
        #                f'{gs.dzmo[ius,jus]: .12f}'
        #            )

        #break
        #print('                 conts:')
        #print(conts)
        #for i in range(8):
        #    for j in range(8):
        #        if i == j:
        #            continue
        #        elif conts[i,j]:
        #            print(f'{i+1}, {j+1}')
        print()
        print()
        print()

        store = False
        #store = True
        if store:
            with open('exact_tdm.csv', 'wt') as stream:
                print('p,q,gam')
                print('p,q,gam', file=stream)
                for p, row in enumerate(tdm):
                    for q, gam in enumerate(row):
                        if abs(gam) > 1E-12:
                            print(f'{p+1},{q+1},{gam:.12f}')
                            print(f'{p+1},{q+1},{gam:.12f}', file=stream)

        #break

    return


if __name__ == '__main__':
    main()
