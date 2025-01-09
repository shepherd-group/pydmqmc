#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from local_utils import (
    readin,
    Outer,
    get_coupled_hamiltonian,
    do_euler_integrator,
    do_symplectic_integrator,
)
from typing import Any
from numpy.typing import NDArray as Array
from matplotlib.backends.backend_pdf import PdfPages

mpl.rc('text', usetex=True)
mpl.rc('savefig', dpi=100)
mpl.rc('lines', lw=2, markersize=6)
mpl.rc('legend', fontsize=8, numpoints=1)
mpl.rc(('axes', 'xtick', 'ytick'), labelsize=8)
mpl.rc('figure', dpi=200, figsize=(3.37, 3.37*(np.sqrt(5)-1)/2))
mpl.rc('font', **{'family': 'serif', 'sans-serif': 'Computer Modern Roman'})


def main() -> None:

    sys = readin(
        #int_file='h2.fcidump',
        int_file='h4.fcidump',
        verbose=True,
        hamiltonian=True,
    )

    print()

    Ei, Psii = np.linalg.eigh(sys.H)
    Ei -= Ei.min()

    outer = Outer(Psii)

    # https://doi.org/10.1063/1.5115323
    driving = 1.0

    state1 = 0
    state2 = 5
    #state1 = 5
    #state2 = 0

    H = get_coupled_hamiltonian(Ei, outer, state1, state2, driving)

    tf = 40.0
    dt = 0.001

    exact = do_euler_integrator(tf, 0.000001, H, state1, state2)

    ei = do_euler_integrator(tf, dt, H, state1, state2)

    si = do_symplectic_integrator(tf, dt, H, state1, state2)

    print(exact)
    print(ei)
    print(si)
    return
    exact.to_csv('exact.csv', index=False)
    ei.to_csv('euler.csv', index=False)
    si.to_csv('symplectic.csv', index=False)

    cmap = mpl.colormaps['plasma'].resampled(7)

    with PdfPages('RabiH4.pdf') as pdf:

        # Beta
        plt.close()

        ncols = 1
        nrows = 1

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=False,
            sharey=False,
        )

        ax0 = axes

        #ax0.plot(
        #    exact.tau,
        #    exact.pop1_real,
        #    lw=1,
        #    ls='-',
        #    color=cmap(0),
        #    label='State 1 (exact)',
        #    zorder=2,
        #)

        #ax0.plot(
        #    exact.tau,
        #    exact.pop2_real,
        #    lw=1,
        #    ls='-',
        #    color=cmap(1),
        #    label='State 2 (exact)',
        #    zorder=2,
        #)

        ax0.plot(
            ei.tau,
            ei.pop1_real,
            lw=1,
            ls='-',
            color=cmap(2),
            label='State 1 (Von Neumann)',
            zorder=2,
        )

        ax0.plot(
            ei.tau,
            ei.pop2_real,
            lw=1,
            ls='-',
            color=cmap(3),
            label='State 2 (Von Neumann)',
            zorder=2,
        )

        ax0.plot(
            si.tau,
            si.pop1_real,
            lw=1,
            ls='--',
            color=cmap(4),
            label='State 1 (Symplectic)',
            zorder=2,
        )

        ax0.plot(
            si.tau,
            si.pop2_real,
            lw=1,
            ls='--',
            color=cmap(5),
            label='State 2 (Symplectic)',
            zorder=2,
        )

        ax0.axhline(
            y=0,
            lw=0.6,
            color='k',
            zorder=1,
        )

        ax0.set_xlabel(r'$\tau$ / [$\hbar/E_{\mathrm{h}}$]')

        ax0.set_ylabel(r'Population / [$N_\mathrm{w}$]')

        fig.subplots_adjust(wspace=0.5, hspace=0.6)
        fig.set_size_inches(ncols*3.37, nrows*3.37*(5**0.5-1)/2)

        fig.legend(
            ncol=2,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.07),
        )

        pdf.savefig(bbox_inches='tight')

    return


if __name__ == '__main__':
    main()
