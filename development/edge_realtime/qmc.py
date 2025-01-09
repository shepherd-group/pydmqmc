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

from qmc_utils import (
    set_numba_seed,
    do_stochastic_euler,
    do_stochastic_bloch,
    do_stochastic_symplectic,
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

    H = get_coupled_hamiltonian(Ei, outer, state1, state2, driving)

    tf = 80.0
    dt = 0.001
    cutoff = 0.01
    nadd = 0.0

    walkers = 1000.0

    seed = 117

    set_numba_seed(seed)

    #exact = do_euler_integrator(tf, 0.0001, H, state1, state2)
    #exact.to_csv('exact.csv')
    exact = pd.read_csv('exact.csv')
    #print(f'{exact.walkers.max() = }')
    #print(f'{exact.walkers.min() = }')

    euler = do_stochastic_euler(
        tf,
        dt,
        H,
        state1,
        state2,
        walkers,
        cutoff,
        nadd,
    )
    #bloch = do_stochastic_bloch(tf, dt, sys.H, state1, state2, walkers, cutoff, nadd)
    sympl = do_stochastic_symplectic(
        tf,
        dt,
        H,
        state1,
        state2,
        walkers,
        cutoff,
        nadd,
    )

    cmap = mpl.colormaps['plasma'].resampled(7)

    with PdfPages('QMCRabiH4.pdf') as pdf:

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

        ax0.plot(
            exact.tau,
            exact.pop1_real,
            lw=1,
            ls='-',
            color=cmap(0),
            label='State 1 (exact)',
            zorder=2,
        )

        ax0.plot(
            exact.tau,
            exact.pop2_real,
            lw=1,
            ls='-',
            color=cmap(1),
            label='State 2 (exact)',
            zorder=2,
        )

        ax0.plot(
            euler.tau,
            euler.pop1_real,
            lw=1,
            ls='-',
            color=cmap(2),
            label='State 1 (Von Neumann)',
            zorder=2,
        )

        ax0.plot(
            euler.tau,
            euler.pop2_real,
            lw=1,
            ls='-',
            color=cmap(3),
            label='State 2 (Von Neumann)',
            zorder=2,
        )

        ax0.plot(
            sympl.tau,
            sympl.pop1_real,
            lw=1,
            ls='--',
            color=cmap(4),
            label='State 1 (Symplectic)',
            zorder=2,
        )

        ax0.plot(
            sympl.tau,
            sympl.pop2_real,
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


        # Walkers
        plt.close()

        ncols = 3
        nrows = 1

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=False,
            sharey=False,
        )

        ax0, ax1, ax2 = axes

        ax0.plot(
            exact.tau,
            exact.walkers,
            lw=1,
            ls='-',
            color=cmap(0),
            label='State 1 (exact)',
            zorder=2,
        )

        ax0.plot(
            euler.tau,
            euler.walkers,
            lw=1,
            ls='-',
            color=cmap(2),
            label='State 1 (Von Neumann)',
            zorder=2,
        )

        ax0.plot(
            sympl.tau,
            sympl.walkers,
            lw=1,
            ls='--',
            color=cmap(4),
            label='State 1 (Symplectic)',
            zorder=2,
        )

        ax0.set_xlabel(r'$\tau$ / [$\hbar/E_{\mathrm{h}}$]')

        ax0.set_yscale('log')
        ax0.set_ylabel(r'Walkers / [$N_\mathrm{w}$]')

        # Real
        ax1.plot(
            exact.tau,
            exact.pop_real,
            lw=1,
            ls='-',
            color=cmap(0),
            #label='State 1 (exact)',
            zorder=2,
        )

        ax1.plot(
            euler.tau,
            euler.pop_real,
            lw=1,
            ls='-',
            color=cmap(2),
            #label='State 1 (Von Neumann)',
            zorder=2,
        )

        ax1.plot(
            sympl.tau,
            sympl.pop_real,
            lw=1,
            ls='--',
            color=cmap(4),
            #label='State 1 (Symplectic)',
            zorder=2,
        )

        ax1.set_xlabel(r'$\tau$ / [$\hbar/E_{\mathrm{h}}$]')

        ax1.set_yscale('log')
        ax1.set_ylabel(r'$\mathrm{Re}(\mathrm{Walkers})$ / [$N_\mathrm{w}$]')

        # Imag
        ax2.plot(
            exact.tau,
            exact.pop_imag,
            lw=1,
            ls='-',
            color=cmap(0),
            #label='State 1 (exact)',
            zorder=2,
        )

        ax2.plot(
            euler.tau,
            euler.pop_imag,
            lw=1,
            ls='-',
            color=cmap(2),
            #label='State 1 (Von Neumann)',
            zorder=2,
        )

        ax2.plot(
            sympl.tau,
            sympl.pop_imag,
            lw=1,
            ls='--',
            color=cmap(4),
            #label='State 1 (Symplectic)',
            zorder=2,
        )

        ax2.set_xlabel(r'$\tau$ / [$\hbar/E_{\mathrm{h}}$]')

        ax2.set_yscale('log')
        ax2.set_ylabel(r'$\mathrm{Im}(\mathrm{Walkers})$ / [$N_\mathrm{w}$]')

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
