#!/usr/bin/env python

import numpy as np
import pandas as pd
import argparse as ap

from sys import argv
from time import time
from uconv import units
from mpi import ParallelHelper

from local_utils import getmem, read_comm_file
from readin_system import D2hSystem, get_dipole_matrices
from psi_wrappers import (
    get_empty_report,
    update_report,
    field_function_wrapper,
    step_and_align_wrapper,
    report_input_parameters,
)

from dipoles import (
    get_transition_dipole_moments,
    get_determinant_basis_dipole_hamiltonians,
)

from stochastic_utils import (
    compress,
    get_connection_map,
)


def parse_command_line_arguments(arguments: list[str]) -> ap.ArgumentParser:
    ''' Parse command-line arguments.

    Parameters
    ----------
    arguments : list of strings
        User provided command-line arguments.

    Returns
    -------
    options : :class:`argparse.ArgumentParser`
        User options read in from the command-line.
    '''
    parser = ap.ArgumentParser(usage=__doc__)

    # Pulse options
    pulse_args = parser.add_argument_group(
        'Pulse options',
        'Options for the field function',
    )
    pulse_args.add_argument(
        '-ff',
        '--field-function',
        action='store',
        default='transform-limited',
        type=str,
        dest='function',
        help=(
            'Select the field function, one of "continuous", '
            '"transform-limited", or "chirped".'
            ' Default: %(default)s'
        ),
    )
    pulse_args.add_argument(
        '-I',
        '--intensity',
        action='store',
        default=1e12,
        type=float,
        dest='intensity',
        help=(
            'The field intensity in units of W/cm^2.'
            ' Default: %(default)s'
        ),
    )
    pulse_args.add_argument(
        '-tf',
        '--final-time',
        action='store',
        default=50,
        type=float,
        dest='final_time',
        help=(
            'The final time in femto-seconds to simulate to.'
            ' Default: %(default)s'
        ),
    )
    pulse_args.add_argument(
        '-fwhm',
        '--fwhm',
        action='store',
        default=10,
        type=float,
        dest='fwhm',
        help=(
            'The Gaussian\'s full width at half maximum, which is used to '
            'calculate the sigma (standard deviation) given as: '
            'sigma = fwhm / (2 * sqrt(2 * ln(2))). (units of femto-seconds)'
            ' Default: %(default)s'
        ),
    )
    pulse_args.add_argument(
        '-c',
        '--center',
        action='store',
        default=12.5,
        type=float,
        dest='center',
        help=(
            'The Gaussian center for the field in femto-seconds.'
            ' Default: %(default)s'
        ),
    )
    pulse_args.add_argument(
        '-w',
        '--omega',
        action='store',
        default=None,
        type=float,
        dest='omega',
        help=(
            'The field frequency in Hartree energy units. '
            'If no value is set the first bright transition energy gap '
            'will be used.'
            ' Default: %(default)s'
        ),
    )
    pulse_args.add_argument(
        '-b',
        '--beta',
        action='store',
        default=0.256,
        type=float,
        dest='beta',
        help=(
            'The chirp parameter in unts of inverse femto-seconds^2.'
            ' Default: %(default)s'
        ),
    )
    pulse_args.add_argument(
        '-dt',
        '--time-step',
        action='store',
        default=1.0,
        type=float,
        dest='time_step',
        help=(
            'The simulation time step in atto-seconds.'
            ' Default: %(default)s'
        ),
    )

    # Iteration parameters
    iter_args = parser.add_argument_group(
        'Iteration options',
        'Options for performing the iteration',
    )
    iter_args.add_argument(
        '-n',
        '--nreports',
        action='store',
        default=2,
        type=int,
        dest='nreports',
        help=(
            'The number of iterations between reports.'
            ' Default: %(default)s'
        ),
    )
    iter_args.add_argument(
        '-m',
        '--method',
        action='store',
        default='euler',
        type=str,
        dest='method',
        help=(
            'Select the iteration method, one of "euler", '
            '"symplectic", or "runge-kutta". For the Runge--Kutta method '
            'see the order and heun parameter(s).'
            ' Default: %(default)s'
        ),
    )
    iter_args.add_argument(
        '-rkn',
        '--runge-kutta-order',
        action='store',
        default=4,
        type=int,
        dest='order',
        help=(
            'Can be one of 1 (same as Euler), 2, or 4. For 2nd order, see '
            'the heun parameter.'
            ' Default: %(default)s'
        ),
    )
    iter_args.add_argument(
        '-heun',
        '--heun',
        action='store_true',
        default=False,
        dest='heun',
        help=(
            'Use Heun\'s tableau. Used for the 2nd order Runge--Kutta '
            'method only.  Otherwise, use the standard midpoint tableau.'
            ' Default: %(default)s'
        ),
    )
    iter_args.add_argument(
        '-com',
        '--comm-file',
        action='store',
        default='RUN.COMM',
        type=str,
        dest='comm_file',
        help=(
            'A filename used to interact with the running calculation. '
            'For more information see `read_comm_file` in local_utils.'
            ' Default: %(default)s'
        ),
    )

    # QMC parameters
    qmc_args = parser.add_argument_group(
        'QMC options',
        'Options for performing quantum Monte Carlo interation',
    )
    qmc_args.add_argument(
        '-s',
        '--stochastic',
        action='store_true',
        default=False,
        dest='stochastic',
        help=(
            'Perform a stochastic integration.'
            ' Default: %(default)s'
        ),
    )
    qmc_args.add_argument(
        '-c0',
        '--c0-populationr',
        action='store',
        default=10000,
        type=float,
        dest='c0_population',
        help=(
            'The population for the reference determinant for the ground '
            'state (t=0) wavefunction, used to rescale the wavefunction '
            'and set the walker population.'
            ' Default: %(default)s'
        ),
    )
    qmc_args.add_argument(
        '-rng',
        '--rng-seed',
        action='store',
        default=117,
        type=int,
        dest='rng_seed',
        help=(
            'The seed for the random number generator.'
            ' Default: %(default)s'
        ),
    )
    qmc_args.add_argument(
        '-co',
        '--cutoff',
        action='store',
        default=0.01,
        type=float,
        dest='cutoff',
        help=(
            'The cutoff used for spawning.'
            ' Default: %(default)s'
        ),
    )
    qmc_args.add_argument(
        '-na',
        '--nareps',
        action='store',
        default=1,
        type=int,
        dest='nareps',
        help=(
            'The number of spawning attempts per walker.'
            ' Default: %(default)s'
        ),
    )
    qmc_args.add_argument(
        '-nc',
        '--ncalcs',
        action='store',
        default=1,
        type=int,
        dest='ncalcs',
        help=(
            'The number of calculations to perform.'
            ' Default: %(default)s'
        ),
    )
    qmc_args.add_argument(
        '-ni',
        '--initiator-threshold',
        action='store',
        default=0.0,
        type=float,
        dest='ninit',
        help=(
            'The initiator population threshold.'
            ' Default: %(default)s'
        ),
    )

    # Storage parameters
    store_args = parser.add_argument_group(
        'Storage options',
        'Options for storing data resulting from the iterations',
    )
    store_args.add_argument(
        '-csv',
        '--csv',
        action='store',
        default='',
        type=str,
        dest='csv',
        help=(
            'A filename to store the calculation data as each calculation '
            'is completed. The format is csv, and the csv suffix need '
            'not be included.'
            ' Default: %(default)s'
        ),
    )

    # Argeparse specific things.
    parser.parse_args(args=None if arguments else ['--help'])

    options = parser.parse_args(arguments)

    if options.csv != '':
        if '.csv' not in options.csv:
            options.csv = f'{options.csv}.csv'

    return options


def get_system(
        ph: ParallelHelper,
        nwfns: int = 202,
        syspath: str = './system',
    ) -> tuple[D2hSystem, np.array, np.array, np.array, np.array]:
    r''' TODO: Write docstring.
    '''
    irreps = [1, 4, 6, 7, 8, 5, 3, 2]

    # TODO: Still needs some attention, bcast on large files causes
    #       OOM errors.
    if ph.parent:
        sys = D2hSystem(
            dump=f'{syspath}/FCIDUMP',
            output=f'{syspath}/FCI.out',
            psi4_output=f'{syspath}/decacene_ints.out',
            psi4_rhf_output=f'{syspath}/decacene_rhf.out',
            verbose=ph.parent,
            cas_orbitals=[
                '4B2g',
                '6B3u',
                '4Au',
                '5B2g',
                '5Au',
                '6B1g',
                '7B3u',
                '7B1g',
                '6B2g',
                '8B3u',
            ],
            hande_fci_files={
                1: {
                    'det': f'{syspath}/01ISYM.DET',
                    'ham': f'{syspath}/01ISYM.HAM',
                    #'diag': True,
                    'wfn': f'{syspath}/01ISYM.WFN',
                    'nwfn': 1,
                },
                5: {
                    'det': f'{syspath}/05ISYM.DET',
                    'ham': f'{syspath}/05ISYM.HAM',
                    #'diag': True,
                    'wfn': f'{syspath}/05ISYM.WFN',
                    'nwfn': nwfns,
                },
            },
            dipole={
                'x': {
                    i: f'{syspath}/{n+1:03d}_irrep{i:03d}_so_mux.npy'
                    for n, i in enumerate(irreps)
                },
                'y': {
                    i: f'{syspath}/{n+1:03d}_irrep{i:03d}_so_muy.npy'
                    for n, i in enumerate(irreps)
                },
                'z': {
                    i: f'{syspath}/{n+1:03d}_irrep{i:03d}_so_muz.npy'
                    for n, i in enumerate(irreps)
                },
            },
            overlap={
                i: f'{syspath}/{n+1:03d}_irrep{i:03d}_so_overlap.npy'
                for n, i in enumerate(irreps)
            },
            coefficients={
                i: f'{syspath}/{n+1:03d}_irrep{i:03d}_so_coefficients.npy'
                for n, i in enumerate(irreps)
            },
        )
    else:
        sys = D2hSystem(
            dump=f'{syspath}/FCIDUMP',
            output=f'{syspath}/FCI.out',
            psi4_output=f'{syspath}/decacene_ints.out',
            psi4_rhf_output=f'{syspath}/decacene_rhf.out',
            verbose=ph.parent,
            cas_orbitals=[
                '4B2g',
                '6B3u',
                '4Au',
                '5B2g',
                '5Au',
                '6B1g',
                '7B3u',
                '7B1g',
                '6B2g',
                '8B3u',
            ],
            hande_fci_files={
                1: {},
                5: {},
            },
            dipole={},
            overlap={},
            coefficients={},
        )

    if ph.parent:
        ph.print(sys)

        _, _, muzmat = get_dipole_matrices(sys)

        ti = time()
        moms = get_transition_dipole_moments(
            sys.M//2,
            sys.det[1],
            sys.det[5],
            sys.wfn[1][0],
            sys.wfn[5],
            muzmat,
        )
        tf = time()
        dt = tf - ti
        ph.print(
            f'Time to calculate excitation moments: {dt/60.0:.6f} (min.)'
        )

        inonzero = np.argwhere(abs(moms) > 1E-6).flatten()[:5] + 1

        ti = time()
        h0, ud = get_determinant_basis_dipole_hamiltonians(
            sys.ham[1][0, 0],
            sys.det[1],
            sys.det[5],
            sys.ham[1],
            sys.ham[5],
            muzmat,
        )
        tf = time()
        dt = tf - ti
        ph.print(
            f'Time to build dipole Hamiltonians: {dt/60.0:.6f} (min.)\n'
        )

    shape = np.array([0, 0, 0], dtype=int)

    if ph.parent:
        shape = np.array(
            [h0.shape[0], h0.shape[1], inonzero.shape[0]],
            dtype=int,
        )

    ph.bcast(shape)

    if not ph.parent:
        assert shape[0] != 0
        assert shape[0] == shape[1]
        assert shape[2] != 0

        inonzero = np.zeros(shape[2], dtype=int)

        h0 = np.zeros((shape[0], shape[1]), dtype=float)
        ud = np.zeros((shape[0], shape[1]), dtype=float)

    ph.bcast(inonzero)
    ph.bcast(h0)
    ph.bcast(ud)

    return sys, inonzero, h0, ud


def run_realtime(
        ph: ParallelHelper,
        sys: D2hSystem,
        inonzero: np.array,
        h0: np.array,
        ud: np.array,
        function: str,
        I: float,
        omega: float,
        sigma: float,
        center: float,
        beta: float,
        tf: float,
        dt: float,
        nits: int,
        nrep: int,
        method: str,
        order: int,
        heun: bool,
        comm_file: str,
        stochastic: bool,
        c0_population: float,
        rng_seed: int,
        cutoff: float,
        nareps: int,
        ncalc: int,
        ninit: float,
        csv: str,
    ) -> None:
    r''' TODO: Write docstring here.
    '''
    report_input_parameters(
        ph.parent,
        function,
        I,
        omega,
        sigma,
        center,
        beta,
        tf,
        dt,
        nits,
        nrep,
        method,
        order,
        heun,
        comm_file,
        stochastic,
        c0_population,
        rng_seed,
        cutoff,
        nareps,
        ninit,
        ncalc,
        csv,
    )

    # Set variables and allocate arrays for MPI.
    ph.setup_job_map(h0.shape[0], ninit)

    ph.print(f'Allocating q and p array(s) with size: {(h0.shape[0], 1)}')

    q = np.zeros((h0.shape[0], 1), dtype=float)
    p = np.zeros((h0.shape[0], 1), dtype=float)

    # TODO: Move this report somewhere else, and add the missing arrays
    #       that are persistent like ncons and icons etc...
    if ph.parent:
        print(
            ' === Approximate memory usage (per processor) ===\n'
            f'                h0: {getmem(h0):.3f} (GB)\n'
            f'                ud: {getmem(ud):.3f} (GB)\n'
            f'                 p: {getmem(p):.3f} (GB)\n'
            f'                 q: {getmem(q):.3f} (GB)\n'
            f'               bra: {getmem(q + 1j*p):.3f} (GB)\n'
            f'               ket: {getmem(np.conjugate(q + 1j*p)):.3f} (GB)\n'
            f'           recvbuf: {getmem(ph.recvbuf):.3f} (GB)\n'
            f'        pq_sendbuf: {getmem(ph.pq_sendbuf):.3f} (GB)\n'
            f'        pq_recvbuf: {getmem(ph.pq_recvbuf):.3f} (GB)\n'
            f'      dpdq_sendbuf: {getmem(ph.dpdq_sendbuf):.3f} (GB)\n'
            f'      dpdq_recvbuf: {getmem(ph.dpdq_recvbuf):.3f} (GB)\n'
            ' **NB: There may be as many as ~2-24 additional intermediates\n'
            '       allocated during iteration with size like p (or q).**\n'
            ' WARNING: It is left to the user to ensure there is enough\n'
            '          memory for the calculation!'
        )

    # Get the valid excitations so we don't recompute at every step.
    ncons, icons = get_connection_map(np.abs(h0) + np.abs(ud))

    # Start looping.
    reports = []
    timing_report = {}
    softexit = False

    for icalc in range(ncalc):
        if softexit:
            break

        t = 0.0
        report = get_empty_report(inonzero)

        q[:, 0] = 0.0
        p[:, 0] = 0.0

        if ph.parent:
            q[:sys.wfn[1][0].shape[0], 0] = sys.wfn[1][0]

        # Get our random number generator
        seed = ph.get_rng_seed(rng_seed, icalc)
        rng = np.random.default_rng(seed)

        if stochastic and ph.parent:
            q /= abs(q).max()/c0_population
            q = compress(q, rng)

        if stochastic:
            # Send the initial q from parent to all children.
            ph.bcast(q)

        report = update_report(
            ph,
            report,
            0,
            0.0,
            t,
            0.0,
            p,
            q,
            sys,
            inonzero,
        )

        itime = time()
        ttime = itime

        for i in range(nits):
            comm_cmds = read_comm_file(comm_file, ph)

            if 'softexit' in comm_cmds:
                softexit = comm_cmds['softexit']

            if softexit:
                break

            Et, Et_hdt, Et_dt = field_function_wrapper(
                function,
                I,
                omega,
                sigma,
                center,
                beta,
                t,
                dt,
            )

            p, q, prep = step_and_align_wrapper(
                i,
                dt,
                Et,
                Et_hdt,
                Et_dt,
                h0,
                ud,
                p,
                q,
                cutoff,
                nareps,
                ncons,
                icons,
                rng,
                nrep,
                ninit,
                stochastic,
                method,
                order,
                heun,
                ph,
            )

            t += dt

            if (i + 1) % nrep == 0:
                wt = time() - itime
                itime = time()

                report = update_report(
                    ph,
                    report,
                    i + 1,
                    wt,
                    t,
                    Et,
                    prep,
                    q,
                    sys,
                    inonzero,
                )

        if ph.parent:
            ttime = time() - ttime
            timing_report[icalc] = ttime/60.0
            ph.print()
            ph.print(f'Total run time: {ttime/60.0:> 8.4f} (min.)')
            ph.print(f'Total runs finished: {icalc + 1:>8d} / {ncalc:<8d}')

            report = pd.DataFrame(report)
            reports.append(report)

            if csv != '':
                ph.print(f'Generating partial dataset in: {csv}')
                tmp = pd.concat(reports)
                tmp.to_csv(csv, index=False)
                del tmp

    if ph.parent:
        if csv != '':
            ph.print(f'Saving final dataset in: {csv}')
            reports = pd.concat(reports)
            reports.to_csv(csv, index=False)

        ph.print('Final timing report per calculation:')
        ph.print(f'{"calculation":>12} {"time":>12}')
        ttime = 0.0
        for icalc, rt in timing_report.items():
            ttime += rt
            ph.print(f'    {icalc+1:>8d} {rt:> 12.4f} (min.)')

        ph.print()
        ph.print(f'Total time: {ttime:> 12.4f} (min.)')

    return


def main(clargs: list[str]) -> None:

    ph = ParallelHelper()

    args = parse_command_line_arguments(clargs)

    # TODO: Set system with arguments.
    sys, inonzero, h0, ud = get_system(ph)

    # Pulse parameters
    # continuous, transform-limited, chirped
    function = args.function
    # W/cm^2 -> Eh^{2}/(\hbar a_{0}^2)
    I = (args.intensity) * units.cm * units.cm * (1/units.W)
    # fs -> \hbar / Eh
    tf = (args.final_time) * (1/units.fs)
    sigma = (args.fwhm / (2 * (2 * np.log(2))**0.5)) * (1/units.fs) 
    # Seemed like ~12.5 fs from the Fig. 5 plot.
    center = (args.center) * (1/units.fs)

    if args.omega is None:
        # (S0 -> S2) Eh -> 1/(\hbar / Eh)
        omega = (sys.fci[5][inonzero[0] - 1] - sys.fci[1][0])
    else:
        omega = args.omega

    # \beta -> 1/(\hbar / Eh)^2
    beta = (args.beta) * (units.fs) * (units.fs)
    # as -> \hbar / Eh
    dt = (args.time_step) * (1/units.ats)

    # Iteration parameters
    nits = int(tf / dt)
    nrep = args.nreports
    # euler, runge-kutta, or symplectic
    method = args.method
    # For runge-kutta only.
    order = args.order
    heun = args.heun
    comm_file = args.comm_file

    # QMC parameters
    stochastic = args.stochastic
    c0_population = args.c0_population
    rng_seed = args.rng_seed
    cutoff = args.cutoff
    nareps = args.nareps
    ncalc = args.ncalcs
    ninit = args.ninit

    # Storage parameters
    csv = args.csv

    if ph.parallel and not stochastic:
        raise RuntimeError(
            'Currently only stochastic is supported with MPI, send patches!'
        )

    run_realtime(
        ph,
        sys,
        inonzero,
        h0,
        ud,
        function,
        I,
        omega,
        sigma,
        center,
        beta,
        tf,
        dt,
        nits,
        nrep,
        method,
        order,
        heun,
        comm_file,
        stochastic,
        c0_population,
        rng_seed,
        cutoff,
        nareps,
        ncalc,
        ninit,
        csv,
    )

    return


if __name__ == '__main__':
    main(argv[1:])
