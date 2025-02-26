#!/usr/bin/env python

import numpy as np
import stochastic_integrators as stoch

from mpi import ParallelHelper
from readin_system import D2hSystem
from field_functions import cw_t, tl_t, c_t

from integrators import (
    euler_step,
    symplectic_step,
    symplectic_align,
    runge_kutta_step,
    get_state_populations,
)


def get_empty_report(inonzero: np.array) -> dict:
    r''' TODO: Write docstring.
    '''
    report = {
        'iteration': [],
        'time': [],
        'Et': [],
        'Psi0': [],
    }

    for idet in inonzero:
        report[f'Psi{idet}'] = []

    report['Psin'] = []
    report['Psik'] = []
    report['real'] = []
    report['imag'] = []
    report['norm'] = []
    report['dets'] = []
    report['wall'] = []

    return report


def update_report(
            ph: ParallelHelper,
            report: dict,
            iteration: int,
            wall: float,
            t: float,
            Et: float,
            p: np.array,
            q: np.array,
            sys: D2hSystem,
            inonzero: np.array,
            stdout: bool = True,
        ) -> dict:
    ''' TODO: Write a docstring.
    '''
    if not ph.parent:
        return report

    if stdout and len(report['time']) == 0:
        l = ' '

        for col, vals in report.items():
            if col in ['iteration', 'time', 'dets', 'wall']:
                fmt = '12'
            else:
                fmt = '22'

            l += f'{col:>{fmt}} '

        print(l)

    report['iteration'].append(iteration)
    report['time'].append(t)
    report['Et'].append(Et)

    norm = (q**2 + p**2).sum()

    pops = get_state_populations(q, p, sys.wfn[1][0], sys.wfn[5])
    pops /= norm

    report['Psi0'].append(pops[0])

    snsum = pops.sum()
    snsum -= report['Psi0'][-1]

    for i in inonzero:
        report[f'Psi{i}'].append(pops[i])
        snsum -= report[f'Psi{i}'][-1]

    report['Psin'].append(snsum)
    report['real'].append(abs(q).sum())
    report['imag'].append(abs(p).sum())
    report['norm'].append(norm)
    report['Psik'].append(report['norm'][-1]/norm - pops.sum())
    report['dets'].append(np.count_nonzero(p + q))
    report['wall'].append(wall)

    if stdout:
        l = ' '

        for col, vals in report.items():
            if col in ['iteration', 'dets']:
                fmt = '12d'
            elif col in ['real', 'imag', 'norm']:
                fmt = '22.12E'
            elif col in ['time', 'wall']:
                fmt = '12.6f'
            else:
                fmt = ' 22.12f'

            l += f'{vals[-1]:>{fmt}} '

        print(l)

    return report


def step_and_align_wrapper(
        it: int,
        dt: float,
        Et: float,
        Et_hdt: float,
        Et_dt: float,
        h0: np.array,
        ud: np.array,
        p: np.array,
        q: np.array,
        cutoff: float,
        nareps: int,
        ncons: np.array,
        icons: np.array,
        rng: np.random.Generator,
        nrep: int,
        ninit: float,
        stochastic: bool,
        method: str,
        order: int,
        heun: bool,
        ph: ParallelHelper,
    ) -> tuple[np.array, np.array, np.array]:
    r''' TODO: Write docstring.
    '''
    if stochastic:
        match method:
            case 'euler':
                p, q = stoch.euler_step(
                    dt,
                    Et,
                    h0,
                    ud,
                    p,
                    q,
                    cutoff,
                    nareps,
                    ninit,
                    ncons,
                    icons,
                    rng,
                    ph,
                )
                prep = p
            case 'runge-kutta':
                p, q = stoch.runge_kutta_step(
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
                    ninit,
                    ncons,
                    icons,
                    rng,
                    ph,
                    order=order,
                    heun=heun,
                )
                prep = p
            case 'symplectic':
                p, q = stoch.symplectic_step(
                    it,
                    dt,
                    Et,
                    Et_hdt,
                    h0,
                    ud,
                    p,
                    q,
                    cutoff,
                    nareps,
                    ninit,
                    ncons,
                    icons,
                    rng,
                    ph,
                )
                if (it + 1) % nrep == 0:
                    prep = p.copy()
                    prep = stoch.symplectic_align(
                        dt,
                        Et,
                        h0,
                        ud,
                        prep,
                        q,
                        cutoff,
                        nareps,
                        ninit,
                        ncons,
                        icons,
                        rng,
                        ph,
                    )
                else:
                    prep = p
            case _:
                raise ValueError(f'Unknown method: {method}, exiting...')
    else:
        match method:
            case 'euler':
                p, q = euler_step(
                    dt,
                    Et,
                    h0,
                    ud,
                    p,
                    q,
                )
                prep = p
            case 'runge-kutta':
                p, q = runge_kutta_step(
                    dt,
                    Et,
                    Et_hdt,
                    Et_dt,
                    h0,
                    ud,
                    p,
                    q,
                    order=order,
                    heun=heun,
                )
                prep = p
            case 'symplectic':
                p, q = symplectic_step(
                    it,
                    dt,
                    Et,
                    Et_hdt,
                    h0,
                    ud,
                    p,
                    q,
                )
                if (it + 1) % nrep == 0:
                    prep = p.copy()
                    prep = symplectic_align(
                        dt,
                        Et,
                        h0,
                        ud,
                        prep,
                        q,
                    )
                else:
                    prep = p
            case _:
                raise ValueError(f'Unknown method: {method}, exiting...')

    return p, q, prep


def field_function_wrapper(
        function: str,
        I: float,
        omega: float,
        sigma: float,
        center: float,
        beta: float,
        t: float,
        dt: float,
    ) -> tuple[float, float, float]:
    r''' TODO: Write a docstring.
    '''
    match function:
        case 'continuous':
            Et = cw_t(I, omega, t)
            Et_hdt = cw_t(I, omega, t + dt/2)
            Et_dt = cw_t(I, omega, t + dt)
        case 'transform-limited':
            Et = tl_t(I, omega, sigma, center, t)
            Et_hdt = tl_t(I, omega, sigma, center, t + dt/2)
            Et_dt = tl_t(I, omega, sigma, center, t + dt)
        case 'chirped':
            Et = c_t(I, omega, sigma, center, beta, t)
            Et_hdt = c_t(I, omega, sigma, center, beta, t + dt/2)
            Et_dt = c_t(I, omega, sigma, center, beta, t + dt)
        case _:
            raise ValueError(f'Unknown field function: {function}, exiting...')

    return Et, Et_hdt, Et_dt


def report_input_parameters(
        parent: bool,
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
        ninit: float,
        ncalc: int,
        csv: str,
    ) -> None:
    r''' TODO: Write docstring here.
    '''
    if parent:
        print(
            f'User input parameters (in atomic units):\n'
            f' === field parameters ===\n'
            f'    function: {function:<22}\n'
            f'    I: {I:< 22.12E}\n'
            f'    omega: {omega:< 22.12E}\n'
            f'    sigma: {sigma:< 22.12E}\n'
            f'    center: {center:< 22.12E}\n'
            f'    beta: {beta:< 22.12E}\n'
            f'    tf: {tf:< 22.12E}\n'
            f'    dt: {dt:< 22.12E}\n'
            f' === iteration parameters ===\n'
            f'    nits: {nits:< 22d}\n'
            f'    nrep: {nrep:< 22d}\n'
            f'    method: {method:<22}\n'
            f'    order: {order:< 22d}\n'
            f'    heun: {"True" if heun else "False":<22}\n'
            f'    comm_file: {comm_file:<22}\n'
            f' === quantum Monte Carlo parameters ===\n'
            f'    stochastic: {"True" if stochastic else "False":<22}\n'
            f'    c0_population: {c0_population:< 22.12E}\n'
            f'    rng_seed: {rng_seed:< 22d}\n'
            f'    cutoff: {cutoff:< 22.12E}\n'
            f'    nareps: {nareps:< 22d}\n'
            f'    ncalc: {ncalc:< 22d}\n'
            f'    ninit: {ninit:< 22.12E}\n'
            f' === storage parameters ===\n'
            f'    csv: {csv:<22}\n'
        )
    return
