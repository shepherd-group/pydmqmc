#!/usr/bin/env python

import os
import sys
import numpy as np
import numba as nb
import pandas as pd

from typing import Any, Tuple
from local_utils import report, get_empty_data_dict
from numpy.typing import NDArray as Array


@nb.njit
def set_numba_seed(seed: int) -> None:
    np.random.seed(seed)
    return


@nb.njit
def get_nattempts(Mij: float, nadd: float = 0.0) -> int:
    mag = abs(Mij)
    return int(mag + np.random.random() if mag > nadd else 0)


@nb.njit
def round_main(M: Array) -> Array:

    ndets = M.shape[0]

    for p in range(ndets):
        for q in range(ndets):
            mag = abs(M[p,q])
            sign = np.sign(M[p,q])

            if mag < 1.0:
                mag = 1.0 if (np.random.random() < mag) else 0.0
                M[p,q] = sign*mag

    return M


@nb.njit
def get_new_random_index(old, nindices) -> int:
    new = old

    while new == old:
        new = np.random.randint(nindices)

    return new


@nb.njit
def attempt_spawning(p, q, na, axis, dt, cutoff, sign, drho, H) -> Array:

    ndets = H.shape[0]

    if axis == 0:
        k = get_new_random_index(p, ndets)
        Hconn = H[k,p].real
        l1, l2 = k, q
    else:
        k = get_new_random_index(q, ndets)
        Hconn = H[q,k].real
        l1, l2 = p, k

    pspawn = Hconn*dt*(ndets - 1)/na
    mspawn = abs(pspawn)

    if mspawn >= cutoff:
        drho[l1,l2] += sign*pspawn
    elif np.random.random() < mspawn/cutoff:
        drho[l1,l2] += cutoff*sign*np.sign(pspawn)

    return drho


@nb.njit
def stochastic_euler(dt, cutoff, nadd, rho, H, drho) -> Any:
    r''' The Von Neumann equation is given as:

        d/dt \rho = -i [ H \rho - \rho H ]

        d/dt \rho_ab = -i [ \sum_k H_ak \rho_kb - \sum_k \rho_ak H_kb ]

    So a given \rho_pq, we contribute to the following locations:

        \rho_kq += H_kp \rho_pq

        \rho_pk -= \rho_pq H_qk
    '''

    na = 1
    ndets = rho.shape[0]

    for p in range(ndets):
        for q in range(ndets):
            rho_pq = rho[p,q]
            rsign = -1j*np.sign(rho_pq.real)
            isign = -1j*np.sign(rho_pq.imag)*1j

            drho[p,q] += rho_pq*(1 - 1j*dt*(H[p,p] - H[q,q]))

            # Real walker spawning
            for attempt in range(na*get_nattempts(rho_pq.real, nadd)):
                # Spawn along rows
                attempt_spawning(p, q, na, 0, dt, cutoff, rsign, drho, H)

                # Spawn along columns
                attempt_spawning(p, q, na, 1, dt, cutoff, -rsign, drho, H)

            # Imaginary walker spawning
            for attempt in range(na*get_nattempts(rho_pq.imag, nadd)):
                # Spawn along rows
                attempt_spawning(p, q, na, 0, dt, cutoff, isign, drho, H)

                # Spawn along columns
                attempt_spawning(p, q, na, 1, dt, cutoff, -isign, drho, H)

    return drho


def do_stochastic_euler(
        tf: float,
        dt: float,
        H: Array,
        state1: int,
        state2: int,
        walkers: float,
        cutoff: float,
        nadd: float,
    ) -> Any:
    # Propagate based on stochastic version of equations in:
    # https://doi.org/10.1063/1.5115323

    rho = np.zeros(H.shape, dtype=complex)

    rho[state1,state1] += walkers

    data = get_empty_data_dict()
    data = report(0, rho, H, state1, state2, data)

    nsteps = int(tf/dt)

    new = np.zeros(rho.shape, dtype=complex)

    for step in range(nsteps):
        new[:,:] = 0.0 + 0.0j

        stochastic_euler(dt, cutoff, nadd, rho, H, new)

        new_real = round_main(new.real)
        new_imag = round_main(new.imag)

        rho[:,:] = new_real + 1j*new_imag

        if (step + 1) % 100 == 0:
            data = report(step+1, rho, H, state1, state2, data)

    data = pd.DataFrame(data)

    data['tau'] = data['step']*dt

    return data


@nb.njit
def stochastic_symplectic(dt, cutoff, nadd, M, H, dM) -> Any:
    r''' The Von Neumann equation is given as:

        d/dt \rho = -i [ H \rho - \rho H ]
    '''

    na = 1
    ndets = M.shape[0]

    for p in range(ndets):
        for q in range(ndets):
            M_pq = M[p,q]
            sign = np.sign(M_pq)

            dM[p,q] += M_pq*dt*(H[p,p].real - H[q,q].real)

            # Real walker spawning
            for attempt in range(na*get_nattempts(M_pq, nadd)):
                # Spawn along rows
                attempt_spawning(p, q, na, 0, dt, cutoff, sign, dM, H)

                # Spawn along columns
                attempt_spawning(p, q, na, 1, dt, cutoff, -sign, dM, H)

    return dM


def do_stochastic_symplectic(
        tf: float,
        dt: float,
        H: Array,
        state1: int,
        state2: int,
        walkers: float,
        cutoff: float,
        nadd: float,
    ) -> Any:
    # Propagate based on stochastic version of equations in:
    # https://doi.org/10.1021/acs.jctc.8b00381

    rho = np.zeros(H.shape, dtype=complex)

    rho[state1,state1] += walkers

    data = get_empty_data_dict()
    data = report(0, rho, H, state1, state2, data)

    nsteps = int(tf/dt)

    p = rho.imag
    q = rho.real

    new = np.zeros(rho.shape, dtype=float)

    for step in range(nsteps):
        new *= 0.0

        stochastic_symplectic(dt, cutoff, nadd, q, H, new)

        p[:,:] = p - (0.5 if step == 0 else 1.0)*new

        p = round_main(p)

        new *= 0.0

        stochastic_symplectic(dt, cutoff, nadd, p, H, new)

        q[:,:] = q + new

        q = round_main(q)

        if (step + 1) % 100 == 0:
            new *= 0.0
            stochastic_symplectic(dt, cutoff, nadd, q, H, new)
            tmp = p - 0.5*new
            tmp = round_main(tmp)
            rho = q + 1j*tmp
            data = report(step+1, rho, H, state1, state2, data)

    data = pd.DataFrame(data)

    data['tau'] = data['step']*dt

    return data


@nb.njit
def stochastic_bloch(dt, cutoff, nadd, rho, H, drho) -> Any:
    r''' The symmetric Bloch equation is given as:

        d/dt \rho = - 0.5 [ H \rho + \rho H ]

    '''

    na = 1
    ndets = rho.shape[0]

    for p in range(ndets):
        for q in range(ndets):
            rho_pq = rho[p,q]
            sign = np.sign(rho_pq)

            drho[p, q] += rho_pq*(1 - 0.5*dt*(H[p,p] + H[q,q]))

            # Real walker spawning
            for attempt in range(na*get_nattempts(rho_pq, nadd)):
                # Spawn along rows
                attempt_spawning(p, q, na, 0, dt/2, cutoff, -sign, drho, H)

                # Spawn along columns
                attempt_spawning(p, q, na, 1, dt/2, cutoff, -sign, drho, H)

    return drho


def do_stochastic_bloch(
        tf: float,
        dt: float,
        H: Array,
        state1: int,
        state2: int,
        walkers: float,
        cutoff: float,
        nadd: float,
    ) -> Any:
    # Propagate based on stochastic version of equations in:
    # https://doi.org/10.1063/1.5115323

    K = H.copy() - (np.diag(H).min()*np.eye(H.shape[0]))

    rho = walkers*np.eye(H.shape[0])/H.shape[0]

    data = get_empty_data_dict()
    data = report(0, rho, H, state1, state2, data)

    nsteps = int(tf/dt)

    new_rho = np.zeros(rho.shape, dtype=float)

    for step in range(nsteps):
        new_rho *= 0.0

        stochastic_bloch(dt, cutoff, nadd, rho, K, new_rho)

        rho[:,:] = new_rho

        if (step + 1) % 100 == 0:
            data = report(step+1, rho, H, state1, state2, data)

    data = pd.DataFrame(data)

    data['tau'] = data['step']*dt

    return data
