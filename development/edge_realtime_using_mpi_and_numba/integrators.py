#!/usr/bin/env python

import numpy as np

from numba import njit, prange


@njit(parallel=True)
def cdot(H: np.array, p: np.array) -> np.array:
    r''' TODO: Docstring here.
    '''
    v = np.zeros(p.shape, dtype=float)

    for i in prange(p.shape[0]):
        v[i, 0] = np.dot(H[i, :], p[:, 0])

    return v


@njit(parallel=True)
def euler_step(
        dt: float,
        Et: float,
        h0: np.array,
        ud: np.array,
        p: np.array,
        q: np.array,
    ) -> tuple[np.array, np.array]:
    r''' TODO: Docstring here.
    '''
    Ht = h0 - ud*Et

    dq = cdot(dt*Ht, p)
    dp = cdot(dt*Ht, q) 

    q = q + dq
    p = p - dp

    return p, q


@njit(parallel=True)
def symplectic_step(
        i: int,
        dt: float,
        Et: float,
        Et_hdt: float,
        h0: np.array,
        ud: np.array,
        p: np.array,
        q: np.array,
    ) -> tuple[np.array, np.array]:
    r''' TODO: Docstring here.
    '''
    Ht = h0 - ud*Et
    Ht_hdt = h0 - ud*Et_hdt

    dp = cdot(Ht, q)

    p = p - (0.5 if i == 0 else 1.0)*dt*dp

    dq = cdot(Ht_hdt, p)

    q = q + dt*dq

    return p, q


@njit(parallel=True)
def symplectic_align(
        dt: float,
        Et: float,
        h0: np.array,
        ud: np.array,
        p: np.array,
        q: np.array,
    ) -> np.array:
    r''' TODO: Docstring here.
    '''
    Ht = h0 - ud*Et

    dp = cdot(Ht, q)

    p = p - 0.5*dt*dp

    return p


@njit(parallel=True)
def runge_kutta_step(
        dt: float,
        Et: float,
        Et_hdt: float,
        Et_dt: float,
        h0: np.array,
        ud: np.array,
        p: np.array,
        q: np.array,
        order: int = 4,
        heun: bool = False,
    ) -> tuple[np.array, np.array]:
    r''' TODO: Docstring here.
    '''
    if order not in [1, 2, 4]:
        print(
            'ERROR: Only the 2nd (midpoint and Heun\'s) or 4th order '
            'RK(n) methods are implemented, please send patches!'
        )
        return p, q

    Ht = h0 - ud*Et
    Ht_hdt = h0 - ud*Et_hdt
    Ht_dt = h0 - ud*Et_dt

    dq_k1 = cdot(Ht, p)
    dp_k1 = cdot(Ht, q)

    if order == 1:
        dq = dt*dq_k1
        dp = dt*dp_k1
    elif order == 2:
        # Heun's seems slightly better in my simple
        # test (closer to 1.0 is better).
        # midpoint final norm: 1.006089835488E+00
        #   Heun's final norm: 1.005793974805E+00
        if heun:
            # Heun's method
            dq_k2 = cdot(Ht_dt, p - dt*dp_k1)
            dp_k2 = cdot(Ht_dt, q + dt*dq_k1)

            dq = (dt/2)*(dq_k1 + dq_k2)
            dp = (dt/2)*(dp_k1 + dp_k2)
        else:
            # Midpoint method
            dq_k2 = cdot(Ht_hdt, p - dt*dp_k1/2)
            dp_k2 = cdot(Ht_hdt, q + dt*dq_k1/2)

            dq = dt*dq_k2
            dp = dt*dp_k2
    elif order == 4:
        dq_k2 = cdot(Ht_hdt, p - dt*dp_k1/2)
        dp_k2 = cdot(Ht_hdt, q + dt*dq_k1/2)

        dq_k3 = cdot(Ht_hdt, p - dt*dp_k2/2)
        dp_k3 = cdot(Ht_hdt, q + dt*dq_k2/2)

        dq_k4 = cdot(Ht_dt, p - dt*dp_k3)
        dp_k4 = cdot(Ht_dt, q + dt*dq_k3)

        dq = (dt/6)*(dq_k1 + 2*dq_k2 + 2*dq_k3 + dq_k4)
        dp = (dt/6)*(dp_k1 + 2*dp_k2 + 2*dp_k3 + dp_k4)

    q = q + dq
    p = p - dp

    return p, q


@njit(parallel=True)
def get_state_populations(
        q: np.array,
        p: np.array,
        Psi0: np.array,
        Psii: np.array,
    ) -> tuple[float, ]:
    r''' TODO: Docstring here.
    '''
    bra = q + 1j*p
    ket = np.conjugate(bra)

    ndets0 = Psi0.shape[0]
    ndetsi = Psii.shape[0]

    pops = np.zeros(1 + ndetsi, dtype=float)

    for indx in prange(1 + ndetsi):
        if indx == 0:
            ovrlp = (ket[:ndets0, 0]*Psi0).sum()
            pops[indx] = ovrlp.real**2 + ovrlp.imag**2
        else:
            ovrlp = (ket[ndets0:, 0]*Psii[indx-1]).sum()
            pops[indx] = ovrlp.real**2 + ovrlp.imag**2

    return pops
