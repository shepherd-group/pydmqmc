#!/usr/bin/env python

import numpy as np

from numba import njit
from mpi import ParallelHelper

from stochastic_utils import (
    cdot_mapped,
    paired_cdot_mapped,
    compress,
    p_merge_spawns,
    q_merge_spawns,
    pq_compress,
    pq_merge_spawns,
    heun_accumulate_spawns,
    rkfour_accumulate_spawns,
)


def euler_step(
        dt: float,
        Et: float,
        h0: np.array,
        ud: np.array,
        p: np.array,
        q: np.array,
        cutoff: float,
        nareps: int,
        ninit: float,
        ncons: np.array,
        icons: np.array,
        rng: np.random.Generator,
        ph: ParallelHelper,
    ) -> tuple[np.array, np.array]:
    r''' TODO: Docstring here.

    Breaking out the math like symplectic

        \Psi = \Re{\Psi} + j \Im{\Psi}

        d/dt \Psi = -j H \Psi

    Assume H is only real, then:

        d/dt \Psi = -j H ( \Re{\Psi} + j \Im{\Psi} )

        d \Psi = -j dt H \Re{\Psi} + -j^2 dt H \Im{\Psi}

        d \Psi = dt H \Im{\Psi} - j dt H \Re{\Psi}

    Which gives

        d \Re{\Psi} = dt H \Im{\Psi}

        d \Im{\Psi} = - dt H \Re{\Psi}
    '''
    dq = cdot_mapped(
        h0,
        ud,
        p,
        Et,
        dt,
        cutoff,
        nareps,
        ninit,
        ncons,
        icons,
        rng,
        ph.imin,
        ph.imax,
    )
    dp = cdot_mapped(
        h0,
        ud,
        q,
        Et,
        dt,
        cutoff,
        nareps,
        ninit,
        ncons,
        icons,
        rng,
        ph.imin,
        ph.imax,
    )

    if ph.parallel:
        ''' TODO: Could reduce to a single comm call, and hash
        for load balancing. Then each processor deals with its own
        compression and initiator checks. Thoughts for a later date.
        '''
        # Accumulate all spawns to parent.
        dp, dq = ph.dpdq_reduce(dp, dq)

        # Merge the spawns into main.
        if ph.parent:
            p, q = pq_merge_spawns(ninit, p, q, dp, dq)
            p, q = pq_compress(p, q, rng)

        # Broadcast the new p/q array to children.
        p, q = ph.pq_bcast(p, q)
    else:
        p, q = pq_merge_spawns(ninit, p, q, dp, dq)
        p, q = pq_compress(p, q, rng)

    return p, q


def symplectic_step(
        i: int,
        dt: float,
        Et: float,
        Et_hdt: float,
        h0: np.array,
        ud: np.array,
        p: np.array,
        q: np.array,
        cutoff: float,
        nareps: int,
        ninit: float,
        ncons: np.array,
        icons: np.array,
        rng: np.random.Generator,
        ph: ParallelHelper,
    ) -> tuple[np.array, np.array]:
    r''' TODO: Docstring here.
    '''
    dp = cdot_mapped(
        h0,
        ud,
        q,
        Et,
        (0.5 if i == 0 else 1.0)*dt,
        cutoff,
        nareps,
        ninit,
        ncons,
        icons,
        rng,
        ph.imin,
        ph.imax,
    )

    if ph.parallel:
        dp = ph.reduce(dp)

        if ph.parent:
            p = p_merge_spawns(ninit, p, q, dp)
            p = compress(p, rng)

        p = ph.bcast(p)
    else:
        p = p_merge_spawns(ninit, p, q, dp)
        p = compress(p, rng)

    dq = cdot_mapped(
        h0,
        ud,
        p,
        Et_hdt,
        dt,
        cutoff,
        nareps,
        ninit,
        ncons,
        icons,
        rng,
        ph.imin,
        ph.imax,
    )

    if ph.parallel:
        dq = ph.reduce(dq)

        if ph.parent:
            q = q_merge_spawns(ninit, p, q, dq)
            q = compress(q, rng)

        q = ph.bcast(q)
    else:
        q = q_merge_spawns(ninit, p, q, dq)
        q = compress(q, rng)

    return p, q


def symplectic_align(
        dt: float,
        Et: float,
        h0: np.array,
        ud: np.array,
        p: np.array,
        q: np.array,
        cutoff: float,
        nareps: int,
        ninit: float,
        ncons: np.array,
        icons: np.array,
        rng: np.random.Generator,
        ph: ParallelHelper,
    ) -> np.array:
    r''' TODO: Docstring here.
    '''
    dp = cdot_mapped(
        h0,
        ud,
        q,
        Et,
        0.5*dt,
        cutoff,
        nareps,
        ninit,
        ncons,
        icons,
        rng,
        ph.imin,
        ph.imax,
    )

    if ph.parallel:
        dp = ph.reduce(dp)

        if ph.parent:
            p = p_merge_spawns(ninit, p, q, dp)
            p = compress(p, rng)

        p = ph.bcast(p)
    else:
        p = p_merge_spawns(ninit, p, q, dp)
        p = compress(p, rng)

    return p


def runge_kutta_step(
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
        ninit: float,
        ncons: np.array,
        icons: np.array,
        rng: np.random.Generator,
        ph: ParallelHelper,
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

    dq_k1 = cdot_mapped(
        h0,
        ud,
        p,
        Et,
        dt,
        cutoff,
        nareps,
        ninit,
        ncons,
        icons,
        rng,
        ph.imin,
        ph.imax,
    )
    dp_k1 = cdot_mapped(
        h0,
        ud,
        q,
        Et,
        dt,
        cutoff,
        nareps,
        ninit,
        ncons,
        icons,
        rng,
        ph.imin,
        ph.imax,
    )

    if order == 1:
        # This is just Euler's method.
        dq = dq_k1
        dp = dp_k1
    elif order == 2:
        r''' Need a local copy for all the guaranteed spawns to
        do intermediates correctly. Initiator probably
        suffers in this case since it can't really participate.
        Also this probably adds a lot of compute time and memory bloat.
        TODO: fix the bloat from all the intermediates etc...
        TODO: there is no way that array slicing is efficient...
        '''
        tmp_dp_k1, tmp_dq_k1 = ph.pq_allreduce(
            dp_k1[:1, :].copy().T,
            dq_k1[:1, :].copy().T,
        )

        if heun:
            # Heun's method
            dq_k2 = paired_cdot_mapped(
                h0,
                ud,
                p,
                tmp_dp_k1,
                -1.0,
                Et_dt,
                dt,
                cutoff,
                nareps,
                ninit,
                ncons,
                icons,
                rng,
                ph.imin,
                ph.imax,
            )
            dp_k2 = paired_cdot_mapped(
                h0,
                ud,
                q,
                tmp_dq_k1,
                1.0,
                Et_dt,
                dt,
                cutoff,
                nareps,
                ninit,
                ncons,
                icons,
                rng,
                ph.imin,
                ph.imax,
            )

            #dq = (1/2)*(dq_k1 + dq_k2)
            #dp = (1/2)*(dp_k1 + dp_k2)
            dp, dq = heun_accumulate_spawns(
                ninit,
                dp_k1,
                dp_k2,
                dq_k1,
                dq_k2,
            )
        else:
            # Midpoint method
            dq_k2 = paired_cdot_mapped(
                h0,
                ud,
                p,
                tmp_dp_k1,
                -0.5,
                Et_hdt,
                dt,
                cutoff,
                nareps,
                ninit,
                ncons,
                icons,
                rng,
                ph.imin,
                ph.imax,
            )
            dp_k2 = paired_cdot_mapped(
                h0,
                ud,
                q,
                tmp_dq_k1,
                0.5,
                Et_hdt,
                dt,
                cutoff,
                nareps,
                ninit,
                ncons,
                icons,
                rng,
                ph.imin,
                ph.imax,
            )

            dq = dq_k2
            dp = dp_k2
    elif order == 4:
        tmp_dp_k1, tmp_dq_k1 = ph.pq_allreduce(
            dp_k1[:1, :].copy().T,
            dq_k1[:1, :].copy().T,
        )

        dq_k2 = paired_cdot_mapped(
            h0,
            ud,
            p,
            tmp_dp_k1,
            -0.5,
            Et_hdt,
            dt,
            cutoff,
            nareps,
            ninit,
            ncons,
            icons,
            rng,
            ph.imin,
            ph.imax,
        )
        dp_k2 = paired_cdot_mapped(
            h0,
            ud,
            q,
            tmp_dq_k1,
            0.5,
            Et_hdt,
            dt,
            cutoff,
            nareps,
            ninit,
            ncons,
            icons,
            rng,
            ph.imin,
            ph.imax,
        )

        tmp_dp_k2, tmp_dq_k2 = ph.pq_allreduce(
            dp_k2[:1, :].copy().T,
            dq_k2[:1, :].copy().T,
        )

        dq_k3 = paired_cdot_mapped(
            h0,
            ud,
            p,
            tmp_dp_k2,
            -0.5,
            Et_hdt,
            dt,
            cutoff,
            nareps,
            ninit,
            ncons,
            icons,
            rng,
            ph.imin,
            ph.imax,
        )
        dp_k3 = paired_cdot_mapped(
            h0,
            ud,
            q,
            tmp_dq_k2,
            0.5,
            Et_hdt,
            dt,
            cutoff,
            nareps,
            ninit,
            ncons,
            icons,
            rng,
            ph.imin,
            ph.imax,
        )

        tmp_dp_k3, tmp_dq_k3 = ph.pq_allreduce(
            dp_k3[:1, :].copy().T,
            dq_k3[:1, :].copy().T,
        )

        dq_k4 = paired_cdot_mapped(
            h0,
            ud,
            p,
            tmp_dp_k3,
            -1.0,
            Et_dt,
            dt,
            cutoff,
            nareps,
            ninit,
            ncons,
            icons,
            rng,
            ph.imin,
            ph.imax,
        )
        dp_k4 = paired_cdot_mapped(
            h0,
            ud,
            q,
            tmp_dq_k3,
            1.0,
            Et_dt,
            dt,
            cutoff,
            nareps,
            ninit,
            ncons,
            icons,
            rng,
            ph.imin,
            ph.imax,
        )

        #dq = (1/6)*(dq_k1 + 2*dq_k2 + 2*dq_k3 + dq_k4)
        #dp = (1/6)*(dp_k1 + 2*dp_k2 + 2*dp_k3 + dp_k4)
        dp, dq = rkfour_accumulate_spawns(
            ninit,
            dp_k1,
            dp_k2,
            dp_k3,
            dp_k4,
            dq_k1,
            dq_k2,
            dq_k3,
            dq_k4,
        )

    if ph.parallel:
        dp, dq = ph.dpdq_reduce(dp, dq)

        if ph.parent:
            p, q = pq_merge_spawns(ninit, p, q, dp, dq)
            p, q = pq_compress(p, q, rng)

        p, q = ph.pq_bcast(p, q)
    else:
        p, q = pq_merge_spawns(ninit, p, q, dp, dq)
        p, q = pq_compress(p, q, rng)

    return p, q
