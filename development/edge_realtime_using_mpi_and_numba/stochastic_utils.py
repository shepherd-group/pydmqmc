#!/usr/bin/env python

import numpy as np

from numba import njit, uint16, prange


@njit
def get_connection_map(H: np.array) -> tuple[np.array, ...]:
    r''' TODO: Docstring here.
    '''
    ndets = H.shape[0]

    ncons = np.zeros(ndets, dtype=uint16)

    for i in range(ndets):
        ival = np.nonzero(H[:, i])[0]
        ival = ival[ival != i]
        ncon = ival.shape[0]
        ncons[i] = ncon

    max_ncon = max(ncons)

    icons = np.zeros((ndets, max_ncon), dtype=uint16)

    for i in range(ndets):
        ival = np.nonzero(H[:, i])[0]
        ival = ival[ival != i]
        icons[i, :ncons[i]] = ival

    return ncons, icons


@njit
def round_main(c: float, rng: np.random.Generator) -> float:
    r''' TODO: Docstring here.
    '''
    nw = abs(c)

    if 0.0 < nw < 1.0:
        if (nw + rng.random()) < 1.0:
            c = 0.0
        else:
            c = np.sign(c)

    return c


@njit
def compress(
        v: np.array,
        rng: np.random.Generator,
    ) -> np.array:
    r''' TODO: Docstring here.
    '''
    for i in range(v.shape[0]):
        v[i, 0] = round_main(v[i, 0], rng)

    return v


@njit
def pq_compress(
        p: np.array,
        q: np.array,
        rng: np.random.Generator,
    ) -> tuple[np.array, np.array]:
    r''' TODO: Docstring here.
    '''
    for i in range(p.shape[0]):
        p[i, 0] = round_main(p[i, 0], rng)
        q[i, 0] = round_main(q[i, 0], rng)

    return p, q


@njit
def p_merge_spawns(
        ninit: float,
        p: np.array,
        q: np.array,
        dp: np.array,
    ) -> np.array:
    r''' TODO: Docstring here.
    '''
    for i in range(p.shape[0]):
        dpi = dp[0, i]

        if ninit > 0.0:
            if (p[i, 0] != 0.0) or (q[i, 0] != 0.0):
                dpi += dp[1, i] + dp[3, i]
            else:
                # dp
                # (-) coherency.
                if dp[2, i] > 1.0:
                    dpi += dp[1, i]
                # (+) coherency.
                if dp[4, i] > 1.0:
                    dpi += dp[3, i]

        p[i, 0] -= dpi

    return p


@njit
def q_merge_spawns(
        ninit: float,
        p: np.array,
        q: np.array,
        dq: np.array,
    ) -> np.array:
    r''' TODO: Docstring here.
    '''
    for i in range(p.shape[0]):
        dqi = dq[0, i]

        if ninit > 0.0:
            if (p[i, 0] != 0.0) or (q[i, 0] != 0.0):
                dqi += dq[1, i] + dq[3, i]
            else:
                # dq
                # (-) coherency.
                if dq[2, i] > 1.0:
                    dqi += dq[1, i]
                # (+) coherency.
                if dq[4, i] > 1.0:
                    dqi += dq[3, i]

        q[i, 0] += dqi

    return q


@njit
def pq_merge_spawns(
        ninit: float,
        p: np.array,
        q: np.array,
        dp: np.array,
        dq: np.array,
    ) -> tuple[np.array, np.array]:
    r''' TODO: Docstring here.
    '''
    for i in range(p.shape[0]):
        dpi = dp[0, i]
        dqi = dq[0, i]

        if ninit > 0.0:
            if (p[i, 0] != 0.0) or (q[i, 0] != 0.0):
                dpi += dp[1, i] + dp[3, i]
                dqi += dq[1, i] + dq[3, i]
            else:
                # dp
                # (-) coherency.
                if dp[2, i] > 1.0:
                    dpi += dp[1, i]
                # (+) coherency.
                if dp[4, i] > 1.0:
                    dpi += dp[3, i]

                # dq
                # (-) coherency.
                if dq[2, i] > 1.0:
                    dqi += dq[1, i]
                # (+) coherency.
                if dq[4, i] > 1.0:
                    dqi += dq[3, i]

        p[i, 0] -= dpi
        q[i, 0] += dqi

    return p, q


@njit
def heun_accumulate_spawns(
        ninit: float,
        dp_k1: np.array,
        dp_k2: np.array,
        dq_k1: np.array,
        dq_k2: np.array,
    ) -> tuple[np.array, np.array]:
    r''' TODO: Docstring here.
    '''
    dp = np.zeros(dp_k1.shape, dtype=float)
    dq = np.zeros(dq_k1.shape, dtype=float)

    if ninit > 0.0:
        for i in range(dp_k1.shape[1]):
            # Counts for (-)/(+) non-initiator spawns
            dp[2, i] = dp_k1[2, i] + dp_k2[2, i]
            dq[2, i] = dq_k1[2, i] + dq_k2[2, i]
            dp[4, i] = dp_k1[4, i] + dp_k2[4, i]
            dq[4, i] = dq_k1[4, i] + dq_k2[4, i]
            # Add up unrestricted and restricted spawns.
            dp[0, i] = 0.5*(dp_k1[0, i] + dp_k2[0, i])
            dq[0, i] = 0.5*(dq_k1[0, i] + dq_k2[0, i])
            dp[1, i] = 0.5*(dp_k1[1, i] + dp_k2[1, i])
            dq[1, i] = 0.5*(dq_k1[1, i] + dq_k2[1, i])
            dp[3, i] = 0.5*(dp_k1[3, i] + dp_k2[3, i])
            dq[3, i] = 0.5*(dq_k1[3, i] + dq_k2[3, i])
    else:
        for i in range(dp_k1.shape[1]):
            # Unrestricted spawning, just add.
            dp[0, i] = 0.5*(dp_k1[0, i] + dp_k2[0, i])
            dq[0, i] = 0.5*(dq_k1[0, i] + dq_k2[0, i])

    return dp, dq


@njit
def rkfour_accumulate_spawns(
        ninit: float,
        dp_k1: np.array,
        dp_k2: np.array,
        dp_k3: np.array,
        dp_k4: np.array,
        dq_k1: np.array,
        dq_k2: np.array,
        dq_k3: np.array,
        dq_k4: np.array,
    ) -> tuple[np.array, np.array]:
    r''' TODO: Docstring here.
    '''
    dp = np.zeros(dp_k1.shape, dtype=float)
    dq = np.zeros(dq_k1.shape, dtype=float)

    if ninit > 0.0:
        for i in range(dp_k1.shape[1]):
            # Counts for (-)/(+) non-initiator spawns
            dp[2, i] = dp_k1[2, i] + dp_k2[2, i] + dp_k3[2, i] + dp_k4[2, i]
            dq[2, i] = dq_k1[2, i] + dq_k2[2, i] + dq_k3[2, i] + dq_k4[2, i]
            dp[4, i] = dp_k1[4, i] + dp_k2[4, i] + dp_k3[4, i] + dp_k4[4, i]
            dq[4, i] = dq_k1[4, i] + dq_k2[4, i] + dq_k3[4, i] + dq_k4[4, i]
            # Add up unrestricted and restricted spawns.
            dp[0, i] = (1/6)*(
                dp_k1[0, i]
                + 2*dp_k2[0, i]
                + 2*dp_k3[0, i]
                + dp_k4[0, i]
            )
            dq[0, i] = (1/6)*(
                dq_k1[0, i]
                + 2*dq_k2[0, i]
                + 2*dq_k3[0, i]
                + dq_k4[0, i]
            )
            dp[1, i] = (1/6)*(
                dp_k1[1, i]
                + 2*dp_k2[1, i]
                + 2*dp_k3[1, i]
                + dp_k4[1, i]
            )
            dq[1, i] = (1/6)*(
                dq_k1[1, i]
                + 2*dq_k2[1, i]
                + 2*dq_k3[1, i]
                + dq_k4[1, i]
            )
            dp[3, i] = (1/6)*(
                dp_k1[3, i]
                + 2*dp_k2[3, i]
                + 2*dp_k3[3, i]
                + dp_k4[3, i]
            )
            dq[3, i] = (1/6)*(
                dq_k1[3, i]
                + 2*dq_k2[3, i]
                + 2*dq_k3[3, i]
                + dq_k4[3, i]
            )
    else:
        for i in range(dp_k1.shape[1]):
            # Unrestricted spawning, just add.
            dp[0, i] = (1/6)*(
                dp_k1[0, i]
                + 2*dp_k2[0, i]
                + 2*dp_k3[0, i]
                + dp_k4[0, i]
            )
            dq[0, i] = (1/6)*(
                dq_k1[0, i]
                + 2*dq_k2[0, i]
                + 2*dq_k3[0, i]
                + dq_k4[0, i]
            )

    return dp, dq


@njit
def round_spawn(
        pspawn: float,
        cutoff: float,
        rng: np.random.Generator,
    ) -> float:
    r''' TODO: Docstring here.
    '''
    mspawn = abs(pspawn)

    if mspawn < cutoff:
        if ((mspawn/cutoff) + rng.random()) < 1.0:
            pspawn = 0.0
        else:
            pspawn = np.sign(pspawn)*cutoff

    return pspawn


@njit
def get_mapped_connections(
        k: int,
        nw: float,
        nareps: int,
        ncons: np.array,
        icons: np.array,
        rng: np.random.Generator,
    ) -> tuple[float, np.array]:
    r''' TODO: Docstring here.
    '''
    # How many random connections to generate (spawn attempts).
    nattempts = nareps*int(abs(nw) + rng.random())

    # Valid connections.
    ncon = ncons[k]
    ival = icons[k, :ncon]
    inv_pgen = ncon/nareps

    # Generate our random selections.
    icons = ival[rng.integers(low=0, high=ncon, size=nattempts)]

    return inv_pgen, icons


@njit
def cdot_mapped(
        h0: np.array,
        ud: np.array,
        p: np.array,
        Et: float,
        dt: float,
        cutoff: float,
        nareps: int,
        ninit: float,
        ncons: np.array,
        icons: np.array,
        rng: np.random.Generator,
        imin: int,
        imax: int,
    ) -> np.array:
    r''' TODO: Docstring here.
    '''
    #nrcons = np.zeros(p.shape[0], dtype=float)

    if ninit > 0.0:
        # 0: unrestricted spawns
        # 1: (-) non-initiator spawns
        # 2: (-) non-initiator counts
        # 3: (+) non-initiator spawns
        # 4: (+) non-initiator counts
        spawns = np.zeros((5, p.shape[0]), dtype=float)
    else:
        spawns = np.zeros((1, p.shape[0]), dtype=float)

    ndets = p.shape[0]

    for k in range(imin, imax + 1):
        nw = p[k, 0]
        isinit = abs(nw) >= ninit

        if nw == 0.0:
            continue

        # Diagonal term.
        spawns[0, k] += dt*(h0[k, k] - ud[k, k]*Et)*nw

        # Then randomly select i from those remaining.
        sign = np.sign(nw)

        inv_pgen, ricons = get_mapped_connections(
            k,
            nw,
            nareps,
            ncons,
            icons,
            rng,
        )

        for icon in ricons:
            nspawn = round_spawn(
                sign*dt*(h0[icon, k] - ud[icon, k]*Et)*inv_pgen,
                cutoff,
                rng,
            )

            if nspawn == 0.0:
                continue

            if isinit:
                spawns[0, icon] += nspawn
            elif nspawn < 0.0:
                spawns[1, icon] += nspawn
                spawns[2, icon] += 1.0
            else:
                spawns[3, icon] += nspawn
                spawns[4, icon] += 1.0

    return spawns


@njit
def paired_cdot_mapped(
        h0: np.array,
        ud: np.array,
        p0: np.array,
        p1: np.array,
        addop: float,
        Et: float,
        dt: float,
        cutoff: float,
        nareps: int,
        ninit: float,
        ncons: np.array,
        icons: np.array,
        rng: np.random.Generator,
        imin: int,
        imax: int,
    ) -> np.array:
    r''' TODO: Docstring here.
    '''
    if ninit > 0.0:
        # 0: unrestricted spawns
        # 1: (-) non-initiator spawns
        # 2: (-) non-initiator counts
        # 3: (+) non-initiator spawns
        # 4: (+) non-initiator counts
        spawns = np.zeros((5, p0.shape[0]), dtype=float)
    else:
        spawns = np.zeros((1, p0.shape[0]), dtype=float)

    ndets = p0.shape[0]

    for k in range(imin, imax + 1):
        nw = p0[k, 0] + addop*p1[k, 0]
        isinit = abs(nw) >= ninit

        if nw == 0.0:
            continue

        # Diagonal term.
        spawns[0, k] += dt*(h0[k, k] - ud[k, k]*Et)*nw

        # Then randomly select i from those remaining.
        sign = np.sign(nw)

        inv_pgen, ricons = get_mapped_connections(
            k,
            nw,
            nareps,
            ncons,
            icons,
            rng,
        )

        for icon in ricons:
            nspawn = round_spawn(
                sign*dt*(h0[icon, k] - ud[icon, k]*Et)*inv_pgen,
                cutoff,
                rng,
            )

            if nspawn == 0.0:
                continue

            if isinit:
                spawns[0, icon] += nspawn
            elif nspawn < 0.0:
                spawns[1, icon] += nspawn
                spawns[2, icon] += 1.0
            else:
                spawns[3, icon] += nspawn
                spawns[4, icon] += 1.0

    return spawns
