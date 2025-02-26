#!/usr/bin/env python

import numpy as np
from numba import njit, prange, uint16


@njit
def get_single_excitation_orbitals(
        adet: np.array,
        bdet: np.array,
    ) -> tuple[int, int, int]:
    r''' TODO: Finish docstring.
    '''
    q = 0
    s = 0
    nex = 0

    for indx in range(adet.shape[0]):
        aorb = adet[indx]
        borb = bdet[indx]

        if aorb not in bdet:
            nex += 1
            q = aorb

        if borb not in adet:
            s = borb

        if nex >= 2 and q != 0 and s != 0:
            break

    return q, s, nex


@njit
def get_single_or_double_excitation_orbitals(
        adet: np.array,
        bdet: np.array,
    ) -> tuple[int, int, int, int, int]:
    r''' TODO: Finish docstring.
    '''
    p = 0
    q = 0
    r = 0
    s = 0
    nex = 0

    for indx in range(adet.shape[0]):
        aorb = adet[indx]
        borb = bdet[indx]

        if aorb not in bdet:
            nex += 1

            if p == 0:
                p = aorb
            else:
                q = aorb

        if borb not in adet:
            if r == 0:
                r = borb
            else:
                s = borb

        if nex >= 3 and p != 0 and q != 0 and r != 0 and s != 0:
            break

    return p, q, r, s, nex


@njit(parallel=True)
def get_transition_dipole_moments(
        M: int,
        A: np.array,
        B: np.array,
        PsiA: np.array,
        Psii: np.array,
        muz: np.array,
    ) -> np.array:
    r''' TODO: Docstring.

    Parameters
    ----------
    M : int
        The number of spatial orbitals in the system.
    A : np.array
        The occupied spin-orbitals (1-indexed) for the determinant basis A.
    B : np.array
        The occupied spin-orbitals (1-indexed) for the determinant basis B.
    PsiA : np.array
        The wavefunction for our reference state, from which we excite from.
    Psii : np.array
        The full set of wavefunctions for determinant basis B.
    muz : np.array
        The dipole integrals for the z-axis, indexed with spatial orbitals.

    Returns
    -------
    moms : np.array
        The transition moment, from the reference in the determinant basis A
        to all excitations in determinant basis B, for the z-axis.
    '''
    Nfac = 2*len(A[0]) - 2
    ndetsA = len(A)
    ndetsB = len(B)
    ndetsk = Psii.shape[0]

    moms = np.zeros(ndetsB, dtype=float)

    # WARNING: prange for i and j silently fails!
    for i in range(ndetsA):
        Ci = PsiA[i]
        aocc = A[i]

        if abs(Ci) < 1E-12:
            continue

        for j in range(ndetsB):
            bocc = B[j]

            q, s, nex = get_single_excitation_orbitals(aocc, bocc)

            if nex == 1:
                perms = (
                    Nfac
                    - np.where(q == aocc)[0][0]
                    - np.where(s == bocc)[0][0]
                )

                qspat = (q - 1)//2
                sspat = (s - 1)//2
                pfac = -1.0 if (perms % 2) else 1.0

                for k in prange(ndetsk):
                    CiCj = Ci*pfac*Psii[k, j]

                    if abs(CiCj) < 1E-12:
                        continue

                    moms[k] += muz[sspat, qspat]*CiCj

    return moms


@njit(parallel=True)
def get_determinant_dipole(adet: np.array, muij: np.array) -> float:
    r''' TODO: Docstring.
    '''
    mu = 0.0

    for i in prange(adet.shape[0]):
        iocc = (adet[i] - 1)//2
        mu += muij[iocc, iocc]

    return mu


@njit(parallel=True)
def get_determinant_basis_dipole_hamiltonians(
        shift: float,
        Di: np.array,
        Dj: np.array,
        Hi: np.array,
        Hj: np.array,
        mu: np.array,
    ) -> tuple[np.array, np.array]:
    r''' Calculate the two constant matrices used to build the time
    dependent Hamiltonian in the Slater determinant basis:
        \hat{H}(t) = \hat{H}_{0} - \hat{{\bf \mu }} \cdot {\bf d } E(t)
    Here our first constant matrix is \hat{H}_{0}, which is the CI
    Hamiltonian, and our second constant matrix is 
    \hat{{\bf \mu }} \cdot {\bf d }, which is the dipole Hamiltonian term.

    One can alternatively construct the matrices in the CI basis, where
    \hat{H}_{0} would be non-zero in the diagonal only and take on values
    for the eigen-energies. Then the dipole matrix would take on values
    for the FCI dipole moment along the diagonal, with the off-diagonal
    terms corresponding to the transition dipole moments. For more
    information see `get_transition_dipole_moments`.
    This would have the benifit of simplifying the state population
    calculations. However, for pragmatic reasons, we simply use the
    Slater basis here instead of the CI basis.

    Parameters
    ----------
    shift : float
        An energy shift applied to the \hat{H}_{0} matrix.

    TODO: Finish.

    Returns
    -------
    h0 : Array
        The first constant matrix.
    ud : Array
        The second constant matrix, which will be scaled by the electric
        field.
    '''
    Nfac = 2*len(Di[0]) - 2

    ndets_i = Di.shape[0]
    ndets_j = Dj.shape[0]
    ndets = ndets_i + ndets_j

    h0 = np.zeros((ndets, ndets), dtype=float)
    ud = np.zeros((ndets, ndets), dtype=float)

    h0[:ndets_i, :ndets_i] = Hi[:, :]
    h0[ndets_i:, ndets_i:] = Hj[:, :]

    for a in prange(ndets):
        if a < ndets_i:
            aocc = Di[a]
        else:
            aocc = Dj[a - ndets_i]

        h0[a, a] -= shift
        ud[a, a] = get_determinant_dipole(aocc, mu)

        for b in prange(ndets):
            if a == b:
                continue

            if b < ndets_i:
                bocc = Di[b]
            else:
                bocc = Dj[b - ndets_i]

            q, s, nex = get_single_excitation_orbitals(aocc, bocc)

            if nex == 1:
                perms = (
                    Nfac
                    - np.where(q == aocc)[0][0]
                    - np.where(s == bocc)[0][0]
                )

                qspat = (q - 1)//2
                sspat = (s - 1)//2
                pfac = -1.0 if (perms % 2) else 1.0

                ud[a, b] = pfac*mu[sspat, qspat]

    return h0, ud
