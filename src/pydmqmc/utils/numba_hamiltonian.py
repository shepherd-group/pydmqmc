"""Numba-compiled Hamiltonian construction helpers."""

import numpy as np
from numba import njit


@njit
def _hij(bitarray1, bitarray2, h0e, h1e, h2e, n_electrons, tolerance):
    removed = [-1, -1]
    added = [-1, -1]
    nremoved = 0
    nadded = 0

    for orbital in range(bitarray1.size):
        if bitarray1[orbital] == 1 and bitarray2[orbital] == 0:
            if nremoved < 2:
                removed[nremoved] = orbital
            nremoved += 1
        elif bitarray1[orbital] == 0 and bitarray2[orbital] == 1:
            if nadded < 2:
                added[nadded] = orbital
            nadded += 1

    if nremoved > 2 or nadded > 2:
        return 0.0

    if nremoved == 0:
        energy = h0e
        for first in range(bitarray1.size):
            if bitarray1[first] != 0:
                energy += h1e[first, first]
                for second in range(bitarray1.size):
                    if bitarray1[second] != 0:
                        energy += 0.5 * (
                            h2e[first, second, first, second]
                            - h2e[first, second, second, first]
                        )
    elif nremoved == 1:
        a = removed[0]
        r = added[0]
        removal_rank = 0
        replacement_rank = 0
        for orbital in range(bitarray1.size):
            if bitarray1[orbital] != 0:
                if orbital == a:
                    break
                removal_rank += 1
        for orbital in range(bitarray2.size):
            if bitarray2[orbital] != 0:
                if orbital == r:
                    break
                replacement_rank += 1

        energy = h1e[a, r]
        for orbital in range(bitarray1.size):
            if bitarray1[orbital] != 0 and orbital != a:
                energy += h2e[a, orbital, r, orbital]
                energy -= h2e[a, orbital, orbital, r]
        permutations = 2 * n_electrons - removal_rank - replacement_rank
        energy *= 1.0 - 2.0 * (permutations % 2)
    else:
        a = removed[0]
        b = removed[1]
        r = added[0]
        s = added[1]
        removal_rank_a = 0
        removal_rank_b = 0
        replacement_rank_r = 0
        replacement_rank_s = 0
        for orbital in range(bitarray1.size):
            if bitarray1[orbital] != 0:
                if orbital == a:
                    break
                removal_rank_a += 1
        for orbital in range(bitarray1.size):
            if bitarray1[orbital] != 0:
                if orbital == b:
                    break
                removal_rank_b += 1
        for orbital in range(bitarray2.size):
            if bitarray2[orbital] != 0:
                if orbital == r:
                    break
                replacement_rank_r += 1
        for orbital in range(bitarray2.size):
            if bitarray2[orbital] != 0:
                if orbital == s:
                    break
                replacement_rank_s += 1

        energy = h2e[a, b, r, s] - h2e[a, b, s, r]
        permutations = (
            4 * n_electrons
            - 2
            - removal_rank_a
            - removal_rank_b
            - replacement_rank_r
            - replacement_rank_s
        )
        energy *= 1.0 - 2.0 * (permutations % 2)

    if abs(energy) <= tolerance:
        return 0.0
    return energy


@njit
def generate_hamiltonian(bitarrays, h0e, h1e, h2e, n_electrons, tolerance=1e-16):
    """Generate a dense Hamiltonian from determinant bitarrays."""
    n_determinants = bitarrays.shape[0]
    hamiltonian = np.zeros((n_determinants, n_determinants), dtype=h1e.dtype)
    for first in range(n_determinants):
        for second in range(first, n_determinants):
            element = _hij(
                bitarrays[first],
                bitarrays[second],
                h0e,
                h1e,
                h2e,
                n_electrons,
                tolerance,
            )
            hamiltonian[first, second] = element
            hamiltonian[second, first] = element
    return hamiltonian
