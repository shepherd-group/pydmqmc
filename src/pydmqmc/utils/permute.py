"""Functions for calculating possible permutations."""

import numpy as np

from numpy.typing import NDArray as Array


def generate_ijab_symmetries_array(
    i: int, j: int, a: int, b: int, eight_fold: bool = True, rhf: bool = True
) -> Array:
    """
    Generate an array of valid symmetry permutations of the orbital indicies.

    Assume physics notation.

    Parameters
    ----------
    i, j, a, b
        Orbital indexes.
    eight_fold
        Whether or not the system is 8-fold spatially symmetric.
    rhf
        Use restricted Hartree--Fock; indicates spin symmetry.

    Returns
    -------
    array
        All valid i, j, a, b permutations
    """
    # (???) are these error checks accurate?
    # Some error checking should be in place if this will be public.
    # It makes sense to have this function available outside of a system
    # though it may make more sense in the utils submodule.
    if a > i:
        raise ValueError(f"Index a {a} cannot excede i {i}.")
    if b > j:
        raise ValueError(f"Index b {b} cannot excede j {j}.")

    if rhf:
        i, j, a, b = i + i, j + j, a + a, b + b
    uhf = not rhf
    nspat = int(4 - 3 * uhf)
    nspin = int(4 - 3 * uhf + 4 * eight_fold - 3 * eight_fold * uhf)

    SS = [
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ]

    FF = [
        [i, j, a, b],
        [a, b, i, j],
        [j, i, b, a],
        [b, a, j, i],
    ]

    EF = [
        [i, j, a, b],
        [j, i, b, a],
        [a, b, i, j],
        [b, a, j, i],
        [a, j, i, b],
        [b, i, j, a],
        [i, b, a, j],
        [j, a, b, i],
    ]

    if eight_fold:
        P = np.repeat(EF, nspat, axis=0)
    else:
        P = np.repeat(FF, nspat, axis=0)

    if rhf:
        P += np.tile(SS, (nspin, 1))

    return P
