"""Functions for manipulating bitarrays.

Notes
-----
A `bitarray` is shorthand for an array of 1's and 0's.
More traditionally referred to as "bitstrings,"
these are used to represent Slater determinants.
We work with memory-hungry numpy arrays as it makes
down-stream work easier.
"""

import numba
import numpy as np

from numpy.typing import NDArray as Array


def bitarray_to_integer(ba: Array) -> int:
    """Generate the integer representation of a given bitarray.

    Parameters
    ----------
    ba :
        The bitarray we wish to convert to an integer.

    Returns
    -------
    int
        The integer form of the bitarray.

    Notes
    -----
    A `bitarray` is shorthand for an array of 1's and 0's.
    More traditionally referred to as "bitstrings,"
    these are used to represent Slater determinants.
    """
    return ba.dot(1 << np.arange(ba.size))


def integer_to_bitarray(iba: int, norb: int) -> Array:
    """Generate a bitarray given an integer representation.

    Parameters
    ----------
    iba
        The integer representation of the bitarray.
    norb
        The total number of spin-orbitals for the system.

    Returns
    -------
    array
        The array of integer bits representing the determinant.

    Notes
    -----
    A `bitarray` is shorthand for an array of 1's and 0's.
    More traditionally referred to as "bitstrings,"
    these are used to represent Slater determinants.
    """
    ba = np.binary_repr(iba, width=norb)
    return np.frombuffer(ba.encode(), dtype="S1").astype(int)[::-1]


def concate_bitarrays_to_label(ba1: Array, ba2: Array) -> int:
    """Concatenate two bitarrays and then convert to integer representation.

    Parameters
    ----------
    ba1
        The first bitarray of the state.
    ba2
        The second bitarray of the state.

    Returns
    -------
    int
        The integer representation of the concatenation of the unique
        bitarray labels for the state

    Notes
    -----
    A `bitarray` is shorthand for an array of 1's and 0's.
    More traditionally referred to as "bitstrings,"
    these are used to represent Slater determinants.
    """
    return bitarray_to_integer(np.concatenate((ba1, ba2), axis=None))


def extract_bitarrays_from_label(label: int, norb: int) -> Array:
    """Extract the bitarrays used to generate a given integer state label.

    Parameters
    ----------
    label
        The integer label of the state.
    norb
        The number of orbitals in the system.

    Returns
    -------
    array
        An array of the two bitarrays used to generate the integer
        representation of the given state

    Notes
    -----
    A `bitarray` is shorthand for an array of 1's and 0's.
    More traditionally referred to as "bitstrings,"
    these are used to represent Slater determinants.
    """
    return np.array_split(integer_to_bitarray(label, 2 * norb), 2)


def get_nex(b1: Array, b2: Array) -> int:
    """Return the number of excitations between two states."""
    return int(np.count_nonzero(b1 != b2) / 2)


def get_occ(b1: Array) -> int:
    return np.nonzero(b1 != 0)[0]


def get_iocc(b1, orb1):
    return np.nonzero(b1 == orb1)[0][0]


def get_single_perm(b1, a, r, nel):
    occ1 = get_occ(b1)
    b2 = np.copy(b1)
    b2[a] = 0
    b2[r] = 1
    occ2 = get_occ(b2)
    perms = int(2 * nel)
    perms -= get_iocc(occ1, a)
    perms -= get_iocc(occ2, r)
    return b2, perms


def get_double_perm(b1, a, b, r, s, nel):
    occ1 = get_occ(b1)
    b2 = np.copy(b1)
    b2[a] = 0
    b2[b] = 0
    b2[r] = 1
    b2[s] = 1
    occ2 = get_occ(b2)
    perms = int(4 * nel) - 2
    perms -= get_iocc(occ1, a)
    perms -= get_iocc(occ1, b)
    perms -= get_iocc(occ2, r)
    perms -= get_iocc(occ2, s)
    return b2, perms


def get_ex_info(b1, b2, nel):
    occ1 = get_occ(b1)
    occ2 = get_occ(b2)
    nex = get_nex(b1, b2)
    excit1 = get_occ(np.logical_and(b2 != b1, b1 != 0))
    excit2 = get_occ(np.logical_and(b1 != b2, b2 != 0))

    perms = 0
    a, b, r, s = [None] * 4
    if nex == 1:
        a = excit1[0]
        r = excit2[0]
        perms += int(2 * nel)
        perms -= get_iocc(occ1, a)
        perms -= get_iocc(occ2, r)
    elif nex == 2:
        a = excit1[0]
        b = excit1[1]
        r = excit2[0]
        s = excit2[1]
        perms += int(4 * nel) - 2
        perms -= get_iocc(occ1, a)
        perms -= get_iocc(occ1, b)
        perms -= get_iocc(occ2, r)
        perms -= get_iocc(occ2, s)

    return nex, [a, b, r, s], perms


@numba.njit
def bitarray_pg(s1, s2, pg):
    """
    Cross product of two point group symmetries with the total point group.

    Use bitwise operations to find the cross product of two point group
    symmetries with the point group of the total symmetry.
    """
    s12 = np.bitwise_xor(s1, s2)
    s = np.bitwise_and(s12, pg)
    return s
