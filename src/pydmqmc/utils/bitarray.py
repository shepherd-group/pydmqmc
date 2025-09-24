"""
Functions for manipulating bitarrays.

Notes
-----
A `bitarray` is shorthand for an array of 1's and 0's.
More traditionally referred to as "bitstrings,"
these are used to represent Slater determinants.
We work with memory-hungry numpy arrays as it makes
down-stream work easier.
"""

import numpy as np

from numpy.typing import NDArray as Array


def bitarray_to_integer(ba: Array) -> int:
    """
    Generate the integer representation of a given bitarray.

    Parameters
    ----------
    ba : array
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
    """
    Generate a bitarray given an integer representation.

    Parameters
    ----------
    iba : int
        The integer representation of the bitarray.
    norb : int
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
    """
    Concatenate two bitarrays and then convert to integer representation.

    Parameters
    ----------
    ba1, ba2 : array
        The first and second bitarrays of the state.

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
    """
    Extract the bitarrays used to generate a given integer state label.

    Parameters
    ----------
    label : int
        The integer label of the state.
    norb : int
        The number of orbitals in the system.

    Returns
    -------
    array
        An array of the two bitarrays used to generate the integer
        representation of the given state.

    Notes
    -----
    A `bitarray` is shorthand for an array of 1's and 0's.
    More traditionally referred to as "bitstrings,"
    these are used to represent Slater determinants.
    """
    return np.array_split(integer_to_bitarray(label, 2 * norb), 2)


def get_nex(b1: Array, b2: Array) -> int:
    """
    Return the number of excitations between two states.

    Parameters
    ----------
    b1, b2 : array
        The bitarrays of the states.

    Returns
    -------
    int
        The number of excitations between the given states.
    """
    return int(np.count_nonzero(b1 != b2) / 2)  # cast out of np.int64


def get_occ(b1: Array) -> Array:
    """
    Return the number of occupied orbitals in a given bitarray.

    Parameters
    ----------
    b1 : array
        The bitarray of the state.

    Returns
    -------
    array
        The indices of the occupied orbitals in the given bitarray.
    """
    return np.nonzero(b1 != 0)[0]


def get_iocc(occ: Array, orb: int) -> int:
    """
    Return the index of an orbital within the set of occupied orbitals.

    Parameters
    ----------
    occ : array
        The indices of the occupied orbitals in the bitarray.
    orb : int
        The occupied orbital whose index we want to find.

    Returns
    -------
    int
        The index of the given orbital in the set of occupied orbitals.

    Examples
    --------
    >>> import numpy as np
    >>> barr = np.array([1, 1, 0, 1])
    >>> occ = get_occ(barr)
    >>> print(occ)
    [0 1 3]
    >>> get_iocc(occ, 3)
    2
    """
    ind = np.nonzero(occ == orb)[0]
    if ind.size == 0:
        raise ValueError("Orbital not in occupied set.")
    return int(ind[0])  # cast out of np.int64


def get_single_perm(b1: Array, a: int, r: int, nel: int) -> tuple[Array, int]:
    """
    Return new bitarray and the number of permutations for a single excitation.

    Parameters
    ----------
    b1 : array
        The bitarray of the initial state.
    a : int
        The index of the occupied orbital to be vacated.
    r : int
        The index of the unoccupied orbital to be filled.
    nel : int
        The number of electrons in the system.

    Returns
    -------
    array
        The new bitarray after the excitation.
    int
        The number of permutations associated with the excitation.
    """
    if b1[a] != 1 or b1[r] != 0:
        raise ValueError("Cannot excite from/to given orbitals.")

    b2 = np.copy(b1)
    b2[a] = 0
    b2[r] = 1

    perms = int(2 * nel)
    occ1 = get_occ(b1)
    occ2 = get_occ(b2)
    perms -= get_iocc(occ1, a)
    perms -= get_iocc(occ2, r)

    return b2, perms


def get_double_perm(
    b1: Array, a: int, b: int, r: int, s: int, nel: int
) -> tuple[Array, int]:
    """
    Return new bitarray and the number of permutations for a double excitation.

    Parameters
    ----------
    b1 : array
        The bitarray of the initial state.
    a, b : int
        The indices of the occupied orbitals to be vacated.
    r, s : int
        The indices of the unoccupied orbitals to be filled.
    nel : int
        The number of electrons in the system.

    Returns
    -------
    array
        The new bitarray after the excitation.
    int
        The number of permutations associated with the excitation.
    """
    if b1[a] != 1 or b1[b] != 1 or b1[r] != 0 or b1[s] != 0:
        raise ValueError("Cannot excite from/to given orbitals.")

    b2 = np.copy(b1)
    b2[a] = 0
    b2[b] = 0
    b2[r] = 1
    b2[s] = 1

    perms = int(4 * nel) - 2
    occ1 = get_occ(b1)
    occ2 = get_occ(b2)
    perms -= get_iocc(occ1, a)
    perms -= get_iocc(occ1, b)
    perms -= get_iocc(occ2, r)
    perms -= get_iocc(occ2, s)

    return b2, perms


def get_ex_info(b1: Array, b2: Array, nel: int) -> tuple[int, list, int]:
    """
    Return information on the excitations between two states.

    This information includes the number of excitations
    between the states (either single or double),
    the currently occupied orbitals a & b, the orbitals to be
    occupied r & s, and the number of permutations associated
    with the excitation.

    Parameters
    ----------
    b1, b2 : array
        The bitarrays of the two states.
    nel : int
        The number of electrons in the system.

    Returns
    -------
    int
        The number of excitations between the two states.
    list
        The list of orbitals involved in the excitation.
        If a single excitation, the list is [a, None, r, None].
        If a double excitation, the list is [a, b, r, s].
    int
        The number of permutations associated with the excitation.

    Raises
    ------
    RuntimeError
        If the number of excitations between the two states is not
        one or two. This ensures the function does not silently fail
        should this edge case arise.
    """
    occ1 = get_occ(b1)
    occ2 = get_occ(b2)
    nex = get_nex(b1, b2)
    excit1 = get_occ(np.logical_and(b2 != b1, b1 != 0))
    excit2 = get_occ(np.logical_and(b1 != b2, b2 != 0))

    perms = 0
    a, b, r, s = [None] * 4
    if nex == 1:
        a = int(excit1[0])
        r = int(excit2[0])
        perms += int(2 * nel)
        perms -= get_iocc(occ1, a)
        perms -= get_iocc(occ2, r)
    elif nex == 2:
        a = int(excit1[0])
        b = int(excit1[1])
        r = int(excit2[0])
        s = int(excit2[1])
        perms += int(4 * nel) - 2
        perms -= get_iocc(occ1, a)
        perms -= get_iocc(occ1, b)
        perms -= get_iocc(occ2, r)
        perms -= get_iocc(occ2, s)
    else:
        raise ValueError(
            f"Unexpected number of excitations ({nex}) between determinants."
        )

    return nex, [a, b, r, s], perms
