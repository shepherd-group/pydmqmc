import numpy as np

from numpy.typing import NDArray as Array


def bitarray_to_integer(ba: Array) -> int:
    """Generate the integer representation of a given bitarray.

    Parameters
    ----------
    ba
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
    return np.frombuffer(ba.encode(), dtype='S1').astype(int)[::-1]


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
    return np.array_split(integer_to_bitarray(label, 2*norb), 2)
