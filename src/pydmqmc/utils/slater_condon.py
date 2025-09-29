"""Slater-Condon rules for generating Hamiltonian elements.

Warnings
--------
Docstrings should be checked for scientific accuracy.
"""

import numpy as np
from numpy.typing import ArrayLike


def sc0(ba: ArrayLike, sys: "Integral") -> float:
    r"""
    Hamiltonian matrix element between a state and itself.

    Parameters
    ----------
    ba : array_like
        Bitarray for the state.
    sys : Integral
        System object with integral information.

    Returns
    -------
    float
        Hamiltonian matrix element.

    Notes
    -----
    Math from Szabo and Ostlund (Table 2.5, Case 1) [1]_:

    .. math::

        <\Psi_{0}|H|\Psi_{0}> = \sum_{a} <a|h|a>
                            + 1/2 \sum_{a,b} <ab|ab> - <ab|ba>

    where :math:`a` and :math:`b` are occupied orbitals.
    Note that this equation is written in physicists' notation.

    References
    ----------
    .. [1] Attila Szabo and Neil S. Ostlund, "Modern Quantum Chemistry:
        Introduction to Advanced Electronic Structure Theory," Dover Books
        on Chemistry, 1996
    """
    E = sys.h0e
    E += np.einsum("a,aa->", ba, sys.h1e)
    E += 0.5 * np.einsum("a,b,abab->", ba, ba, sys.h2e)
    E -= 0.5 * np.einsum("a,b,abba->", ba, ba, sys.h2e)
    return E


def sc1(ba: ArrayLike, a: int, r: int, perms: int, sys: "Integral") -> float:
    r"""
    Hamiltonian matrix element between singly excited states.

    Parameters
    ----------
    ba : array_like
        Bitarray for the state.
    a : int
        Index of the occupied orbital to be vacated.
    r : int
        Index of the unoccupied orbital to be filled.
    perms : int
        Number of permutations associated with the excitation.
    sys : Integral
        System object with integral information.

    Returns
    -------
    float
        Hamiltonian matrix element.

    Notes
    -----
    Math from Szabo and Ostlund (Table 2.5, Case 2) [1]_:

    .. math::
        <\Psi_{0}|H|\Psi_{a}^{r}> = <a|h|r> + \sum_{b} <ab|rb> - <ab|br>

    where :math:`a` and :math:`b` are occupied orbitals
    and :math:`r` is unoccpied. Note that this equation is written in
    physicists' notation.

    References
    ----------
    .. [1] Attila Szabo and Neil S. Ostlund, "Modern Quantum Chemistry:
        Introduction to Advanced Electronic Structure Theory," Dover Books
        on Chemistry, 1996
    """
    ba1 = np.copy(ba)
    ba1[[a, r]] = 0
    E = sys.h1e[a, r]
    E += np.einsum("b,bb->", ba1, sys.h2e[a, :, r, :])
    E -= np.einsum("b,bb->", ba1, sys.h2e[a, :, :, r])
    E *= 1.0 - 2.0 * (perms % 2)
    return E


def sc2(a: int, b: int, r: int, s: int, perms: int, sys: "Integral") -> float:
    r"""
    Hamiltonian matrix element between doubly excited states.

    Parameters
    ----------
    a, b : int
        Indices of the occupied orbitals to be vacated.
    r, s : int
        Indices of the unoccupied orbitals to be filled.
    perms : int
        Number of permutations associated with the excitation.
    sys : Integral
        System object with integral information.

    Returns
    -------
    float
        Hamiltonian matrix element.

    Notes
    -----
    Math from Szabo and Ostlund (Table 2.5, Case 3) [1]_:

    .. math:: <\Psi_{0}|H|\Psi_{ab}^{rs}> = <ab|rs> - <ab|sr>

    where :math:`a` and :math:`b` are occupied orbitals
    and :math:`r` and :math:`s` are unoccpied orbitals.
    Note that this equation is written in physicists' notation.

    The bitarray is not needed as input since the orbitals
    involved are fully specified by the indices.

    References
    ----------
    .. [1] Attila Szabo and Neil S. Ostlund, "Modern Quantum Chemistry:
        Introduction to Advanced Electronic Structure Theory," Dover Books
        on Chemistry, 1996
    """
    E = sys.h2e[a, b, r, s]
    E -= sys.h2e[a, b, s, r]
    E *= 1.0 - 2.0 * (perms % 2)
    return E
