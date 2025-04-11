"""Slater-Condon rules for generating Hamiltonian elements.

Warnings
--------
Docstrings should be checked for scientific accuracy.
"""

import numpy as np
from numpy.typing import ArrayLike


def sc0(ba: ArrayLike, sys) -> float:
    r"""
    Slater-Condon rule for ground state electrons(???).

    Parameters
    ----------
    ba : array_like
        bitarray

    Returns
    -------
    float
        The energy(???)

    Notes
    -----
    Math from Szabo and Ostlund [1]_:

    .. math::

        <\Psi_{0}|H|\Psi_{0}> = \sum_{a} <a|h|a>
                            + 1/2 \sum_{a,b} <ab|ab> - <ab|ba>

    References
    ----------
    .. [1] Attila Szabo and Neil S. Ostlund, "Modern Quantum Chemistry:
        Introduction to Advanced Electronic Structure Theory," Dover Books
        on Chemistry, 1996
    """
    E = sys.h0e
    E += np.einsum('a,aa->', ba, sys.h1e)
    E += 0.5*np.einsum('a,b,abab->', ba, ba, sys.h2e)
    E -= 0.5*np.einsum('a,b,abba->', ba, ba, sys.h2e)
    return E


def sc1(ba, a, r, perms, sys):
    r"""
    Slater-Condon rule for singly excited states(???).

    Notes
    -----
    Math from Szabo and Ostlund [1]_:

    .. math::
        <\Psi_{0}|H|\Psi_{a}^{r}> = <a|h|r> + \sum_{b} <ab|rb> - <ab|br>

    References
    ----------
    .. [1] Attila Szabo and Neil S. Ostlund, "Modern Quantum Chemistry:
        Introduction to Advanced Electronic Structure Theory," Dover Books
        on Chemistry, 1996
    """
    ba1 = np.copy(ba)
    ba1[[a, r]] = 0
    E = sys.h1e[a, r]
    E += np.einsum('b,bb->', ba1, sys.h2e[a, :, r, :])
    E -= np.einsum('b,bb->', ba1, sys.h2e[a, :, :, r])
    E *= (1.0 - 2.0*(perms % 2))
    return E


def sc2(a, b, r, s, perms, sys):
    r"""
    Slater-Condon rule for doubly excited states(???).

    Notes
    -----
    Math from Szabo and Ostlund [1]_:

    .. math:: <\Psi_{0}|H|\Psi_{ab}^{rs}> = <ab|rs> - <ab|sr>

    References
    ----------
    .. [1] Attila Szabo and Neil S. Ostlund, "Modern Quantum Chemistry:
        Introduction to Advanced Electronic Structure Theory," Dover Books
        on Chemistry, 1996
    """
    E = sys.h2e[a, b, r, s]
    E -= sys.h2e[a, b, s, r]
    E *= (1.0 - 2.0*(perms % 2))
    return E
