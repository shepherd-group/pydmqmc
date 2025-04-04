"""Slater-Condon rules for generating Hamiltonian elements.

Warnings
--------
Docstrings should be checked for scientific accuracy.
"""

import numpy as np
from numpy.typing import NDArray as Array

def sc0(ba: Array, sys) -> float:
    """Slater-Condon rule for ground state electrons.

    Math from Szabo and Ostlund:
    <\\Psi_{0}|H|\\Psi_{0}> = \\sum_{a} <a|h|a> 
                            + 1/2 \\sum_{a,b} <ab|ab> - <ab|ba>

    Parameters
    ----------
    ba
        bitarray
    """
    E = sys.h0e
    E += np.einsum('a,aa->',ba,sys.h1e)
    E += 0.5*np.einsum('a,b,abab->',ba,ba,sys.h2e)
    E -= 0.5*np.einsum('a,b,abba->',ba,ba,sys.h2e)
    return E

def sc1(ba, a, r, perms, sys):
    """Slater-Condon rule for singly excited states.

    Math from Szabo and Ostlund:
    <\\Psi_{0}|H|\\Psi_{a}^{r}> = <a|h|r> + \\sum_{b} <ab|rb> - <ab|br>
    """
    ba1 = np.copy(ba)
    ba1[[a,r]] = 0
    E = sys.h1e[a,r]
    E += np.einsum('b,bb->', ba1,sys.h2e[a,:,r,:])
    E -= np.einsum('b,bb->', ba1,sys.h2e[a,:,:,r])
    E *= (1.0 - 2.0*(perms % 2))
    return E

def sc2(a, b, r, s, perms, sys):
    """Slater-Condon rule for doubly excited states.

    Math from Szabo and Ostlund:
    <\\Psi_{0}|H|\\Psi_{ab}^{rs}> = <ab|rs> - <ab|sr>
    """
    E  = sys.h2e[a,b,r,s]
    E -= sys.h2e[a,b,s,r]
    E *= (1.0 - 2.0*(perms % 2))
    return E

def get_hij(b1, b2, sys, tol=1E-16):
    nex, abrs, perms = get_ex_info(b1,b2,sys.nel)
    if nex == 0:
        E = sc0(b1,sys)
    elif nex == 1:
        a, _, r, _ = abrs
        E = sc1(b1,a,r,perms,sys)
    elif nex == 2:
        a, b, r, s = abrs
        E = sc2(a,b,r,s,perms,sys)
    else:
        E = 0.0
    E *= int(abs(E) > tol)
    return E