"""
Point group symmetry functions based off those in the HANDE code.

The forms and functions have been condensed for use in pydmqmc.

See:
    https://github.com/hande-qmc/hande/blob/main/src/pg_symmetry.f90
    https://doi.org/10.1021/acs.jctc.8b01217
"""

import numpy as np

from numpy.typing import ArrayLike


def pg_sym_cross_prod(sym1: int, sym2: int, mask: int) -> int:
    """
    Cross product of two point group symmetries.

    Parameters
    ----------
    sym1, sym2 : int
        The symmetries to find the direct product of. These should be
        the bit representations of the point group symmetries' irreducible
        representations.
    mask : int
        Mask for extracting the point group symmetry from
        the full symmetry representation.

    Returns
    -------
    int
        Direct product of the symmetries' irreducible representations.

    See Also
    --------
    pg_sym_conj:
        Conjugate of a point group symmetry.
    orb_sym:
        Construct the symmetry of a determinant from its orbitals.
    pydmqmc.systems.Integral.random_bitarray_symspace,
    pydmqmc.systems.Integral.generate_renorm_excitation,
    pydmqmc.systems.Integral.print_symmetry_table:
        Example uses of this function.
    """
    return np.bitwise_and(np.bitwise_xor(sym1, sym2), mask)


def orb_sym(orb_syms: ArrayLike, mask: int) -> int:
    """
    Construct the symmetry of a determinant from its orbitals.

    Parameters
    ----------
    orb_syms : array-like of int
        List of the symmetries for the orbitals in the determinant. These should be
        the bit representations of the point group symmetries' irreducible
        representations.
    mask : int
        Mask for extracting the point group symmetry from
        the full symmetry representation.

    Returns
    -------
    int
        The symmetry of the determinant.

    See Also
    --------
    pg_sym_cross_prod:
        Cross product of two point group symmetries.
    pg_sym_conj:
        Conjugate of a point group symmetry.
    pydmqmc.systems.Integral.random_bitarray_symspace:
        Example use of this function.
    """
    sym1 = 0
    for sym2 in orb_syms:
        sym1 = pg_sym_cross_prod(sym1, sym2, mask)
    return int(sym1)


def pg_sym_conj(sym: int, mask: int) -> int:
    """
    Conjugate of a point group symmetry.

    Parameters
    ----------
    sym : int
        The symmetry to be conjugated. This should be the bit
        representation of the point group symmetry's irreducible
        representation.
    mask : int
        Mask for extracting the point group symmetry from
        the full symmetry representation.

    Returns
    -------
    int
        The symmetry conjugate of the given symmetry.

    See Also
    --------
    pg_sym_cross_prod:
        Cross product of two point group symmetries.
    orb_sym:
        Construct the symmetry of a determinant from its orbitals.
    pydmqmc.systems.Integral.generate_renorm_excitation:
        Example use of this function.
    """
    return np.bitwise_or(np.bitwise_and(sym, mask), 0)
