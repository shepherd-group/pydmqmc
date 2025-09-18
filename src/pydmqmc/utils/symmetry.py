"""
Point group symmetry functions based off those in the HANDE code.

The forms and functions have been condensed for use in pydmqmc.

See:
    https://github.com/hande-qmc/hande/blob/main/src/pg_symmetry.f90
    https://doi.org/10.1021/acs.jctc.8b01217
"""

import numpy as np


def cross_prod_sym(sym1, sym2, mask):
    r"""Symmetry checking using bitwise operations in
    this form is original to the HANDE code base.
    The form and function is condensed for
    the purpose of this code.

    ! In:
    !    read_in: information on the symmetries of the basis functions.
    !    sym_i,sym_j: bit string representations of irreducible
    !                 representations of a point group and Lz symmetry
    ! Returns:
    !    The bit string representation of the irreducible representation
    !    formed from the direct product sym_i \cross sym_j.
    !    The Lz part of the symmetry is split off and handled separately from the
    !    rest, and then reintegrated.

    https://github.com/hande-qmc/hande/blob/main/src/pg_symmetry.f90
    https://doi.org/10.1021/acs.jctc.8b01217

    See orb_sym below as well.
    """
    return np.bitwise_and(np.bitwise_xor(sym1, sym2), mask)


def orb_sym(orb_syms, mask):
    """See notes in cross_prod_pg_sym above for more information."""
    sym1 = 0
    for sym2 in orb_syms:
        sym1 = cross_prod_sym(sym1, sym2, mask)
    return sym1


def conj_sym(sym, mask):
    r"""See notes in cross_prod_pg_sym above for more information.
    mask was originally sym and sym.pg_mask

    ! In:
    !   read_in: information on the symmetries of the basis functions.
    !   sym: the bit representation of the irrep of the pg sym including
    !        Lz in its higher bits
    ! Returns:
    !   The symmetry conjugate of the symmetry. For pg symmetry this is the same as
    !   it's Abelian, but we need to take Lz to -Lz here.
    """
    return np.bitwise_or(np.bitwise_and(sym, mask), 0)
