#!/usr/bin/env python

import numpy as np
import numba

@numba.njit
def bitarray_pg(s1,s2,pg):
    """
    Finds the cross product of two point group symmetries
    with the point group of the total symmetry using bitwise operators.
    """
    s12 = np.bitwise_xor(s1,s2)
    s = np.bitwise_and(s12,pg)
    return s
