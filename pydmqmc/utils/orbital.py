import numpy as np

#from pydmqmc.systems import Integral

def get_nvirt_ms_sym(sys, occ):
    """TODO update docstring

    Takes in the sys class object and an occupied orbital array
    and returns relevant information for the virtual orbitals.

    In:
        sys: Class object of our system
        occ: The occupied orbitals for the current determinant
    Out:
        unocc: The unoccupied orbital indexes in our current determinant
        virt_ms: The corresponding spins of unocc
        virt_sym: The corresponding symmetries of unocc
        nvirt: Counts of the number of unoccupied orbitals in each
            spin-symmetry indexed by spin and symmetry.

    (Be warned, there is a always an empty array corresponding to ms = 0)
    """
    unocc = sys.orbitals[np.isin(sys.orbitals, occ, invert=True)]
    virt_ms = sys.ms[unocc]
    virt_sym = sys.orbital_pg_symmetry[unocc]

    nvirt = np.zeros((3, sys.max_symmetry))
    for ms, sym in zip(virt_ms, virt_sym):
        nvirt[ms, sym] += 1

    return unocc, virt_ms, virt_sym, nvirt
