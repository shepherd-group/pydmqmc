#!/usr/bin/env python

import numpy as np
from sympy.utilities.iterables import multiset_permutations as gen_perm_set

# def bitarry_to_integer(ba):
#     r'''
#     Generate the integer representation of a given bitarray.

#     In:
#         ba: The bitarray we wish to convert to an integer
#     Out:
#         None: An integer generated from converting the binary array to
#             its integer form
#     '''
#     return ba.dot(1 << np.arange(ba.size))

# def integer_to_bitarry(iba,norb):
#     r'''
#     Generate a bitarry given the integer representation of that bitarray.

#     In:
#         iba: The integer representation of the bitarray
#         norb: The total number of spin-orbitals for a system
#     Out:
#         None: The array of integer bits representing the determinant
#     '''
#     ba = np.binary_repr(iba,width=norb)
#     return np.fromstring(ba, dtype='S1').astype(int)[::-1]

# def concate_bitarrays_to_label(ba1,ba2):
#     r'''
#     Concatenate two bitarrays and then convert the concatenation to
#     a single integer representation.

#     In:
#         ba1: The first bitarray of the state
#         ba2: The second bitarray of the state
#     Out:
#         None: The integer representation of the concatenation of the unique
#             bitarray labels for the state
#     '''
#     return bitarry_to_integer(np.concatenate((ba1,ba2),axis=None))

# def extract_bitarrays_from_label(label,norb):
#     r'''
#     Extract the two bitarrays used to generate the given integer label of
#     the state.

#     In:
#         label: The integer label of the state
#         norb: The number of orbitals in the system
#     Out:
#         None: An array of the two bitarrays used to generate the integer
#             representation of the given state
#     '''
#     return np.array_split(integer_to_bitarry(label,2*norb),2)

def cross_prod_pg_sym(sym1,sym2,mask):
    r'''
    Symmetry checking using bitwise operations in
    this form is original to the HANDE code base.
    The form and function is condensed for
    the purpose of this code.

    https://github.com/hande-qmc/hande/blob/main/src/pg_symmetry.f90
    https://doi.org/10.1021/acs.jctc.8b01217

    See orb_sym below as well.
    '''
    return np.bitwise_and(np.bitwise_xor(sym1,sym2),mask)

def orb_sym(orb_syms,mask):
    r'''
    See notes in cross_prod_pg_sym above for more information.
    '''
    sym1 = 0
    for sym2 in orb_syms:
        sym1 = cross_prod_pg_sym(sym1,sym2,mask)
    return sym1

def conj_sym(sym,sys):
    r'''
    See notes in cross_prod_pg_sym above for more information.
    '''
    return np.bitwise_or(np.bitwise_and(sym, sys.pg_mask), 0)

def get_nvirt_ms_sym(orbs,ms,orbsym,maxsym,occ):
    r'''
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
    '''
    unocc = orbs[np.isin(orbs, occ, invert=True)]
    virt_ms = ms[unocc]
    virt_sym = orbsym[unocc]
    nvirt = np.zeros((3,maxsym))
    for ms, sym in zip(virt_ms, virt_sym):
        nvirt[ms,sym] += 1
    return unocc, virt_ms, virt_sym, nvirt

def generate_bit_arrays(norb, na, nb, orbsym, pg_mask, symmetry, 
                        use_symmetry_block=True):
    r'''
    Generate all the determinants for a system class object which contains
    the information from "integral_system" used to describe a quantum
    mechanical system.

    It should be noted that my use of "bitstring" is very loose compared
    to the traditional definition.
    We simply choose to work with memory hungry numpy arrays which contain
    the corresponding "bits" as integer 1/0's as it makes work a lot easier
    down stream.
    Changing this to a more traditional implementation is easily doable 
    with numpy functions but for now this is satsifactory.
    This can be considered in the future if so desired.

    The gist of what we are doing is generating a reference bitarray
    for each of the spin channels seperately.
    We then use sympy to generate all those unique combinations of 1/0's
    given the reference.
    Then we loop through all the unique concatenations of alpha/beta bit 
    arrays and check that the point group symmetry of the total bitarray falls
    within our systems possible point group. If it does then we store
    that determinant.

    In:
        use_symmetry_block (default=False): A boolean to control whether
            we will be using the symmetry reduced point group Hamiltonian
            or whether we will generate the entire Hamiltonian containing
            all those point-group symmetries spanned by the system.
    In/Out:
        sys: The integral_system class object, upon returning it will
            contain an array with all those determinants within the provided
            symmetry(s) as numpy arrays of 1/0's. Here 1 is an occupied
            orbital and 0 is an unoccupied orbital. These are exactly
            analagous to bitstrings but I refer to this more chaotic
            version as bitarrays henceforth.
    '''

    aba = np.zeros(int(norb/2),dtype=int)
    aba[:na] = 1
    alpha_bas = list(gen_perm_set(aba))[::-1]

    bba = np.zeros(int(norb/2),dtype=int)
    bba[:nb] = 1
    beta_bas = list(gen_perm_set(bba))[::-1]

    HS_est = len(beta_bas)*len(alpha_bas)
    print('\n Upper bound on hilbert space: {:<22}'.format(HS_est))
    bas = []
    for bba in beta_bas:
        bind = 2*np.nonzero(bba)[0] + 1
        boccsym = orbsym[bind]
        bsym = orb_sym(boccsym,pg_mask)

        for aba in alpha_bas:
            aind = 2*np.nonzero(aba)[0]
            aoccsym = orbsym[aind]
            asym = orb_sym(aoccsym,pg_mask)

            sym = cross_prod_pg_sym(bsym,asym,pg_mask)
            ba = np.zeros(norb,dtype=int)
            ba[np.arange(0,norb,2)] = aba
            ba[np.arange(1,norb,2)] = bba

            if sym == symmetry:
                bas.append(ba)
            elif not use_symmetry_block:
                bas.append(ba)

    HS_est = len(bas)
    print(' Actual size of the hilbert space: {:<22}\n'.format(HS_est))
    ndets = HS_est
    bitarrays = np.array(bas)
    return ndets, bitarrays

def get_nex(b1,b2):
    return int(np.count_nonzero(b1!=b2)/2)

def get_occ(b1):
    return np.nonzero(b1!=0)[0]

def get_iocc(b1,orb1):
    return np.nonzero(b1==orb1)[0][0]

def get_single_perm(b1,a,r,nel):
    occ1 = get_occ(b1)
    b2 = np.copy(b1)
    b2[a] = 0
    b2[r] = 1
    occ2 = get_occ(b2)
    perms = int(2*nel)
    perms -= get_iocc(occ1, a)
    perms -= get_iocc(occ2, r)
    return b2, perms

def get_double_perm(b1,a,b,r,s,nel):
    occ1 = get_occ(b1)
    b2 = np.copy(b1)
    b2[a] = 0
    b2[b] = 0
    b2[r] = 1
    b2[s] = 1
    occ2 = get_occ(b2)
    perms = int(4*nel) - 2
    perms -= get_iocc(occ1, a)
    perms -= get_iocc(occ1, b)
    perms -= get_iocc(occ2, r)
    perms -= get_iocc(occ2, s)
    return b2, perms

def get_ex_info(b1,b2,nel):

    occ1    = get_occ(b1)
    occ2    = get_occ(b2)
    nex     = get_nex(b1,b2)
    excit1  = get_occ(np.logical_and(b2!=b1,b1!=0))
    excit2  = get_occ(np.logical_and(b1!=b2,b2!=0))

    perms = 0
    a,b,r,s = [None]*4
    if nex == 1:
        a = excit1[0]
        r = excit2[0]
        perms += int(2*nel)
        perms -= get_iocc(occ1, a)
        perms -= get_iocc(occ2, r)
    elif nex == 2:
        a = excit1[0]
        b = excit1[1]
        r = excit2[0]
        s = excit2[1]
        perms += int(4*nel) - 2
        perms -= get_iocc(occ1, a)
        perms -= get_iocc(occ1, b)
        perms -= get_iocc(occ2, r)
        perms -= get_iocc(occ2, s)

    return nex, [a,b,r,s], perms

def sc0(ba,sys):
    '''
    Math from Szabo and Ostlund:
    <\Psi_{0}|H|\Psi_{0}> = \sum_{a} <a|h|a> + 1/2 \sum_{a,b} <ab|ab> - <ab|ba>
    '''
    E = sys.h0e
    E += np.einsum('a,aa->',ba,sys.h1e)
    E += 0.5*np.einsum('a,b,abab->',ba,ba,sys.h2e)
    E -= 0.5*np.einsum('a,b,abba->',ba,ba,sys.h2e)
    return E

def sc1(ba,a,r,perms,sys):
    '''
    Math from Szabo and Ostlund:
    <\Psi_{0}|H|\Psi_{a}^{r}> = <a|h|r> + \sum_{b} <ab|rb> - <ab|br>
    '''
    ba1 = np.copy(ba)
    ba1[[a,r]] = 0
    E = sys.h1e[a,r]
    E += np.einsum('b,bb->',ba1,sys.h2e[a,:,r,:])
    E -= np.einsum('b,bb->',ba1,sys.h2e[a,:,:,r])
    E *= (1.0 - 2.0*(perms % 2))
    return E

def sc2(a,b,r,s,perms,sys):
    '''
    Math from Szabo and Ostlund:
    <\Psi_{0}|H|\Psi_{ab}^{rs}> = <ab|rs> - <ab|sr>
    '''
    E  = sys.h2e[a,b,r,s]
    E -= sys.h2e[a,b,s,r]
    E *= (1.0 - 2.0*(perms % 2))
    return E

def get_hij(b1,b2,sys,tol=1E-16):
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

