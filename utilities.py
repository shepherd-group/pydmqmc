#!/usr/bin/env python

import numpy as np
from sympy.utilities.iterables import multiset_permutations as gen_perm_set

def generate_bit_arrays(sys, use_symmetry_block=True):
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

    r'''
    Symmetry checking using bitwise operations in
    this form is original to the HANDE code base.
    The form and function is condensed for
    the purpose of this code.
    https://github.com/hande-qmc/hande/blob/main/src/pg_symmetry.f90
    https://doi.org/10.1021/acs.jctc.8b01217
    '''
    def cross_prod_pg_sym(sym1,sym2):
        return np.bitwise_and(np.bitwise_xor(sym1,sym2),sys.pg_mask)
    def orb_sym(orb_syms):
        sym1 = 0
        for sym2 in orb_syms:
            sym1 = cross_prod_pg_sym(sym1,sym2)
        return sym1

    aba = np.zeros(int(sys.norb/2),dtype=int)
    aba[:sys.na] = 1
    alpha_bas = list(gen_perm_set(aba))[::-1]

    bba = np.zeros(int(sys.norb/2),dtype=int)
    bba[:sys.nb] = 1
    beta_bas = list(gen_perm_set(bba))[::-1]

    HS_est = len(beta_bas)*len(alpha_bas)
    print('\n Upper bound on hilbert space: {:<22}'.format(HS_est))
    bas = []
    for bba in beta_bas:
        bind = 2*np.nonzero(bba)[0] + 1
        boccsym = sys.orbsym[bind]
        bsym = orb_sym(boccsym)

        for aba in alpha_bas:
            aind = 2*np.nonzero(aba)[0]
            aoccsym = sys.orbsym[aind]
            asym = orb_sym(aoccsym)

            sym = cross_prod_pg_sym(bsym,asym)
            ba = np.zeros(sys.norb,dtype=int)
            ba[np.arange(0,sys.norb,2)] = aba
            ba[np.arange(1,sys.norb,2)] = bba

            if not use_symmetry_block:
                bas.append(ba)
            elif sym == sys.symmetry:
                bas.append(ba)

    HS_est = len(bas)
    print(' Actual size of the hilbert space: {:<22}\n'.format(HS_est))
    sys.ndets = HS_est
    sys.bitarrays = np.array(bas)

def get_nex(b1,b2):
    return int(np.count_nonzero(b1!=b2)/2)

def get_occ(b1):
    return np.nonzero(b1!=0)[0]

def get_ex_info(b1,b2,nel):

    def get_iocc(oocc,exs):
        return np.argwhere(oocc == abrs[exs])[0][0]

    occ1    = get_occ(b1)
    occ2    = get_occ(b2)
    occ     = get_occ(np.logical_and(b1==b2,b1!=0))
    nex     = get_nex(b1,b2)
    excit   = get_occ(b1!=b2)
    excit1  = get_occ(np.logical_and(b2!=b1,b1!=0))
    excit2  = get_occ(np.logical_and(b1!=b2,b2!=0))

    perms = 0
    abrs = {'a':None,'b':None,'r':None,'s':None}
    if nex == 1:
        abrs['a'] = excit1[0]
        abrs['r'] = excit2[0]
        perms += int(2*nel)
        perms -= get_iocc(occ1,'a')
        perms -= get_iocc(occ2,'r')
    elif nex == 2:
        abrs['a'] = excit1[0]
        abrs['b'] = excit1[1]
        abrs['r'] = excit2[0]
        abrs['s'] = excit2[1]
        perms += int(4*nel) - 2
        perms -= get_iocc(occ1,'a')
        perms -= get_iocc(occ1,'b')
        perms -= get_iocc(occ2,'r')
        perms -= get_iocc(occ2,'s')

    return nex, occ, abrs, perms

def sc0(ba,occ,sys):
    '''
    Math from Szabo and Ostlund:
    <\Psi_{0}|H|\Psi_{0}> = 
          \sum_{a}^{N} <a|h|a> + 1/2 \sum_{a,b} <ab|ab> - <ab|ba>
    '''
    E = sys.h0e
    for a in occ:
        E += sys.h1e[a,a]
        for b in occ:
            E += 0.5*sys.h2e[a,b,a,b]
            E -= 0.5*sys.h2e[a,b,b,a]
    return E

def sc1(ba,occ,a,r,perms,sys):
    '''
    Math from Szabo and Ostlund:
    <\Psi_{0}|H|\Psi_{a}^{r}> = <a|h|r> + \sum_{b} <ab|rb> - <ab|br>
    '''
    E = sys.h1e[a,r]
    for b in occ:
        E += sys.h2e[a,b,r,b]
        E -= sys.h2e[a,b,b,r]
    if (perms % 2) == 1:
        E = -E
    return E

def sc2(a,b,r,s,perms,sys):
    '''
    Math from Szabo and Ostlund:
    <\Psi_{0}|H|\Psi_{ab}^{rs}> = <ab|rs> - <ab|sr>
    '''
    E  = sys.h2e[a,b,r,s]
    E -= sys.h2e[a,b,s,r]
    if (perms % 2) == 1:
        E = -E
    return E

def get_hij(b1,b2,sys,tol=1E-16):
    nex, occ, abrs, perms = get_ex_info(b1,b2,sys.nel)
    if nex == 0:
        E = sc0(b1,occ,sys)
    elif nex == 1:
        a, _, r, _ = list(abrs.values())
        E = sc1(b1,occ,a,r,perms,sys)
    elif nex == 2:
        a, b, r, s = list(abrs.values())
        E = sc2(a,b,r,s,perms,sys)
    else:
        E = 0.0

    if abs(E) > tol:
        return E
    else:
        return 0.0

