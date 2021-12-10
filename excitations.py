#!/usr/bin/env python

import numpy as np
from utilities import orb_sym as csym
from utilities import cross_prod_pg_sym as xpsym
from utilities import get_nvirt_ms_sym
from utilities import conj_sym
from utilities import sc1
from utilities import sc2
from utilities import get_single_perm
from utilities import get_double_perm

def random_bitarry_symspace(norb,na,nb,pg_mask,symmetry,orbsym):
    r'''
    Generate a random determinant from the full space of all determinants.

    In:
        na: The number of alpha electrons
        nb: The number of beta electrons
        norb: The number of orbitals
    Out:
        ba: The bitarray of the determinant
    '''
    occa = np.random.choice(int(norb/2), na, replace=False)
    syma = csym(orbsym[2*occa], pg_mask)
    occb = np.random.choice(int(norb/2), nb, replace=False)
    symb = csym(orbsym[2*occb+1], pg_mask)

    if not (xpsym(symb, syma, pg_mask) == symmetry):
        return random_bitarry_symspace(sys)

    ba = np.zeros(norb, dtype=int)
    ba[2*occa] = 1
    ba[2*occb+1] = 1
    return ba

def calculate_psingle_pdouble(orbs,ms,orbsym,maxsym,pg_mask,ba_ref):
    r'''
    Calculate the single and double excitation probabilities.
    We assume that the reference state is a good candidate for the
    number of single and double excitations possible for a given system.

    All we are doing here is counting the number of excitations
    which are allowed by symmetry and spin conservation for each electron.

    We do not do anything like consider the value of Hamiltonian element
    between those excitations ect.

    In/Out:
        sys: A system object class, upon return it will contain the
            self.psingle and self.pdouble object
    '''
    occ = orbs[ba_ref == 1]
    unocc, virt_ms, virt_sym, nvirt = get_nvirt_ms_sym(orbs,ms,
                                                        orbsym,maxsym,occ)

    occ_ms = ms[occ]
    occ_sym = orbsym[occ]
    nsingle = np.sum(nvirt[(occ_ms,occ_sym)])

    ndouble = 0
    for i,(ims,isym) in enumerate(zip(occ_ms,occ_sym)):
        for jms,jsym in zip(occ_ms[i+1:],occ_sym[i+1:]):
            for syma in np.arange(maxsym):
                symb = xpsym(isym, jsym, pg_mask)
                symb = xpsym(syma, symb, pg_mask)
                if syma == symb and ims == jms:
                    ndouble += nvirt[ims,syma]*(nvirt[jms,symb]-1)/2
                elif syma == symb:
                    ndouble += nvirt[ims,syma]*nvirt[jms,symb]
                elif syma < symb:
                    ndouble += nvirt[ims,syma]*nvirt[jms,symb]
                    if ims != jms:
                        ndouble += nvirt[jms,syma]*nvirt[ims,symb]

    psingle = nsingle/(nsingle + ndouble)
    pdouble = ndouble/(nsingle + ndouble)
    return psingle, pdouble

def generate_renorm_excitation(sys,ba):
    r'''
    Generate an excitation from the current bitarray.
    '''
    occ = sys.orbs[ba == 1]
    occ_ms, occ_sym = sys.ms[occ], sys.orbsym[occ]
    unocc, virt_ms, virt_sym, nvirt = get_nvirt_ms_sym(sys,occ)

    if np.random.random() < sys.psingle:
        nsources = np.count_nonzero(nvirt[(occ_ms,occ_sym)] > 0)
        if nsources > 0:
            nex = 1
            while True:
                i = np.random.choice(occ)
                ims, isym = sys.ms[i], sys.orbsym[i]
                if nvirt[ims,isym] > 0:
                    break
            allowed = (ims == virt_ms) & (isym == virt_sym)
            a = np.random.choice(unocc[allowed])
            pgen = sys.psingle/(nsources*nvirt[ims,isym])
            ba2, perms = get_single_perm(ba,i,a,sys.nel)
            hij = sc1(ba,i,a,perms,sys)
        else:
            pgen, nex, hij, ba2 = 1.0, None, 0.0, None
    else:
        allowed_excit = False
        i, j = np.random.choice(occ, 2, replace=False)
        ijsym = conj_sym(xpsym(sys.orbsym[i], sys.orbsym[j], sys.pg_mask), sys)
        ijms = sys.ms[i] + sys.ms[j]
        pgen_ij = 2.0/(sys.nel*(sys.nel-1))

        if ijms == -2:
            for syma in range(sys.maxsym):
                symb = conj_sym(xpsym(syma, ijsym, sys.pg_mask) ,sys)
                bool1 = nvirt[-1,syma] > 0
                bool2 = nvirt[-1,symb] > 1
                bool3 = nvirt[-1,symb] == 1 and syma != symb
                if bool1 and (bool2 or bool3):
                    allowed_excit = True
                    break
            fac = 2
            shift = 1
        elif ijms == 0:
            for syma in range(sys.maxsym):
                symb = conj_sym(xpsym(syma, ijsym, sys.pg_mask) ,sys)
                bool1 = nvirt[-1,syma] > 0 and nvirt[1,symb] > 0
                bool2 = nvirt[1,syma] > 0 and nvirt[-1,symb] > 0
                if bool1 or bool2:
                    allowed_excit = True
                    break
            fac = 1
            shift = 0
        elif ijms == 2:
            for syma in range(sys.maxsym):
                symb = conj_sym(xpsym(syma, ijsym, sys.pg_mask) ,sys)
                bool1 = nvirt[1,syma] > 0
                bool2 = nvirt[1,symb] > 1
                bool3 = nvirt[1,symb] == 1 and syma != symb
                if bool1 and (bool2 or bool3):
                    allowed_excit = True
                    break
            fac = 2
            shift = 0

        if allowed_excit:
            nex = 2
            while True:
                a = np.random.choice(unocc[unocc % fac == shift])
                imsb = ijms - sys.ms[a]
                isymb = conj_sym(xpsym(ijsym, sys.orbsym[a], sys.pg_mask), sys)
                bool1 = nvirt[imsb,isymb] > 1
                bool2 = nvirt[imsb,isymb] == 1
                bool3 = isymb != sys.orbsym[a] or ijms == 0
                if bool1 or (bool2 and bool3):
                    allowed = (imsb == virt_ms) & (isymb == virt_sym)
                    b = np.random.choice(unocc[allowed])
                    if b != a:
                        break
            if a > b:
                a, b = b, a

            imsa = sys.ms[a]
            imsb = sys.ms[b]
            if ijms == -2:
                n_aij = int(sys.norb/2) - sys.nb
                for syma in range(sys.maxsym):
                    symb = conj_sym(xpsym(syma, ijsym, sys.pg_mask) ,sys)
                    bool1 = nvirt[-1,symb] == 0
                    bool2 = syma == symb and nvirt[-1,symb] == 1
                    if bool1 or bool2:
                        n_aij -= nvirt[-1,syma]
                if sys.orbsym[a] == sys.orbsym[b]:
                    p_aijb = 1/(nvirt[imsa,sys.orbsym[a]]-1)
                    p_bija = 1/(nvirt[imsb,sys.orbsym[b]]-1)
                else:
                    p_aijb = 1/(nvirt[imsa,sys.orbsym[a]])
                    p_bija = 1/(nvirt[imsb,sys.orbsym[b]])
            elif ijms == 0:
                n_aij = sys.norb - sys.nel
                for syma in range(sys.maxsym):
                    symb = conj_sym(xpsym(syma, ijsym, sys.pg_mask) ,sys)
                    bool1 = nvirt[-1,symb] == 0
                    bool2 = nvirt[1,symb] == 0
                    if bool1:
                        n_aij -= nvirt[1,syma]
                    if bool2:
                        n_aij -= nvirt[-1,syma]
                p_aijb = 1/(nvirt[imsa,sys.orbsym[a]])
                p_bija = 1/(nvirt[imsb,sys.orbsym[b]])
            elif ijms == 2:
                n_aij = int(sys.norb/2) - sys.na
                for syma in range(sys.maxsym):
                    symb = conj_sym(xpsym(syma, ijsym, sys.pg_mask) ,sys)
                    bool1 = nvirt[1,symb] == 0
                    bool2 = syma == symb and nvirt[1,symb] == 1
                    if bool1 or bool2:
                        n_aij -= nvirt[1,syma]
                if sys.orbsym[a] == sys.orbsym[b]:
                    p_aijb = 1/(nvirt[imsa,sys.orbsym[a]]-1)
                    p_bija = 1/(nvirt[imsb,sys.orbsym[b]]-1)
                else:
                    p_aijb = 1/(nvirt[imsa,sys.orbsym[a]])
                    p_bija = 1/(nvirt[imsb,sys.orbsym[b]])

            pgen = sys.pdouble*pgen_ij*(1/n_aij)*(p_bija+p_aijb)
            ba2, perms = get_double_perm(ba,i,j,a,b,sys.nel)
            hij = sc2(i,j,a,b,perms,sys)
        else:
            pgen, nex, hij, ba2 = 1.0, None, 0.0, None

    return pgen, hij, nex, ba2

