#!/usr/bin/env python

import numpy as np
import numba

# The tolerance for Hamiltonian elements, zero otherwise
HMATEL_TOL = 1E-14

@numba.njit
def seed_random_state(seed):
    """
    Seed the numba random state as it is not shared with
    numpy implicitly
    """
    np.random.seed(seed)

@numba.njit
def bitarray_to_integer(ba,norb):
    """
    Converts an array of bits to an integer representation.
    The array of bits takes a non-traditional order in that we
    assume the left most element is the first bit.
    """
    i = 0
    for ib in range(norb):
        b = ba[ib]
        i += int(b * (2 ** ib))
    return i

@numba.njit
def integer_to_bitarray(i,norb):
    """
    Performs the reverse of `bitarray_to_integer` (see above)
    """
    b2 = np.exp2(np.flip(np.arange(norb)))
    ba = np.zeros(norb, dtype=numba.int64)
    for ip in range(norb):
        ib = -ip - 1
        p = b2[ip]
        b = int(i >= p)
        ba[ib] = b
        i -= int(b * p)
        if i == 0: break
    return ba

@numba.njit
def two_bitarray_to_integer(ba1,ba2,norb):
    """
    Convert two bitarrays to a integer which is
    a concatenated integer of the two seperate integer
    representations.
    """
    int1 = bitarray_to_integer(ba1,norb)
    int2 = bitarray_to_integer(ba2,norb)
    int_shift = np.floor(np.log10(int1))
    i = numba.int64(int1**(int_shift+1)) + int2
    return i

@numba.njit
def two_integer_to_integer(int1,int2):
    """
    Convert two integer labels into one concatenated integer label
    """
    int_shift = np.floor(np.log10(int1))
    i = numba.int64(int1**(int_shift+1)) + int2
    return i

@numba.njit
def sc0(ba,norb,h0e,h1e,h2e):
    """
    Calculate a diagonal FCI Hamiltonian element.
    See `sc0` in `utilities.py` for more information.
    """
    E = h0e
    for a in range(norb):
        abit = ba[a]
        E += abit*h1e[a,a]
        for b in range(norb):
            bbit = ba[b]
            E += 0.5*abit*bbit*h2e[a,b,a,b]
            E -= 0.5*abit*bbit*h2e[a,b,b,a]
    E *= (abs(E) > HMATEL_TOL)
    return E

@numba.njit
def sc1(ba,a,r,perms,norb,h1e,h2e):
    """
    Calculate an off-diagonal FCI Hamiltonian element.
    See `sc1` in `utilities.py` for more information.
    """
    E = h1e[a,r]
    for b in range(norb):
        bbit = ba[b]
        E += bbit * (b != a) * (b != r) * h2e[a,b,r,b]
        E -= bbit * (b != a) * (b != r) * h2e[a,b,b,r]
    E *= (1.0 - 2.0*(perms % 2))*(abs(E) > HMATEL_TOL)
    return E

@numba.njit
def sc2(a,b,r,s,perms,h2e):
    """
    Calculate an off-diagonal FCI Hamiltonian element.
    See `sc2` in `utilities.py` for more information.
    """
    E  = h2e[a,b,r,s]
    E -= h2e[a,b,s,r]
    E *= (1.0 - 2.0*(perms % 2))*(abs(E) > HMATEL_TOL)
    return E

@numba.njit
def bitarray_pg(s1,s2,pg):
    """
    Finds the cross product of two point group symmetries
    with the point group of the total symmetry using bitwise operators.
    """
    s12 = np.bitwise_xor(s1,s2)
    s = np.bitwise_and(s12,pg)
    return s

@numba.njit
def spin_channel_pg(symmetries,norb,pg):
    """
    Finds the point group symmetry of an occupation for
    a specific spin channel.
    """
    s = 0
    for i in range(norb):
        s = bitarray_pg(s, symmetries[i], pg)
    return s

@numba.njit
def sym_conj(sym,pg):
    """
    Return the conjugate symmetry, the last bitwise `or` is
    redundant but we may want that functionality eventually.
    """
    conj = np.bitwise_and(sym, pg)
    conj = np.bitwise_or(conj, 0)
    return conj

@numba.njit
def single_perms(ba,a,r,nel,norb):
    """
    Given a valid single excitation, find the permutaions to align
    the parent and child determinants
    """
    perms = nel + nel
    setbits1, setbits2 = 0, 0
    for i in range(norb):
        perms -= setbits1*(i == a)
        perms -= setbits2*(i == r)
        bitset = (ba[i] == 1)
        setbits1 += bitset
        setbits2 += bitset - (i == a)
        if setbits1 == nel and setbits2 == nel:
            break
    return perms

@numba.njit
def double_perms(ba,a,b,r,s,nel,norb):
    """
    Given a valid double excitation, find the permutations to align
    the parent and child determinants
    """
    perms = nel + nel + nel + nel - 2
    setbits1, setbits2 = 0, 0
    for i in range(norb):
        perms -= setbits1*((i == a) + (i == b))
        perms -= setbits2*((i == r) + (i == s))
        bitset = (ba[i] == 1)
        setbits1 += bitset
        setbits2 += bitset - (i == a) - (i == b) + (i == r) + (i == s)
        if setbits1 == nel and setbits2 == nel:
            break
    return perms

@numba.njit
def count_nvirt_ms_sym(ba,orbms,orbsym,norb):
    """
    Count the number of unnocupied orbitals and return the counts
    in an array indexed by the spin and symmetry of those unoccupied
    orbitals.
    """
    nvirt = np.zeros((3,8), dtype=numba.int64)
    for i in range(norb):
        ims = orbms[i]
        isym = orbsym[i]
        setbit = ba[i]
        nvirt[ims,isym] += (1 - setbit)
    return nvirt

@numba.njit
def get_occ(ba,nel,norb):
    """
    Convert a bitarray to occupied indexes and return
    """
    occ = np.zeros(nel, dtype=numba.int64)
    nfound = 0
    for i in range(norb):
        occ[nfound] = i
        nfound += ba[i]
        if nfound == nel:
            break
    return occ

@numba.njit
def get_unocc(ba,nel,norb):
    """
    Convert a bitarray to the unoccupied indexes and return
    """
    unocc = np.zeros(nel, dtype=numba.int64)
    nfound = 0
    for i in range(norb):
        unocc[nfound] = i
        nfound += 1 - ba[i]
        if nfound == nel:
            break
    return unocc

@numba.njit
def random_bitarray(norb,na,nb,pg_mask,symmetry,orbsym):
    """
    Generate a random bitarray in the point group symmetry 
    of the system with uniform probability.
    """
    nsorb = int(norb/2)
    ba = np.zeros(norb, dtype=numba.int64)
    while True:
        ba_a = np.random.choice(nsorb, na, replace=False)
        ba_a = ba_a + ba_a
        ba_b = np.random.choice(nsorb, nb, replace=False)
        ba_b = ba_b + ba_b + 1
        sym_a = spin_channel_pg(orbsym[ba_a], na, pg_mask)
        sym_b = spin_channel_pg(orbsym[ba_b], nb, pg_mask)
        if bitarray_pg(sym_a, sym_b, pg_mask) == symmetry:
            break

    ba[ba_a] = 1
    ba[ba_b] = 1
    return ba

@numba.njit(parallel=True)
def loop_random_bitarray(N,norb,na,nb,pg_mask,symmetry,orbsym):
    """
    A wrapper to loop through generation of random bitarray
    via parallelism and return.
    Unfortunately there is no way to control the RNG state on
    children processors (yet).
    """
    bas = np.zeros((N,norb))
    labels = np.zeros(N)
    for gen_step in numba.prange(N):
        rba = random_bitarray(norb,na,nb,pg_mask,symmetry,orbsym)
        bas[gen_step] = rba
        rlabel = bitarray_to_integer(rba,norb)
        labels[gen_step] = rlabel
    return bas, labels

@numba.njit
def random_excitation(ba,nel,na,nb,norb,nvirt,psingle,pdouble,
                      pg_mask,maxsym,orbms,orbsym):
    """
    Generate a random allowed (single/double) renormalized excitation
    """

    nex, a, b, r, s, perms, pgen = 0, -1, -1, -1, -1, -1, -1.0

    nvirt_ms_sym = count_nvirt_ms_sym(ba,orbms,orbsym,norb)
    single = np.random.random() < psingle
    double = not single
    single = single and np.max(nvirt_ms_sym) > 0

    if single or double:
        occ = get_occ(ba,nel,norb)
        unocc = get_unocc(ba,nel,norb)

    if single:
        nex, b, s = 1, -1, -1

        while True:
            a = np.random.choice(occ)
            ams, asym = orbms[a], orbsym[a]
            if nvirt_ms_sym[ams,asym] > 0:
                break

        while True:
            r = np.random.choice(unocc)
            allowed = (orbms[r] == ams) and (orbsym[r] == asym)
            if allowed:
                break

        nsources = 0
        for iocc in occ:
            tmp_ms = orbms[iocc]
            tmp_sym = bitarray_pg(orbsym[iocc], 0, pg_mask)
            if nvirt_ms_sym[tmp_ms, tmp_sym] > 0:
                nsources += 1

        #nsources = np.count_nonzero(nvirt_ms_sym > 0)
        pgen = psingle/(nsources*nvirt_ms_sym[ams,asym])
        perms = single_perms(ba,a,r,nel,norb)
    elif double:
        allowed, fac, shift = False, -1, 2
        a, b = np.random.choice(occ, 2, replace=False)
        absym = sym_conj(bitarray_pg(orbsym[a], orbsym[b], pg_mask), pg_mask)
        abms = orbms[a] + orbms[b]
        pgen_ab = 2.0/(nel*(nel-1))

        if abms == -2 or abms == 2:
            fac = 2
            shift = int(abms == -2)
            ispin = int(abms/2)

            for rsym in range(maxsym):
                ssym = sym_conj(bitarray_pg(rsym, absym, pg_mask), pg_mask)
                check_one = nvirt_ms_sym[ispin,rsym] > 0
                check_two_a = nvirt_ms_sym[ispin,ssym] > 1
                check_two_b = nvirt_ms_sym[ispin,ssym] == 1 and rsym != ssym
                check_two = check_two_a or check_two_b
                if check_one and check_two:
                    allowed = True
                    break
        elif abms == 0:
            fac = 1
            shift = 0

            for rsym in range(maxsym):
                ssym = sym_conj(bitarray_pg(rsym, absym, pg_mask), pg_mask)
                check_one = nvirt_ms_sym[-1,rsym] > 0
                check_one = check_one and nvirt_ms_sym[1,ssym] > 0
                check_two = nvirt_ms_sym[1,rsym] > 0
                check_two = check_two and nvirt_ms_sym[-1,ssym] > 0
                if check_one and check_two:
                    allowed = True
                    break

        if not allowed:
            return 0, -1, -1, -1, -1, -1, -1.0

        nex = 2
        while True:
            r = np.random.choice(unocc[unocc % fac == shift])
            rsym = orbsym[r]

            sms = abms - orbms[r]
            ssym = sym_conj(bitarray_pg(absym, rsym, pg_mask), pg_mask)
            check_one = nvirt_ms_sym[sms,ssym] > 1
            check_two_a = nvirt_ms_sym[sms,ssym] == 1
            check_two_b = (ssym != rsym) or (abms == 0)
            check_two = check_two_a and check_two_b
            if check_one or check_two:
                while True:
                    s = np.random.choice(unocc)
                    check_ms = sms == orbms[s]
                    check_sym = ssym == orbsym[s]
                    check_orb = r != s
                    if check_ms and check_sym and check_orb:
                        break
                break

        rms, sms = orbms[r], orbms[s]
        if abms == -2 or abms == 2:
            ispin = int(abms/2)
            n_rab = int(norb/2) - int(nb*(abms == -2)) - int(na*(abms == 2))
            for rsym in range(maxsym):
                ssym = sym_conj(bitarray_pg(rsym, absym, pg_mask), pg_mask)
                check_one = nvirt_ms_sym[ispin,ssym] == 0
                check_two = (rsym == ssym) and (nvirt_ms_sym[ispin,ssym] == 1)
                if check_one or check_two:
                    n_rab -= nvirt_ms_sym[ispin,rsym]
            if orbsym[r] == orbsym[s]:
                dcount = -1
            else:
                dcount = 0
            p_rabs = 1/(nvirt_ms_sym[rms,orbsym[r]]+dcount)
            p_sabr = 1/(nvirt_ms_sym[sms,orbsym[s]]+dcount)
        elif abms == 0:
            n_rab = norb - nel
            for rsym in range(maxsym):
                ssym = sym_conj(bitarray_pg(rsym, absym, pg_mask), pg_mask)
                n_rab -= nvirt_ms_sym[1,rsym]*(nvirt_ms_sym[-1,ssym] == 0)
                n_rab -= nvirt_ms_sym[-1,rsym]*(nvirt_ms_sym[1,ssym] == 0)
            p_rabs = 1/(nvirt_ms_sym[rms,orbsym[r]])
            p_sabr = 1/(nvirt_ms_sym[sms,orbsym[s]])

        pgen = pdouble*pgen_ab*(1/n_rab)*(p_rabs+p_sabr)
        perms = double_perms(ba,a,b,r,s,nel,norb)

    return nex, a, b, r, s, perms, pgen

@numba.njit(parallel=True)
def loop_random_excitation(N,ba,nel,na,nb,norb,nvirt,psingle,pdouble,
                           pg_mask,maxsym,orbms,orbsym):
    """
    A wrapper to loop through generation of single/double excitations
    via parallelism and return.
    Unfortunately there is no way to control the RNG state on
    children processors (yet).
    """
    bas = np.zeros((N,norb))
    labels = np.zeros(N)
    pgens = np.zeros(N)
    for gen_step in numba.prange(N):
        exd = random_excitation(ba, nel, na, nb, norb, nvirt, psingle,
                                pdouble, pg_mask, maxsym, orbms, orbsym)
        nex, a, b, r, s, perms, pgen = exd
        rba = np.zeros(norb, dtype=numba.int64)
        rba[:] = ba[:]
        rba[a] = 0
        rba[b] = 0
        rba[r] = 1
        rba[s] = 1
        rlabel = bitarray_to_integer(rba,norb)
        bas[gen_step] = rba
        labels[gen_step] = rlabel
        pgens[gen_step] = pgen
    return bas, labels, pgens

@numba.njit
def find_array_loc(array,label,nslots):
    """
    Find the position of a label within the main or spawn array
    and return it. If we don't find one we return the next open
    position.
    Also return the number of unique elements, old or new.
    """
    label_loc = -1
    if nslots == 1:
        return 0, 2

    for loc in range(nslots):
        if int(array[loc,2]) == label:
            label_loc = loc
            break

    new_nslots = nslots + (label_loc == -1)
    label_loc += new_nslots*(label_loc == -1)

    return label_loc, new_nslots

@numba.njit
def uniform_density_matrix(norb,na,nb,pg_mask,symmetry,orbsym,
                           psips_array,particles,h0e,h1e,h2e):
    """
    Generate determinants and fill them into the main array.
    """
    nslots = 1
    for gen_det in range(particles):
        det = random_bitarray(norb,na,nb,pg_mask,symmetry,orbsym)
        l1 = bitarray_to_integer(det,norb)
        label = two_bitarray_to_integer(det,det,norb)

        loc, new_nslots = find_array_loc(psips_array,label,nslots)

        if nslots == new_nslots:
            psips_array[loc,3] += 1.0
        else:
            nslots = new_nslots
            H = sc0(det,norb,h0e,h1e,h2e)
            psips_array[loc,0] = l1
            psips_array[loc,1] = l1
            psips_array[loc,2] = label
            psips_array[loc,3] = 1.0
            psips_array[loc,4] = 2*H
            psips_array[loc,5] = H

    return psips_array, nslots

@numba.njit(parallel=True)
def calculate_estimates(psips_array,nslots):
    """
    Calculate the various estimates from the density matrix.
    """
    energy = 0.0
    trace = 0.0
    total_particles = 0.0
    for loc in numba.prange(nslots):
    #for loc in range(nslots):
        energy += psips_array[loc,3]*psips_array[loc,5]
        total_particles += abs(psips_array[loc,3])
        trace += psips_array[loc,3]*(psips_array[loc,0] == psips_array[loc,1])
    return energy, trace, total_particles

@numba.njit
def do_symmetric_spawn(nex,ba,a,b,r,s,perms,pgen,norb,h1e,h2e,tau):
    """
    A routine for spawning from either
    rho_{ij} -> rho_{kj} or rho_{ij} -> rho_{ik}
    """
    if nex == 1:
        hij = sc1(ba,a,r,perms,norb,h1e,h2e)
    else:
        hij = sc2(a,b,r,s,perms,h2e)

    p = -(0.5*tau*hij)/pgen
    ps = p - int(p)
    p = int(p)
    if np.random.random() < abs(ps):
        p += (p > 0) - (p < 0)
    p = int(p)

    if p == 0:
        return 0, 0, 0.0

    ba[a] = 0
    ba[b] = (nex == 1)
    ba[r] = 1
    ba[s] = (nex == 2)
    klab = bitarray_to_integer(ba,norb)
    return p, klab, hij

@numba.njit
def symmetric_dmqmc_spawning(spawn_array,psips_array,nslots,norb,nel,na,nb,
                             nvirt,psingle,pdouble,pg_mask,maxsym,orbms,orbsym,
                             h1e,h2e,tau):
    """
    Performs the symmetric spawning step in DMQMC
    """
    tslots = 1
    for idet in range(nslots):
        ilab = int(psips_array[idet,0])
        jlab = int(psips_array[idet,1])
        print(' ilab, jlab')
        print(ilab,jlab)

        iba = integer_to_bitarray(ilab,norb)
        jba = integer_to_bitarray(jlab,norb)
        print(' iba, jba')
        print(iba, jba)

        attempts = abs(int(psips_array[idet,3]))
        detsign = (psips_array[idet,3] > 0) - (psips_array[idet,3] < 0)
        print(' attempts, detsign')
        print(attempts, detsign)

        for ispawn in range(attempts):
            print(' ispawn')
            print(ispawn)
            # Spawn from i -> k, j constant (column spawning)
            ex_data = random_excitation(iba, nel, na, nb, norb, nvirt,
                                        psingle, pdouble, pg_mask, maxsym,
                                        orbms, orbsym)
            nex, a, b, r, s, perms, pgen = ex_data
            print(' nex, a, b, r, s, perms, pgen')
            print(nex, a, b, r, s, perms, pgen)
            allowed = (nex == 1) or (nex == 2)
            if allowed:
                nspawn, klab, hij = do_symmetric_spawn(nex, iba, a, b, r, s,
                                                       perms, pgen, norb, h1e,
                                                       h2e, tau)
                print(' nspawn, klab, hij')
                print(nspawn, klab, hij)
                if nspawn != 0:
                    nspawn = detsign*nspawn
                    kjlabel = two_integer_to_integer(klab,jlab)

                    loc, new_tslots = find_array_loc(spawn_array,
                                                     kjlabel, tslots)
                    print(' nspawn, kjlabel, loc, new_tslots')
                    print(nspawn, kjlabel, loc, new_tslots)
                    if new_tslots == tslots:
                        spawn_array[loc,3] += nspawn
                    else:
                        tslots = new_tslots
                        spawn_array[loc,0] = klab
                        spawn_array[loc,1] = jlab
                        spawn_array[loc,2] = kjlabel
                        spawn_array[loc,3] = nspawn
                        spawn_array[loc,4] = hij

            # Spawn from j -> k, i constant (row spawning)
            #ex_data = random_excitation(jba, nel, na, nb, norb, nvirt,
            #                            psingle, pdouble, pg_mask, maxsym,
            #                            orbms, orbsym)
            #nex, a, b, r, s, perms, pgen = ex_data
            #allowed = (nex == 1) or (nex == 2)
            #if allowed:
            #    nspawn, klab, hij = do_symmetric_spawn(nex, jba, a, b, r, s,
            #                                           perms, pgen, norb, h1e,
            #                                           h2e, tau)
            #    if nspawn != 0:
            #        nspawn = int(detsign*nspawn)
            #        iklabel = two_integer_to_integer(ilab,klab)

            #        loc, new_tslots = find_array_loc(spawn_array,
            #                                         iklabel, tslots)

            #        if new_tslots == tslots:
            #            spawn_array[loc,3] += nspawn
            #        else:
            #            tslots = new_tslots
            #            spawn_array[loc,0] = ilab
            #            spawn_array[loc,1] = klab
            #            spawn_array[loc,2] = iklabel
            #            spawn_array[loc,3] = nspawn
            #            spawn_array[loc,4] = hij

    return spawn_array, tslots

@numba.njit
def symmetric_dmqmc_CD():
    """
    Do cloning and death in symmetric DMQMC
    """
    print('Hello world!')
    return

@numba.njit
def symmetric_dmqmc_annihilation():
    """
    Do the symmetric DMQMC annihilation step.
    """
    print('Hello world!')
    return

