#!/usr/bin/env python

import numpy as np
import pandas as pd
import time
from excitations import random_bitarry_symspace as randomdet
from excitations import calculate_psingle_pdouble as calc_excit_prob
from excitations import generate_renorm_excitation as exc_renorm
from excitations import calculate_psingle_pdouble
from utilities import bitarry_to_integer as btoint
from utilities import orb_sym as csym
from utilities import cross_prod_pg_sym as xpsym
from utilities import get_hij
from utilities import concate_bitarrays_to_label as bas_to_lab
from utilities import extract_bitarrays_from_label as lab_to_bas
from utilities import get_nex
from integrals_readin import integral_system

def regressiontest_random_diagonal_determinant():
    np.random.seed(7)
    failure = False
    test_file = 'tests/inputs/fcidump/STRICT-EIGENVALUES-STO3G-STR-H6.FCIDUMP'
    integral_data = integral_system(
                                int_file = test_file,
                                verbose = True,
                                determinants = True,
                            )
    uniqueness = {}
    for ba1 in integral_data.bitarrays:
        for ba2 in integral_data.bitarrays:
            label = bas_to_lab(ba1,ba2)
            uniqueness[label] = 0
    unique_keys = len(list(uniqueness.keys()))
    matrix_space = int(integral_data.ndets**2)
    if unique_keys != matrix_space:
        print(' Unique keys:',unique_keys)
        print(' Matrix space:',matrix_space)
        raise ValueError(' Unique keys do not span the entire matrix space!')
        failure = True

    initial_walkers = 5000
    walkers = {}
    for ba in integral_data.bitarrays:
        label = bas_to_lab(ba,ba)
        bat1, bat2 = lab_to_bas(label,integral_data.norb)
        walkers[label] = 0
        if not np.array_equal(bat1, bat2) or not np.array_equal(ba,bat1):
            raise ValueError(' Bitarray scrambling in diagonal labels!')
            failure = True

    for walker in range(initial_walkers):
        ba = randomdet(integral_data)
        label = bas_to_lab(ba,ba)
        try:
            walkers[label] += 1
        except KeyError:
            raise KeyError(' Generate label not in symmetry space!')
            failure = True

    nw = sum(list(walkers.values()))
    if initial_walkers != nw:
        raise ValueError(' Walkers lost in generation of space!')
        failure = True
    return failure

def regressiontest_psingle_pdouble():
    benchmark_psingle = 0.13559322
    benchmark_pdouble = 0.86440678
    failure = False
    test_file = 'systems/STRICT-EIGENVALUES-STO3G-STR-H6.FCIDUMP'
    integral_data = integral_system(
                                int_file = test_file,
                                verbose = True,
                                determinants = True,
                            )
    calc_excit_prob(integral_data)
    nsingle = 0
    ndouble = 0
    ba_reference = integral_data.bitarrays[0]
    for i, ba in enumerate(integral_data.bitarrays):
        nex = get_nex(ba_reference, ba)
        if nex == 1:
            nsingle += 1
        elif nex == 2:
            ndouble += 1
    psingle = nsingle/(nsingle + ndouble)
    pdouble = ndouble/(nsingle + ndouble)
    print()
    print(' p_single:', psingle)
    print(' p_double:', pdouble)

def regressiontest_normalized_excitation_generation():
    np.random.seed(7)
    test_file = 'systems/STRICT-EIGENVALUES-STO3G-STR-H6.FCIDUMP'
    sys = integral_system(
                        int_file = test_file,
                        verbose = True,
                        hamiltonian = True,
                    )

    natt = int(100*sys.ndets*np.log(sys.ndets))
    start = time.time()
    print()
    print(' Starting excitation generation test!')
    print(' Running with %s attempts per determinant.' % natt)
    print(' Total determinants to test: %s' % sys.ndets)
    print()

    for i, det in enumerate(sys.bitarrays):
        print(' Running excitation generation test for det: %s' % (i+1))
        print(' Determinant bitarry: %s' %det)
        det_time = time.time()
        index = {}
        counts = []
        pgens = []
        nexs = []
        bitarrays = []

        for det2 in sys.bitarrays[np.arange(sys.ndets) != i]:
            nex = get_nex(det, det2)
            print(nex)
            if nex == 1 or nex == 2:
                label = bas_to_lab(det,det2)
                index[label] = len(index)
                counts.append(0)
                pgens.append(0)
                nexs.append(nex)
                bitarrays.append(det2)

        nsingle = 0
        ndouble = 0
        for gen in np.arange(natt):
            pgen, hij, nex, gen_det = exc_renorm(sys, det)
            if nex == 1 or nex == 2:
                label = bas_to_lab(det,gen_det)
                ind = index[label]
                counts[ind] += 1
                pgens[ind] = pgen
                nsingle += int(nex == 1)
                ndouble += int(nex == 2)

        counts = np.array(counts)
        pgens = np.array(pgens)
        nexs = np.array(nexs)
        counts_norm = counts.astype(float)/pgens
        single = nexs == 1
        double = nexs == 2
        totnormsingle = counts_norm[single].sum()
        totnormdouble = counts_norm[double].sum()
        totsingle = counts[single].sum()
        totdouble = counts[double].sum()
        df = pd.DataFrame(
                {
                    'label':list(index.keys()),
                    'dets':bitarrays,
                    'counts':counts,
                    'pgens':pgens,
                    'count/pgens':counts_norm,
                    'nex':nexs,
                }
            )
        print(df.sort_values('nex').to_string(index=False))
        print()
        print('(1) Estimate. psingle:',totsingle/(totsingle+totdouble))
        print('(2) Estimate. psingle:',nsingle/(nsingle+ndouble))
        print('       Exact. psingle:',sys.psingle)
        print('      Normed. psingle:',totnormsingle/(totnormsingle+totnormdouble))
        print('(1) Estimate. pdouble:',totdouble/(totsingle+totdouble))
        print('(2) Estimate. pdouble:',ndouble/(nsingle+ndouble))
        print('       Exact. pdouble:',sys.pdouble)
        print('      Normed. pdouble:',totnormdouble/(totnormsingle+totnormdouble))
        print('            \sum pgen:',np.sum(pgens))
        
        runtime = time.time() - det_time
        print(' Completed excitation generation test for det: %s' % (i+1))
        print(' Total time for test: %6.4f (min)' % (runtime/(60)))
        print()
    tot_time = time.time() - start
    print(' Testing complete, total time alloted: %s (min)' % (tot_time/60))

def regressiontest_exact_psingle_pdouble():
    np.random.seed(7)
    test_file = 'systems/STRICT-EIGENVALUES-STO3G-STR-H6.FCIDUMP'
    sys = integral_system(
                        int_file = test_file,
                        verbose = True,
                        hamiltonian = True,
                    )

    start = time.time()
    print()
    print(' Starting exact psingle/pdouble test!')
    print(' Total determinants to enumerate: %s' % sys.ndets)
    print()

    nsingle = 0
    ndouble = 0
    for i, det in enumerate(sys.bitarrays):
        det_time = time.time()
        print(' Calculating nsingle/ndouble for det: %s' % (i+1))
        print(' Determinant bitarry: %s' %det)

        for det2 in sys.bitarrays[np.arange(sys.ndets) != i]:
            nex = get_nex(det, det2)
            nsingle += int(nex == 1)
            ndouble += int(nex == 2)
        
        runtime = time.time() - det_time
        print(' Completed excitation generation test for det: %s' % (i+1))
        print(' Total time for test: %6.4f (min)' % (runtime/(60)))
        print()
    ntot = nsingle+ndouble
    psingle = nsingle/ntot
    pdouble = ndouble/ntot
    print()
    print('       Exact. psingle:',psingle)
    print('       Exact. pdouble:',pdouble)
    tot_time = time.time() - start
    print()
    print(' Testing complete, total time alloted: %s (min)' % (tot_time/60))

def regressiontest_histogram_excitations():
    #test_file = 'systems/STRICT-STO3G-STR-H4.FCIDUMP'
    test_file = 'systems/STRICT-EIGENVALUES-STO3G-STR-H6.FCIDUMP'
    sys = integral_system(
                        int_file = test_file,
                        verbose = True,
                        #eigenvalues = True,
                        #reference = [0,1,4,5],
                        hamiltonian = True,
                    )

    ngen = int(10*(500*200*np.log(200)))
    start = time.perf_counter()
    print()
    print(' Starting histogram of excitations test!')
    print()
    for seed in np.arange(1,31,1):
        np.random.seed(seed)
        print()
        print(' Running excitation histogram with seed:',seed)
        print(' Excitation attempts:',ngen)
        print()

        excitations = {}
        for i, det1 in enumerate(sys.bitarrays):
            for j, det2 in enumerate(sys.bitarrays):
                nex = get_nex(det1, det2)
                bsingle = nex == 1
                bdouble = nex == 2
                if bsingle or bdouble:
                    label1 = bas_to_lab(det1,det2)
                    excitations[label1] = [0,0]

        for _ in range(ngen):
            rdet = randomdet(sys)
            pgen, hij, nex, edet = exc_renorm(sys, rdet)
            if nex == 1 or nex == 2:
                label1 = bas_to_lab(rdet,edet)
                excitations[label1][0] += 1/pgen
                excitations[label1][1] += 1

        k, vn = list(excitations.keys()), list(excitations.values())
        v,n = np.transpose(vn)
        df = pd.DataFrame({'label':k,'counts/pgen':v,'counts':n})
        print(df.to_string(index=False))
        break

    tot_time = time.perf_counter() - start
    print()
    print(' Testing complete, total time alloted: %s (min)' % (tot_time/60))

def regressiontest_histogram_diagonal_excitation():
    test_file = 'systems/STRICT-STO3G-STR-H4.FCIDUMP'
    sys = integral_system(
                        int_file = test_file,
                        verbose = True,
                        eigenvalues = True,
                        reference = [0,1,4,5],
                        hamiltonian = True,
                    )

    ngen = int(500*20*np.log(20))
    data = []

    start = time.perf_counter()
    print()
    print(' Starting histogram of diagonal excitations test!')
    print()
    for seed in np.arange(1,31,1):
        np.random.seed(seed)
        print(' Running excitation histogram with seed:',seed)
        print(' Excitation attempts:',ngen)
        print()

        excitations = {}
        for i, det1 in enumerate(sys.bitarrays):
            label1 = bas_to_lab(det1,det1)
            excitations[label1] = 0

        for _ in range(ngen):
            ba = randomdet(sys)
            label = bas_to_lab(ba,ba)
            excitations[label] += 1


        keys = list(excitations.keys())
        vals = list(excitations.values())
        data.append(pd.DataFrame({'label':keys,'count':vals}))

    df = pd.concat(data)
    print(df.to_string(index=False))

    tot_time = time.perf_counter() - start
    print()
    print(' Testing complete, total time alloted: %s (min)' % (tot_time/60))

def main(
        regression_test = True,
        unit_tests = False,
    ):
    #if regression_test:
        #results = regressiontest_random_diagonal_determinant()
        #print(' Regression test of random determinant generation:',results)
    pass

if __name__ == '__main__':
    #regressiontest_normalized_excitation_generation()
    #regressiontest_exact_psingle_pdouble()
    #regressiontest_histogram_excitations()
    #regressiontest_histogram_diagonal_excitation()
    #main()
    regressiontest_random_diagonal_determinant()

