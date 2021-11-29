#!/usr/bin/env python

import numpy as np
from integrals_readin import integral_system as init_readin
from excitations import random_bitarry_symspace as gen_uniform_det
from excitations import generate_renorm_excitation as gen_excit
from dmqmc_object import density_matrix
from functions import stochastic_round_f as stoch_round

seed = 7
np.random.seed(7)
shift = 0.0
tau = 0.01
target = 1.0
nbeta = 10
ncycles = 1
nreports = int(target/(tau*ncycles))
particles0 = int(1E2)

test_file = 'systems/STRICT-STO3G-STR-H4.FCIDUMP'
readin = init_readin(
                    int_file = test_file,
                    verbose = True,
                    eigenvalues = True,
                    reference = [0,1,4,5],
                    hamiltonian = True,
                )

def genr():
    return np.random.random()

# Run through beta loops
for ibeta in range(nbeta):
    # Generate our initial system
    rho = density_matrix(
                    initial_particles = particles0,
                    system = readin,
                )
    rho.update_estimates(0.0)

    # Run through the reports
    for irep in range(nreports):

        # Run through the cycles
        for icyc in range(ncycles):

            # Loop through all determinants
            for label, det in rho.main.items():
                #print(label,det.nw)

                # Spawn
                attempts = int(stoch_round(det.nw))
                for attempt in range(attempts):

                    # Spawning along columns
                    pgen, hij, nex, ba2 = gen_excit(rho.system,det.ba1)
                    r = genr()
                    w = 0.5*tau*hij/pgen
                    s = np.sign(det.nw)
                    if int(w) != 0:
                        if r < abs(w-int(w)):
                            # Success!
                            dnw = s*int(w) + s
                            rho.store_spawns(det.ba1,ba2,dnw,nex,hij)
                    elif r < abs(w):
                        dnw = s
                        rho.store_spawns(det.ba1,ba2,dnw,nex,hij)

                    # Spawning along rows
                    pgen, hij, nex, ba1 = gen_excit(rho.system,det.ba2)
                    r = genr()
                    w = 0.5*tau*hij/pgen
                    s = np.sign(det.nw)
                    if int(w) != 0:
                        if r < abs(w-int(w)):
                            # Success!
                            dnw = s*int(w) + s
                            rho.store_spawns(det.ba2,ba1,dnw,nex,hij)
                    elif r < abs(w):
                            dnw = s
                            rho.store_spawns(det.ba2,ba1,dnw,nex,hij)

                # Clone/Death
                pd = stoch_round(0.5*tau*det.nw*det.T)
                det.update(pd)

            rho.merge_main_and_spawns()

        rho.update_estimates(tau*(irep+1)*ncycles)

    del rho

