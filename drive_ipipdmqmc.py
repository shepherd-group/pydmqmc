#!/usr/bin/env python

import numpy as np
import pandas as pd
import functions as fn
from functions import stochastic_round as SR
from functions import piecewise_ip_bloch as PIP

# "QMC" & System parameters
seed = 7
np.random.seed(seed)
shift = 0.0
cycles = 10
target = 10.0
piecewise = 3.0
tau = 0.001
zeta = 0.05
reports = int(target/(tau*cycles))
beta_loops = 5
Nattempts = int(1E4)
init = 'thermal-uniform'
cutoff = 0.1
n_add = 3.0

# System initalization
hamil_file = 'STRETCHED-H6-STO3G.hamil'
H, Heval, HS, H0 = fn.system_initialize(hamil_file, shift, ip=True)

# Data Saving
data = []
csvname = 'TEST-PIPDMQMC'

for betaloop in range(1,beta_loops+1):
    print(' # Beta Loop:', betaloop)
    iteration, report = 0, 0
    d, occrows, df = fn.initialize_dm(init, Nattempts, piecewise, Heval, HS)
    df = fn.write_report(iteration, tau, shift, d, Heval, df=df, stdout=True)

    for report in range(reports):
        for cycle in range(cycles):
            # Calculate the current beta, and the new number of iterations
            beta = tau*(iteration)
            iteration += 1

            # Reset the shift if we are switching propagators
            if beta == piecewise:
                print(' Switching Propagators.')
                H, shift = Heval - np.eye(HS)*Heval[0,0], 0.0

            # Store this information for initiator check
            rhopop = np.abs(d)

            # Spawn initiator first, which amounts to
            # rho_ij >= n_add spawn normally
            dinit = np.copy(d)
            dinit[rhopop<n_add] = 0.0
            init_spawn = PIP(H,dinit,H0,tau,beta,piecewise)
            init_spawn = SR(init_spawn, threshold=cutoff)

            # Then do the non-initiators, which amounts to
            # rho_ij < n_add spawning normally, then zeoring
            # of spawned elements where rho_ij == 0.0
            dnoninit = np.copy(d)
            dnoninit[rhopop>=n_add] = 0.0
            noninit_spawn = PIP(H,dnoninit,H0,tau,beta,piecewise)
            noninit_spawn = SR(noninit_spawn, threshold=cutoff)
            noninit_spawn[d==0.0] = 0.0

            # Finally update our density matrix
            d += (init_spawn + noninit_spawn)

        fn.write_report(iteration, tau, shift, d, Heval, df=df, stdout=True)
        H, shift = fn.update_shift(shift, df['Nw'][-1], df['Nw'][-2],
                                   zeta, tau, cycles, H, HS)

    data = fn.store_data(data, df, betaloop, beta_loops, csvname, '')
    H, shift = Heval - np.eye(HS)*Heval[0,0], 0.0

