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
cycles = 1
target = 10.0
piecewise = 1.0
tau = 0.001
zeta = 0.05
reports = int(target/(tau*cycles))
beta_loops = 1
Nattempts = int(1E4)
init = 'thermal-uniform'
cutoff = 0.1
n_add = 3.0

# System initalization
hamil_file = 'STRETCHED-H6-STO3G.hamil'
H, Heval, HS, H0 = fn.system_initialize(hamil_file, shift, ip=True)
H_nonzero = np.ones(np.shape(H))
H_nonzero -= np.eye(HS)
H_nonzero[H==0.0] = 0.0

# Data Saving
data = []
csvname = 'TEST-PIPDMQMC'

for betaloop in range(1,beta_loops+1):
    print(' # Beta Loop:', betaloop)
    iteration, report = 0, 0
    d, occrows, df = fn.initialize_dm(init, Nattempts, piecewise, Heval, HS)
    df = fn.write_report(iteration, tau, shift, d, Heval, df=df, stdout=True)
    df['n_fail'] = [0]
    df['n_fail non_init'] = [0]

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

            # Calculate n_fail
            d_nonzero = np.ones(np.shape(d))
            d_nonzero[d==0.0] = 0.0
            C = np.einsum('ij,jk->ij',d_nonzero,H_nonzero)
            d_noninit = np.copy(d_nonzero)
            d_noninit[rhopop>=n_add] = 0.0
            A = np.einsum('ij,ik,jk->ij',d_noninit,d_nonzero,H_nonzero)
            n_fail = C - A
            n_fail = np.divide(n_fail,C,where=C!=0)
            n_fail[d==0.0] = 0.0
            df['n_fail'].append(n_fail.sum()/HS**2)
            n_fail[rhopop>=n_add] = 0.0
            df['n_fail non_init'].append(n_fail.sum()/HS**2)

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

