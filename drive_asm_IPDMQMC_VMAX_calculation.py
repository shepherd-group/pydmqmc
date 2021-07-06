#!/usr/bin/env python

import numpy as np
import pandas as pd
import functions as fn

# "QMC" & System parameters
#seed = np.random.randint(2**32 - 1)
shift = 0.0
cycles = 10
target = 25
tau = 0.1
reports = int(target/(tau*cycles))
beta_loops = 200
Nattempts = 1
init = 'thermal-uniform'

H, Heval, HS = fn.system_initialize('../../STRETCHED-H6-STO3G.hamil')
Heval = fn.unphysical_hamiltonian(Heval)
H = Heval - np.eye(HS)*Heval[0,0]
H0 = np.diag(np.diag(H))
Heval = np.copy(H)

data = []
csvname = 'Vmax-ASYMMETRIC-IPDMQMC-STR-H6'

for betaloop in range(1,beta_loops+1):
    print(' Beta Loop:', betaloop)

    iteration, report = 0, 0
    f, occ, df = fn.initialize_dm(init, 1, target, Heval, HS)
    f = np.zeros((HS,HS))
    f[betaloop-1,betaloop-1] = np.exp(-target*H0[betaloop-1,betaloop-1])
    df = fn.write_report(iteration, tau, shift, f, Heval, df=df, stdout=True)

    for report in range(reports):
        for cycle in range(cycles):
            iteration += 1
    
            '''
                "IP-DMQMC" non-symmetric propagator:

                dd/dtau = (H0 @ f - f @ H)
            '''
            deltaf = tau * ((H0 @ f) - (f @ H))
            f += deltaf

        df = fn.write_report(iteration, tau, shift, f, Heval, df=df, stdout=True)

    data = fn.store_data(data, df, betaloop, beta_loops, csvname, '')

