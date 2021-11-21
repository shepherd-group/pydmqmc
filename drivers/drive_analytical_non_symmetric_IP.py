#!/usr/bin/env python

import numpy as np
import pandas as pd
from functions import *

# "QMC" & System parameters
#seed = np.random.randint(2**32 - 1)
seeds = [7,8,9,10]
shift = 0.0
cycles = 50
targets = [10]
tau = 0.01
#reports = int(target/(tau*cycles))
beta_loops = 1
#attempts = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
#            10000, 20000, 50000, 100000, 200000, 500000, 1000000,
#            2000000]
attempts = [1]
#init_dm = ['thermal-thermal', 'thermal-uniform',
#           'uniform-thermal', 'uniform-uniform']
init_dm = ['thermal-uniform']
#simulations = zip(seeds, [attempts, attempts, attempts, attempts], init_dm)
simulations = zip(seeds, [attempts], init_dm)

H, Heval, HS = system_initialize('EQUILIBRIUM-H4-STO3G.hamil', shift)
#H0 = np.diag(np.diag(H)) + (np.eye(HS)*shift)
W = np.exp(np.diag(H) * -targets[0])
H0 = np.diag(np.diag(H))

for seed, attempts, init in simulations:
    np.random.seed(seed)

    for target in targets:
        reports = int(target/(tau*cycles))

        for Nattempts in attempts:
            print(' Nattempts:',Nattempts)
            # Data Saving
            data = []
            path     = './outputs/1row-convergence/'
            csvout   = 'str-' + init + '-analytical-nonsym-ipdmqmc'
            csvout  += '-Nbeta' + str(beta_loops)
            csvout  += '-Natt' + str(Nattempts)
            csvout  += '-seed' + str(seed)
            csvout  += '-tbeta' + str(target)
            csvout  += '-Shift' + str(shift)
            
            for betaloop in range(1,beta_loops+1):
                print(' Beta Loop:', betaloop)
            
                iteration = 0
                report = 0
                f, occrows, df = initialize_dm(init, Nattempts, target, Heval, HS)
                f = np.diag(W)
                write_report(iteration, tau, shift, f, Heval, df=df, stdout=True)
            
                for report in range(reports):
                
                    for cycle in range(cycles):
                        iteration += 1
                
                        '''
                            "IP-DMQMC"
                
                            df/dtau = H0 @ f - f @ H
                        '''
                        deltaf = tau * ( (H0 @ f) - (f @ H) )
                        
                        f += deltaf
                
                    write_report(iteration, tau, shift, f, Heval, df=df, stdout=True)

#                store_data(data, df, betaloop, beta_loops, csvout, path)

