#!/usr/bin/env python

import numpy as np
import pandas as pd
from functions import *

# "QMC" & System parameters
seed = np.random.randint(2**32 - 1)
np.random.seed(seed)
shift = 0.0
cycles = 1
target = 7
tau = 0.1
reports = int(target/(tau*cycles))
beta_loops = 50
Nattempts = 1000
init = 'uniform-constant'
H, Heval, HS = system_initialize('STRETCHED-H6-STO3G.hamil')
H0 = np.diag(np.diag(H))

# Data Saving
data = []
outname  = './outputs/'
outname += 'ip-dmqmc-init-scan/'
outname += init + '-analytical-nonsym-ipdmqmc'
outname += '-Nbeta' + str(beta_loops)
outname += '-Natt' + str(Nattempts)
outname += '-seed' + str(seed)

for betaloop in range(1,beta_loops+1):
    print(' Beta Loop:', betaloop)

    f, randomrows = initialize_ip(init, Nattempts, target, Heval, HS)

    iteration = 0
    report = 0
    df = { 'Beta':[], 'Shift':[], 'Tr(Hp)':[], 'Tr(p)':[], 'Nw':[], '<E>':[],
           'N_rows':[len(randomrows)]}
    write_report(iteration, tau, shift, f, Heval, df=df, printbool=True)

    for report in range(reports):
    
        for cycle in range(cycles):
            iteration += 1
    
            '''
                "IP-DMQMC"
    
                df/dtau = H0 @ f - f @ H
            '''
    
            deltaf = tau * ( (H0 @ f) - (f @ H) )
            
            f += deltaf
    
            write_report(iteration, tau, shift, f, Heval, df=df, printbool=False)
            df['N_rows'].append(len(randomrows))

    data.append(pd.DataFrame(df))

data = pd.concat(data, ignore_index=True)
data.to_csv(outname + '.csv', index=False)

