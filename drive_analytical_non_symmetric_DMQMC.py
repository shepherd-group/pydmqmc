#!/usr/bin/env python

import numpy as np
import pandas as pd
from functions import *

'''
    H: Shifted Hamiltonian
    Heval: Unaltered Hamiltonian
    HS: Hilbert Space
'''

# "QMC" Parameters
seed = np.random.randint(2**32 - 1)
np.random.seed(seed)
shift = 0.0
cycles = 20
target = 10
tau = 0.005
reports = int(target/(tau*cycles))
beta_loops = 50
Nattempts = 1
H, Heval, HS = system_initialize('STRETCHED-H6-STO3G.hamil')
data = []

for betaloop in range(1,beta_loops+1):
    print(' Beta Loop:', betaloop)

    randomrows = np.random.choice(HS, size=int(Nattempts))
    randomrows = np.unique(randomrows)

    d = empty_array(HS)
    for ii in randomrows:
        d[ii,ii] = 1

    iteration = 0
    report = 0
    df = { 'Beta':[], 'Shift':[], 'Tr(Hp)':[], 'Tr(p)':[], 'Nw':[], '<E>':[]}
    write_report(iteration, tau, shift, d, Heval, df=df, printbool=True)
    
    for report in range(reports):
    
        for cycle in range(cycles):
            iteration += 1
    
            '''
                "DMQMC" non-symmetric propagator:
    
                dd/dtau = -H @ d
            '''
    
            deltad = -tau * (H @ d)
            
            d += deltad
    
            write_report(iteration, tau, shift, d, Heval, df=df, printbool=False)

    data.append(pd.DataFrame(df))

outname = './outputs/seed' + str(seed)
outname += '-nonsymm-dmqmc-analytical-Nbeta50-'
outname += 'Natt' + str(Nattempts) + '.csv'
data = pd.concat(data, ignore_index=True)
data.to_csv(outname, index=False)

