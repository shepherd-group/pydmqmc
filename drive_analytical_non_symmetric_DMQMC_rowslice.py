#!/usr/bin/env python

import numpy as np
import pandas as pd
from functions import *

'''
    Send off the file name of the Hamiltonian
    We get back the:
        H:  Hamiltonian
        HS: Size of the Hilbert Space
        HF: Reference Energy
'''

H, HS, HF = build_hamiltonian('STRETCHED-H6-STO3G.hamil')
H, sorted_diags, sorted_hash = sort_index_by_diagonal(H, HS)
Heval = np.copy(H)
H = H - (np.eye(HS)*HF)

eig, vec = FCI(Heval)

# "QMC" Parameters
shift = 0.0
cycles = 100
tau = 0.001
#cycles = 20
target = 7
#tau = 0.005
reports = int(target/(tau*cycles))
slices = [1,5,10,50,100,200]
datadf = {}

for rowslice in slices:

    d = np.zeros((HS,HS))
    for ii in range(rowslice):
        d[ii,ii] = 1

    iteration = 0
    report = 0
    df = { 'Beta':[], 'Shift':[], 'Tr(Hp)':[],
           'Tr(p)':[], 'Nw':[], '<E>':[], 'Rows':[rowslice]}
    write_report(iteration, tau, shift, d, Heval, df=df, printbool=True)
    
    for report in range(reports):
    
        for cycle in range(cycles):
            iteration += 1
    
            '''
                "DMQMC" non-symmetric propagator:
    
                dp/dtau = -H @ f
            '''
    
            deltad = -tau * (H @ d)
            
            d += deltad
    
        write_report(iteration, tau, shift, d, Heval, df=df, printbool=True)
        df['Rows'].append(rowslice)

    datadf['Row 0:'+str(rowslice)] = pd.DataFrame(df)

df = pd.concat(datadf, ignore_index=True)
df.to_csv('./outputs/non-sym-DMQMC-row-slice.csv', index=False)

