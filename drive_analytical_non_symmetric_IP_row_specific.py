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
Htilde = unphysical_hamiltonian(H)

Heval = np.copy(H)
Htildeeval = np.copy(Htilde)

FCI, v = LA.eigh(Heval)
FCItilde, vtilde = LA.eigh(Htildeeval)

H = H - (np.eye(HS)*HF)
Htilde = Htilde - (np.eye(HS)*HF)

H0 = np.diag(np.diag(H))
Htilde0 = np.diag(np.diag(Htilde))

# Parameters
shift = 0.0
cycles = 1
target = 7
tau = 0.1
reports = int(target/(tau*cycles))

row_slices = [1,5,10,50,100,200]

ftotal = np.diag(np.exp(-target*np.diag(H0)))
alldf = []

for row_slice in row_slices:

    f = np.zeros((HS,HS))
    for i in range(row_slice):
        f[i,i] = ftotal[i,i]

    ftilde = np.copy(f)
    
    iteration = 0
    report = 0
    
    ftdf = { 'Beta':[], 'Shift':[], 'Tr(Hp)':[], 'Tr(p)':[], 'Nw':[], '<E>':[]}
    csvname = 'analytical-ip-strh6sto3g-'+str(row_slice)+'-of-200-rowslice.csv'
    expectation(Heval, f, 'Energy')
    write_report(iteration, tau, shift, f, Heval, df=ftdf, printbool=True)
    
    for report in range(reports):
    
        for cycle in range(cycles):
            iteration += 1
    
            '''
                "IP-DMQMC"
    
                df/dtau = H0 @ f - f @ H
            '''
    
            df = tau * ( (H0 @ f) - (f @ H) )
 
            f += df
    
            dftilde = tau * ( (Htilde0 @ ftilde) - (ftilde @ Htilde) )
    
            ftilde += dftilde
    
        write_report(iteration, tau, shift, f, Heval, df=ftdf, printbool=True)

    pd.DataFrame(ftdf).to_csv('./outputs/'+csvname)

