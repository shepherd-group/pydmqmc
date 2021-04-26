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

eig, vec = FCI(Heval)
eigtilde, vectilde = FCI(Htildeeval)

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
beta_loops = 50

Nattempts = 1e5
Hiiweights = np.exp(-target*np.diag(Heval))
Hiiweights = Hiiweights / Hiiweights.sum()
seed = np.random.randint(2**32 - 1)
np.random.seed(seed)

alldf = {}
for loop in range(beta_loops):
    print('Beta Loop:', loop+1)

    randomrows = np.random.choice(HS, size=int(Nattempts), p=Hiiweights)
    randomrows = np.unique(randomrows)

    f = np.zeros((HS,HS))
    for ii in randomrows:
        f[ii,ii] = np.exp(-target*H[ii,ii])

    ftilde = np.copy(f)
    
    iteration = 0
    report = 0
    
    ftdf = { 'Beta':[], 'Shift':[], 'Tr(Hp)':[], 'Tr(p)':[], 'Nw':[], '<E>':[]}
    csvname = 'analytical-ip-strh6sto3g-random-row.csv'
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
    
        #write_report(iteration, tau, shift, f, Heval, df=ftdf, printbool=False)

    write_report(iteration, tau, shift, f, Heval, df=ftdf, printbool=True)
    #alldf[str(loop)]=pd.DataFrame(ftdf)
    #pd.DataFrame(ftdf).to_csv('./outputs/'+csvname)

#pd.concat(alldf).to_csv('./outputs/alldf.csv')

