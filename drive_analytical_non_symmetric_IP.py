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
H0tilde = np.diag(np.diag(Htilde))

# Parameters
shift = 0.0
cycles = 1
target = 7
tau = 0.1
reports = int(target/(tau*cycles))
beta_loops = 50

for betaloop in range(1,beta_loops+1):

    ftilde = np.copy(f)
    
    iteration = 0
    report = 0
    
    expectation(hamil, dm, observable)
    
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
    

