#!/usr/bin/env python

import numpy as np
from functions import symmetric_bloch as SB
from functions import system_initialize as INIT
from integrals_readin import integral_system as readin

def test_fci():
    r'''
    A simple FCI example demonstrating the new read in functionality
    for pydmqmc.
    '''
    system = readin(
                int_file = 'systems/STRICT-EIGENVALUES-STO3G-STR-H6.FCIDUMP',
                verbose = True,
                hamiltonian = True,
            )
    fci1, vec1 = np.linalg.eigh(system.H)

    H, Heval, HS = INIT('systems/STRETCHED-H6-STO3G.hamil', 0.0)
    fci2, vec2 = np.linalg.eigh(Heval)

    dfci = fci1 - fci2
    dfci = dfci[np.argsort(dfci)]
    print(np.max(np.abs(dfci)))
    # Max delta: 1.2495782186761062e-10
    # This is appropriate considering the strict integrals
    # are better converged than those used to make the .hamil file.

    system = readin(
                int_file = 'systems/STRICT-EIGENVALUES-STO3G-EQM-H6.FCIDUMP',
                verbose = True,
                hamiltonian = True,
            )
    fci1, vec1 = np.linalg.eigh(system.H)

    H, Heval, HS = INIT('systems/EQUILIBRIUM-H6-STO3G.hamil', 0.0)
    fci2, vec2 = np.linalg.eigh(Heval)

    dfci = fci1 - fci2
    dfci = dfci[np.argsort(dfci)]
    print(np.max(np.abs(dfci)))
    # 8.452842870099175e-10
    # This is appropriate considering the strict integrals
    # are better converged than those used to make the .hamil file.

def test_dmqmc():
    r'''
    A simple DMQMC example demonstrating the new read in functionality
    for pydmqmc.
    '''
    system = readin(
                int_file = 'systems/STRICT-EIGENVALUES-STO3G-STR-H6.FCIDUMP',
                verbose = True,
                hamiltonian = True,
            )
    H1  = np.copy(system.H)
    H1 -= np.eye(system.ndets)*H1[0,0]

    H, Heval, HS = INIT('systems/STRETCHED-H6-STO3G.hamil', 0.0)

    tau = 0.001
    cycles = 10
    target = 4
    reports = int(target/(tau*cycles))

    rho = np.eye(200)
    rho1 = np.eye(200)
    for report in range(reports):
        for cycle in range(cycles):
            rho += SB(H,rho,tau)
            rho1 += SB(H1,rho1,tau)

    en = (Heval @ rho).trace() / rho.trace()
    en1 = (system.H @ rho1).trace() / rho1.trace()
    den = en - en1
    print(abs(den))
    # Delta: 8.870815193517956e-11
    # This seems appropriate 

if __name__ == '__main__':
    #test_fci()
    #test_dmqmc()

