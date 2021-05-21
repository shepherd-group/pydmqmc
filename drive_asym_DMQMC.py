#!/usr/bin/env python

import numpy as np
import pandas as pd
import functions as fn

# "QMC" & System parameters
#seed = np.random.randint(2**32 - 1)
seed = 117
np.random.seed(seed)
shift = 0.0
cycles = 20
target = 50
tau = 0.005
reports = int(target/(tau*cycles))
beta_loops = 1
Natt = 1
init = 'specific-uniform'

H, Heval, HS = fn.system_initialize('../../EQUILIBRIUM-H6-STO3G.hamil')
eig, vec = fn.FCI(H)
Heval = H
H0 = np.diag(np.diag(H))

data = []
csvname = str(seed)+'Seed-ROW1-CONSTANT-ROW2-CHANGING-SYMMETRIC-IPDMQMC-DMQMC-HYBRID-EQM-H6'

def gs_signal_cancel(vec1, vec2):

    '''
    Orthogonalize two vectors

        In:
            vec1:
                The first vector we want to orthoganalize
            vec2:
        Out:
            dm:
                The density matrix row with the ground state removed.
    '''

    dot_prod = np.dot(vec1, vec2) / np.dot(vec2, vec2)
    signal = dot_prod*vec2
    return signal

for betaloop in range(1,beta_loops+1):
    print(' Beta Loop:', betaloop)

    iteration, report = 0, 0
    d, occ, df = fn.initialize_dm(init, Natt, target, Heval, HS, rowlist=[0])
    for rep_cyc in range(int(reports*cycles)):
        d += -tau*(d @ H)
    for i in range(1,HS):
        d[i,i] = 1
        #d[i,:] = d[1,:] - gs_signal_cancel(d[i,:], d[0,:])
        dot_prod = np.dot(d[i,:], d[0,:])/np.dot(d[0,:], d[0,:])
        d[i,:] = d[i,:] - dot_prod*d[0,:]

    dtemp = np.zeros((HS,HS))
    dtemp[1:] = d[1:]
    fn.write_report(iteration, tau, shift, dtemp, Heval, df=df, stdout=True, ind_row_evals=True)
    

    for report in range(reports):
        for cycle in range(cycles):
            iteration += 1

            '''
                "DMQMC" non-symmetric propagator:

                dd/dtau = - d @ H
            '''
            # Pure Asymmetric DMQMC Propagation.
            deltad = -tau * (d @ H)

            for i in range(1,HS):
                #deltad[i,:] = deltad[i,:] - gs_signal_cancel(deltad[i,:], d[0,:])
                dot_prod = np.dot(deltad[i,:], d[0,:])/np.dot(d[0,:], d[0,:])
                deltad[i,:] = deltad[i,:] - d[0,:]*dot_prod
                d[i,:] += deltad[i,:]

#            dot_prod = np.dot(deltad[1,:], d[0,:])/np.dot(d[0,:], d[0,:])
#            print('Numerator of Dot product:', np.dot(deltad[1,:], d[0,:]))
#            print('Denominator of Dot Product:', np.dot(d[0,:], d[0,:]))
#            print('Dot Product:', dot_prod)
#            deltad[1,:] = deltad[1,:] - d[0,:]*dot_prod
#            d[1,:] += deltad[1,:]

#            deltadr = -tau * (d @ H)
#            # DMQMC between rows
#            deltadl = -tau * (H @ d)
#            # IP-DMQMC
#            deltaf = tau*(H0 @ d - d @ H)
#
#            d[1,:] += 0.5*(deltadl[1,:] + deltadr[1,:]) + deltaf[1,:]
#            #d[1,:] += 0.5*(deltaf[1,:])

        dtemp = np.zeros((HS,HS))
        dtemp[1:] = d[1:]
        fn.write_report(iteration, tau, shift, dtemp, Heval, df=df, stdout=True, ind_row_evals=True)

    fn.store_data(data, df, betaloop, beta_loops, csvname, '')

