#!/usr/bin/env python

import numpy as np
import pandas as pd
import functions as fn

# "QMC" & System parameters
#seed = np.random.randint(2**32 - 1)
#seed = 2401
seed = 7
np.random.seed(seed)
shift = 0.0
#cycles = 100
#cycles = 20
cycles = 1
target = 25
#tau = 0.001
#tau = 0.005
tau = 0.1
zeta = 0.1
reports = int(target/(tau*cycles))
beta_loops = 1
#Nattempts = int(5E3)
Nattempts = int(5E5)
init = 'uniform-uniform'
det_space = 5

H, Heval, HS = fn.system_initialize('EQUILIBRIUM-H6-STO3G.hamil', shift)
H0 = fn.non_interacting(H)
eye = np.eye(HS)

# Data Saving
data = []
csvname = '5x5exactspace-prop-1jtauHdI-0.5jtauHd'
            
for betaloop in range(1,beta_loops+1):
    print(' # Beta Loop:', betaloop)

    iteration, report = 0, 0
    d, occrows, df = fn.initialize_dm(init, Nattempts, target, Heval, HS)
    d = d.astype('complex128')
    rho_det = np.copy(d)
    df = { 'Beta':[], 'Shift':[], 'Re{Tr(Hp)}':[], 'Im{Tr(Hp)}':[],
           'Re{Tr(p)}':[], 'Im{Tr(p)}':[], 'Re{Nw}':[], 'Im{Nw}':[],
           'Re{<E>}':[], 'Im{<E>}':[], 'N_rows':[]}
    fn.complex_report(iteration, tau, shift, d, Heval, df=df, stdout=True)

    for report in range(reports):
    
        for cycle in range(cycles):
            iteration += 1

            if round(iteration*tau,4) > 3.0:
                '''
                    Python imaginary numbers are given as j.
                    i_prop = ((-1.0j * tau * H) @ (eye - 0.5j * tau * H)) @ d
                '''
                #deltad = -1.0j * tau * (H @ d) * (eye - 0.5j * tau * (H @ d))
                deltad = ((-1.0j * tau * H) @ (eye - 0.5j * tau * H)) @ d

                re_deltad = np.real(deltad)
                re_deltad = fn.stochastic_round(re_deltad)
                im_deltad = np.imag(deltad)
                im_deltad = fn.stochastic_round(im_deltad)
                deltad = re_deltad + 1.0j*im_deltad

                d += deltad
                d[:det_space, :det_space] = rho_D
                #d = d / np.linalg.norm(d, ord='fro')
            else:
                '''
                    "DMQMC" non-symmetric propagator:
                    dd/dtau = -d @ H
                '''
                deltad = -tau * (d @ H)
                deltad = fn.stochastic_round(np.real(deltad))
                d += deltad

                delta_rho_det = -tau * (rho_det @ H)
                rho_det += delta_rho_det

            if round(iteration*tau,4) == 3.0:
                #rho_D = fn.empty_array(HS)
                #rho_D[:det_space, :det_space] = d[:det_space, :det_space]
                rho_D = rho_det[:det_space, :det_space]

        fn.complex_report(iteration, tau, shift, d, Heval, df=df, stdout=True)
        #H, shift = fn.update_shift(H, HS, cycles, tau, df, zeta)

    data = fn.store_data(data, df, betaloop, beta_loops, csvname, '')

