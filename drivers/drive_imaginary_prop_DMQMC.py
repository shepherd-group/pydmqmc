#!/usr/bin/env python

import numpy as np
import pandas as pd
import functions as fn

# "QMC" & System parameters
#seed = np.random.randint(2**32 - 1)
seed = 2401
np.random.seed(seed)
shift = 0.0
#cycles = 100
cycles = 1
target = 10
#tau = 0.001
tau = 0.1
zeta = 0.1
reports = int(target/(tau*cycles))
beta_loops = 1
Nattempts = int(5E5)
init = 'uniform-uniform'
det_space = 50

H, Heval, HS = fn.system_initialize('EQUILIBRIUM-H6-STO3G.hamil', shift)
H0 = fn.non_interacting(H)
eye = np.eye(HS)

# Generate rho_D
rho = np.eye(HS)
for betastep in range(int(3.0/tau)):
    rho += -tau/2 * (H @ rho + rho @ H)
rho_D = rho[:det_space, :det_space]

# Data Saving
data1 = []
data2 = []
#csvname1 = 'imaginary-prop1-round-t01'
#csvname2 = 'imaginary-prop2-round-t01'
csvname1 = 'imaginary-prop1-noround-t01'
csvname2 = 'imaginary-prop2-noround-t01'
            
for betaloop in range(1,beta_loops+1):
    print(' # Beta Loop:', betaloop)

    iteration1, report1 = 0, 0
    d1, occrows1, df1 = fn.initialize_dm(init, Nattempts, target, Heval, HS)
    df1 = { 'Beta':[], 'Shift':[], 'Re{Tr(Hp)}':[], 'Im{Tr(Hp)}':[],
            'Re{Tr(p)}':[], 'Im{Tr(p)}':[], 'Re{Nw}':[], 'Im{Nw}':[],
            'Re{<E>}':[], 'Im{<E>}':[], 'N_rows':[]}
    d1 = np.eye(HS)
    d1 = d1.astype('complex128')
    df1 = fn.complex_report(iteration1, tau, shift, d1, Heval, df=df1, stdout=True)

    iteration2, report2 = 0, 0
    d2, occrows2, df2 = fn.initialize_dm(init, Nattempts, target, Heval, HS)
    df2 = { 'Beta':[], 'Shift':[], 'Re{Tr(Hp)}':[], 'Im{Tr(Hp)}':[],
            'Re{Tr(p)}':[], 'Im{Tr(p)}':[], 'Re{Nw}':[], 'Im{Nw}':[],
            'Re{<E>}':[], 'Im{<E>}':[], 'N_rows':[]}
    d2 = np.eye(HS)
    d2 = d2.astype('complex128')
    df2 = fn.complex_report(iteration2, tau, shift, d2, Heval, df=df2, stdout=False)
    rho = rho.astype('complex128')

    for report in range(reports):
    
        for cycle in range(cycles):

            if round(iteration1*tau, 4) <= 3.0:
                iteration1 += 1
                iteration2 += 1
                rho += -tau/2 * (H @ rho + rho @ H)
                d1 = rho
                d2 = rho
                if round(iteration1*tau, 4) == 3.0:
                    rho_D = rho[:det_space, :det_space]
                    rho_D = rho_D.astype('complex128')
                    d1 = np.eye(HS)
                    d2 = np.eye(HS)
                    d1 = d1.astype('complex128')
                    d2 = d2.astype('complex128')
                    d1[:det_space, :det_space] = rho_D
                    d2[:det_space, :det_space] = rho_D
            else:
                iteration1 += 1
                iteration2 += 1

                '''
                    Python imaginary numbers are given as j.
                    i_prop = I - 1j*tau*H - tau^2 * H^2
                '''
                deltad1 = d1 @ (-1j*tau*H - tau**2 * (H @ H))

                #re_deltad1 = np.real(deltad1)
                #re_deltad1 = fn.stochastic_round(re_deltad1)
                #im_deltad1 = np.imag(deltad1)
                #im_deltad1 = fn.stochastic_round(im_deltad1)
                #deltad1 = re_deltad1 + 1.0j*im_deltad1

                d1 += deltad1
                d1[:det_space, :det_space] = rho_D

                '''
                    Python imaginary numbers are given as j.
                    i_prop = -1j * tau * (H @ d) * (I - 0.5j * tau * (H @ d))
                '''
                #deltad2 = -1.0j * tau * (H @ d2) * (eye - 0.5j * tau * (H @ d2))
                deltad2 = ((-1.0j * tau * H) @ (eye - 0.5j * tau * H)) @ d2

                #re_deltad2 = np.real(deltad2)
                #re_deltad2 = fn.stochastic_round(re_deltad2)
                #im_deltad2 = np.imag(deltad2)
                #im_deltad2 = fn.stochastic_round(im_deltad2)
                #deltad2 = re_deltad2 + 1.0j*im_deltad2

                d2 += deltad2
                d2[:det_space, :det_space] = rho_D

        df1 = fn.complex_report(iteration1, tau, shift, d1, Heval, df=df1, stdout=True)
        df2 = fn.complex_report(iteration2, tau, shift, d2, Heval, df=df2, stdout=True)
        #H, shift = fn.update_shift(H, HS, cycles, tau, df, zeta)

    data1 = fn.store_data(data1, df1, betaloop, beta_loops, csvname1, '')
    data2 = fn.store_data(data2, df2, betaloop, beta_loops, csvname2, '')

