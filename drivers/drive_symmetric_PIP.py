#!/usr/bin/env python

import numpy as np
import pandas as pd
import functions as fn

# The system
hamil_file = '../../STRETCHED-H6-STO3G.hamil'
H, Heval, HS, H0 = fn.system_initialize(hamil_file, 0, ip=True)
h0 = np.diag(H0)

# "QMC" paramters
shift = 0
cycles = 10
pip_target = 1.0
target = 10.0
tau = 0.001
zeta = 0.05
reports = int(target/(tau*cycles))
beta_loops = 10
Nattempts = int(1E4)
init = 'thermal-uniform'
#init = 'deterministic-thermal'
propagator = 'IPDMQMC'
stoch_rounding = True
cutoff = 0.01
variable_shift = True

# Data Saving
data = []
csvname = 'STR-H6-PIP-Analytical-Symmetric-PIP'
seed = 27
np.random.seed(seed)

for betaloop in range(1,beta_loops+1):
    propagator = 'IPDMQMC'
    print(' # Beta Loop:', betaloop)

    N_iter, report, shift = 0, 0, 0
    d, occrows, df = fn.initialize_dm(init, Nattempts, pip_target, Heval, HS)
    df = fn.write_report(N_iter, tau, shift, d, Heval, df=df, stdout=True)

    for report in range(reports):

        for cycle in range(cycles):
            N_iter += 1

            '''
            Symmetric IP-DMQMC:

            Let f = e^{-a*H^0} * e^{-tau*H} * e^{-a*H^0}
                where a = 0.5*(beta-tau)
            Then df /dt = 0.5*{H^0,f}-0.5*(H_I(-a) f + f H_I(a))
                Let rho^0(A) = e^{-A H^0}
                Then H_I (A) =rho^0(-A) H rho^0(A)
            '''

            if propagator == 'IPDMQMC':
                a = (1/2)*(pip_target-tau*(N_iter-1))
                rho0p = np.diag(np.exp(-a*h0))
                rho0n = np.diag(np.exp( a*h0))
                antic = H0 @ d+d @ H0
                inter = rho0p @ H @ rho0n @ d+d @ rho0n @ H @ rho0p
                dd = (tau/2)*antic-(tau/2)*inter
            elif  propagator == 'DMQMC':
                dd = -(tau/2)*(H @ d+d @ H)

            if stoch_rounding:
                dd = fn.stochastic_round(dd, cutoff)

            d += dd

        df = fn.write_report(N_iter, tau, shift, d, Heval, df=df, stdout=True)

        if variable_shift:
            nw_old, nw = df['Nw'][-2:] 
            H, shift = fn.update_shift(shift, nw, nw_old, zeta, tau, cycles, H, HS)

        if df['Beta'][-1] == pip_target:
            H = H + np.eye(HS)*shift
            df['Shift'][-1] = shift = 0.0
            print(' # Changing propagators from IP-DMQMC to Asymmetric DMQMC!')
            propagator = 'DMQMC'

    data = fn.store_data(data, df, betaloop, beta_loops, csvname, '')

