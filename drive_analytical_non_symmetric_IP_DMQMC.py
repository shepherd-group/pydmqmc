#!/usr/bin/env python

import numpy as np
import pandas as pd
from functions import *

# "QMC" & System parameters
#seed = np.random.randint(2**32 - 1)
#np.random.seed(seed)
shift = 0.0
cycles = 20
target = 17
tau = 0.005
reports = int(target/(tau*cycles))
beta_loops = 1

H, Heval, HS = system_initialize('STRETCHED-H6-STO3G.hamil', shift)
H0 = np.diag(np.diag(H))

# Data Saving
data = []
path = './outputs/ip-dmqmc-hybrid/'
csvname = 'ip-dmqmc-hybrid-backprop'

for betaloop in range(1,beta_loops+1):
    print(' Beta Loop:', betaloop)

    iteration = 0
    report = 0
    f, occrows, df = initialize_dm('deterministic-thermal', None, 3, Heval, HS)
    write_report(iteration, tau, shift, f, Heval, df=df, stdout=True)

    for report in range(reports):
    
        for cycle in range(cycles):
            iteration += 1
            curbeta = round(iteration*tau, abs(int(np.log10(tau))))
    
            '''
                "IP-DMQMC"
    
                df/dtau = H0 @ f - f @ H
            '''

            if curbeta < 3.0:
                deltaf = tau * ( (H0 @ f) - (f @ H) )
            
                f += deltaf
    
            elif curbeta >= 3.0 and curbeta <= 10.0:
                deltaf = -(tau/2)*( (H @ f) + (f @ H))

                f += deltaf

            else:
                deltaf = -(-(tau/2)*( (H @ f) + (f @ H)))

                f += deltaf

        write_report(iteration, tau, shift, f, Heval, df=df, stdout=True)

    store_data(data, df, betaloop, beta_loops, csvname, path)

