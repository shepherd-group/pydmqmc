#!/usr/bin/env python

import numpy as np
import pandas as pd
import functions as fn

# "QMC" & System parameters
seed = 7
np.random.seed(seed)
shift = 0.343
cycles = 10
target = 25
tau = 0.001
reports = int(target/(tau*cycles))
beta_loops = 1
Nattempts = 1E3
init = 'uniform-uniform'
Ntarget = 1E3
variable_shift = False

H, Heval, HS = fn.system_initialize('STRETCHED-H6-STO3G.hamil', shift)

# Data Saving
data = []
csvname = 'OG-STR-H6-STO3G-THROUGH-PLAT'

for betaloop in range(1,beta_loops+1):
    print(' Beta Loop:', betaloop)

    iteration, report = 0, 0
    d, occrows, df = fn.initialize_dm(init, Nattempts, target, Heval, HS)
    df = fn.write_report(iteration, tau, shift, d, Heval, df=df, stdout=True)

    for report in range(reports):
    
        for cycle in range(cycles):
            iteration += 1
    
            '''
                "DMQMC" symmetric propagator:

                dd/dtau = -( H @ d + d @ H)
                        = -( (Hp-Hn)@(dp-dn) + (dp-dn)@(Hp-Hn) )
                        = -( Hp@dp - Hp@dn - Hn@dp + Hn@dn
                           + dp@Hp - dp@Hn - dn@Hp + dn@Hn )
                    pos terms:      neg terms:
                        Hp@dp           Hp@dn
                        Hn@dn           Hn@dp
                        dp@Hp           dp@Hn
                        dn@Hn           dn@Hp
            '''
            dp, dn = fn.seperate_signs(d, diagonals=False)
            Hp, Hn = fn.seperate_signs(H, diagonals=False)

            ddp = (tau/2)*(Hp @ dp + Hn @ dn + dp @ Hp + dn @ Hn)
            ddp = fn.stochastic_round(ddp, threshold=1.0)

            ddn = (tau/2)*(Hp @ dn + Hn @ dp + dp @ Hn + dn @ Hp)
            ddn = fn.stochastic_round(ddn, threshold=1.0)

            d -= (ddp - ddn)

            dexact -= (tau/2)*(H @ dexact + dexact @ H)

        df = fn.write_report(iteration, tau, shift, d, Heval, df=df, stdout=True)

        nw = df['Nw'][-1]
        nw_old = df['Nw'][-2]
        if variable_shift or nw > Ntarget:
            H, shift = fn.update_shift(shift, nw, nw_old, 0.05, tau, cycles, H, HS)
            variable_shift = True

    data = fn.store_data(data, df, betaloop, beta_loops, csvname, '')

