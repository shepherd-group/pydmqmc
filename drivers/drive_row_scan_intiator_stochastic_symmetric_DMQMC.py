#!/usr/bin/env python

import numpy as np
import pandas as pd
import functions as fn

def run_simulation(qmc_dict,row,csvname,path):

    verbose = True
    seed = qmc_dict['seed']
    init_shift = qmc_dict['init_shift']
    cycles = qmc_dict['cycles']
    target = qmc_dict['target']
    tau = qmc_dict['tau']
    reports = qmc_dict['reports']
    beta_loops = qmc_dict['beta_loops']
    Nattempts = qmc_dict['Nattempts']
    init = qmc_dict['init']
    Ntarget = qmc_dict['Ntarget']
    variable_shift = qmc_dict['variable_shift']
    n_add = qmc_dict['n_add']
    zeta = qmc_dict['zeta']
    cutoff = qmc_dict['cutoff']
    deterministic = qmc_dict['deterministic']
    np.random.seed(seed)

    for betaloop in range(1,beta_loops+1):

        iteration, report, shift = 0, 0, init_shift*1.0
        H, Heval, HS = fn.system_initialize('../STRETCHED-H6-STO3G.hamil', shift)
        d, occrows, df = fn.initialize_dm(init, Nattempts, target, Heval, HS)
        d = np.zeros((HS,HS))
        d[row,row] = Nattempts
        df = fn.write_report(iteration, tau, shift, d, Heval, df=df, stdout=verbose)

        if deterministic:
            for report in range(reports):
                for cycle in range(cycles):
                    iteration += 1
                    d += -tau*(d @ H)
                df = fn.write_report(iteration, tau, shift, d, Heval, df=df, stdout=verbose)
        else:
            for report in range(reports):
                for cycle in range(cycles):
                    iteration += 1
                    # Spawn initiator first
                    absd = np.abs(d)
                    dinit = np.copy(d)
                    dinit[absd<n_add] = 0.0
                    init_spawn = -tau*(dinit @ H)
                    init_spawn = fn.stochastic_round(init_spawn, threshold=cutoff)
                    # Then do the non-initiators
                    dnoninit = np.copy(d)
                    dnoninit[absd>=n_add] = 0.0
                    noninit_spawn = -tau*(dnoninit @ H)
                    noninit_spawn = fn.stochastic_round(noninit_spawn, threshold=cutoff)
                    noninit_spawn[d==0.0] = 0.0
                    d += (init_spawn + noninit_spawn)
                df = fn.write_report(iteration, tau, shift, d, Heval, df=df, stdout=verbose)
                if variable_shift:
                    nw = df['Nw'][-1]
                    nw_old = df['Nw'][-2]
                    H, shift = fn.update_shift(shift, nw, nw_old, zeta, tau, cycles, H, HS)

        data = []
        data = fn.store_data(data, df, betaloop, beta_loops, csvname, path=path)

# "QMC" & System parameters
qmc = {
        'seed' : 7,
        'init_shift' : 0.0,
        'cycles' : 1,
        'target' : 5,
        'tau' : 0.001,
        'reports' : -1,
        'beta_loops' : 100,
        'Nattempts' : np.inf,
        'init' : 'uniform-uniform',
        'Ntarget' : 1E8,
        'variable_shift' : None,
        'n_add' : 3.0,
        'zeta' : 0.05,
        'cutoff' : 0.1,
        'deterministic' : False,
    }

'''
    Want to run the following simulations per row:
        initiator aDMQMC + Variable Shift + 1E4/200
        full-scheme aDMQMC + Variable Shift + 1E4/200
        initiator aDMQMC + Constant Shift + 1E4/200
        full-scheme aDMQMC + Constant Shift + 1E4/200
        initiator aDMQMC + Variable Shift + 1E4
        full-scheme aDMQMC + Variable Shift + 1E4
        initiator aDMQMC + Constant Shift + 1E4
        full-scheme aDMQMC + Constant Shift + 1E4
        Deterministic
'''
sim_data = {
            'initiator_aDMQMC_variable_shift_5E1' : [3.0,True,5E1],
            }

_, _, HS = fn.system_initialize('../STRETCHED-H6-STO3G.hamil', 0)
#path = 'row_data/'
path = ''
for row in range(HS):
    print(' Row:',row)
    for sim in sim_data:
        print('  Sim:',sim)
        qmc_tmp = qmc.copy()
        n_add_tmp, variable_shift_tmp, Nattempts_tmp = sim_data[sim]
        qmc_tmp['n_add'] = n_add_tmp
        qmc_tmp['variable_shift'] = variable_shift_tmp
        qmc_tmp['Nattempts'] = int(Nattempts_tmp)
        qmc_tmp['reports'] = int(qmc_tmp['target']/(qmc_tmp['tau']*qmc_tmp['cycles']))
        if sim == 'deterministic': qmc_tmp['deterministic'] = True
        csvname = str(row+1).zfill(5) + 'row_' + sim
        run_simulation(qmc_tmp,row,csvname,path)

