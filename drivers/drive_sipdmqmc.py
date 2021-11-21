#!/usr/bin/env python
import functions as fn
import numpy as np

P, H, ndets = fn.system_initialize("EQUILIBRIUM-H4-STO3G.hamil")
eigen_val_exact, eigen_vect_exact = fn.FCI(H)
P0_diag = np.diag(P)
P0 = np.diag(P0_diag)

np.set_printoptions(linewidth=10000)

target_beta = 10
tau = 0.01
nreports = int(target_beta / tau)
init_pop = 1
f = np.exp(-target_beta * P0_diag)
f = np.diag(f)
det_space = 15
semi_stochastic = True

'''
Asymmetric DMQMC 
delta_rho / delta_tau = -H @ rho
delta_rho = -H @ rho * delta_tau

Asymmetric IP-DMQMC
the change in f per change in tau is equal to 
the difference of f time H0 and H times f
where f of 0 is the exponential of 
negative beta times H0

delta_f / delta_tau = (f @ H0) - (H @ f)
delta_f = ((f @ H0) - (H @ f)) * delta_tau
f(0) = exp(-beta * H0)

Note: in the equations above where we have written H
the code would use P the propogator.
'''

for inverse_temp_step in range(1, nreports + 1):
    if semi_stochastic == True:
        if inverse_temp_step < 4:
            delta_f = ((f @ P0) - (P @ f)) * tau
            f = f + delta_f
            continue
#            test_P = np.zeros((ndets, ndets))
#            test_P[:det_space, det_space:] = P[:det_space, det_space:]
#            print("zeros + P12 is\n", test_P)
    
        #P11
        delta_f_P11 = -P[:det_space, :det_space] @ f[:det_space, :] * tau 
        #P0_11
        delta_f_P0_11 = f[:, :det_space] @ P0[:det_space, :det_space] * tau  
        #P22
        delta_f_P22 = -P[det_space:, det_space:] @ f[det_space:, :] * tau
        #P0_22
        delta_f_P0_22 = f[:, det_space:] @ P0[det_space:, det_space:] * tau  
        #P12
        delta_f_P12 = -P[:det_space, det_space:] @ f[det_space:, :] * tau
        #P0_12
        delta_f_P0_12 = f[:, :det_space] @ P0[:det_space, det_space:] * tau
        #P21
        delta_f_P21 = -P[det_space:, :det_space] @ f[:det_space, :] * tau
        #P0_21
        delta_f_P0_21 = f[:, det_space:] @ P0[det_space:, :det_space] * tau
#        test_P21 = np.zeros((ndets, ndets))
#        test_P21[det_space:, :det_space] = P[det_space:, :det_space]
#        test_P0_21 = np.zeros((ndets, ndets))
#        test_P0_21[det_space:, :det_space] = P0[det_space:, :det_space]
#        test_delta_f = -test_P21 @ f * tau
#        #test_delta_f_P0 = f @ test_P0_11 * tau
#        test_delta_f_P0 = f @ test_P0_21  * tau
#        print("all of P0 is\n", P0)
#        print("test P21 is\n", -test_P21 * tau)
#        print("test delta f is\n", test_delta_f)
#        print("delta f is\n", delta_f_P21)
#        print("")
#        print("test P0_21 is\n", test_P0_21 * tau)
#        print("test delta f P0 is\n", test_delta_f_P0)
#        print("delta f P0 is \n", delta_f_P0_21)
#        exit()
        f[:det_space, :] += delta_f_P11 
        f[:, :det_space] += delta_f_P0_11
        f[det_space:, :] += delta_f_P22
        f[:, det_space:] += delta_f_P0_22
        f[:det_space, :] += delta_f_P12
        f[:, det_space:] += delta_f_P0_12
        f[det_space:, :] += delta_f_P21
        f[:, :det_space] += delta_f_P0_21
    elif semi_stochastic == False:
        delta_f = ((f @ P0) - (P @ f)) * tau
        #delta_f = ((P0 @ f) - (f @ P)) * tau
        f = f + delta_f

    if (inverse_temp_step) % 50 == 0:
        energy_estimate = (H @ f).trace()/f.trace()
        total_walkers = (np.abs(f)).sum()
        print(' {:<10}  {:> 0.12e}  {:> 0.12f}'.format(inverse_temp_step * tau, total_walkers, energy_estimate))
#print("exact answer:\n", fn.sum_of_states(eigen_val_exact, inverse_temp_step * tau))
