import functions as fn
import numpy as np

P, H, ndets = fn.system_initialize("EQUILIBRIUM-H4-STO3G.hamil")
eigen_val_exact, eigen_vect_exact = fn.FCI(H)

target_beta = 10
tau = 0.01
nreports = int(target_beta / tau)
init_pop = 1
rho = np.eye(ndets)
det_space = 3
semi_stochastic = True
'''
the change in the density matrix per change in temperature
is equal to the negative of the hamiltonian time the density matrix

delta_rho / delta_tau = -H * rho
delta_rho = -H * rho * delta_tau

'''

for inverse_temp_step in range(1, nreports + 1):

    if semi_stochastic == True:
        '''
        #P11
        delta_rho_1 = -P[:det_space, :det_space] @ rho[:det_space, :det_space] * tau 
        #P22
        delta_rho_2 = -P[det_space:, det_space:] @ rho[det_space:, det_space:] * tau
        #P12
        delta_rho_3 = -P[:det_space, det_space:] @ rho[det_space:, :] * tau
        #P21
        delta_rho_4 = -P[det_space:, :det_space] @ rho[:det_space, :] * tau
        '''
        #P11
        delta_rho_1 = -P[:det_space, :] @ rho[:, :det_space] * tau 
        #P22
        delta_rho_2 = -P[det_space:, :] @ rho[:, det_space:] * tau
        #P12
        delta_rho_3 = -P[:det_space, :] @ rho[:, det_space:] * tau
        #P21
        delta_rho_4 = -P[det_space:, :] @ rho[:, :det_space] * tau
        rho[:det_space, :det_space] += delta_rho_1 
        rho[det_space:, det_space:] += delta_rho_2
        rho[:det_space, det_space:] += delta_rho_3
        rho[det_space:, :det_space] += delta_rho_4
    elif semi_stochastic == False:
        delta_rho = -P @ rho * tau
        rho = rho + delta_rho

    if (inverse_temp_step) % 50 == 0:
        energy_estimate = (H @ rho).trace()/rho.trace()
        total_walkers = (np.abs(rho)).sum()
        print(' {:<10}  {:> 0.12e}  {:> 0.12f}'.format(inverse_temp_step * tau, total_walkers, energy_estimate))
print("exact answer:\n", fn.sum_of_states(eigen_val_exact, inverse_temp_step * tau))
