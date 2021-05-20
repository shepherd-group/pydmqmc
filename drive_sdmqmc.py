#!/usr/bin/env python
import functions as fn
import numpy as np

P, H, ndets = fn.system_initialize("EQUILIBRIUM-H4-STO3G.hamil")
eigen_val_exact, eigen_vect_exact = fn.FCI(H)

target_beta = 10
tau = 0.01
nreports = int(target_beta / tau)
init_pop = 1
rho = np.eye(ndets)
det_space = 15
semi_stochastic = True
symmetric = True
'''
the asymmetric propagator
the change in the density matrix per change in temperature
is equal to the negative of the hamiltonian time the density matrix

delta_rho / delta_tau = -H @ rho
delta_rho = -H @ rho * delta_tau

the symmetric propagator
the change in the density matrix per change in temperature 
is equal to negative one half times the sum of 
the hamiltonian times the density matrix
and the density matrix times the hamiltonian

delta_rho / delta_tau = -1/2 * ((H @ rho) + (rho @ H))
delta_rho = -1/2 * ((H @ rho) + (rho @ H)) * delta_tau

Note: in the equations above where we have written H
the code would use P the propogator.
'''
for inverse_temp_step in range(1, nreports + 1):
    if symmetric == False:
#        np.set_printoptions(linewidth=10000)
        if semi_stochastic == True:
#            delta_rho = -P @ rho * tau
 #           rho = rho + delta_rho
  #          if inverse_temp_step < 4:
   #             continue
            #P11
            delta_rho_1 = -P[:det_space, :det_space] @ rho[:det_space, :] * tau 
            #P22
            delta_rho_2 = -P[det_space:, det_space:] @ rho[det_space:, :] * tau
            #P12
            delta_rho_3 = -P[:det_space, det_space:] @ rho[det_space:, :] * tau
            #P21
            delta_rho_4 = -P[det_space:, :det_space] @ rho[:det_space, :] * tau
            #temp_check = np.zeros((ndets,ndets))
            #temp_check[det_space:, :det_space] = P[det_space:, :det_space]
            #temp_delta_rho = -(temp_check @ rho) * tau
                #print("temp_check\n", temp_check)
            #print("temp_delta_rho\n", temp_delta_rho)
            #print("delta_rho\n", delta_rho_4)
            #print("rho + delta_rho\n", rho)
            #exit()
            rho[:det_space, :] += delta_rho_1 
            rho[det_space:, :] += delta_rho_2
            rho[:det_space, :] += delta_rho_3
            rho[det_space:, :] += delta_rho_4
        elif semi_stochastic == False:
            delta_rho = -P @ rho * tau
            rho = rho + delta_rho
    elif symmetric == True:
        if semi_stochastic == True:
#            np.set_printoptions(linewidth=10000)
  #          delta_rho = -0.5 * tau * ((P @ rho) + (rho @ P))
   #         rho = rho + delta_rho
    #        if inverse_temp_step < 4:
     #           continue
            #P11
            delta_rho_1a = -0.5 * tau * (P[:det_space, :det_space] @ rho[:det_space, :])
            delta_rho_1b = -0.5 * tau * (rho[:, :det_space] @ P[:det_space, :det_space])
            #P22
            delta_rho_2a = -0.5 * tau * (P[det_space:, det_space:] @ rho[det_space:, :])
            delta_rho_2b = -0.5 * tau * (rho[:, det_space:] @ P[det_space:, det_space:])
            #P12
            delta_rho_3a = -0.5 * tau * (P[:det_space, det_space:] @ rho[det_space:, :]) 
            delta_rho_3b = -0.5 * tau * (rho[:, :det_space] @ P[:det_space, det_space:])
            #P21
            delta_rho_4a = -0.5 * tau * (P[det_space:, :det_space] @ rho[:det_space, :])
            delta_rho_4b= -0.5 * tau * (rho[:, det_space:] @ P[det_space:, :det_space])
#            test_P = np.zeros((ndets, ndets))
 #           test_P[det_space:, det_space:] = P[det_space:, det_space:]
  #          print("zeros + P22 is\n", test_P)
   #         print("rho is\n", rho)
    #        test_1 = test_P @ rho
     #       test_2 = rho @ test_P
      #      print("P@rho is\n", test_1)
       #     print("rho@P is\n", test_2)
        #    temp_delta_rho = -0.5 * tau * ((test_1 @ rho) + (rho @ test_2))
         #   print("temp_delta_rho is\n", temp_delta_rho)
          #  print("rho + temp_delta_rho\n", rho + temp_delta_rho)
           # print("delta_rho_3\n", delta_rho_3)
#            print("rho[:det_space, det_space:] += delta_rho_3\n", rho[:det_space, det_space:] + delta_rho_3)
 #           exit()         
            rho[:det_space, :] += delta_rho_1a
            rho[:, :det_space] += delta_rho_1b
            rho[det_space:, :] += delta_rho_2a
            rho[:, det_space:] += delta_rho_2b
            rho[:det_space, :] += delta_rho_3a
            rho[:, det_space:] += delta_rho_3b
            rho[det_space:, :] += delta_rho_4a
            rho[:, :det_space] += delta_rho_4b
        elif semi_stochastic == False:
            delta_rho = -0.5 * tau * ((P @ rho) + (rho @ P))
            rho = rho + delta_rho
    if (inverse_temp_step) % 50 == 0:
        energy_estimate = (H @ rho).trace()/rho.trace()
        total_walkers = (np.abs(rho)).sum()
        print(' {:<10}  {:> 0.12e}  {:> 0.12f}'.format(inverse_temp_step * tau, total_walkers, energy_estimate))
#print("exact answer:\n", fn.sum_of_states(eigen_val_exact, inverse_temp_step * tau))
