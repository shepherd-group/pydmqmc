import functions as fn
import numpy as np

H, Heval, HS = fn.system_initialize("EQUILIBRIUM-H4-STO3G.hamil")
eigen_val_exact, eigen_vect_exact = fn.FCI(Heval)

tau = 0.01
nreports = 1000
init_pop = 1
psi = np.zeros((HS,1))
psi[0,0] = init_pop
det_space = 3
semi_stochastic = True
'''
The propigator is the change is psi for a change in time is equal to 
the negative of our hamiltonian times our current wave function
delta_psi / delta_tau = -H * psi
delta_psi = -H * psi * delta_tau
psi(delat_tau + tau) = psi(tau) + delta_psi

Energy evaluation:
H*psi = E*psi
E = (H[0,:]*psi).sum()/psi[0,0]

Define two new hamiltonians the same dimensions as source hamiltonian
deterministic_hamil_1 will have some elements from the hamiltonian
deterministic_hamil_2 will have the rest of the elements
H = deterministic_hamil_1 + deterministic_hamil_2

'''
'''
det_hamil_1 = np.zeros((HS,HS))
det_hamil_2 = np.zeros((HS,HS))
det_hamil_1[:3,:3] = H[:3,:3]
det_hamil_2 = H - det_hamil_1
'''

for timestep in range(1, nreports + 1):
    ''' 
        
    if (timestep) % 50 == 0:
        energy_estimate_1 = (det_hamil_1[0,:] @ psi).sum()/psi[0,0]
        energy_estimate_2 = (det_hamil_2[0,:] @ psi).sum()/psi[0,0]
        total_energy_estimate = energy_estimate_1 + energy_estimate_2
        total_walkers = (np.abs(psi)).sum()
        print(' {:<10}  {:> 0.12e}  {:> 0.12f}'.format(timestep, total_walkers, total_energy_estimate))
    '''
    if semi_stochastic == True:
        #P11
        delta_psi_1 = -H[:det_space, :det_space] @ psi[:det_space, :] * tau 
        #P22
        delta_psi_2 = -H[det_space:, det_space:] @ psi[det_space:, :] * tau
        #P12
        delta_psi_3 = -H[:det_space, det_space:] @ psi[det_space:, :] * tau
        #P21
        delta_psi_4 = -H[det_space:, :det_space] @ psi[:det_space, :] * tau
        psi[:det_space, :] += delta_psi_1 
        psi[det_space:, :] += delta_psi_2
        psi[:det_space, :] += delta_psi_3
        psi[det_space:, :] += delta_psi_4
    elif semi_stochastic == False:
        delta_psi = -H @ psi * tau
        psi = psi + delta_psi

    if (timestep) % 50 == 0:
        energy_estimate = (Heval[0,:] @ psi).sum()/psi[0,0]
        total_walkers = (np.abs(psi)).sum()
        print(' {:<10}  {:> 0.12e}  {:> 0.12f}'.format(timestep, total_walkers, energy_estimate))

#print("the energy is: \n", (Heval[0,:] @ psi).sum()/psi[0,0])
#print("the exact is: \n", eigen_val_exact[0])
