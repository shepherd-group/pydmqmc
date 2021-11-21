import functions as fn
import numpy as np

np.set_printoptions(linewidth=10000)

P, H, ndets = fn.system_initialize("EQUILIBRIUM-H4-STO3G.hamil")
eigen_val_exact, eigen_vect_exact = fn.FCI(H)

tau = 0.01
nreports = 5000
init_pop = 1
psi = np.zeros((ndets,1))
psi[0,0] = init_pop
'''
The propigator is the change is psi for a change in time is equal to 
the negative of our hamiltonian times our current wave function
delta_psi / delta_tau = -P * psi
delta_psi = -P * psi * delta_tau
psi(delat_tau + tau) = psi(tau) + delta_psi

Energy evaluation:
P*psi = E*psi
E = (P[0,:]*psi).sum()/psi[0,0]

Define two new hamiltonians the same dimensions as source hamiltonian
deterministic_hamil_1 will have some elements from the hamiltonian
deterministic_hamil_2 will have the rest of the elements
P = deterministic_hamil_1 + deterministic_hamil_2

'''
#print("Pi3\n", P[:,3])
#print("\n", [str(i) + ",3" for i in np.arange(ndets)])
#exit()
for timestep in range(1, nreports + 1):
    sum_Hii_ci = np.zeros((ndets, 1))
    sum_Hij_ci = np.zeros((ndets, 1))

    for i in range(ndets):
        sum_Hii_ci[i,0] += P[i,i] * psi[i,0] * -tau

        for j in range(ndets):
            if i == j:
                continue
            #sum_Hij_ci[i,0] += P[i,j] * psi[j,0] * -tau
            #sum_Hij_ci[i,0] += (P[i,j] * np.abs(P[i,j]) / (ndets)) * psi[j,0] * -tau
            #sum_Hij_ci[i,0] += (P[i,j] * ndets) * psi[j,0] * -tau
            #sum_Hij_ci[i,0] += (P[i,j] * ndets / (np.sum(np.abs(P[:,j])))) * psi[j,0] * -tau
            #sum_Hij_ci[i,0] += (P[i,j] * ndets / (np.sum(np.abs(P[j,:])))) * psi[j,0] * -tau
            #sum_Hij_ci[i,0] += (P[i,j] / (ndets * (np.sum(np.abs(P[j,:]))))) * psi[j,0] * -tau
            #sum_Hij_ci[i,0] += (P[i,j] / (ndets * (np.sum(np.abs(P[:,j]))))) * psi[j,0] * -tau
            #sum_Hij_ci[i,0] += (P[i,j] * (np.sum(np.abs(P[:,j])))) / ndets * psi[j,0] * -tau
            #sum_Hij_ci[i,0] += (P[i,j] * (np.sum(np.abs(P[j,:])))) / ndets * psi[j,0] * -tau
            sum_Hij_ci[i,0] += (P[i,j] / (ndets * (1 / np.sum(-P[j,j] + np.abs(P[:,j]))))) * psi[j,0] * -tau
    
    delta_psi = sum_Hii_ci + sum_Hij_ci
    psi += delta_psi
    
    if (timestep) % 50 == 0:
        energy_estimate = (H[0,:] @ psi).sum()/psi[0,0]
        total_walkers = (np.abs(psi)).sum()
        print(' {:<10}  {:> 0.12e}  {:> 0.12f}'.format(timestep, total_walkers, energy_estimate))

#    check_delta_psi = -P @ psi * tau
#    print("delta_psi\n", delta_psi)
#    print("check\n", check_delta_psi)
#    exit()
#print("the energy is: \n", (H[0,:] @ psi).sum()/psi[0,0])
print("the exact is: \n", eigen_val_exact[0])
