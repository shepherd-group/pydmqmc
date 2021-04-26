
from modules import *
from functions import *

'''
    Send off the file name of the Hamiltonian
    We get back the:
        H:  Hamiltonian
        HS: Size of the Hilbert Space
        HF: Reference Energy
'''
H, HS, HF = build_hamiltonian('STRETCHED-H6-STO3G.hamil')
H, sorted_diags, sorted_hash = sort_index_by_diagonal(H, HS)
Htilde = unphysical_hamiltonian(H)

Heval = np.copy(H)
Htildeeval = np.copy(Htilde)

FCI, v = LA.eigh(Heval)
FCItilde, vtilde = LA.eigh(Htildeeval)

H = H - (np.eye(HS)*HF)
Htilde = Htilde - (np.eye(HS)*HF)

H0 = np.diag(np.diag(H))
H0tilde = np.diag(np.diag(Htilde))


# Parameters
shift = 0.0
cycles = 1
target = 7
tau = 0.1
reports = int(target/(tau*cycles))

Nrows = 100000000
seed = np.random.randint(2**32 - 1)
np.random.seed(seed)
beta_loops = 50
rowoutcomes = np.arange(0,HS)
rowweights = np.exp(-target*np.diag(Heval))
rowweights = rowweights / rowweights.sum()

for betaloop in range(1,beta_loops+1):

    randomrows = np.random.choice(rowoutcomes, size=Nrows, p=rowweights)
    randomrows = np.unique(randomrows)

    allrandomrows.append(len(randomrows))
    
    f = np.zeros((HS,HS))
    for row in randomrows:
        f[row,row] = np.exp(-target*H[row,row])
    
    ftilde = np.copy(f)
    
    iteration = 0
    report = 0
    
    expectation(hamil, dm, observable)
    
    for report in range(reports):
    
        for cycle in range(cycles):
            iteration += 1
    
            '''
                "IP-DMQMC"
    
                df/dtau = H0 @ f - f @ H
            '''
    
            df = tau * ( (H0 @ f) - (f @ H) )
            
            f += df
    
            dftilde = tau * ( (Htilde0 @ ftilde) - (ftilde @ Htilde) )
    
            ftilde += dftilde
    
df =    {
        'Beta Loop':loops,
        'e^{-beta H}':phy_energies,
        'e^{-beta Htilde}':unp_energies,
        'e^{-beta H0}':initphy_energies,
        'e^{-beta Htilde0}':initunp_energies,
        'Tr(H@rho)':phy_num,
        'Tr(Htilde@rhotilde)':unp_num,
        'Tr(rho)':phy_den,
        'Tr(rhotilde)':unp_den,
        'Unique Rows':allrandomrows,
        }

df = pd.DataFrame(df)
name  = 'seed'+str(seed)
name += '-'+str(beta_loops)+'loops'
name += '-'+str(Nrows)+'Nrows-Weighted-Choice'
name += '-analytical-IPDMQMC-STRH6STO3G.csv'
df.to_csv(name, index=False)
print('Saved data to:', name)

