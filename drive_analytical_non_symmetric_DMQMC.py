
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
Heval = np.copy(H)
H = H - (np.eye(HS)*HF)

FCI, v = LA.eigh(Heval)

# "QMC" Parameters
shift = 0.0
cycles = 1
target = 10
tau = 0.1
reports = int(target/(tau*cycles))
beta_loops = 1

for betaloop in range(1,beta_loops+1):

    d = np.eye(HS)
    iteration = 0
    report = 0
    write_report(iteration, tau, shift, d, Heval, df=None, printbool=True)
    
    for report in range(reports):
    
        for cycle in range(cycles):
            iteration += 1
    
            '''
                "DMQMC" non-symmetric propagator:
    
                dp/dtau = -H @ f
            '''
    
            deltad = -tau * (H @ d)
            
            d += deltad
    
        write_report(iteration, tau, shift, d, Heval, df=None, printbool=True)

