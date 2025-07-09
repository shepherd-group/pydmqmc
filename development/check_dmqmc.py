# coding: utf-8
import numpy as np
from pydmqmc.systems import Integral
from pydmqmc.methods import AsymmetricBlochDMQMC, SymmetricBlochDMQMC

sys = Integral("tests/inputs/integrals/H2-STO-3G-0.74Ang.fcidump")
mtd = SymmetricBlochDMQMC(sys, rng_seed=42)

#dm_diag = np.load("development/initial_dm.npy")

#mtd.setup("fixed", diag=dm_diag)

mtd.setup("random-uniform", n_particles=int(1e5))
# print("Ref:", dm_diag)
# print("Act:", np.diag(mtd.density_matrix))

mtd.run(final_beta=25,
        dbeta=0.001,
        cycles_per_shift=1000,
        shift_dampening=0.05,
        spawn_cutoff=0.01,
        ilevel=2
        )

print("Trace", mtd.density_matrix.trace())
print("Energy", (mtd.density_matrix @ sys.hamiltonian).trace())
#print((mtd.density_matrix @ mtd.system.hamiltonian).trace())
