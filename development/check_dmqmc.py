# coding: utf-8
import numpy as np
from pydmqmc.systems import Integral
from pydmqmc.methods import InteractionPictureDMQMC

sys = Integral("../tests/inputs/integrals/STRICT-STO3G-STR-H4.FCIDUMP")
mtd = InteractionPictureDMQMC(sys, rng_seed=42)

dm_diag = np.load("initial_dm.npy")

mtd.setup(1.0, "fixed", fixed_diagonal=dm_diag)

# mtd.setup(final_beta=1.0,
#           initialization="random-grand-canonical",
#           spawn_cutoff=0.01,
#           n_particles=int(1e5))
# print("Ref:", dm_diag)
# print("Act:", np.diag(mtd.density_matrix))

mtd.run(dbeta=0.001,
        cycles_per_shift=10,
        shift_dampening=0.05,
        )

print("Trace", mtd.density_matrix.trace())
print("Energy", (mtd.density_matrix @ sys.hamiltonian).trace())
#print((mtd.density_matrix @ mtd.system.hamiltonian).trace())
