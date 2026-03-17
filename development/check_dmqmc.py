# coding: utf-8
import numpy as np
from pydmqmc.systems import Integral
from pydmqmc.methods import AsymmetricBlochDMQMC

sys = Integral("tests/inputs/integrals/STRICT-STO3G-STR-H4.FCIDUMP")
mtd = AsymmetricBlochDMQMC(sys, rng_seed=42, parallel=True)

mtd.setup(25, "random-uniform", n_particles=int(1e5))

mtd.run(dbeta=0.001,
        cycles_per_shift=1000,
        shift_dampening=0.05,
        spawn_cutoff=0.01,
        shift_by_rows=False,
        update_method="rk4"
        )

print("Trace", mtd.density_matrix.trace())
print("Energy", (mtd.density_matrix @ sys.hamiltonian).trace())
#print((mtd.density_matrix @ mtd.system.hamiltonian).trace())
