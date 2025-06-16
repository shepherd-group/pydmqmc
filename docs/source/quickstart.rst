.. _quickstart:

Quickstart Guide
================

Each pydmqmc simulation has three components: the :doc:`System<systems>`,
the :doc:`simulation Method<methods>`, and
outputs derived from the simulation. These components are encapsulated
in three different submodules: :doc:`systems<api_systems>`, 
the :doc:`methods<api_methods>`,
and **not yet implemented**.
You'll need to instantiate an object from each submodule in order to execute
a full pydmqmc simulation.

All :doc:`System<systems>` classes require an input file. The contents 
of this file are used to define the System's Hamiltonian according to the
class used to load the file. For example, if you wish to use integrals in
`FCIDUMP format <https://hande.readthedocs.io/en/stable/manual/integrals.html#fcidump-format>`_,
use the :ref:`Integral<api-system-integral>` class.

The :doc:`Methods<methods>` classes require a System object at initialization.
The various methods fall into two categories: :doc:`iterative<iterative>`
and :doc:`analytic<analytic>`. All Method classes have a `.run()` method
that executes the simulation (or analytic calculation as the case may be). 
The iterative Methods, such as density-matrix quantum Monte Carlo (DMQMC)
also have a `.setup()` method.
As the name suggests, this class method is used to create the initial state
that the Method's `.run()` method will iterate on.

An example simulation with all three parts is below:

.. code-block:: python

    from pydmqmc.systems import Integral
    from pydmqmc.methods import SymmetricBlochDMQMC

    # Instantiate necessary objects
    sys = Integral("tests/inputs/integrals/STRICT-STO3G-STR-H4.FCIDUMP")
    mtd = SymmetricBlochDMQMC(sys, rng_seed=42)

    # Setup and run the simulation
    mtd.setup("uniform-random", n_particles=int(1e5))
    mtd.run(final_beta=25,
            dbeta=0.001,
            cycles_per_shift=1000,
            shift_dampening=0.05,
            spawn_cutoff=0.01,
            shift_by_rows=False
            )
