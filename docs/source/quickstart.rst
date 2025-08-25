.. _quickstart:

Quickstart Guide
================

Each pydmqmc simulation has two components: the :ref:`System <ref-systems>`
being simulated and the simulation :ref:`Method <ref-methods>`.
These components are encapsulated
in two different submodules: :doc:`Systems API<api/systems/system>`, 
the :doc:`Methods API<api/methods/index>`.
You'll need to instantiate an object from each submodule in order to execute
a full pydmqmc simulation. See the additional pages within this documentation
to better understand your options.

All :class:`~pydmqmc.systems.System` classes require an input file. The contents 
of this file are used to define the System's Hamiltonian according to the
class used to load the file. For example, if you wish to use integrals in
`FCIDUMP format <https://hande.readthedocs.io/en/stable/manual/integrals.html#fcidump-format>`_,
use the :class:`~pydmqmc.systems.Integral` class.

The :class:`~pydmqmc.methods.Method` classes require a System object at initialization.
The various methods fall into two categories: :class:`~pydmqmc.methods.Iterative`
and :class:`~pydmqmc.methods.Analytic`. All Method classes have a 
:meth:`~pydmqmc.methods.Method.run()` method
that executes the simulation (or analytic calculation as the case may be). 
The iterative Methods, such as density-matrix quantum Monte Carlo (DMQMC)
also have a :meth:`~pydmqmc.methods.Iterative.setup` method.
As the name suggests, this class method is used to create the initial state
that the Method's :meth:`~pydmqmc.methods.Method.run()` method will iterate on.

An example simulation with all three parts is below:

.. code-block:: python

    from pydmqmc.systems import Integral
    from pydmqmc.methods import SymmetricBlochDMQMC

    # Instantiate necessary objects
    sys = Integral("tests/inputs/integrals/STRICT-STO3G-STR-H4.FCIDUMP")
    mtd = SymmetricBlochDMQMC(sys, rng_seed=42)

    # Setup and run the simulation
    mtd.setup("random-uniform", n_particles=int(1e5))
    mtd.run(final_beta=25,
            dbeta=0.001,
            cycles_per_shift=1000,
            shift_dampening=0.05,
            spawn_cutoff=0.01,
            shift_by_rows=False
            )

    mtd.save_data("test_run")

This will result in two files being saved to disk. The file ``test_run_report.csv``
contains information from each iteration of the 
:class:`~pydmqmc.methods.SymmetricBlochDMQMC` simulation; for example:

.. csv-table::

    beta,trace,energy
    0.0,100000.0,-103794.00723214602
    1.0,45926.587906175104,-61008.49488132428
    2.0,27454.474366465434,-42865.7113992343

The second file, ``test_run_density_matrix.csv``, contains the final density matrix.

Both files can have their format adjusted; 
see :meth:`~pydmqmc.methods.SymmetricBlochDMQMC.save_data` for more.
