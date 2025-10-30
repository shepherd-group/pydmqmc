.. _methods_dmqmc:

Density-Matrix Quantum Monte Carlo Methods
==========================================

.. contents::
    :local:
    :depth: 2

Introduction
------------

The Density-Matrix Quantum Monte Carlo (DMQMC) :footcite:`Blunt2014` methods
stochastically sample the N-body thermal density matrix :math:`\hat{\rho} = e^{-\beta \hat{H}}`.
This sampling allows exact thermodynamic properties to be computed for many-body systems
at finite temperature.

The density matrix :math:`\hat{\rho}` is represented in a discrete basis of Slater determinants.
The matrix elements are stochastically sampled using a population of signed "psi-particles" or "psips".
The initial population of psips is initialized at :math:`\beta = 0`. As :math:`\beta` is increased,
the psips are propagated throughout the density matrix following a process of 
spawning, death, cloning, and annihilation events. Each of these actions has a probability
determined by the Hamiltonian and current state of the density matrix as sampled by the psips.

The relationship :math:`\partial\hat{\rho}/\partial\beta` between the density matrix and the inverse
temperature can be constrained by either the symmetric or asymmetric Bloch equations. Both approaches
are defined by their own classes in pydmqmc: :class:`~pydmqmc.methods.SymmetricBlochDMQMC` and
:class:`~pydmqmc.methods.AsymmetricBlochDMQMC`. The way to use these classes is the same and will be
covered below.

The pydmqmc library also offers Interaction Picture DMQMC (IP-DMQMC) :footcite:`Malone2015`.
As the name suggests, in this variation the density matrix :math:`\hat{\rho}` is sampled in the interaction picture.
This improves sampling---and statistical accuracy--for weakly correlated systems.
The :class:`~pydmqmc.methods.InteractionPictureDMQMC` class enables use of IP-DMQMC.

Summary of Available Classes
++++++++++++++++++++++++++++

Click on any entry for more details about the API or follow the guidelines below for setting up and
running a DMQMC simulation.

* :class:`~pydmqmc.methods.SymmetricBlochDMQMC` - Performs DMQMC using the symmetric Bloch equation.
* :class:`~pydmqmc.methods.AsymmetricBlochDMQMC` - Performs DMQMC using the asymmetric Bloch equation.
* :class:`~pydmqmc.methods.InteractionPictureDMQMC` - Performs DMQMC in the interaction picture.

.. _setup-dmqmc:

Setting up a DMQMC Simulation
-----------------------------

Before starting with a DMQMC system, you must instantiate your desired :ref:`System <ref-systems>`
object. Once your system has been defined, you can create an instance of your desired DMQMC method.
For this example, we'll use :class:`~pydmqmc.methods.SymmetricBlochDMQMC` but this procedure is the
same for all DMQMC classes listed above:

.. code-block:: python

    import pydmqmc.systems
    from pydmqmc.methods import SymmetricBlochDMQMC

    # load your system file with the appropriate class
    sys = pydmqmc.systems...(...)

    # use your system to instantiate your desired DMQMC method
    mtd = SymmetricBlochDMQMC(sys)

Once your desired DMQMC method has been instantiated, you can initialize the density matrix with the
``setup()`` method. This method is used to define the final inverse temperature :math:`\beta` for the
simulation, the framework for defining the initial density matrix, and possible additional
parameters for this initialization depending on the method chosen:

.. code-block:: python

    mtd.setup(
        final_beta = 0.75,
        initialization = "deterministic",
    )

The :class:`~pydmqmc.methods.SymmetricBlochDMQMC` and :class:`~pydmqmc.methods.AsymmetricBlochDMQMC`
methods both offer the same set of frameworks for the initial density matrix: 
the identity matrix (``"deterministic"``),
randomly placing a number of psips along the diagonal following a uniform distribution (``"random-uniform"``),
and a user-supplied initial diagonal (``"fixed"``).
The :class:`~pydmqmc.methods.InteractionPictureDMQMC` method also offers ``"deterministic"`` and
``"random-uniform"`` options but uses the Hartree-Fock thermal weights instead of integer psips. It also offers A
``"random-thermal"`` framework where the diagonal is sampled with probabilities proportional
to the thermal weights and the ``"random-grand-canoncial"`` framework that uses the grand canonical density matrix
to inform sampling.
See the appropriate API documentation for more details on these options and the additional parameters required.

Once the density matrix has been set up, you can view it with the ``density_matrix`` attribute:

.. code-block:: python

    print(mtd.density_matrix)

You can also double check the final :meth:`\beta` that was specified with the ``final_beta`` attribute:

.. code-block:: python

    print(mtd.final_beta)

The ``setup()`` method can be called any number of times so long as the ``run()`` method has not been invoked.
This allows you to change either the final :meth:`\beta` or initial density matrix before running the simulation.
Neither attribute can be changed once the simulation has been run to prevent loss of data.


.. _running-dmqmc:

Running a DMQMC Simulation
--------------------------

Once the density matrix and target inverse temperature have been specified with the ``setup()`` method,
the ``run()`` method can be used to execute the actual DMQMC simulation. The ``run()`` method follows the
same basic structure for all of the DMQMC classes; see their APIs for more detail.
At minimum, all that's needed to run a simulation is to specify
three parameters that control the stability of the iterations as detailed in the next section:

.. code-block:: python

    mtd.run(
        dbeta=0.001,
        cycles_per_shift=10,
        shift_dampening=0.05
    )

This will use the default fields for the :ref:`iteration_report`.

Integration Control
+++++++++++++++++++

DMQMC simulations start from an inverse temperature of :math:`\beta = 0` and increase it gradually to the
final :math:`\beta` specified during setup. Currently this is accomplished using a fixed step size :math:`\Delta\beta`
which is set by the required parameter ``dbeta``.

In DMQMC, an energy shift :math:`S` is applied to the Hamiltonian to keep the normalization approximately constant.
This shift is updated every :math:`n` cycles where :math:`n` is set by the required ``cycles_per_shift`` parameter.
How much the shift is allowed to vary with each update is controlled by the required ``shift_dampening`` parameter,
which maps to the variable :math:`\zeta` in Eqn 16 of the DMQMC paper :footcite:`Blunt2014`. The shift can
either be specified for the Hamiltonian as a whole or independently for each row of Hamiltonian if ``shift_by_rows``
is ``True``.

Since DMQMC is integrating :math:`\partial\hat{\rho}/\partial\beta` to obtain the final density matrix :math:`\hat{\rho}`,
you can specify the integration method with the ``update_method`` parameter. By default this parameter is set to "euler"
but any of the integration methods recognized by :meth:`pydmqmc.methods.Iterative.parse_method` are supported.

.. note:: 
    Claire hasn't actually tested that RK4 with DMQMC, though she's tested both parts independently.

.. _initiator-approximations:

Psip Spawning Control
+++++++++++++++++++++

.. note::
    This text is Claire's best effort to understand these approximation methods 
    based on conversations with Will. Corrections are welcome.

Several of the optional parameters for ``run()`` control how psips are spawned. The basic rules for
spawning are as follows.

For a given site :math:`\rho_{ij}` attempting to spawn a psip at :math:`\rho_{ik}`:

1. If :math:`\rho_{ik} \neq 0`, always let :math:`\rho_{ij}` spawn psips.
2. If :math:`\rho_{ik} = 0`, let :math:`\rho_{ij}` spawn if :math:`|\rho_{ij}| \ge \mathtt{n\_add}`.
3. If neither of the above are satisfied but another :math:`\rho_{pq}`, where :math:`p \neq i` and :math:`q \neq j`, spawns psips at :math:`\rho_{ik}` with the *same sign*, then :math:`\rho_{ij}` may spawn as well.
4. If none of the above are satisfied, nullify the spawn from :math:`\rho_{ij}` to :math:`\rho_{ik}`.
5. If the resulting :math:`\rho_{ik} \le \mathtt{spawn\_cutoff}`, nullify the spawn.

.. warning::
    Claire isn't sure rule 3 is currently implemented in the code.

The **initiator approximation** is represented by step 2 and may be disabled by setting ``n_add = None`` (:math:`n_\mathrm{add} = 0`).
This allows all spawns from :math:`\rho_{ik}`.

Similarly, the lower limit on psip spawning represented by step 5 can be disabled by setting ``spawn_cutoff = 0``.

The ``ilevel`` parameter adds the **initiator level approximation**, which modifies the above rules
to include another rule before 4:
if the number of excitations between the determinant labels for :math:`i` and :math:`j` is
less than or equal to the supplied level, allow the spawn at :math:`\rho_{ik}`.
*Currently, only initiator level zero is supported.*

References
----------
.. footbibliography::