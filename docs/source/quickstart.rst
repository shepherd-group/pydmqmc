.. _quickstart:

Quickstart Guide
================

This guide will get you started using the pydmqmc library to write your own scripts.
If you already feel comfortable with object oriented programming in Python,
you can jump to :ref:`use-in-scripts`. Otherwise, this guide will introduce you to
the design philosophy that pydmqmc is built around.


The Structure of pydmqmc
------------------------

The pydmqmc module contains several "submodules" or "sublibraries"
each dedicated to a different aspect of a simulation or calculation.
The two most important are the :ref:`System submodule<ref-systems>`,
which is used to define the system being studied, and the 
:ref:`Method submodule<ref-methods>`, which defines the calculation
to be executed. You will need elements of both submodules to complete
a pydmqmc calculation. Details about which systems and methods are supported
are detailed in the respective documentation for each submodule.
In particular, note that Methods come in two flavors: Analytic and Iterative.

There is also a :ref:`utility submodule<api-utils>` that contains
functions used by the Systems and Methods submodules. These are available
for your own use if you desire, but direct invocation of these functions
is not necessary for most pydmqmc applications.


.. _oop-primer:

Primer on Object Oriented Programming
-------------------------------------

The pydmqmc library relies heavily on object oriented programming,
a programming paradigm where code is organized into **objects.**
These abstract objects have data associated with them (called **attributes**)
as well as functions that operate on that data (called **methods**).
In this way, object oriented programming leads to software that is
focused on data (where it is stored, how it can be interacted with, etc).

Objects have an associated **type** that defines its attributes and methods.
Some types are inherent to Python, such as integers and lists. Others may
be provided by libraries such as pydmqmc in the form of **classes**.
The :ref:`Systems<ref-systems>` and :ref:`Methods<ref-methods>`
submodules provide many such classes; for example,
we can import the :class:`~pydmqmc.systems.Integral` class from the
Systems submodule for use in our scripts:

.. code-block:: python

    from pydmqmc.systems import Integral

Classes define the *rules* for an object (what data it should contain and how
that data can be interacted with), but usually contain no actual data themselves.
Instead, **instances** of classes are the objects we actually manipulate in our programs.
To create an instance of (or **instantiate**) an object, we must **initialize** it. In Python,
objects are initialized by using their class name followed by parentheses.
A class may require certain parameters to be defined in order for an object to be created.
As with Python functions, some parameters may be optional.
The :class:`~pydmqmc.systems.Integral` class has one required parameter --- the input file
that contains the necessary data (see :ref:`integral-systems` for more information
on what file type is expected):

.. code-block:: python

    # Instantiate an Integral object using its one required parameter
    my_system = Integral("tests/inputs/integrals/STRICT-STO3G-STR-H4.FCIDUMP")

    # Native Python types like dictionaries and floating point numbers 
    # can be instantiated in the same way!
    x = float(4)
    empty_dict = dict()

Our :class:`~pydmqmc.systems.Integral` object, called ``my_system``, has several attributes
associated with it. These are accessible with "dot notation" as shown below:

.. code-block:: python

    print(my_system.n_electrons)

We can use the same notation to invoke the :class:`~pydmqmc.systems.Integral` methods,
such as the method for generating a Hamiltonian matrix. The only difference is the
inclusion of parentheses:

.. code-block:: python

    my_system.generate_hamiltonian()

A full list of attributes and methods (collectively called **members**) 
is visible in the API reference for :class:`~pydmqmc.systems.Integral`.
Note that only the **public** members will be shown; classes may also have
**private** members that are visible only within the source code. Such members
are not accessible outside of the class itself and are designated with a leading underscore.

One last note about object oriented programming: classes can be used to define new, slightly
different classes via **inheritance.** The benefit of inheritance is that it allows code from
one class to be reused in another class. You'll see mentions of inheritance all throughout the
:ref:`api-reference`. There, you will also encounter the concept of **base classes**: classes
that aren't intended to be used as standalone objects but instead only exist for the purposes
of class inheritance. Base classes ensure that all **child classes** have a common set of
attributes and methods.

The pydmqmc library makes extensive use of classes and object oriented programming in order
to fulfill its :ref:`dev-philosophy`. In particular, you'll make heavy use of methods
in your scripts, as shown in the next section.

.. _use-in-scripts:

Using pydmqmc
-------------

Let's put together a script that runs a density-matrix quantum Monte Carlo (DMQMC)
simulation for a system defined by an `FCIDUMP`_ file. Specifically, let's
evolve our DMQMC simulation using the symmetric Bloch equation.

.. _FCIDUMP: https://hande.readthedocs.io/en/stable/manual/integrals.html#fcidump-format

From the documentation on available :ref:`Systems<ref-systems>`, we can see that
the :class:`~pydmqmc.systems.Integral` class is the best fit for our needs.
All Systems require an input file and the :class:`~pydmqmc.systems.Integral` class
supports the `FCIDUMP`_ file format.

Likewise, the :ref:`Methods<ref-methods>` documentation reveals there is a
:class:`~pydmqmc.methods.SymmetricBlochDMQMC` class that can perform a DMQMC
simulation using the symmetric Bloch equation. This class is of the Iterative type.

All Method classes require a System object at initialization and have a
:meth:`~pydmqmc.methods.SymmetricBlochDMQMC.run`
method for executing the calculation. Iterative calculations like
:class:`~pydmqmc.methods.SymmetricBlochDMQMC` also have a
:meth:`~pydmqmc.methods.SymmetricBlochDMQMC.setup` method
that is used to set up the calculation's initial state. In the case of
:class:`~pydmqmc.methods.SymmetricBlochDMQMC`, :meth:`~pydmqmc.methods.SymmetricBlochDMQMC.setup`
does things like construct the initial density matrix.

Once a calculation is finished, the Method's :meth:`~pydmqmc.methods.SymmetricBlochDMQMC.save_data`
method can be used to save any resulting data products to disk. Multiple file formats are supported.

An example simulation with all three parts is below:

.. code-block:: python

    from pydmqmc.systems import Integral
    from pydmqmc.methods import SymmetricBlochDMQMC

    # Instantiate necessary objects
    sys = Integral("tests/inputs/integrals/STRICT-STO3G-STR-H4.FCIDUMP")
    mtd = SymmetricBlochDMQMC(sys, rng_seed=42)

    # Setup and run the simulation
    mtd.setup(
        final_beta=25,
        initialization="random-uniform",
        n_particles=int(1e5)
    )
    mtd.run(
        dbeta=0.001,
        cycles_per_shift=1000,
        shift_dampening=0.05,
        spawn_cutoff=0.01,
        shift_by_rows=False
    )

    mtd.save_data("test_run")

This will result in two files being saved to disk. 
The file ``test_run_density_matrix.csv`` contains the final density matrix.
The second file, ``test_run_report.csv``,
contains information from each iteration of the 
:class:`~pydmqmc.methods.SymmetricBlochDMQMC` simulation; for example:

.. csv-table::

    beta,trace,energy
    0.0,100000.0,-103794.00723214602
    1.0,45926.587906175104,-61008.49488132428
    2.0,27454.474366465434,-42865.7113992343

The quantities included in this table can be adjusted; see :ref:`iteration-report` for more.

Can I Reuse System Objects?
***************************

Let's say you have two very similar systems you want to apply the same method to.
Can you use the same system object in your script and just update one or more of
it's attributes, like its input file?

The short answer is **no**. Each System object represents a unique physical system;
changing even one property would mean it is now a new physical system. Therefore,
pydmqmc requires that you make a new object to represent this new system.

For example, the proper way to create systems with multiple input files would be the following:

.. code-block:: python

    from pydmqmc.systems import MatrixHamiltonian

    sys_eq = MatrixHamiltonian("tests/inputs/hamiltonians/EQUILIBRIUM-H6-STO3G.hamil")
    sys_stretch = MatrixHamiltonian("tests/inputs/hamiltonians/STRETCHED-H6-STO3G.hamil")

Can I Reuse Method Objects with Different Systems?
**************************************************

Every Method object must have a System associated with it, as seen in the example script above.
There is no way to update which System is associated with a given Method object; instead,
**a unique Method object must be created for each unique System.** This keeps data management
straightforward for the Method object.

Extending the example above, where we want to run the same method on many similar systems,
you would need to write your script like this:

.. code-block:: python

    from pydmqmc.methods import FullConfigurationInteraction

    fci_eq = FullConfigurationInteraction(sys_eq)
    fci_stretch = FullConfigurationInteraction(sys_stretch)

    fci_eq.run()
    fci_stretch.run()

    fci_eq.save_data("FCI_EQUILIBRIUM-H6-STO3G")
    fci_stretch.save_data("FCI_STRETCHED-H6-STO3G")

Alternatively, you can use a for-loop:

.. code-block:: python

    from pydmqmc.methods import FullConfigurationInteraction
    from os.path import basename, splitext

    for system in [sys_eq, sys_stretch]:

        # From a path like tests/inputs/hamiltonians/EQUILIBRIUM-H6-STO3G.hamil,
        # extract just the EQUILIBRIUM-H6-STO3G part.
        input_basename = splitext(basename(system.input_file))[0]

        mtd = FullConfigurationInteraction(system)
        mtd.run()
        mtd.save_data("FCI" + input_basename)

Can I Reuse Method Objects for Multiple Monte Carlo Realizations?
*****************************************************************

Since Monte Carlo methods like DMQMC are stochastic, it may be desirable
to run these methods multiple times with different random seeds and
later analyze the variation between runs. **Unique Method objects
will be needed for each calculation.**

To prevent data loss, each Method object can only have its
:meth:`~pydmqmc.methods.Method.run` method called once.
This is true even though some methods like those descended from
:class:`~pydmqmc.methods.DensityMatrixQMC` let you reset the seed.
Additionally, the :meth:`~pydmqmc.methods.Iterative.setup` method for
:class:`~pydmqmc.methods.Iterative`-descended classes can be called
any number of times *before* :meth:`~pydmqmc.methods.Iterative.run`
is called.

To run multiple realizations of something like DMQMC,
you'll need multiple objects:

.. code-block:: python

    from pydmqmc.systems import Integral
    from pydmqmc.methods import SymmetricBlochDMQMC

    # Instantiate necessary objects
    sys = Integral("tests/inputs/integrals/STRICT-STO3G-STR-H4.FCIDUMP")
    sys.generate_hamiltonian()  # Not strictly necessary as is called during method init

    for seed in range(40, 50):

        # Create a unique method object
        mtd = SymmetricBlochDMQMC(sys, rng_seed=seed)

        # Setup and run the simulation
        mtd.setup(
            final_beta=25,
            initialization="random-uniform",
            n_particles=int(1e5)
        )
        mtd.run(
            dbeta=0.001,
            cycles_per_shift=1000,
            shift_dampening=0.05,
            spawn_cutoff=0.01,
            shift_by_rows=False
        )

        mtd.save_data(f"seed_{seed}")


Navigating the Source Code
--------------------------

All information about what classes and their methods require as parameters is available
in the :ref:`api-reference`. That said, sometimes it is useful to look at the source code
to "see the math in action," particularly when learning methods like DMQMC for the first time.

Because of pydmqmc's reliance on class inheritance for its :ref:`dev-philosophy`,
its commitment to a flexible choice of integration methods, and
:ref:`use of Numba<dev-numba>` for better performance, the core elements of an 
:ref:`Iterative method <methods-iterative>`
like :class:`~pydmqmc.methods.SymmetricBlochDMQMC` are somewhat obfuscated.

Within the source code, search for the class's definition. This will look like:

.. code-block:: python

    class SymmetricBlochDMQMC(DensityMatrixQMC):

Within this class definition, look for a method called ``_propagate_core``.
This will contain the code for actually performing an iterative update.

Finding the math at the heart of Analytic methods is easier: simply look for the
``run`` method under the class definition.
