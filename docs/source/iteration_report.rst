.. _iteration-report:

Iteration Progress Reports
==========================

As an :ref:`Iterative <methods-iterative>` calculation runs, you'll
likely want to keep track of several quantities in order to monitor the progress
and convergence of the calculation. During the ``setup()`` stage of an
:ref:`Iterative <methods-iterative>` calculation, you can specify what quantities
you want to track as a list of strings.

This process is enabled by the :data:`pydmqmc.report_registry`, a dictionary-like object
that associates the name of a reportable quantity (such as the
``trace`` or ``energy expectation`` of a matrix being iterated on) with the
function and dependencies used to calculate it.

Specifying Reportable Quantities
--------------------------------

You can specify the quantities you want to track in an :ref:`Iterative <methods-iterative>` calculation
when calling the Method's ``setup()`` function. This is done by providing a list of strings
to the ``report_quants`` parameter:

.. code-block:: python

    import pydmqmc

    # create your system and method objects
    sys = pydmqmc.systems...(...)
    mtd = pydmqmc.methods...(sys)

    # setup your Iterative calculation
    mtd.setup(
        ...     # specify parameters specific to your chosen method
        report_quants = ["trace", "energy numerator", "energy expectation"]
    )

As long as the specified quantities are in ``pydmqmdc.report_registry``,
their associated quantities will be calculated and stored periodically
throughout the iteration.

Note that the **iteration variable is always reported.** For example, for a
:ref:`Density Matrix Quantum Monte Carlo (DMQMC) <methods-dmqmc>` calculation,
this would be the inverse temperature :math:`\beta`.

You may change your mind and rerun the ``setup()`` method any number of times
before starting a calculation with the ``run()`` method.

The Standard Reportable Quantities
++++++++++++++++++++++++++++++++++

The :data:`pydmqmc.report_registry` comes with several pre-defined reportable quantities.
Some of these quantities are expectation values while others like the matrix ``trace``
have no physical meaning but are useful for tracking the progress of a calculation.

The list of standard, pre-defined quantities are listed along with a brief
description in the following table:

*TODO* pull this table from the API docs

Accessing the Iteration Report
------------------------------

Once an :ref:`Iterative <methods-iterative>` calculation has been run,
the report of requested quantities can be accessed with the ``report`` member.
This attribute contains a list of dictionaries, one dictionary for each
report period of the :ref:`Iterative <methods-iterative>` calculation.

.. note::

    The requested report quantities are not recorded every iteration (or "cycle"),
    but rather every few cycles. How many exactly? That depends on the method used;
    for example, for :ref:`DMQMC methods <methods-dmqmc>`, the reporting frequency
    is the same as the ``cycles_per_shift`` parameter passed to the ``run()`` method.

Saving the Iteration Report
+++++++++++++++++++++++++++

Every :ref:`Iterative Method <methods-iterative>` has some sort of ``save_data()``
method that will, at minimum, save the iteration report to disk. Most of these methods
will also save important results of the calcuation. In the case of DMQMC calcuations,
for example, the :meth:`~pydmqmc.methods.DensityMatrixQMC.save_data` method will also save the final density matrix.

The iteration report can be saved with multiple file types: comma-separated value (``.csv``),
text file (``.txt``), or as a Python pickle file (``.pkl``). For example:

.. code-block:: python

    mtd.save_data(
        "my_report",  # base of the filename
        report_filetype = "txt",
        )

Advanced Usage
--------------

The following document advanced use cases for the :data:`pydmqmc.report_registry`.


Defining Your Own Reportable Quantities
+++++++++++++++++++++++++++++++++++++++

You may have a custom function that you'd like to use in an iteration report.
This function may, for instance,
report an observable not included in pydmqmc. The :data:`pydmqmc.report_registry`
allows you to enroll your own functions, assigning them a name and keeping track
of what dependencies that function has. We'll see how to do this in the steps below.

First, you'll need to write a function for your custom observable.
This function should accept, at minimum, a NumPy array. This generic
NumPy array will represent the heart of the :ref:`Iterative <methods-iterative>` calculation,
calculation. For a :ref:`DMQMC <methods-dmqmc>` calculation, for example,
this matrix will be the density matrix.

In the example below, we'll
recreate the existing function for calculating the energy expectation value
(but instead of :func:`~pydmqmc.report.energy_expectation`, we'll call it
``custom_energy``).
This observable requires a second parameter: the Hamiltonian for the system
in question. We'll see how to communicate this dependency in the following
steps; for now, let's just define our function:

.. code-block:: python

    def custom_energy(matrix, hamiltonian):

        eng = np.trace(hamiltonian @ matrix)
        expect = eng / np.trace(matrix)

        return expect

Once we've defined the function for calculating an observable, we need to
enroll it in the :data:`pydmqmc.report_registry`. This can be done in two ways as
outlined below. Regardless of the method used, we'll need to supply the same
information:

    1. The name for our observable
    2. The function used to calculate it
    3. Any dependencies the observable may have (other than the matrix being iterated on)

Note that the dependencies must be members of either a :class:`~pydmqmc.methods.Method` object
or its associated :class:`~pydmqmc.methods.System`. For our example, the ``custom_energy`` function
requires the Hamiltonian. This is associated with a :class:`~pydmqmc.methods.System` object attached
to whatever :class:`~pydmqmc.methods.Method` is generating a report.

The first approach is to use the :meth:`~pydmqmc.report_registry.enroll` method of the :data:`pydmqmc.report_registry`:

.. code-block:: python

    from pydmqmc import report_registry

    report_registry.enroll(
        name = "my_energy",
        function = custom_energy,
        requires = "method.system.hamiltonian"
    )

Note that the dependency (``system.hamiltonian``) is specified as attached to a generic ``method`` object. 
Alternatively, we could just write

.. code-block:: python

    report_registry.enroll(
        name = "my_energy",
        function = custom_energy,
        requires = "hamiltonian"
    )

In this case, the :class:`~pydmqmc.methods.Method` object trying to calculate this observable
and its associated :class:`~pydmqmc.methods.System` will automatically 
be searched for a member called ``hamiltonian``.

We could also enroll our function when defining it by using the ``@enroll`` decorator.
We can optionally pass a name and any requirements to the decorator. If we don't supply a name,
the name of the function itself will be used. For example:

.. code-block:: python

    from pydmqmc import enroll

    @enroll(name = "my_energy", requires = "hamiltonian")
    def custom_energy(matrix, hamiltonian):

        eng = np.trace(hamiltonian @ matrix)
        expect = eng / np.trace(matrix)

        return expect


Using the Report Registry Directly
++++++++++++++++++++++++++++++++++

The :data:`pydmqmc.report_registry` exists to streamline specifying quantities
when setting up an :ref:`Iterative <methods-iterative>` calculation but it
can also be used by you, the user, to easily calculate derived and observable
quantities. The :data:`pydmqmc.report_registry` has several methods available to
facilitate this.

First and foremost, we can use the :meth:`~pydmqmc.report_registry.keys` method
to see which keys are currently registered:

.. code-block:: python

    >>> from pydmqmc import report_registry
    >>> report_registry.keys()
    dict_keys(['trace', 'energy numerator', 'energy expectation', ... ])

Next is the :meth:`~pydmqmc.report_registry.list_requirements` method.
This method returns a tuple of all the dependencies that are required by a function
enrolled in the registry. Note that the matrix being iterated upon is not
considered a dependency. For example:

.. code-block:: python

    >>> report_registry.list_requirements("trace")
    >>> report_registry.list_requirements("energy expectation")
    ('hamiltonian',)

We see from above that the ``"trace"`` key isn't associated with any requirements,
while the ``"energy expectation"`` key has one---the Hamiltonian.

For the simple case of ``"trace"``, we can easily retrieve the associated function
with square bracket notation:

.. code-block:: python

    >>> import numpy as np
    >>> matrix = np.array([[0.2, 0.4],[0.1, 0.3]])
    >>> trace = report_registry["trace"]
    >>> type(trace)
    function
    >>> trace(matrix)
    np.float64(0.5)

We can do the same with ``"energy expectation"``, but recall that this function
requires a Hamiltonian. Let's go ahead an set up a :class:`~pydmqmc.systems.Integral`
object to provide us with our Hamiltonian. Then, we can fetch and call the function
associated with ``"energy expectation"`` by manually supplying its requirements:

.. code-block:: python

    >>> from pydmqmc.systems import Integral
    >>> sys = Integral("pydmqmc/tests/inputs/integrals/H2-STO-3G-0.74Ang.fcidump")
    >>> sys.generate_hamiltonian()
    >>> eng = report_registry["energy expectation"]
    >>> eng(matrix, sys.hamiltonian)
    np.float64(0.01207762620230829)

What if we don't want to manually collect the additional requirements?
If our :class:`~pydmqmc.systems.Integral` has been associated with a
Method (like :class:`~pydmqmc.methods.SymmetricBlochDMQMC`) we can supply
this to :meth:`~pydmqmc.report_registry.get_requirements` to return a dictionary
of requirements that can be fed to function we pull from the registry:

.. code-block:: python

    >>> from pydmqmc.methods import SymmetricBlochDMQMC
    >>> mtd = SymmetricBlochDMQMC(sys)
    >>> reqs = report_registry.get_requirements("energy expectation", mtd)
    >>> reqs
    {'hamiltonian': array([[...]])}
    >>> eng = report_registry["energy expectation"]
    >>> eng(matrix, **reqs)
    np.float64(0.01207762620230829)