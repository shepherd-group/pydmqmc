.. _iteration-report:

Iteration Progress Reports
==========================

As an :ref:`Iterative <methods-iterative>` calculation runs, you'll
likely want to keep track of several quantities in order to track the progress
and convergence of the calculation. During the ``setup()`` stage of an
:ref:`Iterative <methods-iterative>` calculation, you can specify what quantities
you want to track as a list of strings.

This process is enabled by the ``pydmqmc.report_registry``, a dictionary-like object
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
:ref:`Density Matrix Quantum Monte Carlo <methods-dmqmc>` calculation,
this would be the inverse temperature :math:`\beta`.

You may change your mind and rerun the ``setup()`` method any number of times
before starting a calculation with the ``run()`` method.

The Standard Reportable Quantities
++++++++++++++++++++++++++++++++++

The ``pydmqmc.report_registry`` comes with several pre-defined reportable quantities.
Some of these quantities are expectation values while others like the matrix ``trace``
have no physical meaning but are useful for tracking the progress of a calculation.

The list of standard, pre-defined quantities are listed along with a brief
description in the following table:

*TODO* pull this table from the API docs

Accessing the Iteration Report
------------------------------

Saving the Iteration Report
+++++++++++++++++++++++++++

Advanced Usage: Defining Your Own Reportable Quantities
-------------------------------------------------------

