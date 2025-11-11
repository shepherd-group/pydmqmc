.. _api-report:

Report Registry
===============

This submodule contains the :data:`~pydmqmc.report_registry`
that's used for generating :ref:`iteration-report`.
It also contains the functions that come pre-enrolled
in the :data:`~pydmqmc.report_registry` for calculating
traceable quantities.

Please see the documentation on :ref:`iteration-report`
for more information, including how to use the registry.

Accessing the Report Registry
-----------------------------

.. autodata:: pydmqmc.report_registry

.. autosummary::
    :toctree: stubs

    pydmqmc.report_registry.list_requirements
    pydmqmc.report_registry.get_requirements
    pydmqmc.report_registry.enroll
    pydmqmc.report_registry.keys
    pydmqmc.report_registry.functions
    pydmqmc.report_registry.requirements

Functions for Standard Quantities
---------------------------------

These functions are already enrolled in the :data:`~pydmqmc.report_registry`
for use as detailed in the page on :ref:`iteration-report`.

.. autosummary::
    :toctree: stubs

    pydmqmc.report.trace
    pydmqmc.report.energy_numerator
    pydmqmc.report.energy_expectation
    pydmqmc.report.von_neumann_numerator
    pydmqmc.report.von_neumann_expectation
