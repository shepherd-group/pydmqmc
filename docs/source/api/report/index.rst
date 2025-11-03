.. _api-report:

Report Submodule
================

This submodule is used to construct the ``pydmqmc.report_registry``
that's used for generating :ref:`iteration-report`. It contains the
private ``_ReportRegistry`` class. Since it's private, the API for this
class is not documented. Please see the documentation on :ref:`iteration-report`
to add your own function to the ``pydmqmc.report_registry``.

What is documented for this submodule are the standard quantities that are
available as part of the ``pydmqmc.report_registry``; specifically, the functions
used to calculated these quantities.

.. autosummary::
    :toctree: stubs

    pydmqmc.report.trace
    pydmqmc.report.energy_numerator
    pydmqmc.report.energy_expectation
    pydmqmc.report.von_neumann_numerator
    pydmqmc.report.von_neumann_expectation