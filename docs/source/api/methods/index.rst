.. _api-methods:

Methods Submodule
=================

This submodule contains three base classes: 
:class:`~pydmqmc.methods.Method`, :class:`~pydmqmc.methods.Analytic`, and
:class:`~pydmqmc.methods.Iterative`. The former two base classes are used for
the two major families of methods: :ref:`Analytic <methods-analytic>` and
:ref:`Iterative <methods-iterative>`. 

The APIs for these two method families are listed
alphabetically under their respective types. The base classes are listed at the
end of this document.

.. .. contents:: Method Types:
..     :depth: 2
..     :local:

.. _api-analytic-family:

Analytic Methods
----------------
.. toctree::
    :maxdepth: 1

    analytic/full_configuration_interaction

.. _api-iterative-family:

Iterative Methods
-----------------
.. toctree::
    :maxdepth: 2

    iterative/dmqmc/dmqmc

.. _api-method-bases:

Base Classes
------------
.. toctree::
    :maxdepth: 1

    base/method
    base/analytic
    base/iterative