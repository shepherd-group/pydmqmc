.. _dev-new-systems:

Writing New Systems
===================

This is a step-by-step guide for developing new classes in
the :ref:`Systems submodule<ref-systems>`.

When to Add a New System
------------------------

Each user-facing class in the :ref:`Systems submodule<ref-systems>` is
designed to handle a particular file format; therefore, **if you want
a new file format to be able to define a system**, 
you will need a new System class.

The name of this class doesn't have to reference the file format name.
Instead your new class should reference how the system is defined.
For example, while the :class:`~pydmqmc.systems.Integral` class
loads FCIDUMP files the name "Integral" is a reference to how the
system is defined by a set of integrals (rather than, say,
directly specifying a Hamiltonian matrix as with
:class:`~pydmqmc.systems.MatrixHamiltonian`),

Creating a New File
-------------------

Each unique system should be defined in its own file. From the root
pydmqmc directory that you cloned from GitHub, create a Python (.py) file in::

    src/pydmqmc/systems/