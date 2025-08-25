.. _ref-systems:

Available Systems
=================

The systems that are available for use are described below

.. contents:: Systems
    :local:

System Defined by a Hamiltonian Matrix Only
-------------------------------------------

A very minimal :class:`~pydmqmc.systems.System` can be defined using only a Hamiltonian matrix.
This can be accomplished with the :class:`~pydmqmc.systems.MatrixHamiltonian` class,
where the only required argument is the path to a triangular Hamiltonian matrix output
in the style of `HANDE`_.

.. _HANDE: https://hande.readthedocs.io/en/latest/

Note that additional information does need to be specified for a
:class:`~pydmqmc.systems.MatrixHamiltonian` object to use the
following methods:

* :meth:`~pydmqmc.systems.System.generate_determinant_bitarrays`
* :meth:`~pydmqmc.systems.System.generate_excitation_matrix`
* :meth:`~pydmqmc.systems.System.get_bitarray_integers`
* :meth:`~pydmqmc.systems.System.get_virtual_oribtals`

Some :ref:`Methods <ref-methods>` may use these class methods under the hood
and will throw an error if they are not supplied.
See the API reference for :class:`~pydmqmc.systems.MatrixHamiltonian`
to know which optional parameters each of the above methods requires.


Integral-Defined System
-----------------------

Systems may be defined using integral files supplied in the `HANDE`_ `FCIDUMP`_
file format. Such systems are supported by the :class:`~pydmqmc.systems.Integral` class.

.. _FCIDUMP: https://hande.readthedocs.io/en/latest/manual/integrals.html#fcidump-format

The Hamiltonian for :class:`~pydmqmc.systems.Integral` systems must be calculated from the
integrals specified in the class's :attr:`~pydmqmc.systems.Integral.input_file`.
Since this calculation may be computationally intensive, it is not performed by default.
Instead, the :attr:`~pydmqmc.systems.Integral.hamiltonian` attribute is not defined until
:meth:`~pydmqmc.systems.Integral.generate_hamiltonian` is called.
The user may call this function or it will be called automatically
by any :ref:`Method <ref-methods>` that requires a Hamiltonian.
