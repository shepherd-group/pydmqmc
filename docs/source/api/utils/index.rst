.. _api-utils:

Utils Submodule
===============

This submodule contains utility functions used throughout pydmqmc.
While accessible to users, these functions are generally less polished
than the rest of pydmqmc.

.. contents:: Utilities
    :local:
    :depth: 2

Bitarray Manipulation
---------------------

.. autosummary::
    :toctree: stubs

    pydmqmc.utils.bitarray_to_integer
    pydmqmc.utils.integer_to_bitarray
    pydmqmc.utils.concate_bitarrays_to_label
    pydmqmc.utils.extract_bitarrays_from_label
    pydmqmc.utils.get_nex
    pydmqmc.utils.get_occ
    pydmqmc.utils.get_iocc
    pydmqmc.utils.get_single_perm
    pydmqmc.utils.get_double_perm
    pydmqmc.utils.get_ex_info
    pydmqmc.utils.bitarray_pg

Integrators
-----------

.. autosummary::
    :toctree: stubs

    pydmqmc.utils.euler
    pydmqmc.utils.rk4

Permutations
------------

.. autosummary::
    :toctree: stubs

    pydmqmc.utils.generate_ijab_symmetries_array

Point Group Symmetry
--------------------

.. autosummary::
    :toctree: stubs

    pydmqmc.utils.cross_prod_sym
    pydmqmc.utils.orb_sym
    pydmqmc.utils.conj_sym

Saving Data
-----------

.. autosummary:: 
    :toctree: stubs

    pydmqmc.utils.save_array
    pydmqmc.utils.save_report

Slater-Condon Rules
-------------------

.. autosummary::
    :toctree: stubs

    pydmqmc.utils.sc0
    pydmqmc.utils.sc1
    pydmqmc.utils.sc2