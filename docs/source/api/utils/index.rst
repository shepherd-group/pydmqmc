.. _api-utils:

Utils Submodule
===============

This submodule contains utility functions used throughout pydmqmc.
While accessible to users, these functions are generally less polished
than the rest of pydmqmc.

.. contents:: Utilities
    :local:
    :depth: 2

.. _api-bitarrays:

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

.. _api-integrators:

Integrators
-----------

.. autosummary::
    :toctree: stubs

    pydmqmc.utils.euler
    pydmqmc.utils.parallel_euler
    pydmqmc.utils.rk4
    pydmqmc.utils.parallel_rk4

.. _api-parallel:

Parallelism Support
-------------------

.. autosummary::
    :toctree: stubs

    pydmqmc.utils.ParallelHelper

.. _api-permutations:

Permutations
------------

.. autosummary::
    :toctree: stubs

    pydmqmc.utils.generate_ijab_symmetries_array

.. _api-pg-symmetry:

Point Group Symmetry
--------------------

.. autosummary::
    :toctree: stubs

    pydmqmc.utils.pg_sym_cross_prod
    pydmqmc.utils.pg_sym_conj
    pydmqmc.utils.orb_sym

.. _api-saving:

Saving Data
-----------

.. autosummary:: 
    :toctree: stubs

    pydmqmc.utils.save_array
    pydmqmc.utils.save_report

.. _api-slater-condon:

Slater-Condon Rules
-------------------

.. autosummary::
    :toctree: stubs

    pydmqmc.utils.sc0
    pydmqmc.utils.sc1
    pydmqmc.utils.sc2