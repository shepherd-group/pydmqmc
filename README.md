# pydmqmc
Python based density matrix quantum Monte Carlo

## Developer Notes
Notes for Claire's tracking of progress :)
* Functions `random_bitarray_symspace()` and `generate_renorm_excitation()`
in the `Integral` class need tests.
* Internal `Integral` methods `_set_reference`,
`_set_symmetry`, and `_symmetry_check` still need
verification.

Once docs and the above are finished:
* Implement Will's MPI
* Create a speed/scaling page that also compares with HANDE