# pydmqmc
Python based density matrix quantum Monte Carlo

## Developer Notes
Notes for my own tracking of progress :)
* The `bitarray.py`, `slater_condon.py`, and `symmetry.py` files
in the `utils` module lack docstrings, type hinting, and tests.
The tests are lower priority since most if not all of these
functions are used in constructing the `Integral` class.
They should still be added later.
* Functions `random_bitarray_symspace()` and `generate_renorm_excitation()`
in the `Integral` class need docstrings, type hinting, and tests.
They might also be better off in the `System` class but I will migrate
once tests exist.
* Internal `Integral` methods `_set_reference`,
`_set_symmetry`, and `_symmetry_check` still need
testing.