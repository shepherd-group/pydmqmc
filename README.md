# pydmqmc
Python based density matrix quantum Monte Carlo

## Developer Notes
Notes for Claire's tracking of progress :)
* The `utils` package needs complete docstrings, type hints, and tests.
* Functions `random_bitarray_symspace()` and `generate_renorm_excitation()`
in the `Integral` class need docstrings, type hinting, and tests.
They might also be better off in the `System` class but I will migrate
once tests exist.
* Internal `Integral` methods `_set_reference`,
`_set_symmetry`, and `_symmetry_check` still need
testing.
* IP-DMQMC needs verification and updated test answers
