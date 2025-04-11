# pydmqmc
Python based density matrix quantum Monte Carlo

## Developer Notes
Notes for my own tracking of progress :)
* The `bitarray.py`, `slater_condon.py`, and `symmetry.py` files
in the `utils` module lack test coverage. These tests are lower
priority since most if not all of these functions are used in
constructing the `Integral` class. They should still be added later.
* The above files also need docstrings and type hinting.
