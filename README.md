<!-- The coverage badge was generated using the genbadge package: https://smarie.github.io/python-genbadge/ -->
[![Test Status](https://github.com/shepherd-group/pydmqmc/actions/workflows/python-app.yml/badge.svg)](https://github.com/shepherd-group/pydmqmc/actions/workflows/python-app.yml)
![Coverage Status](.coverage-badge.svg?dummy=8484744)
[![Documentation Status](https://readthedocs.org/projects/pydmqmc/badge/?version=latest)](https://pydmqmc.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/19614801.svg)](https://zenodo.org/badge/latestdoi/19614801)

# pydmqmc

The `pydmqmc` package let’s you quickly assemble finite-temperature many-electron calculations. It serves as library with which your own scripts can be written and executed.

`pydmqmc` was inspired by the [HANDE](https://github.com/hande-qmc/hande) code.

## Installation

The `pydmqmc` package is available via PyPI:

``` bash
pip install pydmqmc
```

Any necessary dependencies will be installed automatically!

### Development Version

First, clone this repository and set the branch:

``` bash
git clone https://github.com/shepherd-group/pydmqmc.git
cd pydmqmc
```

Next, set up a Python environment. If you use Conda to manage your Python environments, you can set up a full development environment using the provided `environment.yml`:

``` bash
conda env create -f environment.yml
```

If you prefer to build with pip, you can skip to building `pydmqmc` and all dependencies should be installed automatically.

The final step is to build the `pydmqmc` Python package. Since this project is in early development,
it's advisable to build with the `--editable` or `-e` flag:

```bash
pip install -e .
```

You can test your installation is working by running the test suite:

``` bash
pytest
```

#### Parallel Tests

Note that, by default, `pytest` is configured to skip tests for the parallel (MPI) infrastructure.
This is because the way `pytest` prefers to natively run these tests (i.e., using forking) fails on some version
of MPI, such as OpenMPI on Ubuntu (see the documentation for [mpi-pytest](https://pypi.org/project/mpi-pytest/) for more details).

To run these parallel tests with OpenMPI, please run the following:
``` bash
pytest  # serial tests
bash tests/run_parallel_tests.sh  # parallel tests
```

For all other versions of MPI, you may run:
``` bash
pytest  # serial tests
pytest -m parallel  # parallel tests
```

#### Building the Documentation

The `environment.yml` file includes everything you need to build the documentation yourself.
To do so, run the following:

``` bash
cd docs
make html
```

The documentation will then be available in `pydmqmc/docs/build/html`.
Open `pydmqmc/docs/build/html/index.html` to browse the
documentation like a webpage!

Navigate to the Quickstart page to get started with `pydmqmc`!

## Citing this software

Please cite this software using the following DOI: [10.5281/zenodo.19614801](https://zenodo.org/records/19614801).

Depending on the methods used in your work, please also cite the following method papers (DOIs provided):

* DMQMC: [10.1103/PhysRevB.89.245124](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.89.245124)
* IP-DMQMC: [10.1063/1.4927434](https://pubs.aip.org/aip/jcp/article/143/4/044116/71857/Interaction-picture-density-matrix-quantum-Monte)

## Developer Notes

Please be advised that `pydmqmc` is in a alpha release state and is still being
actively developed. Some features may be missing and the API is subject to change.
You are welcome to submit comments and requests to the lead developer (Claire Kopenhafer, kopenhaf@msu.edu),
ideally as issues on the `pydmqmc` GitHub page (though Slack and email messages also work).

The following are notes from Claire about unfinished work, future planned work, and possible changes.

### Unfinished Work

* Functions `random_bitarray_symspace()` and `generate_renorm_excitation()`
in the `Integral` class need unit tests.
* Internal `Integral` methods `_set_reference`,
`_set_symmetry`, and `_symmetry_check` still need
verification.
* The `report_registry` could have more functions available to it.

### Future Work

* Create a speed/scaling page that also compares with HANDE
* Add more `Methods` from Will's original code.

### Possible API Changes

* The `Method` categories `Iterative` and `Analytic` represent a false dichotomy. `Analytic` may become `Direct`, or `Iterative` could become `Stochastic`. It depends on what other `Method` types get added to `pydmqmc`.