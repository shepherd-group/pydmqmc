import numpy as np
from pytest import fixture, raises
from os.path import dirname, join

from pydmqmc.systems import MatrixHamiltonian, Integral
from pydmqmc.methods import DensityMatrixQMC, AsymmetricBlochDMQMC


@fixture
def matrix_system(request) -> MatrixHamiltonian:
    file = join(dirname(request.path),
                "..", "inputs", "hamiltonians", "EQUILIBRIUM-H4-STO3G.hamil")
    sys = MatrixHamiltonian(file)
    return sys


@fixture
def integral_system_small(request) -> Integral:
    file = join(dirname(request.path),
                "..", "inputs", "integrals", "H2-STO-3G-0.74Ang.fcidump")
    sys = Integral(file)
    return sys


@fixture
def integral_system_large(request) -> Integral:
    file = join(dirname(request.path),
                "..", "inputs", "integrals", "STRICT-STO3G-STR-H4.FCIDUMP")
    sys = Integral(file)
    return sys


@fixture
def dmqmc(matrix_system) -> DensityMatrixQMC:
    mtd = DensityMatrixQMC(matrix_system)
    return mtd


def test_DMQMC_init_MatrixHamiltonian(matrix_system):
    mtd = DensityMatrixQMC(matrix_system)

    assert mtd.system.hamiltonian is matrix_system.hamiltonian


def test_DMQMC_init_Integral(integral_system_small):
    mtd = DensityMatrixQMC(integral_system_small)

    assert mtd.system.hamiltonian is not None
    assert mtd.system.hamiltonian is integral_system_small.hamiltonian


def test_DMQMC_setup_determinitistic(dmqmc):
    dmqmc.setup("deterministic")
    assert np.allclose(dmqmc.density_matrix, 
                       np.eye(dmqmc.system.n_determinants))


def test_DMQMC_setup_uniform_random(dmqmc):
    diag = np.array([0, 3, 0, 0, 1, 0, 0, 0, 2, 0, 
                     0, 0, 0, 2, 0, 1, 0, 1, 0, 0])

    dmqmc.reset_rng(rng_seed=42)
    dmqmc.setup("uniform-random", n_particles=10)
    assert np.allclose(np.diag(dmqmc.density_matrix),
                       diag)


def test_DMQMC_setup_specific_rows(dmqmc):
    rows = [1, 15, 7, 10, 18, 3]
    diag = [0, 1, 0, 1, 0, 0, 0, 1, 0, 0,
            1, 0, 0, 0, 0, 1, 0, 0, 1, 0]

    dmqmc.setup("specific-rows", row_list=rows)
    assert np.allclose(np.diag(dmqmc.density_matrix),
                       diag)


def test_DMQMC_setup_unknown(dmqmc):
    with raises(RuntimeError):
        dmqmc.setup("bad-method")


def test_DMQMC_run(matrix_system):
    mtd = DensityMatrixQMC(matrix_system)
    mtd.setup("deterministic")

    with raises(NotImplementedError):
        mtd.run(25, 10, 10)


def test_DMQMC_run_no_setup(dmqmc):
    with raises(RuntimeError):
        dmqmc.run(25, 10, 10)


def test_AsymmetricBlochDMQMC_basic(integral_system_large):

    pass