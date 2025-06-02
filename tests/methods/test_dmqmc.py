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


def test_DMQMC_setup_fixed(dmqmc):
    diag = [10, 30, 40, 25, 18, 54, 22, 34, 47, 36,
            45, 37, 23, 46, 41, 31, 27, 49, 17, 38]

    dmqmc.setup("fixed", diag=diag)
    assert np.allclose(np.diag(dmqmc.density_matrix),
                       diag)
    assert dmqmc.density_matrix.size == 400


def test_DMQMC_setup_fixed_bad(dmqmc):
    diag = [10, 30, 40]

    with raises(RuntimeError):
        dmqmc.setup("fixed", diag=diag)


def test_DMQMC_setup_unknown(dmqmc):
    with raises(RuntimeError):
        dmqmc.setup("bad-method")


def test_DMQMC_run(matrix_system):
    mtd = DensityMatrixQMC(matrix_system)
    mtd.setup("deterministic")

    with raises(NotImplementedError):
        mtd.run(25, 0.01, 10, 0.05)


def test_DMQMC_run_no_setup(dmqmc):
    with raises(RuntimeError):
        dmqmc.run(25, 0.01, 10, 0.05)


def test_AsymmetricBlochDMQMC_basic(integral_system_large):
    mtd = AsymmetricBlochDMQMC(integral_system_large,
                               rng_seed=42)
    mtd.setup("uniform-random", n_particles=int(1e5))
    mtd.run(final_beta=25,
        dbeta=0.001,
        cycles_per_shift=1000,
        shift_dampening=0.05,
        spawn_cutoff=0.01,
        shift_by_rows=False)

    assert np.isclose(mtd.density_matrix.trace(), 67981.48932281222)
    eng = (mtd.density_matrix @ mtd.system.hamiltonian).trace()
    assert np.isclose(eng, -141115.38639919003)

def test_AsymmetricBlochDMQMC_basic_rbr(integral_system_large):
    mtd = AsymmetricBlochDMQMC(integral_system_large,
                               rng_seed=42)
    mtd.setup("uniform-random", n_particles=int(1e5))
    mtd.run(final_beta=25,
        dbeta=0.001,
        cycles_per_shift=1000,
        shift_dampening=0.05,
        spawn_cutoff=0.01,
        shift_by_rows=True)

    assert np.isclose(mtd.density_matrix.trace(), 22493.37887777515)
    eng = (mtd.density_matrix @ mtd.system.hamiltonian).trace()
    assert np.isclose(eng, -46578.848998115835)
