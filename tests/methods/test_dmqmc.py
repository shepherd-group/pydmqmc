import numpy as np
from pytest import fixture, raises
from os.path import dirname, join

from pydmqmc.systems import MatrixHamiltonian, Integral
from pydmqmc.methods import DensityMatrixQMC, \
        AsymmetricBlochDMQMC, \
        SymmetricBlochDMQMC


@fixture(scope="module")
def integral_system_small(request) -> Integral:
    file = join(dirname(request.path),
                "..", "inputs", "integrals", "H2-STO-3G-0.74Ang.fcidump")
    sys = Integral(file)
    return sys


@fixture(scope="module")
def integral_system_large(request) -> Integral:
    file = join(dirname(request.path),
                "..", "inputs", "integrals", "STRICT-STO3G-STR-H4.FCIDUMP")
    sys = Integral(file)
    return sys


@fixture(scope="module")
def matrix_system(request) -> MatrixHamiltonian:
    file = join(dirname(request.path),
                "..", "inputs", "hamiltonians", "EQUILIBRIUM-H4-STO3G.hamil")
    sys = MatrixHamiltonian(file)
    return sys


@fixture(scope="module")
def dmqmc(matrix_system) -> DensityMatrixQMC:
    mtd = DensityMatrixQMC(matrix_system)
    return mtd


@fixture(scope="module")
def asym_dmqmc(integral_system_large):
    """Make sure to call asym_dmqmc.reset_rng every use!"""
    mtd = AsymmetricBlochDMQMC(integral_system_large)
    return mtd


@fixture(scope="module")
def sym_dmqmc(integral_system_large):
    """Make sure to call sym_dmqmc.reset_rng every use!"""
    mtd = SymmetricBlochDMQMC(integral_system_large)
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
    dmqmc.setup("random-uniform", n_particles=10)
    assert np.allclose(np.diag(dmqmc.density_matrix),
                       diag)


def test_DMQMC_setup_fixed(dmqmc):
    diag = [10, 30, 40, 25, 18, 54, 22, 34, 47, 36,
            45, 37, 23, 46, 41, 31, 27, 49, 17, 38]

    dmqmc.setup("fixed", fixed_diagonal=diag)
    assert np.allclose(np.diag(dmqmc.density_matrix),
                       diag)
    assert dmqmc.density_matrix.size == 400


def test_DMQMC_setup_fixed_bad(dmqmc):
    diag = [10, 30, 40]

    with raises(RuntimeError):
        dmqmc.setup("fixed", fixed_diagonal=diag)


def test_DMQMC_setup_unknown(dmqmc):
    with raises(RuntimeError):
        dmqmc.setup("bad-method")


def test_DMQMC_run(dmqmc):
    dmqmc.setup("deterministic")

    with raises(NotImplementedError):
        dmqmc.run(25, 0.01, 10, 0.05)


def test_DMQMC_run_no_setup(dmqmc):
    with raises(RuntimeError):
        dmqmc.run(25, 0.01, 10, 0.05)


def test_DMQMC_run_bad_ilevel(dmqmc):
    dmqmc.setup("deterministic")

    with raises(TypeError):
        dmqmc.run(25, 0.01, 10, 0.05, ilevel=0.1)


def test_AsymmetricBlochDMQMC_basic(asym_dmqmc):
    """
    Implicitly tests dummy matrix created for ilevel = None and ilevel = 0.
    """
    asym_dmqmc.reset_rng(42)
    asym_dmqmc.setup("random-uniform", n_particles=int(1e5))
    asym_dmqmc.run(final_beta=25,
        dbeta=0.001,
        cycles_per_shift=1000,
        shift_dampening=0.05,
        spawn_cutoff=0.01,
        shift_by_rows=False)

    assert np.isclose(asym_dmqmc.density_matrix.trace(), 67981.48932281222)
    eng = (asym_dmqmc.density_matrix @ asym_dmqmc.system.hamiltonian).trace()
    assert np.isclose(eng, -141115.38639919003)


def test_AsymmetricBlochDMQMC_rbr(asym_dmqmc):
    asym_dmqmc.reset_rng(42)
    asym_dmqmc.setup("random-uniform", n_particles=int(1e5))
    asym_dmqmc.run(final_beta=25,
        dbeta=0.001,
        cycles_per_shift=1000,
        shift_dampening=0.05,
        spawn_cutoff=0.01,
        shift_by_rows=True)

    assert np.isclose(asym_dmqmc.density_matrix.trace(), 22493.37887777515)
    eng = (asym_dmqmc.density_matrix @ asym_dmqmc.system.hamiltonian).trace()
    assert np.isclose(eng, -46578.848998115835)


def test_AsymmetricBlochDMQMC_ilevel_zero(asym_dmqmc):
    """
    Test functionality of ilevel = 0 (and dummy matrix functionality).

    Set density matrix and n_add such that psip spawning 
    using ilevel=0 is emphasized.
    """
    asym_dmqmc.reset_rng(42)
    asym_dmqmc.setup("deterministic")
    asym_dmqmc.run(final_beta=25,
        dbeta=0.001,
        cycles_per_shift=1000,
        shift_dampening=0.05,
        spawn_cutoff=0.01,
        n_add=3,  # strongly limit this spawn channel to emph ilevel
        ilevel=0)

    assert np.isclose(asym_dmqmc.density_matrix.trace(), 14.206870483605295)
    eng = (asym_dmqmc.density_matrix @ asym_dmqmc.system.hamiltonian).trace()
    assert np.isclose(eng, -29.461275823860465)


def test_AsymmetricBlochDMQMC_ilevel_nonzero(asym_dmqmc):
    asym_dmqmc.reset_rng(42)
    asym_dmqmc.setup("random-uniform", n_particles=int(1e5))
    asym_dmqmc.run(final_beta=25,
        dbeta=0.001,
        cycles_per_shift=1000,
        shift_dampening=0.05,
        spawn_cutoff=0.01,
        ilevel=2)

    assert np.isclose(asym_dmqmc.density_matrix.trace(), 67981.48986893434)
    eng = (asym_dmqmc.density_matrix @ asym_dmqmc.system.hamiltonian).trace()
    assert np.isclose(eng, -141115.3875013612)


def test_SymmetricBlochDMQMC_basic(sym_dmqmc):
    sym_dmqmc.reset_rng(42)
    sym_dmqmc.setup("random-uniform", n_particles=int(1e5))
    sym_dmqmc.run(final_beta=25,
        dbeta=0.001,
        cycles_per_shift=1000,
        shift_dampening=0.05,
        spawn_cutoff=0.01,
        shift_by_rows=False)

    assert np.isclose(sym_dmqmc.density_matrix.trace(), 67926.38108893688)
    eng = (sym_dmqmc.density_matrix @ sym_dmqmc.system.hamiltonian).trace()
    assert np.isclose(eng, -141000.99315753282)


def test_SymmetricBlochDMQMC_rbr(sym_dmqmc):
    sym_dmqmc.reset_rng(42)
    sym_dmqmc.setup("random-uniform", n_particles=int(1e5))
    sym_dmqmc.run(final_beta=25,
        dbeta=0.001,
        cycles_per_shift=1000,
        shift_dampening=0.05,
        spawn_cutoff=0.01,
        shift_by_rows=True)

    assert np.isclose(sym_dmqmc.density_matrix.trace(), 20325.670796384442)
    eng = (sym_dmqmc.density_matrix @ sym_dmqmc.system.hamiltonian).trace()
    assert np.isclose(eng, -42058.99263000391)


def test_SymmetricBlochDMQMC_ilevel_zero(sym_dmqmc):
    """
    Test functionality of ilevel = 0 (and dummy matrix functionality).

    Set density matrix and n_add such that psip spawning 
    using ilevel=0 is emphasized.
    """
    sym_dmqmc.reset_rng(42)
    sym_dmqmc.setup("deterministic")
    sym_dmqmc.run(final_beta=25,
                  dbeta=0.001,
                  cycles_per_shift=1000,
                  shift_dampening=0.05,
                  spawn_cutoff=0.01,
                  n_add=3,
                  ilevel=0
                  )

    assert np.isclose(sym_dmqmc.density_matrix.trace(), 13.608499042255167)
    eng = (sym_dmqmc.density_matrix @ sym_dmqmc.system.hamiltonian).trace()
    assert np.isclose(eng, -28.215638757282814)


def test_SymmetricBlochDMQMC_ilevel_nonzero(sym_dmqmc):
    sym_dmqmc.reset_rng(42)
    sym_dmqmc.setup("random-uniform", n_particles=int(1e5))
    sym_dmqmc.run(final_beta=25,
                dbeta=0.001,
                cycles_per_shift=1000,
                shift_dampening=0.05,
                spawn_cutoff=0.01,
                ilevel=2
                )

    assert np.isclose(sym_dmqmc.density_matrix.trace(), 67926.38161070104)
    eng = (sym_dmqmc.density_matrix @ sym_dmqmc.system.hamiltonian).trace()
    assert np.isclose(eng, -141000.99421006895)
