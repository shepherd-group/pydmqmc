import numpy as np
from pytest import fixture, raises, mark
from pytest_lazy_fixtures import lf
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


class TestDMQMC():

    @fixture(autouse=True)
    def _setup(self, matrix_system):
        self._mtd = DensityMatrixQMC(matrix_system)

    @mark.parametrize("system", 
                      [lf('matrix_system'), 
                       lf('integral_system_small')])
    def test_init(self, system):
        mtd = DensityMatrixQMC(system)

        assert mtd.system is system

    def test_setup_determinitistic(self):
        self._mtd.setup(1.0, "deterministic")
        assert np.allclose(self._mtd.density_matrix, 
                        np.eye(self._mtd.system.n_determinants))

    def test_setup_random_uniform(self):
        diag = np.array([0, 3, 0, 0, 1, 0, 0, 0, 2, 0, 
                        0, 0, 0, 2, 0, 1, 0, 1, 0, 0])

        self._mtd.reset_rng(rng_seed=42)
        self._mtd.setup(1.0, "random-uniform", n_particles=10)
        assert np.allclose(np.diag(self._mtd.density_matrix),
                        diag)

    def test_setup_fixed(self):
        diag = [10, 30, 40, 25, 18, 54, 22, 34, 47, 36,
                45, 37, 23, 46, 41, 31, 27, 49, 17, 38]

        self._mtd.setup(1.0, "fixed", fixed_diagonal=diag)
        assert np.allclose(np.diag(self._mtd.density_matrix),
                        diag)
        assert self._mtd.density_matrix.size == 400

    def test_setup_fixed_invalid(self):
        diag = [10, 30, 40]

        with raises(RuntimeError):
            self._mtd.setup(1.0, "fixed", fixed_diagonal=diag)

    def test_setup_unknown(self):
        with raises(RuntimeError):
            self._mtd.setup(1.0, "bad-method")

    def test_run(self):
        self._mtd.setup(1.0, "deterministic")

        with raises(NotImplementedError):
            self._mtd.run(0.01, 10, 0.05)

    def test_run_no_setup(self):
        with raises(RuntimeError):
            self._mtd.run(0.01, 10, 0.05)

    def test_run_invalid_ilevel(self):
        self._mtd.setup(1.0, "deterministic")

        with raises(TypeError):
            self._mtd.run(0.01, 10, 0.05, ilevel=0.1)


class TestAsymmetricBlochDMQMC():

    @fixture(autouse=True)
    def _setup(self, integral_system_large):
        self._mtd = AsymmetricBlochDMQMC(integral_system_large)

    def test_basic(self):
        """
        Implicitly tests dummy matrix created for ilevel = None and ilevel = 0.
        """
        self._mtd.reset_rng(42)
        self._mtd.setup(
            final_beta=25,
            initialization="random-uniform",
            n_particles=int(1e5)
        )
        self._mtd.run(
            dbeta=0.001,
            cycles_per_shift=1000,
            shift_dampening=0.05,
            spawn_cutoff=0.01,
            shift_by_rows=False
        )

        assert np.isclose(self._mtd.density_matrix.trace(), 67981.48932281222)
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        assert np.isclose(eng, -141115.38639919003)

    def test_rbr(self):
        self._mtd.reset_rng(42)
        self._mtd.setup(
            final_beta=25,
            initialization="random-uniform",
            n_particles=int(1e5)
        )
        self._mtd.run(
            dbeta=0.001,
            cycles_per_shift=1000,
            shift_dampening=0.05,
            spawn_cutoff=0.01,
            shift_by_rows=True
        )

        assert np.isclose(self._mtd.density_matrix.trace(), 22493.37887777515)
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        assert np.isclose(eng, -46578.848998115835)

    def test_ilevel_zero(self):
        """
        Test functionality of ilevel = 0 (and dummy matrix functionality).

        Set density matrix and n_add such that psip spawning 
        using ilevel=0 is emphasized.
        """
        self._mtd.reset_rng(42)
        self._mtd.setup(
            final_beta=25,
            initialization="deterministic"
        )
        self._mtd.run(
            dbeta=0.001,
            cycles_per_shift=1000,
            shift_dampening=0.05,
            spawn_cutoff=0.01,
            n_add=3,  # strongly limit this spawn channel to emph ilevel
            ilevel=0
        )

        assert np.isclose(self._mtd.density_matrix.trace(), 14.206870483605295)
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        assert np.isclose(eng, -29.461275823860465)

    def test_ilevel_nonzero(self):
        self._mtd.reset_rng(42)
        self._mtd.setup(
            final_beta=25,
            initialization="random-uniform",
            n_particles=int(1e5)
        )
        self._mtd.run(
            dbeta=0.001,
            cycles_per_shift=1000,
            shift_dampening=0.05,
            spawn_cutoff=0.01,
            ilevel=2
        )

        assert np.isclose(self._mtd.density_matrix.trace(), 67981.48986893434)
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        assert np.isclose(eng, -141115.3875013612)


class TestSymmetricBlochDMQMC():

    @fixture(autouse=True)
    def _setup(self, integral_system_large):
        self._mtd = SymmetricBlochDMQMC(integral_system_large)

    def test_basic(self):
        self._mtd.reset_rng(42)
        self._mtd.setup(
            final_beta=25,
            initialization="random-uniform",
            n_particles=int(1e5)
        )
        self._mtd.run(
            dbeta=0.001,
            cycles_per_shift=1000,
            shift_dampening=0.05,
            spawn_cutoff=0.01,
            shift_by_rows=False
        )

        assert np.isclose(self._mtd.density_matrix.trace(), 67926.38108893688)
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        assert np.isclose(eng, -141000.99315753282)

    def test_rbr(self):
        self._mtd.reset_rng(42)
        self._mtd.setup(
            final_beta=25,
            initialization="random-uniform",
            n_particles=int(1e5)
        )
        self._mtd.run(
            dbeta=0.001,
            cycles_per_shift=1000,
            shift_dampening=0.05,
            spawn_cutoff=0.01,
            shift_by_rows=True
        )

        assert np.isclose(self._mtd.density_matrix.trace(), 20325.670796384442)
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        assert np.isclose(eng, -42058.99263000391)

    def test_ilevel_zero(self):
        """
        Test functionality of ilevel = 0 (and dummy matrix functionality).

        Set density matrix and n_add such that psip spawning 
        using ilevel=0 is emphasized.
        """
        self._mtd.reset_rng(42)
        self._mtd.setup(
            final_beta=25,
            initialization="deterministic"
        )
        self._mtd.run(
            dbeta=0.001,
            cycles_per_shift=1000,
            shift_dampening=0.05,
            spawn_cutoff=0.01,
            n_add=3,
            ilevel=0
        )

        assert np.isclose(self._mtd.density_matrix.trace(), 13.608499042255167)
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        assert np.isclose(eng, -28.215638757282814)

    def test_ilevel_nonzero(self):
        self._mtd.reset_rng(42)
        self._mtd.setup(
            final_beta=25,
            initialization="random-uniform", 
            n_particles=int(1e5)
        )
        self._mtd.run(
            dbeta=0.001,
            cycles_per_shift=1000,
            shift_dampening=0.05,
            spawn_cutoff=0.01,
            ilevel=2
        )

        assert np.isclose(self._mtd.density_matrix.trace(), 67926.38161070104)
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        assert np.isclose(eng, -141000.99421006895)
