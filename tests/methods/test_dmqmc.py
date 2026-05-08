import numpy as np
from pytest import fixture, raises, mark
from pytest_lazy_fixtures import lf
from os.path import dirname, join, exists

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
        assert self._mtd.final_beta == 1.0

    def test_setup_random_uniform(self):
        diag = np.array([0, 3, 0, 0, 1, 0, 0, 0, 2, 0, 
                        0, 0, 0, 2, 0, 1, 0, 1, 0, 0])

        self._mtd.reset_rng(rng_seed=42)
        self._mtd.setup(1.0, "random-uniform", n_particles=10)
        assert np.allclose(np.diag(self._mtd.density_matrix),
                        diag)
        assert self._mtd.final_beta == 1.0

    def test_setup_fixed(self):
        diag = [10, 30, 40, 25, 18, 54, 22, 34, 47, 36,
                45, 37, 23, 46, 41, 31, 27, 49, 17, 38]

        self._mtd.setup(1.0, "fixed", fixed_diagonal=diag)
        assert np.allclose(np.diag(self._mtd.density_matrix),  diag)
        assert self._mtd.density_matrix.size == 400
        assert self._mtd.final_beta == 1.0

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

    def test_save_data(self):
        self._mtd.setup(1.0, "deterministic")
        with raises(RuntimeError):
            self._mtd.save_data("test")


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

        assert np.isclose(self._mtd.density_matrix.trace(), 67981.4893)
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        assert np.isclose(eng, -141115.3864)

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

        assert np.isclose(self._mtd.density_matrix.trace(), 22493.3789)
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        assert np.isclose(eng, -46578.8490)

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

        assert np.isclose(self._mtd.density_matrix.trace(), 14.2069)
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        assert np.isclose(eng, -29.4613)

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

        assert np.isclose(self._mtd.density_matrix.trace(), 67981.4899)
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        assert np.isclose(eng, -141115.3875)

    def test_save_data(self):
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
        self._mtd.save_data("test_asymmetric_bloch_dmqmc",
                            matrix_filetype="csv",
                            report_filetype="csv")

        assert exists("test_asymmetric_bloch_dmqmc_density_matrix.csv")
        assert exists("test_asymmetric_bloch_dmqmc_report.csv")

        loaded_matrix = np.genfromtxt("test_asymmetric_bloch_dmqmc_density_matrix.csv",
                                      delimiter=',')
        assert np.allclose(loaded_matrix.shape, self._mtd.density_matrix.shape)

        loaded_report = np.genfromtxt("test_asymmetric_bloch_dmqmc_report.csv",
                                     delimiter=',', names=True)
        assert loaded_report.shape[0] == len(self._mtd.report)
        assert loaded_report['beta'][0] == self._mtd.report[0]['beta']
        # Numpy converts spaces to underscores
        assert loaded_report['energy_expectation'][1] == self._mtd.report[1]['energy expectation']


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

        assert np.isclose(self._mtd.density_matrix.trace(), 67926.3811)
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        assert np.isclose(eng, -141000.9932)

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

        assert np.isclose(self._mtd.density_matrix.trace(), 20325.6708)
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        assert np.isclose(eng, -42058.9926)
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

        assert np.isclose(self._mtd.density_matrix.trace(), 13.6085)
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        assert np.isclose(eng, -28.2156)

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

        assert np.isclose(self._mtd.density_matrix.trace(), 67926.3816)
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        assert np.isclose(eng, -141000.9942)


    def test_save_data(self):
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
        self._mtd.save_data("test_symmetric_bloch_dmqmc",
                            matrix_filetype="csv",
                            report_filetype="csv")

        assert exists("test_symmetric_bloch_dmqmc_density_matrix.csv")
        assert exists("test_symmetric_bloch_dmqmc_report.csv")

        loaded_matrix = np.genfromtxt("test_symmetric_bloch_dmqmc_density_matrix.csv",
                                      delimiter=',')
        assert np.allclose(loaded_matrix.shape, self._mtd.density_matrix.shape)

        loaded_report = np.genfromtxt("test_symmetric_bloch_dmqmc_report.csv",
                                     delimiter=',', names=True)
        assert loaded_report.shape[0] == len(self._mtd.report)
        assert loaded_report['beta'][0] == self._mtd.report[0]['beta']
        # Numpy converts spaces to underscores
        assert loaded_report['energy_expectation'][1] == self._mtd.report[1]['energy expectation']