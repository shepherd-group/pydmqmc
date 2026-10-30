import numpy as np
from pytest import fixture, raises, mark
from pytest_lazy_fixtures import lf
from os.path import dirname, join, exists
from pytest_mpi import parallel_assert

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


class TestDMQMC_Parallel():

    @fixture(autouse=True)
    def _setup(self, matrix_system):
        self._mtd = DensityMatrixQMC(matrix_system, parallel=True)

    @mark.parametrize("system", 
                      [lf('matrix_system'), 
                       lf('integral_system_small')])
    @mark.parallel([1,2])
    def test_init(self, system):
        mtd = DensityMatrixQMC(system, parallel=True)

        parallel_assert(mtd.system is system)
        parallel_assert(mtd.parallel)
        parallel_assert(self._mtd.parallel_size in [1, 2, 3])
        parallel_assert(self._mtd.parallel_rank in [0, 1, 2])
        parallel_assert(self._mtd.parallel_is_parent in [True, False])
        parallel_assert(self._mtd.is_reporter in [True, False])

    @mark.parallel([1,2])
    def test_setup_determinitistic(self):
        self._mtd.setup(1.0, "deterministic")
        nrows = self._mtd._ph.nrows
        parallel_assert(
            self._mtd.density_matrix.shape
            == (nrows, self._mtd.system.n_determinants),
            msg=f"Density matrix shape: {self._mtd.density_matrix.shape}\n"
            f"Expected: {(nrows, self._mtd.system.n_determinants)}",
        )
        expected = np.zeros_like(self._mtd.density_matrix)
        local_rows = np.arange(self._mtd._ph.imin, self._mtd._ph.imax)
        expected[np.arange(nrows), local_rows] = 1.0
        parallel_assert(
            np.array_equal(self._mtd.density_matrix, expected),
            msg=f"Density matrix:\n{self._mtd.density_matrix}\nExpected:\n{expected}",
        )

    @mark.parallel([1,2])
    def test_setup_random_uniform(self):
        diag = np.array([0, 3, 0, 0, 1, 0, 0, 0, 2, 0, 
                        0, 0, 0, 2, 0, 1, 0, 1, 0, 0])

        self._mtd.reset_rng(rng_seed=42)
        self._mtd.setup(1.0, "random-uniform", n_particles=10)
        local_rows = np.arange(self._mtd._ph.imin, self._mtd._ph.imax)
        local_diag = self._mtd.density_matrix[
            np.arange(self._mtd._ph.nrows), local_rows
        ]
        parallel_assert(
            np.array_equal(local_diag, np.asarray(diag)[local_rows]),
            msg=f"Local diagonal: {local_diag}\nExpected: {np.asarray(diag)[local_rows]}",
        )

    @mark.parallel([1,2])
    def test_setup_fixed(self):
        diag = [10, 30, 40, 25, 18, 54, 22, 34, 47, 36,
                45, 37, 23, 46, 41, 31, 27, 49, 17, 38]

        self._mtd.setup(1.0, "fixed", fixed_diagonal=diag)
        local_rows = np.arange(self._mtd._ph.imin, self._mtd._ph.imax)
        local_diag = self._mtd.density_matrix[
            np.arange(self._mtd._ph.nrows), local_rows
        ]
        parallel_assert(
            np.array_equal(local_diag, np.asarray(diag)[local_rows]),
            msg=f"Local diagonal: {local_diag}\nExpected: {np.asarray(diag)[local_rows]}",
        )
        parallel_assert(
            self._mtd.density_matrix.size
            == self._mtd._ph.nrows * self._mtd.system.n_determinants,
            msg=f"Density matrix size: {self._mtd.density_matrix.size}\n"
            f"Expected: {self._mtd._ph.nrows * self._mtd.system.n_determinants}",
        )

    @mark.parallel([1,2])
    def test_setup_fixed_invalid(self):
        diag = [10, 30, 40]

        with raises(RuntimeError):
            self._mtd.setup(1.0, "fixed", fixed_diagonal=diag)

    @mark.parallel([1,2])
    def test_setup_unknown(self):
        with raises(RuntimeError):
            self._mtd.setup(1.0, "bad-method")

    @mark.parallel([1,2])
    def test_run(self):
        self._mtd.setup(1.0, "deterministic")

        with raises(NotImplementedError):
            self._mtd.run(0.01, 10, 0.05)

    @mark.parallel([1,2])
    def test_run_no_setup(self):
        with raises(RuntimeError):
            self._mtd.run(0.01, 10, 0.05)

    @mark.parallel([1,2])
    def test_run_invalid_ilevel(self):
        self._mtd.setup(1.0, "deterministic")

        with raises(TypeError):
            self._mtd.run(0.01, 10, 0.05, ilevel=0.1)


class TestAsymmetricBlochDMQMC_Parallel():

    @fixture(autouse=True)
    def _setup(self, integral_system_large):
        self._mtd = AsymmetricBlochDMQMC(integral_system_large, parallel=True)

    @mark.parallel([1,2])
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

        traces = {1: 67981.12, 2: 67990.44}
        energies = {1: -141115.79, 2: -141133.32}

        matrix = self._mtd._report_matrix if self._mtd.is_reporter else None
        trace = matrix.trace() if self._mtd.is_reporter else 0.0
        parallel_assert(
            np.isclose(trace, traces[self._mtd.parallel_size]),
            participating=self._mtd.is_reporter,
            msg=f"Density matrix trace: {trace}\n"
            f"Expected: {traces[self._mtd.parallel_size]}",
        )
        eng = (
            (matrix @ self._mtd.system.hamiltonian).trace()
            if self._mtd.is_reporter
            else 0.0
        )
        parallel_assert(np.isclose(eng, energies[self._mtd.parallel_size]),
                        participating=self._mtd.is_reporter,
            msg=f"Energy: {eng}\nExpected: {energies[self._mtd.parallel_size]}")

    @mark.parallel([1,2])
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

        traces = {1: 22497.60,
                  2: 22495.65}
        energies = {1: -46587.10,
                    2: -46583.27}

        matrix = self._mtd._report_matrix if self._mtd.is_reporter else None
        trace = matrix.trace() if self._mtd.is_reporter else 0.0
        parallel_assert(np.isclose(trace, traces[self._mtd.parallel_size]),
            participating=self._mtd.is_reporter,
            msg=f"Density matrix trace: {trace}\n"
            f"Expected: {traces[self._mtd.parallel_size]}")
        eng = (
            (matrix @ self._mtd.system.hamiltonian).trace()
            if self._mtd.is_reporter
            else 0.0
        )
        parallel_assert(np.isclose(eng, energies[self._mtd.parallel_size]),
                participating=self._mtd.is_reporter,
            msg=f"Energy: {eng}\nExpected: {energies[self._mtd.parallel_size]}")
    @mark.parallel([1,2])
    def test_ilevel_zero(self):
        """
        Test functionality of ilevel = 0.

        Set density matrix and n_add such that psip spawning using ilevel=0 is emphasized.
        This does, however, mean the simulation is not well-converged.
        Since the RNG seed varies across ranks, the results will also vary slightly
        with the number of processes.
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

        traces = {1: 13.0736,
                  2: 14.3035}
        energies = {1: -27.1111,
                    2: -29.6655}

        matrix = self._mtd._report_matrix if self._mtd.is_reporter else None
        trace = matrix.trace() if self._mtd.is_reporter else 0.0
        parallel_assert(np.isclose(trace, traces[self._mtd.parallel_size]),
            participating=self._mtd.is_reporter,
            msg=f"Density matrix trace: {trace}\n"
            f"Expected: {traces[self._mtd.parallel_size]}")
        eng = (
            (matrix @ self._mtd.system.hamiltonian).trace()
            if self._mtd.is_reporter
            else 0.0
        )
        parallel_assert(np.isclose(eng, energies[self._mtd.parallel_size]),
                participating=self._mtd.is_reporter,
            msg=f"Energy: {eng}\nExpected: {energies[self._mtd.parallel_size]}")

    @mark.parallel([1,2])
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

        traces = {1: 67973.89,
                  2: 67975.36}
        
        energies = {1: -141098.78,
                    2: -141102.91}

        matrix = self._mtd._report_matrix if self._mtd.is_reporter else None
        trace = matrix.trace() if self._mtd.is_reporter else 0.0
        parallel_assert(np.isclose(trace, traces[self._mtd.parallel_size]),
            participating=self._mtd.is_reporter,
            msg=f"Density matrix trace: {trace}\n"
            f"Expected: {traces[self._mtd.parallel_size]}")
        eng = (
            (matrix @ self._mtd.system.hamiltonian).trace()
            if self._mtd.is_reporter
            else 0.0
        )
        parallel_assert(np.isclose(eng, energies[self._mtd.parallel_size]),
                participating=self._mtd.is_reporter,
                        msg=f"Energy: {eng}\nExpected: {energies[self._mtd.parallel_size]}")

    @mark.parallel([1,2])
    def test_save_data(self, tmp_path):
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
        self._mtd.save_data(str(tmp_path/"test_asymmetric_bloch_dmqmc"),
                            matrix_filetype="csv",
                            report_filetype="csv")

        if self._mtd.is_reporter:

            file_matrix = tmp_path/"test_asymmetric_bloch_dmqmc_density_matrix.csv"
            file_report = tmp_path/"test_asymmetric_bloch_dmqmc_report.csv"

            assert exists(file_matrix)
            assert exists(file_report)

            loaded_matrix = np.genfromtxt(file_matrix, delimiter=',')
            assert loaded_matrix.shape == self._mtd._report_matrix.shape

            loaded_report = np.genfromtxt(file_report, delimiter=',', names=True)
            assert loaded_report.shape[0] == len(self._mtd.report)
            assert loaded_report['beta'][0] == self._mtd.report[0]['beta']
            # Numpy converts spaces to underscores
            assert loaded_report['energy_numerator'][1] == self._mtd.report[1]['energy numerator']


class TestSymmetricBlochDMQMC_Parallel():

    @fixture(autouse=True)
    def _setup(self, integral_system_large):
        self._mtd = SymmetricBlochDMQMC(integral_system_large, parallel=True)

    @mark.parallel([1,2])
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

        traces = {1: 67929.32, 2: 67923.88}
        energies = {1: -141008.19, 2: -140996.96}

        matrix = self._mtd._report_matrix if self._mtd.is_reporter else None
        trace = matrix.trace() if self._mtd.is_reporter else 0.0
        parallel_assert(np.isclose(trace, traces[self._mtd.parallel_size]),
                        participating=self._mtd.is_reporter,
                        msg=f"Density matrix trace: {trace}\n"
                        f"Expected: {traces[self._mtd.parallel_size]}")
        eng = ((matrix @ self._mtd.system.hamiltonian).trace()
               if self._mtd.is_reporter else 0.0)
        parallel_assert(np.isclose(eng, energies[self._mtd.parallel_size]),
                        participating=self._mtd.is_reporter,
                        msg=f"Energy: {eng}\nExpected: {energies[self._mtd.parallel_size]}")

    @mark.parallel([1,2])
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

        traces = {1: 20328.17, 2: 20327.10}
        energies = {1: -42063.25, 2: -42059.50}

        matrix = self._mtd._report_matrix if self._mtd.is_reporter else None
        trace = matrix.trace() if self._mtd.is_reporter else 0.0
        parallel_assert(np.isclose(trace, traces[self._mtd.parallel_size]),
                        participating=self._mtd.is_reporter,
                        msg=f"Density matrix trace: {trace}\n"
                        f"Expected: {traces[self._mtd.parallel_size]}")
        eng = ((matrix @ self._mtd.system.hamiltonian).trace()
               if self._mtd.is_reporter else 0.0)
        parallel_assert(np.isclose(eng, energies[self._mtd.parallel_size]),
                        participating=self._mtd.is_reporter,
                        msg=f"Energy: {eng}\nExpected: {energies[self._mtd.parallel_size]}")

    @mark.parallel([1,2])
    def test_ilevel_zero(self):
        """
        Test functionality of ilevel = 0.

        Set density matrix and n_add such that psip spawning using ilevel=0 is emphasized.
        This does, however, mean the simulation is not well-converged.
        Since the RNG seed varies across ranks, the results will also vary slightly
        with the number of processes.
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

        traces = {1: 14.1179, 2: 13.9435}
        energies = {1: -29.1850, 2: -28.8694}

        matrix = self._mtd._report_matrix if self._mtd.is_reporter else None
        trace = matrix.trace() if self._mtd.is_reporter else 0.0
        parallel_assert(np.isclose(trace, traces[self._mtd.parallel_size]),
                        participating=self._mtd.is_reporter,
                        msg=f"Density matrix trace: {trace}\n"
                        f"Expected: {traces[self._mtd.parallel_size]}")
        eng = ((matrix @ self._mtd.system.hamiltonian).trace()
               if self._mtd.is_reporter else 0.0)
        parallel_assert(np.isclose(eng, energies[self._mtd.parallel_size]),
                        participating=self._mtd.is_reporter,
                        msg=f"Energy: {eng}\nExpected: {energies[self._mtd.parallel_size]}")

    @mark.parallel([1,2])
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

        traces = {1: 67927.84, 2: 67926.33}
        energies = {1: -141004.61, 2: -140999.38}

        matrix = self._mtd._report_matrix if self._mtd.is_reporter else None
        trace = matrix.trace() if self._mtd.is_reporter else 0.0
        parallel_assert(np.isclose(trace, traces[self._mtd.parallel_size]),
                        participating=self._mtd.is_reporter,
                        msg=f"Density matrix trace: {trace}\n"
                        f"Expected: {traces[self._mtd.parallel_size]}")
        eng = ((matrix @ self._mtd.system.hamiltonian).trace()
               if self._mtd.is_reporter else 0.0)
        parallel_assert(np.isclose(eng, energies[self._mtd.parallel_size]),
                        participating=self._mtd.is_reporter,
                        msg=f"Energy: {eng}\nExpected: {energies[self._mtd.parallel_size]}")

    @mark.parallel([1,2])
    def test_save_data(self, tmp_path):
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
        self._mtd.save_data(str(tmp_path/"test_symmetric_bloch_dmqmc"),
                            matrix_filetype="csv",
                            report_filetype="csv")

        if self._mtd.is_reporter:

            file_matrix = tmp_path/"test_symmetric_bloch_dmqmc_density_matrix.csv"
            file_report = tmp_path/"test_symmetric_bloch_dmqmc_report.csv"

            assert exists(file_matrix)
            assert exists(file_report)

            loaded_matrix = np.genfromtxt(file_matrix, delimiter=',')
            assert loaded_matrix.shape == self._mtd._report_matrix.shape

            loaded_report = np.genfromtxt(file_report, delimiter=',', names=True)
            assert loaded_report.shape[0] == len(self._mtd.report)
            assert loaded_report['beta'][0] == self._mtd.report[0]['beta']
            # Numpy converts spaces to underscores
            assert loaded_report['energy_numerator'][1] == self._mtd.report[1]['energy numerator']