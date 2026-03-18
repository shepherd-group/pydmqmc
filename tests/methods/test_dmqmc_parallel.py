import numpy as np
from pytest import fixture, raises, mark
from pytest_lazy_fixtures import lf
from os.path import dirname, join
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

    @mark.parallel([1,2,3])
    def test_setup_determinitistic(self):
        self._mtd.setup(1.0, "deterministic")
        parallel_assert(np.allclose(self._mtd.density_matrix, 
                        np.eye(self._mtd.system.n_determinants)),
                        msg=f"Density matrix:\n{self._mtd.density_matrix}\nExpected:\n{np.eye(self._mtd.system.n_determinants)}")

    @mark.parallel([1,2,3])
    def test_setup_random_uniform(self):
        diag = np.array([0, 3, 0, 0, 1, 0, 0, 0, 2, 0, 
                        0, 0, 0, 2, 0, 1, 0, 1, 0, 0])

        self._mtd.reset_rng(rng_seed=42)
        self._mtd.setup(1.0, "random-uniform", n_particles=10)
        parallel_assert(np.allclose(np.diag(self._mtd.density_matrix),
                        diag),
                        msg=f"Density matrix diagonal:\n{np.diag(self._mtd.density_matrix)}\nExpected:\n{diag}")

    @mark.parallel([1,2,3])
    def test_setup_fixed(self):
        diag = [10, 30, 40, 25, 18, 54, 22, 34, 47, 36,
                45, 37, 23, 46, 41, 31, 27, 49, 17, 38]

        self._mtd.setup(1.0, "fixed", fixed_diagonal=diag)
        parallel_assert(np.allclose(np.diag(self._mtd.density_matrix),
                        diag),
                        msg=f"Density matrix diagonal:\n{np.diag(self._mtd.density_matrix)}\nExpected:\n{diag}")
        parallel_assert(self._mtd.density_matrix.size == 400,
                        msg=f"Density matrix size: {self._mtd.density_matrix.size}\nExpected: 400")

    @mark.parallel([1,2,3])
    def test_setup_fixed_invalid(self):
        diag = [10, 30, 40]

        with raises(RuntimeError):
            self._mtd.setup(1.0, "fixed", fixed_diagonal=diag)

    @mark.parallel([1,2,3])
    def test_setup_unknown(self):
        with raises(RuntimeError):
            self._mtd.setup(1.0, "bad-method")

    @mark.parallel([1,2,3])
    def test_run(self):
        self._mtd.setup(1.0, "deterministic")

        with raises(NotImplementedError):
            self._mtd.run(0.01, 10, 0.05)

    @mark.parallel([1,2,3])
    def test_run_no_setup(self):
        with raises(RuntimeError):
            self._mtd.run(0.01, 10, 0.05)

    @mark.parallel([1,2,3])
    def test_run_invalid_ilevel(self):
        self._mtd.setup(1.0, "deterministic")

        with raises(TypeError):
            self._mtd.run(0.01, 10, 0.05, ilevel=0.1)


class TestAsymmetricBlochDMQMC_Parallel():

    @fixture(autouse=True)
    def _setup(self, integral_system_large):
        self._mtd = AsymmetricBlochDMQMC(integral_system_large, parallel=True)

    @mark.parallel([1,2,3])
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

        parallel_assert(np.isclose(self._mtd.density_matrix.trace(), 67981.4893),
                        msg=f"Density matrix trace: {self._mtd.density_matrix.trace()}\nExpected: 67981.4893")
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        parallel_assert(np.isclose(eng, -141115.3864),
                        msg=f"Energy: {eng}\nExpected: -141115.3864")

    @mark.parallel([1,2,3])
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

        parallel_assert(np.isclose(self._mtd.density_matrix.trace(), 22493.3789),
                        msg=f"Density matrix trace: {self._mtd.density_matrix.trace()}\nExpected: 22493.3789")
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        parallel_assert(np.isclose(eng, -46578.8490),
                        msg=f"Energy: {eng}\nExpected: -46578.8490")

    @mark.parallel([1,2,3])
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

        traces = {1: 14.2069,
                  2: 14.1996,
                  3: 14.2261}
        
        energies = {1: -29.4613,
                    2: -29.4470,
                    3: -29.5008}

        parallel_assert(np.isclose(self._mtd.density_matrix.trace(), traces[self._mtd.parallel_size]),
                        msg=f"Density matrix trace: {self._mtd.density_matrix.trace()}\nExpected: {traces[self._mtd.parallel_size]}")
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        parallel_assert(np.isclose(eng, energies[self._mtd.parallel_size]),
                        msg=f"Energy: {eng}\nExpected: {energies[self._mtd.parallel_size]}")

    @mark.parallel([1,2,3])
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

        parallel_assert(np.isclose(self._mtd.density_matrix.trace(), 67981.4899),
                        msg=f"Density matrix trace: {self._mtd.density_matrix.trace()}\nExpected: 67981.4899")
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        parallel_assert(np.isclose(eng, -141115.3875),
                        msg=f"Energy: {eng}\nExpected: -141115.3875")


class TestSymmetricBlochDMQMC_Parallel():

    @fixture(autouse=True)
    def _setup(self, integral_system_large):
        self._mtd = SymmetricBlochDMQMC(integral_system_large, parallel=True)

    @mark.parallel([1,2,3])
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

        parallel_assert(np.isclose(self._mtd.density_matrix.trace(), 67926.3811),
                        msg=f"Density matrix trace: {self._mtd.density_matrix.trace()}\nExpected: 67926.3811")
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        parallel_assert(np.isclose(eng, -141000.9932),
                        msg=f"Energy: {eng}\nExpected: -141000.9932")

    @mark.parallel([1,2,3])
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

        parallel_assert(np.isclose(self._mtd.density_matrix.trace(), 20325.6708),
                        msg=f"Density matrix trace: {self._mtd.density_matrix.trace()}\nExpected: 20325.6708")
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        parallel_assert(np.isclose(eng, -42058.9926),
                        msg=f"Energy: {eng}\nExpected: -42058.9926")

    @mark.parallel([1,2,3])
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

        traces = {1: 13.6085,
                  2: 13.6090,
                  3: 13.6263}
        energies = {1: -28.2156,
                    2: -28.2128,
                    3: -28.2521}

        parallel_assert(np.isclose(self._mtd.density_matrix.trace(), traces[self._mtd.parallel_size]),
                        msg=f"Density matrix trace: {self._mtd.density_matrix.trace()}\nExpected: {traces[self._mtd.parallel_size]}")
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        parallel_assert(np.isclose(eng, energies[self._mtd.parallel_size]),
                        msg=f"Energy: {eng}\nExpected: {energies[self._mtd.parallel_size]}")

    @mark.parallel([1,2,3])
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

        parallel_assert(np.isclose(self._mtd.density_matrix.trace(), 67926.3816),
                        msg=f"Density matrix trace: {self._mtd.density_matrix.trace()}\nExpected: 67926.3816")
        eng = (self._mtd.density_matrix @ self._mtd.system.hamiltonian).trace()
        parallel_assert(np.isclose(eng, -141000.9942),
                        msg=f"Energy: {eng}\nExpected: -141000.9942")
