import numpy as np
from pytest import fixture, raises, mark
from os.path import dirname, join
from pytest_mpi import parallel_assert

from pydmqmc.systems import Integral
from pydmqmc.methods import InteractionPictureDMQMC


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


class TestIPDMQMC_Parallel():

    @fixture(autouse=True)
    def _setup(self, integral_system_small, integral_system_large):
        self._mtd_sm = InteractionPictureDMQMC(integral_system_small)
        self._mtd_lg = InteractionPictureDMQMC(integral_system_large)
        self._final_beta = 1.0
        self._nparticle = 100000

    @mark.parallel([1,2,3])
    def test_setup_determinitistic(self):
        ref = np.array([[0.82911633, 0.        ],
                        [0.        , 0.17088367]])

        self._mtd_sm.setup(self._final_beta,
                           "deterministic")

        parallel_assert(np.allclose(self._mtd_sm.density_matrix, ref),
                        msg=f"Density matrix:\n{self._mtd_sm.density_matrix}\nRef:\n{ref}")

    @mark.parallel([1,2,3])
    def test_setup_random_uniform(self):
        ref = np.array([[41382.0252,  0.       ],
                        [    0.    ,  8559.3920]])

        self._mtd_sm.reset_rng(rng_seed=42)
        self._mtd_sm.setup(self._final_beta,
                           "random-uniform",
                           n_particles=self._nparticle)

        parallel_assert(np.allclose(self._mtd_sm.density_matrix, ref),
                        msg=f"Density matrix:\n{self._mtd_sm.density_matrix}\nRef:\n{ref}")

    @mark.parallel([1,2,3])
    def test_setup_random_thermal(self):
        ref = np.array([[82945.,     0.],
                        [    0., 17055.]])

        self._mtd_sm.reset_rng(rng_seed=42)
        self._mtd_sm.setup(self._final_beta,
                           "random-thermal",
                           n_particles=self._nparticle)

        parallel_assert(np.allclose(self._mtd_sm.density_matrix, ref),
                        msg=f"Density matrix:\n{self._mtd_sm.density_matrix}\nRef:\n{ref}")

    @mark.parallel([1,2,3])
    def test_setup_random_grand_canonical(self):
        ref = np.array([11521.,  7595.,  7687.,  6636.,  6635.,  6528.,  4917.,  4973.,
                        4627.,  4466.,  4685.,  4528.,  4546.,  3306.,  3364.,  3431.,
                        2840.,  2748.,  2850.,  2121.])

        self._mtd_lg.reset_rng(rng_seed=42)
        self._mtd_lg.setup(self._final_beta,
                           "random-grand-canonical",
                           gc_spawn_cutoff=0.01,
                           n_particles=self._nparticle)

        parallel_assert(np.allclose(np.diag(self._mtd_lg.density_matrix),
                           ref),
                        msg=f"Density matrix diagonal:\n{np.diag(self._mtd_lg.density_matrix)}\nRef:\n{ref}")

    @mark.parallel([1,2,3])
    def test_setup_fixed(self):
        diag = [10, 30]

        self._mtd_sm.setup(self._final_beta,
                           "fixed", fixed_diagonal=diag)
        parallel_assert(np.allclose(np.diag(self._mtd_sm.density_matrix),
                           diag),
                        msg=f"Density matrix diagonal:\n{np.diag(self._mtd_sm.density_matrix)}\nRef:\n{diag}")
        parallel_assert(self._mtd_sm.density_matrix.size == 4,
                        msg=f"Density matrix size: {self._mtd_sm.density_matrix.size}\nExpected: 4")

    @mark.parallel([1,2,3])
    def test_setup_fixed_invalid(self):
        diag = [10, 30, 40]

        with raises(RuntimeError):
            self._mtd_sm.setup(self._final_beta,
                               "fixed", fixed_diagonal=diag)

    @mark.parallel([1,2,3])
    def test_setup_unknown(self):
        with raises(RuntimeError):
            self._mtd_sm.setup(self._final_beta,
                               "bad-method")

    @mark.parallel([1,2,3])
    def test_basic(self):
        """
        Implicitly tests dummy matrix created for ilevel = None and ilevel = 0.
        """
        self._mtd_lg.reset_rng(42)
        self._mtd_lg.setup(final_beta=self._final_beta,
                           initialization="random-grand-canonical",
                           n_particles=self._nparticle,
                           gc_spawn_cutoff=0.01)
        self._mtd_lg.run(dbeta=0.001,
                         cycles_per_shift=10,
                         shift_dampening=0.05,
                         spawn_cutoff=0.01,
                         shift_by_rows=False)

        parallel_assert(np.isclose(self._mtd_lg.density_matrix.trace(), 52340.7535),
                        msg=f"Density matrix trace: {self._mtd_lg.density_matrix.trace()}\nExpected: 52340.7535")
        eng = (self._mtd_lg.density_matrix @ self._mtd_lg.system.hamiltonian).trace()
        parallel_assert(np.isclose(eng, -69386.9260),
                        msg=f"Energy: {eng}\nExpected: -69386.9260")

    @mark.parallel([1,2,3])
    def test_rbr(self):
        self._mtd_lg.reset_rng(42)
        self._mtd_lg.setup(final_beta=self._final_beta,
                           initialization="random-grand-canonical",
                           n_particles=self._nparticle,
                           gc_spawn_cutoff=0.01)
        self._mtd_lg.run(dbeta=0.001,
                         cycles_per_shift=10,
                         shift_dampening=0.05,
                         spawn_cutoff=0.01,
                         shift_by_rows=True)

        parallel_assert(np.isclose(self._mtd_lg.density_matrix.trace(), 53533.6315),
                        msg=f"Density matrix trace: {self._mtd_lg.density_matrix.trace()}\nExpected: 53533.6315")
        eng = (self._mtd_lg.density_matrix @ self._mtd_lg.system.hamiltonian).trace()
        parallel_assert(np.isclose(eng, -73295.6925),
                        msg=f"Energy: {eng}\nExpected: -73295.6925")

    @mark.parallel([1,2,3])
    def test_ilevel_zero(self):
        """
        Test functionality of ilevel = 0 (and dummy matrix functionality).

        Set density matrix and n_add such that psip spawning 
        using ilevel=0 is emphasized.
        """
        self._mtd_lg.reset_rng(42)
        self._mtd_lg.setup(self._final_beta,
                           "deterministic",
                           n_particles=self._nparticle)
        self._mtd_lg.run(dbeta=0.001,
                         cycles_per_shift=10,
                         shift_dampening=0.05,
                         spawn_cutoff=0.01,
                         n_add=3,  # strongly limit this spawn channel to emph ilevel
                         ilevel=0)

        parallel_assert(np.isclose(self._mtd_lg.density_matrix.trace(), 0.54385963),
                        msg=f"Density matrix trace: {self._mtd_lg.density_matrix.trace()}\nExpected: 0.54385963")
        eng = (self._mtd_lg.density_matrix @ self._mtd_lg.system.hamiltonian).trace()
        parallel_assert(np.isclose(eng, -0.72093397),
                        msg=f"Energy: {eng}\nExpected: -0.72093397")

    @mark.parallel([1,2,3])
    def test_ilevel_nonzero(self):
        self._mtd_lg.reset_rng(42)
        self._mtd_lg.setup(self._final_beta,
                           "deterministic",
                           n_particles=self._nparticle)
        self._mtd_lg.run(dbeta=0.001,
                         cycles_per_shift=10,
                         shift_dampening=0.05,
                         spawn_cutoff=0.01,
                         n_add=3,  # strongly limit this spawn channel to emph ilevel
                         ilevel=2)

        parallel_assert(np.isclose(self._mtd_lg.density_matrix.trace(), 0.52477479),
                        msg=f"Density matrix trace: {self._mtd_lg.density_matrix.trace()}\nExpected: 0.52477479")
        eng = (self._mtd_lg.density_matrix @ self._mtd_lg.system.hamiltonian).trace()
        parallel_assert(np.isclose(eng, -0.69688129),
                        msg=f"Energy: {eng}\nExpected: -0.69688129")
