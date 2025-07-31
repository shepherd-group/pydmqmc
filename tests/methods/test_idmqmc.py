import numpy as np
from pytest import fixture, raises, mark
from pytest_lazy_fixtures import lf
from os.path import dirname, join

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


class TestIPDMQMC():

    @fixture(autouse=True)
    def _setup(self, integral_system_small, integral_system_large):
        self._mtd_sm = InteractionPictureDMQMC(integral_system_small)
        self._mtd_lg = InteractionPictureDMQMC(integral_system_large)
        self._final_beta = 1.0
        self._nparticle = 100000

    def test_setup_determinitistic(self):
        ref = np.array([[0.82911633, 0.        ],
                        [0.        , 0.17088367]])

        self._mtd_sm.setup(self._final_beta,
                           "deterministic")

        assert np.allclose(self._mtd_sm.density_matrix, ref)

    def test_setup_random_uniform(self):
        ref = np.array([[41382.02524792,     0.        ],
                        [    0.        ,  8559.39204498]])

        self._mtd_sm.reset_rng(rng_seed=42)
        self._mtd_sm.setup(self._final_beta,
                           "random-uniform",
                           n_particles=self._nparticle)

        assert np.allclose(self._mtd_sm.density_matrix, ref)

    def test_setup_random_thermal(self):
        ref = np.array([[82945.,     0.],
                        [    0., 17055.]])

        self._mtd_sm.reset_rng(rng_seed=42)
        self._mtd_sm.setup(self._final_beta,
                           "random-thermal",
                           n_particles=self._nparticle)

        assert np.allclose(self._mtd_sm.density_matrix, ref)

    def test_setup_random_grand_canonical(self):
        ref = np.array([15386., 10196., 10300.,  8896.,
                        8802.,  8712.,  6525.,  6572.,
                        6188.,  6048.,  6319.,  5985.,
                        6037.,  4437.,  4406.,  4455.,
                        3772.,  3709.,  3700.,  2864.])

        self._mtd_lg.reset_rng(rng_seed=42)
        self._mtd_lg.setup(self._final_beta,
                           "random-grand-canonical",
                           spawn_cutoff=0.01,
                           n_particles=self._nparticle)

        assert np.allclose(np.diag(self._mtd_lg.density_matrix),
                           ref)

    def test_setup_fixed(self):
        diag = [10, 30]

        self._mtd_sm.setup(self._final_beta,
                           "fixed", fixed_diagonal=diag)
        assert np.allclose(np.diag(self._mtd_sm.density_matrix),
                           diag)
        assert self._mtd_sm.density_matrix.size == 4

    def test_setup_fixed_bad(self):
        diag = [10, 30, 40]

        with raises(RuntimeError):
            self._mtd_sm.setup(self._final_beta,
                               "fixed", fixed_diagonal=diag)

    def test_setup_unknown(self):
        with raises(RuntimeError):
            self._mtd_sm.setup(self._final_beta,
                               "bad-method")

    def test_basic(self):
        """
        Implicitly tests dummy matrix created for ilevel = None and ilevel = 0.
        """
        self._mtd_lg.reset_rng(42)
        self._mtd_lg.setup("random-grand-canonical", self._nparticle)
        self._mtd_lg.run(final_beta=self._final_beta,
                         dbeta=0.001,
                         cycles_per_shift=10,
                         shift_dampening=0.05,
                         spawn_cutoff=0.01,
                         shift_by_rows=False)

        assert np.isclose(self._mtd_lg.density_matrix.trace(), 67981.48932281222)
        eng = (self._mtd_lg.density_matrix @ self._mtd_lg.system.hamiltonian).trace()
        assert np.isclose(eng, -141115.38639919003)

    def test_rbr(self):
        self._mtd_lg.reset_rng(42)
        self._mtd_lg.setup("random-grand-canonical", self._nparticle)
        self._mtd_lg.run(final_beta=self._final_beta,
            dbeta=0.001,
            cycles_per_shift=1000,
            shift_dampening=0.05,
            spawn_cutoff=0.01,
            shift_by_rows=True)

        assert np.isclose(self._mtd_lg.density_matrix.trace(), 22493.37887777515)
        eng = (self._mtd_lg.density_matrix @ self._mtd_lg.system.hamiltonian).trace()
        assert np.isclose(eng, -46578.848998115835)

    def test_ilevel_zero(self):
        """
        Test functionality of ilevel = 0 (and dummy matrix functionality).

        Set density matrix and n_add such that psip spawning 
        using ilevel=0 is emphasized.
        """
        self._mtd_lg.reset_rng(42)
        self._mtd_lg.setup("deterministic")
        self._mtd_lg.run(final_beta=self._final_beta,
            dbeta=0.001,
            cycles_per_shift=1000,
            shift_dampening=0.05,
            spawn_cutoff=0.01,
            n_add=3,  # strongly limit this spawn channel to emph ilevel
            ilevel=0)

        assert np.isclose(self._mtd_lg.density_matrix.trace(), 14.206870483605295)
        eng = (self._mtd_lg.density_matrix @ self._mtd_lg.system.hamiltonian).trace()
        assert np.isclose(eng, -29.461275823860465)

    def test_ilevel_nonzero(self):
        self._mtd_lg.reset_rng(42)
        self._mtd_lg.setup("random-grand-canonical", self._nparticle)
        self._mtd_lg.run(final_beta=self._final_beta,
            dbeta=0.001,
            cycles_per_shift=1000,
            shift_dampening=0.05,
            spawn_cutoff=0.01,
            ilevel=2)

        assert np.isclose(self._mtd_lg.density_matrix.trace(), 67981.48986893434)
        eng = (self._mtd_lg.density_matrix @ self._mtd_lg.system.hamiltonian).trace()
        assert np.isclose(eng, -141115.3875013612)
