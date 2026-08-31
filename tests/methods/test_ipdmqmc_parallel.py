import numpy as np
from pytest import fixture, raises, mark
from os.path import dirname, join, exists
from pytest_mpi import parallel_assert

from pydmqmc.systems import Integral
from pydmqmc.methods import InteractionPictureDMQMC, PiecewiseIPDMQMC


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

    @mark.parallel([1,2])
    def test_setup_determinitistic(self):
        ref = np.array([[0.82911633, 0.        ],
                        [0.        , 0.17088367]])

        self._mtd_sm.setup(self._final_beta,
                           "deterministic")

        parallel_assert(np.allclose(self._mtd_sm.density_matrix, ref),
                        msg=f"Density matrix:\n{self._mtd_sm.density_matrix}\nRef:\n{ref}")

    @mark.parallel([1,2])
    def test_setup_random_uniform(self):
        ref = np.array([[41382.0252,  0.       ],
                        [    0.    ,  8559.3920]])

        self._mtd_sm.reset_rng(rng_seed=42)
        self._mtd_sm.setup(self._final_beta,
                           "random-uniform",
                           n_particles=self._nparticle)

        parallel_assert(np.allclose(self._mtd_sm.density_matrix, ref),
                        msg=f"Density matrix:\n{self._mtd_sm.density_matrix}\nRef:\n{ref}")

    @mark.parallel([1,2])
    def test_setup_random_thermal(self):
        ref = np.array([[82945.,     0.],
                        [    0., 17055.]])

        self._mtd_sm.reset_rng(rng_seed=42)
        self._mtd_sm.setup(self._final_beta,
                           "random-thermal",
                           n_particles=self._nparticle)

        parallel_assert(np.allclose(self._mtd_sm.density_matrix, ref),
                        msg=f"Density matrix:\n{self._mtd_sm.density_matrix}\nRef:\n{ref}")

    @mark.parallel([1,2])
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

    @mark.parallel([1,2])
    def test_setup_fixed(self):
        diag = [10, 30]

        self._mtd_sm.setup(self._final_beta,
                           "fixed", fixed_diagonal=diag)
        parallel_assert(np.allclose(np.diag(self._mtd_sm.density_matrix),
                           diag),
                        msg=f"Density matrix diagonal:\n{np.diag(self._mtd_sm.density_matrix)}\nRef:\n{diag}")
        parallel_assert(self._mtd_sm.density_matrix.size == 4,
                        msg=f"Density matrix size: {self._mtd_sm.density_matrix.size}\nExpected: 4")

    @mark.parallel([1,2])
    def test_setup_fixed_invalid(self):
        diag = [10, 30, 40]

        with raises(RuntimeError):
            self._mtd_sm.setup(self._final_beta,
                               "fixed", fixed_diagonal=diag)

    @mark.parallel([1,2])
    def test_setup_unknown(self):
        with raises(RuntimeError):
            self._mtd_sm.setup(self._final_beta,
                               "bad-method")

    @mark.parallel([1,2])
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

        parallel_assert(np.isclose(self._mtd_lg.density_matrix.trace(), 52342.66),
                        msg=f"Density matrix trace: {self._mtd_lg.density_matrix.trace()}\nExpected: 52342.66")
        eng = (self._mtd_lg.density_matrix @ self._mtd_lg.system.hamiltonian).trace()
        parallel_assert(np.isclose(eng, -69388.25),
                        msg=f"Energy: {eng}\nExpected: -69388.25")

    @mark.parallel([1,2])
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

        parallel_assert(np.isclose(self._mtd_lg.density_matrix.trace(), 53532.82),
                        msg=f"Density matrix trace: {self._mtd_lg.density_matrix.trace()}\nExpected: 53532.82")
        eng = (self._mtd_lg.density_matrix @ self._mtd_lg.system.hamiltonian).trace()
        parallel_assert(np.isclose(eng, -73294.37),
                        msg=f"Energy: {eng}\nExpected: -73294.37")

    @mark.parallel([1,2])
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

        parallel_assert(np.isclose(self._mtd_lg.density_matrix.trace(), 0.477507),
                        msg=f"Density matrix trace: {self._mtd_lg.density_matrix.trace()}\nExpected: 0.477507")
        eng = (self._mtd_lg.density_matrix @ self._mtd_lg.system.hamiltonian).trace()
        parallel_assert(np.isclose(eng, -0.634194),
                        msg=f"Energy: {eng}\nExpected: -0.634194")

    @mark.parallel([1,2])
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

        parallel_assert(np.isclose(self._mtd_lg.density_matrix.trace(), 0.442294),
                        msg=f"Density matrix trace: {self._mtd_lg.density_matrix.trace()}\nExpected: 0.442294")
        eng = (self._mtd_lg.density_matrix @ self._mtd_lg.system.hamiltonian).trace()
        parallel_assert(np.isclose(eng, -0.588148),
                        msg=f"Energy: {eng}\nExpected: -0.588148")

    @mark.parallel([1,2])
    def test_save_data(self):
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
        self._mtd_lg.save_data("test_ipdmqmc",
                            matrix_filetype="csv",
                            report_filetype="csv")

        if self._mtd_lg.is_reporter:

            assert exists("test_ipdmqmc_density_matrix.csv")
            assert exists("test_ipdmqmc_report.csv")

            loaded_matrix = np.genfromtxt("test_ipdmqmc_density_matrix.csv",
                                        delimiter=',')
            assert np.allclose(loaded_matrix.shape, self._mtd_lg.density_matrix.shape)

            loaded_report = np.genfromtxt("test_ipdmqmc_report.csv",
                                        delimiter=',', names=True)
            assert loaded_report.shape[0] == len(self._mtd_lg.report)
            assert loaded_report['beta'][0] == self._mtd_lg.report[0]['beta']
            # Numpy converts spaces to underscores
            assert loaded_report['energy_numerator'][1] == self._mtd_lg.report[1]['energy numerator']

@mark.parallel([1,2])
def test_PiecewiseIPDMQMC_switching_parallel(integral_system_small, capsys):
    mtd = PiecewiseIPDMQMC(integral_system_small, rng_seed=42)
    mtd.setup(final_beta=5, initialization="random-uniform",  n_particles=int(1e5))
    mtd.run(
        switch_beta=3.0,
        dbeta=0.001,
        cycles_per_shift=20,
        shift_dampening=0.05,
        spawn_cutoff=0.01,
        shift_by_rows=False,
        update_method="euler",
    )

    captured = capsys.readouterr()
    messages = captured.out.split("\n")

    switched = False
    for line in messages:
        if "switching to DMQMC" in line:
            switched = True
            beta = float(line.split(' ')[1].strip(';'))

    parallel_assert(switched)
    parallel_assert(np.isclose(beta, 3.001),
                    msg=f"Switched beta: {beta}\nExpected: 3.001")  # first beta that is strictly greater than 3.0)

    parallel_assert(np.isclose(mtd.density_matrix.trace(), 45022.49),
                    msg=f"Density matrix trace: {mtd.density_matrix.trace()}\nExpected: 45022.49")
    eng = (mtd.density_matrix @ mtd.system.hamiltonian).trace()
    parallel_assert(np.isclose(eng, -51202.14),
                    msg=f"Energy: {eng}\nExpected: -51202.14")
