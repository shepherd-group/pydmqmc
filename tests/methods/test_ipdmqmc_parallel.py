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
        self._mtd_sm = InteractionPictureDMQMC(integral_system_small, parallel=True)
        self._mtd_lg = InteractionPictureDMQMC(integral_system_large, parallel=True)
        self._final_beta = 1.0
        self._nparticle = 100000

    @mark.parallel([1,2])
    def test_setup_determinitistic(self):
        ref = np.array([[0.82911633, 0.        ],
                        [0.        , 0.17088367]])

        self._mtd_sm.setup(self._final_beta,
                           "deterministic")

        local_rows = np.arange(self._mtd_sm._ph.imin, self._mtd_sm._ph.imax)
        local_diag = self._mtd_sm.density_matrix[
            np.arange(self._mtd_sm._ph.nrows), local_rows
        ]
        parallel_assert(
            np.allclose(local_diag, np.diag(ref)[local_rows]),
            msg=f"Local diagonal: {local_diag}\nExpected: {np.diag(ref)[local_rows]}",
        )

    @mark.parallel([1,2])
    def test_setup_random_uniform(self):
        ref = np.array([[41382.0252,  0.       ],
                        [    0.    ,  8559.3920]])

        self._mtd_sm.reset_rng(rng_seed=42)
        self._mtd_sm.setup(self._final_beta,
                           "random-uniform",
                           n_particles=self._nparticle)

        local_rows = np.arange(self._mtd_sm._ph.imin, self._mtd_sm._ph.imax)
        local_diag = self._mtd_sm.density_matrix[
            np.arange(self._mtd_sm._ph.nrows), local_rows
        ]
        parallel_assert(
            np.allclose(local_diag, np.diag(ref)[local_rows]),
            msg=f"Local diagonal: {local_diag}\nExpected: {np.diag(ref)[local_rows]}",
        )

    @mark.parallel([1,2])
    def test_setup_random_thermal(self):
        ref = np.array([[82945.,     0.],
                        [    0., 17055.]])

        self._mtd_sm.reset_rng(rng_seed=42)
        self._mtd_sm.setup(self._final_beta,
                           "random-thermal",
                           n_particles=self._nparticle)

        local_rows = np.arange(self._mtd_sm._ph.imin, self._mtd_sm._ph.imax)
        local_diag = self._mtd_sm.density_matrix[
            np.arange(self._mtd_sm._ph.nrows), local_rows
        ]
        parallel_assert(
            np.allclose(local_diag, np.diag(ref)[local_rows]),
            msg=f"Local diagonal: {local_diag}\nExpected: {np.diag(ref)[local_rows]}",
        )

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

        local_rows = np.arange(self._mtd_lg._ph.imin, self._mtd_lg._ph.imax)
        local_diag = self._mtd_lg.density_matrix[
            np.arange(self._mtd_lg._ph.nrows), local_rows
        ]
        parallel_assert(
            np.allclose(local_diag, ref[local_rows]),
            msg=f"Local diagonal: {local_diag}\nExpected: {ref[local_rows]}",
        )

    @mark.parallel([1,2])
    def test_setup_fixed(self):
        diag = [10, 30]

        self._mtd_sm.setup(self._final_beta,
                           "fixed", fixed_diagonal=diag)
        local_rows = np.arange(self._mtd_sm._ph.imin, self._mtd_sm._ph.imax)
        local_diag = self._mtd_sm.density_matrix[
            np.arange(self._mtd_sm._ph.nrows), local_rows
        ]
        parallel_assert(
            np.allclose(local_diag, np.asarray(diag)[local_rows]),
            msg=f"Local diagonal: {local_diag}\nExpected: {np.asarray(diag)[local_rows]}",
        )
        parallel_assert(
            self._mtd_sm.density_matrix.size
            == self._mtd_sm._ph.nrows * self._mtd_sm.system.n_determinants,
            msg=f"Density matrix size: {self._mtd_sm.density_matrix.size}\n"
            f"Expected: {self._mtd_sm._ph.nrows * self._mtd_sm.system.n_determinants}",
        )

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

        traces = {1: 52342.66, 2: 52343.33}
        energies = {1: -69388.25, 2: -69391.18}
        matrix = self._mtd_lg._report_matrix if self._mtd_lg.is_reporter else None
        trace = matrix.trace() if self._mtd_lg.is_reporter else 0.0
        parallel_assert(np.isclose(trace, traces[self._mtd_lg.parallel_size]),
                        participating=self._mtd_lg.is_reporter,
                        msg=f"Density matrix trace: {trace}\n"
                        f"Expected: {traces[self._mtd_lg.parallel_size]}")
        eng = ((matrix @ self._mtd_lg.system.hamiltonian).trace()
               if self._mtd_lg.is_reporter else 0.0)
        parallel_assert(np.isclose(eng, energies[self._mtd_lg.parallel_size]),
                        participating=self._mtd_lg.is_reporter)

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

        traces = {1: 53532.82, 2: 53535.61}
        energies = {1: -73294.37, 2: -73296.74}
        matrix = self._mtd_lg._report_matrix if self._mtd_lg.is_reporter else None
        trace = matrix.trace() if self._mtd_lg.is_reporter else 0.0
        parallel_assert(np.isclose(trace, traces[self._mtd_lg.parallel_size]),
                        participating=self._mtd_lg.is_reporter,
                        msg=f"Density matrix trace: {trace}\n"
                        f"Expected: {traces[self._mtd_lg.parallel_size]}")
        eng = ((matrix @ self._mtd_lg.system.hamiltonian).trace()
               if self._mtd_lg.is_reporter else 0.0)
        parallel_assert(np.isclose(eng, energies[self._mtd_lg.parallel_size]),
                        participating=self._mtd_lg.is_reporter)

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

        traces = {1: 0.477507, 2: 0.560680}
        energies = {1: -0.634194, 2: -0.738259}
        matrix = self._mtd_lg._report_matrix if self._mtd_lg.is_reporter else None
        trace = matrix.trace() if self._mtd_lg.is_reporter else 0.0
        parallel_assert(np.isclose(trace, traces[self._mtd_lg.parallel_size]),
                        participating=self._mtd_lg.is_reporter,
                        msg=f"Density matrix trace: {trace}\n"
                        f"Expected: {traces[self._mtd_lg.parallel_size]}")
        eng = ((matrix @ self._mtd_lg.system.hamiltonian).trace()
               if self._mtd_lg.is_reporter else 0.0)
        parallel_assert(np.isclose(eng, energies[self._mtd_lg.parallel_size]),
                        participating=self._mtd_lg.is_reporter)

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

        traces = {1: 0.442294, 2: 0.453955}
        energies = {1: -0.588148, 2: -0.592158}
        matrix = self._mtd_lg._report_matrix if self._mtd_lg.is_reporter else None
        trace = matrix.trace() if self._mtd_lg.is_reporter else 0.0
        parallel_assert(np.isclose(trace, traces[self._mtd_lg.parallel_size]),
                        participating=self._mtd_lg.is_reporter,
                        msg=f"Density matrix trace: {trace}\n"
                        f"Expected: {traces[self._mtd_lg.parallel_size]}")
        eng = ((matrix @ self._mtd_lg.system.hamiltonian).trace()
               if self._mtd_lg.is_reporter else 0.0)
        parallel_assert(np.isclose(eng, energies[self._mtd_lg.parallel_size]),
                        participating=self._mtd_lg.is_reporter)

    @mark.parallel([1,2])
    def test_save_data(self, tmp_path):
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
        self._mtd_lg.save_data(str(tmp_path/"test_ipdmqmc"),
                            matrix_filetype="csv",
                            report_filetype="csv")

        if self._mtd_lg.is_reporter:

            file_matrix = tmp_path/"test_ipdmqmc_density_matrix.csv"
            file_report = tmp_path/"test_ipdmqmc_report.csv"

            assert exists(file_matrix)
            assert exists(file_report)

            loaded_matrix = np.genfromtxt(file_matrix, delimiter=',')
            assert loaded_matrix.shape == self._mtd_lg._report_matrix.shape

            loaded_report = np.genfromtxt(file_report, delimiter=',', names=True)
            assert loaded_report.shape[0] == len(self._mtd_lg.report)
            assert loaded_report['beta'][0] == self._mtd_lg.report[0]['beta']
            # Numpy converts spaces to underscores
            assert loaded_report['energy_numerator'][1] == self._mtd_lg.report[1]['energy numerator']

@mark.parallel([1,2])
def test_PiecewiseIPDMQMC_switching_parallel(integral_system_small, capsys):
    mtd = PiecewiseIPDMQMC(integral_system_small, rng_seed=42, parallel=True)
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
    beta = 0.0
    for line in messages:
        if "switching to DMQMC" in line:
            switched = True
            beta = float(line.split(' ')[1].strip(';'))

    switched = mtd._ph.comm.bcast(switched, root=mtd._ph.root)
    beta = mtd._ph.comm.bcast(beta, root=mtd._ph.root)
    parallel_assert(switched, msg="Piecewise IP-DMQMC did not switch methods")
    parallel_assert(np.isclose(beta, 3.001),
                    msg=f"Switched beta: {beta}\nExpected: 3.001")  # first beta that is strictly greater than 3.0)

    traces = {1: 45022.49, 2: 45016.48}
    energies = {1: -51202.14, 2: -51195.28}

    matrix = mtd._report_matrix if mtd.is_reporter else None
    trace = matrix.trace() if mtd.is_reporter else 0.0
    parallel_assert(
        np.isclose(trace, traces[mtd.parallel_size]),
        participating=mtd.is_reporter,
        msg=f"Density matrix trace: {trace}\nExpected: "
        f"{traces[mtd.parallel_size]}",
    )
    eng = ((matrix @ mtd.system.hamiltonian).trace()
           if mtd.is_reporter else 0.0)
    parallel_assert(
        np.isclose(eng, energies[mtd.parallel_size]),
        participating=mtd.is_reporter,
        msg=f"Energy: {eng}\nExpected: "
        f"{energies[mtd.parallel_size]}",
    )
