from pytest import fixture, raises, mark

import pydmqmc.utils as utils
from pydmqmc.systems import System
from pydmqmc.methods import Method, Analytic, Iterative


class TestSystem():

    @fixture(autouse=True)
    def _setup(self):
        self._sys = System("/dummy/path")

    def test_init(self):
        assert self._sys.input_file == "/dummy/path"
        assert self._sys.is_complex == False
        assert self._sys.n_orbitals is None
        assert self._sys.n_electrons is None
        assert self._sys.n_alpha is None
        assert self._sys.n_beta is None
        assert self._sys.orbital_pg_symmetry is None
        assert self._sys.hamiltonian is None
        assert self._sys.n_determinants is None
        assert self._sys.ref_energy is None
        assert self._sys.eigenvalues is None
        assert self._sys.max_symmetry is None
        assert self._sys.pg_mask is None
        assert self._sys.orbitals is None
        assert self._sys.spin_polarizations is None
        assert self._sys.bitarrays is None
        assert self._sys.excitation_matrix is None

    def test_zero_hamiltonian(self):
        with raises(RuntimeError):
            self._sys.zero_hamiltonian()

    def test_generate_determinant_bitarrays(self):
        with raises(RuntimeError):
            self._sys.generate_determinant_bitarrays()

    def test_get_bitarray_integers(self):
        """Calls generate_determinant_bitarrays."""
        with raises(RuntimeError):
            self._sys.get_bitarray_integers()

    def test_generate_excitation_matrix(self):
        """Calls generate_determinant_bitarrays."""
        with raises(RuntimeError):
            self._sys.generate_excitation_matrix()

    def test_get_virtual_orbitals(self):
        with raises(RuntimeError):
            # Structure of occ parameter doesn't currenly matter
            # because sanity checks have not yet been implemented
            self._sys.get_virtual_orbitals([0, 1])


class TestMethod():

    @fixture(autouse=True)
    def _setup(self):
        self._sys = System("/dummy/path")
        self._mtd = Method(self._sys)

    def test_run(self):
        self._mtd.run()
        assert self._mtd.ran_calculation

    def test_run_twice(self):
        self._mtd.run()
        with raises(RuntimeError):
            self._mtd.run()


class TestAnalytic():

    @fixture(autouse=True)
    def _setup(self):
        self._sys = System("/dummy/path")
        self._mtd = Analytic(self._sys)

    def test_run(self):
        self._mtd.run()
        assert self._mtd.ran_calculation


class TestIterative():

    @fixture(autouse=True)
    def _setup(self):
        self._sys = System("/dummy/path")
        self._mtd = Iterative(self._sys)

    def test_init(self):
        assert self._mtd.system == self._sys
        assert not self._mtd.ran_calculation
        assert self._mtd.report_values is None
        assert self._mtd.report_requirements is None
        assert self._mtd.report is None
        assert not self._mtd.parallel
        assert self._mtd.parallel_size is None
        assert self._mtd.parallel_rank is None
        assert self._mtd.parallel_is_root is None

    def test_setup(self):
        report_lst = ["trace"]
        self._mtd.setup(report_lst)
        assert self._mtd.report_values == report_lst
        assert self._mtd.report_requirements["trace"] == {}
        assert not self._mtd.ran_calculation
        assert not self._mtd.parallel

    def test_setup_invalid_report_values(self):
        with raises(AttributeError):
            self._mtd.setup(["junk"])

    def test_setup_after_run(self):
        self._mtd.setup(["trace"])
        self._mtd.run()
        with raises(RuntimeError):
            self._mtd.setup(["trace"])

    def test_parse_method(self):
        f_euler = self._mtd.parse_method("euler")
        assert f_euler is utils.euler

        f_rk4 = self._mtd.parse_method("rk4")
        assert f_rk4 is utils.rk4

        with raises(RuntimeError):
            self._mtd.parse_method("junk")

class TestIterative_Parallel():

    @fixture(autouse=True)
    def _setup(self):
        self._sys = System("/dummy/path")
        self._mtd = Iterative(self._sys, parallel=True)

    @mark.parallel([1,2,3])
    def test_init(self):
        assert self._mtd.system == self._sys
        assert not self._mtd.ran_calculation
        assert self._mtd.report_values is None
        assert self._mtd.report_requirements is None
        assert self._mtd.report is None
        assert self._mtd.parallel
        assert self._mtd.parallel_size is None
        assert self._mtd.parallel_rank is None
        assert self._mtd.parallel_is_root is None

    @mark.parallel([1,2,3])
    def test_parse_method(self):
        f_euler = self._mtd.parse_method("euler")
        assert f_euler is utils.parallel_euler

        f_rk4 = self._mtd.parse_method("rk4")
        assert f_rk4 is utils.parallel_rk4

        with raises(RuntimeError):
            self._mtd.parse_method("junk")