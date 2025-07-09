from pytest import fixture, raises

import pydmqmc.utils as utils
from pydmqmc.systems import System
from pydmqmc.methods import Method, Analytic, Iterative


@fixture
def dummy_iterative(dummy_system) -> Iterative:
    mtd = Iterative(dummy_system)
    return mtd


class TestSystem():

    @fixture(autouse=True)
    def _setup(self):
        self._sys = System("/dummy/path")

    def test_init(self):
        assert self._sys.input_file == "/dummy/path"
        assert self._sys.is_complex == False
        assert self._sys.ref_energy is None
        assert self._sys.hamiltonian is None
        assert self._sys.n_determinants is None

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
        """Test that errors flag correctly."""
        with raises(NotImplementedError):
            self._mtd.run()


class TestAnalytic():

    @fixture(autouse=True)
    def _setup(self):
        self._sys = System("/dummy/path")
        self._mtd = Analytic(self._sys)

    def test_run(self):
        """Test that errors flag correctly."""

        with raises(NotImplementedError):
            self._mtd.run()


class TestIterative():

    @fixture(autouse=True)
    def _setup(self):
        self._sys = System("/dummy/path")
        self._mtd = Iterative(self._sys)

    def test_run(self):
        """Test that errors flag correctly."""
        with raises(NotImplementedError):
            self._mtd.run()

    def test_setup(self):
        """Test that errors flag correctly."""
        with raises(NotImplementedError):
            self._mtd.setup()

    def test_parse_method(self):
        f_euler = self._mtd.parse_method("euler")
        assert f_euler is utils.euler

        f_rk4 = self._mtd.parse_method("rk4")
        assert f_rk4 is utils.rk4
