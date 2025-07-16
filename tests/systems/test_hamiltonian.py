import numpy as np
from pytest import fixture, raises, warns, mark
from os.path import dirname, join

from numpy.typing import NDArray as Array

from pydmqmc.systems import MatrixHamiltonian


@fixture(scope="module")
def known_diag() -> Array:
    values = [
       -2.11534977839243    , -1.13379264192082    ,
       -1.48665751992125    , -0.644571454998723   ,
       -1.29173833412146    , -0.537126997125704   ,
       -1.13379264192082    , -0.04063719397208587 ,
       -0.559346283023184   ,  0.39433809337647    ,
       -1.48665751992125    , -0.559346283023184   ,
       -0.78502052330756    ,  0.002819642041427484,
       -0.537126997125704   ,  0.267078948771445   ,
       -0.644571454998723   ,  0.394338093376471   ,
        0.002819642041427484,  0.902258118867541   ]

    return np.array(values)


class TestMatrixHamiltonian():
    """Test MatrixHamiltonian without supplying optional arguments."""

    @fixture(autouse=True)
    def _setup(self, request):
        file = join(dirname(request.path),
                    "..", "inputs", "hamiltonians",
                    "EQUILIBRIUM-H4-STO3G.hamil")
        self._input = file
        self._sys = MatrixHamiltonian(file)

    def test_init_default(self, known_diag):
        """Tests __init__()"""

        H = self._sys.hamiltonian

        # Check the Hamiltonian.
        assert H.shape == (20,20)
        assert np.allclose(np.diag(H), known_diag)

        # Check derived quantities about the Hamiltonian.
        assert self._sys.n_determinants == 20
        assert self._sys.ref_energy == known_diag[0]

    def test_zero_hamiltonian(self, known_diag):
        """Tests zero_hamiltonian inherited from System."""

        self._sys.zero_hamiltonian()
        H = self._sys.hamiltonian

        ref = known_diag - self._sys.ref_energy

        assert np.allclose(np.diag(H), ref)

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


class TestExtendedMatrixHamiltonian():
    """Test MatrixHamiltonian with optional arguments."""

    @fixture(autouse=True)
    def _setup(self, request):
        file = join(dirname(request.path),
                    "..", "inputs", "hamiltonians",
                    "EQUILIBRIUM-H4-STO3G.hamil")
        self._input = file

    def test_load_complex(self):
        with raises(NotImplementedError):
            MatrixHamiltonian(self._input, is_complex=True)

    def test_norbitals(self):
        """Setting n_orbitals sets orbitals and spin_polarizations."""
        sys = MatrixHamiltonian(self._input,
                                n_orbitals=4)

        assert sys.n_orbitals == 4
        assert np.allclose(sys.orbitals, np.arange(4))
        assert np.allclose(sys.spin_polarizations, [1, -1, 1, -1])

    def test_bad_nelectron_nalpha_nbeta(self):
        with raises(RuntimeError):
            MatrixHamiltonian(self._input,
                              n_electrons=10,
                              n_alpha=3,
                              n_beta=2)
 
    def test_nelectron_nalpha_nbeta(self):
        sys = MatrixHamiltonian(self._input,
                                n_electrons=10,
                                n_alpha=3,
                                n_beta=7)

        assert sys.n_electrons == 10
        assert sys.n_alpha == 3
        assert sys.n_beta == 7

    def test_nelectron_nalpha(self):
        sys = MatrixHamiltonian(self._input,
                                n_electrons=10,
                                n_alpha=3)

        assert sys.n_electrons == 10
        assert sys.n_alpha == 3
        assert sys.n_beta == 7

    def test_nelectron_nbeta(self):
        sys = MatrixHamiltonian(self._input,
                                n_electrons=10,
                                n_beta=7)

        assert sys.n_electrons == 10
        assert sys.n_alpha == 3
        assert sys.n_beta == 7

    def test_nalpha_nbeta(self):
        sys = MatrixHamiltonian(self._input,
                                n_alpha=3,
                                n_beta=7)

        assert sys.n_electrons == 10
        assert sys.n_alpha == 3
        assert sys.n_beta == 7

    def test_nelectron(self):
        with warns(UserWarning):
            sys = MatrixHamiltonian(self._input,
                                    n_electrons=10)

        assert sys.n_electrons == 10

    def test_nalpha(self):
        with warns(UserWarning):
            sys = MatrixHamiltonian(self._input,
                                    n_alpha=3)

        assert sys.n_alpha == 3

    def test_nbeta(self):
        with warns(UserWarning):
            sys = MatrixHamiltonian(self._input,
                                    n_beta=7)

        assert sys.n_beta == 7

    @mark.parametrize("orb_arg", [[1,3,5], (1,3,5), np.array([1,3,5])])
    def test_orbital_pg_sym(self, orb_arg):
        """Setting orbital_pg_sym sets max_symmetry and pg_mask."""
        sys = MatrixHamiltonian(self._input,
                                orbital_pg_symmetry=orb_arg)

        assert isinstance(sys.orbital_pg_symmetry, np.ndarray)
        assert np.allclose(sys.orbital_pg_symmetry, [1,3,5])
        assert sys.max_symmetry == 8
        assert sys.pg_mask == 7

    @mark.parametrize("eig_arg", [[1,3,5], (1,3,5), np.array([1,3,5])])
    def test_eigenvalues(self, eig_arg):
        sys = MatrixHamiltonian(self._input,
                                eigenvalues=eig_arg)

        assert isinstance(sys.eigenvalues, np.ndarray)
        assert np.allclose(sys.eigenvalues, [1,3,5])
