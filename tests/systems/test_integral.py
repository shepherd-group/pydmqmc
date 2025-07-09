import numpy as np
from pytest import fixture, raises
from os.path import dirname, join, splitext

from numpy.typing import NDArray as Array

from pydmqmc.systems import Integral


@fixture(scope="module")
def hamiltonian() -> Array:
    H = np.array([[-1.11675931,  0.18121046],
                  [ 0.18121046,  0.46261815]])
    return H


@fixture(scope="module")
def eig() -> Array:
    return np.array([-0.5785538598290489, -0.5785538598290489,
                     0.6711434915572507,  0.6711434915572507])

class TestIntegral():

    @fixture(autouse=True)
    def _setup_system(self, request) -> Integral:
        file = join(dirname(request.path),
                    "..", "inputs", "integrals", 
                    "H2-STO-3G-0.74Ang.fcidump")
        self._input = file
        self._sys = Integral(file)

    def test_init_default(self, eig):
        """
        Uses methods:
            _read_integral_file
            _set_reference
            _set_symmetry
            _symmetry_check
            _calculate_psingle_pdouble
            get_virtual_orbitals
        """
        h1e_diag = np.array([-1.253309786667645,
                            -1.253309786667645,
                            -0.4750688491089999,
                            -0.4750688491089999])
        h2e_nonzero = np.array([
        0.6747559268385961, 0.1812104619941235, 0.6747559268385961,
        0.1812104619941235, 0.663711401330187 , 0.1812104619941235,
        0.663711401330187 , 0.1812104619941235, 0.6747559268385961,
        0.1812104619941235, 0.6747559268385961, 0.1812104619941235,
        0.663711401330187 , 0.1812104619941235, 0.663711401330187 ,
        0.1812104619941235, 0.1812104619941235, 0.663711401330187 ,
        0.1812104619941235, 0.663711401330187 , 0.1812104619941235,
        0.6976515044276353, 0.1812104619941235, 0.6976515044276353,
        0.1812104619941235, 0.663711401330187 , 0.1812104619941235,
        0.663711401330187 , 0.1812104619941235, 0.6976515044276353,
        0.1812104619941235, 0.6976515044276353])

        # Values set by _read_integral_file
        assert self._sys.unrestricted_HF == False
        assert self._sys.n_electrons == 2
        assert self._sys.n_alpha == 1
        assert self._sys.n_beta == 1
        assert self._sys.n_orbitals == 4
        assert self._sys.n_virtual == 2
        
        assert (self._sys.orbital_pg_symmetry == [0, 0, 4, 4]).all()
        assert self._sys.spin_polarization == 0
        assert self._sys.ground_state_pg == 0

        assert self._sys.h1e.shape == (4, 4)
        assert np.allclose(np.diag(self._sys.h1e), h1e_diag)

        assert self._sys.h2e.shape == (4, 4, 4, 4)
        mask = np.nonzero(self._sys.h2e)
        assert np.allclose(self._sys.h2e[mask], h2e_nonzero)

        assert np.allclose(self._sys.eigenvalues, eig)
        assert np.isclose(self._sys.h0e, 0.7151043390810812)

        # Values set after _read_integral_file
        assert self._sys.max_symmetry == 8
        assert self._sys.pg_mask == 7
        assert self._sys.symmetry == 0

        assert np.allclose(self._sys.orbitals, np.arange(4))
        assert np.allclose(self._sys.ref_determinant, [1, 1, 0, 0])
        assert np.isclose(self._sys.ref_energy, -1.1167593074156128)

        # Values set by _calculate_psingle_pdouble
        assert np.isclose(self._sys.prob_single, 0.0)
        assert np.isclose(self._sys.prob_double, 1.0)

    def test_init_check_symmetry(self):
        """Tests error checking _set_symmetry."""
        with raises(ValueError):
            sys = Integral(self._input, symmetry=3)

    def test_init_check_reference(self):
        """
        Tests error checking in _symmetry_check.
        
        This error is raised when the reference determinant
        is not within the system's symmetry. We pass an
        allowed symmetry but use the default reference.
        """
        with raises(ValueError):
            sys = Integral(self._input,
                           symmetry=4)

    def test_init_orbital_eigenvalues(self, eig):
        """
        Uses method _generate_orbital_eigenvalues in addition to default.
        """

        # Load the modified input file where
        # oribital eigenvalue information has been removed
        filebase, extension = splitext(self._input)
        file = filebase + '_eigmod' + extension
        sys = Integral(file, orbital_eigenvalues=True)

        assert np.allclose(self._sys.eigenvalues, eig)

    def test_generate_determinant_bitarrays(self):
        bitarray = np.array([[1, 1, 0, 0],
                            [0, 0, 1, 1]])

        self._sys.generate_determinant_bitarrays()

        assert self._sys.n_determinants == 2
        assert np.allclose(self._sys.bitarrays, bitarray)

    def test_generate_hamiltonian(self, hamiltonian):
        self._sys.generate_hamiltonian()

        assert np.allclose(self._sys.hamiltonian, hamiltonian)

    def test_zero_hamiltonian(self, hamiltonian):
        ref = hamiltonian - np.eye(hamiltonian.shape[0]) * hamiltonian[0,0]

        self._sys.generate_hamiltonian()
        self._sys.zero_hamiltonian()

        assert np.allclose(self._sys.hamiltonian, ref)

    def test_generate_excitation_matrix(self):
        nex_mat = np.array([[0, 2],
                            [2, 0]])

        self._sys.generate_excitation_matrix()

        assert np.allclose(self._sys.excitation_matrix, nex_mat)

    def test_get_virtual_orbitals(self):
        """
        This function is run as part of Integral.__init__()
        but we'll also test it individually.
        """
        ref_unocc = np.array([0, 2, 3])
        ref_virt_ms = np.array([1, 1, -1])
        ref_virt_sym = np.array([0, 4, 4])
        ref_nvirt = np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 1., 0., 0., 0.],
                            [0., 0., 0., 0., 1., 0., 0., 0.]])

        unocc, virt_ms, virt_sym, nvirt = self._sys.get_virtual_orbitals([1,1])

        assert np.allclose(unocc, ref_unocc)
        assert np.allclose(virt_ms, ref_virt_ms)
        assert np.allclose(virt_sym, ref_virt_sym)
        assert np.allclose(nvirt, ref_nvirt)

    def test_get_bitarray_integers(self):
        bitints = np.array([3, 12], dtype=np.int64)

        assert np.allclose(self._sys.get_bitarray_integers(), bitints)

# def test_init_reference_symmetry(symmetry_input_file):
#     """
#     Test overriding both symmetry and reference.

#     Both must be set at the same time to avoid an error
#     in _symmetry_check().

#     Tests methods:
#         _set_reference
#         _set_symmetry
#         _symmetry_check
#     """
#     # sys = Integral(symmetry_input_file,
#     #                symmetry=6,
#     #                reference=[1, 36])

#     # failing :c
#     # assert np.isclose(sys.ref_energy, -77.679962263180)
#     pass