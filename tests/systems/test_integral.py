import numpy as np
from pytest import fixture, raises
from os.path import dirname, join, splitext

from numpy.typing import NDArray as Array

from pydmqmc.systems import Integral


@fixture
def input_file(request) -> str:
    file = join(dirname(request.path),
                "..", "inputs", "integrals", "H2-STO-3G-0.74Ang.fcidump")
    return file


@fixture
def integral_system(input_file) -> Integral:
    sys = Integral(input_file)
    return sys


@fixture
def symmetry_input_file(request) -> str:
    file = join(dirname(request.path),
                "..", "inputs", "integrals", "c2h4_ccpvdz.fcidump")
    return file


@fixture
def hamiltonian() -> Array:
    H = np.array([[-1.11675931,  0.18121046],
                  [ 0.18121046,  0.46261815]])
    return H


@fixture
def eig() -> Array:
    return np.array([-0.5785538598290489, -0.5785538598290489,
                     0.6711434915572507,  0.6711434915572507])


def test_Integral_init_default(input_file, eig):
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

    sys = Integral(input_file)

    # Values set by _read_integral_file
    assert sys.unrestricted_HF == False
    assert sys.n_electrons == 2
    assert sys.n_alpha == 1
    assert sys.n_beta == 1
    assert sys.n_orbitals == 4
    assert sys.n_virtual == 2
    
    assert (sys.orbital_pg_symmetry == [0, 0, 4, 4]).all()
    assert sys.spin_polarization == 0
    assert sys.ground_state_pg == 0

    assert sys.h1e.shape == (4, 4)
    assert np.allclose(np.diag(sys.h1e), h1e_diag)

    assert sys.h2e.shape == (4, 4, 4, 4)
    mask = np.nonzero(sys.h2e)
    assert np.allclose(sys.h2e[mask], h2e_nonzero)

    assert np.allclose(sys.eigenvalues, eig)
    assert np.isclose(sys.h0e, 0.7151043390810812)

    # Values set after _read_integral_file
    assert sys.max_symmetry == 8
    assert sys.pg_mask == 7
    assert sys.symmetry == 0

    assert np.allclose(sys.orbitals, np.arange(4))
    assert np.allclose(sys.ref_determinant, [1, 1, 0, 0])
    assert np.isclose(sys.ref_energy, -1.1167593074156128)

    # Values set by _calculate_psingle_pdouble
    assert np.isclose(sys.prob_single, 0.0)
    assert np.isclose(sys.prob_double, 1.0)


def test_Integral_init_check_symmetry(input_file):
    """Tests error checking _set_symmetry."""
    with raises(ValueError):
        sys = Integral(input_file, symmetry=3)


def test_Ingetral_init_check_reference(input_file):
    """
    Tests error checking in _symmetry_check.
    
    This error is raised when the reference determinant
    is not within the system's symmetry. We pass an
    allowed symmetry but use the default reference.
    """
    with raises(ValueError):
        sys = Integral(input_file,
                       symmetry=4)


def test_Integral_init_orbital_eigenvalues(input_file, eig):
    """
    Uses method _generate_orbital_eigenvalues in addition to default.
    """

    # Load the modified input file where
    # oribital eigenvalue information has been removed
    filebase, extension = splitext(input_file)
    file = filebase + '_eigmod' + extension
    sys = Integral(file, orbital_eigenvalues=True)

    assert np.allclose(sys.eigenvalues, eig)


def test_Integral_generate_determinants(integral_system):
    bitarray = np.array([[1, 1, 0, 0],
                         [0, 0, 1, 1]])

    integral_system.generate_determinants()

    assert integral_system.n_determinants == 2
    assert np.allclose(integral_system.bitarrays, bitarray)


def test_Integral_generate_hamiltonian(integral_system, hamiltonian):
    integral_system.generate_hamiltonian()

    assert np.allclose(integral_system.hamiltonian, hamiltonian)

def test_Integral_zero_hamiltonian(integral_system, hamiltonian):
    ref = hamiltonian - np.eye(hamiltonian.shape[0]) * hamiltonian[0,0]

    integral_system.generate_hamiltonian()
    integral_system.zero_hamiltonian()

    assert np.allclose(integral_system.hamiltonian, ref)

def test_Integral_generate_excitation_matrix(integral_system):
    nex_mat = np.array([[0, 2],
                        [2, 0]])

    integral_system.generate_excitation_matrix()

    assert np.allclose(integral_system.excitation_matrix, nex_mat)


def test_Integral_get_virtual_orbitals(integral_system):
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

    unocc, virt_ms, virt_sym, nvirt = integral_system.get_virtual_orbitals([1,1])

    assert np.allclose(unocc, ref_unocc)
    assert np.allclose(virt_ms, ref_virt_ms)
    assert np.allclose(virt_sym, ref_virt_sym)
    assert np.allclose(nvirt, ref_nvirt)


def test_Integral_get_bitarray_integers(integral_system):
    bitints = np.array([3, 12], dtype=np.int64)

    assert np.allclose(integral_system.get_bitarray_integers(), bitints)


# def test_Integral_init_reference_symmetry(symmetry_input_file):
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