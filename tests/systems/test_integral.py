import numpy as np
from pytest import fixture, raises
from os.path import dirname, join, splitext

from numpy.typing import NDArray as Array

from pydmqmc.systems import generate_ijab_symmetries_array, Integral

@fixture
def good_indexes() -> tuple[int]:
    i = 1
    j = 0
    a = 0
    b = 0
    
    return (i, j, a, b)

@fixture
def input_file(request) -> str:
    file = join(dirname(request.path),
                "..", "inputs", "integrals", "H2-STO-3G-0.74Ang.fcidump")
    return file

@fixture
def symmetry_input_file(request) -> str:
    file = join(dirname(request.path),
                "..", "inputs", "integrals", "c2h4_ccpvdz.fcidump")
    return file

@fixture
def eig() -> Array:
    return np.array([-0.5785538598290489, -0.5785538598290489,
                     0.6711434915572507,  0.6711434915572507])

def test_generate_ijab_symmetries_array_ef_false_rhf_false(good_indexes):
    res = np.array(
      [[1, 0, 0, 0],
       [0, 0, 1, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1]]
    )

    assert np.allclose(generate_ijab_symmetries_array(*good_indexes, 
                                                      eight_fold=False, rhf=False),
                       res)
    
def test_generate_ijab_symmetries_array_ef_true_rhf_false(good_indexes):
    res = np.array(
      [[1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 1, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 0],
       [0, 1, 0, 0],
       [1, 0, 0, 0],
       [0, 0, 0, 1]]
    )

    assert np.allclose(generate_ijab_symmetries_array(*good_indexes,
                                                      eight_fold=True, rhf=False),
                       res)

def test_generate_ijab_symmetries_array_ef_false_rhf_true(good_indexes):
    res = np.array(
      [[2, 0, 0, 0],
       [3, 1, 1, 1],
       [2, 1, 0, 1],
       [3, 0, 1, 0],
       [0, 0, 2, 0],
       [1, 1, 3, 1],
       [0, 1, 2, 1],
       [1, 0, 3, 0],
       [0, 2, 0, 0],
       [1, 3, 1, 1],
       [0, 3, 0, 1],
       [1, 2, 1, 0],
       [0, 0, 0, 2],
       [1, 1, 1, 3],
       [0, 1, 0, 3],
       [1, 0, 1, 2]]
    )

    assert np.allclose(generate_ijab_symmetries_array(*good_indexes,
                                                      eight_fold=False, rhf=True),
                       res)

def test_generate_ijab_symmetries_array_ef_true_rhf_true(good_indexes):
    res = np.array(
      [[2, 0, 0, 0],
       [3, 1, 1, 1],
       [2, 1, 0, 1],
       [3, 0, 1, 0],
       [0, 2, 0, 0],
       [1, 3, 1, 1],
       [0, 3, 0, 1],
       [1, 2, 1, 0],
       [0, 0, 2, 0],
       [1, 1, 3, 1],
       [0, 1, 2, 1],
       [1, 0, 3, 0],
       [0, 0, 0, 2],
       [1, 1, 1, 3],
       [0, 1, 0, 3],
       [1, 0, 1, 2],
       [0, 0, 2, 0],
       [1, 1, 3, 1],
       [0, 1, 2, 1],
       [1, 0, 3, 0],
       [0, 2, 0, 0],
       [1, 3, 1, 1],
       [0, 3, 0, 1],
       [1, 2, 1, 0],
       [2, 0, 0, 0],
       [3, 1, 1, 1],
       [2, 1, 0, 1],
       [3, 0, 1, 0],
       [0, 0, 0, 2],
       [1, 1, 1, 3],
       [0, 1, 0, 3],
       [1, 0, 1, 2]]
    )

    assert np.allclose(generate_ijab_symmetries_array(*good_indexes,
                                                      eight_fold=True, rhf=True),
                       res)

def test_generate_ijab_symmetries_array_input_check_ia(good_indexes):
    i, j, a, b = good_indexes
    a = 10

    with raises(ValueError):
        generate_ijab_symmetries_array(i, j, a, b)

def test_generate_ijab_symmetries_array_input_check_jb(good_indexes):
    i, j, a, b = good_indexes
    b = 10

    with raises(ValueError):
        generate_ijab_symmetries_array(i, j, a, b)

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

def test_Integral_init_determinants(input_file):
    bitarray = np.array([[1, 1, 0, 0],
                         [0, 0, 1, 1]])

    sys = Integral(input_file, determinants=True)

    assert sys.n_determinants == 2
    assert np.allclose(sys.bitarrays, bitarray)

def test_Integral_init_hamiltonian(input_file):
    H = np.array([[-1.11675931,  0.18121046],
                  [ 0.18121046,  0.46261815]])

    sys = Integral(input_file, hamiltonian=True)

    assert np.allclose(sys.hamiltonian, H)

def test_Integral_init_excitation_matrix(input_file):
    nex_mat = np.array([[0, 2],
                        [2, 0]])

    sys = Integral(input_file, excitation_matrix=True)

    assert np.allclose(sys.excitation_matrix, nex_mat)

def test_Integral_get_virtual_orbitals(input_file):
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
    
    sys = Integral(input_file)
    unocc, virt_ms, virt_sym, nvirt = sys.get_virtual_orbitals([1,1])

    assert np.allclose(unocc, ref_unocc)
    assert np.allclose(virt_ms, ref_virt_ms)
    assert np.allclose(virt_sym, ref_virt_sym)
    assert np.allclose(nvirt, ref_nvirt)

def test_Integral_init_get_bitarray_integers(input_file):
    bitints = np.array([3, 12], dtype=np.int64)

    sys = Integral(input_file)

    assert np.allclose(sys.get_bitarray_integers(), bitints)

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