import numpy as np
from pytest import fixture, raises
from os.path import dirname, join

from pydmqmc.systems import generate_ijab_symmetries_array, Integral

@fixture
def good_indexes():
    i = 1
    j = 0
    a = 0
    b = 0
    
    return (i, j, a, b)

@fixture
def input_file(request):
    file = join(dirname(request.path),
                "..", "inputs", "integrals", "H2-STO-3G-0.74Ang.fcidump")
    return file

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

def test_Integral_read_basic_load(input_file):

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
    eig = np.array([-0.5785538598290489, -0.5785538598290489,
                     0.6711434915572507,  0.6711434915572507])

    sys = Integral(input_file)

    assert sys.unrestrictd_HF == False

    assert sys.n_electrons == 2
    assert sys.n_alpha == 1
    assert sys.n_beta == 1
    assert sys.n_orbitals == 4
    assert sys.n_virtual == 2

    assert (sys.orbital_pg_symmetry == [0, 0, 4, 4]).all()
    assert sys.spin_polarization == 0
    assert sys.ground_state_pg == 0
    assert sys.max_symmetry == 8 # post _read_integral_file
    assert sys.pg_mask == 7      # post _read_integral_file
    assert sys.symmetry == 0     # post _read_integral_file

    assert sys.h1e.shape == (4, 4)
    assert np.allclose(np.diag(sys.h1e), h1e_diag)

    assert sys.h2e.shape == (4, 4, 4, 4)
    mask = np.nonzero(sys.h2e)
    assert np.allclose(sys.h2e[mask], h2e_nonzero)

    assert np.allclose(sys.eigenvalues, eig)
    assert np.allclose(sys.orbitals, np.arange(4))
    assert np.allclose(sys.ref_determinant, [1, 1, 0, 0])
    assert np.allclose(sys.ms, [1, -1, 1, -1])

    assert np.isclose(sys.h0e, 0.7151043390810812)
    assert np.isclose(sys.ref_energy, -1.1167593074156128)
    assert np.isclose(sys.prob_single, 0.0)
    assert np.isclose(sys.prob_double, 1.0)
