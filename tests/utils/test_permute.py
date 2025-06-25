import numpy as np
from pytest import fixture, raises

from pydmqmc.utils import generate_ijab_symmetries_array


@fixture(scope="module")
def good_indexes() -> tuple[int]:
    i = 1
    j = 0
    a = 0
    b = 0
    
    return (i, j, a, b)


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