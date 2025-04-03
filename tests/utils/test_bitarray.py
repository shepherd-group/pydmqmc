import numpy as np
from pytest import fixture

from pydmqmc.utils.bitarray import *

@fixture
def bitarrays():
    ba1 = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    ba2 = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0])
    return (ba1, ba2)

@fixture
def norbitals(bitarrays):
    return bitarrays[0].size

@fixture
def integers():
    return [63, 363]

@fixture
def label():
    return 1486911

def test_bitarray_to_integer(bitarrays, integers):
    int1 = bitarray_to_integer(bitarrays[0])
    int2 = bitarray_to_integer(bitarrays[1])
    assert int1 == integers[0]
    assert int2 == integers[1]

def test_integer_to_bitarray(integers, norbitals, bitarrays):
    ba2 = integer_to_bitarray(integers[1], norbitals)
    ba1 = integer_to_bitarray(integers[0], norbitals)
    assert np.allclose(ba1, bitarrays[0])
    assert np.allclose(ba2, bitarrays[1])

def test_concate_bitarrays_to_label(bitarrays, label):
    lab = concate_bitarrays_to_label(*bitarrays)
    assert lab == label

def test_extract_bitarrays_from_label(label, norbitals, bitarrays):
    bas = extract_bitarrays_from_label(label, norbitals)
    assert np.allclose(bas, bitarrays)
