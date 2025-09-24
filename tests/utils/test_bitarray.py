import numpy as np
from pytest import fixture, raises

from pydmqmc.utils.bitarray import *


class TestBitarrayUtils():
    """Test the utility functions for bitarrays."""

    @fixture(autouse=True)
    def _setup(self):
        ba1 = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        ba2 = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0])
        self._bitarrays = (ba1, ba2)
        self._norbitals = ba1.size
        self._integers = [63, 363]
        self._label = 1486911

    def test_bitarray_to_integer(self):
        int1 = bitarray_to_integer(self._bitarrays[0])
        int2 = bitarray_to_integer(self._bitarrays[1])
        assert int1 == self._integers[0]
        assert int2 == self._integers[1]

    def test_integer_to_bitarray(self):
        ba2 = integer_to_bitarray(self._integers[1], self._norbitals)
        ba1 = integer_to_bitarray(self._integers[0], self._norbitals)
        assert np.allclose(ba1, self._bitarrays[0])
        assert np.allclose(ba2, self._bitarrays[1])

    def test_concate_bitarrays_to_label(self):
        lab = concate_bitarrays_to_label(*self._bitarrays)
        assert lab == self._label

    def test_extract_bitarrays_from_label(self):
        bas = extract_bitarrays_from_label(self._label, self._norbitals)
        assert np.allclose(bas, self._bitarrays)

    def test_get_nex(self):
        nex = get_nex(*self._bitarrays)
        assert nex == 2

    def test_get_occ(self):
        occ1 = get_occ(self._bitarrays[0])
        occ2 = get_occ(self._bitarrays[1])
        assert np.allclose(occ1, [0, 1, 2, 3, 4, 5])
        assert np.allclose(occ2, [0, 1, 3, 5, 6, 8])

    def test_get_iocc(self):
        occ1 = get_occ(self._bitarrays[0])
        occ2 = get_occ(self._bitarrays[1])
        i1 = get_iocc(occ1, 3)
        i2 = get_iocc(occ2, 3)
        assert i1 == 3
        assert i2 == 2

    def test_get_iocc_invalid(self):
        occ = get_occ(self._bitarrays[0])
        with raises(ValueError):
            get_iocc(occ, 6)

    def test_get_single_perm(self):
        b2, perms = get_single_perm(self._bitarrays[0], 5, 6, 8)
        assert np.allclose(b2, [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
        assert perms == 6

    def test_get_single_perm_invalid(self):
        with raises(ValueError):
            get_single_perm(self._bitarrays[0], 6, 5, 8)

    def test_get_double_perm(self):
        b2, perms = get_double_perm(self._bitarrays[0], 4, 5, 6, 8, 10)
        assert np.allclose(b2, [1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0])
        assert perms == 20

    def test_get_double_perm_invalid(self):
        with raises(ValueError):
            get_double_perm(self._bitarrays[0], 4, 6, 5, 8, 10)

    def test_get_ex_info_single(self):
        ba3 = np.array([1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
        nex, ex_orbs, perms = get_ex_info(self._bitarrays[0], ba3, 6)
        assert nex == 1
        assert ex_orbs == [5, None, 6, None]
        assert perms == 2

    def test_get_ex_info_double(self):
        nex, ex_orbs, perms = get_ex_info(self._bitarrays[0], self._bitarrays[1], 6)
        assert nex == 2
        assert ex_orbs == [2, 4, 6, 8]
        assert perms == 7

    def test_get_ex_info_invalid_nex(self):
        ba3 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        with raises(ValueError):
            get_ex_info(self._bitarrays[0], ba3, 6)
