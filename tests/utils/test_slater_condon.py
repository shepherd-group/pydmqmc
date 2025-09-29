import numpy as np
from pytest import fixture
from os.path import dirname, join

from pydmqmc.systems import Integral
from pydmqmc.utils import get_ex_info
from pydmqmc.utils.slater_condon import *


@fixture(scope="module")
def integral_system_small(request) -> Integral:
    file = join(dirname(request.path),
                "..", "inputs", "integrals", "STRICT-STO3G-STR-H4.FCIDUMP")
    sys = Integral(file)
    return sys


class TestSlaterCondon():
    """Test the Slater-Condon rules."""

    @fixture(autouse=True)
    def _setup(self, integral_system_small):

        self._sys = integral_system_small
        self._sys.generate_determinant_bitarrays()

        # Reassign system attributes for convenience.
        self._norb = self._sys.n_orbitals
        self._nel = self._sys.n_electrons
        self._barr = [arr for arr in self._sys.bitarrays]

    def test_sc0(self):
        elem = sc0(self._barr[0], self._sys)
        assert np.isclose(elem, -1.4289643914806947)

    def test_sc1(self):
        nex, abrs, perms = get_ex_info(self._barr[3], self._barr[5], self._nel)
        a, _, r, _ = abrs
        elem = sc1(self._barr[5], a, r, perms, self._sys)
        assert np.isclose(elem, -0.03764327295193626)

    def test_sc2(self):
        nex, abrs, perms = get_ex_info(self._barr[0], self._barr[1], self._nel)
        elem = sc2(*abrs, perms, self._sys)
        assert np.isclose(elem, 0.08809542744151531)