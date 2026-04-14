import numpy as np
from pytest import mark, fixture
from pytest_mpi import parallel_assert

from pydmqmc.utils.parallel_helper import ParallelHelper
from pydmqmc.utils.parallel_integrators import *


class TestIntegrators_Parallel():

    @fixture(autouse=True)
    def _setup(self):
        self._ph = ParallelHelper(shape=(4, 1))
        self._ph.allocate_reduce_buffers()

    def dxdy(self, y):
        """
        Each process updates a different slice of `y`.
        """
        dy = np.zeros_like(y)
        dy[self._ph.imin:self._ph.imax] = np.exp(y[self._ph.imin:self._ph.imax])
        return dy

    @mark.parallel([1,2])
    def test_euler_parallel(self):
        y = np.arange(4.0).reshape(4, 1)
        y_prime = parallel_euler(self.dxdy, y, 0.01, ph=self._ph)

        answer = y + 0.01*np.exp(y)
        parallel_assert(np.allclose(y_prime, answer), msg=f"y_prime: {y_prime}\nExpected: {answer}")

    @mark.parallel([1,2])
    def test_rk4_parallel(self):
        y = np.arange(4.0).reshape(4, 1)
        y_prime = parallel_rk4(self.dxdy, y, 0.01, ph=self._ph)

        h = 0.01
        k1 = np.exp(y)
        k2 = np.exp(y + h/2*k1)
        k3 = np.exp(y + h/2*k2)
        k4 = np.exp(y + h * k3)
        answer = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)

        parallel_assert(np.allclose(y_prime, answer), msg=f"y_prime: {y_prime}\nExpected: {answer}")
