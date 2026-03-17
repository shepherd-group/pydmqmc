import numpy as np
from pytest import mark, fixture
from pytest_mpi import parallel_assert

from pydmqmc.utils.parallel_helper import ParallelHelper
from pydmqmc.utils.parallel_integrators import *


class TestIntegrators_Parallel():

    @fixture(autouse=True)
    def _setup(self):
        self._ph = ParallelHelper(shape=(1))
        self._ph.allocate_buffers()

    @mark.parallel([1,2,3])
    def test_euler_parallel(self):
        y_prime = parallel_euler(np.exp, 1, 0.01, ph=self._ph)

        answer = 1 + 0.01*np.exp(1)
        parallel_assert(np.isclose(y_prime, answer))

    @mark.parallel([1,2,3])
    def test_rk4_parallel(self):
        y_prime = parallel_rk4(np.exp, 1, 0.01, ph=self._ph)

        y = 1
        h = 0.01
        k1 = np.exp(y)
        k2 = np.exp(y + h/2*k1)
        k3 = np.exp(y + h/2*k2)
        k4 = np.exp(y + h * k3)
        answer = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)

        parallel_assert(np.isclose(y_prime, answer))
