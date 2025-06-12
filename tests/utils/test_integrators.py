import numpy as np
from pytest import fixture

from pydmqmc.utils.integrators import *


def test_euler():
    y_prime = euler(np.exp, 1, 0.01)
    assert np.isclose(y_prime, 1 + 0.01*np.exp(1))