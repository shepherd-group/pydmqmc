import numpy as np

from pydmqmc.utils.integrators import *


def test_euler():
    y_prime = euler(np.exp, 1, 0.01)

    answer = 1 + 0.01*np.exp(1)
    assert np.isclose(y_prime, answer)


def test_rk4():
    y_prime = rk4(np.exp, 1, 0.01)

    y = 1
    h = 0.01
    k1 = np.exp(y)
    k2 = np.exp(y + h/2*k1)
    k3 = np.exp(y + h/2*k2)
    k4 = np.exp(y + h * k3)
    answer = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    assert np.isclose(y_prime, answer)
