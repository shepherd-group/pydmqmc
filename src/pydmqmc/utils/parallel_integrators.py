"""
Generic ODE integrators for use in DMQMC and derivatives.

Each function in this module must have the same call signature.
"""

from collections.abc import Callable


def parallel_euler(func: Callable, y: float, dt: float, *args, **kwargs) -> float:
    """
    Update `y` using Euler's method.

    Function `func` is of the form::

        dy/dt = func(y, ...)
    """
    dx_dt = func(y, *args, **kwargs)
    return y + dt * dx_dt


def parallel_rk4(func: Callable, y: float, dt: float, *args, **kwargs) -> float:
    """
    Update `y` using the fourth-order Runge-Kutta method.

    Function `func` is of the form::

        dy/dt = func(y, ...)
    """
    k1 = func(y, *args, **kwargs)
    k2 = func(y + dt / 2 * k1, *args, **kwargs)
    k3 = func(y + dt / 2 * k2, *args, **kwargs)
    k4 = func(y + dt * k3, *args, **kwargs)

    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
