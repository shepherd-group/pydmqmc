"""
Generic ODE integrators for use in DMQMC and derivatives.

Each function in this module must have the same call signature.
"""

from typing import Callable


def parallel_euler(
    func: Callable[
        [float, ...],
        float,
    ],
    y: float,
    dt: float,
    ph: "ParallelHelper",
    *args,
    **kwargs,
) -> float:
    """
    Update `y` using Euler's method.

    Function `func` is of the form::

        dy/dt = func(y, ...)

    Parameters
    ----------
    func : function
        The function defining the ODE. It should take at least one float
        as it's first argument.
    y : float
        The current value of the variable being integrated.
    dt : float
        The time step for the integration.
    ph : ParallelHelper
        Instance of the ParallelHelper class for performing reductions.
    *args, **kwargs
        Additional arguments to pass to `func`.

    Returns
    -------
    float
        The updated value of `y` after one integration step.
    """
    dx_dt = func(y, *args, **kwargs)
    dx_dt = ph.allreduce_sum(dx_dt)
    return y + dt * dx_dt


def parallel_rk4(
    func: Callable[
        [float, ...],
        float,
    ],
    y: float,
    dt: float,
    ph: "ParallelHelper",
    *args,
    **kwargs,
) -> float:
    """
    Update `y` using the fourth-order Runge-Kutta method.

    Function `func` is of the form::

        dy/dt = func(y, ...)

    Parameters
    ----------
    func : function
        The function defining the ODE. It should take at least one float
        as it's first argument.
    y : float
        The current value of the variable being integrated.
    dt : float
        The time step for the integration.
    ph : ParallelHelper
        Instance of the ParallelHelper class for performing reductions.
    *args, **kwargs
        Additional arguments to pass to `func`.

    Returns
    -------
    float
        The updated value of `y` after one integration step.
    """
    k1 = func(y, *args, **kwargs)
    k1 = ph.allreduce_sum(k1)
    k2 = func(y + dt / 2 * k1, *args, **kwargs)
    k2 = ph.allreduce_sum(k2)
    k3 = func(y + dt / 2 * k2, *args, **kwargs)
    k3 = ph.allreduce_sum(k3)
    k4 = func(y + dt * k3, *args, **kwargs)
    k4 = ph.allreduce_sum(k4)

    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
