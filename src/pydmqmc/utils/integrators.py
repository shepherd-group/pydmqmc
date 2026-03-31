"""
Generic ODE integrators for use in DMQMC and derivatives.

Each function in this module must have the same call signature.
"""

from typing import Callable


def euler(
    func: Callable[
        [float, ...],
        float,
    ],
    y: float,
    dt: float,
    ph_dummy: None = None,
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
    ph_dummy : None
        A dummy argument for a parallel helper, not used in this serial version.
    *args, **kwargs
        Additional arguments to pass to `func`.

    Returns
    -------
    float
        The updated value of `y` after one integration step.
    """
    dx_dt = func(y, *args, **kwargs)
    return y + dt * dx_dt


def rk4(
    func: Callable[
        [float, ...],
        float,
    ],
    y: float,
    dt: float,
    ph_dummy: None = None,
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
    ph_dummy : None
        A dummy argument for a parallel helper, not used in this serial version.
    *args, **kwargs
        Additional arguments to pass to `func`.

    Returns
    -------
    float
        The updated value of `y` after one integration step.
    """
    k1 = func(y, *args, **kwargs)
    k2 = func(y + dt / 2 * k1, *args, **kwargs)
    k3 = func(y + dt / 2 * k2, *args, **kwargs)
    k4 = func(y + dt * k3, *args, **kwargs)

    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# future note: would be good to add symplectic integrator
