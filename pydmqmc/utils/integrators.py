from numpy.typing import NDArray as Array

def euler(func, y, dt, *args, **kwargs):
    """
    Update `y` using Euler's method.

    Function `func` is of the form::

        dy/dt = func(y, ...)
    """
    dx_dt = func(y, *args, **kwargs)
    return y + dt * dx_dt

def rk4(func, y, dt, *args, **kwargs):
    """
    Update `y` using the fourth-order Runge-Kutta method.

    Function `func` is of the form::

        dy/dt = func(y, ...)
    """
    k1 = func(y, *args, **kwargs)
    k2 = func(y + dt/2 * k1, *args, **kwargs)
    k3 = func(y + dt/2 * k2, *args, **kwargs)
    k4 = func(y + dt * k3, *args, **kwargs)

    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
