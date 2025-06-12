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
    pass