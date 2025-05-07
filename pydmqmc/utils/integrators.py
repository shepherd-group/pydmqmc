from numpy.typing import NDArray as Array

def euler(func, x, y, dy, *args, **kwargs):
    """
    Update `x` using Euler's method.

    Function `func` is of the form::

        dx/dy = func(x, y, ...)
    """
    dx_dy = func(x, y, *args, **kwargs)
    return x + dy * dx_dy
