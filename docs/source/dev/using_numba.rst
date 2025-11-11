.. _dev-numba:

Using Numba to Accelerate Methods
=================================

This is a guide to using `Numba's <https://numba.pydata.org/>`_ just in time (JIT) compiler within pydmqmc.

Most of pydmqmc's computational complexity is found within the ``run()`` method
of the various :ref:`Methods<ref-methods>`. After all, this is the function
that performs the actual model calculation, whether it is iterative or analytic.

For better performance, these calculations can be accelerated using
Numba's JIT compiler. This creates a faster version of the calculation
at the (hopefully small) cost of some initial compile time.
It is best used for functions that are used repeatedly, such as the
core of an :ref:`Iterative Method <methods-iterative>`.

Usually, Numba will compile any function to which the ``@njit``
decorator is applied:

.. code-block:: python

    from numba import njit

    @njit
    def some_function(...):
        ...

Applying it to a method in a class, however, takes a few extra steps:

    1. The ``@njit`` decorator can only be applied to `static methods`_
    2. The decorated function cannot call on class members directly (i.e. a wrapper must be used to pass class attributes)

.. _static methods: https://docs.python.org/3/library/functions.html#staticmethod

Static methods are methods that are associated with the class itself, not an instance of it.
It cannot modify the state of the class (i.e. update class attributes). Because Numba can only
be applied to static methods (item 1), this necessarily implies item 2, but we separate them out
conceptually for the sake of this guide.

Let's take the :class:`~pydmqmc.methods.SymmetricBlochDMQMC` method as an example.
This class contains a :meth:`~pydmqmc.methods.SymmetricBlochDMQMC.run` method that
has the following basic structure:

.. note::
    Technically, :class:`~pydmqmc.methods.SymmetricBlochDMQMC` inherits its
    :meth:`~pydmqmc.methods.SymmetricBlochDMQMC.run` method from
    :class:`pydmqmc.methods.DensityMatrixQMC`. If you go look in the source
    code for the structure outlined below,
    it will be under :meth:`pydmqmc.methods.DensityMatrixQMC.run`.

.. code-block:: python
    :emphasize-lines: 27
    :linenos:

    class SymmetricBlochDMQMC(DensityMatrixQMC):

    ...

    def run(
        self,
        dbeta: float,
        cycles_per_shift: int,
        update_method: str = "euler",
        ...
    )
    
    # do some initial setup
    ...

    # self._final_beta is set during the call to setup()
    n_shifts = int(self._final_beta / (dbeta * cycles_per_shift))

    # parse the function we'll be using for integration (e.g. euler, rk4, etc)
    update_func = super().parse_method(update_method)

    # update self._density_matrix every cycle
    # using the parsed method (e.g. euler, rk4, etc)
    for shift in range(n_shifts):
        for cycle in range(cycles_per_shift):
            self._density_matrix = update_func(
                self._propagate,       # function f(dy/dt) to integrate
                self._density_matrix,  # dependent var y
                dbeta,                 # stepsize dt
                ...
            )

Notice the highlighted line. Within the iteration loops on line 26,
we call whichever :ref:`integrator method <api-integrators>` was specified
to ``update_method`` (line 9). One of the arguments for this integrator
is the highlighted ``self._propagate``. This is a function that relates the
density matrix that's being updated to the iteration variable :math:`\beta`.
Since DMQMC calculations 