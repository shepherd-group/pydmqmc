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

    1. The decorated function cannot call on class members directly (i.e. a wrapper must be used to pass class attributes)
    2. The ``@njit`` decorator can only be applied to `static methods`_

.. _static methods: https://docs.python.org/3/library/functions.html#staticmethod

Static methods are methods that are associated with the class itself, not an instance of it.
It cannot modify the state of the class (i.e. update class attributes). Because Numba can only
be applied to static methods (item 2), this necessarily implies item 1, but we separate them out
conceptually in this order for the sake of this guide.

Throughout this guide, we'll be using :class:`pydmqmc.methods.AsymmetricBlochDMQMC` as a case study.
We'll present code blocks that don't reflect the current state of the source code, but rather
demonstrate the logic *behind* :class:`~pydmqmc.methods.AsymmetricBlochDMQMC`'s current form.

What Makes a Good Candidate for Numba?
--------------------------------------

The :class:`~pydmqmc.methods.AsymmetricBlochDMQMC` method is an iterative one. 
It belongs to the family of :ref:`methods-dmqmc`. As such, it evolves a
density matrix :math:`\rho` from a starting inverse temperature of :math:`\beta=0` to some
finite, positive :math:`\beta`. Every step of the iteration, a number of "psi particles" or
"psips" are propagated through the matrix.

This propagation is a good candidate for acceleration with Numba. Since it is the
heart of an iterative calculation, it is run many times. It is also very slow, since
psip propagation isn't easily optimized. **All told, the time it takes to compile this
method is outweighed by the speedup provided.**

Let's outline the structure of such a propagation method:

.. code-block:: python

    class AsymmetricBlochDMQMC(Iterative):

        ...

        def _propagate(self, *args, **kwargs):
            """
            Return drho given the current state of the density matrix rho.
            """
            # This method has an attribute called "_density_matrix"
            # that was constructed elsewhere
            rho = self._density_matrix
            drho = np.zeros_like(self._density_matrix)

            H = self.system.hamiltonian
            dets = H.shape[0]

            for i in range(dets):
                for j in range(dets):
                    drho[i, j] = rho[i, j] * (H[0, 0] - H[i, j])
                    ...

            return drho

.. note::

    Remember that :class:`~pydmqmc.methods.Method` objects must have a :class:`~pydmqmc.systems.System`
    object associated with them. The :class:`~pydmqmc.systems.System` defines information that's necessary
    for the execution of the :class:`~pydmqmc.methods.Method`; in this case, the Hamiltonian.

This method could be used inside :meth:`~pydmqmc.methods.AsymmetricBlochDMQMC.run` as highlighted below:

.. code-block:: python
    :emphasize-lines: 15

    from pydmqmc.utils import euler

    class AsymmetricBlochDMQMC(Iterative):

        ...

        def run(self, dbeta: float, ...)

            # self._final_beta was set during the call to setup()
            n_cycles = int(self._final_beta / dbeta)

            # update self._density_matrix every cycle using Euler's method
            for cycle in range(n_cycles):
                    self._density_matrix = euler(
                        self._propagate,       # function f(dy/dt) to integrate
                        self._density_matrix,  # dependent var y
                        dbeta,                 # stepsize dt
                        ...
                    )

As we can see, ``_propagate`` will be run multiple times. This makes it a good candidate
for acceleration with Numba.

Step 1: Eliminating Calls to ``self``
-------------------------------------

As mentioned at the top of this guide, Numba's JIT compiler can only be applied
to static methods and static methods have no access to the members of the class
they're attached to. To that end, we must remove such references. Let's take
another look at our ``_propagate`` method and highlight the references that need to be removed:

.. code-block:: python
    :emphasize-lines: 5,11,12,14

        class AsymmetricBlochDMQMC(Iterative):

        ...

        def _propagate(self, *args, **kwargs):
            """
            Return drho given the current state of the density matrix rho.
            """
            # This method has an attribute called "_density_matrix"
            # that was constructed elsewhere
            rho = self._density_matrix
            drho = np.zeros_like(self._density_matrix)

            H = self.system.hamiltonian
            dets = H.shape[0]

            for i in range(dets):
                for j in range(dets):
                    drho[i, j] = rho[i, j] * (H[0, 0] - H[i, j])
                    ...

            return drho

The easiest way to remove these references is to create a **wrapper** function. This wrapper
will be able to access the class attributes and pass them to ``_propagate``. We can then
remove references to ``self`` from ``_propagate``:

.. code-block:: python
    :emphasize-lines: 5,11,12,14,26,27

     class AsymmetricBlochDMQMC(Iterative):

        ...

        def _propagate(density_matrix, hamiltonian, *args, **kwargs):
            """
            Return drho given the current state of the density matrix rho.
            """
            # This method has an attribute called "_density_matrix"
            # that was constructed elsewhere
            rho = density_matrix
            drho = np.zeros_like(density_matrix)

            H = hamiltonian
            dets = H.shape[0]

            for i in range(dets):
                for j in range(dets):
                    drho[i, j] = rho[i, j] * (H[0, 0] - H[i, j])
                    ...

            return drho

        def _propagate_wrapper(self):
            return self._propagate(
                self._density_matrix,
                self.system.hamiltonian,
                *args,
                **kwargs
            )

.. note::
    In the actual :class:`~pydmqmc.methods.AsymmetricBlochDMQMC` source code,
    the wrapper function is called ``_propagate`` and the accelerated function
    it wraps is called ``_propagate_core``. We chose to flip the naming convention
    for this tutorial to better match the flow of the explanation.

We can now use this wrapper function with :meth:`~pydmqmc.methods.AsymmetricBlochDMQMC.run`:

.. code-block:: python
    :emphasize-lines: 15

    from pydmqmc.utils import euler

    class AsymmetricBlochDMQMC(Iterative):

        ...

        def run(self, dbeta: float, ...)

            # self._final_beta was set during the call to setup()
            n_cycles = int(self._final_beta / dbeta)

            # update self._density_matrix every cycle using Euler's method
            for cycle in range(n_cycles):
                    self._density_matrix = euler(
                        self._propagate_wrapper,  # function f(dy/dt) to integrate
                        self._density_matrix,     # dependent var y
                        dbeta,                    # stepsize dt
                        ...
                    )

Step 2: Adding Decorators
-------------------------

The hard work is finished; now for the easy part! Our ``_propagate`` method requires
two decorators: one to declare it as a static method and the other to compile it with Numba:

.. code-block:: python
    :emphasize-lines: 7,8

    from numba import njit

    class AsymmetricBlochDMQMC(Iterative):

        ...

        @staticmethod
        @njit
        def _propagate(density_matrix, hamiltonian, *args, **kwargs):
            """
            Return drho given the current state of the density matrix rho.
            """
            # This method has an attribute called "_density_matrix"
            # that was constructed elsewhere
            rho = density_matrix
            drho = np.zeros_like(density_matrix)

            H = hamiltonian
            dets = H.shape[0]

            for i in range(dets):
                for j in range(dets):
                    drho[i, j] = rho[i, j] * (H[0, 0] - H[i, j])
                    ...

            return drho

And with that, our :class:`~pydmqmc.methods.AsymmetricBlochDMQMC` calculations will now run faster!

Takeaways
---------

While Numba's JIT compiler can be used on functions that are members of classes, those methods 
must be static methods and therefore can't actually access any of the classes other members.
It's still worth keeping these functions
inside classes for organizational purposes; for example, :class:`~pydmqmc.methods.AsymmetricBlochDMQMC`
and :class:`~pydmqmc.methods.SyymmetricBlochDMQMC` will each have slightly different propagation functions
that need to be accelerated.

The workaround is to use wrapper methods that pass the necessary class attributes to the compiled function.
This makes the code somewhat more difficult to follow, so the JIT compiler should only be applied
to functions that really need it---like those at the core of an iterative method.

Note that Numba works well with NumPy arrays but cannot handle native Python objects like lists and
dictionaries. For Python code that performs well, you should avoid using these data structures for
computationally intense operations. This is true generally, but particularly true if you want to use
Numba's JIT compiler.