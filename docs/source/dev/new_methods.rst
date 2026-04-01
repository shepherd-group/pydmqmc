.. _dev-new-methods:

Writing New Methods
===================

This is a step-by-step guide for developing new classes in
the :ref:`Methods submodule<ref-methods>`. We advise you
read the :ref:`dev-philosophy` first.

.. note:: 
    The word "method" has two meanings in this guide. When written with a capitalized
    M ("Method"), we're referring to a class in the pydmqmc package that performs
    a particular calculation. When written with a lowercase m ("method"), we're
    referencing part of :ref:`objected oriented programming <oop-primer>`; that is,
    a function that belongs to a particular class. In short, Methods have methods!

.. _what-method-type:

Analytic vs Iterative: What Type is Your Method?
------------------------------------------------

The first step to adding a new Method to pydmqmc is to decide whether is should be
Analytic or Iterative, as this choice has significant consequences for the structure
of your Method and which steps you should follow. Analytic Methods are for calculations
that can be performed directly; for example, the :class:`~pydmqmc.methods.FullConfigurationInteraction`
Method finds the eigenvalues of the Hamiltonian by using NumPy's ``eigh`` function.
Iterative Methods require numerical integration, such as the :class:`~pydmqmc.methods.DensityMatrixQMC`
family.

Adding a New File
-----------------

Each method should be defined in its own Python file located in the
``src/pydmqmc/methods/`` directory.
The filename should be obviously connected to your new class. For example,
the :class:`~pydmqmc.methods.InteractionPictureDMQMC` class is in ``ipdmqmc.py``.
The file name should be written in lowercase letters and use underscores to separate words.

For this example, we'll create ``src/pydmqmc/methods/demo.py`` and open it for editing.

Coding Your Class
-----------------

We'll walk through what's needed to create your class in the sections below.
This includes any import statements, defining the class, writing its core methods,
and what other methods (public or private) you may wish to add. Specifically,
we'll be demonstrating an Iterative Method but we'll note when an Analytic Method
would be handled differently.

Import Statements
+++++++++++++++++

You must import the appropriate base class for your `Method type <what-method-type>`.
For an Iterative Method, this is :class:`pydmqmc.methods.Iterative`
and for an Analytic Method this is :class:`pydmqmc.methods.Analytic`.
In this guide, we are creating an Iterative Method. In addition to importing
the appropriate Method base class, there are a few other standard pydmqmc elements
we'll wish to reference as well:

.. code-block:: python

    from .method import Iterative
    from ..systems import System
    from ..report.registry import report_registry
    from ..utils import save_array, ParallelHelper

.. note::
    These imports (``from .method`` and ``from ..utils``) mean
    we are importing from files rather than the ``pydmqmc`` library. The former imports
    from the ``method.py`` file in the same directory as the file we are editing 
    while the latter imports the ``utils`` folder from the directory above our current one.
    Importing directly from files and directories like this instead of 
    importing from the ``pydmqmc`` package 
    (e.g., ``from pydmqmc.methods import Iterative``) helps prevent circular imports.

For better performance, you may **also wish to import `numba <dev-numba>** using

.. code-block:: python

    import numpy as np
    from numba import njit

**For good type hinting**, you may also
wish to import the following NumPy types depending on your needs:

.. code-block:: python

    from numpy.typing import NDArray as Array
    from numpy.typing import ArrayLike

You may of course import any other packages or routines that you may need to
help you write your code, including `NumPy`_ and `SciPy`_ routines. If you are only
using particular functions from these libraries, please import them individually
using ``from``.

.. _NumPy: https://numpy.org/
.. _SciPy: https://scipy.org/

Defining Your Class
+++++++++++++++++++

After you've set up your imports, its time to define your new class.
We'll go ahead and define the ``Demo`` class.
This definition will include our inheritance of the base
:class:`~pydmqmc.methods.Iterative` class, an example docstring following
the `numpydoc`_ standard, and the standard ``__init__`` initialization function.
All Method classes, whether Iterative or Analytic, must take a System object as
their first parameter. Iterative Methods must also have the ``rng_seed`` and ``parallel``
parameters.

.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html#documenting-classes

.. code-block:: python

    class Demo(Iterative):  # we inherit from the base Iterative class
    r"""
    Write a one-sentence summary of the class here.

    You can elaborate more down below. Perhaps this elaboration involves
    some math or other symbols, which is why we added the 'r' at the beginning
    of this docstring. The rest of this docstring should be formatted
    according to the numpydoc expectations. 
    
    At minimum you should include a "Parameters" section which documents all
    the parameters needed for the __init__ function.
    You will likely not need an "Attributes" section because any publicly
    accessible attributes are actually methods tagged with the `@property`
    decorator.

    Parameters
    ----------
    system : System object
        The predefined System to run the model with.
    rng_seed : int or array_like of ints, optional
        Seed or sequence of seeds for the psuedo-random number generator.
        See :func:`numpy.random.default_rng`. If using MPI parallelization,
        each processor will have a unique seed based on this value.
    parallel : bool, default False
        Whether to use MPI to parallelize the calculation.
    """
    def __init__(
        self,
        system: System,  # type hints
        rng_seed: None | int | ArrayLike = None,
        parallel: bool = False,
    ) -> None:
        # No docstring is needed
        # because we wrote the docstring under the class definition

.. important::
    Notice that the parameters for ``__init__`` are documented in the ``Parameters``
    section of the class docstring. Type hints and default values match between the code
    and the docstring.

.. note::
    Class names should be written using "Pascal case," often stylized as
    ``PascalCase``. As the stylization suggests, the first letter of every
    word should be capitalized and no spaces or other separators should be used.

.. _step-init:

Writing the Initialization Method
+++++++++++++++++++++++++++++++++

Let's dive into the structure of the ``__init__`` method. This is the function that gets
run any time we instantiate a ``Demo`` object; therefore, this method should do
everything we need to setup our system. The base :class:`~pydmqmc.methods.Iterative` class
(or :class:`~pydmqmc.methods.Analytic`) will take care of some of this for us.
The basic structure is as follows:

    1. Call ``super().__init__`` with a subset of the parameters passed to the surrounding ``__init__`` function.
    2. Prepare the associated system if needed; e.g. by generating the Hamiltonian if not already defined.
    3. Set any Method-specific data structures (such as a density matrix) to ``None`` for now.
    4. Perform parallel setup, if applicable.
    5. Set the seed for our random number generator.

.. important::
    Analytic Methods do not need steps 4 and 5.
 
.. note::
    The ``super`` function is a special Python function that resolves complex
    chains of inheritance. It's best practice to use ``super`` to access
    methods from parent classes rather than invoking them directly (e.g.
    ``Iterative.__init__``).

Step 3 is good practice so that all variables that the class will need will be known to the class at initialization, 
even if their values aren't known until other methods are called. Examples of this are the ``self._density_matrix``
and ``self._shift`` attributes below. It's also worth noting that these variables
should probably not be editable by the user since that would interfere with or overwrite the calculation;
that is why both the density matrix and the shift are made private with a leading underscore. We'll talk about
how to let users see these values *without* editing them in the next section.

We can see the five steps above in this sample code block:

.. code-block:: python

    def __init__(
        self,
        system: System,  # type hints
        rng_seed: None | int | ArrayLike = None,
        parallel: bool = False,
    ) -> None:
        # No docstring is needed
        # because we wrote the docstring under the class definition

        # Step 1:
        # Call the base class's initialization function.
        # This will set the system and parallel attributes.
        # Other attributes will be set to None.
        super().__init__(system, parallel)

        # Step 2:
        # Prepare the system, if needed.
        # This step may differ depending on what information your Method requires.
        if self.system.hamiltonian is None:
            print("Generating Hamiltonian.")
            self.system.generate_hamiltonian()

        # Step 3:
        # We'll use these data structures later in our Method.
        # For now, set them to None. Notice the type hints!
        self._density_matrix: Array | None = None
        self._shift: Array | None = None

        # Step 4:
        # Here we set up the ParallelHelper, a utility that makes it easy to
        # parallelize our Iterative Methods with MPI. The ParallelHelper must
        # know how large of a problem we're working on, which could vary
        # from Method to Method.
        if parallel:
            self._ph = ParallelHelper(
                shape=(self.system.n_determinants, self.system.n_determinants)
            )

        # Step 5:
        # Setup the RNG seed. This uses the reset_rng() method from the
        # Iterative base class. If using MPI, each process will get a unique seed.
        self.reset_rng(rng_seed)  # sets self._rng

.. _step-props:

Adding Additional Properties
++++++++++++++++++++++++++++

It's likely your new Method will introduce new attributes that you may want
a user to be able to edit, like ``self._density_matrix`` from the
:ref:`initialization <step-init>` above. Since the density matrix is involved
in our calculation, we chose to make it a private variable and therefore inaccessible
to the user; however, it would still be good for the user to be able to *see* the value
of the density matrix even if they shouldn't be able to edit it.

To let users (or other parts of pydmqmc, for that matter) access the value of class variables
without editing them, we must define **properties**. Properties are very simple methods that
just return the value of a private variable:

.. code-block:: python

    @property  # this decorator defines the behavior of this method
    def density_matrix(self) -> None | Array:
        """Density matrix."""
        return self._density_matrix

.. _step-setup:

Writing the Setup Method for Iterative Calculations
+++++++++++++++++++++++++++++++++++++++++++++++++++

.. warning::
    These instructions only apply to Iterative Methods.

The ``setup`` method accomplishes two important things:

1. Prepping the report infrastructure
2. Initializing any values needed for the calculation

For item 1, we can rely on the :class:`~pydmqmc.methods.Iterative` class through the use of ``super()``.
All we need to do is collect a list of quantities that we should track
through the calculation with the ``report_quants`` argument as shown below.

.. important::

    The iteration variable (e.g. inverse temperature :math:`\beta`
    for DMQMC) is not included in ``report_quants``. Ensuring the iteration
    variable is included in the calculation report is handled during the
    ``run`` step.

For item 2, you'll most likely want to write a separate class method that handles
some of the more intricate initialization. In the example below, this is ``_init_dm``.
Notice that the other data structure needed for our calculation, ``_shift``, is simple
to initialize and doesn't need a dedicated method.

To make your class parallel-friendly, any subroutines like ``_init_dm`` should be
wrapped by the :meth:`~pydmqmc.utils.ParallelHelper.safe_noncollective` method from the
:class:`~pydmqmc.utils.ParallelHelper` class that we constructed back during :ref:`initialization <step-init>`.

Check out the sample setup method below. Note how ``self._density_matrix`` and ``self._shift``, which
were both set to ``None`` during :ref:`initialization <step-init>`, now get values assigned. 

.. code-block:: python

    def setup(
        self,
        ...,  # Method-specific parameters
        report_quants: list[str] = ["trace", "energy expectation"],
    ) -> None:
        r"""
        Make sure to write your docstring!

        Parameters
        ----------
        ...
        report_quants : list, optional
            List of quantities to periodically report while performing
            the calculation. Each item must be recognized by the
            `report_registry`. The iteration variable will automatically
            be included.
        """
        # Step 1:
        # Setup the reporting infrastructure using the parent class
        super().setup(report_quants)

        # Step 2:
        # Initialize values used in this calculation.

        # First, we'll initialize the density matrix.
        # This in a more involved calculation, so the "guts" are encapsulated
        # by a private method, _init_dm
        if self._parallel:
            # When running in parallel, we want to only perform this calculation 
            # on one process and then share it with the others. Since the function
            # doing this initialization could throw an error, we use the
            # safe_noncollective method to ensure all processes will error together.
            self._density_matrix = self._ph.safe_noncollective(
                self._init_dm, ...  # additional arguments as needed
            )
        else:
            self._density_matrix = self._init_dm(...)

        # Initializing the shift is much easier and doesn't need it's own method
        self._shift = np.zeros(self.system.n_determinants, dtype=np.float64)

    def _init_dm(self, ...):
        # Our big complicated initialization method which has a lot of options
        return my_new_density_matrix

Writing the Run Method
++++++++++++++++++++++

The ``run`` method is where the actual calculation is performed. As such,
the structure of this function varies greatly depending on whether you are writing an
Analytic or Iterative Method.

Analytic Methods
****************

There is currently no expectation across pydmqmc for what Analytic ``run`` methods
should look like. Perform your calculation as needed, remembering to save any important
data to private attributes with a :ref:`property function <step-props>`!

Iterative Methods
*****************

For Iterative Methods, the ``run`` method should all follow the same genenral structure:

1. Run ``super().run()`` to ensure data can't be overwritten
2. Perform any sanity checks, such as checking that ``setup`` was run or inputs are appropriate
3. Do prep work based on the arguments to ``run()``
   (i.e., work that couldn't have been done at setup)
4. Report the initial state of the calculation
5. Do the calculation, including periodically reporting on its status

An example of this structure is outlined below. While the specifics may vary
depending on the kind of calculation you're performing, the example outlines
all of the critical elements you should replicate.

.. code-block:: python

    def run(
        self,
        # the following args depend on your calculation
        # but generally you'll need:
        #  - a stop criterion (final_beta)
        #  - a stepsize (dbeta)
        #  - a reporting interval (cycles_per_shift)
        final_beta: float,
        dbeta: float,
        cycles_per_shift: float,
        ...,
        # you will always need an argument for the integration method
        update_method: str = "euler",
        # this argument is not required but is recommended
        quiet: bool = False,
    ):
        r"""
        Write your docstring!

        Parameters
        ----------
        final_beta : float
            Target inverse temperature expressed as
            :math:`\beta = 1 / (k_\mathrm{B} T)`.
        dbeta : float
            Size of a single update step in inverse temperature :math:`\beta`.
        cycles_per_shift : int
            Number of updates to :math:`\beta` made before updating
            the Hamiltonian shift.
        ...
        update_method : str, default "euler"
            One of the supported update methods from
            :meth:`pydmqmc.methods.Iterative.parse_method()`
        quiet : boolean, default False
            Silence printing the iteration report as the simulation runs.
        """

        # Step 1:
        # Run super()'s run method to ensure data safety.
        # That is, if the run() method has already been called,
        # calling it a subsequent time will fail to prevent data loss.
        super().run()

        # Step 2:
        # Perform sanity checks, like making sure setup() has been run...
        if self._density_matrix is None:
            raise RuntimeError("You must first run the setup() method!")

        # ...and that any additional arguments have sane values.

        # Step 3:
        # Additional setup that could not have been done in setup().
        # This includes...

        # ...splitting up the problem in parallel, if applicable;
        # we'll see how the variables start_index and end_index get used later...
        if self._parallel:
            self._ph.allocate_reduce_buffers()
            start_index = self._ph.imin
            end_index = self._ph.imax
        else:
            start_index = 0
            end_index = self.system.n_determinants

        # ...getting the function for our update method...
        update_func = super().parse_method(update_method)

        # ...calculate out how many iterations we'll actually be executing...
        n_shifts = int(final_beta / (dbeta * cycles_per_shift))

        # ...and any other calculations that need to be done before we iterate.

        # Step 4:
        # Now we can report the initial state of our calculation.
        # This includes writing a header out to screen if `quiet` is False.
        # The `self.is_reporter` property is inherited from the Iterative class
        # and ensures that only one process is ever trying to report values.
        if self.is_reporter:
            if not quiet:
                header = f"{'beta':>14}"  # manually include the iteration var
                for value in self._report_quants:  # then print additional quants
                    header += f" {value:>14}"
                print(header)
        # We can then use the inherited `do_report` function to save information
        # on the current state of the calculation. We must specify the name and 
        # value of the iteration value so that it gets saved as the first column
        # of the report. This function will also check the `self.is_reporter`
        # property so that only one process is updating the report.
        self.do_report("beta", 0.0, quiet)

        # Step 5:
        # Time to actually do our calculation!
        for shift in range(n_shifts):
            for cycle in range(cycles_per_shift):
                self._density_matrix = update_func(
                    self._propagate,       # f(dy/dt)
                    self._density_matrix,  # y
                    dbeta,                 # stepsize dt
                    self._ph,  # parallel helper (if applicable)
                    start=start_index,  # kwargs for _propagate
                    end=end_index,
                )
            
            ...

        ...

        self.do_report("beta", (shift + 1) * cycles_per_shift * dbeta, quiet)

The example for Step 5, the actual iteration, deserves elaboration. The exact function
that ``update_func`` references depends on what kind of :ref:`integrator <api-integrators>`
the user requests (e.g. euler, RK4, etc). That said, all integrators expect the same order
of arguments:

1. The function being integrated; of the form :math:`dy/dt = f(y, ...)`
2. The current value of the variable being integrated; that is, :math:`y`
3. The step :math:`dt` being taken
4. The :class:`~pydmqmc.utils.ParallelHelper` for this method, if applicable 
   (note that ``self._ph`` is set to ``None`` when not in parallel)
5. Any additional arguments required for :math:`f(y, ...)`. To have your method run in parallel,
   you should accept (and do something appropriate with) ``start_index`` and ``end_index``

An example of an :math:`f(y, ...)` function is the ``_propagate`` function below.
Developers should also look at :ref:`dev-numba` for advice on improving the
performance of these kinds of functions.

.. code-block:: python

    def _propagate(self, rho, start, end, *args, **kwargs):
        """
        Return drho given the current state of the density matrix rho.
        """
        drho = np.zeros_like(rho)

        H = self.system.hamiltonian
        dets = H.shape[0]

        for i in range(start, end):
            for j in range(dets):
                drho[i, j] = rho[i, j] * (H[0, 0] - H[i, j])
                ...

        return drho

Writing a Save Function
+++++++++++++++++++++++

What's the point of doing all this math is we can't save our results to disk?
The final method you should absolutely add to your class is the ``save_data``
method. The :func:`pydmqmc.utils.save_array` function provides makes it easy
to save NumPy array data to a variety of formats. For Iterative Methods,
``super().save_data()`` will save the iteration report for you as well.

**The following is an example for an Iterative Method.** Analytic Methods
don't have a ``super().save_data()`` function available; instead, make
use of :func:`pydmqmc.utils.save_array`.

.. code-block:: python

    def save_data(
        self,
        basename: str,
        matrix_filetype: str = "csv",
        report_filetype: str = "csv",
        pickle_protocol: int | None = None,
    ) -> None:
        """
        Save the final density matrix and iteration report to file.

        The `basename` and `filetype` parameters will be used to construct
        filenames for all of the data written to file. For example, if
        `basename` is "test_run" and the `matrix_` and `report_filetype`
        are both "csv", the density matrix will be saved to
        "test_run_density_matrix.csv" and the iteration report will be saved to
        "test_run_report.csv".

        Parameters
        ----------
        basename : str
            Base name used to construct the filenames for the density
            matrix and iteration report
        matrix_filetype : str, default "csv"
            File type (aka extension) with which to save the density matrix.
            Supported types are:

            - "csv" : comma-separated value file
            - "npy" : NumPy binary file
            - "pkl" : Python pickle file
            - "txt" : text file (space-delimited)

        report_filetype : str, default "csv"
            File type (aka extension) with which to save the report.
            Supported types are:

            - "csv" : comma-separated value file
            - "txt" : text file (space-delimited)
            - "pkl" : pickle file

        pickle_protocol : unt, optional
            Protocol version to use if either `filetype` is "pkl".
            If none, uses `pickle`'s default.
        """
        if self.is_reporter:
            super().save_data(basename, report_filetype, pickle_protocol, "beta")
            save_array(
                self._density_matrix,
                basename + "_density_matrix",
                matrix_filetype,
                pickle_protocol,
            )

Adding Your Class to the Package
--------------------------------

Now that your class has been created, its time to add it to pydmqmc!

Open the file ``src/pydmqmc/methods/__init__.py``. This is the file
that defines what get included when you run ``import pydmqmc.methods``.

In this file, you'll need to add a line importing your new class from
the file it lives in. For example:

.. code-block:: python

    from .demo import Demo

If you installed pydmqmc in :ref:`editable mode <build-source>`,
the new ``Demo`` class will be available instantly. If not,
you'll need to reinstall pydmqmc.

Either way, you can access your new class is now accessible via the
Methods submodule:

.. code-block:: python

    from pydmqmc.methods import Demo

Linting and Formatting
----------------------

Python developers will use software called "linters" and "formatters" to help
keep their code clean. This improves readability for all developers
on a project---including your future self. Linting focuses on the quality of
your code itself while formatting fixes the visual style of the source code.

The pydmqmc library uses `Ruff`_ for both linting and formatting.
From the top level directory of the pydmqmc package, run:

.. _Ruff: https://docs.astral.sh/ruff/

.. code-block:: bash

    ruff format
    ruff check

The first command will automatically clean up any formatting inconsistencies.
The second will return a report with fixes you should make yourself, such as
removing unused import statements. Some of these may be automatically fixable
with ``ruff check --fix``. Others may need to be ignored; you can add known, ignorable
issues to the ``tool.ruff.lint.per-file-ignores`` of the ``pyproject.toml``.

Writing Tests
-------------

It's a good idea to write unit and regression tests for your new class
both to ensure everything is working as you expect and to ensure future
changes don't break your code.

Tests for the Method classes live in the ``tests/methods/`` directory.
You'll want to create a new file with the prefix ``test_`` followed by the
file name your new class lives in; for example, ``tests/methods/test_demo.py``.

This guide doesn't have the space to explain how to write good tests,
but you should make sure any public methods (including class initialization)
work with a wide range of expected user inputs. You can test for cases where
the code should fail as well as instances where it should give a correct answer
or provide a certain result.

Documenting Your Class
----------------------

Software is worth more if its documented! There are a few places you should
document your new Method class to make it more accessible to future users.
The pydmqmc documentation is written using `reStructuredText`_.
You can always view your newly created documentation by :ref:`build-docs`.

.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html

There are two places your Method should be documented. One is the :ref:`ref-methods`
page which describes the context in which each Method should be used. This page
also notes any caveats about the methods. The second is the :ref:`api-reference`,
which gives details about how classes should be invoked and what members they contain.

Extending the Method Reference
++++++++++++++++++++++++++++++

Under ``docs/source/methods``, create a new file for your new class. The name should
logically connect to your class's name; for this example, ``docs/source/methods/demo.rst``
is a good choice.

At minimum, your page should follow the structure below. You can also look at the documentation
for :ref:`methods-dmqmc` for a great example to follow!

.. code-block:: rst

    .. define a tag for referencing this Method elsewhere
    .. _methods-demo:

    Demo Method
    ===========

    .. contents::
        :local:
        :depth: 2

    Introduction
    ------------

    The Demo Method is for demonstrating the structure of a new pydmqmc Method.
    If it were based on anything real, I'd explain the underlying mathematics here
    and add references accordingly. These references should be added to the
    ``docs/source/references.bib`` file so that I can 
    cite them like this :footcite:`Blunt2014`.
    
    I would also fill out the sections on how to 
    setup, run, and save Demo Method calculations; while pydmqmc's Methods should
    all have a similar structure, the details can vary and the nuances matter!

    Setting up a Demo Simulation
    ----------------------------

    Here is where I talk about what the ``setup`` method expects for this Method
    as well as any initialization options. An example is below:

    .. code-block:: python

        import pydmqmc.systems
        from pydmqmc.methods import Demo

        # load your system file with the appropriate class
        sys = pydmqmc.systems...(...)

        # use your system to instantiate your desired Demo method
        mtd = Demo(sys)

        mtd.setup(
            ...
            report_quants = ["trace", "energy-expectation"]
        )

    After you run ``setup()`` you can check what got initialized with all those
    cool properties I added when writing this class!

    For example:

    .. code-block:: python

        print(mtd.density_matrix)

    Running a Demo Simulation
    -------------------------

    Any parameters that affect how a simulation runs or tweak how a calculation
    is performed should be explained in this section.

    A minimum viable example is below:

    .. code-block:: python

        mtd.run(
            final_beta = 1.0,
            dbeta = 0.001,
            cycles_per_shift = 100
        )

    Saving Simulation Results
    -------------------------

    What gets saved when I run the following code? What file names should I expect?

    .. code-block:: python

        mtd.save_data("my_sim")

    References
    ----------
    .. footbibliography::

Then, based on what kind of Method you've created (Analytic or Iterative), open the
corresponding ``.rst`` file in ``docs/source/methods``. For example, for an Iterative
class called ``Demo`` we would open ``docs/source/methods/iterative.rst``. In the open file, locate the
``toctree`` directive (short for "table of contents tree"). You'll add the name of your
new documentation file without modifying anything else about the block:

.. code-block:: rst

    .. toctree::
        :maxdepth: 2

        dmqmc
        demo

Extending the API Reference
+++++++++++++++++++++++++++

Within the :ref:`api-reference`, there is a page dedicated to the
:ref:`api-methods` that summarizes both the base :class:`~pydmqmc.methods.Method`
class and its child classes. The child classes are described via links
to their own individual pages.

First, create a new individual page for your class in ``docs/source/api/methods/analytic``
or ``docs/source/api/methods/iterative``, as appropriate.
Let's assume you're creating an Iterative method, as this requires the bulk of the code Examples
on this page. The documentation page would then be called ``docs/source/api/methods/iterative/demo.rst``.
The contents of this page are pretty minimal. Replace all references to "demo"
in the following example with your own class name:

.. code-block:: rst

    .. _api-demo:

    Demo
    ====

    .. autoclass:: pydmqmc.methods.Demo

The ``autoclass`` directive means that API documentation will be automatically generated
from the class's source code (including it's docstring!) when the documentation is built.

With the individual page created, open ``docs/source/api/methods/index.rst``.
You'll see three different sections for Analytic and Iterative Methods and the Base Classes.
Look at your appropriate section and locate the ``toctree`` directive, which is short for
"table of contents tree." Individual filenames can be added to this block
in order to create a table of contents on the page. Without modifying anything
else about the block, go ahead an add the name of the file you created above;
for example, for an Iterative Method you would add:

.. code-block:: rst

    Iterative Methods
    -----------------
    .. toctree::
        :maxdepth: 2

        iterative/dmqmc/dmqmc
        iterative/demo