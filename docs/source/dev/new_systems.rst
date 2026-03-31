.. _dev-new-systems:

Writing New Systems
===================

This is a step-by-step guide for developing new classes in
the :ref:`Systems submodule<ref-systems>`. We advise you
read the :ref:`dev-philosophy` first.

When to Add a New System
------------------------

Each user-facing class in the :ref:`Systems submodule<ref-systems>` is
designed to handle a particular file format; therefore, **if you want
a new file format to be able to define a system**, 
you will need a new System class.

The name of this class doesn't have to reference the file format name.
Instead your new class should reference how the system is defined.
For example, while the :class:`~pydmqmc.systems.Integral` class
loads FCIDUMP files the name "Integral" is a reference to how the
system is defined by a set of integrals (rather than, say,
directly specifying a Hamiltonian matrix as with
:class:`~pydmqmc.systems.MatrixHamiltonian`).

For this tutorial, we'll call our class ``Demo`` since it's job
is to demonstrate how to write a new System class.

Adding a New File
-----------------

Each unique system should be defined in its own Python file located in the
``src/pydmqmc/systems/`` directory.
The filename should be obviously connected to your new class. For example,
the :class:`~pydmqmc.systems.MatrixHamiltonian` class is in ``hamiltonian.py``.
The file name should be written in lowercase letters and use underscores to separate words.

For this example, we'll create ``src/pydmqmc/systems/demo.py`` and open it for editing.

Coding Your Class
-----------------

We'll walk through what's needed to create your class in the sections below.
This includes any import statements, defining the class, writing its initialization
method, and what other methods (public or private) you may wish to add.

Import Statements
+++++++++++++++++

**At minimum,** you must import the base :class:`pydmqmc.systems.System`
class for inheritance:

.. code-block:: python

    from .system import System

You may **wish to import the utility module** using

.. code-block:: python

    from .. import utils

All utility functions will be available in your code as ``utils.function_name``.

.. note::
    These imports (``from .system`` and ``from ..``) mean
    we are importing from files rather than the ``pydmqmc`` library. The former imports
    from the ``system.py`` file in the same directory as the file we are editing 
    while the latter imports the ``utils`` folder from the directory above our current one.
    Importing directly from files and directories like this instead of 
    importing from the ``pydmqmc`` package 
    (e.g., ``from pydmqmc.systems import System``) helps prevent circular imports.

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
:class:`~pydmqmc.systems.System` class, an example docstring following
the `numpydoc`_ standard, and the standard ``__init__`` initialization function
Note that the ``input_file`` and ``is_complex`` parameters are required for all Systems:

.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html#documenting-classes

.. code-block:: python

    class Demo(System):  # we inherit from the base System class
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
        input_file : str
            Filename to load.
        n_orbitals : int
            The number of orbitals in our system.
        is_complex : bool, default False
            Whether or not the Hamiltonian is complex.
        eigenvalues : array_like, optional
            Set of eigenvalues for the system.
        """

        def __init__(
            self, 
            input_file: str,  # type hints
            n_orbitals: int,
            is_complex: bool = False,
            eigenvalues: ArrayLike | None = None
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

Structure of the Initialization Method
++++++++++++++++++++++++++++++++++++++

Let's dive into the structure of the ``__init__`` method. This is the function that gets
run any time we instantiate a ``Demo`` object; therefore, this method should do
everything we need to setup our system. The base :class:`~pydmqmc.systems.System` class
will take care of some of this for us. The basic structure is as follows:

    1. Call ``super().__init__`` with a subset of the parameters passed to the surrounding ``__init__`` function.
    2. Set the :ref:`system-attributes` with system-specific initialization, such as reading its particular filetype.
    3. Call ``super()._set_derived_quants()`` to set some additional attributes using ``self._orbsym`` and ``self._norb``.

We can see this in a sample code block:

.. code-block:: python

    def __init__(self, 
                 input_file: str,  # type hints
                 n_orbitals: int,
                 is_complex: bool = False,
                 eigenvalues: ArrayLike | None = None) -> None:
        # No docstring is needed
        # because we wrote the docstring under the class definition

        # First, call the base class's initialization function.
        # This will set the input_file and is_complex attributes.
        # Other attributes will be set to None.
        super().__init__(input_file=input_file,
                         is_complex=is_complex)

        # Now, we need to do initialization that's specific to our class.
        # Some can be initialized from parameters...
        self._norb = n_orbitals

        if eigenvalues is not None:  # requires us to import numpy as np
            if isinstance(eigenvalues, np.ndarray):
                self._eig = eigenvalues
            else:
                self._eig = np.array(eigenvalues)

        # ...others will need to be read in from file
        self._read_file(input_file)

        # The System class has a private function for defining some derived attributes 
        # assuming self._orbsym and self._norb have been defined already.
        # If these attributes haven't been defined, then this function will do nothing.
        # This is the VERY LAST thing that should be done in __init__
        super()._set_derived_quants()

.. note::
    The ``super`` function is a special Python function that resolves complex
    chains of inheritance. It's best practice to use ``super`` to access
    methods from parent classes rather than invoking them directly (e.g.
    ``System.__init__``).

.. _system-attributes:

Expected Private Attributes
***************************

While the :class:`~pydmqmc.systems.System` API documentation outlines how *users* may
access attributes stored within a class, each of these maps to a *private* attribute
that should be used internally by the class. This is to adhere to the design principle
that :ref:`static-systems`. The mapping between user-facing name in the API documentation
and internal private attributes is outlined alphabetically below.

These attributes should be defined through a combination of user arguments
(passed to ``__init__``) and the ``input_file`` read during initialization.
You may allow attributes to go undefined; however, this runs counter to the 
:ref:`interoperability` design guideline.

===================  =================
Public Attribute     Private Attribute
===================  =================
bitarrays            _bitarrays
eigenvalues          _eig
excitation_matrix    _nex_mat
hamiltonian          _H
input_file           _input_file
is_complex           _is_complex
max_symmetery        _maxsym
n_alpha              _na
n_beta               _nb
n_determinants       _ndets
n_electrons          _nel
n_orbitals           _norb
orbital_pg_symmetry  _orbsym
orbitals             _orbs
pg_mask              _pg_mask
ref_energy           _ref_eng
spin_polarizations   _ms
===================  =================

Writing Other Functions
+++++++++++++++++++++++

You are free to add any additional functions that its reasonable for your class
to need or support. For example, in the above section we indicated that ``__init__``
should call a private ``_read_file`` function. Such a function would be defined as follows:

.. code-block:: python

    def _read_file(self) -> None:
        """Private functions aren't required to have docstrings, but they're helpful"""
        with open(self._input_file, "r") as f:
            # process the lines of the file
            ...

You may also wish to define ``generate`` functions such as ``generate_hamiltonian``.
Such functions may be used by Method classes to accomplish tasks that are computationally
expensive to perform at initialization. The DMQMC methods, for instance, count on the
presence of a ``generate_hamiltonian`` method if the ``hamiltonian`` attribute is ``None``.

TODO point to other documentation about what generator methods may be expected.

.. note::
    Functions should be named following the "lower with under" convention
    (stylized as ``lower_with_under``). As the stylization suggests, words should
    be lowercase and underscores should be used between words. Function names that
    start with an underscore will be *private* and only available within class
    definitions (either by the class itself or to any child classes).

Adding Your Class to the Package
--------------------------------

Now that your class has been created, its time to add it to pydmqmc!

Open the file ``src/pydmqmc/systems/__init__.py``. This is the file
that defines what get included when you run ``import pydmqmc.systems``.

In this file, you'll need to add a line importing your new class from
the file it lives in. For example:

.. code-block:: python

    from .demo import Demo

If you installed pydmqmc in :ref:`editable mode<build-source>`,
the new ``Demo`` class will be available instantly. If not,
you'll need to reinstall pydmqmc.

Either way, you can access your new class is now accessible via the
Systems submodule:

.. code-block:: python

    from pydmqmc.systems import Demo

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

Tests for the system classes live in the ``tests/systems/`` directory.
You'll want to create a new file with the prefix ``test_`` followed by the
file name your new class lives in; for example, ``tests/systems/test_demo.py``.

This guide doesn't have the space to explain how to write good tests,
but you should make sure any public methods (including class initialization)
work with a wide range of expected user inputs. You can test for cases where
the code should fail as well as instances where it should give a correct answer
or provide a certain result.

Documenting Your Class
----------------------

Software is worth more if its documented! There are a few places you should
document your new system class to make it more accessible to future users.
The pydmqmc documentation is written using `reStructuredText`_.
You can always view your newly created documentation by :ref:`build-docs`.

.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html

There are two places your system should be documented. One is the :ref:`ref-systems`
page which describes the context in which each system should be used. This page
also notes any caveats about the systems. The second is the :ref:`api-reference`,
which gives details about how classes should be invoked and what members they contain.

Extending the System Reference
++++++++++++++++++++++++++++++

The :ref:`ref-systems` page details all systems that are available within
pydmqmc. This page's source code is found in ``docs/systems.rst``.
You should add your new system here following the existing style,
as shown in the example below:

.. code-block:: rst

    .. define a tag for referencing this section elsewhere
    .. _demo-systems:

    Demo Systems
    ------------

    This class should be used for defining demonstration systems.
    It can optionally accept a list of eigenvalues, which may allow it
    to work with a broader range of Methods. All such caveats should
    be explained here.

Extending the API Reference
+++++++++++++++++++++++++++

Within the :ref:`api-reference`, there is a page dedicated to the
:ref:`api-systems` that summarizes both the base :class:`~pydmqmc.systems.System`
class and its child classes. The child classes are described via links
to their own individual pages.

First, create a new individual page for your class in ``docs/source/api/systems/``.
For the example we've been using here, the page will be called ``docs/source/api/systems/demo.rst``.
The contents of this page are pretty minimal. Replace all references to "demo"
in the following example with your own class name:

.. code-block:: rst

    .. _api-demo:

    Demo
    ====

    .. autoclass:: pydmqmc.systems.Demo

The ``autoclass`` directive means that API documentation will be automatically generated
from the class's source code (including it's docstring!) when the documentation is built.

With the individual page created, open ``docs/source/api/systems/index.rst``.
Near the top of this file is the ``toctree`` directive, which is short for
"table of contents tree." Individual filenames can be added to this block
in order to create a table of contents on the page. Without modifying anything
else about the block, go ahead an add the name of the file you created above:

.. code-block:: rst

    .. toctree::
        :maxdepth: 1

        integral
        matrix-hamiltonian
        demo