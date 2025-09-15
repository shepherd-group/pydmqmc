.. _dev-philosophy:

Design Philosophy
=================

If you want to extend the functionality of pydmqmc, perhaps by adding support for
new :ref:`Systems <ref-systems>` or :ref:`Methods <ref-methods>`, there are a few
key design choices that you should keep in mind. These elements of pydmqmc's
design philosophy are summarized in the following section and explained in
more detail below.
Implementation discussions are kept at a high level; for concrete details on how
to implement your own classes see the :ref:`dev-new-systems` and :ref:`dev-new-methods`
documentation.

Note that pydmqmc uses the object oriented programming design paradigm. If you are
unfamiliar with object oriented programming, check out the :ref:`oop-primer`.

This page covers conceptual design choices but aspiring developers should also read the
:ref:`dev-code-style` for suggestions on writing their actual code.

Overview: Design Principles of pydmqmc
--------------------------------------

    1. :ref:`abstraction`: significant functionality should be implemented once and then reused.
    2. :ref:`interoperability`: any System should be useable with any Method (if enough information is present).
    3. :ref:`templating`: base classes show what quantities Systems and Methods should have defined.
    4. :ref:`static-systems`: systems cannot have their defining traits modified after creation.
    5. :ref:`calculate-once`: once a Method's calculation has been completed, the data can't be modified.

.. _abstraction:

Minimize Code Duplication
-------------------------

If you find yourself writing a piece of code over and over again, maybe it's time
to write a new function instead. This ethos is otherwise known as **abstraction:**
significant pieces of functionality should be implemented only once.

The reasons for this are twofold: reliability and maintainability.
If a piece of functionality is only written once, it's easier to test since
we only need to write one test to trust that it works everywhere. The more you
duplicate functionality, the more tests you have to write to make sure every
unique instance works as intended and doesn't break. This leads into maintainability:
if someone uncovers a bug or wants to implement a new feature, good abstraction
means we only have to update the source code once. We don't have to
track down every instance of the same functionality and make sure they're all updated.

In order to achieve good abstraction, pydmqmc makes heavy use of **class inheritance.**
For instance, the :class:`~pydmqmc.methods.Method` class implements a very minimal
:meth:`~pydmqmc.methods.Method.run` method. This method ensures that 
:meth:`~pydmqmc.methods.Method.run` is only called once per Method object.
Child classes are free to add more functionality (in fact, they should) but
checking whether or not :meth:`~pydmqmc.methods.Method.run` has been called before
does not need to be rewritten for every child class individually.

What does the principle of abstraction mean for you as a developer? You may need to
write your own base classes (like :class:`~pydmqmc.methods.DensityMatrixQMC`)
and inherit from them if you're adding, say, a family of methods. Alternatively,
if you find yourself duplicating functionality already present in an existing class,
consider abstracting it to a shared base class or making your new class the child of
the existing one. Note that Python classes can inherit from multiple classes!

You'll see more detailed examples of how to work with pydmqmc's class inheritance
in the :ref:`dev-new-systems` and :ref:`dev-new-methods` documentation.


.. _interoperability:

Maximize Interoperability
-------------------------

The Systems and Methods in pydmqmc should be written such that any System can be used
with any Method, provided the user has supplied enough information. Any restrictions
should be handled with errors.

Take for example the :class:`~pydmqmc.systems.MatrixHamiltonian` system and the
:class:`~pydmqmc.methods.InteractionPictureDMQMC` method. If we want to use the
``"random-grand-canonical"`` initialization method with IP-DMQMC's
:meth:`~pydmqmc.methods.InteractionPictureDMQMC.setup`, the
:class:`~pydmqmc.systems.MatrixHamiltonian` system must have some notion
of how many electrons are in the system as well as the system's eigenvalues.
Neither of these are *required* for a :class:`~pydmqmc.systems.MatrixHamiltonian` system
to be defined, so :class:`~pydmqmc.methods.InteractionPictureDMQMC` must
check for them before attempting the ``"random-grand-canonical"`` initialization.
If the required values are not found, an error is thrown to alert the user accordingly.

How do you know which values are expected to be in an object? This brings us to the next
design principle.


.. _templating:

Base Classes as Templates
-------------------------

The base classes of pydmqmc don't just allow us to :ref:`abstraction`;
the attributes of a base class serve as a template showing what quantities a 
System or Method object can expect to have defined.

Take a look at the base :class:`~pydmqmc.systems.System` class. Even though this base
class is not intended to hold actual data or be used directly in scripts 
(both characteristics of base classes), we see that it has a number of **attributes** defined.
Within the source code, all of these attributes are set to ``None`` as they should not have a value.

Instead, these attributes are defined within the :class:`~pydmqmc.systems.System` class
to show what attributes a child class *should* have values for. 
A system may not be particularly useful if it has no way of setting the 
:attr:`~pydmqmc.systems.System.hamiltonian` attribute to anything other than ``None``,
for instance. Put another way, child classes are expected to *overwrite* the ``None`` attributes
of the :class:`~pydmqmc.systems.System` class with their own values.

That doesn't mean child classes *have* to have *every* attribute of 
:class:`~pydmqmc.systems.System` set to something other than ``None``.
One example is the :class:`~pydmqmc.systems.MatrixHamiltonian`
system as explained in :ref:`interoperability`. 
Another is the :attr:`~pydmqmc.systems.Integral.hamiltonian`
attribute of the :class:`~pydmqmc.systems.Integral` system. 
It may be computationally expensive to compute the Hamiltonian from the class's integrals,
so the :meth:`~pydmqmc.systems.Integral.generate_hamiltonian` method exists instead.
The user may elect to call this method (or use a :class:`~pydmqmc.methods.Method` object
that calls it for them, such as :class:`~pydmqmc.methods.DensityMatrixQMC`)
whenever the Hamiltonian is actually necessary. Until this function is called,
the :class:`~pydmqmc.systems.Integral` system's :attr:`~pydmqmc.systems.Integral.hamiltonian`
will remain ``None``.

Making base classes like :class:`~pydmqmc.systems.System` contain the attributes
a System object can be *expected* to have (or be able to generate)
is important for the :ref:`interoperability` principle outlined above.
The :class:`~pydmqmc.methods.Method` base classes (including :class:`~pydmqmc.methods.Analytic` 
and :class:`~pydmqmc.methods.Iterative`) also follow this principle,
though they dictate the presence of fewer attributes.
More information on how to follow this principle is contained in the :ref:`dev-new-systems`
and :ref:`dev-new-methods` documentation.


.. _static-systems:

System Objects are Static
-------------------------

The pydmqmc library is written so that every unique physical system is represented
by its own :class:`~pydmqmc.systems.System` object. If you want to work with a new
or slightly modified system, you will need to create a new object.
This behavior can be relied on by Methods, which do not allow users to accidentally
erase data once a :ref:`calculation has been completed<calculate-once>`.

Say, for example, you have a :class:`~pydmqmc.systems.MatrixHamiltonian` system
with 10 electrons. You've run a calculation using this object and now you want to
perform the same calculation using basically the same system but with 20 electrons.
If you've worked with other Python libraries, you might expect to be able to
do something like:

.. code-block:: python

    from pydmqmc.systems import MatrixHamiltonian

    # Start with a 10 electron system
    sys = MatrixHamiltonian("tests/inputs/hamiltonians/EQUILIBRIUM-H6-STO3G.hamil",
                            n_electrons = 10)

    # Do something with our sys object, like include it in a Method object

    # Try to update our system to use 20 electrons
    sys.n_electrons = 20

**This will fail.** In particular, it will throw the following error:

.. code-block:: none

    AttributeError: property 'n_electrons' of 'MatrixHamiltonian' object has no setter

The appropriate way to handle this is to make a new object:

.. code-block:: python

    sys2 = MatrixHamiltonian("tests/inputs/hamiltonians/EQUILIBRIUM-H6-STO3G.hamil",
                             n_electrons = 20)

What does this mean for developers? Take a look at that error message again:
"property 'n_electrons' of 'MatrixHamiltonian' object has no setter." Objects
will often have methods known as **getters** and **setters**. These names
refer to the method's purpose: getters will retrieve the current value of an
attribute while setters will update it. **The pydmqmc library does not
define setter methods for individual attributes.** Attributes can instead
only be set at initialization or through methods like the
:class:`~pydmqmc.methods.Iterative` class's :meth:`~pydmqmc.methods.Iterative.setup`
which perform multiple tasks.

This design choice means that pydmqmc uses code to enforce a particular philosophy
about the physical systems it works with: if a system changes one of its attributes
(such as the total number of electrons) it is now a *different* system. In pydmqmc,
this new, different system must be represented by a new, different object
(``sys2`` in our example).

In more detail, pydmqmc is able to enforce a "no setters" design by making all
attributes **private** internally. Access to these private attributes is mediated
by public getters. Using the :attr:`~pydmqmc.systems.MatrixHamiltonian.n_electrons`
example again, if you look at the source code for :class:`~pydmqmc.systems.MatrixHamiltonian`
you'll see that the private attribute ``_nel`` is used throughout most of the code.
Public access is defined through a special method in the base 
:class:`~pydmqmc.systems.System` class:

.. code-block:: python

    @property
    def n_electrons(self) -> int | None:
        """Total number of electrons."""
        return self._nel

The special ``@property`` decorator means that the ``n_electrons`` method is a 
getter. To a pydmqmc user, this method functions like an attribute named ``n_electrons``
(and will even show up in the :ref:`api-reference` as attribute).
Note that these ``@property`` getter methods enforce what *private* attribute names
should be used by child classes as part of the :ref:`templating` philosophy!

Why go through this process? Why not just give System classes an attribute
called ``n_electrons`` instead of a private ``_nel`` attribute and a getter method?
For example, let's say we have a class like:

.. code-block:: python

    class MySystem():

        def __init__(self, initial_electrons):

            self.num_electrons = initial_electrons

In Python, this public ``num_electrons`` attribute **doesn't need** a getter or setter.
It can be retrieved and set via dot notation:

.. code-block:: python

    my_sys = MySystem(initial_electrons = 10)

    print("Initial electrons:", my_sys.num_electrons)

    my_sys.num_electrons = 20

    print("Current electrons:", my_sys.num_electrons)

This will produce the following output:

.. code-block:: none

    Initial electrons: 10
    Current electrons: 20

This produces a situation that pydmqmc is explicitly trying to *avoid*; 
a system that can be modified after creation.

Confused? Overwhelmed? Don't know what this means for writing your own classes?
Don't worry; explicit steps and guidelines are laid out in the
:ref:`dev-new-systems` documentation.
If you didn't understand this design guideline on the first pass, try coming
back after following along with that guide.


.. _calculate-once:

Calculations Only Run Once
--------------------------

Much like how :ref:`static-systems`, Methods cannot have their data
modified after their :meth:`~pydmqmc.methods.Method.run` methods
have been invoked. This prevents data from being erased.

This is controlled via the :attr:`~pydmqmc.methods.Method.ran_calculation`
attribute. This attribute is set via the base :class:`~pydmqmc.methods.Method`
class's :meth:`~pydmqmc.methods.Method.run` method. This is why using class
inheritance as shown in :ref:`dev-new-methods` is so important; properly
calling the base :meth:`~pydmqmc.methods.Method.run` method will ensure
that data cannot be erased after a calculation has been run.

This is also why calculation data must only be accessible via special
getter methods for private attributes, just like the
:ref:`static System attributes<static-systems>`.
For more details, see :ref:`dev-new-methods`.
