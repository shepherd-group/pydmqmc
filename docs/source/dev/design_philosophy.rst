.. _dev-philosophy:

Design Philosophy
=================

If you want to extend the functionality of pydmqmc, perhaps by adding support for
new :ref:`Systems <ref-systems>` or :ref:`Methods <ref-methods>`, there are a few
key design choices that you should keep in mind. These elements of pydmqmc's
design philosophy are explained below.

.. contents:: Design Principles
    :local:

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
the attributes of a base class serve as a template showing what quantities a Method
can expect to have defined.

Take a look at the base :class:`~pydmqmc.systems.System` class.

.. _static-systems:

Systems are Static
------------------

