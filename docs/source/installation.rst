.. _installation:

Installation
============


.. _install-pypi:

Installing with pip
-------------------

Stable releases of pydmqmc are available via pip: ::

    $ pip install pydmqmc

.. _build-source:

Building from Source
--------------------

First, clone the `repository <https://github.com/shepherd-group/pydmqmc>`_: ::

    $ git clone https://github.com/shepherd-group/pydmqmc.git
    $ cd pydmqmc
    $ git switch release_alpha

Then, build the Python package. Since this project is in early development,
it's advisable to build with the ``--editable`` or ``-e`` flag: ::

    $ pip install -e .

.. _build-docs:

Building the Documentation
**************************

The pydmqmc documentation can be built using `Sphinx <www.sphinx-doc.org>`_.
To build the documentation locally: ::

    $ cd docs
    $ make html

The documentation can then be accessed by opening ``docs/build/html/index.html``
with a web browser.
