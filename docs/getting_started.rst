=====================
Getting Started
=====================

Prequisites
===========

We use Git LFS to store large and binary files. Before you clone the repository, install
`Git LFS <https://git-lfs.github.com/>`_ and run the command :code:`git lfs install`. On Windows, simply run the
installer. On MacOS, extract the contents of the .tar.gz file using :code:`tar xvfz <filename>.tar.gz` and then
run :code:`./install.sh`. The TLOmodel repository can then be cloned as normal.

Installation
============

To get started quickly, we recommend using Anaconda Python, and installing within a fresh environment.
Please use the `Installation Guide <https://github.com/UCL/TLOmodel/wiki/Installation>`_ or, if
you prefer, you can carry out the setup using the command line:

::
    cd TLOmodel
    conda create -n tlo python=3.11
    conda activate tlo
    pip install -r requirements/dev.txt
    pip install -e .

This will install the software in 'editable' mode, so any changes you make to the source will immediately be reflected.
After the initial install, each time you wish to use the model simply activate the environment::

    conda activate tlo

To update dependencies, perform the following steps in the TLOmodel directory:

::

    conda activate tlo
    pip install -r requirements/dev.txt


Documentation
=============

To build the documentation, activate your environment as above then run::

    tox -e docs

The generated HTML documentation will appear in `dist/docs`.

Wiki
====

Please note that we have a `Wiki <https://github.com/UCL/TLOmodel/wiki>`_ which you may wish to refer to. It has information on setup, conventions, checklists and code examples.

Development
===========

To run the Python code tests only::

    pytest

To run all the tests::

    tox

Note, to combine test coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
