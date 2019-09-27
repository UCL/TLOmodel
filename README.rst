========
Overview
========

.. start-badges

.. image:: https://api.travis-ci.com/UCL/TLOmodel.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.com/UCL/TLOmodel

.. end-badges

Thanzi la Onse Epidemiology Model
=================================

This is the main software framework for epidemiology and health system modelling in the Thanzi la Onse project.

See https://thanzi.org for more about the project.

Installation
============

To get started quickly, we recommend using Anaconda Python, and installing within a fresh environment.
Please use the `Wiki Installation Guide <https://github.com/UCL/TLOmodel/wiki/Installation>`_.


Documentation
=============

To build the documentation, activate your environment as above then run::

    tox -e docs

The generated HTML documentation will appear in `dist/docs`.

Development
===========

To use the software interactively in a Jupyter notebook, run::

    jupyter notebook notebooks &

To run just the Python code tests quickly, use::

    pytest

To run all the tests run::

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

