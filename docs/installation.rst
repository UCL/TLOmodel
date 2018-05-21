============
Installation
============

To get started quickly, we recommend using Anaconda Python, and installing within a fresh environment.

::

    conda create -n tlo python=3.6 virtualenv
    source activate tlo
    pip install -r requirements/dev.txt
    pip install -e .

This will install the software in 'editable' mode, so any changes you make to the source will immediately be reflected.
After the initial install, each time you wish to use the model simply activate the environment::

    source activate tlo
