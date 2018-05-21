#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
from os.path import dirname, join

from setuptools import find_packages, setup


def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()


setup(
    name='tlo',
    version='0.1.0',
    description='Thanzi la Onse Epidemiology Model',
    long_description=re.compile(
        '^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
    author='Thanzi la Onse team',
    author_email='j.p.cooper@ucl.ac.uk',
    url='https://github.com/UCL/TLOmodel',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires='>=3',
)
