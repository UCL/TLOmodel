# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]
if os.getenv('SPELLCHECK'):
    extensions += 'sphinxcontrib.spelling',
    spelling_show_suggestions = True
    spelling_lang = 'en_GB'

source_suffix = '.rst'
master_doc = 'index'
project = 'TLOmodel'
year = '2018'
author = 'Jonathan Cooper'
copyright = '{0}, {1}'.format(year, author)
version = release = '0.1.0'

pygments_style = 'trac'
templates_path = ['.']
extlinks = {
    'issue': ('https://github.com/UCL/TLOmodel/issues/%s', '#'),
    'pr': ('https://github.com/UCL/TLOmodel/pull/%s', 'PR #'),
}
# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:  # only set the theme if we're building docs locally
    html_theme = 'sphinx_rtd_theme'

html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = False
html_sidebars = {
    '**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html'],
}
html_short_title = '%s-%s' % (project, version)

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False

autodoc_default_flags = ['members', 'special-members', 'show-inheritance']

# The checker can't see private repos
linkcheck_ignore = ['^https://github.com/UCL/TLOmodel.*']


def setup(app):
    '''
    Tell Sphinx which functions to run when it emits certain events.
    '''

    # The next two lines show two different ways of telling Sphinx to use
    # our local, redefined versions of its internal functions:
    ###AttributeDocumenter.add_directive_header = add_directive_header
    ###app.extensions['sphinx.ext.autodoc'].module.Documenter.add_content =\
    ###    add_content

    # When the autodoc-process-docstring event is emitted, handle it with
    # add_params_to_docstring():
    app.connect("autodoc-process-docstring", add_params_to_docstring)


def add_params_to_docstring(app, what, name, obj, options, lines):
    '''
    This adds a disease's PARAMETERS values in the form of a table.
    We will also want to do the same for its PROPERTIES in TLO.
    Ideally using the same function.
    '''

    if (what == "attribute"):
        # Adds the following list to each attribute docstring encountered
        # Examples of 'name' are:
        # tlo.methods.chronicsyndrome.ChronicSyndrome.rng
        # tlo.methods.antenatal_care.CareOfWomenDuringPregnancy.parameters
        # tlo.methods.epilepsy.Epilepsy.PARAMETERS
        substrings = name.split(".")
        #import pdb; pdb.set_trace()
        disease = substrings[-2]   # e.g. "Epilepsy"
        attribute_name = substrings[-1]  # e.g. "PARAMETERS"
        import pdb
        if lines is None:
            lines = []
        if (attribute_name == "PARAMETERS"):  # and disease != "Module"):
            #lines += create_table(obj, planet_name)
            lines += create_table(obj, disease)
            #pdb.set_trace()
            #lines.append("create table placeholder")


def create_table(mydict, mydisease):
    '''
    Dynamically create a table of arbitrary length.
    `mydict` is the dictionary object.
    `mydisease` is the disease name, e.g. "Epilepsy".
    A key point here is that it splits a string with the
    delimiter " = ", because that is what the Parameter
    class's __repr__() function returns.
    NB Do not change the positioning of items in the
    f-strings below, or things will break!
    '''
    delimiter = " === "

    examplestr = f'''
.. list-table::  Info for {mydisease}
   :widths: 25 25 50
   :header-rows: 1

   * - Item
     - Value
     - Description
'''

    for key in mydict.keys():
        item = str(mydict[key])
        value, description = item.split(delimiter)

        row = f'''   * - {key}
     - {value}
     - {description}
'''
        examplestr += row
    mylist = examplestr.splitlines()
    return mylist

