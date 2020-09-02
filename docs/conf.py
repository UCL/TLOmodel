# -*- coding: utf-8 -*-
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# Helpful page:
# https://medium.com/@eikonomega/getting-started-with-sphinx-autodoc-part-1-2cebbbca5365
#

import docutils
import os
import sys

#from sphinx.ext.autodoc import AttributeDocumenter, SUPPRESS, Documenter, ModuleDocumenter
#from sphinx.util.inspect import object_description

sys.path.insert(0, os.path.abspath('../..')), os.path.abspath('../src')
#from tlo.core import Specifiable, Parameter, Types, Module   #, nullstr

#class_being_tracked = None
#subclasses_of_Module = ['hiv', 'Mockitis', 'Epilepsy']
#common_methods_to_skip = ['on_birth', 'initialise_simulation', 'apply']

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

html_static_path = ['_static']

html_context = {
    'css_files': [
        '_static/theme_overrides.css',  # override wide tables in RTD theme
        ],
     }

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False

# The terms used here are defined at:
# https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html#customizing-templates
# Each value is either True/False or a string which is a comma-separated list
# (this can be None)
# e.g. (from Sphinx documentation):
#     'members': 'var1, var2',
#     'member-order': 'bysource',
#     'special-members': '__init__',
#     'undoc-members': True,
#     'exclude-members': '__weakref__'
# NB some will only take a boolean value.
#autodoc_default_flags = ['members', 'special-members', 'show-inheritance']
# See "autodoc_default_options" at https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
# The supported options are 'members', 'member-order', 'undoc-members',
# 'private-members', 'special-members', 'inherited-members',
# 'show-inheritance', 'ignore-module-all', 'imported-members'
# and 'exclude-members'.
autodoc_default_options = {
    #'members': None,  ##'on_birth',
    #'private-members': None,
    'undoc-members': False,
    #'special-members': None,
    #'show-inheritance': False, ####True,
    #'inherited_members': 'PARAMETERS',

    # Keep HTML output order the same as in the
    # source code, rather than alphabetically:
    'member-order': 'bysource',

    # List below what you don't want to see documented:
    'exclude-members': '__dict__, name, rng, sim'  ##, read_parameters',
}

# The checker can't see private repos
linkcheck_ignore = ['^https://github.com/UCL/TLOmodel.*']


def setup(app):
    '''
    Tell Sphinx which functions to run when it emits certain events.
    '''
    #import pdb; pdb.set_trace()
    #myitems = dir(app)
    if not hasattr(app, 'mydict'):
        app.mydict = dict()

    # Not impl in Documenter base class:
    ####AttributeDocumenter.can_document_member = can_document_member

    # We want to define our own version of Documenter.can_document_member()
    # and put our functionality in there, rather than having our own versions
    # of add_content() and add_directive_header()

    ####app.connect('autodoc-skip-member', skip)

    # When the autodoc-process-docstring event is emitted, handle it with
    # add_dict_to_docstring():
    ####app.connect("autodoc-process-docstring", add_dicts_to_docstring)


