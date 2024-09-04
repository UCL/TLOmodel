# -*- coding: utf-8 -*-
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# Helpful page:
# https://medium.com/@eikonomega/getting-started-with-sphinx-autodoc-part-1-2cebbbca5365
#

import os
import sys
from importlib.metadata import version as get_version

sys.path.insert(0, os.path.abspath('../..')), os.path.abspath('../src')

sys.path.append(os.path.abspath("./_ext"))


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
    'rawfiles'
]

if os.getenv('SPELLCHECK'):
    extensions += 'sphinxcontrib.spelling',
    spelling_show_suggestions = True
    spelling_lang = 'en_GB'

source_suffix = '.rst'
master_doc = 'index'
project = 'TLOmodel'
year = '2021'
author = 'The TLOmodel Team'
copyright = '{0}, {1}'.format(year, author)
release = get_version('tlo')
version = ".".join(release.split('.')[:2])

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
html_show_copyright = False
html_sidebars = {
    '**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html'],
}
html_short_title = '%s-%s' % (project, version)

html_static_path = ['_static']
html_extra_path = ['./.nojekyll', './.gitattributes']
html_favicon = 'favicon.png'

html_css_files = ['theme_overrides.css']

rawfiles = ['.github']

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
# autodoc_default_flags = ['members', 'special-members', 'show-inheritance']
# See "autodoc_default_options" at:
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
# The supported options are 'members', 'member-order', 'undoc-members',
# 'private-members', 'special-members', 'inherited-members',
# 'show-inheritance', 'ignore-module-all', 'imported-members'
# and 'exclude-members'.
autodoc_default_options = {

    'undoc-members': False,

    # Keep HTML output order the same as in the
    # source code, rather than alphabetically:
    'member-order': 'bysource',

    # List below what you don't want to see documented:
    'exclude-members': '__dict__, name, rng, sim'  # , read_parameters',
}

# The checker can't see private repos
linkcheck_ignore = ['^https://github.com/UCL/TLOmodel.*',
                    'https://www.who.int/bulletin/volumes/88/8/09-068213/en/nn']
linkcheck_timeout = 10
linkcheck_anchors = False


# Stop displaying fully-qualified domain names of classes.
# e.g. rather than "class A.B.C", just say "class C".
add_module_names = False
