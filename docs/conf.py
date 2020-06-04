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

from sphinx.ext.autodoc import AttributeDocumenter, SUPPRESS
from sphinx.util.inspect import object_description

sys.path.insert(0, os.path.abspath('../..')), os.path.abspath('../src')
from tlo.core import Specifiable

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

autodoc_default_flags = ['members', 'special-members', 'show-inheritance']

# The checker can't see private repos
linkcheck_ignore = ['^https://github.com/UCL/TLOmodel.*']


def add_content(self, more_content, no_docstring=False):
    """
    Add content from docstrings, attribute documentation and user.

     We adapt the version of this function currently installed at:
     /anaconda3/envs/nicedocs/lib/python3.6/site-packages/sphinx/ext/autodoc/__init__.py
     in the  Documenter class.
    """

    # set sourcename and add content from attribute documentation
    sourcename = self.get_sourcename()

    if self.analyzer:
        attr_docs = self.analyzer.find_attr_docs()
        if self.objpath:
            key = ('.'.join(self.objpath[:-1]), self.objpath[-1])
            # Example key: ('Contraception', 'PARAMETERS')
            if key not in attr_docs and key[1] in ("PARAMETERS", "PROPERTIES"):
                attr_docs[key] = []
            if key in attr_docs:
                no_docstring = True
                docstrings = [attr_docs[key]]
                for i, line in enumerate(self.process_doc(docstrings)):
                    self.add_line(line, sourcename, i)

    # add content from docstrings
    if not no_docstring:
        docstrings = self.get_doc()
        if not docstrings:
            # append at least a dummy docstring, so that the event
            # autodoc-process-docstring is fired and can add some
            # content if desired
            docstrings.append([])
        for i, line in enumerate(self.process_doc(docstrings)):
            self.add_line(line, sourcename, i)

    # add additional content (e.g. from document), if present
    if more_content:
        for line, src in zip(more_content.data, more_content.items):
            self.add_line(line, src[0], src[1])


def add_directive_header(self, sig):
    '''
    As above, we adapt the version of this function currently installed at:
    /anaconda3/envs/nicedocs/lib/python3.6/site-packages/sphinx/ext/autodoc/__init__.py
    in class AttributeDocumenter

    We don't want to display the raw dictionaries of PARAMETERS
    or PROPERTIES; we only want to display those in our nice
    tabular form. So, here, we suppress the raw dict printing.
    '''

    super(AttributeDocumenter, self).add_directive_header(sig)
    sourcename = self.get_sourcename()

    if not self.options.annotation and \
            not (('PARAMETERS' in sourcename) or ('PROPERTIES' in sourcename)):

        if not self._datadescriptor:
            # obtain annotation for this attribute
            annotations = getattr(self.parent, '__annotations__', {})
            if annotations and self.objpath[-1] in annotations:
                objrepr = stringify_typehint(annotations.get(self.objpath[-1]))
                self.add_line('   :type: ' + objrepr, sourcename)
            else:
                key = ('.'.join(self.objpath[:-1]), self.objpath[-1])
                if self.analyzer and key in self.analyzer.annotations:
                    self.add_line('   :type: ' +
                                  self.analyzer.annotations[key],
                                  sourcename)
            try:
                objrepr = object_description(self.object)
                self.add_line('   :value: ' + objrepr, sourcename)
            except ValueError:
                pass

    elif (self.options.annotation is SUPPRESS) or \
            ('PARAMETERS' in sourcename) or ('PROPERTIES' in sourcename):
        pass
    else:
        self.add_line('   :annotation: %s' % self.options.annotation,
                      sourcename)


def setup(app):
    '''
    Tell Sphinx which functions to run when it emits certain events.
    '''

    # The next two lines show two different ways of telling Sphinx to use
    # our local, redefined versions of its internal functions:
    AttributeDocumenter.add_directive_header = add_directive_header
    app.extensions['sphinx.ext.autodoc'].module.Documenter.add_content =\
        add_content
    # The second one could have been written:
    # Documenter.add_content = add_content

    # When the autodoc-process-docstring event is emitted, handle it with
    # add_dict_to_docstring():
    app.connect("autodoc-process-docstring", add_dicts_to_docstring)


def add_dicts_to_docstring(app, what, name, obj, options, lines):
    '''
    This adds a disease's PARAMETERS or PROPERTIES dictionary
    values in the form of a table.
    '''

    if (what == "attribute"):
        # Adds the following list to each attribute docstring encountered
        # Examples of 'name' are:
        # tlo.methods.chronicsyndrome.ChronicSyndrome.rng
        # tlo.methods.antenatal_care.CareOfWomenDuringPregnancy.parameters
        # tlo.methods.epilepsy.Epilepsy.PARAMETERS
        substrings = name.split(".")
        module = substrings[-2]   # e.g. "Epilepsy"
        attribute_name = substrings[-1]  # e.g. "PARAMETERS"

        if lines is None:
            lines = []

        if attribute_name in ("PARAMETERS", "PROPERTIES"):
            lines += create_table(obj)


def create_table(mydict):
    '''
    Dynamically create a table of arbitrary length
    from PROPERTIES and PARAMETERS dictionaries.
    `mydict` is the dictionary object.

    NB Do not change the positioning of items in the
    f-strings below, or things will break!
    '''

    examplestr = f'''
.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Item
     - Type
     - Description
'''
    if len(mydict) == 0:
        row = f'''   * - NO
     - DATA
     - DEFINED
'''
        examplestr += row
    else:
        for key in mydict:
            item = mydict[key]
            description = item.description
            mytype = item.type_  # e.g. <Types.REAL: 4>
            the_type = mytype.name  # e.g. 'REAL'

            if the_type == 'CATEGORICAL':
                description += ".  Possible values are: ["
                mylist = item.categories
                for mything in mylist:
                    description += f'{mything}, '
                description += "]"

            #the_value = mytype.value  # e.g. 4
            row = f'''   * - {key}
     - {the_type}
     - {description}
'''
            examplestr += row
    mylist = examplestr.splitlines()
    return mylist

