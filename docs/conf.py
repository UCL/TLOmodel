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

from sphinx.ext.autodoc import AttributeDocumenter, SUPPRESS, Documenter, ModuleDocumenter
from sphinx.util.inspect import object_description

sys.path.insert(0, os.path.abspath('../..')), os.path.abspath('../src')
from tlo.core import Specifiable, Parameter, Types

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
    #'undoc-members': None, ##False,
    #'special-members': None,
    #'show-inheritance': False, ####True,
    #'inherited_members': 'PARAMETERS',

    # Keep HTML output order the same as in the
    # source code, rather than alphabetically:
    'member-order': 'bysource',

    # List below what you don't want to see documented:
    'exclude-members': '__dict__, name, rng, sim',
}

# The checker can't see private repos
linkcheck_ignore = ['^https://github.com/UCL/TLOmodel.*']

from sphinx.util import inspect

# We keep track of how many times a particular PARAMETERS
# or PROPERTIES dictionary is encountered; we suppress
# output the first time (i.e. the raw docstring), and
# only allow it the second time (i.e. the nice table)
#dict_count = {}

@classmethod
# (cls, member: Any, membername: str, isattr: bool, parent: Any) -> bool:
# We want to stop the raw dictionaries of PARAMETERS and PROPERTIES
# being output. (We still want to show their nice tables.) At the moment
# we suppress the raw dicts using our version of
# AttributeDocumenter.add_directive_header().
def can_document_member(cls, member, membername, isattr, parent):
    """Called to see if a member can be documented by this documenter."""
    # Example values
    # self is not defined
    # cls = <class 'sphinx.ext.autodoc.AttributeDocumenter'>
    # cls.objtype = 'attribute'
    # member = dictionary for table output
    # membername = 'PARAMETERS'
    # parent for PARAMETERS = <sphinx.ext.autodoc.ClassDocumenter object at 0x112bc5b70>
    # member is a dictionary e.g. {}, or with a key like 'prob_seek_care_first_anc' and
    # a value which is a Parameter object
    # e.g. param = member['prob_seek_care_first_anc'],
    #       'prob_seek_care_first_anc' is the Item cell in the table
    #       param.__class__ = <class 'tlo.core.Parameter'>
    #        param.__dict__ = {'type_': <Types.REAL: 4>, 'description': 'Probability a woman will access antenatal care for the first time'}
    # parent.object_name = 'Module', or 'CareOfWomenDuringPregnancy'
    #type(parent) = <class 'sphinx.ext.autodoc.ClassDocumenter'>
    # isattr = False
    # parent = <sphinx.ext.autodoc.ModuleDocumenter object at 0x111fb2240>
    #raise NotImplementedError('must be implemented in subclasses')
    #return False  #True
    # will parent give us what we want?
    if parent.analyzer:
        ad = parent.analyzer.find_attr_docs()
        # ad is a dictionary with a tuple as a key and value as a list
        # e.g. (if just one entry in ad):
        # {('Specifiable', 'PANDAS_TYPE_MAP'): ['Map our Types to Python types.']}
        # we want a key like: ('Contraception', 'PARAMETERS')
        if membername in ("PARAMETERS", "PROPERTIES"):
            #import pdb; pdb.set_trace()
            classname = parent.object_name  # e.g. 'CareOfWomenDuringPregnancy'
            #if classname == 'ChronicSyndrome':
            #    import pdb; pdb.set_trace()
            key = (classname, membername)
            if key not in ad:
                ad[key] = []
            return True

            # I don't think this will work.
            # I think the second time overwrites False to True for
            # this dictionary, so it (both) gets documented
            #global dict_count
            #if key not in dict_count:
            #    dict_count[key] = 1
            #    #import pdb; pdb.set_trace()
            #else:
            #    dict_count[key] += 1
            #    #import pdb; pdb.set_trace()
            #if dict_count[key] > 1:
            #    import pdb; pdb.set_trace()
            #    return True
            #else:
            #    return False
    if inspect.isattributedescriptor(member):
        return True
    elif (not isinstance(parent, ModuleDocumenter) and
          not inspect.isroutine(member) and
          not isinstance(member, type)):
        return True
    else:
        return False


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
    #import pdb; pdb.set_trace()
    #myitems = dir(app)
    if not hasattr(app, 'mydict'):
        app.mydict = dict()

    # The next two lines show two different ways of telling Sphinx to use
    # our local, redefined versions of its internal functions:
    ##AttributeDocumenter.add_directive_header = add_directive_header

    # Not impl in Documenter base class:
    AttributeDocumenter.can_document_member = can_document_member
    #AttributeDocumenter.can_document_member = matt_can_document_member

    # We want to define our own version of Documenter.can_document_member()
    # and put our functionality in there, rather than having our own versions
    # of add_content() and add_directive_header()

    app.connect('autodoc-skip-member', skip)
    app.connect("autodoc-process-docstring", anotherfunc)

                # When the autodoc-process-docstring event is emitted, handle it with
    # add_dict_to_docstring():
    app.connect("autodoc-process-docstring", add_dicts_to_docstring)


def skip(app, what, name, obj, skip, options):

    if name in ('PARAMETERS', 'PROPERTIES'):

        #
        # Don't bother displaying PARAMETERS or
        # PROPERTIES dictionaries if they have no data.
        if not obj:
            return True
        #
        #if len(app.mydict.keys()) > 43:
            #import pdb; pdb.set_trace()
        #    return False

        mykey = ".".join(obj.keys())  # mykey is a dot-separated string of keys from obj
        #import pdb; pdb.set_trace()
        if mykey in app.mydict:  #.values():
            #import pdb; pdb.set_trace()
            app.mydict[mykey] += 1
            #return False
        else:  # New key, so a new dict; skip it.
            app.mydict[mykey] = 1
            #return True

    # From Sphinx docs:
    # "Handlers should return None to fall back
    # to the skipping behavior of autodoc and
    # other enabled extensions."
    return None



#  Emitted when autodoc has read and processed a docstring
# i.e. **too late** to affect first appearance of dict.
def anotherfunc(app, what, name, obj, options, lines):
    pass
    #import pdb; pdb.set_trace()
    #if name.split(".")[-1] in ('PARAMETERS', 'PROPERTIES'):
    #global count
    #if name == 'tlo.methods.antenatal_care.CareOfWomenDuringPregnancy.PARAMETERS':
       # import pdb; pdb.set_trace()
       # if count == 0:
            #import pdb; pdb.set_trace()
            #lines[-1] = '***Goodbye***'
            #obj['something'] = Parameter(Types.REAL, "write something here")
        #count += 1


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

