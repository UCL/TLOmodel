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

from sphinx.ext.autodoc import AttributeDocumenter, SUPPRESS, Documenter, ModuleDocumenter
from sphinx.util.inspect import object_description

sys.path.insert(0, os.path.abspath('../..')), os.path.abspath('../src')
from tlo.core import Specifiable, Parameter, Types, Module   #, nullstr

class_being_tracked = None
#class_object_being_tracked = None
subclasses_of_Module = ['hiv', 'Mockitis', 'Epilepsy']
common_methods_to_skip = ['on_birth', 'initialise_simulation', 'apply']

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


def before_process_signature(app, obj, bound_method):
    '''
    Emitted before autodoc formats a signature for an object.
    The event handler can modify an object to change its signature.

    app = sphinx.application.Sphinx object
    obj = the object itself (e.g.) <function Module.initialise_simulation>
         -> obj.__name__ = initialise_simulation,
         obj.__class__ = <class 'function'>
    bound_method: a boolean indicates if an object is bound method or not
    '''
    #if not bound_method:
    #if 'PARAMETERS' in obj.__name__:
    #    import pdb; pdb.set_trace()
    pass

#from sphinx.ext.autodoc import cut_lines

# Example:
# name: 'tlo.methods.antenatal_care.CareOfWomenDuringPregnancy.PARAMETERS'
# obj = PARAMETERS dictionary
# options = {'members': <object object at 0x105cc86b0>, 'undoc-members': True,
#     'show-inheritance': True,
#     'exclude-members': {'sim', 'name', '__dict__', 'rng'},
#     'member-order': 'bysource'}
# signature : None

def process_signature(app, what, name, obj, options, signature, return_annotation):
    if what == "attribute" and name == "tlo.methods.antenatal_care.CareOfWomenDuringPregnancy.PARAMETERS":

        #"PARAMETERS" in name:
        #obj = {}
        #import pdb; pdb.set_trace()    # 'tlo.core.Module.parameters'
        if 'prob_seek_care_first_anc' in obj:
            del obj['prob_seek_care_first_anc']  # Removes from table as well :-(
            options['undoc-members'] = False
    pass

def source_read_handler(app, docname, source):
    # docname is something like 'index' or 'contributing' or 'reference/tlo.methods'
    # source is a list with one item, the string rep of the page
    #print('do something here...')
    #if docname == 'reference/tlo.methods':
    #    import pdb; pdb.set_trace()

    #or maybe doctree-read() ???

    pass

# or do we want doctree-resolved(app, doctree, docname) ?
# https://stackoverflow.com/questions/39171989/docutils-traverse-sections
# example docnames: 'authors', 'reference/index', 'reference/modules'
# 'reference/tlo.methods', 'temp/tlo.methods'...
# (Pdb) doctree
# <document: <section "tlo.methods package"...>>
# type(doctree): <class 'docutils.nodes.document'>
# sections = [section for section in doctree.traverse(docutils.nodes.section)]
# mysection = sections[3]
# (Pdb) mysection.__class__
# <class 'docutils.nodes.section'>
# (Pdb) mysection.__module__
# 'docutils.nodes'
# (Pdb) mysection.attlist()
# [('ids', ['module-tlo.methods.chronicsyndrome', 'tlo-methods-chronicsyndrome-module']),
#     ('names', ['tlo.methods.chronicsyndrome module'])]
# (Pdb) mysection.shortrepr()
# '<section "tlo.methods.chronicsyndrome module"...>'
# (Pdb) "tlo.methods." in mysection.shortrepr()
# True
# (Pdb) myx = mysection.__getitem__('names')
# (Pdb) myx
# ['tlo.methods.chronicsyndrome module']
# (Pdb) mysection.child_text_separator
# '\n\n'

# items = [item for item in mysection._all_traverse()]
# len(items) = 782, some of which are:
# <desc_name: <#text: 'PROPERTIES'>>, <#text: 'PROPERTIES'>,
# <desc_annotation: <#text: " = {'cs_date_a ...">>,
# <#text: " = {'cs_date_acquired': <tlo.core.Property object>,
# 'cs_date_cur ...">, <desc_content: <table...>>,
# <table: <tgroup...>>,
# <tgroup: <colspec...><colspec...><colspec...><thead...><tbody...>>,
# <colspec: >, <colspec: >, <colspec: >, <thead: <row...>>,
# <row: <entry...><entry...><entry...>>,
# <entry: <paragraph...>>, <paragraph: <#text: 'Item'>>,
# <#text: 'Item'>, <entry: <paragraph...>>,
# <paragraph: <#text: 'Type'>>, <#text: 'Type'>,
# <entry: <paragraph...>>, <paragraph: <#text: 'Description'>>,
# <#text: 'Description'>, <tbody: <row...><row...><row...><row...><row...>>,
# <row: <entry...><entry...><entry...>>, <entry: <paragraph...>>,



def doctree_resolved_handler(app, doctree, docname):
    #if docname == 'reference/tlo.methods':
     #   import pdb; pdb.set_trace()
    pass

def doctree_read_handler(app, doctree):
    #import pdb; pdb.set_trace()

    # doctree.nameids ['tlo.methods.chronicsyndrome module']
    #    = 'tlo-methods-chronicsyndrome-module'
    # etc.
    # (Pdb) doctree
    # <document: <section "tlo.methods package"...>>
    # (Pdb) type(doctree)
    # <class 'docutils.nodes.document'>
    # mylist = [section for section in doctree.traverse(docutils.nodes.section)]
    # typical entry in mylist:   mysection = mylist[3]
    # <section "tlo.methods.chronicsyndrome module": <title...><index...><index...><desc...><index...><desc.. ...>
    # title = mysection.next_node(docutils.nodes.Titular)
    # (Pdb) title.astext()
    # 'tlo.methods.chronicsyndrome module'
    # (Pdb) type(mysection)
    # <class 'docutils.nodes.section'>
    # mysection.astext() --> lots of text
    # (Pdb) doctree.__class__
    # <class 'docutils.nodes.document'>
    #(Pdb) app.doctreedir
    #'/Users/matthewgillman/repos/addtables/TLOmodel/dist/docs/.doctrees'
    # (Pdb) app.outdir
    # '/Users/matthewgillman/repos/addtables/TLOmodel/dist/docs'
    #
    # In folder /Users/matthewgillman/repos/addtables/TLOmodel/dist/docs/.doctrees/reference
    # are various files e.g. tlo.methods.doctree, tlo.events.doctree, tlo.parameters.doctree
    # (Pdb) app.env.get_doctree('reference/tlo.methods')
    # <document: <section "tlo.methods package"...>>

    pass

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
    ###AttributeDocumenter.add_directive_header = add_directive_header

    # Not impl in Documenter base class:
    AttributeDocumenter.can_document_member = can_document_member
    #AttributeDocumenter.can_document_member = matt_can_document_member

    #app.connect('autodoc-process-docstring',
    #            cut_lines(1))  #, what=['tlo.methods.antenatal_care.CareOfWomenDuringPregnancy.PARAMETERS']))

    # We want to define our own version of Documenter.can_document_member()
    # and put our functionality in there, rather than having our own versions
    # of add_content() and add_directive_header()

    app.connect('autodoc-skip-member', skip)
    app.connect("autodoc-process-docstring", anotherfunc)
    #app.connect("autodoc-process-signature", process_signature)

    app.connect('source-read', source_read_handler)

    #app.connect('doctree-read', doctree_read_handler)
    app.connect('doctree-resolved', doctree_resolved_handler)

    # When the autodoc-process-docstring event is emitted, handle it with
    # add_dict_to_docstring():
    app.connect("autodoc-process-docstring", add_dicts_to_docstring)


def skip(app, what, name, obj, skip, options):
    """
    From the docs:
    what – the type of the object which the docstring belongs to
    (one of "module", "class", "exception", "function", "method", "attribute")

    In practice, 'what' seems to be 'class', or sometimes 'module',
    even if obj is a function object.
    So 'what' doesn't seem that useful, unfortunately.

    A function can have a nice short string rep, e.g. <function HSI_Tb_Ipt.apply at 0x105fbf400>
    But can be a monster

    e.g. what is a class 'str', name is PARAMETERS,
    obj is a dict, skip is False
    options is the dictionary autodoc_default_options, except for
    some reason the 'undoc-members' has been changed to True
    e.g. obj = <function Module.on_birth at 0x107a45488>
    # or <function CareOfWomenDuringPregnancy.initialise_simulation at 0x1089709d8>
    """


    '''tlo.methods.* - most class members should not be displayed.
    Have a means to control this. e.g. classes that inherit tlo.Module
    we wouldn't display, say, apply() or name, on_birth(), initialise_simulation()'''

    # Is this a class object?
    # e.g. obj = <class 'tlo.methods.antenatal_care.CareOfWomenDuringPregnancy'>

    if name in ('PARAMETERS', 'PROPERTIES'):
        #
        # Don't bother displaying PARAMETERS or
        # PROPERTIES dictionaries if they have no data.
        if not obj:
            return True

    global class_being_tracked
    #global class_object_being_tracked  # Unreliable :-(

    if "class" in str(obj):
        class_being_tracked = name  # e.g. 'Date', 'Module', 'Parameter', 'Simulation', '__builtins__'
        print (f"Now tracking {class_being_tracked}")  #- rep is " + str(obj))
        if class_being_tracked in ['__builtins__', '__doc__',
                                   'PANDAS_TYPE_MAP', 'PYTHON_TYPE_MAP',
                                   '__module__', '__dict__']:
            #class_object_being_tracked = None
            return True
        #else:
        #class_object_being_tracked = obj  # e.g. instance of tlo.x.x.CareOfWomenDuringPregnancy
        # do we even need to track it?

    else:
        #if "function" in str(obj):
        if (str(obj)).startswith("<function "):
            # e.g. str(obj) = "<function hiv.initialise_simulation at 0x105fbff28>"
            # sometimes no class present e.g. str(obj) = "<function create_age_range_lookup at 0x105442ae8>"
            func = str(obj)
            # Update which class this belongs to, if necessary:
            items = func.split()
            class_and_func = []
            if len(items) > 0:
                class_and_func = items[1]
            parts = class_and_func.split('.')
            if len(parts) > 1:
                class_being_tracked = parts[0]
                this_function = parts[1]
            else:  # No class specified
                this_function = parts[0]
            #if "Epilepsy" == class_being_tracked:
            #    print(f"Got function {func}; {what}; {this_function}; currently tracking {class_being_tracked}")

            # Filter out certain methods in subclasses of tlo.core.Module:
            # Doing this programmatically would be better than this,
            # but unfortunately we can't rely on class_object_being_tracked
            # being up-to-date
            global subclasses_of_Module
            if class_being_tracked in subclasses_of_Module:
                global common_methods_to_skip
                if this_function in common_methods_to_skip:
                    return True



    # From Sphinx docs:
    # "Handlers should return None to fall back
    # to the skipping behavior of autodoc and
    # other enabled extensions."
    return None


#  Emitted when autodoc has read and processed a docstring
# i.e. **too late** to affect first appearance of dict???
def anotherfunc(app, what, name, obj, options, lines):
    pass
    #import pdb; pdb.set_trace()
    #if name.split(".")[-1] in ('PARAMETERS', 'PROPERTIES'):
    #global count
    #if name == 'tlo.methods.antenatal_care.CareOfWomenDuringPregnancy.PARAMETERS':
       # import pdb; pdb.set_trace()
       # if count == 0:
        #options['undoc-members'] = False
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
            #import pdb; pdb.set_trace()
            item = mydict[key]
            #temp = nullstr(key)
            #k = str(temp.sigh())
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

