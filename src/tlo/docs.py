'''The functions used by tlo_methods_rst.py.'''

import inspect
from os import walk


def get_package_name(dirpath):
    '''
    Given a file path to a TLO package, return the name in dot form.

    :param dirpath: the path to the package,
           e.g. (1) "./src/tlo/logging/sublog" or (2) "./src/tlo"
    :return: string of package name in dot form,
           e.g. (1) "tlo.logging.sublog" or (2) "tlo"
    '''
    TLO = "/tlo/"
    if TLO not in dirpath:
        raise ValueError(f"Sorry, {TLO} isn't in dirpath ({dirpath})")
    parts = dirpath.split(TLO)
    # print(f"parts = {parts}")
    runt = parts[-1]  # e.g. "logging/sublog"
    runt = runt.replace("/", ".")  # e.g. logging.sublog
    # print(f"now runt is {runt}")
    if runt:
        package_name = f"tlo.{runt}"
    else:
        package_name = "tlo"
    return package_name


def generate_module_dict(topdir):
    '''
    Given a root directory topdir, iterates recursively
    over top dir and the files and subdirectories within it.
    Returns a dictionary with each key = path to a directory
    (i.e. to topdir or one of its nested subdirectories), and
    value = list of Python .py files in that directory.

    :param topdir: root directory to traverse downwards from, iteratively.
    :returns: dict with key = a directory,
              value = list of Python files in dir `key`.
    '''
    data = {}  # key = path to dir, value = list of .py files

    for (dirpath, dirnames, filenames) in walk(topdir):
        # print(f"**path:{dirpath}, dirnames:{dirnames}, files:{filenames}\n")
        if "__pycache__" in dirpath:
            continue
        if dirpath not in data:
            data[dirpath] = []
        for f in filenames:
            # We can do this as compound-if statements are evaluated
            # left-to-right in Python:
            if (f == "__init__.py") or (f[-4:] == ".pyc") or (f[-3:] != ".py"):
                print(f"skipping {f}")
            else:
                (data[dirpath]).append(f)
    return data


def get_classes_in_module(fqn, module_obj):
    '''
    Generate a list of lists of the classes *defined* in
    the required module (file), in the order in which they
    appear in the module file. Note that this excludes
    any other classes in the module, such as those brought
    in by inheritance, or imported..

    Each entry in the list returned is itself a list of:
    [class name, class object, line number]

    :param fqn: Fully-qualified name of the module,
                e.g. "tlo.methods.mockitis"
    :param module_obj: an object representing this module
    :return: list of entries, each of the form:
                [class name, class object, line number]
    '''
    classes = []
    module_info = inspect.getmembers(module_obj)  # Gets everything
    for name, obj in module_info:

        # Pick out only the classes defined in this module:
        if inspect.isclass(obj) and fqn in str(obj):
            # print(name)  # e.g. MockitisEvent
            # print(obj)  # e.g. <class 'tlo.methods.mockitis.MockitisEvent'>
            source, start_line = inspect.getsourcelines(obj)
            classes.append([name, obj, start_line])
            # print(f"\n\nIn module {name} we find the following:")
            # morestuff = inspect.getmembers(obj)
            # print(morestuff)  # e.g. functions, PARAMETERS dict,...

    # print(f"before sorting, {classes}")
    # https://stackoverflow.com/questions/3169014/inspect-getmembers-in-order
    # Based on answer by Andrew - sort them into order in which they are
    # defined in the module file.
    classes.sort(key=lambda x: x[2])
    # print(f"after sorting, {classes}")
    return classes


def get_fully_qualified_name(filename, context):
    '''
    Given a file name and a context
    return the fully-qualified name of the *module*

    e.g. tlo.methods.mockitis
    This gets used in doc page title.

    :param filename: name of file, e.g. "mockitis.py"
    :param context: "location" of module, e.g. "tlo.methods"
    :return: a string, e.g. "tlo.methods.mockitis"
    '''
    parts = filename.split(".")
    # print(f"getfqn: {filename}, {context}")
    if context == "":
        return parts[0]
    else:
        fqname = f"{context}.{parts[0]}"
        return fqname


def write_rst_file(rst_dir, fqn, mobj):
    '''
    We want to write one .rst file per module in ./src/tlo/methods

    :param rst_dir: directory in which to write the new .rst file
    :param fqn: fully-qualified module name, e.g. "tlo.methods.mockitis"
    :param mobj: the module object
    '''
    filename = f"{rst_dir}/{fqn}.rst"
    with open(filename, 'w') as out:

        # Header
        title = f"{fqn} module"
        out.write(f"{title}\n")
        for i in range(len(title)):
            out.write("=")
        out.write("\n\n")

        out.write(f".. automodule:: {fqn}")

        # NB This gets the classes, but not any docstring e.g. at the top
        # of the module, outside any classes. It also doesn't include
        # any module-level functions. But those items seem to get added in
        # anyway.
        # NB classes_in_module may be empty:
        classes_in_module = get_classes_in_module(fqn, mobj)
        for c in classes_in_module:
            # c is [class name, class object, line number]
            str = get_class_output_string(c)
            out.write(f"{str}\n")
        # Example of use in .rst file:
        # out.write(".. class:: Noodle\n\n")
        # out.write("   Noodle's docstring.\n")


def get_class_output_string(classinfo):
    '''Generate output string for a class to be written to an rst file

    :param classinfo: a list with [class name, class object, line number]
    :return: the string to output

    '''
    class_name, class_obj, _ = classinfo
    str = f"\n\n\n.. autoclass:: {class_name}\n\n"

    # This is needed to keep information neatly aligned
    # with each class.
    numspaces = 5
    spacer = numspaces * ' '

    # Now we want to add base classes, class members, comments and methods
    # Presumably in source file order.
    is_child_of_HSI_Event = False
    is_child_of_Module = False

    (base_str, base_objects) = extract_bases(class_name, class_obj, spacer)

    hsi_event_base_object = None
    # module_base_object = None
    for mybase in base_objects:
        if mybase.__name__ == "HSI_Event":
            is_child_of_HSI_Event = True
            hsi_event_base_object = mybase
        elif mybase.__name__ == "Module":
            is_child_of_Module = True
            # module_base_object = mybase

    # Make sure it has only matched one of the criteria in the loop above:
    if is_child_of_Module and is_child_of_HSI_Event:
        # I don't think this will ever happen, but if it does...
        import pdb
        pdb.set_trace()
        # Maybe raising an assertion would be better?

    str += base_str
    str += "\n\n"

    # Asif says we should probably not exclude __init__ in the following,
    # because some disease classes have custom arguments:
    general_exclusions = ["__class__", "__dict__", "__module__",
                          "__slots__", "__weakref__", ]

    module_inherited_exclusions = ["initialise_population",
                                   "initialise_simulation",
                                   "on_birth", "read_parameters", "apply",
                                   "post_apply_hook", "on_hsi_alert",
                                   "report_daly_values", "run", "did_not_run",
                                   "SYMPTOMS", ]

    # For classes which inherit from HSI_Event, we do not wish to generate docs
    # for the functions which they inherit UNLESS they have produced a new
    # docstring in the child class.
    # Even if HSI_Event's function body is simply "pass", we add it to the
    # list of inherited exclusions.
    # TODO Should __init__ be in this list?
    hsi_event_inherited_functions = ["__init__", "not_available",
                                     "post_apply_hook", "run",
                                     "get_all_consumables",
                                     "make_appt_footprint", "never_ran",
                                     "did_not_run", ]

    # hsi_event_not_implemented = ["apply",  ]  # Not needed ?

    # Asif: I think it's mandatory to a subclass of HSI Event to have an
    # apply() implementation, so I think that should always be displayed
    # actually. To encourage people to add a sensible docstring for that
    # method. The other [post_apply_hook(), which just has pass as its
    # implementation in the parent] is optional, so should only be
    # displayed if it's actually overridden

    # Return all the members of an object in a list of (name, value) pairs
    # sorted by name:
    classdat = inspect.getmembers(class_obj)  # Gets everything

    for name, obj in classdat:
        # We only want to document things defined in this class itself,
        # rather than anything inherited from parent classes (including
        # the basic "object" class).
        # e.g. we don't want:
        # __delattr__ = <slot wrapper '__delattr__' of 'object' objects>
        # Skip over things inherited from object class or Module class
        object_description = f"{obj}"

        if (is_child_of_Module and
            ("of 'object' objects" in object_description
                or "built-in method" in object_description
                or "function Module." in object_description
                or "of 'Module' objects" in object_description
                or name in general_exclusions
                or name in module_inherited_exclusions
                or object_description == 'None')):
            continue

        # if name == "__doc__" and obj is not None:
        #    str += f"**Description:**\n{obj}"
        #    continue
        # print(f"next object in class {class_name} is {name} = {obj}")

        # We want nice tables for PARAMETERS and PROPERTIES
        if name in ("PARAMETERS", "PROPERTIES"):
            table_list = create_table(obj)
            if table_list == []:
                continue
            str += f"{spacer}**{name}:**\n"
            for t in table_list:
                str += f"{spacer}{t}\n"
            str += "\n\n"
            continue

        # Interrogate the object. It's something else, maybe a
        # function which hasn't been filtered out.
        #
        if inspect.isfunction(obj):
            # print(f"DEBUG: got a function: {name}, {object}")

            if is_child_of_HSI_Event and name in hsi_event_inherited_functions:
                # print(f"DEBUG*** {name}, {class_name}")

                if skip_child_doc(hsi_event_base_object, obj, name):
                    continue

            # Document this function if necessary:
            str += f"{spacer}.. automethod:: {name}\n\n"
            continue

        # Anything else?
        # str += f"{name} : {obj}\n\n"
        # print(f"DEBUG: something else... {name}, {obj}")

        # getdoc, getcomments,

    str += "\n\n\n"

    return str


def skip_child_doc(base_obj, func_obj, func_name):
    '''
    Does the child function docstring match that of the parent's?

    :param base_obj: an object instance of the base class
    :param func_obj: the child class function object
    :param func_name: the name of the function concerned.
    :return: True if both present and identical, else False.

    We only want to document the child class's function if its
    docstring is present and different to the parent's, i.e. False.
    We return True when we don't want the child class's function docstring
    included in the docs we generate.

    Obviously, if the child function has a docstring, but the
    parent's function doesn't, we shouldn't skip.
    '''
    # if func_obj is None:
    #     return True
    assert base_obj is not None
    assert func_obj is not None

    func_doc_string = func_obj.__doc__
    if func_doc_string in ["", None]:
        return True

    # Iterate over the base class object.
    base_data = inspect.getmembers(base_obj)
    for datum in base_data:  # Each datum is [name, object]
        base_name = datum[0]
        if base_name is func_name:
            base_func_obj = datum[1]
            if base_func_obj.__doc__ == func_doc_string:
                return True

    return False


def extract_bases(class_name, class_obj, spacer=""):
    '''
    Document which classes this class inherits from,
    except for the object class or this class itself.

    :param class_name: name of the class (e.g. Mockitis) for which we want the
                       bases
    :param class_obj: object with information about this class
    :param spacer: string to use as whitespace padding.
    :return: Tuple with (1) string of base(s) for this class (if any), with
             links to their docs. (2) the list of base class objects
    '''
    # This next line gets the base classes including "object" class,
    # and also the class_name class itself:
    bases = inspect.getmro(class_obj)
    # Typical example of "bases":
    # (<class 'tlo.methods.mockitis.Mockitis'>,
    # <class 'tlo.core.Module'>, <class 'object'>)
    # print(f"bases for {class_name}: {bases}")
    # classtree = inspect.getclasstree(bases)
    # print(f"classtree = {classtree}")

    parents = []
    relevant_bases = []

    for b in bases:
        # Example of this_base_string:
        # `tlo.core.Module <./tlo.core.html#tlo.core.Module>`
        this_base_string = get_base_string(class_name, class_obj, b)
        if this_base_string is not (None or ""):
            parents.append(this_base_string)
        if ("object" not in str(b) and class_name not in str(b)):
            relevant_bases.append(b)

    if len(parents) > 0:
        out = f"{spacer}Bases: {', '.join(parents)}\n"
    else:
        out = ""

    # print(f"DEBUG: extract_bases: str = {str}")
    return (out, relevant_bases)


def get_base_string(class_name, class_obj, base_obj):
    '''
    For this object, representing a base class,
    extract its name and add a hyperlink to it.
    Unless it is the root 'object' class, or the name of the class itself.

    :param class_name: the name of the class (e.g. "Mockitis")
                       for which obj is a base,
    :param class_obj: object representing the class
    :param base_obj: the object representation of the *base* class,
                     e.g. <class 'tlo.core.Module'> or <class 'object'>
                     or <class 'tlo.methods.mockitis.Mockitis'>
    :return: string as hyperlink
    '''
    # Extract fully-qualified name of base (e.g. "tlo.core.Module")
    # from its object representation (e.g. "<class 'tlo.core.Module'>")
    # print (f"DEBUG: before = {class_name}, {base_obj}")
    fqn = (str(base_obj)).replace("<class '", "").replace("'>", "")
    # print(f"DEBUG: after = {class_name}, {fqn}")
    # The next line's getsourcefile() call will raise a TypeError
    # if the object is a built-in module, class, or function:
    # print(f"and filename: {inspect.getsourcefile(base_obj)},
    #  module: {inspect.getmodule(base_obj)}")
    # module like: <module 'tlo.methods.mockitis' from ...

    # fqn will now be something like "tlo.methods.mockitis.Mockitis"
    # or "tlo.core.Module"
    if "." in fqn:
        parts = fqn.split(".")
        name = parts[-1]   # e.g. Mockitis or Module
    else:
        name = fqn

    if name in [class_name, "object"]:
        return ""

    # link = get_link(fqn, base_obj)

    # We want the final HTML to be like:
    # Bases: <a class="reference internal" href="tlo.core.html#tlo.core.Module"
    #     title="tlo.core.Module">
    #     <code class="xref py py-class docutils literal notranslate">
    #     <span class="pre">tlo.core.Module</span></code></a></p>
    # So we need to write the right stuff to the .rst file.
    # We want to return a string with the name of the base class, plus
    # the link to its documentation.
    # e.g. the string might be:
    # "tlo.core.Module <./tlo.core.html#tlo.core.Module>_"
    #
    # 3rd November 2020 - Asif suggests it should be more like:
    # :class: `tlo.events.Event`
    # or :py:class: `tlo.events.Event`
    # to fix the "broken" link issue on Travis (#204)
    # was: mystr = "`" + fqn + " " + link + "`_"
    mystr = f":class:`{fqn}`"
    # print(f"DEBUG: get_base_string(): {mystr}")
    return mystr


def get_link(base_fqn, base_obj):
    '''Given a name like Mockitis and a base_string like
    tlo.core.Module, create a link to the latter's doc page.

    :param base_fqn: fully-qualified name of the base class,
                     e.g. tlo.core.Module
    :param base_obj: the object representing the base class
    :return: a string with the link to the base class's place in the generated
             documentation.
    '''
    # Example of module_defining_class: <module 'tlo.core' from '/Users/...
    module_defining_class = str(inspect.getmodule(base_obj))
    module_pieces = module_defining_class.split(" ")
    if len(module_pieces) > 1:
        module_name = module_pieces[1]  # e.g. 'tlo.core'
    else:
        module_name = module_defining_class
    module_name = module_name.replace("'", "")  # Remove single speech marks
    # print(f"My lovely module is {module_name}")  # e.g. tlo.core

    # Need name of base's file + fqn of base
    # We want a link string like: <./tlo.core.html#tlo.core.Module>
    # i.e. <base's file name#base's fqn>

    # Example link to a base class (from a class in reference directory):
    # http://0.0.0.0:8000/reference/tlo.core.html#tlo.core.Module
    # In this code the link would be <./tlo.core.html#tlo.core.Module>
    link_to_base = "<./" + module_name + ".html" + "#" + base_fqn + ">"
    # print(f"link_to_base = {link_to_base}")
    return link_to_base


def create_table(mydict):
    '''
    Dynamically create a table of arbitrary length
    from a PROPERTIES or PARAMETERS dictionary.

    :param mydict: the dictionary object.
    :return: A list of strings, used for output.

    NB Do not change the positioning of items in the
    f-strings below, or things will break!

    The table is returned as a list of strings.
    If there is no data, an empty list is returned.
    '''

    examplestr = '''
.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Item
     - Type
     - Description
'''
    if len(mydict) == 0:
        return []
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

            row = f'''   * - {key}
     - {the_type}
     - {description}
'''
            examplestr += row
    mylist = examplestr.splitlines()
    return mylist
