"""The functions used by tlo_methods_rst.py."""

import inspect
from inspect import (
    isclass,
    iscode,
    isframe,
    isfunction,
    ismethod,
    ismodule,
    istraceback,
)
from os import walk

from tlo import Module
from tlo.events import Event
from tlo.methods.healthsystem import HSI_Event


def get_package_name(dirpath):
    """
    Given a file path to a TLO package, return the name in dot form.

    :param dirpath: the path to the package,
           e.g. (1) "./src/tlo/logging/sublog" or (2) "./src/tlo"
    :return: string of package name in dot form,
           e.g. (1) "tlo.logging.sublog" or (2) "tlo"
    """
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
    """
    Given a root directory topdir, iterates recursively
    over top dir and the files and subdirectories within it.
    Returns a dictionary with each key = path to a directory
    (i.e. to topdir or one of its nested subdirectories), and
    value = list of Python .py files in that directory.

    :param topdir: root directory to traverse downwards from, iteratively.
    :returns: dict with key = a directory,
              value = list of Python files in dir `key`.
    """
    data = {}  # key = path to dir, value = list of .py files

    for dirpath, dirnames, filenames in walk(topdir):
        # print(f"**path:{dirpath}, dirnames:{dirnames}, files:{filenames}\n")
        if "__pycache__" in dirpath:
            continue
        if dirpath not in data:
            data[dirpath] = []
        for f in filenames:
            # We can do this as compound-if statements are evaluated
            # left-to-right in Python:
            if (f == "__init__.py") or (f[-4:] == ".pyc") or (f[-3:] != ".py"):
                # print(f"skipping {f}")
                pass
            else:
                (data[dirpath]).append(f)
    return data


def get_classes_in_module(fqn, module_obj):
    """
    Generate a list of lists of the classes *defined* in
    the required module (file), in the order in which they
    appear in the module file. Note that this excludes
    any other classes in the module, such as those brought
    in by inheritance, or imported.

    Each entry in the list returned is itself a list of:
    [class name, class object, line number]

    :param fqn: Fully-qualified name of the module,
                e.g. "tlo.methods.mockitis"
    :param module_obj: an object representing this module
    :param return: list of entries, each of the form:
                [class name, class object, line number]
    """
    classes = []
    module_info = inspect.getmembers(module_obj)  # Gets everything
    for name, obj in module_info:

        # Pick out only the classes defined in this module:
        if inspect.isclass(obj) and fqn in str(obj):
            # only generate documentation for tlo subclasses (considers other classes internal implementation detail)
            bases = inspect.getmro(obj)
            # skip this filtering if we're working with classes for testing
            if fqn.startswith("test_docs_data") or any(
                base in bases for base in [Module, Event, HSI_Event]
            ):
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
    """
    Given a file name and a context
    return the fully-qualified name of the *module*

    e.g. tlo.methods.mockitis
    This gets used in doc page title.

    :param filename: name of file, e.g. "mockitis.py"
    :param context: "location" of module, e.g. "tlo.methods"
    :return: a string, e.g. "tlo.methods.mockitis"
    """
    parts = filename.split(".")
    # print(f"getfqn: {filename}, {context}")
    if context == "":
        return parts[0]
    else:
        fqname = f"{context}.{parts[0]}"
        return fqname


def write_rst_file(rst_dir, fqn, mobj):
    """
    We want to write one .rst file per module in ./src/tlo/methods

    :param rst_dir: directory in which to write the new .rst file
    :param fqn: fully-qualified module name, e.g. "tlo.methods.mockitis"
    :param mobj: the module object
    """
    filename = f"{rst_dir}/{fqn}.rst"
    with open(filename, "w") as out:

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
            mystr = get_class_output_string(c)
            out.write(f"{mystr}\n")
        # Example of use in .rst file:
        # out.write(".. class:: Noodle\n\n")
        # out.write("   Noodle's docstring.\n")


def get_class_output_string(classinfo):
    """Generate output string for a single class to be written to an rst file.

    :param classinfo: a list with [class name, class object, line number]
    :return: the string to output
    """
    class_name, class_obj, _ = classinfo
    # mystr = f"\n\n\n" \
    #        f".. autoclass:: {class_name}\n" \
    #        f"   :members:\n\n" \
    #        f"   ..automethod:: __init__\n\n"
    mystr = f"\n\n\n.. autoclass:: {class_name}\n\n"

    # This is needed to keep information neatly aligned
    # with each class.
    numspaces = 5
    spacer = numspaces * " "

    # Now we want to add base classes, class members, comments and methods
    # Presumably in source file order.

    (base_str, base_objects) = extract_bases(class_name, class_obj, spacer)
    mystr += base_str
    mystr += "\n\n"

    # if class_name == 'Matt':  #Mockitis' or class_name == "Matt":
    #     import pdb; pdb.set_trace()
    # class_obj.__dict__ is
    # mappingproxy({'__module__': 'tlo.methods.mockitis',
    #   'another_random_boolean': False, '__doc__': None})

    # Asif says we should probably not exclude __init__ in the following,
    # because some disease classes have custom arguments:
    # general_exclusions = ["__class__", "__dict__", "__module__",
    #                      "__slots__", "__weakref__", ]

    # inherited_exclusions = ["initialise_population", "initialise_simulation",
    #                        "on_birth", "read_parameters", "apply",
    #                        "post_apply_hook", "on_hsi_alert",
    #                        "report_daly_values", "run", "did_not_run",
    #                        "SYMPTOMS", ]

    # Return all the members of an object in a list of (name, value) pairs
    # sorted by name:
    classdat = inspect.getmembers(class_obj)  # Gets everything

    # if class_name == 'Matt':  #Mockitis' or class_name == "Matt":
    #    import pdb; pdb.set_trace()

    # We want to sort classdat by line number, so that functions
    # defined in, or overridden in, this class will be
    # documented in source file order.

    # 25/11 This is the order Asif would like:
    # For subclasses of Module, PARAMETERS & PROPERTIES first as tables
    # (in that order). Then for each of the subclass classes (Module, Event,
    # HSI_Event) attributes of the class (in source-code order if possible,
    # otherwise default) and then functions of the class
    # (in source-code order).
    # Like we did for functions, for attributes, if we can have the same logic
    # (i.e. only show if defined in the subclass), that'd be great. This might
    # be useful: https://stackoverflow.com/a/5253424

    # TODO we only want to do this for descendants of classes
    # Module, Event, and HSI_Event:
    func_objects_to_document = which_functions_to_print(class_obj)
    # attributes_to_document = which_attributes_to_print(class_obj)

    name_func_lines = []  # List of tuples to sort later

    my_attributes = []  # (name, value) pairs of class attributes.

    ignored_attributes = ["__doc__", "__module__", "__weakref__"]

    misc = []

    for name, obj in classdat:
        # First loop. Get PARAMETERS and PROPERTIES dictionary objects only.
        # These are in subclasses of Module only.
        # We want nice tables for PARAMETERS and PROPERTIES.
        # In this case, obj is a dictionary.
        # We want to display PARAMETERS before PROPERTIES. We should get that
        # for free as inspect.getmembers() returns results sorted by name.
        if name in ("PARAMETERS", "PROPERTIES") and name in class_obj.__dict__:
            # (only included if defined/overridden in this class)
            table_list = create_table(obj)
            if table_list == []:
                continue
            mystr += f"{spacer}**{name}:**\n"
            for t in table_list:
                mystr += f"{spacer}{t}\n"
            mystr += "\n\n"

        # Get source-code line numbering where possible.
        elif (
            isfunction(obj)
            and func_objects_to_document
            and obj in func_objects_to_document
            and name in class_obj.__dict__
        ):
            _, start_line_num = inspect.getsourcelines(obj)
            name_func_lines.append((name, obj, start_line_num))

        # Get source-code line numbering where possible.
        # inspect.getsourcelines() only works for module, class, method,
        # function, traceback, frame, or code objects
        elif (
            ismodule(obj)
            or isclass(obj)
            or ismethod(obj)
            or istraceback(obj)
            or isframe(obj)
            or iscode(obj)
        ) and (name in class_obj.__dict__):
            # pass  # _, start_line_num = inspect.getsourcelines(obj)
            misc.append(name, obj, start_line_num)

        elif name == "__dict__":  # Skip over mappingproxy dict.
            continue

        else:
            # We want class attributes but we can't get source code order
            # We want only those which are defined or overridden in this class
            # continue  #attributes.append()
            if name in class_obj.__dict__ and name not in ignored_attributes:
                my_attributes.append((name, obj))

    # Output attributes other than functions
    # for name, obj in classdat:
    #    pass
    if my_attributes:
        mystr += f"\n{spacer}**Class attributes:**\n\n"
        for att in my_attributes:
            mystr += f"{spacer}{att[0]} : {att[1]}\n\n"

    # Sort miscellaneous items:
    if misc:
        misc.sort(key=lambda x: x[2])
        mystr += f"\n{spacer}**Miscellaneous:**\n\n"
        for m in misc:
            mystr += f"{spacer}{m[0]} : {m[1]}\n\n"

    # Sort the functions we wish to document into source-file order:
    name_func_lines.sort(key=lambda x: x[2])

    # if class_name == "Mockitis":
    #     import pdb; pdb.set_trace()

    # New or overridden functions only.
    if func_objects_to_document:
        mystr += (
            f"{spacer}**Functions (defined or overridden in "
            f"class {class_name}):**\n\n"
        )
        for name, obj, _ in name_func_lines:  # Now in source code order

            if obj in func_objects_to_document:  # Should be always True!
                mystr += f"{spacer}.. automethod:: {name}\n\n"
    else:
        # print(f"**DEBUG: no func_objects_to_document in class {class_name}")
        pass

    # Anything else?
    # mystr += f"{name} : {obj}\n\n"
    # print(f"DEBUG: something else... {name}, {obj}")

    # getdoc, getcomments,

    mystr += "\n\n\n"

    return mystr


def which_functions_to_print(clazz):
    """
    Which functions do we want to print?

    :param clazz: class object under consideration
    :return: returns a list of function objects we want to print

    Written by Asif
    """
    # get all the functions in this class
    class_functions = dict(inspect.getmembers(clazz, predicate=inspect.isfunction))

    ok_to_print = []

    # for each function in this class
    for func_name, func_obj in class_functions.items():
        # for func in func_list:
        # func_name, func_obj, _ = func
        # import pdb; pdb.set_trace()
        should_i_print = True

        # for each base class of this class
        for baseclass in clazz.__mro__:
            # skip over the class we're checking
            if baseclass != clazz:
                # get the functions of the base class
                functions_base_class = dict(
                    inspect.getmembers(baseclass, predicate=inspect.isfunction)
                )

                # if there is a function with the same name as base class
                if func_name in functions_base_class:
                    # if the function object is the same
                    # as one defined in a base class
                    if func_obj == functions_base_class[func_name]:
                        # print(f'{func_name} in subclass is same as function in'
                        #       f'{baseclass.__name__} (not overridden)')
                        should_i_print = False
                        break
                    else:
                        # print(f'{func_name} in subclass is not the same '
                        #       f'as one in baseclass {baseclass.__name__}')
                        pass
                else:
                    # print(f'{func_name} is not in '
                    #       f'baseclass {baseclass.__name__}')
                    pass

        if should_i_print:
            # print(f'\t✓✓✓ {func_name} is implemented in the subclass - print ')
            ok_to_print.append(func_obj)
        else:
            # print(f'\txxx {func_name} has been inherited from a subclass'
            #       f'- do not print')
            pass

    return ok_to_print


def extract_bases(class_name, class_obj, spacer=""):
    """
    Document which classes this class inherits from,
    except for the object class or this class itself.

    :param class_name: name of the class (e.g. Mockitis) for which we want the
                       bases
    :param class_obj: object with information about this class
    :param spacer: string to use as whitespace padding.
    :return: Tuple with (1) string of base(s) for this class (if any), with
             links to their docs. (2) the list of base class objects
    """
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

        # We don't want to include the name of the child class.
        # Or the "object" class, which all objects will
        # ultimately inherit from.
        if "object" not in str(b) and class_name not in str(b):
            relevant_bases.append(b)

    if len(parents) > 0:
        out = f"{spacer}Bases: {', '.join(parents)}\n"
    else:
        out = ""

    # print(f"DEBUG: extract_bases: mystr = {mystr}")
    return out, relevant_bases


def get_base_string(class_name, class_obj, base_obj):
    """
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
    """
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
        name = parts[-1]  # e.g. Mockitis or Module
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
    """Given a name like Mockitis and a base_string like
    tlo.core.Module, create a link to the latter's doc page.

    :param base_fqn: fully-qualified name of the base class,
                     e.g. tlo.core.Module
    :param base_obj: the object representing the base class
    :return: a string with the link to the base class's place in the generated
             documentation.
    """
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
    """
    Dynamically create a table of arbitrary length
    from a PROPERTIES or PARAMETERS dictionary.

    :param mydict: the dictionary object.
    :return: A list of strings, used for output.

    NB Do not change the positioning of items in the
    f-strings below, or things will break!

    The table is returned as a list of strings.
    If there is no data, an empty list is returned.
    """

    examplestr = """
.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Item
     - Type
     - Description
"""
    if len(mydict) == 0:
        return []
    else:
        for key in mydict:
            item = mydict[key]
            description = item.description
            mytype = item.type_  # e.g. <Types.REAL: 4>
            the_type = mytype.name  # e.g. 'REAL'

            if the_type == "CATEGORICAL":
                description += ".  Possible values are: ["
                mylist = item.categories
                for mything in mylist:
                    description += f"{mything}, "
                description += "]"

            row = f"""   * - {key}
     - {the_type}
     - {description}
"""
            examplestr += row
    mylist = examplestr.splitlines()
    return mylist
