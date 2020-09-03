# Best to run this from within PyCharm configuration
# with working directory set to (e.g.)
# /Users/matthewgillman/PycharmProjects/TLOmodel
# and Script path
# /Users/matthewgillman/PycharmProjects/TLOmodel/src/tlo/inspector.py
import inspect
import importlib
from os import walk
import tlo
from tlo import Module

from pathlib import Path


# Ideally make these defaults but have command-line options.
root_dir = Path(__file__).resolve().parents[2]
MODULE_DIR = f"{root_dir}/src/tlo/methods"   #./src/tlo/methods"
LEADER = "tlo.methods"
RST_DIR = f"{root_dir}/docs/reference"

# Use this so we can dynamically import, and not need to hard-code it.
#exec("from tlo.methods import mockitis")


def generate_module_list(dir_string):
    '''Obtain all Python module files in this directory
    e.g. mockitis.py, hiv.py, healthsystem.py...
    in an ASCII-sorted list.
    Actually, it returns all files - need to change?

    :param dir_string: string containing path to directory to process.

    For each directory in the tree, walk() yields a 3-tuple
    (dirpath, dirnames, filenames)
    Use break to prevent recursive search.

    Reference:
    https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    '''

    for (dirpath, dirnames, filenames) in walk(dir_string):
        #if '__pycache__' in dirnames:
        #    dirnames.remove('__pycache__')
        # print(f"dirpath = {dirpath}")
        # print(f"dirnames = {dirnames}")
        # print(f"filenames = {filenames}")
        if '__init__.py' in filenames:
            filenames.remove('__init__.py')
        break  # TODO: In this version we don't traverse any subdirectories
               # We will want to change this later so we can handle the
               # case when tlo.methods has submodules (i.e. subdirectories),
               # within which the individual disease modules are located
               # e.g. we might have sub-level tlo.methods.rmnch for modules
               # dealing with reproduction, birth, child health, etc.

    return sorted(filenames)


def get_classes_in_module(fqn, module_obj):
    '''
    Generate a list of lists of the classes *defined* in
    the required module (file), in the order in which they
    appear in the module file. Note that this excludes
    any other classes in the module, such as those brought
    in by inheritance, or imported..

    Each entry in the list returned is itself a list of:
    [class name, class object, line number]

    :param fqn: Fully-qualified name of the module, e.g. "tlo.methods.mockitis"
    :param module_obj: an object representing this module
    '''
    classes = []
    module_info = inspect.getmembers(module_obj)  # Gets everything
    for name, obj in module_info:
        # Pick out only classes, defined in this module:
        if inspect.isclass(obj) and fqn in str(obj):
            #print(name)  # e.g. MockitisEvent
            #print(obj)  # e.g. <class 'tlo.methods.mockitis.MockitisEvent'>
            source, start_line = inspect.getsourcelines(obj)
            classes.append([name, obj, start_line])
            #classes_in_module.append(obj)
            #print(f"\n\nIn module {name} we find the following:")
            #morestuff = inspect.getmembers(obj)
            #print(morestuff)  # e.g. functions, PARAMETERS dict,...
    print(f"before sorting, {classes}")
    # https://stackoverflow.com/questions/3169014/inspect-getmembers-in-order
    # Based on answer by Andrew - sort them into order in which they are
    # defined in the module file.
    classes.sort(key=lambda x: x[2])
    print(f"after sorting, {classes}")
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
    if context == "":
        return parts[0]
    else:
        fqname = context + "." + parts[0]
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
        out.write("\n")

        # TODO: This gets the classes, but not any docstring e.g. at the top
        # of the module, outside any classes. It also doesn't include
        # any module-level functions.
        classes_in_module = get_classes_in_module(fqn, mobj)
        for c in classes_in_module:
            # c is [class name, class object, line number]
            str = get_class_output_string(c)
            out.write(f"{str}\n")
        #out.write(".. class:: Noodle\n\n")
        #out.write("   Noodle's docstring.\n")


def get_class_output_string(classinfo):
    '''Generate output string for a class to be written to an rst file

    :param classinfo: a list with [class name, class object, line number]
    :return: the string to output

    '''
    class_name, class_obj, _ = classinfo
    str = f".. class:: {class_name}\n\n"

    # Now we want to add base classes, class members, comments and methods
    # Presumably in source file order.

    str += extract_bases(class_name, class_obj)
    str += "\n\n"

    #str += "External hyperlinks, like `Python <http://www.python.org/>`_.\n\n"
    #str += "something `tlo.core.Module <./tlo.core.html#tlo.core.Module>`_\n\n"
    #str += "Multiple: `tlo.core.Module <./tlo.core.html#tlo.core.Module>`_, " \
    #       "baa `tlo.methods.contraception.Contraception <./tlo.methods.contraception.html#tlo.methods.contraception.Contraception>`_\n\n"

    general_exclusions = ["__class__", "__dict__", "__init__", "__module__",
                          "__slots__", "__weakref__",]
    inherited_exclusions = ["initialise_population", "initialise_simulation", "on_birth",
                            "read_parameters", "apply", "post_apply_hook",
                            "on_hsi_alert", "report_daly_values", "run", "did_not_run",
                            "SYMPTOMS",]

    classdat = inspect.getmembers(class_obj)  # Gets everything

    for name, obj in classdat:
        # We only want to document things defined in this class itself,
        # rather than anything inherited from parent classes (including
        # the basic "object" class).
        # e.g. we don't want:
        # __delattr__ = <slot wrapper '__delattr__' of 'object' objects>
        # Skip over things inherited from object class or Module class
        object_description = f"{obj}"
        if ("of 'object' objects" in object_description
            or "built-in method" in object_description
            or "function Module." in object_description
            or "of 'Module' objects" in object_description
            or name in general_exclusions
            or name in inherited_exclusions
            or object_description == 'None'):
            continue

        if name == "__doc__" and obj is not None:
            str += f"**Description:**\n{obj}"
            continue
        #print(f"next object in class {class_name} is {name} = {obj}")

        # We want nice tables for PARAMETERS and PROPERTIES
        if name in ("PARAMETERS", "PROPERTIES"):
            table_list = create_table(obj)
            if table_list == []:
                continue
            str += f"**{name}:**\n"
            for t in table_list:
                str += f"{t}\n"
            str += "\n\n"
            continue

        # Anything else?
        str += f"{name} : {obj}\n\n"

        # getdoc, getcomments,
        #pass
    str += "\n\n\n"

    return str


def extract_bases(class_name, class_obj):
    '''
    Document which classes this class inherits from,
    except for the object class or this class itself.
    :param class_name: name of the class (e.g. Mockitis) for which we want the bases
    :param class_obj: object with information about this class
    :return: string of base(s) for this class (if any), with links to their docs.
    '''
    bases = inspect.getmro(class_obj)  # Includes "object" class and the class_name class
    # Typical example of "bases":
    # (<class 'tlo.methods.mockitis.Mockitis'>, <class 'tlo.core.Module'>, <class 'object'>)
    # print(f"bases for {class_name}: {bases}")
    # classtree = inspect.getclasstree(bases)
    # print(f"classtree = {classtree}")

    parents = []

    for b in bases:
        # Example of this_base_string: `tlo.core.Module <./tlo.core.html#tlo.core.Module>`
        this_base_string = get_base_string(class_name, class_obj, b)
        if this_base_string is not (None or ""):
            #link = get_link(class_name, class_obj, this_base_string)
            # str += "something `tlo.core.Module <./tlo.core.html#tlo.core.Module>`_\n\n"
            #mystr =
            parents.append(this_base_string)

    if len(parents) > 0:
        str = "**Bases:**\n\n"
        #str = f"Bases: {parents[0]}"
        numbase = 1
        str += f"Base #{numbase}: {parents[0]}\n\n"
        for p in parents[1:]:
            numbase += 1
            #str += f", {p}"
            str += f"Base #{numbase}: {p}\n\n"
        #str += "\n\n"
    else:
        str = ""

    return str


def get_base_string (class_name, class_obj, base_obj):
    '''
    For this object, representing a base class,
    extract its name and add a hyperlink to it.
    Unless it is the root 'object' class, or the name of the class itself.

    :param class_name: the name of the class (e.g. "Mockitis") for which obj is a base,
    :param class_obj: object representing the class
    :param base_obj: the object representation of the *base* class,
    e.g. <class 'tlo.core.Module'> or <class 'object'>
    or <class 'tlo.methods.mockitis.Mockitis'>
    :return: string with hyperlink, e.g. `tlo.core.Module <./tlo.core.html#tlo.core.Module>`
    '''
    # Extract fully-qualified name of base (e.g. "tlo.core.Module")
    # from its object representation (e.g. "<class 'tlo.core.Module'>")
    print (f"DEBUG: before = {class_name}, {base_obj}")
    fqn = (str(base_obj)).replace("<class '", "").replace("'>", "")
    print(f"DEBUG: after = {class_name}, {fqn}")
    # The next line's getsourcefile() call will raise a TypeError
    # if the object is a built-in module, class, or function:
    # print(f"and filename: {inspect.getsourcefile(base_obj)}, module: {inspect.getmodule(base_obj)}")
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

    link = get_link(fqn, base_obj)

    # We want the final HTML to be like:
    # Bases: <a class="reference internal" href="tlo.core.html#tlo.core.Module"
    #     title="tlo.core.Module">
    #     <code class="xref py py-class docutils literal notranslate">
    #     <span class="pre">tlo.core.Module</span></code></a></p>
    # So we need to write the right stuff to the .rst file.
    # We want to return a string with the name of the base class, plus the link
    # to its documentation.
    # e.g. the string might be "tlo.core.Module <./tlo.core.html#tlo.core.Module>"
    mystr = "`" + fqn + " " + link + "`_"
    # print(f"MYSTR = {mystr}")
    # e.g. MYSTR = `tlo.core.Module <./tlo.core.html#tlo.core.Module>_`
    return mystr


def get_link(base_fqn, base_obj):
    '''Given a name like Mockitis and a base_string like
    tlo.core.Module, create a link to the latter's doc page.
    :param base_fqn: fully-qualified name of the base class, e.g. tlo.core.Module
    :param base_obj: the object representing the base class
    :return: a string with the link to the base class's place in the generated documentation.
    '''
    #file_defining_class = inspect.getsourcefile(base_obj)  # Path to file e.g. /Users/....src/tlo/methods/mockitis.py
    module_defining_class = str(inspect.getmodule(base_obj))  # <module 'tlo.core' from '/Users/...
    module_pieces = module_defining_class.split(" ")
    if len(module_pieces) > 1:
        module_name = module_pieces[1]  # e.g. 'tlo.core'
    else:
        module_name = module_defining_class
    module_name = module_name.replace("'", "")  # Remove single speech marks
    print(f"My lovely module is {module_name}")  # e.g. tlo.core

    # Need name of base's file + fqn of base
    # We want a link string like: <./tlo.core.html#tlo.core.Module>
    # i.e. <base's file name#base's fqn>

    # e.g. if this page is http://0.0.0.0:8000/reference/tlo.methods.mockitis.html
    # here is a link to another class: `newborn_outcomes <./tlo.methods.newborn_outcomes.html>`_
    # http://0.0.0.0:8000/reference/tlo.methods.newborn_outcomes.html
    # or e.g. pointing to http://0.0.0.0:8000/reference/tlo.core.html#tlo.core.Module
    # the link would be <./tlo.core.html#tlo.core.Module>
    link_to_base = "<./" + module_name + ".html" + "#" + base_fqn + ">"
    print(f"link_to_base = {link_to_base}")
    return link_to_base


def create_table(mydict):
    '''
    Dynamically create a table of arbitrary length
    from PROPERTIES and PARAMETERS dictionaries.
    `mydict` is the dictionary object.

    NB Do not change the positioning of items in the
    f-strings below, or things will break!

    The table is returned as a list of strings.
    If there is no data, an empty list is returned.
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
        return []
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



if __name__ == '__main__':

    # From above:
    # root_dir = Path(__file__).resolve().parents[2]
    # MODULE_DIR = f"{root_dir}/src/tlo/methods"   #./src/tlo/methods"
    # LEADER = "tlo.methods"
    # RST_DIR = f"{root_dir}/docs/reference"

    # Add command-line processing here
    context = LEADER
    module_directory = MODULE_DIR
    rst_directory = RST_DIR

    modules = generate_module_list(module_directory)  # List of .py files
    print (modules)
    for m in modules:  # e.g. mockitis.py
        #if m != "mockitis.py":
        #    continue
        fqn = get_fully_qualified_name(m, context)  # e.g. "tlo.methods.mockitis"
        module_obj = importlib.import_module(fqn)  # Object creation from string.
        print (f"module_obj is {module_obj}")
        write_rst_file(rst_directory, fqn, module_obj)



    # From Stef's sphinx_debug.py
    #root_dir = Path(__file__).resolve().parents[2]
    #print(f"root-dir is {root_dir}")
