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
    in an alphabetically-sorted list.

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
               # case when tlo.methods has submodules (i.e. subdirectory),
               # within which the individual disease modules are located
               # e.g. we might have sub-level tlo.methods.rmnch for modules
               # dealing with reproduction, birth, child health, etc.

    return sorted(filenames)


def get_classes_in_module(fqn, module_obj):
    '''
    Generate a list of lists of the classes *defined* in
    the required module, in the order in which they
    appear in the module file. Note that this excludes
    any other classes in the module.

    Each entry in the list returned is itself a list of:
    [class name, class object, line number]

    :param fqn: Fully-qualified name of the module,
    e.g. "tlo.methods.mockitis"

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
    # Based on answer by Andrew
    classes.sort(key=lambda x: x[2])
    print(f"after sorting, {classes}")
    return classes


def get_fully_qualified_name(filename, context):
    '''
    Given a file name and a context
    return the fully-qualified name of the module

    e.g. tlo.methods.mockitis
    This gets used in doc page title.

    :param filename: name of file, e.g. "mockitis.py"
    :param context: "location" of module, e.g. "tlo.methods"
    :return: a string, e.g. "tlo.methods.mockitis"
    '''
    parts = filename.split(".")
    fqname = context + "." + parts[0]
    return fqname


def extract_required_members(module, exclusions):
    '''Which class members do we wish to include in
    the .rst file we will write?

    Skip over the ones we don't want.

    Don't want to display most class members in tlo.methods.*
    For classes which inherit tlo.Module, the PROPERTIES & PARAMETERS
    class dictionary attributes should be displayed only as tables.
    And also not display e.g. apply() or name, on_birth()
   '''
    pass


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

    TODO: stop unwanted output, e.g. certain methods in children of Module.

    '''
    class_name, class_obj, _ = classinfo
    str = f".. class:: {class_name}\n\n"

    # Now we want to add base classes, class members, comments and methods
    # Presumably in source file order.
    classdat = inspect.getmembers(class_obj)  # Gets everything

    bases = inspect.getmro(class_obj)  # Includes "object" class and the class_name class
    print(f"bases for {class_name}: {bases}")
    #classtree = inspect.getclasstree(bases)
    #print(f"classtree = {classtree}")

    if len(bases) > 0:
        str += f"Bases: {bases}\n\n"
        # TODO: Remove object class and the class itself, and tidy up format
        # and we will want links to be generated - is that automatic?

    general_exclusions = ["__class__", "__dict__", "__init__", "__module__",
                          "__slots__", "__weakref__",]
    inherited_exclusions = ["initialise_population", "initialise_simulation", "on_birth",
                            "read_parameters", "apply", "post_apply_hook",
                            "on_hsi_alert", "report_daly_values", "run", "SYMPTOMS",]

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
            str += f"**{name}:**\n"
            table_list = create_table(obj)
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


def create_table(mydict):
    '''
    Dynamically create a table of arbitrary length
    from PROPERTIES and PARAMETERS dictionaries.
    `mydict` is the dictionary object.

    NB Do not change the positioning of items in the
    f-strings below, or things will break!

    Lifted from conf.py
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



if __name__ == '__main__':

    # Add command-line processing here
    context = LEADER
    module_directory = MODULE_DIR
    rst_directory = RST_DIR

    modules = generate_module_list(module_directory)  # List of .py files
    print (modules)
    for m in modules:  # e.g. mockitis.py
        if m != "mockitis.py":
            continue
        fqn = get_fully_qualified_name(m, context)  # e.g. "tlo.methods.mockitis"
        module_obj = importlib.import_module(fqn)  # Object creation from string.
        print (f"module_obj is {module_obj}")
        write_rst_file(rst_directory, fqn, module_obj)



    # From Stef's sphinx_debug.py
    #root_dir = Path(__file__).resolve().parents[2]
    #print(f"root-dir is {root_dir}")

  # inspect.getcomments(): Return in a single string any lines of
    # comments immediately preceding the object’s source code
    # (for a class, function, or method), or at the top of
    # the Python source file (if the object is a module).
    #comments = inspect.getcomments(module_obj)
    #print(f"comments in module = {comments}")
    # This doesn't seem to work
