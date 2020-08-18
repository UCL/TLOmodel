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

MODULE_DIR = "./src/tlo/methods"
LEADER = "tlo.methods"

# Use this so we can dynamically import, and not need to hard-code it.
exec("from tlo.methods import mockitis")


def generate_module_list(dir_string):
    '''Obtain all Python module files in this directory
    e.g. mockitis.py, hiv.py, healthsystem.py...
    in an alphabetically-sorted list.

    :param dir_string: string containing path to directory to process.

    For each directory in the tree, walk yields a 3-tuple
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
    the required module. Note that this excludes
    any other classes in the module.
    Each entry in list returned is itself a list of:
    class name, class object, line number

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
    # Answer by Andrew
    classes.sort(key = lambda x: x[2])
    print(f"after sorting, {classes}")
    return classes


def get_fully_qualified_name(filename):
    '''
    Given a file name (e.g. mockitis.py)
    return the fully-qualified name of the module
    e.g. tlo.methods.mockitis
    This gets used in doc page title.

    :param filename:
    :return:
    '''
    parts = filename.split(".")
    fqname = LEADER + "." + parts[0]
    return fqname


def extract_required_members(module, exclusions):
    '''Which class members do we wish to include in
    the .rst file we will write?

    Skip over the ones we don't want.'''
    pass


def write_rst_file(filename):
    '''We want to write one .rst file per module in ./src/tlo/methods'''
    pass


if __name__ == '__main__':

    # This will print out all classes in mockitis.py
    # including those imported by mockitis.py
    # stuff = inspect.getmembers(mockitis)
    #for name, obj in stuff:
    #    if inspect.isclass(obj):
    #        print(name)
    #        print (obj)

    # Just obtain mockitis.py's classes:
    # Is this robust enough?
    #print ("Classes defined in mockitis.py only:")
    #leader = LEADER + "mockitis"
    #fqdn = get_fully_qualified_name("mockitis.py")
    #stuff = inspect.getmembers(tlo.methods.mockitis)
    #for name, obj in stuff:
    #    if fqdn in str(obj) and inspect.isclass(obj):
    #        print(name)  # e.g. MockitisEvent
    #        print (obj)  # e.g. <class 'tlo.methods.mockitis.MockitisEvent'>
            #print(issubclass(obj, Module))
            #classes.append(obj)

    modules = generate_module_list(MODULE_DIR)  # List of .py files
    print (modules)
    for m in modules:  # e.g. mockitis.py
        #classes_in_module = []
        if m != "mockitis.py":
            continue
        fqn = get_fully_qualified_name(m)  # e.g. "tlo.methods.mockitis"
        module_obj = importlib.import_module(fqn)  # Object creation from string.
        print (f"module_obj is {module_obj}")
        classes_in_this_module = get_classes_in_module(fqn, module_obj)

        # Get the classes defined in this module
        #stuff = inspect.getmembers(module_obj)
        #for name, obj in stuff:
        #   if fqn in str(obj) and inspect.isclass(obj):
        #       print(name)  # e.g. MockitisEvent
        #       print(obj)  # e.g. <class 'tlo.methods.mockitis.MockitisEvent'>
        #       #classes_in_module.append(obj)
        #       print(f"\n\nIn module {name} we find the following:")
        #       morestuff = inspect.getmembers(obj)
        #       print(morestuff)  # e.g. functions, PARAMETERS dict,...

