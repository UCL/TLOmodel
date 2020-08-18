# Best to run this from within PyCharm configuration
# with working directory set to (e.g.)
# /Users/matthewgillman/PycharmProjects/TLOmodel
# and Script path
# /Users/matthewgillman/PycharmProjects/TLOmodel/src/tlo/inspector.py
import inspect
from os import walk

MODULE_DIR = "./src/tlo/methods"

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
        break  # In this version we don't traverse any subdirectories

    return sorted(filenames)


def extract_required_members(module, exclusions):
    '''Which class members do we wish to include in
    the .rst file we will write?

    Skip over the ones we don't want.'''
    pass

def write_rst_file(filename):
    pass

if __name__ == '__main__':

    # This will print out all classes in mockitis.py
    # including those imported by mockitis.py
    stuff = inspect.getmembers(mockitis)
    for name, obj in stuff:
        if inspect.isclass(obj):
            print(name)
            print (obj)

    # Just obtain mockitis.py's classes:
    # Is this robust enough?
    print ("Classes defined in mockitis.py only:")
    leader = "tlo.methods.mockitis"
    for name, obj in stuff:
        if leader in str(obj) and inspect.isclass(obj):
            print (obj)  # e.g. <class 'tlo.methods.mockitis.MockitisEvent'>

    modules = generate_module_list(MODULE_DIR)
    print (modules)
