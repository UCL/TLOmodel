import importlib
import sys

import pytest

sys.path.insert(0, '.')
sys.path.insert(0, './tlo_methods_rst')
sys.path.insert(0, '../src') #/tlo')
#sys.path.insert(0, './tests/')
#from tlo.docs import generate_module_dict, get_package_name, \
#    get_fully_qualified_name, write_rst_file, get_classes_in_module
from tlo.docs import *


def test_generate_module_dict():
    # Gets a dictionary of files in directory tree with
    # key = path to dir, value = list of .py files
    result = generate_module_dict("./tlo_methods_rst/tlo/")
    for dir in result:
        files = result[dir]
        if "/tests/tlo_methods_rst/tlo/more" in dir:
            assert files == ['c.py']
        else:
            if "/tests/tlo_methods_rst/tlo" in dir:
                assert sorted(files) == ['a.py', 'b.py']


@pytest.mark.parametrize(
    "filename, context, result",
    [
        ("fred.py", "some.place.or.other", "some.place.or.other.fred"),
        ("daniel", "somewhere.else", "somewhere.else.daniel"),
        ("roberta", "", "roberta"),
        ("", "", ""),
    ]
)
def test_get_fully_qualified_name(filename, context, result):
    # Get the fully-qualified name of the module (file).
    assert result == get_fully_qualified_name(filename, context)


@pytest.mark.parametrize(
    "dirpath, result",
    [
        ("some/path/or/other/tlo/", "tlo"),
        ("./src/tlo/logging/sublog", "tlo.logging.sublog"),
    ]
)
def test_get_package_name_no_exceptions(dirpath, result):
    assert result == get_package_name(dirpath)


@pytest.mark.parametrize(
    "dirpath",
    [
        ("some/path/or/other/tlo"),
        ("/a/b/c"),
        ("/tlo"),
        ("tlo"),
    ]
)
def test_get_package_name_with_exceptions(dirpath):
    with pytest.raises(ValueError) as e:
        get_package_name(dirpath)
    assert f"Sorry, /tlo/ isn't in dirpath ({dirpath})" == str(e.value)


def get_classes_for_testing():
    '''Utility function.
     Each entry in the list returned is itself a list of:
     [class name, class object, line number]
     The classes are those defined in file for_tlo_methods_rst/tlo/a.py,
     in the same order as they are defined in the source file.
    '''
    fqn = "tlo_methods_rst.tlo.a"
    module_obj = importlib.import_module(fqn)
    return get_classes_in_module(fqn, module_obj)


def test_get_classes_in_module():
    classes = get_classes_for_testing()
    assert len(classes) == 5
    c1, c2, c3, c4, c5 = classes
    assert c1[0] == "Person"
    assert c2[0] == "Employee"
    assert str(c1[1]) == "<class 'tlo_methods_rst.tlo.a.Person'>"
    assert str(c2[1]) == "<class 'tlo_methods_rst.tlo.a.Employee'>"
    assert str(c5[0]) == "Offspring"


def test_extract_bases():
    classes = get_classes_for_testing()
    # Each entry in the list returned is itself a list of:
    # [class name, class object, line number]
    offspring = classes[-1]
    name, obj = offspring[0:2]
    expected = "**Base classes:**\n\n"
    expected += ("Base class #1: `tlo_methods_rst.tlo.a.Father "
                 "<./tlo_methods_rst.tlo.a.html"
                 "#tlo_methods_rst.tlo.a.Father>`_\n\n")
    expected += ("Base class #2: `tlo_methods_rst.tlo.a.Mother "
                 "<./tlo_methods_rst.tlo.a.html"
                 "#tlo_methods_rst.tlo.a.Mother>`_\n\n")
    assert expected == extract_bases(name, obj)


def ignore_this_test_write_rst_file():
    module_directory = "./tlo_methods_rst/tlo/"
    rst_directory = "./tlo_methods_rst/tlo/docs"

    # Need the trailing slash after tlo - it needs "/tlo/":
    # mydata = generate_module_dict("./src/tlo/")
    mydata = get_class_output_string(module_directory)
    for dir in mydata:  # e.g. .../src/tlo/logging/sublog
        package = get_package_name(dir)  # e.g. "tlo.logging.sublog"
        files = mydata[dir]  # e.g. ["fileA.py", "fileB.py", ...]
        print(f"In directory [{dir}]: files are {files}")
        for f in files:
            # e.g. "tlo.logging.sublog.fileA":
            fqn = get_fully_qualified_name(f, package)
            # print(f"DEBUG: dir: {dir}, package:{package}, f:{f}, fqn:{fqn}")
            # Object creation from string:
            module_obj = importlib.import_module(fqn)
            # print(f"module_obj is {module_obj}")
            tmr.write_rst_file(rst_directory, fqn, module_obj)


def test_get_class_output_string():
    classes = get_classes_for_testing()
    person = classes[0]
    result = get_class_output_string(person)
    expected = "\n\n\n.. autoclass:: Person\n\n"
    expected += "\n\n"  # It has no bases to extract.
    numspaces = 5
    spacer = numspaces * ' '

    # NB we expect the following to be in name order
    # i.e. alphabetical rather than source code order.
    expected += f"{spacer}.. automethod:: __init__\n\n"
    expected += f"{spacer}.. automethod:: get_name\n\n"
    expected += f"{spacer}.. automethod:: set_name\n\n"

    expected += "\n\n\n"
    assert result == expected


def test_get_base_string():
    classes = get_classes_for_testing()
    # Each item in classes list has format: [class name, class object, line number]
    # In this test case, Person is the base class of Employee.
    person_info = classes[0]
    # person_name = person_info[0]
    person_object = person_info[1]
    employee_info = classes[1]
    employee_name = employee_info[0]
    employee_object = employee_info[1]
    result = get_base_string(employee_name, employee_object, person_object)
    expected = "`tlo_methods_rst.tlo.a.Person <./tlo_methods_rst.tlo.a.html#tlo_methods_rst.tlo.a.Person>`_"
    assert result == expected


def test_get_link():
    # Example link:
    # <./tlo.core.html#tlo.core.Module>
    classes = get_classes_for_testing()
    base_class = classes[0]
    base_fqn = "tlo_methods_rst.tlo.a.Person"
    base_obj = base_class[1]
    result = get_link(base_fqn, base_obj)
    expected = f"<./tlo_methods_rst.tlo.a.html#{base_fqn}>"
    assert result == expected


def test_create_table():
    pass
