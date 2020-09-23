import pytest
import inspect
import importlib

import docs.inspector as inspector

def test_generate_module_dict():
    # Gets a dictionary of files in directory tree with
    # key = path to dir, value = list of .py files
    result = inspector.generate_module_dict("./for_inspector/tlo/")
    for dir in result:
        files = result[dir]
        if "/tests/for_inspector/tlo/more" in dir:
            assert files == ['c.py']
        else:
            if "/tests/for_inspector/tlo" in dir:
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
    assert result == inspector.get_fully_qualified_name(filename, context)


@pytest.mark.parametrize(
    "dirpath, result",
    [
        ("some/path/or/other/tlo/", "tlo"),
        ("./src/tlo/logging/sublog", "tlo.logging.sublog"),
    ]
)
def test_get_package_name_no_exceptions(dirpath, result):
    assert result == inspector.get_package_name(dirpath)


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
        inspector.get_package_name(dirpath)
    assert f"Sorry, /tlo/ isn't in dirpath ({dirpath})" == str(e.value)


def get_classes_for_testing():
    '''Utility function.
     Each entry in the list returned is itself a list of:
     [class name, class object, line number]
     The classes are those defined in file for_inspector/a.py,
     in the same order as they are defined in the source file.
    '''
    fqn = "for_inspector.tlo.a"
    module_obj = importlib.import_module(fqn)
    return inspector.get_classes_in_module(fqn, module_obj)


def test_get_classes_in_module():
    classes = get_classes_for_testing()
    assert len(classes) == 5
    c1, c2, c3, c4, c5 = classes
    assert c1[0] == "Person"
    assert c2[0] == "Employee"
    assert str(c1[1]) == "<class 'for_inspector.tlo.a.Person'>"
    assert str(c2[1]) == "<class 'for_inspector.tlo.a.Employee'>"
    assert str(c5[0]) == "Offspring"


def test_extract_bases():
    classes = get_classes_for_testing()
    # Each entry in the list returned is itself a list of:
    # [class name, class object, line number]
    offspring = classes[-1]
    name, obj = offspring[0:2]
    expected = "**Bases:**\n\n"
    expected += f"Base #1: `for_inspector.tlo.a.Father <./for_inspector.tlo.a.html#for_inspector.tlo.a.Father>`_\n\n"
    expected += f"Base #2: `for_inspector.tlo.a.Mother <./for_inspector.tlo.a.html#for_inspector.tlo.a.Mother>`_\n\n"
    assert expected == inspector.extract_bases(name, obj)


def ignore_this_test_write_rst_file():
    module_directory = "./for_inspector/tlo/"
    rst_directory = "./for_inspector/tlo/docs"

    # Need the trailing slash after tlo - it needs "/tlo/":
    # mydata = generate_module_dict("./src/tlo/")
    mydata = inspector.get_class_output_string(module_directory)
    for dir in mydata:  # e.g. .../src/tlo/logging/sublog
        package = inspector.get_package_name(dir)  # e.g. "tlo.logging.sublog"
        files = mydata[dir]  # e.g. ["fileA.py", "fileB.py", ...]
        print(f"In directory [{dir}]: files are {files}")
        for f in files:
            # e.g. "tlo.logging.sublog.fileA":
            fqn = inspector.get_fully_qualified_name(f, package)
            # print(f"DEBUG: dir: {dir}, package:{package}, f:{f}, fqn:{fqn}")
            # Object creation from string:
            module_obj = importlib.import_module(fqn)
            # print(f"module_obj is {module_obj}")
            write_rst_file(rst_directory, fqn, module_obj)


def test_get_class_output_string():
    classes = get_classes_for_testing()
    person = classes[0]
    result = inspector.get_class_output_string(person)
    expected = "\n\n\n.. class:: Person\n\n"
    expected += "\n\n"
    expected += "**Description:**\nThe basic Person class."  # <class for_inspector.tlo.a.Person>"
    expected += "\n\n**Function get_name():**\n"
    expected += f"\n\nGet the name. \n\n"
    expected += "\n\n**Function set_name():**\n"
    expected += f"\n\nSet the name. \n\n"
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
    result = inspector.get_base_string(employee_name, employee_object, person_object)
    expected = "`for_inspector.tlo.a.Person <./for_inspector.tlo.a.html#for_inspector.tlo.a.Person>`_"
    assert result == expected


def test_get_link():
    # Example link:
    # <./tlo.core.html#tlo.core.Module>
    classes = get_classes_for_testing()
    base_class = classes[0]
    base_fqn = "for_inspector.tlo.a.Person"
    base_obj = base_class[1]
    result = inspector.get_link(base_fqn, base_obj)
    expected = f"<./for_inspector.tlo.a.html#{base_fqn}>"
    assert result == expected


def test_create_table():
    pass
