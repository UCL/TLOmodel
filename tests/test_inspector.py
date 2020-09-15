import pytest
import inspect
import importlib

import docs.inspector as inspector

def test_generate_module_dict():
    # Gets a dictionary of files in directory tree with
    # key = path to dir, value = list of .py files
    result = inspector.generate_module_dict("./for_inspector")
    for dir in result:
        files = result[dir]
        if "/tests/for_inspector/more" in dir:
            assert files == ['c.py']
        else:
            if "/tests/for_inspector" in dir:
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
    fqn = "for_inspector.a"
    module_obj = importlib.import_module(fqn)
    return inspector.get_classes_in_module(fqn, module_obj)

def test_get_classes_in_module():
    classes = get_classes_for_testing()
    # Each entry in the list returned is itself a list of:
    # [class name, class object, line number]
    assert len(classes) == 5
    c1, c2, c3, c4, c5 = classes
    assert c1[0] == "Person"
    assert c2[0] == "Employee"
    assert str(c1[1]) == "<class 'for_inspector.a.Person'>"
    assert str(c2[1]) == "<class 'for_inspector.a.Employee'>"
    assert str(c5[0]) == "Offspring"


def test_extract_bases():
    classes = get_classes_for_testing()
    # Each entry in the list returned is itself a list of:
    # [class name, class object, line number]
    offspring = classes[-1]
    name, obj = offspring[0:2]
    expected = "**Bases:**\n\n"
    expected += f"Base #1: `for_inspector.a.Father <./for_inspector.a.html#for_inspector.a.Father>`_\n\n"
    expected += f"Base #2: `for_inspector.a.Mother <./for_inspector.a.html#for_inspector.a.Mother>`_\n\n"
    assert expected == inspector.extract_bases(name, obj)


def test_write_rst_file():
    pass


def test_get_class_output_string():
    pass


def test_get_base_string():
    #modules = generate_module_list("./for_inspector")  # List of .py files
    #assert len(modules) = 1
    #m = modules[0]
    #fqn = "for_inspector/a"
    #assert m == m[]
    #for m in modules:  # e.g. mockitis.py
        #fqn = get_fully_qualified_name(m, context)  # e.g. "tlo.methods.mockitis"
        #module_obj = importlib.import_module(fqn)  # Object creation from string.
        #print(f"module_obj is {module_obj}")
        #write_rst_file(rst_directory, fqn, module_obj)
    pass


def test_get_link():
    pass


def test_create_table():
    pass
