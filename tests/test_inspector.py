import pytest
import inspect
import importlib

import tlo.inspector as inspector

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


def test_get_package_name():
    pass


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
    # TODO: this is a bit broken at the moment.
    classes = get_classes_for_testing()
    # Each entry in the list returned is itself a list of:
    # [class name, class object, line number]
    c5 = classes[-1]
    name, obj = c5[0:2]
    expected = "**Bases:**\n\n"
    #expected += f"Base #1: Father {fqn}\n\n"
    expected += "Base #2: Mother\n\n"
    ##assert expected == inspector.extract_bases(name, obj)
    #base_string_1 = inspector.get_base_string(name, obj)
    # Typical example of bases:
    # (<class 'tlo.methods.mockitis.Mockitis'>, <class 'tlo.core.Module'>, <class 'object'>)
    #assert

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
