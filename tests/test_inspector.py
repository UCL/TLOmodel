#import os
#import time
#from pathlib import Path

import pytest
import inspect
import importlib

import tlo.inspector as inspector

@pytest.mark.parametrize(
    "path, result",
    [
        ("./resources", ["ResourceFile_load-parameters.xlsx",
                         "df_at_healthcareseeking.csv",
                         "df_at_init_of_lifestyle.csv",
                         "example_log.txt",
                         ]),
        # ("another_path", [sorted file list]),
        ("./for_inspector", ["a.py"]),
    ],
)
def test_generate_module_list(path, result):
    # Expect result sorted in ASCII order
    assert result == inspector.generate_module_list(path)


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


def test_get_classes_in_module():
    # TODO: move common test code out
    fqn = "for_inspector.a"
    module_obj = importlib.import_module(fqn)
    classes = inspector.get_classes_in_module(fqn, module_obj)
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
    # TODO: move common test code out
    fqn = "for_inspector.a"
    module_obj = importlib.import_module(fqn)
    classes = inspector.get_classes_in_module(fqn, module_obj)
    # Each entry in the list returned is itself a list of:
    # [class name, class object, line number]
    c5 = classes[-1]
    name, obj = c5[0:2]
    expected = "**Bases:**\n\n"
    expected += "Base #1: Father\n\n"
    expected += "Base #2: Mother\n\n"
    #assert expected == inspector.extract_bases(name, obj)
    base_string_1 = inspector.get_base_string(name, obj)
    # Typical example of bases:
    # (<class 'tlo.methods.mockitis.Mockitis'>, <class 'tlo.core.Module'>, <class 'object'>)
    #assert

def test_write_rst_file():
    pass

def test_get_class_output_string():
    pass

def test_get_base_string():
    pass

def test_get_link():
    pass

def test_create_table():
    pass
