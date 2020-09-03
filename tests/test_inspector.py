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
    fqn = "for_inspector.a"
    module_obj = importlib.import_module(fqn)
    classes = inspector.get_classes_in_module(fqn, module_obj)
    # Each entry in the list returned is itself a list of:
    # [class name, class object, line number]
    c1, c2 = classes
    assert c1[0] == "Person"
    assert c2[0] == "Employee"
    assert str(c1[1]) == "<class 'for_inspector.a.Person'>"
    assert str(c2[1]) == "<class 'for_inspector.a.Employee'>"
