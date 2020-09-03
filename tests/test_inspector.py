import os
import time
import inspect  # not sure if required.
from pathlib import Path

import pytest

from tlo import Date, Simulation
import tlo.inspector as inspector
from tlo.methods import (
    antenatal_care,
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    labour,
    newborn_outcomes,
    pregnancy_supervisor,
)

def test_something():
    assert (1 == 1)


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
    assert (result == inspector.generate_module_list(path))


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
    assert (result == inspector.get_fully_qualified_name(filename, context))
