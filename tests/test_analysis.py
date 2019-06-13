import os
from pathlib import Path

from tlo.analysis.utils import parse_log_file

example_log = 'example_log.txt'


def test_parse_log():
    log_file = Path(os.path.dirname(__file__)) / 'resources' / example_log

    p = parse_log_file(log_file)

    assert len(p) == 1
    assert 'tlo.methods.demography' in p
    assert set(p['tlo.methods.demography'].keys()) == {'death', 'population', 'age_range_m',
                                                       'age_range_f', 'on_birth'}
