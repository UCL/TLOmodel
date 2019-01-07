import os

from tlo.analysis.utils import parse_output

example_log = 'example_log.txt'


def test_parse_log():
    log_file = os.path.join(os.path.dirname(__file__),
                            'resources',
                            example_log)

    p = parse_output(log_file)

    assert len(p) == 1
    assert 'tlo.methods.demography' in p
    assert set(p['tlo.methods.demography'].keys()) == {'death', 'population', 'age_range_m',
                                                       'age_range_f', 'on_birth'}

    return p
