from pathlib import Path

from tlo.analysis.utils import parse_log_file


def test_parse_log():
    log_file = Path(__file__).parent / 'resources' / 'structured_log.txt'

    output = parse_log_file(log_file)

    assert 'tlo.methods.epilepsy' in output
    assert set(output['tlo.methods.epilepsy'].keys()) == {'incidence_epilepsy', 'epilepsy_logging'}
