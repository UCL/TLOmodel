import sys
from pathlib import Path


def process(output_dir, scenario_runs_dir, resources_dir):
    from importlib import import_module

    function = getattr(import_module('analysis_all_calibration'), 'apply')
    function(scenario_runs_dir, output_dir, resources_dir)


if __name__ == '__main__':
    _output_dir_for_processed = Path(sys.argv[1])
    _run_directory_name = sys.argv[2]
    _resources_dir = Path(sys.argv[3])
    process(_output_dir_for_processed,
            _output_dir_for_processed.parent / _run_directory_name,
            _resources_dir)
