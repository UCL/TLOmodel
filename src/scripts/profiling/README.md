# Profiling with `pyinstrument`

Activate your developer environment, and navigate to the root of the TLOModel repository.
Run
```sh
python src/scripts/profiling/profile.py HMTL_OUTPUT_LOCATION
```
to run the profiling script (currently only supports `scale_run.py`).
You can also request command-line help using the `-h` or `--help` flags.
If you do not provide the `HTML_OUTPUT_LOCATION`, the script will write the outputs to the default location (`profiling/html`).

## Files within `profiling/`

Utility files:
- `_paths.py`: Defines some absolute paths to ensure that the profiler writes outputs to the correct locations and the script is robust against being run in different working directories.
- `shared.py`: Logging and other processes that are shared across multiple files.

Files that are used to wrap the automatic profiling run:
- `parameters.py`: Parameters for each of the models that the profiler should run, stored as dictionaries.
- `profile.py`: Main profiling script; runs all models that need to be profiled and outputs results.

Models which are run by the profiler:
- `scale_run.py`: A run of the full model at scale using all disease modules considered complete and all
modules for birth / labour / newborn outcome.

Models which are not presently used by the profiler, but can be run locally:
- `batch_test.py`
- `heavy_use_of_bed_days.py`
- `heavy_use_of_spurious_symptoms.py`
- `run_full_model_with_hard_constraints_in_healthsystem.py`
- `run_with_high_intensity_of_HSI_and_simplified_births.py`