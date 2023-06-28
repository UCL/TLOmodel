# Profiling with `pyinstrument`

Activate your developer environment, and navigate to the root of the TLOModel repository.
Run
```sh
python profiling/profile.py HMTL_OUTPUT_LOCATION
```
to run the profiling script (currently only supports `scale_run.py`).
You can also request command-line help using the `-h` or `--help` flags.
If you do not provide the `HTML_OUTPUT_LOCATION`, the script will write the outputs to the default location (`profiling/html`).

## Files within `profiling/`

- `parameters.py`: Parameters for each of the models that the profiler should run, stored as dictionaries.
- `profile.py`: Main profiling script; runs all models that need to be profiled and outputs results.
- `shared.py`: Logging and other processes that are shared across multiple files.

The following files contain models which are run by the profiler:
- `scale_run.py`: A run of the full model at scale using all disease modules considered complete and all
modules for birth / labour / newborn outcome.