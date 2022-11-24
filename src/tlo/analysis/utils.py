"""
General utility functions for TLO analysis
"""
import gzip
import json
import os
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, TextIO, Union

import git
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import squarify

from tlo import logging, util
from tlo.logging.reader import LogData
from tlo.util import create_age_range_lookup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _parse_log_file_inner_loop(filepath, level: int = logging.INFO):
    """Parses the log file and returns dictionary of dataframes"""
    log_data = LogData()
    with gzip.open(filepath, 'rt') as log_file:
        for line in log_file:
            # only parse json entities
            if line.startswith('{'):
                log_data.parse_log_line(line, level)
            else:
                print('FAILURE: found old-style log:')
                print(line)
                raise RuntimeError
    # convert dictionaries to dataframes
    output_logs = {**log_data.get_log_dataframes()}
    return output_logs


def parse_log_file(log_filepath, keep_existing=False):
    """Parses logged output from a TLO run, split it into smaller logfiles and returns a dict-like to access those log
    files. Can handle both gzipped and uncompressed log files. Looks for a gzip file first.

    :param log_filepath: file path to log file
    :param keep_existing: keep any existing module-specific log files and pickled dataframes
    :return: a class containing paths to split logfiles
    """
    print(f'Processing log file {log_filepath}')
    log_filepath = Path(log_filepath)
    uuid_to_module_name: Dict[str, str] = dict()  # uuid to module name
    module_name_to_filehandle: Dict[str, TextIO] = dict()  # module name to file handle

    log_directory = log_filepath.parent
    print(f'Writing module-specific log files to {log_directory}')

    # We've been given a file path, which may or may not be a path to a gzipped file (scenarios produce gzipped log
    # files). Check for both and open the right sort of file handle.
    zipped_log_filepath = log_directory / (log_filepath.name + '.gz')
    if log_filepath.suffix == '.gz':
        log_file = gzip.open(log_filepath, 'rt')
    elif os.path.exists(zipped_log_filepath):
        log_file = gzip.open(zipped_log_filepath, 'rt')
    else:
        log_file = open(log_filepath, 'r')

    module_name_to_filename = {}

    # iterate over each line in the logfile
    for line in log_file:
        # only parse lines that are json log lines (old-style logging is not supported)
        if line.startswith('{'):
            log_data_json = json.loads(line)
            uuid = log_data_json['uuid']
            # if this is a header line (only header lines have a `type` key)
            if 'type' in log_data_json:
                module_name = log_data_json["module"]
                uuid_to_module_name[uuid] = module_name
                # we only need to create the file if we don't already have one for this module
                # and we only need to write the lines if we haven't already got a pickled version
                pickle_file_name = str(log_directory / f"{module_name}.pickle")
                if keep_existing and os.path.exists(pickle_file_name):
                    module_name_to_filename[module_name] = pickle_file_name
                else:
                    # the pickle file for this module doesn't exist. save the log lines to module-specific log file
                    if module_name not in module_name_to_filehandle:
                        module_name_to_filehandle[module_name] = gzip.open(
                            log_directory / f"{module_name}.log.gz", mode="wt"
                        )
                        module_name_to_filename[module_name] = module_name_to_filehandle[module_name].name
            # if we need to save the output of these log line (i.e. we don't have the pickled file)
            if uuid_to_module_name[uuid] in module_name_to_filehandle:
                # copy line from log file to module-specific log file (both headers and non-header lines)
                module_name_to_filehandle[uuid_to_module_name[uuid]].write(line)

    # close all module-specific files
    for file_handle in module_name_to_filehandle.values():
        file_handle.close()

    log_file.close()

    print('Finished writing module-specific log files.')

    # return an object that accepts as an argument a dictionary containing paths to split logfiles
    return LogsDict(module_name_to_filename)


def write_log_to_excel(filename, log_dataframes):
    """Takes the output of parse_log_file() and creates an Excel file from dataframes"""
    metadata = list()
    sheet_count = 0
    for module, dataframes in log_dataframes.items():
        for key, dataframe in dataframes.items():
            if key != '_metadata':
                sheet_count += 1
                metadata.append([module, key, sheet_count, dataframes['_metadata'][module][key]['description']])

    writer = pd.ExcelWriter(filename)
    index = pd.DataFrame(data=metadata, columns=['module', 'key', 'sheet', 'description'])
    index.to_excel(writer, sheet_name='Index')

    sheet_count = 0
    for module, dataframes in log_dataframes.items():
        for key, df in dataframes.items():
            if key != '_metadata':
                sheet_count += 1
                df.to_excel(writer, sheet_name=f'Sheet {sheet_count}')
    writer.save()


def make_calendar_period_lookup():
    """Returns a dictionary mapping calendar year (in years) to five year period
    i.e. { 1950: '1950-1954', 1951: '1950-1954, ...}
    """

    # Recycles the code used to make age-range lookups:
    ranges, lookup = util.create_age_range_lookup(1950, 2100, 5)

    # Removes the '0-1950' category
    ranges.remove('0-1950')

    for year in range(1950):
        lookup.pop(year)

    return ranges, lookup


def make_calendar_period_type():
    """
    Make an ordered categorical type for calendar periods
    Returns CategoricalDType
    """
    keys, _ = make_calendar_period_lookup()
    return pd.CategoricalDtype(categories=keys, ordered=True)


def make_age_grp_lookup():
    """Returns a dictionary mapping age (in years) to five year period
    i.e. { 0: '0-4', 1: '0-4', ..., 119: '100+', 120: '100+' }
    """
    return create_age_range_lookup(min_age=0, max_age=100, range_size=5)


def make_age_grp_types():
    """
    Make an ordered categorical type for age-groups
    Returns CategoricalDType
    """
    keys, _ = create_age_range_lookup(min_age=0, max_age=100, range_size=5)
    return pd.CategoricalDtype(categories=keys, ordered=True)


def to_age_group(_ages: pd.Series):
    """Return a pd.Series with age-group formatted as a categorical type, created from a pd.Series with exact age."""
    _, agegrplookup = make_age_grp_lookup()
    return _ages.map(agegrplookup).astype(make_age_grp_types())


def get_scenario_outputs(scenario_filename: str, outputs_dir: Path) -> list:
    """Returns paths of folders associated with a batch_file, in chronological order."""
    stub = scenario_filename.rstrip('.py')
    f: os.DirEntry
    folders = [Path(f.path) for f in os.scandir(outputs_dir) if f.is_dir() and f.name.startswith(stub)]
    folders.sort()
    return folders


def get_scenario_info(scenario_output_dir: Path) -> dict:
    """Utility function to get the the number draws and the number of runs in a batch set.

    TODO: read the JSON file to get further information
    """
    info = dict()
    f: os.DirEntry
    draw_folders = [f for f in os.scandir(scenario_output_dir) if f.is_dir()]

    info['number_of_draws'] = len(draw_folders)

    run_folders = [f for f in os.scandir(draw_folders[0]) if f.is_dir()]
    info['runs_per_draw'] = len(run_folders)

    return info


def load_pickled_dataframes(results_folder: Path, draw=0, run=0, name=None) -> dict:
    """Utility function to create a dict contaning all the logs from the specified run within a batch set"""
    folder = results_folder / str(draw) / str(run)
    p: os.DirEntry
    pickles = [p for p in os.scandir(folder) if p.name.endswith('.pickle')]
    if name is not None:
        pickles = [p for p in pickles if p.name in f"{name}.pickle"]

    output = dict()
    for p in pickles:
        name = os.path.splitext(p.name)[0]
        with open(p.path, "rb") as f:
            output[name] = pickle.load(f)

    return output


def extract_params(results_folder: Path) -> Optional[pd.DataFrame]:
    """Utility function to get overridden parameters from scenario runs

    Returns dateframe summarizing parameters that change across the draws. It produces a dataframe with index of draw
    and columns of each parameters that is specified to be varied in the batch. NB. This does the extraction from run 0
    in each draw, under the assumption that the over-written parameters are the same in each run.
    """

    try:
        f: os.DirEntry
        # Get the paths for the draws
        draws = [f for f in os.scandir(results_folder) if f.is_dir()]

        list_of_param_changes = list()

        for d in draws:
            p = load_pickled_dataframes(results_folder, d.name, 0, name="tlo.scenario")
            p = p["tlo.scenario"]["override_parameter"]

            p['module_param'] = p['module'] + ':' + p['name']
            p.index = [int(d.name)] * len(p.index)

            list_of_param_changes.append(p[['module_param', 'new_value']])

        params = pd.concat(list_of_param_changes)
        params.index.name = 'draw'
        params = params.rename(columns={'new_value': 'value'})
        params = params.sort_index()
        return params

    except KeyError:
        print("No parameters changed between the runs")
        return None


def extract_results(results_folder: Path,
                    module: str,
                    key: str,
                    column: str = None,
                    index: str = None,
                    custom_generate_series=None,
                    do_scaling: bool = False,
                    ) -> pd.DataFrame:
    """Utility function to unpack results.

    Produces a dataframe from extracting information from a log with the column multi-index for the draw/run.

    If the column to be extracted exists in the log, the name of the `column` is provided as `column`. If the resulting
     dataframe should be based on another column that exists in the log, this can be provided as 'index'.

    If instead, some work must be done to generate a new column from log, then a function can be provided to do this as
     `custom_generate_series`.

    Optionally, with `do_scaling=True`, each element is multiplied by the scaling_factor recorded in the simulation.

    Note that if runs in the batch have failed (such that logs have not been generated), these are dropped silently.
    """

    def get_multiplier(_draw, _run):
        """Helper function to get the multiplier from the simulation, if do_scaling=True.
        Note that if the scaling factor cannot be found a `KeyError` is thrown."""
        if not do_scaling:
            return 1.0
        else:
            return load_pickled_dataframes(results_folder, _draw, _run, 'tlo.methods.population'
                                           )['tlo.methods.population']['scaling_factor']['scaling_factor'].values[0]

    if custom_generate_series is None:
        # If there is no `custom_generate_series` provided, it implies that function required selects a the specified
        # column from the dataframe.
        assert column is not None, "Must specify which column to extract"

        if index is not None:
            _gen_series = lambda _df: _df.set_index(index)[column]  # noqa: 731
        else:
            _gen_series = lambda _df: _df.reset_index(drop=True)[column]  # noqa: 731

    else:
        assert index is None, "Cannot specify an index if using custom_generate_series"
        assert column is None, "Cannot specify a column if using custom_generate_series"
        _gen_series = custom_generate_series

    # get number of draws and numbers of runs
    info = get_scenario_info(results_folder)

    # Collect results from each draw/run
    res = dict()
    for draw in range(info['number_of_draws']):
        for run in range(info['runs_per_draw']):

            draw_run = (draw, run)

            try:
                df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]
                output_from_eval: pd.Series = _gen_series(df)
                assert pd.Series == type(output_from_eval), 'Custom command does not generate a pd.Series'
                res[draw_run] = output_from_eval * get_multiplier(draw, run)

            except KeyError:
                # Some logs could not be found - probably because this run failed.
                res[draw_run] = None

    # Use pd.concat to compile results (skips dict items where the values is None)
    _concat = pd.concat(res, axis=1)
    _concat.columns.names = ['draw', 'run']  # name the levels of the columns multi-index
    return _concat


def summarize(results: pd.DataFrame, only_mean: bool = False, collapse_columns: bool = False) -> pd.DataFrame:
    """Utility function to compute summary statistics

    Finds mean value and 95% interval across the runs for each draw.
    """

    summary = pd.DataFrame(
        columns=pd.MultiIndex.from_product(
            [
                results.columns.unique(level='draw'),
                ["mean", "lower", "upper"]
            ],
            names=['draw', 'stat']),
        index=results.index
    )

    summary.loc[:, (slice(None), "mean")] = results.groupby(axis=1, by='draw').mean().values
    summary.loc[:, (slice(None), "lower")] = results.groupby(axis=1, by='draw').quantile(0.025).values
    summary.loc[:, (slice(None), "upper")] = results.groupby(axis=1, by='draw').quantile(0.975).values

    if only_mean and (not collapse_columns):
        # Remove other metrics and simplify if 'only_mean' across runs for each draw is required:
        om: pd.DataFrame = summary.loc[:, (slice(None), "mean")]
        om.columns = [c[0] for c in om.columns.to_flat_index()]
        return om

    elif collapse_columns and (len(summary.columns.levels[0]) == 1):
        # With 'collapse_columns', if number of draws is 1, then collapse columns multi-index:
        summary_droppedlevel = summary.droplevel('draw', axis=1)
        if only_mean:
            return summary_droppedlevel['mean']
        else:
            return summary_droppedlevel

    else:
        return summary


def get_grid(params: pd.DataFrame, res: pd.Series):
    """Utility function to create the arrays needed to plot a heatmap.

    :param pd.DataFrame params: the dataframe of parameters with index=draw (made using `extract_params()`).
    :param pd.Series res: results of interest with index=draw (can be made using `extract_params()`)
    :returns: grid as dictionary
    """
    res = pd.concat([params.pivot(columns='module_param', values='value'), res], axis=1)
    piv = res.pivot_table(index=res.columns[0], columns=res.columns[1], values=res.columns[2])

    grid = dict()
    grid[res.columns[0]], grid[res.columns[1]] = np.meshgrid(piv.index, piv.columns)
    grid[res.columns[2]] = piv.values

    return grid


def format_gbd(gbd_df: pd.DataFrame):
    """Format GBD data to give standarize categories for age_group and period"""

    # Age-groups:
    gbd_df['Age_Grp'] = gbd_df['Age_Grp'].astype(make_age_grp_types())

    # label periods:
    calperiods, calperiodlookup = make_calendar_period_lookup()
    gbd_df['Period'] = gbd_df['Year'].map(calperiodlookup).astype(make_calendar_period_type())

    return gbd_df


def create_pickles_locally(scenario_output_dir, compressed_file_name_prefix=None):
    """For a run from the Batch system that has not resulted in the creation of the pickles, reconstruct the pickles
     locally."""

    def turn_log_into_pickles(logfile):
        print(f"Opening {logfile}")
        outputs = parse_log_file(logfile)
        for key, output in outputs.items():
            if key.startswith("tlo."):
                print(f" - Writing {key}.pickle")
                with open(logfile.parent / f"{key}.pickle", "wb") as f:
                    pickle.dump(output, f)

    def uncompress_and_save_logfile(compressed_file) -> Path:
        """Uncompress and save a log file and return its path."""
        target = compressed_file.parent / str(compressed_file.name[0:-3])
        with open(target, "wb") as t:
            with gzip.open(compressed_file, 'rb') as s:
                t.write(s.read())
        return target

    f: os.DirEntry
    draw_folders = [f for f in os.scandir(scenario_output_dir) if f.is_dir()]
    for draw_folder in draw_folders:
        run_folders = [f for f in os.scandir(draw_folder) if f.is_dir()]
        for run_folder in run_folders:
            # Find the original log-file written by the simulation
            if compressed_file_name_prefix is None:
                logfile = [x for x in os.listdir(run_folder) if x.endswith('.log')][0]
            else:
                compressed_file_name = [
                    x for x in os.listdir(run_folder) if x.startswith(compressed_file_name_prefix)
                ][0]
                logfile = uncompress_and_save_logfile(Path(run_folder) / compressed_file_name)

            turn_log_into_pickles(logfile)


def compare_number_of_deaths(logfile: Path, resourcefilepath: Path):
    """Helper function to produce tables summarising deaths in the model run (given be a logfile) and the corresponding
    number of deaths in the GBD dataset.
    NB.
    * Requires output from the module `tlo.methods.demography`
    * Will do scaling automatically if the scaling-factor has been computed in the simulation (but not otherwise).
    """
    output = parse_log_file(logfile)

    # 1) Get model outputs:
    # - get scaling factor:
    if 'scaling_factor' in output['tlo.methods.population']:
        sf = output['tlo.methods.population']['scaling_factor']['scaling_factor'].values[0]
    else:
        sf = 1.0

    # - extract number of death by period/sex/age-group
    model = output['tlo.methods.demography']['death'].assign(
        year=lambda x: x['date'].dt.year
    ).groupby(
        ['sex', 'year', 'age', 'label']
    )['person_id'].count().mul(sf)

    # - format categories:
    agegrps, agegrplookup = make_age_grp_lookup()
    calperiods, calperiodlookup = make_calendar_period_lookup()
    model = model.reset_index()
    model['age_grp'] = model['age'].map(agegrplookup).astype(make_age_grp_types())
    model['period'] = model['year'].map(calperiodlookup).astype(make_calendar_period_type())
    model = model.drop(columns=['age', 'year'])

    # - sum over period and divide by five to give yearly averages
    model = model.groupby(['period', 'sex', 'age_grp', 'label']).sum().div(5.0).rename(
        columns={'person_id': 'model'}).replace({0: np.nan})

    # 2) Load comparator GBD datasets
    # - Load data, format and limit to deaths only:
    gbd_dat = format_gbd(pd.read_csv(resourcefilepath / 'gbd' / 'ResourceFile_Deaths_And_DALYS_GBD2019.csv'))
    gbd_dat = gbd_dat.loc[gbd_dat['measure_name'] == 'Deaths']
    gbd_dat = gbd_dat.rename(columns={
        'Sex': 'sex',
        'Age_Grp': 'age_grp',
        'Period': 'period',
        'GBD_Est': 'mean',
        'GBD_Lower': 'lower',
        'GBD_Upper': 'upper'})

    # - Label GBD causes of death by 'label' defined in the simulation
    mapper_from_gbd_causes = pd.Series(
        output['tlo.methods.demography']['mapper_from_gbd_cause_to_common_label'].drop(columns={'date'}).loc[0]
    ).to_dict()
    gbd_dat['label'] = gbd_dat['cause_name'].map(mapper_from_gbd_causes)
    assert not gbd_dat['label'].isna().any()

    # - Create comparable data structure:
    gbd = gbd_dat.groupby(['period', 'sex', 'age_grp', 'label'])[['mean', 'lower', 'upper']].sum().div(5.0)
    gbd = gbd.add_prefix('GBD_')

    # 3) Return summary
    return gbd.merge(model, on=['period', 'sex', 'age_grp', 'label'], how='left')


def flatten_multi_index_series_into_dict_for_logging(ser: pd.Series) -> dict:
    """Helper function that converts a pd.Series with multi-index into a dict format that is suitable for logging.
    It does this by converting the multi-index into keys of type `str` in a format that later be used to reconstruct
    the multi-index (using `unflatten_flattened_multi_index_in_logging`)."""

    assert not ser.index.has_duplicates, "There should not be any duplicates in the multi-index. These will be lost" \
                                         "in the conversion to a dict."

    names_of_multi_index = ser.index.names
    _df = ser.reset_index()
    flat_index = list()
    for _, row in _df.iterrows():
        flat_index.append('|'.join([f"{col}={row[col]}" for col in names_of_multi_index]))
    return dict(zip(flat_index, ser.values))


def unflatten_flattened_multi_index_in_logging(_x: [pd.DataFrame, pd.Index]) -> [pd.DataFrame, pd.Index]:
    """Helper function that recreate the multi-index of logged results from a pd.DataFrame that is generated by
    `parse_log`.

    If a pd.DataFrame created by `parse_log` is the result of repeated logging of a pd.Series with a multi-index that
    was transformed before logging using `flatten_multi_index_series_into_dict_for_logging`, then the pd.DataFrame's
    columns will be those flattened labels. This helper function recreates the original multi-index from which the
    flattened labels were created and applies it to the pd.DataFrame.

    Alternatively, if jus the index of the "flattened" labels is provided, then the equivalent multi-index is returned.
    """

    def gen_mutli_index(_idx: pd.Index):
        """Returns the multi-index represented by the flattened index."""
        index_value_list = list()
        for col in _idx.str.split('|'):
            index_value_list.append(tuple(component.split('=')[1] for component in col))
        index_name_list = tuple(component.split('=')[0] for component in _idx[0].split('|'))
        return pd.MultiIndex.from_tuples(index_value_list, names=index_name_list)

    if isinstance(_x, pd.DataFrame):
        _y = _x.copy()
        _y.columns = gen_mutli_index(_x.columns)
        return _y
    else:
        return gen_mutli_index(_x)


class LogsDict(Mapping):
    """Parses module-specific log files and returns Pandas dataframes.

        The dictionary returned has the format::

            {
                <logger 1 name>: {
                                   <log key 1>: <pandas dataframe>,
                                   <log key 2>: <pandas dataframe>,
                                   <log key 3>: <pandas dataframe>
                                 },

                <logger 2 name>: {
                                   <log key 4>: <pandas dataframe>,
                                   <log key 5>: <pandas dataframe>,
                                   <log key 6>: <pandas dataframe>
                                 },
                ...
            }
    """

    def __init__(self, file_names_and_paths):
        super().__init__()
        # initialise class with module-specific log files paths
        self._logfile_names_and_paths: Dict[str, str] = file_names_and_paths

        # create a dictionary that will contain cached data
        self._results_cache: Dict[str, Dict] = dict()

    def __getitem__(self, key, cache=True):
        # check if the requested key is found in a dictionary containing module name and log file paths. if key
        # is found, return parsed logs else return KeyError
        if key in self._logfile_names_and_paths:
            # check if key is found in cache
            if key not in self._results_cache:
                if self._logfile_names_and_paths[key].endswith('.pickle'):
                    # return the already pickled file
                    result_df = pickle.load(open(self._logfile_names_and_paths[key], 'rb'))
                else:
                    # if the pickled result doesn't exist, we need to create it
                    result_df = _parse_log_file_inner_loop(self._logfile_names_and_paths[key])
                    # get metadata for the selected log file and merge it all with the selected key
                    result_df[key]['_metadata'] = result_df['_metadata']
                    pickle_filename = str((Path(self._logfile_names_and_paths[key])).parent / (key + '.pickle'))
                    pickle.dump(result_df, open(pickle_filename, 'wb'))
                    self._logfile_names_and_paths[key] = pickle_filename

                if not cache:  # check if caching is disallowed
                    return result_df[key]
                self._results_cache[key] = result_df[key]    # add key specific parsed results to cache
            return self._results_cache[key]  # return the added results

        else:
            raise KeyError

    def __contains__(self, k):
        # if key k is a valid logfile entry
        return k in self._logfile_names_and_paths

    def items(self):
        # parse module-specific log file and return results as a generator
        for key in self._logfile_names_and_paths.keys():
            module_specific_logs = self.__getitem__(key, cache=False)
            yield key, module_specific_logs

    def __repr__(self):
        return repr(self._logfile_names_and_paths)

    def __len__(self):
        return len(self._logfile_names_and_paths)

    def keys(self):
        # return dictionary keys
        return self._logfile_names_and_paths.keys()

    def values(self):
        # parse module-specific log file and yield the results
        for key in self._logfile_names_and_paths.keys():
            module_specific_logs = self.__getitem__(key, cache=False)
            yield module_specific_logs

    def __iter__(self):
        return iter(self._logfile_names_and_paths)

    def __getstate__(self):
        # Ensure all items cached before pickling
        for key in self.keys():
            self.__getitem__(key, cache=True)
        return self.__dict__


def get_filtered_treatment_ids(depth: Optional[int] = None) -> List[str]:
    """Return a list of treatment_ids that are defined in the model, filtered to a specified depth."""

    def filter_treatments(_treatments: Iterable[str], depth: int = 1) -> List[str]:
        """Reduce an iterable of `TREATMENT_IDs` by ignoring difference beyond a certain depth of specification and
        adding '_*' to the end to serve as a wild-card.
        N.B., The TREATMENT_ID is defined with each increasing level of specification separated by a `_`. """
        return sorted(list(set(
            [
                "".join(f"{x}_" for i, x in enumerate(t.split('_')) if i < depth).rstrip('_') + '_*'
                for t in set(_treatments)
            ]
        )))

    # Get pd.DataFrame with information of all the defined HSI
    # Import within function to avoid circular import error
    from tlo.analysis.hsi_events import get_all_defined_hsi_events_as_dataframe
    hsi_event_details = get_all_defined_hsi_events_as_dataframe()

    # Return list of TREATMENT_IDs and filter to the resolution needed
    return filter_treatments(hsi_event_details['treatment_id'], depth=depth if depth is not None else np.inf)


def colors_in_matplotlib() -> tuple:
    """Return tuple of the strings for all the colours defined in Matplotlib."""
    return tuple(
        set().union(
            mcolors.BASE_COLORS.keys(),
            mcolors.TABLEAU_COLORS.keys(),
            mcolors.CSS4_COLORS.keys(),
        )
    )


def _define_coarse_appts() -> pd.DataFrame:
    """Define which appointment types fall into which 'coarse appointment' category, the order of the categories and the
    colour of the category.
    Names of colors are selected with reference to: https://i.stack.imgur.com/lFZum.png"""
    return pd.DataFrame.from_records(
        [
            {
                'category': 'Outpatient',
                'appt_types': ['Under5OPD', 'Over5OPD'],
                'color': 'magenta'
            },
            {
                'category': 'Con w/ DCSA',
                'appt_types': ['ConWithDCSA'],
                'color': 'crimson'},
            {
                'category': 'A & E',
                'appt_types': ['AccidentsandEmerg'],
                'color': 'forestgreen'},
            {
                'category': 'Inpatient',
                'appt_types': ['InpatientDays', 'IPAdmission'],
                'color': 'mediumorchid'},
            {
                'category': 'RMNCH',
                'appt_types': ['AntenatalFirst', 'ANCSubsequent', 'NormalDelivery', 'CompDelivery', 'Csection', 'EPI',
                               'FamPlan', 'U5Malnutr'],
                'color': 'gold'},
            {
                'category': 'HIV/AIDS',
                'appt_types': ['VCTNegative', 'VCTPositive', 'MaleCirc', 'NewAdult', 'EstMedCom', 'EstNonCom', 'PMTCT',
                               'Peds'],
                'color': 'darkturquoise'},
            {
                'category': 'Tb',
                'appt_types': ['TBNew', 'TBFollowUp'],
                'color': 'y'},
            {
                'category': 'Dental',
                'appt_types': ['DentAccidEmerg', 'DentSurg', 'DentalU5', 'DentalO5'],
                'color': 'rosybrown'},
            {
                'category': 'Mental Health',
                'appt_types': ['MentOPD', 'MentClinic'],
                'color': 'lightsalmon'},
            {
                'category': 'Surgery / Radiotherapy',
                'appt_types': ['MajorSurg', 'MinorSurg', 'Radiotherapy'],
                'color': 'orange'},
            {
                'category': 'STI',
                'appt_types': ['STI'],
                'color': 'slateblue'},
            {
                'category': 'Lab / Diagnostics',
                'appt_types': ['LabHaem', 'LabPOC', 'LabParasit', 'LabBiochem', 'LabMicrobio', 'LabMolec', 'LabTBMicro',
                               'LabSero', 'LabCyto', 'LabTrans', 'Ultrasound', 'Mammography', 'MRI', 'Tomography',
                               'DiagRadio'],
                'color': 'dodgerblue'}
        ]
    ).set_index('category')


def get_coarse_appt_type(appt_type: str) -> str:
    """Return the `coarser` categorization of appt_types for a given appt_type. """
    for coarse_appt_types, row in _define_coarse_appts().iterrows():
        if appt_type in row['appt_types']:
            return coarse_appt_types


def order_of_coarse_appt(_coarse_appt: Union[str, pd.Index]) -> Union[int, pd.Index]:
    """Define a standard order for the coarse appointment types."""
    order = _define_coarse_appts().index
    if isinstance(_coarse_appt, str):
        return tuple(order).index(_coarse_appt)
    else:
        return order[order.isin(_coarse_appt)]


def get_color_coarse_appt(coarse_appt_type: str) -> str:
    """Return the colour (as matplotlib string) assigned to this appointment type. Returns `np.nan` if appointment-type
    is not recognised.
    Names of colors are selected with reference to: https://i.stack.imgur.com/lFZum.png"""
    colors = _define_coarse_appts().color
    if coarse_appt_type in colors.index:
        return colors.loc[coarse_appt_type]
    else:
        return np.nan


def _define_short_treatment_ids() -> pd.Series:
    """Define the order of the short treatment_ids and the color for each.
    Names of colors are selected with reference to: https://matplotlib.org/stable/gallery/color/named_colors.html"""
    return pd.Series({
        'FirstAttendance*': 'darkgrey',
        'Inpatient*': 'silver',

        'Contraception*': 'darkseagreen',
        'AntenatalCare*': 'green',
        'DeliveryCare*': 'limegreen',
        'PostnatalCare*': 'springgreen',

        'Alri*': 'darkorange',
        'Diarrhoea*': 'tan',
        'Undernutrition*': 'gold',
        'Epi*': 'darkgoldenrod',

        'Hiv*': 'deepskyblue',
        'Malaria*': 'lightsteelblue',
        'Measles*': 'cornflowerblue',
        'Tb*': 'mediumslateblue',
        'Schisto*': 'skyblue',

        'CardioMetabolicDisorders*': 'brown',

        'BladderCancer*': 'orchid',
        'BreastCancer*': 'mediumvioletred',
        'OesophagealCancer*': 'deeppink',
        'ProstateCancer*': 'hotpink',
        'OtherAdultCancer*': 'palevioletred',

        'Depression*': 'indianred',
        'Epilepsy*': 'red',

        'Rti*': 'lightsalmon',
    })


def order_of_short_treatment_ids(_short_treatment_id: Union[str, pd.Index]) -> Union[int, pd.Index]:
    """Define a standard order for short treatment_ids."""
    order = _define_short_treatment_ids().index
    if isinstance(_short_treatment_id, str):
        return tuple(order).index(_short_treatment_id.replace('_*', '*'))
    else:
        return order[order.isin(_short_treatment_id)]


def get_color_short_treatment_id(short_treatment_id: str) -> str:
    """Return the colour (as matplotlib string) assigned to this shorted TREATMENT_ID. Returns `np.nan` if treatment_id
    is not recognised."""
    colors = _define_short_treatment_ids()
    _short_treatment_ids_with_trailing_asterix = short_treatment_id.replace('_*', '*').rstrip('*') + '*'
    if _short_treatment_ids_with_trailing_asterix in colors.index:
        return colors.loc[_short_treatment_ids_with_trailing_asterix]
    else:
        return np.nan


def _define_cause_of_death_labels() -> pd.Series:
    """Define the order of the cause_of_death_labels and the color for each.
    Names of colors are selected with reference to: https://matplotlib.org/stable/gallery/color/named_colors.html"""
    return pd.Series({
        'Maternal Disorders': 'green',
        'Neonatal Disorders': 'springgreen',
        'Congenital birth defects': 'mediumaquamarine',

        'Lower respiratory infections': 'darkorange',
        'Childhood Diarrhoea': 'tan',

        'AIDS': 'deepskyblue',
        'Malaria': 'lightsteelblue',
        'Measles': 'cornflowerblue',
        'non_AIDS_TB': 'mediumslateblue',

        'Heart Disease': 'sienna',
        'Kidney Disease': 'chocolate',
        'Diabetes': 'peru',
        'Stroke': 'burlywood',

        'Cancer (Bladder)': 'deeppink',
        'Cancer (Breast)': 'darkmagenta',
        'Cancer (Oesophagus)': 'mediumvioletred',
        'Cancer (Other)': 'crimson',
        'Cancer (Prostate)': 'hotpink',

        'Depression / Self-harm': 'goldenrod',
        'Epilepsy': 'gold',

        'Transport Injuries': 'lightsalmon',

        'Other': 'dimgrey',
    })


def order_of_cause_of_death_label(_cause_of_death_label: Union[str, pd.Index]) -> Union[int, pd.Index]:
    """Define a standard order for Cause-of-Death labels."""
    order = _define_cause_of_death_labels().index
    if isinstance(_cause_of_death_label, str):
        return tuple(order).index(_cause_of_death_label)
    else:
        return pd.Index(sorted(_cause_of_death_label, key=order_of_cause_of_death_label))


def get_color_cause_of_death_label(cause_of_death_label: str) -> str:
    """Return the colour (as matplotlib string) assigned to this shorted Cause-of-Death Label. Returns `np.nan` if
    label is not recognised."""
    colors = _define_cause_of_death_labels()
    if cause_of_death_label in colors.index:
        return colors.loc[cause_of_death_label]
    else:
        return np.nan


def squarify_neat(sizes: np.array, label: np.array, colormap: Callable, numlabels=5, **kwargs):
    """Pass through to squarify, with some customisation: ...
     * Apply the colormap specified
     * Only give label a selection of the segments
     N.B. The package `squarify` is required.
    """
    # Suppress labels for all but the `numlabels` largest entries.
    to_label = set(pd.Series(index=label, data=sizes).sort_values(ascending=False).iloc[0:numlabels].index)

    squarify.plot(
        sizes=sizes,
        label=[_label if _label in to_label else '' for _label in label],
        color=[colormap(_x) for _x in label],
        **kwargs,
    )


def get_root_path(starter_path: Optional[Path] = None) -> Path:
    """Returns the absolute path of the top level of the repository. `starter_path` optionally gives a reference
    location from which to begin search; if omitted the location of this file is used."""

    def get_git_root(path: Path) -> Path:
        """Return path of git repo. Based on: https://stackoverflow.com/a/41920796"""
        git_repo = git.Repo(path, search_parent_directories=True)
        git_root = git_repo.working_dir
        return Path(git_root)

    if starter_path is None:
        return get_git_root(__file__)
    elif Path(starter_path).exists() and Path(starter_path).is_absolute():
        return get_git_root(starter_path)
    else:
        raise OSError("File Not Found")


def plot_clustered_stacked(dfall, ax, color_for_column_map=None, legends=True, H="/", **kwargs):
    """Given a dict of dataframes, with identical columns and index, create a clustered stacked bar plot.
    * H is the hatch used for identification of the different dataframe.
    * color_for_column_map should return a color for every column in the dataframes
    * legends=False, suppresses generation of the legends
    From: https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars"""

    n_df = len(dfall)
    n_col = len(list(dfall.values())[0].columns)
    n_ind = len(list(dfall.values())[0].index)

    for i, df in enumerate(dfall.values()):  # for each data frame
        ax = df.plot.bar(
            stacked=True,
            ax=ax,
            legend=False,
            color=[color_for_column_map(_label) for _label in df.columns],
            **kwargs
        )

    _handles, _labels = ax.get_legend_handles_labels()  # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col):  # len(h) = n_col * n_df
        for j, pa in enumerate(_handles[i: i+n_col]):
            for rect in pa.patches:  # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col))  # edited part
                rect.set_width(1 / float(n_df + 1))

    ax.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    ax.set_xticklabels(df.index, rotation=0)

    if legends:
        # Add invisible data to add another legend
        n = []
        for i in range(n_df):
            n.append(ax.bar(0, 0, color="gray", hatch=H * i))

        l1 = ax.legend(_handles[:n_col], _labels[:n_col], loc=[1.01, 0.5])
        _ = plt.legend(n, dfall.keys(), loc=[1.01, 0.1])
        ax.add_artist(l1)
