"""
General utility functions for TLO analysis
"""
import os
import pickle
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import logging, util
from tlo.logging.reader import LogData
from tlo.util import create_age_range_lookup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _parse_line(line):
    """
    Parses a single line of logged output. It has the format:
    INFO|<logger name>|<simulation date>|<log key>|<python object>

    It returns the dictionary:
        { 'logger': <logger name>,
          'sim_date': <simulation date>,
          'key': <the key of this log entry>,
          'object': <the logged python object>
        }

    :param line: the full line from log file
    :return: a dictionary with parsed line
    """
    parts = line.split('|')

    if len(parts) != 5:
        return None

    logger.debug(key='debug', data=line)

    try:
        parsed = literal_eval(parts[4])
    except ValueError:
        parsed = eval(parts[4], {'Timestamp': pd.Timestamp, 'nan': np.nan, 'NaT': pd.NaT})

    info = {
        'logger': parts[1],
        'sim_date': parts[2],
        'key': parts[3],
        'object': parsed
    }
    logger.debug(key='debug', data=str(info))
    return info


def parse_log_file(filepath, level: int = logging.INFO):
    """Parses logged output from a TLO run and returns Pandas dataframes.

    The format can be one of two style, old-style TLO logging like ::

        INFO|<logger name>|<simulation datestamp>|<log key>|<python list or dictionary>

    or a JSON representation with the first instance from a log key being a header line, and all following
    rows being data only rows (without column names or metadata).

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

    :param filepath: file path to log file
    :param level: logging level to be parsed for structured logging
    :return: dictionary of parsed log data
    """
    oldstyle_loglines = []
    log_data = LogData()
    with open(filepath) as log_file:
        for line in log_file:
            # only parse json entities
            if line.startswith('{'):
                log_data.parse_log_line(line, level)
            else:
                oldstyle_loglines.append(line)

    # convert dictionaries to dataframes
    output_logs = {**log_data.get_log_dataframes(), **_oldstyle_parse_output(oldstyle_loglines)}
    return output_logs


def _oldstyle_parse_output(list_of_log_lines):
    """Parses logged output from a TLO run and create Pandas dataframes for analysis.

    Used by parse_log_file() to handle old-style TLO logging

    Each input line follows the format:
    INFO|<logger name>|<simulation datestamp>|<log key>|<python list or dictionary>

    e.g.

    [
    'INFO|tlo.methods.demography|2010-11-02 23:00:59.111968|on_birth|{'mother': 17, 'child': 50}',
    'INFO|tlo.methods.demography|2011-01-01 00:00:00|population|{'total': 51, 'male': 21, 'female': 30}',
    'INFO|tlo.methods.demography|2011-01-01 00:00:00|age_range_m|[5, 4, 1, 1, 1, 2, 1, 2, 2, 1, 1, 0]',
    'INFO|tlo.methods.demography|2011-01-01 00:00:00|age_range_f|[4, 7, 5, 1, 5, 1, 2, 0, 1, 2, 0, 1]',
    ]

    The dictionary returned has the format:
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

    :param list_of_log_lines: a list of log lines in the required format
    :return: a dictionary holding logged data as Python objects
    """
    o = dict()

    # for each logged line
    for line in list_of_log_lines:
        # we only parse 'INFO' lines that have 5 parts
        if line.startswith('INFO'):
            i = _parse_line(line.strip())
            # if this line isn't in the right format
            if not i:
                continue
            # add a dictionary for the logger name, if required
            if i['logger'] not in o:
                o[i['logger']] = dict()
            # add a dataframe for the name/key of this log entry, if required
            if i['key'] not in o[i['logger']]:
                # if the logged data is a list, it doesn't have column names
                if isinstance(i['object'], list):
                    # create column names for each entry in the list
                    columns = ['col_%d' % x for x in range(0, len(i['object']))]
                else:
                    # create column names from the keys of the dictionary
                    columns = list(i['object'].keys())
                columns.insert(0, 'date')
                o[i['logger']][i['key']] = pd.DataFrame(columns=columns)

            df = o[i['logger']][i['key']]

            # create a new row to append to the dataframe, add the simulation date
            if isinstance(i['object'], dict):
                row = i['object']
                row['date'] = i['sim_date']
            elif isinstance(i['object'], list):
                if len(df.columns) - 1 != len(i['object']):
                    logger.warning(key='warning', data=f'List to dataframe {i["key"]} number of columns do not match')
                # add list to columns (skip first column, which is date)
                row = dict(zip(df.columns[1:], i['object']))
                row['date'] = i['sim_date']
            else:
                print('Could not parse line: %s' % line)
                continue
            # append the new row to the dataframe for this logger & log name
            o[i['logger']][i['key']] = df.append(row, ignore_index=True)
    return o


def write_log_to_excel(filename, log_dataframes):
    """Takes the output of parse_log_file() and creates an Excel file from dataframes"""
    sheets = list()
    sheet_count = 0
    metadata = log_dataframes['_metadata']
    for module, key_df in log_dataframes.items():
        if module != '_metadata':
            for key, df in key_df.items():
                sheet_count += 1
                sheets.append([module, key, sheet_count, metadata[module][key]['description']])

    writer = pd.ExcelWriter(filename)
    index = pd.DataFrame(data=sheets, columns=['module', 'key', 'sheet', 'description'])
    index.to_excel(writer, sheet_name='Index')

    sheet_count = 0
    for module, key_df in log_dataframes.items():
        if module != '_metadata':
            for key, df in key_df.items():
                sheet_count += 1
                df.to_excel(writer, sheet_name=f'Sheet {sheet_count}')
    writer.save()


def make_calendar_period_lookup():
    """Returns a dictionary mapping calendar year (in years) to five year period
    i.e. { 0: '0-4', 1: '0-4', ..., 119: '100+', 120: '100+' }
    """

    # Recycles the code used to make age-range lookups:
    ranges, lookup = util.create_age_range_lookup(1950, 2100, 5)

    # Removes the '1950-' category
    ranges.remove('0-1950')
    for year in range(1950):
        lookup.pop(year)

    return ranges, lookup


def make_age_grp_types():
    """
    Make an ordered categorical type for age-groups
    Returns CategoricalDType
    """
    keys, _ = create_age_range_lookup(min_age=0, max_age=100, range_size=5)
    return pd.CategoricalDtype(categories=keys, ordered=True)


def make_calendar_period_type():
    """
    Make an ordered categorical type for calendar periods
    Returns CategoricalDType
    """
    keys, _ = make_calendar_period_lookup()
    return pd.CategoricalDtype(categories=keys, ordered=True)


def get_scenario_outputs(scenario_filename: str, outputs_dir: Path) -> list:
    """Returns paths of folders associated with a batch_file, in chronological order."""
    stub = scenario_filename.rstrip('.py')
    folders = [Path(f) for f in os.scandir(outputs_dir) if f.is_dir() and f.name.startswith(stub)]
    folders.sort()
    return folders


def get_scenario_info(scenario_output_dir: Path) -> dict:
    """Utility function to get the the number draws and the number of runs in a batch set.

    TODO: read the JSON file to get further information
    """
    info = dict()
    draw_folders = [f for f in os.scandir(scenario_output_dir) if f.is_dir()]

    info['number_of_draws'] = len(draw_folders)

    run_folders = [f for f in os.scandir(draw_folders[0]) if f.is_dir()]
    info['runs_per_draw'] = len(run_folders)

    return info


def load_pickled_dataframes(results_folder: Path, draw=0, run=0, name=None) -> dict:
    """Utility function to create a dict contaning all the logs from the specified run within a batch set"""
    folder = results_folder / str(draw) / str(run)
    pickles = [p for p in os.scandir(folder) if p.name.endswith('.pickle')]
    if name is not None:
        pickles = [p for p in pickles if p.name in f"{name}.pickle"]

    output = dict()
    for p in pickles:
        name = os.path.splitext(p.name)[0]
        with open(p.path, "rb") as f:
            output[name] = pickle.load(f)

    return output


def extract_params(results_folder: Path) -> pd.DataFrame:
    """Utility function to get overridden parameters from scenario runs

    Returns dateframe summarizing parameters that change across the draws. It produces a dataframe with index of draw
    and columns of each parameters that is specified to be varied in the batch. NB. This does the extraction from run 0
    in each draw, under the assumption that the over-written parameters are the same in each run.
    """

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


def extract_results(results_folder: Path, module: str, key: str, column: str, index: str = None) -> pd.DataFrame:
    """Utility function to unpack results

    Produces a dataframe that summaries one series from the log, with column multi-index for the draw/run. If an 'index'
    component of the log_element is provided, the dataframe uses that index (but note that this will only work if the
    index is the same in each run).
    """

    results_index = None
    if index is not None:
        # extract the index from the first log, and use this ensure that all other are exactly the same.
        filename = f"{module}.pickle"
        df: pd.DataFrame = load_pickled_dataframes(results_folder, draw=0, run=0, name=filename)[module][key]
        results_index = df[index]

    # get number of draws and numbers of runs
    info = get_scenario_info(results_folder)

    results = pd.DataFrame(columns=pd.MultiIndex.from_product(
        [range(info['number_of_draws']), range(info['runs_per_draw'])],
        names=["draw", "run"]
    ))

    for draw in range(info['number_of_draws']):
        for run in range(info['runs_per_draw']):
            try:
                df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]
                results[draw, run] = df[column]

                if index is not None:
                    idx = df[index]
                    assert idx.equals(results_index), "Indexes are not the same between runs"

            except ValueError:
                results[draw, run] = np.nan

    # if 'index' is provided, set this to be the index of the results
    if index is not None:
        results.index = results_index

    return results


def summarize(results: pd.DataFrame, only_mean: bool = False) -> pd.DataFrame:
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

    if only_mean:
        # Remove other metrics and simplify if 'only_mean' is required:
        om = summary.loc[:, (slice(None), "mean")]
        om.columns = [c[0] for c in om.columns.to_flat_index()]
        return om

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
