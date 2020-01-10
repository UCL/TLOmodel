"""
General utility functions for TLO analysis
"""
import logging
from ast import literal_eval
from pathlib import Path

import pandas as pd

from tlo import util
from tlo.util import create_age_range_lookup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_line(line):
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
    logger.debug('%s', line)
    info = {
        'logger': parts[1],
        'sim_date': parts[2],
        'key': parts[3],
        'object': literal_eval(parts[4])
    }
    logger.debug('%s', info)
    return info


def parse_log_file(filepath):
    """
    Parses logged output from a TLO run and create Pandas dataframes for analysis. See
    parse_output() for details

    :param filepath: file path to log file
    :return: dictionary of parsed log data
    """
    with open(filepath) as log_file:
        return parse_output(log_file.readlines())


def parse_output(list_of_log_lines):
    """
    Parses logged output from a TLO run and create Pandas dataframes for analysis.

    Use parse_log_file() wrapper if required

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
            i = parse_line(line.strip())
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
                    logger.warning('List to dataframe %s, number of columns do not match', i['key'])
                # add list to columns (skip first column, which is date)
                row = dict(zip(df.columns[1:], i['object']))
                row['date'] = i['sim_date']
            else:
                print('Could not parse line: %s' % line)
                continue
            # append the new row to the dataframe for this logger & log name
            o[i['logger']][i['key']] = df.append(row, ignore_index=True)
    return o


def scale_to_population(parsed_output, resourcefilepath):
    """
    This helper function scales certain outputs so that they can create statistics for the whole population.
    e.g. Population Size, Number of deaths are scaled by the factor of {Model Pop Size at Start of Simulation} to {
    {Real Population at the same time}.

    NB. This file gives precedence to the Malawi Population Census

    :param parsed_outoput: The outputs from parse_output
    :param resourcefilepath: The resourcefilepath
    :return: a new version of parsed_output that includes certain variables scaled
    """

    # Get information about the real population size (Malawi Census in 2018)
    cens_tot = pd.read_csv(Path(resourcefilepath) / "ResourceFile_PopulationSize_2018Census.csv")['Count'].sum()
    cens_yr = 2018

    # Get information about the model population size in 2018 (and fail if no 2018)
    model_res = parsed_output['tlo.methods.demography']['population']
    model_res['year'] = pd.to_datetime(model_res.date).dt.year

    assert cens_yr in model_res.year.values, "Model results do not contain the year of the census, so cannot scale"
    model_tot = model_res.loc[model_res['year'] == cens_yr, 'total'].values[0]

    # Calculate ratio for scaling
    ratio_data_to_model = cens_tot / model_tot

    # Do the scaling on selected columns in the parsed outputs:
    o = parsed_output.copy()

    # Multiply population count summaries by ratio
    o['tlo.methods.demography']['population']['male'] *= ratio_data_to_model
    o['tlo.methods.demography']['population']['female'] *= ratio_data_to_model
    o['tlo.methods.demography']['population']['total'] *= ratio_data_to_model

    o['tlo.methods.demography']['age_range_m'].iloc[:, 1:] *= ratio_data_to_model
    o['tlo.methods.demography']['age_range_f'].iloc[:, 1:] *= ratio_data_to_model

    # For individual-level reporting, construct groupby's and then multipy by ratio
    # 1) Counts of numbers of death by year/age/cause
    deaths = o['tlo.methods.demography']['death']
    deaths.index = pd.to_datetime(deaths['date'])
    deaths['year'] = deaths.index.year.astype(int)

    deaths_groupby_scaled = deaths[['year', 'sex', 'age', 'cause', 'person_id']].groupby(
        by=['year', 'sex', 'age', 'cause']).count().unstack(fill_value=0).stack() * ratio_data_to_model
    deaths_groupby_scaled.rename(columns={'person_id': 'count'}, inplace=True)
    o['tlo.methods.demography'].update({'death_groupby_scaled': deaths_groupby_scaled})

    # 2) Counts of numbers of births by year/age-of-mother
    births = o['tlo.methods.demography']['on_birth']
    births.index = pd.to_datetime(births['date'])
    births['year'] = births.index.year
    births_groupby_scaled = \
        births[['year', 'mother_age', 'mother']].groupby(by=['year', 'mother_age']).count() \
        * ratio_data_to_model
    births_groupby_scaled.rename(columns={'mother': 'count'}, inplace=True)
    o['tlo.methods.demography'].update({'birth_groupby_scaled': births_groupby_scaled})

    # TODO: Do this kind of manipulation for all things in the log that are /
    #  flagged are being subject to scaling - issue raised.
    return o


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
    return pd.CategoricalIndex(categories=keys, ordered=True)


def make_calendar_period_type():
    """
    Make an ordered categorical type for calendar periods
    Returns CategoricalDType
    """
    keys, _ = make_calendar_period_lookup()
    return pd.CategoricalIndex(categories=keys, ordered=True)
