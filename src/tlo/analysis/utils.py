"""
General utility functions for TLO analysis
"""
import logging
from ast import literal_eval

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_line(line):
    """
    Parses a single line of logged output. It has the format:
    INFO:<logger name>:<simulation data>:<info key>:<python object>
    :param line: the full line from log file
    :return: a dictionary with parsed line
    """
    parts = line.split('|')
    logger.debug('%s', line)
    info = {
        'logger': parts[1],
        'sim_date': parts[2],
        'name': parts[3],
        'object': literal_eval(parts[4])
    }
    logger.debug('%s', info)
    return info


def parse_output(filepath):
    """
    Parses logged output from a TLO run and create Pandas dataframes for analysis
    :param filepath: the full filepath to logged output file
    :return: a dictionary holding logged data as Python objects
    """
    o = dict()
    with open(filepath) as log_file:
        for line in log_file:
            if line.startswith('INFO'):
                i = parse_line(line.strip())
                if i['logger'] not in o:
                    o[i['logger']] = dict()
                if i['name'] not in o[i['logger']]:
                    if isinstance(i['object'], list):
                        columns = ['col_%d' % x for x in range(0, len(i['object']))]
                    else:
                        columns = list(i['object'].keys())
                    columns.insert(0, 'date')
                    o[i['logger']][i['name']] = pd.DataFrame(columns=columns)
                df = o[i['logger']][i['name']]
                if isinstance(i['object'], dict):
                    row = i['object']
                    row['date'] = i['sim_date']
                elif isinstance(i['object'], list):
                    if len(df.columns) - 1 != len(i['object']):
                        logger.warn('List to dataframe %s, number of columns do not match', i['name'])
                    # add list to columns (skip first column, which is date)
                    row = dict(zip(df.columns[1:], i['object']))
                    row['date'] = i['sim_date']
                else:
                    raise ValueError('Cannot handle log object of type %s' % type(i['object']))
                o[i['logger']][i['name']] = df.append(row, ignore_index=True)
    return o
