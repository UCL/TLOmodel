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
    INFO:<logger name>:<simulation date>:<info key>:<python object>

    It returns the dictionary:
        { 'logger': <logger name>,
          'sim_date': <simulation date>,
          'name': <the name/type of this log entry>,
          'object': <the logged python object>
        }

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

    The dictionary returned has the following format:
    {
        <logger 1 name>: {
                           <log key 1 name>: <pandas dataframe>,
                           <log key 2 name>: <pandas dataframe>,
                           <log key 3 name>: <pandas dataframe>
                         },

        <logger 2 name>: {
                           <log key 4 name>: <pandas dataframe>,
                           <log key 5 name>: <pandas dataframe>,
                           <log key 6 name>: <pandas dataframe>
                         },
        ...
    }

    :param filepath: the full filepath to logged output file
    :return: a dictionary holding logged data as Python objects
    """
    o = dict()

    # read logging lines from the file
    with open(filepath) as log_file:
        # for each logged line
        for line in log_file:
            # we only parse 'INFO' lines
            if line.startswith('INFO'):
                i = parse_line(line.strip())
                # add a dictionary for the logger name, if required
                if i['logger'] not in o:
                    o[i['logger']] = dict()
                # add a dataframe for the name/key of this log entry, if required
                if i['name'] not in o[i['logger']]:
                    # if the logged data is a list, it doesn't have column names
                    if isinstance(i['object'], list):
                        # create column names for each entry in the list
                        columns = ['col_%d' % x for x in range(0, len(i['object']))]
                    else:
                        # create column names from the keys of the dictionary
                        columns = list(i['object'].keys())
                    columns.insert(0, 'date')
                    o[i['logger']][i['name']] = pd.DataFrame(columns=columns)

                df = o[i['logger']][i['name']]

                # create a new row to append to the dataframe, add the simulation date
                if isinstance(i['object'], dict):
                    row = i['object']
                    row['date'] = i['sim_date']
                elif isinstance(i['object'], list):
                    if len(df.columns) - 1 != len(i['object']):
                        logger.warning('List to dataframe %s, number of columns do not match', i['name'])
                    # add list to columns (skip first column, which is date)
                    row = dict(zip(df.columns[1:], i['object']))
                    row['date'] = i['sim_date']
                else:
                    raise ValueError('Cannot handle log object of type %s' % type(i['object']))
                # append the new row to the dataframe for this logger & log name
                o[i['logger']][i['name']] = df.append(row, ignore_index=True)
    return o
