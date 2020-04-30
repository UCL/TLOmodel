import json
from collections import defaultdict
from typing import DefaultDict, Dict

import pandas as pd


class LogData:
    """Builds up log data for export as dictionary with dataframes"""
    def __init__(self):
        self.data: DefaultDict[str, Dict[str, Dict[str, list]]] = defaultdict(dict)
        self.allowed_logs = set()

    def parse_log_line(self, log_line: str, level: str):
        """
        Parse LogRow at desired level

        :param log_line: a json line from log file that can either be a header or data row
        :param level: matching level to add to log, other levels will not be added
        """
        # new header line, if this is the right level, then add module and key to log with header and blank data
        log_data = json.loads(log_line)
        log_id = (log_data['module'], log_data['key'])

        if log_data['type'] == 'header':
            if log_data['level'] == level:
                self.allowed_logs.add(log_id)
                self.data[log_data['module']][log_data['key']] = {'header': log_data, 'values': [], 'dates': []}
        # log data row if we allow this logger
        elif log_id in self.allowed_logs:
            self.data[log_data['module']][log_data['key']]['dates'].append(log_data['date'])
            self.data[log_data['module']][log_data['key']]['values'].append(log_data['values'])

    def get_log_dataframes(self) -> DefaultDict[str, Dict[str, pd.DataFrame]]:
        """
        Converts parsed logs of dictionaries to dataframes and then returns all logs

        :return: dictionary of output logs with dataframes for each log key
        """
        output_logs: DefaultDict[str, Dict[str, pd.DataFrame]] = defaultdict(dict)

        for module, log_data in self.data.items():
            for key, data in log_data.items():
                output_logs[module][key] = pd.DataFrame(data['values'], columns=data['header']['columns'].keys())
                output_logs[module][key].insert(0, "date", pd.Series(data["dates"], dtype='datetime64[ns]'))

        return output_logs
