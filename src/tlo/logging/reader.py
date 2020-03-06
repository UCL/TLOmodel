import json
from collections import defaultdict
from typing import DefaultDict, Dict

import pandas as pd


class LogRow:
    """Convenience class for interacting with packet of json log data"""
    is_header = False

    def __init__(self, line: str):
        log_data = json.loads(line)
        self.key = log_data['key']
        self.module = log_data['module']
        self.log_id = f'{self.module}_{self.key}'

        if log_data['type'] == 'header':
            self.is_header = True
            self.level = log_data['level']
            self.header_data = log_data
        else:
            self.date = log_data['date']
            self.values = log_data['values']


class LogData:
    """Builds up log data for export as dictionary with dataframes"""
    def __init__(self):
        self.data: DefaultDict[str, Dict[str, Dict[str, list]]] = defaultdict(dict)
        self.allowed_logs = set()

    def parse_log_row(self, log_row: LogRow, level: str):
        """
        Parse LogRow at desired level

        :param log_row: LogRow that can either be a header or data row
        :param level: matching level to add to log, other levels will not be added
        """
        # new header line, if this is the right level, then add module and key to log with header and blank data
        if log_row.is_header:
            if log_row.level == level:
                self.allowed_logs.add(log_row.log_id)
                self.data[log_row.module][log_row.key] = {'header': log_row.header_data, 'values': [], 'dates': []}
        # log data row if we allow this logger
        elif log_row.log_id in self.allowed_logs:
            self.data[log_row.module][log_row.key]['dates'].append(log_row.date)
            self.data[log_row.module][log_row.key]['values'].append(log_row.values)

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
