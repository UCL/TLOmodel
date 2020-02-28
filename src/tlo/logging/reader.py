import json
from collections import defaultdict
from typing import DefaultDict, Dict

import pandas as pd


class LogRow:
    """Convenience class for interacting with packet of json log data"""
    is_header = False

    def __init__(self, line):
        log_data = json.loads(line)
        self.key = log_data['key']
        self.module = log_data['module']
        self.log_id = f'{self.module}_{self.key}'

        if 'level' in log_data.keys():
            self.is_header = True
            self.level = log_data['level']
            self.header_data = log_data
        else:
            self.date = log_data['date']
            self.values = log_data['values']


class LogData:
    """Class to build up log data"""
    def __init__(self):
        self.data: DefaultDict[str, Dict[str, Dict[str, list]]] = defaultdict(dict)
        self.allowed_logs = set()

    def parse_packet(self, packet, level):
        # new header line, if this is the right level, then add module and key to log with header and blank data
        if packet.is_header:
            if packet.level == level:
                self.allowed_logs.add(packet.log_id)
                self.data[packet.module][packet.key] = {'header': packet.header_data, 'values': [], 'dates': []}
        # log data row if we allow this logger
        elif packet.log_id in self.allowed_logs:
            self.data[packet.module][packet.key]['dates'].append(packet.date)
            self.data[packet.module][packet.key]['values'].append(packet.values)

    def get_log_dataframes(self):
        output_logs: DefaultDict[str, Dict[str, pd.DataFrame]] = defaultdict(dict)

        for module, log_data in self.data.items():
            for key, data in log_data.items():
                output_logs[module][key] = pd.DataFrame(data['values'], columns=data['header']['columns'].keys())
                output_logs[module][key].insert(0, "date", pd.Series(data["dates"], dtype='datetime64[ns]'))

        return output_logs
