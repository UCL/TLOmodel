import json
from collections import defaultdict
from typing import DefaultDict, Dict

import pandas as pd


class Packet:
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
            self.header_line = {'header': log_data, 'values': [], 'dates': []}
        else:
            self.date = log_data['date']
            self.values = log_data['values']


def parse_structured_output(log_lines, level):
    allowed_logs = set()
    parsed_logs: DefaultDict[str, Dict[str, Dict[str, list]]] = defaultdict(dict)
    output_logs: DefaultDict[str, Dict[str, pd.DataFrame]] = defaultdict(dict)

    for line in log_lines:
        # only parse json entities
        if line.startswith('{'):
            packet = Packet(line)
            # new header line, if this is the right level, then add module and key to log with header and blank data
            if packet.is_header:
                if packet.level == level:
                    allowed_logs.add(packet.log_id)
                    parsed_logs[packet.module][packet.key] = packet.header_line
            # log data row if we allow this logger
            elif packet.log_id in allowed_logs:
                parsed_logs[packet.module][packet.key]['dates'].append(packet.date)
                parsed_logs[packet.module][packet.key]['values'].append(packet.values)

    # convert dictionaries to dataframes
    for module, log_keys in parsed_logs.items():
        for key, data in log_keys.items():
            output_logs[module][key] = pd.DataFrame(data['values'], columns=data['header']['columns'].keys())
            output_logs[module][key].insert(0, "date", pd.Series(data["dates"], dtype='datetime64[ns]'))

    return output_logs
