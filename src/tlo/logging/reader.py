from collections import defaultdict

import json
import pandas as pd



def parse_structured_output(log_lines, level):
    # TODO: class of packets
    allowed_logs = set()
    parsed_logs = defaultdict(dict)
    output_logs = defaultdict(dict)

    for line in log_lines:
        # only parse json entities
        if line.startswith('{'):
            packet = json.loads(line)
            # new header line, if this is the right level, then add module and key to log with header and blank data
            if 'level' in packet.keys():
                if packet['level'] == level:
                    allowed_logs.add(f'{packet["module"]}_{packet["key"]}')
                    parsed_logs[packet['module']][packet['key']] = {'header': packet, 'values': [], 'dates': []}
                    continue
            # log data row if we allow this logger
            if f'{packet["module"]}_{packet["key"]}' in allowed_logs:
                parsed_logs[packet['module']][packet['key']]['values'].append(packet['values'])
                parsed_logs[packet['module']][packet['key']]['dates'].append(pd.Timestamp(packet['date']))

    # convert dictionaries to dataframe
    for module, keys in parsed_logs.items():
        for key, data in keys.items():
            dates = pd.DataFrame(data["dates"], columns=['date'], dtype=pd.Timestamp)
            output_logs[module][key] = pd.DataFrame(
                data['values'], columns=data['header']['columns'].keys(), index=data['dates']
            )

    return output_logs
