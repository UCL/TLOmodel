import json
import logging as _logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

import numpy as np
import pandas as pd


class LogData:
    """Builds up log data for export as dictionary with dataframes"""

    def __init__(self):
        self.data: DefaultDict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self.allowed_logs = set()
        self.uuid_to_module = dict()

    def parse_log_line(self, log_line: str, level: int):
        """
        Parse LogRow at desired level

        :param log_line: a json line from log file that can either be a header or data row
        :param level: matching level to add to log, other levels will not be added
        """
        # new header line, if this is the right level, then add module and key to log with header and blank data
        log_data = json.loads(log_line)

        if "type" in log_data and log_data["type"] == "header":
            self.uuid_to_module[log_data["uuid"]] = log_id = (
                log_data["module"],
                log_data["key"],
            )
            if getattr(_logging, log_data["level"]) >= level:
                self.allowed_logs.add(log_id)
                self.data[log_data["module"]][log_data["key"]] = {
                    "header": log_data,
                    "values": [],
                    "dates": [],
                }
        else:
            log_id = (log_data["module"], log_data["key"]) = self.uuid_to_module[
                log_data["uuid"]
            ]
            # log data row if we allow this logger
            if log_id in self.allowed_logs:
                self.data[log_data["module"]][log_data["key"]]["dates"].append(
                    log_data["date"]
                )
                self.data[log_data["module"]][log_data["key"]]["values"].append(
                    log_data["values"]
                )

    def get_log_dataframes(self) -> DefaultDict[str, Dict[str, pd.DataFrame]]:
        """
        Converts parsed logs of dictionaries to dataframes and then returns all logs

        :return: dictionary of output logs with dataframes for each log key
        """
        output_logs: DefaultDict[str, Dict[str, pd.DataFrame]] = defaultdict(dict)

        for module, log_data in self.data.items():
            output_logs["_metadata"][module] = dict()
            for key, data in log_data.items():
                output_logs["_metadata"][module][key] = data["header"]
                if list(data["header"]["columns"].keys()) == ["dataframe"]:
                    output_logs[module][key] = self.parse_logged_dataframe(
                        data["values"], data["dates"]
                    )
                else:
                    output_logs[module][key] = pd.DataFrame(
                        data["values"], columns=data["header"]["columns"].keys()
                    )
                    output_logs[module][key].insert(
                        0,
                        "date",
                        pd.Series(data["dates"], dtype=np.dtype("datetime64[ns]")),
                    )
                # for each column, cast to the correct type if necessary
                for n, t in data["header"]["columns"].items():
                    if t == "Timestamp":
                        output_logs[module][key][n] = output_logs[module][key][
                            n
                        ].astype("datetime64[ns]")
                    elif t == "Categorical":
                        output_logs[module][key][n] = output_logs[module][key][
                            n
                        ].astype("category")
                    elif t == "set":
                        output_logs[module][key][n] = output_logs[module][key][n].apply(
                            set
                        )
        return output_logs

    def parse_logged_dataframe(
        self, values: List[List[Dict[str, Dict[str, Any]]]], dates: List[str]
    ) -> pd.DataFrame:
        """
        Converts log data for an entire dataframe being logged into a mutli-indexed dataframe
        :param values: logged values
        :param dates: list of dates
        :return: Multi-indexed (log_row, df_row) dataframe
        """
        # Convert data to {(log_row_i, df_row_i): {df_col_name_1: df_col_val_1, df_col_name_2: df_col_val_2...}}
        indexed_data = {
            (log_row_i, df_row_i): log_row[df_row_i]
            # Each log row as the first part of multi-index
            for log_row_i in range(len(values))
            # each row is a list with one dictionary as the value
            for log_row in values[0]
            # the index for each row of the logged dataframe
            for df_row_i in log_row.keys()
        }
        # create dataframe from indexed data, and join dates based on the log row index
        log_date = pd.DataFrame(
            pd.Series(dates, name="date", dtype=np.dtype("datetime64[ns]"))
        )
        log_date.index.set_names("log_row", inplace=True)
        logged_df = pd.DataFrame.from_dict(indexed_data, orient="index")
        logged_df.index.set_names(["log_row", "df_row"], inplace=True)
        return log_date.join(logged_df)
