import json

import numpy as np
import pandas as pd


class PandasEncoder(json.JSONEncoder):
    """Encode numpy and pandas objects for writing to json"""
    def default(self, obj):
        # using base classes for numpy numeric types
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, (pd.Categorical, pd.arrays.BooleanArray)):
            # assume only only one categorical / nullable boolean value per cell
            return obj.tolist()[0]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, (type(pd.NaT), type(pd.NA))):
            return None
        # when logging a series directly, numpy datatypes are used
        elif isinstance(obj, np.datetime64):
            return pd.Timestamp(obj).isoformat()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)
