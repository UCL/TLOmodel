import json

import numpy as np
import pandas as pd


class PandasEncoder(json.JSONEncoder):
    """Encode numpy and pandas objects for writing to json"""
    def default(self, obj):
        # using base classes for numpy numeric types
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.signedinteger):
            return int(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Categorical):
            return str(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, type(pd.NaT)):
            return None
        return json.JSONEncoder.default(self, obj)
