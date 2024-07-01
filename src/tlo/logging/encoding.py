import json

import numpy as np
import pandas as pd
from pandas.api.types import is_extension_array_dtype


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
        elif is_extension_array_dtype(obj):
            # for pandas extension dtypes assume length 1 arrays / series are scalars
            return obj.tolist()[0 if len(obj) == 1 else slice(None)]
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
