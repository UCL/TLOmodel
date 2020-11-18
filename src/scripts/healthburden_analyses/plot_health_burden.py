import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Where will outputs be found
outputpath = Path("./outputs")  # folder for convenience of storing outputs
results_filename = outputpath / 'long_run.pickle'

with open(results_filename, 'rb') as f:
    output = pickle.load(f)['output']


