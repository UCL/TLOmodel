""" load the data from ESPEN and extract key data """

import datetime
import pickle
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo.analysis.utils import compare_number_of_deaths

resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs

data = pd.read_excel(outputpath / 'ESPEN_district_data.xlsx', sheet_name='Sheet1')
