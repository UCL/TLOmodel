"""
Extracts DALYs and mortality from the TB module
 """
import datetime
import pickle
from pathlib import Path
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from tlo.analysis.utils import extract_results, summarize
from tlo import Date

resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
#outputpath = Path("./outputs/nic503@york.ac.uk")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")
# with open(outputpath / "default_run.pickle", "rb") as f:
#     output = pickle.load(f)

with open(outputpath / "nocxr.pickle", "rb") as f:
    output = pickle.load(f)
TARGET_PERIOD = (Date(2010, 1, 1), Date(2013, 12, 31))
def get_num_deaths(_df):
    """Return total number of Deaths (total within the TARGET_PERIOD)
    """
    return pd.Series(data=len(_df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)]))
def get_num_dalys(_df):
    """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD)"""
    return pd.Series(
        data=_df
        .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])]
        .drop(columns=['date', 'sex', 'age_range', 'year'])
        .sum().sum()
    )

print(output.keys())
#print(output(tlo.methods.healthsystem.summary).keys())
# output serialises availability of  CXR consumables
cons_available = output['tlo.methods.healthsystem.summary']['Consumables'].drop(columns=[])
cons_available .to_excel(outputpath / "cons_available_nocxr.xlsx")

# Exports TB program indicators
print(f"projected TB incidence{output['tlo.methods.tb']['tb_incidence']}")
TB_incidence= output['tlo.methods.tb']['tb_incidence'].drop(columns=[])
TB_incidence.to_excel(outputpath / "new_TB_cases_nocxr.xlsx")

# output DALYs
print(f"expected dalys{output['tlo.methods.healthburden']['dalys_stacked']}")
#sample_dalys= output['tlo.methods.healthburden']['dalys_stacked'].groupby(['cause', 'sex']).size()
sample_dalys= output['tlo.methods.healthburden']['dalys_stacked'].drop(columns=[])
sample_dalys.to_excel(outputpath / "dalys_nocxr.xlsx")

# output serialises mortality patterns
print(f"expected deaths {output['tlo.methods.demography']['death']}")
sample_deaths = output['tlo.methods.demography']['death'].groupby(['date','cause', 'sex']).size()
#sample_deaths = output['tlo.methods.demography']['death'].drop(columns=[])
sample_deaths.to_excel(outputpath / "mortality_nocxr.xlsx")



