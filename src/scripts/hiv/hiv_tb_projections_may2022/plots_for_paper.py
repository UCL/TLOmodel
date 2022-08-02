"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputspath' results_folder
plots created:
4-panel plot HIV and TB incidence and deaths

"""

import datetime
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

# download all files (and get most recent [-1])
results0 = get_scenario_outputs("scenario0.py", outputspath)[-1]
results1 = get_scenario_outputs("scenario1.py", outputspath)[-1]
results2 = get_scenario_outputs("scenario2.py", outputspath)[-1]
results3 = get_scenario_outputs("scenario3.py", outputspath)[-1]
results4 = get_scenario_outputs("scenario4.py", outputspath)[-1]

# ---------------------------------- Fraction HCW time-------------------------------------

# fraction of HCW time
# output fraction of time by year
def summarise_frac_hcws(results_folder):
    capacity = extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="Capacity",
        column="average_Frac_Time_Used_Overall",
    )

    capacity.columns = capacity.columns.get_level_values(0)
    hcw = pd.DataFrame(index=capacity.index, columns=["median", "lower", "upper"])
    hcw["median"] = capacity.median(axis=1)
    hcw["lower"] = capacity.quantile(q=0.025, axis=1)
    hcw["upper"] = capacity.quantile(q=0.975, axis=1)

    return hcw


hcw0 = summarise_frac_hcws(results0)
hcw1 = summarise_frac_hcws(results1)
hcw2 = summarise_frac_hcws(results2)
hcw3 = summarise_frac_hcws(results3)
hcw4 = summarise_frac_hcws(results4)

# percentage change in HCW time used from baseline (scenario 0)
hcw_diff1 = ((hcw1["median"] - hcw0["median"]) / hcw0["median"]) * 100
hcw_diff2 = ((hcw2["median"] - hcw0["median"]) / hcw0["median"]) * 100
hcw_diff3 = ((hcw3["median"] - hcw0["median"]) / hcw0["median"]) * 100
hcw_diff4 = ((hcw4["median"] - hcw0["median"]) / hcw0["median"]) * 100


# Make plot
fig = plt.figure()

gs=GridSpec(2,3) # 2 rows, 3 columns

ax1 = plt.subplot(gs[0, 0])  # First row, first column
ax2 = plt.subplot(gs[0,1]) # First row, second column
ax3=fig.add_subplot(gs[0,2]) # First row, third column
ax4=fig.add_subplot(gs[1,:]) # Second row, span all columns

fig, ax = plt.subplots()

ax.plot(hcw1.index, hcw_diff1, "-", color="C0")
# ax.fill_between(hcw1.index, hcw_diff1["lower"], hcw_diff1["upper"], color="C0", alpha=0.2)

ax.plot(hcw2.index, hcw_diff2, "-", color="C2")
# ax.fill_between(hcw2.index, hcw_diff2["lower"], hcw_diff2["upper"], color="C2", alpha=0.2)

ax.plot(hcw3.index, hcw_diff3, "-", color="C4")
# ax.fill_between(hcw3.index, hcw_diff3["lower"], hcw_diff3["upper"], color="C4", alpha=0.2)

ax.plot(hcw4.index, hcw_diff4, "-", color="C6")
# ax.fill_between(hcw4.index, hcw_diff4["lower"], hcw_diff4["upper"], color="C6", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.title("Percentage difference in healthcare worker time")
plt.ylabel("% difference in healthcare worker time")
plt.legend(["Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"])

plt.show()
