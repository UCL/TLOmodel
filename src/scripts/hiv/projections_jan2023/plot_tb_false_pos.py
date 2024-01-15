
import datetime
from pathlib import Path
import os

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('default')
matplotlib.use('tkagg')

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import lacroix

from tlo.analysis.utils import (
    extract_results,
    get_scenario_outputs,
)

resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

# download all files (and get most recent [-1])
# todo check file paths to make sure scenario12 not selected
results0 = get_scenario_outputs("scenario0.py", outputspath)[-1]
results1 = get_scenario_outputs("scenario1.py", outputspath)[-1]
results2 = get_scenario_outputs("scenario2.py", outputspath)[-1]

# x = [1, 5, 1.5, 4]
# y = [9, 1.8, 8, 11]
# plt.scatter(x,y)
# plt.show()

plt.close()

plt.close('all')


def tb_false_pos_adults(results_folder):
    false_pos = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_false_positive",
        column="tbPropFalsePositiveAdults",
        index="date",
        do_scaling=False
    )

    false_pos.columns = false_pos.columns.get_level_values(0)
    false_pos_summary = pd.DataFrame(index=false_pos.index, columns=["median", "lower", "upper"])
    false_pos_summary["median"] = false_pos.median(axis=1)
    false_pos_summary["lower"] = false_pos.quantile(q=0.025, axis=1)
    false_pos_summary["upper"] = false_pos.quantile(q=0.975, axis=1)

    return false_pos_summary


false_pos0 = tb_false_pos_adults(results0)
false_pos1 = tb_false_pos_adults(results1)
false_pos2 = tb_false_pos_adults(results2)

berry = lacroix.colorList('CranRaspberry')  # ['#F2B9B8', '#DF7878', '#E40035', '#009A90', '#0054A4', '#001563']
baseline_colour = berry[5]  # '#001563'
sc1_colour = berry[3]  # '#009A90'
sc2_colour = berry[2]  # '#E40035'

years_num = false_pos0.index.year


fig, ax = plt.subplots()

ax.plot(years_num[12:23].values, false_pos0["median"][12:23].values, "-", color=baseline_colour)
ax.fill_between(years_num[12:23].values, false_pos0["lower"][12:23], false_pos0["upper"][12:23],
                color=baseline_colour, alpha=0.2)

ax.plot(years_num[12:23].values, false_pos1["median"][12:23].values, "-", color=sc1_colour)
ax.fill_between(years_num[12:23].values, false_pos1["lower"][12:23], false_pos1["upper"][12:23],
                color=sc1_colour, alpha=0.2)

ax.plot(years_num[12:23].values, false_pos2["median"][12:23].values, "-", color=sc2_colour)
ax.fill_between(years_num[12:23].values, false_pos2["lower"][12:23], false_pos2["upper"][12:23],
                color=sc2_colour, alpha=0.2)

plt.ylabel("Proportion false positives")

plt.xlabel("Year")
plt.ylim((0, 0.35))
plt.title("")

# handles for legend
l_baseline = mlines.Line2D([], [], color=baseline_colour, label="Baseline")
l_sc1 = mlines.Line2D([], [], color=sc1_colour, label="Constrained scale-up")
l_sc2 = mlines.Line2D([], [], color=sc2_colour, label="Unconstrained scale-up")

plt.legend(handles=[l_baseline, l_sc1, l_sc2])
fig.savefig(outputspath / "Tb_false_positives.png")

plt.show()


