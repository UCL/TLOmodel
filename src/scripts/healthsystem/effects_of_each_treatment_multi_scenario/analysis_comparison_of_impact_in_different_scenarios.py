"""This file produces the plots for each set of results from "scenario_effect_of_each_treatment" and plots
any comparisons of the results generated in the different scenarios."""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.scripts.healthsystem.finding_effects_of_each_treatment.analysis_effect_of_each_treatment import (
    apply,
)
from tlo.analysis.utils import get_color_short_treatment_id

outputs = Path("./outputs/tbh03@ic.ac.uk/")

results_folders = {
    'Defaults':
        outputs / "scenario_effect_of_each_treatment_min-2023-02-03T215051Z",
    'Consumables available and Full healthcare seeking':
        outputs / "scenario_effect_of_each_treatment_max-2023-02-03T215023Z",
}


# Generate the results from each results_folder, and capture any results returned.
results = dict()
for k, v in results_folders.items():
    results[k] = apply(results_folder=v, output_folder=v, rtn_results=True)


# %% Make a comparison of the impact (Dalys_averted) of each (coarse) TREATMENT_ID in the different Scenarios.

# Reformat and scale the captured output for plotting:
num_dalys_averted = {
    k: results[k]['num_dalys_averted'].drop(['*']).sort_values('mean') / 1e6
    for k, v in results.items()
}
df = pd.concat(num_dalys_averted.values(), keys=num_dalys_averted.keys())

# Find order of treatment_id bars: lowest-->highest of mean value in the 'Defaults' results.
order_of_treatment_id = df.loc['Defaults']['mean'].sort_values().index.to_list()
yy = dict(zip(order_of_treatment_id, 1.0 + np.arange(0, len(order_of_treatment_id))))  # associate a position on the
#                                                                                         y-axis for each treatment_id

_scenarios = df.index.levels[0]  # the different scenarios

fig, ax = plt.subplots(figsize=(10, 10))
for _tr, ypos_central in yy.items():
    for ypos_adjustment, _sc in zip([-0.15, 0.15], _scenarios):
        _dat = df.loc[(_sc, _tr)]
        level = _dat['mean']
        error = np.array([(_dat['mean'] - _dat['lower'], _dat['upper'] - _dat['mean'])]).T

        ax.barh(
            y=ypos_central + ypos_adjustment,
            width=level,
            xerr=error,
            height=0.25,
            color=get_color_short_treatment_id(_tr),
            alpha=(1.0 if _sc == 'Defaults' else 0.5),
            edgecolor=('k' if _sc == 'Defaults' else 'r'),
            linewidth=2,
        )
ax.set_yticks(list(yy.values()))
ax.set_yticklabels(labels=list(yy.keys()))
ax.yaxis.set_tick_params(labelsize=14)
ax.set_title('Impact of Each TREATMENT_ID', fontsize=30)
ax.set_ylabel('TREATMENT_ID', fontsize=20)
ax.set_xlabel('Number of DALYS Averted (/1e6)', fontsize=20)
ax.set_xlim(0, 10)
ax.set_ylim(0, 22.5)
ax.grid()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.show()
