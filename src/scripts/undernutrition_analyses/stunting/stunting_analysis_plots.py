from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import extract_results, get_scenario_outputs, summarize

# %%
scenario_filename = 'stunting_analysis_scenario.py'
outputspath = Path('./outputs')
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731


# %% Extract results
def __process(x):
    x = x.set_index('date')
    x.index = x.index.year
    age = [int(_.strip("()'").split(',')[0]) for _ in list(x.columns)]
    cat = [_.replace("'", "").strip("( )").split(',')[1].strip() for _ in list(x.columns)]
    x.columns = pd.MultiIndex.from_tuples(list(zip(age, cat)), names=('age', 'cat'))
    return x.stack().stack()


results = summarize(extract_results(results_folder,
                                    module="tlo.methods.stunting",
                                    key="prevalence",
                                    custom_generate_series=__process))

# %% Describe the mean distribution for the baseline case (with HealthSystem working)
r = results.loc[:, (0, "mean")].unstack().unstack().T.reset_index()

years_to_plot = [2010, 2011]
cats = ['HAZ>=-2', '-3<=HAZ<-2', 'HAZ<-3']

fig, axes = plt.subplots(1, len(years_to_plot), sharex=True, sharey=True)
for year, ax in zip(years_to_plot, axes):
    breakdown_this_year = r[['age', 'cat', year]].groupby(['age', 'cat'])[year].sum()\
        .groupby(level=0).apply(lambda x: x/x.sum())
    for cat in cats:
        this_cat_by_age = breakdown_this_year.loc[(slice(None), cat)]
        ax.bar(this_cat_by_age.index, this_cat_by_age.values, label=cat)
    ax.set_xlabel('Age')
    ax.set_ylabel('Proportion')
    ax.set_title(f'{year}')
    ax.legend()
plt.tight_layout()
plt.show()

# todo do stack using "bottom"
