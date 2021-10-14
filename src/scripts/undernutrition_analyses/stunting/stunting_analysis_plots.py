from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import extract_results, get_scenario_outputs, summarize

# %%
scenario_filename = 'analysis_stunting.py'
outputspath = Path('./outputs')
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731


# log = load_pickled_dataframes(results_folder)
# x = log['tlo.methods.stunting']['prevalence']

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

notstunt = r.loc[r.cat == 'HAZ>=-2', ['age', 2010]]
stuntnsev = r.loc[r.cat == '-3<=HAZ<-2', ['age', 2010]]
stuntsev = r.loc[r.cat == '-3<=HAZ', ['age', 2010]]

plt.bar(notstunt.age, notstunt[2010])
plt.bar(stuntnsev.age, stuntnsev[2010], bottom=notstunt[2010])
plt.bar(stuntsev.age, stuntsev[2010], bottom=stuntnsev[2010])
plt.tight_layout()
plt.show()
