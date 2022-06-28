
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)


outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

# download all files (and get most recent [-1])
results0 = get_scenario_outputs("scenario0.py", outputspath)[-1]
results1 = get_scenario_outputs("scenario1.py", outputspath)[-1]
results2 = get_scenario_outputs("scenario2.py", outputspath)[-1]
results3 = get_scenario_outputs("scenario3.py", outputspath)[-1]
results4 = get_scenario_outputs("scenario4.py", outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results0)

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

# plot
# Make plot
fig, ax = plt.subplots()
ax.plot(capacity1.index, capacity1["mean"], "-", color="C3")
ax.fill_between(capacity1.index, capacity1["lower"], capacity1["upper"], color="C3", alpha=0.2)

# ax.plot(capacity2.index, capacity2["mean"], "-", color="C0")
# ax.fill_between(capacity2.index, capacity2["lower"], capacity2["upper"], color="C0", alpha=0.2)

ax.plot(capacity3.index, capacity3["mean"], "-", color="C2")
ax.fill_between(capacity3.index, capacity3["lower"], capacity3["upper"], color="C2", alpha=0.2)
fig.subplots_adjust(left=0.15)
plt.title("Fraction of overall healthcare worker time")
plt.ylabel("Fraction of overall healthcare worker time")
plt.legend(["Scenario 1", "Scenario 2", "Scenario 3"])

plt.show()
# ---------------------------------- HSI Events - TREATMENT ID -------------------------------------

years_of_simulation = 25


def summarise_treatment_counts(df_list, treatment_id):
    """ summarise the treatment counts across all draws/runs for one results folder
        requires a list of dataframes with all treatments listed with associated counts
    """
    number_runs = len(df_list)
    number_HSI_by_run = pd.DataFrame(index=np.arange(years_of_simulation), columns=np.arange(number_runs))
    column_names = [
        treatment_id + "_median",
        treatment_id + "_lower",
        treatment_id + "_upper"]
    out = pd.DataFrame(columns=column_names)

    for i in range(number_runs):
        number_HSI_by_run.iloc[:,i] = pd.Series(df_list[i].loc[:, treatment_id])

    out.iloc[:,0] = number_HSI_by_run.median(axis=1)
    out.iloc[:,1] = number_HSI_by_run.quantile(q=0.025, axis=1)
    out.iloc[:,2] = number_HSI_by_run.quantile(q=0.975, axis=1)

    return out


def treatment_counts(results_folder, module, key, column):
    info = get_scenario_info(results_folder)

    df_list = list()
    for draw in range(info['number_of_draws']):
        for run in range(info['runs_per_draw']):

            # check if anything contained in folder (some runs failed)
            folder = results_folder / str(draw) / str(run)
            p: os.DirEntry
            pickles = [p for p in os.scandir(folder) if p.name.endswith('.pickle')]
            if pickles:

                df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]

                new = df[['date', column]].copy()
                df_list.append(pd.DataFrame(new[column].to_list()))

    # for column in each df, get median
    # list of treatment IDs
    list_tx_id = list(df_list[0].columns)
    # results = pd.DataFrame(index=np.arange(len(list_tx_id)))
    results = pd.DataFrame(index=np.arange(years_of_simulation))

    for treatment_id in list_tx_id:
        tmp = summarise_treatment_counts(df_list, treatment_id)

        # append output to dataframe
        results = results.join(tmp)

    return results


tx_id0 = treatment_counts(results_folder=results0,
               module="tlo.methods.healthsystem.summary",
               key="HSI_Event",
               column="TREATMENT_ID")

tx_id1 = treatment_counts(results_folder=results1,
               module="tlo.methods.healthsystem.summary",
               key="HSI_Event",
               column="TREATMENT_ID")

tx_id2 = treatment_counts(results_folder=results2,
               module="tlo.methods.healthsystem.summary",
               key="HSI_Event",
               column="TREATMENT_ID")

tx_id3 = treatment_counts(results_folder=results3,
               module="tlo.methods.healthsystem.summary",
               key="HSI_Event",
               column="TREATMENT_ID")

tx_id4 = treatment_counts(results_folder=results4,
               module="tlo.methods.healthsystem.summary",
               key="HSI_Event",
               column="TREATMENT_ID")


