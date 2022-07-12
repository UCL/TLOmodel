
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from tlo import Date

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

# Make plot
fig, ax = plt.subplots()
ax.plot(hcw0.index, hcw0["median"], "-", color="C3")
ax.fill_between(hcw0.index, hcw0["lower"], hcw0["upper"], color="C3", alpha=0.2)

ax.plot(hcw1.index, hcw1["median"], "-", color="C0")
ax.fill_between(hcw1.index, hcw1["lower"], hcw1["upper"], color="C0", alpha=0.2)

ax.plot(hcw2.index, hcw2["median"], "-", color="C2")
ax.fill_between(hcw2.index, hcw2["lower"], hcw2["upper"], color="C2", alpha=0.2)

ax.plot(hcw3.index, hcw3["median"], "-", color="C4")
ax.fill_between(hcw3.index, hcw3["lower"], hcw3["upper"], color="C4", alpha=0.2)

ax.plot(hcw4.index, hcw4["median"], "-", color="C6")
ax.fill_between(hcw4.index, hcw4["lower"], hcw4["upper"], color="C6", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.title("Fraction of overall healthcare worker time")
plt.ylabel("Fraction of overall healthcare worker time")
plt.legend(["Scenario 0", "Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"])

plt.show()

# ---------------------------------- HSI Events - TREATMENT ID -------------------------------------

years_of_simulation = 26


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
        if treatment_id in df_list[i].columns:
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


#] Make plot
fig, ax = plt.subplots()
ax.plot(tx_id0.index, tx_id0["Tb_Treatment_median"], "-", color="C3")

ax.plot(tx_id1.index, tx_id1["Tb_Treatment_median"], "-", color="C0")

ax.plot(tx_id2.index, tx_id2["Tb_Treatment_median"], "-", color="C2")

ax.plot(tx_id3.index, tx_id3["Tb_Treatment_median"], "-", color="C4")

ax.plot(tx_id4.index, tx_id4["Tb_Treatment_median"], "-", color="C6")

fig.subplots_adjust(left=0.15)
plt.title("Numbers of TB treatment initiation HSI events")
plt.ylabel("Numbers treated")
plt.legend(["Scenario 0", "Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"])

plt.show()


#---------------------------------------------------------------------------------------------------

def get_annual_num_appts_by_level(results_folder: Path) -> pd.DataFrame:
    """Return pd.DataFrame gives the (mean) simulated annual number of appointments of each type at each level."""

    TARGET_PERIOD = (Date(2010, 1, 1), Date(2035, 12, 31))

    def get_counts_of_appts(_df):
        """Get the mean number of appointments of each type being used each year at each level."""

        def unpack_nested_dict_in_series(_raw: pd.Series):
            return pd.concat(
                {
                  idx: pd.DataFrame.from_dict(mydict) for idx, mydict in _raw.iteritems()
                 }
             ).unstack().fillna(0.0).astype(int)

        return _df \
            .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'Number_By_Appt_Type_Code_And_Level'] \
            .pipe(unpack_nested_dict_in_series) \
            .mean(axis=0)  # mean over each year (row)

    return summarize(
        extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='HSI_Event',
                custom_generate_series=get_counts_of_appts,
                do_scaling=True
            ),
        only_mean=True,
        collapse_columns=True,
        ).unstack().astype(int)


hsi_0 = get_annual_num_appts_by_level(results_folder=results0)


#----------------------------------------------------------------------------------------------------------

cons2 = treatment_counts(results_folder=results2,
               module="tlo.methods.healthsystem.summary",
               key="Consumables",
               column="Item_Available")

cons2NA = treatment_counts(results_folder=results2,
               module="tlo.methods.healthsystem.summary",
               key="Consumables",
               column="Item_NotAvailable")

# Make plot of consumables availability
# item 175 adult tx
fig, ax = plt.subplots()
ax.plot(cons2.index, cons2["175_median"], "-", color="C3")
ax.fill_between(cons2.index, cons2["175_lower"], cons2["175_upper"], color="C3", alpha=0.2)

ax.plot(cons2NA.index, cons2NA["175_median"], "-", color="C0")
ax.fill_between(cons2NA.index, cons2NA["175_lower"], cons2NA["175_upper"], color="C0", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.title("Numbers of TB treatment consumables requested - scenario 2")
plt.ylabel("Numbers of consumables")
plt.legend(["Cons available", "Cons not available"])

plt.show()


# item 2671 "First-line ART regimen: adult"
fig, ax = plt.subplots()
ax.plot(cons2.index, cons2["2671_median"], "-", color="C3")
ax.fill_between(cons2.index, cons2["2671_lower"], cons2["2671_upper"], color="C3", alpha=0.2)

ax.plot(cons2NA.index, cons2NA["2671_median"], "-", color="C0")
ax.fill_between(cons2NA.index, cons2NA["2671_lower"], cons2NA["2671_upper"], color="C0", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.title("Numbers of First-line ART treatment consumables requested - scenario 2")
plt.ylabel("Numbers of consumables")
plt.legend(["Cons available", "Cons not available"])

plt.show()

