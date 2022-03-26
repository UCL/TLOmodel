"""
This file analyses active TB incidence rate from batch file (saved in the 'outputs' results_folder
"""

from pathlib import Path

import datetime
import pandas as pd
import matplotlib.pyplot as plt

from tlo.analysis.utils import (
    extract_results,
    extract_params,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

outputspath = Path("./outputs/lmu17@ic.ac.uk")

# Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("incidence_calibration.py", outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# ------------------------ EXTRACT NUMBER OF NEW ACTIVE TB CASES ---------------------#

new_active_tb = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=False
    ),
    collapse_columns=True,
)

new_active_tb.index = new_active_tb.index.year


# --------------------------------- EXTRACT PERSON-YEARS -----------------------------#

def get_person_years(py_):
    years = pd.to_datetime(py_['date']).dt.year
    py = pd.Series(dtype="int64", index=years)
    for year in years:
        tot_py = (
            (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["M"]).apply(pd.Series) +
            (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["F"]).apply(pd.Series)
        ).transpose()
        tot_py.index = tot_py.index.astype(int)
        py[year] = tot_py.sum().values[0]

    return py


person_years = summarize(
    extract_results(
        results_folder=results_folder,
        module='tlo.methods.demography',
        key='person_years',
        custom_generate_series=get_person_years,
        do_scaling=False
    ),
    collapse_columns=True,
)

# -------------------- CALCULATE & PLOT ACTIVE TB INCIDENCE RATE ---------------------- #

tb_active_incidence_rate = (new_active_tb / person_years) * 100000


def make_plot(
    model=None,
    model_low=None,
    model_high=None,
    data_name=None,
    data_mid=None,
    data_low=None,
    data_high=None,
    xlab=None,
    ylab=None,
    title_str=None,
):
    assert model is not None
    assert title_str is not None

    # Make plot
    fic, ax = plt.subplots()
    ax.plot(model.index, model.values, "-", color="r")

    if (model_low is not None) and (model_high is not None):
        ax.fill_between(model_low.index, model_low, model_high, color="r", alpha=0.2)

    if data_mid is not None:
        ax.plot(data_mid.index, data_mid.values, "-", color="b")

    if (data_low is not None) and (data_high is not None):
        ax.fill_between(data_low.index, data_low, data_high, color="b", alpha=0.2)

    if xlab is not None:
        ax.set_xlabel(xlab)

    if ylab is not None:
        ax.set_xlabel(ylab)

    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.title(title_str)
    plt.legend(["TLO", data_name])


resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")
xls_tb = pd.ExcelFile(resourcefilepath / "ResourceFile_TB.xlsx")

data_who_tb_2020 = pd.read_excel(resourcefilepath / 'ResourceFile_TB.xlsx', sheet_name='WHO_activeTB2020')
data_who_tb_2020.index = data_who_tb_2020["year"]
data_who_tb_2020 = data_who_tb_2020.drop(columns=["year"])

make_plot(
    title_str="Active TB Incidence Rate",
    model=tb_active_incidence_rate[0]["mean"],
    model_low=tb_active_incidence_rate[0]["lower"],
    model_high=tb_active_incidence_rate[0]["upper"],
    data_name="WHO Data",
    data_mid=data_who_tb_2020['incidence_per_100k'],
    data_low=data_who_tb_2020['incidence_per_100k_low'],
    data_high=data_who_tb_2020['incidence_per_100k_high'],
    xlab="Year",
    ylab="Active TB Incidence Rate (per 100k)",
)

plt.show()
