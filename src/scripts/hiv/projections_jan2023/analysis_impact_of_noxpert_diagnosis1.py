"""
Extracts DALYs and mortality from the TB module
 """
# commands for running the analysis script in the terminal
# python src/scripts/hiv/projections_jan2023/analysis_impact_of_noxpert_diagnosis1.py --scenario-outputs-folder outputs/nic503@york.ac.uk --show-figures
# python src/scripts/hiv/projections_jan2023/analysis_impact_of_noxpert_diagnosis.py --scenario-outputs-folder outputs/nic503@york.ac.uk --save-figures

import datetime
from pathlib import Path

#import lacroix
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import statsmodels.api as sm
from tlo.analysis.utils import (
    extract_results,
    get_scenario_outputs,
)

resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")
outputspath = Path("./outputs/nic503@york.ac.uk")
results_folder = get_scenario_outputs('scenario_impact_noXpert_diagnosis.py', outputspath)[-1]

def get_person_years(_df):
    """ extract person-years for each draw/run
    sums across men and women
    will skip column if particular run has failed
    """
    years = pd.to_datetime(_df["date"]).dt.year
    py = pd.Series(dtype="int64", index=years)
    for year in years:
        tot_py = (
            (_df.loc[pd.to_datetime(_df["date"]).dt.year == year]["M"]).apply(pd.Series) +
            (_df.loc[pd.to_datetime(_df["date"]).dt.year == year]["F"]).apply(pd.Series)
        ).transpose()
        py[year] = tot_py.sum().values[0]

    py.index = pd.to_datetime(years, format="%Y")

    return py

py0 = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="person_years",
    custom_generate_series=get_person_years,
    do_scaling=False
)
def summarise_tb_deaths(results_folder, person_years):
    results_deaths = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df["date"].dt.year).groupby(
                ["year", "cause"])["person_id"].count()
            ),
        do_scaling=False,
    )
    # removes multi-index
    results_deaths = results_deaths.reset_index()

    # select only cause AIDS_TB and AIDS_non_TB
    #tmp = results_deaths.loc[results_deaths.cause == "TB"]
  #  tmp = pd.DataFrame(results_deaths.loc[results_deaths.cause])
    tmp = results_deaths.loc[results_deaths['cause'].isin(["AIDS_TB", "AIDS_non_TB", "TB"])]

    # group deaths by year
    tmp2 = pd.DataFrame(tmp.groupby(["year", "cause"]).sum())
    tmp2.to_excel(outputspath / "my_summarised_deaths.xlsx", index=True)

    # # divide each draw/run by the respective person-years from that run
    # # need to reset index as they don't match exactly (date format)
    # tmp3 = tmp2.reset_index(drop=True) / (person_years.reset_index(drop=True))
    #
    # tb_deaths = {}  # empty dict
    #
    # tb_deaths["median_tb_deaths_rate_100kpy"] = (
    #                                                 tmp3.astype(float).quantile(0.5, axis=1)
    #                                             ) * 100000
    # tb_deaths["lower_tb_deaths_rate_100kpy"] = (
    #                                                tmp3.astype(float).quantile(0.025, axis=1)
    #                                            ) * 100000
    # tb_deaths["upper_tb_deaths_rate_100kpy"] = (
    #                                                tmp3.astype(float).quantile(0.975, axis=1)
    #                                            ) * 100000
    #
    # return tb_deaths


# results_deaths = extract_results(
#     results_folder,
#     module="tlo.methods.demography",
#     key="death",
#     custom_generate_series=(
#         lambda df: df.assign(year=df["date"].dt.year).groupby(
#             ["year", "cause"])["person_id"].count()
#     ),
#     do_scaling=False,
#
# )
tb_deaths0 = summarise_tb_deaths(results_folder, py0)
tb_death=pd.DataFrame(tb_deaths0)
tb_deaths0.to_excel (outputspath / "sample_summarised_deaths.xlsx", index=True)

import pandas as pd
import numpy as np


def compute_difference_in_deaths_across_runs(total_deaths, scenario_info, output_file):
    deaths_difference_by_run = [
        total_deaths[0][run_number]["Total"] - total_deaths[1][run_number]["Total"]
        for run_number in range(scenario_info["runs_per_draw"])
    ]

    # Compute mean of deaths difference across runs
    mean_difference = np.mean(deaths_difference_by_run)

    # Create a DataFrame to store the results
    result_df = pd.DataFrame({
        "Difference in Deaths": deaths_difference_by_run
    })

    # Save results to Excel
    result_df.to_excel(outputspath, index=False)

    return mean_difference

# multiply by scaling factor to get numbers of expected infections

# get scaling factor for numbers of tests performed and treatments requested
# scaling factor 145.39609
# sf = extract_results(
#     results_folder,
#     module="tlo.methods.population",
#     key="scaling_factor",
#     column="scaling_factor",
#     index="date",
#     do_scaling=False)

