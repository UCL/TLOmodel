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

# deaths due to TB (not including TB-HIV)
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
    results_deaths.to_excel(outputspath / "sample_summarised_deaths.xlsx", index=True)

    # select only cause AIDS_TB and AIDS_non_TB
    tmp = results_deaths.loc[results_deaths.cause == "TB"]

    # group deaths by year
    tmp2=pd.DataFrame(tmp.groupby(["year"]).sum())
    # divide each draw/run by the respective person-years from that run
    # need to reset index as they don't match exactly (date format)
    tmp3 = tmp2.reset_index(drop=True) / (person_years.reset_index(drop=True))

   tb_deaths = {}  # empty dict

tb_deaths["median_tb_deaths_rate_100kpy"] = (
                                                        tmp3.astype(float).quantile(0.5, axis=1)
                                                    ) * 100000
tb_deaths["median_tb_deaths_rate_100kpy"] = (
                                                       tmp3.astype(float).quantile(0.025, axis=1)
                                                   ) * 100000
tb_deaths["median_tb_deaths_rate_100kpy"] = (
                                                       tmp3.astype(float).quantile(0.975, axis=1)
                                                   ) * 100000
return tb_deaths


tb_deaths.to_excel(outputspath / "summarised_deaths.xlsx", index=True)


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

