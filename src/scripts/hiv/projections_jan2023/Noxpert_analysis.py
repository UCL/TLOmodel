"""This file uses the results of the scenario runs to generate plots

*1 Epi outputs (incidence and mortality)
# python src/scripts/hiv/projections_jan2023/Noxpert_analysis.py --scenario-outputs-folder outputs/nic503@york.ac.uk --show-figures
"""

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

# download all files (and get most recent [-1])
results_folder = get_scenario_outputs("scenario_impact_noXpert_diagnosis.py", outputspath)[-1]


def num_deaths_aids(results_folder):
    extract_deaths = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df["date"].dt.year).groupby(
                ["year", "cause"])["person_id"].count()
        ),
        do_scaling=True,
    )
    # removes multi-index
    extract_deaths = extract_deaths.reset_index()

    # select only cause AIDS_TB and AIDS_non_TB
    num_aids_deaths = extract_deaths.loc[(extract_deaths.year >= 2023)]

    # select years 2023-2035
    num_aids_deaths = num_aids_deaths.loc[
        (num_aids_deaths.cause == "AIDS_TB") | (num_aids_deaths.cause == "AIDS_non_TB")
        ]

    # group deaths by year
    sum_aids_deaths = pd.DataFrame(num_aids_deaths.groupby(["year"]).sum())

    return(sum_aids_deaths)

#
# # differences in deaths
# # sc1_aids_deaths = num_deaths_aids(results1)
# # sc2_aids_deaths = num_deaths_aids(results2)
#
# # sum columns to get total deaths over full time-period by run
# sum_columns_aids_deaths1 = sc1_aids_deaths.sum(axis=0)
# sum_columns_aids_deaths2 = sc2_aids_deaths.sum(axis=0)
#
#
# # extract differences in number deaths by run
# diff = sum_columns_aids_deaths1.subtract(sum_columns_aids_deaths2, fill_value=0)

# # summarise differences in number deaths
# median_aids_deaths = diff.median()
# lower_aids_deaths = diff.quantile(q=0.025)
# upper_aids_deaths = diff.quantile(q=0.975)
