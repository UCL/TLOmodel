"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputspath' results_folder

"""

import datetime
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

# %% Analyse results of runs

# 0) Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("baseline_runs.py", outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# choose which draw to summarise / visualise
draw = 0

# %% extract results
# Load and format model results (with year as integer):

writer = pd.ExcelWriter(outputspath / ("MIHPSA_outputs" + ".xlsx"))


# ---------------------------------- write outputs to excel ---------------------------------- #
# only one run per draw so take mean only

def write_to_excel(key, column_name, scaling=True):
    out = summarize(
        extract_results(
            results_folder,
            module="tlo.methods.hiv",
            key=key,
            column=column_name,
            index="date",
            do_scaling=scaling,
        ),
        # collapse_columns=True,
        only_mean=True,
    )
    out.index = out.index.year
    out.to_excel(writer, sheet_name=column_name)
    writer.save()

# ---------------------------------- population ---------------------------------- #

write_to_excel("summary_inc_and_prev_for_adults_and_children_and_fsw", "pop_male_15plus")
write_to_excel("summary_inc_and_prev_for_adults_and_children_and_fsw", "pop_female_15plus")
write_to_excel("summary_inc_and_prev_for_adults_and_children_and_fsw", "pop_child")
write_to_excel("summary_inc_and_prev_for_adults_and_children_and_fsw", "pop_total")

# ---------------------------------- plhiv ---------------------------------- #

write_to_excel("summary_inc_and_prev_for_adults_and_children_and_fsw", "male_plhiv_15plus")
write_to_excel("summary_inc_and_prev_for_adults_and_children_and_fsw", "female_plhiv_15plus")
write_to_excel("summary_inc_and_prev_for_adults_and_children_and_fsw", "child_plhiv")
write_to_excel("summary_inc_and_prev_for_adults_and_children_and_fsw", "total_plhiv")

# ---------------------------------- new infections ---------------------------------- #

write_to_excel("infections_by_2age_groups_and_sex", "n_new_infections_male_1549")
write_to_excel("infections_by_2age_groups_and_sex", "n_new_infections_female_1524")
write_to_excel("infections_by_2age_groups_and_sex", "n_new_infections_female_2549")
write_to_excel("summary_inc_and_prev_for_adults_and_children_and_fsw", "n_new_infections_adult_1549")

# ---------------------------------- hiv prevalence ---------------------------------- #

write_to_excel("infections_by_2age_groups_and_sex", "male_prev_1524", scaling=False)
write_to_excel("infections_by_2age_groups_and_sex", "male_prev_2549", scaling=False)
write_to_excel("infections_by_2age_groups_and_sex", "female_prev_1524", scaling=False)
write_to_excel("infections_by_2age_groups_and_sex", "female_prev_2549", scaling=False)

write_to_excel("summary_inc_and_prev_for_adults_and_children_and_fsw", "hiv_prev_child", scaling=False)
write_to_excel("summary_inc_and_prev_for_adults_and_children_and_fsw", "hiv_prev_fsw", scaling=False)

write_to_excel("infections_by_2age_groups_and_sex", "total_prev", scaling=False)

# ---------------------------------- hiv incidence ---------------------------------- #

write_to_excel("infections_by_2age_groups_and_sex", "male_inc_1524", scaling=False)
write_to_excel("infections_by_2age_groups_and_sex", "male_inc_2549", scaling=False)
write_to_excel("infections_by_2age_groups_and_sex", "female_inc_1524", scaling=False)
write_to_excel("infections_by_2age_groups_and_sex", "female_inc_2549", scaling=False)
write_to_excel("summary_inc_and_prev_for_adults_and_children_and_fsw", "hiv_adult_inc_1549", scaling=False)

# ---------------------------------- plhiv aware of status ---------------------------------- #

write_to_excel("hiv_program_coverage", "dx_adult", scaling=False)

# ---------------------------------- total tests performed on adults ---------------------------------- #

write_to_excel("hiv_program_coverage", "number_adults_tested")

# ---------------------------------- proportion adults testing positive ---------------------------------- #

write_to_excel("hiv_program_coverage", "testing_yield", scaling=False)

# ---------------------------------- plhiv on art ---------------------------------- #

write_to_excel("hiv_program_coverage", "n_on_art_male_15plus")
write_to_excel("hiv_program_coverage", "n_on_art_female_15plus")
write_to_excel("hiv_program_coverage", "n_on_art_children")
write_to_excel("hiv_program_coverage", "n_on_art_total")

# ---------------------------------- art coverage ---------------------------------- #

write_to_excel("hiv_program_coverage", "art_coverage_child", scaling=False)

# ---------------------------------- adult art and virally suppressed ---------------------------------- #

write_to_excel("hiv_program_coverage", "art_coverage_adult_VL_suppression", scaling=False)

# ---------------------------------- M circumcised ---------------------------------- #

write_to_excel("hiv_program_coverage", "prop_men_circ", scaling=False)

# ---------------------------------- AIDS deaths ---------------------------------- #
# returns floats because of scaling
results_deaths = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=(
        lambda df: df.assign(year=df["date"].dt.year).groupby(
            ["year", "cause"])["person_id"].count()
    ),
    do_scaling=True,
)
results_deaths = results_deaths.reset_index()

aids_tb_deaths_table = results_deaths.loc[
    (results_deaths.cause == "AIDS_TB") | (results_deaths.cause == "AIDS_non_TB")
    ]
# flatten multi-index
aids_tb_deaths_table.reset_index(drop=True, inplace=True)
aids_deaths = pd.DataFrame(aids_tb_deaths_table.groupby(["year"]).sum()).reset_index()
aids_deaths.to_excel(writer, sheet_name="aids_deaths")
writer.save()

# get individual deaths
# todo note differences between cause and label for deaths
deaths_detailed = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=(
        lambda df_: df_.assign(
            year=df_['date'].dt.year
        ).groupby(['sex', 'year', 'age', 'label'])['person_id'].count()
    ),
    do_scaling=True,
)
deaths_detailed = deaths_detailed.reset_index()
deaths_detailed_table = deaths_detailed.loc[
    (deaths_detailed.label == "AIDS")
    ]
# flatten multi-index
deaths_detailed_table.reset_index(drop=True, inplace=True)

deathsM15 = deaths_detailed_table.loc[
    (deaths_detailed_table.sex == "M") &
    (deaths_detailed_table.age >= 15)]
deathsM15 = pd.DataFrame(deathsM15.groupby(["year"]).sum()).reset_index()
deathsM15.to_excel(writer, sheet_name="deathsM15")
writer.save()

deathsF15 = deaths_detailed_table.loc[
    (deaths_detailed_table.sex == "F") &
    (deaths_detailed_table.age >= 15)]
deathsF15 = pd.DataFrame(deathsF15.groupby(["year"]).sum()).reset_index()
deathsF15.to_excel(writer, sheet_name="deathsF15")
writer.save()

deaths_children = deaths_detailed_table.loc[
    (deaths_detailed_table.age < 15)]
deaths_children = pd.DataFrame(deaths_children.groupby(["year"]).sum()).reset_index()
deaths_children.to_excel(writer, sheet_name="deaths_children")
writer.save()

# ---------------------------------- total deaths ---------------------------------- #
# returns floats because of scaling
total_deaths = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=(
        lambda df: df.assign(year=df["date"].dt.year).groupby(
            ["year"])["person_id"].count()
    ),
    do_scaling=True,
)
total_deaths.to_excel(writer, sheet_name="total_deaths")
writer.save()

# total deaths by sex
deaths_detailed = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=(
        lambda df_: df_.assign(
            year=df_['date'].dt.year
        ).groupby(['sex', 'year'])['person_id'].count()
    ),
    do_scaling=True,
)

# flatten multi-index
deaths_detailed = deaths_detailed.reset_index()
deaths_detailed.reset_index(drop=True, inplace=True)

deathsM = deaths_detailed.loc[
    (deaths_detailed.sex == "M")]
deathsM.to_excel(writer, sheet_name="total_deathsM")
writer.save()

deathsF = deaths_detailed.loc[
    (deaths_detailed.sex == "F")]
deathsF.to_excel(writer, sheet_name="total_deathsF")
writer.save()





