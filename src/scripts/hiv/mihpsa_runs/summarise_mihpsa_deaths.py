"""This file uses the results of the batch file to make some summary statistics.
"""

import datetime
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import itertools

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
results_folder = get_scenario_outputs("mihpsa_runs.py", outputspath)[-1]
scaling_factor = load_pickled_dataframes(results_folder, 0, 0, 'tlo.methods.population'
                                         )['tlo.methods.population']['scaling_factor']['scaling_factor'].values[0]


# Collect results from each draw/run
def extract_outputs(results_folder: Path,
                    module: str,
                    key: str,
                    column: str) -> pd.DataFrame:
    # get number of draws and numbers of runs
    info = get_scenario_info(results_folder)

    df_pop = pd.DataFrame()
    df_prev = pd.DataFrame()
    df_inf = pd.DataFrame()
    df_dx = pd.DataFrame()
    df_tx = pd.DataFrame()
    df_vs = pd.DataFrame()
    df_out = pd.DataFrame()

    # 5 draws, 1 run each
    run = 0
    for draw in range(info['number_of_draws']):
        # load the log file
        df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]
        # first column is mean

        tmp = pd.DataFrame(df[column].to_list(), columns=['pop', 'prev',
                                                          "inf", "dx",
                                                          "tx", "vs"])
        df_pop[draw] = tmp["pop"]
        df_prev[draw] = tmp["prev"]
        df_inf[draw] = tmp["inf"]
        df_dx[draw] = tmp["dx"]
        df_tx[draw] = tmp["tx"]
        df_vs[draw] = tmp["vs"]

    df_out["mean_pop"] = df_pop.mean(axis=1) * scaling_factor
    df_out["mean_prev"] = df_prev.mean(axis=1)
    df_out["mean_inf"] = df_inf.mean(axis=1) * scaling_factor
    df_out["mean_dx"] = df_dx.mean(axis=1)
    df_out["mean_tx"] = df_tx.mean(axis=1)
    df_out["mean_vs"] = df_vs.mean(axis=1)

    return df_out


# extract pop size for M, F and Total
baseline_m = extract_outputs(results_folder=results_folder,
                             module="tlo.methods.hiv",
                             key="hiv_baseline_outputs",
                             column="outputs_age15_64_M")

baseline_f = extract_outputs(results_folder=results_folder,
                             module="tlo.methods.hiv",
                             key="hiv_baseline_outputs",
                             column="outputs_age15_64_F")

baseline_all = extract_outputs(results_folder=results_folder,
                               module="tlo.methods.hiv",
                               key="hiv_baseline_outputs",
                               column="outputs_age15_64")


def extract_hiv_pop(results_folder: Path,
                    module: str,
                    key: str,
                    column: str) -> pd.DataFrame:
    # get number of draws and numbers of runs
    info = get_scenario_info(results_folder)

    df_pop = pd.DataFrame()
    df_out = pd.DataFrame()

    # 5 draws, 1 run each
    run = 0
    for draw in range(info['number_of_draws']):
        # load the log file
        df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]
        # first column is mean

        tmp = pd.DataFrame(df[column].to_list(), columns=['num_infected', 'num_not_diagnosed', 'num_dx_no_art',
                    'num_art_not_vs', 'num_art_vs',
                    'num_art_under_6mths_not_vs', 'num_art_under_6mths_vs',
                    'num_art_over_6mths_not_vs', 'num_art_over_6mths_vs',
                    'num_any_interruption'])

        df_pop[draw] = tmp["num_infected"]

    df_out["mean_pop"] = df_pop.mean(axis=1) * scaling_factor

    return df_out


hiv_15_24M = extract_hiv_pop(results_folder=results_folder,
                               module="tlo.methods.hiv",
                               key="hiv_detailed_outputs",
                               column="outputs_age15_24_M")


hiv_15_24F = extract_hiv_pop(results_folder=results_folder,
                               module="tlo.methods.hiv",
                               key="hiv_detailed_outputs",
                               column="outputs_age15_24_F")


hiv_25_34M = extract_hiv_pop(results_folder=results_folder,
                               module="tlo.methods.hiv",
                               key="hiv_detailed_outputs",
                               column="outputs_age25_34_M")

hiv_25_34F = extract_hiv_pop(results_folder=results_folder,
                               module="tlo.methods.hiv",
                               key="hiv_detailed_outputs",
                               column="outputs_age25_34_F")

hiv_35_44M = extract_hiv_pop(results_folder=results_folder,
                               module="tlo.methods.hiv",
                               key="hiv_detailed_outputs",
                               column="outputs_age35_44_M")

hiv_35_44F = extract_hiv_pop(results_folder=results_folder,
                               module="tlo.methods.hiv",
                               key="hiv_detailed_outputs",
                               column="outputs_age35_44_F")

hiv_45M = extract_hiv_pop(results_folder=results_folder,
                               module="tlo.methods.hiv",
                               key="hiv_detailed_outputs",
                               column="outputs_age45_100_M")

hiv_45F = extract_hiv_pop(results_folder=results_folder,
                               module="tlo.methods.hiv",
                               key="hiv_detailed_outputs",
                               column="outputs_age45_100_F")

# extract deaths by age-group
def calculate_counts(df):

    # create empty dataframe with all entries for output
    # Define lists of values for 'age_group', 'year', and 'sex'
    age_groups = ['15-24', '25-34', '35-44', '45+']
    years = list(range(2010, 2046))
    sexes = ['M', 'F']
    # Generate all combinations of 'age_group', 'year', and 'sex'
    combinations = list(itertools.product(age_groups, years, sexes))

    # Create DataFrame from combinations
    df_empty = pd.DataFrame(combinations, columns=['age_group', 'year', 'sex'])

    # Group by 'age_group', 'sex', and 'year', then count the number of entries
    df_total_deaths = df.groupby(['age_group', 'sex', 'year']).size().reset_index(name='total_deaths')

    # undiagnosed
    undx = df[df['hiv_diagnosed'] == False]
    df_undx = undx.groupby(['age_group', 'sex', 'year']).size().reset_index(name='undx')

    # diagnosed without ART initiation
    dx = df[(df['hiv_diagnosed'] == True) & (df['art_status'] == 'not')]
    df_dx = dx.groupby(['age_group', 'sex', 'year']).size().reset_index(name='dx')

    # on ART <6 months after first initiation and CD4<200 at ART start
    art_recent_aids = df[(df['hiv_diagnosed'] == True) &
                         (df['art_status'] != 'not') &
                         (df['on_ART_more_than_6months'] == False) &
                         (df['aids_at_art_start'] == True)]
    df_art_recent_aids = art_recent_aids.groupby(['age_group', 'sex', 'year']).size().reset_index(
        name='art_recent_aids')

    # on ART <6 months after first initiation and CD4>200 at ART start
    art_recent_no_aids = df[(df['hiv_diagnosed'] == True) &
                         (df['art_status'] != 'not') &
                         (df['on_ART_more_than_6months'] == False) &
                         (df['aids_at_art_start'] == False)]
    df_art_recent_no_aids = art_recent_no_aids.groupby(['age_group', 'sex', 'year']).size().reset_index(
        name='art_recent_no_aids')

    # on ART VL <1000
    art_VS= df[(df['hiv_diagnosed'] == True) &
                         (df['art_status'] == 'on_VL_suppressed')]
    df_art_VS = art_VS.groupby(['age_group', 'sex', 'year']).size().reset_index(
        name='art_VS')

    # on ART VL >1000
    art_not_VS= df[(df['hiv_diagnosed'] == True) &
                         (df['art_status'] == 'on_not_VL_suppressed')]
    df_art_not_VS = art_not_VS.groupby(['age_group', 'sex', 'year']).size().reset_index(
        name='art_not_VS')

    # on ART <6 months VL <1000
    art_recent_VL_low = df[(df['hiv_diagnosed'] == True) &
                         (df['art_status'] == 'on_VL_suppressed') &
                         (df['on_ART_more_than_6months'] == False)]
    df_art_recent_VL_low = art_recent_VL_low.groupby(['age_group', 'sex', 'year']).size().reset_index(
        name='art_recent_VL_low')

    # on ART <6 months VL >1000
    art_recent_VL_high = df[(df['hiv_diagnosed'] == True) &
                         (df['art_status'] == 'on_not_VL_suppressed') &
                         (df['on_ART_more_than_6months'] == False)]
    df_art_recent_VL_high = art_recent_VL_high.groupby(['age_group', 'sex', 'year']).size().reset_index(
        name='art_recent_VL_high')

    # on ART >6 months VL <1000
    art_VL_low = df[(df['hiv_diagnosed'] == True) &
                           (df['art_status'] == 'on_VL_suppressed') &
                           (df['on_ART_more_than_6months'] == True)]
    df_art_VL_low = art_VL_low.groupby(['age_group', 'sex', 'year']).size().reset_index(
        name='art_VL_low')

    # on ART >6 months VL >1000
    art_VL_high = df[(df['hiv_diagnosed'] == True) &
                            (df['art_status'] == 'on_not_VL_suppressed') &
                            (df['on_ART_more_than_6months'] == True)]
    df_art_VL_high = art_VL_high.groupby(['age_group', 'sex', 'year']).size().reset_index(
        name='art_VL_high')

    # on ART, CD4 <200 at time of death
    art_aids_at_death = df[(df['hiv_diagnosed'] == True) &
                            (df['art_status'] != 'not') &
                            (df['aids_status'] == True)]
    df_art_aids_at_death = art_aids_at_death.groupby(['age_group', 'sex', 'year']).size().reset_index(
        name='art_aids_at_death')

    # on ART, CD4 >200 at time of death
    art_no_aids_at_death = df[(df['hiv_diagnosed'] == True) &
                            (df['art_status'] != 'not') &
                            (df['aids_status'] == False)]
    df_art_no_aids_at_death = art_no_aids_at_death.groupby(['age_group', 'sex', 'year']).size().reset_index(
        name='art_no_aids_at_death')

    # on ART <6 months, CD4 <200 at time of death
    art_recent_aids_at_death = df[(df['hiv_diagnosed'] == True) &
                           (df['art_status'] != 'not') &
                           (df['aids_status'] == True) &
                           (df['on_ART_more_than_6months'] == False)]
    df_art_recent_aids_at_death = art_recent_aids_at_death.groupby(['age_group', 'sex', 'year']).size().reset_index(
        name='art_recent_aids_at_death')

    # on ART <6 months, CD4 >200 at time of death
    art_recent_no_aids_at_death = df[(df['hiv_diagnosed'] == True) &
                              (df['art_status'] != 'not') &
                              (df['aids_status'] == False) &
                              (df['on_ART_more_than_6months'] == False)]
    df_art_recent_no_aids_at_death = art_recent_no_aids_at_death.groupby(['age_group', 'sex', 'year']).size().reset_index(
        name='art_recent_no_aids_at_death')

    # on ART >6 months, CD4 <200 at time of death
    art_long_aids_at_death = df[(df['hiv_diagnosed'] == True) &
                           (df['art_status'] != 'not') &
                           (df['aids_status'] == True) &
                           (df['on_ART_more_than_6months'] == True)]
    df_art_long_aids_at_death = art_long_aids_at_death.groupby(['age_group', 'sex', 'year']).size().reset_index(
        name='art_long_aids_at_death')

    # on ART >6 months, CD4 >200 at time of death
    art_long_no_aids_at_death = df[(df['hiv_diagnosed'] == True) &
                              (df['art_status'] != 'not') &
                              (df['aids_status'] == False) &
                              (df['on_ART_more_than_6months'] == True)]
    df_art_long_no_aids_at_death = art_long_no_aids_at_death.groupby(['age_group', 'sex', 'year']).size().reset_index(
        name='art_long_no_aids_at_death')

    # List of DataFrames
    dataframes = [df_total_deaths,
                  df_undx, df_dx,
                  df_art_recent_aids, df_art_recent_no_aids,
                  df_art_VS, df_art_not_VS,
                  df_art_recent_VL_low, df_art_recent_VL_high,
                  df_art_VL_low, df_art_VL_high,
                  df_art_aids_at_death, df_art_no_aids_at_death,
                  df_art_recent_aids_at_death, df_art_recent_no_aids_at_death,
                  df_art_long_aids_at_death, df_art_long_no_aids_at_death]

    # Merge the DataFrames iteratively
    merged_df = df_empty
    for df in dataframes[0:]:
        merged_df = pd.merge(merged_df, df, on=['age_group', 'sex', 'year'], how='outer')

    # fill any nans with 0
    merged_df = merged_df.fillna(0)

    return merged_df


# Function to element-wise multiply values in each column by scaling_factor, excluding specified columns
def multiply_column_values(column, scaling_factor):
    if column.name not in ['age_group', 'sex', 'year']:
        return column * scaling_factor  # Multiply each value by the scaling factor
    else:
        return column  # Return the original column values for 'age_group', 'sex', and 'year'


def extract_deaths(results_folder: Path) -> pd.DataFrame:
    module = "tlo.methods.demography"
    key = "death"

    # get number of draws and numbers of runs
    info = get_scenario_info(results_folder)
    dfs = []

    # 5 runs
    draw = 0
    for run in range(info['runs_per_draw']):
        # load the log file
        df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]

        # Define the age groups
        bins = [15, 25, 35, 45, float('inf')]
        labels = ['15-24', '25-34', '35-44', '45+']

        # Create the 'age_group' column based on 'age' values
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

        # sum AIDS deaths by year and age-group
        keep = (df.label == "AIDS")
        aids_deaths = df.loc[keep].copy()
        aids_deaths["year"] = aids_deaths["date"].dt.year  # count by year
        # remove children
        aids_deaths = aids_deaths.loc[aids_deaths.age >= 15]

        # for first draw, create the dataframe
        # subsequent draws need to be added (excluding age_group sex, year)
        df_out = calculate_counts(aids_deaths)

        # Append DataFrame to the list
        dfs.append(pd.DataFrame(df_out))

    # resulting dataframe represents population 5*150_000
    result = dfs[0].copy()
    for df in dfs[1:]:
        for col in result.columns:
            if col not in ['age_group', 'sex', 'year']:
                result[col] += df[col]

    # scale the results
    result = result.apply(multiply_column_values, scaling_factor=scaling_factor/5)

    return result


deaths = extract_deaths(results_folder=results_folder)

# write to excel
with pd.ExcelWriter(outputspath / ("MIHPSA_deaths_Feb2024" + ".xlsx"), engine='openpyxl') as writer:
    baseline_m.to_excel(writer, sheet_name='baseline_m', index=False)
    baseline_f.to_excel(writer, sheet_name='baseline_f', index=False)
    baseline_all.to_excel(writer, sheet_name='baseline_all', index=False)
    deaths.to_excel(writer, sheet_name='deaths', index=False)
writer.save()
