"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputspath' results_folder

if running locally need to parse log files:
tlo parse-log /Users/tmangal/PycharmProjects/TLOmodel/outputs/mihpsa_runs-2025-04-14T130655Z/0/0

this one has the amended log files:
mihpsa_runs-2025-05-14T100705Z

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
compute_summary_statistics
)
from tlo import Date

datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")


# %% Analyse results of runs

# Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("mihpsa_deaths_runs.py", outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder, draw=0)

# get basic information about the results
info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

scaling_factor = log['tlo.methods.population']['scaling_factor'].scaling_factor.values[0]


new_logs = ["POP_15_64_M",
                "POP_15_64_F",
                "POP_15_64",
                "HIV_PREV_15_64_M",
                "HIV_PREV_15_64_F",
                "HIV_PREV_15_64",
                "N_NewHIV_15_64_M",
                "N_NewHIV_15_64_F",
                "N_NewHIV_15_64",
                "prop_dx_15_64_M",
                "prop_dx_15_64_F",
                "prop_dx_15_64",
                "Of_dx_on_ART_15_64_M",
                "Of_dx_on_ART_15_64_F",
                "Of_dx_on_ART_15_64",
                "on_ART_and_VLS_15_64_M",
                "on_ART_and_VLS_15_64_F",
                "on_ART_and_VLS_15_64",
]


output_dict = {}

for log in new_logs:
    result = summarize(
        extract_results(
            results_folder,
            module="tlo.methods.hiv",
            key="mihpsa_15_64",
            column=log,
            index="date",
            do_scaling=False,
        ),
        collapse_columns=False,
        only_mean=True
    )

    for draw in result.columns:
        if draw not in output_dict:
            output_dict[draw] = pd.DataFrame()  # Initialise DataFrame for the draw if not exists

        output_dict[draw][log] = result[draw]

with pd.ExcelWriter(results_folder / "outputs_for_mihpsa_deaths.xlsx", engine='openpyxl') as writer:
    # Iterate over the dictionary and write each DataFrame to a new sheet
    for draw, df in output_dict.items():
        df = df.T  # Switch rows and columns
        # Writing each draw's DataFrame to a new sheet named after the draw
        df.to_excel(writer, sheet_name=f'Draw_{draw}', index=True)





mihpsa_age_breakdown = [
    "Num_HIV_15_34",
    "Num_HIV_15_34_M",
    "Num_HIV_15_34_F",
    "Num_HIV_35_49",
    "Num_HIV_35_49_M",
    "Num_HIV_35_49_F",
    "Num_HIV_50",
    "Num_HIV_50_M",
    "Num_HIV_50_F",
]


age_output_dict = {}

for log in mihpsa_age_breakdown:
    result = summarize(
        extract_results(
            results_folder,
            module="tlo.methods.hiv",
            key="mihpsa_age_breakdown",
            column=log,
            index="date",
            do_scaling=True,
        ),
        collapse_columns=False,
        only_mean=True
    )

    for draw in result.columns:
        if draw not in age_output_dict:
            age_output_dict[draw] = pd.DataFrame()  # Initialise DataFrame for the draw if not exists

        age_output_dict[draw][log] = result[draw]

with pd.ExcelWriter(results_folder / "age_outputs_for_mihpsa_deaths.xlsx", engine='openpyxl') as writer:
    # Iterate over the dictionary and write each DataFrame to a new sheet
    for draw, df in age_output_dict.items():
        df = df.T  # Switch rows and columns
        # Writing each draw's DataFrame to a new sheet named after the draw
        df.to_excel(writer, sheet_name=f'Draw_{draw}', index=True)




# ---------------------------------------------------------------------------------------------------

# DEATHS

results_deaths = compute_summary_statistics(extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death_MIHPSA",
    custom_generate_series=(
        lambda df: df.assign(
            year=df["date"].dt.year,
            age_group=df["age"].apply(
                lambda x: (
                    "child" if x < 15 else
                    "young_adult" if 15 <= x <= 34 else
                    "mid_adult" if 35 <= x <= 49 else
                    "older_adult"
                )
            ),
            has_had_art=~df["date_first_ART_initiation"].isna(),  # True if date_treated not NaT, else False
            has_reinitiated_art=~df["date_ART_reinitiation"].isna(),
        )
        .query("label == 'AIDS'")
        .groupby([
            "year", "label", "age_group", "sex",
            "hiv_status", "hiv_diagnosed", "art_status",
            'less_than_6months_since_art_start',
            'less_than_6months_since_art_reinitiation', 'less_than_6months_since_art_start_or_reinitiation',
            "aids_status", "aids_at_art_start", 'aids_at_art_reinitiation',
            "has_had_art", "has_reinitiated_art",
        ], dropna=False)
        .size()

    ),
    do_scaling=True,
), central_measure='mean', only_central=True)

# remove multi-index on columns
results_deaths = results_deaths.reset_index()
results_deaths = results_deaths.rename(columns={0: "num_deaths"})


# Then write to Excel
with pd.ExcelWriter(results_folder / "deaths_output.xlsx", engine='openpyxl') as writer:
    results_deaths.to_excel(writer, sheet_name='Sheet1', index=False)


############################################################################
# Get deaths by filters

# results already filtered by label=AIDS
def get_deaths_by_filters(
    df,
    age_group=None,
    hiv_diagnosed=None,
    art_status=None,
    less_than_6months_since_art_start=None,
    less_than_6months_since_art_reinitiation=None,
    less_than_6months_since_art_start_or_reinitiation=None,
    aids_status=None,
    aids_at_art_start=None,
    aids_at_art_reinitiation=None,
    has_had_art=None,
    has_reinitiated_art=None,
):

    df_filtered = df.copy()

    # Helper to filter column if argument provided
    def apply_filter(col, val):
        if val is not None:
            if isinstance(val, list):
                return df_filtered[col].isin(val)
            else:
                return df_filtered[col] == val
        return pd.Series(True, index=df_filtered.index)

    # Combine all filters including the new one
    filters = (
        apply_filter('age_group', age_group) &
        apply_filter('hiv_diagnosed', hiv_diagnosed) &
        apply_filter('art_status', art_status) &
        apply_filter('less_than_6months_since_art_start', less_than_6months_since_art_start) &
        apply_filter('less_than_6months_since_art_reinitiation', less_than_6months_since_art_reinitiation) &
        apply_filter('less_than_6months_since_art_start_or_reinitiation',
                     less_than_6months_since_art_start_or_reinitiation) &
        apply_filter('aids_status', aids_status) &
        apply_filter('aids_at_art_start', aids_at_art_start) &
        apply_filter('aids_at_art_reinitiation', aids_at_art_reinitiation) &
        apply_filter('has_had_art', has_had_art) &
        apply_filter('has_reinitiated_art', has_reinitiated_art)
    )

    df_filtered = df_filtered[filters]

    # Get all unique years from full df to keep full range
    years = sorted(df['year'].unique())

    # Group by year and sex and sum deaths
    grouped = (
        df_filtered.groupby(['year', 'sex'])['num_deaths']
        .sum()
        .unstack(fill_value=0)
        .rename(columns={'F': 'Female', 'M': 'Male'})
    )

    # Reindex to include all years and sexes with zeros where missing
    grouped = grouped.reindex(index=years, columns=['Female', 'Male'], fill_value=0)

    # Add total deaths column
    grouped['Total'] = grouped['Female'] + grouped['Male']

    # Now reorder columns exactly
    grouped = grouped[['Total', 'Male', 'Female']]

    return grouped


############
# Define all filter sets as a list of tuples: (name, filter_kwargs)
filter_definitions = [
    ('all', {}),

    ('no_art_no_dx', dict(
        hiv_diagnosed=False,
        art_status='not',
        less_than_6months_since_art_start=None,
        less_than_6months_since_art_reinitiation=None,
        less_than_6months_since_art_start_or_reinitiation=None,
        aids_status=None,
        aids_at_art_start=None,
        aids_at_art_reinitiation=None,
        has_had_art=False,
        has_reinitiated_art=False)),

    ('dx_without_art', dict(
        hiv_diagnosed=True,
        art_status='not',
        less_than_6months_since_art_start=None,
        less_than_6months_since_art_reinitiation=None,
        less_than_6months_since_art_start_or_reinitiation=None,
        aids_status=None,
        aids_at_art_start=None,
        aids_at_art_reinitiation=None,
        has_had_art=False,
        has_reinitiated_art=False)),

    ('on_art_less_6_months_aids', dict(
        hiv_diagnosed=True,
        art_status=['on_not_VL_suppressed', 'on_VL_suppressed'],
        less_than_6months_since_art_start=True,
        less_than_6months_since_art_reinitiation=None,
        less_than_6months_since_art_start_or_reinitiation=None,
        aids_status=None,
        aids_at_art_start=True,
        aids_at_art_reinitiation=None,
        has_had_art=True,
        has_reinitiated_art=None)),

    ('on_art_less_6_months_no_aids', dict(
        hiv_diagnosed=True,
        art_status=['on_not_VL_suppressed', 'on_VL_suppressed'],
        less_than_6months_since_art_start=True,
        less_than_6months_since_art_reinitiation=None,
        less_than_6months_since_art_start_or_reinitiation=None,
        aids_status=None,
        aids_at_art_start=False,
        aids_at_art_reinitiation=None,
        has_had_art=True,
        has_reinitiated_art=None)),

    ('on_art_less_6_months_reinitiation_aids', dict(
        hiv_diagnosed=True,
        art_status=['on_not_VL_suppressed', 'on_VL_suppressed'],
        less_than_6months_since_art_start=None,
        less_than_6months_since_art_reinitiation=True,
        less_than_6months_since_art_start_or_reinitiation=None,
        aids_status=None,
        aids_at_art_start=None,
        aids_at_art_reinitiation=True,
        has_had_art=True,
        has_reinitiated_art=True)),

    ('on_art_less_6_months_reinitiation_no_aids', dict(
        hiv_diagnosed=True,
        art_status=['on_not_VL_suppressed', 'on_VL_suppressed'],
        less_than_6months_since_art_start=None,
        less_than_6months_since_art_reinitiation=True,
        less_than_6months_since_art_start_or_reinitiation=None,
        aids_status=None,
        aids_at_art_start=None,
        aids_at_art_reinitiation=False,
        has_had_art=True,
        has_reinitiated_art=True)),

    ('on_art_lowVL', dict(
        hiv_diagnosed=True,
        art_status='on_VL_suppressed',
        less_than_6months_since_art_start=None,
        less_than_6months_since_art_reinitiation=None,
        less_than_6months_since_art_start_or_reinitiation=None,
        aids_status=None,
        aids_at_art_start=None,
        aids_at_art_reinitiation=None,
        has_had_art=True,
        has_reinitiated_art=None)),

    ('on_art_highVL', dict(
        hiv_diagnosed=True,
        art_status='on_not_VL_suppressed',
        less_than_6months_since_art_start=None,
        less_than_6months_since_art_reinitiation=None,
        less_than_6months_since_art_start_or_reinitiation=None,
        aids_status=None,
        aids_at_art_start=None,
        aids_at_art_reinitiation=None,
        has_had_art=True,
        has_reinitiated_art=None)),

    ('art_less_than_6Months_lowVL', dict(
        hiv_diagnosed=True,
        art_status='on_VL_suppressed',
        less_than_6months_since_art_start=None,
        less_than_6months_since_art_reinitiation=None,
        less_than_6months_since_art_start_or_reinitiation=True,
        aids_status=None,
        aids_at_art_start=None,
        aids_at_art_reinitiation=None,
        has_had_art=True,
        has_reinitiated_art=None)),

    ('art_less_than_6Months_highVL', dict(
        hiv_diagnosed=True,
        art_status='on_not_VL_suppressed',
        less_than_6months_since_art_start=None,
        less_than_6months_since_art_reinitiation=None,
        less_than_6months_since_art_start_or_reinitiation=True,
        aids_status=None,
        aids_at_art_start=None,
        aids_at_art_reinitiation=None,
        has_had_art=True,
        has_reinitiated_art=None)),

    ('art_more_than_6Months_lowVL', dict(
        hiv_diagnosed=True,
        art_status='on_VL_suppressed',
        less_than_6months_since_art_start=None,
        less_than_6months_since_art_reinitiation=None,
        less_than_6months_since_art_start_or_reinitiation=False,
        aids_status=None,
        aids_at_art_start=None,
        aids_at_art_reinitiation=None,
        has_had_art=True,
        has_reinitiated_art=None)),

    ('art_more_than_6Months_highVL', dict(
        hiv_diagnosed=True,
        art_status='on_not_VL_suppressed',
        less_than_6months_since_art_start=None,
        less_than_6months_since_art_reinitiation=None,
        less_than_6months_since_art_start_or_reinitiation=False,
        aids_status=None,
        aids_at_art_start=None,
        aids_at_art_reinitiation=None,
        has_had_art=True,
        has_reinitiated_art=None)),

    ('art_interr', dict(
        hiv_diagnosed=True,
        art_status='not',
        less_than_6months_since_art_start=None,
        less_than_6months_since_art_reinitiation=None,
        less_than_6months_since_art_start_or_reinitiation=None,
        aids_status=None,
        aids_at_art_start=None,
        aids_at_art_reinitiation=None,
        has_had_art=True,
        has_reinitiated_art=None)),

    ('art_curr_CD4_less_than200', dict(
        hiv_diagnosed=True,
        art_status=['on_not_VL_suppressed', 'on_VL_suppressed'],
        less_than_6months_since_art_start=None,
        less_than_6months_since_art_reinitiation=None,
        less_than_6months_since_art_start_or_reinitiation=None,
        aids_status=True,
        aids_at_art_start=None,
        aids_at_art_reinitiation=None,
        has_had_art=True,
        has_reinitiated_art=None)),

    ('art_curr_CD4_more_than200', dict(
        hiv_diagnosed=True,
        art_status=['on_not_VL_suppressed', 'on_VL_suppressed'],
        less_than_6months_since_art_start=None,
        less_than_6months_since_art_reinitiation=None,
        less_than_6months_since_art_start_or_reinitiation=None,
        aids_status=False,
        aids_at_art_start=None,
        aids_at_art_reinitiation=None,
        has_had_art=True,
        has_reinitiated_art=None)),

    ('art_less_than_6months_curr_aids', dict(
        hiv_diagnosed=True,
        art_status=['on_not_VL_suppressed', 'on_VL_suppressed'],
        less_than_6months_since_art_start=True,
        less_than_6months_since_art_reinitiation=None,
        less_than_6months_since_art_start_or_reinitiation=None,
        aids_status=True,
        aids_at_art_start=None,
        aids_at_art_reinitiation=None,
        has_had_art=True,
        has_reinitiated_art=None)),

    ('art_less_than_6months_no_aids', dict(
        hiv_diagnosed=True,
        art_status=['on_not_VL_suppressed', 'on_VL_suppressed'],
        less_than_6months_since_art_start=True,
        less_than_6months_since_art_reinitiation=None,
        less_than_6months_since_art_start_or_reinitiation=None,
        aids_status=False,
        aids_at_art_start=None,
        aids_at_art_reinitiation=None,
        has_had_art=True,
        has_reinitiated_art=None)),

    ('art_curr_no_aids', dict(
        hiv_diagnosed=True,
        art_status=['on_not_VL_suppressed', 'on_VL_suppressed'],
        less_than_6months_since_art_start=False,
        less_than_6months_since_art_reinitiation=None,
        less_than_6months_since_art_start_or_reinitiation=None,
        aids_status=True,
        aids_at_art_start=None,
        aids_at_art_reinitiation=None,
        has_had_art=True,
        has_reinitiated_art=None)),

    ('art_curr_aids', dict(
        hiv_diagnosed=True,
        art_status=['on_not_VL_suppressed', 'on_VL_suppressed'],
        less_than_6months_since_art_start=False,
        less_than_6months_since_art_reinitiation=None,
        less_than_6months_since_art_start_or_reinitiation=None,
        aids_status=False,
        aids_at_art_start=None,
        aids_at_art_reinitiation=None,
        has_had_art=True,
        has_reinitiated_art=None)),
]


# Generate and combine DataFrames per age group
def generate_wide_deaths_table(age_group_label):
    wide_dfs = []

    for label, kwargs in filter_definitions:
        df = get_deaths_by_filters(
            results_deaths,
            age_group=age_group_label,
            **kwargs
        )
        df = df.rename(columns={
            'Total': f'{label}_Total',
            'Male': f'{label}_Male',
            'Female': f'{label}_Female'
        }).reset_index()

        wide_dfs.append(df)

    # Merge all DataFrames on 'year'
    from functools import reduce
    combined = reduce(lambda left, right: pd.merge(left, right, on='year', how='outer'), wide_dfs)

    # Add age_group column once
    combined.insert(1, 'age_group', age_group_label)

    return combined

#  Create outputs
combined_young_adult = generate_wide_deaths_table('young_adult')
combined_mid_adult = generate_wide_deaths_table('mid_adult')
combined_older_adult = generate_wide_deaths_table('older_adult')

with pd.ExcelWriter(results_folder / "young_adult_deaths_output.xlsx", engine='openpyxl') as writer:
    combined_young_adult.to_excel(writer, sheet_name='Sheet1', index=False)

with pd.ExcelWriter(results_folder / "mid_adult_deaths_output.xlsx", engine='openpyxl') as writer:
    combined_mid_adult.to_excel(writer, sheet_name='Sheet1', index=False)

with pd.ExcelWriter(results_folder / "older_adult_deaths_output.xlsx", engine='openpyxl') as writer:
    combined_older_adult.to_excel(writer, sheet_name='Sheet1', index=False)
