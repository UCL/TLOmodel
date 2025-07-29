

import datetime
from pathlib import Path

# import lacroix
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from typing import Tuple

from tlo import Date

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
    make_age_grp_lookup,
    make_age_grp_types,
)

outputspath = Path("./outputs")

# Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("hiv_program_simplification", outputspath)[-1]


# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

#              module_param            value
# draw
# 0     Hiv:type_of_scaleup             none
# 1     Hiv:type_of_scaleup  reduce_HIV_test
# 2     Hiv:type_of_scaleup        remove_VL
# 3     Hiv:type_of_scaleup       remove_IPT
# 4     Hiv:type_of_scaleup       target_IPT
# 5     Hiv:type_of_scaleup    increase_6MMD
# 6     Hiv:type_of_scaleup       remove_all


# number HIV tests
hiv_tests = extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        column="number_adults_tested",
        index="date",
        do_scaling=False
)

# hiv testing rate
testing_rate = extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        column="per_capita_testing_rate",
        index="date",
        do_scaling=False
)

# FSW PY on PrEP
prop_fsw_prep = extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        column="PY_PREP_ORAL_FSW",
        index="date",
        do_scaling=False
)

prop_agyw_prep = extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        column="PY_PREP_ORAL_AGYW",
        index="date",
        do_scaling=False
)

# prop men circ
prop_men_circ = extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        column="prop_men_circ",
        index="date",
        do_scaling=False
)

num_men_circ = extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        column="N_NewVMMC",
        index="date",
        do_scaling=False
)


# num tdf tests
num_tdf_tests = extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        column="n_tdf_tests_performed",
        index="date",
        do_scaling=False
)

# 6MMD

mmd_M = extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="arv_dispensing_intervals",
        column="group=adult_male|interval=6+",
        index="date",
        do_scaling=False
)

mmd_F = extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="arv_dispensing_intervals",
        column="group=adult_female|interval=6+",
        index="date",
        do_scaling=False
)


TARGET_PERIOD = (Date(2025, 1, 1), Date(2034, 1, 1))


def get_parameter_names_from_scenario_file() -> Tuple[str]:
    """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
    from scripts.hiv.program_simplification.analysis_hiv_program_simplification import (
        HIV_Progam_Elements,
    )
    e = HIV_Progam_Elements()
    return tuple(e._scenarios.keys())

def set_param_names_as_column_index_level_0(_df):
    """Set the columns index (level 0) as the param_names."""
    ordered_param_names_no_prefix = {i: x for i, x in enumerate(param_names)}
    names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
    assert len(names_of_cols_level0) == len(_df.columns.levels[0])
    _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
    return _df

param_names = get_parameter_names_from_scenario_file()

# DALYs
def get_num_dalys(_df):
    """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD).
    Throw error if not a record for every year in the TARGET PERIOD (to guard against inadvertently using
    results from runs that crashed mid-way through the simulation.
    """
    years_needed = [i.year for i in TARGET_PERIOD]
    assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
    return pd.Series(
        data=_df
        .loc[_df.year.between(*years_needed)]
        .drop(columns=['date', 'sex', 'age_range', 'year'])
        .sum().sum()
    )


num_dalys = extract_results(
    results_folder,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=get_num_dalys,
    do_scaling=True
).pipe(set_param_names_as_column_index_level_0)


def plot_dalys_barplot(df):
    df.columns = df.columns.get_level_values(0)

    # Reshape the DataFrame
    plot_df = df.T  # transpose to get scenarios as rows
    plot_df.columns = ['num_dalys']  # rename the single column
    plot_df['Scenario'] = plot_df.index  # move index to a column
    plot_df = plot_df.reset_index(drop=True)

    status_quo_value = plot_df.loc[plot_df['Scenario'] == 'Status Quo', 'num_dalys'].values[0]

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x='Scenario', y='num_dalys')

    # Add horizontal line at Status Quo
    plt.axhline(y=status_quo_value, color='grey', linestyle='--', linewidth=1.2, label='Status Quo')

    # Formatting
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('DALYs')
    plt.xlabel('Scenario')
    plt.title(f'DALYs by Scenario')
    plt.tight_layout()
    plt.show()

plot_dalys_barplot(num_dalys)
