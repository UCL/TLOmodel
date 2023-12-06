"""
Read in the output files generated by analysis_scenarios and plot outcomes for comparison
"""

import datetime
from pathlib import Path
import matplotlib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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

# outputspath = Path("./outputs/t.mangal@imperial.ac.uk")
outputspath = Path("./outputs")

# Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("exclude_HTM_services.py", outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# Create a list of strings summarizing the parameter values in the different draws
param_strings = [f"{row.module_param}={row.value}" for _, row in params.iterrows()]

TARGET_PERIOD = (Date(2010, 1, 1), Date(2020, 1, 1))


# extract total deaths
def extract_total_deaths(results_folder, do_scaling=True):
    """ sum all deaths occurring for each run of each draw
    dataframe returned: row=total deaths, column=run/draw
    """
    def extract_deaths_total(df: pd.DataFrame) -> pd.Series:
        return pd.Series({"Total": len(df)})

    return extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=extract_deaths_total,
        do_scaling=do_scaling
    )


def summarise_total_deaths(results_folder, do_scaling=True):
    """ sum all deaths occurring for each run of each draw
    dataframe returned: row=total deaths, column=run/draw
    """
    def extract_deaths_total(df: pd.DataFrame) -> pd.Series:
        return pd.Series({"Total": len(df)})

    return summarize(extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=extract_deaths_total,
        do_scaling=do_scaling
    ),
        only_mean=False)

total_deaths = summarise_total_deaths(results_folder)
rounded_total_deaths = total_deaths.applymap(round_to_nearest_100)



def plot_summarized_total_deaths(summarized_total_deaths, scenario_info, mean_deaths_difference_by_run):
    """ barplot with mean total deaths for each scenario
    with the mean difference in numbers of deaths compared with the baseline
    added above each bar
    """
    list_of_scenarios = list(range(scenario_info['number_of_draws']))
    fig, ax = plt.subplots()
    number_of_draws = scenario_info['number_of_draws']
    statistic_values = {
        s: np.array(
            [summarized_total_deaths[(d, s)].values[0] for d in range(number_of_draws)]
        )
        for s in ["mean", "lower", "upper"]
    }
    ax.bar(
        list_of_scenarios,
        statistic_values["mean"],
        yerr=[
            statistic_values["mean"] - statistic_values["lower"],
            statistic_values["upper"] - statistic_values["mean"]
        ],
        color="mediumaquamarine"
    )
    ax.set_ylim(0, max(statistic_values["upper"]) * 1.1)
    plt.title("Total deaths by scenario (scaled)")

    # add values above bars
    # create gap above bar for value
    gap = statistic_values["mean"][0] * 0.1
    for i in range(len(list_of_scenarios)):
        plt.text(list_of_scenarios[i], (statistic_values["mean"][i]) + gap,
                 mean_deaths_difference_by_run[i], ha="center")

    ax.set_ylabel("Total number of deaths")
    fig.tight_layout()

    return fig, ax


def compute_difference_in_deaths_across_runs(total_deaths, scenario_info):
    """# this computes the mean difference in deaths between each scenario and baseline
    # numbers of deaths are compared run-for-run
    """
    out = [None] * scenario_info["number_of_draws"]
    for scenario in range(scenario_info["number_of_draws"]):
        deaths_difference_by_run = [
            total_deaths[scenario][run_number]["Total"] - total_deaths[0][run_number]["Total"]
            for run_number in range(scenario_info["runs_per_draw"])
        ]

        out[scenario] = np.mean(deaths_difference_by_run)

    return out


def extract_deaths_by_age(results_folder):
    """ produces dataframe with mean (+ 95% UI) number of deaths
    for each draw by age-group
    dataframe returned: rows=age-gp, columns=draw median, draw lower, draw upper
    """
    def extract_deaths_by_age_group(df: pd.DataFrame) -> pd.Series:
        _, age_group_lookup = make_age_grp_lookup()
        df["Age_Grp"] = df["age"].map(age_group_lookup).astype(make_age_grp_types())
        df = df.rename(columns={"sex": "Sex"})
        return df.groupby(["Age_Grp"])["person_id"].count()

    return summarize(extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=extract_deaths_by_age_group,
        do_scaling=True
    ), only_mean=False, collapse_columns=True
    )


# line plot of number deaths by age
def plot_summarized_deaths_by_age(deaths_summarized_by_age):
    fig, ax = plt.subplots()

    for i in range(scenario_info["number_of_draws"]):
        central_values = deaths_summarized_by_age[(i, "mean")].values
        lower_values = deaths_summarized_by_age[(i, "lower")].values
        upper_values = deaths_summarized_by_age[(i, "upper")].values
        ax.plot(
            deaths_summarized_by_age.index, central_values,
            color=f"C{i}",
            label=i
        )
        ax.fill_between(
            deaths_summarized_by_age.index, lower_values, upper_values,
            alpha=0.5,
            color=f"C{i}",
            label="_"
        )
    ax.set(xlabel="Age-Group", ylabel="Total deaths")
    ax.set_xticks(deaths_summarized_by_age.index)
    ax.set_xticklabels(labels=deaths_summarized_by_age.index, rotation=90)
    ax.legend()
    fig.tight_layout()
    return fig, ax


# line plot of number deaths by age
# todo for every run, aggregate numbers deaths, then summarise by age-groups
def barplot_summarized_deaths_by_age(deaths_summarized_by_age, proportion):

    # combine some age-groups
    deaths_summarized_by_age.loc['15-59'] = deaths_summarized_by_age.loc[['15-19', '20-24', '25-29',
                                                                          '30-34', '35-39', '40-44',
                                                                          '45-49', '50-54', '55-59']].sum()
    deaths_summarized_by_age.loc['60+'] = deaths_summarized_by_age.loc[['60-64', '65-69', '70-74',
                                                                        '75-79', '80-84', '85-89',
                                                                        '90-94', '95-99', '100+']].sum()

    # select only age-groups of interest
    deaths_to_plot = deaths_summarized_by_age.loc[['0-4', '5-9', '10-14', '15-59', '60+']]
    deaths_to_plot = deaths_to_plot.stack().reset_index()

    central_values = deaths_to_plot.loc[deaths_to_plot.stat == 'mean']
    central_values = central_values.drop('stat', axis=1)

    if proportion:
        # calculate deaths as a proportion of total
        # get column sums
        sum_deaths = central_values.sum(axis=0)
        d = pd.DataFrame()
        d['Age_Grp'] = central_values['Age_Grp']
        d['0'] = central_values[0] / sum_deaths[0]
        d['1'] = central_values[1] / sum_deaths[1]
        d['2'] = central_values[2] / sum_deaths[2]
        d['3'] = central_values[3] / sum_deaths[3]
        d['4'] = central_values[4] / sum_deaths[4]
        d['5'] = central_values[5] / sum_deaths[5]

        # replace central_values for plotting
        central_values = d

    # x-axis will be the draw
    # stacking variable will be values by age-group
    # switch age-group to columns
    tmp = central_values.T
    # Rename the columns using the first row
    tmp.columns = tmp.iloc[0]
    # Delete the first row
    tmp = tmp.drop(tmp.index[0])
    # Reset the index
    tmp = tmp.reset_index(drop=True)
    tmp.index = ["Status quo", "Excl HIV", "Excl TB",
                       "Excl malaria", "Excl HTM"]

    tmp.plot(kind='bar', stacked=True)

    # Customize the plot
    plt.xticks(range(scenario_info["number_of_draws"]), tmp.index, rotation=0)
    plt.ylabel("Total deaths")
    plt.xlabel("")
    plt.subplots_adjust(bottom=0.2, left=0.15)
    plt.savefig(outputspath / "Deaths_by_age_barplot_excl_htm.png")

    plt.show()


# total deaths in the scenario runs
total_deaths = extract_total_deaths(results_folder, do_scaling=True)
deaths_summarized_by_age = extract_deaths_by_age(results_folder)

# Compute and print the difference between the deaths across the scenario draws
mean_deaths_difference_by_run = compute_difference_in_deaths_across_runs(
    total_deaths, scenario_info
)
format_mean_deaths_difference_by_run = [round(elem) for elem in mean_deaths_difference_by_run]
print(f"Mean difference in total deaths = {format_mean_deaths_difference_by_run}")

# Plot the total deaths across the two scenario draws as a bar plot with error bars
fig_1, ax_1 = plot_summarized_total_deaths(summarize(total_deaths), scenario_info, format_mean_deaths_difference_by_run)
plt.show()

# Plot the total deaths across scenarios by age
fig_1, ax_1 = plot_summarized_deaths_by_age(deaths_summarized_by_age)
fig_1.savefig(outputspath / "Deaths_by_age_excl_htm.png")
plt.show()


# barplot deaths by broad age-group
barplot_summarized_deaths_by_age(deaths_summarized_by_age, proportion=False)


def get_num_deaths_by_cause_label(_df):
    """Return total number of Deaths by label within the TARGET_PERIOD
    values are summed for all ages and aggregated across the runs
    df returned: rows=COD, columns=draw
    """
    return _df \
        .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
        .groupby(_df['label']) \
        .size()


num_deaths_by_cause_label = summarize(
    extract_results(
        results_folder,
        module='tlo.methods.demography',
        key='death',
        custom_generate_series=get_num_deaths_by_cause_label,
        do_scaling=True
    )
)
TARGET_PERIOD = (Date(2010, 1, 1), Date(2020, 1, 1))

mean_deaths_by_cause = num_deaths_by_cause_label.xs('mean', level=1, axis=1)
mean_deaths_by_cause_lower = num_deaths_by_cause_label.xs('lower', level=1, axis=1)
mean_deaths_by_cause_upper = num_deaths_by_cause_label.xs('upper', level=1, axis=1)


# Function to round to the nearest 1000
def round_to_nearest_100(value):
    return round(value, -2)

# Apply the rounding function to the entire DataFrame
rounded_deaths = mean_deaths_by_cause.applymap(round_to_nearest_100)
rounded_deaths_lower = mean_deaths_by_cause_lower.applymap(round_to_nearest_100)
rounded_deaths_upper = mean_deaths_by_cause_upper.applymap(round_to_nearest_100)

# Apply the rounding function to the entire DataFrame
sum_deaths = mean_deaths_by_cause.sum(axis=0)

rounded_deaths.to_csv(outputspath / "deaths_by_cause_excl_htm.csv")



# plot AIDS deaths by yr
def summarise_aids_deaths(results_folder):
    """ returns mean AIDS deaths for each year
    aggregated across all runs of each draw
    AIDS_TB and AIDS_non_TB are combined into one count
    """

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
    # removes multi-index
    results_deaths = results_deaths.reset_index()

    # select only cause AIDS_TB and AIDS_non_TB
    tmp = results_deaths.loc[
        (results_deaths.cause == "AIDS_TB") | (results_deaths.cause == "AIDS_non_TB")
        ]

    # group deaths by year
    tmp = pd.DataFrame(tmp.groupby(["year"]).sum())

    # get mean for each draw
    mean_aids_deaths = pd.concat({'mean': tmp.groupby(level=0, axis=1).mean()}, axis=1).swaplevel(axis=1)

    return mean_aids_deaths


def summarise_deaths_for_one_cause(results_folder, cause):
    """ returns mean deaths for each year of the simulation
    values are aggregated across the runs of each draw
    for the specified cause
    """

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
    # removes multi-index
    results_deaths = results_deaths.reset_index()

    # select only cause specified
    tmp = results_deaths.loc[
        (results_deaths.cause == cause)
    ]

    # group deaths by year
    tmp = pd.DataFrame(tmp.groupby(["year"]).sum())

    # get mean for each draw
    mean_deaths = pd.concat({'mean': tmp.groupby(level=0, axis=1).mean()}, axis=1).swaplevel(axis=1)

    return mean_deaths


aids_deaths = summarise_aids_deaths(results_folder)
tb_deaths = summarise_deaths_for_one_cause(results_folder, 'TB')
malaria_deaths = summarise_deaths_for_one_cause(results_folder, 'Malaria')


