

# tlo batch-download --username tbh03@ic.ac.uk htm_with_and_without_hss-2024-08-06T094422Z


import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    make_age_grp_lookup,
    make_age_grp_types,
    summarize,
)

outputspath = Path("./outputs/tbh03@ic.ac.uk")

# Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("htm_with_and_without_hss.py", outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)



# compare two draws
log4 = load_pickled_dataframes(results_folder, draw=4)



# -----------------------------------------------------------------------------------------
# %% Epi outputs
# -----------------------------------------------------------------------------------------


def summarize_median(results: pd.DataFrame) -> pd.DataFrame:
    """ edit existing utility function to return:
    median and 95% quantiles
    """

    summary = pd.DataFrame(
        columns=pd.MultiIndex.from_product(
            [
                results.columns.unique(level='draw'),
                ["median", "lower", "upper"]
            ],
            names=['draw', 'stat']),
        index=results.index
    )

    summary.loc[:, (slice(None), "median")] = results.groupby(axis=1, by='draw').quantile(0.5).values
    summary.loc[:, (slice(None), "lower")] = results.groupby(axis=1, by='draw').quantile(0.025).values
    summary.loc[:, (slice(None), "upper")] = results.groupby(axis=1, by='draw').quantile(0.975).values

    return summary


# ---------------------------------- HIV ---------------------------------- #

# HIV incidence
hiv_inc = summarize_median(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_adult_inc_1549",
        index="date",
        do_scaling=False
    )
)

hiv_cases = summarize_median(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="n_new_infections_adult_1549",
        index="date",
        do_scaling=False
    )
)

adult_pop = summarize_median(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="pop_total",
        index="date",
        do_scaling=False
    )
)

adult_plhiv = summarize_median(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="total_plhiv",
        index="date",
        do_scaling=False
    )
)


# ---------------------------------- PERSON-YEARS ---------------------------------- #
# for each scenario, return a df with the person-years logged in each draw/run
# to be used for calculating tb incidence or mortality rates


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


# ---------------------------------- TB ---------------------------------- #
# number new active tb cases - convert to incidence rate
def tb_inc_func(results_folder):
    inc = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=False
    )

    inc.columns = inc.columns.get_level_values(0)

    # divide each run of tb incidence by py from that run
    # tb logger starts at 2011-01-01, demog starts at 2010-01-01
    # extract py log from 2011
    py = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="person_years",
        custom_generate_series=get_person_years,
        do_scaling=False
    )
    py.columns = py.columns.get_level_values(0)

    inc_per_py = inc / py
    # remove first row (2010) as nan
    inc_per_py = inc_per_py.iloc[1:]

    # Calculate the mean of each row for each column name
    summary = pd.DataFrame(
        columns=pd.MultiIndex.from_product(
            [
                inc.columns.unique(level='draw'),
                ["median", "lower", "upper"]
            ],
            names=['draw', 'stat']),
        index=inc.index
    )

    summary.loc[:, (slice(None), "median")] = inc_per_py.groupby(axis=1, by='draw').quantile(0.5).values
    summary.loc[:, (slice(None), "lower")] = inc_per_py.groupby(axis=1, by='draw').quantile(0.025).values
    summary.loc[:, (slice(None), "upper")] = inc_per_py.groupby(axis=1, by='draw').quantile(0.975).values

    return summary


tb_inc = tb_inc_func(results_folder)

tb_inc_num = extract_results(
    results_folder,
    module="tlo.methods.tb",
    key="tb_incidence",
    column="num_new_active_tb",
    index="date",
    do_scaling=False
)

tb_mdr_prop = extract_results(
    results_folder,
    module="tlo.methods.tb",
    key="tb_mdr",
    column="tbPropActiveCasesMdr",
    index="date",
    do_scaling=False
)

# ---------------------------------- MALARIA ---------------------------------- #

# malaria incidence
# value is per 1000py
mal_inc = summarize_median(
    extract_results(
        results_folder,
        module="tlo.methods.malaria",
        key="incidence",
        column="inc_1000py_2_10",
        index="date",
        do_scaling=False
    )
)

# ---------------------------------- DEATHS ---------------------------------- #

# get full outputs of numbers of deaths

results_deaths = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=(
        lambda df: df.assign(year=df["date"].dt.year).groupby(
            ["year", "cause"])["person_id"].count()
    )
)


# plot AIDS deaths by yr
def summarise_aids_deaths(results_folder):
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
    tmp = results_deaths.loc[
        (results_deaths.cause == "AIDS_TB") | (results_deaths.cause == "AIDS_non_TB")
        ]

    # group deaths by year
    tmp = pd.DataFrame(tmp.groupby(["year"]).sum())

    py = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="person_years",
        custom_generate_series=get_person_years,
        do_scaling=False
    )
    # years = pd.to_datetime(_df["date"]).dt.year
    # py = pd.Series(dtype="int64", index=years)
    py.index = tmp.index

    deaths_per_py = tmp.iloc[:, 1:26].div(py) * 1000

    # get median and UI
    tmp2 = pd.concat({
        'median': deaths_per_py.groupby(level=0, axis=1).median(0.5),
        'lower': deaths_per_py.groupby(level=0, axis=1).quantile(0.025),
        'upper': deaths_per_py.groupby(level=0, axis=1).quantile(0.975)
    }, axis=1).swaplevel(axis=1)

    return tmp2


def summarise_deaths_for_one_cause(results_folder, cause):
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

    # select only cause specified
    tmp = results_deaths.loc[
        (results_deaths.cause == cause)
    ]

    # group deaths by year
    tmp = pd.DataFrame(tmp.groupby(["year"]).sum())

    py = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="person_years",
        custom_generate_series=get_person_years,
        do_scaling=False
    )

    py.index = tmp.index

    deaths_per_py = tmp.iloc[:, 1:26].div(py) * 1000

    # get median and UI
    tmp2 = pd.concat({
        'median': deaths_per_py.groupby(level=0, axis=1).median(0.5),
        'lower': deaths_per_py.groupby(level=0, axis=1).quantile(0.025),
        'upper': deaths_per_py.groupby(level=0, axis=1).quantile(0.975)
    }, axis=1).swaplevel(axis=1)

    return tmp2


aids_deaths = summarise_aids_deaths(results_folder)
tb_deaths = summarise_deaths_for_one_cause(results_folder, 'TB')
malaria_deaths = summarise_deaths_for_one_cause(results_folder, 'Malaria')

# just get median columns
selected_columns = tb_deaths.loc[:, tb_deaths.columns.get_level_values(1).str.contains('median')]


# remove first yr of deaths data (2010) to align with inc outputs
aids_deaths = aids_deaths.iloc[1:]
tb_deaths = tb_deaths.iloc[1:]
malaria_deaths = malaria_deaths.iloc[1:]

# ---------------------------------- SMOOTH DATA ---------------------------------- #
#  create smoothed lines
data_x = hiv_inc.index.year  # 2011 onwards

start_year = 2011
end_year = 2030
increment = 0.25

# ---------------------------------- PLOTS ---------------------------------- #

plt.style.use('default')  # to reset

font = {'family': 'sans-serif',
        'color': 'black',
        'weight': 'bold',
        'size': 11,
        }

colors = ['#1E22AA',  # Deep Blue
          '#9B26B6',  # Purple
          '#F8485E',  # Red
          '#FF8F1C',  # Orange
          '#30B700',  # Green
          '#1E90FF',  # Dodger Blue
          '#32CD32',  # Lime Green
          '#FFD700',  # Gold
          '#FF4500',  # Orange Red
          '#8B0000']  # Dark Red

# Set x-axis tick positions and labels at every second data point
# xvals_for_ticks = xvals[0::4]
# xlabels_for_ticks = [int(val) for val in xvals_for_ticks]

# colors = sns.color_palette("deep", 5)
gridcol = '#EDEDED'

fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(nrows=2, ncols=4,
                                                                 constrained_layout=True,
                                                                 figsize=(16, 8))
fig.suptitle('')

for i, scenario in enumerate(hiv_inc.filter(like='median')):
    ax1.plot(hiv_inc.index, hiv_inc.filter(like='median')[scenario] * 1000, label=scenario,
             color=colors[i], zorder=2)

ax1.grid(True, linestyle='-', color=gridcol, zorder=1)
ax1.set(title='HIV',
        ylabel='HIV Incidence, per 1000py')
# ax1.set_xticks(xvals_for_ticks)
ax1.set_xticklabels("")
ax1.set_ylim(0, 8)

# TB incidence
for i, scenario in enumerate(tb_inc.filter(like='median')):
    ax2.plot(hiv_inc.index, tb_inc.filter(like='median')[scenario] * 1000, label=scenario,
             color=colors[i], zorder=2)

ax2.grid(True, linestyle='-', color=gridcol, zorder=1)
ax2.set(title='TB',
        ylabel='TB Incidence, per 1000py')
ax2.set_ylim(0, 5)

# Malaria incidence
for i, scenario in enumerate(mal_inc.filter(like='median')):
    ax3.plot(hiv_inc.index, mal_inc.filter(like='median')[scenario], label=scenario,
             color=colors[i], zorder=2)

ax3.grid(True, linestyle='-', color=gridcol, zorder=1)
ax3.set(title='Malaria',
        ylabel='Malaria Incidence, per 1000py')

ax3.set_ylim(0, 600)

# empty plot
ax4.axis('off')

# AIDS deaths
for i, scenario in enumerate(aids_deaths.filter(like='median')):
    ax5.plot(hiv_inc.index, aids_deaths.filter(like='median')[scenario], label=scenario,
             color=colors[i], zorder=2)

ax5.grid(True, linestyle='-', color=gridcol, zorder=1)
ax5.set(title='',
        ylabel='AIDS mortality rate per 1000py')
ax5.set_ylim(0, 3)
ax5.tick_params(axis='x', rotation=70)

# TB deaths
for i, scenario in enumerate(tb_deaths.filter(like='median')):
    ax6.plot(hiv_inc.index, tb_deaths.filter(like='median')[scenario], label=scenario,
             color=colors[i], zorder=2)

ax6.grid(True, linestyle='-', color=gridcol, zorder=1)
ax6.set(title='',
        ylabel='TB mortality rate per 1000py')
ax6.set_ylim(0, 0.75)
ax6.tick_params(axis='x', rotation=70)

# Malaria deaths
for i, scenario in enumerate(malaria_deaths.filter(like='median')):
    ax7.plot(hiv_inc.index, malaria_deaths.filter(like='median')[scenario], label=scenario,
             color=colors[i], zorder=2)

ax7.grid(True, linestyle='-', color=gridcol, zorder=1)
ax7.set(title='',
        ylabel='Malaria mortality rate per 1000py')

ax7.set_ylim(0, 1)
ax7.tick_params(axis='x', rotation=70)
ax7.legend(loc='upper right',
           labels=['Baseline', 'Full HSS', 'HIV no HSS', 'HIV + HSS',
                   'TB no HSS', 'TB + HSS',
                   'Malaria no HSS', 'Malaria + HSS',
                   'HTM no HSS', 'HTM + HSS'],
           bbox_to_anchor=(1.8, 1.0))

# empty plot
ax8.axis('off')

# fig.savefig(outputspath / "Apr2024_HTMresults/Epi_outputs_excl_htm_mortality.png")

plt.show()

# %% ----------------------------------------------------------------------------------
"""
Read in the output files generated by analysis_scenarios and plot outcomes for comparison
"""

# Create a list of strings summarizing the parameter values in the different draws
param_strings = [f"{row.module_param}={row.value}" for _, row in params.iterrows()]

TARGET_PERIOD = (Date(2020, 1, 1), Date(2030, 1, 1))


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


def round_to_nearest_100(x):
    return 100 * round(x / 100)


total_deaths = summarise_total_deaths(results_folder)
rounded_total_deaths = total_deaths.applymap(round_to_nearest_100)


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


person_years = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="person_years",
    custom_generate_series=get_person_years,
    do_scaling=True
)

person_years.index = person_years.index.year
# person_years.to_csv(outputspath / ('Apr2024_HTMresults/py_by_cause_yr_run' + '.csv'))

# get dalys per person_year over the whole simulation
py_totals = person_years.sum(axis=0)


def summarise_total_mortality_rate(results_folder, do_scaling=True):
    """ sum all deaths occurring for each run of each draw
    dataframe returned: row=total deaths, column=run/draw
    """

    def extract_deaths_total(df: pd.DataFrame) -> pd.Series:
        return pd.Series({"Total": len(df)})

    t = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=extract_deaths_total,
        do_scaling=do_scaling
    )

    t2 = t.div(py_totals) * 1000

    return t2


total_mortality_rate = summarise_total_mortality_rate(results_folder)


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
    tmp.index = ["Actual", "No HIV", "No TB",
                 "No malaria", "No HTM"]

    tmp.plot(kind='bar', stacked=True)

    # Customize the plot
    plt.xticks(range(scenario_info["number_of_draws"]), tmp.index, rotation=0)
    plt.ylabel("Total deaths")
    plt.xlabel("")
    plt.subplots_adjust(bottom=0.2, left=0.15)
    # plt.savefig(outputspath / "Apr2024_HTMresults/Deaths_by_age_barplot_excl_htm.png")

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
# fig_1.savefig(outputspath / "Apr2024_HTMresults/Deaths_by_age_excl_htm.png")
plt.show()

# barplot deaths by broad age-group
barplot_summarized_deaths_by_age(deaths_summarized_by_age, proportion=False)


# get mortality rates by cause ------------------------------------------------------------------------

def get_num_deaths_by_cause_label(_df):
    """Return total number of Deaths by label within the TARGET_PERIOD
    values are summed for all ages
    df returned: rows=COD, columns=draw
    """
    return _df \
        .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
        .groupby(_df['label']) \
        .size()


TARGET_PERIOD = (Date(2020, 1, 1), Date(2030, 1, 1))

num_deaths_by_cause_label = extract_results(
    results_folder,
    module='tlo.methods.demography',
    key='death',
    custom_generate_series=get_num_deaths_by_cause_label,
    do_scaling=True
)

# num_deaths_by_cause_label.to_csv(outputspath / "Apr2024_HTMresults/num_deaths_by_cause.csv")


# get median deaths per py to output
t1 = pd.concat({
    'median': num_deaths_by_cause_label.groupby(level=0, axis=1).median(0.5) / 1000,
    'lower': num_deaths_by_cause_label.groupby(level=0, axis=1).quantile(0.025) / 1000,
    'upper': num_deaths_by_cause_label.groupby(level=0, axis=1).quantile(0.975) / 1000
}, axis=1).swaplevel(axis=1)
t1_sorted = t1.sort_index(axis=1, level='draw')


# t1_sorted.to_csv(outputspath / "Apr2024_HTMresults/num_deaths_summarised.csv")


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
    do_scaling=True
)
py_totals = py0.sum(axis=0)

deaths_by_py = num_deaths_by_cause_label.div(py_totals) * 1000
deaths_by_py.to_csv(outputspath / "Apr2024_HTMresults/deaths_by_py_by_run.csv")

# get median deaths per py to output
tmp2 = pd.concat({
    'median': deaths_by_py.groupby(level=0, axis=1).median(0.5),
    'lower': deaths_by_py.groupby(level=0, axis=1).quantile(0.025),
    'upper': deaths_by_py.groupby(level=0, axis=1).quantile(0.975)
}, axis=1).swaplevel(axis=1)
tmp2_sorted = tmp2.sort_index(axis=1, level='draw')
tmp2_sorted.to_csv(outputspath / "Apr2024_HTMresults/deaths_by_py.csv")


# Function to combine values based on specified format with rounding
def combine_values(row):
    combined_values = []
    for i in range(5):  # Iterate over 'draw' levels 0 to 4
        try:
            median = round(row[(i, 'median')], 3)  # Round median value to 3 decimal places
            lower = round(row[(i, 'lower')], 3)  # Round lower value to 3 decimal places
            upper = round(row[(i, 'upper')], 3)  # Round upper value to 3 decimal places
            combined_values.append(f"{median} ({lower}-{upper})")
        except KeyError:
            combined_values.append("")  # Handle missing columns gracefully
    return pd.Series(combined_values, index=['draw0', 'draw1', 'draw2', 'draw3', 'draw4'])


# Apply the function row-wise to create a new DataFrame
new_df = tmp2_sorted.apply(combine_values, axis=1)

new_df.to_csv(outputspath / "Apr2024_HTMresults/SI_deaths_by_py_formatted.csv")

# get mortality rates by cause for plot ------------------------------------------------------------------------

num_deaths_by_cause_label = extract_results(
    results_folder,
    module='tlo.methods.demography',
    key='death',
    custom_generate_series=get_num_deaths_by_cause_label,
    do_scaling=True
)

t1 = num_deaths_by_cause_label.div(py_totals) * 1000

mean_deaths_by_cause = t1.groupby(level=0, axis=1).median(0.5)
mean_deaths_by_cause_lower = t1.groupby(level=0, axis=1).quantile(0.025)
mean_deaths_by_cause_upper = t1.groupby(level=0, axis=1).quantile(0.975)

appended_df = pd.concat([mean_deaths_by_cause, mean_deaths_by_cause_lower, mean_deaths_by_cause_upper],
                        axis=1, keys=['mean_deaths', 'lower_deaths', 'upper_deaths'])
appended_df.to_csv(outputspath / "Apr2024_HTMresults/mortality_rates_per_py_by_cause.csv")


# deaths_by_cause_by_run = extract_results(
#         results_folder,
#         module='tlo.methods.demography',
#         key='death',
#         custom_generate_series=get_num_deaths_by_cause_label,
#         do_scaling=True
#     )
#
# deaths_by_cause_by_run.to_csv(outputspath / "Apr2024_HTMresults/deaths_by_cause_by_run.csv")


# Function to round to the nearest 1000
def round_to_nearest_100(value):
    return round(value, -2)


#
# # Apply the rounding function to the entire DataFrame
# rounded_deaths = mean_deaths_by_cause.applymap(round_to_nearest_100)
# rounded_deaths_lower = mean_deaths_by_cause_lower.applymap(round_to_nearest_100)
# rounded_deaths_upper = mean_deaths_by_cause_upper.applymap(round_to_nearest_100)
#
# # Apply the rounding function to the entire DataFrame
# sum_deaths = mean_deaths_by_cause.sum(axis=0)
#
# rounded_deaths.to_csv(outputspath / "Apr2024_HTMresults/deaths_by_cause_excl_htm.csv")
#

def summarise_deaths_for_one_cause(results_folder, label):
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
                ["year", "label"])["person_id"].count()
        ),
        do_scaling=True,
    )
    # removes multi-index
    results_deaths = results_deaths.reset_index()

    # select only cause specified
    tmp = results_deaths.loc[
        (results_deaths.label == label)
    ]

    # group deaths by year
    tmp = pd.DataFrame(tmp.groupby(["year"]).sum())

    # get mean for each draw
    mean_deaths = pd.concat({'mean': tmp.iloc[:, 1:].groupby(level=0, axis=1).mean()}, axis=1).swaplevel(axis=1)

    return mean_deaths


aids_deaths = summarise_deaths_for_one_cause(results_folder, 'AIDS')
tb_deaths = summarise_deaths_for_one_cause(results_folder, 'TB (non-AIDS)')
malaria_deaths = summarise_deaths_for_one_cause(results_folder, 'Malaria')

# -----------------------------------------------------------------------------------
# plot life expectancy and numbers of deaths by cause

# all cause deaths
total_num_deaths = summarise_total_deaths(results_folder)
# mean_num_total_deaths = total_deaths.loc[:, total_deaths.columns.get_level_values(1) == 'mean']
# lower_num_total_deaths = total_deaths.loc[:, total_deaths.columns.get_level_values(1) == 'lower']
# upper_num_total_deaths = total_deaths.loc[:, total_deaths.columns.get_level_values(1) == 'upper']

# mortality rate per 1000py
mean_total_deaths = total_mortality_rate.groupby(level=0, axis=1).median(0.5)
lower_total_deaths = total_mortality_rate.groupby(level=0, axis=1).quantile(0.025)
upper_total_deaths = total_mortality_rate.groupby(level=0, axis=1).quantile(0.975)

appended_df = pd.concat([mean_total_deaths, lower_total_deaths, upper_total_deaths], axis=0)
appended_df.to_csv(outputspath / "Apr2024_HTMresults/total_mortality_rates_per_py.csv")

# deaths by cause: mean_deaths_by_cause

deaths_for_plot = [mean_deaths_by_cause.loc['AIDS', 0],
                   mean_deaths_by_cause.loc['TB (non-AIDS)', 0],
                   mean_deaths_by_cause.loc['Malaria', 0],
                   mean_total_deaths.values[0][0],
                   ##
                   mean_deaths_by_cause.loc['AIDS', 1],
                   mean_deaths_by_cause.loc['TB (non-AIDS)', 1],
                   mean_deaths_by_cause.loc['Malaria', 1],
                   mean_total_deaths.values[0][1],
                   ##
                   mean_deaths_by_cause.loc['AIDS', 2],
                   mean_deaths_by_cause.loc['TB (non-AIDS)', 2],
                   mean_deaths_by_cause.loc['Malaria', 2],
                   mean_total_deaths.values[0][2],
                   ##
                   mean_deaths_by_cause.loc['AIDS', 3],
                   mean_deaths_by_cause.loc['TB (non-AIDS)', 3],
                   mean_deaths_by_cause.loc['Malaria', 3],
                   mean_total_deaths.values[0][3],
                   ##
                   mean_deaths_by_cause.loc['AIDS', 4],
                   mean_deaths_by_cause.loc['TB (non-AIDS)', 4],
                   mean_deaths_by_cause.loc['Malaria', 4],
                   mean_total_deaths.values[0][4],
                   ]

lower_deaths_for_plot = [mean_deaths_by_cause_lower.loc['AIDS', 0],
                         mean_deaths_by_cause_lower.loc['TB (non-AIDS)', 0],
                         mean_deaths_by_cause_lower.loc['Malaria', 0],
                         lower_total_deaths.values[0][0],
                         ##
                         mean_deaths_by_cause_lower.loc['AIDS', 1],
                         mean_deaths_by_cause_lower.loc['TB (non-AIDS)', 1],
                         mean_deaths_by_cause_lower.loc['Malaria', 1],
                         lower_total_deaths.values[0][1],
                         ##
                         mean_deaths_by_cause_lower.loc['AIDS', 2],
                         mean_deaths_by_cause_lower.loc['TB (non-AIDS)', 2],
                         mean_deaths_by_cause_lower.loc['Malaria', 2],
                         lower_total_deaths.values[0][2],
                         ##
                         mean_deaths_by_cause_lower.loc['AIDS', 3],
                         mean_deaths_by_cause_lower.loc['TB (non-AIDS)', 3],
                         mean_deaths_by_cause_lower.loc['Malaria', 3],
                         lower_total_deaths.values[0][3],
                         ##
                         mean_deaths_by_cause_lower.loc['AIDS', 4],
                         mean_deaths_by_cause_lower.loc['TB (non-AIDS)', 4],
                         mean_deaths_by_cause_lower.loc['Malaria', 4],
                         lower_total_deaths.values[0][4],
                         ]

upper_deaths_for_plot = [mean_deaths_by_cause_upper.loc['AIDS', 0],
                         mean_deaths_by_cause_upper.loc['TB (non-AIDS)', 0],
                         mean_deaths_by_cause_upper.loc['Malaria', 0],
                         upper_total_deaths.values[0][0],
                         ##
                         mean_deaths_by_cause_upper.loc['AIDS', 1],
                         mean_deaths_by_cause_upper.loc['TB (non-AIDS)', 1],
                         mean_deaths_by_cause_upper.loc['Malaria', 1],
                         upper_total_deaths.values[0][1],
                         ##
                         mean_deaths_by_cause_upper.loc['AIDS', 2],
                         mean_deaths_by_cause_upper.loc['TB (non-AIDS)', 2],
                         mean_deaths_by_cause_upper.loc['Malaria', 2],
                         upper_total_deaths.values[0][2],
                         ##
                         mean_deaths_by_cause_upper.loc['AIDS', 3],
                         mean_deaths_by_cause_upper.loc['TB (non-AIDS)', 3],
                         mean_deaths_by_cause_upper.loc['Malaria', 3],
                         upper_total_deaths.values[0][3],
                         ##
                         mean_deaths_by_cause_upper.loc['AIDS', 4],
                         mean_deaths_by_cause_upper.loc['TB (non-AIDS)', 4],
                         mean_deaths_by_cause_upper.loc['Malaria', 4],
                         upper_total_deaths.values[0][4],
                         ]

# Convert lists to numpy arrays
array1 = np.array(deaths_for_plot)
array2 = array1 - np.array(lower_deaths_for_plot)
array3 = np.array(upper_deaths_for_plot) - array1

# ---------------------------------------------------------------------------------------------

target_period = (datetime.date(2019, 1, 1), datetime.date(2020, 1, 1))

# ---------------------------------------------------------------------------------------------
# sum the deaths averted through the single programmes to show difference from joint programmes

# get total deaths for each run 2010-2020
def get_total_deaths(results_folder, do_scaling=True):
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


total_deaths = get_total_deaths(results_folder)


# do run-by-run comparison, deaths in draw1 - draw0
def extract_difference_in_deaths_across_runs(total_deaths, scenario_info):
    """# this computes the mean difference in deaths between each scenario and baseline
    # numbers of deaths are compared run-for-run
    """
    out = [None] * scenario_info["number_of_draws"]
    for scenario in range(scenario_info["number_of_draws"]):
        deaths_difference_by_run = [
            total_deaths[scenario][run_number]["Total"] - total_deaths[0][run_number]["Total"]
            for run_number in range(scenario_info["runs_per_draw"])
        ]

        # Save each value of deaths_difference_by_run instead of computing the mean
        out[scenario] = deaths_difference_by_run
    df = pd.DataFrame(out, index=range(scenario_info["number_of_draws"]))

    return df


# the rows are scenarios, columns are runs
diffs_in_deaths = extract_difference_in_deaths_across_runs(total_deaths, scenario_info)

# to sum deaths averted by scenario:sum the deaths averted in each column 1:3
# this will be the all-cause deaths averted by HIV, TB and malaria programmes separately
# will have double-counting
# should be larger than the joint impact estimates
sum_values = diffs_in_deaths.iloc[1:4, :].sum()

# Add the sum values as a new row to the DataFrame
diffs_in_deaths.loc['Sum'] = sum_values

# Calculate median and percentiles for each row
row_stats = diffs_in_deaths.apply(lambda row: pd.Series([row.median(), row.quantile(0.025), row.quantile(0.975)]),
                                  axis=1)

# Rename the columns for clarity
row_stats.columns = ['Median', '2.5th Percentile', '97.5th Percentile']

print(row_stats)

#########################################################################################
# %% extract numbers of appts delivered for draw

scaling_factor = extract_results(
    results_folder,
    module="tlo.methods.population",
    key="scaling_factor",
    column="scaling_factor",
    index="date",
    do_scaling=False)

module = "tlo.methods.healthsystem.summary"
key = 'HSI_Event'
column = 'TREATMENT_ID'

info = get_scenario_info(results_folder)
# create emtpy dataframe
results = pd.DataFrame()

for draw in range(info['number_of_draws']):
    df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run=0)[module][key]

    new = df[['date', column]].copy()
    filtered_df = new[new['date'] >= '2020-01-01']

    tmp = pd.DataFrame(filtered_df[column].to_list())

    # sum each column to get total appts of each type over the simulation
    tmp2 = pd.DataFrame(tmp.sum(), columns=[f'draw_{draw}'])
    # add results to dataframe for output
    results = pd.concat([results, tmp2], axis=1)

# multiply appt numbers by scaling factor
results = results.mul(scaling_factor.values[0][0])
results.to_csv(outputspath / "tx_outputs.csv")

module = "tlo.methods.healthsystem.summary"
key = 'HSI_Event'
column = 'Number_By_Appt_Type_Code'

info = get_scenario_info(results_folder)
# create emtpy dataframe
results = pd.DataFrame()

for draw in range(info['number_of_draws']):
    df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run=0)[module][key]

    new = df[['date', column]].copy()
    filtered_df = new[new['date'] >= '2020-01-01']

    tmp = pd.DataFrame(filtered_df[column].to_list())

    # sum each column to get total appts of each type over the simulation
    tmp2 = pd.DataFrame(tmp.sum(), columns=[f'draw_{draw}'])
    # add results to dataframe for output
    results = pd.concat([results, tmp2], axis=1)

# multiply appt numbers by scaling factor
results = results.mul(scaling_factor.values[0][0])
results.to_csv(outputspath / "hsi_outputs.csv")

module = "tlo.methods.healthsystem.summary"
key = 'Consumables'
column = 'Item_NotAvailable'

info = get_scenario_info(results_folder)
# create emtpy dataframe
results = pd.DataFrame()

for draw in range(info['number_of_draws']):
    df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run=0)[module][key]

    new = df[['date', column]].copy()
    filtered_df = new[new['date'] >= '2020-01-01']

    tmp = pd.DataFrame(filtered_df[column].to_list())

    # sum each column to get total appts of each type over the simulation
    tmp2 = pd.DataFrame(tmp.sum(), columns=[f'draw_{draw}'])
    # add results to dataframe for output
    results = pd.concat([results, tmp2], axis=1)

# multiply appt numbers by scaling factor
results = results.mul(scaling_factor.values[0][0])
results.to_csv(outputspath / "consNA_outputs.csv")

# %% TB false positives

tb_false_positives_adults = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_false_positive",
        column="tbPropFalsePositiveAdults",
        index="date",
        do_scaling=False
    ),
    only_mean=True
)
tb_false_positives_adults.to_csv(outputspath / "tb_false_pos.csv")

# %% HIV treatment coverage

hiv_tx = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        column="art_coverage_adult",
        index="date",
        do_scaling=False
    ),
    only_mean=True
)
hiv_tx.to_csv(outputspath / "hiv_tx.csv")

mal_tx = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.malaria",
        key="tx_coverage",
        column="treatment_coverage",
        index="date",
        do_scaling=False
    ),
    only_mean=True
)
mal_tx.to_csv(outputspath / "mal_tx.csv")

mal_dx = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.malaria",
        key="tx_coverage",
        column="proportion_diagnosed",
        index="date",
        do_scaling=False
    ),
    only_mean=True
)
mal_dx.to_csv(outputspath / "mal_dx.csv")

tb_tx = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_treatment",
        column="tbTreatmentCoverage",
        index="date",
        do_scaling=False
    ),
    only_mean=True
)
tb_tx.to_csv(outputspath / "tb_tx.csv")

tb_dx = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_treatment",
        column="tbPropDiagnosed",
        index="date",
        do_scaling=False
    ),
    only_mean=True
)
tb_dx.to_csv(outputspath / "tb_dx.csv")
