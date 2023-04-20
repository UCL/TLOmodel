"""Script for plotting basic outputs from the Copd module:
 * 1) Prevalence of each category of lungfunction by age/sex [stacked bar chart by age/sex] at in 2010, 2020, 2030
 * 2) Number of deaths compared to the GBD dataset
"""
import pandas as pd
from matplotlib import pyplot as plt, ticker

from tlo import Simulation, Date
from pathlib import Path

from tlo.analysis.utils import parse_log_file, unflatten_flattened_multi_index_in_logging, compare_number_of_deaths
from tlo.methods import demography, simplified_births, enhanced_lifestyle, healthsystem, symptommanager, \
    healthseekingbehaviour, healthburden, copd

start_date = Date(2010, 1, 1)
end_date = Date(2030, 1, 1)

resourcefilepath = Path("./resources")  # Path to resource files

outputpath = Path('./outputs')  # path to outputs folder


# todo @Emmmanuel....
#  First thing will be to run the model with Copd registered for 30 years. Then,
#  for (1) In the test file I've written an example of how to get this out of the log. We want stacked bars (that sum
#   to 1.0), for the people in each of the lung function categories; and one bar for each age/sex group.
#  for (2) This is done easily using the `compare_deaths` utility function and there are lots of examples of simple
#   plots being made using those results.
def get_simulation(popsize):
    """ Return a simulation object """
    sim = Simulation(
        start_date=start_date,
        seed=0,
        log_config={
            'filename': 'copd_analyses',
            'directory': outputpath,
        },
    )
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable=False,
                                           cons_availability='all'),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 copd.Copd(resourcefilepath=resourcefilepath),
                 )
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    return sim


# sim = get_simulation(1000)
# logs = parse_log_file(sim.log_filepath)
logs = parse_log_file(Path("outputs/copd_analyses__2023-04-20T113539.log"))

lung_func_cat = ['lung function category 0', 'lung function category 1', 'lung function category 2',
                 'lung function category 3', 'lung function category 4', 'lung function category 5',
                 'lung function category 6']


def construct_dfs(lifestyle_log) -> dict:
    """ Create dict of pd.DataFrames containing counts of different lifestyle properties by date, sex and
    age-group """
    return {
        k: unflatten_flattened_multi_index_in_logging(v.set_index('date'))
        for k, v in lifestyle_log.items() if k in ['copd_prevalence']
    }


def plot_lung_function():
    """ plot for all people per each lung funtion """
    re_ordered_df = construct_dfs(logs['tlo.methods.copd'])['copd_prevalence'].reorder_levels([2, 0, 1], axis=1)
    plot_df = pd.DataFrame()
    for _lung_func in ['0', '1', '2', '3', '4', '5', '6']:
        plot_df[_lung_func] = re_ordered_df[_lung_func].sum(axis=1)
    # get totals per year
    plot_df = plot_df.groupby(plot_df.index.year).sum()
    # turn totals into proportions
    plot_df = plot_df.apply(lambda row: row / row.sum(), axis=1)
    # do plotting
    ax = plot_df.plot(kind='bar', stacked=True)
    ax.set_title("Proportion of people in each Lung Function Category")
    ax.legend(lung_func_cat, loc='lower right')
    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion of each lung function category")
    plt.show()


def plot_lung_function_by_gender():
    """ plot for all people per each lung funtion """
    re_ordered_df = construct_dfs(logs['tlo.methods.copd'])['copd_prevalence'].reorder_levels([0, 2, 1], axis=1)
    _gender_desc = {'M': 'Males',
                    'F': 'Females'
                    }
    _df_dict = dict()
    _rows_counter: int = 0  # a counter to set the number of rows when plotting
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    for _gender in ['M', 'F']:
        plot_df = pd.DataFrame()
        for _lung_func in ['0', '1', '2', '3', '4', '5', '6']:
            plot_df[_lung_func] = re_ordered_df[_gender][_lung_func].sum(axis=1)
        _df_dict[_gender] = plot_df
        # get totals per year
        _df_dict[_gender] = _df_dict[_gender].groupby(_df_dict[_gender].index.year).sum()
        # turn totals into proportions
        _df_dict[_gender] = _df_dict[_gender].apply(lambda row: row / row.sum(), axis=1)
        # do plotting
        ax = _df_dict[_gender].plot(kind='bar', stacked=True, ax=axes[_rows_counter],
                                    title=f"{_gender_desc[_gender]} proportion in each Lung Function Category")
        ax.legend(lung_func_cat, loc='lower right')
        ax.set_xlabel("Year")
        ax.set_ylabel("Proportion of each lung function category")
        _rows_counter += 1
    plt.show()


def plot_lung_function_categories_by_age_group():
    """ plot for lung function for all age groups in the year 2022 """
    # select logs from the latest year. In this case we are selecting year 2022
    re_ordered_df = construct_dfs(logs['tlo.methods.copd'])['copd_prevalence'].reorder_levels([2, 0, 1], axis=1)
    plot_df = pd.DataFrame()
    mask = (re_ordered_df.index > pd.to_datetime('2022-01-01')) & (re_ordered_df.index <= pd.to_datetime('2023-01-01'))
    re_ordered_df = re_ordered_df.loc[mask]
    for _lung_func in ['0', '1', '2', '3', '4', '5', '6']:
        plot_df[_lung_func] = re_ordered_df[_lung_func]["M"].sum(axis=0) + re_ordered_df[_lung_func]["F"].sum(axis=0)

    plot_df = plot_df.apply(lambda row: row / row.sum(), axis=1)
    ax = plot_df.plot(kind='bar', stacked=True)
    ax.legend(lung_func_cat, loc="upper right")
    ax.set_title("Proportion of each lung function category per each age group")
    ax.set_xlabel("age group")
    ax.set_ylabel("Proportion of each lung function category")
    plt.show()


def plot_modal_gbd_deaths():
    """ compare copd deaths with GBD data """
    _logs = Path("outputs/copd_analyses__2023-04-20T113539.log")
    death_df = pd.DataFrame()
    death_compare = compare_number_of_deaths(_logs, resourcefilepath)

    # plot for a period of time
    for _period in ['2010-2014', '2015-2019', '2020-2024', '2025-2029']:
        # include all ages and both sexes
        death_df[_period] = death_compare.loc[(_period, slice(None), slice(None))].sum()
    print(f'comparing deaths {death_df}')


# plot lung function categories per each category
plot_lung_function()

# plot lung function by gender
plot_lung_function_by_gender()

# plot lung function categories by age group
plot_lung_function_categories_by_age_group()

# plot modal deaths against GBD deaths
plot_modal_gbd_deaths()
