"""Script for plotting basic outputs from the Copd module:
 * 1) Prevalence of each category of lungfunction by age/sex [stacked bar chart by age/sex] at in 2010, 2020, 2030
 * 2) Number of deaths compared to the GBD dataset
"""
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation
from tlo.analysis.utils import (
    compare_number_of_deaths,
    parse_log_file,
    unflatten_flattened_multi_index_in_logging,
)
from tlo.methods import (
    copd,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)


class CopdAnalyses:
    """ Copd Analyses class responsible for plotting all COPD module outputs. Here we are plotting;
        1. stacked bars for individuals in each of the lung function categories;
        2. stacked bars for individuals in each of the lung function categories grouped by sex
        3. stacked bars for individuals in each of the lung function categories grouped by age group
        4. deaths(Modal against GBD) grouped by sex
        5. deaths(Modal against GBD) grouped by age group """

    def __init__(self, logfile_path):
        """ called each time the CopdAnalyses class is initialised

        :param logfile_path: path to a folder which contains the logfile to be analysed """

        self.__logfile_path = logfile_path  # path to copd logs

        # create copd logs dictionary
        self.__logs_dict = parse_log_file(self.__logfile_path)['tlo.methods.copd']

        # create a DataFrame that contains copd prevalence data
        self.__copd_prev = self.construct_dfs()['copd_prevalence']

        # initialise lung function categories
        self.__lung_func_cats = ['lung function category 0', 'lung function category 1', 'lung function category 2',
                                 'lung function category 3', 'lung function category 4', 'lung function category 5',
                                 'lung function category 6']

        # gender descriptions dictionary
        self.__gender_desc = {'M': 'Males',
                              'F': 'Females'
                              }

    def construct_dfs(self) -> dict:
        """ Create dict of pd.DataFrames containing counts of different lifestyle properties by date, sex and
        age-group """
        return {
            k: unflatten_flattened_multi_index_in_logging(v.set_index('date'))
            for k, v in self.__logs_dict.items() if k in ['copd_prevalence']
        }

    def plot_lung_function(self):
        """ plot for all people per each lung function """
        re_ordered_copd_prev = self.__copd_prev.reorder_levels([2, 0, 1], axis=1)
        plot_df = pd.DataFrame()
        for _lung_func, _ in enumerate(self.__lung_func_cats):
            plot_df[_lung_func] = re_ordered_copd_prev[f'{_lung_func}'].sum(axis=1)
        # get totals per year
        plot_df = plot_df.groupby(plot_df.index.year).sum()
        # turn totals into proportions
        plot_df = plot_df.apply(lambda row: row / row.sum(), axis=1)
        # do plotting
        ax = plot_df.plot(kind='bar', stacked=True)
        ax.set_title("Proportion of people in each Lung Function Category")
        ax.legend(self.__lung_func_cats, loc='lower right')
        ax.set_xlabel("Year")
        ax.set_ylabel("Proportion of each lung function category")
        plt.show()

    def plot_lung_function_by_gender(self):
        """ plot for all people per each lung funtion """
        re_ordered_df = self.__copd_prev.reorder_levels([0, 2, 1], axis=1)
        _df_dict = dict()
        _rows_counter: int = 0  # a counter to set the number of rows when plotting
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        for _gender in ['M', 'F']:
            plot_df = pd.DataFrame()
            for _lung_func, _ in enumerate(self.__lung_func_cats):
                plot_df[_lung_func] = re_ordered_df[_gender][f'{_lung_func}'].sum(axis=1)
            _df_dict[_gender] = plot_df
            # get totals per year
            _df_dict[_gender] = _df_dict[_gender].groupby(_df_dict[_gender].index.year).sum()
            # turn totals into proportions
            _df_dict[_gender] = _df_dict[_gender].apply(lambda row: row / row.sum(), axis=1)
            # do plotting
            ax = _df_dict[_gender].plot(kind='bar', stacked=True, ax=axes[_rows_counter],
                                        title=f"{self.__gender_desc[_gender]} "
                                              f"proportion in each Lung Function Category")
            ax.legend(self.__lung_func_cats, loc='lower right')
            ax.set_xlabel("Year")
            ax.set_ylabel("Proportion of each lung function category")
            _rows_counter += 1
        plt.show()

    def plot_lung_function_categories_by_age_group(self):
        """ plot for lung function for all age groups in the year 2022 """
        # select logs from the latest year. In this case we are selecting year 2022
        re_ordered_df = self.__copd_prev.reorder_levels([2, 0, 1], axis=1)
        plot_df = pd.DataFrame()
        mask = (re_ordered_df.index > pd.to_datetime('2022-01-01')) & (
            re_ordered_df.index <= pd.to_datetime('2023-01-01'))
        re_ordered_df = re_ordered_df.loc[mask]
        for _lung_func, _ in enumerate(self.__lung_func_cats):
            plot_df[f'{_lung_func}'] = re_ordered_df[f'{_lung_func}']["M"].sum(axis=0) + \
                                       re_ordered_df[f'{_lung_func}']["F"].sum(axis=0)

        plot_df = plot_df.apply(lambda row: row / row.sum(), axis=1)
        ax = plot_df.plot(kind='bar', stacked=True)
        ax.legend(self.__lung_func_cats, loc="upper right")
        ax.set_title("Proportion of each lung function category per each age group in 2022")
        ax.set_xlabel("age group")
        ax.set_ylabel("Proportion of each lung function category")
        plt.show()

    def plot_modal_gbd_deaths_by_gender(self):
        """ compare modal and GBD deaths by gender """
        death_compare = compare_number_of_deaths(self.__logfile_path, resourcefilepath)
        fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
        for _col, sex in enumerate(('M', 'F')):
            plot_df = death_compare.loc[(['2010-2014', '2015-2019'], sex, slice(None), 'COPD')].groupby('period').sum()
            ax = plot_df['model'].plot.bar(color='#ADD8E6', label='Model', ax=axs[_col], rot=0)
            ax.errorbar(x=plot_df['model'].index, y=plot_df.GBD_mean,
                        yerr=[plot_df.GBD_lower, plot_df.GBD_upper],
                        fmt='o', color='#23395d', label="GBD")
            ax.set_title(f'{self.__gender_desc[sex]} annual COPD deaths, 2010-2019')
            ax.set_xlabel("Time period")
            ax.set_ylabel("Number of deaths")
            ax.legend(loc=2)
        plt.tight_layout()

        plt.tight_layout()
        plt.show()

    def plot_modal_gbd_deaths_by_age_group(self):
        """ compare modal and GBD deaths by age group """
        death_compare = compare_number_of_deaths(self.__logfile_path, resourcefilepath)
        plot_df = death_compare.loc[(['2010-2014', '2015-2019'], slice(None), slice(None))].groupby(
            'age_grp').sum()
        ax = plot_df['model'].plot.bar(color='#ADD8E6', label='Model', rot=0)
        ax.errorbar(x=plot_df['model'].index, y=plot_df.GBD_mean,
                    yerr=[plot_df.GBD_lower, plot_df.GBD_upper],
                    fmt='o', color='#23395d', label="GBD")
        ax.set_title('Mean annual deaths by age group, 2010-2019')
        ax.set_xlabel("Age group")
        ax.set_ylabel("Number of deaths")
        ax.legend(loc=1)
        plt.tight_layout()

        plt.tight_layout()
        plt.show()


start_date = Date(2010, 1, 1)
end_date = Date(2030, 1, 1)

resourcefilepath = Path("./resources")  # Path to resource files

outputpath = Path('./outputs')  # path to outputs folder


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


# run simulation and store logfile path
sim = get_simulation(1000)
path_to_logfile = sim.log_filepath

# initialise Copd analyses class
copd_analyses = CopdAnalyses(logfile_path=path_to_logfile)

# plot lung function categories per each category
copd_analyses.plot_lung_function()

# plot lung function categories by gender
copd_analyses.plot_lung_function_by_gender()

# plot lung function categories by age group
copd_analyses.plot_lung_function_categories_by_age_group()

# plot modal deaths against GBD deaths by gender
copd_analyses.plot_modal_gbd_deaths_by_gender()

# plot modal deaths against GBD deaths by age group
copd_analyses.plot_modal_gbd_deaths_by_age_group()
