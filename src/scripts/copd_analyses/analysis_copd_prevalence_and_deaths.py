"""Script for plotting basic outputs from the Copd module:
 * 1) Prevalence of each category of lungfunction by age/sex [stacked bar chart by age/sex]
 * 2) Number of deaths compared to the GBD dataset
"""
import datetime
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

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
        2. stacked bars for individuals in each of the lung function categories by sex
        3. stacked bars for individuals in each of the lung function categories by age group
        4. deaths(Modal) grouped by lung function
        5. deaths(Modal against GBD) grouped by sex
        6. deaths(Modal against GBD) grouped by age group """

    def __init__(self, logfile_path):
        """ called each time the Copd analyses class is initialised

        :param logfile_path: path to a folder which contains the logfile to be analysed """

        self.__logfile_path = logfile_path  # path to copd logs

        # create copd logs dictionary
        self.__logs_dict = parse_log_file(self.__logfile_path)['tlo.methods.copd']

        # create a DataFrame that contains copd prevalence data
        self.__copd_prev = self.construct_dfs()['copd_prevalence']

        # initialise lung function categories
        self.__lung_func_cats = ['category 0', 'category 1', 'category 2',
                                 'category 3', 'category 4', 'category 5',
                                 'category 6']

        # a dictionary to describe individual's tobacco status
        self.__smokers_desc = {
            'True': "Smokers",
            'False': 'Non-smokers'
        }

        # gender descriptions dictionary
        self.__gender_desc = {'M': 'Males',
                              'F': 'Females'
                              }

        # colors for plotting
        self.__plot_colors = ['#239B56', '#58D68D', '#ABEBC6', '#EDBB99', '#EB984E', '#D35400', '#641E16']

    def construct_dfs(self) -> dict:
        """ Create dict of pd.DataFrames containing counts of different lifestyle properties by date, sex and
        age-group """
        return {
            k: unflatten_flattened_multi_index_in_logging(v.set_index('date'))
            for k, v in self.__logs_dict.items() if k in ['copd_prevalence']
        }

    def plot_lung_function(self):
        """ plot for all people per each lung function """
        re_ordered_copd_prev = self.__copd_prev.reorder_levels([2, 3, 0, 1], axis=1)
        _col_counter: int = 0  # a counter for plotting. setting rows
        fig, ax = plt.subplots(ncols=2, sharex=True)  # plot setup
        _df = dict()
        for _tob in ['True', 'False']:
            plot_df = pd.DataFrame()
            for _lung_func, _ in enumerate(self.__lung_func_cats):
                plot_df[_lung_func] = re_ordered_copd_prev[_tob][f'{_lung_func}'].sum(axis=1)
            # get totals per year
            plot_df = plot_df.groupby(plot_df.index.year).sum()
            _df[_tob] = plot_df
            # convert totals into proportions
            plot_df = plot_df.apply(lambda row: row / row.sum(), axis=1)

            # do plotting
            plot_df.plot(kind='bar', stacked=True, ax=ax[_col_counter], color=self.__plot_colors,
                         title=f"Proportion of {self.__smokers_desc[_tob].lower()} in each Lung Function Category",
                         xlabel="Year",
                         ylabel="Proportions")

            _col_counter += 1  # increment column counter
        for ax in ax:
            ax.get_legend().remove()
        fontP = FontProperties()
        fontP.set_size('small')
        # add one legend on `plt
        plt.legend(self.__lung_func_cats, title="lung function categories", bbox_to_anchor=(0.7, 0.74),
                   loc='upper left', prop=fontP)
        plt.tight_layout()
        plt.savefig(
            outputpath / ('lung_function_categories' + datestamp + ".pdf"), format="pdf"
        )
        plt.savefig(Path('.outputs'))
        plt.show()

    def mod_sev_exacerb_lung_function_smokers(self):
        """ moderate and severe exacerbation for smokers all people per each lung function """
        re_ordered_copd_prev = self.__copd_prev.reorder_levels([2, 3, 0, 1], axis=1)
        _col_counter: int = 0  # a counter for plotting. setting rows
        fig, ax = plt.subplots(ncols=2, sharex=True)  # plot setup
        for _tob in ['True', 'False']:
            plot_df = pd.DataFrame()
            for _lung_func, _ in enumerate(self.__lung_func_cats):
                plot_df[_lung_func] = re_ordered_copd_prev[_tob][f'{_lung_func}'].sum(axis=1)
            # get totals per year
            plot_df = plot_df.groupby(plot_df.index.year).sum()
            # convert totals into proportions
            plot_df = plot_df.apply(lambda row: row / row.sum(), axis=1)

            # do plotting
            plot_df[[5, 6]].plot(kind='bar', stacked=True, ax=ax[_col_counter], color=self.__plot_colors[5:],
                                 title=f"{self.__smokers_desc[_tob].lower()} "
                                       f"in lung function categories 5 or 6",
                                 xlabel="Year",
                                 ylabel="Proportions",
                                 ylim=[0, 0.23]
                                 )

            _col_counter += 1  # increment column counter
        for ax in ax:
            ax.get_legend().remove()

        fontP = FontProperties()
        fontP.set_size('small')
        # add one legend on `plt
        plt.legend(self.__lung_func_cats[5:], title="lung function categories", bbox_to_anchor=(0.7, 0.74),
                   loc='upper left', prop=fontP)
        plt.tight_layout()
        plt.savefig(
            outputpath / ('lung_function_categories' + datestamp + ".pdf"), format="pdf"
        )
        plt.savefig(Path('.outputs'))
        plt.show()

    def plot_lung_function_by_gender(self):
        """ plot for all people per each lung funtion """
        re_ordered_df = self.__copd_prev.reorder_levels([0, 3, 2, 1], axis=1)
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
            # convert totals into proportions
            _df_dict[_gender] = _df_dict[_gender].apply(lambda row: row / row.sum(), axis=1)
            # do plotting
            ax = _df_dict[_gender].plot(kind='bar', stacked=True, ax=axes[_rows_counter], color=self.__plot_colors,
                                        title=f"{self.__gender_desc[_gender]} "
                                              f"proportion in each Lung Function Category")

            fontP = FontProperties()
            fontP.set_size('small')

            ax.legend(self.__lung_func_cats, title="lung function categories", bbox_to_anchor=(0.7, 0.74),
                      loc='upper left', prop=fontP)
            ax.set_xlabel("Year")
            ax.set_ylabel("Proportion of each lung function category")
            _rows_counter += 1
        plt.tight_layout()
        plt.savefig(
            outputpath / ('lung_function_categories_by_gender' + datestamp + ".pdf"), format="pdf"
        )
        plt.savefig(Path('.outputs'))
        plt.show()

    def sev_mod_exacerb_lung_function_by_gender(self):
        """ moderate and severe exacerbation by gender """
        re_ordered_df = self.__copd_prev.reorder_levels([0, 3, 2, 1], axis=1)
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
            # convert totals into proportions
            _df_dict[_gender] = _df_dict[_gender].apply(lambda row: row / row.sum(), axis=1)
            # do plotting
            ax = _df_dict[_gender][[5, 6]].plot(kind='bar', stacked=True, ax=axes[_rows_counter],
                                                ylim=[0, 0.02],
                                                color=self.__plot_colors[5:],
                                                title=f"{self.__gender_desc[_gender]} "
                                                      f"in lung function categories 5 or 6")

            fontP = FontProperties()
            fontP.set_size('small')

            ax.legend(self.__lung_func_cats[5:], title="lung function categories", bbox_to_anchor=(0.7, 0.74),
                      loc='upper left', prop=fontP)
            ax.set_xlabel("Year")
            ax.set_ylabel("Proportion of each lung function category")
            _rows_counter += 1
        plt.tight_layout()
        plt.savefig(
            outputpath / ('lung_function_categories_by_gender' + datestamp + ".pdf"), format="pdf"
        )
        plt.savefig(Path('.outputs'))
        plt.show()

    def plot_lung_function_categories_by_age_group(self):
        """ plot for lung function for all age groups in the year 2022 """
        # select logs from the latest year. In this case we are selecting year 2022
        re_ordered_df = self.__copd_prev.reorder_levels([3, 0, 2, 1], axis=1)
        plot_df = pd.DataFrame()
        mask = (re_ordered_df.index > pd.to_datetime('2022-01-01')) & (
            re_ordered_df.index <= pd.to_datetime('2023-01-01'))
        re_ordered_df = re_ordered_df.loc[mask]
        _rows_counter: int = 0  # a counter for plotting. setting rows

        fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)  # plot setup
        for _tob in ['True', 'False']:
            for _lung_func, _ in enumerate(self.__lung_func_cats):
                plot_df[f'{_lung_func}'] = re_ordered_df[f'{_lung_func}']['M'][_tob].sum(axis=0) + \
                                           re_ordered_df[f'{_lung_func}']['F'][_tob].sum(axis=0)

            plot_df = plot_df.apply(lambda row: row / row.sum(), axis=1)
            plot_df.plot(kind='bar', ax=ax[_rows_counter], stacked=True, color=self.__plot_colors,
                         title=f"{self.__smokers_desc[_tob]} lung function categories per each age group in 2022",
                         ylabel="Proportions",
                         xlabel="age group",

                         )
            _rows_counter += 1  # increase row number
        # remove all the subplot legends
        for ax in ax:
            ax.get_legend().remove()

        fontP = FontProperties()
        fontP.set_size('small')

        # add one legend on `plt
        plt.legend(self.__lung_func_cats, title="Lung function categories", bbox_to_anchor=(0.7, 0.74),
                   loc='upper left', prop=fontP)
        plt.savefig(
            outputpath / ('lung_function_categories_by_age_group' + datestamp + ".pdf"), format="pdf"
        )
        plt.savefig(Path('.outputs'))
        plt.show()

    def plot_copd_deaths_by_lungfunction(self):
        """ a function to plot COPD deaths by lung function/obstruction """
        # get COPD deaths by lung function from copd logs
        deaths_lung_func = self.__logs_dict['copd_deaths_lung_func']

        # group by date and lung function
        deaths_grouped = deaths_lung_func.groupby(['date', 'lung_function']).size()
        unstack_df = deaths_grouped.unstack()
        plot_lung_func_deaths = unstack_df.groupby(unstack_df.index.year).sum()
        plot_lung_func_deaths = plot_lung_func_deaths.apply(lambda row: row / row.sum(), axis=1)
        # do plotting
        fig, ax = plt.subplots()
        plot_lung_func_deaths.plot(kind='bar', stacked=True, ax=ax, title="COPD deaths by lung function",
                                   color=self.__plot_colors[5:], xlabel="Year",
                                   ylabel="proportion of COPD deaths per each lung function")

        fontP = FontProperties()
        fontP.set_size('small')

        # add one legend on `plt
        plt.legend(self.__lung_func_cats[5:], title="Lung function categories", loc='upper right', prop=fontP)
        plt.savefig(
            outputpath / ('copd_deaths_by_lung_function' + datestamp + ".pdf"), format="pdf"
        )
        plt.savefig(Path('.outputs'))
        plt.show()

    def plot_modal_gbd_deaths_by_gender(self):
        """ compare modal and GBD deaths by gender """
        death_compare = compare_number_of_deaths(self.__logfile_path, resourcefilepath)
        fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
        for _col, sex in enumerate(('M', 'F')):
            plot_df = death_compare.loc[(['2010-2014', '2015-2019'], sex, slice(None), 'COPD')].groupby('period').sum()
            ax = plot_df['model'].plot.bar(color=self.__plot_colors[6], label='Model', ax=axs[_col], rot=0)
            ax.errorbar(x=plot_df['model'].index, y=plot_df.GBD_mean,
                        yerr=[plot_df.GBD_lower, plot_df.GBD_upper],
                        fmt='o', color='#000', label="GBD")
            # ax.set_title(f'{self.__gender_desc[sex]} mean annual deaths, 2010-2019')
            ax.set_title(f'{self.__gender_desc[sex]} COPD deaths, 2010-2019')
            ax.set_xlabel("Time period")
            ax.set_ylabel("Number of deaths")
            ax.legend(loc=2)
        plt.tight_layout()
        plt.savefig(
            outputpath / ('modal_gbd_deaths_by_gender' + datestamp + ".pdf"), format="pdf"
        )
        plt.savefig(Path('.outputs'))
        plt.show()

    def plot_modal_gbd_deaths_by_age_group(self):
        """ compare modal and GBD deaths by age group """
        death_compare = compare_number_of_deaths(self.__logfile_path, resourcefilepath)
        plot_df = death_compare.loc[(['2010-2014', '2015-2019'], slice(None), slice(None), 'COPD')].groupby(
            'age_grp').sum()
        __plot_df = pd.DataFrame(index=['0-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'],
                                 data=[plot_df.iloc[:6].sum(axis=0), plot_df.iloc[6:8].sum(axis=0),
                                       plot_df.iloc[8:10].sum(axis=0), plot_df.iloc[10:12].sum(axis=0),
                                       plot_df.iloc[12:14].sum(axis=0), plot_df.iloc[14:16].sum(axis=0),
                                       plot_df.iloc[16:18].sum(axis=0)])

        ax = __plot_df['model'].plot.bar(color=self.__plot_colors[6], label='Model', rot=0)
        ax.errorbar(x=__plot_df['model'].index, y=__plot_df.GBD_mean,
                    yerr=[__plot_df.GBD_lower, __plot_df.GBD_upper],
                    fmt='o', color='#000', label="GBD")
        # ax.set_title('Mean annual deaths by age group, 2010-2019')
        ax.set_title('COPD deaths by age group, 2010-2019')
        ax.set_xlabel("Age group")
        ax.set_ylabel("Number of deaths")
        ax.legend(loc=1)
        plt.tight_layout()
        plt.savefig(
            outputpath / ('modal_gbd_deaths_by_age_groups' + datestamp + ".pdf"), format="pdf"
        )
        plt.savefig(Path('.outputs'))
        plt.show()


start_date = Date(2010, 1, 1)
end_date = Date(2030, 1, 1)

resourcefilepath = Path("./resources")  # Path to resource files

outputpath = Path('./outputs')  # path to outputs folder

datestamp = datetime.date.today().strftime("__%Y_%m_%d") + datetime.datetime.now().strftime("%H_%M_%S")


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
sim = get_simulation(50_000)
path_to_logfile = sim.log_filepath

# initialise Copd analyses class
copd_analyses = CopdAnalyses(logfile_path=path_to_logfile)

# plot lung function categories per each category
copd_analyses.plot_lung_function()

# plot moderate or severe lung obstruction in smokers and non-smokers
copd_analyses.mod_sev_exacerb_lung_function_smokers()

# plot lung function categories by gender
copd_analyses.plot_lung_function_by_gender()

# plot moderate and severe exacerbation by gender
copd_analyses.sev_mod_exacerb_lung_function_by_gender()

# plot lung function categories by age group
copd_analyses.plot_lung_function_categories_by_age_group()

# plot modal Copd deaths by lung function
copd_analyses.plot_copd_deaths_by_lungfunction()

# plot modal deaths against GBD deaths by gender
copd_analyses.plot_modal_gbd_deaths_by_gender()

# plot modal deaths against GBD deaths by age groups
copd_analyses.plot_modal_gbd_deaths_by_age_group()
