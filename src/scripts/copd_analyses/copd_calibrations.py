"""Script for calibrating the copd module """
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file, unflatten_flattened_multi_index_in_logging
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


class CopdCalibrations:
    """ A class demonstrate how closer the copd model is to the real data """

    def __init__(self, logfile_path):
        """ called each time the Copd calibration class is initialised

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

    def copd_prevalence(self):
        """ calibrate the prevalence of mild, moderate, severe, very severe COPD at mean age 49 (SD 17 years)* """
        re_ordered_copd_prev = self.__copd_prev.reorder_levels([1, 2, 3, 0], axis=1)

        re_ordered_copd_prev = re_ordered_copd_prev.iloc[:, 50:].reorder_levels([2, 3, 0, 1], axis=1)
        plot_df = pd.DataFrame()
        for _lung_func, _ in enumerate(self.__lung_func_cats):
            plot_df[_lung_func] = re_ordered_copd_prev[f'{_lung_func}'].sum(axis=1)

        plot_df['date'] = plot_df.index.year
        plot_df = plot_df.groupby(plot_df.date).sum()
        plot_df = plot_df.apply(lambda row: row / row.sum(), axis=1)

        plot_df[['observed_data_3', 'observed_data_4', 'observed_data_5', 'observed_data_6']] = np.nan
        plot_df.loc[2021, ['observed_data_3', 'observed_data_4', 'observed_data_5', 'observed_data_6']] \
            = [0.098, 0.095, 0.016, 0.006]

        lung_categories = ['mild COPD', 'moderate COPD', 'severe COPD', 'very severe COPD']
        _col_counter: int = 0  # a counter for plotting. setting rows
        fig, axes = plt.subplots(ncols=4, sharey=True)  # plot setup

        for _lung_func in [3, 4, 5, 6]:
            # do plotting
            ax = plot_df.plot.line(ax=axes[_col_counter], title=f"{lung_categories[_col_counter]} plot",
                                   y=[_lung_func],
                                   ylim=[0, 0.18],
                                   xlabel="Year",
                                   ylabel="Proportions")

            plot_df.plot.line(
                y=[f'observed_data_{_lung_func}'],
                marker='^',
                color='red',
                ax=ax
            )
            ax.legend(["modal", "observed data"])
            _col_counter += 1  # increment column counter
        plt.figtext(0.5, 0.01, "Prevalence of mild, moderate, severe, very severe COPD at mean age 49 (SD 17 years)*",
                    ha="center", bbox={"facecolor": "grey", "alpha": 0.5, "pad": 5})
        plt.show()

    def rate_of_death_by_lungfunction_category(self):
        """Make the comparison to Alupo P et al. (2021) study. This study found that the rate of COPD death was as
        follows:
         Persons with Mild COPD Stage: 3.8 all-cause deaths per 100 person-years
         Persons with Moderate COPD Stage: 5.1 all-cause deaths per 100 person-years
         Persons with Severe COPD Stage: 15.3 all-cause deaths per 100 person-years
         Persons with Very Severe COPD Stage: 27.8 all-cause deaths per 100 person-years
        We will compare this with fraction of people that die each year, averaged over many years, as an estimate of the
        risk of persons in each stage dying.
        We assume that the disease stages in the model correspond to the study as:
        Mild - Stages 1, 2
        Moderate - Stages 3, 4
        Severe - Stage 5
        Very Severe - Stage 6
        """

        # Get the number of deaths each year in each category of lung function
        output = parse_log_file(self.__logfile_path)  # parse output file

        # read deaths from demography detail logger
        demog = output['tlo.methods.demography.detail']['properties_of_deceased_persons']

        # only consider deaths from individuals above 30
        demog = demog.loc[demog.age_years > 30]

        # assign a function that groups deaths by year and lung function thereafter do count
        all_deaths = demog.assign(
            year=lambda x: x['date'].dt.year,
            cat=lambda x: x['ch_lungfunction']).groupby(['year', 'cat'])['cause_of_death'].count()

        # re-construct the dataframe by transforming lung function categories into columns
        all_deaths_df = all_deaths.unstack().assign(
            none=lambda x: x[0],
            mild=lambda x: x[1] + x[2],
            moderate=lambda x: x[3] + x[4],
            severe=lambda x: x[5],
            very_severe=lambda x: x[6]).drop(columns=[0, 1, 2, 3, 4, 5, 6])

        # Get the number of person each year in each category of lung function (irrespective of sex/age/smokingstatus)
        # average within the year
        prev = self.construct_dfs()['copd_prevalence']
        prev = (prev.groupby(axis=1, by=prev.columns.droplevel([0, 1, 2])).sum()
                .groupby(axis=0, by=prev.index.year).mean())
        prev['none'] = prev['0']
        prev['mild'] = prev['1'] + prev['2']
        prev['moderate'] = prev['3'] + prev['4']
        prev['severe'] = prev['5']
        prev['very_severe'] = prev['6']
        prev = prev.drop(columns=['0', '1', '2', '3', '4', '5', '6'])

        # Compute fraction that die each year in each category of lung function, average over many years and compare to
        # data
        death_rate_per100 = (100 * all_deaths_df.loc[[2021]] / prev.loc[[2021]]).mean()

        model_and_observed_data_dict = {'model': [death_rate_per100.mild, death_rate_per100.moderate,
                                                  death_rate_per100.severe, death_rate_per100.very_severe],
                                        'data': [3.8, 5.1, 15.3, 27.8]
                                        }

        # plot rate of death (per 100 per year) by COPD stage (mild, moderate, severe, very severe)
        plot_rate_df = pd.DataFrame(index=['mild', 'moderate', 'severe', 'very_severe'],
                                    data=model_and_observed_data_dict)
        fig, axes = plt.subplots(ncols=4, sharey=True)
        _col_counter = 0
        for _label in plot_rate_df.index:
            # do plotting
            ax = plot_rate_df.iloc[_col_counter:1 + _col_counter].plot.line(ax=axes[_col_counter],
                                                                            title=f"{_label} COPD",
                                                                            y='model',
                                                                            marker='o',
                                                                            color='blue',
                                                                            ylabel="rate of COPD death per 100 "
                                                                                   "person years"
                                                                            )

            plot_rate_df.iloc[_col_counter:1 + _col_counter].plot.line(
                y='data',
                marker='^',
                color='red',
                ax=ax
            )
            _col_counter += 1  # increment column counter
        # remove all the subplot legends
        for ax in axes:
            ax.get_legend().remove()

        fontP = FontProperties()
        fontP.set_size('small')

        # set legend
        legend_keys = ['Model (Risk of death per 100 persons)', 'Data (Rate of death per 100 person-years)']
        plt.legend(legend_keys, bbox_to_anchor=(0.1, 0.74), loc='upper left', prop=fontP)
        plt.figtext(0.5, 0.01, "Rate of death (per 100 per year) by COPD stage (mild, moderate, severe, very severe)",
                    ha="center", bbox={"facecolor": "grey", "alpha": 0.5, "pad": 5})

        # show plot
        plt.show()

        # COMPUTE THE RELATIVE RATES TO NONE
        rr_death_rate_per100 = (100 * all_deaths_df.loc[[2023]] / prev.loc[[2023]]).mean()
        rr_rates = rr_death_rate_per100 / rr_death_rate_per100.loc['none']

        rr_model_and_observed_data_dict = {'model': [rr_rates.none, rr_rates.mild, rr_rates.moderate,
                                                     rr_rates.severe + rr_rates.very_severe],
                                           'data': [1.0, 2.4, 3.5, 6.6]
                                           }

        # plot relative rate of (all cause) death according to COPD stage (none, mild, moderate, severe + v severe)
        plot_rr_rate_df = pd.DataFrame(index=['none', 'mild', 'moderate', 'severe_and_very_severe'],
                                       data=rr_model_and_observed_data_dict)
        fig, axes = plt.subplots(ncols=4, sharey=True)
        _col_counter = 0
        for _label in plot_rr_rate_df.index:
            # do plotting
            ax = plot_rr_rate_df.iloc[_col_counter:1 + _col_counter].plot.line(ax=axes[_col_counter],
                                                                               title=f"{_label} COPD",
                                                                               y='model',
                                                                               marker='o',
                                                                               color='blue',
                                                                               ylabel="relative rate of COPD death per "
                                                                                      "100 "
                                                                                      "person years"
                                                                               )

            plot_rr_rate_df.iloc[_col_counter:1 + _col_counter].plot.line(
                y='data',
                marker='^',
                color='red',
                ax=ax
            )
            _col_counter += 1  # increment column counter
        # remove all the subplot legends
        for ax in axes:
            ax.get_legend().remove()

        fontP = FontProperties()
        fontP.set_size('small')
        # set legend
        plt.legend(['Model', 'Data'], bbox_to_anchor=(0.1, 0.74), loc='upper left', prop=fontP)
        plt.figtext(0.5, 0.01, "Relative rate of (all cause) death according to COPD stage (none, mild, moderate, "
                               "severe + v severe)", ha="center", bbox={"facecolor": "grey", "alpha": 0.5, "pad": 5})
        # show plot
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
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography.detail': logging.INFO,
                'tlo.methods.copd': logging.INFO,
            }
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
sim = get_simulation(50000)
path_to_logfile = sim.log_filepath

# initialise Copd analyses class
copd_analyses = CopdCalibrations(logfile_path=path_to_logfile)

# plot lung function categories per each category
copd_analyses.copd_prevalence()

# Examine rate of death by lung function
copd_analyses.rate_of_death_by_lungfunction_category()
