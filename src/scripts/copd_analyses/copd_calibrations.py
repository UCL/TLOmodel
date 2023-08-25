"""Script for calibrating the copd module """
import datetime
from pathlib import Path

import numpy as np
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
        plt.show()

    def plot_death_rate(self):
        """ copd rate of death per 100 per year """
        output = parse_log_file(self.__logfile_path)
        demog = output['tlo.methods.demography']['death']

        # get scaling factor and scale-up the deaths
        if 'scaling_factor' in output['tlo.methods.population']:
            sf = output['tlo.methods.population']['scaling_factor']['scaling_factor'].values[0]
        else:
            sf = 1.0

        # - extract number of death by year/sex/age-group (copied from utils compare deaths function)
        model = demog.assign(
            year=lambda x: x['date'].dt.year
        ).groupby(
            ['year', 'sex', 'age', 'cause']
        )['person_id'].count().mul(sf)

        #  extract copd category 5 and 6 deaths
        copd_deaths_cat5 = model.loc[(slice(None), slice(None), slice(None), ['COPD_cat5'])].groupby('year').sum()
        copd_deaths_cat6 = model.loc[(slice(None), slice(None), slice(None), ['COPD_cat6'])].groupby('year').sum()

        # construct a dataframe with copd deaths, person days, total population and observed data
        copd_deaths_df = pd.DataFrame(data={'copd_severe_modal': copd_deaths_cat5,
                                            'copd_v_severe_modal': copd_deaths_cat6,
                                            'person_years': np.nan,
                                            'total_pop_21': np.nan,
                                            'obs_data_severe_copd': np.nan,
                                            'obs_data_very_severe_copd': np.nan,
                                            })

        # get total population in 2021
        pop_2021 = output['tlo.methods.demography']['population']
        pop = pop_2021.set_index(pd.to_datetime(pop_2021['date']).dt.year)
        pop_21 = pop.loc[2022, "total"]

        # get person days
        py_ = output['tlo.methods.demography']['person_years']
        tot_py = (
            (py_.loc[pd.to_datetime(py_['date']).dt.year == 2021]['M']).apply(pd.Series) +
            (py_.loc[pd.to_datetime(py_['date']).dt.year == 2021]['F']).apply(pd.Series)
        ).transpose()
        total_person_yrs_21 = tot_py.sum().sum()

        # observed according to copd write-up
        obs_data_sev_and_v_sev = [15.3, 27.8]

        # update the dataframe with person years data
        copd_deaths_df.loc[2021, ['total_pop_21', 'person_years', 'obs_data_severe_copd', 'obs_data_very_severe_copd']]\
            = [pop_21, total_person_yrs_21, obs_data_sev_and_v_sev[0], obs_data_sev_and_v_sev[1]]

        print(f'the output is {copd_deaths_df}')


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
# sim = get_simulation(20000)
# path_to_logfile = sim.log_filepath
path_to_logfile = Path('./outputs/copd_analyses__2023-08-24T170704.log')

# initialise Copd analyses class
copd_analyses = CopdCalibrations(logfile_path=path_to_logfile)

# plot lung function categories per each category
copd_analyses.copd_prevalence()

# calibrate copd death rate
copd_analyses.plot_death_rate()
