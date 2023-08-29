"""Script for calibrating the copd module """
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation
from tlo.analysis.utils import (
    make_calendar_period_lookup,
    make_calendar_period_type,
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

    def death_rate(self):
        """ in this function we are calibrating;
            1.  Rate of death (per 100 per year) by COPD stage (mild, moderate, severe, very severe)
            2.  Relative rate of (all cause) death according to COPD stage (none, mild, moderate, severe + v severe)

        For deaths rate calculations
            death rate for copd category 5
                1.  get all deaths in copd category 5 in the period 2020-2024
                2.  get total person days in 2020-2024
                3.  divide deaths in category 5 by person years

             death rate for copd category 6
                1.  get all deaths in copd category 6 in the period 2020-2024
                2.  get total person days in the period 2020-2024
                3.  divide deaths in category 6 by person years

        For relative rate calculations
            relative death rate for copd category 5
                1.  divide deaths in category 5 by person years in the period 2020-2024
                2.  divide non copd deaths by person years in the period 2020-2024
                3.  divide result from step 1 by result from step 2

            relative death rate for copd category 6
                1.  divide deaths in category 6 by person years in the period 2020-2024
                2.  divide non copd deaths by person years in the period 2020-2024
                3.  divide result from step 1 by result from step 2
        """
        output = parse_log_file(self.__logfile_path)  # parse output file
        demog = output['tlo.methods.demography']['death']  # read deaths from demography module

        # create a dataframe that will contain data for calibration
        calibrate_copd_deaths_df = pd.DataFrame()

        # - extract number of death by year/sex/age-group (copied from utils compare deaths function)
        model = demog.assign(
            year=lambda x: x['date'].dt.year).groupby(['year', 'sex', 'age', 'cause'])['person_id'].count()

        # extract copd deaths and multiply them by 100(this is the unit according to observed data). Currently, we only
        # have people dying from copd category 5 and 6
        copd_deaths_cat5 = model.loc[(slice(None), slice(None), slice(None), ['COPD_cat5'])].groupby('year').sum() * 100
        copd_deaths_cat6 = model.loc[(slice(None), slice(None), slice(None), ['COPD_cat6'])].groupby('year').sum() * 100

        # update copd calibration dataframe with copd deaths categories 5 and 6
        calibrate_copd_deaths_df['copd_cat5_deaths'] = copd_deaths_cat5
        calibrate_copd_deaths_df['copd_cat6_deaths'] = copd_deaths_cat6

        # get person days from demography logs. Here we're getting person years for all ages for the simulation period
        py_ = output['tlo.methods.demography']['person_years']
        py_ = py_.set_index(py_.date.dt.year).drop(columns='date')
        tot_py = (
            (py_.loc[py_.index]['M']).apply(pd.Series) +
            (py_.loc[py_.index]['F']).apply(pd.Series)
        )

        # get the total non-copd deaths and add them to copd calibration dataframe
        all_deaths = model.groupby('year').sum() * 100
        non_copd_deaths = all_deaths - (copd_deaths_cat5 + copd_deaths_cat6)
        calibrate_copd_deaths_df['other_deaths'] = non_copd_deaths
        calibrate_copd_deaths_df['person_years'] = tot_py.sum(axis=1)

        # group years in 5-year period
        calperiods, calperiodlookup = make_calendar_period_lookup()
        calibrate_copd_deaths_df = calibrate_copd_deaths_df.reset_index()
        calibrate_copd_deaths_df['period'] = \
            calibrate_copd_deaths_df['year'].map(calperiodlookup).astype(make_calendar_period_type())
        calibrate_copd_deaths_df = calibrate_copd_deaths_df.drop(columns=['year']).groupby('period').sum()

        # add columns for observed data for copd categories 5 and 6
        calibrate_copd_deaths_df['obs_data_copd_cat5'] = np.nan
        calibrate_copd_deaths_df['obs_data_copd_cat6'] = np.nan

        # get observed according to copd write-up
        obs_data_sev_and_v_sev = [15.3, 27.8]

        # update the dataframe with person years data
        calibrate_copd_deaths_df.loc[['2020-2024'], ['obs_data_copd_cat5', 'obs_data_copd_cat6']] \
            = [obs_data_sev_and_v_sev[0], obs_data_sev_and_v_sev[1]]

        calibrate_copd_deaths_df['severe_deaths'] = \
            calibrate_copd_deaths_df.loc[['2020-2024'], 'copd_cat5_deaths'] / \
            calibrate_copd_deaths_df.loc[['2020-2024'], 'person_years']

        calibrate_copd_deaths_df['v_severe_deaths'] = \
            calibrate_copd_deaths_df.loc[['2020-2024'], 'copd_cat6_deaths'] \
            / calibrate_copd_deaths_df.loc[['2020-2024'], 'person_years']

        # do plotting
        fig, axes = plt.subplots(ncols=2, sharey=True)  # plot setup
        ncols = 0

        _titles_dict = {'severe_deaths': 'severe',
                        'v_severe_deaths': 'very severe'}
        for _key, _title in _titles_dict.items():
            ax = calibrate_copd_deaths_df.plot.line(ax=axes[ncols],
                                                    title=f"Rate of death (per 100 per year) by {_title} COPD",
                                                    y=[_key],
                                                    marker='^',
                                                    color='blue',
                                                    xlabel="Year",
                                                    ylabel="death rate")

            calibrate_copd_deaths_df.plot.line(
                y=[f'obs_data_copd_cat{5 + ncols}'],
                marker='^',
                color='red',
                ax=ax
            )
            ax.legend(["modal", "observed data"])
            ncols += 1
        plt.show()

        # ------- PLOT RELATIVE DEATHS--------

        # add columns for observed data
        calibrate_copd_deaths_df['obs_data_rr_copd_cat5'] = np.nan
        calibrate_copd_deaths_df['obs_data_rr_copd_cat6'] = np.nan

        # get observed according to copd write-up
        obs_data_rr_sev_and_v_sev = [3.5, 6.6]

        # update the dataframe with person years data
        calibrate_copd_deaths_df.loc[['2020-2024'], ['obs_data_rr_copd_cat5', 'obs_data_rr_copd_cat6']] \
            = [obs_data_rr_sev_and_v_sev[0], obs_data_rr_sev_and_v_sev[1]]

        # get rate of death for non copd death
        calibrate_copd_deaths_df['r_other_deaths'] = \
            calibrate_copd_deaths_df.loc[['2020-2024'], 'other_deaths'] / \
            calibrate_copd_deaths_df.loc[['2020-2024'], 'person_years']

        # get relative rate of death for severe COPD
        calibrate_copd_deaths_df['rr_severe_deaths'] = \
            calibrate_copd_deaths_df.loc[['2020-2024'], 'severe_deaths'] / \
            calibrate_copd_deaths_df.loc[['2020-2024'], 'r_other_deaths']

        # get relative rate of death very severe COPD
        calibrate_copd_deaths_df['rr_v_severe_deaths'] = \
            calibrate_copd_deaths_df.loc[['2020-2024'], 'v_severe_deaths'] / \
            calibrate_copd_deaths_df.loc[['2020-2024'], 'r_other_deaths']

        # do plotting
        fig, axes = plt.subplots(ncols=2, sharey=True)  # plot setup
        ncols = 0

        _titles_dict = {'rr_severe_deaths': 'severe',
                        'rr_v_severe_deaths': 'very severe'}
        for _key, _title in _titles_dict.items():
            ax = calibrate_copd_deaths_df.plot.line(ax=axes[ncols],
                                                    title=f"relative rate of death according to {_title} COPD",
                                                    y=[_key],
                                                    marker='^',
                                                    color='blue',
                                                    xlabel="Year",
                                                    ylabel="death rate")

            calibrate_copd_deaths_df.plot.line(
                y=[f'obs_data_rr_copd_cat{5 + ncols}'],
                marker='^',
                color='red',
                ax=ax
            )
            ax.legend(["modal", "observed data"])
            ncols += 1
        plt.show()

    def rate_of_death_by_lungfunction_category(self):
        """Make the comparison to Alupo P et al. (2021) study. This study found that the rate of COPD death was as
        follows:
         Persons with Mild COPD Stage: 3.8 deaths per 100 person-years
         Persons with Moderate COPD Stage: 5.1 deaths per 100 person-years
         Persons with Severe COPD Stage: 15.3 deaths per 100 person-years
         Persons with Very Severe COPD Stage: 27.8 deaths per 100 person-years
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
        demog = output['tlo.methods.demography']['death']  # read deaths from demography module

        model = demog.assign(
            year=lambda x: x['date'].dt.year).groupby(['year', 'sex', 'age', 'cause'])['person_id'].count()

        # extract copd deaths:
        copd_deaths = pd.concat(
            {
                'mild': (
                    model.loc[(slice(None), slice(None), slice(None), ['COPD_cat1'])].groupby('year').sum()
                    + model.loc[(slice(None), slice(None), slice(None), ['COPD_cat2'])].groupby('year').sum()
                ),
                'moderate': (
                    model.loc[(slice(None), slice(None), slice(None), ["COPD_cat3"])].groupby("year").sum()
                    + model.loc[(slice(None), slice(None), slice(None), ["COPD_cat4"])].groupby("year").sum()
                ),
                'severe': model.loc[(slice(None), slice(None), slice(None), ['COPD_cat5'])].groupby('year').sum(),
                'very_severe': model.loc[(slice(None), slice(None), slice(None), ['COPD_cat6'])].groupby('year').sum(),
            },
            axis=1
        ).fillna(0).astype(float)

        # Get the number of person each year in each category of lung function (irrespective of sex/age/smokingstatus)
        # average within the year
        prev = self.construct_dfs()['copd_prevalence']
        prev = (prev.groupby(axis=1, by=prev.columns.droplevel([0, 1, 2])).sum()
                .groupby(axis=0, by=prev.index.year).mean())
        prev['mild'] = prev['1'] + prev['2']
        prev['moderate'] = prev['3'] + prev['4']
        prev['severe'] = prev['5']
        prev['very_severe'] = prev['6']
        prev = prev.drop(columns=['0', '1', '2', '3', '4', '5', '6'])

        # Compute fraction that die each year in each category of lung function, average over many years and compare to
        # data
        death_rate_per100 = (100 * copd_deaths / prev).mean()
        print(death_rate_per100)
        # mild           0.000000
        # moderate       0.000000
        # severe         1.674310
        # very_severe    4.507594


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
sim = get_simulation(50000)
path_to_logfile = sim.log_filepath

# initialise Copd analyses class
copd_analyses = CopdCalibrations(logfile_path=path_to_logfile)

# plot lung function categories per each category
copd_analyses.copd_prevalence()

# calibrate copd death rate
copd_analyses.death_rate()

# Examine rate of daath by lung function
copd_analyses.rate_of_death_by_lungfunction_category()
