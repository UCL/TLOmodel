"""
Following the skeleton method for HIV

Q: should treatment be in a separate method?
"""

# import any methods from other modules, e.g. for parameter definitions
from typing import Any, Union

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent

# need to import ART, ART_Event, TB

import numpy as np
import pandas as pd

# NOTES: what should the functions be returning?
# previously they read in the population dataframe and then returned the modified population dataframe
# how to deal with the current_time variable needed in many functions?
# check use of self
# initialise population function was renamed as there were unresolved differences


# read in data files #
# use function read.parameters in class HIV to do this?
file_path = 'Q:\Thanzi la Onse\HIV\Method_HIV.xlsx'

HIV_prev = pd.read_excel(file_path, sheet_name='prevalence2018', header=0)  # July 1st estimates not full year

HIV_death = pd.read_excel(file_path, sheet_name='deaths2009_2021', header=0)

HIV_inc = pd.read_excel(file_path, sheet_name='incidence2009_2021', header=0)

ad_mort = pd.read_excel(file_path, sheet_name='mortality_rates', header=0)

paed_mortART = pd.read_excel(file_path, sheet_name='paediatric_mortality_rates', header=0)

CD4_base_M = pd.read_excel(file_path, sheet_name='CD4_distribution2018', header=0)

age_distr = pd.read_excel(file_path, sheet_name='age_distribution2018', header=0)

inds = pd.read_csv('Q:/Thanzi la Onse/HIV/initial_pop_dataframe2018.csv')
p = inds.shape[0]  # number of rows in pop (# individuals)


# sim_size = int(100)
# current_time = 2018


# HELPER FUNCTION - should these go in class(HIV)?
# are they static methods?

# untreated HIV mortality rates - annual, adults
def log_scale(a0):
    age_scale = 2.55 - 0.025 * (a0 - 30)
    return age_scale


# this class contains all the methods required to set up the baseline population
class HIV(Module):
    """Models HIV incidence, treatment and AIDS-mortality.

    Methods required:
    * `read_parameters(data_folder)`
    * `initialise_population(population)`
    * `initialise_simulation(sim)`
    * `on_birth(mother, child)`
    """

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'prob_infant_fast_progressor': Parameter(
            Types.LIST,
            'Probabilities that infants are fast or slow progressors'),
        'infant_progression_category': Parameter(
            Types.CATEGORICAL,
            'Classification of infants into fast or slow progressors'),
        'exp_rate_mort_infant_fast_progressor': Parameter(
            Types.REAL,
            'Exponential rate parameter for mortality in infants fast progressors'),
        'weibull_scale_mort_infant_slow_progressor': Parameter(
            Types.REAL,
            'Weibull scale parameter for mortality in infants slow progressors'),
        'weibull_shape_mort_infant_slow_progressor': Parameter(
            Types.REAL,
            'weibull shape parameter for mortality in infants slow progressors'),
        'weibull_shape_mort_adult': Parameter(
            Types.REAL,
            'Weibull shape parameter for mortality in adults'),
        'proportion_high_sexual_risk_male': Parameter(
            Types.REAL,
            'proportion of men who have high sexual risk behaviour'),
        'proportion_high_sexual_risk_female': Parameter(
            Types.REAL,
            'proportion of women who have high sexual risk behaviour'),
        'rr_HIV_high_sexual_risk': Parameter(
            Types.REAL,
            'relative risk of acquiring HIV with high risk sexual behaviour'),
        'proportion_on_ART_infectious': Parameter(
            Types.REAL,
            'proportion of people on ART contributing to transmission as not virally suppressed'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'has_HIV': Property(Types.BOOL, 'HIV status'),
        'date_HIV_infection': Property(Types.DATE, 'Date acquired HIV infection'),
        'date_AIDS_death': Property(Types.DATE, 'Projected time of AIDS death if untreated'),
        'on_ART': Property(Types.BOOL, 'Currently on ART'),
        'date_ART_start': Property(Types.DATE, 'Date ART started'),
        'ART_mortality': Property(Types.REAL, 'Mortality rates whilst on ART'),
        'sexual_risk_group': Property(Types.REAL, 'Relative risk of HIV based on sexual risk high/low'),
        'date_death': Property(Types.DATE, 'Date of death'),
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """

        params = self.parameters
        params['prob_infant_fast_progressor'] = [0.36, 1 - 0.36]
        params['infant_progression_category'] = ['FAST', 'SLOW']
        params['exp_rate_mort_infant_fast_progressor'] = 1.08
        params['weibull_scale_mort_infant_slow_progressor'] = 16
        params['weibull_shape_mort_infant_slow_progressor'] = 2.7
        params['weibull_shape_mort_adult'] = 2
        params['proportion_high_sexual_risk_male'] = 0.0913
        params['proportion_high_sexual_risk_female'] = 0.0095
        params['rr_HIV_high_sexual_risk'] = 2
        params['proportion_on_ART_infectious'] = 0.2

    def high_risk(self, df):  # should this be in initialise population?
        """ Stratify the adult (age >15) population in high or low sexual risk """

        params = self.module.parameters

        tmp = df.index[(df.sex == 'M') & (df.age >= 15)]
        tmp2 = np.random.choice(tmp, size=int(round(
            (params['proportion_high_sexual_risk_male'] * len(tmp)))), replace=False)

        tmp3 = df.index[(df.sex == 'F') & (df.age >= 15)]
        tmp4 = np.random.choice(tmp3, size=int(round(
            (params['proportion_high_sexual_risk_male'] * len(tmp3)))), replace=False)

        high_risk_index = np.concatenate([tmp2, tmp4])

        df.loc[high_risk_index, 'sexual_risk_group'] = params['rr_HIV_high_sexual_risk']

    # assign infected status using UNAIDS prevalence 2018 by age
    # randomly allocate time since infection according to CD4 distributions from spectrum
    # should do this separately for infants using CD4%
    # then could include the infant fast progressors
    # currently infant fast progressors will always have time to death shorter than time infected
    def prevalence(self, df, current_time):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        self.current_time = current_time

        self.CD4_times = [self.current_time - 3.2, self.current_time - 7.8,
                          self.current_time - 11.0, self.current_time - 13.9]
        self.prob_CD4 = [0.34, 0.31, 0.27, 0.08]

        for i in range(0, 81):
            # male
            # scale high/low-risk probabilities to sum to 1 for each sub-group
            prob_i = df['sexual_risk_group'][(df.age == i) & (df.sex == 'M')] / \
                     np.sum(df['sexual_risk_group'][(df.age == i) & (df.sex == 'M')])

            # sample from uninfected df using prevalence from UNAIDS
            tmp5 = np.random.choice(df.index[(df.age == i) & (df.sex == 'M')],
                                    size=int(
                                        (HIV_prev['prevalence'][(HIV_prev.year == self.current_time) &
                                                                (HIV_prev.sex == 'M') &
                                                                (HIV_prev.age == i)])),
                                    replace=False, p=prob_i)

            df.loc[tmp5, 'has_HIV'] = 1  # change status to infected

            df.loc[tmp5, 'date_HIV_infection'] = np.random.choice(self.CD4_times, size=len(tmp5),
                                                                          replace=True, p=self.prob_CD4)

            # female
            # scale high/low-risk probabilities to sum to 1 for each sub-group
            prob_i = df['sexual_risk_group'][(df.age == i) & (df.sex == 'F')] / \
                     np.sum(df['sexual_risk_group'][(df.age == i) & (df.sex == 'F')])

            # sample from uninfected df using prevalence from UNAIDS
            tmp6 = np.random.choice(df.index[(df.age == i) & (df.sex == 'F')],
                                    size=int(
                                        (HIV_prev['prevalence'][(HIV_prev.year == self.current_time) &
                                                                (HIV_prev.sex == 'F') &
                                                                (HIV_prev.age == i)])),
                                    replace=False, p=prob_i)

            df.loc[tmp6, 'has_HIV'] = 1  # change status to infected

            df.loc[tmp6, 'date_HIV_infection'] = np.random.choice(self.CD4_times, size=len(tmp6),
                                                                          replace=True, p=self.prob_CD4)

        # check time infected is less than time alive (especially for infants)
        tmp = df.index[(pd.notna(df.date_HIV_infection)) &
                               ((current_time - df.date_HIV_infection) > df.age)]
        tmp2 = current_time - df.loc[tmp, 'age']
        df.loc[tmp, 'date_HIV_infection'] = tmp2  # replace with year of birth

        return df


    # this function needs the ART mortality rates from ART.py
    def initial_pop_deaths(self, df, current_time):

        self.current_time = current_time

        params = self.module.parameters

        # PAEDIATRIC time of death - untreated
        self.hiv_inf = df.index[(df.has_HIV == 1) & (df.on_ART == 0) & (df.age < 3)]

        # need a two parameter Weibull with size parameter, multiply by scale instead
        time_of_death_slow = np.random.weibull(a=params['weibull_size_mort_infant_slow_progressor'],
                                               size=len(self.hiv_inf)) * \
                             params['weibull_scale_mort_infant_slow_progressor']

        # while time of death is shorter than time infected keep redrawing (only for the entries that need it)
        while np.any(
            time_of_death_slow < (self.current_time - df.loc[
                self.hiv_inf, 'date_HIV_infection'])):  # if any condition=TRUE for any rows

            redraw = np.argwhere(
                time_of_death_slow < (self.current_time - df.loc[self.hiv_inf, 'date_HIV_infection']))
            redraw2 = redraw.ravel()

            if len(redraw) == 0:
                break

            # redraw time of death
            time_of_death_slow[redraw2] = np.random.weibull(a=params['weibull_size_mort_infant_slow_progressor'],
                                                            size=len(redraw2)) * \
                                          params['weibull_scale_mort_infant_slow_progressor']

        # subtract time already spent
        df.loc[self.hiv_inf, 'date_AIDS_death'] = self.current_time + time_of_death_slow - (
            self.current_time - df.loc[self.hiv_inf, 'date_HIV_infection'])

        # ADULT time of death, adults are all those aged >3 for untreated mortality rates
        self.hiv_ad = df.index[(df.has_HIV == 1) & (df.on_ART == 0) & (df.age >= 3)]

        time_of_death = np.random.weibull(a=params['weibull_shape_mort_adult'], size=len(self.hiv_ad)) * \
                        np.exp(log_scale(df.loc[self.hiv_ad, 'age']))

        # while time of death is shorter than time infected keep redrawing (only for entries that need it)
        while np.any(
            time_of_death < (self.current_time - df.loc[
                self.hiv_ad, 'date_HIV_infection'])):  # if any condition=TRUE for any rows

            redraw = np.argwhere(
                time_of_death < (self.current_time - df.loc[self.hiv_ad, 'date_HIV_infection']))
            redraw2 = redraw.ravel()

            if len(redraw) < 10:  # this condition needed for older people with long time since infection
                break

            age_index = self.hiv_ad[redraw2]

            time_of_death[redraw2] = np.random.weibull(a=params['weibull_shape_mort_adult'], size=len(redraw2)) * \
                                     np.exp(log_scale(df.loc[age_index, 'age']))

        # subtract time already spent
        df.loc[self.hiv_ad, 'date_AIDS_death'] = self.current_time + time_of_death - \
                                                         (self.current_time - df.loc[
                                                             self.hiv_ad, 'date_HIV_infection'])

        # assign mortality rates on ART
        df['ART_mortality'] = self.ART_mortality_rates(df, self.current_time)

        return df

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        raise NotImplementedError

    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """
        raise NotImplementedError


class HIV_Event(RegularEvent, PopulationScopeEventMixin):
    """HIV infection events

    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module):
        """One line summary here

        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.

        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=1))

    def HIV_infection_adults(self, df, current_time, beta_ad):
        """Apply this event to the population.

        :param population: the current population
        """
        self.current_time = current_time

        params = self.module.parameters

        infected = len(
            df[(df.has_HIV == 1) & (df.on_ART == 0) & (
                    df.age >= 15)])  # number infected untreated

        h_infected = params['proportion_on_ART_infectious'] * len(
            df[(df.has_HIV == 1) & (df.on_ART == 1) & (
                    df.age >= 15)])  # number infected treated

        total_pop = len(df[(df.age >= 15)])  # whole df over 15 years

        foi = beta_ad * ((infected + h_infected) / total_pop)  # force of infection for adults

        # distribute FOI by age
        foi_m = foi * age_distr['age_distribution'][
            (age_distr.year == self.current_time) & (age_distr.sex == 'M')]  # age 15-80+
        foi_f = foi * age_distr['age_distribution'][(age_distr.year == self.current_time) & (age_distr.sex == 'F')]

        for i in range(66):  # ages 15-80
            age_value = i + 14  # adults only FOI

            # males
            susceptible_age = len(
                df[(df.age == age_value) & (df.sex == 'M') & (df.onART == 0)])

            # to determine number of new infections by age
            tmp1 = np.random.binomial(1, p=foi_m[i], size=susceptible_age)

            # allocate infections to people with high/low risk
            # scale high/low-risk probabilities to sum to 1 for each sub-group
            risk = df['sexual_risk_group'][
                       (df.age == age_value) & (df.sex == 'M') & (df.has_HIV == 0)] / \
                   np.sum(df['sexual_risk_group'][
                              (df.age == age_value) & (df.sex == 'M') & (df.has_HIV == 0)])

            tmp2 = np.random.choice(
                df.index[(df.age == age_value) & (df.sex == 'M') & (df.has_HIV == 0)],
                size=len(tmp1), p=risk, replace=False)

            df.loc[tmp2, 'has_HIV'] = 1  # change status to infected
            df.loc[tmp2, 'date_HIV_infection'] = self.current_time

            df.loc[tmp2, 'date_AIDS_death'] = self.current_time + (
                np.random.weibull(a=params['weibull_shape_mort_adult'], size=len(tmp2)) * np.exp(
                log_scale(df.age.iloc[tmp2])))

            # females
            susceptible_age = len(
                df[(df.age == age_value) & (df.sex == 'F') & (df.has_HIV == 0)])

            # to determine number of new infections by age
            tmp3 = np.random.binomial(1, p=foi_f[i], size=susceptible_age)

            # allocate infections to people with high/low risk
            # scale high/low-risk probabilities to sum to 1 for each sub-group
            risk = df['sexual_risk_group'][
                       (df.age == age_value) & (df.sex == 'F') & (df.has_HIV == 0)] / \
                   np.sum(df['sexual_risk_group'][
                              (df.age == age_value) & (df.sex == 'F') & (df.has_HIV == 0)])

            tmp4 = np.random.choice(
                df.index[(df.age == age_value) & (df.sex == 'F') & (df.has_HIV == 0)],
                size=len(tmp3), p=risk, replace=False)

            df.loc[tmp4, 'has_HIV'] = 0  # change status to infected
            df.loc[tmp4, 'date_HIV_infection'] = self.current_time

            df.loc[tmp2, 'date_AIDS_death'] = self.current_time + (
                np.random.weibull(a=params['weibull_shape_mort_adult'], size=len(tmp4)) * np.exp(
                log_scale(df.age.iloc[tmp4])))

        return df


    # run the death functions once a year
    def AIDS_death(self, df, current_time):

        self.current_time = current_time

        # choose which ones die at current_time
        current_time_int = int(round(self.current_time))  # round current_time to nearest year

        tmp = df.index[(round(df.date_AIDS_death) == current_time_int) & (df.on_ART == 0)]

        df.loc[tmp, 'date_death'] = self.current_time

        return df


    def AIDS_death_on_ART(self, df):

        tmp1 = np.random.uniform(low=0, high=1, size=df.shape[0])  # random number for every entry

        tmp2 = df.index[(pd.notna(df.ART_mortality)) & (tmp1 < df['mortality']) &
                          (df.has_HIV == 1) & (df.on_ART == 1)]

        df.loc[tmp2, 'date_death'] = self.current_time

        return df


# # to set up the baseline population
# inds.head(20)
# inds.describe(include='all')
# inds['status'].value_counts()
#
# inds = high_risk_inds(inds)
# inds = prevalence_inds(inds, current_time)
# inds = initART_inds(inds, current_time)
# inds = init_death_inds(inds, current_time)
#
# inds.head(20)
# inds.describe(include='all')
# inds['status'].value_counts()
# inds['treat'].value_counts()
#
# # to run projections set up these commands in a loop
# # example for 2019 projections with placeholder transmission rate beta_ad=0.12
# inds = inf_inds_ad(inds, 2019, 0.12)
# inds = ART_inds(inds, 2019)
# inds = killHIV_inds(inds, 2019)
# inds = killHIV_ART_inds(inds)
#
# inds.describe(include='all')
# inds['status'].value_counts()
# inds['treat'].value_counts()

# TODO: handle births. link child's risk of HIV to mother's HIV status
# TODO: include cotrimoxazole for children
# TODO: code FOI as separate function from infection function
# TODO: incorporate TB as a risk for HIV progression/mortality

