"""
HIV infection event
"""

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent, IndividualScopeEventMixin


class hiv(Module):
    """
    baseline hiv infection
    """

    def __init__(self, name=None, workbook_path=None):
        super().__init__(name)
        self.workbook_path = workbook_path
        self.store = {'Time': [], 'Total_HIV': [], 'HIV_scheduled_deaths': []}
        self.store_DeathsLog = {'DeathEvent_Time': [], 'DeathEvent_Age': [], 'DeathEvent_Cause': []}

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'prob_infant_fast_progressor':
            Parameter(Types.LIST, 'Probabilities that infants are fast or slow progressors'),
        'infant_fast_progression':
            Parameter(Types.BOOL, 'Classification of infants as fast progressor'),
        'exp_rate_mort_infant_fast_progressor':
            Parameter(Types.REAL, 'Exponential rate parameter for mortality in infants fast progressors'),
        'weibull_scale_mort_infant_slow_progressor':
            Parameter(Types.REAL, 'Weibull scale parameter for mortality in infants slow progressors'),
        'weibull_shape_mort_infant_slow_progressor':
            Parameter(Types.REAL, 'Weibull shape parameter for mortality in infants slow progressors'),
        'weibull_shape_mort_adult':
            Parameter(Types.REAL, 'Weibull shape parameter for mortality in adults'),
        'proportion_high_sexual_risk_male':
            Parameter(Types.REAL, 'proportion of men who have high sexual risk behaviour'),
        'proportion_high_sexual_risk_female':
            Parameter(Types.REAL, 'proportion of women who have high sexual risk behaviour'),
        'proportion_female_sex_workers':
            Parameter(Types.REAL, 'proportion of women who engage in transactional sex'),
        'rr_HIV_high_sexual_risk':
            Parameter(Types.REAL, 'relative risk of acquiring HIV with high risk sexual behaviour'),
        'rr_HIV_high_sexual_risk_fsw':
            Parameter(Types.REAL, 'relative risk of acquiring HIV with female sex work'),
        'proportion_on_ART_infectious':
            Parameter(Types.REAL, 'proportion of people on ART contributing to transmission as not virally suppressed'),
        'beta':
            Parameter(Types.REAL, 'transmission rate'),
        'irr_hiv_f':
            Parameter(Types.REAL, 'incidence rate ratio for females vs males'),
        'prob_mtct':
            Parameter(Types.REAL, 'probability of mother to child transmission'),
        'rr_circumcision':
            Parameter(Types.REAL, 'relative reduction in susceptibility due to circumcision'),
        'rr_behaviour_change':
            Parameter(Types.REAL, 'relative reduction in susceptibility due to behaviour modification'),
        'rel_infectiousness_acute':
            Parameter(Types.REAL, 'relative infectiousness during acute stage'),
        'rel_infectiousness_late':
            Parameter(Types.REAL, 'relative infectiousness during late stage')
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'has_hiv': Property(Types.BOOL, 'HIV status'),
        'date_hiv_infection': Property(Types.DATE, 'Date acquired HIV infection'),
        'date_aids_death': Property(Types.DATE, 'Projected time of AIDS death if untreated'),
        'sexual_risk_group': Property(Types.CATEGORICAL, 'Sexual risk groups', categories=['low', 'high', 'sex_work']),
        'cd4_state': Property(Types.CATEGORICAL, 'CD4 state',
                              categories=['CD1000', 'CD750', 'CD500', 'CD350', 'CD250', 'CD200', 'CD100', 'CD50',
                                          'CD0', 'CD30', 'CD26', 'CD21', 'CD16', 'CD11', 'CD5']),
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters
        params['param_list'] = pd.read_excel(self.workbook_path,
                                                           sheet_name='parameters')
        self.param_list.set_index("Parameter", inplace=True)

        params['prob_infant_fast_progressor'] = self.param_list.loc['prob_infant_fast_progressor'].values
        params['infant_progression_category'] = self.param_list.loc['infant_progression_category'].values
        params['exp_rate_mort_infant_fast_progressor'] = \
            self.param_list.loc['exp_rate_mort_infant_fast_progressor', 'Value1']
        params['weibull_scale_mort_infant_slow_progressor'] = \
            self.param_list.loc['weibull_scale_mort_infant_slow_progressor', 'Value1']
        params['weibull_shape_mort_infant_slow_progressor'] = \
            self.param_list.loc['weibull_shape_mort_infant_slow_progressor', 'Value1']
        params['weibull_shape_mort_adult'] = \
            self.param_list.loc['weibull_shape_mort_adult', 'Value1']
        params['proportion_high_sexual_risk_male'] = \
            self.param_list.loc['proportion_high_sexual_risk_male', 'Value1']
        params['proportion_high_sexual_risk_female'] = \
            self.param_list.loc['proportion_high_sexual_risk_female', 'Value1']
        params['proportion_female_sex_workers'] = \
            self.param_list.loc['proportion_female_sex_workers', 'Value1']
        params['rr_HIV_high_sexual_risk'] = \
            self.param_list.loc['rr_HIV_high_sexual_risk', 'Value1']
        params['rr_HIV_high_sexual_risk_fsw'] = \
            self.param_list.loc['rr_HIV_high_sexual_risk_fsw', 'Value1']
        params['proportion_on_ART_infectious'] = \
            self.param_list.loc['proportion_on_ART_infectious', 'Value1']
        params['beta'] = \
            self.param_list.loc['beta', 'Value1']
        params['irr_hiv_f'] = \
            self.param_list.loc['irr_hiv_f', 'Value1']
        params['prob_mtct'] = \
            self.param_list.loc['prob_mtct', 'Value1']
        params['rr_circumcision'] = \
            self.param_list.loc['rr_circumcision', 'Value1']
        params['rr_behaviour_change'] = \
            self.param_list.loc['rr_behaviour_change', 'Value1']
        params['rel_infectiousness_acute'] = \
            self.param_list.loc['rel_infectiousness_acute', 'Value1']
        params['rel_infectiousness_late'] = \
            self.param_list.loc['rel_infectiousness_late', 'Value1']

        # print(self.param_list.head())
        # print(params['infant_progression_category'])
        # print(params['prob_infant_fast_progressor'])
        # print(params['exp_rate_mort_infant_fast_progressor'])

        self.parameters['method_hiv_data'] = pd.read_excel(self.workbook_path,
                                                           sheet_name=None)

        params['hiv_prev'], params['hiv_death'], params['hiv_inc'], params['cd4_base'], params['time_cd4'], \
        params['initial_state_probs'], params['irr_age'] = self.method_hiv_data['prevalence'], \
                                                           self.method_hiv_data['deaths'], \
                                                           self.method_hiv_data['incidence'], \
                                                           self.method_hiv_data['CD4_distribution'], \
                                                           self.method_hiv_data['cd4_unrolled'], \
                                                           self.method_hiv_data['Initial_state_probs'], \
                                                           self.method_hiv_data['IRR']

    def initialise_population(self, population):
        """Set our property values for the initial population.
        """
        df = population.props

        df['has_hiv'] = False
        df['date_hiv_infection'] = pd.NaT
        df['date_aids_death'] = pd.NaT
        df['sexual_risk_group'].values[:] = 'low'

        self.high_risk(population)  # assign high sexual risk
        self.fsw(population)  # allocate proportion of women with very high sexual risk (fsw)
        self.baseline_prevalence(population)  # allocate baseline prevalence
        self.time_since_infection(population)  # find time since infection using CD4 distribution
        self.initial_pop_deaths_children(population)  # add death dates for children
        self.initial_pop_deaths_adults(population)  # add death dates for adults

    def log_scale(self, a0):
        """ helper function for adult mortality rates"""
        age_scale = 2.55 - 0.025 * (a0 - 30)
        return age_scale

    def high_risk(self, population):
        """ Stratify the adult (age >15) population in high or low sexual risk """

        df = population.props
        age = population.age

        male_sample = df[(df.sex == 'M') & (age.years >= 15)].sample(
            frac=self.parameters['proportion_high_sexual_risk_male']).index
        female_sample = df[(df.sex == 'F') & (age.years >= 15)].sample(
            frac=self.parameters['proportion_high_sexual_risk_female']).index

        # these individuals have higher risk of hiv
        df.loc[male_sample | female_sample, 'sexual_risk_group'] = 'high'

    def fsw(self, population):
        """ Assign female sex work to sample of women and change sexual risk to high value
        """

        df = population.props
        age = population.age

        fsw = df[(df.sex == 'F') & (age.years >= 15)].sample(
            frac=self.parameters['proportion_female_sex_workers']).index

        # these individuals have higher risk of hiv
        df.loc[fsw, 'sexual_risk_group'] = 'sex_work'

    def get_index(self, population, has_hiv, sex, age_low, age_high, cd4_state):
        df = population.props
        age = population.age

        index = df.index[
            df.has_hiv &
            (df.sex == sex) &
            (age.years >= age_low) & (age.years < age_high) &
            (df.cd4_state == cd4_state)]

        return index

    def baseline_prevalence(self, population):
        """
        assign baseline hiv prevalence
        """

        now = self.sim.date
        df = population.props

        prevalence = self.hiv_prev.loc[self.hiv_prev.year == now.year, ['age_from', 'sex', 'prev_prop']]

        # add age to population.props
        df_with_age = pd.merge(df, population.age, left_index=True, right_index=True, how='left')

        # merge all susceptible individuals with their hiv probability based on sex and age
        df_with_age_hivprob = df_with_age.merge(prevalence,

                                                left_on=['years', 'sex'],

                                                right_on=['age_from', 'sex'],

                                                how='left')

        # no prevalence in ages 80+ so fill missing values with 0
        df_with_age_hivprob['prev_prop'] = df_with_age_hivprob['prev_prop'].fillna(0)

        # print(df_with_age.head(10))
        # print(df_with_age_hivprob.head(20))
        # df_with_age_hivprob.to_csv('Q:/Thanzi la Onse/HIV/test.csv', sep=',')  # output a test csv file
        # print(list(df_with_age_hivprob.head(0)))  # prints list of column names in merged df

        # assert df_with_age_hivprob.prev_prop.isna().sum() == 0  # check there is a probability for every individual

        # get a list of random numbers between 0 and 1 for each infected individual
        random_draw = self.rng.random_sample(size=len(df_with_age_hivprob))

        # probability of HIV > random number, assign has_hiv = True
        hiv_index = df_with_age_hivprob.index[(df_with_age_hivprob.prev_prop > random_draw)]

        # print(hiv_index)
        # test = hiv_index.isnull().sum()  # sum number of nan
        # print("number of nan: ", test)

        df.loc[hiv_index, 'has_hiv'] = True
        df.loc[hiv_index, 'date_hiv_infection'] = now

    def time_since_infection(self, population):
        """
        calculate the time since infection based on the CD4_state
        this only applies for those aged >=15 years
        assume everyone <15 was infected at birth
        """

        df = population.props
        now = self.sim.date

        # add age to population.props
        df_age = pd.merge(df, population.age, left_index=True, right_index=True, how='left')

        # print(df_age.head(20))

        # for those aged 0-4 years
        cd4_states = ['CD30', 'CD26', 'CD21', 'CD16', 'CD11', 'CD5', 'CD0']
        for sex in ['M', 'F']:
            idx = df_age.index[df_age.has_hiv & (df_age.sex == sex) & (df_age.years < 5)]
            cd4_probs = self.cd4_base.loc[
                (self.cd4_base.sex == sex) & (self.cd4_base.year == now.year) & (self.cd4_base.age == '0_4'),
                'probability']
            df_age.loc[idx, 'cd4_state'] = np.random.choice(cd4_states,
                                                            size=len(idx),
                                                            replace=True,
                                                            p=cd4_probs.values)

        # for those aged 5-14 years
        cd4_states = ['CD1000', 'CD750', 'CD500', 'CD350', 'CD200', 'CD0']

        for sex in ['M', 'F']:
            idx = df_age.index[df_age.has_hiv & (df_age.sex == sex) & (df_age.years >= 5) & (df_age.years <= 14)]
            cd4_probs = self.cd4_base.loc[
                (self.cd4_base.sex == sex) & (self.cd4_base.year == now.year) & (self.cd4_base.age == '5_14'),
                'probability']
            df_age.loc[idx, 'cd4_state'] = np.random.choice(cd4_states,
                                                            size=len(idx),
                                                            replace=True,
                                                            p=cd4_probs.values)

        # for those aged >= 15 years
        cd4_states = ['CD500', 'CD350', 'CD250', 'CD200', 'CD100', 'CD50', 'CD0']

        for sex in ['M', 'F']:
            idx = df_age.index[df_age.has_hiv & (df_age.sex == sex) & (df_age.years >= 15)]
            cd4_probs = self.cd4_base.loc[
                (self.cd4_base.sex == sex) & (self.cd4_base.year == now.year) & (
                    self.cd4_base.age == "15_80"), 'probability']
            df_age.loc[idx, 'cd4_state'] = np.random.choice(cd4_states,
                                                            size=len(idx),
                                                            replace=True,
                                                            p=cd4_probs.values)

        df.cd4_state = df_age.cd4_state  # output this as needed for baseline art mortality rates
        # print(df.cd4_state.head(40))

        # print(df.head(20))

        # hold the index of all individuals with hiv
        infected = df_age.index[df_age.has_hiv & (df_age.years >= 15)]

        # select all individuals with hiv and over 15 years of age
        infected_age = df_age.loc[infected, ['cd4_state', 'sex', 'years']]

        # print(infected_age.head(10))

        # merge all infected individuals with their cd4 infected row based on CD4 state, sex and age
        infected_age_cd4 = infected_age.merge(self.time_cd4,
                                              left_on=['cd4_state', 'sex', 'years'],
                                              right_on=['cd4_state', 'sex', 'age'],
                                              how='left')

        assert len(infected_age_cd4) == len(infected)  # check merged row count

        # print(time_cd4)
        # print(infected_age_cd4.head(20))
        # infected_age_cd4.to_csv('Q:/Thanzi la Onse/HIV/test2.csv', sep=',')  # output a test csv file

        # assert np.array_equal(infected_age.years_exact,
        #                      infected_age_cd4.years_exact)  # check rows are in the same order

        # assert infctd_age_cd4.prob1.isna().sum() == 0  # check that we found a probability for every individual

        # note prob2 is 1-prob1
        # get a list of random numbers between 0 and 1 for each infected individual
        random_draw = self.rng.random_sample(size=len(infected))

        # get a list of time infected which is 'years1' if the random number is less than 'prob1', otherwise 'years2'
        time_infected = infected_age_cd4.years1.where(infected_age_cd4.prob1 < random_draw,
                                                      infected_age_cd4.years2)

        # convert those years to a date in the past
        time_infected = pd.to_timedelta(time_infected, unit='Y')
        date_infected = now - time_infected

        # assign the calculated dates back to the original population dataframe
        # NOTE: we use the '.values' to assign back, ignoring the index of the 'date_infected' series
        df.loc[infected, 'date_hiv_infection'] = date_infected.values

        # del df['cd4_state']  # this doesn't delete the column, just the values in it

        # assign time of infection for children <15 years
        inf_child = df_age.index[df_age.has_hiv & (df_age.years < 15)]
        df.loc[inf_child, 'date_hiv_infection'] = df.date_of_birth.values[inf_child]

        # print(inf_child)

        # check time infected is less than time alive
        early_doi = ((now - df.date_hiv_infection).dt.days / 365.25) > \
                    ((now - df.date_of_birth).dt.days / 365.25)  # time infected earlier than time alive!

        # early_doi.to_csv('Q:/Thanzi la Onse/HIV/test3.csv', sep=',')

        if early_doi.any():
            tmp2 = df.loc[early_doi, 'date_of_birth']
            df.loc[early_doi, 'date_HIV_infection'] = tmp2  # replace with year of birth

    def initial_pop_deaths_children(self, population):
        """ assign death dates to baseline hiv-infected population - INFANTS
        """
        df = population.props
        now = self.sim.date
        age = population.age
        params = self.parameters

        # PAEDIATRIC time of death - untreated
        hiv_inf = df.index[df.has_hiv & (age.years < 3)]
        # print('hiv_inf: ', hiv_inf)

        # need a two parameter Weibull with size parameter, multiply by scale instead
        time_death_slow = self.rng.weibull(a=params['weibull_shape_mort_infant_slow_progressor'],
                                           size=len(hiv_inf)) * params['weibull_scale_mort_infant_slow_progressor']
        # print('time_death_slow: ', time_death_slow)

        time_death_slow = pd.Series(time_death_slow, index=hiv_inf)
        # print('testing_index: ', test)

        time_infected = now - df.loc[hiv_inf, 'date_hiv_infection']
        # print(time_infected)
        # print(time_death_slow)

        # while time of death is shorter than time infected - redraw
        while np.any(time_infected >
                     (pd.to_timedelta(time_death_slow * 365.25, unit='d'))):
            redraw = time_infected.index[time_infected >
                                         (pd.to_timedelta(time_death_slow * 365.25, unit='d'))]
            # print('redraw: ', redraw)

            new_time_death_slow = self.rng.weibull(a=params['weibull_shape_mort_infant_slow_progressor'],
                                                   size=len(redraw)) * params[
                                      'weibull_scale_mort_infant_slow_progressor']
            # print('new_time_death: ', new_time_death_slow)

            time_death_slow[redraw] = new_time_death_slow

        time_death_slow = pd.to_timedelta(time_death_slow * 365.25, unit='d')
        # print(time_death_slow)

        # remove microseconds
        # time_death_slow = pd.to_timedelta(time_death_slow).values.astype('timedelta64[s]')
        # time_death_slow = time_death_slow.floor('s')
        time_death_slow = pd.Series(time_death_slow).dt.floor("S")

        # print(time_death_slow)

        df.loc[hiv_inf, 'date_aids_death'] = df.loc[hiv_inf, 'date_hiv_infection'] + time_death_slow

        # test2 = df.loc[hiv_inf]
        # test2.to_csv('Q:/Thanzi la Onse/HIV/test4.csv', sep=',')  # check data for infants

    def initial_pop_deaths_adults(self, population):
        """ assign death dates to baseline hiv-infected population - ADULTS
        """
        df = population.props
        now = self.sim.date
        age = population.age
        params = self.parameters

        # add age to population.props
        df_age = pd.merge(df, population.age, left_index=True, right_index=True, how='left')

        # ADULT time of death, adults are all those aged >3
        hiv_ad = df.index[df.has_hiv & (age.years >= 3)]
        # print('hiv_ad: ', hiv_ad)

        time_of_death = self.rng.weibull(a=params['weibull_shape_mort_adult'], size=len(hiv_ad)) * \
                        np.exp(self.log_scale(df_age.loc[hiv_ad, 'years']))

        # print('length time_of_death:', len(time_of_death))
        # print('time_death: ', time_of_death)

        time_infected = now - df.loc[hiv_ad, 'date_hiv_infection']
        # print(time_infected)

        # while time of death is shorter than time infected - redraw
        while np.any(time_infected >
                     (pd.to_timedelta(time_of_death * 365.25, unit='d'))):
            redraw = time_infected.index[time_infected >
                                         (pd.to_timedelta(time_of_death * 365.25, unit='d'))]

            # print("redraw: ", redraw)

            new_time_of_death = self.rng.weibull(a=params['weibull_shape_mort_adult'], size=len(redraw)) * \
                                np.exp(self.log_scale(df_age.loc[redraw, 'years']))
            # print('new_time_of_death:', new_time_of_death)

            time_of_death[redraw] = new_time_of_death

        time_of_death = pd.to_timedelta(time_of_death * 365.25, unit='d')
        # print(time_of_death)

        # remove microseconds
        # time_of_death = pd.to_timedelta(time_of_death).values.astype('timedelta64[s]')
        # time_of_death = time_of_death.floor('s')
        time_of_death = pd.Series(time_of_death).dt.floor("S")

        # print(time_death_slow)

        df.loc[hiv_ad, 'date_aids_death'] = df.loc[hiv_ad, 'date_hiv_infection'] + time_of_death

        # test2 = df.loc[hiv_ad]
        # test2.to_csv('Q:/Thanzi la Onse/HIV/test4.csv', sep=',')  # check data for adults

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        sim.schedule_event(hiv_event(self), sim.date + DateOffset(months=12))

        # add an event to log to screen
        sim.schedule_event(hivLoggingEvent(self), sim.date + DateOffset(months=12))

    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """
        params = self.parameters

        child.has_hiv = False
        child.date_hiv_infection = pd.NaT
        child.date_aids_death = pd.NaT
        child.sexual_risk_group.values[:] = 'low'

        # TODO: include risk during breastfeeding period

        random_draw = self.sim.rng.random_sample(size=1)

        if (random_draw < params['prob_mtct']) & mother.has_hiv:
            child.has_hiv = True
            child.date_hiv_infection = self.sim.date


class hiv_event(RegularEvent, PopulationScopeEventMixin):
    """A skeleton class for an event
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
        super().__init__(module, frequency=DateOffset(months=12))  # every 12 months

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        params = self.module.parameters
        now = self.sim.date

        # add age to population.props
        df_age = pd.merge(df, population.age, left_index=True, right_index=True, how='left')

        #  calculate relative infectivity for everyone hiv+
        # stage of infection

        # select all hiv+ and alive
        time_on_treatment = (now - df_age.date_art_start).dt.days / 30.44  # in months
        # print('time_on_treatment:', time_on_treatment)

        time_before_death = (df_age.date_aids_death - now).dt.days / 30.44
        # print('time_before_death:', time_before_death)

        # time since infection
        time_inf = (now - df_age.date_hiv_infection).dt.days / 30.44

        # sum how many untreated in each stage of infection
        acute = len(
            df_age.index[df_age.has_hiv & (df_age.years >= 15) & df_age.is_alive & ~df_age.on_art & (time_inf < 3)])
        # print('acute:', acute)

        chronic = len(
            df_age.index[df_age.has_hiv & (df_age.years >= 15) & df_age.is_alive & ~df_age.on_art & (time_inf >= 3)
                         & (time_before_death >= 19)])
        # print('chronic:', chronic)

        late = len(df_age.index[df_age.has_hiv & (df_age.years >= 15) & df_age.is_alive & ~df_age.on_art & (
                time_before_death >= 10)
                                & (time_before_death < 19)])
        # print('late:', late)

        end = len(df_age.index[df_age.has_hiv & (df_age.years >= 15) & df_age.is_alive & ~df_age.on_art & (
                time_before_death < 10)])
        # print('end:', end)  # no transmission during end stage

        # if early treated, use same relative infectivity as untreated
        acute_early_treated = len(
            df_age.index[df_age.has_hiv & (df_age.years >= 15) & df_age.is_alive & df_age.on_art & time_inf < 3 & (
                    time_on_treatment < 3)])
        # print('acute_early_treated:', acute_early_treated)

        chronic_early_treated = len(
            df_age.index[df_age.has_hiv & (df_age.years >= 15) & df_age.is_alive & df_age.on_art & (time_inf >= 3)
                         & (time_before_death >= 19) & (time_on_treatment < 3)])
        # print('chronic_early_treated:', chronic_early_treated)

        late_early_treated = len(
            df_age.index[
                df_age.has_hiv & (df_age.years >= 15) & df_age.is_alive & df_age.on_art & (time_before_death >= 10)
                & (time_before_death < 19) & (time_on_treatment < 3)])
        # print('late_early_treated:', late_early_treated)

        end_early_treated = len(df_age.index[
                                    df_age.has_hiv & (df_age.years >= 15) & df_age.is_alive & df_age.on_art & (
                                            time_before_death < 10) &
                                    (time_on_treatment < 3)])
        # print('end_early_treated:', end_early_treated)  # no transmission during end stage

        # after 3 months of treatment
        treated = len(
            df_age.index[
                df_age.has_hiv & (df_age.years >= 15) & df_age.is_alive & df_age.on_art & (time_on_treatment >= 3)])
        # print('treated:', treated)


        # calculate force of infection
        infectious_term = ((acute + acute_early_treated) * params['rel_infectiousness_acute']) + \
                          chronic + chronic_early_treated +\
                          ((late + late_early_treated) * params['rel_infectiousness_late']) + \
                          (treated * params['proportion_on_ART_infectious'])

        total_pop = len(df_age[(df_age.years >= 15) & df_age.is_alive])
        foi = params['beta'] * infectious_term / total_pop
        # print('foi:', foi)

        # TODO: reduce force of infection if behaviour change available
        # TODO: reduce FOI due to condom use

        df_with_irr = df_age.merge(params['irr_age'], left_on=['years', 'sex'], right_on=['ages', 'sex'], how='left')
        # print('df: ', df_with_irr.head(30))

        #  incidence rate ratios and relative susceptibility
        eff_susc = pd.Series(0, index=df_with_irr.index)
        eff_susc.loc[
            ~df_with_irr.has_hiv & (
                df_with_irr.years >= 15) & df_with_irr.is_alive] = foi  # foi applied to all HIV- adults
        eff_susc.loc[~df_with_irr.has_hiv & (df_with_irr.years >= 15) & df_with_irr.is_alive] *= df_with_irr.loc[
            ~df_with_irr.has_hiv & (df_with_irr.years >= 15) & df_with_irr.is_alive, 'comb_irr']  # scaled by age IRR
        eff_susc.loc[df_with_irr.sexual_risk_group == 'high'] *= params[
            'rr_HIV_high_sexual_risk']  # scaled by sexual risk group
        eff_susc.loc[df_with_irr.sexual_risk_group == 'sex_work'] *= params[
            'rr_HIV_high_sexual_risk_fsw']  # scaled for fsw
        eff_susc.loc[df_with_irr.is_circumcised] *= params['rr_circumcision']  # scaled for circumcision
        eff_susc.loc[df_with_irr.behaviour_change] *= params['rr_behaviour_change']  # scaled for behaviour counselling

        #  sample using the FOI scaled by relative susceptibility
        newly_infected_index = df_with_irr.index[
            (self.sim.rng.random_sample(size=len(df_with_irr)) < eff_susc) & ~df_with_irr.has_hiv & (
                df_with_irr.years >= 15) & df_with_irr.is_alive]

        # time of death
        death_date = self.sim.rng.weibull(a=params['weibull_shape_mort_adult'], size=len(newly_infected_index)) * \
                     np.exp(self.module.log_scale(df_age.loc[newly_infected_index, 'years']))
        death_date = pd.to_timedelta(death_date * 365.25, unit='d')
        # print('death dates as dates: ', death_date)

        # death_date = death_date.floor('s')  # remove microseconds
        # death_date = pd.to_timedelta(death_date).values.astype('timedelta64[s]')
        death_date = pd.Series(death_date).dt.floor("S")
        # print('death dates without ns: ', death_date)

        df.loc[newly_infected_index, 'has_hiv'] = True
        df.loc[newly_infected_index, 'date_hiv_infection'] = now
        df.loc[newly_infected_index, 'date_aids_death'] = now + death_date

        death_dates = df.date_aids_death[newly_infected_index]

        # schedule the death event
        for i in newly_infected_index:
            person = population[i]
            death = DeathEventHIV(self.module, person, cause='hiv')  # make that death event
            time_death = death_dates[i]
            # print('time_death: ', time_death)
            # print('now: ', now)
            self.sim.schedule_event(death, time_death)  # schedule the death


class DeathEventHIV(Event, IndividualScopeEventMixin):
    """
    Performs the Death operation on an individual and logs it.
    """

    def __init__(self, module, individual, cause):
        super().__init__(module, person=individual)
        self.cause = cause

    def apply(self, individual):
        if individual.is_alive & ~individual.on_art:
            individual.is_alive = False

        # Log the death
        self.module.store_DeathsLog['DeathEvent_Time'].append(self.sim.date)
        self.module.store_DeathsLog['DeathEvent_Age'].append(self.sim.population.age.years[individual.index])
        self.module.store_DeathsLog['DeathEvent_Cause'].append(self.cause)


class hivLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """ produce some outputs to check
        """
        # run this event every 12 months
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props
        now = self.sim.date

        infected_total = len(df[df.has_hiv & df.is_alive])

        date_aids_death = df.loc[df.has_hiv & df.is_alive, 'date_aids_death']
        # print('date_aids_death: ', date_aids_death)
        year_aids_death = date_aids_death.dt.year
        # print('year_aids_death: ', year_aids_death)

        #  this shows the deaths scheduled for this year, including those postponed due to ART
        die = sum(1 for x in year_aids_death if int(x) == now.year)
        # print('die: ', die)

        self.module.store['Time'].append(self.sim.date)
        self.module.store['Total_HIV'].append(infected_total)
        self.module.store['HIV_scheduled_deaths'].append(die)

        # print('hiv outputs: ', self.sim.date, infected_total)
