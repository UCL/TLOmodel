"""
HIV infection event
"""
import os

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent, IndividualScopeEventMixin
from tlo.methods import demography


class hiv(Module):
    """
    baseline hiv infection
    """

    def __init__(self, name=None, resourcefilepath=None, par_est=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.beta_calib = par_est
        self.store = {'Time': [], 'Total_HIV': [], 'HIV_scheduled_deaths': [], 'HIV_new_infections_adult': [],
                      'HIV_new_infections_child': [], 'hiv_prev_adult': [], 'hiv_prev_child': []}
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

        'beta':
            Parameter(Types.REAL, 'transmission rate'),
        'irr_hiv_f':
            Parameter(Types.REAL, 'incidence rate ratio for females vs males'),
        'rr_circumcision':
            Parameter(Types.REAL, 'relative reduction in susceptibility due to circumcision'),
        'rr_behaviour_change':
            Parameter(Types.REAL, 'relative reduction in susceptibility due to behaviour modification'),
        'rel_infectiousness_acute':
            Parameter(Types.REAL, 'relative infectiousness during acute stage'),
        'prob_mtct_untreated':
            Parameter(Types.REAL, 'probability of mother to child transmission'),
        'prob_mtct_treated':
            Parameter(Types.REAL, 'probability of mother to child transmission, mother on ART'),
        'prob_mtct_incident_preg':
            Parameter(Types.REAL, 'probability of mother to child transmission, mother infected during pregnancy'),
        'prob_mtct_incident_post':
            Parameter(Types.REAL, 'probability of mother to child transmission, mother infected during pregnancy'),
        'prob_mtct_breastfeeding_untreated':
            Parameter(Types.REAL, 'probability of mother to child transmission during breastfeeding'),
        'prob_mtct_breastfeeding_treated':
            Parameter(Types.REAL, 'probability of mother to child transmission, mother infected during breastfeeding'),
        'fsw_transition':
            Parameter(Types.REAL, 'probability of returning from sex work to low sexual risk'),
        'or_rural':
            Parameter(Types.REAL, 'odds ratio rural location'),
        'or_windex_poorer':
            Parameter(Types.REAL, 'odds ratio wealth level poorer'),
        'or_windex_middle':
            Parameter(Types.REAL, 'odds ratio wealth level middle'),
        'or_windex_richer':
            Parameter(Types.REAL, 'odds ratio wealth level richer'),
        'or_windex_richest':
            Parameter(Types.REAL, 'odds ratio wealth level richest'),
        'or_sex_f':
            Parameter(Types.REAL, 'odds ratio sex=female'),
        'or_age_gp20':
            Parameter(Types.REAL, 'odds ratio age 20-24'),
        'or_age_gp25':
            Parameter(Types.REAL, 'odds ratio age 25-29'),
        'or_age_gp30':
            Parameter(Types.REAL, 'odds ratio age 30-34'),
        'or_age_gp35':
            Parameter(Types.REAL, 'odds ratio age 35-39'),
        'or_age_gp40':
            Parameter(Types.REAL, 'odds ratio age 40-44'),
        'or_age_gp45':
            Parameter(Types.REAL, 'odds ratio age 45-49'),
        'or_age_gp50':
            Parameter(Types.REAL, 'odds ratio age 50+'),
        'or_edlevel_primary':
            Parameter(Types.REAL, 'odds ratio education primary'),
        'or_edlevel_secondary':
            Parameter(Types.REAL, 'odds ratio education secondary'),
        'or_edlevel_higher':
            Parameter(Types.REAL, 'odds ratio education higher'),
        'hiv_prev_2010':
            Parameter(Types.REAL, 'prevalence hiv in adults'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'hiv_inf': Property(Types.BOOL, 'HIV status'),
        'hiv_date_inf': Property(Types.DATE, 'Date acquired HIV infection'),
        'hiv_date_death': Property(Types.DATE, 'Projected time of AIDS death if untreated'),
        'hiv_sexual_risk': Property(Types.CATEGORICAL, 'Sexual risk groups',
                                    categories=['low', 'sex_work']),
        'hiv_mother_inf': Property(Types.BOOL, 'HIV status of mother'),
        'hiv_mother_art': Property(Types.BOOL, 'ART status of mother')
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """

        workbook = pd.read_excel(os.path.join(self.resourcefilepath,
                                              'Method_HIV.xlsx'), sheet_name=None)

        params = self.parameters
        params['param_list'] = workbook['parameters']

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

        params['proportion_female_sex_workers'] = \
            self.param_list.loc['proportion_female_sex_workers', 'Value1']

        params['rr_HIV_high_sexual_risk_fsw'] = \
            self.param_list.loc['rr_HIV_high_sexual_risk_fsw', 'Value1']

        # params['beta'] = \
        #     self.param_list.loc['beta', 'Value1']
        params['irr_hiv_f'] = \
            self.param_list.loc['irr_hiv_f', 'Value1']
        params['rr_circumcision'] = \
            self.param_list.loc['rr_circumcision', 'Value1']
        params['rr_behaviour_change'] = \
            self.param_list.loc['rr_behaviour_change', 'Value1']

        params['prob_mtct_untreated'] = \
            self.param_list.loc['prob_mtct_untreated', 'Value1']
        params['prob_mtct_treated'] = \
            self.param_list.loc['prob_mtct_treated', 'Value1']
        params['prob_mtct_incident_preg'] = \
            self.param_list.loc['prob_mtct_incident_preg', 'Value1']
        params['prob_mtct_incident_post'] = \
            self.param_list.loc['prob_mtct_incident_post', 'Value1']
        params['prob_mtct_breastfeeding_untreated'] = \
            self.param_list.loc['prob_mtct_breastfeeding_untreated', 'Value1']
        params['prob_mtct_breastfeeding_treated'] = \
            self.param_list.loc['prob_mtct_breastfeeding_treated', 'Value1']
        params['fsw_transition'] = \
            self.param_list.loc['fsw_transition', 'Value1']
        params['hiv_prev_2010'] = \
            self.param_list.loc['hiv_prev_2010', 'Value1']

        params['or_rural'] = \
            self.param_list.loc['or_rural', 'Value1']
        params['or_windex_poorer'] = \
            self.param_list.loc['or_windex_poorer', 'Value1']
        params['or_windex_middle'] = \
            self.param_list.loc['or_windex_middle', 'Value1']
        params['or_windex_richer'] = \
            self.param_list.loc['or_windex_richer', 'Value1']
        params['or_windex_richest'] = \
            self.param_list.loc['or_windex_richest', 'Value1']
        params['or_sex_f'] = \
            self.param_list.loc['or_sex_f', 'Value1']
        params['or_age_gp20'] = \
            self.param_list.loc['or_age_gp20', 'Value1']
        params['or_age_gp25'] = \
            self.param_list.loc['or_age_gp25', 'Value1']
        params['or_age_gp30'] = \
            self.param_list.loc['or_age_gp30', 'Value1']
        params['or_age_gp35'] = \
            self.param_list.loc['or_age_gp35', 'Value1']
        params['or_age_gp40'] = \
            self.param_list.loc['or_age_gp40', 'Value1']
        params['or_age_gp45'] = \
            self.param_list.loc['or_age_gp45', 'Value1']
        params['or_age_gp50'] = \
            self.param_list.loc['or_age_gp50', 'Value1']
        params['or_edlevel_primary'] = \
            self.param_list.loc['or_edlevel_primary', 'Value1']
        params['or_edlevel_secondary'] = \
            self.param_list.loc['or_edlevel_secondary', 'Value1']
        params['or_edlevel_higher'] = \
            self.param_list.loc['or_edlevel_higher', 'Value1']

        # TODO: put beta in worksheet for default

        if self.beta_calib:
            params['beta'] = float(self.beta_calib)
        # print('beta', params['beta'])

        # print(self.param_list.head())
        #
        # self.parameters['method_hiv_data'] = pd.read_excel(self.workbook_path,
        #                                                    sheet_name=None)

        params['hiv_prev'] = workbook['prevalence']

    def initialise_population(self, population):
        """Set our property values for the initial population.
        """
        df = population.props

        df['hiv_inf'] = False
        df['hiv_date_inf'] = pd.NaT
        df['hiv_date_death'] = pd.NaT
        df['hiv_sexual_risk'].values[:] = 'low'
        df['hiv_mother_inf'] = False
        df['hiv_mother_art'].values[:] = '0'

        self.fsw(population)  # allocate proportion of women with very high sexual risk (fsw)
        self.baseline_prevalence(population)  # allocate baseline prevalence
        self.initial_pop_deaths_children(population)  # add death dates for children
        self.initial_pop_deaths_adults(population)  # add death dates for adults

    def log_scale(self, a0):
        """ helper function for adult mortality rates"""
        age_scale = 2.55 - 0.025 * (a0 - 30)
        return age_scale

    def fsw(self, population):
        """ Assign female sex work to sample of women and change sexual risk
        """
        df = population.props

        fsw = df[df.is_alive & (df.sex == 'F') & (df.age_years.between(15, 49))].sample(
            frac=self.parameters['proportion_female_sex_workers']).index

        df.loc[fsw, 'hiv_sexual_risk'] = 'sex_work'

    def baseline_prevalence(self, population):
        """
        assign baseline hiv prevalence
        """

        # TODO: assign baseline prevalence by district
        # odds ratios from Wingston's analysis dropbox Data-MDHS
        now = self.sim.date
        df = population.props
        params = self.parameters

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ ADULT HIV ~~~~~~~~~~~~~~~~~~~~~~~~~~

        prevalence = params['hiv_prev_2010']

        # only for 15-54
        prob_hiv = pd.Series(0, index=df.index)
        prob_hiv.loc[df.is_alive & df.age_years.between(15, 55)] = prevalence  # applied to all adults
        prob_hiv.loc[(df.sex == 'F')] *= params['or_sex_f']
        prob_hiv.loc[df.age_years.between(20, 24)] *= params['or_age_gp20']
        prob_hiv.loc[df.age_years.between(25, 29)] *= params['or_age_gp25']
        prob_hiv.loc[df.age_years.between(30, 34)] *= params['or_age_gp30']
        prob_hiv.loc[df.age_years.between(35, 39)] *= params['or_age_gp35']
        prob_hiv.loc[df.age_years.between(40, 44)] *= params['or_age_gp40']
        prob_hiv.loc[df.age_years.between(45, 50)] *= params['or_age_gp45']
        prob_hiv.loc[(df.age_years >= 50)] *= params['or_age_gp50']
        prob_hiv.loc[~df.li_urban] *= params['or_rural']
        prob_hiv.loc[(df.li_wealth == '2')] *= params['or_windex_poorer']
        prob_hiv.loc[(df.li_wealth == '3')] *= params['or_windex_middle']
        prob_hiv.loc[(df.li_wealth == '4')] *= params['or_windex_richer']
        prob_hiv.loc[(df.li_wealth == '5')] *= params['or_windex_richest']
        prob_hiv.loc[(df.li_ed_lev == '2')] *= params['or_edlevel_primary']
        prob_hiv.loc[(df.li_ed_lev == '3')] *= params['or_edlevel_secondary']  # li_ed_lev=3 secondary and higher

        #  sample scaled by relative risk
        infected_idx = df.index[(self.rng.random_sample(size=len(df)) < prob_hiv)]

        # print('infected_idx', infected_idx)
        test = infected_idx.isnull().sum()  # sum number of nan
        # print("number of nan: ", test)

        df.loc[infected_idx, 'hiv_inf'] = True

        # for time since infection use a reverse weibull distribution
        # this is scaled by current age - should be scaled by age at infection!!
        # hold the index of all adults with hiv
        inf_adult = df.index[df.is_alive & df.hiv_inf & (df.age_years >= 15)]
        times = self.rng.weibull(a=params['weibull_shape_mort_adult'], size=len(inf_adult)) * \
                np.exp(self.log_scale(df.loc[inf_adult, 'age_years']))

        time_inf = pd.to_timedelta(times * 365.25, unit='d')
        df.loc[inf_adult, 'hiv_date_inf'] = now - time_inf

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ CHILD HIV ~~~~~~~~~~~~~~~~~~~~~~~~~~

        # baseline children's prevalence from spectrum outputs
        prevalence = self.hiv_prev.loc[self.hiv_prev.year == now.year, ['age_from', 'sex', 'prev_prop']]

        # merge all susceptible individuals with their hiv probability based on sex and age
        df_hivprob = df.merge(prevalence, left_on=['age_years', 'sex'],

                              right_on=['age_from', 'sex'],

                              how='left')

        # fill missing values with 0 (only relevant for age 80+)
        df_hivprob['prev_prop'] = df_hivprob['prev_prop'].fillna(0)

        assert df_hivprob.prev_prop.isna().sum() == 0  # check there is a probability for every individual

        # get a list of random numbers between 0 and 1 for each infected individual
        random_draw = self.rng.random_sample(size=len(df_hivprob))

        # probability of hiv > random number, assign hiv_inf = True
        # TODO: cluster this by mother's hiv status??
        # if mother hiv+ at time of birth...
        hiv_index = df_hivprob.index[df.is_alive & (df_hivprob.prev_prop > random_draw) & df.age_years.between(0, 14)]
        # print(hiv_index)

        df.loc[hiv_index, 'hiv_inf'] = True
        df.loc[hiv_index, 'hiv_date_inf'] = df.loc[hiv_index, 'date_of_birth']

    def initial_pop_deaths_children(self, population):
        """ assign death dates to baseline hiv-infected population - INFANTS
        """
        df = population.props
        now = self.sim.date
        params = self.parameters

        # PAEDIATRIC time of death - untreated
        infants = df.index[df.is_alive & df.hiv_inf & (df.age_years < 3)]

        # need a two parameter Weibull with size parameter, multiply by scale instead
        time_death_slow = self.rng.weibull(a=params['weibull_shape_mort_infant_slow_progressor'],
                                           size=len(infants)) * params['weibull_scale_mort_infant_slow_progressor']

        time_death_slow = pd.Series(time_death_slow, index=infants)
        time_infected = now - df.loc[infants, 'hiv_date_inf']

        # while time of death is shorter than time infected - redraw
        while np.any(time_infected >
                     (pd.to_timedelta(time_death_slow * 365.25, unit='d'))):
            redraw = time_infected.index[time_infected >
                                         (pd.to_timedelta(time_death_slow * 365.25, unit='d'))]

            new_time_death_slow = self.rng.weibull(a=params['weibull_shape_mort_infant_slow_progressor'],
                                                   size=len(redraw)) * params[
                                      'weibull_scale_mort_infant_slow_progressor']

            time_death_slow[redraw] = new_time_death_slow

        time_death_slow = pd.to_timedelta(time_death_slow * 365.25, unit='d')

        # remove microseconds
        time_death_slow = pd.Series(time_death_slow).dt.floor("S")
        df.loc[infants, 'hiv_date_death'] = df.loc[infants, 'hiv_date_inf'] + time_death_slow
        death_dates = df.loc[infants, 'hiv_date_death']

        # schedule the death event
        for person in infants:
            death = DeathEventHIV(self, individual_id=person, cause='hiv')  # make that death event
            time_death = death_dates[person]
            self.sim.schedule_event(death, time_death)  # schedule the death

    def initial_pop_deaths_adults(self, population):
        """ assign death dates to baseline hiv-infected population - ADULTS
        """
        df = population.props
        now = self.sim.date
        params = self.parameters

        # adults are all those aged >=15
        # TODO: children hiv+ aged >3 have survival draws from adult weibull distr otherwise death date pre-2010
        hiv_ad = df.index[df.is_alive & df.hiv_inf & (df.age_years >= 3)]

        time_of_death = self.rng.weibull(a=params['weibull_shape_mort_adult'], size=len(hiv_ad)) * \
                        np.exp(self.log_scale(df.loc[hiv_ad, 'age_years']))

        time_infected = now - df.loc[hiv_ad, 'hiv_date_inf']

        # while time of death is shorter than time infected - redraw
        while np.any(time_infected >
                     (pd.to_timedelta(time_of_death * 365.25, unit='d'))):
            redraw = time_infected.index[time_infected >
                                         (pd.to_timedelta(time_of_death * 365.25, unit='d'))]

            new_time_of_death = self.rng.weibull(a=params['weibull_shape_mort_adult'], size=len(redraw)) * \
                                np.exp(self.log_scale(df.loc[redraw, 'age_years']))

            time_of_death[redraw] = new_time_of_death

        time_of_death = pd.to_timedelta(time_of_death * 365.25, unit='d')

        # remove microseconds
        time_of_death = pd.Series(time_of_death).dt.floor("S")

        df.loc[hiv_ad, 'hiv_date_death'] = df.loc[hiv_ad, 'hiv_date_inf'] + time_of_death

        death_dates = df.loc[hiv_ad, 'hiv_date_death']

        # schedule the death event
        for person in hiv_ad:
            death = DeathEventHIV(self, individual_id=person, cause='hiv')  # make that death event
            time_death = death_dates[person]
            self.sim.schedule_event(death, time_death)  # schedule the death

    def initialise_simulation(self, sim):
        """Get ready for simulation start.
        """
        sim.schedule_event(hiv_event(self), sim.date + DateOffset(months=12))

        # add an event to log to screen
        sim.schedule_event(hivLoggingEvent(self), sim.date + DateOffset(months=12))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual
        """
        params = self.parameters
        df = self.sim.population.props
        now = self.sim.date

        df.at[child_id, 'hiv_inf'] = False
        df.at[child_id, 'hiv_date_inf'] = pd.NaT
        df.at[child_id, 'hiv_date_death'] = pd.NaT
        df.at[child_id, 'hiv_sexual_risk'] = 'low'

        if df.at[mother_id, 'hiv_inf']:
            df.at[child_id, 'hiv_mother_inf'] = True
            if df.at[mother_id, 'hiv_on_art'] == '2':
                df.at[child_id, 'hiv_mother_art'] = True

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ MTCT ~~~~~~~~~~~~~~~~~~~~~~~~~~

        #  TRANSMISSION DURING PREGNANCY / DELIVERY
        random_draw = self.sim.rng.random_sample(size=1)

        #  mother has incident infection during pregnancy, NO ART
        if (random_draw < params['prob_mtct_incident_preg']) \
            and df.at[child_id, 'is_alive'] \
            and df.at[child_id, 'hiv_mother_inf'] \
            and (df.at[child_id, 'hiv_mother_art'] != '2') \
            and (((now - df.at[mother_id, 'hiv_date_inf']) / np.timedelta64(1, 'M')) < 9):
            df.at[child_id, 'hiv_inf'] = True

        # mother has existing infection, mother NOT ON ART
        if (random_draw < params['prob_mtct_untreated']) \
            and df.at[child_id, 'is_alive'] \
            and df.at[child_id, 'hiv_mother_inf'] \
            and not df.at[child_id, 'hiv_inf'] \
            and (df.at[child_id, 'hiv_mother_art'] != '2'):
            df.at[child_id, 'hiv_inf'] = True

        #  mother has existing infection, mother ON ART
        if (random_draw < params['prob_mtct_treated']) \
            and df.at[child_id, 'is_alive'] \
            and df.at[child_id, 'hiv_mother_inf'] \
            and not df.at[child_id, 'hiv_inf'] \
            and (df.at[child_id, 'hiv_mother_art'] == '2'):
            df.at[child_id, 'hiv_inf'] = True

        #  TRANSMISSION DURING BREASTFEEDING

        # mother NOT ON ART
        if (random_draw < params['prob_mtct_breastfeeding_untreated']) \
            and df.at[child_id, 'is_alive'] \
            and df.at[child_id, 'hiv_mother_inf'] \
            and not df.at[child_id, 'hiv_inf'] \
            and (df.at[child_id, 'hiv_mother_art'] != '2'):
            df.at[child_id, 'hiv_inf'] = True

        # mother ON ART
        if (random_draw < params['prob_mtct_breastfeeding_treated']) \
            and df.at[child_id, 'is_alive'] \
            and df.at[child_id, 'hiv_mother_inf'] \
            and not df.at[child_id, 'hiv_inf'] \
            and (df.at[child_id, 'hiv_mother_art'] == '2'):
            df.at[child_id, 'hiv_inf'] = True

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ ASSIGN DEATHS ~~~~~~~~~~~~~~~~~~~~~~~~~~

        if df.at[child_id, 'is_alive'] and df.at[child_id, 'hiv_inf']:
            df.at[child_id, 'hiv_date_inf'] = self.sim.date

            # assign fast/slow progressor
            progr = self.rng.choice(['FAST', 'SLOW'], size=1, p=[params['prob_infant_fast_progressor'][0],
                                                                 params['prob_infant_fast_progressor'][1]])

            # then draw death date and assign
            if progr == 'SLOW':
                # draw from weibull
                time_death = self.rng.weibull(a=params['weibull_shape_mort_infant_slow_progressor'],
                                              size=1) * params[
                                 'weibull_scale_mort_infant_slow_progressor']

            else:
                # draw from exp
                time_death = self.rng.exponential(scale=params['exp_rate_mort_infant_fast_progressor'],
                                                  size=1)
                # returns an array not a single value!!

            time_death = pd.to_timedelta(time_death[0] * 365.25, unit='d')

            df.at[child_id, 'hiv_date_death'] = now + time_death

            # schedule the death event
            death = DeathEventHIV(self, individual_id=child_id, cause='hiv')  # make that death event
            death_scheduled = df.at[child_id, 'hiv_date_death']
            self.sim.schedule_event(death, death_scheduled)  # schedule the death


class hiv_event(RegularEvent, PopulationScopeEventMixin):
    """ hiv infection event - adults only
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))  # every 12 months

    def apply(self, population):
        df = population.props
        params = self.module.parameters
        now = self.sim.date

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ FOI ~~~~~~~~~~~~~~~~~~~~~~~~~~

        #  calculate relative infectivity for everyone hiv+
        # infective if: hiv_inf & hiv_on_art = none (0) or poor (1)
        infective = len(df.index[df.is_alive & df.hiv_inf & (df.age_years >= 15) & (df.hiv_on_art != '2')])
        # print('infective', infective)
        total_pop = len(df[df.is_alive & (df.age_years >= 15)])
        foi = params['beta'] * infective / total_pop
        # print('foi:', foi)

        #  relative risk
        eff_susc = pd.Series(0, index=df.index)
        eff_susc.loc[df.is_alive & ~df.hiv_inf & (df.age_years >= 15)] = foi  # foi applied to all HIV- adults
        eff_susc.loc[df.hiv_sexual_risk == 'sex_work'] *= params['rr_HIV_high_sexual_risk_fsw']  # fsw
        eff_susc.loc[df.is_circumcised] *= params['rr_circumcision']  # circumcision
        eff_susc.loc[df.behaviour_change] *= params['rr_behaviour_change']  # behaviour counselling
        # TODO: susceptibility=0 if condom use

        #  sample using the FOI scaled by relative susceptibility
        newly_infected_index = df.index[(self.sim.rng.random_sample(size=len(df)) < eff_susc)]
        # print('newly_infected_index', newly_infected_index)

        df.loc[newly_infected_index, 'hiv_inf'] = True
        df.loc[newly_infected_index, 'hiv_date_inf'] = now

        # TODO: if currently breastfeeding mother, further risk to infant

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ TIME OF DEATH ~~~~~~~~~~~~~~~~~~~~~~~~~~
        death_date = self.sim.rng.weibull(a=params['weibull_shape_mort_adult'], size=len(newly_infected_index)) * \
                     np.exp(self.module.log_scale(df.loc[newly_infected_index, 'age_years']))
        death_date = pd.to_timedelta(death_date * 365.25, unit='d')

        death_date = pd.Series(death_date).dt.floor("S")  # remove microseconds
        # print('death dates without ns: ', death_date)
        df.loc[newly_infected_index, 'hiv_date_death'] = now + death_date

        death_dates = df.hiv_date_death[newly_infected_index]

        # schedule the death event
        for person in newly_infected_index:
            # print('person', person)
            death = DeathEventHIV(self.module, individual_id=person, cause='hiv')  # make that death event
            time_death = death_dates[person]
            # print('time_death: ', time_death)
            # print('now: ', now)
            self.sim.schedule_event(death, time_death)  # schedule the death


class hiv_fsw_event(RegularEvent, PopulationScopeEventMixin):
    """ apply risk of fsw to female pop and transition back to non-fsw
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))  # every 12 months

    def apply(self, population):
        df = population.props
        params = self.module.parameters

        # transition those already fsw back to low risk
        remove = df[df.is_alive & (df.sex == 'F') & df.fsw].sample(
            frac=params['fsw_transition']).index

        df.loc[remove, 'hiv_sexual_risk'] = 'low'

        # recruit new fsw, higher weighting for previous sex work?
        # TODO: should propensity for sex work be clustered by wealth / education / location?
        # check if any data to inform this
        # new fsw recruited to replace removed fsw -> constant proportion over time

        # current proportion of F 15-49 classified as fsw
        fsw = len(df[df.is_alive & df.hiv_sexual_risk == 'fsw'])
        eligible = len(df[df.is_alive & (df.sex == 'F') & (df.age_years.between(15, 49))])

        prop = fsw / eligible

        if prop < params['proportion_female_sex_workers']:
            # number new fsw needed
            recruit = round((prop - params['proportion_female_sex_workers']) * eligible)
            fsw_new = df[df.is_alive & (df.sex == 'F') & (df.age_years.between(15, 49))].sample(
                n=recruit).index
            df.loc[fsw_new, 'hiv_sexual_risk'] = 'sex_work'


class DeathEventHIV(Event, IndividualScopeEventMixin):
    """
    Performs the Death operation on an individual and logs it.
    """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)
        self.cause = cause

    def apply(self, individual_id):
        df = self.sim.population.props

        if df.at[individual_id, 'is_alive'] and (df.at[individual_id, 'hiv_on_art'] != '2'):
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, cause='hiv'),
                                    self.sim.date)

        # Log the death
        self.module.store_DeathsLog['DeathEvent_Time'].append(self.sim.date)
        self.module.store_DeathsLog['DeathEvent_Age'].append(df.age_years[individual_id])
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

        infected_total = len(df[df.hiv_inf & df.is_alive])

        mask = (df.loc[(df.age_years > 14), 'hiv_date_inf'] > self.sim.date - DateOffset(months=self.repeat))
        adult_new_inf = mask.sum()
        print('adult new hiv inf', adult_new_inf)

        ad_prev = len(df[df.hiv_inf & df.is_alive & (df.age_years.between(15, 65))]) / len(
            df[df.is_alive & (df.age_years.between(15, 65))])

        mask = (df.loc[(df.age_years < 15), 'hiv_date_inf'] > self.sim.date - DateOffset(months=self.repeat))
        child_new_inf = mask.sum()

        child_prev = len(df[df.hiv_inf & df.is_alive & (df.age_years.between(0, 14))]) / len(
            df[df.is_alive & (df.age_years.between(0, 14))])

        date_aids_death = df.loc[df.hiv_inf & df.is_alive, 'hiv_date_death']
        year_aids_death = date_aids_death.dt.year
        # print('year_aids_death', year_aids_death)

        #  this shows the deaths scheduled for this year, including those postponed due to ART
        die = sum(1 for x in year_aids_death if int(x) == now.year)
        # print('die: ', die)

        self.module.store['Time'].append(self.sim.date)
        self.module.store['Total_HIV'].append(infected_total)
        self.module.store['HIV_scheduled_deaths'].append(die)
        self.module.store['HIV_new_infections_adult'].append(adult_new_inf)
        self.module.store['HIV_new_infections_child'].append(child_new_inf)
        self.module.store['hiv_prev_adult'].append(ad_prev)
        self.module.store['hiv_prev_child'].append(child_prev)

        # print('hiv outputs: ', self.sim.date, infected_total)
