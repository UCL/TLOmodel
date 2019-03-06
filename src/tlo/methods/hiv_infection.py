"""
HIV infection event
"""

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent, IndividualScopeEventMixin
from tlo.methods import demography


class hiv(Module):
    """
    baseline hiv infection
    """

    def __init__(self, name=None, workbook_path=None, par_est=None):
        super().__init__(name)
        self.workbook_path = workbook_path
        self.beta_calib = par_est
        self.store = {'Time': [], 'Total_HIV': [], 'HIV_scheduled_deaths': [], 'HIV_new_infections_adult': [],
                      'HIV_new_infections_child': []}
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
        'rr_circumcision':
            Parameter(Types.REAL, 'relative reduction in susceptibility due to circumcision'),
        'rr_behaviour_change':
            Parameter(Types.REAL, 'relative reduction in susceptibility due to behaviour modification'),
        'rel_infectiousness_acute':
            Parameter(Types.REAL, 'relative infectiousness during acute stage'),
        'prob_mtct':
            Parameter(Types.REAL, 'probability of mother to child transmission'),
        'prob_mtct_treated':
            Parameter(Types.REAL, 'probability of mother to child transmission, mother on ART'),
        'prob_mtct_incident':
            Parameter(Types.REAL, 'probability of mother to child transmission, mother infected during pregnancy'),
        'prob_mtct_breastfeeding':
            Parameter(Types.REAL, 'probability of mother to child transmission during breastfeeding'),
        'prob_mtct_breastfeeding_incident':
            Parameter(Types.REAL, 'probability of mother to child transmission, mother infected during breastfeeding')
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'hiv_inf': Property(Types.BOOL, 'HIV status'),
        'hiv_date_inf': Property(Types.DATE, 'Date acquired HIV infection'),
        'hiv_date_death': Property(Types.DATE, 'Projected time of AIDS death if untreated'),
        'hiv_sexual_risk_group': Property(Types.CATEGORICAL, 'Sexual risk groups', categories=['low', 'high', 'sex_work']),
        'hiv_mother_inf': Property(Types.BOOL, 'HIV status of mother'),
        'hiv_mother_art': Property(Types.BOOL, 'ART status of mother')
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
        # params['beta'] = \
        #     self.param_list.loc['beta', 'Value1']
        params['irr_hiv_f'] = \
            self.param_list.loc['irr_hiv_f', 'Value1']
        params['rr_circumcision'] = \
            self.param_list.loc['rr_circumcision', 'Value1']
        params['rr_behaviour_change'] = \
            self.param_list.loc['rr_behaviour_change', 'Value1']
        params['rel_infectiousness_acute'] = \
            self.param_list.loc['rel_infectiousness_acute', 'Value1']
        params['rel_infectiousness_late'] = \
            self.param_list.loc['rel_infectiousness_late', 'Value1']
        params['prob_mtct'] = \
            self.param_list.loc['prob_mtct', 'Value1']
        params['prob_mtct_treated'] = \
            self.param_list.loc['prob_mtct_treated', 'Value1']
        params['prob_mtct_incident'] = \
            self.param_list.loc['prob_mtct_incident', 'Value1']
        params['prob_mtct_breastfeeding'] = \
            self.param_list.loc['prob_mtct_breastfeeding', 'Value1']
        params['prob_mtct_breastfeeding_treated'] = \
            self.param_list.loc['prob_mtct_breastfeeding_treated', 'Value1']
        # TODO: put beta in worksheet for default

        if self.beta_calib:
            params['beta'] = float(self.beta_calib)
        # print('beta', params['beta'])

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

        df['hiv_inf'] = False
        df['hiv_date_inf'] = pd.NaT
        df['hiv_date_death'] = pd.NaT
        df['hiv_sexual_risk_group'].values[:] = 'low'
        df['hiv_mother_inf'] = False
        df['hiv_mother_art'] = False

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
        # TODO: upper age limit for high sexual risk

        df = population.props

        male_sample = df[df.is_alive & (df.sex == 'M') & (df.age_years >= 15)].sample(
            frac=self.parameters['proportion_high_sexual_risk_male'], replace=False).index
        female_sample = df[df.is_alive &(df.sex == 'F') & (df.age_years >= 15)].sample(
            frac=self.parameters['proportion_high_sexual_risk_female'], replace=False).index

        # these individuals have higher risk of hiv
        df.loc[male_sample | female_sample, 'hiv_sexual_risk_group'] = 'high'

        # TODO: each year EVENT for 15-49 prob of sex work / high risk, also need prob of returning to low sexual risk,
        # may not need age RR on top of that

    def fsw(self, population):
        """ Assign female sex work to sample of women and change sexual risk to high value
        """

        df = population.props

        fsw = df[df.is_alive & (df.sex == 'F') & (df.age_years >= 15)].sample(
            frac=self.parameters['proportion_female_sex_workers']).index

        # these individuals have higher risk of hiv
        df.loc[fsw, 'hiv_sexual_risk_group'] = 'sex_work'

        # TODO: fsw risk should end at certain age


    def baseline_prevalence(self, population):
        """
        assign baseline hiv prevalence
        """

        # TODO: assign baseline prevalence by risk factors age, sex, wealth, education, district etc.
        # odds ratios from Wingston's analysis dropbox Data-MDHS
        now = self.sim.date
        df = population.props
        params = self.parameters

        prevalence = params['hiv_prev_2010']

        # only for 15-54
        prob_hiv = pd.Series(0, index=df.index)
        prob_hiv.loc[df.is_alive & df.age_years.between(15, 55)] = prevalence  # applied to all adults
        prob_hiv.loc[(df.sex == 'F')] *= params['or_sex_f']
        prob_hiv.loc[df.age_years.between(20,24)] *= params['or_age_gp20']
        prob_hiv.loc[df.age_years.between(25,29)] *= params['or_age_gp25']
        prob_hiv.loc[df.age_years.between(30,34)] *= params['or_age_gp30']
        prob_hiv.loc[df.age_years.between(35,39)] *= params['or_age_gp35']
        prob_hiv.loc[df.age_years.between(40,44)] *= params['or_age_gp40']
        prob_hiv.loc[df.age_years.between(45,50)] *= params['or_age_gp45']
        prob_hiv.loc[(df.age_years >= 50)] *= params['or_age_gp50']
        prob_hiv.loc[~df.li_urban] *= params['or_rural']
        prob_hiv.loc[(df.li_wealth == '2')] *= params['or_windex_poorer']
        prob_hiv.loc[(df.li_wealth == '3')] *= params['or_windex_middle']
        prob_hiv.loc[(df.li_wealth == '4')] *= params['or_windex_richer']
        prob_hiv.loc[(df.li_wealth == '5')] *= params['or_windex_richest']
        prob_hiv.loc[(df.li_ed_lev == '2')] *= params['or_edlevel_primary']
        prob_hiv.loc[(df.li_ed_lev == '3')] *= params['or_edlevel_secondary']  # li_ed_lev=3 secondary and higher

        #  sample scaled by relative susceptibility
        infected_idx = df.index[(self.rng.random_sample(size=len(df)) < prob_hiv)]

        print(infected_idx)
        test = infected_idx.isnull().sum()  # sum number of nan
        print("number of nan: ", test)

        df.loc[infected_idx, 'hiv_inf'] = True

        # for date since infection use a reverse weibull distribution
        # hold the index of all adults with hiv
        inf_adult = df.index[df.is_alive & df.hiv_inf & (df.age_years >= 15)]
        times = self.rng.weibull(a=params['weibull_shape_mort_adult'], size=len(inf_adult)) * \
                        np.exp(self.log_scale(df.loc[inf_adult, 'age_years']))

        time_inf = pd.to_timedelta(times * 365.25, unit='d')
        df.loc[inf_adult, 'hiv_date_inf'] = now - time_inf

        # TODO: include baseline children's prevalence
        #  they don't transmit but have long term health implications of early infection
        child_infected_idx = df.index[df.is_alive & (df.age_years < 15) &
                                      (self.rng.random_sample(size=len(df)) < params['child_hiv_prev2010'])]

        # children are infected at time of birth
        df.loc[child_infected_idx, 'hiv_date_inf'] = df.loc[child_infected_idx, 'date_of_birth']


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
        hiv_ad = df.index[df.is_alive & df.hiv_inf & (df.age_years >= 15)]

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

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
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
        df.at[child_id, 'hiv_sexual_risk_group'] = 'low'

        if df.at[mother_id, 'hiv_inf']:
            df.at[child_id, 'hiv_mother_inf'] = True
    # TODO: change the on_art to a categorical variable
            if df.at[mother_id, 'hiv_on_art']:
                df.at[child_id, 'hiv_mother_art'] = True

        # TODO: include risk during breastfeeding period - scheduled event during whole period of exposure

        #  MTCT
        #  transmission during pregnancy / delivery
        random_draw = self.sim.rng.random_sample(size=1)

        # mother not on ART
        if (random_draw < params['prob_mtct']) and df.at[child_id, 'is_alive'] and df.at[child_id, 'hiv_mother_inf'] and not \
            df.at[child_id, 'hiv_mother_art']:
            df.at[child_id, 'hiv_inf'] = True

        #  mother on ART
        if (random_draw < params['prob_mtct_treated']) and df.at[child_id, 'is_alive'] and df.at[
            child_id, 'hiv_mother_inf'] and \
            df.at[child_id, 'hiv_mother_art']:
            df.at[child_id, 'hiv_inf'] = True

        #  mother has incident infection during pregnancy
        if (random_draw < params['prob_mtct_incident']) and df.at[child_id, 'is_alive'] and df.at[
            child_id, 'hiv_mother_inf'] and not \
            df.at[child_id, 'hiv_mother_art'] and \
            (((now - df.at[mother_id, 'hiv_date_inf']) / np.timedelta64(1, 'M')) < 9):
            df.at[child_id, 'hiv_inf'] = True

        #  assign deaths dates and schedule death for newly infected infant
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

# TODO: include high risk / fsw as ongoing event with recruitment and removal back to low risk


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
        total_pop = len(df[df.is_alive & (df.age_years >= 15)])
        foi = params['beta'] * infective / total_pop
        # print('foi:', foi)

        # TODO: reduce force of infection if behaviour change available
        # TODO: reduce FOI due to condom use

        #  incidence rate ratios and relative susceptibility
        eff_susc = pd.Series(0, index=df.index)
        eff_susc.loc[df.is_alive & ~df.has_hiv & (df.age_years >= 15)] = foi  # foi applied to all HIV- adults

        eff_susc.loc[df.sexual_risk_group == 'high'] *= params['rr_HIV_high_sexual_risk']  # sexual risk group

        eff_susc.loc[df.sexual_risk_group == 'sex_work'] *= params['rr_HIV_high_sexual_risk_fsw']  # fsw

        eff_susc.loc[df.is_circumcised] *= params['rr_circumcision']  # circumcision

        eff_susc.loc[df.behaviour_change] *= params['rr_behaviour_change']  # behaviour counselling

        #  sample using the FOI scaled by relative susceptibility
        newly_infected_index = df.index[(self.sim.rng.random_sample(size=len(df)) < eff_susc)]

        df.loc[newly_infected_index, 'hiv_inf'] = True
        df.loc[newly_infected_index, 'hiv_date_inf'] = now

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ TIME OF DEATH ~~~~~~~~~~~~~~~~~~~~~~~~~~
        death_date = self.sim.rng.weibull(a=params['weibull_shape_mort_adult'], size=len(newly_infected_index)) * \
                     np.exp(self.module.log_scale(df.loc[newly_infected_index, 'age_years']))
        death_date = pd.to_timedelta(death_date * 365.25, unit='d')

        death_date = pd.Series(death_date).dt.floor("S")  # remove microseconds
        # print('death dates without ns: ', death_date)
        df.loc[newly_infected_index, 'hiv_date_death'] = now + death_date

        death_dates = df.date_aids_death[newly_infected_index]

        # schedule the death event
        for person in newly_infected_index:
            # print('person', person)
            death = DeathEventHIV(self.module, individual_id=person, cause='hiv')  # make that death event
            time_death = death_dates[person]
            # print('time_death: ', time_death)
            # print('now: ', now)
            self.sim.schedule_event(death, time_death)  # schedule the death


class DeathEventHIV(Event, IndividualScopeEventMixin):
    """
    Performs the Death operation on an individual and logs it.
    """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)
        self.cause = cause

    def apply(self, individual_id):
        df = self.sim.population.props

        if df.at[individual_id, 'is_alive'] and not df.at[individual_id, 'hiv_on_art']:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, cause='hiv'),
                                    self.sim.date)

        # Log the death
        self.module.store_DeathsLog['DeathEvent_Time'].append(self.sim.date)
        self.module.store_DeathsLog['DeathEvent_Age'].append(df.age_years[individual_id])
        self.module.store_DeathsLog['DeathEvent_Cause'].append(self.cause)


class hiv_mtct_event(RegularEvent, PopulationScopeEventMixin):
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
        super().__init__(module, frequency=DateOffset(months=1))  # every 1 months

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        params = self.module.parameters
        now = self.sim.date

        # find uninfected infants < 18 months
        infant_ids = df.index[df.age_exact_years < 1.5 and not df.has_hiv]

        for id in infant_ids:
            #  find mother's hiv status
            mother = df.at[id, 'mother_id']
            if df.at[mother, 'is_alive'] and df.at[mother, 'has_hiv'] and df.at[mother, 'on_art'] and (
                self.sim.rng.random_sample(size=1) < params['prob_mtct_breastfeeding_treated']):
                df.at[id, 'has_hiv'] = True

            #  assign deaths dates and schedule death for newly infected infant
            if df.at[id, 'is_alive'] and df.at[id, 'has_hiv']:
                df.at[id, 'date_hiv_infection'] = self.sim.date

                # assign fast/slow progressor
                progr = self.sim.rng.choice(['FAST', 'SLOW'], size=1, p=[params['prob_infant_fast_progressor'][0],
                                                                         params['prob_infant_fast_progressor'][1]])

                # then draw death date and assign
                if progr == 'SLOW':
                    # draw from weibull
                    time_death = self.sim.rng.weibull(a=params['weibull_shape_mort_infant_slow_progressor'],
                                                      size=1) * params[
                                     'weibull_scale_mort_infant_slow_progressor']

                else:
                    # draw from exp
                    time_death = self.sim.rng.exponential(scale=params['exp_rate_mort_infant_fast_progressor'],
                                                          size=1)
                    # returns an array not a single value!!

                time_death = pd.to_timedelta(time_death[0] * 365.25, unit='d')
                df.at[id, 'date_aids_death'] = now + time_death

                # schedule the death event
                death = DeathEventHIV(self, individual_id=id, cause='hiv')  # make that death event
                death_scheduled = df.at[id, 'date_aids_death']
                self.sim.schedule_event(death, death_scheduled)  # schedule the death


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

        mask = (df.loc[(df.age_years > 14), 'date_hiv_infection'] > self.sim.date - DateOffset(months=self.repeat))
        adult_new_inf = mask.sum()
        # print(adult_new_inf)

        mask = (df.loc[(df.age_years < 15), 'date_hiv_infection'] > self.sim.date - DateOffset(months=self.repeat))
        child_new_inf = mask.sum()

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
        self.module.store['HIV_new_infections_adult'].append(adult_new_inf)
        self.module.store['HIV_new_infections_child'].append(child_new_inf)

        # print('hiv outputs: ', self.sim.date, infected_total)
