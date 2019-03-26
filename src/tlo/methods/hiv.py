"""
HIV infection event
"""
import logging
import os

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent, IndividualScopeEventMixin
from tlo.methods import demography, healthsystem

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        'proportion_female_sex_workers':
            Parameter(Types.REAL, 'proportion of women who engage in transactional sex'),
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
        'level_of_symptoms':
            Parameter(Types.CATEGORICAL, 'level of symptoms that the individual will have'),
        'qalywt_acute':
            Parameter(Types.REAL, 'QALY weighting for acute hiv infection'),
        'qalywt_chronic':
            Parameter(Types.REAL, 'QALY weighting for chronic hiv infection'),
        'qalywt_aids':
            Parameter(Types.REAL, 'QALY weighting for aids'),
        'vls_m': Parameter(Types.INT, 'rates of viral load suppression males'),
        'vls_f': Parameter(Types.INT, 'rates of viral load suppression males'),
        'vls_child': Parameter(Types.INT, 'rates of viral load suppression in children 0-14 years')
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
        'hiv_mother_art': Property(Types.BOOL, 'ART status of mother'),

        # hiv specific symptoms are matched to the levels for qaly weights above - correct??
        'hiv_specific_symptoms': Property(Types.CATEGORICAL, 'Level of symptoms for hiv',
                                          categories=['none', 'acute', 'chronic', 'aids']),
        'hiv_unified_symptom_code': Property(Types.CATEGORICAL, 'level of symptoms on the standardised scale, 0-4',
                                             categories=[0, 1, 2, 3, 4]),

        # TODO: assign these hs properties in initialise population and on_birth
        'hiv_ever_tested': Property(Types.BOOL, 'ever had a hiv test'),
        'hiv_date_tested': Property(Types.DATE, 'date of hiv test'),
        'hiv_number_tests': Property(Types.INT, 'number of hiv tests taken'),
        'hiv_diagnosed': Property(Types.BOOL, 'hiv+ and tested'),
        'hiv_on_art': Property(Types.CATEGORICAL, 'art status', categories=['0', '1', '2']),
        'hiv_date_art_start': Property(Types.DATE, 'date art started'),
        'hiv_viral_load_test': Property(Types.DATE, 'date last viral load test'),
        'hiv_on_cotrim': Property(Types.BOOL, 'on cotrimoxazole'),
        'hiv_date_cotrim': Property(Types.DATE, 'date cotrimoxazole started')

    }

    TREATMENT_ID = 'hiv_treatment'
    TEST_ID = 'hiv_test'

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        :param data_folder: path of a folder supplied to the Simulation containing data files.
        """

        workbook = pd.read_excel(os.path.join(self.resourcefilepath,
                                              'Method_HIV.xlsx'), sheet_name=None)
        # print('workbook', workbook)

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

        # OR for risk of infection, change to RR
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

        params['vls_m'] = self.param_list.loc['vls_m', 'Value1']
        params['vls_f'] = self.param_list.loc['vls_f', 'Value1']
        params['vls_child'] = self.param_list.loc['vls_child', 'Value1']

        # TODO: put beta in worksheet for default

        if self.beta_calib:
            params['beta'] = float(self.beta_calib)
        # print('beta', params['beta'])

        params['hiv_prev'] = workbook['prevalence']  # for child prevalence

        # symptoms
        # do these map to the specific symptoms or the unified symptom code?
        # probs linked to 3 mths acute, end 24 mths aids and remaining time chronic (out of 12 yr inf)
        params['level_of_symptoms'] = pd.DataFrame(
            data={'level': ['none', 'acute', 'chronic', 'aids'], 'probability': [0, 0.02, 0.81, 0.17]})

        # QALY weights
        # TODO: update QALY weights
        params['qalywt_acute'] = self.sim.modules['QALY'].get_qaly_weight(50)
        params['qalywt_chronic'] = self.sim.modules['QALY'].get_qaly_weight(50)
        params['qalywt_aids'] = self.sim.modules['QALY'].get_qaly_weight(50)

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

        df['hiv_specific_symptoms'] = 'none'
        df['hiv_unified_symptom_code'].values[:] = 0

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
        risk_hiv = pd.Series(0, index=df.index)
        risk_hiv.loc[df.is_alive & df.age_years.between(15, 55)] = 1  # applied to all adults
        risk_hiv.loc[(df.sex == 'F')] *= params['or_sex_f']
        risk_hiv.loc[df.age_years.between(20, 24)] *= params['or_age_gp20']
        risk_hiv.loc[df.age_years.between(25, 29)] *= params['or_age_gp25']
        risk_hiv.loc[df.age_years.between(30, 34)] *= params['or_age_gp30']
        risk_hiv.loc[df.age_years.between(35, 39)] *= params['or_age_gp35']
        risk_hiv.loc[df.age_years.between(40, 44)] *= params['or_age_gp40']
        risk_hiv.loc[df.age_years.between(45, 50)] *= params['or_age_gp45']
        risk_hiv.loc[(df.age_years >= 50)] *= params['or_age_gp50']
        risk_hiv.loc[~df.li_urban] *= params['or_rural']
        risk_hiv.loc[(df.li_wealth == '2')] *= params['or_windex_poorer']
        risk_hiv.loc[(df.li_wealth == '3')] *= params['or_windex_middle']
        risk_hiv.loc[(df.li_wealth == '4')] *= params['or_windex_richer']
        risk_hiv.loc[(df.li_wealth == '5')] *= params['or_windex_richest']
        risk_hiv.loc[(df.li_ed_lev == '2')] *= params['or_edlevel_primary']
        risk_hiv.loc[(df.li_ed_lev == '3')] *= params['or_edlevel_secondary']  # li_ed_lev=3 secondary and higher

        # sample 10% prev, but weight the likelihood of being sampled by the relative risk
        eligible = df.index[df.is_alive & df.age_years.between(15, 55)]
        norm_p = np.array(risk_hiv[eligible])
        norm_p /= norm_p.sum()  # normalize
        # print('norm_p', norm_p)
        infected_idx = self.rng.choice(eligible, size=int(prevalence * (len(eligible))), replace=False,
                                       p=norm_p)

        # print('infected_idx', infected_idx)
        # test = infected_idx.isnull().sum()  # sum number of nan
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

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ ASSIGN LEVEL OF SYMPTOMS ~~~~~~~~~~~~~~~~~~~~~~~~~~

        # TODO: this should be related to time infected
        curr_infected = df.index[df.hiv_inf & df.is_alive]
        level_of_symptoms = self.parameters['level_of_symptoms']
        symptoms = self.rng.choice(level_of_symptoms.level, size=len(curr_infected), p=level_of_symptoms.probability)
        df.loc[curr_infected, 'hiv_specific_symptoms'] = symptoms

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
            death = HivDeathEvent(self, individual_id=person, cause='hiv')  # make that death event
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
            death = HivDeathEvent(self, individual_id=person, cause='hiv')  # make that death event
            time_death = death_dates[person]
            self.sim.schedule_event(death, time_death)  # schedule the death

    def initialise_simulation(self, sim):
        """Get ready for simulation start.
        """
        sim.schedule_event(HivEvent(self), sim.date + DateOffset(months=12))
        sim.schedule_event(FswEvent(self), sim.date + DateOffset(months=12))

        sim.schedule_event(HivLoggingEvent(self), sim.date + DateOffset(days=0))

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

        # Schedule the outreach event...
        self.sim.schedule_event(HivOutreachEvent(self), self.sim.date + DateOffset(months=1))

        # Register with the HealthSystem the treatment interventions that this module runs
        # and define the footprint that each intervention has on the common resources
        footprint_for_test = pd.DataFrame(index=np.arange(1), data={
            'Name': hiv.TEST_ID,
            'Nurse_Time': 5,
            'Doctor_Time': 10,
            'Electricity': False,
            'Water': False})

        self.sim.modules['HealthSystem'].register_interventions(footprint_for_test)

        footprint_for_treatment = pd.DataFrame(index=np.arange(1), data={
            'Name': hiv.TREATMENT_ID,
            'Nurse_Time': 5,
            'Doctor_Time': 10,
            'Electricity': False,
            'Water': False})

        self.sim.modules['HealthSystem'].register_interventions(footprint_for_treatment)

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual
        """
        params = self.parameters
        df = self.sim.population.props
        now = self.sim.date

        # default settings
        df.at[child_id, 'hiv_inf'] = False
        df.at[child_id, 'hiv_date_inf'] = pd.NaT
        df.at[child_id, 'hiv_date_death'] = pd.NaT
        df.at[child_id, 'hiv_sexual_risk'] = 'low'
        df.at[child_id, 'hiv_specific_symptoms'] = 'none'
        df.at[child_id, 'hiv_unified_symptom_code'] = 0

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

            # Level of symptoms
            # TODO: should be linked to time infected
            symptoms = self.rng.choice(self.parameters['level_of_symptoms']['level'],
                                       p=self.parameters['level_of_symptoms']['probability'])

            df.at[child_id, 'mi_specific_symptoms'] = symptoms
            df.at[child_id, 'mi_unified_symptom_code'] = '0'

            # schedule the death event
            death = HivDeathEvent(self, individual_id=child_id, cause='hiv')  # make that death event
            death_scheduled = df.at[child_id, 'hiv_date_death']
            self.sim.schedule_event(death, death_scheduled)  # schedule the death

    def query_symptoms_now(self):
        # This is called by the health-care seeking module
        # All modules refresh the symptomology of persons at this time
        # And report it on the unified symptomology scale
        logger.debug("This is hiv, being asked to report unified symptomology")

        # Map the specific symptoms for this disease onto the unified coding scheme
        df = self.sim.population.props  # shortcut to population properties dataframe

        df.loc[df.is_alive, 'hiv_unified_symptom_code'] = df.loc[df.is_alive, 'hiv_specific_symptoms'].map({
            'none': 0,
            'acute': 1,
            'chronic': 2,
            'aids': 3
        })  # no extreme emergency needed?

        return df.loc[df.is_alive, 'hiv_unified_symptom_code']

    def on_healthsystem_interaction(self, person_id, cue_type, disease_specific):
        logger.debug('This is hiv, being asked what to do at a health system appointment for '
                     'person %d triggered by %s', person_id, cue_type)

        df = self.sim.population.props

        gets_test = False  # default value

        # hiv outreach event -> test, could add probability of testing here before query
        if cue_type == 'OutreachEvent' and disease_specific == 'hiv':
            # everyone gets a test, this will always return true
            gets_test = self.sim.modules['HealthSystem'].query_access_to_service(
                person_id, self.TEST_ID
            )

        # other health care seeking poll
        if cue_type == 'HealthCareSeekingPoll':
            # flip a coin to request a test
            request = self.rng.choice(['True', 'False'], p=[0.5, 0.5])

            if request:
                gets_test = self.sim.modules['HealthSystem'].query_access_to_service(
                    person_id, self.TEST_ID
                )

        if gets_test:
            df.at[person_id, 'ever_tested'] = True

            if df.at[person_id, 'hiv_inf']:
                df.at[person_id, 'hiv_diagnosed'] = True

            # Commission treatment for this individual, returns a boolean
            gets_treatment = self.sim.modules['HealthSystem'].query_access_to_service(
                person_id, self.TREATMENT_ID)

            if gets_treatment:
                event = HivTreatmentEvent(self, person_id)
                self.sim.schedule_event(event, self.sim.date)  # can add in delay before treatment here

    # def on_followup_healthsystem_interaction(self, person_id):
    # TODO: the scheduled follow-up appointments, VL testing, repeat prescriptions etc.
    #     logger.debug('This is a follow-up appointment. Nothing to do')

    def report_qaly_values(self):
        # This must send back a dataframe that reports on the HealthStates for all individuals over
        # the past year

        logger.debug('This is hiv reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe

        params = self.parameters
        # TODO: this should be linked to time infected

        health_values = df.loc[df.is_alive, 'hiv_specific_symptoms'].map({
            'none': 0,
            'acute': params['qalywt_acute'],
            'chronic': params['qalywt_chronic'],
            'aids': params['qalywt_aids']
        })

        return health_values.loc[df.is_alive]


class HivEvent(RegularEvent, PopulationScopeEventMixin):
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
        df.loc[newly_infected_index, 'hiv_specific_symptoms'] = 'none'  # all start at none
        df.loc[newly_infected_index, 'hiv_unified_symptom_code'] = 0

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
            death = HivDeathEvent(self.module, individual_id=person, cause='hiv')  # make that death event
            time_death = death_dates[person]
            # print('time_death: ', time_death)
            # print('now: ', now)
            self.sim.schedule_event(death, time_death)  # schedule the death


class FswEvent(RegularEvent, PopulationScopeEventMixin):
    """ apply risk of fsw to female pop and transition back to non-fsw
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))  # every 12 months

    def apply(self, population):
        df = population.props
        params = self.module.parameters

        # transition those already fsw back to low risk
        remove = df[df.is_alive & (df.sex == 'F') & (df.hiv_sexual_risk == 'sex_work')].sample(
            frac=params['fsw_transition']).index

        df.loc[remove, 'hiv_sexual_risk'] = 'low'

        # recruit new fsw, higher weighting for previous sex work?
        # TODO: should propensity for sex work be clustered by wealth / education / location?
        # check if any data to inform this
        # new fsw recruited to replace removed fsw -> constant proportion over time

        # current proportion of F 15-49 classified as fsw
        fsw = len(df[df.is_alive & (df.hiv_sexual_risk == 'sex_work')])
        eligible = len(df[df.is_alive & (df.sex == 'F') & (df.age_years.between(15, 49))])

        prop = fsw / eligible

        if prop < params['proportion_female_sex_workers']:
            # number new fsw needed
            recruit = round((params['proportion_female_sex_workers'] - prop) * eligible)
            fsw_new = df[df.is_alive & (df.sex == 'F') & (df.age_years.between(15, 49))].sample(
                n=recruit).index
            df.loc[fsw_new, 'hiv_sexual_risk'] = 'sex_work'


class HivDeathEvent(Event, IndividualScopeEventMixin):
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


class HivLoggingEvent(RegularEvent, PopulationScopeEventMixin):
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
        sch_deaths = sum(1 for x in year_aids_death if int(x) == now.year)

        logger.info('%s|summary|%s', self.sim.date,
                    {
                        'TotalInf': infected_total,
                        'hiv_prev_adult': ad_prev,
                        'hiv_prev_child': child_prev,
                        'hiv_new_infections_adult': adult_new_inf,
                        'hiv_new_infections_child': child_new_inf,
                        'hiv_scheduled_deaths': sch_deaths
                    })


class HivOutreachEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=3))

    def apply(self, population):
        # target adults age >15
        df = population.props
        mask_for_person_to_be_reached = (df.age_years >= 15)  # can put multiple conditions here

        target = mask_for_person_to_be_reached.loc[df.is_alive]

        # make and run the actual outreach event by the health system
        outreachevent = healthsystem.OutreachEvent(self.module, disease_specific=self.module.name, target=target)

        self.sim.schedule_event(outreachevent, self.sim.date)


class HivTreatmentEvent(Event, IndividualScopeEventMixin):
    """
    Assigns treatment to individuals
    """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        params = self.module.parameters
        df = self.sim.population.props

        # assign ART
        if df.at[individual_id, 'is_alive'] and df.at[individual_id, 'hiv_diagnosed'] and (df.at[
            individual_id, 'age_years'] < 15):
            df.at[individual_id, 'hiv_on_art'] = self.module.rng.choice(['1', '2'],
                                                                        p=[(1 - params['vls_child']),
                                                                           params['vls_child']])

        if df.at[individual_id, 'is_alive'] and df.at[individual_id, 'hiv_diagnosed'] and (df.at[
            individual_id, 'age_years'] >= 15)and (df.at[individual_id, 'sex'] == 'M'):
            df.at[individual_id, 'hiv_on_art'] = self.module.rng.choice(['1', '2'],
                                                                        p=[(1 - params['vls_m']), params['vls_m']])

        if df.at[individual_id, 'is_alive'] and df.at[individual_id, 'hiv_diagnosed'] and (df.at[
            individual_id, 'age_years'] >= 15) and (df.at[individual_id, 'sex'] == 'F'):
            df.at[individual_id, 'hiv_on_art'] = self.module.rng.choice(['1', '2'],
                                                                        p=[(1 - params['vls_f']), params['vls_f']])

        df.at[individual_id, 'hiv_date_art_start'] = self.sim.date


