"""
HIV infection event
"""
import logging
import os

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent, IndividualScopeEventMixin
from tlo.methods import demography

logger = logging.getLogger(__name__)


# TODO: add property hiv_on_nvp_azt for risk of transmission on birth

class hiv(Module):
    """
    baseline hiv infection
    """

    def __init__(self, name=None, resourcefilepath=None, par_est=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.beta_calib = par_est

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'prob_infant_fast_progressor':
            Parameter(Types.LIST, 'Probabilities that infants are fast or slow progressors'),
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
        'monthly_prob_mtct_breastfeeding_untreated':
            Parameter(Types.REAL, 'probability of mother to child transmission during breastfeeding'),
        'monthly_prob_mtct_breastfeeding_treated':
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
        'qalywt_early':
            Parameter(Types.REAL, 'QALY weighting for early hiv infection'),
        'qalywt_chronic':
            Parameter(Types.REAL, 'QALY weighting for chronic hiv infection'),
        'qalywt_aids':
            Parameter(Types.REAL, 'QALY weighting for aids'),
        'vls_m': Parameter(Types.INT, 'rates of viral load suppression males'),
        'vls_f': Parameter(Types.INT, 'rates of viral load suppression males'),
        'vls_child': Parameter(Types.INT, 'rates of viral load suppression in children 0-14 years'),
        'testing_coverage_male': Parameter(Types.REAL, 'proportion of adult male population tested'),
        'testing_coverage_female': Parameter(Types.REAL, 'proportion of adult female population tested'),
        'testing_prob_individual': Parameter(Types.REAL, 'probability of individual being tested after trigger event'),
        'art_coverage': Parameter(Types.DATA_FRAME, 'estimated ART coverage'),
        'rr_testing_high_risk': Parameter(Types.DATA_FRAME,
                                          'relative increase in testing probability if high sexual risk'),
        'rr_testing_female': Parameter(Types.DATA_FRAME, 'relative change in testing for women versus men'),
        'rr_testing_previously_negative': Parameter(Types.DATA_FRAME,
                                                    'relative change in testing if previously negative versus never tested'),
        'rr_testing_previously_positive': Parameter(Types.DATA_FRAME,
                                                    'relative change in testing if previously positive versus never tested'),
        'rr_testing_age25': Parameter(Types.DATA_FRAME, 'relative change in testing for >25 versus <25'),

        'VL_monitoring_times': Parameter(Types.INT, 'times(months) viral load monitoring required after ART start'),
        'annual_prob_symptomatic_adult': Parameter(Types.REAL, 'annual probability of adults becoming symptomatic'),
        'annual_prob_aids_adult': Parameter(Types.REAL, 'annual probability of adults developing aids'),
        'monthly_prob_symptomatic_infant': Parameter(Types.REAL, 'monthly probability of infants becoming symptomatic'),
        'monthly_prob_aids_infant_fast': Parameter(Types.REAL,
                                                   'monthly probability of infants developing aids - fast progressors'),
        'monthly_prob_aids_infant_slow': Parameter(Types.REAL,
                                                   'monthly probability of infants developing aids - slow progressors'),

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
                                          categories=['none', 'early', 'symptomatic', 'aids']),
        'hiv_unified_symptom_code': Property(Types.CATEGORICAL, 'level of symptoms on the standardised scale, 0-4',
                                             categories=[0, 1, 2, 3, 4]),

        'hiv_ever_tested': Property(Types.BOOL, 'ever had a hiv test'),
        'hiv_date_tested': Property(Types.DATE, 'date of hiv test'),
        'hiv_number_tests': Property(Types.INT, 'number of hiv tests taken'),
        'hiv_diagnosed': Property(Types.BOOL, 'hiv+ and tested'),
        'hiv_on_art': Property(Types.CATEGORICAL, 'art status', categories=[0, 1, 2]),
        'hiv_date_art_start': Property(Types.DATE, 'date art started'),
        'hiv_viral_load_test': Property(Types.DATE, 'date last viral load test'),
        'hiv_on_cotrim': Property(Types.BOOL, 'on cotrimoxazole'),
        'hiv_date_cotrim': Property(Types.DATE, 'date cotrimoxazole started'),
        'hiv_fast_progressor': Property(Types.BOOL, 'infant classified as fast progressor')

    }

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

        params['prob_infant_fast_progressor'] = \
            self.param_list.loc['prob_infant_fast_progressor'].values
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
        params['monthly_prob_mtct_breastfeeding_untreated'] = \
            self.param_list.loc['monthly_prob_mtct_breastfeeding_untreated', 'Value1']
        params['monthly_prob_mtct_breastfeeding_treated'] = \
            self.param_list.loc['monthly_prob_mtct_breastfeeding_treated', 'Value1']
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

        # QALY weights
        params['qalywt_early'] = self.sim.modules['QALY'].get_qaly_weight(22)  # Early HIV without anemia
        params['qalywt_chronic'] = self.sim.modules['QALY'].get_qaly_weight(17)  # Symptomatic HIV without anemia
        params['qalywt_aids'] = self.sim.modules['QALY'].get_qaly_weight(
            19)  # AIDS without antiretroviral treatment without anemia

        params['testing_coverage_male'] = self.param_list.loc['testing_coverage_male_2010', 'Value1']
        params['testing_coverage_female'] = self.param_list.loc['testing_coverage_female_2010', 'Value1']
        params['testing_prob_individual'] = self.param_list.loc['testing_prob_individual', 'Value1']  # dummy value
        params['art_coverage'] = self.param_list.loc['art_coverage', 'Value1']

        params['rr_testing_high_risk'] = self.param_list.loc['rr_testing_high_risk', 'Value1']
        params['rr_testing_female'] = self.param_list.loc['rr_testing_female', 'Value1']
        params['rr_testing_previously_negative'] = self.param_list.loc['rr_testing_previously_negative', 'Value1']
        params['rr_testing_previously_positive'] = self.param_list.loc['rr_testing_previously_positive', 'Value1']
        params['rr_testing_age25'] = self.param_list.loc['rr_testing_age25', 'Value1']
        params['vls_m'] = self.param_list.loc['vls_m', 'Value1']
        params['vls_f'] = self.param_list.loc['vls_f', 'Value1']
        params['vls_child'] = self.param_list.loc['vls_child', 'Value1']

        params['annual_prob_symptomatic_adult'] = self.param_list.loc['annual_prob_symptomatic_adult', 'Value1']
        params['annual_prob_aids_adult'] = self.param_list.loc['annual_prob_aids_adult', 'Value1']
        params['monthly_prob_symptomatic_infant'] = self.param_list.loc['monthly_prob_symptomatic_infant', 'Value1']
        params['monthly_prob_aids_infant_fast'] = self.param_list.loc['monthly_prob_aids_infant_fast', 'Value1']
        params['monthly_prob_aids_infant_slow'] = self.param_list.loc['monthly_prob_aids_infant_slow', 'Value1']

        self.parameters['initial_art_coverage'] = workbook['coverage']

        self.parameters['VL_monitoring_times'] = workbook['VL_monitoring']

    def initialise_population(self, population):
        """Set our property values for the initial population.
        """
        df = population.props

        df['hiv_inf'] = False
        df['hiv_date_inf'] = pd.NaT
        df['hiv_date_death'] = pd.NaT
        df['hiv_sexual_risk'].values[:] = 'low'
        df['hiv_mother_inf'] = False
        df['hiv_mother_art'] = False

        df['hiv_specific_symptoms'] = 'none'
        df['hiv_unified_symptom_code'].values[:] = 0

        df['hiv_ever_tested'] = False  # default: no individuals tested
        df['hiv_date_tested'] = pd.NaT
        df['hiv_number_tests'] = 0
        df['hiv_diagnosed'] = False
        df['hiv_on_art'].values[:] = 0
        df['hiv_date_art_start'] = pd.NaT
        df['hiv_viral_load_test'] = pd.NaT
        df['hiv_on_cotrim'] = False
        df['hiv_date_cotrim'] = pd.NaT
        df['hiv_fast_progressor'] = False

        self.fsw(population)  # allocate proportion of women with very high sexual risk (fsw)
        self.baseline_prevalence(population)  # allocate baseline prevalence
        self.initial_pop_deaths_children(population)  # add death dates for children
        self.initial_pop_deaths_adults(population)  # add death dates for adults
        self.assign_symptom_level(population)  # assign symptom level for all infected
        self.baseline_tested(population)  # allocate baseline art coverage
        self.baseline_art(population)  # allocate baseline art coverage

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

        # ----------------------------------- ADULT HIV -----------------------------------

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

        # sample 10% prev, weight the likelihood of being sampled by the relative risk
        eligible = df.index[df.is_alive & df.age_years.between(15, 55)]
        norm_p = np.array(risk_hiv[eligible])
        norm_p /= norm_p.sum()  # normalise
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

        # ----------------------------------- CHILD HIV -----------------------------------

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
        hiv_index = df_hivprob.index[
            df.is_alive & (df_hivprob.prev_prop > random_draw) & df_hivprob.age_years.between(0, 14)]
        # print(hiv_index)

        df.loc[hiv_index, 'hiv_inf'] = True
        df.loc[hiv_index, 'hiv_date_inf'] = df.loc[hiv_index, 'date_of_birth']
        df.loc[hiv_index, 'hiv_fast_progressor'] = False

    def initial_pop_deaths_children(self, population):
        """ assign death dates to baseline hiv-infected population - INFANTS
        assume all are slow progressors, otherwise time to death shorter than time infected
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
        # children hiv+ aged >3 have survival draws from adult weibull distr otherwise death date pre-2010
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

    def assign_symptom_level(self, population):
        """ assign level of symptoms to infected people
        """
        df = population.props
        now = self.sim.date
        params = self.parameters

        # ----------------------------------- ADULT SYMPTOMS -----------------------------------

        # baseline pop - adults
        adults = df[df.is_alive & df.hiv_inf & (df.age_years >= 15)].index
        df.loc[adults, 'hiv_specific_symptoms'] = 'early'
        df.loc[adults, 'hiv_unified_symptom_code'] = 1

        # if <2 years from scheduled death = aids
        time_death = (df.loc[adults, 'hiv_date_death'] - now).dt.days  # returns days
        death_soon = time_death < (2 * 365.25)
        idx = adults[death_soon]
        df.loc[idx, 'hiv_specific_symptoms'] = 'aids'
        df.loc[idx, 'hiv_unified_symptom_code'] = 3

        # if >= 39 months from date infection, 0.5 prob symptomatic
        time_inf = (now - df.loc[adults, 'hiv_date_inf']).dt.days  # returns days
        t2 = time_inf >= (39 * 12)
        keep = adults[t2]
        symp = self.rng.choice(keep, size=int(0.5 * (len(keep))), replace=False)
        df.loc[symp, 'hiv_specific_symptoms'] = 'symptomatic'
        df.loc[symp, 'hiv_unified_symptom_code'] = 2

        # if >= 6.2 yrs from date infection, 0.5 prob of aids
        time_inf = (now - df.loc[adults, 'hiv_date_inf']).dt.days  # returns days
        t1 = time_inf >= (6.2 * 365.25)
        keep = adults[t1]
        aids = self.rng.choice(keep, size=int(0.5 * (len(keep))), replace=False)
        df.loc[aids, 'hiv_specific_symptoms'] = 'aids'
        df.loc[aids, 'hiv_unified_symptom_code'] = 3

        # print(symp)

        # ----------------------------------- CHILD SYMPTOMS -----------------------------------

        # baseline pop - infants, all assumed slow progressors
        infants = df[df.is_alive & df.hiv_inf & (df.age_years < 15)].index
        df.loc[infants, 'hiv_specific_symptoms'] = 'early'
        df.loc[infants, 'hiv_unified_symptom_code'] = 1

        # if <1 years from scheduled death = aids
        time_death = (df.loc[infants, 'hiv_date_death'] - now).dt.days  # returns days
        death_soon = time_death < 365.25
        idx = infants[death_soon]
        df.loc[idx, 'hiv_specific_symptoms'] = 'aids'
        df.loc[idx, 'hiv_unified_symptom_code'] = 3

        # if > 14 months from date infection, 0.5 prob of symptoms
        time_inf = (now - df.loc[infants, 'hiv_date_inf']).dt.days  # returns days
        t1 = time_inf >= (14 * 12)
        keep = infants[t1]
        aids = self.rng.choice(keep, size=int(0.5 * (len(keep))), replace=False)
        df.loc[aids, 'hiv_specific_symptoms'] = 'symptomatic'
        df.loc[aids, 'hiv_unified_symptom_code'] = 2

        # if >= 3.1 years from date infection, 0.5 prob of aids
        time_inf = (now - df.loc[infants, 'hiv_date_inf']).dt.days  # returns days
        t1 = time_inf >= (3.1 * 365.25)
        keep = infants[t1]
        aids = self.rng.choice(keep, size=int(0.5 * (len(keep))), replace=False)
        df.loc[aids, 'hiv_specific_symptoms'] = 'aids'
        df.loc[aids, 'hiv_unified_symptom_code'] = 3

        # print(aids)

    def baseline_tested(self, population):
        """ assign initial hiv testing levels
        """
        now = self.sim.date
        df = population.props

        # get a list of random numbers between 0 and 1 for the whole population
        random_draw = self.sim.rng.random_sample(size=len(df))

        # probability of baseline population ever testing for HIV
        art_index_male = df.index[
            (random_draw < self.parameters['testing_coverage_male']) & df.is_alive & (df.sex == 'M') & (
                df.age_years >= 15)]
        # print('art_index: ', art_index)

        art_index_female = df.index[
            (random_draw < self.parameters['testing_coverage_female']) & df.is_alive & (df.sex == 'F') & (
                df.age_years >= 15)]

        # we don't know date tested, assume date = now
        df.loc[art_index_male | art_index_female, 'hiv_ever_tested'] = True
        df.loc[art_index_male | art_index_female, 'hiv_date_tested'] = now
        df.loc[art_index_male | art_index_female, 'hiv_number_tests'] = 1

        # outcome of test
        diagnosed_idx = df.index[df.hiv_ever_tested & df.is_alive & df.hiv_inf]
        df.loc[diagnosed_idx, 'hiv_diagnosed'] = True

    def baseline_art(self, population):
        """ assign initial art coverage levels
        """
        now = self.sim.date
        df = population.props

        worksheet = self.parameters['initial_art_coverage']

        coverage = worksheet.loc[worksheet.year == now.year, ['year', 'single_age', 'sex', 'prop_coverage']]
        # print('coverage: ', coverage.head(20))

        # merge all susceptible individuals with their coverage probability based on sex and age
        df_art = df.merge(coverage,

                          left_on=['age_years', 'sex'],

                          right_on=['single_age', 'sex'],

                          how='left')

        # no data for ages 100+ so fill missing values with 0
        df_art['prop_coverage'] = df_art['prop_coverage'].fillna(0)
        # print('df_with_age_art_prob: ', df_with_age_art_prob.head(20))

        assert df_art.prop_coverage.isna().sum() == 0  # check there is a probability for every individual

        # get a list of random numbers between 0 and 1 for the whole population
        random_draw = self.sim.rng.random_sample(size=len(df_art))

        # probability of baseline population receiving art: requirement = hiv_diagnosed
        art_idx_child = df_art.index[
            (random_draw < df_art.prop_coverage) & df.is_alive & df_art.hiv_inf & df.hiv_diagnosed &
            df_art.age_years.between(0, 14)]

        df.loc[art_idx_child, 'hiv_on_art'] = 2  # assumes all are adherent at baseline
        df.loc[art_idx_child, 'hiv_date_art_start'] = now

        art_idx_adult = df_art.index[
            (random_draw < df_art.prop_coverage) & df.is_alive & df_art.hiv_inf & df.hiv_diagnosed &
            df_art.age_years.between(15, 64)]

        df.loc[art_idx_adult, 'hiv_on_art'] = 2  # assumes all are adherent, then stratify into category 1/2
        df.loc[art_idx_adult, 'hiv_date_art_start'] = now

        # allocate proportion to non-adherent category
        # if condition added, error with small numbers of children to sample
        if len(df[df.is_alive & (df.hiv_on_art == 2) & (df.age_years.between(0, 14))]) > 5:
            idx_c = df[df.is_alive & (df.hiv_on_art == 2) & (df.age_years.between(0, 14))].sample(
                frac=(1 - self.parameters['vls_child'])).index
            df.loc[idx_c, 'hiv_on_art'] = 1  # change to non=adherent

        idx_m = df[df.is_alive & (df.hiv_on_art == 2) & (df.sex == 'M') & (df.age_years.between(15, 64))].sample(
            frac=(1 - self.parameters['vls_m'])).index
        df.loc[idx_m, 'hiv_on_art'] = 1  # change to non=adherent

        idx_f = df[df.is_alive & (df.hiv_on_art == 2) & (df.sex == 'F') & (df.age_years.between(15, 64))].sample(
            frac=(1 - self.parameters['vls_f'])).index
        df.loc[idx_f, 'hiv_on_art'] = 1  # change to non=adherent

    def initialise_simulation(self, sim):
        """Get ready for simulation start.
        """
        sim.schedule_event(HivEvent(self), sim.date + DateOffset(months=12))
        sim.schedule_event(FswEvent(self), sim.date + DateOffset(months=12))
        sim.schedule_event(SymptomUpdateEventAdult(self), sim.date + DateOffset(months=12))
        sim.schedule_event(SymptomUpdateEventInfant(self), sim.date + DateOffset(months=1))
        # sim.schedule_event(HivOutreachEvent(self), sim.date + DateOffset(months=12))

        sim.schedule_event(HivLoggingEvent(self), sim.date + DateOffset(days=0))

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

        # Schedule the outreach event...
        # self.sim.schedule_event(HivOutreachEvent(self), self.sim.date + DateOffset(months=1))

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
        df.at[child_id, 'hiv_fast_progressor'] = False

        df.at[child_id, 'hiv_ever_tested'] = False
        df.at[child_id, 'hiv_date_tested'] = pd.NaT
        df.at[child_id, 'hiv_number_tests'] = 0
        df.at[child_id, 'hiv_diagnosed'] = False
        df.at[child_id, 'hiv_on_art'] = 0
        df.at[child_id, 'hiv_date_art_start'] = pd.NaT
        df.at[child_id, 'hiv_viral_load_test'] = pd.NaT
        df.at[child_id, 'hiv_on_cotrim'] = False
        df.at[child_id, 'hiv_date_cotrim'] = pd.NaT

        df.at[child_id, 'hiv_mother_inf'] = False
        df.at[child_id, 'hiv_mother_art'] = False

        if df.at[mother_id, 'hiv_inf']:
            df.at[child_id, 'hiv_mother_inf'] = True
            if df.at[mother_id, 'hiv_on_art'] == 2:
                df.at[child_id, 'hiv_mother_art'] = True

        # ----------------------------------- MTCT - PREGNANCY -----------------------------------

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

        # ----------------------------------- ASSIGN DEATHS + FAST/SLOW PROGRESSOR -----------------------------------

        if df.at[child_id, 'is_alive'] and df.at[child_id, 'hiv_inf']:
            df.at[child_id, 'hiv_date_inf'] = self.sim.date

            # assign fast/slow progressor
            progr = self.rng.choice(['FAST', 'SLOW'], size=1, p=[params['prob_infant_fast_progressor'][0],
                                                                 params['prob_infant_fast_progressor'][1]])

            # then draw death date and assign fast/slow progressor
            if progr == 'SLOW':
                # draw from weibull
                time_death = self.rng.weibull(a=params['weibull_shape_mort_infant_slow_progressor'],
                                              size=1) * params[
                                 'weibull_scale_mort_infant_slow_progressor']
                df.at[child_id, 'hiv_specific_symptoms'] = 'early'
                df.at[child_id, 'hiv_unified_symptom_code'] = 1

            else:
                # draw from exp, returns an array not a single value!!
                time_death = self.rng.exponential(scale=params['exp_rate_mort_infant_fast_progressor'],
                                                  size=1)
                df.at[child_id, 'hiv_fast_progressor'] = True
                df.at[child_id, 'hiv_specific_symptoms'] = 'symptomatic'
                df.at[child_id, 'hiv_unified_symptom_code'] = 2

            time_death = pd.to_timedelta(time_death[0] * 365.25, unit='d')
            df.at[child_id, 'hiv_date_death'] = now + time_death

            # schedule the death event
            death = HivDeathEvent(self, individual_id=child_id, cause='hiv')  # make that death event
            death_scheduled = df.at[child_id, 'hiv_date_death']
            self.sim.schedule_event(death, death_scheduled)  # schedule the death

        # ----------------------------------- PMTCT -----------------------------------
        # TODO: PMTCT
        # TODO: check difference between HIV/AIDS, PMTCT and PMTCT. Both have code 90
        # first contact is testing, then schedule treatment / prophylaxis as needed
        if df.at[child_id, 'hiv_mother_inf'] and not df.at[child_id, 'hiv_diagnosed']:
            event = HSI_Hiv_InfantScreening(self, person_id=child_id)
            self.sim.modules['HealthSystem'].schedule_event(event,
                                                            priority=1,
                                                            topen=self.sim.date,
                                                            tclose=self.sim.date + DateOffset(weeks=4)
                                                            )

    # TODO: modify this - include piggy-back appt for other diseases
    # need to include malaria testing at follow-up appts
    def on_healthsystem_interaction(self, person_id, treatment_id):

        logger.debug('This is hiv, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    def report_qaly_values(self):
        # This must send back a dataframe that reports on the HealthStates for all individuals over
        # the past year

        logger.debug('This is hiv reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe
        params = self.parameters

        health_values = df.loc[df.is_alive, 'hiv_specific_symptoms'].map({
            'none': 0,
            'early': params['qalywt_early'],
            'symptomatic': params['qalywt_chronic'],
            'aids': params['qalywt_aids']
        })

        return health_values.loc[df.is_alive]


# ---------------------------------------------------------------------------
#   hiv infection event
# ---------------------------------------------------------------------------

class HivEvent(RegularEvent, PopulationScopeEventMixin):
    """ hiv infection event - adults only
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))  # every 12 months

    def apply(self, population):
        df = population.props
        params = self.module.parameters
        now = self.sim.date

        # ----------------------------------- FOI -----------------------------------

        #  calculate relative infectivity for everyone hiv+
        # infective if: hiv_inf & hiv_on_art = none (0) or poor (1)
        infective = len(df.index[df.is_alive & df.hiv_inf & (df.age_years >= 15) & (df.hiv_on_art != 2)])
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
        df.loc[newly_infected_index, 'hiv_specific_symptoms'] = 'early'  # all start at early
        df.loc[newly_infected_index, 'hiv_unified_symptom_code'] = 1

        # ----------------------------------- TIME OF DEATH -----------------------------------
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


# TODO: monthly risk of hiv with breastfeeding
# depends on PMTCT - assume complete protection?
class HivMtctEvent(RegularEvent, PopulationScopeEventMixin):
    """ hiv infection event during breastfeeding
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        df = population.props
        params = self.module.parameters
        now = self.sim.date

        # TODO: check if any risk once child is on ART
        # mother NOT ON ART & child NOT ON ART
        i1 = df.index[(self.sim.rng.random_sample(size=len(df)) < params[
            'monthly_prob_mtct_breastfeeding_untreated'])
                      & df.is_alive
                      & ~df.hiv_inf
                      & ~df.hiv_on_art
                      & (df.age_exact_years <= 1.5)
                      & df.hiv_mother_inf
                      & ~df.hiv_mother_art]

        # mother ON ART & child NOT ON ART
        i2 = df.index[(self.sim.rng.random_sample(size=len(df)) < params[
            'monthly_prob_mtct_breastfeeding_untreated'])
                      & df.is_alive
                      & ~df.hiv_inf
                      & ~df.hiv_on_art
                      & (df.age_exact_years <= 1.5)
                      & df.hiv_mother_inf
                      & df.hiv_mother_art]

        new_inf = np.concatenate(i1, i2)

        df.loc[new_inf, 'hiv_inf'] = True
        df.loc[new_inf, 'hiv_date_inf'] = now
        df.loc[new_inf, 'hiv_specific_symptoms'] = 'early'  # default - all start at early
        df.loc[new_inf, 'hiv_unified_symptom_code'] = 1

        # ----------------------------------- TIME OF DEATH -----------------------------------
        # assign fast progressor
        fast = self.sim.rng.choice([True, False], size=len(new_inf), p=[params['prob_infant_fast_progressor'][0],
                                                                        params['prob_infant_fast_progressor'][1]])

        fast_idx = new_inf[fast]
        time_death_fast = self.sim.rng.exponential(scale=params['exp_rate_mort_infant_fast_progressor'],
                                                   size=len(fast_idx))
        time_death_fast = pd.to_timedelta(time_death_fast[0] * 365.25, unit='d')
        df.loc[fast_idx, 'hiv_date_death'] = now + time_death_fast
        df.loc[fast_idx, 'hiv_specific_symptoms'] = 'symptomatic'
        df.loc[fast_idx, 'hiv_unified_symptom_code'] = 2

        # assign slow progressor
        slow_idx = new_inf[~fast]
        time_death_slow = self.sim.rng.weibull(a=params['weibull_shape_mort_infant_slow_progressor'],
                                               size=len(slow_idx)) * params[
                              'weibull_scale_mort_infant_slow_progressor']
        time_death_slow = pd.to_timedelta(time_death_slow[0] * 365.25, unit='d')
        df.loc[slow_idx, 'hiv_date_death'] = now + time_death_slow

        # schedule the death event
        for person in new_inf:
            death = HivDeathEvent(self, individual_id=person, cause='hiv')  # make that death event
            time_death = df.loc[person, 'hiv_date_death']
            self.sim.schedule_event(death, time_death)  # schedule the death


class SymptomUpdateEventAdult(RegularEvent, PopulationScopeEventMixin):
    """ update the status of symptoms for infected adults
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))  # every 12 months

    def apply(self, population):
        df = population.props
        params = self.module.parameters

        # ----------------------------------- DEVELOP SYMPTOMS -----------------------------------

        # hazard of moving from early to symptomatic
        symp = df.index[(self.sim.rng.random_sample(size=len(df)) < params[
            'annual_prob_symptomatic_adult']) & df.is_alive & df.hiv_inf & (df.age_years >= 15) & (
                            df.hiv_specific_symptoms == 'early') & ((df.hiv_on_art == 0) | (df.hiv_on_art == 1))]
        df.loc[symp, 'hiv_specific_symptoms'] = 'symptomatic'
        df.loc[symp, 'hiv_unified_symptom_code'] = 2

        # for each person determine whether they will seek care on symptom change
        # get_prob_seek_care will be the healthcare seeking function developed by Wingston
        seeks_care = pd.Series(data=False, index=df.loc[symp].index)
        for i in df.index[symp]:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=2)
            seeks_care[i] = self.module.rng.rand() < prob

        if seeks_care.sum() > 0:
            for person_index in seeks_care.index[seeks_care]:
                logger.debug(
                    'This is HivEvent, scheduling Hiv_PresentsForCareWithSymptoms for person %d',
                    person_index)
                event = HSI_Hiv_PresentsForCareWithSymptoms(self.module, person_id=person_index)
                self.sim.modules['HealthSystem'].schedule_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(weeks=2)
                                                                )
        else:
            logger.debug(
                'This is HivEvent, There is  no one with new severe symptoms so no new healthcare seeking')

        # ----------------------------------- DEVELOP AIDS -----------------------------------

        # hazard of moving to aids (from early or symptomatic)
        aids = df.index[(self.sim.rng.random_sample(size=len(df)) < params[
            'annual_prob_aids_adult']) & df.is_alive & df.hiv_inf & (df.age_years >= 15) & (
                            df.hiv_specific_symptoms != 'aids') & ((df.hiv_on_art == 0) | (df.hiv_on_art == 1))]
        df.loc[aids, 'hiv_specific_symptoms'] = 'aids'
        df.loc[aids, 'hiv_unified_symptom_code'] = 3

        # for each person determine whether they will seek care on symptom change
        seeks_care = pd.Series(data=False, index=df.loc[aids].index)
        for i in df.index[aids]:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=3)
            seeks_care[i] = self.module.rng.rand() < prob

        if seeks_care.sum() > 0:
            for person_index in seeks_care.index[seeks_care]:
                logger.debug(
                    'This is HivEvent, scheduling Hiv_PresentsForCareWithSymptoms for person %d',
                    person_index)
                event = HSI_Hiv_PresentsForCareWithSymptoms(self.module, person_id=person_index)
                self.sim.modules['HealthSystem'].schedule_event(event,
                                                                priority=3,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(weeks=2)
                                                                )


class SymptomUpdateEventInfant(RegularEvent, PopulationScopeEventMixin):
    """ update the status of symptoms for infected children
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # every month

    def apply(self, population):
        df = population.props
        params = self.module.parameters

        # ----------------------------------- DEVELOP SYMPTOMS -----------------------------------

        # hazard of moving from early to symptomatic,  apply to slow progressing infants only
        symp = df.index[(self.sim.rng.random_sample(size=len(df)) < params[
            'monthly_prob_symptomatic_infant']) & df.is_alive & df.hiv_inf & (df.age_years < 15) & (
                            df.hiv_specific_symptoms == 'early') & ((df.hiv_on_art == 0) | (df.hiv_on_art == 1))]
        df.loc[symp, 'hiv_specific_symptoms'] = 'symptomatic'
        df.loc[symp, 'hiv_unified_symptom_code'] = 2

        # for each person determine whether they will seek care on symptom change
        # get_prob_seek_care will be the healthcare seeking function developed by Wingston
        seeks_care = pd.Series(data=False, index=df.loc[symp].index)
        for i in df.index[symp]:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=2)
            seeks_care[i] = self.module.rng.rand() < prob

        if seeks_care.sum() > 0:
            for person_index in seeks_care.index[seeks_care]:
                logger.debug(
                    'This is HivEvent, scheduling Hiv_PresentsForCareWithSymptoms for person %d',
                    person_index)
                event = HSI_Hiv_PresentsForCareWithSymptoms(self.module, person_id=person_index)
                self.sim.modules['HealthSystem'].schedule_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(weeks=2)
                                                                )
        else:
            logger.debug(
                'This is HivEvent, There is  no one with new severe symptoms so no new healthcare seeking')

        # ----------------------------------- DEVELOP AIDS -----------------------------------

        # apply prob of aids to infants, fast and slow
        aids_fast = df.index[(self.sim.rng.random_sample(size=len(df)) < params[
            'monthly_prob_aids_infant_fast']) & df.is_alive & df.hiv_inf & (df.age_years < 15) & (
                                 df.hiv_specific_symptoms != 'aids') & df.hiv_fast_progressor & (
                                 (df.hiv_on_art == 0) | (df.hiv_on_art == 1))]
        df.loc[aids_fast, 'hiv_specific_symptoms'] = 'aids'
        df.loc[aids_fast, 'hiv_unified_symptom_code'] = 3

        aids_slow = df.index[(self.sim.rng.random_sample(size=len(df)) < params[
            'monthly_prob_aids_infant_fast']) & df.is_alive & df.hiv_inf & (df.age_years < 15) & (
                                 df.hiv_specific_symptoms != 'aids') & ~df.hiv_fast_progressor & (
                                 (df.hiv_on_art == 0) | (df.hiv_on_art == 1))]
        df.loc[aids_slow, 'hiv_specific_symptoms'] = 'aids'
        df.loc[aids_slow, 'hiv_unified_symptom_code'] = 3

        aids = np.concatenate([aids_fast, aids_slow])  # join the indices to get all new aids cases

        # for each person determine whether they will seek care on symptom change
        seeks_care = pd.Series(data=False, index=df.loc[aids].index)
        for i in df.index[aids]:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=3)
            seeks_care[i] = self.module.rng.rand() < prob

        if seeks_care.sum() > 0:
            for person_index in seeks_care.index[seeks_care]:
                logger.debug(
                    'This is HivEvent, scheduling Hiv_PresentsForCareWithSymptoms for person %d',
                    person_index)
                event = HSI_Hiv_PresentsForCareWithSymptoms(self.module, person_id=person_index)
                self.sim.modules['HealthSystem'].schedule_event(event,
                                                                priority=3,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(weeks=2)
                                                                )


# ---------------------------------------------------------------------------
#   Health system interactions
# ---------------------------------------------------------------------------

class HSI_Hiv_PresentsForCareWithSymptoms(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It is first appointment that someone has when they present to the healthcare system with the
    symptoms of hiv.
    Outcome is testing
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one outpatient appt
        the_appt_footprint['VCTPositive'] = 1  # Voluntary Counseling and Testing Program - For HIV-Positive

        # Get the consumables required
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = \
            pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'HIV Testing Services', 'Intervention_Pkg_Code'])[
                0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Hiv_Testing'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug(
            'This is HSI_Hiv_PresentsForCareWithSymptoms, giving a test in the first appointment for person %d',
            person_id)

        df = self.sim.population.props

        df.at[person_id, 'hiv_ever_tested'] = True
        df.at[person_id, 'hiv_date_tested'] = self.sim.date
        df.at[person_id, 'hiv_number_tests'] = df.at[person_id, 'hiv_number_tests'] + 1

        # if hiv+ schedule treatment
        if df.at[person_id, 'hiv_inf']:
            df.at[person_id, 'hiv_diagnosed'] = True

            # request treatment
            logger.debug(
                '....This is HSI_Hiv_PresentsForCareWithSymptoms: scheduling hiv treatment for person %d on date %s',
                person_id, self.sim.date)

            treatment = HSI_Hiv_StartTreatment(self.module, person_id=person_id)

            # Request the health system to start treatment
            self.sim.modules['HealthSystem'].schedule_event(treatment,
                                                            priority=2,
                                                            topen=self.sim.date,
                                                            tclose=None)


class HSI_Hiv_InfantScreening(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event - testing of infants exposed to hiv
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Peds'] = 1  # This requires one infant hiv appt

        # Get the consumables required
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = \
            pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'HIV Testing Services', 'Intervention_Pkg_Code'])[
                0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Hiv_TestingInfant'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug('This is HSI_Hiv_InfantScreening, a first appointment for infant %d', person_id)

        df = self.sim.population.props

        df.at[person_id, 'hiv_ever_tested'] = True

        # if hiv+ schedule treatment
        # TODO: give cotrim also
        if df.at[person_id, 'hiv_inf']:
            df.at[person_id, 'hiv_diagnosed'] = True

            # request treatment
            logger.debug('....This is HSI_Hiv_InfantScreening: scheduling hiv treatment for person %d on date %s',
                         person_id, self.sim.date)

            treatment = HSI_Hiv_StartInfantTreatment(self.module, person_id=person_id)

            # Request the health system to start treatment
            self.sim.modules['HealthSystem'].schedule_event(treatment,
                                                            priority=2,
                                                            topen=self.sim.date,
                                                            tclose=None)
        # if hiv- then give cotrim + NVP/AZT
        else:
            # request treatment
            logger.debug('....This is HSI_Hiv_InfantScreening: scheduling hiv treatment for person %d on date %s',
                         person_id, self.sim.date)

            treatment = HSI_Hiv_StartInfantProphylaxis(self.module, person_id=person_id)

            # Request the health system to start treatment
            self.sim.modules['HealthSystem'].schedule_event(treatment,
                                                            priority=2,
                                                            topen=self.sim.date,
                                                            tclose=None)


class HSI_Hiv_StartInfantProphylaxis(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event - start hiv prophylaxis for infants
    cotrim 6 mths + NVP/AZT 6-12 weeks
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Peds'] = 1  # This requires one outpatient appt
        the_appt_footprint['Under5OPD'] = 1  # general child outpatient appt

        # TODO: get the correct consumables listing cotrim + NVP/AZT
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] == 'PMTCT',
                                              'Intervention_Pkg_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Hiv_Infant_Prophylaxis'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug('This is HSI_Hiv_StartInfantProphylaxis: initiating treatment for person %d', person_id)

        df = self.sim.population.props

        df.at[person_id, 'hiv_on_cotrim'] = True
        df.at[person_id, 'hiv_on_art'] = True

        # schedule end date of cotrim after six months
        self.sim.schedule_event(HivCotrimEndEvent(self, person_id), self.sim.date + DateOffset(months=6))

        # schedule end date of ARVs after 6-12 weeks
        self.sim.schedule_event(HivARVEndEvent(self, person_id), self.sim.date + DateOffset(weeks=12))


class HivARVEndEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("Stopping ARVs for person %d", person_id)

        df = self.sim.population.props

        df.at[person_id, 'hiv_on_art'] = False


class HivCotrimEndEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("Stopping cotrim for person %d", person_id)

        df = self.sim.population.props

        df.at[person_id, 'hiv_on_cotrim'] = False


class HSI_Hiv_StartInfantTreatment(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event - start hiv treatment for infants + cotrim
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Peds'] = 1  # This requires one out patient appt
        the_appt_footprint['Under5OPD'] = 1  # hiv-specific appt type

        # TODO: get the correct consumables listing, ART + cotrim
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] == 'PMTCT', 'Intervention_Pkg_Code'])[0]
        pkg_code2 = pd.unique(consumables.loc[
                                  consumables[
                                      'Intervention_Pkg'] == 'Cotrimoxazole for children', 'Intervention_Pkg_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1, pkg_code2],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Hiv_Infant_Treatment_Initiation'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug('This is HSI_Hiv_StartInfantTreatment: initiating treatment for person %d', person_id)

        # ----------------------------------- ASSIGN ART ADHERENCE PROPERTIES -----------------------------------

        params = self.module.parameters
        df = self.sim.population.props

        df.at[person_id, 'hiv_on_cotrim'] = True

        if df.at[person_id, 'is_alive'] and \
            df.at[person_id, 'hiv_diagnosed'] and \
            (df.at[person_id, 'age_years'] < 15):
            df.at[person_id, 'hiv_on_art'] = self.module.rng.choice([1, 2],
                                                                    p=[(1 - params['vls_child']),
                                                                       params['vls_child']])

        df.at[person_id, 'hiv_date_art_start'] = self.sim.date

        # change specific_symptoms to 'none' if virally suppressed and adherent (hiv_on_art = 2)
        if df.at[person_id, 'hiv_on_art'] == 2:
            df.at[person_id, 'hiv_specific_symptoms'] = 'none'

        # ----------------------------------- SCHEDULE VL MONITORING -----------------------------------

        # Create follow-up appointments for VL monitoring
        times = params['VL_monitoring_times']

        logger.debug('....This is HSI_Hiv_StartTreatment: scheduling a follow-up appointment for person %d',
                     person_id)

        followup_appt = HSI_Hiv_TreatmentMonitoring(self.module, person_id=person_id)

        # Request the health system to have this follow-up appointment
        for i in range(0, len(times)):
            followup_appt_date = self.sim.date + DateOffset(months=times.time_months[i])
            self.sim.modules['HealthSystem'].schedule_event(followup_appt,
                                                            priority=2,
                                                            topen=followup_appt_date,
                                                            tclose=followup_appt_date + DateOffset(weeks=2)
                                                            )

        # ----------------------------------- SCHEDULE REPEAT PRESCRIPTIONS -----------------------------------

        date_repeat_prescription = self.sim.date + DateOffset(months=3)

        logger.debug(
            '....This is HSI_Hiv_StartTreatment: scheduling a repeat prescription for person %d on date %s',
            person_id, date_repeat_prescription)

        followup_appt = HSI_Hiv_RepeatPrescription(self.module, person_id=person_id)

        # Request the health system to have this follow-up appointment
        self.sim.modules['HealthSystem'].schedule_event(followup_appt,
                                                        priority=2,
                                                        topen=date_repeat_prescription,
                                                        tclose=date_repeat_prescription + DateOffset(weeks=2)
                                                        )

        # ----------------------------------- SCHEDULE COTRIM END -----------------------------------
        # schedule end date of cotrim after six months
        self.sim.schedule_event(HivCotrimEndEvent(self, person_id), self.sim.date + DateOffset(months=6))


# TODO: check if CD4 counts done routinely
class HSI_Hiv_StartTreatment(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event - start hiv treatment
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient appt
        the_appt_footprint['NewAdult'] = 1  # hiv-specific appt type

        # TODO: get the correct consumables listing
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] == 'First line treatment for new TB cases for adults', 'Intervention_Pkg_Code'])[
            0]
        pkg_code2 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] == 'HIV Testing Services', 'Intervention_Pkg_Code'])[
            0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1, pkg_code2],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Hiv_Treatment_Initiation'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):

        # ----------------------------------- ASSIGN ART ADHERENCE PROPERTIES -----------------------------------

        logger.debug('This is HSI_Hiv_StartTreatment: initiating treatment for person %d', person_id)

        params = self.module.parameters
        df = self.sim.population.props

        # condition: not already on art

        if df.at[person_id, 'is_alive'] and \
            df.at[person_id, 'hiv_diagnosed'] and \
            (df.at[person_id, 'age_years'] < 15) and \
            (df.at[person_id, 'hiv_on_art'] == 0):
            df.at[person_id, 'hiv_on_art'] = self.module.rng.choice([1, 2],
                                                                    p=[(1 - params['vls_child']),
                                                                       params['vls_child']])

        if df.at[person_id, 'is_alive'] and \
            df.at[person_id, 'hiv_diagnosed'] and \
            (df.at[person_id, 'age_years'] >= 15) and \
            (df.at[person_id, 'sex'] == 'M') and \
            (df.at[person_id, 'hiv_on_art'] == 0):
            df.at[person_id, 'hiv_on_art'] = self.module.rng.choice([1, 2],
                                                                    p=[(1 - params['vls_m']), params['vls_m']])

        if df.at[person_id, 'is_alive'] and \
            df.at[person_id, 'hiv_diagnosed'] and \
            (df.at[person_id, 'age_years'] >= 15) and \
            (df.at[person_id, 'sex'] == 'F') and \
            (df.at[person_id, 'hiv_on_art'] == 0):
            df.at[person_id, 'hiv_on_art'] = self.module.rng.choice([1, 2],
                                                                    p=[(1 - params['vls_f']), params['vls_f']])

        df.at[person_id, 'hiv_date_art_start'] = self.sim.date

        # change specific_symptoms to 'none' if virally suppressed and adherent (hiv_on_art = 2)
        if df.at[person_id, 'hiv_on_art'] == 2:
            df.at[person_id, 'hiv_specific_symptoms'] = 'none'
            df.at[person_id, 'hiv_unified_symptom_code'] = 1

        # ----------------------------------- SCHEDULE VL MONITORING -----------------------------------

        if not df.at[person_id, 'hiv_on_art'] == 0:

            # Create follow-up appointments for VL monitoring
            times = params['VL_monitoring_times']

            logger.debug('....This is HSI_Hiv_StartTreatment: scheduling a follow-up appointment for person %d',
                         person_id)

            followup_appt = HSI_Hiv_TreatmentMonitoring(self.module, person_id=person_id)

            # Request the health system to have this follow-up appointment
            for i in range(0, len(times)):
                followup_appt_date = self.sim.date + DateOffset(months=times.time_months[i])
                self.sim.modules['HealthSystem'].schedule_event(followup_appt,
                                                                priority=2,
                                                                topen=followup_appt_date,
                                                                tclose=followup_appt_date + DateOffset(weeks=2)
                                                                )

        # ----------------------------------- SCHEDULE REPEAT PRESCRIPTIONS -----------------------------------

        if not df.at[person_id, 'hiv_on_art'] == 0:
            date_repeat_prescription = self.sim.date + DateOffset(months=3)

            logger.debug(
                '....This is HSI_Hiv_StartTreatment: scheduling a repeat prescription for person %d on date %s',
                person_id, date_repeat_prescription)

            followup_appt = HSI_Hiv_RepeatPrescription(self.module, person_id=person_id)

            # Request the health system to have this follow-up appointment
            self.sim.modules['HealthSystem'].schedule_event(followup_appt,
                                                            priority=2,
                                                            topen=date_repeat_prescription,
                                                            tclose=date_repeat_prescription + DateOffset(weeks=2)
                                                            )


class HSI_Hiv_TreatmentMonitoring(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event for hiv viral load monitoring once on treatment
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one outpatient appt
        the_appt_footprint['EstNonCom'] = 1  # This is an hiv specific appt type

        # Get the consumables required
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] == 'Viral Load', 'Intervention_Pkg_Code'])[
            0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Hiv_TreatmentMonitoring'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug(
            '....This is Hiv_TreatmentMonitoring: giving a viral load test to person %d',
            person_id)


# TODO: find ART in consumables, how long is prescription for?
# schedule next Tx in 3 months
class HSI_Hiv_RepeatPrescription(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event for hiv repeat prescriptions once on treatment
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one outpatient appt
        the_appt_footprint['EstNonCom'] = 1  # This is an hiv specific appt type

        # TODO: get correct consumables listing for ART
        # Get the consumables required
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] == 'Viral Load', 'Intervention_Pkg_Code'])[
            0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Hiv_Treatment'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        date_repeat_prescription = self.sim.date + DateOffset(months=3)

        logger.debug(
            '....This is HSI_Hiv_RepeatPrescription: scheduling a repeat prescription for person %d on date %s',
            person_id, date_repeat_prescription)

        followup_appt = HSI_Hiv_RepeatPrescription(self.module, person_id=person_id)

        # Request the heathsystem to have this follow-up appointment
        self.sim.modules['HealthSystem'].schedule_event(followup_appt,
                                                        priority=2,
                                                        topen=date_repeat_prescription,
                                                        tclose=date_repeat_prescription + DateOffset(weeks=2)
                                                        )


# TODO: include hiv testing event as regular event for those not triggered by symptom change
# this could be an outreach event
# can include propensity for testing and schedule as HSI event


# class HivOutreachEvent(RegularEvent, PopulationScopeEventMixin):
#     def __init__(self, module):
#         super().__init__(module, frequency=DateOffset(months=12))
#
#     def apply(self, population):
#         # target adults age >15
#         df = population.props
#         mask_for_person_to_be_reached = (df.age_years >= 15)  # can put multiple conditions here
#
#         target = mask_for_person_to_be_reached.loc[df.is_alive]
#
#         # make and run the actual outreach event by the health system
#         outreach_event = healthsystem.OutreachEvent(self.module, disease_specific=self.module.name, target=target)
#
#         self.sim.schedule_event(outreach_event, self.sim.date)


# ---------------------------------------------------------------------------
#   Transitions to sex work
# ---------------------------------------------------------------------------

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
        # TODO: include marital status
        # check if any data to inform this
        # new fsw recruited to replace removed fsw -> constant proportion over time

        # current proportion of F 15-49 classified as fsw
        fsw = len(df[df.is_alive & (df.hiv_sexual_risk == 'sex_work')])
        eligible = len(df[df.is_alive & (df.sex == 'F') & (df.age_years.between(15, 49))])

        prop = fsw / eligible

        if prop < params['proportion_female_sex_workers']:
            # number new fsw needed
            recruit = int((params['proportion_female_sex_workers'] - prop) * eligible)
            fsw_new = df[df.is_alive & (df.sex == 'F') & (df.age_years.between(15, 49)) & (df.li_mar_stat == 2)].sample(
                n=recruit).index
            df.loc[fsw_new, 'hiv_sexual_risk'] = 'sex_work'


# ---------------------------------------------------------------------------
#   Scheduling deaths
# ---------------------------------------------------------------------------

class HivDeathEvent(Event, IndividualScopeEventMixin):
    """
    Performs the Death operation on an individual and logs it.
    """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)
        self.cause = cause

    def apply(self, individual_id):
        df = self.sim.population.props

        if df.at[individual_id, 'is_alive'] and (df.at[individual_id, 'hiv_on_art'] != 2):
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, cause='hiv'),
                                    self.sim.date)


# ---------------------------------------------------------------------------
#   Logging
# ---------------------------------------------------------------------------


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

        # adults incidence / prevalence
        mask = (df.loc[(df.age_years >= 15), 'hiv_date_inf'] > self.sim.date - DateOffset(months=self.repeat))
        adult_new_inf = mask.sum()
        # print('adult new hiv inf', adult_new_inf)

        ad_prev = len(df[df.hiv_inf & df.is_alive & (df.age_years.between(15, 65))]) / len(
            df[df.is_alive & (df.age_years.between(15, 65))])

        # children incidence / prevalence
        mask = (df.loc[(df.age_years < 15), 'hiv_date_inf'] > self.sim.date - DateOffset(months=self.repeat))
        child_new_inf = mask.sum()

        child_prev = len(df[df.hiv_inf & df.is_alive & (df.age_years.between(0, 14))]) / len(
            df[df.is_alive & (df.age_years.between(0, 14))])

        # deaths, this shows the deaths scheduled for this year, including those postponed due to ART
        date_aids_death = df.loc[df.hiv_inf & df.is_alive, 'hiv_date_death']
        year_aids_death = date_aids_death.dt.year
        sch_deaths = sum(1 for x in year_aids_death if int(x) == now.year)

        # on treatment, adults + children, good + poor adherence
        art = df.loc[(df.hiv_on_art == 2) | (df.hiv_on_art == 1), 'is_alive'].sum()

        logger.info('%s|summary|%s', self.sim.date,
                    {
                        'TotalInf': infected_total,
                        'hiv_prev_adult': ad_prev,
                        'hiv_prev_child': child_prev,
                        'hiv_new_infections_adult': adult_new_inf,
                        'hiv_new_infections_child': child_new_inf,
                        'hiv_scheduled_deaths': sch_deaths,
                        'hiv_on_art': art
                    })
