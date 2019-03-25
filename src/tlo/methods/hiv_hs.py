"""
A skeleton template for disease methods.
"""

import os

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent, IndividualScopeEventMixin


def create_testing_rates(self):
    years = list(range(2011, 2030))
    curr_testing_rate_adult = [None] * len(years)
    curr_testing_rate_adult[0] = self.parameters['testing_baseline_adult']

    curr_testing_rate_child = [None] * len(years)
    curr_testing_rate_child[0] = self.parameters['testing_baseline_child']

    for ind in range(1, len(years)):
        curr_testing_rate_adult[ind] = curr_testing_rate_adult[ind - 1] + self.parameters['testing_increase']
        curr_testing_rate_child[ind] = curr_testing_rate_child[ind - 1] + self.parameters['testing_increase']

    data = {'year': years, 'testing_rates_adult': curr_testing_rate_adult,
            'testing_rates_child': curr_testing_rate_child}
    testing_rate_df = pd.DataFrame(data)

    return testing_rate_df


class health_system(Module):
    """ routinely tests proportion of the population and
    determines need for ART in HIV+ pop
    """

    def __init__(self, name=None, resourcefilepath=None, par_est1=None, par_est2=None, par_est3=None, par_est4=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.testing_baseline_adult = par_est1
        self.testing_baseline_child = par_est2
        self.treatment_baseline_adult = par_est3
        self.treatment_baseline_child = par_est4

        self.store = {'Time': [], 'Number_tested_adult': [], 'Number_tested_child': [], 'Number_treated_adult': [],
                      'Number_treated_child': []}

    PARAMETERS = {
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
        'testing_baseline_adult': Parameter(Types.REAL, 'baseline testing rate of adults'),
        'testing_baseline_child': Parameter(Types.REAL, 'baseline testing rate of children'),
        'treatment_increase2016': Parameter(Types.REAL,
                                            'increase in treatment rates with eligibility guideline changes'),
        'VL_monitoring_times': Parameter(Types.INT, 'times(months) viral load monitoring required after ART start'),
        'vls_m': Parameter(Types.INT, 'rates of viral load suppression males'),
        'vls_f': Parameter(Types.INT, 'rates of viral load suppression males'),
        'vls_child': Parameter(Types.INT, 'rates of viral load suppression in children 0-14 years')
    }

    PROPERTIES = {
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

    def read_parameters(self, data_folder):

        workbook = pd.read_excel(os.path.join(self.resourcefilepath,
                                              'Method_ART.xlsx'), sheet_name=None)

        params = self.parameters
        params['param_list'] = workbook['parameters']

        self.param_list.set_index("Parameter", inplace=True)

        params['testing_coverage_male'] = self.param_list.loc['testing_coverage_male_2010', 'Value1']
        params['testing_coverage_female'] = self.param_list.loc['testing_coverage_female_2010', 'Value1']
        params['testing_prob_individual'] = self.param_list.loc['testing_prob_individual', 'Value1']  # dummy value
        params['art_coverage'] = self.param_list.loc['art_coverage', 'Value1']

        params['rr_testing_high_risk'] = self.param_list.loc['rr_testing_high_risk', 'Value1']
        params['rr_testing_female'] = self.param_list.loc['rr_testing_female', 'Value1']
        params['rr_testing_previously_negative'] = self.param_list.loc['rr_testing_previously_negative', 'Value1']
        params['rr_testing_previously_positive'] = self.param_list.loc['rr_testing_previously_positive', 'Value1']
        params['rr_testing_age25'] = self.param_list.loc['rr_testing_age25', 'Value1']
        params['testing_increase'] = self.param_list.loc['testing_increase', 'Value1']
        params['treatment_increase2016'] = self.param_list.loc['treatment_increase2016', 'Value1']
        params['vls_m'] = self.param_list.loc['vls_m', 'Value1']
        params['vls_f'] = self.param_list.loc['vls_f', 'Value1']
        params['vls_child'] = self.param_list.loc['vls_child', 'Value1']

        self.parameters['initial_art_coverage'] = workbook['coverage']

        self.parameters['VL_monitoring_times'] = workbook['VL_monitoring']

        params['testing_baseline_adult'] = float(self.testing_baseline_adult)
        params['testing_baseline_child'] = float(self.testing_baseline_child)

        params['treatment_baseline_adult'] = float(self.treatment_baseline_adult)
        params['treatment_baseline_child'] = float(self.treatment_baseline_child)

        # print('testing_baseline_adult', params['testing_baseline_adult'])

        self.parameters['testing_rate_df'] = create_testing_rates(self)
        # print(self.parameters['testing_rate_df'])

    def initialise_population(self, population):
        """ set the default values for the new fields
        """
        df = population.props

        df['hiv_ever_tested'] = False  # default: no individuals tested
        df['hiv_date_tested'] = pd.NaT
        df['hiv_number_tests'] = 0
        df['hiv_diagnosed'] = False
        df['hiv_on_art'].values[:] = '0'
        df['hiv_date_art_start'] = pd.NaT
        df['hiv_viral_load_test'] = pd.NaT
        df['hiv_on_cotrim'] = False
        df['hiv_date_cotrim'] = pd.NaT

        self.baseline_tested(population)  # allocate baseline art coverage
        self.baseline_art(population)  # allocate baseline art coverage

    def baseline_tested(self, population):
        """ assign initial art coverage levels
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

        df.loc[art_idx_child, 'hiv_on_art'] = '2'  # assumes all are adherent at baseline
        df.loc[art_idx_child, 'hiv_date_art_start'] = now

        art_idx_adult = df_art.index[
            (random_draw < df_art.prop_coverage) & df.is_alive & df_art.hiv_inf & df.hiv_diagnosed &
            df_art.age_years.between(15, 64)]

        df.loc[art_idx_adult, 'hiv_on_art'] = '2'  # assumes all are adherent, then stratify into category 1/2
        df.loc[art_idx_adult, 'hiv_date_art_start'] = now

        # allocate proportion to non-adherent category
        # if condition added, error with small numbers of children to sample
        if len(df[df.is_alive & (df.hiv_on_art == '2') & (df.age_years.between(0, 14))]) > 5:
            idx_c = df[df.is_alive & (df.hiv_on_art == '2') & (df.age_years.between(0, 14))].sample(
                frac=(1 - self.parameters['vls_child'])).index
            df.loc[idx_c, 'hiv_on_art'] = '1'  # change to non=adherent

        idx_m = df[df.is_alive & (df.hiv_on_art == '2') & (df.sex == 'M') & (df.age_years.between(15, 64))].sample(
            frac=(1 - self.parameters['vls_m'])).index
        df.loc[idx_m, 'hiv_on_art'] = '1'  # change to non=adherent

        idx_f = df[df.is_alive & (df.hiv_on_art == '2') & (df.sex == 'F') & (df.age_years.between(15, 64))].sample(
            frac=(1 - self.parameters['vls_f'])).index
        df.loc[idx_f, 'hiv_on_art'] = '1'  # change to non=adherent

    def initialise_simulation(self, sim):
        sim.schedule_event(TestingEvent(self), sim.date + DateOffset(months=12))
        sim.schedule_event(TreatmentEvent(self), sim.date + DateOffset(months=12))

        sim.schedule_event(ClinMonitoringEvent(self), sim.date + DateOffset(months=1))
        sim.schedule_event(CotrimoxazoleEvent(self), sim.date + DateOffset(months=12))

        # add an event to log to screen
        sim.schedule_event(HealthSystemLoggingEvent(self), sim.date + DateOffset(months=1))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        """
        df = self.sim.population.props

        df.at[child_id, 'hiv_ever_tested'] = False
        df.at[child_id, 'hiv_date_tested'] = pd.NaT
        df.at[child_id, 'hiv_number_tests'] = 0
        df.at[child_id, 'hiv_diagnosed'] = False
        df.at[child_id, 'hiv_on_art'] = '0'
        df.at[child_id, 'hiv_date_art_start'] = pd.NaT
        df.at[child_id, 'hiv_viral_load_test'] = pd.NaT
        df.at[child_id, 'hiv_on_cotrim'] = False
        df.at[child_id, 'hiv_date_cotrim'] = pd.NaT


class TestingEvent(RegularEvent, PopulationScopeEventMixin):
    """ applies to whole population
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=3))  # every 3 months
        # make sure any rates are annual if frequency of event is annual

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date
        df = population.props

        worksheet = params['testing_rate_df']

        # use iat[0] because the rate is a pd.series value, need to convert to scalar (numpy float)
        current_testing_rate_adult = worksheet.loc[worksheet.year == now.year, 'testing_rates_adult'].iat[0]
        current_testing_rate_child = worksheet.loc[worksheet.year == now.year, 'testing_rates_child'].iat[0]

        # print(current_testing_rate_adult)
        # print(type(current_testing_rate_adult))

        # get a list of random numbers between 0 and 1 for the whole population
        random_draw = self.sim.rng.random_sample(size=len(df))

        # assign relative testing rates
        eff_testing = pd.Series(0, index=df.index)

        eff_testing.loc[(df.age_years >= 15) & df.is_alive] = current_testing_rate_adult
        eff_testing.loc[(df.age_years >= 25)] *= params['rr_testing_age25']  # for ages >= 25
        eff_testing.loc[(df.sex == 'F')] *= params['rr_testing_female']  # for females
        eff_testing.loc[df.hiv_ever_tested & ~df.hiv_diagnosed] *= params[
            'rr_testing_previously_negative']  # tested, previously negative
        eff_testing.loc[df.hiv_ever_tested & df.hiv_diagnosed] *= params[
            'rr_testing_previously_positive']  # tested, previously positive
        eff_testing.loc[(df.hiv_sexual_risk == 'sex_work')] *= params[
            'rr_testing_high_risk']  # for high risk

        eff_testing.loc[(df.age_years < 15) & df.is_alive] = current_testing_rate_child

        # probability of HIV testing, can allow repeat testing
        # if repeat testing, will only store latest test date
        testing_index = df.index[(random_draw < eff_testing) & df.is_alive]
        # print('testing index', testing_index)

        df.loc[testing_index, 'hiv_ever_tested'] = True
        df.loc[testing_index, 'hiv_date_tested'] = now
        df.loc[testing_index, 'hiv_number_tests'] += 1

        diagnosed_index = df.index[(df.hiv_date_tested == now) & df.is_alive & df.hiv_inf]
        # print('diagnosed_index: ', diagnosed_index)

        df.loc[diagnosed_index, 'hiv_diagnosed'] = True

        # TODO: include infant testing (soon after birth? and linked with birth facility?)


# pregnant women can trigger a testing event
# all pregnant women will request hiv test, total number available fixed
class IndividualTesting(Event, IndividualScopeEventMixin):
    """ allows hiv tests for individuals triggered by certain events
    e.g. pregnancy or tb diagnosis
    """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        params = self.module.parameters
        df = self.sim.population.props

        if df.at[individual_id, 'is_alive'] and not df.at[individual_id, 'df.hiv_diagnosed'] & np.random.choice(
            [True, False], size=1,
            p=[params['testing_prob_individual'],
               1 - params['testing_prob_individual']]):

            # probability of HIV testing
            df.at[individual_id, 'hiv_ever_tested'] = True
            df.at[individual_id, 'hiv_date_tested'] = self.sim.date

            if df.at[individual_id, df.hiv_inf]:
                df.at[individual_id, 'hiv_diagnosed'] = True


class TreatmentEvent(RegularEvent, PopulationScopeEventMixin):
    """ assigning ART to diagnosed HIV+ people
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))  # every 12 months
        # make sure any rates are annual if frequency of event is annual

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date
        df = population.props

        curr_treatment_adult = params['treatment_baseline_adult']
        curr_treatment_child = params['treatment_baseline_child']

        if now.year >= 2016:
            curr_treatment_adult = params['treatment_baseline_adult'] * params['treatment_increase2016']
            curr_treatment_child = params['treatment_baseline_child'] * params['treatment_increase2016']

        # print(curr_treatment_adult)

        # get a list of random numbers between 0 and 1 for the whole population
        random_draw = self.sim.rng.random_sample(size=len(df))

        # assign treatment rates
        treatment_rate = pd.Series(0, index=df.index)

        treatment_rate.loc[(df.age_years >= 15) & df.is_alive] = curr_treatment_adult
        treatment_rate.loc[(df.age_years < 15) & df.is_alive] = curr_treatment_child

        # probability of treatment
        treat_idx = df.index[
            (random_draw < treatment_rate) & df.is_alive & (df.hiv_on_art == '0') & df.hiv_diagnosed]
        # print('treatment_index', treatment_index)

        # assign all as good adherence then select the poor adherence proportion m/f
        df.loc[treat_idx, 'hiv_on_art'] = '2'
        df.loc[treat_idx, 'hiv_date_art_start'] = now

        poor_adh_c = df[(df.hiv_date_art_start == now) & (df.age_years.between(0, 15))].sample(
            frac=(1 - params['vls_child'])).index
        df.loc[poor_adh_c, 'hiv_on_art'] = '1'

        poor_adh_m = df[(df.hiv_date_art_start == now) & (df.sex == 'M') & (df.age_years.between(15, 64))].sample(
            frac=(1 - params['vls_m'])).index
        df.loc[poor_adh_m, 'hiv_on_art'] = '1'

        poor_adh_f = df[(df.hiv_date_art_start == now) & (df.sex == 'F') & (df.age_years.between(1, 64))].sample(
            frac=(1 - params['vls_f'])).index
        df.loc[poor_adh_f, 'hiv_on_art'] = '1'


class ClinMonitoringEvent(RegularEvent, PopulationScopeEventMixin):
    """ viral load testing for monitoring people on ART
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # every 1 month

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date
        df = population.props

        vl_times = list(params['VL_monitoring_times'].time_months)
        # print(vl_times)

        # subset pop on ART
        df_art = df[(df.hiv_on_art == '2') & df.is_alive]

        # extract time on art
        time_on_art = (now - df_art['hiv_date_art_start']) / np.timedelta64(1, 'M')
        time_on_art2 = time_on_art.astype(int)

        # request for viral load
        vl_needed = time_on_art2.isin(vl_times)

        # allocate viral load testing using coin flip - later will be linked with hs resources module
        # allocate vl if requested and coin toss is true
        vl_allocated = np.random.choice([True, False], size=len(df_art), p=[0.5, 0.5])

        vl_index = df_art.index[vl_needed & vl_allocated]
        # print('vl_index', vl_index)

        df.loc[vl_index, 'hiv_viral_load_test'] = now


class CotrimoxazoleEvent(RegularEvent, PopulationScopeEventMixin):
    """ cotrimoxazole prophylaxis for HIV-exposed infants and HIV+ people (all ages)
    prioritise if limited resources
    only for people on art and taking it well
    then it will be lifelong
    """

    # TODO: if none allocated to a group can throw errors, include more checks

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # every 1 month

    def apply(self, population):
        now = self.sim.date
        df = population.props

        # 1. HIV-exposed infants, from 4 weeks to 18 months
        df_inf = df[
            df.is_alive & ~df.hiv_inf & df.hiv_mother_inf & ~df.hiv_on_cotrim & (df.age_exact_years > 0.083) & (
                df.age_exact_years < 1.5)]

        # request for cotrim, dependent on access to health facility / demographic characteristics?
        cotrim_needed = pd.Series(np.random.choice([True, False], size=len(df_inf), p=[1, 0]),
                                  index=df_inf)

        # allocate cotrim using coin flip - later will be linked with hs resources module
        # allocate cotrim if requested and coin toss is true
        cotrim_allocated = pd.Series(np.random.choice([True, False], size=len(df_inf), p=[1, 0.]),
                                     index=df_inf)
        # print('df_inf', df_inf.head(30))
        z = [a and b for a, b in zip(cotrim_needed, cotrim_allocated)]
        if len(z):
            ct_inf_index = df_inf.index[z]
            # print('ct_inf_index', ct_inf_index)
            df.loc[ct_inf_index, 'hiv_on_cotrim'] = True
            df.loc[ct_inf_index, 'hiv_date_cotrim'] = now

        # 2. HIV+ children <15 years
        df_child = df[
            df.is_alive & df.hiv_inf & ~df.hiv_on_cotrim & (df.age_exact_years >= 1.5) & (df.age_exact_years < 15)]
        cotrim_needed = pd.Series(np.random.choice([True, False], size=len(df_child), p=[1, 0]),
                                  index=df_child)
        cotrim_allocated = pd.Series(
            np.random.choice([True, False], size=len(df_child), p=[1, 0]), index=df_child)

        z = [a and b for a, b in zip(cotrim_needed, cotrim_allocated)]
        if len(z):
            ct_child_index = df_child.index[z]
            df.loc[ct_child_index, 'hiv_on_cotrim'] = True
            df.loc[ct_child_index, 'hiv_date_cotrim'] = now

        # 3. TB/HIV+ adults
        df_coinf = df[df.is_alive & df.hiv_inf & (df.tb_inf == 'Active') & ~df.hiv_on_cotrim & (df.age_years >= 15)]
        cotrim_needed = pd.Series(np.random.choice([True, False], size=len(df_coinf), p=[1, 0]),
                                  index=df_coinf)
        cotrim_allocated = pd.Series(
            np.random.choice([True, False], size=len(df_coinf), p=[1, 0]), index=df_coinf)

        z = [a and b for a, b in zip(cotrim_needed, cotrim_allocated)]
        if len(z):
            ct_coinf_index = df_coinf.index[z]
            df.loc[ct_coinf_index, 'hiv_on_cotrim'] = True
            df.loc[ct_coinf_index, 'hiv_date_cotrim'] = now

        # 4. pregnant women with HIV
        df_preg = df[df.is_alive & df.hiv_inf & df.is_pregnant & ~df.hiv_on_cotrim & (df.age_years >= 15)]
        cotrim_needed = pd.Series(np.random.choice([True, False], size=len(df_preg), p=[1, 0]),
                                  index=df_preg)
        cotrim_allocated = pd.Series(np.random.choice([True, False], size=len(df_preg), p=[1, 0]),
                                     index=df_preg)
        z = [a and b for a, b in zip(cotrim_needed, cotrim_allocated)]
        if len(z):
            ct_preg_index = df_preg.index[z]
            df.loc[ct_preg_index, 'hiv_on_cotrim'] = True
            df.loc[ct_preg_index, 'hiv_date_cotrim'] = now

        # 5. all adults with HIV
        df_adult = df[df.is_alive & df.hiv_inf & ~df.hiv_on_cotrim & (df.age_years >= 15)]
        cotrim_needed = pd.Series(np.random.choice([True, False], size=len(df_adult), p=[1, 0]),
                                  index=df_adult)
        cotrim_allocated = pd.Series(
            np.random.choice([True, False], size=len(df_adult), p=[1, 0]), index=df_adult)
        z = [a and b for a, b in zip(cotrim_needed, cotrim_allocated)]
        if len(z):
            ct_adult_index = df_adult.index[z]
            df.loc[ct_adult_index, 'hiv_on_cotrim'] = True
            df.loc[ct_adult_index, 'hiv_date_cotrim'] = now


class HealthSystemLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """ produce some outputs to check
        """
        # run this event every 12 months (every year)
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        mask = (df.loc[(df.age_years >= 15), 'hiv_date_tested'] > self.sim.date - DateOffset(months=self.repeat))
        recently_tested_adult = mask.sum()

        mask = (df.loc[(df.age_years < 15), 'hiv_date_tested'] > self.sim.date - DateOffset(months=self.repeat))
        recently_tested_child = mask.sum()

        mask = (df.loc[(df.age_years >= 15), 'hiv_date_art_start'] > self.sim.date - DateOffset(months=self.repeat))
        recently_treated_adult = mask.sum()

        mask = (df.loc[(df.age_years < 15), 'hiv_date_art_start'] > self.sim.date - DateOffset(months=self.repeat))
        recently_treated_child = mask.sum()

        currently_on_art = len(df[(df.hiv_on_art == '2') & df.is_alive])

        self.module.store['Time'].append(self.sim.date)
        self.module.store['Number_tested_adult'].append(recently_tested_adult)
        self.module.store['Number_tested_child'].append(recently_tested_child)
        self.module.store['Number_treated_adult'].append(recently_treated_adult)
        self.module.store['Number_treated_child'].append(recently_treated_child)
