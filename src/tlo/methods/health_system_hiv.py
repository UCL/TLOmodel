"""
A skeleton template for disease methods.
"""
import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent, IndividualScopeEventMixin


class health_system(Module):
    """ routinely tests proportion of the population and
    determines availability of ART for HIV+ dependent on UNAIDS coverage estimates
    """

    def __init__(self, name=None, workbook_path=None, par_est1=None, par_est2=None, par_est3=None, par_est4=None):
        super().__init__(name)
        self.workbook_path = workbook_path
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
        'testing_baseline_child': Parameter(Types.REAL, 'baseline testing rate of children')
    }

    PROPERTIES = {
        'ever_tested': Property(Types.BOOL, 'ever had a hiv test'),
        'date_tested': Property(Types.DATE, 'date of hiv test'),
        'number_hiv_tests': Property(Types.INT, 'number of hiv tests taken'),
        'hiv_diagnosed': Property(Types.BOOL, 'hiv+ and tested'),
        'on_art': Property(Types.BOOL, 'on art'),
        'date_art_start': Property(Types.DATE, 'date art started')
    }

    def read_parameters(self, data_folder):
        params = self.parameters
        params['param_list'] = pd.read_excel(self.workbook_path,
                                             sheet_name='parameters')

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

        self.parameters['initial_art_coverage'] = pd.read_excel(self.workbook_path,
                                                                sheet_name='coverage')

        # self.parameters['art_initiation'] = pd.read_excel(self.workbook_path,
        #                                                   sheet_name='art_initiators')

        params['testing_baseline_adult'] = float(self.testing_baseline_adult)
        params['testing_baseline_child'] = float(self.testing_baseline_child)

        params['treatment_baseline_adult'] = float(self.treatment_baseline_adult)
        params['treatment_baseline_child'] = float(self.treatment_baseline_child)

        # print('testing_baseline_adult', params['testing_baseline_adult'])

    def initialise_population(self, population):
        """ set the default values for the new fields
        """
        df = population.props

        df['ever_tested'] = False  # default: no individuals tested
        df['date_tested'] = pd.NaT
        df['number_hiv_tests'] = 0
        df['hiv_diagnosed'] = False
        df['on_art'] = False
        df['date_art_start'] = pd.NaT

        self.baseline_tested(population)  # allocate baseline art coverage
        self.baseline_art(population)  # allocate baseline art coverage

    def baseline_tested(self, population):
        """ assign initial art coverage levels
        """
        now = self.sim.date
        df = population.props

        # add age to population.props
        # df_age = pd.merge(df, population.age, left_index=True, right_index=True, how='left')

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
        df.loc[art_index_male | art_index_female, 'ever_tested'] = True
        df.loc[art_index_male | art_index_female, 'date_tested'] = now

        # outcome of test
        diagnosed_idx = df.index[df.ever_tested & df.is_alive & df.has_hiv]
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

        # probability of baseline population receiving art: requirement = ever_tested
        art_index = df_art.index[
            (random_draw < df_art.prop_coverage) & df_art.has_hiv & df.ever_tested & df.is_alive]
        # print('art_index: ', art_index)

        df.loc[art_index, 'on_art'] = True
        df.loc[art_index, 'date_art_start'] = now

    def initialise_simulation(self, sim):
        sim.schedule_event(TestingEvent(self), sim.date + DateOffset(months=12))
        sim.schedule_event(TreatmentEvent(self), sim.date + DateOffset(months=12))

        # add an event to log to screen
        sim.schedule_event(HealthSystemLoggingEvent(self), sim.date + DateOffset(months=12))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        """
        df = self.sim.population.props

        df.at[child_id, 'ever_tested'] = False
        df.at[child_id, 'date_tested'] = pd.NaT
        df.at[child_id, 'number_hiv_tests'] = 0
        df.at[child_id, 'hiv_diagnosed'] = False
        df.at[child_id, 'on_art'] = False
        df.at[child_id, 'date_art_start'] = pd.NaT


class TestingEvent(RegularEvent, PopulationScopeEventMixin):
    """ applies to whole population
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))  # every 12 months
        # make sure any rates are annual if frequency of event is annual

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date
        df = population.props

        # get a list of random numbers between 0 and 1 for the whole population
        random_draw = self.sim.rng.random_sample(size=len(df))

        # assign relative testing rates
        eff_testing = pd.Series(0, index=df.index)

        eff_testing.loc[(df.age_years >= 15) & df.is_alive] = params['testing_baseline_adult']  # baseline adult
        eff_testing.loc[(df.age_years >= 25)] *= params['rr_testing_age25']  # for ages >= 25
        eff_testing.loc[(df.sex == 'F')] *= params['rr_testing_female']  # for females
        eff_testing.loc[df.ever_tested & ~df.hiv_diagnosed] *= params[
            'rr_testing_previously_negative']  # tested, previously negative
        eff_testing.loc[df.ever_tested & df.hiv_diagnosed] *= params[
            'rr_testing_previously_positive']  # tested, previously positive
        eff_testing.loc[(df.sexual_risk_group == 'high') | (df.sexual_risk_group == 'sex_work')] *= params[
            'rr_testing_high_risk']  # for high risk

        eff_testing.loc[(df.age_years < 15) & df.is_alive] = params['testing_baseline_child']  # baseline children

        # probability of HIV testing, can allow repeat testing
        # if repeat testing, will only store latest test date
        testing_index = df.index[(random_draw < eff_testing) & df.is_alive]
        # print('testing index', testing_index)

        df.loc[testing_index, 'ever_tested'] = True
        df.loc[testing_index, 'date_tested'] = now
        df.loc[testing_index, 'number_hiv_tests'] = df.loc[testing_index, 'number_hiv_tests'] + 1

        diagnosed_index = df.index[(df.date_tested == now) & df.is_alive & df.has_hiv]
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

        if df.at[individual_id.is_alive & ~individual_id.hiv_diagnosed]:
            # probability of HIV testing
            df.at[individual_id, 'ever_tested'] = np.random.choice([True, False], size=1,
                                                                   p=[params['testing_prob_individual'],
                                                                      1 - params['testing_prob_individual']])

            df.at[individual_id, 'date_tested'] = self.sim.date

            df.at[individual_id.ever_tested & individual_id.is_alive & individual_id.has_hiv, 'hiv_diagnosed'] = True


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

        # get a list of random numbers between 0 and 1 for the whole population
        random_draw = self.sim.rng.random_sample(size=len(df))

        # assign treatment rates
        treatment_rate = pd.Series(0, index=df.index)

        treatment_rate.loc[(df.age_years >= 15) & df.is_alive] = params['treatment_baseline_adult']  # baseline adult
        treatment_rate.loc[(df.age_years < 15) & df.is_alive] = params['treatment_baseline_child']  # baseline adult

        # probability of treatment
        # if repeat testing, will only store latest test date
        treatment_index = df.index[(random_draw < treatment_rate) & df.is_alive & ~df.on_art]
        # print('treatment_index', treatment_index)

        df.loc[treatment_index, 'on_art'] = True
        df.loc[treatment_index, 'date_art_start'] = now


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

        mask = (df.loc[(df.age_years >= 15), 'date_tested'] > self.sim.date - DateOffset(months=self.repeat))
        recently_tested_adult = mask.sum()

        mask = (df.loc[(df.age_years < 15), 'date_tested'] > self.sim.date - DateOffset(months=self.repeat))
        recently_tested_child = mask.sum()

        mask = (df.loc[(df.age_years >= 15), 'date_art_start'] > self.sim.date - DateOffset(months=self.repeat))
        recently_treated_adult = mask.sum()

        mask = (df.loc[(df.age_years < 15), 'date_art_start'] > self.sim.date - DateOffset(months=self.repeat))
        recently_treated_child = mask.sum()

        currently_on_art = len(df[df.on_art & df.is_alive])

        self.module.store['Time'].append(self.sim.date)
        self.module.store['Number_tested_adult'].append(recently_tested_adult)
        self.module.store['Number_tested_child'].append(recently_tested_child)
        self.module.store['Number_treated_adult'].append(recently_treated_adult)
        self.module.store['Number_treated_child'].append(recently_treated_child)
