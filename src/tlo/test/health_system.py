"""
A skeleton template for disease methods.
"""
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent


class health_system(Module):
    """ routinely tests proportion of the population and
    determines availability of ART for HIV+ dependent on UNAIDS coverage estimates
    """

    def __init__(self, name=None, workbook_path=None):
        super().__init__(name)
        self.workbook_path = workbook_path
        self.store = {'Time': [], 'Number_tested': [], 'Number_treated': []}

    PARAMETERS = {
        'testing_coverage': Parameter(Types.REAL, 'proportion of population tested'),
        'art_coverage': Parameter(Types.DATA_FRAME, 'estimated ART coverage')
    }

    PROPERTIES = {
        'ever_tested': Property(Types.BOOL, 'ever had a hiv test'),
        'date_tested': Property(Types.DATE, 'date of hiv test'),
        'hiv_diagnosed': Property(Types.BOOL, 'hiv+ and tested'),
        'on_art': Property(Types.BOOL, 'on art'),
        'date_art_start': Property(Types.DATE, 'date art started')

    }

    def read_parameters(self, data_folder):
        params = self.parameters
        params['testing_coverage'] = 0.2  # dummy value
        params['art_coverage'] = 0.5  # dummy value

        self.parameters['initial_art'] = pd.read_excel(self.workbook_path,
                                                       sheet_name='coverage')

    def initialise_population(self, population):
        """ set the default values for the new fields
        """
        df = population.props

        df['ever_tested'] = False  # default: no individuals tested
        df['date_tested'] = pd.NaT
        df['hiv_diagnosed'] = False
        df['on_art'] = False
        df['date_art_start'] = pd.NaT

        self.baseline_art(population)  # allocate baseline art coverage

    def baseline_art(self, population):
        """ assign initial art coverage levels
        """
        now = self.sim.date
        df = population.props

        worksheet = self.parameters['initial_art']

        coverage = worksheet.loc[worksheet.year == now.year, ['year', 'single_age', 'sex', 'prop_coverage']]
        # print('coverage: ', coverage.head(20))

        # add age to population.props
        df_with_age = pd.merge(df, population.age, left_index=True, right_index=True, how='left')
        # print('df_with_age: ', df_with_age.head(10))

        # merge all susceptible individuals with their hiv probability based on sex and age
        df_with_age = df_with_age.merge(coverage,

                                                 left_on=['years', 'sex'],

                                                 right_on=['single_age', 'sex'],

                                                 how='left')

        # no data for ages 100+ so fill missing values with 0
        df_with_age['prop_coverage'] = df_with_age['prop_coverage'].fillna(0)
        # print('df_with_age_art_prob: ', df_with_age_art_prob.head(20))

        assert df_with_age.prop_coverage.isna().sum() == 0  # check there is a probability for every individual

        # get a list of random numbers between 0 and 1 for the whole population
        random_draw = self.sim.rng.random_sample(size=len(df_with_age))

        # probability of baseline population receiving art
        art_index = df_with_age.index[
            (random_draw < df_with_age.prop_coverage) & ~df_with_age.has_hiv & df.is_alive]
        print('art_index: ', art_index)

        # we don't know proportion tested but not treated at baseline, assume same proportion
        df.loc[art_index, 'ever_tested'] = True
        df.loc[art_index, 'date_tested'] = now
        df.loc[art_index, 'hiv_diagnosed'] = True
        df.loc[art_index, 'on_art'] = True
        df.loc[art_index, 'date_art_start'] = now


    def initialise_simulation(self, sim):
        sim.schedule_event(TestingEvent(self), sim.date + DateOffset(months=12))
        sim.schedule_event(TreatmentEvent(self), sim.date + DateOffset(months=12))

        # add an event to log to screen
        sim.schedule_event(HealthSystemLoggingEvent(self), sim.date + DateOffset(months=1))

    def on_birth(self, mother, child):
        pass


class TestingEvent(RegularEvent, PopulationScopeEventMixin):
    """A skeleton class for an event
    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module):
        """One line summary here

        """
        super().__init__(module, frequency=DateOffset(months=12))  # every 12 months
        # make sure any rates are annual if frequency of event is annual

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """

        params = self.module.parameters
        now = self.sim.date
        df = population.props

        # get a list of random numbers between 0 and 1 for the whole population
        random_draw = self.sim.rng.random_sample(size=len(df))

        # probability of HIV testing
        testing_index = df.index[(random_draw < params['testing_coverage']) & ~df.ever_tested & df.is_alive]
        testing_diagnosed_index = df.index[
            (random_draw < params['testing_coverage']) & ~df.ever_tested & df.is_alive & df.has_hiv]
        # print('testing_index: ', testing_index)
        # print('diagnosed_index: ', testing_diagnosed_index)

        df.loc[testing_index, 'ever_tested'] = True
        df.loc[testing_index, 'date_tested'] = now
        df.loc[testing_diagnosed_index, 'hiv_diagnosed'] = True


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

        # probability of HIV treatment
        treatment_index = df.index[(random_draw < params['art_coverage']) & df.has_hiv & df.hiv_diagnosed &
                                   df.is_alive & ~df.on_art]
        # print('treatment_index: ', treatment_index)

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

        mask = (df['date_tested'] > self.sim.date - DateOffset(months=self.repeat))
        recently_tested = mask.sum()

        currently_on_art = len(df[df.on_art & df.is_alive])

        self.module.store['Time'].append(self.sim.date)
        self.module.store['Number_tested'].append(recently_tested)
        self.module.store['Number_treated'].append(currently_on_art)
