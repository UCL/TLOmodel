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
        'hiv_diagnosed': Property(Types.BOOL, 'hiv+ and tested')
    }

    def read_parameters(self, data_folder):
        params = self.parameters
        params['testing_coverage'] = 0.2  # dummy value

        # TODO: check the sheet name
        # self.parameters['art_coverage'] = pd.read_excel(self.workbook_path,
        #                                                 sheet_name='art_coverage')

    def initialise_population(self, population):
        """ set the default values for the new fields
        """
        df = population.props

        df['ever_tested'] = False  # default: no individuals tested
        df['date_tested'] = pd.NaT
        df['hiv_diagnosed'] = False


    def initialise_simulation(self, sim):

        sim.schedule_event(TestingEvent(self), sim.date + DateOffset(months=12))

        # add an event to log to screen
        sim.schedule_event(HealthSystemLoggingEvent(self), sim.date + DateOffset(months=12))


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
        print('testing_index: ', testing_index)
        print('diagnosed_index: ', testing_diagnosed_index)

        df.loc[testing_index, 'ever_tested'] = True
        df.loc[testing_index, 'date_tested'] = now
        df.loc[testing_diagnosed_index, 'hiv_diagnosed'] = True


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

        self.module.store['Time'].append(self.sim.date)
        self.module.store['Number_tested'].append(recently_tested)

