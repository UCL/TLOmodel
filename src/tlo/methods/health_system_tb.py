"""
A skeleton template for disease methods.
"""
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent


class health_system_tb(Module):
    """ routinely tests proportion of the population and
    determines availability of ART for HIV+ dependent on UNAIDS coverage estimates
    """

    def __init__(self, name=None):
        super().__init__(name)
        self.store = {'Time': [], 'Number_tested_tb': []}

    PARAMETERS = {
        'tb_testing_coverage': Parameter(Types.REAL, 'proportion of population tested')
    }

    PROPERTIES = {
        'tb_ever_tested': Property(Types.BOOL, 'ever had a tb test'),
        'tb_date_tested': Property(Types.DATE, 'date of tb test'),
        'tb_diagnosed': Property(Types.BOOL, 'active tb and tested')
    }

    def read_parameters(self, data_folder):
        params = self.parameters
        params['tb_testing_coverage'] = 0.1  # dummy value


    def initialise_population(self, population):
        """ set the default values for the new fields
        """
        df = population.props

        df['tb_ever_tested'] = False  # default: no individuals tested
        df['tb_date_tested'] = pd.NaT
        df['tb_diagnosed'] = False



    def initialise_simulation(self, sim):
        sim.schedule_event(TbTestingEvent(self), sim.date + DateOffset(months=12))

        # add an event to log to screen
        sim.schedule_event(TbHealthSystemLoggingEvent(self), sim.date + DateOffset(months=1))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        """
        df = self.sim.population.props

        df.at[child_id, 'tb_ever_tested'] = False
        df.at[child_id, 'tb_date_tested'] = pd.NaT
        df.at[child_id, 'tb_diagnosed'] = False


class TbTestingEvent(RegularEvent, PopulationScopeEventMixin):
    """ Testing for TB
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

        # probability of TB testing
        testing_index = df.index[(random_draw < params['tb_testing_coverage']) & ~df.tb_ever_tested & df.is_alive]
        df.loc[testing_index, 'tb_ever_tested'] = True
        df.loc[testing_index, 'tb_date_tested'] = now

        diagnosed_index = df.index[df.tb_ever_tested & df.is_alive & (df.has_tb == 'Active')]
        df.loc[diagnosed_index, 'tb_diagnosed'] = True




class TbHealthSystemLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """ produce some outputs to check
        """
        # run this event every 12 months (every year)
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        mask = (df['tb_date_tested'] > self.sim.date - DateOffset(months=self.repeat))
        recently_tested = mask.sum()

        self.module.store['Time'].append(self.sim.date)
        self.module.store['Number_tested_tb'].append(recently_tested)
