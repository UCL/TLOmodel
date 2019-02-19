"""
Assigning ART to HIV+
"""
import pandas as pd

from tlo import DateOffset, Module, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography


# this class contains all the methods required to set up the baseline population
class art(Module):
    """ Models ART initiation.
    """

    def __init__(self, name=None, workbook_path=None):
        super().__init__(name)
        self.workbook_path = workbook_path
        self.store = {'Time': [], 'Number_dead_art': []}

    # PARAMETERS = {}

    PROPERTIES = {
        'art_mortality_rate': Property(Types.REAL, 'Mortality rates whilst on art'),
        'early_art': Property(Types.BOOL, 'If art was started >2 years from scheduled death'),
        'time_on_art': Property(Types.CATEGORICAL, 'length of time on art',
                                categories=['0_6months', '7_12months', '12months']),
    }

    def read_parameters(self, data_folder):
        """ Read parameter values from file
        """
        # values early / late starters
        self.parameters['art_mortality_data'] = pd.read_excel(self.workbook_path,
                                                              sheet_name='mortality_rates_long')

        # values by CD4 state
        self.parameters['initial_art_mortality'] = pd.read_excel(self.workbook_path,
                                                                 sheet_name='mortality_CD4')

    def initialise_population(self, population):
        """ assign mortality rates on ART for baseline population by cd4 state
        """

        df = population.props

        df.art_mortality_rate = None  # default no values
        df.art_early = None
        df.time_on_art = None

        worksheet = self.parameters['initial_art_mortality']

        # assume all treated for at least 12 months
        mort_rates = worksheet.loc[worksheet.time_on_treatment == '12months+', ['state', 'age', 'sex', 'mortality']]
        # print(mort_rates)

        # merge the mortality rates by age, sex and cd4 state, all ages
        df_mort = pd.merge(df, mort_rates, left_on=['age_years', 'sex', 'cd4_state'],
                           right_on=['age', 'sex', 'state'], how='left')
        # print('df_mort: ', df_mort.head(20))
        # df_mort.to_csv('P:/Documents/TLO/test.csv', sep=',')

        # retrieve index to assign mortality rates
        idx = df_mort.index[df_mort.on_art]
        df.loc[idx, 'art_mortality_rate'] = df_mort.loc[idx, 'mortality']
        # print(idx)

    def initialise_simulation(self, sim):
        """ Get ready for simulation start.
        """
        event = ArtMortalityEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=1))

        sim.schedule_event(HivArtDeathEvent(self), sim.date + DateOffset(
            months=12))

    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """
        pass


class ArtMortalityEvent(RegularEvent, PopulationScopeEventMixin):
    """ assigning ART mortality rates to new initiators
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=2))  # every 12 months
        # make sure any rates are annual if frequency of event is annual

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date
        df = population.props

        worksheet = self.module.parameters['art_mortality_data']

        # find index of all 0-6 months on treatment
        idx = df.index[df.on_art & ((now - df.date_art_start).dt.days < 182.6)]
        df.loc[idx, 'time_on_art'] = '0_6months'

        # find index of all 7-12 months on treatment
        idx = df.index[df.on_art & ((now - df.date_art_start).dt.days >= 182.6)
                       & ((now - df.date_art_start).dt.days < 365.25)]
        df.loc[idx, 'time_on_art'] = '7_12months'

        # find index of all >12 months on treatment
        idx = df.index[df.on_art & ((now - df.date_art_start).dt.days >= 365.25)]
        df.loc[idx, 'time_on_art'] = '12months'

        # find index of all early starters
        idx = df.index[df.on_art & ((df.date_aids_death - df.date_art_start).dt.days >= 730.5)]
        df.loc[idx, 'early_art'] = True

        # find index of all late starters
        idx = df.index[df.on_art & ((df.date_aids_death - df.date_art_start).dt.days < 730.5)]
        df.loc[idx, 'early_art'] = False

        # add age to population.props
        # df_age = pd.merge(df, population.age, left_index=True, right_index=True, how='left')

        # add updated mortality rates
        df_mort = pd.merge(df, worksheet, left_on=['age_years', 'sex', 'time_on_art', 'early_art'],
                           right_on=['age', 'sex', 'time_on_art', 'early'], how='left')
        # df_mort.to_csv('P:/Documents/TLO/test.csv', sep=',')
        # print('df_mort with art mortality: ', df_mort.head(30))

        idx = df.index[df.on_art]
        # print(idx)
        df.loc[idx, 'art_mortality_rate'] = df_mort.loc[idx, 'rate']
        # df.to_csv('P:/Documents/TLO/test2.csv', sep=',')


class HivArtDeathEvent(RegularEvent, PopulationScopeEventMixin):
    """ The regular event that actually kills people.
    """

    def __init__(self, module):
        """ Create a new random death event.
        """
        super().__init__(module, frequency=DateOffset(months=12))

    def apply(self, population):
        """Apply this event to the population.

        For efficiency, we use pandas operations to scan the entire population
        and kill individuals at random.

        :param population: the current population
        """
        params = self.module.parameters
        df = population.props
        now = self.sim.date
        rng = self.module.rng

        # fill missing values with 0
        df['art_mortality_rate'] = df['art_mortality_rate'].fillna(0)

        # Generate a series of random numbers, one per individual
        probs = rng.rand(len(df))
        deaths = df.is_alive & (probs < df.art_mortality_rate)
        will_die = (df[deaths]).index

        for person in will_die:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, person, cause='aids'), now)

        total_deaths = len(will_die)
        self.module.store['Number_dead_art'].append(total_deaths)
        self.module.store['Time'].append(self.sim.date)
