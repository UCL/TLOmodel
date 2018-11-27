"""
Assigning ART to HIV+
"""
import pandas as pd

from tlo import DateOffset, Module, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent


# this class contains all the methods required to set up the baseline population
class art(Module):
    """Models ART initiation.
    """

    def __init__(self, name=None, workbook_path=None):
        super().__init__(name)
        self.workbook_path = workbook_path

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    # PARAMETERS = {}

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'art_mortality_rate': Property(Types.REAL, 'Mortality rates whilst on art'),
        'art_early': Property(Types.BOOL, 'If art was started >2 years from scheduled death'),
        'time_on_art': Property(Types.CATEGORICAL, 'length of time on art', categories=['0_6', '7_12', '12']),
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

        worksheet = self.parameters['initial_art_mortality']

        # assume all treated for at least 12 months
        mort_rates = worksheet.loc[worksheet.time_on_treatment == '12months+', ['state', 'age', 'sex', 'mortality']]
        # print(mort_rates)

        # add age to population.props
        df_age = pd.merge(df, population.age, left_index=True, right_index=True, how='left')

        # merge the mortality rates by age, sex and cd4 state, all ages
        df_age = pd.merge(df_age, mort_rates, left_on=['years', 'sex', 'cd4_state'],
                          right_on=['age', 'sex', 'state'], how='left')
        # print('df_age: ', df_age.head(20))
        # df_age.to_csv('P:/Documents/TLO/test.csv', sep=',')

        # retrieve index to assign mortality rates
        idx = df_age.index[df_age.on_art]
        df.loc[idx, 'art_mortality_rate'] = df_age.loc[idx, 'mortality']
        print(idx)

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        pass

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
        super().__init__(module, frequency=DateOffset(months=12))  # every 12 months
        # make sure any rates are annual if frequency of event is annual

    # HELPER FUNCTION
    def get_index(population, age, has_hiv, on_art, current_time,
                  length_treatment_low, length_treatment_high,
                  optarg1=None, optarg2=None, optarg3=None):
        # optargs not needed for infant mortality rates (yet)
        # optarg1 = time from treatment start to death lower bound
        # optarg2 = time from treatment start to death upper bound
        # optarg3 = sex

        df = population.props

        index = df.index[
            (df.age == age) & (df.sex == optarg3) &
            df.has_hiv & df.on_art &
            ((current_time - df.date_art_start) > length_treatment_low) &
            ((current_time - df.date_art_start) <= length_treatment_high) &
            (df.date_aids_death - df.date_art_start >= optarg1) &
            (df.date_aids_death - df.date_art_start < optarg2)]

        return index

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date
        df = population.props

        worksheet = self.module.parameters['art_mortality_data']

        # add age to population.props
        df_age = pd.merge(df, population.age, left_index=True, right_index=True, how='left')

        # find index of all 0-6 months on treatment
        idx = df.index[df.on_art & ((now - df_age.date_art_start).dt.days < 182.6)]
        df.loc[idx, 'time_on_art'] = '0_6'

        # find index of all 7-12 months on treatment
        idx = df.index[df.on_art & ((now - df_age.date_art_start).dt.days >= 182.6)
                       & ((now - df_age.date_art_start).dt.days < 365.25)]
        df.loc[idx, 'time_on_art'] = '7_12'

        # find index of all >12 months on treatment
        idx = df.index[df.on_art & ((now - df_age.date_art_start).dt.days >= 365.25)]
        df.loc[idx, 'time_on_art'] = '12'




