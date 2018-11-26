"""
Assigning ART to HIV+
"""
import pandas as pd

from tlo import Module, Property, Types


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
        'ART_mortality_rate': Property(Types.REAL, 'Mortality rates whilst on ART'),
    }

    def read_parameters(self, data_folder):
        """ Read parameter values from file
        """
        # values early / late starters
        self.parameters['art_mortality_rates'] = pd.read_excel(self.workbook_path,
                                                               sheet_name='mortality_rates_long')

        # values by CD4 state
        self.parameters['initial_art_mortality_inf'] = pd.read_excel(self.workbook_path,
                                                                     sheet_name='paed_mortality_on_ART')
        self.parameters['initial_art_mortality'] = pd.read_excel(self.workbook_path,
                                                                 sheet_name='mortality_CD4')

    def initialise_population(self, population):
        """ assign mortality rates on ART for baseline population by cd4 state
        """

        df = population.props

        df.ART_mortality_rate = None  # default no values

        worksheet = self.parameters['initial_art_mortality']
        worksheet2 = self.parameters['initial_art_mortality_inf']

        # assume all treated for at least 12 months
        mort_rates = worksheet.loc[worksheet.time_on_treatment == '12months+', ['state', 'age', 'sex', 'mortality']]
        # print(mort_rates)

        # add age to population.props
        df_age = pd.merge(df, population.age, left_index=True, right_index=True, how='left')

        # merge the mortality rates by age, sex and cd4 for ages >=5
        df_age = pd.merge(df_age, mort_rates, left_on=['years', 'sex', 'cd4_state'],
                          right_on=['age', 'sex', 'state'], how='left')
        # print('df_age: ', df_age.head(20))
        # df_age.to_csv('P:/Documents/TLO/test.csv', sep=',')

        # for infants
        mort_rates_inf = worksheet2.loc[worksheet2.time_on_ART == "12"]
        df_age = pd.merge(df_age, mort_rates_inf, left_on=['years'],
                          right_on=['age'], how='left')
        # df_age.to_csv('P:/Documents/TLO/test.csv', sep=',')




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
