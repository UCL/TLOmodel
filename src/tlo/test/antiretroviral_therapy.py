"""
Assigning ART to HIV+
"""
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
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
        'ART_mortality_rate': Property(Types.REAL, 'Mortality rates whilst on ART'),
    }


    def read_parameters(self, data_folder):
        """ Read parameter values from file
        """
        # values early / late starters
        self.parameters['art_mortality_rates'] = pd.read_excel(self.workbook_path,
                                                               sheet_name='mortality_on_art')

        # values by CD4 state
        self.parameters['initial_art_mortality'] = pd.read_excel(self.workbook_path,
                                                                 sheet_name='mortality_CD4')


    def initialise_population(self, population):
        """ assign mortality rates on ART for baseline population
        """

        df = population.props
        now = self.sim.date

        worksheet = self.parameters['initial_art_mortality']

        # add age to population.props
        df_age = pd.merge(df, population.age, left_index=True, right_index=True, how='left')

        # for those aged >=5 years
        # hold the index of all individuals on art at baseline
        treated = df_age.index[df_age.on_art & (df_age.years >= 5)]

        # select all individuals on art and over 5 years of age
        treated_age = df_age.loc[treated, ['cd4_state', 'sex', 'years']]

        print(treated_age.head(10))

        # merge all infected individuals with their mortality rates by cd4, sex and age
        treated_age_cd4 = treated_age.merge(worksheet,
                                              left_on=['cd4_state', 'sex', 'years'],
                                              right_on=['state', 'sex', 'age'],
                                              how='left')

        assert len(treated_age_cd4) == len(treated)  # check merged row count
        print(treated_age_cd4.head(30))


        # for infants




    # helper function
    def get_index(df, age_low, age_high, has_hiv, on_ART, current_time,
                  length_treatment_low, length_treatment_high,
                  optarg1=None, optarg2=None, optarg3=None):
        # optargs not needed for infant mortality rates (yet)
        # optarg1 = time from treatment start to death lower bound
        # optarg2 = time from treatment start to death upper bound
        # optarg3 = sex

        if optarg1 != None:

            index = df.index[
                (df.age >= age_low) & (df.age < age_high) & (df.sex == optarg3) &
                (df.has_hiv == 1) & (df.on_ART == on_ART) &
                ((current_time - df.date_ART_start) > length_treatment_low) &
                ((current_time - df.date_ART_start) <= length_treatment_high) &
                (df.date_AIDS_death - df.date_ART_start >= optarg1) &
                (df.date_AIDS_death - df.date_ART_start < optarg2)]
        else:
            index = df.index[(df.age >= age_low) & (df.age < age_high) &
                             (df.has_hiv == has_hiv) & (df.on_ART == on_ART) &
                             ((current_time - df.date_ART_start) > length_treatment_low) &
                             ((current_time - df.date_ART_start) <= length_treatment_high)]

        return index


    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        raise NotImplementedError

    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """
        raise NotImplementedError
