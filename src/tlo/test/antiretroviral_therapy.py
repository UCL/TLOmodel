"""
Assigning ART to HIV+
"""
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent


# this class contains all the methods required to set up the baseline population
class ART(Module):
    """Models ART initiation.
    """

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    # PARAMETERS = {}

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'ART_mortality_rate': Property(Types.REAL, 'Mortality rates whilst on ART'),
    }

    # HELPER FUNCTION
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

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters
        params['p_infection'] = 0.01
        params['p_cure'] = 0.01
        params['initial_prevalence'] = 0.05
        self.parameters['interpolated_pop'] = pd.read_excel(self.workbook_path,
                                                            sheet_name='Interpolated Pop Structure')



    # initial number on ART ordered by longest duration of infection
    # ART numbers are divided by sim_size
    def initial_ART_allocation(self, df, current_time):

        self.current_time = current_time

        # select data for baseline year 2018 - or replace with self.current_time
        self.hiv_art_f = HIV_ART['ART'][
            (HIV_ART.Year == self.current_time) & (HIV_ART.Sex == 'F')]  # returns vector ordered by age
        self.hiv_art_m = HIV_ART['ART'][(HIV_ART.Year == self.current_time) & (HIV_ART.Sex == 'M')]

        for i in range(0, 81):
            # male
            # select each age-group
            subgroup = df[(df.age == i) & (df.has_HIV == 1) & (df.sex == 'M')]
            # order by longest time infected
            subgroup.sort_values(by='date_HIV_infection', ascending=False, na_position='last')
            art_slots = int(self.hiv_art_m.iloc[i])
            tmp = subgroup.id[0:art_slots]
            df.loc[tmp, 'on_ART'] = 1
            df.loc[tmp, 'date_ART_start'] = self.current_time

            # female
            # select each age-group
            subgroup2 = df[(df.age == i) & (df.has_HIV == 1) & (df.sex == 'F')]
            # order by longest time infected
            subgroup2.sort_values(by='date_HIV_infection', ascending=False, na_position='last')
            art_slots2 = int(self.hiv_art_f.iloc[i])
            tmp2 = subgroup2.id[0:art_slots2]
            df.loc[tmp2, 'on_ART'] = 1
            df.loc[tmp2, 'date_ART_start'] = self.current_time

        return df




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


