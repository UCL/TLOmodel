"""
Following the skeleton method for HIV

Q: should treatment be in a separate method?
"""

# import any methods from other modules, e.g. for parameter definitions
from typing import Any, Union

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent

# need to import HIV, HIV_Event

import numpy as np
import pandas as pd

# NOTES: what should the functions be returning?
# previously they read in the population dataframe and then returned the modified population dataframe
# how to deal with the current_time variable needed in many functions?
# check use of self
# initialise population function was renamed as there were unresolved differences


# read in data files #
# use function read.parameters in class HIV to do this?
file_path = 'Q:\Thanzi la Onse\HIV\Method_HIV.xlsx'

HIV_ART = pd.read_excel(file_path, sheet_name='ART2009_2021', header=0)

ART_totals = pd.read_excel(file_path, sheet_name='aggregate_number_ART', header=0)

ad_mort = pd.read_excel(file_path, sheet_name='mortality_rates', header=0)

paed_mortART = pd.read_excel(file_path, sheet_name='paediatric_mortality_rates', header=0)

inds = pd.read_csv('Q:/Thanzi la Onse/HIV/initial_pop_dataframe2018.csv')
p = inds.shape[0]  # number of rows in pop (# individuals)


# sim_size = int(100)
# current_time = 2018


# HELPER FUNCTION - should these go in class(ART)?
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


# this class contains all the methods required to set up the baseline population
class ART(Module):
    """Models ART initiation.

    Methods required:
    * `read_parameters(data_folder)`
    * `initialise_population(population)`
    * `initialise_simulation(sim)`
    * `on_birth(mother, child)`
    """

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    # PARAMETERS = {}

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'on_ART': Property(Types.BOOL, 'Currently on ART'),
        'date_ART_start': Property(Types.DATE, 'Date ART started'),
        'ART_mortality': Property(Types.REAL, 'Mortality rates whilst on ART'),
    }

    # def read_parameters(self, data_folder):
    #     """Read parameter values from file, if required.
    #     :param data_folder: path of a folder supplied to the Simulation containing data files.
    #       Typically modules would read a particular file within here.
    #     """


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

    # assign mortality rates for those on ART
    def ART_mortality_rates(self, df, current_time):

        self.current_time = current_time

        # INFANTS
        # treated infant mortality averaged over all CD4%
        # could stratify by early / late treatment

        # infants 0-6 months on treatment
        # age < 1
        df.loc[get_index(df, 0, 1, 'I', 1, self.current_time, 0, 0.5), 'mortality'] = \
            paed_mortART['paed_mort'][(paed_mortART.time_on_ART == '0_6months') & (paed_mortART.age == '0')]

        # age 1-2
        df.loc[get_index(df, 1, 3, 'I', 1, self.current_time, 0, 0.5), 'mortality'] = \
            paed_mortART['paed_mort'][(paed_mortART.time_on_ART == '0_6months') & (paed_mortART.age == '1_2')]

        # age 3-4
        df.loc[get_index(df, 3, 5, 'I', 1, self.current_time, 0, 0.5), 'mortality'] = \
            paed_mortART['paed_mort'][(paed_mortART.time_on_ART == '0_6months') & (paed_mortART.age == '3_4')]

        # infants 7-12 months on treatment by age
        # age < 1
        df.loc[get_index(df, 0, 1, 'I', 1, self.current_time, 0.5, 1), 'mortality'] = \
            paed_mortART['paed_mort'][(paed_mortART.time_on_ART == '7_12months') & (paed_mortART.age == '0')]

        # age 1-2
        df.loc[get_index(df, 1, 3, 'I', 1, self.current_time, 0.5, 1), 'mortality'] = \
            paed_mortART['paed_mort'][(paed_mortART.time_on_ART == '7_12months') & (paed_mortART.age == '1_2')]

        # age 3-4
        df.loc[get_index(df, 3, 5, 'I', 1, self.current_time, 0.5, 1), 'mortality'] = \
            paed_mortART['paed_mort'][(paed_mortART.time_on_ART == '7_12months') & (paed_mortART.age == '3_4')]

        # infants >12 months on treatment by age
        # age < 1
        df.loc[get_index(df, 0, 1, 'I', 1, self.current_time, 1, np.Inf), 'mortality'] = \
            paed_mortART['paed_mort'][(paed_mortART.time_on_ART == '12months') & (paed_mortART.age == '0')]

        # age 1-2
        df.loc[get_index(df, 1, 3, 'I', 1, self.current_time, 1, np.Inf), 'mortality'] = \
            paed_mortART['paed_mort'][(paed_mortART.time_on_ART == '12months') & (paed_mortART.age == '1_2')]

        # age 3-4
        df.loc[get_index(df, 3, 5, 'I', 1, self.current_time, 1, np.Inf), 'mortality'] = \
            paed_mortART['paed_mort'][(paed_mortART.time_on_ART == '12months') & (paed_mortART.age == '3_4')]

        # ADULTS
        # early starters > 2 years to death when starting treatment
        # 0-6 months on treatment by four age groups
        # male age <25
        df.loc[
            get_index(df, 5, 25, 'I', 1, self.current_time, 0, 0.5, optarg1=2, optarg2=np.Inf, optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age15_24') & (ad_mort.sex == 'M') &
                                           (ad_mort.ART == 'Y0_6E')]

        # male age 25-34
        df.loc[
            get_index(df, 25, 35, 'I', 1, self.current_time, 0, 0.5, optarg1=2, optarg2=np.Inf, optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age25_34') & (ad_mort.sex == 'M') &
                                           (ad_mort.ART == 'Y0_6E')]

        # male age 35-44
        df.loc[
            get_index(df, 35, 45, 'I', 1, self.current_time, 0, 0.5, optarg1=2, optarg2=np.Inf, optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age35_44') & (ad_mort.sex == 'M') &
                                           (ad_mort.ART == 'Y0_6E')]

        # male age >= 45
        df.loc[
            get_index(df, 45, np.Inf, 'I', 1, self.current_time, 0, 0.5, optarg1=2, optarg2=np.Inf,
                      optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age45') & (ad_mort.sex == 'M') &
                                           (ad_mort.ART == 'Y0_6E')]

        # female age <25
        df.loc[
            get_index(df, 5, 25, 'I', 1, self.current_time, 0, 0.5, optarg1=2, optarg2=np.Inf, optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age15_24') & (ad_mort.sex == 'F') &
                                           (ad_mort.ART == 'Y0_6E')]

        # female age 25-34
        df.loc[
            get_index(df, 25, 35, 'I', 1, self.current_time, 0, 0.5, optarg1=2, optarg2=np.Inf, optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age25_34') & (ad_mort.sex == 'F') &
                                           (ad_mort.ART == 'Y0_6E')]

        # female age 35-44
        df.loc[
            get_index(df, 35, 45, 'I', 1, self.current_time, 0, 0.5, optarg1=2, optarg2=np.Inf, optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age35_44') & (ad_mort.sex == 'F') &
                                           (ad_mort.ART == 'Y0_6E')]

        # female age >= 45
        df.loc[
            get_index(df, 45, np.Inf, 'I', 1, self.current_time, 0, 0.5, optarg1=2, optarg2=np.Inf,
                      optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age45') & (ad_mort.sex == 'F') & (ad_mort.ART == 'Y0_6E')]

        # 7-12 months on treatment by four age groups
        # male age <25
        df.loc[
            get_index(df, 5, 25, 'I', 1, self.current_time, 0.5, 2, optarg1=2, optarg2=np.Inf, optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age15_24') & (ad_mort.sex == 'M') &
                                           (ad_mort.ART == 'Y7_12E')]

        # male age 25-34
        df.loc[
            get_index(df, 25, 35, 'I', 1, self.current_time, 0.5, 2, optarg1=2, optarg2=np.Inf, optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age25_34') & (ad_mort.sex == 'M') &
                                           (ad_mort.ART == 'Y7_12E')]

        # male age 35-44
        df.loc[
            get_index(df, 35, 45, 'I', 1, self.current_time, 0.5, 2, optarg1=2, optarg2=np.Inf, optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age35_44') & (ad_mort.sex == 'M') &
                                           (ad_mort.ART == 'Y7_12E')]

        # male age >= 45
        df.loc[
            get_index(df, 45, np.Inf, 'I', 1, self.current_time, 0.5, 2, optarg1=2, optarg2=np.Inf,
                      optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age45') & (ad_mort.sex == 'M') &
                                           (ad_mort.ART == 'Y7_12E')]

        # female age <25
        df.loc[
            get_index(df, 5, 25, 'I', 1, self.current_time, 0.5, 2, optarg1=2, optarg2=np.Inf, optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age15_24') & (ad_mort.sex == 'F') &
                                           (ad_mort.ART == 'Y7_12E')]

        # female age 25-34
        df.loc[
            get_index(df, 25, 35, 'I', 1, self.current_time, 0.5, 2, optarg1=2, optarg2=np.Inf, optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age25_34') & (ad_mort.sex == 'F') &
                                           (ad_mort.ART == 'Y7_12E')]

        # female age 35-44
        df.loc[
            get_index(df, 35, 45, 'I', 1, self.current_time, 0.5, 2, optarg1=2, optarg2=np.Inf, optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age35_44') & (ad_mort.sex == 'F') &
                                           (ad_mort.ART == 'Y7_12E')]

        # female age >= 45
        df.loc[
            get_index(df, 45, np.Inf, 'I', 1, self.current_time, 0.5, 2, optarg1=2, optarg2=np.Inf,
                      optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age45') & (ad_mort.sex == 'F') & (ad_mort.ART == 'Y7_12E')]

        # > 12 months on treatment by four age groups
        # male age <25
        df.loc[
            get_index(df, 5, 25, 'I', 1, self.current_time, 2, np.Inf, optarg1=2, optarg2=np.Inf, optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age15_24') & (ad_mort.sex == 'M') & (ad_mort.ART == 'Y12E')]

        # male age 25-34
        df.loc[
            get_index(df, 25, 35, 'I', 1, self.current_time, 2, np.Inf, optarg1=2, optarg2=np.Inf, optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age25_34') & (ad_mort.sex == 'M') &
                                           (ad_mort.ART == 'Y7_12E')]

        # male age 35-44
        df.loc[
            get_index(df, 35, 45, 'I', 1, self.current_time, 2, np.Inf, optarg1=2, optarg2=np.Inf, optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age35_44') & (ad_mort.sex == 'M') & (ad_mort.ART == 'Y12E')]

        # male age >= 45
        df.loc[
            get_index(df, 45, np.Inf, 'I', 1, self.current_time, 2, np.Inf, optarg1=2, optarg2=np.Inf,
                      optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age45') & (ad_mort.sex == 'M') & (ad_mort.ART == 'Y12E')]

        # female age <25
        df.loc[
            get_index(df, 5, 25, 'I', 1, self.current_time, 2, np.Inf, optarg1=2, optarg2=np.Inf, optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age15_24') & (ad_mort.sex == 'F') & (ad_mort.ART == 'Y12E')]

        # female age 25-34
        df.loc[
            get_index(df, 25, 35, 'I', 1, self.current_time, 2, np.Inf, optarg1=2, optarg2=np.Inf, optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age25_34') & (ad_mort.sex == 'F') & (ad_mort.ART == 'Y12E')]

        # female age 35-44
        df.loc[
            get_index(df, 35, 45, 'I', 1, self.current_time, 2, np.Inf, optarg1=2, optarg2=np.Inf, optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age35_44') & (ad_mort.sex == 'F') & (ad_mort.ART == 'Y12E')]

        # female age >= 45
        df.loc[
            get_index(df, 45, np.Inf, 'I', 1, self.current_time, 2, np.Inf, optarg1=2, optarg2=np.Inf,
                      optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age45') & (ad_mort.sex == 'F') & (ad_mort.ART == 'Y12E')]

        # late starters < 2 years to death when starting treatment
        # 0-6 months on treatment by four age groups
        # male age <25
        df.loc[
            get_index(df, 5, 25, 'I', 1, self.current_time, 0, 0.5, optarg1=0, optarg2=2, optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age15_24') & (ad_mort.sex == 'M') &
                                           (ad_mort.ART == 'Y0_6L')]

        # male age 25-34
        df.loc[
            get_index(df, 25, 35, 'I', 1, self.current_time, 0, 0.5, optarg1=0, optarg2=2, optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age25_34') & (ad_mort.sex == 'M') &
                                           (ad_mort.ART == 'Y0_6L')]

        # male age 35-44
        df.loc[
            get_index(df, 35, 45, 'I', 1, self.current_time, 0, 0.5, optarg1=0, optarg2=2, optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age35_44') & (ad_mort.sex == 'M') &
                                           (ad_mort.ART == 'Y0_6L')]

        # male age >= 45
        df.loc[
            get_index(df, 45, np.Inf, 'I', 1, self.current_time, 0, 0.5, optarg1=0, optarg2=2, optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age45') & (ad_mort.sex == 'M') &
                                           (ad_mort.ART == 'Y0_6L')]

        # female age <25
        df.loc[
            get_index(df, 5, 25, 'I', 1, self.current_time, 0, 0.5, optarg1=0, optarg2=2, optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age15_24') & (ad_mort.sex == 'F') &
                                           (ad_mort.ART == 'Y0_6L')]

        # female age 25-34
        df.loc[
            get_index(df, 25, 35, 'I', 1, self.current_time, 0, 0.5, optarg1=0, optarg2=2, optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age25_34') & (ad_mort.sex == 'F') &
                                           (ad_mort.ART == 'Y0_6L')]

        # female age 35-44
        df.loc[
            get_index(df, 35, 45, 'I', 1, self.current_time, 0, 0.5, optarg1=0, optarg2=2, optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age35_44') & (ad_mort.sex == 'F') &
                                           (ad_mort.ART == 'Y0_6L')]

        # female age >= 45
        df.loc[
            get_index(df, 45, np.Inf, 'I', 1, self.current_time, 0, 0.5, optarg1=0, optarg2=2, optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age45') & (ad_mort.sex == 'F') &
                                           (ad_mort.ART == 'Y0_6L')]

        # 7-12 months on treatment by four age groups
        # male age <25
        df.loc[
            get_index(df, 5, 25, 'I', 1, self.current_time, 0.5, 2, optarg1=0, optarg2=2, optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age15_24') & (ad_mort.sex == 'M') &
                                           (ad_mort.ART == 'Y7_12L')]

        # male age 25-34
        df.loc[
            get_index(df, 25, 35, 'I', 1, self.current_time, 0.5, 2, optarg1=0, optarg2=2, optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age25_34') & (ad_mort.sex == 'M') &
                                           (ad_mort.ART == 'Y7_12L')]

        # male age 35-44
        df.loc[
            get_index(df, 35, 45, 'I', 1, self.current_time, 0.5, 2, optarg1=0, optarg2=2, optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age35_44') & (ad_mort.sex == 'M') &
                                           (ad_mort.ART == 'Y7_12L')]

        # male age >= 45
        df.loc[
            get_index(df, 45, np.Inf, 'I', 1, self.current_time, 0.5, 2, optarg1=0, optarg2=2, optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age45') & (ad_mort.sex == 'M') &
                                           (ad_mort.ART == 'Y7_12EL')]

        # female age <25
        df.loc[
            get_index(df, 5, 25, 'I', 1, self.current_time, 0.5, 2, optarg1=0, optarg2=2, optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age15_24') & (ad_mort.sex == 'F') &
                                           (ad_mort.ART == 'Y7_12L')]

        # female age 25-34
        df.loc[
            get_index(df, 25, 35, 'I', 1, self.current_time, 0.5, 2, optarg1=0, optarg2=2, optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age25_34') & (ad_mort.sex == 'F') &
                                           (ad_mort.ART == 'Y7_12L')]

        # female age 35-44
        df.loc[
            get_index(df, 35, 45, 'I', 1, self.current_time, 0.5, 2, optarg1=0, optarg2=2, optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age35_44') & (ad_mort.sex == 'F') &
                                           (ad_mort.ART == 'Y7_12L')]

        # female age >= 45
        df.loc[
            get_index(df, 45, np.Inf, 'I', 1, self.current_time, 0.5, 2, optarg1=0, optarg2=2, optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age45') & (ad_mort.sex == 'F') &
                                           (ad_mort.ART == 'Y7_12L')]

        # > 12 months on treatment by four age groups
        # male age <25
        df.loc[
            get_index(df, 5, 25, 'I', 1, self.current_time, 2, np.Inf, optarg1=0, optarg2=2, optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age15_24') & (ad_mort.sex == 'M') &
                                           (ad_mort.ART == 'Y12L')]

        # male age 25-34
        df.loc[
            get_index(df, 25, 35, 'I', 1, self.current_time, 2, np.Inf, optarg1=0, optarg2=2, optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age25_34') & (ad_mort.sex == 'M') &
                                           (ad_mort.ART == 'Y7_12L')]

        # male age 35-44
        df.loc[
            get_index(df, 35, 45, 'I', 1, self.current_time, 2, np.Inf, optarg1=0, optarg2=2, optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age35_44') & (ad_mort.sex == 'M') &
                                           (ad_mort.ART == 'Y12L')]

        # male age >= 45
        df.loc[
            get_index(df, 45, np.Inf, 'I', 1, self.current_time, 2, np.Inf, optarg1=0, optarg2=2, optarg3='M'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age45') & (ad_mort.sex == 'M') & (ad_mort.ART == 'Y12L')]

        # female age <25
        df.loc[
            get_index(df, 5, 25, 'I', 1, self.current_time, 2, np.Inf, optarg1=0, optarg2=2, optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age15_24') & (ad_mort.sex == 'F') &
                                           (ad_mort.ART == 'Y12L')]

        # female age 25-34
        df.loc[
            get_index(df, 25, 35, 'I', 1, self.current_time, 2, np.Inf, optarg1=0, optarg2=2, optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age25_34') & (ad_mort.sex == 'F') &
                                           (ad_mort.ART == 'Y12L')]

        # female age 35-44
        df.loc[
            get_index(df, 35, 45, 'I', 1, self.current_time, 2, np.Inf, optarg1=0, optarg2=2, optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age35_44') & (ad_mort.sex == 'F') &
                                           (ad_mort.ART == 'Y12L')]

        # female age >= 45
        df.loc[
            get_index(df, 45, np.Inf, 'I', 1, self.current_time, 2, np.Inf, optarg1=0, optarg2=2, optarg3='F'),
            'mortality'] = ad_mort['rate'][(ad_mort.age == 'age45') & (ad_mort.sex == 'F') & (ad_mort.ART == 'Y12L')]

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


class ART_Event(RegularEvent, PopulationScopeEventMixin):
    """HIV infection events

    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module):
        """One line summary here

        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.

        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=1))


    def allocate_ART(self, df, current_time):
        # look at how many slots are currently taken
        # then check number available for current year
        # remember to divide by sim_size
        # allocate any unfilled ART slots by longest time infected

        self.current_time = current_time

        # total number of ART slots available 2018
        self.ART_infants = int(
            ART_totals['number_on_ART'][(ART_totals.year == self.current_time) & (ART_totals.age == '0_14')])
        self.ART_adults = int(
            ART_totals['number_on_ART'][(ART_totals.year == self.current_time) & (ART_totals.age == '15_80')])

        # infants - this treats older kids first as they've been infected longer
        # less likely to have infants treated close to birth/infection
        tmp1 = len(df[(df.on_ART == 1) & (df.age < 15)])  # current number on ART
        diff_inf = self.ART_infants - tmp1

        if diff_inf < 0:
            diff_inf = 0  # replace negative values with zero

        subgroup = df[(df.age < 15) & (df.on_ART == 0) & (df.has_HIV == 1)]
        subgroup = subgroup.sort_values(by='date_HIV_infection', ascending=True,
                                        na_position='last')  # order by earliest time infected
        tmp2 = subgroup.id[0:(diff_inf + 1)]
        df.loc[tmp2, 'on_ART'] = 1
        df.loc[tmp2, 'date_ART_start'] = self.current_time

        # adults
        tmp3 = len(df[(df.on_ART == 1) & (df.age >= 15)])  # current number on ART
        diff_ad = self.ART_adults - tmp3

        if diff_ad < 0:
            diff_ad = 0  # replace negative values with zero

        subgroup2 = df[(df.age >= 15) & (df.on_ART == 0) & (df.has_HIV == 1)]
        subgroup2 = subgroup2.sort_values(by='date_HIV_infection', ascending=True,
                                          na_position='last')  # order by earliest time infected
        tmp4 = subgroup2.id[0:(diff_ad + 1)]
        df.loc[tmp4, 'on_ART'] = 1
        df.loc[tmp4, 'date_ART_start'] = self.current_time

        return df


# inds.describe(include='all')
# inds['status'].value_counts()
# inds['treat'].value_counts()

# TODO: ART allocation for infants - influenced by mother's ART status
# TODO: include ART start date if possible - would influence next year's deaths

