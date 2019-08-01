"""
Childhood diarrhoea module
Documentation: 04 - Methods Repository/Method_Child_EntericInfection.xlsx
"""
import logging

import numpy as np
import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin
from tlo.methods.iCCM import HSI_Sick_Child_Seeks_Care_From_HSA

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ChildhoodDiarrhoea(Module):
    PARAMETERS = {
        'base_incidence_diarrhoea_by_rotavirus': Parameter(Types.LIST,
         'incidence of diarrhoea caused by rotavirus in age groups 0-11, 12-23, 24-59 months'),
        'base_prev_dysentery': Parameter(Types.REAL,
         'initial prevalence of dysentery, among children aged 0-11 months,'
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no access to clean water, no improved sanitation'
         ),
        'rp_dysentery_age12to23mo': Parameter(Types.REAL,
         'relative prevalence of dysentery for age 12 to 23 months'
         ),
        'rp_dysentery_age24to59mo': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for age 24 to 59 months'
         ),
        'rp_dysentery_HIV': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for HIV positive'
         ),
        'rp_dysentery_SAM': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for severe acute malnutrition'
         ),
        'rp_dysentery_excl_breast': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for exclusive breastfeeding upto 6 months'
         ),
        'rp_dysentery_cont_breast': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for continued breastfeeding upto 23 months'
         ),
        'rp_dysentery_HHhandwashing': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for household handwashing'
         ),
        'rp_dysentery_clean_water': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for access to clean water'
         ),
        'rp_dysentery_improved_sanitation': Parameter
        (Types.REAL,
         'relative prevalence of dysentery for improved sanitation'
         ),
        'base_incidence_dysentery': Parameter
        (Types.REAL,
         'baseline incidence of dysentery, among children aged < 11 months, '
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no clean water source, no improved sanitation'
         ),
        'rr_dysentery_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of dysentery for age 12 to 23 months'
         ),
        'rr_dysentery_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of dysentery for age 24 to 59 months'
         ),
        'rr_dysentery_HIV': Parameter
        (Types.REAL,
         'relative rate of dysentery for HIV positive'
         ),
        'rr_dysentery_SAM': Parameter
        (Types.REAL,
         'relative rate of dysentery for severe acute malnutrition'
         ),
        'rr_dysentery_excl_breast': Parameter
        (Types.REAL,
         'relative rate of dysentery for exclusive breastfeeding upto 6 months'
         ),
        'rr_dysentery_cont_breast': Parameter
        (Types.REAL,
         'relative rate of dysentery for continued breastfeeding 6 months to 2 years'
         ),
        'rr_dysentery_HHhandwashing': Parameter
        (Types.REAL,
         'relative rate of dysentery for household handwashing'
         ),
        'rr_dysentery_clean_water': Parameter
        (Types.REAL,
         'relative rate of dysentery for access to clean water'
         ),
        'rr_dysentery_improved_sanitation': Parameter
        (Types.REAL,
         'relative rate of dysentery for improved sanitation'
         ),
        'base_prev_acute_diarrhoea': Parameter
        (Types.REAL,
         'initial prevalence of acute watery diarrhoea, among children aged < 11 months, '
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no clean water source, no improved sanitation'
         ),
        'rp_acute_diarrhoea_age12to23mo': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for age 12 to 23 months'
         ),
        'rp_acute_diarrhoea_age24to59mo': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for age 24 to 59 months'
         ),
        'rp_acute_diarrhoea_HIV': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for HIV positive'
         ),
        'rp_acute_diarrhoea_SAM': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for severe acute malnutrition'
         ),
        'rp_acute_diarrhoea_excl_breast': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for exclusive breastfeeding upto 6 months'
         ),
        'rp_acute_diarrhoea_cont_breast': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for continued breastfeeding 6 months to 2 years'
         ),
        'rp_acute_diarrhoea_HHhandwashing': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for household handwashing'
         ),
        'rp_acute_diarrhoea_clean_water': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for access to clean water'
         ),
        'rp_acute_diarrhoea_improved_sanitation': Parameter
        (Types.REAL,
         'relative prevalence of acute watery diarrhoea for improved sanitation'
         ),
        'base_incidence_acute_diarrhoea': Parameter
        (Types.REAL,
         'baseline incidence of acute watery diarrhoea, among children aged < 11 months, '
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no clean water source, no improved sanitation'
         ),
        'rr_acute_diarrhoea_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for age 12 to 23 months'
         ),
        'rr_acute_diarrhoea_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for age 24 to 59 months'
         ),
        'rr_acute_diarrhoea_HIV': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for HIV positive'
         ),
        'rr_acute_diarrhoea_SAM': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for severe acute malnutrition'
         ),
        'rr_acute_diarrhoea_excl_breast': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for exclusive breastfeeding upto 6 months'
         ),
        'rr_acute_diarrhoea_cont_breast': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for continued breastfeeding 6 months to 2 years'
         ),
        'rr_acute_diarrhoea_HHhandwashing': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for household handwashing'
         ),
        'rr_acute_diarrhoea_clean_water': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for access to clean water'
         ),
        'rr_acute_diarrhoea_improved_sanitation': Parameter
        (Types.REAL,
         'relative rate of acute watery diarrhoea for improved sanitation'
         ),
        'base_prev_persistent_diarrhoea': Parameter
        (Types.REAL,
         'initial prevalence of persistent diarrhoea, among children aged < 11 months,'
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no access to clean water, no improved sanitation'
         ),
        'rp_persistent_diarrhoea_age12to23mo': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for age 12 to 23 months'
         ),
        'rp_persistent_diarrhoea_age24to59mo': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for age 24 to 59 months'
         ),
        'rp_persistent_diarrhoea_HIV': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for HIV positive'
         ),
        'rp_persistent_diarrhoea_SAM': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for severe acute malnutrition'
         ),
        'rp_persistent_diarrhoea_excl_breast': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for exclusive breastfeeding upto 6 months'
         ),
        'rp_persistent_diarrhoea_cont_breast': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for continued breastfeeding 6 months to 2 years'
         ),
        'rp_persistent_diarrhoea_HHhandwashing': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for household handwashing'
         ),
        'rp_persistent_diarrhoea_clean_water': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for access to clean water'
         ),
        'rp_persistent_diarrhoea_improved_sanitation': Parameter
        (Types.REAL,
         'relative prevalence of persistent diarrhoea for improved sanitation'
         ),
        'r_death_dysentery': Parameter
        (Types.REAL,
         'death rate from dysentery among children aged less than 11 months, '
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no access to clean water, no improved sanitation'
         ),
        'rr_death_dysentery_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of death from dysentery for age 12 to 23 months'
         ),
        'rr_death_dysentery_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of death from dysentery for age 24 to 59 months'
         ),
        'rr_death_dysentery_HIV': Parameter
        (Types.REAL,
         'relative rate of death from dysentery for HIV positive status'
         ),
        'rr_death_dysentery_SAM': Parameter
        (Types.REAL,
         'relative rate of death from dysentery for severe acute malnutrition'
         ),
        'rr_death_dysentery_excl_breast': Parameter
        (Types.REAL,
         'relative rate of death from dysentery for exclusive breastfeeding'
         ),
        'rr_death_dysentery_cont_breast': Parameter
        (Types.REAL,
         'relative rate of death from dysentery for continued breastfeeding'
         ),
        'r_death_acute_diarrhoea': Parameter
        (Types.REAL,
         'death rate from acute watery diarrhoea among children aged less than 11 months, '
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no access to clean water, no improved sanitation'
         ),
        'rr_death_acute_diar_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of death from acute watery diarrhoea for age 12 to 23 months'
         ),
        'rr_death_acute_diar_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of death from acute watery diarrhoea for age 24 to 59 months'
         ),
        'rr_death_acute_diar_HIV': Parameter
        (Types.REAL,
         'relative rate of death from acute watery diarrhoea for HIV positive status'
         ),
        'rr_death_acute_diar_SAM': Parameter
        (Types.REAL,
         'relative rate of death from acute watery diarrhoea for severe acute malnutrition'
         ),
        'r_death_persistent_diarrhoea': Parameter
        (Types.REAL,
         'death rate from persistent diarrhoea among children aged less than 11 months, '
         'HIV negative, no SAM'
         ),
        'rr_death_persistent_diar_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of death from persistent diarrhoea for age 12 to 23 months'
         ),
        'rr_death_persistent_diar_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of death from persistent diarrhoea for age 24 to 59 months'
         ),
        'rr_death_persistent_diar_HIV': Parameter
        (Types.REAL,
         'relative rate of death from persistent diarrhoea for HIV positive status'
         ),
        'rr_death_persistent_diar_SAM': Parameter
        (Types.REAL,
         'relative rate of death from persistent diarrhoea for severe acute malnutrition'
         ),
        'r_dysentery_prog_persistent': Parameter
        (Types.REAL,
         'rate of progression from dysentery to persistent diarrhoea among children aged 0-11 months, '
         'HIV negative, no SAM '
         ),
        'rr_dysentery_prog_persistent_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of progression from dysentery to persistent diarrhoea for age between 12 to 23 months'
         ),
        'rr_dysentery_prog_persistent_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of progression from dysentery to persistent diarrhoea for age between 24 to 59 months'
         ),
        'rr_dysentery_prog_persistent_HIV': Parameter
        (Types.REAL,
         'relative rate of progression from dysentery to persistent diarrhoea for HIV positive status'
         ),
        'rr_dysentery_prog_persistent_SAM': Parameter
        (Types.REAL,
         'relative rate of progression from dysentery to persistent diarrhoea for severe acute malnutrition'
         ),
        'r_acute_diarr_prog_persistent': Parameter
        (Types.REAL,
         'baseline recovery rate from acute watery diarrhoea among children ages 2 to 11 months, '
         'HIV negative, no SAM'
         ),
        'rr_acute_diarr_prog_persistent_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of progression from acute watery diarrhoea persistent diarrhoea for age between 12 to 23 months'
         ),
        'rr_acute_diarr_prog_persistent_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of progression from acute watery diarrhoea persistent diarrhoea for age between 24 to 59 months'
         ),
        'rr_acute_diarr_prog_persistent_HIV': Parameter
        (Types.REAL,
         'relative rate of progression from acute watery diarrhoea persistent diarrhoea for HIV positive status'
         ),
        'rr_acute_diarr_prog_persistent_SAM': Parameter
        (Types.REAL,
         'relative rate of progression from acute watery diarrhoea persistent diarrhoea for severe acute malnutrition'
         ),
        'init_prop_diarrhoea_status': Parameter
        (Types.LIST,
         'initial proportions in ei_diarrhoea_status categories '
         'for children aged 2-11 months, HIV negative, no SAM, '
         'not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no indoor air pollution'
         )
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'gi_diarrhoea_status': Property(Types.BOOL, 'symptomatic infection - diarrhoea disease'),
        'gi_diarrhoea_acute_type': Property(Types.CATEGORICAL, 'clinical diarrhoea type',
                                        categories=['dysentery', 'acute watery diarrhoea']),
        'gi_dehydration_status': Property(Types.CATEGORICAL, 'dehydration status',
                                          categories=['no dehydration', 'some dehydration', 'severe dehydration']),
        'gi_persistent_diarrhoea': Property(Types.BOOL, 'diarrhoea episode longer than 14 days - persistent type'),

        'gi_diarrhoea_death': Property(Types.BOOL, 'death caused by diarrhoea'),
        'date_of_onset_diarrhoea': Property(Types.DATE, 'date of onset of diarrhoea'),
        'gi_recovered_date': Property(Types.DATE, 'date of recovery from enteric infection'),
        'gi_diarrhoea_death_date': Property(Types.DATE, 'date of death from enteric infection'),
        'gi_diarrhoea_count': Property(Types.REAL, 'number of diarrhoea episodes per individual'),
        'has_hiv': Property(Types.BOOL, 'temporary property - has hiv'),
        'malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
        # symptoms of diarrhoea for care seeking
        'di_diarrhoea_loose_watery_stools': Property(Types.BOOL, 'diarrhoea symptoms - loose or watery stools'),
        'di_blood_in_stools': Property(Types.BOOL, 'dysentery symptoms - blood in the stools'),
        'di_diarrhoea_over14days': Property(Types.BOOL, 'persistent diarrhoea - diarrhoea for 14 days or more'),
    }

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module
        """
        p = self.parameters

        p['base_incidence_diarrhoea_by_rotavirus'] = [0.061, 0.02225, 0.00125]
        p['base_prev_dysentery'] = 0.3
        p['rp_dysentery_age12to23mo'] = 0.8
        p['rp_dysentery_age24to59mo'] = 0.5
        p['rp_dysentery_HIV'] = 1.4
        p['rp_dysentery_SAM'] = 1.25
        p['rp_dysentery_excl_breast'] = 0.5
        p['rp_dysentery_cont_breast'] = 0.7
        p['rp_dysentery_HHhandwashing'] = 0.5
        p['rp_dysentery_clean_water'] = 0.6
        p['rp_dysentery_improved_sanitation'] = 0.6
        p['base_incidence_dysentery'] = 0.4
        p['rr_dysentery_age12to23mo'] = 0.8
        p['rr_dysentery_age24to59mo'] = 0.5
        p['rr_dysentery_HIV'] = 1.4
        p['rr_dysentery_SAM'] = 1.25
        p['rr_dysentery_excl_breast'] = 0.5
        p['rr_dysentery_cont_breast'] = 0.7
        p['rr_dysentery_HHhandwashing'] = 0.5
        p['rr_dysentery_clean_water'] = 0.6
        p['rr_dysentery_improved_sanitation'] = 0.4
        p['base_prev_acute_diarrhoea'] = 0.4
        p['rp_acute_diarrhoea_age12to23mo'] = 0.8
        p['rp_acute_diarrhoea_age24to59mo'] = 0.5
        p['rp_acute_diarrhoea_HIV'] = 1.3
        p['rp_acute_diarrhoea_SAM'] = 1.3
        p['rp_acute_diarrhoea_excl_breast'] = 0.5
        p['rp_acute_diarrhoea_cont_breast'] = 0.7
        p['rp_acute_diarrhoea_HHhandwashing'] = 0.8
        p['rp_acute_diarrhoea_clean_water'] = 0.6
        p['rp_acute_diarrhoea_improved_sanitation'] = 0.6
        p['base_incidence_acute_diarrhoea'] = 0.5
        p['rr_acute_diarrhoea_age12to23mo'] = 0.8
        p['rr_acute_diarrhoea_age24to59mo'] = 0.5
        p['rr_acute_diarrhoea_HIV'] = 1.3
        p['rr_acute_diarrhoea_SAM'] = 1.3
        p['rr_acute_diarrhoea_excl_breast'] = 0.5
        p['rr_acute_diarrhoea_cont_breast'] = 0.7
        p['rr_acute_diarrhoea_HHhandwashing'] = 0.8
        p['rr_acute_diarrhoea_clean_water'] = 0.6
        p['rr_acute_diarrhoea_improved_sanitation'] = 0.7
        p['base_prev_persistent_diarrhoea'] = 0.2
        p['rp_persistent_diarrhoea_age12to23mo'] = 0.8
        p['rp_persistent_diarrhoea_age24to59mo'] = 0.5
        p['rp_persistent_diarrhoea_HIV'] = 1.3
        p['rp_persistent_diarrhoea_SAM'] = 1.3
        p['rp_persistent_diarrhoea_excl_breast'] = 0.5
        p['rp_persistent_diarrhoea_cont_breast'] = 0.7
        p['rp_persistent_diarrhoea_HHhandwashing'] = 0.8
        p['rp_persistent_diarrhoea_clean_water'] = 0.6
        p['rp_persistent_diarrhoea_improved_sanitation'] = 0.7
        p['r_dysentery_prog_persistent'] = 0.5
        p['rr_dysentery_prog_persistent_age12to23mo'] = 1.6
        p['rr_dysentery_prog_persistent_age24to59mo'] = 1.7
        p['rr_dysentery_prog_persistent_HIV'] = 1.5
        p['rr_dysentery_prog_persistent_SAM'] = 1.6
        p['rr_acute_diarr_prog_persistent_age12to23mo'] = 1.6
        p['rr_acute_diarr_prog_persistent_age24to59mo'] = 1.7
        p['rr_acute_diarr_prog_persistent_HIV'] = 1.5
        p['rr_acute_diarr_prog_persistent_SAM'] = 1.6
        p['init_prop_diarrhoea_status'] = [0.2, 0.2, 0.2]
        p['r_death_dysentery'] = 0.3
        p['rr_death_dysentery_age12to23mo'] = 0.7
        p['rr_death_dysentery_age24to59mo'] = 0.5
        p['rr_death_dysentery_HIV'] = 1.3
        p['rr_death_dysentery_SAM'] = 1.4
        p['rr_death_dysentery_excl_breast'] = 0.6
        p['rr_death_dysentery_cont_breast'] = 0.8
        p['r_death_acute_diarrhoea'] = 0.3
        p['rr_death_acute_diar_age12to23mo'] = 0.7
        p['rr_death_acute_diar_age24to59mo'] = 0.5
        p['rr_death_acute_diar_HIV'] = 1.3
        p['rr_death_acute_diar_SAM'] = 1.4
        p['rr_death_acute_diar_excl_breast'] = 0.6
        p['rr_death_acute_diar_cont_breast'] = 0.8
        p['r_death_persistent_diarrhoea'] = 0.3
        p['rr_death_persistent_diar_age12to23mo'] = 0.7
        p['rr_death_persistent_diar_age24to59mo'] = 0.5
        p['rr_death_persistent_diar_HIV'] = 1.3
        p['rr_death_persistent_diar_SAM'] = 1.4
        p['rr_death_persistent_diar_excl_breast'] = 0.6
        p['rr_death_persistent_diar_cont_breast'] = 0.8

    def initialise_population(self, population):
        """Set our property values for the initial population.
        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.
        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals
        m = self
        rng = m.rng
        now = self.sim.date

        # DEFAULTS
        df['gi_diarrhoea_status'] = 'none'
        df['gi_dehydration_status'] = 'no dehydration'
        df['date_of_onset_diarrhoea'] = pd.NaT
        df['gi_recovered_date'] = pd.NaT
        df['gi_diarrhoea_death_date'] = pd.NaT
        df['gi_diarrhoea_count'] = 0
        df['gi_diarrhoea_death'] = False
        df['malnutrition'] = False
        df['has_hiv'] = False
        df['exclusive_breastfeeding'] = False
        df['continued_breastfeeding'] = False

        # -------------------- ASSIGN DIARRHOEA STATUS AT BASELINE (PREVALENCE) -----------------------

        df_under5 = df.age_years < 5 & df.is_alive
        under5_idx = df.index[(df.age_years < 5) & df.is_alive]

        # create dataframe of the probabilities of ei_diarrhoea_status for children
        # aged 0-11 months, HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding,
        # no household handwashing, no access to clean water, no improved sanitation

        p_dysentery_status = pd.Series(self.init_prop_diarrhoea_status[0], index=under5_idx)
        p_acute_diarrhoea_status = pd.Series(self.init_prop_diarrhoea_status[1], index=under5_idx)
        p_persistent_diarrhoea_status = pd.Series(self.init_prop_diarrhoea_status[2], index=under5_idx)

        # create probabilities of dysentery for all age under 5
        p_dysentery_status.loc[(df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] \
            *= self.rp_dysentery_age12to23mo
        p_dysentery_status.loc[(df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] \
            *= self.rp_dysentery_age24to59mo
        p_dysentery_status.loc[(df.has_hiv == True) & df_under5] *= self.rp_dysentery_HIV
        p_dysentery_status.loc[(df.malnutrition == True) & df_under5] *= self.rp_dysentery_SAM
        p_dysentery_status.loc[(df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) & df.is_alive] \
            *= self.rp_dysentery_excl_breast
        p_dysentery_status.loc[(df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) &
                               (df.age_exact_years < 2) & df.is_alive] *= self.rp_dysentery_cont_breast
        p_dysentery_status.loc[(df.li_no_access_handwashing == False) & df_under5] *= self.rp_dysentery_HHhandwashing
        p_dysentery_status.loc[(df.li_no_clean_drinking_water == False) & df_under5] *= self.rp_dysentery_clean_water
        p_dysentery_status.loc[(df.li_unimproved_sanitation == False) & df_under5] *= self.rp_dysentery_improved_sanitation

        # create probabilities of acute watery diarrhoea for all age under 5
        p_acute_diarrhoea_status.loc[(df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] \
            *= self.rp_acute_diarrhoea_age12to23mo
        p_acute_diarrhoea_status.loc[(df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] \
            *= self.rp_acute_diarrhoea_age24to59mo
        p_acute_diarrhoea_status.loc[(df.has_hiv == True) & df_under5] *= self.rp_acute_diarrhoea_HIV
        p_acute_diarrhoea_status.loc[(df.malnutrition == True) & df_under5] *= self.rp_acute_diarrhoea_SAM
        p_acute_diarrhoea_status.loc[(df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) & df.is_alive] \
            *= self.rp_acute_diarrhoea_excl_breast
        p_acute_diarrhoea_status.loc[(df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) &
                                     (df.age_exact_years < 2) & df.is_alive] *= self.rp_acute_diarrhoea_cont_breast
        p_acute_diarrhoea_status.loc[(df.li_no_clean_drinking_water == False) & df_under5] \
            *= self.rp_acute_diarrhoea_clean_water
        p_acute_diarrhoea_status.loc[(df.li_unimproved_sanitation == False) & df_under5] \
            *= self.rp_acute_diarrhoea_improved_sanitation

        # create probabilities of persistent diarrhoea for all age under 5
        p_persistent_diarrhoea_status.loc[(df.age_exact_years >= 1) & ( df.age_exact_years < 2) & df.is_alive] \
            *= self.rp_persistent_diarrhoea_age12to23mo
        p_persistent_diarrhoea_status.loc[(df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] \
            *= self.rp_persistent_diarrhoea_age24to59mo
        p_persistent_diarrhoea_status.loc[(df.has_hiv == True) & df_under5] *= self.rp_persistent_diarrhoea_HIV
        p_persistent_diarrhoea_status.loc[(df.malnutrition == True) & df_under5] *= self.rp_persistent_diarrhoea_SAM
        p_persistent_diarrhoea_status.loc[(df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) &
                                          df.is_alive] *= self.rp_persistent_diarrhoea_excl_breast
        p_persistent_diarrhoea_status.loc[(df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) &
                                          (df.age_exact_years < 2) & df.is_alive] \
            *= self.rp_persistent_diarrhoea_cont_breast
        p_persistent_diarrhoea_status.loc[(df.li_no_clean_drinking_water == False) & df_under5] \
            *= self.rp_persistent_diarrhoea_clean_water
        p_persistent_diarrhoea_status.loc[(df.li_unimproved_sanitation == False) & df_under5] \
            *= self.rp_persistent_diarrhoea_improved_sanitation

        # randomly select some individuals to assign diarrhoeal disease status at the start of simulation
        random_draw = pd.Series(rng.random_sample(size=len(under5_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive])

        # create a temporary dataframe called dfx to hold values of probabilities and random draw
        dfx = pd.concat([p_dysentery_status, p_acute_diarrhoea_status, p_persistent_diarrhoea_status, random_draw],
                        axis=1)
        dfx.columns = ['p_dysentery', 'p_acute_diarrhoea', 'p_persistent_diarrhoea', 'random_draw']

        dfx['p_none'] = 1 - (dfx.p_dysentery + dfx.p_acute_diarrhoea + dfx.p_persistent_diarrhoea)

        # based on probabilities of being in each category, define cut-offs to determine status from
        # random draw uniform(0,1)

        # assign baseline values of ei_diarrhoea_status based on probabilities and value of random draw

        idx_none = dfx.index[dfx.p_none > dfx.random_draw]
        idx_dysentery = dfx.index[(dfx.p_none < dfx.random_draw) & ((dfx.p_none + dfx.p_dysentery) > dfx.random_draw)]
        idx_acute_diarrhoea = dfx.index[((dfx.p_none + dfx.p_dysentery) < dfx.random_draw) &
                                        (dfx.p_none + dfx.p_dysentery + dfx.p_acute_diarrhoea) > dfx.random_draw]
        idx_persistent_diarrhoea = dfx.index[
            ((dfx.p_none + dfx.p_dysentery + dfx.p_acute_diarrhoea) < dfx.random_draw) &
            (dfx.p_none + dfx.p_dysentery + dfx.p_acute_diarrhoea +
             dfx.p_persistent_diarrhoea) > dfx.random_draw]

        df.loc[idx_none, 'gi_diarrhoea_status'] = 'none'
        df.loc[idx_dysentery, 'gi_diarrhoea_status'] = 'dysentery'
        df.loc[idx_acute_diarrhoea, 'gi_diarrhoea_status'] = 'acute watery diarrhoea'
        df.loc[idx_persistent_diarrhoea, 'gi_diarrhoea_status'] = 'persistent diarrhoea'

        # get all the individuals with diarrhoea
        # diarrhoea_count = df.loc[(df.gi_diarrhoea_status == 'dysentery') |
        #                          (df.gi_diarrhoea_status == 'acute watery diarrhoea') |
        #                          (df.gi_diarrhoea_status == 'persistent diarrhoea')].sum()
        # print(diarrhoea_count)

    def initialise_simulation(self, sim):
        """
        Get ready for simulation start.
        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event for dysentery ---------------------------------------------------
        event_dysentery = DysenteryEvent(self)
        sim.schedule_event(event_dysentery, sim.date + DateOffset(months=2))

        # add an event to log to screen
        # sim.schedule_event(DysenteryLoggingEvent(self), sim.date + DateOffset(months=6))

        # add the basic event for acute watery diarrhoea ---------------------------------------
        event_acute_diar = AcuteDiarrhoeaEvent(self)
        sim.schedule_event(event_acute_diar, sim.date + DateOffset(months=3))

        '''# death event
        death_all_diarrhoea = DeathDiarrhoeaEvent(self)
        sim.schedule_event(death_all_diarrhoea, sim.date)

        # add an event to log to screen
        sim.schedule_event(AcuteDiarrhoeaLoggingEvent(self), sim.date + DateOffset(months=6))

        # add the basic event for persistent diarrhoea ------------------------------------------
        event_persistent_diar = PersistentDiarrhoeaEvent(self)
        sim.schedule_event(event_persistent_diar, sim.date + DateOffset(months=3))

        # add an event to log to screen
        sim.schedule_event(PersistentDiarrhoeaLoggingEvent(self), sim.date + DateOffset(months=6))
        '''

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        pass

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is Diarrhoea, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)


class DysenteryEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=2))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = m.rng

        # --------------------------------------------------------------------------------------------------------
        # UPDATING FOR CHILDREN UNDER 5 WITH CURRENT STATUS 'NONE' TO DYSENTERY
        # --------------------------------------------------------------------------------------------------------

        no_diarrhoea = df.is_alive & (df.gi_diarrhoea_status == 'none')
        no_diarrhoea_under5 = df.is_alive & (df.gi_diarrhoea_status == 'none') & (df.age_years < 5)

        eff_prob_gi_dysentery = pd.Series(m.base_incidence_dysentery,
                                          index=df.index[no_diarrhoea_under5])

        eff_prob_gi_dysentery.loc[no_diarrhoea & (df.age_exact_years >= 1) & (df.age_exact_years < 2)] \
            *= m.rr_dysentery_age12to23mo
        eff_prob_gi_dysentery.loc[no_diarrhoea & (df.age_exact_years >= 2) & (df.age_exact_years < 5)]\
            *= m.rr_dysentery_age24to59mo
        eff_prob_gi_dysentery.loc[no_diarrhoea_under5 & df.li_no_access_handwashing == False] \
            *= m.rr_dysentery_HHhandwashing
        eff_prob_gi_dysentery.loc[no_diarrhoea_under5 & (df.has_hiv == True)] *= m.rr_dysentery_HIV
        eff_prob_gi_dysentery.loc[no_diarrhoea_under5 & df.malnutrition == True] *= m.rr_dysentery_SAM
        eff_prob_gi_dysentery.loc[no_diarrhoea & df.exclusive_breastfeeding == True & (df.age_exact_years <= 0.5)] \
            *= m.rr_dysentery_excl_breast
        eff_prob_gi_dysentery.loc[no_diarrhoea & df.continued_breastfeeding == True & (df.age_exact_years > 0.5) &
                                  (df.age_exact_years < 2)] *= m.rr_dysentery_cont_breast
        eff_prob_gi_dysentery.loc[no_diarrhoea_under5 & df.li_no_clean_drinking_water == False] \
            *= m.rr_dysentery_clean_water
        eff_prob_gi_dysentery.loc[no_diarrhoea_under5 & df.li_unimproved_sanitation == False]\
            *= m.rr_dysentery_improved_sanitation

        di_current_none_idx = df.index[no_diarrhoea_under5]

        random_draw_01 = pd.Series(rng.random_sample(size=len(di_current_none_idx)),
                                   index=df.index[no_diarrhoea_under5])

        incident_dysentery = eff_prob_gi_dysentery > random_draw_01
        idx_get_dysentery = eff_prob_gi_dysentery.index[incident_dysentery]

        df.loc[idx_get_dysentery, 'gi_diarrhoea_status'] = 'dysentery'

        # WHEN THEY GET DYSENTERY - DATE
        random_draw_days = np.random.randint(0, 60, size=len(idx_get_dysentery))
        adding_days = pd.to_timedelta(random_draw_days, unit='d')
        date_of_aquisition = self.sim.date + adding_days
        df.loc[idx_get_dysentery, 'date_of_onset_diarrhoea'] = date_of_aquisition

        # # # # # # ASSIGN DEHYDRATION LEVELS FOR DYSENTERY # # # # # #
        under5_dysentery_idx = df.index[(df.age_years < 5) & df.is_alive & (df.gi_diarrhoea_status == 'dysentery')]

        eff_prob_some_dehydration_dysentery = pd.Series(0.5, index=under5_dysentery_idx)
        eff_prob_severe_dehydration_dysentery = pd.Series(0.3, index=under5_dysentery_idx)
        random_draw_a = pd.Series(self.sim.rng.random_sample(size=len(under5_dysentery_idx)),
                                  index=under5_dysentery_idx)

        no_dehydration_dysentery = 1 - (eff_prob_some_dehydration_dysentery + eff_prob_severe_dehydration_dysentery)
        some_dehydration_dysentery = \
            (random_draw_a > no_dehydration_dysentery) & \
            (random_draw_a < (no_dehydration_dysentery + eff_prob_some_dehydration_dysentery))
        severe_dehydration_dysentery = \
            ((no_dehydration_dysentery + eff_prob_some_dehydration_dysentery) < random_draw_a) & \
            ((no_dehydration_dysentery + eff_prob_some_dehydration_dysentery + eff_prob_severe_dehydration_dysentery) >
             random_draw_a)

        idx_some_dehydration_dysentery = eff_prob_some_dehydration_dysentery.index[some_dehydration_dysentery]
        idx_severe_dehydration_dysentery = eff_prob_severe_dehydration_dysentery.index[severe_dehydration_dysentery]
        df.loc[idx_some_dehydration_dysentery, 'gi_dehydration_status'] = 'some dehydration'
        df.loc[idx_severe_dehydration_dysentery, 'gi_dehydration_status'] = 'severe dehydration'

        # # # # # # # # SYMPTOMS FROM DYSENTERY # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        df.loc[idx_get_dysentery, 'di_diarrhoea_loose_watery_stools'] = True
        df.loc[idx_get_dysentery, 'di_blood_in_stools'] = True
        df.loc[idx_get_dysentery, 'di_diarrhoea_over14days'] = False

        # --------------------------------------------------------------------------------------------------------
        # SEEKING CARE FOR ACUTE BLOODY DIARRHOEA
        # --------------------------------------------------------------------------------------------------------

        dysentery_symptoms = df.index[df.is_alive & (df.age_years < 5) & (df.di_diarrhoea_loose_watery_stools == True) &
                                      (df.di_blood_in_stools == True) & df.di_diarrhoea_over14days == False]

        seeks_care = pd.Series(data=False, index=dysentery_symptoms)
        for individual in dysentery_symptoms:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(individual, symptom_code=1)
            seeks_care[individual] = self.module.rng.rand() < prob
            event = HSI_Sick_Child_Seeks_Care_From_HSA(self.module, person_id=individual)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(weeks=2)
                                                                )

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # GET DYSENTERY, SYMPTOMS, SEEK CARE, IF NOT SEEK CARE PROGRESS TO PERSISTENT, SELF-RECOVER OR DEATH #
        # self.sim.schedule_event(DiarrhoeaDeathEvent(self, person_id))


class AcuteDiarrhoeaEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(weeks=2))

    def apply(self, population):

        df = population.props
        m = self.module
        rng = m.rng

        # --------------------------------------------------------------------------------------------------------
        # UPDATING FOR CHILDREN UNDER 5 WITH CURRENT STATUS 'NONE' TO DYSENTERY
        # --------------------------------------------------------------------------------------------------------

        no_diarrhoea = df.is_alive & (df.gi_diarrhoea_status == 'none')
        no_diarrhoea_under5 = df.is_alive & (df.gi_diarrhoea_status == 'none') & (df.age_years < 5)

        eff_prob_ei_acute_diarrhoea = pd.Series(m.base_incidence_acute_diarrhoea,
                                                index=df.index[no_diarrhoea_under5])

        eff_prob_ei_acute_diarrhoea.loc[no_diarrhoea & (df.age_exact_years >= 1) & (df.age_exact_years < 2)] \
            *= m.rr_acute_diarrhoea_age12to23mo
        eff_prob_ei_acute_diarrhoea.loc[no_diarrhoea & (df.age_exact_years >= 2) & (df.age_exact_years < 5)] \
            *= m.rr_acute_diarrhoea_age24to59mo
        eff_prob_ei_acute_diarrhoea.loc[no_diarrhoea_under5 & df.li_no_access_handwashing == False] \
            *= m.rr_acute_diarrhoea_HHhandwashing
        eff_prob_ei_acute_diarrhoea.loc[no_diarrhoea_under5 & (df.has_hiv == True)] \
            *= m.rr_acute_diarrhoea_HIV
        eff_prob_ei_acute_diarrhoea.loc[no_diarrhoea_under5 & df.malnutrition == True] \
            *= m.rr_acute_diarrhoea_SAM
        eff_prob_ei_acute_diarrhoea.loc[no_diarrhoea & df.exclusive_breastfeeding == True & (df.age_exact_years <= 0.5)]\
            *= m.rr_acute_diarrhoea_excl_breast
        eff_prob_ei_acute_diarrhoea.loc[no_diarrhoea & df.continued_breastfeeding == True & (df.age_exact_years > 0.5) &
                                        (df.age_exact_years < 2)] *= m.rr_acute_diarrhoea_cont_breast
        eff_prob_ei_acute_diarrhoea.loc[no_diarrhoea_under5 & df.li_no_clean_drinking_water == False] \
            *= m.rr_acute_diarrhoea_clean_water
        eff_prob_ei_acute_diarrhoea.loc[no_diarrhoea_under5 & df.li_unimproved_sanitation == False] \
            *= m.rr_acute_diarrhoea_improved_sanitation

        di_current_none_idx = \
            df.index[no_diarrhoea_under5]

        random_draw = pd.Series(rng.random_sample(size=len(di_current_none_idx)), index=df.index[no_diarrhoea_under5])

        incident_acute_diarrhoea = eff_prob_ei_acute_diarrhoea > random_draw
        idx_get_acute_diarrhoea = eff_prob_ei_acute_diarrhoea.index[incident_acute_diarrhoea]
        df.loc[idx_get_acute_diarrhoea, 'gi_diarrhoea_status'] = 'acute watery diarrhoea'

        # WHEN THEY GET ACUTE WATERY DIARRHOEA - DATE
        random_draw_days = np.random.randint(0, 60, size=len(incident_acute_diarrhoea))
        adding_days = pd.to_timedelta(random_draw_days, unit='d')
        date_of_aquisition = self.sim.date + adding_days
        df.loc[idx_get_acute_diarrhoea, 'date_of_onset_diarrhoea'] = date_of_aquisition

        # # # # # # ASSIGN DEHYDRATION LEVELS FOR ACUTE WATERY DIARRHOEA # # # # # #

        under5_acute_diarrhoea_idx = df.index[
            (df.age_years < 5) & df.is_alive & (df.gi_diarrhoea_status == 'acute watery diarrhoea')]

        eff_prob_some_dehydration_acute_diarrhoea = pd.Series(0.5, index=under5_acute_diarrhoea_idx)
        eff_prob_severe_dehydration_acute_diarrhoea = pd.Series(0.3, index=under5_acute_diarrhoea_idx)
        random_draw_b = pd.Series(self.sim.rng.random_sample(size=len(under5_acute_diarrhoea_idx)),
                                  index=under5_acute_diarrhoea_idx)

        no_dehydration_acute_diarrhoea = \
            1 - (eff_prob_some_dehydration_acute_diarrhoea + eff_prob_severe_dehydration_acute_diarrhoea)
        some_dehydration_acute_diarrhoea = \
            (random_draw_b > no_dehydration_acute_diarrhoea) & \
            (random_draw_b < (no_dehydration_acute_diarrhoea + eff_prob_some_dehydration_acute_diarrhoea))
        severe_dehydration_acute_diarrhoea = \
            ((no_dehydration_acute_diarrhoea + eff_prob_some_dehydration_acute_diarrhoea) < random_draw_b) & \
            ((no_dehydration_acute_diarrhoea + eff_prob_some_dehydration_acute_diarrhoea +
              eff_prob_severe_dehydration_acute_diarrhoea) > random_draw_b)

        idx_some_dehydration_acute_diarrhoea = eff_prob_some_dehydration_acute_diarrhoea.index[
            some_dehydration_acute_diarrhoea]
        idx_severe_dehydration_acute_diarrhoea = eff_prob_severe_dehydration_acute_diarrhoea.index[
            severe_dehydration_acute_diarrhoea]

        df.loc[idx_some_dehydration_acute_diarrhoea, 'gi_dehydration_status'] = 'some dehydration'
        df.loc[idx_severe_dehydration_acute_diarrhoea, 'gi_dehydration_status'] = 'severe dehydration'

        # # # # # # # # SYMPTOMS FROM ACUTE WATERY DIARRHOEA # # # # # # # # # # # # # # # # # # # # # # #
        df.loc[idx_get_acute_diarrhoea, 'di_diarrhoea_loose_watery_stools'] = True
        df.loc[idx_get_acute_diarrhoea, 'di_blood_in_stools'] = False
        df.loc[idx_get_acute_diarrhoea, 'di_diarrhoea_over14days'] = False

        # --------------------------------------------------------------------------------------------------------
        # SEEKING CARE FOR ACUTE WATERY DIARRHOEA
        # --------------------------------------------------------------------------------------------------------

        acute_diarrhoea_symptoms = \
            df.index[df.is_alive & (df.age_years < 5) & (df.di_diarrhoea_loose_watery_stools == True) &
                     (df.di_blood_in_stools == False) & df.di_diarrhoea_over14days == False]

        seeks_care = pd.Series(data=False, index=acute_diarrhoea_symptoms)
        for individual in acute_diarrhoea_symptoms:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(individual, symptom_code=1)
            seeks_care[individual] = self.module.rng.rand() < prob
            event = HSI_Sick_Child_Seeks_Care_From_HSA(self.module, person_id=individual)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(weeks=2)
                                                                )


class ProgressPersistentDiarrhoeaEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module):
        super().__init__(module)

    def apply(self, population):

        df = population.props
        m = self.module
        rng = m.rng

        # --------------------------------------------------------------------------------------------------------
        # UPDATING FOR CHILDREN UNDER 5 PROGRESSING FROM DYSENTERY TO PERSISTENT DIARRHOEA
        # --------------------------------------------------------------------------------------------------------

        current_dysentery = df.is_alive & (df.ei_diarrhoea_status == 'dysentery')
        current_dysentery_under5 = df.is_alive & (df.ei_diarrhoea_status == 'dysentery') & (df.age_years < 5)
        eff_prob_dysentery_prog_persistent = pd.Series(m.r_dysentery_prog_persistent,
                                                       index=df.index[current_dysentery_under5])

        eff_prob_dysentery_prog_persistent.loc[current_dysentery & (df.age_exact_years >= 1) &
                                               (df.age_exact_years < 2)] *= m.rr_persistent_diarrhoea_age12to23mo
        eff_prob_dysentery_prog_persistent.loc[current_dysentery & (df.age_exact_years >= 2) &
                                               (df.age_exact_years < 5)] *= m.rr_persistent_diarrhoea_age24to59mo
        eff_prob_dysentery_prog_persistent.loc[current_dysentery_under5 & df.li_no_access_handwashing == False] \
            *= m.rr_persistent_diarrhoea_HHhandwashing
        eff_prob_dysentery_prog_persistent.loc[current_dysentery_under5 & (df.has_hiv == True)] \
            *= m.rr_persistent_diarrhoea_HIV
        eff_prob_dysentery_prog_persistent.loc[current_dysentery_under5 & df.malnutrition == True] \
            *= m.rr_persistent_diarrhoea_SAM
        eff_prob_dysentery_prog_persistent.loc[current_dysentery & df.exclusive_breastfeeding == True
                                               & (df.age_exact_years <= 0.5)] *= m.rr_persistent_diarrhoea_excl_breast
        eff_prob_dysentery_prog_persistent.loc[current_dysentery & df.continued_breastfeeding == True &
                                               (df.age_exact_years > 0.5) & (df.age_exact_years < 2)] \
            *= m.rr_persistent_diarrhoea_cont_breast
        eff_prob_dysentery_prog_persistent.loc[current_dysentery_under5 & df.li_no_clean_drinking_water == False]\
            *= m.rr_persistent_diarrhoea_clean_water
        eff_prob_dysentery_prog_persistent.loc[current_dysentery_under5 & df.li_unimproved_sanitation == False] \
            *= m.rr_persistent_diarrhoea_improved_sanitation

        idx_current_dysentery_under5 = df.index[current_dysentery_under5]
        random_draw = pd.Series(rng.random_sample(size=len(idx_current_dysentery_under5)),
                                index=idx_current_dysentery_under5)

        dysentery_prog_persistent = eff_prob_dysentery_prog_persistent > random_draw
        idx_dysentery_prog_persistent = df.index[dysentery_prog_persistent]
        df.loc[idx_dysentery_prog_persistent, 'gi_diarrhoea_status'] = 'persistent diarrhoea'

        # --------------------------------------------------------------------------------------------------------
        # UPDATING FOR CHILDREN UNDER 5 PROGRESSING FROM ACUTE WATERY DIARRHOEA TO PERSISTENT DIARRHOEA
        # --------------------------------------------------------------------------------------------------------

        current_acute_diarrhoea = df.is_alive & (df.ei_diarrhoea_status == 'acute watery diarrhoea')
        current_acute_diarrhoea_under5 = df.is_alive & (df.ei_diarrhoea_status == 'acute watery diarrhoea') & \
                                         (df.age_years < 5)
        eff_prob_acute_diarr_prog_persistent = pd.Series(m.r_dysentery_prog_persistent,
                                                       index=df.index[current_acute_diarrhoea_under5])

        eff_prob_acute_diarr_prog_persistent.loc[current_acute_diarrhoea & (df.age_exact_years >= 1) &
                                               (df.age_exact_years < 2)] *= m.rr_persistent_diarrhoea_age12to23mo
        eff_prob_acute_diarr_prog_persistent.loc[current_acute_diarrhoea & (df.age_exact_years >= 2) &
                                               (df.age_exact_years < 5)] *= m.rr_persistent_diarrhoea_age24to59mo
        eff_prob_acute_diarr_prog_persistent.loc[current_acute_diarrhoea_under5 & df.li_no_access_handwashing == False] \
            *= m.rr_persistent_diarrhoea_HHhandwashing
        eff_prob_acute_diarr_prog_persistent.loc[current_acute_diarrhoea_under5 & (df.has_hiv == True)] \
            *= m.rr_persistent_diarrhoea_HIV
        eff_prob_acute_diarr_prog_persistent.loc[current_acute_diarrhoea_under5 & df.malnutrition == True] \
            *= m.rr_persistent_diarrhoea_SAM
        eff_prob_acute_diarr_prog_persistent.loc[current_acute_diarrhoea & df.exclusive_breastfeeding == True &
                                                 (df.age_exact_years <= 0.5)] *= m.rr_persistent_diarrhoea_excl_breast
        eff_prob_acute_diarr_prog_persistent.loc[current_acute_diarrhoea & df.continued_breastfeeding == True &
                                                 (df.age_exact_years > 0.5) & (df.age_exact_years < 2)] \
            *= m.rr_persistent_diarrhoea_cont_breast
        eff_prob_acute_diarr_prog_persistent.loc[current_acute_diarrhoea_under5 & df.li_no_clean_drinking_water == False] \
            *= m.rr_persistent_diarrhoea_clean_water
        eff_prob_acute_diarr_prog_persistent.loc[current_acute_diarrhoea_under5 & df.li_unimproved_sanitation == False] \
            *= m.rr_persistent_diarrhoea_improved_sanitation

        idx_current_acute_diarrhoea_under5 = df.index[current_acute_diarrhoea_under5]
        random_draw = pd.Series(rng.random_sample(size=len(idx_current_acute_diarrhoea_under5)),
                                index=idx_current_acute_diarrhoea_under5)

        acute_diarr_prog_persistent = eff_prob_acute_diarr_prog_persistent > random_draw
        idx_acute_diarr_prog_persistent = df.index[acute_diarr_prog_persistent]
        df.loc[idx_acute_diarr_prog_persistent, 'gi_diarrhoea_status'] = 'persistent diarrhoea'

        # WHEN THEY GET PROGRESS TO PERSISTENT DIARRHOEA - DATE ---------------------------------------------
        if date_of_onset_diarrhoea + DateOffset(weeks=2): # HERE I WANT THOSE WHO STILL HAVE DIARRHOEA AFTER 2 WEEKS
        # SINCE THE START OF DIARRHOEA, THEN THEY WILL GO THROUGH THE EVENT = PROGRESSPERSISTENTDIARRHOEA

        # # # # # # ASSIGN DEHYDRATION LEVELS FOR PERSISTENT DIARRHOEA # # # # # #
        # this should be according to the dehydration level previously assigned when in acute phase - but I changed
        # the status to persistent already....????
        # and some percentage may increase in severity

        under5_persistent_diarrhoea_idx = df.index[
            (df.age_years < 5) & df.is_alive & (df.ei_diarrhoea_status == 'persistent diarrhoea')]

        # # # # # # # # SYMPTOMS FROM ACUTE WATERY DIARRHOEA # # # # # # # # # # # # # # # # # # # # # # #
        df.loc[idx_get_persistent_diarrhoea, 'di_diarrhoea_loose_watery_stools'] = True
        df.loc[idx_get_persistent_diarrhoea, 'di_blood_in_stools'] = True or False
        df.loc[idx_get_persistent_diarrhoea, 'di_diarrhoea_over14days'] = True

        # --------------------------------------------------------------------------------------------------------
        # SEEKING CARE FOR PERSISTENT DIARRHOEA
        # --------------------------------------------------------------------------------------------------------

        persistent_diarrhoea_symptoms = \
            df.index[df.is_alive & (df.age_years < 5) & (df.di_diarrhoea_loose_watery_stools == True) &
                     (df.di_blood_in_stools == True | False) & df.di_diarrhoea_over14days == True]

        seeks_care = pd.Series(data=False, index=persistent_diarrhoea_symptoms)
        for individual in persistent_diarrhoea_symptoms:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(individual, symptom_code=1)
            seeks_care[individual] = self.module.rng.rand() < prob
            event = HSI_Sick_Child_Seeks_Care_From_HSA(self.module, person_id=individual)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(weeks=2)
                                                                )


class DeathDiarrhoeaEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, population):

        df = population.props
        m = self.module
        rng = m.rng

        # ------------------------------------------------------------------------------------------------------
        # DEATH DUE TO ACUTE BLOODY DIARRHOEA - DYSENTERY
        # ------------------------------------------------------------------------------------------------------
        eff_prob_death_dysentery = \
            pd.Series(m.r_death_dysentery,
                      index=df.index[df.is_alive & (df.gi_diarrhoea_status == 'dysentery') & (df.age_years < 5)])

        eff_prob_death_dysentery.loc[df.is_alive & (df.gi_diarrhoea_status == 'dysentery') &
                                     (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= \
            m.rr_death_dysentery_age12to23mo
        eff_prob_death_dysentery.loc[df.is_alive & (df.gi_diarrhoea_status == 'dysentery') &
                                     (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_death_dysentery_age24to59mo
        eff_prob_death_dysentery.loc[df.is_alive & (df.gi_diarrhoea_status == 'dysentery') &
                                     (df.has_hiv == True) & (df.age_exact_years < 5)] *= m.rr_death_dysentery_HIV
        eff_prob_death_dysentery.loc[df.is_alive & (df.gi_diarrhoea_status == 'dysentery') &
                                     df.malnutrition == True & (df.age_exact_years < 5)] *= m.rr_death_dysentery_SAM

        under5_dysentery_idx = df.index[(df.age_years < 5) & df.is_alive & (df.gi_diarrhoea_status == 'dysentery')]

        random_draw = pd.Series(rng.random_sample(size=len(under5_dysentery_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.gi_diarrhoea_status == 'dysentery')])
        dfx = pd.concat([eff_prob_death_dysentery, random_draw], axis=1)
        dfx.columns = ['eff_prob_death_dysentery', 'random_draw']

        for person_id in under5_dysentery_idx:
            if dfx.index[dfx.eff_prob_death_dysentery > dfx.random_draw]:
                df.at[person_id, 'gi_diarrhoea_death'] = True
            else:
                df.at[person_id, 'gi_diarrhoea_status'] = 'none'

        # ------------------------------------------------------------------------------------------------------
        # DEATH DUE TO ACUTE WATERY DIARRHOEA
        # ------------------------------------------------------------------------------------------------------

        eff_prob_death_acute_diarrhoea = \
            pd.Series(m.r_death_acute_diarrhoea,
                      index=df.index[
                          df.is_alive & (df.gi_diarrhoea_status == 'acute watery diarrhoea') & (df.age_years < 5)])
        eff_prob_death_acute_diarrhoea.loc[df.is_alive & (df.gi_diarrhoea_status == 'acute watery diarrhoea') &
                                           (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= \
            m.rr_death_acute_diar_age12to23mo
        eff_prob_death_acute_diarrhoea.loc[df.is_alive & (df.gi_diarrhoea_status == 'acute watery diarrhoea') &
                                           (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_death_acute_diar_age24to59mo
        eff_prob_death_acute_diarrhoea.loc[df.is_alive & (df.gi_diarrhoea_status == 'acute watery diarrhoea') &
                                           (df.has_hiv == True) & (df.age_exact_years < 5)] *= \
            m.rr_death_acute_diar_HIV
        eff_prob_death_acute_diarrhoea.loc[df.is_alive & (df.gi_diarrhoea_status == 'acute watery diarrhoea') &
                                           df.malnutrition == True & (df.age_exact_years < 5)] *= \
            m.rr_death_acute_diar_SAM

        under5_acute_diarrhoea_idx = df.index[(df.age_years < 5) & df.is_alive &
                                              (df.gi_diarrhoea_status == 'acute watery diarrhoea')]

        random_draw = pd.Series(rng.random_sample(size=len(under5_acute_diarrhoea_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.gi_diarrhoea_status == 'acute watery diarrhoea')])
        dfx = pd.concat([eff_prob_death_acute_diarrhoea, random_draw], axis=1)
        dfx.columns = ['eff_prob_death_acute_diarrhoea', 'random_draw']

        for person_id in under5_acute_diarrhoea_idx:
            if dfx.index[dfx.eff_prob_death_acute_diarrhoea > dfx.random_draw]:
                df.at[person_id, 'gi_diarrhoea_death'] = True
            else:
                df.at[person_id, 'gi_diarrhoea_status'] = 'none'

        # ------------------------------------------------------------------------------------------------------
        # DEATH DUE TO PERSISTENT DIARRHOEA
        # ------------------------------------------------------------------------------------------------------

        eff_prob_death_persistent_diarrhoea = \
            pd.Series(m.r_death_persistent_diarrhoea,
                      index=df.index[
                          df.is_alive & (df.gi_diarrhoea_status == 'persistent diarrhoea') & (df.age_years < 5)])
        eff_prob_death_persistent_diarrhoea.loc[df.is_alive & (df.gi_diarrhoea_status == 'persistent diarrhoea') &
                                                (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= \
            m.rr_death_persistent_diar_age12to23mo
        eff_prob_death_persistent_diarrhoea.loc[df.is_alive & (df.gi_diarrhoea_status == 'persistent diarrhoea') &
                                                (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_death_persistent_diar_age24to59mo
        eff_prob_death_persistent_diarrhoea.loc[df.is_alive & (df.gi_diarrhoea_status == 'persistent diarrhoea') &
                                                (df.has_hiv == True) & (df.age_exact_years < 5)] *= \
            m.rr_death_persistent_diar_HIV
        eff_prob_death_persistent_diarrhoea.loc[df.is_alive & (df.gi_diarrhoea_status == 'persistent diarrhoea') &
                                                df.malnutrition == True & (df.age_exact_years < 5)] *= \
            m.rr_death_persistent_diar_SAM

        under5_persistent_diarrhoea_idx = df.index[(df.age_years < 5) & df.is_alive &
                                                   (df.gi_diarrhoea_status == 'persistent diarrhoea')]

        random_draw = pd.Series(rng.random_sample(size=len(under5_persistent_diarrhoea_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.gi_diarrhoea_status == 'persistent diarrhoea')])
        dfx = pd.concat([eff_prob_death_persistent_diarrhoea, random_draw], axis=1)
        dfx.columns = ['eff_prob_death_persistent_diarrhoea', 'random_draw']

        for person_id in under5_persistent_diarrhoea_idx:
            if dfx.index[dfx.eff_prob_death_persistent_diarrhoea > dfx.random_draw]:
                df.at[person_id, 'gi_diarrhoea_death'] = True
            else:
                df.at[person_id, 'ei_diarrhoea_status'] = 'none'

        death_this_period = df.index[(df.gi_diarrhoea_death == True)]
        for individual_id in death_this_period:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, 'ChildhoodDiarrhoea'),
                                    self.sim.date)


class DysenteryLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Handles lifestyle logging"""

    def __init__(self, module):
        """schedule logging to repeat every 3 months
        """
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        """Apply this event to the population.
        """
        df = population.props

        diarrhoea_total = df.loc[(df.ei_diarrhoea_status == 'dysentery') |
                                 (df.ei_diarrhoea_status == 'acute watery diarrhoea') |
                                 (df.ei_diarrhoea_status == 'persistent diarrhoea') & df.is_alive].sum()

        logger.info('%s|summary|%s', self.sim.date,
                    {
                        'people_who_got_diarrhoea': diarrhoea_total
                    })


class AcuteDiarrhoeaLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Handles lifestyle logging"""

    def __init__(self, module):
        """schedule logging to repeat every 3 months
        """
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        """Apply this event to the population.
        """
        pass


class PersistentDiarrhoeaLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Handles lifestyle logging"""

    def __init__(self, module):
        """schedule logging to repeat every 3 months
        """
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        """Apply this event to the population.
        """
        pass
#         if len(idx_incident_dysentery):
#             for child in idx_incident_dysentery:
#                 logger.info('%s|start_dysentery|%s', self.sim.date,
#                             {
#                                 'child_index': child,
#                                 'diarrhoea_type': df.at[child, 'ei_diarrhoea_status'],
#                                 'died': df.at[child, 'ei_diarrhoea_death']
#                             })
