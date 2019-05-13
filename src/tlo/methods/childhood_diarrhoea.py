"""
Childhood diarrhoea module
Documentation: 04 - Methods Repository/Method_Child_EntericInfection.xlsx
"""
import logging

import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ChildhoodDiarrhoea(Module):
    PARAMETERS = {
        'base_prev_dysentery': Parameter
        (Types.REAL,
         'initial prevalence of dysentery, among children aged 0-11 months,'
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no access to clean water, no improved sanitation'
         ),
        'rp_dysentery_age12to23mo': Parameter
        (Types.REAL,
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
        'base_incidence_persistent_diarrhoea': Parameter
        (Types.REAL,
         'initial prevalence of persistent diarrhoea, among children aged < 11 months,'
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no access to clean water, no improved sanitation'
         ),
        'rr_persistent_diarrhoea_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for age 12 to 23 months'
         ),
        'rr_persistent_diarrhoea_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for age 24 to 59 months'
         ),
        'rr_persistent_diarrhoea_HIV': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for HIV positive'
         ),
        'rr_persistent_diarrhoea_SAM': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for severe acute malnutrition'
         ),
        'rr_persistent_diarrhoea_excl_breast': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for exclusive breastfeeding upto 6 months'
         ),
        'rr_persistent_diarrhoea_cont_breast': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for continued breastfeeding 6 months to 2 years'
         ),
        'rr_persistent_diarrhoea_HHhandwashing': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for household handwashing'
         ),
        'rr_persistent_diarrhoea_clean_water': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for access to clean water'
         ),
        'rr_persistent_diarrhoea_improved_sanitation': Parameter
        (Types.REAL,
         'relative rate of persistent diarrhoea for improved sanitation'
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
        'r_recovery_dysentery': Parameter
        (Types.REAL,
         'recovery rate from acute watery diarrhoea among children aged 0-11 months, '
         'HIV negative, no SAM '
         ),
        'rr_recovery_dysentery_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of recovery from dysentery for age between 12 to 23 months'
         ),
        'rr_recovery_dysentery_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of recovery from dysentery for age between 24 to 59 months'
         ),
        'rr_recovery_dysentery_HIV': Parameter
        (Types.REAL,
         'relative rate of recovery from dysentery for HIV positive status'
         ),
        'rr_recovery_dysentery_SAM': Parameter
        (Types.REAL,
         'relative rate of recovery from dysentery for severe acute malnutrition'
         ),
        'r_recovery_acute_diarrhoea': Parameter
        (Types.REAL,
         'baseline recovery rate from acute watery diarrhoea among children ages 2 to 11 months, '
         'HIV negative, no SAM'
         ),
        'rr_recovery_acute_diarrhoea_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of recovery from acute watery diarrhoea for age between 12 to 23 months'
         ),
        'rr_recovery_acute_diarrhoea_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of recovery from acute watery diarrhoea for age between 24 to 59 months'
         ),
        'rr_recovery_acute_diarrhoea_HIV': Parameter
        (Types.REAL,
         'relative rate of recovery from acute watery diarrhoea for HIV positive status'
         ),
        'rr_recovery_acute_diarrhoea_SAM': Parameter
        (Types.REAL,
         'relative rate of recovery from acute watery diarrhoea for severe acute malnutrition'
         ),
        'r_recovery_persistent_diarrhoea': Parameter
        (Types.REAL,
         'baseline recovery rate from persistent diarrhoea among children less than 11 months, '
         'HIV negative, no SAM'
         ),
        'rr_recovery_persistent_diarrhoea_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of recovery from acute watery diarrhoea for age between 12 to 23 months'
         ),
        'rr_recovery_persistent_diarrhoea_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of recovery from acute watery diarrhoea for age between 24 to 59 months'
         ),
        'rr_recovery_persistent_diarrhoea_HIV': Parameter
        (Types.REAL,
         'relative rate of recovery from acute watery diarrhoea for HIV positive status'
         ),
        'rr_recovery_persistent_diarrhoea_SAM': Parameter
        (Types.REAL,
         'relative rate of recovery from acute watery diarrhoea for severe acute malnutrition'
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
        'ei_diarrhoea_status': Property(Types.CATEGORICAL, 'enteric infections - diarrhoea status',
                                        categories=['none', 'dysentery', 'acute watery diarrhoea',
                                                    'persistent diarrhoea']),
        'di_dehydration_status': Property(Types.CATEGORICAL, 'dehydration status',
                                          categories=['no dehydration', 'some dehydration', 'severe dehydration']),
        'ei_diarrhoea_death': Property(Types.BOOL, 'death caused by diarrhoea'),
        'ei_infected_date': Property(Types.DATE, 'date of latest infection'),
        'ei_recovered_date': Property(Types.DATE, 'date of recovery from enteric infection'),
        'ei_diarrhoea_death_date': Property(Types.DATE, 'date of death from enteric infection'),
        'ei_diarrhoea_count': Property(Types.REAL, 'number of diarrhoea episodes per individual'),
        'has_hiv': Property(Types.BOOL, 'temporary property - has hiv'),
        'malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
    }

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module
        """
        p = self.parameters

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
        p['base_incidence_dysentery'] = 0.3
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
        p['base_incidence_persistent_diarrhoea'] = 0.5
        p['rr_persistent_diarrhoea_age12to23mo'] = 0.8
        p['rr_persistent_diarrhoea_age24to59mo'] = 0.5
        p['rr_persistent_diarrhoea_HIV'] = 1.3
        p['rr_persistent_diarrhoea_SAM'] = 1.3
        p['rr_persistent_diarrhoea_excl_breast'] = 0.5
        p['rr_persistent_diarrhoea_cont_breast'] = 0.7
        p['rr_persistent_diarrhoea_HHhandwashing'] = 0.8
        p['rr_persistent_diarrhoea_clean_water'] = 0.6
        p['rr_persistent_diarrhoea_improved_sanitation'] = 0.7
        p['init_prop_diarrhoea_status'] = [0.2, 0.2, 0.2]
        p['r_recovery_dysentery'] = 0.9
        p['rr_recovery_dysentery_age12to23mo'] = 0.9
        p['rr_recovery_dysentery_age24to59mo'] = 1.1
        p['rr_recovery_dysentery_HIV'] = 0.5
        p['rr_recovery_dysentery_SAM'] = 0.5
        p['r_recovery_acute_diarrhoea'] = 0.6
        p['rr_recovery_acute_diarrhoea_age12to23mo'] = 0.9
        p['rr_recovery_acute_diarrhoea_age24to59mo'] = 1.2
        p['rr_recovery_acute_diarrhoea_HIV'] = 0.6
        p['rr_recovery_acute_diarrhoea_SAM'] = 0.3
        p['r_recovery_persistent_diarrhoea'] = 0.8
        p['rr_recovery_persistent_diarrhoea_age12to23mo'] = 1.2
        p['rr_recovery_persistent_diarrhoea_age24to59mo'] = 1.3
        p['rr_recovery_persistent_diarrhoea_HIV'] = 0.6
        p['rr_recovery_persistent_diarrhoea_SAM'] = 0.5
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

        # DEFAULTS
        df['ei_diarrhoea_status'] = 'none'
        df['di_dehydration_status'] = 'no dehydration'
        df['ei_infected_date'] = pd.NaT
        df['ei_recovered_date'] = pd.NaT
        df['ei_diarrhoea_death_date'] = pd.NaT
        df['ei_diarrhoea_count'] = 0
        df['ei_diarrhoea_death'] = False
        df['malnutrition'] = False
        df['has_hiv'] = False
        df['exclusive_breastfeeding'] = False
        df['continued_breastfeeding'] = False

        # -------------------- ASSIGN DIARRHOEA STATUS AT BASELINE (PREVALENCE) -----------------------

        under5_idx = df.index[(df.age_years < 5) & df.is_alive]

        # create dataframe of the probabilities of ei_diarrhoea_status for children
        # aged 0-11 months, HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding,
        # no household handwashing, no access to clean water, no improved sanitation

        p_dysentery_status = pd.Series(self.init_prop_diarrhoea_status[0], index=under5_idx)
        p_acute_diarrhoea_status = pd.Series(self.init_prop_diarrhoea_status[1], index=under5_idx)
        p_persistent_diarrhoea_status = pd.Series(self.init_prop_diarrhoea_status[2], index=under5_idx)

        # create probabilities of dysentery for all age under 5
        p_dysentery_status.loc[
            (df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] *= self.rp_dysentery_age12to23mo
        p_dysentery_status.loc[
            (df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] *= self.rp_dysentery_age24to59mo
        p_dysentery_status.loc[
            (df.has_hiv == True) & (df.age_years < 5) & df.is_alive] *= self.rp_dysentery_HIV
        p_dysentery_status.loc[
            (df.malnutrition == True) & (df.age_years < 5) & df.is_alive] *= self.rp_dysentery_SAM
        p_dysentery_status.loc[
            (df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) & df.is_alive] \
            *= self.rp_dysentery_excl_breast
        p_dysentery_status.loc[
            (df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) & (df.age_exact_years < 2) &
            df.is_alive] *= self.rp_dysentery_cont_breast
        p_dysentery_status.loc[
            (df.li_no_access_handwashing == False) & (
                    df.age_years < 5) & df.is_alive] *= self.rp_dysentery_HHhandwashing
        p_dysentery_status.loc[
            (df.li_no_clean_drinking_water == False) & (df.age_years < 5) & df.is_alive] *= self.rp_dysentery_clean_water
        p_dysentery_status.loc[
            (df.li_unimproved_sanitation == False) & (df.age_years < 5) &
            df.is_alive] *= self.rp_dysentery_improved_sanitation

        # create probabilities of acute watery diarrhoea for all age under 5
        p_acute_diarrhoea_status.loc[
            (df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] *= self.rp_acute_diarrhoea_age12to23mo
        p_acute_diarrhoea_status.loc[
            (df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] *= self.rp_acute_diarrhoea_age24to59mo
        p_acute_diarrhoea_status.loc[
            (df.has_hiv == True) & (df.age_years < 5) & df.is_alive] *= self.rp_acute_diarrhoea_HIV
        p_acute_diarrhoea_status.loc[
            (df.malnutrition == True) & (df.age_years < 5) & df.is_alive] *= self.rp_acute_diarrhoea_SAM
        p_acute_diarrhoea_status.loc[
            (df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) & df.is_alive] \
            *= self.rp_acute_diarrhoea_excl_breast
        p_acute_diarrhoea_status.loc[
            (df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) & (df.age_exact_years < 2) &
            df.is_alive] *= self.rp_acute_diarrhoea_cont_breast
        p_acute_diarrhoea_status.loc[
            (df.li_no_clean_drinking_water == False) & (df.age_years < 5) & df.is_alive] *= self.rp_acute_diarrhoea_clean_water
        p_acute_diarrhoea_status.loc[
            (df.li_unimproved_sanitation == False) & (df.age_years < 5) &
            df.is_alive] *= self.rp_acute_diarrhoea_improved_sanitation

        # create probabilities of persistent diarrhoea for all age under 5
        p_persistent_diarrhoea_status.loc[
            (df.age_exact_years >= 1) & (
                    df.age_exact_years < 2) & df.is_alive] *= self.rp_persistent_diarrhoea_age12to23mo
        p_persistent_diarrhoea_status.loc[
            (df.age_exact_years >= 2) & (
                    df.age_exact_years < 5) & df.is_alive] *= self.rp_persistent_diarrhoea_age24to59mo
        p_persistent_diarrhoea_status.loc[
            (df.has_hiv == True) & (df.age_years < 5) & df.is_alive] *= self.rp_persistent_diarrhoea_HIV
        p_persistent_diarrhoea_status.loc[
            (df.malnutrition == True) & (df.age_years < 5) & df.is_alive] *= self.rp_persistent_diarrhoea_SAM
        p_persistent_diarrhoea_status.loc[
            (df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) & df.is_alive] \
            *= self.rp_persistent_diarrhoea_excl_breast
        p_persistent_diarrhoea_status.loc[
            (df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) & (df.age_exact_years < 2) &
            df.is_alive] *= self.rp_persistent_diarrhoea_cont_breast
        p_persistent_diarrhoea_status.loc[
            (df.li_no_clean_drinking_water == False) & (df.age_years < 5) & df.is_alive] *= self.rp_persistent_diarrhoea_clean_water
        p_persistent_diarrhoea_status.loc[
            (df.li_unimproved_sanitation == False) & (df.age_years < 5) &
            df.is_alive] *= self.rp_persistent_diarrhoea_improved_sanitation

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

        df.loc[idx_none, 'ei_diarrhoea_status'] = 'none'
        df.loc[idx_dysentery, 'ei_diarrhoea_status'] = 'dysentery'
        df.loc[idx_acute_diarrhoea, 'ei_diarrhoea_status'] = 'acute watery diarrhoea'
        df.loc[idx_persistent_diarrhoea, 'ei_diarrhoea_status'] = 'persistent diarrhoea'

        # get all the individuals with diarrhoea
        # diarrhoea_count = df.loc[(df.ei_diarrhoea_status == 'dysentery') |
        #                          (df.ei_diarrhoea_status == 'acute watery diarrhoea') |
        #                          (df.ei_diarrhoea_status == 'persistent diarrhoea')].sum()
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
        sim.schedule_event(event_dysentery, sim.date + DateOffset(weeks=2))

        # add an event to log to screen
        sim.schedule_event(DysenteryLoggingEvent(self), sim.date + DateOffset(months=6))

        # add the basic event for acute watery diarrhoea ---------------------------------------
        event_acute_diar = AcuteDiarrhoeaEvent(self)
        sim.schedule_event(event_acute_diar, sim.date + DateOffset(weeks=2))

        # add an event to log to screen
        sim.schedule_event(AcuteDiarrhoeaLoggingEvent(self), sim.date + DateOffset(months=6))

        # add the basic event for persistent diarrhoea ------------------------------------------
        event_persistent_diar = PersistentDiarrhoeaEvent(self)
        sim.schedule_event(event_persistent_diar, sim.date + DateOffset(weeks=4))

        # add an event to log to screen
        sim.schedule_event(PersistentDiarrhoeaLoggingEvent(self), sim.date + DateOffset(months=6))

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        pass


class DysenteryEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all enteric infection properties for population
    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(weeks=2))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = m.rng

        # -------------------------- UPDATING DYSENTERY DIARRHOEA STATUS OVER TIME ---------------------------
        # updating dysentery for children under 5 with current status 'none'

        eff_prob_ei_dysentery = pd.Series(m.base_incidence_dysentery,
                                          index=df.index[
                                              df.is_alive & (df.ei_diarrhoea_status == 'none') & (
                                                  df.age_years < 5)])

        eff_prob_ei_dysentery.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                  (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= m.rr_dysentery_age12to23mo
        eff_prob_ei_dysentery.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                  (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= m.rr_dysentery_age24to59mo
        eff_prob_ei_dysentery.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                  df.li_no_access_handwashing == False & (df.age_years < 5)] *= m.rr_dysentery_HHhandwashing
        eff_prob_ei_dysentery.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                  (df.has_hiv == True) & (df.age_years < 5)] *= m.rr_dysentery_HIV
        eff_prob_ei_dysentery.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                  df.malnutrition == True & (df.age_years < 5)] *= m.rr_dysentery_SAM
        eff_prob_ei_dysentery.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                  df.exclusive_breastfeeding == True & (
                                          df.age_exact_years <= 0.5)] *= m.rr_dysentery_excl_breast
        eff_prob_ei_dysentery.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                  df.continued_breastfeeding == True & (df.age_exact_years > 0.5) &
                                  (df.age_exact_years < 2)] *= m.rr_dysentery_cont_breast
        eff_prob_ei_dysentery.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                  df.li_no_clean_drinking_water == False & (df.age_years < 5)] *= m.rr_dysentery_clean_water
        eff_prob_ei_dysentery.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                  df.li_unimproved_sanitation == False & (df.age_years < 5)] *= m.rr_dysentery_improved_sanitation

        di_current_none_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'none') & (df.age_years < 5)]

        random_draw_01 = pd.Series(rng.random_sample(size=len(di_current_none_idx)),
                                   index=df.index[
                                       (df.age_years < 5) & df.is_alive & (df.ei_diarrhoea_status == 'none')])

        dfx = pd.concat([eff_prob_ei_dysentery, random_draw_01], axis=1)
        dfx.columns = ['eff_prob_ei_dysentery', 'random_draw_01']

        idx_incident_dysentery = dfx.index[dfx.eff_prob_ei_dysentery > dfx.random_draw_01]
        df.loc[idx_incident_dysentery, 'ei_diarrhoea_status'] = 'dysentery'

        # updating death due to dysentery

        eff_prob_death_dysentery = \
            pd.Series(m.r_death_dysentery,
                      index=df.index[df.is_alive & (df.ei_diarrhoea_status == 'dysentery') & (df.age_years < 5)])

        eff_prob_death_dysentery.loc[df.is_alive & (df.ei_diarrhoea_status == 'dysentery') &
                                     (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= \
            m.rr_death_dysentery_age12to23mo
        eff_prob_death_dysentery.loc[df.is_alive & (df.ei_diarrhoea_status == 'dysentery') &
                                     (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_death_dysentery_age24to59mo
        eff_prob_death_dysentery.loc[df.is_alive & (df.ei_diarrhoea_status == 'dysentery') &
                                     (df.has_hiv == True) & (df.age_exact_years < 5)] *= m.rr_death_dysentery_HIV
        eff_prob_death_dysentery.loc[df.is_alive & (df.ei_diarrhoea_status == 'dysentery') &
                                     df.malnutrition == True & (df.age_exact_years < 5)] *= m.rr_death_dysentery_SAM

        under5_dysentery_idx = df.index[(df.age_years < 5) & df.is_alive & (df.ei_diarrhoea_status == 'dysentery')]

        random_draw = pd.Series(rng.random_sample(size=len(under5_dysentery_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.ei_diarrhoea_status == 'dysentery')])

        dfx = pd.concat([eff_prob_death_dysentery, random_draw], axis=1)
        dfx.columns = ['eff_prob_death_dysentery', 'random_draw']
        idx_incident_death_dysentery = dfx.index[dfx.eff_prob_death_dysentery > dfx.random_draw]
        df.loc[idx_incident_death_dysentery, 'ei_diarrhoea_death'] = True

        after_death_dysentery_idx = df.index[(df.age_years < 5) & df.is_alive & (df.ei_diarrhoea_status == 'dysentery')]

        if self.sim.date + DateOffset(weeks=2):
            df.loc[after_death_dysentery_idx, 'ei_diarrhoea_status'] == 'none'

        if len(idx_incident_dysentery):
            for child in idx_incident_dysentery:
                logger.info('%s|start_dysentery|%s', self.sim.date,
                            {
                                'child_index': child,
                                'diarrhoea_type': df.at[child, 'ei_diarrhoea_status'],
                                'died': df.at[child, 'ei_diarrhoea_death']
                            })


class AcuteDiarrhoeaEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(weeks=2))

    def apply(self, population):

        df = population.props
        m = self.module
        rng = m.rng

        # -------------------------- UPDATING ACUTE WATERY DIARRHOEA STATUS OVER TIME ---------------------------
        # updating acute watery diarrhoea for children under 5 with current status 'none'

        eff_prob_ei_acute_diarrhoea = pd.Series(m.base_incidence_acute_diarrhoea,
                                                index=df.index[
                                                    df.is_alive & (df.ei_diarrhoea_status == 'none') & (
                                                        df.age_years < 5)])

        eff_prob_ei_acute_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                        (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= m.rr_acute_diarrhoea_age12to23mo
        eff_prob_ei_acute_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                        (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= m.rr_acute_diarrhoea_age24to59mo
        eff_prob_ei_acute_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                        df.li_no_access_handwashing == False & (df.age_years < 5)] *= m.rr_acute_diarrhoea_HHhandwashing
        eff_prob_ei_acute_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                        (df.has_hiv == True) & (df.age_years < 5)] *= m.rr_acute_diarrhoea_HIV
        eff_prob_ei_acute_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                  df.malnutrition == True & (df.age_years < 5)] *= m.rr_acute_diarrhoea_SAM
        eff_prob_ei_acute_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                  df.exclusive_breastfeeding == True & (
                                          df.age_exact_years <= 0.5)] *= m.rr_acute_diarrhoea_excl_breast
        eff_prob_ei_acute_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                        df.continued_breastfeeding == True & (df.age_exact_years > 0.5) &
                                        (df.age_exact_years < 2)] *= m.rr_acute_diarrhoea_cont_breast
        eff_prob_ei_acute_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                        df.li_no_clean_drinking_water == False & (df.age_years < 5)] *= m.rr_acute_diarrhoea_clean_water
        eff_prob_ei_acute_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                        df.li_unimproved_sanitation == False & (df.age_years < 5)] *= m.rr_acute_diarrhoea_improved_sanitation

        di_current_none_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'none') & (df.age_years < 5)]

        random_draw = pd.Series(rng.random_sample(size=len(di_current_none_idx)),
                                   index=df.index[
                                       (df.age_years < 5) & df.is_alive & (df.ei_diarrhoea_status == 'none')])

        dfx = pd.concat([eff_prob_ei_acute_diarrhoea, random_draw], axis=1)
        dfx.columns = ['eff_prob_ei_acute_diarrhoea', 'random_draw']

        idx_incident_acute_diarrhoea = dfx.index[dfx.eff_prob_ei_acute_diarrhoea > dfx.random_draw]
        df.loc[idx_incident_acute_diarrhoea, 'ei_diarrhoea_status'] = 'acute watery diarrhoea'

        # updating death due to acute watery diarrhoea

        eff_prob_death_acute_diarrhoea = \
            pd.Series(m.r_death_acute_diarrhoea,
                      index=df.index[
                          df.is_alive & (df.ei_diarrhoea_status == 'acute watery diarrhoea') & (df.age_years < 5)])
        eff_prob_death_acute_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'acute watery diarrhoea') &
                                           (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= \
            m.rr_death_acute_diar_age12to23mo
        eff_prob_death_acute_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'acute watery diarrhoea') &
                                           (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_death_acute_diar_age24to59mo
        eff_prob_death_acute_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'acute watery diarrhoea') &
                                           (df.has_hiv == True) & (df.age_exact_years < 5)] *= \
            m.rr_death_acute_diar_HIV
        eff_prob_death_acute_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'acute watery diarrhoea') &
                                           df.malnutrition == True & (df.age_exact_years < 5)] *= \
            m.rr_death_acute_diar_SAM

        under5_acute_diarrhoea_idx = df.index[(df.age_years < 5) & df.is_alive &
                                              (df.ei_diarrhoea_status == 'acute watery diarrhoea')]

        random_draw = pd.Series(rng.random_sample(size=len(under5_acute_diarrhoea_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.ei_diarrhoea_status == 'acute watery diarrhoea')])

        dfx = pd.concat([eff_prob_death_acute_diarrhoea, random_draw], axis=1)
        dfx.columns = ['eff_prob_death_acute_diarrhoea', 'random_draw']
        idx_incident_death_acute_diarrhoea = dfx.index[dfx.eff_prob_death_acute_diarrhoea > dfx.random_draw]
        df.loc[idx_incident_death_acute_diarrhoea, 'ei_diarrhoea_death'] = True

        after_death_acute_diarrhoea_idx = df.index[(df.age_years < 5) & df.is_alive &
                                                   (df.ei_diarrhoea_status == 'acute watery diarrhoea')]

        if self.sim.date + DateOffset(weeks=2):
            df.loc[after_death_acute_diarrhoea_idx, 'ei_diarrhoea_status'] == 'none'


class PersistentDiarrhoeaEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(weeks=2))

    def apply(self, population):

        df = population.props
        m = self.module
        rng = m.rng

        # -------------------------- UPDATING PERSISTENT DIARRHOEA STATUS OVER TIME ---------------------------
        # updating persistent diarrhoea for children under 5 with current status 'none'

        eff_prob_ei_persistent_diarrhoea = pd.Series(m.base_incidence_persistent_diarrhoea,
                                                     index=df.index[
                                                         df.is_alive & (df.ei_diarrhoea_status == 'none') & (
                                                             df.age_years < 5)])

        eff_prob_ei_persistent_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                             (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= m.rr_persistent_diarrhoea_age12to23mo
        eff_prob_ei_persistent_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                             (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= m.rr_persistent_diarrhoea_age24to59mo
        eff_prob_ei_persistent_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                             df.li_no_access_handwashing == False & (df.age_years < 5)] *= m.rr_persistent_diarrhoea_HHhandwashing
        eff_prob_ei_persistent_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                             (df.has_hiv == True) & (df.age_years < 5)] *= m.rr_persistent_diarrhoea_HIV
        eff_prob_ei_persistent_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                             df.malnutrition == True & (df.age_years < 5)] *= m.rr_persistent_diarrhoea_SAM
        eff_prob_ei_persistent_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                             df.exclusive_breastfeeding == True & (df.age_exact_years <= 0.5)] *= \
            m.rr_persistent_diarrhoea_excl_breast
        eff_prob_ei_persistent_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                             df.continued_breastfeeding == True & (df.age_exact_years > 0.5) &
                                             (df.age_exact_years < 2)] *= m.rr_persistent_diarrhoea_cont_breast
        eff_prob_ei_persistent_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                             df.li_no_clean_drinking_water == False & (df.age_years < 5)] *= m.rr_persistent_diarrhoea_clean_water
        eff_prob_ei_persistent_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'none') &
                                             df.li_unimproved_sanitation == False & (df.age_years < 5)] *= \
            m.rr_persistent_diarrhoea_improved_sanitation

        di_current_none_idx = \
            df.index[df.is_alive & (df.ei_diarrhoea_status == 'none') & (df.age_years < 5)]

        random_draw = pd.Series(rng.random_sample(size=len(di_current_none_idx)),
                                index=df.index[
                                    (df.age_years < 5) & df.is_alive & (df.ei_diarrhoea_status == 'none')])

        dfx = pd.concat([eff_prob_ei_persistent_diarrhoea, random_draw], axis=1)
        dfx.columns = ['eff_prob_ei_persistent_diarrhoea', 'random_draw']

        idx_incident_persistent_diarrhoea = dfx.index[dfx.eff_prob_ei_persistent_diarrhoea > dfx.random_draw]
        df.loc[idx_incident_persistent_diarrhoea, 'ei_diarrhoea_status'] = 'persistent diarrhoea'

        # updating death due to persistent diarrhoea

        eff_prob_death_persistent_diarrhoea = \
            pd.Series(m.r_death_persistent_diarrhoea,
                      index=df.index[
                          df.is_alive & (df.ei_diarrhoea_status == 'persistent diarrhoea') & (df.age_years < 5)])
        eff_prob_death_persistent_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'persistent diarrhoea') &
                                                (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= \
            m.rr_death_persistent_diar_age12to23mo
        eff_prob_death_persistent_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'persistent diarrhoea') &
                                                (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_death_persistent_diar_age24to59mo
        eff_prob_death_persistent_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'persistent diarrhoea') &
                                                (df.has_hiv == True) & (df.age_exact_years < 5)] *= \
            m.rr_death_persistent_diar_HIV
        eff_prob_death_persistent_diarrhoea.loc[df.is_alive & (df.ei_diarrhoea_status == 'persistent diarrhoea') &
                                                df.malnutrition == True & (df.age_exact_years < 5)] *= \
            m.rr_death_persistent_diar_SAM

        under5_persistent_diarrhoea_idx = df.index[(df.age_years < 5) & df.is_alive &
                                                   (df.ei_diarrhoea_status == 'persistent diarrhoea')]

        random_draw = pd.Series(rng.random_sample(size=len(under5_persistent_diarrhoea_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.ei_diarrhoea_status == 'persistent diarrhoea')])

        dfx = pd.concat([eff_prob_death_persistent_diarrhoea, random_draw], axis=1)
        dfx.columns = ['eff_prob_death_persistent_diarrhoea', 'random_draw']
        idx_incident_death_persistent_diarrhoea = dfx.index[dfx.eff_prob_death_persistent_diarrhoea > dfx.random_draw]
        df.loc[idx_incident_death_persistent_diarrhoea, 'ei_diarrhoea_death'] = True

        after_death_persistent_diarrhoea_idx = df.index[(df.age_years < 5) & df.is_alive &
                                                   (df.ei_diarrhoea_status == 'persistent diarrhoea')]

        death_this_period = df.index[(df.ei_diarrhoea_death == True)]
        for individual_id in death_this_period:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, 'ChildhoodDiarrhoea'),
                                    self.sim.date)

        if self.sim.date + DateOffset(weeks=4):
            df.loc[after_death_persistent_diarrhoea_idx, 'ei_diarrhoea_status'] == 'none'


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
