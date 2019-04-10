"""
Childhood Pneumonia module
Documentation: 04 - Methods Repository/Method_Child_RespiratoryInfection.xlsx
"""
import logging

import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ChildhoodPneumonia(Module):
    PARAMETERS = {
        'base_prev_pneumonia': Parameter
        (Types.REAL,
         'initial prevalence of non-severe pneumonia, among children aged 2-11 months,'
         'HIV negative, no SAM, not exclusively breastfeeding or continued breatfeeding, '
         'no household handwashing, no indoor air pollution, wealth level 3'
         ),
        'rp_pneumonia_agelt2mo': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for age < 2 months'
         ),
        'rp_pneumonia_age12to23mo': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for age 12 to 23 months'
         ),
        'rp_pneumonia_age24to59mo': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for age 24 to 59 months'
         ),
        'rp_pneumonia_HIV': Parameter
        (Types.REAL,
         'relative prevalence of pneumonia for HIV positive'
         ),
        'rp_pneumonia_SAM': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for severe acute malnutrition'
         ),
        'rp_pneumonia_excl_breast': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for exclusive breastfeeding upto 6 months'
         ),
        'rp_pneumonia_cont_breast': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for continued breastfeeding upto 23 months'
         ),
        'rp_pneumonia_HHhandwashing': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for household handwashing'
         ),
        'rp_pneumonia_IAP': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for indoor air pollution'
         ),
        'rp_pneumonia_wealth1': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for wealth level 1'
         ),
        'rp_pneumonia_wealth2': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for wealth level 2'
         ),
        'rp_pneumonia_wealth4': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for wealth level 4'
         ),
        'rp_pneumonia_wealth5': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for wealth level 5'
         ),
        'base_incidence_pneumonia': Parameter
        (Types.REAL,
         'baseline incidence of non-severe pneumonia, among children aged 2-11 months, '
         'HIV negative, no SAM, not exclusively breastfeeding or continued breatfeeding, '
         'no household handwashing, no indoor air pollution, wealth level 3'
         ),
        'rr_pneumonia_agelt2mo': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for age < 2 months'
         ),
        'rr_pneumonia_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for age 12 to 23 months'
         ),
        'rr_pneumonia_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for age 24 to 59 months'
         ),
        'rr_pneumonia_HIV': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for HIV positive'
         ),
        'rr_pneumonia_SAM': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for severe acute malnutrition'
         ),
        'rr_pneumonia_excl_breast': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for exclusive breastfeeding upto 6 months'
         ),
        'rr_pneumonia_cont_breast': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for continued breastfeeding upto 23 months'
         ),
        'rr_pneumonia_HHhandwashing': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for household handwashing'
         ),
        'rr_pneumonia_IAP': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for indoor air pollution'
         ),
        'rr_pneumonia_wealth1': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for wealth level 1'
         ),
        'rr_pneumonia_wealth2': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for wealth level 2'
         ),
        'rr_pneumonia_wealth4': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for wealth level 4'
         ),
        'rr_pneumonia_wealth5': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for wealth level 5'
         ),
        'base_prev_severe_pneumonia': Parameter
        (Types.REAL,
         'initial prevalence of severe pneumonia, among children aged 2-11 months,'
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no indoor air pollution, wealth level 3'
         ),
        'rp_severe_pneum_agelt2mo': Parameter
        (Types.REAL, 'relative prevalence of severe pneumonia for age <2 months'
         ),
        'rp_severe_pneum_age12to23mo': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for age 12 to 23 months'
         ),
        'rp_severe_pneum_age24to59mo': Parameter
        (Types.REAL, 'relative prevalence of severe pneumonia for age 24 to 59 months'
         ),
        'rp_severe_pneum_HIV': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for HIV positive status'
         ),
        'rp_severe_pneum_SAM': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for severe acute malnutrition'
         ),
        'rp_severe_pneum_excl_breast': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for exclusive breastfeeding upto 6 months'
         ),
        'rp_severe_pneum_cont_breast': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for continued breastfeeding upto 23 months'
         ),
        'rp_severe_pneum_HHhandwashing': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for household handwashing'
         ),
        'rp_severe_pneum_IAP': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for indoor air pollution'
         ),
        'rp_severe_pneum_wealth1': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for wealth level 1'
         ),
        'rp_severe_pneum_wealth2': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for wealth level 2'
         ),
        'rp_severe_pneum_wealth4': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for wealth level 4'
         ),
        'rp_severe_pneum_wealth5': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for wealth level 5'
         ),
        'base_incidence_severe_pneum': Parameter
        (Types.REAL,
         'baseline incidence of severe pneumonia, among children aged 2-11 months, '
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no indoor air pollution, wealth level 3'
         ),
        'rr_severe_pneum_agelt2mo': Parameter
        (Types.REAL,
         'relative rate of severe pneumonia for age <2 months'
         ),
        'rr_severe_pneum_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of severe pneumonia for age 12 to 23 months'
         ),
        'rr_severe_pneum_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of severe pneumonia for age 24 to 59 months'
         ),
        'rr_severe_pneum_HIV': Parameter
        (Types.REAL,
         'relative rate of severe pneumonia for HIV positive status'
         ),
        'rr_severe_pneum_SAM': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for severe acute malnutrition'
         ),
        'rr_severe_pneum_excl_breast': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for exclusive breastfeeding upto 6 months'
         ),
        'rr_severe_pneum_cont_breast': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for continued breastfeeding upto 23 months'
         ),
        'rr_severe_pneum_HHhandwashing': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for household handwashing'
         ),
        'rr_severe_pneum_IAP': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for indoor air pollution'
         ),
        'rr_severe_pneum_wealth1': Parameter
        (Types.REAL,
         'relative rate of severe pneumonia for wealth level 1'
         ),
        'rr_severe_pneum_wealth2': Parameter
        (Types.REAL,
         'relative rate of severe pneumonia for wealth level 2'
         ),
        'rr_severe_pneum_wealth4': Parameter
        (Types.REAL,
         'relative rate of severe pneumonia for wealth level 4'
         ),
        'rr_severe_pneum_wealth5': Parameter
        (Types.REAL,
         'relative rate of severe pneumonia for wealth level 5'
         ),
        'r_progress_to_severe_pneum': Parameter
        (Types.REAL,
         'probability of progressing from non-severe to severe pneumonia among children aged 2-11 months, '
         'HIV negative, no SAM, wealth level 3'
         ),
        'rr_progress_severe_pneum_agelt2mo': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for age <2 months'
         ),
        'rr_progress_severe_pneum_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for age 12 to 23 months'
         ),
        'rr_progress_severe_pneum_age24to59mo': Parameter
        (Types.REAL, 'relative rate of progression to severe pneumonia for age 24 to 59 months'
         ),
        'rr_progress_severe_pneum_HIV': Parameter
        (Types.REAL,
         'relative risk of progression to severe pneumonia for HIV positive status'
         ),
        'rr_progress_severe_pneum_SAM': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for severe acute malnutrition'
         ),
        'rr_progress_severe_pneum_wealth1': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for wealth level 1'
         ),
        'rr_progress_severe_pneum_wealth2': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for wealth level 2'
         ),
        'rr_progress_severe_pneum_wealth4': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for wealth level 4'
         ),
        'rr_progress_severe_pneum_wealth5': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for wealth level 5'
         ),
        'r_death_pneumonia': Parameter
        (Types.REAL,
         'death rate from pneumonia disease among children aged 2-11 months, '
         'HIV negative, no SAM, wealth level 3 '
         ),
        'rr_death_pneumonia_agelt2months': Parameter
        (Types.REAL,
         'relative rate of death from pneumonia disease for age < 2 months'
         ),
        'rr_death_pneumonia_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of death from pneumonia disease for age 12 to 23 months'
         ),
        'rr_death_pneumonia_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of death from pneumonia disease for age 24 to 59 months'
         ),
        'rr_death_pneumonia_HIV': Parameter
        (Types.REAL,
         'relative rate of death from pneumonia disease for HIV positive'
         ),
        'rr_death_pneumonia_SAM': Parameter
        (Types.REAL,
         'relative rate of death from pneumonia disease for severe acute malnutrition'
         ),
        'rr_death_pneumonia_wealth1': Parameter
        (Types.REAL,
         'relative rate of death from pneumonia disease for wealth level 1'
         ),
        'rr_death_pneumonia_wealth2': Parameter
        (Types.REAL,
         'relative rate of death from pneumonia disease for wealth level 2'
         ),
        'rr_death_pneumonia_wealth4': Parameter
        (Types.REAL,
         'relative rate of death from pneumonia disease for wealth level 4'
         ),
        'rr_death_pneumonia_wealth5': Parameter
        (Types.REAL,
         'relative rate of death from pneumonia disease for wealth level 5'
         ),
        'r_recovery_pneumonia': Parameter
        (Types.REAL,
         'recovery rate from pneumonia among children aged 2-11 months, '
         'HIV negative, no SAM,  '
         ),
        'rr_recovery_pneumonia_agelt2mo': Parameter
        (Types.REAL,
         'relative rate of recovery from pneumonia for age < 2 months'
         ),
        'rr_recovery_pneumonia_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of recovery from pneumonia for age between 12 to 23 months'
         ),
        'rr_recovery_pneumonia_age24to59mo': Parameter
        (Types.REAL, 'relative rate of recovery from pneumonia for age between 24 to 59 months'
         ),
        'rr_recovery_pneumonia_HIV': Parameter
        (Types.REAL,
         'relative rate of recovery from pneumonia for HIV positive status'
         ),
        'rr_recovery_pneumonia_SAM': Parameter
        (Types.REAL,
         'relative rate of recovery from pneumonia for severe acute malnutrition'
         ),
        'r_recovery_severe_pneumonia': Parameter
        (Types.REAL,
         'baseline recovery rate from severe pneumonia among children ages 2 to 11 months, '
         'HIV negative, no SAM, no indoor air pollution'
         ),
        'rr_recovery_severe_pneum_agelt2mo': Parameter
        (Types.REAL,
         'relative rate of recovery from severe pneumonia for age <2 months'
         ),
        'rr_recovery_severe_pneum_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of recovery from severe pneumonia for age between 12 to 23 months'
         ),
        'rr_recovery_severe_pneum_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of recovery from severe pneumonia for age between 24 to 59 months'
         ),
        'rr_recovery_severe_pneum_HIV': Parameter
        (Types.REAL,
         'relative rate of recovery from severe pneumonia for HIV positive status'
         ),
        'rr_recovery_severe_pneum_SAM': Parameter
        (Types.REAL,
         'relative rate of recovery from severe pneumonia for severe acute malnutrition'
         ),
        'init_prop_pneumonia_status': Parameter
        (Types.LIST,
         'initial proportions in ri_pneumonia_status categories '
         'for children aged 2-11 months, HIV negative, no SAM, '
         'not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no indoor air pollution, wealth level 3'
         )
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'ri_pneumonia_status': Property(Types.CATEGORICAL, 'lower respiratory infection - pneumonia status',
                                        categories=['none', 'pneumonia', 'severe pneumonia']),
        'has_hiv': Property(Types.BOOL, 'temporary property - has hiv'),
        'malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'indoor_air_pollution': Property(Types.BOOL, 'temporary property - indoor air pollution'),
        'exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
        'HHhandwashing': Property(Types.BOOL, 'temporary property - household handwashing'),
        'ri_pneumonia_death': Property(Types.BOOL, 'death from pneumonia disease')
    }

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module
        """
        p = self.parameters

        p['base_prev_pneumonia'] = 0.2
        p['rp_pneumonia_agelt2mo'] = 1.2
        p['rp_pneumonia_age12to23mo'] = 0.8
        p['rp_pneumonia_age24to59mo'] = 0.5
        p['rp_pneumonia_HIV'] = 1.4
        p['rp_pneumonia_SAM'] = 1.25
        p['rp_pneumonia_excl_breast'] = 0.5
        p['rp_pneumonia_cont_breast'] = 0.7
        p['rp_pneumonia_HHhandwashing'] = 0.5
        p['rp_pneumonia_IAP'] = 1.1
        p['rp_pneumonia_wealth1'] = 0.8
        p['rp_pneumonia_wealth2'] = 0.9
        p['rp_pneumonia_wealth4'] = 1.2
        p['rp_pneumonia_wealth5'] = 1.3
        p['base_incidence_pneumonia'] = 0.5
        p['rr_pneumonia_agelt2mo'] = 1.2
        p['rr_pneumonia_age12to23mo'] = 0.8
        p['rr_pneumonia_age24to59mo'] = 0.5
        p['rr_pneumonia_HIV'] = 1.4
        p['rr_pneumonia_SAM'] = 1.25
        p['rr_pneumonia_excl_breast'] = 0.6
        p['rr_pneumonia_cont_breast'] = 0.8
        p['rr_pneumonia_HHhandwashing'] = 0.5
        p['rr_pneumonia_IAP'] = 1.1
        p['rr_pneumonia_wealth1'] = 0.8
        p['rr_pneumonia_wealth2'] = 0.9
        p['rr_pneumonia_wealth4'] = 1.2
        p['rr_pneumonia_wealth5'] = 1.3
        p['base_prev_severe_pneumonia'] = 0.5
        p['rp_severe_pneum_agelt2mo'] = 1.3
        p['rp_severe_pneum_age12to23mo'] = 0.8
        p['rp_severe_pneum_age24to59mo'] = 0.5
        p['rp_severe_pneum_HIV'] = 1.3
        p['rp_severe_pneum_SAM'] = 1.3
        p['rp_severe_pneum_excl_breast'] = 0.5
        p['rp_severe_pneum_cont_breast'] = 0.7
        p['rp_severe_pneum_HHhandwashing'] = 0.8
        p['rp_severe_pneum_IAP'] = 1.1
        p['rp_severe_pneum_wealth1'] = 0.8
        p['rp_severe_pneum_wealth2'] = 0.9
        p['rp_severe_pneum_wealth4'] = 1.1
        p['rp_severe_pneum_wealth5'] = 1.2
        p['base_incidence_severe_pneum'] = 0.5
        p['rr_severe_pneum_agelt2mo'] = 1.3
        p['rr_severe_pneum_age12to23mo'] = 0.8
        p['rr_severe_pneum_age24to59mo'] = 0.5
        p['rr_severe_pneum_HIV'] = 1.3
        p['rr_severe_pneum_SAM'] = 1.3
        p['rr_severe_pneum_excl_breast'] = 0.6
        p['rr_severe_pneum_cont_breast'] = 0.8
        p['rr_severe_pneum_HHhandwashing'] = 0.3
        p['rr_severe_pneum_IAP'] = 1.1
        p['rr_severe_pneum_wealth1'] = 0.8
        p['rr_severe_pneum_wealth2'] = 0.9
        p['rr_severe_pneum_wealth4'] = 1.1
        p['rr_severe_pneum_wealth5'] = 1.2
        p['r_progress_to_severe_pneum'] = 0.05
        p['rr_progress_severe_pneum_agelt2mo'] = 1.3
        p['rr_progress_severe_pneum_age12to23mo'] = 0.9
        p['rr_progress_severe_pneum_age24to59mo'] = 0.6
        p['rr_progress_severe_pneum_HIV'] = 1.2
        p['rr_progress_severe_pneum_SAM'] = 1.1
        p['rr_progress_severe_pneum_wealth1'] = 0.8
        p['rr_progress_severe_pneum_wealth2'] = 0.9
        p['rr_progress_severe_pneum_wealth4'] = 1.1
        p['rr_progress_severe_pneum_wealth5'] = 1.3
        p['r_death_pneumonia'] = 0.5
        p['rr_death_pneumonia_agelt2mo'] = 1.2
        p['rr_death_pneumonia_age12to23mo'] = 0.8
        p['rr_death_pneumonia_age24to59mo'] = 0.04
        p['rr_death_pneumonia_HIV'] = 1.4
        p['rr_death_pneumonia_SAM'] = 1.3
        p['rr_death_pneumonia_wealth1'] = 0.7
        p['rr_death_pneumonia_wealth2'] = 0.8
        p['rr_death_pneumonia_wealth4'] = 1.2
        p['rr_death_pneumonia_wealth5'] = 1.3
        p['r_recovery_pneumonia'] = 0.5
        p['rr_recovery_pneumonia_agelt2mo'] = 0.3
        p['rr_recovery_pneumonia_age12to23mo'] = 0.7
        p['rr_recovery_pneumonia_age24to59mo'] = 0.8
        p['rr_recovery_pneumonia_HIV'] = 0.3
        p['rr_recovery_pneumonia_SAM'] = 0.4
        p['r_recovery_severe_pneumonia'] = 0.2
        p['rr_recovery_severe_pneum_agelt2mo'] = 0.6
        p['rr_recovery_severe_pneum_age12to23mo'] = 1.2
        p['rr_recovery_severe_pneum_age24to59mo'] = 1.5
        p['rr_recovery_severe_pneum_HIV'] = 0.5
        p['rr_recovery_severe_pneum_SAM'] = 0.6
        p['init_prop_pneumonia_status'] = [0.2, 0.1]

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

        # -------------------- DEFAULTS ------------------------------------------------------------

        df['ri_pneumonia_status'] = 'none'
        df['malnutrition'] = False
        df['has_HIV'] = False
        df['indoor_air_pollution'] = False
        df['HHhandwashing'] = False
        df['exclusive_breastfeeding'] = False
        df['continued_breastfeeding'] = False

        # -------------------- ASSIGN VALUES OF RESPIRATORY INFECTION STATUS AT BASELINE -----------

        under5_idx = df.index[(df.age_years < 5) & df.is_alive]

        # create data-frame of the probabilities of ri_pneumonia_status for children
        # aged 2-11 months, HIV negative, no SAM, no indoor air pollution
        p_pneumonia_status = pd.Series(self.init_prop_pneumonia_status[0], index=under5_idx)
        p_sev_pneum_status = pd.Series(self.init_prop_pneumonia_status[1], index=under5_idx)

        # create probabilities of pneumonia for all age under 5
        p_pneumonia_status.loc[
            (df.age_exact_years < 0.1667) & df.is_alive] *= self.rp_pneumonia_agelt2mo
        p_pneumonia_status.loc[
            (df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] *= self.rp_pneumonia_age12to23mo
        p_pneumonia_status.loc[
            (df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] *= self.rp_pneumonia_age24to59mo
        p_pneumonia_status.loc[
            (df.has_hiv == True) & (df.age_years < 5) & df.is_alive] *= self.rp_pneumonia_HIV
        p_pneumonia_status.loc[
            (df.malnutrition == True) & (df.age_years < 5) & df.is_alive] *= self.rp_pneumonia_SAM
        p_pneumonia_status.loc[
            (df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) & df.is_alive] \
            *= self.rp_pneumonia_excl_breast
        p_pneumonia_status.loc[
            (df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) & (df.age_exact_years < 2) &
            df.is_alive] *= self.rp_pneumonia_cont_breast
        p_pneumonia_status.loc[
            (df.indoor_air_pollution == True) & (df.age_years < 5) & df.is_alive] *= self.rp_pneumonia_IAP
        p_pneumonia_status.loc[
            (df.li_wealth == 1) & (df.age_years < 5) & df.is_alive] *= self.rp_pneumonia_wealth1
        p_pneumonia_status.loc[
            (df.li_wealth == 2) & (df.age_years < 5) & df.is_alive] *= self.rp_pneumonia_wealth2
        p_pneumonia_status.loc[
            (df.li_wealth == 4) & (df.age_years < 5) & df.is_alive] *= self.rp_pneumonia_wealth4
        p_pneumonia_status.loc[
            (df.li_wealth == 5) & (df.age_years < 5) & df.is_alive] *= self.rp_pneumonia_wealth5

        # create probabilities of severe pneumonia for all age under 5
        p_sev_pneum_status.loc[
            (df.age_exact_years < 0.1667) & df.is_alive] *= self.rp_severe_pneum_agelt2mo
        p_sev_pneum_status.loc[
            (df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] *= self.rp_severe_pneum_age12to23mo
        p_sev_pneum_status.loc[
            (df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] *= self.rp_severe_pneum_age24to59mo
        p_sev_pneum_status.loc[
            (df.has_hiv == True) & (df.age_years < 5) & df.is_alive] *= self.rp_severe_pneum_HIV
        p_sev_pneum_status.loc[
            (df.malnutrition == True) & (df.age_years < 5) & df.is_alive] *= self.rp_severe_pneum_SAM
        p_sev_pneum_status.loc[
            (df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) & df.is_alive] \
            *= self.rp_severe_pneum_excl_breast
        p_sev_pneum_status.loc[
            (df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) & (df.age_exact_years < 2) &
            df.is_alive] *= self.rp_severe_pneum_cont_breast
        p_sev_pneum_status.loc[
            (df.indoor_air_pollution == True) & (df.age_years < 5) & df.is_alive] *= self.rp_severe_pneum_IAP
        p_sev_pneum_status.loc[
            (df.li_wealth == 1) & (df.age_years < 5) & df.is_alive] *= self.rp_severe_pneum_wealth1
        p_sev_pneum_status.loc[
            (df.li_wealth == 2) & (df.age_years < 5) & df.is_alive] *= self.rp_severe_pneum_wealth2
        p_sev_pneum_status.loc[
            (df.li_wealth == 4) & (df.age_years < 5) & df.is_alive] *= self.rp_severe_pneum_wealth4
        p_sev_pneum_status.loc[
            (df.li_wealth == 5) & (df.age_years < 5) & df.is_alive] *= self.rp_severe_pneum_wealth5

        random_draw = pd.Series(rng.random_sample(size=len(under5_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive])

        # create a temporary dataframe called dfx to hold values of probabilities and random draw
        dfx = pd.concat([p_pneumonia_status, p_sev_pneum_status, random_draw], axis=1)
        dfx.columns = ['p_pneumonia', 'p_severe_pneumonia', 'random_draw']

        dfx['p_none'] = 1 - (dfx.p_pneumonia + dfx.p_severe_pneumonia)

        # based on probabilities of being in each category, define cut-offs to determine status from
        # random draw uniform(0,1)

        # assign baseline values of ri_resp_infection_stat based on probabilities and value of random draw

        idx_none = dfx.index[dfx.p_none > dfx.random_draw]
        idx_pneumonia = dfx.index[(dfx.p_none < dfx.random_draw) & ((dfx.p_none + dfx.p_pneumonia) > dfx.random_draw)]
        idx_severe_pneumonia = dfx.index[((dfx.p_none + dfx.p_pneumonia) < dfx.random_draw) &
                                         (dfx.p_none + dfx.p_pneumonia + dfx.p_severe_pneumonia) > dfx.random_draw]

        df.loc[idx_none, 'ri_pneumonia_status'] = 'none'
        df.loc[idx_pneumonia, 'ri_pneumonia_status'] = 'pneumonia'
        df.loc[idx_severe_pneumonia, 'ri_pneumonia_status'] = 'severe pneumonia'

    def initialise_simulation(self, sim):
        """
        Get ready for simulation start.
        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event
        event = RespInfectionEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(weeks=1))

        # add an event to log to screen
        sim.schedule_event(RespInfectionLoggingEvent(self), sim.date + DateOffset(weeks=1))

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        df.at[child_id, 'ri_pneumonia_status'] = 'none'


class RespInfectionEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all Respiratory Infection properties for population
    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module):
        """schedule to run every 7 days
        note: if change this offset from 1 week need to consider code conditioning on age.years_exact
        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.
        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(days=7))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = m.rng

        df['ri_pneumonia_death'] = False

        # ------------------- UPDATING OF LOWER RESPIRATORY INFECTION - PNEUMONIA STATUS OVER TIME -------------------

        # updating for children under 5 with current status 'none'

        pn_current_none_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'none') & (df.age_years < 5)]
        pn_current_none_agelt2mo_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'none') & (df.age_exact_years < 0.1667)]
        pn_current_none_age12to23mo_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     (df.age_exact_years >= 1) & (df.age_exact_years < 2)]
        pn_current_none_age24to59mo_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     (df.age_exact_years >= 2) & (df.age_exact_years < 5)]
        pn_current_none_handwashing_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     df.HHhandwashing & (df.age_years < 5)]
        pn_current_none_HIV_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     (df.has_hiv) & (df.age_years < 5)]
        pn_current_none_SAM_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     df.malnutrition & (df.age_years < 5)]
        pn_current_none_excl_breast_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     df.exclusive_breastfeeding & (df.age_exact_years <= 0.5)]
        pn_current_none_cont_breast_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     df.continued_breastfeeding & (df.age_exact_years > 0.5) & (df.age_exact_years < 2)]
        pn_current_none_IAP_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     df.indoor_air_pollution & (df.age_years < 5)]
        pn_current_none_wealth1_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     (df.li_wealth == 1) & (df.age_years < 5)]
        pn_current_none_wealth2_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     (df.li_wealth == 2) & (df.age_years < 5)]
        pn_current_none_wealth4_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     (df.li_wealth == 4) & (df.age_years < 5)]
        pn_current_none_wealth5_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                     (df.li_wealth == 5) & (df.age_years < 5)]

        eff_prob_ri_pneumonia = pd.Series(m.base_incidence_pneumonia,
                                          index=df.index[
                                              df.is_alive & (df.ri_pneumonia_status == 'none') & (
                                                  df.age_years < 5)])

        eff_prob_ri_pneumonia.loc[pn_current_none_agelt2mo_idx] *= m.rr_pneumonia_agelt2mo
        eff_prob_ri_pneumonia.loc[pn_current_none_age12to23mo_idx] *= m.rr_pneumonia_age12to23mo
        eff_prob_ri_pneumonia.loc[pn_current_none_age24to59mo_idx] *= m.rr_pneumonia_age24to59mo
        eff_prob_ri_pneumonia.loc[pn_current_none_handwashing_idx] *= m.rr_pneumonia_HHhandwashing
        eff_prob_ri_pneumonia.loc[pn_current_none_HIV_idx] *= m.rr_pneumonia_HIV
        eff_prob_ri_pneumonia.loc[pn_current_none_SAM_idx] *= m.rr_pneumonia_SAM
        eff_prob_ri_pneumonia.loc[pn_current_none_excl_breast_idx] *= m.rr_pneumonia_excl_breast
        eff_prob_ri_pneumonia.loc[pn_current_none_cont_breast_idx] *= m.rr_pneumonia_cont_breast
        eff_prob_ri_pneumonia.loc[pn_current_none_IAP_idx] *= m.rr_pneumonia_IAP
        eff_prob_ri_pneumonia.loc[pn_current_none_wealth1_idx] *= m.rr_pneumonia_wealth1
        eff_prob_ri_pneumonia.loc[pn_current_none_wealth2_idx] *= m.rr_pneumonia_wealth2
        eff_prob_ri_pneumonia.loc[pn_current_none_wealth4_idx] *= m.rr_pneumonia_wealth4
        eff_prob_ri_pneumonia.loc[pn_current_none_wealth5_idx] *= m.rr_pneumonia_wealth5

        eff_prob_ri_severe_pneumonia = pd.Series(m.base_incidence_severe_pneum,
                                                 index=df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                                                (df.age_years < 5)])
        eff_prob_ri_severe_pneumonia.loc[pn_current_none_agelt2mo_idx] *= m.rr_severe_pneum_agelt2mo
        eff_prob_ri_severe_pneumonia.loc[pn_current_none_age12to23mo_idx] *= m.rr_severe_pneum_age12to23mo
        eff_prob_ri_severe_pneumonia.loc[pn_current_none_age24to59mo_idx] *= m.rr_severe_pneum_age24to59mo
        eff_prob_ri_severe_pneumonia.loc[pn_current_none_handwashing_idx] *= m.rr_severe_pneum_HHhandwashing
        eff_prob_ri_severe_pneumonia.loc[pn_current_none_HIV_idx] *= m.rr_severe_pneum_HIV
        eff_prob_ri_severe_pneumonia.loc[pn_current_none_SAM_idx] *= m.rr_severe_pneum_SAM
        eff_prob_ri_severe_pneumonia.loc[pn_current_none_excl_breast_idx] *= m.rr_severe_pneum_excl_breast
        eff_prob_ri_severe_pneumonia.loc[pn_current_none_cont_breast_idx] *= m.rr_severe_pneum_cont_breast
        eff_prob_ri_severe_pneumonia.loc[pn_current_none_IAP_idx] *= m.rr_severe_pneum_IAP
        eff_prob_ri_severe_pneumonia.loc[pn_current_none_wealth1_idx] *= m.rr_pneumonia_wealth1
        eff_prob_ri_severe_pneumonia.loc[pn_current_none_wealth2_idx] *= m.rr_pneumonia_wealth2
        eff_prob_ri_severe_pneumonia.loc[pn_current_none_wealth4_idx] *= m.rr_pneumonia_wealth4
        eff_prob_ri_severe_pneumonia.loc[pn_current_none_wealth5_idx] *= m.rr_pneumonia_wealth5

        random_draw_01 = pd.Series(rng.random_sample(size=len(pn_current_none_idx)),
                                   index=df.index[
                                       (df.age_years < 5) & df.is_alive & (df.ri_pneumonia_status == 'none')])

        dfx = pd.concat([eff_prob_ri_pneumonia, eff_prob_ri_severe_pneumonia, random_draw_01], axis=1)
        dfx.columns = ['eff_prob_ri_pneumonia', 'eff_prob_ri_severe_pneumonia', 'random_draw_01']

        dfx['ri_none'] = 1 - (dfx.eff_prob_ri_pneumonia + dfx.eff_prob_ri_severe_pneumonia)

        idx_incident_none = dfx.index[dfx.eff_prob_ri_pneumonia > dfx.random_draw_01]
        idx_incident_pneumonia = dfx.index[
            (dfx.ri_none < dfx.random_draw_01) & ((dfx.ri_none + dfx.eff_prob_ri_pneumonia) > dfx.random_draw_01)]
        idx_incident_severe_pneumonia = dfx.index[((dfx.ri_none + dfx.eff_prob_ri_pneumonia) < dfx.random_draw_01) &
                                                  (dfx.ri_none + dfx.eff_prob_ri_pneumonia +
                                                   dfx.eff_prob_ri_severe_pneumonia) > dfx.random_draw_01]

        df.loc[idx_incident_none, 'ri_pneumonia_status'] = 'none'
        df.loc[idx_incident_pneumonia, 'ri_pneumonia_status'] = 'pneumonia'
        df.loc[idx_incident_severe_pneumonia, 'ri_pneumonia_status'] = 'severe pneumonia'

        # ---------- updating for children under 5 with current status 'pneumonia' to 'severe pneumonia'----------

        pn_current_pneumonia_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') & (df.age_years < 5)]
        pn_current_pneumonia_agelt2mo_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') & (df.age_exact_years < 0.1667)]
        pn_current_pneumonia_age12to23mo_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                     (df.age_exact_years >= 1) & (df.age_exact_years < 2)]
        pn_current_pneumonia_age24to59mo_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                     (df.age_exact_years >= 2) & (df.age_exact_years < 5)]
        pn_current_pneumonia_HIV_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                     df.has_hiv & (df.age_years < 5)]
        pn_current_pneumonia_SAM_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                     df.malnutrition & (df.age_years < 5)]
        pn_current_pneumonia_wealth1_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                     (df.li_wealth == 1) & (df.age_years < 5)]
        pn_current_pneumonia_wealth2_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                     (df.li_wealth == 2) & (df.age_years < 5)]
        pn_current_pneumonia_wealth4_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                     (df.li_wealth == 4) & (df.age_years < 5)]
        pn_current_pneumonia_wealth5_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                     (df.li_wealth == 5) & (df.age_years < 5)]

        eff_prob_prog_severe_pneumonia = pd.Series(m.r_progress_to_severe_pneum,
                                                   index=df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia')
                                                                  & (df.age_years < 5)])
        eff_prob_prog_severe_pneumonia.loc[pn_current_pneumonia_agelt2mo_idx] *= \
            m.rr_progress_severe_pneum_agelt2mo
        eff_prob_prog_severe_pneumonia.loc[pn_current_pneumonia_age12to23mo_idx] *= \
            m.rr_progress_severe_pneum_age12to23mo
        eff_prob_prog_severe_pneumonia.loc[pn_current_pneumonia_age24to59mo_idx] *= \
            m.rr_progress_severe_pneum_age24to59mo
        eff_prob_prog_severe_pneumonia.loc[pn_current_pneumonia_HIV_idx] *= \
            m.rr_progress_severe_pneum_HIV
        eff_prob_prog_severe_pneumonia.loc[pn_current_pneumonia_SAM_idx] *= \
            m.rr_progress_severe_pneum_SAM
        eff_prob_prog_severe_pneumonia.loc[pn_current_pneumonia_wealth1_idx] *= \
            m.rr_progress_severe_pneum_wealth1
        eff_prob_prog_severe_pneumonia.loc[pn_current_pneumonia_wealth2_idx] *= \
            m.rr_progress_severe_pneum_wealth2
        eff_prob_prog_severe_pneumonia.loc[pn_current_pneumonia_wealth4_idx] *= \
            m.rr_progress_severe_pneum_wealth4
        eff_prob_prog_severe_pneumonia.loc[pn_current_pneumonia_wealth5_idx] *= \
            m.rr_progress_severe_pneum_wealth5

        random_draw_02 = pd.Series(rng.random_sample(size=len(pn_current_pneumonia_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.ri_pneumonia_status == 'pneumonia')])
        dfx = pd.concat([eff_prob_ri_severe_pneumonia, random_draw_02], axis=1)
        dfx.columns = ['eff_prob_prog_severe_pneumonia', 'random_draw_02']
        idx_ri_progress_severe_pneumonia = dfx.index[dfx.eff_prob_prog_severe_pneumonia > dfx.random_draw_02]
        df.loc[idx_ri_progress_severe_pneumonia, 'ri_pneumonia_status'] = 'severe pneumonia'

        # -------------------- UPDATING OF RI_PNEUMONIA_STATUS RECOVERY OVER TIME --------------------------------
        # recovery from non-severe pneumonia
        pn1_current_pneumonia_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') & (df.age_years < 5)]
        pn1_current_pneumonia_agelt2mo_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') & (df.age_exact_years < 0.1667)]
        pn1_current_pneumonia_age12to23mo_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                     (df.age_exact_years >= 1) & (df.age_exact_years < 2)]
        pn1_current_pneumonia_age24to59mo_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                     (df.age_exact_years >= 2) & (df.age_exact_years < 5)]
        pn1_current_pneumonia_HIV_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                     df.has_hiv & (df.age_years < 5)]
        pn1_current_pneumonia_SAM_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                     df.malnutrition & (df.age_years < 5)]

        eff_prob_recovery_pneumonia = pd.Series(m.r_recovery_pneumonia,
                                                index=df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia')
                                                               & (df.age_years < 5)])

        eff_prob_recovery_pneumonia.loc[pn1_current_pneumonia_agelt2mo_idx] *= \
            m.rr_recovery_pneumonia_agelt2mo
        eff_prob_recovery_pneumonia.loc[pn1_current_pneumonia_age12to23mo_idx] *= \
            m.rr_recovery_pneumonia_age12to23mo
        eff_prob_recovery_pneumonia.loc[pn1_current_pneumonia_age24to59mo_idx] *= \
            m.rr_recovery_pneumonia_age24to59mo
        eff_prob_recovery_pneumonia.loc[pn1_current_pneumonia_HIV_idx] *= \
            m.rr_recovery_pneumonia_HIV
        eff_prob_recovery_pneumonia.loc[pn1_current_pneumonia_SAM_idx] *= \
            m.rr_recovery_pneumonia_SAM

        random_draw_03 = pd.Series(rng.random_sample(size=len(pn1_current_pneumonia_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.ri_pneumonia_status == 'pneumonia')])
        dfx = pd.concat([eff_prob_recovery_pneumonia, random_draw_03], axis=1)
        dfx.columns = ['eff_prob_recovery_pneumonia', 'random_draw_03']
        idx_recovery_pneumonia = dfx.index[dfx.eff_prob_recovery_pneumonia > dfx.random_draw_03]
        df.loc[idx_recovery_pneumonia, 'ri_pneumonia_status'] = 'none'

        # recovery from severe pneumonia

        pn_current_severe_pneumonia_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') & (df.age_years < 5)]
        pn_current_severe_pneum_agelt2mo_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') & (df.age_exact_years < 0.1667)]
        pn_current_severe_pneum_age12to23mo_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') &
                     (df.age_exact_years >= 1) & (df.age_exact_years < 2)]
        pn_current_severe_pneum_age24to59mo_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') &
                     (df.age_exact_years >= 2) & (df.age_exact_years < 5)]
        pn_current_severe_pneum_HIV_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') &
                     df.has_hiv & (df.age_years < 5)]
        pn_current_severe_pneum_SAM_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') &
                     df.malnutrition & (df.age_years < 5)]

        eff_prob_recovery_severe_pneum = \
            pd.Series(m.r_recovery_severe_pneumonia,
                      index=df.index[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') & (df.age_years < 5)])

        eff_prob_recovery_severe_pneum.loc[pn_current_severe_pneum_agelt2mo_idx] *= \
            m.rr_recovery_severe_pneum_agelt2mo
        eff_prob_recovery_severe_pneum.loc[pn_current_severe_pneum_age12to23mo_idx] *= \
            m.rr_recovery_severe_pneum_age12to23mo
        eff_prob_recovery_severe_pneum.loc[pn_current_severe_pneum_age24to59mo_idx] *= \
            m.rr_recovery_severe_pneum_age24to59mo
        eff_prob_recovery_severe_pneum.loc[pn_current_severe_pneum_HIV_idx] *= \
            m.rr_recovery_severe_pneum_HIV
        eff_prob_recovery_severe_pneum.loc[pn_current_severe_pneum_SAM_idx] *= \
            m.rr_recovery_severe_pneum_SAM

        random_draw_04 = pd.Series(rng.random_sample(size=len(pn_current_severe_pneumonia_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.ri_pneumonia_status == 'severe pneumonia')])
        dfx = pd.concat([eff_prob_recovery_severe_pneum, random_draw_04], axis=1)
        dfx.columns = ['eff_prob_recovery_severe_pneum', 'random_draw_04']
        idx_recovery_pneumonia = dfx.index[dfx.eff_prob_recovery_severe_pneum > dfx.random_draw_04]
        df.loc[idx_recovery_pneumonia, 'ri_pneumonia_status'] = 'none'

        # ---------------------------- DEATH FROM PNEUMONIA DISEASE ---------------------------------------

        pn1_current_severe_pneumonia_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') & (df.age_years < 5)]
        pn1_current_severe_pneum_agelt2mo_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') & (df.age_exact_years < 0.1667)]
        pn1_current_severe_pneum_age12to23mo_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') &
                     (df.age_exact_years >= 1) & (df.age_exact_years < 2)]
        pn1_current_severe_pneum_age24to59mo_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') &
                     (df.age_exact_years >= 2) & (df.age_exact_years < 5)]
        pn1_current_severe_pneum_HIV_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') &
                     df.has_hiv & (df.age_years < 5)]
        pn1_current_severe_pneum_SAM_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') &
                     df.malnutrition & (df.age_years < 5)]

        eff_prob_death_pneumonia = \
            pd.Series(m.r_death_pneumonia,
                      index=df.index[df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia') & (df.age_years < 5)])
        eff_prob_death_pneumonia.loc[pn1_current_severe_pneum_agelt2mo_idx] *= \
            m.rr_death_pneumonia_agelt2mo
        eff_prob_death_pneumonia.loc[pn1_current_severe_pneum_age12to23mo_idx] *= \
            m.rr_death_pneumonia_age12to23mo
        eff_prob_death_pneumonia.loc[pn1_current_severe_pneum_age24to59mo_idx] *= \
            m.rr_death_pneumonia_age24to59mo
        eff_prob_death_pneumonia.loc[pn1_current_severe_pneum_HIV_idx] *= \
            m.rr_death_pneumonia_HIV
        eff_prob_death_pneumonia.loc[pn1_current_severe_pneum_SAM_idx] *= \
            m.rr_death_pneumonia_SAM

        random_draw_05 = pd.Series(rng.random_sample(size=len(pn1_current_severe_pneumonia_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.ri_pneumonia_status == 'severe pneumonia')])

        dfx = pd.concat([eff_prob_death_pneumonia, random_draw_05], axis=1)
        dfx.columns = ['eff_prob_death_pneumonia', 'random_draw_05']

        dfx['pneumonia_death'] = False
        dfx.loc[dfx.eff_prob_death_pneumonia > dfx.random_draw_05, 'pneumonia_death'] = True
        df.loc[pn1_current_severe_pneumonia_idx, 'ri_pneumonia_death'] = dfx['pneumonia_death']

        death_this_period = df.index[df.ri_pneumonia_death]
        for individual_id in death_this_period:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, 'ChildhoodPneumonia'),
                                    self.sim.date)

        logger.debug('%s|person_one|%s',
                     self.sim.date,
                     df.loc[0].to_dict())

class RespInfectionLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Handles lifestyle logging"""

    def __init__(self, module):
        """schedule logging to repeat every 3 months
        """
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        # get some summary statistics
