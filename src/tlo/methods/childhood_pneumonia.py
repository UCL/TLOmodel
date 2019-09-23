"""
Childhood Pneumonia module
Documentation: 04 - Methods Repository/Method_Child_RespiratoryInfection.xlsx
"""
import logging

import numpy as np
import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin
from tlo.methods import demography
from tlo.methods.iCCM import HSI_Sick_Child_Seeks_Care_From_HSA

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ChildhoodPneumonia(Module):
    PARAMETERS = {
        'base_prev_pneumonia': Parameter
        (Types.REAL,
         'initial prevalence of non-severe pneumonia, among children aged 2-11 months,'
         'HIV negative, no SAM, no exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing practice, indoor air pollution'
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
        'base_incidence_pneumonia': Parameter
        (Types.REAL,
         'baseline incidence of non-severe pneumonia, among children aged 2-11 months, '
         'HIV negative, no SAM, not exclusively breastfeeding or continued breatfeeding, '
         'no household handwashing, no indoor air pollution, wealth level 3'
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
        'base_prev_very_severe_pneumonia': Parameter
        (Types.REAL,
         'initial prevalence of severe pneumonia, among children aged 2-11 months,'
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no indoor air pollution, wealth level 3'
         ),
        'rp_very_severe_pneum_agelt2mo': Parameter
        (Types.REAL, 'relative prevalence of severe pneumonia for age <2 months'
         ),
        'rp_very_severe_pneum_age12to23mo': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for age 12 to 23 months'
         ),
        'rp_very_severe_pneum_age24to59mo': Parameter
        (Types.REAL, 'relative prevalence of severe pneumonia for age 24 to 59 months'
         ),
        'rp_very_severe_pneum_HIV': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for HIV positive status'
         ),
        'rp_very_severe_pneum_SAM': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for severe acute malnutrition'
         ),
        'rp_very_severe_pneum_excl_breast': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for exclusive breastfeeding upto 6 months'
         ),
        'rp_very_severe_pneum_cont_breast': Parameter
        (Types.REAL,
         'relative prevalence of non-severe pneumonia for continued breastfeeding upto 23 months'
         ),
        'rp_very_severe_pneum_HHhandwashing': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for household handwashing'
         ),
        'rp_very_severe_pneum_IAP': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for indoor air pollution'
         ),
        'base_incidence_very_severe_pneum': Parameter
        (Types.REAL,
         'baseline incidence of severe pneumonia, among children aged 2-11 months, '
         'HIV negative, no SAM, not exclusively breastfeeding or continued breastfeeding, '
         'no household handwashing, no indoor air pollution, wealth level 3'
         ),
        'rr_very_severe_pneum_agelt2mo': Parameter
        (Types.REAL,
         'relative rate of severe pneumonia for age <2 months'
         ),
        'rr_very_severe_pneum_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of severe pneumonia for age 12 to 23 months'
         ),
        'rr_very_severe_pneum_age24to59mo': Parameter
        (Types.REAL,
         'relative rate of severe pneumonia for age 24 to 59 months'
         ),
        'rr_very_severe_pneum_HIV': Parameter
        (Types.REAL,
         'relative rate of severe pneumonia for HIV positive status'
         ),
        'rr_very_severe_pneum_SAM': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for severe acute malnutrition'
         ),
        'rr_very_severe_pneum_excl_breast': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for exclusive breastfeeding upto 6 months'
         ),
        'rr_very_severe_pneum_cont_breast': Parameter
        (Types.REAL,
         'relative rate of non-severe pneumonia for continued breastfeeding upto 23 months'
         ),
        'rr_very_severe_pneum_HHhandwashing': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for household handwashing'
         ),
        'rr_very_severe_pneum_IAP': Parameter
        (Types.REAL,
         'relative prevalence of severe pneumonia for indoor air pollution'
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
                                        categories=['none', 'pneumonia', 'severe pneumonia', 'very severe pneumonia']),
        'has_hiv': Property(Types.BOOL, 'temporary property - has hiv'),
        'malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
        'ri_pneumonia_death': Property(Types.BOOL, 'death from pneumonia disease'),
        'date_of_acquiring_pneumonia': Property(Types.DATE, 'date of acquiring pneumonia infection'),
        'pn_fever': Property(Types.BOOL, 'fever from non-severe pneumonia, severe pneumonia or very severe pneumonia'),
        'pn_cough': Property(Types.BOOL, 'cough from non-severe pneumonia, severe pneumonia or very severe pneumonia'),
        'pn_difficult_breathing': Property(Types.BOOL, 'difficult breathing from non-severe pneumonia, severe pneumonia or very severe pneumonia'),
        'pn_fast_breathing': Property(Types.BOOL, 'fast breathing from non-severe pneumonia'),
        'pn_chest_indrawing': Property(Types.BOOL, 'chest indrawing from severe pneumonia or very severe pneumonia'),
        'pn_any_general_danger_sign': Property(Types.BOOL, 'any danger sign - lethargic/uncounscious, not able to drink/breastfeed, convulsions and vomiting everything'),
        'pn_stridor_in_calm_child': Property(Types.BOOL, 'stridor in calm child from very severe pneumonia')
    }

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module
        """
        p = self.parameters

        p['base_prev_pneumonia'] = 0.4
        p['rp_pneumonia_age12to23mo'] = 0.8
        p['rp_pneumonia_age24to59mo'] = 0.5
        p['rp_pneumonia_HIV'] = 1.4
        p['rp_pneumonia_SAM'] = 1.25
        p['rp_pneumonia_excl_breast'] = 0.5
        p['rp_pneumonia_cont_breast'] = 0.7
        p['rp_pneumonia_HHhandwashing'] = 0.5
        p['rp_pneumonia_IAP'] = 1.1
        p['base_incidence_pneumonia'] = 0.5
        p['rr_pneumonia_age12to23mo'] = 0.8
        p['rr_pneumonia_age24to59mo'] = 0.5
        p['rr_pneumonia_HIV'] = 1.4
        p['rr_pneumonia_SAM'] = 1.25
        p['rr_pneumonia_excl_breast'] = 0.6
        p['rr_pneumonia_cont_breast'] = 0.8
        p['rr_pneumonia_HHhandwashing'] = 0.5
        p['rr_pneumonia_IAP'] = 1.1
        p['base_prev_severe_pneumonia'] = 0.4
        p['rp_severe_pneum_agelt2mo'] = 1.3
        p['rp_severe_pneum_age12to23mo'] = 0.8
        p['rp_severe_pneum_age24to59mo'] = 0.5
        p['rp_severe_pneum_HIV'] = 1.3
        p['rp_severe_pneum_SAM'] = 1.3
        p['rp_severe_pneum_excl_breast'] = 0.5
        p['rp_severe_pneum_cont_breast'] = 0.7
        p['rp_severe_pneum_HHhandwashing'] = 0.8
        p['rp_severe_pneum_IAP'] = 1.1
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
        p['base_prev_very_severe_pneumonia'] = 0.4
        p['rp_very_severe_pneum_agelt2mo'] = 1.3
        p['rp_very_severe_pneum_age12to23mo'] = 0.8
        p['rp_very_severe_pneum_age24to59mo'] = 0.5
        p['rp_very_severe_pneum_HIV'] = 1.3
        p['rp_very_severe_pneum_SAM'] = 1.3
        p['rp_very_severe_pneum_excl_breast'] = 0.5
        p['rp_very_severe_pneum_cont_breast'] = 0.7
        p['rp_very_severe_pneum_HHhandwashing'] = 0.8
        p['rp_very_severe_pneum_IAP'] = 1.1
        p['base_incidence_very_severe_pneum'] = 0.5
        p['rr_very_severe_pneum_agelt2mo'] = 1.3
        p['rr_very_severe_pneum_age12to23mo'] = 0.8
        p['rr_very_severe_pneum_age24to59mo'] = 0.5
        p['rr_very_severe_pneum_HIV'] = 1.3
        p['rr_very_severe_pneum_SAM'] = 1.3
        p['rr_very_severe_pneum_excl_breast'] = 0.6
        p['rr_very_severe_pneum_cont_breast'] = 0.8
        p['rr_very_severe_pneum_HHhandwashing'] = 0.3
        p['rr_very_severe_pneum_IAP'] = 1.1
        p['r_progress_to_severe_pneum'] = 0.05
        p['rr_progress_severe_pneum_agelt2mo'] = 1.3
        p['rr_progress_severe_pneum_age12to23mo'] = 0.9
        p['rr_progress_severe_pneum_age24to59mo'] = 0.6
        p['rr_progress_severe_pneum_HIV'] = 1.2
        p['rr_progress_severe_pneum_SAM'] = 1.1
        p['r_progress_to_very_severe_pneum'] = 0.5
        p['rr_progress_very_severe_pneum_agelt2mo'] = 1.3
        p['rr_progress_very_severe_pneum_age12to23mo'] = 0.9
        p['rr_progress_very_severe_pneum_age24to59mo'] = 0.6
        p['rr_progress_very_severe_pneum_HIV'] = 1.2
        p['rr_progress_very_severe_pneum_SAM'] = 1.1
        p['r_death_pneumonia'] = 0.5
        p['rr_death_pneumonia_agelt2mo'] = 1.2
        p['rr_death_pneumonia_age12to23mo'] = 0.8
        p['rr_death_pneumonia_age24to59mo'] = 0.04
        p['rr_death_pneumonia_HIV'] = 1.4
        p['rr_death_pneumonia_SAM'] = 1.3
        p['init_prop_pneumonia_status'] = [0.4, 0.3, 0.2]

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

        # defaults
        df['ri_pneumonia_status'] = 'none'
        df['malnutrition'] = False
        df['has_HIV'] = False
        df['exclusive_breastfeeding'] = False
        df['continued_breastfeeding'] = False
        df['ri_pneumonia_death'] = False
        df['pn_cough'] = False
        df['pn_difficult_breathing'] = False
        df['pn_fast_breathing'] = False
        df['date_of_acquiring_pneumonia'] = pd.NaT

        # --------------------------------------------------------------------------------------------------------
        # ------------------------- ASSIGN VALUES OF LRI - PNEUMONIA STATUS AT BASELINE --------------------------
        # --------------------------------------------------------------------------------------------------------
        df_under5 = df.age_years < 5 & df.is_alive
        under5_idx = df.index[df_under5]

        # create data-frame of the probabilities of ri_pneumonia_status for children
        # aged 2-11 months, HIV negative, no SAM, no indoor air pollution
        p_pneumonia_status = pd.Series(self.init_prop_pneumonia_status[0], index=under5_idx)
        p_sev_pneum_status = pd.Series(self.init_prop_pneumonia_status[1], index=under5_idx)
        p_very_sev_pneum_status = pd.Series(self.init_prop_pneumonia_status[2], index=under5_idx)

        # create probabilities of pneumonia for all age under
        p_pneumonia_status.loc[(df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] \
            *= self.rp_pneumonia_age12to23mo
        p_pneumonia_status.loc[(df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] \
            *= self.rp_pneumonia_age24to59mo
        p_pneumonia_status.loc[(df.has_hiv == True) & df_under5] *= self.rp_pneumonia_HIV
        p_pneumonia_status.loc[(df.malnutrition == True) & df_under5] *= self.rp_pneumonia_SAM
        p_pneumonia_status.loc[(df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) & df.is_alive] \
            *= self.rp_pneumonia_excl_breast
        p_pneumonia_status.loc[(df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) &
                               (df.age_exact_years < 2) & df.is_alive] *= self.rp_pneumonia_cont_breast
        p_pneumonia_status.loc[(df.li_wood_burn_stove == False) & df_under5] *= self.rp_pneumonia_IAP

        # create probabilities of severe pneumonia for all age under 5
        p_sev_pneum_status.loc[(df.age_exact_years < 0.1667) & df.is_alive] *= self.rp_severe_pneum_agelt2mo
        p_sev_pneum_status.loc[(df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] \
            *= self.rp_severe_pneum_age12to23mo
        p_sev_pneum_status.loc[(df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] \
            *= self.rp_severe_pneum_age24to59mo
        p_sev_pneum_status.loc[(df.has_hiv == True) & df_under5] *= self.rp_severe_pneum_HIV
        p_sev_pneum_status.loc[(df.malnutrition == True) & df_under5] *= self.rp_severe_pneum_SAM
        p_sev_pneum_status.loc[(df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) & df.is_alive] \
            *= self.rp_severe_pneum_excl_breast
        p_sev_pneum_status.loc[(df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) &
                               (df.age_exact_years < 2) & df.is_alive] *= self.rp_severe_pneum_cont_breast
        p_sev_pneum_status.loc[(df.li_wood_burn_stove == False) & df_under5] *= self.rp_severe_pneum_IAP

        # create probabilities of very severe pneumonia for all age under 5
        p_very_sev_pneum_status.loc[(df.age_exact_years < 0.1667) & df.is_alive] *= self.rp_very_severe_pneum_agelt2mo
        p_very_sev_pneum_status.loc[(df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] \
            *= self.rp_very_severe_pneum_age12to23mo
        p_very_sev_pneum_status.loc[(df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] \
            *= self.rp_very_severe_pneum_age24to59mo
        p_very_sev_pneum_status.loc[(df.has_hiv == True) & df_under5] *= self.rp_very_severe_pneum_HIV
        p_very_sev_pneum_status.loc[(df.malnutrition == True) & df_under5] *= self.rp_very_severe_pneum_SAM
        p_very_sev_pneum_status.loc[(df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) & df.is_alive] \
            *= self.rp_very_severe_pneum_excl_breast
        p_very_sev_pneum_status.loc[(df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) &
                                    (df.age_exact_years < 2) & df.is_alive] *= self.rp_very_severe_pneum_cont_breast
        p_very_sev_pneum_status.loc[(df.li_wood_burn_stove == False) & df_under5] *= self.rp_very_severe_pneum_IAP

        random_draw = pd.Series(rng.random_sample(size=len(under5_idx)), index=under5_idx)

        # create a temporary dataframe called dfx to hold values of probabilities and random draw
        dfx = pd.concat([p_pneumonia_status, p_sev_pneum_status, p_very_sev_pneum_status, random_draw], axis=1)
        dfx.columns = ['p_pneumonia', 'p_severe_pneumonia', 'p_very_severe_pneumonia', 'random_draw']
        dfx['p_none'] = 1 - (dfx.p_pneumonia + dfx.p_severe_pneumonia + dfx.p_very_severe_pneumonia)

        idx_none = dfx.index[dfx.p_none > dfx.random_draw]
        idx_pneumonia = dfx.index[(dfx.p_none < dfx.random_draw) & ((dfx.p_none + dfx.p_pneumonia) > dfx.random_draw)]
        idx_severe_pneumonia = dfx.index[((dfx.p_none + dfx.p_pneumonia) < dfx.random_draw) &
                                         (dfx.p_none + dfx.p_pneumonia + dfx.p_severe_pneumonia) > dfx.random_draw]
        idx_very_severe_pneumonia = dfx.index[
            ((dfx.p_none + dfx.p_pneumonia + dfx.p_severe_pneumonia) < dfx.random_draw) &
            (dfx.p_none + dfx.p_pneumonia + dfx.p_severe_pneumonia + dfx.p_very_severe_pneumonia) > dfx.random_draw]

        df.loc[idx_none, 'ri_pneumonia_severity'] = 'none'
        df.loc[idx_pneumonia, 'ri_pneumonia_severity'] = 'pneumonia'
        df.loc[idx_severe_pneumonia, 'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[idx_very_severe_pneumonia, 'ri_pneumonia_severity'] = 'very severe pneumonia'

        # # # # # # # # # DIAGNOSED AND TREATED BASED ON CARE SEEKING AND IMCI EFFECTIVENESS # # # # # # # # #

        init_pneumonia_idx = df.index[df.is_alive & df.age_exact_years < 5 & (df.ri_pneumonia_status is True)]
        random_draw = self.sim.rng.random_sample(size=len(init_pneumonia_idx))
        prob_sought_care = pd.Series(self.dhs_care_seeking_2010, index=init_pneumonia_idx)
        sought_care = prob_sought_care > random_draw
        sought_care_idx = prob_sought_care.index[sought_care]

        for i in sought_care_idx:
            random_draw1 = self.sim.rng.random_sample(size=len(sought_care_idx))
            diagnosed_and_treated = df.index[
                df.is_alive & (random_draw1 < self.parameters['IMCI_effectiveness_2010'])
                & (df.age_years < 5)]
            df.at[diagnosed_and_treated[i], 'ri_pneumonia_status'] = False

        # # # # # # # # # # ASSIGN RECOVERY AND DEATH TO BASELINE PNEUMONIA CASES # # # # # # # # # #

        not_treated_pneumonia_idx = df.index[df.is_alive & df.age_exact_years < 5 & (df.ri_pneumonia_status is True)]
        for i in not_treated_pneumonia_idx:
            random_draw2 = self.sim.rng.random_sample(size=len(not_treated_pneumonia_idx))
            death_pneumonia = df.index[
                df.is_alive & (random_draw2 < self.parameters['r_death_pneumonia'])
                & (df.age_years < 5)]
            if death_pneumonia[i]:
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, i, 'NewPneumonia'), self.sim.date)
                df.at[i, 'ri_pneumonia_status'] = False
            else:
                df.at[i, 'ri_pneumonia_status'] = False
# --------------------------------------------------------------------------------------------------------------------

        df_under5 = df.age_years < 5 & df.is_alive
        under5_idx = df.index[df_under5]

        # create data-frame of the probabilities of ri_pneumonia_status for children
        # aged 2-11 months, HIV negative, no SAM, no indoor air pollution
        p_pneumonia_status = pd.Series(self.init_prop_pneumonia_status[0], index=under5_idx)
        p_sev_pneum_status = pd.Series(self.init_prop_pneumonia_status[1], index=under5_idx)
        p_very_sev_pneum_status = pd.Series(self.init_prop_pneumonia_status[2], index=under5_idx)

        # create probabilities of pneumonia for all age under
        p_pneumonia_status.loc[(df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] \
            *= self.rp_pneumonia_age12to23mo
        p_pneumonia_status.loc[(df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] \
            *= self.rp_pneumonia_age24to59mo
        p_pneumonia_status.loc[(df.has_hiv == True) & df_under5] *= self.rp_pneumonia_HIV
        p_pneumonia_status.loc[(df.malnutrition == True) & df_under5] *= self.rp_pneumonia_SAM
        p_pneumonia_status.loc[(df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) & df.is_alive] \
            *= self.rp_pneumonia_excl_breast
        p_pneumonia_status.loc[(df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) &
                               (df.age_exact_years < 2) & df.is_alive] *= self.rp_pneumonia_cont_breast
        p_pneumonia_status.loc[(df.li_wood_burn_stove == False) & df_under5] *= self.rp_pneumonia_IAP

        # create probabilities of severe pneumonia for all age under 5
        p_sev_pneum_status.loc[(df.age_exact_years < 0.1667) & df.is_alive] *= self.rp_severe_pneum_agelt2mo
        p_sev_pneum_status.loc[(df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] \
            *= self.rp_severe_pneum_age12to23mo
        p_sev_pneum_status.loc[(df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] \
            *= self.rp_severe_pneum_age24to59mo
        p_sev_pneum_status.loc[(df.has_hiv == True) & df_under5] *= self.rp_severe_pneum_HIV
        p_sev_pneum_status.loc[(df.malnutrition == True) & df_under5] *= self.rp_severe_pneum_SAM
        p_sev_pneum_status.loc[(df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) & df.is_alive] \
            *= self.rp_severe_pneum_excl_breast
        p_sev_pneum_status.loc[(df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) &
                               (df.age_exact_years < 2) & df.is_alive] *= self.rp_severe_pneum_cont_breast
        p_sev_pneum_status.loc[(df.li_wood_burn_stove == False) & df_under5] *= self.rp_severe_pneum_IAP

        # create probabilities of very severe pneumonia for all age under 5
        p_very_sev_pneum_status.loc[(df.age_exact_years < 0.1667) & df.is_alive] *= self.rp_very_severe_pneum_agelt2mo
        p_very_sev_pneum_status.loc[(df.age_exact_years >= 1) & (df.age_exact_years < 2) & df.is_alive] \
            *= self.rp_very_severe_pneum_age12to23mo
        p_very_sev_pneum_status.loc[(df.age_exact_years >= 2) & (df.age_exact_years < 5) & df.is_alive] \
            *= self.rp_very_severe_pneum_age24to59mo
        p_very_sev_pneum_status.loc[(df.has_hiv == True) & df_under5] *= self.rp_very_severe_pneum_HIV
        p_very_sev_pneum_status.loc[(df.malnutrition == True) & df_under5] *= self.rp_very_severe_pneum_SAM
        p_very_sev_pneum_status.loc[(df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5) & df.is_alive] \
            *= self.rp_very_severe_pneum_excl_breast
        p_very_sev_pneum_status.loc[(df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) &
                                    (df.age_exact_years < 2) & df.is_alive] *= self.rp_very_severe_pneum_cont_breast
        p_very_sev_pneum_status.loc[(df.li_wood_burn_stove == False) & df_under5] *= self.rp_very_severe_pneum_IAP

        random_draw = pd.Series(rng.random_sample(size=len(under5_idx)), index=under5_idx)

        # create a temporary dataframe called dfx to hold values of probabilities and random draw
        dfx = pd.concat([p_pneumonia_status, p_sev_pneum_status, p_very_sev_pneum_status, random_draw], axis=1)
        dfx.columns = ['p_pneumonia', 'p_severe_pneumonia', 'p_very_severe_pneumonia', 'random_draw']

        dfx['p_none'] = 1 - (dfx.p_pneumonia + dfx.p_severe_pneumonia + dfx.p_very_severe_pneumonia)

        # based on probabilities of being in each category, define cut-offs to determine status from
        # random draw uniform(0,1)

        # assign baseline values of ri_resp_infection_stat based on probabilities and value of random draw

        idx_none = dfx.index[dfx.p_none > dfx.random_draw]
        idx_pneumonia = dfx.index[(dfx.p_none < dfx.random_draw) & ((dfx.p_none + dfx.p_pneumonia) > dfx.random_draw)]
        idx_severe_pneumonia = dfx.index[((dfx.p_none + dfx.p_pneumonia) < dfx.random_draw) &
                                         (dfx.p_none + dfx.p_pneumonia + dfx.p_severe_pneumonia) > dfx.random_draw]
        idx_very_severe_pneumonia = dfx.index[((dfx.p_none + dfx.p_pneumonia + dfx.p_severe_pneumonia) < dfx.random_draw) &
                                         (dfx.p_none + dfx.p_pneumonia + dfx.p_severe_pneumonia + dfx.p_very_severe_pneumonia) > dfx.random_draw]

        df.loc[idx_none, 'ri_pneumonia_status'] = 'none'
        df.loc[idx_pneumonia, 'ri_pneumonia_status'] = 'pneumonia'
        df.loc[idx_severe_pneumonia, 'ri_pneumonia_status'] = 'severe pneumonia'
        df.loc[idx_very_severe_pneumonia, 'ri_pneumonia_status'] = 'very severe pneumonia'

    def initialise_simulation(self, sim):
        """
        Get ready for simulation start.
        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event
        sim.schedule_event(PneumoniaEvent(self), sim.date + DateOffset(months=3))
        sim.schedule_event(SeverePneumoniaEvent(self), sim.date + DateOffset(months=3))
        sim.schedule_event(VerySeverePneumoniaEvent(self), sim.date + DateOffset(months=3))

        # add an event to log to screen
        # sim.schedule_event(PneumoniaLoggingEvent(self), sim.date + DateOffset(months=3))
        # sim.schedule_event(SeverePneumoniaLoggingEvent(self), sim.date + DateOffset(months=3))
        # sim.schedule_event(VerySeverePneumoniaLoggingEvent(self), sim.date + DateOffset(months=3))

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def on_birth(self, mother_id, child_id):
        pass

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is Pneumonia, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)


class PneumoniaEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=3))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = m.rng

        # 1. DO I NEED TO GET THOSE WHO ARE CURRENTLY WITH PNEUMONIA STATUS BEFORE NEXT STEP???

        # --------------------------------------------------------------------------------------------------------
        # UPDATING FOR CHILDREN UNDER 5 WITH CURRENT STATUS 'NONE' TO NON-SEVERE PNEUMONIA
        # --------------------------------------------------------------------------------------------------------

        no_pneumonia = df.is_alive & (df.ri_pneumonia_status == 'none')
        no_pneumonia_age2to59mo = df.is_alive & (df.ri_pneumonia_status == 'none') & \
                                  (df.age_exact_years > 0.1667) & (df.age_years < 5)

        eff_prob_ri_pneumonia = pd.Series(m.base_incidence_pneumonia,
                                          index=df.index[no_pneumonia_age2to59mo])

        eff_prob_ri_pneumonia.loc[no_pneumonia & (df.age_exact_years >= 1) & (df.age_exact_years < 2)]\
            *= m.rr_pneumonia_age12to23mo
        eff_prob_ri_pneumonia.loc[no_pneumonia & (df.age_exact_years >= 2) & (df.age_exact_years < 5)] \
            *= m.rr_pneumonia_age24to59mo
        eff_prob_ri_pneumonia.loc[no_pneumonia_age2to59mo & df.li_no_access_handwashing == False]\
            *= m.rr_pneumonia_HHhandwashing
        eff_prob_ri_pneumonia.loc[no_pneumonia_age2to59mo & (df.has_hiv == True)] *= m.rr_pneumonia_HIV
        eff_prob_ri_pneumonia.loc[no_pneumonia_age2to59mo & df.malnutrition == True] *= m.rr_pneumonia_SAM
        eff_prob_ri_pneumonia.loc[no_pneumonia & df.exclusive_breastfeeding == True & (df.age_exact_years <= 0.5)] \
            *= m.rr_pneumonia_excl_breast
        eff_prob_ri_pneumonia.loc[no_pneumonia & (df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) &
                                  (df.age_exact_years < 2)] *= m.rr_pneumonia_cont_breast
        eff_prob_ri_pneumonia.loc[no_pneumonia_age2to59mo & df.li_wood_burn_stove == False] *= m.rr_pneumonia_IAP

        pn_current_none_idx = df.index[no_pneumonia_age2to59mo]

        random_draw_01 = pd.Series(rng.random_sample(size=len(pn_current_none_idx)),
                                   index=df.index[no_pneumonia_age2to59mo])

        get_pneumonia = eff_prob_ri_pneumonia > random_draw_01
        idx_get_pn = eff_prob_ri_pneumonia.index[get_pneumonia]

        df.loc[idx_get_pn, 'ri_pneumonia_status'] = 'pneumonia'

        # # # # # # # WHEN THEY GET THE DISEASE - DATE -----------------------------------------------------------

        random_draw_days = np.random.randint(0, 90, size=len(get_pneumonia))
        td = pd.to_timedelta(random_draw_days, unit='d')
        date_of_aquisition = self.sim.date + td
        df.loc[idx_get_pn, 'date_of_acquiring_pneumonia'] = date_of_aquisition

        # # # # # # # # # SYMPTOMS FROM NON-SEVERE PNEUMONIA # # # # # # # # # # # # # # # # # #

        pn_current_pneumonia_idx = df.index[df.is_alive & (df.age_exact_years > 0.1667) & (df.age_years < 5) &
                                            (df.ri_pneumonia_status == 'pneumonia')] # non-severe pneumonia only in 2-59 months
        # fast breathing
        df.loc[idx_get_pn, 'pn_fast_breathing'] = True

        # cough
        eff_prob_cough = pd.Series(0.89, index=pn_current_pneumonia_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_pneumonia_idx)),
                                index=df.index[(df.age_exact_years > 0.1667) & (df.age_years < 5) & df.is_alive &
                                               (df.ri_pneumonia_status == 'pneumonia')])
        dfx = pd.concat([eff_prob_cough, random_draw], axis=1)
        dfx.columns = ['eff_prob_cough', 'random number']
        idx_cough = dfx.index[dfx.eff_prob_cough > random_draw]
        df.loc[idx_cough, 'pn_cough'] = True

        # difficult breathing
        eff_prob_difficult_breathing = pd.Series(0.89, index=pn_current_pneumonia_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_pneumonia_idx)),
                                index=df.index[(df.age_exact_years > 0.1667) &
                                               (df.age_years < 5) & df.is_alive & (df.ri_pneumonia_status == 'pneumonia')])
        dfx = pd.concat([eff_prob_difficult_breathing, random_draw], axis=1)
        dfx.columns = ['eff_prob_difficult_breathing', 'random number']
        idx_difficult_breathing = dfx.index[dfx.eff_prob_difficult_breathing > random_draw]
        df.loc[idx_difficult_breathing, 'pn_difficult_breathing'] = True

        # --------------------------------------------------------------------------------------------------------
        # SEEKING CARE FOR NON-SEVERE PNEUMONIA
        # --------------------------------------------------------------------------------------------------------

        pneumonia_symptoms = df.index[df.is_alive & (df.pn_cough == True) | (df.pn_difficult_breathing == True) |
                                      (df.pn_fast_breathing == True) | (df.pn_chest_indrawing == False)]

        seeks_care = pd.Series(data=False, index=pneumonia_symptoms)
        for individual in pneumonia_symptoms:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(individual, symptom_code=1)
            seeks_care[individual] = self.module.rng.rand() < prob
            event = HSI_Sick_Child_Seeks_Care_From_HSA(self.module, person_id=individual)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                            priority=2,
                                                            topen=self.sim.date,
                                                            tclose=self.sim.date + DateOffset(weeks=2)
                                                            )

# For those with non-sev pneumonia, calculate the probability of disease progression: ------------------------

        # --------------------------------------------------------------------------------------------------------
        # UPDATING FOR CHILDREN UNDER 5 WITH CURRENT STATUS 'PNEUMONIA' TO 'SEVERE PNEUMONIA'
        # --------------------------------------------------------------------------------------------------------

        eff_prob_prog_severe_pneumonia = pd.Series(m.r_progress_to_severe_pneum,
                                                   index=df.index[df.is_alive & (df.ri_pneumonia_status == 'pneumonia')
                                                                  & (df.age_years < 5)])

        eff_prob_prog_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                           (df.age_exact_years >= 1) & (
                                                   df.age_exact_years < 2)] *= m.rr_progress_severe_pneum_age12to23mo
        eff_prob_prog_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                           (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_progress_severe_pneum_age24to59mo
        eff_prob_prog_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                           df.has_hiv == True & (df.age_years < 5)] *= \
            m.rr_progress_severe_pneum_HIV
        eff_prob_prog_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                           df.malnutrition == True & (df.age_years < 5)] *= \
            m.rr_progress_severe_pneum_SAM

        pn_current_pneumonia_idx = df.index[df.is_alive & (df.age_years < 5) & (df.ri_pneumonia_status == 'pneumonia')]

        random_draw_03 = pd.Series(rng.random_sample(size=len(pn_current_pneumonia_idx)),
                                   index=df.index[(df.age_years < 5) & df.is_alive &
                                                  (df.ri_pneumonia_status == 'pneumonia')])

        dfx = pd.concat([eff_prob_prog_severe_pneumonia, random_draw_03], axis=1)
        dfx.columns = ['eff_prob_prog_severe_pneumonia', 'random_draw_03']
        idx_ri_progress_severe_pneumonia = dfx.index[dfx.eff_prob_prog_severe_pneumonia > dfx.random_draw_03]
        df.loc[idx_ri_progress_severe_pneumonia, 'ri_pneumonia_status'] = 'severe pneumonia'

        # for those that have pneumonia, probability of those who will progress TO SEVERE PNEUMONIA:
        # if ['ri_pneumonia_status' == 'pneumonia']:
        # date_disease_progression = date_of_acquiring_pneum + DateOffset(days=5) + DateOffset(
        #   days=int((self.module.rng.rand() - 0.5) * 10))
        # event = ProgressToSeverePneumoniaEvent(self, person_id) ##### NOT NEEDED
        # self.sim.schedule_event(event, date_of_death) ##### NOT NEEDED

        # else: RECOVERY
        # date_of_recovery = date_of_getting_disease + DateOffset(days=14) + DateOffset(
        # days=int((self.module.rng.rand() - 0.5) * 10))
        # df[person_id, 'date_of_recovery'] = date_of_recovery
        # df[person_id, 'ri_pneumonia_status'] = 'none'

        after_progression_pneumonia_idx = df.index[df.is_alive &
                                                   (df.ri_pneumonia_status == 'pneumonia') & (df.age_years < 5)]

        if self.sim.date + DateOffset(weeks=2):
            df.loc[after_progression_pneumonia_idx, 'ri_pneumonia_status'] == 'none'


class SeverePneumoniaEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(weeks=2))

    def apply(self, population):
        df = population.props
        m = self.module
        rng = m.rng

        # --------------------------------------------------------------------------------------------------------
        # UPDATING FOR CHILDREN UNDER 5 WITH CURRENT STATUS 'NONE' TO 'SEVERE PNEUMONIA'
        # --------------------------------------------------------------------------------------------------------

        eff_prob_ri_severe_pneumonia = pd.Series(m.base_incidence_severe_pneum,
                                                  index=df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                                                 (df.age_years < 5)])
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         (df.age_exact_years < 0.1667)] *= m.rr_severe_pneum_agelt2mo
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         (df.age_exact_years >= 1) & (
                                                  df.age_exact_years < 2)] *= m.rr_severe_pneum_age12to23mo
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         (df.age_exact_years >= 2) & (
                                                  df.age_exact_years < 5)] *= m.rr_severe_pneum_age24to59mo
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         df.li_no_access_handwashing == False & (
                                                  df.age_years < 5)] *= m.rr_severe_pneum_HHhandwashing
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         (df.has_hiv == True) & (df.age_years < 5)] *= m.rr_severe_pneum_HIV
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         df.malnutrition == True & (df.age_years < 5)] *= m.rr_severe_pneum_SAM
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         df.exclusive_breastfeeding == True & (
                                                 df.age_exact_years <= 0.5)] *= m.rr_severe_pneum_excl_breast
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.continued_breastfeeding == True) &
                                         (df.age_exact_years > 0.5) & (
                                                  df.age_exact_years < 2)] *= m.rr_severe_pneum_cont_breast
        eff_prob_ri_severe_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         df.li_wood_burn_stove == False & (df.age_years < 5)] *= m.rr_severe_pneum_IAP

        pn1_current_none_idx = df.index[df.is_alive & (df.age_years < 5) & (df.ri_pneumonia_status == 'none')]

        random_draw_02 = pd.Series(rng.random_sample(size=len(pn1_current_none_idx)),
                                    index=df.index[
                                        (df.age_years < 5) & df.is_alive & (df.ri_pneumonia_status == 'none')])

        dfx = pd.concat([eff_prob_ri_severe_pneumonia, random_draw_02], axis=1)
        dfx.columns = ['eff_prob_ri_severe_pneumonia', 'random_draw_02']

        idx_incident_severe_pneumonia = dfx.index[dfx.eff_prob_ri_severe_pneumonia > dfx.random_draw_02]

        df.loc[idx_incident_severe_pneumonia, 'ri_pneumonia_status'] = 'severe pneumonia'

        # # # # # # # # # SYMPTOMS FROM SEVERE PNEUMONIA # # # # # # # # # # # # # # # # # #

        pn_current_severe_pneum_idx = df.index[df.is_alive & (df.age_years < 5) &
                                               (df.ri_pneumonia_status == 'severe pneumonia')]
        for individual in pn_current_severe_pneum_idx:
            df.at[individual, 'pn_chest_indrawing'] = True

        eff_prob_cough = pd.Series(0.96, index=pn_current_severe_pneum_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_severe_pneum_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.ri_pneumonia_status == 'severe pneumonia')])
        dfx = pd.concat([eff_prob_cough, random_draw], axis=1)
        dfx.columns = ['eff_prob_cough', 'random number']
        idx_cough = dfx.index[dfx.eff_prob_cough > random_draw]
        df.loc[idx_cough, 'pn_cough'] = True

        eff_prob_difficult_breathing = pd.Series(0.40, index=pn_current_severe_pneum_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_severe_pneum_idx)),
                                index=df.index[
                                    (df.age_years < 5) & df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia')])
        dfx = pd.concat([eff_prob_difficult_breathing, random_draw], axis=1)
        dfx.columns = ['eff_prob_difficult_breathing', 'random number']
        idx_difficult_breathing = dfx.index[dfx.eff_prob_difficult_breathing > random_draw]
        df.loc[idx_difficult_breathing, 'pn_difficult_breathing'] = True

        eff_prob_fast_breathing = pd.Series(0.96, index=pn_current_severe_pneum_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_severe_pneum_idx)),
                                index=df.index[
                                    (df.age_years < 5) & df.is_alive & (df.ri_pneumonia_status == 'severe pneumonia')])
        dfx = pd.concat([eff_prob_fast_breathing, random_draw], axis=1)
        dfx.columns = ['eff_prob_fast_breathing', 'random number']
        idx_fast_breathing = dfx.index[dfx.eff_prob_fast_breathing > random_draw]
        df.loc[idx_fast_breathing, 'pn_fast_breathing'] = True

        # --------------------------------------------------------------------------------------------------------
        # SEEKING CARE FOR SEVERE PNEUMONIA
        # --------------------------------------------------------------------------------------------------------

        severe_pneumonia_symptoms = df.index[df.is_alive & (df.pn_cough == True) | (df.pn_difficult_breathing == True) |
                                           (df.pn_fast_breathing == True) | (df.pn_chest_indrawing == True)]

        seeks_care = pd.Series(data=False, index=severe_pneumonia_symptoms)
        for individual in severe_pneumonia_symptoms:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(individual, symptom_code=1)
            seeks_care[individual] = self.module.rng.rand() < prob
            event = HSI_Sick_Child_Seeks_Care_From_HSA(self.module['iCCM'], person_id=individual)
            self.sim.modules['HealthSystem'].schedule_event(event,
                                                            priority=2,
                                                            topen=self.sim.date,
                                                            tclose=self.sim.date + DateOffset(weeks=2)
                                                            )

        # --------------------------------------------------------------------------------------------------------
        # UPDATING FOR CHILDREN UNDER 5 WITH CURRENT STATUS 'SEVERE PNEUMONIA' TO 'VERY SEVERE PNEUMONIA'
        # --------------------------------------------------------------------------------------------------------

        eff_prob_prog_very_sev_pneumonia = pd.Series(m.r_progress_to_very_severe_pneum,
                                                     index=df.index[
                                                         df.is_alive & (df.ri_pneumonia_status == 'pneumonia')
                                                         & (df.age_years < 5)])

        eff_prob_prog_very_sev_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                             (df.age_years < 5)] *= m.rr_progress_very_severe_pneum_agelt2mo
        eff_prob_prog_very_sev_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                             (df.age_exact_years >= 1) & (
                                                 df.age_exact_years < 2)] *= m.rr_progress_very_severe_pneum_age12to23mo
        eff_prob_prog_very_sev_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                             (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_progress_very_severe_pneum_age24to59mo
        eff_prob_prog_very_sev_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                             df.has_hiv == True & (df.age_years < 5)] *= \
            m.rr_progress_very_severe_pneum_HIV
        eff_prob_prog_very_sev_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'pneumonia') &
                                             df.malnutrition == True & (df.age_years < 5)] *= \
            m.rr_progress_very_severe_pneum_SAM

        pn_current_severe_pneumonia_idx = df.index[df.is_alive & (df.age_years < 5) &
                                                   (df.ri_pneumonia_status == 'severe pneumonia')]

        random_draw_03 = pd.Series(rng.random_sample(size=len(pn_current_severe_pneumonia_idx)),
                                   index=df.index[(df.age_years < 5) & df.is_alive &
                                                  (df.ri_pneumonia_status == 'severe pneumonia')])
        dfx = pd.concat([eff_prob_prog_very_sev_pneumonia, random_draw_03], axis=1)
        dfx.columns = ['eff_prob_prog_very_severe_pneumonia', 'random_draw_03']
        idx_ri_progress_very_sev_pneumonia = dfx.index[dfx.eff_prob_prog_very_severe_pneumonia > dfx.random_draw_03]
        df.loc[idx_ri_progress_very_sev_pneumonia, 'ri_pneumonia_status'] = 'very severe pneumonia'


class VerySeverePneumoniaEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(weeks=2))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = m.rng

        # --------------------------------------------------------------------------------------------------------
        # UPDATING FOR CHILDREN UNDER 5 WITH CURRENT STATUS 'NONE' TO 'VERY SEVERE PNEUMONIA'
        # --------------------------------------------------------------------------------------------------------

        eff_prob_ri_very_sev_pneumonia = pd.Series(m.base_incidence_very_severe_pneum,
                                                 index=df.index[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                                                (df.age_years < 5)])
        eff_prob_ri_very_sev_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         (df.age_exact_years < 0.1667)] *= m.rr_very_severe_pneum_agelt2mo
        eff_prob_ri_very_sev_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         (df.age_exact_years >= 1) & (
                                             df.age_exact_years < 2)] *= m.rr_very_severe_pneum_age12to23mo
        eff_prob_ri_very_sev_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         (df.age_exact_years >= 2) & (
                                             df.age_exact_years < 5)] *= m.rr_very_severe_pneum_age24to59mo
        eff_prob_ri_very_sev_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         df.li_no_access_handwashing == False & (
                                             df.age_years < 5)] *= m.rr_very_severe_pneum_HHhandwashing
        eff_prob_ri_very_sev_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         (df.has_hiv == True) & (df.age_years < 5)] *= m.rr_very_severe_pneum_HIV
        eff_prob_ri_very_sev_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         df.malnutrition == True & (df.age_years < 5)] *= m.rr_very_severe_pneum_SAM
        eff_prob_ri_very_sev_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         df.exclusive_breastfeeding == True & (
                                             df.age_exact_years <= 0.5)] *= m.rr_very_severe_pneum_excl_breast
        eff_prob_ri_very_sev_pneumonia.loc[df.is_alive & (df.continued_breastfeeding == True) &
                                         (df.age_exact_years > 0.5) & (
                                             df.age_exact_years < 2)] *= m.rr_very_severe_pneum_cont_breast
        eff_prob_ri_very_sev_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'none') &
                                         df.li_wood_burn_stove == False & (df.age_years < 5)] *= m.rr_very_severe_pneum_IAP

        pn1_current_none_idx = df.index[df.is_alive & (df.age_years < 5) & (df.ri_pneumonia_status == 'none')]

        random_draw_03 = pd.Series(rng.random_sample(size=len(pn1_current_none_idx)),
                                   index=df.index[
                                       (df.age_years < 5) & df.is_alive & (df.ri_pneumonia_status == 'none')])

        dfx = pd.concat([eff_prob_ri_very_sev_pneumonia, random_draw_03], axis=1)
        dfx.columns = ['eff_prob_ri_very_severe_pneumonia', 'random_draw_03']

        idx_incident_severe_pneumonia = dfx.index[dfx.eff_prob_ri_very_severe_pneumonia > dfx.random_draw_03]

        df.loc[idx_incident_severe_pneumonia, 'ri_pneumonia_status'] = 'very severe pneumonia'

        # # # # # # # # # SYMPTOMS FROM VERY SEVERE PNEUMONIA # # # # # # # # # # # # # # # # # #

        pn_current_very_sev_pneum_idx = df.index[df.is_alive & (df.age_years < 5) &
                                               (df.ri_pneumonia_status == 'very severe pneumonia')]

        eff_prob_cough = pd.Series(0.857, index=pn_current_very_sev_pneum_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_very_sev_pneum_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.ri_pneumonia_status == 'very severe pneumonia')])
        dfx = pd.concat([eff_prob_cough, random_draw], axis=1)
        dfx.columns = ['eff_prob_cough', 'random number']
        idx_cough = dfx.index[dfx.eff_prob_cough > random_draw]
        df.loc[idx_cough, 'pn_cough'] = True

        eff_prob_difficult_breathing = pd.Series(0.43, index=pn_current_very_sev_pneum_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_very_sev_pneum_idx)),
                                index=df.index[
                                    (df.age_years < 5) & df.is_alive & (df.ri_pneumonia_status == 'very severe pneumonia')])
        dfx = pd.concat([eff_prob_difficult_breathing, random_draw], axis=1)
        dfx.columns = ['eff_prob_difficult_breathing', 'random number']
        idx_difficult_breathing = dfx.index[dfx.eff_prob_difficult_breathing > random_draw]
        df.loc[idx_difficult_breathing, 'pn_difficult_breathing'] = True

        eff_prob_fast_breathing = pd.Series(0.857, index=pn_current_very_sev_pneum_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_very_sev_pneum_idx)),
                                index=df.index[
                                    (df.age_years < 5) & df.is_alive & (df.ri_pneumonia_status == 'very severe pneumonia')])
        dfx = pd.concat([eff_prob_fast_breathing, random_draw], axis=1)
        dfx.columns = ['eff_prob_fast_breathing', 'random number']
        idx_fast_breathing = dfx.index[dfx.eff_prob_fast_breathing > random_draw]
        df.loc[idx_fast_breathing, 'pn_fast_breathing'] = True

        eff_prob_chest_indrawing = pd.Series(0.76, index=pn_current_very_sev_pneum_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_very_sev_pneum_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.ri_pneumonia_status == 'very severe pneumonia')])
        dfx = pd.concat([eff_prob_chest_indrawing, random_draw], axis=1)
        dfx.columns = ['eff_prob_chest_indrawing', 'random number']
        idx_chest_indrawing = dfx.index[dfx.eff_prob_chest_indrawing > random_draw]
        df.loc[idx_chest_indrawing, 'pn_chest_indrawing'] = True

        # --------------------------------------------------------------------------------------------------------
        # SEEKING CARE FOR VERY SEVERE PNEUMONIA
        # --------------------------------------------------------------------------------------------------------

        very_sev_pneum_symptoms = df.index[df.is_alive & (df.pn_cough == True) | (df.pn_difficult_breathing == True) |
                                           (df.pn_fast_breathing == True) | (df.pn_chest_indrawing == True) |
                                           (df.pn_any_general_danger_sign == True)]

        seeks_care = pd.Series(data=False, index=very_sev_pneum_symptoms)
        for individual in very_sev_pneum_symptoms:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(individual, symptom_code=1) # what happens with multiple symptoms of different severity??
            seeks_care[individual] = self.module.rng.rand() < prob
            event = HSI_Sick_Child_Seeks_Care_From_HSA(self.module['ICCM'], person_id=individual)
            self.sim.modules['HealthSystem'].schedule_event(event,
                                                            priority=2,
                                                            topen=self.sim.date,
                                                            tclose=self.sim.date + DateOffset(weeks=2)
                                                            )

        # Determine if anyone with symptoms will seek care
        # now work out who will seek care for having severe diarrhoea
        # loop through everyone who has severe diarhoea
        # for each person with severe diarrhoa, seek if they seek care ( by using the prob_seek_care function of health system)
        #       ( that gives a probability, flip a coin to work out if they actually do seek care)
        # if a person is seeking care, create an event 'HSI_Sick_Child_Seeks_Care_From_CHW()'
        # submit this event to the health system scheduler
        # (the health system scheduler will do the rest....)


# class RecoveryPneumoniaDisease_Event(Event, IndividualScopeEventMixin):

class DeathFromPneumoniaDisease(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        rng = m.rng

        eff_prob_death_pneumonia = \
            pd.Series(m.r_death_pneumonia,
                      index=df.index[df.is_alive & (df.ri_pneumonia_status == 'very severe pneumonia') & (df.age_years < 5)])
        eff_prob_death_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'very severe pneumonia') &
                     (df.age_years < 5)] *= m.rr_death_pneumonia_agelt2mo
        eff_prob_death_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'very severe pneumonia') &
                     (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= \
            m.rr_death_pneumonia_age12to23mo
        eff_prob_death_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'very severe pneumonia') &
                     (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_death_pneumonia_age24to59mo
        eff_prob_death_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'very severe pneumonia') &
                     df.has_hiv == True & (df.age_years < 5)] *= \
            m.rr_death_pneumonia_HIV
        eff_prob_death_pneumonia.loc[df.is_alive & (df.ri_pneumonia_status == 'very severe pneumonia') &
                     df.malnutrition == True & (df.age_years < 5)] *= \
            m.rr_death_pneumonia_SAM

        pn1_current_very_severe_pneumonia_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_status == 'very severe pneumonia') & (df.age_years < 5)]

        random_draw_04 = pd.Series(rng.random_sample(size=len(pn1_current_very_severe_pneumonia_idx)),
                                   index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.ri_pneumonia_status == ' very severe pneumonia')])

        dfx = pd.concat([eff_prob_death_pneumonia, random_draw_04], axis=1)
        dfx.columns = ['eff_prob_death_pneumonia', 'random_draw_04']
        dfx['pneumonia_death'] = False

        if dfx.loc[dfx.eff_prob_death_pneumonia > dfx.random_draw_04]:
            dfx['pneumonia_death'] = True
            df.loc[pn1_current_very_severe_pneumonia_idx, 'ri_pneumonia_death'] = dfx['pneumonia_death']
            for individual_id in df.index[df.ri_pneumonia_death]:
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, 'ChildhoodPneumonia'),
                                        self.sim.date)
        else:
            df.loc[pn1_current_very_severe_pneumonia_idx, 'ri_pneumonia_status'] = 'none'


''' 
class PneumoniaLoggingEvent(RegularEvent, PopulationScopeEventMixin):
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

        df = population.props
        m = self.module
        rng = m.rng

        all_children_pneumonia_idx = df.index[df.age_exact_years < 5 & df.ri_pneumonia_status == 'pneumonia']
        for child_p in all_children_pneumonia_idx:
            logger.info('%s|acquired_pneumonia|%s',
                        self.sim.date,
                        {
                            'child_index': child_p,
                            'progression': df.at[child_p, 'severe pneumonia'],
                        })


class SeverePneumoniaLoggingEvent(RegularEvent, PopulationScopeEventMixin):
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



# Here define HSI_Sick_Child_Seeks_Care_From_CHW()
# This will specify the interaction in which a child is assesed by the CHW
# HSI events need to have the following things defined...
# priority = 0 (very high priority for a sick child)
# treatment_id = 'Sick child presents for care'
# appt_footprint = 'Under five out patient appointmnet'
# levels at which this appt can have: level 0
# consumables  = none

# in the apply() part of the event:
# here is where we have the CHW going through the algorithm

# will_CHW_ask_about_fever = rand()<0.5
# will_CHW_ask_about_cough = rand()<0.5

# fever_is_detected = (df[person_id,'fever'] is True) and will_ask_CHW_ask_about_fever
# cough_is_detected = (df[person_id,'cough'] is True) and will_CHW_ask_about_cough

# if fever_is_detected:
#   if cough_is_detected:
#       -- child has bouth fever and cough
        # make a event for the treatment for this condition
        # HSI_Treatment_For_Fever_And_Cough


# define HSI_Treatment_For_fever_and_Cough

# prioruty ==0
# appt_footprint = unde5 classmethod *3 + dispensing
# appt_consumables = ORS
# levels at wich the appint can occur = 0
# in the apply section of this event..... implmenet the effect of the drug '''

