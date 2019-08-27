"""
Childhood pneumonia module
Documentation: 04 - Methods Repository/Method_Child_RespiratoryInfection.xlsx
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin
from tlo.methods import demography
from tlo.methods.Childhood_interventions import HSI_ICCM

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NewPneumonia(Module):
    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    PARAMETERS = {
        'base_incidence_pneumonia_by_RSV': Parameter
        (Types.LIST, 'incidence of pneumonia caused by Respiratory Syncytial Virus in age groups 0-11, 12-59 months'
         ),
        'base_incidence_pneumonia_by_rhinovirus': Parameter
        (Types.LIST, 'incidence of pneumonia caused by rhinovirus in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_hMPV': Parameter
        (Types.LIST, 'incidence of pneumonia caused by hMPV in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_parainfluenza': Parameter
        (Types.LIST, 'incidence of pneumonia caused by parainfluenza in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_streptococcus': Parameter
        (Types.LIST, 'incidence of pneumonia caused by streptoccocus 40/41 in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_hib': Parameter
        (Types.LIST, 'incidence of pneumonia caused by hib in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_TB': Parameter
        (Types.LIST, 'incidence of pneumonia caused by TB in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_staph': Parameter
        (Types.LIST, 'incidence of pneumonia caused by Staphylococcus aureus in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_influenza': Parameter
        (Types.LIST, 'incidence of pneumonia caused by influenza in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_jirovecii': Parameter
        (Types.LIST, 'incidence of pneumonia caused by P. jirovecii in age groups 0-11, 12-59 months'
         ),
        'rr_ri_pneumonia_HHhandwashing': Parameter
        (Types.REAL, 'relative rate of pneumonia with household handwashing with soap'
         ),
        'rr_ri_pneumonia_HIV': Parameter
        (Types.REAL, 'relative rate of pneumonia for HIV positive status'
         ),
        'rr_ri_pneumonia_SAM': Parameter
        (Types.REAL, 'relative rate of pneumonia for severe malnutrition'
         ),
        'rr_ri_pneumonia_excl_breast': Parameter
        (Types.REAL, 'relative rate of pneumonia for exclusive breastfeeding upto 6 months'
         ),
        'rr_ri_pneumonia_cont_breast': Parameter
        (Types.REAL, 'relative rate of pneumonia for continued breastfeeding 6 months to 2 years'
         ),
        'rr_ri_pneumonia_indoor_air_pollution': Parameter
        (Types.REAL, 'relative rate of pneumonia for indoor air pollution'
         ),
        'rr_ri_pneumonia_pneumococcal_vaccine': Parameter
        (Types.REAL, 'relative rate of pneumonia for pneumonococcal vaccine'
         ),
        'rr_ri_pneumonia_hib_vaccine': Parameter
        (Types.REAL, 'relative rate of pneumonia for hib vaccine'
         ),
        'rr_progress_severe_pneum_viral': Parameter
        (Types.REAL, 'relative rate of pneumonia for viral pathogen'
         ),
        'r_progress_to_severe_pneum': Parameter
        (Types.REAL,
         'probability of progressing from non-severe to severe pneumonia among children aged 2-11 months, '
         'HIV negative, no SAM'
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
        'rr_progress_very_sev_pneum_viral': Parameter
        (Types.REAL, 'relative rate of pneumonia for viral pathogen'
         ),
        'r_progress_to_very_sev_pneum': Parameter
        (Types.REAL,
         'probability of progressing from non-severe to severe pneumonia among children aged 2-11 months, '
         'HIV negative, no SAM'
         ),
        'rr_progress_very_sev_pneum_agelt2mo': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for age <2 months'
         ),
        'rr_progress_very_sev_pneum_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for age 12 to 23 months'
         ),
        'rr_progress_very_sev_pneum_age24to59mo': Parameter
        (Types.REAL, 'relative rate of progression to severe pneumonia for age 24 to 59 months'
         ),
        'rr_progress_very_sev_pneum_HIV': Parameter
        (Types.REAL,
         'relative risk of progression to severe pneumonia for HIV positive status'
         ),
        'rr_progress_very_sev_pneum_SAM': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for severe acute malnutrition'
         ),
    }

    PROPERTIES = {
        'ri_pneumonia_status': Property
        (Types.BOOL, 'symptomatic ALRI - pneumonia disease'
         ),
        'ri_pneumonia_severity': Property
        (Types.CATEGORICAL, 'severity of pneumonia disease',
         categories=['none', 'pneumonia', 'severe pneumonia', 'very severe pneumonia']
         ),
        'ri_pneumonia_pathogen': Property
        (Types.CATEGORICAL, 'attributable pathogen for pneumonia',
         categories=['RSV', 'rhinovirus', 'hMPV', 'parainfluenza', 'streptococcus',
                     'hib', 'TB', 'staph', 'influenza', 'P. jirovecii']
         ),
        'ri_pneumonia_pathogen_type': Property
        (Types.CATEGORICAL, 'bacterial or viral pathogen', categories=['bacterial', 'viral']
         ),
        'ri_scheduled_date_death': Property(Types.DATE, 'date of death from pneumonia disease'),
        'has_hiv': Property(Types.BOOL, 'temporary property - has hiv'),
        'malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
        'pneumococcal_vaccination': Property(Types.BOOL, 'temporary property - streptococcus pneumoniae vaccine'),
        'hib_vaccination': Property(Types.BOOL, 'temporary property - H. influenzae type b vaccine'),
        'influenza_vaccination': Property(Types.BOOL, 'temporary property - flu vaccine'),
        # symptoms of diarrhoea for care seeking
        'ri_pneumonia_death': Property(Types.BOOL, 'death from pneumonia disease'),
        'date_of_acquiring_pneumonia': Property(Types.DATE, 'date of acquiring pneumonia infection'),
        'pn_fever': Property(Types.BOOL, 'fever from non-severe pneumonia, severe pneumonia or very severe pneumonia'),
        'pn_cough': Property(Types.BOOL, 'cough from non-severe pneumonia, severe pneumonia or very severe pneumonia'),
        'pn_difficult_breathing': Property
        (Types.BOOL, 'difficult breathing from non-severe pneumonia, severe pneumonia or very severe pneumonia'),
        'pn_fast_breathing': Property(Types.BOOL, 'fast breathing from non-severe pneumonia'),
        'pn_chest_indrawing': Property(Types.BOOL, 'chest indrawing from severe pneumonia or very severe pneumonia'),
        'pn_any_general_danger_sign': Property
        (Types.BOOL, 'any danger sign - lethargic/uncounscious, not able to drink/breastfeed, convulsions and vomiting everything'),
        'pn_stridor_in_calm_child': Property(Types.BOOL, 'stridor in calm child from very severe pneumonia')
    }

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module
        """
        p = self.parameters
        dfd = pd.read_excel(
            Path(self.resourcefilepath) / 'ResourceFile_Childhood_Pneumonia.xlsx', sheet_name='Parameter_values')
        dfd.set_index("Parameter_name", inplace=True)

        p['init_prop_pneumonia_status'] = [
            dfd.loc['rp_pneumonia_age12to23mo', 'value1'],
            dfd.loc['rp_pneumonia_age12to23mo', 'value2'],
            dfd.loc['rp_pneumonia_age12to23mo', 'value3']
        ]
        p['rp_pneumonia_age12to23mo'] = dfd.loc['rp_pneumonia_age12to23mo', 'value1']
        p['rp_pneumonia_age24to59mo'] = dfd.loc['rp_pneumonia_age24to59mo', 'value1']
        p['rp_pneumonia_HIV'] = dfd.loc['rp_pneumonia_HIV', 'value1']
        p['rp_pneumonia_SAM'] = dfd.loc['rp_pneumonia_SAM', 'value1']
        p['rp_pneumonia_excl_breast'] = dfd.loc['rp_pneumonia_excl_breast', 'value1']
        p['rp_pneumonia_cont_breast'] = dfd.loc['rp_pneumonia_cont_breast', 'value1']
        p['rp_pneumonia_HHhandwashing'] = dfd.loc['rp_pneumonia_HHhandwashing', 'value1']
        p['rp_pneumonia_IAP'] = dfd.loc['rp_pneumonia_IAP', 'value1']
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
        p['base_incidence_pneumonia_by_RSV'] = [
            dfd.loc['base_incidence_pneumonia_by_RSV', 'value1'],
            dfd.loc['base_incidence_pneumonia_by_RSV', 'value2']
            ]
        p['base_incidence_pneumonia_by_rhinovirus'] = [
            dfd.loc['base_incidence_pneumonia_by_rhinovirus', 'value1'],
            dfd.loc['base_incidence_pneumonia_by_rhinovirus', 'value2']
            ]
        p['base_incidence_pneumonia_by_hMPV'] = [
            dfd.loc['base_incidence_pneumonia_by_hmpv', 'value1'],
            dfd.loc['base_incidence_pneumonia_by_hmpv', 'value2']
            ]
        p['base_incidence_pneumonia_by_parainfluenza'] = [
            dfd.loc['base_incidence_pneumonia_by_parainfluenza', 'value1'],
            dfd.loc['base_incidence_pneumonia_by_parainfluenza', 'value2']
            ]
        p['base_incidence_pneumonia_by_streptococcus'] = [
            dfd.loc['base_incidence_pneumonia_by_streptococcus', 'value1'],
            dfd.loc['base_incidence_pneumonia_by_streptococcus', 'value2']
            ]
        p['base_incidence_pneumonia_by_hib'] = [
            dfd.loc['base_incidence_pneumonia_by_RSV', 'value1'],
            dfd.loc['base_incidence_pneumonia_by_RSV', 'value2']
            ]
        p['base_incidence_pneumonia_by_TB'] = [
            dfd.loc['base_incidence_pneumonia_by_TB', 'value1'],
            dfd.loc['base_incidence_pneumonia_by_TB', 'value2']
            ]
        p['base_incidence_pneumonia_by_staph'] = [
            dfd.loc['base_incidence_pneumonia_by_staph', 'value1'],
            dfd.loc['base_incidence_pneumonia_by_staph', 'value2']
            ]
        p['base_incidence_pneumonia_by_influenza'] = [
            dfd.loc['base_incidence_pneumonia_by_influenza', 'value1'],
            dfd.loc['base_incidence_pneumonia_by_influenza', 'value2']
            ]
        p['base_incidence_pneumonia_by_jirovecii'] = [
            dfd.loc['base_incidence_pneumonia_by_jirovecii', 'value1'],
            dfd.loc['base_incidence_pneumonia_by_jirovecii', 'value2']
            ]
        p['rr_ri_pneumonia_HHhandwashing'] = dfd.loc['base_incidence_pneumonia_by_jirovecii', 'value1']
        p['rr_ri_pneumonia_HIV'] = dfd.loc['rr_ri_pneumonia_HIV', 'value1']
        p['rr_ri_pneumonia_SAM'] = dfd.loc['rr_ri_pneumonia_malnutrition', 'value1']
        p['rr_ri_pneumonia_excl_breast'] = dfd.loc['rr_ri_pneumonia_excl_breastfeeding', 'value1']
        p['rr_ri_pneumonia_cont_breast'] = dfd.loc['rr_ri_pneumonia_cont_breast', 'value1']
        p['rr_ri_pneumonia_indoor_air_pollution'] = dfd.loc['rr_ri_pneumonia_indoor_air_pollution', 'value1']
        p['rr_ri_pneumonia_pneumococcal_vaccine'] = dfd.loc['rr_ri_pneumonia_pneumococcal_vaccine', 'value1']
        p['rr_ri_pneumonia_hib_vaccine'] = dfd.loc['rr_ri_pneumonia_hib_vaccine', 'value1']
        p['rr_ri_pneumonia_influenza_vaccine'] = dfd.loc['rr_ri_pneumonia_influenza_vaccine', 'value1']
        p['rr_progress_severe_pneum_viral'] = dfd.loc['rr_progress_severe_pneum_viral', 'value1']
        p['r_progress_to_severe_pneum'] = dfd.loc['r_progress_to_severe_pneumonia', 'value1']

        p['rr_progress_severe_pneum_age12to23mo'] = dfd.loc['rr_progress_severe_pneum_age12to23mo', 'value1']
        p['rr_progress_severe_pneum_age24to59mo'] = dfd.loc['rr_progress_severe_pneum_age24to59mo', 'value1']
        p['rr_progress_severe_pneum_HIV'] = dfd.loc['rr_progress_severe_pneum_HIV', 'value1']
        p['rr_progress_severe_pneum_SAM'] = dfd.loc['rr_progress_severe_pneum_SAM', 'value1']
        p['r_progress_to_very_sev_pneum'] = dfd.loc['r_progress_to_very_sev_pneumonia', 'value1']
        p['rr_progress_very_sev_pneum_viral'] = dfd.loc['rr_progress_very_sev_pneum_viral', 'value1']
        p['rr_progress_very_sev_pneum_age12to23mo'] = dfd.loc['rr_progress_very_sev_pneum_age12to23mo', 'value1']
        p['rr_progress_very_sev_pneum_age24to59mo'] = dfd.loc['rr_progress_very_sev_pneum_age24to59mo', 'value1']
        p['rr_progress_very_sev_pneum_HIV'] = dfd.loc['rr_progress_very_sev_pneum_HIV', 'value1']
        p['rr_progress_very_sev_pneum_SAM'] = dfd.loc['rr_progress_very_sev_pneum_SAM', 'value1']
        p['r_death_pneumonia'] = dfd.loc['r_death_pneumonia', 'value1']
        p['IMCI_effectiveness_2010'] = 0.5
        p['dhs_care_seeking_2010'] = 0.6

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
        df['ri_pneumonia_status'] = False
        df['ri_pneumonia_severity'] = 'none'
        df['malnutrition'] = False
        df['has_hiv'] = False
        df['exclusive_breastfeeding'] = False
        df['continued_breastfeeding'] = False
        df['date_of_acquiring_pneumonia'] = pd.NaT

        # -------------------- ASSIGN PNEUMONIA STATUS AT BASELINE (PREVALENCE) -----------------------

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

    def initialise_simulation(self, sim):
        """
        Get ready for simulation start.
        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event for pneumonia ---------------------------------------------------
        event_pneumonia = PneumoniaEvent(self)
        sim.schedule_event(event_pneumonia, sim.date + DateOffset(months=3))

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

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

        logger.debug('This is Pneumonia, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug('This is pneumonia reporting my health values')


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

        # DEFAULTS
        df['ri_pneumonia_status'] = False
        df['ri_pneumonia_severity'] = 'none'
        df['malnutrition'] = False
        df['has_hiv'] = False
        df['exclusive_breastfeeding'] = False
        df['continued_breastfeeding'] = False
        df['date_of_acquiring_pneumonia'] = pd.NaT

        # # # # # # # # # # # # # # PNEUMONIA INCIDENCE BY ATTRIBUTABLE PATHOGEN # # # # # # # # # # # # # #

        no_pneumonia0 = df.is_alive & (df.ri_pneumonia_status == False) & (df.age_exact_years < 1)
        no_pneumonia1 = df.is_alive & (df.ri_pneumonia_status == False) &\
                        (df.age_exact_years >= 1) & (df.age_exact_years < 5)
        no_pneumonia_under5 = df.is_alive & (df.ri_pneumonia_status == False) & (df.age_years < 5)
        current_no_pneumonia = df.index[no_pneumonia_under5]

        pneumonia_RSV0 = pd.Series(m.base_incidence_pneumonia_by_RSV[0], index=df.index[no_pneumonia0])
        pneumonia_RSV1 = pd.Series(m.base_incidence_pneumonia_by_RSV[1], index=df.index[no_pneumonia1])

        pneumonia_rhinovirus0 = pd.Series(m.base_incidence_pneumonia_by_rhinovirus[0], index=df.index[no_pneumonia0])
        pneumonia_rhinovirus1 = pd.Series(m.base_incidence_pneumonia_by_rhinovirus[1], index=df.index[no_pneumonia1])

        pneumonia_hMPV0 = pd.Series(m.base_incidence_pneumonia_by_hMPV[0], index=df.index[no_pneumonia0])
        pneumonia_hMPV1 = pd.Series(m.base_incidence_pneumonia_by_hMPV[1], index=df.index[no_pneumonia1])

        pneumonia_parainfluenza0 = pd.Series(m.base_incidence_pneumonia_by_parainfluenza[0],
                                             index=df.index[no_pneumonia0])
        pneumonia_parainfluenza1 = pd.Series(m.base_incidence_pneumonia_by_parainfluenza[1],
                                             index=df.index[no_pneumonia1])

        pneumonia_streptococcus0 = pd.Series(m.base_incidence_pneumonia_by_streptococcus[0],
                                             index=df.index[no_pneumonia0])
        pneumonia_streptococcus1 = pd.Series(m.base_incidence_pneumonia_by_streptococcus[1],
                                             index=df.index[no_pneumonia1])

        pneumonia_hib0 = pd.Series(m.base_incidence_pneumonia_by_hib[0], index=df.index[no_pneumonia0])
        pneumonia_hib1 = pd.Series(m.base_incidence_pneumonia_by_hib[1], index=df.index[no_pneumonia1])

        pneumonia_TB0 = pd.Series(m.base_incidence_pneumonia_by_TB[0], index=df.index[no_pneumonia0])
        pneumonia_TB1 = pd.Series(m.base_incidence_pneumonia_by_TB[1], index=df.index[no_pneumonia1])

        pneumonia_staph0 = pd.Series(m.base_incidence_pneumonia_by_staph[0], index=df.index[no_pneumonia0])
        pneumonia_staph1 = pd.Series(m.base_incidence_pneumonia_by_staph[1], index=df.index[no_pneumonia1])

        pneumonia_influenza0 = pd.Series(m.base_incidence_pneumonia_by_influenza[0], index=df.index[no_pneumonia0])
        pneumonia_influenza1 = pd.Series(m.base_incidence_pneumonia_by_influenza[1], index=df.index[no_pneumonia1])

        pneumonia_jirovecii0 = pd.Series(m.base_incidence_pneumonia_by_jirovecii[0], index=df.index[no_pneumonia0])
        pneumonia_jirovecii1 = pd.Series(m.base_incidence_pneumonia_by_jirovecii[1], index=df.index[no_pneumonia1])

        # concatenating plus sorting
        eff_prob_RSV = pd.concat([pneumonia_RSV0, pneumonia_RSV1], axis=0).sort_index()
        eff_prob_rhinovirus = pd.concat([pneumonia_rhinovirus0, pneumonia_rhinovirus1], axis=0).sort_index()
        eff_prob_hMPV = pd.concat([pneumonia_hMPV0, pneumonia_hMPV1], axis=0).sort_index()
        eff_prob_parainfluenza = pd.concat([pneumonia_parainfluenza0, pneumonia_parainfluenza1], axis=0).sort_index()
        eff_prob_streptococcus = pd.concat([pneumonia_streptococcus0, pneumonia_streptococcus1], axis=0).sort_index()
        eff_prob_hib = pd.concat([pneumonia_hib0, pneumonia_hib1], axis=0).sort_index()
        eff_prob_TB = pd.concat([pneumonia_TB0, pneumonia_TB1], axis=0).sort_index()
        eff_prob_staph = pd.concat([pneumonia_staph0, pneumonia_staph1], axis=0).sort_index()
        eff_prob_influenza = pd.concat([pneumonia_influenza0, pneumonia_influenza1], axis=0).sort_index()
        eff_prob_jirovecii = pd.concat([pneumonia_jirovecii0, pneumonia_jirovecii1], axis=0).sort_index()

        eff_prob_all_pathogens = \
            pd.concat([eff_prob_RSV, eff_prob_rhinovirus, eff_prob_hMPV, eff_prob_parainfluenza, eff_prob_streptococcus,
                       eff_prob_hib, eff_prob_TB, eff_prob_staph, eff_prob_influenza, eff_prob_jirovecii], axis=1)

        # adding the effects of risk factors
        eff_prob_all_pathogens.loc[no_pneumonia_under5 & df.li_no_access_handwashing == False] \
            *= m.rr_ri_pneumonia_HHhandwashing
        eff_prob_all_pathogens.loc[no_pneumonia_under5 & (df.has_hiv == True)] \
            *= m.rr_ri_pneumonia_HIV
        eff_prob_all_pathogens.loc[no_pneumonia_under5 & df.malnutrition == True] \
            *= m.rr_ri_pneumonia_SAM
        eff_prob_all_pathogens.loc[
            no_pneumonia_under5 & df.exclusive_breastfeeding == True & (df.age_exact_years <= 0.5)] \
            *= m.rr_ri_pneumonia_excl_breast
        eff_prob_all_pathogens.loc[
            no_pneumonia_under5 & df.continued_breastfeeding == True & (df.age_exact_years > 0.5) &
            (df.age_exact_years < 2)] *= m.rr_ri_pneumonia_cont_breast
        eff_prob_all_pathogens.loc[
            no_pneumonia_under5 & (df.li_wood_burn_stove == False)] *= m.rr_ri_pneumonia_indoor_air_pollution
        eff_prob_all_pathogens.loc[
            no_pneumonia_under5 & (df.pneumococcal_vaccination == True)] *= m.rr_ri_pneumonia_pneumococcal_vaccine
        eff_prob_all_pathogens.loc[
            no_pneumonia_under5 & (df.hib_vaccination == True)] *= m.rr_ri_pneumonia_hib_vaccine
        eff_prob_all_pathogens.loc[
            no_pneumonia_under5 & (df.influenza_vaccination == True)] *= m.rr_ri_pneumonia_influenza_vaccine

        # # # # # # # # OPTION 1  # # # # # # # #
        # # # # # # # #  # # # # # # # #  # # # # # # # #

        random_draw_all = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        dfx = pd.concat([eff_prob_all_pathogens, random_draw_all], axis=1)
        dfx.columns = ['eff_prob_RSV', 'eff_prob_rhinovirus', 'eff_prob_hMPV', 'eff_prob_parainfluenza',
                       'eff_prob_streptococcus', 'eff_prob_hib', 'eff_prob_TB', 'eff_prob_staph',
                       'eff_prob_influenza', 'eff_prob_jirovecii', 'random_draw_all']
        dfx['eff_prob_none'] = 1 - (dfx.eff_prob_RSV + dfx.eff_prob_rhinovirus + dfx.eff_prob_hMPV +
                                    dfx.eff_prob_parainfluenza + dfx.eff_prob_streptococcus + dfx.eff_prob_hib +
                                    dfx.eff_prob_TB + dfx.eff_prob_staph + dfx.eff_prob_influenza +
                                    dfx.eff_prob_jirovecii)

        pneum_RSV_idx = dfx.index[(dfx.eff_prob_none < dfx.random_draw_all) &
                      ((dfx.eff_prob_none + dfx.eff_prob_RSV) > dfx.random_draw_all)]
        df.loc[pneum_RSV_idx, 'ri_pneumonia_pathogen'] = 'RSV'
        df.loc[pneum_RSV_idx, 'ri_pneumonia_status'] = True
        df.loc[(df.ri_pneumonia_pathogen == 'RSV') & (df.age_exact_years < 0.1667),
               'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[(df.ri_pneumonia_pathogen == 'RSV') & (df.age_exact_years > 0.1667) & (
            df.age_years < 5), 'ri_pneumonia_severity'] = 'pneumonia'

        pneum_rhinovirus_idx = \
            dfx.index[(dfx.eff_prob_none + dfx.eff_prob_RSV < dfx.random_draw_all) &
                      ((dfx.eff_prob_none + dfx.eff_prob_RSV + dfx.eff_prob_rhinovirus) > dfx.random_draw_all)]
        df.loc[pneum_rhinovirus_idx, 'ri_pneumonia_pathogen'] = 'rhinovirus'
        df.loc[pneum_rhinovirus_idx, 'ri_pneumonia_status'] = True
        df.loc[(df.ri_pneumonia_pathogen == 'rhinovirus') & (df.age_exact_years < 0.1667),
               'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[(df.ri_pneumonia_pathogen == 'rhinovirus') & (df.age_exact_years > 0.1667) & (
            df.age_years < 5), 'ri_pneumonia_severity'] = 'pneumonia'

        pneum_hMPV_idx = \
            dfx.index[(dfx.eff_prob_none + dfx.eff_prob_RSV + dfx.eff_prob_rhinovirus < dfx.random_draw_all) &
                      ((dfx.eff_prob_none + dfx.eff_prob_RSV + dfx.eff_prob_rhinovirus + dfx.eff_prob_hMPV) >
                       dfx.random_draw_all)]
        df.loc[pneum_hMPV_idx, 'ri_pneumonia_pathogen'] = 'hMPV'
        df.loc[pneum_hMPV_idx, 'ri_pneumonia_status'] = True
        df.loc[(df.ri_pneumonia_pathogen == 'hMPV') & (df.age_exact_years < 0.1667),
               'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[(df.ri_pneumonia_pathogen == 'hMPV') & (df.age_exact_years > 0.1667) & (
            df.age_years < 5), 'ri_pneumonia_severity'] = 'pneumonia'

        pneum_parainfluenza_idx = \
            dfx.index[(dfx.eff_prob_none + dfx.eff_prob_RSV + dfx.eff_prob_rhinovirus + dfx.eff_prob_hMPV
                       < dfx.random_draw_all) & ((dfx.eff_prob_none + dfx.eff_prob_RSV + dfx.eff_prob_rhinovirus +
                                                  dfx.eff_prob_hMPV + dfx.eff_prob_parainfluenza) > dfx.random_draw_all)]
        df.loc[pneum_parainfluenza_idx, 'ri_pneumonia_pathogen'] = 'parainfluenza'
        df.loc[pneum_parainfluenza_idx, 'ri_pneumonia_status'] = True
        df.loc[(df.ri_pneumonia_pathogen == 'parainfluenza') & (df.age_exact_years < 0.1667),
               'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[(df.ri_pneumonia_pathogen == 'parainfluenza') & (df.age_exact_years > 0.1667) & (
            df.age_years < 5), 'ri_pneumonia_severity'] = 'pneumonia'

        pneum_streptococcus_idx = \
            dfx.index[(dfx.eff_prob_none + dfx.eff_prob_RSV + dfx.eff_prob_rhinovirus + + dfx.eff_prob_hMPV +
                       dfx.eff_prob_parainfluenza < dfx.random_draw_all) &
                      ((dfx.eff_prob_none + dfx.eff_prob_RSV + dfx.eff_prob_rhinovirus + dfx.eff_prob_hMPV +
                        dfx.eff_prob_parainfluenza + dfx.eff_prob_streptococcus) > dfx.random_draw_all)]
        df.loc[pneum_streptococcus_idx, 'ri_pneumonia_pathogen'] = 'streptococcus'
        df.loc[pneum_streptococcus_idx, 'ri_pneumonia_status'] = True
        df.loc[(df.ri_pneumonia_pathogen == 'streptococcus') & (df.age_exact_years < 0.1667),
               'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[(df.ri_pneumonia_pathogen == 'streptococcus') & (df.age_exact_years > 0.1667) & (
            df.age_years < 5), 'ri_pneumonia_severity'] = 'pneumonia'

        pneum_hib_idx = \
            dfx.index[(dfx.eff_prob_none + dfx.eff_prob_RSV + dfx.eff_prob_rhinovirus + + dfx.eff_prob_hMPV +
                       dfx.eff_prob_parainfluenza + dfx.eff_prob_streptococcus < dfx.random_draw_all) &
                      ((dfx.eff_prob_none + dfx.eff_prob_RSV + dfx.eff_prob_rhinovirus + dfx.eff_prob_hMPV +
                        dfx.eff_prob_parainfluenza + dfx.eff_prob_streptococcus + dfx.eff_prob_hib) >
                       dfx.random_draw_all)]
        df.loc[pneum_hib_idx, 'ri_pneumonia_pathogen'] = 'hib'
        df.loc[pneum_hib_idx, 'ri_pneumonia_status'] = True
        df.loc[(df.ri_pneumonia_pathogen == 'hib') & (df.age_exact_years < 0.1667),
               'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[(df.ri_pneumonia_pathogen == 'hib') & (df.age_exact_years > 0.1667) & (
            df.age_years < 5), 'ri_pneumonia_severity'] = 'pneumonia'

        pneum_TB_idx = \
            dfx.index[(dfx.eff_prob_none + dfx.eff_prob_RSV + dfx.eff_prob_rhinovirus + + dfx.eff_prob_hMPV +
                       dfx.eff_prob_parainfluenza + dfx.eff_prob_streptococcus + dfx.eff_prob_streptococcus +
                       dfx.eff_prob_hib < dfx.random_draw_all) &
                      ((dfx.eff_prob_none + dfx.eff_prob_RSV + dfx.eff_prob_rhinovirus + dfx.eff_prob_hMPV +
                        dfx.eff_prob_parainfluenza + dfx.eff_prob_streptococcus + dfx.eff_prob_hib + dfx.eff_prob_TB) >
                       dfx.random_draw_all)]
        df.loc[pneum_TB_idx, 'ri_pneumonia_pathogen'] = 'TB'
        df.loc[pneum_TB_idx, 'ri_pneumonia_status'] = True
        df.loc[(df.ri_pneumonia_pathogen == 'TB') & (df.age_exact_years < 0.1667),
               'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[(df.ri_pneumonia_pathogen == 'TB') & (df.age_exact_years > 0.1667) & (
            df.age_years < 5), 'ri_pneumonia_severity'] = 'pneumonia'

        pneum_staph_idx = \
            dfx.index[(dfx.eff_prob_none + dfx.eff_prob_RSV + dfx.eff_prob_rhinovirus + + dfx.eff_prob_hMPV +
                       dfx.eff_prob_parainfluenza + dfx.eff_prob_streptococcus + dfx.eff_prob_streptococcus +
                       dfx.eff_prob_hib + dfx.eff_prob_TB < dfx.random_draw_all) &
                      ((dfx.eff_prob_none + dfx.eff_prob_RSV + dfx.eff_prob_rhinovirus + dfx.eff_prob_hMPV +
                        dfx.eff_prob_parainfluenza + dfx.eff_prob_streptococcus + dfx.eff_prob_hib + dfx.eff_prob_TB +
                        dfx.eff_prob_staph) > dfx.random_draw_all)]
        df.loc[pneum_staph_idx, 'ri_pneumonia_pathogen'] = 'staph'
        df.loc[pneum_staph_idx, 'ri_pneumonia_status'] = True
        df.loc[(df.ri_pneumonia_pathogen == 'staph') & (df.age_exact_years < 0.1667),
               'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[(df.ri_pneumonia_pathogen == 'staph') & (df.age_exact_years > 0.1667) & (
            df.age_years < 5), 'ri_pneumonia_severity'] = 'pneumonia'

        pneum_influenza_idx = \
            dfx.index[(dfx.eff_prob_none + dfx.eff_prob_RSV + dfx.eff_prob_rhinovirus + + dfx.eff_prob_hMPV +
                       dfx.eff_prob_parainfluenza + dfx.eff_prob_streptococcus + dfx.eff_prob_streptococcus +
                       dfx.eff_prob_hib + dfx.eff_prob_TB + dfx.eff_prob_staph < dfx.random_draw_all) &
                      ((dfx.eff_prob_none + dfx.eff_prob_RSV + dfx.eff_prob_rhinovirus + dfx.eff_prob_hMPV +
                        dfx.eff_prob_parainfluenza + dfx.eff_prob_streptococcus + dfx.eff_prob_hib + dfx.eff_prob_TB +
                        dfx.eff_prob_staph + dfx.eff_prob_influenza) > dfx.random_draw_all)]
        df.loc[pneum_influenza_idx, 'ri_pneumonia_pathogen'] = 'influenza'
        df.loc[pneum_influenza_idx, 'ri_pneumonia_status'] = True
        df.loc[(df.ri_pneumonia_pathogen == 'influenza') & (df.age_exact_years < 0.1667),
               'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[(df.ri_pneumonia_pathogen == 'influenza') & (df.age_exact_years > 0.1667) & (
            df.age_years < 5), 'ri_pneumonia_severity'] = 'pneumonia'

        pneum_jirovecii_idx = \
            dfx.index[(dfx.eff_prob_none + dfx.eff_prob_RSV + dfx.eff_prob_rhinovirus + + dfx.eff_prob_hMPV +
                       dfx.eff_prob_parainfluenza + dfx.eff_prob_streptococcus + dfx.eff_prob_streptococcus +
                       dfx.eff_prob_hib + dfx.eff_prob_TB + dfx.eff_prob_staph + dfx.eff_prob_influenza
                       < dfx.random_draw_all) &
                      ((dfx.eff_prob_none + dfx.eff_prob_RSV + dfx.eff_prob_rhinovirus + dfx.eff_prob_hMPV +
                        dfx.eff_prob_parainfluenza + dfx.eff_prob_streptococcus + dfx.eff_prob_hib + dfx.eff_prob_TB +
                        dfx.eff_prob_staph + dfx.eff_prob_influenza + dfx.eff_prob_jirovecii) > dfx.random_draw_all)]
        df.loc[pneum_jirovecii_idx, 'ri_pneumonia_pathogen'] = 'P. jirovecii'
        df.loc[pneum_jirovecii_idx, 'ri_pneumonia_status'] = True
        df.loc[(df.ri_pneumonia_pathogen == 'P. jirovecii') & (df.age_exact_years < 0.1667),
               'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[(df.ri_pneumonia_pathogen == 'P. jirovecii') & (df.age_exact_years > 0.1667) & (
            df.age_years < 5), 'ri_pneumonia_severity'] = 'pneumonia'

        # # # # # # # #  # # # # # # # #  # # # # # # # #
        # # # # # # # # OPTION 2  # # # # # # # #
        # # # # # # # #  # # # # # # # #  # # # # # # # #
        random_draw1 = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        pneumonia_by_RSV = eff_prob_RSV > random_draw1
        pneum_RSV_idx = eff_prob_RSV.index[pneumonia_by_RSV]
        df.loc[pneum_RSV_idx, 'ri_pneumonia_pathogen'] = 'RSV'
        df.loc[pneum_RSV_idx, 'ri_pneumonia_status'] = True
        df.loc[(df.ri_pneumonia_pathogen == 'RSV') & (df.age_exact_years < 0.1667),
               'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[(df.ri_pneumonia_pathogen == 'RSV') & (df.age_exact_years > 0.1667) & (
            df.age_years < 5), 'ri_pneumonia_severity'] = 'pneumonia'

        random_draw2 = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        pneumonia_by_rhinovirus = eff_prob_rhinovirus > random_draw2
        pneum_rhinovirus_idx = eff_prob_rhinovirus.index[pneumonia_by_rhinovirus]
        df.loc[pneum_rhinovirus_idx, 'ri_pneumonia_pathogen'] = 'rhinovirus'
        df.loc[pneum_rhinovirus_idx, 'ri_pneumonia_status'] = True
        df.loc[(df.ri_pneumonia_pathogen == 'rhinovirus') & (df.age_exact_years < 0.1667),
               'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[(df.ri_pneumonia_pathogen == 'rhinovirus') & (df.age_exact_years > 0.1667) & (
            df.age_years < 5), 'ri_pneumonia_severity'] = 'pneumonia'

        random_draw3 = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        pneumonia_by_hMPV = eff_prob_hMPV > random_draw3
        pneum_hMPV_idx = eff_prob_hMPV.index[pneumonia_by_hMPV]
        df.loc[pneum_hMPV_idx, 'ri_pneumonia_pathogen'] = 'hMPV'
        df.loc[pneum_hMPV_idx, 'ri_pneumonia_status'] = True
        df.loc[(df.ri_pneumonia_pathogen == 'hMPV') & (df.age_exact_years < 0.1667),
               'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[(df.ri_pneumonia_pathogen == 'hMPV') & (df.age_exact_years > 0.1667) & (
            df.age_years < 5), 'ri_pneumonia_severity'] = 'pneumonia'

        random_draw4 = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        pneumonia_by_parainfluenza = eff_prob_parainfluenza > random_draw4
        pneum_parainfluenza_idx = eff_prob_parainfluenza.index[pneumonia_by_parainfluenza]
        df.loc[pneum_parainfluenza_idx, 'ri_pneumonia_pathogen'] = 'parainfluenza'
        df.loc[pneum_parainfluenza_idx, 'ri_pneumonia_status'] = True
        df.loc[(df.ri_pneumonia_pathogen == 'parainfluenza') & (df.age_exact_years < 0.1667),
               'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[(df.ri_pneumonia_pathogen == 'parainfluenza') & (df.age_exact_years > 0.1667) & (
            df.age_years < 5), 'ri_pneumonia_severity'] = 'pneumonia'

        random_draw5 = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        pneumonia_by_streptococcus = eff_prob_streptococcus > random_draw5
        pneum_streptococcus_idx = eff_prob_streptococcus.index[pneumonia_by_streptococcus]
        df.loc[pneum_streptococcus_idx, 'ri_pneumonia_pathogen'] = 'streptococcus'
        df.loc[pneum_streptococcus_idx, 'ri_pneumonia_status'] = True
        df.loc[(df.ri_pneumonia_pathogen == 'streptococcus') & (df.age_exact_years < 0.1667),
               'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[(df.ri_pneumonia_pathogen == 'streptococcus') & (df.age_exact_years > 0.1667) & (
            df.age_years < 5), 'ri_pneumonia_severity'] = 'pneumonia'

        random_draw6 = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        pneumonia_by_hib = eff_prob_hib > random_draw6
        pneum_hib_idx = eff_prob_hib.index[pneumonia_by_hib]
        df.loc[pneum_hib_idx, 'ri_pneumonia_pathogen'] = 'hib'
        df.loc[pneum_hib_idx, 'ri_pneumonia_status'] = True
        df.loc[(df.ri_pneumonia_pathogen == 'hib') & (df.age_exact_years < 0.1667),
               'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[(df.ri_pneumonia_pathogen == 'hib') & (df.age_exact_years > 0.1667) & (
            df.age_years < 5), 'ri_pneumonia_severity'] = 'pneumonia'

        random_draw7 = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        pneumonia_by_TB = eff_prob_TB > random_draw7
        pneum_TB_idx = eff_prob_TB.index[pneumonia_by_TB]
        df.loc[pneum_TB_idx, 'ri_pneumonia_pathogen'] = 'TB'
        df.loc[pneum_TB_idx, 'ri_pneumonia_status'] = True
        df.loc[(df.ri_pneumonia_pathogen == 'TB') & (df.age_exact_years < 0.1667),
               'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[(df.ri_pneumonia_pathogen == 'TB') & (df.age_exact_years > 0.1667) & (
            df.age_years < 5), 'ri_pneumonia_severity'] = 'pneumonia'

        random_draw8 = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        pneumonia_by_staph = eff_prob_staph > random_draw8
        pneum_staph_idx = eff_prob_staph.index[pneumonia_by_staph]
        df.loc[pneum_staph_idx, 'ri_pneumonia_pathogen'] = 'staph'
        df.loc[pneum_staph_idx, 'ri_pneumonia_status'] = True
        df.loc[(df.ri_pneumonia_pathogen == 'staph') & (df.age_exact_years < 0.1667),
               'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[(df.ri_pneumonia_pathogen == 'staph') & (df.age_exact_years > 0.1667) & (
            df.age_years < 5), 'ri_pneumonia_severity'] = 'pneumonia'

        random_draw9 = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        pneumonia_by_influenza = eff_prob_influenza > random_draw9
        pneum_influenza_idx = eff_prob_influenza.index[pneumonia_by_influenza]
        df.loc[pneum_influenza_idx, 'ri_pneumonia_pathogen'] = 'influenza'
        df.loc[pneum_influenza_idx, 'ri_pneumonia_status'] = True
        df.loc[(df.ri_pneumonia_pathogen == 'influenza') & (df.age_exact_years < 0.1667),
               'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[(df.ri_pneumonia_pathogen == 'influenza') & (df.age_exact_years > 0.1667) & (
            df.age_years < 5), 'ri_pneumonia_severity'] = 'pneumonia'

        random_draw10 = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        pneumonia_by_jirovecii = eff_prob_jirovecii > random_draw10
        pneum_jirovecii_idx = eff_prob_jirovecii.index[pneumonia_by_jirovecii]
        df.loc[pneum_jirovecii_idx, 'ri_pneumonia_pathogen'] = 'P. jirovecii'
        df.loc[pneum_jirovecii_idx, 'ri_pneumonia_status'] = True
        df.loc[(df.ri_pneumonia_pathogen == 'P. jirovecii') & (df.age_exact_years < 0.1667),
               'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[(df.ri_pneumonia_pathogen == 'P. jirovecii') & (df.age_exact_years > 0.1667) & (
                df.age_years < 5), 'ri_pneumonia_severity'] = 'pneumonia'

        # NOTE: NON-SEVERE PNEUMONIA ONLY IN 2-59 MONTHS
        pn_current_pneumonia_idx = df.index[
            df.is_alive & (df.age_exact_years > 0.1667) & (df.age_years < 5) &
            (df.ri_pneumonia_severity == 'pneumonia') | (df.is_alive & (df.age_exact_years < 0.1667) &
                                                         (df.ri_pneumonia_severity == 'severe pneumonia'))]

        # # # # # # # WHEN THEY GET THE DISEASE - DATE -----------------------------------------------------------

        random_draw_days = np.random.randint(0, 90, size=len(pn_current_pneumonia_idx))
        td = pd.to_timedelta(random_draw_days, unit='d')
        date_of_aquisition = self.sim.date + td
        df.loc[pn_current_pneumonia_idx, 'date_of_acquiring_pneumonia'] = date_of_aquisition

        # # # # # # # # # # # # # # # # # # SYMPTOMS FROM NON-SEVERE PNEUMONIA # # # # # # # # # # # # # # # # # #
        # fast breathing
        df.loc[pn_current_pneumonia_idx, 'pn_fast_breathing'] = True

        # cough
        eff_prob_cough = pd.Series(0.89, index=pn_current_pneumonia_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_pneumonia_idx)),
                                index=df.index[(df.age_exact_years > 0.1667) & (df.age_years < 5) & df.is_alive &
                                               (df.ri_pneumonia_severity == 'pneumonia')])
        dfx = pd.concat([eff_prob_cough, random_draw], axis=1)
        dfx.columns = ['eff_prob_cough', 'random number']
        idx_cough = dfx.index[dfx.eff_prob_cough > random_draw]
        df.loc[idx_cough, 'pn_cough'] = True

        # difficult breathing
        eff_prob_difficult_breathing = pd.Series(0.89, index=pn_current_pneumonia_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_pneumonia_idx)),
                                index=df.index[(df.age_exact_years > 0.1667) &
                                               (df.age_years < 5) & df.is_alive & (
                                                       df.ri_pneumonia_severity == 'pneumonia')])
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
            event = HSI_ICCM(self.module, person_id=individual)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=None
                                                                )

        # ---------------------------------------------------------------------------------------------------
        # / # / # / # / # / # / # / # / # / PROGRESS TO SEVERE PNEUMONIA # / # / # / # / # / # / # / # / # /
        # ---------------------------------------------------------------------------------------------------
        eff_prob_prog_severe_pneum = pd.Series(m.r_progress_to_severe_pneum,
                                               index=df.index[df.is_alive & (df.ri_pneumonia_severity == 'pneumonia')
                                                              & (df.age_exact_years > 0.1667) & (df.age_years < 5)])

        eff_prob_prog_severe_pneum.loc[
            df.is_alive & (df.ri_pneumonia_severity == 'pneumonia') & (df.age_exact_years >= 1) &
            (df.age_exact_years < 2)] *= m.rr_progress_severe_pneum_age12to23mo
        eff_prob_prog_severe_pneum.loc[
            df.is_alive & (df.ri_pneumonia_severity == 'pneumonia') & (df.age_exact_years >= 2) &
            (df.age_exact_years < 5)] *= m.rr_progress_severe_pneum_age24to59mo
        eff_prob_prog_severe_pneum.loc[
            df.is_alive & (df.ri_pneumonia_severity == 'pneumonia') & df.has_hiv == True &
            (df.age_years < 5)] *= m.rr_progress_severe_pneum_HIV
        eff_prob_prog_severe_pneum.loc[
            df.is_alive & (df.ri_pneumonia_severity == 'pneumonia') & df.malnutrition == True &
            (df.age_years < 5)] *= m.rr_progress_severe_pneum_SAM
        eff_prob_prog_severe_pneum.loc[
            df.is_alive & (df.ri_pneumonia_severity == 'pneumonia') & (df.ri_pneumonia_pathogen_type == 'viral') &
            (df.age_years < 5)] *= m.rr_progress_severe_pneum_viral

        pn_current_pneumonia_idx = \
            df.index[df.is_alive & (df.age_years < 5) & (df.ri_pneumonia_severity == 'pneumonia')]

        random_draw = pd.Series(rng.random_sample(size=len(pn_current_pneumonia_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.ri_pneumonia_severity == 'pneumonia')])
        progress_severe_pneum = eff_prob_prog_severe_pneum > random_draw
        progress_severe_pneum_idx = eff_prob_prog_severe_pneum.index[progress_severe_pneum]
        df.loc[progress_severe_pneum_idx, 'ri_pneumonia_severity'] = 'severe pneumonia'

        # date of progression to severe pneumonia for 2-59 months
        random_draw_days = np.random.randint(0, 14, size=len(progress_severe_pneum_idx))
        td = pd.to_timedelta(random_draw_days, unit='d')
        date_prog_severe_pneum = date_of_aquisition + td

        # # # # # # # # # # # # # # # # # # SYMPTOMS FROM SEVERE PNEUMONIA # # # # # # # # # # # # # # # # # #

        pn_current_severe_pneum_idx = df.index[df.is_alive & (df.age_years < 5) &
                                               (df.ri_pneumonia_severity == 'severe pneumonia')]
        for individual in pn_current_severe_pneum_idx:
            df.at[individual, 'pn_chest_indrawing'] = True

        eff_prob_cough = pd.Series(0.96, index=pn_current_severe_pneum_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_severe_pneum_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.ri_pneumonia_severity == 'severe pneumonia')])
        dfx = pd.concat([eff_prob_cough, random_draw], axis=1)
        dfx.columns = ['eff_prob_cough', 'random number']
        idx_cough = dfx.index[dfx.eff_prob_cough > random_draw]
        df.loc[idx_cough, 'pn_cough'] = True

        eff_prob_difficult_breathing = pd.Series(0.40, index=pn_current_severe_pneum_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_severe_pneum_idx)),
                                index=df.index[
                                    (df.age_years < 5) & df.is_alive & (df.ri_pneumonia_severity == 'severe pneumonia')])
        dfx = pd.concat([eff_prob_difficult_breathing, random_draw], axis=1)
        dfx.columns = ['eff_prob_difficult_breathing', 'random number']
        idx_difficult_breathing = dfx.index[dfx.eff_prob_difficult_breathing > random_draw]
        df.loc[idx_difficult_breathing, 'pn_difficult_breathing'] = True

        eff_prob_fast_breathing = pd.Series(0.96, index=pn_current_severe_pneum_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_severe_pneum_idx)),
                                index=df.index[
                                    (df.age_years < 5) & df.is_alive & (df.ri_pneumonia_severity == 'severe pneumonia')])
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
        if seeks_care.sum() > 0:
            for person_index in seeks_care.index[seeks_care is True]:
                logger.debug(
                    'This is PneumoniaEvent, scheduling HSI_Sick_Child_Seeks_Care_From_HSA for person %d',
                    person_index)
                event = HSI_ICCM(self.module['iCCM'], person_id=person_index)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=2,
                                                                    topen=date_prog_severe_pneum,
                                                                    tclose=date_prog_severe_pneum + DateOffset(weeks=2)
                                                                    )
        else:
            logger.debug(
                'This is PneumoniaEvent, There is no one with new pneumonia symptoms so no new healthcare seeking')

        # --------------------------------------------------------------------------------------------------------
        # / # / # / # / # / # / # / # / # / PROGRESS TO VERY SEVERE PNEUMONIA # / # / # / # / # / # / # / # / # /
        # --------------------------------------------------------------------------------------------------------
        eff_prob_prog_very_sev_pneum = \
            pd.Series(m.r_progress_to_very_sev_pneum, index=df.index[
                df.is_alive & (df.ri_pneumonia_severity == 'severe pneumonia') & (df.age_years < 5)])

        eff_prob_prog_very_sev_pneum.loc[
            df.is_alive & (df.ri_pneumonia_severity == 'severe pneumonia') & (df.age_exact_years >= 1) &
            (df.age_exact_years < 2)] *= m.rr_progress_very_sev_pneum_age12to23mo
        eff_prob_prog_very_sev_pneum.loc[
            df.is_alive & (df.ri_pneumonia_severity == 'severe pneumonia') & (df.age_exact_years >= 2) &
            (df.age_exact_years < 5)] *= m.rr_progress_very_sev_pneum_age24to59mo
        eff_prob_prog_very_sev_pneum.loc[
            df.is_alive & (df.ri_pneumonia_severity == 'severe pneumonia') & df.has_hiv == True &
            (df.age_years < 5)] *= m.rr_progress_very_sev_pneum_HIV
        eff_prob_prog_very_sev_pneum.loc[
            df.is_alive & (df.ri_pneumonia_severity == 'severe pneumonia') & df.malnutrition == True &
            (df.age_years < 5)] *= m.rr_progress_very_sev_pneum_SAM
        eff_prob_prog_very_sev_pneum.loc[
            df.is_alive & (df.ri_pneumonia_severity == 'severe pneumonia') & (df.ri_pneumonia_pathogen_type == 'viral')
            & (df.age_years < 5)] *= m.rr_progress_very_sev_pneum_viral

        pn_current_severe_pneumonia_idx = \
            df.index[df.is_alive & (df.age_years < 5) & (df.ri_pneumonia_severity == 'severe pneumonia')]

        random_draw = pd.Series(rng.random_sample(size=len(pn_current_severe_pneumonia_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.ri_pneumonia_severity == 'severe pneumonia')])
        progress_very_sev_pneum = eff_prob_prog_very_sev_pneum > random_draw
        progress_very_sev_pneum_idx = eff_prob_prog_very_sev_pneum.index[progress_very_sev_pneum]
        df.loc[progress_very_sev_pneum_idx, 'ri_pneumonia_severity'] = 'very severe pneumonia'

        # date of progression to very severe pneumonia for 2-59 months
        random_draw_days = np.random.randint(0, 7, size=len(progress_very_sev_pneum_idx))
        td = pd.to_timedelta(random_draw_days, unit='d')
        date_prog_very_sev_pneum = date_prog_severe_pneum + td

        # # # # # # # # # # # # # # # # # # SYMPTOMS FROM VERY SEVERE PNEUMONIA # # # # # # # # # # # # # # # # # #

        pn_current_very_sev_pneum_idx = df.index[df.is_alive & (df.age_years < 5) &
                                                 (df.ri_pneumonia_severity == 'very severe pneumonia')]

        eff_prob_cough = pd.Series(0.857, index=pn_current_very_sev_pneum_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_very_sev_pneum_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.ri_pneumonia_severity == 'very severe pneumonia')])
        dfx = pd.concat([eff_prob_cough, random_draw], axis=1)
        dfx.columns = ['eff_prob_cough', 'random number']
        idx_cough = dfx.index[dfx.eff_prob_cough > random_draw]
        df.loc[idx_cough, 'pn_cough'] = True

        eff_prob_difficult_breathing = pd.Series(0.43, index=pn_current_very_sev_pneum_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_very_sev_pneum_idx)),
                                index=df.index[
                                    (df.age_years < 5) & df.is_alive & (
                                            df.ri_pneumonia_severity == 'very severe pneumonia')])
        dfx = pd.concat([eff_prob_difficult_breathing, random_draw], axis=1)
        dfx.columns = ['eff_prob_difficult_breathing', 'random number']
        idx_difficult_breathing = dfx.index[dfx.eff_prob_difficult_breathing > random_draw]
        df.loc[idx_difficult_breathing, 'pn_difficult_breathing'] = True

        eff_prob_fast_breathing = pd.Series(0.857, index=pn_current_very_sev_pneum_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_very_sev_pneum_idx)),
                                index=df.index[
                                    (df.age_years < 5) & df.is_alive & (
                                            df.ri_pneumonia_severity == 'very severe pneumonia')])
        dfx = pd.concat([eff_prob_fast_breathing, random_draw], axis=1)
        dfx.columns = ['eff_prob_fast_breathing', 'random number']
        idx_fast_breathing = dfx.index[dfx.eff_prob_fast_breathing > random_draw]
        df.loc[idx_fast_breathing, 'pn_fast_breathing'] = True

        eff_prob_chest_indrawing = pd.Series(0.76, index=pn_current_very_sev_pneum_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_very_sev_pneum_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.ri_pneumonia_severity == 'very severe pneumonia')])
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
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(individual, symptom_code=1)  # what happens with multiple symptoms of different severity??
            seeks_care[individual] = self.module.rng.rand() < prob
        if seeks_care.sum() > 0:
            for person_index in seeks_care.index[seeks_care is True]:
                logger.debug(
                    'This is PneumoniaEvent, scheduling HSI_Sick_Child_Seeks_Care_From_HSA for person %d',
                    person_index)
                event = HSI_ICCM(self.module['iCCM'], person_id=person_index)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=2,
                                                                    topen=date_prog_very_sev_pneum,
                                                                    tclose=date_prog_very_sev_pneum + DateOffset(weeks=2)
                                                                    )
        # if doesn't seek care, probability of death and probability of recovery

        # schedule death events for very severe pneumonia
        for child in pn_current_very_sev_pneum_idx:
            death_event = DeathFromPneumoniaDisease(self.module, person_id=child)
            self.sim.schedule_event(death_event, df.at[child, 'ri_scheduled_date_death'])


class DeathFromPneumoniaDisease(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        rng = m.rng

        logger.info('This is DeathFromPneumoniaDisease Event determining if person %d will die from their disease',
                    person_id)

        eff_prob_death_pneumonia = \
            pd.Series(m.r_death_pneumonia,
                      index=df.index[df.is_alive & (df.ri_pneumonia_severity == 'very severe pneumonia') &
                                     (df.age_years < 5)])
        eff_prob_death_pneumonia.loc[
            df.is_alive & (df.ri_pneumonia_severity == 'very severe pneumonia') &
            (df.age_years < 5)] *= m.rr_death_pneumonia_agelt2mo
        eff_prob_death_pneumonia.loc[
            df.is_alive & (df.ri_pneumonia_severity == 'very severe pneumonia') & (df.age_exact_years >= 1) &
            (df.age_exact_years < 2)] *= m.rr_death_pneumonia_age12to23mo
        eff_prob_death_pneumonia.loc[df.is_alive & (df.ri_pneumonia_severity == 'very severe pneumonia') &
                     (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_death_pneumonia_age24to59mo
        eff_prob_death_pneumonia.loc[df.is_alive & (df.ri_pneumonia_severity == 'very severe pneumonia') &
                     df.has_hiv == True & (df.age_years < 5)] *= \
            m.rr_death_pneumonia_HIV
        eff_prob_death_pneumonia.loc[df.is_alive & (df.ri_pneumonia_severity == 'very severe pneumonia') &
                     df.malnutrition == True & (df.age_years < 5)] *= \
            m.rr_death_pneumonia_SAM

        pn1_current_very_severe_pneumonia_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_severity == 'very severe pneumonia') & (df.age_years < 5)]

        random_draw = \
            pd.Series(rng.random_sample(size=len(pn1_current_very_severe_pneumonia_idx)),
                      index=df.index[(df.age_years < 5) & df.is_alive & (df.ri_pneumonia_severity == 'very severe pneumonia')])
        very_sev_pneum_death = eff_prob_death_pneumonia > random_draw
        very_sev_pneum_death_idx = eff_prob_death_pneumonia.index[very_sev_pneum_death]

        for individual_id in very_sev_pneum_death_idx:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, 'NewPneumonia'),
                                    self.sim.date)
        else:
            df.loc[pn1_current_very_severe_pneumonia_idx, 'ri_pneumonia_severity'] = 'none'
