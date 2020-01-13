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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Pneumonia(Module):
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
         categories=['pneumonia', 'severe pneumonia', 'very severe pneumonia']
         ),
        'ri_pneumonia_pathogen': Property
        (Types.CATEGORICAL, 'attributable pathogen for pneumonia',
         categories=['RSV', 'rhinovirus', 'hMPV', 'parainfluenza', 'streptococcus',
                     'hib', 'TB', 'staph', 'influenza', 'P. jirovecii', 'other pathogens', 'other cause']
         ),
        'ri_pneumonia_pathogen_type': Property
        (Types.CATEGORICAL, 'attributable pathogen for pneumonia',
         categories=['bacterial', 'viral']
         ),
        'date_of_acquiring_pneumonia': Property
        (Types.DATE, 'date of acquiring pneumonia infection'
         ),
        'date_of_progression_severe_pneum': Property
        (Types.DATE, 'date of progression of disease to severe pneumonia'
         ),
        'date_of_progression_very_sev_pneum': Property
        (Types.DATE, 'date of progression of disease to severe pneumonia'
         ),
        'ri_pneumonia_death_date': Property
        (Types.BOOL, 'death from pneumonia disease'
         ),
        'ri_pneum_recovered_date':  Property
        (Types.DATE, 'date of recovery from pneumonia disease'
         ),
        'has_hiv': Property(Types.BOOL, 'temporary property - has hiv'),
        'malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
        'pneumococcal_vaccination': Property(Types.BOOL, 'temporary property - streptococcus pneumoniae vaccine'),
        'hib_vaccination': Property(Types.BOOL, 'temporary property - H. influenzae type b vaccine'),
        'influenza_vaccination': Property(Types.BOOL, 'temporary property - flu vaccine'),
        # symptoms of diarrhoea for care seeking
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

        p['base_incidence_pneumonia_by_agecat'] = [
            dfd.loc['base_incidence_pneumonia_by_agecat', 'value1'],
            dfd.loc['base_incidence_pneumonia_by_agecat', 'value2'],
            dfd.loc['base_incidence_pneumonia_by_agecat', 'value3']
            ]
        p['r_progress_to_severe_penum'] = [
            dfd.loc['r_progress_to_severe_penum', 'value1'],
            dfd.loc['r_progress_to_severe_penum', 'value2'],
            dfd.loc['r_progress_to_severe_penum', 'value3']
        ]
        p['r_progress_to_very_sev_penum'] = [
            dfd.loc['r_progress_to_very_sev_penum', 'value1'],
            dfd.loc['r_progress_to_very_sev_penum', 'value2'],
            dfd.loc['r_progress_to_very_sev_penum', 'value3']
        ]
        p['pn_attributable_fraction_RSV'] = [
            dfd.loc['pn_attributable_fraction_RSV', 'value1'],
            dfd.loc['pn_attributable_fraction_RSV', 'value2'],
            dfd.loc['pn_attributable_fraction_RSV', 'value3']
        ]
        p['pn_attributable_fraction_rhinovirus'] = [
            dfd.loc['pn_attributable_fraction_rhinovirus', 'value1'],
            dfd.loc['pn_attributable_fraction_rhinovirus', 'value2'],
            dfd.loc['pn_attributable_fraction_rhinovirus', 'value3']
            ]
        p['pn_attributable_fraction_hmpv'] = [
            dfd.loc['pn_attributable_fraction_hmpv', 'value1'],
            dfd.loc['pn_attributable_fraction_hmpv', 'value2'],
            dfd.loc['pn_attributable_fraction_hmpv', 'value3']
        ]
        p['pn_attributable_fraction_streptococcus'] = [
            dfd.loc['pn_attributable_fraction_streptococcus', 'value1'],
            dfd.loc['pn_attributable_fraction_streptococcus', 'value2'],
            dfd.loc['pn_attributable_fraction_streptococcus', 'value3']
        ]
        p['pn_attributable_fraction_parainfluenza'] = [
            dfd.loc['pn_attributable_fraction_parainfluenza', 'value1'],
            dfd.loc['pn_attributable_fraction_parainfluenza', 'value2'],
            dfd.loc['pn_attributable_fraction_parainfluenza', 'value3']
        ]
        p['pn_attributable_fraction_hib'] = [
            dfd.loc['pn_attributable_fraction_hib', 'value1'],
            dfd.loc['pn_attributable_fraction_hib', 'value2'],
            dfd.loc['pn_attributable_fraction_hib', 'value3']
        ]
        p['pn_attributable_fraction_TB'] = [
            dfd.loc['pn_attributable_fraction_TB', 'value1'],
            dfd.loc['pn_attributable_fraction_TB', 'value2'],
            dfd.loc['pn_attributable_fraction_TB', 'value3']
        ]
        p['pn_attributable_fraction_staph'] = [
            dfd.loc['pn_attributable_fraction_staph', 'value1'],
            dfd.loc['pn_attributable_fraction_staph', 'value2'],
            dfd.loc['pn_attributable_fraction_staph', 'value3']
        ]
        p['pn_attributable_fraction_influenza'] = [
            dfd.loc['pn_attributable_fraction_influenza', 'value1'],
            dfd.loc['pn_attributable_fraction_influenza', 'value2'],
            dfd.loc['pn_attributable_fraction_influenza', 'value3']
        ]
        p['base_incidence_pneumonia_by_hMPV'] = [
            dfd.loc['base_incidence_pneumonia_by_hmpv', 'value1'],
            dfd.loc['base_incidence_pneumonia_by_hmpv', 'value2']
            ]
        p['pn_attributable_fraction_jirovecii'] = [
            dfd.loc['pn_attributable_fraction_jirovecii', 'value1'],
            dfd.loc['pn_attributable_fraction_jirovecii', 'value2'],
            dfd.loc['pn_attributable_fraction_jirovecii', 'value3']
        ]
        p['pn_attributable_fraction_other_pathogens'] = [
            dfd.loc['pn_attributable_fraction_other_pathogens', 'value1'],
            dfd.loc['pn_attributable_fraction_other_pathogens', 'value2'],
            dfd.loc['pn_attributable_fraction_other_pathogens', 'value3']
        ]
        p['pn_attributable_fraction_other_cause'] = [
            dfd.loc['pn_attributable_fraction_other_cause', 'value1'],
            dfd.loc['pn_attributable_fraction_other_cause', 'value2'],
            dfd.loc['pn_attributable_fraction_other_cause', 'value3']
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
        p['rr_death_pneumonia_agelt2mo'] = 1.4
        p['rr_death_pneumonia_age12to23mo'] = 0.8
        p['rr_death_pneumonia_age24to59mo'] = 0.3
        p['rr_death_pneumonia_HIV'] = 1.4
        p['rr_death_pneumonia_SAM'] = 1.4
        p['IMCI_effectiveness_2010'] = 0.5
        p['dhs_care_seeking_2010'] = 0.6
        p['case_fatality_rate'] = 0.15

        '''# symptoms prevalence
        p['symptoms_nonsev_pneum_lt2mo'] = pd.DataFrame(
            data={
                'symptoms': ['fast_breathing', 'chest_indrawing', 'grunting', 'nasal flaring', 'head nod', 'cyanosis',
                             'sleepy', 'not breastfeeding', 'not drinking', 'wheeze', 'stridor', 'convulsions'],
                'prevalence': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            })
        p['symptoms_severe_pneum_lt2mo'] = pd.DataFrame(
            data={
                'symptoms': ['fast_breathing', 'chest_indrawing', 'grunting', 'nasal flaring', 'head nod', 'cyanosis',
                             'sleepy', 'not breastfeeding', 'not drinking', 'wheeze', 'stridor', 'convulsions'],
                'prevalence': [0.9, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            })
        p['symptoms_very_sev_pneum_lt2mo'] = pd.DataFrame(
            data={
                'symptoms': ['fast_breathing', 'chest_indrawing', 'grunting', 'nasal flaring', 'head nod', 'cyanosis',
                             'sleepy', 'not breastfeeding', 'not drinking', 'wheeze', 'stridor', 'convulsions'],
                'prevalence': [0.9, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
            })
            '''

        # DALY weights
        if 'HealthBurden' in self.sim.modules.keys():
            p['daly_pneumonia'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=47)
            p['daly_severe_pneumonia'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=47)
            p['daly_very_severe_pneumonia'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=46)

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
        # df['ri_pneumonia_severity'] = 'none'
        df['malnutrition'] = False
        df['has_hiv'] = False
        df['exclusive_breastfeeding'] = False
        df['continued_breastfeeding'] = False
        df['date_of_acquiring_pneumonia'] = pd.NaT
        df['date_of_progression_severe_pneum'] = pd.NaT
        df['date_of_progression_very_sev_pneum'] = pd.NaT
        # -------------------- ASSIGN PNEUMONIA STATUS AT BASELINE (PREVALENCE) -----------------------

        df_under5 = df.age_years < 5 & df.is_alive
        under5_idx = df.index[df_under5]

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
        sim.schedule_event(event_pneumonia, sim.date + DateOffset(days=0))

        # Register this disease module with the health system
        # self.sim.modules['HealthSystem'].register_disease_module(self)

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
        df = self.sim.population.props
        p = self.parameters

        health_values = df.loc[df.is_alive, 'ri_specific_symptoms'].map({
            'none': 0,
            'pneumonia':  p['daly_severe_pneumonia'],
            'severe pneumonia': p['daly_severe_pneumonia'],
            'very severe pneumonia': p['daly_very_severe_pneumonia']
        })
        health_values.name = 'Pneumonia Symptoms'  # label the cause of this disability

        return health_values.loc[df.is_alive]  # returns the series


class PneumoniaEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=3))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = self.module.rng
        p = self.module.parameters
        now = self.sim.date

        # DEFAULTS
        df['ri_pneumonia_status'] = False
        df['malnutrition'] = False
        df['has_hiv'] = False
        df['exclusive_breastfeeding'] = False
        df['continued_breastfeeding'] = False
        df['date_of_acquiring_pneumonia'] = pd.NaT

        # # # # # # # # # # # # # # PNEUMONIA INCIDENCE BY ATTRIBUTABLE PATHOGEN # # # # # # # # # # # # # #

        no_pneumonia0 = df.is_alive & (df.ri_pneumonia_status == False) & (df.age_exact_years < 1)
        no_pneumonia1 = df.is_alive & (df.ri_pneumonia_status == False) & \
                        (df.age_exact_years >= 1) & (df.age_exact_years < 2)
        no_pneumonia2 = df.is_alive & (df.ri_pneumonia_status == False) & \
                        (df.age_exact_years >= 2) & (df.age_exact_years < 5)
        no_pneumonia_under5 = df.is_alive & (df.ri_pneumonia_status == False) & (df.age_years < 5)
        current_no_pneumonia = df.index[no_pneumonia_under5]

        # Incidence of pneumonia (all severity) by age groups
        pneumonia_incidence_0_11mo = pd.Series(m.base_incidence_pneumonia_by_agecat[0], index=df.index[no_pneumonia0])
        pneumonia_incidence_12_23mo = pd.Series(m.base_incidence_pneumonia_by_agecat[1], index=df.index[no_pneumonia1])
        pneumonia_incidence_24_59mo = pd.Series(m.base_incidence_pneumonia_by_agecat[1], index=df.index[no_pneumonia2])

        # concatenating plus sorting
        eff_prob_pneum_all_ages = pd.concat([pneumonia_incidence_0_11mo, pneumonia_incidence_12_23mo,
                                             pneumonia_incidence_24_59mo], axis=0).sort_index()

        # adding the effects of risk factors
        eff_prob_pneum_all_ages.loc[no_pneumonia_under5 & (df.li_no_access_handwashing == False)] \
            *= m.rr_ri_pneumonia_HHhandwashing
        eff_prob_pneum_all_ages.loc[no_pneumonia_under5 & (df.has_hiv == True)] \
            *= m.rr_ri_pneumonia_HIV
        eff_prob_pneum_all_ages.loc[no_pneumonia_under5 & (df.malnutrition == True)] \
            *= m.rr_ri_pneumonia_SAM
        eff_prob_pneum_all_ages.loc[
            no_pneumonia_under5 & (df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5)] \
            *= m.rr_ri_pneumonia_excl_breast
        eff_prob_pneum_all_ages.loc[
            no_pneumonia_under5 & (df.continued_breastfeeding == True) & (df.age_exact_years > 0.5) &
            (df.age_exact_years < 2)] *= m.rr_ri_pneumonia_cont_breast
        eff_prob_pneum_all_ages.loc[
            no_pneumonia_under5 & (df.li_wood_burn_stove == False)] *= m.rr_ri_pneumonia_indoor_air_pollution
        eff_prob_pneum_all_ages.loc[
            no_pneumonia_under5 & (df.pneumococcal_vaccination == True)] *= m.rr_ri_pneumonia_pneumococcal_vaccine
        eff_prob_pneum_all_ages.loc[
            no_pneumonia_under5 & (df.hib_vaccination == True)] *= m.rr_ri_pneumonia_hib_vaccine
        eff_prob_pneum_all_ages.loc[
            no_pneumonia_under5 & (df.influenza_vaccination == True)] *= m.rr_ri_pneumonia_influenza_vaccine

        random_draw = pd.Series(rng.random_sample(size=len(current_no_pneumonia)), index=current_no_pneumonia)
        incident_pneum = eff_prob_pneum_all_ages > random_draw
        incident_pneum_idx = eff_prob_pneum_all_ages.index[incident_pneum]
        df.loc[incident_pneum_idx, 'ri_pneumonia_status'] = True

        # assign status for any individuals that are infected
        df.loc[df.ri_pneumonia_status & (df.age_exact_years < 1/6),
               'ri_pneumonia_severity'] = 'severe pneumonia'
        df.loc[df.ri_pneumonia_status & (df.age_exact_years >= 1/6) & (
            df.age_years < 5), 'ri_pneumonia_severity'] = 'pneumonia'

        # NOTE: NON-SEVERE PNEUMONIA ONLY IN 2-59 MONTHS
        first_incidence_pneum_idx = \
            df.index[df.is_alive & (df.age_exact_years < 1/6) & (df.ri_pneumonia_severity == 'severe pneumonia') |
                     (df.is_alive & (df.age_exact_years >= 1/6) & (df.ri_pneumonia_severity == 'pneumonia'))]

        # Give a date of acquiring pneumonia
        random_draw_days = np.random.randint(0, 90, size=len(first_incidence_pneum_idx))
        to_date = pd.to_timedelta(random_draw_days, unit='d')
        df.loc[first_incidence_pneum_idx, 'date_of_acquiring_pneumonia'] = self.sim.date + to_date

        # # # # # # # # # # # # # # # # # # SYMPTOMS FROM NON-SEVERE PNEUMONIA # # # # # # # # # # # # # # # # # #
        # fast breathing
        df.loc[first_incidence_pneum_idx, 'pn_fast_breathing'] = True

        # ---------------------------------------------------------------------------------------------------
        # / # / # / # / # / # / # / # / # / PROGRESS TO SEVERE PNEUMONIA # / # / # / # / # / # / # / # / # /
        # ---------------------------------------------------------------------------------------------------
        # Progression in pneumonia severity by age groups
        severe_pneum_prog_2_11mo =\
            pd.Series(m.r_progress_to_severe_penum[0],
                      index=df.index[df.is_alive & (df.age_exact_years >= 1/6) & (df.age_exact_years < 1) &
                                     df.ri_pneumonia_status & (df.ri_pneumonia_severity == 'pneumonia')])
        severe_pneum_prog_12_23mo = \
            pd.Series(m.r_progress_to_severe_penum[1],
                      index=df.index[df.is_alive & (df.age_exact_years >= 1) & (df.age_exact_years < 2) &
                                     df.ri_pneumonia_status & (df.ri_pneumonia_severity == 'pneumonia')])
        severe_pneum_prog_24_59mo = \
            pd.Series(m.r_progress_to_severe_penum[2],
                      index=df.index[df.is_alive & (df.age_exact_years >= 2) & (df.age_exact_years < 5) &
                                     df.ri_pneumonia_status & (df.ri_pneumonia_severity == 'pneumonia')])
        # concatenating plus sorting
        eff_prob_prog_severe_pneum = pd.concat([severe_pneum_prog_2_11mo, severe_pneum_prog_12_23mo,
                                                severe_pneum_prog_24_59mo], axis=0).sort_index()

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
            df.index[df.is_alive & (df.age_exact_years >= 1/6) & (df.age_exact_years < 5) &
                     df.ri_pneumonia_status & (df.ri_pneumonia_severity == 'pneumonia')]

        random_draw = pd.Series(rng.random_sample(size=len(pn_current_pneumonia_idx)),
                                index=pn_current_pneumonia_idx)
        progress_severe_pneum = eff_prob_prog_severe_pneum > random_draw
        progress_severe_pneum_idx = eff_prob_prog_severe_pneum.index[progress_severe_pneum]
        df.loc[progress_severe_pneum_idx, 'ri_pneumonia_severity'] = 'severe pneumonia'
        self_recovery_nonsev_pneum = eff_prob_prog_severe_pneum <= random_draw
        self_recovery_nonsev_pneum_idx = eff_prob_prog_severe_pneum.index[self_recovery_nonsev_pneum]

        # date of progression to severe pneumonia for 2-59 months
        df.loc[progress_severe_pneum_idx, 'date_of_progression_severe_pneum'] = \
            df['date_of_acquiring_pneumonia'] + pd.DateOffset(days=int(rng.random_integers(0, 7)))

        # keep the same date acquiring/progression for under 2 months
        under2mo_severe_pneum_idx = df.index[df.is_alive & (df.age_exact_years < 1/6) & df.ri_pneumonia_status &
                                             (df.ri_pneumonia_severity == 'severe pneumonia')]
        df.loc[under2mo_severe_pneum_idx, 'date_of_progression_severe_pneum'] = df['date_of_acquiring_pneumonia']

        # schedule recovery from non-severe pneumonia
        for person_id in self_recovery_nonsev_pneum_idx:
            self.sim.schedule_event(SelfRecoverEvent(self.module, person_id=person_id),
                                    (df.at[person_id, 'date_of_acquiring_pneumonia'] + DateOffset(
                                        days=int(rng.random_integers(3, 7)))))

            # random_day = self.sim.date + DateOffset(days=int(rng.random_integers(1, 28)))

        # # # ASSIGN THE ATTRIBUTABLE PATHOGEN # # #
        severe_pneum_0_11mo = df.index[df.is_alive & df.ri_pneumonia_status &
                                       (df.ri_pneumonia_severity == 'severe pneumonia') & (df.age_exact_years < 1)]
        severe_pneum_12_23mo = \
            df.index[df.is_alive & df.ri_pneumonia_status & (df.ri_pneumonia_severity == 'severe pneumonia') &
                     (df.age_exact_years >= 1) & (df.age_exact_years < 2)]
        severe_pneum_24_59mo = \
            df.index[df.is_alive & df.ri_pneumonia_status & (df.ri_pneumonia_severity == 'severe pneumonia') &
                     (df.age_exact_years >= 2) & (df.age_exact_years < 5)]

        severe_pneum_RSV0 = pd.Series(m.pn_attributable_fraction_RSV[0], index=severe_pneum_0_11mo)
        severe_pneum_RSV1 = pd.Series(m.pn_attributable_fraction_RSV[1], index=severe_pneum_12_23mo)
        severe_pneum_RSV2 = pd.Series(m.pn_attributable_fraction_RSV[2], index=severe_pneum_24_59mo)

        severe_pneum_rhino0 = pd.Series(m.pn_attributable_fraction_rhinovirus[0], index=severe_pneum_0_11mo)
        severe_pneum_rhino1 = pd.Series(m.pn_attributable_fraction_rhinovirus[1], index=severe_pneum_12_23mo)
        severe_pneum_rhino2 = pd.Series(m.pn_attributable_fraction_rhinovirus[2], index=severe_pneum_24_59mo)

        severe_pneum_hmpv0 = pd.Series(m.pn_attributable_fraction_hmpv[0], index=severe_pneum_0_11mo)
        severe_pneum_hmpv1 = pd.Series(m.pn_attributable_fraction_hmpv[1], index=severe_pneum_12_23mo)
        severe_pneum_hmpv2 = pd.Series(m.pn_attributable_fraction_hmpv[2], index=severe_pneum_24_59mo)

        severe_pneum_para0 = pd.Series(m.pn_attributable_fraction_parainfluenza[0], index=severe_pneum_0_11mo)
        severe_pneum_para1 = pd.Series(m.pn_attributable_fraction_parainfluenza[1], index=severe_pneum_12_23mo)
        severe_pneum_para2 = pd.Series(m.pn_attributable_fraction_parainfluenza[2], index=severe_pneum_24_59mo)

        severe_pneum_strep0 = pd.Series(m.pn_attributable_fraction_streptococcus[0], index=severe_pneum_0_11mo)
        severe_pneum_strep1 = pd.Series(m.pn_attributable_fraction_streptococcus[1], index=severe_pneum_12_23mo)
        severe_pneum_strep2 = pd.Series(m.pn_attributable_fraction_streptococcus[2], index=severe_pneum_24_59mo)

        severe_pneum_hib0 = pd.Series(m.pn_attributable_fraction_hib[0], index=severe_pneum_0_11mo)
        severe_pneum_hib1 = pd.Series(m.pn_attributable_fraction_hib[1], index=severe_pneum_12_23mo)
        severe_pneum_hib2 = pd.Series(m.pn_attributable_fraction_hib[2], index=severe_pneum_24_59mo)

        severe_pneum_TB0 = pd.Series(m.pn_attributable_fraction_TB[0], index=severe_pneum_0_11mo)
        severe_pneum_TB1 = pd.Series(m.pn_attributable_fraction_TB[1], index=severe_pneum_12_23mo)
        severe_pneum_TB2 = pd.Series(m.pn_attributable_fraction_TB[2], index=severe_pneum_24_59mo)

        severe_pneum_staph0 = pd.Series(m.pn_attributable_fraction_staph[0], index=severe_pneum_0_11mo)
        severe_pneum_staph1 = pd.Series(m.pn_attributable_fraction_staph[1], index=severe_pneum_12_23mo)
        severe_pneum_staph2 = pd.Series(m.pn_attributable_fraction_staph[2], index=severe_pneum_24_59mo)

        severe_pneum_influenza0 = pd.Series(m.pn_attributable_fraction_influenza[0], index=severe_pneum_0_11mo)
        severe_pneum_influenza1 = pd.Series(m.pn_attributable_fraction_influenza[1], index=severe_pneum_12_23mo)
        severe_pneum_influenza2 = pd.Series(m.pn_attributable_fraction_influenza[2], index=severe_pneum_24_59mo)

        severe_pneum_jirovecii0 = pd.Series(m.pn_attributable_fraction_jirovecii[0], index=severe_pneum_0_11mo)
        severe_pneum_jirovecii1 = pd.Series(m.pn_attributable_fraction_jirovecii[1], index=severe_pneum_12_23mo)
        severe_pneum_jirovecii2 = pd.Series(m.pn_attributable_fraction_jirovecii[2], index=severe_pneum_24_59mo)

        severe_pneum_other_patho0 = pd.Series(m.pn_attributable_fraction_other_pathogens[0], index=severe_pneum_0_11mo)
        severe_pneum_other_patho1 = pd.Series(m.pn_attributable_fraction_other_pathogens[1], index=severe_pneum_12_23mo)
        severe_pneum_other_patho2 = pd.Series(m.pn_attributable_fraction_other_pathogens[2], index=severe_pneum_24_59mo)

        severe_pneum_other_cause0 = pd.Series(m.pn_attributable_fraction_other_cause[0], index=severe_pneum_0_11mo)
        severe_pneum_other_cause1 = pd.Series(m.pn_attributable_fraction_other_cause[1], index=severe_pneum_12_23mo)
        severe_pneum_other_cause2 = pd.Series(m.pn_attributable_fraction_other_cause[2], index=severe_pneum_24_59mo)

        eff_prob_severe_pneum_RSV = \
            pd.concat([severe_pneum_RSV0, severe_pneum_RSV1, severe_pneum_RSV2], axis=0).sort_index()
        eff_prob_severe_pneum_rhinovirus = \
            pd.concat([severe_pneum_rhino0, severe_pneum_rhino1, severe_pneum_rhino2], axis=0).sort_index()
        eff_prob_severe_pneum_hMPV =\
            pd.concat([severe_pneum_hmpv0, severe_pneum_hmpv1, severe_pneum_hmpv2], axis=0).sort_index()
        eff_prob_severe_pneum_parainfluenza \
            = pd.concat([severe_pneum_para0, severe_pneum_para1, severe_pneum_para2], axis=0).sort_index()
        eff_prob_severe_pneum_strep = \
            pd.concat([severe_pneum_strep0, severe_pneum_strep1, severe_pneum_strep2], axis=0).sort_index()
        eff_prob_severe_pneum_hib = \
            pd.concat([severe_pneum_hib0, severe_pneum_hib1, severe_pneum_hib2], axis=0).sort_index()
        eff_prob_severe_pneum_TB = \
            pd.concat([severe_pneum_TB0, severe_pneum_TB1, severe_pneum_TB2], axis=0).sort_index()
        eff_prob_severe_pneum_staph = \
            pd.concat([severe_pneum_staph0, severe_pneum_staph1, severe_pneum_staph2], axis=0).sort_index()
        eff_prob_severe_pneum_influenza = \
            pd.concat([severe_pneum_influenza0, severe_pneum_influenza1, severe_pneum_influenza2], axis=0).sort_index()
        eff_prob_severe_pneum_jirovecii =\
            pd.concat([severe_pneum_jirovecii0, severe_pneum_jirovecii1, severe_pneum_jirovecii2], axis=0).sort_index()
        eff_prob_severe_pneum_other_patho =\
            pd.concat([severe_pneum_other_patho0, severe_pneum_other_patho1, severe_pneum_other_patho2], axis=0).sort_index()
        eff_prob_severe_pneum_other_cause = \
            pd.concat([severe_pneum_other_cause0, severe_pneum_other_cause1, severe_pneum_other_cause2], axis=0).sort_index()

        eff_prob_sev_pneum_all_pathogens = \
            pd.concat([eff_prob_severe_pneum_RSV, eff_prob_severe_pneum_rhinovirus, eff_prob_severe_pneum_hMPV,
                       eff_prob_severe_pneum_parainfluenza, eff_prob_severe_pneum_strep, eff_prob_severe_pneum_hib,
                       eff_prob_severe_pneum_TB, eff_prob_severe_pneum_staph, eff_prob_severe_pneum_influenza,
                       eff_prob_severe_pneum_jirovecii, eff_prob_severe_pneum_other_patho,
                       eff_prob_severe_pneum_other_cause], axis=1)

        severe_pneum_0_59mo = df.index[df.is_alive & df.ri_pneumonia_status &
                                       (df.ri_pneumonia_severity == 'severe pneumonia') & (df.age_exact_years < 5)]

        # cumulative sum to determine which pathogen is the cause of severe pneumonia
        random_draw_all = \
            pd.Series(rng.random_sample(size=len(severe_pneum_0_59mo)), index=severe_pneum_0_59mo)
        eff_prob_none = 1 - eff_prob_sev_pneum_all_pathogens.sum(axis=1)
        dfx = pd.concat([eff_prob_none, eff_prob_sev_pneum_all_pathogens], axis=1)
        dfx = dfx.cumsum(axis=1)
        dfx.columns = ['prob_none', 'RSV', 'rhinovirus', 'hMPV', 'parainfluenza', 'streptococcus', 'hib', 'TB',
                       'staph', 'influenza', 'P. jirovecii', 'other pathogens', 'other cause']
        dfx['random_draw_all'] = random_draw_all

        for i, column in enumerate(dfx.columns):
            # go through each pathogen and assign the pathogen and status
            if column in ('prob_none', 'random_draw_all'):
                # skip probability of none and random draw columns
                continue

            idx_attributable_patho = dfx.index[
                ((dfx.iloc[:, i - 1] < dfx.random_draw_all)
                 & (dfx.loc[:, column] >= dfx.random_draw_all))]
            df.loc[idx_attributable_patho, 'ri_pneumonia_pathogen'] = column

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

        '''severe_pneumonia_symptoms = df.index[df.is_alive & (df.pn_cough == True) | (df.pn_difficult_breathing == True) |
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
                '''

        # --------------------------------------------------------------------------------------------------------
        # / # / # / # / # / # / # / # / # / PROGRESS TO VERY SEVERE PNEUMONIA # / # / # / # / # / # / # / # / # /
        # --------------------------------------------------------------------------------------------------------
        # Progression in pneumonia severity by age groups
        current_sev_pneum = df.is_alive & (df.age_exact_years < 5) & df.ri_pneumonia_status & \
                            (df.ri_pneumonia_severity == 'severe pneumonia')

        very_sev_pneum_prog_0_11mo = \
            pd.Series(m.r_progress_to_very_sev_penum[0],
                      index=df.index[current_sev_pneum & (df.age_years < 1)])
        very_sev_pneum_prog_12_23mo = \
            pd.Series(m.r_progress_to_very_sev_penum[1],
                      index=df.index[current_sev_pneum & (df.age_exact_years >= 1) & (df.age_years < 2)])
        very_sev_pneum_prog_24_59mo = \
            pd.Series(m.r_progress_to_very_sev_penum[2],
                      index=df.index[current_sev_pneum & (df.age_exact_years >= 2) & (df.age_years < 5)])
        # concatenating plus sorting
        eff_prob_prog_very_sev_pneum = pd.concat([very_sev_pneum_prog_0_11mo, very_sev_pneum_prog_12_23mo,
                                                  very_sev_pneum_prog_24_59mo], axis=0).sort_index()

        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.has_hiv == True) & (df.age_years < 5)] *=\
            m.rr_progress_very_sev_pneum_HIV
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & df.malnutrition == True & (df.age_years < 5)] *=\
            m.rr_progress_very_sev_pneum_SAM
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'RSV') &
                                         (df.age_years < 5)] *= 0.7159
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'rhinovirus') &
                                         (df.age_years < 5)] *= 0.9506
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'hMPV') &
                                         (df.age_years < 5)] *= 0.9512
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'parainfluenza') &
                                         (df.age_years < 5)] *= 0.5556
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'streptococcus') &
                                         (df.age_years < 5)] *= 2.1087
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'hib') &
                                         (df.age_years < 5)] *= 1.6122
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'TB') &
                                         (df.age_years < 5)] *= 1.1667
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'staphylococcus') &
                                         (df.age_years < 5)] *= 5.2727
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'influenza') &
                                         (df.age_years < 5)] *= 1.4
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'P. jirovecii') &
                                         (df.age_years < 5)] *= 1.9167

        pn_current_severe_pneumonia_idx = df.index[current_sev_pneum]

        random_draw = pd.Series(rng.random_sample(size=len(pn_current_severe_pneumonia_idx)),
                                index=pn_current_severe_pneumonia_idx)
        progress_very_sev_pneum = eff_prob_prog_very_sev_pneum > random_draw
        progress_very_sev_pneum_idx = eff_prob_prog_very_sev_pneum.index[progress_very_sev_pneum]
        df.loc[progress_very_sev_pneum_idx, 'ri_pneumonia_severity'] = 'very severe pneumonia'
        recover_from_severe_pneum = eff_prob_prog_very_sev_pneum <= random_draw
        recover_from_severe_pneum_idx = eff_prob_prog_very_sev_pneum.index[recover_from_severe_pneum]

        # date of progression to very severe pneumonia for 0-59 months
        df.loc[progress_severe_pneum_idx, 'date_of_progression_very_sev_pneum'] = \
            df['date_of_progression_severe_pneum'] + pd.DateOffset(days=int(rng.random_integers(0, 3)))

        # Schedule recovery for severe pneumonia
        for person_id in recover_from_severe_pneum_idx:
            self.sim.schedule_event(SelfRecoverEvent(self.module, person_id=person_id),
                                    df.at[person_id, 'date_of_progression_severe_pneum'] +
                                    DateOffset(days=int(rng.random_integers(0, 4))))

        # log the information on attributable pathogens
        pathogen_count = df[df.is_alive & df.age_years.between(0, 5)].groupby('ri_pneumonia_pathogen').size()
        under5 = df[df.is_alive & df.age_years.between(0, 5)]
        logger.info('%s|pneumonia_pathogens|%s', self.sim.date,
                    {'total': sum(pathogen_count),
                     'RSV': pathogen_count['RSV'],
                     'rhinovirus': pathogen_count['rhinovirus'],
                     'hMPV': pathogen_count['hMPV'],
                     'parainfluenza': pathogen_count['parainfluenza'],
                     'strep': pathogen_count['streptococcus'],
                     'hib': pathogen_count['hib'],
                     'TB': pathogen_count['TB'],
                     'staph': pathogen_count['staph'],
                     'influenza': pathogen_count['influenza'],
                     'jirovecii': pathogen_count['P. jirovecii'],
                     })

        # incidence rate by pathogen per 100 child-years
        logger.info('%s|pneumo_incidence_by_patho|%s', self.sim.date,
                    {'total': (sum(pathogen_count) * 4 * 100) / len(under5),
                     'RSV': (pathogen_count['RSV'] * 4 * 100) / len(under5),
                     'rhinovirus': (pathogen_count['rhinovirus'] * 4 * 100) / len(under5),
                     'hMPV': (pathogen_count['hMPV'] * 4 * 100) / len(under5),
                     'parainfluenza': (pathogen_count['parainfluenza'] * 4 * 100) / len(under5),
                     'strep': (pathogen_count['streptococcus'] * 4 * 100) / len(under5),
                     'hib': (pathogen_count['hib'] * 4 * 100) / len(under5),
                     'TB': (pathogen_count['TB'] * 4 * 100) / len(under5),
                     'staph': (pathogen_count['staph'] * 4 * 100) / len(under5),
                     'influenza': (pathogen_count['influenza'] * 4 * 100) / len(under5),
                     'jirovecii': (pathogen_count['P. jirovecii'] * 4 * 100) / len(under5),
                     })

        # TODO: make a graph showing the proportions of pathogens causing severe vs very severe pneum
        # log the proportions of pathogens causing severe and very severe
        severity_pneum_count = df[df.is_alive & df.age_years.between(0, 5)].groupby('ri_pneumonia_severity').size()
        logger.info('%s|severity_pneumonia|%s', self.sim.date,
                    {'total': sum(severity_pneum_count),
                     'pneumonia': severity_pneum_count['pneumonia'],
                     'severe': severity_pneum_count['severe pneumonia'],
                     'very_severe': severity_pneum_count['very severe pneumonia']
                     })

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

        '''seeks_care = pd.Series(data=False, index=very_sev_pneum_symptoms)
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
        '''

        # # # # # # ASSIGN DEATH PROBABILITIES BASED ON AGE, SEVERITY AND CO-MORBIDITIES # # # # # #
        # schedule death events for very severe pneumonia
        current_very_sev_pneumonia_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_severity == 'very severe pneumonia') & (df.age_exact_years < 5)]

        # base group 2-11 months of age
        eff_prob_death_pneumonia = \
            pd.Series(m.r_death_pneumonia,
                      index=current_very_sev_pneumonia_idx)
        eff_prob_death_pneumonia.loc[
            df.is_alive & (df.age_exact_years < 1/6)] *= m.rr_death_pneumonia_agelt2mo
        eff_prob_death_pneumonia.loc[
            df.is_alive & (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= m.rr_death_pneumonia_age12to23mo
        eff_prob_death_pneumonia.loc[df.is_alive & (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_death_pneumonia_age24to59mo
        eff_prob_death_pneumonia.loc[df.is_alive & df.has_hiv == True & (df.age_exact_years < 5)] *= \
            m.rr_death_pneumonia_HIV
        eff_prob_death_pneumonia.loc[df.is_alive & df.malnutrition == True & (df.age_exact_years < 5)] *= \
            m.rr_death_pneumonia_SAM

        random_draw_death = \
            pd.Series(rng.random_sample(size=len(current_very_sev_pneumonia_idx)),
                      index=current_very_sev_pneumonia_idx)
        pneum_death = eff_prob_death_pneumonia > random_draw_death
        pneum_death_idx = eff_prob_death_pneumonia.index[pneum_death]
        recover_from_very_sev_pneum = eff_prob_death_pneumonia <= random_draw_death
        recover_from_very_sev_pneum_idx = eff_prob_death_pneumonia.index[recover_from_very_sev_pneum]

        # schedule recovery event from very severe pneumonia
        for person_id in recover_from_very_sev_pneum_idx:
            random_date = rng.randint(low=1, high=2)
            random_days = pd.to_timedelta(random_date, unit='d')
            self.sim.schedule_event(SelfRecoverEvent(self.module, person_id=person_id),
                                    (df.at[person_id, 'date_of_progression_very_sev_pneum'] + random_days))

        # schedule death event
        for person_id in pneum_death_idx:
            random_date = rng.randint(low=1, high=2)
            random_days = pd.to_timedelta(random_date, unit='d')
            self.sim.schedule_event(DeathFromPneumoniaDisease(self.module, person_id=person_id, cause='pneumonia'),
                                    (df.at[person_id, 'date_of_progression_very_sev_pneum'] + random_days))


class SelfRecoverEvent(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        # set everything back to default
        df.at[person_id, 'ri_pneumonia_status'] = False
        df.at[person_id, 'ri_pneumonia_severity'] = np.nan
        df.at[person_id, 'date_of_acquiring_pneumonia'] = pd.NaT
        df.at[person_id, 'date_of_progression_severe_pneum'] = pd.NaT
        df.at[person_id, 'date_of_progression_very_sev_pneum'] = pd.NaT
        df.at[person_id, 'ri_pneumonia_pathogen'] = np.nan


class DeathFromPneumoniaDisease(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id, cause):
        super().__init__(module, person_id=person_id)
        self.cause = cause

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        logger.info('This is DeathFromPneumoniaDisease Event determining if person %d on date %s will die '
                    'from their disease', person_id, self.sim.date)
        # TODO: those who are meant to die in the natural history (historical mortality rate) are sent here,
        #  but before entering this event, the mortality rate is multiplied by effect of treatment and interventions

        if df.at[person_id, 'is_alive']:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, person_id, cause='pneumonia'),
                                    self.sim.date)
            logger.info('%s|pneumo_death|%s', self.sim.date,
                        {
                            'age': df.at[person_id, 'age_years'],
                            'child': person_id,
                            'cause': self.cause
                        })

'''
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
            '''
# # # # # # # # # # # # # # # # # # SCHEDULE SELF-RECOVERY # # # # # # # # # # # # # # # # # #

'''viral_infection = df.index[df.is_alive & df.ri_pneumonia_status & (df.age_years < 5) &
                           (('ri_pneumonia_pathogen' == 'RSV') | ('ri_pneumonia_pathogen' == 'rhinovirus') |
                            ('ri_pneumonia_pathogen' == 'hMPV') | ('ri_pneumonia_pathogen' == 'influenza'))]
bacterial_infection = df.index[df.is_alive & df.ri_pneumonia_status & (df.age_years < 5) &
                               (('ri_pneumonia_pathogen' == 'streptococcus') | ('ri_pneumonia_pathogen' == 'hib') |
                                ('ri_pneumonia_pathogen' == 'TB') | ('ri_pneumonia_pathogen' == 'staph'))]

eff_prob_nonsev_pneum_recovery = pd.Series(0.7, index=first_incidence_pneum_idx)  # apply a rate of recovery
eff_prob_nonsev_pneum_recovery.loc[no_pneumonia_under5 & (df.li_no_access_handwashing == False)] \
    *= m.rr_ri_pneumonia_HHhandwashing
eff_prob_nonsev_pneum_recovery.loc[no_pneumonia_under5 & (df.has_hiv == True)] \
    *= m.rr_ri_pneumonia_HIV
eff_prob_nonsev_pneum_recovery.loc[no_pneumonia_under5 & (df.malnutrition == True)] \
    *= m.rr_ri_pneumonia_SAM
eff_prob_nonsev_pneum_recovery.loc[
    no_pneumonia_under5 & (df.exclusive_breastfeeding == True) & (df.age_exact_years <= 0.5)] \
    *= m.rr_ri_pneumonia_excl_breast

for child in viral_infection:
    self.sim.schedule_event(SelfRecoverEvent(self.module, child),
                            df.date_of_acquiring_pneumonia + DateOffset(days=3))
for child in bacterial_infection:
    self.sim.schedule_event(SelfRecoverEvent(self.module, child),
                            df.date_of_acquiring_pneumonia + DateOffset(days=7))
                            
                            
                            
                             random_draw_days1 = np.random.randint(0, 7, size=len(progress_severe_pneum_idx))
        td = pd.to_timedelta(random_draw_days1, unit='d')
        
            def assign_symptoms(self, population, eligible_idx, severity):
        """
        Assigns multiple symptoms to the initial population.
        :param eligible_idx: indices of infected individuals
        :param population:
        :param severity: level of severity, non-severe, severe or very severe pneumonia
        """
        assert severity in ['nonsev_pneumonia', 'severe_pneumonia', 'very_sev_pneumonia'], \
            "Incorrect severity level. Can't assign symptoms."

        if len(eligible_idx):
            df = population.props
            p = self.parameters
            symptoms_dict = p['symptoms_' + severity].set_index('symptoms').to_dict()[
                'prevalence']  # create a dictionary from a df
            symptoms_column = severity + '_specific_symptoms'

            for symptom in symptoms_dict.keys():
                p = symptoms_dict[symptom]  # get the prevalence of the symptom among the infected population
                # find who should get this symptom assigned - get p indices
                s_idx = self.rng.choice(eligible_idx, size=int(p * len(eligible_idx)), replace=False)
                df.loc[s_idx, symptoms_column] = df.loc[s_idx, symptoms_column].apply(
                    lambda x: add_elements(x, [symptom]))
        
'''
