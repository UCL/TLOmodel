"""
Childhood ALRI module
Documentation: 04 - Methods Repository/ResourceFile_Childhood_Pneumonia.xlsx
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import demography

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

class ALRI(Module):
    # Declare the pathogens that this module will simulate:
    pathogens = {
        'RSV',
        'rhinovirus',
        'hMPV',
        'parainfluenza',
        'streptococcus',
        'hib',
        'TB',
        'staphylococcus',
        'influenza',
        'jirovecii',
        'adenovirus',
        'coronavirus',
        'bocavirus',
        'other_pathogens'
    }

    complications = {'pneumothorax', 'pleural_effusion', 'empyema', 'lung_abscess',
                     'sepsis', 'meningitis', 'respiratory_failure'}

    # Declare the severity levels of the disease:
    pathogen_type = {
        'viral': 'RSV' 'rhinovirus' 'hMPV' 'parainfluenza' 'influenza' 'adenovirus' 'coronavirus' 'bocavirus',
        'bacterial': 'streptococcus' 'hib' 'TB' 'staphylococcus'
    }

    disease_type = {
        'bacterial_pneumonia', 'viral_pneumonia', 'bronchiolitis'
    }


    PARAMETERS = {
        'base_incidence_rate_ALRI_by_RSV': Parameter
        (Types.LIST, 'incidence of ALRI caused by Respiratory Syncytial Virus in age groups 0-11, 12-59 months'
         ),
        'base_inc_rate_ALRI_by_rhinovirus': Parameter
        (Types.LIST, 'incidence of ALRI caused by rhinovirus in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_hMPV': Parameter
        (Types.LIST, 'incidence of ALRI caused by hMPV in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_parainfluenza': Parameter
        (Types.LIST, 'incidence of ALRI caused by parainfluenza in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_streptococcus': Parameter
        (Types.LIST, 'incidence of ALRI caused by streptoccocus in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_hib': Parameter
        (Types.LIST, 'incidence of ALRI caused by hib in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_TB': Parameter
        (Types.LIST, 'incidence of ALRI caused by TB in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_staphylococcus': Parameter
        (Types.LIST, 'incidence of ALRI caused by Staphylococcus aureus in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_influenza': Parameter
        (Types.LIST, 'incidence of ALRI caused by influenza in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_jirovecii': Parameter
        (Types.LIST, 'incidence of ALRI caused by P. jirovecii in age groups 0-11, 12-59 months'
         ),
        'base_inc_rate_ALRI_by_adenovirus': Parameter
        (Types.LIST, 'incidence of ALRI caused by P. adenovirus in age groups 0-11, 12-59 months'
         ),
        'base_inc_rate_ALRI_by_coronavirus': Parameter
        (Types.LIST, 'incidence of ALRI caused by P. coronavirus in age groups 0-11, 12-59 months'
         ),
        'base_inc_rate_ALRI_by_bocavirus': Parameter
        (Types.LIST, 'incidence of ALRI caused by bocavirus in age groups 0-11, 12-59 months'
         ),
        'base_inc_rate_ALRI_by_other_pathogens': Parameter
        (Types.LIST, 'incidence of ALRI caused by other pathogens in age groups 0-11, 12-59 months'
         ),
        'rr_ALRI_HHhandwashing': Parameter
        (Types.REAL, 'relative rate of ALRI with household handwashing with soap'
         ),
        'rr_ALRI_HIV_untreated': Parameter
        (Types.REAL, 'relative rate of ALRI for HIV positive status'
         ),
        'rr_ALRI_underweight': Parameter
        (Types.REAL, 'relative rate of ALRI for underweight'
         ),
        'rr_ALRI_not_excl_breastfeeding': Parameter
        (Types.REAL, 'relative rate of ALRI for not exclusive breastfeeding upto 6 months'
         ),
        'rr_ALRI_indoor_air_pollution': Parameter
        (Types.REAL, 'relative rate of ALRI for indoor air pollution'
         ),
        'rr_ALRI_pneumococcal_vaccine': Parameter
        (Types.REAL, 'relative rate of ALRI for pneumonococcal vaccine'
         ),
        'rr_ALRI_hib_vaccine': Parameter
        (Types.REAL, 'relative rate of ALRI for hib vaccine'
         ),
        'rr_ALRI_influenza_vaccine': Parameter
        (Types.REAL, 'relative rate of ALRI for influenza vaccine'
         ),
        'r_progress_to_severe_ALRI': Parameter
        (Types.LIST,
         'probability of progressing from non-severe to severe ALRI by age category '
         'HIV negative, no SAM'
         ),
        'prob_respiratory_failure_by_viral_pneumonia': Parameter
        (Types.REAL, 'probability of respiratory failure caused by primary viral ALRI'
         ),
        'prob_respiratory_failure_by_bacterial_pneumonia': Parameter
        (Types.REAL, 'probability of respiratory failure caused by primary or secondary bacterial pneumonia'
         ),
        'prob_respiratory_failure_to_multiorgan_dysfunction': Parameter
        (Types.REAL, 'probability of respiratory failure causing multi-organ dysfunction'
         ),
        'prob_sepsis_by_viral_pneumonia': Parameter
        (Types.REAL, 'probability of sepsis caused by primary viral pneumonia'
         ),
        'prob_sepsis_by_bacterial_pneumonia': Parameter
        (Types.REAL, 'probability of sepsis caused by primary or secondary bacterial pneumonia'
         ),
        'prob_sepsis_to_septic_shock': Parameter
        (Types.REAL, 'probability of sepsis causing septic shock'
         ),
        'prob_septic_shock_to_multiorgan_dysfunction': Parameter
        (Types.REAL, 'probability of septic shock causing multi-organ dysfunction'
         ),
        'prob_meningitis_by_bacterial_ALRI': Parameter
        (Types.REAL, 'probability of meningitis caused by primary or secondary bacterial pneumonia'
         ),
        'prob_pleural_effusion_by_bacterial_pneumonia': Parameter
        (Types.REAL, 'probability of pleural effusion caused by primary or secondary bacterial pneumonia'
         ),
        'prob_pleural_effusion_to_empyema': Parameter
        (Types.REAL, 'probability of pleural effusion developing into empyema'
         ),
        'prob_empyema_to_sepsis': Parameter
        (Types.REAL, 'probability of empyema causing sepsis'
         ),
        'prob_lung_abscess_by_bacterial_pneumonia': Parameter
        (Types.REAL, 'probability of a lung abscess caused by primary or secondary bacterial pneumonia'
         ),
        'prob_pneumothorax_by_bacterial_pneumonia': Parameter
        (Types.REAL, 'probability of pneumothorax caused by primary or secondary bacterial pneumonia'
         ),
        'prob_pneumothorax_to_respiratory_failure': Parameter
        (Types.REAL, 'probability of pneumothorax causing respiratory failure'
         ),
        'prob_lung_abscess_to_sepsis': Parameter
        (Types.REAL, 'probability of lung abscess causing sepsis'
         ),
        'r_death_from_ALRI_due_to_meningitis': Parameter
        (Types.REAL, 'death rate from ALRI due to meningitis'
         ),
        'r_death_from_ALRI_due_to_sepsis': Parameter
        (Types.REAL, 'death rate from ALRI due to sepsis'
         ),
        'r_death_from_ALRI_due_to_respiratory_failure': Parameter
        (Types.REAL, 'death rate from ALRI due to respiratory failure'
         ),
        # 'rr_death_ALRI_agelt2mo': Parameter
        # (Types.REAL,
        #  'death rate of ALRI'
        #  ),
        'rr_death_ALRI_age12to23mo': Parameter
        (Types.REAL,
         'death rate of ALRI for children aged 12 to 23 months'
         ),
        'rr_death_ALRI_age24to59mo': Parameter
        (Types.REAL,
         'death rate of ALRI for children aged 24 to 59 months'
         ),
        'rr_death_ALRI_HIV': Parameter
        (Types.REAL,
         'death rate of ALRI for children with HIV not on ART'
         ),
        'rr_death_ALRI_SAM': Parameter
        (Types.REAL,
         'death rate of ALRI for children with severe acute malnutrition'
         ),
        'rr_death_ALRI_low_birth_weight': Parameter
        (Types.REAL,
         'death rate of ALRI for children with low birth weight (applicable to infants)'
         ),
    }

    PROPERTIES = {
        # ---- The pathogen which is the attributed cause of ALRI ----
        'ri_primary_ALRI_pathogen': Property(Types.CATEGORICAL,
                                             'Attributable pathogen for the current ALRI event',
                                             categories=list(pathogens) + ['none']),

        # ---- The bacterial pathogen which is the attributed cause of ALRI ----
        'ri_secondary_bacterial_pathogen': Property(Types.CATEGORICAL,
                                               'Secondary bacterial pathogen for the current ALRI event',
                                               categories=list(pathogen_type['bacterial']) + ['none']),

        # ---- The underlying ALRI condition ----
        'ri_ALRI_disease_type': Property(Types.CATEGORICAL, 'underlying ALRI condition',
                                         categories=['viral pneumonia', 'bacterial pneumonia', 'co-infection',
                                                     'bronchiolitis']),

        # ---- Complications associated with ALRI ----
        'ri_ALRI_complications': Property(Types.LIST,
                                                  'complications that arose from the current ALRI event',
                                                  categories=['pneumothorax', 'pleural_effusion', 'empyema',
                                                              'lung_abscess', 'sepsis', 'meningitis',
                                                              'respiratory_failure'] + ['none']
                                                  ),

        # ---- Symptoms associated with ALRI ----
        'ri_current_ALRI_symptoms': Property(Types.LIST,
                                             'symptoms of current ALRI event',
                                             categories=['fever', 'cough', 'difficult_breathing',
                                                         'fast_breathing', 'chest_indrawing', 'grunting',
                                                         'cyanosis', 'severe_respiratory_distress', 'hypoxia',
                                                         'danger_signs']
                                             ),

        # ---- Internal variables to schedule onset and deaths due to ALRI ----
        'ri_ALRI_event_date_of_onset': Property(Types.DATE, 'date of onset of current ALRI event'),
        'ri_ALRI_event_recovered_date': Property(Types.DATE, 'date of recovery from current ALRI event'),
        'ri_ALRI_event_death_date': Property(Types.DATE, 'date of death caused by current ALRI event'),

        # ---- Temporary Variables: To be replaced with the properties of other modules ----
        'tmp_malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'tmp_low_birth_weight': Property(Types.BOOL, 'temporary property - low birth weight'),
        'tmp_hv_inf': Property(Types.BOOL, 'temporary property - hiv infection'),
        'tmp_exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'tmp_continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
        'tmp_pneumococcal_vaccination': Property(Types.BOOL, 'temporary property - streptococcus ALRIe vaccine'),
        'tmp_hib_vaccination': Property(Types.BOOL, 'temporary property - H. influenzae type b vaccine'),
        'tmp_influenza_vaccination': Property(Types.BOOL, 'temporary property - flu vaccine'),

        # ---- Treatment properties ----
        # TODO; Ines -- you;ve introduced these but not initialised them and don;t use them. do you need them?
        'ri_ALRI_treatment': Property(Types.BOOL, 'currently on ALRI treatment'),
        'ri_ALRI_tx_start_date': Property(Types.DATE, 'start date of ALRI treatment for current event'),
    }

    # declare the symptoms that this module will cause:
    SYMPTOMS = {'fever', 'cough', 'difficult_breathing', 'fast_breathing', 'chest_indrawing', 'grunting',
                'cyanosis', 'severe_respiratory_distress', 'hypoxia', 'danger_signs', 'stridor'}

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # equations for the incidence of ALRI by pathogen:
        self.incidence_equations_by_pathogen = dict()

        # equations for the proportions of ALRI diseases:
        self.proportions_of_ALRI_disease_types_by_pathogen = dict()

        # equations for the probabilities of secondary bacterial co-infection:
        self.prob_secondary_bacterial_infection = dict()

        # equations for the development of ALRI-associated complications:
        self.risk_of_developing_ALRI_complications = dict()

        # Linear Model for predicting the risk of death:
        self.risk_of_death_severe_ALRI = dict()

        # dict to hold the probability of onset of different types of symptom given underlying complications:
        self.prob_symptoms_uncomplicated_ALRI = dict()
        self.prob_extra_symptoms_complications = dict()

        # dict to hold the DALY weights
        self.daly_wts = dict()

        # dict to hold counters for the number of ALRI events by pathogen and age-group
        # (0yrs, 1yrs, 2-4yrs)
        blank_counter = dict(zip(self.pathogens, [list() for _ in self.pathogens]))

        self.incident_case_tracker_blank = {
            '0y': copy.deepcopy(blank_counter),
            '1y': copy.deepcopy(blank_counter),
            '2-4y': copy.deepcopy(blank_counter),
            '5+y': copy.deepcopy(blank_counter)
        }
        self.incident_case_tracker = copy.deepcopy(self.incident_case_tracker_blank)

        zeros_counter = dict(zip(self.pathogens, [0]*len(self.pathogens)))
        self.incident_case_tracker_zeros = {
            '0y': copy.deepcopy(zeros_counter),
            '1y': copy.deepcopy(zeros_counter),
            '2-4y': copy.deepcopy(zeros_counter),
            '5+y': copy.deepcopy(zeros_counter)
        }

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module
        """
        p = self.parameters
        self.load_parameters_from_dataframe(
            pd.read_excel(
                Path(self.resourcefilepath) / 'ResourceFile_ALRI.xlsx', sheet_name='Parameter_values'))

        p['prob_secondary_bacterial_infection_from_bronchiolitis'] = 0.03
        p['prob_secondary_bacterial_infection_from_viral_pneumonia'] = 0.3

        # Check that every value has been read-in successfully
        for param_name, param_type in self.PARAMETERS.items():
            assert param_name in self.parameters, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert param_name is not None, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert isinstance(self.parameters[param_name],
                          type.python_type), f'Parameter "{param_name}" is not read in correctly from the resourcefile.'

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

        # # Declare symptoms that this modules will cause and which are not included in the generic symptoms:
        # generic_symptoms = self.sim.modules['SymptomManager'].parameters['generic_symptoms']
        # for symptom_name in self.symptoms:
        #     if symptom_name not in generic_symptoms:
        #         self.sim.modules['SymptomManager'].register_symptom(
        #             Symptom(name=symptom_name)  # (give non-generic symptom 'average' healthcare seeking)
        #         )

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

        # ---- Key Current Status Classification Properties ----
        df['ri_primary_ALRI_pathogen'].values[:] = 'none'
        df['ri_current_ALRI_symptoms'] = 'not_applicable'
        df['ri_secondary_bacterial_pathogen'] = 'none'
        df['ri_ALRI_disease_type'] = 'not_applicable'
        df['ri_ALRI_complications'] = 'none'

        # ---- Internal values ----
        df['ri_ALRI_event_date_of_onset'] = pd.NaT
        df['ri_ALRI_event_recovered_date'] = pd.NaT
        df['ri_ALRI_event_death_date'] = pd.NaT

        df['ri_ALRI_treatment'] = False
        df['ri_ALRI_tx_start_date'] = pd.NaT

        # ---- Temporary values ----
        df['tmp_malnutrition'] = False
        df['tmp_hv_inf'] = False
        df['tmp_low_birth_weight'] = False
        df['tmp_exclusive_breastfeeding'] = False
        df['tmp_continued_breastfeeding'] = False

    def initialise_simulation(self, sim):
        """
        Get ready for simulation start.
        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # Schedule the main polling event (to first occur immidiately)
        sim.schedule_event(AcuteLowerRespiratoryInfectionPollingEvent(self), sim.date)

        # Schedule the main logging event (to first occur in one year)
        sim.schedule_event(AcuteLowerRespiratoryInfectionLoggingEvent(self), sim.date + DateOffset(years=1))

        # Get DALY weights
        get_daly_weight = self.sim.modules['HealthBurden'].get_daly_weight
        if 'HealthBurden' in self.sim.modules.keys():
            self.daly_wts['daly_ALRI'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=47)
            self.daly_wts['daly_severe_ALRI'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=47)
            self.daly_wts['daly_very_severe_ALRI'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=46)

        # =====================================================================================================
        # APPLY A LINEAR MODEL FOR THE ACQUISITION OF A PRIMARY PATHOGEN FOR ALRI
        # --------------------------------------------------------------------------------------------
        # Make a dict to hold the equations that govern the probability that a person acquires ALRI
        # that is caused (primarily) by a pathogen
        p = self.parameters

        def make_scaled_linear_model(patho):
            """Makes the unscaled linear model with default intercept of 1. Calculates the mean incidents rate for
            0-year-olds and then creates a new linear model with adjusted intercept so incidents in 0-year-olds
            matches the specified value in the model when averaged across the population
            """
            def make_linear_model(patho, intercept=1.0):
                base_inc_rate = f'base_inc_rate_ALRI_by_{patho}'
                return LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    intercept,
                    Predictor('age_years')
                        .when('.between(0,0)', p[base_inc_rate][0])
                        .when('.between(1,1)', p[base_inc_rate][1])
                        .when('.between(2,4)', p[base_inc_rate][2])
                        .otherwise(0.0),
                    Predictor('li_no_access_handwashing').when(False, p['rr_ALRI_HHhandwashing']),
                    Predictor('li_wood_burn_stove').when(False, p['rr_ALRI_indoor_air_pollution']),
                    Predictor('tmp_hv_inf').when(True, p['rr_ALRI_HIV_untreated']),
                    Predictor('tmp_malnutrition').when(True, p['rr_ALRI_underweight']),
                    Predictor('tmp_exclusive_breastfeeding').when(False, p['rr_ALRI_excl_breastfeeding'])
                )

            df = self.sim.population.props
            unscaled_lm = make_linear_model(patho)
            target_mean = p[f'base_inc_rate_diarrhoea_by_{patho}'][0]
            actual_mean = unscaled_lm.predict(df.loc[df.is_alive & (df.age_years == 0)]).mean()
            scaled_intercept = 1.0 * (target_mean / actual_mean)
            scaled_lm = make_linear_model(patho, intercept=scaled_intercept)
            # check by applying the model to mean incidence of 0-year-olds
            assert (target_mean - scaled_lm.predict(df.loc[df.is_alive & (df.age_years == 0)]).mean()) < 1e-10
            return scaled_lm

        for pathogen in ALRI.pathogens:
            self.incidence_equations_by_pathogen[pathogen] = make_scaled_linear_model(pathogen)

        # check that equations have been declared for each pathogens
        assert self.pathogens == set(list(self.incidence_equations_by_pathogen.keys()))

        # --------------------------------------------------------------------------------------------
        # Linear models for determining the underlying condition as viral or bacterial pneumonia, and bronchiolitis
        # caused by each primary pathogen
        def determine_ALRI_type(patho):
            return {
                'viral_pneumonia': p[f'proportion_viral_pneumonia_caused_by_{patho}'],
                'bronchiolitis': 1- p[f'proportion_viral_pneumonia_caused_by_{patho}'],
                'bacterial_pneumonia': p[f'proportion_bacterial_pneumonia_caused_by_{patho}']
            }
        for pathogen in ALRI.pathogens:
            self.proportions_of_ALRI_disease_types_by_pathogen[pathogen] = determine_ALRI_type(pathogen)

        # check that equations have been declared for each pathogens
        assert self.pathogens == set(list(self.proportions_of_ALRI_disease_types_by_pathogen.keys()))

        # =====================================================================================================
        # APPLY PROBABILITY OF SECONDARY BACTERIAL INFECTION
        # -----------------------------------------------------------------------------------------------------
        # Create linear model equation for the probability of a secondary bacterial infection
        self.prob_secondary_bacterial_infection = \
            LinearModel(LinearModelType.MULTIPLICATIVE,
                        1.0,
                        Predictor()
                        .when(
                            "ri_primary_ALRI_pathogen.isin(['RSV', 'rhinovirus', 'hMPV', "
                            "'parainfluenza', 'influenza']) & "
                            "ri_current_ALRI_disease_type == 'viral pneumonia'",
                            p[
                                'prob_secondary_bacterial_infection_from_viral_pneumonia'])
                        .otherwise(0.0),
                        Predictor()
                        .when(
                            "ri_primary_ALRI_pathogen.isin(['RSV', 'rhinovirus', 'hMPV', "
                            "'parainfluenza', 'influenza']) & "
                            "ri_current_ALRI_disease_type == 'bronchiolitis'",
                            p[
                                'prob_secondary_bacterial_infection_from_bronchiolitis'])
                        .otherwise(0.0)
                        )

        # =====================================================================================================
        # APPLY LINEAR MODEL TO DETERMINE PROBABILITY OF COMPLICATIONS
        # -----------------------------------------------------------------------------------------------------
        # Create linear models for the risk of acquiring complications from uncomplicated ALRI
        self.risk_of_developing_ALRI_complications.update({
            'pneumothorax':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_primary_ALRI_pathogen')
                            .when(
                                ".isin(['streptococcus', 'hib', 'TB', 'staphylococcus', 'other_pathogens'])",
                                p['prob_pneumothorax_by_bacterial_ALRI'])
                            .otherwise(0.0)
                            ),

            'pleural_effusion':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_primary_ALRI_pathogen')
                            .when(
                                ".isin(['streptococcus', 'hib', 'TB', 'staphylococcus', 'other_pathogens'])",
                                p['prob_pleural_effusion_by_bacterial_ALRI'])
                            .otherwise(0.0)
                            ),

            'empyema':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_current_ALRI_complications')
                            .when('pleural_effusion', p['prob_pleural_effusion_to_empyema'])
                            .otherwise(0.0)
                            ),

            'lung_abscess':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_primary_ALRI_pathogen')
                            .when(
                                ".isin(['streptococcus', 'hib', 'TB', 'staphylococcus', 'other_pathogens'])",
                                p['prob_lung_abscess_by_bacterial_ALRI'])
                            .otherwise(0.0)
                            ),

            'sepsis':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_primary_ALRI_pathogen')
                            .when(
                                ".isin(['streptococcus', 'hib', 'TB', 'staphylococcus', 'other_pathogens'])",
                                p['prob_sepsis_by_bacterial_ALRI'])
                            .when(
                                ".isin(['RSV', 'rhinovirus', 'hMPV', 'parainfluenza', 'influenza'])",
                                p['prob_sepsis_by_viral_ALRI'])
                            .otherwise(0.0)
                            ),

            'meningitis':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_primary_ALRI_pathogen')
                            .when(
                                ".isin(['streptococcus', 'hib', 'TB', 'staphylococcus', 'other_pathogens'])",
                                p['prob_meningitis_by_bacterial_ALRI'])
                            .when(
                                ".isin(['RSV', 'rhinovirus', 'hMPV', 'parainfluenza', 'influenza'])",
                                p['prob_meningitis_by_viral_ALRI'])
                            .otherwise(0.0)
                            ),

            'respiratory_failure':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_primary_ALRI_pathogen')
                            .when(
                                ".isin(['streptococcus', 'hib', 'TB', 'staphylococcus', 'other_pathogens'])",
                                p['prob_respiratory_failure_by_bacterial_ALRI'])

                            .when(
                                ".isin(['RSV', 'rhinovirus', 'hMPV', 'parainfluenza', 'influenza'])",
                                p['prob_respiratory_failure_by_viral_ALRI'])
                            .otherwise(0.0)
                            ),
        }),

        # check that equations have been declared for each complication
        assert self.complications == set(list(self.risk_of_developing_ALRI_complications.keys()))

        # =====================================================================================================
        # APPLY PROBABILITY OF SYMPTOMS TO EACH ALRI DISEASE TYPE (UNCOMPLICATED AND WITH COMPLICATIONS)
        # -----------------------------------------------------------------------------------------------------
        # Make a dict containing the probability of symptoms given acquisition of (uncomplicated) ALRI,
        # by disease type
        def make_symptom_probs(disease_type):
            return {
                'fever': p[f'prob_fever_uncomplicated_{disease_type}'],
                'cough': p[f'prob_cough_uncomplicated_{disease_type}'],
                'difficult_breathing': p[f'prob_difficult_breathing_uncomplicated_{disease_type}'],
                'fast_breathing': p[f'prob_fast_breathing_uncomplicated_{disease_type}'],
                'chest_indrawing': p[f'prob_chest_indrawing_uncomplicated_{disease_type}'],
                'danger_signs': p[f'prob_danger_signs_uncomplicated_{disease_type}'],
            }

        for disease in ALRI.disease_type:
            self.prob_symptoms_uncomplicated_ALRI[disease] = make_symptom_probs(disease)

        # Check that each ALRI type has a risk of developing each symptom
        assert self.disease_type == set(list(self.prob_symptoms_uncomplicated_ALRI.keys()))

        # -----------------------------------------------------------------------------------------------------
        # Make a dict containing the probability of additional symptoms given acquisition of complications
        # probability by complication
        def add_complication_symptom_probs(complicat):
            return {
                'fast_breathing': p[f'prob_fast_breathing_adding_from_{complicat}'],
                'chest_indrawing': p[f'prob_chest_indrawing_adding_from_{complicat}'],
                'convulsions': p[f'prob_conculsions_adding_from_{complicat}'],
                'severe_respiratory_distress': p[f'prob_severe_respiratory_distress_adding_from_{complicat}'],
                'grunting': p[f'prob_grunting_adding_from_{complicat}'],
                'hypoxia': p[f'prob_hypoxia_adding_from_{complicat}'],
            }

        for complication in ALRI.complications:
            self.prob_extra_symptoms_complications[complication] = add_complication_symptom_probs(complication)

        # Check that each complication has a risk of developing each symptom
        assert self.complications == set(list(self.prob_extra_symptoms_complications.keys()))

        # =====================================================================================================
        # APPLY A LINEAR MODEL FOR THE RISK OF DEATH DUE TO ALRI (SEVERE COMPLICATIONS)
        # -----------------------------------------------------------------------------------------------------
        # Create a linear model for the risk of dying due to complications: sepsis, meningitis, respiratory failure
        def death_risk(complications_list):
            total = 0
            if 'sepsis' in complications_list:
                total += p['r_death_from_ALRI_due_to_sepsis']
            if 'respiratory_failure' in complications_list:
                total += p['r_death_from_ALRI_due_to_respiratory_failure']
            if 'meningitis' in complications_list:
                total += p['r_death_from_ALRI_due_to_meningitis']
            return total

        self.risk_of_death_severe_ALRI = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1.0,
            Predictor('ri_current_ALRI_complications').apply(death_risk),
            Predictor('tmp_hv_inf').when(True, p['rr_death_ALRI_HIV']),
            Predictor('tmp_malnutrition').when(True, p['rr_death_ALRI_SAM']),
            Predictor('tmp_low_birth_weight').when(True, p['rr_death_ALRI_low_birth_weight']),
            Predictor('age_years')
                .when('.between(1,1)', p['rr_death_ALRI_age12to23mo'])
                .when('.between(2,4)', p['rr_death_ALRI_age24to59mo'])
        )
        # -----------------------------------------------------------------------------------------------------

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        # ---- Key Current Status Classification Properties ----
        df.at[child_id, 'ri_primary_ALRI_pathogen'] = 'none'
        df.at[child_id, 'ri_current_ALRI_symptoms'] = 'not_applicable'
        df.at[child_id, 'ri_secondary_bacterial_pathogen'] = 'none'
        df.at[child_id, 'ri_ALRI_disease_type'] = 'not_applicable'
        df.at[child_id, 'ri_ALRI_complications'] ='none'

        # ---- Internal values ----
        df.at[child_id, 'ri_ALRI_event_date_of_onset'] = pd.NaT
        df.at[child_id, 'ri_ALRI_event_recovered_date'] = pd.NaT
        df.at[child_id, 'ri_ALRI_event_death_date'] = pd.NaT

        # ---- Temporary values ----
        df.at[child_id, 'tmp_malnutrition'] = False
        df.at[child_id, 'tmp_hv_inf'] = False
        df.at[child_id, 'tmp_low_birth_weight'] = False
        df.at[child_id, 'tmp_exclusive_breastfeeding'] = False
        df.at[child_id, 'tmp_continued_breastfeeding'] = False


    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

    pass

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.
        pass

        # logger.debug('This is ALRI reporting my health values')
        # df = self.sim.population.props
        # p = self.parameters
        #
        # total_daly_values = pd.Series(data=0.0, index=df.loc[df['is_alive']].index)
        # total_daly_values.loc[
        #     self.sim.modules['SymptomManager'].who_has('fast_breathing')
        # ] = self.daly_wts['daly_ALRI']
        # total_daly_values.loc[
        #     self.sim.modules['SymptomManager'].who_has('chest_indrawing')
        # ] = self.daly_wts['daly_severe_ALRI']
        # total_daly_values.loc[
        #     self.sim.modules['SymptomManager'].who_has('danger_signs')
        # ] = self.daly_wts['daly_severe_ALRI']
        #
        # # health_values = df.loc[df.is_alive, 'ri_specific_symptoms'].map({
        # #     'none': 0,
        # #     'ALRI': p['daly_ALRI'],
        # #     'severe ALRI': p['daly_severe_ALRI'],
        # #     'very severe ALRI': p['daly_very_severe_ALRI']
        # # })
        # # health_values.name = 'Pneumonia Symptoms'  # label the cause of this disability
        # # return health_values.loc[df.is_alive]  # returns the series
        #
        # # Split out by pathogen that causes the ALRI
        # dummies_for_pathogen = pd.get_dummies(df.loc[total_daly_values.index,
        #                                              'ri_primary_ALRI_pathogen'],
        #                                       dtype='float')
        # daly_values_by_pathogen = dummies_for_pathogen.mul(total_daly_values, axis=0).drop(columns='none')
        #
        # return daly_values_by_pathogen

    def cancel_death_date(self, person_id):
        """
        Cancels a scheduled date of death due to diarrhoea for a person. This is called prior to the scheduling the
        CureEvent to prevent deaths happening in the time between a treatment being given and the cure event occurring.
        :param person_id:
        :return:
        """
        df = self.sim.population.props
        df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------
class AcuteLowerRespiratoryInfectionPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is the main event that runs the acquisition of pathogens that cause ALRI.
        It determines who is infected and when and schedules individual IncidentCase events to represent onset.

        A known issue is that diarrhoea events are scheduled based on the risk of current age but occur a short time
        later when the children will be slightly older. This means that when comparing the model output with data, the
        model slightly under-represents incidence among younger age-groups and over-represents incidence among older
        age-groups. This is a small effect when the frequency of the polling event is high.
    """
    # TODO: how to fix this

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=3))
        # NB. The frequency of the occurrences of this event can be edited safely.

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = self.module.rng

        # Compute the incidence rate for each person getting ALRI and then convert into a probability
        # getting all children that do not currently have an ALRI episode (never had or last episode resolved)
        mask_could_get_new_alri_event = \
            df['is_alive'] & (df['age_years'] < 5) & ((df['ri_ALRI_event_recovered_date'] <= self.sim.date) |
                                                      pd.isnull(df['ri_ALRI_event_recovered_date']))

        # Compute the incidence rate for each person acquiring ALRI
        inc_of_acquiring_alri = pd.DataFrame(index=df.loc[mask_could_get_new_alri_event].index)
        for pathogen in m.pathogens:
            inc_of_acquiring_alri[pathogen] = m.incidence_equations_by_pathogen[pathogen] \
                .predict(df.loc[mask_could_get_new_alri_event])

        # Convert the incidence rates that are predicted by the model into risk of an event occurring before the next
        # polling event
        fraction_of_a_year_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(
            1, 'Y')
        days_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'D')
        probs_of_acquiring_pathogen = 1 - np.exp(-inc_of_acquiring_alri * fraction_of_a_year_until_next_polling_event)

        # Create the probability of getting 'any' pathogen:
        # (Assumes that pathogens are mutually exclusive); Prevents probability being greater than 1.0.
        prob_of_acquiring_any_pathogen = probs_of_acquiring_pathogen.sum(axis=1).clip(upper=1.0)
        assert all(prob_of_acquiring_any_pathogen < 1.0)

        # Determine which persons will acquire any pathogen:
        person_id_that_acquire_pathogen = prob_of_acquiring_any_pathogen.index[
            rng.rand(len(prob_of_acquiring_any_pathogen)) < prob_of_acquiring_any_pathogen]

        # Determine which pathogen each person will acquire (among those who will get a pathogen)
        # and create the event for the onset of new infection
        for person_id in person_id_that_acquire_pathogen:
            # ----------------------- Allocate a pathogen to the person ----------------------
            p_by_pathogen = probs_of_acquiring_pathogen.loc[person_id].values
            # print(sum(p_by_pathogen))
            normalised_p_by_pathogen = p_by_pathogen / sum(p_by_pathogen)
            # print(sum(normalised_p_by_pathogen))
            pathogen = rng.choice(probs_of_acquiring_pathogen.columns, p=normalised_p_by_pathogen)

            # ----------------------- Allocate a date of onset of ALRI ----------------------
            date_onset = self.sim.date + DateOffset(days=np.random.randint(0, days_until_next_polling_event))
            # duration
            duration_in_days_of_alri = max(1, int(
                7 + (-2 + 4 * rng.rand())))  # assumes uniform interval around mean duration with range 4 days

            # ----------------------- Allocate symptoms to onset of ALRI ----------------------
            # possible_symptoms_by_severity = m.prob_symptoms_uncomplicated_ALRI
            # symptoms_for_this_person = list()
            # for symptom, prob in possible_symptoms_by_severity.items():
            #     if rng.rand() < prob:
            #         symptoms_for_this_person.append(symptom)

            # ----------------------- Create the event for the onset of infection -------------------
            # NB. The symptoms are scheduled by the SymptomManager to 'autoresolve' after the duration
            #       of the ALRI.
            self.sim.schedule_event(
                event=AcuteLowerRespiratoryInfectionIncidentCase(
                    module=self.module,
                    person_id=person_id,
                    pathogen=pathogen,
                    duration_in_days=duration_in_days_of_alri,
                    # symptoms=symptoms_for_this_person
                ),
                date=date_onset
            )


class AcuteLowerRespiratoryInfectionIncidentCase(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of the infection that causes ALRI.
    """

    def __init__(self, module, person_id, pathogen, duration_in_days):
        super().__init__(module, person_id=person_id)
        self.pathogen = pathogen
        self.duration_in_days = duration_in_days
        # self.symptoms = symptoms

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        rng = self.module.rng

        # The event should not run if the person is not currently alive
        if not df.at[person_id, 'is_alive']:
            return

        # Update the properties in the dataframe:
        df.at[person_id, 'ri_primary_ALRI_pathogen'] = self.pathogen
        df.at[person_id, 'ri_ALRI_event_date_of_onset'] = self.sim.date
        df.at[person_id, 'ri_current_ALRI_complications'] = 'none' # all disease start as non-severe symptoms
        df.at[person_id, 'ri_current_ALRI_symptoms'] = self.symptoms

        for pathogen, prob in m.proportions_of_ALRI_disease_types_by_pathogen.items():
            if rng.rand() < prob:
                df.at[person_id, 'ri_current_ALRI_disease_type'] = 'primarily bacterial'
        # # determine if it is viral or bacterial ALRI based on pathogen
        # if self.pathogen == self.module.pathogen_type['viral']:
        #     if
        #     df.at[person_id, 'ri_current_ALRI_disease_type'] = 'primarily viral'
        # if self.pathogen == self.module.pathogen_type['bacterial']:
        #     df.at[person_id, 'ri_current_ALRI_disease_type'] = 'primarily bacterial'

        # ----------------------- Allocate symptoms to onset of ALRI ----------------------
        possible_symptoms_by_severity = m.prob_symptoms_uncomplicated_ALRI[disease_type]
        symptoms_for_this_person = list()
        for symptom, prob in possible_symptoms_by_severity.items():
            if rng.rand() < prob:
                symptoms_for_this_person.append(symptom)

        # Onset symptoms:
        for symptom in symptoms_for_this_person:
            self.module.sim.modules['SymptomManager'].change_symptom(
                person_id=person_id,
                symptom_string=symptom,
                add_or_remove='+',
                disease_module=self.module,
                duration_in_days=self.duration_in_days
            )

        date_of_outcome = self.module.sim.date + DateOffset(days=self.duration_in_days)

        complications_for_this_person = list()
        for complication in self.module.complications:
            prob_developing_each_complication = m.risk_of_developing_ALRI_complications[complication].predict(
                df.loc[[person_id]]).values[0]
            if rng.rand() < prob_developing_each_complication:
                complications_for_this_person.append(complication)
                df.at[person_id, 'ri_ALRI_event_recovered_date'] = pd.NaT
                df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT
            else:
                df.at[person_id, 'ri_ALRI_event_recovered_date'] = date_of_outcome
                df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT

        if len(complications_for_this_person) != 0:
            for i in complications_for_this_person:
                date_onset_complications = self.module.sim.date + DateOffset(
                    days=np.random.randint(3, high=self.duration_in_days))
                print(i, date_onset_complications)
                self.sim.schedule_event(PneumoniaWithComplicationsEvent(
                    self.module, person_id, duration_in_days=self.duration_in_days, symptoms=self.symptoms,
                    complication=complications_for_this_person), date_onset_complications)

        self.sim.modules['DxAlgorithmChild'].imnci_as_gold_standard(person_id=person_id)

        # Add this incident case to the tracker
        age = df.loc[person_id, ['age_years']]
        if age.values[0] < 5:
            age_grp = age.map({0: '0y', 1: '1y', 2: '2-4y', 3: '2-4y', 4: '2-4y'}).values[0]
        else:
            age_grp = '5+y'
        self.module.incident_case_tracker[age_grp][self.pathogen].append(self.sim.date)


class PneumoniaWithComplicationsEvent(Event, IndividualScopeEventMixin):
        """
            This Event is for the onset of Clinical Severe Pneumonia. For some untreated children,
            this occurs a set number of days after onset of disease.
            It sets the property 'ri_current_ALRI_complications' to each complication and schedules the death.
            """

        def __init__(self, module, person_id, duration_in_days, symptoms, complication):
            super().__init__(module, person_id=person_id)
            self.duration_in_days = duration_in_days
            self.complication = complication
            self.symptoms = symptoms

        def apply(self, person_id):
            df = self.sim.population.props  # shortcut to the dataframe
            m = self.module
            rng = self.module.rng

            # terminate the event if the person has already died.
            if not df.at[person_id, 'is_alive']:
                return

            # complications for this person
            df.at[person_id, 'ri_current_ALRI_complications'] = list(self.complication)

            # add to the initial list of uncomplicated ALRI symptoms
            all_symptoms_for_this_person = list(self.symptoms)  # original uncomplicated symptoms list to add to

            # keep only the probabilities for the complications of the person:
            possible_symptoms_by_complication = {key: val for key, val in
                                                 self.module.prob_extra_symptoms_complications.items()
                                                 if key in list(self.complication)}
            print(possible_symptoms_by_complication)
            symptoms_from_complications = list()
            for complication in possible_symptoms_by_complication:
                for symptom, prob in possible_symptoms_by_complication[complication].items():
                    if self.module.rng.rand() < prob:
                        symptoms_from_complications.append(symptom)
                    for i in symptoms_from_complications:
                        # add symptoms from complications to the list
                        all_symptoms_for_this_person.append(
                            i) if i not in all_symptoms_for_this_person \
                            else all_symptoms_for_this_person

            print(all_symptoms_for_this_person)
            df.at[person_id, 'ri_current_ALRI_symptoms'] = all_symptoms_for_this_person

            for symptom in all_symptoms_for_this_person:
                self.module.sim.modules['SymptomManager'].change_symptom(
                    person_id=person_id,
                    symptom_string=symptom,
                    add_or_remove='+',
                    disease_module=self.module,
                    duration_in_days=self.duration_in_days
                )

            # Determine death outcome -------------------------------------------------------------------------
            date_of_outcome = \
                df.at[person_id, 'ri_ALRI_event_date_of_onset'] + DateOffset(days=self.duration_in_days)

            prob_death_from_ALRI =\
                m.risk_of_death_severe_ALRI.predict(df.loc[[person_id]]).values[0]
            death_outcome = rng.rand() < prob_death_from_ALRI

            if death_outcome:
                df.at[person_id, 'ri_ALRI_event_recovered_date'] = pd.NaT
                df.at[person_id, 'ri_ALRI_event_death_date'] = date_of_outcome
                self.sim.schedule_event(PneumoniaDeathEvent(self.module, person_id),
                                        date_of_outcome)
            else:
                df.at[person_id, 'ri_ALRI_event_recovered_date'] = date_of_outcome
                df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT


class PneumoniaCureEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("PneumoniaCureEvent: Stopping ALRI treatment and curing person %d", person_id)
        df = self.sim.population.props

        # terminate the event if the person has already died.
        if not df.at[person_id, 'is_alive']:
            return

        # Stop the person from dying of ALRI (if they were going to die)
        df.at[person_id, 'ri_ALRI_event_recovered_date'] = self.sim.date
        df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT

        # clear the treatment prperties
        df.at[person_id, 'ri_ALRI_treatment'] = False
        df.at[person_id, 'ri_ALRI_tx_start_date'] = pd.NaT

        # Resolve all the symptoms immediately
        self.sim.modules['SymptomManager'].clear_symptoms(person_id=person_id,
                                                          disease_module=self.sim.modules['Pneumonia'])


class PneumoniaDeathEvent(Event, IndividualScopeEventMixin):
    """
    This Event is for the death of someone that is caused by the infection with a pathogen that causes ALRI.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        # Check if person should still die of ALRI
        if (df.at[person_id, 'is_alive']) and \
            (df.at[person_id, 'ri_ALRI_event_death_date'] == self.sim.date):
            self.sim.schedule_event(demography.InstantaneousDeath(self.module,
                                                                  person_id,
                                                                  cause='Pneumonia_' + df.at[
                                                                      person_id, 'ri_primary_ALRI_pathogen']
                                                                  ), self.sim.date)


# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
# ---------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
# ---------------------------------------------------------------------------------------------------------

class AcuteLowerRespiratoryInfectionLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This Event logs the number of incident cases that have occurred since the previous logging event.
    Analysis scripts expect that the frequency of this logging event is once per year.
    """

    def __init__(self, module):
        # This event to occur every year
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        self.date_last_run = self.sim.date

    def apply(self, population):
        df = self.sim.population.props
        # Convert the list of timestamps into a number of timestamps
        # and check that all the dates have occurred since self.date_last_run
        counts = copy.deepcopy(self.module.incident_case_tracker_zeros)

        for age_grp in self.module.incident_case_tracker.keys():
            for pathogen in self.module.pathogens:
                list_of_times = self.module.incident_case_tracker[age_grp][pathogen]
                counts[age_grp][pathogen] = len(list_of_times)
                for t in list_of_times:
                    assert self.date_last_run <= t <= self.sim.date

        logger.info('%s|incidence_count_by_pathogen|%s',
                    self.sim.date,
                    counts
                    )

        imci_classification_count = \
            df[df.is_alive & df.age_years.between(0, 5)].groupby('ri_pneumonia_IMCI_classification').size()

        logger.info('%s|imci_classicications_count|%s',
                    self.sim.date,
                    imci_classification_count
                    )

        # Reset the counters and the date_last_run
        self.module.incident_case_tracker = copy.deepcopy(self.module.incident_case_tracker_blank)
        self.date_last_run = self.sim.date
