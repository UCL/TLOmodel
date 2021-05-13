"""
Childhood wasting module
Documentation: '04 - Methods Repository/Undernutrition module - Description.docx'

Overview
=======
This module applies the prevalence of wasting at the population-level, and schedules new incidences of wasting

"""
import copy
from pathlib import Path
from scipy.stats import norm

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, demography
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom
from tlo.util import create_age_range_lookup
from tlo.methods.dxmanager import DxTest

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------


class Wasting(Module):
    """
    This module applies the prevalence of wasting at the population-level,
    based on the Malawi DHS Survey 2015-2016.
    The definitions:
    - moderate wasting: height-for-age Z-score (WHZ) <-2 SD from the reference mean
    - severe wasting: height-for-age Z-score (WHZ) <-3 SD from the reference mean

    """

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    acute_malnutrition_states = ['MAM', 'SAM']

    PARAMETERS = {
        # prevalence of wasting by age group
        'prev_WHZ_distribution_age_0_5mo': Parameter(
            Types.LIST, 'distribution of WHZ among less than 6 months of age in 2015'),
        'prev_WHZ_distribution_age_6_11mo': Parameter(
            Types.LIST, 'distribution of WHZ among 6 months and 1 year of age in 2015'),
        'prev_WHZ_distribution_age_12_23mo': Parameter(
            Types.LIST, 'distribution of WHZ among 1 year olds in 2015'),
        'prev_WHZ_distribution_age_24_35mo': Parameter(
            Types.LIST, 'distribution of WHZ among 2 year olds in 2015'),
        'prev_WHZ_distribution_age_36_47mo': Parameter(
            Types.LIST, 'distribution of WHZ among 3 year olds in 2015'),
        'prev_WHZ_distribution_age_48_59mo': Parameter(
            Types.LIST, 'distribution of WHZ among 4 year olds  in 2015'),
        # effect of risk factors on wasting prevalence
        'or_wasting_motherBMI_underweight': Parameter(
            Types.REAL, 'odds ratio of wasting if mother has low BMI, ref group high BMI (overweight)'),
        'or_wasting_hhwealth_Q5': Parameter(
            Types.REAL, 'odds ratio of wasting if household wealth is poorest Q5, ref group Q1'),
        'or_wasting_hhwealth_Q4': Parameter(
            Types.REAL, 'odds ratio of wasting if household wealth is poorer Q4, ref group Q1'),
        'or_wasting_hhwealth_Q3': Parameter(
            Types.REAL, 'odds ratio of wasting if household wealth is middle Q3, ref group Q1'),
        'or_wasting_hhwealth_Q2': Parameter(
            Types.REAL, 'odds ratio of wasting if household wealth is richer Q2, ref group Q1'),
        'or_wasting_preterm_and_AGA': Parameter(
            Types.REAL, 'odds ratio of wasting if born preterm and adequate for gestational age'),
        'or_wasting_SGA_and_term': Parameter(
            Types.REAL, 'relative risk of wasting if born term and small for geatational age'),
        'or_wasting_SGA_and_preterm': Parameter(
            Types.REAL, 'odds ratio of wasting if born preterm and small for gestational age'),
        # incidence parameters
        'base_inc_rate_wasting_by_agegp': Parameter(
            Types.LIST, 'List with baseline incidence of wasting by age group'),
        'rr_wasting_preterm_and_AGA': Parameter(
            Types.REAL, 'relative risk of wasting if born preterm and adequate for gestational age'),
        'rr_wasting_SGA_and_term': Parameter(
            Types.REAL, 'relative risk of wasting if born term and small for geatational age'),
        'rr_wasting_SGA_and_preterm': Parameter(
            Types.REAL, 'relative risk of wasting if born preterm and small for gestational age'),
        'min_days_duration_of_wasting': Parameter(
            Types.REAL, 'minimum duration in days of wasting (MAM and SAM)'),
        'average_duration_of_untreated_MAM': Parameter(
            Types.REAL, 'average duration of untreated MAM'),
        # progression to severe parameters
        'progression_severe_wasting_by_agegp': Parameter(
            Types.LIST, 'List with progression rates to severe wasting by age group'),
        'rr_progress_severe_wasting_preterm_and_AGA': Parameter(
            Types.REAL, 'relative risk of severe wasting if born preterm and adequate for gestational age'),
        'rr_progress_severe_wasting_SGA_and_term': Parameter(
            Types.REAL, 'relative risk of severe wasting if born term and small for geatational age'),
        'rr_progress_severe_wasting_SGA_and_preterm': Parameter(
            Types.REAL, 'relative risk of severe wasting if born preterm and small for gestational age'),
        'average_duration_of_untreated_SAM': Parameter(
            Types.REAL, 'average duration of untreated SAM'),
        'prob_complications_in_SAM': Parameter(
            Types.REAL, 'probability of medical complications in SAM '),
        # recovery parameters
        'recovery_rate_with_standard_RUTF': Parameter(
            Types.REAL, 'probability of recovery from wasting following treatment with standard RUTF'),
        'recovery_rate_with_soy_RUSF': Parameter(
            Types.REAL, 'probability of recovery from wasting following treatment with soy RUSF'),
        'recovery_rate_with_CSB++': Parameter(
            Types.REAL, 'probability of recovery from wasting following treatment with CSB++'),
        'recovery_rate_with_inpatient_care': Parameter(
            Types.REAL, 'probability of recovery from wasting following treatment with inpatient care'),
        # MUAC distributions
        'MUAC_distribution_WHZ<-3': Parameter(
            Types.LIST, 'mean and standard deviation of a normal distribution of MUAC measurements for WHZ<-3'),
        'MUAC_distribution_-3<=WHZ<-2': Parameter(
            Types.LIST, 'mean and standard deviation of a normal distribution of MUAC measurements for -3<=WHZ<-2'),
        'MUAC_distribution_WHZ>=-2': Parameter(
            Types.LIST, 'mean and standard deviation of a normal distribution of MUAC measurements for WHZ>=-2'),

        'proportion_WHZ<-3_with_MUAC<115mm': Parameter(
            Types.REAL, 'proportion of severe weight-for-height Z-score with MUAC<115mm'),
        'proportion_-3<=WHZ<-2_with_MUAC<115mm': Parameter(
            Types.REAL, 'proportion of moderate weight-for-height Z-score with MUAC<115mm'),
        'proportion_-3<=WHZ<-2_with_MUAC_115-<125mm': Parameter(
            Types.REAL, 'proportion of moderate weight-for-height Z-score with MUAC between 115mm and 125mm'),
        'proportion_mam_with_MUAC_115-<125mm_and_normal_whz': Parameter(
            Types.REAL, 'proportion of mam case with MUAC between 115mm and 125mm and normal/mild WHZ'),

        # bilateral oedema
        'prevalence_nutritional_oedema': Parameter(
            Types.REAL, 'prevalence of nutritional oedema in children under 5 in Malawi'),
        'proportion_oedema_with_WHZ<-2': Parameter(
            Types.REAL, 'proportion of oedematous malnutrition with concurrent wasting'),

        # death CFR, risk factors
        'base_death_rate_untreated_SAM': Parameter(
            Types.REAL, 'baseline death rate of untreated SAM'),
        'cfr_SAM_inpatient_care': Parameter(
            Types.REAL, 'case-fatality-rate of complicated SAM in inpatient care'),
        'cfr_SAM_outpatient_care': Parameter(
            Types.REAL, 'case-fatality-rate of uncomplicated SAM in outpatient care'),
        'rr_SAM_death_WHZ<-3_only': Parameter(
            Types.REAL, 'relative risk of death from SAM if indices of WHZ<-3, compared to MUAC<115mm'),
        'rr_SAM_death_both_WHZ<-3_&_MUAC<115mm': Parameter(
            Types.REAL, 'relative risk of death from SAM if both indices WHZ<-3 & MUAC<115mm are present, '
                        'compared to MUAC<115mm alone'),
        'rr_SAM_death_kwashiorkor_only': Parameter(
            Types.REAL, 'relative risk of death from SAM if bilateral oedema present (kwashiorkor), '
                        'compared to MUAC<115mm alone'),
        'rr_SAM_death_kwashiorkor_MUAC<115mm_only': Parameter(
            Types.REAL, 'relative risk of death from SAM if bilateral oedema present and MUAC<115mm, '
                        'compared to MUAC<115mm alone'),
        'rr_SAM_death_kwashiorkor_WHZ<-3_only': Parameter(
            Types.REAL, 'relative risk of death from SAM if bilateral oedema present and WHZ<-3, '
                        'compared to MUAC<115mm alone'),
        'rr_SAM_death_kwashiorkor_both_WHZ<-3_&_MUAC<115mm': Parameter(
            Types.REAL, 'relative risk of death from SAM if bilateral oedema present, WHZ<-3 and MUAC<115mm, '
                        'compared to MUAC<115mm alone'),
        # treatment parameters
        'coverage_supplementary_feeding_program': Parameter(
            Types.REAL, 'coverage of supplementary feeding program for MAM in health centres'),
        'coverage_outpatient_therapeutic_care': Parameter(
            Types.REAL, 'coverage of outpatient therapeutic care for SAM in health centres'),
        'coverage_inpatient_care': Parameter(
            Types.REAL, 'coverage of inpatient care for complicated SAM in hospitals'),

    }

    PROPERTIES = {
        # Properties related to wasting
        'un_ever_wasted': Property(Types.BOOL, 'had wasting before WHZ <-2'),
        'un_WHZ_category': Property(Types.CATEGORICAL, 'height-for-age z-score group',
                                    categories=['WHZ<-3', '-3<=WHZ<-2', 'WHZ>=-2']),
        'un_last_wasting_date_of_onset': Property(Types.DATE, 'date of onset of lastest wasting episode'),
        'un_wasting_death_date': Property(Types.DATE, 'death date from wasting'),

        # Properties related to clinical acute malnutrition
        'un_clinical_acute_malnutrition': Property(Types.CATEGORICAL, 'clinical acute malnutrition state based on WHZ',
                                                   categories=['MAM', 'SAM'] + ['well']),
        'un_am_bilateral_oedema': Property(Types.BOOL, 'bilateral oedema present in wasting'),
        'un_am_MUAC_category': Property(Types.CATEGORICAL, 'MUAC measurement categories',
                                        categories=['<115mm', '115-<125mm', '>=125mm']),
        'un_SAM_with_complications': Property(Types.BOOL, 'medical complications in SAM'),
        'un_acute_malnutrition_tx_start_date': Property(Types.DATE, 'intervention start date'),
        'un_am_treatment_type': Property(Types.CATEGORICAL, 'treatment types for acute malnutrition',
                                         categories=['standard_RUTF', 'soy_RUSF', 'CSB++', 'inpatient_care']),
        'un_SAM_management_facility_type': Property(Types.CATEGORICAL, 'treatment for SAM at facility level',
                                                    categories=['outpatient', 'inpatient']),
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # dict to hold the probability of onset of different symptoms:
        self.prob_symptoms = dict()

        # Store the symptoms that this module will use:
        self.symptoms = {
            'palmar_pallor',
            'oedema',
            'poor_appetite',
            'lethargic',
            'dehydration'
        }

        # dict to hold counters for the number of episodes by wasting-type and age-group
        blank_counter = dict(zip(self.acute_malnutrition_states, [list() for _ in self.acute_malnutrition_states]))
        self.wasting_incident_case_tracker_blank = {
            '0y': copy.deepcopy(blank_counter),
            '1y': copy.deepcopy(blank_counter),
            '2y': copy.deepcopy(blank_counter),
            '3y': copy.deepcopy(blank_counter),
            '4y': copy.deepcopy(blank_counter),
            '5+y': copy.deepcopy(blank_counter)
        }
        self.wasting_incident_case_tracker = copy.deepcopy(self.wasting_incident_case_tracker_blank)

        zeros_counter = dict(zip(self.acute_malnutrition_states, [0] * len(self.acute_malnutrition_states)))
        self.wasting_incident_case_tracker_zeros = {
            '0y': copy.deepcopy(zeros_counter),
            '1y': copy.deepcopy(zeros_counter),
            '2y': copy.deepcopy(zeros_counter),
            '3y': copy.deepcopy(zeros_counter),
            '4y': copy.deepcopy(zeros_counter),
            '5+y': copy.deepcopy(zeros_counter)
        }

        # dict to hold the DALY weights
        self.daly_wts = dict()

        # # Create the diagnostic representing the assessment for whether a person is diagnosed with acute malnutrition
        # # NB. Specificity is assumed to be 100%
        # self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
        #     assess_acute_malnutrition=DxTest(
        #         property='un_clinical_acute_malnutrition',
        #         sensitivity=self.parameters['coverage_inpatient_care'],
        #     )
        # )

        # --------------------- linear models of the natural history --------------------- #

        # set the linear model equations for prevalence and incidence
        self.prevalence_equations_by_age = dict()
        self.wasting_incidence_equation = dict()

        # set the linear model for progression to severe wasting
        self.severe_wasting_progression_equation = dict()

        # set the linear model for death from severe acute malnutrition
        self.wasting_death_equation = dict()

        # --------------------- linear models following HSI interventions --------------------- #

        # set the linear models for MAM and SAM recovery by intervention
        self.acute_malnutrition_recovery_based_on_interventions = dict()

    def read_parameters(self, data_folder):
        """
        :param data_folder: path of a folder supplied to the Simulation containing data files.
              Typically modules would read a particular file within here.
        :return:
        """
        # Update parameters from the resource dataframe
        # Read parameters from the resourcefile
        self.load_parameters_from_dataframe(
            pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Undernutrition.xlsx',
                          sheet_name='Parameter_values_AM'))
        p = self.parameters

        # Check that every value has been read-in successfully
        for param_name, param_type in self.PARAMETERS.items():
            assert param_name in p, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert param_name is not None, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert isinstance(p[param_name],
                              param_type.python_type), f'Parameter "{param_name}" is not read in correctly from the ' \
                                                       f'resourcefile.'

        # Declare symptoms that this module will cause and which are not included in the generic symptoms:
        generic_symptoms = self.sim.modules['SymptomManager'].parameters['generic_symptoms']
        for symptom_name in self.symptoms:
            if symptom_name not in generic_symptoms:
                self.sim.modules['SymptomManager'].register_symptom(
                    Symptom(name=symptom_name)  # (give non-generic symptom 'average' healthcare seeking)
                )

    def muac_cutoff_by_WHZ(self, idx, whz):
        """
        Proportion of MUAC<115mm in WHZ<-3 and -3<=WHZ<-2,
        and proportion of wasted children with oedematous malnutrition (Kwashiokor, marasmic-kwashiorkor)
        :param idx:
        :param whz:
        :return:
        """
        df = self.sim.population.props
        p = self.parameters

        # get those children with MUAC cuf-off
        children_with_wasting = idx.intersection(df.un_WHZ_category != 'WHZ>=-2'.index)
        children_without_wasting = idx.intersection(df.un_WHZ_category == 'WHZ>=-2'.index)

        # ---------- MUAC <115mm in severe wasting (WHZ<-3) and moderate (-3<=WHZ<-2) ----------
        if whz=='WHZ<-3':
            # apply probability of MUAC<115mm in severe wasting
            low_muac_in_severe_wasting = self.rng.random_sample(size=len(idx)) < p['proportion_WHZ<-3_with_MUAC<115mm']
            df.loc[idx[low_muac_in_severe_wasting], 'un_am_MUAC_category'] = '<115mm'
            # other severe wasting will have MUAC between 115-<125mm
            df.loc[idx[low_muac_in_severe_wasting==False], 'un_am_MUAC_category'] = '115-<125mm'

        if whz=='-3<=WHZ<-2':
            # apply probability of MUAC<115mm in moderate wasting
            low_muac_in_moderate_wasting = self.rng.random_sample(size=len(idx)) < p[
                'proportion_-3<=WHZ<-2_with_MUAC<115mm']
            df.loc[idx[low_muac_in_moderate_wasting], 'un_am_MUAC_category'] = '<115mm'
            # apply probability of MUAC between 115-<125mm in moderate wasting
            moderate_low_muac_in_moderate_wasting = self.rng.random_sample(size=len(
                idx[low_muac_in_moderate_wasting==False])) < p['proportion_-3<=WHZ<-2_with_MUAC_115-<125mm']
            df.loc[idx[low_muac_in_moderate_wasting==False][moderate_low_muac_in_moderate_wasting],
                   'un_am_MUAC_category'] = '115-<125mm'
            # other moderate wasting will have normal MUAC
            df.loc[idx[low_muac_in_moderate_wasting==False][moderate_low_muac_in_moderate_wasting==False],
                   'un_am_MUAC_category'] = '115-<125mm'

    def nutritional_oedema_present(self, idx):
        """
        This function applies the probability of bilateral oedema present in wasting and non-wasted cases
        :param idx:
        :return:
        """
        df = self.sim.population.props
        p = self.parameters

        # Knowing the prevalence of nutritional oedema in under 5 population, apply the probability of oedema in WHZ<-2
        # get those children with wasting
        children_with_wasting = idx.intersection(df.un_WHZ_category != 'WHZ>=-2'.index)
        children_without_wasting = idx.intersection(df.un_WHZ_category == 'WHZ>=-2'.index)

        # oedema among wasted children
        oedema_in_wasted_children = self.rng.random_sample(size=len(children_with_wasting)) < \
                                    p['prevalence_nutritional_oedema'] * p['proportion_oedema_with_WHZ<-2']
        df.loc[children_with_wasting[oedema_in_wasted_children], 'un_am_bilateral_oedema'] = True
        df.loc[children_with_wasting[oedema_in_wasted_children==False], 'un_am_bilateral_oedema'] = False

        # oedema among non-wasted children
        oedema_in_non_wasted = self.rng.random_sample(size=len(children_without_wasting)) < \
                               p['prevalence_nutritional_oedema'] * (1 - p['proportion_oedema_with_WHZ<-2'])
        df.loc[children_without_wasting[oedema_in_non_wasted], 'un_am_bilateral_oedema'] = True
        df.loc[children_without_wasting[oedema_in_non_wasted==False], 'un_am_bilateral_oedema'] = False

    def clinical_acute_malnutrition_state(self, person_id):
        """
        This fuction will determine the clinical acute malnutrition status (MAM, SAM) based on anthropometric indices
        and presence of bilateral oedema (Kwashiorkor);
        And help determine whether the individual will have medical complications, applicable to SAM cases only,
        requiring inpatient care.
        :param person_id:
        :return:
        """
        df = self.sim.population.props
        p = self.parameters

        # check if person is not wasted
        if (df.at[person_id, 'un_WHZ_category'] == 'WHZ>=-2') & (
            df.at[person_id, 'un_am_MUAC_category'] == '>=125mm') & (
            df.at[person_id, 'un_am_bilateral_oedema'] == False):
            df.at[person_id, 'un_clinical_acute_malnutrition'] = 'well'

        # severe acute malnutrition - MUAC<115mm and/or WHZ<-3 and/or bilateral oedema
        if ((df.at[person_id, 'un_am_MUAC_category'] == '<115mm') | (
            df.at[person_id, 'un_WHZ_category'] == 'WHZ<-3') | (
            df.at[person_id, 'un_am_bilateral_oedema'] == True)):
            df.at[person_id, 'un_clinical_acute_malnutrition'] = 'SAM'
        elif ((df.at[person_id, 'un_am_MUAC_category'] == '115-<125mm') | (
            df.at[person_id, 'un_WHZ_category'] == '-3<=WHZ<-2')) & (
            df.at[person_id, 'un_am_bilateral_oedema'] == False):
            df.at[person_id, 'un_clinical_acute_malnutrition'] = 'MAM'

        # -------------------------------------------------------------------------------
        # Determine those SAM with complications that need inpatient treatment
        if df.at[person_id, 'un_clinical_acute_malnutrition'] == 'SAM':
            if self.rng.rand() < p['prob_complications_in_SAM']:
                df.at[person_id, 'un_SAM_with_complications'] = True

    def date_of_outcome_for_untreated_am(self, person_id, am_severity):
        """
        helper funtion to get the duration and the wasting episode and date of outcome (recovery, progression, or death)
        :param person_id:
        :param am_severity:
        :return:
        """
        df = self.sim.population.props
        p = self.parameters

        # moderate wasting (for progression to severe, or recovery from MAM) -----
        if am_severity == 'MAM':
            # Allocate the duration of the moderate wasting episode
            duration_mam = int(max(p['min_days_duration_of_wasting'], p['average_duration_of_untreated_MAM']))
            # Allocate a date of outcome (progression, recovery or death)
            date_of_outcome = df.at[person_id, 'un_last_wasting_date_of_onset'] + DateOffset(days=duration_mam)
            return date_of_outcome

        # severe wasting (for death, or recovery to moderate wasting) -----
        if am_severity == 'SAM':
            # determine the duration of SAM episode
            duration_sam = int(max(p['min_days_duration_of_wasting'], p['average_duration_of_untreated_MAM'] +
                                   p['average_duration_of_untreated_SAM']))
            # Allocate a date of outcome (progression, recovery or death)
            date_of_outcome = df.at[person_id, 'un_last_wasting_date_of_onset'] + DateOffset(days=duration_sam)
            return date_of_outcome

    def initialise_population(self, population):
        """
        Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population:
        :return:
        """
        df = population.props
        p = self.parameters

        # Set initial properties
        df.loc[df.is_alive, 'un_ever_wasted'] = False
        df.loc[df.is_alive, 'un_WHZ_category'] = 'WHZ>=-2'  # not undernourished
        df.loc[df.is_alive, 'un_clinical_acute_malnutrition'] = 'well'
        df.loc[df.is_alive, 'un_last_wasting_date_of_onset'] = pd.NaT
        df.loc[df.is_alive, 'un_wasting_death_date'] = pd.NaT
        df.loc[df.is_alive, 'un_am_bilateral_oedema'] = False
        df.loc[df.is_alive, 'un_am_MUAC_category'] = '>=125mm'
        # df.loc[df.is_alive, 'un_am_treatment_type'] = 'not_applicable'
        # df.loc[df.is_alive, 'un_SAM_management_facility_type'] = 'not_applicable'

        # -----------------------------------------------------------------------------------------------------
        # # # # # # allocate initial prevalence of wasting at the start of the simulation # # # # #
        def make_scaled_linear_model_wasting(agegp):
            """ Makes the unscaled linear model with intercept of baseline odds of wasting (WHZ <-2).
            Calculates the mean odds of wasting by age group and then creates a new linear model
            with adjusted intercept so odds in 1-year-olds matches the specified value in the model
            when averaged across the population
            """
            def get_odds_wasting(agegp):
                """
                This function will calculate the WHZ scores by categories and return the odds of wasting
                :param agegp: age grouped in months
                :return:
                """
                # generate random numbers from N(meean, sd)
                baseline_WHZ_prevalence_by_agegp = f'prev_WHZ_distribution_age_{agegp}'
                WHZ_normal_distribution = norm(loc=p[baseline_WHZ_prevalence_by_agegp][0],
                                               scale=p[baseline_WHZ_prevalence_by_agegp][1])

                # get all wasting: WHZ <-2
                probability_over_or_equal_minus2sd = WHZ_normal_distribution.sf(-2)
                probability_less_than_minus2sd = 1 - probability_over_or_equal_minus2sd

                # convert probability to odds
                base_odds_of_wasting = probability_less_than_minus2sd / (1 - probability_less_than_minus2sd)

                return base_odds_of_wasting

            def make_linear_model_wasting(agegp, intercept=get_odds_wasting(agegp=agegp)):
                return LinearModel(
                    LinearModelType.LOGISTIC,
                    intercept,  # baseline odds: get_odds_wasting(agegp=agegp)
                    Predictor('li_wealth').when(2, p['or_wasting_hhwealth_Q2'])
                        .when(3, p['or_wasting_hhwealth_Q3'])
                        .when(4, p['or_wasting_hhwealth_Q4'])
                        .when(5, p['or_wasting_hhwealth_Q5'])
                        .otherwise(1.0),
                    # Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                    #                  '& (nb_late_preterm == False) & (nb_early_preterm == False)',
                    #                  p['or_wasting_SGA_and_term']),
                    # Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                    #                  '& (nb_late_preterm == True) | (nb_early_preterm == True)',
                    #                  p['or_wasting_SGA_and_preterm']),
                    # Predictor().when('(nb_size_for_gestational_age == "average_for_gestational_age") '
                    #                  '& (nb_late_preterm == True) | (nb_early_preterm == True)',
                    #                  p['or_wasting_preterm_and_AGA'])
                )

            unscaled_lm = make_linear_model_wasting(agegp, intercept=get_odds_wasting(agegp=agegp))
            target_mean = get_odds_wasting(agegp='12_23mo')
            actual_mean = unscaled_lm.predict(df.loc[df.is_alive & (df.age_years == 1)]).mean()
            scaled_intercept = get_odds_wasting(agegp) * (target_mean / actual_mean) if \
                (target_mean != 0 and actual_mean != 0) else get_odds_wasting(agegp)
            scaled_lm = make_linear_model_wasting(agegp, intercept=scaled_intercept)
            return scaled_lm

        # the linear model returns the probability that is implied by the model prob = odds / (1 + odds)
        for agegp in ['0_5mo', '6_11mo', '12_23mo', '24_35mo', '36_47mo', '48_59mo']:
            self.prevalence_equations_by_age[agegp] = make_scaled_linear_model_wasting(agegp)

        # get the initial prevalence values for each age group using the lm equation (scaled)
        prevalence_of_wasting = pd.DataFrame(index=df.loc[df.is_alive & (df.age_exact_years < 5)].index)
        prevalence_of_wasting['0_5mo'] = self.prevalence_equations_by_age['0_5mo'] \
            .predict(df.loc[df.is_alive & (df.age_exact_years < 0.5)])
        prevalence_of_wasting['6_11mo'] = self.prevalence_equations_by_age['6_11mo'] \
            .predict(df.loc[df.is_alive & ((df.age_exact_years >= 0.5) & (df.age_exact_years < 1))])
        prevalence_of_wasting['12_23mo'] = self.prevalence_equations_by_age['12_23mo'] \
            .predict(df.loc[df.is_alive & ((df.age_exact_years >= 1) & (df.age_exact_years < 2))])
        prevalence_of_wasting['24_35mo'] = self.prevalence_equations_by_age['24_35mo'] \
            .predict(df.loc[df.is_alive & ((df.age_exact_years >= 2) & (df.age_exact_years < 3))])
        prevalence_of_wasting['36_47mo'] = self.prevalence_equations_by_age['36_47mo'] \
            .predict(df.loc[df.is_alive & ((df.age_exact_years >= 3) & (df.age_exact_years < 4))])
        prevalence_of_wasting['48_59mo'] = self.prevalence_equations_by_age['48_59mo'] \
            .predict(df.loc[df.is_alive & ((df.age_exact_years >= 4) & (df.age_exact_years < 5))])

        # -----------------------------------------------------------------------------------------------------
        # # # # # # further categorize into moderate (-3<=WHZ<-2) or severe (WHZ<-3) wasting # # # # #
        def get_prob_severe_in_overall_wasting(agegp):
            """
            This function will calculate the WHZ scores by categories and return probability of severe wasting
            for those with wasting status
            :param agegp: age grouped in months
            :return:
            """
            # generate random numbers from N(meean, sd)
            baseline_WHZ_prevalence_by_agegp = f'prev_WHZ_distribution_age_{agegp}'
            WHZ_normal_distribution = norm(loc=p[baseline_WHZ_prevalence_by_agegp][0],
                                           scale=p[baseline_WHZ_prevalence_by_agegp][1])

            # get all wasting: WHZ <-2
            probability_over_or_equal_minus2sd = WHZ_normal_distribution.sf(-2)
            probability_less_than_minus2sd = 1 - probability_over_or_equal_minus2sd

            # get severe wasting zcores: WHZ <-3
            probability_over_or_equal_minus3sd = WHZ_normal_distribution.sf(-3)
            probability_less_than_minus3sd = 1 - probability_over_or_equal_minus3sd

            # get moderate wasting zcores: <=-3 WHZ <-2
            probability_between_minus3_minus2sd = \
                probability_over_or_equal_minus3sd - probability_over_or_equal_minus2sd

            # make WHZ <-2 as the 100% and get the adjusted probability of severe wasting within overall wasting
            proportion_severe_in_overall_wasting = probability_less_than_minus3sd * probability_less_than_minus2sd

            # get the probability of severe wasting
            return proportion_severe_in_overall_wasting

        # further differentiate between severe wasting and moderate wasting, and normal WHZ
        for agegp in ['0_5mo', '6_11mo', '12_23mo', '24_35mo', '36_47mo', '48_59mo']:
            wasted = self.rng.random_sample(len(prevalence_of_wasting[agegp])) < prevalence_of_wasting[agegp]
            for id in wasted[wasted].index:
                probability_of_severe = get_prob_severe_in_overall_wasting(agegp)
                wasted_category = self.rng.choice(['WHZ<-3', '-3<=WHZ<-2'],
                                                  p=[probability_of_severe, 1 - probability_of_severe])
                df.at[id, 'un_WHZ_category'] = wasted_category
            df.loc[wasted[wasted == False].index, 'un_WHZ_category'] = 'WHZ>=-2'

        # -----------------------------------------------------------------------------------------------------
        # # # # # # Give MUAC measurements categories based on WHZ and age group # # # # #
        for whz in ['WHZ<-3', '-3<=WHZ<-2']:
            index_under5_with_wasting = df.index[df.is_alive & ((df.age_exact_years >= 0.5) & (
                df.age_exact_years < 5)) & (df.un_WHZ_category == whz)]
            self.muac_cutoff_by_WHZ(idx=index_under5_with_wasting, whz=whz)

        # Give MUAC distribution for WHZ>=-2 ('well' group) ---------
        muac_distribution_in_well_group = norm(loc=p['MUAC_distribution_WHZ>=-2'][0],
                                               scale=p['MUAC_distribution_WHZ>=-2'][1])
        # get probability of MUAC <115mm
        probability_over_or_equal_115 = muac_distribution_in_well_group.sf(11.5)
        probability_over_or_equal_125 = muac_distribution_in_well_group.sf(12.5)

        prob_less_than_115 = 1 - probability_over_or_equal_115
        pro_between_115_125 = probability_over_or_equal_115 - probability_over_or_equal_125

        for id in df.index[df.is_alive & ((df.age_exact_years >= 0.5) & (
            df.age_exact_years < 5)) & (df.un_WHZ_category == 'WHZ>=-2')]:
            muac_cat = self.rng.choice(['<115mm', '115-<125mm', '>=125mm'],
                                       p=[prob_less_than_115, pro_between_115_125, probability_over_or_equal_125])
            df.at[id, 'un_am_MUAC_category'] = muac_cat

        # update nutritional oedema status
        index_under5 = df.index[df.is_alive & ((df.age_exact_years >= 0.5) & (df.age_exact_years < 5))]
        self.nutritional_oedema_present(idx=index_under5)

        # -----------------------------------------------------------------------------------------------------
        # Check clinical acute malnutrition state
        for person in df.index[df.is_alive & (
            (df.age_exact_years >= 0.5) & (df.age_exact_years < 5)) & (df.un_WHZ_category != 'WHZ>=-2')]:
            self.clinical_acute_malnutrition_state(person)

        # -----------------------------------------------------------------------------------------------------
        # # # # # Treatment coverage and cure rates at initiation # # # # #
        # inpatient care
        sam_requiring_inpatient_care = df.index[df.is_alive & (df.age_exact_years < 5) & (
            df.un_clinical_acute_malnutrition == 'SAM') & df.un_SAM_with_complications]
        recovered_complic_sam = self.rng.random_sample(len(sam_requiring_inpatient_care)) < (
            p['coverage_inpatient_care'] * p['recovery_rate_with_inpatient_care'])
        # schedule recovery, and reset properties
        for person in sam_requiring_inpatient_care[recovered_complic_sam]:
            self.sim.schedule_event(
                event=WastingRecoveryEvent(module=self, person_id=person),
                date=self.sim.date + DateOffset(days=self.rng.randint(0, 90)))  # in the next 3 months

        # outpatient care
        uncomplicated_sam = df.index[df.is_alive & (df.age_exact_years < 5) & (
            df.un_clinical_acute_malnutrition == 'SAM') & df.un_SAM_with_complications == False]
        recovered_uncompl_sam = self.rng.random_sample(len(uncomplicated_sam)) < (
            p['coverage_outpatient_therapeutic_care'] * p['recovery_rate_with_standard_RUTF'])
        for person in uncomplicated_sam[recovered_uncompl_sam]:
            self.sim.schedule_event(
                event=WastingRecoveryEvent(module=self, person_id=person),
                date=self.sim.date + DateOffset(days=self.rng.randint(0, 90)))  # in the next 3 months

        # supplementary feeding for MAM
        children_with_mam = df.index[df.is_alive & (df.age_exact_years < 5) & (
            df.un_clinical_acute_malnutrition == 'MAM') & df.un_SAM_with_complications == False]
        recovered_mam = self.rng.random_sample(len(children_with_mam)) < (
            p['coverage_supplementary_feeding_program'] * p['recovery_rate_with_CSB++'])
        for person in children_with_mam[recovered_mam]:
            self.sim.schedule_event(
                event=WastingRecoveryEvent(module=self, person_id=person),
                date=self.sim.date + DateOffset(days=self.rng.randint(0, 90)))  # in the next 3 months

    def initialise_simulation(self, sim):
        """Prepares for simulation:
        * Schedules the main polling event
        * Schedules the main logging event
        * Establishes the incidence linear models and other data structures using the parameters that have been read-in
        * Store the consumables that are required in each of the HSI
        """
        df = self.sim.population.props
        p = self.parameters

        event = WastingPollingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=6))

        event = WastingLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=12))

        # Get DALY weights
        get_daly_weight = self.sim.modules['HealthBurden'].get_daly_weight
        if 'HealthBurden' in self.sim.modules.keys():
            # self.daly_wts['MAM_w/o_oedema'] = get_daly_weight(sequlae_code=460)  ## no value given
            self.daly_wts['MAM_with_oedema'] = get_daly_weight(sequlae_code=461)
            self.daly_wts['SAM_w/o_oedema'] = get_daly_weight(sequlae_code=462)
            self.daly_wts['SAM_with_oedema'] = get_daly_weight(sequlae_code=463)

        # --------------------------------------------------------------------------------------------
        # Make a linear model equation that govern the probability that a person becomes wasted WHZ<-2
        def make_scaled_lm_wasting_incidence():
            """
            Makes the unscaled linear model with default intercept of 1. Calculates the mean incidents rate for
            1-year-olds and then creates a new linear model with adjusted intercept so incidents in 1-year-olds
            matches the specified value in the model when averaged across the population
            """
            def make_lm_wasting_incidence(intercept=1.0):
                return LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    intercept,
                    Predictor('age_exact_years')
                        .when('<0.5', p['base_inc_rate_wasting_by_agegp'][0])
                        .when('.between(0.5,0.9999)', p['base_inc_rate_wasting_by_agegp'][1]),
                    Predictor('age_years')
                        .when('.between(1,1)', p['base_inc_rate_wasting_by_agegp'][2])
                        .when('.between(2,2)', p['base_inc_rate_wasting_by_agegp'][3])
                        .when('.between(3,3)', p['base_inc_rate_wasting_by_agegp'][4])
                        .when('.between(4,4)', p['base_inc_rate_wasting_by_agegp'][5])
                        .otherwise(0.0),
                    # Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                    #                  '& (nb_late_preterm == False) & (nb_early_preterm == False)',
                    #                  p['rr_wasting_SGA_and_term']),
                    # Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                    #                  '& (nb_late_preterm == True) | (nb_early_preterm == True)',
                    #                  p['rr_wasting_SGA_and_preterm']),
                    # Predictor().when('(nb_size_for_gestational_age == "average_for_gestational_age") '
                    #                  '& (nb_late_preterm == True) | (nb_early_preterm == True)',
                    #                  p['rr_wasting_preterm_and_AGA'])
                )

            unscaled_lm = make_lm_wasting_incidence()
            target_mean = p[f'base_inc_rate_wasting_by_agegp'][2]
            actual_mean = unscaled_lm.predict(df.loc[df.is_alive & (df.age_years == 1) &
                                                     (df.un_WHZ_category == 'WHZ>=-2')]).mean()
            scaled_intercept = 1.0 * (target_mean / actual_mean) if \
                (target_mean != 0 and actual_mean != 0) else 1.0
            scaled_lm = make_lm_wasting_incidence(intercept=scaled_intercept)
            return scaled_lm

        self.wasting_incidence_equation = make_scaled_lm_wasting_incidence()

        # --------------------------------------------------------------------------------------------
        # Linear model for the probability of progression to severe wasting
        # (natural history only, no interventions)
        self.severe_wasting_progression_equation = \
            LinearModel(LinearModelType.MULTIPLICATIVE,
                        1.0,
                        Predictor('age_exact_years')
                        .when('<0.5', p['progression_severe_wasting_by_agegp'][0])
                        .when('.between(0.5,0.9999)', p['progression_severe_wasting_by_agegp'][1]),
                        Predictor('age_years')
                        .when('.between(1,1)', p['progression_severe_wasting_by_agegp'][2])
                        .when('.between(2,2)', p['progression_severe_wasting_by_agegp'][3])
                        .when('.between(3,3)', p['progression_severe_wasting_by_agegp'][4])
                        .when('.between(4,4)', p['progression_severe_wasting_by_agegp'][5])
                        .otherwise(0.0),
                        # Predictor('un_am_treatment_type').when('soy_RUSF', 1 - p['recovery_rate_with_soy_RUSF'])
                        # .when('CSB++', 1 - p['recovery_rate_with_CSB++']).otherwise(0.0),
                        # Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                        #                  '& (nb_late_preterm == False) & (nb_early_preterm == False)',
                        #                  p['rr_progress_severe_wasting_SGA_and_term']),
                        # Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                        #                  '& (nb_late_preterm == True) | (nb_early_preterm == True)',
                        #                  p['rr_progress_severe_wasting_SGA_and_preterm']),
                        # Predictor().when('(nb_size_for_gestational_age == "average_for_gestational_age") '
                        #                  '& (nb_late_preterm == True) | (nb_early_preterm == True)',
                        #                  p['rr_progress_severe_wasting_preterm_and_AGA'])
                        )

        # --------------------------------------------------------------------------------------------
        # Linear model for the probability of recovery based on interventions
        self.acute_malnutrition_recovery_based_on_interventions.update({
            'MAM':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('un_am_treatment_type').when('soy_RUSF', p['recovery_rate_with_soy_RUSF'])
                            .when('CSB++', p['recovery_rate_with_CSB++'])
                            .otherwise(0.0),
                            ),
            'SAM':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('un_am_treatment_type')
                            .when('standard_RUTF', p['recovery_rate_with_standard_RUTF'])
                            .when('inpatient_care', p['recovery_rate_with_inpatient_care'])
                            .otherwise(0.0),
                            )
        })
        # --------------------------------------------------------------------------------------------
        # Make a linear model equation of death from severe acute malnutrition
        def make_scaled_lm_wasting_death():
            """
            Makes the unscaled linear model with default intercept of 1. Calculates the mean death rate for
            1-year-olds and then creates a new linear model with adjusted intercept so incidents in 1-year-olds
            matches the specified value in the model when averaged across the population
            """
            def make_lm_wasting_death(intercept=1.0):
                return LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    intercept,
                    Predictor('un_SAM_management_facility_type')
                        .when('inpatient', p['cfr_SAM_inpatient_care'])  # proxy for complicated SAM
                        .when('outpatient', p['cfr_SAM_outpatient_care'])  # proxy for uncomplicated SAM
                        .otherwise(p['base_death_rate_untreated_SAM']),
                    Predictor().when('(un_am_bilateral_oedema == False) & '
                                     '(un_WHZ_category == "WHZ<-3") & (un_am_MUAC_category == "115-<125mm")',
                                     p['rr_SAM_death_WHZ<-3_only']),
                    Predictor().when('(un_am_bilateral_oedema == False) & '
                                     '(un_WHZ_category == "WHZ<-3") & (un_am_MUAC_category == "<115mm")',
                                     p['rr_SAM_death_both_WHZ<-3_&_MUAC<115mm']),
                    Predictor().when('(un_am_bilateral_oedema == True) & '
                                     '(un_WHZ_category == "-3<=WHZ<-2") & (un_am_MUAC_category == "<115mm")',
                                     p['rr_SAM_death_kwashiorkor_only']),
                    Predictor().when('(un_am_bilateral_oedema == True) & '
                                     '(un_WHZ_category == "WHZ<-3") & (un_am_MUAC_category == "115-<125mm")',
                                     p['rr_SAM_death_kwashiorkor_WHZ<-3_only']),
                    Predictor().when('(un_am_bilateral_oedema == True) & '
                                     '(un_WHZ_category == "WHZ<-3") & (un_am_MUAC_category == "<115mm")',
                                     p['rr_SAM_death_kwashiorkor_both_WHZ<-3_&_MUAC<115mm']),
                    Predictor().when('(un_am_bilateral_oedema == True) & '
                                     '(un_WHZ_category == "-3<=WHZ<-2") & (un_am_MUAC_category == "<115mm")',
                                     p['rr_SAM_death_kwashiorkor_MUAC<115mm_only']),
                )

            unscaled_lm = make_lm_wasting_death()
            target_mean = p[f'cfr_SAM_inpatient_care']
            actual_mean = unscaled_lm.predict(
                df.loc[df.is_alive & ((df.age_exact_years > 0.5) & (df.age_exact_years < 5)) &
                       (df.un_WHZ_category != 'WHZ>=-2') & df.un_SAM_with_complications]).mean()
            scaled_intercept = 1.0 * (target_mean / actual_mean) if \
                (target_mean != 0 and actual_mean != 0) else 1.0
            scaled_lm = make_lm_wasting_death(intercept=scaled_intercept)
            return scaled_lm

        self.wasting_death_equation = make_scaled_lm_wasting_death()

        # --------------------------------------------------------------------------------------------
        # Make a linear model equations that govern the probability of progression to severe wasting
        # TODO: LINEAR MODEL OF RECOVERY BASED ON TREATMENT

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        df.at[child_id, 'un_WHZ_category'] = 'WHZ>=-2'

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        logger.debug(key='message',
                     data=f'This is Wasting, being alerted about a health system interaction for person'
                          f'{person_id} and treatment {treatment_id}')

    def report_daly_values(self):
        """
        This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        experienced by persons in the previous month. Only rows for alive-persons must be returned.
        The names of the series of columns is taken to be the label of the cause of this disability.
        It will be recorded by the healthburden module as <ModuleName>_<Cause>.
        """
        df = self.sim.population.props

        total_daly_values = pd.Series(data=0.0, index=df.index[df.is_alive])
        total_daly_values.loc[df.is_alive & (df.un_WHZ_category == 'WHZ<-3') &
                              (df.un_am_bilateral_oedema == True)] = self.daly_wts['SAM_with_oedema']
        total_daly_values.loc[df.is_alive & (df.un_WHZ_category == 'WHZ<-3') &
                              (df.un_am_bilateral_oedema == False)] = self.daly_wts['SAM_w/o_oedema']
        total_daly_values.loc[df.is_alive & (df.un_WHZ_category == '-3<=WHZ<-2') &
                              (df.un_am_bilateral_oedema == True)] = self.daly_wts['MAM_with_oedema']
        # total_daly_values.loc[df.is_alive & (df.un_WHZ_category == '-3<=WHZ<-2') &
        #                       (df.un_am_bilateral_oedema == False)] = self.daly_wts['MAM_w/o_oedema']

        return total_daly_values

    def wasting_clinical_symptoms(self, person_id):
        """
        assign clinical symptoms to new acute malnutrition cases
        :param person_id:
        """
        df = self.sim.population.props
        p = self.parameters
        rng = self.rng
        now = self.sim.date

        if df.at[person_id, 'un_clinical_acute_malnutrition'] != 'SAM':
            return

        # currently symptoms list is applied to all
        for symptom in self.symptoms:
            # this also schedules symptom resolution in 5 days
            self.sim.modules["SymptomManager"].change_symptom(
                person_id=person_id,
                symptom_string=symptom,
                add_or_remove="+",
                disease_module=self,
                duration_in_days=None,
            )

    def do_when_acute_malnutrition(self, person_id, hsi_event):
        """
        This is called by the a generic HSI event when acute malnutrition is checked.
        :param person_id:
        :param hsi_event: The HSI event that has called this event
        :return:
        """
        df = self.sim.population.props
        p = self.parameters

        # get the clinical states
        clinical_am = df.at[person_id, 'un_clinical_acute_malnutrition']
        complications = df.at[person_id, 'un_SAM_with_complications']

        # Interventions for MAM
        if clinical_am == 'MAM':
            # Check for coverage of supplementary feeding
            if p['coverage_supplementary_feeding_program'] < self.module.rng.rand():
                # schedule HSI for supplementary feeding program for MAM
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event=HSI_supplementary_feeding_programme_for_MAM
                    (module=self,
                     person_id=person_id,
                     facility_level=hsi_event.ACCEPTED_FACILITY_LEVEL),
                    priority=0,
                    topen=self.sim.date
                )

        # Interventions for uncomplicated SAM
        if (clinical_am == 'SAM') & (complications == False):
            # Check for coverage of outpatient therapeutic care
            if p['coverage_outpatient_therapeutic_care'] < self.module.rng.rand():
                # schedule HSI for supplementary feeding program for MAM
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event=HSI_outpatient_therapeutic_programme_for_SAM
                    (module=self,
                     person_id=person_id,
                     facility_level=hsi_event.ACCEPTED_FACILITY_LEVEL),
                    priority=0,
                    topen=self.sim.date
                )

        # Interventions for complicated SAM
        if (clinical_am == 'SAM') & (complications == True):
            # Check for coverage of outpatient therapeutic care
            if p['coverage_inpatient_care'] < self.module.rng.rand():
                # schedule HSI for supplementary feeding program for MAM
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event=HSI_inpatient_care_for_complicated_SAM
                    (module=self,
                     person_id=person_id,
                     facility_level=hsi_event.ACCEPTED_FACILITY_LEVEL),
                    priority=0,
                    topen=self.sim.date
                )

    def do_wasting_onset_poll(self, person_id):

        df = self.sim.population.props
        p = self.parameters

        df.at[person_id, 'un_ever_wasted'] = True
        df.at[person_id, 'un_last_wasting_date_of_onset'] = self.sim.date
        df.at[person_id, 'un_WHZ_category'] = '-3<=WHZ<-2'  # start as moderate wasting

        # Give MUAC measurement category for -3<=WHZ<-2, and determine the presence of bilateral oedema -----
        self.muac_cutoff_by_WHZ(idx=df.loc[[person_id]].index, whz='-3<=WHZ<-2')
        self.nutritional_oedema_present(idx=df.loc[[person_id]].index)

        # determine the clinical state of acute malnutrition, and check complications if SAM
        self.clinical_acute_malnutrition_state(person_id)


class WastingPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that determines new cases of wasting (WHZ<-2) to the under-5 population,
    and schedules individual incident cases to represent onset.
    It determines those who will progress to severe wasting (WHZ<-3) and schedules the event to update on properties.
    It also schedules death event.
    """

    def __init__(self, module):
        """schedule to run every 6 months
        :param module: the module that created this event
        """
        self.repeat_months = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat_months))
        assert isinstance(module, Wasting)

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = m.rng
        p = m.parameters

        days_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'D')

        # # # # # # # # # # # # # # # # # # # # # INCIDENCE # # # # # # # # # # # # # # # # # # # # #

        # Determine who will be onset with wasting among those who are not currently wasted -------------
        incidence_of_wasting = self.module.wasting_incidence_equation.predict(
            df.loc[df.is_alive & (df.age_exact_years < 5) & (df.un_WHZ_category == 'WHZ>=-2')])
        wasted = rng.random_sample(len(incidence_of_wasting)) < incidence_of_wasting

        # determine the time of onset and other disease characteristics for each individual
        for person_id in wasted[wasted].index:
            # Allocate a date of onset for wasting episode
            # date_onset = self.sim.date + DateOffset(days=rng.randint(0, 30))  # new cases all occur in the same month
            # # Create the event for the onset of wasting
            # self.sim.schedule_event(
            #     event=WastingOnsetEvent(module=self.module,
            #                             person_id=person_id), date=date_onset)
            self.module.do_wasting_onset_poll(person_id)

        # # # # # # # # # # # # # # # # # # # # # PROGRESS TO SEVERE # # # # # # # # # # # # # # # # # # # # #

        # Determine those that will progress to severe wasting ( WHZ<-3) -----------------------------------
        progression_severe_wasting = m.severe_wasting_progression_equation.predict(
            df.loc[df.is_alive & (df.age_exact_years < 5) & (df.un_WHZ_category == '-3<=WHZ<-2')])
        severely_wasted = rng.random_sample(len(progression_severe_wasting)) < progression_severe_wasting

        # determine those individuals who will progress to severe wasting and time of progression
        for person in severely_wasted[severely_wasted].index:
            # for wasting cases in current polling event
            if self.sim.date <= df.at[person, 'un_last_wasting_date_of_onset'] < \
                self.sim.date + DateOffset(days=days_until_next_polling_event):
                # schedule severe wasting WHZ<-3 onset
                self.sim.schedule_event(
                    event=ProgressionSevereWastingEvent(module=self.module, person_id=person),
                    date=self.module.date_of_outcome_for_untreated_am(person_id=person, am_severity='MAM'))
            # for wasting cases from previous polling events
            if df.at[person, 'un_last_wasting_date_of_onset'] < self.sim.date:
                # schedule severe wasting WHZ<-3 onset to be today
                self.sim.schedule_event(
                    event=ProgressionSevereWastingEvent(module=self.module, person_id=person), date=self.sim.date)

        # # # # # # # # # # # # # # # # # # # # # RECOVERY FROM MAM # # # # # # # # # # # # # # # # # # # # #
        # MAM = moderate wasting (-3<=WHZ<-2) and/or MUAC between 115-125mm, no nutritional oedema

        # get index of MAM cases, exclude SAM cases with MUAC<115mm and oedema
        mam_cases = severely_wasted[severely_wasted==False].index.intersection(
            df.index[(df.un_am_MUAC_category != '<115mm') & (df.un_am_bilateral_oedema == False)])

        # moderate wasting not progressed to severe, schedule recovery
        for person in mam_cases:
            # for wasting cases in current polling event
            if self.sim.date <= df.at[person, 'un_last_wasting_date_of_onset'] < \
                self.sim.date + DateOffset(days=days_until_next_polling_event):
                self.sim.schedule_event(
                    event=WastingRecoveryEvent(module=self.module, person_id=person),
                    date=self.module.date_of_outcome_for_untreated_am(person_id=person, am_severity='MAM'))
            # for wasting cases from previous polling events
            if df.at[person, 'un_last_wasting_date_of_onset'] < self.sim.date:
                # schedule recovery to be today
                self.sim.schedule_event(
                    event=WastingRecoveryEvent(module=self.module, person_id=person), date=self.sim.date)

        # # # # # # # # # # # # # # # # # # # # # DEATH # # # # # # # # # # # # # # # # # # # # #

        # Determine those that will die -----------------------------------------------
        prob_death = m.wasting_death_equation.predict(df.loc[df.is_alive & (df.age_exact_years < 5) &
                                                             (df.un_WHZ_category != 'WHZ>=-2')])
        will_die = rng.random_sample(len(prob_death)) < prob_death

        # schedule death date
        for person in will_die[will_die]:
            # for wasting cases in current polling event
            if self.sim.date <= df.at[person, 'un_last_wasting_date_of_onset'] < \
                DateOffset(days=days_until_next_polling_event):
                # schedule death
                self.sim.schedule_event(
                    event=SevereWastingDeathEvent(module=self.module, person_id=person),
                    date=self.module.date_of_outcome_for_untreated_am(person_id=person, am_severity='SAM'))
            # for wasting cases from previous polling events
            if df.at[person, 'un_last_wasting_date_of_onset'] < self.sim.date:
                # schedule death for today
                self.sim.schedule_event(
                    event=SevereWastingDeathEvent(module=self.module, person_id=person), date=self.sim.date)

        # # # # # # # # # # # # # # # # # # # # # RECOVERY FROM SAM # # # # # # # # # # # # # # # # # # # # #
        # SAM = severe wasting (WHZ<-3) and/or MUAC <115mm, or nutritional oedema

        # get index of SAM, include oedema cases without severe wasting
        sam_by_muac_or_oedema = df.loc[(df.un_am_MUAC_category != '>=125mm') | df.un_am_bilateral_oedema].index
        sam_cases = will_die[will_die==False].index.intersection(sam_by_muac_or_oedema)

        # schedule recovery for those not scheduled to die
        for person in sam_cases:
            # for wasting cases in current polling event
            if self.sim.date < df.at[person, 'un_last_wasting_date_of_onset'] < \
                DateOffset(days=days_until_next_polling_event):
                # schedule recovery date
                self.sim.schedule_event(
                    event=WastingRecoveryEvent(module=self.module, person_id=person),
                    date=self.module.date_of_outcome_for_untreated_am(person_id=person, am_severity='SAM'))
            # for wasting cases from previous polling events
            if df.at[person, 'un_last_wasting_date_of_onset'] < self.sim.date:
                # schedule recovery for today
                self.sim.schedule_event(
                    event=WastingRecoveryEvent(module=self.module, person_id=person), date=self.sim.date)


class WastingOnsetEvent(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of wasting (WHZ <-2).
     * Updates all properties so that they pertain to this current episode of wasting
     * Imposes the symptoms
     * Schedules relevant natural history event {(ProgressionSAMEvent) and
       (either WastingRecoveryEvent or WastingDeathEvent)}
    """
    # TODO: UPDATE ABOVE DESCRIPTION

    AGE_GROUPS = {0: '0y', 1: '1y', 2: '2y', 3: '3y', 4: '4y'}

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        p = m.parameters
        rng = m.rng

        df.at[person_id, 'un_ever_wasted'] = True
        df.at[person_id, 'un_last_wasting_date_of_onset'] = self.sim.date
        df.at[person_id, 'un_WHZ_category'] = '-3<=WHZ<-2'  # start as moderate wasting

        # Give MUAC measurement category for -3<=WHZ<-2, and determine the presence of bilateral oedema -----
        self.module.muac_cutoff_by_WHZ(idx=df.loc[[person_id]].index, whz='-3<=WHZ<-2')
        self.module.nutritional_oedema_present(idx=df.loc[[person_id]].index)

        # determine the clinical state of acute malnutrition, and check complications if SAM
        self.module.clinical_acute_malnutrition_state(person_id)

        # -------------------------------------------------------------------------------------------
        # Add this incident case to the tracker
        clinical_am_state = df.at[person_id, 'un_clinical_acute_malnutrition']
        age_group = WastingOnsetEvent.AGE_GROUPS.get(df.loc[person_id].age_years, '5+y')
        if clinical_am_state != 'well':
            m.wasting_incident_case_tracker[age_group][clinical_am_state].append(self.sim.date)


class ProgressionSevereWastingEvent(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of wasting (SAM with WHZ <-3).
     * Refreshes all the properties so that they pertain to this current episode of wasting
     * Imposes the symptoms
     * Schedules relevant natural history event {(either WastingRecoveryEvent or WastingDeathEvent)}
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        rng = m.rng

        # before progression to severe wasting, check recovery of those who started/had supplementary feeding programme
        mam_recovery = self.module.acute_malnutrition_recovery_based_on_interventions['MAM'].predict(
            df.loc[[person_id]])
        if mam_recovery < rng.rand():
            # schedule recovery date
            self.sim.schedule_event(
                event=WastingRecoveryEvent(module=self.module, person_id=person_id),
                date=df.at[person_id, 'un_acute_malnutrition_tx_start_date'] + DateOffset(weeks=3))

        # continue with progression to severe if not recovered
        else:
            # update properties
            df.at[person_id, 'un_WHZ_category'] = 'WHZ<-3'

            # Give MUAC measurement category for WHZ<-3, and determine the presence of bilateral oedema ------------
            self.module.muac_cutoff_by_WHZ(idx=df.loc[[person_id]].index, whz='WHZ<-3')
            self.module.nutritional_oedema_present(idx=df.loc[[person_id]].index)

            # determine the clinical state of acute malnutrition, and check complications if SAM
            self.module.clinical_acute_malnutrition_state(person_id)

            # update clinical symptoms for severe wasting
            self.module.wasting_clinical_symptoms(person_id=person_id)

            # -------------------------------------------------------------------------------------------
            # Add this incident case to the tracker
            clinical_am_state = df.at[person_id, 'un_clinical_acute_malnutrition']
            age_group = WastingOnsetEvent.AGE_GROUPS.get(df.loc[person_id].age_years, '5+y')
            if clinical_am_state != 'well':
                m.wasting_incident_case_tracker[age_group][clinical_am_state].append(self.sim.date)


class SevereWastingDeathEvent(Event, IndividualScopeEventMixin):
    """
    This event applies the death function
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        p = m.parameters
        rng = m.rng

        # Before scheduling death, check recovery of those who had outpatient/inpatient care
        sam_recovery = self.module.acute_malnutrition_recovery_based_on_interventions['SAM'].predict(
            df.loc[[person_id]])
        if sam_recovery < rng.rand():
            # schedule recovery date
            self.sim.schedule_event(
                event=WastingRecoveryEvent(module=self.module, person_id=person_id),
                date=df.at[person_id, 'un_acute_malnutrition_tx_start_date'] + DateOffset(weeks=3))

        else:
            # Implement the death:
            self.sim.schedule_event(
                demography.InstantaneousDeath(
                    self.module,
                    person_id,
                    cause='Wasting'
                ), self.sim.date)


class WastingRecoveryEvent(Event, IndividualScopeEventMixin):
    """
    This event sets the properties back to normal state
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module

        if not df.at[person_id, 'is_alive']:
            return

        df.at[person_id, 'un_WHZ_category'] = 'WHZ>=-2'  # not undernourished
        df.at[person_id, 'un_clinical_acute_malnutrition'] = 'well'
        df.at[person_id, 'un_wasting_death_date'] = pd.NaT
        df.at[person_id, 'un_am_bilateral_oedema'] = False
        df.at[person_id, 'un_am_MUAC_category'] = '>=125mm'
        df.at[person_id, 'un_SAM_with_complications'] = False

        # this will clear all wasting symptoms
        self.sim.modules["SymptomManager"].clear_symptoms(
            person_id=person_id, disease_module=self.module
        )


class HSI_supplementary_feeding_programme_for_MAM(HSI_Event, IndividualScopeEventMixin):
    """
    This is the supplementary feeding programme for MAM without complications
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Wasting)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'supplementary_feeding_programme_for_MAM'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props

        # Stop the person from dying of acute malnutrition (if they were going to die)
        if not df.at[person_id, 'is_alive']:
            return

        # Do here whatever happens to an individual during this health system interaction event
        # ~~~~~~~~~~~~~~~~~~~~~~
        # Make request for some consumables
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        # whole package of interventions
        pkg_code_mam = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Management of moderate acute malnutrition (children)',
                            'Intervention_Pkg_Code'])[0]  # This package includes only CSB(or supercereal or CSB++)
        # individual items
        item_code1 = pd.unique(
            consumables.loc[consumables['Items'] == 'Corn Soya Blend (or Supercereal - CSB++)', 'Item_Code'])[0]

        consumables_needed = {'Intervention_Package_Code': {pkg_code_mam: 1}, 'Item_Code': {item_code1: 1}}

        # check availability of consumables
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)
        # answer comes back in the same format, but with quantities replaced with bools indicating availability
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_mam]:
            logger.debug(key='debug', data='PkgCode1 is available, so use it.')
            # Log that the treatment is provided:
            df.at[person_id, 'un_acute_malnutrition_tx_start_date'] = self.sim.date
            df.at[person_id, 'un_am_treatment_type'] = 'CSB++'

        else:
            logger.debug(key='debug', data="PkgCode1 is not available, so can't use it.")

        # --------------------------------------------------------------------------------------------------
        # check to see if all consumables returned (for demonstration purposes):
        all_available = (outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_mam]) and \
                        (outcome_of_request_for_consumables['Item_Code'][item_code1])
        # use helper function instead (for demonstration purposes)
        all_available_using_helper_function = self.get_all_consumables(
            item_codes=[item_code1],
            pkg_codes=[pkg_code_mam]
        )
        # Demonstrate equivalence
        assert all_available == all_available_using_helper_function

    def did_not_run(self):
        logger.debug("supplementary_feeding_programme_for_MAM: did not run")
        pass


class HSI_outpatient_therapeutic_programme_for_SAM(HSI_Event, IndividualScopeEventMixin):
    """
    This is the outpatient management of SAM without any medical complications
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Wasting)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint['U5Malnutr'] = 1

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'outpatient_therapeutic_programme_for_SAM'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props

        # Stop the person from dying of acute malnutrition (if they were going to die)
        if not df.at[person_id, 'is_alive']:
            return

        # Do here whatever happens to an individual during this health system interaction event
        # ~~~~~~~~~~~~~~~~~~~~~~
        # Make request for some consumables
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        # whole package of interventions
        pkg_code_sam = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Management of severe malnutrition (children)',
                            'Intervention_Pkg_Code'])[0]
        # individual items
        item_code1 = pd.unique(
            consumables.loc[consumables['Items'] == 'SAM theraputic foods', 'Item_Code'])[0]
        item_code2 = pd.unique(
            consumables.loc[consumables['Items'] == 'SAM medicines', 'Item_Code'])[0]

        consumables_needed = {'Intervention_Package_Code': {pkg_code_sam: 1}, 'Item_Code': {item_code1: 1,
                                                                                            item_code2: 1}}

        # check availability of consumables
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)
        # answer comes back in the same format, but with quantities replaced with bools indicating availability
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_sam]:
            logger.debug(key='debug', data='PkgCode1 is available, so use it.')
            # Log that the treatment is provided:
            df.at[person_id, 'un_acute_malnutrition_tx_start_date'] = self.sim.date
            df.at[person_id, 'un_am_treatment_type'] = 'standard_RUTF'
            df.at[person_id, 'un_SAM_management_facility_type'] = 'outpatient'
            assert df.at[person_id, 'un_acute_malnutrition_tx_start_date'] < df.at[person_id, 'un_wasting_death_date']
        else:
            logger.debug(key='debug', data="PkgCode1 is not available, so can't use it.")
        # --------------------------------------------------------------------------------------------------
        # check to see if all consumables returned (for demonstration purposes):
        all_available = (outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_sam]) and \
                        (outcome_of_request_for_consumables['Item_Code'][item_code1][item_code2])
        # use helper function instead (for demonstration purposes)
        all_available_using_helper_function = self.get_all_consumables(
            item_codes=[item_code1, item_code2],
            pkg_codes=[pkg_code_sam]
        )
        # Demonstrate equivalence
        assert all_available == all_available_using_helper_function

    def did_not_run(self):
        logger.debug("HSI_outpatient_therapeutic_programme_for_SAM: did not run")
        pass


class HSI_inpatient_care_for_complicated_SAM(HSI_Event, IndividualScopeEventMixin):
    """
    This is the inpatient management of SAM with medical complications
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Wasting)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint['U5Malnutr'] = 1

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'inpatient_care_for_complicated_SAM'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2
        self.ALERT_OTHER_DISEASES = []
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 7})

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props

        # Stop the person from dying of acute malnutrition (if they were going to die)
        if not df.at[person_id, 'is_alive']:
            return

        # Do here whatever happens to an individual during this health system interaction event
        # ~~~~~~~~~~~~~~~~~~~~~~
        # Make request for some consumables
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        # whole package of interventions
        pkg_code_sam = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Management of severe malnutrition (children)',
                            'Intervention_Pkg_Code'])[0]
        # individual items
        item_code1 = pd.unique(
            consumables.loc[consumables['Items'] == 'SAM theraputic foods', 'Item_Code'])[0]
        item_code2 = pd.unique(
            consumables.loc[consumables['Items'] == 'SAM medicines', 'Item_Code'])[0]

        consumables_needed = {'Intervention_Package_Code': {pkg_code_sam: 1}, 'Item_Code': {item_code1: 1,
                                                                                            item_code2: 1}}

        # check availability of consumables
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)
        # answer comes back in the same format, but with quantities replaced with bools indicating availability
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_sam]:
            logger.debug(key='debug', data='PkgCode1 is available, so use it.')
            # Log that the treatment is provided:
            df.at[person_id, 'un_acute_malnutrition_tx_start_date'] = self.sim.date
            df.at[person_id, 'un_am_treatment_type'] = 'inpatient_care'
            df.at[person_id, 'un_SAM_management_facility_type'] = 'inpatient'
            assert df.at[person_id, 'un_acute_malnutrition_tx_start_date'] < df.at[person_id, 'un_wasting_death_date']
        else:
            logger.debug(key='debug', data="PkgCode1 is not available, so can't use it.")
        # --------------------------------------------------------------------------------------------------
        # check to see if all consumables returned (for demonstration purposes):
        all_available = (outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_sam]) and \
                        (outcome_of_request_for_consumables['Item_Code'][item_code1][item_code2])
        # use helper function instead (for demonstration purposes)
        all_available_using_helper_function = self.get_all_consumables(
            item_codes=[item_code1, item_code2],
            pkg_codes=[pkg_code_sam]
        )
        # Demonstrate equivalence
        assert all_available == all_available_using_helper_function

    def did_not_run(self):
        logger.debug("HSI_inpatient_care_for_complicated_SAM: did not run")
        pass


class WastingLoggingEvent(RegularEvent, PopulationScopeEventMixin):
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
        counts_am = copy.deepcopy(self.module.wasting_incident_case_tracker_zeros)

        for age_grp in self.module.wasting_incident_case_tracker.keys():
            for state in self.module.acute_malnutrition_states:
                list_of_times = self.module.wasting_incident_case_tracker[age_grp][state]
                counts_am[age_grp][state] = len(list_of_times)
                for t in list_of_times:
                    assert self.date_last_run <= t <= self.sim.date

        logger.info(key='wasting_incidence_count', data=counts_am)

        # Reset the counters and the date_last_run
        self.module.wasting_incident_case_tracker = copy.deepcopy(self.module.wasting_incident_case_tracker_blank)
        self.date_last_run = self.sim.date

        # Wasting totals (prevalence at logging time)
        currently_wasted_age_0_5mo = (df.is_alive & (df.age_exact_years < 0.5) &
                                      (df.un_WHZ_category != 'WHZ>=-2')).sum()
        currently_wasted_age_6_11mo = (df.is_alive & ((df.age_exact_years >= 0.5) & (
            df.age_exact_years < 1)) & (df.un_WHZ_category != 'WHZ>=-2')).sum()
        currently_wasted_age_12_23mo = (df.is_alive & ((df.age_exact_years >= 1) & (
            df.age_exact_years < 2)) & (df.un_WHZ_category != 'WHZ>=-2')).sum()
        currently_wasted_age_24_35mo = (df.is_alive & ((df.age_exact_years >= 2) & (
            df.age_exact_years < 3)) & (df.un_WHZ_category != 'WHZ>=-2')).sum()
        currently_wasted_age_36_47mo = (df.is_alive & ((df.age_exact_years >= 3) & (
            df.age_exact_years < 4)) & (df.un_WHZ_category != 'WHZ>=-2')).sum()
        currently_wasted_age_48_59mo = (df.is_alive & ((df.age_exact_years >= 4) & (
            df.age_exact_years < 5)) & (df.un_WHZ_category != 'WHZ>=-2')).sum()

        currently_wasted = {'0_5mo': currently_wasted_age_0_5mo,
                            '6_11mo': currently_wasted_age_6_11mo,
                            '12_23mo': currently_wasted_age_12_23mo,
                            '24_35mo': currently_wasted_age_24_35mo,
                            '36_47mo': currently_wasted_age_36_47mo,
                            '48_59mo': currently_wasted_age_48_59mo}

        logger.info(key='wasting_prevalence_count', data=currently_wasted)

        # Proportion of ever wasted who have recovered
        ever_wasted_and_recovered = (df.is_alive & df.un_ever_wasted & (df.un_WHZ_category != 'WHZ>=-2')).sum()

        # # Numbers experiencing interventions
        # got_treatment_in_past_year = (self.sim.date - DateOffset(months=self.repeat)
        #                               ) < df.un_acute_malnutrition_tx_start_date < self.sim.date
        # proportion_got_treatment = (self.sim.date - DateOffset(months=self.repeat)
        #                             ) < df.un_last_wasting_date_of_onset < self.sim.date

