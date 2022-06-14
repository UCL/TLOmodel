"""
Road traffic injury module.

"""
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RTI(Module):
    """
    The road traffic injuries module for the TLO model, handling all injuries related to road traffic accidents.
    """

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.ASSIGN_INJURIES_AND_DALY_CHANGES = None
        self.item_codes_for_consumables_required = dict()

    INIT_DEPENDENCIES = {"SymptomManager",
                         "HealthBurden"}

    ADDITIONAL_DEPENDENCIES = {
        'Demography',
        'Lifestyle',
        'HealthSystem',
    }

    INJURY_COLUMNS = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                      'rt_injury_7', 'rt_injury_8']

    INJURY_CODES = ['none', '112', '113', '133a', '133b', '133c', '133d', '134a', '134b', '135', '1101', '1114', '211',
                    '212', '241', '2101', '2114', '291', '342', '343', '361', '363', '322', '323', '3101', '3113',
                    '412', '414', '461', '463', '453a', '453b', '441', '442', '443', '4101', '4113', '552', '553',
                    '554', '5101', '5113', '612', '673a', '673b', '674a', '674b', '675a', '675b', '676', '712a',
                    '712b', '712c', '722', '782a', '782b', '782c', '783', '7101', '7113', '811', '813do', '812',
                    '813eo', '813a', '813b', '813bo', '813c', '813co', '822a', '822b', '882', '883', '884', '8101',
                    '8113', 'P133a', 'P133b', 'P133c', 'P133d', 'P134a', 'P134b', 'P135', 'P673a', 'P673b', 'P674a',
                    'P674b', 'P675a', 'P675b', 'P676', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883', 'P884']

    SWAPPING_CODES = ['712b', '812', '3113', '4113', '5113', '7113', '8113', '813a', '813b', 'P673a', 'P673b', 'P674a',
                      'P674b', 'P675a', 'P675b', 'P676', 'P782b', 'P783', 'P883', 'P884', '813bo', '813co', '813do',
                      '813eo']

    INJURIES_REQ_IMAGING = ['112', '113', '211', '212', '412', '414', '612', '712a', '712b', '712c', '811', '812',
                            '813a', '813b', '813c', '822a', '822b', '813bo', '813co', '813do', '813eo', '673', '674',
                            '675', '676', '322', '323', '722', '342', '343', '441', '443', '453', '133', '134', '135',
                            '552', '553', '554', '342', '343', '441', '443', '453', '361', '363', '461', '463']

    FRACTURE_CODES = ['112', '113', '211', '212', '412', '414', '612', '712', '811', '812', '813']

    NO_TREATMENT_RECOVERY_TIMES_IN_DAYS = {
        '112': 49,
        '113': 49,
        '1101': 7,
        '211': 49,
        '212': 49,
        '241': 7,
        '2101': 7,
        '291': 7,
        '342': 42,
        '343': 42,
        '361': 7,
        '363': 14,
        '322': 42,
        '323': 42,
        '3101': 7,
        '3113': 56,
        '412': 35,
        '414': 365,
        '461': 7,
        '463': 14,
        '453a': 84,
        '453b': 84,
        '441': 14,
        '442': 14,
        '4101': 7,
        '552': 90,
        '553': 90,
        '554': 90,
        '5101': 7,
        '5113': 56,
        '612': 63,
        '712a': 70,
        '712b': 70,
        '722': 84,
        '7101': 7,
        '7113': 56,
        '813do': 240,
        '811': 70,
        '812': 70,
        '813eo': 240,
        '813bo': 240,
        '813co': 240,
        '822a': 60,
        '822b': 180,
        '8101': 7,
        '8113': 56
    }

    # Module parameters
    PARAMETERS = {

        'base_rate_injrti': Parameter(
            Types.REAL,
            'Base rate of RTI per year',
        ),
        'rr_injrti_age04': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 0-4 compared to base rate of RTI'
        ),
        'rr_injrti_age59': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 5-9 compared to base rate of RTI'
        ),
        'rr_injrti_age1017': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 10-17 compared to base rate of RTI'
        ),
        'rr_injrti_age1829': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 18-29 compared to base rate of RTI',
        ),
        'rr_injrti_age3039': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 30-39 compared to base rate of RTI',
        ),
        'rr_injrti_age4049': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 40-49 compared to base rate of RTI',
        ),
        'rr_injrti_age5059': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 50-59 compared to base rate of RTI',
        ),
        'rr_injrti_age6069': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 60-69 compared to base rate of RTI',
        ),
        'rr_injrti_age7079': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 70-79 compared to base rate of RTI',
        ),
        'rr_injrti_male': Parameter(
            Types.REAL,
            'risk ratio of RTI when male compared to females',
        ),
        'rr_injrti_excessalcohol': Parameter(
            Types.REAL,
            'risk ratio of RTI in those that consume excess alcohol compared to those who do not'
        ),
        'imm_death_proportion_rti': Parameter(
            Types.REAL,
            'Proportion of those involved in an RTI that die at site of accident or die before seeking medical '
            'intervention'
        ),
        'prob_bleeding_leads_to_shock': Parameter(
            Types.REAL,
            'The proportion of those with heavily bleeding injuries who go into shock'
        ),
        'prob_death_iss_less_than_9': Parameter(
            Types.REAL,
            'Proportion of people who pass away in the following month after medical treatment for injuries with an ISS'
            'score less than or equal to 9'
        ),
        'prob_death_iss_10_15': Parameter(
            Types.REAL,
            'Proportion of people who pass away in the following month after medical treatment for injuries with an ISS'
            'score from 10 to 15'
        ),
        'prob_death_iss_16_24': Parameter(
            Types.REAL,
            'Proportion of people who pass away in the following month after medical treatment for injuries with an ISS'
            'score from 16 to 24'
        ),
        'prob_death_iss_25_35': Parameter(
            Types.REAL,
            'Proportion of people who pass away in the following month after medical treatment for injuries with an ISS'
            'score from 25 to 34'
        ),
        'prob_death_iss_35_plus': Parameter(
            Types.REAL,
            'Proportion of people who pass away in the following month after medical treatment for injuries with an ISS'
            'score 35 and above'
        ),
        'prob_perm_disability_with_treatment_severe_TBI': Parameter(
            Types.REAL,
            'probability that someone with a treated severe TBI is permanently disabled'
        ),
        'prob_death_TBI_SCI_no_treatment': Parameter(
            Types.REAL,
            'probability that someone with a spinal cord injury will die without treatment'
        ),
        'prop_death_burns_no_treatment': Parameter(
            Types.REAL,
            'probability that someone with a burn injury will die without treatment'
        ),
        'prob_death_fractures_no_treatment': Parameter(
            Types.REAL,
            'probability that someone with a fracture injury will die without treatment'
        ),
        'prob_TBI_require_craniotomy': Parameter(
            Types.REAL,
            'probability that someone with a traumatic brain injury will require a craniotomy surgery'
        ),
        'prob_exploratory_laparotomy': Parameter(
            Types.REAL,
            'probability that someone with an internal organ injury will require a exploratory_laparotomy'
        ),
        'prob_depressed_skull_fracture': Parameter(
            Types.REAL,
            'Probability that a skull fracture will be depressed and therefore require surgery'
        ),
        'prob_mild_burns': Parameter(
            Types.REAL,
            'Probability that a burn within a region will result in < 10% total body surface area'
        ),
        'prob_dislocation_requires_surgery': Parameter(
            Types.REAL,
            'Probability that a dislocation will require surgery to relocate the joint.'
        ),
        'number_of_injured_body_regions_distribution': Parameter(
            Types.LIST,
            'The distribution of number of injured AIS body regions, used to decide how many injuries a person has'
        ),
        'injury_location_distribution': Parameter(
            Types.LIST,
            'The distribution of where injuries are located in the body, based on the AIS body region definition'
        ),
        # Length of stay
        'mean_los_ISS_less_than_4': Parameter(
            Types.REAL,
            'Mean length of stay for someone with an ISS score < 4'
        ),
        'sd_los_ISS_less_than_4': Parameter(
            Types.REAL,
            'Standard deviation in length of stay for someone with an ISS score < 4'
        ),
        'mean_los_ISS_4_to_8': Parameter(
            Types.REAL,
            'Mean length of stay for someone with an ISS score between 4 and 8'
        ),
        'sd_los_ISS_4_to_8': Parameter(
            Types.REAL,
            'Standard deviation in length of stay for someone with an ISS score between 4 and 8'
        ),
        'mean_los_ISS_9_to_15': Parameter(
            Types.REAL,
            'Mean length of stay for someone with an ISS score between 9 and 15'
        ),
        'sd_los_ISS_9_to_15': Parameter(
            Types.REAL,
            'Standard deviation in length of stay for someone with an ISS score between 9 and 15'
        ),
        'mean_los_ISS_16_to_24': Parameter(
            Types.REAL,
            'Mean length of stay for someone with an ISS score between 16 and 24'
        ),
        'sd_los_ISS_16_to_24': Parameter(
            Types.REAL,
            'Standard deviation in length of stay for someone with an ISS score between 16 and 24'
        ),
        'mean_los_ISS_more_than_25': Parameter(
            Types.REAL,
            'Mean length of stay for someone with an ISS score between 16 and 24'
        ),
        'sd_los_ISS_more_that_25': Parameter(
            Types.REAL,
            'Standard deviation in length of stay for someone with an ISS score between 16 and 24'
        ),
        # DALY weights
        'daly_wt_unspecified_skull_fracture': Parameter(
            Types.REAL,
            'daly_wt_unspecified_skull_fracture - code 1674'
        ),
        'daly_wt_basilar_skull_fracture': Parameter(
            Types.REAL,
            'daly_wt_basilar_skull_fracture - code 1675'
        ),
        'daly_wt_epidural_hematoma': Parameter(
            Types.REAL,
            'daly_wt_epidural_hematoma - code 1676'
        ),
        'daly_wt_subdural_hematoma': Parameter(
            Types.REAL,
            'daly_wt_subdural_hematoma - code 1677'
        ),
        'daly_wt_subarachnoid_hematoma': Parameter(
            Types.REAL,
            'daly_wt_subarachnoid_hematoma - code 1678'
        ),
        'daly_wt_brain_contusion': Parameter(
            Types.REAL,
            'daly_wt_brain_contusion - code 1679'
        ),
        'daly_wt_intraventricular_haemorrhage': Parameter(
            Types.REAL,
            'daly_wt_intraventricular_haemorrhage - code 1680'
        ),
        'daly_wt_diffuse_axonal_injury': Parameter(
            Types.REAL,
            'daly_wt_diffuse_axonal_injury - code 1681'
        ),
        'daly_wt_subgaleal_hematoma': Parameter(
            Types.REAL,
            'daly_wt_subgaleal_hematoma - code 1682'
        ),
        'daly_wt_midline_shift': Parameter(
            Types.REAL,
            'daly_wt_midline_shift - code 1683'
        ),
        'daly_wt_facial_fracture': Parameter(
            Types.REAL,
            'daly_wt_facial_fracture - code 1684'
        ),
        'daly_wt_facial_soft_tissue_injury': Parameter(
            Types.REAL,
            'daly_wt_facial_soft_tissue_injury - code 1685'
        ),
        'daly_wt_eye_injury': Parameter(
            Types.REAL,
            'daly_wt_eye_injury - code 1686'
        ),
        'daly_wt_neck_soft_tissue_injury': Parameter(
            Types.REAL,
            'daly_wt_neck_soft_tissue_injury - code 1687'
        ),
        'daly_wt_neck_internal_bleeding': Parameter(
            Types.REAL,
            'daly_wt_neck_internal_bleeding - code 1688'
        ),
        'daly_wt_neck_dislocation': Parameter(
            Types.REAL,
            'daly_wt_neck_dislocation - code 1689'
        ),
        'daly_wt_chest_wall_bruises_hematoma': Parameter(
            Types.REAL,
            'daly_wt_chest_wall_bruises_hematoma - code 1690'
        ),
        'daly_wt_hemothorax': Parameter(
            Types.REAL,
            'daly_wt_hemothorax - code 1691'
        ),
        'daly_wt_lung_contusion': Parameter(
            Types.REAL,
            'daly_wt_lung_contusion - code 1692'
        ),
        'daly_wt_diaphragm_rupture': Parameter(
            Types.REAL,
            'daly_wt_diaphragm_rupture - code 1693'
        ),
        'daly_wt_rib_fracture': Parameter(
            Types.REAL,
            'daly_wt_rib_fracture - code 1694'
        ),
        'daly_wt_flail_chest': Parameter(
            Types.REAL,
            'daly_wt_flail_chest - code 1695'
        ),
        'daly_wt_chest_wall_laceration': Parameter(
            Types.REAL,
            'daly_wt_chest_wall_laceration - code 1696'
        ),
        'daly_wt_closed_pneumothorax': Parameter(
            Types.REAL,
            'daly_wt_closed_pneumothorax - code 1697'
        ),
        'daly_wt_open_pneumothorax': Parameter(
            Types.REAL,
            'daly_wt_open_pneumothorax - code 1698'
        ),
        'daly_wt_surgical_emphysema': Parameter(
            Types.REAL,
            'daly_wt_surgical_emphysema aka subcuteal emphysema - code 1699'
        ),
        'daly_wt_abd_internal_organ_injury': Parameter(
            Types.REAL,
            'daly_wt_abd_internal_organ_injury - code 1700'
        ),
        'daly_wt_spinal_cord_lesion_neck_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_spinal_cord_lesion_neck_with_treatment - code 1701'
        ),
        'daly_wt_spinal_cord_lesion_neck_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_spinal_cord_lesion_neck_without_treatment - code 1702'
        ),
        'daly_wt_spinal_cord_lesion_below_neck_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_spinal_cord_lesion_below_neck_with_treatment - code 1703'
        ),
        'daly_wt_spinal_cord_lesion_below_neck_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_spinal_cord_lesion_below_neck_without_treatment - code 1704'
        ),
        'daly_wt_vertebrae_fracture': Parameter(
            Types.REAL,
            'daly_wt_vertebrae_fracture - code 1705'
        ),
        'daly_wt_clavicle_scapula_humerus_fracture': Parameter(
            Types.REAL,
            'daly_wt_clavicle_scapula_humerus_fracture - code 1706'
        ),
        'daly_wt_hand_wrist_fracture_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_hand_wrist_fracture_with_treatment - code 1707'
        ),
        'daly_wt_hand_wrist_fracture_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_hand_wrist_fracture_without_treatment - code 1708'
        ),
        'daly_wt_radius_ulna_fracture_short_term_with_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_radius_ulna_fracture_short_term_with_without_treatment - code 1709'
        ),
        'daly_wt_radius_ulna_fracture_long_term_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_radius_ulna_fracture_long_term_without_treatment - code 1710'
        ),
        'daly_wt_dislocated_shoulder': Parameter(
            Types.REAL,
            'daly_wt_dislocated_shoulder - code 1711'
        ),
        'daly_wt_amputated_finger': Parameter(
            Types.REAL,
            'daly_wt_amputated_finger - code 1712'
        ),
        'daly_wt_amputated_thumb': Parameter(
            Types.REAL,
            'daly_wt_amputated_thumb - code 1713'
        ),
        'daly_wt_unilateral_arm_amputation_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_unilateral_arm_amputation_with_treatment - code 1714'
        ),
        'daly_wt_unilateral_arm_amputation_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_unilateral_arm_amputation_without_treatment - code 1715'
        ),
        'daly_wt_bilateral_arm_amputation_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_bilateral_arm_amputation_with_treatment - code 1716'
        ),
        'daly_wt_bilateral_arm_amputation_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_bilateral_arm_amputation_without_treatment - code 1717'
        ),
        'daly_wt_foot_fracture_short_term_with_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_foot_fracture_short_term_with_without_treatment - code 1718'
        ),
        'daly_wt_foot_fracture_long_term_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_foot_fracture_long_term_without_treatment - code 1719'
        ),
        'daly_wt_patella_tibia_fibula_fracture_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_patella_tibia_fibula_fracture_with_treatment - code 1720'
        ),
        'daly_wt_patella_tibia_fibula_fracture_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_patella_tibia_fibula_fracture_without_treatment - code 1721'
        ),
        'daly_wt_hip_fracture_short_term_with_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_hip_fracture_short_term_with_without_treatment - code 1722'
        ),
        'daly_wt_hip_fracture_long_term_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_hip_fracture_long_term_with_treatment - code 1723'
        ),
        'daly_wt_hip_fracture_long_term_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_hip_fracture_long_term_without_treatment - code 1724'
        ),
        'daly_wt_pelvis_fracture_short_term': Parameter(
            Types.REAL,
            'daly_wt_pelvis_fracture_short_term - code 1725'
        ),
        'daly_wt_pelvis_fracture_long_term': Parameter(
            Types.REAL,
            'daly_wt_pelvis_fracture_long_term - code 1726'
        ),
        'daly_wt_femur_fracture_short_term': Parameter(
            Types.REAL,
            'daly_wt_femur_fracture_short_term - code 1727'
        ),
        'daly_wt_femur_fracture_long_term_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_femur_fracture_long_term_without_treatment - code 1728'
        ),
        'daly_wt_dislocated_hip': Parameter(
            Types.REAL,
            'daly_wt_dislocated_hip - code 1729'
        ),
        'daly_wt_dislocated_knee': Parameter(
            Types.REAL,
            'daly_wt_dislocated_knee - code 1730'
        ),
        'daly_wt_amputated_toes': Parameter(
            Types.REAL,
            'daly_wt_amputated_toes - code 1731'
        ),
        'daly_wt_unilateral_lower_limb_amputation_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_unilateral_lower_limb_amputation_with_treatment - code 1732'
        ),
        'daly_wt_unilateral_lower_limb_amputation_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_unilateral_lower_limb_amputation_without_treatment - code 1733'
        ),
        'daly_wt_bilateral_lower_limb_amputation_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_bilateral_lower_limb_amputation_with_treatment - code 1734'
        ),
        'daly_wt_bilateral_lower_limb_amputation_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_bilateral_lower_limb_amputation_without_treatment - code 1735'
        ),
        'rt_emergency_care_ISS_score_cut_off': Parameter(
            Types.INT,
            'A parameter to determine which level of injury severity corresponds to the emergency health care seeking '
            'symptom and which to the non-emergency generic injury symptom'
        ),
        'prob_death_MAIS3': Parameter(
            Types.REAL,
            'A parameter to determine the probability of death without medical intervention with a military AIS'
            'score of 3'
        ),
        'prob_death_MAIS4': Parameter(
            Types.REAL,
            'A parameter to determine the probability of death without medical intervention with a military AIS'
            'score of 4'
        ),
        'prob_death_MAIS5': Parameter(
            Types.REAL,
            'A parameter to determine the probability of death without medical intervention with a military AIS'
            'score of 5'
        ),
        'prob_death_MAIS6': Parameter(
            Types.REAL,
            'A parameter to determine the probability of death without medical intervention with a military AIS'
            'score of 6'
        ),
        'femur_fracture_skeletal_traction_mean_los': Parameter(
            Types.INT,
            'The mean length of stay for a person with a femur fracture being treated with skeletal traction'
        ),
        'other_skeletal_traction_los': Parameter(
            Types.INT,
            'The mean length of stay for a person with a non-femur fracture being treated with skeletal traction'
        ),
        'prob_foot_frac_require_cast': Parameter(
            Types.REAL,
            'The probability that a person with a foot fracture will be treated with a plaster cast'
        ),
        'prob_foot_frac_require_maj_surg': Parameter(
            Types.REAL,
            'The probability that a person with a foot fracture will be treated with a major surgery'
        ),
        'prob_foot_frac_require_min_surg': Parameter(
            Types.REAL,
            'The probability that a person with a foot fracture will be treated with a major surgery'
        ),
        'prob_foot_frac_require_amp': Parameter(
            Types.REAL,
            'The probability that a person with a foot fracture will be treated with amputation via a major surgery'
        ),
        'prob_tib_fib_frac_require_cast': Parameter(
            Types.REAL,
            'The probability that a person with a tibia/fibula fracture will be treated with a plaster cast'
        ),
        'prob_tib_fib_frac_require_maj_surg': Parameter(
            Types.REAL,
            'The probability that a person with a tibia/fibula fracture will be treated with a major surgery'
        ),
        'prob_tib_fib_frac_require_min_surg': Parameter(
            Types.REAL,
            'The probability that a person with a tibia/fibula fracture will be treated with a minor surgery'
        ),
        'prob_tib_fib_frac_require_amp': Parameter(
            Types.REAL,
            'The probability that a person with a tibia/fibula fracture will be treated with an amputation via major '
            'surgery'
        ),
        'prob_tib_fib_frac_require_traction': Parameter(
            Types.REAL,
            'The probability that a person with a tibia/fibula fracture will be treated with skeletal traction'
        ),
        'prob_femural_fracture_require_major_surgery': Parameter(
            Types.REAL,
            'The probability that a person with a femur fracture will be treated with major surgery'
        ),
        'prob_femural_fracture_require_minor_surgery': Parameter(
            Types.REAL,
            'The probability that a person with a femur fracture will be treated with minor surgery'
        ),
        'prob_femural_fracture_require_cast': Parameter(
            Types.REAL,
            'The probability that a person with a femur fracture will be treated with a plaster cast'
        ),
        'prob_femural_fracture_require_amputation': Parameter(
            Types.REAL,
            'The probability that a person with a femur fracture will be treated with amputation via major surgery'
        ),
        'prob_femural_fracture_require_traction': Parameter(
            Types.REAL,
            'The probability that a person with a femur fracture will be treated with skeletal traction'
        ),
        'prob_pelvis_fracture_traction': Parameter(
            Types.REAL,
            'The probability that a person with a pelvis fracture will be treated with skeletal traction'
        ),
        'prob_pelvis_frac_major_surgery': Parameter(
            Types.REAL,
            'The probability that a person with a pelvis fracture will be treated with major surgery'
        ),
        'prob_pelvis_frac_minor_surgery': Parameter(
            Types.REAL,
            'The probability that a person with a pelvis fracture will be treated with minor surgery'
        ),
        'prob_pelvis_frac_cast': Parameter(
            Types.REAL,
            'The probability that a person with a pelvis fracture will be treated with a cast'
        ),
        'prob_dis_hip_require_maj_surg': Parameter(
            Types.REAL,
            'The probability that a person with a dislocated hip will be treated with a major surgery'
        ),
        'prob_dis_hip_require_cast': Parameter(
            Types.REAL,
            'The probability that a person with a dislocated hip will be treated with a plaster cast'
        ),
        'prob_hip_dis_require_traction': Parameter(
            Types.REAL,
            'The probability that a person with a dislocated hip will be treated with skeletal traction'
        ),
        'hdu_cut_off_iss_score': Parameter(
            Types.INT,
            'The ISS score used as a criteria to admit patients to the HDU/ICU units'
        ),
        'mean_icu_days': Parameter(
            Types.REAL,
            'The mean length of stay in the ICUfor those without TBI'
        ),
        'sd_icu_days': Parameter(
            Types.REAL,
            'The standard deviation in length of stay in the ICU for those without TBI'
        ),
        'mean_tbi_icu_days': Parameter(
            Types.REAL,
            'The mean length of stay in the ICU for those with TBI'
        ),
        'sd_tbi_icu_days': Parameter(
            Types.REAL,
            'The standard deviation in length of stay in the ICU for those with TBI'
        ),
        'prob_open_fracture_contaminated': Parameter(
            Types.REAL,
            'The probability that an open fracture will be contaminated'
        ),
        'allowed_interventions': Parameter(
            Types.LIST,
            'List of additional interventions that can be included when performing model analysis'
        ),
        'head_prob_112': Parameter(
            Types.REAL,
            "The probability that this person's head injury is a skull fracture"
        ),
        'head_prob_113': Parameter(
            Types.REAL,
            "The probability that this person's head injury is a basilar skull fracture"
        ),
        'head_prob_133a': Parameter(
            Types.REAL,
            "The probability that this person's head injury is a Subarachnoid hematoma"
        ),
        'head_prob_133b': Parameter(
            Types.REAL,
            "The probability that this person's head injury is a Brain contusion"
        ),
        'head_prob_133c': Parameter(
            Types.REAL,
            "The probability that this person's head injury is an Intraventricular haemorrhage"
        ),
        'head_prob_133d': Parameter(
            Types.REAL,
            "The probability that this person's head injury is a Subgaleal hematoma"
        ),
        'head_prob_134a': Parameter(
            Types.REAL,
            "The probability that this person's head injury is an Epidural hematoma"
        ),
        'head_prob_134b': Parameter(
            Types.REAL,
            "The probability that this person's head injury is a Subdural hematoma"
        ),
        'head_prob_135': Parameter(
            Types.REAL,
            "The probability that this person's head injury is a Diffuse axonal injury/midline shift"
        ),
        'head_prob_1101': Parameter(
            Types.REAL,
            "The probability that this person's head injury is a laceration"
        ),
        'head_prob_1114': Parameter(
            Types.REAL,
            "The probability that this person's head injury is a burn"
        ),
        'face_prob_211': Parameter(
            Types.REAL,
            "The probability that this person's face injury is a Facial fracture (nasal/unspecified)"
        ),
        'face_prob_212': Parameter(
            Types.REAL,
            "The probability that this person's face injury is a Facial fracture (mandible/zygomatic)"
        ),
        'face_prob_241': Parameter(
            Types.REAL,
            "The probability that this person's face injury is a soft tissue injury"
        ),
        'face_prob_2101': Parameter(
            Types.REAL,
            "The probability that this person's face injury is a laceration"
        ),
        'face_prob_2114': Parameter(
            Types.REAL,
            "The probability that this person's face injury is a burn"
        ),
        'face_prob_291': Parameter(
            Types.REAL,
            "The probability that this person's face injury is an eye injury"
        ),
        'neck_prob_3101': Parameter(
            Types.REAL,
            "The probability that this person's neck injury is a laceration"
        ),
        'neck_prob_3113': Parameter(
            Types.REAL,
            "The probability that this person's neck injury is a burn"
        ),
        'neck_prob_342': Parameter(
            Types.REAL,
            "The probability that this person's neck injury is a Soft tissue injury in neck (vertebral artery "
            "laceration)"
        ),
        'neck_prob_343': Parameter(
            Types.REAL,
            "The probability that this person's neck injury is a Soft tissue injury in neck (pharynx contusion)"
        ),
        'neck_prob_361': Parameter(
            Types.REAL,
            "The probability that this person's neck injury is a Sternomastoid m. hemorrhage/ Hemorrhage, "
            "supraclavicular triangle/Hemorrhage, posterior triangle/Anterior vertebral vessel hemorrhage/ Neck muscle "
            "hemorrhage"
        ),
        'neck_prob_363': Parameter(
            Types.REAL,
            "The probability that this person's neck injury is a Hematoma in carotid sheath/Carotid sheath hemorrhage"
        ),
        'neck_prob_322': Parameter(
            Types.REAL,
            "The probability that this person's neck injury is an Atlanto-occipital subluxation"
        ),
        'neck_prob_323': Parameter(
            Types.REAL,
            "The probability that this person's neck injury is an Atlanto-axial subluxation"
        ),
        'thorax_prob_4101': Parameter(
            Types.REAL,
            "The probability that this person's thorax injury is a laceration"
        ),
        'thorax_prob_4113': Parameter(
            Types.REAL,
            "The probability that this person's thorax injury is a burn"
        ),
        'thorax_prob_461': Parameter(
            Types.REAL,
            "The probability that this person's thorax injury is Chest wall bruises/haematoma"
        ),
        'thorax_prob_463': Parameter(
            Types.REAL,
            "The probability that this person's thorax injury is Haemothorax"
        ),
        'thorax_prob_453a': Parameter(
            Types.REAL,
            "The probability that this person's thorax injury is a Lung contusion"
        ),
        'thorax_prob_453b': Parameter(
            Types.REAL,
            "The probability that this person's thorax injury is a Diaphragm rupture"
        ),
        'thorax_prob_412': Parameter(
            Types.REAL,
            "The probability that this person's thorax injury is a rib fracture"
        ),
        'thorax_prob_414': Parameter(
            Types.REAL,
            "The probability that this person's thorax injury is flail chest"
        ),
        'thorax_prob_441': Parameter(
            Types.REAL,
            "The probability that this person's thorax injury is a Chest wall lacerations/avulsions"
        ),
        'thorax_prob_442': Parameter(
            Types.REAL,
            "The probability that this person's thorax injury is a Surgical emphysema"
        ),
        'thorax_prob_443': Parameter(
            Types.REAL,
            "The probability that this person's thorax injury is a Closed pneumothorax/ open pneumothorax"
        ),
        'abdomen_prob_5101': Parameter(
            Types.REAL,
            "The probability that this person's abdomen injury is a laceration"
        ),
        'abdomen_prob_5113': Parameter(
            Types.REAL,
            "The probability that this person's thorax injury is a burn"
        ),
        'abdomen_prob_552': Parameter(
            Types.REAL,
            "The probability that this person's thorax injury is a skull fracture"
        ),
        'abdomen_prob_553': Parameter(
            Types.REAL,
            "The probability that this person's thorax injury is an Injury to stomach/intestines/colon"
        ),
        'abdomen_prob_554': Parameter(
            Types.REAL,
            "The probability that this person's thorax injury is an Injury to spleen/Urinary bladder/Liver/Urethra/"
            "Diaphragm"
        ),
        'spine_prob_612': Parameter(
            Types.REAL,
            "The probability that this person's spine injury is a vertabrae fracture"
        ),
        'spine_prob_673a': Parameter(
            Types.REAL,
            "The probability that this person's spine injury is a Spinal cord injury at neck level"
        ),
        'spine_prob_673b': Parameter(
            Types.REAL,
            "The probability that this person's spine injury is a Spinal cord injury below neck level"
        ),
        'spine_prob_674a': Parameter(
            Types.REAL,
            "The probability that this person's spine injury is a Spinal cord injury at neck level"
        ),
        'spine_prob_674b': Parameter(
            Types.REAL,
            "The probability that this person's spine injury is a Spinal cord injury below neck level"
        ),
        'spine_prob_675a': Parameter(
            Types.REAL,
            "The probability that this person's spine injury is a Spinal cord injury at neck level"
        ),
        'spine_prob_675b': Parameter(
            Types.REAL,
            "The probability that this person's spine injury is a Spinal cord injury below neck level"
        ),
        'spine_prob_676': Parameter(
            Types.REAL,
            "The probability that this person's spine injury is a Spinal cord injury at neck level"
        ),
        'upper_ex_prob_7101': Parameter(
            Types.REAL,
            "The probability that this person's upper extremity injury is a laceration"
        ),
        'upper_ex_prob_7113': Parameter(
            Types.REAL,
            "The probability that this person's upper extremity injury is a burn"
        ),
        'upper_ex_prob_712a': Parameter(
            Types.REAL,
            "The probability that this person's upper extremity injury is a Fracture to Clavicle, scapula, humerus"
        ),
        'upper_ex_prob_712b': Parameter(
            Types.REAL,
            "The probability that this person's upper extremity injury is a Fracture to Hand/wrist"
        ),
        'upper_ex_prob_712c': Parameter(
            Types.REAL,
            "The probability that this person's upper extremity injury is a Fracture to Radius/ulna"
        ),
        'upper_ex_prob_722': Parameter(
            Types.REAL,
            "The probability that this person's upper extremity injury is a dislocated shoulder"
        ),
        'upper_ex_prob_782a': Parameter(
            Types.REAL,
            "The probability that this person's upper extremity injury is an Amputated finger"
        ),
        'upper_ex_prob_782b': Parameter(
            Types.REAL,
            "The probability that this person's upper extremity injury is a Unilateral arm amputation"
        ),
        'upper_ex_prob_782c': Parameter(
            Types.REAL,
            "The probability that this person's upper extremity injury is a Thumb amputation"
        ),
        'upper_ex_prob_783': Parameter(
            Types.REAL,
            "The probability that this person's upper extremity injury is a bilateral arm amputation"
        ),
        'lower_ex_prob_8101': Parameter(
            Types.REAL,
            "The probability that this person's lower extremity injury is a laceration"
        ),
        'lower_ex_prob_8113': Parameter(
            Types.REAL,
            "The probability that this person's lower extremity injury is a burn"
        ),
        'lower_ex_prob_811': Parameter(
            Types.REAL,
            "The probability that this person's lower extremity injury is a foot fracture"
        ),
        'lower_ex_prob_813do': Parameter(
            Types.REAL,
            "The probability that this person's lower extremity injury is an open foot fracture"
        ),
        'lower_ex_prob_812': Parameter(
            Types.REAL,
            "The probability that this person's lower extremity injury is a Fracture to patella, tibia, fibula, ankle"
        ),
        'lower_ex_prob_813eo': Parameter(
            Types.REAL,
            "The probability that this person's lower extremity injury is an open Fracture to patella, tibia, fibula, "
            "ankle"
        ),
        'lower_ex_prob_813a': Parameter(
            Types.REAL,
            "The probability that this person's lower extremity injury is a Hip fracture"
        ),
        'lower_ex_prob_813b': Parameter(
            Types.REAL,
            "The probability that this person's lower extremity injury is a Pelvis fracture"
        ),
        'lower_ex_prob_813bo': Parameter(
            Types.REAL,
            "The probability that this person's lower extremity injury is an open Pelvis fracture"
        ),
        'lower_ex_prob_813c': Parameter(
            Types.REAL,
            "The probability that this person's lower extremity injury is a Femur fracture"
        ),
        'lower_ex_prob_813co': Parameter(
            Types.REAL,
            "The probability that this person's lower extremity injury is an open Femur fracture"
        ),
        'lower_ex_prob_822a': Parameter(
            Types.REAL,
            "The probability that this person's lower extremity injury is a Dislocated hip"
        ),
        'lower_ex_prob_822b': Parameter(
            Types.REAL,
            "The probability that this person's lower extremity injury is a Dislocated knee"
        ),
        'lower_ex_prob_882': Parameter(
            Types.REAL,
            "The probability that this person's lower extremity injury is an Amputation of toes"
        ),
        'lower_ex_prob_883': Parameter(
            Types.REAL,
            "The probability that this person's lower extremity injury is a Unilateral leg amputation"
        ),
        'lower_ex_prob_884': Parameter(
            Types.REAL,
            "The probability that this person's lower extremity injury is a Bilateral leg amputation "
        ),
        'blocked_interventions': Parameter(
            Types.LIST,
            "A list of interventions that are blocked in a simulation"
        ),
        'unavailable_treatment_mortality_mais_cutoff': Parameter(
            Types.INT,
            "A cut-off score above which an injury will result in additional mortality if the person has "
            "sought healthcare and not received it."
        ),
        'consider_death_no_treatment_ISS_cut_off': Parameter(
            Types.INT,
            "A cut-off score above which an injuries will be considered severe enough to cause mortality in those who"
            "have not sought care."
        )

    }

    # Define the module's parameters
    PROPERTIES = {
        'rt_road_traffic_inc': Property(Types.BOOL, 'involved in a road traffic injury'),
        'rt_inj_severity': Property(Types.CATEGORICAL,
                                    'Injury status relating to road traffic injury: none, mild, severe',
                                    categories=['none', 'mild', 'severe'],
                                    ),
        'rt_injury_1': Property(Types.CATEGORICAL, 'Codes for injury 1 from RTI', categories=INJURY_CODES),
        'rt_injury_2': Property(Types.CATEGORICAL, 'Codes for injury 2 from RTI', categories=INJURY_CODES),
        'rt_injury_3': Property(Types.CATEGORICAL, 'Codes for injury 3 from RTI', categories=INJURY_CODES),
        'rt_injury_4': Property(Types.CATEGORICAL, 'Codes for injury 4 from RTI', categories=INJURY_CODES),
        'rt_injury_5': Property(Types.CATEGORICAL, 'Codes for injury 5 from RTI', categories=INJURY_CODES),
        'rt_injury_6': Property(Types.CATEGORICAL, 'Codes for injury 6 from RTI', categories=INJURY_CODES),
        'rt_injury_7': Property(Types.CATEGORICAL, 'Codes for injury 7 from RTI', categories=INJURY_CODES),
        'rt_injury_8': Property(Types.CATEGORICAL, 'Codes for injury 8 from RTI', categories=INJURY_CODES),
        'rt_in_shock': Property(Types.BOOL, 'A property determining if this person is in shock'),
        'rt_death_from_shock': Property(Types.BOOL, 'whether this person died from shock'),
        'rt_injuries_to_cast': Property(Types.LIST, 'A list of injuries that are to be treated with casts'),
        'rt_injuries_for_minor_surgery': Property(Types.LIST, 'A list of injuries that are to be treated with a minor'
                                                              'surgery'),
        'rt_injuries_for_major_surgery': Property(Types.LIST, 'A list of injuries that are to be treated with a minor'
                                                              'surgery'),
        'rt_injuries_to_heal_with_time': Property(Types.LIST, 'A list of injuries that heal without further treatment'),
        'rt_injuries_for_open_fracture_treatment': Property(Types.LIST, 'A list of injuries that with open fracture '
                                                                        'treatment'),
        'rt_ISS_score': Property(Types.INT, 'The ISS score associated with the injuries resulting from a road traffic'
                                            'accident'),
        'rt_perm_disability': Property(Types.BOOL, 'whether the injuries from an RTI result in permanent disability'),
        'rt_polytrauma': Property(Types.BOOL, 'polytrauma from RTI'),
        'rt_imm_death': Property(Types.BOOL, 'death at scene True/False'),
        'rt_diagnosed': Property(Types.BOOL, 'Person has had their injuries diagnosed'),
        'rt_date_to_remove_daly': Property(Types.LIST, 'List of dates to remove the daly weight associated with each '
                                                       'injury'),
        'rt_post_med_death': Property(Types.BOOL, 'death in following month despite medical intervention True/False'),
        'rt_no_med_death': Property(Types.BOOL, 'death in following month without medical intervention True/False'),
        'rt_unavailable_med_death': Property(Types.BOOL, 'death in the following month without medical intervention '
                                                         'being able to be provided'),
        'rt_recovery_no_med': Property(Types.BOOL, 'recovery without medical intervention True/False'),
        'rt_disability': Property(Types.REAL, 'disability weight for current month'),
        'rt_date_inj': Property(Types.DATE, 'date of latest injury'),
        'rt_med_int': Property(Types.BOOL, 'whether this person is currently undergoing medical treatment'),
        'rt_in_icu_or_hdu': Property(Types.BOOL, 'whether this person is currently in ICU for RTI'),
        'rt_MAIS_military_score': Property(Types.INT, 'the maximum AIS-military score, used as a proxy to calculate the'
                                                      'probability of mortality without medical intervention'),
        'rt_date_death_no_med': Property(Types.DATE, 'the date which the person has is scheduled to die without medical'
                                                     'intervention'),
        'rt_debugging_DALY_wt': Property(Types.REAL, 'The true value of the DALY weight burden'),
        'rt_injuries_left_untreated': Property(Types.LIST, 'A list of injuries that have been left untreated due to a '
                                                           'blocked intervention')
    }

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,  # Disease modules: Any disease module should carry this label.
        Metadata.USES_SYMPTOMMANAGER,  # The 'Symptom Manager' recognises modules with this label.
        Metadata.USES_HEALTHSYSTEM,  # The 'HealthSystem' recognises modules with this label.
        Metadata.USES_HEALTHBURDEN  # The 'HealthBurden' module recognises modules with this label.
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'RTI_death_without_med': Cause(gbd_causes='Road injuries', label='Transport Injuries'),
        'RTI_death_with_med': Cause(gbd_causes='Road injuries', label='Transport Injuries'),
        'RTI_unavailable_med': Cause(gbd_causes='Road injuries', label='Transport Injuries'),
        'RTI_imm_death': Cause(gbd_causes='Road injuries', label='Transport Injuries'),
        'RTI_death_shock': Cause(gbd_causes='Road injuries', label='Transport Injuries'),
    }

    # Declare Causes of Death and Disability
    CAUSES_OF_DISABILITY = {
        'RTI': Cause(gbd_causes='Road injuries', label='Transport Injuries')
    }

    def read_parameters(self, data_folder):
        """ Reads the parameters used in the RTI module"""
        p = self.parameters

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_RTI.xlsx', sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)
        if "HealthBurden" in self.sim.modules:
            # get the DALY weights of the seq associated with road traffic injuries
            daly_sequlae_codes = {
                'daly_wt_unspecified_skull_fracture': 1674,
                'daly_wt_basilar_skull_fracture': 1675,
                'daly_wt_epidural_hematoma': 1676,
                'daly_wt_subdural_hematoma': 1677,
                'daly_wt_subarachnoid_hematoma': 1678,
                'daly_wt_brain_contusion': 1679,
                'daly_wt_intraventricular_haemorrhage': 1680,
                'daly_wt_diffuse_axonal_injury': 1681,
                'daly_wt_subgaleal_hematoma': 1682,
                'daly_wt_midline_shift': 1683,
                'daly_wt_facial_fracture': 1684,
                'daly_wt_facial_soft_tissue_injury': 1685,
                'daly_wt_eye_injury': 1686,
                'daly_wt_neck_soft_tissue_injury': 1687,
                'daly_wt_neck_internal_bleeding': 1688,
                'daly_wt_neck_dislocation': 1689,
                'daly_wt_chest_wall_bruises_hematoma': 1690,
                'daly_wt_hemothorax': 1691,
                'daly_wt_lung_contusion': 1692,
                'daly_wt_diaphragm_rupture': 1693,
                'daly_wt_rib_fracture': 1694,
                'daly_wt_flail_chest': 1695,
                'daly_wt_chest_wall_laceration': 1696,
                'daly_wt_closed_pneumothorax': 1697,
                'daly_wt_open_pneumothorax': 1698,
                'daly_wt_surgical_emphysema': 1699,
                'daly_wt_abd_internal_organ_injury': 1700,
                'daly_wt_spinal_cord_lesion_neck_with_treatment': 1701,
                'daly_wt_spinal_cord_lesion_neck_without_treatment': 1702,
                'daly_wt_spinal_cord_lesion_below_neck_with_treatment': 1703,
                'daly_wt_spinal_cord_lesion_below_neck_without_treatment': 1704,
                'daly_wt_vertebrae_fracture': 1705,
                'daly_wt_clavicle_scapula_humerus_fracture': 1706,
                'daly_wt_hand_wrist_fracture_with_treatment': 1707,
                'daly_wt_hand_wrist_fracture_without_treatment': 1708,
                'daly_wt_radius_ulna_fracture_short_term_with_without_treatment': 1709,
                'daly_wt_radius_ulna_fracture_long_term_without_treatment': 1710,
                'daly_wt_dislocated_shoulder': 1711,
                'daly_wt_amputated_finger': 1712,
                'daly_wt_amputated_thumb': 1713,
                'daly_wt_unilateral_arm_amputation_with_treatment': 1714,
                'daly_wt_unilateral_arm_amputation_without_treatment': 1715,
                'daly_wt_bilateral_arm_amputation_with_treatment': 1716,
                'daly_wt_bilateral_arm_amputation_without_treatment': 1717,
                'daly_wt_foot_fracture_short_term_with_without_treatment': 1718,
                'daly_wt_foot_fracture_long_term_without_treatment': 1719,
                'daly_wt_patella_tibia_fibula_fracture_with_treatment': 1720,
                'daly_wt_patella_tibia_fibula_fracture_without_treatment': 1721,
                'daly_wt_hip_fracture_short_term_with_without_treatment': 1722,
                'daly_wt_hip_fracture_long_term_with_treatment': 1723,
                'daly_wt_hip_fracture_long_term_without_treatment': 1724,
                'daly_wt_pelvis_fracture_short_term': 1725,
                'daly_wt_pelvis_fracture_long_term': 1726,
                'daly_wt_femur_fracture_short_term': 1727,
                'daly_wt_femur_fracture_long_term_without_treatment': 1728,
                'daly_wt_dislocated_hip': 1729,
                'daly_wt_dislocated_knee': 1730,
                'daly_wt_amputated_toes': 1731,
                'daly_wt_unilateral_lower_limb_amputation_with_treatment': 1732,
                'daly_wt_unilateral_lower_limb_amputation_without_treatment': 1733,
                'daly_wt_bilateral_lower_limb_amputation_with_treatment': 1734,
                'daly_wt_bilateral_lower_limb_amputation_without_treatment': 1735,
                'daly_wt_burns_greater_than_20_percent_body_area': 1736,
                'daly_wt_burns_less_than_20_percent_body_area_with_treatment': 1737,
                'daly_wt_burns_less_than_20_percent_body_area_without_treatment': 1738,
            }

            hb = self.sim.modules["HealthBurden"]
            for key, value in daly_sequlae_codes.items():
                p[key] = hb.get_daly_weight(sequlae_code=value)

        # ================== Test the parameter distributions to see whether they sum to roughly one ===============
        # test the distribution of the number of injured body regions
        assert 0.9999 < sum(p['number_of_injured_body_regions_distribution'][1]) < 1.0001, \
            "The number of injured body region distribution doesn't sum to one"
        # test the injury location distribution
        assert 0.9999 < sum(p['injury_location_distribution'][1]) < 1.0001, \
            "The injured body region distribution doesn't sum to one"
        # test the distributions to assign injuries to certain body regions
        # get the first characters of the parameter names
        body_part_strings = ['head_prob_', 'face_prob_', 'neck_prob_', 'thorax_prob_', 'abdomen_prob_',
                             'spine_prob_', 'upper_ex_prob_', 'lower_ex_prob_']
        # iterate over each body part, check the probabilities add to one
        for body_part in body_part_strings:
            probabilities_to_assign_injuries = [val for key, val in p.items() if body_part in key]
            sum_probabilities = sum(probabilities_to_assign_injuries)
            assert (sum_probabilities % 1 < 0.0001) or (sum_probabilities % 1 > 0.9999), "The probabilities" \
                                                                                         "chosen for assigning" \
                                                                                         "injuries don't" \
                                                                                         "sum to one"
        # Check all other probabilities are between 0 and 1
        probabilities = [val for key, val in p.items() if 'prob_' in key]
        for probability in probabilities:
            assert 0 <= probability <= 1, "Probability is not a feasible value"
        # create a generic severe trauma symptom, which forces people into the health system
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(
                name='severe_trauma',
                emergency_in_adults=True,
                emergency_in_children=True
            )
        )
        # create an injury lookup table to handle all assigning injuries/daly weights and daly weight changes. The table
        # is writted in the following format: [[1], 2, 3, 4]. [1] contains information used in assigning injuries e.g.
        # probability of injury occuring followed by information used in logging, specifically injury location, injury
        # category and injury severity. 2 contains the daly weight initially assigned to people who have this injury.
        # 3 contains any potential changes to the persons health burden upon treatment. 4 contains the daly weight to
        # remove once an injury is healed.

        self.ASSIGN_INJURIES_AND_DALY_CHANGES = {
            'none': [0, 0, 0, 0],
            # injuries to the head
            '112': [[p['head_prob_112'], 1, 1, 2, 3], p['daly_wt_unspecified_skull_fracture'], 0,
                    - p['daly_wt_unspecified_skull_fracture']],
            '113': [[p['head_prob_113'], 1, 1, 3, 4], p['daly_wt_basilar_skull_fracture'], 0,
                    - p['daly_wt_basilar_skull_fracture']],
            '133a': [[p['head_prob_133a'], 1, 3, 3, 4], p['daly_wt_subarachnoid_hematoma'], 0,
                     - p['daly_wt_subarachnoid_hematoma']],
            '133b': [[p['head_prob_133b'], 1, 3, 3, 4], p['daly_wt_brain_contusion'], 0,
                     - p['daly_wt_brain_contusion']],
            '133c': [[p['head_prob_133c'], 1, 3, 3, 4], p['daly_wt_intraventricular_haemorrhage'], 0,
                     - p['daly_wt_intraventricular_haemorrhage']],
            '133d': [[p['head_prob_133d'], 1, 3, 3, 4], p['daly_wt_subgaleal_hematoma'], 0,
                     - p['daly_wt_subgaleal_hematoma']],
            '134a': [[p['head_prob_134a'], 1, 3, 4, 5], p['daly_wt_epidural_hematoma'], 0,
                     - p['daly_wt_epidural_hematoma']],
            '134b': [[p['head_prob_134b'], 1, 3, 4, 5], p['daly_wt_subdural_hematoma'], 0,
                     - p['daly_wt_subdural_hematoma']],
            '135': [[p['head_prob_135'], 1, 3, 5, 6], p['daly_wt_diffuse_axonal_injury'], 0,
                    - p['daly_wt_diffuse_axonal_injury']],
            '1101': [[p['head_prob_1101'], 1, 10, 1, 2], p['daly_wt_facial_soft_tissue_injury'], 0,
                     - p['daly_wt_facial_soft_tissue_injury']],
            '1114': [[p['head_prob_1114'], 1, 11, 4, 5], p['daly_wt_burns_greater_than_20_percent_body_area'], 0,
                     - p['daly_wt_burns_greater_than_20_percent_body_area']],
            # injuries to the face
            '211': [[p['face_prob_211'], 2, 1, 1, 2], p['daly_wt_facial_fracture'], 0, - p['daly_wt_facial_fracture']],
            '212': [[p['face_prob_212'], 2, 1, 2, 3], p['daly_wt_facial_fracture'], 0, - p['daly_wt_facial_fracture']],
            '241': [[p['face_prob_241'], 2, 4, 1, 2], p['daly_wt_facial_soft_tissue_injury'], 0,
                    - p['daly_wt_facial_soft_tissue_injury']],
            '2101': [[p['face_prob_2101'], 2, 10, 1, 2], p['daly_wt_facial_soft_tissue_injury'], 0,
                     - p['daly_wt_facial_soft_tissue_injury']],
            '2114': [[p['face_prob_2114'], 2, 11, 4, 5], p['daly_wt_burns_greater_than_20_percent_body_area'], 0,
                     - p['daly_wt_burns_greater_than_20_percent_body_area']],
            '291': [[p['face_prob_291'], 2, 9, 1, 2], p['daly_wt_eye_injury'], 0, - p['daly_wt_eye_injury']],
            # injuries to the neck
            '3101': [[p['neck_prob_3101'], 3, 10, 1, 2], p['daly_wt_facial_soft_tissue_injury'], 0,
                     - p['daly_wt_facial_soft_tissue_injury']],
            '3113': [[p['neck_prob_3113'], 3, 11, 3, 4],
                     p['daly_wt_burns_less_than_20_percent_body_area_without_treatment'],
                     - p['daly_wt_burns_less_than_20_percent_body_area_without_treatment'] +
                     p['daly_wt_burns_less_than_20_percent_body_area_with_treatment'],
                     - p['daly_wt_burns_less_than_20_percent_body_area_with_treatment']],
            '342': [[p['neck_prob_342'], 3, 4, 2, 3], p['daly_wt_neck_internal_bleeding'], 0,
                    - p['daly_wt_neck_internal_bleeding']],
            '343': [[p['neck_prob_343'], 3, 4, 3, 4], p['daly_wt_neck_internal_bleeding'], 0,
                    - p['daly_wt_neck_internal_bleeding']],
            '361': [[p['neck_prob_361'], 3, 6, 1, 2], p['daly_wt_neck_internal_bleeding'], 0,
                    - p['daly_wt_neck_internal_bleeding']],
            '363': [[p['neck_prob_363'], 3, 6, 3, 4], p['daly_wt_neck_internal_bleeding'], 0,
                    - p['daly_wt_neck_internal_bleeding']],
            '322': [[p['neck_prob_322'], 3, 2, 2, 3], p['daly_wt_neck_dislocation'], 0,
                    - p['daly_wt_neck_dislocation']],
            '323': [[p['neck_prob_323'], 3, 2, 3, 4], p['daly_wt_neck_dislocation'], 0,
                    - p['daly_wt_neck_dislocation']],
            # injuries to the chest
            '4101': [[p['thorax_prob_4101'], 4, 10, 1, 2], p['daly_wt_facial_soft_tissue_injury'], 0,
                     - p['daly_wt_facial_soft_tissue_injury']],
            '4113': [[p['thorax_prob_4113'], 4, 11, 3, 4],
                     p['daly_wt_burns_less_than_20_percent_body_area_without_treatment'],
                     - p['daly_wt_burns_less_than_20_percent_body_area_without_treatment'] +
                     p['daly_wt_burns_less_than_20_percent_body_area_with_treatment'],
                     - p['daly_wt_burns_less_than_20_percent_body_area_with_treatment']],
            '461': [[p['thorax_prob_461'], 4, 6, 1, 2], p['daly_wt_chest_wall_bruises_hematoma'], 0,
                    - p['daly_wt_chest_wall_bruises_hematoma']],
            '463': [[p['thorax_prob_463'], 4, 6, 3, 4], p['daly_wt_hemothorax'], 0, - p['daly_wt_hemothorax']],
            '453a': [[p['thorax_prob_453a'], 4, 5, 3, 4], p['daly_wt_diaphragm_rupture'], 0,
                     - p['daly_wt_diaphragm_rupture']],
            '453b': [[p['thorax_prob_453b'], 4, 5, 3, 4], p['daly_wt_lung_contusion'], 0,
                     - p['daly_wt_lung_contusion']],
            '412': [[p['thorax_prob_412'], 4, 1, 2, 3], p['daly_wt_rib_fracture'], 0, - p['daly_wt_rib_fracture']],
            '414': [[p['thorax_prob_414'], 4, 1, 4, 5], p['daly_wt_flail_chest'], 0, - p['daly_wt_flail_chest']],
            '441': [[p['thorax_prob_441'], 4, 4, 1, 2], p['daly_wt_closed_pneumothorax'], 0,
                    - p['daly_wt_closed_pneumothorax']],
            '442': [[p['thorax_prob_442'], 4, 4, 2, 3], p['daly_wt_surgical_emphysema'], 0,
                    - p['daly_wt_surgical_emphysema']],
            '443': [[p['thorax_prob_443'], 4, 4, 3, 4], p['daly_wt_open_pneumothorax'], 0,
                    - p['daly_wt_open_pneumothorax']],
            # injuries to the abdomen
            '5101': [[p['abdomen_prob_5101'], 5, 10, 1, 2], p['daly_wt_facial_soft_tissue_injury'], 0,
                     - p['daly_wt_facial_soft_tissue_injury']],
            '5113': [[p['abdomen_prob_5113'], 5, 11, 3, 4],
                     p['daly_wt_burns_less_than_20_percent_body_area_without_treatment'],
                     - p['daly_wt_burns_less_than_20_percent_body_area_without_treatment'] +
                     p['daly_wt_burns_less_than_20_percent_body_area_with_treatment'],
                     - p['daly_wt_burns_less_than_20_percent_body_area_with_treatment']],
            '552': [[p['abdomen_prob_552'], 5, 5, 2, 3], p['daly_wt_abd_internal_organ_injury'], 0,
                    - p['daly_wt_abd_internal_organ_injury']],
            '553': [[p['abdomen_prob_553'], 5, 5, 3, 4], p['daly_wt_abd_internal_organ_injury'], 0,
                    - p['daly_wt_abd_internal_organ_injury']],
            '554': [[p['abdomen_prob_554'], 5, 5, 4, 5], p['daly_wt_abd_internal_organ_injury'], 0,
                    - p['daly_wt_abd_internal_organ_injury']],
            # injuries to the spine
            '612': [[p['spine_prob_612'], 6, 1, 2, 3], p['daly_wt_vertebrae_fracture'], 0,
                    - p['daly_wt_vertebrae_fracture']],
            '673a': [[p['spine_prob_673a'], 6, 7, 3, 4], p['daly_wt_spinal_cord_lesion_neck_without_treatment'],
                     - p['daly_wt_spinal_cord_lesion_neck_without_treatment'] +
                     p['daly_wt_spinal_cord_lesion_neck_with_treatment'], 0],
            '673b': [[p['spine_prob_673b'], 6, 7, 3, 4], p['daly_wt_spinal_cord_lesion_below_neck_without_treatment'],
                     - p['daly_wt_spinal_cord_lesion_below_neck_without_treatment'] +
                     p['daly_wt_spinal_cord_lesion_below_neck_with_treatment'], 0],
            '674a': [[p['spine_prob_674a'], 6, 7, 4, 5], p['daly_wt_spinal_cord_lesion_neck_without_treatment'],
                     - p['daly_wt_spinal_cord_lesion_neck_without_treatment'] +
                     p['daly_wt_spinal_cord_lesion_neck_with_treatment'], 0],
            '674b': [[p['spine_prob_674b'], 6, 7, 4, 5], p['daly_wt_spinal_cord_lesion_below_neck_without_treatment'],
                     - p['daly_wt_spinal_cord_lesion_below_neck_without_treatment'] +
                     p['daly_wt_spinal_cord_lesion_below_neck_with_treatment'], 0],
            '675a': [[p['spine_prob_675a'], 6, 7, 5, 6], p['daly_wt_spinal_cord_lesion_neck_without_treatment'],
                     - p['daly_wt_spinal_cord_lesion_neck_without_treatment'] +
                     p['daly_wt_spinal_cord_lesion_neck_with_treatment'], 0],
            '675b': [[p['spine_prob_675b'], 6, 7, 5, 6], p['daly_wt_spinal_cord_lesion_below_neck_without_treatment'],
                     - p['daly_wt_spinal_cord_lesion_below_neck_without_treatment'] +
                     p['daly_wt_spinal_cord_lesion_below_neck_with_treatment'], 0],
            '676': [[p['spine_prob_676'], 6, 7, 6, 6], p['daly_wt_spinal_cord_lesion_neck_without_treatment'],
                    - p['daly_wt_spinal_cord_lesion_neck_without_treatment'] +
                    p['daly_wt_spinal_cord_lesion_neck_with_treatment'], 0],
            # injuries to the upper extremities
            '7101': [[p['upper_ex_prob_7101'], 7, 10, 1, 2], p['daly_wt_facial_soft_tissue_injury'], 0,
                     - p['daly_wt_facial_soft_tissue_injury']],
            '7113': [[p['upper_ex_prob_7113'], 7, 11, 3, 4],
                     p['daly_wt_burns_less_than_20_percent_body_area_without_treatment'],
                     - p['daly_wt_burns_less_than_20_percent_body_area_without_treatment'] +
                     p['daly_wt_burns_less_than_20_percent_body_area_with_treatment'],
                     - p['daly_wt_burns_less_than_20_percent_body_area_with_treatment']],
            '712a': [[p['upper_ex_prob_712a'], 7, 1, 2, 3], p['daly_wt_clavicle_scapula_humerus_fracture'], 0,
                     - p['daly_wt_clavicle_scapula_humerus_fracture']],
            '712b': [[p['upper_ex_prob_712b'], 7, 1, 2, 3], p['daly_wt_hand_wrist_fracture_without_treatment'],
                     - p['daly_wt_hand_wrist_fracture_without_treatment'] +
                     p['daly_wt_hand_wrist_fracture_with_treatment'],
                     - p['daly_wt_hand_wrist_fracture_with_treatment']],
            '712c': [[p['upper_ex_prob_712c'], 7, 1, 2, 3],
                     p['daly_wt_radius_ulna_fracture_short_term_with_without_treatment'], 0,
                     - p['daly_wt_radius_ulna_fracture_short_term_with_without_treatment']],
            '722': [[p['upper_ex_prob_722'], 7, 2, 2, 3], p['daly_wt_dislocated_shoulder'], 0,
                    - p['daly_wt_dislocated_shoulder']],
            '782a': [[p['upper_ex_prob_782a'], 7, 8, 2, 3], p['daly_wt_amputated_finger'], 0, 0],
            '782b': [[p['upper_ex_prob_782b'], 7, 8, 2, 3], p['daly_wt_unilateral_arm_amputation_without_treatment'],
                     - p['daly_wt_unilateral_arm_amputation_without_treatment'] +
                     p['daly_wt_unilateral_arm_amputation_with_treatment'], 0],
            '782c': [[p['upper_ex_prob_782c'], 7, 8, 2, 3], p['daly_wt_amputated_thumb'], 0, 0],
            '783': [[p['upper_ex_prob_783'], 7, 8, 3, 4], p['daly_wt_bilateral_arm_amputation_without_treatment'],
                    - p['daly_wt_bilateral_arm_amputation_without_treatment'] +
                    p['daly_wt_bilateral_arm_amputation_with_treatment'], 0],
            # injuries to the lower extremities
            '8101': [[p['lower_ex_prob_8101'], 8, 10, 1, 2], p['daly_wt_facial_soft_tissue_injury'], 0,
                     - p['daly_wt_facial_soft_tissue_injury']],
            '8113': [[p['lower_ex_prob_8113'], 8, 11, 3, 4],
                     p['daly_wt_burns_less_than_20_percent_body_area_without_treatment'],
                     - p['daly_wt_burns_less_than_20_percent_body_area_without_treatment'] +
                     p['daly_wt_burns_less_than_20_percent_body_area_with_treatment'],
                     - p['daly_wt_burns_less_than_20_percent_body_area_with_treatment']],
            # foot fracture, can be open or not, open is more severe
            '811': [[p['lower_ex_prob_811'], 8, 1, 1, 2], p['daly_wt_foot_fracture_short_term_with_without_treatment'],
                    0, - p['daly_wt_foot_fracture_short_term_with_without_treatment']],
            '813do': [[p['lower_ex_prob_813do'], 8, 1, 3, 4],
                      p['daly_wt_foot_fracture_short_term_with_without_treatment']
                      + p['daly_wt_facial_soft_tissue_injury'], 0,
                      - p['daly_wt_foot_fracture_short_term_with_without_treatment'] -
                      p['daly_wt_facial_soft_tissue_injury']],
            # lower leg fracture can be open or not
            '812': [[p['lower_ex_prob_812'], 8, 1, 2, 3], p['daly_wt_patella_tibia_fibula_fracture_without_treatment'],
                    - p['daly_wt_patella_tibia_fibula_fracture_without_treatment'] +
                    p['daly_wt_patella_tibia_fibula_fracture_with_treatment'],
                    - p['daly_wt_patella_tibia_fibula_fracture_with_treatment']],
            '813eo': [[p['lower_ex_prob_813eo'], 8, 1, 3, 4],
                      p['daly_wt_patella_tibia_fibula_fracture_without_treatment']
                      + p['daly_wt_facial_soft_tissue_injury'], 0,
                      - p['daly_wt_patella_tibia_fibula_fracture_without_treatment'] -
                      p['daly_wt_facial_soft_tissue_injury']],
            '813a': [[p['lower_ex_prob_813a'], 8, 1, 3, 4], p['daly_wt_hip_fracture_short_term_with_without_treatment'],
                     - p['daly_wt_hip_fracture_short_term_with_without_treatment'] +
                     p['daly_wt_hip_fracture_long_term_with_treatment'],
                     - p['daly_wt_hip_fracture_long_term_with_treatment']],
            # pelvis fracture can be open or closed
            '813b': [[p['lower_ex_prob_813b'], 8, 1, 3, 4], p['daly_wt_pelvis_fracture_short_term'],
                     - p['daly_wt_pelvis_fracture_short_term'] + p['daly_wt_pelvis_fracture_long_term'],
                     - p['daly_wt_pelvis_fracture_long_term']],
            '813bo': [[p['lower_ex_prob_813bo'], 8, 1, 3, 4], p['daly_wt_pelvis_fracture_short_term'] +
                      p['daly_wt_facial_soft_tissue_injury'], - p['daly_wt_pelvis_fracture_short_term'] +
                      p['daly_wt_pelvis_fracture_long_term'], - p['daly_wt_pelvis_fracture_long_term'] -
                      p['daly_wt_facial_soft_tissue_injury']],
            # femur fracture can be open or closed
            '813c': [[p['lower_ex_prob_813c'], 8, 1, 3, 4], p['daly_wt_femur_fracture_short_term'], 0,
                     - p['daly_wt_femur_fracture_short_term']],
            '813co': [[p['lower_ex_prob_813co'], 8, 1, 3, 4], p['daly_wt_femur_fracture_short_term'] +
                      p['daly_wt_facial_soft_tissue_injury'], 0, - p['daly_wt_femur_fracture_short_term'] -
                      p['daly_wt_facial_soft_tissue_injury']],
            '822a': [[p['lower_ex_prob_822a'], 8, 2, 2, 3], p['daly_wt_dislocated_hip'], 0,
                     - p['daly_wt_dislocated_hip']],
            '822b': [[p['lower_ex_prob_822b'], 8, 2, 2, 3], p['daly_wt_dislocated_knee'], 0,
                     - p['daly_wt_dislocated_knee']],
            '882': [[p['lower_ex_prob_882'], 8, 8, 2, 3], p['daly_wt_amputated_toes'], 0, 0],
            '883': [[p['lower_ex_prob_883'], 8, 8, 3, 4],
                    p['daly_wt_unilateral_lower_limb_amputation_without_treatment'],
                    - p['daly_wt_unilateral_lower_limb_amputation_without_treatment'] +
                    p['daly_wt_unilateral_lower_limb_amputation_with_treatment'], 0],
            '884': [[p['lower_ex_prob_884'], 8, 8, 4, 5],
                    p['daly_wt_bilateral_lower_limb_amputation_without_treatment'],
                    - p['daly_wt_bilateral_lower_limb_amputation_without_treatment'] +
                    p['daly_wt_bilateral_lower_limb_amputation_with_treatment'], 0]
        }
        # The vast majority of the injuries should have a total change of daly weights that sum to zero, meaning that
        # a person recieves an injury and has the health burden which will eventually be removed once the injury has
        # healed. However some injuries are permanent so the person will always have some level of health burden. The
        # injury codes for permanent injuries are given below.
        permanent_injuries = ['673a', '673b', '674a', '674b', '675a', '675b', '676', '782a', '782b', '782c', '783',
                              '882', '883', '884']
        # We need to check that the changes to all other DALY weights over the course of treatment sum to zero, do so
        # using pandas, convert dictionary into a dataframe
        check_daly_change_df = pd.DataFrame(self.ASSIGN_INJURIES_AND_DALY_CHANGES)
        # drop the row of the dataframe used to assign people injuries
        check_daly_change_df = check_daly_change_df.drop([0], axis=0)
        # calculate the sum of the dataframe
        sum_check_daly_change_df = check_daly_change_df.sum()
        # find the injuries where the change in daly weights does not sum to zero
        non_zero_total_daly_change = sum_check_daly_change_df.where(sum_check_daly_change_df > 0).dropna().index
        # ensure that these injuries are the permanent injuries
        assert non_zero_total_daly_change.to_list() == permanent_injuries

    def rti_injury_diagnosis(self, person_id, the_appt_footprint):
        """
        A function used to alter the appointment footprint of the generic first appointments, based on the needs of
        the patient to be properly diagnosed. Specifically, this function will assign x-rays/ct-scans for injuries
        that require those diagnosis tools.
        :param person_id: the person in a generic appointment with an injury
        :param the_appt_footprint: the current appointment footprint to be altered
        :return: the altered appointment footprint
        """
        df = self.sim.population.props
        # Filter the dataframe by the columns the injuries are stored in
        persons_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]

        # Injuries that require x rays are: fractures, spinal cord injuries, dislocations, soft tissue injuries in neck
        # and soft tissue injury in thorax/ lung injury
        codes_requiring_xrays = ['112', '113', '211', '212', '412', '414', '612', '712a', '712b', '712c', '811', '812',
                                 '813a', '813b', '813c', '822a', '822b', '813bo', '813co', '813do', '813eo', '673',
                                 '674', '675', '676', '322', '323', '722', '342', '343', '441', '443', '453']
        # Injuries that require a ct scan are TBIs, abdominal trauma, soft tissue injury in neck, soft tissue injury in
        # thorax/ lung injury and abdominal trauma
        codes_requiring_ct_scan = ['133', '134', '135', '552', '553', '554', '342', '343', '441', '443', '453', '361',
                                   '363', '461', '463']

        def adjust_appt_footprint(_codes, _requirement):
            _, counts = self.rti_find_and_count_injuries(persons_injuries, _codes)
            if counts > 0:
                the_appt_footprint[_requirement] = 1

        adjust_appt_footprint(codes_requiring_xrays, 'DiagRadio')
        adjust_appt_footprint(codes_requiring_ct_scan, 'Tomography')

    def initialise_population(self, population):
        """Sets up the default properties used in the RTI module and applies them to the dataframe. The default state
        for the RTI module is that people haven't been involved in a road traffic accident and are therefor alive and
        healthy."""
        df = population.props
        df.loc[df.is_alive, 'rt_road_traffic_inc'] = False
        df.loc[df.is_alive, 'rt_inj_severity'] = "none"  # default: no one has been injured in a RTI
        df.loc[df.is_alive, 'rt_injury_1'] = "none"
        df.loc[df.is_alive, 'rt_injury_2'] = "none"
        df.loc[df.is_alive, 'rt_injury_3'] = "none"
        df.loc[df.is_alive, 'rt_injury_4'] = "none"
        df.loc[df.is_alive, 'rt_injury_5'] = "none"
        df.loc[df.is_alive, 'rt_injury_6'] = "none"
        df.loc[df.is_alive, 'rt_injury_7'] = "none"
        df.loc[df.is_alive, 'rt_injury_8'] = "none"
        df.loc[df.is_alive, 'rt_in_shock'] = False
        df.loc[df.is_alive, 'rt_death_from_shock'] = False
        df.loc[df.is_alive, 'rt_polytrauma'] = False
        df.loc[df.is_alive, 'rt_ISS_score'] = 0
        df.loc[df.is_alive, 'rt_perm_disability'] = False
        df.loc[df.is_alive, 'rt_imm_death'] = False  # default: no one is dead on scene of crash
        df.loc[df.is_alive, 'rt_diagnosed'] = False
        df.loc[df.is_alive, 'rt_recovery_no_med'] = False  # default: no recovery without medical intervention
        df.loc[df.is_alive, 'rt_post_med_death'] = False  # default: no death after medical intervention
        df.loc[df.is_alive, 'rt_no_med_death'] = False
        df.loc[df.is_alive, 'rt_unavailable_med_death'] = False
        df.loc[df.is_alive, 'rt_disability'] = 0  # default: no DALY
        df.loc[df.is_alive, 'rt_date_inj'] = pd.NaT
        df.loc[df.is_alive, 'rt_med_int'] = False
        df.loc[df.is_alive, 'rt_in_icu_or_hdu'] = False
        df.loc[df.is_alive, 'rt_MAIS_military_score'] = 0
        df.loc[df.is_alive, 'rt_date_death_no_med'] = pd.NaT
        df.loc[df.is_alive, 'rt_debugging_DALY_wt'] = 0
        alive_count = sum(df.is_alive)
        df.loc[df.is_alive, 'rt_date_to_remove_daly'] = pd.Series([[pd.NaT] * 8 for _ in range(alive_count)])
        df.loc[df.is_alive, 'rt_injuries_to_cast'] = pd.Series([[] for _ in range(alive_count)])
        df.loc[df.is_alive, 'rt_injuries_for_minor_surgery'] = pd.Series([[] for _ in range(alive_count)])
        df.loc[df.is_alive, 'rt_injuries_for_major_surgery'] = pd.Series([[] for _ in range(alive_count)])
        df.loc[df.is_alive, 'rt_injuries_to_heal_with_time'] = pd.Series([[] for _ in range(alive_count)])
        df.loc[df.is_alive, 'rt_injuries_for_open_fracture_treatment'] = pd.Series([[] for _ in range(alive_count)])
        df.loc[df.is_alive, 'rt_injuries_left_untreated'] = pd.Series([[] for _ in range(alive_count)])

    def initialise_simulation(self, sim):
        """At the start of the simulation we schedule a logging event, which records the relevant information
        regarding road traffic injuries in the last month.

        Afterwards, we schedule three RTI events, the first is the main RTI event which takes parts
        of the population and assigns them to be involved in road traffic injuries and providing they survived will
        begin the interaction with the healthcare system. This event runs monthly.

        The second is the begin scheduling the RTI recovery event, which looks at those in the population who have been
        injured in a road traffic accident, checking every day whether enough time has passed for their injuries to have
        healed. When the injury has healed the associated daly weight is removed.

        The final event is one which checks if this person has not sought sought care or been given care, if they
        haven't then it asks whether they should die away from their injuries
        """
        # Begin modelling road traffic injuries
        sim.schedule_event(RTIPollingEvent(self), sim.date + DateOffset(months=0))
        # Begin checking whether the persons injuries are healed
        sim.schedule_event(RTI_Recovery_Event(self), sim.date + DateOffset(months=0))
        # Begin checking whether those with untreated injuries die
        sim.schedule_event(RTI_Check_Death_No_Med(self), sim.date + DateOffset(months=0))
        # Begin logging the RTI events
        sim.schedule_event(RTI_Logging_Event(self), sim.date + DateOffset(months=1))

    def rti_do_when_diagnosed(self, person_id):
        """
        This function is called by the generic first appointments when an injured person has been diagnosed
        in A&E and needs to progress further in the health system. The injured person will then be scheduled a generic
        'medical intervention' appointment which serves three purposes. The first is to determine what treatments they
        require for their injuries and shedule those, the second is to contain them in the health care system with
        inpatient days and finally, the appointment treats injuries that heal over time without further need for
        resources in the health system.

        :param person_id: the person requesting medical care
        :return: n/a
        """
        df = self.sim.population.props
        # Check to see whether they have been sent here from A and E
        assert df.at[person_id, 'rt_diagnosed']
        # Get the relevant information about their injuries
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        # check this person is injured, search they have an injury code that isn't "none"
        _, counts = RTI.rti_find_and_count_injuries(person_injuries, RTI.INJURY_CODES[1:])
        # also test whether the regular injury symptom has been given to the person via spurious symptoms
        assert (counts > 0) or self.sim.modules['SymptomManager'].spurious_symptoms, \
            'This person has asked for medical treatment despite not being injured'

        # If they meet the requirements, send them to HSI_RTI_MedicalIntervention for further treatment
        # Using counts condition to stop spurious symptoms progressing people through the model
        if counts > 0:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Medical_Intervention(module=self, person_id=person_id),
                priority=0,
                topen=self.sim.date
            )

    def rti_do_for_major_surgeries(self, person_id, count):
        """
        Function called in HSI_RTI_MedicalIntervention to schedule a major surgery if required. In
        HSI_RTI_MedicalIntervention, we determine that they need a surgery. In this function, further to scheduling the
        surgery, we double check that they do meet the conditions for needing a surgery. The conditions for needing a
        surgery is that they are alive, currently seeking medical intervention and have an injury that is treated by
        surgery.
        :param person_id: The person requesting major surgeries
        :param count: The amount of major surgeries required, used when scheduling surgeries to ensure that two major
                      surgeries aren't scheduled on the same day
        :return: n/a
        """
        df = self.sim.population.props
        p = self.parameters
        if df.at[person_id, 'is_alive']:
            person = df.loc[person_id]
            # Check to see whether they have been sent here from RTI_MedicalIntervention and they haven't died due to
            # rti
            assert person.rt_med_int, 'person sent here not been through RTI_MedInt'
            # Determine what injuries are able to be treated by surgery by checking the injury codes which are currently
            # treated in this simulation, it seems there is a limited available to treat spinal cord injuries and chest
            # trauma in Malawi, so these are initially left out, but we will test different scenarios to see what
            # happens when we include those treatments
            surgically_treated_codes = ['112', '811', '812', '813a', '813b', '813c', '133a', '133b', '133c', '133d',
                                        '134a', '134b', '135', '552', '553', '554', '342', '343', '414', '361', '363',
                                        '782', '782a', '782b', '782c', '783', '822a', '882', '883', '884', 'P133a',
                                        'P133b', 'P133c', 'P133d', 'P134a', 'P134b', 'P135', 'P782a', 'P782b', 'P782c',
                                        'P783', 'P882', 'P883', 'P884']

            # If we allow surgical treatment of spinal cord injuries, extend the surgically treated codes to include
            # spinal cord injury codes
            if 'include_spine_surgery' in p['allowed_interventions']:
                additional_codes = ['673a', '673b', '674a', '674b', '675a', '675b', '676', 'P673a', 'P673b', 'P674',
                                    'P674a', 'P674b', 'P675', 'P675a', 'P675b', 'P676']
                surgically_treated_codes.extend(additional_codes)
            # If we allow surgical treatment of chest trauma, extend the surgically treated codes to include chest
            # trauma codes.
            if 'include_thoroscopy' in p['allowed_interventions']:
                additional_codes = ['441', '443', '453', '453a', '453b', '463']
                surgically_treated_codes.extend(additional_codes)
            # check this person has an injury which should be treated here
            if count == 0:
                assert len(set(person.rt_injuries_for_major_surgery) & set(surgically_treated_codes)) > 0, \
                    'This person has asked for surgery but does not have an appropriate injury'
            # isolate the relevant injury information
            person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
            # Check whether the person sent to surgery has an injury which actually requires surgery
            _, counts = RTI.rti_find_and_count_injuries(person_injuries, surgically_treated_codes)
            if counts == 0:
                logger.debug(key='rti_general_message',
                             data=f"This is rti do for major surgery person {person_id} asked for treatment but "
                                  f"doesn't need it.")
            # for each injury which has been assigned to be treated by major surgery make sure that the injury hasn't
            # already been treated
            for code in person.rt_injuries_for_major_surgery:
                column, found_code = self.rti_find_injury_column(person_id, [code])
                index_in_rt_recovery_dates = int(column[-1]) - 1
                if not pd.isnull(df.at[person_id, 'rt_date_to_remove_daly'][index_in_rt_recovery_dates]):
                    logger.debug(key='rti_general_message',
                                 data=f"person {person_id} was assigned for a minor surgery but had already received "
                                      f"treatment")
                    return
            # schedule major surgeries
            if 'Major Surgery' not in p['blocked_interventions']:
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event=HSI_RTI_Major_Surgeries(module=self,
                                                      person_id=person_id),
                    priority=0,
                    topen=self.sim.date + DateOffset(days=count),
                    tclose=self.sim.date + DateOffset(days=15))
            else:
                if count == 0:
                    df.at[person_id, 'rt_injuries_left_untreated'] = df.at[person_id, 'rt_injuries_for_major_surgery']
                    # remove the injury code from this treatment option
                    df.at[person_id, 'rt_injuries_for_major_surgery'] = []
                    # reset the time to check whether the person has died from their injuries
                    df.loc[person_id, 'rt_date_death_no_med'] = self.sim.date + DateOffset(days=1)

    def rti_do_for_minor_surgeries(self, person_id, count):
        """
        Function called in HSI_RTI_MedicalIntervention to schedule a minor surgery if required. In
        HSI_RTI_MedicalIntervention, we determine that they need a surgery. In this function, further to scheduling the
        surgery, we double check that they do meet the conditions for needing a surgery. The conditions for needing a
        surgery is that they are alive, currently seeking medical intervention and have an injury that is treated by
        surgery.
        :param person_id: The person requesting major surgeries
        :param count: The amount of major surgeries required, used when scheduling surgeries to ensure that two minor
                      surgeries aren't scheduled on the same day
        :return:
        """
        df = self.sim.population.props
        # Check to see whether they have been sent here from RTI_MedicalIntervention and they haven't been killed by the
        # RTI module
        assert df.at[person_id, 'rt_med_int'], 'Person sent for treatment did not go through rti med int'
        # Isolate the person
        if df.at[person_id, 'is_alive']:
            # state the codes treated by minor surgery
            surgically_treated_codes = ['211', '212', '291', '241', '322', '323', '722', '811', '812', '813a',
                                        '813b', '813c']
            # check that the person requesting surgery has an injury in their minor surgery treatment plan
            assert len(df.at[person_id, 'rt_injuries_for_minor_surgery']) > 0 or \
                   len(df.at[person_id, 'rt_injuries_left_untreated']) > 0, 'this person has asked for a minor ' \
                                                                            'surgery but does not need it'
            # check that for each injury due to be treated with a minor surgery, the injury hasn't previously been
            # treated
            for code in df.at[person_id, 'rt_injuries_for_minor_surgery']:
                column, found_code = self.rti_find_injury_column(person_id, [code])
                index_in_rt_recovery_dates = int(column[-1]) - 1
                if not pd.isnull(df.at[person_id, 'rt_date_to_remove_daly'][index_in_rt_recovery_dates]):
                    logger.debug(key='rti_general_message',
                                 data=f"person {person_id} was assigned for a minor surgery but had already received "
                                      f"treatment")
                    return

            # check that this person's injuries that were decided to be treated with a minor surgery and the injuries
            # actually treated by minor surgeries coincide
            if count == 0:
                assert len(set(df.at[person_id, 'rt_injuries_for_minor_surgery']) & set(surgically_treated_codes)) > 0,\
                    'This person has asked for a minor surgery but does not need it'
            # Isolate the relevant injury information
            person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
            # Check whether the person requesting minor surgeries has an injury that requires minor surgery
            _, counts = RTI.rti_find_and_count_injuries(person_injuries, surgically_treated_codes)
            if counts == 0:
                logger.debug(key='rti_general_message',
                             data=f"person {person_id} was assigned for a minor surgery but has no injury")
                return
            # schedule the minor surgery
            if 'Minor Surgery' not in self.parameters['blocked_interventions']:
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event=HSI_RTI_Minor_Surgeries(module=self,
                                                      person_id=person_id),
                    priority=0,
                    topen=self.sim.date + DateOffset(days=count),
                    tclose=self.sim.date + DateOffset(days=15))
            else:
                if count == 0:
                    df.at[person_id, 'rt_injuries_left_untreated'] = df.at[person_id, 'rt_injuries_for_minor_surgery']
                    # remove the injury code from this treatment option
                    df.at[person_id, 'rt_injuries_for_minor_surgery'] = []
                    # reset the time to check whether the person has died from their injuries
                    df.loc[person_id, 'rt_date_death_no_med'] = self.sim.date + DateOffset(days=1)

    def rti_acute_pain_management(self, person_id):
        """
        Function called in HSI_RTI_MedicalIntervention to request pain management. This should be called for every alive
        injured person, regardless of what their injuries are. In this function we test whether they meet the
        requirements to recieve for pain relief, that is they are alive and currently receiving medical treatment.
        :param person_id: The person requesting pain management
        :return: n/a
        """
        df = self.sim.population.props

        if df.at[person_id, 'is_alive']:
            # Check to see whether they have been sent here from RTI_MedicalIntervention and they haven't died due to
            # rti
            assert df.at[person_id, 'rt_med_int'], 'person sent here not been through rti med int'
            # Isolate the relevant injury information
            person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
            # check this person is injured, search they have an injury code that isn't "none".
            idx, counts = RTI.rti_find_and_count_injuries(person_injuries,
                                                          self.PROPERTIES.get('rt_injury_1').categories[1:])
            if counts == 0:
                logger.debug(key='rti_general_message',
                             data=f"person {person_id} requested pain relief but does not need it")
                return
            # schedule pain management
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Acute_Pain_Management(module=self,
                                                        person_id=person_id),
                priority=0,
                topen=self.sim.date + DateOffset(days=1),
                tclose=self.sim.date + DateOffset(days=15))

    def rti_ask_for_suture_kit(self, person_id):
        """
        Function called by HSI_RTI_MedicalIntervention to centralise all suture kit requests. This function checks
        that the person asking for a suture kit meets the requirements to get one. That is they are alive, currently
        being treated for their injuries and that they have a laceration which needs stitching.
        :param person_id: The person asking for a suture kit
        :return: n/a
        """
        df = self.sim.population.props
        if df.at[person_id, 'is_alive']:
            # Check to see whether they have been sent here from RTI_MedicalIntervention and they haven't died due to
            # rti
            assert df.at[person_id, 'rt_med_int'], 'person sent here not been through rti med int'
            # Isolate the relevant injury information
            person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
            laceration_codes = ['1101', '2101', '3101', '4101', '5101', '6101', '7101', '8101']
            # Check they have a laceration which needs stitches
            _, counts = RTI.rti_find_and_count_injuries(person_injuries, laceration_codes)
            if counts == 0:
                logger.debug(key='rti_general_message',
                             data=f"person {person_id} requested a suture but does not need it")
                return
            # request suture
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Suture(module=self,
                                         person_id=person_id),
                priority=0,
                topen=self.sim.date + DateOffset(days=1),
                tclose=self.sim.date + DateOffset(days=15)
            )

    def rti_ask_for_shock_treatment(self, person_id):
        """
        A function called by the generic emergency appointment to treat the onset of hypovolemic shock
        :param person_id:
        :return:
        """
        df = self.sim.population.props
        if df.at[person_id, 'is_alive']:
            assert df.at[person_id, 'rt_in_shock'], 'person requesting shock treatment is not in shock'

            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Shock_Treatment(module=self,
                                                  person_id=person_id),
                priority=0,
                topen=self.sim.date + DateOffset(days=1),
                tclose=self.sim.date + DateOffset(days=15)
            )

    def rti_ask_for_imaging(self, person_id):
        """
        A function called by the generic emergency appointment to order imaging for diagnosis
        :param person_id:
        :return:
        """
        df = self.sim.population.props
        if df.at[person_id, 'is_alive']:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Imaging_Event(module=self, person_id=person_id),
                priority=0,
                topen=self.sim.date + DateOffset(days=1),
                tclose=self.sim.date + DateOffset(days=15)
            )

    def rti_ask_for_burn_treatment(self, person_id):
        """
        Function called by HSI_RTI_MedicalIntervention to centralise all burn treatment requests. This function
        schedules burn treatments for the person if they meet the requirements, that is they are alive, currently being
        treated, and they have a burn which needs to be treated.
        :param person_id: The person requesting burn treatment
        :return: n/a
        """
        df = self.sim.population.props

        if df.at[person_id, 'is_alive']:
            # Check to see whether they have been sent here from RTI_MedicalIntervention and they haven't died due to
            # rti
            assert df.at[person_id, 'rt_med_int'], 'person not been through rti med int'
            # Isolate the relevant injury information
            person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
            burn_codes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113']
            # Check to see whether they have a burn which needs treatment
            _, counts = RTI.rti_find_and_count_injuries(person_injuries, burn_codes)
            if counts == 0:
                logger.debug(key='rti_general_message',
                             data=f"person {person_id} requested burn treatment but does not need it")
                return
            # if this person is alive ask for the hsi event
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Burn_Management(module=self,
                                                  person_id=person_id),
                priority=0,
                topen=self.sim.date + DateOffset(days=1),
                tclose=self.sim.date + DateOffset(days=15)
            )

    def rti_ask_for_fracture_casts(self, person_id):
        """
        Function called by HSI_RTI_MedicalIntervention to centralise all fracture casting. This function schedules the
        fracture cast treatment if they meet the requirements to ask for it. That is they are alive, currently being
        treated and they have a fracture that needs casting (Note that this also handles slings for upper arm/shoulder
        fractures).
        :param person_id: The person asking for fracture cast/sling
        :return: n/a
        """
        df = self.sim.population.props
        if df.at[person_id, 'is_alive']:
            # Check to see whether they have been sent here from RTI_MedicalIntervention and they haven't died due to
            # rti
            assert df.at[person_id, 'rt_med_int'], 'person sent here not been through rti med int'
            # Isolate the relevant injury information
            person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
            fracture_codes = ['712a', '712b', '712c', '811', '812', '813a', '813b', '813c', '822a', '822b']
            # check that the codes assigned for treatment by rt_injuries_to_cast and the codes treated by
            # rti_fracture_cast coincide
            assert len(set(df.loc[person_id, 'rt_injuries_to_cast']) & set(fracture_codes)) > 0, \
                'This person has asked for a fracture cast'
            # Check they have an injury treated by HSI_RTI_Fracture_Cast
            _, counts = RTI.rti_find_and_count_injuries(person_injuries, fracture_codes)
            if counts == 0:
                logger.debug(key='rti_general_message',
                             data=f"person {person_id} requested a fracture cast but does not need it")
                return
            # if this person is alive request the hsi
            if 'Fracture Casts' not in self.parameters['blocked_interventions']:
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event=HSI_RTI_Fracture_Cast(module=self,
                                                    person_id=person_id),
                    priority=0,
                    topen=self.sim.date + DateOffset(days=1),
                    tclose=self.sim.date + DateOffset(days=15)
                )
            else:
                df.at[person_id, 'rt_injuries_left_untreated'] = df.at[person_id, 'rt_injuries_to_cast']
                df.at[person_id, 'rt_injuries_to_cast'] = []
                # reset the time to check whether the person has died from their injuries
                df.loc[person_id, 'rt_date_death_no_med'] = self.sim.date + DateOffset(days=1)

    def rti_ask_for_open_fracture_treatment(self, person_id, counts):
        """Function called by HSI_RTI_MedicalIntervention to centralise open fracture treatment requests. This function
        schedules an open fracture event, conditional on whether they are alive, being treated and have an appropriate
        injury.

        :param person_id: the person requesting a tetanus jab
        :param counts: the number of open fractures that requires a treatment
        :return: n/a
        """
        df = self.sim.population.props
        if df.at[person_id, 'is_alive']:
            # Check to see whether they have been sent here from RTI_MedicalIntervention and are haven't died due to rti
            assert df.at[person_id, 'rt_med_int'], 'person sent here not been through rti med int'
            # Isolate the relevant injury information
            open_fracture_codes = ['813bo', '813co', '813do', '813eo']
            person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
            # Check that they have an open fracture
            _, counts = RTI.rti_find_and_count_injuries(person_injuries, open_fracture_codes)
            if counts == 0:
                logger.debug(key='rti_general_message',
                             data=f"This is rti_ask_for_open_frac person {person_id} asked for treatment but doesn't"
                                  f"need it.")
                return
            # if the person is alive request the hsi
            for i in range(0, counts):
                # schedule the treatments, say the treatments occur a day apart for now
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event=HSI_RTI_Open_Fracture_Treatment(module=self, person_id=person_id),
                    priority=0,
                    topen=self.sim.date + DateOffset(days=0 + i),
                    tclose=self.sim.date + DateOffset(days=15 + i)
                )

    def rti_ask_for_tetanus(self, person_id):
        """
        Function called by HSI_RTI_MedicalIntervention to centralise all tetanus requests. This function schedules a
        tetanus event, conditional on whether they are alive, being treated and have an injury that requires a tetanus
        vaccine, i.e. a burn or a laceration.

        :param person_id: the person requesting a tetanus jab
        :return: n/a
        """
        df = self.sim.population.props
        if df.at[person_id, 'is_alive']:
            # Check to see whether they have been sent here from RTI_MedicalIntervention and are haven't died due to rti
            assert df.at[person_id, 'rt_med_int'], 'person sent here not been through rti med int'
            # Isolate the relevant injury information
            codes_for_tetanus = ['1101', '2101', '3101', '4101', '5101', '7101', '8101',
                                 '1114', '2114', '3113', '4113', '5113', '7113', '8113']
            person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
            # Check that they have a burn/laceration
            _, counts = RTI.rti_find_and_count_injuries(person_injuries, codes_for_tetanus)
            if counts == 0:
                logger.debug(key='rti_general_message',
                             data=f"This is rti_ask_for_tetanus person {person_id} asked for treatment but doesn't"
                                  f"need it.")
                return

            # if this person is alive, ask for the hsi
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Tetanus_Vaccine(module=self,
                                                  person_id=person_id),
                priority=0,
                topen=self.sim.date + DateOffset(days=1),
                tclose=self.sim.date + DateOffset(days=15)
            )

    def schedule_hsi_event_for_tomorrow(self, hsi_event: HSI_Event = None):
        """
        A function to reschedule requested events for the following day if they have failed to run
        :return:
        """
        self.sim.modules['HealthSystem'].schedule_hsi_event(hsi_event, topen=self.sim.date + DateOffset(days=1),
                                                            tclose=self.sim.date + DateOffset(days=15), priority=0)

    def rti_find_injury_column(self, person_id, codes):
        """
        This function is a tool to find the injury column an injury code occurs in, when calling this funtion
        you will need to guarentee that the person has at least one of the code you are searching for, else this
        function will raise an assertion error.
        To call this function you need to provide the person who you want to perform the search on and the injury
        codes which you want to find the corresponding injury column for. The function/search will return the injury
        code which the person has from the list of codes you supplied, and which injury column from rt_injury_1 through
        to rt_injury_8, the code appears in.

        :param person_id: The person the search is being performed for
        :param codes: The injury codes being searched for
        :return: which column out of rt_injury_1 to rt_injury_8 the injury code occurs in, and the injury code itself
        """
        df = self.sim.population.props
        person_injuries = df.loc[person_id, RTI.INJURY_COLUMNS]
        injury_column = ''
        injury_code = ''
        for code in codes:
            for col in RTI.INJURY_COLUMNS:
                if person_injuries[col] == code:
                    injury_column = col
                    injury_code = code
                    break
        # Check that the search found the injury column
        assert injury_column != '', df
        # Return the found column for the injury code
        return injury_column, injury_code

    def rti_find_all_columns_of_treated_injuries(self, person_id, codes):
        """
        This function searches for treated injuries (supplied by the parameter codes) for a specific person, finding and
        returning all the columns with treated injuries and all the injury codes for the treated injuries.

        :param person_id: The person the search is being performed on
        :param codes: The treated injury codes
        :return: All columns and codes of the successfully treated injuries
        """
        df = self.sim.population.props
        person_injuries = df.loc[person_id, RTI.INJURY_COLUMNS]
        # create empty variables to return the columns and codes of the treated injuries
        columns_to_return = []
        codes_to_return = []
        # iterate over the codes in the list codes and also the injury columns
        for col, val in person_injuries.iteritems():
            # Search a sub-dataframe that is non-empty if the code is present is in that column and empty if not
            if val in codes:
                columns_to_return.append(col)
                codes_to_return.append(val)

        return columns_to_return, codes_to_return

    def rti_assign_daly_weights(self, injured_index):
        """
        This function assigns DALY weights associated with each injury when they happen.

        By default this function gives the DALY weight for each condition without treatment, this will then be swapped
        for the DALY weight associated with the injury with treatment when treatment occurs.

        The properties that this function alters are rt_disability, which is the property used to report the
        disability burden that this person has and rt_debugging_DALY_wt, which stores the true value of the
        the disability.

        :param injured_index: The people who have been involved in a road traffic accident for the current month and did
                              not die on the scene of the crash
        :return: n/a
        """
        df = self.sim.population.props

        # ==============================================================================================================
        # Check that those sent here have been involved in a road traffic accident
        assert df.loc[injured_index, 'rt_road_traffic_inc'].all()
        # Check everyone here has at least one injury to be given a daly weight to
        assert (df.loc[injured_index, 'rt_injury_1'] != "none").all()
        # Check everyone here is alive and hasn't died due to rti
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death',
                      'RTI_death_shock']
        assert (sum(~df.loc[injured_index, 'cause_of_death'].isin(rti_deaths)) == len(injured_index)) & \
               (sum(df.loc[injured_index, 'rt_imm_death']) == 0)
        selected_for_rti_inj = df.loc[injured_index, RTI.INJURY_COLUMNS]

        daly_change = selected_for_rti_inj.applymap(
            lambda code: self.ASSIGN_INJURIES_AND_DALY_CHANGES[code][1]
        ).sum(axis=1)
        df.loc[injured_index, 'rt_disability'] += daly_change

        # Store the true sum of DALY weights in the df
        df.loc[injured_index, 'rt_debugging_DALY_wt'] = df.loc[injured_index, 'rt_disability']
        # Find who's disability burden is greater than one
        DALYweightoverlimit = df.index[df['rt_disability'] > 1]
        # Set the total daly weights to one in this case
        df.loc[DALYweightoverlimit, 'rt_disability'] = 1
        # Find who's disability burden is less than one
        DALYweightunderlimit = df.index[df.rt_road_traffic_inc & ~ df.rt_imm_death & (df['rt_disability'] <= 0)]
        # Check that no one has a disability burden less than or equal to zero
        assert len(DALYweightunderlimit) == 0, ('Someone has not been given an injury burden',
                                                selected_for_rti_inj.loc[DALYweightunderlimit])
        df.loc[DALYweightunderlimit, 'rt_disability'] = 0
        assert (df.loc[injured_index, 'rt_disability'] > 0).all()

    def rti_alter_daly_post_treatment(self, person_id, codes):
        """
        This function removes the DALY weight associated with each injury code after treatment is complete. This
        function is called by RTI_Recovery_event which removes asks to remove the DALY weight when the injury has
        healed

        The properties that this function alters are rt_disability, which is the property used to report the
        disability burden that this person has and rt_debugging_DALY_wt, which stores the true value of the
        the disability.

        :param person_id: The person who needs a daly weight removed as their injury has healed
        :param codes: The injury codes for the healed injury/injuries
        :return: n/a
        """

        df = self.sim.population.props
        # Check everyone here has at least one injury to be alter the daly weight to
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        # check this person is injured, search they have an injury code that isn't "none"
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries,
                                                      self.PROPERTIES.get('rt_injury_1').categories[1:])
        assert counts > 0, 'This person has asked for medical treatment despite not being injured'
        # Check everyone here is alive and hasn't died on scene
        assert not df.at[person_id, 'rt_imm_death']

        # ------------------------------- Remove the daly weights for treated injuries ---------------------------------
        # update the total values of the daly weights
        df.at[person_id, 'rt_debugging_DALY_wt'] += \
            sum([self.ASSIGN_INJURIES_AND_DALY_CHANGES[code][3] for code in codes])
        # round off any potential floating point errors
        df.at[person_id, 'rt_debugging_DALY_wt'] = np.round(df.at[person_id, 'rt_debugging_DALY_wt'], 4)
        # if the person's true total for daly weights is greater than one, report rt_disability as one, if not
        # report the true disability burden.
        if df.at[person_id, 'rt_debugging_DALY_wt'] > 1:
            df.at[person_id, 'rt_disability'] = 1
        else:
            df.at[person_id, 'rt_disability'] = df.at[person_id, 'rt_debugging_DALY_wt']
        # if the reported daly weight is below zero add make the model report the true (and always positive) daly weight
        if df.at[person_id, 'rt_disability'] < 0:
            df.at[person_id, 'rt_disability'] = df.at[person_id, 'rt_debugging_DALY_wt']
        # Make sure the true disability burden is greater or equal to zero
        if df.at[person_id, 'rt_debugging_DALY_wt'] < 0:
            logger.debug(key='rti_general_message',
                         data=f"person {person_id} has had too many daly weights removed")
            df.at[person_id, 'rt_debugging_DALY_wt'] = 0
        # the reported disability should satisfy 0<=disability<=1, check that they do
        assert df.at[person_id, 'rt_disability'] >= 0, 'Negative disability burden'
        assert df.at[person_id, 'rt_disability'] <= 1, 'Too large disability burden'
        # remover the treated injury code from the person using rti_treated_injuries
        RTI.rti_treated_injuries(self, person_id, codes)

    def rti_swap_injury_daly_upon_treatment(self, person_id, codes):
        """
        This function swaps certain DALY weight codes upon when a person receives treatment(s). Some injuries have a
        different daly weight associated with them for the treated and untreated injuries. If an injury is 'swap-able'
        then this function removes the old daly weight for the untreated injury and gives the daly weight for the
        treated injury.

        The properties that this function alters are rt_disability, which is the property used to report the
        disability burden that this person has and rt_debugging_DALY_wt, which stores the true value of the
        the disability.


        :param person_id: The person who has received treatment
        :param codes: the 'swap-able' injury code
        :return: n/a

        """
        df = self.sim.population.props
        # Check the people that are sent here have had medical treatment
        assert df.at[person_id, 'rt_med_int']
        # Check they have an appropriate injury code to swap
        swapping_codes = RTI.SWAPPING_CODES[:]
        relevant_codes = np.intersect1d(codes, swapping_codes)
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        # check this person is injured, search they have an injury code that is swappable
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries, list(relevant_codes))
        assert counts > 0, 'This person has asked to swap an injury code, but it is not swap-able'
        # If there are any permanent injuries which are due to be swapped, remove the "P" writted at the start of injury
        # code in order to access the injury dictionary
        relevant_codes = [code.replace('P', '') for code in relevant_codes]
        # swap the relevant code's daly weight, from the daly weight associated with the injury without treatment
        # and the daly weight for the disability with treatment.
        # keep track of the changes to the daly weights
        # update the disability burdens
        df.at[person_id, 'rt_debugging_DALY_wt'] += \
            sum([self.ASSIGN_INJURIES_AND_DALY_CHANGES[code][2] for code in relevant_codes])
        df.at[person_id, 'rt_debugging_DALY_wt'] = np.round(df.at[person_id, 'rt_debugging_DALY_wt'], 4)
        # TODO: the injury '5113' seems to being treated multiple times for certain people, causing a repeated DALY
        #  weight swap which ultimately results in a negative daly weight. I need to work out why this is happening, the
        #  if statement below is a temporary fix
        # Check that the person's true disability burden is positive
        if df.at[person_id, 'rt_debugging_DALY_wt'] < 0:
            logger.debug(key='rti_general_message',
                         data=f"person {person_id} has had too many daly weights removed")
            df.at[person_id, 'rt_debugging_DALY_wt'] = 0
        # catch rounding point errors where the disability weights should be zero but aren't
        if df.at[person_id, 'rt_disability'] < 0:
            df.at[person_id, 'rt_disability'] = 0
        # Catch cases where the disability burden is greater than one in reality but needs to be
        # capped at one, if not report the true disability burden
        if df.at[person_id, 'rt_debugging_DALY_wt'] > 1:
            df.at[person_id, 'rt_disability'] = 1
        else:
            df.at[person_id, 'rt_disability'] = df.at[person_id, 'rt_debugging_DALY_wt']
        # Check the daly weights fall within the accepted bounds
        assert df.at[person_id, 'rt_disability'] >= 0, 'Negative disability burden'
        assert df.at[person_id, 'rt_disability'] <= 1, 'Too large disability burden'

    def rti_determine_LOS(self, person_id):
        """
        This function determines the length of stay a person sent to the health care system will require, based on how
        severe their injuries are (determined by the person's ISS score). Currently I use data from China, but once a
        more appropriate source of data is found I can swap this over.
        :param person_id: The person who needs their LOS determined
        :return: the inpatient days required to treat this person (Their LOS)
        """
        p = self.parameters
        df = self.sim.population.props

        def draw_days(_mean, _sd):
            return int(self.rng.normal(_mean, _sd, 1))

        # Create the length of stays required for each ISS score boundaries and check that they are >=0
        rt_iss_score = df.at[person_id, 'rt_ISS_score']

        if rt_iss_score < 4:
            days_until_treatment_end = draw_days(p["mean_los_ISS_less_than_4"], p["sd_los_ISS_less_than_4"])
        elif 4 <= rt_iss_score < 9:
            days_until_treatment_end = draw_days(p["mean_los_ISS_4_to_8"], p["sd_los_ISS_4_to_8"])
        elif 9 <= rt_iss_score < 16:
            days_until_treatment_end = draw_days(p["mean_los_ISS_9_to_15"], p["sd_los_ISS_9_to_15"])
        elif 16 <= rt_iss_score < 25:
            days_until_treatment_end = draw_days(p["mean_los_ISS_16_to_24"], p["sd_los_ISS_16_to_24"])
        elif 25 <= rt_iss_score:
            days_until_treatment_end = draw_days(p["mean_los_ISS_more_than_25"], p["sd_los_ISS_more_that_25"])
        else:
            days_until_treatment_end = 0
        # Make sure inpatient days is less that max available
        if days_until_treatment_end > 150:
            days_until_treatment_end = 150
        # Return the LOS
        return max(days_until_treatment_end, 0)

    @staticmethod
    def rti_find_and_count_injuries(persons_injury_properties: pd.DataFrame, injury_codes: list):
        """
        A function that searches a user given dataframe for a list of injuries (injury_codes). If the injury code is
        found in the dataframe, this function returns the index for who has the injury/injuries and the number of
        injuries found. This function works much faster if the dataframe is smaller, hence why the searched dataframe
        is a parameter in the function.

        :param persons_injury_properties: The dataframe to search for the tlo injury codes in
        :param injury_codes: The injury codes to search for in the data frame
        :return: the df index of who has the injuries and how many injuries in the search were found.
        """
        assert isinstance(persons_injury_properties, pd.DataFrame)
        assert isinstance(injury_codes, list)
        injury_counts = persons_injury_properties.isin(injury_codes).sum(axis=1)
        people_with_given_injuries = injury_counts[injury_counts > 0]
        return people_with_given_injuries.index, people_with_given_injuries.sum()

    def rti_treated_injuries(self, person_id, tlo_injury_codes):
        """
        A function that takes a person with treated injuries and removes the injury code from the properties rt_injury_1
        to rt_injury_8

        The properties that this function alters are rt_injury_1 through rt_injury_8 and the symptoms properties

        :param person_id: The person who needs an injury code removed
        :param tlo_injury_codes: the injury code(s) to be removed
        :return: n/a
        """
        df = self.sim.population.props
        # Isolate the relevant injury information
        permanent_injuries = {'P133', 'P133a', 'P133b', 'P133c', 'P133d', 'P134', 'P134a', 'P134b', 'P135', 'P673',
                              'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675', 'P675a', 'P675b', 'P676', 'P782a',
                              'P782b', 'P782c', 'P783', 'P882', 'P883', 'P884'}
        person_injuries = df.loc[person_id, RTI.INJURY_COLUMNS]

        # only remove non-permanent injuries
        codes_to_remove = [c for c in tlo_injury_codes if c not in permanent_injuries]

        # get injury columns for all codes to remove
        injury_cols = person_injuries.index[person_injuries.isin(codes_to_remove)].tolist()

        # if no injuries to reset, exit
        if len(injury_cols) == 0:
            return

        # Reset the treated injury code to "none"
        df.loc[person_id, injury_cols] = "none"

        # Reset symptoms so that after being treated for an injury the person won't interact with the
        # health system again.
        if df.at[person_id, 'sy_injury'] != 0:
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=person_id,
                disease_module=self.sim.modules['RTI'],
                add_or_remove='-',
                symptom_string='injury')

        if df.at[person_id, 'sy_severe_trauma'] != 0:
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=person_id,
                disease_module=self.sim.modules['RTI'],
                add_or_remove='-',
                symptom_string='severe_trauma')

    def on_birth(self, mother_id, child_id):
        """
        When a person is born this function sets up the default properties for the road traffic injuries module
        :param mother_id: The mother
        :param child_id: The newborn
        :return: n/a
        """
        df = self.sim.population.props
        df.at[child_id, 'rt_road_traffic_inc'] = False
        df.at[child_id, 'rt_inj_severity'] = "none"  # default: no one has been injured in a RTI
        df.at[child_id, 'rt_injury_1'] = "none"
        df.at[child_id, 'rt_injury_2'] = "none"
        df.at[child_id, 'rt_injury_3'] = "none"
        df.at[child_id, 'rt_injury_4'] = "none"
        df.at[child_id, 'rt_injury_5'] = "none"
        df.at[child_id, 'rt_injury_6'] = "none"
        df.at[child_id, 'rt_injury_7'] = "none"
        df.at[child_id, 'rt_injury_8'] = "none"
        df.at[child_id, 'rt_in_shock'] = False
        df.at[child_id, 'rt_death_from_shock'] = False
        df.at[child_id, 'rt_injuries_to_cast'] = []
        df.at[child_id, 'rt_injuries_for_minor_surgery'] = []
        df.at[child_id, 'rt_injuries_for_major_surgery'] = []
        df.at[child_id, 'rt_injuries_to_heal_with_time'] = []
        df.at[child_id, 'rt_injuries_for_open_fracture_treatment'] = []
        df.at[child_id, 'rt_polytrauma'] = False
        df.at[child_id, 'rt_ISS_score'] = 0
        df.at[child_id, 'rt_imm_death'] = False
        df.at[child_id, 'rt_perm_disability'] = False
        df.at[child_id, 'rt_med_int'] = False  # default: no one has a had medical intervention
        df.at[child_id, 'rt_in_icu_or_hdu'] = False
        df.at[child_id, 'rt_date_to_remove_daly'] = [pd.NaT] * 8
        df.at[child_id, 'rt_diagnosed'] = False
        df.at[child_id, 'rt_recovery_no_med'] = False  # default: no recovery without medical intervention
        df.at[child_id, 'rt_post_med_death'] = False  # default: no death after medical intervention
        df.at[child_id, 'rt_no_med_death'] = False
        df.at[child_id, 'rt_unavailable_med_death'] = False
        df.at[child_id, 'rt_disability'] = 0  # default: no disability due to RTI
        df.at[child_id, 'rt_date_inj'] = pd.NaT
        df.at[child_id, 'rt_MAIS_military_score'] = 0
        df.at[child_id, 'rt_date_death_no_med'] = pd.NaT
        df.at[child_id, 'rt_debugging_DALY_wt'] = 0
        df.at[child_id, 'rt_injuries_left_untreated'] = []

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        logger.debug(key='rti_general_message',
                     data=f"This is RTI, being alerted about a health system interaction person %d for: %s, {person_id}"
                          f", {treatment_id}"
                     )

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.
        logger.debug(key='rti_general_message', data='This is RTI reporting my daly values')
        df = self.sim.population.props
        disability_series_for_alive_persons = df.loc[df.is_alive, "rt_disability"]
        return disability_series_for_alive_persons

    def rti_assign_injuries(self, number):
        """
        A function that can be called specifying the number of people affected by RTI injuries
         and provides outputs for the number of injuries each person experiences from a RTI event, the location of the
         injury, the TLO injury categories and the severity of the injuries. The severity of the injuries will then be
         used to calculate the injury severity score (ISS), which will then inform mortality and disability from road
         traffic injuries with treatment and the military abreviated injury score (MAIS) which will be used to predict
         mortality without medical intervention.


        :param number: The number of people who need injuries assigned to them
        :return: injurydescription - a dataframe for the injury/injuries summarised in the TLO injury code form along
                                     with data on their ISS score, used for calculating mortality and whether they have
                                     polytrauma or not.
        Todo: see if we can include the following factors for injury severity (taken from a preprint sent to me after
            MOH meeting):
            - The setting of the patient (rural/urban) as rural location was a predictor for severe injury AOR 2.41
            (1.49-3.90)
            - Seatbelt use (none compared to using AOR 4.49 (1.47-13.76))
            - Role of person in crash (Different risk for different type, see paper)
            - Alcohol use (AOR 1.74 (1.11-2.74) compared to none)
        """
        p = self.parameters
        # Import the fitted distribution of injured body regions
        number_of_injured_body_regions_distribution = p['number_of_injured_body_regions_distribution']
        # Get the probability distribution for the likelihood of a certain body region being injured.
        injlocdist = p['injury_location_distribution']
        # create a dataframe to store the injury persons information in
        inj_df = pd.DataFrame(columns=['Injury_codes', 'AIS_scores', 'ISS', 'Polytrauma', 'MAIS', 'Number_of_injuries'])
        # Create empty lists to store information of each person's injuries to be used in logging:
        # predicted injury location
        predinjlocs = []
        # predicted injury severity
        predinjsev = []
        # predicted injury category
        predinjcat = []
        # create empty lists to store the qualitative description of injury severity and the number of injuries
        # each person has
        severity_category = []
        # ============================= Begin assigning injuries to people =====================================
        # Iterate over the total number of injured people
        for n in range(0, number):
            # Generate a random number which will decide how many injuries the person will have,
            ninj = self.rng.choice(number_of_injured_body_regions_distribution[0],
                                   p=number_of_injured_body_regions_distribution[1])
            # create an empty list which stores the injury chosen for this person
            injuries_chosen = []
            # Create an empty vector which will store the severity of the injuries
            injais = []
            # Create an empty vector to store injury MAIS scores in
            injmais = []
            # generate the locations of the injuries for this person (chosed without replacement, meaning that each
            # injury corresponds to a single body region)
            injurylocation = self.rng.choice(injlocdist[0], ninj, p=injlocdist[1], replace=False)
            # iterate over the chosen injury locations to determine the exact injuries that this person will have
            for injlocs in injurylocation:
                # get a list of the injuries that occur at this location to filter the dictionary
                # self.ASSIGN_INJURIES_AND_DALY_CHANGES to the relevant injury information
                injuries_at_location = [injury for injury in self.ASSIGN_INJURIES_AND_DALY_CHANGES.keys() if
                                        injury.startswith(str(injlocs))]
                # find the probability of each injury based on the above filter
                prob_of_each_injury_at_location = [self.ASSIGN_INJURIES_AND_DALY_CHANGES[injury][0][0] for injury in
                                                   injuries_at_location]
                # make sure there are no rounding errors (meaning all probabilities sum to one)
                prob_of_each_injury_at_location = np.divide(prob_of_each_injury_at_location,
                                                            sum(prob_of_each_injury_at_location))
                # chose an injury to occur at this location
                injury_chosen = self.rng.choice(injuries_at_location, p=prob_of_each_injury_at_location)
                # store this persons chosen injury at this location
                injuries_chosen.append(injury_chosen)
                # Store this person's injury location (used in logging)
                predinjlocs.append(self.ASSIGN_INJURIES_AND_DALY_CHANGES[injury_chosen][0][1])
                # store the injury category chosen (used in logging)
                predinjcat.append(self.ASSIGN_INJURIES_AND_DALY_CHANGES[injury_chosen][0][2])
                # store the severity of the injury chosen
                injais.append(self.ASSIGN_INJURIES_AND_DALY_CHANGES[injury_chosen][0][3])
                # store the MAIS score
                injmais.append(self.ASSIGN_INJURIES_AND_DALY_CHANGES[injury_chosen][0][4])
            # create the data needed for an additional row, the injuries chosen for this person, their corresponding
            # AIS scores, their ISS score (calculated as the summed square of their three most severly injured body
            # regions, whether or not they have polytrauma (calculated if they have two or more body regions with an ISS
            # score greater than 2), their MAIS score and the number of injuries they have
            new_row = {'Injury_codes': injuries_chosen,
                       'AIS_scores': injais,
                       'ISS': sum(sorted(np.square(injais))[-3:]),
                       'Polytrauma': sum(i > 2 for i in injais) > 1,
                       'MAIS': max(injmais),
                       'Number_of_injuries': ninj}
            inj_df = inj_df.append(new_row, ignore_index=True)
            # If person has an ISS score less than 15 they have a mild injury, otherwise severe
            if new_row['ISS'] < 15:
                severity_category.append('mild')
            else:
                severity_category.append('severe')
            # Store this person's injury information into the lists which house each individual person's injury
            # information
            predinjsev.append(injais)
        # If there is at least one injured person, expand the returned dataframe so that each injury has it's own column
        if len(inj_df) > 0:
            # create a copy of the injury codes
            listed_injuries = inj_df.copy()
            # expand the injury codes into their own columns
            listed_injuries = listed_injuries['Injury_codes'].apply(pd.Series)
            # rename the columns
            listed_injuries = listed_injuries.rename(columns=lambda x: 'rt_injury_' + str(x + 1))
            # join the expanded injuries to the injury df
            inj_df = inj_df.join(listed_injuries)
        # Fill dataframe entries where a person has not had an injury assigned with 'none'
        inj_df = inj_df.fillna("none")

        # Begin logging injury information
        # ============================ Injury category incidence ======================================================
        # log the incidence of each injury category
        # count the number of injuries that fall in each category
        amputationcounts = sum(1 for i in predinjcat if i == '8')
        burncounts = sum(1 for i in predinjcat if i == '11')
        fraccounts = sum(1 for i in predinjcat if i == '1')
        tbicounts = sum(1 for i in predinjcat if i == '3')
        minorinjurycounts = sum(1 for i in predinjcat if i == '10')
        spinalcordinjurycounts = sum(1 for i in predinjcat if i == '7')
        other_counts = sum(1 for i in predinjcat if i in ['2', '4', '5', '6', '9'])
        # calculate the incidence of this injury in the population
        df = self.sim.population.props
        n_alive = len(df.is_alive)
        inc_amputations = amputationcounts / ((n_alive - amputationcounts) * 1 / 12) * 100000
        inc_burns = burncounts / ((n_alive - burncounts) * 1 / 12) * 100000
        inc_fractures = fraccounts / ((n_alive - fraccounts) * 1 / 12) * 100000
        inc_tbi = tbicounts / ((n_alive - tbicounts) * 1 / 12) * 100000
        inc_sci = spinalcordinjurycounts / ((n_alive - spinalcordinjurycounts) * 1 / 12) * 100000
        inc_minor = minorinjurycounts / ((n_alive - minorinjurycounts) * 1 / 12) * 100000
        inc_other = other_counts / ((n_alive - other_counts) * 1 / 12) * 100000
        tot_inc_all_inj = inc_amputations + inc_burns + inc_fractures + inc_tbi + inc_sci + inc_minor + inc_other
        if number > 0:
            number_of_injuries = inj_df['Number_of_injuries'].tolist()
        else:
            number_of_injuries = 0
        dict_to_output = {'inc_amputations': inc_amputations,
                          'inc_burns': inc_burns,
                          'inc_fractures': inc_fractures,
                          'inc_tbi': inc_tbi,
                          'inc_sci': inc_sci,
                          'inc_minor': inc_minor,
                          'inc_other': inc_other,
                          'tot_inc_injuries': tot_inc_all_inj,
                          'number_of_injuries': number_of_injuries}

        logger.info(key='Inj_category_incidence',
                    data=dict_to_output,
                    description='Incidence of each injury grouped as per the GBD definition')
        # Log injury information
        # Get injury severity information in an easily interpreted form to be logged.
        # create a list of the predicted injury severity scores
        flattened_injury_ais = [str(item) for sublist in predinjsev for item in sublist]

        injury_info = {'Number_of_injuries': number_of_injuries,
                       'Location_of_injuries': predinjlocs,
                       'Injury_category': predinjcat,
                       'Per_injury_severity': flattened_injury_ais,
                       'Per_person_injury_severity': inj_df['ISS'].to_list(),
                       'Per_person_MAIS_score': inj_df['MAIS'].to_list(),
                       'Per_person_severity_category': severity_category
                       }
        logger.info(key='Injury_information',
                    data=injury_info,
                    description='Relevant information on the injuries from road traffic accidents when they are '
                                'assigned')
        # log the fraction of lower extremity fractions that are open
        flattened_injuries = [str(item) for sublist in inj_df['Injury_codes'].to_list() for item in sublist]
        lx_frac_codes = ['811', '813do', '812', '813eo', '813', '813a', '813b', '813bo', '813c', '813co']
        lx_open_frac_codes = ['813do', '813eo', '813bo', '813co']
        n_lx_fracs = len([inj for inj in flattened_injuries if inj in lx_frac_codes])
        n_open_lx_fracs = len([inj for inj in flattened_injuries if inj in lx_open_frac_codes])
        if n_lx_fracs > 0:
            proportion_lx_fracture_open = n_open_lx_fracs / n_lx_fracs
        else:
            proportion_lx_fracture_open = 'no_lx_fractures'
        injury_info = {'Proportion_lx_fracture_open': proportion_lx_fracture_open}
        logger.info(key='Open_fracture_information',
                    data=injury_info,
                    description='The proportion of fractures that are open in specific body regions')
        # Finally return the injury description information
        return inj_df

    def do_rti_diagnosis_and_treatment(self, person_id):
        """Things to do upon a person presenting at a Non-Emergency Generic HSI if they have an injury."""
        df = self.sim.population.props
        persons_injuries = df.loc[person_id, RTI.INJURY_COLUMNS]
        if pd.isnull(df.at[person_id, 'cause_of_death']) and not df.at[person_id, 'rt_diagnosed']:
            if len(set(RTI.INJURIES_REQ_IMAGING).intersection(persons_injuries)) > 0:
                self.rti_ask_for_imaging(person_id)
            df.at[person_id, 'rt_diagnosed'] = True
            self.rti_do_when_diagnosed(person_id=person_id)
            if df.at[person_id, 'rt_in_shock']:
                self.rti_ask_for_shock_treatment(person_id)


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
#
#   These are the events which drive the simulation of the disease. It may be a regular event that updates
#   the status of all the population of subsections of it at one time. There may also be a set of events
#   that represent disease events for particular persons.
# ---------------------------------------------------------------------------------------------------------


class RTIPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """The regular RTI event which handles all the initial RTI related changes to the dataframe. It can be thought of
     as the actual road traffic accident occurring. Specifically the event decides who is involved in a road traffic
     accident every month (via the linear model helper class), whether those involved in a road traffic accident die on
     scene or are given injuries (via the assign_injuries function) which they will attempt to interact with the health
     system with for treatment.

     Those who don't die on scene and are injured then attempt to go to an emergency generic first appointment

    This event will change the rt_ properties:
    1) rt_road_traffic_inc - False when not involved in a collision, True when RTI_Event decides they are in a collision

    2) rt_date_inj - Change to current date if the person has been involved in a road traffic accident

    3) rt_imm_death - True if they die on the scene of the crash, false otherwise

    4) rt_injury_1 through to rt_injury_8 - a series of 8 properties which stores the injuries that need treating as a
                                            code

    5) rt_ISS_score - The metric used to calculate the probability of mortality from the person's injuries

    6) rt_MAIS_military_score - The metric used to calculate the probability of mortality without medical intervention

    7) rt_disability - after injuries are assigned to a person, RTI_event calls rti_assign_daly_weights to match the
                       person's injury codes in rt_injury_1 through 8 to their corresponding DALY weights

    8) rt_polytrauma - If the person's injuries fit the definition for polytrauma we keep track of this here and use it
                        to calculate the probability for mortality later on.
    9) rt_date_death_no_med - the projected date to determine mortality for those who haven't sought medical care

    10) rt_inj_severity - The qualitative description of the severity of this person's injuries

    11) the symptoms this person has
    """

    def __init__(self, module):
        """Schedule to take place every month
        """
        super().__init__(module, frequency=DateOffset(months=1))
        p = module.parameters
        # Parameters which transition the model between states
        self.base_1m_prob_rti = (p['base_rate_injrti'] / 12)
        if 'reduce_incidence' in p['allowed_interventions']:
            self.base_1m_prob_rti = self.base_1m_prob_rti * 0.335
        self.rr_injrti_age04 = p['rr_injrti_age04']
        self.rr_injrti_age59 = p['rr_injrti_age59']
        self.rr_injrti_age1017 = p['rr_injrti_age1017']
        self.rr_injrti_age1829 = p['rr_injrti_age1829']
        self.rr_injrti_age3039 = p['rr_injrti_age3039']
        self.rr_injrti_age4049 = p['rr_injrti_age4049']
        self.rr_injrti_age5059 = p['rr_injrti_age5059']
        self.rr_injrti_age6069 = p['rr_injrti_age6069']
        self.rr_injrti_age7079 = p['rr_injrti_age7079']
        self.rr_injrti_male = p['rr_injrti_male']
        self.rr_injrti_excessalcohol = p['rr_injrti_excessalcohol']
        self.imm_death_proportion_rti = p['imm_death_proportion_rti']
        self.prob_bleeding_leads_to_shock = p['prob_bleeding_leads_to_shock']
        self.rt_emergency_care_ISS_score_cut_off = p['rt_emergency_care_ISS_score_cut_off']

    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """
        df = population.props
        now = self.sim.date
        # Reset injury properties after death, get an index of people who have died due to RTI, all causes
        diedfromrtiidx = df.index[df.rt_imm_death | df.rt_post_med_death | df.rt_no_med_death | df.rt_death_from_shock |
                                  df.rt_unavailable_med_death]
        df.loc[diedfromrtiidx, "rt_imm_death"] = False
        df.loc[diedfromrtiidx, "rt_post_med_death"] = False
        df.loc[diedfromrtiidx, "rt_no_med_death"] = False
        df.loc[diedfromrtiidx, "rt_unavailable_med_death"] = False
        df.loc[diedfromrtiidx, "rt_disability"] = 0
        df.loc[diedfromrtiidx, "rt_med_int"] = False
        df.loc[diedfromrtiidx, 'rt_in_icu_or_hdu'] = False
        for index, row in df.loc[diedfromrtiidx].iterrows():
            df.at[index, 'rt_date_to_remove_daly'] = [pd.NaT] * 8
            df.at[index, 'rt_injuries_to_cast'] = []
            df.at[index, 'rt_injuries_for_minor_surgery'] = []
            df.at[index, 'rt_injuries_for_major_surgery'] = []
            df.at[index, 'rt_injuries_to_heal_with_time'] = []
            df.at[index, 'rt_injuries_for_open_fracture_treatment'] = []
            df.at[index, 'rt_injuries_left_untreated'] = []
        df.loc[diedfromrtiidx, "rt_diagnosed"] = False
        df.loc[diedfromrtiidx, "rt_polytrauma"] = False
        df.loc[diedfromrtiidx, "rt_inj_severity"] = "none"
        df.loc[diedfromrtiidx, "rt_perm_disability"] = False
        df.loc[diedfromrtiidx, "rt_injury_1"] = "none"
        df.loc[diedfromrtiidx, "rt_injury_2"] = "none"
        df.loc[diedfromrtiidx, "rt_injury_3"] = "none"
        df.loc[diedfromrtiidx, "rt_injury_4"] = "none"
        df.loc[diedfromrtiidx, "rt_injury_5"] = "none"
        df.loc[diedfromrtiidx, "rt_injury_6"] = "none"
        df.loc[diedfromrtiidx, "rt_injury_7"] = "none"
        df.loc[diedfromrtiidx, "rt_injury_8"] = "none"
        df.loc[diedfromrtiidx, 'rt_date_death_no_med'] = pd.NaT
        df.loc[diedfromrtiidx, 'rt_MAIS_military_score'] = 0
        df.loc[diedfromrtiidx, 'rt_debugging_DALY_wt'] = 0
        df.loc[diedfromrtiidx, 'rt_in_shock'] = False
        # reset whether they have been selected for an injury this month
        df['rt_road_traffic_inc'] = False

        # --------------------------------- UPDATING OF RTI OVER TIME -------------------------------------------------
        # Currently we have the following conditions for being able to be involved in a road traffic injury, they are
        # alive, they aren't currently injured, they didn't die immediately in
        # a road traffic injury in the last month and finally, they aren't currently being treated for a road traffic
        # injury.
        rt_current_non_ind = df.index[df.is_alive & ~df.rt_road_traffic_inc & ~df.rt_imm_death & ~df.rt_med_int &
                                      (df.rt_inj_severity == "none")]

        # ========= Update for people currently not involved in a RTI, make some involved in a RTI event ==============
        # Use linear model helper class
        eq = LinearModel(LinearModelType.MULTIPLICATIVE,
                         self.base_1m_prob_rti,
                         Predictor('sex').when('M', self.rr_injrti_male),
                         Predictor(
                             'age_years',
                             conditions_are_mutually_exclusive=True
                         )
                         .when('.between(0,4)', self.rr_injrti_age04)
                         .when('.between(5,9)', self.rr_injrti_age59)
                         .when('.between(10,17)', self.rr_injrti_age1017)
                         .when('.between(18,29)', self.rr_injrti_age1829)
                         .when('.between(30,39)', self.rr_injrti_age3039)
                         .when('.between(40,49)', self.rr_injrti_age4049)
                         .when('.between(50,59)', self.rr_injrti_age5059)
                         .when('.between(60,69)', self.rr_injrti_age6069)
                         .when('.between(70,79)', self.rr_injrti_age7079),
                         Predictor('li_ex_alc').when(True, self.rr_injrti_excessalcohol)
                         )
        pred = eq.predict(df.loc[rt_current_non_ind])
        random_draw_in_rti = self.module.rng.random_sample(size=len(rt_current_non_ind))
        selected_for_rti = rt_current_non_ind[pred > random_draw_in_rti]
        # Update to say they have been involved in a rti
        df.loc[selected_for_rti, 'rt_road_traffic_inc'] = True
        # Set the date that people were injured to now
        df.loc[selected_for_rti, 'rt_date_inj'] = now
        # ========================= Take those involved in a RTI and assign some to death ==============================
        # This section accounts for pre-hospital mortality, where a person is so severy injured that they die before
        # being able to seek medical care
        selected_to_die = selected_for_rti[self.imm_death_proportion_rti >
                                           self.module.rng.random_sample(size=len(selected_for_rti))]
        # Keep track of who experience pre-hospital mortality with the property rt_imm_death
        df.loc[selected_to_die, 'rt_imm_death'] = True
        # For each person selected to experience pre-hospital mortality, schedule an InstantaneosDeath event
        for individual_id in selected_to_die:
            self.sim.modules['Demography'].do_death(individual_id=individual_id, cause="RTI_imm_death",
                                                    originating_module=self.module)
        # ============= Take those remaining people involved in a RTI and assign injuries to them ==================
        # Drop those who have died immediately
        selected_for_rti_inj_idx = selected_for_rti.drop(selected_to_die)
        # Check that none remain
        assert len(selected_for_rti_inj_idx.intersection(selected_to_die)) == 0
        # take a copy dataframe, used to get the index of the population affected by RTI
        selected_for_rti_inj = df.loc[selected_for_rti_inj_idx]
        # Again make sure that those who have injuries assigned to them are alive, involved in a crash and didn't die on
        # scene
        selected_for_rti_inj = selected_for_rti_inj.loc[df.is_alive & df.rt_road_traffic_inc & ~df.rt_imm_death]
        # To stop people who have died from causes outside of the RTI module progressing through the model, remove
        # any person with the condition 'cause_of_death' is not null
        died_elsewhere_index = selected_for_rti_inj[~ selected_for_rti_inj['cause_of_death'].isnull()].index
        # drop the died_elsewhere_index from selected_for_rti_inj
        selected_for_rti_inj.drop(died_elsewhere_index, inplace=True)
        # Create shorthand link to RTI module
        road_traffic_injuries = self.sim.modules['RTI']

        # if people have been chosen to be injured, assign the injuries using the assign injuries function
        description = road_traffic_injuries.rti_assign_injuries(len(selected_for_rti_inj))
        # replace the nan values with 'none', this is so that the injuries can be copied over from this temporarily used
        # pandas dataframe will fit in with the categories in the columns rt_injury_1 through rt_injury_8
        description = description.replace('nan', 'none')
        # set the index of the description dataframe, so that we can join it to the selected_for_rti_inj dataframe
        description = description.set_index(selected_for_rti_inj.index)
        # copy over values from the assign injury dataframe to self.sim.population.props

        df.loc[selected_for_rti_inj.index, 'rt_ISS_score'] = \
            description.loc[selected_for_rti_inj.index, 'ISS'].astype(int)
        df.loc[selected_for_rti_inj.index, 'rt_MAIS_military_score'] = \
            description.loc[selected_for_rti_inj.index, 'MAIS'].astype(int)
        # ======================== Apply the injuries to the population dataframe ======================================
        # Find the corresponding column names
        injury_columns = pd.Index(RTI.INJURY_COLUMNS)
        matching_columns = description.columns.intersection(injury_columns)
        for col in matching_columns:
            df.loc[selected_for_rti_inj.index, col] = description.loc[selected_for_rti_inj.index, col]
        # Run assert statements to make sure the model is behaving as it should
        # All those who are injured in a road traffic accident have this noted in the property 'rt_road_traffic_inc'
        assert df.loc[selected_for_rti, 'rt_road_traffic_inc'].all()
        # All those who are involved in a road traffic accident have these noted in the property 'rt_date_inj'
        assert (df.loc[selected_for_rti, 'rt_date_inj'] != pd.NaT).all()
        # All those who are injures and do not die immediately have an ISS score > 0
        assert len(df.loc[df.rt_road_traffic_inc & ~df.rt_imm_death, 'rt_ISS_score'] > 0) == \
               len(df.loc[df.rt_road_traffic_inc & ~df.rt_imm_death])
        # ========================== Determine who will experience shock from blood loss ==============================
        internal_bleeding_codes = ['361', '363', '461', '463', '813bo', '813co', '813do', '813eo']
        df = self.sim.population.props

        potential_shock_index, _ = \
            road_traffic_injuries.rti_find_and_count_injuries(df.loc[df.rt_road_traffic_inc, RTI.INJURY_COLUMNS],
                                                              internal_bleeding_codes)
        rand_for_shock = self.module.rng.random_sample(len(potential_shock_index))
        shock_index = potential_shock_index[self.prob_bleeding_leads_to_shock > rand_for_shock]
        df.loc[shock_index, 'rt_in_shock'] = True
        # log the percentage of those with RTIs in shock
        percent_in_shock = \
            len(shock_index) / len(selected_for_rti_inj) if len(selected_for_rti_inj) > 0 else 'none_injured'
        logger.info(key='Percent_of_shock_in_rti',
                    data={'Percent_of_shock_in_rti': percent_in_shock},
                    description='The percentage of those assigned injuries who were also assign the shock property')
        # ========================== Decide survival time without medical intervention ================================
        # todo: find better time for survival data without med int for ISS scores
        # Assign a date in the future for which when the simulation reaches that date, the person's mortality will be
        # checked if they haven't sought care
        df.loc[selected_for_rti_inj.index, 'rt_date_death_no_med'] = now + DateOffset(days=7)
        # ============================ Injury severity classification =================================================
        # Find those with mild injuries and update the rt_inj_severity property so they have a mild injury
        injured_this_month = df.loc[selected_for_rti_inj.index]
        mild_rti_idx = injured_this_month.index[injured_this_month.is_alive & injured_this_month['rt_ISS_score'] < 15]
        df.loc[mild_rti_idx, 'rt_inj_severity'] = 'mild'
        # Find those with severe injuries and update the rt_inj_severity property so they have a severe injury
        severe_rti_idx = injured_this_month.index[injured_this_month['rt_ISS_score'] >= 15]
        df.loc[severe_rti_idx, 'rt_inj_severity'] = 'severe'
        # check that everyone who has been assigned an injury this month has an associated injury severity
        assert sum(df.loc[df.rt_road_traffic_inc & ~df.rt_imm_death & (df.rt_date_inj == now), 'rt_inj_severity']
                   != 'none') == len(selected_for_rti_inj.index)
        # Find those with polytrauma and update the rt_polytrauma property so they have polytrauma
        polytrauma_idx = description.loc[description.Polytrauma].index
        df.loc[polytrauma_idx, 'rt_polytrauma'] = True
        # Assign daly weights for each person's injuries with the function rti_assign_daly_weights
        road_traffic_injuries.rti_assign_daly_weights(selected_for_rti_inj.index)

        # =============================== Health seeking behaviour set up =======================================
        # Set up health seeking behaviour. Two symptoms are used in the RTI module, the generic injury symptom and an
        # emergency symptom 'severe_trauma'.

        # The condition to be sent to the health care system: 1) They must be alive 2) They must have been involved in a
        # road traffic accident 3) they must have not died immediately in the accident 4) they must not have been to an
        # A and E department previously and been diagnosed

        # The symptom they are assigned depends injury severity, those with mild injuries will be assigned the generic
        # symptom, those with severe injuries will have the emergency injury symptom

        # Create the logical conditions for each symptom
        condition_to_be_sent_to_em = \
            df.is_alive & df.rt_road_traffic_inc & ~df.rt_diagnosed & ~df.rt_imm_death & (df.rt_date_inj == now) & \
            (df.rt_injury_1 != "none") & (df.rt_ISS_score >= self.rt_emergency_care_ISS_score_cut_off)
        condition_to_be_sent_to_begin_non_emergency = \
            df.is_alive & df.rt_road_traffic_inc & ~df.rt_diagnosed & ~df.rt_imm_death & (df.rt_date_inj == now) & \
            (df.rt_injury_1 != "none") & (df.rt_ISS_score < self.rt_emergency_care_ISS_score_cut_off)
        # check that all those who meet the conditions to try and seek healthcare have at least one injury
        assert sum(df.loc[condition_to_be_sent_to_em, 'rt_injury_1'] != "none") == \
               len(df.loc[condition_to_be_sent_to_em])
        assert sum(df.loc[condition_to_be_sent_to_begin_non_emergency, 'rt_injury_1'] != "none") == \
               len(df.loc[condition_to_be_sent_to_begin_non_emergency])
        # create indexes of people to be assigned each rti symptom
        em_idx = df.index[condition_to_be_sent_to_em]
        non_em_idx = df.index[condition_to_be_sent_to_begin_non_emergency]
        # Assign the symptoms
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=em_idx.tolist(),
            disease_module=self.module,
            add_or_remove='+',
            symptom_string='severe_trauma',
        )
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=non_em_idx.tolist(),
            disease_module=self.module,
            add_or_remove='+',
            symptom_string='injury',
        )


class RTI_Check_Death_No_Med(RegularEvent, PopulationScopeEventMixin):
    """
    A regular event which organises whether a person who has not received medical treatment should die as a result of
    their injuries. This even makes use of the maximum AIS-military score, a trauma scoring system developed for
    injuries in a military environment, assumed here to be an indicator of the probability of mortality without
    access to a medical system.

    The properties this function changes are:
    1) rt_no_med_death - the boolean property tracking who dies from road traffic injuries without medical intervention

    2) rt_date_death_no_med - resetting the date to check the person's mortality without medical intervention if
                              they survive
    3) rt_disability - if the person survives a non-fatal injury then this injury may heal and therefore the disability
                       burden is changed
    4) rt_debugging_DALY_wt - if the person survives a non-fatal injury then this injury may heal and therefore the
                              disability burden is changed, this property keeping track of the true disability burden
    5) rt_date_to_remove_daly - In the event of recovering from a non-fatal injury without medical intervention
                                a recovery date will scheduled

    If the person is sent here and they don't die, we need to correctly model the level of disability they experience
    from their untreated injuries, some injuries that are left untreated will have an associated daly weight for long
    term disability without treatment, others don't.

    # todo: consult with a doctor about the likelihood of survival without medical treatment

    Currently I am assuming that anyone with an injury severity score of 9 or higher will seek care and have an
    emergency symptom, that means that I have to decide what to do with the following injuries:

    Lacerations - [1101, 2101, 3101, 4101, 5101, 7101, 8101]
    What would a laceration do without stitching? Take longer to heal most likely
    Fractures - ['112', '211', '212, '412', '612', '712', '712a', '712b', '712c', '811', '812']

    Some fractures have an associated level of disability to them, others do not. So things like fractured radius/ulna
    have a code to swap, but others don't. Some are of the no treatment type, such as fractured skulls, fractured ribs
    or fractured vertebrae, so we can just add the same recovery time for these injuries. So '112', '412' and '612' will
    need to have recovery events checked and recovered.
    Dislocations will presumably be popped back into place, the injury will feasably be able to heal but most likely
    with more pain and probably with more time
    Amputations - ['782','782a', '782b', '782c', '882']
    Amputations will presumably trigger emergency health seeking behaviour so they shouldn't turn up here really
    soft tissue injuries - ['241', '342', '441', '442']
    Presumably soft tissue injuries that turn up here will heal over time but with more pain
    Internal organ injury - ['552']
    Injury to the gastrointestinal organs can result in complications later on, but
    Internal bleedings - ['361', '461']
    Surviving internal bleeding is concievably possible, these are comparitively minor bleeds


    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(days=1))
        assert isinstance(module, RTI)
        p = module.parameters
        # Load parameters used by this event
        self.prob_death_MAIS3 = p['prob_death_MAIS3']
        self.prob_death_MAIS4 = p['prob_death_MAIS4']
        self.prob_death_MAIS5 = p['prob_death_MAIS5']
        self.prob_death_MAIS6 = p['prob_death_MAIS6']
        self.daly_wt_radius_ulna_fracture_short_term_with_without_treatment = \
            p['daly_wt_radius_ulna_fracture_short_term_with_without_treatment']
        self.daly_wt_radius_ulna_fracture_long_term_without_treatment = \
            p['daly_wt_radius_ulna_fracture_long_term_without_treatment']
        self.daly_wt_foot_fracture_short_term_with_without_treatment = \
            p['daly_wt_foot_fracture_short_term_with_without_treatment']
        self.daly_wt_foot_fracture_long_term_without_treatment = \
            p['daly_wt_foot_fracture_long_term_without_treatment']
        self.daly_wt_hip_fracture_short_term_with_without_treatment = \
            p['daly_wt_hip_fracture_short_term_with_without_treatment']
        self.daly_wt_hip_fracture_long_term_without_treatment = \
            p['daly_wt_hip_fracture_long_term_without_treatment']
        self.daly_wt_pelvis_fracture_short_term = p['daly_wt_pelvis_fracture_short_term']
        self.daly_wt_pelvis_fracture_long_term = \
            p['daly_wt_pelvis_fracture_long_term']
        self.daly_wt_femur_fracture_short_term = p['daly_wt_femur_fracture_short_term']
        self.daly_wt_femur_fracture_long_term_without_treatment = \
            p['daly_wt_femur_fracture_long_term_without_treatment']
        self.no_treatment_mortality_mais_cutoff = p['unavailable_treatment_mortality_mais_cutoff']
        self.no_treatment_ISS_cut_off = p['consider_death_no_treatment_ISS_cut_off']

    def apply(self, population):
        df = population.props
        now = self.sim.date
        probabilities_of_death = {
            '1': 0,
            '2': 0,
            '3': self.prob_death_MAIS3,
            '4': self.prob_death_MAIS4,
            '5': self.prob_death_MAIS5,
            '6': self.prob_death_MAIS6
        }
        # check if anyone is due to have their mortality without medical intervention determined today
        if len(df.loc[df['rt_date_death_no_med'] == now]) > 0:
            # Get an index of those scheduled to have their mortality checked
            due_to_die_today_without_med_int = df.loc[df['rt_date_death_no_med'] == now].index
            # iterate over those scheduled to die
            for person in due_to_die_today_without_med_int:
                # Create a random number to determine mortality
                rand_for_death = self.module.rng.random_sample(1)
                # create a variable to show if a person has died due to their untreated injuries
                # find which injuries have been untreated
                untreated_injuries = []
                persons_injuries = df.loc[[person], RTI.INJURY_COLUMNS]
                non_empty_injuries = persons_injuries[persons_injuries != "none"]
                non_empty_injuries = non_empty_injuries.dropna(axis=1)
                for col in non_empty_injuries:
                    if pd.isnull(df.loc[person, 'rt_date_to_remove_daly'][int(col[-1]) - 1]):
                        untreated_injuries.append(df.at[person, col])
                mais_scores = [1]
                for injury in untreated_injuries:
                    mais_scores.append(self.module.ASSIGN_INJURIES_AND_DALY_CHANGES[injury][0][-1])
                max_untreated_injury = max(mais_scores)
                prob_death = probabilities_of_death[str(max_untreated_injury)]
                if df.loc[person, 'rt_med_int'] and (max_untreated_injury < self.no_treatment_mortality_mais_cutoff):
                    # filter out non serious injuries from the consideration of mortality
                    prob_death = 0
                if (rand_for_death < prob_death) and (df.at[person, 'rt_ISS_score'] > self.no_treatment_ISS_cut_off):
                    # If determined to die, schedule a death without med
                    df.loc[person, 'rt_no_med_death'] = True
                    self.sim.modules['Demography'].do_death(individual_id=person, cause="RTI_death_without_med",
                                                            originating_module=self.module)
                else:
                    # If the people do not die from their injuries despite not getting care, we have to decide when and
                    # to what degree their injuries will heal.
                    df.loc[[person], 'rt_recovery_no_med'] = True
                    # Reset the date to check if they die
                    df.loc[[person], 'rt_date_death_no_med'] = pd.NaT
                    swapping_codes = ['712c', '811', '813a', '813b', '813c']
                    # create a dictionary to reference changes to daly weights done here
                    swapping_daly_weights_lookup = {
                        '712c': (- self.daly_wt_radius_ulna_fracture_short_term_with_without_treatment +
                                 self.daly_wt_radius_ulna_fracture_long_term_without_treatment),
                        '811': (- self.daly_wt_foot_fracture_short_term_with_without_treatment +
                                self.daly_wt_foot_fracture_long_term_without_treatment),
                        '813a': (- self.daly_wt_hip_fracture_short_term_with_without_treatment +
                                 self.daly_wt_hip_fracture_long_term_without_treatment),
                        '813b': - self.daly_wt_pelvis_fracture_short_term + self.daly_wt_pelvis_fracture_long_term,
                        '813c': (- self.daly_wt_femur_fracture_short_term +
                                 self.daly_wt_femur_fracture_long_term_without_treatment),
                        'none': 0
                    }
                    road_traffic_injuries = self.sim.modules['RTI']
                    # If those who haven't sought health care have an injury for which we have a daly code
                    # associated with that injury long term without treatment, swap it
                    # Iterate over the person's injuries
                    injuries = df.loc[[person], RTI.INJURY_COLUMNS].values.tolist()
                    # Cannot iterate correctly over list like [[1,2,3]], so need to flatten
                    flattened_injuries = [item for sublist in injuries for item in sublist if item != 'none']
                    if df.loc[person, 'rt_med_int']:
                        flattened_injuries = [injury for injury in flattened_injuries if injury in
                                              df.loc[person, 'rt_injuries_left_untreated']]
                    persons_injuries = df.loc[[person], RTI.INJURY_COLUMNS]
                    for code in flattened_injuries:
                        swapable_code = np.intersect1d(code, swapping_codes)
                        if len(swapable_code) > 0:
                            swapable_code = swapable_code[0]
                        else:
                            swapable_code = 'none'
                        # check that the person has the injury code
                        _, counts = road_traffic_injuries.rti_find_and_count_injuries(persons_injuries, [code])
                        assert counts > 0
                        df.loc[person, 'rt_debugging_DALY_wt'] += swapping_daly_weights_lookup[swapable_code]
                        if df.loc[person, 'rt_debugging_DALY_wt'] > 1:
                            df.loc[person, 'rt_disability'] = 1
                        else:
                            df.loc[person, 'rt_disability'] = df.loc[person, 'rt_debugging_DALY_wt']
                        # if the code is swappable, swap it
                        if df.loc[person, 'rt_disability'] < 0:
                            df.loc[person, 'rt_disability'] = 0
                        if df.loc[person, 'rt_disability'] > 1:
                            df.loc[person, 'rt_disability'] = 1
                        # If they don't have a swappable code, schedule the healing of the injury
                        # get the persons injuries
                        persons_injuries = df.loc[[person], RTI.INJURY_COLUMNS]
                        non_empty_injuries = persons_injuries[persons_injuries != "none"]
                        non_empty_injuries = non_empty_injuries.dropna(axis=1)
                        injury_columns = non_empty_injuries.columns
                        columns = \
                            injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person, [code])[0])
                        # assign a recovery date
                        # not all injuries have an assigned duration of recovery. These are more serious injuries that
                        # would normally be sent directly to the health system. In the instance that a serious injury
                        # occurs and no treatment is recieved but the person survives assume they will be disabled for
                        # the duration of the simulation

                        # if they haven't sought care at all we don't need to specify which injuries need a recovery
                        # date assigned
                        if not df.loc[person, 'rt_med_int']:
                            if code in self.module.NO_TREATMENT_RECOVERY_TIMES_IN_DAYS.keys():
                                df.loc[person, 'rt_date_to_remove_daly'][columns] = \
                                    self.sim.date + DateOffset(
                                        days=self.module.NO_TREATMENT_RECOVERY_TIMES_IN_DAYS[code]
                                    )
                            else:
                                df.loc[person, 'rt_date_to_remove_daly'][columns] = self.sim.end_date + \
                                                                                    DateOffset(days=1)
                        else:
                            # if they have sought medical care and it hasn't been provided, we need to make sure only
                            # the untreated injuries have a recovery date assigned here
                            code_has_recovery_time = code in self.module.NO_TREATMENT_RECOVERY_TIMES_IN_DAYS.keys()
                            code_is_left_untreated = code in df.loc[person, 'rt_injuries_left_untreated']
                            if code_has_recovery_time & code_is_left_untreated:
                                df.loc[person, 'rt_date_to_remove_daly'][columns] = \
                                    self.sim.date + DateOffset(
                                        days=self.module.NO_TREATMENT_RECOVERY_TIMES_IN_DAYS[code]
                                    )
                            else:
                                df.loc[person, 'rt_date_to_remove_daly'][columns] = self.sim.end_date + \
                                                                                    DateOffset(days=1)
                        # remove the injury code from columns to be treated, as they have not sought care and have
                        # survived without treatment
                        if code in df.loc[person, 'rt_injuries_left_untreated']:
                            if code in df.loc[person, 'rt_injuries_to_cast']:
                                df.loc[person, 'rt_injuries_to_cast'].remove(code)
                            if code in df.loc[person, 'rt_injuries_for_minor_surgery']:
                                df.loc[person, 'rt_injuries_for_minor_surgery'].remove(code)
                            if code in df.loc[person, 'rt_injuries_for_major_surgery']:
                                df.loc[person, 'rt_injuries_for_major_surgery'].remove(code)
                            if code in df.loc[person, 'rt_injuries_to_heal_with_time']:
                                df.loc[person, 'rt_injuries_to_heal_with_time'].remove(code)
                            if code in df.loc[person, 'rt_injuries_to_heal_with_time']:
                                df.loc[person, 'rt_injuries_to_heal_with_time'].remove(code)
                            if code in df.loc[person, 'rt_injuries_for_open_fracture_treatment']:
                                df.loc[person, 'rt_injuries_for_open_fracture_treatment'].remove(code)
                            assert df.loc[person, 'rt_date_to_remove_daly'][columns] > self.sim.date


class RTI_Recovery_Event(RegularEvent, PopulationScopeEventMixin):
    """
    A regular event which checks the recovery date determined by each injury in columns rt_injury_1 through
    rt_injury_8, which is being stored in rt_date_to_remove_daly, a list property with 8 entries. This event
    checks the dates stored in rt_date_to_remove_daly property, when the date matches one of the entries,
    the daly weight is removed and the injury is fully healed.

    The properties changed in this functions is:

    1) rt_date_to_remove_daly - resetting the date to remove the daly weight for each injury once the date is
                                reached in the sim

    2) rt_inj_severity - resetting the person's injury severity once and injury is healed

    3) rt_injuries_to_heal_with_time - resetting the list of injuries that are due to heal over time once healed

    4) rt_injuries_for_minor_surgery - resetting the list of injuries that are treated with minor surgery once
                                       healed
    5) rt_injuries_for_major_surgery - resetting the list of injuries that are treated with major surgery once
                                       healed
    6) rt_injuries_for_open_fracture_treatment - resetting the list of injuries that are treated with open fracture
                                                 treatment once healed
    7) rt_injuries_to_cast - resetting the list of injuries that are treated with fracture cast treatment once healed
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(days=1))
        assert isinstance(module, RTI)

    def apply(self, population):
        road_traffic_injuries = self.module
        df = population.props
        now = self.sim.date
        # # Isolate the relevant population
        any_not_null = df.loc[df.is_alive, 'rt_date_to_remove_daly'].apply(lambda x: pd.notnull(x).any())
        relevant_population = any_not_null.index[any_not_null]
        # Isolate the relevant information
        recovery_dates = df.loc[relevant_population]['rt_date_to_remove_daly']
        default_recovery = [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT]
        # Iterate over all the injured people who are having medical treatment
        for person in recovery_dates.index:
            # Iterate over all the dates in 'rt_date_to_remove_daly'
            for date in df.loc[person, 'rt_date_to_remove_daly']:
                # check that a recovery date hasn't been assigned to the past
                if not pd.isnull(date):
                    assert date >= self.sim.date, 'recovery date assigned to past'
                # check if the recovery date is today
                if date == now:
                    # find the index for the injury which the person has recovered from
                    dateindex = df.loc[person, 'rt_date_to_remove_daly'].index(date)
                    # find the injury code associated with the healed injury
                    code_to_remove = [df.loc[person, f'rt_injury_{dateindex + 1}']]
                    # Set the healed injury recovery data back to the default state
                    df.loc[person, 'rt_date_to_remove_daly'][dateindex] = pd.NaT
                    # Remove the daly weight associated with the healed injury code
                    person_injuries = df.loc[[person], RTI.INJURY_COLUMNS]
                    _, counts = RTI.rti_find_and_count_injuries(person_injuries, self.module.INJURY_CODES[1:])
                    if counts == 0:
                        pass
                    else:
                        road_traffic_injuries.rti_alter_daly_post_treatment(person, code_to_remove)
                    # Check whether all their injuries are healed so the injury properties can be reset
                    if df.loc[person, 'rt_date_to_remove_daly'] == default_recovery:
                        # remove the injury severity as person is uninjured
                        df.loc[person, 'rt_inj_severity'] = "none"
            # Check that the date to remove dalys is removed if the date to remove the daly is today
            assert now not in df.loc[person, 'rt_date_to_remove_daly']
            # finally ensure the reported disability burden is an appropriate value
            if df.loc[person, 'rt_disability'] < 0:
                df.loc[person, 'rt_disability'] = 0
            if df.loc[person, 'rt_disability'] > 1:
                df.loc[person, 'rt_disability'] = 1


# ---------------------------------------------------------------------------------------------------------
#   RTI SPECIFIC HEALTH SYSTEM INTERACTION EVENTS
#
#   Here are all the different Health System Interactions Events that this module will use.
# ---------------------------------------------------------------------------------------------------------
class HSI_RTI_Imaging_Event(HSI_Event, IndividualScopeEventMixin):
    """This HSI event is triggered by the generic first appointments. After first arriving into the health system at
    either level 0 or level 1, should the injured person require a imaging to diagnose their injuries this HSI
    event is caused and x-ray or ct scans are provided as needed"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, RTI)

        self.TREATMENT_ID = 'Rti_Imaging'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'DiagRadio': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

    def apply(self, person_id, squeeze_factor):
        self.sim.population.props.at[person_id, 'rt_diagnosed'] = True
        road_traffic_injuries = self.sim.modules['RTI']
        road_traffic_injuries.rti_injury_diagnosis(person_id, self.EXPECTED_APPT_FOOTPRINT)
        if 'Tomography' in list(self.EXPECTED_APPT_FOOTPRINT.keys()):
            self.ACCEPTED_FACILITY_LEVEL = '3'

    def did_not_run(self, *args, **kwargs):
        pass


class HSI_RTI_Medical_Intervention(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event.
    An appointment of a person who has experienced a road traffic injury, had their injuries diagnosed through A&E
    and now needs treatment.

    This appointment is designed to organise the treatments needed. In the __init__ section the appointment footprint
    is altered to fit the requirements of the person's treatment need. In this section we count the number of
    minor/major surgeries required and determine how long they will be in the health system for. For some injuries,
    the treatment plan is not entirely set into stone and may vary, for example, some skull fractures will need surgery
    whilst some will not. The treatment plan in its entirety is designed here.

    In the apply section, we send those who need surgeries to either HSI_RTI_Major_Surgery or HSI_RTI_Minor_Surgery,
    those who need stitches to HSI_RTI_Suture, those who need burn treatment to HSI_RTI_Burn_Management and those who
    need fracture casts to HSI_RTI_Casting.

    Pain medication is also requested here with HSI_RTI_Acute_Pain_Management.

    The properties changed in this event are:

    rt_injuries_for_major_surgery - the injuries that are determined to be treated by major surgery are stored in
                                    this list property
    rt_injuries_for_minor_surgery - the injuries that are determined to be treated by minor surgery are stored in
                                    this list property
    rt_injuries_to_cast - the injuries that are determined to be treated with fracture casts are stored in this list
                          property
    rt_injuries_for_open_fracture_treatment - the injuries that are determined to be treated with open fractre treatment
                                              are stored in this list property
    rt_injuries_to_heal_with_time - the injuries that are determined to heal with time are stored in this list property

    rt_date_to_remove_daly - recovery dates for the heal with time injuries are set here

    rt_date_death_no_med - the date to check mortality without medical intervention is removed as this person has
                           sought medical care
    rt_med_int - the bool property that shows whether a person has sought medical care or not
    """

    # TODO: include treatment or at least transfer between facilities, e.g. at KCH "Most patients transferred from
    #  either a health center, 2463 (47.2%), or district hospital, 1996 (38.3%)"

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'Rti_MedicalIntervention'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'AccidentsandEmerg': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 8})

        p = module.parameters
        # Load the parameters used in this event
        self.prob_depressed_skull_fracture = p['prob_depressed_skull_fracture']  # proportion of depressed skull
        # fractures in https://doi.org/10.1016/j.wneu.2017.09.084
        self.prob_mild_burns = p['prob_mild_burns']  # proportion of burns accross SSA with TBSA < 10
        # https://doi.org/10.1016/j.burns.2015.04.006
        self.prob_TBI_require_craniotomy = p['prob_TBI_require_craniotomy']
        self.prob_exploratory_laparotomy = p['prob_exploratory_laparotomy']
        self.prob_dislocation_requires_surgery = p['prob_dislocation_requires_surgery']
        self.allowed_interventions = p['allowed_interventions']
        self.prob_perm_disability_with_treatment_severe_TBI = p['prob_perm_disability_with_treatment_severe_TBI']
        # Create an empty list for injuries that are potentially healed without further medical intervention
        self.heal_with_time_injuries = []

    def apply(self, person_id, squeeze_factor):
        road_traffic_injuries = self.sim.modules['RTI']
        df = self.sim.population.props
        p = self.sim.modules['RTI'].parameters
        person = df.loc[person_id]
        # ======================= Design treatment plan, appointment type =============================================
        """ Here, RTI_MedInt designs the treatment plan of the person's injuries, the following determines what the
        major and minor surgery requirements will be

        """
        # Create variables to count how many major or minor surgeries will be required to treat this person
        major_surgery_counts = 0
        minor_surgery_counts = 0
        # Isolate the relevant injury information
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]

        # todo: work out if the amputations need to be included as a swap or if they already exist

        # create a dictionary to store the probability of each possible treatment for applicable injuries, we are
        # assuming that any amputation treatment plan will just be a major surgery for now
        treatment_plans = {
            # Treatment plan options for skull fracture
            '112': [[self.prob_depressed_skull_fracture, 1 - self.prob_depressed_skull_fracture], ['major', 'HWT']],
            '113': [[1], ['HWT']],
            # Treatment plan for facial fractures
            '211': [[1], ['minor']],
            '212': [[1], ['minor']],
            # Treatment plan for rib fractures
            '412': [[1], ['HWT']],
            # Treatment plan for flail chest
            '414': [[1], ['major']],
            # Treatment plan options for foot fractures
            '811': [[p['prob_foot_frac_require_cast'], p['prob_foot_frac_require_maj_surg'],
                     p['prob_foot_frac_require_min_surg'], p['prob_foot_frac_require_amp']],
                    ['cast', 'major', 'minor', 'major']],
            # Treatment plan options for lower leg fractures
            '812': [[p['prob_tib_fib_frac_require_cast'], p['prob_tib_fib_frac_require_maj_surg'],
                     p['prob_tib_fib_frac_require_min_surg'], p['prob_tib_fib_frac_require_traction'],
                     p['prob_tib_fib_frac_require_amp']],
                    ['cast', 'major', 'minor', 'HWT', 'major']],
            # Treatment plan options for femur/hip fractures
            '813a': [[p['prob_femural_fracture_require_major_surgery'],
                      p['prob_femural_fracture_require_minor_surgery'], p['prob_femural_fracture_require_cast'],
                      p['prob_femural_fracture_require_traction'], p['prob_femural_fracture_require_amputation']],
                     ['major', 'minor', 'cast', 'HWT', 'major']],
            # Treatment plan options for femur/hip fractures
            '813c': [[p['prob_femural_fracture_require_major_surgery'],
                      p['prob_femural_fracture_require_minor_surgery'], p['prob_femural_fracture_require_cast'],
                      p['prob_femural_fracture_require_traction'], p['prob_femural_fracture_require_amputation']],
                     ['major', 'minor', 'cast', 'HWT', 'major']],
            # Treatment plan options for pelvis fractures
            '813b': [[p['prob_pelvis_fracture_traction'], p['prob_pelvis_frac_major_surgery'],
                      p['prob_pelvis_frac_minor_surgery'], p['prob_pelvis_frac_cast']],
                     ['HWT', 'major', 'minor', 'cast']],
            # Treatment plan options for open fractures
            '813bo': [[1], ['open']],
            '813co': [[1], ['open']],
            '813do': [[1], ['open']],
            '813eo': [[1], ['open']],
            # Treatment plan options for traumatic brain injuries
            '133a': [[self.prob_TBI_require_craniotomy, 1 - self.prob_TBI_require_craniotomy], ['major', 'HWT']],
            '133b': [[self.prob_TBI_require_craniotomy, 1 - self.prob_TBI_require_craniotomy], ['major', 'HWT']],
            '133c': [[self.prob_TBI_require_craniotomy, 1 - self.prob_TBI_require_craniotomy], ['major', 'HWT']],
            '133d': [[self.prob_TBI_require_craniotomy, 1 - self.prob_TBI_require_craniotomy], ['major', 'HWT']],
            '134a': [[self.prob_TBI_require_craniotomy, 1 - self.prob_TBI_require_craniotomy], ['major', 'HWT']],
            '134b': [[self.prob_TBI_require_craniotomy, 1 - self.prob_TBI_require_craniotomy], ['major', 'HWT']],
            '135': [[self.prob_TBI_require_craniotomy, 1 - self.prob_TBI_require_craniotomy], ['major', 'HWT']],
            # Treatment plan options for abdominal injuries
            '552': [[self.prob_exploratory_laparotomy, 1 - self.prob_exploratory_laparotomy], ['major', 'HWT']],
            '553': [[self.prob_exploratory_laparotomy, 1 - self.prob_exploratory_laparotomy], ['major', 'HWT']],
            '554': [[self.prob_exploratory_laparotomy, 1 - self.prob_exploratory_laparotomy], ['major', 'HWT']],
            # Treatment plan for vertebrae fracture
            '612': [[1], ['HWT']],
            # Treatment plan for dislocations
            '822a': [[p['prob_dis_hip_require_maj_surg'], p['prob_hip_dis_require_traction'],
                      p['prob_dis_hip_require_cast']], ['major', 'HWT', 'cast']],
            '322': [[self.prob_dislocation_requires_surgery, 1 - self.prob_dislocation_requires_surgery],
                    ['minor', 'HWT']],
            '323': [[self.prob_dislocation_requires_surgery, 1 - self.prob_dislocation_requires_surgery],
                    ['minor', 'HWT']],
            '722': [[self.prob_dislocation_requires_surgery, 1 - self.prob_dislocation_requires_surgery],
                    ['minor', 'HWT']],
            # Soft tissue injury in neck treatment plan
            '342': [[1], ['major']],
            '343': [[1], ['major']],
            # Treatment plan for surgical emphysema
            '442': [[1], ['HWT']],
            # Treatment plan for internal bleeding
            '361': [[1], ['major']],
            '363': [[1], ['major']],
            '461': [[1], ['HWT']],
            # Treatment plan for amputations
            '782a': [[1], ['major']],
            '782b': [[1], ['major']],
            '782c': [[1], ['major']],
            '783': [[1], ['major']],
            '882': [[1], ['major']],
            '883': [[1], ['major']],
            '884': [[1], ['major']],
            # Treatment plan for eye injury
            '291': [[1], ['minor']],
            # Treatment plan for soft tissue injury
            '241': [[1], ['minor']],
            # treatment plan for simple fractures and dislocations
            '712a': [[1], ['cast']],
            '712b': [[1], ['cast']],
            '712c': [[1], ['cast']],
            '822b': [[1], ['cast']]

        }
        # store number of open fractures for use later
        open_fractures = 0
        # check if they have an injury for which we need to find the treatment plan for

        for code in treatment_plans.keys():
            _, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, [code])
            if counts > 0:
                treatment_choice = self.module.rng.choice(treatment_plans[code][1], p=treatment_plans[code][0])
                if treatment_choice == 'cast':
                    df.loc[person_id, 'rt_injuries_to_cast'].append(code)
                if treatment_choice == 'major':
                    df.loc[person_id, 'rt_injuries_for_major_surgery'].append(code)
                    major_surgery_counts += 1
                if treatment_choice == 'minor':
                    df.loc[person_id, 'rt_injuries_for_minor_surgery'].append(code)
                    minor_surgery_counts += 1
                if treatment_choice == 'HWT':
                    df.loc[person_id, 'rt_injuries_to_heal_with_time'].append(code)
                if treatment_choice == 'open':
                    open_fractures += 1
                    df.loc[person_id, 'rt_injuries_for_open_fracture_treatment'].append(code)

        # -------------------------------- Spinal cord injury requirements --------------------------------------------
        # Check whether they have a spinal cord injury, if we allow spinal cord surgery capacilities here, ask for a
        # surgery, otherwise make the injury permanent
        codes = ['673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676']
        # Ask if this person has a spinal cord injury
        _, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if (counts > 0) & ('include_spine_surgery' in self.allowed_interventions):
            # if this person has a spinal cord injury and we allow surgeries, determine their exact injury
            actual_injury = np.intersect1d(codes, person_injuries.values)
            # update the number of major surgeries
            major_surgery_counts += 1
            # add the injury to the injuries to be treated by major surgery
            df.loc[person_id, 'rt_injuries_for_major_surgery'].append(actual_injury[0])
        elif counts > 0:
            # if no surgery assume that the person will be permanently disabled
            df.at[person_id, 'rt_perm_disability'] = True
            # Find the column and code where the permanent injury is stored
            column, code = road_traffic_injuries.rti_find_injury_column(person_id=person_id, codes=codes)
            # make the injury permanent by adding a 'P' before the code
            df.loc[person_id, column] = "P" + code
            code = df.loc[person_id, column]
            # find which property the injury is stored in
            columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id, [code])
            for col in columns:
                # schedule the recovery date for the permanent injury for beyond the end of the simulation (making
                # it permanent)
                df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = self.sim.end_date + \
                                                                                DateOffset(days=1)
                assert df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] > self.sim.date

        # --------------------------------- Soft tissue injury in thorax/ lung injury ----------------------------------
        # Check whether they have any soft tissue injuries in the thorax, if so schedule surgery if required else make
        # the injuries heal over time without further medical care
        codes = ['441', '443', '453', '453a', '453b']
        # check if they have chest traume
        _, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if (counts > 0) & ('include_thoroscopy' in self.allowed_interventions):
            # work out the exact injury they have
            actual_injury = np.intersect1d(codes, person_injuries.values)
            # update the number of major surgeries required
            major_surgery_counts += 1
            # add the injury to the injuries to be treated with major surgery so they aren't treated elsewhere
            df.loc[person_id, 'rt_injuries_for_major_surgery'].append(actual_injury[0])

        # -------------------------------- Internal bleeding -----------------------------------------------------------
        # check if they have internal bleeding in the thorax, and if the surgery is available, schedule a major surgery
        codes = ['463']
        _, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if (counts > 0) & ('include_thoroscopy' in self.allowed_interventions):
            # update the number of major surgeries needed
            major_surgery_counts += 1
            # add the injury to the injuries to be treated with major surgery.
            df.loc[person_id, 'rt_injuries_for_major_surgery'].append('463')
        # ================ Determine how long the person will be in hospital based on their ISS score ==================
        inpatient_days = road_traffic_injuries.rti_determine_LOS(person_id)
        # If the patient needs skeletal traction for their injuries they need to stay at minimum 6 weeks,
        # average length of stay for those with femur skeletal traction found from Kramer et al. 2016:
        # https://doi.org/10.1007/s00264-015-3081-3
        # todo: put in complications from femur fractures
        femur_fracture_skeletal_traction_mean_los = p['femur_fracture_skeletal_traction_mean_los']
        other_skeletal_traction_los = p['other_skeletal_traction_los']
        min_los_for_traction = {
            '813c': femur_fracture_skeletal_traction_mean_los,
            '813b': other_skeletal_traction_los,
            '813a': other_skeletal_traction_los,
            '812': other_skeletal_traction_los,
        }
        traction_injuries = [injury for injury in df.loc[person_id, 'rt_injuries_to_heal_with_time'] if injury in
                             min_los_for_traction.keys()]
        if len(traction_injuries) > 0:
            if inpatient_days < min_los_for_traction[traction_injuries[0]]:
                inpatient_days = min_los_for_traction[traction_injuries[0]]

        # Specify the type of bed days needed? not sure if necessary
        self.BEDDAYS_FOOTPRINT.update({'general_bed': inpatient_days})
        # update the expected appointment foortprint
        if inpatient_days > 0:
            self.EXPECTED_APPT_FOOTPRINT.update({'InpatientDays': inpatient_days})
        # ================ Determine whether the person will require ICU days =========================================
        # Percentage of RTIs that required ICU stay 2.7% at KCH : https://doi.org/10.1007/s00268-020-05853-z
        # Percentage of RTIs that require HDU stay 3.3% at KCH
        # Assume for now that ICU admission is entirely dependent on injury severity so that only the 2.7% of most
        # severe injuries get admitted to ICU and the following 3.3% of most severe injuries get admitted to HDU
        # NOTE: LEAVING INPATIENT DAYS IN PLACE TEMPORARILY
        # Seems only one level of care above normal so adjust accordingly
        # self.icu_cut_off_iss_score = 38
        self.hdu_cut_off_iss_score = p['hdu_cut_off_iss_score']
        # Malawi ICU data: doi: 10.1177/0003134820950282
        # General length of stay from Malawi source, not specifically for injuries though
        # mean = 4.8, s.d. = 6, TBI admission mean = 8.4, s.d. = 6.4
        # mortality percentage = 51.2 overall, 50% for TBI admission and 49% for hemorrhage
        # determine the number of ICU days used to treat patient

        if df.loc[person_id, 'rt_ISS_score'] > self.hdu_cut_off_iss_score:
            mean_icu_days = p['mean_icu_days']
            sd_icu_days = p['sd_icu_days']
            mean_tbi_icu_days = p['mean_tbi_icu_days']
            sd_tbi_icu_days = p['sd_tbi_icu_days']
            codes = ['133', '133a', '133b', '133c', '133d' '134', '134a', '134b', '135']
            _, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
            if counts > 0:
                self.icu_days = int(self.module.rng.normal(mean_tbi_icu_days, sd_tbi_icu_days, 1))
            else:
                self.icu_days = int(self.module.rng.normal(mean_icu_days, sd_icu_days, 1))
            # if the number of ICU days is less than zero make it zero
            if self.icu_days < 0:
                self.icu_days = 0
            # update the property showing if a person is in ICU
            df.loc[person_id, 'rt_in_icu_or_hdu'] = True
            # update the bed days footprint
            self.BEDDAYS_FOOTPRINT.update({'general_bed': self.icu_days})
            # store the injury information of patients in ICU
            logger.info(key='ICU_patients',
                        data=person_injuries,
                        description='The injuries of ICU patients')
        # Check that each injury has only one treatment plan assigned to it
        treatment_plan = \
            person['rt_injuries_for_minor_surgery'] + person['rt_injuries_for_major_surgery'] + \
            person['rt_injuries_to_heal_with_time'] + person['rt_injuries_for_open_fracture_treatment'] + \
            person['rt_injuries_to_cast']
        assert len(treatment_plan) == len(set(treatment_plan))

        # Other test admission protocol. Basing ICU admission of whether they have a TBI
        # 17.3% of head injury patients in KCH were admitted to ICU/HDU (7.9 and 9.4% respectively)

        # Injury characteristics of patients admitted to ICU in Tanzania:
        # 97.8% had lacerations
        # 32.4% had fractures
        # 21.5% had TBI
        # 13.1% had abdominal injuries
        # 2.9% had burns
        # 3.8% had 'other' injuries
        # https://doi.org/10.1186/1757-7241-19-61

        if not df.at[person_id, 'is_alive']:
            return self.make_appt_footprint({})
        # Remove the scheduled death without medical intervention
        df.loc[person_id, 'rt_date_death_no_med'] = pd.NaT
        # Isolate relevant injury information
        person = df.loc[person_id]
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        non_empty_injuries = person_injuries[person_injuries != "none"]
        non_empty_injuries = non_empty_injuries.dropna(axis=1)
        injury_columns = person_injuries.keys()
        # Check that those who arrive here are alive and have been through the first generic appointment, and didn't
        # die due to rti
        assert person['rt_diagnosed'], 'person sent here has not been through A and E'
        # Check that those who arrive here have at least one injury
        _, counts = RTI.rti_find_and_count_injuries(person_injuries,
                                                    self.module.PROPERTIES.get('rt_injury_1').categories[1:-1])
        if counts == 0:
            logger.debug(key='rti_general_message',
                         data=f"This is RTIMedicalInterventionEvent person {person_id} asked for treatment but doesn't"
                              f"need it.")
            return self.make_appt_footprint({})

        # log the number of injuries this person has
        logger.info(key='number_of_injuries_in_hospital',
                    data={'number_of_injuries': counts},
                    description='The number of injuries of people in the healthsystem')
        # update the model's properties to reflect that this person has sought medical care
        df.at[person_id, 'rt_med_int'] = True
        # =============================== Make 'healed with time' injuries disappear ===================================
        # these are the injuries actually treated in this HSI
        heal_with_time_recovery_times_in_days = {
            # using estimated 6 weeks PLACEHOLDER FOR neck dislocations
            '322': 42,
            '323': 42,
            # using estimated 12 weeks placeholder for dislocated shoulders
            '722': 84,
            # using estimated 2 month placeholder for dislocated knees
            '822a': 60,
            # using estimated 7 weeks PLACEHOLDER FOR SKULL FRACTURE
            '112': 49,
            '113': 49,
            # using estimated 5 weeks PLACEHOLDER FOR rib FRACTURE
            '412': 35,
            # using estimated 9 weeks PLACEHOLDER FOR Vertebrae FRACTURE
            '612': 63,
            # using estimated 9 weeks PLACEHOLDER FOR skeletal traction for tibia/fib
            '812': 63,
            # using estimated 9 weeks PLACEHOLDER FOR skeletal traction for hip
            '813a': 63,
            # using estimated 9 weeks PLACEHOLDER FOR skeletal traction for pelvis
            '813b': 63,
            # using estimated 9 weeks PLACEHOLDER FOR skeletal traction for femur
            '813c': 63,
            # using estimated 3 month PLACEHOLDER FOR abdominal trauma
            '552': 90,
            '553': 90,
            '554': 90,
            # using 1 week placeholder for surgical emphysema
            '442': 7,
            # 2 week placeholder for chest wall bruising
            '461': 14

        }
        tbi = ['133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135']
        if len(df.at[person_id, 'rt_injuries_to_heal_with_time']) > 0:
            # check whether the heal with time injuries include dislocations, which may have been sent to surgery
            for code in person['rt_injuries_to_heal_with_time']:
                # temporarily dealing with TBI heal dates seporately
                if code in tbi:
                    pass
                else:
                    columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, [code])[0])
                    df.loc[person_id, 'rt_date_to_remove_daly'][columns] = \
                        self.sim.date + DateOffset(days=heal_with_time_recovery_times_in_days[code])
                    assert df.loc[person_id, 'rt_date_to_remove_daly'][columns] > self.sim.date
            heal_with_time_codes = []
            # Check whether the heal with time injury is a skull fracture, which may have been sent to surgery
            tbi = ['133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135']
            tbi_injury = [injury for injury in tbi if injury in person['rt_injuries_to_heal_with_time']]
            if len(tbi_injury) > 0:
                columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, tbi_injury)[0])
                # ask if this injury will be permanent
                perm_injury = self.module.rng.random_sample(size=1)
                if perm_injury < self.prob_perm_disability_with_treatment_severe_TBI:
                    # injury is permanent so find where the injury is located
                    column, code = road_traffic_injuries.rti_find_injury_column(person_id=person_id, codes=tbi_injury)
                    # put a P in front of the code to show it will be a perm injury
                    df.loc[person_id, column] = "P" + code
                    # store the heal with time injury in heal_with_time_codes
                    heal_with_time_codes.append("P" + code)
                    # update the property 'rt_injuries_to_heal_with_time' to contain the new code
                    df.loc[person_id, 'rt_injuries_to_heal_with_time'].remove(code)
                    df.loc[person_id, 'rt_injuries_to_heal_with_time'].append("P" + code)
                    # schedule a recover date beyond this simulation's end
                    df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.end_date + DateOffset(days=1)
                    assert df.loc[person_id, 'rt_date_to_remove_daly'][columns] > self.sim.date
                else:
                    heal_with_time_codes.append(tbi_injury[0])
                    # using estimated 6 months PLACEHOLDER FOR TRAUMATIC BRAIN INJURY
                    df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(months=6)
                    assert df.loc[person_id, 'rt_date_to_remove_daly'][columns] > self.sim.date
            # swap potentially swappable codes
            swapping_codes = RTI.SWAPPING_CODES[:]
            # remove codes that will be treated elsewhere
            for code in person['rt_injuries_for_minor_surgery']:
                if code in swapping_codes:
                    swapping_codes.remove(code)
            for code in person['rt_injuries_for_major_surgery']:
                if code in swapping_codes:
                    swapping_codes.remove(code)
            for code in person['rt_injuries_to_cast']:
                if code in swapping_codes:
                    swapping_codes.remove(code)
            for code in person['rt_injuries_for_open_fracture_treatment']:
                if code in swapping_codes:
                    swapping_codes.remove(code)
            # drop injuries potentially treated elsewhere
            codes_to_swap = [code for code in heal_with_time_codes if code in swapping_codes]
            if len(codes_to_swap) > 0:
                road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, codes_to_swap)
            # check every heal with time injury has a recovery date associated with it
            for code in person['rt_injuries_to_heal_with_time']:
                columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, [code])
                                                 [0])
                assert not pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]), \
                    'no recovery date given for this injury' + code
                # check injury heal time is in the future
                assert df.loc[person_id, 'rt_date_to_remove_daly'][columns] > self.sim.date
                # remove code from heal with time injury list

            df.loc[person_id, 'rt_injuries_to_heal_with_time'].clear()
        # schedule treatments of all injuries here

        # ======================================= Schedule surgeries ==================================================
        # Schedule the surgeries by calling the functions rti_do_for_major/minor_surgeries which in turn schedules the
        # surgeries, people can have multiple surgeries scheduled so schedule surgeries seperate to the rest of the
        # treatment plans
        # Check they haven't died from another source
        if not pd.isnull(df.loc[person_id, 'cause_of_death']):
            pass
        else:
            if major_surgery_counts > 0:
                # schedule major surgeries
                for count in range(0, major_surgery_counts):
                    road_traffic_injuries.rti_do_for_major_surgeries(person_id=person_id, count=count)
            if minor_surgery_counts > 0:
                # shedule minor surgeries
                for count in range(0, minor_surgery_counts):
                    road_traffic_injuries.rti_do_for_minor_surgeries(person_id=person_id, count=count)
        # Schedule all other treatments here
        # Fractures are sometimes treated via major/minor surgeries. Need to establish which injuries are due to be
        # treated via fracture cast
        frac_codes = ['712', '712a', '712b', '712c', '811', '812', '813a', '813b', '813c', '822a', '822b']
        p = df.loc[person_id]
        codes_treated_elsewhere = \
            p['rt_injuries_for_minor_surgery'] + p['rt_injuries_for_major_surgery'] + \
            p['rt_injuries_to_heal_with_time'] + p['rt_injuries_for_open_fracture_treatment']
        frac_codes = [code for code in frac_codes if code not in codes_treated_elsewhere]
        # Create a lookup table for treatment methods and the injuries that they are due to treat
        single_option_treatments = {
            'suture': ['1101', '2101', '3101', '4101', '5101', '7101', '8101'],
            'burn': ['1114', '2114', '3113', '4113', '5113', '7113', '8113'],
            'fracture': frac_codes,
            'tetanus': ['1101', '2101', '3101', '4101', '5101', '7101', '8101', '1114', '2114', '3113', '4113', '5113',
                        '7113', '8113'],
            'pain': self.module.PROPERTIES.get('rt_injury_1').categories[1:],
            'open': ['813bo', '813co', '813do', '813eo']
        }
        # find this person's untreated injuries
        untreated_injury_cols = []
        idx_for_untreated_injuries = []
        for index, time in enumerate(df.loc[person_id, 'rt_date_to_remove_daly']):
            if pd.isnull(time):
                idx_for_untreated_injuries.append(index)
        for idx in idx_for_untreated_injuries:
            untreated_injury_cols.append(RTI.INJURY_COLUMNS[idx])
        person_untreated_injuries = df.loc[[person_id], untreated_injury_cols]

        for treatment in single_option_treatments:
            # If a person has an injury that hasn't been deliberately left untreated then schedule a treatment, or if
            # the treatment is pain management
            untreated_injuries = list(non_empty_injuries.values[0])
            deliberately_untreated_injuries = df.loc[person_id, 'rt_injuries_left_untreated']
            injuries_left_to_treat = [injury for injury in untreated_injuries if injury not in
                                      deliberately_untreated_injuries]
            no_injuries_for_this_treatment = (len(set(injuries_left_to_treat) &
                                                  set(single_option_treatments[treatment])) == 0)
            condition_to_skip = no_injuries_for_this_treatment & (treatment != 'pain')
            if condition_to_skip:
                pass
            else:
                _, inj_counts = road_traffic_injuries.rti_find_and_count_injuries(person_untreated_injuries,
                                                                                  single_option_treatments[treatment])
                if inj_counts > 0 & df.loc[person_id, 'is_alive']:
                    if treatment == 'suture':
                        road_traffic_injuries.rti_ask_for_suture_kit(person_id=person_id)
                    if treatment == 'burn':
                        road_traffic_injuries.rti_ask_for_burn_treatment(person_id=person_id)
                    if treatment == 'fracture':
                        road_traffic_injuries.rti_ask_for_fracture_casts(person_id=person_id)
                    if treatment == 'tetanus':
                        road_traffic_injuries.rti_ask_for_tetanus(person_id=person_id)
                    if treatment == 'pain':
                        road_traffic_injuries.rti_acute_pain_management(person_id=person_id)
                    if treatment == 'open':
                        road_traffic_injuries.rti_ask_for_open_fracture_treatment(person_id=person_id,
                                                                                  counts=open_fractures)

        treatment_plan = \
            p['rt_injuries_for_minor_surgery'] + p['rt_injuries_for_major_surgery'] + \
            p['rt_injuries_to_heal_with_time'] + p['rt_injuries_for_open_fracture_treatment'] + \
            p['rt_injuries_to_cast']
        # make sure injuries are treated in one place only
        assert len(treatment_plan) == len(set(treatment_plan))
        # ============================== Ask if they die even with treatment ===========================================
        self.sim.schedule_event(RTI_Medical_Intervention_Death_Event(self.module, person_id), self.sim.date +
                                DateOffset(days=inpatient_days))
        logger.debug(key='rti_general_message',
                     data=f"This is RTIMedicalInterventionEvent scheduling a potential death on date "
                          f"{self.sim.date + DateOffset(days=inpatient_days)} (end of treatment) for person "
                          f"{person_id}")

    def did_not_run(self):
        person_id = self.target
        df = self.sim.population.props
        logger.debug(key='rti_general_message',
                     data=f"RTIMedicalInterventionEvent did not run on date {self.sim.date} (end of treatment) for "
                          f"person {person_id}")
        injurycodes = {'First injury': df.loc[person_id, 'rt_injury_1'],
                       'Second injury': df.loc[person_id, 'rt_injury_2'],
                       'Third injury': df.loc[person_id, 'rt_injury_3'],
                       'Fourth injury': df.loc[person_id, 'rt_injury_4'],
                       'Fifth injury': df.loc[person_id, 'rt_injury_5'],
                       'Sixth injury': df.loc[person_id, 'rt_injury_6'],
                       'Seventh injury': df.loc[person_id, 'rt_injury_7'],
                       'Eight injury': df.loc[person_id, 'rt_injury_8']}
        logger.debug(key='rti_injury_profile_of_untreated_person', data=injurycodes)
        # reset the treatment plan
        df.loc[person_id, 'rt_injuries_for_major_surgery'] = []
        df.loc[person_id, 'rt_injuries_for_minor_surgery'] = []
        df.loc[person_id, 'rt_injuries_to_cast'] = []
        df.loc[person_id, 'rt_injuries_to_heal_with_time'] = []
        df.loc[person_id, 'rt_injuries_for_open_fracture_treatment'] = []


class HSI_RTI_Shock_Treatment(HSI_Event, IndividualScopeEventMixin):
    """
    This HSI event handles the process of treating hypovolemic shock, as recommended by the pediatric
    handbook for Malawi and (TODO: FIND ADULT REFERENCE)
    Currently this HSI_Event is described only and not used, as I still need to work out how to model the occurrence
    of shock
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, RTI)

        self.TREATMENT_ID = 'Rti_ShockTreatment'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'AccidentsandEmerg': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        # determine if this is a child
        if df.loc[person_id, 'age_years'] < 15:
            is_child = True
        else:
            is_child = False
        if not df.at[person_id, 'is_alive']:
            return self.make_appt_footprint({})
        get_item_code = self.sim.modules['HealthSystem'].get_item_code_from_item_name
        # TODO: find a more complete list of required consumables for adults
        if is_child:
            self.module.item_codes_for_consumables_required['shock_treatment_child'] = {
                get_item_code("ringer's lactate (Hartmann's solution), 1000 ml_12_IDA"): 1,
                get_item_code("Dextrose (glucose) 5%, 1000ml_each_CMST"): 1,
                get_item_code('Cannula iv  (winged with injection pot) 18_each_CMST'): 1,
                get_item_code('Blood, one unit'): 1,
                get_item_code("Oxygen, 1000 liters, primarily with oxygen cylinders"): 1
            }
            is_cons_available = self.get_consumables(
                self.module.item_codes_for_consumables_required['shock_treatment_child']
            )
        else:
            self.module.item_codes_for_consumables_required['shock_treatment_adult'] = {
                get_item_code("ringer's lactate (Hartmann's solution), 1000 ml_12_IDA"): 1,
                get_item_code('Cannula iv  (winged with injection pot) 18_each_CMST'): 1,
                get_item_code('Blood, one unit'): 1,
                get_item_code("Oxygen, 1000 liters, primarily with oxygen cylinders"): 1
            }
            is_cons_available = self.get_consumables(
                self.module.item_codes_for_consumables_required['shock_treatment_adult']
            )

        if is_cons_available:
            logger.debug(key='rti_general_message',
                         data=f"Hypovolemic shock treatment available for person {person_id}")
            df.at[person_id, 'rt_in_shock'] = False
        else:
            self.sim.modules['RTI'].schedule_hsi_event_for_tomorrow(self)
            return self.make_appt_footprint({})

    def did_not_run(self):
        # Assume that untreated shock leads to death for now
        # Schedule the death
        df = self.sim.population.props
        person_id = self.target

        df.at[person_id, 'rt_death_from_shock'] = True
        self.sim.modules['Demography'].do_death(individual_id=person_id, cause="RTI_death_shock",
                                                originating_module=self.module)
        # Log the death
        logger.debug(key='rti_general_message',
                     data=f"This is RTI_Shock_Treatment scheduling a death for person {person_id} who did not recieve "
                          f"treatment for shock on {self.sim.date}"
                     )


class HSI_RTI_Fracture_Cast(HSI_Event, IndividualScopeEventMixin):
    """
    This HSI event handles fracture casts/giving slings for those who need it. The HSI event tests whether the injured
    person has an appropriate injury code, determines how many fractures the person and then requests fracture
    treatment as required.

    The injury codes dealt with in this HSI event are:
    '712a' - broken clavicle, scapula, humerus
    '712b' - broken hand/wrist
    '712c' - broken radius/ulna
    '811' - Fractured foot
    '812' - broken tibia/fibula
    '813a' - Broken hip
    '813b' - broken pelvis
    '813c' - broken femur

    '822a' - dislocated hip
    '822b' - dislocated knee

    The properties altered by this function are
    rt_date_to_remove_daly - setting recovery dates for injuries treated with fracture casts
    rt_injuries_to_cast - once treated the codes used to denote injuries to be treated by fracture casts are removed
                          from the list of injuries due to be treated with fracture casts
    rt_med_int - the property used to denote whether a person getting treatment for road traffic injuries

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, RTI)

        self.TREATMENT_ID = 'Rti_FractureCast'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'AccidentsandEmerg': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

    def apply(self, person_id, squeeze_factor):
        # Get the population and health system
        df = self.sim.population.props
        p = df.loc[person_id]
        # if the person isn't alive return a blank footprint
        if not df.at[person_id, 'is_alive']:
            return self.make_appt_footprint({})
        # get a shorthand reference to RTI and consumables modules
        road_traffic_injuries = self.sim.modules['RTI']
        get_item_code = self.sim.modules['HealthSystem'].get_item_code_from_item_name
        # isolate the relevant injury information
        # Find the untreated injuries
        untreated_injury_cols = \
            [RTI.INJURY_COLUMNS[i] for i, v in enumerate(df.at[person_id, 'rt_date_to_remove_daly']) if pd.isnull(v)]
        person_injuries = df.loc[[person_id], untreated_injury_cols]
        # check if they have a fracture that requires a cast
        codes = ['712b', '712c', '811', '812', '813a', '813b', '813c', '822a', '822b']
        _, fracturecastcounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        # check if they have a fracture that requires a sling
        codes = ['712a']
        _, slingcounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        # Check the person sent here is alive, been through the generic first appointment,
        # been through the RTI med intervention
        assert p['rt_diagnosed'], 'person sent here has not been diagnosed'
        assert p['rt_med_int'], 'person sent here has not been treated'
        # Check that the person sent here has an injury treated by this module
        assert fracturecastcounts + slingcounts > 0
        # Check this person has an injury intended to be treated here
        assert len(p['rt_injuries_to_cast']) > 0
        # Check this injury assigned to be treated here is actually had by the person
        assert all(injuries in person_injuries.values for injuries in p['rt_injuries_to_cast'])
        # If they have a fracture that needs a cast, ask for plaster of paris
        self.module.item_codes_for_consumables_required['fracture_treatment'] = {
            get_item_code('Plaster of Paris (POP) 10cm x 7.5cm slab_12_CMST'): fracturecastcounts,
            get_item_code('Bandage, crepe 7.5cm x 1.4m long , when stretched'): slingcounts,
        }
        is_cons_available = self.get_consumables(
            self.module.item_codes_for_consumables_required['fracture_treatment']
        )
        # if the consumables are available then the appointment can run
        if is_cons_available:
            logger.debug(key='rti_general_message',
                         data=f"Fracture casts available for person %d's {fracturecastcounts + slingcounts} fractures, "
                              f"{person_id}"
                         )
            # update the property rt_med_int to indicate they are recieving treatment
            df.at[person_id, 'rt_med_int'] = True
            # Find the persons injuries
            non_empty_injuries = person_injuries[person_injuries != "none"]
            non_empty_injuries = non_empty_injuries.dropna(axis=1)
            # Find the injury codes treated by fracture casts/slings
            codes = ['712a', '712b', '712c', '811', '812', '813a', '813b', '813c', '822a', '822b']
            # Some TLO codes have daly weights associated with treated and non-treated injuries, copy the list of
            # swapping codes
            swapping_codes = RTI.SWAPPING_CODES[:]
            # find the relevant swapping codes for this treatment
            swapping_codes = [code for code in swapping_codes if code in codes]
            # remove codes that will be treated elsewhere
            injuries_treated_elsewhere = \
                p['rt_injuries_for_minor_surgery'] + p['rt_injuries_for_major_surgery'] + \
                p['rt_injuries_to_heal_with_time'] + p['rt_injuries_for_open_fracture_treatment']
            # remove codes that are being treated elsewhere
            swapping_codes = [code for code in swapping_codes if code not in injuries_treated_elsewhere]
            # find any potential codes this person has that are due to be swapped and then swap with
            # rti_swap_injury_daly_upon_treatment
            relevant_codes = np.intersect1d(non_empty_injuries.values, swapping_codes)
            if len(relevant_codes) > 0:
                road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, relevant_codes)
            # Find the injuries that have been treated and then schedule a recovery date
            columns, codes = \
                road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id, df.loc[person_id,
                                                                                                 'rt_injuries_to_cast'])
            # check that for each injury to be treated by this event we have a corresponding column
            assert len(columns) == len(df.loc[person_id, 'rt_injuries_to_cast'])
            # iterate over the columns of injuries treated here and assign a recovery date
            for col in columns:
                # todo: update this with recovery times for casted broken hips/pelvis/femurs
                # todo: update this with recovery times for casted dislocated hip
                df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = self.sim.date + \
                                                                                DateOffset(weeks=7)
                # make sure the assigned injury recovery date is in the future
                assert df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] > self.sim.date
            person_injuries = df.loc[person_id, RTI.INJURY_COLUMNS]
            injury_columns = person_injuries.keys()
            for code in df.loc[person_id, 'rt_injuries_to_cast']:
                columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, [code])[0])
                assert not pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]), \
                    'no recovery date given for this injury'
            # remove codes from fracture cast list
            df.loc[person_id, 'rt_injuries_to_cast'].clear()
            df.loc[person_id, 'rt_date_death_no_med'] = pd.NaT
        else:
            self.sim.modules['RTI'].schedule_hsi_event_for_tomorrow(self)
            if pd.isnull(df.loc[person_id, 'rt_date_death_no_med']):
                df.loc[person_id, 'rt_date_death_no_med'] = self.sim.date + DateOffset(days=7)
            logger.debug(key='rti_general_message',
                         data=f"Person {person_id} has {fracturecastcounts + slingcounts} fractures without treatment"
                         )
            return self.make_appt_footprint({})

    def did_not_run(self):
        person_id = self.target

        logger.debug(key='rti_general_message',
                     data=f"Fracture casts unavailable for person {person_id}")


class HSI_RTI_Open_Fracture_Treatment(HSI_Event, IndividualScopeEventMixin):
    """
    This HSI event handles fracture casts/giving slings for those who need it. The HSI event tests whether the injured
    person has an appropriate injury code, determines how many fractures the person and then requests fracture
    treatment as required.

    The injury codes dealt with in this HSI event are:
    '813bo' - Open fracture of the pelvis
    '813co' - Open fracture of the femur
    '813do' - Open fracture of the foot
    '813eo' - Open fracture of the tibia/fibula/ankle/patella

    The properties altered by this function are:
    rt_med_int - to denote that this person is recieving treatment
    rt_injuries_for_open_fracture_treatment - removing codes that have been treated by open fracture treatment
    rt_date_to_remove_daly - to schedule recovery dates for open fractures that have recieved treatment
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, RTI)

        self.TREATMENT_ID = 'Rti_OpenFractureTreatment'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'MinorSurg': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return self.make_appt_footprint({})
        road_traffic_injuries = self.sim.modules['RTI']
        get_item_code = self.sim.modules['HealthSystem'].get_item_code_from_item_name
        # isolate the relevant injury information
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        # check if they have a fracture that requires a cast
        codes = ['813bo', '813co', '813do', '813eo']
        _, open_fracture_counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        assert open_fracture_counts > 0
        # Check the person sent here is alive, been through the generic first appointment,
        # been through the RTI med intervention
        assert df.loc[person_id, 'rt_diagnosed'], 'person sent here has not been diagnosed'
        assert df.loc[person_id, 'rt_med_int'], 'person sent here has not been treated'

        # If they have an open fracture, ask for consumables to treat fracture
        if open_fracture_counts > 0:
            self.module.item_codes_for_consumables_required['open_fracture_treatment'] = {
                get_item_code('Ceftriaxone 1g, PFR_each_CMST'): 1,
                get_item_code('Cetrimide 15% + chlorhexidine 1.5% solution.for dilution _5_CMST'): 1,
                get_item_code("Gauze, absorbent 90cm x 40m_each_CMST"): 1,
                get_item_code('Suture pack'): 1,
            }
            # If wound is "grossly contaminated" administer Metronidazole
            # todo: parameterise the probability of wound contamination
            p = self.module.parameters
            prob_open_fracture_contaminated = p['prob_open_fracture_contaminated']
            rand_for_contamination = self.module.rng.random_sample(size=1)
            if rand_for_contamination < prob_open_fracture_contaminated:
                self.module.item_codes_for_consumables_required['open_fracture_treatment'].update(
                    {get_item_code('Metronidazole, injection, 500 mg in 100 ml vial'): 1}
                )
        # Check that there are enough consumables to treat this person's fractures
        is_cons_available = self.get_consumables(
            self.module.item_codes_for_consumables_required['open_fracture_treatment']
        )

        if is_cons_available:
            logger.debug(key='rti_general_message',
                         data=f"Fracture casts available for person {person_id} {open_fracture_counts} open fractures"
                         )
            person = df.loc[person_id]
            # update the dataframe to show this person is recieving treatment
            df.loc[person_id, 'rt_med_int'] = True
            # Find the persons injuries to be treated
            non_empty_injuries = person['rt_injuries_for_open_fracture_treatment']
            columns, code = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(
                person_id, non_empty_injuries
            )
            # Some TLO codes have daly weights associated with treated and non-treated injuries
            if code[0] == '813bo':
                road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, code[0])
            # Schedule a recovery date for the injury
            # estimated 6-9 months recovery times for open fractures
            df.loc[person_id, 'rt_date_to_remove_daly'][int(columns[0][-1]) - 1] = self.sim.date + DateOffset(months=7)
            assert df.loc[person_id, 'rt_date_to_remove_daly'][int(columns[0][-1]) - 1] > self.sim.date
            assert not pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][int(columns[0][-1]) - 1]), \
                'no recovery date given for this injury'
            df.loc[person_id, 'rt_date_death_no_med'] = pd.NaT
            # remove code from open fracture list
            if code[0] in df.loc[person_id, 'rt_injuries_for_open_fracture_treatment']:
                df.loc[person_id, 'rt_injuries_for_open_fracture_treatment'].remove(code[0])
        else:
            self.sim.modules['RTI'].schedule_hsi_event_for_tomorrow(self)
            if pd.isnull(df.loc[person_id, 'rt_date_death_no_med']):
                df.loc[person_id, 'rt_date_death_no_med'] = self.sim.date + DateOffset(days=7)
            logger.debug(key='rti_general_message',
                         data=f"Person {person_id}'s has {open_fracture_counts} open fractures without treatment",
                         )

    def did_not_run(self):
        person_id = self.target

        logger.debug(key='rti_general_message',
                     data=f"Open fracture treatment unavailable for person {person_id}")


class HSI_RTI_Suture(HSI_Event, IndividualScopeEventMixin):
    """
    This HSI event handles lacerations giving suture kits for those who need it. The HSI event tests whether the injured
    person has an appropriate injury code, determines how many lacerations the person and then requests suture kits
     as required.


    The codes dealt with are:
    '1101' - Laceration to the head
    '2101' - Laceration to the face
    '3101' - Laceration to the neck
    '4101' - Laceration to the thorax
    '5101' - Laceration to the abdomen
    '7101' - Laceration to the upper extremity
    '8101' - Laceration to the lower extremity

    The properties altered by this function are:
    rt_med_int - to denote that this person is recieving treatment
    rt_date_to_remove_daly - to schedule recovery dates for lacerations treated in this hsi
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, RTI)

        self.TREATMENT_ID = 'Rti_Suture'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({
            ('Under5OPD' if self.sim.population.props.at[person_id, "age_years"] < 5 else 'Over5OPD'): 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

    def apply(self, person_id, squeeze_factor):
        get_item_code = self.sim.modules['HealthSystem'].get_item_code_from_item_name
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return self.make_appt_footprint({})
        road_traffic_injuries = self.sim.modules['RTI']

        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        codes = ['1101', '2101', '3101', '4101', '5101', '7101', '8101']
        _, lacerationcounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        # Check the person sent here didn't die due to rti, has been through A&E, through Med int
        assert df.loc[person_id, 'rt_diagnosed'], 'person sent here has not been through A and E'
        assert df.loc[person_id, 'rt_med_int'], 'person sent here has not been treated'
        # Check that the person sent here has an injury that is treated by this HSI event
        assert lacerationcounts > 0
        if lacerationcounts > 0:
            self.module.item_codes_for_consumables_required['laceration_treatment'] = {
                get_item_code('Suture pack'): lacerationcounts,
                get_item_code('Cetrimide 15% + chlorhexidine 1.5% solution.for dilution _5_CMST'): lacerationcounts,

            }
            # check the number of suture kits required and request them
            is_cons_available = self.get_consumables(
                self.module.item_codes_for_consumables_required['laceration_treatment']
            )

            # Availability of consumables determines if the intervention is delivered...
            if is_cons_available:
                logger.debug(key='rti_general_message',
                             data=f"This facility has open wound treatment available which has been used for person "
                                  f"{person_id}."
                             )
                logger.debug(key='rti_general_message',
                             data=f"This facility treated their {lacerationcounts} open wounds")

                columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id, codes)
                for col in columns:
                    # heal time for lacerations is roughly two weeks according to:
                    # https://www.facs.org/~/media/files/education/patient%20ed/wound_lacerations.ashx#:~:text=of%20
                    # wound%20and%20your%20general,have%20a%20weakened%20immune%20system.
                    df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = self.sim.date + \
                                                                                    DateOffset(days=14)
                    assert df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] > self.sim.date
                df.loc[person_id, 'rt_date_death_no_med'] = pd.NaT
            else:
                self.sim.modules['RTI'].schedule_hsi_event_for_tomorrow(self)
                if pd.isnull(df.loc[person_id, 'rt_date_death_no_med']):
                    df.loc[person_id, 'rt_date_death_no_med'] = self.sim.date + DateOffset(days=7)
                logger.debug(key='rti_general_message',
                             data="This facility has no treatment for open wounds available.")
                return self.make_appt_footprint({})

    def did_not_run(self):
        person_id = self.target

        logger.debug(key='rti_general_message',
                     data=f"Suture kits unavailable for person {person_id}")


class HSI_RTI_Burn_Management(HSI_Event, IndividualScopeEventMixin):
    """
    This HSI event handles burns giving treatment for those who need it. The HSI event tests whether the injured
    person has an appropriate injury code, determines how many burns the person and then requests appropriate treatment
     as required.



    The codes dealt with in this HSI event are:
    '1114' - Burns to the head
    '2114' - Burns to the face
    '3113' - Burns to the neck
    '4113' - Burns to the thorax
    '5113' - Burns to the abdomen
    '7113' - Burns to the upper extremities
    '8113' - Burns to the lower extremities

    The properties treated by this module are:
    rt_med_int - to denote that this person is recieving treatment for their injuries
    rt_date_to_remove_daly - to schedule recovery dates for injuries treated here
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, RTI)

        self.TREATMENT_ID = 'Rti_BurnManagement'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'MinorSurg': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 1})

        p = self.module.parameters
        self.prob_mild_burns = p['prob_mild_burns']

    def apply(self, person_id, squeeze_factor):
        get_item_code = self.sim.modules['HealthSystem'].get_item_code_from_item_name
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return self.make_appt_footprint({})
        road_traffic_injuries = self.sim.modules['RTI']

        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        codes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113']
        _, burncounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        # check the person sent here has an injury treated by this module
        assert burncounts > 0
        # check the person sent here didn't die due to rti, has been through A and E and had RTI_med_int
        assert df.loc[person_id, 'rt_diagnosed'], 'this person has not been through a and e'
        assert df.loc[person_id, 'rt_med_int'], 'this person has not been treated'
        if burncounts > 0:
            # Request materials for burn treatment
            self.module.item_codes_for_consumables_required['burn_treatment'] = {
                get_item_code("Gauze, absorbent 90cm x 40m_each_CMST"): burncounts,
                get_item_code('Cetrimide 15% + chlorhexidine 1.5% solution.for dilution _5_CMST'): burncounts,

            }
            possible_large_TBSA_burn_codes = ['7113', '8113', '4113', '5113']
            idx2, bigburncounts = \
                road_traffic_injuries.rti_find_and_count_injuries(person_injuries, possible_large_TBSA_burn_codes)
            random_for_severe_burn = self.module.rng.random_sample(size=1)
            # ======================== If burns severe enough then give IV fluid replacement ===========================
            if (burncounts > 1) or ((len(idx2) > 0) & (random_for_severe_burn > self.prob_mild_burns)):
                # check if they have multiple burns, which implies a higher burned total body surface area (TBSA) which
                # will alter the treatment plan
                self.module.item_codes_for_consumables_required['burn_treatment'].update(
                    {get_item_code("ringer's lactate (Hartmann's solution), 1000 ml_12_IDA"): 1}
                )

            is_cons_available = self.get_consumables(
                self.module.item_codes_for_consumables_required['burn_treatment']
            )
            if is_cons_available:
                logger.debug(key='rti_general_message',
                             data=f"This facility has burn treatment available which has been used for person "
                                  f"{person_id}")
                logger.debug(key='rti_general_message',
                             data=f"This facility treated their {burncounts} burns")
                df.at[person_id, 'rt_med_int'] = True
                person = df.loc[person_id]
                injury_columns = person_injuries.keys()
                columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, codes)[0])
                # estimate burns take 4 weeks to heal
                df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=4)
                assert df.loc[person_id, 'rt_date_to_remove_daly'][columns] > self.sim.date
                persons_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
                non_empty_injuries = persons_injuries[persons_injuries != "none"]
                non_empty_injuries = non_empty_injuries.dropna(axis=1)
                swapping_codes = RTI.SWAPPING_CODES[:]
                swapping_codes = [code for code in swapping_codes if code in codes]
                # remove codes that will be treated elsewhere
                treatment_plan = (
                    person['rt_injuries_for_major_surgery'] + person['rt_injuries_for_minor_surgery'] +
                    person['rt_injuries_for_minor_surgery'] + person['rt_injuries_to_cast'] +
                    person['rt_injuries_to_heal_with_time'] + person['rt_injuries_for_open_fracture_treatment']
                )
                swapping_codes = [code for code in swapping_codes if code not in treatment_plan]
                relevant_codes = np.intersect1d(non_empty_injuries.values, swapping_codes)
                if len(relevant_codes) > 0:
                    road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, relevant_codes)

                assert df.loc[person_id, 'rt_date_to_remove_daly'][columns] > self.sim.date, \
                    'recovery date assigned to past'
                df.loc[person_id, 'rt_date_death_no_med'] = pd.NaT
            else:
                self.sim.modules['RTI'].schedule_hsi_event_for_tomorrow(self)
                if pd.isnull(df.loc[person_id, 'rt_date_death_no_med']):
                    df.loc[person_id, 'rt_date_death_no_med'] = self.sim.date + DateOffset(days=7)
                logger.debug(key='rti_general_message',
                             data="This facility has no treatment for burns available.")

    def did_not_run(self):
        person_id = self.target

        logger.debug(key='rti_general_message',
                     data=f"Burn treatment unavailable for person {person_id}")


class HSI_RTI_Tetanus_Vaccine(HSI_Event, IndividualScopeEventMixin):
    """
    This HSI event handles tetanus vaccine requests, the idea being that by separating these from the burn and
    laceration and burn treatments, those treatments can go ahead without the availability of tetanus stopping the event

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, RTI)

        self.TREATMENT_ID = 'Rti_TetanusVaccine'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'EPI': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return self.make_appt_footprint({})
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        # check the person sent here hasn't died due to rti, has been through A and E and had RTI_med_int
        assert df.loc[person_id, 'rt_diagnosed'], 'This person has not been through a and e'
        assert df.loc[person_id, 'rt_med_int'], 'This person has not been through rti med int'
        # check the person sent here has an injury treated by this module
        codes_for_tetanus = ['1101', '2101', '3101', '4101', '5101', '7101', '8101',
                             '1114', '2114', '3113', '4113', '5113', '7113', '8113']
        _, counts = RTI.rti_find_and_count_injuries(person_injuries, codes_for_tetanus)
        if counts == 0:
            logger.debug(key='rti_general_message',
                         data=f"This is RTI tetanus vaccine person {person_id} asked for treatment but doesn't"
                              f"need it.")
            return self.make_appt_footprint({})
        # If they have a laceration/burn ask request the tetanus vaccine
        if counts > 0:
            get_item_code = self.sim.modules['HealthSystem'].get_item_code_from_item_name
            self.module.item_codes_for_consumables_required['tetanus_treatment'] = {
                get_item_code('Tetanus toxoid, injection'): 1
            }
            is_tetanus_available = self.get_consumables(
                self.module.item_codes_for_consumables_required['tetanus_treatment']
            )
            if is_tetanus_available:
                logger.debug(key='rti_general_message',
                             data=f"Tetanus vaccine requested for person {person_id} and given")
            else:
                self.sim.modules['RTI'].schedule_hsi_event_for_tomorrow(self)
                logger.debug(key='rti_general_message',
                             data=f"Tetanus vaccine requested for person {person_id}, not given")
                return self.make_appt_footprint({})

    def did_not_run(self):
        person_id = self.target

        logger.debug(key='rti_general_message',
                     data=f"Tetanus vaccine unavailable for person {person_id}")


class HSI_RTI_Acute_Pain_Management(HSI_Event, IndividualScopeEventMixin):
    """ This HSI event handles all requests for pain management here, all injuries will pass through here and the pain
    medicine required will be set to manage the level of pain they are experiencing, with mild pain being managed with
    paracetamol/NSAIDS, moderate pain being managed with tramadol and severe pain being managed with morphine.

     "There is a mismatch between the burden of musculoskeletal pain conditions and appropriate health policy response
     and planning internationally that can be addressed with an integrated research and policy agenda."
     SEE doi: 10.2105/AJPH.2018.304747
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, RTI)

        self.TREATMENT_ID = 'Rti_AcutePainManagement'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({
            ('Under5OPD' if self.sim.population.props.at[person_id, "age_years"] < 5 else 'Over5OPD'): 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return self.make_appt_footprint({})
        # Check that the person sent here is alive, has been through A&E and RTI_Med_int
        assert df.loc[person_id, 'rt_diagnosed'], 'This person has not been through a and e'
        assert df.loc[person_id, 'rt_med_int'], 'This person has not been through rti med int'
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        get_item_code = self.sim.modules['HealthSystem'].get_item_code_from_item_name
        road_traffic_injuries = self.sim.modules['RTI']
        pain_level = "none"
        # create a dictionary to associate the level of pain to the codes
        pain_dict = {
            'severe': ['1114', '2114', '3113', '4113', '5113', '7113', '8113',  # burns
                       'P782', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883', 'P884',  # amputations
                       '673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676',
                       'P673', 'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675', 'P675a', 'P675b', 'P676',  # SCI
                       '552', '553', '554',  # abdominal trauma
                       '463', '453', '453a', '453b', '441', '443'  # severe chest trauma
                       ],
            'moderate': ['112', '113', '211', '212', '412', '414', '612', '712', '712a', '712b', '712c',
                         '811', '812', '813', '813a', '813b', '813c',  # fractures
                         '322', '323', '722', '822', '822a', '822b',  # dislocations
                         '342', '343', '361', '363',  # neck trauma
                         '461',  # chest wall bruising
                         '813bo', '813co', '813do', '813eo'  # open fractures
                         ],
            'mild': ['1101', '2101', '3101', '4101', '5101', '7101', '8101',  # lacerations
                     '241',  # Minor soft tissue injuries
                     '133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135',  # TBI
                     'P133', 'P133a', 'P133b', 'P133c', 'P133d', 'P134', 'P134a', 'P134b', 'P135',  # Perm TBI
                     '291',  # Eye injury
                     '442'
                     ]
        }
        # iterate over the dictionary to find the pain level, going from highest pain to lowest pain in a for loop,
        # then find the highest level of pain this person has by breaking the for loop
        for severity in pain_dict.keys():
            _, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, pain_dict[severity])
            if counts > 0:
                pain_level = severity
                break
        if pain_level == "mild":
            # Multiple options, some are conditional
            # Give paracetamol
            # Give NSAIDS such as aspirin (unless they are under 16) for soft tissue pain, but not if they are pregnant
            dict_to_output = {'person': person_id,
                              'pain level': pain_level}
            logger.info(key='Requested_Pain_Management',
                        data=dict_to_output,
                        description='Summary of the pain medicine requested by each person')
            if df.loc[person_id, 'age_years'] < 16:
                self.module.item_codes_for_consumables_required['pain_management'] = {
                    get_item_code("Paracetamol 500mg_1000_CMST"): 1
                }
                cond = self.get_consumables(
                    self.module.item_codes_for_consumables_required['pain_management']
                )
            else:
                self.module.item_codes_for_consumables_required['pain_management'] = {
                    get_item_code("diclofenac sodium 25 mg, enteric coated_1000_IDA"): 1
                }
                cond1 = self.get_consumables(
                    self.module.item_codes_for_consumables_required['pain_management']
                )
                self.module.item_codes_for_consumables_required['pain_management'] = {
                    get_item_code("Paracetamol 500mg_1000_CMST"): 1
                }
                cond2 = self.get_consumables(
                    self.module.item_codes_for_consumables_required['pain_management']
                )
                if (cond1 is True) & (cond2 is True):
                    which = self.module.rng.random_sample(size=1)
                    if which <= 0.5:
                        cond = cond1
                        logger.debug(key='rti_general_message',
                                     data=f"Person {person_id} requested paracetamol for their pain relief")
                    else:
                        cond = cond2
                        logger.debug(key='rti_general_message',
                                     data=f"Person {person_id} requested diclofenac for their pain relief")
                elif (cond1 is True) & (cond2 is False):
                    cond = cond1
                    logger.debug(key='rti_general_message',
                                 data=f"Person {person_id} requested paracetamol for their pain relief")
                elif (cond1 is False) & (cond2 is True):
                    cond = cond2
                    logger.debug(key='rti_general_message',
                                 data=f"Person {person_id} requested diclofenac for their pain relief")
                else:
                    which = self.module.rng.random_sample(size=1)
                    if which <= 0.5:
                        cond = cond1
                        logger.debug(key='rti_general_message',
                                     data=f"Person {person_id} requested paracetamol for their pain relief")
                    else:
                        cond = cond2
                        logger.debug(key='rti_general_message',
                                     data=f"Person {person_id} requested diclofenac for their pain relief")
            # Availability of consumables determines if the intervention is delivered...
            if cond:
                logger.debug(key='rti_general_message',
                             data=f"This facility has pain management available for mild pain which has been used for "
                                  f"person {person_id}.")
                dict_to_output = {'person': person_id,
                                  'pain level': pain_level}
                logger.info(key='Successful_Pain_Management',
                            data=dict_to_output,
                            description='Pain medicine successfully provided to the person')
            else:
                self.sim.modules['RTI'].schedule_hsi_event_for_tomorrow(self)
                logger.debug(key='rti_general_message',
                             data=f"This facility has no pain management available for their mild pain, person "
                                  f"{person_id}.")
                return self.make_appt_footprint({})

        if pain_level == "moderate":
            dict_to_output = {'person': person_id,
                              'pain level': pain_level}
            logger.info(key='Requested_Pain_Management',
                        data=dict_to_output,
                        description='Summary of the pain medicine requested by each person')
            self.module.item_codes_for_consumables_required['pain_management'] = {
                get_item_code("tramadol HCl 100 mg/2 ml, for injection_100_IDA"): 1
            }
            is_cons_available = self.get_consumables(
                self.module.item_codes_for_consumables_required['pain_management']
            )
            logger.debug(key='rti_general_message',
                         data=f"Person {person_id} has requested tramadol for moderate pain relief")

            if is_cons_available:
                logger.debug(key='rti_general_message',
                             data=f"This facility has pain management available for moderate pain which has been used "
                                  f"for person {person_id}.")
                dict_to_output = {'person': person_id,
                                  'pain level': pain_level}
                logger.info(key='Successful_Pain_Management',
                            data=dict_to_output,
                            description='Pain medicine successfully provided to the person')
            else:
                self.sim.modules['RTI'].schedule_hsi_event_for_tomorrow(self)
                logger.debug(key='rti_general_message',
                             data=f"This facility has no pain management available for moderate pain for person "
                                  f"{person_id}.")
                return self.make_appt_footprint({})

        if pain_level == "severe":
            dict_to_output = {'person': person_id,
                              'pain level': pain_level}
            logger.info(key='Requested_Pain_Management',
                        data=dict_to_output,
                        description='Summary of the pain medicine requested by each person')
            # give morphine
            self.module.item_codes_for_consumables_required['pain_management'] = {
                get_item_code("morphine sulphate 10 mg/ml, 1 ml, injection (nt)_10_IDA"): 1
            }
            is_cons_available = self.get_consumables(
                self.module.item_codes_for_consumables_required['pain_management']
            )
            logger.debug(key='rti_general_message',
                         data=f"Person {person_id} has requested morphine for severe pain relief")

            if is_cons_available:
                logger.debug(key='rti_general_message',
                             data=f"This facility has pain management available for severe pain which has been used for"
                                  f" person {person_id}")
                dict_to_output = {'person': person_id,
                                  'pain level': pain_level}
                logger.info(key='Successful_Pain_Management',
                            data=dict_to_output,
                            description='Pain medicine successfully provided to the person')
            else:
                self.sim.modules['RTI'].schedule_hsi_event_for_tomorrow(self)
                logger.debug(key='rti_general_message',
                             data=f"This facility has no pain management available for severe pain for person "
                                  f"{person_id}.")
                return self.make_appt_footprint({})

    def did_not_run(self):
        person_id = self.target

        df = self.sim.population.props
        logger.debug(key='rti_general_message',
                     data=f"Pain relief unavailable for person {person_id}")
        injurycodes = {'First injury': df.loc[person_id, 'rt_injury_1'],
                       'Second injury': df.loc[person_id, 'rt_injury_2'],
                       'Third injury': df.loc[person_id, 'rt_injury_3'],
                       'Fourth injury': df.loc[person_id, 'rt_injury_4'],
                       'Fifth injury': df.loc[person_id, 'rt_injury_5'],
                       'Sixth injury': df.loc[person_id, 'rt_injury_6'],
                       'Seventh injury': df.loc[person_id, 'rt_injury_7'],
                       'Eight injury': df.loc[person_id, 'rt_injury_8']}
        logger.debug(key='rti_general_message',
                     data=f"Injury profile of person {person_id}, {injurycodes}")


class HSI_RTI_Major_Surgeries(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event.
        An appointment of a person who has experienced a road traffic injury, had their injuries diagnosed through
        A and E and requires major surgery.

        Major surgeries are defined here as surgeries that include extensive work such as entering a body cavity,
        removing an organ or altering the body’s anatomy

        The injuries treated in this module are as follows:

        FRACTURES:
        While district hospitals can provide some
        emergency trauma care and surgeries, only central hospitals
        are equipped to provide advanced orthopaedic surgery. - Lavy et al. 2007

        '112' - Depressed skull fracture - reported use of surgery in Eaton et al. 2017
        '811' - fractured foot - reported use of surgery in Chagomerana et al. 2017
        '812' - fracture tibia/fibula - reported use of surgery in Chagomerana et al. 2017
        '813a' - Fractured hip - reported use of surgery and Lavy et al. 2007
        '813b' - Fractured pelvis - reported use of surgery and Lavy et al. 2007
        '813c' - Fractured femur - reported use of surgery and Lavy et al. 2007
        '414' - Flail chest - https://www.sciencedirect.com/science/article/abs/pii/S0020138303002900

        SOFT TISSUE INJURIES:
        '342' - Soft tissue injury of the neck
        '343' - Soft tissue injury of the neck

        Thoroscopy treated injuries:
        https://www.ncbi.nlm.nih.gov/nlmcatalog/101549743
        Ref from pediatric handbook for Malawi
        '441' - Closed pneumothorax
        '443' - Open pneumothorax
        '463' - Haemothorax
        '453a' - Diaphragm rupture
        '453b' - Lung contusion

        INTERNAL BLEEDING:
        '361' - Internal bleeding in neck
        '363' - Internal bleeding in neck


        TRAUMATIC BRAIN INJURIES THAT REQUIRE A CRANIOTOMOY - reported use of surgery in Eaton et al 2017 and Lavy et
        al. 2007

        '133a' - Subarachnoid hematoma
        '133b' - Brain contusion
        '133c' - Intraventricular haemorrhage
        '133d' - Subgaleal hematoma
        '134a' - Epidural hematoma
        '134b' - Subdural hematoma
        '135' - diffuse axonal injury

        Laparotomy - Recorded in Lavy et al. 2007 and here: https://www.ajol.info/index.php/mmj/article/view/174378

        '552' - Injury to Intestine, stomach and colon
        '553' - Injury to Spleen, Urinary bladder, Liver, Urethra, Diaphragm
        '554' - Injury to kidney


        SPINAL CORD LESIONS, REQUIRING LAMINOTOMY/FORAMINOTOMY/INTERSPINOUS PROCESS SPACER
        Quote from Eaton et al. 2019:
        "No patients received thoracolumbar braces or underwent spinal surgery."
        https://journals.sagepub.com/doi/pdf/10.1177/0049475518808969
        So those with spinal cord injuries are not likely to be treated here in RTI_Major_Surgeries..

        '673a' - Spinal cord lesion at neck level
        '673b' - Spinal cord lesion below neck level
        '674a' - Spinal cord lesion at neck level
        '674b' - Spinal cord lesion below neck level
        '675a' - Spinal cord lesion at neck level
        '675b' - Spinal cord lesion below neck level
        '676' - Spinal cord lesion at neck level

        AMPUTATIONS - Reported in Crudziak et al. 2019
        '782a' - Amputated finger
        '782b' - Unilateral arm amputation
        '782c' - Amputated thumb
        '783' - Bilateral arm amputation
        '882' - Amputated toe
        '883' - Unilateral lower limb amputation
        '884' - Bilateral lower limb amputation

        Dislocations - Reported in Chagomerana et al. 2017
        '822a' Hip dislocation

        The properties altered in this function are:
        rt_injury_1 through rt_injury_8 - in the incidence that despite treatment the person treated is left
                                          permanently disabled we need to update the injury code to inform the
                                          model that the disability burden associated with the permanently
                                          disabling injury shouldn't be removed
        rt_perm_disability - when a person is decided to be permanently disabled we update this property to reflect this
        rt_date_to_remove_daly - assign recovery dates for the injuries treated with the surgery
        rt_injuries_for_major_surgery - to remove codes due to be treated by major surgery when that injury recieves
                                        a treatment.
        """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, RTI)

        self.TREATMENT_ID = 'Rti_MajorSurgeries'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'MajorSurg': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({})

        p = self.module.parameters
        self.prob_perm_disability_with_treatment_severe_TBI = p['prob_perm_disability_with_treatment_severe_TBI']
        self.allowed_interventions = p['allowed_interventions']
        self.treated_code = 'none'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        rng = self.module.rng
        road_traffic_injuries = self.sim.modules['RTI']
        get_item_code = self.sim.modules['HealthSystem'].get_item_code_from_item_name
        # Request first draft of consumables used in major surgery
        self.module.item_codes_for_consumables_required['major_surgery'] = {
            # request a general anaesthetic
            get_item_code("Halothane (fluothane)_250ml_CMST"): 1,
            # clean the site of the surgery
            get_item_code("Chlorhexidine 1.5% solution_5_CMST"): 1,
            # tools to begin surgery
            get_item_code("Scalpel blade size 22 (individually wrapped)_100_CMST"): 1,
            # administer an IV
            get_item_code('Cannula iv  (winged with injection pot) 18_each_CMST'): 1,
            get_item_code("Giving set iv administration + needle 15 drops/ml_each_CMST"): 1,
            get_item_code("ringer's lactate (Hartmann's solution), 1000 ml_12_IDA"): 1,
            # repair incision made
            get_item_code("Suture pack"): 1,
            get_item_code("Gauze, absorbent 90cm x 40m_each_CMST"): 1,
            # administer pain killer
            get_item_code('Pethidine, 50 mg/ml, 2 ml ampoule'): 1,
            # administer antibiotic
            get_item_code("Ampicillin injection 500mg, PFR_each_CMST"): 1,
            # equipment used by surgeon, gloves and facemask
            get_item_code('Disposables gloves, powder free, 100 pieces per box'): 1,
            get_item_code('surgical face mask, disp., with metal nose piece_50_IDA'): 1,
            # request syringe
            get_item_code("Syringe, Autodisable SoloShot IX "): 1
        }

        request_outcome = self.get_consumables(
            self.module.item_codes_for_consumables_required['major_surgery']
        )

        if not df.at[person_id, 'is_alive']:
            return self.make_appt_footprint({})
        # todo: think about consequences of certain consumables not being available for major surgery and model health
        #  outcomes
        # Isolate the relevant injury information
        surgically_treated_codes = ['112', '811', '812', '813a', '813b', '813c', '133a', '133b', '133c', '133d', '134a',
                                    '134b', '135', '552', '553', '554', '342', '343', '414', '361', '363', '782',
                                    '782a', '782b', '782c', '783', '822a', '882', '883', '884', 'P133a', 'P133b',
                                    'P133c', 'P133d', 'P134a', 'P134b', 'P135', 'P782a', 'P782b', 'P782c', 'P783',
                                    'P882', 'P883', 'P884']
        # If we have allowed spinal cord surgeries to be treated in this simulation, include the associated injury
        # codes here
        if 'include_spine_surgery' in self.allowed_interventions:
            additional_codes = ['673a', '673b', '674a', '674b', '675a', '675b', '676', 'P673a', 'P673b', 'P674',
                                'P674a', 'P674b', 'P675', 'P675a', 'P675b', 'P676']
            for code in additional_codes:
                surgically_treated_codes.append(code)
        # If we have allowed greater access to thoroscopy, include the codes treated by thoroscopy here
        if 'include_thoroscopy' in self.allowed_interventions:
            additional_codes = ['441', '443', '453', '453a', '453b', '463']
            for code in additional_codes:
                surgically_treated_codes.append(code)
        persons_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        injuries_to_be_treated = df.loc[person_id, 'rt_injuries_for_major_surgery']
        assert len(set(injuries_to_be_treated) & set(surgically_treated_codes)) > 0, \
            'This person has asked for surgery but does not have an appropriate injury'
        # check the people sent here have at least one injury treated by this HSI event
        _, counts = road_traffic_injuries.rti_find_and_count_injuries(persons_injuries, surgically_treated_codes)
        if counts == 0:
            logger.debug(key='rti_general_message',
                         data=f"This is RTI major surgery person {person_id} asked for treatment but doesn't"
                              f"need it.")
            return self.make_appt_footprint({})

        # People can be sent here for multiple surgeries, but only one injury can be treated at a time. Decide which
        # injury is being treated in this surgery
        # find index for untreated injuries
        idx_for_untreated_injuries = np.where(pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly']))
        # find untreated injury codes that are treated with major surgery
        relevant_codes = np.intersect1d(injuries_to_be_treated, surgically_treated_codes)
        # check that the person sent here has an appropriate code(s)
        assert len(relevant_codes) > 0, (persons_injuries.values[0], idx_for_untreated_injuries, person_id,
                                         persons_injuries.values[0][idx_for_untreated_injuries])
        # choose a code at random
        self.treated_code = rng.choice(relevant_codes)
        if request_outcome:
            # check the people sent here hasn't died due to rti, have had their injuries diagnosed and been through
            # RTI_Med
            assert df.loc[person_id, 'rt_diagnosed'], 'This person has not been through a and e'
            assert df.loc[person_id, 'rt_med_int'], 'This person has not been through rti med int'
            # ------------------------ Track permanent disabilities with treatment -------------------------------------
            # --------------------------------- Perm disability from TBI -----------------------------------------------
            codes = ['133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135']

            """ Of patients that survived, 80.1% (n 148) had a good recovery with no appreciable clinical neurologic
            deficits, 13.1% (n 24) had a moderate disability with deficits that still allowed the patient to live
            independently, 4.9% (n 9) had severe disability which will require assistance with activities of daily life,
            and 1.1% (n 2) were in a vegetative state
            """
            # Check whether the person having treatment for their tbi will be left permanently disabled
            if self.treated_code in codes:
                prob_perm_disability = self.module.rng.random_sample(size=1)
                if prob_perm_disability < self.prob_perm_disability_with_treatment_severe_TBI:
                    # Track whether they are permanently disabled
                    df.at[person_id, 'rt_perm_disability'] = True
                    # Find the column and code where the permanent injury is stored
                    column, code = road_traffic_injuries.rti_find_injury_column(person_id=person_id, codes=codes)
                    logger.debug(key='rti_general_message',
                                 data=f"@@@@@@@@@@ Person {person_id} had intervention for TBI on {self.sim.date} but "
                                      f"still disabled!!!!!!")
                    # Update the code to make the injury permanent, so it will not have the associated daly weight
                    # removed later on
                    code_to_drop_index = injuries_to_be_treated.index(self.treated_code)
                    injuries_to_be_treated.pop(code_to_drop_index)
                    # remove the old code from rt_injuries_for_major_surgery
                    self.treated_code = "P" + self.treated_code
                    df.loc[person_id, column] = self.treated_code
                    # include the new code in rt_injuries_for_major_surgery
                    df.loc[person_id, 'rt_injuries_for_major_surgery'].append(self.treated_code)
                    assert len(injuries_to_be_treated) == len(df.loc[person_id, 'rt_injuries_for_major_surgery'])

                columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id,
                                                                                                [self.treated_code])

                # schedule the recovery date for the permanent injury for beyond the end of the simulation (making
                # it permanent)
                df.loc[person_id, 'rt_date_to_remove_daly'][int(columns[0][-1]) - 1] = \
                    self.sim.end_date + DateOffset(days=1)
                assert df.loc[person_id, 'rt_date_to_remove_daly'][int(columns[0][-1]) - 1] > self.sim.date
            # ------------------------------------- Perm disability from SCI -------------------------------------------
            if 'include_spine_surgery' in self.allowed_interventions:
                codes = ['673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676']
                if self.treated_code in codes:
                    # Track whether they are permanently disabled
                    df.at[person_id, 'rt_perm_disability'] = True
                    # Find the column and code where the permanent injury is stored
                    column, code = road_traffic_injuries.rti_find_injury_column(person_id=person_id,
                                                                                codes=[self.treated_code])
                    logger.debug(key='rti_general_message',
                                 data=f"@@@@@@@@@@ Person {person_id} had intervention for SCI on {self.sim.date} but "
                                      f"still disabled!!!!!!")
                    code_to_drop_index = injuries_to_be_treated.index(self.treated_code)
                    injuries_to_be_treated.pop(code_to_drop_index)
                    # remove the code from 'rt_injuries_for_major_surgery'
                    df.loc[person_id, 'rt_injuries_for_major_surgery'].remove(self.treated_code)
                    self.treated_code = "P" + self.treated_code
                    # update the code for 'rt_injuries_for_major_surgery'
                    df.loc[person_id, 'rt_injuries_for_major_surgery'].append(self.treated_code)
                    df.loc[person_id, column] = self.treated_code
                    for injury in injuries_to_be_treated:
                        if injury not in df.loc[person_id, 'rt_injuries_for_major_surgery']:
                            df.loc[person_id, 'rt_injuries_for_major_surgery'].append(injury)
                    assert len(injuries_to_be_treated) == len(df.loc[person_id, 'rt_injuries_for_major_surgery'])
                    columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id,
                                                                                                    [self.treated_code])

                    # schedule the recovery date for the permanent injury for beyond the end of the simulation (making
                    # it permanent)
                    df.loc[person_id, 'rt_date_to_remove_daly'][int(columns[0][-1]) - 1] = \
                        self.sim.end_date + DateOffset(days=1)
                    assert df.loc[person_id, 'rt_date_to_remove_daly'][int(columns[0][-1]) - 1] > self.sim.date

            # ------------------------------------- Perm disability from amputation ------------------------------------
            codes = ['782', '782a', '782b', '782c', '783', '882', '883', '884']
            if self.treated_code in codes:
                # Track whether they are permanently disabled
                df.at[person_id, 'rt_perm_disability'] = True
                # Find the column and code where the permanent injury is stored
                column, code = road_traffic_injuries.rti_find_injury_column(person_id=person_id,
                                                                            codes=[self.treated_code])
                logger.debug(key='rti_general_message',
                             data=f"@@@@@@@@@@ Person {person_id} had intervention for an amputation on {self.sim.date}"
                                  f" but still disabled!!!!!!")
                # Update the code to make the injury permanent, so it will not have the associated daly weight removed
                # later on
                code_to_drop_index = injuries_to_be_treated.index(self.treated_code)
                injuries_to_be_treated.pop(code_to_drop_index)
                # remove the old code from rt_injuries_for_major_surgery
                self.treated_code = "P" + self.treated_code
                # add the new code to rt_injuries_for_major_surgery
                df.loc[person_id, 'rt_injuries_for_major_surgery'].append(self.treated_code)
                df.loc[person_id, column] = self.treated_code
                for injury in injuries_to_be_treated:
                    if injury not in df.loc[person_id, 'rt_injuries_for_major_surgery']:
                        df.loc[person_id, 'rt_injuries_for_major_surgery'].append(injury)
                assert len(injuries_to_be_treated) == len(df.loc[person_id, 'rt_injuries_for_major_surgery'])
                columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id,
                                                                                                [self.treated_code])
                # Schedule recovery for the end of the simulation, thereby making the injury permanent

                df.loc[person_id, 'rt_date_to_remove_daly'][int(columns[0][-1]) - 1] = \
                    self.sim.end_date + DateOffset(days=1)
                assert df.loc[person_id, 'rt_date_to_remove_daly'][int(columns[0][-1]) - 1] > self.sim.date

            # ============================== Schedule the recovery dates for the non-permanent injuries ================
            injury_columns = persons_injuries.columns
            maj_surg_recovery_time_in_days = {
                '112': 42,
                '552': 90,
                '553': 90,
                '554': 90,
                '822a': 270,
                '811': 63,
                '812': 63,
                '813a': 270,
                '813b': 70,
                '813c': 120,
                '133a': 42,
                '133b': 42,
                '133c': 42,
                '133d': 42,
                '134a': 42,
                '134b': 42,
                '135': 42,
                '342': 42,
                '343': 42,
                '414': 365,
                '441': 14,
                '443': 14,
                '453a': 42,
                '453b': 42,
                '361': 7,
                '363': 7,
                '463': 7,
            }
            # find the column of the treated injury
            if self.treated_code in maj_surg_recovery_time_in_days.keys():
                columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                              [self.treated_code])[0])
                if pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]):
                    df.loc[person_id, 'rt_date_to_remove_daly'][columns] = \
                        self.sim.date + DateOffset(days=maj_surg_recovery_time_in_days[self.treated_code])
                    assert df.loc[person_id, 'rt_date_to_remove_daly'][columns] > self.sim.date

            # some injuries have a daly weight that swaps upon treatment, get list of those codes
            swapping_codes = RTI.SWAPPING_CODES[:]
            # isolate that swapping codes that will be treated here
            swapping_codes = [code for code in swapping_codes if code in surgically_treated_codes]
            # find the injuries this person will have treated in other forms of treatment
            person = df.loc[person_id]
            treatment_plan = (
                person['rt_injuries_for_minor_surgery'] + person['rt_injuries_to_cast'] +
                person['rt_injuries_to_heal_with_time'] + person['rt_injuries_for_open_fracture_treatment']
            )
            # remove codes that will be treated elsewhere
            swapping_codes = [code for code in swapping_codes if code not in treatment_plan]
            # swap the daly weight for any applicable injuries
            if self.treated_code in swapping_codes:
                road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, [self.treated_code])
            # Check that every injury treated has a recovery time
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [self.treated_code])[0])
            assert not pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]), \
                'no recovery date given for this injury'
            assert df.loc[person_id, 'rt_date_to_remove_daly'][columns] > self.sim.date
            logger.debug(key='rti_general_message',
                         data=f"This is RTI_Major_Surgeries supplying surgery for person {person_id} on date "
                              f"{self.sim.date}!!!!!!, removing code")
            # remove code from major surgeries list
            if self.treated_code in df.loc[person_id, 'rt_injuries_for_major_surgery']:
                df.loc[person_id, 'rt_injuries_for_major_surgery'].remove(self.treated_code)
            assert self.treated_code not in df.loc[person_id, 'rt_injuries_for_major_surgery'], \
                ['Treated injury code not removed', self.treated_code]
            df.loc[person_id, 'rt_date_death_no_med'] = pd.NaT
        else:
            self.sim.modules['RTI'].schedule_hsi_event_for_tomorrow(self)
            if pd.isnull(df.loc[person_id, 'rt_date_death_no_med']):
                df.loc[person_id, 'rt_date_death_no_med'] = self.sim.date + DateOffset(days=7)
            return self.make_appt_footprint({})

    def did_not_run(self):
        person_id = self.target

        df = self.sim.population.props
        logger.debug(key='rti_general_message',
                     data=f"Major surgery not scheduled for person {person_id}")
        injurycodes = {'First injury': df.loc[person_id, 'rt_injury_1'],
                       'Second injury': df.loc[person_id, 'rt_injury_2'],
                       'Third injury': df.loc[person_id, 'rt_injury_3'],
                       'Fourth injury': df.loc[person_id, 'rt_injury_4'],
                       'Fifth injury': df.loc[person_id, 'rt_injury_5'],
                       'Sixth injury': df.loc[person_id, 'rt_injury_6'],
                       'Seventh injury': df.loc[person_id, 'rt_injury_7'],
                       'Eight injury': df.loc[person_id, 'rt_injury_8']}
        logger.debug(key='rti_general_message',
                     data=f"Injury profile of person {person_id}, {injurycodes}")


class HSI_RTI_Minor_Surgeries(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event.
        An appointment of a person who has experienced a road traffic injury, had their injuries diagnosed through
        A and E, treatment plan organised by RTI_MedInt and requires minor surgery.

        Minor surgeries are defined here as surgeries are generally superficial and do not require penetration of a
        body cavity. They do not involve assisted breathing or anesthesia and are usually performed by a single doctor.

        The injuries treated in this module are as follows:

        Evidence for all from Mkandawire et al. 2008:
        https://link.springer.com/article/10.1007%2Fs11999-008-0366-5
        '211' - Facial fractures
        '212' - Facial fractures
        '291' - Injury to the eye
        '241' - Soft tissue injury of the face

        '322' - Dislocation in the neck
        '323' - Dislocation in the neck

        '722' - Dislocated shoulder

        External fixation of fractures
        '811' - fractured foot
        '812' - fractures tibia/fibula
        '813a' - Fractured hip
        '813b' - Fractured pelvis
        '813C' - Fractured femur
        The properties altered in this function are:
        rt_med_int - update to show this person is being treated for their injuries.
        rt_date_to_remove_daly - assign recovery dates for the injuries treated with the surgery
        rt_injuries_for_minor_surgery - to remove codes due to be treated by minor surgery when that injury recieves
                                        a treatment.
        """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, RTI)

        self.TREATMENT_ID = 'Rti_MinorSurgeries'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'MinorSurg': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return self.make_appt_footprint({})
        get_item_code = self.sim.modules['HealthSystem'].get_item_code_from_item_name
        # Request first draft of consumables used in major surgery
        self.module.item_codes_for_consumables_required['minor_surgery'] = {
            # request a local anaesthetic
            get_item_code("Halothane (fluothane)_250ml_CMST"): 1,
            # clean the site of the surgery
            get_item_code("Chlorhexidine 1.5% solution_5_CMST"): 1,
            # tools to begin surgery
            get_item_code("Scalpel blade size 22 (individually wrapped)_100_CMST"): 1,
            # administer an IV
            get_item_code('Cannula iv  (winged with injection pot) 18_each_CMST'): 1,
            get_item_code("Giving set iv administration + needle 15 drops/ml_each_CMST"): 1,
            get_item_code("ringer's lactate (Hartmann's solution), 1000 ml_12_IDA"): 1,
            # repair incision made
            get_item_code("Suture pack"): 1,
            get_item_code("Gauze, absorbent 90cm x 40m_each_CMST"): 1,
            # administer pain killer
            get_item_code('Pethidine, 50 mg/ml, 2 ml ampoule'): 1,
            # administer antibiotic
            get_item_code("Ampicillin injection 500mg, PFR_each_CMST"): 1,
            # equipment used by surgeon, gloves and facemask
            get_item_code('Disposables gloves, powder free, 100 pieces per box'): 1,
            get_item_code('surgical face mask, disp., with metal nose piece_50_IDA'): 1,
            # request syringe
            get_item_code("Syringe, Autodisable SoloShot IX "): 1
        }
        rng = self.module.rng
        road_traffic_injuries = self.sim.modules['RTI']
        surgically_treated_codes = ['322', '211', '212', '323', '722', '291', '241', '811', '812', '813a', '813b',
                                    '813c']
        persons_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        person = df.loc[person_id]
        # =========================================== Tests ============================================================
        # check the people sent here hasn't died due to rti, have had their injuries diagnosed and been through RTI_Med
        assert person['rt_diagnosed'], 'This person has not been through a and e'
        assert person['rt_med_int'], 'This person has not been through rti med int'
        # check they have at least one injury treated by minor surgery
        _, counts = road_traffic_injuries.rti_find_and_count_injuries(persons_injuries, surgically_treated_codes)
        if counts == 0:
            logger.debug(key='rti_general_message',
                         data=f"This is RTI minor surgery person {person_id} asked for treatment but doesn't"
                              f"need it.")
            return self.make_appt_footprint({})
            # find the injuries which will be treated here
        relevant_codes = np.intersect1d(df.loc[person_id, 'rt_injuries_for_minor_surgery'], surgically_treated_codes)
        # Check that a code has been selected to be treated
        assert len(relevant_codes) > 0
        # choose an injury to treat
        treated_code = rng.choice(relevant_codes)
        # need to determine whether this person has an injury which will treated with external fixation
        # external_fixation_codes = ['811', '812', '813a', '813b', '813c']
        request_outcome = self.get_consumables(
            self.module.item_codes_for_consumables_required['minor_surgery']
        )
        # todo: think about consequences of certain consumables not being available for minor surgery and model health
        #  outcomes
        if request_outcome:
            injury_columns = persons_injuries.columns
            # create a dictionary to store the recovery times for each injury in days
            minor_surg_recov_time_days = {
                '322': 180,
                '323': 180,
                '722': 49,
                '211': 49,
                '212': 49,
                '291': 7,
                '241': 7,
                '811': 63,
                '812': 63,
                '813a': 63,
                '813b': 63,
                '813c': 63,
            }

            # assign a recovery time for the treated person from the dictionary, get the column which the injury is
            # stored in
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, [treated_code])[0])
            # assign a recovery date
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = \
                self.sim.date + DateOffset(days=minor_surg_recov_time_days[treated_code])
            # make sure the injury recovery date is in the future
            assert df.loc[person_id, 'rt_date_to_remove_daly'][columns] > self.sim.date

            # some injuries have a change in daly weight if they are treated, find all possible swappable codes
            swapping_codes = RTI.SWAPPING_CODES[:]
            # exclude any codes that could be swapped but are due to be treated elsewhere
            treatment_plan = (
                person['rt_injuries_for_minor_surgery'] + person['rt_injuries_to_cast'] +
                person['rt_injuries_to_heal_with_time'] + person['rt_injuries_for_open_fracture_treatment']
            )
            swapping_codes = [code for code in swapping_codes if code not in treatment_plan]
            if treated_code in swapping_codes:
                road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, [treated_code])
            logger.debug(key='rti_general_message',
                         data=f"This is RTI_Minor_Surgeries supplying minor surgeries for person {person_id} on date "
                              f"{self.sim.date}!!!!!!")
            # update the dataframe to reflect that this person is recieving medical care
            df.at[person_id, 'rt_med_int'] = True
            # Check if the injury has been given a recovery date
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, [treated_code])[0])
            assert not pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]), \
                'no recovery date given for this injury'
            # remove code from minor surgeries list as it has now been treated
            if treated_code in df.loc[person_id, 'rt_injuries_for_minor_surgery']:
                df.loc[person_id, 'rt_injuries_for_minor_surgery'].remove(treated_code)
            assert treated_code not in df.loc[person_id, 'rt_injuries_for_minor_surgery'], \
                ['Injury treated not removed', treated_code]
            df.loc[person_id, 'rt_date_death_no_med'] = pd.NaT
        else:
            self.sim.modules['RTI'].schedule_hsi_event_for_tomorrow(self)
            if pd.isnull(df.loc[person_id, 'rt_date_death_no_med']):
                df.loc[person_id, 'rt_date_death_no_med'] = self.sim.date + DateOffset(days=7)
            logger.debug(key='rti_general_message',
                         data=f"This is RTI_Minor_Surgeries failing to provide minor surgeries for person {person_id} "
                              f"on date {self.sim.date}!!!!!!")
            return self.make_appt_footprint({})

    def did_not_run(self):
        person_id = self.target

        df = self.sim.population.props
        logger.debug(key='rti_general_message',
                     data=f"Minor surgery not scheduled for person {person_id}")
        injurycodes = {'First injury': df.loc[person_id, 'rt_injury_1'],
                       'Second injury': df.loc[person_id, 'rt_injury_2'],
                       'Third injury': df.loc[person_id, 'rt_injury_3'],
                       'Fourth injury': df.loc[person_id, 'rt_injury_4'],
                       'Fifth injury': df.loc[person_id, 'rt_injury_5'],
                       'Sixth injury': df.loc[person_id, 'rt_injury_6'],
                       'Seventh injury': df.loc[person_id, 'rt_injury_7'],
                       'Eight injury': df.loc[person_id, 'rt_injury_8']}
        logger.debug(key='rti_injury_profile_of_untreated_person',
                     data=injurycodes)


class RTI_Medical_Intervention_Death_Event(Event, IndividualScopeEventMixin):
    """This is the MedicalInterventionDeathEvent. It is scheduled by the MedicalInterventionEvent to occur at the end of
     the person's determined length of stay. The risk of mortality for the person wil medical intervention is determined
     by the persons ISS score and whether they have polytrauma.

     The properties altered by this event are:
     rt_post_med_death - updated to reflect when a person dies from their injuries
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        p = self.module.parameters
        self.prob_death_iss_less_than_9 = p['prob_death_iss_less_than_9']
        self.prob_death_iss_10_15 = p['prob_death_iss_10_15']
        self.prob_death_iss_16_24 = p['prob_death_iss_16_24']
        self.prob_death_iss_25_35 = p['prob_death_iss_25_35']
        self.prob_death_iss_35_plus = p['prob_death_iss_35_plus']

    def apply(self, person_id):
        df = self.sim.population.props

        randfordeath = self.module.rng.random_sample(size=1)
        # ======================================== Tests ==============================================================
        assert df.loc[person_id, 'rt_ISS_score'] > 0
        mortality_checked = False
        probabilities_of_death = {
            '1-4': [range(1, 5), 0],
            '5-9': [range(5, 10), self.prob_death_iss_less_than_9],
            '10-15': [range(10, 16), self.prob_death_iss_10_15],
            '16-24': [range(16, 25), self.prob_death_iss_16_24],
            '25-35': [range(25, 36), self.prob_death_iss_25_35],
            '35-75': [range(25, 76), self.prob_death_iss_35_plus]
        }
        # Schedule death for those who died from their injuries despite medical intervention
        if df.loc[person_id, 'cause_of_death'] == 'Other':
            pass
        for range_boundaries in probabilities_of_death.keys():
            if df.loc[person_id].rt_ISS_score in probabilities_of_death[range_boundaries][0]:
                if randfordeath < probabilities_of_death[range_boundaries][1]:
                    mortality_checked = True
                    df.loc[person_id, 'rt_post_med_death'] = True
                    dict_to_output = {'person': person_id,
                                      'First injury': df.loc[person_id, 'rt_injury_1'],
                                      'Second injury': df.loc[person_id, 'rt_injury_2'],
                                      'Third injury': df.loc[person_id, 'rt_injury_3'],
                                      'Fourth injury': df.loc[person_id, 'rt_injury_4'],
                                      'Fifth injury': df.loc[person_id, 'rt_injury_5'],
                                      'Sixth injury': df.loc[person_id, 'rt_injury_6'],
                                      'Seventh injury': df.loc[person_id, 'rt_injury_7'],
                                      'Eight injury': df.loc[person_id, 'rt_injury_8']}
                    logger.info(key='RTI_Death_Injury_Profile',
                                data=dict_to_output,
                                description='The injury profile of those who have died due to rtis despite medical care'
                                )
                    # Schedule the death
                    self.sim.modules['Demography'].do_death(individual_id=person_id, cause="RTI_death_with_med",
                                                            originating_module=self.module)
                    # Log the death
                    logger.debug(key='rti_general_message',
                                 data=f"This is RTIMedicalInterventionDeathEvent scheduling a death for person "
                                      f"{person_id} who was treated for their injuries but still died on date "
                                      f"{self.sim.date}")
                else:
                    mortality_checked = True

        assert mortality_checked, 'Something missing in criteria'


class RTI_No_Lifesaving_Medical_Intervention_Death_Event(Event, IndividualScopeEventMixin):
    """This is the NoMedicalInterventionDeathEvent. It is scheduled by the MedicalInterventionEvent which determines the
    resources required to treat that person and if they aren't present, the person is sent here. This function is also
    called by the did not run function for rti_major_surgeries for certain injuries, implying that if life saving
    surgery is not available for the person, then we have to ask the probability of them dying without having this life
    saving surgery.

    some information on time to craniotomy here:
    https://thejns.org/focus/view/journals/neurosurg-focus/45/6/article-pE2.xml?body=pdf-10653


    The properties altered by this event are:
    rt_unavailable_med_death - to denote that this person has died due to medical interventions not being available
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        p = self.module.parameters
        # load the parameteres used for this event
        self.prob_death_TBI_SCI_no_treatment = p['prob_death_TBI_SCI_no_treatment']
        self.prob_death_fractures_no_treatment = p['prob_death_fractures_no_treatment']
        self.prop_death_burns_no_treatment = p['prop_death_burns_no_treatment']
        self.prob_death_MAIS3 = p['prob_death_MAIS3']
        self.prob_death_MAIS4 = p['prob_death_MAIS4']
        self.prob_death_MAIS5 = p['prob_death_MAIS5']
        self.prob_death_MAIS6 = p['prob_death_MAIS6']

        self.allowed_interventions = p['allowed_interventions']

    def apply(self, person_id):
        probabilities_of_death = {
            '1': 0,
            '2': 0,
            '3': self.prob_death_MAIS3,
            '4': self.prob_death_MAIS4,
            '5': self.prob_death_MAIS5,
            '6': self.prob_death_MAIS6
        }
        life_threatening_injuries = ['133a', '133b', '133c', '133d', '134a', '134b', '135',  # TBI
                                     '112',  # Depressed skull fracture
                                     'P133a', 'P133b', 'P133c', 'P133d', 'P134a', 'P134b', 'P135',  # Perm TBI
                                     '342', '343', '361', '363',  # Injuries to neck
                                     '414', '441', '443', '463', '453a', '453b',  # Severe chest trauma
                                     '782b',  # Unilateral arm amputation
                                     '783',  # Bilateral arm amputation
                                     '883',  # Unilateral lower limb amputation
                                     '884',  # Bilateral lower limb amputation
                                     '552', '553', '554'  # Internal organ injuries
                                     ]

        df = self.sim.population.props
        untreated_injuries = []
        persons_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        non_empty_injuries = persons_injuries[persons_injuries != "none"]
        non_empty_injuries = non_empty_injuries.dropna(axis=1)
        # drop injuries that have a treatment scheduled
        person = df.loc[person_id]
        treatment_plan = (
            person['rt_injuries_for_minor_surgery'] + person['rt_injuries_to_cast'] +
            person['rt_injuries_to_heal_with_time'] + person['rt_injuries_for_open_fracture_treatment']
        )
        maj_surg_codes = ['112', '811', '812', '813a', '813b', '813c', '133a', '133b', '133c', '133d', '134a', '134b',
                          '135', '552', '553', '554', '342', '343', '414', '361', '363', '782', '782a', '782b', '782c',
                          '783', '822a', '882', '883', '884', 'P133a', 'P133b', 'P133c', 'P133d', 'P134a', 'P134b',
                          'P135', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883', 'P884']
        # If we have allowed spinal cord surgeries to be treated in this simulation, include the associated injury
        # codes here
        if 'include_spine_surgery' in self.allowed_interventions:
            additional_codes = ['673a', '673b', '674a', '674b', '675a', '675b', '676', 'P673a', 'P673b', 'P674',
                                'P674a', 'P674b', 'P675', 'P675a', 'P675b', 'P676']
            for code in additional_codes:
                maj_surg_codes.append(code)
        # If we have allowed greater access to thoroscopy, include the codes treated by thoroscopy here
        if 'include_thoroscopy' in self.allowed_interventions:
            additional_codes = ['441', '443', '453', '453a', '453b', '463']
            for code in additional_codes:
                maj_surg_codes.append(code)
        for col in non_empty_injuries:
            # create the conditions to ignore untreated injuries
            injury_treated_elsewhere = non_empty_injuries[col].values.to_list()[0] in treatment_plan
            injury_not_treated_by_major_surgery = non_empty_injuries[col].values.to_list()[0] not in maj_surg_codes
            condition_to_remove_column = injury_treated_elsewhere or injury_not_treated_by_major_surgery
            if condition_to_remove_column:
                non_empty_injuries = non_empty_injuries.drop(col, axis=1)
        for col in non_empty_injuries:
            if pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1]):
                untreated_injuries.append(df.at[person_id, col])
        untreated_injuries = [code for code in untreated_injuries if code in life_threatening_injuries]
        mais_scores = [1]
        for injury in untreated_injuries:
            mais_scores.append(self.module.ASSIGN_INJURIES_AND_DALY_CHANGES[injury][0][-1])
        max_untreated_injury = max(mais_scores)
        prob_death = probabilities_of_death[str(max_untreated_injury)]

        randfordeath = self.module.rng.random_sample(size=1)
        if randfordeath < prob_death:
            df.loc[person_id, 'rt_unavailable_med_death'] = True
            self.sim.modules['Demography'].do_death(individual_id=person_id, cause="RTI_unavailable_med",
                                                    originating_module=self.module)
            # Log the death
            logger.debug(key='rti_general_message',
                         data=f"This is RTINoMedicalInterventionDeathEvent scheduling a death for person {person_id} on"
                              f" date {self.sim.date}")
        else:
            # person has survived their injuries despite the lack of treatment. Assign a recovery date to their injuries
            # If a spinal injury, amputation, TBI is untreated, assign this injury's recovery time to the end of the
            # simulation
            codes = ['673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676',
                     '782a', '782b', '782c', '783', '882', '883', '884', '133', '133a', '133b', '133c', '133d', '134',
                     '134a', '134b', '135']
            for injury in untreated_injuries:
                if injury in codes:
                    # Track whether they are permanently disabled
                    df.at[person_id, 'rt_perm_disability'] = True
                    # Find the column and code where the permanent injury is stored
                    df.loc[person_id, col] = "P" + injury
                    # schedule the recovery date for the permanent injury for beyond the end of the simulation (making
                    # it permanent)
                    df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = \
                        self.sim.end_date + DateOffset(days=1)
                    assert df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] > self.sim.date
                    # all injuries are handled by major surgery here, remove the untreated injury code
                    df.loc[person_id, 'rt_injuries_for_major_surgery'].remove(injury)
                else:
                    # check if the injury has a heal time associated with no treamtent
                    if injury in self.module.NO_TREATMENT_RECOVERY_TIMES_IN_DAYS.keys():
                        df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = \
                            self.sim.end_date + DateOffset(days=self.module.NO_TREATMENT_RECOVERY_TIMES_IN_DAYS[injury])
                        if injury in df.loc[person_id, 'rt_injuries_for_major_surgery']:
                            df.loc[person_id, 'rt_injuries_for_major_surgery'].remove(injury)


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
#
#   Put the logging events here. There should be a regular logger outputting current states of the
#   population. There may also be a loggig event that is driven by particular events.
# ---------------------------------------------------------------------------------------------------------


class RTI_Logging_Event(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        """Produce a summary of the numbers of people with respect to the action of this module.
        This is a regular event that can output current states of people or cumulative events since last logging event.
        """

        # run this event every month
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, RTI)
        # Create variables used to store simulation data in
        # Number of injured body region data
        self.tot1inj = 0
        self.tot2inj = 0
        self.tot3inj = 0
        self.tot4inj = 0
        self.tot5inj = 0
        self.tot6inj = 0
        self.tot7inj = 0
        self.tot8inj = 0
        # Injury category data
        self.totfracnumber = 0
        self.totdisnumber = 0
        self.tottbi = 0
        self.totsoft = 0
        self.totintorg = 0
        self.totintbled = 0
        self.totsci = 0
        self.totamp = 0
        self.toteye = 0
        self.totextlac = 0
        self.totburns = 0
        # Injury location on body data
        self.totAIS1 = 0
        self.totAIS2 = 0
        self.totAIS3 = 0
        self.totAIS4 = 0
        self.totAIS5 = 0
        self.totAIS6 = 0
        self.totAIS7 = 0
        self.totAIS8 = 0
        # Injury severity data
        self.totmild = 0
        self.totsevere = 0
        # More model progression data
        self.totinjured = 0
        self.deathonscene = 0
        self.soughtmedcare = 0
        self.deathaftermed = 0
        self.deathwithoutmed = 0
        self.permdis = 0
        self.ISSscore = []
        self.severe_pain = 0
        self.moderate_pain = 0
        self.mild_pain = 0
        # Create variables for averages over time in the model
        self.numerator = 0
        self.denominator = 0
        self.death_inc_numerator = 0
        self.death_in_denominator = 0
        self.fracdenominator = 0
        # Create variables to measure where certain injuries are located on the body
        self.fracdist = [0, 0, 0, 0, 0, 0, 0, 0]
        self.openwounddist = [0, 0, 0, 0, 0, 0, 0, 0]
        self.burndist = [0, 0, 0, 0, 0, 0, 0, 0]

    def apply(self, population):
        # Make some summary statistics
        # Get the dataframe and isolate the important information
        df = population.props
        # dump dataframe each month if population size is large (used to find the minimum viable population size)
        time_stamped_file_name = "df_at_" + str(self.sim.date.month) + "_" + str(self.sim.date.year)
        if len(df.loc[df.is_alive]) > 750000:
            df.to_csv(f"C:/Users/Robbie Manning Smith/Documents/Dataframe_dump/{time_stamped_file_name}.csv")
        thoseininjuries = df.loc[df.rt_road_traffic_inc]
        # ================================= Injury severity ===========================================================
        sev = thoseininjuries['rt_inj_severity']
        rural_injuries = df.loc[df.rt_road_traffic_inc & ~df.li_urban]
        if len(rural_injuries) > 0:
            percent_sev_rural = \
                len(rural_injuries.loc[rural_injuries['rt_inj_severity'] == 'severe']) / len(rural_injuries)
        else:
            percent_sev_rural = 'none_injured'
        urban_injuries = df.loc[df.rt_road_traffic_inc & df.li_urban]
        if len(urban_injuries) > 0:
            percent_sev_urban = \
                len(urban_injuries.loc[urban_injuries['rt_inj_severity'] == 'severe']) / len(urban_injuries)
        else:
            percent_sev_urban = 'none_injured'
        severity, severitycount = np.unique(sev, return_counts=True)
        if 'mild' in severity:
            idx = np.where(severity == 'mild')
            self.totmild += len(idx)
        if 'severe' in severity:
            idx = np.where(severity == 'severe')
            self.totsevere += len(idx)
        dict_to_output = {
            'total_mild_injuries': self.totmild,
            ''
            '_severe_injuries': self.totsevere,
            'Percent_severe_rural': percent_sev_rural,
            'Percent_severe_urban': percent_sev_urban
        }
        logger.info(key='injury_severity',
                    data=dict_to_output,
                    description='severity of injuries in simulation')
        # ==================================== Incidence ==============================================================
        # How many were involved in a RTI
        n_in_RTI = df.rt_road_traffic_inc.sum()
        children_in_RTI = len(df.loc[df.rt_road_traffic_inc & (df['age_years'] < 19)])
        children_alive = len(df.loc[df['age_years'] < 19])
        self.numerator += n_in_RTI
        self.totinjured += n_in_RTI
        # How many were disabled
        n_perm_disabled = (df.is_alive & df.rt_perm_disability).sum()
        # self.permdis += n_perm_disabled
        n_alive = df.is_alive.sum()
        self.denominator += (n_alive - n_in_RTI) * (1 / 12)
        n_immediate_death = (df.rt_road_traffic_inc & df.rt_imm_death).sum()
        self.deathonscene += n_immediate_death
        diedfromrtiidx = df.index[df.rt_imm_death | df.rt_post_med_death | df.rt_no_med_death | df.rt_death_from_shock |
                                  df.rt_unavailable_med_death]
        n_sought_care = (df.rt_road_traffic_inc & df.rt_med_int).sum()
        self.soughtmedcare += n_sought_care
        n_death_post_med = df.rt_post_med_death.sum()
        self.deathaftermed += n_death_post_med
        self.deathwithoutmed += df.rt_no_med_death.sum()
        self.death_inc_numerator += n_immediate_death + n_death_post_med + len(df.loc[df.rt_no_med_death])
        self.death_in_denominator += (n_alive - (n_immediate_death + n_death_post_med + len(df.loc[df.rt_no_med_death])
                                                 )) * \
                                     (1 / 12)
        if self.numerator > 0:
            percent_accidents_result_in_death = \
                (self.deathonscene + self.deathaftermed + self.deathwithoutmed) / self.numerator
        else:
            percent_accidents_result_in_death = 'none injured'
        maleinrti = len(df.loc[df.rt_road_traffic_inc & (df['sex'] == 'M')])
        femaleinrti = len(df.loc[df.rt_road_traffic_inc & (df['sex'] == 'F')])

        divider = min(maleinrti, femaleinrti)
        if divider > 0:
            maleinrti = maleinrti / divider
            femaleinrti = femaleinrti / divider
        else:
            maleinrti = 1
            femaleinrti = 0
        mfratio = [maleinrti, femaleinrti]
        if (n_in_RTI - len(df.loc[df.rt_imm_death])) > 0:
            percent_sought_care = n_sought_care / (n_in_RTI - len(df.loc[df.rt_imm_death]))
        else:
            percent_sought_care = 'none_injured'

        if n_sought_care > 0:
            percent_died_post_care = n_death_post_med / n_sought_care
        else:
            percent_died_post_care = 'none_injured'

        if n_sought_care > 0:
            percentage_admitted_to_ICU_or_HDU = len(df.loc[df.rt_med_int & df.rt_in_icu_or_hdu]) / n_sought_care
        else:
            percentage_admitted_to_ICU_or_HDU = 'none_injured'
        if (n_alive - n_in_RTI) > 0:
            inc_rti = (n_in_RTI / ((n_alive - n_in_RTI) * (1 / 12))) * 100000
        else:
            inc_rti = 0
        if (children_alive - children_in_RTI) > 0:
            inc_rti_in_children = (children_in_RTI / ((children_alive - children_in_RTI) * (1 / 12))) * 100000
        else:
            inc_rti_in_children = 0
        if (n_alive - len(diedfromrtiidx)) > 0:
            inc_rti_death = (len(diedfromrtiidx) / ((n_alive - len(diedfromrtiidx)) * (1 / 12))) * 100000
        else:
            inc_rti_death = 0
        if (n_alive - len(df.loc[df.rt_post_med_death])) > 0:
            inc_post_med_death = (len(df.loc[df.rt_post_med_death]) / ((n_alive - len(df.loc[df.rt_post_med_death])) *
                                                                       (1 / 12))) * 100000
        else:
            inc_post_med_death = 0
        if (n_alive - len(df.loc[df.rt_imm_death])) > 0:
            inc_imm_death = (len(df.loc[df.rt_imm_death]) / ((n_alive - len(df.loc[df.rt_imm_death])) * (1 / 12))) * \
                            100000
        else:
            inc_imm_death = 0
        if (n_alive - len(df.loc[df.rt_no_med_death])) > 0:
            inc_death_no_med = (len(df.loc[df.rt_no_med_death]) /
                                ((n_alive - len(df.loc[df.rt_no_med_death])) * (1 / 12))) * 100000
        else:
            inc_death_no_med = 0
        if (n_alive - len(df.loc[df.rt_unavailable_med_death])) > 0:
            inc_death_unavailable_med = (len(df.loc[df.rt_unavailable_med_death]) /
                                         ((n_alive - len(df.loc[df.rt_unavailable_med_death])) * (1 / 12))) * 100000
        else:
            inc_death_unavailable_med = 0
        if self.fracdenominator > 0:
            frac_incidence = (self.totfracnumber / self.fracdenominator) * 100000
        else:
            frac_incidence = 0
        # calculate case fatality ratio for those injured who don't seek healthcare
        did_not_seek_healthcare = len(df.loc[df.rt_road_traffic_inc & ~df.rt_med_int & ~df.rt_diagnosed])
        died_no_healthcare = \
            len(df.loc[df.rt_road_traffic_inc & df.rt_no_med_death & ~df.rt_med_int & ~df.rt_diagnosed])
        if did_not_seek_healthcare > 0:
            cfr_no_med = died_no_healthcare / did_not_seek_healthcare
        else:
            cfr_no_med = 'all_sought_care'
        # calculate incidence rate per 100,000 of deaths on scene
        if n_alive > 0:
            inc_death_on_scene = (len(df.loc[df.rt_imm_death]) / n_alive) * 100000 * (1 / 12)
        else:
            inc_death_on_scene = 0
        dict_to_output = {
            'number involved in a rti': n_in_RTI,
            'incidence of rti per 100,000': inc_rti,
            'incidence of rti per 100,000 in children': inc_rti_in_children,
            'incidence of rti death per 100,000': inc_rti_death,
            'incidence of death post med per 100,000': inc_post_med_death,
            'incidence of prehospital death per 100,000': inc_imm_death,
            'incidence of death on scene per 100,000': inc_death_on_scene,
            'incidence of death without med per 100,000': inc_death_no_med,
            'incidence of death due to unavailable med per 100,000': inc_death_unavailable_med,
            'incidence of fractures per 100,000': frac_incidence,
            'number alive': n_alive,
            'number immediate deaths': n_immediate_death,
            'number deaths post med': n_death_post_med,
            'number deaths without med': len(df.loc[df.rt_no_med_death]),
            'number deaths unavailable med': len(df.loc[df.rt_unavailable_med_death]),
            'number rti deaths': len(diedfromrtiidx),
            'number permanently disabled': n_perm_disabled,
            'percent of crashes that are fatal': percent_accidents_result_in_death,
            'male:female ratio': mfratio,
            'percent sought healthcare': percent_sought_care,
            'percentage died after med': percent_died_post_care,
            'percent admitted to ICU or HDU': percentage_admitted_to_ICU_or_HDU,
            'cfr_no_med': cfr_no_med,
        }
        logger.info(key='summary_1m',
                    data=dict_to_output,
                    description='Summary of the rti injuries in the last month')
        # =========================== Get population demographics of those with RTIs ==================================
        columnsOfInterest = ['sex', 'age_years', 'li_ex_alc']
        injuredDemographics = df.loc[df.rt_road_traffic_inc]

        injuredDemographics = injuredDemographics.loc[:, columnsOfInterest]
        try:
            percent_related_to_alcohol = len(injuredDemographics.loc[injuredDemographics.li_ex_alc]) / \
                                         len(injuredDemographics)
        except ZeroDivisionError:
            percent_related_to_alcohol = 0
        injured_demography_summary = {
            'males_in_rti': injuredDemographics['sex'].value_counts()['M'],
            'females_in_rti': injuredDemographics['sex'].value_counts()['F'],
            'age': injuredDemographics['age_years'].values.tolist(),
            'male_age': injuredDemographics.loc[injuredDemographics['sex'] == 'M', 'age_years'].values.tolist(),
            'female_age': injuredDemographics.loc[injuredDemographics['sex'] == 'F', 'age_years'].values.tolist(),
            'percent_related_to_alcohol': percent_related_to_alcohol,
        }
        logger.info(key='rti_demography',
                    data=injured_demography_summary,
                    description='Demographics of those in rti')

        # =================================== Flows through the model ==================================================
        dict_to_output = {'total_injured': self.totinjured,
                          'total_died_on_scene': self.deathonscene,
                          'total_sought_medical_care': self.soughtmedcare,
                          'total_died_after_medical_intervention': self.deathaftermed,
                          'total_permanently_disabled': n_perm_disabled}
        logger.info(key='model_progression',
                    data=dict_to_output,
                    description='Flows through the rti module')
