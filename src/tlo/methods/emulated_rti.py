"""
Road traffic injury module.

"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
from collections import Counter

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging, Date

from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.hsi_generic_first_appts import GenericFirstAppointmentsMixin
from tlo.methods.symptommanager import Symptom
from tlo.util import read_csv_files

from sdv.single_table import CTGANSynthesizer
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.sampling import Condition

if TYPE_CHECKING:
    from tlo.methods.hsi_generic_first_appts import HSIEventScheduler
    from tlo.population import IndividualProperties

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EmulatedRTI(Module, GenericFirstAppointmentsMixin):
    """
    The road traffic injuries module for the TLO model, handling all injuries related to road traffic accidents.
    """

    def __init__(self, name=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.
        super().__init__(name)

    INIT_DEPENDENCIES = {"SymptomManager",
                         "HealthBurden"}

    ADDITIONAL_DEPENDENCIES = {
        'Demography',
        'Lifestyle',
        'HealthSystem',
    }
    
    # ================================================================================
    # EMULATOR PARAMETERS
    # Counters tracking use of HealthSystem by RTI module under use of emulator
    HS_Use_Type = [
        'Level2_AccidentsandEmerg', 'Level2_DiagRadio', 'Level2_EPI',
        'Level2_IPAdmission', 'Level2_InpatientDays', 'Level2_MajorSurg',
        'Level2_MinorSurg', 'Level2_Over5OPD', 'Level2_Tomography', 'Level2_Under5OPD'
    ]

    # Initialize the counter with all items set to 0
    HS_Use_by_RTI = Counter({col: 0 for col in HS_Use_Type})
 
    Rti_Services = ['Rti_AcutePainManagement','Rti_BurnManagement','Rti_FractureCast','Rti_Imaging','Rti_MajorSurgeries','Rti_MedicalIntervention','Rti_MinorSurgeries','Rti_OpenFractureTreatment','Rti_ShockTreatment','Rti_Suture','Rti_TetanusVaccine']

    HS_conditions = {}

    RTI_emulator = None
    # ================================================================================


    INJURY_INDICES = range(1, 9)

    INJURY_COLUMNS = [f'rt_injury_{i}' for i in INJURY_INDICES]

    DATE_TO_REMOVE_DALY_COLUMNS = [f'rt_date_to_remove_daly_{i}' for i in INJURY_INDICES]

    # Bi-directional map from/to injury columns to/from date to remove daly columns
    INJURY_DATE_COLUMN_MAP = {
        **{f'rt_injury_{i}': f'rt_date_to_remove_daly_{i}' for i in INJURY_INDICES},
        **{f'rt_date_to_remove_daly_{i}': f'rt_injury_{i}' for i in INJURY_INDICES},
    }

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
        'prob_death_non_serious': Parameter(
            Types.REAL,
            'A parameter to determine the probability of death for non serious condition'
        ),        'prob_death_MAIS1': Parameter(
            Types.REAL,
            'A parameter to determine the probability of death without medical intervention with a military AIS'
            'score of 1'
        ),
        'prob_death_MAIS2': Parameter(
            Types.REAL,
            'A parameter to determine the probability of death without medical intervention with a military AIS'
            'score of 2'
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
        ),
        'maximum_number_of_times_HSI_events_should_run': Parameter(
            Types.INT,
            "limit on the number of times an HSI event can run"
        ),
        'use_RTI_emulator': Parameter(
            Types.BOOL,
            "Replace module with RTI emulator, valid if running in mode 1 with actual consumable availability"
        ),
        'hsi_schedule_window_days': Parameter(
            Types.INT,
            'Number of days window to schedule HSI appointment'
        ),
        'main_polling_frequency': Parameter(
            Types.INT,
            'Frequency in months for RTI polling events that determine new injuries'
        ),
        'rti_check_death_no_med_event_frequency_days': Parameter(
            Types.INT,
            'Frequency in days for RTI check death no med event'
        ),
        'rti_recovery_event_frequency_days': Parameter(
            Types.INT,
            'Frequency in days for RTI recovery event'
        ),
        'incidence_rate_frequency': Parameter(
            Types.INT,
            'Number of months per year for rate conversion calculations'
        ),
        'incidence_rate_per_population': Parameter(
            Types.INT,
            'Population base for incidence rate calculations'
        ),
        'days_to_death_without_treatment': Parameter(
            Types.INT,
            'Number of days until death for untreated RTI patients'
        ),
        'max_treatment_duration_days': Parameter(
            Types.INT,
            'Maximum number of days for RTI treatment duration'
        ),
        'intervention_incidence_reduction_factor': Parameter(
            Types.REAL,
            'Factor by which interventions reduce RTI incidence '
            '(applied when reduce_incidence in allowed_interventions)'
        ),
        'laceration_recovery_days': Parameter(
            Types.INT,
            'Number of days for recovery assessment period'
        ),
        'hsi_opening_delay_days': Parameter(
            Types.INT,
            'Number of days delay before HSI appointments can be scheduled'
        ),
        'main_polling_initialisation_delay_months': Parameter(
            Types.INT,
            'Delay in months at initialisation for first main polling event'
        ),
        'rti_recovery_initialisation_delay_months': Parameter(
            Types.INT,
            'Delay in months at initialisation for rti recovery event'
        ),
        'rti_check_death_no_med_initialisation_delay_months': Parameter(
            Types.INT,
            'Delay in months at initialisation for rti check death no med event'
        ),
        'non_permanent_tbi_recovery_months': Parameter(
            Types.INT,
            'Recovery duration in months for tbi if non permanent; sets  date for date_to_remove_daly_column'
        ),
        'rti_fracture_cast_recovery_weeks': Parameter(
            Types.INT,
            'Recovery duration in weeks for fracture cast; sets  date for date_to_remove_daly_column'
        ),
        'rti_open_fracture_recovery_months': Parameter(
            Types.INT,
            'Recovery duration in months for open fracture; sets  date for date_to_remove_daly_column'
        ),
        'rti_burn_recovery_weeks': Parameter(
            Types.INT,
            'Recovery duration in weeks for rti burn; sets  date for date_to_remove_daly_column'
        ),

    }

    PROPERTIES = {
        'rt_disability': Property(Types.REAL, 'disability weight for current month'),
        'rt_disability_permanent': Property(Types.REAL, 'disability weight incurred permanently'),
        'rt_date_inj': Property(Types.DATE, 'date of latest injury'),
        'rt_road_traffic_inc': Property(Types.BOOL, 'involved in a road traffic injury'),
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

    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        """ Reads the parameters used in the RTI module"""
        p = self.parameters

        dfd = read_csv_files(resourcefilepath / 'ResourceFile_RTI', files='parameter_values')
        self.load_parameters_from_dataframe(dfd)
        
        # Load emulator
        self.RTI_emulator = CTGANSynthesizer.load(filepath= resourcefilepath / 'ResourceFile_RTI/RTI_emulator.pkl')

    def initialise_population(self, population):
        """Sets up the default properties used in the RTI module and applies them to the dataframe. The default state
        for the RTI module is that people haven't been involved in a road traffic accident and are therefor alive and
        healthy."""
        df = population.props
        df.loc[df.is_alive, 'rt_disability'] = 0  # default: no DALY
        df.loc[df.is_alive, 'rt_disability_permanent'] = 0  # default: no DALY
        df.loc[df.is_alive, 'rt_road_traffic_inc'] = False
        df.loc[df.is_alive, 'rt_date_inj'] = pd.NaT
        
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
        p = self.parameters
        # Begin modelling road traffic injuries

        sim.schedule_event(RTIPollingEvent(self), sim.date +
                           DateOffset(months=p['main_polling_initialisation_delay_months']))
    
        # If all services are included, set everything to True
        if sim.modules['HealthSystem'].service_availability == ['*']:
            for i in sim.modules['EmulatedRTI'].Rti_Services:
                sim.modules['EmulatedRTI'].HS_conditions[i] = True
        else:
            for i in sim.modules['EmulatedRTI'].Rti_Services:
                if (i + '_*') in sim.modules['HealthSystem'].service_availability:
                    sim.modules['EmulatedRTI'].HS_conditions[i] = True
                else:
                    sim.modules['EmulatedRTI'].HS_conditions[i] = False


    def on_birth(self, mother_id, child_id):
        """
        When a person is born this function sets up the default properties for the road traffic injuries module
        :param mother_id: The mother
        :param child_id: The newborn
        :return: n/a
        """
        df.at[child_id, 'rt_disability'] = 0  # default: no DALY
        df.at[child_id, 'rt_disability_permanent'] = 0  # default: no DALY
        df.at[child_id, 'rt_road_traffic_inc'] = False
        df.at[child_id, 'rt_date_inj'] = pd.NaT

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.
        logger.debug(key='rti_general_message', data='This is RTI reporting my daly values')
        df = self.sim.population.props
        disability_series_for_alive_persons = df.loc[df.is_alive, "rt_disability"]
        return disability_series_for_alive_persons

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
        """Schedule to take place
        """
        super().__init__(module, frequency=DateOffset(months=module.parameters['main_polling_frequency']))
        p = module.parameters
        # Parameters which transition the model between states
        self.base_1m_prob_rti = (p['base_rate_injrti'] / 12)
        if 'reduce_incidence' in p['allowed_interventions']:
            self.base_1m_prob_rti = self.base_1m_prob_rti * p['intervention_incidence_reduction_factor']
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
        
        rt_current_non_ind = df.index[df.is_alive & ~df.rt_road_traffic_inc]
        
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
        
        if len(selected_for_rti)>0:
            # This is where we want to replace normal course of events for RTI with emulator.

            # First, sample outcomes for individuals which were selected_for_rti.
            # For now, don't consider properties of individual when sampling outcome. All we care about is the number of samples.
            condition_for_Rti = Condition(
                num_rows=len(selected_for_rti),
                column_values=self.sim.modules['EmulatedRTI'].HS_conditions
            )
            NN_model = self.sim.modules['EmulatedRTI'].RTI_emulator.sample_from_conditions(
                conditions=[condition_for_Rti],
            )

            # HS USAGE
            # Get the total number of different types of appts that will be accessed as a result of this polling event and add to rolling count.
            for column in self.sim.modules['EmulatedRTI'].HS_Use_Type:
                self.sim.modules['EmulatedRTI'].HS_Use_by_RTI[column] += NN_model[column].sum()  # Sum all values in the column
     
            # Change current properties of the individual and schedule resolution of event.
            count = 0
            for person_id in selected_for_rti:

                # These properties are determined by the NN sampling
                is_alive_after_RTI = NN_model.loc[count,'is_alive_after_RTI']
        
                # Why does this require an int wrapper to work with DateOffset?
                duration_days = int(NN_model.loc[count,'duration_days'])
                
                # Individual experiences an immediate death
                if is_alive_after_RTI is False and duration_days == 0:

                    # For each person selected to experience pre-hospital mortality, schedule an InstantaneosDeath event
                    self.sim.modules['Demography'].do_death(individual_id=individual_id, cause="RTI_imm_death",
                                                            originating_module=self.module)
                                                            
                # Else individual doesn't immediately die, therefore schedule resolution
                else:
                    # Set disability to what will be the average over duration of the episodef
                    df.loc[person_id,'rt_disability'] = NN_model.loc[count,'rt_disability_average']

                    # Make sure this person is not 'discoverable' by polling event next month.
                    #df.loc[person_id,'rt_inj_severity'] = 'mild' # instead of "none", but actually we don't know how severe it is
                    
                    # Schedule resolution
                    if is_alive_after_RTI:
                        # Store permanent disability incurred now to be accessed when Recovery Event is invoked.
                        df.loc[person_id,'rt_disability_permanent'] = NN_model.loc[count,'rt_disability_permanent']
                        self.sim.schedule_event(RTI_NNResolution_Recovery_Event(self.module, person_id), df.loc[person_id, 'rt_date_inj']  + DateOffset(days=duration_days))
                    else:
                        self.sim.schedule_event(RTI_NNResolution_Death_Event(self.module, person_id), df.loc[person_id, 'rt_date_inj']  + DateOffset(days=duration_days))
                    
                count += 1

class RTI_NNResolution_Death_Event(Event, IndividualScopeEventMixin):
    """This is an individual-level event that determines the end of the incindent for individual via death"""
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
    def apply(self, person_id):
        # For now invoking RTI_death_with_med, although technically we don't know whether individual accessed HSI or not.
        # How finely do we want to really resolve RTI deaths?
        self.sim.modules['Demography'].do_death(individual_id=person_id, cause="RTI_death_with_med",
                                            originating_module=self.module)
 
class RTI_NNResolution_Recovery_Event(Event, IndividualScopeEventMixin):
    """This is an individual-level event that determines the end of the incindent for individual via death"""
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
    def apply(self, person_id):
        df  = self.sim.population.props
        # Updat rt disability with long term outcome of accident
        df.loc[person_id,'rt_disability'] = df.loc[person_id,'rt_disability_permanent']
        # Ensure that this person will be 'eligible' for rti injury again
        df.loc[person_id,'rt_road_traffic_inc'] = False

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
        assert isinstance(module, EmulatedRTI)
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
        population_with_injuries = df.loc[df.rt_road_traffic_inc]
        # ================================= Injury severity ===========================================================
        population_subsets_with_injuries = {
            "rural": population_with_injuries.loc[~population_with_injuries.li_urban],
            "urban": population_with_injuries.loc[population_with_injuries.li_urban],
        }
        proportion_severely_injured = {
            label: (
                len(pop_subset.loc[pop_subset['rt_inj_severity'] == 'severe'])
                / len(pop_subset)
            ) if len(pop_subset) > 0 else float("nan")
            for label, pop_subset in population_subsets_with_injuries.items()
        }
        self.totmild += (population_with_injuries.rt_inj_severity == "mild").sum()
        self.totsevere += (population_with_injuries.rt_inj_severity == "severe").sum()
        dict_to_output = {
            'total_mild_injuries': self.totmild,
            'total_severe_injuries': self.totsevere,
            'proportion_severe_rural': proportion_severely_injured["rural"],
            'proportion_severe_urban': proportion_severely_injured["urban"],
        }
        logger.info(key='injury_severity',
                    data=dict_to_output,
                    description='severity of injuries in simulation')
        # ==================================== Incidence ==============================================================
        # How many were involved in a RTI
        n_in_RTI = int(df.rt_road_traffic_inc.sum())
        children_in_RTI = len(df.loc[df.rt_road_traffic_inc & (df['age_years'] < 19)])
        children_alive = len(df.loc[df['age_years'] < 19])
        self.numerator += n_in_RTI
        self.totinjured += n_in_RTI
        # How many were disabled
        n_perm_disabled = int((df.is_alive & df.rt_perm_disability).sum())
        # self.permdis += n_perm_disabled
        n_alive = int(df.is_alive.sum())
        self.denominator += (n_alive - n_in_RTI) * (1 / 12)
        n_immediate_death = int((df.rt_road_traffic_inc & df.rt_imm_death).sum())
        self.deathonscene += n_immediate_death
        diedfromrtiidx = df.index[df.rt_imm_death | df.rt_post_med_death | df.rt_no_med_death | df.rt_death_from_shock |
                                  df.rt_unavailable_med_death]
        n_sought_care = int((df.rt_road_traffic_inc & df.rt_med_int).sum())
        self.soughtmedcare += n_sought_care
        n_death_post_med = int(df.rt_post_med_death.sum())
        self.deathaftermed += n_death_post_med
        self.deathwithoutmed += int(df.rt_no_med_death.sum())
        self.death_inc_numerator += n_immediate_death + n_death_post_med + len(df.loc[df.rt_no_med_death])
        self.death_in_denominator += (n_alive - (n_immediate_death + n_death_post_med + len(df.loc[df.rt_no_med_death])
                                                 )) * \
                                     (1 / 12)
        if self.numerator > 0:
            percent_accidents_result_in_death = \
                (self.deathonscene + self.deathaftermed + self.deathwithoutmed) / self.numerator
        else:
            percent_accidents_result_in_death = float("nan")
        maleinrti = len(df.loc[df.rt_road_traffic_inc & (df['sex'] == 'M')])
        femaleinrti = len(df.loc[df.rt_road_traffic_inc & (df['sex'] == 'F')])

        divider = min(maleinrti, femaleinrti)
        if divider > 0:
            maleinrti = maleinrti / divider
            femaleinrti = femaleinrti / divider
        else:
            maleinrti = 1.0
            femaleinrti = 0.0
        mfratio = [maleinrti, femaleinrti]
        if (n_in_RTI - len(df.loc[df.rt_imm_death])) > 0:
            percent_sought_care = n_sought_care / (n_in_RTI - len(df.loc[df.rt_imm_death]))
        else:
            percent_sought_care = float("nan")

        if n_sought_care > 0:
            percent_died_post_care = n_death_post_med / n_sought_care
        else:
            percent_died_post_care = float("nan")

        if n_sought_care > 0:
            percentage_admitted_to_ICU_or_HDU = len(df.loc[df.rt_med_int & df.rt_in_icu_or_hdu]) / n_sought_care
        else:
            percentage_admitted_to_ICU_or_HDU = float("nan")
        if (n_alive - n_in_RTI) > 0:
            inc_rti = (n_in_RTI / ((n_alive - n_in_RTI) * (1 / 12))) * 100000
        else:
            inc_rti = 0.0
        if (children_alive - children_in_RTI) > 0:
            inc_rti_in_children = (children_in_RTI / ((children_alive - children_in_RTI) * (1 / 12))) * 100000
        else:
            inc_rti_in_children = 0.0
        if (n_alive - len(diedfromrtiidx)) > 0:
            inc_rti_death = (len(diedfromrtiidx) / ((n_alive - len(diedfromrtiidx)) * (1 / 12))) * 100000
        else:
            inc_rti_death = 0.0
        if (n_alive - len(df.loc[df.rt_post_med_death])) > 0:
            inc_post_med_death = (len(df.loc[df.rt_post_med_death]) / ((n_alive - len(df.loc[df.rt_post_med_death])) *
                                                                       (1 / 12))) * 100000
        else:
            inc_post_med_death = 0
        if (n_alive - len(df.loc[df.rt_imm_death])) > 0:
            inc_imm_death = (len(df.loc[df.rt_imm_death]) / ((n_alive - len(df.loc[df.rt_imm_death])) * (1 / 12))) * \
                            100000
        else:
            inc_imm_death = 0.0
        if (n_alive - len(df.loc[df.rt_no_med_death])) > 0:
            inc_death_no_med = (len(df.loc[df.rt_no_med_death]) /
                                ((n_alive - len(df.loc[df.rt_no_med_death])) * (1 / 12))) * 100000
        else:
            inc_death_no_med = 0.0
        if (n_alive - len(df.loc[df.rt_unavailable_med_death])) > 0:
            inc_death_unavailable_med = (len(df.loc[df.rt_unavailable_med_death]) /
                                         ((n_alive - len(df.loc[df.rt_unavailable_med_death])) * (1 / 12))) * 100000
        else:
            inc_death_unavailable_med = 0.0
        if self.fracdenominator > 0:
            frac_incidence = (self.totfracnumber / self.fracdenominator) * 100000
        else:
            frac_incidence = 0.0
        # calculate case fatality ratio for those injured who don't seek healthcare
        did_not_seek_healthcare = len(df.loc[df.rt_road_traffic_inc & ~df.rt_med_int & ~df.rt_diagnosed])
        died_no_healthcare = \
            len(df.loc[df.rt_road_traffic_inc & df.rt_no_med_death & ~df.rt_med_int & ~df.rt_diagnosed])
        if did_not_seek_healthcare > 0:
            cfr_no_med = died_no_healthcare / did_not_seek_healthcare
        else:
            cfr_no_med = float("nan")
        # calculate incidence rate per 100,000 of deaths on scene
        if n_alive > 0:
            inc_death_on_scene = (len(df.loc[df.rt_imm_death]) / n_alive) * 100000 * (1 / 12)
        else:
            inc_death_on_scene = 0.0
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
            percent_related_to_alcohol = 0.0
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
