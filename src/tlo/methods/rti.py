"""
Road traffic injury module.

"""
from pathlib import Path
import pandas as pd
import numpy as np
from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent, Event
from tlo.methods import demography, Metadata
from tlo.methods.healthsystem import HSI_Event
from tlo.lm import LinearModel, LinearModelType, Predictor
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
        'prob_death_with_med_mild': Parameter(
            Types.REAL,
            'Proportion of people who pass away in the following month after medical treatment for injuries with an ISS'
            'score less than or equal to 15'
        ),
        'prob_death_with_med_severe': Parameter(
            Types.REAL,
            'Proportion of people who pass away in the following month after medical treatment for injuries with an ISS'
            'score of 15 '
        ),
        'prob_perm_disability_with_treatment_severe_TBI': Parameter(
            Types.REAL,
            'probability that someone with a treated severe TBI is permanently disabled'
        ),
        'prob_perm_disability_with_treatment_sci': Parameter(
            Types.REAL,
            'probability that someone with a treated spinal cord injury is permanently disabled'
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
        'rr_injrti_mortality_polytrauma': Parameter(
            Types.REAL,
            'Relative risk of mortality for those with polytrauma'
        ),
        'number_of_injured_body_regions_distribution': Parameter(
            Types.LIST,
            'The distribution of number of injured AIS body regions, used to decide how many injuries a person has'
        ),
        'injury_location_distribution': Parameter(
            Types.LIST,
            'The distribution of where injuries are located in the body, based on the AIS body region definition'
        ),
        'head_prob_skin_wound': Parameter(
            Types.REAL,
            'Proportion of head wounds that result in a skin wound'
        ),
        'head_prob_skin_wound_open': Parameter(
            Types.REAL,
            'Proportion of skin wounds in the head that result in an open wound'
        ),
        'head_prob_skin_wound_burn': Parameter(
            Types.REAL,
            'Proportion of skin wounds in the head that result in an open wound'
        ),
        'head_prob_fracture': Parameter(
            Types.REAL,
            'Proportion of head wounds that result in a fractured skull'
        ),
        'head_prob_fracture_unspecified': Parameter(
            Types.REAL,
            'Proportion of skull fractures in an unspecified location in the skull, carrying a lower AIS score'
        ),
        'head_prob_fracture_basilar': Parameter(
            Types.REAL,
            'Proportion of skull fractures in the base of the skull, carrying a higher AIS score'
        ),
        'head_prob_TBI': Parameter(
            Types.REAL,
            'Proportion of head injuries that result in traumatic brain injury'
        ),
        'head_prob_TBI_AIS3': Parameter(
            Types.REAL,
            'Proportion of traumatic brain injuries with an AIS score of 3'
        ),
        'head_prob_TBI_AIS4': Parameter(
            Types.REAL,
            'Proportion of traumatic brain injuries with an AIS score of 4'
        ),
        'head_prob_TBI_AIS5': Parameter(
            Types.REAL,
            'Proportion of traumatic brain injuries with an AIS score of 3'
        ),
        'face_prob_skin_wound': Parameter(
            Types.REAL,
            'Proportion of facial wounds that result in a skin wound'
        ),
        'face_prob_skin_wound_open': Parameter(
            Types.REAL,
            'Proportion of skin wounds in the face that result in an open wound'
        ),
        'face_prob_skin_wound_burn': Parameter(
            Types.REAL,
            'Proportion of skin wounds in the face that result in an open wound'
        ),
        'face_prob_fracture': Parameter(
            Types.REAL,
            'Proportion of facial wounds that result in a fractured skull'
        ),
        'face_prob_fracture_AIS1': Parameter(
            Types.REAL,
            'Proportion of facial fractures with an AIS score of 1'
        ),
        'face_prob_fracture_AIS2': Parameter(
            Types.REAL,
            'Proportion of facial fractures with an AIS score of 2'
        ),
        'face_prob_soft_tissue_injury': Parameter(
            Types.REAL,
            'Proportion of facial injuries that result in soft tissue injury'
        ),
        'face_prob_eye_injury': Parameter(
            Types.REAL,
            'Proportion of facial injuries that result in eye injury'
        ),
        'neck_prob_skin_wound': Parameter(
            Types.REAL,
            'Proportion of neck injuries that result in skin wounds'
        ),
        'neck_prob_skin_wound_open': Parameter(
            Types.REAL,
            'Proportion of skin wounds in the neck that are open wounds'
        ),
        'neck_prob_skin_wound_burn': Parameter(
            Types.REAL,
            'Proportion of skin wounds in the neck that are burns'
        ),
        'neck_prob_soft_tissue_injury': Parameter(
            Types.REAL,
            'Proportion of neck injuries that result in soft tissue injury'
        ),
        'neck_prob_soft_tissue_injury_AIS2': Parameter(
            Types.REAL,
            'Proportion of soft tissue injuries with an AIS score of 2'
        ),
        'neck_prob_soft_tissue_injury_AIS3': Parameter(
            Types.REAL,
            'Proportion of soft tissue injuries with an AIS score of 3'
        ),
        'neck_prob_internal_bleeding': Parameter(
            Types.REAL,
            'Proportion of neck injuries that result in internal bleeding'
        ),
        'neck_prob_internal_bleeding_AIS1': Parameter(
            Types.REAL,
            'Proportion of internal bleeding in the neck with an AIS score of 1'
        ),
        'neck_prob_internal_bleeding_AIS3': Parameter(
            Types.REAL,
            'Proportion of internal bleeding in the neck with an AIS score of 3'
        ),
        'neck_prob_dislocation': Parameter(
            Types.REAL,
            'Proportion of neck injuries that result in a dislocated neck vertebrae'
        ),
        'neck_prob_dislocation_AIS2': Parameter(
            Types.REAL,
            'Proportion dislocated neck vertebrae with an AIS score of 2'
        ),
        'neck_prob_dislocation_AIS3': Parameter(
            Types.REAL,
            'Proportion dislocated neck vertebrae with an AIS score of 3'
        ),
        'thorax_prob_skin_wound': Parameter(
            Types.REAL,
            'Proportion of thorax injuries that result in a skin wound'
        ),
        'thorax_prob_skin_wound_open': Parameter(
            Types.REAL,
            'Proportion of thorax skin wounds that are open wounds'
        ),
        'thorax_prob_skin_wound_burn': Parameter(
            Types.REAL,
            'Proportion of thorax skin wounds that are burns'
        ),
        'thorax_prob_internal_bleeding': Parameter(
            Types.REAL,
            'Proportion of thorax injuries that result in internal bleeding'
        ),
        'thorax_prob_internal_bleeding_AIS1': Parameter(
            Types.REAL,
            'Proportion of internal bleeding in thorax with AIS score of 1'
        ),
        'thorax_prob_internal_bleeding_AIS3': Parameter(
            Types.REAL,
            'Proportion of internal bleeding in thorax with AIS score of 3'
        ),
        'thorax_prob_internal_organ_injury': Parameter(
            Types.REAL,
            'Proportion of thorax injuries that result in internal organ injuries'
        ),
        'thorax_prob_fracture': Parameter(
            Types.REAL,
            'Proportion of thorax injuries that result in rib fractures/ flail chest'
        ),
        'thorax_prob_fracture_ribs': Parameter(
            Types.REAL,
            'Proportion of rib fractures in  thorax fractures'
        ),
        'thorax_prob_fracture_flail_chest': Parameter(
            Types.REAL,
            'Proportion of flail chest in thorax fractures'
        ),
        'thorax_prob_soft_tissue_injury': Parameter(
            Types.REAL,
            'Proportion of thorax injuries resulting in soft tissue injury'
        ),
        'thorax_prob_soft_tissue_injury_AIS1': Parameter(
            Types.REAL,
            'Proportion of soft tissue injuries in the thorax with an AIS score of 1'
        ),
        'thorax_prob_soft_tissue_injury_AIS2': Parameter(
            Types.REAL,
            'Proportion of soft tissue injuries in the thorax with an AIS score of 2'
        ),
        'thorax_prob_soft_tissue_injury_AIS3': Parameter(
            Types.REAL,
            'Proportion of soft tissue injuries in the thorax with an AIS score of 3'
        ),
        'abdomen_prob_skin_wound': Parameter(
            Types.REAL,
            'Proportion of abdomen injuries that are skin wounds'
        ),
        'abdomen_prob_skin_wound_open': Parameter(
            Types.REAL,
            'Proportion skin wounds to the abdomen that are open wounds'
        ),
        'abdomen_prob_skin_wound_burn': Parameter(
            Types.REAL,
            'Proportion skin wounds to the abdomen that are burns'
        ),
        'abdomen_prob_internal_organ_injury': Parameter(
            Types.REAL,
            'Proportion of abdomen injuries that result in internal organ injury'
        ),
        'abdomen_prob_internal_organ_injury_AIS2': Parameter(
            Types.REAL,
            'Proportion of abdomen injuries that result in internal organ injury with an AIS score of 2'
        ),
        'abdomen_prob_internal_organ_injury_AIS3': Parameter(
            Types.REAL,
            'Proportion of abdomen injuries that result in internal organ injury with an AIS score of 2'
        ),
        'abdomen_prob_internal_organ_injury_AIS4': Parameter(
            Types.REAL,
            'Proportion of abdomen injuries that result in internal organ injury with an AIS score of 2'
        ),
        'spine_prob_spinal_cord_lesion': Parameter(
            Types.REAL,
            'Proportion of injuries to spine that result in spinal cord lesions'
        ),
        'spine_prob_spinal_cord_lesion_neck_level': Parameter(
            Types.REAL,
            'Proportion of spinal cord lesions that happen at neck level'
        ),
        'spine_prob_spinal_cord_lesion_neck_level_AIS3': Parameter(
            Types.REAL,
            'Proportion of spinal cord lesions that happen at neck level with an AIS score of 3'
        ),
        'spine_prob_spinal_cord_lesion_neck_level_AIS4': Parameter(
            Types.REAL,
            'Proportion of spinal cord lesions that happen at neck level with an AIS score of 4'
        ),
        'spine_prob_spinal_cord_lesion_neck_level_AIS5': Parameter(
            Types.REAL,
            'Proportion of spinal cord lesions that happen at neck level with an AIS score of 5'
        ),
        'spine_prob_spinal_cord_lesion_neck_level_AIS6': Parameter(
            Types.REAL,
            'Proportion of spinal cord lesions that happen at neck level with an AIS score of 6'
        ),
        'spine_prob_spinal_cord_lesion_below_neck_level': Parameter(
            Types.REAL,
            'Proportion of spinal cord lesions that happen below neck level'
        ),
        'spine_prob_spinal_cord_lesion_below_neck_level_AIS3': Parameter(
            Types.REAL,
            'Proportion of spinal cord lesions that happen below neck level with an AIS score of 3'
        ),
        'spine_prob_spinal_cord_lesion_below_neck_level_AIS4': Parameter(
            Types.REAL,
            'Proportion of spinal cord lesions that happen below neck level with an AIS score of 4'
        ),
        'spine_prob_spinal_cord_lesion_below_neck_level_AIS5': Parameter(
            Types.REAL,
            'Proportion of spinal cord lesions that happen below neck level with an AIS score of 5'
        ),
        'spine_prob_fracture': Parameter(
            Types.REAL,
            'Proportion of spinal injuries that result in vertebrae fractures'
        ),
        'upper_ex_prob_skin_wound': Parameter(
            Types.REAL,
            'Proportion of upper extremity injuries that result in skin wounds'
        ),
        'upper_ex_prob_skin_wound_open': Parameter(
            Types.REAL,
            'Proportion of upper extremity injuries that result in open wounds'
        ),
        'upper_ex_prob_skin_wound_burn': Parameter(
            Types.REAL,
            'Proportion of upper extremity injuries that result in burns'
        ),
        'upper_ex_prob_fracture': Parameter(
            Types.REAL,
            'Proportion of upper extremity injuries that result in fractures'
        ),
        'upper_ex_prob_dislocation': Parameter(
            Types.REAL,
            'Proportion of upper extremity injuries that result in dislocation'
        ),
        'upper_ex_prob_amputation': Parameter(
            Types.REAL,
            'Proportion of upper extremity injuries that result in amputation'
        ),
        'upper_ex_prob_amputation_AIS2': Parameter(
            Types.REAL,
            'Proportion of upper extremity injuries that result in amputation with AIS 2'
        ),
        'upper_ex_prob_amputation_AIS3': Parameter(
            Types.REAL,
            'Proportion of upper extremity injuries that result in amputation with AIS 3'
        ),
        'lower_ex_prob_skin_wound': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in skin wounds'
        ),
        'lower_ex_prob_skin_wound_open': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in open wounds'
        ),
        'lower_ex_prob_skin_wound_burn': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in burns'
        ),
        'lower_ex_prob_fracture': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in fractures'
        ),
        'lower_ex_prob_fracture_AIS1': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in fractures with an AIS of 1'
        ),
        'lower_ex_prob_fracture_AIS2': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in fractures with an AIS of 2'
        ),
        'lower_ex_prob_fracture_AIS3': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in fractures with an AIS of 3'
        ),
        'lower_ex_prob_dislocation': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in dislocation'
        ),
        'lower_ex_prob_amputation': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in amputation'
        ),
        'lower_ex_prob_amputation_AIS2': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in amputation with AIS 2'
        ),
        'lower_ex_prob_amputation_AIS3': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in amputation with AIS 3'
        ),
        'lower_ex_prob_amputation_AIS4': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in amputation with AIS 4'
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
        'daly_dist_code_133': Parameter(
            Types.LIST,
            'Mapping parameter of injury code 133 to the various injuries associated with the code'
        ),
        'daly_dist_code_134': Parameter(
            Types.LIST,
            'Mapping parameter of injury code 134 to the various injuries associated with the code'
        ),
        'daly_dist_code_453': Parameter(
            Types.LIST,
            'Mapping parameter of injury code 453 to the various injuries associated with the code'
        ),
        'daly_dist_code_673': Parameter(
            Types.LIST,
            'Mapping parameter of injury code 673 to the various injuries associated with the code'
        ),
        'daly_dist_codes_674_675': Parameter(
            Types.LIST,
            'Mapping parameter of injury code 674/675 to the various injuries associated with the codes'
        ),
        'daly_dist_code_712': Parameter(
            Types.LIST,
            'Mapping parameter of injury code 712 to the various injuries associated with the code'
        ),
        'daly_dist_code_782': Parameter(
            Types.LIST,
            'Mapping parameter of injury code 782 to the various injuries associated with the code'
        ),
        'daly_dist_code_813': Parameter(
            Types.LIST,
            'Mapping parameter of injury code 813 to the various injuries associated with the code'
        ),
        'daly_dist_code_822': Parameter(
            Types.LIST,
            'Mapping parameter of injury code 822 to the various injuries associated with the code'
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
        'prob_foot_fracture_open': Parameter(
            Types.REAL,
            'The probability that a foot fracture will be open'
        ),
        'prob_patella_tibia_fibula_ankle_fracture_open': Parameter(
            Types.REAL,
            'The probability that a patella/tibia/fibula/ankle fracture will be open'
        ),
        'prob_pelvis_fracture_open': Parameter(
            Types.REAL,
            'The probability that a pelvis fracture will be open'
        ),
        'prob_femur_fracture_open': Parameter(
            Types.REAL,
            'The probability that a femur fracture will be open'
        ),
        'prob_open_fracture_contaminated': Parameter(
            Types.REAL,
            'The probability that an open fracture will be contaminated'
        ),
        'allowed_interventions': Parameter(
            Types.LIST,
            'List of additional interventions that can be included when performing model analysis'
        )
    }

    # Define the module's parameters
    PROPERTIES = {
        'rt_road_traffic_inc': Property(Types.BOOL, 'involved in a road traffic injury'),
        'rt_inj_severity': Property(Types.CATEGORICAL,
                                    'Injury status relating to road traffic injury: none, mild, severe',
                                    categories=['none', 'mild', 'severe'],
                                    ),
        'rt_injury_1': Property(Types.CATEGORICAL, 'Codes for injury 1 from RTI',
                                categories=['none', '112', '113', '133', '133a', '133b', '133c', '133d', '134', '134a',
                                            '134b', '135', '1101', '1114', '211', '212', '241', '2101', '2114', '291',
                                            '342', '343', '361', '363', '322', '323', '3101', '3113', '412', '414',
                                            '461', '463', '453', '453a', '453b', '441', '442', '443', '4101', '4113',
                                            '552', '553', '554', '5101', '5113', '612', '673', '673a', '673b', '674',
                                            '674a', '674b', '675', '675a', '675b', '676', '712', '712a', '712b', '712c',
                                            '722', '782', '782a', '782b', '782c', '783', '7101', '7113', '811', '813do',
                                            '812', '813eo', '813', '813a', '813b', '813bo', '813c', '813co', '822',
                                            '822a', '822b', '882', '883', '884', '8101', '8113', 'P133', 'P133a',
                                            'P133b', 'P133c', 'P133d', 'P134', 'P134a', 'P134b', 'P135', 'P673',
                                            'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675', 'P675a', 'P675b',
                                            'P676', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883', 'P884']),
        'rt_injury_2': Property(Types.CATEGORICAL, 'Codes for injury 2 from RTI',
                                categories=['none', '112', '113', '133', '133a', '133b', '133c', '133d', '134', '134a',
                                            '134b', '135', '1101', '1114', '211', '212', '241', '2101', '2114', '291',
                                            '342', '343', '361', '363', '322', '323', '3101', '3113', '412', '414',
                                            '461', '463', '453', '453a', '453b', '441', '442', '443', '4101', '4113',
                                            '552', '553', '554', '5101', '5113', '612', '673', '673a', '673b', '674',
                                            '674a', '674b', '675', '675a', '675b', '676', '712', '712a', '712b', '712c',
                                            '722', '782', '782a', '782b', '782c', '783', '7101', '7113', '811', '813do',
                                            '812', '813eo', '813', '813a', '813b', '813bo', '813c', '813co', '822',
                                            '822a', '822b', '882', '883', '884', '8101', '8113', 'P133', 'P133a',
                                            'P133b', 'P133c', 'P133d', 'P134', 'P134a', 'P134b', 'P135', 'P673',
                                            'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675', 'P675a', 'P675b',
                                            'P676', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883', 'P884']),
        'rt_injury_3': Property(Types.CATEGORICAL, 'Codes for injury 3 from RTI',
                                categories=['none', '112', '113', '133', '133a', '133b', '133c', '133d', '134', '134a',
                                            '134b', '135', '1101', '1114', '211', '212', '241', '2101', '2114', '291',
                                            '342', '343', '361', '363', '322', '323', '3101', '3113', '412', '414',
                                            '461', '463', '453', '453a', '453b', '441', '442', '443', '4101', '4113',
                                            '552', '553', '554', '5101', '5113', '612', '673', '673a', '673b', '674',
                                            '674a', '674b', '675', '675a', '675b', '676', '712', '712a', '712b', '712c',
                                            '722', '782', '782a', '782b', '782c', '783', '7101', '7113', '811', '813do',
                                            '812', '813eo', '813', '813a', '813b', '813bo', '813c', '813co', '822',
                                            '822a', '822b', '882', '883', '884', '8101', '8113', 'P133', 'P133a',
                                            'P133b', 'P133c', 'P133d', 'P134', 'P134a', 'P134b', 'P135', 'P673',
                                            'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675', 'P675a', 'P675b',
                                            'P676', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883', 'P884']),
        'rt_injury_4': Property(Types.CATEGORICAL, 'Codes for injury 4 from RTI',
                                categories=['none', '112', '113', '133', '133a', '133b', '133c', '133d', '134', '134a',
                                            '134b', '135', '1101', '1114', '211', '212', '241', '2101', '2114', '291',
                                            '342', '343', '361', '363', '322', '323', '3101', '3113', '412', '414',
                                            '461', '463', '453', '453a', '453b', '441', '442', '443', '4101', '4113',
                                            '552', '553', '554', '5101', '5113', '612', '673', '673a', '673b', '674',
                                            '674a', '674b', '675', '675a', '675b', '676', '712', '712a', '712b', '712c',
                                            '722', '782', '782a', '782b', '782c', '783', '7101', '7113', '811', '813do',
                                            '812', '813eo', '813', '813a', '813b', '813bo', '813c', '813co', '822',
                                            '822a', '822b', '882', '883', '884', '8101', '8113', 'P133', 'P133a',
                                            'P133b', 'P133c', 'P133d', 'P134', 'P134a', 'P134b', 'P135', 'P673',
                                            'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675', 'P675a', 'P675b',
                                            'P676', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883', 'P884']),
        'rt_injury_5': Property(Types.CATEGORICAL, 'Codes for injury 5 from RTI',
                                categories=['none', '112', '113', '133', '133a', '133b', '133c', '133d', '134', '134a',
                                            '134b', '135', '1101', '1114', '211', '212', '241', '2101', '2114', '291',
                                            '342', '343', '361', '363', '322', '323', '3101', '3113', '412', '414',
                                            '461', '463', '453', '453a', '453b', '441', '442', '443', '4101', '4113',
                                            '552', '553', '554', '5101', '5113', '612', '673', '673a', '673b', '674',
                                            '674a', '674b', '675', '675a', '675b', '676', '712', '712a', '712b', '712c',
                                            '722', '782', '782a', '782b', '782c', '783', '7101', '7113', '811', '813do',
                                            '812', '813eo', '813', '813a', '813b', '813bo', '813c', '813co', '822',
                                            '822a', '822b', '882', '883', '884', '8101', '8113', 'P133', 'P133a',
                                            'P133b', 'P133c', 'P133d', 'P134', 'P134a', 'P134b', 'P135', 'P673',
                                            'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675', 'P675a', 'P675b',
                                            'P676', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883', 'P884']),
        'rt_injury_6': Property(Types.CATEGORICAL, 'Codes for injury 6 from RTI',
                                categories=['none', '112', '113', '133', '133a', '133b', '133c', '133d', '134', '134a',
                                            '134b', '135', '1101', '1114', '211', '212', '241', '2101', '2114', '291',
                                            '342', '343', '361', '363', '322', '323', '3101', '3113', '412', '414',
                                            '461', '463', '453', '453a', '453b', '441', '442', '443', '4101', '4113',
                                            '552', '553', '554', '5101', '5113', '612', '673', '673a', '673b', '674',
                                            '674a', '674b', '675', '675a', '675b', '676', '712', '712a', '712b', '712c',
                                            '722', '782', '782a', '782b', '782c', '783', '7101', '7113', '811', '813do',
                                            '812', '813eo', '813', '813a', '813b', '813bo', '813c', '813co', '822',
                                            '822a', '822b', '882', '883', '884', '8101', '8113', 'P133', 'P133a',
                                            'P133b', 'P133c', 'P133d', 'P134', 'P134a', 'P134b', 'P135', 'P673',
                                            'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675', 'P675a', 'P675b',
                                            'P676', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883', 'P884']),
        'rt_injury_7': Property(Types.CATEGORICAL, 'Codes for injury 7 from RTI',
                                categories=['none', '112', '113', '133', '133a', '133b', '133c', '133d', '134', '134a',
                                            '134b', '135', '1101', '1114', '211', '212', '241', '2101', '2114', '291',
                                            '342', '343', '361', '363', '322', '323', '3101', '3113', '412', '414',
                                            '461', '463', '453', '453a', '453b', '441', '442', '443', '4101', '4113',
                                            '552', '553', '554', '5101', '5113', '612', '673', '673a', '673b', '674',
                                            '674a', '674b', '675', '675a', '675b', '676', '712', '712a', '712b', '712c',
                                            '722', '782', '782a', '782b', '782c', '783', '7101', '7113', '811', '813do',
                                            '812', '813eo', '813', '813a', '813b', '813bo', '813c', '813co', '822',
                                            '822a', '822b', '882', '883', '884', '8101', '8113', 'P133', 'P133a',
                                            'P133b', 'P133c', 'P133d', 'P134', 'P134a', 'P134b', 'P135', 'P673',
                                            'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675', 'P675a', 'P675b',
                                            'P676', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883', 'P884']),
        'rt_injury_8': Property(Types.CATEGORICAL, 'Codes for injury 8 from RTI',
                                categories=['none', '112', '113', '133', '133a', '133b', '133c', '133d', '134', '134a',
                                            '134b', '135', '1101', '1114', '211', '212', '241', '2101', '2114', '291',
                                            '342', '343', '361', '363', '322', '323', '3101', '3113', '412', '414',
                                            '461', '463', '453', '453a', '453b', '441', '442', '443', '4101', '4113',
                                            '552', '553', '554', '5101', '5113', '612', '673', '673a', '673b', '674',
                                            '674a', '674b', '675', '675a', '675b', '676', '712', '712a', '712b', '712c',
                                            '722', '782', '782a', '782b', '782c', '783', '7101', '7113', '811', '813do',
                                            '812', '813eo', '813', '813a', '813b', '813bo', '813c', '813co', '822',
                                            '822a', '822b', '882', '883', '884', '8101', '8113', 'P133', 'P133a',
                                            'P133b', 'P133c', 'P133d', 'P134', 'P134a', 'P134b', 'P135', 'P673',
                                            'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675', 'P675a', 'P675b',
                                            'P676', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883', 'P884']),
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
        'rt_debugging_DALY_wt': Property(Types.REAL, 'The true value of the DALY weight burden')
    }

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,  # Disease modules: Any disease module should carry this label.
        Metadata.USES_SYMPTOMMANAGER,  # The 'Symptom Manager' recognises modules with this label.
        Metadata.USES_HEALTHSYSTEM,  # The 'HealthSystem' recognises modules with this label.
        Metadata.USES_HEALTHBURDEN  # The 'HealthBurden' module recognises modules with this label.
    }

    def read_parameters(self, data_folder):
        """ Reads the parameters used in the RTI module"""
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_RTI.xlsx', sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)
        if "HealthBurden" in self.sim.modules.keys():
            # get the DALY weights of the seq associated with road traffic injuries
            self.parameters["daly_wt_unspecified_skull_fracture"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1674
            )
            self.parameters["daly_wt_basilar_skull_fracture"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1675
            )
            self.parameters["daly_wt_epidural_hematoma"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1676
            )
            self.parameters["daly_wt_subdural_hematoma"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1677
            )
            self.parameters["daly_wt_subarachnoid_hematoma"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1678
            )
            self.parameters["daly_wt_brain_contusion"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1679
            )
            self.parameters["daly_wt_intraventricular_haemorrhage"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1680
            )
            self.parameters["daly_wt_diffuse_axonal_injury"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1681
            )
            self.parameters["daly_wt_subgaleal_hematoma"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1682
            )
            self.parameters["daly_wt_midline_shift"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1683
            )
            self.parameters["daly_wt_facial_fracture"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1684
            )
            self.parameters["daly_wt_facial_soft_tissue_injury"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1685
            )
            self.parameters["daly_wt_eye_injury"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1686
            )
            self.parameters["daly_wt_neck_soft_tissue_injury"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1687
            )
            self.parameters["daly_wt_neck_internal_bleeding"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1688
            )
            self.parameters["daly_wt_neck_dislocation"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1689
            )
            self.parameters["daly_wt_chest_wall_bruises_hematoma"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1690
            )
            self.parameters["daly_wt_hemothorax"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1691
            )
            self.parameters["daly_wt_lung_contusion"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1692
            )
            self.parameters["daly_wt_diaphragm_rupture"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1693
            )
            self.parameters["daly_wt_rib_fracture"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1694
            )
            self.parameters["daly_wt_flail_chest"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1695
            )
            self.parameters["daly_wt_chest_wall_laceration"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1696
            )
            self.parameters["daly_wt_closed_pneumothorax"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1697
            )
            self.parameters["daly_wt_open_pneumothorax"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1698
            )
            self.parameters["daly_wt_surgical_emphysema"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1699
            )
            self.parameters["daly_wt_abd_internal_organ_injury"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1700
            )
            self.parameters["daly_wt_spinal_cord_lesion_neck_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1701
            )
            self.parameters["daly_wt_spinal_cord_lesion_neck_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1702
            )
            self.parameters["daly_wt_spinal_cord_lesion_below_neck_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1703
            )
            self.parameters["daly_wt_spinal_cord_lesion_below_neck_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1704
            )
            self.parameters["daly_wt_vertebrae_fracture"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1705
            )
            self.parameters["daly_wt_clavicle_scapula_humerus_fracture"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1706
            )
            self.parameters["daly_wt_hand_wrist_fracture_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1707
            )
            self.parameters["daly_wt_hand_wrist_fracture_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1708
            )
            self.parameters["daly_wt_radius_ulna_fracture_short_term_with_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1709
            )
            self.parameters["daly_wt_radius_ulna_fracture_long_term_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1710
            )
            self.parameters["daly_wt_dislocated_shoulder"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1711
            )
            self.parameters["daly_wt_amputated_finger"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1712
            )
            self.parameters["daly_wt_amputated_thumb"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1713
            )
            self.parameters["daly_wt_unilateral_arm_amputation_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1714
            )
            self.parameters["daly_wt_unilateral_arm_amputation_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1715
            )
            self.parameters["daly_wt_bilateral_arm_amputation_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1716
            )
            self.parameters["daly_wt_bilateral_arm_amputation_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1717
            )
            self.parameters["daly_wt_foot_fracture_short_term_with_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1718
            )
            self.parameters["daly_wt_foot_fracture_long_term_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1719
            )
            self.parameters["daly_wt_patella_tibia_fibula_fracture_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1720
            )
            self.parameters["daly_wt_patella_tibia_fibula_fracture_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1721
            )
            self.parameters["daly_wt_hip_fracture_short_term_with_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1722
            )
            self.parameters["daly_wt_hip_fracture_long_term_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1723
            )
            self.parameters["daly_wt_hip_fracture_long_term_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1724
            )
            self.parameters["daly_wt_pelvis_fracture_short_term"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1725
            )
            self.parameters["daly_wt_pelvis_fracture_long_term"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1726
            )
            self.parameters["daly_wt_femur_fracture_short_term"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1727
            )
            self.parameters["daly_wt_femur_fracture_long_term_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1728
            )
            self.parameters["daly_wt_dislocated_hip"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1729
            )
            self.parameters["daly_wt_dislocated_knee"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1730
            )
            self.parameters["daly_wt_amputated_toes"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1731
            )
            self.parameters["daly_wt_unilateral_lower_limb_amputation_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1732
            )
            self.parameters["daly_wt_unilateral_lower_limb_amputation_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1733
            )
            self.parameters["daly_wt_bilateral_lower_limb_amputation_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1734
            )
            self.parameters["daly_wt_bilateral_lower_limb_amputation_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1735
            )
            self.parameters["daly_wt_burns_greater_than_20_percent_body_area"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1736
            )
            self.parameters["daly_wt_burns_less_than_20_percent_body_area_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1737
            )
            self.parameters["daly_wt_burns_less_than_20_percent_body_area_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1738
            )
            # self.sim.modules["HealthSystem"].register_disease_module(self)
        p = self.parameters
        # ================== Test the parameter distributions to see whether they sum to roughly one ===============
        # test the distribution of the number of injured body regions
        assert 0.9999 < sum(p['number_of_injured_body_regions_distribution'][1]) < 1.0001, \
            "The number of injured body region distribution doesn't sum to one"
        # test the injury location distribution
        assert 0.9999 < sum(p['injury_location_distribution'][1]) < 1.0001, \
            "The injured body region distribution doesn't sum to one"
        # test the distributions used to assign daly weights for certain injury codes
        daly_weight_distributions = [val for key, val in p.items() if 'daly_dist_code_' in key]
        for dist in daly_weight_distributions:
            assert 0.9999 < sum(dist) < 1.0001, 'daly weight distribution does not sum to one'
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
        # create a generic severe trauma symtom, which forces people into the health system
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(
                name='severe_trauma',
                emergency_in_adults=True,
                emergency_in_children=True
            )
        )

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
        columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                   'rt_injury_7', 'rt_injury_8']
        persons_injuries = df.loc[[person_id], columns]
        # ================================ Fractures require x-rays ============================================
        # Do they have a fracture which needs an x ray
        fracture_codes = ['112', '113', '211', '212', '412', '414', '612',
                          '712a', '712b', '712c', '811', '812', '813a', '813b', '813c', '822a', '822b',
                          '813bo', '813co', '813do', '813eo']
        idx, counts = self.rti_find_and_count_injuries(persons_injuries, fracture_codes)
        if len(idx) > 0:
            the_appt_footprint['DiagRadio'] = 1
        # ========================= Traumatic brain injuries require ct scan ===================================
        # Do they have a TBI which needs a ct-scan
        codes = ['133', '134', '135']
        idx, counts = self.rti_find_and_count_injuries(persons_injuries, codes)
        if len(idx) > 0:
            the_appt_footprint['Tomography'] = 1  # This appointment requires a ct scan
        # ============================= Abdominal trauma requires ct scan ======================================
        # Do they have abdominal trauma
        codes = ['552', '553', '554']
        idx, counts = self.rti_find_and_count_injuries(persons_injuries, codes)
        if len(idx) > 0:
            the_appt_footprint['Tomography'] = 1

        # ============================== Spinal cord injury require x ray ======================================
        # Do they have a spinal cord injury
        codes = ['673', '674', '675', '676']
        idx, counts = self.rti_find_and_count_injuries(persons_injuries, codes)
        if len(idx) > 0:
            the_appt_footprint['DiagRadio'] = 1  # This appointment requires an x-ray

        # ============================== Dislocations require x ray ============================================
        # Do they have a dislocation
        codes = ['322', '323', '722', '822']
        idx, counts = self.rti_find_and_count_injuries(persons_injuries, codes)
        if len(idx) > 0:
            the_appt_footprint['DiagRadio'] = 1  # This appointment requires an x-ray

        # --------------------------------- Soft tissue injury in neck -----------------------------------------
        # Do they have soft tissue injury in the neck which requires a ct scan and an x ray
        codes = ['342', '343']
        idx, counts = self.rti_find_and_count_injuries(persons_injuries, codes)
        if len(idx) > 0:
            the_appt_footprint['Tomography'] = 1  # This appointment requires a ct scan
            the_appt_footprint['DiagRadio'] = 1  # This appointment requires an x ray

        # --------------------------------- Soft tissue injury in thorax/ lung injury --------------------------
        # Do they have soft tissue injury in the thorax which requires a ct scan and x ray
        codes = ['441', '443', '453']
        idx, counts = self.rti_find_and_count_injuries(persons_injuries, codes)
        if len(idx) > 0:
            the_appt_footprint['Tomography'] = 1  # This appointment requires a ct scan
            the_appt_footprint['DiagRadio'] = 1  # This appointment requires an x ray

        # ----------------------------- Internal bleeding ------------------------------------------------------
        # Do they have internal bleeding which requires a ct scan
        codes = ['361', '363', '461', '463']
        idx, counts = self.rti_find_and_count_injuries(persons_injuries, codes)
        if len(idx) > 0:
            the_appt_footprint['Tomography'] = 1  # This appointment requires a ct scan

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
        df.loc[df.is_alive, 'rt_MAIS_military_score'] = 0
        df.loc[df.is_alive, 'rt_date_death_no_med'] = pd.NaT
        df.loc[df.is_alive, 'rt_debugging_DALY_wt'] = 0
        for index, row in df.iterrows():
            df.at[index, 'rt_date_to_remove_daly'] = [pd.NaT] * 8  # no one has any injuries to remove dalys for
            df.at[index, 'rt_injuries_to_cast'] = []
            df.at[index, 'rt_injuries_for_minor_surgery'] = []
            df.at[index, 'rt_injuries_for_major_surgery'] = []
            df.at[index, 'rt_injuries_to_heal_with_time'] = []
            df.at[index, 'rt_injuries_for_open_fracture_treatment'] = []

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
        # Begin logging the RTI events
        event = RTI_Logging_Event(self)
        sim.schedule_event(event, sim.date + DateOffset(months=0))
        # Begin modelling road traffic injuries
        event = RTIPollingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=0))
        # Begin checking whether the persons injuries are healed
        event = RTI_Recovery_Event(self)
        sim.schedule_event(event, sim.date + DateOffset(months=0))
        # Begin checking whether those with untreated injuries die
        event = RTI_Check_Death_No_Med(self)
        sim.schedule_event(event, sim.date + DateOffset(months=0))

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
        # Check to see whether they have been sent here from A and E and haven't been died due to the rti module
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person seeking treatment for RTI already died'
        assert df.loc[person_id, 'rt_diagnosed']
        # Get the relevant information about their injuries
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        # check this person is injured, search they have an injury code that isn't "none"
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries,
                                                      self.PROPERTIES.get('rt_injury_1').categories[1:])
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
        # Check to see whether they have been sent here from RTI_MedicalIntervention and they haven't died due to rti
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'
        assert df.at[person_id, 'rt_med_int'], 'person sent here not been through RTI_MedInt'
        # Determine what injuries are able to be treated by surgery by checking the injury codes which are currently
        # treated in this simulation, it seems there is a limited available to treat spinal cord injuries and chest
        # trauma in Malawi, so these are initially left out, but we will test different scenarios to see what happens
        # when we include those treatments
        surgically_treated_codes = ['112', '811', '812', '813a', '813b', '813c', '133a', '133b', '133c', '133d', '134a',
                                    '134b', '135', '552', '553', '554', '342', '343', '414', '361', '363',
                                    '782', '782a', '782b', '782c', '783', '822a', '882', '883', '884',
                                    'P133a', 'P133b', 'P133c', 'P133d', 'P134a', 'P134b', 'P135', 'P782a', 'P782b',
                                    'P782c', 'P783', 'P882', 'P883', 'P884'
                                    ]

        # If we allow surgical treatment of spinal cord injuries, extend the surgically treated codes to include spinal
        # cord injury codes
        if 'include_spine_surgery' in self.allowed_interventions:
            additional_codes = ['673a', '673b', '674a', '674b', '675a', '675b', '676', 'P673a', 'P673b', 'P674',
                                'P674a', 'P674b', 'P675', 'P675a', 'P675b', 'P676']
            for code in additional_codes:
                surgically_treated_codes.append(code)
        # If we allow surgical treatment of chest trauma, extend the surgically treated codes to include chest trauma
        # codes.
        if 'include_thoroscopy' in self.allowed_interventions:
            additional_codes = ['441', '443', '453', '453a', '453b', '463']
            for code in additional_codes:
                surgically_treated_codes.append(code)
        assert len(set(df.loc[person_id, 'rt_injuries_for_major_surgery']) & set(surgically_treated_codes)) > 0, \
            'This person has asked for surgery but does not have an appropriate injury'
        person = df.iloc[person_id]
        # isolate the relevant injury information
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        # Check whether the person sent to surgery has an injury which actually requires surgery
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries, surgically_treated_codes)
        assert counts > 0, 'This person has been sent to major surgery without the right injuries'
        # If this person is alive schedule major surgeries
        if person.is_alive:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Major_Surgeries(module=self,
                                                  person_id=person_id),
                priority=0,
                topen=self.sim.date + DateOffset(days=count),
                tclose=self.sim.date + DateOffset(days=15))

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
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'
        assert df.at[person_id, 'rt_med_int'], 'Person sent for treatment did not go through rti med int'
        # Isolate the person
        person = df.iloc[person_id]
        # state the codes treated by minor surgery
        surgically_treated_codes = ['211', '212', '291', '241', '322', '323', '722', '811', '812', '813a',
                                    '813b', '813c']
        # check that this person's injuries that were decided to be treated with a minor surgery and the injuries
        # actually treated by minor surgeries coincide
        assert len(set(df.loc[person_id, 'rt_injuries_for_minor_surgery']) & set(surgically_treated_codes)) > 0, \
            'This person has asked for a minor surgery but does not need it'
        # Isolate the relevant injury information
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        # Check whether the person requesting minor surgeries has an injury that requires minor surgery
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries, surgically_treated_codes)
        assert counts > 0
        # if this person is alive schedule the minor surgery
        if person.is_alive:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Minor_Surgeries(module=self,
                                                  person_id=person_id),
                priority=0,
                topen=self.sim.date + DateOffset(days=count),
                tclose=self.sim.date + DateOffset(days=15))

    def rti_acute_pain_management(self, person_id):
        """
        Function called in HSI_RTI_MedicalIntervention to request pain management. This should be called for every alive
        injured person, regardless of what their injuries are. In this function we test whether they meet the
        requirements to recieve for pain relief, that is they are alive and currently receiving medical treatment.
        :param person_id: The person requesting pain management
        :return: n/a
        """
        df = self.sim.population.props
        # Check to see whether they have been sent here from RTI_MedicalIntervention and they haven't died due to rti
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'
        assert df.at[person_id, 'rt_med_int'], 'person sent here not been through rti med int'
        # Isolate the relevant injury information
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        # check this person is injured, search they have an injury code that isn't "none".
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries,
                                                      self.PROPERTIES.get('rt_injury_1').categories[1:])
        assert counts > 0, 'This person has asked for pain relief despite not being injured'
        person = df.iloc[person_id]
        # if the person is alive schedule pain management
        if person.is_alive:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Acute_Pain_Management(module=self,
                                                        person_id=person_id),
                priority=0,
                topen=self.sim.date,
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
        # Check to see whether they have been sent here from RTI_MedicalIntervention and they haven't died due to rti
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'
        assert df.at[person_id, 'rt_med_int'], 'person sent here not been through rti med int'
        # Isolate the relevant injury information
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        laceration_codes = ['1101', '2101', '3101', '4101', '5101', '6101', '7101', '8101']
        # Check they have a laceration which needs stitches
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries, laceration_codes)
        assert counts > 0, "This person has asked for stiches, but doens't have a laceration"
        person = df.iloc[person_id]
        # if the person is alive request the hsi event
        if person.is_alive:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Suture(module=self,
                                         person_id=person_id),
                priority=0,
                topen=self.sim.date,
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
        # Check to see whether they have been sent here from RTI_MedicalIntervention and they haven't died due to rti
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'
        assert df.at[person_id, 'rt_med_int'], 'person not been through rti med int'
        # Isolate the relevant injury information
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        burn_codes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113']
        # Check to see whether they have a burn which needs treatment
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries, burn_codes)
        assert counts > 0, "This person has asked for burn treatment, but doens't have any burns"
        person = df.iloc[person_id]
        # if this person is alive ask for the hsi event
        if person.is_alive:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Burn_Management(module=self,
                                                  person_id=person_id),
                priority=0,
                topen=self.sim.date,
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
        # Check to see whether they have been sent here from RTI_MedicalIntervention and they haven't died due to rti
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'
        assert df.at[person_id, 'rt_med_int'], 'person sent here not been through rti med int'
        # Isolate the relevant injury information
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        fracture_codes = ['712a', '712b', '712c', '811', '812', '813a', '813b', '813c', '822a', '822b']
        # check that the codes assigned for treatment by rt_injuries_to_cast and the codes treated by rti_fracture_cast
        # coincide
        assert len(set(df.loc[person_id, 'rt_injuries_to_cast']) & set(fracture_codes)) > 0, \
            'This person has asked for a fracture cast'
        # Check they have an injury treated by HSI_RTI_Fracture_Cast
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries, fracture_codes)
        assert counts > 0, "This person has asked for fracture treatment, but doens't have appropriate fractures"
        person = df.iloc[person_id]
        # if this person is alive request the hsi
        if person.is_alive:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Fracture_Cast(module=self,
                                                person_id=person_id),
                priority=0,
                topen=self.sim.date,
                tclose=self.sim.date + DateOffset(days=15)
            )

    def rti_ask_for_open_fracture_treatment(self, person_id, counts):
        """Function called by HSI_RTI_MedicalIntervention to centralise open fracture treatment requests. This function
        schedules an open fracture event, conditional on whether they are alive, being treated and have an appropriate
        injury.

        :param person_id: the person requesting a tetanus jab
        :param counts: the number of open fractures that requires a treatment
        :return: n/a
        """
        df = self.sim.population.props
        # Check to see whether they have been sent here from RTI_MedicalIntervention and are haven't died due to rti
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'
        assert df.at[person_id, 'rt_med_int'], 'person sent here not been through rti med int'
        # Isolate the relevant injury information
        person = df.iloc[person_id]
        open_fracture_codes = ['813bo', '813co', '813do', '813eo']
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        # Check that they have an open fracture
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries, open_fracture_codes)
        assert counts > 0, "This person has requested open fracture treatment but doesn't require one"
        # if the person is alive request the hsi
        if person.is_alive:
            for i in range(0, counts):
                # shedule the treatments, say the treatments occur a day appart for now
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
        # Check to see whether they have been sent here from RTI_MedicalIntervention and are haven't died due to rti
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'
        assert df.at[person_id, 'rt_med_int'], 'person sent here not been through rti med int'
        person = df.iloc[person_id]
        # Isolate the relevant injury information
        codes_for_tetanus = ['1101', '2101', '3101', '4101', '5101', '7101', '8101',
                             '1114', '2114', '3113', '4113', '5113', '7113', '8113']
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        # Check that they have a burn/laceration
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries, codes_for_tetanus)
        assert counts > 0, "This person has requested a tetanus jab but doesn't require one"
        # if this person is alive, ask for the hsi
        if person.is_alive:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Tetanus_Vaccine(module=self,
                                                  person_id=person_id),
                priority=0,
                topen=self.sim.date,
                tclose=self.sim.date + DateOffset(days=15)
            )

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
        # Isolate the relevant injury information
        columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                   'rt_injury_7', 'rt_injury_8']
        df = df.loc[[person_id], columns]
        # Set up the number to iterate over
        injury_numbers = range(1, 9)
        # create empty variables to return if the search doesn't find a code/column
        injury_column = ''
        injury_code = ''
        # Iterate over the list of codes to begin the search
        for code in codes:
            # Iterate over the injury columns
            for injury_number in injury_numbers:
                # Create a dataframe where the rows are those who have injury code 'code' within the column
                # 'rt_injury_(injury_number)', if the person doesn't have 'code' in column 'rt_injury_(injury_number)',
                # then the dataframe is empty
                found = df[df[f"rt_injury_{injury_number}"].str.contains(code)]
                # check if the dataframe is non-empty
                if len(found) > 0:
                    # if the dataframe is non-empty, then we have found the injury column corresponding to the injury
                    # code for person 'person_id'. Assign the found column/code to injury_column and injury_code and
                    # break the for loop.
                    injury_column = f"rt_injury_{injury_number}"
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
        # Isolate the relevant injury information
        columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                   'rt_injury_7', 'rt_injury_8']
        df = df.loc[[person_id], columns]
        # create empty variables to return the columns and codes of the treated injuries
        columns_to_return = []
        codes_to_return = []
        injury_numbers = range(1, 9)
        # iterate over the codes in the list codes and also the injury columns
        for code in codes:
            for injury_number in injury_numbers:
                # Search a sub-dataframe that is non-empty if the code is present is in that column and empty if not
                found = len(df[df[f"rt_injury_{injury_number}"] == code])
                if found > 0:
                    # if the code is in the column, store the column and code in columns_to_return and codes_to_return
                    # respectively
                    columns_to_return.append(f"rt_injury_{injury_number}")
                    codes_to_return.append(code)

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
        # Check that those sent here have been involved in a road traffic accident
        assert sum(df.loc[injured_index, 'rt_road_traffic_inc']) == len(injured_index)
        # Check everyone here has at least one injury to be given a daly weight to
        assert sum(df.loc[injured_index, 'rt_injury_1'] != "none") == len(injured_index)
        # Check everyone here is alive and hasn't died due to rti
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert (sum(~df.loc[injured_index, 'cause_of_death'].isin(rti_deaths)) == len(injured_index)) & \
               (sum(df.loc[injured_index, 'rt_imm_death']) == 0)
        columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                   'rt_injury_7', 'rt_injury_8']
        selected_for_rti_inj = df.loc[injured_index, columns]
        # =============================== AIS region 1: head ==========================================================
        # ------ Find those with skull fractures and update rt_fracture to match and call for treatment ---------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['112'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_unspecified_skull_fracture
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['113'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_basilar_skull_fracture
        # ------ Find those with traumatic brain injury and update rt_tbi to match and call the TBI treatment ---------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['133a'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_subarachnoid_hematoma
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['133b'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_brain_contusion
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['133c'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_intraventricular_haemorrhage
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['133d'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_subgaleal_hematoma

        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['134a'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_epidural_hematoma
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['134b'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_subdural_hematoma

        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['135'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_diffuse_axonal_injury

        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['1101'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_facial_soft_tissue_injury
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['1114'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_burns_greater_than_20_percent_body_area

        # =============================== AIS region 2: face ==========================================================
        # ----------------------- Find those with facial fractures and assign DALY weight -----------------------------
        codes = ['211', '212']
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, codes)
        if counts > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_facial_fracture

        # ----------------- Find those with lacerations/soft tissue injuries and assign DALY weight -------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['2101'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_facial_soft_tissue_injury

        # ----------------- Find those with eye injuries and assign DALY weight ---------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['291'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_eye_injury
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['241'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_facial_soft_tissue_injury

        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['2114'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_burns_greater_than_20_percent_body_area
        # =============================== AIS region 3: Neck ==========================================================
        # -------------------------- soft tissue injuries and internal bleeding----------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['342', '343', '361', '363'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_neck_internal_bleeding
        # -------------------------------- neck vertebrae dislocation ------------------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['322', '323'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_neck_dislocation
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['3101'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_facial_soft_tissue_injury
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['3113'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_burns_less_than_20_percent_body_area_without_treatment
        # ================================== AIS region 4: Thorax =====================================================
        # --------------------------------- fractures & flail chest ---------------------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['412'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_rib_fracture
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['414'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_flail_chest
        # ------------------------------------ Internal bleeding ------------------------------------------------------
        # chest wall bruises/hematoma
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['461'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_chest_wall_bruises_hematoma
        # hemothorax
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['463'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_hemothorax
        # -------------------------------- Internal organ injury ------------------------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['453a'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_diaphragm_rupture
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['453b'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_lung_contusion
        # ----------------------------------- Soft tissue injury ------------------------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['442'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_surgical_emphysema
        # ---------------------------------- Pneumothoraxs ------------------------------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['441'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_closed_pneumothorax
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['443'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_open_pneumothorax
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['4101'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_facial_soft_tissue_injury
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['4113'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_burns_less_than_20_percent_body_area_without_treatment
        # ================================== AIS region 5: Abdomen ====================================================
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['552', '553', '554'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_abd_internal_organ_injury
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['5101'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_facial_soft_tissue_injury
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['5113'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_burns_less_than_20_percent_body_area_without_treatment
        # =================================== AIS region 6: spine =====================================================
        # ----------------------------------- vertebrae fracture ------------------------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['612'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_vertebrae_fracture
        # ---------------------------------- Spinal cord injuries -----------------------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['673a'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_spinal_cord_lesion_neck_without_treatment
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['673b'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_spinal_cord_lesion_below_neck_without_treatment
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['674a', '675a'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_spinal_cord_lesion_neck_without_treatment
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['674b', '675b'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_spinal_cord_lesion_below_neck_without_treatment
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['676'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_spinal_cord_lesion_neck_without_treatment

        # ============================== AIS body region 7: upper extremities ======================================
        # ------------------------------------------ fractures ------------------------------------------------------
        # Fracture to Clavicle, scapula, humerus, Hand/wrist, Radius/ulna
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['712a'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_clavicle_scapula_humerus_fracture
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['712b'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_hand_wrist_fracture_without_treatment
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['712c'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_radius_ulna_fracture_short_term_with_without_treatment

        # ------------------------------------ Dislocation of shoulder ---------------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['722'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_dislocated_shoulder
        # ------------------------------------------ Amputations -----------------------------------------------------
        # Amputation of fingers, Unilateral upper limb amputation, Thumb amputation
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['782a'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_amputated_finger
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['782b'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_unilateral_arm_amputation_without_treatment
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['782c'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_amputated_thumb

        # Bilateral upper limb amputation
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['783'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_bilateral_arm_amputation_without_treatment
        # ----------------------------------- cuts and bruises --------------------------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['7101'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_facial_soft_tissue_injury
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['7113'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_burns_less_than_20_percent_body_area_without_treatment
        # ============================== AIS body region 8: Lower extremities ========================================
        # ------------------------------------------ Fractures -------------------------------------------------------
        # Broken foot
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['811'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_foot_fracture_short_term_with_without_treatment
        # Broken foot (open), currently combining the daly weight used for open wounds and the fracture
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['813do'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_foot_fracture_short_term_with_without_treatment + \
                                            self.daly_wt_facial_soft_tissue_injury
        # Broken patella, tibia, fibula
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['812'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_patella_tibia_fibula_fracture_without_treatment
        # Broken foot (open), currently combining the daly weight used for open wounds and the fracture
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['813eo'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_patella_tibia_fibula_fracture_without_treatment + \
                                            self.daly_wt_facial_soft_tissue_injury
        # Broken Hip, Pelvis, Femur other than femoral neck
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['813a'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_hip_fracture_short_term_with_without_treatment
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['813b'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_pelvis_fracture_short_term
        # broken pelvis (open)
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['813bo'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_pelvis_fracture_short_term + \
                                            self.daly_wt_facial_soft_tissue_injury
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['813c'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_femur_fracture_short_term
        # broken femur (open)
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['813co'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_femur_fracture_short_term + \
                                            self.daly_wt_facial_soft_tissue_injury
        # -------------------------------------- Dislocations -------------------------------------------------------
        # Dislocated hip, knee
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['822a'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_dislocated_hip
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['822b'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_dislocated_knee
        # --------------------------------------- Amputations ------------------------------------------------------
        # toes
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['882'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_amputated_toes
        # Unilateral lower limb amputation
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['883'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_unilateral_lower_limb_amputation_without_treatment
        # Bilateral lower limb amputation
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['884'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_bilateral_lower_limb_amputation_without_treatment
        # ------------------------------------ cuts and bruises -----------------------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['8101'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_facial_soft_tissue_injury

        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['8113'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_burns_less_than_20_percent_body_area_without_treatment
        # Store the true sum of DALY weights in the df
        df.loc[injured_index, 'rt_debugging_DALY_wt'] = df.loc[injured_index, 'rt_disability']
        # Find who's disability burden is greater than one
        DALYweightoverlimit = df.index[df['rt_disability'] > 1]
        # Set the total daly weights to one in this case
        df.loc[DALYweightoverlimit, 'rt_disability'] = 1
        # Find who's disability burden is less than one
        DALYweightunderlimit = df.index[df.rt_road_traffic_inc & ~ df.rt_imm_death & (df['rt_disability'] <= 0)]
        # Check that no one has a disability burden less than or equal to zero
        assert len(DALYweightunderlimit) == 0, 'Someone has not been given an injury burden' + \
                                               selected_for_rti_inj.loc[DALYweightunderlimit]
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
        # Check that people who are sent here have had medical treatment
        assert df.loc[person_id, 'rt_med_int'] or df.loc[person_id, 'rt_recovery_no_med']
        # Check everyone here has at least one injury to be alter the daly weight to
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        # check this person is injured, search they have an injury code that isn't "none"
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries,
                                                      self.PROPERTIES.get('rt_injury_1').categories[1:])
        assert counts > 0, 'This person has asked for medical treatment despite not being injured'
        # Check everyone here is alive and hasn't died on scene
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'
        # ------------------------------- Remove the daly weights for treated injuries ---------------------------------
        # ==================================== heal with time injuries =================================================
        for code in codes:
            # keep track of the change in daly weights
            change_in_daly_weights = 0
            if code == '322' or code == '323':
                change_in_daly_weights -= self.daly_wt_neck_dislocation
            if code == '822a':
                change_in_daly_weights -= self.daly_wt_dislocated_hip
            if code == '822b':
                change_in_daly_weights -= self.daly_wt_dislocated_knee
            if code == '112':
                change_in_daly_weights -= self.daly_wt_unspecified_skull_fracture
            if code == '113':
                change_in_daly_weights -= self.daly_wt_basilar_skull_fracture
            if code == '552' or code == '553' or code == '554':
                change_in_daly_weights -= self.daly_wt_abd_internal_organ_injury
            if code == '412':
                change_in_daly_weights -= self.daly_wt_rib_fracture
            if code == '442':
                change_in_daly_weights -= self.daly_wt_surgical_emphysema
            if code == '461':
                change_in_daly_weights -= self.daly_wt_chest_wall_bruises_hematoma
            if code == '612':
                change_in_daly_weights -= self.daly_wt_vertebrae_fracture
            # ========================== Codes 'treated' with stitches  ==============================================
            if code == '1101':
                change_in_daly_weights -= self.daly_wt_facial_soft_tissue_injury
            if code == '2101':
                change_in_daly_weights -= self.daly_wt_facial_soft_tissue_injury
            if code == '3101':
                change_in_daly_weights -= self.daly_wt_facial_soft_tissue_injury
            if code == '4101':
                change_in_daly_weights -= self.daly_wt_facial_soft_tissue_injury
            if code == '5101':
                change_in_daly_weights -= self.daly_wt_facial_soft_tissue_injury
            if code == '7101':
                change_in_daly_weights -= self.daly_wt_facial_soft_tissue_injury
            if code == '8101':
                change_in_daly_weights -= self.daly_wt_facial_soft_tissue_injury
            # ============================== Codes 'treated' with fracture casts ======================================
            if code == '712a':
                change_in_daly_weights -= self.daly_wt_clavicle_scapula_humerus_fracture
            if code == '712b':
                change_in_daly_weights -= self.daly_wt_hand_wrist_fracture_with_treatment
            if code == '712c':
                change_in_daly_weights -= self.daly_wt_radius_ulna_fracture_short_term_with_without_treatment
            if code == '811':
                change_in_daly_weights -= self.daly_wt_foot_fracture_short_term_with_without_treatment
            if code == '812':
                change_in_daly_weights -= self.daly_wt_patella_tibia_fibula_fracture_with_treatment
            # ============================== Codes 'treated' with minor surgery =======================================
            if code == '722':
                change_in_daly_weights -= self.daly_wt_dislocated_shoulder
            if code == '291':
                change_in_daly_weights -= self.daly_wt_eye_injury
            if code == '241':
                change_in_daly_weights -= self.daly_wt_facial_soft_tissue_injury
            if code == '211' or code == '212':
                change_in_daly_weights -= self.daly_wt_facial_fracture
            # ============================== Codes 'treated' with burn management ======================================
            if code == '1114':
                change_in_daly_weights -= self.daly_wt_burns_greater_than_20_percent_body_area
            if code == '2114':
                change_in_daly_weights -= self.daly_wt_burns_greater_than_20_percent_body_area
            if code == '3113':
                change_in_daly_weights -= self.daly_wt_burns_less_than_20_percent_body_area_with_treatment
            if code == '4113':
                change_in_daly_weights -= self.daly_wt_burns_less_than_20_percent_body_area_with_treatment
            if code == '5113':
                change_in_daly_weights -= self.daly_wt_burns_less_than_20_percent_body_area_with_treatment
            if code == '7113':
                change_in_daly_weights -= self.daly_wt_burns_less_than_20_percent_body_area_with_treatment
            if code == '8113':
                change_in_daly_weights -= self.daly_wt_burns_less_than_20_percent_body_area_with_treatment
            # ============================== Codes 'treated' with major surgery ========================================
            if code == '813a':
                change_in_daly_weights -= self.daly_wt_hip_fracture_long_term_with_treatment
            if code == '813b':
                change_in_daly_weights -= self.daly_wt_pelvis_fracture_long_term
            if code == '813c':
                change_in_daly_weights -= self.daly_wt_femur_fracture_short_term
            if code == '133a':
                change_in_daly_weights -= self.daly_wt_subarachnoid_hematoma
            if code == '133b':
                change_in_daly_weights -= self.daly_wt_brain_contusion
            if code == '133c':
                change_in_daly_weights -= self.daly_wt_intraventricular_haemorrhage
            if code == '133d':
                change_in_daly_weights -= self.daly_wt_subgaleal_hematoma
            if code == '134a':
                change_in_daly_weights -= self.daly_wt_epidural_hematoma
            if code == '134b':
                change_in_daly_weights -= self.daly_wt_subdural_hematoma
            if code == '135':
                change_in_daly_weights -= self.daly_wt_diffuse_axonal_injury
            if code == '342' or code == '343' or code == '361' or code == '363':
                change_in_daly_weights -= self.daly_wt_neck_internal_bleeding
            if code == '414':
                change_in_daly_weights -= self.daly_wt_flail_chest
            if code == '441':
                change_in_daly_weights -= self.daly_wt_closed_pneumothorax
            if code == '443':
                change_in_daly_weights -= self.daly_wt_open_pneumothorax
            if code == '453a':
                change_in_daly_weights -= self.daly_wt_diaphragm_rupture
            if code == '453b':
                change_in_daly_weights -= self.daly_wt_lung_contusion
            if code == '463':
                change_in_daly_weights -= self.daly_wt_hemothorax
            # ----------------------------- Codes treated with open fracture treatment --------------------------------
            if code == '813bo':
                change_in_daly_weights -= self.daly_wt_pelvis_fracture_long_term + \
                                          self.daly_wt_facial_soft_tissue_injury
            if code == '813co':
                change_in_daly_weights -= self.daly_wt_femur_fracture_short_term + \
                                          self.daly_wt_facial_soft_tissue_injury
            if code == '813do':
                change_in_daly_weights -= self.daly_wt_foot_fracture_short_term_with_without_treatment + \
                                          self.daly_wt_facial_soft_tissue_injury
            if code == '813eo':
                change_in_daly_weights -= self.daly_wt_patella_tibia_fibula_fracture_without_treatment + \
                                          self.daly_wt_facial_soft_tissue_injury

            # update the total values of the daly weights
            df.loc[person_id, 'rt_debugging_DALY_wt'] += change_in_daly_weights
        # if the person's true total for daly weights is greater than one, report rt_disability as one, if not
        # report the true disability burden.
        if df.loc[person_id, 'rt_debugging_DALY_wt'] > 1:
            df.loc[person_id, 'rt_disability'] = 1
        else:
            df.loc[person_id, 'rt_disability'] = df.loc[person_id, 'rt_debugging_DALY_wt']
        # round off any potential floating point errors
        df.loc[person_id, 'rt_debugging_DALY_wt'] = np.round(df.loc[person_id, 'rt_debugging_DALY_wt'], 4)
        # if the reported daly weight is below zero add make the model report the true (and always positive) daly weight
        if df.loc[person_id, 'rt_disability'] < 0:
            df.loc[person_id, 'rt_disability'] = df.loc[person_id, 'rt_debugging_DALY_wt']
        # Make sure the true disability burden is greater or equal to zero
        assert df.loc[person_id, 'rt_debugging_DALY_wt'] >= 0, (person_injuries.values,
                                                                df.loc[person_id, 'rt_debugging_DALY_wt'])
        # the reported disability should satisfy 0<=disability<=1, check that they do
        assert df.loc[person_id, 'rt_disability'] >= 0, 'Negative disability burden'
        assert df.loc[person_id, 'rt_disability'] <= 1, 'Too large disability burden'
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
        assert df.loc[person_id, 'rt_med_int']
        # Check they haven't died due to rti
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'
        # Check they have an appropriate injury code to swap

        swapping_codes = ['712b', '812', '3113', '4113', '5113', '7113', '8113', '813a', '813b', 'P673a',
                          'P673b', 'P674a', 'P674b', 'P675a', 'P675b', 'P676', 'P782b', 'P783', 'P883', 'P884',
                          '813bo', '813co', '813do', '813eo']
        relevant_codes = np.intersect1d(codes, swapping_codes)
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        # check this person is injured, search they have an injury code that is swappable
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries, relevant_codes)
        assert counts > 0, 'This person has asked to swap an injury code, but it is not swap-able'
        # Iterate over the relevant codes
        for code in relevant_codes:
            # swap the relevant code's daly weight, from the daly weight associated with the injury without treatment
            # and the daly weight for the disability with treatment.
            # keep track of the changes to the daly weights
            change_in_daly_weights = 0
            if code == '712b':
                change_in_daly_weights += - self.daly_wt_hand_wrist_fracture_without_treatment + \
                                          self.daly_wt_hand_wrist_fracture_with_treatment
            if code == '812':
                change_in_daly_weights += - self.daly_wt_patella_tibia_fibula_fracture_without_treatment + \
                                          self.daly_wt_patella_tibia_fibula_fracture_with_treatment
            if code == '3113':
                change_in_daly_weights += \
                    - self.daly_wt_burns_less_than_20_percent_body_area_without_treatment \
                    + self.daly_wt_burns_less_than_20_percent_body_area_with_treatment

            if code == '4113':
                change_in_daly_weights += \
                    - self.daly_wt_burns_less_than_20_percent_body_area_without_treatment \
                    + self.daly_wt_burns_less_than_20_percent_body_area_with_treatment
            if code == '5113':
                change_in_daly_weights += \
                    - self.daly_wt_burns_less_than_20_percent_body_area_without_treatment \
                    + self.daly_wt_burns_less_than_20_percent_body_area_with_treatment
            if code == '7113':
                change_in_daly_weights += \
                    - self.daly_wt_burns_less_than_20_percent_body_area_without_treatment \
                    + self.daly_wt_burns_less_than_20_percent_body_area_with_treatment
            if code == '8113':
                change_in_daly_weights += \
                    - self.daly_wt_burns_less_than_20_percent_body_area_without_treatment \
                    + self.daly_wt_burns_less_than_20_percent_body_area_with_treatment
            if code == '813a':
                change_in_daly_weights += - self.daly_wt_hip_fracture_short_term_with_without_treatment + \
                                          self.daly_wt_hip_fracture_long_term_with_treatment
            if code == '813b':
                change_in_daly_weights += - self.daly_wt_pelvis_fracture_short_term + \
                                          self.daly_wt_pelvis_fracture_long_term
            if code == '813bo':
                change_in_daly_weights += - self.daly_wt_pelvis_fracture_short_term + \
                                          self.daly_wt_pelvis_fracture_long_term
            if code == 'P673a':
                change_in_daly_weights += - self.daly_wt_spinal_cord_lesion_neck_without_treatment + \
                                          self.daly_wt_spinal_cord_lesion_neck_with_treatment
            if code == 'P673b':
                change_in_daly_weights += - self.daly_wt_spinal_cord_lesion_below_neck_without_treatment + \
                                          self.daly_wt_spinal_cord_lesion_below_neck_with_treatment
            if code == 'P674a':
                change_in_daly_weights += - self.daly_wt_spinal_cord_lesion_neck_without_treatment + \
                                          self.daly_wt_spinal_cord_lesion_neck_with_treatment
            if code == 'P674b':
                change_in_daly_weights += - self.daly_wt_spinal_cord_lesion_below_neck_without_treatment + \
                                          self.daly_wt_spinal_cord_lesion_below_neck_with_treatment
            if code == 'P675a':
                change_in_daly_weights += - self.daly_wt_spinal_cord_lesion_neck_without_treatment + \
                                          self.daly_wt_spinal_cord_lesion_neck_with_treatment
            if code == 'P675b':
                change_in_daly_weights += - self.daly_wt_spinal_cord_lesion_below_neck_without_treatment + \
                                          self.daly_wt_spinal_cord_lesion_below_neck_with_treatment
            if code == 'P676':
                change_in_daly_weights += - self.daly_wt_spinal_cord_lesion_neck_without_treatment + \
                                          self.daly_wt_spinal_cord_lesion_neck_with_treatment
            if code == 'P782b':
                change_in_daly_weights += - self.daly_wt_unilateral_arm_amputation_without_treatment + \
                                          self.daly_wt_unilateral_arm_amputation_with_treatment
            if code == 'P783':
                change_in_daly_weights += - self.daly_wt_bilateral_arm_amputation_without_treatment + \
                                          self.daly_wt_bilateral_arm_amputation_with_treatment
            if code == 'P883':
                change_in_daly_weights += - self.daly_wt_unilateral_lower_limb_amputation_without_treatment \
                                          + self.daly_wt_unilateral_lower_limb_amputation_with_treatment
            if code == 'P884':
                change_in_daly_weights += - self.daly_wt_bilateral_lower_limb_amputation_without_treatment \
                                          + self.daly_wt_bilateral_lower_limb_amputation_with_treatment
            # update the disability burdens
            df.loc[person_id, 'rt_debugging_DALY_wt'] += change_in_daly_weights
        # Check that the person's true disability burden is positive
        assert np.round(df.loc[person_id, 'rt_debugging_DALY_wt'], 4) >= 0, (person_injuries.values,
                                                                             df.loc[person_id, 'rt_debugging_DALY_wt'])
        # catch rounding point errors where the disability weights should be zero but aren't
        if df.loc[person_id, 'rt_disability'] < 0:
            df.loc[person_id, 'rt_disability'] = np.round(df.loc[person_id, 'rt_debugging_DALY_wt'], 4)
        # Catch cases where the disability burden is greater than one in reality but needs to be
        # capped at one, if not report the true disability burden
        if df.loc[person_id, 'rt_debugging_DALY_wt'] > 1:
            df.loc[person_id, 'rt_disability'] = 1
        else:
            df.loc[person_id, 'rt_disability'] = df.loc[person_id, 'rt_debugging_DALY_wt']
        # Check the daly weights fall within the accepted bounds
        assert df.loc[person_id, 'rt_disability'] >= 0, 'Negative disability burden'
        assert df.loc[person_id, 'rt_disability'] <= 1, 'Too large disability burden'

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
        if df.loc[person_id, 'cause_of_death'] != '':
            pass
        # Check that those sent here have been injured and have not died due to rti
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'
        # Load the parameters needed to determine the length of stay
        mean_los_ISS_less_than_4 = p['mean_los_ISS_less_than_4']
        sd_los_ISS_less_than_4 = p['sd_los_ISS_less_than_4']
        mean_los_ISS_4_to_8 = p['mean_los_ISS_4_to_8']
        sd_los_ISS_4_to_8 = p['sd_los_ISS_4_to_8']
        mean_los_ISS_9_to_15 = p['mean_los_ISS_9_to_15']
        sd_los_ISS_9_to_15 = p['sd_los_ISS_9_to_15']
        mean_los_ISS_16_to_24 = p['mean_los_ISS_16_to_24']
        sd_los_ISS_16_to_24 = p['sd_los_ISS_16_to_24']
        mean_los_ISS_more_than_25 = p['mean_los_ISS_more_than_25']
        sd_los_ISS_more_that_25 = p['sd_los_ISS_more_that_25']
        days_until_treatment_end = 0  # default value to be changed
        # Create the length of stays required for each ISS score boundaries and check that they are >=0
        if df.iloc[person_id]['rt_ISS_score'] < 4:
            inpatient_days_ISS_less_than_4 = int(self.rng.normal(mean_los_ISS_less_than_4,
                                                                 sd_los_ISS_less_than_4, 1))
            days_until_treatment_end = inpatient_days_ISS_less_than_4
        if 4 <= df.iloc[person_id]['rt_ISS_score'] < 9:
            inpatient_days_ISS_4_to_8 = int(self.rng.normal(mean_los_ISS_4_to_8,
                                                            sd_los_ISS_4_to_8, 1))
            days_until_treatment_end = inpatient_days_ISS_4_to_8
        if 9 <= df.iloc[person_id]['rt_ISS_score'] < 16:
            inpatient_days_ISS_9_to_15 = int(self.rng.normal(mean_los_ISS_9_to_15,
                                                             sd_los_ISS_9_to_15, 1))
            days_until_treatment_end = inpatient_days_ISS_9_to_15
        if 16 <= df.iloc[person_id]['rt_ISS_score'] < 25:
            inpatient_days_ISS_16_to_24 = int(self.rng.normal(mean_los_ISS_16_to_24,
                                                              sd_los_ISS_16_to_24, 1))
            days_until_treatment_end = inpatient_days_ISS_16_to_24
        if 25 <= df.iloc[person_id]['rt_ISS_score']:
            inpatient_days_ISS_more_than_25 = int(self.rng.normal(mean_los_ISS_more_than_25,
                                                                  sd_los_ISS_more_that_25, 1))
            days_until_treatment_end = inpatient_days_ISS_more_than_25
        if days_until_treatment_end < 0:
            days_until_treatment_end = 0
        # Return the LOS
        return days_until_treatment_end

    @staticmethod
    def rti_find_and_count_injuries(dataframe, tloinjcodes):
        """
        A function that searches a user given dataframe for a list of injuries (tloinjcodes). If the injury code is
        found in the dataframe, this function returns the index for who has the injury/injuries and the number of
        injuries found. This function works much faster if the dataframe is smaller, hence why the searched dataframe
        is a parameter in the function.

        :param dataframe: The dataframe to search for the tlo injury codes in
        :param tloinjcodes: The injury codes to search for in the data frame
        :return: the df index of who has the injuries and how many injuries in the search were found.
        """
        # create empty index and outputs to append to
        index = pd.Index([])
        # set the number of found injuries to zero by default
        counts = 0
        # reformat the person's injury dataframes to a list
        peoples_injuries = [item for sublist in dataframe.values.tolist() for item in sublist]
        # get the relevant codes to search over
        relevant_codes = np.intersect1d(peoples_injuries, tloinjcodes)
        for code in relevant_codes:
            for col in dataframe.columns:
                # Find where a searched for injury code is in the columns, store the matches in counts
                counts += len(dataframe[dataframe[col] == code])
                if len(dataframe[dataframe[col] == code]) > 0:
                    # If you find a matching code, update the index to include the matching person
                    inj = dataframe.apply(lambda row: row.astype(str).str.contains(code).any(0), axis=1)
                    injidx = inj.index[inj]
                    index = index.union(injidx)
        # return the idx of people with the corresponding injuries and the number of corresponding injuries found
        return index, counts

    def rti_treated_injuries(self, person_id, tloinjcodes):
        """
        A function that takes a person with treated injuries and removes the injury code from the properties rt_injury_1
        to rt_injury_8

        The properties that this function alters are rt_injury_1 through rt_injury_8 and the symptoms properties

        :param person_id: The person who needs an injury code removed
        :param tloinjcodes: the injury code(s) to be removed
        :return: n/a
        """
        df = self.sim.population.props
        # Isolate the relevant injury information
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        permanent_injuries = ['P133', 'P133a', 'P133b', 'P133c', 'P133d', 'P134', 'P134a', 'P134b', 'P135', 'P673',
                              'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675', 'P675a', 'P675b', 'P676', 'P782a',
                              'P782b', 'P782c', 'P783', 'P882', 'P883', 'P884']
        person_injuries = df.loc[[person_id], cols]
        # Iterate over the codes
        for code in tloinjcodes:
            if code in permanent_injuries:
                # checks if the injury is permanent, if so the injury code is not removed.
                pass
            else:
                # Find which columns have treated injuries
                injury_cols = person_injuries.columns[(person_injuries.values == code).any(0)].tolist()
                # Reset the treated injury code to "none"
                df.loc[person_id, injury_cols] = "none"
                # Reset symptoms so that after being treated for an injury the person won't interact with the
                # healthsystem again.
                if df.loc[person_id, 'sy_injury'] != 0:
                    self.sim.modules['SymptomManager'].change_symptom(
                        person_id=person_id,
                        disease_module=self.sim.modules['RTI'],
                        add_or_remove='-',
                        symptom_string='injury',
                    )
                if df.loc[person_id, 'sy_severe_trauma'] != 0:
                    self.sim.modules['SymptomManager'].change_symptom(
                        person_id=person_id,
                        disease_module=self.sim.modules['RTI'],
                        add_or_remove='-',
                        symptom_string='severe_trauma',
                    )

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

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        logger.debug(
            'This is RTI, being alerted about a health system interaction person %d for: %s',
            person_id,
            treatment_id,
        )

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.
        logger.debug('This is RTI reporting my daly values')
        df = self.sim.population.props
        disability_series_for_alive_persons = df.loc[df.is_alive, "rt_disability"]
        return disability_series_for_alive_persons

    def assign_injuries(self, number):
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
        # Import the distribution of injured body regions from the VIBES study
        number_of_injured_body_regions_distribution = p['number_of_injured_body_regions_distribution']
        # Create empty lists to store information on the person's injuries
        # predicted injury location
        predinjlocs = []
        # predicted injury severity
        predinjsev = []
        # predicted injury category
        predinjcat = []
        # predicted injury ISS score
        predinjiss = []
        # whether the prediction injuries are classed as polytrauma
        predpolytrauma = []
        # whether this predicted injury requires a greater level of detail that can't be determined by location
        # category and severity alone
        predinjdetail = []
        # Create empty lists which will be used to combine the injury location, category, severity and detail
        # information
        injlocstring = []
        injcatstring = []
        injaisstring = []
        injdetailstr = []
        # create empty lists to store the qualitative description of injury severity and the number of injuries
        # each person has
        severity_category = []
        number_of_injuries = []
        # ============================= Begin assigning injuries to people =====================================
        # Iterate over the total number of injured people
        for n in range(0, number):
            # Get the distribution of body regions which can be injured for each iteration.
            injlocdist = p['injury_location_distribution']
            # Convert the parameter to a numpy array
            injlocdist = np.array(injlocdist)
            # Generate a random number which will decide how many injuries the person will have,
            ninj = self.rng.choice(number_of_injured_body_regions_distribution[0],
                                   p=number_of_injured_body_regions_distribution[1])
            # store the number of injuries this person recieves
            number_of_injuries.append(ninj)
            # Create an empty vector which will store the injury locations (numerically coded using the
            # abbreviated injury scale coding system, where 1 corresponds to head, 2 to face, 3 to neck, 4 to
            # thorax, 5 to abdomen, 6 to spine, 7 to upper extremity and 8 to lower extremity
            allinjlocs = []
            # Create an empty vector to store the type of injury
            injcat = []
            # Create an empty vector which will store the severity of the injuries
            injais = []
            # Create an empty vector to store the exact nature of the injury
            injdetail = []
            # generate the locations of the injuries for this person
            injurylocation = self.rng.choice(injlocdist[0], ninj, p=injlocdist[1], replace=False)
            # iterate over the chosen injury locations to determine the exact injuries that this person will have
            for injlocs in injurylocation:
                # Store this person's injury location
                allinjlocs.append(int(injlocs))
                # create a random variable which will determine the category of the injury
                cat = self.rng.uniform(0, 1)
                # create a random variable which will determine the severity of the injury
                severity = self.rng.uniform(0, 1)
                # create an empty string to be used for adding detail to the injury if more than one injury maps to the
                # code system
                detail = 'none'
                # create a random variable that will decide if certain fractures are open or not
                open_frac = self.rng.uniform(0, 1)

                # In injury categories I will use the following mapping:
                # Fracture - 1
                # Dislocation - 2
                # Traumatic brain injury - 3
                # Soft tissue injury - 4
                # Internal organ injury - 5
                # Internal bleeding - 6
                # Spinal cord injury - 7
                # Amputation - 8
                # Eye injury - 9
                # Cuts etc (minor wounds) - 10
                # Burns - 11

                # if the injury is located in the head (injloc 1) determine what the injury is
                if injlocs == 1:
                    # todo: redo the assign injury function with the self.rng.choice function
                    # Decide what the injury to the head is going to be:
                    # determine if injury is a skin wound
                    if cat <= self.head_prob_skin_wound:
                        # determine if it is a laceration else it is a burn
                        if severity <= self.head_prob_skin_wound_open:
                            # Open wound so store corresponding category and ais score
                            injcat.append(int(10))
                            injais.append(1)
                        else:
                            # Burn so store corresponding category and ais score
                            injcat.append(int(11))
                            injais.append(4)
                    # determine if injury is a skull fracture
                    elif self.head_prob_skin_wound < cat <= self.head_prob_skin_wound + self.head_prob_fracture:
                        # Injury is a skull fracture so update the injury category
                        injcat.append(int(1))
                        # determine how severe the skull fracture will be, either having an AIS score of 2 or 3,
                        # store the severity in injais
                        if severity <= self.head_prob_fracture_unspecified:
                            injais.append(2)
                        else:
                            injais.append(3)
                    # Determine if this injury is a traumatic brain injury
                    elif self.head_prob_skin_wound + self.head_prob_fracture < cat:
                        # The injury is a traumatic brain injury so store the injury category
                        injcat.append(int(3))
                        # Decide how severe the traumatic brain injury will be
                        if severity <= self.head_prob_TBI_AIS3:
                            # Mild TBI, store the severity score in injais
                            injais.append(3)
                            # multiple TBIs have an AIS score of 3, but have different daly weights, determine
                            # the exact TBI injury here and distinguish between them using the injury detail
                            # list.
                            # Choose the exact TBI and store the information in variable detail
                            probabilities = p['daly_dist_code_133']
                            detail_add_on = ['a', 'b', 'c', 'd']
                            detail = self.rng.choice(detail_add_on, p=probabilities)
                        elif self.head_prob_TBI_AIS3 < severity <= self.head_prob_TBI_AIS3 + self.head_prob_TBI_AIS4:
                            # Moderate TBI, store the injury severity in injais
                            injais.append(4)
                            # multiple TBIs have an AIS score of 4, but have different daly weights, determine
                            # the exact TBI injury here and distinguish between them using the injury detail
                            # list.
                            # Choose the exact TBI and store the information in variable detail
                            probabilities = p['daly_dist_code_134']
                            detail_add_on = ['a', 'b']
                            detail = self.rng.choice(detail_add_on, p=probabilities)
                        elif self.head_prob_TBI_AIS3 + self.head_prob_TBI_AIS4 < severity:
                            # Severe TBI, store the severity score in injais
                            injais.append(5)
                # if the injury is located in the face (injloc 2) determine what the injury is
                if injlocs == 2:
                    # Decide what the injury to the face will be
                    # determine if it is a skin wound
                    # label probability boundaries used for assigning facial injuries
                    boundary_1 = self.face_prob_skin_wound
                    boundary_2 = self.face_prob_skin_wound + self.face_prob_fracture
                    boundary_3 = self.face_prob_skin_wound + self.face_prob_fracture + self.face_prob_soft_tissue_injury
                    if cat <= self.face_prob_skin_wound:
                        # decide whether it will be a laceration or a burn
                        if severity <= boundary_1:
                            # Open wound, store the category and severity information
                            injcat.append(int(10))
                            injais.append(1)
                        else:
                            # Burn, store the category and severity information
                            injcat.append(int(11))
                            injais.append(4)
                    # Ask if it is a facial fracture
                    elif boundary_1 < cat <= boundary_2:
                        # Facial fracture, update the injury category list
                        injcat.append(int(1))
                        # decide how severe the injury will be
                        if severity <= self.face_prob_fracture_AIS1:
                            # Nasal and unspecified fractures of the face, AIS score of 1 store severity information
                            injais.append(1)
                        else:
                            # Mandible and Zygomatic fractures, AIS score of 2 store severity information
                            injais.append(2)
                    # Ask if it will be a soft tissue injury
                    elif boundary_2 < cat < boundary_3:
                        # soft tissue injury, store the injury category and injury information
                        injcat.append(int(4))
                        injais.append(1)
                    # If none of the above the injury is an eye injury
                    elif boundary_3 < cat:
                        # eye injury, store the injury category and injury information
                        injcat.append(int(9))
                        injais.append(1)
                # Decide what the injury to the neck will be
                if injlocs == 3:
                    # ask if the injury is a skin wound
                    # label probability boundaries used to assign neck injuries
                    boundary_1 = self.neck_prob_skin_wound
                    boundary_2 = self.neck_prob_skin_wound + self.neck_prob_soft_tissue_injury
                    boundary_3 = boundary_2 + self.neck_prob_internal_bleeding
                    if cat <= self.neck_prob_skin_wound:
                        # ask if it is a laceration, else it will be a burn
                        if severity <= boundary_1:
                            # Open wound, store the injury category and injury information
                            injcat.append(int(10))
                            injais.append(1)
                        else:
                            # Burn, store the injury category and injury information
                            injcat.append(int(11))
                            injais.append(3)
                    # determine if the injury is a soft tissue injury
                    elif boundary_1 < cat <= boundary_2:
                        # Soft tissue injuries of the neck, store the injury category information
                        injcat.append(int(4))
                        # decide how severe the injury is
                        if severity <= self.neck_prob_soft_tissue_injury_AIS2:
                            # Vertebral artery laceration, AIS score of 2, store information in injais
                            injais.append(2)
                        else:
                            # Pharynx contusion, AIS score of 2, store information in injais
                            injais.append(3)
                    # determine if the injury is internal bleeding
                    elif boundary_2 < cat <= boundary_3:
                        # Internal bleeding, so store injury category in injcat
                        injcat.append(int(6))
                        # Decide how severe the injury will be
                        if severity <= self.neck_prob_internal_bleeding_AIS1:
                            # Sternomastoid m. hemorrhage,
                            # Hemorrhage, supraclavicular triangle
                            # Hemorrhage, posterior triangle
                            # Anterior vertebral vessel hemorrhage
                            # Neck muscle hemorrhage
                            # all have AIS score of 1 so store severity information in injais
                            injais.append(1)
                        else:
                            # Hematoma in carotid sheath
                            # Carotid sheath hemorrhage
                            # have AIS score of 3 so store severity information in injais
                            injais.append(3)
                    # Determine if the injury is a dislocation
                    elif boundary_3 < cat:
                        # Dislocation, store the category information in injcat
                        injcat.append(int(2))
                        # Decide how severe the injury will be
                        if severity <= self.neck_prob_dislocation_AIS3:
                            # Atlanto-axial subluxation has an AIS score of 3, store injury severity in injais
                            injais.append(3)
                        else:
                            # Atlanto-occipital subluxation has an AIS score of 2, store injury severity in injais
                            injais.append(2)
                # If the injury is a thorax injury, determine what the injury is
                if injlocs == 4:
                    # Determine if the injury is a skin wound
                    # label probability boundaries used to assign neck injuries
                    boundary_1 = self.thorax_prob_skin_wound
                    boundary_2 = self.thorax_prob_skin_wound + self.thorax_prob_internal_bleeding
                    boundary_3 = boundary_2 + self.thorax_prob_internal_organ_injury
                    boundary_4 = boundary_3 + self.thorax_prob_fracture
                    boundary_5 = boundary_4 + self.thorax_prob_soft_tissue_injury
                    if cat <= self.thorax_prob_skin_wound:
                        # Decide if the injury is a laceration or a burn
                        if severity <= boundary_1:
                            # Open wound, so update the injury category and severity information
                            injcat.append(int(10))
                            injais.append(1)
                        else:
                            # Burn, so update the injury category and severity information
                            injcat.append(int(11))
                            injais.append(3)
                    # Decide if the injury is internal bleeding
                    elif boundary_1 < cat <= boundary_2:
                        # Internal Bleeding, so update the injury category information
                        injcat.append(int(6))
                        # Decide how severe the injury will be
                        if severity <= self.thorax_prob_internal_bleeding_AIS1:
                            # Chest wall bruises/haematoma has an AIS score of 1, update injais
                            injais.append(1)
                        else:
                            # Haemothorax has an AIS score of 3, update injais
                            injais.append(3)
                    # Decide if the injury is an internal organ injury
                    elif boundary_2 < cat <= boundary_3:
                        # Internal organ injury, so update the injury category information
                        injcat.append(int(5))
                        # Lung contusion and Diaphragm rupture both have an AIS score of 3, update the injury
                        # severity information
                        injais.append(3)
                        # As both lung contusion and diaphragm have the same AIS score, but different daly weights,
                        # distiguish between the two with the detail variable
                        probabilities = p['daly_dist_code_453']
                        detail_add_on = ['a', 'b']
                        # store the specific details of the injury in the variable detail
                        detail = self.rng.choice(detail_add_on, p=probabilities)
                    # Determine if the injury is a fracture/flail chest
                    elif boundary_3 < cat <= boundary_4:
                        # Fractures to ribs and flail chest, so update the injury category information
                        injcat.append(int(1))
                        # Decide how severe the injury is
                        if severity <= self.thorax_prob_fracture_ribs:
                            # fracture to rib(s) have an AIS score of 2, store this in injais
                            injais.append(2)
                        else:
                            # flail chest has an AIS score of 4, store this in injais
                            injais.append(4)
                    # Determine if the injury is a soft tissue injury
                    elif boundary_4 < cat <= boundary_5:
                        # Soft tissue injury, update the injury catregory information
                        injcat.append(int(4))
                        # Decide how severe the injury is
                        # create boundaries used for assigning injury severity
                        sev_boundary_1 = self.thorax_prob_soft_tissue_injury_AIS1
                        sev_boundary_2 = sev_boundary_1 + self.thorax_prob_soft_tissue_injury_AIS2
                        if severity <= sev_boundary_1:
                            # Chest wall lacerations/avulsions have an AIS score of 1, store this in injais
                            injais.append(1)
                        elif sev_boundary_1 < severity <= sev_boundary_2:
                            # surgical emphysema has an AIS score of 2, store this in injais
                            injais.append(2)
                        else:
                            # Open/closed pneumothorax has an AIS score of 3, store this in injais
                            injais.append(3)
                # If the injury is to the abdomen, determine what the injury will be
                if injlocs == 5:
                    # Decide if it will be a skin wound, otherwise will be an internal organ injury
                    if cat <= self.abdomen_prob_skin_wound:
                        # Decide if the skin wound is a laceration or a burn
                        if severity <= self.abdomen_prob_skin_wound_open:
                            # Open wound, store the injury category and severity information
                            injcat.append(int(10))
                            injais.append(1)
                        else:
                            # Burn, store the injury category and severity information
                            injcat.append(int(11))
                            injais.append(3)
                    else:
                        # Internal organ injuries, store the injury category information
                        injcat.append(int(5))
                        # create boundaries used for assigning injury severity
                        sev_boundary_1 = self.abdomen_prob_internal_organ_injury_AIS2
                        sev_boundary_2 = sev_boundary_1 + self.abdomen_prob_internal_organ_injury_AIS3
                        # Decide how severe the injury is, all abdominal injuries share the same daly weight so
                        # there is no need to specify further details beyond severity, category and location
                        if severity <= self.abdomen_prob_internal_organ_injury_AIS2:
                            # Intestines, Stomach and colon injury have an AIS score of 2, store this in injais
                            injais.append(2)
                        elif sev_boundary_1 < severity <= sev_boundary_2:
                            # Spleen, bladder, liver, urethra and diaphragm injury have an AIS score of 3, store this
                            # in injais
                            injais.append(3)
                        else:
                            # Kidney injury has an AIS score of 2, store this in injais
                            injais.append(4)
                # If the injury is to the spine, determine what the injury will be
                if injlocs == 6:
                    # Determine if the injury is a vertebrae fracture
                    if cat <= self.spine_prob_fracture:
                        # Fracture to vertebrae, store the injury category and AIS score
                        injcat.append(int(1))
                        injais.append(2)
                    # Ask if it is a spinal cord lesion
                    else:
                        # Spinal cord injury, store the injury category
                        injcat.append(int(7))
                        # create probability boundaries to decide what the injury severity of the spinal cord
                        # injury will be
                        base1 = \
                            self.spine_prob_spinal_cord_lesion_neck_level * \
                            self.spine_prob_spinal_cord_lesion_neck_level_AIS3 + \
                            self.spine_prob_spinal_cord_lesion_below_neck_level * \
                            self.spine_prob_spinal_cord_lesion_below_neck_level_AIS3
                        base2 = \
                            self.spine_prob_spinal_cord_lesion_neck_level * \
                            self.spine_prob_spinal_cord_lesion_neck_level_AIS4 + \
                            self.spine_prob_spinal_cord_lesion_below_neck_level * \
                            self.spine_prob_spinal_cord_lesion_below_neck_level_AIS4
                        base3 = \
                            self.spine_prob_spinal_cord_lesion_neck_level * \
                            self.spine_prob_spinal_cord_lesion_neck_level_AIS5 + \
                            self.spine_prob_spinal_cord_lesion_below_neck_level * \
                            self.spine_prob_spinal_cord_lesion_below_neck_level_AIS5
                        # Decide how severe the injury is
                        if severity <= base1:
                            # This spinal cord has been determined to have an AIS score of 3, store the score in
                            # injais
                            injais.append(3)
                            # After determining how severe the injury to the spine is, we need to determine where
                            # on the spine the injury is located as spinal cord injuries above the neck carry a
                            # greater disability burden
                            probabilities = p['daly_dist_code_673']
                            detail_add_on = ['a', 'b']
                            # Choose where on the spine the spinal cord laceration is located and store the specific
                            # injury details in the variable detail
                            detail = self.rng.choice(detail_add_on, p=probabilities)
                        elif base1 < cat <= base1 + base2:
                            # This spinal cord has been determined to have an AIS score of 4, store the score in
                            # injais
                            injais.append(4)
                            # After determining how severe the injury to the spine is, we need to determine where
                            # on the spine the injury is located as spinal cord injuries above the neck carry a
                            # greater disability burden
                            probabilities = p['daly_dist_codes_674_675']
                            detail_add_on = ['a', 'b']
                            # Choose where on the spine the spinal cord laceration is located and store the specific
                            # injury details in the variable detail
                            detail = self.rng.choice(detail_add_on, p=probabilities)
                        elif base1 + base2 < cat <= base1 + base2 + base3:
                            # This spinal cord has been determined to have an AIS score of 5, store the score in
                            # injais
                            injais.append(5)
                            # After determining how severe the injury to the spine is, we need to determine where
                            # on the spine the injury is located as spinal cord injuries above the neck carry a
                            # greater disability burden
                            probabilities = p['daly_dist_codes_674_675']
                            detail_add_on = ['a', 'b']
                            # Choose where on the spine the spinal cord laceration is located and store the specific
                            # injury details in the variable detail
                            detail = self.rng.choice(detail_add_on, p=probabilities)
                        else:
                            # This spinal cord laceration has an AIS score of 6. Store the informaiton in injais
                            injais.append(6)
                            # All spinal cord lacerations with an AIS score of 6 are at the neck level, so there is
                            # no need to specify further detail.
                # If the injury is to the upper extermities, determine what the injury will be
                if injlocs == 7:
                    # Decide if the injury will be a skin wound
                    # create probability boundaries used to assign injuries
                    boundary_1 = self.upper_ex_prob_skin_wound
                    boundary_2 = boundary_1 + self.upper_ex_prob_fracture
                    boundary_3 = boundary_2 + self.upper_ex_prob_dislocation
                    if cat <= self.upper_ex_prob_skin_wound:
                        # Decide if the injury will be a laceration or a burn
                        if severity <= boundary_1:
                            # Open wound, update the injury category and severity information
                            injcat.append(int(10))
                            injais.append(1)
                        else:
                            # Burn, update the injury category and severity information
                            injcat.append(int(11))
                            injais.append(3)
                    # Determine if the injury is a fracture
                    elif boundary_1 < cat <= boundary_2:
                        # Fracture to arm, update the injury category and severity information
                        injcat.append(int(1))
                        injais.append(2)
                        # Multiple arm fractures have an AIS score of 2, but have different daly weights associated
                        # with them, need to specify further which daly weight to assign to this injury
                        probabilities = p['daly_dist_code_712']
                        detail_add_on = ['a', 'b', 'c']
                        # Store the specific injury details in the variable detail
                        detail = self.rng.choice(detail_add_on, p=probabilities)
                    # Determine if the injury is a dislocation
                    elif boundary_2 < cat <= boundary_3:
                        # Dislocation to arm, update the injury category and severity information
                        injcat.append(int(2))
                        injais.append(2)
                    # Determine if the injury is an amputation
                    elif boundary_3 < cat:
                        # Amputation in upper limb, store the injury category information
                        injcat.append(int(8))
                        # Determine how severe the injury will be
                        if severity <= self.upper_ex_prob_amputation_AIS2:
                            # Amputation to finger/thumb/unilateral arm
                            injais.append(2)
                            # amputated thumbs/fingers/ arms have the same AIS score (apparently), but have different
                            # disability burdens, need to specify what the exact injury is
                            probabilities = p['daly_dist_code_782']
                            detail_add_on = ['a', 'b', 'c']
                            # Store the specific injury details in the variable detail
                            detail = self.rng.choice(detail_add_on, p=probabilities)
                        else:
                            # Amputation, arm, bilateral, store the injury severity in injais
                            injais.append(3)
                # If the injury is to the lower extermities, determine what the injury will be
                if injlocs == 8:
                    # Determine if the injury is a skin wound
                    # create probability boundaries used to assign lower extremity injuries
                    boundary_1 = self.lower_ex_prob_skin_wound
                    boundary_2 = boundary_1 + self.lower_ex_prob_fracture
                    boundary_3 = boundary_2 + self.lower_ex_prob_dislocation
                    if cat <= boundary_1:
                        # decide if the injury is a laceration or a burn
                        if severity <= self.lower_ex_prob_skin_wound_open:
                            # Open wound, store the relevant category and severity information
                            injcat.append(int(10))
                            injais.append(1)
                        else:
                            # Burn, store the relevant category and severity information
                            injcat.append(int(11))
                            injais.append(3)
                    # Decide if the injury is a fracture
                    elif boundary_1 < cat <= boundary_2:
                        # Fracture, so update the injury category information
                        injcat.append(int(1))
                        # Decide how severe the injury will be
                        # create probability boundaries for assigning injury severity
                        sev_boundary_1 = self.lower_ex_prob_fracture_AIS1
                        sev_boundary_2 = sev_boundary_1 + self.lower_ex_prob_fracture_AIS2
                        if severity < sev_boundary_1:
                            # Foot fracture
                            # determine if the foot fracture is open
                            prob_foot_frac_open = p['prob_foot_fracture_open']
                            if open_frac < prob_foot_frac_open:
                                # update the injury details to represent the fact that this foot fracture is open
                                detail = 'do'
                                # store the severity information in injais
                                injais.append(3)
                            else:
                                # Foot fracture is not open so just need to store the injury information in injais
                                injais.append(1)
                        elif sev_boundary_1 < severity <= sev_boundary_2:
                            # Lower leg fracture
                            # determine is the lower leg fracture is open
                            prob_tib_fib_frac_open = p['prob_patella_tibia_fibula_ankle_fracture_open']
                            if open_frac < prob_tib_fib_frac_open:
                                # update the injury details to represent the fact that this lower leg fracture is open
                                detail = 'eo'
                                # store the severity information in injais
                                injais.append(3)
                            else:
                                # lower leg fracture is not open so just need to store the injury information in injais
                                injais.append(2)
                        else:
                            # Upper leg fractures all have an AIS score of 3, store this in injais
                            injais.append(3)
                            # determine what the injury is in greater detail, i.e. is this a pelvis/femur/hip fracture
                            probabilities = p['daly_dist_code_813']
                            detail_add_on = ['a', 'b', 'c']
                            # store the exact injury information in the variable detail
                            detail = self.rng.choice(detail_add_on, p=probabilities)
                            # some of the upper leg fractures can be open fractures
                            # if the fracture is a pelvis fracture, check if it is open
                            if detail == 'b':
                                # determine if the pelvis fracture is open
                                prob_pelvis_frac_open = p['prob_pelvis_fracture_open']
                                if open_frac < prob_pelvis_frac_open:
                                    # update the injury detail information to reflect the fact that the fracture is open
                                    detail = detail + 'o'
                            # if the fracture is a femur fracture, check if it is open
                            if detail == 'c':
                                # determine if the femur fracture is open
                                prob_femur_frac_open = p['prob_femur_fracture_open']
                                if open_frac < prob_femur_frac_open:
                                    # update the injury detail information to reflect the fact that the fracture is open
                                    detail = detail + 'o'
                    # Determine if the injury is a dislocation
                    elif boundary_2 < cat <= boundary_3:
                        # dislocation of hip or knee, store the injury category information in injcat
                        injcat.append(int(2))
                        # both dislocated hips and knees have an AIS score of 2, store this in injais
                        injais.append(2)
                        probabilities = p['daly_dist_code_822']
                        detail_add_on = ['a', 'b']
                        # specify whether this injury is a hip or knee dislocation in the variable detail
                        detail = self.rng.choice(detail_add_on, p=probabilities)
                    # Determine if the injury is an amputation
                    elif boundary_3 < cat:
                        # Amputation, so store the injury category in injcat
                        injcat.append(int(8))
                        # Determine how severe the injury is
                        # create probability boundaries to assign injury severity
                        sev_boundary_1 = self.lower_ex_prob_amputation_AIS2
                        sev_boundary_2 = sev_boundary_1 + self.lower_ex_prob_amputation_AIS3
                        if severity <= sev_boundary_1:
                            # Toe/toes amputation have an AIS score of 2, store this in injais
                            injais.append(2)
                        elif sev_boundary_1 < severity <= sev_boundary_2:
                            # Unilateral limb amputation has an AIS score of 3, store this in injais
                            injais.append(3)
                        else:
                            # Bilateral limb amputation has an AIS score of 2, store this in injais
                            injais.append(4)
                # Add the injury detail to the injury detail list, doing this at the end of the loop stores the
                # 'default' detail 'none' if no injury specific details are required, and the specific injury
                # details where applicable.
                injdetail.append(detail)
            # Check that all the relevant injury information has been decided by checking there is a injury category,
            # AIS score, injury location and injury detail specifics for all injuries
            assert len(injcat) == ninj
            assert len(injais) == ninj
            assert len(allinjlocs) == ninj
            assert len(injdetail) == ninj

            # Create a dataframe that stores the injury location and severity for each person, the point of this
            # dataframe is to use some of the pandas tools to manipulate the generated injury data to calculate
            # the ISS score and from this, the probability of mortality resulting from the injuries.
            injlocstring.append(' '.join(map(str, allinjlocs)))
            injcatstring.append(' '.join(map(str, injcat)))
            injaisstring.append(' '.join(map(str, injais)))
            injdetailstr.append(' '.join(map(str, injdetail)))
            injdata = {'AIS location': allinjlocs, 'AIS severity': injais}
            df = pd.DataFrame(injdata, columns=['AIS location', 'AIS severity'])
            # Find the most severe injury to the person in each body region, creates a new column containing the
            # maximum AIS value of each injured body region
            df['Severity max'] = df.groupby(['AIS location'], sort=False)['AIS severity'].transform(max)
            # column no longer needed and will get in the way of future calculations
            df = df.drop(columns='AIS severity')
            # drops the duplicate values in the location data, preserving the most severe injuries in each body
            # location.
            df = df.drop_duplicates(['AIS location'], keep='first')
            # Finds the AIS score for the most severely injured body regions and stores them in a new dataframe z
            # (variable name arbitraty, but only used in the next few lines)
            z = df.nlargest(3, 'Severity max', 'first')
            # Find the 3 most severely injured body regions
            z = z.iloc[:3]
            # Need to determine whether the persons injuries qualify as polytrauma as such injuries have a different
            # prognosis, set default as False. Polytrauma is defined via the new Berlin definition, 'when two or more
            # injuries have an AIS severity score of 3 or higher'.
            # set polytrauma as False by default
            polytrauma = False
            # Determine where more than one injured body region has occurred
            if len(z) > 1:
                # Find where the injuries have an AIS score of 3 or higher
                cond = np.where(z['Severity max'] > 2)
                if len(z.iloc[cond]) > 1:
                    # if two or more injuries have a AIS score of 3 or higher then this person has polytrauma.
                    polytrauma = True
            # Calculate the squares of the AIS scores for the three most severely injured body regions
            z['sqrsev'] = z['Severity max'] ** 2
            # From the squared AIS scores, calculate the ISS score
            ISSscore = sum(z['sqrsev'])
            if ISSscore < 15:
                severity_category.append('mild')
            else:
                severity_category.append('severe')
            # Turn the vectors into a string to store as one entry in a dataframe
            allinjlocs = np.array(allinjlocs)
            allinjlocs = allinjlocs.astype(int)
            allinjlocs = ''.join([str(elem) for elem in allinjlocs])
            predinjlocs.append(allinjlocs)
            predinjsev.append(injais)
            predinjcat.append(injcat)
            predinjiss.append(ISSscore)
            predpolytrauma.append(polytrauma)
            predinjdetail.append(injdetail)
        # create a new data frame
        injdf = pd.DataFrame()
        # store the predicted injury locations
        injdf['Injury locations'] = predinjlocs
        # store the predicted injury locations as a string
        injdf['Injury locations string'] = injlocstring
        # store the predicted injury severity scores
        injdf['Injury AIS'] = predinjsev
        # store the predicted injury severity scores as strings
        injdf['Injury AIS string'] = injaisstring
        # Store the injury category information
        injdf['Injury category'] = predinjcat
        # Store the injury category information as strings
        injdf['Injury category string'] = injcatstring
        # Store the predicted ISS scores
        injdf['ISS'] = predinjiss
        # Store the predicted occurence of polytrauma
        injdf['Polytrauma'] = predpolytrauma
        # Store the predicted injury details
        injdf['Detail'] = injdetailstr
        # set the type of injdf['Injury category string'] to strings
        injdf['Injury category string'] = injdf['Injury category string'].astype(str)
        # Split the injury category information to combine all information as one code later on
        injurycategories = injdf['Injury category string'].str.split(expand=True)
        # set the type of injdf['Injury locations string'] to strings
        injdf['Injury locations string'] = injdf['Injury locations string'].astype(str)
        # Split the injury location information to combine all information as one code later on
        injurylocations = injdf['Injury locations string'].str.split(expand=True)
        # set the type of injdf['Injury AIS string'] to strings
        injdf['Injury AIS string'] = injdf['Injury AIS string'].astype(str)
        # Split the injury severity information to combine all information as one code later on
        injuryais = injdf['Injury AIS string'].str.split(expand=True)
        # set the type of injdf['Detail'] to strings
        injdf['Detail'] = injdf['Detail'].astype(str)
        # Split the injury detail information to combine all information as one code later on
        injurydetails = injdf['Detail'].str.split(expand=True)
        # Change any instances where the injury detail was none to be an empty string, making the TLO injury codes
        # more readable
        injurydetails = injurydetails.replace(to_replace='none', value='')
        # Create the TLO injury codes by combining the injury, category, severity and details information
        injurydescription = injurylocations + injurycategories + injuryais + injurydetails
        # Set the injury description values to be strings
        injurydescription = injurydescription.astype(str)
        # Name the columns in the dateframe to return, needs to be able to react to the maximum number of injuries
        # assigned to people this month i.e. if everyone has one injury, only rename one injury column, if someone
        # have 8 injuries, need to renames injury columns 1 through 8
        for (columnname, columndata) in injurydescription.iteritems():
            injurydescription.rename(
                columns={injurydescription.columns[columnname]: "Injury " + str(columnname + 1)},
                inplace=True)
        # Store the predicted injury ISS scores this month
        injurydescription['ISS'] = predinjiss
        # Store the predicted occurence of polytrauma this month
        injurydescription['Polytrauma'] = predpolytrauma
        # create empty list to store the Military AIS scores used to predict morality without medical care
        MAIS = []
        # iterate of the injur AIS scores and calculate the associated MAIS score
        for item in injdf['Injury AIS'].tolist():
            MAIS.append(max(item) + 1)
        # Store the predicted Military AIS scores
        injurydescription['MAIS_M'] = MAIS
        # Fill dataframe entries where a person has not had an injury assigned with 'none'
        injurydescription = injurydescription.fillna("none")
        # Get injury information in an easily interpreted form to be logged.
        # create a list of the predicted injury locations
        flattened_injury_locations = [item for sublist in injurylocations.values.tolist()
                                      for item in sublist if type(item) is str]
        # create a list of the predicted injury categories
        flattened_injury_category = [item for sublist in injurycategories.values.tolist()
                                     for item in sublist if type(item) is str]
        # create a list of the predicted injury severity scores
        flattened_injury_ais = [item for sublist in injuryais.values.tolist()
                                for item in sublist if type(item) is str]
        # create a list of the predicted injury details
        flattened_injury_detail = [item for sublist in injurydetails.values.tolist()
                                   for item in sublist if type(item) is str]
        # ============================ Injury category incidence ======================================================
        df = self.sim.population.props
        # log the incidence of each injury category
        n_alive = len(df.is_alive)
        amputationcounts = sum(1 for i in flattened_injury_category if i == '8')
        burncounts = sum(1 for i in flattened_injury_category if i == '11')
        fraccounts = sum(1 for i in flattened_injury_category if i == '1')
        tbicounts = sum(1 for i in flattened_injury_category if i == '3')
        minorinjurycounts = sum(1 for i in flattened_injury_category if i == '10')
        spinalcordinjurycounts = sum(1 for i in flattened_injury_category if i == '7')
        other_counts = sum(1 for i in flattened_injury_category if i in ['2', '4', '5', '6', '9'])
        inc_amputations = amputationcounts / ((n_alive - amputationcounts) * 1 / 12) * 100000
        inc_burns = burncounts / ((n_alive - burncounts) * 1 / 12) * 100000
        inc_fractures = fraccounts / ((n_alive - fraccounts) * 1 / 12) * 100000
        inc_tbi = tbicounts / ((n_alive - tbicounts) * 1 / 12) * 100000
        inc_sci = spinalcordinjurycounts / ((n_alive - spinalcordinjurycounts) * 1 / 12) * 100000
        inc_minor = minorinjurycounts / ((n_alive - minorinjurycounts) * 1 / 12) * 100000
        inc_other = other_counts / ((n_alive - other_counts) * 1 / 12) * 100000
        tot_inc_all_inj = inc_amputations + inc_burns + inc_fractures + inc_tbi + inc_sci + inc_minor + inc_other
        dict_to_output = {'inc_amputations': inc_amputations,
                          'inc_burns': inc_burns,
                          'inc_fractures': inc_fractures,
                          'inc_tbi': inc_tbi,
                          'inc_sci': inc_sci,
                          'inc_minor': inc_minor,
                          'inc_other': inc_other,
                          'tot_inc_injuries': tot_inc_all_inj}

        logger.info(key='Inj_category_incidence',
                    data=dict_to_output,
                    description='Incidence of each injury grouped as per the GBD definition')
        # Log injury information
        injury_info = {'Number_of_injuries': number_of_injuries,
                       'Location_of_injuries': flattened_injury_locations,
                       'Injury_category': flattened_injury_category,
                       'Per_injury_severity': flattened_injury_ais,
                       'Per_person_injury_severity': predinjiss,
                       'Per_person_MAIS_score': MAIS,
                       'Per_person_severity_category': severity_category
                       }
        logger.info(key='Injury_information',
                    data=injury_info,
                    description='Relevant information on the injuries from road traffic accidents when they are '
                                'assigned')
        # log the fraction of lower extremity fractions that are open
        # find lower extremity injuries
        index_pos_lx = [i for i in range(len(flattened_injury_locations)) if flattened_injury_locations[i] == '8']
        # find lower extremity fractures
        lx_fractures = [i for i in index_pos_lx if flattened_injury_category[i] == '1']
        # open lx_fractures
        open_lx_fractures = [i for i in lx_fractures if 'o' in flattened_injury_detail[i]]
        # log the proportion of lower extermity fractures that are open
        if len(lx_fractures) > 0:
            proportion_lx_fracture_open = len(open_lx_fractures) / len(lx_fractures)
        else:
            proportion_lx_fracture_open = 'no_lx_fractures'
        injury_info = {'Proportion_lx_fracture_open': proportion_lx_fracture_open}
        logger.info(key='Open_fracture_information',
                    data=injury_info,
                    description='The proportion of fractures that are open in specific body regions')
        # Finally return the injury description information
        return injurydescription


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
        self.prob_perm_disability_with_treatment_severe_TBI = p['prob_perm_disability_with_treatment_severe_TBI']
        self.prob_perm_disability_with_treatment_sci = p['prob_perm_disability_with_treatment_sci']

        # Parameters used to assign injuries in the injrandomizer function
        # Injuries to AIS region 1
        self.head_prob_skin_wound = p['head_prob_skin_wound']
        self.head_prob_skin_wound_open = p['head_prob_skin_wound_open']
        self.head_prob_skin_wound_burn = p['head_prob_skin_wound_burn']
        self.head_prob_fracture = p['head_prob_fracture']
        self.head_prob_fracture_unspecified = p['head_prob_fracture_unspecified']
        self.head_prob_fracture_basilar = p['head_prob_fracture_basilar']
        self.head_prob_TBI = p['head_prob_TBI']
        self.head_prob_TBI_AIS3 = p['head_prob_TBI_AIS3']
        self.head_prob_TBI_AIS4 = p['head_prob_TBI_AIS4']
        self.head_prob_TBI_AIS5 = p['head_prob_TBI_AIS5']
        # Injuries to AIS region 2
        self.face_prob_skin_wound = p['face_prob_skin_wound']
        self.face_prob_skin_wound_open = p['face_prob_skin_wound_open']
        self.face_prob_skin_wound_burn = p['face_prob_skin_wound_burn']
        self.face_prob_fracture = p['face_prob_fracture']
        self.face_prob_fracture_AIS1 = p['face_prob_fracture_AIS1']
        self.face_prob_fracture_AIS2 = p['face_prob_fracture_AIS2']
        self.face_prob_soft_tissue_injury = p['face_prob_soft_tissue_injury']
        self.face_prob_eye_injury = p['face_prob_eye_injury']
        # Injuries to AIS region 3
        self.neck_prob_skin_wound = p['neck_prob_skin_wound']
        self.neck_prob_skin_wound_open = p['neck_prob_skin_wound_open']
        self.neck_prob_skin_wound_burn = p['neck_prob_skin_wound_burn']
        self.neck_prob_soft_tissue_injury = p['neck_prob_soft_tissue_injury']
        self.neck_prob_soft_tissue_injury_AIS2 = p['neck_prob_soft_tissue_injury_AIS2']
        self.neck_prob_soft_tissue_injury_AIS3 = p['neck_prob_soft_tissue_injury_AIS3']
        self.neck_prob_internal_bleeding = p['neck_prob_internal_bleeding']
        self.neck_prob_internal_bleeding_AIS1 = p['neck_prob_internal_bleeding_AIS1']
        self.neck_prob_internal_bleeding_AIS3 = p['neck_prob_internal_bleeding_AIS3']
        self.neck_prob_dislocation = p['neck_prob_dislocation']
        self.neck_prob_dislocation_AIS2 = p['neck_prob_dislocation_AIS2']
        self.neck_prob_dislocation_AIS3 = p['neck_prob_dislocation_AIS3']
        # Injuries to AIS region 4
        self.thorax_prob_skin_wound = p['thorax_prob_skin_wound']
        self.thorax_prob_skin_wound_open = p['thorax_prob_skin_wound_open']
        self.thorax_prob_skin_wound_burn = p['thorax_prob_skin_wound_burn']
        self.thorax_prob_internal_bleeding = p['thorax_prob_internal_bleeding']
        self.thorax_prob_internal_bleeding_AIS1 = p['thorax_prob_internal_bleeding_AIS1']
        self.thorax_prob_internal_bleeding_AIS3 = p['thorax_prob_internal_bleeding_AIS3']
        self.thorax_prob_internal_organ_injury = p['thorax_prob_internal_organ_injury']
        self.thorax_prob_fracture = p['thorax_prob_fracture']
        self.thorax_prob_fracture_ribs = p['thorax_prob_fracture_ribs']
        self.thorax_prob_fracture_flail_chest = p['thorax_prob_fracture_flail_chest']
        self.thorax_prob_soft_tissue_injury = p['thorax_prob_soft_tissue_injury']
        self.thorax_prob_soft_tissue_injury_AIS1 = p['thorax_prob_soft_tissue_injury_AIS1']
        self.thorax_prob_soft_tissue_injury_AIS2 = p['thorax_prob_soft_tissue_injury_AIS2']
        self.thorax_prob_soft_tissue_injury_AIS3 = p['thorax_prob_soft_tissue_injury_AIS3']
        # Injuries to AIS region 5
        self.abdomen_prob_skin_wound = p['abdomen_prob_skin_wound']
        self.abdomen_prob_skin_wound_open = p['abdomen_prob_skin_wound_open']
        self.abdomen_prob_skin_wound_burn = p['abdomen_prob_skin_wound_burn']
        self.abdomen_prob_internal_organ_injury = p['abdomen_prob_internal_organ_injury']
        self.abdomen_prob_internal_organ_injury_AIS2 = p['abdomen_prob_internal_organ_injury_AIS2']
        self.abdomen_prob_internal_organ_injury_AIS3 = p['abdomen_prob_internal_organ_injury_AIS3']
        self.abdomen_prob_internal_organ_injury_AIS4 = p['abdomen_prob_internal_organ_injury_AIS4']
        # Injuries to AIS region 6
        self.spine_prob_spinal_cord_lesion = p['spine_prob_spinal_cord_lesion']
        self.spine_prob_spinal_cord_lesion_neck_level = p['spine_prob_spinal_cord_lesion_neck_level']
        self.spine_prob_spinal_cord_lesion_neck_level_AIS3 = p['spine_prob_spinal_cord_lesion_neck_level_AIS3']
        self.spine_prob_spinal_cord_lesion_neck_level_AIS4 = p['spine_prob_spinal_cord_lesion_neck_level_AIS4']
        self.spine_prob_spinal_cord_lesion_neck_level_AIS5 = p['spine_prob_spinal_cord_lesion_neck_level_AIS5']
        self.spine_prob_spinal_cord_lesion_neck_level_AIS6 = p['spine_prob_spinal_cord_lesion_neck_level_AIS6']
        self.spine_prob_spinal_cord_lesion_below_neck_level = p['spine_prob_spinal_cord_lesion_below_neck_level']
        self.spine_prob_spinal_cord_lesion_below_neck_level_AIS3 = \
            p['spine_prob_spinal_cord_lesion_below_neck_level_AIS3']
        self.spine_prob_spinal_cord_lesion_below_neck_level_AIS4 = \
            p['spine_prob_spinal_cord_lesion_below_neck_level_AIS4']
        self.spine_prob_spinal_cord_lesion_below_neck_level_AIS5 = \
            p['spine_prob_spinal_cord_lesion_below_neck_level_AIS5']
        self.spine_prob_fracture = p['spine_prob_fracture']
        # Injuries to AIS region 7
        self.upper_ex_prob_skin_wound = p['upper_ex_prob_skin_wound']
        self.upper_ex_prob_skin_wound_open = p['upper_ex_prob_skin_wound_open']
        self.upper_ex_prob_skin_wound_burn = p['upper_ex_prob_skin_wound_burn']
        self.upper_ex_prob_fracture = p['upper_ex_prob_fracture']
        self.upper_ex_prob_dislocation = p['upper_ex_prob_dislocation']
        self.upper_ex_prob_amputation = p['upper_ex_prob_amputation']
        self.upper_ex_prob_amputation_AIS2 = p['upper_ex_prob_amputation_AIS2']
        self.upper_ex_prob_amputation_AIS3 = p['upper_ex_prob_amputation_AIS3']
        # Injuries to AIS region 8
        self.lower_ex_prob_skin_wound = p['lower_ex_prob_skin_wound']
        self.lower_ex_prob_skin_wound_open = p['lower_ex_prob_skin_wound_open']
        self.lower_ex_prob_skin_wound_burn = p['lower_ex_prob_skin_wound_burn']
        self.lower_ex_prob_fracture = p['lower_ex_prob_fracture']
        self.lower_ex_prob_fracture_AIS1 = p['lower_ex_prob_fracture_AIS1']
        self.lower_ex_prob_fracture_AIS2 = p['lower_ex_prob_fracture_AIS2']
        self.lower_ex_prob_fracture_AIS3 = p['lower_ex_prob_fracture_AIS3']
        self.lower_ex_prob_dislocation = p['lower_ex_prob_dislocation']
        self.lower_ex_prob_amputation = p['lower_ex_prob_amputation']
        self.lower_ex_prob_amputation_AIS2 = p['lower_ex_prob_amputation_AIS2']
        self.lower_ex_prob_amputation_AIS3 = p['lower_ex_prob_amputation_AIS3']
        self.lower_ex_prob_amputation_AIS4 = p['lower_ex_prob_amputation_AIS3']

        # DALY weights
        self.daly_wt_unspecified_skull_fracture = p['daly_wt_unspecified_skull_fracture']
        self.daly_wt_basilar_skull_fracture = p['daly_wt_basilar_skull_fracture']
        self.daly_wt_epidural_hematoma = p['daly_wt_epidural_hematoma']
        self.daly_wt_subdural_hematoma = p['daly_wt_subdural_hematoma']
        self.daly_wt_subarachnoid_hematoma = p['daly_wt_subarachnoid_hematoma']
        self.daly_wt_brain_contusion = p['daly_wt_brain_contusion']
        self.daly_wt_intraventricular_haemorrhage = p['daly_wt_intraventricular_haemorrhage']
        self.daly_wt_diffuse_axonal_injury = p['daly_wt_diffuse_axonal_injury']
        self.daly_wt_subgaleal_hematoma = p['daly_wt_subgaleal_hematoma']
        self.daly_wt_midline_shift = p['daly_wt_midline_shift']
        self.daly_wt_facial_fracture = p['daly_wt_facial_fracture']
        self.daly_wt_facial_soft_tissue_injury = p['daly_wt_facial_soft_tissue_injury']
        self.daly_wt_eye_injury = p['daly_wt_eye_injury']
        self.daly_wt_neck_soft_tissue_injury = p['daly_wt_neck_soft_tissue_injury']
        self.daly_wt_neck_internal_bleeding = p['daly_wt_neck_internal_bleeding']
        self.daly_wt_neck_dislocation = p['daly_wt_neck_dislocation']
        self.daly_wt_chest_wall_bruises_hematoma = p['daly_wt_chest_wall_bruises_hematoma']
        self.daly_wt_hemothorax = p['daly_wt_hemothorax']
        self.daly_wt_lung_contusion = p['daly_wt_lung_contusion']
        self.daly_wt_diaphragm_rupture = p['daly_wt_diaphragm_rupture']
        self.daly_wt_rib_fracture = p['daly_wt_rib_fracture']
        self.daly_wt_flail_chest = p['daly_wt_flail_chest']
        self.daly_wt_chest_wall_laceration = p['daly_wt_chest_wall_laceration']
        self.daly_wt_closed_pneumothorax = p['daly_wt_closed_pneumothorax']
        self.daly_wt_open_pneumothorax = p['daly_wt_open_pneumothorax']
        self.daly_wt_surgical_emphysema = p['daly_wt_surgical_emphysema']
        self.daly_wt_abd_internal_organ_injury = p['daly_wt_abd_internal_organ_injury']
        self.daly_wt_spinal_cord_lesion_neck_with_treatment = p['daly_wt_spinal_cord_lesion_neck_with_treatment']
        self.daly_wt_spinal_cord_lesion_neck_without_treatment = p['daly_wt_spinal_cord_lesion_neck_without_treatment']
        self.daly_wt_spinal_cord_lesion_below_neck_with_treatment = p[
            'daly_wt_spinal_cord_lesion_below_neck_with_treatment']
        self.daly_wt_spinal_cord_lesion_below_neck_without_treatment = p[
            'daly_wt_spinal_cord_lesion_below_neck_without_treatment']
        self.daly_wt_vertebrae_fracture = p['daly_wt_vertebrae_fracture']
        self.daly_wt_clavicle_scapula_humerus_fracture = p['daly_wt_clavicle_scapula_humerus_fracture']
        self.daly_wt_hand_wrist_fracture_with_treatment = p['daly_wt_hand_wrist_fracture_with_treatment']
        self.daly_wt_hand_wrist_fracture_without_treatment = p['daly_wt_hand_wrist_fracture_without_treatment']
        self.daly_wt_radius_ulna_fracture_short_term_with_without_treatment = p[
            'daly_wt_radius_ulna_fracture_short_term_with_without_treatment']
        self.daly_wt_radius_ulna_fracture_long_term_without_treatment = p[
            'daly_wt_radius_ulna_fracture_long_term_without_treatment']
        self.daly_wt_dislocated_shoulder = p['daly_wt_dislocated_shoulder']
        self.daly_wt_amputated_finger = p['daly_wt_amputated_finger']
        self.daly_wt_amputated_thumb = p['daly_wt_amputated_thumb']
        self.daly_wt_unilateral_arm_amputation_with_treatment = p['daly_wt_unilateral_arm_amputation_with_treatment']
        self.daly_wt_unilateral_arm_amputation_without_treatment = p[
            'daly_wt_unilateral_arm_amputation_without_treatment']
        self.daly_wt_bilateral_arm_amputation_with_treatment = p['daly_wt_bilateral_arm_amputation_with_treatment']
        self.daly_wt_bilateral_arm_amputation_without_treatment = p[
            'daly_wt_bilateral_arm_amputation_without_treatment']
        self.daly_wt_foot_fracture_short_term_with_without_treatment = p[
            'daly_wt_foot_fracture_short_term_with_without_treatment']
        self.daly_wt_foot_fracture_long_term_without_treatment = p['daly_wt_foot_fracture_long_term_without_treatment']
        self.daly_wt_patella_tibia_fibula_fracture_with_treatment = p[
            'daly_wt_patella_tibia_fibula_fracture_with_treatment']
        self.daly_wt_patella_tibia_fibula_fracture_without_treatment = p[
            'daly_wt_patella_tibia_fibula_fracture_without_treatment']
        self.daly_wt_hip_fracture_short_term_with_without_treatment = p[
            'daly_wt_hip_fracture_short_term_with_without_treatment']
        self.daly_wt_hip_fracture_long_term_with_treatment = p['daly_wt_hip_fracture_long_term_with_treatment']
        self.daly_wt_hip_fracture_long_term_without_treatment = p['daly_wt_hip_fracture_long_term_without_treatment']
        self.daly_wt_pelvis_fracture_short_term = p['daly_wt_pelvis_fracture_short_term']
        self.daly_wt_pelvis_fracture_long_term = p['daly_wt_pelvis_fracture_long_term']
        self.daly_wt_femur_fracture_short_term = p['daly_wt_femur_fracture_short_term']
        self.daly_wt_femur_fracture_long_term_without_treatment = p[
            'daly_wt_femur_fracture_long_term_without_treatment']
        self.daly_wt_dislocated_hip = p['daly_wt_dislocated_hip']
        self.daly_wt_dislocated_knee = p['daly_wt_dislocated_knee']
        self.daly_wt_amputated_toes = p['daly_wt_amputated_toes']
        self.daly_wt_unilateral_lower_limb_amputation_with_treatment = p[
            'daly_wt_unilateral_lower_limb_amputation_with_treatment']
        self.daly_wt_unilateral_lower_limb_amputation_without_treatment = p[
            'daly_wt_unilateral_lower_limb_amputation_without_treatment']
        self.daly_wt_bilateral_lower_limb_amputation_with_treatment = p[
            'daly_wt_bilateral_lower_limb_amputation_with_treatment']
        self.daly_wt_bilateral_lower_limb_amputation_without_treatment = p[
            'daly_wt_bilateral_lower_limb_amputation_without_treatment']
        self.daly_wt_burns_greater_than_20_percent_body_area = p['daly_wt_burns_greater_than_20_percent_body_area']
        self.daly_wt_burns_less_than_20_percent_body_area_with_treatment = p[
            'daly_wt_burns_less_than_20_percent_body_area_with_treatment']
        self.daly_wt_burns_less_than_20_percent_body_area_without_treatment = p[
            'daly_wt_burns_less_than_20_percent_body_area_with_treatment']
        self.rt_emergency_care_ISS_score_cut_off = p['rt_emergency_care_ISS_score_cut_off']

    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """
        df = population.props
        now = self.sim.date
        # Reset injury properties after death, get an index of people who have died due to RTI, all causes
        immdeathidx = df.index[df.rt_imm_death]
        deathwithmedidx = df.index[df.rt_post_med_death]
        deathwithoutmedidx = df.index[df.rt_no_med_death]
        deathunavailablemedidx = df.index[df.rt_unavailable_med_death]
        diedfromrtiidx = immdeathidx.union(deathwithmedidx)
        diedfromrtiidx = diedfromrtiidx.union(deathwithoutmedidx)
        diedfromrtiidx = diedfromrtiidx.union(deathunavailablemedidx)
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
        # reset whether they have been selected for an injury this month
        df['rt_road_traffic_inc'] = False
        # reset whether they have sought care this month
        df['rt_diagnosed'] = False
        # reset whether they have been given care this month
        df['rt_med_int'] = False
        df.loc[df.is_alive, 'rt_post_med_death'] = False
        df.loc[df.is_alive, 'rt_no_med_death'] = False
        df.loc[df.is_alive, 'rt_unavailable_med_death'] = False

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
                         Predictor('age_years').when('.between(0,4)', self.rr_injrti_age04),
                         Predictor('age_years').when('.between(5,9)', self.rr_injrti_age59),
                         Predictor('age_years').when('.between(10,17)', self.rr_injrti_age1017),
                         Predictor('age_years').when('.between(18,29)', self.rr_injrti_age1829),
                         Predictor('age_years').when('.between(30,39)', self.rr_injrti_age3039),
                         Predictor('age_years').when('.between(40,49)', self.rr_injrti_age4049),
                         Predictor('age_years').when('.between(50,59)', self.rr_injrti_age5059),
                         Predictor('age_years').when('.between(60,69)', self.rr_injrti_age6069),
                         Predictor('age_years').when('.between(70,79)', self.rr_injrti_age7079),
                         Predictor('li_ex_alc').when(True, self.rr_injrti_excessalcohol)
                         )
        pred = eq.predict(df.iloc[rt_current_non_ind])
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
            self.sim.schedule_event(
                demography.InstantaneousDeath(self.module, individual_id, "RTI_imm_death"),
                self.sim.date
            )
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
        # any person with the condition 'cause_of_death' != ''
        died_elsewhere_index = selected_for_rti_inj[selected_for_rti_inj['cause_of_death'] != ''].index
        # drop the died_elsewhere_index from selected_for_rti_inj
        selected_for_rti_inj.drop(died_elsewhere_index, inplace=True)
        # Create shorthand link to RTI module
        road_traffic_injuries = self.sim.modules['RTI']

        # Assign the injuries using the assign injuries function
        description = road_traffic_injuries.assign_injuries(len(selected_for_rti_inj))
        # replace the nan values with 'none', this is so that the injuries can be copied over from this temporarily used
        # pandas dataframe will fit in with the categories in the columns rt_injury_1 through rt_injury_8
        description = description.replace('nan', 'none')
        # set the index of the description dataframe, so that we can join it to the selected_for_rti_inj dataframe
        description = description.set_index(selected_for_rti_inj.index)
        # join the description dataframe, which stores information on people's injuries to the copy of
        # self.sim.population.props which contains the index of people involved in RTIs
        selected_for_rti_inj = selected_for_rti_inj.join(description.set_index(selected_for_rti_inj.index))
        # begin copying the results from the selected_for_rti_inj dataframe to self.sim.population.props
        for person_id in selected_for_rti_inj.index:
            # copy over injury severity
            df.loc[person_id, 'rt_ISS_score'] = description.loc[person_id, 'ISS']
            df.loc[person_id, 'rt_MAIS_military_score'] = description.loc[person_id, 'MAIS_M']
        # ======================== Apply the injuries to the population dataframe ======================================
        # Copy entries from selected_for_rti_inj dataframe to self.sim.population.props.
        injury_columns = pd.Index(['Injury 1', 'Injury 2', 'Injury 3', 'Injury 4', 'Injury 5', 'Injury 6', 'Injury 7',
                                   'Injury 8'])
        # iterate over the number of injury columns in description
        for ninjuries in range(0, len(description.columns.intersection(injury_columns))):
            # copy over injuries column by column
            for person_id in selected_for_rti_inj.index:
                # copy over injuries person by person... may be slower than optimal
                if ninjuries == 0:
                    df.loc[person_id, 'rt_injury_1'] = description.loc[person_id, 'Injury 1']
                if ninjuries == 1:
                    df.loc[person_id, 'rt_injury_2'] = description.loc[person_id, 'Injury 2']
                if ninjuries == 2:
                    df.loc[person_id, 'rt_injury_3'] = description.loc[person_id, 'Injury 3']
                if ninjuries == 3:
                    df.loc[person_id, 'rt_injury_4'] = description.loc[person_id, 'Injury 4']
                if ninjuries == 4:
                    df.loc[person_id, 'rt_injury_5'] = description.loc[person_id, 'Injury 5']
                if ninjuries == 5:
                    df.loc[person_id, 'rt_injury_6'] = description.loc[person_id, 'Injury 6']
                if ninjuries == 6:
                    df.loc[person_id, 'rt_injury_7'] = description.loc[person_id, 'Injury 7']
                if ninjuries == 7:
                    df.loc[person_id, 'rt_injury_8'] = description.loc[person_id, 'Injury 8']
        # Run assert statements to make sure the model is behaving as it should
        # All those who are injured in a road traffic accident have this noted in the property 'rt_road_traffic_inc'
        assert sum(df.loc[selected_for_rti, 'rt_road_traffic_inc']) == len(selected_for_rti)
        # All those who are involved in a road traffic accident have these noted in the property 'rt_date_inj'
        assert len(df.loc[selected_for_rti, 'rt_date_inj'] != pd.NaT) == len(selected_for_rti)
        # All those who are injures and do not die immediately have an ISS score > 0
        assert len(df.loc[df.rt_road_traffic_inc & ~df.rt_imm_death, 'rt_ISS_score'] > 0) == \
               len(df.loc[df.rt_road_traffic_inc & ~df.rt_imm_death])
        # ========================== Decide survival time without medical intervention ================================
        # todo: find better time for survival data without med int for ISS scores
        df.loc[selected_for_rti_inj.index, 'rt_date_death_no_med'] = now + DateOffset(days=7)
        # ============================ Injury severity classification =================================================
        # Find those with mild injuries and update the rt_roadtrafficinj property so they have a mild injury
        mild_rti_idx = selected_for_rti_inj.index[selected_for_rti_inj.is_alive & selected_for_rti_inj['ISS'] < 15]
        df.loc[mild_rti_idx, 'rt_inj_severity'] = 'mild'
        # Find those with severe injuries and update the rt_roadtrafficinj property so they have a severe injury
        severe_rti_idx = selected_for_rti_inj.index[selected_for_rti_inj['ISS'] >= 15]
        df.loc[severe_rti_idx, 'rt_inj_severity'] = 'severe'
        # check that everyone who has been assigned an injury this month has an associated injury severity
        assert sum(df.loc[df.rt_road_traffic_inc & ~df.rt_imm_death & (df.rt_date_inj == now), 'rt_inj_severity']
                   != 'none') == len(selected_for_rti_inj.index)
        # Find those with polytrauma and update the rt_polytrauma property so they have polytrauma
        polytrauma_idx = selected_for_rti_inj.loc[selected_for_rti_inj.Polytrauma].index
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

    def apply(self, population):
        df = population.props
        now = self.sim.date
        # check if anyone is due to have their mortality without medical intervention determined today
        if len(df.loc[df['rt_date_death_no_med'] == now]) > 0:
            # Get an index of those scheduled to have their mortality checked
            due_to_die_today_without_med_int = df.loc[df['rt_date_death_no_med'] == now].index
            # iterate over those scheduled to die
            for person in due_to_die_today_without_med_int:
                # Create a random number to determine mortality
                rand_for_death = self.sim.rng.random_sample(1)
                # create a variable to show if a person has died due to their untreated injuries
                died = False
                # for each rt_MAIS_military_score, determine mortality
                if (df.loc[person, 'rt_MAIS_military_score'] == 3) & (rand_for_death < self.prob_death_MAIS3):
                    died = True
                elif (df.loc[person, 'rt_MAIS_military_score'] == 4) & (rand_for_death < self.prob_death_MAIS4):
                    died = True
                elif (df.loc[person, 'rt_MAIS_military_score'] == 5) & (rand_for_death < self.prob_death_MAIS5):
                    died = True
                elif (df.loc[person, 'rt_MAIS_military_score'] == 6) & (rand_for_death < self.prob_death_MAIS6):
                    died = True
                if died:
                    # If determined to die, schedule a death without med
                    df.loc[person, 'rt_no_med_death'] = True
                    self.sim.schedule_event(demography.InstantaneousDeath(self.module, person,
                                                                          cause='RTI_death_without_med'), self.sim.date)
                else:
                    # If the people do not die from their injuries despite not getting care, we have to decide when and
                    # to what degree their injuries will heal.
                    df.loc[[person], 'rt_recovery_no_med'] = True
                    # Reset the date to check if they die
                    df.loc[[person], 'rt_date_death_no_med'] = pd.NaT
                    cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                            'rt_injury_7', 'rt_injury_8']
                    swapping_codes = ['712c', '811', '813a', '813b', '813c']
                    road_traffic_injuries = self.sim.modules['RTI']
                    # If those who haven't sought health care have an injury for which we have a daly code
                    # associated with that injury long term without treatment, swap it
                    # Iterate over the person's injuries
                    injuries = df.loc[[person], cols].values.tolist()
                    # Cannot iterate correctly over list like [[1,2,3]], so need to flatten
                    flattened_injuries = [item for sublist in injuries for item in sublist if item != 'none']
                    persons_injuries = df.loc[[person], cols]
                    for code in flattened_injuries:
                        swapable_code = np.intersect1d(code, swapping_codes)
                        if len(swapable_code) > 0:
                            swapable_code = swapable_code[0]
                        else:
                            swapable_code = 'none'
                        # check that the person has the injury code
                        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(persons_injuries, [code])
                        assert counts > 0
                        # if the code is swappable, swap it
                        if swapable_code == '712c':
                            df.loc[person, 'rt_disability'] -= \
                                self.daly_wt_radius_ulna_fracture_short_term_with_without_treatment
                            df.loc[person, 'rt_disability'] += \
                                self.daly_wt_radius_ulna_fracture_long_term_without_treatment
                            df.loc[person, 'rt_debugging_DALY_wt'] -= \
                                self.daly_wt_radius_ulna_fracture_short_term_with_without_treatment
                            df.loc[person, 'rt_debugging_DALY_wt'] += \
                                self.daly_wt_radius_ulna_fracture_long_term_without_treatment
                        if swapable_code == '811':
                            df.loc[person, 'rt_disability'] -= \
                                self.daly_wt_foot_fracture_short_term_with_without_treatment
                            df.loc[person, 'rt_disability'] += \
                                self.daly_wt_foot_fracture_long_term_without_treatment
                            df.loc[person, 'rt_debugging_DALY_wt'] -= \
                                self.daly_wt_foot_fracture_short_term_with_without_treatment
                            df.loc[person, 'rt_debugging_DALY_wt'] += \
                                self.daly_wt_foot_fracture_long_term_without_treatment
                        if swapable_code == '813a':
                            df.loc[person, 'rt_disability'] -= \
                                self.daly_wt_hip_fracture_short_term_with_without_treatment
                            df.loc[person, 'rt_disability'] += \
                                self.daly_wt_hip_fracture_long_term_without_treatment
                            df.loc[person, 'rt_debugging_DALY_wt'] -= \
                                self.daly_wt_hip_fracture_short_term_with_without_treatment
                            df.loc[person, 'rt_debugging_DALY_wt'] += \
                                self.daly_wt_hip_fracture_long_term_without_treatment
                        if swapable_code == '813b':
                            df.loc[person, 'rt_disability'] -= \
                                self.daly_wt_pelvis_fracture_short_term
                            df.loc[person, 'rt_disability'] += \
                                self.daly_wt_pelvis_fracture_long_term
                            df.loc[person, 'rt_debugging_DALY_wt'] -= \
                                self.daly_wt_pelvis_fracture_short_term
                            df.loc[person, 'rt_debugging_DALY_wt'] += \
                                self.daly_wt_pelvis_fracture_long_term
                        if swapable_code == '813c':
                            df.loc[person, 'rt_disability'] -= \
                                self.daly_wt_femur_fracture_short_term
                            df.loc[person, 'rt_disability'] += \
                                self.daly_wt_femur_fracture_long_term_without_treatment
                            df.loc[person, 'rt_debugging_DALY_wt'] -= \
                                self.daly_wt_femur_fracture_short_term
                            df.loc[person, 'rt_debugging_DALY_wt'] += \
                                self.daly_wt_femur_fracture_long_term_without_treatment
                        if df.loc[person, 'rt_disability'] < 0:
                            df.loc[person, 'rt_disability'] = 0
                        if df.loc[person, 'rt_disability'] > 1:
                            df.loc[person, 'rt_disability'] = 1
                        # If they don't have a swappable code, schedule the healing of the injury
                        # get the persons injuries
                        persons_injuries = df.loc[[person], cols]
                        non_empty_injuries = persons_injuries[persons_injuries != "none"]
                        non_empty_injuries = non_empty_injuries.dropna(axis=1)
                        injury_columns = non_empty_injuries.columns
                        # Fractures
                        if code == '112':
                            # schedule a recovery date for the skull fracture
                            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person,
                                                                                                          [code])[0]
                                                             )
                            # using estimated 7 weeks PLACEHOLDER FOR SKULL FRACTURE
                            df.loc[person, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                                weeks=7)
                        if code == '211' or code == '212':
                            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person,
                                                                                                          [code])[
                                                                 0])
                            # using estimated 7 weeks to recover from facial fracture surgery
                            df.loc[person, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                                weeks=7)
                        if code == '412':
                            # schedule a recovery date for the rib fracture
                            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person,
                                                                                                          [code])[0]
                                                             )
                            # using estimated 5 weeks PLACEHOLDER FOR rib FRACTURE
                            df.loc[person, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                                weeks=5)
                        if code == '612':
                            # schedule a recovery date for the vertebrae fracture
                            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person,
                                                                                                          [code])[0]
                                                             )
                            # using estimated 9 weeks PLACEHOLDER FOR Vertebrae FRACTURE
                            df.loc[person, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                                weeks=9)
                        castable_fractures = ['712a', '712a', '712b', '712c', '811', '812']
                        if code in castable_fractures:
                            # schedule a recovery date for the castable fractures
                            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person,
                                                                                                          [code])[0]
                                                             )
                            # using estimated 10 weeks PLACEHOLDER non casted, 'castable' fracture
                            df.loc[person, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                                weeks=10)
                        # Dislocations
                        if code == '322':
                            # estimated six week recovery for neck dislocation
                            columns = injury_columns.get_loc(
                                road_traffic_injuries.rti_find_injury_column(person,
                                                                             [code])[0])
                            df.loc[person, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                                weeks=6)
                        if code == '722':
                            columns = injury_columns.get_loc(
                                road_traffic_injuries.rti_find_injury_column(person,
                                                                             [code])[0])
                            # using estimated 12 weeks to recover from dislocated shoulder
                            df.loc[person, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                                weeks=12)
                        if code == '822a':
                            columns = injury_columns.get_loc(
                                road_traffic_injuries.rti_find_injury_column(person,
                                                                             [code])[0])
                            # using estimated 2 months to recover from dislocated hip
                            df.loc[person, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                                months=2)
                        if code == '822b':
                            columns = injury_columns.get_loc(
                                road_traffic_injuries.rti_find_injury_column(person,
                                                                             [code])[0])
                            # using estimated 6 months to recover from dislocated knee
                            df.loc[person, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                                weeks=6)
                        # Soft tissue injuries
                        if code == '241':
                            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person,
                                                                                                          [code])[
                                                                 0])
                            # using estimated 1 week to recover from soft tissue injury
                            df.loc[person, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                                month=1)
                        if code == '342':
                            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person,
                                                                                                          [code])[
                                                                 0])
                            # using estimated 6 weeks PLACEHOLDER FOR VERTEBRAL ARTERY LACERATION
                            df.loc[person, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                                weeks=6)
                        if code == '441':
                            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person,
                                                                                                          [code])[
                                                                 0])
                            # using estimated 1 - 2 week recovery time for pneumothorax
                            df.loc[person, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                                weeks=2)
                        if code == '442':
                            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person,
                                                                                                          [code])[
                                                                 0])
                            # using estimated 1 - 2 week recovery time for Surgical emphysema
                            df.loc[person, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                                weeks=2)
                        # Internal organ injuries
                        if code == '552':
                            # Schedule the recovery date for the injury
                            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person,
                                                                                                          [code])[0]
                                                             )
                            # using estimated 3 months PLACEHOLDER FOR ABDOMINAL TRAUMA
                            df.loc[person, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                                months=3)
                        # Internal bleeding
                        if code == '361' or code == '461':
                            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person,
                                                                                                          [code])[
                                                                 0])
                            # using estimated 1 weeks PLACEHOLDER FOR INTERNAL BLEEDING
                            df.loc[person, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                                weeks=1)
                        # Eye injuries
                        if code == '291':
                            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person,
                                                                                                          [code])[
                                                                 0])
                            # using estimated 1 week to recover from eye surgery injury
                            df.loc[person, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                                weeks=1)
                        # Lacerations
                        laceration_codes = ['1101', '2101', '3101', '4101', '5101', '6101', '7101', '8101']
                        if code in laceration_codes:
                            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person,
                                                                                                          [code])[
                                                                 0])
                            # using estimated 1 week to recover from laceration
                            df.loc[person, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                                weeks=1)


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
        # Isolate the relevant population
        treated_persons = df.loc[(df.is_alive & df.rt_med_int) | (df.is_alive & df.rt_recovery_no_med)]
        # Isolate the relevant information
        recovery_dates = treated_persons['rt_date_to_remove_daly']
        default_recovery = [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT]
        injury_cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                       'rt_injury_7', 'rt_injury_8']
        # Iterate over all the injured people who are having medical treatment
        for person in recovery_dates.index:
            # Iterate over all the dates in 'rt_date_to_remove_daly'
            persons_injuries = df.loc[person, injury_cols]
            for date in treated_persons.loc[person, 'rt_date_to_remove_daly']:
                # check if the recovery date is today
                if date == now:
                    # find the index for the injury which the person has recovered from
                    dateindex = treated_persons.loc[person, 'rt_date_to_remove_daly'].index(date)
                    # find the injury code associated with the healed injury
                    code_to_remove = [df.loc[person, f'rt_injury_{dateindex + 1}']]
                    # Set the healed injury recovery data back to the default state
                    df.loc[person, 'rt_date_to_remove_daly'][dateindex] = pd.NaT
                    # Remove the daly weight associated with the healed injury code
                    road_traffic_injuries.rti_alter_daly_post_treatment(person, code_to_remove)
                    # Check whether all their injuries are healed so the injury properties can be reset
                    if df.loc[person, 'rt_date_to_remove_daly'] == default_recovery:
                        # remove the injury severity as person is uninjured
                        df.loc[person, 'rt_inj_severity'] = "none"
                        # check that all codes in rt_injuries_to_heal_with_time are removed
                        for code in df.loc[person, 'rt_injuries_to_heal_with_time']:
                            idx, counts = road_traffic_injuries.rti_find_and_count_injuries(persons_injuries, [code])
                            if counts == 0:
                                # if for some reason the code hasn't been removed, remove it
                                df.loc[person, 'rt_injuries_to_heal_with_time'].remove(code)
                        assert df.loc[person, 'rt_injuries_to_heal_with_time'] == [], \
                            df.loc[person, 'rt_injuries_to_heal_with_time']
                        # check that all codes in rt_injuries_for_minor_surgery are removed
                        for code in df.loc[person, 'rt_injuries_for_minor_surgery']:
                            idx, counts = road_traffic_injuries.rti_find_and_count_injuries(persons_injuries, [code])
                            if counts == 0:
                                # if for some reason the code hasn't been removed, remove it
                                df.loc[person, 'rt_injuries_for_minor_surgery'].remove(code)
                        assert df.loc[person, 'rt_injuries_for_minor_surgery'] == [], \
                            df.loc[person, 'rt_injuries_for_minor_surgery']
                        # check that all codes in rt_injuries_for_major_surgery are removed
                        for code in df.loc[person, 'rt_injuries_for_major_surgery']:
                            idx, counts = road_traffic_injuries.rti_find_and_count_injuries(persons_injuries, [code])
                            if counts == 0:
                                # if for some reason the code hasn't been removed, remove it
                                df.loc[person, 'rt_injuries_for_major_surgery'].remove(code)
                        assert df.loc[person, 'rt_injuries_for_major_surgery'] == [], \
                            df.loc[person, 'rt_injuries_for_major_surgery']
                        # check that all codes in rt_injuries_for_open_fracture_treatment are removed
                        for code in df.loc[person, 'rt_injuries_for_open_fracture_treatment']:
                            idx, counts = road_traffic_injuries.rti_find_and_count_injuries(persons_injuries, [code])
                            if counts == 0:
                                # if for some reason the code hasn't been removed, remove it
                                df.loc[person, 'rt_injuries_for_open_fracture_treatment'].remove(code)
                        assert df.loc[person, 'rt_injuries_for_open_fracture_treatment'] == [], \
                            df.loc[person, 'rt_injuries_for_open_fracture_treatment']
                        # check that all codes in rt_injuries_to_cast are removed
                        for code in df.loc[person, 'rt_injuries_to_cast']:
                            idx, counts = road_traffic_injuries.rti_find_and_count_injuries(persons_injuries, [code])
                            if counts == 0:
                                # if for some reason the code hasn't been removed, remove it
                                df.loc[person, 'rt_injuries_to_cast'].remove(code)
                        assert df.loc[person, 'rt_injuries_to_cast'] == [], \
                            df.loc[person, 'rt_injuries_to_cast']

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

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        road_traffic_injuries = self.sim.modules['RTI']

        df = self.sim.population.props
        p = module.parameters
        # Load the parameters used in this event
        self.prob_depressed_skull_fracture = p['prob_depressed_skull_fracture']  # proportion of depressed skull
        # fractures in https://doi.org/10.1016/j.wneu.2017.09.084
        self.prob_mild_burns = p['prob_mild_burns']  # proportion of burns accross SSA with TBSA < 10
        # https://doi.org/10.1016/j.burns.2015.04.006
        self.prob_TBI_require_craniotomy = p['prob_TBI_require_craniotomy']
        self.prob_exploratory_laparotomy = p['prob_exploratory_laparotomy']
        self.prob_death_with_med_mild = p['prob_death_with_med_mild']
        self.prob_death_with_med_severe = p['prob_death_with_med_severe']
        self.prob_dislocation_requires_surgery = p['prob_dislocation_requires_surgery']
        self.allowed_interventions = p['allowed_interventions']
        self.prob_perm_disability_with_treatment_severe_TBI = p['prob_perm_disability_with_treatment_severe_TBI']
        # Create an empty list for injuries that are potentially healed without further medical intervention
        self.heal_with_time_injuries = []
        # Define the call on resources of this treatment event: Time of Officers (Appointments)
        #   - get an 'empty' foot
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_accepted_facility_level = 1
        # Place holder appointment footprints to ensure there is at least one
        is_child = self.sim.population.props.at[person_id, 'age_years'] < 5.0
        if is_child:
            the_appt_footprint['Under5OPD'] = 1.0  # Child out-patient appointment
        else:
            the_appt_footprint['Over5OPD'] = 1.0  # Adult out-patient appointment

        # ======================= Design treatment plan, appointment type =============================================
        """ Here, RTI_MedInt designs the treatment plan of the person's injuries, the following determines what the
        major and minor surgery requirements will be

        """
        # Create variables to count how many major or minor surgeries will be required to treat this person
        self.major_surgery_counts = 0
        self.minor_surgery_counts = 0
        # Isolate the relevant injury information
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        # ------------------------------- Skull fractures -------------------------------------------------------------
        # Check if the person has a skull fracture and whether the skull fracture is a depressed skull fracture. If the
        # fracture is depressed, schedule a surgery.
        codes = ['112']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        require_surgery = self.module.rng.random_sample(size=1)
        if counts > 0:
            # given that they have a skull fracture check if they need a surgery
            if require_surgery < self.prob_depressed_skull_fracture:
                # update the number of surgeries needed to treat this person
                self.major_surgery_counts += 1
                # add the injury to the injuries to be treated with major surgery so they aren't treated elsewhere
                df.loc[person_id, 'rt_injuries_for_major_surgery'].append('112')
            else:
                # if they don't need surgery then injury will heal over time
                self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, '112')
        # check if this person has a basilar skull fracture and needs the injury to heal with time
        codes = ['113']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            # add the injury to the heal with time injuries
            self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, '113')
        # -------------------------------- Facial fractures -----------------------------------------------------------
        # Check whether the person has facial fractures, then if they do schedule a surgery to treat it
        codes = ['211', '212']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            # determine the exact injury they have
            actual_injury = np.intersect1d(codes, person_injuries.values)
            # update the number of minor surgeries needed
            self.minor_surgery_counts += 1
            # store the injury to the injuries treated by minor surgery to make sure they aren't treated elsewhere
            df.loc[person_id, 'rt_injuries_for_minor_surgery'].append(actual_injury[0])
        # consumables required: closed reduction. In some cases surgery
        # --------------------------------- Thorax Fractures -----------------------------------------------------------
        # Check whether the person has a broken rib (and therefor needs no further medical care apart from pain
        # management) or if they have flail chest, a life threatening condition which will require surgery.

        # check if they have a rib fracture
        codes = ['412']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            # if they have a rib fracture, add this to the heal with time injuries
            self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, '412')
        codes = ['414']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            # update the number of major surgeries required
            self.major_surgery_counts += 1
            # add the injury to the injuries to be treated by major surgeries so it isn't treated elsewhere
            df.loc[person_id, 'rt_injuries_for_major_surgery'].append('414')
        # --------------------------------- Lower extremity fractures --------------------------------------------------
        # todo: work out if the amputations need to be included as a swap or if they already exist
        # Three treatment options in use currently for treating foot fractures:
        # 1) major surgery - open reduction internal fixation
        # 2) minor surgery - external fixation
        # 3) fracture cast
        # Later I hope to include amputation of foot fractures which occurs in some case. The information used to
        # predict the treatment plan comes from Chagomerana et al. 2017, a hospital report from a hospital in Lilongwe
        # Design treatment plan for foot fractures: first specify the treatments that are available and store them for
        # use in the apply section
        self.foot_frac_major_surg = False
        self.foot_frac_minor_surg = False
        self.foot_frac_amputation = False
        self.foot_frac_cast = False
        # Load the parameters used to determine which treatment option to use
        prob_foot_frac_require_cast = p['prob_foot_frac_require_cast']
        prob_foot_frac_require_maj_surg = p['prob_foot_frac_require_maj_surg']
        prob_foot_frac_require_min_surg = p['prob_foot_frac_require_min_surg']
        # Check if this person has a foot fracture
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, ['811'])
        if counts > 0:
            # Use a random variable to choose which treatment to use
            treatment_plan = self.module.rng.random_sample(size=1)
            # check if this tibia fracture can be treated with a cast
            if treatment_plan < prob_foot_frac_require_cast:
                # update treatment plan
                self.foot_frac_cast = True
                # put the injury in the injuries to be cast property to stop them being treated elsewhere
                df.loc[person_id, 'rt_injuries_to_cast'].append('811')
            # Check if this fracture needs to be treated with external fixation (minor surgery)
            elif treatment_plan < prob_foot_frac_require_cast + prob_foot_frac_require_min_surg:
                # update treatment plan
                self.foot_frac_minor_surg = True
                # update the number of minor surgeries needed
                self.minor_surgery_counts += 1
                # put the injury in the injuries for minor surgery property to stop them being treated elsewhere
                df.loc[person_id, 'rt_injuries_for_minor_surgery'].append('811')
            # Check if this fracture needs to be treated with open reduction internal fixation (major surg)
            elif treatment_plan < prob_foot_frac_require_cast + prob_foot_frac_require_min_surg + \
                prob_foot_frac_require_maj_surg:
                # update the treatment plan
                self.foot_frac_major_surg = True
                # update the number of major surgeries
                self.major_surgery_counts += 1
                # put the injury in the injuries to be cast property to stop them being treated elsewhere
                df.loc[person_id, 'rt_injuries_for_major_surgery'].append('811')
            # Check if this fracture needs to be treated by amputation
            else:
                # self.foot_frac_amputation = True
                # for the time being, assume all amputations are major surgeries
                # update the treatment plan
                self.foot_frac_major_surg = True
                # update the number of major surgeries
                self.major_surgery_counts += 1
                # put the injury in the injuries to be cast property to stop them being treated elsewhere
                df.loc[person_id, 'rt_injuries_for_major_surgery'].append('811')

            # check that no more than one treatment plan has been chosen for the foot fracture
            treatment_options = [self.foot_frac_major_surg, self.foot_frac_minor_surg, self.foot_frac_amputation,
                                 self.foot_frac_cast]
            assert sum(treatment_options) <= 1, 'multiple treatment options assigned to treat this foot fracture'

        # Design treatment plan for tibia/fibula fractures
        # Four treatment options in use currently for treating tibia/fibula fractures:
        # 1) major surgery - open reduction internal fixation
        # 2) minor surgery - external fixation
        # 3) fracture cast
        # 4) skeletal traction - a 'heal with time' treatment
        # Set up treatment options
        self.tib_fib_frac_major_surg = False
        self.tib_fib_frac_minor_surg = False
        self.tib_fib_frac_amputation = False
        self.tib_fib_frac_traction = False
        self.tib_fib_frac_cast = False
        # Load the parameters used to determine treatment for tibia/fibula fractures
        prob_tib_fib_frac_require_cast = p['prob_tib_fib_frac_require_cast']
        prob_tib_fib_frac_require_maj_surg = p['prob_tib_fib_frac_require_maj_surg']
        prob_tib_fib_frac_require_min_surg = p['prob_tib_fib_frac_require_min_surg']
        prob_tib_fib_frac_require_traction = p['prob_tib_fib_frac_require_traction']
        # Check if the person has a broken tibia/fibula
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, ['812'])
        if counts > 0:
            # determine what the treatment for this person will be
            treatment_plan = self.module.rng.random_sample(size=1)
            # check if this tibia fracture can be treated with a cast
            if treatment_plan < prob_tib_fib_frac_require_cast:
                # update the treatment plan
                self.tib_fib_frac_cast = True
                # put the injury in the injuries to be cast property to stop them being treated elsewhere
                df.loc[person_id, 'rt_injuries_to_cast'].append('812')
            # Check if this fracture needs to be treated with external fixation (minor surgery)
            elif treatment_plan < prob_tib_fib_frac_require_cast + \
                prob_tib_fib_frac_require_min_surg:
                # update the treatment plan
                self.tib_fib_frac_minor_surg = True
                # put the injury in the injuries for minor surgery property to stop them being treated elsewhere
                df.loc[person_id, 'rt_injuries_for_minor_surgery'].append('812')
                # update the number of minor surgeries needed
                self.minor_surgery_counts += 1
            # Check if this fracture needs to be treated with open reduction internal fixation (major surg)
            elif treatment_plan < prob_tib_fib_frac_require_cast + prob_tib_fib_frac_require_min_surg + \
                prob_tib_fib_frac_require_maj_surg:
                # update the treatment plan
                self.tib_fib_frac_major_surg = True
                # put the injury in the injuries for major surgery property to stop them being treated elsewhere
                df.loc[person_id, 'rt_injuries_for_major_surgery'].append('812')
                # update the number of major surgeries needed
                self.major_surgery_counts += 1
            # Check if this fracture needs to be treated with traction
            elif treatment_plan < prob_tib_fib_frac_require_cast + prob_tib_fib_frac_require_min_surg + \
                prob_tib_fib_frac_require_maj_surg + prob_tib_fib_frac_require_traction:
                # update the treatment plan
                self.tib_fib_frac_traction = True
                # update the list of heal with time injuries
                self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, '812')
            else:
                # self.tib_fib_frac_amputation = True
                # self.tib_fib_frac_traction = True
                # self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, '812')
                # for now, just assume that all amputations are major surgeries
                self.tib_fib_frac_major_surg = True
                # put the injury in the injuries for major surgery property to stop them being treated elsewhere
                df.loc[person_id, 'rt_injuries_for_major_surgery'].append('812')
                # update the number of major surgeries needed
                self.major_surgery_counts += 1
            # make sure that there is only one treatment plan chosed to treat this injury
            treatment_options = [self.tib_fib_frac_major_surg, self.tib_fib_frac_minor_surg,
                                 self.tib_fib_frac_amputation, self.tib_fib_frac_traction, self.tib_fib_frac_cast]
            assert sum(treatment_options) <= 1, 'multiple treatment options assigned to treat this tib/fib fracture'

        # Design treatment plan for femur fractures / hip fractures
        # for femur/hip fractures there are currently four treatment options included in the model:
        # 1) skeletal traction
        # 2) minor surgery (external fixation)
        # 3) major surgery (open reduction internal fixation)
        # 4) fracture cast
        # in the future I hope to include amputations

        # Check if the person has a broken femur/hip/pelvis which will require a major/minor surgery or traction.
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, ['813a', '813c'])
        # create the treatment plan options
        self.femur_skeletal_traction = False
        self.femur_minor_surgery = False
        self.femur_major_surgery = False
        self.femur_cast = False
        self.femur_amputation = False
        # Load the parameters used to determine the treatment plan
        prob_femural_fracture_require_major_surgery = p['prob_femural_fracture_require_major_surgery']
        prob_femural_fracture_require_minor_surgery = p['prob_femural_fracture_require_minor_surgery']
        prob_femural_fracture_require_cast = p['prob_femural_fracture_require_cast']
        if counts > 0:
            # create a treatment plan
            treatment_plan = self.module.rng.random_sample(size=1)
            # work out if this injury is a femur fracture or a hip fracture
            actual_injury = np.intersect1d(['813a', '813c'], person_injuries.values)
            # check if femur fracture needs major surgery
            if treatment_plan < prob_femural_fracture_require_major_surgery:
                # update the number of major surgeries required
                self.major_surgery_counts += 1
                # update the treatment plan
                self.femur_major_surgery = True
                # store the injury in the major surgeries property so it won't be treated elsewhere
                df.loc[person_id, 'rt_injuries_for_major_surgery'].append(actual_injury[0])
            # check if femur fracture needs minor surgery
            elif treatment_plan < prob_femural_fracture_require_major_surgery + \
                prob_femural_fracture_require_minor_surgery:
                # update the number of minor surgeries required
                self.minor_surgery_counts += 1
                # update the treatment plan
                self.femur_minor_surgery = True
                # store the injury in the minor surgeries property so it won't be treated elsewhere
                df.loc[person_id, 'rt_injuries_for_minor_surgery'].append(actual_injury[0])
            # check if femur fracture needs casting
            elif treatment_plan < prob_femural_fracture_require_major_surgery + \
                prob_femural_fracture_require_minor_surgery + prob_femural_fracture_require_cast:
                # update the treatment plan
                self.femur_cast = True
                # put the injury in the injuries to be cast property to stop them being treated elsewhere
                df.loc[person_id, 'rt_injuries_to_cast'].append(actual_injury[0])
            # Check if femur fracture it treated using skeletal traction
            else:
                # store the injury in the heal with time injuries
                self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, actual_injury)
                self.femur_skeletal_traction = True
            # check that only one treatment option has been chosed for this femur/hip fracture
            treatment_options = [self.femur_skeletal_traction, self.femur_minor_surgery, self.femur_major_surgery,
                                 self.femur_cast, self.femur_amputation]
            assert sum(treatment_options) <= 1, 'multiple treatment options assigned to treat this femur/hip fracture'

        # Design the treatment plan for pelvis fractures
        # Three treatment options in use currently for treating pelvis fractures:
        # 1) major surgery - open reduction internal fixation
        # 2) minor surgery - external fixation
        # 3) fracture cast
        # 4) skeletal traction
        # set up the treatment options
        self.pelvis_skeletal_traction = False
        self.pelvis_major_surgery = False
        self.pelvis_minor_surgery = False
        self.pelvis_fracture_cast = False
        # Load the parameters used to determine pelvis fracture treatment plans
        prob_pelvis_fracture_traction = p['prob_pelvis_fracture_traction']
        prob_pelvis_frac_major_surgery = p['prob_pelvis_frac_major_surgery']
        prob_pelvis_frac_minor_surgery = p['prob_pelvis_frac_minor_surgery']

        # See if this person has a pelvis fracture
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, ['813b'])
        if counts > 0:
            # create a treatment plan option
            treatment_plan = self.module.rng.random_sample(size=1)
            # Determine if this pelvis fracture will be treated with skeletal traction
            if treatment_plan < prob_pelvis_fracture_traction:
                # add the pelvis fracture to the heal with time injuries
                self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, '813b')
                # update the treatment plan
                self.pelvis_skeletal_traction = True
            # Determine if this pelvis fracture will be treated with a minor surgery
            elif treatment_plan < prob_pelvis_fracture_traction + prob_pelvis_frac_minor_surgery:
                # update the number of minor surgeries required
                self.minor_surgery_counts += 1
                # update the treatment plan
                self.pelvis_minor_surgery = True
                # add the injury to the injuries to be treated with minor surgeries so they aren't treated elsewhere
                df.loc[person_id, 'rt_injuries_for_minor_surgery'].append('813b')
            # Determine if the pelvis fracture will be treated with major surgery
            elif treatment_plan < prob_pelvis_fracture_traction + prob_pelvis_frac_minor_surgery + \
                prob_pelvis_frac_major_surgery:
                # update the number of major surgeries required
                self.major_surgery_counts += 1
                # update the treatment plan
                self.pelvis_major_surgery = True
                # add the injury to the injuries to be treated by major surgery so it isn't treated elsewhere
                df.loc[person_id, 'rt_injuries_for_major_surgery'].append('813b')
            # Determine if the injury will be treated with a cast
            else:
                # update the treatment plan
                self.pelvis_fracture_cast = True
                # add the injury to be treated by fracture cast so it isn't treated elsewhere
                df.loc[person_id, 'rt_injuries_to_cast'].append('813b')
            # Make sure that only one treatment plant for the pelvis fracture has been chosen
            treatment_options = [self.pelvis_skeletal_traction, self.pelvis_major_surgery, self.pelvis_minor_surgery,
                                 self.pelvis_fracture_cast]
            assert sum(treatment_options) <= 1, 'multiple treatment options assigned to treat this femur/hip fracture'
        # -------------------------------------- Open fractures -------------------------------------------------------
        self.open_fractures = 0
        open_fracture_codes = ['813bo', '813co', '813do', '813eo']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, open_fracture_codes)
        if len(idx) > 0:
            # find the exact injury
            actual_injury = np.intersect1d(open_fracture_codes, person_injuries.values)
            # update the number of open fracture treatments needed
            self.open_fractures += counts
            # add the injury to the injuries to be treated by major surgery so they aren't treated elsewhere
            df.loc[person_id, 'rt_injuries_for_open_fracture_treatment'].append(actual_injury[0])
        # ------------------------------ Traumatic brain injury requirements ------------------------------------------
        # Check whether the person has a severe traumatic brain injury, which in some cases will require a major surgery
        # to treat
        codes = ['133', '133a', '133b', '133c', '133d' '134', '134a', '134b', '135']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        # create variable to determine treatment plan
        require_surgery = self.module.rng.random_sample(size=1)
        if counts > 0:
            # find the exact injury
            actual_injury = np.intersect1d(codes, person_injuries.values)
            # determine if the tbi will be treated with a surgery, or it whether it will heal with time
            if require_surgery < self.prob_TBI_require_craniotomy:
                # update the number of major surgeries needed
                self.major_surgery_counts += 1
                # add the injury to the injuries to be treated by major surgery so they aren't treated elsewhere
                df.loc[person_id, 'rt_injuries_for_major_surgery'].append(actual_injury[0])
            else:
                self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, actual_injury)
        # ------------------------------ Abdominal organ injury requirements ------------------------------------------
        # Check if the person has any abodominal organ injuries, if they do, determine whether they require a surgery or
        # not
        codes = ['552', '553', '554']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        # create variable to determine treatment plan
        require_surgery = self.module.rng.random_sample(size=1)
        if counts > 0:
            # find the exact injury
            actual_injury = np.intersect1d(codes, person_injuries.values)
            # Check if abdominal injury will require surgery, otherwise will heal over time
            if require_surgery < self.prob_exploratory_laparotomy:
                # update the number of major surgeries needed
                self.major_surgery_counts += 1
                df.loc[person_id, 'rt_injuries_for_major_surgery'].append(actual_injury[0])
            else:
                actual_injury = np.intersect1d(codes, person_injuries.values)
                self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, actual_injury)

        # -------------------------------- Spinal cord injury requirements --------------------------------------------
        # Check whether they have a spinal cord injury, if we allow spinal cord surgery capacilities here, ask for a
        # surgery, otherwise make the injury permanent
        codes = ['673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676']
        # Ask if this person has a spinal cord injury
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if (counts > 0) & ('include_spine_surgery' in self.allowed_interventions):
            # if this person has a spinal cord injury and we allow surgeries, determine their exact injury
            actual_injury = np.intersect1d(codes, person_injuries.values)
            # update the number of major surgeries
            self.major_surgery_counts += 1
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
        # check if the person has a vertebrae fracture
        codes = ['612']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            # add the injury to the heal with time injuries
            self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, '612')
        # --------------------------------- Dislocations --------------------------------------------------------------
        # Check if they have a dislocation, will require surgery but otherwise they can be taken care of in the RTI med
        # app
        # Create treatment plan for dislocated hip, there are currently three options:
        # 1) major surgery - open reduction internal fixation
        # 2) casting
        # 3) skeletal traction
        # Set up the treatment options
        self.hip_dis_require_major_surg = False
        self.hip_dis_require_cast = False
        self.hip_dis_require_traction = False
        # load the parameters used to determine the treatment plan
        prob_dis_hip_require_maj_surg = p['prob_dis_hip_require_maj_surg']
        prob_hip_dis_require_traction = p['prob_hip_dis_require_traction']
        # See if person has dislocated hip
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, ['822a'])
        if counts > 0:
            # Set up the treatment plan
            treatment_plan = self.module.rng.random_sample(size=1)
            # See if this person is treated with skeletal traction
            if treatment_plan < prob_hip_dis_require_traction:
                # Add the injury to the heal with time injuries
                self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, '822a')
                # update the treatment plan
                self.hip_dis_require_traction = True
            # See if this person is treated with a major surgery
            elif treatment_plan < prob_hip_dis_require_traction + prob_dis_hip_require_maj_surg:
                # update the number of major surgeries needed
                self.major_surgery_counts += 1
                # update the treatment plan
                self.hip_dis_require_major_surg = True
                # add the injury to the injuries that need to be treated with major surgeries
                df.loc[person_id, 'rt_injuries_for_major_surgery'].append('822a')
            # Determine if the injury will be treated with a cast
            else:
                # Update the treatment plan
                self.hip_dis_require_cast = True
                # add the injury to this injuries that need to be cast
                df.loc[person_id, 'rt_injuries_to_cast'].append('822a')
            # Make sure only one treatment plan has been chosen
            treatment_options = [self.hip_dis_require_major_surg, self.hip_dis_require_cast,
                                 self.hip_dis_require_traction]
            assert sum(treatment_options) <= 1, 'person had multiple treatment options for hip dislocation assigned'
        # Knee dislocations will be treated by casting
        # Determine the treatment plans for dislocated necks ('322', '323') and dislocated shoulders ('722')
        codes = ['322', '323', '722']
        # check if they have a neck or shoulder dislocation
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            # determine if they will require surgery
            require_surgery = self.module.rng.random_sample(size=1)
            # determine the exact injury they have
            actual_injury = np.intersect1d(codes, person_injuries.values)
            # Check if they require a minor surgery
            if require_surgery < self.prob_dislocation_requires_surgery:
                # update the number of minor surgeries required
                self.minor_surgery_counts += 1
                # add the injury to the injuries to be treated in minor surgery so they aren't treated elsewhere
                df.loc[person_id, 'rt_injuries_for_minor_surgery'].append(actual_injury[0])
            else:
                # if the injury isn't treated in surgery, add the injury to the heal with time injuries
                self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, actual_injury)
        # --------------------------------- Soft tissue injury in neck -------------------------------------------------
        # check whether they have a soft tissue/internal bleeding injury in the neck. If so schedule a surgery.
        codes = ['342', '343']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            # find the exact injury they have
            actual_injury = np.intersect1d(codes, person_injuries.values)
            # update the number of major surgeries required.
            self.major_surgery_counts += 1
            # add these injuries to the injuries to be treated by major surgery so they aren't treated elsewhere
            df.loc[person_id, 'rt_injuries_for_major_surgery'].append(actual_injury[0])
        # --------------------------------- Soft tissue injury in thorax/ lung injury ----------------------------------
        # Check whether they have any soft tissue injuries in the thorax, if so schedule surgery if required else make
        # the injuries heal over time without further medical care
        codes = ['441', '443', '453', '453a', '453b']
        # check if they have chest traume
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if (counts > 0) & ('include_thoroscopy' in self.allowed_interventions):
            # work out the exact injury they have
            actual_injury = np.intersect1d(codes, person_injuries.values)
            # update the number of major surgeries required
            self.major_surgery_counts += 1
            # add the injury to the injuries to be treated with major surgery so they aren't treated elsewhere
            df.loc[person_id, 'rt_injuries_for_major_surgery'].append(actual_injury[0])
        # check if they have chest trauma which is just air in the chest cavity which resolves on its own
        codes = ['442']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            # add the injury to the heal with time injuries
            self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, '442')
        # -------------------------------- Internal bleeding -----------------------------------------------------------
        # Check if they have any internal bleeding in the neck, if so schedule a major surgery
        codes = ['361', '363']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            # Work out what the injury is
            actual_injury = np.intersect1d(codes, person_injuries.values)
            # update the number of major surgeries required
            self.major_surgery_counts += 1
            # add the injury to the injuries to be treated with major surgery so it isn't treated elsewhere
            df.loc[person_id, 'rt_injuries_for_major_surgery'].append(actual_injury[0])
        # check if they have internal bleeding in the thorax, and if the surgery is available, schedule a major surgery
        codes = ['463']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if (counts > 0) & ('include_thoroscopy' in self.allowed_interventions):
            # update the number of major surgeries needed
            self.major_surgery_counts += 1
            # add the injury to the injuries to be treated with major surgery.
            df.loc[person_id, 'rt_injuries_for_major_surgery'].append('463')
        # check if this person has minor internal bleeding which will heal without intervention
        codes = ['461']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            # add the injury to the heal with time injuries
            self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, '461')
        # ------------------------------------- Amputations ------------------------------------------------------------
        # Check if they have or need an amputation in the upper extremities
        codes = ['782', '782a', '782b', '782c', '783']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            # determine the injuries the person has
            actual_injury = np.intersect1d(codes, person_injuries.values)
            self.major_surgery_counts += 1
            df.loc[person_id, 'rt_injuries_for_major_surgery'].append(actual_injury[0])
        # Check if they have or need an amputation in the lower extremities
        codes = ['882', '883', '884']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            # determine the injuries the person has
            actual_injury = np.intersect1d(codes, person_injuries.values)
            self.major_surgery_counts += 1
            df.loc[person_id, 'rt_injuries_for_major_surgery'].append(actual_injury[0])
        # --------------------------------------- Eye injury -----------------------------------------------------------
        # check if they have an eye injury and schedule a minor surgery if so schedule a minor surgery
        codes = ['291']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            # update the number of minor surgeries needed
            self.minor_surgery_counts += 1
            # add the injury to the list of injuries treated by minor surgeries
            df.loc[person_id, 'rt_injuries_for_minor_surgery'].append('291')

        # ------------------------------ Soft tissue injury in face ----------------------------------------------------
        # check if they have any facial soft tissue damage and schedule a minor surgery if so.
        codes = ['241']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            # update the number of minor surgeries needed
            self.minor_surgery_counts += 1
            # add the injury to the list of injuries treated by minor surgeries
            df.loc[person_id, 'rt_injuries_for_minor_surgery'].append('241')
        # store the heal with time injuries in the dataframe
        for injury in self.heal_with_time_injuries:
            df.loc[person_id, 'rt_injuries_to_heal_with_time'].append(injury)

        # Store the fractures that need treatment from casts
        codes = ['712a', '712b', '712c', '822b']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            # determine the injuries the person has
            actual_injury = np.intersect1d(codes, person_injuries.values)
            df.loc[person_id, 'rt_injuries_to_cast'].append(actual_injury[0])
        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'RTI_MedicalIntervention'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []

        # ================ Determine how long the person will be in hospital based on their ISS score ==================
        self.inpatient_days = road_traffic_injuries.rti_determine_LOS(person_id)
        # If the patient needs skeletal traction for their injuries they need to stay at minimum 6 weeks,
        # average length of stay for those with femur skeletal traction found from Kramer et al. 2016:
        # https://doi.org/10.1007/s00264-015-3081-3
        # todo: put in complications from femur fractures
        self.femur_fracture_skeletal_traction_mean_los = p['femur_fracture_skeletal_traction_mean_los']
        self.other_skeletal_traction_los = p['other_skeletal_traction_los']
        if self.femur_skeletal_traction & (self.inpatient_days < self.femur_fracture_skeletal_traction_mean_los):
            self.inpatient_days = self.femur_fracture_skeletal_traction_mean_los
        if self.pelvis_skeletal_traction & (self.inpatient_days < self.other_skeletal_traction_los):
            self.inpatient_days = self.other_skeletal_traction_los
        if self.tib_fib_frac_traction & (self.inpatient_days < self.other_skeletal_traction_los):
            self.inpatient_days = self.other_skeletal_traction_los
        if self.hip_dis_require_traction & (self.inpatient_days < self.other_skeletal_traction_los):
            self.inpatient_days = self.other_skeletal_traction_los
        # Specify the type of bed days needed? not sure if necessary
        self.BEDDAYS_FOOTPRINT.update({'general_bed': self.inpatient_days})
        # update the expected appointment foortprint
        self.EXPECTED_APPT_FOOTPRINT.update({'InpatientDays': self.inpatient_days})
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
            idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
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
            self.BEDDAYS_FOOTPRINT.update({'high_dependency_bed': self.icu_days})
            # store the injury information of patients in ICU
            logger.info(key='ICU_patients',
                        data=person_injuries,
                        description='The injuries of ICU patients')

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

    def apply(self, person_id, squeeze_factor):
        road_traffic_injuries = self.sim.modules['RTI']
        df = self.sim.population.props
        # Remove the scheduled death without medical intervention
        df.loc[person_id, 'rt_date_death_no_med'] = pd.NaT
        # Isolate relevant injury information
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        non_empty_injuries = person_injuries[person_injuries != "none"]
        injury_columns = non_empty_injuries.columns
        # Check that those who arrive here are alive and have been through the first generic appointment, and didn't
        # die due to rti
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'
        assert df.loc[person_id, 'rt_diagnosed'], 'person sent here has not been through A and E'
        # Check that those who arrive here have at least one injury
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries,
                                                      self.module.PROPERTIES.get('rt_injury_1').categories[1:-1])
        assert counts > 0, 'This person has asked for medical treatment despite not being injured'
        # update the model's properties to reflect that this person has sought medical care
        df.at[person_id, 'rt_med_int'] = True
        # =============================== Make 'healed with time' injuries disappear ===================================
        if len(self.heal_with_time_injuries) > 0:
            # check whether the heal with time injuries include dislocations, which may have been sent to surgery
            heal_with_time_codes = []
            dislocations = ['322', '323', '722', '822', '822a']
            dislocations_injury = [injury for injury in dislocations if injury in self.heal_with_time_injuries]
            if len(dislocations_injury) > 0:
                for code in dislocations_injury:
                    heal_with_time_codes.append(code)
                    # if the heal with time injury is a dislocation, schedule a recovery date
                    if code == '322' or code == '323':
                        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                                      [code])[0])
                        # using estimated 6 weeks to recover from dislocated neck
                        df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=6)
                    elif code == '722':
                        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                                      [code])[0])
                        # using estimated 12 weeks to recover from dislocated shoulder
                        df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=12)
                    elif code == '822a':
                        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                                      [code])[0])
                        # using estimated 2 months to recover from dislocated hip
                        df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(months=2)

            # Check whether the heal with time injury is a skull fracture, which may have been sent to surgery
            fractures = ['112', '113', '412', '612', '812', '813a', '813b', '813c']
            fractures_injury = [injury for injury in fractures if injury in self.heal_with_time_injuries]
            if len(fractures_injury) > 0:
                for code in fractures_injury:
                    heal_with_time_codes.append(code)
                    if code == '112' or code == '113':
                        # schedule a recovery date for the skull fracture
                        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                                      [code])[0]
                                                         )
                        # using estimated 7 weeks PLACEHOLDER FOR SKULL FRACTURE
                        df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=7)
                    if code == '412':
                        # schedule a recovery date for the rib fracture
                        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                                      [code])[0]
                                                         )
                        # using estimated 5 weeks PLACEHOLDER FOR rib FRACTURE
                        df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=5)
                    if code == '612':
                        # schedule a recovery date for the vertebrae fracture
                        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                                      [code])[0]
                                                         )
                        # using estimated 9 weeks PLACEHOLDER FOR Vertebrae FRACTURE
                        df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                            weeks=9)
                    if (code == '812') & self.tib_fib_frac_traction:
                        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                                      [code])[0]
                                                         )
                        # using estimated 9 weeks PLACEHOLDER FOR skeletal traction for hip
                        df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                            weeks=9)
                    if (code == '813a') & self.femur_skeletal_traction:
                        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                                      [code])[0]
                                                         )
                        # using estimated 9 weeks PLACEHOLDER FOR skeletal traction for hip
                        df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                            weeks=9)
                    if (code == '813b') & self.pelvis_skeletal_traction:
                        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                                      [code])[0]
                                                         )
                        # using estimated 9 weeks PLACEHOLDER FOR skeletal traction for pelvis
                        df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                            weeks=9)
                    if (code == '813c') & self.femur_skeletal_traction:
                        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                                      [code])[0]
                                                         )
                        # using estimated 9 weeks PLACEHOLDER FOR skeletal traction for femur
                        df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(
                            weeks=9)

            abdominal = ['552', '553', '554']
            abdominal_injury = [injury for injury in abdominal if injury in self.heal_with_time_injuries]
            # check whether the heal with time injury is an abdominal injury
            if len(abdominal_injury) > 0:
                for code in abdominal_injury:
                    heal_with_time_codes.append(code)
                    if code == '552' or code == '553' or code == '554':
                        # Schedule the recovery date for the injury
                        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                                      [code])[0]
                                                         )
                        # using estimated 3 months PLACEHOLDER FOR ABDOMINAL TRAUMA
                        df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(months=3)
            tbi = ['133', '133a', '133b', '133c', '133d' '134', '134a', '134b', '135']
            tbi_injury = [injury for injury in tbi if injury in self.heal_with_time_injuries]
            if len(tbi_injury) > 0:
                columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, tbi_injury)
                                                 [0])
                # ask if this injury will be permanent
                perm_injury = self.module.rng.random_sample(size=1)
                if perm_injury < self.prob_perm_disability_with_treatment_severe_TBI:
                    column, code = road_traffic_injuries.rti_find_injury_column(person_id=person_id, codes=tbi_injury)
                    df.loc[person_id, column] = "P" + code
                    heal_with_time_codes.append("P" + code)
                else:
                    heal_with_time_codes.append(tbi_injury[0])
                    # using estimated 3 months PLACEHOLDER FOR ABDOMINAL TRAUMA
                    df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(months=3)

                # using estimated 6 months PLACEHOLDER FOR TRAUMATIC BRAIN INJURY
                df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(months=6)
            surgical_emphysema = ['442']
            empysema_injury = [injury for injury in surgical_emphysema if injury in self.heal_with_time_injuries]
            if len(empysema_injury) > 0:
                heal_with_time_codes.append(surgical_emphysema[0])
                columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                              empysema_injury)[0])
                # use a 1 week placeholder for surgical emphysema
                df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=1)

            int_bleeding = ['461']
            int_b_injury = [injury for injury in int_bleeding if injury in self.heal_with_time_injuries]
            if len(int_b_injury) > 0:
                heal_with_time_codes.append(int_b_injury[0])
                columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, int_b_injury)
                                                 [0])
                # use a 2 week placeholder for chest wall bruising
                df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=2)

            # swap potentially swappable codes
            swapping_codes = ['712b', '812', '3113', '4113', '5113', '7113', '8113', '813a', '813b', 'P673a',
                              'P673b', 'P674a', 'P674b', 'P675a', 'P675b', 'P676', 'P782b', 'P783', 'P883', 'P884',
                              '813bo', '813co', '813do', '813eo']
            # remove codes that will be treated elsewhere
            for code in df.loc[person_id, 'rt_injuries_for_minor_surgery']:
                if code in swapping_codes:
                    swapping_codes.remove(code)
            for code in df.loc[person_id, 'rt_injuries_for_major_surgery']:
                if code in swapping_codes:
                    swapping_codes.remove(code)
            for code in df.loc[person_id, 'rt_injuries_to_cast']:
                if code in swapping_codes:
                    swapping_codes.remove(code)
            for code in df.loc[person_id, 'rt_injuries_for_open_fracture_treatment']:
                if code in swapping_codes:
                    swapping_codes.remove(code)
            # drop injuries potentially treated elsewhere
            codes_to_swap = [code for code in heal_with_time_codes if code in swapping_codes]
            if len(codes_to_swap) > 0:
                road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, codes_to_swap)
            # check every heal with time injury has a recovery date associated with it
            for code in self.heal_with_time_injuries:
                columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, [code])
                                                 [0])
                assert not pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]), \
                    'no recovery date given for this injury'
                # remove code from heal with time injury list
                if code in df.loc[person_id, 'rt_injuries_to_heal_with_time']:
                    df.loc[person_id, 'rt_injuries_to_heal_with_time'].remove(code)

        # ======================================= Schedule surgeries ==================================================
        # Schedule the surgeries by calling the functions rti_do_for_major/minor_surgeries which in turn schedules the
        # surgeries
        # Check they haven't died from another source
        if df.loc[person_id, 'cause_of_death'] != '':
            pass
        else:
            if self.major_surgery_counts > 0:
                # schedule major surgeries
                for count in range(0, self.major_surgery_counts):
                    road_traffic_injuries.rti_do_for_major_surgeries(person_id=person_id, count=count)
            if self.minor_surgery_counts > 0:
                # shedule minor surgeries
                for count in range(0, self.minor_surgery_counts):
                    road_traffic_injuries.rti_do_for_minor_surgeries(person_id=person_id, count=count)

        # --------------------------- Lacerations will get stitches here -----------------------------------------------
        # Schedule the laceration sutures by calling the functions rti_ask_for_stitches which in turn schedules the
        # treatment
        codes = ['1101', '2101', '3101', '4101', '5101', '7101', '8101']
        idx, lacerationcounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if lacerationcounts > 0 & (df.loc[person_id, 'cause_of_death'] == ''):
            # Schedule laceration treatment
            road_traffic_injuries.rti_ask_for_suture_kit(person_id=person_id)

        # =================================== Burns consumables =======================================================
        # Schedule the burn treatments  by calling the functions rti_ask_for_burn_treatment which in turn schedules the
        # treatment
        codes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113']
        idx, burncounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)

        if burncounts > 0 & (df.loc[person_id, 'cause_of_death'] != ''):
            # schedule burn treatment
            road_traffic_injuries.rti_ask_for_burn_treatment(person_id=person_id)

        # ==================================== Fractures ==============================================================
        # ------------------------------ Cast-able fractures ----------------------------------------------------------
        # Schedule the fracture treatments by calling the functions rti_ask_for_fracture_casts which in turn schedules
        # the treatment
        codes = ['712', '712a', '712b', '712c', '822a', '822b']
        treated_injury_cols = []
        idx_for_treated_injuries = []
        for index, time in enumerate(df.loc[person_id, 'rt_date_to_remove_daly']):
            if ~pd.isnull(time):
                idx_for_treated_injuries.append(index)
        for idx in idx_for_treated_injuries:
            treated_injury_cols.append(cols[idx])
        person_treated_injuries = df.loc[[person_id], treated_injury_cols]

        codes = [code for code in codes if
                 road_traffic_injuries.rti_find_and_count_injuries(person_treated_injuries, code)[1] > 1]
        for injury in df.loc[person_id, 'rt_injuries_to_cast']:
            codes.append(injury)
        idx, fracturecounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if fracturecounts > 0 & (df.loc[person_id, 'cause_of_death'] == ''):
            road_traffic_injuries.rti_ask_for_fracture_casts(person_id=person_id)
        # ------------------------------------ Open fractures ---------------------------------------------------------
        if self.open_fractures > 0 & (df.loc[person_id, 'cause_of_death'] == ''):
            road_traffic_injuries.rti_ask_for_open_fracture_treatment(person_id=person_id, counts=self.open_fractures)
        # ============================== Generic injury management =====================================================

        # ================================= Pain management ============================================================
        # Most injuries will require some level of pain relief, we need to determine:
        # 1) What drug the person will require
        # 2) What to do if the drug they are after isn't available
        # 3) Whether to run the event even if the drugs aren't available
        # Determine whether this person dies with medical treatment or not with the RTIMediaclInterventionDeathEvent

        # Check that the person hasn't died from another source
        if df.loc[person_id, 'cause_of_death'] != '':
            pass
        else:
            road_traffic_injuries.rti_acute_pain_management(person_id=person_id)
        # ==================================== Tetanus management ======================================================
        # Check if they have had a laceration or a burn, if so request a tetanus jab
        codes_for_tetanus = ['1101', '2101', '3101', '4101', '5101', '7101', '8101',
                             '1114', '2114', '3113', '4113', '5113', '7113', '8113']

        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes_for_tetanus)
        if counts > 0 & (df.loc[person_id, 'cause_of_death'] == ''):
            road_traffic_injuries.rti_ask_for_tetanus(person_id=person_id)
        # ============================== Ask if they die even with treatment ===========================================
        self.sim.schedule_event(RTI_Medical_Intervention_Death_Event(self.module, person_id), self.sim.date +
                                DateOffset(days=self.inpatient_days))
        logger.debug('This is RTIMedicalInterventionEvent scheduling a potential death on date %s (end of treatment)'
                     ' for person %d', self.sim.date + DateOffset(days=self.inpatient_days), person_id)

    def did_not_run(self):
        person_id = self.target
        df = self.sim.population.props
        logger.debug('RTI_MedicalInterventionEvent: did not run for person  %d on date %s',
                     person_id, self.sim.date)
        injurycodes = {'First injury': df.loc[person_id, 'rt_injury_1'],
                       'Second injury': df.loc[person_id, 'rt_injury_2'],
                       'Third injury': df.loc[person_id, 'rt_injury_3'],
                       'Fourth injury': df.loc[person_id, 'rt_injury_4'],
                       'Fifth injury': df.loc[person_id, 'rt_injury_5'],
                       'Sixth injury': df.loc[person_id, 'rt_injury_6'],
                       'Seventh injury': df.loc[person_id, 'rt_injury_7'],
                       'Eight injury': df.loc[person_id, 'rt_injury_8']}
        logger.debug(f'Injury profile of person %d, {injurycodes}', person_id)


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
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # Placeholder requirement
        the_accepted_facility_level = 1
        self.TREATMENT_ID = 'RTI_Fracture_Cast'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        road_traffic_injuries = self.sim.modules['RTI']
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        # isolate the relevant injury information
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        # Find the untreated injuries
        idx_for_untreated_injuries = []
        untreated_injury_cols = []
        for index, time in enumerate(df.loc[person_id, 'rt_date_to_remove_daly']):
            if pd.isnull(time):
                idx_for_untreated_injuries.append(index)
        for idx in idx_for_untreated_injuries:
            untreated_injury_cols.append(cols[idx])
        person_injuries = df.loc[[person_id], untreated_injury_cols]

        # check if they have a fracture that requires a cast
        codes = ['712b', '712c', '811', '812', '813a', '813b', '813c', '822a', '822b']
        idx, fracturecastcounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        # check if they have a fracture that requires a sling
        codes = ['712a']
        idx, slingcounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        consumables_fractures = {'Intervention_Package_Code': dict(), 'Item_Code': dict()}
        # Check the person sent here is alive, been through the generic first appointment,
        # been through the RTI med intervention
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'
        assert df.loc[person_id, 'rt_diagnosed'], 'person sent here has not been diagnosed'
        assert df.loc[person_id, 'rt_med_int'], 'person sent here has not been treated'
        # Check that the person sent here has an injury treated by this module
        # assert fracturecastcounts + slingcounts > 0
        assert len(df.loc[person_id, 'rt_injuries_to_cast']) > 0
        # If they have a fracture that needs a cast, ask for plaster of paris
        if fracturecastcounts > 0:
            plaster_of_paris_code = pd.unique(
                consumables.loc[consumables['Items'] ==
                                'Plaster of Paris (POP) 10cm x 7.5cm slab_12_CMST', 'Item_Code'])[0]
            consumables_fractures['Item_Code'].update({plaster_of_paris_code: fracturecastcounts})
        # If they have a fracture that needs a sling, ask for bandage.

        if slingcounts > 0:
            sling_code = pd.unique(
                consumables.loc[consumables['Items'] ==
                                'Bandage, crepe 7.5cm x 1.4m long , when stretched', 'Item_Code'])[0]
            consumables_fractures['Item_Code'].update({sling_code: slingcounts})
        # Check that there are enough consumables to treat this person's fractures
        is_cons_available = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self,
            cons_req_as_footprint=consumables_fractures,
            to_log=True)
        if is_cons_available:
            logger.debug(f"Fracture casts available for person %d's {fracturecastcounts + slingcounts} fractures",
                         person_id)
            df.at[person_id, 'rt_med_int'] = True
            # Find the persons injuries
            non_empty_injuries = person_injuries[person_injuries != "none"]
            non_empty_injuries = non_empty_injuries.dropna(axis=1)
            # Find the injury codes treated by fracture casts/slings
            codes = ['712a', '712b', '712c', '811', '812', '813a', '813b', '813c', '822a', '822b']
            # Some TLO codes have daly weights associated with treated and non-treated injuries, swap-able codes are
            # listed below
            swapping_codes = ['712b', '812', '3113', '4113', '5113', '7113', '8113', '813a', '813b', 'P673a',
                              'P673b', 'P674a', 'P674b', 'P675a', 'P675b', 'P676', 'P782b', 'P783', 'P883', 'P884',
                              '813bo', '813co', '813do', '813eo']
            swapping_codes = [code for code in swapping_codes if code in codes]
            # remove codes that will be treated elsewhere
            for code in df.loc[person_id, 'rt_injuries_for_minor_surgery']:
                if code in swapping_codes:
                    swapping_codes.remove(code)
            for code in df.loc[person_id, 'rt_injuries_for_major_surgery']:
                if code in swapping_codes:
                    swapping_codes.remove(code)
            for code in df.loc[person_id, 'rt_injuries_to_heal_with_time']:
                if code in swapping_codes:
                    swapping_codes.remove(code)
            for code in df.loc[person_id, 'rt_injuries_for_open_fracture_treatment']:
                if code in swapping_codes:
                    swapping_codes.remove(code)
            relevant_codes = np.intersect1d(non_empty_injuries.values, swapping_codes)
            if len(relevant_codes) > 0:
                road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, relevant_codes)

            # Find the injuries that have been treated and then schedule a recovery date
            columns, codes = \
                road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id, df.loc[person_id,
                                                                                                 'rt_injuries_to_cast'])
            assert len(columns) == len(df.loc[person_id, 'rt_injuries_to_cast'])
            for col in columns:
                # todo: update this with recovery times for casted broken hips/pelvis/femurs
                # todo: update this with recovery times for casted dislocated hip
                df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = self.sim.date + \
                                                                                DateOffset(weeks=7)
            person_injuries = df.loc[person_id, cols]
            non_empty_injuries = person_injuries[person_injuries != "none"]
            injury_columns = non_empty_injuries.keys()

            for code in df.loc[person_id, 'rt_injuries_to_cast']:
                columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, [code])[0])
                assert not pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]), \
                    'no recovery date given for this injury'
                # remove code from fracture cast list
                if code in df.loc[person_id, 'rt_injuries_to_cast']:
                    df.loc[person_id, 'rt_injuries_to_cast'].remove(code)
        else:
            logger.debug(f"Person %d's has {fracturecastcounts + slingcounts} fractures without treatment",
                         person_id)

    def did_not_run(self, person_id):
        logger.debug('Fracture casts unavailable for person %d', person_id)


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
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # Placeholder requirement
        the_appt_footprint['MinorSurg'] = 1  # wound debridement requires minor surgery
        the_accepted_facility_level = 1
        self.TREATMENT_ID = 'RTI_Open_Fracture_Treatment'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        road_traffic_injuries = self.sim.modules['RTI']
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        # isolate the relevant injury information
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        # check if they have a fracture that requires a cast
        codes = ['813bo', '813co', '813do', '813eo']
        idx, open_fracture_counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)

        consumables_fractures = {'Intervention_Package_Code': dict(), 'Item_Code': dict()}
        # Check the person sent here is alive, been through the generic first appointment,
        # been through the RTI med intervention
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'
        assert df.loc[person_id, 'rt_diagnosed'], 'person sent here has not been diagnosed'
        assert df.loc[person_id, 'rt_med_int'], 'person sent here has not been treated'
        # Check that the person sent here has an injury treated by this module
        assert open_fracture_counts > 0
        # If they have a fracture that needs a cast, ask for plaster of paris
        if open_fracture_counts > 0:
            # Ask for ceftriaxon antibiotics as first choice.
            first_choice_antibiotic_code = pd.unique(
                consumables.loc[consumables['Items'] ==
                                'ceftriaxon 500 mg, powder for injection_10_IDA',
                                'Item_Code'])[0]
            consumables_fractures['Item_Code'].update({first_choice_antibiotic_code: 1})
            # Ask for sterilized gauze
            item_code_cetrimide_chlorhexidine = pd.unique(
                consumables.loc[consumables['Items'] ==
                                'Cetrimide 15% + chlorhexidine 1.5% solution.for dilution _5_CMST', 'Item_Code'])[0]
            consumables_fractures['Item_Code'].update({item_code_cetrimide_chlorhexidine: 1})
            item_code_gauze = pd.unique(
                consumables.loc[
                    consumables['Items'] == "Dressing, paraffin gauze 9.5cm x 9.5cm (square)_packof 36_CMST",
                    'Item_Code'])[0]
            consumables_fractures['Item_Code'].update({item_code_gauze: 1})
            # Ask for suture kit
            item_code_suture_kit = pd.unique(
                consumables.loc[consumables['Items'] == 'Suture pack', 'Item_Code'])[0]
            consumables_fractures['Item_Code'].update({item_code_suture_kit: 1})
            # If wound is "grossly contaminated" administer Metronidazole
            # todo: parameterise the probability of wound contamination
            p = self.module.parameters
            prob_open_fracture_contaminated = p['prob_open_fracture_contaminated']
            rand_for_contamination = self.module.rng.random_sample(size=1)
            if rand_for_contamination < prob_open_fracture_contaminated:
                conaminated_wound_metronidazole_code = pd.unique(
                    consumables.loc[consumables['Items'] ==
                                    'Metronidazole, injection, 500 mg in 100 ml vial',
                                    'Item_Code'])[0]
                consumables_fractures['Item_Code'].update({conaminated_wound_metronidazole_code: 1})

        # Check that there are enough consumables to treat this person's fractures
        is_cons_available = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self,
            cons_req_as_footprint=consumables_fractures,
            to_log=True)
        if is_cons_available:
            logger.debug(f"Fracture casts available for person %d's {open_fracture_counts} open fractures",
                         person_id)
            df.at[person_id, 'rt_med_int'] = True
            # Find the persons injuries to be treated
            non_empty_injuries = df.loc[person_id, 'rt_injuries_for_open_fracture_treatment']

            # Find the injury codes treated by fracture casts/slings
            swapping_codes = ['712b', '812', '3113', '4113', '5113', '7113', '8113', '813a', '813b', 'P673a',
                              'P673b', 'P674a', 'P674b', 'P675a', 'P675b', 'P676', 'P782b', 'P783', 'P883', 'P884',
                              '813bo', '813co', '813do', '813eo']
            # remove codes that will be treated elsewhere
            for code in df.loc[person_id, 'rt_injuries_for_minor_surgery']:
                if code in swapping_codes:
                    swapping_codes.remove(code)
            for code in df.loc[person_id, 'rt_injuries_for_major_surgery']:
                if code in swapping_codes:
                    swapping_codes.remove(code)
            for code in df.loc[person_id, 'rt_injuries_to_cast']:
                if code in swapping_codes:
                    swapping_codes.remove(code)
            for code in df.loc[person_id, 'rt_injuries_to_heal_with_time']:
                if code in swapping_codes:
                    swapping_codes.remove(code)
            relevant_codes = np.intersect1d(non_empty_injuries, swapping_codes)
            treated_code = self.module.rng.choice(relevant_codes)
            # Some TLO codes have daly weights associated with treated and non-treated injuries
            if treated_code == '813bo':
                road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, treated_code)
            # Find the injury that has been treated and then schedule a recovery date
            columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id, [treated_code])
            for col in columns:
                # estimated 6-9 months recovery times for open fractures
                df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = self.sim.date + \
                                                                                DateOffset(months=7)
            else:
                logger.debug(f"Person %d's has {open_fracture_counts} open fractures without treatment",
                             person_id)
            non_empty_injuries = person_injuries[person_injuries != "none"]

            injury_columns = non_empty_injuries.columns

            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, [treated_code])[0])

            assert not pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]), \
                'no recovery date given for this injury'
            # remove code from open fracture list
            if treated_code in df.loc[person_id, 'rt_injuries_for_open_fracture_treatment']:
                df.loc[person_id, 'rt_injuries_for_open_fracture_treatment'].remove(treated_code)

    def did_not_run(self, person_id):
        logger.debug('Open fracture treatment unavailable for person %d', person_id)


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
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # Placeholder requirement
        the_accepted_facility_level = 1
        self.TREATMENT_ID = 'RTI_Suture'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        df = self.sim.population.props
        road_traffic_injuries = self.sim.modules['RTI']

        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        codes = ['1101', '2101', '3101', '4101', '5101', '7101', '8101']
        idx, lacerationcounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        # Check the person sent here didn't die due to rti, has been through A&E, through Med int
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'
        assert df.loc[person_id, 'rt_diagnosed'], 'person sent here has not been through A and E'
        assert df.loc[person_id, 'rt_med_int'], 'person sent here has not been treated'
        # Check that the person sent here has an injury that is treated by this HSI event
        assert lacerationcounts > 0
        if lacerationcounts > 0:
            # check the number of suture kits required and request them
            item_code_suture_kit = pd.unique(
                consumables.loc[consumables['Items'] == 'Suture pack', 'Item_Code'])[0]
            item_code_cetrimide_chlorhexidine = pd.unique(
                consumables.loc[consumables['Items'] ==
                                'Cetrimide 15% + chlorhexidine 1.5% solution.for dilution _5_CMST', 'Item_Code'])[0]
            consumables_open_wound_1 = {
                'Intervention_Package_Code': dict(),
                'Item_Code': {item_code_suture_kit: lacerationcounts,
                              item_code_cetrimide_chlorhexidine: lacerationcounts}
            }

            is_cons_available_1 = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=consumables_open_wound_1,
                to_log=True)['Item_Code']

            cond = is_cons_available_1

            # Availability of consumables determines if the intervention is delivered...
            if cond[item_code_suture_kit]:
                logger.debug('This facility has open wound treatment available which has been used for person %d.',
                             person_id)
                logger.debug(f'This facility treated their {lacerationcounts} open wounds')
                if cond[item_code_cetrimide_chlorhexidine]:
                    logger.debug('This laceration was cleaned before stitching')
                    df.at[person_id, 'rt_med_int'] = True
                    columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id, codes)
                    for col in columns:
                        # heal time for lacerations is roughly two weeks according to:
                        # https://www.facs.org/~/media/files/education/patient%20ed/wound_lacerations.ashx#:~:text=of%20
                        # wound%20and%20your%20general,have%20a%20weakened%20immune%20system.
                        df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = self.sim.date + \
                                                                                        DateOffset(days=14)
                else:
                    logger.debug("This laceration wasn't cleaned before stitching, person %d is at risk of infection",
                                 person_id)
                    df.at[person_id, 'rt_med_int'] = True
                    columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id, codes)
                    for col in columns:
                        df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = self.sim.date + \
                                                                                        DateOffset(days=14)

            else:
                logger.debug('This facility has no treatment for open wounds available.')

        else:
            logger.debug("Did event run????")
            logger.debug(person_id)

            pass

    def did_not_run(self, person_id):
        logger.debug('Suture kits unavailable for person %d', person_id)


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
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['MinorSurg'] = 1
        the_accepted_facility_level = 1
        self.TREATMENT_ID = 'RTI_Burn_Management'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []
        p = self.module.parameters
        self.prob_mild_burns = p['prob_mild_burns']

    def apply(self, person_id, squeeze_factor):
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        df = self.sim.population.props
        road_traffic_injuries = self.sim.modules['RTI']

        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        codes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113']
        idx, burncounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        # check the person sent here didn't die due to rti, has been through A and E and had RTI_med_int
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'
        assert df.loc[person_id, 'rt_diagnosed'], 'this person has not been through a and e'
        assert df.loc[person_id, 'rt_med_int'], 'this person has not been treated'
        # check the person sent here has an injury treated by this module
        assert burncounts > 0
        if burncounts > 0:
            # Request materials for burn treatment
            possible_large_TBSA_burn_codes = ['7113', '8113', '4113', '5113']
            idx2, bigburncounts = \
                road_traffic_injuries.rti_find_and_count_injuries(person_injuries, possible_large_TBSA_burn_codes)
            item_code_cetrimide_chlorhexidine = pd.unique(
                consumables.loc[consumables['Items'] ==
                                'Cetrimide 15% + chlorhexidine 1.5% solution.for dilution _5_CMST', 'Item_Code'])[0]
            item_code_gauze = pd.unique(
                consumables.loc[
                    consumables['Items'] == "Dressing, paraffin gauze 9.5cm x 9.5cm (square)_packof 36_CMST",
                    'Item_Code'])[0]

            random_for_severe_burn = self.module.rng.random_sample(size=1)
            # ======================== If burns severe enough then give IV fluid replacement ===========================
            if (burncounts > 1) or ((len(idx2) > 0) & (random_for_severe_burn > self.prob_mild_burns)):
                # check if they have multiple burns, which implies a higher burned total body surface area (TBSA) which
                # will alter the treatment plan

                item_code_fluid_replacement = pd.unique(
                    consumables.loc[consumables['Items'] ==
                                    "ringer's lactate (Hartmann's solution), 500 ml_20_IDA", 'Item_Code'])[0]
                consumables_burns = {
                    'Intervention_Package_Code': dict(),
                    'Item_Code': {item_code_cetrimide_chlorhexidine: burncounts,
                                  item_code_fluid_replacement: 1, item_code_gauze: burncounts}}

            else:
                consumables_burns = {
                    'Intervention_Package_Code': dict(),
                    'Item_Code': {item_code_cetrimide_chlorhexidine: burncounts,
                                  item_code_gauze: burncounts}}
            is_cons_available = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=consumables_burns,
                to_log=True)
            logger.debug(is_cons_available)
            cond = is_cons_available
            if all(value == 1 for value in cond.values()):
                logger.debug('This facility has burn treatment available which has been used for person %d.',
                             person_id)
                logger.debug(f'This facility treated their {burncounts} burns')
                df.at[person_id, 'rt_med_int'] = True
                non_empty_injuries = person_injuries[person_injuries != "none"]
                injury_columns = non_empty_injuries.columns
                columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, codes)[0])
                # estimate burns take 4 weeks to heal
                df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=4)

                columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                           'rt_injury_7', 'rt_injury_8']
                persons_injuries = df.loc[[person_id], columns]
                non_empty_injuries = persons_injuries[persons_injuries != "none"]
                non_empty_injuries = non_empty_injuries.dropna(axis=1)
                swapping_codes = ['712b', '812', '3113', '4113', '5113', '7113', '8113', '813a', '813b', 'P673a',
                                  'P673b', 'P674a', 'P674b', 'P675a', 'P675b', 'P676', 'P782b', 'P783', 'P883', 'P884',
                                  '813bo', '813co', '813do', '813eo']
                swapping_codes = [code for code in swapping_codes if code in codes]
                # remove codes that will be treated elsewhere
                for code in df.loc[person_id, 'rt_injuries_for_major_surgery']:
                    if code in swapping_codes:
                        swapping_codes.remove(code)
                for code in df.loc[person_id, 'rt_injuries_for_minor_surgery']:
                    if code in swapping_codes:
                        swapping_codes.remove(code)
                for code in df.loc[person_id, 'rt_injuries_to_cast']:
                    if code in swapping_codes:
                        swapping_codes.remove(code)
                for code in df.loc[person_id, 'rt_injuries_to_heal_with_time']:
                    if code in swapping_codes:
                        swapping_codes.remove(code)
                for code in df.loc[person_id, 'rt_injuries_for_open_fracture_treatment']:
                    if code in swapping_codes:
                        swapping_codes.remove(code)
                relevant_codes = np.intersect1d(non_empty_injuries.values, swapping_codes)
                for code in relevant_codes:
                    if code in swapping_codes:
                        road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, relevant_codes)
                        break
            else:
                logger.debug('This facility has no treatment for burns available.')

        else:
            logger.debug("Did event run????")
            logger.debug(person_id)
            pass

    def did_not_run(self, person_id):
        logger.debug('Burn treatment unavailable for person %d', person_id)


class HSI_RTI_Tetanus_Vaccine(HSI_Event, IndividualScopeEventMixin):
    """
    This HSI event handles tetanus vaccine requests, the idea being that by separating these from the burn and
    laceration and burn treatments, those treatments can go ahead without the availability of tetanus stopping the event

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, RTI)
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # Placeholder requirement
        the_accepted_facility_level = 1
        self.TREATMENT_ID = 'RTI_Tetanus_Vaccine'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        # check the person sent here hasn't died due to rti, has been through A and E and had RTI_med_int
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'
        assert df.loc[person_id, 'rt_diagnosed'], 'This person has not been through a and e'
        assert df.loc[person_id, 'rt_med_int'], 'This person has not been through rti med int'
        # check the person sent here has an injury treated by this module
        codes_for_tetanus = ['1101', '2101', '3101', '4101', '5101', '7101', '8101',
                             '1114', '2114', '3113', '4113', '5113', '7113', '8113']

        idx, counts = RTI.rti_find_and_count_injuries(person_injuries, codes_for_tetanus)
        assert counts > 0
        # If they have a laceration/burn ask request the tetanus vaccine
        if counts > 0:
            consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
            item_code_tetanus = pd.unique(
                consumables.loc[consumables['Items'] == 'Tetanus toxin vaccine (TTV)', 'Item_Code'])[0]
            consumables_tetanus = {
                'Intervention_Package_Code': dict(),
                'Item_Code': {item_code_tetanus: 1}
            }
            is_tetanus_available = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=consumables_tetanus,
                to_log=True)
            if is_tetanus_available:
                logger.debug("Tetanus vaccine requested for person %d and given", person_id)

    def did_not_run(self, person_id):
        logger.debug('Tetanus vaccine unavailable for person %d', person_id)


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

        # Define the call on resources of this treatment event: Time of Officers (Appointments)
        #   - get an 'empty' footprint:
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        #   - update to reflect the appointments that are required
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient

        # Define the facilities at which this event can occur (only one is allowed)
        # Choose from: list(pd.unique(self.sim.modules['HealthSystem'].parameters['Facilities_For_Each_District']
        #                            ['Facility_Level']))
        the_accepted_facility_level = 1

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'RTI_Acute_Pain_Management'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        # Check that the person sent here is alive, has been through A&E and RTI_Med_int
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'
        assert df.loc[person_id, 'rt_diagnosed'], 'This person has not been through a and e'
        assert df.loc[person_id, 'rt_med_int'], 'This person has not been through rti med int'
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        road_traffic_injuries = self.sim.modules['RTI']
        pain_level = "none"
        # Injuries causing mild pain include: Lacerations, mild soft tissue injuries, TBI (for now), eye injury
        Mild_Pain_Codes = ['1101', '2101', '3101', '4101', '5101', '7101', '8101',  # lacerations
                           '241',  # Minor soft tissue injuries
                           '133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135',  # TBI
                           'P133', 'P133a', 'P133b', 'P133c', 'P133d', 'P134', 'P134a', 'P134b', 'P135',  # Perm TBI
                           '291',  # Eye injury
                           '442'
                           ]
        mild_idx, mild_counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, Mild_Pain_Codes)
        # Injuries causing moderate pain include: Fractures, dislocations, soft tissue and neck trauma
        Moderate_Pain_Codes = ['112', '113', '211', '212', '412', '414', '612', '712', '712a', '712b', '712c',
                               '811', '812', '813', '813a', '813b', '813c',  # fractures
                               '322', '323', '722', '822', '822a', '822b',  # dislocations
                               '342', '343', '361', '363',  # neck trauma
                               '461',  # chest wall bruising
                               '813bo', '813co', '813do', '813eo'  # open fractures
                               ]
        moderate_idx, moderate_counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries,
                                                                                          Moderate_Pain_Codes)
        # Injuries causing severe pain include: All burns, amputations, spinal cord injuries, abdominal trauma see
        # (https://bestbets.org/bets/bet.php?id=1247), severe chest trauma
        Severe_Pain_Codes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113',  # burns
                             'P782', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883', 'P884',  # amputations
                             '673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676',
                             'P673', 'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675', 'P675a', 'P675b', 'P676',
                             # SCI
                             '552', '553', '554',  # abdominal trauma
                             '463', '453', '453a', '453b', '441', '443'  # severe chest trauma
                             ]
        severe_idx, severe_counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries,
                                                                                      Severe_Pain_Codes)
        # check that the people here have at least one injury
        assert mild_counts + moderate_counts + severe_counts > 0
        if len(severe_idx) > 0:
            pain_level = "severe"
        elif len(moderate_idx) > 0:
            pain_level = "moderate"
        elif len(mild_idx) > 0:
            pain_level = "mild"

        if pain_level == "mild":
            # Multiple options, some are conditional
            # Give paracetamol
            # Give NSAIDS such as aspirin (unless they are under 16) for soft tissue pain, but not if they are pregnant
            dict_to_output = {'person': person_id,
                              'pain level': pain_level}
            logger.info(key='Requested_Pain_Management',
                        data=dict_to_output,
                        description='Summary of the pain medicine requested by each person')
            item_code_paracetamol = pd.unique(
                consumables.loc[consumables['Items'] == "Paracetamol 500mg_1000_CMST",
                                'Item_Code'])[0]
            item_code_diclofenac = pd.unique(
                consumables.loc[consumables['Items'] == "diclofenac sodium 25 mg, enteric coated_1000_IDA",
                                'Item_Code'])[0]

            pain_management_strategy_paracetamol = {
                'Intervention_Package_Code': dict(),
                'Item_Code': {item_code_paracetamol: 1}}
            pain_management_strategy_diclofenac = {
                'Intervention_Package_Code': dict(),
                'Item_Code': {item_code_diclofenac: 1}}

            if (df.iloc[person_id]['age_years'] < 16):
                # or df.iloc[person_id]['is_pregnant']
                # If they are under 16 or pregnant only give them paracetamol
                logger.debug(pain_management_strategy_paracetamol)
                is_paracetamol_available = self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=self,
                    cons_req_as_footprint=pain_management_strategy_paracetamol,
                    to_log=True)['Item_Code'][item_code_paracetamol]
                cond = is_paracetamol_available
                logger.debug('Person %d requested paracetamol for their pain relief', person_id)
            else:
                # Multiple options, give them what's available or random pick between them (for now)
                is_diclofenac_available = self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=self,
                    cons_req_as_footprint=pain_management_strategy_diclofenac,
                    to_log=True)['Item_Code'][item_code_diclofenac]

                is_paracetamol_available = self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=self,
                    cons_req_as_footprint=pain_management_strategy_paracetamol,
                    to_log=True)['Item_Code'][item_code_paracetamol]

                cond1 = is_paracetamol_available
                cond2 = is_diclofenac_available
                if (cond1 is True) & (cond2 is True):
                    which = self.module.rng.random_sample(size=1)
                    if which <= 0.5:
                        cond = cond1
                        logger.debug('Person %d requested paracetamol for their pain relief', person_id)
                    else:
                        cond = cond2
                        logger.debug('Person %d requested diclofenac for their pain relief', person_id)
                elif (cond1 is True) & (cond2 is False):
                    cond = cond1
                    logger.debug('Person %d requested paracetamol for their pain relief', person_id)
                elif (cond1 is False) & (cond2 is True):
                    cond = cond2
                    logger.debug('Person %d requested diclofenac for their pain relief', person_id)
                else:
                    which = self.module.rng.random_sample(size=1)
                    if which <= 0.5:
                        cond = cond1
                        logger.debug('Person %d requested paracetamol for their pain relief', person_id)
                    else:
                        cond = cond2
                        logger.debug('Person %d requested diclofenac for their pain relief', person_id)
            # Availability of consumables determines if the intervention is delivered...
            if cond:
                logger.debug('This facility has pain management available for mild pain which has been used for '
                             'person %d.', person_id)
                dict_to_output = {'person': person_id,
                                  'pain level': pain_level}
                logger.info(key='Successful_Pain_Management',
                            data=dict_to_output,
                            description='Pain medicine successfully provided to the person')
            else:
                logger.debug('This facility has no pain management available for their mild pain, person %d.',
                             person_id)

        if pain_level == "moderate":
            dict_to_output = {'person': person_id,
                              'pain level': pain_level}
            logger.info(key='Requested_Pain_Management',
                        data=dict_to_output,
                        description='Summary of the pain medicine requested by each person')
            item_code_tramadol = pd.unique(
                consumables.loc[consumables['Items'] == "tramadol HCl 100 mg/2 ml, for injection_100_IDA",
                                'Item_Code'])[0]

            pain_management_strategy_tramadol = {
                'Intervention_Package_Code': dict(),
                'Item_Code': {item_code_tramadol: 1}}

            is_cons_available = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=pain_management_strategy_tramadol,
                to_log=True)['Item_Code'][item_code_tramadol]
            cond = is_cons_available
            logger.debug('Person %d has requested tramadol for moderate pain relief', person_id)

            if cond:
                logger.debug('This facility has pain management available for moderate pain which has been used for '
                             'person %d.', person_id)
                dict_to_output = {'person': person_id,
                                  'pain level': pain_level}
                logger.info(key='Successful_Pain_Management',
                            data=dict_to_output,
                            description='Pain medicine successfully provided to the person')
            else:
                logger.debug('This facility has no pain management available for moderate pain for person %d.',
                             person_id)

        if pain_level == "severe":
            dict_to_output = {'person': person_id,
                              'pain level': pain_level}
            logger.info(key='Requested_Pain_Management',
                        data=dict_to_output,
                        description='Summary of the pain medicine requested by each person')
            # give morphine
            item_code_morphine = pd.unique(
                consumables.loc[consumables['Items'] == "morphine sulphate 10 mg/ml, 1 ml, injection (nt)_10_IDA",
                                'Item_Code'])[0]

            pain_management_strategy = {
                'Intervention_Package_Code': dict(),
                'Item_Code': {item_code_morphine: 1}}

            is_cons_available = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=pain_management_strategy,
                to_log=True)
            cond = is_cons_available
            logger.debug('Person %d has requested morphine for severe pain relief', person_id)

            if cond:
                logger.debug('This facility has pain management available for severe pain which has been used for '
                             'person %d.', person_id)
                dict_to_output = {'person': person_id,
                                  'pain level': pain_level}
                logger.info(key='Successful_Pain_Management',
                            data=dict_to_output,
                            description='Pain medicine successfully provided to the person')
            else:
                logger.debug('This facility has no pain management available for severe pain for person %d.', person_id)

    def did_not_run(self, person_id):
        df = self.sim.population.props
        logger.debug('Pain relief unavailable for person %d', person_id)
        injurycodes = {'First injury': df.loc[person_id, 'rt_injury_1'],
                       'Second injury': df.loc[person_id, 'rt_injury_2'],
                       'Third injury': df.loc[person_id, 'rt_injury_3'],
                       'Fourth injury': df.loc[person_id, 'rt_injury_4'],
                       'Fifth injury': df.loc[person_id, 'rt_injury_5'],
                       'Sixth injury': df.loc[person_id, 'rt_injury_6'],
                       'Seventh injury': df.loc[person_id, 'rt_injury_7'],
                       'Eight injury': df.loc[person_id, 'rt_injury_8']}
        logger.debug(f'Injury profile of person %d, {injurycodes}', person_id)


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
        https://www.unthsc.edu/texas-college-of-osteopathic-medicine/wp-content/uploads/sites/9/Pediatric_Handbook_for_Malawi.pdf
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
        self.TREATMENT_ID = 'RTI_Major_Surgeries'
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['MajorSurg'] = 1  # This requires major surgery

        the_accepted_facility_level = 1
        p = self.module.parameters

        # Define the necessary information for an HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []
        self.prob_perm_disability_with_treatment_severe_TBI = p['prob_perm_disability_with_treatment_severe_TBI']
        self.prob_perm_disability_with_treatment_sci = p['prob_perm_disability_with_treatment_sci']
        self.allowed_interventions = p['allowed_interventions']
        self.treated_code = 'none'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        rng = self.module.rng
        road_traffic_injuries = self.sim.modules['RTI']
        # check the people sent here hasn't died due to rti, have had their injuries diagnosed and been through RTI_Med
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'
        assert df.loc[person_id, 'rt_diagnosed'], 'This person has not been through a and e'
        assert df.loc[person_id, 'rt_med_int'], 'This person has not been through rti med int'
        # Isolate the relevant injury information
        columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                   'rt_injury_7', 'rt_injury_8']
        surgically_treated_codes = ['112', '811', '812', '813a', '813b', '813c', '133a', '133b', '133c', '133d', '134a',
                                    '134b', '135', '552', '553', '554', '342', '343', '414', '361', '363',
                                    '782', '782a', '782b', '782c', '783', '822a', '882', '883', '884',
                                    'P133a', 'P133b', 'P133c', 'P133d', 'P134a', 'P134b', 'P135', 'P782a', 'P782b',
                                    'P782c', 'P783', 'P882', 'P883', 'P884']
        # If we have allowed spinal cord surgeries to be treated in this simulation, include the associated injury codes
        # here
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
        persons_injuries = df.loc[[person_id], columns]
        injuries_to_be_treated = df.loc[person_id, 'rt_injuries_for_major_surgery']
        assert len(set(injuries_to_be_treated) & set(surgically_treated_codes)) > 0, \
            'This person has asked for surgery but does not have an appropriate injury'
        # check the people sent here have at least one injury treated by this HSI event
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(persons_injuries, surgically_treated_codes)
        assert counts > 0, (persons_injuries.to_dict(), surgically_treated_codes)
        # People can be sent here for multiple surgeries, but only one injury can be treated at a time. Decide which
        # injury is being treated in this surgery
        idx_for_untreated_injuries = []
        for index, time in enumerate(df.loc[person_id, 'rt_date_to_remove_daly']):
            if pd.isnull(time):
                idx_for_untreated_injuries.append(index)

        relevant_codes = np.intersect1d(injuries_to_be_treated, surgically_treated_codes)
        assert len(relevant_codes) > 0, (persons_injuries.values[0], idx_for_untreated_injuries, person_id,
                                         persons_injuries.values[0][idx_for_untreated_injuries])
        self.treated_code = rng.choice(relevant_codes)
        # ------------------------ Track permanent disabilities with treatment ----------------------------------------
        # --------------------------------- Perm disability from TBI --------------------------------------------------
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
                logger.debug('@@@@@@@@@@ Person %d had intervention for TBI on %s but still disabled!!!!!!',
                             person_id, self.sim.date)
                # Update the code to make the injury permanent, so it will not have the associated daly weight removed
                # later on
                df.loc[person_id, column] = "P" + self.treated_code
                code_to_drop_index = injuries_to_be_treated.index(self.treated_code)
                injuries_to_be_treated.pop(code_to_drop_index)
                injuries_to_be_treated.append("P" + self.treated_code)
                # df.loc[person_id, 'rt_injuries_for_major_surgery'] = injuries_to_be_treated
                for injury in injuries_to_be_treated:
                    if injury not in df.loc[person_id, 'rt_injuries_for_major_surgery']:
                        df.loc[person_id, 'rt_injuries_for_major_surgery'].append(injury)
                assert len(injuries_to_be_treated) == len(df.loc[person_id, 'rt_injuries_for_major_surgery'])

            columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id,
                                                                                            [self.treated_code])
            for col in columns:
                # schedule the recovery date for the permanent injury for beyond the end of the simulation (making
                # it permanent)
                df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = self.sim.end_date + DateOffset(days=1)
        # ------------------------------------- Perm disability from SCI ----------------------------------------------
        if 'include_spine_surgery' in self.allowed_interventions:
            codes = ['673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676']
            if self.treated_code in codes:
                # Track whether they are permanently disabled
                df.at[person_id, 'rt_perm_disability'] = True
                # Find the column and code where the permanent injury is stored
                column, code = road_traffic_injuries.rti_find_injury_column(person_id=person_id,
                                                                            codes=[self.treated_code])
                logger.debug('@@@@@@@@@@ Person %d had intervention for SCI on %s but still disabled!!!!!!',
                             person_id, self.sim.date)
                df.loc[person_id, column] = "P" + self.treated_code
                code_to_drop_index = injuries_to_be_treated.index(self.treated_code)
                injuries_to_be_treated.pop(code_to_drop_index)
                injuries_to_be_treated.append("P" + self.treated_code)
                # df.loc[person_id, 'rt_injuries_for_major_surgery'] = injuries_to_be_treated
                for injury in injuries_to_be_treated:
                    if injury not in df.loc[person_id, 'rt_injuries_for_major_surgery']:
                        df.loc[person_id, 'rt_injuries_for_major_surgery'].append(injury)
                assert len(injuries_to_be_treated) == len(df.loc[person_id, 'rt_injuries_for_major_surgery'])
                code = df.loc[person_id, column]
                columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id, [code])
                for col in columns:
                    # schedule the recovery date for the permanent injury for beyond the end of the simulation (making
                    # it permanent)
                    df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = self.sim.end_date + \
                                                                                    DateOffset(days=1)
        # ------------------------------------- Perm disability from amputation ----------------------------------------
        codes = ['782', '782a', '782b', '782c', '783', '882', '883', '884']
        if self.treated_code in codes:
            # Track whether they are permanently disabled
            df.at[person_id, 'rt_perm_disability'] = True
            # Find the column and code where the permanent injury is stored
            column, code = road_traffic_injuries.rti_find_injury_column(person_id=person_id, codes=[self.treated_code])
            logger.debug('@@@@@@@@@@ Person %d had intervention for an amputation on %s but still disabled!!!!!!',
                         person_id, self.sim.date)
            # Update the code to make the injury permanent, so it will not have the associated daly weight removed
            # later on
            df.loc[person_id, column] = "P" + self.treated_code
            code_to_drop_index = injuries_to_be_treated.index(self.treated_code)

            injuries_to_be_treated.pop(code_to_drop_index)
            injuries_to_be_treated.append("P" + self.treated_code)
            # df.loc[person_id, 'rt_injuries_for_major_surgery'] = injuries_to_be_treated
            for injury in injuries_to_be_treated:
                if injury not in df.loc[person_id, 'rt_injuries_for_major_surgery']:
                    df.loc[person_id, 'rt_injuries_for_major_surgery'].append(injury)
            assert len(injuries_to_be_treated) == len(df.loc[person_id, 'rt_injuries_for_major_surgery'])
            code = df.loc[person_id, column]
            columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id,
                                                                                            [code])
            # Schedule recovery for the end of the simulation, thereby making the injury permanent
            for col in columns:
                df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = self.sim.end_date + \
                                                                                DateOffset(days=1)
        # ============================== Schedule the recovery dates for the non-permanent injuries ==================
        non_empty_injuries = persons_injuries[persons_injuries != "none"]
        non_empty_injuries = non_empty_injuries.dropna(axis=1)
        injury_columns = non_empty_injuries.columns
        if self.treated_code == '112':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [self.treated_code])[0])
            # using estimated 6 weeks to recover from brain/head injury surgery

            # performing check to see whether an injury is deemed to heal over time, if it is, then we change the code
            # this scheduled surgery treats
            if pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]):
                df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=6)
            else:
                non_empty_injuries.drop(non_empty_injuries.columns[columns], axis=1, inplace=True)
                relevant_codes = np.intersect1d(non_empty_injuries.values, surgically_treated_codes)
                self.treated_code = rng.choice(relevant_codes)
        if self.treated_code == '552' or self.treated_code == '553' or self.treated_code == '554':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [self.treated_code])[0])
            # using estimated 3 months to recover from laparotomy
            if pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]):
                df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(months=3)
            else:
                non_empty_injuries.drop(non_empty_injuries.columns[columns], axis=1, inplace=True)
                relevant_codes = np.intersect1d(non_empty_injuries.values, surgically_treated_codes)
                assert len(relevant_codes) > 0
                self.treated_code = rng.choice(relevant_codes)
        if self.treated_code == '822a':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [self.treated_code])[0])
            # using estimated 6 - 12 months to recover from a hip dislocation
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(months=9)
        if self.treated_code == '811':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [self.treated_code])[0])
            # using estimated 9 weeks to recover from a foot fracture treated with surgery
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=9)
        if self.treated_code == '812':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [self.treated_code])[0])
            # using estimated 9 weeks to recover from a tibia/fibula fracture treated with surgery
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=9)
        if self.treated_code == '813a':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [self.treated_code])[0])
            # using estimated 6 - 12 months to recover from a hip fracture
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(months=9)
        if self.treated_code == '813b':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [self.treated_code])[0])
            # using estimated 8 - 12 weeks to recover from a pelvis fracture
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=10)
        if self.treated_code == '813c':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [self.treated_code])[0])
            # using estimated 3 - 6 months to recover from a femur fracture
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(months=4)
        tbi_codes = ['133a', '133b', '133c', '133d', '134a', '134b', '135']
        if self.treated_code in tbi_codes:
            columns = injury_columns.get_loc(
                road_traffic_injuries.rti_find_injury_column(person_id, [self.treated_code])[0])
            # using estimated 6 weeks to recover from brain/head injury surgery
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=6)
        if self.treated_code == '342':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [self.treated_code])[0])
            # using estimated 6 weeks PLACEHOLDER FOR VERTEBRAL ARTERY LACERATION
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=6)
        if self.treated_code == '343':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [self.treated_code])[0])
            # using estimated 6 weeks PLACEHOLDER FOR PHARYNX CONTUSION
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=6)
        if self.treated_code == '414':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [self.treated_code])[0])
            # using estimated 1 year recovery for flail chest
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(years=1)
        if self.treated_code == '441' or self.treated_code == '443':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [self.treated_code])[0])
            # using estimated 1 - 2 week recovery time for pneumothorax
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=2)
        if self.treated_code == '453a' or self.treated_code == '453b':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [self.treated_code])[0])
            # using estimated 6 weeks PLACEHOLDER FOR DIAPHRAGM RUPTURE
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=6)
        if self.treated_code == '361' or self.treated_code == '363' or self.treated_code == '463':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [self.treated_code])[0])
            # using estimated 1 weeks PLACEHOLDER FOR INTERNAL BLEEDING
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=1)
        swapping_codes = ['712b', '812', '3113', '4113', '5113', '7113', '8113', '813a', '813b', 'P673a',
                          'P673b', 'P674a', 'P674b', 'P675a', 'P675b', 'P676', 'P782b', 'P783', 'P883', 'P884',
                          '813bo', '813co', '813do', '813eo']
        swapping_codes = [code for code in swapping_codes if code in surgically_treated_codes]
        # remove codes that will be treated elsewhere
        for code in df.loc[person_id, 'rt_injuries_for_minor_surgery']:
            if code in swapping_codes:
                swapping_codes.remove(code)
        for code in df.loc[person_id, 'rt_injuries_to_cast']:
            if code in swapping_codes:
                swapping_codes.remove(code)
        for code in df.loc[person_id, 'rt_injuries_to_heal_with_time']:
            if code in swapping_codes:
                swapping_codes.remove(code)
        for code in df.loc[person_id, 'rt_injuries_for_open_fracture_treatment']:
            if code in swapping_codes:
                swapping_codes.remove(code)
        if self.treated_code in swapping_codes:
            road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, [self.treated_code])
        # Check that every injury treated has a recovery time
        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                      [self.treated_code])[0])
        assert not pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]), \
            'no recovery date given for this injury'
        logger.debug('This is RTI_Major_Surgeries supplying surgery for person %d on date %s!!!!!!, removing code %s',
                     person_id, self.sim.date)
        # remove code from major surgeries list
        if self.treated_code in df.loc[person_id, 'rt_injuries_for_major_surgery']:
            df.loc[person_id, 'rt_injuries_for_major_surgery'].remove(self.treated_code)

    def did_not_run(self, person_id):
        df = self.sim.population.props
        logger.debug('Major surgery not scheduled for person %d', person_id)
        injurycodes = {'First injury': df.loc[person_id, 'rt_injury_1'],
                       'Second injury': df.loc[person_id, 'rt_injury_2'],
                       'Third injury': df.loc[person_id, 'rt_injury_3'],
                       'Fourth injury': df.loc[person_id, 'rt_injury_4'],
                       'Fifth injury': df.loc[person_id, 'rt_injury_5'],
                       'Sixth injury': df.loc[person_id, 'rt_injury_6'],
                       'Seventh injury': df.loc[person_id, 'rt_injury_7'],
                       'Eight injury': df.loc[person_id, 'rt_injury_8']}
        logger.debug(f'Injury profile of person %d, {injurycodes}', person_id)

        # If the surgery was a life-saving surgery, then send them to RTI_No_Medical_Intervention_Death_Event
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
        if (self.treated_code in life_threatening_injuries) & df.loc[person_id, 'is_alive']:
            self.sim.schedule_event(RTI_No_Lifesaving_Medical_Intervention_Death_Event(self.module, person_id),
                                    self.sim.date)


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
        self.TREATMENT_ID = 'RTI_Minor_Surgeries'
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['MinorSurg'] = 1  # This requires major surgery

        the_accepted_facility_level = 1

        # Define the necessary information for an HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        rng = self.module.rng
        road_traffic_injuries = self.sim.modules['RTI']
        columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                   'rt_injury_7', 'rt_injury_8']
        surgically_treated_codes = ['322', '211', '212', '323', '722', '291', '241', '811', '812', '813a', '813b',
                                    '813c']
        persons_injuries = df.loc[[person_id], columns]
        # =========================================== Tests ============================================================
        # check the people sent here hasn't died due to rti, have had their injuries diagnosed and been through RTI_Med
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'
        assert df.loc[person_id, 'rt_diagnosed'], 'This person has not been through a and e'
        assert df.loc[person_id, 'rt_med_int'], 'This person has not been through rti med int'
        # check they have at least one injury treated by minor surgery
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(persons_injuries, surgically_treated_codes)
        assert counts > 0
        non_empty_injuries = persons_injuries[persons_injuries != "none"]
        non_empty_injuries = non_empty_injuries.dropna(axis=1)

        relevant_codes = np.intersect1d(df.loc[person_id, 'rt_injuries_for_minor_surgery'], surgically_treated_codes)
        # Check that a code has been selected to be treated
        assert len(relevant_codes) > 0
        treated_code = rng.choice(relevant_codes)

        injury_columns = non_empty_injuries.columns

        # elsewhere
        if treated_code == '322' or treated_code == '323':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [treated_code])[0])

            # using estimated 6 months to recover from neck surgery
            if pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]):
                df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(months=6)
            else:
                non_empty_injuries.drop(non_empty_injuries.columns[columns], axis=1, inplace=True)
                relevant_codes = np.intersect1d(non_empty_injuries.values, surgically_treated_codes)
                assert len(relevant_codes) > 0
                treated_code = rng.choice(relevant_codes)

        if treated_code == '722':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [treated_code])[0])
            # using estimated 7 weeks to recover from dislocated shoulder surgery
            if pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]):
                df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=7)
            else:
                non_empty_injuries.drop(non_empty_injuries.columns[columns], axis=1, inplace=True)
                relevant_codes = np.intersect1d(non_empty_injuries.values, surgically_treated_codes)
                treated_code = rng.choice(relevant_codes)
        if treated_code == '211' or treated_code == '212':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [treated_code])[0])
            # using estimated 7 weeks to recover from facial fracture surgery
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=7)
        if treated_code == '291':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [treated_code])[0])
            # using estimated 1 week to recover from eye surgery
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=1)
        if treated_code == '241':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [treated_code])[0])
            # using estimated 1 week to soft tissue injury to face
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=1)
        if treated_code == '811':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [treated_code])[0])
            # using estimated 9 weeks for external fixation
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=9)
        if treated_code == '812':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [treated_code])[0])
            # using estimated 9 weeks for external fixation
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=9)
        if treated_code == '813a':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [treated_code])[0])
            # using estimated 9 weeks for external fixation
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=9)
        if treated_code == '813b':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [treated_code])[0])
            # using estimated 9 weeks for external fixation
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=9)
        if treated_code == '813c':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [treated_code])[0])
            # using estimated 9 weeks for external fixation
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=9)
        swapping_codes = ['712b', '812', '3113', '4113', '5113', '7113', '8113', '813a', '813b', 'P673a',
                          'P673b', 'P674a', 'P674b', 'P675a', 'P675b', 'P676', 'P782b', 'P783', 'P883', 'P884',
                          '813bo', '813co', '813do', '813eo']
        for code in df.loc[person_id, 'rt_injuries_for_minor_surgery']:
            if code in swapping_codes:
                swapping_codes.remove(code)
        for code in df.loc[person_id, 'rt_injuries_to_cast']:
            if code in swapping_codes:
                swapping_codes.remove(code)
        for code in df.loc[person_id, 'rt_injuries_to_heal_with_time']:
            if code in swapping_codes:
                swapping_codes.remove(code)
        for code in df.loc[person_id, 'rt_injuries_for_open_fracture_treatment']:
            if code in swapping_codes:
                swapping_codes.remove(code)

        if code in swapping_codes:
            road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, [code])
        logger.debug('This is RTI_Minor_Surgeries supplying minor surgeries for person %d on date %s!!!!!!',
                     person_id, self.sim.date)
        df.at[person_id, 'rt_med_int'] = True
        # Check if the injury has been given a recovery date['211']
        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, [treated_code])[0])
        assert not pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]), \
            'no recovery date given for this injury'
        # remove code from minor surgeries list
        if treated_code in df.loc[person_id, 'rt_injuries_for_minor_surgery']:
            df.loc[person_id, 'rt_injuries_for_minor_surgery'].remove(treated_code)

    def did_not_run(self, person_id):
        df = self.sim.population.props
        logger.debug('Minor surgery not scheduled for person %d', person_id)
        injurycodes = {'First injury': df.loc[person_id, 'rt_injury_1'],
                       'Second injury': df.loc[person_id, 'rt_injury_2'],
                       'Third injury': df.loc[person_id, 'rt_injury_3'],
                       'Fourth injury': df.loc[person_id, 'rt_injury_4'],
                       'Fifth injury': df.loc[person_id, 'rt_injury_5'],
                       'Sixth injury': df.loc[person_id, 'rt_injury_6'],
                       'Seventh injury': df.loc[person_id, 'rt_injury_7'],
                       'Eight injury': df.loc[person_id, 'rt_injury_8']}
        logger.debug(f'Injury profile of person %d, {injurycodes}', person_id)


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
        self.prob_death_with_med_mild = p['prob_death_with_med_mild']
        self.prob_death_with_med_severe = p['prob_death_with_med_severe']
        self.rr_injrti_mortality_polytrauma = p['rr_injrti_mortality_polytrauma']

    def apply(self, person_id):
        df = self.sim.population.props
        randfordeath = self.module.rng.random_sample(size=1)
        # ======================================== Tests ==============================================================
        # Check the person sent here hasn't died due to rti
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'

        # Schedule death for those who died from their injuries despite medical intervention
        if df.loc[person_id, 'cause_of_death'] == 'Other':
            pass
        if df.loc[person_id, 'rt_inj_severity'] == 'mild':
            # Check if the people with mild injuries have the polytrauma property, if so, judge mortality accordingly
            if df.loc[person_id, 'rt_polytrauma'] is True:
                if randfordeath < self.prob_death_with_med_mild * self.rr_injrti_mortality_polytrauma:
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
                    self.sim.schedule_event(demography.InstantaneousDeath(self.module, person_id,
                                                                          cause='RTI_death_with_med'), self.sim.date)
                    # Log the death
                    logger.debug('This is RTIMedicalInterventionDeathEvent scheduling a death for person %d who was '
                                 'treated for their injuries but still died on date %s',
                                 person_id, self.sim.date)
            elif randfordeath < self.prob_death_with_med_mild:
                # No polytrauma
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
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, person_id,
                                                                      cause='RTI_death_with_med'), self.sim.date)
                # Log the death
                logger.debug('This is RTIMedicalInterventionDeathEvent scheduling a death for person %d who was'
                             'treated for their injuries but still died on date %s',
                             person_id, self.sim.date)
            else:
                logger.debug('RTIMedicalInterventionDeathEvent determining that person %d was treated for injuries and '
                             'survived on date %s',
                             person_id, self.sim.date)
        if df.loc[person_id, 'rt_inj_severity'] == 'severe':
            if df.loc[person_id, 'rt_polytrauma'] is True:
                # Predict death if they have polytrauma for severe injuries
                if randfordeath < self.prob_death_with_med_severe * self.rr_injrti_mortality_polytrauma:
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
                    self.sim.schedule_event(demography.InstantaneousDeath(self.module, person_id,
                                                                          cause='RTI_death_with_med'), self.sim.date)
                    # Log the death
                    logger.debug('This is RTIMedicalInterventionDeathEvent scheduling a death for person %d who was '
                                 'treated for their injuries but still died on date %s',
                                 person_id, self.sim.date)
                elif randfordeath < self.prob_death_with_med_severe:
                    # Predict death without polytrauma for severe injuries
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
                    self.sim.schedule_event(demography.InstantaneousDeath(self.module, person_id,
                                                                          cause='RTI_death_with_med'), self.sim.date)
                    # Log the death
                    logger.debug('This is RTIMedicalInterventionDeathEvent scheduling a death for person %d who was'
                                 'treated for their injuries but still died on date %s',
                                 person_id, self.sim.date)
            else:
                logger.debug('RTIMedicalInterventionDeathEvent has determined person %d was '
                             'treated for injuries and survived on date %s',
                             person_id, self.sim.date)


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
        # self.scheduled_death = 0

    def apply(self, person_id):
        # self.scheduled_death = 0
        df = self.sim.population.props
        # =========================================== Tests ============================================================
        # check the people sent here hasn't died due to rti
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        assert df.loc[person_id, 'cause_of_death'] not in rti_deaths, 'person sent for treatment died from rti module'
        columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                   'rt_injury_7', 'rt_injury_8']
        non_lethal_injuries = ['241', '291', '322', '323', '461', '442', '1101', '2101', '3101', '4101', '5101', '7101',
                               '8101', '722', '822a', '822b']
        severeinjurycodes = ['133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135',
                             '342', '343', '361', '363', '414', '441', '443', '453a', '453b', '463', '552',
                             '553', '554', '673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b',
                             '676', '782a', '782b', '782c', '783', '882', '883', '884', 'P133', 'P133a', 'P133b',
                             'P133c', 'P133d', 'P134', 'P134a', 'P134b', 'P135', 'P673', 'P673a', 'P673b', 'P674',
                             'P674a', 'P674b', 'P675', 'P675a', 'P675b', 'P676', 'P782a', 'P782b', 'P782c', 'P783',
                             'P882', 'P883', 'P884']
        fractureinjurycodes = ['112', '113', '211', '212', '412', '612', '712', '712a', '712b', '712c', '811',
                               '812', '813', '813a', '813b', '813c']
        burninjurycodes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113']
        persons_injuries = df.loc[[person_id], columns]
        non_empty_injuries = persons_injuries[persons_injuries != "none"]
        non_empty_injuries = non_empty_injuries.dropna(axis=1)
        untreated_injuries = []
        prob_death = 0
        # Find which injuries are left untreated by finding injuries which haven't been set a recovery time
        for col in non_empty_injuries:
            if pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1]):
                untreated_injuries.append(df.at[person_id, col])
        for injury in untreated_injuries:
            if injury in severeinjurycodes:
                if prob_death < self.prob_death_TBI_SCI_no_treatment:
                    prob_death = self.prob_death_TBI_SCI_no_treatment
            elif injury in fractureinjurycodes:
                if prob_death < self.prob_death_fractures_no_treatment:
                    prob_death = self.prob_death_fractures_no_treatment
            elif injury in burninjurycodes:
                if prob_death < self.prop_death_burns_no_treatment:
                    prob_death = self.prop_death_burns_no_treatment
            elif injury in non_lethal_injuries:
                pass
        randfordeath = self.module.rng.random_sample(size=1)
        if randfordeath < prob_death:
            df.loc[person_id, 'rt_unavailable_med_death'] = True
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, person_id,
                                                                  cause='RTI_unavailable_med'), self.sim.date)
            # Log the death
            logger.debug(
                'This is RTINoMedicalInterventionDeathEvent scheduling a death for person %d on date %s',
                person_id, self.sim.date)


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
        road_traffic_injuries = self.sim.modules['RTI']
        columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                   'rt_injury_7', 'rt_injury_8']
        thoseininjuries = df.loc[df.rt_road_traffic_inc]
        df_injuries = thoseininjuries.loc[:, columns]
        # ==================================== Number of injuries =====================================================
        #
        oneinjury = len(df_injuries.loc[df_injuries['rt_injury_2'] == 'none'])
        self.tot1inj += oneinjury
        twoinjury = len(df_injuries.loc[(df_injuries['rt_injury_2'] != 'none') &
                                        (df_injuries['rt_injury_3'] == 'none')])
        self.tot2inj += twoinjury
        threeinjury = len(df_injuries.loc[(df_injuries['rt_injury_3'] != 'none') &
                                          (df_injuries['rt_injury_4'] == 'none')])
        self.tot3inj += threeinjury
        fourinjury = len(df_injuries.loc[(df_injuries['rt_injury_4'] != 'none') &
                                         (df_injuries['rt_injury_5'] == 'none')])
        self.tot4inj += fourinjury
        fiveinjury = len(df_injuries.loc[(df_injuries['rt_injury_5'] != 'none') &
                                         (df_injuries['rt_injury_6'] == 'none')])
        self.tot5inj += fiveinjury
        sixinjury = len(df_injuries.loc[(df_injuries['rt_injury_6'] != 'none') &
                                        (df_injuries['rt_injury_7'] == 'none')])
        self.tot6inj += sixinjury
        seveninjury = len(df_injuries.loc[(df_injuries['rt_injury_7'] != 'none') &
                                          (df_injuries['rt_injury_8'] == 'none')])
        self.tot7inj += seveninjury
        eightinjury = len(df_injuries.loc[df_injuries['rt_injury_8'] != 'none'])
        self.tot8inj += eightinjury
        dict_to_output = {
            'total_one_injured_body_region': self.tot1inj,
            'total_two_injured_body_region': self.tot2inj,
            'total_three_injured_body_region': self.tot3inj,
            'total_four_injured_body_region': self.tot4inj,
            'total_five_injured_body_region': self.tot5inj,
            'total_six_injured_body_region': self.tot6inj,
            'total_seven_injured_body_region': self.tot7inj,
            'total_eight_injured_body_region': self.tot8inj,
        }
        logger.info(key='number_of_injuries',
                    data=dict_to_output,
                    description='The total number of injured body regions in the simulation')
        # ====================================== AIS body regions =====================================================
        AIS1codes = ['112', '113', '133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135', '1101', '1114']
        AIS2codes = ['211', '212', '2101', '291', '241', '2114']
        AIS3codes = ['342', '343', '361', '363', '322', '323', '3101', '3113']
        AIS4codes = ['412', '414', '461', '463', '453', '453a', '453b', '441', '442', '443', '4101', '4114']
        AIS5codes = ['552', '553', '554', '5101', '5114']
        AIS6codes = ['612', '673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676']
        AIS7codes = ['712', '712a', '712b', '712c', '722', '782', '782a', '782b', '782c', '783', '7101', '7114']
        AIS8codes = ['811', '812', '813', '813a', '813b', '813c', '822', '822a', '822b', '882', '883', '884', '8101',
                     '8114', '813bo', '813co', '813do', '813eo']
        idx, AIS1counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, AIS1codes)
        self.totAIS1 += AIS1counts
        idx, AIS2counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, AIS2codes)
        self.totAIS2 += AIS2counts
        idx, AIS3counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, AIS3codes)
        self.totAIS3 += AIS3counts
        idx, AIS4counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, AIS4codes)
        self.totAIS4 += AIS4counts
        idx, AIS5counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, AIS5codes)
        self.totAIS5 += AIS5counts
        idx, AIS6counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, AIS6codes)
        self.totAIS6 += AIS6counts
        idx, AIS7counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, AIS7codes)
        self.totAIS7 += AIS7counts
        idx, AIS8counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, AIS8codes)
        self.totAIS8 += AIS8counts
        dict_to_output = {'total_number_of_head_injuries': self.totAIS1,
                          'total_number_of_facial_injuries': self.totAIS2,
                          'total_number_of_neck_injuries': self.totAIS3,
                          'total_number_of_thorax_injuries': self.totAIS4,
                          'total_number_of_abdomen_injuries': self.totAIS5,
                          'total_number_of_spinal_injuries': self.totAIS6,
                          'total_number_of_upper_ex_injuries': self.totAIS7,
                          'total_number_of_lower_ex_injuries': self.totAIS8}
        logger.info(key='injury_location_data',
                    data=dict_to_output,
                    description='Data on distribution of the injury location on the body')
        # Log where the fractures occur in the body
        skullfracs = ['112', '113']
        idx, skullfraccounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, skullfracs)
        self.fracdist[0] += skullfraccounts
        facefracs = ['211', '212']
        idx, facefraccounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, facefracs)
        self.fracdist[1] += facefraccounts
        thoraxfracs = ['412', '414']
        idx, thorfraccounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, thoraxfracs)
        self.fracdist[3] += thorfraccounts
        spinefracs = ['612']
        idx, spinfraccounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, spinefracs)
        self.fracdist[5] += spinfraccounts
        upperexfracs = ['712', '712a', '712b', '712c']
        idx, upperexfraccounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, upperexfracs)
        self.fracdist[6] += upperexfraccounts
        lowerexfracs = ['811', '812', '813', '813a', '813b', '813c', '813bo', '813co', '813do', '813eo']
        idx, lowerexfraccounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, lowerexfracs)
        self.fracdist[7] += lowerexfraccounts
        dict_to_output = {
            'total_head_fractures': self.fracdist[0],
            'total_facial_fractures': self.fracdist[1],
            'total_thorax_fractures': self.fracdist[3],
            'total_spinal_fractures': self.fracdist[5],
            'total_upper_ex_fractures': self.fracdist[6],
            'total_lower_ex_fractures': self.fracdist[7]
        }
        logger.info(key='fracture_location_data',
                    data=dict_to_output,
                    description='data on where the fractures occurred on the body')
        # Log where the lacerations are located on the body
        skullopen = ['1101']
        idx, skullopencounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, skullopen)
        self.openwounddist[0] += skullopencounts
        faceopen = ['2101']
        idx, faceopencounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, faceopen)
        self.openwounddist[1] += faceopencounts
        neckopen = ['3101']
        idx, neckopencounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, neckopen)
        self.openwounddist[2] += neckopencounts
        thoraxopen = ['4101']
        idx, thoropencounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, thoraxopen)
        self.openwounddist[3] += thoropencounts
        abdopen = ['5101']
        idx, abdopencounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, abdopen)
        self.openwounddist[4] += abdopencounts
        upperexopen = ['7101']
        idx, upperexopencounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, upperexopen)
        self.openwounddist[6] += upperexopencounts
        lowerexopen = ['8101']
        idx, lowerexopencounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, lowerexopen)
        self.openwounddist[7] += lowerexopencounts
        dict_to_output = {
            'total_head_laceration': self.openwounddist[0],
            'total_facial_laceration': self.openwounddist[1],
            'total_neck_laceration': self.openwounddist[2],
            'total_thorax_laceration': self.openwounddist[3],
            'total_abdomen_laceration': self.openwounddist[4],
            'total_upper_ex_laceration': self.openwounddist[6],
            'total_lower_ex_laceration': self.openwounddist[7]
        }
        logger.info(key='laceration_location_data',
                    data=dict_to_output,
                    description='data on where the lacerations occurred on the body')
        # Log where the burns are located on the body
        burncodes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113']
        idx, skullburncounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, burncodes[0])
        self.burndist[0] = skullburncounts
        idx, faceburncounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, burncodes[1])
        self.burndist[1] = faceburncounts
        idx, neckburncounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, burncodes[2])
        self.burndist[2] = neckburncounts
        idx, thorburncounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, burncodes[3])
        self.burndist[3] = thorburncounts
        idx, abdburncounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, burncodes[4])
        self.burndist[4] = abdburncounts
        idx, upperexburncounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, burncodes[5])
        self.burndist[6] = upperexburncounts
        idx, lowerexburncounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, burncodes[6])
        self.burndist[7] = lowerexburncounts
        dict_to_output = {
            'total_head_laceration': self.burndist[0],
            'total_facial_laceration': self.burndist[1],
            'total_neck_laceration': self.burndist[2],
            'total_thorax_laceration': self.burndist[3],
            'total_abdomen_laceration': self.burndist[4],
            'total_upper_ex_laceration': self.burndist[6],
            'total_lower_ex_laceration': self.burndist[7]
        }
        logger.info(key='burn_location_data',
                    data=dict_to_output,
                    description='data on where the burns occurred on the body')
        # ===================================== Pain severity =========================================================
        # Injuries causing mild pain include: Lacerations, mild soft tissue injuries, TBI (for now), eye injury
        Mild_Pain_Codes = ['1101', '2101', '3101', '4101', '5101', '7101', '8101',  # lacerations
                           '241',  # Minor soft tissue injuries
                           '133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135',  # TBI
                           '291',  # Eye injury
                           ]
        mild_idx, mild_counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, Mild_Pain_Codes)
        # Injuries causing moderate pain include: Fractures, dislocations, soft tissue and neck trauma
        Moderate_Pain_Codes = ['112', '113', '211', '212', '412', '414', '612', '712', '712a', '712b', '712c',
                               '811', '812', '813', '813a', '813b', '813c',  # fractures
                               '322', '323', '722', '822', '822a', '822b',  # dislocations
                               '342', '343', '361', '363',  # neck trauma
                               '813bo', '813co', '813do', '813eo'  # open fractures
                               ]
        moderate_idx, moderate_counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries,
                                                                                          Moderate_Pain_Codes)
        # Injuries causing severe pain include: All burns, amputations, spinal cord injuries, abdominal trauma,
        # severe chest trauma
        Severe_Pain_Codes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113',  # burns
                             '782', '782a', '782b', '782c', '783', '882', '883', '884',  # amputations
                             '673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676',  # sci
                             '552', '553', '554',  # abdominal trauma
                             '461', '463', '453', '453a', '453b', '441', '442', '443',  # severe chest trauma
                             ]
        severe_idx, severe_counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, Severe_Pain_Codes)
        in_severe_pain = severe_idx
        self.severe_pain += len(in_severe_pain)
        in_moderate_pain = moderate_idx.difference(moderate_idx.intersection(severe_idx))
        self.moderate_pain += len(in_moderate_pain)
        in_mild_pain = mild_idx.difference(moderate_idx.union(severe_idx))
        self.mild_pain += len(in_mild_pain)
        dict_to_output = {'mild_pain': mild_counts,
                          'moderate_pain': moderate_counts,
                          'severe_pain': severe_counts}
        logger.info(key='pain_information',
                    data=dict_to_output,
                    description='data on the pain level from injuries in the simulation'
                    )
        # ================================== Injury characteristics ===================================================
        allfraccodes = ['112', '113', '211', '212', '412', '414', '612', '712', '712a', '712b', '712c',
                        '811', '812', '813', '813a', '813b', '813c', '813bo', '813co', '813do', '813eo']
        idx, fraccounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, allfraccodes)
        self.totfracnumber += fraccounts
        n_alive = df.is_alive.sum()
        self.fracdenominator += (n_alive - fraccounts) / 12
        dislocationcodes = ['322', '323', '722', '822', '822a', '822b']
        idx, dislocationcounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, dislocationcodes)
        self.totdisnumber += dislocationcounts
        allheadinjcodes = ['133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135']
        idx, tbicounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, allheadinjcodes)
        self.tottbi += tbicounts
        softtissueinjcodes = ['241', '342', '343', '441', '442', '443']
        idx, softtissueinjcounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, softtissueinjcodes)
        self.totsoft += softtissueinjcounts
        organinjurycodes = ['453', '453a', '453b', '552', '553', '554']
        idx, organinjurycounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, organinjurycodes)
        self.totintorg += organinjurycounts
        internalbleedingcodes = ['361', '363', '461', '463']
        idx, internalbleedingcounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries,
                                                                                        internalbleedingcodes)
        self.totintbled += internalbleedingcounts
        spinalcordinjurycodes = ['673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676']
        idx, spinalcordinjurycounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries,
                                                                                        spinalcordinjurycodes)
        self.totsci += spinalcordinjurycounts
        amputationcodes = ['782', '782a', '782b', '783', '882', '883', '884']
        idx, amputationcounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, amputationcodes)
        self.totamp += amputationcounts
        eyecodes = ['291']
        idx, eyecounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, eyecodes)
        self.toteye += eyecounts
        externallacerationcodes = ['1101', '2101', '3101', '4101', '5101', '7101', '8101']
        idx, externallacerationcounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries,
                                                                                          externallacerationcodes)
        self.totextlac += externallacerationcounts
        burncodes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113']
        idx, burncounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, burncodes)
        self.totburns += burncounts
        totalinj = \
            fraccounts + dislocationcounts + tbicounts + softtissueinjcounts + organinjurycounts + \
            internalbleedingcounts + spinalcordinjurycounts + amputationcounts + externallacerationcounts + burncounts

        dict_to_output = {
            'total_fractures': self.totfracnumber,
            'total_dislocations': self.totdisnumber,
            'total_traumatic_brain_injuries': self.tottbi,
            'total_soft_tissue_injuries': self.totsoft,
            'total_internal_organ_injuries': self.totintorg,
            'total_internal_bleeding': self.totintbled,
            'total_spinal_cord_injuries': self.totsci,
            'total_amputations': self.totamp,
            'total_eye_injuries': self.toteye,
            'total_lacerations': self.totextlac,
            'total_burns': self.totburns,
        }
        logger.info(key='injury_characteristics',
                    data=dict_to_output,
                    description='the injury categories produced in the simulation')
        # ================================= Injury severity ===========================================================
        sev = df.loc[df.rt_road_traffic_inc]
        sev = sev['rt_inj_severity']
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
        ISSlist = thoseininjuries['rt_ISS_score'].tolist()
        ISSlist = list(filter(lambda num: num != 0, ISSlist))
        self.ISSscore += ISSlist
        severity, severitycount = np.unique(sev, return_counts=True)
        if 'mild' in severity:
            idx = np.where(severity == 'mild')
            self.totmild += len(idx)
        if 'severe' in severity:
            idx = np.where(severity == 'severe')
            self.totsevere += len(idx)
        dict_to_output = {
            'total_mild_injuries': self.totmild,
            'total_severe_injuries': self.totsevere,
            'ISS_score': self.ISSscore,
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
        immdeathidx = df.index[df.rt_imm_death]
        deathwithmedidx = df.index[df.rt_post_med_death]
        deathwithoutmedidx = df.index[df.rt_no_med_death]
        diedfromrtiidx = immdeathidx.union(deathwithmedidx)
        diedfromrtiidx = diedfromrtiidx.union(deathwithoutmedidx)
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
        incidence_of_injuries = (totalinj / (n_alive - n_in_RTI)) * 12 * 100000
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

        dict_to_output = {
            'number involved in a rti': n_in_RTI,
            'incidence of rti per 100,000': (n_in_RTI / ((n_alive - n_in_RTI) * (1 / 12))) * 100000,
            'incidence of rti per 100,000 in children': (children_in_RTI /
                                                         ((children_alive - children_in_RTI) * (1 / 12))) * 100000,
            'incidence of rti death per 100,000': (len(diedfromrtiidx) /
                                                   ((n_alive - len(diedfromrtiidx)) * (1 / 12))) * 100000,
            'incidence of death post med per 100,000': (len(df.loc[df.rt_post_med_death]) /
                                                        ((n_alive -
                                                          len(df.loc[df.rt_post_med_death])) * (1 / 12))) * 100000,
            'incidence of prehospital death per 100,000': (len(df.loc[df.rt_imm_death]) /
                                                           ((n_alive -
                                                             len(df.loc[df.rt_imm_death])) * (1 / 12))) * 100000,
            'incidence of death without med per 100,000': (len(df.loc[df.rt_no_med_death]) /
                                                           ((n_alive -
                                                             len(df.loc[df.rt_no_med_death])) * (1 / 12))) * 100000,
            'incidence of death due to unavailable med per 100,000': (len(df.loc[df.rt_unavailable_med_death]) /
                                                                      ((n_alive -
                                                                        len(df.loc[df.rt_unavailable_med_death])) * (
                                                                           1 / 12))) * 100000,
            'incidence of fractures per 100,000': (self.totfracnumber / self.fracdenominator) * 100000,
            'injury incidence per 100,000': incidence_of_injuries,
            'number alive': n_alive,
            'number immediate deaths': n_immediate_death,
            'number deaths post med': n_death_post_med,
            'number deaths without med': len(df.loc[df.rt_no_med_death]),
            'number deaths unavailable med': len(df.loc[df.rt_unavailable_med_death]),
            'number permanently disabled': n_perm_disabled,
            'percent of crashes that are fatal': percent_accidents_result_in_death,
            'total injuries': totalinj,
            'male:female ratio': mfratio,
            'percent sought healthcare': percent_sought_care,
            'percentage died after med': percent_died_post_care,
            'percent admitted to ICU or HDU': percentage_admitted_to_ICU_or_HDU,
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
