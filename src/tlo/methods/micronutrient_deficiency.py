"""
Micronutrient Deficiency module
Documentation: '04 - Methods Repository/Undernutrition module - Description.docx'

Overview
=======
This module is apply the prevalence of micronutrient deficiency levels at the population-level

"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, demography
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------


class MicronutrientDeficiency(Module):
    """
    This module applies the levels of micronutrients deficiency at the population-level,
    based on the Malawi Micronutrient Survey 2015-2016.
    The micronutrients included are:
    - iron deficiency
    - zinc deficiency
    - vitamin A deficiency
    - iodine deficiency
    - selenium deficiency ??
    - folic acid and B12 status ??

    Applied to the age groups:
    - Preschool children 9age 6-59 months)
    - School-aged children (5-14 yo)
    - Women of reproductive age (15-49 yo)
    - Men (20-55 yo)

    """

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    PARAMETERS = {
        # Vitamin A deficiency
        'vitamin_A_deficiency_PSC': Parameter(
            Types.REAL, 'prevalence of vitamin A deficiency (low retinol binding protein) in preschool children'),
        'vitamin_A_deficiency_SAC': Parameter(
            Types.REAL, 'prevalence of vitamin A deficiency (low retinol binding protein) in school-aged children'),
        'vitamin_A_deficiency_nonpreg_women': Parameter(
            Types.REAL, 'prevalence of vitamin A deficiency (low retinol binding protein) in non-pregnant women'),
        'vitamin_A_deficiency_men': Parameter(
            Types.REAL, 'prevalence of vitamin A deficiency (low retinol binding protein) in men'),
        # Zinc deficiency
        'zinc_deficiency_PSC': Parameter(
            Types.REAL, 'prevalence of zinc deficiency in preschool children'),
        'zinc_deficiency_SAC': Parameter(
            Types.REAL, 'prevalence of zinc deficiency in school-aged children'),
        'zinc_deficiency_nonpreg_women': Parameter(
            Types.REAL, 'prevalence of zinc deficiency in non-pregnant women'),
        'zinc_deficiency_men': Parameter(
            Types.REAL, 'prevalence of zinc deficiency in men')


    }

    PROPERTIES = {

    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        p = self.parameters

        p['vitamin_A_deficiency_PSC'] = 0.036  # 3.6%
        p['vitamin_A_deficiency_PAC'] = 0.009  # 0.9%
        p['vitamin_A_deficiency_nonpreg_women'] = 0.003  # 0.3%
        p['vitamin_A_deficiency_men'] = 0.001  # 0.1%
        p['zinc_deficiency_PSC'] = 0.60
        p['zinc_deficiency_SAC'] = 0.60  # note stat significant difference in SAC - 5-10y 56.9%, 11-14y 66.4%
        p['zinc_deficiency_nonpreg_women'] = 0.63
        p['zinc_deficiency_men'] = 0.66

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother_id, child_id):
        pass
