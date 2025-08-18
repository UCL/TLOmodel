from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Population, Property, Simulation, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, cardio_metabolic_disorders
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.hsi_generic_first_appts import HSIEventScheduler
from tlo.methods.symptommanager import Symptom
from tlo.population import IndividualProperties

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CMDChronicKidneyDisease(Module):
    """This is the Chronic Kidney Disease module for kidney transplants."""

    INIT_DEPENDENCIES = {'SymptomManager', 'Lifestyle', 'HealthSystem', 'CardioMetabolicDisorders'}
    ADDITIONAL_DEPENDENCIES = set()

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN,
    }

    PARAMETERS = {
        "rate_onset_to_stage1-4": Parameter(Types.REAL,
                                            "Probability of people who get diagnosed with CKD stage 1-4"),
        "rate_stage1-4_to_stage5": Parameter(Types.REAL,
                                             "Probability of people who get diagnosed with stage 5 (ESRD)"),
        "init_prob_any_ckd": Parameter(Types.LIST, "Initial probability of anyone with diabetic retinopathy"),
        "rp_ckd_ex_alc": Parameter(
            Types.REAL, "relative prevalence at baseline of CKD if excessive alcohol"
        ),
        "rp_ckd_high_sugar": Parameter(
            Types.REAL, "relative prevalence at baseline of CKD if high sugar"
        ),
        "rp_ckd_low_ex": Parameter(
            Types.REAL, "relative prevalence at baseline of CKD if low exercise"
        ),
    }

    PROPERTIES = {
        "ckd_status": Property(
            Types.CATEGORICAL,
            "CKD status",
            categories=["none", "stage1-4", "stage5"],
        ),
        "ckd_on_treatment": Property(
            Types.BOOL, "Whether this person is on CKD treatment",
        ),
        "ckd_date_treatment": Property(
            Types.DATE,
            "date of first receiving CKD treatment (pd.NaT if never started treatment)"
        ),
        "ckd_stage_at_which_treatment_given": Property(
            Types.CATEGORICAL,
            "The CKD stage at which treatment was given (used to apply stage-specific treatment effect)",
            categories=["stage1-4", "stage5", "chronic_dialysis"]
        ),
        "ckd_diagnosed": Property(
            Types.BOOL, "Whether this person has been diagnosed with any CKD"
        ),
        "ckd_date_diagnosis": Property(
            Types.DATE,
            "The date of diagnosis of CKD (pd.NaT if never diagnosed)"
        ),
        "ckd_total_dialysis_sessions": Property(Types.INT,
                                                "total number of dialysis sessions the person has ever had"),
    }

    def __init__(self):
        super().__init__()
        self.cons_item_codes = None  # (Will store consumable item codes)

    def read_parameters(self, data_folder: str | Path) -> None:
        """ initialise module parameters. Here we are assigning values to all parameters defined at the beginning of
        this module.

        :param data_folder: Path to the folder containing parameter values

        """
        # TODO Read from resourcefile
        self.parameters['rate_onset_to_stage1-4'] = 0.29
        self.parameters['rate_stage1-4_to_stage5'] = 0.4

        self.parameters['init_prob_any_ckd'] = [0.5, 0.3, 0.2]

        self.parameters['rp_ckd_ex_alc'] = 1.1
        self.parameters['rp_ckd_high_sugar'] = 1.2
        self.parameters['rp_ckd_low_ex'] = 1.3

        #todo use chronic_kidney_disease_symptoms from cardio_metabolic_disorders.py

        # self.sim.modules['SymptomManager'].register_symptom(
        #     Symptom(name='blindness_partial'),
        #     Symptom(name='blindness_full')
        # )

    def initialise_population(self, population: Population) -> None:
        """ Set property values for the initial population

        :param population: all individuals in the model

        """
        df = population.props
        # p = self.parameters

        alive_ckd_idx = df.loc[df.is_alive & df.nc_chronic_kidney_disease].index

        # any_dr_idx = alive_diabetes_idx[
        #     self.rng.random_sample(size=len(alive_diabetes_idx)) < self.parameters['init_prob_any_dr']
        #     ]
        # no_dr_idx = set(alive_diabetes_idx) - set(any_dr_idx)
        #
        # proliferative_dr_idx = any_dr_idx[
        #     self.rng.random_sample(size=len(any_dr_idx)) < self.parameters['init_prob_proliferative_dr']
        #     ]
        #
        # mild_dr_idx = set(any_dr_idx) - set(proliferative_dr_idx)

        # write to property:
        df.loc[df.is_alive & ~df.nc_chronic_kidney_disease, 'ckd_status'] = 'none'

        df.loc[list(alive_ckd_idx), "ckd_on_treatment"] = False
        df.loc[list(alive_ckd_idx), "ckd_diagnosed"] = False
        df.loc[list(alive_ckd_idx), "ckd_date_treatment"] = pd.NaT
        df.loc[list(alive_ckd_idx), "ckd_stage_at_which_treatment_given"] = "none"
        df.loc[list(alive_ckd_idx), "ckd_date_diagnosis"] = pd.NaT

        # -------------------- ckd_status -----------
        # Determine who has CKD at all stages:
        # check parameters are sensible: probability of having any CKD stage cannot exceed 1.0
        assert sum(self.parameters['init_prob_any_ckd']) <= 1.0

        lm_init_ckd_status_any_ckd = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            sum(self.parameters['init_prob_any_ckd']),
            Predictor('li_ex_alc').when(True, self.parameters['rp_ckd_ex_alc']),
            Predictor('li_high_sugar').when(True, self.parameters['rp_ckd_high_sugar']),
            Predictor('li_low_ex').when(True, self.parameters['rp_ckd_low_ex']),
        )

        # Get boolean Series of who has ckd
        has_ckd = lm_init_ckd_status_any_ckd.predict(df.loc[df.is_alive & df.nc_chronic_kidney_disease], self.rng)

        # Get indices of those with DR
        ckd_idx = has_ckd[has_ckd].index if has_ckd.any() else pd.Index([])

        if not ckd_idx.empty:
            # Get non-none categories
            categories = [cat for cat in df.ckd_status.cat.categories if cat != 'none']

            # Verify probabilities match categories
            assert len(categories) == len(self.parameters['init_prob_any_ckd'])

            # Normalize probabilities
            total_prob = sum(self.parameters['init_prob_any_ckd'])
            probs = [p / total_prob for p in self.parameters['init_prob_any_ckd']]

            # Assign DR stages
            df.loc[ckd_idx, 'ckd_status'] = self.rng.choice(
                categories,
                size=len(ckd_idx),
                p=probs
            )

    def initialise_simulation(self, sim: Simulation) -> None:
        """ This is where you should include all things you want to be happening during simulation
        * Schedule the main polling event
        * Schedule the main logging event
        * Call the LinearModels
        """
        sim.schedule_event(CMDChronicKidneyDiseasePollEvent(self), date=sim.date)
        sim.schedule_event(CMDChronicKidneyDiseaseLoggingEvent(self), sim.date + DateOffset(months=1))
        self.make_the_linear_models()
        self.look_up_consumable_item_codes()

    def report_daly_values(self) -> pd.Series:
        return pd.Series(index=self.sim.population.props.index, data=0.0)

    def on_birth(self, mother_id: int, child_id: int) -> None:
        """ Set properties of a child when they are born.
        :param child_id: the new child
        """
        self.sim.population.props.at[child_id, 'ckd_status'] = 'none'
        self.sim.population.props.at[child_id, 'ckd_on_treatment'] = False
        self.sim.population.props.at[child_id, 'ckd_date_treatment'] = pd.NaT
        self.sim.population.props.at[child_id, 'ckd_stage_at_which_treatment_given'] = 'none'
        self.sim.population.props.at[child_id, 'ckd_diagnosed'] = False
        self.sim.population.props.at[child_id, 'ckd_date_diagnosis'] = pd.NaT

    def on_simulation_end(self) -> None:
        pass

    def make_the_linear_models(self) -> None:
        """Make and save LinearModels that will be used when the module is running"""
        pass

    def look_up_consumable_item_codes(self):
        """Look up the item codes used in the HSI of this module"""
        pass


class CMDChronicKidneyDiseasePollEvent(RegularEvent, PopulationScopeEventMixin):
    """An event that controls the development process of Diabetes Retinopathy (DR) and logs current states. DR diagnosis
    begins at least after 3 years of being infected with Diabetes Mellitus."""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population: Population) -> None:
        pass

    def do_at_generic_first_appt(
        self,
        person_id: int,
        individual_properties: IndividualProperties,
        schedule_hsi_event: HSIEventScheduler,
        **kwargs,
    ) -> None:
        pass


class CMDChronicKidneyDiseaseLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """The only logging event for this module"""

    def __init__(self, module):
        """schedule logging to repeat every 1 month
        """
        self.repeat = 30
        super().__init__(module, frequency=DateOffset(days=self.repeat))

    def apply(self, population):
        """Compute statistics regarding the current status of persons and output to the logger
        """
        pass
