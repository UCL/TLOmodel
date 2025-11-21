from pathlib import Path
# from typing import Union

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Population, Property, Simulation, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.hsi_generic_first_appts import HSIEventScheduler
# from tlo.methods.symptommanager import Symptom
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
        "rate_onset_to_stage1_4": Parameter(Types.REAL,
                                            "Probability of people who get diagnosed with CKD stage 1-4"),
        "rate_stage1_4_to_stage5": Parameter(Types.REAL,
                                             "Probability of people who get diagnosed with stage 5 (ESRD)"),
        "init_prob_any_ckd": Parameter(Types.LIST, "Initial probability of anyone with CKD"),
        "rp_ckd_nc_diabetes": Parameter(
            Types.REAL, "relative prevalence at baseline of CKD if person has diabetes"
        ),
        "rp_ckd_hiv_infection": Parameter(
            Types.REAL, "relative prevalence at baseline of CKD if person has an HIV infection"
        ),
        "rp_ckd_li_bmi": Parameter(
            Types.REAL, "relative prevalence at baseline of CKD if person li_bmi is true"
        ),
        "rp_ckd_nc_hypertension": Parameter(
            Types.REAL, "relative prevalence at baseline of CKD if person has hypertension"
        ),
        "rp_ckd_nc_chronic_ischemic_hd": Parameter(
            Types.REAL, "relative prevalence at baseline of CKD if person has heart disease"
        ),
        "prob_ckd_renal_clinic": Parameter(
            Types.REAL, "Proportion of eligible population that goes for Renal Clinic & Medication"
        ),
        "prob_ckd_transplant_eval": Parameter(
            Types.REAL, "Proportion of eligible population that goes for Kidney Transplant Evaluation"
        ),
        "prop_herbal_use_ckd": Parameter(
            Types.REAL, "Proportion of people with ckd who use herbal medicine"
        ),
        "rp_ckd_herbal_use_baseline": Parameter(
            Types.REAL, "Relative risk of having CKD at baseline if a person uses herbal medicine"
        ),
    }

    PROPERTIES = {
        "ckd_status": Property(
            Types.CATEGORICAL,
            "CKD status",
            categories=["pre_diagnosis", "stage1_4", "stage5"],
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
            categories=["pre_diagnosis", "stage1_4", "stage5"]
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
        "uses_herbal_medicine": Property(
            Types.BOOL, "Whether this person uses herbal medicine"
        ),
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
        self.parameters['rate_onset_to_stage1_4'] = 0.29
        self.parameters['rate_stage1_4_to_stage5'] = 0.4

        self.parameters['init_prob_any_ckd'] = [0.6, 0.4]

        self.parameters['rp_ckd_nc_diabetes'] = 1.1
        self.parameters['rp_ckd_hiv_infection'] = 1.2
        self.parameters['rp_ckd_li_bmi'] = 1.3
        self.parameters['rp_ckd_nc_hypertension'] = 1.3
        self.parameters['rp_ckd_nc_chronic_ischemic_hd'] = 1.2
        self.parameters['rp_ckd_herbal_use_baseline'] = 1.35

        #
        self.parameters['prob_ckd_renal_clinic'] = 0.7
        self.parameters['prob_ckd_transplant_eval'] = 0.3

        self.parameters['prop_herbal_use_ckd'] = 0.35



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

        # write to property:
        df.loc[df.is_alive & ~df.nc_chronic_kidney_disease, 'ckd_status'] = 'pre_diagnosis'

        df.loc[list(alive_ckd_idx), "ckd_on_treatment"] = False
        df.loc[list(alive_ckd_idx), "ckd_diagnosed"] = False
        df.loc[list(alive_ckd_idx), "ckd_date_treatment"] = pd.NaT
        df.loc[list(alive_ckd_idx), "ckd_stage_at_which_treatment_given"] = "pre_diagnosis"
        df.loc[list(alive_ckd_idx), "ckd_date_diagnosis"] = pd.NaT

        df.loc[list(alive_ckd_idx), "uses_herbal_medicine"] = \
            self.rng.random(len(alive_ckd_idx)) < self.parameters['prop_herbal_use_ckd']
        df.loc[list(df.loc[df.is_alive & ~df.nc_chronic_kidney_disease].index),
        "uses_herbal_medicine"] = False

        # -------------------- ckd_status -----------
        # Determine who has CKD at all stages:
        # check parameters are sensible: probability of having any CKD stage cannot exceed 1.0
        assert sum(self.parameters['init_prob_any_ckd']) <= 1.0

        lm_init_ckd_status_any_ckd = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            sum(self.parameters['init_prob_any_ckd']),
            Predictor('nc_diabetes').when(True, self.parameters['rp_ckd_nc_diabetes']),
            Predictor('hv_inf').when(True, self.parameters['rp_ckd_hiv_infection']),
            Predictor('li_bmi').when(True, self.parameters['rp_ckd_li_bmi']),
            Predictor('nc_hypertension').when(True, self.parameters['rp_ckd_nc_hypertension']),
            Predictor('nc_chronic_ischemic_hd').when(True, self.parameters['rp_ckd_nc_chronic_ischemic_hd']),
            Predictor('uses_herbal_medicine').when(True, self.parameters['rp_ckd_herbal_use_baseline']),

        )

        # Get boolean Series of who has ckd
        has_ckd = lm_init_ckd_status_any_ckd.predict(df.loc[df.is_alive & df.nc_chronic_kidney_disease], self.rng)

        # Get indices of those with CKD
        ckd_idx = has_ckd[has_ckd].index if has_ckd.any() else pd.Index([])

        if not ckd_idx.empty:
            # Get non-pre_diagnosis categories
            categories = [cat for cat in df.ckd_status.cat.categories if cat != 'pre_diagnosis']

            # Verify probabilities match categories
            assert len(categories) == len(self.parameters['init_prob_any_ckd'])

            # Normalize probabilities
            total_prob = sum(self.parameters['init_prob_any_ckd'])
            probs = [p / total_prob for p in self.parameters['init_prob_any_ckd']]

            # Assign CKD stages
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
        self.sim.population.props.at[child_id, 'ckd_status'] = 'pre_diagnosis'
        self.sim.population.props.at[child_id, 'ckd_on_treatment'] = False
        self.sim.population.props.at[child_id, 'ckd_date_treatment'] = pd.NaT
        self.sim.population.props.at[child_id, 'ckd_stage_at_which_treatment_given'] = 'pre_diagnosis'
        self.sim.population.props.at[child_id, 'ckd_diagnosed'] = False
        self.sim.population.props.at[child_id, 'ckd_date_diagnosis'] = pd.NaT

    def on_simulation_end(self) -> None:
        pass

    def make_the_linear_models(self) -> None:
        """Make and save LinearModels that will be used when the module is running"""
        self.lm = dict()

        self.lm['onset_stage1_4'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            self.parameters['rate_onset_to_stage1_4'],
            Predictor('uses_herbal_medicine')
            .when(True, self.parameters['rp_ckd_herbal_use_baseline'])
        )

        self.lm['stage1_to_4_stage5'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            self.parameters['rate_stage1_4_to_stage5'],
            Predictor('had_treatment_during_this_stage', external=True)
            .when(True, 0.0).otherwise(1.0)
        )

    def look_up_consumable_item_codes(self):
        """Look up the item codes used in the HSI of this module"""
        get_item_codes = self.sim.modules['HealthSystem'].get_item_code_from_item_name

        self.cons_item_codes = dict()
        self.cons_item_codes['renal_consumables'] = {
            get_item_codes("Sterile syringe"): 1,
            get_item_codes('Sterile drapes and supplies'): 3,
            get_item_codes('Gloves, exam, latex, disposable, pair'): 4,
            get_item_codes("Catheter"): 1,
            get_item_codes("Disinfectant"): 1
        },
        self.cons_item_codes['kidney_transplant_eval_cons'] = {
            get_item_codes("Blood collection tube"): 1,
            get_item_codes('Reagents'): 3,
            get_item_codes('Radiopharmaceuticals'): 4
        }



class CMDChronicKidneyDiseasePollEvent(RegularEvent, PopulationScopeEventMixin):
    """An event that controls the development process of Chronic Kidney Disease (CKD) and logs current states."""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population: Population) -> None:
        df = population.props

        had_treatment_during_this_stage = \
            df.is_alive & ~pd.isnull(df.ckd_date_treatment) & \
            (df.ckd_status == df.ckd_stage_at_which_treatment_given)

        ckd_and_alive_prediagnosis = (
            df.loc)[df.is_alive & df.nc_chronic_kidney_disease & (df.ckd_status == 'pre_diagnosis')]
        ckd_and_alive_stage1_to_4 = df.loc[df.is_alive & df.nc_chronic_kidney_disease & (df.ckd_status == 'stage1_4')]

        will_progress = self.module.lm['onset_stage1_4'].predict(ckd_and_alive_prediagnosis, self.module.rng)
        will_progress_idx = df.index[np.where(will_progress)[0]]
        df.loc[will_progress_idx, 'ckd_status'] = 'stage1_4'

        stage1_to_4_to_stage5 = self.module.lm['stage1_to_4_stage5'].predict(
            ckd_and_alive_stage1_to_4,
            self.module.rng,
            had_treatment_during_this_stage=had_treatment_during_this_stage)
        stage1_to_4_to_stage5_idx = df.index[np.where(stage1_to_4_to_stage5)[0]]
        df.loc[stage1_to_4_to_stage5_idx, 'ckd_status'] = 'stage5'

        # ----------------------------SELECTING INDIVIDUALS FOR CKD Diagnosis by stage----------------------------#

        eligible_population = (
            (df.is_alive & df.nc_chronic_kidney_disease) &
            (df.ckd_status == 'pre_diagnosis') &
            (df.age_years >= 20) &
            (pd.isna(df.ckd_date_diagnosis))
        )

        eligible_idx = df.loc[eligible_population].index

        if not eligible_idx.empty:
            probs = [
                self.module.parameters['prob_ckd_renal_clinic'],
                self.module.parameters['prob_ckd_transplant_eval']
            ]

            hsi_choices = self.module.rng.choice(
                ['renal_clinic', 'transplant_eval'],
                size=len(eligible_idx),
                p=probs
            )

            for person_id, hsi_choice in zip(eligible_idx, hsi_choices):
                if hsi_choice == 'renal_clinic':
                    self.sim.modules['HealthSystem'].schedule_hsi_event(
                        hsi_event=HSI_Renal_Clinic_and_Medication(self.module, person_id),
                        priority=1,
                        topen=self.sim.date,
                        tclose=self.sim.date + DateOffset(months=1),
                    )
                elif hsi_choice == 'transplant_eval':
                    self.sim.modules['HealthSystem'].schedule_hsi_event(
                        hsi_event=HSI_Kidney_Transplant_Evaluation(self.module, person_id),
                        priority=1,
                        topen=self.sim.date,
                        tclose=self.sim.date + DateOffset(months=1),
                    )

    def do_at_generic_first_appt(
        self,
        person_id: int,
        individual_properties: IndividualProperties,
        schedule_hsi_event: HSIEventScheduler,
        **kwargs,
    ) -> None:
        pass


class HSI_Renal_Clinic_and_Medication(HSI_Event, IndividualScopeEventMixin):
    """
    This is the event when a person undergoes the optical coherence topography before being given the anti-vegf
    injection. Given to individuals with dr_status of severe and proliferative
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CMDChronicKidneyDisease)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Dr_CMD_Renal_Medication'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1, 'NewAdult': 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'  #todo Facility Level?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(key='debug',
                     data=f'This is HSI_Renal_Clinic_and_Medication for person {person_id}')
        df = self.sim.population.props
        person = df.loc[person_id]
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            # The person is not alive, the event did not happen: so return a blank footprint
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        # if person already on treatment or not yet diagnosed, do nothing
        if person["ckd_on_treatment"] or not person["ckd_diagnosed"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        assert pd.isnull(df.at[person_id, 'ckd_date_treatment'])

        is_cons_available = self.get_consumables(
            self.module.cons_item_codes['renal_consumables']
        )

        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='renal_clinic_test',
            hsi_event=self
        )

        if dx_result and is_cons_available:
            # record date of diagnosis
            df.at[person_id, 'ckd_date_diagnosis'] = self.sim.date
            df.at[person_id, 'ckd_date_treatment'] = self.sim.date
            df.at[person_id, 'ckd_stage_at_which_treatment_given'] = df.at[person_id, 'ckd_status']

            next_session = self.sim.date + pd.DateOffset(months=1)
            self.sim.modules['HealthSystem'].schedule_hsi_event(self,
                                                                topen=next_session,
                                                                tclose=None,
                                                                priority=1)


class HSI_Kidney_Transplant_Evaluation(HSI_Event, IndividualScopeEventMixin):
    """
    This is the event a person undergoes in order to determine whether an individual is eligible for a kidney transplant
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CMDChronicKidneyDisease)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Dr_CMD_Kidney_Transplant_Evaluation'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1, 'NewAdult': 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(key='debug',
                     data=f'This is HSI_Kidney_Transplant_Evaluation for person {person_id}')
        df = self.sim.population.props
        person = df.loc[person_id]
        hs = self.sim.modules["HealthSystem"]
        if not df.at[person_id, 'is_alive']:
            # The person is not alive, the event did not happen: so return a blank footprint
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        # if person already on treatment or not yet diagnosed, do nothing
        if person["ckd_on_treatment"] or not person["ckd_diagnosed"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        assert pd.isnull(df.at[person_id, 'ckd_date_treatment'])

        is_cons_available = self.get_consumables(
            self.module.cons_item_codes['kidney_transplant_eval_cons']
        )

        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='kidney_transplant_eval_tests',
            hsi_event=self
        )

        if dx_result and is_cons_available:
            # record date of diagnosis
            df.at[person_id, 'ckd_date_diagnosis'] = self.sim.date
            df.at[person_id, 'ckd_date_treatment'] = self.sim.date
            df.at[person_id, 'ckd_stage_at_which_treatment_given'] = df.at[person_id, 'ckd_status']
            

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
