from pathlib import Path
# from typing import Union

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Population, Property, Simulation, Types, logging
from collections import deque
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
        "prob_transplant_success": Parameter(
            Types.REAL, "Probability that kidney transplant surgery is successful"
        ),
        "max_surgeries_per_month": Parameter(
            Types.INT,
            "Maximum number of kidney transplant surgeries that can be performed in a month"
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
        "nc_ckd_total_dialysis_sessions": Property(Types.INT,
                                                   "total number of dialysis sessions the person has ever had"),
    }

    def __init__(self):
        super().__init__()
        self.cons_item_codes = None  # (Will store consumable item codes)
        self.kidney_transplant_waiting_list = deque()
        self.daly_wts = dict()

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

        self.parameters['prob_transplant_success'] = 0.8
        self.parameters['max_surgeries_per_month'] = 3

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
        df.loc[list(alive_ckd_idx), "nc_ckd_total_dialysis_sessions"] = 0

        df.loc[list(alive_ckd_idx), "uses_herbal_medicine"] = \
            self.rng.random(len(alive_ckd_idx)) < self.parameters['prop_herbal_use_ckd']
        df.loc[list(df.loc[df.is_alive & ~df.nc_chronic_kidney_disease].index), "uses_herbal_medicine"] = False

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

        # ----- DISABILITY-WEIGHTS -----
        if "HealthBurden" in self.sim.modules:
            # For those with End-stage renal disease, with kidney transplant (any stage after stage1_4)
            self.daly_wts["stage5_ckd"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=977
            )

    def report_daly_values(self):
        # return pd.Series(index=self.sim.population.props.index, data=0.0)
        df = self.sim.population.props  # shortcut to population properties dataframe for alive persons

        disability_series_for_alive_persons = pd.Series(index=df.index[df.is_alive], data=0.0)

        # Assign daly_wt to those with CKD stage stage5 and have had a kidney transplant
        disability_series_for_alive_persons.loc[
            (df.ckd_status == "stage5") &
            (~pd.isnull(df.ckd_date_transplant))
            ] = self.daly_wts['stage5_ckd']

        return disability_series_for_alive_persons

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
        self.sim.population.props.at[child_id, 'nc_ckd_total_dialysis_sessions'] = 0

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

        self.cons_item_codes['kidney_transplant_surgery_cons'] = {
            # Prepare surgical instruments
            # administer an IV
            get_item_codes('Cannula iv  (winged with injection pot) 18_each_CMST'): 1,
            get_item_codes("Giving set iv administration + needle 15 drops/ml_each_CMST"): 1,
            get_item_codes("ringer's lactate (Hartmann's solution), 1000 ml_12_IDA"): 2000,
            # request a general anaesthetic
            get_item_codes("Halothane (fluothane)_250ml_CMST"): 100,
            # Position patient in lateral position with kidney break
            # clean the site of the surgery
            get_item_codes("Chlorhexidine 1.5% solution_5_CMST"): 600,
            # tools to begin surgery
            get_item_codes("Scalpel blade size 22 (individually wrapped)_100_CMST"): 1,
            # repair incision made
            get_item_codes("Suture pack"): 1,
            get_item_codes("Gauze, absorbent 90cm x 40m_each_CMST"): 100,
            # administer pain killer
            get_item_codes('Pethidine, 50 mg/ml, 2 ml ampoule'): 6,
            # administer antibiotic
            get_item_codes("Ampicillin injection 500mg, PFR_each_CMST"): 2,
            #change this to immunosuppressive drugs
            # equipment used by surgeon, gloves and facemask
            get_item_codes('Disposables gloves, powder free, 100 pieces per box'): 1,
            get_item_codes('surgical face mask, disp., with metal nose piece_50_IDA'): 1,
            # request syringe
            get_item_codes("Syringe, Autodisable SoloShot IX "): 1
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
            #todo stage1_4 should go to renal_clinic and stage5 should go to transplant_eval and/or haemodialysis
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

        surgeries_done = 0

        while (
            surgeries_done < self.module.parameters['max_surgeries_per_month']
            and len(self.module.kidney_transplant_waiting_list) > 0
        ):
            next_person = self.module.kidney_transplant_waiting_list.popleft()

            # Skip if dead
            if not population.props.at[next_person, 'is_alive']:
                continue

            # Schedule the transplant
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Kidney_Transplant_Surgery(self.module, next_person),
                priority=1,
                topen=self.sim.date,
                tclose=self.sim.date + DateOffset(months=1)
            )
            # increments until max_surgeries_per_month is reached
            surgeries_done += 1

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
    This is an event where a CKD patient is managed conservatively; attending clinic monthly for symptom management"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CMDChronicKidneyDisease)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'CKD_Renal_Medication'
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


class HSI_Haemodialysis_Refill(HSI_Event, IndividualScopeEventMixin):
    """This is an event in which a person goes for dialysis sessions 2 times a week
    adding up to 8 times a month."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'CKD_Treatment_Haemodialysis'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        if not df.at[person_id, 'is_alive'] or not df.at[person_id, 'nc_chronic_kidney_disease']:
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        # Increment total number of dialysis sessions the person has ever had in their lifetime
        df.at[person_id, 'nc_ckd_total_dialysis_sessions'] += 1

        self.add_equipment({'Chair', 'Dialysis Machine', 'Dialyser (Artificial Kidney)',
                            'Bloodlines', 'Dialysate solution', 'Dialysis water treatment system'})

        next_session_date = self.sim.date + pd.DateOffset(days=3)
        self.sim.modules['HealthSystem'].schedule_hsi_event(self,
                                                            topen=next_session_date,
                                                            tclose=next_session_date + pd.DateOffset(days=1),
                                                            priority=1
                                                            )

    def never_ran(self) -> None:
        """What to do if the event is never run by the HealthSystem"""
        # Reschedule this HSI to happen again 3 days time.
        next_session_date = self.sim.date + pd.DateOffset(days=3)
        self.sim.modules['HealthSystem'].schedule_hsi_event(self,
                                                            topen=next_session_date,
                                                            tclose=next_session_date + pd.DateOffset(days=1),
                                                            priority=1
                                                            )


class HSI_Kidney_Transplant_Evaluation(HSI_Event, IndividualScopeEventMixin):
    """
    This is the event a person undergoes in order to determine whether an individual is eligible for a kidney transplant
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CMDChronicKidneyDisease)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'CKD_Kidney_Transplant_Evaluation'
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

            # Append to waiting list if treatment is successful
            self.module.kidney_transplant_waiting_list.append(person_id)

        else:
            hs.schedule_hsi_event(
                HSI_Haemodialysis_Refill(self.module, person_id),
                topen=self.sim.date,
                tclose=self.sim.date + pd.DateOffset(days=1),
                priority=1
            )


class HSI_Kidney_Transplant_Surgery(HSI_Event, IndividualScopeEventMixin):
    """
    This is the event a person undergoes in order to determine whether an individual is eligible for a kidney transplant
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CMDChronicKidneyDisease)

        # Define the necessary information for an HSI
        # todo need to update priority number in resource files
        self.TREATMENT_ID = 'CKD_Kidney_Transplant'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1, 'NewAdult': 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(key='debug',
                     data=f'This is HSI_Kidney_Transplant_Surgery for person {person_id}')
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
            self.module.cons_item_codes['kidney_transplant_surgery_cons']
        )

        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='kidney_transplant_surgery_tests',
            hsi_event=self
        )

        # Check if transplant is successful
        transplant_successful = self.module.rng.random() < self.module.parameters['prob_transplant_success']

        if dx_result and is_cons_available:
            self.add_equipment({'Patient monitors', 'Infusion pump', 'Dialysis machine', 'Bloodlines', 'Water tank',
                                'Reverse osmosis machine', 'Water softener', 'Carbon filter', '5 micro filter',
                                'ventilator', 'Electrocautery unit', 'Suction machine', 'theatre bed',
                                'cold static storage' 'perfusion machine', 'Ultrasound machine', 'drip stand',
                                'trolley', })

        if transplant_successful:
            # df.at[person_id, 'ckd_transplant_successful'] = True
            df.at[person_id, 'ckd_date_transplant'] = self.sim.date
            # df.at[person_id, 'ckd_on_anti_rejection_drugs'] = True
            # df.at[person_id, 'ckd_on_transplant_waiting_list'] = False
            # df.at[person_id, 'ckd_stage_at_which_treatment_given'] = df.at[person_id, 'ckd_status']

            # Schedule monthly anti-rejection drug refill
            next_month = self.sim.date + pd.DateOffset(months=1)
            hs.schedule_hsi_event(
                HSI_AntiRejectionDrug_Refill(self.module, person_id),
                topen=next_month,
                tclose=next_month + pd.DateOffset(days=5),
                priority=1
            )

        else:
            df.at[person_id, 'ckd_date_transplant'] = self.sim.date

            self.sim.modules['Demography'].do_death(
                individual_id=person_id,
                cause='chronic_kidney_disease_death',
                originating_module=self.module
            )


class HSI_AntiRejectionDrug_Refill(HSI_Event, IndividualScopeEventMixin):
    """This is an event where a kidney transplant recipient gets drugs every month for the rest of their lives """

    def __init__(self, module, person_id):
        super().__init__(module, person_id)
        self.TREATMENT_ID = 'CKD_AntiRejectionDrug_Refill'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '2'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        if not df.at[person_id, 'is_alive']:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        # Person stays on anti-rejection drugs for life
        next_month = self.sim.date + pd.DateOffset(months=1)
        self.sim.modules["HealthSystem"].schedule_hsi_event(
            self,
            topen=next_month,
            tclose=next_month + pd.DateOffset(days=5),
            priority=1
        )


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
