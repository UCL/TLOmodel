from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Population, Property, Simulation, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent, Event
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, cardio_metabolic_disorders
from tlo.methods.dxmanager import DxTest
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.hsi_generic_first_appts import HSIEventScheduler
# from tlo.methods.symptommanager import Symptom
from tlo.population import IndividualProperties
from tlo.util import read_csv_files

# from typing import Union


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CMDChronicKidneyDisease(Module):
    """This is the Chronic Kidney Disease module for kidney transplants."""

    INIT_DEPENDENCIES = {'SymptomManager', 'Lifestyle', 'HealthSystem', 'CardioMetabolicDisorders'}

    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden', 'Hiv', 'Tb', 'Epi'}

    ADDITIONAL_DEPENDENCIES = {'Depression'}

    # ADDITIONAL_DEPENDENCIES = set()

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
        "prob_staging_referral": Parameter(
            Types.REAL,
            "Proportion of eligible population that gets referred for CKD staging"
        ),
        "prob_stage5": Parameter(
            Types.REAL,
            "Proportion of eligible population referred for CKD staging that gets CKD stage 5"
        ),
        "sensitivity_of_ckd_staging_test": Parameter(
            Types.REAL, "sensitivity of psa staging test for CKD"
        ),
        "sensitivity_of_renal_clinic_test": Parameter(
            Types.REAL, "sensitivity of renal clinic test for CKD"
        ),
        "sensitivity_of_kidney_transplant_eval_tests": Parameter(
            Types.REAL, "sensitivity of kidney transplant evaluation tests for CKD"
        ),
        "sensitivity_of_kidney_transplant_surgery_tests": Parameter(
            Types.REAL, "sensitivity of kidney transplant surgery tests for CKD"
        ),
        "prob_dialysis_death_1_year": Parameter(
            Types.REAL,
            "Probability of death for those who have been on dialysis for at least 1 year"
        ),
        "prob_dialysis_death_5_years": Parameter(
            Types.REAL,
            "Probability of death for those who have been on dialysis for at least 5 year"
        ),
        "prob_dialysis_death_10_years": Parameter(
            Types.REAL,
            "Probability of death for those who have been on dialysis for at least 10 year"
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
                                                   "Total number of dialysis sessions the person has ever had"),
        "nc_ckd_on_dialysis": Property(Types.BOOL,
                                       "Whether this person is on dialysis"),
        "ckd_date_transplant": Property(
            Types.DATE,
            "The date of kidney transplant (pd.NaT if never transplanted)"
        ),
        "ckd_death_event_scheduled": Property(Types.BOOL,
            "Whether death event has been been scheduled for person"),

    }

    def __init__(self):
        super().__init__()
        self.cons_item_codes = None  # (Will store consumable item codes)
        self.kidney_transplant_waiting_list = deque()
        self.daly_wts = dict()

    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        """Setup parameters used by the module"""

        self.load_parameters_from_dataframe(
            read_csv_files(resourcefilepath / "ResourceFile_CMD_Chronic_Kidney_Disease",
                           files="parameter_values")
        )

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
        p = self.parameters

        alive_ckd_idx = df.loc[df.is_alive & df.nc_chronic_kidney_disease].index

        # write to property:
        df.loc[df.is_alive & ~df.nc_chronic_kidney_disease, 'ckd_status'] = 'pre_diagnosis'

        df.loc[list(alive_ckd_idx), "ckd_on_treatment"] = False
        df.loc[list(alive_ckd_idx), "ckd_diagnosed"] = False
        df.loc[list(alive_ckd_idx), "ckd_date_treatment"] = pd.NaT
        df.loc[list(alive_ckd_idx), "ckd_stage_at_which_treatment_given"] = "pre_diagnosis"
        df.loc[list(alive_ckd_idx), "ckd_date_diagnosis"] = pd.NaT
        df.loc[list(alive_ckd_idx), "nc_ckd_total_dialysis_sessions"] = 0
        df.loc[list(alive_ckd_idx), "nc_ckd_on_dialysis"] = False
        df.loc[list(alive_ckd_idx), "ckd_date_transplant"] = pd.NaT
        df.loc[list(alive_ckd_idx), "ckd_death_event_scheduled"] = False


        df.loc[list(alive_ckd_idx), "uses_herbal_medicine"] = \
            self.rng.random(len(alive_ckd_idx)) < p['prop_herbal_use_ckd']
        df.loc[list(df.loc[df.is_alive & ~df.nc_chronic_kidney_disease].index), "uses_herbal_medicine"] = False

        # -------------------- ckd_status -----------
        # Determine who has CKD at all stages:
        # check parameters are sensible: probability of having any CKD stage cannot exceed 1.0
        assert sum(p['init_prob_any_ckd']) <= 1.0

        lm_init_ckd_status_any_ckd = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            sum(p['init_prob_any_ckd']),
            Predictor('nc_diabetes').when(True, p['rp_ckd_nc_diabetes']),
            Predictor().when('hv_inf', p['rp_ckd_hiv_infection']),
            # todo find parameters for all 5 categories of bmi if its a direct risk factor
            # Predictor('li_bmi').when(True, p['rp_ckd_li_bmi']),
            Predictor().when('nc_hypertension', p['rp_ckd_nc_hypertension']),
            Predictor().when('nc_chronic_ischemic_hd', p['rp_ckd_nc_chronic_ischemic_hd']),
            Predictor('uses_herbal_medicine').when(True, p['rp_ckd_herbal_use_baseline']),

        )

        # Get boolean Series of who has ckd
        has_ckd = lm_init_ckd_status_any_ckd.predict(df.loc[df.is_alive & df.nc_chronic_kidney_disease], self.rng)

        # Get indices of those with CKD
        ckd_idx = has_ckd[has_ckd].index if has_ckd.any() else pd.Index([])

        if not ckd_idx.empty:
            # Get non-pre_diagnosis categories
            categories = [cat for cat in df.ckd_status.cat.categories if cat != 'pre_diagnosis']

            # Verify probabilities match categories
            assert len(categories) == len(p['init_prob_any_ckd'])

            # Normalize probabilities
            total_prob = sum(p['init_prob_any_ckd'])
            probs = [p / total_prob for p in p['init_prob_any_ckd']]

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

        # ----- DX TESTS -----
        # Create the diagnostic test representing ckd staging
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            ckd_staging_test=DxTest(
                property='ckd_status',
                sensitivity=self.parameters['sensitivity_of_ckd_staging_test'],
                target_categories=["stage1_4", "stage5"]
            )
        )
        # Create the diagnostic test representing ckd renal clinic and medication
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            renal_clinic_test=DxTest(
                property='ckd_status',
                sensitivity=self.parameters['sensitivity_of_renal_clinic_test'],
                # todo do we really need a test for this? People who take this are already
                #  in stage1_4 from HSI_CKD_Staging
                target_categories=["stage1_4"]
            )
        )

        # Create the diagnostic test representing ckd kidney transplant evaluation
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            kidney_transplant_eval_tests=DxTest(
                property='ckd_status',
                sensitivity=self.parameters['sensitivity_of_kidney_transplant_eval_tests'],
                target_categories=["stage5"]
            )
        )

        # Create the diagnostic test representing ckd kidney transplant surgery tests
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            kidney_transplant_surgery_tests=DxTest(
                property='ckd_status',
                sensitivity=self.parameters['sensitivity_of_kidney_transplant_surgery_tests'],
                target_categories=["stage5"]
            )
        )


        # ----- DISABILITY-WEIGHTS -----
        if "HealthBurden" in self.sim.modules:
            # For those with End-stage renal disease, with kidney transplant (any stage after stage1_4)
            self.daly_wts["stage5_ckd_with_transplant"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=977
            )
            self.daly_wts["stage5_ckd_on_dialysis"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=987
            )

    def report_daly_values(self):
        # return pd.Series(index=self.sim.population.props.index, data=0.0)
        df = self.sim.population.props  # shortcut to population properties dataframe for alive persons

        disability_series_for_alive_persons = pd.Series(index=df.index[df.is_alive], data=0.0)

        # Assign daly_wt to those with CKD stage stage5 and have had a kidney transplant
        disability_series_for_alive_persons.loc[
            (df.ckd_status == "stage5") &
            (~pd.isnull(df.ckd_date_transplant))
            ] = self.daly_wts['stage5_ckd_with_transplant']

        # Assign daly_wt to those with CKD stage stage5 and are on dialysis
        disability_series_for_alive_persons.loc[
            (df.ckd_status == "stage5") &
            df.nc_ckd_on_dialysis
            ] = self.daly_wts['stage5_ckd_on_dialysis']

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
        self.sim.population.props.at[child_id, 'nc_ckd_on_dialysis'] = False
        self.sim.population.props.at[child_id, 'ckd_date_transplant'] = pd.NaT
        self.sim.population.props.at[child_id, 'ckd_death_event_scheduled'] = False

    def on_simulation_end(self) -> None:
        pass

    def make_the_linear_models(self) -> None:
        """Make and save LinearModels that will be used when the module is running"""
        self.lm = dict()
        p = self.parameters

        self.lm['onset_stage1_4'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['rate_onset_to_stage1_4'],
            Predictor('uses_herbal_medicine')
            .when(True, p['rp_ckd_herbal_use_baseline'])
        )

        self.lm['stage1_to_4_stage5'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['rate_stage1_4_to_stage5'],
            Predictor('had_treatment_during_this_stage', external=True)
            .when(True, 0.0).otherwise(1.0)
        )

    def look_up_consumable_item_codes(self):
        """Look up the item codes used in the HSI of this module"""
        get_item_codes = self.sim.modules['HealthSystem'].get_item_code_from_item_name

        self.cons_item_codes = dict()

        self.cons_item_codes['ckd_staging_consumables'] = {
            get_item_codes("Glove disposable latex medium_100_CMST"): 1,
            get_item_codes("Blood collection tube"): 1,
            get_item_codes("Reagents"): 1
        }
        self.cons_item_codes['renal_consumables'] = {
            get_item_codes("Gloves, exam, latex, disposable, pair"): 4,
            get_item_codes("Catheter Foley's + urine bag (2000ml) 14g_each_CMST"): 1,
            get_item_codes("Disinfectant for hands, alcohol-based,  1 litre bottle"): 1
        }
        self.cons_item_codes['kidney_transplant_eval_cons'] = {
            get_item_codes("Blood collection tube"): 1,
            get_item_codes("Reagents"): 3,
            get_item_codes("Radiopharmaceuticals"): 4
        }
        self.cons_item_codes['kidney_transplant_surgery_cons'] = {
            # Prepare surgical instruments
            # administer an IV
            get_item_codes("Cannula iv  (winged with injection pot) 18_each_CMST"): 1,
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
            get_item_codes("Disposables gloves, powder free, 100 pieces per box"): 1,
            get_item_codes("surgical face mask, disp., with metal nose piece_50_IDA"): 1,
            # request syringe
            get_item_codes("Syringe, Autodisable SoloShot IX "): 1
        }

    def schedule_dialysis_death_event(self, person_id):
        """Start periodic dialysis mortality checking for a person."""
        df = self.sim.population.props

        if not df.at[person_id, 'is_alive']:
            return

        if not df.at[person_id, 'nc_ckd_on_dialysis']:
            return

        # Just to prevent double-scheduling
        if df.at[person_id, 'ckd_death_event_scheduled']:
            return

        # Schedule first death check 6 months from now
        first_check = self.sim.date + DateOffset(months=6)
        self.sim.schedule_event(
            CKD_DialysisDeathEvent(self, person_id),
            first_check
        )

        df.at[person_id, 'ckd_death_event_scheduled'] = True


class CMDChronicKidneyDiseasePollEvent(RegularEvent, PopulationScopeEventMixin):
    """An event that controls the development process of Chronic Kidney Disease (CKD) and logs current states."""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population: Population) -> None:
        df = population.props
        p = self.module.parameters

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

        # ------------SELECTING INDIVIDUALS already on dialysis from cardio_metabolic_disorders module-----------#
        people_on_dialysis_from_cmd = (
            df.is_alive &
            df.nc_chronic_kidney_disease &
            (df.nc_ckd_total_dialysis_sessions > 0) &
            (df.ckd_status.isin(['pre_diagnosis', 'stage1_4']))
        )

        # Set these people as on dialysis and on stage5
        dialysis_idx = df.loc[people_on_dialysis_from_cmd].index
        if not dialysis_idx.empty:
            df.loc[dialysis_idx, 'nc_ckd_on_dialysis'] = True
            df.loc[dialysis_idx, 'ckd_status'] = 'stage5'
            df.loc[dialysis_idx, 'ckd_diagnosed'] = True
            df.loc[dialysis_idx, 'ckd_date_diagnosis'] = self.sim.date

            # Schedule death events for those who are newly identified as on dialysis
            for person_id in dialysis_idx:
                if not df.at[person_id, 'ckd_death_event_scheduled']:
                    self.module.schedule_dialysis_death_event(person_id)

        # ----------------------------SELECTING INDIVIDUALS FOR CKD Diagnosis staging----------------------------#
        staging_eligible_population = (
            (df.is_alive & df.nc_chronic_kidney_disease) &
            (df.ckd_status == 'pre_diagnosis') &
            (df.age_years >= 18) &
            (pd.isna(df.ckd_date_diagnosis)) &
            (~df.nc_ckd_on_dialysis)
        )

        staging_eligible_idx = df.loc[staging_eligible_population].index

        if not staging_eligible_idx.empty:
            selected_for_staging = self.module.rng.random(len(staging_eligible_idx)) < p['prob_staging_referral']
            staging_idx = staging_eligible_idx[selected_for_staging]

            for person_id in staging_idx:
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event=HSI_CKD_Staging(self.module, person_id),
                    priority=0,
                    topen=self.sim.date,
                    tclose=None,
                )

        surgeries_done = 0

        while (
            surgeries_done < p['max_surgeries_per_month']
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


class CKD_DialysisDeathEvent(Event, IndividualScopeEventMixin):
    """
    Performs death for individuals who have been on dialysis for certain durations.
    Uses dialysis session count to determine time on dialysis.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        p = self.module.parameters

        if not df.at[person_id, "is_alive"]:
            return

        # Check if person is still on dialysis
        if not df.at[person_id, "nc_ckd_on_dialysis"]:
            return

        # Calculate approximate years on dialysis based on session count
        # Assuming 3 sessions per week = 144 sessions per year
        total_sessions = df.at[person_id, "nc_ckd_total_dialysis_sessions"]
        approx_years_on_dialysis = total_sessions / 144.0

        # Determine if death should occur based on the year thresholds
        should_die = False

        if approx_years_on_dialysis >= 10:
            # Person has been on dialysis for at least 10 years
            rand_val = self.module.rng.random()
            # Person dies if random value is less than probability
            should_die = rand_val < p['prob_dialysis_death_10_years']

        elif approx_years_on_dialysis >= 5:
            # Person has been on dialysis for at least 5 years
            rand_val = self.module.rng.random()
            should_die = rand_val < p['prob_dialysis_death_5_years']

        elif approx_years_on_dialysis >= 1:
            # Person has been on dialysis for at least 1 year
            rand_val = self.module.rng.random()
            should_die = rand_val < p['prob_dialysis_death_1_year']

        if should_die:
            logger.debug(key="CKD_DialysisDeathEvent",
                         data=f"CKD_DialysisDeathEvent: scheduling death for person {person_id} "
                              f"on dialysis for {approx_years_on_dialysis:.1f} years on {self.sim.date}")

            self.sim.modules['Demography'].do_death(
                individual_id=person_id,
                cause="chronic_kidney_disease_death",
                originating_module=self.module
            )

            # Reset the flag since person is now dead
            df.at[person_id, 'ckd_death_event_scheduled'] = False
        else:
            # If person survived this check, reschedule for another check in 6 months
            next_check_date = self.sim.date + DateOffset(months=6)
            self.sim.schedule_event(
                CKD_DialysisDeathEvent(self.module, person_id),
                next_check_date
            )


class HSI_CKD_Staging(HSI_Event, IndividualScopeEventMixin):
    """
    This is an event where CKD is diagnosed and the person could be in any stage from 1 to 5"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CMDChronicKidneyDisease)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'CardioMetabolicDisorders_CKD_Staging'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1, 'NewAdult': 1})
        self.ACCEPTED_FACILITY_LEVEL = '2'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(key='debug',
                     data=f'This is HSI_CKD_Staging for person {person_id}')
        df = self.sim.population.props
        person = df.loc[person_id]
        hs = self.sim.modules["HealthSystem"]
        p = self.module.parameters

        if not df.at[person_id, 'is_alive']:
            # The person is not alive, the event did not happen: so return a blank footprint
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        # if person has already been diagnosed, do nothing
        if person["ckd_diagnosed"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        is_cons_available = self.get_consumables(
            self.module.cons_item_codes['ckd_staging_consumables']
        )

        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='ckd_staging_test',
            hsi_event=self
        )

        if dx_result and is_cons_available:
            self.add_equipment({'Weighing scale', 'Blood pressure machine', 'Red blood bottle',
                                'Urine dip Stick', 'Ultrasound scanning machine'})

            if self.module.rng.random() < p['prob_stage5']:
                df.at[person_id, 'ckd_status'] = 'stage5'
            else:
                df.at[person_id, 'ckd_status'] = 'stage1_4'

            df.at[person_id, 'ckd_diagnosed'] = True
            df.at[person_id, 'ckd_date_diagnosis'] = self.sim.date

            if df.at[person_id, 'ckd_status'] == 'stage1_4':
                # Conservative management at renal clinic
                hs.schedule_hsi_event(
                    hsi_event=HSI_Renal_Clinic_and_Medication(self.module, person_id),
                    priority=1,
                    topen=self.sim.date,
                    tclose=self.sim.date + DateOffset(months=1),
                )

            elif df.at[person_id, 'ckd_status'] == 'stage5':
                hs.schedule_hsi_event(
                    hsi_event=HSI_Kidney_Transplant_Evaluation(self.module, person_id),
                    priority=1,
                    topen=self.sim.date,
                    tclose=self.sim.date + DateOffset(months=1),
                )


class HSI_Renal_Clinic_and_Medication(HSI_Event, IndividualScopeEventMixin):
    """
    This is an event where a CKD patient is managed conservatively; attending clinic monthly for symptom management"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CMDChronicKidneyDisease)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'CardioMetabolicDisorders_CKD_Renal_Medication'
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
            # record date of treatment
            df.at[person_id, 'ckd_on_treatment'] = True
            df.at[person_id, 'ckd_date_treatment'] = self.sim.date
            df.at[person_id, 'ckd_stage_at_which_treatment_given'] = df.at[person_id, 'ckd_status']
            self.add_equipment({'Weighing scale', 'Blood pressure machine', 'Purple blood bottle', 'Red blood bottle'
                                'Ultrasound scanning machine', 'Electrocardiogram', 'Oxygen concentrator'})

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
        self.TREATMENT_ID = 'CardioMetabolicDisorders_CKD_Kidney_Transplant_Evaluation'
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

        # if person not yet diagnosed, do nothing
        if not person["ckd_diagnosed"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        is_cons_available = self.get_consumables(
            self.module.cons_item_codes['kidney_transplant_eval_cons']
        )

        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='kidney_transplant_eval_tests',
            hsi_event=self
        )

        if dx_result and is_cons_available:
            # record date of treatment
            df.at[person_id, 'ckd_date_treatment'] = self.sim.date  #todo should be transplant eval date?
            df.at[person_id, 'ckd_stage_at_which_treatment_given'] = df.at[person_id, 'ckd_status']

            # Append to waiting list if evaluation is successful
            self.module.kidney_transplant_waiting_list.append(person_id)

        else:
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                hsi_event=cardio_metabolic_disorders.HSI_CardioMetabolicDisorders_Dialysis_Refill(
                    person_id=person_id,
                    module=self.sim.modules["CardioMetabolicDisorders"],
                ),
                priority=1,
                topen=self.sim.date,
                tclose=self.sim.date + pd.DateOffset(days=1)
            )
            # Set person to be on dialysis
            df.at[person_id, 'nc_ckd_on_dialysis'] = True
            self.schedule_dialysis_death_event(person_id)


# class HSI_Haemodialysis_Refill(HSI_Event, IndividualScopeEventMixin):
#     """This is an event in which a person goes for dialysis sessions 2 times a week
#     adding up to 8 times a month."""
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#
#         self.TREATMENT_ID = 'CKD_Treatment_Haemodialysis'
#         self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
#         self.ACCEPTED_FACILITY_LEVEL = '3'
#
#     def apply(self, person_id, squeeze_factor):
#         df = self.sim.population.props
#
#         if not df.at[person_id, 'is_alive'] or not df.at[person_id, 'nc_chronic_kidney_disease']:
#             return self.sim.modules['HealthSystem'].get_blank_appt_footprint()
#
#         if not df.at[person_id, 'nc_ckd_on_dialysis']:
#             df.at[person_id, 'nc_ckd_on_dialysis'] = True
#
#         # Increment total number of dialysis sessions the person has ever had in their lifetime
#         df.at[person_id, 'nc_ckd_total_dialysis_sessions'] += 1
#
#         self.add_equipment({'Chair', 'Dialysis machine', 'Dialyser (Artificial Kidney)',
#                             'Bloodlines', 'Dialysate solution', 'Dialysis water treatment system'})
#
#         next_session_date = self.sim.date + pd.DateOffset(days=3)
#         self.sim.modules['HealthSystem'].schedule_hsi_event(self,
#                                                             topen=next_session_date,
#                                                             tclose=next_session_date + pd.DateOffset(days=1),
#                                                             priority=1
#                                                             )

    def never_ran(self) -> None:
        """What to do if the event is never run by the HealthSystem"""
        # Reschedule this HSI to happen again 3 days time.
        next_session_date = self.sim.date + pd.DateOffset(days=3)
        self.sim.modules['HealthSystem'].schedule_hsi_event(self,
                                                            topen=next_session_date,
                                                            tclose=next_session_date + pd.DateOffset(days=1),
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
        # todo need to update priority number in resource files (for all HSIs)
        self.TREATMENT_ID = 'CardioMetabolicDisorders_CKD_Kidney_Transplant_Surgery'
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
            self.add_equipment({'Patient monitor', 'Infusion pump', 'Dialysis machine', 'Bloodlines', 'Water tank',
                                'Reverse osmosis machine', 'Water softener', 'Carbon filter', '5 micro filter',
                                'Ventilator', 'Electrocautery unit', 'Suction machine', 'Theatre bed',
                                'Cold static storage' 'Perfusion machine', 'Ultrasound scanning machine', 'Drip stand',
                                'Trolley, patient'})

        if transplant_successful:
            # df.at[person_id, 'ckd_transplant_successful'] = True
            df.at[person_id, 'ckd_date_transplant'] = self.sim.date
            df.at[person_id, 'nc_ckd_on_dialysis'] = False
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
            df.at[person_id, 'ckd_date_transplant'] = self.sim.date  #todo Upile look at this logic, related to DALYs

            self.sim.modules['Demography'].do_death(
                individual_id=person_id,
                cause='chronic_kidney_disease_death',
                originating_module=self.module
            )


class HSI_AntiRejectionDrug_Refill(HSI_Event, IndividualScopeEventMixin):
    """This is an event where a kidney transplant recipient gets drugs every month for the rest of their lives """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        self.TREATMENT_ID = 'CardioMetabolicDisorders_CKD_AntiRejectionDrug_Refill'
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
