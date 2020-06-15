"""
Prostate Cancer Disease Module

Limitations to note:
* Footprints of HSI -- pending input from expert on resources required.
"""

from pathlib import Path

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import demography
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ProstateCancer(Module):
    """Prostate Cancer Disease Module"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.linear_models_for_progression_of_prostate_cancer_status = dict()
        self.lm_prostate_ca_onset_urunary_symptoms = None
        self.daly_wts = dict()

    PARAMETERS = {
        "init_prop_prostate_ca_stage": Parameter(
            Types.LIST,
            "initial proportions in prostate cancer stages for men aged 35-49"
        ),
        "init_prop_urinary_symptoms_by_stage": Parameter(
            Types.LIST, "initial proportions of those in prostate ca stages that have urinary symptoms"
        ),
        "init_prop_with_urinary_symptoms_diagnosed_prostate_ca_by_stage": Parameter(
            Types.LIST, "initial proportions of people with prostate ca and urinary symptoms that have been diagnosed"
        ),
        "init_prop_treatment_status_prostate_ca": Parameter(
            Types.LIST, "initial proportions of people with prostate ca that had received treatment"
        ),
        "init_prob_palliative_care": Parameter(
            Types.REAL, "initial probability of being under palliative care if at metastatic stage"
        ),
        "r_prostate_confined_prostate_ca_none": Parameter(
            Types.REAL,
            "probabilty per 3 months of incident (prostate confined) prostate cancer amongst people with no "
            "prostate ca (men, age35-49)",
        ),
        "rr_prostate_confined_prostate_ca_age5069": Parameter(
            Types.REAL, "rate ratio for incident (prostate confined) prostate cancer amongst men aged 50-69"
        ),
        "rr_prostate_confined_prostate_ca_agege70": Parameter(
            Types.REAL, "rate ratio for incident (prostate confined) prostate cancer amongst men aged ge 70"
        ),
        "r_local_ln_prostate_ca_prostate_confined": Parameter(
            Types.REAL,
            "probabilty per 3 months of local lymph node involved prostate ca amongst people with prostate confined "
            "prostate ca"
        ),
        "rr_local_ln_prostate_ca_undergone_curative_treatment": Parameter(
            Types.REAL,
            "rate ratio for local lymph node involved prostate ca for people with prostate confined prostate ca"
            "due to undergoing curative treatment"
        ),
        "r_metastatic_prostate_ca_local_ln": Parameter(
            Types.REAL, "probabilty per 3 months of metastatic prostate cancer amongst people with local lymph node"
                        "involved prostate ca"
        ),
        "rr_metastatic_prostate_ca_undergone_curative_treatment": Parameter(
            Types.REAL,
            "rate ratio for metastatic prostate cancer stage 1 for people with lymph node involved prostate ca due to"
            "undergoing curative treatment"
        ),
        "rate_palliative_care_metastatic_prostate_ca": Parameter(
            Types.REAL, "prob palliative care this 3 month period if metastatic prostate ca"
        ),
        "r_death_metastatic_prostate_cancer": Parameter(
            Types.REAL,
            "probabilty per 3 months of death from prostate cancer mongst people with metastatic prostate cancer",
        ),
        "r_urinary_symptoms_prostate_ca": Parameter(
            Types.REAL, "rate of urinary symptoms if have prostate confined prostate ca"
        ),
        "rr_urinary_symptoms_local_ln_or_metastatic_prostate_cancer": Parameter(
            Types.REAL, "rate ratio of urinary symptoms in a person with local lymph node or metastatuc prostate cancer "
                        "compared with prostate confined prostate ca"
        ),
        "r_pelvic_pain_symptoms_local_ln_prostate_ca": Parameter(
            Types.REAL, "rate of pelvic pain or numbness symptoms if have local lymph node involved prostate cancer"
        ),
        "rr_pelvic_pain_metastatic_prostate_cancer": Parameter(
            Types.REAL,
            "rate ratio of pelvic pain or numbness in a person with metastatic prostate cancer compared with lymph node involved"
            "prostate cancer"
        ),
        "rp_prostate_cancer_age5069": Parameter(
            Types.REAL, "stage-specific relative prevalence at baseline of prostate cancer for age 50-69"
        ),
        "rp_prostate_cancer_agege 70": Parameter(
            Types.REAL, "stage-specific relative prevalence at baseline of prostate cancer for age 70+"
        ),
        "sensitivity_of_psa_test_for_prostate_confined_prostate_ca": Parameter(
            Types.REAL, "sensitivity of psa test for prostate confined_prostate cancer"
        ),
        "sensitivity_of_psa_test_for_local_ln_prostate_ca": Parameter(
            Types.REAL, "sensitivity of psa test for local lymph node involved prostate cancer"
        ),
        "sensitivity_of_psa_test_for_metastatic_prostate_ca": Parameter(
            Types.REAL, "sensitivity of psa test for metastatic prostate cancer"
        ),
        "sensitivity_of_biopsy_for_prostate_confined_prostate_ca": Parameter(
            Types.REAL, "sensitivity of biopsy for prostate confined prostate cancer"
        ),
        "sensitivity_of_biopsy_for_local_ln_prostate_ca": Parameter(
            Types.REAL, "sensitivity of biopsy for local lymph node involved prostate cancer"
        ),
        "sensitivity_of_biopsy_for_metastatic_prostate_ca": Parameter(
            Types.REAL, "sensitivity of biopsy for metastatic prostate cancer"
        ),
    }

    PROPERTIES = {
        "pc_status": Property(
            Types.CATEGORICAL,
            "Current status of the health condition, prostate cancer",
            categories=["none", "prostate_confined", "local_ln", "metastatic"],
        ),

        "pc_date_diagnosis": Property(
            Types.DATE,
            "the date of diagnsosis of the prostate cancer (pd.NaT if never diagnosed)"
        ),

        "pc_date_treatment": Property(
            Types.DATE,
            "date of first receiving attempted curative treatment (pd.NaT if never started treatment)"
        ),

        "pc_stage_at_which_treatment_given": Property(
            Types.CATEGORICAL,
            "the cancer stage at which treatment is given (because the treatment only has an effect during the stage"
            "at which it is given.",
            categories=["prostate_confined", "local_ln", "metastatic"],
        ),

        "pc_date_palliative_care": Property(
            Types.DATE,
            "date of first receiving palliative care (pd.NaT is never had palliative care)"
        ),
    }

    # Symptoms that this module will use
    SYMPTOMS = {'urinary', 'pelvic_pain'}

    def read_parameters(self, data_folder):
        """Setup parameters used by the module, now including disability weights"""

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

        # Update parameters from the resourcefile
        self.load_parameters_from_dataframe(
            pd.read_excel(Path(self.resourcefilepath) / "ResourceFile_Prostate_Cancer.xlsx",
                          sheet_name="parameter_values")
        )

    def initialise_population(self, population):
        """Set property values for the initial population."""
        df = population.props  # a shortcut to the data-frame
        p = self.parameters

        # defaults
        df.loc[df.is_alive, "pc_status"] = "none"
        df.loc[df.is_alive, "pc_date_diagnosis"] = pd.NaT
        df.loc[df.is_alive, "pc_date_treatment"] = pd.NaT
        df.loc[df.is_alive, "pc_stage_at_which_treatment_given"] = "none"
        df.loc[df.is_alive, "pc_date_palliative_care"] = pd.NaT

        # -------------------- pc_status -----------
        # Determine who has cancer at ANY cancer stage:
        # check parameters are sensible: probability of having any cancer stage cannot exceed 1.0
        assert sum(p['init_prop_oes_cancer_stage']) <= 1.0

        lm_init_oc_status_any_dysplasia_or_cancer = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            sum(p['init_prop_oes_cancer_stage']),
            Predictor('li_ex_alc').when(True, p['rp_oes_cancer_ex_alc']),
            Predictor('li_tob').when(True, p['rp_oes_cancer_tobacco']),
            Predictor('age_years').apply(lambda x: ((x - 20) ** p['rp_oes_cancer_per_year_older']) if x > 20 else 0.0)
        )

        oc_status_any_dysplasia_or_cancer = \
            lm_init_oc_status_any_dysplasia_or_cancer.predict(df.loc[df.is_alive], self.rng)

        # Determine the stage of the cancer for those who do have a cancer:
        if oc_status_any_dysplasia_or_cancer.sum():
            sum_probs = sum(p['init_prop_oes_cancer_stage'])
            if sum_probs > 0:
                prob_by_stage_of_cancer_if_cancer = [i/sum_probs for i in p['init_prop_oes_cancer_stage']]
                assert (sum(prob_by_stage_of_cancer_if_cancer) - 1.0) < 1e-10
                df.loc[oc_status_any_dysplasia_or_cancer, "oc_status"] = self.rng.choice(
                    [val for val in df.oc_status.cat.categories if val != 'none'],
                    size=oc_status_any_dysplasia_or_cancer.sum(),
                    p=prob_by_stage_of_cancer_if_cancer
                )

        # -------------------- SYMPTOMS -----------
        # ----- Impose the symptom of random sample of those in each cancer stage to have the symptom of dysphagia:
        lm_init_disphagia = LinearModel.multiplicative(
            Predictor('oc_status')  .when("none", 0.0)
                                    .when("low_grade_dysplasia",
                                          p['init_prop_dysphagia_oes_cancer_by_stage'][0])
                                    .when("high_grade_dysplasia",
                                          p['init_prop_dysphagia_oes_cancer_by_stage'][1])
                                    .when("stage1",
                                          p['init_prop_dysphagia_oes_cancer_by_stage'][2])
                                    .when("stage2",
                                          p['init_prop_dysphagia_oes_cancer_by_stage'][3])
                                    .when("stage3",
                                          p['init_prop_dysphagia_oes_cancer_by_stage'][4])
                                    .when("stage4",
                                          p['init_prop_dysphagia_oes_cancer_by_stage'][5])
        )
        has_dysphagia_at_init = lm_init_disphagia.predict(df.loc[df.is_alive], self.rng)
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=has_dysphagia_at_init.index[has_dysphagia_at_init].tolist(),
            symptom_string='dysphagia',
            add_or_remove='+',
            disease_module=self
        )

        # -------------------- oc_date_diagnosis -----------
        lm_init_diagnosed = LinearModel.multiplicative(
            Predictor('oc_status')  .when("none", 0.0)
                                    .when("low_grade_dysplasia",
                                          p['init_prop_with_dysphagia_diagnosed_oes_cancer_by_stage'][0])
                                    .when("high_grade_dysplasia",
                                          p['init_prop_with_dysphagia_diagnosed_oes_cancer_by_stage'][1])
                                    .when("stage1",
                                          p['init_prop_with_dysphagia_diagnosed_oes_cancer_by_stage'][2])
                                    .when("stage2",
                                          p['init_prop_with_dysphagia_diagnosed_oes_cancer_by_stage'][3])
                                    .when("stage3",
                                          p['init_prop_with_dysphagia_diagnosed_oes_cancer_by_stage'][4])
                                    .when("stage4",
                                          p['init_prop_with_dysphagia_diagnosed_oes_cancer_by_stage'][5])
        )
        ever_diagnosed = lm_init_diagnosed.predict(df.loc[df.is_alive], self.rng)

        # ensure that persons who have not ever had the symptom dysphagia are diagnosed:
        ever_diagnosed.loc[~has_dysphagia_at_init] = False

        # For those that have been diagnosed, set data of diagnosis to today's date
        df.loc[ever_diagnosed, "oc_date_diagnosis"] = self.sim.date

        # -------------------- oc_date_treatment -----------
        lm_init_treatment_for_those_diagnosed = LinearModel.multiplicative(
            Predictor('oc_status')  .when("none", 0.0)
                                    .when("low_grade_dysplasia",
                                          p['init_prop_treatment_status_oes_cancer'][0])
                                    .when("high_grade_dysplasia",
                                          p['init_prop_treatment_status_oes_cancer'][1])
                                    .when("stage1",
                                          p['init_prop_treatment_status_oes_cancer'][2])
                                    .when("stage2",
                                          p['init_prop_treatment_status_oes_cancer'][3])
                                    .when("stage3",
                                          p['init_prop_treatment_status_oes_cancer'][4])
                                    .when("stage4",
                                          p['init_prop_treatment_status_oes_cancer'][5])
        )
        treatment_initiated = lm_init_treatment_for_those_diagnosed.predict(df.loc[df.is_alive], self.rng)

        # prevent treatment having been initiated for anyone who is not yet diagnosed
        treatment_initiated.loc[pd.isnull(df.oc_date_diagnosis)] = False

        # assume that the stage at which treatment is begun is the stage the person is in now;
        df.loc[treatment_initiated, "oc_stage_at_which_treatment_applied"] = df.loc[treatment_initiated, "oc_status"]

        # set date at which treatment began: same as diagnosis (NB. no HSI is established for this)
        df.loc[treatment_initiated, "oc_date_treatment"] = df.loc[treatment_initiated, "oc_date_diagnosis"]

        # -------------------- oc_date_palliative_care -----------
        in_stage4_diagnosed = df.index[df.is_alive & (df.oc_status == 'stage4') & ~pd.isnull(df.oc_date_diagnosis)]

        select_for_care = self.rng.random_sample(size=len(in_stage4_diagnosed)) < p['init_prob_palliative_care']
        select_for_care = in_stage4_diagnosed[select_for_care]

        # set date of palliative care being initiated: same as diagnosis (NB. future HSI will be scheduled for this)
        df.loc[select_for_care, "oc_date_palliative_care"] = df.loc[select_for_care, "oc_date_diagnosis"]

    def initialise_simulation(self, sim):
        """
        * Schedule the main polling event
        * Schedule the main logging event
        * Define the LinearModels
        * Define the Diagnostic used
        * Define the Disability-weights
        * Schedule the palliative care appointments for those that are on palliative care at initiation
        """

        # ----- SCHEDULE LOGGING EVENTS -----
        # Schedule logging event to happen immediately
        sim.schedule_event(OesCancerLoggingEvent(self), sim.date + DateOffset(months=0))

        # ----- SCHEDULE MAIN POLLING EVENTS -----
        # Schedule main polling event to happen immediately
        sim.schedule_event(OesCancerMainPollingEvent(self), sim.date + DateOffset(months=0))

        # ----- LINEAR MODELS -----
        # Define LinearModels for the progression of cancer, in each 3 month period
        # NB. The effect being produced is that treatment only has the effect for during the stage at which the
        # treatment was received.

        df = sim.population.props
        p = self.parameters
        lm = self.linear_models_for_progession_of_oc_status

        lm['low_grade_dysplasia'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_low_grade_dysplasia_none'],
            Predictor('age_years').apply(
                lambda x: 0 if x < 20 else (x - 20) ** p['rr_low_grade_dysplasia_none_per_year_older']
            ),
            Predictor('sex').when('F', p['rr_low_grade_dysplasia_none_female']),
            Predictor('li_tob').when(True, p['rr_low_grade_dysplasia_none_tobacco']),
            Predictor('li_ex_alc').when(True, p['rr_low_grade_dysplasia_none_ex_alc']),
            Predictor('oc_status').when('none', 1.0)
                                  .otherwise(0.0)
        )

        lm['high_grade_dysplasia'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_high_grade_dysplasia_low_grade_dysp'],
            Predictor('had_treatment_during_this_stage',
                      external=True).when(True, p['rr_high_grade_dysp_undergone_curative_treatment']),
            Predictor('oc_status').when('low_grade_dysplasia', 1.0)
                                  .otherwise(0.0)
        )

        lm['stage1'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage1_high_grade_dysp'],
            Predictor('had_treatment_during_this_stage',
                      external=True).when(True, p['rr_stage1_undergone_curative_treatment']),
            Predictor('oc_status').when('high_grade_dysplasia', 1.0)
                                  .otherwise(0.0)
        )

        lm['stage2'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage2_stage1'],
            Predictor('had_treatment_during_this_stage',
                      external=True).when(True, p['rr_stage2_undergone_curative_treatment']),
            Predictor('oc_status').when('stage1', 1.0)
                                  .otherwise(0.0)
        )

        lm['stage3'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage3_stage2'],
            Predictor('had_treatment_during_this_stage',
                      external=True).when(True, p['rr_stage3_undergone_curative_treatment']),
            Predictor('oc_status').when('stage2', 1.0)
                                  .otherwise(0.0)
        )

        lm['stage4'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage4_stage3'],
            Predictor('had_treatment_during_this_stage',
                      external=True).when(True, p['rr_stage4_undergone_curative_treatment']),
            Predictor('oc_status').when('stage3', 1.0)
                                  .otherwise(0.0)
        )

        # Check that the dict labels are correct as these are used to set the value of oc_status
        assert set(lm).union({'none'}) == set(df.oc_status.cat.categories)

        # Linear Model for the onset of dysphagia, in each 3 month period
        self.lm_onset_dysphagia = LinearModel.multiplicative(
            Predictor('oc_status').when('low_grade_dysplasia',
                                        p['rr_dysphagia_low_grade_dysp'] * p['r_dysphagia_stage1'])
                                  .when('high_grade_dysplaisa',
                                        p['rr_dysphagia_high_grade_dysp'] * p['r_dysphagia_stage1'])
                                  .when('stage1', p['r_dysphagia_stage1'])
                                  .when('stage2', p['rr_dysphagia_stage2'] * p['r_dysphagia_stage1'])
                                  .when('stage3', p['rr_dysphagia_stage3'] * p['r_dysphagia_stage1'])
                                  .when('stage4', p['rr_dysphagia_stage4'] * p['r_dysphagia_stage1'])
                                  .otherwise(0.0)
        )

        # ----- DX TESTS -----
        # Create the diagnostic test representing the use of an endoscope to oc_status
        # This properties of conditional on the test being done only to persons with the Symptom, 'dysphagia'.
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            endoscopy_for_oes_cancer_given_dysphagia=DxTest(
                property='oc_status',
                sensitivity=self.parameters['sensitivity_of_endoscopy_for_oes_cancer_with_dysphagia'],
                target_categories=["low_grade_dysplasia", "high_grade_dysplasia",
                                   "stage1", "stage2", "stage3", "stage4"]
            )
        )

        # ----- DISABILITY-WEIGHT -----
        if "HealthBurden" in self.sim.modules:
            # For those with cancer (any stage prior to stage 4) and never treated
            self.daly_wts["stage_1_3"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=550
                # "Diagnosis and primary therapy phase of esophageal cancer":
                #  "Cancer, diagnosis and primary therapy ","has pain, nausea, fatigue, weight loss and high anxiety."
            )

            # For those with cancer (any stage prior to stage 4) and has been treated
            self.daly_wts["stage_1_3_treated"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=547
                # "Controlled phase of esophageal cancer,Generic uncomplicated disease":
                # "worry and daily medication,has a chronic disease that requires medication every day and causes some
                #   worry but minimal interference with daily activities".
            )

            # For those in stage 4: no palliative care
            self.daly_wts["stage4"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=549
                # "Metastatic phase of esophageal cancer:
                # "Cancer, metastatic","has severe pain, extreme fatigue, weight loss and high anxiety."
            )

            # For those in stage 4: with palliative care
            self.daly_wts["stage4_palliative_care"] = self.daly_wts["stage_1_3"]
            # By assumption, we say that that the weight for those in stage 4 with palliative care is the same as
            # that for those with stage 1-3 cancers.

        # ----- HSI FOR PALLIATIVE CARE -----
        on_palliative_care_at_initiation = df.index[df.is_alive & ~pd.isnull(df.oc_date_palliative_care)]
        for person_id in on_palliative_care_at_initiation:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_OesophagealCancer_PalliativeCare(module=self, person_id=person_id),
                priority=0,
                topen=self.sim.date + DateOffset(months=1),
                tclose=self.sim.date + DateOffset(months=1) + DateOffset(weeks=1)
            )

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props
        df.at[child_id, "oc_status"] = "none"
        df.at[child_id, "oc_date_diagnosis"] = pd.NaT
        df.at[child_id, "oc_date_treatment"] = pd.NaT
        df.at[child_id, "oc_stage_at_which_treatment_applied"] = "none"
        df.at[child_id, "oc_date_palliative_care"] = pd.NaT

    def on_hsi_alert(self, person_id, treatment_id):
        pass

    def report_daly_values(self):
        # This must send back a dataframe that reports on the HealthStates for all individuals over
        # the past month

        df = self.sim.population.props  # shortcut to population properties dataframe for alive persons

        disability_series_for_alive_persons = pd.Series(index=df.index[df.is_alive], data=0.0)

        # Assign daly_wt to those with cancer stages before stage4 and have either never been treated or are no longer
        # in the stage in which they were treated
        disability_series_for_alive_persons.loc[
            (
                (df.oc_status == "stage1") |
                (df.oc_status == "stage2") |
                (df.oc_status == "stage3")
            )
        ] = self.daly_wts['stage_1_3']

        # Assign daly_wt to those with cancer stages before stage4 and who have been treated and who are still in the
        # stage in which they were treated.
        disability_series_for_alive_persons.loc[
            (
                ~pd.isnull(df.oc_date_treatment) & (
                    (df.oc_status == "stage1") |
                    (df.oc_status == "stage2") |
                    (df.oc_status == "stage3")
                ) & (df.oc_status == df.oc_stage_at_which_treatment_applied)
            )
        ] = self.daly_wts['stage_1_3_treated']

        # Assign daly_wt to those in stage4 cancer (who have not had palliative care)
        disability_series_for_alive_persons.loc[
            (df.oc_status == "stage4") &
            (pd.isnull(df.oc_date_palliative_care))
            ] = self.daly_wts['stage4']

        # Assign daly_wt to those in stage4 cancer, who have had palliative care
        disability_series_for_alive_persons.loc[
            (df.oc_status == "stage4") &
            (~pd.isnull(df.oc_date_palliative_care))
            ] = self.daly_wts['stage4_palliative_care']

        return disability_series_for_alive_persons


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------

class OesCancerMainPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all Oesophageal cancer properties for population:
    * Acquisition and progression of Oesophageal Cancer
    * Symptom Development according to stage of Oesophageal Cancer
    * Deaths from Oesophageal Cancer for those in stage4
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=3))
        # scheduled to run every 3 months: do not change as this is hard-wired into the values of all the parameters.

    def apply(self, population):
        df = population.props  # shortcut to dataframe
        m = self.module
        rng = m.rng

        # -------------------- ACQUISITION AND PROGRESSION OF CANCER (oc_status) -----------------------------------

        # determine if the person had a treatment during this stage of cancer (nb. treatment only has an effect on
        #  reducing progression risk during the stage at which is received.
        had_treatment_during_this_stage = \
            df.is_alive & ~pd.isnull(df.oc_date_treatment) & \
            (df.oc_status == df.oc_stage_at_which_treatment_applied)

        for stage, lm in self.module.linear_models_for_progession_of_oc_status.items():
            gets_new_stage = lm.predict(df.loc[df.is_alive], rng,
                                        had_treatment_during_this_stage=had_treatment_during_this_stage)
            idx_gets_new_stage = gets_new_stage[gets_new_stage].index
            df.loc[idx_gets_new_stage, 'oc_status'] = stage

        # -------------------- UPDATING OF SYMPTOM OF DYSPHAGIA OVER TIME --------------------------------
        # Each time this event is called (event 3 months) individuals may develop the symptom of dysphagia.
        # Once the symptom is developed it never resolves naturally. It may trigger health-care-seeking behaviour.
        onset_dysphagia = self.module.lm_onset_dysphagia.predict(df.loc[df.is_alive], rng)
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=onset_dysphagia[onset_dysphagia].index.tolist(),
            symptom_string='dysphagia',
            add_or_remove='+',
            disease_module=self.module
        )

        # -------------------- DEATH FROM OESOPHAGEAL CANCER ---------------------------------------
        # There is a risk of death for those in stage4 only. Death is assumed to go instantly.
        stage4_idx = df.index[df.is_alive & (df.oc_status == "stage4")]
        selected_to_die = stage4_idx[
            rng.random_sample(size=len(stage4_idx)) < self.module.parameters['r_death_oesoph_cancer']]

        for person_id in selected_to_die:
            self.sim.schedule_event(
                demography.InstantaneousDeath(self.module, person_id, "OesophagealCancer"), self.sim.date
            )


# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
# ---------------------------------------------------------------------------------------------------------

class HSI_OesophagealCancer_Investigation_Following_Dysphagia(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_GenericFirstApptAtFacilityLevel1 following presentation for care with the symptom
    dysphagia.
    This event begins the investigation that may result in diagnosis of Oesophageal Cancer and the scheduling of
    treatment or palliative care.
    It is for people with the symptom dysphagia.
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["Over5OPD"] = 1

        self.TREATMENT_ID = "OesophagealCancer_Investigation_Following_Dysphagia"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        # Ignore this event if the person is no longer alive:
        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # Check that this event has been called for someone with the symptom dysphagia
        assert 'dysphagia' in self.sim.modules['SymptomManager'].has_what(person_id)

        # If the person is already diagnosed, then take no action:
        if not pd.isnull(df.at[person_id, "oc_date_diagnosis"]):
            return hs.get_blank_appt_footprint()

        # Use an endoscope to diagnose whether the person has Oesophageal Cancer:
        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='endoscopy_for_oes_cancer_given_dysphagia',
            hsi_event=self
        )

        if dx_result:
            # record date of diagnosis:
            df.at[person_id, 'oc_date_diagnosis'] = self.sim.date

            # Check if is in stage4:
            in_stage4 = df.at[person_id, 'oc_status'] == 'stage4'
            # If the diagnosis does detect cancer, it is assumed that the classification as stage4 is made accurately.

            if not in_stage4:
                # start treatment:
                hs.schedule_hsi_event(
                    hsi_event=HSI_OesophagealCancer_StartTreatment(
                        module=self.module,
                        person_id=person_id
                    ),
                    priority=0,
                    topen=self.sim.date,
                    tclose=None
                )

            else:
                # start palliative care:
                hs.schedule_hsi_event(
                    hsi_event=HSI_OesophagealCancer_PalliativeCare(
                        module=self.module,
                        person_id=person_id
                    ),
                    priority=0,
                    topen=self.sim.date,
                    tclose=None
                )

    def did_not_run(self):
        pass


class HSI_OesophagealCancer_StartTreatment(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_OesophagealCancer_Investigation_Following_Dysphagia following a diagnosis of
    Oesophageal Cancer. It initiates the treatment of Oesophageal Cancer.
    It is only for persons with a cancer that is not in stage4 and who have been diagnosed.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["Over5OPD"] = 1

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "OesophagealCancer_StartTreatment"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 3
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # Check that the person has cancer, not in stage4, has been diagnosed and is not on treatment
        assert not df.at[person_id, "oc_status"] == 'none'
        assert not df.at[person_id, "oc_status"] == 'stage4'
        assert not pd.isnull(df.at[person_id, "oc_date_diagnosis"])
        assert pd.isnull(df.at[person_id, "oc_date_treatment"])

        # Record date and stage of starting treatment
        df.at[person_id, "oc_date_treatment"] = self.sim.date
        df.at[person_id, "oc_stage_at_which_treatment_applied"] = df.at[person_id, "oc_status"]

        # Schedule a post-treatment check for 12 months:
        hs.schedule_hsi_event(
            hsi_event=HSI_OesophagealCancer_PostTreatmentCheck(
                module=self.module,
                person_id=person_id,
            ),
            topen=self.sim.date + DateOffset(years=12),
            tclose=None,
            priority=0
        )

    def did_not_run(self):
        pass


class HSI_OesophagealCancer_PostTreatmentCheck(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_OesophagealCancer_StartTreatment and itself.
    It is only for those who have undergone treatment for Oesophageal Cancer.
    If the person has developed cancer to stage4, the patient is initiated on palliative care; otherwise a further
    appointment is scheduled for one year.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["Over5OPD"] = 1

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "OesophagealCancer_MonitorTreatment"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 3
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # Check that the person is has cancer and is on treatment
        assert not df.at[person_id, "oc_status"] == 'none'
        assert not pd.isnull(df.at[person_id, "oc_date_diagnosis"])
        assert not pd.isnull(df.at[person_id, "oc_date_treatment"])

        if df.at[person_id, 'oc_status'] == 'stage4':
            # If has progressed to stage4, then start Palliative Care immediately:
            hs.schedule_hsi_event(
                hsi_event=HSI_OesophagealCancer_PalliativeCare(
                    module=self.module,
                    person_id=person_id
                ),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )

        else:
            # Schedule another HSI_OesophagealCancer_PostTreatmentCheck event in one month
            hs.schedule_hsi_event(
                hsi_event=HSI_OesophagealCancer_PostTreatmentCheck(
                    module=self.module,
                    person_id=person_id
                ),
                topen=self.sim.date + DateOffset(years=1),
                tclose=None,
                priority=0
            )

    def did_not_run(self):
        pass


class HSI_OesophagealCancer_PalliativeCare(HSI_Event, IndividualScopeEventMixin):
    """
    This is the event for palliative care. It does not affect the patients progress but does affect the disability
     weight and takes resources from the healthsystem.
    This event is scheduled by either:
    * HSI_OesophagealCancer_Investigation_Following_Dysphagia following a diagnosis of Oesophageal Cancer at stage4.
    * HSI_OesophagealCancer_PostTreatmentCheck following progression to stage4 during treatment.
    * Itself for the continuance of care.
    It is only for persons with a cancer in stage4.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["Over5OPD"] = 1

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "OesophagealCancer_PalliativeCare"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 3
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # Check that the person is in stage4
        assert df.at[person_id, "oc_status"] == 'stage4'

        # Record the start of palliative care if this is first appointment
        if pd.isnull(df.at[person_id, "oc_date_palliative_care"]):
            df.at[person_id, "oc_date_palliative_care"] = self.sim.date

        # Schedule another instance of the event for one month
        hs.schedule_hsi_event(
            hsi_event=HSI_OesophagealCancer_PalliativeCare(
                module=self.module,
                person_id=person_id
            ),
            topen=self.sim.date + DateOffset(months=1),
            tclose=None,
            priority=0
        )

    def did_not_run(self):
        pass


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
# ---------------------------------------------------------------------------------------------------------

class OesCancerLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """The only logging event for this module"""

    def __init__(self, module):
        """schedule logging to repeat every 3 months
        """
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        """Compute statistics regarding the current status of persons and output to the logger
        """
        df = population.props

        # CURRENT STATUS COUNTS
        # Create dictionary for each subset, adding prefix to key name, and adding to make a flat dict for logging.
        out = {}

        # Current counts, total
        out.update({
            f'total_{k}': v for k, v in df.loc[df.is_alive].oc_status.value_counts().items()})

        # Current counts, undiagnosed
        out.update({f'undiagnosed_{k}': v for k, v in df.loc[df.is_alive].loc[
            pd.isnull(df.oc_date_diagnosis), 'oc_status'].value_counts().items()})

        # Current counts, diagnosed
        out.update({f'diagnosed_{k}': v for k, v in df.loc[df.is_alive].loc[
            ~pd.isnull(df.oc_date_diagnosis), 'oc_status'].value_counts().items()})

        # Current counts, on treatment (excl. palliative care)
        out.update({f'treatment_{k}': v for k, v in df.loc[df.is_alive].loc[(~pd.isnull(
            df.oc_date_treatment) & pd.isnull(
            df.oc_date_palliative_care)), 'oc_status'].value_counts().items()})

        # Current counts, on palliative care
        out.update({f'palliative_{k}': v for k, v in df.loc[df.is_alive].loc[
            ~pd.isnull(df.oc_date_palliative_care), 'oc_status'].value_counts().items()})

        # Counts of those that have been diagnosed, started treatment or started palliative care since last logging
        # event:
        date_now = self.sim.date
        date_lastlog = self.sim.date - pd.DateOffset(months=self.repeat)

        out.update({
            'diagnosed_since_last_log': df.oc_date_diagnosis.between(date_lastlog, date_now).sum(),
            'treated_since_last_log': df.oc_date_treatment.between(date_lastlog, date_now).sum(),
            'palliative_since_last_log': df.oc_date_palliative_care.between(date_lastlog, date_now).sum()
        })

        logger.info('%s|summary_stats|%s', self.sim.date, out)
