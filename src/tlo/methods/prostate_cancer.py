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
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.demography import InstantaneousDeath
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ProstateCancer(Module):
    """Prostate Cancer Disease Module"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.linear_models_for_progression_of_pc_status = dict()
        self.lm_prostate_ca_onset_urinary_symptoms = None
        self.lm_onset_pelvic_pain = None
        self.daly_wts = dict()

    INIT_DEPENDENCIES = {'Demography', 'HealthSystem', 'SymptomManager'}

    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden'}

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'ProstateCancer': Cause(gbd_causes='Prostate cancer', label='Cancer (Prostate)'),
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'ProstateCancer': Cause(gbd_causes='Prostate cancer', label='Cancer (Prostate)'),
    }

    PARAMETERS = {
        "init_prop_prostate_ca_stage": Parameter(
            Types.LIST,
            "initial proportions in prostate cancer stages for men aged 35-49"
        ),
        "init_prop_urinary_symptoms_by_stage": Parameter(
            Types.LIST, "initial proportions of those in prostate ca stages that have urinary symptoms"
        ),
        "init_prop_pelvic_pain_symptoms_by_stage": Parameter(
            Types.LIST, "initial proportions of those in prostate ca stages that have pelvic pain symptoms"
        ),
        "init_prop_with_urinary_symptoms_diagnosed_prostate_ca_by_stage": Parameter(
            Types.LIST, "initial proportions of people with prostate ca and urinary symptoms that have been diagnosed"
        ),
        "init_prop_with_pelvic_pain_symptoms_diagnosed_prostate_ca_by_stage": Parameter(
            Types.LIST,
            "initial proportions of people with prostate ca and pelvic pain symptoms that have been diagnosed"
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
            Types.REAL,
            "rate ratio of urinary symptoms in a person with local lymph node or metastatuc prostate cancer "
            "compared with prostate confined prostate ca"
        ),
        "r_pelvic_pain_symptoms_local_ln_prostate_ca": Parameter(
            Types.REAL, "rate of pelvic pain or numbness symptoms if have local lymph node involved prostate cancer"
        ),
        "rr_pelvic_pain_metastatic_prostate_cancer": Parameter(
            Types.REAL,
            "rate ratio of pelvic pain or numbness in a person with metastatic prostate cancer compared with "
            "lymph node involved prostate cancer"
        ),
        "rp_prostate_cancer_age5069": Parameter(
            Types.REAL, "stage-specific relative prevalence at baseline of prostate cancer for age 50-69"
        ),
        "rp_prostate_cancer_agege70": Parameter(
            Types.REAL, "stage-specific relative prevalence at baseline of prostate cancer for age 70+"
        ),
        "sensitivity_of_psa_test_for_prostate_ca": Parameter(
            Types.REAL, "sensitivity of psa test for prostate cancer"
        ),
        "sensitivity_of_biopsy_for_prostate_ca": Parameter(
            Types.REAL, "sensitivity of biopsy for prostate cancer"
        ),
    }

    PROPERTIES = {
        "pc_status": Property(
            Types.CATEGORICAL,
            "Current status of the health condition, prostate cancer",
            categories=["none", "prostate_confined", "local_ln", "metastatic"],
        ),
        "pc_date_psa_test": Property(
            Types.DATE,
            "the date of psa test in response to symptoms"
        ),
        "pc_date_biopsy": Property(
            Types.DATE,
            "the date of biopsy in response to symptoms and positive psa test"
        ),
        "pc_date_diagnosis": Property(
            Types.DATE,
            "the date of diagnosis of the prostate cancer (pd.NaT if never diagnosed)"
        ),

        "pc_date_treatment": Property(
            Types.DATE,
            "date of first receiving attempted curative treatment (pd.NaT if never started treatment)"
        ),

        "pc_stage_at_which_treatment_given": Property(
            Types.CATEGORICAL,
            "the cancer stage at which treatment is given (because the treatment only has an effect during the stage"
            "at which it is given.",
            categories=["none", "prostate_confined", "local_ln", "metastatic"],
        ),

        "pc_date_palliative_care": Property(
            Types.DATE,
            "date of first receiving palliative care (pd.NaT is never had palliative care)"
        ),
        "pc_date_death": Property(
            Types.DATE,
            "date pc death"
        )
    }

    def read_parameters(self, data_folder):
        """Setup parameters used by the module, now including disability weights"""

        # Update parameters from the resourcefile
        self.load_parameters_from_dataframe(
            pd.read_excel(Path(self.resourcefilepath) / "ResourceFile_Prostate_Cancer.xlsx",
                          sheet_name="parameter_values")
        )

        # Register Symptom that this module will use
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='urinary',
                    odds_ratio_health_seeking_in_adults=4.00,
                    no_healthcareseeking_in_children=True)
        )

        # Register Symptom that this module will use
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='pelvic_pain',
                    odds_ratio_health_seeking_in_adults=4.00,
                    no_healthcareseeking_in_children=True)
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
        df.loc[df.is_alive, "pc_date_death"] = pd.NaT
        df.loc[df.is_alive, "pc_date_psa_test"] = pd.NaT
        df.loc[df.is_alive, "pc_date_biopsy"] = pd.NaT

        # -------------------- pc_status -----------
        # Determine who has cancer at ANY cancer stage:
        # check parameters are sensible: probability of having any cancer stage cannot exceed 1.0
        assert sum(p['init_prop_prostate_ca_stage']) <= 1.0

        lm_init_prop_prostate_cancer_stage = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            sum(p['init_prop_prostate_ca_stage']),
            Predictor('sex').when('M', 1.0).otherwise(0.0),
            Predictor('age_years', conditions_are_mutually_exclusive=True)
            .when('.between(50,69)', p['rp_prostate_cancer_age5069'])
            .when('.between(70,120)', p['rp_prostate_cancer_agege70'])
            .when('.between(0,34)', 0.0)
        )

        pc_status_ = \
            lm_init_prop_prostate_cancer_stage.predict(df.loc[df.is_alive], self.rng)

        # Determine the stage of the cancer for those who do have a cancer:
        if pc_status_.sum():
            sum_probs = sum(p['init_prop_prostate_ca_stage'])
            if sum_probs > 0:
                prob_by_stage_of_cancer_if_cancer = [i/sum_probs for i in p['init_prop_prostate_ca_stage']]
                assert (sum(prob_by_stage_of_cancer_if_cancer) - 1.0) < 1e-10
                df.loc[pc_status_, "pc_status"] = self.rng.choice(
                    [val for val in df.pc_status.cat.categories if val != 'none'],
                    size=pc_status_.sum(),
                    p=prob_by_stage_of_cancer_if_cancer
                )

        # -------------------- SYMPTOMS -----------
        # ----- Impose the symptom of random sample of those in each cancer stage to have pelvic pain:
        lm_init_pelvic_pain = LinearModel.multiplicative(
            Predictor(
                'pc_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when("none", 0.0)
            .when("prostate_confined", p['init_prop_pelvic_pain_symptoms_by_stage'][0])
            .when("local_ln", p['init_prop_pelvic_pain_symptoms_by_stage'][1])
            .when("metastatic", p['init_prop_pelvic_pain_symptoms_by_stage'][2])
        )

        has_pelvic_pain_symptoms_at_init = lm_init_pelvic_pain.predict(df.loc[df.is_alive], self.rng)
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=has_pelvic_pain_symptoms_at_init.index[has_pelvic_pain_symptoms_at_init].tolist(),
            symptom_string='pelvic_pain',
            add_or_remove='+',
            disease_module=self
        )

#       above code replaced with below when running for n=1 -

#       self.sim.modules['SymptomManager'].change_symptom(
#           person_id=1,
#           symptom_string='pelvic_pain',
#           add_or_remove='+',
#           disease_module=self
#       )

        # ----- Impose the symptom of random sample of those in each cancer stage to have urinary symptoms:
        lm_init_urinary = LinearModel.multiplicative(
            Predictor(
                'pc_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when("none", 0.0)
            .when("prostate_confined", p['init_prop_urinary_symptoms_by_stage'][0])
            .when("local_ln", p['init_prop_urinary_symptoms_by_stage'][1])
            .when("metastatic",  p['init_prop_urinary_symptoms_by_stage'][2])
        )
        has_urinary_symptoms_at_init = lm_init_urinary.predict(df.loc[df.is_alive], self.rng)
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=has_urinary_symptoms_at_init.index[has_urinary_symptoms_at_init].tolist(),
            symptom_string='urinary',
            add_or_remove='+',
            disease_module=self
        )

#       above code replaced with below when running for n=1 -

#       self.sim.modules['SymptomManager'].change_symptom(
#           person_id=1,
#           symptom_string='pelvic_pain',
#           add_or_remove='+',
#           disease_module=self
#       )
        # -------------------- pc_date_diagnosis -----------

        # for those with symptoms set to initially diagnosed
        # For those that have been diagnosed, set data of diagnosis to today's date
        df.loc[has_urinary_symptoms_at_init, "pc_date_diagnosis"] = self.sim.date
        df.loc[has_pelvic_pain_symptoms_at_init, "pc_date_diagnosis"] = self.sim.date
        df.loc[has_urinary_symptoms_at_init, "pc_date_psa_test"] = self.sim.date
        df.loc[has_pelvic_pain_symptoms_at_init, "pc_date_psa_test"] = self.sim.date
        df.loc[has_urinary_symptoms_at_init, "pc_date_biopsy"] = self.sim.date
        df.loc[has_pelvic_pain_symptoms_at_init, "pc_date_biopsy"] = self.sim.date

        # -------------------- pc_date_treatment -----------
        lm_init_treatment_for_those_diagnosed = LinearModel.multiplicative(
            Predictor(
                'pc_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when("none", 0.0)
            .when("prostate_confined", p['init_prop_treatment_status_prostate_ca'][0])
            .when("local_ln", p['init_prop_treatment_status_prostate_ca'][1])
            .when("metastatic", 0.0)
        )
        treatment_initiated = lm_init_treatment_for_those_diagnosed.predict(df.loc[df.is_alive], self.rng)

        # prevent treatment having been initiated for anyone who is not yet diagnosed
        treatment_initiated.loc[pd.isnull(df.pc_date_diagnosis)] = False

        # assume that the stage at which treatment is begun is the stage the person is in now;
        df.loc[treatment_initiated, "pc_stage_at_which_treatment_given"] = df.loc[treatment_initiated, "pc_status"]

        # set date at which treatment began: same as diagnosis (NB. no HSI is established for this)
        df.loc[treatment_initiated, "pc_date_treatment"] = df.loc[treatment_initiated, "pc_date_diagnosis"]

        # -------------------- pc_date_palliative_care -----------
        in_metastatic_diagnosed = df.index[
            df.is_alive & (df.pc_status == 'metastatic') & ~pd.isnull(df.pc_date_diagnosis)
            ]

        select_for_care = self.rng.random_sample(size=len(in_metastatic_diagnosed)) < p['init_prob_palliative_care']
        select_for_care = in_metastatic_diagnosed[select_for_care]

        # set date of palliative care being initiated: same as diagnosis (NB. future HSI will be scheduled for this)
        df.loc[select_for_care, "pc_date_palliative_care"] = df.loc[select_for_care, "pc_date_diagnosis"]

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
        sim.schedule_event(ProstateCancerLoggingEvent(self), sim.date + DateOffset(months=0))

        # ----- SCHEDULE MAIN POLLING EVENTS -----
        # Schedule main polling event to happen immediately
        sim.schedule_event(ProstateCancerMainPollingEvent(self), sim.date + DateOffset(months=1))

        # ----- LINEAR MODELS -----
        # Define LinearModels for the progression of cancer, in each 3 month period
        # NB. The effect being produced is that treatment only has the effect for during the stage at which the
        # treatment was received.

        df = sim.population.props
        p = self.parameters
        lm = self.linear_models_for_progression_of_pc_status

        lm['prostate_confined'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_prostate_confined_prostate_ca_none'],
            Predictor('sex').when('F', 0),
            Predictor('pc_status').when('prostate_confined', 0),
            Predictor('pc_status').when('local_ln', 0),
            Predictor('pc_status').when('metastatic', 0),
            Predictor('age_years', conditions_are_mutually_exclusive=True)
            .when('.between(50,69)', p['rr_prostate_confined_prostate_ca_age5069'])
            .when('.between(70,120)', p['rr_prostate_confined_prostate_ca_agege70'])
            .when('.between(0,34)', 0.0)
        )

        lm['local_ln'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_local_ln_prostate_ca_prostate_confined'],
            Predictor('had_treatment_during_this_stage',
                      external=True).when(True, p['rr_local_ln_prostate_ca_undergone_curative_treatment']),
            Predictor('pc_status').when('prostate_confined', 1.0)
                                  .otherwise(0.0)
        )

        lm['metastatic'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_metastatic_prostate_ca_local_ln'],
            Predictor('had_treatment_during_this_stage',
                      external=True).when(True, p['rr_metastatic_prostate_ca_undergone_curative_treatment']),
            Predictor('pc_status').when('local_ln', 1.0)
                                  .otherwise(0.0)
        )

        # Check that the dict labels are correct as these are used to set the value of pc_status
        assert set(lm).union({'none'}) == set(df.pc_status.cat.categories)

        # Linear Model for the onset of urinary symptoms in each 3 month period
        self.lm_onset_urinary_symptoms = LinearModel.multiplicative(
            Predictor(
                'pc_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when('prostate_confined', p['r_urinary_symptoms_prostate_ca'])
            .when(
                'local_ln',
                p['rr_urinary_symptoms_local_ln_or_metastatic_prostate_cancer']
                * p['r_urinary_symptoms_prostate_ca']
            )
            .when(
                'metastatic',
                p['rr_urinary_symptoms_local_ln_or_metastatic_prostate_cancer'] *
                p['r_urinary_symptoms_prostate_ca']
            )
            .when('none', 0.0)
        )

        # Linear Model for the onset of pelvic pain symptoms in each 3 month period
        self.lm_onset_pelvic_pain = LinearModel.multiplicative(
            Predictor('pc_status', conditions_are_mutually_exclusive=True)
            .when('local_ln', p['r_pelvic_pain_symptoms_local_ln_prostate_ca'])
            .when(
                'metastatic',
                p['rr_pelvic_pain_metastatic_prostate_cancer']
                * p['r_pelvic_pain_symptoms_local_ln_prostate_ca']
            )
            .otherwise(0.0)
        )

        # ----- DX TESTS -----
        # Create the diagnostic test representing the use of psa test to diagnose prostate cancer
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            psa_for_prostate_cancer=DxTest(
                property='pc_status',
                sensitivity=self.parameters['sensitivity_of_psa_test_for_prostate_ca'],
                target_categories=["prostate_confined", "local_ln", "metastatic"]
            )
        )

        # todo: consider that sensitivity depends on underlying stage

        # Create the diagnostic test representing the use of biopsy to diagnose prostate cancer
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            biopsy_for_prostate_cancer=DxTest(
                property='pc_status',
                sensitivity=self.parameters['sensitivity_of_biopsy_for_prostate_ca'],
                target_categories=["prostate_confined", "local_ln", "metastatic"]
            )
        )

        # ----- DISABILITY-WEIGHT -----
        if "HealthBurden" in self.sim.modules:
            # For those with cancer (any stage prior to stage 4) and never treated
            self.daly_wts["prostate_confined_or_local_ln_untreated"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=550
                # "Diagnosis and primary therapy phase of prostate cancer":
                #  "Cancer, diagnosis and primary therapy ","has pain, nausea, fatigue, weight loss and high anxiety."
            )

            # For those with cancer (any stage prior to stage 4) and has been treated
            self.daly_wts["prostate_confined_or_local_ln_treated"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=547
                # "Controlled phase of prostate cancer,Generic uncomplicated disease":
                # "worry and daily medication,has a chronic disease that requires medication every day and causes some
                #   worry but minimal interference with daily activities".
            )

            # For those in metastatic: no palliative care
            self.daly_wts["metastatic"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=549
                # "Metastatic phase of metastatic cancer:
                # "Cancer, metastatic","has severe pain, extreme fatigue, weight loss and high anxiety."
            )

            # For those in metastatic: with palliative care
            self.daly_wts["metastatic_palliative_care"] = self.daly_wts["prostate_confined_or_local_ln_untreated"]
            # By assumption, we say that that the weight for those with metastatic with palliative care is the same as
            # that for those with prostate_confined_or_local_ln_untreated cancers.

        # ----- HSI FOR PALLIATIVE CARE -----
        on_palliative_care_at_initiation = df.index[df.is_alive & (df.pc_status == "metastatic") &
                                                    ~pd.isnull(df.pc_date_palliative_care)]
        for person_id in on_palliative_care_at_initiation:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_ProstateCancer_PalliativeCare(module=self, person_id=person_id),
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
        df.at[child_id, "pc_status"] = "none"
        df.at[child_id, "pc_date_diagnosis"] = pd.NaT
        df.at[child_id, "pc_date_treatment"] = pd.NaT
        df.at[child_id, "pc_stage_at_which_treatment_given"] = "none"
        df.at[child_id, "pc_date_palliative_care"] = pd.NaT
        df.at[child_id, "pc_date_death"] = pd.NaT
        df.at[child_id, "pc_date_psa_test"] = pd.NaT
        df.at[child_id, "pc_date_biopsy"] = pd.NaT

    def on_hsi_alert(self, person_id, treatment_id):
        pass

    def report_daly_values(self):
        # This must send back a dataframe that reports on the HealthStates for all individuals over
        # the past month

        df = self.sim.population.props  # shortcut to population properties dataframe for alive persons

        disability_series_for_alive_persons = pd.Series(index=df.index[df.is_alive], data=0.0)

        # Assign daly_wt to those with cancer stages before metastatic and have either never been treated or are no
        # longer in the stage in which they were treated
        disability_series_for_alive_persons.loc[
            (
                (df.pc_status == "prostate_confined") |
                (df.pc_status == "local_ln")
            )
        ] = self.daly_wts['prostate_confined_or_local_ln_untreated']

        # Assign daly_wt to those with cancer stages before metastatic and who have been treated and who are still
        # in the
        # stage in which they were treated.
        disability_series_for_alive_persons.loc[
            (
                ~pd.isnull(df.pc_date_treatment) &
                (
                    (df.pc_status == "prostate_confined") |
                    (df.pc_status == "local_ln")
                ) &
                (df.pc_status == df.pc_stage_at_which_treatment_given)
            )
        ] = self.daly_wts['prostate_confined_or_local_ln_treated']

        # Assign daly_wt to those in metastatic cancer (who have not had palliative care)
        disability_series_for_alive_persons.loc[
            (df.pc_status == "metastatic") &
            (pd.isnull(df.pc_date_palliative_care))
            ] = self.daly_wts['metastatic']

        # Assign daly_wt to those in metastatic cancer, who have had palliative care
        disability_series_for_alive_persons.loc[
            (df.pc_status == "metastatic") &
            (~pd.isnull(df.pc_date_palliative_care))
            ] = self.daly_wts['metastatic_palliative_care']

        return disability_series_for_alive_persons


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------

class ProstateCancerMainPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all prostate cancer properties for population:
    * Acquisition and progression of prostate cancer
    * Symptom Development according to stage of prostate cancer
    * Deaths from prostate cancer for those in metastatic
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        # scheduled to run every 1 months: do not change as this is hard-wired into the values of all the parameters.

    def apply(self, population):
        df = population.props  # shortcut to dataframe
        m = self.module
        rng = m.rng

        # -------------------- ACQUISITION AND PROGRESSION OF CANCER (pc_status) -----------------------------------

        # determine if the person had a treatment during this stage of cancer (nb. treatment only has an effect on
        #  reducing progression risk during the stage at which is received.
        had_treatment_during_this_stage = \
            df.is_alive & ~pd.isnull(df.pc_date_treatment) & \
            (df.pc_status == df.pc_stage_at_which_treatment_given)

        # todo: people can move through more than one stage per month (this event runs every month)
        # todo: I am guessing this is somehow a consequence of this way of looping through the stages
        # todo: I imagine this issue is the same for bladder cancer and oesophageal cancer
        for stage, lm in self.module.linear_models_for_progression_of_pc_status.items():
            gets_new_stage = lm.predict(df.loc[df.is_alive], rng,
                                        had_treatment_during_this_stage=had_treatment_during_this_stage)
            idx_gets_new_stage = gets_new_stage[gets_new_stage].index
            df.loc[idx_gets_new_stage, 'pc_status'] = stage

        # -------------------- UPDATING OF SYMPTOM OF URINARY OVER TIME --------------------------------
        # Each time this event is called (event 3 months) individuals may develop urinary symptoms.
        # Once the symptom is developed it resolves with treatment and may trigger health-care-seeking behaviour.
        onset_urinary_symptoms = self.module.lm_onset_urinary_symptoms.predict(df.loc[df.is_alive], rng)
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=onset_urinary_symptoms[onset_urinary_symptoms].index.tolist(),
            symptom_string='urinary',
            add_or_remove='+',
            disease_module=self.module
        )

        # -------------------- UPDATING OF SYMPTOM OF PELVIC PAIN OVER TIME --------------------------------
        # Each time this event is called (event 3 months) individuals may develop pelvic pain symptoms.
        # Once the symptom is developed it resolves with treatment and may trigger health-care-seeking behaviour.
        onset_pelvic_pain = self.module.lm_onset_pelvic_pain.predict(df.loc[df.is_alive], rng)
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=onset_pelvic_pain[onset_pelvic_pain].index.tolist(),
            symptom_string='pelvic_pain',
            add_or_remove='+',
            disease_module=self.module
        )

        # -------------------- DEATH FROM PROSTATE CANCER ---------------------------------------
        # There is a risk of death for those in metastatic only. Death is assumed to go instantly.
        metastatic_idx = df.index[df.is_alive & (df.pc_status == "metastatic")]
        selected_to_die = metastatic_idx[
            rng.random_sample(size=len(metastatic_idx)) < self.module.parameters['r_death_metastatic_prostate_cancer']]

        for person_id in selected_to_die:
            self.sim.schedule_event(
                InstantaneousDeath(self.module, person_id, "ProstateCancer"), self.sim.date
            )
        df.loc[selected_to_die, 'pc_date_death'] = self.sim.date


# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
# ---------------------------------------------------------------------------------------------------------

class HSI_ProstateCancer_Investigation_Following_Urinary_Symptoms(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_GenericFirstApptAtFacilityLevel1 following presentation for care with the symptom
    urinary symptoms.
    This event begins the investigation that may result in diagnosis of prostate cancer and the scheduling of
    treatment or palliative care.
    It is for men with the symptom urinary.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "ProstateCancer_Investigation"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

        # biopsy equipment needed (perhaps ultrasound to guide).  histology lab equipment.

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        # Ignore this event if the person is no longer alive:
        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # Check that this event has been called for someone with the urinary symptoms
        assert 'urinary' in self.sim.modules['SymptomManager'].has_what(person_id)

        # If the person is already diagnosed, then take no action:
        if not pd.isnull(df.at[person_id, "pc_date_diagnosis"]):
            return hs.get_blank_appt_footprint()

        df.at[person_id, 'pc_date_psa_test'] = self.sim.date

        # todo: stratify by pc_status
        # Use a psa test to assess whether the person has prostate cancer:
        dx_result = hs.dx_manager.run_dx_test(
                dx_tests_to_run='psa_for_prostate_cancer',
                hsi_event=self
            )

        if dx_result:
            # send for biopsy
            hs.schedule_hsi_event(
                hsi_event=HSI_ProstateCancer_Investigation_Following_psa_positive(
                    module=self.module,
                    person_id=person_id
                ),
                priority=0,
                topen=self.sim.date,
                tclose=None
            )


class HSI_ProstateCancer_Investigation_Following_Pelvic_Pain(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "ProstateCancer_Investigation"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

    # biopsy equipment needed (perhaps ultrasound to guide).  histology lab equipment.

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        # Ignore this event if the person is no longer alive:
        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # Check that this event has been called for someone with the pelvic pain
        assert 'pelvic_pain' in self.sim.modules['SymptomManager'].has_what(person_id)

        # If the person is already diagnosed, then take no action:
        if not pd.isnull(df.at[person_id, "pc_date_diagnosis"]):
            return hs.get_blank_appt_footprint()

        df.at[person_id, 'pc_date_psa_test'] = self.sim.date

        # todo: stratify by pc_status
        # Use a psa test to assess whether the person has prostate cancer:
        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='psa_for_prostate_cancer',
            hsi_event=self
        )

        if dx_result:
            # send for biopsy
            hs.schedule_hsi_event(
                    hsi_event=HSI_ProstateCancer_Investigation_Following_psa_positive(
                        module=self.module,
                        person_id=person_id
                    ),
                    priority=0,
                    topen=self.sim.date,
                    tclose=None
            )


class HSI_ProstateCancer_Investigation_Following_psa_positive(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "ProstateCancer_Investigation"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

        # biopsy equipment needed (perhaps ultrasound to guide).  histology lab equipment.

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        # Ignore this event if the person is no longer alive:
        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # If the person is already diagnosed, then take no action:
        if not pd.isnull(df.at[person_id, "pc_date_diagnosis"]):
            return hs.get_blank_appt_footprint()

        df.at[person_id, 'pc_date_biopsy'] = self.sim.date

        # todo: stratify by pc_status
        # Use a psa test to assess whether the person has prostate cancer:
        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='biopsy_for_prostate_cancer',
            hsi_event=self
        )

        if dx_result:
            # record date of diagnosis:
            df.at[person_id, 'pc_date_diagnosis'] = self.sim.date

            # Check if is in metastatic stage:
            in_metastatic = df.at[person_id, 'pc_status'] == 'metastatic'
            # If the diagnosis does detect cancer, it is assumed that the classification as metastatic is made
            # accurately.

            if not in_metastatic:
                # start treatment:
                hs.schedule_hsi_event(
                    hsi_event=HSI_ProstateCancer_StartTreatment(
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
                    hsi_event=HSI_ProstateCancer_PalliativeCare(
                        module=self.module,
                        person_id=person_id
                    ),
                    priority=0,
                    topen=self.sim.date,
                    tclose=None
                )


class HSI_ProstateCancer_StartTreatment(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_ProstateOesophagealCancer_Investigation_Following_Urinary_Symptoms.
    It initiates the treatment of prostate cancer.
    It is only for persons with a cancer that is not in metastatic and who have been diagnosed.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "ProstateCancer_Treatment"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"MajorSurg": 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({"general_bed": 5})

        # equipment as required for surgery

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # we don't treat if cancer is metastatic
        if df.at[person_id, "pc_status"] == 'metastatic':
            logger.warning(key="warning", data="Cancer is metastatic- aborting HSI_ProstateCancer_StartTreatment,"
                                               "scheduling HSI_ProstateCancer_PalliativeCare")
            hs.schedule_hsi_event(
                hsi_event=HSI_ProstateCancer_PalliativeCare(
                    module=self.module,
                    person_id=person_id,
                ),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )
            return self.make_appt_footprint({})

        # Check that the person has cancer, not in metastatic, has been diagnosed and is not on treatment
        assert not df.at[person_id, "pc_status"] == 'none'
        # todo: check this line below
        assert not pd.isnull(df.at[person_id, "pc_date_diagnosis"])
        assert pd.isnull(df.at[person_id, "pc_date_treatment"])

        # Record date and stage of starting treatment
        df.at[person_id, "pc_date_treatment"] = self.sim.date
        df.at[person_id, "pc_stage_at_which_treatment_given"] = df.at[person_id, "pc_status"]

        # Schedule a post-treatment check for 12 months:
        hs.schedule_hsi_event(
            hsi_event=HSI_ProstateCancer_PostTreatmentCheck(
                module=self.module,
                person_id=person_id,
            ),
            topen=self.sim.date + DateOffset(months=12),
            tclose=None,
            priority=0
        )


class HSI_ProstateCancer_PostTreatmentCheck(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_ProstateCancer_StartTreatment and itself.
    It is only for those who have undergone treatment for prostate cancer.
    If the person has developed cancer to metastatic, the patient is initiated on palliative care; otherwise a further
    appointment is scheduled for one year.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "ProstateCancer_Treatment"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'

        # possibly biopsy and histology

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # Check that the person is has prostate cancer and is on treatment
        assert not df.at[person_id, "pc_status"] == 'none'
        assert not pd.isnull(df.at[person_id, "pc_date_diagnosis"])
        assert not pd.isnull(df.at[person_id, "pc_date_treatment"])

        if df.at[person_id, 'pc_status'] == 'metastatic':
            # If has progressed to metastatic, then start Palliative Care immediately:
            hs.schedule_hsi_event(
                hsi_event=HSI_ProstateCancer_PalliativeCare(
                    module=self.module,
                    person_id=person_id
                ),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )

        else:
            # Schedule another HSI_ProstateCancer_PostTreatmentCheck event in one month
            hs.schedule_hsi_event(
                hsi_event=HSI_ProstateCancer_PostTreatmentCheck(
                    module=self.module,
                    person_id=person_id
                ),
                topen=self.sim.date + DateOffset(years=1),
                tclose=None,
                priority=0
            )


class HSI_ProstateCancer_PalliativeCare(HSI_Event, IndividualScopeEventMixin):
    """
    This is the event for palliative care. It does not affect the patients progress but does affect the disability
     weight and takes resources from the healthsystem.
    This event is scheduled by either:
    * HSI_ProstateCancer_Investigation_Following_Urinary_Symptoms following a diagnosis of metastatic Prostate Cancer .
    * HSI_ProstateCancer_PostTreatmentCheck following progression to metastatic cancer during treatment.
    * Itself for the continuance of care.
    It is only for persons with metastatic cancer.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "ProstateCancer_PalliativeCare"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
        self.ACCEPTED_FACILITY_LEVEL = '2'
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 15})

        # generally not sure equipment is required as therapy is with drug, but can require castration surgery

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # Check that the person is in metastatic
        assert df.at[person_id, "pc_status"] == 'metastatic'

        # Record the start of palliative care if this is first appointment
        if pd.isnull(df.at[person_id, "pc_date_palliative_care"]):
            df.at[person_id, "pc_date_palliative_care"] = self.sim.date

        # Schedule another instance of the event for one month
        hs.schedule_hsi_event(
            hsi_event=HSI_ProstateCancer_PalliativeCare(
                module=self.module,
                person_id=person_id
            ),
            topen=self.sim.date + DateOffset(months=1),
            tclose=None,
            priority=0
        )


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
# ---------------------------------------------------------------------------------------------------------

class ProstateCancerLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """The only logging event for this module"""

    def __init__(self, module):
        """schedule logging to repeat every 1 months
        """
        self.repeat = 30
        super().__init__(module, frequency=DateOffset(days=self.repeat))

    def apply(self, population):
        """Compute statistics regarding the current status of persons and output to the logger
        """
        df = population.props

        # CURRENT STATUS COUNTS
        # Create dictionary for each subset, adding prefix to key name, and adding to make a flat dict for logging.
        out = {}

        # Current counts, total
        out.update({
            f'total_{k}': v for k, v in df.loc[df.is_alive].pc_status.value_counts().items()})

        # Current counts, undiagnosed
        out.update({f'undiagnosed_{k}': v for k, v in df.loc[df.is_alive].loc[
            pd.isnull(df.pc_date_diagnosis), 'pc_status'].value_counts().items()})

        # Current counts, diagnosed
        out.update({f'diagnosed_{k}': v for k, v in df.loc[df.is_alive].loc[
            ~pd.isnull(df.pc_date_diagnosis), 'pc_status'].value_counts().items()})

        # Current counts, on treatment (excl. palliative care)
        out.update({f'treatment_{k}': v for k, v in df.loc[df.is_alive].loc[(~pd.isnull(
            df.pc_date_treatment) & pd.isnull(
            df.pc_date_palliative_care)), 'pc_status'].value_counts().items()})

        # Current counts, on palliative care
        out.update({f'palliative_{k}': v for k, v in df.loc[df.is_alive].loc[
            ~pd.isnull(df.pc_date_palliative_care), 'pc_status'].value_counts().items()})

        # Counts of those that have been diagnosed, started treatment or started palliative care since last logging
        # event:
        date_now = self.sim.date
        date_lastlog = self.sim.date - pd.DateOffset(days=29)

        n_ge35_m = (df.is_alive & (df.age_years >= 35) & (df.sex == 'M')).sum()

        # todo: the .between function I think includes the two dates so events on these dates counted twice
        # todo:_ I think we need to replace with date_lastlog <= x < date_now
        n_newly_diagnosed_prostate_confined = (
                df.pc_date_diagnosis.between(date_lastlog, date_now) & (df.pc_status == 'prostate_confined')).sum()
        n_newly_diagnosed_local_ln = (
                df.pc_date_diagnosis.between(date_lastlog, date_now) & (df.pc_status == 'local_ln')).sum()
        n_newly_diagnosed_metastatic = (
                df.pc_date_diagnosis.between(date_lastlog, date_now) & (df.pc_status == 'metastatic')).sum()

        n_diagnosed = (df.is_alive & ~pd.isnull(df.pc_date_diagnosis)).sum()

        out.update({
            'diagnosed_since_last_log': df.pc_date_diagnosis.between(date_lastlog, date_now).sum(),
            'treated_since_last_log': df.pc_date_treatment.between(date_lastlog, date_now).sum(),
            'palliative_since_last_log': df.pc_date_palliative_care.between(date_lastlog, date_now).sum(),
            'death_prostate_cancer_since_last_log': df.pc_date_death.between(date_lastlog, date_now).sum(),
            'n_men age 35+': n_ge35_m,
            'n_newly_diagnosed_prostate_confined1': n_newly_diagnosed_prostate_confined,
            'n_newly_diagnosed_local_ln': n_newly_diagnosed_local_ln,
            'n_newly_diagnosed_metastatic': n_newly_diagnosed_metastatic,
            'n_diagnosed': n_diagnosed
        })

        logger.info(key='summary_stats',
                    data=out)
        #       logger.info('%s|person_one|%s',
        #                    self.sim.date,
        #                    df.loc[ 8].to_dict())
