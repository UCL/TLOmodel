"""
Breast Cancer Disease Module

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


class BreastCancer(Module):
    """Breast Cancer Disease Module"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.linear_models_for_progession_of_brc_status = dict()
        self.lm_onset_breast_lump_discernible = None
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
        'BreastCancer': Cause(gbd_causes='Breast cancer', label='Cancer'),
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'BreastCancer': Cause(gbd_causes='Breast cancer', label='Cancer'),
    }

    PARAMETERS = {
        "init_prop_breast_cancer_stage": Parameter(
            Types.LIST,
            "initial proportions in cancer categories for woman aged 15-29"
        ),
        "init_prop_breast_lump_discernible_breast_cancer_by_stage": Parameter(
            Types.LIST, "initial proportions of those with cancer categories that have the symptom breast_lump"
                        "_discernible"
        ),
        "init_prop_with_breast_lump_discernible_diagnosed_breast_cancer_by_stage": Parameter(
            Types.LIST, "initial proportions of people that have breast_lump_discernible that have been diagnosed"
        ),
        "init_prop_treatment_status_breast_cancer": Parameter(
            Types.LIST, "initial proportions of people with breast cancer previously treated"
        ),
        "init_prob_palliative_care": Parameter(
            Types.REAL, "initial probability of being under palliative care if in stage 4"
        ),
        "r_stage1_none": Parameter(
            Types.REAL,
            "probabilty per 3 months of incident stage 1 breast, amongst people with no "
            "breast cancer",
        ),
        "rr_stage1_none_age3049": Parameter(
            Types.REAL, "rate ratio for stage1 breast cancer for age 30-49"
        ),
        "rr_stage1_none_agege50": Parameter(
            Types.REAL, "rate ratio for stage1 breast cancer for age 50+"
        ),
        "r_stage2_stage1": Parameter(
            Types.REAL, "probabilty per 3 months of stage 2 breast cancer amongst people with stage 1"
        ),
        "rr_stage2_undergone_curative_treatment": Parameter(
            Types.REAL,
            "rate ratio for stage 2 breast cancer for people with stage 1 "
            "breast cancer if had curative treatment at stage 1",
        ),
        "r_stage3_stage2": Parameter(
            Types.REAL, "probabilty per 3 months of stage 3 breast cancer amongst people with stage 2"
        ),
        "rr_stage3_undergone_curative_treatment": Parameter(
            Types.REAL,
            "rate ratio for stage 3 breast cancer for people with stage 2 "
            "breast cancer if had curative treatment at stage 2",
        ),
        "r_stage4_stage3": Parameter(
            Types.REAL, "probabilty per 3 months of stage 4 breast cancer amongst people with stage 3"
        ),
        "rr_stage4_undergone_curative_treatment": Parameter(
            Types.REAL,
            "rate ratio for stage 4 breast cancer for people with stage 3 "
            "breast cancer if had curative treatment at stage 3",
        ),
        "r_death_breast_cancer": Parameter(
            Types.REAL,
            "probabilty per 3 months of death from breast cancer amongst people with stage 4 breast cancer",
        ),
        "r_breast_lump_discernible_stage1": Parameter(
            Types.REAL, "rate ratio for breast_lump_discernible if have stage 1 breast cancer"
        ),
        "rr_breast_lump_discernible_stage2": Parameter(
            Types.REAL, "rate ratio for breast_lump_discernible if have stage 2 breast cancer"
        ),
        "rr_breast_lump_discernible_stage3": Parameter(
            Types.REAL, "rate ratio for breast_lump_discernible if have stage 3 breast cancer"
        ),
        "rr_breast_lump_discernible_stage4": Parameter(
            Types.REAL, "rate ratio for breast_lump_discernible if have stage 4 breast cancer"
        ),
        "rp_breast_cancer_age3049": Parameter(
            Types.REAL, "relative prevalence at baseline of breast cancer if age3049"
        ),
        "rp_breast_cancer_agege50": Parameter(
            Types.REAL, "relative prevalence at baseline of breast cancer if agege50"
        ),
        "sensitivity_of_biopsy_for_stage1_breast_cancer": Parameter(
            Types.REAL, "sensitivity of biopsy_for diagnosis of stage 1 breast cancer"
        ),
        "sensitivity_of_biopsy_for_stage2_breast_cancer": Parameter(
            Types.REAL, "sensitivity of biopsy_for diagnosis of stage 2 breast cancer"
        ),
        "sensitivity_of_biopsy_for_stage3_breast_cancer": Parameter(
            Types.REAL, "sensitivity of biopsy_for diagnosis of stage 3 breast cancer"
        ),
        "sensitivity_of_biopsy_for_stage4_breast_cancer": Parameter(
            Types.REAL, "sensitivity of biopsy_for diagnosis of stage 4 breast cancer"
        ),
    }

    PROPERTIES = {
        "brc_status": Property(
            Types.CATEGORICAL,
            "Current status of the health condition, breast cancer",
            categories=["none", "stage1", "stage2", "stage3", "stage4"],
        ),

        "brc_date_diagnosis": Property(
            Types.DATE,
            "the date of diagnosis of the breast_cancer (pd.NaT if never diagnosed)"
        ),

        "brc_date_treatment": Property(
            Types.DATE,
            "date of first receiving attempted curative treatment (pd.NaT if never started treatment)"
        ),
        "brc_breast_lump_discernible_investigated": Property(
            Types.BOOL,
            "whether a breast_lump_discernible has been investigated, and cancer missed"
        ),
        "brc_stage_at_which_treatment_given": Property(
            Types.CATEGORICAL,
            "the cancer stage at which treatment is given (because the treatment only has an effect during the stage"
            "at which it is given).",
            categories=["none", "stage1", "stage2", "stage3", "stage4"],
        ),
        "brc_date_palliative_care": Property(
            Types.DATE,
            "date of first receiving palliative care (pd.NaT is never had palliative care)"
        ),
        "brc_date_death": Property(
            Types.DATE,
            "date of brc death"
        ),
        "brc_new_stage_this_month": Property(
            Types.BOOL,
            "new_stage_this month"
        )
    }

    def read_parameters(self, data_folder):
        """Setup parameters used by the module, now including disability weights"""

        # Update parameters from the resourcefile
        self.load_parameters_from_dataframe(
            pd.read_excel(Path(self.resourcefilepath) / "ResourceFile_Breast_Cancer.xlsx",
                          sheet_name="parameter_values")
        )

        # Register Symptom that this module will use
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='breast_lump_discernible',
                    odds_ratio_health_seeking_in_adults=4.00)
        )

    def initialise_population(self, population):
        """Set property values for the initial population."""
        df = population.props  # a shortcut to the data-frame
        p = self.parameters

        # defaults
        df.loc[df.is_alive, "brc_status"] = "none"
        df.loc[df.is_alive, "brc_date_diagnosis"] = pd.NaT
        df.loc[df.is_alive, "brc_date_treatment"] = pd.NaT
        df.loc[df.is_alive, "brc_stage_at_which_treatment_given"] = "none"
        df.loc[df.is_alive, "brc_date_palliative_care"] = pd.NaT
        df.loc[df.is_alive, "brc_date_death"] = pd.NaT
        df.loc[df.is_alive, "brc_breast_lump_discernible_investigated"] = False
        df.loc[df.is_alive, "brc_new_stage_this_month"] = False

        # -------------------- brc_status -----------
        # Determine who has cancer at ANY cancer stage:
        # check parameters are sensible: probability of having any cancer stage cannot exceed 1.0
        assert sum(p['init_prop_breast_cancer_stage']) <= 1.0

        lm_init_brc_status_any_stage = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            sum(p['init_prop_breast_cancer_stage']),
            Predictor('sex').when('F', 1.0).otherwise(0.0),
            Predictor('age_years', conditions_are_mutually_exclusive=True)
            .when('.between(30,49)', p['rp_breast_cancer_age3049'])
            .when('.between(0,14)', 0.0)
            .when('.between(50,120)', p['rp_breast_cancer_agege50']),
        )

        brc_status_any_stage = \
            lm_init_brc_status_any_stage.predict(df.loc[df.is_alive], self.rng)

        # Determine the stage of the cancer for those who do have a cancer:
        if brc_status_any_stage.sum():
            sum_probs = sum(p['init_prop_breast_cancer_stage'])
            if sum_probs > 0:
                prob_by_stage_of_cancer_if_cancer = [i/sum_probs for i in p['init_prop_breast_cancer_stage']]
                assert (sum(prob_by_stage_of_cancer_if_cancer) - 1.0) < 1e-10
                df.loc[brc_status_any_stage, "brc_status"] = self.rng.choice(
                    [val for val in df.brc_status.cat.categories if val != 'none'],
                    size=brc_status_any_stage.sum(),
                    p=prob_by_stage_of_cancer_if_cancer
                )

        # -------------------- SYMPTOMS -----------
        # ----- Impose the symptom of random sample of those in each cancer stage to have the symptom of breast_
        # lump_discernible:
        # todo: note dysphagia was mis-spelled here in oesophageal cancer module in master so may not be working
        # Create shorthand variable for the initial proportion of discernible breast cancer lumps in the population
        bc_init_prop_discernible_lump = p['init_prop_breast_lump_discernible_breast_cancer_by_stage']
        lm_init_breast_lump_discernible = LinearModel.multiplicative(
            Predictor(
                'brc_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when("none", 0.0)
            .when("stage1", bc_init_prop_discernible_lump[0])
            .when("stage2", bc_init_prop_discernible_lump[1])
            .when("stage3", bc_init_prop_discernible_lump[2])
            .when("stage4", bc_init_prop_discernible_lump[3])
        )

        has_breast_lump_discernible_at_init = lm_init_breast_lump_discernible.predict(df.loc[df.is_alive], self.rng)
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=has_breast_lump_discernible_at_init.index[has_breast_lump_discernible_at_init].tolist(),
            symptom_string='breast_lump_discernible',
            add_or_remove='+',
            disease_module=self
        )

        # -------------------- brc_date_diagnosis -----------
        # Create shorthand variable for the initial proportion of the population with a discernible breast lump that has
        # been diagnosed
        bc_initial_prop_diagnosed_discernible_lump = \
            p['init_prop_with_breast_lump_discernible_diagnosed_breast_cancer_by_stage']
        lm_init_diagnosed = LinearModel.multiplicative(
            Predictor(
                'brc_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when("none", 0.0)
            .when("stage1", bc_initial_prop_diagnosed_discernible_lump[0])
            .when("stage2", bc_initial_prop_diagnosed_discernible_lump[1])
            .when("stage3", bc_initial_prop_diagnosed_discernible_lump[2])
            .when("stage4", bc_initial_prop_diagnosed_discernible_lump[3])
        )
        ever_diagnosed = lm_init_diagnosed.predict(df.loc[df.is_alive], self.rng)

        # ensure that persons who have not ever had the symptom breast_lump_discernible are diagnosed:
        ever_diagnosed.loc[~has_breast_lump_discernible_at_init] = False

        # For those that have been diagnosed, set data of diagnosis to today's date
        df.loc[ever_diagnosed, "brc_date_diagnosis"] = self.sim.date

        # -------------------- brc_date_treatment -----------
        # create short hand variable for the predicting the initial occurence of various breast
        # cancer stages in the population
        bc_inital_treament_status = p['init_prop_treatment_status_breast_cancer']
        lm_init_treatment_for_those_diagnosed = LinearModel.multiplicative(
            Predictor(
                'brc_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when("none", 0.0)
            .when("stage1", bc_inital_treament_status[0])
            .when("stage2", bc_inital_treament_status[1])
            .when("stage3", bc_inital_treament_status[2])
            .when("stage4", bc_inital_treament_status[3])
        )
        treatment_initiated = lm_init_treatment_for_those_diagnosed.predict(df.loc[df.is_alive], self.rng)

        # prevent treatment having been initiated for anyone who is not yet diagnosed
        treatment_initiated.loc[pd.isnull(df.brc_date_diagnosis)] = False

        # assume that the stage at which treatment is begun is the stage the person is in now;
        df.loc[treatment_initiated, "brc_stage_at_which_treatment_given"] = df.loc[treatment_initiated, "brc_status"]

        # set date at which treatment began: same as diagnosis (NB. no HSI is established for this)
        df.loc[treatment_initiated, "brc_date_treatment"] = df.loc[treatment_initiated, "brc_date_diagnosis"]

        # -------------------- brc_date_palliative_care -----------
        in_stage4_diagnosed = df.index[df.is_alive & (df.brc_status == 'stage4') & ~pd.isnull(df.brc_date_diagnosis)]

        select_for_care = self.rng.random_sample(size=len(in_stage4_diagnosed)) < p['init_prob_palliative_care']
        select_for_care = in_stage4_diagnosed[select_for_care]

        # set date of palliative care being initiated: same as diagnosis (NB. future HSI will be scheduled for this)
        df.loc[select_for_care, "brc_date_palliative_care"] = df.loc[select_for_care, "brc_date_diagnosis"]

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
        sim.schedule_event(BreastCancerLoggingEvent(self), sim.date + DateOffset(months=0))

        # ----- SCHEDULE MAIN POLLING EVENTS -----
        # Schedule main polling event to happen immediately
        sim.schedule_event(BreastCancerMainPollingEvent(self), sim.date + DateOffset(months=1))

        # ----- LINEAR MODELS -----
        # Define LinearModels for the progression of cancer, in each 3 month period
        # NB. The effect being produced is that treatment only has the effect for during the stage at which the
        # treatment was received.

        df = sim.population.props
        p = self.parameters
        lm = self.linear_models_for_progession_of_brc_status

        lm['stage1'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage1_none'],
            Predictor('sex').when('M', 0.0),
            Predictor('brc_status').when('none', 1.0).otherwise(0.0),
            Predictor('age_years', conditions_are_mutually_exclusive=True)
            .when('.between(0,14)', 0.0)
            .when('.between(30,49)', p['rr_stage1_none_age3049'])
            .when('.between(50,120)', p['rr_stage1_none_agege50'])
        )

        lm['stage2'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage2_stage1'],
            Predictor('had_treatment_during_this_stage',
                      external=True).when(True, p['rr_stage2_undergone_curative_treatment']),
            Predictor('brc_status').when('stage1', 1.0).otherwise(0.0),
            Predictor('brc_new_stage_this_month').when(True, 0.0).otherwise(1.0)
        )

        lm['stage3'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage3_stage2'],
            Predictor('had_treatment_during_this_stage',
                      external=True).when(True, p['rr_stage3_undergone_curative_treatment']),
            Predictor('brc_status').when('stage2', 1.0).otherwise(0.0),
            Predictor('brc_new_stage_this_month').when(True, 0.0).otherwise(1.0)
        )

        lm['stage4'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage4_stage3'],
            Predictor('had_treatment_during_this_stage',
                      external=True).when(True, p['rr_stage4_undergone_curative_treatment']),
            Predictor('brc_status').when('stage3', 1.0).otherwise(0.0),
            Predictor('brc_new_stage_this_month').when(True, 0.0).otherwise(1.0)
        )

        # Check that the dict labels are correct as these are used to set the value of brc_status
        assert set(lm).union({'none'}) == set(df.brc_status.cat.categories)

        # Linear Model for the onset of breast_lump_discernible, in each 3 month period
        # Create variables for used to predict the onset of discernible breast lumps at
        # various stages of the disease
        stage1 = p['r_breast_lump_discernible_stage1']
        stage2 = p['rr_breast_lump_discernible_stage2'] * p['r_breast_lump_discernible_stage1']
        stage3 = p['rr_breast_lump_discernible_stage3'] * p['r_breast_lump_discernible_stage1']
        stage4 = p['rr_breast_lump_discernible_stage4'] * p['r_breast_lump_discernible_stage1']
        self.lm_onset_breast_lump_discernible = LinearModel.multiplicative(
            Predictor(
                'brc_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when('stage1', stage1)
            .when('stage2', stage2)
            .when('stage3', stage3)
            .when('stage4', stage4)
            .when('none', 0.0)
        )

        # ----- DX TESTS -----
        # Create the diagnostic test representing the use of a biopsy to brc_status
        # This properties of conditional on the test being done only to persons with the Symptom, 'breast_lump_
        # discernible'.
        # todo: depends on underlying stage not symptoms
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            biopsy_for_breast_cancer_given_breast_lump_discernible=DxTest(
                property='brc_status',
                sensitivity=self.parameters['sensitivity_of_biopsy_for_stage1_breast_cancer'],
                target_categories=["stage1", "stage2", "stage3", "stage4"]
            )
        )

        # todo: possibly un-comment out below when can discuss with Tim
        """
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            biopsy_for_breast_cancer_stage2=DxTest(
                property='brc_status',
                sensitivity=self.parameters['sensitivity_of_biopsy_for_stage2_breast_cancer'],
                target_categories=["stage1", "stage2", "stage3", "stage4"]
            )
        )

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            biopsy_for_breast_cancer_stage3=DxTest(
                property='brc_status',
                sensitivity=self.parameters['sensitivity_of_biopsy_for_stage3_breast_cancer'],
                target_categories=["stage1", "stage2", "stage3", "stage4"]
            )
        )

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            biopsy_for_breast_cancer_stage4=DxTest(
                property='brc_status',
                sensitivity=self.parameters['sensitivity_of_biopsy_for_stage4_breast_cancer'],
                target_categories=["stage1", "stage2", "stage3", "stage4"]
            )
        )
        """
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
        on_palliative_care_at_initiation = df.index[df.is_alive & ~pd.isnull(df.brc_date_palliative_care)]
        for person_id in on_palliative_care_at_initiation:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_BreastCancer_PalliativeCare(module=self, person_id=person_id),
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
        df.at[child_id, "brc_status"] = "none"
        df.at[child_id, "brc_date_diagnosis"] = pd.NaT
        df.at[child_id, "brc_date_treatment"] = pd.NaT
        df.at[child_id, "brc_stage_at_which_treatment_given"] = "none"
        df.at[child_id, "brc_date_palliative_care"] = pd.NaT
        df.at[child_id, "brc_new_stage_this_month"] = False
        df.at[child_id, "brc_breast_lump_discernible_investigated"] = False
        df.at[child_id, "brc_date_death"] = pd.NaT

    def on_hsi_alert(self, person_id, treatment_id):
        pass

    def report_daly_values(self):

        # This must send back a dataframe that reports on the HealthStates for all individuals over the past month

        df = self.sim.population.props  # shortcut to population properties dataframe for alive persons

        disability_series_for_alive_persons = pd.Series(index=df.index[df.is_alive], data=0.0)

        # Assign daly_wt to those with cancer stages before stage4 and have either never been treated or are no longer
        # in the stage in which they were treated
        disability_series_for_alive_persons.loc[
            (
                (df.brc_status == "stage1") |
                (df.brc_status == "stage2") |
                (df.brc_status == "stage3")
            )
        ] = self.daly_wts['stage_1_3']

        # Assign daly_wt to those with cancer stages before stage4 and who have been treated and who are still in the
        # stage in which they were treated.
        disability_series_for_alive_persons.loc[
            (
                ~pd.isnull(df.brc_date_treatment) & (
                    (df.brc_status == "stage1") |
                    (df.brc_status == "stage2") |
                    (df.brc_status == "stage3")
                ) & (df.brc_status == df.brc_stage_at_which_treatment_given)
            )
        ] = self.daly_wts['stage_1_3_treated']

        # Assign daly_wt to those in stage4 cancer (who have not had palliative care)
        disability_series_for_alive_persons.loc[
            (df.brc_status == "stage4") &
            (pd.isnull(df.brc_date_palliative_care))
            ] = self.daly_wts['stage4']

        # Assign daly_wt to those in stage4 cancer, who have had palliative care
        disability_series_for_alive_persons.loc[
            (df.brc_status == "stage4") &
            (~pd.isnull(df.brc_date_palliative_care))
            ] = self.daly_wts['stage4_palliative_care']

        return disability_series_for_alive_persons


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------

class BreastCancerMainPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all breast cancer properties for population:
    * Acquisition and progression of breast Cancer
    * Symptom Development according to stage of breast Cancer
    * Deaths from breast Cancer for those in stage4
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        # scheduled to run every 3 months: do not change as this is hard-wired into the values of all the parameters.

    def apply(self, population):
        df = population.props  # shortcut to dataframe
        m = self.module
        rng = m.rng

        # -------------------- ACQUISITION AND PROGRESSION OF CANCER (brc_status) -----------------------------------

        df.brc_new_stage_this_month = False

        # determine if the person had a treatment during this stage of cancer (nb. treatment only has an effect on
        #  reducing progression risk during the stage at which is received.
        had_treatment_during_this_stage = \
            df.is_alive & ~pd.isnull(df.brc_date_treatment) & \
            (df.brc_status == df.brc_stage_at_which_treatment_given)

        for stage, lm in self.module.linear_models_for_progession_of_brc_status.items():
            gets_new_stage = lm.predict(df.loc[df.is_alive], rng,
                                        had_treatment_during_this_stage=had_treatment_during_this_stage)
            idx_gets_new_stage = gets_new_stage[gets_new_stage].index
            df.loc[idx_gets_new_stage, 'brc_status'] = stage
            df.loc[idx_gets_new_stage, 'brc_new_stage_this_month'] = True

        # todo: people can move through more than one stage per month (this event runs every month)
        # todo: I am guessing this is somehow a consequence of this way of looping through the stages
        # todo: I imagine this issue is the same for bladder cancer and oesophageal cancer

        # -------------------- UPDATING OF SYMPTOM OF breast_lump_discernible OVER TIME --------------------------------
        # Each time this event is called (event 3 months) individuals may develop the symptom of breast_lump_
        # discernible.
        # Once the symptom is developed it never resolves naturally. It may trigger health-care-seeking behaviour.
        onset_breast_lump_discernible = self.module.lm_onset_breast_lump_discernible.predict(df.loc[df.is_alive], rng)
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=onset_breast_lump_discernible[onset_breast_lump_discernible].index.tolist(),
            symptom_string='breast_lump_discernible',
            add_or_remove='+',
            disease_module=self.module
        )

        # -------------------- DEATH FROM breast CANCER ---------------------------------------
        # There is a risk of death for those in stage4 only. Death is assumed to go instantly.
        stage4_idx = df.index[df.is_alive & (df.brc_status == "stage4")]
        selected_to_die = stage4_idx[
            rng.random_sample(size=len(stage4_idx)) < self.module.parameters['r_death_breast_cancer']]

        for person_id in selected_to_die:
            self.sim.schedule_event(
                InstantaneousDeath(self.module, person_id, "BreastCancer"), self.sim.date
            )
            df.loc[selected_to_die, 'brc_date_death'] = self.sim.date

    # ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
# ---------------------------------------------------------------------------------------------------------


class HSI_BreastCancer_Investigation_Following_breast_lump_discernible(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_GenericFirstApptAtFacilityLevel1 following presentation for care with the symptom
    breast_lump_discernible.
    This event begins the investigation that may result in diagnosis of breast Cancer and the scheduling of
    treatment or palliative care.
    It is for people with the symptom breast_lump_discernible.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "BreastCancer_Investigation"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1, "Mammography": 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'  # Mammography only available at level 3 and above.

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        # Ignore this event if the person is no longer alive:
        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # Check that this event has been called for someone with the symptom breast_lump_discernible
        assert 'breast_lump_discernible' in self.sim.modules['SymptomManager'].has_what(person_id)

        # If the person is already diagnosed, then take no action:
        if not pd.isnull(df.at[person_id, "brc_date_diagnosis"]):
            return hs.get_blank_appt_footprint()

        df.brc_breast_lump_discernible_investigated = True

        # Use a biopsy to diagnose whether the person has breast Cancer:
        # todo: request consumables needed for this

        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='biopsy_for_breast_cancer_given_breast_lump_discernible',
            hsi_event=self
        )

        if dx_result:
            # record date of diagnosis:
            df.at[person_id, 'brc_date_diagnosis'] = self.sim.date

            # Check if is in stage4:
            in_stage4 = df.at[person_id, 'brc_status'] == 'stage4'
            # If the diagnosis does detect cancer, it is assumed that the classification as stage4 is made accurately.

            if not in_stage4:
                # start treatment:
                hs.schedule_hsi_event(
                    hsi_event=HSI_BreastCancer_StartTreatment(
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
                    hsi_event=HSI_BreastCancer_PalliativeCare(
                        module=self.module,
                        person_id=person_id
                    ),
                    priority=0,
                    topen=self.sim.date,
                    tclose=None
                )

#   todo: we would like to note that the symptom has been investigated in a diagnostic test and the diagnosis was
#   todo: was missed, so the same test will not likely be repeated, at least not in the short term, so we even
#   todo: though the symptom remains we don't want to keep repeating the HSI which triggers the diagnostic test


class HSI_BreastCancer_StartTreatment(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_BreastCancer_Investigation_Following_breast_lump_discernible following a diagnosis of
    breast Cancer. It initiates the treatment of breast Cancer.
    It is only for persons with a cancer that is not in stage4 and who have been diagnosed.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "BreastCancer_Treatment"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"MajorSurg": 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({"general_bed": 5})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        # todo: request consumables needed for this

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # If the status is already in `stage4`, start palliative care (instead of treatment)
        if df.at[person_id, "brc_status"] == 'stage4':
            logger.warning(key="warning", data="Cancer is in stage 4 - aborting HSI_breastCancer_StartTreatment,"
                                               "scheduling HSI_BreastCancer_PalliativeCare")

            hs.schedule_hsi_event(
                hsi_event=HSI_BreastCancer_PalliativeCare(
                     module=self.module,
                     person_id=person_id,
                ),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )
            return self.make_appt_footprint({})

        # Check that the person has been diagnosed and is not on treatment
        assert not df.at[person_id, "brc_status"] == 'none'
        assert not df.at[person_id, "brc_status"] == 'stage4'
        assert not pd.isnull(df.at[person_id, "brc_date_diagnosis"])
        assert pd.isnull(df.at[person_id, "brc_date_treatment"])

        # Record date and stage of starting treatment
        df.at[person_id, "brc_date_treatment"] = self.sim.date
        df.at[person_id, "brc_stage_at_which_treatment_given"] = df.at[person_id, "brc_status"]

        # Schedule a post-treatment check for 12 months:
        hs.schedule_hsi_event(
            hsi_event=HSI_BreastCancer_PostTreatmentCheck(
                module=self.module,
                person_id=person_id,
            ),
            topen=self.sim.date + DateOffset(months=12),
            tclose=None,
            priority=0
        )


class HSI_BreastCancer_PostTreatmentCheck(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_BreastCancer_StartTreatment and itself.
    It is only for those who have undergone treatment for breast Cancer.
    If the person has developed cancer to stage4, the patient is initiated on palliative care; otherwise a further
    appointment is scheduled for one year.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "BreastCancer_Treatment"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # Check that the person is has cancer and is on treatment
        assert not df.at[person_id, "brc_status"] == 'none'
        assert not pd.isnull(df.at[person_id, "brc_date_diagnosis"])
        assert not pd.isnull(df.at[person_id, "brc_date_treatment"])

        if df.at[person_id, 'brc_status'] == 'stage4':
            # If has progressed to stage4, then start Palliative Care immediately:
            hs.schedule_hsi_event(
                hsi_event=HSI_BreastCancer_PalliativeCare(
                    module=self.module,
                    person_id=person_id
                ),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )

        else:
            # Schedule another HSI_BreastCancer_PostTreatmentCheck event in one month
            hs.schedule_hsi_event(
                hsi_event=HSI_BreastCancer_PostTreatmentCheck(
                    module=self.module,
                    person_id=person_id
                ),
                topen=self.sim.date + DateOffset(months=3),
                tclose=None,
                priority=0
            )


class HSI_BreastCancer_PalliativeCare(HSI_Event, IndividualScopeEventMixin):
    """
    This is the event for palliative care. It does not affect the patients progress but does affect the disability
     weight and takes resources from the healthsystem.
    This event is scheduled by either:
    * HSI_BreastCancer_Investigation_Following_breast_lump_discernible following a diagnosis of breast Cancer at stage4.
    * HSI_BreastCancer_PostTreatmentCheck following progression to stage4 during treatment.
    * Itself for the continuance of care.
    It is only for persons with a cancer in stage4.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "BreastCancer_PalliativeCare"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
        self.ACCEPTED_FACILITY_LEVEL = '2'
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 15})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        # todo: request consumables needed for this

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # Check that the person is in stage4
        assert df.at[person_id, "brc_status"] == 'stage4'

        # Record the start of palliative care if this is first appointment
        if pd.isnull(df.at[person_id, "brc_date_palliative_care"]):
            df.at[person_id, "brc_date_palliative_care"] = self.sim.date

        # Schedule another instance of the event for one month
        hs.schedule_hsi_event(
            hsi_event=HSI_BreastCancer_PalliativeCare(
                module=self.module,
                person_id=person_id
            ),
            topen=self.sim.date + DateOffset(months=3),
            tclose=None,
            priority=0
        )


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
# ---------------------------------------------------------------------------------------------------------

class BreastCancerLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """The only logging event for this module"""

    def __init__(self, module):
        """schedule logging to repeat every 1 month
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
            f'total_{k}': v for k, v in df.loc[df.is_alive].brc_status.value_counts().items()})

        # Current counts, undiagnosed
        out.update({f'undiagnosed_{k}': v for k, v in df.loc[df.is_alive].loc[
            pd.isnull(df.brc_date_diagnosis), 'brc_status'].value_counts().items()})

        # Current counts, diagnosed
        out.update({f'diagnosed_{k}': v for k, v in df.loc[df.is_alive].loc[
            ~pd.isnull(df.brc_date_diagnosis), 'brc_status'].value_counts().items()})

        # Current counts, on treatment (excl. palliative care)
        out.update({f'treatment_{k}': v for k, v in df.loc[df.is_alive].loc[(~pd.isnull(
            df.brc_date_treatment) & pd.isnull(
            df.brc_date_palliative_care)), 'brc_status'].value_counts().items()})

        # Current counts, on palliative care
        out.update({f'palliative_{k}': v for k, v in df.loc[df.is_alive].loc[
            ~pd.isnull(df.brc_date_palliative_care), 'brc_status'].value_counts().items()})

        # Counts of those that have been diagnosed, started treatment or started palliative care since last logging
        # event:
        date_now = self.sim.date
        date_lastlog = self.sim.date - pd.DateOffset(days=29)

        n_ge15_f = (df.is_alive & (df.age_years >= 15) & (df.sex == 'F')).sum()

        # todo: the .between function I think includes the two dates so events on these dates counted twice
        # todo:_ I think we need to replace with date_lastlog <= x < date_now
        n_newly_diagnosed_stage1 = \
            (df.brc_date_diagnosis.between(date_lastlog, date_now) & (df.brc_status == 'stage1')).sum()
        n_newly_diagnosed_stage2 = \
            (df.brc_date_diagnosis.between(date_lastlog, date_now) & (df.brc_status == 'stage2')).sum()
        n_newly_diagnosed_stage3 = \
            (df.brc_date_diagnosis.between(date_lastlog, date_now) & (df.brc_status == 'stage3')).sum()
        n_newly_diagnosed_stage4 = \
            (df.brc_date_diagnosis.between(date_lastlog, date_now) & (df.brc_status == 'stage4')).sum()

        n_diagnosed_age_15_29 = (df.is_alive & (df.age_years >= 15) & (df.age_years < 30)
                                 & ~pd.isnull(df.brc_date_diagnosis)).sum()
        n_diagnosed_age_30_49 = (df.is_alive & (df.age_years >= 30) & (df.age_years < 50)
                                 & ~pd.isnull(df.brc_date_diagnosis)).sum()
        n_diagnosed_age_50p = (df.is_alive & (df.age_years >= 50) & ~pd.isnull(df.brc_date_diagnosis)).sum()

        n_diagnosed = (df.is_alive & ~pd.isnull(df.brc_date_diagnosis)).sum()

        out.update({
            'diagnosed_since_last_log': df.brc_date_diagnosis.between(date_lastlog, date_now).sum(),
            'treated_since_last_log': df.brc_date_treatment.between(date_lastlog, date_now).sum(),
            'palliative_since_last_log': df.brc_date_palliative_care.between(date_lastlog, date_now).sum(),
            'death_breast_cancer_since_last_log': df.brc_date_death.between(date_lastlog, date_now).sum(),
            'n women age 15+': n_ge15_f,
            'n_newly_diagnosed_stage1': n_newly_diagnosed_stage1,
            'n_newly_diagnosed_stage2': n_newly_diagnosed_stage2,
            'n_newly_diagnosed_stage3': n_newly_diagnosed_stage3,
            'n_newly_diagnosed_stage4': n_newly_diagnosed_stage4,
            'n_diagnosed_age_15_29': n_diagnosed_age_15_29,
            'n_diagnosed_age_30_49':  n_diagnosed_age_30_49,
            'n_diagnosed_age_50p': n_diagnosed_age_50p,
            'n_diagnosed': n_diagnosed
        })

        logger.info(key='summary_stats',
                    description='summary statistics for breast cancer',
                    data=out)
