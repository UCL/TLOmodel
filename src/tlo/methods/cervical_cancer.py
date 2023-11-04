"""
Cervical Cancer Disease Module

Limitations to note:
* Footprints of HSI -- pending input from expert on resources required.
"""

from pathlib import Path

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods.causes import Cause
from tlo.methods.demography import InstantaneousDeath
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom
from tlo.methods import Metadata


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CervicalCancer(Module):
    """Cervical Cancer Disease Module"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.linear_models_for_progression_of_hpv_cc_status = dict()
        self.lm_onset_vaginal_bleeding = None
        self.daly_wts = dict()

    INIT_DEPENDENCIES = {
        'Demography', 'SimplifiedBirths', 'HealthSystem', 'Lifestyle', 'SymptomManager'
    }

    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden', 'HealthSeekingBehaviour'}

    ADDITIONAL_DEPENDENCIES = {'Tb', 'Hiv'}

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'CervicalCancer': Cause(gbd_causes='Cervical cancer', label='Cancer (Cervix)'),
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'CervicalCancer': Cause(gbd_causes='Cervical cancer', label='Cancer (Cervix)'),
    }

    PARAMETERS = {
        "init_prop_hpv_cc_stage_age1524": Parameter(
            Types.LIST,
            "initial proportions in cancer categories for women aged 15-24"
        ),
        "init_prop_hpv_cc_stage_age2549": Parameter(
            Types.LIST,
            "initial proportions in cancer categories for women aged 25-49"
        ),
        "init_prop_vaginal_bleeding_by_cc_stage": Parameter(
            Types.LIST, "initial proportions of those with cervical cancer that have the symptom vaginal_bleeding"
        ),
        "init_prop_with_vaginal_bleeding_diagnosed_cervical_cancer": Parameter(
            Types.REAL, "initial proportions of people that have vaginal bleeding that have been diagnosed"
        ),
        "init_prop_prev_treatment_cervical_cancer": Parameter(
            Types.LIST, "initial proportions of people with cervical cancer previously treated"
        ),
        "init_prob_palliative_care": Parameter(
            Types.REAL, "initial probability of being under palliative care if in stage 4"
        ),
        "r_vp_hpv": Parameter(
            Types.REAL,
            "probabilty per month of incident vaccine preventable hpv infection",
        ),
        "r_nvp_hpv": Parameter(
            Types.REAL,
            "probabilty per month of incident non-vaccine preventable hpv infection",
        ),
        "r_cin1_hpv": Parameter(
            Types.REAL,
            "probabilty per month of incident cin1 amongst people with hpv",
        ),
        "r_cin2_cin1": Parameter(
            Types.REAL,
            "probabilty per month of incident cin2 amongst people with cin1",
        ),
        "r_cin3_cin2": Parameter(
            Types.REAL,
            "probabilty per month of incident cin3 amongst people with cin2",
        ),
        "r_stage1_cin3": Parameter(
            Types.REAL,
            "probabilty per month of incident stage1 cervical cancer amongst people with cin3",
        ),
        "r_stage2a_stage1": Parameter(
            Types.REAL,
            "probabilty per month of incident stage2a cervical cancer amongst people with stage1",
        ),
        "r_stage2b_stage2a": Parameter(
            Types.REAL,
            "probabilty per month of incident stage2b cervical cancer amongst people with stage2a",
        ),
        "r_stage3_stage2b": Parameter(
            Types.REAL,
            "probabilty per month of incident stage3 cervical cancer amongst people with stage2b",
        ),
        "r_stage4_stage3": Parameter(
            Types.REAL,
            "probabilty per month of incident stage4 cervical cancer amongst people with stage3",
        ),
        "rr_progress_cc_hiv": Parameter(
            Types.REAL, "rate ratio for progressing through cin and cervical cancer stages if have unsuppressed hiv"
        ),
        "rr_hpv_vaccinated": Parameter(
            Types.REAL,
            "rate ratio for hpv if vaccinated - this is combined effect of probability the hpv is "
            "vaccine-preventable and vaccine efficacy against vaccine-preventable hpv ",
        ),
         "rr_progression_cc_undergone_curative_treatment": Parameter(
            Types.REAL,
            "rate ratio for progression to next cervical cancer stage if had curative treatment at current stage",
        ),
         "r_death_cervical_cancer": Parameter(
            Types.REAL,
            "probabilty per 3 months of death from cervical cancer amongst people with stage 4 cervical cancer",
        ),
        "r_vaginal_bleeding_cc_stage1": Parameter(
            Types.REAL, "rate of vaginal bleeding if have stage 1 cervical cancer"
        ),
        "rr_vaginal_bleeding_cc_stage2a": Parameter(
            Types.REAL, "rate ratio for vaginal bleeding if have stage 2a cervical cancer"
        ),
        "rr_vaginal_bleeding_cc_stage2b": Parameter(
            Types.REAL, "rate ratio for vaginal bleeding if have stage 2b cervical cancer"
        ),
        "rr_vaginal_bleeding_cc_stage3": Parameter(
            Types.REAL, "rate ratio for vaginal bleeding if have stage 3 cervical cancer"
        ),
        "rr_vaginal_bleeding_cc_stage4": Parameter(
            Types.REAL, "rate ratio for vaginal bleeding if have stage 4 cervical cancer"
        ),
        "sensitivity_of_biopsy_for_cervical_cancer": Parameter(
            Types.REAL, "sensitivity of biopsy for diagnosis of cervical cancer"
        ),
        "sensitivity_of_xpert_for_hpv_cin_cc": Parameter(
            Types.REAL, "sensitivity of xpert for presence of hpv, cin or cervical cancer"
        ),
        "sensitivity_of_via_for_cin_cc": Parameter(
            Types.REAL, "sensitivity of via for cin and cervical cancer bu stage"
        )
    }

    """
    note: hpv vaccination is in epi.py
    """

    PROPERTIES = {
        "ce_hpv_cc_status": Property(
            Types.CATEGORICAL,
            "Current hpv / cervical cancer status",
            categories=["none", "hpv", "cin1", "cin2", "cin3", "stage1", "stage2a", "stage2b", "stage3", "stage4"],
        ),
        "ce_hpv_vp": Property(
            Types.BOOL,
            "if ce_hpv_cc_status = hpv, is it vaccine preventable?"
        ),
        "ce_date_diagnosis": Property(
            Types.DATE,
            "the date of diagnosis of cervical cancer (pd.NaT if never diagnosed)"
        ),
        "ce_date_via": Property(
            Types.DATE,
            "the date of last visual inspection with acetic acid (pd.NaT if never diagnosed)"
        ),
        "ce_date_xpert": Property(
            Types.DATE,
            "the date of last hpv test using xpert (pd.NaT if never diagnosed)"
        ),
        "ce_date_cin_removal": Property(
            Types.DATE,
            "the date of last cin removal (pd.NaT if never diagnosed)"
        ),
        "ce_date_treatment": Property(
            Types.DATE,
            "date of first receiving attempted curative treatment (pd.NaT if never started treatment)"
        ),
        "ce_vaginal_bleeding_investigated": Property(
            Types.BOOL,
            "whether vaginal bleeding has been investigated, and cancer missed"
        ),
# currently this property has levels to match ce_hov_cc_status to enable the code as written, even
# though can only be treated when in stage 1-3
        "ce_stage_at_which_treatment_given": Property(
            Types.CATEGORICAL,
            "the cancer stage at which treatment was given (because the treatment only has an effect during the stage"
            "at which it is given).",
            categories=["none", "hpv", "cin1", "cin2", "cin3", "stage1", "stage2a", "stage2b", "stage3", "stage4"],
        ),
        "ce_date_palliative_care": Property(
            Types.DATE,
            "date of first receiving palliative care (pd.NaT is never had palliative care)"
        ),
        "ce_date_death": Property(
            Types.DATE,
            "date of cervical cancer death"
        ),
        "ce_new_stage_this_month": Property(
            Types.BOOL,
            "new_stage_this month"
        )
    }

    def read_parameters(self, data_folder):
        """Setup parameters used by the module, now including disability weights"""
        # todo: add disability weights to resource file

        # Update parameters from the resourcefile
        self.load_parameters_from_dataframe(
            pd.read_excel(Path(self.resourcefilepath) / "ResourceFile_Cervical_Cancer.xlsx",
                          sheet_name="parameter_values")
        )

        # Register Symptom that this module will use
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='vaginal_bleeding',
                    odds_ratio_health_seeking_in_adults=4.00)
        )

    def initialise_population(self, population):
        """Set property values for the initial population."""
        df = population.props  # a shortcut to the data-frame
        p = self.parameters

        # defaults
        df.loc[df.is_alive, "ce_hpv_cc_status"] = "none"
        df.loc[df.is_alive, "ce_date_diagnosis"] = pd.NaT
        df.loc[df.is_alive, "ce_date_treatment"] = pd.NaT
        df.loc[df.is_alive, "ce_stage_at_which_treatment_given"] = "none"
        df.loc[df.is_alive, "ce_date_palliative_care"] = pd.NaT
        df.loc[df.is_alive, "ce_date_death"] = pd.NaT
        df.loc[df.is_alive, "ce_vaginal_bleeding_investigated"] = False
        df.loc[df.is_alive, "ce_new_stage_this_month"] = False

        # -------------------- ce_hpv_cc_status -----------
        # Determine who has cancer at ANY cancer stage:
        # check parameters are sensible: probability of having any cancer stage cannot exceed 1.0
        assert sum(p['init_prop_hpv_cc_stage_age1524']) <= 1.0
        assert sum(p['init_prop_hpv_cc_stage_age2549']) <= 1.0

    # todo: create ce_hpv_cc_status for all at baseline using init_prop_hpv_cc_stage_age1524
    #       and init_prop_hpv_cc_stage_age2549 - currently everyone incorrectly starts as "none"

#       df.ce_hpv_cc_status = 'none'

        # -------------------- SYMPTOMS -----------
        # Create shorthand variable for the initial proportion of discernible cervical cancer lumps in the population
        init_prop_vaginal_bleeding = p['init_prop_vaginal_bleeding_by_cc_stage']
        lm_init_vaginal_bleeding = LinearModel.multiplicative(
            Predictor(
                'ce_hpv_cc_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when("none", 0.0)
            .when("hpv", 0.0)
            .when("cin1", 0.0)
            .when("cin2", 0.0)
            .when("cin3", 0.0)
            .when("stage1", init_prop_vaginal_bleeding[0])
            .when("stage2a", init_prop_vaginal_bleeding[1])
            .when("stage2b", init_prop_vaginal_bleeding[2])
            .when("stage3", init_prop_vaginal_bleeding[3])
            .when("stage4", init_prop_vaginal_bleeding[4])
        )

        has_vaginal_bleeding_at_init = lm_init_vaginal_bleeding.predict(df.loc[df.is_alive], self.rng)
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=has_vaginal_bleeding_at_init.index[has_vaginal_bleeding_at_init].tolist(),
            symptom_string='vaginal_bleeding',
            add_or_remove='+',
            disease_module=self
        )

        # -------------------- ce_date_diagnosis -----------
        # Create shorthand variable for the initial proportion of the population with vaginal bleeding that has
        # been diagnosed
        initial_prop_diagnosed_vaginal_bleeding = \
            p['init_prop_with_vaginal_bleeding_diagnosed_cervical_cancer']
        lm_init_diagnosed = LinearModel.multiplicative(
            Predictor(
                'ce_hpv_cc_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when("none", 0.0)
            .when("hpv", 0.0)
            .when("cin1", 0.0)
            .when("cin2", 0.0)
            .when("cin3", 0.0)
            .when("stage1", initial_prop_diagnosed_vaginal_bleeding)
            .when("stage2a", initial_prop_diagnosed_vaginal_bleeding)
            .when("stage2b", initial_prop_diagnosed_vaginal_bleeding)
            .when("stage3", initial_prop_diagnosed_vaginal_bleeding)
            .when("stage4", initial_prop_diagnosed_vaginal_bleeding)
        )
        ever_diagnosed_cc = lm_init_diagnosed.predict(df.loc[df.is_alive], self.rng)

        # ensure that persons who have not ever had the symptom vaginal bleeding are not diagnosed:
        ever_diagnosed_cc.loc[~has_vaginal_bleeding_at_init] = False

        # For those that have been diagnosed, set data of diagnosis to today's date
        df.loc[ever_diagnosed_cc, "ce_date_diagnosis"] = self.sim.date

        # -------------------- ce_date_treatment -----------

        ce_inital_treament_status = p['init_prop_prev_treatment_cervical_cancer']
        lm_init_treatment_for_those_diagnosed = LinearModel.multiplicative(
            Predictor(
                'ce_hpv_cc_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when("none", 0.0)
            .when("hpv", 0.0)
            .when("stage1", ce_inital_treament_status[0])
            .when("stage2a", ce_inital_treament_status[1])
            .when("stage2b", ce_inital_treament_status[2])
            .when("stage3", ce_inital_treament_status[3])
            .when("stage4", ce_inital_treament_status[4])
        )
        treatment_initiated = lm_init_treatment_for_those_diagnosed.predict(df.loc[df.is_alive], self.rng)

        # prevent treatment having been initiated for anyone who is not yet diagnosed
        treatment_initiated.loc[pd.isnull(df.ce_date_diagnosis)] = False

        # assume that the stage at which treatment is begun is the stage the person is in now;
        df.loc[treatment_initiated, "ce_stage_at_which_treatment_given"] = df.loc[treatment_initiated, "ce_hpv_cc_status"]

        # set date at which treatment began: same as diagnosis (NB. no HSI is established for this)
        df.loc[treatment_initiated, "ce_date_treatment"] = df.loc[treatment_initiated, "ce_date_diagnosis"]

        # -------------------- ce_date_palliative_care -----------
        in_stage4_diagnosed = df.index[df.is_alive & (df.ce_hpv_cc_status == 'stage4') & ~pd.isnull(df.ce_date_diagnosis)]

        select_for_care = self.rng.random_sample(size=len(in_stage4_diagnosed)) < p['init_prob_palliative_care']
        select_for_care = in_stage4_diagnosed[select_for_care]

        # set date of palliative care being initiated: same as diagnosis (NB. future HSI will be scheduled for this)
        df.loc[select_for_care, "ce_date_palliative_care"] = df.loc[select_for_care, "ce_date_diagnosis"]


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
        sim.schedule_event(CervicalCancerLoggingEvent(self), sim.date + DateOffset(months=0))

        # ----- SCHEDULE MAIN POLLING EVENTS -----
        # Schedule main polling event to happen immediately
        sim.schedule_event(CervicalCancerMainPollingEvent(self), sim.date + DateOffset(months=1))

        # ----- LINEAR MODELS -----
        # Define LinearModels for the progression of cancer, in each 1 month period
        # NB. The effect being produced is that treatment only has the effect in the stage at which the
        # treatment was received.

        df = sim.population.props
        p = self.parameters
        lm = self.linear_models_for_progression_of_hpv_cc_status

        rate_hpv = p['r_nvp_hpv'] + p['r_vp_hpv']

        lm['hpv'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            rate_hpv,
            Predictor('va_hpv')
            .when(1, p['rr_hpv_vaccinated'])
            .when(2, p['rr_hpv_vaccinated']),
            Predictor('sex').when('M', 0.0),
            Predictor('ce_hpv_cc_status').when('none', 1.0).otherwise(0.0),
            Predictor('hv_art', conditions_are_mutually_exclusive=True)
            .when('not', p['rr_progress_cc_hiv'])
            .when('on_not_VL_suppressed', p['rr_progress_cc_hiv'])
            .when('on_VL_suppressed', 1.0)
        )

        lm['cin1'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_cin1_hpv'],
            Predictor('ce_hpv_cc_status').when('hpv', 1.0).otherwise(0.0),
            Predictor('hv_art', conditions_are_mutually_exclusive=True)
            .when('not', p['rr_progress_cc_hiv'])
            .when('on_not_VL_suppressed', p['rr_progress_cc_hiv'])
            .when('on_VL_suppressed', 1.0)
        )

        lm['cin2'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_cin2_cin1'],
            Predictor('ce_hpv_cc_status').when('cin1', 1.0).otherwise(0.0),
            Predictor('hv_art', conditions_are_mutually_exclusive=True)
            .when('not', p['rr_progress_cc_hiv'])
            .when('on_not_VL_suppressed', p['rr_progress_cc_hiv'])
            .when('on_VL_suppressed', 1.0)
        )

        lm['cin3'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_cin3_cin2'],
            Predictor('ce_hpv_cc_status').when('cin2', 1.0).otherwise(0.0),
            Predictor('hv_art', conditions_are_mutually_exclusive=True)
            .when('not', p['rr_progress_cc_hiv'])
            .when('on_not_VL_suppressed', p['rr_progress_cc_hiv'])
            .when('on_VL_suppressed', 1.0)
        )

        lm['stage1'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage1_cin3'],
            Predictor('ce_hpv_cc_status').when('cin3', 1.0).otherwise(0.0),
            Predictor('hv_art', conditions_are_mutually_exclusive=True)
            .when('not', p['rr_progress_cc_hiv'])
            .when('on_not_VL_suppressed', p['rr_progress_cc_hiv'])
            .when('on_VL_suppressed', 1.0)
        )

        lm['stage2a'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage2a_stage1'],
            Predictor('ce_hpv_cc_status').when('stage1', 1.0).otherwise(0.0),
            Predictor('had_treatment_during_this_stage',
                      external=True).when(True, p['rr_progression_cc_undergone_curative_treatment']),
            Predictor('hv_art', conditions_are_mutually_exclusive=True)
            .when('not', p['rr_progress_cc_hiv'])
            .when('on_not_VL_suppressed', p['rr_progress_cc_hiv'])
            .when('on_VL_suppressed', 1.0),
            Predictor('ce_new_stage_this_month').when(True, 0.0).otherwise(1.0)
        )

        lm['stage2b'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage2b_stage2a'],
            Predictor('ce_hpv_cc_status').when('stage2a', 1.0).otherwise(0.0),
            Predictor('had_treatment_during_this_stage',
                      external=True).when(True, p['rr_progression_cc_undergone_curative_treatment']),
            Predictor('hv_art', conditions_are_mutually_exclusive=True)
            .when('not', p['rr_progress_cc_hiv'])
            .when('on_not_VL_suppressed', p['rr_progress_cc_hiv'])
            .when('on_VL_suppressed', 1.0),
            Predictor('ce_new_stage_this_month').when(True, 0.0).otherwise(1.0)
        )

        lm['stage3'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage3_stage2b'],
            Predictor('ce_hpv_cc_status').when('stage2b', 1.0).otherwise(0.0),
            Predictor('had_treatment_during_this_stage',
                      external=True).when(True, p['rr_progression_cc_undergone_curative_treatment']),
            Predictor('hv_art', conditions_are_mutually_exclusive=True)
            .when('not', p['rr_progress_cc_hiv'])
            .when('on_not_VL_suppressed', p['rr_progress_cc_hiv'])
            .when('on_VL_suppressed', 1.0),
            Predictor('ce_new_stage_this_month').when(True, 0.0).otherwise(1.0)
        )

        lm['stage4'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage4_stage3'],
            Predictor('ce_hpv_cc_status').when('stage3', 1.0).otherwise(0.0),
            Predictor('had_treatment_during_this_stage',
                      external=True).when(True, p['rr_progression_cc_undergone_curative_treatment']),
            Predictor('hv_art', conditions_are_mutually_exclusive=True)
            .when('not', p['rr_progress_cc_hiv'])
            .when('on_not_VL_suppressed', p['rr_progress_cc_hiv'])
            .when('on_VL_suppressed', 1.0),
            Predictor('ce_new_stage_this_month').when(True, 0.0).otherwise(1.0)
        )

        # Check that the dict labels are correct as these are used to set the value of ce_hpv_cc_status
        assert set(lm).union({'none'}) == set(df.ce_hpv_cc_status.cat.categories)

        # Linear Model for the onset of vaginal bleeding, in each 1 month period
        # Create variables for used to predict the onset of vaginal bleeding at
        # various stages of the disease

        stage1 = p['r_vaginal_bleeding_cc_stage1']
        stage2a = p['rr_vaginal_bleeding_cc_stage2a'] * p['r_vaginal_bleeding_cc_stage1']
        stage2b = p['rr_vaginal_bleeding_cc_stage2b'] * p['r_vaginal_bleeding_cc_stage1']
        stage3 = p['rr_vaginal_bleeding_cc_stage3'] * p['r_vaginal_bleeding_cc_stage1']
        stage4 = p['rr_vaginal_bleeding_cc_stage4'] * p['r_vaginal_bleeding_cc_stage1']

# todo: do we need to restrict to women without pre-existing vaginal bleeding ?

        self.lm_onset_vaginal_bleeding = LinearModel.multiplicative(
            Predictor(
                'ce_hpv_cc_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when('none', 0.0)
            .when('cin1', 0.0)
            .when('cin2', 0.0)
            .when('cin3', 0.0)
            .when('stage1', stage1)
            .when('stage2a', stage2a)
            .when('stage2b', stage2b)
            .when('stage3', stage3)
            .when('stage4', stage4)
        )

        # ----- DX TESTS -----
        # Create the diagnostic test representing the use of a biopsy
        # This properties of conditional on the test being done only to persons with the Symptom, 'vaginal_bleeding!

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            biopsy_for_cervical_cancer_given_vaginal_bleeding=DxTest(
                property='ce_hpv_cc_status',
                sensitivity=self.parameters['sensitivity_of_biopsy_for_cervical_cancer'],
                target_categories=["stage1", "stage2a", "stage2b", "stage3", "stage4"]
            )
        )

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            screening_with_via_for_hpv_and_cervical_cancer=DxTest(
                property='ce_hpv_cc_status',
                sensitivity=self.parameters['sensitivity_of_xpert_for_hpv_cin_cc'],
                target_categories=["cin1", "cin2", "cin3", "stage1", "stage2a", "stage2b", "stage3", "stage4"]
            )
        )

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            screening_with_xpert_for_hpv_and_cervical_cancer=DxTest(
                property='ce_hpv_cc_status',
                sensitivity=self.parameters['sensitivity_of_via_for_cin_cc'],
                target_categories=["hpv", "cin1", "cin2", "cin3", "stage1", "stage2a", "stage2b", "stage3", "stage4"]
            )
        )

        # ----- DISABILITY-WEIGHT -----
        if "HealthBurden" in self.sim.modules:
            # For those with cancer (any stage prior to stage 4) and never treated
            self.daly_wts["stage_1_3"] = self.sim.modules["HealthBurden"].get_daly_weight(
                # todo: review the sequlae numbers
                sequlae_code=550
                # "Diagnosis and primary therapy phase of cervical cancer":
                #  "Cancer, diagnosis and primary therapy ","has pain, nausea, fatigue, weight loss and high anxiety."
            )

            # For those with cancer (any stage prior to stage 4) and has been treated
            self.daly_wts["stage_1_3_treated"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=547
                # "Controlled phase of cervical cancer,Generic uncomplicated disease":
                # "worry and daily medication,has a chronic disease that requires medication every day and causes some
                #   worry but minimal interference with daily activities".
            )

            # For those in stage 4: no palliative care
            self.daly_wts["stage4"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=549
                # "Metastatic phase of cervical cancer:
                # "Cancer, metastatic","has severe pain, extreme fatigue, weight loss and high anxiety."
            )

            # For those in stage 4: with palliative care
            self.daly_wts["stage4_palliative_care"] = self.daly_wts["stage_1_3"]
            # By assumption, we say that that the weight for those in stage 4 with palliative care is the same as
            # that for those with stage 1-3 cancers.

        # ----- HSI FOR PALLIATIVE CARE -----
        on_palliative_care_at_initiation = df.index[df.is_alive & ~pd.isnull(df.ce_date_palliative_care)]
        for person_id in on_palliative_care_at_initiation:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_CervicalCancer_PalliativeCare(module=self, person_id=person_id),
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
        df.at[child_id, "ce_hpv_cc_status"] = "none"
        df.at[child_id, "ce_hpv_vp"] = False
        df.at[child_id, "ce_date_treatment"] = pd.NaT
        df.at[child_id, "ce_stage_at_which_treatment_given"] = "none"
        df.at[child_id, "ce_date_diagnosis"] = pd.NaT
        df.at[child_id, "ce_new_stage_this_month"] = False
        df.at[child_id, "ce_vaginal_bleeding_investigated"] = False
        df.at[child_id, "ce_date_palliative_care"] = pd.NaT
        df.at[child_id, "ce_date_xpert"] = pd.NaT
        df.at[child_id, "ce_date_via"] = pd.NaT
        df.at[child_id, "ce_date_death"] = pd.NaT
        df.at[child_id, "ce_date_cin_removal"] = pd.NaT
        df.at[child_id, "ce_date_treatment"] = pd.NaT

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
                (df.ce_hpv_cc_status == "stage1") |
                (df.ce_hpv_cc_status == "stage2a") |
                (df.ce_hpv_cc_status == "stage2b") |
                (df.ce_hpv_cc_status == "stage3")
            )
        ] = self.daly_wts['stage_1_3']

        # Assign daly_wt to those with cancer stages before stage4 and who have been treated and who are still in the
        # stage in which they were treated.
        disability_series_for_alive_persons.loc[
            (
                ~pd.isnull(df.ce_date_treatment) & (
                    (df.ce_hpv_cc_status == "stage1") |
                    (df.ce_hpv_cc_status == "stage2a") |
                    (df.ce_hpv_cc_status == "stage2b") |
                    (df.ce_hpv_cc_status == "stage3")
                ) & (df.ce_hpv_cc_status == df.ce_stage_at_which_treatment_given)
            )
        ] = self.daly_wts['stage_1_3_treated']

        # Assign daly_wt to those in stage4 cancer (who have not had palliative care)
        disability_series_for_alive_persons.loc[
            (df.ce_hpv_cc_status == "stage4") &
            (pd.isnull(df.ce_date_palliative_care))
            ] = self.daly_wts['stage4']

        # Assign daly_wt to those in stage4 cancer, who have had palliative care
        disability_series_for_alive_persons.loc[
            (df.ce_hpv_cc_status == "stage4") &
            (~pd.isnull(df.ce_date_palliative_care))
            ] = self.daly_wts['stage4_palliative_care']

        return disability_series_for_alive_persons


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------

class CervicalCancerMainPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all cervical cancer properties for population:
    * Acquisition and progression of hpv, cin, cervical cancer
    * Symptom Development according to stage of cervical Cancer
    * Deaths from cervical cancer for those in stage4
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        # scheduled to run every 1 month: do not change as this is hard-wired into the values of all the parameters.

    def apply(self, population):
        df = population.props  # shortcut to dataframe
        m = self.module
        rng = m.rng

        # -------------------- ACQUISITION AND PROGRESSION OF CANCER (ce_hpv_cc_status) -----------------------------------

        df.ce_new_stage_this_month = False

        # determine if the person had a treatment during this stage of cancer (nb. treatment only has an effect on
        #  reducing progression risk during the stage at which is received.
        had_treatment_during_this_stage = \
            df.is_alive & ~pd.isnull(df.ce_date_treatment) & \
            (df.ce_hpv_cc_status == df.ce_stage_at_which_treatment_given)

# todo: still need to derive the lm to make this work

        for stage, lm in self.module.linear_models_for_progression_of_hpv_cc_status.items():
            gets_new_stage = lm.predict(df.loc[df.is_alive], rng,
                                        had_treatment_during_this_stage=had_treatment_during_this_stage)
            idx_gets_new_stage = gets_new_stage[gets_new_stage].index
            df.loc[idx_gets_new_stage, 'ce_hpv_cc_status'] = stage
            df.loc[idx_gets_new_stage, 'ce_new_stage_this_month'] = True

        # -------------------- UPDATING OF SYMPTOM OF vaginal bleeding OVER TIME --------------------------------
        # Each time this event is called (every month) individuals with cervical cancer may develop the symptom of
        # vaginal bleeding.  Once the symptom is developed it never resolves naturally. It may trigger
        # health-care-seeking behaviour.
        onset_vaginal_bleeding = self.module.lm_onset_vaginal_bleeding.predict(df.loc[df.is_alive], rng)
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=onset_vaginal_bleeding[onset_vaginal_bleeding].index.tolist(),
            symptom_string='vaginal_bleeding',
            add_or_remove='+',
            disease_module=self.module
        )

        # -------------------- DEATH FROM cervical CANCER ---------------------------------------
        # There is a risk of death for those in stage4 only. Death is assumed to go instantly.
        stage4_idx = df.index[df.is_alive & (df.ce_hpv_cc_status == "stage4")]
        selected_to_die = stage4_idx[
            rng.random_sample(size=len(stage4_idx)) < self.module.parameters['r_death_cervical_cancer']]

        for person_id in selected_to_die:
            self.sim.schedule_event(
                InstantaneousDeath(self.module, person_id, "CervicalCancer"), self.sim.date
            )
            df.loc[selected_to_die, 'ce_date_death'] = self.sim.date

# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
# ---------------------------------------------------------------------------------------------------------

#  todo: hsi for routine screening (ie the hsi is health system-initiated) using hpv xpert and/or via,
#  todo: with cin removal - need to agree how to do this

class HSI_CervicalCancer_Investigation_Following_vaginal_bleeding(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_GenericFirstApptAtFacilityLevel1 following presentation for care with the symptom
    vaginal bleeding.
    This event begins the investigation that may result in diagnosis of cervical Cancer and the scheduling of
    treatment or palliative care.
    It is for people with the symptom vaginal_bleeding.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        print(person_id, self.sim.date, 'vaginal_bleeding_hsi_called -1')

        self.TREATMENT_ID = "CervicalCancer_Investigation"

        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        # Ignore this event if the person is no longer alive:
        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        print(person_id, self.sim.date, 'vaginal_bleeding_hsi_called -2')

        # Check that this event has been called for someone with the symptom vaginal_bleeding
        assert 'vaginal_bleeding' in self.sim.modules['SymptomManager'].has_what(person_id)

        # If the person is already diagnosed, then take no action:
        if not pd.isnull(df.at[person_id, "ce_date_diagnosis"]):
            return hs.get_blank_appt_footprint()

        df.loc[person_id, 'ce_vaginal_bleeding_investigated'] = True

        # Use a biopsy to diagnose whether the person has cervical cancer
        # todo: request consumables needed for this

        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='biopsy_for_cervical_cancer_given_vaginal_bleeding',
            hsi_event=self
        )

        if dx_result:
            # record date of diagnosis:
            df.at[person_id, 'ce_date_diagnosis'] = self.sim.date

            # Check if is in stage4:
            in_stage4 = df.at[person_id, 'ce_hpv_cc_status'] == 'stage4'
            # If the diagnosis does detect cancer, it is assumed that the classification as stage4 is made accurately.

            if not in_stage4:
                # start treatment:
                hs.schedule_hsi_event(
                    hsi_event=HSI_CervicalCancer_StartTreatment(
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
                    hsi_event=HSI_CervicalCancer_PalliativeCare(
                        module=self.module,
                        person_id=person_id
                    ),
                    priority=0,
                    topen=self.sim.date,
                    tclose=None
                )

class HSI_CervicalCancer_StartTreatment(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_CervicalCancer_Investigation_Following_vaginal_bleeding following a diagnosis of
    cervical Cancer. It initiates the treatment of cervical Cancer.
    It is only for persons with a cancer that is not in stage4 and who have been diagnosed.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "CervicalCancer_Treatment"
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
        if df.at[person_id, "ce_hpv_cc_status"] == 'stage4':
            logger.warning(key="warning", data="Cancer is in stage 4 - aborting HSI_CervicalCancer_StartTreatment,"
                                               "scheduling HSI_CervicalCancer_PalliativeCare")

            hs.schedule_hsi_event(
                hsi_event=HSI_CervicalCancer_PalliativeCare(
                     module=self.module,
                     person_id=person_id,
                ),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )
            return self.make_appt_footprint({})

        # Check that the person has been diagnosed and is not on treatment
        assert not df.at[person_id, "ce_hpv_cc_status"] == 'none'
        assert not df.at[person_id, "ce_hpv_cc_status"] == 'hpv'
        assert not df.at[person_id, "ce_hpv_cc_status"] == 'cin1'
        assert not df.at[person_id, "ce_hpv_cc_status"] == 'cin2'
        assert not df.at[person_id, "ce_hpv_cc_status"] == 'cin3'
        assert not df.at[person_id, "ce_hpv_cc_status"] == 'stage4'
        assert not pd.isnull(df.at[person_id, "ce_date_diagnosis"])
        assert pd.isnull(df.at[person_id, "ce_date_treatment"])

        # Record date and stage of starting treatment
        df.at[person_id, "ce_date_treatment"] = self.sim.date
        df.at[person_id, "ce_stage_at_which_treatment_given"] = df.at[person_id, "ce_hpv_cc_status"]

        # todo: maybe have a probability of going to status=none rather than a relative rate of progression

        # Schedule a post-treatment check for 3 months:
        hs.schedule_hsi_event(
            hsi_event=HSI_CervicalCancer_PostTreatmentCheck(
                module=self.module,
                person_id=person_id,
            ),
            topen=self.sim.date + DateOffset(months=3),
            tclose=None,
            priority=0
        )

class HSI_CervicalCancer_PostTreatmentCheck(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_CervicalCancer_StartTreatment and itself.
    It is only for those who have undergone treatment for cervical Cancer.
    If the person has developed cancer to stage4, the patient is initiated on palliative care; otherwise a further
    appointment is scheduled for one year.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "CervicalCancer_Treatment"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # Check that the person has cancer and is on treatment
        assert not df.at[person_id, "ce_hpv_cc_status"] == 'none'
        assert not df.at[person_id, "ce_hpv_cc_status"] == 'hpv'
        assert not df.at[person_id, "ce_hpv_cc_status"] == 'cin1'
        assert not df.at[person_id, "ce_hpv_cc_status"] == 'cin2'
        assert not df.at[person_id, "ce_hpv_cc_status"] == 'cin3'
        assert not pd.isnull(df.at[person_id, "ce_date_diagnosis"])
        assert not pd.isnull(df.at[person_id, "ce_date_treatment"])

        if df.at[person_id, 'ce_hpv_cc_status'] == 'stage4':
            # If has progressed to stage4, then start Palliative Care immediately:
            hs.schedule_hsi_event(
                hsi_event=HSI_CervicalCancer_PalliativeCare(
                    module=self.module,
                    person_id=person_id
                ),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )

        else:
            # Schedule another HSI_CervicalCancer_PostTreatmentCheck event in 3 monthw
            hs.schedule_hsi_event(
                hsi_event=HSI_CervicalCancer_PostTreatmentCheck(
                    module=self.module,
                    person_id=person_id
                ),
                topen=self.sim.date + DateOffset(months=3),
                tclose=None,
                priority=0
            )


class HSI_CervicalCancer_PalliativeCare(HSI_Event, IndividualScopeEventMixin):
    """
    This is the event for palliative care. It does not affect the patients progress but does affect the disability
     weight and takes resources from the healthsystem.
    This event is scheduled by either:
    * HSI_CervicalCancer_Investigation_Following_vaginal_bleeding following a diagnosis of cervical Cancer at stage4.
    * HSI_CervicalCancer_PostTreatmentCheck following progression to stage4 during treatment.
    * Itself for the continuance of care.
    It is only for persons with a cancer in stage4.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "CervicalCancer_PalliativeCare"
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
        assert df.at[person_id, "ce_hpv_cc_status"] == 'stage4'

        # Record the start of palliative care if this is first appointment
        if pd.isnull(df.at[person_id, "ce_date_palliative_care"]):
            df.at[person_id, "ce_date_palliative_care"] = self.sim.date

        # Schedule another instance of the event for one month
        hs.schedule_hsi_event(
            hsi_event=HSI_CervicalCancer_PalliativeCare(
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

class CervicalCancerLoggingEvent(RegularEvent, PopulationScopeEventMixin):
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
            f'total_{k}': v for k, v in df.loc[df.is_alive].ce_hpv_cc_status.value_counts().items()})

        # Current counts, undiagnosed
        out.update({f'undiagnosed_{k}': v for k, v in df.loc[df.is_alive].loc[
            pd.isnull(df.ce_date_diagnosis), 'ce_hpv_cc_status'].value_counts().items()})

        # Current counts, diagnosed
        out.update({f'diagnosed_{k}': v for k, v in df.loc[df.is_alive].loc[
            ~pd.isnull(df.ce_date_diagnosis), 'ce_hpv_cc_status'].value_counts().items()})

        # Current counts, on treatment (excl. palliative care)
        out.update({f'treatment_{k}': v for k, v in df.loc[df.is_alive].loc[(~pd.isnull(
            df.ce_date_treatment) & pd.isnull(
            df.ce_date_palliative_care)), 'ce_hpv_cc_status'].value_counts().items()})

        # Current counts, on palliative care
        out.update({f'palliative_{k}': v for k, v in df.loc[df.is_alive].loc[
            ~pd.isnull(df.ce_date_palliative_care), 'ce_hpv_cc_status'].value_counts().items()})

        # Counts of those that have been diagnosed, started treatment or started palliative care since last logging
        # event:
        date_now = self.sim.date
        date_lastlog = self.sim.date - pd.DateOffset(days=29)

        n_ge15_f = (df.is_alive & (df.age_years >= 15) & (df.sex == 'F')).sum()

        n_newly_diagnosed_stage1 = \
            (df.ce_date_diagnosis.between(date_lastlog, date_now - DateOffset(days=1)) & (df.ce_hpv_cc_status == 'stage1')).sum()
        n_newly_diagnosed_stage2a = \
            (df.ce_date_diagnosis.between(date_lastlog, date_now - DateOffset(days=1)) & (df.ce_hpv_cc_status == 'stage2a')).sum()
        n_newly_diagnosed_stage2b = \
            (df.ce_date_diagnosis.between(date_lastlog, date_now - DateOffset(days=1)) & (df.ce_hpv_cc_status == 'stage2b')).sum()
        n_newly_diagnosed_stage3 = \
            (df.ce_date_diagnosis.between(date_lastlog, date_now - DateOffset(days=1)) & (df.ce_hpv_cc_status == 'stage3')).sum()
        n_newly_diagnosed_stage4 = \
            (df.ce_date_diagnosis.between(date_lastlog, date_now - DateOffset(days=1)) & (df.ce_hpv_cc_status == 'stage4')).sum()

# todo: add outputs for cin,  xpert testing and via and removal of cin

        n_diagnosed_age_15_29 = (df.is_alive & (df.age_years >= 15) & (df.age_years < 30)
                                 & ~pd.isnull(df.ce_date_diagnosis)).sum()
        n_diagnosed_age_30_49 = (df.is_alive & (df.age_years >= 30) & (df.age_years < 50)
                                 & ~pd.isnull(df.ce_date_diagnosis)).sum()
        n_diagnosed_age_50p = (df.is_alive & (df.age_years >= 50) & ~pd.isnull(df.ce_date_diagnosis)).sum()

        n_diagnosed = (df.is_alive & ~pd.isnull(df.ce_date_diagnosis)).sum()

        out.update({
            'diagnosed_since_last_log': df.ce_date_diagnosis.between(date_lastlog, date_now).sum(),
            'treated_since_last_log': df.ce_date_treatment.between(date_lastlog, date_now).sum(),
            'palliative_since_last_log': df.ce_date_palliative_care.between(date_lastlog, date_now).sum(),
            'death_cervical_cancer_since_last_log': df.ce_date_death.between(date_lastlog, date_now).sum(),
            'n women age 15+': n_ge15_f,
            'n_newly_diagnosed_stage1': n_newly_diagnosed_stage1,
            'n_newly_diagnosed_stage2a': n_newly_diagnosed_stage2a,
            'n_newly_diagnosed_stage2b': n_newly_diagnosed_stage2b,
            'n_newly_diagnosed_stage3': n_newly_diagnosed_stage3,
            'n_newly_diagnosed_stage4': n_newly_diagnosed_stage4,
            'n_diagnosed_age_15_29': n_diagnosed_age_15_29,
            'n_diagnosed_age_30_49':  n_diagnosed_age_30_49,
            'n_diagnosed_age_50p': n_diagnosed_age_50p,
            'n_diagnosed': n_diagnosed
        })

#       df = df.rename(columns={'ce_stage_at_which_treatment_given': 'treatment_stage'})

        print(self.sim.date)
        selected_columns = ['ce_hpv_cc_status', 'ce_hpv_vp']
        selected_rows = df[(df['sex'] == 'F') & (df['age_years'] > 15)]
        print(selected_rows[selected_columns])

#       df = df.rename(columns={'treatment_stage': 'ce_stage_at_which_treatment_given'})

        logger.info(key='summary_stats',
                    description='summary statistics for cervical cancer',
                    data=out)
