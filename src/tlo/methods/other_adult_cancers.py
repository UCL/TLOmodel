"""
Other_adult Cancer Disease Module

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


class OtherAdultCancer(Module):
    """Other Adult Cancers Disease Module"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.linear_models_for_progession_of_oac_status = dict()
        self.lm_onset_early_other_adult_ca_symptom = None
        self.daly_wts = dict()

    INIT_DEPENDENCIES = {'Demography', 'HealthSystem', 'SymptomManager'}

    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden'}

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    gbd_causes_of_cancer_represented_in_this_module = [
        'Other malignant neoplasms',
        'Nasopharynx cancer',
        'Other pharynx cancer',
        'Gallbladder and biliary tract cancer',
        'Pancreatic cancer',
        'Malignant skin melanoma',
        'Non-melanoma skin cancer',
        'Ovarian cancer',
        'Testicular cancer',
        'Kidney cancer',
        # 'Brain and central nervous system cancer',
        'Thyroid cancer',
        'Mesothelioma',
        'Hodgkin lymphoma',
        'Non-Hodgkin lymphoma',
        'Multiple myeloma',
        'Leukemia',
        'Other neoplasms',
        'Cervical cancer',
        'Uterine cancer',
        'Colon and rectum cancer',
        'Lip and oral cavity cancer',
        'Stomach cancer',
        'Liver cancer']

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'OtherAdultCancer': Cause(gbd_causes=gbd_causes_of_cancer_represented_in_this_module, label='Cancer')
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'OtherAdultCancer': Cause(gbd_causes=gbd_causes_of_cancer_represented_in_this_module, label='Cancer')
    }

    PARAMETERS = {
        "init_prop_early_other_adult_ca_symptom_other_adult_cancer_by_stage": Parameter(
            Types.LIST, "initial proportions of those with other adult cancer categories that have other "
                        "adult_ca_symptom"
        ),
        "in_prop_other_adult_cancer_stage": Parameter(
            Types.LIST,
            "initial proportions in other adult cancer stages for person aged 15-29"
        ),
        "init_prop_with_early_other_adult_ca_symptom_diagnosed_by_stage": Parameter(
            Types.LIST, "initial proportions of people that have symptom of other_adult_ca_symptom that "
                        "have been diagnosed"
        ),
        "init_prop_treatment_status_other_adult_cancer": Parameter(
            Types.LIST, "initial proportions of people with other_adult cancer previously given treatment"
        ),
        "init_prob_palliative_care": Parameter(
            Types.REAL, "initial probability of being under palliative care if in metastatic"
        ),
        "r_site_confined_none": Parameter(
            Types.REAL,
            "probabilty per 3 months of incident site_confined other_adult cancer amongst people age 15-29 with no "
            "other adult cancer",
        ),
        "rr_site_confined_age3049": Parameter(
            Types.REAL, "rate ratio for site-confined other_adult cancer for age 30-49"
        ),
        "rr_site_confined_age5069": Parameter(
            Types.REAL, "rate ratio for site-confined other_adult cancer for age 50-69"
        ),
        "rr_site_confined_agege70": Parameter(
            Types.REAL, "rate ratio for site-confined other_adult cancer for age ge 70"
        ),
        "r_local_ln_site_confined_other_adult_ca": Parameter(
            Types.REAL,
            "probabilty per 3 months of local ln other_adult cancer amongst people with site confined",
        ),
        "rr_local_ln_other_adult_ca_undergone_curative_treatment": Parameter(
            Types.REAL,
            "rate ratio for local_ln for people with site_confined "
            "if had curative treatment at site_confined stage",
        ),
        "r_metastatic_local_ln": Parameter(
            Types.REAL, "probabilty per 3 months of metastatic other_adult cancer amongst people with local_ln"
        ),
        "rr_metastatic_undergone_curative_treatment": Parameter(
            Types.REAL,
            "rate ratio for metastatic Other_adult cancer for people with local_ln"
            "Other_adult cancer if had curative treatment at local_ln",
        ),
        "r_death_other_adult_cancer": Parameter(
            Types.REAL,
            "probabilty per 3 months of death from other adult cancer mongst people with metastatic other_adult cancer",
        ),
        "r_early_other_adult_ca_symptom_site_confined_other_adult_ca": Parameter(
            Types.REAL,
            "probability per 3 months of other_adult_ca_symptom in a person with site confined other_adult cancer"
        ),
        "rr_early_other_adult_ca_symptom_local_ln_other_adult_ca": Parameter(
            Types.REAL, "rate ratio for other_adult_ca_symptom if have high grade other_adult cancer"
        ),
        "rr_early_other_adult_ca_symptom_metastatic_other_adult_ca": Parameter(
            Types.REAL, "rate ratio for other_adult_ca_symptom if have metastatic other_adult cancer"
        ),
        "rp_other_adult_cancer_age3049": Parameter(
            Types.REAL, "relative prevalence at baseline of bladder cancer/cancer age 30-49"
        ),
        "rp_other_adult_cancer_age5069": Parameter(
            Types.REAL, "relative prevalence at baseline of bladder cancer/cancer age 50-69"
        ),
        "rp_other_adult_cancer_agege70": Parameter(
            Types.REAL, "relative prevalence at baseline of bladder cancer/cancer age 70+"
        ),
        "sensitivity_of_diagnostic_device_for_other_adult_cancer_with_other_adult_ca_site_confined": Parameter(
            Types.REAL, "sensitivity of diagnostic_device_for diagnosis of other_adult cancer for those with "
                        "other_adult_ca_site-confined"
        ),
        "sensitivity_of_diagnostic_device_for_other_adult_cancer_with_other_adult_ca_local_ln": Parameter(
            Types.REAL, "sensitivity of diagnostic_device_for diagnosis of other_adult cancer for those with "
                        "other_adult_local_ln"
        ),
        "sensitivity_of_diagnostic_device_for_other_adult_cancer_with_other_adult_ca_metastatic": Parameter(
            Types.REAL, "sensitivity of diagnostic_device_for diagnosis of other_adult cancer for those with "
                        "other_adult_ca metastatic"
        )
    }

    PROPERTIES = {
        "oac_status": Property(
            Types.CATEGORICAL,
            "Current status of the health condition, other_adult cancer",
            categories=["none", "site_confined", "local_ln", "metastatic"],
        ),

        "oac_date_diagnosis": Property(
            Types.DATE,
            "the date of diagnosis of the other_adult_cancer (pd.NaT if never diagnosed)"
        ),

        "oac_date_treatment": Property(
            Types.DATE,
            "date of first receiving attempted curative treatment (pd.NaT if never started treatment)"
        ),

        "oac_stage_at_which_treatment_given": Property(
            Types.CATEGORICAL,
            "the cancer stage at which treatment is given (because the treatment only has an effect during the stage"
            "at which it is given.",
            categories=["none", "site_confined", "local_ln", "metastatic"],
        ),
        "oac_date_palliative_care": Property(
            Types.DATE,
            "date of first receiving palliative care (pd.NaT is never had palliative care)"
        ),
        "oac_date_death": Property(
            Types.DATE,
            "date of oac death"
        )
    }

    def read_parameters(self, data_folder):
        """Setup parameters used by the module, now including disability weights"""

        # Update parameters from the resourcefile
        self.load_parameters_from_dataframe(
            pd.read_excel(Path(self.resourcefilepath) / "ResourceFile_Other_Adult_Cancers.xlsx",
                          sheet_name="parameter_values")
        )

        # Register Symptom that this module will use
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='early_other_adult_ca_symptom',
                    odds_ratio_health_seeking_in_adults=4.00,
                    no_healthcareseeking_in_children=True)
        )

    def initialise_population(self, population):
        """Set property values for the initial population."""
        df = population.props  # a shortcut to the data-frame
        p = self.parameters

        # defaults
        df.loc[df.is_alive, "oac_status"] = "none"
        df.loc[df.is_alive, "oac_date_diagnosis"] = pd.NaT
        df.loc[df.is_alive, "oac_date_treatment"] = pd.NaT
        df.loc[df.is_alive, "oac_stage_at_which_treatment_given"] = "none"
        df.loc[df.is_alive, "oac_date_palliative_care"] = pd.NaT
        df.loc[df.is_alive, "oac_date_death"] = pd.NaT

        # -------------------- oac_status -----------
        # Determine who has cancer at ANY cancer stage:
        # check parameters are sensible: probability of having any cancer stage cannot exceed 1.0
        assert sum(p['in_prop_other_adult_cancer_stage']) <= 1.0

        lm_init_oac_status_any_stage = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            sum(p['in_prop_other_adult_cancer_stage']),
            Predictor('age_years', conditions_are_mutually_exclusive=True)
            .when('.between(30,49)', p['rp_other_adult_cancer_age3049'])
            .when('.between(50,69)', p['rp_other_adult_cancer_age5069'])
            .when('.between(70,120)', p['rp_other_adult_cancer_agege70'])
            .when('.between(0,14)', 0.0)
        )

        oac_status_ = lm_init_oac_status_any_stage.predict(df.loc[df.is_alive], self.rng)

        # Determine the stage of the cancer for those who do have a cancer:
        if sum(oac_status_):
            sum_probs = sum(p['in_prop_other_adult_cancer_stage'])
            if sum_probs > 0:
                prob_by_stage_of_cancer_if_cancer = [i/sum_probs for i in p['in_prop_other_adult_cancer_stage']]
                assert (sum(prob_by_stage_of_cancer_if_cancer) - 1.0) < 1e-10
                df.loc[oac_status_, "oac_status"] = self.rng.choice(
                    [val for val in df.oac_status.cat.categories if val != 'none'],
                    size=sum(oac_status_),
                    p=prob_by_stage_of_cancer_if_cancer
                )

        # -------------------- SYMPTOMS -----------
        # ----- Impose the symptom of random sample of those in each cancer stage to have the symptom of
        # other_adult_ca_symptom:
        lm_init_early_other_adult_ca_symptom = LinearModel.multiplicative(
            Predictor(
                'oac_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when("none", 0.0)
            .when(
                "site_confined",
                p['init_prop_early_other_adult_ca_symptom_other_adult_cancer_by_stage'][0]
            )
            .when(
                "local_ln",
                p['init_prop_early_other_adult_ca_symptom_other_adult_cancer_by_stage'][1]
            )
            .when(
                "metastatic",
                p['init_prop_early_other_adult_ca_symptom_other_adult_cancer_by_stage'][2]
            )
        )
        has_early_other_adult_ca_symptom_at_init = lm_init_early_other_adult_ca_symptom.predict(
            df.loc[df.is_alive], self.rng
        )
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=has_early_other_adult_ca_symptom_at_init.index[has_early_other_adult_ca_symptom_at_init].tolist(),
            symptom_string='early_other_adult_ca_symptom',
            add_or_remove='+',
            disease_module=self
        )

        # -------------------- oac_date_diagnosis -----------
        lm_init_diagnosed = LinearModel.multiplicative(
            Predictor(
                'oac_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when("none", 0.0)
            .when(
                "site_confined",
                p['init_prop_with_early_other_adult_ca_symptom_diagnosed_by_stage'][0]
            )
            .when(
                "local_ln",
                p['init_prop_with_early_other_adult_ca_symptom_diagnosed_by_stage'][1]
            )
            .when(
                "metastatic",
                p['init_prop_with_early_other_adult_ca_symptom_diagnosed_by_stage'][2]
            )
        )
        ever_diagnosed = lm_init_diagnosed.predict(df.loc[df.is_alive], self.rng)

        # ensure that persons who have not ever had the symptom other_adult_ca_symptom are diagnosed:
        ever_diagnosed.loc[~has_early_other_adult_ca_symptom_at_init] = False

        # For those that have been diagnosed, set data of diagnosis to today's date
        df.loc[ever_diagnosed, "oac_date_diagnosis"] = self.sim.date

        # -------------------- oac_date_treatment -----------
        lm_init_treatment_for_those_diagnosed = LinearModel.multiplicative(
            Predictor(
                'oac_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when("none", 0.0)
            .when("site_confined", p['init_prop_treatment_status_other_adult_cancer'][0])
            .when("local_ln", p['init_prop_treatment_status_other_adult_cancer'][1])
            .when("metastatic", p['init_prop_treatment_status_other_adult_cancer'][2])
        )
        treatment_initiated = lm_init_treatment_for_those_diagnosed.predict(df.loc[df.is_alive], self.rng)

        # prevent treatment having been initiated for anyone who is not yet diagnosed
        treatment_initiated.loc[pd.isnull(df.oac_date_diagnosis)] = False

        # assume that the stage at which treatment is begun is site_confined;
        df.loc[treatment_initiated, "oac_stage_at_which_treatment_given"] = "site_confined"

        # set date at which treatment began: same as diagnosis (NB. no HSI is established for this)
        df.loc[treatment_initiated, "oac_date_treatment"] = df.loc[treatment_initiated, "oac_date_diagnosis"]

        # -------------------- oac_date_palliative_care -----------
        in_metastatic_diagnosed = df.index[
            df.is_alive & (df.oac_status == 'metastatic') & ~pd.isnull(df.oac_date_diagnosis)
            ]

        select_for_care = self.rng.random_sample(size=len(in_metastatic_diagnosed)) < p['init_prob_palliative_care']
        select_for_care = in_metastatic_diagnosed[select_for_care]

        # set date of palliative care being initiated: same as diagnosis (NB. future HSI will be scheduled for this)
        df.loc[select_for_care, "oac_date_palliative_care"] = df.loc[select_for_care, "oac_date_diagnosis"]

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
        sim.schedule_event(OtherAdultCancerLoggingEvent(self), sim.date + DateOffset(months=0))

        # ----- SCHEDULE MAIN POLLING EVENTS -----
        # Schedule main polling event to happen immediately
        sim.schedule_event(OtherAdultCancerMainPollingEvent(self), sim.date + DateOffset(months=1))

        # ----- LINEAR MODELS -----
        # Define LinearModels for the progression of cancer, in each 3 month period
        # NB. The effect being produced is that treatment only has the effect for during the stage at which the
        # treatment was received.

        df = sim.population.props
        p = self.parameters
        lm = self.linear_models_for_progession_of_oac_status

        lm['site_confined'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_site_confined_none'],
            Predictor('age_years', conditions_are_mutually_exclusive=True)
            .when('.between(30,49)', p['rr_site_confined_age3049'])
            .when('.between(50,69)', p['rr_site_confined_age5069'])
            .when('.between(0,14)', 0.0)
            .when('.between(70,120)', p['rr_site_confined_agege70']),
            Predictor('oac_status').when('none', 1.0).otherwise(0.0)
        )

        lm['local_ln'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_local_ln_site_confined_other_adult_ca'],
            Predictor('had_treatment_during_this_stage',
                      external=True).when(True, p['rr_local_ln_other_adult_ca_undergone_curative_treatment']),
            Predictor('oac_status').when('site_confined', 1.0)
                                   .otherwise(0.0)
        )

        lm['metastatic'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_metastatic_local_ln'],
            Predictor('had_treatment_during_this_stage',
                      external=True).when(True, p['rr_metastatic_undergone_curative_treatment']),
            Predictor('oac_status').when('local_ln', 1.0)
                                   .otherwise(0.0)
        )

        # Check that the dict labels are correct as these are used to set the value of oac_status
        assert set(lm).union({'none'}) == set(df.oac_status.cat.categories)

        # Linear Model for the onset of other_adult_ca_symptom, in each 3 month period
        self.lm_onset_early_other_adult_ca_symptom = LinearModel.multiplicative(
            Predictor(
                'oac_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when(
                'site_confined',
                p['r_early_other_adult_ca_symptom_site_confined_other_adult_ca']
            )
            .when(
                'local_ln',
                p['rr_early_other_adult_ca_symptom_local_ln_other_adult_ca'] *
                p['r_early_other_adult_ca_symptom_site_confined_other_adult_ca']
            )
            .when(
                'metastatic',
                p['rr_early_other_adult_ca_symptom_metastatic_other_adult_ca'] *
                p['r_early_other_adult_ca_symptom_site_confined_other_adult_ca']
            )
            .when('none', 0.0)
        )

        # ----- DX TESTS -----
        # Create the diagnostic test representing the use of an diagnostic_device to oac_status
        # This properties of conditional on the test being done only to persons with the Symptom,
        # 'early_other_adult_ca_symptom'.
        # todo: note dependent on underlying status not symptoms + add for other stages
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
             diagnostic_device_for_other_adult_cancer_given_other_adult_ca_symptom=DxTest(
                property='oac_status',
                sensitivity=self.parameters['sensitivity_of_diagnostic_device_for_other_adult_cancer_with_other_'
                                            'adult_ca_site_confined'],
                target_categories=["site_confined", "local_ln", "metastatic"]
             )
        )

        # ----- DISABILITY-WEIGHT -----
        if "HealthBurden" in self.sim.modules:
            # For those with cancer (any stage prior to metastatic) and never treated
            self.daly_wts["site_confined_local_ln"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=550
                #  todo: maybe this is too high for early cancer
                # "Diagnosis and primary therapy phase of esophageal cancer":
                #  "Cancer, diagnosis and primary therapy ","has pain, nausea, fatigue, weight loss and high anxiety."
            )

            # For those with cancer (any stage prior to metastatic) and has been treated
            self.daly_wts["site_confined_local_ln_treated"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=547
                # "Controlled phase of cancer,Generic uncomplicated disease":
                # "worry and daily medication,has a chronic disease that requires medication every day and causes some
                #   worry but minimal interference with daily activities".
            )

            # For those in metastatic: no palliative care
            self.daly_wts["metastatic"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=549
                # "Metastatic phase of cancer:
                # "Cancer, metastatic","has severe pain, extreme fatigue, weight loss and high anxiety."
            )

            # For those in metastatic: with palliative care
            self.daly_wts["metastatic_palliative_care"] = self.daly_wts["site_confined_local_ln_treated"]
            # By assumption, we say that that the weight for those in metastatic with palliative care is the same as
            # that for those with site confined local ln cancers

        # ----- HSI FOR PALLIATIVE CARE -----
        on_palliative_care_at_initiation = df.index[df.is_alive & ~pd.isnull(df.oac_date_palliative_care)]
        for person_id in on_palliative_care_at_initiation:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_OtherAdultCancer_PalliativeCare(module=self, person_id=person_id),
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
        df.at[child_id, "oac_status"] = "none"
        df.at[child_id, "oac_date_diagnosis"] = pd.NaT
        df.at[child_id, "oac_date_treatment"] = pd.NaT
        df.at[child_id, "oac_stage_at_which_treatment_given"] = "none"
        df.at[child_id, "oac_date_palliative_care"] = pd.NaT
        df.at[child_id, "oac_date_death"] = pd.NaT

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
                (df.oac_status == "site_confined") |
                (df.oac_status == "local_ln")
            )
        ] = self.daly_wts['site_confined_local_ln']

        # Assign daly_wt to those with cancer stages before metastatic and who have been treated and who are still in
        # the stage in which they were treated.
        disability_series_for_alive_persons.loc[
            (
                ~pd.isnull(df.oac_date_treatment) & (
                    (df.oac_status == "site_confined") |
                    (df.oac_status == "local_ln")
                ) & (df.oac_status == df.oac_stage_at_which_treatment_given)
            )
        ] = self.daly_wts['site_confined_local_ln_treated']

        # Assign daly_wt to those in metastatic cancer (who have not had palliative care)
        disability_series_for_alive_persons.loc[
            (df.oac_status == "metastatic") &
            (pd.isnull(df.oac_date_palliative_care))
            ] = self.daly_wts['metastatic']

        # Assign daly_wt to those in metastatic cancer, who have had palliative care
        disability_series_for_alive_persons.loc[
            (df.oac_status == "metastatic") &
            (~pd.isnull(df.oac_date_palliative_care))
            ] = self.daly_wts['metastatic_palliative_care']

        return disability_series_for_alive_persons


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------

class OtherAdultCancerMainPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all Other_adult cancer properties for population:
    * Acquisition and progression of Other_adult Cancer
    * Symptom Development according to stage of Other_adult Cancer
    * Deaths from Other_adult Cancer for those in metastatic
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        # scheduled to run every 1 month : do not change as this is hard-wired into the values of all the parameters.

    def apply(self, population):
        df = population.props  # shortcut to dataframe
        m = self.module
        rng = m.rng

        # -------------------- ACQUISITION AND PROGRESSION OF CANCER (oac_status) -----------------------------------

        # determine if the person had a treatment during this stage of cancer (nb. treatment only has an effect on
        #  reducing progression risk during the stage at which is received.

        # todo: people can move through more than one stage per month (this event runs every month)
        # todo: I am guessing this is somehow a consequence of this way of looping through the stages
        # todo: I imagine this issue is the same for bladder cancer and oesophageal cancer

        had_treatment_during_this_stage = \
            df.is_alive & ~pd.isnull(df.oac_date_treatment) & \
            (df.oac_status == df.oac_stage_at_which_treatment_given)

        for stage, lm in self.module.linear_models_for_progession_of_oac_status.items():
            gets_new_stage = lm.predict(df.loc[df.is_alive], rng,
                                        had_treatment_during_this_stage=had_treatment_during_this_stage)
            idx_gets_new_stage = gets_new_stage[gets_new_stage].index
            df.loc[idx_gets_new_stage, 'oac_status'] = stage

        # -------------------- UPDATING OF SYMPTOM OF early_other_adult_ca_symptom OVER TIME ---------------------------
        # Each time this event is called (event 3 months) individuals may develop the symptom of other_adult_ca_symptom.
        # Once the symptom is developed it never resolves naturally. It may trigger health-care-seeking behaviour.
        onset_early_other_adult_ca_symptom = self.module.lm_onset_early_other_adult_ca_symptom.predict(
            df.loc[df.is_alive], rng
        )
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=onset_early_other_adult_ca_symptom[onset_early_other_adult_ca_symptom].index.tolist(),
            symptom_string='early_other_adult_ca_symptom',
            add_or_remove='+',
            disease_module=self.module
        )

        # -------------------- DEATH FROM Other_adult CANCER ---------------------------------------
        # There is a risk of death for those in metastatic only. Death is assumed to go instantly.
        metastatic_idx = df.index[df.is_alive & (df.oac_status == "metastatic")]
        selected_to_die = metastatic_idx[
            rng.random_sample(size=len(metastatic_idx)) < self.module.parameters['r_death_other_adult_cancer']]

        for person_id in selected_to_die:
            self.sim.schedule_event(
                    InstantaneousDeath(self.module, person_id, "OtherAdultCancer"), self.sim.date
            )
        df.loc[selected_to_die, 'oac_date_death'] = self.sim.date


# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
# ---------------------------------------------------------------------------------------------------------

class HSI_OtherAdultCancer_Investigation_Following_early_other_adult_ca_symptom(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_GenericFirstApptAtFacilityLevel1 following presentation for care with the symptom
    other_adult_ca_symptom.
    This event begins the investigation that may result in diagnosis of Other_adult Cancer and the scheduling of
    treatment or palliative care.
    It is for people with the symptom other_adult_ca_symptom.
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "OtherAdultCancer_Investigation"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        # Ignore this event if the person is no longer alive:
        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # Check that this event has been called for someone with the symptom other_adult_ca_symptom
        assert 'early_other_adult_ca_symptom' in self.sim.modules['SymptomManager'].has_what(person_id)

        # If the person is already diagnosed, then take no action:
        if not pd.isnull(df.at[person_id, "oac_date_diagnosis"]):
            return hs.get_blank_appt_footprint()

        # Use a diagnostic_device to diagnose whether the person has other adult cancer:
        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='diagnostic_device_for_other_adult_cancer_given_other_adult_ca_symptom',
            hsi_event=self
        )

        if dx_result:
            # record date of diagnosis:
            df.at[person_id, 'oac_date_diagnosis'] = self.sim.date

            # Check if is in metastatic:
            in_metastatic = df.at[person_id, 'oac_status'] == 'metastatic'
            # If the diagnosis does detect cancer, it is assumed that the classification as metastatic is
            # made accurately.

            if not in_metastatic:
                # start treatment:
                hs.schedule_hsi_event(
                    hsi_event=HSI_OtherAdultCancer_StartTreatment(
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
                    hsi_event=HSI_OtherAdultCancer_PalliativeCare(
                        module=self.module,
                        person_id=person_id
                    ),
                    priority=0,
                    topen=self.sim.date,
                    tclose=None
                )


class HSI_OtherAdultCancer_StartTreatment(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_OtherAdultCancer_Investigation_Following_other_adult_ca_symptom following a diagnosis
    of Other_adult Cancer. It initiates the treatment of Other_adult Cancer.
    It is only for persons with a cancer that is not in metastatic and who have been diagnosed.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "OtherAdultCancer_Treatment"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"MajorSurg": 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({"general_bed": 5})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        if df.at[person_id, "oac_status"] == 'metastatic':
            logger.warning(key="warning", data="Cancer is metastatic- aborting HSI_OtherAdultCancer_StartTreatment,"
                                               "scheduling HSI_OtherAdultCancer_PalliativeCare")

            hs.schedule_hsi_event(
                hsi_event=HSI_OtherAdultCancer_PalliativeCare(
                    module=self.module,
                    person_id=person_id,
                ),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )
            return self.make_appt_footprint({})
        # Check that the person has cancer, not in metastatic, has been diagnosed and is not on treatment
        assert not df.at[person_id, "oac_status"] == 'none'
        assert not pd.isnull(df.at[person_id, "oac_date_diagnosis"])
        assert pd.isnull(df.at[person_id, "oac_date_treatment"])

        # Record date and stage of starting treatment
        df.at[person_id, "oac_date_treatment"] = self.sim.date
        df.at[person_id, "oac_stage_at_which_treatment_given"] = df.at[person_id, "oac_status"]

        # Schedule a post-treatment check for 12 months:
        hs.schedule_hsi_event(
            hsi_event=HSI_OtherAdultCancer_PostTreatmentCheck(
                module=self.module,
                person_id=person_id,
            ),
            topen=self.sim.date + DateOffset(months=3),
            tclose=None,
            priority=0
        )

    def did_not_run(self):
        pass


class HSI_OtherAdultCancer_PostTreatmentCheck(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_OtherAdultCancer_StartTreatment and itself.
    It is only for those who have undergone treatment for Other_adult Cancer.
    If the person has developed cancer to metastatic, the patient is initiated on palliative care; otherwise a further
    appointment is scheduled for one year.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "OtherAdultCancer_Treatment"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # Check that the person is has cancer and is on treatment
        assert not df.at[person_id, "oac_status"] == 'none'
        assert not pd.isnull(df.at[person_id, "oac_date_diagnosis"])
        assert not pd.isnull(df.at[person_id, "oac_date_treatment"])

        if df.at[person_id, 'oac_status'] == 'metastatic':
            # If has progressed to metastatic, then start Palliative Care immediately:
            hs.schedule_hsi_event(
                hsi_event=HSI_OtherAdultCancer_PalliativeCare(
                    module=self.module,
                    person_id=person_id
                ),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )

        else:
            # Schedule another HSI_OtherAdultCancer_PostTreatmentCheck event in one month
            hs.schedule_hsi_event(
                hsi_event=HSI_OtherAdultCancer_PostTreatmentCheck(
                    module=self.module,
                    person_id=person_id
                ),
                topen=self.sim.date + DateOffset(months=3),
                tclose=None,
                priority=0
            )

    def did_not_run(self):
        pass


class HSI_OtherAdultCancer_PalliativeCare(HSI_Event, IndividualScopeEventMixin):
    """
    This is the event for palliative care. It does not affect the patients progress but does affect the disability
     weight and takes resources from the healthsystem.
    This event is scheduled by either:
    * HSI_OtherAdultCancer_Investigation_Following_other_adult_ca_symptom following a diagnosis of Other_adult Cancer
      at metastatic.
    * HSI_OtherAdultCancer_PostTreatmentCheck following progression to metastatic during treatment.
    * Itself for the continuance of care.
    It is only for persons with a cancer in metastatic.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "OtherAdultCancer_PalliativeCare"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
        self.ACCEPTED_FACILITY_LEVEL = '2'
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 15})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # Check that the person is in metastatic
        assert df.at[person_id, "oac_status"] == 'metastatic'

        # Record the start of palliative care if this is first appointment
        if pd.isnull(df.at[person_id, "oac_date_palliative_care"]):
            df.at[person_id, "oac_date_palliative_care"] = self.sim.date

        # Schedule another instance of the event for one month
        hs.schedule_hsi_event(
            hsi_event=HSI_OtherAdultCancer_PalliativeCare(
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

class OtherAdultCancerLoggingEvent(RegularEvent, PopulationScopeEventMixin):
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
            f'total_{k}': v for k, v in df.loc[df.is_alive].oac_status.value_counts().items()})

        # Current counts, undiagnosed
        out.update({f'undiagnosed_{k}': v for k, v in df.loc[df.is_alive].loc[
            pd.isnull(df.oac_date_diagnosis), 'oac_status'].value_counts().items()})

        # Current counts, diagnosed
        out.update({f'diagnosed_{k}': v for k, v in df.loc[df.is_alive].loc[
            ~pd.isnull(df.oac_date_diagnosis), 'oac_status'].value_counts().items()})

        # Current counts, on treatment (excl. palliative care)
        out.update({f'treatment_{k}': v for k, v in df.loc[df.is_alive].loc[(~pd.isnull(
            df.oac_date_treatment) & pd.isnull(
            df.oac_date_palliative_care)), 'oac_status'].value_counts().items()})

        # Current counts, on palliative care
        out.update({f'palliative_{k}': v for k, v in df.loc[df.is_alive].loc[
            ~pd.isnull(df.oac_date_palliative_care), 'oac_status'].value_counts().items()})

        # Counts of those that have been diagnosed, started treatment or started palliative care since last logging
        # event:
        date_now = self.sim.date
        date_lastlog = self.sim.date - pd.DateOffset(days=29)

        n_ge15 = (df.is_alive & (df.age_years >= 15)).sum()

        # todo: the .between function I think includes the two dates so events on these dates counted twice
        # todo:_ I think we need to replace with date_lastlog <= x < date_now
        n_newly_diagnosed_site_confined = (
                df.oac_date_diagnosis.between(date_lastlog, date_now) & (df.oac_status == 'site_confined1')).sum()
        n_newly_diagnosed_local_ln = (
                df.oac_date_diagnosis.between(date_lastlog, date_now) & (df.oac_status == 'local_ln')).sum()
        n_newly_diagnosed_metastatic = (
                df.oac_date_diagnosis.between(date_lastlog, date_now) & (df.oac_status == 'metastatic')).sum()

        n_sy_early_other_adult_ca_symptom = (df.is_alive & (df.sy_early_other_adult_ca_symptom >= 1)).sum()

        n_diagnosed_age_15_29 = (df.is_alive & (df.age_years >= 15) & (df.age_years < 30)
                                 & ~pd.isnull(df.oac_date_diagnosis)).sum()
        n_diagnosed_age_30_49 = (df.is_alive & (df.age_years >= 30) & (df.age_years < 50)
                                 & ~pd.isnull(df.oac_date_diagnosis)).sum()
        n_diagnosed_age_50p = (df.is_alive & (df.age_years >= 50) & ~pd.isnull(df.oac_date_diagnosis)).sum()

        n_diagnosed = (df.is_alive & ~pd.isnull(df.oac_date_diagnosis)).sum()

        out.update({
            'diagnosed_since_last_log': df.oac_date_diagnosis.between(date_lastlog, date_now).sum(),
            'treated_since_last_log': df.oac_date_treatment.between(date_lastlog, date_now).sum(),
            'palliative_since_last_log': df.oac_date_palliative_care.between(date_lastlog, date_now).sum(),
            'death_other_adult_cancer_since_last_log': df.oac_date_death.between(date_lastlog, date_now).sum(),
            'n age 15+': n_ge15,
            'n_newly_diagnosed_site_confined': n_newly_diagnosed_site_confined,
            'n_newly_diagnosed_local_ln': n_newly_diagnosed_local_ln,
            'n_newly_diagnosed_metastatic': n_newly_diagnosed_metastatic,
            'n_diagnosed_age_15_29': n_diagnosed_age_15_29,
            'n_diagnosed_age_30_49': n_diagnosed_age_30_49,
            'n_diagnosed_age_50p': n_diagnosed_age_50p,
            'n_sy_early_other_adult_ca_symptom': n_sy_early_other_adult_ca_symptom,
            'n_diagnosed': n_diagnosed
        })

        # logger.info('%s|summary_stats|%s', self.sim.date, out)
        logger.info(key='summary_stats',
                    data=out,
                    description='The summary information for the other adult cancer modules')
#       logger.info('%s|person_one|%s',
#                        self.sim.date,
#                       df.loc[3].to_dict())
