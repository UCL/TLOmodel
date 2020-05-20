"""
Oesophageal Cancer Disease Module

TODO:
* Consider adding palliative care : todo -- seems to be in there already!?
* Disability weights need to be updated?
* Needs to represent the the DxTest 'endoscopy_dysphagia_oes_cancer' requires use of an endoscope
* The age effect is very aggressive in the initiaisation: is that right?
* No benefit in daly_wt of palliative care -- should there be?
* we are sending these people to a specific HSI rather than a generic HSI. I think that's fine but we'll want to keep track of these decision and making sure that all is consistent
* representation of palliative care in the healthsystem: how often appointments, etc.
* property 'ca_stage_at_which_curative_treatment_was_begun' is never used.
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

import logging
from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent, Event
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import demography
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OesophagealCancer(Module):
    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    PARAMETERS = {
        "r_low_grade_dysplasia_none": Parameter(
            Types.REAL,
            "probabilty per 3 months of incident low grade oesophageal dysplasia, amongst people with no "
            "oesophageal dysplasia (men, age20, no excess alcohol, no tobacco)",
        ),
        "rr_low_grade_dysplasia_none_female": Parameter(
            Types.REAL, "rate ratio for low grade oesophageal dysplasia for females"
        ),
        "rr_low_grade_dysplasia_none_per_year_older": Parameter(
            Types.REAL, "rate ratio for low grade oesophageal dysplasia per year older from age 20"
        ),
        "rr_low_grade_dysplasia_none_tobacco": Parameter(
            Types.REAL, "rate ratio for low grade oesophageal dysplasia for tobacco smokers"
        ),
        "rr_low_grade_dysplasia_none_ex_alc": Parameter(
            Types.REAL, "rate ratio for low grade oesophageal dysplasia for no excess alcohol"
        ),
        "r_high_grade_dysplasia_low_grade_dysp": Parameter(
            Types.REAL,
            "probabilty per 3 months of high grade oesophageal dysplasia, amongst people with low grade dysplasia",
        ),
        "rr_high_grade_dysp_undergone_curative_treatment": Parameter(
            Types.REAL,
            "rate ratio for high grade dysplasia for people with low grade dysplasia "
            "if had curative treatment at low grade dysplasia stage",
        ),
        "r_stage1_high_grade_dysp": Parameter(
            Types.REAL, "probabilty per 3 months of stage 1 oesophageal cancer amongst people with high grade dysplasia"
        ),
        "rr_stage1_undergone_curative_treatment": Parameter(
            Types.REAL,
            "rate ratio for stage 1 oesophageal cancer for people with high grade "
            "dysplasia if had curative treatment at high grade dysplasia stage",
        ),
        "r_stage2_stage1": Parameter(
            Types.REAL, "probabilty per 3 months of stage 2 oesophageal cancer amongst people with stage 1"
        ),
        "rr_stage2_undergone_curative_treatment": Parameter(
            Types.REAL,
            "rate ratio for stage 2 oesophageal cancer for people with stage 1 "
            "oesophageal cancer if had curative treatment at stage 1",
        ),
        "r_stage3_stage2": Parameter(
            Types.REAL, "probabilty per 3 months of stage 3 oesophageal cancer amongst people with stage 2"
        ),
        "rr_stage3_undergone_curative_treatment": Parameter(
            Types.REAL,
            "rate ratio for stage 3 oesophageal cancer for people with stage 2 "
            "oesophageal cancer if had curative treatment at stage 2",
        ),
        "r_stage4_stage3": Parameter(
            Types.REAL, "probabilty per 3 months of stage 4 oesophageal cancer amongst people with stage 3"
        ),
        "rr_stage4_undergone_curative_treatment": Parameter(
            Types.REAL,
            "rate ratio for stage 4 oesophageal cancer for people with stage 3 "
            "oesophageal cancer if had curative treatment at stage 3",
        ),
        "rate_palliative_care_stage4": Parameter(
            Types.REAL, "prob palliative care this 3 month period if stage4"
        ),
        "r_death_oesoph_cancer": Parameter(
            Types.REAL,
            "probabilty per 3 months of death from oesophageal cancer mongst people with stage 4 oesophageal cancer",
        ),
        "rr_dysphagia_low_grade_dysp": Parameter(
            Types.REAL, "probability per 3 months of dysphagia in a person with low grade oesophageal dysplasia"
        ),
        "rr_dysphagia_high_grade_dysp": Parameter(
            Types.REAL, "rate ratio for dysphagia if have high grade oesophageal dysplasia"
        ),
        "r_dysphagia_stage1": Parameter(
            Types.REAL, "rate ratio for dysphagia if have stage 1 oesophageal cancer"
        ),
        "rr_dysphagia_stage2": Parameter(
            Types.REAL, "rate ratio for dysphagia if have stage 2 oesophageal cancer"
        ),
        "rr_dysphagia_stage3": Parameter(
            Types.REAL, "rate ratio for dysphagia if have stage 3 oesophageal cancer"
        ),
        "rr_dysphagia_stage4": Parameter(
            Types.REAL, "rate ratio for dysphagia if have stage 4 oesophageal cancer"
        ),
        "prob_present_dysphagia": Parameter(
            Types.REAL, "probability of presenting if have dysphagia"
        ),
        "init_prop_oes_cancer_stage": Parameter(
            Types.LIST,
            "initial proportions in ca_oesophagus categories for man aged 20 with no excess alcohol and no tobacco",
        ),
        "rp_oes_cancer_female": Parameter(
            Types.REAL, "relative prevalence at baseline of oesophageal dysplasia/cancer if female "
        ),
        "rp_oes_cancer_per_year_older": Parameter(
            Types.REAL, "relative prevalence at baseline of oesophageal dysplasia/cancer per year older than 20 "
        ),
        "rp_oes_cancer_tobacco": Parameter(
            Types.REAL, "relative prevalence at baseline of oesophageal dysplasia/cancer if tobacco "
        ),
        "rp_oes_cancer_ex_alc": Parameter(
            Types.REAL, "relative prevalence at baseline of oesophageal dysplasia/cancer "
        ),
        "init_prop_dysphagia_oes_cancer_by_stage": Parameter(
            Types.LIST, "initial proportions of people with oesophageal dysplasia/cancer diagnosed"
        ),
        "init_prop_with_dysphagia_diagnosed_oes_cancer_by_stage": Parameter(
            Types.LIST, "initial proportions of people with diagnosed oesophageal cancer with dysplasia"
        ),
        "init_prop_treatment_status_oes_cancer": Parameter(
            Types.LIST, "initial proportions of people with oesophageal dysplasia/cancer treated"
        ),
        "init_prob_palliative_care": Parameter(
            Types.REAL, "initial probability of being under palliative care if stage 4"
        ),
        "sensitivity_of_endoscopy_for_oes_cancer_with_dysphagia": Parameter(
            Types.REAL, "sensitivity of endoscopy_for diagnosis of oes_cancer_with_dysphagia"
        ),
        # these definitions for disability weights are the ones in the global burden of disease list (Salomon)
        "daly_wt_oes_cancer_controlled": Parameter(
            Types.REAL, "disability weight for oesophageal cancer controlled phase"
        ),
        "daly_wt_oes_cancer_terminal": Parameter(
            Types.REAL, "disability weight for oesophageal cancer terminal"
        ),
        "daly_wt_oes_cancer_metastatic": Parameter(
            Types.REAL, "disability weight for oesophageal cancer metastatic"
        ),
        "daly_wt_oes_cancer_primary_therapy": Parameter(
            Types.REAL, "disability weight for oesophageal cancer primary therapy"
        ),
    }

    PROPERTIES = {
        "ca_oesophagus": Property(
            Types.CATEGORICAL,
            "Current status of the health condition, oesophageal dysplasia",
            categories=["none", "low_grade_dysplasia", "high_grade_dysplasia", "stage1", "stage2", "stage3", "stage4"],
        ),

        "ca_oesophagus_any": Property(Types.BOOL,
                                      "Current status of having any Oesophageal Cancer (equal to: ~(ca_oesophagus=='none')"),

        "ca_date_oes_cancer_diagnosis": Property(
            Types.DATE,
            "the date of diagnsosis of the oes_cancer (pd.NaT if never diagnosed)"
        ),

        "ca_date_treatment_oesophageal_cancer": Property(
            Types.DATE,
            "date of first receiving attempted curative treatment (pd.NaT if never started treatment)"),

        "ca_stage_at_which_curative_treatment_was_begun": Property(
            Types.CATEGORICAL,
            "The stage of cancer when curative treatment was initiated (if it was initiated)",
            categories=["never", "low_grade_dysplasia", "high_grade_dysplasia", "stage1", "stage2", "stage3"],
        ),  # todo - this is never used!?

        "ca_date_palliative_care": Property(
            Types.DATE,
            "date of first receiving palliative care (pd.NaT is never had palliative care)"),
        # todo - is is right that it never ends?
    }

    # Symptom that this module will use
    SYMPTOMS = {'dysphagia'}    # (dysphagia means problems swallowing)

    def read_parameters(self, data_folder):
        """Setup parameters used by the module, now including disability weights
        """

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

        # Update parameters from the resource dataframe
        self.load_parameters_from_dataframe(
            pd.read_excel(Path(self.resourcefilepath) / "ResourceFile_Oesophageal_Cancer.xlsx",
                          sheet_name="parameter_values")
        )

        # Get DALY weight values:
        if "HealthBurden" in self.sim.modules.keys():
            self.parameters["daly_wt_oes_cancer_stage_1_3"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=550
            )
            self.parameters["daly_wt_oes_cancer_stage4"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=549
            )
            self.parameters["daly_wt_treated_oes_cancer"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=547
            )

    def initialise_population(self, population):
        """Set our property values for the initial population."""
        df = population.props  # a shortcut to the data-frame storing data for individuals

        # -------------------- ASSIGN VALUES OF OESOPHAGEAL DYSPLASIA/CANCER STATUS AT BASELINE -----------

        # ----- Determine who has cancer at ANY cancer stage:
        # defaults:
        df.loc[df.is_alive, "ca_oesophagus"] = "none"
        df.loc[df.is_alive, "ca_oesophagus_any"] = False

        lm_init_ca_oesophagus_any = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            sum(self.parameters['init_prop_oes_cancer_stage']),
            Predictor('li_ex_alc').when(True, self.parameters['rp_oes_cancer_ex_alc']),
            Predictor('li_tob').when(True, self.parameters['rp_oes_cancer_tobacco']),
            Predictor('age_years').apply(
                lambda x: ((x - 20) ** self.parameters['rp_oes_cancer_per_year_older']) if x > 20 else 0.0)
        )
        df.loc[df.is_alive, "ca_oesophagus_any"] = lm_init_ca_oesophagus_any.predict(df, self.rng)

        # Determine the stage of the cancer for those who do have a cancer:
        df.loc[(df.is_alive) & (df.ca_oesophagus_any), "ca_oesophagus"] = self.rng.choice(
            [val for val in self.PROPERTIES['ca_oesophagus'].categories if val != 'none'],
            df.loc[df.is_alive, "ca_oesophagus_any"].sum(),
            p=[p / sum(self.parameters['init_prop_oes_cancer_stage']) for p in
               self.parameters['init_prop_oes_cancer_stage']]
        )

        # ----- Impose the symptom of random sample of those in each cancer stage to have the symptom of dysphagia:
        lm_init_disphagia = LinearModel.multiplicative(
            Predictor('ca_oesophagus')
                .when("none", 0.0)
                .when("low_grade_dysplasia", self.parameters['init_prop_dysphagia_oes_cancer_by_stage'][0])
                .when("high_grade_dysplasia", self.parameters['init_prop_dysphagia_oes_cancer_by_stage'][1])
                .when("stage1", self.parameters['init_prop_dysphagia_oes_cancer_by_stage'][2])
                .when("stage2", self.parameters['init_prop_dysphagia_oes_cancer_by_stage'][3])
                .when("stage3", self.parameters['init_prop_dysphagia_oes_cancer_by_stage'][4])
                .when("stage4", self.parameters['init_prop_dysphagia_oes_cancer_by_stage'][5])
        )
        has_dysphagia_at_init = lm_init_disphagia.predict(df, self.rng)
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=has_dysphagia_at_init.index[has_dysphagia_at_init].tolist(),
            symptom_string='dysphagia',
            add_or_remove='+',
            disease_module=self
        )

        # ----- Determine which persons with oes_cancer have been diagnosed
        # default:
        df.loc[df.is_alive, "ca_date_oes_cancer_diagnosis"] = pd.NaT

        lm_init_diagnosed = LinearModel.multiplicative(
            Predictor('ca_oesophagus')
                .when("none", 0.0)
                .when("low_grade_dysplasia",
                      self.parameters['init_prop_with_dysphagia_diagnosed_oes_cancer_by_stage'][0])
                .when("high_grade_dysplasia",
                      self.parameters['init_prop_with_dysphagia_diagnosed_oes_cancer_by_stage'][1])
                .when("stage1", self.parameters['init_prop_with_dysphagia_diagnosed_oes_cancer_by_stage'][2])
                .when("stage2", self.parameters['init_prop_with_dysphagia_diagnosed_oes_cancer_by_stage'][3])
                .when("stage3", self.parameters['init_prop_with_dysphagia_diagnosed_oes_cancer_by_stage'][4])
                .when("stage4", self.parameters['init_prop_with_dysphagia_diagnosed_oes_cancer_by_stage'][5])
        )
        ever_diagnosed = lm_init_diagnosed.predict(df, self.rng)

        # Set data of diagnosis (300-600 days ago)
        df["ca_date_oes_cancer_diagnosis"] = pd.Series(
            index=ever_diagnosed.index,
            data=[pd.NaT if (not x) else (self.sim.date - DateOffset(days=self.rng.randint(300, 600))) for x in
                  ever_diagnosed]
        )

        # ----- Determine which persons that have been diagnosed have started a treatment
        # defaults:
        df.loc[df.is_alive, "ca_date_treatment_oesophageal_cancer"] = pd.NaT
        df.loc[df.is_alive, "ca_stage_at_which_curative_treatment_was_begun"] = "never"

        lm_init_treatment_for_those_diagnosed = LinearModel.multiplicative(
            Predictor('ca_oesophagus')
                .when("none", 0.0)
                .when("low_grade_dysplasia", self.parameters['init_prop_treatment_status_oes_cancer'][0])
                .when("high_grade_dysplasia", self.parameters['init_prop_treatment_status_oes_cancer'][1])
                .when("stage1", self.parameters['init_prop_treatment_status_oes_cancer'][2])
                .when("stage2", self.parameters['init_prop_treatment_status_oes_cancer'][3])
                .when("stage3", self.parameters['init_prop_treatment_status_oes_cancer'][4])
                .when("stage4", self.parameters['init_prop_treatment_status_oes_cancer'][5]),
        )
        treatment_initiated = lm_init_treatment_for_those_diagnosed.predict(df, self.rng)

        # prevent treatment having been initiated for anyone who is not yet diagnosed
        treatment_initiated.loc[pd.isnull(df.ca_date_oes_cancer_diagnosis)] = False

        # prevent treatment having been initiated from anyone who is not in stage4
        treatment_initiated.loc[df.ca_oesophagus == 'stage4'] = False

        # set date at which treatment began (100-300 days ago)
        df["ca_date_treatment_oesophageal_cancer"] = pd.Series(
            index=treatment_initiated.index,
            data=[pd.NaT if (not x) else (self.sim.date - DateOffset(days=self.rng.randint(100, 300))) for x in
                  treatment_initiated]
        )

        # Assume that the treatment was begun in the stage of cancer that they are in now (i.e. treatment was begun
        # recently)
        df.loc[treatment_initiated, 'ca_stage_at_which_curative_treatment_was_begun'].values[:] = df.loc[
            treatment_initiated, 'ca_oesophagus']

        # ----- Determine which persons that have been diagnosed and started a palliative care (only those in stage4)
        # default:
        df.loc[df.is_alive, "ca_date_palliative_care"] = pd.NaT

        persons_in_stage4 = df.index[df.is_alive & (df.ca_oesophagus == 'stage4')]
        selected_for_palliative_care = persons_in_stage4[
            self.rng.random_sample(size=len(persons_in_stage4)) < self.parameters['init_prob_palliative_care']
            ]
        palliative_care_initiated = pd.Series(index=df.index, data=False)
        palliative_care_initiated[selected_for_palliative_care] = True

        # set date of palliative care being initiated (in the last 100 days)
        df["ca_date_palliative_care"] = pd.Series(
            index=palliative_care_initiated.index,
            data=[pd.NaT if (not x) else (self.sim.date - DateOffset(days=self.rng.randint(0, 100))) for x in
                  palliative_care_initiated]
        )

    def initialise_simulation(self, sim):
        """
        * Schedule the main polling event
        * Shcedule the main logging event
        * Define the LinearModels for ca_oesphagus progression, onset_dysphagia
        * Define the diagnostic used
        """

        # Schedule main polling event to happen immediately
        sim.schedule_event(OesCancerMainPollingEvent(self), sim.date + DateOffset(months=0))

        # Schedule logging event to happen immediately
        sim.schedule_event(OesCancerLoggingEvent(self), sim.date + DateOffset(months=0))

        # Define LinearModels for the progression of cancer
        self.linear_models_for_progession_of_ca_oesophagus = dict()
        self.linear_models_for_progession_of_ca_oesophagus[
            'low_grade_dysplasia'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            self.parameters['r_low_grade_dysplasia_none'],  # todo - rename
            Predictor('age_years')
                .apply(lambda x: 0 if x < 20 else (x - 20) ** self.parameters['rr_low_grade_dysplasia_none_per_year_older']),
            Predictor('sex')
                .when('F', self.parameters['rr_low_grade_dysplasia_none_female']),
            Predictor('li_tob')
                .when(True, self.parameters['rr_low_grade_dysplasia_none_tobacco']),
            Predictor('li_ex_alc')
                .when(True, self.parameters['rr_low_grade_dysplasia_none_ex_alc']),
            Predictor('ca_oesophagus')
                .when('none', 1.0)
                .otherwise(0.0)
        )

        self.linear_models_for_progession_of_ca_oesophagus[
            'high_grade_dysplasia'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            self.parameters['r_high_grade_dysplasia_low_grade_dysp'],  # todo - rename
            Predictor('currently_on_treatment', external=True)
                .when(True, self.parameters['rr_high_grade_dysp_undergone_curative_treatment']),
            Predictor('ca_oesophagus')
                .when('low_grade_dysplasia', 1.0)
                .otherwise(0.0)
        )

        self.linear_models_for_progession_of_ca_oesophagus[
            'stage1'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            self.parameters['r_stage1_high_grade_dysp'],  # todo - rename
            Predictor('currently_on_treatment', external=True)
                .when(True, self.parameters['rr_stage1_undergone_curative_treatment']),
            Predictor('ca_oesophagus')
                .when('high_grade_dysplasia', 1.0)
                .otherwise(0.0)
        )

        self.linear_models_for_progession_of_ca_oesophagus[
            'stage2'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            self.parameters['r_stage2_stage1'],  # todo - rename
            Predictor('currently_on_treatment', external=True)
                .when(True, self.parameters['rr_stage2_undergone_curative_treatment']),
            Predictor('ca_oesophagus')
                .when('stage1', 1.0)
                .otherwise(0.0)
        )

        self.linear_models_for_progession_of_ca_oesophagus[
            'stage3'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            self.parameters['r_stage3_stage2'],  # todo - rename
            Predictor('currently_on_treatment', external=True)
                .when(True, self.parameters['rr_stage3_undergone_curative_treatment']),
            Predictor('ca_oesophagus')
                .when('stage2', 1.0)
                .otherwise(0.0)
        )

        self.linear_models_for_progession_of_ca_oesophagus[
            'stage4'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            self.parameters['r_stage4_stage3'],  # todo - rename
            Predictor('currently_on_treatment', external=True)
                .when(True, self.parameters['rr_stage4_undergone_curative_treatment']),
            Predictor('ca_oesophagus')
                .when('stage3', 1.0)
                .otherwise(0.0)
        )

        # Check that the dict labels are correct as these are used to set the value of ca_oesophagus
        assert (set(self.linear_models_for_progession_of_ca_oesophagus.keys()).union({'none'})) == \
               set(self.PROPERTIES['ca_oesophagus'].categories)

        # Linear Model for the onset of dysphagia # todo - rename to specify that it is 3mo
        self.lm_onset_dysphagia = LinearModel.multiplicative(
            Predictor('ca_oesophagus')
                .when('low_grade_dysplasia',
                      self.parameters['rr_dysphagia_low_grade_dysp'] * self.parameters['r_dysphagia_stage1'])
                .when('high_grade_dysplaisa',
                      self.parameters['rr_dysphagia_high_grade_dysp'] * self.parameters['r_dysphagia_stage1'])
                .when('stage1', self.parameters['r_dysphagia_stage1'])
                .when('stage2', self.parameters['rr_dysphagia_stage2'] * self.parameters['r_dysphagia_stage1'])
                .when('stage3', self.parameters['rr_dysphagia_stage3'] * self.parameters['r_dysphagia_stage1'])
                .when('stage4', self.parameters['rr_dysphagia_stage4'] * self.parameters['r_dysphagia_stage1'])
                .otherwise(0.0)
        )

        # ----- DX TESTS -----
        # Create the diagnostic test representing the use of an endoscope to ca_oesophagus
        # This properties of conditional on the test being done only to persons with the Symptom, 'dysphagia'.
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            endoscopy_for_oes_cancer_given_dysphagia=DxTest(
                property='ca_oesophagus_any',
                sensitivity=self.parameters['sensitivity_of_endoscopy_for_oes_cancer_with_dysphagia']
            )
        )



    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props
        df.at[child_id, "ca_oesophagus"] = "none"
        df.at[child_id, "ca_oesophagus_any"] = False
        df.at[child_id, "ca_date_oes_cancer_diagnosis"] = pd.NaT
        df.at[child_id, "cca_date_treatment_oesophageal_cancer"] = pd.NaT
        df.at[child_id, "ca_stage_at_which_curative_treatment_was_begun"] = "never"
        df.at[child_id, "ca_date_palliative_care"] = pd.NaT

    def on_hsi_alert(self, person_id, treatment_id):
        pass

    def report_daly_values(self):
        # This must send back a dataframe that reports on the HealthStates for all individuals over
        # the past month

        df = self.sim.population.props  # shortcut to population properties dataframe

        disability_series_for_alive_persons = pd.Series(index=df.index[df.is_alive], data=0.0)

        # Assign daly_wt to those with cancer stages before stage4 and never treated
        disability_series_for_alive_persons.loc[
            (
                pd.isnull(df.ca_date_treatment_oesophageal_cancer) & (
                    (df.ca_oesophagus == "low_grade_dysplasia") |
                    (df.ca_oesophagus == "high_grade_dysplasia") |
                    (df.ca_oesophagus == "stage1") |
                    (df.ca_oesophagus == "stage2") |
                    (df.ca_oesophagus == "stage3")
                )
            )
        ] = self.parameters['daly_wt_oes_cancer_stage_1_3']

        # Assign daly_wt to those with cancer stages before stage4 and who have been treated
        disability_series_for_alive_persons.loc[
            (
                ~pd.isnull(df.ca_date_treatment_oesophageal_cancer) & (
                    (df.ca_oesophagus == "low_grade_dysplasia") |
                    (df.ca_oesophagus == "high_grade_dysplasia") |
                    (df.ca_oesophagus == "stage1") |
                    (df.ca_oesophagus == "stage2") |
                    (df.ca_oesophagus == "stage3")
                )
            )
        ] = self.parameters['daly_wt_treated_oes_cancer']

        # Assign daly_wt to those in stage4 cancer (irrespective of whether receiving palliative care)
        disability_series_for_alive_persons.loc[
            (df.ca_oesophagus == "stage4")
        ] = self.parameters['daly_wt_oes_cancer_stage4']

        return disability_series_for_alive_persons


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------

class OesCancerMainPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all oesophageal cancer properties for population:
    * Acquisition and progression of Oesophagel Cancer
    * Sympton Development according to stage of Oesophagel Cancer
    * Deaths from Oesophagel Cancer for those in stage4
    """

    def __init__(self, module):
        """scheduled to run every 3 months: do not change."""
        super().__init__(module, frequency=DateOffset(months=3))

    def apply(self, population):

        df = population.props
        m = self.module
        rng = m.rng

        # -------------------- ACQUISITION AND PROGRESSION OF CA-OESOPHAGUS -----------------------------------
        currently_on_treatment = ~pd.isnull(df.ca_date_treatment_oesophageal_cancer)
        for stage, lm in self.module.linear_models_for_progession_of_ca_oesophagus.items():
            gets_new_stage = lm.predict(df, rng, currently_on_treatment=currently_on_treatment)
            idx_gets_new_stage = gets_new_stage[gets_new_stage].index
            df.loc[idx_gets_new_stage, 'ca_oesophagus'] = stage
            df.loc[idx_gets_new_stage, 'ca_oesophagus_any'] = True

        # -------------------- UPDATING OF DYSPHAGIA OVER TIME --------------------------------
        # Each time this event is called (event 3 months) individuals may develop the symptom of dysphagia.
        # Once the symptom is developed it never resolves naturally. It may trigger health-care-seeking behaviour.
        onset_dysphagia = self.module.lm_onset_dysphagia.predict(df, rng)
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=onset_dysphagia[onset_dysphagia].index.tolist(),
            symptom_string='dysphagia',
            add_or_remove='+',
            disease_module=self.module
        )

        # -------------------- DEATH FROM OESOPHAGEAL CANCER ---------------------------------------
        # There is a risk of death for those in stage4 only. Death is assumed to go instantly.
        stage4_idx = df.index[df.is_alive & (df.ca_oesophagus == "stage4")]
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
    It begins the investigation that may result in diagnosis of Oesophageal Cancer and the scheduling of treatment or
    palliative care.
    It is for people with the symptom dysphagia
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

        # Check that this event has been called for someone with the symptom dysphagia
        assert 'dysphagia' in self.sim.modules['SymptomManager'].has_what(person_id)

        # If the person is already diagnosed, then take no action:
        if not pd.isnull(self.sim.population.props.at[person_id, "ca_date_oes_cancer_diagnosis"]):
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        # Ignore this event if the person is no longer alive:
        if not self.sim.population.props.at[person_id, 'is_alive']:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        # Check if is in stage4
        in_stage4 = self.sim.population.props.at[person_id, 'ca_oesophagus'] == 'stage4'

        # Use an endoscope to diagnose whether the person has Oesophageal Cancer:
        # If the diagnsosis does detect cancer, it is assumed that the classification as stage4 is made accurately.
        dx_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
            dx_tests_to_run='endoscopy_for_oes_cancer_given_dysphagia',
            hsi_event=self
        )

        if dx_result and (not in_stage4):
            # record date of diagnosis:
            self.sim.population.props.at[person_id, 'ca_date_oes_cancer_diagnosis'] = self.sim.date

            # start treatment:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event= HSI_OesophagealCancer_StartTreatment(
                    module=self.module,
                    person_id=person_id
                ),
                priority=0,
                topen=self.sim.date,
                tclose=None
            )

        elif dx_result and in_stage4:
            # record date of diagnosis:
            self.sim.population.props.at[person_id, 'ca_date_oes_cancer_diagnosis'] = self.sim.date

            # start palliative care
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event= HSI_OesophagealCancer_PalliativeCare(
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
    Oesophageal Cancer. It is only for persons with a cancer that is not in stage4.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["Over5OPD"] = 1
        the_appt_footprint["MajorSurg"] = 3

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "OesophagealCancer_StartTreatment"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 3
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        if not self.sim.population.props.at[person_id, 'is_alive']:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        # Check that the person has cancer, not in stage4, and is not on treatment
        assert self.sim.population.props.at[person_id, "ca_oesophagus_any"]
        assert not self.sim.population.props.at[person_id, "ca_oesophagus"] == 'stage4'
        assert pd.isnull(self.sim.population.props.at[person_id, "ca_date_treatment_oesophageal_cancer"])

        # Record date of starting treatment
        df = self.sim.population.props
        df.at[person_id, "ca_date_treatment_oesophageal_cancer"] = self.sim.date
        df.at[person_id, "ca_stage_at_which_curative_treatment_was_begun"] = df.at[person_id, "ca_oesophagus"]

        self.sim.modules['HealthSystem'].schedule_hsi_event(
            hsi_event=HSI_OesophagealCancer_MonitorTreatment(
                module=self.module,
                person_id=person_id,
            ),
            topen=self.sim.date + DateOffset(months=1),
            tclose=None,
            priority=0
        )

    def did_not_run(self):
        pass

class HSI_OesophagealCancer_MonitorTreatment(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_OesophagealCancer_StartTreatment and itself.
    It is only for those undergoing treatment for Oesophageal Cancer.
    If the person has developed cancer to stage4, then treatment is stopped and palliative care is begun.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["Over5OPD"] = 1
        the_appt_footprint["MajorSurg"] = 3

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "OesophagealCancer_MonitorTreatment"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 3
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        if not self.sim.population.props.at[person_id, 'is_alive']:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        # Check that the person is has cancer and is on treatment
        assert self.sim.population.props.at[person_id, "ca_oesophagus_any"]
        assert not pd.isnull(self.sim.population.props.at[person_id, "ca_date_treatment_oesophageal_cancer"])


        if self.sim.population.props.at[person_id, 'ca_oesophagus'] == 'stage4':
            # If has progressed to stage4, then start Palliative Care immediately:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_OesophagealCancer_PalliativeCare(
                    module=self.module,
                    person_id=person_id
                ),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )

        else:
            # Schedule another HSI_OesophagealCancer_MonitorTreatment event in one month
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_OesophagealCancer_MonitorTreatment(
                    module=self.module,
                    person_id=person_id
                ),
                topen=self.sim.date + DateOffset(months=1),
                tclose=None,
                priority=0
            )

    def did_not_run(self):
        pass

class HSI_OesophagealCancer_PalliativeCare(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by either:
    * HSI_OesophagealCancer_Investigation_Following_Dysphagia following a diagnosis of Oesophageal Cancer at stage4.
    * HSI_OesophagealCancer_MonitorTreatment following progression to stage4 during treatment.
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

        if not self.sim.population.props.at[person_id, 'is_alive']:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        # Check that the person is in stage4
        assert self.sim.population.props.at[person_id, "ca_oesophagus"] == 'stage4'

        # Record the start of palliative care if this is first appointment
        if pd.isnull(self.sim.population.props.at[person_id, "ca_date_palliative_care"]):
            self.sim.population.props.at[person_id, "ca_date_palliative_care"] = self.sim.date

        # Schedule another instance of the event for one month
        self.sim.modules['HealthSystem'].schedule_hsi_event(
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
        dict_for_output = {}

        # Current counts, total
        dict_for_output.update({f'total_{k}': v for k, v in df.ca_oesophagus.value_counts().to_dict().items()})

        # Current counts, undiagnosed
        dict_for_output.update({f'undiagnosed_{k}': v for k, v in df.loc[pd.isnull(df.ca_date_oes_cancer_diagnosis),'ca_oesophagus'].value_counts().to_dict().items()})

        # Current counts, diagnosed
        dict_for_output.update({f'diagnosed_{k}': v for k, v in df.loc[~pd.isnull(df.ca_date_oes_cancer_diagnosis),'ca_oesophagus'].value_counts().to_dict().items()})

        # Current counts, on treatment
        dict_for_output.update({f'treatment_{k}': v for k, v in df.loc[(~pd.isnull(df.ca_date_treatment_oesophageal_cancer) & pd.isnull(df.ca_date_palliative_care)),'ca_oesophagus'].value_counts().to_dict().items()})

        # Current counts, on palliative treatment
        dict_for_output.update({f'palliative_{k}': v for k, v in df.loc[~pd.isnull(df.ca_date_palliative_care), 'ca_oesophagus'].value_counts().to_dict().items()})

        # Counts of those that have been diagnosed, started treatment or started palliative care since last logging event:
        date_now = self.sim.date
        date_lastlog = self.sim.date - pd.DateOffset(months=self.repeat)

        dict_for_output.update({
            'diagnosed_since_last_log': df.ca_date_oes_cancer_diagnosis.between(date_lastlog, date_now).sum(),
            'treated_since_last_log': df.ca_date_treatment_oesophageal_cancer.between(date_lastlog, date_now).sum(),
            'palliative_since_last_log': df.ca_date_palliative_care.between(date_lastlog, date_now).sum()
        })

        logger.info('%s|summary_stats|%s', self.sim.date, dict_for_output)
