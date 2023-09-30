"""
Bladder Cancer Disease Module

Limitations to note:
* Needs to represent the the DxTest 'cytoscopy_blood_urine_bladder_cancer' requires use of a cytoscope
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


class BladderCancer(Module):
    """Bladder Cancer Disease Module"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.linear_models_for_progession_of_bc_status = dict()
        self.lm_onset_blood_urine = None
        self.lm_onset_pelvic_pain = None
        self.daly_wts = dict()

    INIT_DEPENDENCIES = {'Demography', 'Lifestyle', 'HealthSystem', 'SymptomManager'}

    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden'}

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'BladderCancer': Cause(gbd_causes='Bladder cancer', label='Cancer (Bladder)'),
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'BladderCancer': Cause(gbd_causes='Bladder cancer', label='Cancer (Bladder)'),
    }

    PARAMETERS = {
        "init_prop_bladder_cancer_stage": Parameter(
            Types.LIST,
            "initial proportions in bladder cancer categories for person aged 15-19 and no tobacco and no schisto_h"
        ),
        "init_prop_blood_urine_bladder_cancer_by_stage": Parameter(
            Types.LIST, "initial proportions of those with bladder cancer categories that have the symptom blood urine"
        ),
        "init_prop_pelvic_pain_bladder_cancer_by_stage": Parameter(
            Types.LIST, "initial proportions of those with bladder cancer categories that have pelvic pain"
        ),
        "init_prop_with_blood_urine_diagnosed_bladder_cancer_by_stage": Parameter(
            Types.LIST, "initial proportions of people that have symptom of blood urine that have been diagnosed"
        ),
        "init_prop_with_pelvic_pain_diagnosed_bladder_cancer_by_stage": Parameter(
            Types.LIST, "initial proportions of people that have symptom of pelvic pain that have been diagnosed"
        ),
        "init_prop_treatment_status_bladder_cancer": Parameter(
            Types.LIST, "initial proportions of people with bladder cancer that had initiated treatment"
        ),
        "init_prob_palliative_care": Parameter(
            Types.REAL, "initial probability of being under palliative care if in metastatic"
        ),
        "r_tis_t1_bladder_cancer_none": Parameter(
            Types.REAL,
            "probability per 3 months of incident tis_t1 bladder cancer, amongst people with no bladder cancer"
            "(for person aged 15-19 and no tobacco and no schisto_h)",
        ),
        "rr_tis_t1_bladder_cancer_none_age3049": Parameter(
            Types.REAL, "rate ratio for tis_t1 bladder cancer for age 30-49"
        ),
        "rr_tis_t1_bladder_cancer_none_age5069": Parameter(
            Types.REAL, "rate ratio for tis_t1 bladder cancer for age 50-69"
        ),
        "rr_tis_t1_bladder_cancer_none_agege70": Parameter(
            Types.REAL, "rate ratio for tis_t1 bladder cancer for age 70+"
        ),
        "rr_tis_t1_bladder_cancer_none_tobacco": Parameter(
            Types.REAL, "rate ratio for tis_t1 bladder cancer for tobacco smokers"
        ),
        "rr_tis_t1_bladder_cancer_none_schisto_h": Parameter(
            Types.REAL, "rate ratio for tis_t1 bladder cancer for schisto_h"
        ),
        "r_t2p_bladder_cancer_tis_t1": Parameter(
            Types.REAL,
            "probability per 3 months of t2+ bladder cancer, amongst people with tis_t1 bladder cancer",
        ),
        "rr_t2p_bladder_cancer_undergone_curative_treatment": Parameter(
            Types.REAL,
            "rate ratio for t2+ bladder cancer for people with tis_t1 bladder cancer "
            "if had curative treatment at tis_t1 bladder cancer stage",
        ),
        "r_metastatic_t2p_bladder_cancer": Parameter(
            Types.REAL, "probability per 3 months of metastatic bladder cancer amongst people with t2+ bladder cancer"
        ),
        "rr_metastatic_undergone_curative_treatment": Parameter(
            Types.REAL,
            "rate ratio for metastatic bladder cancer for people with t2+ bladder cancer "
            "if had curative treatment at t2+ bladder cancer stage",
        ),
        "rate_palliative_care_metastatic": Parameter(
            Types.REAL, "prob palliative care this 3 month period if metastatic bladder cancer"
        ),
        "r_death_bladder_cancer": Parameter(
            Types.REAL,
            "probability per 3 months of death from bladder cancer amongst people with metastatic bladder cancer",
        ),
        "r_blood_urine_tis_t1_bladder_cancer": Parameter(
            Types.REAL, "probability per 3 months of blood_urine in a person with tis_t1 bladder cancer"
        ),
        "rr_blood_urine_t2p_bladder_cancer": Parameter(
            Types.REAL, "rate ratio for blood_urine if have t2p bladder cancer"
        ),
        "rr_blood_urine_metastatic_bladder_cancer": Parameter(
            Types.REAL, "rate ratio for blood_urine if have metastatic bladder cancer"
        ),
        "r_pelvic_pain_tis_t1_bladder_cancer": Parameter(
            Types.REAL, "probability per 3 months of pelvic_pain in a person with tis_t1 bladder cancer"
        ),
        "rr_pelvic_pain_t2p_bladder_cancer": Parameter(
            Types.REAL, "rate ratio for pelvic_pain if have t2p bladder cancer"
        ),
        "rr_pelvic_pain_metastatic_bladder_cancer": Parameter(
            Types.REAL, "rate ratio for pelvic_pain if have metastatic bladder cancer"
        ),
        "rp_bladder_cancer_age3049": Parameter(
            Types.REAL, "relative prevalence at baseline of bladder cancer/cancer age 30-49"
        ),
        "rp_bladder_cancer_age5069": Parameter(
            Types.REAL, "relative prevalence at baseline of bladder cancer/cancer age 50-69"
        ),
        "rp_bladder_cancer_agege70": Parameter(
            Types.REAL, "relative prevalence at baseline of bladder cancer/cancer age 70+"
        ),
        "rp_bladder_cancer_tobacco": Parameter(
            Types.REAL, "relative prevalence at baseline of bladder cancer if tobacco"
        ),
        "rp_bladder_cancer_schisto_h": Parameter(
            Types.REAL, "relative prevalence at baseline of bladder cancer if schisto_h"
        ),
        "sensitivity_of_cytoscopy_for_bladder_cancer_blood_urine": Parameter(
            Types.REAL, "sensitivity of cytoscopy_for diagnosis of bladder cancer given blood urine"
        ),
        "sensitivity_of_cytoscopy_for_bladder_cancer_pelvic_pain": Parameter(
            Types.REAL, "sensitivity of cytoscopy_for diagnosis of bladder cancer given pelvic pain"
        )
    }

    PROPERTIES = {
        "bc_status": Property(
            Types.CATEGORICAL,
            "Current status of the health condition, bladder cancer",
            categories=["none", "tis_t1", "t2p", "metastatic"],
        ),
        "bc_date_diagnosis": Property(
            Types.DATE,
            "the date of diagnosis of the bladder cancer (pd.NaT if never diagnosed)"
        ),
        "bc_date_treatment": Property(
            Types.DATE,
            "date of first receiving attempted curative treatment (pd.NaT if never started treatment)"
        ),
        "bc_stage_at_which_treatment_given": Property(
            Types.CATEGORICAL,
            "the cancer stage at which treatment is given (because the treatment only has an effect during the stage"
            "at which it is given ",
            categories=["none", "tis_t1", "t2p", "metastatic"],
        ),
        "bc_date_palliative_care": Property(
            Types.DATE,
            "date of first receiving palliative care (pd.NaT is never had palliative care)"
        ),
        "bc_date_death": Property(
            Types.DATE,
            "date bc death"
        )
    }

    def read_parameters(self, data_folder):
        """Setup parameters used by the module, now including disability weights"""

        # Update parameters from the resourcefile
        self.load_parameters_from_dataframe(
            pd.read_excel(Path(self.resourcefilepath) / "ResourceFile_Bladder_Cancer.xlsx",
                          sheet_name="parameter_values")
        )

        # Register Symptom that this module will use
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='blood_urine',
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
        df.loc[df.is_alive, "bc_status"] = "none"
        df.loc[df.is_alive, "bc_date_diagnosis"] = pd.NaT
        df.loc[df.is_alive, "bc_date_treatment"] = pd.NaT
        df.loc[df.is_alive, "bc_stage_at_which_treatment_given"] = "none"
        df.loc[df.is_alive, "bc_date_palliative_care"] = pd.NaT
        df.loc[df.is_alive, "bc_date_death"] = pd.NaT

        # -------------------- bc_status -----------
        # Determine who has cancer at ANY cancer stage:
        # check parameters are sensible: probability of having any cancer stage cannot exceed 1.0
        assert sum(p['init_prop_bladder_cancer_stage']) <= 1.0

        lm_init_bc_status_any_stage = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            sum(p['init_prop_bladder_cancer_stage']),
            Predictor('li_tob').when(True, p['rp_bladder_cancer_tobacco']),
            # todo: add line when schisto is merged
            # Predictor('sh_infection_status').when('High-infection', p['rp_bladder_cancer_schisto_h']),
            Predictor('age_years', conditions_are_mutually_exclusive=True)
            .when('.between(30,49)', p['rp_bladder_cancer_age3049'])
            .when('.between(50,69)', p['rp_bladder_cancer_age5069'])
            .when('.between(70,120)', p['rp_bladder_cancer_agege70'])
            .when('.between(0,14)', 0.0)
        )

        bc_status_any_stage = lm_init_bc_status_any_stage.predict(df.loc[df.is_alive], self.rng)

        # Determine the stage of the cancer for those who do have a cancer:
        if bc_status_any_stage.sum():
            sum_probs = sum(p['init_prop_bladder_cancer_stage'])
            if sum_probs > 0:
                prob_by_stage_of_cancer_if_cancer = [i/sum_probs for i in p['init_prop_bladder_cancer_stage']]
                assert (sum(prob_by_stage_of_cancer_if_cancer) - 1.0) < 1e-10
                df.loc[bc_status_any_stage, "bc_status"] = self.rng.choice(
                    [val for val in df.bc_status.cat.categories if val != 'none'],
                    size=bc_status_any_stage.sum(),
                    p=prob_by_stage_of_cancer_if_cancer
                )

        # -------------------- SYMPTOMS -----------
        # ----- Impose the symptom of random sample of those in each cancer stage to have the symptom of blood_urine:
        lm_init_blood_urine = LinearModel.multiplicative(
            Predictor(
                'bc_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when("none", 0.0)
            .when("tis_t1", p['init_prop_blood_urine_bladder_cancer_by_stage'][0])
            .when("t2p", p['init_prop_blood_urine_bladder_cancer_by_stage'][1])
            .when("metastatic", p['init_prop_blood_urine_bladder_cancer_by_stage'][2])
        )
        has_blood_urine_at_init = lm_init_blood_urine.predict(df.loc[df.is_alive], self.rng)
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=has_blood_urine_at_init.index[has_blood_urine_at_init].tolist(),
            symptom_string='blood_urine',
            add_or_remove='+',
            disease_module=self
        )

        # ----- Impose the symptom of random sample of those in each cancer stage to have the symptom of pelvic pain:
        lm_init_pelvic_pain = LinearModel.multiplicative(
            Predictor(
                'bc_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when("none", 0.0)
            .when("tis_t1", p['init_prop_pelvic_pain_bladder_cancer_by_stage'][0])
            .when("t2p", p['init_prop_pelvic_pain_bladder_cancer_by_stage'][1])
            .when("metastatic", p['init_prop_pelvic_pain_bladder_cancer_by_stage'][2])
        )
        has_pelvic_pain_at_init = lm_init_pelvic_pain.predict(df.loc[df.is_alive], self.rng)
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=has_pelvic_pain_at_init.index[has_pelvic_pain_at_init].tolist(),
            symptom_string='pelvic_pain',
            add_or_remove='+',
            disease_module=self
        )

        # -------------------- bc_date_diagnosis -----------
        lm_init_diagnosed = LinearModel.multiplicative(
            Predictor(
                'bc_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when("none", 0.0)
            .when("tis_t1", p['init_prop_with_blood_urine_diagnosed_bladder_cancer_by_stage'][0])
            .when("t2p", p['init_prop_with_blood_urine_diagnosed_bladder_cancer_by_stage'][1])
            .when("metastatic", p['init_prop_with_blood_urine_diagnosed_bladder_cancer_by_stage'][2])
        )
        ever_diagnosed = lm_init_diagnosed.predict(df.loc[df.is_alive], self.rng)

        # ensure that persons who have not ever had the symptom blood_urine are diagnosed:
        ever_diagnosed.loc[~has_blood_urine_at_init] = False

        # For those that have been diagnosed, set date of diagnosis to today's date
        df.loc[ever_diagnosed, "bc_date_diagnosis"] = self.sim.date

        # -------------------- bc_date_treatment -----------
        lm_init_treatment_for_those_diagnosed = LinearModel.multiplicative(
            Predictor(
                'bc_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when("none", 0.0)
            .when("tis_t1", p['init_prop_treatment_status_bladder_cancer'][0])
            .when("t2p", p['init_prop_treatment_status_bladder_cancer'][1])
            .when("metastatic", p['init_prop_treatment_status_bladder_cancer'][2])
        )
        treatment_initiated = lm_init_treatment_for_those_diagnosed.predict(df.loc[df.is_alive], self.rng)

        # prevent treatment having been initiated for anyone who is not yet diagnosed
        treatment_initiated.loc[pd.isnull(df.bc_date_diagnosis)] = False

        # assume that the stage at which treatment is begun is the stage the person is in now;
        # df.loc[treatment_initiated, "bc_stage_at_which_treatment_given"] = df.loc[treatment_initiated, "bc_status"]
        df.loc[treatment_initiated, "bc_stage_at_which_treatment_given"] = "t2p"

        # set date at which treatment began: same as diagnosis (NB. no HSI is established for this)
        df.loc[treatment_initiated, "bc_date_treatment"] = df.loc[treatment_initiated, "bc_date_diagnosis"]

        # -------------------- bc_date_palliative_care -----------
        in_metastatic_diagnosed = df.index[
            df.is_alive &
            (df.bc_status == 'metastatic') &
            ~pd.isnull(df.bc_date_diagnosis)
        ]

        select_for_care = self.rng.random_sample(size=len(in_metastatic_diagnosed)) < p['init_prob_palliative_care']
        select_for_care = in_metastatic_diagnosed[select_for_care]

        # set date of palliative care being initiated: same as diagnosis (NB. future HSI will be scheduled for this)
        df.loc[select_for_care, "bc_date_palliative_care"] = df.loc[select_for_care, "bc_date_diagnosis"]

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
        sim.schedule_event(BladderCancerLoggingEvent(self), sim.date + DateOffset(months=0))

        # ----- SCHEDULE MAIN POLLING EVENTS -----
        # Schedule main polling event to happen immediately
        sim.schedule_event(BladderCancerMainPollingEvent(self), sim.date + DateOffset(months=0))

        # ----- LINEAR MODELS -----
        # Define LinearModels for the progression of cancer, in each 3 month period
        # NB. The effect being produced is that treatment only has the effect for during the stage at which the
        # treatment was received.

        df = sim.population.props
        p = self.parameters
        lm = self.linear_models_for_progession_of_bc_status

        lm['tis_t1'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_tis_t1_bladder_cancer_none'],
            # todo: add in when schisto is in
            # Predictor('sh_infection_status').when('High-infection', p['rp_bladder_cancer_schisto_h']),
            Predictor('age_years', conditions_are_mutually_exclusive=True)
            .when('.between(30,49)', p['rp_bladder_cancer_age3049'])
            .when('.between(50,69)', p['rp_bladder_cancer_age5069'])
            .when('.between(70,120)', p['rp_bladder_cancer_agege70'])
            .when('.between(0,14)', 0.0),
            Predictor('li_tob').when(True, p['rr_tis_t1_bladder_cancer_none_tobacco']),
            # todo: add in when schisto module in master
            # Predictor('sh_').when(True, p['rr_tis_t1_bladder_cancer_none_ex_alc']),
            Predictor('bc_status').when('none', 1.0).otherwise(0.0)
        )

        lm['t2p'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_t2p_bladder_cancer_tis_t1'],
            Predictor('had_treatment_during_this_stage',
                      external=True).when(True, p['rr_t2p_bladder_cancer_undergone_curative_treatment']),
            Predictor('bc_status').when('tis_t1', 1.0)
                                  .otherwise(0.0)
        )

        lm['metastatic'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_metastatic_t2p_bladder_cancer'],
            Predictor('had_treatment_during_this_stage',
                      external=True).when(True, p['rr_metastatic_undergone_curative_treatment']),
            Predictor('bc_status').when('t2p', 1.0)
                                  .otherwise(0.0)
        )

        # Check that the dict labels are correct as these are used to set the value of bc_status
        assert set(lm).union({'none'}) == set(df.bc_status.cat.categories)

        # Linear Model for the onset of blood_urine, in each 3 month period
        self.lm_onset_blood_urine = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_blood_urine_tis_t1_bladder_cancer'],
            Predictor(
                'bc_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when('tis_t1', 1.0)
            .when('t2p', p['rr_blood_urine_t2p_bladder_cancer'])
            .when('metastatic', p['rr_blood_urine_metastatic_bladder_cancer'])
            .when('none', 0.0)
        )

        # Linear Model for the onset of pelvic_pain, in each 3 month period
        self.lm_onset_pelvic_pain = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_pelvic_pain_tis_t1_bladder_cancer'],
            Predictor(
                'bc_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when('tis_t1', 1.0)
            .when('t2p', p['rr_pelvic_pain_t2p_bladder_cancer'])
            .when('metastatic', p['rr_pelvic_pain_metastatic_bladder_cancer'])
            .when('none', 0.0)
        )

        # ----- DX TESTS -----
        # Create the diagnostic test representing the use of a cytoscope to diagnose bladder cancer
        # This properties of conditional on the test being done only to persons with the Symptom, 'blood_urine'.

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            cytoscopy_for_bladder_cancer_given_blood_urine=DxTest(
                property='bc_status',
                sensitivity=self.parameters['sensitivity_of_cytoscopy_for_bladder_cancer_blood_urine'],
                target_categories=["tis_t1", "t2p", "metastatic"]
            )
        )

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            cytoscopy_for_bladder_cancer_given_pelvic_pain=DxTest(
                property='bc_status',
                sensitivity=self.parameters['sensitivity_of_cytoscopy_for_bladder_cancer_pelvic_pain'],
                target_categories=["tis_t1", "t2p", "metastatic"]
            )
         )

        # ----- DISABILITY-WEIGHT -----
        if "HealthBurden" in self.sim.modules:
            health_burden = self.sim.modules["HealthBurden"]

            # For those with cancer (any stage prior to metastatic) and never treated
            self.daly_wts["tis_t1_t2p"] = health_burden.get_daly_weight(
                sequlae_code=550
                # todo: may need to consider reducing daly weight for early (tis_t1) as physical symptoms are unlikely
                # "Diagnosis and primary therapy phase of bladder cancer":
                # "Cancer, diagnosis and primary therapy ","has pain, nausea, fatigue, weight loss and high anxiety."
            )

            # For those with cancer (any stage prior to metastatic) and has been treated
            self.daly_wts["tis_t1_t2p_treated"] = health_burden.get_daly_weight(
                sequlae_code=547
                # "Controlled phase of bladder cancer,Generic uncomplicated disease":
                # "worry and daily medication,has a chronic disease that requires medication every day and causes some
                #  worry but minimal interference with daily activities".
            )

            # For those in metastatic: no palliative care
            self.daly_wts["metastatic"] = health_burden.get_daly_weight(
                sequlae_code=549
                # "Metastatic phase of esophageal cancer:
                # "Cancer, metastatic","has severe pain, extreme fatigue, weight loss and high anxiety."
            )

            # For those in metastatic: with palliative care
            self.daly_wts["metastatic_palliative_care"] = self.daly_wts["tis_t1_t2p"]
            # By assumption, we say that that the weight for those in metastatic with palliative care is the same as
            # that for those with earlier stage cancers. (this may be over-optimistic)

        # ----- HSI FOR PALLIATIVE CARE -----
        on_palliative_care_at_initiation = df.index[df.is_alive & ~pd.isnull(df.bc_date_palliative_care)]
        for person_id in on_palliative_care_at_initiation:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_BladderCancer_PalliativeCare(module=self, person_id=person_id),
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
        df.at[child_id, "bc_status"] = "none"
        df.at[child_id, "bc_date_diagnosis"] = pd.NaT
        df.at[child_id, "bc_date_treatment"] = pd.NaT
        df.at[child_id, "bc_stage_at_which_treatment_given"] = "none"
        df.at[child_id, "bc_date_palliative_care"] = pd.NaT
        df.at[child_id, "bc_date_death"] = pd.NaT

    def on_hsi_alert(self, person_id, treatment_id):
        pass

    def report_daly_values(self):
        # This must send back a dataframe that reports on the HealthStates for all individuals over the past month

        df = self.sim.population.props  # shortcut to population properties dataframe for alive persons

        disability_series_for_alive_persons = pd.Series(index=df.index[df.is_alive], data=0.0)

        # Assign daly_wt to those with cancer stages before metastatic and have either never been treated or
        # are no longer in the stage in which they were treated
        disability_series_for_alive_persons.loc[
            (
                (df.bc_status == "tis_t1") | (df.bc_status == "t2p")
            )
        ] = self.daly_wts['tis_t1_t2p']

        # Assign daly_wt to those with cancer stages before metastatic and who have been treated and
        # who are still in the stage in which they were treated.
        disability_series_for_alive_persons.loc[
            (
                ~pd.isnull(df.bc_date_treatment) &
                ((df.bc_status == "tis_t1") | (df.bc_status == "t2p")) &
                (df.bc_status == df.bc_stage_at_which_treatment_given)
            )
        ] = self.daly_wts['tis_t1_t2p_treated']

        # Assign daly_wt to those in metastatic cancer (who have not had palliative care)
        disability_series_for_alive_persons.loc[
            (df.bc_status == "metastatic") &
            (pd.isnull(df.bc_date_palliative_care))
        ] = self.daly_wts['metastatic']

        # Assign daly_wt to those in metastatic cancer, who have had palliative care
        disability_series_for_alive_persons.loc[
            (df.bc_status == "metastatic") &
            (~pd.isnull(df.bc_date_palliative_care))
        ] = self.daly_wts['metastatic_palliative_care']

        return disability_series_for_alive_persons


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------

class BladderCancerMainPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all bladder cancer properties for population:
    * Acquisition and progression of bladder Cancer
    * Symptom Development according to stage of bladder Cancer
    * Deaths from bladder Cancer for those in metastatic
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        # scheduled to run every 3 months: do not change as this is hard-wired into the values of all the parameters.

    def apply(self, population):
        df = population.props  # shortcut to dataframe
        m = self.module
        rng = m.rng

        # -------------------- ACQUISITION AND PROGRESSION OF CANCER (bc_status) -----------------------------------
        # determine if the person had a treatment during this stage of cancer (nb. treatment only has an effect on
        # reducing progression risk during the stage at which is received.
        had_treatment_during_this_stage = (
            df.is_alive & ~pd.isnull(df.bc_date_treatment) & (df.bc_status == df.bc_stage_at_which_treatment_given)
        )

        for stage, lm in m.linear_models_for_progession_of_bc_status.items():
            gets_new_stage = lm.predict(df.loc[df.is_alive], rng,
                                        had_treatment_during_this_stage=had_treatment_during_this_stage)
            df.loc[gets_new_stage[gets_new_stage].index, 'bc_status'] = stage

        # -------------------- UPDATING OF SYMPTOM OF blood_urine OVER TIME --------------------------------
        # Each time this event is called (event 3 months) individuals may develop the symptom of blood_urine.
        # Once the symptom is developed it never resolves naturally. It may trigger health-care-seeking behaviour.
        onset_blood_urine = m.lm_onset_blood_urine.predict(df.loc[df.is_alive], rng)
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=onset_blood_urine[onset_blood_urine].index.tolist(),
            symptom_string='blood_urine',
            add_or_remove='+',
            disease_module=m
        )

        # -------------------- UPDATING OF SYMPTOM OF PELVIC PAIN OVER TIME --------------------------------
        # Each time this event is called (event 3 months) individuals may develop the symptom of pelvic pain.
        # Once the symptom is developed it never resolves naturally. It may trigger health-care-seeking behaviour.
        onset_pelvic_pain = m.lm_onset_pelvic_pain.predict(df.loc[df.is_alive], rng)
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=onset_pelvic_pain[onset_pelvic_pain].index.tolist(),
            symptom_string='pelvic_pain',
            add_or_remove='+',
            disease_module=m
        )

        # -------------------- DEATH FROM bladder CANCER ---------------------------------------
        # There is a risk of death for those in metastatic only. Death is assumed to go instantly.
        metastatic_idx = df.index[df.is_alive & (df.bc_status == "metastatic")]
        selected_to_die = metastatic_idx[
            rng.random_sample(size=len(metastatic_idx)) < m.parameters['r_death_bladder_cancer']]

        for person_id in selected_to_die:
            self.sim.schedule_event(
                InstantaneousDeath(m, person_id, "BladderCancer"), self.sim.date
            )
            df.loc[selected_to_die, 'bc_date_death'] = self.sim.date


# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
# ---------------------------------------------------------------------------------------------------------

class HSI_BladderCancer_Investigation_Following_Blood_Urine(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_GenericFirstApptAtFacilityLevel1 following presentation for care with the symptom
    blood_urine.
    This event begins the investigation that may result in diagnosis of Bladder Cancer and the scheduling of
    treatment or palliative care.
    It is for people with the symptom blood_urine.
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "BladderCancer_Investigation"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

        # equipment: (ultrsound guided) biopsy, lab equipment for histology

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        # Ignore this event if the person is no longer alive:
        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # Check that this event has been called for someone with the symptom blood_urine
        assert 'blood_urine' in self.sim.modules['SymptomManager'].has_what(person_id)

        # If the person is already diagnosed, then take no action:
        if not pd.isnull(df.at[person_id, "bc_date_diagnosis"]):
            return hs.get_blank_appt_footprint()

        # Use a cytoscope to diagnose whether the person has bladder Cancer:
        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='cytoscopy_for_bladder_cancer_given_blood_urine',
            hsi_event=self
        )

        if dx_result:
            # record date of diagnosis:
            df.at[person_id, 'bc_date_diagnosis'] = self.sim.date

            # Check if is in metastatic:
            in_metastatic = df.at[person_id, 'bc_status'] == 'metastatic'

            # If diagnosis detects cancer, we assume classification as metastatic is accurate
            if not in_metastatic:
                # start treatment:
                hs.schedule_hsi_event(
                    hsi_event=HSI_BladderCancer_StartTreatment(
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
                    hsi_event=HSI_BladderCancer_PalliativeCare(
                        module=self.module,
                        person_id=person_id
                    ),
                    priority=0,
                    topen=self.sim.date,
                    tclose=None
                )


class HSI_BladderCancer_Investigation_Following_pelvic_pain(HSI_Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "BladderCancer_Investigation"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

        # equipment: (ultrsound guided) biopsy, lab equipment for histology

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        # Ignore this event if the person is no longer alive:
        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # Check that this event has been called for someone with the symptom pelvic_pain
        assert 'pelvic_pain' in self.sim.modules['SymptomManager'].has_what(person_id)

        # If the person is already diagnosed, then take no action:
        if not pd.isnull(df.at[person_id, "bc_date_diagnosis"]):
            return hs.get_blank_appt_footprint()

        # Use a cytoscope to diagnose whether the person has bladder Cancer:
        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='cytoscopy_for_bladder_cancer_given_pelvic_pain',
            hsi_event=self
        )

        if dx_result:
            # record date of diagnosis:
            df.at[person_id, 'bc_date_diagnosis'] = self.sim.date

            # Check if is in metastatic:
            in_metastatic = df.at[person_id, 'bc_status'] == 'metastatic'

            # If diagnosis detects cancer, we assume classification as metastatic is accurate
            if not in_metastatic:
                # start treatment:
                hs.schedule_hsi_event(
                    hsi_event=HSI_BladderCancer_StartTreatment(
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
                    hsi_event=HSI_BladderCancer_PalliativeCare(
                        module=self.module,
                        person_id=person_id
                    ),
                    priority=0,
                    topen=self.sim.date,
                    tclose=None
                )


class HSI_BladderCancer_StartTreatment(HSI_Event, IndividualScopeEventMixin):
    """
    Scheduled by HSI_bladderCancer_Investigation_Following_blood_urine or pelvic pain following a
    diagnosis of bladder Cancer using cytoscopy. It initiates the treatment of bladder Cancer.
    It is only for persons with a cancer that is not in metastatic and who have been diagnosed.
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "BladderCancer_Treatment"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'MajorSurg': 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 5})

        # equipment: standard equipment for surgery

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # If the status is metastatic, start palliative care (instead of treatment)
        if df.at[person_id, "bc_status"] == 'metastatic':
            logger.warning(key="warning", data="Cancer is metastatic - aborting HSI_BladderCancer_StartTreatment,"
                                               "scheduling HSI_BladderCancer_PalliativeCare")

            hs.schedule_hsi_event(
                hsi_event=HSI_BladderCancer_PalliativeCare(
                    module=self.module,
                    person_id=person_id,
                ),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )
            return self.make_appt_footprint({})

        # Check that the person has cancer, has been diagnosed and is not on treatment
        assert not df.at[person_id, "bc_status"] == 'none'
        assert not pd.isnull(df.at[person_id, "bc_date_diagnosis"])
        assert pd.isnull(df.at[person_id, "bc_date_treatment"])

        # Record date and stage of starting treatment
        df.at[person_id, "bc_date_treatment"] = self.sim.date
        df.at[person_id, "bc_stage_at_which_treatment_given"] = df.at[person_id, "bc_status"]

        # Schedule a post-treatment check for 12 months:
        hs.schedule_hsi_event(
            hsi_event=HSI_BladderCancer_PostTreatmentCheck(
                module=self.module,
                person_id=person_id,
            ),
            topen=self.sim.date + DateOffset(years=12),
            tclose=None,
            priority=0
        )


class HSI_BladderCancer_PostTreatmentCheck(HSI_Event, IndividualScopeEventMixin):
    """
    Scheduled by HSI_BladderCancer_StartTreatment and itself.
    It is only for those who have undergone treatment for Bladder Cancer.
    If the person has developed cancer to metastatic, the patient is initiated on palliative care; otherwise a further
    appointment is scheduled for one year.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "BladderCancer_Treatment"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'

        # I assume ultrasound (Ultrasound scanning machine) and biopsy

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # Check that the person is has cancer and is on treatment
        assert not df.at[person_id, "bc_status"] == 'none'
        assert not pd.isnull(df.at[person_id, "bc_date_diagnosis"])
        assert not pd.isnull(df.at[person_id, "bc_date_treatment"])

        if df.at[person_id, 'bc_status'] == 'metastatic':
            # If has progressed to metastatic, then start Palliative Care immediately:
            hs.schedule_hsi_event(
                hsi_event=HSI_BladderCancer_PalliativeCare(
                    module=self.module,
                    person_id=person_id
                ),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )

        else:
            # Schedule another HSI_BladderCancer_PostTreatmentCheck event in one month
            hs.schedule_hsi_event(
                hsi_event=HSI_BladderCancer_PostTreatmentCheck(
                    module=self.module,
                    person_id=person_id
                ),
                topen=self.sim.date + DateOffset(years=1),
                tclose=None,
                priority=0
            )


class HSI_BladderCancer_PalliativeCare(HSI_Event, IndividualScopeEventMixin):
    """
    This is the event for palliative care. It does not affect the patients progress but does affect the disability
     weight and takes resources from the healthsystem.
    This event is scheduled by either:
    * HSI_bladderCancer_Investigation_Following_blood_urine following a diagnosis of bladder Cancer at metastatic.
    * HSI_bladderCancer_PostTreatmentCheck following progression to metastatic during treatment.
    * Itself for the continuance of care.
    It is only for persons with a cancer in metastatic.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "BladderCancer_PalliativeCare"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
        self.ACCEPTED_FACILITY_LEVEL = '2'
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 15})

        # no equipment as far as I am aware

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # Check that the person is in metastatic
        assert df.at[person_id, "bc_status"] == 'metastatic'

        # Record the start of palliative care if this is first appointment
        if pd.isnull(df.at[person_id, "bc_date_palliative_care"]):
            df.at[person_id, "bc_date_palliative_care"] = self.sim.date

        # Schedule another instance of the event for one month
        hs.schedule_hsi_event(
            hsi_event=HSI_BladderCancer_PalliativeCare(
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

class BladderCancerLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """The only logging event for this module"""
    def __init__(self, module):
        """schedule logging to repeat every 1 month
        """
        self.repeat = 1
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
            f'total_{k}': v for k, v in df.loc[df.is_alive].bc_status.value_counts().items()})

        # Current counts, undiagnosed
        out.update({f'undiagnosed_{k}': v for k, v in df.loc[df.is_alive].loc[
            pd.isnull(df.bc_date_diagnosis), 'bc_status'].value_counts().items()})

        # Current counts, diagnosed
        out.update({f'diagnosed_{k}': v for k, v in df.loc[df.is_alive].loc[
            ~pd.isnull(df.bc_date_diagnosis), 'bc_status'].value_counts().items()})

        # Current counts, on treatment (excl. palliative care)
        out.update({f'treatment_{k}': v for k, v in df.loc[df.is_alive].loc[(~pd.isnull(
            df.bc_date_treatment) & pd.isnull(
            df.bc_date_palliative_care)), 'bc_status'].value_counts().items()})

        # Counts of those that have been diagnosed, started treatment or started palliative care since last logging
        # event:
        date_now = self.sim.date
        date_lastlog = self.sim.date - pd.DateOffset(months=self.repeat)

        out.update({
            'diagnosed_since_last_log': df.bc_date_diagnosis.between(date_lastlog, date_now).sum(),
            'treated_since_last_log': df.bc_date_treatment.between(date_lastlog, date_now).sum(),
            'palliative_since_last_log': df.bc_date_palliative_care.between(date_lastlog, date_now).sum(),
            'death_bladder_cancer_since_last_log': df.bc_date_death.between(date_lastlog, date_now).sum()
        })

        logger.info(key="summary_stats", data=out)
