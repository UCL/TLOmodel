"""
Oesophageal Cancer Disease Module

TODO:
* Consider adding palliative care : todo -- seems to be in there already!?
* Disability weights need to be updated?
* Needs to represent the the DxTest 'endoscopy_dysphagia_oes_cancer' requires use of an endoscope
* The age effect is very aggressive in the initiaisation: is that right?
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


class Oesophageal_Cancer(Module):
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
            Types.REAL, "disability weight for oesophageal cancer controlled phase - code 547"
        ),
        "daly_wt_oes_cancer_terminal": Parameter(
            Types.REAL, "disability weight for oesophageal cancer terminal - code 548"
        ),
        "daly_wt_oes_cancer_metastatic": Parameter(
            Types.REAL, "disability weight for oesophageal cancer metastatic - code 549"
        ),
        "daly_wt_oes_cancer_primary_therapy": Parameter(
            Types.REAL, "disability weight for oesophageal cancer primary therapy - code 550"
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
        ),

        "ca_date_palliative_care": Property(
            Types.DATE,
            "date of first receiving palliative care (pd.NaT is never had palliative care)"),   # todo - is is right that it never end?
    }

    # Symptom that this module will use: (dysphagia means problems swallowing)
    SYMPTOMS = {'dysphagia'}

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
            # get the DALY weight for oes cancer
            self.parameters["daly_wt_oes_dysp_diagnosed"] = 0.03
            self.parameters["daly_wt_oes_cancer_stage_1_3"] = self.sim.modules["HealthBurden"].get_daly_weight(
            sequlae_code=550
            )
            # todo: lower disability weight if palliative care (or higher if no palliative care)
            self.parameters["daly_wt_oes_cancer_stage4"] = self.sim.modules["HealthBurden"].get_daly_weight(
            sequlae_code=549
            )
            self.parameters["daly_wt_treated_oes_cancer"] = self.sim.modules["HealthBurden"].get_daly_weight(
            sequlae_code=547
            )


    def initialise_population(self, population):
        """Set our property values for the initial population."""
        df = population.props  # a shortcut to the data-frame storing data for individuals

        # -------------------- DEFAULTS ------------------------------------------------------------
        df.loc[df.is_alive, "ca_oesophagus"] = "none"
        df.loc[df.is_alive, "ca_oesophagus_any"] = False
        df.loc[df.is_alive, "ca_date_oes_cancer_diagnosis"] = pd.NaT
        df.loc[df.is_alive, "ca_date_treatment_oesophageal_cancer"] = pd.NaT
        df.loc[df.is_alive, "ca_stage_at_which_curative_treatment_was_begun"] = "never"
        df.loc[df.is_alive, "ca_date_palliative_care"] = pd.NaT

        # -------------------- ASSIGN VALUES OF OESOPHAGEAL DYSPLASIA/CANCER STATUS AT BASELINE -----------

        # Determine who has cancer at ANY cancer stage:
        lm_init_ca_oesophagus_any = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            sum(self.parameters['init_prop_oes_cancer_stage']),
            Predictor('li_ex_alc').when(True, self.parameters['rp_oes_cancer_ex_alc']),
            Predictor('li_tob').when(True, self.parameters['rp_oes_cancer_tobacco']),
            Predictor('age_years').apply(lambda x: ((x - 20) ** self.parameters['rp_oes_cancer_per_year_older']) if x > 20 else 0.0)
        )
        df.loc[df.is_alive, "ca_oesophagus_any"] = lm_init_ca_oesophagus_any.predict(df, self.rng)

        # Determine the stage of the cancer for those who do have a cancer:
        df.loc[(df.is_alive) & (df.ca_oesophagus_any), "ca_oesophagus"] = self.rng.choice(
            [val for val in self.PROPERTIES['ca_oesophagus'].categories if val != 'none'],
            df.loc[df.is_alive, "ca_oesophagus_any"].sum(),
            p=[p/sum(self.parameters['init_prop_oes_cancer_stage']) for p in self.parameters['init_prop_oes_cancer_stage']]
        )

        # Impose the symptom of random sample of those in each cancer stage to have the symptom of dysphagia:
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


        # Determine which persons with oes_cancer have been diagnosed (if diagnosed, diagnosis was in last 600 days)
        lm_init_diagnosed = LinearModel.multiplicative(
            Predictor('ca_oesophagus')
                .when("none", 0.0)
                .when("low_grade_dysplasia", self.parameters['init_prop_with_dysphagia_diagnosed_oes_cancer_by_stage'][0])
                .when("high_grade_dysplasia", self.parameters['init_prop_with_dysphagia_diagnosed_oes_cancer_by_stage'][1])
                .when("stage1", self.parameters['init_prop_with_dysphagia_diagnosed_oes_cancer_by_stage'][2])
                .when("stage2", self.parameters['init_prop_with_dysphagia_diagnosed_oes_cancer_by_stage'][3])
                .when("stage3", self.parameters['init_prop_with_dysphagia_diagnosed_oes_cancer_by_stage'][4])
                .when("stage4", self.parameters['init_prop_with_dysphagia_diagnosed_oes_cancer_by_stage'][5])
        )
        ever_diagnosed = lm_init_diagnosed.predict(df, self.rng)
        date_diagnosed = pd.Series(
            index=ever_diagnosed.index,
            data=[pd.NaT if (not x) else (self.sim.date - DateOffset(days=self.rng.random_integers(0, 600))) for x in ever_diagnosed.values]
        )
        # TODO- GOT TO HERE!
        df["ca_date_oes_cancer_diagnosis"]


        # -------------------- ASSIGN VALUES OESOPHAGEAL_CANCER_DIAGNOSED AT BASELINE --------------------------------

        def set_diagnosed(stage):
            """samples diagnosed status based on stage of cancer"""
            # get the positional offset of the stage (from definition of category in PROPERTIES)
            offset = df.ca_oesophagus.cat.categories.get_loc(stage) - 1
            # get the probability of diagnosis at this stage of cancer, given sy_dysphagia
            p_diagnosed = m.init_prop_with_dysphagia_diagnosed_oes_cancer_by_stage[offset]
            # randomly select some to have been diagnosed
            subset = df.is_alive & (df.ca_oesophagus == stage) & df.sy_dysphagia
            df.loc[subset, 'ca_oesophagus_diagnosed'] = rng.random_sample(size=subset.sum()) < p_diagnosed

        for stage in cancer_stages:
            set_diagnosed(stage)

        # -------------------- ASSIGN VALUES CA_OESOPHAGUS_CURATIVE_TREATMENT AT BASELINE -------------------

        def set_curative_treatment(stage):
            """sets the curative treatment flag for given stage of cancer"""
            idx = df.index[df.is_alive & (df.ca_oesophagus == stage) & df.ca_oesophagus_diagnosed]
            offset = df.ca_oesophagus.cat.categories.get_loc(stage) - 1
            p_curative_treatment = m.init_prop_treatment_status_oes_cancer[offset]
            selected = idx[p_curative_treatment > rng.random_sample(size=len(idx))]
            df.loc[selected, "ca_oesophagus_curative_treatment"] = stage

        # NOTE: excludes stage4 cancer
        for stage in cancer_stages[:-1]:
            set_curative_treatment(stage)

        # -------------------- ASSIGN VALUES CA_PALLIATIVE_CARE AT BASELINE -------------------

        idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage4')]
        selected = idx[m.init_prob_palliative_care > rng.random_sample(size=len(idx))]
        df.loc[selected, "ca_palliative_care"] = True


    def initialise_simulation(self, sim):
        """Add lifestyle events to the simulation
        """

        # Schedule main polling event to happen immediately
        sim.schedule_event(OesCancerEvent(self), sim.date + DateOffset(months=0))

        # Schedule logging event to happen immediately
        sim.schedule_event(OesCancerLoggingEvent(self), sim.date + DateOffset(months=0))

        # Create the diagnostic test representing the use of an endoscope to assess oes_cancer
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
        df.at[child_id, "ca_palliative_care"] = False
        df.at[child_id, "ca_oesophagus_diagnosed"] = False
        df.at[child_id, "ca_date_oes_cancer_diagnosis"] = pd.NaT
        df.at[child_id, "ca_oesophagus_curative_treatment"] = "never"
        df.at[child_id, "ca_incident_oes_cancer_diagnosis_this_3_month_period"] = False
        df.at[child_id, "ca_oesophagus_curative_treatment_requested"] = False
        df.at[child_id, "ca_date_treatment_oesophageal_cancer"] = pd.NaT
        df.at[child_id, "sy_dysphagia"] = False
        df.at[child_id, "ca_oesophageal_cancer_death"] = False

    def query_symptoms_now(self):
        # This is called by the health-care seeking module
        # All modules refresh the symptomology of persons at this time
        # And report it on the unified symptomology scale
        # Map the specific symptoms for this disease onto the unified coding scheme
        df = self.sim.population.props  # shortcut to population properties dataframe

        return pd.Series("1", index=df.index[df.is_alive])

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        pass

    def report_daly_values(self):
        # This must send back a dataframe that reports on the HealthStates for all individuals over
        # the past month
        #       logger.debug('This is Oesophageal Cancer reporting my health values')
        df = self.sim.population.props  # shortcut to population properties dataframe

        disability_series_for_alive_persons = pd.Series(index=df.index[df.is_alive],data=0.0)

        disability_series_for_alive_persons.loc[df.is_alive &
                                                (df.ca_oesophagus == "low_grade_dysplasia") &
                                                df.ca_oesophagus_diagnosed] = self.parameters['daly_wt_oes_cancer_stage_1_3']

        disability_series_for_alive_persons.loc[df.is_alive &
                                                (df.ca_oesophagus == "high_grade_dysplasia") &
                                                df.ca_oesophagus_diagnosed] = self.parameters['daly_wt_oes_cancer_stage_1_3']

        disability_series_for_alive_persons.loc[df.is_alive & (df.ca_oesophagus == "stage1")
                                                ] = self.parameters['daly_wt_oes_cancer_stage_1_3']

        disability_series_for_alive_persons.loc[df.is_alive & (df.ca_oesophagus == "stage2")
                                                ] = self.parameters['daly_wt_oes_cancer_stage_1_3']

        disability_series_for_alive_persons.loc[df.is_alive & (df.ca_oesophagus == "stage3")
                                                ] = self.parameters['daly_wt_oes_cancer_stage_1_3']

        disability_series_for_alive_persons.loc[df.is_alive & (df.ca_oesophagus == "stage4")
                                                ] = self.parameters['daly_wt_oes_cancer_stage4']

        disability_series_for_alive_persons.loc[df.is_alive
           & (df.ca_oesophagus_curative_treatment == 'low_grade_dysplasia') & (df.ca_oesophagus == 'low_grade_dysplasia')
        ] = self.parameters['daly_wt_treated_oes_cancer']

        disability_series_for_alive_persons.loc[df.is_alive
           & (df.ca_oesophagus_curative_treatment == 'high_grade_dysplasia') & (df.ca_oesophagus == 'high_grade_dysplasia')
        ] = self.parameters['daly_wt_treated_oes_cancer']

        disability_series_for_alive_persons.loc[df.is_alive
           & (df.ca_oesophagus_curative_treatment == 'stage1') & (df.ca_oesophagus == 'stage1')
        ] = self.parameters['daly_wt_treated_oes_cancer']

        disability_series_for_alive_persons.loc[df.is_alive
           & (df.ca_oesophagus_curative_treatment == 'stage2') & (df.ca_oesophagus == 'stage2')
        ] = self.parameters['daly_wt_treated_oes_cancer']

        disability_series_for_alive_persons.loc[df.is_alive
           & (df.ca_oesophagus_curative_treatment == 'stage3') & (df.ca_oesophagus == 'stage3')
        ] = self.parameters['daly_wt_treated_oes_cancer']

        return disability_series_for_alive_persons

class OesCancerEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all oesophagealcancer properties for population
    """

    def __init__(self, module):
        """schedule to run every 3 months
        note: if change this offset from 3 months need to consider code conditioning on age.years_exact
        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=3))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = m.rng

        # -------------------- UPDATING of CA-OESOPHAGUS OVER TIME -----------------------------------

        # create indexes of subgroups of people with different cancer statuses
        ca_oes_current_none_idx = df.index[df.is_alive &
                                           (df.ca_oesophagus == "none") &
                                           (df.age_years >= 20)]
        ca_oes_current_low_grade_dysp_idx = df.index[df.is_alive &
                                                     (df.ca_oesophagus == "low_grade_dysplasia")]
        ca_oes_current_high_grade_dysp_idx = df.index[df.is_alive &
                                                      (df.ca_oesophagus == "high_grade_dysplasia") &
                                                      (df.age_years >= 20)]
        ca_oes_current_stage1_idx = df.index[df.is_alive & (df.ca_oesophagus == "stage1")]
        ca_oes_current_stage2_idx = df.index[df.is_alive & (df.ca_oesophagus == "stage2")]
        ca_oes_current_stage3_idx = df.index[df.is_alive & (df.ca_oesophagus == "stage3")]

        # updating for people aged over 20 with current status 'none'
        # create series of the parameter r_low_grade_dysplasia_none which is the probability of low grade dysplasia for
        # men, age20, no excess alcohol, no tobacco
        eff_prob_low_grade_dysp = pd.Series(m.r_low_grade_dysplasia_none, index=ca_oes_current_none_idx)

        # update this "effective" probability by multiplyng by the rate ratio being female for the females
        eff_prob_low_grade_dysp.loc[df.is_alive &
                                    (df.ca_oesophagus == "none") &
                                    (df.age_years >= 20) &
                                    (df.sex == "F")] *= m.rr_low_grade_dysplasia_none_female

        # update this "effective" probability by multiplyng by the rate ratio being a tobacco user for the tobacco users
        eff_prob_low_grade_dysp.loc[df.is_alive &
                                    (df.ca_oesophagus == "none") &
                                    (df.age_years >= 20) &
                                    df.li_tob] *= m.rr_low_grade_dysplasia_none_tobacco

        # as above for excess alcohol users
        eff_prob_low_grade_dysp.loc[df.is_alive &
                                    (df.ca_oesophagus == "none") &
                                    (df.age_years >= 20) &
                                    df.li_ex_alc] *= m.rr_low_grade_dysplasia_none_ex_alc

        # create series which is rate ratio to be applied for each persons age - rate rate by age is continuous
        eff_prob_low_grade_dysp *= (
            m.rr_low_grade_dysplasia_none_per_year_older ** (df.loc[ca_oes_current_none_idx, 'age_years'] - 20)
        )

        # based on the random draw determine who develops low grade dysplasia in this update
        selected = ca_oes_current_none_idx[
            eff_prob_low_grade_dysp > rng.random_sample(size=len(eff_prob_low_grade_dysp))
            ]
        df.loc[selected, "ca_oesophagus"] = "low_grade_dysplasia"

        # updating for people aged over 20 with current stage to next stage
        def progress_stage(index, current_stage, next_stage, r_next_stage, rr_curative_treatment):

            eff_prob_next_stage = pd.Series(r_next_stage, index=index)

            eff_prob_next_stage.loc[df.is_alive &
                                   (df.ca_oesophagus == current_stage) &
                                   (df.age_years >= 20) &
                                    (df.ca_oesophagus_curative_treatment == current_stage)
                                    ] *= rr_curative_treatment
            selected = index[eff_prob_next_stage > rng.random_sample(size=len(eff_prob_next_stage))]
            df.loc[selected, 'ca_oesophagus'] = next_stage

        progress_stage(ca_oes_current_low_grade_dysp_idx,
                       'low_grade_dysplasia', 'high_grade_dysplasia',
                       m.r_high_grade_dysplasia_low_grade_dysp, m.rr_high_grade_dysp_undergone_curative_treatment)
        progress_stage(ca_oes_current_high_grade_dysp_idx,
                       'high_grade_dysplasia', 'stage1',
                       m.r_stage1_high_grade_dysp, m.rr_stage1_undergone_curative_treatment)
        progress_stage(ca_oes_current_stage1_idx,
                       'stage1', 'stage2',
                       m.r_stage2_stage1, m.rr_stage2_undergone_curative_treatment)
        progress_stage(ca_oes_current_stage2_idx,
                       'stage2', 'stage3',
                       m.r_stage3_stage2, m.rr_stage3_undergone_curative_treatment)
        progress_stage(ca_oes_current_stage3_idx,
                       'stage3', 'stage4',
                       m.r_stage4_stage3, m.rr_stage4_undergone_curative_treatment)

        # -------------------- UPDATING OF SY_DYSPHAGIA OVER TIME --------------------------------
        def update_dysphagia(current_stage, r_dysphagia):
            idx = df.index[df.is_alive &
                           (df.ca_oesophagus == current_stage)]
            selected = idx[r_dysphagia > rng.random_sample(size=len(idx))]
            df.loc[selected, 'sy_dysphagia'] = True

        update_dysphagia('low_grade_dysplasia', m.r_dysphagia_stage1 * m.rr_dysphagia_low_grade_dysp)
        update_dysphagia('high_grade_dysplasia', m.r_dysphagia_stage1 * m.rr_dysphagia_high_grade_dysp)
        update_dysphagia('stage1', m.r_dysphagia_stage1)
        update_dysphagia('stage2', m.r_dysphagia_stage1 * m.rr_dysphagia_stage2)
        update_dysphagia('stage3', m.r_dysphagia_stage1 * m.rr_dysphagia_stage3)
        update_dysphagia('stage4', m.r_dysphagia_stage1 * m.rr_dysphagia_stage4)


        # -------------------- UPDATING VALUES OF CA_OESOPHAGUS_DIAGNOSED  -------------------
        # update ca_oesophagus_diagnosed for those with dysphagia

        def update_diagnosis_status(p_present_dysphagia):

            idx2 = df.index[df.is_alive & df.sy_dysphagia & ~df.ca_oesophagus_diagnosed]
            selected2 = idx2[p_present_dysphagia > rng.random_sample(size=len(idx2))]

            # generate the HSI Events whereby persons present for care and get diagnosed
            for person_id in selected2:
                # For this person, determine when they will seek care (uniform distribution [0,91]days from now)
                date_seeking_care = self.sim.date + pd.DateOffset(days=int(rng.uniform(0, 91)))
                # For this person, create the HSI Event for their presentation for care
                hsi_present_for_care = HSI_Dysphagia_PresentForCare(self.module, person_id)
                # Enter this event to the HealthSystem
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    hsi_present_for_care, priority=0, topen=date_seeking_care, tclose=None
                )

        update_diagnosis_status(m.prob_present_dysphagia)

        # -------------------- UPDATING VALUES OF CA_OESOPHAGUS_CURATIVE_TREATMENT -------------------
        # update ca_oesophagus_curative_treatment for diagnosed, untreated people
        # this uses the approach descibed in detail above for updating diagnosis status
        def update_curative_treatment(current_stage):

        # todo: to disallow treatment for calibration purposes changed age cut off below from 20 to 200
            idx = df.index[df.is_alive &
                           (df.ca_oesophagus == current_stage) &
                           (df.age_years >= 200) &
                           df.ca_oesophagus_diagnosed &
                           (df.ca_oesophagus_curative_treatment == 'never')]

            # generate the HSI Events whereby diagnosed person gets treatment
            for person_id in idx:
                # For this person, determine when they will receive treatment (if there is health system capacity)
                date_oes_cancer_curative_treatment = self.sim.date + pd.DateOffset(days=int(rng.uniform(0, 91)))
                # For this person, create the HSI Event for their treatment
                hsi_oes_cancer_curative_treatment = HSI_OesCancer_StartTreatment(self.module, person_id)
                # Enter this event to the HealthSystem
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    hsi_oes_cancer_curative_treatment , priority=0, topen=date_oes_cancer_curative_treatment, tclose=None
                )

        update_curative_treatment('low_grade_dysplasia')
        update_curative_treatment('high_grade_dysplasia')
        update_curative_treatment('stage1')
        update_curative_treatment('stage2')
        update_curative_treatment('stage3')



        # -------------------- UPDATING VALUES OF CA_PALLIATIVE_CARE  -------------------

        def update_palliative_care_status(p):
            idx = df.index[df.is_alive & (df.ca_oesophagus == "stage4") & (~df.ca_palliative_care)]
            selected3 = idx[p > rng.random_sample(size=len(idx))]

            # generate the HSI Events whereby persons seek/need palliative care
            for person_id in selected3:
                # For this person, determine when they will seek/need palliative care
                # (uniform distribution [0,91]days from now)
                date_palliative_care = self.sim.date + pd.DateOffset(days=int(rng.uniform(0, 91)))
                # For this person, create the HSI Event for their palliative care
                hsi_cancer_palliative_care = HSI_Cancer_PalliativeCare(self.module, person_id)
                # Enter this event to the HealthSystem
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    hsi_cancer_palliative_care, priority=0, topen=date_palliative_care, tclose=None
                )

        update_palliative_care_status(m.rate_palliative_care_stage4)


        # -------------------- DEATH FROM OESOPHAGEAL CANCER ---------------------------------------
        stage4_idx = df.index[df.is_alive & (df.ca_oesophagus == "stage4")]

        selected_to_die = stage4_idx[m.r_death_oesoph_cancer > rng.random_sample(size=len(stage4_idx))]

        df.loc[selected_to_die, "ca_oesophageal_cancer_death"] = True

        for individual_id in selected_to_die:
            self.sim.schedule_event(
                demography.InstantaneousDeath(self.module, individual_id, "Oesophageal_cancer"), self.sim.date
            )



class HSI_Dysphagia_PresentForCare(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["Over5OPD"] = 1
        the_appt_footprint['MinorSurg'] = 1
        # todo: this appointment type for endoscopy and biopsy to be checked

        # This requires one out patient appt  # Define the necessary information for an HSI
        self.TREATMENT_ID = "assess_dysphagia"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1  # Enforces that this apppointment must happen at level 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        df.at[person_id, "ca_oesophagus_diagnosed"] = True
        df.at[person_id, "ca_date_oes_cancer_diagnosis"] = self.sim.date
        df.at[person_id, "ca_incident_oes_cancer_diagnosis_this_3_month_period"] = True

class HSI_OesCancer_StartTreatment(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        # todo: return to this below
        the_appt_footprint["Over5OPD"] = 1
        the_appt_footprint["MajorSurg"] = 3

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "start_treatment_oes_cancer"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 3  # Enforces that this apppointment must happen at level 3
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        df.at[person_id, "ca_oesophagus_curative_treatment"] = df.at[person_id, "ca_oesophagus"]
        df.at[person_id, "ca_date_treatment_oesophageal_cancer"] = self.sim.date

class HSI_Cancer_PalliativeCare(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()

            # todo: return to this below
            the_appt_footprint["Over5OPD"] = 1

            # Define the necessary information for an HSI
            self.TREATMENT_ID = "cancer_palliative_care"
            self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
            # todo: return to this below
            self.ACCEPTED_FACILITY_LEVEL = 3  # Enforces that this apppointment must happen at level 3
            self.ALERT_OTHER_DISEASES = []

        def apply(self, person_id, squeeze_factor):
            df = self.sim.population.props

            df.at[person_id, "ca_palliative_care"] = True


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

        dict_for_output = {
            # -- Current Status --

            # TODO: should be just count_values on ; ca_oesophagus - for those who have and have not been diagnosed


            'n_stage4': (df.is_alive & (df.ca_oesophagus == 'stage4')).sum(),

            'n_low_grade_dysplasia': (df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia')).sum(),

            'n_high_grade_dysplasia': (df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia')).sum(),

            'n_stage1_oc': (df.is_alive & (df.ca_oesophagus == 'stage1')).sum(),
            'n_stage2_oc': (df.is_alive & (df.ca_oesophagus == 'stage2')).sum(),
            'n_stage3_oc': (df.is_alive & (df.ca_oesophagus == 'stage3')).sum(),
            'n_stage4_oc': (df.is_alive & (df.ca_oesophagus == 'stage4')).sum(),

            'n_stage4_undiagnosed_oc': (
                    df.is_alive & (df.ca_oesophagus == 'stage4') & ~df.ca_oesophagus_diagnosed).sum(),

            'n_low_grade_dysplasia_diag':
                (df.is_alive & df.ca_oesophagus_diagnosed &
                 (df.ca_oesophagus == 'low_grade_dysplasia')).sum(),
            'n_high_grade_dysplasia_diag':
                (df.is_alive & df.ca_oesophagus_diagnosed
                 & (df.ca_oesophagus == 'high_grade_dysplasia')).sum(),

            'n_stage1_oc_diag': (df.is_alive & df.ca_oesophagus_diagnosed & (df.ca_oesophagus == 'stage1')).sum(),
            'n_stage2_oc_diag': (df.is_alive & df.ca_oesophagus_diagnosed & (df.ca_oesophagus == 'stage2')).sum(),
            'n_stage3_oc_diag': (df.is_alive & df.ca_oesophagus_diagnosed & (df.ca_oesophagus == 'stage3')).sum(),
            'n_stage4_oc_diag': (df.is_alive & df.ca_oesophagus_diagnosed & (df.ca_oesophagus == 'stage4')).sum(),

            'n_palliative_care': (df.is_alive & df.ca_palliative_care).sum(),

            # -- Events in last 3 months --  # todo: change this to date
            'n_incident_low_grade_dys_diag':
                (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
                 & (df.ca_oesophagus == 'low_grade_dysplasia')).sum(),

            'n_incident_high_grade_dys_diag':
                (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
                 & (df.ca_oesophagus == 'high_grade_dysplasia')).sum(),

            'n_incident_oc_stage1_diag':
                (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
                 & (df.ca_oesophagus == 'stage1')).sum(),

            'n_incident_oc_stage2_diag':
                (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
                 & (df.ca_oesophagus == 'stage2')).sum(),

            'n_incident_oc_stage3_diag':
                (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
                 & (df.ca_oesophagus == 'stage3')).sum(),

            'n_incident_oc_stage4_diag':
                (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
                 & (df.ca_oesophagus == 'stage4')).sum(),

            'n_incident_oes_cancer_diagnosis_any_stage':
                (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
                 & (df.ca_oesophague.isin(['stage1', 'stage2', 'stage3', 'stage4']))).sum(),

            'n_received_trt_this_period_low_grade_dysplasia':
                (df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia')
                 & df.ca_date_treatment_oesophageal_cancer == self.sim.date).sum(),
            'n_received_trt_this_period_high_grade_dysplasia':
                (df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia')
                 & df.ca_date_treatment_oesophageal_cancer == self.sim.date).sum(),
            'n_received_trt_this_period_stage1':
                (df.is_alive & (df.ca_oesophagus == 'stage1')
                 & df.ca_date_treatment_oesophageal_cancer == self.sim.date).sum(),
            'n_received_trt_this_period_stage2':
                (df.is_alive & (df.ca_oesophagus == 'stage2')
                 & df.ca_date_treatment_oesophageal_cancer == self.sim.date).sum(),
            'n_received_trt_this_period_stage3':
                (df.is_alive & (
                    df.ca_oesophagus == 'stage3') & df.ca_date_treatment_oesophageal_cancer == self.sim.date).sum()

        }


        # Check that no np.nan's are included in the output dict:
        for key, value in dict_for_output.items():
            dict_for_output[key] = value if not np.isnan(value) else 0

        logger.info('%s|summary_stats|%s', self.sim.date, dict_for_output)



        # TODO; work out what this is intendeing to do.
        # ---------------------------------------------------------------------------------------------------
        # # set ca_incident_oes_cancer_diagnosis_this_3_month_period back to False so is False for next 3 month
        # # period (this was not working when placed higher up
        #
        # idx = df.index[df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period]
        # df.loc[idx, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] = False
        #
        # idx = df.index[df.is_alive & df.ca_oesophageal_cancer_death]
        # df.loc[idx, "ca_oesophageal_cancer_death"] = False
        # ---------------------------------------------------------------------------------------------------
