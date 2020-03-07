"""
Oesophageal Cancer - module
Documentation: 04 - Methods Repository/Method_Oesophageal_Cancer.xlsx
"""
import logging
from pathlib import Path

import pandas as pd

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

    # todo: consider adding palliative care ;
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
        "r_death_oesoph_cancer": Parameter(
            Types.REAL,
            "probabilty per 3 months of death from oesophageal cancer mongst people with stage 4 oesophageal cancer",
        ),
        "r_curative_treatment_low_grade_dysp": Parameter(
            Types.REAL,
            "probabilty per 3 months of receiving medical treatment aimed at cure if have low grade "
            "dysplasia, given diagnosis (surgery, radiotherapy and/or chemotherapy",
        ),
        "rr_curative_treatment_high_grade_dysp": Parameter(
            Types.REAL,
            "relative rate of receiving medical treatment aimed at cure if have high grade "
            "dysplasia, given diagnosis (surgery, radiotherapy and/or chemotherapy",
        ),
        "rr_curative_treatment_stage1": Parameter(
            Types.REAL,
            "relative rate of receiving medical treatment aimed at cure if have stage1, "
            "given diagnosis (surgery, radiotherapy and/or chemotherapy",
        ),
        "rr_curative_treatment_stage2": Parameter(
            Types.REAL,
            "relative rate of receiving medical treatment aimed at cure if have stage2, "
            "given diagnosis (surgery, radiotherapy and/or chemotherapy",
        ),
        "rr_curative_treatment_stage3": Parameter(
            Types.REAL,
            "relative rate of receiving medical treatment aimed at cure if have stage3, "
            "given diagnosis (surgery, radiotherapy and/or chemotherapy",
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
        "init_prop_treatment_status_oes_cancer": Parameter(
            Types.LIST, "initial proportions of people with oesophageal dysplasia/cancer treated"
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

    # todo:  Dysplasia is assumed to persist indefinitely so if diagnosis is not made as a result of an initial
    # todo: presentation (diagnostic algorithm) there is a possibility  of re - presentation at a later  point at
    # todo: which there is again a certain probability of diagnosis.

    PROPERTIES = {
        "ca_oesophagus": Property(
            Types.CATEGORICAL,
            "oesophageal dysplasia / cancer stage: none, low_grade_dysplasia"
            "high_grade_dysplasia, stage1, stage2, stage3, stage4",
            categories=["none", "low_grade_dysplasia", "high_grade_dysplasia", "stage1", "stage2", "stage3", "stage4"],
        ),
        "ca_oesophagus_curative_treatment_requested": Property(
            Types.BOOL, "curative treatment requested of health care system this 3 month period"
        ),
        "ca_oesophagus_curative_treatment": Property(
            Types.CATEGORICAL,
            "oesophageal dysplasia / cancer stage at"
            "time of attempted curative treatment: never had treatment"
            "low grade dysplasia"
            "high grade dysplasia, stage 1, stage 2, stage 3",
            categories=["never", "low_grade_dysplasia", "high_grade_dysplasia", "stage1", "stage2", "stage3"],
        ),
        "sy_dysphagia": Property(Types.BOOL, "has dysphagia"),
        "ca_oesophagus_diagnosed": Property(Types.BOOL, "diagnosed with oesophageal dysplasia / cancer"),
        "ca_oesophageal_cancer_death": Property(Types.BOOL, "death from oesophageal cancer"),
        "ca_date_oes_cancer_diagnosis": Property(Types.DATE, "date incident oesophageal cancer"),
        "ca_date_treatment_oesophageal_cancer": Property(Types.DATE, "date of receiving attempted curative treatment")
    }

    # Symptom that this module will use
    # NB. The 'em_' prefix means that the onset of this symptom leads to an GenericEmergencyAppt
    SYMPTOMS = {'sy_dysphagia'}


    def read_parameters(self, data_folder):
        """Setup parameters used by the module, now including disability weights
        """
        # Update parameters from the resource dataframe
        dfd = pd.read_excel(
            Path(self.resourcefilepath) / "ResourceFile_Oesophageal_Cancer.xlsx", sheet_name="parameter_values"
        )
        self.load_parameters_from_dataframe(dfd)

        # Get DALY weight values:

        if "HealthBurden" in self.sim.modules.keys():
            # get the DALY weight for oes cancer
            self.parameters["daly_wt_oes_dysp_diagnosed"] = 0.03
            self.parameters["daly_wt_oes_cancer_stage_1_3"] = self.sim.modules["HealthBurden"].get_daly_weight(
            sequlae_code=550
            )
            self.parameters["daly_wt_oes_cancer_stage4"] = self.sim.modules["HealthBurden"].get_daly_weight(
            sequlae_code=549
            )
            self.parameters["daly_wt_treated_oes_cancer"] = self.sim.modules["HealthBurden"].get_daly_weight(
            sequlae_code=547
            )

            # Register this disease module with the health system
            self.sim.modules['HealthSystem'].register_disease_module(self)

    def initialise_population(self, population):
        """Set our property values for the initial population.
        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals
        m = self
        rng = m.rng

        cancer_stages = ["low_grade_dysplasia", "high_grade_dysplasia", "stage1", "stage2", "stage3", "stage4"]

        # -------------------- DEFAULTS ------------------------------------------------------------
        df.loc[df.is_alive, "ca_oesophagus"] = "none"
        df.loc[df.is_alive, "ca_oesophagus_diagnosed"] = False
        df.loc[df.is_alive, "ca_oesophagus_curative_treatment"] = "never"
        df.loc[df.is_alive, "ca_date_oes_cancer_diagnosis"] = pd.NaT
        df.loc[df.is_alive, "ca_oesophagus_curative_treatment_requested"] = False
        df.loc[df.is_alive, "ca_date_treatment_oesophageal_cancer"] = pd.NaT
        df.loc[df.is_alive, "sy_dysphagia"] = False

        # -------------------- ASSIGN VALUES OF OESOPHAGEAL DYSPLASIA/CANCER STATUS AT BASELINE -----------
        agege20_idx = df.index[(df.age_years >= 20) & df.is_alive]

        # create dataframe of the probabilities of ca_oesophagus status for 20 year old males, no ex alcohol, no tobacco
        p_oes_dys_can = pd.DataFrame(data=[m.init_prop_oes_cancer_stage], columns=cancer_stages, index=agege20_idx)

        # create probabilities of oes dysplasia and oe cancer for all over age 20
        p_oes_dys_can.loc[(df.sex == "F") & (df.age_years >= 20) & df.is_alive] *= m.rp_oes_cancer_female
        p_oes_dys_can.loc[df.li_ex_alc & (df.age_years >= 20) & df.is_alive] *= m.rp_oes_cancer_ex_alc
        p_oes_dys_can.loc[df.li_tob & (df.age_years >= 20) & df.is_alive] *= m.rp_oes_cancer_tobacco

        # apply multiplier
        p_oes_dys_can_age_muliplier = m.rp_oes_cancer_per_year_older ** (df.loc[agege20_idx, 'age_years'] - 20)
        p_oes_dys_can = p_oes_dys_can.multiply(p_oes_dys_can_age_muliplier, axis="index")

        # add column for the probability of no cancer at start of dataframe
        p_oes_dys_can.insert(0, 'none', 1 - p_oes_dys_can.sum(axis=1))

        # check the categories match and line up
        assert df.ca_oesophagus.cat.categories.all() == p_oes_dys_can.columns.all()

        # for each row, make a choice
        stage = p_oes_dys_can.apply(lambda p_oes: rng.choice(p_oes_dys_can.columns, p=p_oes), axis=1)

        # set for those that have cancer
        df.loc[stage.index[stage != 'none'], 'ca_oesophagus'] = stage[stage != 'none']

        # -------------------- ASSIGN VALUES SY_DYSPHAGIA AT BASELINE --------------------------------

        def set_dysphagia(stage):
            """samples dysphagia status based on stage of cancer"""
            # get the positional offset of the stage (from definition of category in PROPERTIES)
            offset = df.ca_oesophagus.cat.categories.get_loc(stage) - 1
            # get the probability of dysphagia  this stage of cancer
            p_dysphagia = m.init_prop_dysphagia_oes_cancer_by_stage[offset]
            # randomly select some to have been diagnosed
            subset = df.is_alive & (df.ca_oesophagus == stage)
            df.loc[subset, 'sy_dysphagia'] = rng.random_sample(size=subset.sum()) < p_dysphagia

        for stage in cancer_stages:
            set_dysphagia(stage)

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

    def initialise_simulation(self, sim):
        """Add lifestyle events to the simulation
        """
        # start simulation immediately - so above values are updated immediately
        event = OesCancerEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=0))
        event = OesCancerLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=0))

        # Create the diagnostic representing the assessment for whether a person with dysphagia is diagnosed with
        # oes cancer
        # assume specificity = 100%
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_dysphagia=DxTest(
                property='ca_oesophagus',
                sensitivity=self.parameters['sensitivity_of_assessment_of_oes_cancer_with_dysphagia'],
            )
        )

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props
        df.at[child_id, "ca_oesophagus"] = "none"
        df.at[child_id, "ca_oesophagus_diagnosed"] = False
        df.at[child_id, "ca_date_oes_cancer_diagnosis"] = pd_NaT
        df.at[child_id, "ca_oesophagus_curative_treatment"] = "never"
        df.at[child_id, "ca_incident_oes_cancer_diagnosis_this_3_month_period"] = False
        df.at[child_id, "ca_oesophagus_curative_treatment_requested"] = False
        df.at[child_id, "ca_date_treatment_oesophageal_cancer"] = pd.NaT
        df.at[child_id, "sy_dysphagia"] = False

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
        logger.debug(
            "This is oesophageal_cancer, being alerted about a health system interaction person %d for: %s",
            person_id,
            treatment_id,
        )
        pass

    def report_daly_values(self):
        # This must send back a dataframe that reports on the HealthStates for all individuals over
        # the past month
        #       logger.debug('This is Oesophageal Cancer reporting my health values')
        df = self.sim.population.props  # shortcut to population properties dataframe

        disability_series_for_alive_persons = pd.Series(index=df.index[df.is_alive],data=0.0)

        disability_series_for_alive_persons.loc[df.is_alive &
                                                ((df.ca_oesophagus == "low_grade_dysplasia") or
                                                (df.ca_oesophagus == "high_grade_dysplasia")) &
                                                df.ca.oesophagus_diagnosed] = 'daly_wt_oes_dysp_diagnosed'

        disability_series_for_alive_persons.loc[df.is_alive &
                                                ((df.ca_oesophagus == "stage1") or
                                                (df.ca_oesophagus == "stage2") or (df.ca_oesophagus == "stage3"))
                                                ] = 'daly_wt_oes_cancer_stage_1_3'

        disability_series_for_alive_persons.loc[df.is_alive & (df.ca_oesophagus == "stage4")
                                                ] = 'daly_wt_oes_cancer_stage4'

        disability_series_for_alive_persons.loc[df.is_alive
           & (df.ca_oesophagus_curative_treatment == 'low_grade_dysplasia') & (df.ca_oesophagus == 'low_grade_dysplasia')
        ] = 'daly_wt_treated_oes_cancer'

        disability_series_for_alive_persons.loc[df.is_alive
           & (df.ca_oesophagus_curative_treatment == 'high_grade_dysplasia') & (df.ca_oesophagus == 'high_grade_dysplasia')
        ] = 'daly_wt_treated_oes_cancer'

        disability_series_for_alive_persons.loc[df.is_alive
           & (df.ca_oesophagus_curative_treatment == 'stage1') & (df.ca_oesophagus == 'stage1')
        ] = 'daly_wt_treated_oes_cancer'

        disability_series_for_alive_persons.loc[df.is_alive
           & (df.ca_oesophagus_curative_treatment == 'stage2') & (df.ca_oesophagus == 'stage2')
        ] = 'daly_wt_treated_oes_cancer'

        disability_series_for_alive_persons.loc[df.is_alive
           & (df.ca_oesophagus_curative_treatment == 'stage3') & (df.ca_oesophagus == 'stage3')
        ] = 'daly_wt_treated_oes_cancer'

        return disability_series_for_alive_persons

    def do_when_dysphagia(self, person_id, hsi_event):
        """
        This is called by a generic HSI event when dysphagia is present
        :param person_id:
        :param hsi_event: The HSI event that has called this event
        :return:
        """
        # Assess for oes cancer
        if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='assess_oes_cancer_dysphagia',
                                                                   hsi_event=hsi_event
                                                                   ):
            self.sim.population.props.at[person_id, 'ca_oesophagus_diagnosed'] = True

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

        df.loc[df.is_alive, "ca_incident_oes_cancer_diagnosis_this_3_month_period"] = False

        # -------------------- UPDATING of CA-OESOPHAGUS OVER TIME -----------------------------------

        # create indexes of subgroups of people with different cancert statuses
        ca_oes_current_none_idx = df.index[df.is_alive &
                                           (df.ca_oesophagus == "none") &
                                           (df.age_years >= 20)]
        ca_oes_current_low_grade_dysp_idx = df.index[df.is_alive &
                                                     (df.ca_oesophagus == "low_grade_dysplasia") &
                                                     (df.age_years >= 20)]
        ca_oes_current_high_grade_dysp_idx = df.index[df.is_alive &
                                                      (df.ca_oesophagus == "high_grade_dysplasia") &
                                                      (df.age_years >= 20)]
        ca_oes_current_stage1_idx = df.index[df.is_alive & (df.ca_oesophagus == "stage1") & (df.age_years >= 20)]
        ca_oes_current_stage2_idx = df.index[df.is_alive & (df.ca_oesophagus == "stage2") & (df.age_years >= 20)]
        ca_oes_current_stage3_idx = df.index[df.is_alive & (df.ca_oesophagus == "stage3") & (df.age_years >= 20)]

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
            df.loc[selected, 'ca_date_oes_cancer_diagnosis'] = self.sim.date

        update_dysphagia('low_grade_dysphagia', m.r_dysphagia_stage1 * m.rr_dysphagia_low_grade_dysp)
        update_dysphagia('high_grade_dysphagia', m.r_dysphagia_stage1 * m.rr_dysphagia_high_grade_dysp)
        update_dysphagia('stage1', m.r_dysphagia_stage1)
        update_dysphagia('stage2', m.r_dysphagia_stage1 * m.rr_dysphagia_stage2)
        update_dysphagia('stage3', m.r_dysphagia_stage1 * m.rr_dysphagia_stage3)
        update_dysphagia('stage4', m.r_dysphagia_stage1 * m.rr_dysphagia_stage4)

        # -------------------- UPDATING VALUES OF CA_OESOPHAGUS_CURATIVE_TREATMENT -------------------
        # update ca_oesophagus_curative_treatment for diagnosed, untreated people with low grade dysplasia w
        # this uses the approach descibed in detail above for updating diagosis status

        def update_curative_treatment(current_stage, p_curative_treatment):
            idx = df.index[df.is_alive &
                           (df.ca_oesophagus == current_stage) &
                           (df.age_years >= 20) &
                           df.ca_oesophagus_diagnosed &
                           (df.ca_oesophagus_curative_treatment == 'never')]
            selected = idx[p_curative_treatment > rng.random_sample(size=len(idx))]
            df.loc[selected, 'ca_oesophagus_curative_treatment_requested'] = True

            # generate the HSI Events whereby persons present for care and get treatment
            for person_id in idx:
                # For this person, determine when they will seek care (uniform distibition [0,91]days from now)
                date_seeking_care = self.sim.date + pd.DateOffset(days=int(rng.uniform(0, 91)))
                # For this person, create the HSI Event for their presentation for care
                hsi_present_for_care = HSI_OesCancer_StartTreatmentLowGradeOesDysplasia(self.module, person_id)
                # Enter this event to the HealthSystem
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    hsi_present_for_care, priority=0, topen=date_seeking_care, tclose=None
                )

        update_curative_treatment('low_grade_dysplasia', m.r_curative_treatment_low_grade_dysp)
        update_curative_treatment('high_grade_dysplasia',
                                  m.r_curative_treatment_low_grade_dysp * m.rr_curative_treatment_high_grade_dysp)
        update_curative_treatment('stage1', m.r_curative_treatment_low_grade_dysp * m.rr_curative_treatment_stage1)
        update_curative_treatment('stage2', m.r_curative_treatment_low_grade_dysp * m.rr_curative_treatment_stage2)
        update_curative_treatment('stage3', m.r_curative_treatment_low_grade_dysp * m.rr_curative_treatment_stage3)

        # -------------------- DEATH FROM OESOPHAGEAL CANCER ---------------------------------------
        stage4_idx = df.index[df.is_alive & (df.ca_oesophagus == "stage4")]
        selected_to_die = stage4_idx[m.r_death_oesoph_cancer > rng.random_sample(size=len(stage4_idx))]
        for individual_id in selected_to_die:
            self.sim.schedule_event(
                demography.InstantaneousDeath(self.module, individual_id, "Oesophageal_cancer"), self.sim.date
            )



class HSI_Dysphagia_PresentForCare(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["Over5OPD"] = 1  # This requires one out patient appt
        # Define the necessary information for an HSI
        self.TREATMENT_ID = "assess_dysphagia_for_oes_cancer"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1  # Enforces that this apppointment must happen at level 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        df.at[person_id, "ca_oesophagus_diagnosed"] = True
        df.at[person_id, "ca_date_oes_cancer_diagnosis"] = self.sim.date


class HSI_OesCancer_StartTreatment(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        # todo this below will change to another type of appointment
        the_appt_footprint["Over5OPD"] = 1  # This requires one out patient appt
        # Define the necessary information for an HSI
        self.TREATMENT_ID = "start_treatment_oes_cancer"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1  # Enforces that this apppointment must happen at level 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        df.at[person_id, "ca_oesophagus_curative_treatment"] = df.ca_oesophagus
        df.at[person_id, "ca_date_treatment_oesophageal_cancer"] = self.sim.date


    #todo: HSIs to add
""""
Clinic appointment: (present with symptoms but) failure to diagnose 
Clinic appointment: diagnosis 

Attempt at curative treatment – pre surgery
Attempt at curative treatment – surgery
Attempt at curative treatment – chemotherapy clinic appointment
Attempt at curative treatment – radiotherapy clinic appointment
Attempt at curative treatment – initiation of endocrine treatment
Clinic appointment: monitoring - no new action 

Clinic appointment: initiate palliative care

Pharmacy: drug pick up 

"""


class OesCancerLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Handles logging"""

    def __init__(self, module):
        """schedule logging to repeat every 3 months
        """
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        # get some summary statistics
        df = population.props
        # n_alive = df.is_alive.sum()
        # n_alive_ge20 = (df.is_alive & (df.age_years >= 20)).sum()
        # n_incident_low_grade_dys_diag = (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
        #                             & (df.ca_oesophagus == 'low_grade_dysplasia')).sum()
        # n_incident_high_grade_dys_diag = (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
        #                             & (df.ca_oesophagus == 'high_grade_dysplasia')).sum()
        # n_incident_oc_stage1_diag = (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
        #                             & (df.ca_oesophagus == 'stage1')).sum()
        # n_incident_oc_stage2_diag = (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
        #                             & (df.ca_oesophagus == 'stage2')).sum()
        # n_incident_oc_stage3_diag = (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
        #                             & (df.ca_oesophagus == 'stage3')).sum()
        # n_incident_oc_stage4_diag = (df.is_alive & df.ca_incident_oes_cancer_diagnosis_this_3_month_period
        #                             & (df.ca_oesophagus == 'stage4')).sum()
        # n_incident_oes_cancer_diagnosis = n_incident_oc_stage1_diag \
        #                                   + n_incident_oc_stage2_diag + n_incident_oc_stage3_diag + \
        #                                   n_incident_oc_stage4_diag
        # n_low_grade_dysplasia = (df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia')).sum()
        # n_high_grade_dysplasia = (df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia')).sum()
        # n_stage1_oc = (df.is_alive & (df.ca_oesophagus == 'stage1')).sum()
        # n_stage2_oc = (df.is_alive & (df.ca_oesophagus == 'stage2')).sum()
        # n_stage3_oc = (df.is_alive & (df.ca_oesophagus == 'stage3')).sum()
        # n_stage4_oc = (df.is_alive & (df.ca_oesophagus == 'stage4')).sum()
        # n_stage4_undiagnosed_oc = (df.is_alive & (df.ca_oesophagus == 'stage4') & ~df.ca_oesophagus_diagnosed).sum()
        # n_low_grade_dysplasia_diag = (df.is_alive & df.ca_oesophagus_diagnosed &
        # (df.ca_oesophagus == 'low_grade_dysplasia')).sum()
        # n_high_grade_dysplasia_diag = (df.is_alive & df.ca_oesophagus_diagnosed
        # & (df.ca_oesophagus == 'high_grade_dysplasia')).sum()
        # n_stage1_oc_diag = (df.is_alive & df.ca_oesophagus_diagnosed & (df.ca_oesophagus == 'stage1')).sum()
        # n_stage2_oc_diag = (df.is_alive & df.ca_oesophagus_diagnosed & (df.ca_oesophagus == 'stage2')).sum()
        # n_stage3_oc_diag = (df.is_alive & df.ca_oesophagus_diagnosed & (df.ca_oesophagus == 'stage3')).sum()
        # n_stage4_oc_diag = (df.is_alive & df.ca_oesophagus_diagnosed & (df.ca_oesophagus == 'stage4')).sum()
        # n_received_trt_this_period_low_grade_dysplasia = (df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia')
        #                                     & df.ca_date_treatment_oesophageal_cancer == self.sim.date).sum()
        # n_received_trt_this_period_high_grade_dysplasia = (df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia')
        #                                     & df.ca_date_treatment_oesophageal_cancer == self.sim.date).sum()
        # n_received_trt_this_period_stage1 = (df.is_alive & (df.ca_oesophagus == 'stage1')
        #                                     & df.ca_date_treatment_oesophageal_cancer == self.sim.date).sum()
        # n_received_trt_this_period_stage2 = (df.is_alive & (df.ca_oesophagus == 'stage2')
        #                                     & df.ca_date_treatment_oesophageal_cancer == self.sim.date).sum()
        # n_received_trt_this_period_stage3 = (df.is_alive &
        # (df.ca_oesophagus == 'stage3') & df.ca_date_treatment_oesophageal_cancer == self.sim.date).sum()
        # n_oc_death = df.ca_oesophageal_cancer_death.sum()
        # cum_deaths = (~df.is_alive).sum()
        # logger.info('%s| n_alive_ge20 |%s| n_incident_oes_cancer_diagnosis|%s| n_incident_low_grade_dys_diag|%s| '
        #             'n_incident_high_grade_dys_diag|%s| n_incident_oc_stage1_diag|%s| '
        #             'n_incident_oc_stage2_diag|%s| n_incident_oc_stage3_diag|%s|n_incident_oc_stage4_diag|%s| '
        #             'n_low_grade_dysplasia|%s| n_high_grade_dysplasia|%s|  n_stage1_oc|%s| n_stage2_oc|%s| '
        #             'n_stage3_oc|%s| n_stage4_oc|%s| n_low_grade_dysplasia_diag|%s| n_high_grade_dysplasia_diag'
        #             '|%s| n_stage1_oc_diag|%s| n_stage2_oc_diag|%s| n_stage3_oc_diag|%s| n_stage4_oc_diag |%s|'
        #             'n_received_trt_this_period_low_grade_dysplasia |%s| '
        #             'n_received_trt_this_period_high_grade_dysplasia'
        #             '|%s| n_received_trt_this_period_stage1|%s| n_received_trt_this_period_stage2|%s|'
        #             'n_received_trt_this_period_stage3|%s|n_stage4_undiagnosed_oc|%s| cum_deaths |%s |n_alive|%s|'
        #             'n_oc_death|%s',
        #             self.sim.date, n_alive_ge20, n_incident_oes_cancer_diagnosis, n_incident_low_grade_dys_diag,
        #             n_incident_high_grade_dys_diag, n_incident_oc_stage1_diag,
        #             n_incident_oc_stage2_diag, n_incident_oc_stage3_diag,n_incident_oc_stage4_diag,
        #             n_low_grade_dysplasia, n_high_grade_dysplasia,  n_stage1_oc, n_stage2_oc,
        #             n_stage3_oc, n_stage4_oc, n_low_grade_dysplasia_diag, n_high_grade_dysplasia_diag,
        #             n_stage1_oc_diag, n_stage2_oc_diag, n_stage3_oc_diag, n_stage4_oc_diag,
        #             n_received_trt_this_period_low_grade_dysplasia, n_received_trt_this_period_high_grade_dysplasia,
        #             n_received_trt_this_period_stage1, n_received_trt_this_period_stage2,
        #             n_received_trt_this_period_stage3, n_stage4_undiagnosed_oc, cum_deaths, n_alive, n_oc_death
        #             )

        logger.info("%s|person_one|%s", self.sim.date, df.loc[0].to_dict())
