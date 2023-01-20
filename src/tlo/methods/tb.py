"""
    This module schedules TB infection and natural history
    It schedules TB treatment and follow-up appointments along with preventive therapy
    for eligible people (HIV+ and paediatric contacts of active TB cases
"""

import os

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, hiv
from tlo.methods.causes import Cause
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HealthSystemChangeParameters, HSI_Event
from tlo.methods.symptommanager import Symptom
from tlo.util import random_date

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Tb(Module):
    """Set up the baseline population with TB prevalence"""

    def __init__(self, name=None, resourcefilepath=None, run_with_checks=False):
        super().__init__(name)

        self.resourcefilepath = resourcefilepath
        self.daly_wts = dict()
        self.lm = dict()
        self.footprints_for_consumables_required = dict()
        self.symptom_list = {"fever", "respiratory_symptoms", "fatigue", "night_sweats"}
        self.district_list = list()
        self.item_codes_for_consumables_required = dict()

        assert isinstance(run_with_checks, bool)
        self.run_with_checks = run_with_checks

        # tb outputs needed for calibration/
        keys = ["date",
                "num_new_active_tb",
                "tbPrevLatent"
                ]
        # initialise empty dict with set keys
        self.tb_outputs = {k: [] for k in keys}

    INIT_DEPENDENCIES = {"Demography", "HealthSystem", "Lifestyle", "SymptomManager", "Epi"}

    OPTIONAL_INIT_DEPENDENCIES = {"HealthBurden", "Hiv"}

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN,
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        "TB": Cause(gbd_causes="Tuberculosis", label="non_AIDS_TB"),
        "AIDS_TB": Cause(gbd_causes="HIV/AIDS", label="AIDS"),
    }

    CAUSES_OF_DISABILITY = {
        "TB": Cause(gbd_causes="Tuberculosis", label="non_AIDS_TB"),
    }

    # Declaration of the specific symptoms that this module will use
    SYMPTOMS = {"fatigue", "night_sweats"}

    PROPERTIES = {
        # ------------------ natural history ------------------ #
        "tb_inf": Property(
            Types.CATEGORICAL,
            categories=[
                "uninfected",
                "latent",
                "active",
            ],
            description="tb status",
        ),
        "tb_strain": Property(
            Types.CATEGORICAL,
            categories=[
                "none",
                "ds",
                "mdr",
            ],
            description="tb strain: drug-susceptible (ds) or multi-drug resistant (mdr)",
        ),
        "tb_date_latent": Property(
            Types.DATE, "Date acquired tb infection (latent stage)"
        ),
        "tb_scheduled_date_active": Property(
            Types.DATE, "Date active tb is scheduled to start"
        ),
        "tb_date_active": Property(Types.DATE, "Date active tb started"),
        "tb_smear": Property(
            Types.BOOL,
            "smear positivity with active infection: False=negative, True=positive",
        ),
        # ------------------ testing status ------------------ #
        # todo
        "tb_date_tested": Property(Types.DATE, "Date of last tb test"),
        "tb_diagnosed": Property(
            Types.BOOL, "person has current diagnosis of active tb"
        ),
        "tb_date_diagnosed": Property(Types.DATE, "date most recent tb diagnosis"),
        "tb_diagnosed_mdr": Property(
            Types.BOOL, "person has current diagnosis of active mdr-tb"
        ),
        # ------------------ treatment status ------------------ #
        "tb_on_treatment": Property(Types.BOOL, "on tb treatment regimen"),
        "tb_date_treated": Property(
            Types.DATE, "date most recent tb treatment started"
        ),
        "tb_treatment_regimen": Property(
            Types.CATEGORICAL,
            categories=[
                "none",
                "tb_tx_adult",
                "tb_tx_child",
                "tb_tx_child_shorter",
                "tb_retx_adult",
                "tb_retx_child",
                "tb_mdrtx"
            ],
            description="current tb treatment regimen",
        ),
        "tb_ever_treated": Property(Types.BOOL, "if ever treated for active tb"),
        "tb_treatment_failure": Property(Types.BOOL, "failed first line tb treatment"),
        "tb_treated_mdr": Property(Types.BOOL, "on tb treatment MDR regimen"),
        "tb_date_treated_mdr": Property(Types.DATE, "date tb MDR treatment started"),
        "tb_on_ipt": Property(Types.BOOL, "if currently on ipt"),
        "tb_date_ipt": Property(Types.DATE, "date ipt started"),
    }

    PARAMETERS = {
        "beta": Parameter(
            Types.REAL,
            "transmission rate for TB",
        ),
        "importation_rate_ds": Parameter(
            Types.REAL,
            "monthly rate of importation of ds-tb",
        ),
        "importation_rate_mdr": Parameter(
            Types.REAL,
            "monthly rate of importation of ds-tb",
        ),
        # ------------------ baseline population ------------------ #
        "prop_mdr2010": Parameter(
            Types.REAL,
            "Proportion of active tb cases with multidrug resistance in 2010",
        ),
        # ------------------ workbooks ------------------ #
        "who_incidence_estimates": Parameter(
            Types.REAL, "WHO estimated active TB incidence per 100,000 population"
        ),
        "followup_times": Parameter(
            Types.DATA_FRAME,
            "times(weeks) tb treatment monitoring required after tx start",
        ),
        "tb_high_risk_distr": Parameter(Types.LIST, "list of ten high-risk districts"),
        "ipt_coverage": Parameter(
            Types.DATA_FRAME,
            "national estimates of coverage of IPT in PLHIV and paediatric contacts",
        ),
        # ------------------ natural history ------------------ #
        "incidence_active_tb_2010": Parameter(
            Types.REAL, "incidence of active tb in 2010 in all ages"
        ),
        "rr_tb_child": Parameter(
            Types.REAL, "relative risk of tb infection if under 16 years of age"
        ),
        "rr_bcg_inf": Parameter(
            Types.REAL, "relative risk of tb infection with bcg vaccination"
        ),
        "monthly_prob_relapse_tx_complete": Parameter(
            Types.REAL, "monthly probability of relapse once treatment complete"
        ),
        "monthly_prob_relapse_tx_incomplete": Parameter(
            Types.REAL, "monthly probability of relapse if treatment incomplete"
        ),
        "monthly_prob_relapse_2yrs": Parameter(
            Types.REAL,
            "monthly probability of relapse 2 years after treatment complete",
        ),
        "rr_relapse_hiv": Parameter(
            Types.REAL, "relative risk of relapse for HIV-positive people"
        ),
        # ------------------ active disease ------------------ #
        "duration_active_disease_years": Parameter(
            Types.REAL, "duration of active disease from onset to cure or death"
        ),
        # ------------------ clinical features ------------------ #
        "prop_smear_positive": Parameter(
            Types.REAL, "proportion of new active cases that will be smear-positive"
        ),
        "prop_smear_positive_hiv": Parameter(
            Types.REAL, "proportion of hiv+ active tb cases that will be smear-positive"
        ),
        # ------------------ mortality ------------------ #
        # untreated
        "death_rate_smear_pos_untreated": Parameter(
            Types.REAL,
            "probability of death in smear-positive tb cases with untreated tb",
        ),
        "death_rate_smear_neg_untreated": Parameter(
            Types.REAL,
            "probability of death in smear-negative tb cases with untreated tb",
        ),
        # treated
        "death_rate_child0_4_treated": Parameter(
            Types.REAL, "probability of death in child aged 0-4 years with treated tb"
        ),
        "death_rate_child5_14_treated": Parameter(
            Types.REAL, "probability of death in child aged 5-14 years with treated tb"
        ),
        "death_rate_adult_treated": Parameter(
            Types.REAL, "probability of death in adult aged >=15 years with treated tb"
        ),
        # ------------------ progression to active disease ------------------ #
        "rr_tb_bcg": Parameter(
            Types.REAL,
            "relative risk of progression to active disease for children with BCG vaccine",
        ),
        "rr_tb_hiv": Parameter(
            Types.REAL, "relative risk of progression to active disease for PLHIV"
        ),
        "rr_tb_aids": Parameter(
            Types.REAL,
            "relative risk of progression to active disease for PLHIV with AIDS",
        ),
        "rr_tb_art_adult": Parameter(
            Types.REAL,
            "relative risk of progression to active disease for adults with HIV on ART",
        ),
        "rr_tb_art_child": Parameter(
            Types.REAL,
            "relative risk of progression to active disease for adults with HIV on ART",
        ),
        "rr_tb_obese": Parameter(
            Types.REAL, "relative risk of progression to active disease if obese"
        ),
        "rr_tb_diabetes1": Parameter(
            Types.REAL,
            "relative risk of progression to active disease with type 1 diabetes",
        ),
        "rr_tb_alcohol": Parameter(
            Types.REAL,
            "relative risk of progression to active disease with heavy alcohol use",
        ),
        "rr_tb_smoking": Parameter(
            Types.REAL, "relative risk of progression to active disease with smoking"
        ),
        "rr_ipt_adult": Parameter(
            Types.REAL, "relative risk of active TB with IPT in adults"
        ),
        "rr_ipt_child": Parameter(
            Types.REAL, "relative risk of active TB with IPT in children"
        ),
        "rr_ipt_adult_hiv": Parameter(
            Types.REAL, "relative risk of active TB with IPT in adults with hiv"
        ),
        "rr_ipt_child_hiv": Parameter(
            Types.REAL, "relative risk of active TB with IPT in children with hiv"
        ),
        "rr_ipt_art_adult": Parameter(
            Types.REAL, "relative risk of active TB with IPT and ART in adults"
        ),
        "rr_ipt_art_child": Parameter(
            Types.REAL, "relative risk of active TB with IPT and ART in children"
        ),
        # ------------------ diagnostic tests ------------------ #
        "sens_xpert_smear_negative": Parameter(
            Types.REAL, "sensitivity of Xpert test in smear negative TB cases"),
        "sens_xpert_smear_positive": Parameter(
            Types.REAL, "sensitivity of Xpert test in smear positive TB cases"),
        "spec_xpert_smear_negative": Parameter(
            Types.REAL, "specificity of Xpert test in smear negative TB cases"),
        "spec_xpert_smear_positive": Parameter(
            Types.REAL, "specificity of Xpert test in smear positive TB cases"),
        "sens_sputum_smear_positive": Parameter(
            Types.REAL,
            "sensitivity of sputum smear microscopy in sputum positive cases",
        ),
        "spec_sputum_smear_positive": Parameter(
            Types.REAL,
            "specificity of sputum smear microscopy in sputum positive cases",
        ),
        "sens_clinical": Parameter(
            Types.REAL, "sensitivity of clinical diagnosis in detecting active TB"
        ),
        "spec_clinical": Parameter(
            Types.REAL, "specificity of clinical diagnosis in detecting TB"
        ),
        "sens_xray_smear_negative": Parameter(
            Types.REAL, "sensitivity of x-ray diagnosis in smear negative TB cases"
        ),
        "sens_xray_smear_positive": Parameter(
            Types.REAL, "sensitivity of x-ray diagnosis in smear positive TB cases"
        ),
        "spec_xray_smear_negative": Parameter(
            Types.REAL, "specificity of x-ray diagnosis in smear negative TB cases"
        ),
        "spec_xray_smear_positive": Parameter(
            Types.REAL, "specificity of x-ray diagnosis in smear positive TB cases"
        ),
        # ------------------ treatment success rates ------------------ #
        "prob_tx_success_ds": Parameter(
            Types.REAL, "Probability of treatment success for new and relapse TB cases"
        ),
        "prob_tx_success_mdr": Parameter(
            Types.REAL, "Probability of treatment success for MDR-TB cases"
        ),
        "prob_tx_success_0_4": Parameter(
            Types.REAL, "Probability of treatment success for children aged 0-4 years"
        ),
        "prob_tx_success_5_14": Parameter(
            Types.REAL, "Probability of treatment success for children aged 5-14 years"
        ),
        "prob_tx_success_shorter": Parameter(
            Types.REAL, "Probability of treatment success for children aged <16 years on shorter regimen"
        ),
        # ------------------ testing rates ------------------ #
        "rate_testing_general_pop": Parameter(
            Types.REAL,
            "rate of screening / testing per month in general population",
        ),
        "rate_testing_active_tb": Parameter(
            Types.DATA_FRAME,
            "rate of screening / testing per month in population with active tb",
        ),
        # ------------------ treatment regimens ------------------ #
        "ds_treatment_length": Parameter(
            Types.REAL,
            "length of treatment for drug-susceptible tb (first case) in months",
        ),
        "ds_retreatment_length": Parameter(
            Types.REAL,
            "length of treatment for drug-susceptible tb (secondary case) in months",
        ),
        "mdr_treatment_length": Parameter(
            Types.REAL, "length of treatment for mdr-tb in months"
        ),
        "child_shorter_treatment_length": Parameter(
            Types.REAL, "length of treatment for shorter paediatric regimen in months"
        ),
        "prob_retained_ipt_6_months": Parameter(
            Types.REAL,
            "probability of being retained on IPT every 6 months if still eligible",
        ),
        "age_eligibility_for_ipt": Parameter(
            Types.REAL,
            "eligibility criteria (years of age) for IPT given to contacts of TB cases",
        ),
        "ipt_start_date": Parameter(
            Types.INT,
            "year from which IPT is available for paediatric contacts of diagnosed active TB cases",
        ),
        "scenario": Parameter(
            Types.INT,
            "integer value labelling the scenario to be run: default is 0"
        ),
        "scenario_start_date": Parameter(
            Types.DATE,
            "date from which different scenarios are run"
        ),
        "first_line_test": Parameter(
            Types.STRING,
            "name of first test to be used for TB diagnosis"
        ),
        "second_line_test": Parameter(
            Types.STRING,
            "name of second test to be used for TB diagnosis"
        ),
        "probability_access_to_xray": Parameter(
            Types.REAL,
            "probability a person will have access to chest x-ray"
        ),
        "prob_tb_referral_in_generic_hsi": Parameter(
            Types.REAL,
            "probability of referral to TB screening HSI if presenting with TB-related symptoms"
        ),
        "scenario_SI": Parameter(
            Types.STRING,
            "sub-set of scenarios used for sensitivity analysis"
        ),
    }

    def read_parameters(self, data_folder):
        """
        * 1) Reads the ResourceFiles
        * 2) Declares the DALY weights
        * 3) Declares the Symptoms
        """

        # 1) Read the ResourceFiles
        workbook = pd.read_excel(
            os.path.join(self.resourcefilepath, "ResourceFile_TB.xlsx"), sheet_name=None
        )
        self.load_parameters_from_dataframe(workbook["parameters"])

        p = self.parameters

        # assume cases distributed equally across districts
        # todo updated WHO data
        p["who_incidence_estimates"] = workbook["WHO_activeTB2023"]

        # use NTP reported treatment rates as testing rates (perfect referral)
        p["rate_testing_active_tb"] = workbook["NTP2019"]
        p["followup_times"] = workbook["followup"]

        # if using national-level model, include all districts in IPT coverage
        # p['tb_high_risk_distr'] = workbook['IPTdistricts']
        p["tb_high_risk_distr"] = workbook["all_districts"]

        p["ipt_coverage"] = workbook["ipt_coverage"]

        self.district_list = (
            self.sim.modules["Demography"]
                .parameters["pop_2010"]["District"]
                .unique()
                .tolist()
        )

        # 2) Get the DALY weights
        if "HealthBurden" in self.sim.modules.keys():
            # HIV-negative
            # Drug-susceptible tuberculosis, not HIV infected
            self.daly_wts["daly_tb"] = self.sim.modules["HealthBurden"].get_daly_weight(
                0
            )
            # multi-drug resistant tuberculosis, not HIV infected
            self.daly_wts["daly_mdr_tb"] = self.sim.modules[
                "HealthBurden"
            ].get_daly_weight(1)

            # HIV-positive
            # Drug-susceptible Tuberculosis, HIV infected without anaemia
            self.daly_wts["daly_tb_hiv"] = self.sim.modules[
                "HealthBurden"
            ].get_daly_weight(7)
            # Multi-drug resistant Tuberculosis, HIV infected and anemia, moderate
            self.daly_wts["daly_mdr_tb_hiv"] = self.sim.modules[
                "HealthBurden"
            ].get_daly_weight(9)

        # 3) Declare the Symptoms
        # additional healthcare-seeking behaviour with these symptoms
        self.sim.modules["SymptomManager"].register_symptom(
            Symptom(
                name="fatigue",
                odds_ratio_health_seeking_in_adults=5.0,
                odds_ratio_health_seeking_in_children=5.0,
            )
        )

        self.sim.modules["SymptomManager"].register_symptom(
            Symptom(
                name="night_sweats",
                odds_ratio_health_seeking_in_adults=5.0,
                odds_ratio_health_seeking_in_children=5.0,
            )
        )

    def pre_initialise_population(self):
        """
        * Establish the Linear Models
        """
        p = self.parameters

        self.lm["active_tb"] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1,
            Predictor("age_years").when("<=15", p["rr_tb_child"]),
            # -------------- LIFESTYLE -------------- #
            Predictor().when(
                'va_bcg_all_doses &'
                '(hv_inf == False) &'
                '(age_years <10)',
                p["rr_tb_bcg"]  # child with bcg
            ),
            Predictor("li_bmi").when(">=4", p["rr_tb_obese"]),
            # Predictor('diabetes').when(True, p['rr_tb_diabetes1']),
            Predictor("li_ex_alc").when(True, p["rr_tb_alcohol"]),
            Predictor("li_tob").when(True, p["rr_tb_smoking"]),
            # -------------- IPT -------------- #
            Predictor().when(
                '~hv_inf &'
                'tb_on_ipt & '
                'age_years <= 15',
                p["rr_ipt_child"]),  # hiv- child on ipt
            Predictor().when(
                '~hv_inf &'
                'tb_on_ipt & '
                'age_years > 15',
                p["rr_ipt_adult"]),  # hiv- adult on ipt
            # -------------- PLHIV -------------- #
            Predictor("hv_inf").when(True, p["rr_tb_hiv"]),
            Predictor("sy_aids_symptoms").when(">0", p["rr_tb_aids"]),
            # on ART, no IPT
            Predictor().when(
                'hv_inf & '
                '(hv_art == "on_VL_suppressed") &'
                '~tb_on_ipt & '
                'age_years <= 15',
                p["rr_tb_art_child"]),  # hiv+ child on ART
            Predictor().when(
                'hv_inf & '
                '(hv_art == "on_VL_suppressed") &'
                '~tb_on_ipt & '
                'age_years > 15',
                p["rr_tb_art_adult"]),  # hiv+ adult on ART
            # on ART, on IPT
            Predictor().when(
                'tb_on_ipt & '
                'hv_inf & '
                'age_years <= 15 &'
                '(hv_art == "on_VL_suppressed")',
                (p["rr_tb_art_child"] * p["rr_ipt_art_child"]),  # hiv+ child on ART+IPT
            ),
            Predictor().when(
                'tb_on_ipt & '
                'hv_inf & '
                'age_years > 15 &'
                '(hv_art == "on_VL_suppressed")',
                (p["rr_tb_art_adult"] * p["rr_ipt_art_adult"]),  # hiv+ adult on ART+IPT
            ),
            # not on ART, on IPT
            Predictor().when(
                'tb_on_ipt & '
                'hv_inf & '
                'age_years <= 15 &'
                '(hv_art != "on_VL_suppressed")',
                p["rr_ipt_child_hiv"],  # hiv+ child IPT only
            ),
            Predictor().when(
                'tb_on_ipt & '
                'hv_inf & '
                'age_years > 15 &'
                '(hv_art != "on_VL_suppressed")',
                p["rr_ipt_adult_hiv"],  # hiv+ adult IPT only
            ),
        )

        self.lm["risk_relapse_2yrs"] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p["monthly_prob_relapse_tx_complete"],
            Predictor("hv_inf").when(True, p["rr_relapse_hiv"]),
            Predictor("tb_treatment_failure")
            .when(True, (p["monthly_prob_relapse_tx_incomplete"] / p["monthly_prob_relapse_tx_complete"])),
            Predictor().when(
                'tb_on_ipt & '
                'age_years <= 15',
                p["rr_ipt_child"]),
            Predictor().when(
                'tb_on_ipt & '
                'age_years > 15',
                p["rr_ipt_adult"]),
        )

        # risk of relapse if >=2 years post treatment
        self.lm["risk_relapse_late"] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p["monthly_prob_relapse_2yrs"],
            Predictor("hv_inf").when(True, p["rr_relapse_hiv"]),
            Predictor().when(
                'tb_on_ipt & '
                'age_years <= 15',
                p["rr_ipt_child"]),
            Predictor().when(
                'tb_on_ipt & '
                'age_years > 15',
                p["rr_ipt_adult"]),
        )

        # probability of death
        self.lm["death_rate"] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1,
            Predictor().when(
                "(tb_on_treatment == True) & "
                "(age_years <=4)",
                p["death_rate_child0_4_treated"],
            ),
            Predictor().when(
                "(tb_on_treatment == True) & "
                "(age_years <=14)",
                p["death_rate_child5_14_treated"],
            ),
            Predictor().when(
                "(tb_on_treatment == True) & "
                "(age_years >=15)",
                p["death_rate_adult_treated"],
            ),
            Predictor().when(
                "(tb_on_treatment == False) & "
                "(tb_smear == True)",
                p["death_rate_smear_pos_untreated"],
            ),
            Predictor().when(
                "(tb_on_treatment == False) & "
                "(tb_smear == False)",
                p["death_rate_smear_neg_untreated"],
            ),
        )

    def send_for_screening_general(self, population):

        df = population.props
        p = self.parameters
        rng = self.rng

        random_draw = rng.random_sample(size=len(df))

        # randomly select some individuals for screening and testing
        # this may include some newly infected active tb cases (that's fine)
        screen_idx = df.index[
            df.is_alive
            & ~df.tb_diagnosed
            & ~df.tb_on_treatment
            & (random_draw < p["rate_testing_general_pop"])
            ]

        for person in screen_idx:
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_ScreeningAndRefer(person_id=person, module=self),
                topen=random_date(self.sim.date, self.sim.date + DateOffset(months=1), self.rng),
                tclose=None,
                priority=0,
            )

    def select_tb_test(self, person_id):

        df = self.sim.population.props
        p = self.parameters
        person = df.loc[person_id]

        # xpert tests limited to 60% coverage
        # if selected test is xpert, check for availability
        # give sputum smear as back-up
        # assume sputum smear always available

        # previously diagnosed/treated or hiv+ -> xpert
        if person["tb_ever_treated"] or person["hv_diagnosed"] or (p["first_line_test"] == 'xpert'):
            return "xpert"
        else:
            return "sputum"

    def get_consumables_for_dx_and_tx(self):
        p = self.parameters
        # consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
        hs = self.sim.modules["HealthSystem"]

        # TB Sputum smear test
        # assume that if smear-positive, sputum smear test is 100% specific and sensitive
        self.item_codes_for_consumables_required['sputum_test'] = \
            hs.get_item_codes_from_package_name("Microscopy Test")

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            tb_sputum_test_smear_positive=DxTest(
                property='tb_inf',
                target_categories=["active"],
                sensitivity=p["sens_sputum_smear_positive"],
                specificity=p["spec_sputum_smear_positive"],
                item_codes=self.item_codes_for_consumables_required['sputum_test']
            )
        )
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            tb_sputum_test_smear_negative=DxTest(
                property='tb_inf',
                target_categories=["active"],
                sensitivity=0.0,
                specificity=1.0,
                item_codes=self.item_codes_for_consumables_required['sputum_test']
            )
        )

        # TB GeneXpert
        self.item_codes_for_consumables_required['xpert_test'] = \
            hs.get_item_codes_from_package_name("Xpert test")

        # sensitivity/specificity set for smear status of cases
        self.sim.modules["HealthSystem"].dx_manager.register_dx_test(
            tb_xpert_test_smear_positive=DxTest(
                property="tb_inf",
                target_categories=["active"],
                sensitivity=p["sens_xpert_smear_positive"],
                specificity=p["spec_xpert_smear_positive"],
                item_codes=self.item_codes_for_consumables_required['xpert_test']
            )
        )
        self.sim.modules["HealthSystem"].dx_manager.register_dx_test(
            tb_xpert_test_smear_negative=DxTest(
                property="tb_inf",
                target_categories=["active"],
                sensitivity=p["sens_xpert_smear_negative"],
                specificity=p["spec_xpert_smear_negative"],
                item_codes=self.item_codes_for_consumables_required['xpert_test']
            )
        )

        # TB Chest x-ray
        self.item_codes_for_consumables_required['chest_xray'] = {
            hs.get_item_code_from_item_name("X-ray"): 1}

        # sensitivity/specificity set for smear status of cases
        self.sim.modules["HealthSystem"].dx_manager.register_dx_test(
            tb_xray_smear_positive=DxTest(
                property="tb_inf",
                target_categories=["active"],
                sensitivity=p["sens_xray_smear_positive"],
                specificity=p["spec_xray_smear_positive"],
                item_codes=self.item_codes_for_consumables_required['chest_xray']
            )
        )
        self.sim.modules["HealthSystem"].dx_manager.register_dx_test(
            tb_xray_smear_negative=DxTest(
                property="tb_inf",
                target_categories=["active"],
                sensitivity=p["sens_xray_smear_negative"],
                specificity=p["spec_xray_smear_negative"],
                item_codes=self.item_codes_for_consumables_required['chest_xray']
            )
        )

        # TB clinical diagnosis
        self.sim.modules["HealthSystem"].dx_manager.register_dx_test(
            tb_clinical=DxTest(
                property="tb_inf",
                target_categories=["active"],
                sensitivity=p["sens_clinical"],
                specificity=p["spec_clinical"],
                item_codes=[]
            )
        )

        # 4) -------- Define the treatment options --------
        # adult treatment - primary
        # self.item_codes_for_consumables_required['tb_tx_adult'] = \
        #     hs.get_item_codes_from_package_name("First line treatment for new TB cases for adults")
        self.item_codes_for_consumables_required['tb_tx_adult'] = \
            hs.get_item_code_from_item_name("Cat. I & III Patient Kit A")

        # child treatment - primary
        # self.item_codes_for_consumables_required['tb_tx_child'] = \
        #     hs.get_item_codes_from_package_name("First line treatment for new TB cases for children")
        self.item_codes_for_consumables_required['tb_tx_child'] = \
            hs.get_item_code_from_item_name("Cat. I & III Patient Kit B")

        # child treatment - primary, shorter regimen
        # self.item_codes_for_consumables_required['tb_tx_child_shorter'] = \
        #     hs.get_item_codes_from_package_name("First line treatment for new TB cases for children shorter regimen")
        self.item_codes_for_consumables_required['tb_tx_child_shorter'] = \
            hs.get_item_code_from_item_name("Cat. I & III Patient Kit B")

        # adult treatment - secondary
        # self.item_codes_for_consumables_required['tb_retx_adult'] = \
        #     hs.get_item_codes_from_package_name("First line treatment for retreatment TB cases for adults")
        self.item_codes_for_consumables_required['tb_retx_adult'] = \
            hs.get_item_code_from_item_name("Cat. II Patient Kit A1")

        # child treatment - secondary
        # self.item_codes_for_consumables_required['tb_retx_child'] = \
        #     hs.get_item_codes_from_package_name("First line treatment for retreatment TB cases for children")
        self.item_codes_for_consumables_required['tb_retx_child'] = \
            hs.get_item_code_from_item_name("Cat. II Patient Kit A2")

        # mdr treatment
        self.item_codes_for_consumables_required['tb_mdrtx'] = {
            hs.get_item_code_from_item_name("Treatment: second-line drugs"): 1}

        # ipt
        self.item_codes_for_consumables_required['tb_ipt'] = {
            hs.get_item_code_from_item_name("Isoniazid/Pyridoxine, tablet 300 mg"): 1}

    def initialise_population(self, population):

        df = population.props
        p = self.parameters

        # if HIV is not registered, create a dummy property
        if "Hiv" not in self.sim.modules:
            population.make_test_property("hv_inf", Types.BOOL)
            population.make_test_property("sy_aids_symptoms", Types.INT)
            population.make_test_property("hv_art", Types.STRING)

            df["hv_inf"] = False
            df["sy_aids_symptoms"] = 0
            df["hv_art"] = "not"

        # Set our property values for the initial population
        df["tb_inf"].values[:] = "uninfected"
        df["tb_strain"].values[:] = "none"

        df["tb_date_latent"] = pd.NaT
        df["tb_scheduled_date_active"] = pd.NaT
        df["tb_date_active"] = pd.NaT
        df["tb_smear"] = False

        # ------------------ testing status ------------------ #
        # todo
        df["tb_date_tested"] = pd.NaT
        df["tb_diagnosed"] = False
        df["tb_date_diagnosed"] = pd.NaT
        df["tb_diagnosed_mdr"] = False

        # ------------------ treatment status ------------------ #
        df["tb_on_treatment"] = False
        df["tb_date_treated"] = pd.NaT
        df["tb_treatment_regimen"].values[:] = "none"
        df["tb_ever_treated"] = False
        df["tb_treatment_failure"] = False

        df["tb_on_ipt"] = False
        df["tb_date_ipt"] = pd.NaT

        # # ------------------ infection status ------------------ #
        # todo set incidence for full yr 2010 then run poll from jan 2011
        # WHO estimates of active TB for 2010
        # need an infected initial population
        inc_estimates = p["who_incidence_estimates"]
        incidence_year = (inc_estimates.loc[
            (inc_estimates.year == self.sim.date.year), "incidence_per_100k"
        ].values[0]) / 100000

        # todo change to incidence year
        self.assign_baseline_active_tb(
            population,
            strain="ds",
            incidence=incidence_year)

        # todo changes prop_mdr to 0.0186 (error in resourcefile)
        # todo change to incidence year
        self.assign_baseline_active_tb(
            population,
            strain="mdr",
            incidence=incidence_year * p['prop_mdr2010'])

        self.send_for_screening_general(
            population
        )  # send some baseline population for screening

    def initialise_simulation(self, sim):
        """
        * 1) Schedule the regular TB events
        * 2) Schedule the scenario change
        * 3) Define the DxTests and treatment options
        """

        # 1) Regular events
        sim.schedule_event(TbActiveEvent(self), sim.date + DateOffset(days=0))
        sim.schedule_event(TbTreatmentAndRelapseEvents(self), sim.date + DateOffset(days=0))
        sim.schedule_event(TbSelfCureEvent(self), sim.date + DateOffset(days=0))
        sim.schedule_event(TbActiveCasePoll(self), sim.date + DateOffset(years=1))

        # log at the end of the year
        sim.schedule_event(TbLoggingEvent(self), sim.date + DateOffset(years=1))

        # 2) Scenario change
        sim.schedule_event(ScenarioSetupEvent(self), self.parameters["scenario_start_date"])

        # 3) Define the DxTests and get the consumables required
        self.get_consumables_for_dx_and_tx()

        # 4) (Optionally) Schedule the event to check the configuration of all properties
        if self.run_with_checks:
            sim.schedule_event(
                TbCheckPropertiesEvent(self), sim.date + pd.DateOffset(months=1)
            )

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual
        allocate IPT for child if mother diagnosed with TB
        """

        df = self.sim.population.props
        now = self.sim.date

        df.at[child_id, "tb_inf"] = "uninfected"
        df.at[child_id, "tb_strain"] = "none"

        df.at[child_id, "tb_date_latent"] = pd.NaT
        df.at[child_id, "tb_scheduled_date_active"] = pd.NaT
        df.at[child_id, "tb_date_active"] = pd.NaT
        df.at[child_id, "tb_smear"] = False

        # ------------------ testing status ------------------ #
        # todo
        df.at[child_id, "tb_date_tested"] = pd.NaT

        df.at[child_id, "tb_diagnosed"] = False
        df.at[child_id, "tb_date_diagnosed"] = pd.NaT
        df.at[child_id, "tb_diagnosed_mdr"] = False

        # ------------------ treatment status ------------------ #
        df.at[child_id, "tb_on_treatment"] = False
        df.at[child_id, "tb_date_treated"] = pd.NaT
        df.at[child_id, "tb_treatment_regimen"] = "none"
        df.at[child_id, "tb_treatment_failure"] = False
        df.at[child_id, "tb_ever_treated"] = False

        df.at[child_id, "tb_on_ipt"] = False
        df.at[child_id, "tb_date_ipt"] = pd.NaT

        if "Hiv" not in self.sim.modules:
            df.at[child_id, "hv_inf"] = False
            df.at[child_id, "sy_aids_symptoms"] = 0
            df.at[child_id, "hv_art"] = "not"

        # if mother is diagnosed with TB, give IPT to infant
        mother_id = mother_id if mother_id != -1 else self.rng.choice(
            df.index[df.is_alive & (df.sex == "F") & (df.age_years > 16)])
        assert mother_id != -1

        if df.at[mother_id, "tb_diagnosed"]:
            event = HSI_Tb_Start_or_Continue_Ipt(self, person_id=child_id)
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                event,
                priority=1,
                topen=now,
                tclose=now + DateOffset(days=28),
            )

    def report_daly_values(self):
        """
        This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        experienced by persons in the previous month. Only rows for alive-persons must be returned.
        The names of the series of columns is taken to be the label of the cause of this disability.
        It will be recorded by the healthburden module as <ModuleName>_<Cause>.
        """
        df = self.sim.population.props  # shortcut to population properties dataframe

        # to avoid errors when hiv module not running
        df_tmp = df.loc[df.is_alive]
        health_values = pd.Series(0, index=df_tmp.index)

        # hiv-negative
        health_values.loc[
            (df_tmp.tb_inf == "active")
            & (df_tmp.tb_strain == "ds")
            & ~df_tmp.hv_inf
            ] = self.daly_wts["daly_tb"]

        health_values.loc[
            (df_tmp.tb_inf == "active")
            & (df_tmp.tb_strain == "mdr")
            & ~df_tmp.hv_inf
            ] = self.daly_wts["daly_tb"]

        # hiv-positive
        health_values.loc[
            (df_tmp.tb_inf == "active")
            & (df_tmp.tb_strain == "ds")
            & df_tmp.hv_inf
            ] = self.daly_wts["daly_tb_hiv"]

        health_values.loc[
            (df_tmp.tb_inf == "active")
            & (df_tmp.tb_strain == "mdr")
            & df_tmp.hv_inf
            ] = self.daly_wts["daly_mdr_tb_hiv"]

        return health_values.loc[df.is_alive]

    def assign_baseline_active_tb(self, population, strain, incidence):
        """
        select individuals to be infected during baseline year
        assign scheduled date of active tb onset
        update properties as needed
        symptoms and smear status are assigned in the TbActiveEvent
        """
        # todo changed this to assign all 2010 tb cases
        df = population.props
        rng = self.rng
        now = self.sim.date
        p = self.parameters

        # ------------------ infection status ------------------ #
        # identify eligible people, not currently with active tb infection
        eligible = df.loc[
            df.is_alive
            & (df.tb_inf != "active")
            ].index

        # weight risk by individual characteristics
        # Compute chance that each susceptible person becomes infected:
        rr_of_infection = self.lm["active_tb"].predict(
            df.loc[eligible]
        )
        # todo add this scaling
        # scale to get overall prevalence correct
        scaled_rr_of_infection = rr_of_infection / rr_of_infection.mean()

        # todo add scaled rr
        #  probability of infection
        p_infection = (scaled_rr_of_infection * incidence)

        # New infections:
        will_be_infected = (
            self.rng.random_sample(len(p_infection)) < p_infection
        )
        idx_new_infection = will_be_infected[will_be_infected].index

        df.loc[idx_new_infection, "tb_strain"] = strain

        # todo change timeframe to one yr
        # schedule onset of active tb, time now up to 1 yr
        for person_id in idx_new_infection:
            date_progression = now + pd.DateOffset(
                days=rng.randint(0, 365)
            )

            # todo reset
            # set date of active tb - properties will be updated at TbActiveEvent every month
            df.at[person_id, "tb_scheduled_date_active"] = date_progression

    def assign_active_tb(self, population, strain):
        """
        select individuals to be infected - strain-specific
        assign scheduled date of active tb onset
        update properties as needed
        symptoms and smear status are assigned in the TbActiveEvent
        """
        df = population.props
        rng = self.rng
        now = self.sim.date
        p = self.parameters

        # ----------------------------------- TRANSMISSION MODEL -----------------------------------
        # Count current number of alive people with active TB
        # including children and adults - equally transmissible (assumed)
        # assume those on treatment not infectious (from day 1 of tx)
        n_smear_pos = len(
            df.loc[
                df.is_alive
                & (df.tb_inf == "active")
                & (df.tb_strain == strain)
                & ~df.tb_on_treatment
                & df.tb_smear
                ]
        )
        n_smear_neg = len(
            df.loc[
                df.is_alive
                & (df.tb_inf == "active")
                & (df.tb_strain == strain)
                & ~df.tb_on_treatment
                & ~df.tb_smear
                ]
        )

        # add in mdr cases on incorrect treatment - will continue to transmit infection
        if strain == "mdr":
            mdr_smear_pos_on_wrong_tx = len(
                df.loc[
                    df.is_alive
                    & (df.tb_inf == "active")
                    & (df.tb_strain == strain)
                    & df.tb_on_treatment
                    & (df.tb_treatment_regimen != "tb_mdrtx")
                    & df.tb_smear
                    ]
            )
            n_smear_pos = n_smear_pos + mdr_smear_pos_on_wrong_tx

            mdr_smear_neg_on_wrong_tx = len(
                df.loc[
                    df.is_alive
                    & (df.tb_inf == "active")
                    & (df.tb_strain == strain)
                    & df.tb_on_treatment
                    & (df.tb_treatment_regimen != "tb_mdrtx")
                    & ~df.tb_smear
                    ]
            )
            n_smear_neg = n_smear_neg + mdr_smear_neg_on_wrong_tx

        if n_smear_pos > 0:

            # identify susceptible people, not currently with active tb infection
            # can be latent (prior infection) or never infected
            susc_idx = df.loc[
                df.is_alive
                & (df.tb_inf != "active")
                ].index
            n_susc = len(susc_idx)

            # weight risk by individual characteristics
            # Compute chance that each susceptible person becomes infected:
            rr_of_infection = self.lm["active_tb"].predict(
                df.loc[susc_idx]
            )
            # todo add brackets around n_smear_neg*0.2
            #  - probability of infection = beta * I/N
            # relative infectiousness of smear-negative is lower
            p_infection = (
                rr_of_infection * p['beta'] *
                (
                    (n_smear_pos + (n_smear_neg * 0.2)) /
                    (n_smear_pos + n_smear_neg + n_susc)
                )
            )

            # New infections:
            will_be_infected = (
                self.rng.random_sample(len(p_infection)) < p_infection
            )
            idx_new_infection = will_be_infected[will_be_infected].index

            df.loc[idx_new_infection, "tb_strain"] = strain

            # todo change to within one month
            # schedule onset of active tb, time now up to 1 month
            for person_id in idx_new_infection:
                date_progression = now + pd.DateOffset(
                    days=rng.randint(0, 30)
                )

                # set date of active tb - properties will be updated at TbActiveEvent every month
                df.at[person_id, "tb_scheduled_date_active"] = date_progression

    def import_tb_cases(self, population, strain, import_rate):
        """
        select individuals to be infected by importation of infection - strain-specific
        risk of infection is still weighted by individual risk factors
        assign scheduled date of active tb onset
        update properties as needed
        symptoms and smear status are assigned in the TbActiveEvent
        """
        df = population.props
        rng = self.rng
        now = self.sim.date

        # todo add condition not currently with active tb
        # apply risk to all susceptible people
        susc_idx = df.loc[
            df.is_alive
            & (df.tb_inf != "active")
            ].index

        # weight risk by individual characteristics
        # Compute chance that each susceptible person becomes infected:
        rr_of_infection = self.lm["active_tb"].predict(
            df.loc[susc_idx]
        )

        #  probability of infection
        p_infection = rr_of_infection * import_rate

        # New infections:
        will_be_infected = (
            self.rng.random_sample(len(p_infection)) < p_infection
        )
        idx_new_infection = will_be_infected[will_be_infected].index

        df.loc[idx_new_infection, "tb_strain"] = strain

        # schedule onset of active tb, time now up to 1 year
        # if already active -> do nothing
        # if already scheduled active -> do nothing
        for person_id in idx_new_infection:
            if df.at[person_id, "tb_inf"] == "active":
                return

            # todo importation occurs within this month
            # if person doesn't already have scheduled date active...
            if df.at[person_id, "tb_scheduled_date_active"] == pd.NaT:
                date_progression = now + pd.DateOffset(
                    days=rng.randint(0, 30)
                )

                # set date of active tb - properties will be updated at TbActiveEvent every month
                df.at[person_id, "tb_scheduled_date_active"] = date_progression

    def consider_ipt_for_those_initiating_art(self, person_id):
        """
        this is called by HIV when person is initiating ART
        checks whether person is eligible for IPT
        """
        df = self.sim.population.props

        if df.loc[person_id, "tb_diagnosed"] or df.loc[person_id, "tb_diagnosed_mdr"]:
            pass

        high_risk_districts = self.parameters["tb_high_risk_distr"]
        district = df.at[person_id, "district_of_residence"]
        eligible = df.at[person_id, "tb_inf"] != "active"

        # select coverage rate by year:
        # todo add this condition
        now = self.sim.date
        year = now.year if now.year <= 2050 else 2050

        ipt = self.parameters["ipt_coverage"]
        # todo change to == year
        ipt_year = ipt.loc[ipt.year == year]
        ipt_coverage_plhiv = ipt_year.coverage_plhiv

        if (
            (district in high_risk_districts.district_name.values)
            & eligible
            & (self.rng.rand() < ipt_coverage_plhiv.values)
        ):
            # Schedule the TB treatment event:
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_Start_or_Continue_Ipt(self, person_id=person_id),
                priority=1,
                topen=self.sim.date,
                tclose=None,
            )

    def relapse_event(self, population):
        """The Tb Regular Relapse Event
        runs every month to randomly sample amongst those previously infected with active tb
        * Schedules persons who have previously been infected to relapse with a set probability
        * Sets a scheduled_date_active which is picked up by TbActiveEvent
        """

        df = population.props
        rng = self.rng
        now = self.sim.date

        # need a monthly relapse for every person in df
        # should return risk=0 for everyone not eligible for relapse

        # risk of relapse if <2 years post treatment start, includes risk if HIV+
        risk_of_relapse_early = self.lm["risk_relapse_2yrs"].predict(
            df.loc[df.is_alive
                   & df.tb_ever_treated
                   & (df.tb_inf == "latent")
                   & (now < (df.tb_date_treated + pd.DateOffset(years=2)))]
        )

        will_relapse = (
            rng.random_sample(len(risk_of_relapse_early)) < risk_of_relapse_early
        )
        idx_will_relapse_early = will_relapse[will_relapse].index

        # risk of relapse if >=2 years post treatment start, includes risk if HIV+
        risk_of_relapse_later = self.lm["risk_relapse_late"].predict(
            df.loc[df.is_alive
                   & df.tb_ever_treated
                   & (df.tb_inf == "latent")
                   & (now >= (df.tb_date_treated + pd.DateOffset(years=2)))]
        )

        will_relapse_later = (
            rng.random_sample(len(risk_of_relapse_later)) < risk_of_relapse_later
        )
        idx_will_relapse_late2 = will_relapse_later[will_relapse_later].index

        # join both indices
        idx_will_relapse = idx_will_relapse_early.union(
            idx_will_relapse_late2
        ).drop_duplicates()

        # set date of scheduled active tb
        # properties will be updated at TbActiveEvent every month
        df.loc[idx_will_relapse, "tb_scheduled_date_active"] = now

    def end_treatment(self, population):
        """
         * check for those eligible to finish treatment
         * sample for treatment failure and refer for follow-up screening/testing
         * if treatment has finished, change individual properties
         """

        df = population.props
        rng = self.rng
        now = self.sim.date
        p = self.parameters

        # check across population on tb treatment and end treatment if required
        # if current date is after (treatment start date + treatment length) -> end tx

        # ---------------------- treatment end: first case ds-tb (6 months) ---------------------- #
        # end treatment for new tb (ds) cases
        end_ds_tx_idx = df.loc[
            df.is_alive
            & df.tb_on_treatment
            & ((df.tb_treatment_regimen == "tb_tx_adult") | (df.tb_treatment_regimen == "tb_tx_child"))
            & (
                now
                > (df.tb_date_treated + pd.DateOffset(months=p["ds_treatment_length"]))
            )
            ].index

        # ---------------------- treatment end: retreatment ds-tb (7 months) ---------------------- #
        # end treatment for retreatment cases
        end_ds_retx_idx = df.loc[
            df.is_alive
            & df.tb_on_treatment
            & ((df.tb_treatment_regimen == "tb_retx_adult") | (df.tb_treatment_regimen == "tb_retx_child"))
            & (
                now
                > (
                    df.tb_date_treated
                    + pd.DateOffset(months=p["ds_retreatment_length"])
                )
            )
            ].index

        # ---------------------- treatment end: mdr-tb (24 months) ---------------------- #
        # end treatment for mdr-tb cases
        end_mdr_tx_idx = df.loc[
            df.is_alive
            & df.tb_on_treatment
            & (df.tb_treatment_regimen == "tb_mdrtx")
            & (
                now
                > (df.tb_date_treated + pd.DateOffset(months=p["mdr_treatment_length"]))
            )
            ].index

        # ---------------------- treatment end: shorter paediatric regimen ---------------------- #
        # end treatment for paediatric cases on 4 month regimen
        end_tx_shorter_idx = df.loc[
            df.is_alive
            & df.tb_on_treatment
            & (df.tb_treatment_regimen == "tb_tx_child_shorter")
            & (
                now
                > (df.tb_date_treated + pd.DateOffset(months=p["child_shorter_treatment_length"]))
            )
            ].index

        # join indices
        end_tx_idx = end_ds_tx_idx.union(end_ds_retx_idx)
        end_tx_idx = end_tx_idx.union(end_mdr_tx_idx)
        end_tx_idx = end_tx_idx.union(end_tx_shorter_idx)

        # ---------------------- treatment failure ---------------------- #
        # sample some to have treatment failure
        # assume all retreatment cases will cure
        random_var = rng.random_sample(size=len(df))

        # children aged 0-4 ds-tb
        ds_tx_failure0_4_idx = df.loc[
            (df.index.isin(end_ds_tx_idx))
            & (df.age_years < 5)
            & (random_var < (1 - p["prob_tx_success_0_4"]))
            ].index

        # children aged 5-14 ds-tb
        ds_tx_failure5_14_idx = df.loc[
            (df.index.isin(end_ds_tx_idx))
            & (df.age_years.between(5, 14))
            & (random_var < (1 - p["prob_tx_success_5_14"]))
            ].index

        # children aged <16 and on shorter regimen
        ds_tx_failure_shorter_idx = df.loc[
            (df.index.isin(end_tx_shorter_idx))
            & (df.age_years < 16)
            & (random_var < (1 - p["prob_tx_success_shorter"]))
            ].index

        # adults ds-tb
        ds_tx_failure_adult_idx = df.loc[
            (df.index.isin(end_ds_tx_idx))
            & (df.age_years >= 15)
            & (random_var < (1 - p["prob_tx_success_ds"]))
            ].index

        # all mdr cases on ds tx will fail
        failure_in_mdr_with_ds_tx_idx = df.loc[
            (df.index.isin(end_ds_tx_idx))
            & (df.tb_strain == "mdr")
            ].index

        # some mdr cases on mdr treatment will fail
        failure_due_to_mdr_idx = df.loc[
            (df.index.isin(end_mdr_tx_idx))
            & (df.tb_strain == "mdr")
            & (random_var < (1 - p["prob_tx_success_mdr"]))

            ].index

        # join indices of failing cases together
        tx_failure = (
            list(ds_tx_failure0_4_idx)
            + list(ds_tx_failure5_14_idx)
            + list(ds_tx_failure_shorter_idx)
            + list(ds_tx_failure_adult_idx)
            + list(failure_in_mdr_with_ds_tx_idx)
            + list(failure_due_to_mdr_idx)
        )

        if tx_failure:
            df.loc[tx_failure, "tb_treatment_failure"] = True
            df.loc[
                tx_failure, "tb_ever_treated"
            ] = True  # ensure classed as retreatment case

            for person in tx_failure:
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    HSI_Tb_ScreeningAndRefer(person_id=person, module=self),
                    topen=self.sim.date,
                    tclose=None,
                    priority=0,
                )

        # remove any treatment failure indices from the treatment end indices
        cure_idx = list(set(end_tx_idx) - set(tx_failure))

        # change individual properties for all to off treatment
        df.loc[end_tx_idx, "tb_diagnosed"] = False
        df.loc[end_tx_idx, "tb_on_treatment"] = False
        df.loc[end_tx_idx, "tb_treated_mdr"] = False
        # this will indicate that this person has had one complete course of tb treatment
        # subsequent infections will be classified as retreatment
        df.loc[end_tx_idx, "tb_ever_treated"] = True

        # if cured, move infection status back to latent
        # leave tb_strain property set in case of relapse
        df.loc[cure_idx, "tb_inf"] = "latent"
        df.loc[cure_idx, "tb_date_latent"] = now
        df.loc[cure_idx, "tb_smear"] = False

        # this will clear all tb symptoms
        self.sim.modules["SymptomManager"].clear_symptoms(
            person_id=cure_idx, disease_module=self
        )

        # if HIV+ and on ART (virally suppressed), remove AIDS symptoms if cured of TB
        hiv_tb_infected = set(cure_idx).intersection(
            df.loc[
                df.is_alive
                & df.hv_inf
                & (df.hv_art == "on_VL_suppressed")
                ].index
        )

        self.sim.modules["SymptomManager"].clear_symptoms(
            person_id=hiv_tb_infected, disease_module=self.sim.modules["Hiv"]
        )

    def check_config_of_properties(self):
        """check that the properties are currently configured correctly"""
        df = self.sim.population.props
        df_alive = df.loc[df.is_alive]

        # basic check types of columns and dtypes
        orig = self.sim.population.new_row
        assert (df.dtypes == orig.dtypes).all()

        def is_subset(col_for_set, col_for_subset):
            # Confirms that the series of col_for_subset is true only for a subset of the series for col_for_set
            return set(col_for_subset.loc[col_for_subset].index).issubset(
                col_for_set.loc[col_for_set].index
            )

        # Check that core properties of current status are never None/NaN/NaT
        assert not df_alive.tb_inf.isna().any()
        assert not df_alive.tb_strain.isna().any()
        assert not df_alive.tb_smear.isna().any()
        assert not df_alive.tb_on_treatment.isna().any()
        assert not df_alive.tb_treatment_regimen.isna().any()
        assert not df_alive.tb_ever_treated.isna().any()
        assert not df_alive.tb_on_ipt.isna().any()

        # Check that the core TB properties are 'nested' in the way expected.
        assert is_subset(
            col_for_set=(df_alive.tb_inf != "uninfected"), col_for_subset=df_alive.tb_diagnosed
        )
        assert is_subset(
            col_for_set=df_alive.tb_diagnosed, col_for_subset=df_alive.tb_on_treatment
        )

        # Check that if person is infected, the dates of active TB is NOT missing
        assert not df.loc[(df.tb_inf == "active"), "tb_date_active"].isna().all()


# # ---------------------------------------------------------------------------
# #   TB infection event
# # ---------------------------------------------------------------------------
class ScenarioSetupEvent(RegularEvent, PopulationScopeEventMixin):
    """ This event exists to change parameters or functions
    depending on the scenario for projections which has been set
    * scenario 0 is the default which uses baseline parameters
    * scenario 1 optimistic, achieving all program targets
    * scenario 2 optimistic with program constraints
    * scenario 3 optimistic with program constraints and additional measures to reduce incidence
    * scenario 4 optimistic and additional measures to reduce incidence

    It only occurs once at param: scenario_start_date,
    called by initialise_simulation

    the sensitivity analysis is determined by parameter scenario_SI which redacts one intervention at a time
    using parameter values "a"-"i"
    currently this is only called for scenario 4 runs, otherwise the default scenario_SI value is "z"
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(years=100))

    def apply(self, population):

        p = self.module.parameters
        scenario = p["scenario"]

        logger.debug(
            key="message", data=f"ScenarioSetupEvent: scenario {scenario}"
        )

        # baseline scenario 0: no change to parameters/functions
        if scenario == 0:
            return

        # all scenarios 1-4 have scale-up of testing/treatment
        if scenario > 0:

            # HIV
            if p["scenario_SI"] != "a":
                # increase testing/diagnosis rates, default 2020 0.03/0.25 -> 93% dx
                self.sim.modules["Hiv"].parameters["hiv_testing_rates"]["annual_testing_rate_children"] = 0.1
                self.sim.modules["Hiv"].parameters["hiv_testing_rates"]["annual_testing_rate_adults"] = 0.3

                # ANC testing - value for mothers and infants testing
                self.sim.modules["Hiv"].parameters["prob_anc_test_at_delivery"] = 0.95

                # prob ART start if dx, this is already 95% at 2020
                # self.sim.modules["Hiv"].parameters["prob_start_art_after_hiv_test"] = 0.95

            if p["scenario_SI"] != "b":
                # viral suppression rates
                # adults already at 95% by 2020
                # change all column values
                self.sim.modules["Hiv"].parameters["prob_start_art_or_vs"]["virally_suppressed_on_art"] = 95

            # TB
            if p["scenario_SI"] != "c":
                # use NTP treatment rates
                self.sim.modules["Tb"].parameters["rate_testing_active_tb"]["treatment_coverage"] = 90

            if p["scenario_SI"] != "d":
                # increase tb treatment success rates
                self.sim.modules["Tb"].parameters["prob_tx_success_ds"] = 0.9
                self.sim.modules["Tb"].parameters["prob_tx_success_mdr"] = 0.9
                self.sim.modules["Tb"].parameters["prob_tx_success_0_4"] = 0.9
                self.sim.modules["Tb"].parameters["prob_tx_success_5_14"] = 0.9
                self.sim.modules["Tb"].parameters["prob_tx_success_shorter"] = 0.9

            if p["scenario_SI"] != "e":
                # change first-line testing for TB to xpert
                p["first_line_test"] = "xpert"
                p["second_line_test"] = "sputum"

        # remove consumables constraints, all cons available
        if (scenario == 1) or (scenario == 4):
            # list only things that change: constraints on consumables and personnel
            new_parameters = {
                'cons_availability': 'all',  # use cons availability from LMIS
            }
            self.sim.schedule_event(
                HealthSystemChangeParameters(
                    self.sim.modules['HealthSystem'], parameters=new_parameters),
                self.sim.date)

        # improve preventive measures
        if (scenario == 3) or (scenario == 4):

            # HIV
            if p["scenario_SI"] != "f":
                # reduce risk of HIV - applies to whole adult population
                self.sim.modules["Hiv"].parameters["beta"] = self.sim.modules["Hiv"].parameters["beta"] * 0.9

            if p["scenario_SI"] != "g":
                # increase PrEP coverage for FSW after HIV test
                self.sim.modules["Hiv"].parameters["prob_prep_for_fsw_after_hiv_test"] = 0.5

                # prep poll for AGYW - target to the highest risk
                # increase retention to 75% for FSW and AGYW
                self.sim.modules["Hiv"].parameters["prob_prep_for_agyw"] = 0.1
                self.sim.modules["Hiv"].parameters["probability_of_being_retained_on_prep_every_3_months"] = 0.75

            if p["scenario_SI"] != "h":
                # increase probability of VMMC after hiv test
                self.sim.modules["Hiv"].parameters["prob_circ_after_hiv_test"] = 0.25

            # TB
            if p["scenario_SI"] != "i":
                # change IPT eligibility for TB contacts to all years
                p["age_eligibility_for_ipt"] = 100

                # increase coverage of IPT
                p["ipt_coverage"]["coverage_plhiv"] = 0.6
                # todo change from 0.8 to 80
                p["ipt_coverage"]["coverage_paediatric"] = 80  # this will apply to contacts of all ages

                # retention on IPT (PLHIV)
                self.sim.modules["Tb"].parameters["prob_retained_ipt_6_months"] = 0.99


class TbActiveCasePoll(RegularEvent, PopulationScopeEventMixin):
    """The Tb Regular Poll Event for assigning active infections
    * selects people for active infection and schedules onset of active tb
    assign_active_tb uses a transmission model to assign new cases
    import_tb simulates importation of active tb independent of current prevalence
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        p = self.module.parameters

        # transmission ds-tb
        self.module.assign_active_tb(population, strain="ds")

        # transmission mdr-tb, around 1% of total tb incidence
        self.module.assign_active_tb(population, strain="mdr")

        # importation of new ds cases - independent of current prevalence
        self.module.import_tb_cases(population, strain="ds", import_rate=p["importation_rate_ds"])

        # importation of new mdr cases - independent of current prevalence
        self.module.import_tb_cases(population, strain="mdr", import_rate=p["importation_rate_mdr"])


class TbTreatmentAndRelapseEvents(RegularEvent, PopulationScopeEventMixin):
    """ This event runs each month and calls three functions:
    * scheduling TB screening for the general population
    * ending treatment if end of treatment regimen has been reached
    * determining who will relapse after a primary infection
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        # schedule some background rates of tb testing (non-symptom-driven)
        self.module.send_for_screening_general(population)

        self.module.end_treatment(population)
        self.module.relapse_event(population)


class TbActiveEvent(RegularEvent, PopulationScopeEventMixin):
    """
    * check for those with dates of active tb onset within last time-period
    *1 change individual properties for active disease
    *2 assign symptoms
    *3 if HIV+, assign smear status and schedule AIDS onset
    *4 if HIV-, assign smear status and schedule death
    *5 schedule screening for symptomatic active cases
    """

    def __init__(self, module):

        self.repeat = 1
        super().__init__(module, frequency=DateOffset(days=self.repeat))

    def apply(self, population):
        df = population.props
        now = self.sim.date
        p = self.module.parameters
        rng = self.module.rng

        # find people eligible for progression to active disease
        # date of active disease scheduled to occur this week
        # some will be scheduled for future dates
        # if on IPT or treatment - do nothing
        active_idx = df.loc[
            df.is_alive
            & (df.tb_scheduled_date_active < (now + DateOffset(days=self.repeat)))
            & (df.tb_scheduled_date_active >= now)
            & ~df.tb_on_ipt
            & ~df.tb_on_treatment
            ].index

        if active_idx.empty:
            return

        # -------- 1) change individual properties for active disease --------
        df.loc[active_idx, "tb_inf"] = "active"
        df.loc[active_idx, "tb_date_active"] = now
        df.loc[active_idx, "tb_smear"] = False  # default property

        # -------- 2) assign symptoms --------
        self.sim.modules["SymptomManager"].change_symptom(
            person_id=active_idx,
            symptom_string=self.module.symptom_list,
            add_or_remove="+",
            disease_module=self.module,
            duration_in_days=None,
        )

        # -------- 3) if HIV+ assign smear status and schedule AIDS onset --------
        active_and_hiv = df.loc[
            (df.index.isin(active_idx) & df.hv_inf)].index

        # higher probability of being smear positive than HIV-
        smear_pos = (
            rng.random_sample(len(active_and_hiv)) < p["prop_smear_positive_hiv"]
        )
        active_and_hiv_smear_pos = active_and_hiv[smear_pos]
        df.loc[active_and_hiv_smear_pos, "tb_smear"] = True

        if "Hiv" in self.sim.modules:
            for person_id in active_and_hiv:
                self.sim.schedule_event(
                    hiv.HivAidsOnsetEvent(
                        self.sim.modules["Hiv"], person_id, cause="AIDS_TB"
                    ),
                    now,
                )
        else:
            # if Hiv not registered, give HIV+ person same time to death as HIV-
            for person_id in active_and_hiv:
                date_of_tb_death = self.sim.date + pd.DateOffset(
                    months=int(rng.uniform(low=1, high=6))
                )
                self.sim.schedule_event(
                    event=TbDeathEvent(person_id=person_id, module=self.module, cause="AIDS_TB"),
                    date=date_of_tb_death,
                )

        # -------- 4) if HIV- assign smear status and schedule death --------
        active_no_hiv = active_idx[~active_idx.isin(active_and_hiv)]
        smear_pos = rng.random_sample(len(active_no_hiv)) < p["prop_smear_positive"]
        active_no_hiv_smear_pos = active_no_hiv[smear_pos]
        df.loc[active_no_hiv_smear_pos, "tb_smear"] = True

        for person_id in active_no_hiv:
            date_of_tb_death = self.sim.date + pd.DateOffset(
                months=int(rng.uniform(low=1, high=6))
            )
            self.sim.schedule_event(
                event=TbDeathEvent(person_id=person_id, module=self.module, cause="TB"),
                date=date_of_tb_death,
            )

        # -------- 5) schedule screening for asymptomatic and symptomatic people --------
        # sample from all new active cases (active_idx) and determine whether they will seek a test
        # year = now.year if now.year < 2050 else 2050
        year = now.year if now.year < 2020 else 2019
        if now.year == 2010:
            year = 2011

        active_testing_rates = p["rate_testing_active_tb"]

        # change to NTP testing rates
        current_active_testing_rate = active_testing_rates.loc[
                                          (
                                              active_testing_rates.year == year),
                                          "treatment_coverage"].values[
                                          0] / 100

        # multiply testing rate by average treatment availability to match treatment coverage
        current_active_testing_rate = current_active_testing_rate * (1 / 0.6)

        random_draw = rng.random_sample(size=len(df))

        # randomly select some symptomatic individuals for screening and testing
        # would only be screened if have symptoms for >= 14 days
        # sample some of active_idx to go for screening
        screen_active_idx = df.loc[
            (df.index.isin(active_idx) & (random_draw < current_active_testing_rate))].index

        # TB screening checks for symptoms lasting at least 14 days, so add delay
        for person in screen_active_idx:
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_ScreeningAndRefer(person_id=person, module=self.module),
                topen=self.sim.date + DateOffset(days=14),
                tclose=None,
                priority=0,
            )


class TbSelfCureEvent(RegularEvent, PopulationScopeEventMixin):
    """annual event which allows some individuals to self-cure
    approximate time from infection to self-cure is 3 years
    HIV+ and not virally suppressed cannot self-cure
    note that frequency can't be changed here as parameters are set to annual values
    """

    def __init__(self, module):
        # note frequency must remain at 12 months or edit code below for duration active disease
        super().__init__(module, frequency=DateOffset(months=12))

    def apply(self, population):
        p = self.module.parameters
        now = self.sim.date
        rng = self.module.rng

        df = population.props

        prob_self_cure = 1 / p["duration_active_disease_years"]

        # self-cure - move from active to latent, excludes cases that just became active
        random_draw = rng.random_sample(size=len(df))

        # hiv-negative
        self_cure = df.loc[
            (df.tb_inf == "active")
            & df.is_alive
            & ~df.hv_inf
            & (df.tb_date_active < now)
            & (random_draw < prob_self_cure)
            ].index

        # hiv-positive, on art and virally suppressed
        self_cure_art = df.loc[
            (df.tb_inf == "active")
            & df.is_alive
            & df.hv_inf
            & (df.hv_art == "on_VL_suppressed")
            & (df.tb_date_active < now)
            & (random_draw < prob_self_cure)
            ].index

        # resolve symptoms and change properties
        all_self_cure = [*self_cure, *self_cure_art]

        # leave tb strain set in case of relapse
        df.loc[all_self_cure, "tb_inf"] = "latent"
        df.loc[all_self_cure, "tb_diagnosed"] = False
        df.loc[all_self_cure, "tb_smear"] = False

        # this will clear all tb symptoms
        self.sim.modules["SymptomManager"].clear_symptoms(
            person_id=all_self_cure, disease_module=self.module
        )

        # resolve AIDS symptoms if virally suppressed
        self.sim.modules["SymptomManager"].clear_symptoms(
            person_id=self_cure_art, disease_module=self.sim.modules["Hiv"]
        )


# ---------------------------------------------------------------------------
#   Health System Interactions (HSI)
# ---------------------------------------------------------------------------


class HSI_Tb_ScreeningAndRefer(HSI_Event, IndividualScopeEventMixin):
    """
    The is the Screening-and-Refer HSI.
    A positive outcome from symptom-based screening will prompt referral to tb tests (sputum/xpert/xray)
    no consumables are required for screening (4 clinical questions)

    This event is scheduled by:
        * the main event poll,
        * when someone presents for care through a Generic HSI with tb-like symptoms
        * active screening / contact tracing programmes

    If this event is called within another HSI, it may be desirable to limit the functionality of the HSI: do this
    using the arguments:
        * suppress_footprint=True : the HSI will not have any footprint

    This event will:
    * screen individuals for TB symptoms
    * administer appropriate TB test
    * schedule treatment if needed
    * give IPT for paediatric contacts of diagnosed case
    """

    def __init__(self, module, person_id, suppress_footprint=False):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Tb)

        assert isinstance(suppress_footprint, bool)
        self.suppress_footprint = suppress_footprint

        self.TREATMENT_ID = "Tb_Test_Screening"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        """Do the screening and referring to next tests"""

        df = self.sim.population.props
        now = self.sim.date
        p = self.module.parameters
        person = df.loc[person_id]

        # If the person is dead or already diagnosed, do nothing do not occupy any resources
        if not person["is_alive"] or person["tb_diagnosed"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        logger.debug(
            key="message", data=f"HSI_Tb_ScreeningAndRefer: person {person_id}"
        )

        smear_status = person["tb_smear"]

        # If the person is already on treatment and not failing, do nothing do not occupy any resources
        if person["tb_on_treatment"] and not person["tb_treatment_failure"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        # ------------------------- screening ------------------------- #

        # check if patient has: cough, fever, night sweat, weight loss
        # if none of the above conditions are present, no further action
        persons_symptoms = self.sim.modules["SymptomManager"].has_what(person_id)
        if not any(x in self.module.symptom_list for x in persons_symptoms):
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        # ------------------------- testing ------------------------- #
        # if screening indicates presumptive tb
        test = None
        test_result = None
        ACTUAL_APPT_FOOTPRINT = self.EXPECTED_APPT_FOOTPRINT

        # refer for HIV testing: all ages
        # todo exclude if hiv test within last week
        # do not run if already HIV diagnosed or had test in last week
        if not person["hv_diagnosed"] or (person["hv_last_test_date"] >= (now - DateOffset(days=7))):
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                hsi_event=hiv.HSI_Hiv_TestAndRefer(
                    person_id=person_id, module=self.sim.modules["Hiv"], referred_from='Tb'
                ),
                priority=1,
                topen=now,
                tclose=None,
            )

        # child under 5 -> chest x-ray, but access is limited
        # if xray not available, HSI_Tb_Xray_level1b will refer
        if person["age_years"] < 5:
            ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint(
                {"Under5OPD": 1}
            )

            # this HSI will choose relevant sensitivity/specificity depending on person's smear status
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_Xray_level1b(person_id=person_id, module=self.module),
                topen=now,
                tclose=None,
                priority=0,
            )
            test_result = False  # to avoid calling a clinical diagnosis

        # for all presumptive cases over 5 years of age
        else:
            # this selects a test for the person
            # if selection is xpert, will check for availability and return sputum if xpert not available
            test = self.module.select_tb_test(person_id)
            assert test in ["sputum", "xpert"]

            if test == "sputum":
                ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint(
                    {"Over5OPD": 1, "LabTBMicro": 1}
                )

                # relevant test depends on smear status (changes parameters on sensitivity/specificity
                if smear_status:
                    test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                        dx_tests_to_run="tb_sputum_test_smear_positive", hsi_event=self
                    )
                else:
                    # if smear-negative, sputum smear should always return negative
                    # run the dx test to log the consumable
                    test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                        dx_tests_to_run="tb_sputum_test_smear_negative", hsi_event=self
                    )
                    # if negative, check for presence of all symptoms (clinical diagnosis)
                    if all(x in self.module.symptom_list for x in persons_symptoms):
                        test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                            dx_tests_to_run="tb_clinical", hsi_event=self
                        )

            elif test == "xpert":
                ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint(
                    {"Over5OPD": 1}
                )
                # relevant test depends on smear status (changes parameters on sensitivity/specificity
                if smear_status:
                    test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                        dx_tests_to_run="tb_xpert_test_smear_positive", hsi_event=self
                    )
                # for smear-negative people
                else:
                    test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                        dx_tests_to_run="tb_xpert_test_smear_negative", hsi_event=self
                    )

        # ------------------------- testing referrals ------------------------- #

        # if none of the tests are available, try again for sputum
        # requires another appointment - added in ACTUAL_APPT_FOOTPRINT
        if test_result is None:
            if smear_status:
                test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                    dx_tests_to_run="tb_sputum_test_smear_positive", hsi_event=self
                )
            else:
                test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                    dx_tests_to_run="tb_sputum_test_smear_negative", hsi_event=self
                )

            ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint(
                {"Over5OPD": 2, "LabTBMicro": 1}
            )

        # if still no result available, rely on clinical diagnosis
        if test_result is None:
            test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                dx_tests_to_run="tb_clinical", hsi_event=self
            )

        # ------------------------- testing outcomes ------------------------- #

        # diagnosed with mdr-tb - only if xpert used
        if test_result and (test == "xpert") and (person["tb_strain"] == "mdr"):
            df.at[person_id, "tb_diagnosed_mdr"] = True

        # if a test has been performed, update person's properties
        if test_result is not None:
            # todo
            df.at[person_id, "tb_date_tested"] = now

        # if any test returns positive result, refer for appropriate treatment
        if test_result:
            df.at[person_id, "tb_diagnosed"] = True
            df.at[person_id, "tb_date_diagnosed"] = now

            logger.debug(
                key="message",
                data=f"schedule HSI_Tb_StartTreatment for person {person_id}",
            )

            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_StartTreatment(person_id=person_id, module=self.module),
                topen=now,
                tclose=None,
                priority=0,
            )

            # ------------------------- give IPT to contacts ------------------------- #
            # if diagnosed, trigger ipt outreach event for up to 5 contacts of case
            # only high-risk districts are eligible
            # todo add condition for year
            year = now.year if now.year < 2020 else 2019

            district = person["district_of_residence"]
            ipt = self.module.parameters["ipt_coverage"]
            # todo change to == year
            ipt_year = ipt.loc[ipt.year == year]
            ipt_coverage_paed = ipt_year.coverage_paediatric.values[0] / 100

            if (district in p["tb_high_risk_distr"].district_name.values) & (
                self.module.rng.rand() < ipt_coverage_paed
            ):
                # randomly sample from eligible population within district
                ipt_eligible = df.loc[
                    (df.age_years <= p["age_eligibility_for_ipt"])
                    & ~df.tb_diagnosed
                    & df.is_alive
                    & (df.district_of_residence == district)
                    ].index

                if ipt_eligible.any():

                    # select persons at highest risk of tb
                    rr_of_tb = self.module.lm["active_tb"].predict(
                        df.loc[ipt_eligible]
                    )

                    # choose top 5 highest risk contacts
                    ipt_sample = rr_of_tb.sort_values(ascending=False).head(5).index

                    for person_id in ipt_sample:
                        logger.debug(
                            key="message",
                            data=f"HSI_Tb_ScreeningAndRefer: scheduling IPT for person {person_id}",
                        )

                        ipt_event = HSI_Tb_Start_or_Continue_Ipt(
                            self.module, person_id=person_id
                        )
                        self.sim.modules["HealthSystem"].schedule_hsi_event(
                            ipt_event,
                            priority=1,
                            topen=now,
                            tclose=None,
                        )

        # Return the footprint. If it should be suppressed, return a blank footprint.
        if self.suppress_footprint:
            return self.make_appt_footprint({})
        else:
            return ACTUAL_APPT_FOOTPRINT


class HSI_Tb_Xray_level1b(HSI_Event, IndividualScopeEventMixin):
    """
    The is the x-ray HSI
    usually used for testing children unable to produce sputum
    positive result will prompt referral to start treatment

    """

    def __init__(self, module, person_id, suppress_footprint=False):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Tb)

        assert isinstance(suppress_footprint, bool)
        self.suppress_footprint = suppress_footprint

        self.TREATMENT_ID = "Tb_Test_Xray"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"DiagRadio": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props

        if not df.at[person_id, "is_alive"] or df.at[person_id, "tb_diagnosed"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        ACTUAL_APPT_FOOTPRINT = self.EXPECTED_APPT_FOOTPRINT

        smear_status = df.at[person_id, "tb_smear"]

        # select sensitivity/specificity of test based on smear status
        if smear_status:
            test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                dx_tests_to_run="tb_xray_smear_positive", hsi_event=self
            )
        else:
            test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                dx_tests_to_run="tb_xray_smear_negative", hsi_event=self
            )

        # if consumables not available, either refer to level 2 or use clinical diagnosis
        if test_result is None:

            # if smear-positive, assume symptoms strongly predictive of TB
            if smear_status:
                test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                    dx_tests_to_run="tb_clinical", hsi_event=self
                )
                # add another clinic appointment
                ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint(
                    {"Under5OPD": 1, "DiagRadio": 1}
                )

            # if smear-negative, assume still some uncertainty around dx, refer for another x-ray
            else:
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    HSI_Tb_Xray_level2(person_id=person_id, module=self.module),
                    topen=self.sim.date + pd.DateOffset(weeks=1),
                    tclose=None,
                    priority=0,
                )

        # if test returns positive result, refer for appropriate treatment
        if test_result:
            df.at[person_id, "tb_diagnosed"] = True
            df.at[person_id, "tb_date_diagnosed"] = self.sim.date

            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_StartTreatment(person_id=person_id, module=self.module),
                topen=self.sim.date,
                tclose=None,
                priority=0,
            )

        # Return the footprint. If it should be suppressed, return a blank footprint.
        if self.suppress_footprint:
            return self.make_appt_footprint({})
        else:
            return ACTUAL_APPT_FOOTPRINT


class HSI_Tb_Xray_level2(HSI_Event, IndividualScopeEventMixin):
    """
    The is the x-ray HSI performed at level 2
    usually used for testing children unable to produce sputum
    positive result will prompt referral to start treatment
    """

    def __init__(self, module, person_id, suppress_footprint=False):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Tb)

        assert isinstance(suppress_footprint, bool)
        self.suppress_footprint = suppress_footprint

        self.TREATMENT_ID = "Tb_Test_Xray"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"DiagRadio": 1})
        self.ACCEPTED_FACILITY_LEVEL = '2'

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props

        if not df.at[person_id, "is_alive"] or df.at[person_id, "tb_diagnosed"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        ACTUAL_APPT_FOOTPRINT = self.EXPECTED_APPT_FOOTPRINT

        smear_status = df.at[person_id, "tb_smear"]

        # select sensitivity/specificity of test based on smear status
        if smear_status:
            test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                dx_tests_to_run="tb_xray_smear_positive", hsi_event=self
            )
        else:
            test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                dx_tests_to_run="tb_xray_smear_negative", hsi_event=self
            )

        # if consumables not available, rely on clinical diagnosis
        if test_result is None:
            test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                dx_tests_to_run="tb_clinical", hsi_event=self
            )
            # add another clinic appointment
            ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint(
                {"Under5OPD": 1, "DiagRadio": 1}
            )

        # if test returns positive result, refer for appropriate treatment
        if test_result:
            df.at[person_id, "tb_diagnosed"] = True
            df.at[person_id, "tb_date_diagnosed"] = self.sim.date

            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_StartTreatment(person_id=person_id, module=self.module),
                topen=self.sim.date,
                tclose=None,
                priority=0,
            )

        # Return the footprint. If it should be suppressed, return a blank footprint.
        if self.suppress_footprint:
            return self.make_appt_footprint({})
        else:
            return ACTUAL_APPT_FOOTPRINT


# # ---------------------------------------------------------------------------
# #   Treatment
# # ---------------------------------------------------------------------------
# # the consumables at treatment initiation include the cost for the full course of treatment
# # so the follow-up appts don't need to account for consumables, just appt time


class HSI_Tb_StartTreatment(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Tb)

        self.TREATMENT_ID = "Tb_Treatment"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"TBNew": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.number_of_occurrences = 0

    def apply(self, person_id, squeeze_factor):
        """This is a Health System Interaction Event - start TB treatment
        select appropriate treatment and request
        if available, change person's properties
        """
        df = self.sim.population.props
        now = self.sim.date
        person = df.loc[person_id]
        self.number_of_occurrences += 1  # The current appointment is included in the count.

        if not person["is_alive"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        # if person already on treatment or not yet diagnosed, do nothing
        if person["tb_on_treatment"] or not person["tb_diagnosed"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        treatment_regimen = self.select_treatment(person_id)
        treatment_available = self.get_consumables(
            item_codes=self.module.item_codes_for_consumables_required[treatment_regimen]
        )

        if treatment_available:
            # start person on tb treatment - update properties
            df.at[person_id, "tb_on_treatment"] = True
            df.at[person_id, "tb_date_treated"] = now
            df.at[person_id, "tb_treatment_regimen"] = treatment_regimen

            if person["tb_diagnosed_mdr"]:
                df.at[person_id, "tb_treated_mdr"] = True
                df.at[person_id, "tb_date_treated_mdr"] = now

            # schedule first follow-up appointment
            follow_up_date = self.sim.date + DateOffset(months=1)
            logger.debug(
                key="message",
                data=f"HSI_Tb_StartTreatment: scheduling first follow-up "
                     f"for person {person_id} on {follow_up_date}",
            )

            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_FollowUp(person_id=person_id, module=self.module),
                topen=follow_up_date,
                tclose=None,
                priority=0,
            )

        # if treatment not available, return for treatment start in 1 week
        # cap repeated visits at 5
        else:
            if self.number_of_occurrences <= 5:
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    HSI_Tb_StartTreatment(person_id=person_id, module=self.module),
                    topen=self.sim.date + DateOffset(weeks=1),
                    tclose=None,
                    priority=0,
                )

    def select_treatment(self, person_id):
        """
        helper function to select appropriate treatment and check whether
        consumables are available to start drug course
        treatment will always be for ds-tb unless mdr has been identified
        :return: drug_available [BOOL]
        """
        df = self.sim.population.props
        person = df.loc[person_id]

        treatment_regimen = None  # default return value

        # -------- MDR-TB -------- #

        if person["tb_diagnosed_mdr"]:

            treatment_regimen = "tb_mdrtx"

        # -------- First TB infection -------- #
        # could be undiagnosed mdr or ds-tb: treat as ds-tb

        elif not person["tb_ever_treated"]:

            if person["age_years"] >= 15:
                # treatment for ds-tb: adult
                treatment_regimen = "tb_tx_adult"
            else:
                # treatment for ds-tb: child
                treatment_regimen = "tb_tx_child"

        # -------- Secondary TB infection -------- #
        # person has been treated before
        # possible treatment failure or subsequent reinfection
        else:

            if person["age_years"] >= 15:
                # treatment for reinfection ds-tb: adult
                treatment_regimen = "tb_retx_adult"

            else:
                # treatment for reinfection ds-tb: child
                treatment_regimen = "tb_retx_child"

        # -------- SHINE Trial shorter paediatric regimen -------- #
        if (self.module.parameters["scenario"] == 5) \
                & (self.sim.date >= self.module.parameters["scenario_start_date"]) \
                & (person["age_years"] <= 16) \
                & ~(person["tb_smear"]) \
                & ~person["tb_ever_treated"] \
                & ~person["tb_diagnosed_mdr"]:
            # shorter treatment for child with minimal tb
            treatment_regimen = "tb_tx_child_shorter"

        return treatment_regimen


# # ---------------------------------------------------------------------------
# #   Follow-up appts
# # ---------------------------------------------------------------------------
class HSI_Tb_FollowUp(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event
    clinical monitoring for tb patients on treatment
    will schedule sputum smear test if needed
    if positive sputum smear, schedule xpert test for drug sensitivity
    then schedule the next follow-up appt if needed
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Tb)

        self.TREATMENT_ID = "Tb_Test_FollowUp"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"TBFollowUp": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        p = self.module.parameters
        df = self.sim.population.props
        person = df.loc[person_id]

        # Do not run if the person is not alive, or is not currently on treatment
        if (not person["is_alive"]) or (not person["tb_on_treatment"]):
            return

        ACTUAL_APPT_FOOTPRINT = self.EXPECTED_APPT_FOOTPRINT

        # months since treatment start - to compare with monitoring schedule
        # make sure it's an integer value
        months_since_tx = int(
            (self.sim.date - df.at[person_id, "tb_date_treated"]).days / 30.5
        )

        logger.debug(
            key="message",
            data=f"HSI_Tb_FollowUp: person {person_id} on month {months_since_tx} of treatment",
        )

        # default clinical monitoring schedule for first infection ds-tb
        xperttest_result = None
        follow_up_times = p["followup_times"]
        sputum_fup = follow_up_times["ds_sputum"].dropna()
        treatment_length = p["ds_treatment_length"]

        # if previously treated:
        if ((person["tb_treatment_regimen"] == "tb_retx_adult") or
                (person["tb_treatment_regimen"] == "tb_retx_child")):

            # if strain is ds and person previously treated:
            sputum_fup = follow_up_times["ds_retreatment_sputum"].dropna()
            treatment_length = p["ds_retreatment_length"]

        # if person diagnosed with mdr - this treatment schedule takes precedence
        elif person["tb_treatment_regimen"] == "tb_mdrtx":

            sputum_fup = follow_up_times["mdr_sputum"].dropna()
            treatment_length = p["mdr_treatment_length"]

        # if person on shorter paediatric regimen
        elif person["tb_treatment_regimen"] == "tb_tx_child_shorter":
            sputum_fup = follow_up_times["shine_sputum"].dropna()
            treatment_length = p["shine_treatment_length"]

        # check schedule for sputum test and perform if necessary
        if months_since_tx in sputum_fup:
            ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint(
                {"TBFollowUp": 1, "LabTBMicro": 1}
            )

            # choose test parameters based on smear status
            if person["tb_smear"]:
                test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                    dx_tests_to_run="tb_sputum_test_smear_positive", hsi_event=self
                )
            else:
                test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                    dx_tests_to_run="tb_sputum_test_smear_negative", hsi_event=self
                )

            # if sputum test was available and returned positive and not diagnosed with mdr, schedule xpert test
            if test_result and not person["tb_diagnosed_mdr"]:
                ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint(
                    {"TBFollowUp": 1, "LabTBMicro": 1, "LabMolec": 1}
                )
                if person["tb_smear"]:
                    xperttest_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                        dx_tests_to_run="tb_xpert_test_smear_positive", hsi_event=self
                    )
                else:
                    xperttest_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                        dx_tests_to_run="tb_xpert_test_smear_negative", hsi_event=self
                    )

        # if xpert test returns new mdr-tb diagnosis
        if xperttest_result and (df.at[person_id, "tb_strain"] == "mdr"):
            df.at[person_id, "tb_diagnosed_mdr"] = True
            # already diagnosed with active tb so don't update tb_date_diagnosed
            df.at[person_id, "tb_treatment_failure"] = True

            # restart treatment (new regimen) if newly diagnosed with mdr-tb
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_StartTreatment(person_id=person_id, module=self.module),
                topen=self.sim.date,
                tclose=None,
                priority=0,
            )

        # for all ds cases and known mdr cases:
        # schedule next clinical follow-up appt if still within treatment length
        elif months_since_tx < treatment_length:
            follow_up_date = self.sim.date + DateOffset(months=1)
            logger.debug(
                key="message",
                data=f"HSI_Tb_FollowUp: scheduling next follow-up "
                     f"for person {person_id} on {follow_up_date}",
            )

            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_FollowUp(person_id=person_id, module=self.module),
                topen=follow_up_date,
                tclose=None,
                priority=0,
            )

        return ACTUAL_APPT_FOOTPRINT


# ---------------------------------------------------------------------------
#   IPT
# ---------------------------------------------------------------------------
class HSI_Tb_Start_or_Continue_Ipt(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event - give ipt to reduce risk of active TB
    It can be scheduled by:
    * HIV.HSI_Hiv_StartOrContinueTreatment for PLHIV, diagnosed and on ART
    * Tb.HSI_Tb_StartTreatment for up to 5 contacts of diagnosed active TB case

    if person referred by ART initiation (HIV+), IPT given for 36 months
    paediatric IPT is 6-9 months
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        self.TREATMENT_ID = "Tb_Prevention_Ipt"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.number_of_occurrences = 0

    def apply(self, person_id, squeeze_factor):

        logger.debug(key="message", data=f"Starting IPT for person {person_id}")
        self.number_of_occurrences += 1

        df = self.sim.population.props  # shortcut to the dataframe

        person = df.loc[person_id]

        # Do not run if the person is not alive or already on IPT or diagnosed active infection
        if (
            (not person["is_alive"])
            or person["tb_on_ipt"]
            or person["tb_diagnosed"]
        ):
            return

        # if currently have symptoms of TB, refer for screening/testing
        persons_symptoms = self.sim.modules["SymptomManager"].has_what(person_id)
        if any(x in self.module.symptom_list for x in persons_symptoms):

            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_ScreeningAndRefer(person_id=person_id, module=self.module),
                topen=self.sim.date,
                tclose=self.sim.date + pd.DateOffset(days=14),
                priority=0,
            )

        else:
            # Check/log use of consumables, and give IPT if available
            # if not available, reschedule IPT start
            if self.get_consumables(
                item_codes=self.module.item_codes_for_consumables_required["tb_ipt"]
            ):
                # Update properties
                df.at[person_id, "tb_on_ipt"] = True
                df.at[person_id, "tb_date_ipt"] = self.sim.date

                # schedule decision to continue or end IPT after 6 months
                self.sim.schedule_event(
                    Tb_DecisionToContinueIPT(self.module, person_id),
                    self.sim.date + DateOffset(months=6),
                )

            else:
                # Reschedule this HSI to occur again, up to a 3 times in total
                if self.number_of_occurrences < 3:
                    self.sim.modules["HealthSystem"].schedule_hsi_event(
                        self,
                        topen=self.sim.date + pd.DateOffset(days=1),
                        tclose=self.sim.date + pd.DateOffset(days=14),
                        priority=0,
                    )


class Tb_DecisionToContinueIPT(Event, IndividualScopeEventMixin):
    """Helper event that is used to 'decide' if someone on IPT should continue or end
    This event is scheduled by 'HSI_Tb_Start_or_Continue_Ipt' after 6 months

    * end IPT for all
    * schedule further IPT for HIV+ if still eligible (no active TB diagnosed, <36 months IPT)
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id]
        m = self.module

        if not (person["is_alive"]):
            return

        # default update properties for all
        df.at[person_id, "tb_on_ipt"] = False

        # decide whether PLHIV will continue
        if (
            person["hv_diagnosed"]
            and (not person["tb_diagnosed"])
            and (person["tb_date_ipt"] < (self.sim.date - pd.DateOffset(days=36 * 30.5)))
            and (m.rng.random_sample() < m.parameters["prob_retained_ipt_6_months"])
        ):
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_Start_or_Continue_Ipt(person_id=person_id, module=m),
                topen=self.sim.date,
                tclose=self.sim.date + pd.DateOffset(days=14),
                priority=0,
            )


# ---------------------------------------------------------------------------
#   Deaths
# ---------------------------------------------------------------------------


class TbDeathEvent(Event, IndividualScopeEventMixin):
    """
    The scheduled death for a tb case
    check whether this death should occur using a linear model
    will depend on treatment status, smear status and age
    """

    def __init__(self, module, person_id, cause):
        super().__init__(module, person_id=person_id)
        self.cause = cause

    def apply(self, person_id):
        df = self.sim.population.props

        if not df.at[person_id, "is_alive"]:
            return

        if not df.at[person_id, "tb_inf"] == "active":
            return

        logger.debug(
            key="message",
            data=f"TbDeathEvent: checking whether death should occur for person {person_id}",
        )

        # use linear model to determine whether this person will die:
        # todo change death rate for treated to 10.7%
        rng = self.module.rng
        result = self.module.lm["death_rate"].predict(df.loc[[person_id]], rng=rng)

        if result:
            logger.debug(
                key="message",
                data=f"TbDeathEvent: cause this death for person {person_id}",
            )

            self.sim.modules["Demography"].do_death(
                individual_id=person_id,
                cause=self.cause,
                originating_module=self.module,
            )


# ---------------------------------------------------------------------------
#   Logging
# ---------------------------------------------------------------------------


class TbLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """produce some outputs to check"""
        # run this event every 12 months
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props
        now = self.sim.date

        # ------------------------------------ INCIDENCE ------------------------------------
        # total number of new active cases in last year - ds + mdr
        # may have died in the last year but still counted as active case for the year

        # number of new active cases
        new_tb_cases = len(
            df[(df.tb_date_active >= (now - DateOffset(months=self.repeat)))]
        )
        # # todo remove
        print("active", new_tb_cases)
        scheduled_tb_cases = len(
            df[(df.tb_scheduled_date_active >= (now - DateOffset(months=self.repeat)))]
        )
        print("scheduled", scheduled_tb_cases)

        # number of latent cases
        new_latent_cases = len(
            df[(df.tb_date_latent >= (now - DateOffset(months=self.repeat)))]
        )

        # number of new active cases in HIV+
        inc_active_hiv = len(
            df[
                (df.tb_date_active >= (now - DateOffset(months=self.repeat)))
                & df.hv_inf
                ]
        )

        # proportion of active TB cases in the last year who are HIV-positive
        prop_hiv = inc_active_hiv / new_tb_cases if new_tb_cases else 0

        logger.info(
            key="tb_incidence",
            description="Number new active and latent TB cases, total and in PLHIV",
            data={
                "num_new_active_tb": new_tb_cases,
                "num_new_latent_tb": new_latent_cases,
                "num_new_active_tb_in_hiv": inc_active_hiv,
                "prop_active_tb_in_plhiv": prop_hiv,
            },
        )

        # save outputs to dict for calibration
        self.module.tb_outputs["date"] += [self.sim.date.year]
        self.module.tb_outputs["num_new_active_tb"] += [new_tb_cases]

        # ------------------------------------ PREVALENCE ------------------------------------
        # number of current active cases divided by population alive

        # ACTIVE
        num_active_tb_cases = len(df[(df.tb_inf == "active") & df.is_alive])
        prev_active = num_active_tb_cases / len(df[df.is_alive])

        assert prev_active <= 1

        # prevalence of active TB in adults
        num_active_adult = len(
            df[(df.tb_inf == "active") & (df.age_years >= 15) & df.is_alive]
        )
        prev_active_adult = num_active_adult / len(
            df[(df.age_years >= 15) & df.is_alive]
        ) if len(
            df[(df.age_years >= 15) & df.is_alive]
        ) else 0
        assert prev_active_adult <= 1

        # prevalence of active TB in children
        num_active_child = len(
            df[(df.tb_inf == "active") & (df.age_years < 15) & df.is_alive]
        )
        prev_active_child = num_active_child / len(
            df[(df.age_years < 15) & df.is_alive]
        ) if len(
            df[(df.age_years < 15) & df.is_alive]
        ) else 0
        assert prev_active_child <= 1

        # LATENT
        # proportion of population with latent TB - all pop
        num_latent = len(df[(df.tb_inf == "latent") & df.is_alive])
        prev_latent = num_latent / len(df[df.is_alive])
        assert prev_latent <= 1

        # proportion of population with latent TB - adults
        num_latent_adult = len(
            df[(df.tb_inf == "latent") & (df.age_years >= 15) & df.is_alive]
        )
        prev_latent_adult = num_latent_adult / len(
            df[(df.age_years >= 15) & df.is_alive]
        ) if len(
            df[(df.age_years >= 15) & df.is_alive]
        ) else 0
        assert prev_latent_adult <= 1

        # proportion of population with latent TB - children
        num_latent_child = len(
            df[(df.tb_inf == "latent") & (df.age_years < 15) & df.is_alive]
        )
        prev_latent_child = num_latent_child / len(
            df[(df.age_years < 15) & df.is_alive]
        ) if len(
            df[(df.age_years < 15) & df.is_alive]
        ) else 0
        assert prev_latent_child <= 1

        logger.info(
            key="tb_prevalence",
            description="Prevalence of active and latent TB cases, total and in PLHIV",
            data={
                "tbPrevActive": prev_active,
                "tbPrevActiveAdult": prev_active_adult,
                "tbPrevActiveChild": prev_active_child,
                "tbPrevLatent": prev_latent,
                "tbPrevLatentAdult": prev_latent_adult,
                "tbPrevLatentChild": prev_latent_child,
            },
        )

        # save outputs to dict for calibration
        self.module.tb_outputs["tbPrevLatent"] += [prev_latent]

        # ------------------------------------ MDR ------------------------------------
        # number new mdr tb cases
        new_mdr_cases = len(
            df[
                (df.tb_strain == "mdr")
                & (df.tb_date_active >= (now - DateOffset(months=self.repeat)))
                ]
        )
        # # todo remove
        # print("mdr", new_mdr_cases)

        if new_mdr_cases:
            prop_mdr = new_mdr_cases / new_tb_cases
        else:
            prop_mdr = 0

        logger.info(
            key="tb_mdr",
            description="Incidence of new active MDR cases and the proportion of TB cases that are MDR",
            data={
                "tbNewActiveMdrCases": new_mdr_cases,
                "tbPropActiveCasesMdr": prop_mdr,
            },
        )

        # ------------------------------------ CASE NOTIFICATIONS ------------------------------------
        # number diagnoses (new, relapse, reinfection) in last timeperiod
        new_tb_diagnosis = len(
            df[
                (df.tb_date_active >= (now - DateOffset(months=self.repeat)))
                & (df.tb_date_diagnosed >= (now - DateOffset(months=self.repeat)))]
        )

        if new_tb_diagnosis:
            prop_dx = new_tb_diagnosis / new_tb_cases
        else:
            prop_dx = 0

        # ------------------------------------ TREATMENT ------------------------------------
        # number of tb cases who became active in last timeperiod and initiated treatment
        new_tb_tx = len(
            df[
                (df.tb_date_active >= (now - DateOffset(months=self.repeat)))
                & (df.tb_date_treated >= (now - DateOffset(months=self.repeat)))
                ]
        )

        # treatment coverage: if became active and was treated in last timeperiod
        if new_tb_cases:
            tx_coverage = new_tb_tx / new_tb_cases
            # assert tx_coverage <= 1
        else:
            tx_coverage = 0

        # ipt coverage
        new_tb_ipt = len(
            df[
                (df.tb_date_ipt >= (now - DateOffset(months=self.repeat)))
            ]
        )

        # this will give ipt among whole population - not just eligible pop
        if new_tb_ipt:
            ipt_coverage = new_tb_ipt / len(df[df.is_alive])
        else:
            ipt_coverage = 0

        logger.info(
            key="tb_treatment",
            description="TB treatment coverage",
            data={
                "tbNewDiagnosis": new_tb_diagnosis,
                "tbPropDiagnosed": prop_dx,
                "tbTreatmentCoverage": tx_coverage,
                "tbIptCoverage": ipt_coverage,
            },
        )

        # ------------------------------------ TREATMENT DELAYS ------------------------------------
        # for every person initiated on treatment, record time from onset to treatment
        # each year a series of intervals in days (treatment date - onset date) are recorded
        # convert to list
        # this will include false positives as Nan or negative or delay > 3 years

        # adults
        # get index of adults starting tx in last time-period
        # note tb onset may have been up to 3 years prior to treatment
        adult_tx_idx = df.loc[(df.age_years >= 16) &
                              (df.tb_date_treated >= (now - DateOffset(months=self.repeat)))].index

        # calculate treatment_date - onset_date for each person in index
        adult_tx_delays = (df.loc[adult_tx_idx, "tb_date_treated"] - df.loc[adult_tx_idx, "tb_date_active"]).dt.days
        adult_tx_delays = adult_tx_delays.tolist()

        # children
        child_tx_idx = df.loc[(df.age_years < 16) &
                              (df.tb_date_treated >= (now - DateOffset(months=self.repeat)))].index
        child_tx_delays = (df.loc[child_tx_idx, "tb_date_treated"] - df.loc[child_tx_idx, "tb_date_active"]).dt.days
        child_tx_delays = child_tx_delays.tolist()

        logger.info(
            key="tb_treatment_delays",
            description="TB time from onset to treatment",
            data={
                "tbTreatmentDelayAdults": adult_tx_delays,
                "tbTreatmentDelayChildren": child_tx_delays,
            },
        )

        # ------------------------------------ FALSE POSITIVES ------------------------------------
        # from the numbers on treatment, extract those who did not have active TB infection
        # they will be diagnosed as positive, but tb_inf != active
        # proportion of new treatments which are false positives

        # adults
        # tb_date_active is not within last 3 years (or pd.NaT)
        adult_num_false_positive = len(
            df[
                ~(df.tb_date_active >= (now - DateOffset(months=36)))
                & (df.tb_date_treated >= (now - DateOffset(months=self.repeat)))
                & (df.age_years >= 16)
                ]
        )

        # these are all new adults treated, regardless of tb status
        new_tb_tx_adult = len(
            df[
                (df.tb_date_treated >= (now - DateOffset(months=self.repeat)))
                & (df.age_years >= 16)
                ]
        )

        # proportion of adults starting on treatment who are false positive
        if adult_num_false_positive:
            adult_prop_false_positive = adult_num_false_positive / new_tb_tx_adult
        else:
            adult_prop_false_positive = 0

        # children
        child_num_false_positive = len(
            df[
                ~(df.tb_date_active >= (now - DateOffset(months=36)))
                & (df.tb_date_treated >= (now - DateOffset(months=self.repeat)))
                & (df.age_years < 16)
                ]
        )

        # these are all new children treated, regardless of tb status
        new_tb_tx_child = len(
            df[
                (df.tb_date_treated >= (now - DateOffset(months=self.repeat)))
                & (df.age_years < 16)
                ]
        )

        # proportion of children starting on treatment who are false positive
        if child_num_false_positive:
            child_prop_false_positive = child_num_false_positive / new_tb_tx_child
        else:
            child_prop_false_positive = 0

        logger.info(
            key="tb_false_positive",
            description="TB numbers on treatment without disease",
            data={
                "tbNumFalsePositiveAdults": adult_num_false_positive,
                "tbNumFalsePositiveChildren": child_num_false_positive,
                "tbPropFalsePositiveAdults": adult_prop_false_positive,
                "tbPropFalsePositiveChildren": child_prop_false_positive,
            },
        )


# ---------------------------------------------------------------------------
#   Debugging / Checking Events
# ---------------------------------------------------------------------------


class TbCheckPropertiesEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # runs every month

    def apply(self, population):
        self.module.check_config_of_properties()


# ---------------------------------------------------------------------------
#   Dummy Version of the Module
# ---------------------------------------------------------------------------


class DummyTbModule(Module):
    """Dummy TB Module - it's only job is to create and maintain the 'tb_inf' property.
    This can be used in test files."""

    INIT_DEPENDENCIES = {"Demography"}
    ALTERNATIVE_TO = {"Tb"}

    PROPERTIES = {
        "tb_inf": Property(
            Types.CATEGORICAL,
            categories=[
                "uninfected",
                "latent",
                "active",
            ],
            description="tb status",
        ),
    }

    def __init__(self, name=None, active_tb_prev=0.001):
        super().__init__(name)
        self.active_tb_prev = active_tb_prev

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        df = population.props

        tb_idx = df.index[
            df.is_alive & (self.rng.random_sample(len(df.is_alive)) < self.active_tb_prev)
            ]
        df.loc[tb_idx, "tb_inf"] = "active"

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother, child):
        child_infected = (self.rng.random_sample() < self.active_tb_prev)
        if child_infected:
            self.sim.population.props.at[child, "tb_inf"] = "active"
