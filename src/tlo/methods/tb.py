"""
The TB Module is under development. Here the contents of the module are commented out whilst work proceeds. It is
included because this version should replace what is currently in Master.
"""

from tlo import Module, logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
import os

import pandas as pd

from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography, hiv
from tlo.methods.healthsystem import HSI_Event
"""


class Tb(Module):
    """ Set up the baseline population with TB prevalence
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
    """
    PARAMETERS = {
        # workbooks
        "prop_latent_2010": Parameter(
            Types.REAL, "Proportion of population with latent tb in 2010"
        ),
        "prop_active_2010": Parameter(
            Types.REAL, "Proportion of population with active tb in 2010"
        ),
        "pulm_tb": Parameter(Types.REAL, "probability of pulmonary tb"),
        "followup_times": Parameter(
            Types.INT, "times(weeks) tb treatment monitoring required after tx start"
        ),
        "tb_high_risk_distr": Parameter(Types.LIST, "list of ten high-risk districts"),
        "ipt_contact_cov": Parameter(
            Types.REAL,
            "coverage of IPT among contacts of TB cases in high-risk districts",
        ),
        "bcg_coverage_year": Parameter(
            Types.REAL, "bcg coverage estimates in children <1 years by calendar year"
        ),
        "initial_bcg_coverage": Parameter(
            Types.REAL, "bcg coverage by age in baseline population"
        ),
        # baseline population
        "prop_mdr2010": Parameter(
            Types.REAL,
            "Proportion of active tb cases with multidrug resistance in 2010",
        ),
        # natural history
        "transmission_rate": Parameter(Types.REAL, "TB transmission rate, calibrated"),
        "rel_inf_smear_ng": Parameter(
            Types.REAL, "relative infectiousness of tb in hiv+ compared with hiv-"
        ),
        "rel_inf_poor_tx": Parameter(
            Types.REAL,
            "relative infectiousness of tb in treated people with poor adherence",
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
        "r_trans_mdr": Parameter(
            Types.REAL,
            "relative transmissibility of MDR compared with drug-susceptible TB",
        ),
        "p_mdr_new": Parameter(
            Types.REAL,
            "probability of MDR emergence per treatment course – new patients",
        ),
        "p_mdr_retreat": Parameter(
            Types.REAL,
            "multiplier for probability of MDR emergence per treatment course – re-treatment",
        ),
        "p_mdr_tx_fail": Parameter(
            Types.REAL,
            "multiplier for probability of MDR emergence per treatment course – treatment failure",
        ),
        # progression
        "prop_fast_progressor": Parameter(
            Types.REAL,
            "Proportion of infections that progress directly to active stage",
        ),
        "prop_fast_progressor_hiv": Parameter(
            Types.REAL,
            "proportion of HIV+ people not on ART progressing directly to active TB disease after infection",
        ),
        "prog_active": Parameter(
            Types.REAL, "risk of progressing to active tb within two years"
        ),
        "prog_1yr": Parameter(
            Types.REAL, "proportion children aged <1 year progressing to active disease"
        ),
        "prog_1_2yr": Parameter(
            Types.REAL,
            "proportion children aged 1-2 year2 progressing to active disease",
        ),
        "prog_2_5yr": Parameter(
            Types.REAL,
            "proportion children aged 2-5 years progressing to active disease",
        ),
        "prog_5_10yr": Parameter(
            Types.REAL,
            "proportion children aged 5-10 years progressing to active disease",
        ),
        "prog_10yr": Parameter(
            Types.REAL,
            "proportion children aged 10-15 years progressing to active disease",
        ),
        "monthly_prob_self_cure": Parameter(
            Types.REAL, "monthly probability of self-cure"
        ),
        "monthly_prob_self_cure_hiv": Parameter(
            Types.REAL, "monthly probability of self-cure in plhiv"
        ),
        # clinical features
        "prop_smear_positive": Parameter(
            Types.REAL, "proportion of new active cases that will be smear-positive"
        ),
        "prop_smear_positive_hiv": Parameter(
            Types.REAL, "proportion of hiv+ active tb cases that will be smear-positive"
        ),
        # mortality
        "monthly_prob_tb_mortality": Parameter(
            Types.REAL, "mortality rate with active tb"
        ),
        "monthly_prob_tb_mortality_hiv": Parameter(
            Types.REAL, "mortality from tb with concurrent HIV"
        ),
        "mort_cotrim": Parameter(
            Types.REAL, "reduction in mortality rates due to cotrimoxazole prophylaxis"
        ),
        "mort_tx": Parameter(
            Types.REAL, "reduction in mortality rates due to effective tb treatment"
        ),
        # relative risks of progression to active disease
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
        "rr_tb_overweight": Parameter(
            Types.REAL, "relative risk of progression to active disease if overweight"
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
        "dur_prot_ipt": Parameter(
            Types.REAL,
            "duration in days of protection conferred by IPT against active TB",
        ),
        "dur_prot_ipt_infant": Parameter(
            Types.REAL,
            "duration days of protection conferred by IPT against active TB in infants",
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
        # health system interactions
        "sens_xpert": Parameter(Types.REAL, "sensitivity of Xpert test"),
        "sens_sputum_pos": Parameter(
            Types.REAL,
            "sensitivity of sputum smear microscopy in sputum positive cases",
        ),
        "sens_sputum_neg": Parameter(
            Types.REAL,
            "sensitivity of sputum smear microscopy in sputum negative cases",
        ),
        "sens_clinical": Parameter(
            Types.REAL, "sensitivity of clinical diagnosis in detecting active TB"
        ),
        "spec_clinical": Parameter(
            Types.REAL, "specificity of clinical diagnosis in detecting TB"
        ),
        "prob_tx_success_new": Parameter(
            Types.REAL, "Probability of treatment success for new TB cases"
        ),
        "prob_tx_success_prev": Parameter(
            Types.REAL, "Probability of treatment success for previously treated cases"
        ),
        "prob_tx_success_hiv": Parameter(
            Types.REAL, "Probability of treatment success for PLHIV"
        ),
        "prob_tx_success_mdr": Parameter(
            Types.REAL, "Probability of treatment success for MDR-TB cases"
        ),
        "prob_tx_success_extra": Parameter(
            Types.REAL, "Probability of treatment success for extrapulmonary TB cases"
        ),
        "prob_tx_success_0_4": Parameter(
            Types.REAL, "Probability of treatment success for children aged 0-4 years"
        ),
        "prob_tx_success_5_14": Parameter(
            Types.REAL, "Probability of treatment success for children aged 5-14 years"
        ),
        "prop_ltfu_tx": Parameter(
            Types.REAL, "Proportion lost to follow-up during initial treatment"
        ),
        "prop_ltfu_retx": Parameter(
            Types.REAL, "Proportion lost to follow-up during retreatment"
        ),
        "rate_testing_tb": Parameter(
            Types.REAL,
            "rate of presenting for TB screening and testing for people with active TB",
        ),
        "rr_testing_non_tb": Parameter(
            Types.REAL,
            "rate ratio for TB testing without active TB compared with ative TB cases",
        ),
        "rate_testing_hiv": Parameter(
            Types.REAL,
            "rate of presenting for TB screening and testing for HIV-positive people with active TB",
        ),
        "presump_testing": Parameter(
            Types.REAL, "probability of an individual without tb requesting tb test"
        ),
        # daly weights, no daly weight for latent tb
        "daly_wt_susc_tb": Parameter(
            Types.REAL, "Drug-susecptible tuberculosis, not HIV infected"
        ),
        "daly_wt_resistant_tb": Parameter(
            Types.REAL, "multidrug-resistant tuberculosis, not HIV infected"
        ),
        "daly_wt_susc_tb_hiv_severe_anaemia": Parameter(
            Types.REAL,
            "# Drug-susecptible Tuberculosis, HIV infected and anemia, severe",
        ),
        "daly_wt_susc_tb_hiv_moderate_anaemia": Parameter(
            Types.REAL,
            "Drug-susecptible Tuberculosis, HIV infected and anemia, moderate",
        ),
        "daly_wt_susc_tb_hiv_mild_anaemia": Parameter(
            Types.REAL, "Drug-susecptible Tuberculosis, HIV infected and anemia, mild"
        ),
        "daly_wt_susc_tb_hiv": Parameter(
            Types.REAL, "Drug-susecptible Tuberculosis, HIV infected"
        ),
        "daly_wt_resistant_tb_hiv_severe_anaemia": Parameter(
            Types.REAL,
            "Multidrug resistant Tuberculosis, HIV infected and anemia, severe",
        ),
        "daly_wt_resistant_tb_hiv": Parameter(
            Types.REAL, "Multidrug resistant Tuberculosis, HIV infected"
        ),
        "daly_wt_resistant_tb_hiv_moderate_anaemia": Parameter(
            Types.REAL,
            "Multidrug resistant Tuberculosis, HIV infected and anemia, moderate",
        ),
        "daly_wt_resistant_tb_hiv_mild_anaemia": Parameter(
            Types.REAL,
            "Multidrug resistant Tuberculosis, HIV infected and anemia, mild",
        ),
    }

    PROPERTIES = {
        "tb_inf": Property(
            Types.CATEGORICAL,
            categories=[
                "uninfected",
                "latent_susc_new",
                "active_susc_new",
                "latent_susc_tx",
                "active_susc_tx",
                "latent_mdr_new",
                "active_mdr_new",
                "latent_mdr_tx",
                "active_mdr_tx",
            ],
            description="tb status",
        ),
        "tb_date_active": Property(Types.DATE, "Date active tb started"),
        "tb_any_resp_symptoms": Property(Types.DATE, "Date of symptom start"),
        "tb_smear": Property(Types.BOOL, "smear positivity with active infection"),
        "tb_date_latent": Property(
            Types.DATE, "Date acquired tb infection (latent stage)"
        ),
        "tb_ever_tb": Property(Types.BOOL, "if ever had active drug-susceptible tb"),
        "tb_ever_tb_mdr": Property(
            Types.BOOL, "if ever had active multi-drug resistant tb"
        ),
        "tb_stage": Property(
            Types.CATEGORICAL,
            "Level of symptoms for tb",
            categories=["none", "latent", "active_pulm", "active_extra"],
        ),
        "tb_ever_tested": Property(Types.BOOL, "ever had a tb test"),
        "tb_smear_test": Property(Types.BOOL, "ever had a tb smear test"),
        "tb_result_smear_test": Property(Types.BOOL, "result from tb smear test"),
        "tb_date_smear_test": Property(Types.DATE, "date of tb smear test"),
        "tb_xpert_test": Property(Types.BOOL, "ever had a tb Xpert test"),
        "tb_result_xpert_test": Property(Types.BOOL, "result from tb Xpert test"),
        "tb_date_xpert_test": Property(Types.DATE, "date of tb Xpert test"),
        "tb_diagnosed": Property(Types.BOOL, "current diagnosis of active tb"),
        "tb_mdr_diagnosed": Property(Types.BOOL, "current diagnosis of active tb_mdr"),
        "tb_on_treatment": Property(Types.BOOL, "on tb treatment regimen"),
        "tb_date_treated": Property(Types.DATE, "date tb treatment started"),
        "tb_treatment_failure": Property(Types.BOOL, "failed first line tb treatment"),
        "tb_treated_mdr": Property(Types.BOOL, "on tb treatment MDR regimen"),
        "tb_date_treated_mdr": Property(Types.DATE, "date tb MDR treatment started"),
        "tb_on_ipt": Property(Types.BOOL, "if currently on ipt"),
        "tb_date_ipt": Property(Types.DATE, "date ipt started"),
        "tb_date_death": Property(Types.DATE, "date of tb death"),
        "tb_date_death_occurred": Property(Types.DATE, "date of tb death"),
        "tb_bcg": Property(Types.BOOL, "received BCG vaccination"),
        "tb_symptoms": Property(Types.BOOL, "tb-like symptoms present"),
    }

    # cough and fever are part of generic symptoms
    SYMPTOMS = {"fatigue", "night_sweats"}

    """
    def read_parameters(self, data_folder):
        pass
        """
        workbook = pd.read_excel(
            os.path.join(self.resourcefilepath, "ResourceFile_TB.xlsx"), sheet_name=None
        )
        self.load_parameters_from_dataframe(workbook["parameters"])

        p = self.parameters

        # workbooks
        p["prop_active_2010"], p["prop_latent_2010"] = (
            workbook["cases2010district"],
            workbook["Latent_TB_prob"],
        )
        p["pulm_tb"] = workbook["pulm_tb"]
        p["followup_times"] = workbook["followup"]
        p["tb_high_risk_distr"] = workbook["IPTdistricts"]
        p["ipt_contact_cov"] = workbook["ipt_coverage"]
        p["bcg_coverage_year"] = workbook["BCG"]
        p["initial_bcg_coverage"] = workbook["BCG_baseline"]

        # get the DALY weight that this module will use from the weight database
        if "HealthBurden" in self.sim.modules.keys():
            p["daly_wt_susc_tb"] = self.sim.modules["HealthBurden"].get_daly_weight(
                0
            )  # Drug-susecptible tuberculosis, not HIV infected
            p["daly_wt_resistant_tb"] = self.sim.modules[
                "HealthBurden"
            ].get_daly_weight(
                1
            )  # multidrug-resistant tuberculosis, not HIV infected
            p["daly_wt_susc_tb_hiv_severe_anaemia"] = self.sim.modules[
                "HealthBurden"
            ].get_daly_weight(
                4
            )  # Drug-susecptible Tuberculosis, HIV infected and anemia, severe
            p["daly_wt_susc_tb_hiv_moderate_anaemia"] = self.sim.modules[
                "HealthBurden"
            ].get_daly_weight(
                5
            )  # Drug-susecptible Tuberculosis, HIV infected and anemia, moderate
            p["daly_wt_susc_tb_hiv_mild_anaemia"] = self.sim.modules[
                "HealthBurden"
            ].get_daly_weight(
                6
            )  # Drug-susecptible Tuberculosis, HIV infected and anemia, mild
            p["daly_wt_susc_tb_hiv"] = self.sim.modules["HealthBurden"].get_daly_weight(
                7
            )  # Drug-susecptible Tuberculosis, HIV infected
            p["daly_wt_resistant_tb_hiv_severe_anaemia"] = self.sim.modules[
                "HealthBurden"
            ].get_daly_weight(
                8
            )  # Multidrug resistant Tuberculosis, HIV infected and anemia, severe
            p["daly_wt_resistant_tb_hiv"] = self.sim.modules[
                "HealthBurden"
            ].get_daly_weight(
                9
            )  # Multidrug resistant Tuberculosis, HIV infected
            p["daly_wt_resistant_tb_hiv_moderate_anaemia"] = self.sim.modules[
                "HealthBurden"
            ].get_daly_weight(
                10
            )  # Multidrug resistant Tuberculosis, HIV infected and anemia, moderate
            p["daly_wt_resistant_tb_hiv_mild_anaemia"] = self.sim.modules[
                "HealthBurden"
            ].get_daly_weight(
                11
            )  # Multidrug resistant Tuberculosis, HIV infected and anemia, mild
        """

    def initialise_population(self, population):
        pass
        """
        Set our property values for the initial population.

        df = population.props
        now = self.sim.date
        params = self.parameters

        # set-up baseline population
        df["tb_inf"].values[:] = "uninfected"
        df["tb_date_active"] = pd.NaT
        df["tb_any_resp_symptoms"] = pd.NaT
        df["tb_smear"] = False
        df["tb_date_latent"] = pd.NaT

        df["tb_ever_tb"] = False
        df["tb_ever_tb_mdr"] = False

        df["tb_stage"].values[:] = "none"

        df["tb_ever_tested"] = False  # default: no individuals tested
        df["tb_smear_test"] = False
        df["tb_result_smear_test"] = False
        df["tb_date_smear_test"] = pd.NaT
        df["tb_xpert_test"] = False
        df["tb_result_xpert_test"] = False
        df["tb_date_xpert_test"] = pd.NaT
        df["tb_diagnosed"] = False
        df["tb_mdr_diagnosed"] = False
        df["tb_on_treatment"] = False
        df["tb_date_treated"] = pd.NaT
        df["tb_treatment_failure"] = False
        df["tb_treated_mdr"] = False
        df["tb_date_treated_mdr"] = pd.NaT
        df["tb_on_ipt"] = False
        df["tb_date_ipt"] = pd.NaT
        df["tb_date_death"] = pd.NaT
        df["tb_date_death_occurred"] = pd.NaT
        df["tb_bcg"] = False
        df["tb_symptoms"] = False

        self.bcg(population)  # assign bcg vaccine to baseline population

        # TB infections - active / latent
        # baseline infections not weighted by RR, randomly assigned
        active_tb_data = self.parameters["prop_active_2010"]
        latent_tb_data = self.parameters["prop_latent_2010"]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ LATENT ~~~~~~~~~~~~~~~~~~~~~~~~~~

        # merge all susceptible individuals with their hiv probability based on sex and age
        df_tbprob = df.merge(
            latent_tb_data,
            left_on=["age_years", "sex"],
            right_on=["age", "sex"],
            how="left",
        )

        # fill missing values with 0 (only relevant for age 80+)
        df_tbprob["prob_latent_tb"] = df_tbprob["prob_latent_tb"].fillna(0)

        assert (
            df_tbprob.prob_latent_tb.isna().sum() == 0
        )  # check there is a probability for every individual

        # assign risk of latent tb
        risk_tb = pd.Series(1, index=df.index)
        risk_tb.loc[df.is_alive & df.tb_bcg & (df.age_years < 10)] *= params[
            "rr_bcg_inf"
        ]

        # weight the likelihood of being sampled by the relative risk
        # norm_p = pd.Series(risk_tb)
        # norm_p /= norm_p.sum()  # normalise

        # get a list of random numbers between 0 and 1 for each infected individual
        random_draw = self.rng.random_sample(size=len(df_tbprob))
        tb_idx = df_tbprob.index[df.is_alive & (random_draw < df_tbprob.prob_latent_tb)]
        # tb_idx = df_tbprob.index[df.is_alive & (random_draw < (df_tbprob.prob_latent_tb * norm_p))]

        df.loc[tb_idx, "tb_inf"] = "latent_susc_new"
        df.loc[tb_idx, "tb_date_latent"] = now
        df.loc[tb_idx, "tb_stage"] = "latent"

        # allocate some latent infections as mdr-tb
        if len(df[df.is_alive & (df.tb_inf == "latent_susc_new")]) > 10:
            idx_c = (
                df[df.is_alive & (df.tb_inf == "latent_susc_new")]
                .sample(frac=self.parameters["prop_mdr2010"])
                .index
            )

            df.loc[idx_c, "tb_inf"] = "latent_mdr_new"  # change to mdr-tb
            df.loc[idx_c, "tb_stage"] = "latent"

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ ACTIVE ~~~~~~~~~~~~~~~~~~~~~~~~~~
        # prob of active case by district
        df_active_prob = df.merge(
            active_tb_data,
            left_on=["district_of_residence"],
            right_on=["district"],
            how="left",
        )
        assert (
            df_active_prob.general_prob.isna().sum() == 0
        )  # check there is a probability for every individual

        # get a list of random numbers between 0 and 1 for each infected individual
        random_draw = self.rng.random_sample(size=len(df_active_prob))
        active_idx = df_active_prob.index[
            df.is_alive & (random_draw < df_active_prob.general_prob)
        ]

        # if >10 active cases, sample some mdr cases
        if len(active_idx) > 10:

            # sample from active to get mdr-tb cases
            idx = df.loc[active_idx].sample(frac=self.parameters["prop_mdr2010"]).index

            # if some mdr-cases
            if not idx.isnull().all():

                for person_id in active_idx:
                    # print(person_id)
                    # random draw of days 0-365
                    random_day = self.rng.randint(low=0, high=364)
                    pd_day = now + pd.to_timedelta(random_day, unit="d")

                    # if not sampled for mdr, schedule onset active drug-susceptible
                    if person_id not in idx:

                        logger.debug(
                            "This is Tb, scheduling active disease for person %d on date %s",
                            person_id,
                            pd_day,
                        )

                        # schedule active disease
                        self.sim.schedule_event(TbActiveEvent(self, person_id), pd_day)

                    else:
                        logger.debug(
                            "This is Tb, scheduling mdr-active disease for person %d on date %s",
                            person_id,
                            pd_day,
                        )

                        # schedule active disease
                        self.sim.schedule_event(
                            TbMdrActiveEvent(self, person_id), pd_day
                        )

            # if no mdr cases have been sampled
            else:
                for person_id in active_idx:
                    # print(person_id)
                    # random draw of days 0-365
                    random_day = self.rng.randint(low=0, high=364)
                    pd_day = now + pd.to_timedelta(random_day, unit="d")

                    logger.debug(
                        "This is Tb, scheduling active disease for person %d on date %s",
                        person_id,
                        pd_day,
                    )

                    # schedule active disease
                    self.sim.schedule_event(TbActiveEvent(self, person_id), pd_day)

        # if <10 active cases
        else:
            for person_id in active_idx:
                # print(person_id)
                # random draw of days 0-365
                random_day = self.rng.randint(low=0, high=364)
                pd_day = now + pd.to_timedelta(random_day, unit="d")

                logger.debug(
                    "This is Tb, scheduling active disease for person %d on date %s",
                    person_id,
                    pd_day,
                )

                # schedule active disease
                self.sim.schedule_event(TbActiveEvent(self, person_id), pd_day)
        """

    def bcg(self, population):
        pass
        """
        Assign BCG vaccination to baseline population
        may have already been assigned active TB, should be ok as bcg not 100% protective

        df = population.props
        params = self.parameters

        bcg_cov = params["initial_bcg_coverage"]

        df_bcg = df.merge(bcg_cov, left_on=["age_years"], right_on=["Age"], how="left")

        # fill missing values with 0 (only relevant for age 80+)
        df_bcg["bcg_coverage"] = df_bcg["bcg_coverage"].fillna(0)
        assert (
            df_bcg.bcg_coverage.isna().sum() == 0
        )  # check there is a probability for every individual

        # get a list of random numbers between 0 and 1 for each infected individual
        random_draw = self.rng.random_sample(size=len(df_bcg))

        # probability of bcg > random number, assign bcg = True
        bcg_idx = df_bcg.index[df.is_alive & (random_draw < df_bcg.bcg_coverage)]

        df.loc[bcg_idx, "tb_bcg"] = True
        """

    def initialise_simulation(self, sim):
        pass
        """
        sim.schedule_event(TbEvent(self), sim.date + DateOffset(months=12))
        sim.schedule_event(TbRelapseEvent(self), sim.date + DateOffset(months=1))
        sim.schedule_event(TbSelfCureEvent(self), sim.date + DateOffset(months=1))

        sim.schedule_event(TbMdrEvent(self), sim.date + DateOffset(months=12))
        sim.schedule_event(TbMdrRelapseEvent(self), sim.date + DateOffset(months=1))
        sim.schedule_event(TbMdrSelfCureEvent(self), sim.date + DateOffset(months=1))

        # sim.schedule_event(TbScheduleTesting(self), sim.date + DateOffset(days=1))

        # sim.schedule_event(NonTbSymptomsEvent(self), sim.date + DateOffset(months=1))

        sim.schedule_event(TbDeathEvent(self), sim.date + DateOffset(months=1))
        sim.schedule_event(TbMdrDeathEvent(self), sim.date + DateOffset(months=1))

        # Logging
        sim.schedule_event(TbLoggingEvent(self), sim.date + DateOffset(days=0))
        """

    def on_birth(self, mother_id, child_id):
        pass
        """
        Initialise our properties for a newborn individual.

        df = self.sim.population.props
        now = self.sim.date

        df.at[child_id, "tb_inf"] = "uninfected"
        df.at[child_id, "tb_date_active"] = pd.NaT
        df.at[child_id, "tb_any_resp_symptoms"] = pd.NaT
        df.at[child_id, "tb_smear"] = False
        df.at[child_id, "tb_date_latent"] = pd.NaT
        df.at[child_id, "tb_ever_tb"] = False
        df.at[child_id, "tb_ever_tb_mdr"] = False
        df.at[child_id, "tb_stage"] = "none"

        df.at[child_id, "tb_ever_tested"] = False  # default: not tested
        df.at[child_id, "tb_smear_test"] = False
        df.at[child_id, "tb_result_smear_test"] = False
        df.at[child_id, "tb_date_smear_test"] = pd.NaT
        df.at[child_id, "tb_xpert_test"] = False
        df.at[child_id, "tb_result_xpert_test"] = False
        df.at[child_id, "tb_date_xpert_test"] = pd.NaT
        df.at[child_id, "tb_diagnosed"] = False
        df.at[child_id, "tb_mdr_diagnosed"] = False
        df.at[child_id, "tb_on_treatment"] = False
        df.at[child_id, "tb_date_treated"] = pd.NaT
        df.at[child_id, "tb_treatment_failure"] = False
        df.at[child_id, "tb_treated_mdr"] = False
        df.at[child_id, "tb_date_treated_mdr"] = pd.NaT
        df.at[child_id, "tb_on_ipt"] = False
        df.at[child_id, "tb_date_ipt"] = pd.NaT
        df.at[child_id, "tb_date_death"] = pd.NaT
        df.at[child_id, "tb_date_death_occurred"] = pd.NaT
        df.at[child_id, "tb_bcg"] = False
        df.at[child_id, "tb_symptoms"] = False

        # if mother is diagnosed with TB, give IPT to infant
        if df.at[mother_id, "tb_diagnosed"]:
            event = HSI_Tb_Ipt(self, person_id=child_id)
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                event,
                priority=1,
                topen=self.sim.date,
                tclose=self.sim.date + DateOffset(weeks=4),
            )

        # assign bcg according to current coverage
        bcg_cov_yr = self.parameters["bcg_coverage_year"]

        bcg_curr_yr = (
            bcg_cov_yr.loc[bcg_cov_yr.Year == now.year, ["bcg_coverage"]]
        ).values[0] / 100

        if self.rng.random_sample(size=1) < bcg_curr_yr:
            df.at[child_id, "tb_bcg"] = True
        """

    def on_hsi_alert(self, person_id, treatment_id):
        pass
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.

        logger.debug(
            "This is TB, being alerted about a health system interaction "
            "person %d for: %s",
            person_id,
            treatment_id,
        )

        # create list of appts which link to tb screening
        id_list = [
            "Hiv_Testing",
            "Hiv_TestingInfant",
            "Hiv_InfantTreatmentInitiation",
            "Hiv_TreatmentInitiation",
            "Hiv_Treatment",
        ]

        if treatment_id in id_list:
            piggy_back_dx_at_appt = HSI_Tb_Screening(self, person_id)
            piggy_back_dx_at_appt.TREATMENT_ID = "Tb_Screening_PiggybackAppt"

            # Arbitrarily reduce the size of appt footprint to reflect that this is a piggy back appt
            for key in piggy_back_dx_at_appt.EXPECTED_APPT_FOOTPRINT:
                piggy_back_dx_at_appt.EXPECTED_APPT_FOOTPRINT[key] = (
                    piggy_back_dx_at_appt.EXPECTED_APPT_FOOTPRINT[key] * 0.5
                )

            self.sim.modules["HealthSystem"].schedule_hsi_event(
                piggy_back_dx_at_appt, priority=0, topen=self.sim.date, tclose=None
            )
        """

    def report_daly_values(self):
        pass
        """
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.
        logger.debug("This is tb reporting my health values")

        df = self.sim.population.props  # shortcut to population properties dataframe
        params = self.parameters

        health_values = pd.Series(0, index=df.index)

        health_values.loc[
            df["tb_inf"].str.contains("active_susc") & ~df.hv_inf & df.is_alive
        ] = params["daly_wt_susc_tb"]
        health_values.loc[
            df["tb_inf"].str.contains("active_mdr") & ~df.hv_inf & df.is_alive
        ] = params["daly_wt_resistant_tb"]

        health_values.loc[
            df["tb_inf"].str.contains("active_susc") & df.hv_inf & df.is_alive
        ] = params["daly_wt_susc_tb_hiv"]
        health_values.loc[
            df["tb_inf"].str.contains("active_mdr") & df.hv_inf & df.is_alive
        ] = params["daly_wt_resistant_tb_hiv"]

        health_values.name = "tb Symptoms"  # label the cause of this disability

        return health_values.loc[df.is_alive]
    """

#
# # ---------------------------------------------------------------------------
# #   TB infection event
# # ---------------------------------------------------------------------------
# # TODO should transmission be limited within each district?
# # TODO add relative infectiousness for those on poor tx [prop tx failure]
# # TODO rel inf for smear negative - weight inf by prop smear negative
# # TODO age/sex distribution for new cases?
# class TbEvent(RegularEvent, PopulationScopeEventMixin):
#     """ tb infection events
#     """
#
#     def __init__(self, module):
#         super().__init__(module, frequency=DateOffset(months=1))  # every 1 month
#
#     def apply(self, population):
#         params = self.module.parameters
#         now = self.sim.date
#         rng = self.module.rng
#
#         df = population.props
#
#         # ----------------------------------- FORCE OF INFECTION -----------------------------------
#
#         # apply a force of infection to produce new latent cases
#         # no age distribution for FOI but the relative risks would affect distribution of active infections
#         # smear-positive cases by district
#         df["tmp"] = pd.Series(0, index=df.index)
#         df.loc[
#             df.is_alive & df["tb_inf"].str.contains("active_susc") & df.tb_smear, "tmp"
#         ] = 1
#         smear_pos = df.groupby(["district_of_residence"])[["tmp"]].sum()
#         smear_pos = smear_pos.iloc[:, 0]  # convert to series
#
#         # smear-negative cases by district
#         df["tmp2"] = pd.Series(0, index=df.index)
#         df.loc[
#             df.is_alive & df["tb_inf"].str.contains("active_susc") & ~df.tb_smear,
#             "tmp2",
#         ] = 1
#         smear_neg = df.groupby(["district_of_residence"])[["tmp2"]].sum()
#         smear_neg = smear_neg.iloc[:, 0]  # convert to series
#
#         # uninfected cases by district
#         df["tmp3"] = pd.Series(0, index=df.index)
#         df.loc[df.is_alive & (df.tb_inf == "uninfected"), "tmp3"] = 1
#         uninfected = df.groupby(["district_of_residence"])[["tmp3"]].sum()
#         uninfected = uninfected.iloc[:, 0]  # convert to series
#
#         # population by district
#         df["tmp4"] = pd.Series(0, index=df.index)
#         df.loc[df.is_alive, "tmp4"] = 1
#         pop = df.groupby(["district_of_residence"])[["tmp4"]].sum()
#         pop = pop.iloc[:, 0]  # convert to series
#
#         del df["tmp"]
#         del df["tmp2"]
#         del df["tmp3"]
#         del df["tmp4"]
#
#         # calculate foi by district
#         districts = df["district_of_residence"].unique()
#
#         # if any smear_pos or smear_neg are 0 they will return foi=0
#         smear_pos[smear_pos == 0] = 1
#         smear_neg[smear_neg == 0] = 1
#
#         foi = pd.Series(0, index=districts)
#         foi = (
#             params["transmission_rate"]
#             * smear_pos
#             * (smear_neg * params["rel_inf_smear_ng"])
#             * uninfected
#         ) / pop
#
#         assert foi.isna().sum() == 0  # check there is a foi for every district
#
#         foi_df = pd.DataFrame(foi, columns=["foi"])
#
#         # ----------------------------------- NEW LATENT INFECTIONS -----------------------------------
#         df_distr = df.merge(
#             foi_df, left_on=["district_of_residence"], right_index=True, how="left"
#         )
#
#         assert (
#             df_distr.foi.isna().sum() == 0
#         )  # check there is a district-level foi for every person
#
#         risk_tb = pd.Series(df_distr.foi, index=df.index)
#         risk_tb.loc[~df.is_alive] *= 0
#         risk_tb.loc[df.tb_bcg & df.age_years < 10] *= params["rr_bcg_inf"]
#
#         # get a list of random numbers between 0 and 1 for each infected individual
#         random_draw = rng.random_sample(size=len(df))
#
#         tb_idx = df.index[
#             df.is_alive & (df.tb_inf == "uninfected") & (random_draw < risk_tb)
#         ]
#
#         df.loc[tb_idx, "tb_inf"] = "latent_susc_new"
#         df.loc[tb_idx, "tb_date_latent"] = now
#         df.loc[tb_idx, "tb_stage"] = "latent"
#
#         logger.debug(f"TbEvent:new latent infections for ids {tb_idx}")
#
#         # ----------------------------------- RE-INFECTIONS -----------------------------------
#
#         # pop at risk = latent_susc_tx and latent_mdr (new & tx)
#         repeat_case = df.index[
#             df.is_alive & (df.tb_inf == "latent_susc_tx")
#             | df["tb_inf"].str.contains("latent_mdr") & (random_draw < risk_tb)
#         ]
#
#         # unchanged status, high risk of relapse as if just recovered
#         df.loc[repeat_case, "tb_inf"] = "latent_susc_tx"
#
#         df.loc[repeat_case, "tb_date_latent"] = now
#         df.loc[repeat_case, "tb_stage"] = "latent"
#
#         logger.debug(f"TbEvent: reinfections for ids {tb_idx}")
#
#         del df_distr
#
#         # -----------------------------------------------------------------------------------------------------
#         # PROGRESSION TO ACTIVE DISEASE
#
#         # ----------------------------------- ADULT PROGRESSORS TO ACTIVE DISEASE -----------------------------------
#
#         # probability of active disease
#         prob_prog = pd.Series(0, index=df.index)
#         prob_prog.loc[df.is_alive & df.age_years.between(15, 100)] = params[
#             "prog_active"
#         ]
#         prob_prog.loc[df.hv_inf] *= params["rr_tb_hiv"]
#         prob_prog.loc[df.hv_inf] *= params["rr_tb_aids"]
#         prob_prog.loc[(df.hv_on_art == 2)] *= (
#             params["rr_tb_hiv"] * params["rr_tb_art_adult"]
#         )
#         # prob_prog.loc[df.xxxx] *= params['rr_tb_overweight']
#         prob_prog.loc[df.li_bmi >= 4] *= params["rr_tb_obese"]
#         # prob_prog.loc[df.xxxx] *= params['rr_tb_diabetes1']
#         prob_prog.loc[df.li_ex_alc] *= params["rr_tb_alcohol"]
#         prob_prog.loc[df.li_tob] *= params["rr_tb_smoking"]
#
#         # impact of IPT
#         dur_ipt = pd.to_timedelta(params["dur_prot_ipt"], unit="d")
#
#         prob_prog.loc[(df.tb_date_ipt < (now - dur_ipt)) & ~df.hv_inf] *= params[
#             "rr_ipt_adult"
#         ]
#         prob_prog.loc[
#             (df.tb_date_ipt < (now - dur_ipt)) & df.hv_inf & (df.hv_on_art != 2)
#         ] *= params["rr_ipt_adult_hiv"]
#         prob_prog.loc[
#             (df.tb_date_ipt < (now - dur_ipt)) & df.hv_inf & (df.hv_on_art == 2)
#         ] *= params["rr_ipt_art_adult"]
#
#         # new latent cases including repeat infections
#         new_latent = len(
#             df[(df.tb_date_latent == now) & (df.age_years >= 15) & df.is_alive].index
#         )
#
#         if new_latent:
#
#             # index of all new susc cases
#             idx = df[
#                 (df.tb_date_latent == now) & (df.age_years >= 15) & df.is_alive
#             ].index
#
#             # for each person determine if fast progressor and schedule onset of active disease
#             fast = pd.Series(data=False, index=df.loc[idx].index)
#
#             for i in df.index[idx]:
#                 fast[i] = rng.rand() < params["prop_fast_progressor"]
#
#             # need to scale the prob of progressing to active so overall pop prob of
#             #  progression = params['prog_active']
#             # mean / scale = scaling factor
#             # don't include children here
#             mean = sum(prob_prog[df.is_alive & df.age_years.between(15, 100)]) / len(
#                 prob_prog[df.is_alive & df.age_years.between(15, 100)]
#             )
#             scaling_factor = mean / params["prog_active"]
#
#             prob_prog_scaled = prob_prog.divide(other=scaling_factor)
#             assert len(prob_prog) == len(prob_prog_scaled)
#
#             if fast.sum() > 0:
#                 for person_id in fast.index[fast]:
#                     # print('fast', person_id)
#
#                     logger.debug(
#                         "This is TbEvent, scheduling active disease for fast progressing person %d on date %s",
#                         person_id,
#                         now,
#                     )
#                     # schedule active disease now
#                     self.sim.schedule_event(TbActiveEvent(self.module, person_id), now)
#
#             # slow progressors
#             for person_id in idx[~fast]:
#                 # print('slow', person_id)
#                 # decide if person will develop active tb
#                 active = rng.rand() < prob_prog_scaled[person_id]
#
#                 if active:
#                     # randomly select date of onset of active disease
#                     # random draw of days 0-732
#                     random_date = rng.randint(low=0, high=732)
#                     sch_date = now + pd.to_timedelta(random_date, unit="d")
#
#                     logger.debug(
#                         "This is TbEvent, scheduling active disease for slow progressing person %d on date %s",
#                         person_id,
#                         sch_date,
#                     )
#
#                     # schedule active disease
#                     self.sim.schedule_event(
#                         TbActiveEvent(self.module, person_id), sch_date
#                     )
#
#         # ----------------------------------- CHILD PROGRESSORS TO ACTIVE DISEASE -----------------------------------
#         # probability of active disease
#         prob_prog_child = pd.Series(0, index=df.index)
#         prob_prog_child.loc[df.is_alive & (df.age_years < 1)] = params["prog_1yr"]
#         prob_prog_child.loc[df.is_alive & (df.age_years.between(1, 2))] = params[
#             "prog_1_2yr"
#         ]
#         prob_prog_child.loc[df.is_alive & (df.age_years.between(2, 5))] = params[
#             "prog_2_5yr"
#         ]
#         prob_prog_child.loc[df.is_alive & (df.age_years.between(5, 10))] = params[
#             "prog_5_10yr"
#         ]
#         prob_prog_child.loc[df.is_alive & (df.age_years.between(10, 15))] = params[
#             "prog_10yr"
#         ]
#         prob_prog_child.loc[df.is_alive & df.tb_bcg & df.age_years <= 10] *= params[
#             "rr_tb_bcg"
#         ]
#         prob_prog_child.loc[(df.hv_on_art == 2)] *= params["rr_tb_art_child"]
#
#         dur_ipt_inf = pd.to_timedelta(params["dur_prot_ipt_infant"], unit="d")
#
#         prob_prog_child.loc[
#             (df.tb_date_ipt < (now - dur_ipt_inf)) & ~df.hv_inf
#         ] *= params["rr_ipt_child"]
#         prob_prog_child.loc[
#             (df.tb_date_ipt < (now - dur_ipt_inf)) & df.hv_inf & (df.hv_on_art != 2)
#         ] *= params["rr_ipt_child_hiv"]
#         prob_prog_child.loc[
#             (df.tb_date_ipt < (now - dur_ipt_inf)) & df.hv_inf & (df.hv_on_art == 2)
#         ] *= params["rr_ipt_art_child"]
#
#         # no direct progression
#         # progression within 1 year
#         new_latent_child = len(
#             df.index[(df.tb_date_latent == now) & (df.age_years < 15) & df.is_alive]
#         )
#         # print(new_latent)
#
#         # some proportion schedule active now, else schedule active at random date
#         if new_latent_child:
#             prog = df[
#                 (df.tb_date_latent == now) & (df.age_years < 15) & df.is_alive
#             ].index
#
#             for person_id in prog:
#                 # decide if person will develop active tb
#                 active = rng.rand() < prob_prog_child[person_id]
#
#                 if active:
#                     # random draw of days 0-365
#                     # random_date = rng.choice(list(range(0, 365)), size=1, p=[(1 / 365)] * 365)
#                     random_date = rng.randint(low=0, high=182)
#                     # convert days into years
#                     random_days = pd.to_timedelta(random_date, unit="d")
#
#                     sch_date = now + random_days
#
#                     logger.debug(
#                         "This is TbEvent, scheduling active disease for child %d on date %s",
#                         person_id,
#                         sch_date,
#                     )
#
#                     # schedule active disease
#                     self.sim.schedule_event(
#                         TbActiveEvent(self.module, person_id), sch_date
#                     )
#
#
# class NonTbSymptomsEvent(RegularEvent, PopulationScopeEventMixin):
#     """ symptom onset and healthcare seeking of people with tb-like symptoms but no active tb infection
#     """
#
#     def __init__(self, module):
#         super().__init__(module, frequency=DateOffset(months=1))  # every 1 month
#
#     def apply(self, population):
#         params = self.module.parameters
#         rng = self.module.rng
#
#         df = population.props
#
#         # select proportion of eligible non-tb patients to test for tb
#         test_non_tb = df.index[
#             (rng.random_sample(size=len(df)) < params["presump_testing"])
#             & df.is_alive
#             & ~(df.tb_inf.str.contains("active"))
#         ]
#
#         df.loc[test_non_tb, "tb_symptoms"] = True
#
#         # schedule tb screening and test
#         if len(test_non_tb):
#
#             # ----------------------------------- SYMPTOMS -----------------------------------
#             # ACTIVE CASES - smear positive and smear negative
#
#             df.loc[test_non_tb, "tb_any_resp_symptoms"] = self.sim.date
#
#             self.sim.modules["SymptomManager"].change_symptom(
#                 person_id=list(test_non_tb),
#                 symptom_string="fever",
#                 add_or_remove="+",
#                 disease_module=self.module,
#                 duration_in_days=None,
#             )
#
#             self.sim.modules["SymptomManager"].change_symptom(
#                 person_id=list(test_non_tb),
#                 symptom_string="respiratory_symptoms",
#                 add_or_remove="+",
#                 disease_module=self.module,
#                 duration_in_days=None,
#             )
#
#             # ----------------------------------- SCHEDULE TB SCREENING -----------------------------------
#
#             for person in test_non_tb:
#                 logger.debug(
#                     "This is NonTbSymptomsEvent, scheduling HSI_Tb_Screening for person %d",
#                     person,
#                 )
#
#                 event = HSI_Tb_Screening(self.module, person_id=person)
#                 self.sim.modules["HealthSystem"].schedule_hsi_event(
#                     event,
#                     priority=2,
#                     topen=self.sim.date + DateOffset(weeks=2),
#                     tclose=None,
#                 )
#
#
# class TbActiveEvent(Event, IndividualScopeEventMixin):
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#
#     def apply(self, person_id):
#         logger.debug(f"Onset of active TB for {person_id} on {self.sim.date}")
#
#         df = self.sim.population.props
#         params = self.sim.modules["Tb"].parameters
#         rng = self.module.rng
#         now = self.sim.date
#
#         # todo add tb_stage == active_extra with prob based on HIV status
#         if (
#             not df.at[person_id, "tb_on_ipt"]
#             and not df.at[person_id, "tb_on_treatment"]
#         ):
#
#             df.at[person_id, "tb_date_active"] = self.sim.date
#             df.at[person_id, "tb_symptoms"] = True
#             df.at[person_id, "tb_ever_tb"] = True
#             df.at[person_id, "tb_stage"] = "active_pulm"
#
#             # check if new infection or re-infection
#             # latent_susc_new or latent_susc_tx
#             if df.at[person_id, "tb_inf"] == "latent_susc_new":
#                 df.at[person_id, "tb_inf"] = "active_susc_new"
#             elif df.at[person_id, "tb_inf"] == "latent_susc_tx":
#                 df.at[person_id, "tb_inf"] = "active_susc_tx"
#
#             # decide smear positive / negative
#             # depends on HIV status
#             if df.at[person_id, "hv_inf"] & (
#                 rng.rand() < params["prop_smear_positive_hiv"]
#             ):
#                 df.at[person_id, "tb_smear"] = True
#
#             # hiv-negative
#             elif rng.rand() < params["prop_smear_positive"]:
#                 df.at[person_id, "tb_smear"] = True
#
#             # ----------------------------------- SYMPTOMS -----------------------------------
#             # ACTIVE CASES - smear positive and smear negative
#
#             df.at[person_id, "tb_any_resp_symptoms"] = now
#
#             self.sim.modules["SymptomManager"].change_symptom(
#                 person_id=person_id,
#                 symptom_string="fever",
#                 add_or_remove="+",
#                 disease_module=self.module,
#                 duration_in_days=None,
#             )
#
#             self.sim.modules["SymptomManager"].change_symptom(
#                 person_id=person_id,
#                 symptom_string="respiratory_symptoms",
#                 add_or_remove="+",
#                 disease_module=self.module,
#                 duration_in_days=None,
#             )
#
#             self.sim.modules["SymptomManager"].change_symptom(
#                 person_id=person_id,
#                 symptom_string="fatigue",
#                 add_or_remove="+",
#                 disease_module=self.module,
#                 duration_in_days=None,
#             )
#
#             self.sim.modules["SymptomManager"].change_symptom(
#                 person_id=person_id,
#                 symptom_string="night_sweats",
#                 add_or_remove="+",
#                 disease_module=self.module,
#                 duration_in_days=None,
#             )
#
#             # ----------------------------------- ACTIVE CASES SEEKING CARE -----------------------------------
#             # TODO leave this in here for now until diagnostic algorithm updated
#             # TODO: calibrate / input these data from MOH reports
#             prob_care = 0.47
#             if now.year == 2013:
#                 prob_care = 0.44
#             elif now.year == 2014:
#                 prob_care = 0.45
#             elif now.year == 2015:
#                 prob_care = 0.5
#             elif now.year == 2016:
#                 prob_care = 0.57
#             elif now.year > 2016:
#                 prob_care = 0.73
#
#             seeks_care = rng.rand() < prob_care
#
#             if seeks_care:
#                 logger.debug(
#                     "This is TbActiveEvent, scheduling HSI_Tb_Screening for person %d",
#                     person_id,
#                 )
#
#                 event = HSI_Tb_Screening(self.module, person_id=person_id)
#                 self.sim.modules["HealthSystem"].schedule_hsi_event(
#                     event,
#                     priority=2,
#                     topen=self.sim.date + DateOffset(weeks=2),
#                     tclose=None,
#                 )
#             else:
#                 logger.debug(
#                     "This is TbActiveEvent, person %d is not seeking care", person_id
#                 )
#
#             # ----------------------------------- HIV+ SCHEDULE AIDS ONSET -----------------------------------
#
#             if df.at[person_id, "hv_inf"]:
#                 logger.debug(
#                     "This is TbActiveEvent scheduling aids onset for person %d",
#                     person_id,
#                 )
#
#                 aids = hiv.HivAidsEvent(self.module, person_id)
#                 self.sim.schedule_event(aids, self.sim.date)
#
#
# class TbRelapseEvent(RegularEvent, PopulationScopeEventMixin):
#     """ relapse from latent to active
#     """
#
#     def __init__(self, module):
#         super().__init__(module, frequency=DateOffset(months=1))  # every 1 month
#
#     def apply(self, population):
#
#         df = self.sim.population.props
#         params = self.module.parameters
#         now = self.sim.date
#         rng = self.module.rng
#
#         # ----------------------------------- RELAPSE -----------------------------------
#         random_draw = rng.random_sample(size=len(df))
#
#         # HIV-NEGATIVE
#
#         # relapse after treatment completion, tb_date_treated + six months
#         relapse_tx_complete = df[
#             (df.tb_inf == "latent_susc_tx")
#             & ~df.tb_on_ipt
#             & df.is_alive
#             & ~df.hv_inf
#             & (self.sim.date - df.tb_date_treated > pd.to_timedelta(182.625, unit="d"))
#             & (self.sim.date - df.tb_date_treated < pd.to_timedelta(732.5, unit="d"))
#             & ~df.tb_treatment_failure
#             & (random_draw < params["monthly_prob_relapse_tx_complete"])
#         ].index
#
#         # relapse after treatment default, tb_treatment_failure=True, but make sure not tb-mdr
#         relapse_tx_incomplete = df[
#             (df.tb_inf == "latent_susc_tx")
#             & ~df.tb_on_ipt
#             & df.is_alive
#             & ~df.hv_inf
#             & df.tb_treatment_failure
#             & (self.sim.date - df.tb_date_treated > pd.to_timedelta(182.625, unit="d"))
#             & (self.sim.date - df.tb_date_treated < pd.to_timedelta(732.5, unit="d"))
#             & (random_draw < params["monthly_prob_relapse_tx_incomplete"])
#         ].index
#
#         # relapse after >2 years following completion of treatment (or default)
#         # use tb_date_treated + 2 years + 6 months of treatment
#         relapse_tx_2yrs = df[
#             (df.tb_inf == "latent_susc_tx")
#             & ~df.tb_on_ipt
#             & df.is_alive
#             & ~df.hv_inf
#             & (self.sim.date - df.tb_date_treated >= pd.to_timedelta(913.125, unit="d"))
#             & (random_draw < params["monthly_prob_relapse_2yrs"])
#         ].index
#
#         df.loc[
#             relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, "tb_inf"
#         ] = "active_susc_tx"
#         df.loc[
#             relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs,
#             "tb_date_active",
#         ] = now
#         df.loc[
#             relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, "tb_ever_tb"
#         ] = True
#         df.loc[
#             relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, "tb_stage"
#         ] = "active_pulm"
#
#         # HIV-POSITIVE
#
#         # relapse after treatment completion, tb_date_treated + six months
#         relapse_tx_complete = df[
#             (df.tb_inf == "latent_susc_tx")
#             & ~df.tb_on_ipt
#             & df.is_alive
#             & df.hv_inf
#             & (self.sim.date - df.tb_date_treated > pd.to_timedelta(182.625, unit="d"))
#             & (self.sim.date - df.tb_date_treated < pd.to_timedelta(732.5, unit="d"))
#             & ~df.tb_treatment_failure
#             & (
#                 random_draw
#                 < (
#                     params["monthly_prob_relapse_tx_complete"]
#                     * params["rr_relapse_hiv"]
#                 )
#             )
#         ].index
#
#         # relapse after treatment default, tb_treatment_failure=True, but make sure not tb-mdr
#         relapse_tx_incomplete = df[
#             (df.tb_inf == "latent_susc_tx")
#             & ~df.tb_on_ipt
#             & df.is_alive
#             & df.hv_inf
#             & df.tb_treatment_failure
#             & (self.sim.date - df.tb_date_treated > pd.to_timedelta(182.625, unit="d"))
#             & (self.sim.date - df.tb_date_treated < pd.to_timedelta(732.5, unit="d"))
#             & (
#                 random_draw
#                 < (
#                     params["monthly_prob_relapse_tx_complete"]
#                     * params["rr_relapse_hiv"]
#                 )
#             )
#         ].index
#
#         # relapse after >2 years following completion of treatment (or default)
#         # use tb_date_treated + 2 years + 6 months of treatment
#         relapse_tx_2yrs = df[
#             (df.tb_inf == "latent_susc_tx")
#             & ~df.tb_on_ipt
#             & df.is_alive
#             & df.hv_inf
#             & (self.sim.date - df.tb_date_treated >= pd.to_timedelta(913.125, unit="d"))
#             & (
#                 random_draw
#                 < (
#                     params["monthly_prob_relapse_tx_complete"]
#                     * params["rr_relapse_hiv"]
#                 )
#             )
#         ].index
#
#         df.loc[
#             relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, "tb_inf"
#         ] = "active_susc_tx"
#         df.loc[
#             relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs,
#             "tb_date_active",
#         ] = now
#         df.loc[
#             relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, "tb_ever_tb"
#         ] = True
#         df.loc[
#             relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, "tb_stage"
#         ] = "active_pulm"
#         df.loc[
#             relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, "tb_symptoms"
#         ] = True
#
#         all_relapse_complete = relapse_tx_complete.append(relapse_tx_2yrs)
#         all_relapse = all_relapse_complete.append(relapse_tx_incomplete)
#
#         # ----------------------------------- SYMPTOMS -----------------------------------
#         df.loc[all_relapse, "tb_any_resp_symptoms"] = now
#
#         self.sim.modules["SymptomManager"].change_symptom(
#             person_id=list(all_relapse),
#             symptom_string="fever",
#             add_or_remove="+",
#             disease_module=self.module,
#             duration_in_days=None,
#         )
#
#         self.sim.modules["SymptomManager"].change_symptom(
#             person_id=list(all_relapse),
#             symptom_string="respiratory_symptoms",
#             add_or_remove="+",
#             disease_module=self.module,
#             duration_in_days=None,
#         )
#
#         self.sim.modules["SymptomManager"].change_symptom(
#             person_id=list(all_relapse),
#             symptom_string="fatigue",
#             add_or_remove="+",
#             disease_module=self.module,
#             duration_in_days=None,
#         )
#
#         self.sim.modules["SymptomManager"].change_symptom(
#             person_id=list(all_relapse),
#             symptom_string="night_sweats",
#             add_or_remove="+",
#             disease_module=self.module,
#             duration_in_days=None,
#         )
#
#         # ----------------------------------- RELAPSE CASES SEEKING CARE -----------------------------------
#         # relapse after complete treatment course - refer for xpert testing
#         # TODO leave this in here for now until diagnostic algorithm updated
#         # TODO: calibrate / input these data from MOH reports
#         prob_care = 0.47
#         if now.year == 2013:
#             prob_care = 0.44
#         elif now.year == 2014:
#             prob_care = 0.45
#         elif now.year == 2015:
#             prob_care = 0.5
#         elif now.year == 2016:
#             prob_care = 0.57
#         elif now.year > 2016:
#             prob_care = 0.73
#
#         # if relapse after complete treatment course
#         seeks_care = pd.Series(data=False, index=df.loc[relapse_tx_2yrs].index)
#         for i in df.loc[all_relapse_complete].index:
#             prob = rng.rand() < prob_care
#             seeks_care[i] = rng.rand() < prob
#
#         if seeks_care.sum() > 0:
#             for person_index in seeks_care.index[seeks_care]:
#                 logger.debug(
#                     f"This is TbRelapseEvent, scheduling HSI_Tb_XpertTest for {person_index}",
#                     person_index,
#                 )
#                 event = HSI_Tb_XpertTest(self.module, person_id=person_index)
#                 self.sim.modules["HealthSystem"].schedule_hsi_event(
#                     event,
#                     priority=2,
#                     topen=self.sim.date,
#                     tclose=self.sim.date + DateOffset(weeks=2),
#                 )
#
#                 # add back-up check if xpert is not available, then schedule sputum smear
#                 self.sim.schedule_event(
#                     TbCheckXpert(self.module, person_index),
#                     self.sim.date + DateOffset(weeks=2),
#                 )
#
#         # relapse after incomplete treatment course - repeat treatment course
#         seeks_care = pd.Series(data=False, index=df.loc[relapse_tx_incomplete].index)
#         for i in df.loc[relapse_tx_incomplete].index:
#             prob = rng.rand() < prob_care
#             seeks_care[i] = rng.rand() < prob
#
#         if seeks_care.sum() > 0:
#             for person_index in seeks_care.index[seeks_care]:
#                 if df.at[person_index, "age_years"] < 15:
#
#                     logger.debug(
#                         f"TbRelapseEvent, scheduling HSI_Tb_StartTreatmentChild for relapsed child {person_index}"
#                     )
#
#                     event = HSI_Tb_StartTreatmentChild(
#                         self.module, person_id=person_index
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         event,
#                         priority=2,
#                         topen=self.sim.date,
#                         tclose=self.sim.date + DateOffset(weeks=2),
#                     )
#                 else:
#                     logger.debug(
#                         f"TbRelapseEvent, scheduling HSI_Tb_StartTreatmentAdult for relapsed adult {person_index}"
#                     )
#                     event = HSI_Tb_StartTreatmentAdult(
#                         self.module, person_id=person_index
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         event,
#                         priority=2,
#                         topen=self.sim.date,
#                         tclose=self.sim.date + DateOffset(weeks=2),
#                     )
#
#
# class TbScheduleTesting(RegularEvent, PopulationScopeEventMixin):
#     """ additional TB testing happening outside the symptom-driven generic HSI event
#     to increase tx coverage up to reported levels
#     """
#
#     def __init__(self, module):
#         super().__init__(module, frequency=DateOffset(months=1))
#
#     def apply(self, population):
#         df = population.props
#         now = self.sim.date
#         p = self.module.parameters
#
#         # select symptomatic people to go for testing (and subsequent tx)
#         # random sample to match clinical case tx coverage
#         test = df.index[
#             (self.module.rng.random_sample(size=len(df)) < p["rate_testing_tb"])
#             & df.is_alive
#             & (
#                 (df.tb_inf == "active_susc_new")
#                 | (df.tb_inf == "active_susc_tx")
#                 | (df.tb_inf == "active_mdr_new")
#                 | (df.tb_inf == "active_mdr_tx")
#             )
#             & ~(df.tb_diagnosed | df.tb_mdr_diagnosed)
#         ]
#
#         for person_index in test:
#             logger.debug(
#                 f"TbScheduleTesting: scheduling HSI_Tb_Screening for person {person_index}"
#             )
#
#             event = HSI_Tb_Screening(self.module, person_id=person_index)
#             self.sim.modules["HealthSystem"].schedule_hsi_event(
#                 event, priority=1, topen=now, tclose=None
#             )
#
#
# class TbCheckXpert(Event, IndividualScopeEventMixin):
#     """ if person has not received xpert test as prescribed, schedule sputum smear
#     """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#
#     def apply(self, person_id):
#         logger.debug(
#             "This is TbCheckXpert checking if person %d received xpert test", person_id
#         )
#
#         df = self.sim.population.props
#         now = self.sim.date
#
#         # if no xpert test
#         if not df.at[person_id, "tb_xpert_test"]:
#             logger.debug(
#                 "This is TbCheckXpert, scheduling HSI_Tb_SputumTest for person %d",
#                 person_id,
#             )
#
#             event = HSI_Tb_SputumTest(self.module, person_id=person_id)
#             self.sim.modules["HealthSystem"].schedule_hsi_event(
#                 event,
#                 priority=2,
#                 topen=self.sim.date,
#                 tclose=self.sim.date + DateOffset(weeks=2),
#             )
#
#         # if no xpert in last 14 days
#         if (now - df.at[person_id, "tb_date_xpert_test"]) > pd.to_timedelta(
#             14, unit="d"
#         ):
#             logger.debug(
#                 "This is TbCheckXpert, scheduling HSI_Tb_SputumTest for person %d",
#                 person_id,
#             )
#
#             event = HSI_Tb_SputumTest(self.module, person_id=person_id)
#             self.sim.modules["HealthSystem"].schedule_hsi_event(
#                 event,
#                 priority=2,
#                 topen=self.sim.date,
#                 tclose=self.sim.date + DateOffset(weeks=2),
#             )
#
#
# class TbCheckXray(Event, IndividualScopeEventMixin):
#     """ if person has not received chest x-ray as prescribed, treat if still presumptive case
#     """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#
#     def apply(self, person_id):
#         logger.debug(
#             "This is TbCheckXray checking if person %d received xpert test", person_id
#         )
#
#         df = self.sim.population.props
#         now = self.sim.date
#
#         # if not started treatment in last 14 days and still have symptoms
#         if not df.at[person_id, "tb_date_treated"] or (
#             now - df.at[person_id, "tb_date_treated"]
#         ) < pd.to_timedelta(14, unit="d"):
#             logger.debug(
#                 "This is TbCheckXray, scheduling HSI_Tb_StartTreatmentChild for person %d",
#                 person_id,
#             )
#
#             event = HSI_Tb_StartTreatmentChild(self.module, person_id=person_id)
#             self.sim.modules["HealthSystem"].schedule_hsi_event(
#                 event,
#                 priority=2,
#                 topen=self.sim.date,
#                 tclose=self.sim.date + DateOffset(weeks=2),
#             )
#
#
# class TbSelfCureEvent(RegularEvent, PopulationScopeEventMixin):
#     """ tb self-cure events
#     """
#
#     def __init__(self, module):
#         super().__init__(module, frequency=DateOffset(months=1))  # every 1 month
#
#     def apply(self, population):
#         params = self.module.parameters
#         now = self.sim.date
#         rng = self.module.rng
#
#         df = population.props
#
#         # self-cure - move from active to latent, excludes cases that just became active
#         random_draw = rng.random_sample(size=len(df))
#
#         # hiv-negative
#         self_cure = df[
#             df["tb_inf"].str.contains("active_susc")
#             & df.is_alive
#             & ~df.hv_inf
#             & (df.tb_date_active < now)
#             & (random_draw < params["monthly_prob_self_cure"])
#         ].index
#         df.loc[self_cure, "tb_inf"] = "latent_susc_new"
#         df.loc[self_cure, "tb_stage"] = "latent"
#         df.loc[self_cure, "tb_symptoms"] = False
#         df.loc[self_cure, "tb_diagnosed"] = False
#
#         # hiv-positive, not on art
#         self_cure_hiv = df[
#             df["tb_inf"].str.contains("active_susc")
#             & df.is_alive
#             & df.hv_inf
#             & (df.hv_on_art != 2)
#             & (df.tb_date_active < now)
#             & (random_draw < params["monthly_prob_self_cure_hiv"])
#         ].index
#         df.loc[self_cure_hiv, "tb_inf"] = "latent_susc_new"
#         df.loc[self_cure_hiv, "tb_stage"] = "latent"
#         df.loc[self_cure_hiv, "tb_symptoms"] = False
#         df.loc[self_cure_hiv, "tb_diagnosed"] = False
#
#         # hiv-positive, on art
#         self_cure_art = df[
#             df["tb_inf"].str.contains("active_susc")
#             & df.is_alive
#             & df.hv_inf
#             & (df.hv_on_art == 2)
#             & (df.tb_date_active < now)
#             & (random_draw < params["monthly_prob_self_cure"])
#         ].index
#         df.loc[self_cure_art, "tb_inf"] = "latent_susc_new"
#         df.loc[self_cure_art, "tb_stage"] = "latent"
#         df.loc[self_cure_art, "tb_symptoms"] = False
#         df.loc[self_cure_art, "tb_diagnosed"] = False
#
#         # check that tb symptoms are present and caused by tb before resolving
#         # all_self_cure = pd.concat(self_cure, self_cure_hiv, self_cure_art)
#         # all_self_cure = self_cure + self_cure_hiv + self_cure_art
#         all_self_cure = [*self_cure, *self_cure_hiv, *self_cure_art]
#
#         for person_id in all_self_cure:
#             if (
#                 "respiratory_symptoms"
#                 in self.sim.modules["SymptomManager"].has_what(person_id)
#             ) & (
#                 "Tb"
#                 in self.sim.modules["SymptomManager"].causes_of(
#                     person_id, "respiratory_symptoms"
#                 )
#             ):
#                 # this will clear all tb symptoms
#                 self.sim.modules["SymptomManager"].clear_symptoms(
#                     person_id=person_id, disease_module=self.module
#                 )
#
#
# # ---------------------------------------------------------------------------
# #   TB MDR infection event
# # ---------------------------------------------------------------------------
# # TODO should transmission be limited within each district? background foi for transmission risk between districts
# # TODO add relative infectiousness for those on poor tx [prop tx failure]
# # TODO rel inf for smear negative - weight inf by prop smear negative
# # TODO age/sex distribution for new active cases? may emerge through rr active
# class TbMdrEvent(RegularEvent, PopulationScopeEventMixin):
#     """ tb-mdr infection events
#     """
#
#     def __init__(self, module):
#         super().__init__(module, frequency=DateOffset(months=1))  # every 1 month
#
#     def apply(self, population):
#         params = self.module.parameters
#         now = self.sim.date
#         rng = self.module.rng
#
#         df = population.props
#
#         # ----------------------------------- FORCE OF INFECTION -----------------------------------
#
#         # apply a force of infection to produce new latent cases
#         # no age distribution for FOI but the relative risks would affect distribution of active infections
#
#         # infectious people are active_pulm
#         # hiv-positive and hiv-negative
#         active_sm_pos = int(
#             len(
#                 df[
#                     df["tb_inf"].str.contains("active_mdr")
#                     & (df.tb_stage == "active_pulm")
#                     & df.is_alive
#                 ]
#             )
#             * params["prop_smear_positive"]
#         )
#
#         active_sm_neg = int(
#             len(
#                 df[
#                     df["tb_inf"].str.contains("active_mdr")
#                     & (df.tb_stage == "active_pulm")
#                     & df.is_alive
#                 ]
#             )
#             * (1 - params["prop_smear_positive"])
#         )
#
#         # population at-risk of new infection = uninfected
#         uninfected_total = len(df[(df.tb_inf == "uninfected") & df.is_alive])
#         total_population = len(df[df.is_alive])
#
#         # in small pops, may get zero values which results in FOI = 0
#         if active_sm_pos == 0:
#             active_sm_pos = 1
#
#         if active_sm_neg == 0:
#             active_sm_neg = 1
#
#         force_of_infection = (
#             params["transmission_rate"]
#             * active_sm_pos
#             * (active_sm_neg * params["rel_inf_smear_ng"])
#             * uninfected_total
#         ) / total_population
#         # print('force_of_infection: ', force_of_infection)
#
#         # ----------------------------------- NEW INFECTIONS -----------------------------------
#
#         # pop at risk = susceptible and latent_susc, latent_mdr_primary only
#         #  no age/sex effect on risk of latent infection
#         prob_tb_new = pd.Series(0, index=df.index)
#         prob_tb_new.loc[
#             df.is_alive & (df.tb_inf == "uninfected")
#         ] = force_of_infection  # print('prob_tb_new: ', prob_tb_new)
#
#         # assign risk of latent tb
#         risk_tb = pd.Series(1, index=df.index)
#         risk_tb.loc[df.is_alive & df.tb_bcg & df.age_years < 10] *= params["rr_bcg_inf"]
#
#         # weight the likelihood of being sampled by the relative risk
#         norm_p = pd.Series(risk_tb)
#         norm_p /= norm_p.sum()  # normalise
#
#         # get a list of random numbers between 0 and 1 for each infected individual
#         random_draw = rng.random_sample(size=len(df))
#
#         tb_idx = df.index[df.is_alive & (random_draw < (prob_tb_new * norm_p))]
#
#         df.loc[tb_idx, "tb_inf"] = "latent_mdr_new"
#         df.loc[tb_idx, "tb_date_latent"] = now
#         df.loc[tb_idx, "tb_stage"] = "latent"
#
#         # ----------------------------------- RE-INFECTIONS -----------------------------------
#
#         # pop at risk = latent_mdr_secondary, latent_susc (primary & secondary)
#         prob_tb_reinf = pd.Series(0, index=df.index)
#         prob_tb_reinf.loc[
#             (df.tb_inf == "latent_mdr_tx")
#             | df["tb_inf"].str.contains("latent_susc") & df.is_alive
#         ] = force_of_infection
#
#         repeat_case = df.index[df.is_alive & (random_draw < (prob_tb_reinf * norm_p))]
#
#         # unchanged status, high risk of relapse as if just recovered
#         df.loc[repeat_case, "tb_inf"] = "latent_mdr_tx"
#
#         df.loc[repeat_case, "tb_date_latent"] = now
#         df.loc[repeat_case, "tb_stage"] = "latent"
#
#         # -----------------------------------------------------------------------------------------------------
#         # PROGRESSION TO ACTIVE MDR-TB DISEASE
#
#         # ----------------------------------- ADULT PROGRESSORS TO ACTIVE DISEASE -----------------------------------
#
#         # probability of active disease
#         prob_prog = pd.Series(0, index=df.index)
#         prob_prog.loc[df.is_alive & df.age_years.between(15, 100)] = params[
#             "prog_active"
#         ]
#         prob_prog.loc[df.hv_inf] *= params["rr_tb_hiv"]
#         prob_prog.loc[(df.hv_on_art == 2)] *= (
#             params["rr_tb_hiv"] * params["rr_tb_art_adult"]
#         )
#         # prob_prog.loc[df.xxxx] *= params['rr_tb_overweight']
#         prob_prog.loc[df.li_bmi >= 4] *= params["rr_tb_obese"]
#         # prob_prog.loc[df.xxxx] *= params['rr_tb_diabetes1']
#         prob_prog.loc[df.li_ex_alc] *= params["rr_tb_alcohol"]
#         prob_prog.loc[df.li_tob] *= params["rr_tb_smoking"]
#
#         # ipt - protection against active disease for one yr (6-month regimen)
#         dur_ipt = pd.to_timedelta(params["dur_prot_ipt"], unit="d")
#
#         prob_prog.loc[(df.tb_date_ipt < (now - dur_ipt)) & ~df.hv_inf] *= params[
#             "rr_ipt_adult"
#         ]
#         prob_prog.loc[
#             (df.tb_date_ipt < (now - dur_ipt)) & df.hv_inf & (df.hv_on_art != 2)
#         ] *= params["rr_ipt_adult_hiv"]
#         prob_prog.loc[
#             (df.tb_date_ipt < (now - dur_ipt)) & df.hv_inf & (df.hv_on_art == 2)
#         ] *= params["rr_ipt_art_adult"]
#
#         # scale probability of progression to match overall population values
#         mean = sum(prob_prog[df.is_alive & df.age_years.between(15, 100)]) / len(
#             prob_prog[df.is_alive & df.age_years.between(15, 100)]
#         )
#         scaling_factor = mean / params["prog_active"]
#
#         prob_prog_scaled = prob_prog.divide(other=scaling_factor)
#         assert len(prob_prog) == len(prob_prog_scaled)
#
#         # if any newly infected latent cases, 14% become active directly, only for new infections
#         new_latent = len(
#             df[
#                 (df.tb_inf == "latent_mdr_new")
#                 & (df.tb_date_latent == now)
#                 & (df.age_years >= 15)
#                 & df.is_alive
#             ].index
#         )
#         # print(new_latent)
#
#         # some proportion schedule active now, else schedule active at random date
#         if new_latent:
#
#             # index of all new mdr cases
#             idx = df[
#                 (df.tb_inf == "latent_mdr_new")
#                 & (df.tb_date_latent == now)
#                 & (df.age_years >= 15)
#                 & df.is_alive
#             ].index
#
#             # for each person determine if fast progressor and schedule onset of active disease
#             fast = pd.Series(data=False, index=df.loc[idx].index)
#
#             for i in df.index[idx]:
#                 fast[i] = rng.rand() < params["prop_fast_progressor"]
#
#             # print('fast', fast)
#
#             if fast.sum() > 0:
#                 for person_id in fast.index[fast]:
#                     # print('fast', person_id)
#
#                     logger.debug(
#                         "This is TbMdrEvent, scheduling active disease for fast progressing person %d on date %s",
#                         person_id,
#                         now,
#                     )
#                     # schedule active disease now
#                     self.sim.schedule_event(
#                         TbMdrActiveEvent(self.module, person_id), now
#                     )
#
#             for person_id in idx[~fast]:
#                 # print('slow', person_id)
#                 # decide if person will develop active tb
#                 active = rng.rand() < prob_prog_scaled[person_id]
#
#                 if active:
#                     # randomly select date of onset of active disease
#                     # random draw of days 0-732
#                     random_date = rng.randint(low=0, high=732)
#                     sch_date = now + pd.to_timedelta(random_date, unit="d")
#
#                     logger.debug(
#                         "This is TbMdrEvent, scheduling active disease for slow progressing person %d on date %s",
#                         person_id,
#                         sch_date,
#                     )
#
#                     # schedule active disease
#                     self.sim.schedule_event(
#                         TbMdrActiveEvent(self.module, person_id), sch_date
#                     )
#
#         # ----------------------------------- CHILD PROGRESSORS TO ACTIVE DISEASE -----------------------------------
#         # probability of active disease
#         prob_prog_child = pd.Series(0, index=df.index)
#         prob_prog_child.loc[df.is_alive & (df.age_years < 1)] = params["prog_1yr"]
#         prob_prog_child.loc[df.is_alive & (df.age_years.between(1, 2))] = params[
#             "prog_1_2yr"
#         ]
#         prob_prog_child.loc[df.is_alive & (df.age_years.between(2, 5))] = params[
#             "prog_2_5yr"
#         ]
#         prob_prog_child.loc[df.is_alive & (df.age_years.between(5, 10))] = params[
#             "prog_5_10yr"
#         ]
#         prob_prog_child.loc[df.is_alive & (df.age_years.between(10, 15))] = params[
#             "prog_10yr"
#         ]
#         prob_prog_child.loc[df.tb_bcg & df.age_years < 10] *= params["rr_tb_bcg"]
#         prob_prog_child.loc[(df.hv_on_art == 2)] *= params["rr_tb_art_child"]
#
#         # ipt - protection against active disease for one yr (6-month regimen)
#         dur_ipt_inf = pd.to_timedelta(params["dur_prot_ipt_infant"], unit="d")
#
#         prob_prog_child.loc[
#             (df.tb_date_ipt < (now - dur_ipt_inf)) & ~df.hv_inf
#         ] *= params["rr_ipt_child"]
#         prob_prog_child.loc[
#             (df.tb_date_ipt < (now - dur_ipt_inf)) & df.hv_inf & (df.hv_on_art != 2)
#         ] *= params["rr_ipt_child_hiv"]
#         prob_prog_child.loc[
#             (df.tb_date_ipt < (now - dur_ipt_inf)) & df.hv_inf & (df.hv_on_art == 2)
#         ] *= params["rr_ipt_art_child"]
#
#         # no direct progression
#         # progression within 1 year
#         new_latent_child = df[
#             (df.tb_inf == "latent_mdr_new")
#             & (df.tb_date_latent == now)
#             & (df.age_years < 15)
#             & df.is_alive
#         ].sum()
#         # print(new_latent)
#
#         # some proportion schedule active now, else schedule active at random date
#         if new_latent_child.any():
#             prog = df[
#                 (df.tb_inf == "latent_mdr_new")
#                 & (df.tb_date_latent == now)
#                 & (df.age_years < 15)
#                 & df.is_alive
#             ].index
#
#             for person_id in prog:
#                 # decide if person will develop active tb
#                 active = rng.rand() < prob_prog_child[person_id]
#
#                 if active:
#                     # random draw of days 0-182
#                     random_date = rng.randint(low=0, high=182)
#                     # convert days into years
#                     random_days = pd.to_timedelta(random_date, unit="d")
#
#                     sch_date = now + random_days
#
#                     logger.debug(
#                         "This is TbMdrEvent, scheduling active disease for child %d on date %s",
#                         person_id,
#                         sch_date,
#                     )
#
#                     # schedule active disease
#                     self.sim.schedule_event(
#                         TbMdrActiveEvent(self.module, person_id), sch_date
#                     )
#
#
# class TbMdrActiveEvent(Event, IndividualScopeEventMixin):
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#
#     def apply(self, person_id):
#         logger.debug("Onset of active MDR-TB for person %d", person_id)
#
#         df = self.sim.population.props
#         params = self.sim.modules["Tb"].parameters
#         # prob_pulm = params['pulm_tb']
#         rng = self.module.rng
#         now = self.sim.date
#
#         # todo add in pulm / extrapulm probabilities
#
#         # check not on ipt now or on tb treatment
#         if not df.at[person_id, "tb_on_ipt"] or not df.at[person_id, "tb_on_treatment"]:
#
#             df.at[person_id, "tb_date_active"] = self.sim.date
#             df.at[person_id, "tb_symptoms"] = True
#             df.at[person_id, "tb_ever_tb"] = True
#             df.at[person_id, "tb_ever_tb_mdr"] = True
#             df.at[person_id, "tb_stage"] = "active_pulm"
#
#             # check if new infection or re-infection
#             # latent_susc_new or latent_susc_tx
#             if df.at[person_id, "tb_inf"] == "latent_mdr_new":
#                 df.at[person_id, "tb_inf"] = "active_mdr_new"
#             elif df.at[person_id, "tb_inf"] == "latent_mdr_tx":
#                 df.at[person_id, "tb_inf"] = "active_mdr_tx"
#
#             # decide smear positive / negative
#             # depends on HIV status
#             if df.at[person_id, "hv_inf"] & (
#                 rng.rand() < params["prop_smear_positive_hiv"]
#             ):
#                 df.at[person_id, "tb_smear"] = True
#
#             elif rng.rand() < params["prop_smear_positive"]:
#                 df.at[person_id, "tb_smear"] = True
#
#             # ----------------------------------- ACTIVE CASES SEEKING CARE -----------------------------------
#
#             # determine whether they will seek care on symptom change
#             # prob_care = self.sim.modules['HealthSystem'].get_prob_seek_care(person_id, symptom_code=2)
#
#             # this will be removed once healthcare seeking model is in place
#             prob_care = 0.47
#             if now.year == 2013:
#                 prob_care = 0.44
#             elif now.year == 2014:
#                 prob_care = 0.45
#             elif now.year == 2015:
#                 prob_care = 0.5
#             elif now.year == 2016:
#                 prob_care = 0.57
#             elif now.year > 2016:
#                 prob_care = 0.73
#
#             seeks_care = rng.rand() < prob_care
#
#             if seeks_care:
#                 logger.debug(
#                     "This is TbMdrActiveEvent, scheduling HSI_Tb_Screening for person %d",
#                     person_id,
#                 )
#
#                 event = HSI_Tb_Screening(self.sim.modules["Tb"], person_id=person_id)
#                 self.sim.modules["HealthSystem"].schedule_hsi_event(
#                     event,
#                     priority=2,
#                     topen=self.sim.date,
#                     tclose=self.sim.date + DateOffset(weeks=2),
#                 )
#             else:
#                 logger.debug(
#                     "This is TbMdrActiveEvent, person %d is not seeking care", person_id
#                 )
#
#             if df.at[person_id, "hv_inf"]:
#                 logger.debug(
#                     "This is TbActiveEvent scheduling aids onset for person %d",
#                     person_id,
#                 )
#
#                 aids = hiv.HivAidsEvent(self.module, person_id)
#                 self.sim.schedule_event(aids, self.sim.date)
#
#
# class TbMdrRelapseEvent(RegularEvent, PopulationScopeEventMixin):
#     """ relapse from latent to active
#     """
#
#     def __init__(self, module):
#         super().__init__(module, frequency=DateOffset(months=1))  # every 1 month
#
#     def apply(self, population):
#
#         df = self.sim.population.props
#         params = self.module.parameters
#         now = self.sim.date
#         rng = self.module.rng
#
#         # ----------------------------------- RELAPSE -----------------------------------
#         random_draw = rng.random_sample(size=len(df))
#
#         # relapse after treatment completion, tb_date_treated + six months
#         relapse_tx_complete = df[
#             (df.tb_inf == "latent_mdr_tx")
#             & ~df.tb_on_ipt
#             & df.is_alive
#             & (self.sim.date - df.tb_date_treated > pd.to_timedelta(182.625, unit="d"))
#             & (self.sim.date - df.tb_date_treated < pd.to_timedelta(732.5, unit="d"))
#             & ~df.tb_treatment_failure
#             & (random_draw < params["monthly_prob_relapse_tx_complete"])
#         ].index
#
#         # relapse after treatment default, tb_treatment_failure=True, but make sure not tb-mdr
#         relapse_tx_incomplete = df[
#             (df.tb_inf == "latent_mdr_tx")
#             & ~df.tb_on_ipt
#             & df.is_alive
#             & df.tb_treatment_failure
#             & (self.sim.date - df.tb_date_treated > pd.to_timedelta(182.625, unit="d"))
#             & (self.sim.date - df.tb_date_treated < pd.to_timedelta(732.5, unit="d"))
#             & (random_draw < params["monthly_prob_relapse_tx_incomplete"])
#         ].index
#
#         # relapse after >2 years following completion of treatment (or default)
#         # use tb_date_treated + 2 years + 6 months of treatment
#         relapse_tx_2yrs = df[
#             (df.tb_inf == "latent_mdr_secondary")
#             & ~df.tb_on_ipt
#             & df.is_alive
#             & (self.sim.date - df.tb_date_treated >= pd.to_timedelta(732.5, unit="d"))
#             & (random_draw < params["monthly_prob_relapse_2yrs"])
#         ].index
#
#         df.loc[
#             relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, "tb_inf"
#         ] = "active_mdr_tx"
#         df.loc[
#             relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs,
#             "tb_date_active",
#         ] = now
#         df.loc[
#             relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, "tb_ever_tb"
#         ] = True
#         df.loc[
#             relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, "tb_stage"
#         ] = "active_pulm"
#         df.loc[
#             relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, "tb_symptoms"
#         ] = True
#
#         all_relapse_complete = relapse_tx_complete.append(relapse_tx_2yrs)
#         all_relapse = all_relapse_complete.append(relapse_tx_incomplete)
#
#         # ----------------------------------- SYMPTOMS -----------------------------------
#         df.loc[all_relapse, "tb_any_resp_symptoms"] = now
#
#         self.sim.modules["SymptomManager"].change_symptom(
#             person_id=list(all_relapse),
#             symptom_string="fever",
#             add_or_remove="+",
#             disease_module=self.module,
#             duration_in_days=None,
#         )
#
#         self.sim.modules["SymptomManager"].change_symptom(
#             person_id=list(all_relapse),
#             symptom_string="respiratory_symptoms",
#             add_or_remove="+",
#             disease_module=self.module,
#             duration_in_days=None,
#         )
#
#         self.sim.modules["SymptomManager"].change_symptom(
#             person_id=list(all_relapse),
#             symptom_string="fatigue",
#             add_or_remove="+",
#             disease_module=self.module,
#             duration_in_days=None,
#         )
#
#         self.sim.modules["SymptomManager"].change_symptom(
#             person_id=list(all_relapse),
#             symptom_string="night_sweats",
#             add_or_remove="+",
#             disease_module=self.module,
#             duration_in_days=None,
#         )
#
#         # ----------------------------------- RELAPSE CASES SEEKING CARE -----------------------------------
#         # relapse after complete treatment course - refer for xpert testing
#         # TODO leave this in here for now until diagnostic algorithm updated
#         # TODO: calibrate / input these data from MOH reports
#         prob_care = 0.47
#         if now.year == 2013:
#             prob_care = 0.44
#         elif now.year == 2014:
#             prob_care = 0.45
#         elif now.year == 2015:
#             prob_care = 0.5
#         elif now.year == 2016:
#             prob_care = 0.57
#         elif now.year > 2016:
#             prob_care = 0.73
#
#         # if relapse after complete treatment course
#         seeks_care = pd.Series(data=False, index=df.loc[relapse_tx_2yrs].index)
#         for i in df.loc[all_relapse_complete].index:
#             prob = rng.rand() < prob_care
#             seeks_care[i] = rng.rand() < prob
#
#         if seeks_care.sum() > 0:
#             for person_index in seeks_care.index[seeks_care]:
#                 logger.debug(
#                     "This is TbMdrRelapseEvent, scheduling HSI_Tb_XpertTest for person %d",
#                     person_index,
#                 )
#                 event = HSI_Tb_XpertTest(self.module, person_id=person_index)
#                 self.sim.modules["HealthSystem"].schedule_hsi_event(
#                     event,
#                     priority=2,
#                     topen=self.sim.date,
#                     tclose=self.sim.date + DateOffset(weeks=2),
#                 )
#
#                 # add back-up check if xpert is not available, then schedule sputum smear
#                 self.sim.schedule_event(
#                     TbCheckXpert(self.module, person_index),
#                     self.sim.date + DateOffset(weeks=2),
#                 )
#
#         # relapse after incomplete treatment course - repeat treatment course
#         seeks_care = pd.Series(data=False, index=df.loc[relapse_tx_incomplete].index)
#         for i in df.loc[relapse_tx_incomplete].index:
#             prob = rng.rand() < prob_care
#             seeks_care[i] = rng.rand() < prob
#
#         if seeks_care.sum() > 0:
#             for person_index in seeks_care.index[seeks_care]:
#                 if df.at[person_index, "age_years"] < 15:
#
#                     logger.debug(
#                         "This is TbMdrActiveEvent, scheduling HSI_Tb_StartTreatmentChild for relapsed child %d",
#                         person_index,
#                     )
#                     event = HSI_Tb_StartTreatmentChild(
#                         self.module, person_id=person_index
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         event,
#                         priority=2,
#                         topen=self.sim.date,
#                         tclose=self.sim.date + DateOffset(weeks=2),
#                     )
#                 else:
#                     logger.debug(
#                         "This is TbMdrActiveEvent, scheduling HSI_Tb_StartTreatmentChild for relapsed adult %d",
#                         person_index,
#                     )
#                     event = HSI_Tb_StartTreatmentAdult(
#                         self.module, person_id=person_index
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         event,
#                         priority=2,
#                         topen=self.sim.date,
#                         tclose=self.sim.date + DateOffset(weeks=2),
#                     )
#
#
# class TbMdrSelfCureEvent(RegularEvent, PopulationScopeEventMixin):
#     """ tb-mdr self-cure events
#     """
#
#     def __init__(self, module):
#         super().__init__(module, frequency=DateOffset(months=1))  # every 1 month
#
#     def apply(self, population):
#         params = self.module.parameters
#         now = self.sim.date
#
#         df = population.props
#
#         # self-cure - move from active to latent_secondary, make sure it's not the ones that just became active
#         random_draw = self.module.rng.random_sample(size=len(df))
#
#         # hiv-negative
#         self_cure = df[
#             df["tb_inf"].str.contains("active_mdr")
#             & df.is_alive
#             & ~df.hv_inf
#             & (df.tb_date_active < now)
#             & (random_draw < params["monthly_prob_self_cure"])
#         ].index
#         df.loc[self_cure, "tb_inf"] = "latent_susc_new"
#         df.loc[self_cure, "tb_stage"] = "latent"
#         df.loc[self_cure, "tb_symptoms"] = False
#         df.loc[self_cure, "tb_diagnosed"] = False
#
#         # hiv-positive, not on art
#         self_cure_hiv = df[
#             df["tb_inf"].str.contains("active_mdr")
#             & df.is_alive
#             & df.hv_inf
#             & (df.hv_on_art != 2)
#             & (df.tb_date_active < now)
#             & (random_draw < params["monthly_prob_self_cure_hiv"])
#         ].index
#         df.loc[self_cure_hiv, "tb_inf"] = "latent_susc_new"
#         df.loc[self_cure_hiv, "tb_stage"] = "latent"
#         df.loc[self_cure_hiv, "tb_symptoms"] = False
#         df.loc[self_cure_hiv, "tb_diagnosed"] = False
#
#         # hiv-positive, on art
#         self_cure_art = df[
#             df["tb_inf"].str.contains("active_mdr")
#             & df.is_alive
#             & df.hv_inf
#             & (df.hv_on_art == 2)
#             & (df.tb_date_active < now)
#             & (random_draw < params["monthly_prob_self_cure"])
#         ].index
#         df.loc[self_cure_art, "tb_inf"] = "latent_susc_new"
#         df.loc[self_cure_art, "tb_stage"] = "latent"
#         df.loc[self_cure_art, "tb_symptoms"] = False
#         df.loc[self_cure_art, "tb_diagnosed"] = False
#
#         # check that tb symptoms are present and caused by tb before resolving
#         # all_self_cure = pd.concat(self_cure, self_cure_hiv, self_cure_art)
#         # all_self_cure = self_cure + self_cure_hiv + self_cure_art
#         all_self_cure = [*self_cure, *self_cure_hiv, *self_cure_art]
#
#         for person_id in all_self_cure:
#             if (
#                 "respiratory_symptoms"
#                 in self.sim.modules["SymptomManager"].has_what(person_id)
#             ) & (
#                 "Tb"
#                 in self.sim.modules["SymptomManager"].causes_of(
#                     person_id, "respiratory_symptoms"
#                 )
#             ):
#                 # this will clear all tb symptoms
#                 self.sim.modules["SymptomManager"].clear_symptoms(
#                     person_id=person_id, disease_module=self.module
#                 )
#
#
# # ---------------------------------------------------------------------------
# #   HEALTH SYSTEM INTERACTIONS
# # ---------------------------------------------------------------------------
#
# # ---------------------------------------------------------------------------
# #   Testing
# # ---------------------------------------------------------------------------
#
#
# class HSI_Tb_Screening(HSI_Event, IndividualScopeEventMixin):
#     """
#     This is a Health System Interaction Event.
#     It is the screening event that occurs before a sputum smear test or xpert is offered
#     """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#         assert isinstance(module, Tb)
#
#         # Get a blank footprint and then edit to define call on resources of this event
#         the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
#         the_appt_footprint[
#             "Over5OPD"
#         ] = 0.5  # This requires a few minutes of an outpatient appt
#
#         # Define the necessary information for an HSI
#         self.TREATMENT_ID = "Tb_Screening"
#         self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
#         self.ACCEPTED_FACILITY_LEVEL = '1a'
#         self.ALERT_OTHER_DISEASES = ["Hiv"]
#
#     def apply(self, person_id, squeeze_factor):
#         logger.debug(
#             "HSI_TbScreening: a screening appointment for person %d", person_id
#         )
#
#         df = self.sim.population.props
#
#         # check across all disease modules if patient has: cough, fever, night sweat, weight loss
#         # if any of the above conditions are present, label as presumptive tb case and request appropriate test
#
#         # hiv-negative adults or undiagnosed hiv-positive
#         if (
#             (df.at[person_id, "tb_stage"] == "active_pulm")
#             and (df.at[person_id, "age_exact_years"] >= 5)
#             and not (df.at[person_id, "hv_diagnosed"])
#         ):
#
#             logger.debug(
#                 "HSI_TbScreening: scheduling sputum test for person %d", person_id
#             )
#
#             test = HSI_Tb_SputumTest(self.module, person_id=person_id)
#
#             # Request the health system to give xpert test
#             self.sim.modules["HealthSystem"].schedule_hsi_event(
#                 test, priority=1, topen=self.sim.date, tclose=None
#             )
#
#         # hiv-positive adults, diagnosed only
#         elif (
#             (df.at[person_id, "tb_stage"] == "active_pulm")
#             and (df.at[person_id, "age_exact_years"] >= 5)
#             and (df.at[person_id, "hv_diagnosed"])
#         ):
#
#             logger.debug(
#                 "HSI_TbScreening: scheduling xpert test for person %d", person_id
#             )
#
#             test = HSI_Tb_XpertTest(self.module, person_id=person_id)
#
#             # Request the health system to give xpert test
#             self.sim.modules["HealthSystem"].schedule_hsi_event(
#                 test, priority=1, topen=self.sim.date, tclose=None
#             )
#
#             # add back-up check if xpert is not available, then schedule sputum smear
#             self.sim.schedule_event(
#                 TbCheckXpert(self.module, person_id),
#                 self.sim.date + DateOffset(weeks=2),
#             )
#
#         # if child <5 schedule chest x-ray for diagnosis and add check if x-ray not available
#         elif (df.at[person_id, "tb_stage"] == "active_pulm") and (
#             df.at[person_id, "age_exact_years"] < 5
#         ):
#
#             logger.debug(
#                 "HSI_TbScreening: scheduling chest xray for person %d", person_id
#             )
#
#             test = HSI_Tb_Xray(self.module, person_id=person_id)
#
#             # Request the health system to give xpert test
#             self.sim.modules["HealthSystem"].schedule_hsi_event(
#                 test, priority=1, topen=self.sim.date, tclose=None
#             )
#
#             # add back-up check if chest x-ray is not available, then treat if still symptomatic
#             self.sim.schedule_event(
#                 TbCheckXray(self.module, person_id), self.sim.date + DateOffset(weeks=2)
#             )
#
#     def did_not_run(self):
#         logger.debug("HSI_TbScreening: did not run")
#         pass
#
#
# class HSI_Tb_SputumTest(HSI_Event, IndividualScopeEventMixin):
#     """
#     This is a sputum test for presumptive tb cases
#     """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#         assert isinstance(module, Tb)
#
#         # Get a blank footprint and then edit to define call on resources of this event
#         the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
#         the_appt_footprint[
#             "ConWithDCSA"
#         ] = 1  # This requires one generic outpatient appt
#         the_appt_footprint[
#             "LabTBMicro"
#         ] = 1  # This requires one lab appt for microscopy
#
#         # Define the necessary information for an HSI
#         self.TREATMENT_ID = "Tb_SputumTest"
#         self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
#         self.ACCEPTED_FACILITY_LEVEL = '1a'
#         self.ALERT_OTHER_DISEASES = ["Hiv"]
#
#     def apply(self, person_id, squeeze_factor):
#         logger.debug(
#             "This is HSI_Tb_SputumTest, giving a sputum test to person %d", person_id
#         )
#
#         df = self.sim.population.props
#         params = self.sim.modules["Tb"].parameters
#         now = self.sim.date
#
#         # Get the consumables required
#         consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
#         pkg_code1 = pd.unique(
#             consumables.loc[
#                 consumables["Intervention_Pkg"] == "Microscopy Test",
#                 "Intervention_Pkg_Code",
#             ]
#         )[0]
#
#         the_cons_footprint = {
#             "Intervention_Package_Code": [{pkg_code1: 1}],
#             "Item_Code": [],
#         }
#
#         is_cons_available = self.sim.modules["HealthSystem"].request_consumables(
#             hsi_event=self, cons_req_as_footprint=the_cons_footprint
#         )
#
#         if is_cons_available:
#
#             df.at[person_id, "tb_ever_tested"] = True
#             df.at[person_id, "tb_smear_test"] = True
#             df.at[person_id, "tb_date_smear_test"] = now
#             df.at[person_id, "tb_result_smear_test"] = False
#
#             # ----------------------------------- OUTCOME OF TEST -----------------------------------
#
#             # if smear-positive, sensitivity is 1
#             if (df.at[person_id, "tb_stage"] == "active_pulm") and df.at[
#                 person_id, "tb_smear"
#             ]:
#                 df.at[person_id, "tb_result_smear_test"] = True
#                 df.at[person_id, "tb_diagnosed"] = True
#
#             # ----------------------------------- REFERRALS FOR TREATMENT -----------------------------------
#             if df.at[person_id, "tb_diagnosed"]:
#
#                 # request child treatment
#                 if (
#                     (df.at[person_id, "tb_inf"] == "active_susc_new")
#                     | (df.at[person_id, "tb_inf"] == "active_mdr_new")
#                 ) & (df.at[person_id, "age_years"] < 15):
#                     logger.debug(
#                         "This is HSI_Tb_SputumTest scheduling HSI_Tb_StartTreatmentChild for person %d",
#                         person_id,
#                     )
#
#                     treatment = HSI_Tb_StartTreatmentChild(
#                         self.module, person_id=person_id
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         treatment,
#                         priority=1,
#                         topen=self.sim.date + DateOffset(days=1),
#                         tclose=None,
#                     )
#                 # request adult treatment
#                 if (
#                     (df.at[person_id, "tb_inf"] == "active_susc_new")
#                     | (df.at[person_id, "tb_inf"] == "active_mdr_new")
#                 ) & (df.at[person_id, "age_years"] >= 15):
#                     logger.debug(
#                         "This is HSI_Tb_SputumTest scheduling HSI_Tb_StartTreatmentAdult for person %d",
#                         person_id,
#                     )
#
#                     treatment = HSI_Tb_StartTreatmentAdult(
#                         self.module, person_id=person_id
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         treatment,
#                         priority=1,
#                         topen=self.sim.date + DateOffset(days=1),
#                         tclose=None,
#                     )
#
#                 if (
#                     (df.at[person_id, "tb_inf"] == "active_susc_tx")
#                     | (df.at[person_id, "tb_inf"] == "active_mdr_tx")
#                 ) & (df.at[person_id, "age_years"] < 15):
#                     # request child retreatment
#                     logger.debug(
#                         "This is HSI_Tb_SputumTest scheduling HSI_Tb_RetreatmentChild for person %d",
#                         person_id,
#                     )
#
#                     treatment = HSI_Tb_RetreatmentChild(
#                         self.module, person_id=person_id
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         treatment,
#                         priority=1,
#                         topen=self.sim.date + DateOffset(days=1),
#                         tclose=None,
#                     )
#
#                 if (
#                     (df.at[person_id, "tb_inf"] == "active_susc_tx")
#                     | (df.at[person_id, "tb_inf"] == "active_mdr_tx")
#                 ) & (df.at[person_id, "age_years"] >= 15):
#                     # request adult retreatment
#                     logger.debug(
#                         "This is HSI_Tb_SputumTest scheduling HSI_Tb_RetreatmentAdult for person %d",
#                         person_id,
#                     )
#
#                     treatment = HSI_Tb_RetreatmentAdult(
#                         self.module, person_id=person_id
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         treatment,
#                         priority=1,
#                         topen=self.sim.date + DateOffset(days=1),
#                         tclose=None,
#                     )
#
#                 # ----------------------------------- REFERRALS FOR IPT -----------------------------------
#                 # todo check these coverage levels relate to paed contacts not HIV cases
#                 if self.sim.date.year >= 2014:
#                     # if diagnosed, trigger ipt outreach event for all paediatric contacts of case
#                     district = df.at[person_id, "district_of_residence"]
#                     ipt_cov = params["ipt_contact_cov"]
#                     ipt_cov_year = ipt_cov.loc[
#                         ipt_cov.year == self.sim.date.year
#                     ].coverage.values
#
#                     if (district in params["tb_high_risk_distr"].values) & (
#                         self.module.rng.rand() < ipt_cov_year
#                     ):
#
#                         # check enough contacts available for sample
#                         if (
#                             len(
#                                 df[
#                                     (df.age_years <= 5)
#                                     & ~df.tb_ever_tb
#                                     & ~df.tb_ever_tb_mdr
#                                     & df.is_alive
#                                     & (df.district_of_residence == district)
#                                 ].index
#                             )
#                             > 5
#                         ):
#
#                             # randomly sample from <5 yr olds within district
#                             ipt_sample = (
#                                 df[
#                                     (df.age_years <= 5)
#                                     & ~df.tb_ever_tb
#                                     & ~df.tb_ever_tb_mdr
#                                     & df.is_alive
#                                     & (df.district_of_residence == district)
#                                 ]
#                                 .sample(n=5, replace=False)
#                                 .index
#                             )
#
#                             for person_id in ipt_sample:
#                                 logger.debug(
#                                     "HSI_Tb_Sputum: scheduling IPT for person %d",
#                                     person_id,
#                                 )
#
#                                 ipt_event = HSI_Tb_Ipt(self.module, person_id=person_id)
#                                 self.sim.modules["HealthSystem"].schedule_hsi_event(
#                                     ipt_event,
#                                     priority=1,
#                                     topen=self.sim.date,
#                                     tclose=None,
#                                 )
#
#     def did_not_run(self):
#         logger.debug("HSI_Tb_SputumTest: did not run")
#         pass
#
#
# class HSI_Tb_XpertTest(HSI_Event, IndividualScopeEventMixin):
#     """
#         This is a Health System Interaction Event - tb xpert test
#         """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#         assert isinstance(module, Tb)
#
#         # Get a blank footprint and then edit to define call on resources of this treatment event
#         the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
#         the_appt_footprint["TBFollowUp"] = 1
#         the_appt_footprint["LabTBMicro"] = 1
#
#         # Define the necessary information for an HSI
#         self.TREATMENT_ID = "Tb_XpertTest"
#         self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
#         self.ACCEPTED_FACILITY_LEVEL = '1a'
#         self.ALERT_OTHER_DISEASES = ["Hiv"]
#
#     def apply(self, person_id, squeeze_factor):
#         logger.debug("HSI_Tb_XpertTest: giving xpert test for person %d", person_id)
#
#         df = self.sim.population.props
#         now = self.sim.date
#         params = self.sim.modules["Tb"].parameters
#
#         consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
#         pkg_code1 = pd.unique(
#             consumables.loc[
#                 consumables["Intervention_Pkg"] == "Xpert test", "Intervention_Pkg_Code"
#             ]
#         )[0]
#
#         the_cons_footprint = {
#             "Intervention_Package_Code": [{pkg_code1: 1}],
#             "Item_Code": [],
#         }
#
#         is_cons_available = self.sim.modules["HealthSystem"].request_consumables(
#             hsi_event=self, cons_req_as_footprint=the_cons_footprint
#         )
#
#         if is_cons_available:
#             df.at[person_id, "tb_ever_tested"] = True
#             df.at[person_id, "tb_xpert_test"] = True
#             df.at[person_id, "tb_date_xpert_test"] = now
#             df.at[person_id, "tb_result_xpert_test"] = False
#             df.at[person_id, "tb_diagnosed_mdr"] = False  # default
#
#             # a further 15% of TB cases fail to be diagnosed with Xpert (sensitivity of test)
#             # they will present back to the health system with some delay (2-4 weeks)
#             if df.at[person_id, "tb_inf"].startswith("active"):
#
#                 diagnosed = self.module.rng.choice(
#                     [True, False],
#                     size=1,
#                     p=[params["sens_xpert"], (1 - params["sens_xpert"])],
#                 )
#
#                 if diagnosed:
#                     df.at[person_id, "tb_result_xpert_test"] = True
#                     df.at[person_id, "tb_diagnosed"] = True
#
#                     if df.at[person_id, "tb_inf"].startswith("active_mdr"):
#                         df.at[person_id, "tb_diagnosed_mdr"] = True
#
#                     # ----------------------------------- REFERRALS FOR IPT -----------------------------------
#                     # todo check these coverage levels relate to paed contacts not HIV cases
#                     if self.sim.date.year >= 2014:
#                         # if diagnosed, trigger ipt outreach event for all paediatric contacts of case
#                         district = df.at[person_id, "district_of_residence"]
#                         ipt_cov = params["ipt_contact_cov"]
#                         ipt_cov_year = ipt_cov.loc[
#                             ipt_cov.year == self.sim.date.year
#                         ].coverage.values
#
#                         if (district in params["tb_high_risk_distr"].values) & (
#                             self.module.rng.rand() < ipt_cov_year
#                         ):
#
#                             # check enough contacts available for sample
#                             if (
#                                 len(
#                                     df[
#                                         (df.age_years <= 5)
#                                         & ~df.tb_ever_tb
#                                         & ~df.tb_ever_tb_mdr
#                                         & df.is_alive
#                                         & (df.district_of_residence == district)
#                                     ].index
#                                 )
#                                 > 5
#                             ):
#
#                                 # randomly sample from <5 yr olds within district
#                                 ipt_sample = (
#                                     df[
#                                         (df.age_years <= 5)
#                                         & ~df.tb_ever_tb
#                                         & ~df.tb_ever_tb_mdr
#                                         & df.is_alive
#                                         & (df.district_of_residence == district)
#                                     ]
#                                     .sample(n=5, replace=False)
#                                     .index
#                                 )
#
#                                 for person_id in ipt_sample:
#                                     logger.debug(
#                                         "HSI_Tb_XpertTest: scheduling IPT for person %d",
#                                         person_id,
#                                     )
#
#                                     ipt_event = HSI_Tb_Ipt(
#                                         self.module, person_id=person_id
#                                     )
#                                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                                         ipt_event,
#                                         priority=1,
#                                         topen=self.sim.date,
#                                         tclose=None,
#                                     )
#
#                 else:
#                     # Request the health system for screening again in 2 weeks if still symptomatic
#                     logger.debug(
#                         "HSI_Tb_XpertTest: with negative result for person %d",
#                         person_id,
#                     )
#
#                     followup = HSI_Tb_SputumTest(self.module, person_id=person_id)
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         followup,
#                         priority=1,
#                         topen=self.sim.date + DateOffset(weeks=2),
#                         tclose=None,
#                     )
#
#                 # ----------------------------------- REFERRALS FOR TREATMENT -----------------------------------
#             if df.at[person_id, "tb_diagnosed"]:
#
#                 if (df.at[person_id, "tb_inf"] == "active_susc_new") & (
#                     df.at[person_id, "age_years"] < 15
#                 ):
#                     # request child treatment
#                     logger.debug(
#                         "HSI_Tb_XpertTest: scheduling HSI_Tb_StartTreatmentChild for person %d",
#                         person_id,
#                     )
#
#                     treatment = HSI_Tb_StartTreatmentChild(
#                         self.module, person_id=person_id
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         treatment, priority=1, topen=self.sim.date, tclose=None
#                     )
#
#                 if (df.at[person_id, "tb_inf"] == "active_susc_new") & (
#                     df.at[person_id, "age_years"] >= 15
#                 ):
#                     # request adult treatment
#                     logger.debug(
#                         "HSI_Tb_XpertTest: scheduling HSI_Tb_StartTreatmentAdult for person %d",
#                         person_id,
#                     )
#
#                     treatment = HSI_Tb_StartTreatmentAdult(
#                         self.module, person_id=person_id
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         treatment, priority=1, topen=self.sim.date, tclose=None
#                     )
#
#                 if (df.at[person_id, "tb_inf"] == "active_susc_tx") & (
#                     df.at[person_id, "age_years"] < 15
#                 ):
#                     # request child retreatment
#                     logger.debug(
#                         "HSI_Tb_XpertTest: scheduling HSI_Tb_StartTreatmentChild for person %d",
#                         person_id,
#                     )
#
#                     treatment = HSI_Tb_RetreatmentChild(
#                         self.module, person_id=person_id
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         treatment, priority=1, topen=self.sim.date, tclose=None
#                     )
#
#                 if (df.at[person_id, "tb_inf"] == "active_susc_tx") & (
#                     df.at[person_id, "age_years"] >= 15
#                 ):
#                     # request adult retreatment
#                     logger.debug(
#                         "HSI_Tb_XpertTest: scheduling HSI_Tb_StartTreatmentAdult for person %d",
#                         person_id,
#                     )
#
#                     treatment = HSI_Tb_RetreatmentAdult(
#                         self.module, person_id=person_id
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         treatment, priority=1, topen=self.sim.date, tclose=None
#                     )
#
#                 if df.at[person_id, "tb_diagnosed"] & df.at[
#                     person_id, "tb_inf"
#                 ].startswith("active_mdr"):
#                     # request treatment
#                     logger.debug(
#                         "This is HSI_Tb_XpertTest scheduling HSI_Tb_StartMdrTreatment for person %d",
#                         person_id,
#                     )
#
#                     treatment = HSI_Tb_StartMdrTreatment(
#                         self.module, person_id=person_id
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         treatment, priority=1, topen=self.sim.date, tclose=None
#                     )
#
#                 # ----------------------------------- REFERRALS FOR IPT -----------------------------------
#                 # trigger ipt outreach event for all paediatric contacts of diagnosed case
#                 # randomly sample from <5 yr olds, match by district
#                 if self.sim.date.year >= 2014:
#                     # if diagnosed, trigger ipt outreach event for all paediatric contacts of case
#                     district = df.at[person_id, "district_of_residence"]
#                     ipt_cov = params["ipt_contact_cov"]
#                     ipt_cov_year = ipt_cov.loc[
#                         ipt_cov.year == self.sim.date.year
#                     ].coverage.values
#
#                     if (district in params["tb_high_risk_distr"].values) & (
#                         self.module.rng.rand() < ipt_cov_year
#                     ):
#
#                         if (
#                             len(
#                                 df[
#                                     (df.age_years <= 5)
#                                     & ~df.tb_ever_tb
#                                     & ~df.tb_ever_tb_mdr
#                                     & df.is_alive
#                                     & (df.district_of_residence == district)
#                                 ].index
#                             )
#                             > 5
#                         ):
#
#                             ipt_sample = (
#                                 df[
#                                     (df.age_years <= 5)
#                                     & ~(df.tb_inf.str.contains("active"))
#                                     & df.is_alive
#                                     & (df.district_of_residence == district)
#                                 ]
#                                 .sample(n=5, replace=False)
#                                 .index
#                             )
#
#                             for person_id in ipt_sample:
#                                 logger.debug(
#                                     "HSI_Tb_XpertTest: scheduling HSI_Tb_Ipt for person %d",
#                                     person_id,
#                                 )
#
#                                 ipt_event = HSI_Tb_Ipt(self.module, person_id=person_id)
#                                 self.sim.modules["HealthSystem"].schedule_hsi_event(
#                                     ipt_event,
#                                     priority=1,
#                                     topen=self.sim.date + DateOffset(days=1),
#                                     tclose=None,
#                                 )
#
#     def did_not_run(self):
#         logger.debug("HSI_Tb_XpertTest: did not run")
#         pass
#
#
# class HSI_Tb_Xray(HSI_Event, IndividualScopeEventMixin):
#     """
#     This is a chest x-ray for presumptive tb cases
#     """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#         assert isinstance(module, Tb)
#
#         # Get a blank footprint and then edit to define call on resources of this event
#         the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
#         the_appt_footprint[
#             "Under5OPD"
#         ] = 1  # This requires one paediatric outpatient appt
#         the_appt_footprint["DiagRadio"] = 1  # This requires one x-ray appt
#
#         # Define the necessary information for an HSI
#         self.TREATMENT_ID = "Tb_Xray"
#         self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
#         self.ACCEPTED_FACILITY_LEVEL = '1a'
#         self.ALERT_OTHER_DISEASES = ["Hiv"]
#
#     def apply(self, person_id, squeeze_factor):
#         logger.debug("This is HSI_Tb_Xray, a chest x-ray for person %d", person_id)
#
#         df = self.sim.population.props
#         params = self.module.parameters
#
#         # Get the consumables required
#         consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
#         pkg_code1 = pd.unique(
#             consumables.loc[consumables["Item_Code"] == 175, "Intervention_Pkg_Code"]
#         )[0]
#
#         the_cons_footprint = {
#             "Intervention_Package_Code": [{pkg_code1: 1}],
#             "Item_Code": [],
#         }
#
#         is_cons_available = self.sim.modules["HealthSystem"].request_consumables(
#             hsi_event=self, cons_req_as_footprint=the_cons_footprint
#         )
#
#         if is_cons_available:
#
#             # ----------------------------------- OUTCOME OF TEST -----------------------------------
#
#             # active tb
#             if df.at[person_id, "tb_stage"] == "active_pulm":
#                 df.at[person_id, "tb_diagnosed"] = True
#
#             # ------------------------- REFERRALS FOR TREATMENT -----------------------------------
#
#             if (
#                 df.at[person_id, "tb_diagnosed"]
#                 & (df.at[person_id, "tb_inf"] == "active_susc_new")
#                 & (df.at[person_id, "age_years"] < 15)
#             ):
#                 # request child treatment
#                 logger.debug(
#                     "This is HSI_Tb_Xray scheduling HSI_Tb_StartTreatmentChild for person %d",
#                     person_id,
#                 )
#
#                 treatment = HSI_Tb_StartTreatmentChild(self.module, person_id=person_id)
#                 self.sim.modules["HealthSystem"].schedule_hsi_event(
#                     treatment,
#                     priority=1,
#                     topen=self.sim.date + DateOffset(days=1),
#                     tclose=None,
#                 )
#
#             if (
#                 df.at[person_id, "tb_diagnosed"]
#                 & (df.at[person_id, "tb_inf"] == "active_susc_tx")
#                 & (df.at[person_id, "age_years"] < 15)
#             ):
#
#                 # request child retreatment
#                 logger.debug(
#                     "This is HSI_Tb_Xray scheduling HSI_Tb_RetreatmentChild for person %d",
#                     person_id,
#                 )
#
#                 treatment = HSI_Tb_RetreatmentChild(self.module, person_id=person_id)
#                 self.sim.modules["HealthSystem"].schedule_hsi_event(
#                     treatment,
#                     priority=1,
#                     topen=self.sim.date + DateOffset(days=1),
#                     tclose=None,
#                 )
#
#                 # ----------------------------------- REFERRALS FOR IPT -----------------------------------
#                 # todo check these coverage levels relate to paed contacts not HIV cases
#                 if self.sim.date.year >= 2014:
#                     # if diagnosed, trigger ipt outreach event for all paediatric contacts of case
#                     district = df.at[person_id, "district_of_residence"]
#                     ipt_cov = params["ipt_contact_cov"]
#                     ipt_cov_year = ipt_cov.loc[
#                         ipt_cov.year == self.sim.date.year
#                     ].coverage.values
#
#                     if (district in params["tb_high_risk_distr"].values) & (
#                         self.module.rng.rand() < ipt_cov_year
#                     ):
#
#                         # check enough contacts available for sample
#                         if (
#                             len(
#                                 df[
#                                     (df.age_years <= 5)
#                                     & ~df.tb_ever_tb
#                                     & ~df.tb_ever_tb_mdr
#                                     & df.is_alive
#                                     & (df.district_of_residence == district)
#                                 ].index
#                             )
#                             > 5
#                         ):
#
#                             # randomly sample from <5 yr olds within district
#                             ipt_sample = (
#                                 df[
#                                     (df.age_years <= 5)
#                                     & ~df.tb_ever_tb
#                                     & ~df.tb_ever_tb_mdr
#                                     & df.is_alive
#                                     & (df.district_of_residence == district)
#                                 ]
#                                 .sample(n=5, replace=False)
#                                 .index
#                             )
#
#                             for person_id in ipt_sample:
#                                 logger.debug(
#                                     "HSI_Tb_Xray: scheduling IPT for person %d",
#                                     person_id,
#                                 )
#
#                                 ipt_event = HSI_Tb_Ipt(self.module, person_id=person_id)
#                                 self.sim.modules["HealthSystem"].schedule_hsi_event(
#                                     ipt_event,
#                                     priority=1,
#                                     topen=self.sim.date,
#                                     tclose=None,
#                                 )
#
#     def did_not_run(self):
#         pass
#
#
# # ---------------------------------------------------------------------------
# #   Treatment
# # ---------------------------------------------------------------------------
# # the consumables at treatment initiation include the cost for the full course of treatment
# # so the follow-up appts don't need to account for consumables, just appt time
#
#
# class HSI_Tb_StartTreatmentAdult(HSI_Event, IndividualScopeEventMixin):
#     """
#     This is a Health System Interaction Event - start tb treatment
#     """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#         assert isinstance(module, Tb)
#
#         # Get a blank footprint and then edit to define call on resources of this treatment event
#         the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
#         the_appt_footprint[
#             "TBNew"
#         ] = 1  # New tb treatment initiation appt, this include pharmacist time
#
#         # Define the necessary information for an HSI
#         self.TREATMENT_ID = "Tb_TreatmentInitiationAdult"
#         self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
#         self.ACCEPTED_FACILITY_LEVEL = '1a'
#         self.ALERT_OTHER_DISEASES = []
#
#     def apply(self, person_id, squeeze_factor):
#         logger.debug("We are now ready to treat this tb case %d", person_id)
#
#         now = self.sim.date
#         df = self.sim.population.props
#         params = self.module.parameters
#
#         consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
#         pkg_code1 = pd.unique(
#             consumables.loc[
#                 consumables["Intervention_Pkg"]
#                 == "First line treatment for new TB cases for adults",
#                 "Intervention_Pkg_Code",
#             ]
#         )[0]
#
#         the_cons_footprint = {
#             "Intervention_Package_Code": [{pkg_code1: 1}],
#             "Item_Code": [],
#         }
#
#         is_cons_available = self.sim.modules["HealthSystem"].request_consumables(
#             hsi_event=self, cons_req_as_footprint=the_cons_footprint
#         )
#
#         if is_cons_available:
#
#             # treatment allocated for pulmonary tb
#             if (
#                 df.at[person_id, "is_alive"]
#                 and df.at[person_id, "tb_diagnosed"]
#                 and df.at[person_id, "tb_stage"] == "active_pulm"
#             ):
#                 df.at[person_id, "tb_on_treatment"] = True
#                 df.at[person_id, "tb_date_treated"] = now
#
#                 # schedule a 6-month event where people are cured, symptoms return to latent or not cured
#                 self.sim.schedule_event(
#                     TbCureEvent(self.module, person_id),
#                     self.sim.date + DateOffset(months=6),
#                 )
#
#                 # follow-up appts
#                 logger.debug(
#                     "This is HSI_Tb_StartTreatmentAdult: scheduling follow-up appointments for person %d",
#                     person_id,
#                 )
#
#                 # clinical monitoring
#                 followup_clin = HSI_Tb_FollowUp(self.module, person_id=person_id)
#
#                 # Request the health system to have this follow-up appointment
#                 weeks = params["followup_times"]["treatment_clinical"]
#                 cleaned_list = [x for x in weeks if x == x]  # remove nan
#
#                 for i in range(0, len(cleaned_list)):
#                     followup_appt_date = self.sim.date + DateOffset(
#                         months=cleaned_list[i]
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         followup_clin,
#                         priority=1,
#                         topen=followup_appt_date,
#                         tclose=followup_appt_date + DateOffset(days=3),
#                     )
#
#                 # repeat sputum smear tests
#                 followup_smear = HSI_Tb_FollowUp_SputumTest(
#                     self.module, person_id=person_id
#                 )
#
#                 # Request the health system to have this follow-up appointment
#                 weeks = params["followup_times"]["treatment_sputum"]
#                 cleaned_list = [x for x in weeks if x == x]  # remove nan
#
#                 for i in range(0, len(cleaned_list)):
#                     followup_appt_date = self.sim.date + DateOffset(
#                         months=cleaned_list[i]
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         followup_smear,
#                         priority=1,
#                         topen=followup_appt_date,
#                         tclose=followup_appt_date + DateOffset(days=3),
#                     )
#
#             # treatment allocated for extrapulmonary tb
#             if (
#                 df.at[person_id, "is_alive"]
#                 and df.at[person_id, "tb_diagnosed"]
#                 and df.at[person_id, "tb_stage"] == "active_extra"
#             ):
#
#                 df.at[person_id, "tb_on_treatment"] = True
#                 df.at[person_id, "tb_date_treated"] = now
#
#                 # schedule a 6-month event where people are cured, symptoms return to latent or not cured
#                 self.sim.schedule_event(
#                     TbCureEvent(self.module, person_id),
#                     self.sim.date + DateOffset(months=6),
#                 )
#
#                 # follow-up appts
#                 logger.debug(
#                     "This is HSI_Tb_StartTreatmentAdult: scheduling follow-up appointments for person %d",
#                     person_id,
#                 )
#
#                 # clinical monitoring
#                 followup_clin = HSI_Tb_FollowUp(self.module, person_id=person_id)
#
#                 # Request the health system to have this follow-up appointment
#                 weeks = params["followup_times"]["treatment_extrapulm"]
#                 cleaned_list = [x for x in weeks if x == x]  # remove nan
#
#                 for i in range(0, len(cleaned_list)):
#                     followup_appt_date = self.sim.date + DateOffset(
#                         months=cleaned_list[i]
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         followup_clin,
#                         priority=1,
#                         topen=followup_appt_date,
#                         tclose=followup_appt_date + DateOffset(days=3),
#                     )
#
#                 # repeat sputum smear tests
#                 followup_smear = HSI_Tb_FollowUp_SputumTest(
#                     self.module, person_id=person_id
#                 )
#
#                 # Request the health system to have this follow-up appointment
#                 weeks = params["followup_times"]["treatment_sputum"]
#                 cleaned_list = [x for x in weeks if x == x]  # remove nan
#
#                 for i in range(0, len(cleaned_list)):
#                     followup_appt_date = self.sim.date + DateOffset(
#                         months=cleaned_list[i]
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         followup_smear,
#                         priority=1,
#                         topen=followup_appt_date,
#                         tclose=followup_appt_date + DateOffset(days=3),
#                     )
#
#     def did_not_run(self):
#         pass
#
#
# class HSI_Tb_StartTreatmentChild(HSI_Event, IndividualScopeEventMixin):
#     """
#     This is a Health System Interaction Event - start tb treatment
#     """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#         assert isinstance(module, Tb)
#
#         # Get a blank footprint and then edit to define call on resources of this treatment event
#         the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
#         the_appt_footprint["TBNew"] = 1  # New tb treatment initiation appt
#
#         # Define the necessary information for an HSI
#         self.TREATMENT_ID = "Tb_TreatmentInitiationChild"
#         self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
#         self.ACCEPTED_FACILITY_LEVEL = '1a'
#         self.ALERT_OTHER_DISEASES = []
#
#     def apply(self, person_id, squeeze_factor):
#         logger.debug("We are now ready to treat this tb case %d", person_id)
#
#         now = self.sim.date
#         df = self.sim.population.props
#         params = self.module.parameters
#
#         consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
#         pkg_code1 = pd.unique(
#             consumables.loc[
#                 consumables["Intervention_Pkg"]
#                 == "First line treatment for new TB cases for children",
#                 "Intervention_Pkg_Code",
#             ]
#         )[0]
#
#         the_cons_footprint = {
#             "Intervention_Package_Code": [{pkg_code1: 1}],
#             "Item_Code": [],
#         }
#
#         is_cons_available = self.sim.modules["HealthSystem"].request_consumables(
#             hsi_event=self, cons_req_as_footprint=the_cons_footprint
#         )
#
#         if is_cons_available:
#
#             # treatment allocated for pulmonary tb
#             if (
#                 df.at[person_id, "is_alive"]
#                 and df.at[person_id, "tb_diagnosed"]
#                 and df.at[person_id, "tb_stage"] == "active_pulm"
#             ):
#
#                 df.at[person_id, "tb_on_treatment"] = True
#                 df.at[person_id, "tb_date_treated"] = now
#
#                 # schedule a 6-month event where people are cured, symptoms return to latent or not cured
#                 self.sim.schedule_event(
#                     TbCureEvent(self.module, person_id),
#                     self.sim.date + DateOffset(months=6),
#                 )
#
#                 # follow-up appts
#                 logger.debug(
#                     "....This is HSI_Tb_StartTreatmentChild: scheduling follow-up appointments for person %d",
#                     person_id,
#                 )
#
#                 # clinical monitoring
#                 followup_clin = HSI_Tb_FollowUp(self.module, person_id=person_id)
#
#                 # Request the health system to have this follow-up appointment
#                 weeks = params["followup_times"]["treatment_clinical"]
#                 cleaned_list = [x for x in weeks if x == x]  # remove nan
#
#                 for i in range(0, len(cleaned_list)):
#                     followup_appt_date = self.sim.date + DateOffset(
#                         months=cleaned_list[i]
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         followup_clin,
#                         priority=1,
#                         topen=followup_appt_date,
#                         tclose=followup_appt_date + DateOffset(days=3),
#                     )
#
#                 # repeat sputum smear tests
#                 followup_smear = HSI_Tb_FollowUp_SputumTest(
#                     self.module, person_id=person_id
#                 )
#
#                 # Request the health system to have this follow-up appointment
#                 weeks = params["followup_times"]["treatment_sputum"]
#                 cleaned_list = [x for x in weeks if x == x]  # remove nan
#
#                 for i in range(0, len(cleaned_list)):
#                     followup_appt_date = self.sim.date + DateOffset(
#                         months=cleaned_list[i]
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         followup_smear,
#                         priority=1,
#                         topen=followup_appt_date,
#                         tclose=followup_appt_date + DateOffset(days=3),
#                     )
#
#             # treatment allocated for extrapulmonary tb
#             if (
#                 df.at[person_id, "is_alive"]
#                 and df.at[person_id, "tb_diagnosed"]
#                 and df.at[person_id, "tb_stage"] == "active_extra"
#             ):
#                 df.at[person_id, "tb_on_treatment"] = True
#                 df.at[person_id, "tb_date_treated"] = now
#
#                 # schedule a 6-month event where people are cured, symptoms return to latent or not cured
#                 self.sim.schedule_event(
#                     TbCureEvent(self.module, person_id),
#                     self.sim.date + DateOffset(months=6),
#                 )
#
#                 # follow-up appts
#                 logger.debug(
#                     "....This is HSI_Tb_StartTreatmentChild: scheduling follow-up appointments for person %d",
#                     person_id,
#                 )
#
#                 # clinical monitoring
#                 followup_clin = HSI_Tb_FollowUp(self.module, person_id=person_id)
#
#                 # Request the health system to have this follow-up appointment
#                 weeks = params["followup_times"]["treatment_extrapulm"]
#                 cleaned_list = [x for x in weeks if x == x]  # remove nan
#
#                 for i in range(0, len(cleaned_list)):
#                     followup_appt_date = self.sim.date + DateOffset(
#                         months=cleaned_list[i]
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         followup_clin,
#                         priority=1,
#                         topen=followup_appt_date,
#                         tclose=followup_appt_date + DateOffset(days=3),
#                     )
#
#                 # repeat sputum smear tests
#                 followup_smear = HSI_Tb_FollowUp_SputumTest(
#                     self.module, person_id=person_id
#                 )
#
#                 # Request the health system to have this follow-up appointment
#                 weeks = params["followup_times"]["treatment_sputum"]
#                 cleaned_list = [x for x in weeks if x == x]  # remove nan
#
#                 for i in range(0, len(cleaned_list)):
#                     followup_appt_date = self.sim.date + DateOffset(
#                         months=cleaned_list[i]
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         followup_smear,
#                         priority=1,
#                         topen=followup_appt_date,
#                         tclose=followup_appt_date + DateOffset(days=3),
#                     )
#
#     def did_not_run(self):
#         pass
#
#
# class HSI_Tb_StartMdrTreatment(HSI_Event, IndividualScopeEventMixin):
#     """
#     This is a Health System Interaction Event - start tb-mdr treatment
#     """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#         assert isinstance(module, Tb)
#
#         # Get a blank footprint and then edit to define call on resources of this treatment event
#         the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
#         the_appt_footprint["TBFollowUp"] = 1
#
#         # Define the necessary information for an HSI
#         self.TREATMENT_ID = "Tb_MdrTreatmentInitiation"
#         self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
#         self.ACCEPTED_FACILITY_LEVEL = '1a'
#         self.ALERT_OTHER_DISEASES = []
#
#     def apply(self, person_id, squeeze_factor):
#         logger.debug("We are now ready to treat this tb case %d", person_id)
#
#         now = self.sim.date
#         df = self.sim.population.props
#         params = self.module.parameters
#
#         consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
#         pkg_code1 = pd.unique(
#             consumables.loc[
#                 consumables["Intervention_Pkg"] == "Case management of MDR cases",
#                 "Intervention_Pkg_Code",
#             ]
#         )[0]
#
#         the_cons_footprint = {
#             "Intervention_Package_Code": [{pkg_code1: 1}],
#             "Item_Code": [],
#         }
#
#         is_cons_available = self.sim.modules["HealthSystem"].request_consumables(
#             hsi_event=self, cons_req_as_footprint=the_cons_footprint
#         )
#
#         if is_cons_available:
#
#             # treatment allocated
#             if df.at[person_id, "is_alive"] and df.at[person_id, "tb_diagnosed"]:
#                 df.at[person_id, "tb_treated_mdr"] = True
#                 df.at[person_id, "tb_date_treated_mdr"] = now
#
#             # schedule a 6-month event where people are cured, symptoms return to latent or not cured
#             self.sim.schedule_event(
#                 TbCureEvent(self.module, person_id),
#                 self.sim.date + DateOffset(months=6),
#             )
#
#             # follow-up appts
#             logger.debug(
#                 "....This is HSI_Tb_StartMdrTreatment: scheduling follow-up appointments for person %d",
#                 person_id,
#             )
#
#             # clinical monitoring
#             followup_clin = HSI_Tb_FollowUp(self.module, person_id=person_id)
#
#             # Request the health system to have this follow-up appointment
#             weeks = params["followup_times"]["mdr_clinical"]
#             cleaned_list = [x for x in weeks if x == x]  # remove nan
#
#             for i in range(0, len(cleaned_list)):
#                 followup_appt_date = self.sim.date + DateOffset(months=cleaned_list[i])
#                 self.sim.modules["HealthSystem"].schedule_hsi_event(
#                     followup_clin,
#                     priority=1,
#                     topen=followup_appt_date,
#                     tclose=followup_appt_date + DateOffset(days=3),
#                 )
#
#             # repeat sputum smear tests
#             followup_smear = HSI_Tb_FollowUp_SputumTest(
#                 self.module, person_id=person_id
#             )
#
#             # Request the health system to have this follow-up appointment
#             weeks = params["followup_times"]["mdr_sputum"]
#             cleaned_list = [x for x in weeks if x == x]  # remove nan
#
#             for i in range(0, len(cleaned_list)):
#                 followup_appt_date = self.sim.date + DateOffset(months=cleaned_list[i])
#                 self.sim.modules["HealthSystem"].schedule_hsi_event(
#                     followup_smear,
#                     priority=1,
#                     topen=followup_appt_date,
#                     tclose=followup_appt_date + DateOffset(days=3),
#                 )
#
#     def did_not_run(self):
#         pass
#
#
# class HSI_Tb_RetreatmentAdult(HSI_Event, IndividualScopeEventMixin):
#     """
#     This is a Health System Interaction Event - start tb treatment
#     """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#         assert isinstance(module, Tb)
#
#         # Get a blank footprint and then edit to define call on resources of this treatment event
#         the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
#         the_appt_footprint["TBNew"] = 1  # This requires one out patient appt
#
#         # Define the necessary information for an HSI
#         self.TREATMENT_ID = "Tb_RetreatmentAdult"
#         self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
#         self.ACCEPTED_FACILITY_LEVEL = '1a'
#         self.ALERT_OTHER_DISEASES = []
#
#     def apply(self, person_id, squeeze_factor):
#         logger.debug("We are now ready to treat this tb case %d", person_id)
#
#         params = self.sim.modules["Tb"].parameters
#         now = self.sim.date
#         df = self.sim.population.props
#
#         consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
#         pkg_code1 = pd.unique(
#             consumables.loc[
#                 consumables["Intervention_Pkg"]
#                 == "First line treatment for retreatment TB cases for adults",
#                 "Intervention_Pkg_Code",
#             ]
#         )[0]
#
#         the_cons_footprint = {
#             "Intervention_Package_Code": [{pkg_code1: 1}],
#             "Item_Code": [],
#         }
#
#         is_cons_available = self.sim.modules["HealthSystem"].request_consumables(
#             hsi_event=self, cons_req_as_footprint=the_cons_footprint
#         )
#
#         if is_cons_available:
#
#             # treatment allocated
#             if df.at[person_id, "is_alive"] and df.at[person_id, "tb_diagnosed"]:
#                 df.at[person_id, "tb_on_treatment"] = True
#                 df.at[person_id, "tb_date_treated"] = now
#
#                 # schedule a 6-month event where people are cured, symptoms return to latent or not cured
#                 self.sim.schedule_event(
#                     TbCureEvent(self.module, person_id),
#                     self.sim.date + DateOffset(months=6),
#                 )
#
#                 # follow-up appts
#                 logger.debug(
#                     "....This is HSI_Tb_RetreatmentAdult: scheduling follow-up appointments for person %d",
#                     person_id,
#                 )
#
#                 # clinical monitoring
#                 followup_clin = HSI_Tb_FollowUp(self.module, person_id=person_id)
#
#                 # Request the health system to have this follow-up appointment
#                 weeks = params["followup_times"]["retreatment_clinical"]
#                 cleaned_list = [x for x in weeks if x == x]  # remove nan
#
#                 for i in range(0, len(cleaned_list)):
#                     followup_appt_date = self.sim.date + DateOffset(
#                         months=cleaned_list[i]
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         followup_clin,
#                         priority=1,
#                         topen=followup_appt_date,
#                         tclose=followup_appt_date + DateOffset(days=3),
#                     )
#
#                 # repeat sputum smear tests
#                 followup_smear = HSI_Tb_FollowUp_SputumTest(
#                     self.module, person_id=person_id
#                 )
#
#                 # Request the health system to have this follow-up appointment
#                 weeks = params["followup_times"]["retreatment_sputum"]
#                 cleaned_list = [x for x in weeks if x == x]  # remove nan
#
#                 for i in range(0, len(cleaned_list)):
#                     followup_appt_date = self.sim.date + DateOffset(
#                         months=cleaned_list[i]
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         followup_smear,
#                         priority=1,
#                         topen=followup_appt_date,
#                         tclose=followup_appt_date + DateOffset(days=3),
#                     )
#
#     def did_not_run(self):
#         pass
#
#
# class HSI_Tb_RetreatmentChild(HSI_Event, IndividualScopeEventMixin):
#     """
#     This is a Health System Interaction Event - start tb treatment
#     """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#         assert isinstance(module, Tb)
#
#         # Get a blank footprint and then edit to define call on resources of this treatment event
#         the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
#         the_appt_footprint["TBNew"] = 1  # New tb treatment initiation appt
#
#         # Define the necessary information for an HSI
#         self.TREATMENT_ID = "Tb_RetreatmentChild"
#         self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
#         self.ACCEPTED_FACILITY_LEVEL = '1a'
#         self.ALERT_OTHER_DISEASES = []
#
#     def apply(self, person_id, squeeze_factor):
#         logger.debug("We are now ready to treat this tb case %d", person_id)
#
#         params = self.sim.modules["Tb"].parameters
#         now = self.sim.date
#         df = self.sim.population.props
#
#         consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
#         pkg_code1 = pd.unique(
#             consumables.loc[
#                 consumables["Intervention_Pkg"]
#                 == "First line treatment for retreatment TB cases for children",
#                 "Intervention_Pkg_Code",
#             ]
#         )[0]
#
#         the_cons_footprint = {
#             "Intervention_Package_Code": [{pkg_code1: 1}],
#             "Item_Code": [],
#         }
#
#         is_cons_available = self.sim.modules["HealthSystem"].request_consumables(
#             hsi_event=self, cons_req_as_footprint=the_cons_footprint
#         )
#
#         if is_cons_available:
#
#             # treatment allocated
#             if df.at[person_id, "is_alive"] and df.at[person_id, "tb_diagnosed"]:
#                 df.at[person_id, "tb_on_treatment"] = True
#                 df.at[person_id, "tb_date_treated"] = now
#
#                 # schedule a 6-month event where people are cured, symptoms return to latent or not cured
#                 self.sim.schedule_event(
#                     TbCureEvent(self.module, person_id),
#                     self.sim.date + DateOffset(months=6),
#                 )
#
#                 # follow-up appts
#                 logger.debug(
#                     "....This is HSI_Tb_RetreatmentChild: scheduling follow-up appointments for person %d",
#                     person_id,
#                 )
#
#                 # clinical monitoring
#                 followup_clin = HSI_Tb_FollowUp(self.module, person_id=person_id)
#
#                 # Request the health system to have this follow-up appointment
#                 weeks = params["followup_times"]["retreatment_clinical"]
#                 cleaned_list = [x for x in weeks if x == x]  # remove nan
#
#                 for i in range(0, len(cleaned_list)):
#                     followup_appt_date = self.sim.date + DateOffset(
#                         months=cleaned_list[i]
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         followup_clin,
#                         priority=1,
#                         topen=followup_appt_date,
#                         tclose=followup_appt_date + DateOffset(days=3),
#                     )
#
#                 # repeat sputum smear tests
#                 followup_smear = HSI_Tb_FollowUp_SputumTest(
#                     self.module, person_id=person_id
#                 )
#
#                 # Request the health system to have this follow-up appointment
#                 weeks = params["followup_times"]["retreatment_sputum"]
#                 cleaned_list = [x for x in weeks if x == x]  # remove nan
#
#                 for i in range(0, len(cleaned_list)):
#                     followup_appt_date = self.sim.date + DateOffset(
#                         months=cleaned_list[i]
#                     )
#                     self.sim.modules["HealthSystem"].schedule_hsi_event(
#                         followup_smear,
#                         priority=1,
#                         topen=followup_appt_date,
#                         tclose=followup_appt_date + DateOffset(days=3),
#                     )
#
#     def did_not_run(self):
#         pass
#
#
# # ---------------------------------------------------------------------------
# #   Follow-up appts
# # ---------------------------------------------------------------------------
# class HSI_Tb_FollowUp(HSI_Event, IndividualScopeEventMixin):
#     """
#     This is a Health System Interaction Event - start tb treatment
#     """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#         assert isinstance(module, Tb)
#
#         # Get a blank footprint and then edit to define call on resources of this treatment event
#         the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
#         the_appt_footprint["TBFollowUp"] = 1
#
#         # Define the necessary information for an HSI
#         self.TREATMENT_ID = "Tb_FollowUp"
#         self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
#         self.ACCEPTED_FACILITY_LEVEL = '1a'
#         self.ALERT_OTHER_DISEASES = []
#
#     def apply(self, person_id, squeeze_factor):
#         # nothing needs to happen here, just log the appt
#         logger.debug("Follow up appt for tb case %d", person_id)
#
#     def did_not_run(self):
#         pass
#
#
# class HSI_Tb_FollowUp_SputumTest(HSI_Event, IndividualScopeEventMixin):
#     """
#     This is a follow-up sputum test for confirmed tb cases
#     doesn't change any properties except for date latest sputum test
#     """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#         assert isinstance(module, Tb)
#
#         # Get a blank footprint and then edit to define call on resources of this event
#         the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
#         the_appt_footprint[
#             "ConWithDCSA"
#         ] = 1  # This requires one generic outpatient appt
#         the_appt_footprint[
#             "LabTBMicro"
#         ] = 1  # This requires one lab appt for microscopy
#
#         # Define the necessary information for an HSI
#         self.TREATMENT_ID = "Tb_FollowUpSputumTest"
#         self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
#         self.ACCEPTED_FACILITY_LEVEL = '1a'
#         self.ALERT_OTHER_DISEASES = []
#
#     def apply(self, person_id, squeeze_factor):
#         logger.debug(
#             "This is HSI_Tb_FollowUp_SputumTest, a follow-up sputum smear test for person %d",
#             person_id,
#         )
#
#         df = self.sim.population.props
#         now = self.sim.date
#
#         # Get the consumables required
#         consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
#         pkg_code1 = pd.unique(
#             consumables.loc[
#                 consumables["Intervention_Pkg"] == "Microscopy Test",
#                 "Intervention_Pkg_Code",
#             ]
#         )[0]
#
#         the_cons_footprint = {
#             "Intervention_Package_Code": [{pkg_code1: 1}],
#             "Item_Code": [],
#         }
#
#         is_cons_available = self.sim.modules["HealthSystem"].request_consumables(
#             hsi_event=self, cons_req_as_footprint=the_cons_footprint
#         )
#
#         if is_cons_available:
#             df.at[person_id, "tb_ever_tested"] = True
#             df.at[person_id, "tb_smear_test"] = True
#             df.at[person_id, "tb_date_smear_test"] = now
#
#     def did_not_run(self):
#         pass
#
#
# # ---------------------------------------------------------------------------
# #   Cure
# # ---------------------------------------------------------------------------
#
#
# class TbCureEvent(Event, IndividualScopeEventMixin):
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#
#     def apply(self, person_id):
#         logger.debug("Stopping tb treatment and curing person %d", person_id)
#
#         df = self.sim.population.props
#         params = self.sim.modules["Tb"].parameters
#
#         if df.at[person_id, "is_alive"]:
#
#             # after six months of treatment, stop
#             df.at[person_id, "tb_on_treatment"] = False
#             cured = False
#
#             # ADULTS: if drug-susceptible and pulmonary tb:
#             # prob of tx success
#             if df.at[person_id, "tb_inf"].startswith("active_susc") and (
#                 df.at[person_id, "age_exact_years"] >= 15
#             ):
#
#                 # new case
#                 if (
#                     df.at[person_id, "tb_inf"] == "active_susc_new"
#                     and not df.at[person_id, "hv_inf"]
#                 ):
#                     cured = (
#                         self.module.rng.random_sample(size=1)
#                         < params["prob_tx_success_new"]
#                     )
#
#                 # previously treated case
#                 elif (
#                     df.at[person_id, "tb_inf"] == "active_susc_prev"
#                     and not df.at[person_id, "hv_inf"]
#                 ):
#                     cured = (
#                         self.module.rng.random_sample(size=1)
#                         < params["prob_tx_success_prev"]
#                     )
#
#                 # hiv+
#                 elif df.at[person_id, "hv_inf"]:
#                     cured = (
#                         self.module.rng.random_sample(size=1)
#                         < params["prob_tx_success_hiv"]
#                     )
#
#             # if under 5 years old
#             if df.at[person_id, "tb_inf"].startswith("active_susc") and (
#                 df.at[person_id, "age_exact_years"] < 4
#             ):
#                 cured = (
#                     self.module.rng.random_sample(size=1)
#                     < params["prob_tx_success_0_4"]
#                 )
#
#             # if between 5-14 years old
#             if (
#                 df.at[person_id, "tb_inf"].startswith("active_susc")
#                 and (df.at[person_id, "age_exact_years"] > 4)
#                 and (df.at[person_id, "age_exact_years"] < 15)
#             ):
#                 cured = (
#                     self.module.rng.random_sample(size=1)
#                     < params["prob_tx_success_5_14"]
#                 )
#
#             # if cured change properties
#             if cured:
#                 df.at[person_id, "tb_inf"] = "latent_susc_tx"
#                 df.at[person_id, "tb_diagnosed"] = False
#                 df.at[person_id, "tb_stage"] = "latent"
#                 df.at[person_id, "tb_treatment_failure"] = False
#                 df.at[person_id, "tb_symptoms"] = False
#                 df.at[person_id, "tb_smear"] = False
#                 df.at[person_id, "tb_diagnosed"] = False
#
#                 # check that tb symptoms are present and caused by tb before resolving
#                 if (
#                     "respiratory_symptoms"
#                     in self.sim.modules["SymptomManager"].has_what(person_id)
#                 ) & (
#                     "Tb"
#                     in self.sim.modules["SymptomManager"].causes_of(
#                         person_id, "respiratory_symptoms"
#                     )
#                 ):
#                     # this will clear all tb symptoms
#                     self.sim.modules["SymptomManager"].clear_symptoms(
#                         person_id=person_id, disease_module=self.module
#                     )
#
#             else:
#                 df.at[person_id, "tb_treatment_failure"] = True
#                 # request a repeat / Xpert test - follow-up
#                 # this will include drug-susceptible treatment failures and mdr-tb cases
#                 secondary_test = HSI_Tb_XpertTest(self.module, person_id=person_id)
#
#                 # Request the health system to give xpert test
#                 self.sim.modules["HealthSystem"].schedule_hsi_event(
#                     secondary_test, priority=1, topen=self.sim.date, tclose=None
#                 )
#
#                 # add back-up check if xpert is not available, then schedule sputum smear
#                 self.sim.schedule_event(
#                     TbCheckXpert(self.module, person_id),
#                     self.sim.date + DateOffset(weeks=2),
#                 )
#
#
# class TbCureMdrEvent(Event, IndividualScopeEventMixin):
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#
#     def apply(self, person_id):
#         logger.debug("Stopping tb-mdr treatment and curing person %d", person_id)
#
#         df = self.sim.population.props
#         params = self.sim.modules["Tb"].parameters
#
#         df.at[person_id, "tb_treated_mdr"] = False
#
#         cured = self.module.rng.random_sample(size=1) < params["prob_tx_success_mdr"]
#
#         if cured:
#             df.at[person_id, "tb_inf"] = "latent_mdr_tx"
#             df.at[person_id, "tb_diagnosed"] = False
#             df.at[person_id, "tb_stage"] = "latent"
#             df.at[person_id, "tb_treatment_failure"] = False
#             df.at[person_id, "tb_symptoms"] = False
#             df.at[person_id, "tb_smear"] = False
#             df.at[person_id, "tb_diagnosed"] = False
#
#             # check that tb symptoms are present and caused by tb before resolving
#             if (
#                 "respiratory_symptoms"
#                 in self.sim.modules["SymptomManager"].has_what(person_id)
#             ) & (
#                 "Tb"
#                 in self.sim.modules["SymptomManager"].causes_of(
#                     person_id, "respiratory_symptoms"
#                 )
#             ):
#                 # this will clear all tb symptoms
#                 self.sim.modules["SymptomManager"].clear_symptoms(
#                     person_id=person_id, disease_module=self.module
#                 )
#
#         else:
#             df.at[person_id, "tb_treatment_failure"] = True
#
#             # request a repeat / Xpert test - follow-up
#             secondary_test = HSI_Tb_XpertTest(self.module, person_id=person_id)
#
#             # Request the health system to give xpert test
#             self.sim.modules["HealthSystem"].schedule_hsi_event(
#                 secondary_test, priority=1, topen=self.sim.date, tclose=None
#             )
#
#             # add back-up check if xpert is not available, then schedule sputum smear
#             self.sim.schedule_event(
#                 TbCheckXpert(self.module, person_id),
#                 self.sim.date + DateOffset(weeks=2),
#             )
#
#
# #
# # ---------------------------------------------------------------------------
# #   IPT
# # ---------------------------------------------------------------------------
# class HSI_Tb_Ipt(HSI_Event, IndividualScopeEventMixin):
#     """
#         This is a Health System Interaction Event - give ipt to contacts of tb cases for 6 months
#     """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#         assert isinstance(module, Tb)
#
#         # Get a blank footprint and then edit to define call on resources of this treatment event
#         the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
#         the_appt_footprint["Over5OPD"] = 1  # This requires one out patient appt
#
#         # Define the necessary information for an HSI
#         self.TREATMENT_ID = "Tb_Ipt"
#         self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
#         self.ACCEPTED_FACILITY_LEVEL = '1a'
#         self.ALERT_OTHER_DISEASES = []
#
#     def apply(self, person_id, squeeze_factor):
#         logger.debug("Starting IPT for person %d", person_id)
#
#         df = self.sim.population.props
#
#         consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
#         pkg_code1 = pd.unique(
#             consumables.loc[
#                 consumables["Items"] == "Isoniazid Preventive Therapy",
#                 "Intervention_Pkg_Code",
#             ]
#         )[0]
#
#         the_cons_footprint = {
#             "Intervention_Package_Code": [{pkg_code1: 1}],
#             "Item_Code": [],
#         }
#
#         is_cons_available = self.sim.modules["HealthSystem"].request_consumables(
#             hsi_event=self, cons_req_as_footprint=the_cons_footprint
#         )
#
#         if is_cons_available:
#             df.at[person_id, "tb_on_ipt"] = True
#             df.at[person_id, "tb_date_ipt"] = self.sim.date
#
#             # schedule end date of ipt after six months
#             self.sim.schedule_event(
#                 TbIptEndEvent(self.module, person_id),
#                 self.sim.date + DateOffset(months=6),
#             )
#
#     def did_not_run(self):
#         pass
#
#
# class HSI_Tb_IptHiv(HSI_Event, IndividualScopeEventMixin):
#     """
#         This is a Health System Interaction Event - give ipt to hiv+ persons
#         called by hiv module when starting ART (adults and children)
#         """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#         # assert isinstance(module, Tb)
#
#         # Get a blank footprint and then edit to define call on resources of this treatment event
#         the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
#         the_appt_footprint["Over5OPD"] = 1  # This requires one out patient appt
#
#         # Define the necessary information for an HSI
#         self.TREATMENT_ID = "Tb_IptHiv"
#         self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
#         self.ACCEPTED_FACILITY_LEVEL = '1a'
#         self.ALERT_OTHER_DISEASES = []
#
#     def apply(self, person_id, squeeze_factor):
#
#         df = self.sim.population.props
#
#         consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
#         pkg_code1 = pd.unique(
#             consumables.loc[
#                 consumables["Intervention_Pkg"]
#                 == "Isoniazid preventative therapy for HIV+ no TB",
#                 "Intervention_Pkg_Code",
#             ]
#         )[0]
#
#         the_cons_footprint = {
#             "Intervention_Package_Code": [{pkg_code1: 1}],
#             "Item_Code": [],
#         }
#
#         is_cons_available = self.sim.modules["HealthSystem"].request_consumables(
#             hsi_event=self, cons_req_as_footprint=the_cons_footprint
#         )
#
#         if is_cons_available:
#
#             # only if not currently tb_active
#             if not df.at[person_id, "tb_inf"].startswith("active"):
#
#                 logger.debug(
#                     "HSI_Tb_IptHiv: starting IPT for HIV+ person %d", person_id
#                 )
#
#                 df.at[person_id, "tb_on_ipt"] = True
#                 df.at[person_id, "tb_date_ipt"] = self.sim.date
#
#                 # schedule end date of ipt after six months and repeat call
#                 self.sim.schedule_event(
#                     TbIptEndEvent(self.module, person_id),
#                     self.sim.date + DateOffset(months=6),
#                 )
#
#             else:
#                 logger.debug(
#                     "HSI_Tb_IptHiv: person %d has active TB and can't start IPT",
#                     person_id,
#                 )
#
#     def did_not_run(self):
#         pass
#
#
# class TbIptEndEvent(Event, IndividualScopeEventMixin):
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#
#     def apply(self, person_id):
#         logger.debug("Stopping ipt for person %d", person_id)
#
#         df = self.sim.population.props
#
#         df.at[person_id, "tb_on_ipt"] = False
#
#         # if hiv+ reschedule HSI_Tb_IptHiv to continue IPT
#         if df.at[person_id, "hv_inf"]:
#             logger.debug(
#                 "TbIptEndEvent: scheduling further IPT for person %d on date %s",
#                 person_id,
#                 self.sim.date,
#             )
#
#             ipt_start = HSI_Tb_IptHiv(self.module, person_id=person_id)
#
#             # Request the health system to have this follow-up appointment
#             self.sim.modules["HealthSystem"].schedule_hsi_event(
#                 ipt_start, priority=1, topen=self.sim.date, tclose=None
#             )
#
#
# # ---------------------------------------------------------------------------
# #   Deaths
# # ---------------------------------------------------------------------------
#
#
# class TbDeathEvent(RegularEvent, PopulationScopeEventMixin):
#     """The regular event that kills people with active drug-susceptible tb
#     HIV-positive deaths due to TB counted as HIV deaths
#     """
#
#     def __init__(self, module):
#         super().__init__(module, frequency=DateOffset(months=1))
#
#     def apply(self, population):
#         params = self.module.parameters
#         df = population.props
#         now = self.sim.date
#         rng = self.module.rng
#
#         # ---------------------------------- TB DEATHS - HIV-NEGATIVE ------------------------------------
#         # only active infections result in death
#         mortality_rate = pd.Series(0, index=df.index)
#
#         # hiv-negative, tb untreated
#         mortality_rate.loc[
#             df["tb_inf"].str.contains("active_susc")
#             & ~df.hv_inf
#             & (~df.tb_on_treatment | ~df.tb_treated_mdr)
#             & ~df.hv_on_cotrim
#         ] = params["monthly_prob_tb_mortality"]
#
#         # hiv-negative on cotrim - shouldn't be any, tb untreated
#         mortality_rate.loc[
#             df["tb_inf"].str.contains("active_susc")
#             & ~df.hv_inf
#             & (~df.tb_on_treatment | ~df.tb_treated_mdr)
#             & df.hv_on_cotrim
#         ] = (params["monthly_prob_tb_mortality"] * params["mort_cotrim"])
#
#         # hiv-negative, tb treated
#         mortality_rate.loc[
#             df["tb_inf"].str.contains("active_susc")
#             & ~df.hv_inf
#             & (df.tb_on_treatment | df.tb_treated_mdr)
#             & ~df.hv_on_cotrim
#         ] = params["monthly_prob_tb_mortality"]
#
#         # Generate a series of random numbers, one per individual
#         probs = rng.rand(len(df))
#         deaths = df.is_alive & (probs < mortality_rate)
#         # print('deaths: ', deaths)
#         will_die = (df[deaths]).index
#         # print('will_die: ', will_die)
#
#         for person in will_die:
#             if df.at[person, "is_alive"]:
#                 df.at[person, "tb_date_death_occurred"] = self.sim.date
#
#                 self.sim.schedule_event(
#                     demography.InstantaneousDeath(
#                         self.module, individual_id=person, cause="tb"
#                     ),
#                     now,
#                 )
#                 df.at[person, "tb_date_death"] = now
#
#         # ---------------------------------- HIV-TB DEATHS ------------------------------------
#         # only active infections result in death, no deaths on treatment
#         mort_hiv = pd.Series(0, index=df.index)
#
#         # hiv-positive, untreated
#         mort_hiv.loc[
#             df["tb_inf"].str.contains("active_susc")
#             & df.hv_inf
#             & (~df.tb_on_treatment | ~df.tb_treated_mdr)
#             & (df.hv_on_art != 2)
#             & ~df.hv_on_cotrim
#         ] = params["monthly_prob_tb_mortality_hiv"]
#
#         # hiv-positive, on ART
#         mort_hiv.loc[
#             df["tb_inf"].str.contains("active_susc")
#             & df.hv_inf
#             & (~df.tb_on_treatment | ~df.tb_treated_mdr)
#             & (df.hv_on_art == 2)
#             & ~df.hv_on_cotrim
#         ] = params["monthly_prob_tb_mortality"]
#
#         # hiv-positive on cotrim and not ART - shouldn't happen
#         mort_hiv.loc[
#             df["tb_inf"].str.contains("active_susc")
#             & df.hv_inf
#             & (~df.tb_on_treatment | ~df.tb_treated_mdr)
#             & (df.hv_on_art != 2)
#             & df.hv_on_cotrim
#         ] = (params["monthly_prob_tb_mortality_hiv"] * params["mort_cotrim"])
#
#         # Generate a series of random numbers, one per individual
#         probs = rng.rand(len(df))
#         deaths = df.is_alive & (probs < mortality_rate)
#         # print('deaths: ', deaths)
#         will_die = (df[deaths]).index
#
#         for person in will_die:
#             if df.at[person, "is_alive"]:
#                 df.at[person, "tb_date_death_occurred"] = self.sim.date
#
#                 self.sim.schedule_event(
#                     demography.InstantaneousDeath(
#                         self.module, individual_id=person, cause="hiv"
#                     ),
#                     now,
#                 )
#                 df.at[person, "tb_date_death"] = now
#
#
# class TbMdrDeathEvent(RegularEvent, PopulationScopeEventMixin):
#     """The regular event that kills people with active MDR tb
#     HIV-positive deaths due to TB counted as HIV deaths
#     """
#
#     def __init__(self, module):
#         super().__init__(module, frequency=DateOffset(months=1))
#
#     def apply(self, population):
#         params = self.module.parameters
#         df = population.props
#         now = self.sim.date
#         rng = self.module.rng
#
#         # ---------------------------------- TB DEATHS - HIV-NEGATIVE ------------------------------------
#         # only active infections result in death
#         mortality_rate = pd.Series(0, index=df.index)
#
#         # hiv-negative, tb untreated
#         mortality_rate.loc[
#             df["tb_inf"].str.contains("active_mdr")
#             & ~df.hv_inf
#             & ~df.tb_treated_mdr
#             & ~df.hv_on_cotrim
#         ] = params["monthly_prob_tb_mortality"]
#
#         # hiv-negative on cotrim - shouldn't be any, tb untreated
#         mortality_rate.loc[
#             df["tb_inf"].str.contains("active_mdr")
#             & ~df.hv_inf
#             & ~df.tb_treated_mdr
#             & df.hv_on_cotrim
#         ] = (params["monthly_prob_tb_mortality"] * params["mort_cotrim"])
#
#         # hiv-negative, tb treated
#         mortality_rate.loc[
#             df["tb_inf"].str.contains("active_mdr")
#             & ~df.hv_inf
#             & df.tb_treated_mdr
#             & ~df.hv_on_cotrim
#         ] = (params["monthly_prob_tb_mortality"] * params["mort_tx"])
#
#         # Generate a series of random numbers, one per individual
#         probs = rng.rand(len(df))
#         deaths = df.is_alive & (probs < mortality_rate)
#         # print('deaths: ', deaths)
#         will_die = (df[deaths]).index
#         # print('will_die: ', will_die)
#
#         for person in will_die:
#             if df.at[person, "is_alive"]:
#                 df.at[person, "tb_date_death_occurred"] = self.sim.date
#
#                 self.sim.schedule_event(
#                     demography.InstantaneousDeath(
#                         self.module, individual_id=person, cause="tb"
#                     ),
#                     now,
#                 )
#                 df.at[person, "tb_date_death"] = now
#
#         # ---------------------------------- HIV-TB DEATHS ------------------------------------
#         # only active infections result in death, no deaths on treatment
#         mort_hiv = pd.Series(0, index=df.index)
#
#         # hiv-positive no ART, tb untreated
#         mort_hiv.loc[
#             df["tb_inf"].str.contains("active_mdr")
#             & df.hv_inf
#             & (~df.tb_on_treatment | ~df.tb_treated_mdr)
#             & (df.hv_on_art != 2)
#             & ~df.hv_on_cotrim
#         ] = params["monthly_prob_tb_mortality_hiv"]
#
#         # hiv-positive on ART, tb untreated
#         mort_hiv.loc[
#             df["tb_inf"].str.contains("active_mdr")
#             & df.hv_inf
#             & (~df.tb_on_treatment | ~df.tb_treated_mdr)
#             & (df.hv_on_art == 2)
#             & ~df.hv_on_cotrim
#         ] = params["monthly_prob_tb_mortality"]
#
#         # hiv-positive on cotrim and ART, tb untreated
#         mort_hiv.loc[
#             df["tb_inf"].str.contains("active_mdr")
#             & df.hv_inf
#             & (~df.tb_on_treatment | ~df.tb_treated_mdr)
#             & (df.hv_on_art == 2)
#             & df.hv_on_cotrim
#         ] = (params["monthly_prob_tb_mortality_hiv"] * params["mort_cotrim"])
#
#         # hiv-positive no ART, tb treated
#         mort_hiv.loc[
#             df["tb_inf"].str.contains("active_mdr")
#             & df.hv_inf
#             & df.tb_treated_mdr
#             & (df.hv_on_art == 2)
#             & ~df.hv_on_cotrim
#         ] = (params["monthly_prob_tb_mortality"] * params["mort_tx"])
#
#         # Generate a series of random numbers, one per individual
#         probs = rng.rand(len(df))
#         deaths = df.is_alive & (probs < mortality_rate)
#         # print('deaths: ', deaths)
#         will_die = (df[deaths]).index
#
#         for person in will_die:
#             if df.at[person, "is_alive"]:
#                 df.at[person, "tb_date_death_occurred"] = self.sim.date
#
#                 self.sim.schedule_event(
#                     demography.InstantaneousDeath(
#                         self.module, individual_id=person, cause="hiv"
#                     ),
#                     now,
#                 )
#                 df.at[person, "tb_date_death"] = now
#
#
# # ---------------------------------------------------------------------------
# #   Logging
# # ---------------------------------------------------------------------------
#
#
# class TbLoggingEvent(RegularEvent, PopulationScopeEventMixin):
#     def __init__(self, module):
#         """ produce some outputs to check
#         """
#         # run this event every 12 months
#         self.repeat = 12
#         super().__init__(module, frequency=DateOffset(months=self.repeat))
#
#     def apply(self, population):
#         # get some summary statistics
#         df = population.props
#         now = self.sim.date
#
#         # ------------------------------------ INCIDENCE ------------------------------------
#         # total number of new active cases in last year - susc + mdr
#         # may have died in the last year but still counted as active case for the year
#         new_tb_cases = len(
#             df[(df.tb_date_active > (now - DateOffset(months=self.repeat)))]
#         )
#
#         # incidence per 100k
#         inc100k = (new_tb_cases / len(df[df.is_alive])) * 100000
#
#         # latent cases
#         new_latent_cases = len(
#             df[(df.tb_date_latent > (now - DateOffset(months=self.repeat)))]
#         )
#
#         # percentage of active TB cases in the last year who are HIV-positive
#         inc_active_hiv = len(
#             df[(df.tb_date_active > (now - DateOffset(months=self.repeat))) & df.hv_inf]
#         )
#
#         prop_hiv = inc_active_hiv / new_tb_cases if new_tb_cases else 0
#
#         # incidence of TB-HIV per 100k
#         inc100k_hiv = (inc_active_hiv / len(df[df.is_alive])) * 100000
#
#         # proportion of new active cases that are mdr-tb
#         # technically this is EVER HAD MDR doesn't mean the last episode necessarily
#         inc_active_mdr = len(
#             df[
#                 df.tb_ever_tb_mdr
#                 & (df.tb_date_active > (now - DateOffset(months=self.repeat)))
#             ]
#         )
#
#         if new_tb_cases > 0:
#             prop_inc_active_mdr = inc_active_mdr / new_tb_cases
#         else:
#             prop_inc_active_mdr = 0
#
#         assert prop_inc_active_mdr <= 1
#
#         logger.info(
#             "%s|tb_incidence|%s",
#             now,
#             {
#                 "tbNewActiveCases": new_tb_cases,
#                 "tbNewLatentCases": new_latent_cases,
#                 "tbIncActive100k": inc100k,
#                 "tbIncActive100k_hiv": inc100k_hiv,
#                 "tbPropIncActiveMdr": prop_inc_active_mdr,
#                 "tb_prop_hiv_pos": prop_hiv,
#             },
#         )
#
#         # ------------------------------------ PREVALENCE ------------------------------------
#         # prevalence should be the number of clinically active cases that occurred in the past year
#         # ACTIVE
#         new_tb_cases = len(
#             df[(df.tb_date_active > (now - DateOffset(months=self.repeat)))]
#         )
#
#         # num_active = len(df[(df.tb_inf.str.contains('active')) & df.is_alive])
#         prop_active = new_tb_cases / len(df[df.is_alive])
#
#         assert prop_active <= 1
#
#         # todo: change to adjusted prevalence counting # episodes / pop
#         # proportion of adults with active tb
#         num_active_adult = len(
#             df[(df.tb_inf.str.contains("active")) & (df.age_years >= 15) & df.is_alive]
#         )
#         prop_active_adult = num_active_adult / len(
#             df[(df.age_years >= 15) & df.is_alive]
#         )
#         assert prop_active_adult <= 1
#
#         # proportion of children with active tb
#         num_active_child = len(
#             df[(df.tb_inf.str.contains("active")) & (df.age_years < 15) & df.is_alive]
#         )
#         prop_active_child = num_active_child / len(
#             df[(df.age_years < 15) & df.is_alive]
#         )
#         assert prop_active_child <= 1
#
#         # proportion of hiv cases with active tb
#         num_active_hiv = len(
#             df[(df.tb_inf.str.contains("active")) & df.hv_inf & df.is_alive]
#         )
#
#         if (num_active_hiv > 0) & (len(df[df.hv_inf & df.is_alive]) > 0):
#             prop_hiv_tb = num_active_hiv / len(df[df.hv_inf & df.is_alive])
#         else:
#             prop_hiv_tb = 0
#
#         # proportion of population with active TB by age/sex - compare to WHO 2018
#         m_num_cases = (
#             df[df.is_alive & (df.sex == "M") & (df.tb_inf.str.contains("active"))]
#             .groupby("age_range")
#             .size()
#         )
#         f_num_cases = (
#             df[df.is_alive & (df.sex == "F") & (df.tb_inf.str.contains("active"))]
#             .groupby("age_range")
#             .size()
#         )
#
#         m_pop = df[df.is_alive & (df.sex == "M")].groupby("age_range").size()
#         f_pop = df[df.is_alive & (df.sex == "F")].groupby("age_range").size()
#
#         m_prop_active = m_num_cases / m_pop
#         m_prop_active = m_prop_active.fillna(0)
#         f_prop_active = f_num_cases / f_pop
#         f_prop_active = f_prop_active.fillna(0)
#
#         logger.info("%s|tb_propActiveTbMale|%s", self.sim.date, m_prop_active.to_dict())
#
#         logger.info(
#             "%s|tb_propActiveTbFemale|%s", self.sim.date, f_prop_active.to_dict()
#         )
#
#         # LATENT
#
#         # proportion of population with latent TB - all pop
#         num_latent = len(df[(df.tb_inf.str.contains("latent")) & df.is_alive])
#         prop_latent = num_latent / len(df[df.is_alive])
#         assert prop_latent <= 1
#
#         # proportion of population with latent TB - adults
#         num_latent_adult = len(
#             df[(df.tb_inf.str.contains("latent")) & (df.age_years >= 15) & df.is_alive]
#         )
#         prop_latent_adult = num_latent_adult / len(
#             df[(df.age_years >= 15) & df.is_alive]
#         )
#         assert prop_latent_adult <= 1
#
#         # proportion of population with latent TB - children
#         num_latent_child = len(
#             df[(df.tb_inf.str.contains("latent")) & (df.age_years < 15) & df.is_alive]
#         )
#         prop_latent_child = num_latent_child / len(
#             df[(df.age_years < 15) & df.is_alive]
#         )
#         assert prop_latent_child <= 1
#
#         logger.info(
#             "%s|tb_prevalence|%s",
#             now,
#             {
#                 "tbPropActive": prop_active,
#                 "tbPropActiveAdult": prop_active_adult,
#                 "tbPropActiveChild": prop_active_child,
#                 "tbPropActiveHiv": prop_hiv_tb,
#                 "tbPropLatent": prop_latent,
#                 "tbPropLatentAdult": prop_latent_adult,
#                 "tbPropLatentChild": prop_latent_child,
#             },
#         )
#
#         # ------------------------------------ TREATMENT ------------------------------------
#         # number of new cases in the last time period / number new treatment starts in last time period
#
#         # percentage all cases which occurred in last time period and treated in last time period
#         new_tx = len(df[(df.tb_date_treated > (now - DateOffset(months=self.repeat)))])
#
#         percent_treated = ((new_tx / new_tb_cases) * 100) if new_tb_cases else 0
#         # assert percent_treated <= 100
#
#         # percentage all adult cases which occurred in last time period and treated in last time period
#         new_tb_cases_adult = len(
#             df[
#                 (df.age_years >= 15)
#                 & (df.tb_date_active > (now - DateOffset(months=self.repeat)))
#             ]
#         )
#
#         new_tb_tx_adult = len(
#             df[
#                 (df.age_years >= 15)
#                 & (df.tb_date_treated > (now - DateOffset(months=self.repeat)))
#             ]
#         )
#
#         percent_treated_adult = (
#             ((new_tb_tx_adult / new_tb_cases_adult) * 100) if new_tb_cases_adult else 0
#         )
#         # assert percent_treated_adult <= 100
#
#         # percentage all child cases which occurred in last time period and treated in last time period
#         new_tb_cases_child = len(
#             df[
#                 (df.age_years < 15)
#                 & (df.tb_date_active > (now - DateOffset(months=self.repeat)))
#             ]
#         )
#
#         new_tb_tx_child = len(
#             df[
#                 (df.age_years < 15)
#                 & (df.tb_date_treated > (now - DateOffset(months=self.repeat)))
#             ]
#         )
#
#         percent_treated_child = (
#             ((new_tb_tx_child / new_tb_cases_child) * 100) if new_tb_cases_child else 0
#         )
#         # assert percent_treated_child <= 100
#
#         logger.info(
#             "%s|tb_treatment|%s",
#             now,
#             {
#                 "tbTreat": percent_treated,
#                 "tbTreatAdult": percent_treated_adult,
#                 "tbTreatChild": percent_treated_child,
#             },
#         )
#
#         # ------------------------------------ BCG ------------------------------------
#         # bcg vaccination coverage in <1 year old children
#         bcg = len(df[df.is_alive & df.tb_bcg & (df.age_years <= 1)])
#         infants = len(df[df.is_alive & (df.age_years <= 1)])
#
#         coverage = ((bcg / infants) * 100) if infants else 0
#         assert coverage <= 100
#
#         logger.info(
#             "%s|tb_bcg|%s",
#             now,
#             {
#                 "tbNumInfantsBcg": bcg,
#                 "tbNumInfantsEligibleBcg": infants,
#                 "tbBcgCoverage": coverage,
#             },
#         )
#
#         # ------------------------------------ MORTALITY ------------------------------------
#         # tb deaths (incl hiv) reported in the last 12 months / pop size
#         deaths = len(df[(df.tb_date_death > (now - DateOffset(months=self.repeat)))])
#
#         mort_rate100k = (deaths / len(df[df.is_alive])) * 100000
#
#         # tb deaths (hiv+ only) reported in the last 12 months / pop size
#         deaths_tb_hiv = len(
#             df[df.hv_inf & (df.tb_date_death > (now - DateOffset(months=self.repeat)))]
#         )
#
#         mort_rate_tb_hiv100k = (deaths_tb_hiv / len(df[df.is_alive])) * 100000
#
#         logger.info(
#             "%s|tb_mortality|%s",
#             now,
#             {
#                 "tbMortRate100k": mort_rate100k,
#                 "tbMortRateHiv100k": mort_rate_tb_hiv100k,
#             },
#         )
