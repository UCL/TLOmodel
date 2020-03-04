"""
HIV infection event
"""
import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd

from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography, tb
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Hiv(Module):
    """
    baseline hiv infection
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        # self.beta_calib = par_est

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        # baseline characteristics
        "hiv_prev_2010": Parameter(Types.REAL, "adult hiv prevalence in 2010"),
        "time_inf": Parameter(
            Types.REAL, "prob of time since infection for baseline adult pop"
        ),
        "child_hiv_prev2010": Parameter(Types.REAL, "adult hiv prevalence in 2010"),
        "testing_coverage_male": Parameter(
            Types.REAL, "proportion of adult male population tested"
        ),
        "testing_coverage_female": Parameter(
            Types.REAL, "proportion of adult female population tested"
        ),
        "initial_art_coverage": Parameter(Types.REAL, "coverage of ART at baseline"),
        "vls_m": Parameter(Types.REAL, "rates of viral load suppression males"),
        "vls_f": Parameter(Types.REAL, "rates of viral load suppression males"),
        "vls_child": Parameter(
            Types.REAL, "rates of viral load suppression in children 0-14 years"
        ),
        # natural history
        "beta": Parameter(Types.REAL, "transmission rate"),
        "exp_rate_mort_infant_fast_progressor": Parameter(
            Types.REAL,
            "Exponential rate parameter for mortality in infants fast progressors",
        ),
        "weibull_scale_mort_infant_slow_progressor": Parameter(
            Types.REAL,
            "Weibull scale parameter for mortality in infants slow progressors",
        ),
        "weibull_shape_mort_infant_slow_progressor": Parameter(
            Types.REAL,
            "Weibull shape parameter for mortality in infants slow progressors",
        ),
        "weibull_shape_mort_adult": Parameter(
            Types.REAL, "Weibull shape parameter for mortality in adults"
        ),
        "prob_mtct_untreated": Parameter(
            Types.REAL, "probability of mother to child transmission"
        ),
        "prob_mtct_treated": Parameter(
            Types.REAL, "probability of mother to child transmission, mother on ART"
        ),
        "prob_mtct_incident_preg": Parameter(
            Types.REAL,
            "probability of mother to child transmission, mother infected during pregnancy",
        ),
        "monthly_prob_mtct_bf_untreated": Parameter(
            Types.REAL,
            "probability of mother to child transmission during breastfeeding",
        ),
        "monthly_prob_mtct_bf_treated": Parameter(
            Types.REAL,
            "probability of mother to child transmission, mother infected during breastfeeding",
        ),
        # behavioural parameters
        "proportion_female_sex_workers": Parameter(
            Types.REAL, "proportion of women who engage in transactional sex"
        ),
        "fsw_transition": Parameter(
            Types.REAL, "annual rate at which women leave sex work"
        ),
        # relative risk of HIV acquisition
        "rr_fsw": Parameter(Types.REAL, "relative risk of hiv with female sex work"),
        "rr_circumcision": Parameter(
            Types.REAL, "relative risk of hiv with circumcision"
        ),
        "rr_behaviour_change": Parameter(
            Types.REAL, "relative risk of hiv with behaviour modification"
        ),
        "rr_condom": Parameter(Types.REAL, "relative risk hiv with condom use"),
        "rr_rural": Parameter(Types.REAL, "relative risk of hiv in rural location"),
        "rr_windex_poorer": Parameter(
            Types.REAL, "relative risk of hiv with wealth level poorer"
        ),
        "rr_windex_middle": Parameter(
            Types.REAL, "relative risk of hiv with wealth level middle"
        ),
        "rr_windex_richer": Parameter(
            Types.REAL, "relative risk of hiv with wealth level richer"
        ),
        "rr_windex_richest": Parameter(
            Types.REAL, "relative risk of hiv with wealth level richest"
        ),
        "rr_sex_f": Parameter(Types.REAL, "relative risk of hiv if female"),
        "rr_age_gp20": Parameter(
            Types.REAL, "relative risk of hiv if age 20-24 compared with 15-19"
        ),
        "rr_age_gp25": Parameter(Types.REAL, "relative risk of hiv if age 25-29"),
        "rr_age_gp30": Parameter(Types.REAL, "relative risk of hiv if age 30-34"),
        "rr_age_gp35": Parameter(Types.REAL, "relative risk of hiv if age 35-39"),
        "rr_age_gp40": Parameter(Types.REAL, "relative risk of hiv if age 40-44"),
        "rr_age_gp45": Parameter(Types.REAL, "relative risk of hiv if age 45-49"),
        "rr_age_gp50": Parameter(Types.REAL, "relative risk of hiv if age 50+"),
        "rr_edlevel_primary": Parameter(
            Types.REAL, "relative risk of hiv with primary education"
        ),
        "rr_edlevel_secondary": Parameter(
            Types.REAL, "relative risk of hiv with secondary education"
        ),
        "rr_edlevel_higher": Parameter(
            Types.REAL, "relative risk of hiv with higher education"
        ),
        "hv_behav_mod": Parameter(
            Types.REAL, "change in force of infection with behaviour modification"
        ),
        "testing_adj": Parameter(
            Types.REAL, "additional HIV testing outside generic appts"
        ),
        "treatment_prob": Parameter(
            Types.REAL, "probability of requesting ART following positive HIV test"
        ),
        # health system interactions
        "prob_high_to_low_art": Parameter(
            Types.REAL, "prob of transitioning from good adherence to poor adherence"
        ),
        "prob_low_to_high_art": Parameter(
            Types.REAL, "prob of transitioning from poor adherence to good adherence"
        ),
        "prob_off_art": Parameter(Types.REAL, "prob of transitioning off ART"),
        "vl_monitoring_times": Parameter(
            Types.INT, "times(months) viral load monitoring required after ART start"
        ),
        "fsw_prep": Parameter(Types.REAL, "prob of fsw receiving PrEP"),
        "hiv_art_ipt": Parameter(
            Types.REAL, "proportion of hiv-positive cases on ART also on IPT"
        ),
        "tb_high_risk_distr": Parameter(Types.REAL, "high-risk districts giving IPT"),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        "hv_inf": Property(Types.BOOL, "hiv status"),
        "hv_date_inf": Property(Types.DATE, "Date acquired hiv infection"),
        "hv_date_symptoms": Property(Types.DATE, "Date of symptom start"),
        "hv_proj_date_death": Property(
            Types.DATE, "Projected time of AIDS death if untreated"
        ),
        "hv_sexual_risk": Property(
            Types.CATEGORICAL, "Sexual risk groups", categories=["low", "sex_work"]
        ),
        "hv_mother_inf_by_birth": Property(Types.BOOL, "hiv status of mother"),
        "hv_mother_art": Property(
            Types.CATEGORICAL, "mother's art status", categories=[0, 1, 2]
        ),
        "hv_proj_date_symp": Property(Types.DATE, "Date becomes symptomatic"),
        "hv_proj_date_aids": Property(Types.DATE, "Date develops AIDS"),
        "hv_ever_tested": Property(Types.BOOL, "ever had a hiv test"),
        "hv_date_tested": Property(Types.DATE, "date of hiv test"),
        "hv_number_tests": Property(Types.INT, "number of hiv tests taken"),
        "hv_diagnosed": Property(Types.BOOL, "hiv+ and tested"),
        "hv_on_art": Property(Types.CATEGORICAL, "art status", categories=[0, 1, 2]),   # TODO: turn these into string categories to do with adherance
        "hv_date_art_start": Property(Types.DATE, "date art started"),
        "hv_viral_load": Property(Types.DATE, "date last viral load test"),
        "hv_on_cotrim": Property(Types.BOOL, "on cotrimoxazole"),
        "hv_date_cotrim": Property(Types.DATE, "date cotrimoxazole started"),
        "hv_fast_progressor": Property(Types.BOOL, "infant fast progressor"),
        "hv_behaviour_change": Property(
            Types.BOOL, "Exposed to hiv prevention counselling"
        ),
        "hv_date_death_occurred": Property(
            Types.DATE, "date death due to AIDS actually occurred"
        ),
        "hv_on_prep": Property(Types.BOOL, "person is currently receiving PrEP")
    }

    SYMPTOMS = {"pre_aids", "aids"}   # TODO; rename this and use them consistently

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        :param data_folder: path of a folder supplied to the Simulation containing data files.
        """

        workbook = pd.read_excel(
            os.path.join(self.resourcefilepath, "ResourceFile_HIV.xlsx"),
            sheet_name=None,
        )
        self.load_parameters_from_dataframe(workbook["parameters"])

        p = self.parameters

        # Build Linear Models:
        # TODO: Check this linear model - esp the ed-level which looks odd in it selection of levels
        self.LinearModels = dict()
        self.LinearModels['relative_risk_of_acquiring_HIV_among_those_not_infected'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1.0,
            Predictor('hv_sexual_risk').when('sex_work', p["rr_fsw"]),
            Predictor('mc_is_circumcised').when(True, p["rr_circumcision"]),
            Predictor('li_urban').when(False, p["rr_rural"]),
            Predictor('sex').when('F', p["rr_sex_f"]),
            Predictor('hv_behaviour_change').when(True, p["rr_behaviour_change"]),
            Predictor('li_ed_lev')  .when("2", p["rr_edlevel_primary"])
                                    .when("3", p["rr_edlevel_secondary"]),
            Predictor('li_wealth')  .when("2", p["rr_windex_poorer"])
                                    .when("3", p["rr_windex_middle"])
                                    .when("4", p["rr_windex_richer"])
                                    .when("5", p["rr_windex_richest"]),
            Predictor('age_years')  .when('.between(20, 24)', p["rr_age_gp20"])
                                    .when('.between(25, 29)', p["rr_age_gp25"])
                                    .when('.between(30, 34)', p["rr_age_gp30"])
                                    .when('.between(35, 39)', p["rr_age_gp35"])
                                    .when('.between(40, 44)', p["rr_age_gp40"])
                                    .when('.between(45, 50)', p["rr_age_gp45"])
                                    .when('> 50', p["rr_age_gp50"])
        )

        self.LinearModels['risk_of_transmitting_HIV_by_those_infected'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p["beta"],
            Predictor('age_years')  .when('>=15', 1.0)
                                    .otherwise(0.0),
            # Predictor('hv_on_art').when(True, 0.0)              # <---TODO: depends on the useage of hv_on_art property
        )


        # Load Tables for establishing the baseline characteristics
        p["hiv_prev"] = pd.read_csv(
            Path(self.resourcefilepath) / "ResourceFile_HIV_prevalence.csv"
        )
        p["time_inf"] = workbook["timeSinceInf2010"]
        p["initial_art_coverage"] = pd.read_csv(
            Path(self.resourcefilepath) / "ResourceFile_HIV_coverage.csv"
        )


        # DALY Weights
        self.daly_wts = dict()
        if "HealthBurden" in self.sim.modules.keys():
            # Symptomatic HIV without anemia
            self.daly_wts["pre_aids"] = self.sim.modules["HealthBurden"].get_daly_weight(17)

            # AIDS without antiretroviral treatment without anemia
            self.daly_wts["aids"] = self.sim.modules["HealthBurden"].get_daly_weight(19)

            assert set(self.daly_wts.keys()) == self.SYMPTOMS


        # health system interactions
        p["vl_monitoring_times"] = workbook["VL_monitoring"]
        p["tb_high_risk_distr"] = workbook["IPTdistricts"]

        # Register this disease module with the health system
        self.sim.modules["HealthSystem"].register_disease_module(self)

    def initialise_population(self, population):
        """Set our property values for the initial population.
        """
        df = population.props

        df["hv_inf"] = False
        df["hv_date_inf"] = pd.NaT
        df["hv_date_symptoms"] = pd.NaT
        df["hv_proj_date_death"] = pd.NaT
        df["hv_sexual_risk"].values[:] = "low"
        df["hv_mother_inf_by_birth"] = False
        df["hv_mother_art"].values[:] = 0

        df["hv_proj_date_symp"] = pd.NaT
        df["hv_proj_date_aids"] = pd.NaT

        df["hv_ever_tested"] = False  # default: no individuals tested
        df["hv_date_tested"] = pd.NaT
        df["hv_number_tests"] = 0
        df["hv_diagnosed"] = False
        df["hv_on_art"].values[:] = 0
        df["hv_date_art_start"] = pd.NaT
        df["hv_on_cotrim"] = False
        df["hv_date_cotrim"] = pd.NaT
        df["hv_fast_progressor"] = False
        df["hv_behaviour_change"] = False
        df["hv_date_death_occurred"] = pd.NaT
        df["hv_on_prep"] = False

        self.baseline_prevalence(population)            # allocate baseline prevalence
        self.fsw(population)                            # allocate some women as female-sex-workers
        self.baseline_tested(population)                # allocate baseline art coverage
        self.baseline_art(population)                   # allocate baseline art coverage
        self.initial_pop_deaths_children(population)    # add death dates for children
        self.initial_pop_deaths_adults(population)      # add death dates for adults
        self.assign_symptom_level(population)           # assign symptom level for all infected

        # Set the natural history events for those who are infected
        for person_id in df.loc[df['hv_inf'] & df['is_alive']].index:
            self.set_natural_history(person_id=person_id)

    def log_scale(self, a0):
        """ helper function for adult mortality rates"""
        age_scale = 2.55 - 0.025 * (a0 - 30)
        return age_scale

    def fsw(self, population):
        """ Assign female sex work to sample of women and change sexual risk
        """
        df = population.props

        fsw = (
            df[
                df.is_alive
                & (df.sex == "F")
                & (df.age_years.between(15, 49))
                & (df.li_mar_stat != 2)
            ]
            .sample(
                frac=self.parameters["proportion_female_sex_workers"], replace=False
            )
            .index
        )

        df.loc[fsw, "hv_sexual_risk"] = "sex_work"

    def baseline_prevalence(self, population):
        """
        Assign baseline HIV prevalence and Date of Infection
        """

        now = self.sim.date
        df = population.props
        params = self.parameters

        # ----------------------------------- ADULT HIV -----------------------------------
        # TODO A linear model here
        prevalence = params["hiv_prev_2010"]

        # only for 15-54
        risk_hiv = pd.Series(0, index=df.index)
        risk_hiv.loc[
            df.is_alive & df.age_years.between(15, 55)
        ] = 1  # applied to all adults
        risk_hiv.loc[(df.hv_sexual_risk == "sex_work")] *= params["rr_fsw"]
        risk_hiv.loc[df.mc_is_circumcised] *= params["rr_circumcision"]
        # risk_hiv.loc[(df.contraception == 'condom')] *= params['rr_condom']
        risk_hiv.loc[~df.li_urban] *= params["rr_rural"]
        risk_hiv.loc[(df.li_wealth == "2")] *= params["rr_windex_poorer"]
        risk_hiv.loc[(df.li_wealth == "3")] *= params["rr_windex_middle"]
        risk_hiv.loc[(df.li_wealth == "4")] *= params["rr_windex_richer"]
        risk_hiv.loc[(df.li_wealth == "5")] *= params["rr_windex_richest"]
        risk_hiv.loc[(df.sex == "F")] *= params["rr_sex_f"]
        risk_hiv.loc[df.age_years.between(20, 24)] *= params["rr_age_gp20"]
        risk_hiv.loc[df.age_years.between(25, 29)] *= params["rr_age_gp25"]
        risk_hiv.loc[df.age_years.between(30, 34)] *= params["rr_age_gp30"]
        risk_hiv.loc[df.age_years.between(35, 39)] *= params["rr_age_gp35"]
        risk_hiv.loc[df.age_years.between(40, 44)] *= params["rr_age_gp40"]
        risk_hiv.loc[df.age_years.between(45, 50)] *= params["rr_age_gp45"]
        risk_hiv.loc[(df.age_years >= 50)] *= params["rr_age_gp50"]
        risk_hiv.loc[(df.li_ed_lev == "2")] *= params["rr_edlevel_primary"]
        risk_hiv.loc[(df.li_ed_lev == "3")] *= params[
            "rr_edlevel_secondary"
        ]  # li_ed_lev=3 secondary and higher

        # sample 10% prev, weight the likelihood of being sampled by the relative risk
        eligible = df.index[df.is_alive & df.age_years.between(15, 80)]
        norm_p = pd.Series(risk_hiv[eligible])
        norm_p /= norm_p.sum()  # normalise
        infected_idx = self.rng.choice(
            eligible, size=int(prevalence * (len(eligible))), replace=False, p=norm_p
        )

        # print('infected_idx', infected_idx)
        # test = infected_idx.isnull().sum()  # sum number of nan
        # print("number of nan: ", test)
        df.loc[infected_idx, "hv_inf"] = True

        # for time since infection use prob of incident inf 2000-2010
        inf_adult = df.index[df.is_alive & df.hv_inf & (df.age_years >= 15)]

        year_inf = self.rng.choice(
            self.time_inf["year"],
            size=len(inf_adult),
            replace=True,
            p=self.time_inf["scaled_prob"],
        )

        df.loc[inf_adult, "hv_date_inf"] = now - pd.to_timedelta(year_inf, unit="y")



        # ----------------------------------- CHILD HIV -----------------------------------

        # baseline children's prevalence from spectrum outputs
        prevalence = self.hiv_prev.loc[
            self.hiv_prev.year == now.year, ["age_from", "sex", "prev_prop"]
        ]

        # merge all susceptible individuals with their hiv probability based on sex and age
        df_hivprob = df.merge(
            prevalence,
            left_on=["age_years", "sex"],
            right_on=["age_from", "sex"],
            how="left",
        )

        # fill missing values with 0 (only relevant for age 80+)
        df_hivprob["prev_prop"] = df_hivprob["prev_prop"].fillna(0)

        assert (
            df_hivprob.prev_prop.isna().sum() == 0
        )  # check there is a probability for every individual

        # get a list of random numbers between 0 and 1 for each infected individual
        random_draw = self.rng.random_sample(size=len(df_hivprob))

        # probability of hiv > random number, assign hiv_inf = True
        # TODO: cluster this by mother's hiv status?? currently no linked mother pre-baseline year
        hiv_index = df_hivprob.index[
            df.is_alive
            & (random_draw < df_hivprob.prev_prop)
            & df_hivprob.age_years.between(0, 14)
        ]
        # print(hiv_index)

        df.loc[hiv_index, "hv_inf"] = True
        df.loc[hiv_index, "hv_date_inf"] = df.loc[hiv_index, "date_of_birth"]
        df.loc[hiv_index, "hv_fast_progressor"] = False

    def baseline_tested(self, population):
        """ assign initial hiv testing levels, only for adults
        """
        now = self.sim.date
        df = population.props

        # get a list of random numbers between 0 and 1 for the whole population
        random_draw = self.rng.random_sample(size=len(df))

        # probability of baseline population ever testing for HIV
        test_index_male = df.index[
            (random_draw < self.parameters["testing_coverage_male"])
            & df.is_alive
            & (df.sex == "M")
            & (df.age_years >= 15)
        ]
        # print('art_index: ', art_index)

        test_index_female = df.index[
            (random_draw < self.parameters["testing_coverage_female"])
            & df.is_alive
            & (df.sex == "F")
            & (df.age_years >= 15)
        ]

        # we don't know date tested, assume date = now
        df.loc[test_index_male | test_index_female, "hv_ever_tested"] = True
        df.loc[test_index_male | test_index_female, "hv_date_tested"] = now
        df.loc[test_index_male | test_index_female, "hv_number_tests"] = 1

        # outcome of test
        diagnosed_idx = df.index[df.hv_ever_tested & df.is_alive & df.hv_inf]
        df.loc[diagnosed_idx, "hv_diagnosed"] = True

    def baseline_art(self, population):
        """ assign initial art coverage levels
        """
        now = self.sim.date
        df = population.props

        worksheet = self.parameters["initial_art_coverage"]

        coverage = worksheet.loc[
            worksheet.year == now.year, ["year", "single_age", "sex", "prop_coverage"]
        ]
        # print('coverage: ', coverage.head(20))

        # merge all susceptible individuals with their coverage probability based on sex and age
        df_art = df.merge(
            coverage,
            left_on=["age_years", "sex"],
            right_on=["single_age", "sex"],
            how="left",
        )

        # no data for ages 100+ so fill missing values with 0
        df_art["prop_coverage"] = df_art["prop_coverage"].fillna(0)
        # print('df_with_age_art_prob: ', df_with_age_art_prob.head(20))

        assert (
            df_art.prop_coverage.isna().sum() == 0
        )  # check there is a probability for every individual

        # get a list of random numbers between 0 and 1 for the whole population
        random_draw = self.rng.random_sample(size=len(df_art))

        # ----------------------------------- ART - CHILDREN -----------------------------------

        # no testing rates for children so assign ever_tested=True if allocated treatment
        art_idx_child = df_art.index[
            (random_draw < df_art.prop_coverage)
            & df_art.is_alive
            & df_art.hv_inf
            & df_art.age_years.between(0, 14)
        ]

        df.loc[art_idx_child, "hv_on_art"] = 2  # assumes all are adherent at baseline
        df.loc[art_idx_child, "hv_date_art_start"] = now
        df.loc[art_idx_child, "hv_ever_tested"] = True
        df.loc[art_idx_child, "hv_date_tested"] = now
        df.loc[art_idx_child, "hv_number_tests"] = 1
        df.loc[art_idx_child, "hv_diagnosed"] = True

        # ----------------------------------- ART - ADULTS -----------------------------------

        # probability of adult baseline population receiving art: requirement = hiv_diagnosed
        art_idx_adult = df_art.index[
            (random_draw < df_art.prop_coverage)
            & df_art.is_alive
            & df_art.hv_inf
            & df_art.hv_diagnosed
            & df_art.age_years.between(15, 64)
        ]

        df.loc[
            art_idx_adult, "hv_on_art"
        ] = 2  # assumes all are adherent, then stratify into category 1/2
        df.loc[art_idx_adult, "hv_date_art_start"] = now

        del df_art
        # ----------------------------------- ADHERENCE -----------------------------------

        # allocate proportion to non-adherent category
        # if condition not added, error with small numbers to sample
        if (
            len(df[df.is_alive & (df.hv_on_art == 2) & (df.age_years.between(0, 14))])
            > 5
        ):
            idx_c = (
                df[df.is_alive & (df.hv_on_art == 2) & (df.age_years.between(0, 14))]
                .sample(frac=(1 - self.parameters["vls_child"]))
                .index
            )
            df.loc[idx_c, "hv_on_art"] = 1  # change to non=adherent

        if (
            len(
                df[
                    df.is_alive
                    & (df.hv_on_art == 2)
                    & (df.sex == "M")
                    & (df.age_years.between(15, 64))
                ]
            )
            > 5
        ):
            idx_m = (
                df[
                    df.is_alive
                    & (df.hv_on_art == 2)
                    & (df.sex == "M")
                    & (df.age_years.between(15, 64))
                ]
                .sample(frac=(1 - self.parameters["vls_m"]))
                .index
            )
            df.loc[idx_m, "hv_on_art"] = 1  # change to non=adherent

        if (
            len(
                df[
                    df.is_alive
                    & (df.hv_on_art == 2)
                    & (df.sex == "F")
                    & (df.age_years.between(15, 64))
                ]
            )
            > 5
        ):
            idx_f = (
                df[
                    df.is_alive
                    & (df.hv_on_art == 2)
                    & (df.sex == "F")
                    & (df.age_years.between(15, 64))
                ]
                .sample(frac=(1 - self.parameters["vls_f"]))
                .index
            )
            df.loc[idx_f, "hv_on_art"] = 1  # change to non=adherent

    def initial_pop_deaths_children(self, population):
        """ assign death dates to baseline hiv-infected population - INFANTS
        assume all are slow progressors, otherwise time to death shorter than time infected
        only assign death dates if not already on ART
        """
        df = population.props
        now = self.sim.date
        params = self.parameters

        # PAEDIATRIC time of death - untreated
        infants = df.index[
            df.is_alive & df.hv_inf & (df.hv_on_art != 2) & (df.age_years < 15)
        ]

        # need a two parameter Weibull with size parameter, multiply by scale instead
        time_death_slow = (
            self.rng.weibull(
                a=params["weibull_shape_mort_infant_slow_progressor"], size=len(infants)
            )
            * params["weibull_scale_mort_infant_slow_progressor"]
        )

        time_death_slow = pd.Series(time_death_slow, index=infants)
        time_infected = now - df.loc[infants, "hv_date_inf"]

        # while time of death is shorter than time infected - redraw
        while np.any(
            time_infected > (pd.to_timedelta(time_death_slow * 365.25, unit="d"))
        ):
            redraw = time_infected.index[
                time_infected > (pd.to_timedelta(time_death_slow * 365.25, unit="d"))
            ]

            new_time_death_slow = (
                self.rng.weibull(
                    a=params["weibull_shape_mort_infant_slow_progressor"],
                    size=len(redraw),
                )
                * params["weibull_scale_mort_infant_slow_progressor"]
            )

            time_death_slow[redraw] = new_time_death_slow

        time_death_slow = pd.to_timedelta(time_death_slow * 365.25, unit="d")

        # remove microseconds
        time_death_slow = pd.Series(time_death_slow).dt.floor("S")
        df.loc[infants, "hv_proj_date_death"] = (
            df.loc[infants, "hv_date_inf"] + time_death_slow
        )

    def initial_pop_deaths_adults(self, population):
        """ assign death dates to baseline hiv-infected population - ADULTS
        only assign if not on ART
        """
        df = population.props
        now = self.sim.date
        params = self.parameters

        # adults are all those aged >=15
        hiv_ad = df.index[
            df.is_alive & df.hv_inf & (df.hv_on_art != 2) & (df.age_years >= 15)
        ]

        time_of_death = self.rng.weibull(
            a=params["weibull_shape_mort_adult"], size=len(hiv_ad)
        ) * np.exp(self.log_scale(df.loc[hiv_ad, "age_years"]))

        time_infected = now - df.loc[hiv_ad, "hv_date_inf"]

        # while time of death is shorter than time infected - redraw
        while np.any(
            time_infected > (pd.to_timedelta(time_of_death * 365.25, unit="d"))
        ):
            redraw = time_infected.index[
                time_infected > (pd.to_timedelta(time_of_death * 365.25, unit="d"))
            ]

            new_time_of_death = self.rng.weibull(
                a=params["weibull_shape_mort_adult"], size=len(redraw)
            ) * np.exp(self.log_scale(df.loc[redraw, "age_years"]))

            time_of_death[redraw] = new_time_of_death

        time_of_death = pd.to_timedelta(time_of_death * 365.25, unit="d")

        # remove microseconds
        time_of_death = pd.Series(time_of_death).dt.floor("S")

        df.loc[hiv_ad, "hv_proj_date_death"] = (
            df.loc[hiv_ad, "hv_date_inf"] + time_of_death
        )

    def assign_symptom_level(self, population):
        """ assign level of symptoms to infected people: chronic or aids
        only for untreated people
        """
        pass

    def schedule_symptoms(self, population):
        """ assign level of symptoms to infected people
        """
        pass

    def initialise_simulation(self, sim):
        """Get ready for simulation start.
        """

        # Schedule the events for the natural history of persons already infected
        df = sim.population.props
        for person_id in df.loc[df['is_alive'] & df['hv_inf']].index:
            self.set_natural_history(person_id)

        # Schedule Main Polling Event
        sim.schedule_event(HivMainPollingEvent(self), sim.date + DateOffset(months=12))

        # Schedule Logging Event
        sim.schedule_event(HivLoggingEvent(self), sim.date)

        # sim.schedule_event(HivMtctEvent(self), sim.date + DateOffset(months=12))
        # sim.schedule_event(
        #     HivArtGoodToPoorAdherenceEvent(self), sim.date + DateOffset(months=12)
        # )
        # sim.schedule_event(
        #     HivArtPoorToGoodAdherenceEvent(self), sim.date + DateOffset(months=12)
        # )
        # sim.schedule_event(
        #     HivTransitionOffArtEvent(self), sim.date + DateOffset(months=12)
        # )
        # sim.schedule_event(FswEvent(self), sim.date + DateOffset(months=12))
        #
        # sim.schedule_event(HivScheduleTesting(self), sim.date + DateOffset(days=1))


        # # Schedule the event that will launch the Outreach event
        # outreach_event = HivLaunchOutreachEvent(self)
        # self.sim.schedule_event(outreach_event, self.sim.date + DateOffset(months=12))

        # To launch the Behaviour change event across a target population:
        # def target_fn(person_id, population):
        #     # Receives a person_id and returns True/False to indicate whether that person is to be included
        #     return 15 <= population.at[person_id, 'age_years'] <= 50 and (not population.at[person_id, 'hv_inf'])
        #
        # population_level_HSI_event = HSI_Hiv_PopulationWideBehaviourChange(self, target_fn=target_fn)
        #
        # self.sim.modules['HealthSystem'].schedule_hsi_event(hsi_event=population_level_HSI_event,
        #                                                     priority=0,
        #                                                     topen=self.sim.date + DateOffset(months=12),
        #                                                     tclose=None)

        # # Schedule the occurrence of a population wide change in risk that goes through the health system:
        # popwide_hsi_event = HSI_Hiv_PopulationWideBehaviourChange(self, target_fn=None)
        # self.sim.modules["HealthSystem"].schedule_hsi_event(
        #     popwide_hsi_event, priority=1, topen=self.sim.date, tclose=None
        # )
        # logger.debug(
        #     "HSI_Hiv_PopulationWideBehaviourChange has been scheduled successfully!"
        # )
        #
        # # Schedule the event that will launch the PrEP event (2018 onwards)
        # prep_event = HivLaunchPrepEvent(self)
        # self.sim.schedule_event(prep_event, self.sim.date + DateOffset(years=8))
        #
        # df = sim.population.props
        # inf = df.index[df.is_alive & df.hv_inf & (df.hv_on_art != 2)]




    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual
        """
        params = self.parameters
        df = self.sim.population.props
        now = self.sim.date

        # default settings
        df.at[child_id, "hv_inf"] = False
        df.at[child_id, "hv_date_inf"] = pd.NaT
        df.at[child_id, "hv_date_symptoms"] = pd.NaT
        df.at[child_id, "hv_proj_date_death"] = pd.NaT
        df.at[child_id, "hv_sexual_risk"] = "low"
        df.at[child_id, "hv_mother_inf_by_birth"] = False
        df.at[child_id, "hv_mother_art"] = 0

        df.at[child_id, "hv_specific_symptoms"] = "none"

        df.at[child_id, "hv_proj_date_symp"] = pd.NaT
        df.at[child_id, "hv_proj_date_aids"] = pd.NaT

        df.at[child_id, "hv_ever_tested"] = False  # default: no individuals tested
        df.at[child_id, "hv_date_tested"] = pd.NaT
        df.at[child_id, "hv_number_tests"] = 0
        df.at[child_id, "hv_diagnosed"] = False
        df.at[child_id, "hv_on_art"] = 0
        df.at[child_id, "hv_date_art_start"] = pd.NaT
        df.at[child_id, "hv_on_cotrim"] = False
        df.at[child_id, "hv_date_cotrim"] = pd.NaT
        df.at[child_id, "hv_fast_progressor"] = False
        df.at[child_id, "hv_behaviour_change"] = False
        df.at[child_id, "hv_date_death_occurred"] = pd.NaT
        df.at[child_id, "hv_on_prep"] = False

        if df.at[mother_id, "hv_inf"]:
            df.at[child_id, "hv_mother_inf_by_birth"] = True
            if df.at[mother_id, "hv_on_art"] == 2:
                df.at[child_id, "hv_mother_art"] = 2

        # ----------------------------------- MTCT - PREGNANCY -----------------------------------

        #  TRANSMISSION DURING PREGNANCY / DELIVERY
        random_draw = self.rng.random_sample(size=1)

        #  mother has incident infection during pregnancy, NO ART
        if (
            (random_draw < params["prob_mtct_incident_preg"])
            and df.at[child_id, "is_alive"]
            and df.at[child_id, "hv_mother_inf_by_birth"]
            and (df.at[child_id, "hv_mother_art"] != 2)
            and (((now - df.at[mother_id, "hv_date_inf"]) / np.timedelta64(1, "M")) < 9)
        ):
            df.at[child_id, "hv_inf"] = True

        # mother has existing infection, mother NOT ON ART
        if (
            (random_draw < params["prob_mtct_untreated"])
            and df.at[child_id, "is_alive"]
            and df.at[child_id, "hv_mother_inf_by_birth"]
            and not df.at[child_id, "hv_inf"]
            and (df.at[child_id, "hv_mother_art"] != 2)
        ):
            df.at[child_id, "hv_inf"] = True

        #  mother has existing infection, mother ON ART
        if (
            (random_draw < params["prob_mtct_treated"])
            and df.at[child_id, "is_alive"]
            and df.at[child_id, "hv_mother_inf_by_birth"]
            and not df.at[child_id, "hv_inf"]
            and (df.at[child_id, "hv_mother_art"] == 2)
        ):
            df.at[child_id, "hv_inf"] = True

        # ----------------------------------- ASSIGN DEATHS  -----------------------------------

        if df.at[child_id, "is_alive"] and df.at[child_id, "hv_inf"]:
            df.at[child_id, "hv_date_inf"] = self.sim.date

            # assume all FAST PROGRESSORS, draw from exp, returns an array not a single value!!
            time_death = self.rng.exponential(
                scale=params["exp_rate_mort_infant_fast_progressor"], size=1
            )
            df.at[child_id, "hv_fast_progressor"] = True
            df.at[child_id, "hv_specific_symptoms"] = "aids"

            time_death = pd.to_timedelta(time_death[0] * 365.25, unit="d")
            df.at[child_id, "hv_proj_date_death"] = now + time_death

            # schedule the death event
            death = HivAidsDeathEvent(
                self, individual_id=child_id, cause="hiv"
            )  # make that death event
            death_scheduled = df.at[child_id, "hv_proj_date_death"]
            self.sim.schedule_event(death, death_scheduled)  # schedule the death

        # ----------------------------------- PMTCT -----------------------------------
        # first contact is testing, then schedule treatment / prophylaxis as needed
        # then if child infected, schedule ART
        if (
            df.at[child_id, "hv_mother_inf_by_birth"]
            and not df.at[child_id, "hv_diagnosed"]
        ):
            event = HSI_Hiv_InfantScreening(self, person_id=child_id)
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                event,
                priority=1,
                topen=self.sim.date,
                tclose=self.sim.date + DateOffset(weeks=4),
            )

    def on_hsi_alert(self, person_id, treatment_id):

        logger.debug(
            "This is hiv, being alerted about a health system interaction "
            "person %d for: %s",
            person_id,
            treatment_id,
        )

        if treatment_id == "Tb_Testing":
            piggy_back_dx_at_appt = HSI_Hiv_PresentsForCareWithSymptoms(self, person_id)
            piggy_back_dx_at_appt.TREATMENT_ID = "Hiv_PiggybackAppt"

            # Arbitrarily reduce the size of appt footprint
            for key in piggy_back_dx_at_appt.EXPECTED_APPT_FOOTPRINT:
                piggy_back_dx_at_appt.EXPECTED_APPT_FOOTPRINT[key] = (
                    piggy_back_dx_at_appt.EXPECTED_APPT_FOOTPRINT[key] * 0.25
                )

            self.sim.modules["HealthSystem"].schedule_hsi_event(
                piggy_back_dx_at_appt, priority=0, topen=self.sim.date, tclose=None
            )

    def report_daly_values(self):
        """
        DALYS are based on whether the person is experiencing symptoms of pre-AIDS or AIDS
        :return:
        """
        df = self.sim.population.props
        health_values = pd.Series(index=df[df.is_alive].index, data=0.0)

        for symptom_string in self.SYMPTOMS:
            health_values.loc[self.sim.modules['SymptomManager'].who_has(symptom_string)] += self.daly_wts[symptom_string]

        health_values.name = ""  # label the cause of this disability

        return health_values.loc[df.is_alive]

    def set_natural_history(self, person_id):
        """
        This helper function sets the natural history events of HIV and is called when someone is infected.
        It will:
        * Schedule the time of Death if there is not treatment (based on a draw from a distribution)
        * Schedule the time of AIDS (one year prior to death)
        * Schedule the time of Pre-AIDS (one year prior to AIDS)

        The key inputs (date-of-birth and date-of-infection and found in sim.population.props)
        :return: None
        """

        df = self.sim.population.props
        assert df.at[person_id, 'is_alive'], 'The person is not alive.'

        # Find date of AIDS Death
        date_of_aidsdeath = self.sim.date + DateOffset(years=10)
        date_of_aids = date_of_aidsdeath - DateOffset(years=1)
        date_of_preaids = date_of_aids - DateOffset(years=1)

        # Schedule events for these stages of disease progression:
        self.sim.schedule_event(HivAidsDeathEvent(self, person_id), date_of_aidsdeath)
        self.sim.schedule_event(HivAidsEvent(self, person_id), date_of_aids)
        self.sim.schedule_event( HivPreAidsEvent(self, person_id), date_of_aids)


        """ # ----------------------------------- TIME OF DEATH -----------------------------------
        death_date = rng.weibull(
            a=params["weibull_shape_mort_adult"], size=len(newly_infected_index)
        ) * np.exp(self.module.log_scale(df.loc[newly_infected_index, "age_years"]))

        death_date = pd.to_timedelta(death_date * 365.25, unit="d")

        death_date = pd.Series(death_date).dt.floor("S")  # remove microseconds
        df.loc[newly_infected_index, "hv_proj_date_death"] = (
            df.loc[newly_infected_index, "hv_date_inf"] + death_date
        )

        death_dates = df.hv_proj_date_death[newly_infected_index]

        # schedule the death event
        for person in newly_infected_index:
            # print('person', person)
            death = HivAidsDeathEvent(
                self.module, individual_id=person, cause="hiv"
            )  # make that death event
            time_death = death_dates[person]
            # print('time_death: ', time_death)
            # print('now: ', now)
            self.sim.schedule_event(death, time_death)  # schedule the death

        # ----------------------------------- PROGRESSION TO SYMPTOMATIC -----------------------------------
        df.loc[newly_infected_index, "hv_proj_date_symp"] = df.loc[
            newly_infected_index, "hv_proj_date_death"
        ] - DateOffset(days=732.5)

        # schedule the symptom update event for each person
        for person_index in newly_infected_index:
            symp_event = HivPreAidsEvent(self.module, person_index)

            if df.at[person_index, "hv_proj_date_symp"] < self.sim.date:
                df.at[person_index, "hv_proj_date_symp"] = self.sim.date + DateOffset(
                    days=1
                )
            # print('symp_date', df.at[person_index, 'hv_proj_date_symp'])
            self.sim.schedule_event(
                symp_event, df.at[person_index, "hv_proj_date_symp"]
            )

        # ----------------------------------- PROGRESSION TO AIDS -----------------------------------
        df.loc[newly_infected_index, "hv_proj_date_aids"] = df.loc[
            newly_infected_index, "hv_proj_date_death"
        ] - DateOffset(days=365.25)

        # schedule the symptom update event for each person
        for person_index in newly_infected_index:
            aids_event = HivAidsEvent(self.module, person_index)

            if df.at[person_index, "hv_proj_date_aids"] < self.sim.date:
                df.at[person_index, "hv_proj_date_aids"] = self.sim.date + DateOffset(
                    days=1
                )
            # print('aids_date', df.at[person_index, 'hv_proj_date_aids'])
            self.sim.schedule_event(
                aids_event, df.at[person_index, "hv_proj_date_aids"]
            )"""

        """

        # now = self.sim.date
        # params = self.parameters

        # ----------------------------------- PROGRESSION TO SYMPTOMATIC -----------------------------------
        # not needed if already chronic/aids, so symptoms = 'none' only
        idx = df.index[
            df.is_alive
            & df.hv_inf
            & (df.hv_specific_symptoms == "none")
            & (df.hv_on_art != 2)
        ]

        df.loc[idx, "hv_proj_date_symp"] = df.loc[
            idx, "hv_proj_date_death"
        ] - DateOffset(days=732.5)

        # schedule the symptom update event for each person
        for person_index in idx:
            symp_event = HivPreAidsEvent(self, person_index)
            if df.at[person_index, "hv_proj_date_symp"] < self.sim.date:
                df.at[person_index, "hv_proj_date_symp"] = self.sim.date + DateOffset(
                    days=1
                )
            self.sim.schedule_event(
                symp_event, df.at[person_index, "hv_proj_date_symp"]
            )

        # ----------------------------------- PROGRESSION TO AIDS -----------------------------------
        # not needed if already aids, so symptoms = 'none' or 'chronic' only
        idx = df.index[
            df.is_alive
            & df.hv_inf
            & (df.hv_specific_symptoms != "aids")
            & (df.hv_on_art != 2)
        ]

        df.loc[idx, "hv_proj_date_aids"] = df.loc[
            idx, "hv_proj_date_death"
        ] - DateOffset(days=365.25)

        # schedule the symptom update event for each person
        for person_index in idx:
            aids_event = HivAidsEvent(self, person_index)
            if df.at[person_index, "hv_proj_date_aids"] < self.sim.date:
                df.at[person_index, "hv_proj_date_aids"] = self.sim.date + DateOffset(
                    days=1
                )
            self.sim.schedule_event(
                aids_event, df.at[person_index, "hv_proj_date_aids"]
            )

        """


        """        df = population.props
        now = self.sim.date
        # params = self.parameters

        # ----------------------------------- ADULT SYMPTOMS -----------------------------------
        adults = df[
            df.is_alive & df.hv_inf & (df.hv_on_art != 2) & (df.age_years >= 15)
        ].index

        # if <2 years from scheduled death = chronic
        time_death = (
            df.loc[adults, "hv_proj_date_death"] - now
        ).dt.days  # returns days
        chronic = time_death < (2 * 365.25)
        idx = adults[chronic]
        df.loc[idx, "hv_specific_symptoms"] = "symp"

        # if <1 year from scheduled death = aids
        time_death = (
            df.loc[adults, "hv_proj_date_death"] - now
        ).dt.days  # returns days
        aids = time_death < 365.25
        idx = adults[aids]
        df.loc[idx, "hv_specific_symptoms"] = "aids"

        # ----------------------------------- CHILD SYMPTOMS -----------------------------------
        # baseline pop - infants, all assumed slow progressors

        infants = df[
            df.is_alive & df.hv_inf & (df.hv_on_art != 2) & (df.age_years < 15)
        ].index

        # if <2 years from scheduled death = chronic
        time_death = (
            df.loc[infants, "hv_proj_date_death"] - now
        ).dt.days  # returns days
        chronic = time_death < (2 * 365.25)
        idx = infants[chronic]
        df.loc[idx, "hv_specific_symptoms"] = "symp"

        # if <1 year from scheduled death = aids
        time_death = (
            df.loc[infants, "hv_proj_date_death"] - now
        ).dt.days  # returns days
        aids = time_death < 365.25
        idx = infants[aids]
        df.loc[idx, "hv_specific_symptoms"] = "aids"
        """


# ---------------------------------------------------------------------------
#   Events
# ---------------------------------------------------------------------------
class HivMainPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This the Main Polling Event. It occurs once per year and it will:
    * TODO FILL THIS IN
    *
    *
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))  # every 12 months

    def apply(self, population):

        df = population.props

        p = self.module.parameters

        lm = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1.0,
            Predictor('hv_sexual_risk').when('sex_work', p["rr_fsw"]),
            Predictor('mc_is_circumcised').when(True, p["rr_circumcision"]),
            Predictor('li_urban').when(False, p["rr_rural"]),
            Predictor('sex').when('F', p["rr_sex_f"]),
            Predictor('hv_behaviour_change').when(True, p["rr_behaviour_change"]),
            Predictor('li_ed_lev')  .when("2", p["rr_edlevel_primary"])
                                    .when("3", p["rr_edlevel_secondary"]),
            Predictor('li_wealth')  .when("2", p["rr_windex_poorer"])
                                    .when("3", p["rr_windex_middle"])
                                    .when("4", p["rr_windex_richer"])
                                    .when("5", p["rr_windex_richest"]),
            Predictor('age_years')  .when('.between(20, 24)', p["rr_age_gp20"])
                                    .when('.between(25, 29)', p["rr_age_gp25"])
                                    .when('.between(30, 34)', p["rr_age_gp30"])
                                    .when('.between(35, 39)', p["rr_age_gp35"])
                                    .when('.between(40, 44)', p["rr_age_gp40"])
                                    .when('.between(45, 50)', p["rr_age_gp45"])
                                    .when('> 50', p["rr_age_gp50"])
        )

        lm.predict(df.loc[df['is_alive'] & ~df['hv_inf']])

        # ----------------------------------- NEW INFECTIONS -----------------------------------
        # Relative risk of acquiring HIV
        rr_aq_hiv = self.module.LinearModels['relative_risk_of_acquiring_HIV_among_those_not_infected'].predict(
            df.loc[df['is_alive'] & ~df['hv_inf']]
        )

        # Relative risk of transmitting HIV
        rr_tr_hiv = self.module.LinearModels['risk_of_transmitting_HIV_by_those_infected'].predict(
            df.loc[df['is_alive'] & df['hv_inf']]
        )

        # Calc the risk of each uninfected person becoming infected:
        # TODO when/if apply: params["hv_behav_mod"]
        prob_infection = rr_aq_hiv * rr_tr_hiv.sum()
        newly_infected = (self.module.rng.rand(len(prob_infection)) < prob_infection).index

        # Schedule the HIV Infection Event for some time in the next year:
        for person_id in newly_infected:
            self.sim.schedule_event(HivInfectionEvent(self.module, person_id),
                                    self.sim.date + DateOffset(days=rng.randint(0,365))
                                    )


class HivInfectionEvent(Event, IndividualScopeEventMixin):
    """
    This is the event for someone who is becoming infected. It will:
    * Update the 'hv_inf' and 'hiv_inf_date' properties
    * Schedule the events of Pre-AIDS, AIDS and AIDSDeath that will run if the person has not started treatment
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props

        if not df.at[person_id, 'is_alive']:
            return

        assert not df.at[person_id, 'hv_inf'], 'The person is already HIV-positive'

        # Update flags
        df.at[person_id, "hv_inf"] = True
        df.at[person_id, "hv_inf_date"] = self.sim.date

        # Set the natural history events of Pre-AIDS, AIDS and AIDS Death
        date_of_infection = self.sim.date
        age_years_at_infection = (date_of_infection - df.at[person_id, 'date_of_birth']).dt.years   #TODO check that this works

        self.module.set_natural_history(
            person_id=person_id,
            age_years_at_infection=age_years_at_infection,
            date_of_infection=date_of_infection
        )


class HivPreAidsEvent(Event, IndividualScopeEventMixin):
    """     Enacts the onset of pre-AIDS for the person if they are not on treatment.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):

        df = self.sim.population.props

        if not df.at[person_id, "is_alive"]:
            # TODO: Also add a block if the person is on treatment
            return

        self.sim.modules["SymptomManager"].change_symptom(
            person_id=person_id,
            symptom_string="pre_aids",
            add_or_remove="+",
            disease_module=self.module,
            duration_in_days=None,
        )

        # # prob = self.sim.modules['HealthSystem'].get_prob_seek_care(person_id, symptom_code=2)
        # prob = 1.0  # Do not use get_prob_seek_care()
        # seeks_care = self.module.rng.random_sample() < prob
        #
        # if seeks_care:
        #     logger.debug(
        #         "HivPreAidsEvent: scheduling Hiv_PresentsForCareWithSymptoms for person %d",
        #         person_id,
        #     )
        #     event = HSI_Hiv_PresentsForCareWithSymptoms(
        #         self.module, person_id=person_id
        #     )
        #     self.sim.modules["HealthSystem"].schedule_hsi_event(
        #         event,
        #         priority=2,
        #         topen=self.sim.date,
        #         tclose=self.sim.date + DateOffset(weeks=2),
        #     )


class HivAidsEvent(Event, IndividualScopeEventMixin):
    """     Enacts the onset of AIDS for the person if they are not on treatment.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):

        df = self.sim.population.props

        if not df.at[person_id, "is_alive"]:
            # TODO: Also add a block if the person is on treatment
            return

        df.at[person_id, "hv_specific_symptoms"] = "aids"

        self.sim.modules["SymptomManager"].change_symptom(
            person_id=person_id,
            symptom_string="aids",
            add_or_remove="+",
            disease_module=self.sim.modules["Hiv"],
            duration_in_days=None,
        )

        # # prob = self.sim.modules['HealthSystem'].get_prob_seek_care(person_id, symptom_code=3)
        # prob = 0.5  # NB. Do not use get_prob_seek_care(). For non-generic symptoms do inside the module.
        # seeks_care = self.module.rng.random_sample() < prob
        #
        # if seeks_care:
        #     logger.debug(
        #         "This is HivAidsEvent, scheduling Hiv_PresentsForCareWithSymptoms for person %d",
        #         person_id,
        #     )
        #     event = HSI_Hiv_PresentsForCareWithSymptoms(
        #         self.sim.modules["Hiv"], person_id=person_id
        #     )
        #     self.sim.modules["HealthSystem"].schedule_hsi_event(
        #         event,
        #         priority=2,
        #         topen=self.sim.date,
        #         tclose=self.sim.date + DateOffset(weeks=2),
        #     )


class HivAidsDeathEvent(Event, IndividualScopeEventMixin):
    """
    Enacts the death of the person if they are not on treatment.
    """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, person_id):
        df = self.sim.population.props

        if not df.at[person_id, "is_alive"]:
            # TODO: Also add a block if the person is on treatment
            return

        self.sim.schedule_event(
            demography.InstantaneousDeath(self.module, person_id, cause="AIDS"),
            self.sim.date,
        )


class HivMtctEvent(RegularEvent, PopulationScopeEventMixin):
    """ hiv infection event in infants during breastfeeding
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        df = population.props
        params = self.module.parameters
        now = self.sim.date

        # mother NOT ON ART & child NOT ON ART
        i1 = df.index[
            (
                self.module.rng.random_sample(size=len(df))
                < params["monthly_prob_mtct_bf_untreated"]
            )
            & df.is_alive
            & ~df.hv_inf
            & (df.hv_on_art != 2)
            & (df.age_exact_years <= 1.5)
            & df.hv_mother_inf_by_birth
            & (df.hv_mother_art != 2)
        ]

        # mother ON ART & assume child on azt/nvp
        i2 = df.index[
            (
                self.module.rng.random_sample(size=len(df))
                < params["monthly_prob_mtct_bf_untreated"]
            )
            & df.is_alive
            & ~df.hv_inf
            & (df.hv_on_art != 2)
            & (df.age_exact_years <= 1.5)
            & df.hv_mother_inf_by_birth
            & (df.hv_mother_art == 2)
        ]

        new_inf = i1.append(i2)

        df.loc[new_inf, "hv_inf"] = True
        df.loc[new_inf, "hv_date_inf"] = now
        df.loc[new_inf, "hv_fast_progressor"] = False

        # ----------------------------------- TIME OF DEATH -----------------------------------
        # assign slow progressor
        if len(new_inf):
            time_death_slow = (
                self.module.rng.weibull(
                    a=params["weibull_shape_mort_infant_slow_progressor"],
                    size=len(new_inf),
                )
                * params["weibull_scale_mort_infant_slow_progressor"]
            )

            time_death_slow = pd.to_timedelta(time_death_slow[0] * 365.25, unit="d")
            df.loc[new_inf, "hv_proj_date_death"] = now + time_death_slow

            # schedule the death event
            for person in new_inf:
                death = HivAidsDeathEvent(
                    self.module, individual_id=person, cause="hiv"
                )  # make that death event
                time_death = df.loc[person, "hv_proj_date_death"]
                self.sim.schedule_event(death, time_death)  # schedule the death

            # ----------------------------------- PROGRESSION TO SYMPTOMATIC -----------------------------------
            for person_index in new_inf:
                df.at[person_index, "hv_proj_date_symp"] = df.at[
                    person_index, "hv_proj_date_death"
                ] - DateOffset(days=732.5)

                if df.at[person_index, "hv_proj_date_symp"] < self.sim.date:
                    df.at[
                        person_index, "hv_proj_date_symp"
                    ] = self.sim.date + DateOffset(days=1)

                    # schedule the symptom update event for each person
                    symp_event = HivPreAidsEvent(self.module, person_index)
                    self.sim.schedule_event(
                        symp_event, df.at[person_index, "hv_proj_date_symp"]
                    )

                # ----------------------------------- PROGRESSION TO AIDS -----------------------------------
                df.at[person_index, "hv_proj_date_aids"] = df.at[
                    person_index, "hv_proj_date_death"
                ] - DateOffset(days=365.25)

                if df.at[person_index, "hv_proj_date_aids"] < self.sim.date:
                    df.at[
                        person_index, "hv_proj_date_aids"
                    ] = self.sim.date + DateOffset(days=1)

                # schedule the symptom update event for each person
                aids_event = HivAidsEvent(self.module, person_index)
                self.sim.schedule_event(
                    aids_event, df.at[person_index, "hv_proj_date_aids"]
                )




# ---------------------------------------------------------------------------
#   Other Events
# ---------------------------------------------------------------------------

class HivScheduleTesting(RegularEvent, PopulationScopeEventMixin):
    """ additional HIV testing happening outside the symptom-driven generic HSI event
    to increase tx coverage up to reported levels
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        df = population.props
        now = self.sim.date
        p = self.module.parameters

        # TODO this fixes the current ART coverage for projections
        # otherwise ART coverage reaches 95% in 2025
        if self.sim.date.year <= 2018:

            # select people to go for testing (and subsequent tx)
            # random sample 0.4 to match clinical case tx coverage
            test = df.index[
                (self.module.rng.random_sample(size=len(df)) < p["testing_adj"])
                & df.is_alive
                & df.hv_inf
            ]

            for person_index in test:
                logger.debug(
                    f"HivScheduleTesting: scheduling HSI_Hiv_PresentsForCareWithSymptoms for person {person_index}"
                )

                event = HSI_Hiv_PresentsForCareWithSymptoms(
                    self.module, person_id=person_index
                )
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    event, priority=1, topen=now, tclose=None
                )


class HivLaunchOutreachEvent(Event, PopulationScopeEventMixin):
    """
    this is voluntary testing and counselling
    It will now submit the individual HSI events that occur when each individual is met.
    """

    def __init__(self, module):
        super().__init__(module)

    def apply(self, population):
        df = self.sim.population.props

        # TODO: here we can add testing patterns from quarterly reports
        # Find the person_ids who are going to get the outreach
        # open to any adults not currently on ART
        gets_outreach = df.index[
            (df["is_alive"]) & (df["hv_on_art"] != 0) & (df.age_years.between(15, 80))
        ]

        for person_id in gets_outreach:
            # make the outreach event
            outreach_event_for_individual = HSI_Hiv_OutreachIndividual(
                self.module, person_id=person_id
            )

            self.sim.modules["HealthSystem"].schedule_hsi_event(
                outreach_event_for_individual,
                priority=0,
                topen=self.sim.date,
                tclose=self.sim.date + DateOffset(weeks=12),
            )

        # schedule next outreach event
        outreach_event = HivLaunchOutreachEvent(self.module)
        self.sim.schedule_event(outreach_event, self.sim.date + DateOffset(months=12))


class HivLaunchBehavChangeEvent(Event, PopulationScopeEventMixin):
    """
    this is all behaviour change interventions that will reduce risk of HIV
    """

    def __init__(self, module):
        super().__init__(module)

    def apply(self, population):
        df = self.sim.population.props

        # could do this by district
        df.loc[(df.age_years >= 15), "hv_behaviour_change"] = True

        # Find the person_ids who are going to get the behaviour change intervention
        # open to any adults not currently infected
        # gets_outreach = df.index[(df['is_alive']) & ~df.hv_inf & (df.age_years.between(15, 80))]
        # for person_id in gets_outreach:
        #     # make the outreach event
        #     outreach_event_for_individual = HSI_Hiv_BehaviourChange(self.module, person_id=person_id)
        #
        #     self.sim.modules['HealthSystem'].schedule_hsi_event(outreach_event_for_individual,
        #                                                         priority=0,
        #                                                         topen=self.sim.date,
        #                                                         tclose=self.sim.date + DateOffset(weeks=12))

        # schedule next behav change launch event
        # behav_change_event = HivLaunchBehavChangeEvent(self)
        # self.sim.schedule_event(behav_change_event, self.sim.date + DateOffset(months=12))


class HivLaunchPrepEvent(Event, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module)

    def apply(self, population):
        df = self.sim.population.props
        params = self.sim.modules["Hiv"].parameters

        # Find the person_ids who are going to get prep
        # open to fsw only
        if (
            len(df[df.is_alive & (df.sex == "F") & (df.hv_sexual_risk == "sex_work")])
            > 10
        ):

            gets_prep = (
                df[df.is_alive & (df.sex == "F") & (df.hv_sexual_risk == "sex_work")]
                .sample(frac=params["fsw_prep"])
                .index
            )

            if len(gets_prep) >= 1:
                for person_id in gets_prep:
                    # make the outreach event
                    prep_event = HSI_Hiv_Prep(self.module, person_id=person_id)

                    self.sim.modules["HealthSystem"].schedule_hsi_event(
                        prep_event,
                        priority=0,
                        topen=self.sim.date,
                        tclose=self.sim.date + DateOffset(weeks=12),
                    )

        # schedule next prep launch event
        next_prep_event = HivLaunchPrepEvent(self.module)
        self.sim.schedule_event(next_prep_event, self.sim.date + DateOffset(months=12))


# ---------------------------------------------------------------------------
#   Health System Interactions
# ---------------------------------------------------------------------------


class HSI_Hiv_PresentsForCareWithSymptoms(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It is first appointment that someone has when they present to the healthcare system with the
    symptoms of hiv.
    Outcome is testing
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(
            module, Hiv
        )  # this fails if event is called by tb -> HivAidsEvent -> HSI_Hiv...

        # Get a blank footprint and then edit to define call on resources of this event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["Over5OPD"] = 1  # This requires one outpatient appt
        the_appt_footprint[
            "VCTPositive"
        ] = 1  # Voluntary Counseling and Testing Program - For HIV-Positive

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Hiv_Testing"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(
            f"HSI_Hiv_PresentsForCareWithSymptoms: giving a test for {person_id}"
        )

        df = self.sim.population.props
        params = self.module.parameters

        # Get the consumables required
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables["Intervention_Pkg"] == "HIV Testing Services",
                "Intervention_Pkg_Code",
            ]
        )[0]

        the_cons_footprint = {
            "Intervention_Package_Code": {pkg_code1: 1},
            "Item_Code": {},
        }

        is_cons_available = self.sim.modules["HealthSystem"].request_consumables(
            hsi_event=self, cons_req_as_footprint=the_cons_footprint
        )

        if is_cons_available:
            logger.debug(
                "HSI_Hiv_PresentsForCareWithSymptoms: all consumables available"
            )

            df.at[person_id, "hv_ever_tested"] = True
            df.at[person_id, "hv_date_tested"] = self.sim.date
            df.at[person_id, "hv_number_tests"] = (
                df.at[person_id, "hv_number_tests"] + 1
            )

            # if hiv+ schedule treatment and not already on treatment
            if df.at[person_id, "hv_inf"] and (df.at[person_id, "hv_on_art"] == 0):
                df.at[person_id, "hv_diagnosed"] = True

                # request treatment
                # pre-2012, only AIDS patients received ART
                if self.sim.date.year <= 2012:
                    logger.debug(
                        "HSI_Hiv_PresentsForCareWithSymptoms: scheduling treatment for person %d on date %s",
                        person_id,
                        self.sim.date,
                    )

                    if df.at[person_id, "hv_specific_symptoms"] == "aids":
                        treatment = HSI_Hiv_StartTreatment(
                            self.module, person_id=person_id
                        )

                        # Request the health system to start treatment
                        self.sim.modules["HealthSystem"].schedule_hsi_event(
                            treatment, priority=1, topen=self.sim.date, tclose=None
                        )
                    # if not eligible for art, give ipt
                    else:

                        district = df.at[person_id, "district_of_residence"]

                        if (
                            (district in params["tb_high_risk_distr"].values)
                            & (self.sim.date.year > 2012)
                            & (self.module.rng.rand() < params["hiv_art_ipt"])
                        ):
                            logger.debug(
                                "HSI_Hiv_PresentsForCareWithSymptoms: scheduling IPT for person %d on date %s",
                                person_id,
                                self.sim.date,
                            )

                            ipt_start = tb.HSI_Tb_IptHiv(
                                self.module, person_id=person_id
                            )

                            # Request the health system to have this follow-up appointment
                            self.sim.modules["HealthSystem"].schedule_hsi_event(
                                ipt_start, priority=1, topen=self.sim.date, tclose=None
                            )

                # post-2012, treat all with some probability
                else:
                    if self.module.rng.random_sample(size=1) < params["treatment_prob"]:
                        logger.debug(
                            "HSI_Hiv_PresentsForCareWithSymptoms: scheduling art for person %d on date %s",
                            person_id,
                            self.sim.date,
                        )
                        treatment = HSI_Hiv_StartTreatment(
                            self.module, person_id=person_id
                        )

                        # Request the health system to start treatment
                        self.sim.modules["HealthSystem"].schedule_hsi_event(
                            treatment, priority=1, topen=self.sim.date, tclose=None
                        )

    def did_not_run(self):
        logger.debug("HSI_Hiv_PresentsForCareWithSymptoms: did not run")

        return True


# TODO infant screening needs to be linked up to ANC/delivery
class HSI_Hiv_InfantScreening(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event - testing of infants exposed to hiv
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Hiv)

        # Get a blank footprint and then edit to define call on resources of this event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["Peds"] = 1  # This requires one infant hiv appt

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Hiv_TestingInfant"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        logger.debug(
            "HSI_Hiv_InfantScreening: a first appointment for infant %d", person_id
        )

        df = self.sim.population.props

        # Get the consumables required
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
        item_code1 = pd.unique(
            consumables.loc[
                consumables["Items"] == "Blood collecting tube, 5 ml", "Item_Code"
            ]
        )[0]
        item_code2 = pd.unique(
            consumables.loc[
                consumables["Items"] == "Gloves, exam, latex, disposable, pair",
                "Item_Code",
            ]
        )[0]
        item_code3 = pd.unique(
            consumables.loc[consumables["Items"] == "HIV EIA Elisa test", "Item_Code"]
        )[0]

        the_cons_footprint = {
            "Intervention_Package_Code": {},
            "Item_Code": {item_code1: 1, item_code2: 1, item_code3: 1}
        }

        is_cons_available = self.sim.modules["HealthSystem"].request_consumables(
            hsi_event=self, cons_req_as_footprint=the_cons_footprint
        )
        if is_cons_available:
            logger.debug("HSI_Hiv_InfantScreening: all consumables available")

            df.at[person_id, "hv_ever_tested"] = True

            # if hiv+ schedule treatment
            if df.at[person_id, "hv_inf"]:
                df.at[person_id, "hv_diagnosed"] = True

                # request treatment
                logger.debug(
                    "HSI_Hiv_InfantScreening: scheduling hiv treatment for person %d on date %s",
                    person_id,
                    self.sim.date,
                )

                treatment = HSI_Hiv_StartInfantTreatment(
                    self.module, person_id=person_id
                )

                # Request the health system to start treatment
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    treatment, priority=2, topen=self.sim.date, tclose=None
                )
            # if hiv- then give cotrim + NVP/AZT
            else:
                # request treatment
                logger.debug(
                    "HSI_Hiv_InfantScreening: scheduling hiv treatment for person %d on date %s",
                    person_id,
                    self.sim.date,
                )

                treatment = HSI_Hiv_StartInfantProphylaxis(
                    self.module, person_id=person_id
                )

                # Request the health system to start treatment
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    treatment, priority=2, topen=self.sim.date, tclose=None
                )

    def did_not_run(self):
        pass


class HSI_Hiv_PopulationWideBehaviourChange(HSI_Event, PopulationScopeEventMixin):
    """
    This is a Population-Wide Health System Interaction Event - will change the variables to do with behaviour
    """

    def __init__(self, module, target_fn=None):
        super().__init__(module)
        assert isinstance(module, Hiv)

        # If no "target_fn" is provided, then let this event pertain to everyone
        if target_fn is None:

            def target_fn(person_id):
                return True

        self.target_fn = target_fn

        # # Define the necessary information for an HSI (Population level)
        self.TREATMENT_ID = "Hiv_PopLevel_BehavChange"

    def apply(self, population, squeeze_factor):
        logger.debug(
            "HSI_Hiv_PopulationWideBehaviourChange: modifying parameter hv_behav_mod"
        )

        # Label the relevant people as having had contact with the 'behaviour change' intervention
        # NB. An alternative approach would be for, at this point, a property in the module to be changed.

        # reduce the chance of acquisition per year (due to behaviour change)
        self.module.parameters["hv_behav_mod"] = (
            self.module.parameters["hv_behav_mod"] * 0.9
        )

        # schedule the next behaviour change event
        popwide_hsi = HSI_Hiv_PopulationWideBehaviourChange(self.module, target_fn=None)

        # Request the health system to start treatment
        self.sim.modules["HealthSystem"].schedule_hsi_event(
            popwide_hsi,
            priority=1,
            topen=self.sim.date + DateOffset(months=12),
            tclose=self.sim.date + DateOffset(months=13),
        )

    def did_not_run(self):
        pass


class HSI_Hiv_OutreachIndividual(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Hiv)

        # Get a blank footprint and then edit to define call on resources of this event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        # the_appt_footprint['ConWithDCSA'] = 1  # This requires small amount of time with DCSA
        # -- CANNOT HAVE ConWithDCSA at Level 1
        the_appt_footprint[
            "VCTPositive"
        ] = 1  # Voluntary Counseling and Testing Program - For HIV-Positive

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Hiv_Testing"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(
            "HSI_Hiv_OutreachIndividual: giving a test in the first appointment for person %d",
            person_id,
        )

        df = self.sim.population.props

        # Get the consumables required
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables["Intervention_Pkg"] == "HIV Testing Services",
                "Intervention_Pkg_Code",
            ]
        )[0]

        the_cons_footprint = {
            "Intervention_Package_Code": {pkg_code1: 1},
            "Item_Code": {},
        }

        is_cons_available = self.sim.modules["HealthSystem"].request_consumables(
            hsi_event=self, cons_req_as_footprint=the_cons_footprint
        )
        if is_cons_available:
            logger.debug("HSI_Hiv_OutreachIndividual: all consumables available")

            df.at[person_id, "hv_ever_tested"] = True
            df.at[person_id, "hv_date_tested"] = self.sim.date
            df.at[person_id, "hv_number_tests"] = (
                df.at[person_id, "hv_number_tests"] + 1
            )

            # if hiv+ schedule treatment
            if df.at[person_id, "hv_inf"]:
                df.at[person_id, "hv_diagnosed"] = True

                # request treatment
                logger.debug(
                    "HSI_Hiv_OutreachIndividual: scheduling hiv treatment for person %d on date %s",
                    person_id,
                    self.sim.date,
                )

                treatment = HSI_Hiv_StartTreatment(self.module, person_id=person_id)

                # Request the health system to start treatment
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    treatment, priority=1, topen=self.sim.date, tclose=None
                )

    def did_not_run(self):
        pass


class HSI_Hiv_Prep(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Hiv)

        # Get a blank footprint and then edit to define call on resources of this event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint[
            "ConWithDCSA"
        ] = 1  # This requires small amount of time with DCSA
        the_appt_footprint[
            "VCTPositive"
        ] = 1  # Voluntary Counseling and Testing Program - For HIV-Positive

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Hiv_Prep"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug("HSI_Hiv_Prep: giving a test and PrEP for person %d", person_id)

        df = self.sim.population.props

        # check if test available first
        # Get the consumables required
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        pkg_code1 = pd.unique(
            consumables.loc[
                consumables["Intervention_Pkg"] == "HIV Testing Services",
                "Intervention_Pkg_Code",
            ]
        )[0]

        item_code1 = pd.unique(
            consumables.loc[
                consumables["Items"]
                == "Tenofovir (TDF)/Emtricitabine (FTC), tablet, 300/200 mg",
                "Item_Code",
            ]
        )[0]

        the_cons_footprint = {
            "Intervention_Package_Code": {pkg_code1: 1},
            "Item_Code": {item_code1: 1},
        }

        # query if consumables are available before logging their use (will depend on test results)
        is_cons_available = self.sim.modules["HealthSystem"].request_consumables(
            hsi_event=self, cons_req_as_footprint=the_cons_footprint, to_log=False
        )

        # if testing is available:
        if is_cons_available["Intervention_Package_Code"][pkg_code1]:
            logger.debug("HSI_Hiv_Prep: testing is available")

            df.at[person_id, "hv_ever_tested"] = True
            df.at[person_id, "hv_date_tested"] = self.sim.date
            df.at[person_id, "hv_number_tests"] = (
                df.at[person_id, "hv_number_tests"] + 1
            )

            # if hiv+ schedule treatment
            if df.at[person_id, "hv_inf"]:
                df.at[person_id, "hv_diagnosed"] = True

                # request treatment
                logger.debug(
                    "HSI_Hiv_Prep: scheduling hiv treatment for person %d on date %s",
                    person_id,
                    self.sim.date,
                )

                treatment = HSI_Hiv_StartTreatment(self.module, person_id=person_id)

                # Request the health system to start treatment
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    treatment, priority=1, topen=self.sim.date, tclose=None
                )

                # in this case, only the hiv test is used, so reset the cons_footprint
                the_cons_footprint = {
                    "Intervention_Package_Code": {pkg_code1: 1},
                    "Item_Code": {},
                }

                cons_logged = self.sim.modules["HealthSystem"].request_consumables(
                    hsi_event=self,
                    cons_req_as_footprint=the_cons_footprint,
                    to_log=True,
                )

                if cons_logged:
                    logger.debug(
                        f"HSI_Hiv_Prep: {person_id} is HIV+, requesting treatment"
                    )

            # if HIV-, check PREP available, give PREP and assume good adherence
            else:
                if is_cons_available["Item_Code"][item_code1]:
                    logger.debug("HSI_Hiv_Prep: Prep is available")
                    df.at[person_id, "hv_on_art"] = 2

                    cons_logged = self.sim.modules["HealthSystem"].request_consumables(
                        hsi_event=self,
                        cons_req_as_footprint=the_cons_footprint,
                        to_log=True,
                    )

                    if cons_logged:
                        logger.debug(f"HSI_Hiv_Prep: {person_id} is HIV-, giving PrEP")

        else:
            logger.debug("HSI_Hiv_Prep: testing is not available")

    def did_not_run(self):
        pass


class HSI_Hiv_StartInfantProphylaxis(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event - start hiv prophylaxis for infants
    cotrim 6 mths + NVP/AZT 6-12 weeks
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Hiv)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["Peds"] = 1  # This requires one outpatient appt
        the_appt_footprint["Under5OPD"] = 1  # general child outpatient appt

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Hiv_InfantProphylaxis"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(
            "HSI_Hiv_StartInfantProphylaxis: initiating treatment for person %d",
            person_id,
        )

        df = self.sim.population.props

        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
        item_code1 = pd.unique(
            consumables.loc[
                consumables["Items"] == "Zidovudine (AZT), capsule, 100 mg", "Item_Code"
            ]
        )[0]
        item_code2 = pd.unique(
            consumables.loc[
                consumables["Items"] == "Nevirapine (NVP), tablet, 200 mg", "Item_Code"
            ]
        )[0]
        item_code3 = pd.unique(
            consumables.loc[
                consumables["Items"]
                == "Cotrimoxazole preventive therapy for TB HIV+ patients",
                "Item_Code",
            ]
        )[0]

        the_cons_footprint = {
            "Intervention_Package_Code": {},
            "Item_Code": {item_code1: 1, item_code2: 1, item_code3: 1}
        }

        is_cons_available = self.sim.modules["HealthSystem"].request_consumables(
            hsi_event=self, cons_req_as_footprint=the_cons_footprint
        )
        if is_cons_available:
            df.at[person_id, "hv_on_cotrim"] = True
            df.at[person_id, "hv_on_art"] = True

            # schedule end date of cotrim after six months
            self.sim.schedule_event(
                HivCotrimEndEvent(self.module, person_id),
                self.sim.date + DateOffset(months=6),
            )

            # schedule end date of ARVs after 6-12 weeks
            self.sim.schedule_event(
                HivARVEndEvent(self.module, person_id),
                self.sim.date + DateOffset(weeks=12),
            )

    def did_not_run(self):
        pass


class HSI_Hiv_StartInfantTreatment(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event - start hiv treatment for infants + cotrim
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Hiv)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["Peds"] = 1  # This requires one out patient appt
        the_appt_footprint["Under5OPD"] = 1  # hiv-specific appt type

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Hiv_InfantTreatmentInitiation"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(
            "This is HSI_Hiv_StartInfantTreatment: initiating treatment for person %d",
            person_id,
        )

        # ----------------------------------- ASSIGN ART ADHERENCE PROPERTIES -----------------------------------

        params = self.module.parameters
        df = self.sim.population.props

        df.at[person_id, "hv_on_cotrim"] = True

        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
        item = "Lamiduvine/Zidovudine/Nevirapine (3TC + AZT + NVP), tablet, 150 + 300 + 200 mg"
        item_code1 = pd.unique(
            consumables.loc[consumables["Items"] == item, "Item_Code"]
        )[0]
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables["Intervention_Pkg"] == "Cotrimoxazole for children",
                "Intervention_Pkg_Code",
            ]
        )[0]

        the_cons_footprint = {
            "Intervention_Package_Code": {pkg_code1: 1},
            "Item_Code": {item_code1: 1}
        }

        is_cons_available = self.sim.modules["HealthSystem"].request_consumables(
            hsi_event=self, cons_req_as_footprint=the_cons_footprint
        )

        if is_cons_available:

            if (
                df.at[person_id, "is_alive"]
                and df.at[person_id, "hv_diagnosed"]
                and (df.at[person_id, "age_years"] < 15)
            ):
                df.at[person_id, "hv_on_art"] = self.module.rng.choice(
                    [1, 2], p=[(1 - params["vls_child"]), params["vls_child"]]
                )

            df.at[person_id, "hv_date_art_start"] = self.sim.date

            # change specific_symptoms to 'none' if virally suppressed and adherent (hiv_on_art = 2)
            if df.at[person_id, "hv_on_art"] == 2:
                df.at[person_id, "hv_specific_symptoms"] = "none"

            # ----------------------------------- SCHEDULE VL MONITORING -----------------------------------

            # Create follow-up appointments for VL monitoring
            times = params["vl_monitoring_times"]

            logger.debug(
                "....This is HSI_Hiv_StartTreatment: scheduling a follow-up appointment for person %d",
                person_id,
            )

            followup_appt = HSI_Hiv_VLMonitoring(self.module, person_id=person_id)

            # Request the health system to have this follow-up appointment
            for i in range(0, len(times)):
                followup_appt_date = self.sim.date + DateOffset(
                    months=times.time_months[i]
                )
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    followup_appt,
                    priority=2,
                    topen=followup_appt_date,
                    tclose=followup_appt_date + DateOffset(weeks=2),
                )

            # ----------------------------------- SCHEDULE REPEAT PRESCRIPTIONS -----------------------------------

            date_repeat_prescription = self.sim.date + DateOffset(months=3)

            logger.debug(
                "HSI_Hiv_StartTreatment: scheduling a repeat prescription for person %d on date %s",
                person_id,
                date_repeat_prescription,
            )

            followup_appt = HSI_Hiv_RepeatARV(self.module, person_id=person_id)

            # Request the health system to have this follow-up appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                followup_appt,
                priority=2,
                topen=date_repeat_prescription,
                tclose=date_repeat_prescription + DateOffset(weeks=2),
            )

            # ----------------------------------- SCHEDULE COTRIM END -----------------------------------
            # schedule end date of cotrim after six months
            self.sim.schedule_event(
                HivCotrimEndEvent(self.module, person_id),
                self.sim.date + DateOffset(months=6),
            )

            # ----------------------------------- SCHEDULE IPT START -----------------------------------
            # df.at[person_id, 'tb_inf'].startswith("active"):
            district = df.at[person_id, "district_of_residence"]

            if (
                (district in params["tb_high_risk_distr"].values)
                & (self.sim.date.year > 2012)
                & (self.module.rng.rand() < params["hiv_art_ipt"])
            ):

                if (
                    not df.at[person_id, "hv_on_art"] == 0
                    and not (df.at[person_id, "tb_inf"].startswith("active"))
                    and (self.module.rng.random_sample(size=1) < params["hiv_art_ipt"])
                ):
                    logger.debug(
                        "HSI_Hiv_StartTreatment: scheduling IPT for person %d on date %s",
                        person_id,
                        self.sim.date,
                    )

                    ipt_start = tb.HSI_Tb_IptHiv(self.module, person_id=person_id)

                    # Request the health system to have this follow-up appointment
                    self.sim.modules["HealthSystem"].schedule_hsi_event(
                        ipt_start, priority=1, topen=self.sim.date, tclose=None
                    )

    def did_not_run(self):
        pass


class HSI_Hiv_StartTreatment(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event - start hiv treatment
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Hiv)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["Over5OPD"] = 1  # This requires one out patient appt
        the_appt_footprint["NewAdult"] = 1  # hiv-specific appt type

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Hiv_TreatmentInitiation"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        logger.debug(
            "HSI_Hiv_StartTreatment: initiating treatment for person %d", person_id
        )

        # params = self.module.parameters  # why doesn't this command work post-2011?
        params = self.sim.modules["Hiv"].parameters
        df = self.sim.population.props
        now = self.sim.date

        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
        item_code1 = pd.unique(
            consumables.loc[
                consumables["Items"] == "Adult First line 1A d4T-based", "Item_Code"
            ]
        )[0]

        the_cons_footprint = {
            "Intervention_Package_Code": {},
            "Item_Code": {item_code1: 1},
        }

        is_cons_available = self.sim.modules["HealthSystem"].request_consumables(
            hsi_event=self, cons_req_as_footprint=the_cons_footprint
        )

        if is_cons_available:
            # ----------------------------------- ASSIGN ART ADHERENCE PROPERTIES -----------------------------------

            # condition: not already on art
            if (
                df.at[person_id, "is_alive"]
                and df.at[person_id, "hv_diagnosed"]
                and (df.at[person_id, "age_years"] < 15)
            ):
                df.at[person_id, "hv_on_art"] = self.sim.modules["Hiv"].rng.choice(
                    [1, 2], p=[(1 - params["vls_child"]), params["vls_child"]]
                )

            if (
                df.at[person_id, "is_alive"]
                and df.at[person_id, "hv_diagnosed"]
                and (df.at[person_id, "age_years"] >= 15)
                and (df.at[person_id, "sex"] == "M")
            ):
                df.at[person_id, "hv_on_art"] = self.sim.modules["Hiv"].rng.choice(
                    [1, 2], p=[(1 - params["vls_m"]), params["vls_m"]]
                )

            if (
                df.at[person_id, "is_alive"]
                and df.at[person_id, "hv_diagnosed"]
                and (df.at[person_id, "age_years"] >= 15)
                and (df.at[person_id, "sex"] == "F")
            ):
                df.at[person_id, "hv_on_art"] = self.sim.modules["Hiv"].rng.choice(
                    [1, 2], p=[(1 - params["vls_f"]), params["vls_f"]]
                )

            df.at[person_id, "hv_date_art_start"] = now

            # change specific_symptoms to 'none' if virally suppressed and adherent (hiv_on_art = 2)
            if df.at[person_id, "hv_on_art"] == 2:
                df.at[person_id, "hv_specific_symptoms"] = "none"

            # ----------------------------------- SCHEDULE VL MONITORING -----------------------------------

            if not df.at[person_id, "hv_on_art"] == 0:

                # Create follow-up appointments for VL monitoring
                times = params["vl_monitoring_times"]

                logger.debug(
                    "HSI_Hiv_StartTreatment: scheduling a follow-up appointment for person %d",
                    person_id,
                )

                followup_appt = HSI_Hiv_VLMonitoring(self.module, person_id=person_id)

                # Request the health system to have this follow-up appointment
                for i in range(0, len(times)):
                    followup_appt_date = self.sim.date + DateOffset(
                        months=times.time_months[i]
                    )
                    self.sim.modules["HealthSystem"].schedule_hsi_event(
                        followup_appt,
                        priority=2,
                        topen=followup_appt_date,
                        tclose=followup_appt_date + DateOffset(weeks=2),
                    )

            # ----------------------------------- SCHEDULE REPEAT PRESCRIPTIONS -----------------------------------

            if not df.at[person_id, "hv_on_art"] == 0:
                date_repeat_prescription = now + DateOffset(months=3)

                logger.debug(
                    "HSI_Hiv_StartTreatment: scheduling a repeat prescription for person %d on date %s",
                    person_id,
                    date_repeat_prescription,
                )

                followup_appt = HSI_Hiv_RepeatARV(self.module, person_id=person_id)

                # Request the health system to have this follow-up appointment
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    followup_appt,
                    priority=2,
                    topen=date_repeat_prescription,
                    tclose=date_repeat_prescription + DateOffset(weeks=2),
                )

            # ----------------------------------- SCHEDULE IPT START -----------------------------------
            district = df.at[person_id, "district_of_residence"]

            if (
                (district in params["tb_high_risk_distr"].values)
                & (self.sim.date.year > 2012)
                & (self.sim.modules["Hiv"].rng.rand() < params["hiv_art_ipt"])
            ):

                if (
                    not df.at[person_id, "hv_on_art"] == 0
                    and not (df.at[person_id, "tb_inf"].startswith("active"))
                    and (
                        self.sim.modules["Hiv"].rng.random_sample(size=1)
                        < params["hiv_art_ipt"]
                    )
                ):
                    logger.debug(
                        "HSI_Hiv_StartTreatment: scheduling IPT for person %d on date %s",
                        person_id,
                        now,
                    )

                    ipt_start = tb.HSI_Tb_IptHiv(self.module, person_id=person_id)

                    # Request the health system to have this follow-up appointment
                    self.sim.modules["HealthSystem"].schedule_hsi_event(
                        ipt_start, priority=1, topen=now, tclose=None
                    )
        else:
            # if not ARVs currently available, repeat call to start treatment in 2 weeks time
            # request treatment

            logger.debug(
                "HSI_Hiv_StartTreatment: rescheduling hiv treatment for person %d on date %s",
                person_id,
                self.sim.date,
            )

            treatment = HSI_Hiv_StartTreatment(self.module, person_id=person_id)

            # Request the health system to start treatment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                treatment,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=2),
                tclose=None,
            )

    def did_not_run(self):
        pass


class HSI_Hiv_VLMonitoring(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event for hiv viral load monitoring once on treatment
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Hiv)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["LabSero"] = 1  # This requires one lab appt
        the_appt_footprint["EstNonCom"] = 1  # This is an hiv specific appt type

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Hiv_TreatmentMonitoring"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(
            "Hiv_TreatmentMonitoring: giving a viral load test to person %d", person_id
        )

        # Get the consumables required
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables["Intervention_Pkg"] == "Viral Load", "Intervention_Pkg_Code"
            ]
        )[0]

        the_cons_footprint = {
            "Intervention_Package_Code": {pkg_code1: 1},
            "Item_Code": {},
        }

        self.sim.modules["HealthSystem"].request_consumables(
            hsi_event=self, cons_req_as_footprint=the_cons_footprint, to_log=True
        )

    def did_not_run(self):
        pass


class HSI_Hiv_RepeatARV(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event for hiv repeat prescriptions once on treatment
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Hiv)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        # TODO need a pharmacy appt
        the_appt_footprint["EstNonCom"] = 1  # This is an hiv specific appt type

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Hiv_Treatment"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(
            "HSI_Hiv_RepeatPrescription: giving repeat prescription for person %d",
            person_id,
        )

        # Get the consumables required
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
        item_code1 = pd.unique(
            consumables.loc[
                consumables["Items"] == "Adult First line 1A d4T-based", "Item_Code"
            ]
        )[0]

        the_cons_footprint = {
            "Intervention_Package_Code": {},
            "Item_Code": {item_code1: 1},
        }

        request_cons = self.sim.modules["HealthSystem"].request_consumables(
            hsi_event=self, cons_req_as_footprint=the_cons_footprint, to_log=True
        )

        if request_cons:
            logger.debug(f"HSI_Hiv_RepeatPrescription: giving ARVs to {person_id}")

        date_repeat_prescription = self.sim.date + DateOffset(months=3)

        logger.debug(
            f"HSI_Hiv_RepeatPrescription: repeat prescription for{person_id} on {date_repeat_prescription}"
        )

        followup_appt = HSI_Hiv_RepeatARV(self.module, person_id=person_id)

        # Request the heathsystem to have this follow-up appointment
        self.sim.modules["HealthSystem"].schedule_hsi_event(
            followup_appt,
            priority=2,
            topen=date_repeat_prescription,
            tclose=date_repeat_prescription + DateOffset(weeks=2),
        )

    def did_not_run(self):
        pass


# ---------------------------------------------------------------------------
#   Transitions on/off treatment
# ---------------------------------------------------------------------------


class HivARVEndEvent(Event, IndividualScopeEventMixin):
    """ scheduled end of ARV provision (infant prophylaxis)
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("Stopping ARVs for person %d", person_id)

        df = self.sim.population.props

        df.at[person_id, "hv_on_art"] = False


class HivCotrimEndEvent(Event, IndividualScopeEventMixin):
    """ scheduled end of cotrimoxazole provision (infant prophylaxis)
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("Stopping cotrim for person %d", person_id)

        df = self.sim.population.props

        df.at[person_id, "hv_on_cotrim"] = False


class HivArtGoodToPoorAdherenceEvent(RegularEvent, PopulationScopeEventMixin):
    """ apply risk of transitioning from good to poor ART adherence
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))  # every 12 months

    def apply(self, population):
        df = population.props
        params = self.module.parameters

        # transition from good adherence to poor adherence
        # currently placeholder value=0 for all ages until data arrives
        if len(df[df.is_alive & (df.hv_on_art == 2)]) > 1:
            poor = (
                df[df.is_alive & (df.hv_on_art == 2)]
                .sample(frac=params["prob_high_to_low_art"])
                .index
            )

            df.loc[poor, "hv_on_art"] = 1

            # ----------------------------------- RESCHEDULE DEATH -----------------------------------
            # if now poor adherence, re-schedule death and symptom onset as if not on treatment
            if len(poor):

                for person in poor:
                    logger.debug(
                        "This is ArtGoodToPoorAdherenceEvent: transitioning to poor adherence for person %d",
                        person,
                    )

                    if df.at[person, "age_years"] < 3:
                        time_death_slow = (
                            self.module.rng.weibull(
                                a=params["weibull_shape_mort_infant_slow_progressor"],
                                size=1,
                            )
                            * params["weibull_scale_mort_infant_slow_progressor"]
                        )
                        time_death_slow = pd.to_timedelta(
                            time_death_slow[0] * 365.25, unit="d"
                        )
                        df.at[person, "hv_proj_date_death"] = (
                            self.sim.date + time_death_slow
                        )

                    else:
                        death_date = self.module.rng.weibull(
                            a=params["weibull_shape_mort_adult"], size=1
                        ) * np.exp(self.module.log_scale(df.at[person, "age_years"]))
                        death_date = pd.to_timedelta(death_date * 365.25, unit="d")

                        df.at[person, "hv_proj_date_death"] = self.sim.date + death_date

                    # schedule the death event
                    death = HivAidsDeathEvent(
                        self.module, individual_id=person, cause="hiv"
                    )  # make that death event
                    time_death = df.at[person, "hv_proj_date_death"]
                    self.sim.schedule_event(death, time_death)  # schedule the death

                    # ----------------------------------- RESCHEDULE PROGRESSION TO SYMPTOMATIC ----
                    if df.at[person, "hv_specific_symptoms"] == "none":
                        df.at[person, "hv_proj_date_symp"] = df.at[
                            person, "hv_proj_date_death"
                        ] - DateOffset(days=732.5)

                        # schedule the symptom update event for each person
                        symp_event = HivPreAidsEvent(self.module, person)

                        if df.at[person, "hv_proj_date_symp"] < self.sim.date:
                            df.at[
                                person, "hv_proj_date_symp"
                            ] = self.sim.date + DateOffset(days=1)
                        # print('symp_date', df.at[person, 'hv_proj_date_symp'])
                        self.sim.schedule_event(
                            symp_event, df.at[person, "hv_proj_date_symp"]
                        )

                    # ----------------------------------- RESCHEDULE PROGRESSION TO AIDS -----------
                    if df.at[person, "hv_specific_symptoms"] != "aids":
                        df.at[person, "hv_proj_date_aids"] = df.at[
                            person, "hv_proj_date_death"
                        ] - DateOffset(days=365.25)

                        # schedule the symptom update event for each person
                        aids_event = HivAidsEvent(self.module, person)
                        if df.at[person, "hv_proj_date_aids"] < self.sim.date:
                            df.at[
                                person, "hv_proj_date_aids"
                            ] = self.sim.date + DateOffset(days=1)
                        # print('aids_date', df.at[person, 'hv_proj_date_aids'])
                        self.sim.schedule_event(
                            aids_event, df.at[person, "hv_proj_date_aids"]
                        )


class HivArtPoorToGoodAdherenceEvent(RegularEvent, PopulationScopeEventMixin):
    """ apply risk of transitioning from poor to good ART adherence
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))  # every 12 months

    def apply(self, population):
        df = population.props
        params = self.module.parameters

        # transition from poor adherence to good adherence
        # currently placeholder value=0 for all ages until data arrives
        # this is probably going to be driven by symptoms worsening
        if len(df[df.is_alive & (df.hv_on_art == 1)]) > 1:
            good = (
                df[df.is_alive & (df.hv_on_art == 2)]
                .sample(frac=params["prob_low_to_high_art"])
                .index
            )

            df.loc[good, "hv_on_art"] = 2


class HivTransitionOffArtEvent(RegularEvent, PopulationScopeEventMixin):
    """ apply risk of stopping ART for people with hiv
    this is likely to vary by good/poor adherence along with personal characteristics
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))  # every 12 months

    def apply(self, population):
        df = population.props
        params = self.module.parameters

        if len(df[df.is_alive & (df.hv_on_art != 0)]) > 1:
            off_art = (
                df[df.is_alive & (df.hv_on_art == 2)]
                .sample(frac=params["prob_off_art"])
                .index
            )

            df.loc[off_art, "hv_on_art"] = 0

            # ----------------------------------- RESCHEDULE DEATH -----------------------------------
            # if now poor adherence, re-schedule death and symptom onset as if not on treatment
            if len(off_art):
                for person in off_art:
                    logger.debug(
                        "This is HivTransitionOffArtEvent: transitioning off ART for person %d",
                        person,
                    )

                    if df.at[person, "age_years"] < 3:
                        time_death_slow = (
                            self.module.rng.weibull(
                                a=params["weibull_shape_mort_infant_slow_progressor"],
                                size=1,
                            )
                            * params["weibull_scale_mort_infant_slow_progressor"]
                        )
                        time_death_slow = pd.to_timedelta(
                            time_death_slow[0] * 365.25, unit="d"
                        )
                        df.at[person, "hv_proj_date_death"] = (
                            self.sim.date + time_death_slow
                        )

                    else:
                        death_date = self.module.rng.weibull(
                            a=params["weibull_shape_mort_adult"], size=1
                        ) * np.exp(self.module.log_scale(df.at[person, "age_years"]))
                        death_date = pd.to_timedelta(death_date * 365.25, unit="d")

                        df.at[person, "hv_proj_date_death"] = self.sim.date + death_date

                    # schedule the death event
                    death = HivAidsDeathEvent(
                        self.module, individual_id=person, cause="hiv"
                    )  # make that death event
                    time_death = df.at[person, "hv_proj_date_death"]
                    self.sim.schedule_event(death, time_death)  # schedule the death

                    # ----------------------------------- RESCHEDULE PROGRESSION TO SYMPTOMATIC ----
                    if df.at[person, "hv_specific_symptoms"] == "none":
                        df.at[person, "hv_proj_date_symp"] = df.at[
                            person, "hv_proj_date_death"
                        ] - DateOffset(days=732.5)

                        # schedule the symptom update event for each person
                        symp_event = HivPreAidsEvent(self.module, person)
                        if df.at[person, "hv_proj_date_symp"] < self.sim.date:
                            df.at[
                                person, "hv_proj_date_symp"
                            ] = self.sim.date + DateOffset(days=1)
                        # print('symp_date', df.at[person, 'hv_proj_date_symp'])
                        self.sim.schedule_event(
                            symp_event, df.at[person, "hv_proj_date_symp"]
                        )

                    # ----------------------------------- RESCHEDULE PROGRESSION TO AIDS -----------
                    if df.at[person, "hv_specific_symptoms"] != "aids":
                        df.at[person, "hv_proj_date_aids"] = df.at[
                            person, "hv_proj_date_death"
                        ] - DateOffset(days=365.25)

                        # schedule the symptom update event for each person
                        aids_event = HivAidsEvent(self.module, person)
                        if df.at[person, "hv_proj_date_aids"] < self.sim.date:
                            df.at[
                                person, "hv_proj_date_aids"
                            ] = self.sim.date + DateOffset(days=1)
                        # print('aids_date', df.at[person, 'hv_proj_date_aids'])
                        self.sim.schedule_event(
                            aids_event, df.at[person, "hv_proj_date_aids"]
                        )


# ---------------------------------------------------------------------------
#   Transitions to sex work
# ---------------------------------------------------------------------------


class FswEvent(RegularEvent, PopulationScopeEventMixin):
    """ apply risk of fsw to female pop and transition back to non-fsw
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))  # every 12 months

    def apply(self, population):
        df = population.props
        params = self.module.parameters

        # transition those already fsw back to low risk
        if (
            len(df[df.is_alive & (df.sex == "F") & (df.hv_sexual_risk == "sex_work")])
            > 1
        ):
            remove = (
                df[df.is_alive & (df.sex == "F") & (df.hv_sexual_risk == "sex_work")]
                .sample(frac=params["fsw_transition"])
                .index
            )

            df.loc[remove, "hv_sexual_risk"] = "low"

        # recruit new fsw, higher weighting for previous sex work?
        # TODO: should propensity for sex work be clustered by wealth / education / location?
        # check if any data to inform this
        # new fsw recruited to replace removed fsw -> constant proportion over time

        # current proportion of F 15-49 classified as fsw
        fsw = len(df[df.is_alive & (df.hv_sexual_risk == "sex_work")])
        eligible = len(
            df[
                df.is_alive
                & (df.sex == "F")
                & (df.age_years.between(15, 49))
                & (df.li_mar_stat != 2)
            ]
        )

        prop = fsw / eligible

        if prop < params["proportion_female_sex_workers"]:
            # number new fsw needed
            recruit = int((params["proportion_female_sex_workers"] - prop) * eligible)
            fsw_new = (
                df[
                    df.is_alive
                    & (df.sex == "F")
                    & (df.age_years.between(15, 49))
                    & (df.li_mar_stat != 2)
                ]
                .sample(n=recruit)
                .index
            )
            df.loc[fsw_new, "hv_sexual_risk"] = "sex_work"




# ---------------------------------------------------------------------------
#   Logging
# ---------------------------------------------------------------------------


class HivLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """ produce some outputs to check
        """
        # run this event every 12 months
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props
        now = self.sim.date

        def divide_but_zero_if_nan(x, y):
            if y > 0:
                return x / y
            else:
                return 0

        # ------------------------------------ INC / PREV ------------------------------------
        # New HIV infections among Adults aged 15-49
        num_new_infections_15_to_49 = len(
            df.loc[
                (df.age_years.between(15, 49))
                & df.is_alive
                & (df.hv_date_inf > (now - DateOffset(months=self.repeat)))
                & (df.hv_date_inf <= now)
            ]
        )

        pop_15_to_49 = len(df[df.is_alive & (df.age_years.between(15, 49))])
        inc_percent_15_to_49 = (num_new_infections_15_to_49 / pop_15_to_49)
        # TODO: denominator should be person-years at risk not population and not as a 'percentage'
        # See: https://github.com/UCL/TLOmodel/issues/116

        # New HIV infections among Children age 0-14 years
        num_new_infections_O_to_14 = len(
            df.loc[
                (df.age_years.between(0, 14))
                & df.is_alive
                & (df.hv_date_inf > (now - DateOffset(months=self.repeat)))
                & (df.hv_date_inf <= now)
            ]
        )
        pop_O_to_14 = len(df[df.is_alive & df.age_years.between(0, 14)])
        inc_percent_0_to_14 = (num_new_infections_O_to_14 / pop_O_to_14)

        # New HIV infections among FSW (aged 15+)
        num_new_infections_fsw = len(
            df.loc[
                (df.age_years > 15)
                & (df.hv_sexual_risk == "sex_work")
                & df.is_alive
                & (df.hv_date_inf > (now - DateOffset(months=self.repeat)))
                & (df.hv_date_inf <= now)
                ]
        )
        pop_fsw = len(df[df.is_alive & (df.age_years > 15) & (df.hv_sexual_risk == "sex_work")])
        inc_percent_fsw = (num_new_infections_fsw / pop_fsw) if not pop_fsw == 0 else 0
        assert inc_percent_fsw <= 1

        # HIV prevalence among 15-49yo
        prev_15_to_49 = (
            len(df[df.hv_inf & df.is_alive & (df.age_years.between(15, 49))])
            / len(df[df.is_alive & (df.age_years.between(15, 49))])
        )
        assert prev_15_to_49 <= 1

        # HIV prevalence among 0-14yo
        prev_0_to_14 = (
            len(df[df.hv_inf & df.is_alive & (df.age_years.between(0, 14))])
            / len(df[df.is_alive & (df.age_years.between(0, 14))])
        )
        assert prev_0_to_14 <= 1

        # HIV prevalence among FSW
        num_fsw = len(df[df.is_alive & (df.age_years>15) & (df.hv_sexual_risk == "sex_work")])
        prev_fsw = divide_but_zero_if_nan(
            len(df[df.hv_inf & df.is_alive & (df.age_years>15) & (df.hv_sexual_risk == "sex_work")]),
            num_fsw
        )
        assert prev_fsw <= 1

        # Proportion of women aged 15-49 that are classified as sexual_risk=='sex_work'
        prop_women_in_sex_work = len(df[df.is_alive & (df.sex == "F") & (df.age_years.between(15, 49)) & (df.hv_sexual_risk == "sex_work")]) / len(df[df.is_alive & (df.sex == "F") & (df.age_years.between(15, 49))])
        assert prop_women_in_sex_work <= 1

        dict_of_outputs = {
            "num_new_infections_15_to_49": num_new_infections_15_to_49,
            "num_new_infections_0_to_14": num_new_infections_O_to_14,
            "num_new_infections_fsw": num_new_infections_fsw,
            "inc_percent_15_to_49": inc_percent_15_to_49,
            "inc_percent_0_to_49": inc_percent_0_to_14,
            "inc_percent_fsw": num_new_infections_fsw,
            "prev_15_to_49": prev_15_to_49,
            "prev_0_to_49": prev_0_to_14,
            "prev_fsw": prev_fsw,
            "prop_women_in_sex_work": prop_women_in_sex_work
        }

        logger.info(
            "%s|hiv_epidemiology|%s",
            self.sim.date,
            dict_of_outputs,
        )

        # ------------------------------------ PREVALENCE BY AGE ------------------------------------
        # Output number of People Living With HIV (PLHIV) by age group and seperately for men and women
        plhiv_m = df.loc[
                df.is_alive
                & (df.sex == "M")
                & df.hv_inf
                ].groupby("age_range").size()

        plhiv_f = df.loc[
                df.is_alive
                & (df.sex == "F")
                & df.hv_inf
                ].groupby("age_range").size()

        logger.info("%s|plhiv_m|%s", self.sim.date, plhiv_m.to_dict())
        logger.info("%s|plhiv_f|%s", self.sim.date, plhiv_f.to_dict())

        # ------------------------------------ TREATMENT ------------------------------------
        # Numbers on ART by category
        on_art_15plus = df.loc[
            df.is_alive &
            (df.hv_on_art==1) &   # TODO: update this to be the bool
            (df.age_years >= 15)
        ].groupby(by=["hv_on_art"]).size()

        on_art_0_to_4 = df.loc[
            df.is_alive &
            (df.hv_on_art==1) &  # TODO: update this to be the bool
            (df['age_years'].between(0,4))
        ].groupby(by=["hv_on_art"]).size()

        logger.info(
            "%s|hiv_treatment|%s",
            self.sim.date,
            {
                "on_art_15plus": on_art_15plus.to_dict(),
                "on_art_0_to_15": on_art_0_to_4.to_dict(),
            },
        )

        # ------------------------------------ OTHER INTERVENTIONS ------------------------------------
        # Proportion exposed to behaviour change
        prop_exposed_to_behaviour_change_15plus = len(
            df[df.is_alive & df.hv_behaviour_change & (df.age_years >= 15)]
        ) / len(df[df.is_alive & (df.age_years >= 15)])

        # PREP
        prop_fsw_on_prep = divide_but_zero_if_nan(
            (df.is_alive & df.hv_on_prep & (df.hv_sexual_risk == "sex_work") & (df.age_years >= 15)).sum(),
            (df.is_alive & (df.hv_sexual_risk == "sex_work") & (df.age_years >= 15)).sum()
        )

        # Proportion of Adult PLHIV who have been diagnosed
        # TODO: maybe elaborate this so that we can track the route through which the person was diagnosed.
        prop_15plus_diagnosed = divide_but_zero_if_nan(
            (df.is_alive & (df.age_years > 15) & df.hv_inf & df.hv_diagnosed).sum(),
            (df.is_alive & (df.age_years > 15) & df.hv_inf).sum()
        )

        logger.info(
            "%s|hiv_intvs|%s",
            self.sim.date,
            {
                "prop_exposed_to_behaviour_change_15plus": prop_exposed_to_behaviour_change_15plus,
                "prop_fsw_on_prep": prop_fsw_on_prep,
                "prop_15plus_diagnosed": prop_15plus_diagnosed
            },
        )


