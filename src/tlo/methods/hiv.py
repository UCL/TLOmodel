"""
The HIV Module

Overview:
HIV infection ---> AIDS onset Event (defined by the presence of those symptoms) --> AIDS Death Event
Testing is spontaneously taken-up and can lead to accessing intervention services; ART, VMMC, PrEP

# Things to note
* Need to incorporate testing for HIV at first ANC appointment (as it does in generic HSI)
* Need to incorporate testing for infants born to HIV-positive mothers (currently done in on_birth here).
* Cotrimoxazole is not included - either in effect of consumption of the drug (because the effect is not known).

# TODO before PR
* isort and flake8
* last tests - see test script
* calibration plots

---
* Decide the relationship between AIDS and VL suppression (which blocks the AIDSOnsetEvent and AIDSDeathEvent -
currently either does)
* Assume that any ART removes the aids_symptoms? does this depend on VL status??
* What to happen with stock-outs
* note that if consumables not available for several days, could then have several appointments.
"""

import os

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, demography
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom
from tlo.util import create_age_range_lookup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Hiv(Module):
    """
    The HIV Disease Module
    """

    def __init__(self, name=None, resourcefilepath=None, run_with_checks=False):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        assert isinstance(run_with_checks, bool)
        self.run_with_checks = run_with_checks

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    PROPERTIES = {
        # --- Core Properties
        "hv_inf": Property(Types.BOOL,
                           "Is person currently infected with HIV (NB. AIDS status is determined by prescence of the "
                           "AIDS Symptom."),
        "hv_art": Property(Types.CATEGORICAL,
                           "ART status of person, whether on ART or not; and whether viral load is suppressed or not if"
                           " on ART.",
                           categories=["not", "on_VL_suppressed", "on_not_VL_suppressed"]),
        'hv_is_on_prep': Property(Types.BOOL,
                                  'Whether or not the person is currently taking and receiving a protective effect from'
                                  ' Pre-Exposure Prophylaxsis.'),
        "hv_behaviour_change": Property(Types.BOOL,
                                        "Has this person been exposed to HIV prevention counselling following a "
                                        "negative HIV test result"),
        "hv_diagnosed": Property(Types.BOOL, "knows that they are hiv+: i.e. is hiv+ and tested as hiv+"),
        "hv_number_tests": Property(Types.INT, "number of hiv tests ever taken"),

        # # --- Dates on which things have happened:
        "hv_date_inf": Property(Types.DATE, "Date infected with hiv"),

        # -- Temporary variable for breastfeeding:
        "tmp_breastfed": Property(Types.BOOL, "Is the person currently receiving breast milk from mother")
    }

    PARAMETERS = {
        # baseline characteristics
        "time_inf": Parameter(Types.DATA_FRAME, "prob of time since infection for baseline adult pop"),
        "art_coverage": Parameter(Types.DATA_FRAME, "coverage of ART at baseline"),

        "fraction_of_those_infected_that_have_aids_at_initiation": Parameter(Types.REAL,
                                                                             "Fraction of persons living with HIV at "
                                                                             "baseline that have developed AIDS"),
        "testing_coverage_male": Parameter(
            Types.REAL, "proportion of adult male population tested"),
        "testing_coverage_female": Parameter(
            Types.REAL, "proportion of adult female population tested"),

        # natural history - transmission - overall rates
        "beta": Parameter(
            Types.REAL, "transmission rate"),
        "prob_mtct_untreated": Parameter(
            Types.REAL, "probability of mother to child transmission"),
        "prob_mtct_treated": Parameter(
            Types.REAL, "probability of mother to child transmission, mother on ART"),
        "prob_mtct_incident_preg": Parameter(
            Types.REAL,
            "probability of mother to child transmission, mother infected during pregnancy"),
        "monthly_prob_mtct_bf_untreated": Parameter(
            Types.REAL,
            "probability of mother to child transmission during breastfeeding"),
        "monthly_prob_mtct_bf_treated": Parameter(
            Types.REAL,
            "probability of mother to child transmission, mother infected during breastfeeding"),

        # natural history - transmission - relative risk of HIV acquisition (non-intervention)
        "rr_fsw": Parameter(Types.REAL, "relative risk of hiv with female sex work"),
        "rr_circumcision": Parameter(
            Types.REAL, "relative risk of hiv with circumcision"
        ),
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

        # natural history - transmission - relative risk of HIV acquisition (interventions)
        "rr_behaviour_change": Parameter(
            Types.REAL, "relative risk of hiv with behaviour modification"
        ),
        "proportion_reduction_in_risk_of_hiv_aq_if_on_prep": Parameter(
            Types.REAL, "proportion reduction in risk of HIV acquisition if on PrEP. 0 for no efficacy; "
                        "1.0 for perfect efficacy."),

        # natural history - survival (adults)
        "mean_months_between_aids_and_death": Parameter(Types.REAL,
                                                        "Mean number of months (distributed exponentially) for the "
                                                        "time between AIDS and AIDS Death"),
        "infection_to_death_weibull_shape_1519": Parameter(Types.REAL,
                                                           "Shape parameters for the weibill distribution describing"
                                                           " time between infection and death for those aged 15-19"
                                                           " years"),
        "infection_to_death_weibull_shape_2024": Parameter(Types.REAL,
                                                           "Shape parameters for the weibill distribution describing"
                                                           " time between infection and death for those aged 20-24"
                                                           " years"),
        "infection_to_death_weibull_shape_2529": Parameter(Types.REAL,
                                                           "Shape parameters for the weibill distribution describing"
                                                           " time between infection and death for those aged 25-29"
                                                           " years"),
        "infection_to_death_weibull_shape_3034": Parameter(Types.REAL,
                                                           "Shape parameters for the weibill distribution describing"
                                                           " time between infection and death for those aged 30-34"
                                                           " years"),
        "infection_to_death_weibull_shape_3539": Parameter(Types.REAL,
                                                           "Shape parameters for the weibill distribution describing"
                                                           " time between infection and death for those aged 35-39"
                                                           " years"),
        "infection_to_death_weibull_shape_4044": Parameter(Types.REAL,
                                                           "Shape parameters for the weibill distribution describing"
                                                           " time between infection and death for those aged 40-44"
                                                           " years"),
        "infection_to_death_weibull_shape_4549": Parameter(Types.REAL,
                                                           "Shape parameters for the weibill distribution describing"
                                                           " time between infection and death for those aged 45-49"
                                                           " years"),
        "infection_to_death_weibull_scale_1519": Parameter(Types.REAL,
                                                           "Scale parameters for the weibill distribution describing"
                                                           " time between infection and death for those aged 15-19"
                                                           " years"),
        "infection_to_death_weibull_scale_2024": Parameter(Types.REAL,
                                                           "Scale parameters for the weibill distribution describing"
                                                           " time between infection and death for those aged 20-24"
                                                           " years"),
        "infection_to_death_weibull_scale_2529": Parameter(Types.REAL,
                                                           "Scale parameters for the weibill distribution describing"
                                                           " time between infection and death for those aged 25-29"
                                                           " years"),
        "infection_to_death_weibull_scale_3034": Parameter(Types.REAL,
                                                           "Scale parameters for the weibill distribution describing"
                                                           " time between infection and death for those aged 30-34"
                                                           " years"),
        "infection_to_death_weibull_scale_3539": Parameter(Types.REAL,
                                                           "Scale parameters for the weibill distribution describing"
                                                           " time between infection and death for those aged 35-39"
                                                           " years"),
        "infection_to_death_weibull_scale_4044": Parameter(Types.REAL,
                                                           "Scale parameters for the weibill distribution describing"
                                                           " time between infection and death for those aged 40-44"
                                                           " years"),
        "infection_to_death_weibull_scale_4549": Parameter(Types.REAL,
                                                           "Scale parameters for the weibill distribution describing"
                                                           " time between infection and death for those aged 45-49"
                                                           " years"),

        # natural history - survival (children)
        "mean_survival_for_infants_infected_prior_to_birth": Parameter(
            Types.REAL,
            "Exponential rate parameter for mortality in infants who are infected before birth"),
        "infection_to_death_infant_infection_after_birth_weibull_scale": Parameter(
            Types.REAL,
            "Weibull scale parameter for mortality in infants who are infected after birth"),
        "infection_to_death_infant_infection_after_birth_weibull_shape": Parameter(
            Types.REAL,
            "Weibull shape parameter for mortality in infants who are infected after birth"),

        # Uptake of Interventions
        "prob_spontaneous_test_12m": Parameter(
            Types.REAL, "probability that a person will seek HIV testing per 12 month period."),
        "prob_start_art_after_hiv_test": Parameter(
            Types.REAL, "probability that a person will start treatment, if HIV-positive, following testing"),
        "rr_start_art_if_aids_symptoms": Parameter(
            Types.REAL, "relative probability of a person starting treatment if they have aids_symptoms compared to if"
                        "they do not."
        ),
        "prob_behav_chg_after_hiv_test": Parameter(
            Types.REAL, "probability that a person will change risk behaviours, if HIV-negative, following testing"),
        "prob_prep_for_fsw_after_hiv_test": Parameter(
            Types.REAL, "probability that a FSW will start PrEP, if HIV-negative, following testing"),
        "prob_circ_after_hiv_test": Parameter(
            Types.REAL, "probability that a male will be circumcised, if HIV-negative, following testing"),
        "probability_of_being_retained_on_prep_every_3_months": Parameter(
            Types.REAL, "probability that someone who has initiated on prep will attend an appointment and be on prep "
                        "for the next 3 months, until the next appointment."),
        "probability_of_being_retained_on_art_every_6_months": Parameter(
            Types.REAL, "probability that someone who has initiated on treatment will attend an appointment and be on "
                        "treatment for next 6 months, until the next appointment."),
        "vls_m": Parameter(
            Types.REAL, "rates of viral load suppression males"),
        "vls_f": Parameter(
            Types.REAL, "rates of viral load suppression males"),
        "vls_child": Parameter(
            Types.REAL, "rates of viral load suppression in children 0-14 years")
    }

    def read_parameters(self, data_folder):
        """
        * 1) Reads the ResourceFiles
        * 2) Declare the Symptoms
        """

        # 1) Read the ResourceFiles

        # Short cut to parameters dict
        p = self.parameters

        workbook = pd.read_excel(
            os.path.join(self.resourcefilepath, "ResourceFile_HIV.xlsx"),
            sheet_name=None,
        )
        self.load_parameters_from_dataframe(workbook["parameters"])

        # Load data on HIV prevalence
        p["hiv_prev"] = workbook["hiv_prevalence"]

        # Load assumed time since infected at baseline (year 2010)
        p["time_inf"] = workbook["time_since_infection_at_baselin"]

        # Load assumed ART coverage at baseline (year 2010)
        p["art_coverage"] = workbook["art_coverage"]

        # DALY weights
        # get the DALY weight that this module will use from the weight database (these codes are just random!)
        if "HealthBurden" in self.sim.modules.keys():
            # Chronic infection but not AIDS (including if on ART)
            # (taken to be equal to "Symptomatic HIV without anaemia")
            self.daly_wts = dict()
            self.daly_wts['hiv_infection_but_not_aids'] = self.sim.modules["HealthBurden"].get_daly_weight(17)

            #  AIDS without anti-retroviral treatment without anemia
            self.daly_wts['aids'] = self.sim.modules["HealthBurden"].get_daly_weight(19)

        # 2)  Declare the Symptoms.
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name="aids_symptoms",
                    odds_ratio_health_seeking_in_adults=3.0,  # High chance of seeking care when aids_symptoms onset
                    odds_ratio_health_seeking_in_children=3.0)  # High chance of seeking care when aids_symptoms onset
        )

    def pre_initialise_population(self):
        """
        * Establish the Linear Models
        *
        """
        p = self.parameters

        # ---- LINEAR MODELS -----
        # LinearModel for the relative risk of becoming infected during the simulation
        self.rr_of_infection = LinearModel.multiplicative(
            Predictor('age_years')  .when('<15', 0.0)
                                    .when('<20', 1.0)
                                    .when('<25', p["rr_age_gp20"])
                                    .when('<30', p["rr_age_gp25"])
                                    .when('<35', p["rr_age_gp30"])
                                    .when('<40', p["rr_age_gp35"])
                                    .when('<45', p["rr_age_gp40"])
                                    .when('<50', p["rr_age_gp45"])
                                    .when('<80', p["rr_age_gp50"])
                                    .otherwise(0.0),
            Predictor('sex').when('F', p["rr_sex_f"]),
            Predictor('li_is_sexworker').when(True, p["rr_fsw"]),
            Predictor('li_is_circ').when(True, p["rr_circumcision"]),
            Predictor('hv_is_on_prep').when(True, 1.0 - p['proportion_reduction_in_risk_of_hiv_aq_if_on_prep']),
            Predictor('li_urban').when(False, p["rr_rural"]),
            Predictor('li_wealth')  .when(2, p["rr_windex_poorer"])
                                    .when(3, p["rr_windex_middle"])
                                    .when(4, p["rr_windex_richer"])
                                    .when(5, p["rr_windex_richest"]),
            Predictor('li_ed_lev')  .when(2, p["rr_edlevel_primary"])
                                    .when(3, p["rr_edlevel_secondary"]),
            Predictor('hv_behaviour_change').when(True, p["rr_behaviour_change"])
        )

        # LinearModels to give the shape and scale for the Weibull distribution describing time from infection to death
        self.scale_parameter_for_infection_to_death = LinearModel.multiplicative(
            Predictor('age_years')  .when('<20', p["infection_to_death_weibull_scale_1519"])
                                    .when('<25', p["infection_to_death_weibull_scale_2024"])
                                    .when('<30', p["infection_to_death_weibull_scale_2529"])
                                    .when('<35', p["infection_to_death_weibull_scale_3034"])
                                    .when('<40', p["infection_to_death_weibull_scale_3539"])
                                    .when('<45', p["infection_to_death_weibull_scale_4044"])
                                    .when('<50', p["infection_to_death_weibull_scale_4549"])
                                    .otherwise(p["infection_to_death_weibull_scale_4549"])
        )

        self.shape_parameter_for_infection_to_death = LinearModel.multiplicative(
            Predictor('age_years')  .when('<20', p["infection_to_death_weibull_shape_1519"])
                                    .when('<25', p["infection_to_death_weibull_shape_2024"])
                                    .when('<30', p["infection_to_death_weibull_shape_2529"])
                                    .when('<35', p["infection_to_death_weibull_shape_3034"])
                                    .when('<40', p["infection_to_death_weibull_shape_3539"])
                                    .when('<45', p["infection_to_death_weibull_shape_4044"])
                                    .when('<50', p["infection_to_death_weibull_shape_4549"])
                                    .otherwise(p["infection_to_death_weibull_shape_4549"])
        )

        # -- Linear Models for the Uptake of Services
        # Linear model that give the probability of seeking a 'Spontaneous' Test for HIV
        # (= sum of probabilities for accessing any HIV service when not ill)

        self.lm_spontaneous_test_12m = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p["prob_spontaneous_test_12m"],
            Predictor('hv_diagnosed').when(True, 0.0).otherwise(1.0)
        )

        # Linear model if the person will start ART, following when the person has been diagnosed:
        self.lm_art = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p["prob_start_art_after_hiv_test"],
            Predictor('hv_inf').when(True, 1.0).otherwise(0.0),
            Predictor('has_aids_symptoms', external=True).when(True, p["rr_start_art_if_aids_symptoms"])
        )

        # Linear model for changing behaviour following an HIV-negative test
        self.lm_behavchg = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p["prob_behav_chg_after_hiv_test"],
            Predictor('hv_inf').when(False, 1.0).otherwise(0.0)
        )

        # Linear model for starting PrEP (if F/sex-workers), following when the person has tested HIV -ve:
        self.lm_prep = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p["prob_prep_for_fsw_after_hiv_test"],
            Predictor('hv_inf').when(False, 1.0).otherwise(0.0),
            Predictor('sex').when('F', 1.0).otherwise(0.0),
            Predictor('li_is_sexworker').when(True, 1.0).otherwise(0.0)
        )

        # Linear model for circumcision (if M) following when the person has been diagnosed:
        self.lm_circ = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p["prob_circ_after_hiv_test"],
            Predictor('hv_inf').when(False, 1.0).otherwise(0.0),
            Predictor('sex').when('M', 1.0).otherwise(0.0),
        )

    def initialise_population(self, population):
        """Set our property values for the initial population.
        """

        df = population.props

        # --- Current status
        df["hv_inf"] = False
        df["hv_art"].values[:] = "not"
        df["hv_is_on_prep"] = False
        df["hv_behaviour_change"] = False
        df["hv_diagnosed"] = False
        df["hv_number_tests"] = 0

        # --- Dates on which things have happened
        df["hv_date_inf"] = pd.NaT

        # -- Temporary --
        df["tmp_breastfed"] = False

        # Launch sub-routines for allocating the right number of people into each category
        self.initialise_baseline_prevalence(population)  # allocate baseline prevalence
        self.initialise_baseline_tested(population)  # allocate baseline art coverage
        self.initialise_baseline_art(population)  # allocate baseline art coverage

    def initialise_baseline_prevalence(self, population):
        """
        Assign baseline HIV prevalence, according to age, sex and key other variables (established in analysis of DHS
        data).
        """

        params = self.parameters
        df = population.props

        # prob of infection based on age and sex in baseline year (2010:
        prevalence_db = params["hiv_prev"]
        prev_2010 = prevalence_db.loc[prevalence_db.year == 2010, ['age_from', 'sex', 'prev_prop']]
        prev_2010 = prev_2010.rename(columns={'age_from': 'age_years'})
        prob_of_infec = df.loc[df.is_alive, ['age_years', 'sex']].merge(prev_2010, on=['age_years', 'sex'], how='left')[
            'prev_prop']

        # probability based on risk factors
        rel_prob_by_risk_factor = LinearModel.multiplicative(
            Predictor("li_is_sexworker").when(True, params["rr_fsw"]),
            Predictor("li_is_circ").when(True, params["rr_circumcision"]),
            Predictor("li_urban").when(False, params["rr_rural"]),
            Predictor("li_wealth")  .when(2, params["rr_windex_poorer"])
                                    .when(3, params["rr_windex_middle"])
                                    .when(4, params["rr_windex_richer"])
                                    .when(5, params["rr_windex_richest"]),
            Predictor("li_ed_lev")  .when(2, params["rr_edlevel_primary"])
                                    .when(3, params["rr_edlevel_secondary"])
        ).predict(df.loc[df.is_alive])

        # Rescale relative probability of infection so that its average is 1.0 within each age/sex group
        p = pd.DataFrame({
            'age_years': df['age_years'],
            'sex': df['sex'],
            'prob_of_infec': prob_of_infec,
            'rel_prob_by_risk_factor': rel_prob_by_risk_factor
        })

        p['mean_of_rel_prob_within_age_sex_group'] = p.groupby(['age_years', 'sex'])[
            'rel_prob_by_risk_factor'].transform('mean')
        p['scaled_rel_prob_by_risk_factor'] = p['rel_prob_by_risk_factor'] / p['mean_of_rel_prob_within_age_sex_group']
        p['overall_prob_of_infec'] = p['scaled_rel_prob_by_risk_factor'] * p['prob_of_infec']
        infec = self.rng.rand(len(p['overall_prob_of_infec'])) < p['overall_prob_of_infec']

        # Assign the designated person as infected in the population.props dataframe:
        df.loc[infec, 'hv_inf'] = True

        # Assign date that persons were infected by drawing from assumed distribution (for adults)
        # Clipped to prevent dates of infection before before the person was born.
        years_ago_inf = self.rng.choice(
            self.time_inf["year"],
            size=len(infec),
            replace=True,
            p=self.time_inf["scaled_prob"],
        )

        hv_date_inf = pd.Series(self.sim.date - pd.to_timedelta(years_ago_inf, unit="y"))
        df.loc[infec, "hv_date_inf"] = hv_date_inf.clip(lower=df.date_of_birth)

    def initialise_baseline_tested(self, population):
        """ assign initial hiv testing levels, only for adults
        """
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

        test_index_female = df.index[
            (random_draw < self.parameters["testing_coverage_female"])
            & df.is_alive
            & (df.sex == "F")
            & (df.age_years >= 15)
            ]

        # we don't know date tested, assume date = now
        df.loc[test_index_male | test_index_female, "hv_number_tests"] = 1

        # person assumed to be diagnosed if they have had a test and are currently HIV positive:
        df.loc[((df.hv_number_tests > 0) & df.is_alive & df.hv_inf), "hv_diagnosed"] = True
        # df.loc[((df.hv_number_tests > 0) & df.is_alive & df.hv_inf), "hv_date_diagnosed"] = now

    def initialise_baseline_art(self, population):
        """ assign initial art coverage levels
        """
        df = population.props

        # 1) Determine who is currently on ART
        worksheet = self.parameters["art_coverage"]
        art_data = worksheet.loc[
            worksheet.year == 2010, ["year", "single_age", "sex", "prop_coverage"]
        ]

        # merge all susceptible individuals with their coverage probability based on sex and age
        prob_art = df.loc[df.is_alive, ['age_years', 'sex']].merge(
            art_data,
            left_on=["age_years", "sex"],
            right_on=["single_age", "sex"],
            how="left",
        )['prop_coverage']

        prob_art = prob_art.fillna(0)

        # no testing rates for children so assign ever_tested=True if allocated treatment
        art_idx = prob_art.index[
            (self.rng.rand(len(prob_art)) < prob_art)
            & df.is_alive
            & df.hv_inf
            ]

        # 2) Determine adherence levels for those currently on ART, for each of adult men, adult women and children
        adult_f_art_idx = df.loc[(df.index.isin(art_idx) & (df.sex == 'F') & (df.age_years >= 15))].index
        adult_m_art_idx = df.loc[(df.index.isin(art_idx) & (df.sex == 'M') & (df.age_years >= 15))].index
        child_art_idx = df.loc[(df.index.isin(art_idx) & (df.age_years < 15))].index

        suppr = list()  # list of all indices for persons on ART and suppressed
        notsuppr = list()  # list of all indices for persons on ART and not suppressed

        def split_into_vl_and_notvl(all_idx, prob):
            vl_suppr = self.rng.rand(len(all_idx)) < prob
            suppr.extend(all_idx[vl_suppr])
            notsuppr.extend(all_idx[~vl_suppr])

        split_into_vl_and_notvl(adult_f_art_idx, self.parameters['vls_f'])
        split_into_vl_and_notvl(adult_m_art_idx, self.parameters['vls_m'])
        split_into_vl_and_notvl(child_art_idx, self.parameters['vls_child'])

        # Set ART status:
        df.loc[suppr, "hv_art"] = "on_VL_suppressed"
        df.loc[notsuppr, "hv_art"] = "on_not_VL_suppressed"

        # check that everyone on ART is labelled as such
        assert not (df.loc[art_idx, "hv_art"] == "not").any()

        # assume that all persons currently on ART started on thre current date
        # df.loc[art_idx, "hv_date_art_start"] = self.sim.date

        # for logical consistency, ensure that all persons on ART have been tested and diagnosed
        df.loc[art_idx, "hv_number_tests"] = 1
        df.loc[art_idx, "hv_diagnosed"] = True
        # df.loc[art_idx, "hv_date_diagnosed"] = self.sim.date

    def initialise_simulation(self, sim):
        """
        * 1) Schedule the Main HIV Regular Polling Event
        * 2) Schedule the Logging Event
        * 3) Determine who has AIDS and impose the Symptoms 'aids_symptoms'
        * 4) Schedule the AIDS onset events and AIDS death event for those infected already
        * 5) (Optionally) Schedule the event to check the configuration of all properties
        * 6) Define the DxTests
        * 7) Look-up and save the codes for consumables
        """
        df = sim.population.props

        # 1) Schedule the Main HIV Regular Polling Event
        sim.schedule_event(HivRegularPollingEvent(self), sim.date)

        # 2) Schedule the Logging Event
        sim.schedule_event(HivLoggingEvent(self), sim.date)

        # 3) Determine who has AIDS and impose the Symptoms 'aids_symptoms'

        # Those on ART currently (will not get any further events scheduled):
        on_art_idx = df.loc[
            df.is_alive &
            df.hv_inf &
            (df.hv_art != "not")
            ].index

        # Those that lived more than ten years and not currently on ART are assumed to currently have AIDS
        #  (will have AIDS Death event scheduled)
        has_aids_idx = df.loc[
            df.is_alive &
            df.hv_inf &
            ((self.sim.date - df.hv_date_inf).dt.days > 10 * 365) &
            (df.hv_art == "not")
            ].index

        # Those that are in neither category are "before AIDS" (will have AIDS Onset Event scheduled)
        before_aids_idx = set(df.loc[df.is_alive & df.hv_inf].index) - set(has_aids_idx) - set(on_art_idx)

        # Impose the symptom to those that have AIDS (the symptom is the definition of having AIDS)
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=has_aids_idx.tolist(),
            symptom_string='aids_symptoms',
            add_or_remove='+',
            disease_module=self
        )

        # 4) Schedule the AIDS onset events and AIDS death event for those infected already
        # AIDS Onset Event for those who are infected but not yet AIDS and have not ever started ART
        # NB. This means that those on ART at the start of the simulation may not have an AIDS event --
        # like it happened at some point in the past

        for person_id in before_aids_idx:
            # get days until develops aids, repeating sampling until a positive number is obtained.
            days_until_aids = 0
            while days_until_aids <= 0:
                days_since_infection = (self.sim.date - df.at[person_id, 'hv_date_inf']).days
                days_infection_to_aids = np.round((self.get_time_from_infection_to_aids(person_id)).months * 30.5)
                days_until_aids = days_infection_to_aids - days_since_infection

            date_onset_aids = self.sim.date + pd.DateOffset(days=days_until_aids)
            sim.schedule_event(
                HivAidsOnsetEvent(person_id=person_id, module=self),
                date=date_onset_aids
            )

        # Schedule the AIDS death events for those who have got AIDS already
        for person_id in has_aids_idx:
            date_aids_death = self.sim.date + self.get_time_from_aids_to_death()  # (assumes AIDS onset on this day)
            sim.schedule_event(
                HivAidsDeathEvent(person_id=person_id, module=self),
                date=date_aids_death
            )

        # 5) (Optionally) Schedule the event to check the configuration of all properties
        if self.run_with_checks:
            sim.schedule_event(HivCheckPropertiesEvent(self), sim.date + pd.DateOffset(months=1))

        # 6) Define the DxTests
        # HIV Rapid Diagnostic Test:

        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
        pkg_code_hiv_rapid_test = consumables.loc[
            consumables["Intervention_Pkg"] == "HIV Testing Services",
            "Intervention_Pkg_Code"].values[0]

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            hiv_rapid_test=DxTest(
                property='hv_inf',
                sensitivity=1.0,
                specificity=1.0,
                cons_req_as_footprint={'Intervention_Package_Code': {pkg_code_hiv_rapid_test: 1}, 'Item_Code': {}}
            )
        )

        # Test for Early Infect Diagnosis
        #  - Consumables required:
        item1 = pd.unique(
            consumables.loc[
                consumables["Items"] == "Blood collecting tube, 5 ml", "Item_Code"
            ]
        )[0]
        item2 = pd.unique(
            consumables.loc[
                consumables["Items"] == "Gloves, exam, latex, disposable, pair",
                "Item_Code",
            ]
        )[0]
        item3 = pd.unique(
            consumables.loc[consumables["Items"] == "HIV EIA Elisa test", "Item_Code"]
        )[0]

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            hiv_early_infant_test=DxTest(
                property='hv_inf',
                sensitivity=1.0,
                specificity=1.0,
                cons_req_as_footprint={'Intervention_Package_Code': {}, 'Item_Code': {item1: 1, item2: 1, item3: 1}}
            )
        )

        # 7) Look-up and store the codes for the consumables used in the interventions.
        self.pkg_codes_for_consumables_required = dict()
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        # Circumcison:
        self.pkg_codes_for_consumables_required['circ'] = pd.unique(
            consumables.loc[
                consumables["Intervention_Pkg"] == "Male circumcision ",
                "Intervention_Pkg_Code",
            ]
        )[0]

        # PrEP:
        self.item_code_for_prep = pd.unique(
            consumables.loc[
                consumables["Items"]
                == "Tenofovir (TDF)/Emtricitabine (FTC), tablet, 300/200 mg",
                "Item_Code",
            ]
        )[0]

        # In relation to ART:
        # ART for adults
        self.item_code_for_art = pd.unique(
            consumables.loc[
                consumables["Items"] == "Adult First line 1A d4T-based", "Item_Code"
            ]
        )[0]

        # ART for children:
        self.cons_footprint_for_infant_art = {
            "Intervention_Package_Code": {
                pd.unique(consumables.loc[
                              consumables["Intervention_Pkg"] == "Cotrimoxazole for children",
                              "Intervention_Pkg_Code"])[0]: 1},
            "Item_Code": {
                pd.unique(consumables.loc[
                              consumables[
                                  "Items"] == "Lamiduvine/Zidovudine/Nevirapine (3TC + AZT + NVP), tablet, 150 + 300"
                                              " + 200 mg", "Item_Code"])[
                    0]: 1}
        }

        # Viral Load monitoring
        self.item_code_for_viral_load = pd.unique(
            consumables.loc[
                consumables["Intervention_Pkg"] == "Viral Load", "Intervention_Pkg_Code"
            ]
        )[0]

    def on_birth(self, mother_id, child_id):
        """
        * Initialise our properties for a newborn individual;
        * schedule testing;
        * schedule infection during breastfeeding
        """
        params = self.parameters
        df = self.sim.population.props

        # Default Settings:
        # --- Current status
        df.at[child_id, "hv_inf"] = False
        df.at[child_id, "hv_art"] = "not"
        df.at[child_id, "hv_is_on_prep"] = False
        df.at[child_id, "hv_behaviour_change"] = False
        df.at[child_id, "hv_diagnosed"] = False
        df.at[child_id, "hv_number_tests"] = 0

        # --- Dates on which things have happened
        df.at[child_id, "hv_date_inf"] = pd.NaT

        # -- Temporary
        df.at[child_id, "tmp_breastfed"] = True

        # ----- Schedule routine HIV test for those born to mothers that are HIV-positive (a time of giving birth)
        # TODO: this to be subsumed into post-natal care
        if df.at[mother_id, 'hv_inf']:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_Hiv_TestAndRefer(person_id=child_id, module=self),
                topen=self.sim.date + pd.DateOffset(months=1),
                tclose=self.sim.date + pd.DateOffset(months=2),
                priority=1
            )

        # ----------------------------------- MTCT - AT OR PRIOR TO BIRTH --------------------------
        #  DETERMINE IF THE CHILD IS INFECTED WITH HIV FROM THEIR MOTHER DURING PREGNANCY / DELIVERY
        mother_infected_prior_to_pregnancy = \
            df.at[mother_id, 'hv_inf'] & (
                df.at[mother_id, 'hv_date_inf'] <= df.at[mother_id, 'date_of_last_pregnancy']
            )
        mother_infected_during_pregnancy = \
            df.at[mother_id, 'hv_inf'] & (
                df.at[mother_id, 'hv_date_inf'] > df.at[mother_id, 'date_of_last_pregnancy']
            )

        if mother_infected_prior_to_pregnancy:
            if (df.at[mother_id, "hv_art"] == "on_VL_suppressed"):
                #  mother has existing infection, mother ON ART and VL suppressed at time of delivery
                child_infected = self.rng.rand() < params["prob_mtct_treated"]
            else:
                # mother was infected prior to prgenancy but is not on VL suppressed at time of delivery
                child_infected = self.rng.rand() < params["prob_mtct_untreated"]

        elif mother_infected_during_pregnancy:
            #  mother has incident infection during pregnancy, NO ART
            child_infected = self.rng.rand() < params["prob_mtct_incident_preg"]

        else:
            # mother is not infected
            child_infected = False

        if child_infected:
            self.do_new_infection(child_id)

        # ----------------------------------- MTCT - DURING BREASTFEEDING --------------------------
        # If child is not infected and is being breastfed, then expose them to risk of MTCT through breastfeeding
        if (~child_infected and df.at[child_id, "tmp_breastfed"] and df.at[mother_id, "hv_inf"]):
            self.mtct_during_breastfeeding(mother_id, child_id)

    def on_hsi_alert(self, person_id, treatment_id):
        raise NotImplementedError

    def report_daly_values(self):
        """Report DALYS for HIV, based on current symptomatic state of persons."""
        df = self.sim.population.props

        dalys = pd.Series(data=0, index=df.loc[df.is_alive].index)

        # All those infected get the 'infected but not AIDS' daly_wt:
        dalys.loc[df.hv_inf] = self.daly_wts['hiv_infection_but_not_aids']

        # Overwrite the value for those that currently have symptoms of AIDS with the 'AIDS' daly_wt:
        dalys.loc[self.sim.modules['SymptomManager'].who_has('aids_symptoms')] = self.daly_wts['aids']

        dalys.name = 'hiv'
        return dalys

    def mtct_during_breastfeeding(self, mother_id, child_id):
        """
        Compute risk of mother-to-child transmission and schedule HivInfectionDuringBreastFeedingEvent.

        If the child is breastfeeding currently, consider the time-until-infection assuming a constantly monthly risk of
         transmission. If the breastfeeding has ceased by the time of the scheduled infection, then it will not run.
        (This means that this event can be run at birth or at the time of the mother's infection without the need for
        further polling etc.)
        """

        df = self.sim.population.props
        params = self.parameters

        if df.at[mother_id, "hv_art"] == "on_VL_suppressed":
            monthly_prob_mtct_bf = params["monthly_prob_mtct_bf_treated"]
        else:
            monthly_prob_mtct_bf = params["monthly_prob_mtct_bf_untreated"]

        if monthly_prob_mtct_bf > 0.0:
            months_to_infection = int(self.rng.exponential(1 / monthly_prob_mtct_bf))
            date_of_infection = self.sim.date + pd.DateOffset(months=months_to_infection)
            self.sim.schedule_event(
                HivInfectionDuringBreastFeedingEvent(person_id=child_id, module=self),
                date_of_infection
            )

    def do_new_infection(self, person_id):
        """
        Enact that this person is infected with HIV
        * Update their hv_inf status and hv_date_inf
        * Schedule the AIDS onset event for this person
        """
        df = self.sim.population.props

        # Update HIV infection status for this person
        df.at[person_id, "hv_inf"] = True
        df.at[person_id, "hv_date_inf"] = self.sim.date

        # Schedule AIDS onset events for this person
        date_onset_aids = self.sim.date + self.get_time_from_infection_to_aids(person_id=person_id)
        self.sim.schedule_event(event=HivAidsOnsetEvent(self, person_id), date=date_onset_aids)

    def get_time_from_infection_to_aids(self, person_id):
        """Gives time between onset of infection and AIDS, returning a pd.DateOffset.
        For those infected prior to, or at, birth: (this is a draw from an exponential distribution)
        For those infected after birth but before reaching age 5.0 (this is drawn from a weibull distribution)
        For adults: (this is a drawn from a weibull distribution (with scale depending on age);

        * NB. It is further assumed that the time from aids to death is 18 months.
        """

        df = self.sim.population.props
        age = df.at[person_id, 'age_exact_years']
        p = self.parameters

        if age == 0.0:
            # The person is infected prior to, or at, birth:
            months_to_aids = int(
                max(0.0, self.rng.exponential(scale=p["mean_survival_for_infants_infected_prior_to_birth"]) * 12))
        elif age < 5.0:
            # The person is infected after birth but before age 5.0:
            months_to_aids = int(max(0.0,
                                     self.rng.weibull(
                                         p["infection_to_death_infant_infection_after_birth_weibull_shape"])
                                     * p["infection_to_death_infant_infection_after_birth_weibull_scale"] * 12
                                     ))
        else:
            # The person is infected after age 5.0
            # - get the shape parameters (unit: years)
            scale = self.scale_parameter_for_infection_to_death.predict(
                self.sim.population.props.loc[[person_id]]).values[0]
            # - get the scale parameter (unit: years)
            shape = self.shape_parameter_for_infection_to_death.predict(
                self.sim.population.props.loc[[person_id]]).values[0]
            # - draw from Weibull and convert to months
            months_to_death = self.rng.weibull(shape) * scale * 12
            # - compute months to aids, which is somewhat shorter than the months to death
            months_to_aids = int(
                max(0.0, np.round(months_to_death - self.parameters['mean_months_between_aids_and_death'])))

        return pd.DateOffset(months=months_to_aids)

    def get_time_from_aids_to_death(self):
        """Gives time between onset of AIDS and death, returning a pd.DateOffset.
        Assumes that the time between onset of AIDS symptoms and deaths is exponentially distributed.
        """
        mean = self.parameters['mean_months_between_aids_and_death']
        draw_number_of_months = int(np.round(self.rng.exponential(mean)))
        return pd.DateOffset(months=draw_number_of_months)

    def check_config_of_properties(self):
        """check that the properties are currently configured correctly"""
        df = self.sim.population.props

        def is_subset(col_for_set, col_for_subset):
            # Confirms that the series of col_for_subset is true only for a subset of the series for col_for_set
            return set(col_for_subset.loc[col_for_subset].index).issubset(col_for_set.loc[col_for_set].index)

        # Check that core properties of current status are never None/NaN/NaT
        assert not df.hv_inf.isna().any()
        assert not df.hv_art.isna().any()
        assert not df.hv_behaviour_change.isna().any()
        assert not df.hv_diagnosed.isna().any()
        assert not df.hv_number_tests.isna().any()

        # Check that the core HIV properties are 'nested' in the way expected.
        assert is_subset(col_for_set=df.hv_inf, col_for_subset=df.hv_diagnosed)
        assert is_subset(col_for_set=df.hv_diagnosed, col_for_subset=(df.hv_art != "not"))

        # Check that if person is not infected, the dates of HIV events are None/NaN/NaT
        assert df.loc[~df.hv_inf, "hv_date_inf"].isna().all()

        # Check that dates consistent for those infected with HIV
        assert not df.loc[df.hv_inf].hv_date_inf.isna().any()
        assert (df.loc[df.hv_inf].hv_date_inf >= df.loc[df.hv_inf].date_of_birth).all()

        # Check alignment between AIDS Symptoms and status and infection
        assert set(self.sim.modules['SymptomManager'].who_has('aids_symptoms')).issubset(
            df.loc[df.is_alive & df.hv_inf].index)


# ---------------------------------------------------------------------------
#   Main Polling Event
# ---------------------------------------------------------------------------

class HivRegularPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """ The HIV Regular Polling Events
    * Schedules persons becoming newly infected through horizontal transmission
    * Schedules who will present for voluntary ("spontaneous") testing
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))  # repeats every 12 months, but this can be changed

    def apply(self, population):

        df = population.props
        params = self.module.parameters

        fraction_of_year_between_polls = self.frequency.months / 12
        beta = params["beta"] * fraction_of_year_between_polls

        # ----------------------------------- HORIZONTAL TRANSMISSION -----------------------------------
        def horizontal_transmission(to_sex, from_sex):
            # Count current number of alive 15-80 year-olds at risk of transmission
            # (= those infected and not VL suppressed):
            n_infectious = len(df.loc[
                                   df.is_alive &
                                   df.age_years.between(15, 80) &
                                   df.hv_inf &
                                   (df.hv_art != "on_VL_suppressed") &
                                   (df.sex == from_sex)
                                   ])

            # Get Susceptible (non-infected alive 15-80 year-old) persons:
            susc_idx = df.loc[df.is_alive & ~df.hv_inf & df.age_years.between(15, 80) & (df.sex == to_sex)].index
            n_susceptible = len(susc_idx)

            # Compute chance that each susceptible person becomes infected:
            #  - relative chance of infection (acts like a scaling-factor on 'beta')
            rr_of_infection = self.module.rr_of_infection.predict(df.loc[susc_idx])

            #  - probability of infection = beta * I/N
            p_infection = rr_of_infection * beta * (n_infectious / (n_infectious + n_susceptible))

            # New infections:
            will_be_infected = self.module.rng.rand(len(p_infection)) < p_infection
            idx_new_infection = will_be_infected[will_be_infected].index

            # Schedule the date of infection for each new infection:
            for idx in idx_new_infection:
                date_of_infection = self.sim.date + \
                                    pd.DateOffset(days=self.module.rng.randint(0, 365 * fraction_of_year_between_polls))
                self.sim.schedule_event(HivInfectionEvent(self.module, idx), date_of_infection)

        # Horizontal transmission: Male --> Female
        horizontal_transmission(from_sex='M', to_sex='F')

        # Horizontal transmission: Female --> Male
        horizontal_transmission(from_sex='F', to_sex='M')

        # ----------------------------------- SPONTANEOUS TESTING -----------------------------------
        prob_spontaneous_test = self.module.lm_spontaneous_test_12m.predict(
            df.loc[df.is_alive]) * fraction_of_year_between_polls
        will_test = self.module.rng.rand(len(prob_spontaneous_test)) < prob_spontaneous_test
        idx_will_test = will_test[will_test].index

        for person_id in idx_will_test:
            date_test = self.sim.date + \
                        pd.DateOffset(days=self.module.rng.randint(0, 365 * fraction_of_year_between_polls))
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_Hiv_TestAndRefer(person_id=person_id, module=self.module),
                priority=1,
                topen=date_test,
                tclose=self.sim.date + pd.DateOffset(months=self.frequency.months)  # (to occur before next polling)
            )


# ---------------------------------------------------------------------------
#   Natural History Events
# ---------------------------------------------------------------------------

class HivInfectionEvent(Event, IndividualScopeEventMixin):
    """ This person will become infected.
    * Do the infection process
    * Check for onward transmission through MTCT if the infection is to a mother who is currently breastfeeding.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props

        # Check person is_alive
        if not df.at[person_id, 'is_alive']:
            return

        # Onset the infection for this person (which will schedule progression etc)
        self.module.do_new_infection(person_id)

        # Consider mother-to-child-transmission (MTCT) from this person to their children:
        children_of_this_person_being_breastfed = df.loc[(df.mother_id == person_id) & df.tmp_breastfed].index
        # - Do the MTCT routine for each child:
        for child_id in children_of_this_person_being_breastfed:
            self.module.mtct_during_breastfeeding(person_id, child_id)


class HivInfectionDuringBreastFeedingEvent(Event, IndividualScopeEventMixin):
    """ This person will become infected during breastfeeding
    * Do the infection process
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props

        # Check person is_alive
        if not df.at[person_id, 'is_alive']:
            return

        # Check person is breastfed currently
        if not df.at[person_id, "tmp_breastfed"]:
            return

        # Onset the infection for this person (which will schedule progression etc)
        self.module.do_new_infection(person_id)


class HivAidsOnsetEvent(Event, IndividualScopeEventMixin):
    """ This person has developed AIDS.
    * Update their symptomatic status
    * Record the date at which AIDS onset
    * Schedule the AIDS death
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):

        df = self.sim.population.props

        # Check person is_alive
        if not df.at[person_id, 'is_alive']:
            return

        # Do nothing if person is now on ART and VL suppressed
        if df.at[person_id, "hv_art"] == "on_VL_suppressed":
            return

        # Update Symptoms
        self.sim.modules["SymptomManager"].change_symptom(
            person_id=person_id,
            symptom_string="aids_symptoms",
            add_or_remove="+",
            disease_module=self.module,
        )

        # Schedule AidsDeath
        date_of_aids_death = self.sim.date + self.module.get_time_from_aids_to_death()
        self.sim.schedule_event(event=HivAidsDeathEvent(person_id=person_id, module=self.module),
                                date=date_of_aids_death)


class HivAidsDeathEvent(Event, IndividualScopeEventMixin):
    """
    Causes someone to die of AIDS
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props

        # Check person is_alive
        if not df.at[person_id, 'is_alive']:
            return

        # Do nothing if person is now on ART and VL suppressed
        if df.at[person_id, "hv_art"] == "on_VL_suppressed":
            # todo - reconsider if the VL_suppression should block the death or if this should happen through VL
            #  suppression removing the AIDS symptoms
            return

        # Confirm that the person has the symptoms of AIDS
        assert 'aids_symptoms' in self.sim.modules['SymptomManager'].has_what(person_id)

        # Cause the death - to happen immediately
        demography.InstantaneousDeath(self.module, individual_id=person_id, cause="AIDS").apply(person_id)


class Hiv_DecisionToContinueOnPrEP(Event, IndividualScopeEventMixin):
    """Helper event that is used to 'decide' if someone on PrEP should continue on PrEP.
    This event is scheduled by 'HSI_Hiv_StartOrContinueOnPrep' 3 months after it is run.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id]

        if not person["is_alive"]:
            return

        # If the person is no longer a sex worker they will not continue on PrEP
        if not person["li_is_sexworker"]:
            return

        # Check that there are on PrEP currently:
        if not person["hv_is_on_prep"]:
            logger.warning('This event should not be running')

        # Determine if this appointment is actually attended by the person who has already started on PrEP
        if (self.module.rng.rand() < self.module.parameters['probability_of_being_retained_on_prep_every_3_months']):
            # Continue on PrEP - and schedule an HSI for a refill appointment today
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Hiv_StartOrContinueOnPrep(person_id=person_id, module=self.module),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )

        else:
            # Defaults to being off PrEP - reset flag and take no further action
            df.at[person_id, "hv_is_on_prep"] = False


class Hiv_DecisionToContinueTreatment(Event, IndividualScopeEventMixin):
    """Helper event that is used to 'decide' if someone on Treatment should continue on Treatment.
    This event is scheduled by 'HSI_Hiv_StartOrContinueTreatment' 6 months after it is run.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id]

        if not person["is_alive"]:
            return

        # Check that there are on Treatment currently:
        if not (person["hv_art"] in ["on_VL_suppressed", "on_not_VL_suppressed"]):
            logger.warning('This event should not be running')

        # Determine if this appointment is actually attended by the person who has already started on PrEP
        if (self.module.rng.rand() < self.module.parameters['probability_of_being_retained_on_art_every_6_months']):
            # Continue on Treatment - and schedule an HSI for a continuation appointment today
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Hiv_StartOrContinueTreatment(person_id=person_id, module=self.module),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )

        else:
            # Defaults to being off Treatment - reset flag and take no further action
            df.at[person_id, "hv_art"] = "not"


# ---------------------------------------------------------------------------
#   Health System Interactions (HSI)
# ---------------------------------------------------------------------------

class HSI_Hiv_TestAndRefer(HSI_Event, IndividualScopeEventMixin):
    """
    The is the Test-and-Refer HSI. Individuals may seek an HIV test at any time. From this, they can be referred on to
    other services.

    This event is scheduled by:
        * the main event poll,
        * when someone presents for any care through a Generic HSI.
        * when an infant is born to an HIV-positive mother

    Following the test, they may or may not go on to present for uptake an HIV service: ART (if HIV-positive), VMMC (if
    HIV-negative and male) or PrEP (if HIV-negative and a female sex worker).

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(
            module, Hiv
        )

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Hiv_TestAndRefer"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'VCTNegative': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """Do the testing and referring to other services"""

        df = self.sim.population.props
        person = df.loc[person_id]

        if not person['is_alive']:
            return

        # If the person has previously been diagnosed do nothing do not occupy any resources
        if person['hv_diagnosed']:
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        # Run test
        if person['age_years'] < 1.0:
            test_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run='hiv_early_infant_test',
                hsi_event=self
            )
        else:
            test_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run='hiv_rapid_test',
                hsi_event=self
            )

        # Update number of tests:
        df.at[person_id, 'hv_number_tests'] += 1

        # Offer services as needed:
        if not np.isnan(test_result):

            if test_result:
                # The test_result is HIV positive
                ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint({'VCTPositive': 1})

                # Update diagnosis if the person is indeed HIV positive
                if person['hv_inf']:
                    df.at[person_id, 'hv_diagnosed'] = True

                    # Consider if the person will be referred to start ART
                    has_aids_symptoms = 'aids_symptoms' in self.sim.modules['SymptomManager'].has_what(person_id)
                    if self.module.lm_art.predict(df=df.loc[[person_id]],
                                                  rng=self.module.rng,
                                                  has_aids_symptoms=has_aids_symptoms
                                                  ):
                        self.sim.modules['HealthSystem'].schedule_hsi_event(
                            HSI_Hiv_StartOrContinueTreatment(person_id=person_id, module=self.module),
                            topen=self.sim.date,
                            tclose=None,
                            priority=0
                        )

            else:
                # The test_result is HIV negative
                ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint({'VCTNegative': 1})

                # Consider if the person's risk will be reduced by behaviour change counselling
                if self.module.lm_behavchg.predict(df.loc[[person_id]], self.module.rng):
                    df.at[person_id, 'hv_behaviour_change'] = True

                # If person is a man, and not circumcised, then consider referring to VMMC
                if (person['sex'] == 'M') & (~person['li_is_circ']):
                    if self.module.lm_circ.predict(df.loc[[person_id]], self.module.rng):
                        self.sim.modules['HealthSystem'].schedule_hsi_event(
                            HSI_Hiv_Circ(person_id=person_id, module=self.module),
                            topen=self.sim.date,
                            tclose=None,
                            priority=0
                        )

                # If person is a woman and FSW, and not currently on PrEP then consider referring to PrEP
                if (person['sex'] == 'F') & (person['li_is_sexworker']) & (~person['hv_is_on_prep']):
                    if self.module.lm_prep.predict(df.loc[[person_id]], self.module.rng):
                        self.sim.modules['HealthSystem'].schedule_hsi_event(
                            HSI_Hiv_StartOrContinueOnPrep(person_id=person_id, module=self.module),
                            topen=self.sim.date,
                            tclose=None,
                            priority=0
                        )

        return ACTUAL_APPT_FOOTPRINT


class HSI_Hiv_Circ(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "Hiv_Circumcision"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"MinorSurg": 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """Do the circumcision for this man"""

        df = self.sim.population.props  # shortcut to the dataframe

        person = df.loc[person_id]

        # Do not run if the person is not alive or is already circumcised
        if not (person["is_alive"] & ~person["li_is_circ"]):
            return

        # Check/log use of consumables, and do circumcision if materials available
        # NB. If materials not available, it is assumed that the procedure is not carried out for this person following
        # this particular referral.
        if self.get_all_consumables(pkg_codes=self.module.pkg_codes_for_consumables_required['circ']):
            # Update circumcision state
            df.at[person_id, "li_is_circ"] = True


class HSI_Hiv_StartOrContinueOnPrep(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Hiv)

        self.TREATMENT_ID = "Hiv_StartOrContinueOnPrep"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """Start PrEP for this person; or continue them on PrEP for 3 more months"""

        df = self.sim.population.props
        person = df.loc[person_id]

        # Do not run if the person is not alive or is not currently a sex worker
        if not person["is_alive"]:
            return

        # Run an HIV test
        test_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
            dx_tests_to_run='hiv_rapid_test',
            hsi_event=self
        )
        person['hv_number_tests'] += 1

        # If test is positive, flag as diagnosed and refer to ART
        if test_result is True:
            # label as diagnosed
            df.at[person_id, 'hv_diagnosed'] = True
            # Consider if the person will be referred to start ART
            if self.module.lm_art.predict(df.loc[[person_id]], self.module.rng):
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    HSI_Hiv_StartOrContinueTreatment(person_id=person_id, module=self.module),
                    topen=self.sim.date,
                    tclose=None,
                    priority=0
                )
            return self.make_appt_footprint({"Over5OPD": 1, "VCTPositive": 1})

        # Check that PrEP is available and if it is, initiate PrEP:
        if self.get_all_consumables(item_codes=self.module.item_code_for_prep):
            df.at[person_id, "hv_is_on_prep"] = True

        # Schedule 'decision about whether to continue on PrEP' for 3 months time
        self.sim.schedule_event(
            Hiv_DecisionToContinueOnPrEP(person_id=person_id, module=self.module),
            self.sim.date + pd.DateOffset(months=3)
        )

    def did_not_run(self, *args, **kwargs):
        # If this HSI cannot run, then the person will not be on PrEP and no further appointments are made:
        df = self.sim.population.props
        person_id = self.target
        df.at[person_id, "hv_is_on_prep"] = False
        return False  # to prevent this event from being rescheduled


class HSI_Hiv_StartOrContinueTreatment(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Hiv)

        self.TREATMENT_ID = "Hiv_TreatmentInitiation"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1, "NewAdult": 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """This is a Health System Interaction Event - start or continue HIV treatment for 6 more months"""
        df = self.sim.population.props
        person = df.loc[person_id]

        if not person["is_alive"]:
            return

        if person["hv_art"] == "not":
            self.do_at_initiation(person_id)
        else:
            self.do_at_continuation(person_id)

        # Schedule 'decision about whether to continue on PrEP' for 6 months time
        self.sim.schedule_event(
            Hiv_DecisionToContinueTreatment(person_id=person_id, module=self.module),
            self.sim.date + pd.DateOffset(months=6)
        )

    def get_drugs(self, age_of_person):
        """Helper function to get the ART according to the age of the person being treated"""
        if age_of_person < 5.0:
            # Formulation for children
            drugs_available = self.get_all_consumables(
                item_codes=self.module.cons_footprint_for_infant_art['Item_Code'],
                pkg_codes=self.module.cons_footprint_for_infant_art['Intervention_Package_Code']
            )
        else:
            # Formulation for adults
            drugs_available = self.get_all_consumables(item_codes=self.module.item_code_for_art)

        return drugs_available

    def do_at_initiation(self, person_id):
        """Things to do when this the first appointment ART"""
        df = self.sim.population.props
        person = df.loc[person_id]

        # Do a confirmatory test and do not run if negative
        test_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
            dx_tests_to_run='hiv_rapid_test',
            hsi_event=self
        )
        person['hv_number_tests'] += 1

        if not test_result:
            return  # todo - this return will not end the HSI

        # Check if drugs are available, and provide drugs:
        drugs_available = self.get_drugs(age_of_person=person['age_years'])

        if drugs_available:
            # Assign person to be have suppressed or un-suppressed viral load
            if person["age_years"] < 15:
                prob_vs = self.module.parameters["vls_child"]
            else:
                if person["sex"] == "M":
                    prob_vs = self.module.parameters["vls_m"]
                else:
                    prob_vs = self.module.parameters["vls_f"]

            if self.module.rng.rand() < prob_vs:
                df.at[person_id, "hv_art"] = "on_VL_suppressed"
            else:
                df.at[person_id, "hv_art"] = "on_not_VL_suppressed"

        else:
            # If drugs not available, schedule a new appointment:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Hiv_StartOrContinueTreatment(person_id=person_id, module=self.module),
                topen=self.sim.date + pd.DateOffset(days=1),
                tclose=self.sim.date + pd.DateOffset(days=14),
                priority=1
            )

    def do_at_continuation(self, person_id):
        """Things to do when the person is already on ART"""

        df = self.sim.population.props
        person = df.loc[person_id]

        # Viral Load Monitoring
        # NB. This does not have a direct effect on outcomes for the person.
        _ = self.get_all_consumables(item_codes=self.module.item_code_for_viral_load)

        # Check if drugs are available, and provide drugs:
        drugs_available = self.get_drugs(age_of_person=person['age_years'])

        if not drugs_available:
            # If drugs not available, schedule a new appointment:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Hiv_StartOrContinueTreatment(person_id=person_id, module=self.module),
                topen=self.sim.date + pd.DateOffset(days=1),
                tclose=self.sim.date + pd.DateOffset(days=14),
                priority=1
            )

    def consider_tb(self, person_id):
        # todo - TB treatment when person starts ART:
        pass
        """
        Consider whether IPT is needed at this time

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
            """


# ---------------------------------------------------------------------------
#   Logging
# ---------------------------------------------------------------------------

class HivLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """ Log Current status of the population, every year
        """

        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props
        now = self.sim.date

        # ------------------------------------ SUMMARIES ------------------------------------
        # adult prevalence
        adult_prev = (
            len(df[df.hv_inf & df.is_alive & (df.age_years >= 15)])
            / len(df[df.is_alive & (df.age_years >= 15)])
        )

        # child prevalence
        child_prev = (
            len(df[df.hv_inf & df.is_alive & (df.age_years < 15)])
            / len(df[df.is_alive & (df.age_years < 15)])
        )

        # incidence in the period since the last log for 15-49 year-olds (denominator is approximate)
        n_new_infections_adult = len(
            df.loc[
                (df.age_years >= 15)
                & df.is_alive
                & (df.hv_date_inf > (now - DateOffset(months=self.repeat)))
                ]
        )
        denom_adults = len(df[df.is_alive & ~df.hv_inf & (df.age_years >= 15)])
        adult_inc = n_new_infections_adult / denom_adults

        # incidence in the period since the last log for 0-14 year-olds (denominator is approximate)
        n_new_infections_children = len(
            df.loc[
                (df.age_years < 15)
                & df.is_alive
                & (df.hv_date_inf > (now - DateOffset(months=self.repeat)))
                ]
        )
        denom_children = len(df[df.is_alive & ~df.hv_inf & (df.age_years < 15)])
        child_inc = (n_new_infections_children / denom_children)

        # hiv prev among female sex workers (aged 15-49)
        n_fsw = len(df.loc[
                        df.is_alive &
                        df.li_is_sexworker &
                        (df.sex == "F") &
                        df.age_years.between(15, 49)
                        ])
        prev_hiv_fsw = 0 if n_fsw == 0 else \
            len(df.loc[
                    df.is_alive &
                    df.hv_inf &
                    df.li_is_sexworker &
                    (df.sex == "F") &
                    df.age_years.between(15, 49)
                    ]) / n_fsw

        logger.info(key='summary_inc_and_prev_for_adults_and_children_and_fsw',
                    description='Summary of HIV among adult (15+) and children (0-14s) and female sex workers (15-49)',
                    data={
                        "hiv_prev_adult": adult_prev,
                        "hiv_prev_child": child_prev,
                        "hiv_adult_inc": adult_inc,
                        "hiv_child_inc": child_inc,
                        "hiv_prev_fsw": prev_hiv_fsw
                    }
                    )

        # ------------------------------------ PREVALENCE BY AGE and SEX  ------------------------------------

        # Prevalence by Age/Sex (to make every category be output, do separately by 'sex')
        prev_by_age_and_sex = {}
        for sex in ['F', 'M']:
            n_hiv = df.loc[df.sex == sex].groupby(by=['age_range'])['hv_inf'].sum()
            n_pop = df.loc[df.sex == sex].groupby(by=['age_range'])['hv_inf'].count()
            prev_by_age_and_sex[sex] = (n_hiv / n_pop).to_dict()

        logger.info(key='prev_by_age_and_sex',
                    data=prev_by_age_and_sex,
                    description='Prevalence of HIV split by age and sex')

        # ------------------------------------ TREATMENT ------------------------------------
        plhiv_adult = len(df.loc[df.is_alive & df.hv_inf & (df.age_years >= 15)])
        plhiv_children = len(df.loc[df.is_alive & df.hv_inf & (df.age_years < 15)])

        # proportion of adults (15+) living with HIV that are diagnosed:
        dx_adult = len(df.loc[df.is_alive & df.hv_inf & df.hv_diagnosed & (df.age_years >= 15)]) / plhiv_adult

        # proportion of children (15+) living with HIV that are diagnosed:
        dx_children = len(df.loc[df.is_alive & df.hv_inf & df.hv_diagnosed & (df.age_years >= 15)]) / plhiv_children

        # proportions of adults (15+) living with HIV on treatment:
        art_adult = len(df.loc[df.is_alive & df.hv_inf & (df.hv_art != "not") & (df.age_years >= 15)])
        art_cov_adult = art_adult / plhiv_adult if plhiv_adult > 0 else 0

        # proportion of adults (15+) on treatment that have good VL suppression
        art_adult_vs = len(df.loc[df.is_alive & df.hv_inf & (df.hv_art == "on_VL_suppressed") & (df.age_years >= 15)])
        art_cov_vs_adult = art_adult_vs / art_adult if art_adult > 0 else 0

        # proportions of children (0-14) living with HIV on treatment:
        art_children = len(df.loc[df.is_alive & df.hv_inf & (df.hv_art != "not") & (df.age_years < 15)])
        art_cov_children = art_children / plhiv_children if plhiv_adult > 0 else 0

        # proportion of children (0-14) living with HIV on treatment and with VL suppression
        art_children_vs = len(df.loc[df.is_alive & df.hv_inf & (df.hv_art == "on_VL_suppressed") & (df.age_years < 15)])
        art_cov_vs_children = art_children_vs / art_children if art_children > 0 else 0

        # ------------------------------------ BEHAVIOUR CHANGE ------------------------------------

        # proportion of adults (15+) exposed to behaviour change intervention
        prop_adults_exposed_to_behav_intv = len(
            df[df.is_alive & df.hv_behaviour_change & (df.age_years >= 15)]
        ) / len(df[df.is_alive & (df.age_years >= 15)])

        # ------------------------------------ PREP AMONG FSW ------------------------------------
        prop_fsw_on_prep = 0 if n_fsw == 0 else len(
            df[df.is_alive & df.li_is_sexworker & (df.age_years >= 15) & df.hv_is_on_prep]
        ) / len(df[df.is_alive & df.li_is_sexworker & (df.age_years >= 15)])

        # ------------------------------------ MALE CIRCUMCISION ------------------------------------
        # NB. Among adult men
        prop_men_circ = len(
            df[df.is_alive & (df.sex == 'M') & (df.age_years >= 15) & df.li_is_circ]
        ) / len(df[df.is_alive & (df.sex == 'M') & (df.age_years >= 15)])

        logger.info(key='hiv_program_coverage',
                    description='Coverage of interventions for HIV among adult (15+) and children (0-14s)',
                    data={
                        "dx_adult": dx_adult,
                        "dx_childen": dx_children,
                        "art_coverage_adult": art_cov_adult,
                        "art_coverage_adult_VL_suppression": art_cov_vs_adult,
                        "art_coverage_child": art_cov_children,
                        "art_coverage_child_VL_suppression": art_cov_vs_children,
                        "prop_adults_exposed_to_behav_intv": prop_adults_exposed_to_behav_intv,
                        "prop_fsw_on_prep": prop_fsw_on_prep,
                        "prop_men_circ": prop_men_circ
                    }
                    )


# ---------------------------------------------------------------------------
#   Helper functions for analysing outputs
# ---------------------------------------------------------------------------

def set_age_group(ser):
    AGE_RANGE_CATEGORIES, AGE_RANGE_LOOKUP = create_age_range_lookup(
        min_age=demography.MIN_AGE_FOR_RANGE,
        max_age=demography.MAX_AGE_FOR_RANGE,
        range_size=demography.AGE_RANGE_SIZE
    )
    ser = ser.astype("category")
    AGE_RANGE_CATEGORIES_filtered = [a for a in AGE_RANGE_CATEGORIES if a in ser.values]
    return ser.cat.reorder_categories(AGE_RANGE_CATEGORIES_filtered)


def map_to_age_group(ser):
    AGE_RANGE_CATEGORIES, AGE_RANGE_LOOKUP = create_age_range_lookup(
        min_age=demography.MIN_AGE_FOR_RANGE,
        max_age=demography.MAX_AGE_FOR_RANGE,
        range_size=demography.AGE_RANGE_SIZE
    )
    ser = ser.map(AGE_RANGE_LOOKUP)
    ser = set_age_group(ser)
    return ser


def unpack_raw_output_dict(raw_dict):
    x = pd.DataFrame.from_dict(data=raw_dict, orient='index')
    x = x.reset_index()
    x.rename(columns={'index': 'age_group', 0: 'value'}, inplace=True)
    x['age_group'] = set_age_group(x['age_group'])
    return x


# ---------------------------------------------------------------------------
#   Debugging / Checking Events
# ---------------------------------------------------------------------------

class HivCheckPropertiesEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # runs every month

    def apply(self, population):
        self.module.check_config_of_properties()
