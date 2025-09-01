"""
this is the malaria module which assigns malaria infections to the population: asymptomatic, clinical and severe
it also holds the hsi events pertaining to malaria testing and treatment
including the malaria RDT using DxTest

"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Union

import pandas as pd

from tlo import Date, DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, Predictor
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.dxmanager import DxTest
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.hsi_generic_first_appts import GenericFirstAppointmentsMixin
from tlo.methods.symptommanager import Symptom
from tlo.util import random_date, read_csv_files

if TYPE_CHECKING:
    from tlo.methods.hsi_generic_first_appts import DiagnosisFunction, HSIEventScheduler
    from tlo.population import IndividualProperties

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Malaria(Module, GenericFirstAppointmentsMixin):
    def __init__(self, name=None, resourcefilepath=None):
        """Create instance of Malaria module

        :param name: Name of this module (optional, defaults to name of class)
        :param resourcefilepath: Path to the TLOmodel `resources` directory
        """
        super().__init__(name)
        self.resourcefilepath = Path(resourcefilepath)

        # cleaned coverage values for IRS and ITN (populated in `read_parameters`)
        self.itn_irs = None
        self.all_inc = None
        self.item_codes_for_consumables_required = dict()
        self.lm = dict()

    INIT_DEPENDENCIES = {
        'Contraception', 'Demography', 'HealthSystem', 'SymptomManager'
    }

    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden', 'Hiv'}

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN,
        Metadata.USES_SYMPTOMMANAGER
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'Malaria': Cause(gbd_causes='Malaria', label='Malaria'),
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'Malaria': Cause(gbd_causes='Malaria', label='Malaria')
    }

    PARAMETERS = {
        'itn_district': Parameter(
            Types.REAL, 'data frame of ITN usage rates by district'
        ),
        'irs_district': Parameter(
            Types.REAL, 'data frame of IRS usage rates by district'
        ),
        'sev_symp_prob': Parameter(
            Types.REAL, 'probabilities of each symptom for severe malaria cases'
        ),
        'sensitivity_rdt': Parameter(Types.REAL, 'Sensitivity of rdt'),
        'cfr': Parameter(Types.REAL, 'case-fatality rate for severe malaria'),
        'dur_asym': Parameter(Types.REAL, 'duration (days) of asymptomatic malaria'),
        'dur_clin': Parameter(
            Types.REAL, 'duration (days) of clinical symptoms of malaria'
        ),
        'dur_clin_para': Parameter(
            Types.REAL, 'duration (days) of parasitaemia for clinical malaria cases'
        ),
        'treatment_adjustment': Parameter(
            Types.REAL, 'probability of death from severe malaria if on treatment'
        ),
        'p_sev_anaemia_preg': Parameter(
            Types.REAL,
            'probability of severe anaemia in pregnant women with clinical malaria',
        ),
        'itn_proj': Parameter(
            Types.REAL, 'coverage of ITN for projections 2020 onwards'
        ),
        'mortality_adjust': Parameter(
            Types.REAL, 'adjustment of case-fatality rate to match WHO/MAP'
        ),
        'data_end': Parameter(
            Types.REAL, 'final year of ICL malaria model outputs, after 2018 = projections'
        ),
        'irs_rates_boundary': Parameter(
            Types.REAL, 'threshold for indoor residual spraying coverage'
        ),
        'irs_rates_upper': Parameter(
            Types.REAL, 'indoor residual spraying high coverage'
        ),
        'irs_rates_lower': Parameter(
            Types.REAL, 'indoor residual spraying low coverage'
        ),
        'prob_malaria_case_tests': Parameter(
            Types.REAL, 'probability that a malaria case will have a scheduled rdt'
        ),
        'itn': Parameter(
            Types.REAL, 'projected future itn coverage'
        ),
        'rdt_testing_rates': Parameter(
            Types.REAL,
            'per capita rdt testing rate of general population',
        ),
        'scaling_factor_for_rdt_availability': Parameter(
            Types.REAL,
            'scaling factor applied to the reports of rdt usage to compensate for'
            'non-availability of rdts at some facilities'
        ),
        'duration_iptp_protection_weeks': Parameter(
            Types.REAL,
            'duration of protection against clinical malaria conferred by each dose of IPTp'
        ),
        'rr_clinical_malaria_hiv_under5': Parameter(
            Types.REAL,
            'relative risk of clinical malaria if HIV+ and aged under 5 years'
        ),
        'rr_clinical_malaria_hiv_over5': Parameter(
            Types.REAL,
            'relative risk of clinical malaria if HIV+ and aged over 5 years'
        ),
        'rr_clinical_malaria_hiv_pregnant': Parameter(
            Types.REAL,
            'relative risk of clinical malaria if HIV+ and pregnant'
        ),
        'rr_clinical_malaria_cotrimoxazole': Parameter(
            Types.REAL,
            'relative risk of clinical malaria if on cotrimoxazole'
        ),
        'rr_clinical_malaria_art': Parameter(
            Types.REAL,
            'relative risk of clinical malaria if HIV+ and on ART and virally suppressed'
        ),
        'rr_clinical_malaria_iptp': Parameter(
            Types.REAL,
            'relative risk of clinical malaria with each dose of IPTp'
        ),
        'rr_severe_malaria_hiv_under5': Parameter(
            Types.REAL,
            'relative risk of severe malaria if HIV+ and aged under 5 years'
        ),
        'rr_severe_malaria_hiv_over5': Parameter(
            Types.REAL,
            'relative risk of severe malaria if HIV+ and aged over 5 years'
        ),
        'rr_severe_malaria_hiv_pregnant': Parameter(
            Types.REAL,
            'relative risk of clinical malaria if HIV+ and pregnant'
        ),
        'rr_severe_malaria_iptp': Parameter(
            Types.REAL,
            'relative risk of severe malaria with each dose of IPTp'
        ),
        'prob_of_treatment_success': Parameter(
            Types.REAL,
            'probability that treatment will clear malaria symptoms'
        ),
        "type_of_scaleup": Parameter(
            Types.STRING, "argument to determine type scale-up of program which will be implemented, "
                          "can be 'none', 'target' or 'max'",
        ),
        "scaleup_start_year": Parameter(
            Types.INT,
            "the year when the scale-up starts (it will occur on 1st January of that year)"
        ),
        "scaleup_parameters": Parameter(
            Types.DATA_FRAME,
            "the parameters and values changed in scenario analysis"
        )
    }

    PROPERTIES = {
        'ma_is_infected': Property(
            Types.BOOL, 'Current status of malaria, infected with malaria parasitaemia'
        ),
        'ma_date_infected': Property(Types.DATE, 'Date of latest infection'),
        'ma_date_symptoms': Property(
            Types.DATE, 'Date of symptom start for clinical infection'
        ),
        'ma_date_death': Property(Types.DATE, 'Date of death due to malaria'),
        'ma_tx': Property(Types.CATEGORICAL, 'Type of anti-malarial treatment person is currently using',
                          categories=['none', 'uncomplicated', 'complicated']),
        'ma_date_tx': Property(
            Types.DATE, 'Date treatment started for most recent malaria episode'
        ),
        'ma_inf_type': Property(
            Types.CATEGORICAL,
            'specific symptoms with malaria infection',
            categories=['none', 'asym', 'clinical', 'severe'],
        ),
        'ma_age_edited': Property(
            Types.REAL, 'age values redefined to match with malaria data'
        ),
        'ma_clinical_counter': Property(
            Types.INT, 'annual counter for malaria clinical episodes'
        ),
        'ma_dx_counter': Property(
            Types.INT, 'annual counter for malaria diagnoses'
        ),
        'ma_tx_counter': Property(
            Types.INT, 'annual counter for malaria treatment episodes'
        ),
        'ma_clinical_preg_counter': Property(
            Types.INT, 'annual counter for malaria clinical episodes in pregnant women'
        ),
        'ma_iptp': Property(Types.BOOL, 'if woman has IPTp in current pregnancy'),
    }

    def read_parameters(self, data_folder):
        # workbook = pd.read_excel(self.resourcefilepath / 'malaria' / 'ResourceFile_malaria.xlsx', sheet_name=None)
        workbook = read_csv_files(self.resourcefilepath / 'malaria' / 'ResourceFile_malaria', files=None)
        self.load_parameters_from_dataframe(workbook['parameters'])

        p = self.parameters

        # baseline characteristics
        p['itn_district'] = workbook['MAP_ITNrates']
        p['irs_district'] = workbook['MAP_IRSrates']

        p['sev_symp_prob'] = workbook['severe_symptoms']
        p['rdt_testing_rates'] = workbook['WHO_TestData2023']
        p['highrisk_districts'] = workbook['highrisk_districts']

        inf_inc_sheet = pd.read_csv(self.resourcefilepath / 'malaria' / 'ResourceFile_malaria_InfInc_expanded.csv')
        clin_inc_sheet = pd.read_csv(self.resourcefilepath / 'malaria' / 'ResourceFile_malaria_ClinInc_expanded.csv')
        sev_inc_sheet = pd.read_csv(self.resourcefilepath / 'malaria' / 'ResourceFile_malaria_SevInc_expanded.csv')

        # load parameters for scale-up projections
        p['scaleup_parameters'] = workbook["scaleup_parameters"]

        # check itn projected values are <=0.7 and rounded to 1dp for matching to incidence tables
        p['itn'] = round(p['itn'], 1)
        assert (p['itn'] <= 0.7)

        # ===============================================================================
        # single dataframe for itn and irs district/year data; set index for fast lookup
        # ===============================================================================
        itn_curr = p['itn_district']
        itn_curr.rename(columns={'itn_rates': 'itn_rate'}, inplace=True)
        itn_curr['itn_rate'] = itn_curr['itn_rate'].round(decimals=1)
        # maximum itn is 0.7; see comment https://github.com/UCL/TLOmodel/pull/165#issuecomment-699625290
        itn_curr.loc[itn_curr.itn_rate > 0.7, 'itn_rate'] = 0.7
        itn_curr = itn_curr.set_index(['District', 'Year'])

        irs_curr = p['irs_district']
        irs_curr.rename(columns={'irs_rates': 'irs_rate'}, inplace=True)
        irs_curr.drop(['Region'], axis=1, inplace=True)
        irs_curr['irs_rate'] = irs_curr['irs_rate'].round(decimals=1)
        irs_curr.loc[irs_curr.irs_rate > p['irs_rates_boundary'], 'irs_rate'] = p['irs_rates_upper']
        irs_curr.loc[irs_curr.irs_rate <= p['irs_rates_boundary'], 'irs_rate'] = p['irs_rates_lower']
        irs_curr = irs_curr.set_index(['District', 'Year'])

        itn_irs = pd.concat([itn_curr, irs_curr], axis=1)

        # Substitute District Num for District Name
        mapper_district_name_to_num = \
            {v: k for k, v in self.sim.modules['Demography'].parameters['district_num_to_district_name'].items()}
        self.itn_irs = itn_irs.reset_index().assign(
            District_Num=lambda x: x['District'].map(mapper_district_name_to_num)
        ).drop(columns=['District']).set_index(['District_Num', 'Year'])

        # ===============================================================================
        # put the all incidence data into single table with month/admin/llin/irs index
        # ===============================================================================
        inf_inc = inf_inc_sheet.set_index(['month', 'admin', 'llin', 'irs', 'age'])
        inf_inc = inf_inc.loc[:, ['monthly_prob_inf']]

        clin_inc = clin_inc_sheet.set_index(['month', 'admin', 'llin', 'irs', 'age'])
        clin_inc = clin_inc.loc[:, ['monthly_prob_clin']]

        sev_inc = sev_inc_sheet.set_index(['month', 'admin', 'llin', 'irs', 'age'])
        sev_inc = sev_inc.loc[:, ['monthly_prob_sev']]

        all_inc = pd.concat([inf_inc, clin_inc, sev_inc], axis=1)
        # we don't want age to be part of index
        all_inc = all_inc.reset_index()

        all_inc['district_num'] = all_inc['admin'].map(mapper_district_name_to_num)
        assert not all_inc['district_num'].isna().any()

        self.all_inc = all_inc.drop(columns=['admin']).set_index(['month', 'district_num', 'llin', 'irs'])

        # get the DALY weight that this module will use from the weight database
        if 'HealthBurden' in self.sim.modules:
            p['daly_wt_clinical'] = self.sim.modules['HealthBurden'].get_daly_weight(218)
            p['daly_wt_severe'] = self.sim.modules['HealthBurden'].get_daly_weight(213)

        # ----------------------------------- DECLARE THE SYMPTOMS -------------------------------------------
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom('severe_anaemia'),  # nb. will cause care seeking as much as a typical symptom
            Symptom.emergency('severe_malaria'),  # emergency
        )

    def pre_initialise_population(self):
        """Do things required before the population is created
        * Build the LinearModels"""
        self._build_linear_models()

    def _build_linear_models(self):
        """Establish the Linear Models

        if HIV is registered, the conditional predictors will apply
        otherwise only IPTp will affect risk of clinical/severe malaria
        """
        p = self.parameters

        # ---- LINEAR MODELS -----
        # LinearModel for the relative risk of clinical malaria infection
        predictors = [
            Predictor("ma_iptp").when(True, p["rr_clinical_malaria_iptp"]),
        ]

        # people with HIV
        conditional_predictors = [
            Predictor().when('(hv_inf == True) & (age_years <= 5) & (is_pregnant == False)',
                             p['rr_clinical_malaria_hiv_under5']),
            Predictor().when('(hv_inf == True) & (age_years > 5) & (is_pregnant == False)',
                             p['rr_clinical_malaria_hiv_over5']),
            Predictor().when('(hv_inf == True) & (is_pregnant == True)',
                             p['rr_clinical_malaria_hiv_pregnant']),
            # treatment effects
            # assume same effect of cotrim if pregnant
            Predictor("hv_art").when('on_VL_suppressed', p["rr_clinical_malaria_art"]).otherwise(1.0),
            Predictor("hv_on_cotrimoxazole").when(True, p["rr_clinical_malaria_cotrimoxazole"]),
        ] if "Hiv" in self.sim.modules else []

        self.lm["rr_of_clinical_malaria"] = LinearModel.multiplicative(
            *(predictors + conditional_predictors))

        # LinearModel for the relative risk of severe malaria infection
        predictors = [
            Predictor("ma_iptp").when(True, p["rr_severe_malaria_iptp"]),
        ]

        # people with HIV
        conditional_predictors = [
            Predictor().when('(hv_inf == True) & (age_years <= 5) & (is_pregnant == False)',
                             p['rr_severe_malaria_hiv_under5']),
            Predictor().when('(hv_inf == True) & (age_years > 5) & (is_pregnant == False)',
                             p['rr_severe_malaria_hiv_over5']),
            Predictor().when('(hv_inf == True) & (is_pregnant == True)',
                             p['rr_severe_malaria_hiv_pregnant']),
        ] if "Hiv" in self.sim.modules else []

        self.lm["rr_of_severe_malaria"] = LinearModel.multiplicative(
            *(predictors + conditional_predictors))

    def initialise_population(self, population):
        df = population.props

        # ----------------------------------- INITIALISE THE POPULATION-----------------------------------
        # Set default for properties
        df.loc[df.is_alive, 'ma_is_infected'] = False
        df.loc[df.is_alive, 'ma_date_infected'] = pd.NaT
        df.loc[df.is_alive, 'ma_date_symptoms'] = pd.NaT
        df.loc[df.is_alive, 'ma_date_death'] = pd.NaT
        df.loc[df.is_alive, 'ma_tx'] = 'none'
        df.loc[df.is_alive, 'ma_date_tx'] = pd.NaT
        df.loc[df.is_alive, 'ma_inf_type'] = 'none'
        df.loc[df.is_alive, 'ma_age_edited'] = 0.0

        df.loc[df.is_alive, 'ma_clinical_counter'] = 0
        df.loc[df.is_alive, 'ma_dx_counter'] = 0
        df.loc[df.is_alive, 'ma_tx_counter'] = 0
        df.loc[df.is_alive, 'ma_clinical_preg_counter'] = 0
        df.loc[df.is_alive, 'ma_iptp'] = False

    def malaria_poll2(self, population):
        df = population.props
        p = self.parameters
        now = self.sim.date
        rng = self.rng

        # ----------------------------------- DISTRICT INTERVENTION COVERAGE -----------------------------------
        # fix values for 2018 onwards
        current_year = min(now.year, p['data_end'])

        # get itn_irs rows for current year; slice multiindex for all districts & current_year
        itn_irs_curr = self.itn_irs.loc[pd.IndexSlice[:, current_year], :]
        itn_irs_curr = itn_irs_curr.reset_index().drop('Year', axis=1)  # we don't use the year column
        itn_irs_curr.insert(0, 'month', now.month)  # add current month for the incidence index lookup

        # replace itn coverage with projected coverage levels from 2019 onwards
        if now.year > p['data_end']:
            itn_irs_curr['itn_rate'] = self.parameters['itn']

        month_districtnum_itn_irs_lookup = [
            tuple(r) for r in itn_irs_curr.values]  # every row is a key in incidence table

        # ----------------------------------- DISTRICT INCIDENCE ESTIMATES -----------------------------------
        # get all corresponding rows from the incidence table; drop unneeded column; set new index
        curr_inc = self.all_inc.loc[month_districtnum_itn_irs_lookup]
        curr_inc = curr_inc.reset_index().drop(['month', 'llin', 'irs'], axis=1).set_index(['district_num', 'age'])

        # ----------------------------------- DISTRICT NEW INFECTIONS -----------------------------------
        def _draw_incidence_for(_col, _where):
            """a helper function to perform random draw for selected individuals on column of probabilities"""
            # create an index from the individuals to lookup entries in the current incidence table
            district_age_lookup = pd.MultiIndex.from_frame(df.loc[_where, ['district_num_of_residence', 'ma_age_edited']])

            # get the monthly incidence probabilities for these individuals
            monthly_prob = curr_inc.loc[district_age_lookup, _col]
            # update the index so it's the same as the original population dataframe for these individuals
            monthly_prob = monthly_prob.set_axis(df.index[_where])

            # the linear models only apply to clinical and severe malaria risk
            if _col == 'monthly_prob_inf':
                # select individuals for infection
                random_draw = rng.random_sample(_where.sum()) < monthly_prob

            else:
                linear_model = self.lm["rr_of_clinical_malaria"] if _col == 'monthly_prob_clin' else self.lm[
                    "rr_of_severe_malaria"]

                # apply linear model to get individual risk
                individual_risk = linear_model.predict(
                    df.loc[_where]
                )

                random_draw = rng.random_sample(_where.sum()) < (monthly_prob * individual_risk)

            selected = _where & random_draw

            return selected

        # we don't have incidence data for over 80s
        alive = df.is_alive & (df.age_years < 80)

        alive_over_one = alive & (df.age_exact_years >= 1)
        df.loc[alive & df.age_exact_years.between(0, 0.5), 'ma_age_edited'] = 0.0
        df.loc[alive & df.age_exact_years.between(0.5, 1), 'ma_age_edited'] = 0.5
        df.loc[alive_over_one, 'ma_age_edited'] = df.loc[alive_over_one, 'age_years'].astype(float)

        # select new infections
        # eligible: uninfected or asym
        alive_uninfected = alive & df.ma_inf_type.isin(['none', 'asym'])
        now_infected = _draw_incidence_for('monthly_prob_inf', alive_uninfected)
        df.loc[now_infected, 'ma_inf_type'] = 'asym'

        # draw from currently asymptomatic to allocate clinical cases
        # this can include people who became infected/asym in previous polls
        alive_infected_asym = alive & (df.ma_inf_type == 'asym')
        now_clinical = _draw_incidence_for('monthly_prob_clin', alive_infected_asym)
        df.loc[now_clinical, 'ma_inf_type'] = 'clinical'

        # draw from clinical cases to allocate severe cases - draw from all currently clinical cases
        alive_infected_clinical = alive & (df.ma_inf_type == 'clinical')
        now_severe = _draw_incidence_for('monthly_prob_sev', alive_infected_clinical)
        df.loc[now_severe, 'ma_inf_type'] = 'severe'

        # ----------------------------------- ASSIGN INFECTION DATES -----------------------------------

        # index now_clinical does not always include now_severe
        # new severe infections may have been drawn from those assigned clinical in previous poll
        new_clinical = df.loc[now_clinical].index
        new_severe = df.loc[now_severe].index
        new_infections = df.loc[now_infected].index

        # create list of all new infections
        all_new_infections = list(new_infections)
        all_new_infections.extend(x for x in new_clinical if x not in all_new_infections)
        all_new_infections.extend(x for x in new_severe if x not in all_new_infections)

        # scatter infection dates across the month
        # now_infected includes all new infections this month
        # join all indices (some clinical infections drawn from asymptomatic infections from previous months)
        for idx in all_new_infections:
            date_of_infection = now + pd.DateOffset(days=self.rng.randint(1, 30))
            df.at[idx, 'ma_date_infected'] = date_of_infection

        assert (df.loc[all_new_infections, 'ma_date_infected'] >= self.sim.date).all()

        # assign date of symptom onset
        df.loc[new_clinical, 'ma_date_symptoms'] = df.loc[new_clinical, 'ma_date_infected'] + DateOffset(days=7)
        df.loc[new_severe, 'ma_date_symptoms'] = df.loc[new_severe, 'ma_date_infected'] + DateOffset(days=7)

        # ----------------------------------- CLINICAL MALARIA SYMPTOMS -----------------------------------

        # check symptom onset occurs in one week
        if len(new_clinical):
            assert (df.loc[new_clinical, 'ma_date_infected'] < df.loc[new_clinical, 'ma_date_symptoms']).all()
            assert not pd.isnull(df.loc[new_clinical, 'ma_date_symptoms']).all()

        # ----------------------------------- SCHEDULED DEATHS -----------------------------------
        # schedule deaths within the next week
        # Assign time of infections across the month

        # the cfr applies to all severe malaria
        random_draw = rng.random_sample(size=len(new_severe))
        death = df.index[new_severe][random_draw < (p['cfr'] * p['mortality_adjust'])]

        for person in death:
            logger.debug(key='message',
                         data=f'MalariaEvent: scheduling malaria death for person {person}')

            # death occurs 1-7 days after symptom onset
            date_death = df.at[person, 'ma_date_symptoms'] + DateOffset(days=rng.randint(low=1, high=7))

            death_event = MalariaDeathEvent(
                self, person_id=person, cause='Malaria'
            )  # make that death event
            self.sim.schedule_event(
                death_event, date_death
            )  # schedule the death

    def general_population_rdt_scheduler(self, population):
        """
        schedule rdt for general population - performed in the community by DCSAs
        independent of any current symptoms
        rates are set to match rdt usage reports from WHO / NMCP
        """
        df = population.props
        rng = self.rng
        p = self.parameters

        # extract annual testing rates from NMCP reports
        # this is the # rdts issued divided by population size
        year = self.sim.date.year if self.sim.date.year <= 2024 else 2024

        test_rates = (
            p['rdt_testing_rates'].set_index('Year')['Rate_rdt_testing'].dropna()
        )
        rdt_rate = test_rates.loc[min(test_rates.index.max(), year)] / 12

        # adjust rdt usage reported rate to reflect consumables availability
        rdt_rate = rdt_rate * p['scaling_factor_for_rdt_availability']

        # testing trends independent of any demographic characteristics
        # no rdt offered if currently on anti-malarials
        random_draw = rng.random_sample(size=len(df))
        will_test_idx = df.loc[df.is_alive & (df.ma_tx == 'none') & (random_draw < rdt_rate)].index

        for person_id in will_test_idx:
            date_test = self.sim.date + pd.DateOffset(
                days=self.rng.randint(0, 30)
            )
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_Malaria_rdt_community(person_id=person_id, module=self),
                priority=1,
                topen=date_test,
                tclose=date_test + pd.DateOffset(days=7),
            )

    def initialise_simulation(self, sim):
        """
        * 1) Schedule the Main Regular Polling Events
        * 2) Define the DxTests
        * 3) Look-up and save the codes for consumables
        """

        # 1) ----------------------------------- REGULAR EVENTS -----------------------------------

        sim.schedule_event(MalariaPollingEventDistrict(self), sim.date + DateOffset(days=0))

        sim.schedule_event(MalariaUpdateEvent(self), sim.date + DateOffset(days=0))
        sim.schedule_event(MalariaParasiteClearanceEvent(self), sim.date + DateOffset(months=1))

        if 'CareOfWomenDuringPregnancy' not in self.sim.modules:
            sim.schedule_event(MalariaIPTp(self), sim.date + DateOffset(days=30.5))

        # add logger events
        sim.schedule_event(MalariaLoggingEvent(self), sim.date + DateOffset(years=1))
        sim.schedule_event(MalariaTxLoggingEvent(self), sim.date + DateOffset(years=1))
        sim.schedule_event(MalariaPrevDistrictLoggingEvent(self), sim.date + DateOffset(months=1))

        # Optional: Schedule the scale-up of programs
        if self.parameters["type_of_scaleup"] != 'none':
            scaleup_start_date = Date(self.parameters["scaleup_start_year"], 1, 1)
            assert scaleup_start_date >= self.sim.start_date, f"Date {scaleup_start_date} is before simulation starts."
            sim.schedule_event(MalariaScaleUpEvent(self), scaleup_start_date)

        # 2) ----------------------------------- DIAGNOSTIC TESTS -----------------------------------
        # Create the diagnostic test representing the use of RDT for malaria diagnosis
        # and registers it with the Diagnostic Test Manager

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            malaria_rdt=DxTest(
                property='ma_is_infected',
                item_codes=self.sim.modules['HealthSystem'].get_item_code_from_item_name('Malaria test kit (RDT)'),
                sensitivity=self.parameters['sensitivity_rdt'],
            )
        )

        # 3) ----------------------------------- CONSUMABLES -----------------------------------
        get_item_code = self.sim.modules['HealthSystem'].get_item_code_from_item_name

        # malaria rdt
        self.item_codes_for_consumables_required['malaria_rdt'] = get_item_code('Malaria test kit (RDT)')

        # malaria treatment uncomplicated children <15kg
        self.item_codes_for_consumables_required['malaria_uncomplicated_young_children'] = get_item_code(
            'Lumefantrine 120mg/Artemether 20mg,  30x18_540_CMST')

        self.item_codes_for_consumables_required['paracetamol_syrup'] = get_item_code(
            'Paracetamol syrup 120mg/5ml_0.0119047619047619_CMST')

        # malaria treatment uncomplicated children >15kg
        self.item_codes_for_consumables_required['malaria_uncomplicated_older_children'] = get_item_code(
            'Lumefantrine 120mg/Artemether 20mg,  30x18_540_CMST')

        # malaria treatment uncomplicated adults >36kg
        self.item_codes_for_consumables_required['malaria_uncomplicated_adult'] = get_item_code(
            'Lumefantrine 120mg/Artemether 20mg,  30x18_540_CMST')

        self.item_codes_for_consumables_required['paracetamol'] = get_item_code('Paracetamol 500mg_1000_CMST')

        # malaria treatment complicated - same consumables for adults and children
        self.item_codes_for_consumables_required['malaria_complicated_artesunate'] = get_item_code('Injectable artesunate')

        self.item_codes_for_consumables_required['malaria_complicated_optional_items'] = [
            get_item_code('Malaria test kit (RDT)'),
            get_item_code('Cannula iv  (winged with injection pot) 18_each_CMST'),
            get_item_code('Disposables gloves, powder free, 100 pieces per box'),
            get_item_code('Gauze, absorbent 90cm x 40m_each_CMST'),
            get_item_code('Water for injection, 10ml_Each_CMST')
        ]

        # malaria IPTp for pregnant women
        self.item_codes_for_consumables_required['malaria_iptp'] = get_item_code(
            'Fansidar (sulphadoxine / pyrimethamine tab)'
        )

    def update_parameters_for_program_scaleup(self):
        """ options for program scale-up are 'target' or 'max' """
        p = self.parameters
        scaled_params_workbook = p["scaleup_parameters"]

        if p['type_of_scaleup'] == 'target':
            scaled_params = scaled_params_workbook.set_index('parameter')['target_value'].to_dict()
        else:
            scaled_params = scaled_params_workbook.set_index('parameter')['max_value'].to_dict()

        # scale-up malaria program
        # increase testing
        # prob_malaria_case_tests=0.4 default
        p["prob_malaria_case_tests"] = scaled_params["prob_malaria_case_tests"]

        # gen pop testing rates
        # annual Rate_rdt_testing=0.64 at 2023
        p["rdt_testing_rates"]["Rate_rdt_testing"] = scaled_params["rdt_testing_rates"]

        # treatment reaches XX
        # no default between testing and treatment, governed by tx availability

        # coverage IPTp reaches XX
        # given during ANC visits and MalariaIPTp Event which selects ALL eligible women

        # treatment success reaches 1 - default is currently 1 also
        p["prob_of_treatment_success"] = scaled_params["prob_of_treatment_success"]

        # bednet and ITN coverage
        # set IRS for 4 high-risk districts
        # lookup table created in malaria read_parameters
        # produces self.itn_irs called by malaria poll to draw incidence
        # need to overwrite this
        highrisk_distr_num = p["highrisk_districts"]["district_num"]

        # Find indices where District_Num is in highrisk_distr_num
        mask = self.itn_irs['irs_rate'].index.get_level_values('District_Num').isin(
            highrisk_distr_num)

        # IRS values can be 0 or 0.8 - no other value in lookup table
        self.itn_irs['irs_rate'].loc[mask] = scaled_params["irs_district"]

        # set ITN for all districts
        # Set these values to 0.7 - this is the max value possible in lookup table
        # equivalent to 0.7 of all pop sleeping under bednet
        # household coverage could be 100%, but not everyone in household sleeping under bednet
        self.itn_irs['itn_rate'] = scaled_params["itn_district"]

        # itn rates for 2019 onwards
        p["itn"] = scaled_params["itn"]

        # update exising linear models to use new scaled-up parameters
        self._build_linear_models()

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props
        df.at[child_id, 'ma_is_infected'] = False
        df.at[child_id, 'ma_date_infected'] = pd.NaT
        df.at[child_id, 'ma_date_symptoms'] = pd.NaT
        df.at[child_id, 'ma_date_death'] = pd.NaT
        df.at[child_id, 'ma_tx'] = 'none'
        df.at[child_id, 'ma_date_tx'] = pd.NaT
        df.at[child_id, 'ma_inf_type'] = 'none'
        df.at[child_id, 'ma_age_edited'] = 0.0
        df.at[child_id, 'ma_clinical_counter'] = 0
        df.at[child_id, 'ma_clinical_preg_counter'] = 0
        df.at[child_id, 'ma_dx_counter'] = 0
        df.at[child_id, 'ma_tx_counter'] = 0
        df.at[child_id, 'ma_iptp'] = False

        # reset mother's IPTp status to False
        if mother_id >= 0:  # exclude direct births
            df.at[mother_id, 'ma_iptp'] = False

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug(key='message',
                     data='This is malaria reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe
        p = self.parameters

        health_values = df.loc[df.is_alive, 'ma_inf_type'].map(
            {
                'none': 0,
                'asym': 0,
                'clinical': p['daly_wt_clinical'],
                'severe': p['daly_wt_severe'],
            }
        )
        health_values.name = 'Malaria'  # label the cause of this disability

        return health_values.loc[df.is_alive]  # returns the series

    def check_if_fever_is_caused_by_malaria(
        self,
        true_malaria_infection_type: str,
        diagnosis_function: DiagnosisFunction,
        person_id: Optional[int] = None,
        fever_is_a_symptom: Optional[bool] = True,
        patient_age: Optional[Union[int, float]] = None,
        facility_level: Optional[str] = None,
        treatment_id: Optional[str] = None,
    ) -> Literal["severe_malaria", "clinical_malaria", "negative_malaria_test"]:
        """
        Run by an HSI when an adult presents with fever.
        Determine if the cause is malaria.

        Optional arguments are used by the logger,
        and are not needed in the diagnosis.
        """
        # Call the DxTest RDT to diagnose malaria
        dx_result = diagnosis_function('malaria_rdt')

        # Log the test: line-list of summary information about each test
        logger.info(
            key="rdt_log",
            data=_data_for_rdt_log(
                person_id=person_id,
                age=patient_age,
                fever_is_a_symptom=fever_is_a_symptom,
                dx_result=dx_result,
                facility_level=facility_level,
                treatment_id=treatment_id
            )
        )

        # Severe malaria infection always returns positive RDT
        if true_malaria_infection_type == 'severe':
            return 'severe_malaria'
        elif dx_result and true_malaria_infection_type in ('clinical', 'asym'):
            return 'clinical_malaria'
        else:
            return 'negative_malaria_test'

    def do_at_generic_first_appt(
        self,
        person_id: int,
        individual_properties: IndividualProperties,
        symptoms: List[str],
        schedule_hsi_event: HSIEventScheduler,
        diagnosis_function: DiagnosisFunction,
        facility_level: str,
        treatment_id: str,
        **kwargs,
    ) -> None:
        malaria_associated_symptoms = {
            "fever",
            "headache",
            "stomachache",
            "diarrhoea",
            "vomiting",
        }
        if (
            bool(set(symptoms) & malaria_associated_symptoms)
            and individual_properties["ma_tx"] == "none"
        ):
            malaria_test_result = self.check_if_fever_is_caused_by_malaria(
                true_malaria_infection_type=individual_properties["ma_inf_type"],
                diagnosis_function=diagnosis_function,
                person_id=person_id,
                fever_is_a_symptom="fever" in symptoms,
                patient_age=individual_properties["age_years"],
                facility_level=facility_level,
                treatment_id=treatment_id,
            )
            # Treat / refer based on diagnosis
            if malaria_test_result == "severe_malaria":
                individual_properties["ma_dx_counter"] += 1
                event = HSI_Malaria_Treatment_Complicated(person_id=person_id, module=self)
                schedule_hsi_event(
                    event, priority=0, topen=self.sim.date
                )

            # return type 'clinical_malaria' includes asymptomatic infection
            elif malaria_test_result == "clinical_malaria":
                individual_properties["ma_dx_counter"] += 1
                event = HSI_Malaria_Treatment(person_id=person_id, module=self)
                schedule_hsi_event(
                    event, priority=1, topen=self.sim.date
                )

    def do_at_generic_first_appt_emergency(
        self,
        person_id: int,
        individual_properties: IndividualProperties,
        symptoms: List[str],
        schedule_hsi_event: HSIEventScheduler,
        diagnosis_function: DiagnosisFunction,
        facility_level: str,
        treatment_id: str,
        **kwargs,
    ) -> None:
        # This is called for a person (of any age) that attends an
        # emergency generic HSI and has a fever.
        # (Quick diagnosis algorithm - just perfectly recognises the
        # symptoms of severe malaria.)
        if 'severe_malaria' in symptoms:
            if individual_properties["ma_tx"] == 'none':
                # Check if malaria parasitaemia:
                malaria_test_result = self.check_if_fever_is_caused_by_malaria(
                    true_malaria_infection_type=individual_properties["ma_inf_type"],
                    diagnosis_function=diagnosis_function,
                    person_id=person_id,
                    fever_is_a_symptom="fever" in symptoms,
                    patient_age=individual_properties["age_years"],
                    facility_level=facility_level,
                    treatment_id=treatment_id,
                )

                # if any symptoms indicative of malaria and they have parasitaemia (would return a positive rdt)
                if malaria_test_result in ('severe_malaria', 'clinical_malaria'):
                    individual_properties['ma_dx_counter'] += 1

                    # Launch the HSI for treatment for Malaria, HSI_Malaria_Treatment will determine correct treatment
                    event = HSI_Malaria_Treatment_Complicated(
                        person_id=person_id, module=self,
                    )
                    schedule_hsi_event(
                        event, priority=0, topen=self.sim.date
                    )


class MalariaPollingEventDistrict(RegularEvent, PopulationScopeEventMixin):
    """
    this calls functions to assign new malaria infections
    and schedules rdt at a community level (non-symptom driven)
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        logger.debug(key='message', data='MalariaEvent: tracking the disease progression of the population')

        # assigns new malaria infections
        self.module.malaria_poll2(population)

        # schedule rdt for general population, rate increases over time
        self.module.general_population_rdt_scheduler(population)


class MalariaScaleUpEvent(Event, PopulationScopeEventMixin):
    """ This event exists to change parameters or functions
    depending on the scenario for projections which has been set
    It only occurs once on date: scaleup_start_date,
    called by initialise_simulation
    """

    def __init__(self, module):
        super().__init__(module)

    def apply(self, population):

        self.module.update_parameters_for_program_scaleup()


class MalariaIPTp(RegularEvent, PopulationScopeEventMixin):
    """
    malaria prophylaxis for pregnant women
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        df = population.props
        now = self.sim.date

        # select currently pregnant women without IPTp, malaria-negative, not on cotrimoxazole
        p1_condition = (
            df.is_alive
            & df.is_pregnant
            & ~df.ma_is_infected
            & ~df.ma_iptp
            & (
                ~df.hv_on_cotrimoxazole
                if 'Hiv' in self.sim.modules
                else True
            )
        )

        p1 = df.index[p1_condition]

        for person_index in p1:
            logger.debug(key='message',
                         data=f'MalariaIPTp: scheduling HSI_Malaria_IPTp for person {person_index}')

            event = HSI_MalariaIPTp(self.module, person_id=person_index)
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                event, priority=1, topen=now, tclose=None
            )


class MalariaEndIPTpProtection(Event, IndividualScopeEventMixin):
    """
    This resets the properties of a person on IPTp
    the protective effects ends after 6 weeks and so the property is reset to prevent the
    malaria poll assuming that this person still has reduced susceptibility to malaria infection
    """

    def __init__(self, module, person_id, ):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props

        if not df.at[person_id, 'is_alive'] or not df.at[person_id, 'ma_iptp']:
            return

        # reset the IPTp property
        df.at[person_id, 'ma_iptp'] = False


class MalariaDeathEvent(Event, IndividualScopeEventMixin):
    """
    Performs the Death operation on an individual and logs it.
    """

    def __init__(self, module, person_id, cause):
        super().__init__(module, person_id=person_id)
        self.cause = cause

    def apply(self, person_id):
        df = self.sim.population.props

        if not df.at[person_id, 'is_alive'] or (df.at[person_id, 'ma_inf_type'] == 'none'):
            return

        # if on treatment for severe malaria, will reduce probability of death
        # use random number generator - currently param treatment_adjustment set to 0.5
        if df.at[person_id, 'ma_tx'] == 'complicated':

            prob = self.module.rng.rand()

            # if draw -> death
            if prob < self.module.parameters['treatment_adjustment']:
                self.sim.modules['Demography'].do_death(
                    individual_id=person_id, cause=self.cause, originating_module=self.module)

                df.at[person_id, 'ma_date_death'] = self.sim.date

            # else if draw does not result in death -> cure
            else:
                df.at[person_id, 'ma_tx'] = 'none'
                df.at[person_id, 'ma_inf_type'] = 'none'
                df.at[person_id, 'ma_is_infected'] = False

                # clear symptoms
                self.sim.modules['SymptomManager'].clear_symptoms(
                    person_id=person_id, disease_module=self.module
                )

        # if not on treatment - death will occur
        else:
            self.sim.modules['Demography'].do_death(
                individual_id=person_id, cause=self.cause, originating_module=self.module)

            df.at[person_id, 'ma_date_death'] = self.sim.date


# ---------------------------------------------------------------------------------
# Health System Interaction Events
# ---------------------------------------------------------------------------------


class HSI_Malaria_rdt(HSI_Event, IndividualScopeEventMixin):
    """
    this is a point-of-care malaria rapid diagnostic test, with results within 2 minutes
    default facility level is 1a unless specified
    """

    def __init__(self, module, person_id, facility_level='1a'):

        super().__init__(module, person_id=person_id)
        assert isinstance(module, Malaria)

        self.TREATMENT_ID = 'Malaria_Test'
        self.facility_level = facility_level

        df = self.sim.population.props
        person_age_years = df.at[self.target, 'age_years']
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({
            'Under5OPD' if person_age_years < 5 else 'Over5OPD': 1}
        )
        self.ACCEPTED_FACILITY_LEVEL = '1a' if (self.facility_level == '1a') else '1b'

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        hs = self.sim.modules['HealthSystem']

        # Ignore this event if the person is no longer alive or already on treatment
        if not df.at[person_id, 'is_alive'] or (df.at[person_id, 'ma_tx'] != 'none'):
            return hs.get_blank_appt_footprint()

        district = df.at[person_id, 'district_num_of_residence']
        logger.debug(key='message',
                     data=f'HSI_Malaria_rdt: rdt test for person {person_id} '
                          f'in district num {district}')

        # call the DxTest RDT to diagnose malaria
        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='malaria_rdt',
            hsi_event=self
        )

        # Log the test: line-list of summary information about each test
        fever_present = 'fever' in self.sim.modules["SymptomManager"].has_what(person_id=person_id)
        person_details_for_test = _data_for_rdt_log(
            person_id=person_id,
            age=df.at[person_id, 'age_years'],
            fever_is_a_symptom=fever_present,
            dx_result=dx_result,
            facility_level=self.ACCEPTED_FACILITY_LEVEL,
            treatment_id=self.TREATMENT_ID,
        )
        logger.info(key='rdt_log', data=person_details_for_test)

        if dx_result:
            # ----------------------------------- SEVERE MALARIA -----------------------------------

            df.at[person_id, 'ma_dx_counter'] += 1

            # if severe malaria, treat for complicated malaria
            if df.at[person_id, 'ma_inf_type'] == 'severe':

                logger.debug(key='message',
                             data=f'HSI_Malaria_rdt: scheduling HSI_Malaria_Treatment_Complicated {person_id}'
                                  f'on date {self.sim.date}')

                treat = HSI_Malaria_Treatment_Complicated(
                    self.sim.modules['Malaria'], person_id=person_id
                )
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    treat, priority=0, topen=self.sim.date, tclose=None
                )

            # ----------------------------------- TREATMENT CLINICAL DISEASE -----------------------------------

            # clinical malaria - not severe
            # this will allow those with asym malaria (positive RDT) to also be treated
            else:
                logger.debug(key='message',
                             data=f'HSI_Malaria_rdt scheduling HSI_Malaria_Treatment for person {person_id}'
                                  f'on date {self.sim.date}')

                treat = HSI_Malaria_Treatment(self.module, person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    treat, priority=1, topen=self.sim.date, tclose=None
                )

        elif dx_result is None:

            # repeat appt for rdt and move to level 1b regardless of current facility level
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Malaria_rdt(person_id=person_id, module=self.module, facility_level='1b'),
                topen=self.sim.date + pd.DateOffset(days=1),
                tclose=None,
                priority=0,
            )
            # Test was not possible, set blank footprint and schedule another test
            ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint({})

            return ACTUAL_APPT_FOOTPRINT


class HSI_Malaria_rdt_community(HSI_Event, IndividualScopeEventMixin):
    """
    this is a point-of-care malaria rapid diagnostic test, with results within 2 minutes
    this is performed in the community at facility level 0 by a DCSA
    positive result will schedule a referral to HSI_Malaria_rdt at facility level 1a
    where a confirmatory rdt will be performed and treatment will be scheduled
    """

    def __init__(self, module, person_id):

        super().__init__(module, person_id=person_id)
        assert isinstance(module, Malaria)

        self.TREATMENT_ID = 'Malaria_Test'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ConWithDCSA': 1})
        self.ACCEPTED_FACILITY_LEVEL = '0'

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        hs = self.sim.modules['HealthSystem']

        # Ignore this event if the person is no longer alive or already on treatment
        if not df.at[person_id, 'is_alive'] or not (df.at[person_id, 'ma_tx'] == 'none'):
            return hs.get_blank_appt_footprint()

        # call the DxTest RDT to diagnose malaria
        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='malaria_rdt',
            hsi_event=self
        )

        # Log the test: line-list of summary information about each test
        fever_present = 'fever' in self.sim.modules["SymptomManager"].has_what(person_id=person_id)
        person_details_for_test = _data_for_rdt_log(
            person_id=person_id,
            age=df.at[person_id, 'age_years'],
            fever_is_a_symptom=fever_present,
            dx_result=dx_result,
            facility_level=self.ACCEPTED_FACILITY_LEVEL,
            treatment_id=self.TREATMENT_ID,
        )

        logger.info(key='rdt_log', data=person_details_for_test)

        # if positive, refer for a confirmatory test at level 1a
        if dx_result:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_Malaria_rdt(person_id=person_id, module=self.module, facility_level='1a'),
                priority=1,
                topen=self.sim.date,
                tclose=self.sim.date + pd.DateOffset(days=1),
            )


class HSI_Malaria_Treatment(HSI_Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for all ages. Includes treatment plus one rdt
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Malaria)

        self.TREATMENT_ID = 'Malaria_Treatment'

        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({
            ('Under5OPD' if self.sim.population.props.at[person_id, "age_years"] < 5 else 'Over5OPD'): 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        person = df.loc[person_id]

        # if not on treatment already - request treatment
        if person['ma_tx'] == 'none':

            logger.debug(key='message',
                         data=f'HSI_Malaria_Treatment: requesting malaria treatment for {person_id}')

            # Check if drugs are available, and provide drugs:
            drugs_available = self.get_drugs(age_of_person=person['age_years'])

            if drugs_available:

                logger.debug(key='message',
                             data=f'HSI_Malaria_Treatment: giving malaria treatment for {person_id}')

                if df.at[person_id, 'is_alive']:
                    df.at[person_id, 'ma_tx'] = 'uncomplicated'
                    df.at[person_id, 'ma_date_tx'] = self.sim.date
                    df.at[person_id, 'ma_tx_counter'] += 1

                # rdt is offered as part of the treatment package
                # Log the test: line-list of summary information about each test
                fever_present = 'fever' in self.sim.modules["SymptomManager"].has_what(person_id=person_id)
                person_details_for_test = _data_for_rdt_log(
                    person_id=person_id,
                    age=df.at[person_id, 'age_years'],
                    fever_is_a_symptom=fever_present,
                    dx_result=True,
                    facility_level=self.ACCEPTED_FACILITY_LEVEL,
                    treatment_id=self.TREATMENT_ID,
                )
                logger.info(key='rdt_log', data=person_details_for_test)

    def get_drugs(self, age_of_person):
        """
        :param age_of_person:
        :return:

        Helper function to get treatment according to the age of the person being treated. Returns bool to indicate
        whether drugs were available"""

        # non-complicated malaria
        if age_of_person < 5:
            # Formulation for young children
            # 5–14kg: 1 tablet(120mg Lumefantrine / 20mg Artemether) every 12 hours for 3 days
            # paracetamol syrup in 1ml doses, 10ml 4x per day, 3 days
            drugs_available = self.get_consumables(
                item_codes={self.module.item_codes_for_consumables_required['malaria_uncomplicated_young_children']: 6},
                optional_item_codes={self.module.item_codes_for_consumables_required['paracetamol_syrup']: 120,
                                     self.module.item_codes_for_consumables_required['malaria_rdt']: 1}
            )

        elif 5 <= age_of_person <= 15:
            # Formulation for older children
            # 35–44 kg: 4 tablets every 12 hours for 3 days
            # paracetamol syrup in 1ml doses, 15ml 4x per day, 3 days
            drugs_available = self.get_consumables(
                item_codes={self.module.item_codes_for_consumables_required['malaria_uncomplicated_older_children']: 24},
                optional_item_codes={self.module.item_codes_for_consumables_required['paracetamol_syrup']: 180,
                                     self.module.item_codes_for_consumables_required['malaria_rdt']: 1}
            )

        else:
            # Formulation for adults
            # 4 tablets every 12 hours for 3 day
            # paracetamol in 1 mg doses, 4g per day for 3 days
            drugs_available = self.get_consumables(
                item_codes={self.module.item_codes_for_consumables_required['malaria_uncomplicated_adult']: 24},
                optional_item_codes={self.module.item_codes_for_consumables_required['paracetamol']: 12_000,
                                     self.module.item_codes_for_consumables_required['malaria_rdt']: 1}
            )

        return drugs_available

    def did_not_run(self):
        logger.debug(key='message',
                     data='HSI_Malaria_Treatment: did not run')
        pass


class HSI_Malaria_Treatment_Complicated(HSI_Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for complicated malaria in all ages
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Malaria)

        self.TREATMENT_ID = 'Malaria_Treatment_Complicated'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({
            ('Under5OPD' if self.sim.population.props.at[person_id, "age_years"] < 5 else 'Over5OPD'): 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 5})

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props

        # if person is not on treatment and still alive
        if (df.at[person_id, 'ma_tx'] == 'none') and df.at[person_id, 'is_alive']:

            logger.debug(key='message',
                         data=f'HSI_Malaria_Treatment_Complicated: requesting complicated malaria treatment for '
                              f' {person_id}')

            # dosage in 60mg artesunate ampoules
            # First dose: 2.4 mg/kg × 25 kg = 60 mg (administered IV or IM).
            # Repeat 60 mg after 12 hours and then again at 24 hours.
            if self.get_consumables(
                item_codes={self.module.item_codes_for_consumables_required['malaria_complicated_artesunate']: 3},
                optional_item_codes=self.module.item_codes_for_consumables_required[
                    'malaria_complicated_optional_items']
            ):
                logger.debug(key='message',
                             data=f'HSI_Malaria_Treatment_Complicated: giving complicated malaria treatment for '
                                  f' {person_id}')

                df.at[person_id, 'ma_tx'] = 'complicated'
                df.at[person_id, 'ma_date_tx'] = self.sim.date
                df.at[person_id, 'ma_tx_counter'] += 1

                # Add used equipment
                self.add_equipment({'Drip stand', 'Haemoglobinometer',
                                       'Analyser, Combined Chemistry and Electrolytes'})

                # rdt is offered as part of the treatment package
                # Log the test: line-list of summary information about each test
                fever_present = 'fever' in self.sim.modules["SymptomManager"].has_what(person_id=person_id)
                person_details_for_test = _data_for_rdt_log(
                    person_id=person_id,
                    age=df.at[person_id, 'age_years'],
                    fever_is_a_symptom=fever_present,
                    dx_result=True,
                    facility_level=self.ACCEPTED_FACILITY_LEVEL,
                    treatment_id=self.TREATMENT_ID,
                )
                logger.info(key='rdt_log', data=person_details_for_test)

                # schedule ACT to follow inpatient care, this is delivered through outpatient facility
                continue_to_treat = HSI_Malaria_Treatment(self.module, person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    continue_to_treat, priority=1, topen=self.sim.date, tclose=None
                )

    def did_not_run(self):
        logger.debug(key='message',
                     data='HSI_Malaria_Treatment_Complicated: did not run')


class HSI_MalariaIPTp(HSI_Event, IndividualScopeEventMixin):
    """
    this is IPTp for pregnant women
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Malaria)

        self.TREATMENT_ID = 'Malaria_Prevention_Iptp'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        p = self.module.parameters

        if not df.at[person_id, 'is_alive'] or (df.at[person_id, 'ma_tx'] != 'none'):
            return

        # IPTp contra-indicated if currently on cotrimoxazole
        if 'Hiv' in self.sim.modules and df.at[person_id, "hv_on_cotrimoxazole"]:
            return

        logger.debug(key='message',
                     data=f'HSI_MalariaIPTp: requesting IPTp for person {person_id}')

        # request the treatment
        # dosage is one tablet
        if self.get_consumables(self.module.item_codes_for_consumables_required['malaria_iptp']):
            logger.debug(key='message',
                         data=f'HSI_MalariaIPTp: giving IPTp for person {person_id}')

            df.at[person_id, 'ma_iptp'] = True

            # if currently infected, IPTp will clear the infection
            df.at[person_id, 'ma_is_infected'] = False
            df.at[person_id, 'ma_inf_type'] = 'none'

            # clear any symptoms
            self.sim.modules['SymptomManager'].clear_symptoms(
                person_id=person_id, disease_module=self.module
            )

            # If person has been placed/continued on IPTp, schedule end of protective period
            self.sim.schedule_event(
                MalariaEndIPTpProtection(
                    person_id=person_id, module=self.module
                ),
                self.sim.date + pd.DateOffset(days=7 * p["duration_iptp_protection_weeks"]),
            )

    def did_not_run(self):

        logger.debug(key='message',
                     data='HSI_MalariaIPTp: did not run')
        pass


# ---------------------------------------------------------------------------------
# Recovery Events
# ---------------------------------------------------------------------------------
class MalariaUpdateEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(days=1))

    def apply(self, population):
        """
        this is a regular event for clinical and severe cases which:
        * assigns symptoms
        * schedules rdt
        * cures people currently on treatment for malaria
        * clears symptoms for those not on treatment but self-cured
        * clears parasites if treated
        """

        logger.debug(key='message', data='MalariaUpdateEvent')

        df = self.sim.population.props
        p = self.module.parameters
        now = self.sim.date

        # assign symptoms
        # find those with schedule date of symptoms = today
        new_symptomatic_clinical = df.loc[
            df.is_alive
            & (df.ma_inf_type == 'clinical')
            & (df.ma_date_symptoms == now)].index

        new_symptomatic_severe = df.loc[
            df.is_alive
            & (df.ma_inf_type == 'severe')
            & (df.ma_date_symptoms == now)].index

        new_symptomatic_pregnant = df.loc[
            df.is_alive
            & ((df.ma_inf_type == 'clinical') | (df.ma_inf_type == 'severe'))
            & df.is_pregnant
            & (df.ma_date_symptoms == now)].index

        # assign clinical symptoms
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=new_symptomatic_clinical,
            symptom_string=['fever', 'headache', 'vomiting', 'stomachache'],
            add_or_remove='+',
            disease_module=self.module,
            date_of_onset=now,
            duration_in_days=None,  # remove duration as symptoms cleared by MalariaCureEvent
        )

        # assign symptoms if pregnant
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=new_symptomatic_pregnant,
            symptom_string='severe_anaemia',
            add_or_remove='+',
            disease_module=self.module,
            date_of_onset=now,
            duration_in_days=None,  # remove duration as symptoms cleared by MalariaCureEvent
        )

        # assign severe symptom
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=new_symptomatic_severe,
            symptom_string='severe_malaria',
            add_or_remove='+',
            disease_module=self.module,
            date_of_onset=now,
            duration_in_days=None,  # remove duration as symptoms cleared by MalariaCureEvent
        )

        # create list of all new symptomatic cases
        all_new_infections = sorted(set(new_symptomatic_clinical).union(new_symptomatic_severe))

        # clinical counter
        df.loc[all_new_infections, 'ma_clinical_counter'] += 1
        df.loc[all_new_infections, 'ma_is_infected'] = True

        # sample those scheduled for rdt
        eligible_for_rdt = df.loc[df.is_alive & (df.ma_date_symptoms == now)].index
        selected_for_rdt = self.module.rng.random_sample(size=len(eligible_for_rdt)) < p['prob_malaria_case_tests']

        for idx in eligible_for_rdt[selected_for_rdt]:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Malaria_rdt(self.module, person_id=idx, facility_level='1a'),
                priority=1,
                topen=random_date(now + DateOffset(days=1),
                                  now + DateOffset(days=4),
                                  self.module.rng),
                tclose=None
            )

        # TREATED
        # select people with clinical malaria and treatment for at least 5 days
        # if treated, will clear symptoms and parasitaemia
        # this will also clear parasitaemia for asymptomatic cases picked up by routine rdt
        random_draw = self.module.rng.random_sample(size=len(df))

        clinical_and_treated = df.index[
            df.is_alive
            & (df.ma_date_tx < (self.sim.date - DateOffset(days=5)))
            & (df.ma_inf_type == 'clinical')
            & (random_draw < p['prob_of_treatment_success'])
        ]

        # select people with severe malaria and treatment for at least 7 days
        severe_and_treated = df.index[
            df.is_alive
            & (df.ma_date_tx < (self.sim.date - DateOffset(days=7)))
            & (df.ma_inf_type == 'severe')
            & (random_draw < p['prob_of_treatment_success'])
        ]

        # create list of all cases to be resolved through treatment
        infections_to_clear = sorted(set(clinical_and_treated).union(severe_and_treated))

        self.sim.modules['SymptomManager'].clear_symptoms(
            person_id=infections_to_clear, disease_module=self.module
        )

        # change properties
        df.loc[infections_to_clear, 'ma_tx'] = 'none'
        df.loc[infections_to_clear, 'ma_is_infected'] = False
        df.loc[infections_to_clear, 'ma_inf_type'] = 'none'

        # UNTREATED or TREATMENT FAILURE
        # if not treated or treatment failed, self-cure occurs after 6 days of symptoms
        # but parasites remain in blood
        clinical_not_treated = df.index[
            df.is_alive
            & (df.ma_inf_type == 'clinical')
            & (df.ma_date_symptoms < (self.sim.date - DateOffset(days=6)))
        ]

        self.sim.modules['SymptomManager'].clear_symptoms(
            person_id=clinical_not_treated, disease_module=self.module
        )

        # change properties
        df.loc[clinical_not_treated, 'ma_inf_type'] = 'asym'


class MalariaParasiteClearanceEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(days=30.5))

    def apply(self, population):
        logger.debug(key='message', data='MalariaParasiteClearanceEvent: parasite clearance for malaria cases')

        df = self.sim.population.props
        p = self.module.parameters

        # select people infected at least a period ago equal to the duration of asymptomatic infection
        asym_inf = df.index[df.is_alive &
                            (df.ma_inf_type == 'asym') &
                            (df.ma_date_infected < (self.sim.date - DateOffset(days=p['dur_asym'])))]

        df.loc[asym_inf, 'ma_inf_type'] = 'none'
        df.loc[asym_inf, 'ma_is_infected'] = False


# ---------------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------------

class MalariaLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props
        now = self.sim.date

        # ------------------------------------ INCIDENCE ------------------------------------

        # infected in the last time-step, clinical and severe cases only
        # incidence rate per 1000 person-years
        # include those cases that have died in the case load
        tmp = sum(df.ma_date_symptoms > (now - DateOffset(months=self.repeat)))
        pop = sum(df.is_alive)

        inc_1000py = ((tmp / pop) * 1000) if pop else 0

        # incidence rate clinical (inc severe) in 2-10 yr olds
        tmp2 = len(
            df.loc[
                (df.age_years.between(2, 10))
                & (df.ma_date_symptoms > (now - DateOffset(months=self.repeat)))
                ]
        )

        pop2_10 = sum(df.is_alive & (df.age_years.between(2, 10)))
        inc_1000py_2_10 = ((tmp2 / pop2_10) * 1000) if pop2_10 else 0

        inc_1000py_hiv = 0  # if running without hiv/tb

        # using clinical counter
        # sum all the counters for previous year
        clin_episodes = df[
            'ma_clinical_counter'
        ].sum()  # clinical episodes (inc severe)
        inc_counter_1000py = (clin_episodes / pop) * 1000

        clin_preg_episodes = df[
            'ma_clinical_preg_counter'
        ].sum()  # clinical episodes in pregnant women (inc severe)

        summary = {
            'number_new_cases': tmp,
            'population': pop,
            'inc_1000py': inc_1000py,
            'inc_1000py_hiv': inc_1000py_hiv,
            'new_cases_2_10': tmp2,
            'population2_10': pop2_10,
            'inc_1000py_2_10': inc_1000py_2_10,
            'inc_clin_counter': inc_counter_1000py,
            'clinical_preg_counter': clin_preg_episodes,
        }

        logger.info(key='incidence',
                    data=summary,
                    description='Summary of incident malaria cases')

        # ------------------------------------ RUNNING COUNTS ------------------------------------

        counts = {'none': 0, 'asym': 0, 'clinical': 0, 'severe': 0}
        counts.update(df.loc[df.is_alive, 'ma_inf_type'].value_counts().to_dict())

        logger.info(key='status_counts',
                    data=counts,
                    description='Running counts of incident malaria cases')

        # ------------------------------------ PARASITE PREVALENCE BY AGE ------------------------------------

        # includes all parasite positive cases: some may have low parasitaemia (undetectable)
        child2_10_inf = len(
            df[df.is_alive & (df.ma_inf_type != 'none') & (df.age_years.between(2, 10))]
        )

        # population size - children
        child2_10_pop = sum(df.is_alive & (df.age_years.between(2, 10)))

        # prevalence in children aged 2-10
        child_prev = child2_10_inf / child2_10_pop if child2_10_pop else 0

        # prevalence of clinical including severe in all ages
        total_clin = len(
            df[
                df.is_alive
                & ((df.ma_inf_type == 'clinical') | (df.ma_inf_type == 'severe'))
                ]
        )
        pop2 = sum(df.is_alive)
        prev_clin = total_clin / pop2

        prev = {
            'child2_10_prev': child_prev,
            'clinical_prev': prev_clin,
        }

        logger.info(key='prevalence',
                    data=prev,
                    description='Prevalence malaria cases')

        # ------------------------------------ CO-INFECTION PREVALENCE ------------------------------------
        if "Hiv" in self.sim.modules:
            # number of people with both HIV and clinical/severe malaria
            # output is malaria prevalence in HIV pop
            coinfection_num = len(
                df[df.is_alive & (df.ma_inf_type != 'none') & df.hv_inf]
            )

            # hiv population
            hiv_infected = len(
                df[df.is_alive & df.hv_inf]
            )

            # prevalence of malaria in HIV population
            prev_malaria_in_hiv_population = coinfection_num / hiv_infected if hiv_infected else 0

            # proportion of malaria cases with concurrent HIV infection
            malaria_infected = len(
                df[df.is_alive & (df.ma_inf_type != 'none')]
            )

            prop_malaria_cases_with_hiv = coinfection_num / malaria_infected if malaria_infected else 0

            coinfection_prevalence = {
                'coinfection_num': coinfection_num,
                'prev_malaria_in_hiv_population': prev_malaria_in_hiv_population,
                'prop_malaria_cases_with_hiv': prop_malaria_cases_with_hiv,
            }

            logger.info(key='coinfection_prevalence',
                        data=coinfection_prevalence,
                        description='Co-infection prevalence')


class MalariaTxLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        # ------------------------------------ TREATMENT COVERAGE ------------------------------------
        # prop clinical episodes which had treatment, all ages

        # sum all the counters for previous year
        dx = df['ma_dx_counter'].sum()  # treatment (inc severe)
        tx = df['ma_tx_counter'].sum()  # treatment (inc severe)
        clin = df['ma_clinical_counter'].sum()  # clinical episodes (inc severe)

        dx_coverage = dx / clin if clin else 0
        tx_coverage = tx / clin if clin else 0

        treatment = {
            'number_diagnosed': dx,
            'number_treated': tx,
            'number_clinical episodes': clin,
            'proportion_diagnosed': dx_coverage,
            'treatment_coverage': tx_coverage,
        }

        logger.info(key='tx_coverage',
                    data=treatment,
                    description='Treatment of malaria cases')

        # reset all counters
        logger.debug(key='message',
                     data=f'Resetting the malaria counter {self.sim.date}')

        df['ma_clinical_counter'] = 0
        df['ma_tx_counter'] = 0
        df['ma_dx_counter'] = 0
        df['ma_clinical_preg_counter'] = 0


class MalariaPrevDistrictLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        # ------------------------------------ PREVALENCE OF INFECTION ------------------------------------
        infected = (
            df.district_num_of_residence[df.is_alive & df.ma_is_infected].value_counts()
        )
        pop = df.district_num_of_residence[df.is_alive].value_counts()
        prev = infected / pop
        prev_ed = prev.fillna(0)
        assert prev_ed.all() >= 0  # checks
        assert prev_ed.all() <= 1

        logger.info(key='prev_district',
                    data=prev_ed.to_dict(),
                    description='District estimates of malaria prevalence')

        logger.info(key='pop_district',
                    data=pop.to_dict(),
                    description='District population sizes')


def _data_for_rdt_log(
    person_id: int,
    age: int,
    fever_is_a_symptom: bool,
    dx_result: Union[bool, None],
    facility_level: str,
    treatment_id: str,
):
    return {
        "person_id": person_id,
        "age": age,
        "fever_present": fever_is_a_symptom,
        "rdt_result": pd.array([dx_result], dtype="boolean"),
        "facility_level": facility_level,
        "called_by": treatment_id,
    }
