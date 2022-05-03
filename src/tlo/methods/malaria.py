"""
this is the malaria module which assigns malaria infections to the population: asymptomatic, clinical and severe
it also holds the hsi events pertaining to malaria testing and treatment
including the malaria RDT using DxTest

"""
from pathlib import Path

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Malaria(Module):
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

    INIT_DEPENDENCIES = {
        'Contraception', 'Demography', 'HealthSystem', 'SymptomManager'
    }

    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden'}

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
        "mal_inc": Parameter(Types.REAL, "monthly incidence of malaria in all ages"),
        "interv": Parameter(Types.REAL, "data frame of intervention coverage by year"),
        "clin_inc": Parameter(
            Types.REAL,
            "data frame of clinical incidence by age, district, intervention coverage",
        ),
        "inf_inc": Parameter(
            Types.REAL,
            "data frame of infection incidence by age, district, intervention coverage",
        ),
        "sev_inc": Parameter(
            Types.REAL,
            "data frame of severe case incidence by age, district, intervention coverage",
        ),
        "itn_district": Parameter(
            Types.REAL, "data frame of ITN usage rates by district"
        ),
        "irs_district": Parameter(
            Types.REAL, "data frame of IRS usage rates by district"
        ),
        "sev_symp_prob": Parameter(
            Types.REAL, "probabilities of each symptom for severe malaria cases"
        ),
        # "p_infection": Parameter(
        #     Types.REAL, "Probability that an uninfected individual becomes infected"
        # ),
        "sensitivity_rdt": Parameter(Types.REAL, "Sensitivity of rdt"),
        "cfr": Parameter(Types.REAL, "case-fatality rate for severe malaria"),
        "dur_asym": Parameter(Types.REAL, "duration (days) of asymptomatic malaria"),
        "dur_clin": Parameter(
            Types.REAL, "duration (days) of clinical symptoms of malaria"
        ),
        "dur_clin_para": Parameter(
            Types.REAL, "duration (days) of parasitaemia for clinical malaria cases"
        ),
        "rr_hiv": Parameter(
            Types.REAL, "relative risk of clinical malaria if hiv-positive"
        ),
        "treatment_adjustment": Parameter(
            Types.REAL, "probability of death from severe malaria if on treatment"
        ),
        "p_sev_anaemia_preg": Parameter(
            Types.REAL,
            "probability of severe anaemia in pregnant women with clinical malaria",
        ),
        "itn_proj": Parameter(
            Types.REAL, "coverage of ITN for projections 2020 onwards"
        ),
        "mortality_adjust": Parameter(
            Types.REAL, "adjustment of case-fatality rate to match WHO/MAP"
        ),
        "data_end": Parameter(
            Types.REAL, "final year of ICL malaria model outputs, after 2018 = projections"
        ),
        # "prob_sev": Parameter(
        #     Types.REAL, "probability of infected case becoming severe"
        # ),
        "irs_rates_boundary": Parameter(
            Types.REAL, "threshold for indoor residual spraying coverage"
        ),
        "irs_rates_upper": Parameter(
            Types.REAL, "indoor residual spraying high coverage"
        ),
        "irs_rates_lower": Parameter(
            Types.REAL, "indoor residual spraying low coverage"
        ),
        "testing_adj": Parameter(
            Types.REAL, "adjusted testing rates to match rdt/tx levels"
        ),
        "itn": Parameter(
            Types.REAL, "projected future itn coverage"
        ),
    }

    PROPERTIES = {
        "ma_is_infected": Property(Types.BOOL, "Current status of malaria"),
        "ma_date_infected": Property(Types.DATE, "Date of latest infection"),
        "ma_date_symptoms": Property(
            Types.DATE, "Date of symptom start for clinical infection"
        ),
        "ma_date_death": Property(Types.DATE, "Date of death due to malaria"),
        "ma_tx": Property(Types.BOOL, "Currently on anti-malarial treatment"),
        "ma_date_tx": Property(
            Types.DATE, "Date treatment started for most recent malaria episode"
        ),
        "ma_inf_type": Property(
            Types.CATEGORICAL,
            "specific symptoms with malaria infection",
            categories=["none", "asym", "clinical", "severe"],
        ),
        "ma_age_edited": Property(
            Types.REAL, "age values redefined to match with malaria data"
        ),
        "ma_clinical_counter": Property(
            Types.INT, "annual counter for malaria clinical episodes"
        ),
        "ma_tx_counter": Property(
            Types.INT, "annual counter for malaria treatment episodes"
        ),
        "ma_clinical_preg_counter": Property(
            Types.INT, "annual counter for malaria clinical episodes in pregnant women"
        ),
        "ma_iptp": Property(Types.BOOL, "if woman has IPTp in current pregnancy"),
    }

    # TODO reset ma_iptp after delivery

    def read_parameters(self, data_folder):
        workbook = pd.read_excel(self.resourcefilepath / "ResourceFile_malaria.xlsx", sheet_name=None)
        self.load_parameters_from_dataframe(workbook["parameters"])

        p = self.parameters

        # baseline characteristics
        p["mal_inc"] = workbook["incidence"]
        p["interv"] = workbook["interventions"]
        p["itn_district"] = workbook["MAP_ITNrates"]
        p["irs_district"] = workbook["MAP_IRSrates"]

        p["sev_symp_prob"] = workbook["severe_symptoms"]

        p["inf_inc"] = pd.read_csv(self.resourcefilepath / "ResourceFile_malaria_InfInc_expanded.csv")
        p["clin_inc"] = pd.read_csv(self.resourcefilepath / "ResourceFile_malaria_ClinInc_expanded.csv")
        p["sev_inc"] = pd.read_csv(self.resourcefilepath / "ResourceFile_malaria_SevInc_expanded.csv")

        # check itn projected values are <=0.7 and rounded to 1dp for matching to incidence tables
        p["itn"] = round(p["itn"], 1)
        assert (p["itn"] <= 0.7)

        # ===============================================================================
        # single dataframe for itn and irs district/year data; set index for fast lookup
        # ===============================================================================
        itn_curr = p["itn_district"]
        itn_curr.rename(columns={"itn_rates": "itn_rate"}, inplace=True)
        itn_curr["itn_rate"] = itn_curr["itn_rate"].round(decimals=1)
        # maximum itn is 0.7; see comment https://github.com/UCL/TLOmodel/pull/165#issuecomment-699625290
        itn_curr.loc[itn_curr.itn_rate > 0.7, "itn_rate"] = 0.7
        itn_curr = itn_curr.set_index(["District", "Year"])

        irs_curr = p["irs_district"]
        irs_curr.rename(columns={"irs_rates": "irs_rate"}, inplace=True)
        irs_curr.drop(["Region"], axis=1, inplace=True)
        irs_curr["irs_rate"] = irs_curr["irs_rate"].round(decimals=1)
        irs_curr.loc[irs_curr.irs_rate > p["irs_rates_boundary"], "irs_rate"] = p["irs_rates_upper"]
        irs_curr.loc[irs_curr.irs_rate <= p["irs_rates_boundary"], "irs_rate"] = p["irs_rates_lower"]
        irs_curr = irs_curr.set_index(["District", "Year"])

        itn_irs = pd.concat([itn_curr, irs_curr], axis=1)

        # Subsitute District Num for District Name
        mapper_district_name_to_num = \
            {v: k for k, v in self.sim.modules['Demography'].parameters['district_num_to_district_name'].items()}
        self.itn_irs = itn_irs.reset_index().assign(
            District_Num=lambda x: x['District'].map(mapper_district_name_to_num)
        ).drop(columns=['District']).set_index(['District_Num', 'Year'])

        # ===============================================================================
        # put the all incidence data into single table with month/admin/llin/irs index
        # ===============================================================================
        inf_inc = p["inf_inc"].set_index(["month", "admin", "llin", "irs", "age"])
        inf_inc = inf_inc.loc[:, ["monthly_prob_inf"]]

        clin_inc = p["clin_inc"].set_index(["month", "admin", "llin", "irs", "age"])
        clin_inc = clin_inc.loc[:, ["monthly_prob_clin"]]

        sev_inc = p["sev_inc"].set_index(["month", "admin", "llin", "irs", "age"])
        sev_inc = sev_inc.loc[:, ["monthly_prob_sev"]]

        all_inc = pd.concat([inf_inc, clin_inc, sev_inc], axis=1)
        # we don't want age to be part of index
        all_inc = all_inc.reset_index()

        all_inc['district_num'] = all_inc['admin'].map(mapper_district_name_to_num)
        assert not all_inc['district_num'].isna().any()

        self.all_inc = all_inc.drop(columns=['admin']).set_index(["month", "district_num", "llin", "irs"])

        # get the DALY weight that this module will use from the weight database
        if "HealthBurden" in self.sim.modules:
            p["daly_wt_clinical"] = self.sim.modules["HealthBurden"].get_daly_weight(218)
            p["daly_wt_severe"] = self.sim.modules["HealthBurden"].get_daly_weight(213)

        # ----------------------------------- DECLARE THE SYMPTOMS -------------------------------------------
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom("jaundice"),  # nb. will cause care seeking as much as a typical symptom
            Symptom("severe_anaemia"),  # nb. will cause care seeking as much as a typical symptom
            Symptom("acidosis", emergency_in_children=True, emergency_in_adults=True),
            Symptom("coma_convulsions", emergency_in_children=True, emergency_in_adults=True),
            Symptom("renal_failure", emergency_in_children=True, emergency_in_adults=True),
            Symptom("shock", emergency_in_children=True, emergency_in_adults=True)
        )

    def initialise_population(self, population):
        df = population.props

        # ----------------------------------- INITIALISE THE POPULATION-----------------------------------
        # Set default for properties
        df.loc[df.is_alive, "ma_is_infected"] = False
        df.loc[df.is_alive, "ma_date_infected"] = pd.NaT
        df.loc[df.is_alive, "ma_date_symptoms"] = pd.NaT
        df.loc[df.is_alive, "ma_date_death"] = pd.NaT
        df.loc[df.is_alive, "ma_tx"] = False
        df.loc[df.is_alive, "ma_date_tx"] = pd.NaT
        df.loc[df.is_alive, "ma_inf_type"] = "none"
        df.loc[df.is_alive, "ma_age_edited"] = 0.0

        df.loc[df.is_alive, "ma_clinical_counter"] = 0
        df.loc[df.is_alive, "ma_tx_counter"] = 0
        df.loc[df.is_alive, "ma_clinical_preg_counter"] = 0
        df.loc[df.is_alive, "ma_iptp"] = False

        self.malaria_poll2(population)

    def malaria_poll2(self, population):
        df = population.props
        p = self.parameters
        now = self.sim.date
        rng = self.rng

        # ----------------------------------- DISTRICT INTERVENTION COVERAGE -----------------------------------
        # fix values for 2018 onwards
        current_year = min(now.year, p["data_end"])

        # get itn_irs rows for current year; slice multiindex for all districts & current_year
        itn_irs_curr = self.itn_irs.loc[pd.IndexSlice[:, current_year], :]
        itn_irs_curr = itn_irs_curr.reset_index().drop("Year", axis=1)  # we don"t use the year column
        itn_irs_curr.insert(0, "month", now.month)  # add current month for the incidence index lookup

        # replace itn coverage with projected coverage levels from 2019 onwards
        if now.year > p["data_end"]:
            itn_irs_curr['itn_rate'] = self.parameters["itn"]

        month_districtnum_itn_irs_lookup = [
            tuple(r) for r in itn_irs_curr.values]  # every row is a key in incidence table

        # ----------------------------------- DISTRICT INCIDENCE ESTIMATES -----------------------------------
        # get all corresponding rows from the incidence table; drop unneeded column; set new index
        curr_inc = self.all_inc.loc[month_districtnum_itn_irs_lookup]
        curr_inc = curr_inc.reset_index().drop(["month", "llin", "irs"], axis=1).set_index(["district_num", "age"])

        # ----------------------------------- DISTRICT NEW INFECTIONS -----------------------------------
        def _draw_incidence_for(_col, _where):
            """a helper function to perform random draw for selected individuals on column of probabilities"""
            # create an index from the individuals to lookup entries in the current incidence table
            district_age_lookup = df[_where].set_index(["district_num_of_residence", "ma_age_edited"]).index
            # get the monthly incidence probabilities for these individuals
            monthly_prob = curr_inc.loc[district_age_lookup, _col]
            # update the index so it"s the same as the original population dataframe for these individuals
            monthly_prob = monthly_prob.set_axis(df.index[_where], inplace=False)
            # select individuals for infection
            random_draw = rng.random_sample(_where.sum()) < monthly_prob
            selected = _where & random_draw
            return selected

        # we don't have incidence data for over 80s
        alive = df.is_alive & (df.age_years < 80)

        alive_over_one = alive & (df.age_exact_years >= 1)
        df.loc[alive & df.age_exact_years.between(0, 0.5), "ma_age_edited"] = 0.0
        df.loc[alive & df.age_exact_years.between(0.5, 1), "ma_age_edited"] = 0.5
        df.loc[alive_over_one, "ma_age_edited"] = df.loc[alive_over_one, "age_years"].astype(float)

        # select new infections
        alive_uninfected = alive & ~df.ma_is_infected
        now_infected = _draw_incidence_for("monthly_prob_inf", alive_uninfected)
        df.loc[now_infected, "ma_is_infected"] = True
        df.loc[now_infected, "ma_date_infected"] = now  # TODO: scatter dates across month
        df.loc[now_infected, "ma_inf_type"] = "asym"

        # select all currently infected
        alive_infected = alive & df.ma_is_infected

        # draw from currently asymptomatic to allocate clinical cases
        alive_infected_asym = alive_infected & (df.ma_inf_type == "asym")
        now_clinical = _draw_incidence_for("monthly_prob_clin", alive_infected_asym)
        df.loc[now_clinical, "ma_inf_type"] = "clinical"
        df.loc[now_clinical, "ma_date_infected"] = now  # updated infection date
        df.loc[now_clinical, "ma_clinical_counter"] += 1

        # draw from clinical cases to allocate severe cases - draw from all currently clinical cases
        alive_infected_clinical = alive_infected & (df.ma_inf_type == "clinical")
        now_severe = _draw_incidence_for("monthly_prob_sev", alive_infected_clinical)
        df.loc[now_severe, "ma_inf_type"] = "severe"
        df.loc[now_severe, "ma_date_infected"] = now  # updated infection date

        alive_now_infected_pregnant = now_clinical & (df.ma_date_infected == now) & df.is_pregnant
        df.loc[alive_now_infected_pregnant, "ma_clinical_preg_counter"] += 1

        # ----------------------------------- CLINICAL MALARIA SYMPTOMS -----------------------------------
        # clinical - can't use now_clinical, because some clinical may have become severe
        clin = df.index[df.is_alive & (df.ma_inf_type == "clinical") & (df.ma_date_infected == now)]

        # update clinical symptoms for all new clinical infections
        self.clinical_symptoms(df, clin)
        # check symptom onset occurs in one week
        assert (df.loc[clin, "ma_date_infected"] < df.loc[clin, "ma_date_symptoms"]).all()

        # ----------------------------------- SEVERE MALARIA SYMPTOMS -----------------------------------

        # SEVERE CASES
        severe = df.is_alive & (df.ma_inf_type == "severe") & (df.ma_date_infected == now)

        children = severe & (df.age_exact_years < 5)
        adult = severe & (df.age_exact_years >= 5)

        # update symptoms for all new severe infections
        self.severe_symptoms(df, df.index[children], child=True)
        self.severe_symptoms(df, df.index[adult], child=False)
        # check symptom onset occurs in one week
        assert (df.loc[severe, "ma_date_infected"] < df.loc[severe, "ma_date_symptoms"]).all()

        # ----------------------------------- SCHEDULED DEATHS -----------------------------------
        # schedule deaths within the next week
        # Assign time of infections across the month

        # the cfr applies to all severe malaria
        random_draw = rng.random_sample(size=severe.sum())
        death = df.index[severe][random_draw < (p["cfr"] * p["mortality_adjust"])]

        for person in death:

            logger.debug(key='message',
                         data=f'MalariaEvent: scheduling malaria death for person {person}')

            # symptom onset occurs one week after infection
            # death occurs 1-7 days after symptom onset, 8+ days after infection
            random_date = rng.randint(low=8, high=14)
            random_days = pd.to_timedelta(random_date, unit="d")

            death_event = MalariaDeathEvent(
                self, individual_id=person, cause="Malaria"
            )  # make that death event
            self.sim.schedule_event(
                death_event, self.sim.date + random_days
            )  # schedule the death

    def initialise_simulation(self, sim):
        """
        * 1) Schedule the Main Regular Polling Events
        * 2) Define the DxTests
        * 3) Look-up and save the codes for consumables
        """

        # 1) ----------------------------------- REGULAR EVENTS -----------------------------------

        sim.schedule_event(MalariaPollingEventDistrict(self), sim.date + DateOffset(months=1))

        sim.schedule_event(MalariaScheduleTesting(self), sim.date + DateOffset(days=1))

        if 'CareOfWomenDuringPregnancy' not in self.sim.modules:
            sim.schedule_event(MalariaIPTp(self), sim.date + DateOffset(days=30.5))

        sim.schedule_event(MalariaCureEvent(self), sim.date + DateOffset(days=5))
        sim.schedule_event(MalariaParasiteClearanceEvent(self), sim.date + DateOffset(days=30.5))

        sim.schedule_event(MalariaResetCounterEvent(self), sim.date + DateOffset(days=365))  # 01 jan each year

        # add an event to log to screen - 31st Dec each year
        sim.schedule_event(MalariaLoggingEvent(self), sim.date + DateOffset(days=364))
        sim.schedule_event(MalariaTxLoggingEvent(self), sim.date + DateOffset(days=364))
        sim.schedule_event(MalariaPrevDistrictLoggingEvent(self), sim.date + DateOffset(days=30.5))

        # 2) ----------------------------------- DIAGNOSTIC TESTS -----------------------------------
        # Create the diagnostic test representing the use of RDT for malaria diagnosis
        # and registers it with the Diagnostic Test Manager

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            malaria_rdt=DxTest(
                property='ma_is_infected',
                item_codes=self.sim.modules['HealthSystem'].get_item_code_from_item_name("Malaria test kit (RDT)"),
                sensitivity=self.parameters['sensitivity_rdt'],
            )
        )

        # 3) ----------------------------------- CONSUMABLES -----------------------------------
        get_item_code = self.sim.modules['HealthSystem'].get_item_code_from_item_name

        # malaria treatment uncomplicated children <15kg
        self.item_codes_for_consumables_required['malaria_uncomplicated_young_children'] = {
            get_item_code("Malaria test kit (RDT)"): 1,
            get_item_code("Lumefantrine 120mg/Artemether 20mg,  30x18_540_CMST"): 1,
            get_item_code("Paracetamol syrup 120mg/5ml_0.0119047619047619_CMST"): 18
        }

        # malaria treatment uncomplicated children >15kg
        self.item_codes_for_consumables_required['malaria_uncomplicated_older_children'] = {
            get_item_code("Malaria test kit (RDT)"): 1,
            get_item_code("Lumefantrine 120mg/Artemether 20mg,  30x18_540_CMST"): 3,
            get_item_code("Paracetamol syrup 120mg/5ml_0.0119047619047619_CMST"): 18
        }

        # malaria treatment uncomplicated adults >36kg
        self.item_codes_for_consumables_required['malaria_uncomplicated_adult'] = {
            get_item_code("Malaria test kit (RDT)"): 1,
            get_item_code("Lumefantrine 120mg/Artemether 20mg,  30x18_540_CMST"): 4,
            get_item_code("Paracetamol 500mg_1000_CMST"): 18
        }

        # malaria treatment complicated - same consumables for adults and children
        self.item_codes_for_consumables_required['malaria_complicated'] = {
            get_item_code("Injectable artesunate"): 1,
            get_item_code("Cannula iv  (winged with injection pot) 18_each_CMST"): 3,
            get_item_code("Disposables gloves, powder free, 100 pieces per box"): 1,
            get_item_code("Gauze, absorbent 90cm x 40m_each_CMST"): 3,
            get_item_code("Water for injection, 10ml_Each_CMST"): 3,
        }

        # malaria IPTp for pregnant women
        self.item_codes_for_consumables_required['malaria_iptp'] = {
            get_item_code("Sulfamethoxazole + trimethropin, tablet 400 mg + 80 mg"): 6
        }

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props
        df.at[child_id, "ma_is_infected"] = False
        df.at[child_id, "ma_date_infected"] = pd.NaT
        df.at[child_id, "ma_date_symptoms"] = pd.NaT
        df.at[child_id, "ma_date_death"] = pd.NaT
        df.at[child_id, "ma_tx"] = False
        df.at[child_id, "ma_date_tx"] = pd.NaT
        df.at[child_id, "ma_inf_type"] = "none"
        df.at[child_id, "ma_age_edited"] = 0.0
        df.at[child_id, "ma_clinical_counter"] = 0
        df.at[child_id, "ma_clinical_preg_counter"] = 0
        df.at[child_id, "ma_tx_counter"] = 0
        df.at[child_id, "ma_iptp"] = False

    def on_hsi_alert(self, person_id, treatment_id):
        """This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        logger.debug(key='message',
                     data=f'This is Malaria, being alerted about a health system interaction for person'
                          f'{person_id} and treatment {treatment_id}')

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug(key='message',
                     data='This is malaria reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe
        p = self.parameters

        health_values = df.loc[df.is_alive, "ma_inf_type"].map(
            {
                "none": 0,
                "asym": 0,
                "clinical": p["daly_wt_clinical"],
                "severe": p["daly_wt_severe"],
            }
        )
        health_values.name = "Malaria"  # label the cause of this disability

        return health_values.loc[df.is_alive]  # returns the series

    def clinical_symptoms(self, population, clinical_index):
        """assign clinical symptoms to new clinical malaria cases and schedule symptom resolution

        :param population:
        :param clinical_index:
        """
        df = population
        p = self.parameters
        rng = self.rng
        now = self.sim.date

        date_symptom_onset = now + pd.DateOffset(days=7)

        df.loc[clinical_index, "ma_date_symptoms"] = date_symptom_onset

        symptom_list = {"fever", "headache", "vomiting", "stomachache"}

        # this also schedules symptom resolution in 5 days
        self.sim.modules["SymptomManager"].change_symptom(
            person_id=list(clinical_index),
            symptom_string=symptom_list,
            add_or_remove="+",
            disease_module=self,
            date_of_onset=date_symptom_onset,
            duration_in_days=None,  # remove duration as symptoms cleared by MalariaCureEvent
        )

        # additional risk of severe anaemia in pregnancy
        pregnant_infected = df.is_alive & (df.ma_inf_type == "clinical") & (df.ma_date_infected == now) & df.is_pregnant
        if pregnant_infected.sum() > 0:
            random_draw = rng.random_sample(size=pregnant_infected.sum())
            preg_infected = df.index[pregnant_infected][random_draw < p["p_sev_anaemia_preg"]]
            if len(preg_infected) > 0:
                self.sim.modules["SymptomManager"].change_symptom(
                    person_id=list(preg_infected),
                    symptom_string="severe_anaemia",
                    add_or_remove="+",
                    disease_module=self,
                    date_of_onset=date_symptom_onset,
                    duration_in_days=None,
                )

    def severe_symptoms(self, population, severe_index, child=False):
        """assign clinical symptoms to new severe malaria cases. Symptoms can only be resolved by treatment
        handles both adult and child (using the child parameter) symptoms

        :param population: the population dataframe
        :param severe_index: the indices of new clinical cases
        :param child: to apply severe symptoms to children (otherwise applied to adults)
        """
        # If no indices specified exit straight away
        if len(severe_index) == 0:
            return

        df = population
        p = self.parameters
        rng = self.rng
        date_symptom_onset = self.sim.date + pd.DateOffset(days=7)

        df.loc[severe_index, "ma_date_symptoms"] = date_symptom_onset

        # general symptoms - applied to all
        symptom_list = {"fever", "headache", "vomiting", "stomachache"}

        self.sim.modules["SymptomManager"].change_symptom(
            person_id=list(severe_index),
            symptom_string=symptom_list,
            add_or_remove="+",
            disease_module=self,
            date_of_onset=date_symptom_onset,
            duration_in_days=None,
        )

        # symptoms specific to severe cases
        # get range of probabilities of each symptom for severe cases for children and adults
        range_symp = p["sev_symp_prob"]

        if child:
            range_symp = range_symp.loc[range_symp.age_group == "0_5"]
        else:
            range_symp = range_symp.loc[range_symp.age_group == "5_60"]
        range_symp = range_symp.set_index("symptom")
        symptom_list_severe = list(range_symp.index)

        # assign symptoms

        for symptom in symptom_list_severe:
            # Let u ~ Uniform(0, 1) and p ~ Uniform(prop_lower, prop_upper),
            # then the probability of the event (u < p) is (prop_lower + prop_upper) / 2
            # That is the probability of b == True in the following code snippet
            #     b = rng.uniform() < rng.uniform(low=prop_lower, high=prop_upper)
            # and this one
            #     b = rng.uniform() < (prop_lower + prop_upper) / 2
            # are equivalent.
            persons_gaining_symptom = severe_index[
                rng.uniform(size=len(severe_index))
                < (
                    range_symp.at[symptom, "prop_lower"]
                    + range_symp.at[symptom, "prop_upper"]
                ) / 2
            ]
            # schedule symptom onset
            self.sim.modules["SymptomManager"].change_symptom(
                person_id=persons_gaining_symptom,
                symptom_string=symptom,
                add_or_remove="+",
                disease_module=self,
                duration_in_days=None,
            )

    def check_if_fever_is_caused_by_malaria(self, person_id, hsi_event):
        """Run by an HSI when an adult presents with fever"""

        # Call the DxTest RDT to diagnose malaria
        dx_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
            dx_tests_to_run='malaria_rdt',
            hsi_event=hsi_event
        )

        true_malaria_infection_type = self.sim.population.props.at[person_id, "ma_inf_type"]

        # severe malaria infection always returns positive RDT
        if true_malaria_infection_type == "severe":
            return "severe_malaria"

        elif dx_result and true_malaria_infection_type in ("clinical", "asym"):
            return "clinical_malaria"

        else:
            return "negative_malaria_test"


class MalariaPollingEventDistrict(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        logger.debug(key='message', data='MalariaEvent: tracking the disease progression of the population')
        self.module.malaria_poll2(population)


class MalariaScheduleTesting(RegularEvent, PopulationScopeEventMixin):
    """ additional malaria testing happening outside the symptom-driven generic HSI event
    to increase tx coverage up to reported levels
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        df = population.props
        now = self.sim.date
        p = self.module.parameters

        # select people to go for testing (and subsequent tx)
        # random sample 0.4 to match clinical case tx coverage
        # this sample will include asymptomatic infections too to account for
        # unnecessary treatments  and uninfected people
        alive = df.is_alive
        test = df.index[alive][self.module.rng.random_sample(size=alive.sum()) < p["testing_adj"]]

        for person_index in test:
            logger.debug(key='message',
                         data=f'MalariaScheduleTesting: scheduling HSI_Malaria_rdt for person {person_index}')
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Malaria_rdt(self.module, person_id=person_index),
                priority=1,
                topen=now, tclose=None
            )


# TODO link this with ANC appts
class MalariaIPTp(RegularEvent, PopulationScopeEventMixin):
    """ malaria prophylaxis for pregnant women
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        df = population.props
        now = self.sim.date

        # select currently pregnant women without IPTp, malaria-negative
        p1 = df.index[df.is_alive & df.is_pregnant & ~df.ma_is_infected & ~df.ma_iptp]

        for person_index in p1:
            logger.debug(key='message',
                         data=f'MalariaIPTp: scheduling HSI_Malaria_IPTp for person {person_index}')

            event = HSI_MalariaIPTp(self.module, person_id=person_index)
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                event, priority=1, topen=now, tclose=None
            )


class MalariaDeathEvent(Event, IndividualScopeEventMixin):
    """
    Performs the Death operation on an individual and logs it.
    """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)
        self.cause = cause

    def apply(self, individual_id):
        df = self.sim.population.props

        if not df.at[individual_id, "is_alive"]:
            return

        # death should only occur if severe malaria case
        assert df.at[individual_id, "ma_inf_type"] == "severe"

        # if on treatment, will reduce probability of death
        # use random number generator - currently param treatment_adjustment set to 0.5
        if df.at[individual_id, "ma_tx"]:
            prob = self.module.rng.rand()

            # if draw -> death
            if prob < self.module.parameters["treatment_adjustment"]:
                self.sim.modules['Demography'].do_death(
                    individual_id=individual_id, cause=self.cause, originating_module=self.module)

                df.at[individual_id, "ma_date_death"] = self.sim.date

            # else if draw does not result in death -> cure
            else:
                df.at[individual_id, "ma_tx"] = False
                df.at[individual_id, "ma_inf_type"] = "none"
                df.at[individual_id, "ma_is_infected"] = False

                # clear symptoms
                self.sim.modules["SymptomManager"].clear_symptoms(
                    person_id=individual_id, disease_module=self.module
                )

        # if not on treatment - death will occur
        else:
            self.sim.modules['Demography'].do_death(
                individual_id=individual_id, cause=self.cause, originating_module=self.module)

            df.at[individual_id, "ma_date_death"] = self.sim.date


# ---------------------------------------------------------------------------------
# Health System Interaction Events
# ---------------------------------------------------------------------------------


class HSI_Malaria_rdt(HSI_Event, IndividualScopeEventMixin):
    """
    this is a point-of-care malaria rapid diagnostic test, with results within 2 minutes
    """

    def __init__(self, module, person_id):

        super().__init__(module, person_id=person_id)
        assert isinstance(module, Malaria)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["ConWithDCSA"] = 1
        # print(the_appt_footprint)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Malaria_RDT"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = '0'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        params = self.module.parameters
        hs = self.sim.modules["HealthSystem"]

        # Ignore this event if the person is no longer alive:
        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        district = df.at[person_id, "district_num_of_residence"]
        logger.debug(key='message',
                     data=f'HSI_Malaria_rdt: rdt test for person {person_id} '
                          f'in district num {district}')

        # call the DxTest RDT to diagnose malaria
        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='malaria_rdt',
            hsi_event=self
        )

        if dx_result:

            # check if currently on treatment
            if not df.at[person_id, "ma_tx"]:

                # ----------------------------------- SEVERE MALARIA -----------------------------------

                # if severe malaria, treat for complicated malaria
                if df.at[person_id, "ma_inf_type"] == "severe":

                    # paediatric severe malaria case
                    if df.at[person_id, "age_years"] < 15:

                        logger.debug(key='message',
                                     data=f'HSI_Malaria_rdt: scheduling HSI_Malaria_tx_compl_child {person_id}'
                                          f'on date {self.sim.date}')

                        treat = HSI_Malaria_complicated_treatment_child(
                            self.sim.modules["Malaria"], person_id=person_id
                        )
                        self.sim.modules["HealthSystem"].schedule_hsi_event(
                            treat, priority=1, topen=self.sim.date, tclose=None
                        )

                    else:
                        # adult severe malaria case
                        logger.debug(key='message',
                                     data='HSI_Malaria_rdt: scheduling HSI_Malaria_tx_compl_adult for person '
                                          f'{person_id} on date {self.sim.date}')

                        treat = HSI_Malaria_complicated_treatment_adult(
                            self.module, person_id=person_id
                        )
                        self.sim.modules["HealthSystem"].schedule_hsi_event(
                            treat, priority=1, topen=self.sim.date, tclose=None
                        )

                # ----------------------------------- TREATMENT CLINICAL DISEASE -----------------------------------

                # clinical malaria - not severe
                elif df.at[person_id, "ma_inf_type"] == "clinical":

                    # diagnosis of clinical disease dependent on RDT sensitivity
                    diagnosed = self.sim.rng.choice(
                        [True, False],
                        size=1,
                        p=[params["sensitivity_rdt"], (1 - params["sensitivity_rdt"])],
                    )

                    # diagnosis / treatment for children <5
                    if diagnosed & (df.at[person_id, "age_years"] < 5):
                        logger.debug(key='message',
                                     data=f'HSI_Malaria_rdt scheduling HSI_Malaria_tx_0_5 for person {person_id}'
                                          f'on date {self.sim.date}')

                        treat = HSI_Malaria_non_complicated_treatment_age0_5(self.module, person_id=person_id)
                        self.sim.modules["HealthSystem"].schedule_hsi_event(
                            treat, priority=1, topen=self.sim.date, tclose=None
                        )

                    # diagnosis / treatment for children 5-15
                    if diagnosed & (df.at[person_id, "age_years"] >= 5) & (df.at[person_id, "age_years"] < 15):
                        logger.debug(key='message',
                                     data=f'HSI_Malaria_rdt: scheduling HSI_Malaria_tx_5_15 for person {person_id}'
                                          f'on date {self.sim.date}')

                        treat = HSI_Malaria_non_complicated_treatment_age5_15(self.module, person_id=person_id)
                        self.sim.modules["HealthSystem"].schedule_hsi_event(
                            treat, priority=1, topen=self.sim.date, tclose=None
                        )

                    # diagnosis / treatment for adults
                    if diagnosed & (df.at[person_id, "age_years"] >= 15):
                        logger.debug(key='message',
                                     data=f'HSI_Malaria_rdt: scheduling HSI_Malaria_tx_adult for person {person_id}'
                                          f'on date {self.sim.date}')

                        treat = HSI_Malaria_non_complicated_treatment_adult(self.module, person_id=person_id)
                        self.sim.modules["HealthSystem"].schedule_hsi_event(
                            treat, priority=1, topen=self.sim.date, tclose=None
                        )

    def did_not_run(self):
        logger.debug(key='message',
                     data='HSI_Malaria_rdt: did not run')
        pass


class HSI_Malaria_non_complicated_treatment_age0_5(HSI_Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for children <15 kg. Includes treatment plus one rdt
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Malaria)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["ConWithDCSA"] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Malaria_treatment_child0_5"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = '0'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        if not df.at[person_id, "ma_tx"]:

            logger.debug(key='message',
                         data=f'HSI_Malaria_tx_0_5: requesting malaria treatment for child {person_id}')

            if self.get_consumables(
                self.module.item_codes_for_consumables_required['malaria_uncomplicated_young_children']
            ):

                logger.debug(key='message',
                             data=f'HSI_Malaria_tx_0_5: giving malaria treatment for child {person_id}')

                if df.at[person_id, "is_alive"]:
                    df.at[person_id, "ma_tx"] = True
                    df.at[person_id, "ma_date_tx"] = self.sim.date
                    df.at[person_id, "ma_tx_counter"] += 1

    def did_not_run(self):
        logger.debug(key='message',
                     data='HSI_Malaria_tx_0_5: did not run')
        pass


class HSI_Malaria_non_complicated_treatment_age5_15(HSI_Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for children >15 kg. Includes treatment plus one rdt
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Malaria)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["Under5OPD"] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Malaria_treatment_child5_15"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        if not df.at[person_id, "ma_tx"]:

            logger.debug(key='message',
                         data=f'HSI_Malaria_tx_5_15: requesting malaria treatment for child {person_id}')

            if self.get_consumables(
                self.module.item_codes_for_consumables_required['malaria_uncomplicated_older_children']
            ):

                logger.debug(key='message',
                             data=f'HSI_Malaria_tx_5_15: giving malaria treatment for child {person_id}')

                if df.at[person_id, "is_alive"]:
                    df.at[person_id, "ma_tx"] = True
                    df.at[person_id, "ma_date_tx"] = self.sim.date
                    df.at[person_id, "ma_tx_counter"] += 1

    def did_not_run(self):
        logger.debug(key='message',
                     data='HSI_Malaria_tx_5_15: did not run')
        pass


class HSI_Malaria_non_complicated_treatment_adult(HSI_Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for adults. Includes treatment plus one rdt
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Malaria)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["Over5OPD"] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Malaria_treatment_adult"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        if not df.at[person_id, "ma_tx"] and df.at[person_id, "is_alive"]:

            logger.debug(key='message',
                         data=f'HSI_Malaria_tx_adult: requesting malaria treatment for person {person_id}')

            if self.get_consumables(self.module.item_codes_for_consumables_required['malaria_uncomplicated_adult']):
                logger.debug(key='message',
                             data=f'HSI_Malaria_tx_adult: giving malaria treatment for person {person_id}')

                df.at[person_id, "ma_tx"] = True
                df.at[person_id, "ma_date_tx"] = self.sim.date
                df.at[person_id, "ma_tx_counter"] += 1

    def did_not_run(self):
        logger.debug(key='message',
                     data='HSI_Malaria_tx_adult: did not run')
        pass


class HSI_Malaria_complicated_treatment_child(HSI_Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for complicated malaria in children
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Malaria)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["InpatientDays"] = 5

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Malaria_treatment_complicated_child"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = '1b'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        if not df.at[person_id, "ma_tx"] and df.at[person_id, "is_alive"]:

            logger.debug(key='message',
                         data=f'HSI_Malaria_tx_compl_child: requesting complicated malaria treatment for '
                              f'child {person_id}')

            if self.get_consumables(self.module.item_codes_for_consumables_required['malaria_complicated']):
                logger.debug(key='message',
                             data=f'HSI_Malaria_tx_compl_child: giving complicated malaria treatment for '
                                  f'child {person_id}')

                df.at[person_id, "ma_tx"] = True
                df.at[person_id, "ma_date_tx"] = self.sim.date
                df.at[person_id, "ma_tx_counter"] += 1

    def did_not_run(self):
        logger.debug(key='message',
                     data='HSI_Malaria_tx_compl_child: did not run')
        pass


class HSI_Malaria_complicated_treatment_adult(HSI_Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for complicated malaria in adults
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Malaria)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["InpatientDays"] = 5

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Malaria_treatment_complicated_adult"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = '1b'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        if not df.at[person_id, "ma_tx"] and df.at[person_id, "is_alive"]:

            logger.debug(key='message',
                         data=f'HSI_Malaria_tx_compl_adult: requesting complicated malaria treatment '
                              f'for person {person_id}')

            if self.get_consumables(self.module.item_codes_for_consumables_required['malaria_complicated']):
                logger.debug(key='message',
                             data=f'HSI_Malaria_tx_compl_adult: giving complicated malaria treatment '
                                  f'for person {person_id}')

                df.at[person_id, "ma_tx"] = True
                df.at[person_id, "ma_date_tx"] = self.sim.date
                df.at[person_id, "ma_tx_counter"] += 1

    def did_not_run(self):
        logger.debug(key='message',
                     data='HSI_Malaria_tx_compl_adult: did not run')
        pass


class HSI_MalariaIPTp(HSI_Event, IndividualScopeEventMixin):
    """
    this is IPTp for pregnant women
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Malaria)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["ANCSubsequent"] = 0.25  # This requires part of an ANC appt

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Malaria_IPTp"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props

        if not df.at[person_id, "is_alive"] or df.at[person_id, "ma_tx"]:
            return

        else:

            logger.debug(key='message',
                         data=f'HSI_MalariaIPTp: requesting IPTp for person {person_id}')

            # request the treatment
            if self.get_consumables(self.module.item_codes_for_consumables_required['malaria_iptp']):
                logger.debug(key='message',
                             data=f'HSI_MalariaIPTp: giving IPTp for person {person_id}')

                df.at[person_id, "ma_iptp"] = True

    def did_not_run(self):

        logger.debug(key='message',
                     data='HSI_MalariaIPTp: did not run')
        pass


# ---------------------------------------------------------------------------------
# Recovery Events
# ---------------------------------------------------------------------------------
class MalariaCureEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(days=3))

    def apply(self, population):
        """
        this is a regular event which cures people currently on treatment for malaria
        and clears symptoms for those not on treatment
        it also clears parasites if treated
        """

        logger.debug(key='message', data='MalariaCureEvent: symptom resolution for malaria cases')

        df = self.sim.population.props

        # TREATED
        # select people with malaria and treatment for at least 3 days
        # if treated, will clear symptoms and parasitaemia
        # this will also clear parasitaemia for asymptomatic cases picked up by routine rdt
        infected_and_treated = df.index[df.is_alive &
                                        (df.ma_date_tx < (self.sim.date - DateOffset(days=3))) &
                                        (df.ma_inf_type != "severe")]

        self.sim.modules["SymptomManager"].clear_symptoms(
            person_id=infected_and_treated, disease_module=self.module
        )

        # change properties
        df.loc[infected_and_treated, "ma_tx"] = False
        df.loc[infected_and_treated, "ma_is_infected"] = False
        df.loc[infected_and_treated, "ma_inf_type"] = "none"

        # UNTREATED
        # if not treated, self-cure occurs after 6 days of symptoms
        # but parasites remain in blood
        clinical_not_treated = df.index[df.is_alive &
                                        (df.ma_inf_type == "clinical") &
                                        (df.ma_date_infected < (self.sim.date - DateOffset(days=6))) &
                                        ~df.ma_tx]

        self.sim.modules["SymptomManager"].clear_symptoms(
            person_id=clinical_not_treated, disease_module=self.module
        )

        # change properties
        df.loc[clinical_not_treated, "ma_inf_type"] = "asym"


class MalariaParasiteClearanceEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(days=30.5))

    def apply(self, population):
        logger.debug(key='message', data='MalariaParasiteClearanceEvent: parasite clearance for malaria cases')

        df = self.sim.population.props
        p = self.module.parameters

        # select people infected at least 100 days ago
        asym_inf = df.index[df.is_alive &
                            (df.ma_inf_type == "asym") &
                            (df.ma_date_infected < (self.sim.date - DateOffset(days=p["dur_asym"])))]

        df.loc[asym_inf, "ma_inf_type"] = "none"
        df.loc[asym_inf, "ma_is_infected"] = False


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
        tmp = len(
            df.loc[(df.ma_date_symptoms > (now - DateOffset(months=self.repeat)))]
        )
        pop = len(df[df.is_alive])

        inc_1000py = (tmp / pop) * 1000

        # incidence rate clinical (inc severe) in 2-10 yr olds
        tmp2 = len(
            df.loc[
                (df.age_years.between(2, 10))
                & (df.ma_date_symptoms > (now - DateOffset(months=self.repeat)))
                ]
        )

        pop2_10 = len(df[df.is_alive & (df.age_years.between(2, 10))])
        inc_1000py_2_10 = (tmp2 / pop2_10) * 1000

        inc_1000py_hiv = 0  # if running without hiv/tb

        # using clinical counter
        # sum all the counters for previous year
        clin_episodes = df[
            "ma_clinical_counter"
        ].sum()  # clinical episodes (inc severe)
        inc_counter_1000py = (clin_episodes / pop) * 1000

        clin_preg_episodes = df[
            "ma_clinical_preg_counter"
        ].sum()  # clinical episodes in pregnant women (inc severe)

        summary = {
            "number_new_cases": tmp,
            "population": pop,
            "inc_1000py": inc_1000py,
            "inc_1000py_hiv": inc_1000py_hiv,
            "new_cases_2_10": tmp2,
            "population2_10": pop2_10,
            "inc_1000py_2_10": inc_1000py_2_10,
            "inc_clin_counter": inc_counter_1000py,
            "clinical_preg_counter": clin_preg_episodes,
        }

        logger.info(key='incidence',
                    data=summary,
                    description='Summary of incident malaria cases')

        # ------------------------------------ RUNNING COUNTS ------------------------------------

        counts = {"none": 0, "asym": 0, "clinical": 0, "severe": 0}
        counts.update(df.loc[df.is_alive, "ma_inf_type"].value_counts().to_dict())

        logger.info(key='status_counts',
                    data=counts,
                    description='Running counts of incident malaria cases')

        # ------------------------------------ PARASITE PREVALENCE BY AGE ------------------------------------

        # includes all parasite positive cases: some may have low parasitaemia (undetectable)
        child2_10_inf = len(
            df[df.is_alive & df.ma_is_infected & (df.age_years.between(2, 10))]
        )

        # population size - children
        child2_10_pop = len(df[df.is_alive & (df.age_years.between(2, 10))])

        # prevalence in children aged 2-10
        child_prev = child2_10_inf / child2_10_pop if child2_10_pop else 0

        # prevalence of clinical including severe in all ages
        total_clin = len(
            df[
                df.is_alive
                & ((df.ma_inf_type == "clinical") | (df.ma_inf_type == "severe"))
                ]
        )
        pop2 = len(df[df.is_alive])
        prev_clin = total_clin / pop2

        prev = {
            "child2_10_prev": child_prev,
            "clinical_prev": prev_clin,
        }

        logger.info(key='prevalence',
                    data=prev,
                    description='Prevalence malaria cases')


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
        tx = df["ma_tx_counter"].sum()  # treatment (inc severe)
        clin = df["ma_clinical_counter"].sum()  # clinical episodes (inc severe)

        tx_coverage = tx / clin if clin else 0

        treatment = {
            "number_treated": tx,
            "number_clinical episodes": clin,
            "treatment_coverage": tx_coverage,
        }

        logger.info(key='tx_coverage',
                    data=treatment,
                    description='Treatment of malaria cases')


class MalariaPrevDistrictLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        # ------------------------------------ PREVALENCE OF INFECTION ------------------------------------
        infected = (
            df[df.is_alive & df.ma_is_infected].groupby("district_num_of_residence").size()
        )
        pop = df[df.is_alive].groupby("district_num_of_residence").size()
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


# ---------------------------------------------------------------------------------
# Reset counters
# ---------------------------------------------------------------------------------
class MalariaResetCounterEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # reset all the counters to zero each year
        df = population.props
        now = self.sim.date

        logger.info(key='message',
                    data=f'Resetting the malaria counter {now}')

        df["ma_clinical_counter"] = 0
        df["ma_tx_counter"] = 0
        df["ma_clinical_preg_counter"] = 0
