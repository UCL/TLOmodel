from pathlib import Path

import numpy as np
import pandas as pd

from tlo import Date, DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO Vitamin A
class Epi(Module):
    """This is the expanded programme on immunisation module
    it sets up the vaccination schedule for all children from birth
    """

    INIT_DEPENDENCIES = {'Demography', 'HealthSystem'}

    # Declare Metadata
    METADATA = {Metadata.USES_HEALTHSYSTEM}

    PARAMETERS = {
        "baseline_coverage": Parameter(Types.DATA_FRAME, "baseline vaccination coverage (all vaccines)"),
        "vaccine_schedule": Parameter(Types.SERIES, "vaccination schedule applicable from 2018 onwards"),
        "prob_facility_level_for_vaccine": Parameter(Types.LIST,
                                                     "The probability of going to each facility-level (0 / 1a / 1b / 2)"
                                                     " for child having vaccines given through childhood immunisation "
                                                     "schedule. The probabilities must sum to 1.0 ")
    }

    PROPERTIES = {
        # -- Properties for the number of doses received of each vaccine --
        "va_bcg": Property(Types.INT, "number of doses of BCG vaccination"),
        "va_opv": Property(Types.INT, "number of doses of OPV vaccine received"),
        "va_dtp": Property(Types.INT, "number of doses of DTP vaccine received"),
        "va_hib": Property(Types.INT, "number of doses of Hib vaccine received"),
        "va_hep": Property(Types.INT, "number of doses of HepB vaccine received (infant series)"),
        "va_pneumo": Property(Types.INT, "number of doses of pneumococcal vaccine received"),
        "va_rota": Property(Types.INT, "number of doses of rotavirus vaccine received"),
        "va_measles": Property(Types.INT, "number of doses of measles vaccine received"),
        "va_rubella": Property(Types.INT, "number of doses of rubella vaccine received"),
        "va_hpv": Property(Types.INT, "number of doses of hpv vaccine received"),
        "va_td": Property(Types.INT, "number of doses of tetanus/diphtheria vaccine received by pregnant women"),

        # -- Properties to inidcate whether the full number of doses have been received --
        "va_bcg_all_doses": Property(Types.BOOL, "whether all doses have been received of the vaccine bcg"),
        "va_opv_all_doses": Property(Types.BOOL, "whether all doses have been received of the OPV vaccine"),
        "va_dtp_all_doses": Property(Types.BOOL, "whether all doses have been received of the DTP vaccine"),
        "va_hib_all_doses": Property(Types.BOOL, "whether all doses have been received of the Hib vaccine"),
        "va_hep_all_doses": Property(Types.BOOL,
                                     "whether all doses have been received of the HepB vaccine (infant series)"),
        "va_pneumo_all_doses": Property(Types.BOOL, "whether all doses have been received of the pneumococcal vaccine"),
        "va_rota_all_doses": Property(Types.BOOL, "whether all doses have been received of the rotavirus vaccine"),
        "va_measles_all_doses": Property(Types.BOOL, "whether all doses have been received of the measles vaccine"),
        "va_rubella_all_doses": Property(Types.BOOL, "whether all doses have been received of the rubella vaccine"),
        "va_hpv_all_doses": Property(Types.BOOL, "whether all doses have been received of the HPV vaccine"),
        "va_td_all_doses": Property(Types.BOOL,
                                    "whether all doses have been received of the tetanus/diphtheria vaccine"),
    }

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.all_doses = dict()
        self.cons_item_codes = dict()  # (will store dict giving item_codes for each vaccine)

    def read_parameters(self, data_folder):
        p = self.parameters
        workbook = pd.read_excel(
            Path(self.resourcefilepath) / 'ResourceFile_EPI_WHO_estimates.xlsx', sheet_name=None
        )

        self.load_parameters_from_dataframe(workbook["parameters"])

        p["baseline_coverage"] = workbook["WHO_estimates"]
        p["vaccine_schedule"] = workbook["vaccine_schedule"].set_index('vaccine')['date_administration_days']

        # Declare definitions of how many doses is labelled as "all doses"
        self.all_doses.update({
            'bcg': 1,
            'opv': 4,
            'dtp': 3,
            'hib': 3,
            'hep': 3,
            'pneumo': 3,
            'rota': 2,
            'measles': 2,
            'rubella': 2,
            'hpv': 1,
            'td': 2
        })

    def initialise_population(self, population):
        df = population.props
        p = self.parameters

        # Set default for properties
        df.loc[df.is_alive, [
            "va_bcg",
            "va_opv",
            "va_dtp",
            "va_hib",
            "va_hep",
            "va_pneumo",
            "va_rota",
            "va_measles",
            "va_rubella",
            "va_hpv",
            "va_td"]
        ] = 0

        df.loc[df.is_alive, [
            "va_bcg_all_doses",
            "va_opv_all_doses",
            "va_dtp_all_doses",
            "va_hib_all_doses",
            "va_hep_all_doses",
            "va_pneumo_all_doses",
            "va_rota_all_doses",
            "va_measles_all_doses",
            "va_rubella_all_doses",
            "va_hpv_all_doses",
            "va_td_all_doses"]
        ] = False

        # BCG
        # from 1981-2009 average bcg coverage is 92% (WHO estimates)
        # use vaccine coverage estimates for each year prior to 2010
        # assuming only <1 yr olds were vaccinated each year
        # match up vaccine coverage for each person based on their age
        # anyone over age 29 will not have matching vaccine coverage estimates
        # therefore no vaccinations for them
        df_vaccine_baseline = df.loc[df.is_alive, ['age_years']].reset_index().merge(
            p["baseline_coverage"],
            left_on=["age_years"],
            right_on=["AgeOn01Jan2010"],
            how="left"
        ).set_index('person')

        # use same random draw value for all vaccines - will induce correlations (good)
        # there are individuals who have high probability of getting all vaccines
        # some individuals will have consistently poor coverage
        random_draw = self.rng.random_sample(len(df_vaccine_baseline))
        df.loc[df.is_alive & (random_draw < df_vaccine_baseline["BCG"]), "va_bcg"] = 1

        # Polio OPV
        # from 1980-2009 average opv3 coverage is 79.5% (WHO estimates): all 3 doses OPV
        # assume no partial protection if < 3 doses (all-or-nothing response)
        df.loc[df.is_alive & (random_draw < df_vaccine_baseline["Pol3"]), "va_opv"] = 3

        # DTP3
        # available since 1980
        df.loc[df.is_alive & (random_draw < df_vaccine_baseline["DTP3"]), "va_dtp"] = 3

        # Hep3
        # available since 2002
        # by Jan 2010, anyone <9 years has 87.5% prob of having vaccine
        df.loc[df.is_alive & (random_draw < df_vaccine_baseline["HepB3"]), "va_hep"] = 3

        # Hib3
        # available since 2002
        # by Jan 2010, anyone <9 years has 87.5% prob of having vaccine
        df.loc[df.is_alive & (random_draw < df_vaccine_baseline["Hib3"]), "va_hib"] = 3

        # Measles
        # available since 1980
        # second dose only started in 2015
        # by Jan 2010, anyone <=30 years has 77.2% prob of having vaccine
        df.loc[df.is_alive & (random_draw < df_vaccine_baseline["MCV1"]), "va_measles"] = 1

        # update the 'all_doses' properties for initial population
        for vaccine, max_dose in self.all_doses.items():
            df.loc[df.is_alive, f"va_{vaccine}_all_doses"] = df.loc[df.is_alive, f"va_{vaccine}"] >= max_dose

    def initialise_simulation(self, sim):
        # add an event to log to screen
        sim.schedule_event(EpiLoggingEvent(self), sim.date + DateOffset(years=1))

        # HPV vaccine given from 2018 onwards
        sim.schedule_event(HpvScheduleEvent(self), Date(2018, 1, 1))

        # Look up item codes for consumables
        self.get_item_codes()

        # Check that the values enetered for 'prob_facility_level_for_vaccine' sum to 1.0
        probs = self.parameters['prob_facility_level_for_vaccine']
        assert all(np.isfinite(probs)) and np.isclose(sum(probs), 1.0)

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual

        birth doses occur within 24 hours of delivery

        2012 data is patchy, no record of Hep vaccine but it was used before 2012
        assume hepB3 coverage in 2012 same as 2011
        same with Hib

        Measles - first dose, only one dose pre-2016 and no rubella
        Measles, rubella - first dose, 2018 onwards

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """
        df = self.sim.population.props  # shortcut to the population props dataframe
        p = self.parameters

        # Initialise all the properties that this module looks after:
        df.loc[child_id, [
            "va_bcg",
            "va_opv",
            "va_dtp",
            "va_hib",
            "va_hep",
            "va_pneumo",
            "va_rota",
            "va_measles",
            "va_rubella",
            "va_hpv",
            "va_td"]
        ] = 0

        df.loc[child_id, [
            "va_bcg_all_doses",
            "va_opv_all_doses",
            "va_dtp_all_doses",
            "va_hib_all_doses",
            "va_hep_all_doses",
            "va_pneumo_all_doses",
            "va_rota_all_doses",
            "va_measles_all_doses",
            "va_rubella_all_doses",
            "va_hpv_all_doses",
            "va_td_all_doses"]
        ] = False

        # ----------------------------------- 2010-2018 -----------------------------------
        vax_date = p["vaccine_schedule"]

        # each entry is (hsi event class, [days to administration key 1, days to administration key 2, ...]
        vax_schedule = [
            # schedule bcg - now dependent on health system capacity / stocks
            (HSI_BcgVaccine, ['bcg']),
            # OPV doses 2-4 are given during the week 6, 10, 14 penta, pneumo, rota appts
            (HSI_opv, ['opv1', 'opv2', 'opv3', 'opvIpv4']),
            (HSI_PneumoVaccine, ['pneumo1', 'pneumo2', 'pneumo3']),
            (HSI_RotaVaccine, ['rota1', 'rota2']),
            (HSI_DtpHibHepVaccine, ['dtpHibHep1', 'dtpHibHep2', 'dtpHibHep3']),
            (HSI_MeaslesRubellaVaccine, ['MR1', 'MR2'])
        ]

        # choose facility level where child will receive all their childhood vaccinations
        # Determine the level at which care is sought
        facility_levels = ('0', '1a', '1b', '2')
        facility_level_for_vaccines = self.rng.choice(
            facility_levels,
            p=self.parameters['prob_facility_level_for_vaccine']
        )

        for each_vax in vax_schedule:
            vax_hsi_event, admin_schedule = each_vax
            for admin_key in admin_schedule:
                vax_event_instance = vax_hsi_event(self, person_id=child_id,
                                                   facility_level_of_this_hsi=facility_level_for_vaccines)
                scheduled_date = vax_date[admin_key]
                # Request the health system to have this vaccination appointment
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    vax_event_instance,
                    priority=1,
                    topen=self.sim.date + DateOffset(days=scheduled_date),
                    tclose=None
                )

    def report_daly_values(self):
        """ epi module returns dalys=0 for all persons alive """
        logger.debug(key="debug", data="This is epi reporting my health values")

        df = self.sim.population.props  # shortcut to population properties dataframe

        health_values = pd.Series(index=df.index[df.is_alive], data=0)
        return health_values  # returns the series

    def increment_dose(self, person_id, vaccine):
        df = self.sim.population.props
        df.at[person_id, f"va_{vaccine}"] += 1
        if df.at[person_id, f"va_{vaccine}"] >= self.all_doses[vaccine]:
            df.at[person_id, f"va_{vaccine}_all_doses"] = True

    def get_item_codes(self):
        """Look-up the item-codes for each vaccine and update `self.cons_item_codes`"""
        get_item_code_from_item_name = self.sim.modules['HealthSystem'].get_item_code_from_item_name

        self.cons_item_codes['bcg'] = get_item_code_from_item_name("Syringe, autodisposable, BCG, 0.1 ml, with needle")
        self.cons_item_codes['opv'] = get_item_code_from_item_name("Polio vaccine")
        self.cons_item_codes['pentavalent_vaccine'] = get_item_code_from_item_name(
            "Pentavalent vaccine (DPT, Hep B, Hib)")
        self.cons_item_codes["rota"] = get_item_code_from_item_name("Rotavirus vaccine")
        self.cons_item_codes["pneumo"] = get_item_code_from_item_name("Pneumococcal vaccine")
        self.cons_item_codes["measles_and_rubella"] = get_item_code_from_item_name("Measles vaccine")
        self.cons_item_codes["hpv"] = get_item_code_from_item_name("HPV vaccine")
        self.cons_item_codes['td'] = get_item_code_from_item_name("Tetanus toxoid, injection")
        self.cons_item_codes['syringes'] = [
            get_item_code_from_item_name("Syringe, Autodisable SoloShot IX "),
            get_item_code_from_item_name("Safety box for used syringes/needles, 5 liter")]


# ---------------------------------------------------------------------------------
# Schedule vaccines outside EPI
# ---------------------------------------------------------------------------------

class HpvScheduleEvent(RegularEvent, PopulationScopeEventMixin):
    """ HPV vaccine event - each year sample from 9 year old girls and schedule vaccine
    stagger vaccine administration across the year
    coverage estimates dependent on health system capacity
    average around 85% for 2018
    WHO recommends 2 doses
    schedule doses 1 month apart
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))

    def apply(self, population):
        logger.debug(key="debug", data="HpvScheduleEvent selecting eligible 9-yr olds for HPV vaccine")

        df = population.props
        now = self.sim.date

        #  sample using the prob_inf scaled by relative susceptibility
        hpv_vax = df.index[df.is_alive & (df.age_years == 9) & (df.sex == "F")]

        # scatter vaccination days across the year
        # todo: HPV vaccine may be offered on scheduled clinic days / weeks - check
        random_day = self.module.rng.randint(365, size=len(hpv_vax))

        scheduled_vax_dates = now + pd.to_timedelta(random_day, unit="d")

        for index, person_id in enumerate(hpv_vax):
            logger.debug(key="debug", data=f"HpvScheduleEvent scheduling HPV vaccine for {person_id}")

            # find the index in hpv_vax
            # then select that value from the scheduled_vax_date
            vax_date = scheduled_vax_dates[index]

            facility_levels = ('0', '1a', '1b', '2')
            facility_level_for_vaccines = self.module.rng.choice(
                facility_levels,
                p=self.module.parameters['prob_facility_level_for_vaccine']
            )

            # first dose
            event = HSI_HpvVaccine(self.module, person_id=person_id,
                                   facility_level_of_this_hsi=facility_level_for_vaccines)

            self.sim.modules["HealthSystem"].schedule_hsi_event(
                event,
                priority=2,
                topen=vax_date,
                tclose=vax_date + DateOffset(weeks=2),
            )

            # second dose
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                event,
                priority=2,
                topen=vax_date + DateOffset(weeks=4),
                tclose=vax_date + DateOffset(weeks=6),
            )


# ---------------------------------------------------------------------------------
# Health System Interaction Events
# ---------------------------------------------------------------------------------


class HsiBaseVaccine(HSI_Event, IndividualScopeEventMixin):
    """This is a base class for all vaccination HSI_Events. Handles initialisation and requesting consumables needed
    for the vaccination. For custom behaviour, you can override __init__ in subclasses and implemented your own
    constructors (or inherit directly from HSI_Event)
    unless specified, default facility level is 1a
    unless specified, footprint returned in 1 EPI appt
    if vaccine occurs as part of a treatment package within another appointment, use suppress_footprint=True
    """

    def __init__(self, module, person_id, facility_level_of_this_hsi="1a", suppress_footprint=False):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Epi)

        assert isinstance(suppress_footprint, bool)
        self.suppress_footprint = suppress_footprint

        self.TREATMENT_ID = self.treatment_id()
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi

    def treatment_id(self):
        """subclasses should implement this method to return the TREATMENT_ID"""
        raise NotImplementedError

    def apply(self, *args, **kwargs):
        """must be implemented by subclasses"""
        raise NotImplementedError

    @property
    def EXPECTED_APPT_FOOTPRINT(self):
        """Returns the EPI appointment footprint for this person according to the vaccine:
        * tetanus/diphtheria for pregnant women uses 1 EPI appt
        * childhood vaccines can occur in bundles at birth, weeks 6, 10 and 14
        * measles/rubella always given in one appt (`HSI_MeaslesRubellaVaccine`) in months 9, 15
        * hpv given for adolescents uses 1 EPI appt
        * if a vaccine is given at the same time as other vaccines, decrease the EPI footprint to 0.5
        """

        if self.suppress_footprint:
            return self.make_appt_footprint({})

        # these vaccines are always given jointly with other childhood vaccines.
        # NB. If p["vaccine_schedule"] changes, this would need to be updated.
        vaccine_bundle = ['Epi_Childhood_Bcg', 'Epi_Childhood_Opv', 'Epi_Childhood_DtpHibHep', 'Epi_Childhood_Rota',
                          'Epi_Childhood_Pneumo']

        # determine whether this HSI gives a vaccine as part of a vaccine bundle
        # if vaccine is in list of vaccine bundles, return EPI footprint 0.5
        # all other vaccines use 1 full EPI appt
        if self.treatment_id() in vaccine_bundle:
            return self.make_appt_footprint({"EPI": 0.5})
        else:
            return self.make_appt_footprint({"EPI": 1})


class HSI_BcgVaccine(HsiBaseVaccine):
    """gives bcg vaccine 24 hours after birth or as soon as possible afterwards"""

    def treatment_id(self):
        return "Epi_Childhood_Bcg"

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if df.at[person_id, "va_bcg"] < self.module.all_doses["bcg"]:
            if self.get_consumables(item_codes=self.module.cons_item_codes["bcg"],
                                    optional_item_codes=self.module.cons_item_codes["syringes"]):
                self.module.increment_dose(person_id, "bcg")


class HSI_opv(HsiBaseVaccine):
    """gives poliovirus vaccine 24 hours after birth, plus weeks 6, 10, 14 or as soon as possible afterwards"""

    def treatment_id(self):
        return "Epi_Childhood_Opv"

    def apply(self, person_id, squeeze_factor):
        if self.get_consumables(item_codes=self.module.cons_item_codes["opv"],
                                optional_item_codes=self.module.cons_item_codes["syringes"]):
            self.module.increment_dose(person_id, "opv")


class HSI_DtpHibHepVaccine(HsiBaseVaccine):
    """ gives DTP-Hib_HepB vaccine """

    def treatment_id(self):
        return "Epi_Childhood_DtpHibHep"

    def apply(self, person_id, squeeze_factor):
        if self.get_consumables(item_codes=self.module.cons_item_codes['pentavalent_vaccine'],
                                optional_item_codes=self.module.cons_item_codes["syringes"]):
            self.module.increment_dose(person_id, "dtp")
            self.module.increment_dose(person_id, "hib")
            self.module.increment_dose(person_id, "hep")


class HSI_RotaVaccine(HsiBaseVaccine):
    """ gives Rotavirus vaccine 6 and 10 weeks after birth """

    def treatment_id(self):
        return "Epi_Childhood_Rota"

    def apply(self, person_id, squeeze_factor):
        logger.debug(key="debug", data=f"HSI_RotaVaccine: requesting vaccines for {person_id}")
        # rotavirus - oral vaccine
        # only 2 doses rotavirus given (week 6 and 10)
        # available from 2012 onwards
        df = self.sim.population.props
        if df.at[person_id, "va_rota"] < self.module.all_doses["rota"]:
            if self.get_consumables(item_codes=self.module.cons_item_codes["rota"],
                                    optional_item_codes=self.module.cons_item_codes["syringes"]):
                self.module.increment_dose(person_id, "rota")


class HSI_PneumoVaccine(HsiBaseVaccine):
    """ gives Pneumococcal vaccine 6, 10 and 14 weeks after birth """

    def treatment_id(self):
        return "Epi_Childhood_Pneumo"

    def apply(self, person_id, squeeze_factor):
        if self.get_consumables(item_codes=self.module.cons_item_codes["pneumo"],
                                optional_item_codes=self.module.cons_item_codes["syringes"]):
            self.module.increment_dose(person_id, "pneumo")


class HSI_MeaslesRubellaVaccine(HsiBaseVaccine):
    """ administers measles+rubella vaccine """

    def treatment_id(self):
        return "Epi_Childhood_MeaslesRubella"

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        # give vaccine if first dose or if second dose and after 2016
        if (df.at[person_id, "va_measles"] == 0) or (
                (df.at[person_id, "va_measles"] == 1) and (self.sim.date.year >= 2016)):

            if self.get_consumables(item_codes=self.module.cons_item_codes["measles_and_rubella"],
                                    optional_item_codes=self.module.cons_item_codes["syringes"]):
                self.module.increment_dose(person_id, "measles")

                # rubella contained in vaccine from 2018
                if self.sim.date.year >= 2018:
                    self.module.increment_dose(person_id, "rubella")


class HSI_HpvVaccine(HsiBaseVaccine):
    """ gives HPV vaccine to 9 year old girls; recommended 2 doses (WHO) """

    def treatment_id(self):
        return "Epi_Adolescent_Hpv"

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if df.at[person_id, "va_hpv"] < self.module.all_doses["hpv"]:
            if self.get_consumables(item_codes=self.module.cons_item_codes["hpv"],
                                    optional_item_codes=self.module.cons_item_codes["syringes"]):
                self.module.increment_dose(person_id, "hpv")


class HSI_TdVaccine(HsiBaseVaccine):
    """ gives tetanus/diphtheria vaccine to pregnant women as part of routine antenatal care
    recommended 2+ doses (WHO)
    """

    def treatment_id(self):
        return "Epi_Pregnancy_Td"

    def apply(self, person_id, squeeze_factor):
        if self.get_consumables(item_codes=self.module.cons_item_codes["td"],
                                optional_item_codes=self.module.cons_item_codes["syringes"]):
            self.module.increment_dose(person_id, "td")


# ---------------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------------


class EpiLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """ output vaccine coverage every year """
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, Epi)

    def apply(self, population):
        df = population.props

        def get_coverage(condition, subset):
            total = sum(subset)
            has_condition = sum(condition & subset)
            coverage = has_condition / total * 100 if total else 0
            assert coverage <= 100
            return coverage

        under_ones = df.is_alive & (df.age_years <= 1)
        bcg_coverage = get_coverage(df.va_bcg, under_ones)

        # the eligible group for coverage estimates will be those from 14 weeks and older
        # younger infants won't have had the three-dose course yet
        from_14_weeks_to_one = df.is_alive & (df.age_years <= 1) & (df.age_exact_years >= 0.27)
        dtp3_coverage = get_coverage(df.va_dtp >= 3, from_14_weeks_to_one)
        opv3_coverage = get_coverage(df.va_opv >= 3, from_14_weeks_to_one)
        hib3_coverage = get_coverage(df.va_hib >= 3, from_14_weeks_to_one)
        hep3_coverage = get_coverage(df.va_hep >= 3, from_14_weeks_to_one)
        pneumo3_coverage = get_coverage(df.va_pneumo >= 3, from_14_weeks_to_one)
        rota_coverage = get_coverage(df.va_rota >= 2, from_14_weeks_to_one)

        # measles vaccination coverage in <2 year old children - 1 dose
        # first dose is at 9 months, second dose is 15 months
        # so check coverage in 15 month -2 year olds
        from_15_months_to_two = df.is_alive & (df.age_exact_years >= 1.25) & (df.age_years <= 2)
        measles_coverage = get_coverage(df.va_measles >= 1, from_15_months_to_two)
        measles2_coverage = get_coverage(df.va_measles >= 2, from_15_months_to_two)

        # rubella vaccination coverage in <2 year old children - 1 dose
        # first dose is at 9 months, second dose is 15 months
        rubella_coverage = get_coverage(df.va_rubella >= 1, from_15_months_to_two)

        # HPV vaccination coverage in adolescent girls - 1 dose
        # first dose is at 9 years
        girls_from_ten_to_twelve = df.is_alive & (df.sex == 'F') & (df.age_exact_years >= 10) & (df.age_years <= 12)
        hpv_coverage = get_coverage(df.va_hpv >= 1, girls_from_ten_to_twelve)

        logger.info(
            key="ep_vaccine_coverage",
            data={
                "epNumInfantsUnder1": sum(from_14_weeks_to_one),
                "epBcgCoverage": bcg_coverage,
                "epDtp3Coverage": dtp3_coverage,
                "epOpv3Coverage": opv3_coverage,
                "epHib3Coverage": hib3_coverage,
                "epHep3Coverage": hep3_coverage,
                "epPneumo3Coverage": pneumo3_coverage,
                "epRota2Coverage": rota_coverage,
                "epMeaslesCoverage": measles_coverage,
                "epMeasles2Coverage": measles2_coverage,
                "epRubellaCoverage": rubella_coverage,
                "epHpvCoverage": hpv_coverage
            }
        )
