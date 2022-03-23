from pathlib import Path

import pandas as pd

from tlo import Date, DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
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
        "district_vaccine_coverage": Parameter(Types.DATA_FRAME, "coverage of each vaccine type by year and district")
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

        p["baseline_coverage"] = workbook["WHO_estimates"]
        p["vaccine_schedule"] = workbook["vaccine_schedule"].set_index('vaccine')['date_administration_days']

        p["district_vaccine_coverage"] = pd.read_csv(
            Path(self.resourcefilepath) / "ResourceFile_EPI_vaccine_coverage.csv"
        )

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
        df.loc[df.is_alive & (random_draw < df_vaccine_baseline["MCV1"]), "va_measles"] = 3

        # update the 'all_doses' properties for initial population
        for vaccine, max_dose in self.all_doses.items():
            df.loc[df.is_alive, f"va_{vaccine}_all_doses"] = df.loc[df.is_alive, f"va_{vaccine}"] >= max_dose

    def initialise_simulation(self, sim):
        # add an event to log to screen
        sim.schedule_event(EpiLoggingEvent(self), sim.date + DateOffset(years=1))
        # HPV vaccine given from 2018 onwards
        sim.schedule_event(HpvScheduleEvent(self), Date(2018, 1, 1))

        # Update paramerer "district_vaccine_coverage" to use district_num rather than the name of the district;
        self.parameters["district_vaccine_coverage"]["District"] = \
            self.parameters["district_vaccine_coverage"]["District"].map(
                {v: k for k, v in self.sim.modules['Demography'].parameters['district_num_to_district_name'].items()}
            )

        # Look up item codes for consumables
        self.get_item_codes()

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual

        birth doses occur within 24 hours of delivery

        from 2010-2018 data on vaccine coverage are used to determine probability of receiving vaccine
        vaccinations are scheduled to occur with a probability dependent on the year and district
        from 2019 onwards, probability will be determined by personnel and vaccine availability

        2012 data is patchy, no record of Hep vaccine but it was used before 2012
        assume hepB3 coverage in 2012 same as 2011
        same with Hib

        for births up to end 2018 schedule the vaccine as individual event (not HSI)
        schedule the dates as the exact due date
        then from 2019 use the HSI events - only need the current vaccines in use that way

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """
        df = self.sim.population.props  # shortcut to the population props dataframe
        p = self.parameters
        year = self.sim.date.year

        # look up coverage of every vaccine
        # anything delivered after 12months needs the estimate from the following year
        district = df.at[child_id, 'district_num_of_residence']

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

        # from 2010-2018 use the reported vaccine coverage values and schedule individual events (not HSI)
        # no consumables will be tracked up to end-2018
        if year <= 2018:

            # lookup the correct table of vaccine estimates for this child
            vax_coverage = p["district_vaccine_coverage"]
            ind_vax_coverage = vax_coverage.loc[(vax_coverage.District == district) & (vax_coverage.Year == year)]
            assert not ind_vax_coverage.empty

            # schedule bcg birth dose according to current coverage
            # some values are >1
            # each entry is (coverage prob key, event class, [days_to_1_administration, days_to_2_administration, etc])
            pre_2018_vax_schedule = [
                ('BCG', BcgVaccineEvent, ['bcg']),
                # assign OPV first dose according to current coverage
                # OPV doses 2-4 are given during the week 6, 10, 14 penta, pneumo, rota appts
                # coverage estimates for 3 doses reported, use these for doses 2-4
                ('OPV3', OpvEvent, ['opv1']),
                ('OPV3', OpvEvent, ['opv2', 'opv3', 'opvIpv4']),

                # DTP1_HepB - up to and including 2012, then replaced by pentavalent vaccine
                ('DTP1', DtpHepVaccineEvent, ['dtpHibHep1']),

                # DTP2_HepB - up to and including 2012
                # second doses not reported - same coverage for second and third doses
                ('DTP3', DtpHepVaccineEvent, ['dtpHibHep2', 'dtpHibHep3']),

                ('Hib3', HibVaccineEvent, ['dtpHibHep1', 'dtpHibHep2', 'dtpHibHep3']),

                # PNEUMO - all three doses reported separately
                ('Pneumo1', PneumococcalVaccineEvent, ['pneumo1']),
                ('Pneumo2', PneumococcalVaccineEvent, ['pneumo2']),
                ('Pneumo3', PneumococcalVaccineEvent, ['pneumo3']),

                # ROTA - doses 1 and 2 reported separately
                ('Rotavirus1', RotavirusVaccineEvent, ['rota1']),
                ('Rotavirus2', RotavirusVaccineEvent, ['rota2']),

                # PENTA1
                ('DTPHepHib1', DtpHibHepVaccineEvent, ['dtpHibHep1']),
                # PENTA2 - second dose not reported so use 3 dose coverage
                ('DTPHepHib3', DtpHibHepVaccineEvent, ['dtpHibHep2', 'dtpHibHep3']),

                # Measles, rubella - first dose, 2018 onwards
                ('MCV1_MR1', MeaslesRubellaVaccineEvent, ['MR1']),
                ('MCV2_MR2', MeaslesRubellaVaccineEvent, ['MR2']),

                # Measles - first dose, only one dose pre-2017 and no rubella
                ('MCV1', MeaslesVaccineEvent, ['MR1'])
            ]

            for each_vax in pre_2018_vax_schedule:
                coverage_key, vax_event, admin_schedule = each_vax
                if self.rng.random_sample() < ind_vax_coverage[coverage_key].values:
                    vax_event_instance = vax_event(self, person_id=child_id)
                    for admin_key in admin_schedule:
                        days_to_admin = vax_date[admin_key]
                        self.sim.schedule_event(vax_event_instance, self.sim.date + DateOffset(days=days_to_admin))

        # ----------------------------------- 2019 onwards -----------------------------------
        else:

            # after 2018
            # each entry is (hsi event class, [days to administration key 1, days to administration key 2, ...]
            post_2018_vax_schedule = [
                # schedule bcg - now dependent on health system capacity / stocks
                (HSI_BcgVaccine, ['bcg']),
                # OPV doses 2-4 are given during the week 6, 10, 14 penta, pneumo, rota appts
                (HSI_opv, ['opv1', 'opv2', 'opv3', 'opvIpv4']),
                (HSI_PneumoVaccine, ['pneumo1', 'pneumo2', 'pneumo3']),
                (HSI_RotaVaccine, ['rota1', 'rota2']),
                (HSI_DtpHibHepVaccine, ['dtpHibHep1', 'dtpHibHep2', 'dtpHibHep3']),
                (HSI_MeaslesRubellaVaccine, ['MR1', 'MR2'])
            ]

            for each_vax in post_2018_vax_schedule:
                vax_hsi_event, admin_schedule = each_vax
                for admin_key in admin_schedule:
                    vax_event_instance = vax_hsi_event(self, person_id=child_id)
                    scheduled_date = vax_date[admin_key]
                    # Request the health system to have this vaccination appointment
                    self.sim.modules['HealthSystem'].schedule_hsi_event(
                        vax_event_instance,
                        priority=1,
                        topen=self.sim.date + DateOffset(days=scheduled_date),
                        tclose=None
                    )

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        pass
        # TODO: consider here how early interventions are bundled
        # TODO: routine infant check-ups may occur alongside vaccinations

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
        get_item_codes_from_package_name = self.sim.modules['HealthSystem'].get_item_codes_from_package_name
        get_item_code_from_item_name = self.sim.modules['HealthSystem'].get_item_code_from_item_name

        self.cons_item_codes['bcg'] = [
            get_item_code_from_item_name("Syringe, autodisposable, BCG, 0.1 ml, with needle"),
            get_item_code_from_item_name("Safety box for used syringes/needles, 5 liter")]
        self.cons_item_codes['opv'] = get_item_code_from_item_name("Polio vaccine")
        self.cons_item_codes['pentavalent_vaccine'] = [
                get_item_code_from_item_name("Pentavalent vaccine (DPT, Hep B, Hib)"),
                get_item_code_from_item_name("Syringe, Autodisable SoloShot IX "),
                get_item_code_from_item_name("Safety box for used syringes/needles, 5 liter")]
        self.cons_item_codes["rota"] = get_item_code_from_item_name("Rotavirus vaccine")
        self.cons_item_codes["pneumo"] = [
                get_item_code_from_item_name("Pneumococcal vaccine"),
                get_item_code_from_item_name("Syringe, Autodisable SoloShot IX "),
                get_item_code_from_item_name("Safety box for used syringes/needles, 5 liter")]
        self.cons_item_codes['measles_and_rubella'] = get_item_codes_from_package_name("Measles rubella vaccine")
        self.cons_item_codes['hpv'] = get_item_codes_from_package_name("HPV vaccine")
        self.cons_item_codes['td'] = get_item_codes_from_package_name("Tetanus toxoid (pregnant women)")


# ---------------------------------------------------------------------------------
# Individually Scheduled Vaccine Events
# ---------------------------------------------------------------------------------


# BCG
class BcgVaccineEvent(Event, IndividualScopeEventMixin):
    """ give BCG vaccine at birth """
    def apply(self, person_id):
        self.module.increment_dose(person_id, "bcg")


class OpvEvent(Event, IndividualScopeEventMixin):
    """ give oral poliovirus vaccine (OPV) """
    def apply(self, person_id):
        self.module.increment_dose(person_id, "opv")


class DtpHepVaccineEvent(Event, IndividualScopeEventMixin):
    """ give DTP_Hep vaccine """
    def apply(self, person_id):
        self.module.increment_dose(person_id, "dtp")
        self.module.increment_dose(person_id, "hep")


class DtpHibHepVaccineEvent(Event, IndividualScopeEventMixin):
    """ give DTP_Hib_Hep vaccine """
    def apply(self, person_id):
        self.module.increment_dose(person_id, "dtp")
        self.module.increment_dose(person_id, "hep")
        self.module.increment_dose(person_id, "hib")


class RotavirusVaccineEvent(Event, IndividualScopeEventMixin):
    """ give Rotavirus vaccine """
    def apply(self, person_id):
        self.module.increment_dose(person_id, "rota")


class PneumococcalVaccineEvent(Event, IndividualScopeEventMixin):
    """ give Pneumococcal vaccine (PCV) """
    def apply(self, person_id):
        self.module.increment_dose(person_id, "pneumo")


class HibVaccineEvent(Event, IndividualScopeEventMixin):
    """ give Haemophilus influenza B vaccine """
    def apply(self, person_id):
        self.module.increment_dose(person_id, "hib")


class MeaslesVaccineEvent(Event, IndividualScopeEventMixin):
    """ give measles vaccine """
    def apply(self, person_id):
        self.module.increment_dose(person_id, "measles")


class MeaslesRubellaVaccineEvent(Event, IndividualScopeEventMixin):
    """ give measles/rubella vaccine """
    def apply(self, person_id):
        self.module.increment_dose(person_id, "measles")
        self.module.increment_dose(person_id, "rubella")


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

            # first dose
            event = HSI_HpvVaccine(self.module, person_id=person_id)

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
# TODO: note syringe disposal units will accommodate ~100 syringes
# request a box with each vaccine but don't need to condition HSI on availability
# likely always safety boxes available
# could request 1/100 of a box with each vaccine

class HsiBaseVaccine(HSI_Event, IndividualScopeEventMixin):
    """This is a base class for all vaccination HSI_Events. Handles initialisation and requesting consumables needed
    for the vaccination. For custom behaviour, you can override __init__ in subclasses and implemented your own
    constructors (or inherit directly from HSI_Event)"""
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Epi)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = self.treatment_id()
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"ConWithDCSA": 1})
        self.ACCEPTED_FACILITY_LEVEL = '0'
        self.ALERT_OTHER_DISEASES = []

    def treatment_id(self):
        """subclasses should implement this method to return the TREATMENT_ID"""
        raise NotImplementedError

    def apply(self, *args, **kwargs):
        """must be implemented by subclasses"""
        raise NotImplementedError

    def did_not_run(self):
        logger.debug(key="debug", data=f"{self.__class__.__name__}: did not run")


class HSI_BcgVaccine(HsiBaseVaccine):
    """gives bcg vaccine 24 hours after birth or as soon as possible afterwards"""
    def treatment_id(self):
        return "Epi_bcg"

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if df.at[person_id, "va_bcg"] < self.module.all_doses["bcg"]:
            if self.get_consumables(item_codes=self.module.cons_item_codes["bcg"]):
                self.module.increment_dose(person_id, "bcg")


class HSI_opv(HsiBaseVaccine):
    """gives poliovirus vaccine 24 hours after birth, plus weeks 6, 10, 14 or as soon as possible afterwards"""
    def treatment_id(self):
        return "Epi_opv"

    def apply(self, person_id, squeeze_factor):
        if self.get_consumables(item_codes=self.module.cons_item_codes["opv"]):
            self.module.increment_dose(person_id, "opv")


class HSI_DtpHibHepVaccine(HsiBaseVaccine):
    """ gives DTP-Hib_HepB vaccine """
    def treatment_id(self):
        return "Epi_DtpHibHep"

    def apply(self, person_id, squeeze_factor):
        if self.get_consumables(item_codes=self.module.cons_item_codes['pentavalent_vaccine']):
            self.module.increment_dose(person_id, "dtp")
            self.module.increment_dose(person_id, "hib")
            self.module.increment_dose(person_id, "hep")


class HSI_RotaVaccine(HsiBaseVaccine):
    """ gives Rotavirus vaccine 6 and 10 weeks after birth """
    def treatment_id(self):
        return "Epi_Rota"

    def apply(self, person_id, squeeze_factor):
        logger.debug(key="debug", data=f"HSI_RotaVaccine: requesting vaccines for {person_id}")
        # rotavirus - oral vaccine
        # only 2 doses rotavirus given (week 6 and 10)
        # available from 2012 onwards
        df = self.sim.population.props
        if df.at[person_id, "va_rota"] < self.module.all_doses["rota"]:
            if self.get_consumables(item_codes=self.module.cons_item_codes["rota"]):
                self.module.increment_dose(person_id, "rota")


class HSI_PneumoVaccine(HsiBaseVaccine):
    """ gives Pneumococcal vaccine 6, 10 and 14 weeks after birth """
    def treatment_id(self):
        return "Epi_Pneumo"

    def apply(self, person_id, squeeze_factor):
        if self.get_consumables(item_codes=self.module.cons_item_codes["pneumo"]):
            self.module.increment_dose(person_id, "pneumo")


class HSI_MeaslesRubellaVaccine(HsiBaseVaccine):
    """ administers measles+rubella vaccine """
    def treatment_id(self):
        return "Epi_MeaslesRubella"

    def apply(self, person_id, squeeze_factor):
        if self.get_consumables(item_codes=self.module.cons_item_codes["measles_and_rubella"]):
            self.module.increment_dose(person_id, "measles")
            self.module.increment_dose(person_id, "rubella")


class HSI_HpvVaccine(HsiBaseVaccine):
    """ gives HPV vaccine to 9 year old girls; recommended 2 doses (WHO) """
    def treatment_id(self):
        return "Epi_hpv"

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if df.at[person_id, "va_hpv"] < self.module.all_doses["hpv"]:
            if self.get_consumables(item_codes=self.module.cons_item_codes["hpv"]):
                self.module.increment_dose(person_id, "hpv")


# TODO this will be called by the antenatal care module as part of routine care: currently not implemented
class HSI_TdVaccine(HsiBaseVaccine):
    """ gives tetanus/diphtheria vaccine to pregnant women as part of routine antenatal care
    recommended 2+ doses (WHO)
    """
    def treatment_id(self):
        return "Epi_Td"

    def apply(self, person_id, squeeze_factor):
        if self.get_consumables(item_codes=self.module.cons_item_codes["td"]):
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
                "epRubellaCoverage": rubella_coverage,
                "epHpvCoverage": hpv_coverage
            }
        )
