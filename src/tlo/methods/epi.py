import os
from pathlib import Path

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO Vitamin A
class Epi(Module):
    """
    This is the expanded programme on immunisation module
    it sets up the vaccination schedule for all children from birth
    """

    PARAMETERS = {
        "baseline_coverage": Parameter(
            Types.DATA_FRAME, "baseline vaccination coverage (all vaccines)"
        ),
        "vaccine_schedule": Parameter(
            Types.SERIES, "vaccination schedule applicable from 2018 onwards"
        ),

        "district_vaccine_coverage": Parameter(Types.DATA_FRAME, "coverage of each vaccine type by year and district")
        }

    PROPERTIES = {
        "ep_bcg": Property(Types.BOOL, "received bcg vaccination"),
        "ep_opv": Property(Types.INT, "number of doses of OPV vaccine received"),
        "ep_dtp": Property(Types.INT, "number of doses of DTP vaccine received"),
        "ep_hib": Property(Types.INT, "number of doses of Hib vaccine received"),
        "ep_hep": Property(Types.INT, "number of doses of HepB vaccine received"),
        "ep_pneumo": Property(
            Types.INT, "number of doses of pneumococcal vaccine received"),
        "ep_rota": Property(Types.INT, "number of doses of rotavirus vaccine received"),
        "ep_measles": Property(Types.INT, "number of doses of measles vaccine received"),
        "ep_rubella": Property(Types.INT, "number of doses of rubella vaccine received"),
        "ep_hpv": Property(Types.INT, "number of doses of hpv vaccine received"),
        "ep_td": Property(Types.INT, "number of doses of tetanus/diphtheria vaccine received by pregnant women"),
    }

    # Declaration of the symptoms that this module will use
    SYMPTOMS = {}

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):

        p = self.parameters

        workbook = pd.read_excel(os.path.join(self.resourcefilepath,
                                              'ResourceFile_EPI_WHO_estimates.xlsx'), sheet_name=None)

        p["baseline_coverage"] = workbook["WHO_estimates"]
        p["vaccine_schedule"] = workbook["vaccine_schedule"].set_index('vaccine')['date_administration_days']

        p["district_vaccine_coverage"] = pd.read_csv(
            Path(self.resourcefilepath) / "ResourceFile_EPI_vaccine_coverage.csv"
        )

        # ---- Register this module ----
        # Register this disease module with the health system
        self.sim.modules["HealthSystem"].register_disease_module(self)

    def initialise_population(self, population):

        df = population.props
        p = self.parameters

        # Set default for properties
        df.loc[df.is_alive, "ep_bcg"] = False
        df.loc[df.is_alive, "ep_opv"] = 0
        df.loc[df.is_alive, "ep_dtp"] = 0
        df.loc[df.is_alive, "ep_hib"] = 0
        df.loc[df.is_alive, "ep_hep"] = 0
        df.loc[df.is_alive, "ep_pneumo"] = 0
        df.loc[df.is_alive, "ep_rota"] = 0
        df.loc[df.is_alive, "ep_measles"] = 0
        df.loc[df.is_alive, "ep_rubella"] = 0
        df.loc[df.is_alive, "ep_hpv"] = 0
        df.loc[df.is_alive, "ep_td"] = 0

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

        # what happens with the nan values in the df for vaccine coverage (age >30)??
        # seems fine!!
        # will have a susceptible older population though
        # use same random draw value for all vaccines - will induce correlations (good)
        # there are individuals who have high probability of getting all vaccines
        # some individuals will have consistently poor coverage
        random_draw = self.rng.random_sample(len(df_vaccine_baseline))
        df.loc[df.is_alive & (random_draw < df_vaccine_baseline["BCG"]), "ep_bcg"] = True

        # Polio OPV
        # from 1980-2009 average opv3 coverage is 79.5% (WHO estimates): all 3 doses OPV
        # assume no partial protection if < 3 doses (all-or-nothing response)
        df.loc[df.is_alive & (random_draw < df_vaccine_baseline["Pol3"]), "ep_opv"] = 3

        # DTP3
        # available since 1980
        df.loc[df.is_alive & (random_draw < df_vaccine_baseline["DTP3"]), "ep_dtp"] = 3

        # Hep3
        # available since 2002
        # by Jan 2010, anyone <9 years has 87.5% prob of having vaccine
        df.loc[df.is_alive & (random_draw < df_vaccine_baseline["HepB3"]), "ep_hep"] = 3

        # Hib3
        # available since 2002
        # by Jan 2010, anyone <9 years has 87.5% prob of having vaccine
        df.loc[df.is_alive & (random_draw < df_vaccine_baseline["Hib3"]), "ep_hib"] = 3

        # Measles
        # available since 1980
        # second dose only started in 2015
        # by Jan 2010, anyone <=30 years has 77.2% prob of having vaccine
        df.loc[df.is_alive & (random_draw < df_vaccine_baseline["MCV1"]), "ep_measles"] = 3

    def initialise_simulation(self, sim):
        # add an event to log to screen
        sim.schedule_event(EpiLoggingEvent(self), sim.date + DateOffset(years=1))
        # HPV vaccine given from 2018 onwards
        sim.schedule_event(HpvScheduleEvent(self), pd.to_datetime("2018/01/01", format="%Y/%m/%d"))

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
        district = df.at[child_id, 'district_of_residence']

        # Initialise all the properties that this module looks after:
        df.at[child_id, "ep_bcg"] = False
        df.at[child_id, "ep_opv"] = 0
        df.at[child_id, "ep_dtp"] = 0
        df.at[child_id, "ep_hib"] = 0
        df.at[child_id, "ep_hep"] = 0
        df.at[child_id, "ep_pneumo"] = 0
        df.at[child_id, "ep_rota"] = 0
        df.at[child_id, "ep_measles"] = 0
        df.at[child_id, "ep_rubella"] = 0
        df.at[child_id, "ep_hpv"] = 0
        df.at[child_id, "ep_td"] = 0

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
        # TODO: consider here how early interventions are bundled
        # TODO: routine infant check-ups may occur alongside vaccinations

    def report_daly_values(self):
        """ epi module returns dalys=0 for all persons alive """
        logger.debug(key="debug", data="This is epi reporting my health values")

        df = self.sim.population.props  # shortcut to population properties dataframe

        health_values = pd.Series(index=df.index[df.is_alive], data=0)
        return health_values  # returns the series


# ---------------------------------------------------------------------------------
# Individually Scheduled Vaccine Events
# ---------------------------------------------------------------------------------

# BCG
class BcgVaccineEvent(Event, IndividualScopeEventMixin):
    """ give BCG vaccine at birth """
    def apply(self, person_id):
        df = self.sim.population.props
        df.at[person_id, "ep_bcg"] = True


class OpvEvent(Event, IndividualScopeEventMixin):
    """ give oral poliovirus vaccine (OPV) """
    def apply(self, person_id):
        df = self.sim.population.props
        df.at[person_id, "ep_opv"] += 1


class DtpHepVaccineEvent(Event, IndividualScopeEventMixin):
    """ give DTP_Hep vaccine """
    def apply(self, person_id):
        df = self.sim.population.props
        df.at[person_id, "ep_dtp"] += 1
        df.at[person_id, "ep_hep"] += 1


class DtpHibHepVaccineEvent(Event, IndividualScopeEventMixin):
    """ give DTP_Hib_Hep vaccine """
    def apply(self, person_id):
        df = self.sim.population.props
        df.at[person_id, "ep_dtp"] += 1
        df.at[person_id, "ep_hep"] += 1
        df.at[person_id, "ep_hib"] += 1


class RotavirusVaccineEvent(Event, IndividualScopeEventMixin):
    """ give Rotavirus vaccine """
    def apply(self, person_id):
        df = self.sim.population.props
        df.at[person_id, "ep_rota"] += 1


class PneumococcalVaccineEvent(Event, IndividualScopeEventMixin):
    """ give Pneumococcal vaccine (PCV) """
    def apply(self, person_id):
        df = self.sim.population.props
        df.at[person_id, "ep_pneumo"] += 1


class HibVaccineEvent(Event, IndividualScopeEventMixin):
    """ give Haemophilus influenza B vaccine """
    def apply(self, person_id):
        df = self.sim.population.props
        df.at[person_id, "ep_hib"] += 1


class MeaslesVaccineEvent(Event, IndividualScopeEventMixin):
    """ give measles vaccine """
    def apply(self, person_id):
        df = self.sim.population.props
        df.at[person_id, "ep_measles"] += 1


class MeaslesRubellaVaccineEvent(Event, IndividualScopeEventMixin):
    """ give measles/rubella vaccine """
    def apply(self, person_id):
        df = self.sim.population.props
        df.at[person_id, "ep_measles"] += 1
        df.at[person_id, "ep_rubella"] += 1


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
        random_day = self.module.rng.choice(
            range(0, 365),
            size=len(hpv_vax),
            replace=True,
            p=[(1 / 365)] * 365,
        )

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

class HSI_BcgVaccine(HSI_Event, IndividualScopeEventMixin):
    """
    gives bcg vaccine 24 hours after birth or as soon as possible afterwards
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Epi)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["ConWithDCSA"] = 1  # This requires one ConWithDCSA appt

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Epi_bcg"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(key="debug", data=f"HSI_bcg: giving bcg to {person_id}")

        df = self.sim.population.props

        if not df.at[person_id, "ep_bcg"]:

            # Make request for some consumables
            consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

            item_code1 = pd.unique(
                consumables.loc[consumables["Items"] == "BCG vaccine", "Item_Code"]
            )[0]

            item_code2 = pd.unique(
                consumables.loc[
                    consumables["Items"]
                    == "Syringe, autodisposable, BCG, 0.1 ml, with needle",
                    "Item_Code",
                ]
            )[0]

            item_code3 = pd.unique(
                consumables.loc[
                    consumables["Items"]
                    == "Safety box for used syringes/needles, 5 liter",
                    "Item_Code",
                ]
            )[0]

            # assume 100 needles can be disposed of in each safety box
            consumables_needed = {
                "Intervention_Package_Code": {},
                "Item_Code": {item_code1: 1, item_code2: 1, item_code3: 1},
            }

            outcome_of_request_for_consumables = self.sim.modules[
                "HealthSystem"
            ].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed
            )

            if outcome_of_request_for_consumables:
                df.at[person_id, "ep_bcg"] = True

    def did_not_run(self):
        logger.debug(key="debug", data="HSI_bcg: did not run")


class HSI_opv(HSI_Event, IndividualScopeEventMixin):
    """
    gives poliovirus vaccine 24 hours after birth, plus weeks 6, 10, 14 or as soon as possible afterwards
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Epi)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["ConWithDCSA"] = 1  # This requires one ConWithDCSA appt

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Epi_opv"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(key="debug", data=f"HSI_opv: giving opv to {person_id}")

        df = self.sim.population.props

        # Make request for some consumables
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        item_code1 = pd.unique(
            consumables.loc[consumables["Items"] == "Polio vaccine", "Item_Code"]
        )[0]

        consumables_needed = {
            "Intervention_Package_Code": {},
            "Item_Code": {item_code1: 1},
        }

        outcome_of_request_for_consumables = self.sim.modules[
            "HealthSystem"
        ].request_consumables(hsi_event=self, cons_req_as_footprint=consumables_needed)

        if outcome_of_request_for_consumables:
            df.at[person_id, "ep_opv"] += 1

    def did_not_run(self):
        logger.debug(key="debug", data="HSI_opv: did not run")


class HSI_DtpHibHepVaccine(HSI_Event, IndividualScopeEventMixin):
    """
    gives DTP-Hib_HepB vaccine
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Epi)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["ConWithDCSA"] = 1  # This requires one ConWithDCSA appt

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Epi_DtpHibHep"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(key="debug", data=f"HSI_DtpHibHepVaccine: requesting vaccines for {person_id}")

        df = self.sim.population.props

        # Make request for some consumables
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        penta_vax = pd.unique(
            consumables.loc[
                consumables["Items"] == "Pentavalent vaccine (DPT, Hep B, Hib)",
                "Item_Code",
            ]
        )[0]

        syringe = pd.unique(
            consumables.loc[
                consumables["Items"] == "Syringe, needle + swab", "Item_Code",
            ]
        )[0]

        disposal = pd.unique(
            consumables.loc[
                consumables["Items"] == "Safety box for used syringes/needles, 5 liter",
                "Item_Code",
            ]
        )[0]

        # assume 100 needles can be disposed of in each safety box
        consumables_needed = {
            "Intervention_Package_Code": {},
            "Item_Code": {
                penta_vax: 1,
                syringe: 2,
                disposal: 1,
            },
        }

        outcome_of_request_for_consumables = self.sim.modules[
            "HealthSystem"
        ].request_consumables(hsi_event=self, cons_req_as_footprint=consumables_needed)

        # check if Penta and syringes available
        if (
            outcome_of_request_for_consumables["Item_Code"][penta_vax]
            & outcome_of_request_for_consumables["Item_Code"][syringe]
        ):
            logger.debug(key="debug", data=f"Penta vax is available, so administer to {person_id}")

            df.at[person_id, "ep_dtp"] += 1
            df.at[person_id, "ep_hib"] += 1
            df.at[person_id, "ep_hep"] += 1
        else:
            logger.debug(key="debug", data=f"Penta vax is not available for person {person_id}")

    def did_not_run(self):
        logger.debug(key="debug", data="HSI_DtpHibHepVaccine: did not run")


class HSI_RotaVaccine(HSI_Event, IndividualScopeEventMixin):
    """
    gives Rotavirus vaccine 6 and 10 weeks after birth
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Epi)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["ConWithDCSA"] = 1  # This requires one ConWithDCSA appt

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Epi_Rota"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(key="debug", data=f"HSI_RotaVaccine: requesting vaccines for {person_id}")

        df = self.sim.population.props

        # rotavirus - oral vaccine
        # only 2 doses rotavirus given (week 6 and 10)
        # available from 2012 onwards
        if df.at[person_id, "ep_rota"] < 2:

            # Make request for some consumables
            consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

            rotavirus_vax = pd.unique(
                consumables.loc[
                    consumables["Items"] == "Rotavirus vaccine", "Item_Code",
                ]
            )[0]

            consumables_needed = {
                "Intervention_Package_Code": {},
                "Item_Code": {rotavirus_vax: 1},
            }

            # check if rotavirus vaccine available
            outcome_of_request_for_consumables = self.sim.modules[
                "HealthSystem"
            ].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed
            )

            if outcome_of_request_for_consumables["Item_Code"][rotavirus_vax]:
                logger.debug(key="debug", data=f"Rotavirus vaccine is available, so administer to {person_id}")

                df.at[person_id, "ep_rota"] += 1
            else:
                logger.debug(key="debug", data=f"Rotavirus vaccine is not available for person {person_id}")

    def did_not_run(self):
        logger.debug(key="debug", data="HSI_RotaVaccine: did not run")


class HSI_PneumoVaccine(HSI_Event, IndividualScopeEventMixin):
    """
    gives Pneumococcal vaccine 6, 10 and 14 weeks after birth
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Epi)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["ConWithDCSA"] = 1  # This requires one ConWithDCSA appt

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Epi_Pneumo"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(key="debug", data=f"HSI_PneumoVaccine: requesting vaccines for {person_id}")

        df = self.sim.population.props

        # Make request for some consumables
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        # pneumococcal vaccine
        pneumo_vax = pd.unique(
            consumables.loc[
                consumables["Items"] == "Pneumococcal vaccine", "Item_Code",
            ]
        )[0]

        syringe = pd.unique(
            consumables.loc[
                consumables["Items"] == "Syringe, needle + swab", "Item_Code",
            ]
        )[0]

        disposal = pd.unique(
            consumables.loc[
                consumables["Items"] == "Safety box for used syringes/needles, 5 liter",
                "Item_Code",
            ]
        )[0]

        # assume 100 needles can be disposed of in each safety box
        consumables_needed = {
            "Intervention_Package_Code": {},
            "Item_Code": {
                pneumo_vax: 1,
                syringe: 2,
                disposal: 1,
            },
        }

        outcome_of_request_for_consumables = self.sim.modules[
            "HealthSystem"
        ].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

        # check if pneumococcal vaccine available and current year 2012 onwards
        if outcome_of_request_for_consumables:
            logger.debug(key="debug", data=f"Pneumococcal vaccine is available, so administer to {person_id}")

            df.at[person_id, "ep_pneumo"] += 1

    def did_not_run(self):
        logger.debug(key="debug", data="HSI_PneumoVaccine: did not run")


class HSI_MeaslesRubellaVaccine(HSI_Event, IndividualScopeEventMixin):
    """
    administers measles+rubella vaccine
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Epi)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["ConWithDCSA"] = 1  # This requires one ConWithDCSA appt

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Epi_MeaslesRubella"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(key="debug",
                     data=f"HSI_MeaslesRubellaVaccine: checking measles+rubella vaccine availability for {person_id}")

        df = self.sim.population.props

        # Make request for some consumables
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        pkg_code1 = pd.unique(
            consumables.loc[
                consumables["Intervention_Pkg"] == "Measles rubella vaccine",
                "Intervention_Pkg_Code",
            ]
        )[0]

        consumables_needed = {
            "Intervention_Package_Code": {pkg_code1: 1},
            "Item_Code": {},
        }

        outcome_of_request_for_consumables = self.sim.modules[
            "HealthSystem"
        ].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

        if outcome_of_request_for_consumables:
            logger.debug(key="debug",
                         data=f"HSI_MeaslesRubellaVaccine: measles+rubella vaccine is available for {person_id}")

            df.at[person_id, "ep_measles"] += 1
            df.at[person_id, "ep_rubella"] += 1
        else:
            logger.debug(key="debug",
                         data=f"HSI_MeaslesRubellaVaccine: measles+rubella vaccine is NOT available for {person_id}")

    def did_not_run(self):
        logger.debug(key="debug", data="HSI_MeaslesRubellaVaccine: did not run")


class HSI_HpvVaccine(HSI_Event, IndividualScopeEventMixin):
    """
    gives HPV vaccine to 9 year old girls
    recommended 2 doses (WHO)
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Epi)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["ConWithDCSA"] = 1  # This requires one ConWithDCSA appt

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Epi_hpv"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(key="debug", data=f"HSI_HpvVaccine: giving hpv vaccine to {person_id}")

        df = self.sim.population.props

        if df.at[person_id, "ep_hpv"] < 2:

            # Make request for some consumables
            consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

            pkg_code1 = pd.unique(
                consumables.loc[
                    consumables["Intervention_Pkg"] == "HPV vaccine",
                    "Intervention_Pkg_Code",
                ]
            )[0]

            # assume 100 needles can be disposed of in each safety box
            consumables_needed = {
                "Intervention_Package_Code": {pkg_code1: 1},
                "Item_Code": {},
            }

            outcome_of_request_for_consumables = self.sim.modules[
                "HealthSystem"
            ].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed
            )

            if outcome_of_request_for_consumables:
                df.at[person_id, "ep_hpv"] += 1

    def did_not_run(self):
        logger.debug(key="debug", data="HSI_HpvVaccine: did not run")


# TODO this will be called by the antenatal care module as part of routine care
# currently not implemented
class HSI_TdVaccine(HSI_Event, IndividualScopeEventMixin):
    """
    gives tetanus/diphtheria vaccine to pregnant women as part of routine antenatal care
    recommended 2+ doses (WHO)
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Epi)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["ConWithDCSA"] = 1  # This requires one ConWithDCSA appt

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Epi_Td"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(key="debug", data=f"HSI_TdVaccine: giving Td vaccine to {person_id}")

        df = self.sim.population.props

        # Make request for some consumables
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        # note this vaccine may not contain diphtheria - it should!
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables["Intervention_Pkg"] == "Tetanus toxoid (pregnant women)",
                "Intervention_Pkg_Code",
            ]
        )[0]

        consumables_needed = {
            "Intervention_Package_Code": {pkg_code1: 1},
            "Item_Code": {},
        }

        outcome_of_request_for_consumables = self.sim.modules[
            "HealthSystem"
        ].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

        if outcome_of_request_for_consumables:
            df.at[person_id, "ep_td"] += 1

    def did_not_run(self):
        logger.debug(key="debug", data="HSI_TdVaccine: did not run")

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

        def get_coverage(condition, total):
            has_condition = sum(condition)
            coverage = has_condition / total * 100 if total else 0
            assert coverage <= 100
            return coverage

        under_ones = df.is_alive & (df.age_years <= 1)
        from_14_weeks_to_one = df.is_alive & (df.age_years <= 1) & (df.age_exact_years >= 0.27)
        from_15_months_to_two = df.is_alive & (df.age_exact_years >= 1.25) & (df.age_years <= 2)
        girls_from_ten_to_twelve = df.is_alive & (df.sex == 'F') & (df.age_exact_years >= 10) & (df.age_years <= 12)

        bcg_coverage = get_coverage(under_ones & df.ep_bcg, sum(under_ones))

        # the eligible group for coverage estimates will be those from 14 weeks and older
        # younger infants won't have had the three-dose course yet
        # infants_denom = sum(from_14_weeks_to_one)
        toddler_count = sum(from_14_weeks_to_one)
        dtp3_coverage = get_coverage(from_14_weeks_to_one & (df.ep_dtp >= 3), toddler_count)
        opv3_coverage = get_coverage(from_14_weeks_to_one & (df.ep_opv >= 3), toddler_count)
        hib3_coverage = get_coverage(from_14_weeks_to_one & (df.ep_hib >= 3), toddler_count)
        hep3_coverage = get_coverage(from_14_weeks_to_one & (df.ep_hep >= 3), toddler_count)
        pneumo3_coverage = get_coverage(from_14_weeks_to_one & (df.ep_pneumo >= 3), toddler_count)
        rota_coverage = get_coverage(from_14_weeks_to_one & (df.ep_rota >= 2), toddler_count)

        # measles vaccination coverage in <2 year old children - 1 dose
        # first dose is at 9 months, second dose is 15 months
        # so check coverage in 15 month -2 year olds
        measles_coverage = get_coverage(from_15_months_to_two & (df.ep_measles >= 1), sum(from_15_months_to_two))

        # rubella vaccination coverage in <2 year old children - 1 dose
        # first dose is at 9 months, second dose is 15 months
        rubella_coverage = get_coverage(from_15_months_to_two & (df.ep_rubella >= 1), sum(from_15_months_to_two))

        # HPV vaccination coverage in adolescent girls - 1 dose
        # first dose is at 9 years
        hpv_coverage = get_coverage(girls_from_ten_to_twelve & (df.ep_hpv >= 1), sum(girls_from_ten_to_twelve))

        logger.info(
            key="ep_vaccine_coverage",
            data={
                "epNumInfantsUnder1": toddler_count,
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
