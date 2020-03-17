import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import (
    Event,
    IndividualScopeEventMixin,
    PopulationScopeEventMixin,
    RegularEvent,
)
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Epi(Module):
    """
    This is the expanded programme on immunisation module
    it sets up the vaccination schedule for all children from birth
    """

    PARAMETERS = {
        "bcg_coverage": Parameter(Types.REAL, "dummy vax coverage value"),
        "opv1_coverage": Parameter(Types.REAL, "dummy vax coverage value"),
        "opv2_coverage": Parameter(Types.REAL, "dummy vax coverage value"),
        "opv3_coverage": Parameter(Types.REAL, "dummy vax coverage value"),
        "opv4_coverage": Parameter(Types.REAL, "dummy vax coverage value"),
        "penta1_coverage": Parameter(Types.REAL, "dummy vax coverage value"),
        "penta2_coverage": Parameter(Types.REAL, "dummy vax coverage value"),
        "penta3_coverage": Parameter(Types.REAL, "dummy vax coverage value"),
        "measles_rubella1_coverage": Parameter(Types.REAL, "dummy vax coverage value"),
        "measles_rubella2_coverage": Parameter(Types.REAL, "dummy vax coverage value"),

        # baseline vaccination coverage for general population
        "opv3_baseline_coverage": Parameter(
            Types.REAL, "baseline opv3 vaccination coverage for population"
        ),
        "bcg_baseline_coverage": Parameter(
            Types.REAL, "baseline bcg vaccination coverage for population"
        ),
        "dtp3_baseline_coverage": Parameter(
            Types.REAL, "baseline dtp3 vaccination coverage for population"
        ),
        "HepB3_baseline_coverage": Parameter(
            Types.REAL, "baseline HepB3 vaccination coverage for population"
        ),
        "Hib3_baseline_coverage": Parameter(
            Types.REAL, "baseline Hib3 vaccination coverage for population"
        ),
        "measles_baseline_coverage": Parameter(
            Types.REAL, "baseline measles vaccination coverage for population"
        ),
        "rubella_baseline_coverage": Parameter(
            Types.REAL, "baseline rubella vaccination coverage for population"
        ),
    }

    PROPERTIES = {
        "ep_bcg": Property(Types.BOOL, "received bcg vaccination"),
        "ep_opv": Property(Types.INT, "number of doses of OPV vaccine received"),
        "ep_dtp": Property(Types.INT, "number of doses of DTP vaccine received"),
        "ep_hib": Property(Types.INT, "number of doses of Hib vaccine received"),
        "ep_hep": Property(Types.INT, "number of doses of HepB vaccine received"),
        "ep_pneumo": Property(
            Types.INT, "number of doses of pneumococcal vaccine received"
        ),
        "ep_rota": Property(Types.INT, "number of doses of rotavirus vaccine received"),
        "ep_measles": Property(Types.INT, "number of doses of measles vaccine received"),
        "ep_rubella": Property(Types.INT, "number of doses of rubella vaccine received"),
    }

    # Declaration of the symptoms that this module will use
    SYMPTOMS = {}

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):

        # baseline coverage = average coverage estimates from 1980 (opv) or 1981 (bcg)
        self.parameters["opv3_baseline_coverage"] = 0.795
        self.parameters["bcg_baseline_coverage"] = 0.92
        self.parameters["dtp3_baseline_coverage"] = 0.807
        self.parameters["hep3_baseline_coverage"] = 0.875
        self.parameters["hib3_baseline_coverage"] = 0.875
        self.parameters["measles_baseline_coverage"] = 0.772

        # district-level coverage estimates from 2010 to go here
        self.parameters["bcg_coverage"] = 1
        self.parameters["opv1_coverage"] = 1
        self.parameters["opv2_coverage"] = 1
        self.parameters["opv3_coverage"] = 1
        self.parameters["opv4_coverage"] = 1
        self.parameters["penta1_coverage"] = 1
        self.parameters["penta2_coverage"] = 1
        self.parameters["penta3_coverage"] = 1
        self.parameters["measles_rubella1_coverage"] = 1
        self.parameters["measles_rubella2_coverage"] = 1


        # ---- Register this module ----
        # Register this disease module with the health system
        self.sim.modules["HealthSystem"].register_disease_module(self)

    def initialise_population(self, population):

        df = population.props
        p = self.parameters

        # Set default for properties
        df.at[df.is_alive, "ep_bcg"] = False
        df.at[df.is_alive, "ep_opv"] = 0
        df.at[df.is_alive, "ep_dtp"] = 0
        df.at[df.is_alive, "ep_hib"] = 0
        df.at[df.is_alive, "ep_hep"] = 0
        df.at[df.is_alive, "ep_pneumo"] = 0
        df.at[df.is_alive, "ep_rota"] = 0
        df.at[df.is_alive, "ep_measles"] = 0
        df.at[df.is_alive, "ep_rubella"] = 0

        # BCG
        # from 1981-2009 average bcg coverage is 92% (WHO estimates)
        # by Jan 2010, anyone <30 years has 92% probability of being vaccinated
        # assuming only <1 yr olds were vaccinated each year
        random_draw = self.rng.random_sample(size=len(df))
        bcg_idx = df.index[
            df.is_alive
            & (df.age_years <= 29)
            & (random_draw < p["bcg_baseline_coverage"])
        ]

        df.at[bcg_idx, "ep_bcg"] = True

        # Polio OPV
        # from 1980-2009 average opv3 coverage is 79.5% (WHO estimates): all 3 doses OPV
        # assume no partial protection if < 3 doses (all-or-nothing response)
        # by Jan 2010, anyone <31 years has 79.5% probability of being vaccinated
        # assuming only <1 yr olds were vaccinated each year
        random_draw = self.rng.random_sample(size=len(df))
        opv3_idx = df.index[
            df.is_alive
            & (df.age_years <= 30)
            & (random_draw < p["opv3_baseline_coverage"])
        ]

        df.at[opv3_idx, "ep_opv"] = 3

        # DTP3
        # available since 1980
        random_draw = self.rng.random_sample(size=len(df))
        dtp3_idx = df.index[
            df.is_alive
            & (df.age_years <= 30)
            & (random_draw < p["dtp3_baseline_coverage"])
        ]

        df.at[dtp3_idx, "ep_dtp"] = 3

        # Hep3
        # available since 2002
        # by Jan 2010, anyone <9 years has 87.5% prob of having vaccine
        random_draw = self.rng.random_sample(size=len(df))
        hep3_idx = df.index[
            df.is_alive
            & (df.age_years <= 8)
            & (random_draw < p["hep3_baseline_coverage"])
        ]

        df.at[hep3_idx, "ep_hep"] = 3

        # Hib3
        # available since 2002
        # by Jan 2010, anyone <9 years has 87.5% prob of having vaccine
        random_draw = self.rng.random_sample(size=len(df))
        hib3_idx = df.index[
            df.is_alive
            & (df.age_years <= 8)
            & (random_draw < p["hib3_baseline_coverage"])
        ]

        df.at[hib3_idx, "ep_hib"] = 3

        # Measles
        # available since 1980
        # second dose only started in 2015
        # by Jan 2010, anyone <=30 years has 77.2% prob of having vaccine
        random_draw = self.rng.random_sample(size=len(df))
        measles_idx = df.index[
            df.is_alive
            & (df.age_years <= 30)
            & (random_draw < p["measles_baseline_coverage"])
        ]

        df.at[measles_idx, "ep_measles"] = 3

    def initialise_simulation(self, sim):

        # add an event to log to screen
        sim.schedule_event(EpiLoggingEvent(self), sim.date + DateOffset(months=1))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual

        all vaccinations are scheduled to occur with a probability dependent on the year and district
        birth doses occur within 24 hours of delivery

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """
        df = self.sim.population.props  # shortcut to the population props dataframe

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

        # assign bcg according to current coverage
        # TODO use current coverage estimates by district
        if self.rng.random_sample(size=1) < self.parameters["bcg_coverage"]:
            bcg_appt = HSI_bcg(self, person_id=child_id)

            # Request the health system to have this bcg vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                bcg_appt,
                priority=1,
                topen=self.sim.date + DateOffset(days=1),
                tclose=None,
            )

        # assign OPV first dose according to current coverage
        # OPV doses 2-4 are given during the week 6, 10, 14 penta, pneumo, rota appts
        # TODO use current coverage estimates by district
        if self.rng.random_sample(size=1) < self.parameters["opv1_coverage"]:
            opv1_appt = HSI_opv1(self, person_id=child_id)

            # Request the health system to have this bcg vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                opv1_appt,
                priority=1,
                topen=self.sim.date + DateOffset(days=1),
                tclose=None,
            )

        # PENTA1, OPV2, PNEUMO1, ROTA1
        # TODO use current coverage estimates by district
        if self.rng.random_sample(size=1) < self.parameters["penta1_coverage"]:
            penta1_appt = HSI_DtpHibHepVaccine(self, person_id=child_id)

            # Request the health system to have this bcg vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                penta1_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=6),
                tclose=None,
            )

        # PENTA2, OPV3, PNEUMO2, ROTA2
        # TODO use current coverage estimates by district
        if self.rng.random_sample(size=1) < self.parameters["penta2_coverage"]:
            penta2_appt = HSI_DtpHibHepVaccine(self, person_id=child_id)

            # Request the health system to have this bcg vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                penta2_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=10),
                tclose=None,
            )

        # PENTA3, OPV4, PNEUMO3
        # TODO use current coverage estimates by district
        if self.rng.random_sample(size=1) < self.parameters["penta3_coverage"]:
            penta3_appt = HSI_DtpHibHepVaccine(self, person_id=child_id)

            # Request the health system to have this bcg vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                penta3_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=14),
                tclose=None,
            )

        # Measles, rubella - first dose
        # TODO use current coverage estimates by district
        if self.rng.random_sample(size=1) < self.parameters["measles_rubella1_coverage"]:
            mr_appt = HSI_MeaslesRubellaVaccine(self, person_id=child_id)

            # Request the health system to have this bcg vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                mr_appt,
                priority=1,
                topen=self.sim.date + DateOffset(months=9),
                tclose=None,
            )

        # Measles, rubella - second dose
        # TODO use current coverage estimates by district
        if self.rng.random_sample(size=1) < self.parameters["measles_rubella2_coverage"]:
            mr_appt = HSI_MeaslesRubellaVaccine(self, person_id=child_id)

            # Request the health system to have this bcg vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                mr_appt,
                priority=1,
                topen=self.sim.date + DateOffset(months=15),
                tclose=None,
            )

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        # TODO: consider here how early interventions are bundled
        # TODO: routine infant check-ups may occur alongside vaccinations

    def report_daly_values(self):
        """ epi module returns dalys=0 for all persons alive """

        logger.debug("This is epi reporting my health values")

        df = self.sim.population.props  # shortcut to population properties dataframe

        health_values = pd.Series(index=df.index[df.is_alive], data=0)
        return health_values  # returns the series


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# Health System Interaction Events


class HSI_bcg(HSI_Event, IndividualScopeEventMixin):
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
        self.TREATMENT_ID = "HSI_bcg"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(f"HSI_bcg: giving bcg to {person_id}")

        df = self.sim.population.props

        if df.at[person_id, "ep_bcg"] == False:

            # Make request for some consumables
            consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

            item_code1 = pd.unique(
                consumables.loc[consumables["Items"] == "BCG vaccine", "Item_Code",]
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
        logger.debug("HSI_bcg: did not run")


class HSI_opv1(HSI_Event, IndividualScopeEventMixin):
    """
    gives first dose vaccine 24 hours after birth or as soon as possible afterwards
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Epi)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["ConWithDCSA"] = 1  # This requires one ConWithDCSA appt

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "HSI_opv1"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(f"HSI_opv1: giving bcg to {person_id}")

        df = self.sim.population.props

        # Make request for some consumables
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        item_code1 = pd.unique(
            consumables.loc[consumables["Items"] == "Polio vaccine", "Item_Code",]
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
        logger.debug("HSI_opv1: did not run")


class HSI_DtpHibHepVaccine(HSI_Event, IndividualScopeEventMixin):
    """
    gives DTP-Hib_HepB, OPV, Pneumococcal and Rotavirus vaccine 6 weeks after birth
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Epi)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["ConWithDCSA"] = 1  # This requires one ConWithDCSA appt

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "HSI_DtpHibHepVaccine"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(f"HSI_DtpHibHepVaccine: requesting vaccines for {person_id}")

        df = self.sim.population.props

        # Make request for some consumables
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        penta_vax = pd.unique(
            consumables.loc[
                consumables["Items"] == "Pentavalent vaccine (DPT, Hep B, Hib)",
                "Item_Code",
            ]
        )[0]

        # OPV - oral vaccine
        opv = pd.unique(
            consumables.loc[consumables["Items"] == "Polio vaccine", "Item_Code",]
        )[0]

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
                penta_vax: 1,
                opv: 1,
                pneumo_vax: 1,
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
            logger.debug(f"Penta vax is available, so administer to {person_id}")

            df.at[person_id, "ep_dtp"] += 1
            df.at[person_id, "ep_hib"] += 1
            df.at[person_id, "ep_hep"] += 1

        # check if OPV available
        if outcome_of_request_for_consumables["Item_Code"][opv]:
            logger.debug("OPV is available, so administer")

            df.at[person_id, "ep_opv"] += 1

        # check if pneumococcal vaccine available and current year 2012 onwards
        if (
            (self.sim.date.year >= 2012)
            & outcome_of_request_for_consumables["Item_Code"][pneumo_vax]
            & outcome_of_request_for_consumables["Item_Code"][syringe]
        ):
            logger.debug(
                f"Pneumococcal vaccine is available, so administer to {person_id}"
            )

            df.at[person_id, "ep_pneumo"] += 1

        # rotavirus - oral vaccine
        # only 2 doses rotavirus given (week 6 and 10)
        # available from 2012 onwards
        if (df.at[person_id, "ep_rota"] < 2) & (self.sim.date.year >= 2012):

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
                logger.debug(
                    f"Rotavirus vaccine is available, so administer to {person_id}"
                )

                df.at[person_id, "ep_rota"] += 1

    def did_not_run(self):
        logger.debug("HSI_DtpHibHepVaccine: did not run")


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
        self.TREATMENT_ID = "HSI_MeaslesRubellaVaccine"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(f"HSI_MeaslesRubellaVaccine: giving measles+rubella vaccine to {person_id}")

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
            df.at[person_id, "ep_measles"] += 1
            df.at[person_id, "ep_rubella"] += 1

    def did_not_run(self):
        logger.debug("HSI_MeaslesRubellaVaccine: did not run")


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------


class EpiLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """ output vaccine coverage every year
        """

        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, Epi)

    def apply(self, population):
        df = population.props
        now = self.sim.date

        infants = len(df[df.is_alive & (df.age_years <= 1)])

        # bcg vaccination coverage in <1 year old children
        bcg = len(df[df.is_alive & df.ep_bcg & (df.age_years <= 1)])
        bcg_coverage = ((bcg / infants) * 100) if infants else 0
        assert bcg_coverage <= 100

        # dtp3 vaccination coverage in <1 year old children
        dtp3 = len(df[df.is_alive & (df.ep_dtp >= 3) & (df.age_years <= 1)])
        dtp3_coverage = ((dtp3 / infants) * 100) if infants else 0
        assert dtp3_coverage <= 100

        # opv3 vaccination coverage in <1 year old children
        opv3 = len(df[df.is_alive & (df.ep_opv >= 3) & (df.age_years <= 1)])
        opv3_coverage = ((opv3 / infants) * 100) if infants else 0
        assert opv3_coverage <= 100

        # hib3 vaccination coverage in <1 year old children
        hib3 = len(df[df.is_alive & (df.ep_hib >= 3) & (df.age_years <= 1)])
        hib3_coverage = ((hib3 / infants) * 100) if infants else 0
        assert hib3_coverage <= 100

        # hep3 vaccination coverage in <1 year old children
        hep3 = len(df[df.is_alive & (df.ep_hep >= 3) & (df.age_years <= 1)])
        hep3_coverage = ((hep3 / infants) * 100) if infants else 0
        assert hep3_coverage <= 100

        # pneumo3 vaccination coverage in <1 year old children
        pneumo3 = len(df[df.is_alive & (df.ep_pneumo >= 3) & (df.age_years <= 1)])
        pneumo3_coverage = ((pneumo3 / infants) * 100) if infants else 0
        assert pneumo3_coverage <= 100

        # rota vaccination coverage in <1 year old children
        rota2 = len(df[df.is_alive & (df.ep_rota >= 2) & (df.age_years <= 1)])
        rota_coverage = ((rota2 / infants) * 100) if infants else 0
        assert rota_coverage <= 100

        # measles vaccination coverage in <1 year old children - 1 dose
        measles = len(df[df.is_alive & (df.ep_measles >= 1) & (df.age_years <= 1)])
        measles_coverage = ((measles / infants) * 100) if infants else 0
        assert measles_coverage <= 100

        # rubella vaccination coverage in <1 year old children - 1 dose
        rubella = len(df[df.is_alive & (df.ep_rubella >= 1) & (df.age_years <= 1)])
        rubella_coverage = ((rubella / infants) * 100) if infants else 0
        assert rubella_coverage <= 100

        logger.info(
            "%s|ep_vaccine_coverage|%s",
            now,
            {
                "epNumInfantsUnder1": infants,
                "epBcgCoverage": bcg_coverage,
                "epDtp3Coverage": dtp3_coverage,
                "epOpv3Coverage": opv3_coverage,
                "epHib3Coverage": hib3_coverage,
                "epHep3Coverage": hep3_coverage,
                "epPneumo3Coverage": pneumo3_coverage,
                "epRota2Coverage": rota_coverage,
                "epMeaslesCoverage": measles_coverage,
                "epRubellaCoverage": rubella_coverage,
            },
        )
