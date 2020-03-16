import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import (
    Event,
    IndividualScopeEventMixin,
    PopulationScopeEventMixin,
    RegularEvent,
)
from tlo.methods.demography import InstantaneousDeath
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
        "pol1_coverage": Parameter(Types.REAL, "dummy vax coverage value"),
        "penta1_coverage": Parameter(Types.REAL, "dummy vax coverage value"),
        # baseline vaccination coverage for general population
        "pol3_baseline_coverage": Parameter(
            Types.REAL, "baseline pol3 vaccination coverage for population"
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
    }

    PROPERTIES = {
        "ep_bcg": Property(Types.BOOL, "received bcg vaccination"),
        "ep_pol1": Property(Types.BOOL, "received first dose of OPV vaccine"),
        "ep_pol2": Property(Types.BOOL, "received second dose of OPV vaccine"),
        "ep_pol3": Property(
            Types.BOOL, "received three doses of OPV vaccine - now protected"
        ),
        "ep_dtp1": Property(Types.BOOL, "received first dose of DTP vaccine"),
        "ep_dtp2": Property(Types.BOOL, "received second dose of DTP vaccine"),
        "ep_dtp3": Property(
            Types.BOOL, "received three doses of DTP vaccine - now protected"
        ),
        "ep_hib1": Property(Types.BOOL, "received first dose of Hib vaccine"),
        "ep_hib2": Property(Types.BOOL, "received second dose of Hib vaccine"),
        "ep_hib3": Property(
            Types.BOOL, "received three doses of Hib vaccine - now protected"
        ),
        "ep_hep1": Property(Types.BOOL, "received first dose of HepB vaccine"),
        "ep_hep2": Property(Types.BOOL, "received second dose of HepB vaccine"),
        "ep_hep3": Property(
            Types.BOOL, "received three doses of HepB vaccine - now protected"
        ),
    }

    # Declaration of the symptoms that this module will use
    SYMPTOMS = {}

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):

        # baseline coverage = average coverage estimates from 1980 (opv) or 1981 (bcg)
        self.parameters["pol3_baseline_coverage"] = 0.795
        self.parameters["bcg_baseline_coverage"] = 0.92
        self.parameters["dtp3_baseline_coverage"] = 0.807
        self.parameters["hep3_baseline_coverage"] = 0.875
        self.parameters["hib3_baseline_coverage"] = 0.875

        # district-level coverage estimates from 2010 to go here
        self.parameters["bcg_coverage"] = 0.8
        self.parameters["pol1_coverage"] = 0.8
        self.parameters["penta1_coverage"] = 0.8

        # ---- Register this module ----
        # Register this disease module with the health system
        self.sim.modules["HealthSystem"].register_disease_module(self)

    def initialise_population(self, population):

        df = population.props
        p = self.parameters

        # Set default for properties
        df.at[df.is_alive, "ep_bcg"] = False
        df.at[df.is_alive, "ep_pol1"] = False
        df.at[df.is_alive, "ep_pol2"] = False
        df.at[df.is_alive, "ep_pol3"] = False
        df.at[df.is_alive, "ep_dtp1"] = False
        df.at[df.is_alive, "ep_dtp2"] = False
        df.at[df.is_alive, "ep_dtp3"] = False
        df.at[df.is_alive, "ep_hib1"] = False
        df.at[df.is_alive, "ep_hib2"] = False
        df.at[df.is_alive, "ep_hib3"] = False
        df.at[df.is_alive, "ep_hep1"] = False
        df.at[df.is_alive, "ep_hep2"] = False
        df.at[df.is_alive, "ep_hep3"] = False

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
        # from 1980-2009 average pol3 coverage is 79.5% (WHO estimates): all 3 doses OPV
        # assume no partial protection if < 3 doses (all-or-nothing response)
        # by Jan 2010, anyone <31 years has 79.5% probability of being vaccinated
        # assuming only <1 yr olds were vaccinated each year
        random_draw = self.rng.random_sample(size=len(df))
        pol3_idx = df.index[
            df.is_alive
            & (df.age_years <= 30)
            & (random_draw < p["pol3_baseline_coverage"])
            ]

        df.at[pol3_idx, "ep_pol3"] = True

        # DTP3
        # available since 1980
        random_draw = self.rng.random_sample(size=len(df))
        dtp3_idx = df.index[
            df.is_alive
            & (df.age_years <= 30)
            & (random_draw < p["dtp3_baseline_coverage"])
            ]

        df.at[dtp3_idx, "ep_dtp3"] = True

        # Hep3
        # available since 2002
        # by Jan 2010, anyone <9 years has 87.5% prob of having vaccine
        random_draw = self.rng.random_sample(size=len(df))
        hep3_idx = df.index[
            df.is_alive
            & (df.age_years <= 8)
            & (random_draw < p["hep3_baseline_coverage"])
            ]

        df.at[hep3_idx, "ep_hep3"] = True

        # Hib3
        # available since 2002
        # by Jan 2010, anyone <9 years has 87.5% prob of having vaccine
        random_draw = self.rng.random_sample(size=len(df))
        hib3_idx = df.index[
            df.is_alive
            & (df.age_years <= 8)
            & (random_draw < p["hib3_baseline_coverage"])
            ]

        df.at[hib3_idx, "ep_hib3"] = True

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
        df.at[child_id, "ep_pol1"] = False
        df.at[child_id, "ep_pol2"] = False
        df.at[child_id, "ep_pol3"] = False
        df.at[child_id, "ep_dtp1"] = False
        df.at[child_id, "ep_dtp2"] = False
        df.at[child_id, "ep_dtp3"] = False
        df.at[child_id, "ep_hib1"] = False
        df.at[child_id, "ep_hib2"] = False
        df.at[child_id, "ep_hib3"] = False
        df.at[child_id, "ep_hep1"] = False
        df.at[child_id, "ep_hep2"] = False
        df.at[child_id, "ep_hep3"] = False

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
        # TODO use current coverage estimates by district
        if self.rng.random_sample(size=1) < self.parameters["pol1_coverage"]:
            pol1_appt = HSI_pol1(self, person_id=child_id)

            # Request the health system to have this bcg vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                pol1_appt,
                priority=1,
                topen=self.sim.date + DateOffset(days=1),
                tclose=None,
            )

        # PENTA
        # TODO use current coverage estimates by district
        if self.rng.random_sample(size=1) < self.parameters["penta1_coverage"]:
            penta1_appt = HSI_6WeekVaccines(self, person_id=child_id)

            # Request the health system to have this bcg vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                penta1_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=6),
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


class HSI_pol1(HSI_Event, IndividualScopeEventMixin):
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
        self.TREATMENT_ID = "HSI_pol1"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(f"HSI_pol1: giving bcg to {person_id}")

        df = self.sim.population.props

        if df.at[person_id, "ep_pol1"] == False:

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
            ].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed
            )

            if outcome_of_request_for_consumables:
                df.at[person_id, "ep_pol1"] = True

    def did_not_run(self):
        logger.debug("HSI_pol1: did not run")


class HSI_6WeekVaccines(HSI_Event, IndividualScopeEventMixin):
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
        self.TREATMENT_ID = "HSI_6WeekVaccines"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(f"HSI_6WeekVaccines: giving bcg to {person_id}")

        df = self.sim.population.props

        if not df.at[person_id, "ep_dtp1"]:

            # Make request for some consumables
            consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

            penta_vax = pd.unique(
                consumables.loc[
                    consumables["Items"] == "Pentavalent vaccine (DTP, Hep B, Hib)",
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

            # rotavirus - oral vaccine
            rotavirus_vax = pd.unique(
                consumables.loc[
                    consumables["Items"] == "Rotavirus vaccine", "Item_Code",
                ]
            )[0]

            syringe = pd.unique(
                consumables.loc[
                    consumables["Items"] == "Syringe, needle + swab", "Item_Code",
                ]
            )[0]

            disposal = pd.unique(
                consumables.loc[
                    consumables["Items"]
                    == "Safety box for used syringes/needles, 5 liter",
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
                    rotavirus_vax: 1,
                    syringe: 2,
                    disposal: 1,
                },
            }

            outcome_of_request_for_consumables = self.sim.modules[
                "HealthSystem"
            ].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed
            )

            # check if Penta and syringes available
            if (
                outcome_of_request_for_consumables["Item_Code"][penta_vax]
                & outcome_of_request_for_consumables["Item_Code"][syringe]
            ):
                logger.debug("Penta vax is available, so administer")

                df.at[person_id, "ep_dtp1"] = True
                df.at[person_id, "ep_hib1"] = True
                df.at[person_id, "ep_hep1"] = True

            # check if OPV available
            if outcome_of_request_for_consumables["Item_Code"][opv]:
                logger.debug("OPV is available, so administer")

                df.at[person_id, "ep_opv1"] = True

            # check if pneumococcal vaccine available
            if outcome_of_request_for_consumables["Item_Code"][pneumo_vax]:
                logger.debug("Pneumococcal vaccine is available, so administer")

                df.at[person_id, "ep_pneumo1"] = True

            # check if rotavirus vaccine available
            if outcome_of_request_for_consumables["Item_Code"][rotavirus_vax]:
                logger.debug("Rotavirus vaccine is available, so administer")

                df.at[person_id, "ep_rota1"] = True

    def did_not_run(self):
        logger.debug("HSI_6WeekVaccines: did not run")


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

        # pol3 vaccination coverage in <1 year old children
        pol3 = len(df[df.is_alive & df.ep_pol3 & (df.age_years <= 1)])
        pol3_coverage = ((pol3 / infants) * 100) if infants else 0
        assert pol3_coverage <= 100

        logger.info(
            "%s|ep_vaccine_coverage|%s",
            now,
            {
                "epNumInfantsUnder1": infants,
                "epBcgCoverage": bcg_coverage,
                "epPol3Coverage": pol3_coverage,
            },
        )
