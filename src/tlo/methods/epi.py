import numpy as np
import pandas as pd
import os
from pathlib import Path

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

        "district_vaccine_coverage": Parameter(Types.DATA_FRAME, "reported vaccine coverage estimates by district"),

        # baseline vaccination coverage
        "baseline_coverage": Parameter(
            Types.REAL, "baseline vaccination coverage (all vaccines)"
        )
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
        "ep_district_edited": Property(Types.STRING, "edited district of residence to match EPI district list")
    }

    # Declaration of the symptoms that this module will use
    SYMPTOMS = {}

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):

        p = self.parameters

        # district-level coverage estimates from 1980-2009
        workbook = pd.read_excel(
            Path(self.resourcefilepath, "ResourceFile_EPI.xlsx"), sheet_name=None
        )
        p["baseline_coverage"] = workbook["WHO_Estimates"]
        p["district_vaccine_coverage"] = pd.read_csv(
            Path(self.resourcefilepath) / "ResourceFile_EPI_summary.csv"
        )

        # tmp values for current vaccine coverage
        # also limited by consumables stocks
        p["bcg_coverage"] = 1
        p["opv1_coverage"] = 1
        p["opv2_coverage"] = 1
        p["opv3_coverage"] = 1
        p["opv4_coverage"] = 1
        p["penta1_coverage"] = 1
        p["penta2_coverage"] = 1
        p["penta3_coverage"] = 1
        p["measles_rubella1_coverage"] = 1
        p["measles_rubella2_coverage"] = 1

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
        # use vaccine coverage estimates for each year prior to 2010
        # assuming only <1 yr olds were vaccinated each year
        # match up vaccine coverage for each person based on their age
        # anyone over age 29 will not have matching vaccine coverage estimates
        # therefore no vaccinations for them
        df_vaccine_baseline = df.merge(p["baseline_coverage"],
                                       left_on=["age_years"],
                                       right_on=["AgeOn01Jan2010"],
                                       how="left")

        # what happens with the nan values in the df for vaccine coverage (age >30)??
        # seems fine!!
        # will have a susceptible older population though
        # use same random draw for all vaccines - will induce correlations
        # there are individuals who have high probability of getting all vaccines
        # some individuals will have consistently poor coverage
        random_draw = self.rng.random_sample(size=len(df_vaccine_baseline))
        bcg_idx = df_vaccine_baseline.index[
            df_vaccine_baseline.is_alive
            & (random_draw < df_vaccine_baseline["BCG"])
        ]

        df.at[bcg_idx, "ep_bcg"] = True

        # Polio OPV
        # from 1980-2009 average opv3 coverage is 79.5% (WHO estimates): all 3 doses OPV
        # assume no partial protection if < 3 doses (all-or-nothing response)
        opv3_idx = df_vaccine_baseline.index[
            df_vaccine_baseline.is_alive
            & (random_draw < df_vaccine_baseline["Pol3"])
        ]

        df.at[opv3_idx, "ep_opv"] = 3

        # DTP3
        # available since 1980
        dtp3_idx = df_vaccine_baseline.index[
            df_vaccine_baseline.is_alive
            & (random_draw < df_vaccine_baseline["DTP3"])
        ]

        df.at[dtp3_idx, "ep_dtp"] = 3

        # Hep3
        # available since 2002
        # by Jan 2010, anyone <9 years has 87.5% prob of having vaccine
        hep3_idx = df_vaccine_baseline.index[
            df_vaccine_baseline.is_alive
            & (random_draw < df_vaccine_baseline["HepB3"])
        ]

        df.at[hep3_idx, "ep_hep"] = 3

        # Hib3
        # available since 2002
        # by Jan 2010, anyone <9 years has 87.5% prob of having vaccine
        hib3_idx = df_vaccine_baseline.index[
            df_vaccine_baseline.is_alive
            & (random_draw < df_vaccine_baseline["Hib3"])
        ]

        df.at[hib3_idx, "ep_hib"] = 3

        # Measles
        # available since 1980
        # second dose only started in 2015
        # by Jan 2010, anyone <=30 years has 77.2% prob of having vaccine
        measles_idx = df_vaccine_baseline.index[
            df_vaccine_baseline.is_alive
            & (random_draw < df_vaccine_baseline["MCV1"])
        ]

        df.at[measles_idx, "ep_measles"] = 3

    def initialise_simulation(self, sim):

        # add an event to log to screen
        sim.schedule_event(EpiLoggingEvent(self), sim.date + DateOffset(days=364))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual

        birth doses occur within 24 hours of delivery

        from 2010-2018 data on vaccine coverage are used to determine probability of receiving vaccine
        vaccinations are scheduled to occur with a probability dependent on the year and district
        from 2019 onwards, probability will be 1 and coverage determined by vaccine availability

        how to stop consumables limiting vaccine availability before 2019??
        don't use outcome_of_request_for_consumables - just assume it's received

        2012 data is patchy, no record of Hep vaccine but it was used before 2012
        assume hepB3 coverage in 2012 same as 2011
        same with Hib

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """
        df = self.sim.population.props  # shortcut to the population props dataframe
        p = self.parameters
        year = self.sim.date.year

        # keep coverage estimates for 2018 for now
        if year > 2018:
            year = 2018

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

        # rename districts to match EPI data
        df.at[child_id, "ep_district_edited"] = df.at[child_id, "district_of_residence"]

        if df.at[child_id, "ep_district_edited"] == "Lilongwe City":
            df.at[child_id, "ep_district_edited"] = "Lilongwe"

        elif df.at[child_id, "ep_district_edited"] == "Blantyre City":
            df.at[child_id, "ep_district_edited"] = "Blantyre"

        elif df.at[child_id, "ep_district_edited"] == "Zomba City":
            df.at[child_id, "ep_district_edited"] = "Zomba"

        elif df.at[child_id, "ep_district_edited"] == "Mzuzu City":
            df.at[child_id, "ep_district_edited"] = "Mzimba"

        elif df.at[child_id, "ep_district_edited"] == "Mzuzu":
            df.at[child_id, "ep_district_edited"] = "Mzimba"

        elif df.at[child_id, "ep_district_edited"] == "Nkhata Bay":
            df.at[child_id, "ep_district_edited"] = "Mzimba"

        # look up coverage of every vaccine
        # anything delivered after 12months needs the estimate from the following year
        district = df.at[child_id, 'ep_district_edited']
        # todo note: Mzuzu is not in EPI database (Mzimba)

        # lookup the correct table of vaccine estimates for this child
        vax_coverage = p["district_vaccine_coverage"]
        ind_vax_coverage = vax_coverage.loc[(vax_coverage.District == district) & (vax_coverage.Year == year)]
        # print(district)
        assert not ind_vax_coverage.empty

        # assign bcg according to current coverage
        # some values are >1
        if self.rng.random_sample(size=1) < ind_vax_coverage.BCG.values:
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
        if self.rng.random_sample(size=1) < ind_vax_coverage.OPV3.values:
            opv1_appt = HSI_opv(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                opv1_appt,
                priority=1,
                topen=self.sim.date + DateOffset(days=1),
                tclose=None,
            )

        # OPV2
        if self.rng.random_sample(size=1) < ind_vax_coverage.OPV3.values:
            opv2_appt = HSI_opv(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                opv2_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=6),
                tclose=None,
            )

        # OPV3
        if self.rng.random_sample(size=1) < ind_vax_coverage.OPV3.values:
            opv3_appt = HSI_opv(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                opv3_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=10),
                tclose=None,
            )

        # OPV4
        if self.rng.random_sample(size=1) < ind_vax_coverage.OPV3.values:
            opv4_appt = HSI_opv(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                opv4_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=14),
                tclose=None,
            )

        # DTP1_HepB - up to and including 2012, then replaced by pentavalent vaccine
        if self.rng.random_sample(size=1) < ind_vax_coverage.DTP1.values:
            dtp1_appt = HSI_DtpHepVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                dtp1_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=6),
                tclose=None,
            )

        # DTP2_HepB - up to and including 2012
        if self.rng.random_sample(size=1) < ind_vax_coverage.DTP3.values:
            dtp2_appt = HSI_DtpHepVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                dtp2_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=10),
                tclose=None,
            )

        # DTP3_HepB - up to and including 2012
        if self.rng.random_sample(size=1) < ind_vax_coverage.DTP3.values:
            dtp3_appt = HSI_DtpHepVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                dtp3_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=14),
                tclose=None,
            )

        # HIB1
        if self.rng.random_sample(size=1) < ind_vax_coverage.Hib3.values:
            Hib1_appt = HSI_HibVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                Hib1_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=6),
                tclose=None,
            )

        # Hib2
        if self.rng.random_sample(size=1) < ind_vax_coverage.Hib3.values:
            Hib2_appt = HSI_HibVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                Hib2_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=10),
                tclose=None,
            )

        # Hib3
        if self.rng.random_sample(size=1) < ind_vax_coverage.Hib3.values:
            Hib3_appt = HSI_HibVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                Hib3_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=14),
                tclose=None,
            )

        # PNEUMO1
        if self.rng.random_sample(size=1) < ind_vax_coverage.Pneumo1.values:
            pneumo1_appt = HSI_PneumoVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                pneumo1_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=6),
                tclose=None,
            )

        # PNEUMO2
        if self.rng.random_sample(size=1) < ind_vax_coverage.Pneumo2.values:
            pneumo2_appt = HSI_PneumoVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                pneumo2_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=10),
                tclose=None,
            )

        # PNEUMO3
        if self.rng.random_sample(size=1) < ind_vax_coverage.Pneumo3.values:
            pneumo3_appt = HSI_PneumoVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                pneumo3_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=14),
                tclose=None,
            )

        # ROTA1
        if self.rng.random_sample(size=1) < ind_vax_coverage.Rotavirus1.values:
            rota1_appt = HSI_RotaVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                rota1_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=6),
                tclose=None,
            )

        # ROTA2
        if self.rng.random_sample(size=1) < ind_vax_coverage.Rotavirus2.values:
            rota2_appt = HSI_RotaVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                rota2_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=10),
                tclose=None,
            )

        # PENTA1
        if self.rng.random_sample(size=1) < ind_vax_coverage.DTPHepHib1.values:
            # print("PENTA1 TRUE")
            penta1_appt = HSI_DtpHibHepVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                penta1_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=6),
                tclose=None
            )
        # else:
        #     print(ind_vax_coverage.DTPHepHib1.values, "Penta1 FALSE")

        # PENTA2
        if self.rng.random_sample(size=1) < ind_vax_coverage.DTPHepHib3.values:
            # print("PENTA2 TRUE")
            penta2_appt = HSI_DtpHibHepVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                penta2_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=10),
                tclose=None,
            )
        # else:
        #     print(ind_vax_coverage.DTPHepHib3.values, "Penta2 FALSE")

        # PENTA3
        # print (ind_vax_coverage.DTPHepHib3.values)
        if self.rng.random_sample(size=1) < ind_vax_coverage.DTPHepHib3.values:
            # print("PENTA3 TRUE")
            penta3_appt = HSI_DtpHibHepVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                penta3_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=14),
                tclose=None,
            )
        # else:
            # print(ind_vax_coverage.DTPHepHib3.values, "Penta3 FALSE")

        # Measles, rubella - first dose, 2018 onwards
        if self.rng.random_sample(size=1) < ind_vax_coverage.MCV1_MR1.values:
            mr_appt = HSI_MeaslesRubellaVaccine(self, person_id=child_id)

            # Request the health system to have this bcg vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                mr_appt,
                priority=1,
                topen=self.sim.date + DateOffset(months=9),
                tclose=None,
            )

        # Measles, rubella - second dose
        if self.rng.random_sample(size=1) < ind_vax_coverage.MCV2_MR2.values:
            mr_appt = HSI_MeaslesRubellaVaccine(self, person_id=child_id)

            # Request the health system to have this measles/rubella vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                mr_appt,
                priority=1,
                topen=self.sim.date + DateOffset(months=15),
                tclose=None,
            )

        # Measles - first dose, only one dose pre-2017 and no rubella
        if self.rng.random_sample(size=1) < ind_vax_coverage.MCV1.values:
            m_appt = HSI_MeaslesVaccine(self, person_id=child_id)

            # Request the health system to have this measles/rubella vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                m_appt,
                priority=1,
                topen=self.sim.date + DateOffset(months=9),
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
        self.TREATMENT_ID = "HSI_opv"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(f"HSI_opv: giving opv to {person_id}")

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
        logger.debug("HSI_opv: did not run")


class HSI_DtpHepVaccine(HSI_Event, IndividualScopeEventMixin):
    """
    gives DTP vaccine 6 weeks after birth
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Epi)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["ConWithDCSA"] = 1  # This requires one ConWithDCSA appt

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "HSI_DtpVaccine"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(f"HSI_DtpHepVaccine: requesting vaccines for {person_id}")

        df = self.sim.population.props

        # Make request for some consumables
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        dtp_vax = pd.unique(
            consumables.loc[
                consumables["Items"] == "DTP vaccine",
                "Item_Code",
            ]
        )[0]

        hep_vax = pd.unique(
            consumables.loc[
                consumables["Items"] == "Hepatitis B vaccine",
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
                dtp_vax: 1,
                hep_vax: 1,
                syringe: 2,
                disposal: 1,
            },
        }

        outcome_of_request_for_consumables = self.sim.modules[
            "HealthSystem"
        ].request_consumables(hsi_event=self, cons_req_as_footprint=consumables_needed)

        # check if DTP and syringes available
        if outcome_of_request_for_consumables:
            logger.debug(f"DTP_Hep vax is available, so administer to {person_id}")

            df.at[person_id, "ep_dtp"] += 1
            df.at[person_id, "ep_hep"] += 1

    def did_not_run(self):
        logger.debug("HSI_DtpHepVaccine: did not run")


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
            logger.debug(f"Penta vax is available, so administer to {person_id}")

            df.at[person_id, "ep_dtp"] += 1
            df.at[person_id, "ep_hib"] += 1
            df.at[person_id, "ep_hep"] += 1
        else:
            logger.debug(f"Penta vax is not available for person {person_id}")

    def did_not_run(self):
        logger.debug("HSI_DtpHibHepVaccine: did not run")


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
        self.TREATMENT_ID = "HSI_RotaVaccine"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(f"HSI_RotaVaccine: requesting vaccines for {person_id}")

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
                logger.debug(
                    f"Rotavirus vaccine is available, so administer to {person_id}"
                )

                df.at[person_id, "ep_rota"] += 1
            else:
                logger.debug(f"Rotavirus vaccine is not available for person {person_id}")

    def did_not_run(self):
        logger.debug("HSI_RotaVaccine: did not run")


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
        self.TREATMENT_ID = "HSI_PneumoVaccine"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(f"HSI_PneumoVaccine: requesting vaccines for {person_id}")

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
            logger.debug(
                f"Pneumococcal vaccine is available, so administer to {person_id}"
            )

            df.at[person_id, "ep_pneumo"] += 1

    def did_not_run(self):
        logger.debug("HSI_PneumoVaccine: did not run")

class HSI_HibVaccine(HSI_Event, IndividualScopeEventMixin):
    """
    gives Hib vaccine 6, 10 and 14 weeks after birth
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Epi)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["ConWithDCSA"] = 1  # This requires one ConWithDCSA appt

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "HSI_HibVaccine"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(f"HSI_HibVaccine: requesting vaccines for {person_id}")

        df = self.sim.population.props

        # Make request for some consumables
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        # Hib vaccine
        hib_vax = pd.unique(
            consumables.loc[
                consumables["Items"] == "Hib vaccine", "Item_Code",
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
                hib_vax: 1,
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
            logger.debug(
                f"Hib vaccine is available, so administer to {person_id}"
            )

            df.at[person_id, "ep_hib"] += 1

    def did_not_run(self):
        logger.debug("HSI_HibVaccine: did not run")


class HSI_MeaslesVaccine(HSI_Event, IndividualScopeEventMixin):
    """
    administers single measles vaccine pre-2017 when measles+rubella vaccine became available
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Epi)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["ConWithDCSA"] = 1  # This requires one ConWithDCSA appt

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "HSI_MeaslesVaccine"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(f"HSI_MeaslesVaccine: checking measles vaccine availability for {person_id}")

        df = self.sim.population.props

        # Make request for some consumables
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        # todo this should be the single measles vaccine - not currently in consumables list
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
            logger.debug(f"HSI_MeaslesVaccine: measles vaccine is available for {person_id}")

            df.at[person_id, "ep_measles"] += 1
        else:
            logger.debug(f"HSI_MeaslesVaccine: measles vaccine is NOT available for {person_id}")

    def did_not_run(self):
        logger.debug("HSI_MeaslesVaccine: did not run")


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
        logger.debug(f"HSI_MeaslesRubellaVaccine: checking measles+rubella vaccine availability for {person_id}")

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
            logger.debug(f"HSI_MeaslesRubellaVaccine: measles+rubella vaccine is available for {person_id}")

            df.at[person_id, "ep_measles"] += 1
            df.at[person_id, "ep_rubella"] += 1
        else:
            logger.debug(f"HSI_MeaslesRubellaVaccine: measles+rubella vaccine is NOT available for {person_id}")

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

        # measles vaccination coverage in <2 year old children - 1 dose
        # first dose is at 9 months, second dose is 15 months
        # so check coverage in 1-2 year olds
        toddlers = len(df[df.is_alive & (df.age_years >= 1) & (df.age_years <= 2)])

        measles = len(df[df.is_alive & (df.ep_measles >= 1) & (df.age_years >= 1) & (df.age_years <= 2)])
        measles_coverage = ((measles / toddlers) * 100) if toddlers else 0
        assert measles_coverage <= 100

        # rubella vaccination coverage in <2 year old children - 1 dose
        # first dose is at 9 months, second dose is 15 months
        rubella = len(df[df.is_alive & (df.ep_rubella >= 1) & (df.age_years >= 1) & (df.age_years <= 2)])
        rubella_coverage = ((rubella / toddlers) * 100) if toddlers else 0
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
