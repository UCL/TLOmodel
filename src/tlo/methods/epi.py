import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

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

        p["baseline_coverage"] = pd.read_csv(
            Path(self.resourcefilepath) / "ResourceFile_EPI_WHO_estimates.csv"
        )

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
        df.at[df.is_alive, "ep_bcg"] = False
        df.at[df.is_alive, "ep_opv"] = 0
        df.at[df.is_alive, "ep_dtp"] = 0
        df.at[df.is_alive, "ep_hib"] = 0
        df.at[df.is_alive, "ep_hep"] = 0
        df.at[df.is_alive, "ep_pneumo"] = 0
        df.at[df.is_alive, "ep_rota"] = 0
        df.at[df.is_alive, "ep_measles"] = 0
        df.at[df.is_alive, "ep_rubella"] = 0
        df.at[df.is_alive, "ep_hpv"] = 0
        df.at[df.is_alive, "ep_td"] = 0

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
        # use same random draw value for all vaccines - will induce correlations (good)
        # there are individuals who have high probability of getting all vaccines
        # some individuals will have consistently poor coverage
        random_draw = self.rng.rand(len(df_vaccine_baseline))
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

        # from 2010-2018 use the reported vaccine coverage values and schedule individual events (not HSI)
        # no consumables will be tracked up to end-2018
        if year <= 2018:

            # lookup the correct table of vaccine estimates for this child
            vax_coverage = p["district_vaccine_coverage"]
            ind_vax_coverage = vax_coverage.loc[(vax_coverage.District == district) & (vax_coverage.Year == year)]
            assert not ind_vax_coverage.empty

            # schedule bcg birth dose according to current coverage
            # some values are >1
            if self.rng.rand(1) < ind_vax_coverage.BCG.values:

                bcg_event = BcgEvent(self, child_id)
                self.sim.schedule_event(bcg_event, self.sim.date + DateOffset(days=1))

            # assign OPV first dose according to current coverage
            # OPV doses 2-4 are given during the week 6, 10, 14 penta, pneumo, rota appts
            if self.rng.rand(1) < ind_vax_coverage.OPV3.values:
                opv1_event = OpvEvent(self, child_id)
                self.sim.schedule_event(opv1_event, self.sim.date + DateOffset(days=1))

            # OPV2
            # coverage estimates for 3 doses reported, use these for doses 2-4
            if self.rng.rand(1) < ind_vax_coverage.OPV3.values:
                opv2_event = OpvEvent(self, child_id)
                self.sim.schedule_event(opv2_event, self.sim.date + DateOffset(weeks=6))
                self.sim.schedule_event(opv2_event, self.sim.date + DateOffset(weeks=10))
                self.sim.schedule_event(opv2_event, self.sim.date + DateOffset(weeks=14))

            # DTP1_HepB - up to and including 2012, then replaced by pentavalent vaccine
            if self.rng.rand(1) < ind_vax_coverage.DTP1.values:
                dtp1_event = DTP_HepEvent(self, child_id)
                self.sim.schedule_event(dtp1_event, self.sim.date + DateOffset(weeks=6))

            # DTP2_HepB - up to and including 2012
            # second doses not reported - same coverage for second and third doses
            if self.rng.rand(1) < ind_vax_coverage.DTP3.values:
                dtp2_event = DTP_HepEvent(self, child_id)
                self.sim.schedule_event(dtp2_event, self.sim.date + DateOffset(weeks=10))
                self.sim.schedule_event(dtp2_event, self.sim.date + DateOffset(weeks=14))

            # HIB1
            if self.rng.rand(1) < ind_vax_coverage.Hib3.values:
                hib_event = HibEvent(self, child_id)
                self.sim.schedule_event(hib_event, self.sim.date + DateOffset(weeks=6))
                self.sim.schedule_event(hib_event, self.sim.date + DateOffset(weeks=10))
                self.sim.schedule_event(hib_event, self.sim.date + DateOffset(weeks=14))

            # PNEUMO1 - all three doses reported separately
            if self.rng.rand(1) < ind_vax_coverage.Pneumo1.values:
                pneumo1_event = PneumococcalEvent(self, child_id)
                self.sim.schedule_event(pneumo1_event, self.sim.date + DateOffset(weeks=6))

            # PNEUMO2
            if self.rng.rand(1) < ind_vax_coverage.Pneumo2.values:
                pneumo2_event = PneumococcalEvent(self, child_id)
                self.sim.schedule_event(pneumo2_event, self.sim.date + DateOffset(weeks=10))

            # PNEUMO3
            if self.rng.rand(1) < ind_vax_coverage.Pneumo3.values:
                pneumo3_event = PneumococcalEvent(self, child_id)
                self.sim.schedule_event(pneumo3_event, self.sim.date + DateOffset(weeks=14))

            # ROTA1 - doses 1 and 2 reported separately
            if self.rng.rand(1) < ind_vax_coverage.Rotavirus1.values:
                rota1_event = RotavirusEvent(self, child_id)
                self.sim.schedule_event(rota1_event, self.sim.date + DateOffset(weeks=6))

            # ROTA2
            if self.rng.rand(1) < ind_vax_coverage.Rotavirus2.values:
                rota2_event = RotavirusEvent(self, child_id)
                self.sim.schedule_event(rota2_event, self.sim.date + DateOffset(weeks=10))

            # PENTA1
            if self.rng.rand(1) < ind_vax_coverage.DTPHepHib1.values:
                # print("PENTA1 TRUE")
                penta1_event = DTP_Hib_HepEvent(self, child_id)
                self.sim.schedule_event(penta1_event, self.sim.date + DateOffset(weeks=6))

            # PENTA2 - second dose not reported so use 3 dose coverage
            if self.rng.rand(1) < ind_vax_coverage.DTPHepHib3.values:
                penta2_event = DTP_Hib_HepEvent(self, child_id)
                self.sim.schedule_event(penta2_event, self.sim.date + DateOffset(weeks=10))
                self.sim.schedule_event(penta2_event, self.sim.date + DateOffset(weeks=14))

            # Measles, rubella - first dose, 2018 onwards
            if self.rng.rand(1) < ind_vax_coverage.MCV1_MR1.values:
                mr1_event = MeaslesRubellaEvent(self, child_id)
                self.sim.schedule_event(mr1_event, self.sim.date + DateOffset(months=9))

            # Measles, rubella - second dose
            if self.rng.rand(1) < ind_vax_coverage.MCV2_MR2.values:
                mr1_event = MeaslesRubellaEvent(self, child_id)
                self.sim.schedule_event(mr1_event, self.sim.date + DateOffset(months=15))

            # Measles - first dose, only one dose pre-2017 and no rubella
            if self.rng.rand(1) < ind_vax_coverage.MCV1.values:
                m1_event = MeaslesEvent(self, child_id)
                self.sim.schedule_event(m1_event, self.sim.date + DateOffset(months=9))

        # ----------------------------------- 2019 onwards -----------------------------------
        else:
            # schedule bcg - now dependent on health system capacity / stocks
            bcg_appt = HSI_bcg(self, person_id=child_id)

            # Request the health system to have this bcg vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                bcg_appt,
                priority=1,
                topen=self.sim.date + DateOffset(days=1),
                tclose=None,
            )

            # OPV
            # OPV doses 2-4 are given during the week 6, 10, 14 penta, pneumo, rota appts
            opv1_appt = HSI_opv(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                opv1_appt,
                priority=1,
                topen=self.sim.date + DateOffset(days=1),
                tclose=None,
            )

            # OPV2
            opv2_appt = HSI_opv(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                opv2_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=6),
                tclose=None,
            )

            # OPV3
            opv3_appt = HSI_opv(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                opv3_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=10),
                tclose=None,
            )

            # OPV4
            opv4_appt = HSI_opv(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                opv4_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=14),
                tclose=None,
            )

            # PNEUMO1
            pneumo1_appt = HSI_PneumoVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                pneumo1_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=6),
                tclose=None,
            )

            # PNEUMO2
            pneumo2_appt = HSI_PneumoVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                pneumo2_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=10),
                tclose=None,
            )

            # PNEUMO3
            pneumo3_appt = HSI_PneumoVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                pneumo3_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=14),
                tclose=None,
            )

            # ROTA1
            rota1_appt = HSI_RotaVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                rota1_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=6),
                tclose=None,
            )

            # ROTA2
            rota2_appt = HSI_RotaVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                rota2_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=10),
                tclose=None,
            )

            # PENTA1
            penta1_appt = HSI_DtpHibHepVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                penta1_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=6),
                tclose=None
            )

            # PENTA2
            penta2_appt = HSI_DtpHibHepVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                penta2_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=10),
                tclose=None,
            )

            # PENTA3
            penta3_appt = HSI_DtpHibHepVaccine(self, person_id=child_id)

            # Request the health system to have this vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                penta3_appt,
                priority=1,
                topen=self.sim.date + DateOffset(weeks=14),
                tclose=None,
            )

            # Measles, rubella - first dose, 2018 onwards
            mr_appt = HSI_MeaslesRubellaVaccine(self, person_id=child_id)

            # Request the health system to have this bcg vaccination appointment
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                mr_appt,
                priority=1,
                topen=self.sim.date + DateOffset(months=9),
                tclose=None,
            )

            # Measles, rubella - second dose
            mr_appt = HSI_MeaslesRubellaVaccine(self, person_id=child_id)

            # Request the health system to have this measles/rubella vaccination appointment
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
# Individually Scheduled Vaccine Events
# ---------------------------------------------------------------------------------

# BCG
class BcgEvent(Event, IndividualScopeEventMixin):
    """ give BCG vaccine at birth
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props

        df.at[person_id, "ep_bcg"] = True


# OPV
class OpvEvent(Event, IndividualScopeEventMixin):
    """ give oral poliovirus vaccine
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props

        df.at[person_id, "ep_opv"] += 1


# DTP_Hep
class DTP_HepEvent(Event, IndividualScopeEventMixin):
    """ give DTP_Hep vaccine
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props

        df.at[person_id, "ep_dtp"] += 1
        df.at[person_id, "ep_hep"] += 1


# DTP_Hib_Hep
class DTP_Hib_HepEvent(Event, IndividualScopeEventMixin):
    """ give DTP_Hib_Hep vaccine
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props

        df.at[person_id, "ep_dtp"] += 1
        df.at[person_id, "ep_hep"] += 1
        df.at[person_id, "ep_hib"] += 1


# Rotavirus
class RotavirusEvent(Event, IndividualScopeEventMixin):
    """ give Rotavirus vaccine
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props

        df.at[person_id, "ep_rota"] += 1


# Penumococcal vaccine (PCV)
class PneumococcalEvent(Event, IndividualScopeEventMixin):
    """ give Pneumococcal vaccine
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props

        df.at[person_id, "ep_pneumo"] += 1


# Hib vaccine
class HibEvent(Event, IndividualScopeEventMixin):
    """ give Haemophilus influenza B vaccine
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props

        df.at[person_id, "ep_hib"] += 1


# Measles vaccine
class MeaslesEvent(Event, IndividualScopeEventMixin):
    """ give measles vaccine
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props

        df.at[person_id, "ep_measles"] += 1


# Measles/Rubella vaccine
class MeaslesRubellaEvent(Event, IndividualScopeEventMixin):
    """ give measles/rubella vaccine
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props

        df.at[person_id, "ep_measles"] += 1
        df.at[person_id, "ep_rubella"] += 1

# ---------------------------------------------------------------------------------
# Health System Interaction Events
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
        logger.debug(
            "HpvScheduleEvent selecting eligible 9-yr olds for HPV vaccine")

        df = population.props
        now = self.sim.date

        #  sample using the prob_inf scaled by relative susceptibility
        hpv_vax = df.index[(df.age_years == 9) & (df.sex == "F")]

        # scatter vaccination days across the year
        # todo: HPV vaccine may be offered on scheduled clinic days / weeks - check
        random_day = self.module.rng.choice(
            list(range(0, 365)),
            size=len(hpv_vax),
            replace=True,
            p=[(1 / 365)] * 365,
        )

        scheduled_vax_date = now + pd.to_timedelta(random_day, unit="d")

        for person_index in hpv_vax:
            logger.debug(
                f"HpvScheduleEvent scheduling HPV vaccine for {person_index}")

            # find the index in hpv_vax
            # then select that value from the scheduled_vax_date
            tmp = np.where(hpv_vax == person_index)
            tmp_date = scheduled_vax_date[tmp]

            # first dose
            event = HSI_HpvVaccine(
                self.module, person_id=person_index
            )
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                event,
                priority=2,
                topen=tmp_date,
                tclose=tmp_date + DateOffset(weeks=2),
            )

            # second dose
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                event,
                priority=2,
                topen=tmp_date + DateOffset(weeks=4),
                tclose=tmp_date + DateOffset(weeks=6),
            )

# ---------------------------------------------------------------------------------
# Health System Interaction Events
# ---------------------------------------------------------------------------------


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
        self.TREATMENT_ID = "Epi_bcg"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(f"HSI_bcg: giving bcg to {person_id}")

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
        self.TREATMENT_ID = "Epi_opv"
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
        self.TREATMENT_ID = "Epi_Rota"
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
        self.TREATMENT_ID = "Epi_Pneumo"
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
        logger.debug(f"HSI_HpvVaccine: giving hpv vaccine to {person_id}")

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
        logger.debug("HSI_HpvVaccine: did not run")


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
        logger.debug(f"HSI_TdVaccine: giving Td vaccine to {person_id}")

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
        logger.debug("HSI_TdVaccine: did not run")

# ---------------------------------------------------------------------------------
# LOGGING
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

        birth_dose_denom = len(df[df.is_alive & (df.age_years <= 1)])

        # the eligible group for coverage estimates will be those from 14 weeks and older
        # younger infants won't have had the three-dose course yet
        infants_denom = len(df[df.is_alive & (df.age_years <= 1) & (df.age_exact_years >= 0.27)])

        # bcg vaccination coverage in <1 year old children
        bcg = len(df[df.is_alive & df.ep_bcg & (df.age_years <= 1)])
        bcg_coverage = ((bcg / birth_dose_denom) * 100) if birth_dose_denom else 0
        assert bcg_coverage <= 100

        # dtp3 vaccination coverage in children aged between 14 weeks and 1 year
        dtp3 = len(df[df.is_alive & (df.ep_dtp >= 3) & (df.age_years <= 1) & (df.age_exact_years >= 0.27)])
        dtp3_coverage = ((dtp3 / infants_denom) * 100) if infants_denom else 0
        assert dtp3_coverage <= 100

        # opv3 vaccination coverage in children aged between 14 weeks and 1 year
        opv3 = len(df[df.is_alive & (df.ep_opv >= 3) & (df.age_years <= 1) & (df.age_exact_years >= 0.27)])
        opv3_coverage = ((opv3 / infants_denom) * 100) if infants_denom else 0
        assert opv3_coverage <= 100

        # hib3 vaccination coverage in children aged between 14 weeks and 1 yearn
        hib3 = len(df[df.is_alive & (df.ep_hib >= 3) & (df.age_years <= 1) & (df.age_exact_years >= 0.27)])
        hib3_coverage = ((hib3 / infants_denom) * 100) if infants_denom else 0
        assert hib3_coverage <= 100

        # hep3 vaccination coverage in children aged between 14 weeks and 1 yearn
        hep3 = len(df[df.is_alive & (df.ep_hep >= 3) & (df.age_years <= 1) & (df.age_exact_years >= 0.27)])
        hep3_coverage = ((hep3 / infants_denom) * 100) if infants_denom else 0
        assert hep3_coverage <= 100

        # pneumo3 vaccination coverage in children aged between 14 weeks and 1 year
        pneumo3 = len(df[df.is_alive & (df.ep_pneumo >= 3) & (df.age_years <= 1) & (df.age_exact_years >= 0.27)])
        pneumo3_coverage = ((pneumo3 / infants_denom) * 100) if infants_denom else 0
        assert pneumo3_coverage <= 100

        # rota vaccination coverage in <1 children aged between 14 weeks and 1 year
        rota2 = len(df[df.is_alive & (df.ep_rota >= 2) & (df.age_years <= 1) & (df.age_exact_years >= 0.27)])
        rota_coverage = ((rota2 / infants_denom) * 100) if infants_denom else 0
        assert rota_coverage <= 100

        # measles vaccination coverage in <2 year old children - 1 dose
        # first dose is at 9 months, second dose is 15 months
        # so check coverage in 15 month -2 year olds
        toddlers = len(df[df.is_alive & (df.age_exact_years >= 1.25) & (df.age_years <= 2)])

        measles = len(df[df.is_alive & (df.ep_measles >= 1) & (df.age_exact_years >= 1.25) & (df.age_years <= 2)])
        measles_coverage = ((measles / toddlers) * 100) if toddlers else 0
        assert measles_coverage <= 100

        # rubella vaccination coverage in <2 year old children - 1 dose
        # first dose is at 9 months, second dose is 15 months
        rubella = len(df[df.is_alive & (df.ep_rubella >= 1) & (df.age_exact_years >= 1.25) & (df.age_years <= 2)])
        rubella_coverage = ((rubella / toddlers) * 100) if toddlers else 0
        assert rubella_coverage <= 100

        # HPV vaccination coverage in adolescent girls - 1 dose
        # first dose is at 9 years
        # check coverage in girls aged 10-12
        adol_girls = len(df[df.is_alive & (df.age_exact_years >= 10) & (df.age_years <= 12)])
        hpv = len(df[df.is_alive & (df.ep_hpv >= 1) & (df.age_exact_years >= 10) & (df.age_years <= 12)])
        hpv_coverage = ((hpv / adol_girls) * 100) if adol_girls else 0
        assert hpv_coverage <= 100

        logger.info(
            "%s|ep_vaccine_coverage|%s",
            now,
            {
                "epNumInfantsUnder1": infants_denom,
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
            },
        )
