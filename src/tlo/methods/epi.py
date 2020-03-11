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
        "p_coverage": Parameter(Types.REAL, "dummy vax coverage value"),
    }

    PROPERTIES = {
        "ep_bcg": Property(Types.BOOL, "received bcg vaccination"),
    }

    # Declaration of the symptoms that this module will use
    SYMPTOMS = {}

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        For now, we are going to hard code them explicity
        """
        self.parameters["p_coverage"] = 0.8

        # ---- Register this module ----
        # Register this disease module with the health system
        self.sim.modules["HealthSystem"].register_disease_module(self)

        # # Register this disease module with the symptom manager and declare the symptoms
        # self.sim.modules['SymptomManager'].register_disease_symptoms(module=self,
        #                                                            list_of_symptoms=['coughing_and_irritable'])

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """
        pass

    def initialise_simulation(self, sim):

        # add an event to log to screen
        sim.schedule_event(EpiLoggingEvent(self), sim.date + DateOffset(months=1))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """
        df = self.sim.population.props  # shortcut to the population props dataframe

        # Initialise all the properties that this module looks after:
        df.at[child_id, "cs_has_cs"] = False
        df.at[child_id, "cs_status"] = "N"
        df.at[child_id, "cs_date_acquired"] = pd.NaT
        df.at[child_id, "cs_scheduled_date_death"] = pd.NaT
        df.at[child_id, "cs_date_cure"] = pd.NaT

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug(
            "This is ChronicSyndrome, being alerted about a health system interaction "
            "person %d for: %s",
            person_id,
            treatment_id,
        )

        # To simulate a "piggy-backing" appointment, whereby additional treatment and test are done
        # for another disease, schedule another appointment (with smaller resources than a full appointmnet)
        # and set it to priority 0 (to give it highest possible priority).

        if treatment_id == "Mockitis_TreatmentMonitoring":
            piggy_back_dx_at_appt = HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment(
                self, person_id
            )
            piggy_back_dx_at_appt.TREATMENT_ID = "ChronicSyndrome_PiggybackAppt"

            # Arbitrarily reduce the size of appt footprint to reflect that this is a piggy back appt
            for key in piggy_back_dx_at_appt.EXPECTED_APPT_FOOTPRINT:
                piggy_back_dx_at_appt.EXPECTED_APPT_FOOTPRINT[key] = (
                    piggy_back_dx_at_appt.EXPECTED_APPT_FOOTPRINT[key] * 0.25
                )

            self.sim.modules["HealthSystem"].schedule_hsi_event(
                piggy_back_dx_at_appt, priority=0, topen=self.sim.date, tclose=None
            )

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug("This is chronicsyndrome reporting my health values")

        df = self.sim.population.props  # shortcut to population properties dataframe

        health_values_df = pd.DataFrame(index=df.index[df.is_alive])

        for symptom, daly_wt in self.parameters["daly_wts"].items():
            health_values_df.loc[
                self.sim.modules["SymptomManager"].who_has(symptom), symptom
            ] = daly_wt

        health_values_df.fillna(0, inplace=True)

        return health_values_df


class ChronicSyndromeEvent(RegularEvent, PopulationScopeEventMixin):

    # This event is occuring regularly at one monthly intervals

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        assert isinstance(module, ChronicSyndrome)

    def apply(self, population):
        df = population.props
        p = self.module.parameters
        rng: np.random.RandomState = self.module.rng

        # 1. get (and hold) index of currently infected and uninfected individuals
        # currently_cs = df.index[df.cs_has_cs & df.is_alive]
        currently_not_cs = df.index[~df.cs_has_cs & df.is_alive]

        # 2. handle new cases
        p_aq = p["p_acquisition_per_year"] / 12.0
        now_acquired = rng.random_sample(size=len(currently_not_cs)) < p_aq

        # if any are new cases
        if now_acquired.sum():
            newcases_idx = currently_not_cs[now_acquired]

            death_years_ahead = rng.exponential(scale=20, size=now_acquired.sum())
            death_td_ahead = pd.to_timedelta(death_years_ahead, unit="y")

            df.loc[newcases_idx, "cs_has_cs"] = True
            df.loc[newcases_idx, "cs_status"].values[:] = "C"
            df.loc[newcases_idx, "cs_date_acquired"] = self.sim.date
            df.loc[newcases_idx, "cs_scheduled_date_death"] = (
                self.sim.date + death_td_ahead
            )
            df.loc[newcases_idx, "cs_date_cure"] = pd.NaT

            # schedule death events for new cases
            for person_index in newcases_idx:
                death_event = ChronicSyndromeDeathEvent(self.module, person_index)
                self.sim.schedule_event(
                    death_event, df.at[person_index, "cs_scheduled_date_death"]
                )

            # Assign symptoms:
            for symp in self.module.parameters["prob_of_symptoms"]:
                # persons who will have symptoms (each can occur independently)
                persons_id_with_symp = np.array(newcases_idx)[
                    self.module.rng.rand(len(newcases_idx))
                    < self.module.parameters["prob_of_symptoms"][symp]
                ]

                self.sim.modules["SymptomManager"].change_symptom(
                    person_id=list(persons_id_with_symp),
                    symptom_string=symp,
                    add_or_remove="+",
                    disease_module=self.module,
                )

        # 3) Handle progression to severe symptoms
        curr_cs_but_not_craving_sandwiches = list(
            set(df.index[df.cs_has_cs & df.is_alive])
            - set(self.sim.modules["SymptomManager"].who_has("em_craving_sandwiches"))
        )

        become_severe = (
            self.module.rng.random_sample(size=len(curr_cs_but_not_craving_sandwiches))
            < p["prob_dev_severe_symptoms_per_year"] / 12
        )
        become_severe_idx = np.array(curr_cs_but_not_craving_sandwiches)[become_severe]

        self.sim.modules["SymptomManager"].change_symptom(
            person_id=list(become_severe_idx),
            symptom_string="em_craving_sandwiches",
            add_or_remove="+",
            disease_module=self.module,
        )


class ChronicSyndromeDeathEvent(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        # Apply checks to ensure that this death should occur
        if df.at[person_id, "cs_status"] == "C":
            # Fire the centralised death event:
            death = InstantaneousDeath(self.module, person_id, cause="ChronicSyndrome")
            self.sim.schedule_event(death, self.sim.date)


class ChronicSyndrome_LaunchOutreachEvent(Event, PopulationScopeEventMixin):
    """
    This is the event that is run by ChronicSyndrome and it is the Outreach Event.
    It will now submit the individual HSI events that occur when each individual is met.
    (i.e. Any large campaign that involves contct with individual is composed of many individual outreach events).
    """

    def __init__(self, module):
        super().__init__(module)
        assert isinstance(module, ChronicSyndrome)

    def apply(self, population):
        df = self.sim.population.props

        # Find the person_ids who are going to get the outreach
        gets_outreach = df.index[(df["is_alive"]) & (df["sex"] == "F")]
        for person_id in gets_outreach:
            # make the outreach event (let this disease module be alerted about it, and also Mockitis)
            outreach_event_for_individual = HSI_ChronicSyndrome_Outreach_Individual(
                self.module, person_id=person_id
            )

            self.sim.modules["HealthSystem"].schedule_hsi_event(
                outreach_event_for_individual,
                priority=1,
                topen=self.sim.date,
                tclose=None,
            )


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# Health System Interaction Events


class HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment(
    HSI_Event, IndividualScopeEventMixin
):
    """
    This is a Health System Interaction Event.
    It is the event when a person with the severe symptoms of chronic syndrome presents for emergency care
    and is immediately provided with treatment.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, ChronicSyndrome)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint["Over5OPD"] = 1  # This requires one out patient appt
        # the_appt_footprint['AccidentsandEmerg'] = 0  # Plus, an amount of resources similar to an A&E

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(
            "This is HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment: We are now ready to treat this person %d.",
            person_id,
        )
        logger.debug(
            "This is HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment: The squeeze-factor is %d.",
            squeeze_factor,
        )

        if squeeze_factor < 0.5:
            # If squeeze factor is not too large:
            logger.debug("Treatment will be provided.")
            df = self.sim.population.props
            treatmentworks = self.module.rng.rand() < self.module.parameters["p_cure"]

            if treatmentworks:
                df.at[person_id, "cs_has_cs"] = False
                df.at[person_id, "cs_status"] = "P"

                # (in this we nullify the death event that has been scheduled.)
                df.at[person_id, "cs_scheduled_date_death"] = pd.NaT
                df.at[person_id, "cs_date_cure"] = self.sim.date

                # remove all symptoms instantly
                self.sim.modules["SymptomManager"].clear_symptoms(
                    person_id=person_id, disease_module=self.module
                )
        else:
            # Squeeze factor is too large
            logger.debug("Treatment will not be provided due to squeeze factor.")

    def did_not_run(self):
        logger.debug(
            "HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment: did not run"
        )
        pass


class HSI_ChronicSyndrome_Outreach_Individual(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    This event can be used to simulate the occurrence of an 'outreach' intervention.

    NB. This needs to be created and run for each individual that benefits from the outreach campaign.

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, ChronicSyndrome)

        logger.debug("Outreach event being created.")

        # Define the necessary information for an HSI
        # (These are blank when created; but these should be filled-in by the module that calls it)
        self.TREATMENT_ID = "ChronicSyndrome_Outreach_Individual"

        # APPP_FOOTPRINT: outreach event takes small amount of time for DCSA
        appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        appt_footprint["ConWithDCSA"] = 0.5
        self.EXPECTED_APPT_FOOTPRINT = appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at facility-level 0
        self.ALERT_OTHER_DISEASES = ["*"]

    def apply(self, person_id, squeeze_factor):
        logger.debug("Outreach event running now for person: %s", person_id)

        # Do here whatever happens during an outreach event with an individual
        # ~~~~~~~~~~~~~~~~~~~~~~

        # Make request for some consumables
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables["Intervention_Pkg"]
                == "First line treatment for new TB cases for adults",
                "Intervention_Pkg_Code",
            ]
        )[0]
        pkg_code2 = pd.unique(
            consumables.loc[
                consumables["Intervention_Pkg"]
                == "MDR notification among previously treated patients",
                "Intervention_Pkg_Code",
            ]
        )[0]

        item_code1 = pd.unique(
            consumables.loc[
                consumables["Items"] == "Ketamine hydrochloride 50mg/ml, 10ml",
                "Item_Code",
            ]
        )[0]
        item_code2 = pd.unique(
            consumables.loc[consumables["Items"] == "Underpants", "Item_Code"]
        )[0]

        consumables_needed = {
            "Intervention_Package_Code": {pkg_code1: 1, pkg_code2: 4},
            "Item_Code": {item_code1: 1, item_code2: 10},
        }

        outcome_of_request_for_consumables = self.sim.modules[
            "HealthSystem"
        ].request_consumables(hsi_event=self, cons_req_as_footprint=consumables_needed)

        # answer comes back in the same format, but with quantities replaced with bools indicating availability
        if outcome_of_request_for_consumables["Intervention_Package_Code"][pkg_code1]:
            logger.debug("PkgCode1 is available, so use it.")
        else:
            logger.debug("PkgCode1 is not available, so can" "t use it.")

        # Return the actual appt footprints
        actual_appt_footprint = (
            self.EXPECTED_APPT_FOOTPRINT
        )  # The actual time take is double what is expected
        actual_appt_footprint["ConWithDCSA"] = actual_appt_footprint["ConWithDCSA"] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug("HSI_ChronicSyndrome_Outreach_Individual: did not run")
        pass


class HSI_ChronicSyndrome_PopulationWideBehaviourChange(
    HSI_Event, PopulationScopeEventMixin
):
    """
    This is a Population-Wide Health System Interaction Event - will change the variables to do with risk for
    ChronicSyndrome
    """

    def __init__(self, module):
        super().__init__(module)
        assert isinstance(module, ChronicSyndrome)

        # Define the necessary information for a Population level HSI
        self.TREATMENT_ID = "ChronicSyndrome_PopulationWideBehaviourChange"

    def apply(self, population, squeeze_factor):
        logger.debug("This is HSI_ChronicSyndrome_PopulationWideBehaviourChange")

        # As an example, we will reduce the chance of acquisition per year (due to behaviour change)
        self.module.parameters["p_acquisition_per_year"] = (
            self.module.parameters["p_acquisition_per_year"] * 0.5
        )


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------


class EpiLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """ There is no logging done here.
        """
        # run this event every month
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, ChronicSyndrome)

    def apply(self, population):
        pass
