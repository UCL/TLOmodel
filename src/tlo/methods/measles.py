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


class Measles(Module):
    """
        Sets up the baseline population with measles and reads in all necessary parameters
        measles vaccine is found in module epi
    """

    PARAMETERS = {
        "p_measles": Parameter(
            Types.REAL, "Probability that an uninfected individual becomes infected"
        ),

        # daly weights
        "daly_wt_moderate_measles": Parameter(
            Types.REAL, "DALY weights for moderate measles infection"
        ),
        "daly_wt_severe_measles": Parameter(Types.REAL, "DALY weights for severe measles"),
    }

    PROPERTIES = {
        "me_has_measles": Property(Types.BOOL, "Current status of measles"),
    }

    # Declaration of the symptoms that this module will use
    SYMPTOMS = {
        "spots",  # will trigger healthcare seeking behaviour
        "em_lots_of_spots",  # will trigger emergency care
    }

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)+
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        For now, we are going to hard code them explicity
        """
        p = self.parameters

        p["p_measles"] = 0.10

        if "HealthBurden" in self.sim.modules.keys():
            p["daly_wt_moderate_measles"] = self.sim.modules["HealthBurden"].get_daly_weight(
                205
            )
            p["daly_wt_severe_measles"] = self.sim.modules["HealthBurden"].get_daly_weight(
                206
            )

        # ---- Register this module ----
        # Register this disease module with the health system
        self.sim.modules["HealthSystem"].register_disease_module(self)

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """
        df = (
            population.props
        )  # a shortcut to the dataframe storing data for individiuals
        p = self.parameters

        # Set default for properties
        df.loc[df.is_alive, "cs_has_cs"] = False  # default: no individuals infected
        df.loc[df.is_alive, "cs_status"].values[:] = "N"  # default: never infected
        df.loc[df.is_alive, "cs_date_acquired"] = pd.NaT  # default: not a time
        df.loc[df.is_alive, "cs_scheduled_date_death"] = pd.NaT  # default: not a time
        df.loc[df.is_alive, "cs_date_cure"] = pd.NaT  # default: not a time

        # randomly selected some individuals as infected
        num_alive = df.is_alive.sum()
        df.loc[df.is_alive, "cs_has_cs"] = (
            self.rng.random_sample(size=num_alive) < p["initial_prevalence"]
        )
        df.loc[df.cs_has_cs, "cs_status"].values[:] = "C"

        # Assign time of infections and dates of scheduled death for all those infected
        # get all the infected individuals
        acquired_count = df.cs_has_cs.sum()

        # Assign level of symptoms to each person with cd:
        person_id_all_with_cs = list(df[df.cs_has_cs].index)

        for symp in self.parameters["prob_of_symptoms"]:
            # persons who will have symptoms (each can occur independently)
            persons_id_with_symp = np.array(person_id_all_with_cs)[
                self.rng.rand(len(person_id_all_with_cs))
                < self.parameters["prob_of_symptoms"][symp]
            ]

            self.sim.modules["SymptomManager"].change_symptom(
                person_id=list(persons_id_with_symp),
                symptom_string=symp,
                add_or_remove="+",
                disease_module=self,
            )

        # date acquired cs
        # sample years in the past
        acquired_years_ago = self.rng.exponential(scale=10, size=acquired_count)

        # pandas requires 'timedelta' type for date calculations
        acquired_td_ago = pd.to_timedelta(acquired_years_ago, unit="y")

        # date of death of the infected individuals (in the future)
        death_years_ahead = self.rng.exponential(scale=20, size=acquired_count)
        death_td_ahead = pd.to_timedelta(death_years_ahead, unit="y")

        # set the properties of infected individuals
        df.loc[df.cs_has_cs, "cs_date_acquired"] = self.sim.date - acquired_td_ago
        df.loc[df.cs_has_cs, "cs_scheduled_date_death"] = self.sim.date + death_td_ahead

    def initialise_simulation(self, sim):

        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event (we will implement below)
        event = ChronicSyndromeEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=1))

        # add an event to log to screen
        sim.schedule_event(
            ChronicSyndromeLoggingEvent(self), sim.date + DateOffset(months=6)
        )

        # # add the death event of individuals with ChronicSyndrome
        df = (
            sim.population.props
        )  # a shortcut to the dataframe storing data for individiuals
        indicies_of_persons_who_will_die = df.index[df.cs_has_cs]
        for person_index in indicies_of_persons_who_will_die:
            death_event = ChronicSyndromeDeathEvent(self, person_index)
            self.sim.schedule_event(
                death_event, df.at[person_index, "cs_scheduled_date_death"]
            )

        # Schedule the event that will launch the Outreach event
        outreach_event = ChronicSyndrome_LaunchOutreachEvent(self)
        self.sim.schedule_event(outreach_event, self.sim.date + DateOffset(months=6))

        # Schedule the occurance of a population wide change in risk that goes through the health system:
        popwide_hsi_event = HSI_ChronicSyndrome_PopulationWideBehaviourChange(self)
        self.sim.modules["HealthSystem"].schedule_hsi_event(
            popwide_hsi_event, priority=1, topen=self.sim.date, tclose=None
        )
        logger.debug("The population wide HSI event has been scheduled succesfully!")

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
