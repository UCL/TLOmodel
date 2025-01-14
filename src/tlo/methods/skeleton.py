"""
A skeleton template for disease methods.

"""
from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.hsi_event import HSI_Event

from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Skeleton(Module):
    """
    One line summary goes here...

    If it another kind of module use base class Module

    All disease modules need to be implemented as a class inheriting from Module.
    They need to provide several methods which will be called by the simulation
    framework:

        - `read_parameters(data_folder)`
        - `initialise_population(population)`
        - `initialise_simulation(sim)`
        - `on_birth(mother, child)`
        - `on_hsi_alert(person_id, treatment_id)` [If this is disease module]
        -  `report_daly_values()` [If this is disease module]
    """

    # Declares modules that need to be registered in simulation and initialised before
    # this module
    INIT_DEPENDENCIES = {'Demography'}

    # Declares optiona;l modules that need to be registered in simulation and
    # initialised before this module if present
    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden'}

    # Declares any modules that need to be registered in simulation in addition to those
    # in INIT_DEPENDENCIES to allow running simulation
    ADDITIONAL_DEPENDENCIES = {'HealthSystem'}

    # Declare Metadata (this is for a typical 'Disease Module')
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'tlo_name_of_a_cause_of_death_in_this_module':
            Cause(gbd_causes={'set_of_strings_of_gbd_causes_to_which_this_cause_corresponds'},
                  label='the_category_of_which_this_cause_is_a_part')
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'tlo_name_of_a_cause_of_disability_in_this_module':
            Cause(gbd_causes={'set_of_strings_of_gbd_causes_to_which_this_cause_corresponds'},
                  label='the_category_of_which_this_cause_is_a_part')
    }

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'parameter_a': Parameter(
            Types.REAL, 'Description of parameter a'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.

    # Note that all properties must have a two letter prefix that identifies them to this module.

    PROPERTIES = {
        'sk_property_a': Property(Types.BOOL, 'Description of property a'),
    }

    # Declare the non-generic symptoms that this module will use.
    # It will not be able to use any that are not declared here. They do not need to be unique to this module.
    # You should not declare symptoms that are generic here (i.e. in the generic list of symptoms)
    SYMPTOMS = {}

    def __init__(self, name=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.store = {'Proportion_infected': []}

    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        """Read parameter values from file, if required.
        To access files use: Path(self.resourcefilepath) / file_name
        """
        pass

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """
        raise NotImplementedError

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.

        """
        raise NotImplementedError

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        raise NotImplementedError

    def report_daly_values(self):
        """
        This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        experienced by persons in the previous month. Only rows for alive-persons must be returned.
        If multiple causes in CAUSES_OF_DISABILITY are defined, a pd.DataFrame must be returned with a column
        corresponding to each cause (but if only one cause in CAUSES_OF_DISABILITY is defined, the pd.Series does not
        need to be given a specific name).

        To return a value of 0.0 (fully health) for everyone, use:
        df = self.sim.population.props
        return pd.Series(index=df.index[df.is_alive],data=0.0)
        """
        raise NotImplementedError

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
#
#   These are the events which drive the simulation of the disease. It may be a regular event that updates
#   the status of all the population of subsections of it at one time. There may also be a set of events
#   that represent disease events for particular persons.
# ---------------------------------------------------------------------------------------------------------

class Skeleton_Event(RegularEvent, PopulationScopeEventMixin):
    """A skeleton class for an event

    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module):
        """One line summary here

        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.

        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=1))
        assert isinstance(module, Skeleton)

    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
#
#   Put the logging events here. There should be a regular logger outputting current states of the
#   population. There may also be a logging event that is driven by particular events.
# ---------------------------------------------------------------------------------------------------------

class Skeleton_LoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summary of the numbers of people with respect to the action of this module.
        This is a regular event that can output current states of people or cumulative events since last logging event.
        """

        # run this event every year
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, Skeleton)

    def apply(self, population):
        # Make some summary statistics

        dict_to_output = {
            'Metric_One': 1.0,
            'Metric_Two': 2.0
        }

        logger.info(key='summary_12m', data=dict_to_output)


# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
#
#   Here are all the different Health System Interactions Events that this module will use.
# ---------------------------------------------------------------------------------------------------------

class HSI_Skeleton_Example_Interaction(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event. An interaction with the healthsystem are encapsulated in events
    like this.
    It must begin HSI_<Module_Name>_Description
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Skeleton)

        # Define the appointments types needed
        the_appt_footprint = self.make_appt_footprint({'Over5OPD': 1})  # This requires one adult out-patient appt.

        # Define the facilities at which this event can occur (only one is allowed)
        # Choose from: list(pd.unique(self.sim.modules['HealthSystem'].parameters['Facilities_For_Each_District']
        #                            ['Facility_Level']))
        the_accepted_facility_level = 0

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Skeleton_Example_Interaction'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """
        Do the action that take place in this health system interaction, in light of prevailing conditions in the
        healthcare system:

            * squeeze_factor (an argument provided to the event) indicates the extent to which this HSI_Event is being
              run in the context of an over-burdened healthcare facility.
            * bed_days_allocated_to_this_event (a property of the event) indicates the number and types of bed-days
              that have been allocated to this event.

        Can return an updated APPT_FOOTPRINT if this differs from the declaration in self.EXPECTED_APPT_FOOTPRINT

        To request consumables use - self.get_all_consumables(item_codes=my_item_codes)
        """
        pass

    def did_not_run(self):
        """
        Do any action that is necessary each time when the health system interaction is not run.
        This is called each day that the HSI is 'due' but not run due to insufficient health system capabilities.
        Return False to cause this HSI event not to be rescheduled and to therefore never be run.
        (Returning nothing or True will cause this event to be rescheduled for the next day.)
        """
        pass

    def never_ran(self):
        """
        Do any action that is necessary when it is clear that the HSI event will never be run. This will occur if it
        has not run and the simulation date has passed the date 'tclose' by which the event was scheduled to have
        occurred.
        Do not return anything.
        """
        pass
