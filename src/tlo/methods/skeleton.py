"""
A skeleton template for disease methods.

"""

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.healthsystem import HSI_Event

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Skeleton(Module):
    """
    One line summary goes here...

    All disease modules need to be implemented as a class inheriting from Module.
    They need to provide several methods which will be called by the simulation
    framework:
    * `read_parameters(data_folder)`
    * `initialise_population(population)`
    * `initialise_simulation(sim)`
    * `on_birth(mother, child)`
    * `on_hsi_alert(person_id, treatment_id)` [If this is disease module]
    *  `report_daly_values()` [If this is disease module]

    """

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

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.store = {'Proportion_infected': []}

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        To access files use: Path(self.resourcefilepath) / file_name
        """
        pass

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        If this is a disease module, register this disease module with the healthsystem:
        self.sim.modules['HealthSystem'].register_disease_module(self)

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
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        # To return a value of 0.0 (fully health) for everyone, use:
        # df = self.sim.popultion.props
        # return pd.Series(index=df.index[df.is_alive],data=0.0)

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
#   population. There may also be a loggig event that is driven by particular events.
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
        # Make some summary statitics

        dict_to_output = {
            'Metric_One': 1.0,
            'Metric_Two': 2.0
        }

        logger.info('%s|summary_12m|%s', self.sim.date, dict_to_output)


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

        # Define the call on resources of this treatment event: Time of Officers (Appointments)
        #   - get an 'empty' footprint:
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        #   - update to reflect the appointments that are required
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient

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
        Do the action that take place in this health system interaction, in light of squeeze_factor
        Can reutrn an updated APPT_FOOTPRINT if this differs from the declaration in self.EXPECTED_APPT_FOOTPRINT
        """
        pass

    def did_not_run(self):
        """
        Do any action that is neccessary when the health system interaction is not run.
        This is called each day that the HSI is 'due' but not run due to insufficient health system capabilities.
        Return False to cause this HSI event not to be rescheduled and to therefore never be run.
        (Returning nothing or True will cause this event to be rescheduled for the next day.)
        """
        pass
