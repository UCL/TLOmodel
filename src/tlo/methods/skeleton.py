"""
A skeleton template for disease methods.
"""

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent


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

    And, if the module represents a disease:
    * It must register itself: self.sim.modules['HealthSystem'].register_disease_module(self)
    * `query_symptoms_now(self)`
    * `report_qaly_values(self)`
    * `on_healthsystem_interaction(self, person_id, cue_type=None, disease_specific=None)`

    If this module represents a form of treatment:
    * TREATMENT_ID: must be defined
    * It must register the treatment: self.sim.modules['HealthSystem'].register_interventions(footprint_for_treatment)
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
    PROPERTIES = {
        'property_a': Property(Types.BOOL, 'Description of property a'),
    }

    # Declaration of how we will refer to any treatments that are related to this disease.
    TREATMENT_ID = ''

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
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

        If this is a disease module, register this disease module with the healthsystem:
        e.g. self.sim.modules['HealthSystem'].register_disease_module(self)"

        If this is an interveton module: register the footprints with the healthsystem:
        e.g.    footprint_for_treatment = pd.DataFrame(index=np.arange(1), data={
                 'Name': self.TREATMENT_ID,
                 'Nurse_Time': 5,
                 'Doctor_Time': 10,
                 'Electricity': False,
                 'Water': False})
             self.sim.modules['HealthSystem'].register_interventions(footprint_for_treatment)

        """

        raise NotImplementedError

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        raise NotImplementedError

    def query_symptoms_now(self):
        """
        If this is a registered disease module, this is called by the HealthCareSeekingPoll in order to determine the
        healthlevel of each person. It can be called at any time and must return a Series with length equal to the
        number of persons alive and index matching sim.population.props. The entries encode the symptoms on the
        following "unified symptom scale":
        0=None; 1=Mild; 2=Moderate; 3=Severe; 4=Extreme_Emergency
        """

        raise NotImplementedError

    def report_qaly_values(self):
        """
        If this is a registered disease module, this is called periodically by the QALY module in order to compute the
        total 'Quality of Life' for all alive persons. Each disease module must return a Series with length equal to the
        number of persons alive and index matching sim.population.props. The entries encode a QALY weight, between zero
        and 1, which summarise the quality of life for that persons for the total of the past 12 months. Note that this
        can be called at any time.

        Disease modules should look-up the weights to use by calling QALY.get_qaly_weight(sequaluecode). The sequalue
        code to use can be found in the ResourceFile_DALYWeights. ie. Find the appropriate sequalue in that file, and
        then hard-code the sequale code in this call.
        e.g. p['qalywt_mild_sneezing'] = self.sim.modules['QALY'].get_qaly_weight(50)

        """

        raise NotImplementedError

    def on_healthsystem_interaction(self, person_id, cue_type=None, disease_specific=None):
        """
        If this is a registered disease module, this is called whenever there is any interaction between an individual
        and the healthsystem. All disease modules are notified of all interactions with the healthsystem but can choose
        if they will respond by looking at the arguments that are passed.

        * cue_type: determines what has caused the interaction and can be "HealthCareSeekingPoll", "OutreachEvent",
            "InitialDiseaseCall" or "FollowUp".
        * disease_specific: determines if this interaction has been triggered by, or is otherwise intended to be,
            specfifc to a particular disease. If will either take the value None or the name of the registered disease
            module.


        """
        pass


class SkeletonEvent(RegularEvent, PopulationScopeEventMixin):
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

    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """
        raise NotImplementedError
