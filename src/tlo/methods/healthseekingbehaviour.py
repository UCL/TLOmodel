"""
A skeleton template for disease methods.

"""

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.hsi_generic_first_appts import HSI_GenericFirstApptAtFacilityLevel1
from tlo.population import logger

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

class HealthSeekingBehaviour(Module):
    """
    This modules determines if the onset of generic symptoms will lead to that person presenting at the health
    facility for a HSI_GenericFirstAppointment.

    """

    # No parameters to declare
    PARAMETERS = {}

    # No properties to declare
    PROPERTIES = {}

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        To access files use: Path(self.resourcefilepath) / file_name
        """
        pass

    def initialise_population(self, population):
        """Nothing to initialise in the population
        """
        #
        pass

    def initialise_simulation(self, sim):
        """Initialise the simulation: set the first occurance of the repeating HealthSeekingBehaviourPoll
        """
        sim.schedule_event(HealthSeekingBehaviourPoll(self),sim.date)


    def on_birth(self, mother_id, child_id):
        """Nothing to handle on_birth
        """
        pass


# ---------------------------------------------------------------------------------------------------------
#   REGULAR POLLING EVENT
# ---------------------------------------------------------------------------------------------------------

class HealthSeekingBehaviourPoll(RegularEvent, PopulationScopeEventMixin):
    """This event occurs every day and determines if persons with newly onset acute generic symptoms will seek care.

    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module):
        """Initialise the HealthSeekingBehaviourPoll

        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.

        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(days=1))
        assert isinstance(module, HealthSeekingBehaviour)

    def apply(self, population):
        """Determine if persons with newly onset acute generic symptoms will seek care

        :param population: the current population
        """

        # get the list of person_ids who have onset generic acute symptoms in the last day
        person_ids_with_new_symptoms = self.module.sim.modules['SymptomManager'].persons_with_newly_onset_acute_generic_symptoms

        # clear the list of person_ids with onset generic acute symptoms (as now dealt with here)
        self.module.sim.modules['SymptomManager'].persons_with_newly_onset_acute_generic_symptoms= list()

        for person_id in person_ids_with_new_symptoms:
            # For each individual person_id, look at the symptoms and determine if will seek care

            symptom_profile = self.sim.population.props.loc[person_id,self.sim.population.props.columns.str.startswith('sy_')]

            coefficients = [0.05, 0.1, - 0.5]
            intercept = 0.02
            #prob_seeking_care = 0.02 + sy_a * 0.05 + sy_b * 0.1 + sy_c * -0.5

            prob_seeking_care = max(0.0, min(1.0, intercept + sum(symptom_profile.values * coefficients)))

            if prob_seeking_care > self.module.rng.rand():
                # Create HSI_GenericFirstAppt for this person to represent them presenting at the facility
                # NB. Here we can specifify which type of facility they would attend if we need to

                hsi_genericfirstappt = HSI_GenericFirstApptAtFacilityLevel1(self.module, person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(hsi_genericfirstappt,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=None)





