"""
Health Seeking Behaviour Module
This module determines if care is sought once a symptom is developed.
"""
import numpy as np

from tlo import DateOffset, Module
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.methods.hsi_generic_first_appts import (
    HSI_GenericEmergencyFirstApptAtFacilityLevel1,
    HSI_GenericFirstApptAtFacilityLevel1,
)

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------
from tlo.util import choose_outcome_from_a_dict


class HealthSeekingBehaviour(Module):
    """
    This modules determines if the onset of generic symptoms will lead to that person presenting at the health
    facility for a HSI_GenericFirstAppointment.

    """

    # No parameters to declare
    PARAMETERS = {}

    # No properties to declare
    PROPERTIES = {}

    def __init__(self, name=None, resourcefilepath=None, force_any_symptom_to_lead_to_healthcareseeking=False):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # "force_any_symptom_to_lead_to_healthcareseeking"=True will mean that probability of health care seeking is 1.0
        # for anyone with newly onset symptoms
        assert isinstance(force_any_symptom_to_lead_to_healthcareseeking, bool)
        self.force_any_symptom_to_lead_to_healthcareseeking = force_any_symptom_to_lead_to_healthcareseeking

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
        sim.schedule_event(HealthSeekingBehaviourPoll(self), sim.date)

    def on_birth(self, mother_id, child_id):
        """Nothing to handle on_birth
        """
        pass


# ---------------------------------------------------------------------------------------------------------
#   REGULAR POLLING EVENT
# ---------------------------------------------------------------------------------------------------------

class HealthSeekingBehaviourPoll(RegularEvent, PopulationScopeEventMixin):
    """This event occurs every day and determines if persons with newly onset symptoms will seek care.
    """

    def __init__(self, module):
        """Initialise the HealthSeekingBehaviourPoll
        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(days=1))
        assert isinstance(module, HealthSeekingBehaviour)

    def apply(self, population):
        """Determine if persons with newly onset acute generic symptoms will seek care.

        :param population: the current population
        """

        # get the list of person_ids who have onset generic acute symptoms in the last day, extracting any person_ids
        #    that have died (since the onset of symptoms)
        alive_person_ids = list(self.sim.population.props.index[self.sim.population.props.is_alive])
        person_ids_with_new_symptoms = list(
            self.module.sim.modules['SymptomManager'].persons_with_newly_onset_symptoms.
                intersection(alive_person_ids)
        )

        # clear the list of person_ids with newly onset symptoms
        self.module.sim.modules['SymptomManager'].persons_with_newly_onset_symptoms = set()

        for person_id in person_ids_with_new_symptoms:
            # For each individual person_id, with at least one new onset symptom, look at the symptoms and determine if
            # will seek care.
            # This is run looking at all symptoms even if only one is newly onset.
            # If one symptom is an 'emergency symptom' a generic emergency appointment is scheduled
            #   (and no non-emergency appointment)

            symps_this_person = self.module.sim.modules['SymptomManager'].has_what(person_id)

            # ~~~~~~ HEALTH CARE SEEKING IN RESPONSE TO EMERGENCY SYMPTOMS ~~~~~~~~
            if any([s.startswith('em_') for s in symps_this_person]):
                hsi_genericemergencyfirstappt = HSI_GenericEmergencyFirstApptAtFacilityLevel1(self.module,
                                                                                              person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(hsi_genericemergencyfirstappt,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=None)
            else:
                # ~~~~~~ HEALTH CARE SEEKING IN RESPONSE TO GENERIC NON-EMERGENCY SYMPTOMS ~~~~~~~~
                person_profile = self.sim.population.props.loc[person_id]

                # Build up the RHS of the logistic regresssion equation: 'f' is the linear term f(beta*x +... )
                # collate indicator variables to match the HSB equation (from Ng'ambi et al)
                f = np.log(3.237729)  # 'Constant' term from STATA is the baseline odds.

                # Region
                if person_profile['region_of_residence'] == 'Northern':
                    f += np.log(1.00)
                elif person_profile['region_of_residence'] == 'Central':
                    f += np.log(0.61)
                elif person_profile['region_of_residence'] == 'Southern':
                    f += np.log(0.67)
                else:
                    raise Exception('region_of_residence not recognised')

                # Urban/Rural residence
                if person_profile['li_urban'] is False:
                    f += np.log(1.00)
                else:
                    f += np.log(1.63)

                # Sex
                if person_profile['sex'] == 'M':
                    f += np.log(1.00)
                else:
                    f += np.log(1.19)

                # Age-group
                if person_profile['age_years'] < 5:
                    f += np.log(1.00)
                elif person_profile['age_years'] < 15:
                    f += np.log(0.64)
                elif person_profile['age_years'] < 35:
                    f += np.log(0.51)
                elif person_profile['age_years'] < 60:
                    f += np.log(0.54)
                else:
                    f += np.log(0.44)

                # Year
                # - not encoded: this effect ignored

                # Chronic conditions
                # - not encoded: awaiting suitable variable to include.
                #   (effect size = 1.44 if pre-existing chronic_condition)

                # Symptoms (testing for empty or non-empty set) - (can have more than one)
                if person_profile['sy_fever']:
                    f += np.log(1.86)

                if person_profile['sy_vomiting']:
                    f += np.log(1.28)

                if (person_profile['sy_stomachache']) or (person_profile['sy_diarrhoea']):
                    f += np.log(0.76)

                if person_profile['sy_sore_throat']:
                    f += np.log(0.89)

                if person_profile['sy_respiratory_symptoms']:
                    f += np.log(0.71)

                if person_profile['sy_headache']:
                    f += np.log(0.52)

                if person_profile['sy_skin_complaint']:
                    f += np.log(2.31)

                if person_profile['sy_dental_complaint']:
                    f += np.log(0.94)

                if person_profile['sy_backache']:
                    f += np.log(1.01)

                if person_profile['sy_injury']:
                    f += np.log(1.02)

                if person_profile['sy_eye_complaint']:
                    f += np.log(1.33)

                # TODO - symptom that are specific to a disease module need to be registered, or otherwise safely
                #  handled here
                def optional_symptom(name, value):
                    if name in person_profile:
                        return value
                    return 0

                # arbitrarily large value for health seeking for dysphagia)
                f += optional_symptom('sy_dysphagia', np.log(4.00))

                # convert into a probability of seeking care:
                prob_seeking_care = 1 / (1 + np.exp(-f))

                if self.module.force_any_symptom_to_lead_to_healthcareseeking or (self.module.rng.rand() < prob_seeking_care):
                    # Create HSI_GenericFirstAppt for this person to represent them presenting at the facility
                    # NB. Here we can specify which type of facility they would attend if we need to

                    delay_to_seeking_care_in_days = self.module.rng.randint(0, 1)  # Uniform interval 0-1 days
                    date_of_seeking_care = self.sim.date + DateOffset(days=delay_to_seeking_care_in_days)

                    hsi_genericfirstappt = HSI_GenericFirstApptAtFacilityLevel1(self.module, person_id=person_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(hsi_genericfirstappt,
                                                                        priority=0,
                                                                        topen=date_of_seeking_care,
                                                                        tclose=None)
