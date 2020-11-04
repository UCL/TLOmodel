"""
Health Seeking Behaviour Module
This module determines if care is sought once a symptom is developed.

The write-up of these estimates is: Health-seeking behaviour estimates for adults and children.docx

"""
from pathlib import Path

import pandas as pd

from tlo import DateOffset, Module, Parameter, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.hsi_generic_first_appts import (
    HSI_GenericEmergencyFirstApptAtFacilityLevel1,
    HSI_GenericFirstApptAtFacilityLevel1,
)

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------


class HealthSeekingBehaviour(Module):
    """
    This modules determines if the onset of symptoms will lead to that person presenting at the health
    facility for a HSI_GenericFirstAppointment.

    An equation gives the probability of seeking care in response to the "average" symptom. This is modified according
    to if the symptom is associated with a particular effect.

    """

    # Declare Metadata
    METADATA = {Metadata.USES_HEALTHSYSTEM}

    # No parameters to declare
    PARAMETERS = {
        'baseline_odds_of_healthcareseeking_children': Parameter(Types.REAL, 'odds of health-care seeking (children:'
                                                                             ' 0-14) if male, 0-5 years-old, living in'
                                                                             ' a rural setting in the Northern region,'
                                                                             ' and not in the wealth categories 4 or '
                                                                             '5'),
        'odds_ratio_children_sex_Female': Parameter(Types.REAL, 'odds ratio for health-care seeking (children) if sex'
                                                                ' is Female'),
        'odds_ratio_children_age_5to14': Parameter(Types.REAL, 'odds ratio for health-care seeking (children) if aged'
                                                               ' 5-14'),
        'odds_ratio_children_setting_urban': Parameter(Types.REAL, 'odds ratio for health-care seeking (children) if'
                                                                   ' setting is Urban'),
        'odds_ratio_children_region_Central': Parameter(Types.REAL, 'odds ratio for health-care seeking (children) if'
                                                                    ' region is Central'),
        'odds_ratio_children_region_Southern': Parameter(Types.REAL, 'odds ratio for health-care seeking (children) if'
                                                                     ' region is Southern'),
        'odds_ratio_children_wealth_higher': Parameter(Types.REAL, 'odds ratio for health-care seeking (children) if '
                                                                   'wealth is in categories 4 or 5'),
        'baseline_odds_of_healthcareseeking_adults': Parameter(Types.REAL, 'odds of health-care seeking (adults: 15+) '
                                                                           'if male, 15-34 year-olds, living in a rural'
                                                                           ' setting in the Northern region, and not in'
                                                                           ' the wealth categories 4 or 5.'),
        'odds_ratio_adults_sex_Female': Parameter(Types.REAL, 'odds ratio for health-care seeking (adults) if sex is'
                                                              ' Female'),
        'odds_ratio_adults_age_35to59': Parameter(Types.REAL, 'odds ratio for health-care seeking (adults) if aged'
                                                              ' 35-59'),
        'odds_ratio_adults_age_60plus': Parameter(Types.REAL, 'odds ratio for health-care seeking (adults) if aged'
                                                              ' 60+'),
        'odds_ratio_adults_setting_urban': Parameter(Types.REAL, 'odds ratio for health-care seeking (adults) if '
                                                                 'setting is Urban'),
        'odds_ratio_adults_region_Central': Parameter(Types.REAL, 'odds ratio for health-care seeking (adults) if '
                                                                  'region is Central'),
        'odds_ratio_adults_region_Southern': Parameter(Types.REAL, 'odds ratio for health-care seeking (adults) if '
                                                                   'region is Southern'),
        'odds_ratio_adults_wealth_higher': Parameter(Types.REAL, 'odds ratio for health-care seeking (adults) if wealth'
                                                                 ' is in categories 4 or 5'),
        'max_days_delay_to_generic_HSI_after_symptoms': Parameter(Types.INT,
                                                                  'Maximum days delay between symptom onset and first'
                                                                  'generic HSI. Actual delay is sample between 0 and '
                                                                  'this value.')
    }

    # No properties to declare
    PROPERTIES = {}

    def __init__(self, name=None, resourcefilepath=None, force_any_symptom_to_lead_to_healthcareseeking=False):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        self.hsb = dict()
        self.odds_ratio_health_seeking_in_children = dict()
        self.odds_ratio_health_seeking_in_adults = dict()
        self.no_healthcareseeking_in_children = set()
        self.emergency_in_children = set()
        self.no_healthcareseeking_in_adults = set()
        self.emergency_in_adults = set()

        # "force_any_symptom_to_lead_to_healthcareseeking"=True will mean that probability of health care seeking is 1.0
        # for anyone with newly onset symptoms
        assert isinstance(force_any_symptom_to_lead_to_healthcareseeking, bool)
        self.force_any_symptom_to_lead_to_healthcareseeking = force_any_symptom_to_lead_to_healthcareseeking

    def read_parameters(self, data_folder):
        """Construct the LinearModel for healthcare seeking of the 'average symptom'.
        This is from the Ng'ambia et al.
        """

        # Load parameters from resource file:
        self.load_parameters_from_dataframe(
            pd.DataFrame(pd.read_csv(Path(self.resourcefilepath) / 'ResourceFile_HealthSeekingBehaviour.csv'))
        )

        # Define the LinearModel for health seeking behaviour for the 'average symptom'
        p = self.parameters

        self.hsb['children'] = LinearModel(
            LinearModelType.LOGISTIC,
            p['baseline_odds_of_healthcareseeking_children'],
            Predictor('li_urban').when(True, p['odds_ratio_children_setting_urban']),
            Predictor('sex').when('F', p['odds_ratio_children_sex_Female']),
            Predictor('age_years').when('>=5', p['odds_ratio_children_age_5to14']),
            Predictor('region_of_residence').when('Central', p['odds_ratio_children_region_Central'])
                                            .when('Southern', p['odds_ratio_children_region_Southern']),
            Predictor('li_wealth').when(4, p['odds_ratio_children_wealth_higher'])
                                  .when(5, p['odds_ratio_children_wealth_higher'])
        )

        self.hsb['adults'] = LinearModel(
            LinearModelType.LOGISTIC,
            p['baseline_odds_of_healthcareseeking_adults'],
            Predictor('li_urban').when(True, p['odds_ratio_adults_setting_urban']),
            Predictor('sex').when('F', p['odds_ratio_adults_sex_Female']),
            Predictor('age_years').when('.between(35,59)', p['odds_ratio_adults_age_35to59'])
                                  .when('>=60', p['odds_ratio_adults_age_60plus']),
            Predictor('region_of_residence').when('Central', p['odds_ratio_adults_region_Central'])
                                            .when('Southern', p['odds_ratio_adults_region_Southern']),
            Predictor('li_wealth').when(4, p['odds_ratio_adults_wealth_higher'])
                                  .when(5, p['odds_ratio_adults_wealth_higher'])
        )

    def initialise_population(self, population):
        """Nothing to initialise in the population
        """
        pass

    def initialise_simulation(self, sim):
        """
        * set the first occurrence of the repeating HealthSeekingBehaviourPoll
        * assemble the health-care seeking information from the symptoms that have been registered
        """

        # Schedule the HealthSeekingBehaviourPoll
        sim.schedule_event(HealthSeekingBehaviourPoll(self), sim.date)

        # Assemble the health-care seeking information from the symptoms that have been registered
        for symptom in self.sim.modules['SymptomManager'].all_registered_symptoms:
            # Children:
            if symptom.no_healthcareseeking_in_children:
                self.no_healthcareseeking_in_children.add(symptom.name)
            elif symptom.emergency_in_children:
                self.emergency_in_children.add(symptom.name)
            else:
                self.odds_ratio_health_seeking_in_children[symptom.name] = symptom.odds_ratio_health_seeking_in_children

            # Adults:
            if symptom.no_healthcareseeking_in_adults:
                self.no_healthcareseeking_in_adults.add(symptom.name)
            elif symptom.emergency_in_adults:
                self.emergency_in_adults.add(symptom.name)
            else:
                self.odds_ratio_health_seeking_in_adults[symptom.name] = symptom.odds_ratio_health_seeking_in_adults

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
        m = self.module
        df = population.props
        symptom_manager = self.sim.modules["SymptomManager"]
        health_system = self.sim.modules["HealthSystem"]

        # get rows of alive persons with new symptoms
        alive_with_symptoms = df.loc[sorted(symptom_manager.persons_with_newly_onset_symptoms), 'is_alive']
        selected_persons = df.loc[alive_with_symptoms[alive_with_symptoms].index]

        # clear the list of persons with newly onset symptoms
        symptom_manager.persons_with_newly_onset_symptoms.clear()

        # calculate the baseline probability for adults and children to seek care & put them in single series
        # (index remains the person_id)
        baseline_prob_child = m.hsb['children'].predict(selected_persons[selected_persons.age_years < 15])
        baseline_prob_adult = m.hsb['adults'].predict(selected_persons[selected_persons.age_years >= 15])
        baseline_prob = baseline_prob_child.append(baseline_prob_adult, verify_integrity=True).rename('baseline_prob')

        # this replaces calls to `symptom_manager.has_what(person_id)` inside the loop. get all the symptoms for
        # all persons and skip the checks/options in the `has_what()` method
        # todo: move this to symptom manager i.e. have_what(person_ids) (but better name)
        persons_symptoms = selected_persons.apply(
            lambda p: [s for s in symptom_manager.symptom_names if p[f'sy_{s}'] > 0], axis=1
        ).rename('symptoms')

        # make dataframe for processing below:
        # person_id (index), baseline_prob, age_years < 15 (i.e. is child), symptoms (list object)
        persons = pd.concat([baseline_prob, selected_persons.age_years < 15, persons_symptoms], axis=1)

        # For each individual person_id, with at least one new onset symptom,
        # look at the symptoms and determine if will seek care.
        # This is run looking at all symptoms even if only one is newly onset.
        for person_id, person in persons.iterrows():
            symptoms = person['symptoms']
            is_child = person['age_years']

            # determine whether symptoms cause person to seek care
            seek_emergency_care = False
            seek_non_emergency_care = False

            # Look at joint effect on health-care seeking of all symptoms. For a given symptoms list (of symptoms
            # that do causes care seeking), get the associated odds ratio for health seeking behaviour: this is the
            # product of the odds ratio associated with each symptom.
            care_seeking_odds = 1

            if is_child:
                for symptom in symptoms:
                    if symptom in m.emergency_in_children:
                        seek_emergency_care = True
                    elif symptom not in m.no_healthcareseeking_in_children:
                        seek_non_emergency_care = True
                        care_seeking_odds *= m.odds_ratio_health_seeking_in_children[symptom]
            else:
                for symptom in symptoms:
                    if symptom in m.emergency_in_adults:
                        seek_emergency_care = True
                    elif symptom not in m.no_healthcareseeking_in_adults:
                        seek_non_emergency_care = True
                        care_seeking_odds *= m.odds_ratio_health_seeking_in_adults[symptom]

            # Generate an emergency HSI, if any of the symptoms is an emergency
            if seek_emergency_care:
                health_system.schedule_hsi_event(
                    HSI_GenericEmergencyFirstApptAtFacilityLevel1(m, person_id=person_id),
                    priority=0,
                    topen=self.sim.date,
                    tclose=None
                )

            # if any non-emergency symptoms - consider making a generic non-emergency HSI:
            if seek_non_emergency_care:
                # For a baseline probability and an effect odds ratio, compute probability that is implied
                baseline_odds = person['baseline_prob'] / (1 - person['baseline_prob'])
                odds = baseline_odds * care_seeking_odds
                prob_hsb = odds / (1 + odds)

                if (m.rng.rand() < prob_hsb) or m.force_any_symptom_to_lead_to_healthcareseeking:
                    # Schedule a generic non-emergency appointment. Occurs after a delay of 0-4 days, or immediately
                    # if using 'force_any_symptom_to_lead_to_healthcareseeking'.
                    if m.force_any_symptom_to_lead_to_healthcareseeking:
                        date_of_seeking_care = self.sim.date
                    else:
                        max_days_delay = m.parameters['max_days_delay_to_generic_HSI_after_symptoms']
                        date_of_seeking_care = self.sim.date + DateOffset(days=m.rng.randint(0, max_days_delay))

                    health_system.schedule_hsi_event(
                        HSI_GenericFirstApptAtFacilityLevel1(m, person_id=person_id),
                        priority=0,
                        topen=date_of_seeking_care,
                        tclose=None
                    )
