"""
Health Seeking Behaviour Module
This module determines if care is sought once a symptom is developed.

The write-up of these estimates is: Health-seeking behaviour estimates for adults and children.docx

"""
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods.hsi_generic_first_appts import (
    HSI_GenericEmergencyFirstApptAtFacilityLevel1,
    HSI_GenericFirstApptAtFacilityLevel1,
)

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------


class HealthSeekingBehaviour(Module):
    """
    This modules determines if the onset of generic symptoms will lead to that person presenting at the health
    facility for a HSI_GenericFirstAppointment.

    """

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
        'odds_ratio_children_wealth_higher': Parameter(Types.REAL, 'odds ratio for health-care seeking (children) if'
                                                                    ' wealth is in categories 4 or 5'),
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
                                                                 ' is in categories 4 or 5')
    }

    # No properties to declare
    PROPERTIES = {}

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

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
        self.hsb = dict()
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
        self.no_healthcareseeking_in_children = set()
        self.emergency_in_children = set()
        self.odds_ratio_health_seeking_in_children = dict()
        self.no_healthcareseeking_in_adults = set()
        self.emergency_in_adults = set()
        self.odds_ratio_health_seeking_in_adults = dict()

        for symptom in self.sim.modules['SymptomManager'].all_registered_symptoms:
            # Children:
            if symptom.no_healthcareseeking_in_children:
                self.no_healthcareseeking_in_children = self.no_healthcareseeking_in_children.union({symptom.name})

            elif symptom.emergency_in_children:
                self.emergency_in_children = self.emergency_in_children.union({symptom.name})
            else:
                self.odds_ratio_health_seeking_in_children[symptom.name] = symptom.odds_ratio_health_seeking_in_children

            # Adults:
            if symptom.no_healthcareseeking_in_adults:
                self.no_healthcareseeking_in_adults = self.no_healthcareseeking_in_adults.union({symptom.name})

            elif symptom.emergency_in_adults:
                self.emergency_in_adults = self.emergency_in_adults.union({symptom.name})
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

        # get the list of person_ids who have onset generic acute symptoms in the last day, extracting any person_ids
        #    that have died (since the onset of symptoms)
        alive_person_ids = list(self.sim.population.props.index[self.sim.population.props.is_alive])
        person_ids_with_new_symptoms = list(
            m.sim.modules['SymptomManager'].persons_with_newly_onset_symptoms.intersection(alive_person_ids)
        )

        # clear the list of person_ids with newly onset symptoms
        m.sim.modules['SymptomManager'].persons_with_newly_onset_symptoms = set()

        def make_generic_emergency_first_appt(person_id):
            """Schedule a generic emergency first appt for the current date."""
            hsi_genericemergencyfirstappt = HSI_GenericEmergencyFirstApptAtFacilityLevel1(self.module,
                                                                                          person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(hsi_genericemergencyfirstappt,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=None)

        def make_generic_non_emergency_first_appt_at_facility_level1(person_id):
            """Schedule a generic non-emergency appointment for a delay of 0-4 days."""
            date_of_seeking_care = self.sim.date + DateOffset(days=m.rng.randint(0, 4))
            hsi_genericfirstappt = HSI_GenericFirstApptAtFacilityLevel1(m, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(hsi_genericfirstappt,
                                                                priority=0,
                                                                topen=date_of_seeking_care,
                                                                tclose=None)

        def get_joint_odds_ratio_of_all_symptoms(symptoms_list, is_child):
            """For a given symptoms list (of symptoms that do causes care seeking), get the associated odds ratio for
            health seeking behaviour: this is the product of the odds ratio associated with each symptom."""
            odds_ratio_list = []
            for s in symptoms_list:
                if is_child:
                    assert s not in m.emergency_in_children
                    assert s not in m.no_healthcareseeking_in_children
                    assert s in m.odds_ratio_health_seeking_in_children
                    odds_ratio_list.append(m.odds_ratio_health_seeking_in_children[s])
                else:
                    assert s not in m.emergency_in_adults
                    assert s not in m.no_healthcareseeking_in_adults
                    assert s in m.odds_ratio_health_seeking_in_adults
                    odds_ratio_list.append(m.odds_ratio_health_seeking_in_adults[s])

            return np.array(odds_ratio_list).prod()

        def compute_prob_from_baseline_prob_and_odds_ratio(baseline_prob, odds_ratio):
            """For a baseline probability and an effect odds ratio, compute probability that is implied"""
            baseline_odds = baseline_prob / (1 - baseline_prob)
            odds = baseline_odds * odds_ratio
            return odds / (1 + odds)

        for person_id in person_ids_with_new_symptoms:
            # For each individual person_id, with at least one new onset symptom, look at the symptoms and determine if
            # will seek care.
            # This is run looking at all symptoms even if only one is newly onset.

            symptoms = m.sim.modules['SymptomManager'].has_what(person_id)
            is_child = population.props.at[person_id, 'age_years'] < 15

            # sort the symptoms into the different categories:
            emergency_care_seeking_symptoms = []
            non_care_seeking_symptoms = []
            care_seeking_symptoms = []

            for symptom in symptoms:
                if is_child:
                    if symptom in m.emergency_in_children:
                        emergency_care_seeking_symptoms.append(symptom)
                    elif symptom in m.no_healthcareseeking_in_children:
                        non_care_seeking_symptoms.append(symptom)
                    else:
                        care_seeking_symptoms.append(symptom)

                if not is_child:
                    if symptom in m.emergency_in_adults:
                        emergency_care_seeking_symptoms.append(symptom)
                    elif symptom in m.no_healthcareseeking_in_adults:
                        non_care_seeking_symptoms.append(symptom)
                    else:
                        care_seeking_symptoms.append(symptom)

            assert len(symptoms) == (
                len(emergency_care_seeking_symptoms) + len(non_care_seeking_symptoms) + len(care_seeking_symptoms)
            )

            # if any of the symptom is an emergency - generate an emergency HSI:
            if len(emergency_care_seeking_symptoms) > 0:
                make_generic_emergency_first_appt(person_id)

            # if any non-emergency symptoms - consider making a generic non-emergency HSI:
            if len(care_seeking_symptoms) > 0:
                # Look at joint effect on health-care seeking of all symptoms
                if is_child:
                    baseline_prob = m.hsb['children'].predict(population.props.loc[[person_id]]).values[0]
                else:
                    baseline_prob = m.hsb['adults'].predict(population.props.loc[[person_id]]).values[0]
                odds_ratio = get_joint_odds_ratio_of_all_symptoms(care_seeking_symptoms, is_child)
                prob_hsb = compute_prob_from_baseline_prob_and_odds_ratio(baseline_prob, odds_ratio)
                if m.rng.rand() < prob_hsb:
                    make_generic_non_emergency_first_appt_at_facility_level1(person_id)
