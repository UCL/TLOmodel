"""
Health Seeking Behaviour Module
This module determines if care is sought once a symptom is developed.

The write-up of these estimates is: Health-seeking behaviour estimates for adults and children.docx

"""
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import Date, DateOffset, Module, Parameter, Types
from tlo.events import PopulationScopeEventMixin, Priority, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.hsi_generic_first_appts import (
    HSI_GenericEmergencyFirstApptAtFacilityLevel1,
    HSI_GenericFirstApptAtFacilityLevel0,
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

    INIT_DEPENDENCIES = {'Demography', 'HealthSystem', 'SymptomManager'}
    ADDITIONAL_DEPENDENCIES = {'Lifestyle'}

    # Declare Metadata
    METADATA = {Metadata.USES_HEALTHSYSTEM}

    # No parameters to declare
    PARAMETERS = {
        'force_any_symptom_to_lead_to_healthcareseeking': Parameter(
            Types.BOOL, "Whether every symptom should always lead to healthcare seeking (ignoring the other parameters "
                        "that determine the probability of seeking care."),
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

    def __init__(self, name=None, resourcefilepath=None, force_any_symptom_to_lead_to_healthcareseeking=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        self.hsb_linear_models = dict()
        self.odds_ratio_health_seeking_in_children = dict()
        self.odds_ratio_health_seeking_in_adults = dict()
        self.no_healthcareseeking_in_children = set()
        self.emergency_in_children = set()
        self.non_emergency_healthcareseeking_in_children = set()
        self.no_healthcareseeking_in_adults = set()
        self.emergency_in_adults = set()
        self.non_emergency_healthcareseeking_in_adults = set()

        # "force_any_symptom_to_lead_to_healthcareseeking"=True will mean that probability of health care seeking is 1.0
        # for anyone with newly onset symptoms. Note that if this is not specified, then the value is taken from the
        # ResourceFile.
        if force_any_symptom_to_lead_to_healthcareseeking is not None:
            assert isinstance(force_any_symptom_to_lead_to_healthcareseeking, bool)
        self.arg_force_any_symptom_to_lead_to_healthcareseeking = force_any_symptom_to_lead_to_healthcareseeking

    def read_parameters(self, data_folder):
        """Read in ResourceFile"""
        # Load parameters from resource file:
        self.load_parameters_from_dataframe(
            pd.read_csv(Path(self.resourcefilepath) / 'ResourceFile_HealthSeekingBehaviour.csv')
        )

        # Check that force_any_symptom_to_lead_to_healthcareseeking is a bool (this is returned in
        # `self.force_any_symptom_to_lead_to_healthcareseeking` without any further checking).
        assert isinstance(self.parameters['force_any_symptom_to_lead_to_healthcareseeking'], bool)

    def initialise_population(self, population):
        """Nothing to initialise in the population
        """
        pass

    def initialise_simulation(self, sim):
        """
        * define the linear models that govern healthcare seeking
        * set the first occurrence of the repeating HealthSeekingBehaviourPoll
        * assemble the health-care seeking information from the registered symptoms
        """

        # Define the linear models that govern healthcare seeking
        self.define_linear_models()

        # Schedule the HealthSeekingBehaviourPoll
        self.theHealthSeekingBehaviourPoll = HealthSeekingBehaviourPoll(self)
        sim.schedule_event(self.theHealthSeekingBehaviourPoll, sim.date)

        # Assemble the health-care seeking information from the registered symptoms
        for symptom in self.sim.modules['SymptomManager'].all_registered_symptoms:
            # Children:
            if symptom.no_healthcareseeking_in_children:
                self.no_healthcareseeking_in_children.add(symptom.name)
            elif symptom.emergency_in_children:
                self.emergency_in_children.add(symptom.name)
            else:
                self.non_emergency_healthcareseeking_in_children.add(symptom.name)
                self.odds_ratio_health_seeking_in_children[symptom.name] = (
                    symptom.odds_ratio_health_seeking_in_children
                )

            # Adults:
            if symptom.no_healthcareseeking_in_adults:
                self.no_healthcareseeking_in_adults.add(symptom.name)
            elif symptom.emergency_in_adults:
                self.emergency_in_adults.add(symptom.name)
            else:
                self.non_emergency_healthcareseeking_in_adults.add(symptom.name)
                self.odds_ratio_health_seeking_in_adults[symptom.name] = (
                    symptom.odds_ratio_health_seeking_in_adults
                )

    def on_birth(self, mother_id, child_id):
        """Nothing to handle on_birth
        """
        pass

    def define_linear_models(self):
        """Define linear models for health seeking behaviour for children and adults"""
        p = self.parameters
        for subgroup, age_predictor, odds_ratios, care_seeking_symptoms in zip(
            ('children', 'adults'),
            (
                Predictor('age_years').when('>=5', p['odds_ratio_children_age_5to14']),
                Predictor('age_years', conditions_are_mutually_exclusive=True)
                .when('.between(35,59)', p['odds_ratio_adults_age_35to59'])
                .when('>=60', p['odds_ratio_adults_age_60plus']),
            ),
            (
                self.odds_ratio_health_seeking_in_children,
                self.odds_ratio_health_seeking_in_adults
            ),
            (
                self.non_emergency_healthcareseeking_in_children,
                self.non_emergency_healthcareseeking_in_adults
            ),
        ):
            self.hsb_linear_models[subgroup] = LinearModel(
                LinearModelType.LOGISTIC,
                p[f'baseline_odds_of_healthcareseeking_{subgroup}'],
                # First set of predictors are for behaviour due to the 'average symptom'
                # This is from the Ng'ambia et al. papers
                Predictor('li_urban').when(
                    True, p[f'odds_ratio_{subgroup}_setting_urban']
                ),
                Predictor('sex').when('F', p[f'odds_ratio_{subgroup}_sex_Female']),
                age_predictor,
                Predictor('region_of_residence', conditions_are_mutually_exclusive=True)
                .when('Central', p[f'odds_ratio_{subgroup}_region_Central'])
                .when('Southern', p[f'odds_ratio_{subgroup}_region_Southern']),
                Predictor('li_wealth', conditions_are_mutually_exclusive=True)
                .when(4, p[f'odds_ratio_{subgroup}_wealth_higher'])
                .when(5, p[f'odds_ratio_{subgroup}_wealth_higher']),
                # Second set of predictors are the symptom specific odd ratios
                *(
                    Predictor(f'sy_{symptom}').when('>0', odds_ratios[symptom])
                    for symptom in care_seeking_symptoms
                )
            )

    @property
    def force_any_symptom_to_lead_to_healthcareseeking(self):
        """Returns the parameter value stored for `force_any_symptom_to_lead_to_healthcareseeking` unless this has
         been over-ridden by an argument to the module."""
        if self.arg_force_any_symptom_to_lead_to_healthcareseeking is None:
            return self.parameters['force_any_symptom_to_lead_to_healthcareseeking']
        else:
            return self.arg_force_any_symptom_to_lead_to_healthcareseeking

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
        super().__init__(module, frequency=DateOffset(days=1), priority=Priority.LAST_HALF_OF_DAY)
        assert isinstance(module, HealthSeekingBehaviour)

    @staticmethod
    def _has_any_symptoms(persons, symptoms):
        """Which rows in `persons` have non-zero values for columns in `symptoms`."""
        if len(symptoms) == 0:
            raise ValueError('At least one symptom must be specified')
        return (persons[[f'sy_{symptom}' for symptom in symptoms]] != 0).any(axis=1)

    def apply(self, population):
        """Determine if persons with newly onset acute generic symptoms will seek care. This event runs second-to-last
        every day (i.e., just before the `HealthSystemScheduler`) in order that symptoms arising this day can lead to
        FirstAttendance on the same day.

        :param population: the current population
        """
        # Define some shorter aliases
        module = self.module
        symptom_manager = self.sim.modules["SymptomManager"]
        health_system = self.sim.modules["HealthSystem"]
        max_delay = module.parameters['max_days_delay_to_generic_HSI_after_symptoms']
        routine_hsi_event_class = HSI_GenericFirstApptAtFacilityLevel0
        emergency_hsi_event_class = HSI_GenericEmergencyFirstApptAtFacilityLevel1

        # Get IDs of alive persons with new symptoms
        person_ids_with_newly_onset_symptoms = sorted(
            symptom_manager.get_persons_with_newly_onset_symptoms())
        symptomatic_persons = population.props.loc[person_ids_with_newly_onset_symptoms]
        alive_symptomatic_persons = symptomatic_persons[symptomatic_persons.is_alive]

        # Clear the list of persons with newly onset symptoms
        symptom_manager.reset_persons_with_newly_onset_symptoms()

        # Split alive symptomatic persons into child and adult subgroups
        are_under_15 = alive_symptomatic_persons.age_years < 15
        alive_symptomatic_children = alive_symptomatic_persons[are_under_15]
        alive_symptomatic_adults = alive_symptomatic_persons[~are_under_15]

        # Separately schedule HSI events for child and adult subgroups
        for subgroup, emergency_symptoms, care_seeking_symptoms, hsb_model in zip(
            (alive_symptomatic_children, alive_symptomatic_adults),
            (module.emergency_in_children, module.emergency_in_adults),
            (
                module.non_emergency_healthcareseeking_in_children,
                module.non_emergency_healthcareseeking_in_adults
            ),
            (module.hsb_linear_models['children'], module.hsb_linear_models['adults'])
        ):
            if len(emergency_symptoms) > 0:
                # Generate an emergency HSI event if any of the symptoms is an emergency
                is_emergency_care_seeking = self._has_any_symptoms(
                    subgroup, emergency_symptoms
                )
                health_system.schedule_batch_of_individual_hsi_events(
                    hsi_event_class=emergency_hsi_event_class,
                    person_ids=subgroup[is_emergency_care_seeking].index,
                    priority=0,
                    topen=self.sim.date,
                    tclose=None,
                    module=module
                )
                # If a person has had an emergency appointment scheduled this day
                # already due to emergency symptoms, then do not allow a non-emergency
                # appointment to be scheduled in addition, so select the subgroup
                # who are not emergency care-seeking
                not_emergency_care_seeking_subgroup = subgroup[
                    ~is_emergency_care_seeking
                ]
            else:
                not_emergency_care_seeking_subgroup = subgroup

            # Check if no symptoms initiating (non-emergency) care seeking specified
            if len(care_seeking_symptoms) == 0:
                continue
            # Symptoms in non-emergency care seeking set may or may not generate an
            # associated HSI event, we first select all persons in
            # not_emergency_care_seeking_subgroup who have any symptoms which may lead
            # to a HSI event being generated. From here onwards care seeking should be
            # taken to mean specifically *non-emergency* care seeking
            possibly_care_seeking_subgroup = not_emergency_care_seeking_subgroup[
                self._has_any_symptoms(
                    not_emergency_care_seeking_subgroup, care_seeking_symptoms
                )
            ]

            if module.force_any_symptom_to_lead_to_healthcareseeking:
                # This HSB module flag causes a generic non-emergency appointment to be
                # scheduled for any symptom immediately
                health_system.schedule_batch_of_individual_hsi_events(
                    hsi_event_class=routine_hsi_event_class,
                    person_ids=possibly_care_seeking_subgroup.index,
                    priority=0,
                    topen=self.sim.date,
                    tclose=None,
                    module=module
                )
            else:
                # All in-patients with symptoms always generate a HSI event
                care_seeking_inpatients = possibly_care_seeking_subgroup[
                    possibly_care_seeking_subgroup.hs_is_inpatient
                ]
                # For non-in-patients with symptoms use HSB linear model to (randomly)
                # select subset seeking care and so generating a HSI event
                possibly_care_seeking_non_inpatients = possibly_care_seeking_subgroup[
                    ~possibly_care_seeking_subgroup.hs_is_inpatient
                ]
                care_seeking_non_inpatients = possibly_care_seeking_non_inpatients[
                    hsb_model.predict(
                        possibly_care_seeking_non_inpatients,
                        module.rng,
                        squeeze_single_row_output=False,
                    )
                ]
                for care_seeking_ids in (
                    care_seeking_inpatients.index, care_seeking_non_inpatients.index
                ):
                    # Schedule generic non-emergency appointments after a random delay
                    care_seeking_dates = (
                        # Create NumPy datetime with day unit to allow directly adding
                        # array of generated integer delays in [0, max_delay]
                        np.array(self.sim.date, dtype='datetime64[D]')
                        + module.rng.randint(0, max_delay, size=len(care_seeking_ids))
                    )
                    health_system.schedule_batch_of_individual_hsi_events(
                        hsi_event_class=routine_hsi_event_class,
                        person_ids=care_seeking_ids,
                        priority=0,
                        topen=map(Date, care_seeking_dates),
                        tclose=None,
                        module=module
                    )
