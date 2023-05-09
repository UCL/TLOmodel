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
from tlo.lm import LinearModel
from tlo.methods import Metadata
from tlo.methods.hsi_generic_first_appts import (
    HSI_GenericEmergencyFirstApptAtFacilityLevel1,
    HSI_GenericFirstApptAtFacilityLevel0,
)

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

HIGH_ODDS_RATIO = 1e5


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
            Types.BOOL, "Whether every symptom [except those that declare they should not lead to any healthcare "
                        "seeking] should always lead to healthcare seeking immediately."),
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

        self.odds_ratio_health_seeking_in_children = dict()
        self.odds_ratio_health_seeking_in_adults = dict()
        self.prob_seeks_emergency_appt_in_children = dict()
        self.prob_seeks_emergency_appt_in_adults = dict()

        self.hsb_linear_models = dict()
        self.emergency_appt_linear_models = dict()

        # "force_any_symptom_to_lead_to_healthcareseeking"=True will mean that probability of health care seeking is 1.0
        # for anyone with newly onset symptoms (excepting symptoms explicitly declared to have no healthcareseeking
        # behaviour) and the care is sought on the same day.
        # (Note that if this is not specified, then the value is taken from the ResourceFile.)
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

        # Schedule the HealthSeekingBehaviourPoll
        self.theHealthSeekingBehaviourPoll = HealthSeekingBehaviourPoll(self)
        sim.schedule_event(self.theHealthSeekingBehaviourPoll, sim.date)

        # Assemble the health-care seeking information from the registered symptoms
        for symptom in self.sim.modules['SymptomManager'].all_registered_symptoms:
            # Children:
            if not symptom.no_healthcareseeking_in_children:
                self.odds_ratio_health_seeking_in_children[symptom.name] = (
                    symptom.odds_ratio_health_seeking_in_children
                )
                self.prob_seeks_emergency_appt_in_children[symptom.name] = (
                    symptom.prob_seeks_emergency_appt_in_children
                )

            # Adults:
            if not symptom.no_healthcareseeking_in_adults:
                self.odds_ratio_health_seeking_in_adults[symptom.name] = (
                    symptom.odds_ratio_health_seeking_in_adults
                )
                self.prob_seeks_emergency_appt_in_adults[symptom.name] = (
                    symptom.prob_seeks_emergency_appt_in_adults
                )

        # Define the linear models that govern healthcare seeking
        self.define_linear_models()

    def on_birth(self, mother_id, child_id):
        """Nothing to handle on_birth
        """
        pass

    def define_linear_models(self):
        """Define linear models for health seeking behaviour for children and adults"""
        p = self.parameters

        # Use a custom function to represent the linear model for healthcare seeking
        def predict_healthcareseeking(
            self, df, rng=None, subgroup=None, care_seeking_odds_ratios=None
        ):
            if subgroup is None or care_seeking_odds_ratios is None:
                raise ValueError("subgroup and care_seeking_odds_ratios must both be specified")

            result = pd.Series(data=p[f'baseline_odds_of_healthcareseeking_{subgroup}'], index=df.index)
            # Predict behaviour due to the 'average symptom'
            if subgroup == 'children':
                result[df.age_years >= 5] *= p['odds_ratio_children_age_5to14']
            if subgroup == 'adults':
                result[df.age_years.between(35, 59)] *= p['odds_ratio_adults_age_35to59']
                result[df.age_years >= 60] *= p['odds_ratio_adults_age_60plus']
            result[df.li_urban] *= p[f'odds_ratio_{subgroup}_setting_urban']
            result[df.sex == 'F'] *= p[f'odds_ratio_{subgroup}_sex_Female']
            result[df.region_of_residence == 'Central'] *= p[f'odds_ratio_{subgroup}_region_Central']
            result[df.region_of_residence == 'Southern'] *= p[f'odds_ratio_{subgroup}_region_Southern']
            result[(df.li_wealth == 4) | (df.li_wealth == 5)] *= p[f'odds_ratio_{subgroup}_wealth_higher']
            # Predict for symptom-specific odd ratios
            for symptom, odds in care_seeking_odds_ratios.items():
                result[df[f'sy_{symptom}'] > 0] *= odds
            result = (1 / (1 + 1 / result))
            # If a random number generator is supplied provide boolean outcomes, not probabilities
            if rng:
                outcome = rng.random_sample(len(result)) < result
                return outcome
            else:
                return result

        for subgroup in (
            'children',
            'adults'
        ):
            self.hsb_linear_models[subgroup] = LinearModel.custom(predict_function=predict_healthcareseeking)

        # Model for the care-seeking (if it occurs) to be for an EMERGENCY Appointment:
        def custom_predict(self, df, rng=None, **externals) -> pd.Series:
            """Custom predict function for LinearModel. This finds the probability that a person seeks emergency care
            by finding the highest probability of seeking emergency care for all symptoms they have currently."""
            prob = pd.Series(
                (
                    (df[[f'sy_{s}' for s in self.prob_emergency_appt]].to_numpy() > 0)
                    * np.array(list(self.prob_emergency_appt.values()))
                ).max(axis=1),
                df.index
            )
            return prob > rng.random_sample(len(prob))

        for subgroup, prob_emergency_appt in zip(
            (
                'children',
                'adults'
            ),
            (
                self.prob_seeks_emergency_appt_in_children,
                self.prob_seeks_emergency_appt_in_adults
            ),
        ):
            self.emergency_appt_linear_models[subgroup] = LinearModel.custom(predict_function=custom_predict,
                                                                             prob_emergency_appt=prob_emergency_appt)

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
        newly_symptomatic_persons = population.props.loc[person_ids_with_newly_onset_symptoms]
        alive_newly_symptomatic_persons = newly_symptomatic_persons[newly_symptomatic_persons.is_alive]

        # Clear the list of persons with newly onset symptoms
        symptom_manager.reset_persons_with_newly_onset_symptoms()

        # Split alive newly symptomatic persons into child and adult subgroups
        are_under_15 = alive_newly_symptomatic_persons.age_years < 15
        alive_newly_symptomatic_children = alive_newly_symptomatic_persons[are_under_15]
        alive_newly_symptomatic_adults = alive_newly_symptomatic_persons[~are_under_15]

        idx_where_true = lambda series: series.loc[series].index  # noqa: E731

        # Separately schedule HSI events for child and adult subgroups
        for subgroup, subgroup_name, care_seeking_odds_ratios, hsb_model, emergency_appt_model in zip(
            (
                alive_newly_symptomatic_children,
                alive_newly_symptomatic_adults
            ),
            (
                'children',
                'adults'
             ),
            (
                module.odds_ratio_health_seeking_in_children,
                module.odds_ratio_health_seeking_in_adults,
            ),
            (
                module.hsb_linear_models['children'],
                module.hsb_linear_models['adults']
            ),
            (
                module.emergency_appt_linear_models['children'],
                module.emergency_appt_linear_models['adults']
            ),
        ):
            symptoms_that_allow_healthcareseeking = care_seeking_odds_ratios.keys()
            # Determine who will seek care:
            if module.force_any_symptom_to_lead_to_healthcareseeking:
                # If forcing any person with symptoms to seek care, find all those with any symptoms which cause
                # any degree of healthcare seeking (i.e., excluding symptoms declared to have no healthcare-seeking
                # behaviour).
                will_seek_care = idx_where_true(self._has_any_symptoms(subgroup, symptoms_that_allow_healthcareseeking))
            else:
                # If not forcing, run the custom model to predict which persons will seek care, from among those
                # with symptoms that cause any degree of healthcare seeking.
                will_seek_care = idx_where_true(
                    hsb_model.predict(
                        subgroup.loc[self._has_any_symptoms(subgroup, symptoms_that_allow_healthcareseeking)],
                        module.rng,
                        subgroup=subgroup_name, care_seeking_odds_ratios=care_seeking_odds_ratios)
                )

                # Force the addition to this set those who are already in-patient. (In-patients will always get the
                # notional "FirstAppointment" for a new symptom.)
                will_seek_care = will_seek_care.union(idx_where_true(subgroup.hs_is_inpatient))

            # Determine if the care sought will be emergency care (for those that seek care):
            will_seek_emergency_care = idx_where_true(
                emergency_appt_model.predict(subgroup.loc[will_seek_care], module.rng, squeeze_single_row_output=False))

            # Determine who will seek non-emergency care (those that did not seek emergency care):
            will_seek_non_emergency_care = will_seek_care.difference(will_seek_emergency_care)

            # Schedule Emergency Care for same day
            health_system.schedule_batch_of_individual_hsi_events(
                hsi_event_class=emergency_hsi_event_class,
                person_ids=sorted(will_seek_emergency_care),
                priority=0,
                topen=self.sim.date,
                tclose=None,
                module=module
            )

            # Schedule Non-Emergency Care for "soon" (after a random delay), or the same day if using
            # `force_any_symptom_to_lead_to_healthcareseeking`.
            if not module.force_any_symptom_to_lead_to_healthcareseeking:
                care_seeking_dates = (
                    # Create NumPy datetime with day unit to allow directly adding
                    # array of generated integer delays in [0, max_delay]
                    np.array(self.sim.date, dtype='datetime64[D]')
                    + module.rng.randint(0, max_delay + 1, size=len(will_seek_non_emergency_care))
                    # (The +1 is because `randint` takes the upper bound to be excluded.)
                )
            else:
                care_seeking_dates = np.full(len(will_seek_non_emergency_care), self.sim.date)

            health_system.schedule_batch_of_individual_hsi_events(
                hsi_event_class=routine_hsi_event_class,
                person_ids=sorted(will_seek_non_emergency_care),
                priority=0,
                topen=map(Date, care_seeking_dates),
                tclose=None,
                module=module
            )
