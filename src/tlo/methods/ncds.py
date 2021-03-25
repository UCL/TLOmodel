"""
The joint NCDs model by Tim Hallett and Britta Jewell, October 2020

"""
import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd

import tlo.methods.demography as de
from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, demography
from tlo.methods.demography import InstantaneousDeath
from tlo.methods.healthsystem import HSI_Event

from itertools import combinations

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Ncds(Module):
    """
    NCDs module covers a subset of NCD conditions and events. Conditions are binary, and individuals experience a risk
    of acquiring or losing a condition based on annual probability and demographic/lifestyle risk factors.

    """
    # save a master list of the events that are covered in this module
    conditions = ['diabetes',
                  'hypertension',
                  'chronic_kidney_disease',
                  'chronic_lower_back_pain',
                  'chronic_ischemic_hd']

    # save a master list of the events that are covered in this module
    events = ['ever_stroke',
              'ever_heart_attack']

    # Declare Metadata (this is for a typical 'Disease Module')
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    # create separate dicts for params for conditions and events
    onset_conditions_param_dicts = {
        f"{p}_onset": Parameter(Types.DICT, f"all the parameters that specify the linear models for onset of {p}")
        for p in conditions
    }
    removal_conditions_param_dicts = {
        f"{p}_removal": Parameter(Types.DICT, f"all the parameters that specify the linear models for removal of {p}")
        for p in conditions
    }
    onset_events_param_dicts = {
        f"{p}_onset": Parameter(Types.DICT, f"all the parameters that specify the linear models for onset of {p}")
        for p in events
    }
    death_conditions_param_dicts = {
        f"{p}_death": Parameter(Types.DICT, f"all the parameters that specify the linear models for death from {p}")
        for p in conditions
    }
    death_events_param_dicts = {
        f"{p}_death": Parameter(Types.DICT, f"all the parameters that specify the linear models for death from {p}")
        for p in events
    }
    initial_prev_param_dicts = {
        f"{p}_initial_prev": Parameter(Types.DICT, 'initial prevalence of condition') for p in conditions
    }
    other_params_dict = {
        'interval_between_polls': Parameter(Types.INT, 'months between the main polling event')
    }

    PARAMETERS = {**onset_conditions_param_dicts, **removal_conditions_param_dicts, **onset_events_param_dicts,
                  **death_conditions_param_dicts, **death_events_param_dicts, **initial_prev_param_dicts,
                  **other_params_dict
                  }

    # convert conditions and events to dicts and merge together into PROPERTIES
    condition_list = {
        f"nc_{p}": Property(Types.BOOL, f"Whether or not someone has {p}") for p in conditions
    }
    event_list = {
        f"nc_{p}": Property(Types.BOOL, f"Whether or not someone has had a {p}") for p in events}

    PROPERTIES = {**condition_list, **event_list,
                  'nc_depression': Property(Types.BOOL,
                                            'whether or not the person currently has depression'
                                            ),
                  'nc_cancers': Property(Types.BOOL,
                                         'whether or not the person currently has any form of cancer'
                                         )
                  }

    # TODO: we will have to later gather from the others what the symptoms are in each state - for now leave blank
    SYMPTOMS = {}

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        self.conditions = list(Ncds.condition_list)
        self.events = list(Ncds.event_list)

        # create list that includes conditions modelled by other modules
        self.extended_conditions = list(Ncds.condition_list)
        self.extended_conditions.append("nc_depression")

    def read_parameters(self, data_folder):
        """Read parameter values from files for condition onset, removal, deaths, and initial prevalence.
        To access files use: Path(self.resourcefilepath) / file_name
        """

        for condition in self.conditions:
            # get onset parameters
            params_onset = pd.read_excel(Path(self.resourcefilepath) / "ncds" /
                                         "ResourceFile_NCDs_condition_onset.xlsx",
                                         sheet_name=f"{condition}")
            # replace NaNs with 1
            params_onset['value'] = params_onset['value'].replace(np.nan, 1)
            self.parameters[f'{condition}_onset'] = params_onset

            # get removal parameters
            params_removal = pd.read_excel(Path(self.resourcefilepath) / "ncds" /
                                           "ResourceFile_NCDs_condition_removal.xlsx",
                                           sheet_name=f"{condition}")
            # replace NaNs with 1
            params_removal['value'] = params_removal['value'].replace(np.nan, 1)
            self.parameters[f'{condition}_removal'] = params_removal

            # get death parameters
            params_death = pd.read_excel(Path(self.resourcefilepath) / "ncds" /
                                         "ResourceFile_NCDs_condition_death.xlsx",
                                         sheet_name=f"{condition}")
            # replace NaNs with 1
            params_death['value'] = params_death['value'].replace(np.nan, 1)
            self.parameters[f'{condition}_death'] = params_death

            # get parameters for initial prevalence by age/sex
            params_prevalence = pd.read_excel(
                Path(self.resourcefilepath) / "ncds" / "ResourceFile_NCDs_condition_prevalence.xlsx",
                sheet_name=f"{condition}")
            params_prevalence['value'] = params_prevalence['value'].replace(np.nan, 0)
            self.parameters[f'{condition}_initial_prev'] = params_prevalence

        for event in self.events:
            # get onset parameters
            params_onset = pd.read_excel(Path(self.resourcefilepath) / "ncds" / "ResourceFile_NCDs_events.xlsx",
                                         sheet_name=f"{event}")
            # replace NaNs with 1
            params_onset['value'] = params_onset['value'].replace(np.nan, 1)
            self.parameters[f'{event}_onset'] = params_onset

            # get death parameters
            params_death = pd.read_excel(Path(self.resourcefilepath) / "ncds" / "ResourceFile_NCDs_events_death.xlsx",
                                         sheet_name=f"{event}")
            # replace NaNs with 1
            params_death['value'] = params_death['value'].replace(np.nan, 1)
            self.parameters[f'{event}_death'] = params_death

        # Check that every value has been read-in successfully
        p = self.parameters
        for param_name, param_type in self.PARAMETERS.items():
            if param_name != 'interval_between_polls':  # this param is manually entered in the code
                param_name = f'nc_{param_name}'
                assert param_name in p, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
                assert param_name is not None, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'

        # Set the interval (in months) between the polls
        self.parameters['interval_between_polls'] = 3

    def initialise_population(self, population):
        """Set our property values for the initial population.
        """
        # dict to hold counters for the number of episodes by condition-type and age-group
        self.age_index = self.sim.modules['Demography'].AGE_RANGE_CATEGORIES
        self.df_incidence_tracker_zeros = pd.DataFrame(0, index=self.age_index, columns=self.conditions)
        self.df_incidence_tracker = copy.deepcopy(self.df_incidence_tracker_zeros)

        df = population.props
        for condition in self.conditions:
            p = self.parameters[f'{condition}_initial_prev'].set_index('parameter_name').T.to_dict('records')[0]
            # Set age min and max to get correct age group later
            age_min = 0
            age_max = 4
            for age_grp in self.age_index:
                # Select all eligible individuals
                eligible_pop_m = df.index[
                    df.is_alive & (df.age_years.between(age_min, age_max)) & (df.sex == 'M') & ~df[f'{condition}']]
                init_prev_m = self.rng.choice([True, False], size=len(eligible_pop_m),
                                              p=[p[f'm_{age_grp}'], 1 - p[f'm_{age_grp}']])
                eligible_pop_f = df.index[
                    df.is_alive & (df.age_years.between(age_min, age_max)) & (df.sex == 'F') & ~df[f'{condition}']]
                init_prev_f = self.rng.choice([True, False], size=len(eligible_pop_f),
                                              p=[p[f'f_{age_grp}'], 1 - p[f'f_{age_grp}']])
                # if any have condition
                if init_prev_m.sum():
                    condition_idx_m = eligible_pop_m[init_prev_m]
                    df.loc[condition_idx_m, f'{condition}'] = True
                if init_prev_f.sum():
                    condition_idx_f = eligible_pop_f[init_prev_f]
                    df.loc[condition_idx_f, f'{condition}'] = True
                if age_grp != '100+':
                    age_min = age_min + 5
                    age_max = age_max + 5
                else:
                    age_min = age_min + 5
                    age_max = age_max + 20

    def initialise_simulation(self, sim):
        """Schedule:
        * Main Polling Event
        * Main Logging Event
        * Build the LinearModels for the onset/removal of each condition:
        """
        sim.schedule_event(Ncds_MainPollingEvent(self, self.parameters['interval_between_polls']), sim.date)
        sim.schedule_event(Ncds_LoggingEvent(self), sim.date)

        df = self.sim.population.props
        df['nc_depression'] = df['de_depr']
        # cancers_idx = df.index[df['de_depr']]

        # Create Tracker for the number of different types of events
        self.eventsTracker = dict()
        for event in self.events:
            event_name = event.replace('nc_ever_', '')
            self.eventsTracker.update({f'{event_name}_events': 0})

        # Build the LinearModel for onset/removal/deaths for each condition
        self.lms_onset = dict()
        self.lms_removal = dict()
        self.lms_death = dict()

        # Build the LinearModel for occurrence of events
        self.lms_event_onset = dict()
        self.lms_event_death = dict()

        for condition in self.conditions:
            self.lms_dict[condition] = self.build_linear_model(condition,
                                                               self.parameters['interval_between_polls'],
                                                               lm_type='onset')
            self.lms_onset[condition] = self.lms_dict[condition]

            self.lms_dict[condition] = self.build_linear_model(condition,
                                                               self.parameters['interval_between_polls'],
                                                               lm_type='removal')
            self.lms_removal[condition] = self.lms_dict[condition]

            self.lms_dict[condition] = self.build_linear_model(condition,
                                                               self.parameters['interval_between_polls'],
                                                               lm_type='death')
            self.lms_death[condition] = self.lms_dict[condition]

        for event in self.events:
            self.lms_dict[event] = self.build_linear_model(event,
                                                           self.parameters['interval_between_polls'],
                                                           lm_type='event')
            self.lms_event_onset[event] = self.lms_dict[event]

            self.lms_dict[event] = self.build_linear_model(event,
                                                           self.parameters['interval_between_polls'],
                                                           lm_type='death')
            self.lms_event_death[event] = self.lms_dict[event]

    def build_linear_model(self, condition, interval_between_polls, lm_type):
        """
        :param_dict: the dict read in from the resourcefile
        :param interval_between_polls: the duration (in months) between the polls
        :return: a linear model
        """

        # use temporary empty dict to save results
        self.lms_dict = dict()

        # load parameters for correct condition/event
        p = self.parameters
        if lm_type == 'onset':
            p = self.parameters[f'{condition}_onset'].set_index('parameter_name').T.to_dict('records')[0]
        elif lm_type == 'removal':
            p = self.parameters[f'{condition}_removal'].set_index('parameter_name').T.to_dict('records')[0]
        elif lm_type == 'event':
            p = self.parameters[f'{condition}_onset'].set_index('parameter_name').T.to_dict('records')[0]
        elif lm_type == 'death':
            p = self.parameters[f'{condition}_death'].set_index('parameter_name').T.to_dict('records')[0]

        p['baseline_annual_probability'] = p['baseline_annual_probability'] * (interval_between_polls / 12)

        self.lms_dict[condition] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['baseline_annual_probability'],
            Predictor().when('(sex=="M")', p['rr_male']),
            Predictor('age_years').when('.between(0, 4)', p['rr_0_4'])
                .when('.between(5, 9)', p['rr_5_9'])
                .when('.between(10, 14)', p['rr_10_14'])
                .when('.between(15, 19)', p['rr_15_19'])
                .when('.between(20, 24)', p['rr_20_24'])
                .when('.between(25, 29)', p['rr_25_29'])
                .when('.between(30, 34)', p['rr_30_34'])
                .when('.between(35, 39)', p['rr_35_39'])
                .when('.between(40, 44)', p['rr_40_44'])
                .when('.between(45, 49)', p['rr_45_49'])
                .when('.between(50, 54)', p['rr_50_54'])
                .when('.between(55, 59)', p['rr_55_59'])
                .when('.between(60, 64)', p['rr_60_64'])
                .when('.between(65, 69)', p['rr_65_69'])
                .when('.between(70, 74)', p['rr_70_74'])
                .when('.between(75, 79)', p['rr_75_79'])
                .when('.between(80, 84)', p['rr_80_84'])
                .when('.between(85, 89)', p['rr_85_89'])
                .when('.between(90, 94)', p['rr_90_94'])
                .when('.between(95, 99)', p['rr_95_99'])
                .otherwise(p['rr_100']),
            Predictor('li_urban').when(True, p['rr_urban']),
            Predictor('li_wealth').when('==1', p['rr_wealth_1'])
                .when('2', p['rr_wealth_2'])
                .when('3', p['rr_wealth_3'])
                .when('4', p['rr_wealth_4'])
                .when('5', p['rr_wealth_5']),
            Predictor('li_bmi').when('==1', p['rr_bmi_1'])
                .when('2', p['rr_bmi_2'])
                .when('3', p['rr_bmi_3'])
                .when('4', p['rr_bmi_4'])
                .when('5', p['rr_bmi_5']),
            Predictor('li_low_ex').when(True, p['rr_low_exercise']),
            Predictor('li_high_salt').when(True, p['rr_high_salt']),
            Predictor('li_high_sugar').when(True, p['rr_high_sugar']),
            Predictor('li_tob').when(True, p['rr_tobacco']),
            Predictor('li_ex_alc').when(True, p['rr_alcohol']),
            Predictor('li_mar_stat').when('1', p['rr_marital_status_1'])
                .when('2', p['rr_marital_status_2'])
                .when('3', p['rr_marital_status_3']),
            Predictor('li_in_ed').when(True, p['rr_in_education']),
            Predictor('li_ed_lev').when('==1', p['rr_current_education_level_1'])
                .when('2', p['rr_current_education_level_2'])
                .when('3', p['rr_current_education_level_3']),
            Predictor('li_unimproved_sanitation').when(True, p['rr_unimproved_sanitation']),
            Predictor('li_no_access_handwashing').when(True, p['rr_no_access_handwashing']),
            Predictor('li_no_clean_drinking_water').when(True, p['rr_no_clean_drinking_water']),
            Predictor('li_wood_burn_stove').when(True, p['rr_wood_burning_stove']),
            Predictor('nc_diabetes').when(True, p['rr_diabetes']),
            Predictor('nc_hypertension').when(True, p['rr_hypertension']),
            Predictor('de_depr').when(True, p['rr_depression']),
            Predictor('nc_chronic_kidney_disease').when(True, p['rr_chronic_kidney_disease']),
            Predictor('nc_chronic_lower_back_pain').when(True, p['rr_chronic_lower_back_pain']),
            Predictor('nc_chronic_ischemic_hd').when(True, p['rr_chronic_ischemic_heart_disease'])
            # Predictor('nc_cancers').when(True, p['rr_cancers'])
        )

        return self.lms_dict[condition]

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        # TODO: @britta - assuming that the all children have nothing when they are born
        df = self.sim.population.props
        for condition in self.conditions:
            df.at[child_id, condition] = False

    def report_daly_values(self):
        """Report DALY values to the HealthBurden module"""
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        # To return a value of 0.0 (fully health) for everyone, use:
        # df = self.sim.popultion.props
        # return pd.Series(index=df.index[df.is_alive],data=0.0)

        # TODO: @britta add in functionality to fetch daly weight from resourcefile

        df = self.sim.population.props
        any_condition = df.loc[df.is_alive, self.conditions].any(axis=1)

        return any_condition * 0.1

        pass

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

class Ncds_MainPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """The Main Polling Event.
    * Establishes onset of each condition
    * Establishes removal of each condition
    * Schedules events that arise, according the condition.
    """

    def __init__(self, module, interval_between_polls):
        """The Main Polling Event of the NCDs Module

        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=interval_between_polls))
        assert isinstance(module, Ncds)

    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = m.rng

        current_incidence_df = pd.DataFrame(index=self.module.age_index, columns=self.module.conditions)

        # Update depression status
        df['nc_depression'] = df['de_depr']

        # Determine onset/removal of conditions
        for condition in self.module.conditions:

            # onset:
            eligible_population = df.is_alive & ~df[condition]
            acquires_condition = self.module.lms_onset[condition].predict(df.loc[eligible_population], rng)
            idx_acquires_condition = acquires_condition[acquires_condition].index
            df.loc[idx_acquires_condition, condition] = True

            # Add incident cases to the tracker
            current_incidence_df[condition] = df.loc[idx_acquires_condition, :].groupby('age_range')[condition].count()

            # -------------------------------------------------------------------------------------------

            # removal:
            eligible_population = df.is_alive & df[condition]
            loses_condition = self.module.lms_removal[condition].predict(df.loc[eligible_population], rng)
            if loses_condition.any():  # catch in case no one loses condition
                idx_loses_condition = loses_condition[loses_condition].index
                df.loc[idx_loses_condition, condition] = False

            # -------------------- DEATH FROM NCD CONDITION ---------------------------------------
            # There is a risk of death for those who have an NCD condition. Death is assumed to happen instantly.

            # Strip leading 'nc_' from condition name
            condition_name = condition.replace('nc_', '')

            eligible_population = df.is_alive & df[condition]
            selected_to_die = self.module.lms_death[condition].predict(df.loc[eligible_population], rng)
            if selected_to_die.any():  # catch in case no one dies
                idx_selected_to_die = selected_to_die[selected_to_die].index

                for person_id in idx_selected_to_die:
                    self.sim.schedule_event(
                        InstantaneousDeath(self.module, person_id, f"{condition_name}"), self.sim.date
                    )

        # add the new incidence numbers to tracker
        self.module.df_incidence_tracker = self.module.df_incidence_tracker.add(current_incidence_df)

        # Determine occurrence of events
        for event in self.module.events:

            eligible_population_for_event = df.is_alive
            has_event = self.module.lms_event_onset[event].predict(df.loc[eligible_population_for_event], rng)
            if has_event.any():  # catch in case no one has event
                idx_has_event = has_event[has_event].index

                for person_id in idx_has_event:
                    self.sim.schedule_event(NcdEvent(self.module, person_id, event),
                                            self.sim.date + DateOffset(days=self.module.rng.randint(0, 90)))

            # -------------------- DEATH FROM NCD EVENT ---------------------------------------
            # There is a risk of death for those who have had an NCD event. Death is assumed to happen instantly.

            # Strip leading 'nc_' from condition name
            event_name = event.replace('nc_ever_', '')

            eligible_population = df.is_alive & df[event]
            selected_to_die = self.module.lms_event_death[event].predict(df.loc[eligible_population], rng)
            if selected_to_die.any():  # catch in case no one dies
                idx_selected_to_die = selected_to_die[selected_to_die].index

                for person_id in idx_selected_to_die:
                    self.sim.schedule_event(
                        InstantaneousDeath(self.module, person_id, f"{event_name}"), self.sim.date
                    )


class NcdEvent(Event, IndividualScopeEventMixin):
    """
    This is a Stroke event. It has been scheduled to occur by the Ncds_MainPollingEvent.
    """

    def __init__(self, module, person_id, event):
        super().__init__(module, person_id=person_id)
        self.event = event
        self.event_name = event.replace('nc_ever_', '')

    def apply(self, person_id):
        if not self.sim.population.props.at[person_id, 'is_alive']:
            return

        self.module.eventsTracker[f'{self.event_name}_events'] += 1
        self.sim.population.props.at[person_id, f'{self.event}'] = True

        # TODO: @britta add functionality to add symptoms

        ## Add the outward symptom to the SymptomManager. This will result in emergency care being sought
        # self.sim.modules['SymptomManager'].change_symptom(
        #    person_id=person_id,
        #    disease_module=self.module,
        #    add_or_remove='+',
        #    symptom_string='Damage_From_Stroke'
        # )


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
#
#   Put the logging events here. There should be a regular logger outputting current states of the
#   population. There may also be a loggig event that is driven by particular events.
# ---------------------------------------------------------------------------------------------------------

class Ncds_LoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summary of the numbers of people with respect to the action of this module.
        """

        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        self.date_last_run = self.sim.date
        self.AGE_RANGE_LOOKUP = de.Demography.AGE_RANGE_LOOKUP
        assert isinstance(module, Ncds)

    def apply(self, population):

        # update depression status
        self.sim.population.props['nc_depression'] = self.sim.population.props['de_depr']

        # Convert incidence tracker to dict to pass through logger
        logger.info(key='incidence_count_by_condition', data=self.module.df_incidence_tracker.to_dict())
        # Reset the counter
        self.module.df_incidence_tracker = copy.deepcopy(self.module.df_incidence_tracker_zeros)

        def age_cats(ages_in_years):
            dem = demography.Demography()
            AGE_RANGE_CATEGORIES = dem.AGE_RANGE_CATEGORIES
            AGE_RANGE_LOOKUP = dem.AGE_RANGE_LOOKUP

            age_cats = pd.Series(
                pd.Categorical(ages_in_years.map(AGE_RANGE_LOOKUP),
                               categories=AGE_RANGE_CATEGORIES, ordered=True)
            )
            return age_cats

        # Output the person-years lived by single year of age in the past year
        df = self.sim.population.props
        delta = pd.DateOffset(years=1)
        for cond in self.module.conditions:
            # mask is a Series restricting dataframe to individuals who do not have the condition, which is passed to
            # demography module to calculate person-years lived without the condition
            mask = (df.is_alive & ~df[f'{cond}'])
            py = de.Demography.calc_py_lived_in_last_year(self, delta, mask)
            py['age_range'] = age_cats(py.index)
            py = py.groupby('age_range').sum()
            logger.info(key=f'person_years_{cond}', data=py.to_dict())

        # Make some summary statistics for prevalence by age/sex for each condition
        df = self.sim.population.props
        # make custom age range cats
        df_age_cats = df
        df_age_cats['custom_age_range'] = df_age_cats['age_range']
        df_age_cats['custom_age_range'] = np.where(df_age_cats['age_years'] >= 80, '80+', df_age_cats['age_range'])

        def proportion_of_something_in_a_groupby_ready_for_logging(df, something, groupbylist):
            dfx = df.groupby(groupbylist).apply(lambda dft: pd.Series(
                {'something': dft[something].sum(), 'not_something': (~dft[something]).sum()}))
            pr = dfx['something'] / dfx.sum(axis=1)

            # create into a dict with keys as strings
            pr = pr.reset_index()
            pr['flat_index'] = ''
            for i in range(len(pr)):
                pr.at[i, 'flat_index'] = '__'.join([f"{col}={pr.at[i, col]}" for col in groupbylist])
            pr = pr.set_index('flat_index', drop=True)
            pr = pr.drop(columns=groupbylist)
            return pr[0].to_dict()

        # Prevalence of conditions broken down by sex and age

        for condition in self.module.conditions:
            # Strip leading 'nc_' from condition name
            condition_name = condition.replace('nc_', '')
            prev_age_sex = proportion_of_something_in_a_groupby_ready_for_logging(df_age_cats, f'{condition}',
                                                                                  ['sex', 'custom_age_range'])

            # Prevalence of conditions broken down by sex and age
            logger.info(
                key=f'{condition_name}_prevalence_by_age_and_sex',
                description='current fraction of the population classified as having condition, by sex and age',
                data={'data': prev_age_sex}
            )

            # Prevalence of conditions by adults aged 20 or older
            adult_prevalence = {
                'prevalence': len(df[df[f'{condition}'] & df.is_alive & (df.age_years >= 20)]) / len(
                    df[df.is_alive & (df.age_years >= 20)])}

            logger.info(
                key=f'{condition_name}_prevalence',
                description='current fraction of the adult population classified as having condition',
                data=adult_prevalence
            )

        # Counter for number of co-morbidities
        df = self.sim.population.props
        mask = df.is_alive
        df.loc[mask, 'n_conditions'] = df.loc[mask, self.module.extended_conditions].sum(axis=1)
        n_comorbidities_all = pd.DataFrame(index=self.module.age_index,
                                           columns=list(range(0, len(self.module.extended_conditions) + 1)))
        df = df[['age_range', 'n_conditions']]

        for num in range(0, len(self.module.extended_conditions) + 1):
            col = df.loc[df['n_conditions'] == num].groupby(['age_range']).apply(lambda x: pd.Series(
                {'count': x['n_conditions'].count()}))
            n_comorbidities_all.loc[:, num] = col['count']

        prop_comorbidities_all = n_comorbidities_all.div(n_comorbidities_all.sum(axis=1), axis=0)

        logger.info(key='mm_prevalence_by_age_all',
                    description='annual summary of multi-morbidities by age for all',
                    data=prop_comorbidities_all.to_dict()
                    )

        # output combinations of different conditions

        df = self.sim.population.props
        df = df[df.is_alive]
        df['four_age_bins'] = pd.cut(x=df['age_years'], bins=[0, 20, 45, 65, 120], right=False)
        df['all_adults'] = pd.cut(x=df['age_years'], bins=[0, 20, 120], right=False)

        combos = combinations(self.module.extended_conditions, 2)
        condition_combos = list(combos)

        n_combos = pd.DataFrame(index=df['four_age_bins'].value_counts().sort_index().index)

        for i in range(0, len(condition_combos)):
            df['condition_combos'] = np.where(df[condition_combos[i][0]] & df[condition_combos[i][1]], True, False)
            col = df.loc[df['condition_combos']].groupby(['four_age_bins'])['condition_combos'].count()
            n_combos.reset_index()
            n_combos.loc[:, (f'{condition_combos[i][0]}' + '_' + f'{condition_combos[i][1]}')] = col.values

        # output proportions of different combinations of conditions

        prop_combos = n_combos.div(df.groupby(['four_age_bins'])['four_age_bins'].count(), axis=0)
        prop_combos.index = prop_combos.index.astype(str)

        logger.info(key='prop_combos',
                    description='proportion of combinations of morbidities',
                    data=prop_combos.to_dict()
                    )

        # output entire dataframe for logistic regression

        df = self.sim.population.props
        df.to_csv('df_for_regression.csv')
