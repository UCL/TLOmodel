"""
The joint NCDs model by Tim Hallett and Britta Jewell, October 2020

"""
from pathlib import Path

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods.healthsystem import HSI_Event

import pandas as pd
import copy

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Ncds(Module):
    """
    One line summary goes here...

    """
    # Declare Metadata (this is for a typical 'Disease Module')
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }


    PARAMETERS = {
        'interval_between_polls': Parameter(Types.INT, 'months between the main polling event'),
        'baseline_annual_probability': Parameter(Types.REAL, 'baseline annual probability of acquiring/losing condition'),
        'rr_male': Parameter(Types.REAL, 'rr if male'),
        'rr_0_4': Parameter(Types.REAL, 'rr if 0-4'),
        'rr_5_9': Parameter(Types.REAL, 'rr if 5-9'),
        'rr_10_14': Parameter(Types.REAL, 'rr if 10-14'),
        'rr_15_19': Parameter(Types.REAL, 'rr if 15-19'),
        'rr_20_24': Parameter(Types.REAL, 'rr if 20-24'),
        'rr_25_29': Parameter(Types.REAL, 'rr if 25-29'),
        'rr_30_34': Parameter(Types.REAL, 'rr if 30-34'),
        'rr_35_39': Parameter(Types.REAL, 'rr if 35-39'),
        'rr_40_44': Parameter(Types.REAL, 'rr if 40-44'),
        'rr_45_49': Parameter(Types.REAL, 'rr if 45-49'),
        'rr_50_54': Parameter(Types.REAL, 'rr if 50-54'),
        'rr_55_59': Parameter(Types.REAL, 'rr if 55-59'),
        'rr_60_64': Parameter(Types.REAL, 'rr if 60-64'),
        'rr_65_69': Parameter(Types.REAL, 'rr if 65-69'),
        'rr_70_74': Parameter(Types.REAL, 'rr if 70-74'),
        'rr_75_79': Parameter(Types.REAL, 'rr if 75-79'),
        'rr_80_84': Parameter(Types.REAL, 'rr if 80-84'),
        'rr_85_89': Parameter(Types.REAL, 'rr if 85-89'),
        'rr_90_94': Parameter(Types.REAL, 'rr if 90-94'),
        'rr_95_99': Parameter(Types.REAL, 'rr if 95-99'),
        'rr_100': Parameter(Types.REAL, 'rr if 100+'),
        'rr_urban': Parameter(Types.REAL, 'rr if living in an urban area'),
        'rr_wealth_1': Parameter(Types.REAL, 'rr if wealth 1'),
        'rr_wealth_2': Parameter(Types.REAL, 'rr if wealth 2'),
        'rr_wealth_3': Parameter(Types.REAL, 'rr if wealth 3'),
        'rr_wealth_4': Parameter(Types.REAL, 'rr if wealth 4'),
        'rr_wealth_5': Parameter(Types.REAL, 'rr if wealth 5'),
        'rr_bmi_1': Parameter(Types.REAL, 'rr if bmi 1'),
        'rr_bmi_2': Parameter(Types.REAL, 'rr if bmi 2'),
        'rr_bmi_3': Parameter(Types.REAL, 'rr if bmi 3'),
        'rr_bmi_4': Parameter(Types.REAL, 'rr if bmi 4'),
        'rr_bmi_5': Parameter(Types.REAL, 'rr if bmi 5'),
        'rr_low_exercise': Parameter(Types.REAL, 'rr if low exercise'),
        'rr_high_salt': Parameter(Types.REAL, 'rr if high salt'),
        'rr_high_sugar': Parameter(Types.REAL, 'rr if high sugar'),
        'rr_tobacco': Parameter(Types.REAL, 'rr if tobacco'),
        'rr_alcohol': Parameter(Types.REAL, 'rr if alcohol'),
        'rr_marital_status_1': Parameter(Types.REAL, 'rr if never married'),
        'rr_marital_status_2': Parameter(Types.REAL, 'rr if currently married'),
        'rr_marital_status_3': Parameter(Types.REAL, 'rr if widowed or divorced'),
        'rr_in_education': Parameter(Types.REAL, 'rr if in education'),
        'rr_current_education_level_1': Parameter(Types.REAL, 'rr if education level 1'),
        'rr_current_education_level_2': Parameter(Types.REAL, 'rr if education level 2'),
        'rr_current_education_level_3': Parameter(Types.REAL, 'rr if education level 3'),
        'rr_unimproved_sanitation': Parameter(Types.REAL, 'rr if unimproved sanitation'),
        'rr_no_access_handwashing': Parameter(Types.REAL, 'rr if no access to handwashing'),
        'rr_no_clean_drinking_water': Parameter(Types.REAL, 'rr if no access to drinking water'),
        'rr_wood_burning_stove': Parameter(Types.REAL, 'rr if wood-burning stove'),
        'rr_ldl_hdl': Parameter(Types.REAL, 'rr if currently has ldl/hdl'),
        'rr_chronic_inflammation': Parameter(Types.REAL, 'rr if currently has chronic inflammation'),
        'rr_diabetes': Parameter(Types.REAL, 'rr if currently has diabetes'),
        'rr_hypertension': Parameter(Types.REAL, 'rr if currently has hypertension'),
        'rr_depression': Parameter(Types.REAL, 'rr if currently has depression'),
        'rr_chronic_kidney_disease': Parameter(Types.REAL, 'rr if currently has chronic kidney disease'),
        'rr_chronic_ischemic_heart_disease': Parameter(Types.REAL, 'rr if currently has chronic ischemic heart disease'),
        'rr_stroke': Parameter(Types.REAL, 'rr if currently has stroke'),
        'rr_cancers': Parameter(Types.REAL, 'rr if currently has cancers'),
    }

    # Note that all properties must have a two letter prefix that identifies them to this module.

    PROPERTIES = {
        # These are all the states:
        'nc_ldl_hdl': Property(Types.BOOL, 'Whether or not someone currently has LDL/HDL'),
        'nc_chronic_inflammation': Property(Types.BOOL, 'Whether or not someone currently has chronic inflammation'),
        'nc_diabetes': Property(Types.BOOL, 'Whether or not someone currently has diabetes'),
        'nc_hypertension': Property(Types.BOOL, 'Whether or not someone currently has hypertension'),
        'nc_depression': Property(Types.BOOL, 'Whether or not someone currently has depression'),
        #'nc_muscoskeletal': Property(Types.BOOL, 'Whether or not someone currently has muscoskeletal conditions'),
        #'nc_frailty': Property(Types.BOOL, 'Whether or not someone currently has frailty'),
        'nc_chronic_lower_back_pain': Property(Types.BOOL, 'Whether or not someone currently has chronic lower back pain'),
        #'nc_arthritis': Property(Types.BOOL, 'Whether or not someone currently has arthritis'),
        #'nc_vision_disorders': Property(Types.BOOL, 'Whether or not someone currently has vision disorders'),
        #'nc_chronic_liver_disease': Property(Types.BOOL, 'Whether or not someone currently has chronic liver disease'),
        'nc_chronic_kidney_disease': Property(Types.BOOL, 'Whether or not someone currently has chronic kidney disease'),
        'nc_chronic_ischemic_hd': Property(Types.BOOL, 'Whether or not someone currently has chronic ischemic heart disease'),
        #'nc_lower_extremity_disease': Property(Types.BOOL, 'Whether or not someone currently has lower extremity disease'),
        #'nc_dementia': Property(Types.BOOL, 'Whether or not someone currently has dementia'),
        #'nc_bladder_cancer': Property(Types.BOOL, 'Whether or not someone currently has bladder cancer'),
        #'nc_oesophageal_cancer': Property(Types.BOOL, 'Whether or not someone currently has oesophageal cancer'),
        #'nc_breast_cancer': Property(Types.BOOL, 'Whether or not someone currently has breast cancer'),
        #'nc_prostate_cancer': Property(Types.BOOL, 'Whether or not someone currently has prostate cancer'),
        'nc_cancers': Property(Types.BOOL, 'Whether or not someone currently has cancers'),
        #'nc_chronic_respiratory_disease': Property(Types.BOOL, 'Whether or not someone currently has chronic respiratory disease'),
        #'nc_other_infections': Property(Types.BOOL, 'Whether or not someone currently has other infections'),
    }

    # TODO: we will have to later gather from the others what the symptoms are in each state - for now leave blank
    SYMPTOMS = {}

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # save a list of the conditions that covered in this module (extracted from PROPERTIES)
        self.conditions = list(self.PROPERTIES.keys())

        # dict to hold counters for the number of episodes by condition-type and age-group
        # (0yrs, 1yrs, 2-4yrs)
        blank_counter = dict(zip(self.conditions, [list() for _ in self.conditions]))
        self.incident_case_tracker_blank = {
            '0-10y': copy.deepcopy(blank_counter),
            '11-20y': copy.deepcopy(blank_counter),
            '21-30y': copy.deepcopy(blank_counter),
            '31-40y': copy.deepcopy(blank_counter),
            '41-50y': copy.deepcopy(blank_counter),
            '51-60y': copy.deepcopy(blank_counter),
            '61-70y': copy.deepcopy(blank_counter),
            '71-80y': copy.deepcopy(blank_counter),
            '81-90y': copy.deepcopy(blank_counter),
            '91-100y': copy.deepcopy(blank_counter),
            '100+y': copy.deepcopy(blank_counter)
        }
        self.incident_case_tracker = copy.deepcopy(self.incident_case_tracker_blank)

        zeros_counter = dict(zip(self.conditions, [0] * len(self.conditions)))
        self.incident_case_tracker_zeros = {
            '0-10y': copy.deepcopy(zeros_counter),
            '11-20y': copy.deepcopy(zeros_counter),
            '21-30y': copy.deepcopy(zeros_counter),
            '31-40y': copy.deepcopy(zeros_counter),
            '41-50y': copy.deepcopy(zeros_counter),
            '51-60y': copy.deepcopy(zeros_counter),
            '61-70y': copy.deepcopy(zeros_counter),
            '71-80y': copy.deepcopy(zeros_counter),
            '81-90y': copy.deepcopy(zeros_counter),
            '91-100y': copy.deepcopy(zeros_counter),
            '100+y': copy.deepcopy(zeros_counter)
        }


    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        To access files use: Path(self.resourcefilepath) / file_name
        """
        xls = pd.ExcelFile(
            Path(self.resourcefilepath / 'ResourceFile_NCDs2.xlsx')
        )

        # check that we have got parameters for each of the conditions
        # assert {xls.sheet_names} == {self.conditions}

        def read_excel_sheet(df):
            """Helper function to read in the sheet"""
            pass

        self.params_dict = dict()
        for condition in self.conditions:
            self.params_dict[condition] = read_excel_sheet(pd.read_excel(xls, condition))


        # Set the interval (in months) between the polls
        self.parameters['interval_between_polls'] = 3


    def initialise_population(self, population):
        """Set our property values for the initial population.
        """
        # TODO: @britta - we might need to gather this info from the others too: or it might that we have to find
        #  this through fitting. For now, let there be no conditions for anyone

        df = population.props
        for condition in self.conditions:
            df[condition] = False


    def initialise_simulation(self, sim):
        """Schedule:
        * Main Polling Event
        * Main Logging Event
        * Build the LinearModels for the onset/removal of each condition:
        """
        sim.schedule_event(Ncds_MainPollingEvent(self, self.parameters['interval_between_polls']), sim.date)
        sim.schedule_event(Ncds_LoggingEvent(self), sim.date)

        # Build the LinearModel for onset/removal of each condition
        self.lms_onset = dict()
        self.lms_removal = dict()

        for condition in self.conditions:
            self.lms_onset[condition] = self.build_linear_model(condition, self.parameters['interval_between_polls'])
            #self.lms_removal[condition] = self.build_linear_model(condition, self.parameters['interval_between_polls'])


    def build_linear_model(self, condition, interval_between_polls):
        """
        :param_dict: the dict read in from the resourcefile
        :param interval_between_polls: the duration (in months) between the polls
        :return: a linear model
        """

        # read in parameters from resource file
        # ResourceFile_NCDs2.xlsx = simplified version with no removal of conditions

        if condition == 'nc_ldl_hdl':
            self.load_parameters_from_dataframe(
                pd.read_excel(Path(self.resourcefilepath) / "ResourceFile_NCDs2.xlsx",
                            sheet_name="nc_ldl_hdl")
            )
        elif condition == 'nc_chronic_inflammation':
            self.load_parameters_from_dataframe(
                pd.read_excel(Path(self.resourcefilepath) / "ResourceFile_NCDs2.xlsx",
                              sheet_name="nc_chronic_inflammation")
            )
        elif condition == 'nc_diabetes':
            self.load_parameters_from_dataframe(
                pd.read_excel(Path(self.resourcefilepath) / "ResourceFile_NCDs2.xlsx",
                              sheet_name="nc_diabetes")
            )
        elif condition == 'nc_hypertension':
            self.load_parameters_from_dataframe(
                pd.read_excel(Path(self.resourcefilepath) / "ResourceFile_NCDs2.xlsx",
                              sheet_name="nc_hypertension")
            )
        elif condition == 'nc_depression':
            self.load_parameters_from_dataframe(
                pd.read_excel(Path(self.resourcefilepath) / "ResourceFile_NCDs2.xlsx",
                              sheet_name="nc_depression")
            )
        elif condition == 'nc_chronic_lower_back_pain':
            self.load_parameters_from_dataframe(
                pd.read_excel(Path(self.resourcefilepath) / "ResourceFile_NCDs2.xlsx",
                              sheet_name="nc_chronic_lower_back_pain")
            )
        elif condition == 'nc_chronic_kidney_disease':
            self.load_parameters_from_dataframe(
                pd.read_excel(Path(self.resourcefilepath) / "ResourceFile_NCDs2.xlsx",
                              sheet_name="nc_chronic_kidney_disease")
            )
        elif condition == 'nc_chronic_ischemic_hd':
            self.load_parameters_from_dataframe(
                pd.read_excel(Path(self.resourcefilepath) / "ResourceFile_NCDs2.xlsx",
                              sheet_name="nc_chronic_ischemic_hd")
            )
        elif condition == 'nc_stroke':
            self.load_parameters_from_dataframe(
                pd.read_excel(Path(self.resourcefilepath) / "ResourceFile_NCDs2.xlsx",
                              sheet_name="nc_stroke")
            )
        elif condition == 'nc_cancers':
            self.load_parameters_from_dataframe(
                pd.read_excel(Path(self.resourcefilepath) / "ResourceFile_NCDs2.xlsx",
                              sheet_name="nc_cancers")
            )

        p = self.parameters
        p['baseline_annual_probability'] = p['baseline_annual_probability'] * (interval_between_polls / 12)

        self.lms_onset[condition] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            self.parameters['baseline_annual_probability'],
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
                .when('==2', p['rr_wealth_2'])
                .when('==3', p['rr_wealth_3'])
                .when('==4', p['rr_wealth_4'])
                .when('==5', p['rr_wealth_5']),
            Predictor('li_bmi').when('==1', p['rr_bmi_1'])
                .when('==2', p['rr_bmi_2'])
                .when('==3', p['rr_bmi_3'])
                .when('==4', p['rr_bmi_4'])
                .when('==5', p['rr_bmi_5']),
            Predictor('li_low_ex').when(True, p['rr_low_exercise']),
            Predictor('li_high_salt').when(True, p['rr_high_salt']),
            Predictor('li_high_sugar').when(True, p['rr_high_sugar']),
            Predictor('li_tob').when(True, p['rr_tobacco']),
            Predictor('li_ex_alc').when(True, p['rr_alcohol']),
            Predictor('li_mar_stat').when('==1', p['rr_marital_status_1'])
                .when('==2', p['rr_marital_status_2'])
                .when('==3', p['rr_marital_status_3']),
            Predictor('li_in_ed').when(True, p['rr_in_education']),
            Predictor('li_ed_lev').when('==1', p['rr_current_education_level_1'])
                .when('==2', p['rr_current_education_level_2'])
                .when('==3', p['rr_current_education_level_3']),
            Predictor('li_unimproved_sanitation').when(True, p['rr_unimproved_sanitation']),
            Predictor('li_no_access_handwashing').when(True, p['rr_no_access_handwashing']),
            Predictor('li_no_clean_drinking_water').when(True, p['rr_no_clean_drinking_water']),
            Predictor('li_wood_burn_stove').when(True, p['rr_wood_burning_stove'])
        )

        #self.lms_removal[condition] = LinearModel(
            #LinearModelType.MULTIPLICATIVE,
            #self.parameters['baseline_prob_ldl_hdl'],
            #Predictor().when('(sex=="M")', p['rr_male'])
        #)

        return self.lms_onset[condition]



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
        #This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        # To return a value of 0.0 (fully health) for everyone, use:
        # df = self.sim.popultion.props
        # return pd.Series(index=df.index[df.is_alive],data=0.0)

        # TODO: @britta - we will also have to gather information for daly weights for each condition and do a simple
        #  mapping to them from the properties. For now, anyone who has anything has a daly_wt of 0.1

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

        # Determine onset/removal of conditions
        for condition in self.module.conditions:

            # onset:
            eligible_population = df.is_alive & ~df[condition]
            acquires_condition = self.module.lms_onset[condition].predict(df.loc[df.is_alive], rng,
                                                                          eligible_population=eligible_population)
            idx_acquires_condition = acquires_condition[acquires_condition].index
            df.loc[idx_acquires_condition, condition] = True

            # -------------------------------------------------------------------------------------------
            # Add this incident case to the tracker
            age = df.loc[idx_acquires_condition, ['age_years']]
            if age.values[0] < 11:
                age_grp = '0-10y'
            elif age.values[0] >= 11 & age.values[0] < 20:
                age_grp = '11-20y'
            elif age.values[0] >= 21 & age.values[0] < 30:
                age_grp = '21-30y'
            elif age.values[0] >= 31 & age.values[0] < 40:
                age_grp = '31-40y'
            elif age.values[0] >= 41 & age.values[0] < 50:
                age_grp = '41-50y'
            elif age.values[0] >= 51 & age.values[0] < 60:
                age_grp = '51-60y'
            elif age.values[0] >= 61 & age.values[0] < 70:
                age_grp = '61-70y'
            elif age.values[0] >= 71 & age.values[0] < 80:
                age_grp = '71-80y'
            elif age.values[0] >= 81 & age.values[0] < 90:
                age_grp = '81-90y'
            elif age.values[0] >= 91 & age.values[0] < 100:
                age_grp = '91-100y'
            else:
                age_grp = '100+y'
            self.module.incident_case_tracker[age_grp][condition].append(self.sim.date)
            # -------------------------------------------------------------------------------------------

            # removal:
            # df.loc[
                # self.module.lms_removal[condition].predict(df.loc[df.is_alive & df[condition]
                                                                # ],
                                                         # self.module.rng), condition] = False




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
        assert isinstance(module, Ncds)

    def apply(self, population):

        # Convert the list of timestamps into a number of timestamps
        # and check that all the dates have occurred since self.date_last_run
        counts = copy.deepcopy(self.module.incident_case_tracker_zeros)

        for age_grp in self.module.incident_case_tracker.keys():
            for condition in self.module.conditions:
                list_of_times = self.module.incident_case_tracker[age_grp][condition]
                counts[age_grp][condition] = len(list_of_times)
                for t in list_of_times:
                    assert self.date_last_run <= t <= self.sim.date

        logger.info(key='incidence_count_by_condition', data=counts)

        # Reset the counters and the date_last_run
        self.module.incident_case_tracker = copy.deepcopy(self.module.incident_case_tracker_blank)
        self.date_last_run = self.sim.date

        # Make some summary statistics for prevalence by age/sex for each condition
        df = population.props

        for condition in self.module.conditions:
            dict_for_output = pd.DataFrame(df.loc[df.is_alive].groupby(by=['sex', 'age_range'])[condition].mean()).reset_index().to_dict()
            logger.info(key=f'prevalence_{condition}', data=dict_for_output)



        # TODO: @Britta - need to do incidence do, by age. I think the counter approach in Diarrhoea would work
