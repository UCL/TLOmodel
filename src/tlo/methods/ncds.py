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
        'baseline_prob_ldl_hdl': Parameter(Types.REAL, 'baseline prob of ldl hdl'),
        'rr_male': Parameter(Types.REAL, 'rr if male')
    }

    # Note that all properties must have a two letter prefix that identifies them to this module.

    PROPERTIES = {
        # These are all the states:
        'nc_ldl_hdl': Property(Types.BOOL, 'Whether or not someone currently has LDL/HDL'),
        'nc_chronic_inflammation': Property(Types.BOOL, 'Whether or not someone currently has chronic inflammation'),
        'nc_diabetes': Property(Types.BOOL, 'Whether or not someone currently has diabetes'),
        'nc_hypertension': Property(Types.BOOL, 'Whether or not someone currently has hypertension'),
        'nc_depression': Property(Types.BOOL, 'Whether or not someone currently has depression'),
        'nc_muscoskeletal': Property(Types.BOOL, 'Whether or not someone currently has muscoskeletal conditions'),
        'nc_frailty': Property(Types.BOOL, 'Whether or not someone currently has frailty'),
        'nc_chronic_lower_back_pain': Property(Types.BOOL, 'Whether or not someone currently has chronic lower back pain'),
        'nc_arthritis': Property(Types.BOOL, 'Whether or not someone currently has arthritis'),
        'nc_vision_disorders': Property(Types.BOOL, 'Whether or not someone currently has vision disorders'),
        'nc_chronic_liver_disease': Property(Types.BOOL, 'Whether or not someone currently has chronic liver disease'),
        'nc_chronic_kidney_disease': Property(Types.BOOL, 'Whether or not someone currently has chronic kidney disease'),
        'nc_chronic_ischemic_hd': Property(Types.BOOL, 'Whether or not someone currently has chronic ischemic heart disease'),
        'nc_lower_extremity_disease': Property(Types.BOOL, 'Whether or not someone currently has lower extremity disease'),
        'nc_dementia': Property(Types.BOOL, 'Whether or not someone currently has dementia'),
        'nc_bladder_cancer': Property(Types.BOOL, 'Whether or not someone currently has bladder cancer'),
        'nc_oesophageal_cancer': Property(Types.BOOL, 'Whether or not someone currently has oesophageal cancer'),
        'nc_breast_cancer': Property(Types.BOOL, 'Whether or not someone currently has breast cancer'),
        'nc_prostate_cancer': Property(Types.BOOL, 'Whether or not someone currently has prostate cancer'),
        'nc_other_cancers': Property(Types.BOOL, 'Whether or not someone currently has other cancers'),
        'nc_chronic_respiratory_disease': Property(Types.BOOL, 'Whether or not someone currently has chronic respiratory disease'),
        'nc_other_infections': Property(Types.BOOL, 'Whether or not someone currently has other infections'),
    }

    # TODO: we will have to later gather from the others what the symptoms are in each state - for now leave blank
    SYMPTOMS = {}

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # save a list of the conditions that covered in this module (extracted from PROPERTIES)
        self.conditions = list(self.PROPERTIES.keys())


    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        To access files use: Path(self.resourcefilepath) / file_name
        """
        xls = pd.ExcelFile(
            Path(self.resourcefilepath / 'ResourceFile_NCDs.xlsx')
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

        self.parameters['baseline_prob_ldl_hdl'] = 0.06
        self.parameters['rr_male'] = 1.2


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
            self.lms_onset[condition] = self.build_linear_model(condition,self.parameters['interval_between_polls'])
            self.lms_removal[condition] = self.build_linear_model(condition,self.parameters['interval_between_polls'])


    def build_linear_model(self, condition, interval_between_polls):
        """
        :param_dict: the dict read in from the resourcefile
        :param interval_between_polls: the duration (in months) between the polls
        :return: a linear model
        """

        # read in parameters from resource file

        xls = pd.ExcelFile(
            Path(self.resourcefilepath / 'ResourceFile_NCDs.xlsx')
        )

        #def read_excel_sheet(df):
            #"""Helper function to read in the sheet"""
            #pass

        p = self.parameters

        self.lms_onset[condition] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            self.parameters['baseline_prob_ldl_hdl'],
            Predictor().when('(sex=="M")', p['rr_male'])
        )

        self.lms_removal[condition] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            self.parameters['baseline_prob_ldl_hdl'],
            Predictor().when('(sex=="M")', p['rr_male'])
        )

        return self.lms_onset, self.lms_removal

        # todo: @Britta - adjust the rates according to the frequency at which the MainPollingEvent will be called



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

        # Determine onset/removal of conditions
        for condition in self.module.conditions:
            # onset:
            df.loc[
                self.module.lms_onset[condition].predict(df.loc[df.is_alive & ~df[condition]
                                                                ],
                                                         self.module.rng), condition] = True

            # removal:
            df.loc[
                self.module.lms_removal[condition].predict(df.loc[df.is_alive & df[condition]
                                                                ],
                                                         self.module.rng), condition] = False




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
        assert isinstance(module, Ncds)

    def apply(self, population):
        # Make some summary statistics for prevalence by age/sex for each condition
        df = population.props

        for condition in self.module.conditions:
            dict_for_output = pd.DataFrame(df.loc[df.is_alive].groupby(by=['sex', 'age-group'])[condition].mean()).reset_index().to_dict()
            logger.info(key=f'prevalence_{condition}', data=dict_for_output)



        # TODO: @Britta - need to do incidence do, by age. I think the counter approach in Diarrhoea would work
