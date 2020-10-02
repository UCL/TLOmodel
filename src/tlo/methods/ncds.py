"""
A skeleton template for disease methods.

"""
from pathlib import Path

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
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
        'interval_between_polls': Parameter(Types.INT, 'months between the main polling event')
    }

    # Note that all properties must have a two letter prefix that identifies them to this module.

    PROPERTIES = {
        # These are all the states:
        'diabetes': Property(Types.BOOL, 'Whether or not someone currently has diabetes'),
        'hypertension': Property(Types.BOOL, 'Whether or not someone currently has hypertension'),
        'frailty': Property(Types.BOOL, 'Whether or not someone currently has frailty'),
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
            Path(self.resourcefilepath / 'ResourceFile_Ncds.xlsx')
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
        sim.schedule_event(Ncds_MainPollingEvent(self, self.parameters['interval_between_polls']), self.date)
        sim.schedule_event(Ncds_LoggingEvent(self), self.date)

        # Build the LinearModel for onset/removal of each condition
        self.lms_onset = dict()
        self.lms_removal = dict()
        for condition in self.conditions:
            self.lms_onset[condition] = self.build_linear_model(self.params_dict[condition]['onset'])
            self.lms_removal[condition] = self.build_linear_model(self.params_dict[condition]['removal'])


    def build_linear_model(self, params_dict, interval_between_polls):
        """
        :param_dict: the dict read in from the resourcefile
        :param interval_between_polls: the duration (in months) between the polls
        :return: a linear model
        """

        # todo: @Britta - adjust the rates according to the freqyency at which the MainPollingEvent will be called



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
            logger.info(key=f'prevalence_{condition}', data=dict_to_output)



        # TODO: @Britta - need to do incidence do, by age. I think the counter approach in Diarrhoea would work
