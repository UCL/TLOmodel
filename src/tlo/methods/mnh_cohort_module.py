from pathlib import Path

import numpy as np
import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.methods import Metadata
from tlo.analysis.utils import parse_log_file
from tlo.events import Event, IndividualScopeEventMixin


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MaternalNewbornHealthCohort(Module):
    """

    """

    # INIT_DEPENDENCIES = {'Demography'}
    #
    # OPTIONAL_INIT_DEPENDENCIES = {''}
    #
    # ADDITIONAL_DEPENDENCIES = {''}

    # Declare Metadata (this is for a typical 'Disease Module')
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }
    CAUSES_OF_DEATH = {}
    CAUSES_OF_DISABILITY = {}
    PARAMETERS = {}
    PROPERTIES = {}

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        To access files use: Path(self.resourcefilepath) / file_name
        """
        pass

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        log_file = parse_log_file(
             '/Users/j_collins/PycharmProjects/TLOmodel/outputs/sejjj49@ucl.ac.uk/'
             'fullmodel_200k_cohort-2024-04-24T072206Z/0/0/fullmodel_200k_cohort__2024-04-24T072516.log',
             level=logging.DEBUG)['tlo.methods.contraception']

        all_pregnancies = log_file['properties_of_pregnant_person'].loc[
            log_file['properties_of_pregnant_person'].date.dt.year == 2024].drop(columns=['date'])
        all_pregnancies.index = [x for x in range(len(all_pregnancies))]

        preg_pop = all_pregnancies.loc[0:(len(self.sim.population.props))-1]

        props_dtypes = self.sim.population.props.dtypes
        preg_pop_final = preg_pop.astype(props_dtypes.to_dict())
        preg_pop_final.index.name = 'person'

        self.sim.population.props = preg_pop_final

        df = self.sim.population.props
        population = df.loc[df.is_alive]
        df.loc[population.index, 'date_of_last_pregnancy'] = self.sim.start_date
        df.loc[population.index, 'co_contraception'] = "not_using"


    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.

        """
        df = self.sim.population.props

        # Clear HSI queue for events scheduled during initialisation
        sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

        # Clear HSI queue for events scheduled during initialisation
        updated_event_queue = [item for item in self.sim.event_queue.queue
                               if not isinstance(item[3], IndividualScopeEventMixin)]

        self.sim.event_queue.queue = updated_event_queue

        # Prevent additional pregnancies from occurring
        self.sim.modules['Contraception'].processed_params['p_pregnancy_with_contraception_per_month'].iloc[:] = 0
        self.sim.modules['Contraception'].processed_params['p_pregnancy_no_contraception_per_month'].iloc[:] = 0

        # Set labour date for cohort women
        for person in df.index:
                self.sim.modules['Labour'].set_date_of_labour(person)

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        pass

    def report_daly_values(self):
        """
        This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        experienced by persons in the previous month. Only rows for alive-persons must be returned.
        If multiple causes in CAUSES_OF_DISABILITY are defined, a pd.DataFrame must be returned with a column
        corresponding to each cause (but if only one cause in CAUSES_OF_DISABILITY is defined, the pd.Series does not
        need to be given a specific name).

        To return a value of 0.0 (fully health) for everyone, use:
        df = self.sim.population.props
        return pd.Series(index=df.index[df.is_alive],data=0.0)
        """
        pass

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        pass
