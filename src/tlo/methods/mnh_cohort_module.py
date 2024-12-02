from pathlib import Path

import numpy as np
import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.methods import Metadata
from tlo.analysis.utils import parse_log_file
from tlo.events import Event, IndividualScopeEventMixin


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MaternalNewbornHealthCohort(Module):
    """
    When registered this module overrides the population data frame with a cohort of pregnant women. Cohort properties
    are sourced from a long run of the full model in which the properties of all newly pregnant women per year were
    logged. The cohort represents women in 2024.
    """

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
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        # Read in excel sheet with cohort
        all_preg_df = pd.read_excel(Path(f'{self.resourcefilepath}/maternal cohort') /
                                    'ResourceFile_All2024PregnanciesCohortModel.xlsx')

        # Only select rows equal to the desired population size
        if len(self.sim.population.props) <= len(all_preg_df):
            preg_pop = all_preg_df.loc[0:(len(self.sim.population.props))-1]
        else:
            # Calculate the number of rows needed to reach the desired length
            additional_rows = len(self.sim.population.props) - len(all_preg_df)

            # Initialize an empty DataFrame for additional rows
            rows_to_add = pd.DataFrame(columns=all_preg_df.columns)

            # Loop to fill the required additional rows
            while additional_rows > 0:
                if additional_rows >= len(all_preg_df):
                    rows_to_add = pd.concat([rows_to_add, all_preg_df], ignore_index=True)
                    additional_rows -= len(all_preg_df)
                else:
                    rows_to_add = pd.concat([rows_to_add, all_preg_df.iloc[:additional_rows]], ignore_index=True)
                    additional_rows = 0

            # Concatenate the original DataFrame with the additional rows
            preg_pop = pd.concat([all_preg_df, rows_to_add], ignore_index=True)


        # Set the dtypes and index of the cohort dataframe
        props_dtypes = self.sim.population.props.dtypes
        preg_pop_final = preg_pop.astype(props_dtypes.to_dict())
        preg_pop_final.index.name = 'person'

        # For the below columns we manually overwrite the dtypes
        for column in ['rt_injuries_for_minor_surgery', 'rt_injuries_for_major_surgery',
                       'rt_injuries_to_heal_with_time', 'rt_injuries_for_open_fracture_treatment',
                       'rt_injuries_left_untreated', 'rt_injuries_to_cast']:
            preg_pop_final[column] = [[] for _ in range(len(preg_pop_final))]

        # Set the population.props dataframe to the new cohort
        self.sim.population.props = preg_pop_final

        # Update key pregnancy properties
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

        # Clear the individual event queue for events scheduled during initialisation
        updated_event_queue = [item for item in self.sim.event_queue.queue
                               if not isinstance(item[3], IndividualScopeEventMixin)]
        self.sim.event_queue.queue = updated_event_queue

        # Prevent additional pregnancies from occurring during the cohort tun
        self.sim.modules['Contraception'].processed_params['p_pregnancy_with_contraception_per_month'].iloc[:] = 0
        self.sim.modules['Contraception'].processed_params['p_pregnancy_no_contraception_per_month'].iloc[:] = 0

        # Set labour date for cohort women
        for person in df.index:
                self.sim.modules['Labour'].set_date_of_labour(person)

    def on_birth(self, mother_id, child_id):
        pass

    def report_daly_values(self):
        pass

    def on_hsi_alert(self, person_id, treatment_id):
        pass
