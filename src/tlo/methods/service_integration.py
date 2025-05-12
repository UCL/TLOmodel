from __future__ import annotations

from typing import TYPE_CHECKING, List
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DAYS_IN_YEAR, Date, DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.methods.hsi_generic_first_appts import GenericFirstAppointmentsMixin
from tlo.util import read_csv_files
from tlo.lm import LinearModel, LinearModelType

from tlo.methods.labour import LabourAndPostnatalCareAnalysisEvent
from tlo.methods.contraception import StartInterventions

if TYPE_CHECKING:
    from tlo.methods.hsi_generic_first_appts import HSIEventScheduler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ServiceIntegration(Module, GenericFirstAppointmentsMixin):
    """
    """

    # Declare modules that need to be registered in simulation and initialised before
    # this module
    INIT_DEPENDENCIES = {'Demography'}


    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {}

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
    }

    PARAMETERS = {
        'serv_int_screening': Parameter(Types.LIST, 'Blank by default. Listed conditions are those for '
                                                    'which screening is increased as part of integration modelling'),
        'serv_int_chronic': Parameter(Types.BOOL, 'specify whether chronic care pathway is implemented'),
        'serv_int_mch': Parameter(Types.LIST, 'Blank by default. Listed conditions are those for '
                                                    'which maternal and child health care is increased as part of'
                                              ' integration modelling'),
        'integration_date': Parameter(Types.DATE, 'Date on which parameters are overidden for integration '
                                                  'modelling'),

    }

    PROPERTIES = {
    }

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.accepted_conditions = ['hiv', 'tb', 'htn', 'dm', 'fp', 'cc', 'mal', 'ncds', 'depression', 'epilepsy',
                                    'pnc', 'epi']

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        For now, we are going to hard code them explicity.
        Register the module with the health system and register the symptoms
        """
        parameter_dataframe = read_csv_files(self.resourcefilepath/'service integration',
                                             files='parameter_values')
        self.load_parameters_from_dataframe(parameter_dataframe)

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        pass

    def initialise_simulation(self, sim):

        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        params = self.parameters

        event = ServiceIntegrationParameterUpdateEvent(self)
        sim.schedule_event(event, params['integration_date'])

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """

        pass

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        pass

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        pass


class ServiceIntegrationParameterUpdateEvent(Event, PopulationScopeEventMixin):

    # This event is occuring regularly at one monthly intervals

    def __init__(self, module):
        super().__init__(module)
        assert isinstance(module, ServiceIntegration)

    def apply(self, population):
        params = self.module.parameters

        logger.info(key='event_runs', data='ServiceIntegrationParameterUpdateEvent is running')

        for p in [params['serv_int_screening'], params['serv_int_mch']]:
            if p:
                assert all(item in self.module.accepted_conditions for item in p)

        # TODO: rebuild linear models
        # TODO: check correct service names provided

        if not params['serv_int_screening'] and not params['serv_int_chronic'] and not params['serv_int_mch']:
            logger.info(key='event_cancelled', data='ServiceIntegrationParameterUpdateEvent did not run')
            return

        # ---------------------------------------------- SCREENING ----------------------------------------------------
        if 'htn' in params['serv_int_screening']:
            # Annual community screening in over 50s increased to 100%
            self.sim.modules['CardioMetabolicDisorders'].parameters['hypertension_hsi']['pr_assessed_other_symptoms'] = 1.0
            # Probability of screening when presenting to any generic first appointment set to 100%
            self.sim.modules['CardioMetabolicDisorders'].lms_testing['hypertension'] = LinearModel(LinearModelType.MULTIPLICATIVE, 1.0)

        if 'dm' in params['serv_int_screening']:
            # Probability of screening when presenting to any generic first appointment and not sympotmatic set to 100%
            self.sim.modules['CardioMetabolicDisorders'].parameters['diabetes_hsi']['pr_assessed_other_symptoms'] = 1.0

        if 'fp' in params['serv_int_screening']:
            # Here we use the in-built functionality of the contraception model to increase the coverage of modern
            # methods of contraception. When 'fp' is listed in params['serv_int_screening'] the probability of
            # initiation in the general female population is increased. See updates to contraception.py

            # Todo: may need to increase coverage further!
            self.sim.schedule_event(StartInterventions(self.sim.modules['Contraception']), Date(self.sim.date))

        if 'mal' in params['serv_int_screening']:
            self.sim.modules['Stunting'].parameters['prob_stunting_diagnosed_at_generic_appt'] = 1.0

        if 'hiv' in params['serv_int_screening']:
            # annual testing rate used in HIV scale-up scenarios, default average (2010-2020) is 0.25
            p["hiv_testing_rates"]["annual_testing_rate_adults"] = 0.4

        if 'tb' in params['serv_int_screening']:
            # increase treatment coverage rate used to infer rate testing for active tb, default is 0.75
            p["rate_testing_active_tb"]["treatment_coverage"] = 90

        # ------------------------------------ MATERNAL AND CHILD HEALTH CLINIC ---------------------------------------
        if 'pnc' in params['serv_int_mch']:
            #
            self.sim.modules['Labour'].current_parameters['alternative_pnc_coverage'] = True
            self.sim.modules['Labour'].current_parameters['pnc_availability_odds'] = 15.0
            self.sim.schedule_event(LabourAndPostnatalCareAnalysisEvent(self.sim.modules['Labour']), Date(self.sim.date))

        if 'fp' in params['serv_int_mch']:
            # Here we use the in-built functionality of the contraception model to increase the coverage of modern
            # methods of contraception. When 'fp' is listed in params['serv_int_mch'] the probability of
            # initiation following birth is increased. See updates to contraception.py
            self.sim.schedule_event(StartInterventions(self.sim.modules['Contraception']), Date(self.sim.date))

        if 'mal' in params['serv_int_mch']:
            self.sim.modules['Stunting'].parameters['prob_stunting_diagnosed_at_generic_appt'] = 1.0

        # Todo: EPI intervention
        if 'epi' in params['serv_int_mch']:
            pass

        # ------------------------------------- CHRONIC CARE CLINIC ---------------------------------------------------
        if params['serv_int_chronic']:
            self.sim.modules['Hiv'].parameters['virally_suppressed_on_art'] = 1.0
            self.sim.modules['Tb'].parameters['tb_prob_tx_success_ds'] = 0.9
            self.sim.modules['Tb'].parameters['tb_prob_tx_success_mdr'] = 0.9
            self.sim.modules['Tb'].parameters['tb_prob_tx_success_0_4'] = 0.9
            self.sim.modules['Tb'].parameters['tb_prob_tx_success_5_14'] = 0.9
            self.sim.modules['Epilepsy'].parameters['prob_start_anti_epilep_when_seizures_detected_in_generic_first_appt'] = 1.0
            self.sim.modules['Depression'].parameters['pr_assessed_for_depression_in_generic_appt_level1'] = 1.0





