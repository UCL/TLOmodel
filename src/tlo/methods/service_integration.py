from __future__ import annotations

from typing import TYPE_CHECKING, List
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DAYS_IN_YEAR, DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.methods.hsi_generic_first_appts import GenericFirstAppointmentsMixin
from tlo.util import read_csv_files
from tlo.lm import LinearModel, LinearModelType


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
        self.accepted_conditions = ['hiv', 'tb', 'htn', 'gdm', 'fp', 'cc', 'mal', 'ncds', 'depression', 'epilepsy',
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
        df = self.sim.population.props
        params = self.module.parameters

        logger.debug(key='event_runs', data='ServiceIntegrationParameterUpdateEvent is running')

        for p in [params['serv_int_screening'], params['serv_int_chronic'], params['serv_int_mch']]:
            if p:
                assert all(item in self.module.accepted_conditions for item in p)

        # TODO: rebuild linear models
        # TODO: check correct service names provided

        # hiv_p = self.sim.modules['Hiv'].parameters
        # tb_p = self.sim.modules['Tb'].parameters
        # htn_p = self.sim.modules['TB'].parameters
        # dm_p = self.sim.modules['TB'].parameters
        # fp_p = self.sim.modules['Contraception'].parameters
        # cc_p = self.sim.modules['TB'].parameters
        # mal_p = self.sim.modules['TB'].parameters

        if not params['serv_int_screening'] and not params['serv_int_chronic'] and not params['serv_int_mch']:
            logger.debug(key='event_cancelled', data='ServiceIntegrationParameterUpdateEvent did not run')
            return

        if 'htn' in params['serv_int_screening']:
            pass
            # self.sim.modules['cmd'].parameters['hypertension_hsi']['pr_assessed_other_symptoms'] = 1.0
            # self.sim.modules['cmd'].lms_testing['hypertension'] = LinearModel(LinearModelType.MULTIPLICATIVE, 1.0)

        if 'dm' in params['serv_int_screening']:
            pass
            # self.sim.modules['cmd'].parameters['diabetes_hsi']['pr_assessed_other_symptoms'] = 1.0

        if 'hiv' in params['serv_int_screening']:
            pass
        if 'tb' in params['serv_int_screening']:
            pass


        if 'ncd' in params['serv_int_screening']:
             pass
        if 'fp' in params['serv_int_screening']:
            pass
        if 'cc' in params['serv_int_screening']:
            pass
        if 'mal' in params['serv_int_screening']:
            pass


        if 'hiv' in params['serv_int_chronic']:
            pass
        if 'tb' in params['serv_int_chronic']:
            pass
        if 'ncd' in params['serv_int_chronic']:
            pass
        if 'epilepsy' in params['serv_int_chronic']:
            pass
        if 'depression' in params['serv_int_chronic']:
            pass


        if 'pnc' in params['serv_int_mch']:
            pass
        if 'epi' in params['serv_int_mch']:
            pass
        if 'mal' in params['serv_int_mch']:
            pass
        if 'fp' in params['serv_int_mch']:
            pass







