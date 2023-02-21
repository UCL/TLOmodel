"""
The Chronic Obstructive Pulmonary Disease(COPD) module
Overview:
    - COPD is the name for a group of lung conditions that cause breathing difficulties.
    - It includes:
        emphysema – damage to the air sacs in the lungs
        chronic bronchitis – long-term inflammation of the airways
"""
from pathlib import Path

import pandas as pd
from tlo.methods import Metadata

from tlo import Module, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.causes import Cause
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Copd(Module):
    """ The module responsible for infecting individuals with Chronic Obstructive Pulmonary Diseases. It defines and initialises parameters and
    properties associated with COPD plus functions and events related to COPD
    """

    INIT_DEPENDENCIES = {}  # any other COPD dependency

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN}

    # a list of causes of death
    gbd_causes_of_copd_represented_in_this_module = [
        'Asthma',
        'Lower respiratory infections',
        'Upper respiratory infections',
        'Other chronic respiratory diseases'
        ]
    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'ChronicObstructivePulmonaryDiseases':
            Cause(gbd_causes={gbd_causes_of_copd_represented_in_this_module},
                  label='copd')
    }

    PARAMETERS = {

    }

    PROPERTIES = {
        'co_copd_diag': Property(
            Types.BOOL, 'Has the COPD ever been diagnosed'),
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = Path(resourcefilepath)

    def read_parameters(self, data_folder):
        """ A function to read all COPD parameters """
        # Create and register symptoms that this module will use:
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(
                name='breathless_mild',
            ),
            Symptom(
                name='breathless_severe',
            ),
        )

        p = self.parameters
        df = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_copd.xlsx')
        self.load_parameters_from_dataframe(df)

    def initialise_population(self, population):
        """ set COPD baseline values for all individuals

        :param population: the population of individuals
        """
        assert NotImplementedError

    def initialise_simulation(self, sim):
        """ Get ready for simulation start.

        :param sim: simulation object
        """

        assert NotImplementedError

    def on_birth(self, mother_id, child_id):
        """ Initialise COPD properties for a newborn individual.

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """

        assert NotImplementedError


class COPD_Event(RegularEvent, PopulationScopeEventMixin):
    """
    An event that controls the COPD infection process.
    """

    assert NotImplementedError


class HSI_COPD(HSI_Event, IndividualScopeEventMixin):
    """
    An event to that contains all COPD interactions with the Health System.
    """

    assert NotImplementedError
