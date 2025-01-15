"""Placeholder for childhood wasting module."""
from tlo import Module, Property, Types, logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Wasting(Module):
    """Placeholder for childhood wasting module.

    Provides dummy values for properties required by other modules.
    """

    INIT_DEPENDENCIES = {'Demography'}

    PROPERTIES = {
        'un_clinical_acute_malnutrition': Property(Types.CATEGORICAL,
                                                   'temporary property', categories=['MAM', 'SAM', 'well']),
        'un_ever_wasted': Property(Types.BOOL, 'temporary property')
    }

    def __init__(self, name=None):
        super().__init__(name=name)

    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        pass

    def initialise_population(self, population):
        df = population.props
        df.loc[df.is_alive, 'un_clinical_acute_malnutrition'] = 'well'
        df.loc[df.is_alive, 'un_ever_wasted'] = False

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother, child):
        df = self.sim.population.props
        df.at[child, 'un_clinical_acute_malnutrition'] = 'well'
        df.at[child, 'un_ever_wasted'] = False
