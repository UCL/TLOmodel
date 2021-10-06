"""Placeholder for childhood stunting module."""

from tlo import Module, Property, Types, logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Stunting(Module):
    """Placeholder for childhood stunting module.

    Provides dummy values for properties required by other modules.
    """

    INIT_DEPENDENCIES = {'Demography'}

    PROPERTIES = {
        'un_HAZ_category':
        Property(
            Types.CATEGORICAL,
            'temporary property',
            categories=['HAZ<-3', '-3<=HAZ<-2', 'HAZ>=-2']
        ),
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name=name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        df = population.props
        df.loc[df.is_alive, 'un_HAZ_category'] = 'HAZ>=-2'

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother, child):
        df = self.sim.population.props
        df.at[child, 'un_HAZ_category'] = 'HAZ>=-2'
