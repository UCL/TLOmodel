"""Placeholder for childhood wasting module."""

from tlo import Module, Property, Types, logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Wasting(Module):
    """Placeholder for childhood wasting module.

    Provides dummy values for properties required by other modules.
    """

    INIT_DEPENDENCIES = {'Demography'}

    PROPERTIES = {
        "un_clinical_acute_malnutrition": Property(
            Types.CATEGORICAL,
            "temporary property",
            categories=["MAM", "SAM", "well"],
            default_property_value="well",
        ),
        "un_ever_wasted": Property(Types.BOOL, "temporary property"),
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name=name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        pass

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother, child):
        df = self.sim.population.props
        df.at[child, 'un_clinical_acute_malnutrition'] = 'well'
        df.at[child, 'un_ever_wasted'] = False
