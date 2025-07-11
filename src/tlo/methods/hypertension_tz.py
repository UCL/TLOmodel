from pathlib import Path

from tlo import Parameter, Types, Property, Module, Population, Simulation
from tlo.methods import Metadata


class HypertensionTz(Module):
    """
        Hypertension Module looking at cost and effectiveness of lifestyle interventions
        for hypertension in Tanzania
    """
    def __init__(self, resourcefilepath=None):
        super().__init__()

    METADATA = {Metadata.DISEASE_MODULE}

    PARAMETERS = {
        'prob_hyp': Parameter(Types.REAL, 'probability of having hypertension '),
        'prob_hyp_mild' : Parameter(Types.LIST, 'probabilities of hypertension classification'),
        'prob_hyp_severity': Parameter(Types.LIST, 'probabilities hypertension severity')
    }

    PROPERTIES = {
        'hyp_classification': Property(Types.CATEGORICAL, 'hypertension classification',
                                       categories=['primary', 'secondary']),
        'hyp_severity': Property(Types.CATEGORICAL, 'hypertension severity',
                                 categories=['mild', 'moderate', 'severe'])
    }

    def read_parameters(self, data_folder: str | Path) -> None:
        """ reading and assigning values to all parameters defined within this module """
        pass

    def initialise_population(self, population: Population) -> None:
        pass

    def initialise_simulation(self, sim: Simulation) -> None:
        pass

    def on_birth(self, mother_id: int, child_id: int) -> None:
        pass

