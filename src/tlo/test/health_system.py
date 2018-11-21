"""
A skeleton template for disease methods.
"""
import pandas as pd

from tlo import Module, Parameter, Types


class health_system(Module):
    """ determines availability of ART for HIV+ dependent on UNAIDS coverage estimates
    """

    def __init__(self, name=None, workbook_path=None):
        super().__init__(name)
        self.workbook_path = workbook_path
        self.store = {'Time': [], 'Number_treated': []}

    PARAMETERS = {
        'art_coverage': Parameter(Types.DATA_FRAME, 'estimated ART coverage')
    }

    # no PROPERTIES

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """

        # TODO: check the sheet name
        self.parameters['art_coverage'] = pd.read_excel(self.workbook_path,
                                                        sheet_name='art_coverage')

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother, child):
        pass

    def treatment_available(self, person, art):
        pass
