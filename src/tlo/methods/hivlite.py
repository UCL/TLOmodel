# include comments on what this module is all about here
from pathlib import Path

import pandas as pd

from tlo import Module, Types, Parameter, Property, Population, Simulation
from tlo.events import RegularEvent, PopulationScopeEventMixin
from tlo.methods import Metadata


class HivLite(Module):
    def __init__(self, resourcefilepath=None):
        super().__init__()

    # define metadata
    METADATA = {
        Metadata.DISEASE_MODULE
    }
    #  define hivlite module parameters
    PARAMETERS = {
        'inf_rate': Parameter(Types.REAL, description='hiv infection rate'),
        'aids_prog_rate': Parameter(Types.REAL, description='aids progression rate'),
    }
    PROPERTIES = {
        'hl_hiv_status': Property(Types.STRING, description='hiv status'),
        'hl_date_inf': Property(Types.DATE, description='date of infection')
    }

    def read_parameters(self, data_folder: str | Path) -> None:
        param = self.parameters
        param['inf_rate'] = 0.05
        param['aids_prog_rate'] = 0.01

    def initialise_population(self, population: Population) -> None:
        df = population.props
        df.loc[df.is_alive, 'hl_hiv_status'] = 'Non-Reactive'
        df.loc[df.is_alive, 'hl_date_inf'] = pd.NaT

    def initialise_simulation(self, sim: Simulation) -> None:
        sim.schedule_event(HivInfectionEvent(self), sim.date + pd.DateOffset(months=1))

    def on_birth(self, mother_id: int, child_id: int) -> None:
        df = self.sim.population.props
        df.at[child_id, 'hl_hiv_status'] = 'Non-Reactive'
        df.at[child_id, 'hl_date_inf'] = pd.NaT

class HivInfectionEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=pd.DateOffset(months=1))
    def apply(self, population):
        df = population.props

        selected_for_hiv_inf = df.loc[df.is_alive & (df.hl_hiv_status == 'Non-Reactive')]
        random_selection = self.module.rng.choice([True, False],
                                                  size=len(selected_for_hiv_inf),
                                                  p=[self.module.parameters['inf_rate'], 1 - self.module.parameters['inf_rate']])
        # get index of those who are to be infected
        inf_index = selected_for_hiv_inf.index[random_selection]
        # update their hiv status to Reactive and assign a date
        df.loc[inf_index, 'hl_hiv_status'] = 'Reactive'
        df.loc[inf_index, 'hl_date_inf'] = self.module.sim.date

        print(df)



