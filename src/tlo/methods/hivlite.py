# include comments on what this module is all about here
from pathlib import Path

import pandas as pd
from pandas import DateOffset

from tlo import Module, Types, Parameter, Property, Population, Simulation
from tlo.events import RegularEvent, PopulationScopeEventMixin, IndividualScopeEventMixin
from tlo.methods import Metadata


class HivLite(Module):
    def __init__(self, resourcefilepath=None):
        super().__init__()

    # define metadata
    METADATA = {
        Metadata.DISEASE_MODULE
    }
    # CAUSES_OF_DEATH = ''
    #  define parameters that this module will use
    PARAMETERS = {
        'inf_rate': Parameter(Types.REAL, description='hiv infection rate'),
        'aids_prog_rate': Parameter(Types.REAL, description='aids progression rate'),
    }
    # define properties that this module will use
    PROPERTIES = {
        'hl_hiv_status': Property(Types.STRING, description='hiv status'),
        'hl_hiv_stage': Property(Types.CATEGORICAL, 'the current HIV stage',
                                 categories=['stage1', 'stage2', 'stage3']),
        'hl_date_inf': Property(Types.DATE, description='date of infection')
    }

    def read_parameters(self, data_folder: str | Path) -> None:
        param = self.parameters
        param['inf_rate'] = 0.2
        param['aids_prog_rate'] = 0.01

    def initialise_population(self, population: Population) -> None:
        df = population.props
        df.loc[df.is_alive, 'hl_hiv_status'] = 'Non-Reactive'
        df.loc[df.is_alive, 'hl_date_inf'] = pd.NaT
        df.loc[df.is_alive, 'hl_hiv_stage'] = pd.NA

    def initialise_simulation(self, sim: Simulation) -> None:
        sim.schedule_event(HivInfectionEvent(self), sim.date + pd.DateOffset(months=1))
        sim.schedule_event(ChangeHivStage(self), sim.date + pd.DateOffset(months=1))
        sim.schedule_event(HealthCareCosting(self), sim.date + pd.DateOffset(months=1))

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
                                                  p=[self.module.parameters['inf_rate'],
                                                     1 - self.module.parameters['inf_rate']])
        # get index of those who are to be infected
        inf_index = selected_for_hiv_inf.index[random_selection]
        # update their hiv status to Reactive and assign a date
        df.loc[inf_index, 'hl_hiv_status'] = 'Reactive'
        df.loc[inf_index, 'hl_date_inf'] = self.module.sim.date
        # assign hiv stage1 to all newly reactive individuals
        df.loc[inf_index, 'hl_hiv_stage'] = 'stage1'

class ChangeHivStage(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=pd.DateOffset(months=3))
    def apply(self, population):
        # Stage mappings
        stage_order = {'stage1': 1, 'stage2': 2, 'stage3': 3}
        # reverse the key value map
        reverse_stage_order = {v: k for k, v in stage_order.items()}
        df = population.props
        # get hiv stage
        hiv_3_mon = df.loc[
            ((self.sim.date - pd.DateOffset(months=3)) >= df['hl_date_inf']) &
            (df['hl_hiv_stage'] != 'stage3')
            ]

        assert (hiv_3_mon['hl_hiv_stage'] != 'stage3').all()
        # select some that are not on ART to progress to a higher hiv status
        to_progress = self.module.rng.choice([True, False], size=len(hiv_3_mon), p=[0.2, 0.8])
        to_progress_idx = hiv_3_mon.index[to_progress]
        # Increment their stage
        current_stage = df.loc[to_progress_idx, 'hl_hiv_stage']
        df.loc[to_progress_idx, 'hl_hiv_stage'] = (
            current_stage.map(lambda x: reverse_stage_order.get(stage_order[x] + 1))
        )

class HealthCareSeekingEvent(RegularEvent, IndividualScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=3))

     def apply(self, population):



class HealthCareCosting(RegularEvent, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id, frequency=pd.DateOffset(months=3))

    def apply(self, population):
        df = population.props
        # select all individuals in stage 3
        stg3_individuals = df.loc[df.is_alive & (df.hl_hiv_stage == 3)]
        print(stg3_individuals)

