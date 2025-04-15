# include comments on what this module is all about here
from pathlib import Path

import pandas as pd

from tlo import Module, Types, Parameter, Property, Population, Simulation, logging
from tlo.events import RegularEvent, PopulationScopeEventMixin
from tlo.methods import Metadata

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
        'art_coverage': Parameter(Types.REAL, description='ART coverage'),
    }
    PROPERTIES = {
        'hl_hiv_status': Property(Types.STRING, description='hiv status'),
        'hl_on_art': Property(Types.BOOL, description='whether on art or not'),
        'hl_date_inf': Property(Types.DATE, description='date of infection'),
        'hl_date_prog_aids': Property(Types.DATE, description='date of progression to aids')
    }

    def read_parameters(self, data_folder: str | Path) -> None:
        param = self.parameters
        param['inf_rate'] = 0.05
        param['aids_prog_rate'] = 0.5
        param['art_coverage'] = 0.02


    def initialise_population(self, population: Population) -> None:
        df = population.props
        df.loc[df.is_alive, 'hl_hiv_status'] = 'Non-Reactive'
        df.loc[df.is_alive, 'hl_on_art'] = False
        df.loc[df.is_alive, 'hl_date_inf'] = pd.NaT
        df.loc[df.is_alive, 'hl_date_prog_aids'] = pd.NaT

    def initialise_simulation(self, sim: Simulation) -> None:
        sim.schedule_event(HivInfectionEvent(self), sim.date + pd.DateOffset(months=1))
        sim.schedule_event(AidsProgressEvent(self), sim.date + pd.DateOffset(months=1))
        sim.schedule_event(HivLiteloogingevent(self), sim.date + pd.DateOffset(months=1))

    def on_birth(self, mother_id: int, child_id: int) -> None:
        df = self.sim.population.props
        df.at[child_id, 'hl_hiv_status'] = 'Non-Reactive'
        df.at[child_id, 'hl_on_art'] = False
        df.at[child_id, 'hl_date_inf'] = pd.NaT
        df.at[child_id, 'hl_date_prog_aids'] = pd.NaT

class HivInfectionEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=pd.DateOffset(months=1))
        self.params = self.module.parameters
    def apply(self, population):
        df = population.props

        selected_for_hiv_inf = df.loc[df.is_alive & (df.hl_hiv_status == 'Non-Reactive')]
        random_selection = self.module.rng.choice([True, False],
                                                  size=len(selected_for_hiv_inf),
                                                  p=[self.params['inf_rate'], 1 - self.params['inf_rate']])
        # get index of those who are to be infected
        inf_index = selected_for_hiv_inf.index[random_selection]
        # update their hiv status to Reactive and assign a date
        df.loc[inf_index, 'hl_hiv_status'] = 'Reactive'
        df.loc[inf_index, 'hl_date_inf'] = self.module.sim.date

        # start ART
        select_for_art = df.loc[df.is_alive & (df.hl_hiv_status == 'Reactive') & ~df.hl_on_art]
        start_art = self.module.rng.random_sample(size=len(select_for_art)) < self.params['art_coverage']
        prog_index = select_for_art.index[start_art]
        df.loc[prog_index, 'hl_on_art'] = True

class AidsProgressEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=pd.DateOffset(months=1))
        self.params = module.parameters
    def apply(self, population):
        def selected_for_progression(df, progression_rate):
            random_selection = self.module.rng.choice([True, False],
                                                      size=len(df),
                                                      p=[progression_rate, 1 - progression_rate])
            return df.index[random_selection]

        df = population.props
        selected_for_progression_not_art = df.loc[df.is_alive &
                                          (df.hl_hiv_status == 'Reactive') &
                                          (df.hl_date_inf + pd.DateOffset(months=3) < self.sim.date) &
                                          ~df.hl_on_art]

        selected_for_progression_art = df.loc[df.is_alive &
                                              (df.hl_hiv_status == 'Reactive') &
                                              (df.hl_date_inf + pd.DateOffset(months=3) < self.sim.date) &
                                              df.hl_on_art]

        # get index of those who are to be infected
        prog_index_not_art = selected_for_progression(selected_for_progression_not_art,
                                                      self.module.parameters['aids_prog_rate'])
        prog_index_art = selected_for_progression(selected_for_progression_art,
                                                  self.params['aids_prog_rate'] * (1 - self.params['art_coverage']))

        all_index = prog_index_not_art.append(prog_index_art)
        if len(all_index) != 0:
            # update their hiv status to Reactive and assign a date
            df.loc[all_index, 'hl_hiv_status'] = 'AIDS'
            df.loc[all_index, 'hl_date_prog_aids'] = self.module.sim.date



class HivLiteloogingevent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=pd.DateOffset(months=12))

    def apply(self, population):
        df = population.props
        alive_indv = df.loc[df.is_alive]

        indiv_0_14 = alive_indv.loc[df.age_years.between(0, 14) & (df.hl_hiv_status == 'AIDS')]
        indiv_15_24 = alive_indv.loc[df.age_years.between(15, 24) & (df.hl_hiv_status == 'AIDS')]
        indiv_25_34 = alive_indv.loc[df.age_years.between(25, 34) & (df.hl_hiv_status == 'AIDS')]
        indiv_35_44 = alive_indv.loc[df.age_years.between(35, 44) & (df.hl_hiv_status == 'AIDS')]
        indiv_45_54 = alive_indv.loc[df.age_years.between(45, 54) & (df.hl_hiv_status == 'AIDS')]
        indiv_55_64 = alive_indv.loc[df.age_years.between(55, 64) & (df.hl_hiv_status == 'AIDS')]
        indiv_65 = alive_indv.loc[(df.age_years >= 65) & (df.hl_hiv_status == 'AIDS')]

        hiv_infections_cases_by_age_group = {
            'indiv_0_14': len(indiv_0_14),
            'indiv_15_24': len(indiv_15_24),
            'indiv_25_34': len(indiv_25_34),
            'indiv_35_44': len(indiv_35_44),
            'indiv_45_54': len(indiv_45_54),
            'indiv_55_64': len(indiv_55_64),
            'indiv_65': len(indiv_65),
        }

        logger.info(key='aids_cases', data=hiv_infections_cases_by_age_group, description='AIDS cases')

