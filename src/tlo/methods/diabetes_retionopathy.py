from pathlib import Path

import pandas as pd

from tlo import Module, Simulation, Parameter, Types, Property, Population
from tlo.events import RegularEvent, PopulationScopeEventMixin
from tlo.methods import Metadata


class Diabetes_Retinopathy(Module):
    """ This is Diabetes Retinopathy module. It seeks to skeleton of blindness due to diabetes. """

    INIT_DEPENDENCIES = {'SymptomManager', 'Lifestyle', 'HealthSystem', 'CardioMetabolicDisorders'}
    ADDITIONAL_DEPENDENCIES = set()

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN,
    }

    # define a dictionary of parameters this module will use
    PARAMETERS = {
        'rate_onset_to_early_dr': Parameter(Types.REAL, 'Probability to dr'),
        'rate_progression_to_dr': Parameter(Types.REAL, 'Probability to dr'),
        'prob_fast_dr': Parameter(Types.REAL, 'Probability to dr'),
        'init_prob_any_dr': Parameter(Types.REAL, 'Probability to dr'),
        'init_prob_late_dr': Parameter(Types.REAL, 'Probability to dr'),
    }

    # define a dictionary of properties this module will use
    PROPERTIES = {
        "dr_status": Property(
            Types.CATEGORICAL,
            categories=[
                "none",
                "early",
                "late",
            ],
            description="dr status",
        ),
    }

    def __init__(self):
        # this method is included to define all things that should be initialised first
        super().__init__()


    def read_parameters(self, data_folder: str | Path) -> None:
        """ initialise module parameters. Here we are assigning values to all parameters defined at the beginning of
        this module. For this demo module, we will manually assign values to parameters but in the
        Thanzi model we do this by reading from an Excel file containing parameter names and values

        :param data_folder: Path to the folder containing parameter values

        """
        self.parameters['rate_onset_to_early_dr'] = 0.5
        self.parameters['rate_progression_to_dr'] = 0.5
        self.parameters['prob_fast_dr'] = 0.5
        self.parameters['init_prob_any_dr'] = 0.5
        self.parameters['init_prob_late_dr'] = 0.5

    def initialise_population(self, population: Population) -> None:
        """ set the initial state of the population. The state will be update over time

        :param population: all individuals in the model

        """
        df = population.props
        p = self.parameters


        alive_diabetes_idx = df.loc[df.is_alive & df.nc_diabetes].index

        any_dr_idx = alive_diabetes_idx[
            self.rng.random_sample(size=len(alive_diabetes_idx)) < self.parameters['init_prob_any_dr']
        ]
        no_dr_idx = set(alive_diabetes_idx) - set(any_dr_idx)

        late_dr_idx = any_dr_idx[
            self.rng.random_sample(size=len(any_dr_idx)) < self.parameters['init_prob_late_dr']
        ]

        early_dr_idx = set(any_dr_idx) - set(late_dr_idx)

        # write to property:
        df.loc[df.is_alive & ~df.nc_diabetes, 'dr_status'] = 'none'
        df.loc[list(no_dr_idx), 'dr_status'] = 'none'
        df.loc[list(early_dr_idx), "dr_status"] = "early"
        df.loc[list(late_dr_idx), "dr_status"] = "late"





    def initialise_simulation(self, sim: Simulation) -> None:
        """ This is where you should include all things you want to be happening during simulation

        :param sim: simulation object

        """
        # schedule an event to infect people with tb
        sim.schedule_event(DrPollEvent(self), date=sim.date + pd.DateOffset(years=3))
        # sim.schedule_event(TblInfectionEvent(self), date=sim.date + pd.DateOffset(months=1))
        # schedule an event to cure people from tb
        sim.schedule_event(TblCureEvent(self), date=sim.date + pd.DateOffset(months=2))

    def on_birth(self, mother_id: int, child_id: int) -> None:
        """ set properties of a child when they are born. """
        pass

    def on_simulation_end(self) -> None:
        tb_inc_and_prev = pd.DataFrame(index=list(self.incidence_tb.keys()),
                                       data={'incidence': self.incidence_tb.values(),
                                             'prevalence': self.prevalence_tb.values(),
                                             'total_pop': len(self.sim.population.props)})

        print(f'\n\nrunning the model with infection probability at {self.parameters["p_infection"]}')
        print(tb_inc_and_prev)


class DrPollEvent(RegularEvent, PopulationScopeEventMixin):
    """An event that controls the development process of Diabetes Retionpathy (DR) and logs current states. DR diagnosis
    begins at least after 3 years of being infected with Diabetes Mellitus."""

    def __init__(self, module):
        super().__init__(module, frequency=pd.DateOffset(months=3))

class TblInfectionEvent(RegularEvent, PopulationScopeEventMixin):
    """ cause individuals to be infected by Tb. This event will run every one month """

    def __init__(self, module: Module) -> None:
        self.repeat = 1
        super().__init__(module, frequency=pd.DateOffset(months=self.repeat))

    def apply(self, population: Population) -> None:
        """ actions that should be applied to the population when this event is triggered """
        df = population.props

        # select individuals to infect. should be those without tb at the present time
        individuals_to_infect = ~df['tbl_is_infected']
        random_selection = self.module.rng.choice([True, False], size=len(individuals_to_infect),
                                                  p=[self.module.parameters['p_infection'],
                                                     1 - self.module.parameters['p_infection']])

        # update the properties of individuals that have been selected for tb infection
        idx_individuals_to_infect = individuals_to_infect.index[random_selection]
        df.loc[idx_individuals_to_infect, 'tbl_is_infected'] = True
        df.loc[idx_individuals_to_infect, 'tbl_date_infected'] = self.sim.date

        # incidence of Tb
        self.module.incidence_tb.update({self.sim.date: len(idx_individuals_to_infect)})
        self.module.prevalence_tb.update({self.sim.date: df.tbl_is_infected.sum()})


class TblCureEvent(RegularEvent, PopulationScopeEventMixin):
    """ cause individuals to recover from Tb. This event will run every one month """

    def __init__(self, module: Module) -> None:
        self.repeat = 1
        super().__init__(module, frequency=pd.DateOffset(months=self.repeat))
        self.module = module

    def apply(self, population: Population) -> None:
        """ actions that should be applied to the population when this event is triggered """
        df = population.props

        # select individuals to recover. should be those with tb and infected not less than a month ago
        individuals_to_recover = df.loc[df.tbl_is_infected &
                                        (self.sim.date - pd.DateOffset(months=1) > df.tbl_date_infected)]

        random_selection = self.module.rng.choice([True, False], size=len(individuals_to_recover),
                                                  p=[self.module.parameters['p_cure'],
                                                     1 - self.module.parameters['p_cure']])

        # update the properties of individuals that have been selected for tb cure.
        # they should be well and have a date of recovery
        idx_individuals_to_recover = individuals_to_recover.index[random_selection]
        df.loc[idx_individuals_to_recover, 'tbl_is_infected'] = False
        df.loc[idx_individuals_to_recover, 'tbl_date_cure'] = self.sim.date
