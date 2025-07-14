from pathlib import Path

import pandas as pd

from tlo import Parameter, Types, Property, Module, Population, Simulation
from tlo.events import PopulationScopeEventMixin, RegularEvent, IndividualScopeEventMixin
from tlo.methods import Metadata


class HypertensionTz(Module):
    """
        Hypertension Module looking at cost and effectiveness of lifestyle interventions
        for hypertension in Tanzania
    """
    def __init__(self, resourcefilepath=None):
        super().__init__()
        self.health_care_cost = {}

    METADATA = {Metadata.DISEASE_MODULE}

    PARAMETERS = {
        'prob_hyp': Parameter(Types.REAL, 'probability of having hypertension '),
        'prob_hyp_classification' : Parameter(Types.LIST, 'probabilities of hypertension classification'),
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
        df = pd.read_csv(data_folder / 'Resourcefile_hypertension.csv')
        self.load_parameters_from_dataframe(df)


    def initialise_population(self, population: Population) -> None:
        df = population.props
        df.loc[df.is_alive, 'hyp_classification'] = pd.NA
        df.loc[df.is_alive, 'hyp_severity'] = pd.NA

    def initialise_simulation(self, sim: Simulation) -> None:
        sim.schedule_event(HypertensionEVent(self), sim.date)

    def on_birth(self, mother_id: int, child_id: int) -> None:
        """ child hypetension properties """
        df = self.sim.population.props
        df.at[child_id, 'hyp_classification'] = pd.NA
        df.at[child_id, 'hyp_severity'] = pd.NA

    def on_simulation_end(self) -> None:
        df = self.sim.population.props
        cost_df = pd.DataFrame(data={'healthcare_cost(TSH)': self.health_care_cost.values()}, index=self.health_care_cost.keys())
        print(f'the population dataframe is {df.loc[df.is_alive, ["district_of_residence", "hyp_classification", "hyp_severity"]]}')
        print(f'the healthcare cost {cost_df}')

    def healthcare_event(self, person_id):
        # provide treatment and update healthcare cost
        df = self.sim.population.props
        df.loc[person_id, 'hyp_severity'] = 'mild'
        self.health_care_cost.update({self.sim.date: self.rng.randint(1000, 3000)})


class HypertensionEVent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module) -> None:
        super().__init__(module, frequency=pd.DateOffset(months=1))

    def apply(self, population):
        df = population.props
        self.param = self.module.parameters

        # select population, alive and aged 15+
        pop_over_15 = df.loc[df.is_alive & (df.age_years > 15) & pd.isna(df.hyp_classification)]
        # select those to have hypertension
        hypert_pop = self.module.rng.choice([True, False],
                                            size=len(pop_over_15),
                                            p=[self.param['prob_hyp'], 1-self.param['prob_hyp']]
                                            )
        idx_hyp_pop = pop_over_15.index[hypert_pop]
        # assign hypertension clasification
        df.loc[idx_hyp_pop, 'hyp_classification'] = self.module.rng.choice(['primary', 'secondary'],
                                            size=len(idx_hyp_pop),
                                            p=self.param['prob_hyp_classification']
                                            )
        # assign hypertension severity
        df.loc[idx_hyp_pop, 'hyp_severity'] = self.module.rng.choice(['mild', 'moderate', 'severe'],
                                                                           size=len(idx_hyp_pop),
                                                                           p=self.param['prob_hyp_severity']
                                                                           )
        # offer treatment to severe and update health care cost
        for idx in idx_hyp_pop:
            if df.loc[idx, 'hyp_severity'] == 'severe':
                self.module.healthcare_event(person_id=idx)
