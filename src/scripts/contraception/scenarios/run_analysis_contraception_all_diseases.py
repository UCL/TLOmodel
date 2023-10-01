"""
This file defines a batch run of the full model to get sims results to be used by the analysis_contraception_plot_table.
* Consumables Availability: All
* Use Interventions: to be set as True or False

Run on the remote batch system using:
```tlo batch-submit src/scripts/contraception/scenarios/run_analysis_contraception_all_diseases.py```
or locally using:
```tlo scenario-run src/scripts/contraception/scenarios/run_analysis_contraception_all_diseases.py```
"""

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class RunAnalysisCo(BaseScenario):
    def __init__(self):
        super().__init__(
            seed=0,
            start_date=Date(2010, 1, 1),
            end_date=Date(2050, 12, 31),
            initial_population_size=1_000,
            number_of_draws=1,  # <- one scenario
            runs_per_draw=1,  # <- repeated this many times
        )

    def log_configuration(self):
        return {
            'filename': 'run_analysis_contraception_all_diseases',  # <- (specified only for local running)
            'directory': './outputs/run_on_laptop',  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.contraception": logging.DEBUG,
                "tlo.methods.demography": logging.INFO
            }
        }

    def modules(self):
        return fullmodel(
            resourcefilepath=self.resources,
            module_kwargs={
                "Healthsystem": {"cons_availability": "all"},
                "Contraception": {"use_interventions": True}
            }
        )

    def draw_parameters(self, draw_number, rng):
        return  # Using default parameters in all cases


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
