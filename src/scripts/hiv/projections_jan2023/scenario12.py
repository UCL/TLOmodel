"""
This file defines a batch run through which the hiv and tb modules are run across a grid of parameter values
check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/hiv/PaperProjectionsNov2022/scenario12.py
Run on the batch system using:
tlo batch-submit src/scripts/hiv/projections_jan2023/scenario12.py
Display information about a job:
tlo batch-job tlo_q1_demo-123 --tasks
Download result files for a completed job:
tlo batch-download scenario1-2022-04-20T112503Z
9th June
Job ID: scenario0-2022-06-09T170155Z
"""

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class TestScenario(BaseScenario):
    # this imports the resource filepath automatically

    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2034, 1, 1)
        self.pop_size = 100_000
        self.number_of_draws = 5
        self.runs_per_draw = 2

    def log_configuration(self):
        return {
            "filename": "scenario12",
            "directory": "./outputs",
            "custom_levels": {
                "*": logging.WARNING,
                "tlo.methods.hiv": logging.INFO,
                "tlo.methods.tb": logging.INFO,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
            },
        }

    def modules(self):
        return *fullmodel(
            resourcefilepath=self.resources,
            use_simplified_births=False,
            module_kwargs={
                "SymptomManager": {"spurious_symptoms": True},
                "HealthSystem": {"disable": False,
                                 "service_availability": ["*"],
                                 "mode_appt_constraints": 2,  # changed
                                 "policy_name": "HivTbProgrammes",  # changed
                                 "cons_availability": "default",
                                 "beds_availability": "all",
                                 "ignore_priority": False,
                                 "use_funded_or_actual_staffing": "funded",  # changed
                                 "capabilities_coefficient": None},  # changed
            },
        ),

    def draw_parameters(self, draw_number, rng):
        return {
            'Tb': {
                'scenario': 2,
                'scaling_factor_WHO': [1.603029372, 1.507636969, 1.673313118, 1.739471024, 1.702850618][draw_number]

            },
            'Hiv': {
                'beta': [0.143785163, 0.137641638, 0.130761192, 0.141134768, 0.11937769][draw_number]
            },
        }


if __name__ == "__main__":
    from tlo.cli import scenario_run

    scenario_run([__file__])
