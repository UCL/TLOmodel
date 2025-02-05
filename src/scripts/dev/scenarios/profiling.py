from pathlib import Path

from tlo import Date, logging
from tlo.analysis.performance import PerformanceMonitor
from tlo.methods import fullmodel
from tlo.scenario import BaseScenario

class Profiling(BaseScenario):
    """getting scale_run setup as a scenario"""
    def __init__(self):
        super().__init__()
        self.seed = 655123742
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2015, 1, 1)
        self.pop_size = 250000
        self.number_of_draws = 1
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            "directory": Path("."),
            "custom_levels": {"*": getattr(logging, "INFO"), "tlo.profiling": logging.INFO, "tlo.analysis.performance": logging.INFO},
            "suppress_stdout": False
        }

    def modules(self):
        fm = fullmodel.fullmodel(
            resourcefilepath=Path("./resources"),
            use_simplified_births=False,
            module_kwargs={
                "HealthSystem": {
                    "disable": False,
                    "mode_appt_constraints": 1,
                    "capabilities_coefficient": None,
                    "hsi_event_count_log_period": None
                },
                "SymptomManager": {"spurious_symptoms": True},
            }
        )
        fm.append(PerformanceMonitor())
        return fm

if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
