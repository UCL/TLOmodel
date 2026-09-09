"""
Committed once at src/scripts/smac_optimisation/smac_scenario_suspend.py

Used ONLY for pre-resume checkpoint generation (see
generate_checkpoint_job() in optimisation_pipeline.py) - NEVER for real
trials, which use smac_scenario.py's TloOptimisationScenario instead.

Runs the model with a single, fixed, no-scale-up scenario
('type_of_scaleup': 'none') up to SUSPEND_DATE (see
optimisation_pipeline.py), then gets suspended via `--suspend-date` on
the remote task command. Since NO config parameters vary here (this
scenario always represents the shared, config-independent "first part"
of the simulation - see the whole suspend/resume design discussion),
there's no configspace-driven setattr loop needed the way
smac_scenario.py's real-trial class has one - draw_parameters() always
returns the same single, hardcoded parameter set.

CLASS NAME deliberately differs from smac_scenario.py's own
TloOptimisationScenario (that file's class is used for real trials -
this one is checkpoint-generation only) so both can be imported directly
into optimisation_pipeline.py without needing an import alias to avoid a
collision.

pop_size/start_date MUST match smac_scenario.py's own values exactly -
once a real trial resumes from a checkpoint generated here, the
simulation's population/state is already fixed from whatever was
pickled; a resumed trial cannot retroactively change the population
size or start date it was checkpointed under. If either file's
pop_size/start_date is ever edited, the other must be updated to match,
or resumed trials will silently be running a population inconsistent
with what a fresh, non-resumed run would have used.
"""

from pathlib import Path
from typing import Dict

#import sys
#sys.path.insert(0, str(Path(__file__).resolve().parent))  # ensures sibling
    # modules in this same directory (optimisation_parameters.py, and
    # anything else this file imports directly) are importable regardless
    # of the working directory the process was actually launched from.
    # Needed specifically because TLO's own `tlo scenario-run`/
    # `tlo batch-run` load this file dynamically, by path, from wherever
    # they're invoked (typically the repo root) - unlike running
    # optimisation_pipeline.py directly (python's own automatic
    # sys.path[0] behaviour, which is why that file's own sibling imports
    # never hit this problem), that dynamic loading does NOT automatically
    # add this file's own directory to sys.path.

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario
from optimisation_parameters import YEAR_START_DATE, YEAR_END_DATE, CONFIG_YEAR_START_DATE, POP_SIZE

class TloCheckpointScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0  # placeholder - overwritten with the target
                       # checkpoint seed by generate_checkpoint_job()
                       # before every real submission
        self.start_date = Date(2010, 1, 1)  # MUST match smac_scenario.py - see module docstring
        self.end_date = Date(2012, 1, 1)    # not functionally load-bearing (the remote
                                              # command's --suspend-date interrupts the run
                                              # well before this regardless), kept aligned
                                              # with the real scenario's own horizon for clarity
        self.pop_size = 1000  # MUST match smac_scenario.py - see module docstring
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)  # always 1 - single, fixed scenario
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'hiv_program_simplification',
            'directory': Path('./outputs'),
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem': logging.WARNING,
                'tlo.methods.healthsystem.summary': logging.INFO,
                'tlo.methods.hiv': logging.INFO,
                'tlo.methods.tb': logging.INFO,
            }
        }

    def modules(self):
        return (
            fullmodel(use_simplified_births=True,
                      module_kwargs={"HealthSystem": {"equip_availability": 'all'}})
        )

    def draw_parameters(self, draw_number, rng):
        if draw_number < len(self._scenarios):
            return list(self._scenarios.values())[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """
        Return the Dict with values for the parameters that are
        changed, keyed by a name for the scenario.
        """
        return {
            "Baseline": {
                "Hiv": {
                    "type_of_scaleup": "none",
                    "config_start_year": 2011,
                }
            }
        }

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
