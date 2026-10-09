"""
Committed once at src/scripts/smac_optimisation/smac_scenario_baseline.py

Used ONLY for the one-off baseline run (see submit_baseline_job() in
optimisation_pipeline.py) - NEVER for real trials (smac_scenario.py) or
checkpoint generation (smac_scenario_suspend.py). Runs a single, fixed,
no-scale-up scenario ('type_of_scaleup': 'none') as BASELINE_RUNS_PER_DRAW
FULL, differently-seeded, COMPLETE runs (below) - NOT suspended/resumed
at any point. end_date IS genuinely load-bearing here (unlike
smac_scenario_suspend.py's own end_date, which the remote command's
--suspend-date interrupts before it's ever reached) - the whole point
of this scenario is to run status-quo behaviour all the way through to
the end of YEAR_END_DATE's own calendar year, so its results are
directly comparable to what a real trial's own full run would look
like, for deriving the budget constraints those real trials get
checked against (see postprocess_output.compute_and_save_baseline_budgets()).
end_date is set one year PAST YEAR_END_DATE (not AT it) specifically so
the simulation actually runs through YEAR_END_DATE's own full calendar
year rather than stopping at its very first day - matches
smac_scenario.py's own end_date for the identical reason.

Since NO config parameters vary here (this scenario always represents
a single, shared, config-independent "status quo" comparison point),
there's no configspace-driven setattr loop needed the way
smac_scenario.py's real-trial class has one - draw_parameters() always
returns the same single, hardcoded parameter set.

CLASS NAME is literally identical to smac_scenario_suspend.py's own
TloCheckpointScenario (a leftover of this file originally being cloned
from that one) - NOT a problem in practice, since optimisation_pipeline.py
imports this one under an alias (TloBaselineScenario) specifically to
avoid the collision - but worth knowing if this file is ever read in
isolation.

pop_size/start_date are set from the SAME shared constants
smac_scenario.py/smac_scenario_suspend.py use (optimisation_parameters.py) -
not because a resumed trial's technical correctness depends on it (this
scenario is never checkpointed or resumed at all, unlike
smac_scenario_suspend.py's own pop_size/start_date, which genuinely
does carry that hard requirement) but so the baseline's own results stay
genuinely comparable to what a real trial's own full run represents,
for the budget-deriving purpose above.
"""

from pathlib import Path
from typing import Dict
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))  # ensures sibling
    # modules in this same directory (optimisation_parameters.py, and
    # anything else this file imports directly) are importable regardless
    # of the working directory the process was actually launched from.
    # Needed specifically because TLO's own `tlo scenario-run`/
    # `tlo batch-run` load this file dynamically, by path, from wherever
    # they're invoked (typically the repo root) - unlike running this
    # file directly (python's own automatic sys.path[0] behaviour, which
    # is why optimisation_pipeline.py's own sibling imports never hit
    # this problem), that dynamic loading does NOT automatically add this
    # file's own directory to sys.path.
from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario
from optimisation_parameters import YEAR_START_DATE, YEAR_END_DATE, POP_SIZE

BASELINE_RUNS_PER_DRAW = 10  # exported so callers outside this file (e.g.
    # postprocess_output.compute_and_save_baseline_budgets()'s own
    # sanity check that it actually received this many runs) can
    # reference the SAME value, rather than duplicating the literal 10
    # in a second place where it could silently drift out of sync with
    # this class's own runs_per_draw below.

class TloCheckpointScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0  # NEVER overwritten - unlike smac_scenario.py/
                       # smac_scenario_suspend.py's own placeholders,
                       # submit_baseline_job() never sets this. Stays 0,
                       # genuinely used: low_bias_32(0 + sample_number)
                       # for sample_number=0..9 (runs_per_draw=10, below)
                       # gives the 10 runs 10 genuinely different actual
                       # seeds - matching each other isn't important
                       # here, only that they genuinely differ.
        self.start_date = Date(YEAR_START_DATE, 1, 1)  # from the SAME shared
                                                          # constant as smac_scenario.py -
                                                          # see module docstring for why
        self.end_date = Date(YEAR_END_DATE+1, 1, 1)  # GENUINELY load-bearing here -
                                                      # unlike smac_scenario_suspend.py's
                                                      # own end_date, this scenario is
                                                      # NEVER suspended, so this IS where
                                                      # the simulation actually stops.
                                                      # +1 so YEAR_END_DATE's own full
                                                      # calendar year is actually
                                                      # simulated, not just its first day.
        self.pop_size = POP_SIZE  # from the SAME shared constant - see module docstring
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)  # always 1 - single, fixed scenario
        self.runs_per_draw = BASELINE_RUNS_PER_DRAW

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
                }
            }
        }

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
