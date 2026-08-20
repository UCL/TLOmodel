"""
Committed once at src/scripts/smac_optimisation/smac_scenario.py

This is the ONE scenario file the ask-tell loop in
ask_tell_azure_example.py submits repeatedly - never regenerated or
re-committed per SMAC trial. Configuration values are set directly as
Python attributes on the instantiated object by the submitting code
(see submit_azure_job() in ask_tell_azure_example.py) - no CLI/argparse
round-trip, since that submitting code is itself Python, not a shell.

number_of_draws=1 because each submission represents exactly one
SMAC-proposed configuration. runs_per_draw=1 because each submission is
a single (config, seed) physical realisation - see SEEDING below for
why repeated evaluation of a config is handled by SMAC's intensifier
rather than by TLO's own multi-seed averaging.

SEEDING - leveraging SMAC's own intensification
--------------------------------------------------
scenario.seed is set directly from SMAC's info.seed at submission time
(see submit_azure_job() in ask_tell_azure_example.py) - NOT fixed here.
runs_per_draw is 1: each submission is a single (config, seed) physical
realisation, not a pre-averaged bundle.

This is deliberate: with the SMAC Scenario set to deterministic=False,
SMAC's own intensifier decides whether a given config's result is
promising enough to warrant re-evaluating with an additional seed
before trusting it as a challenger to the incumbent - compute is spent
proportionally to how competitive a config looks, rather than averaging
every proposed config over a fixed number of seeds regardless of merit.
Fixing runs_per_draw>1 here would double up on that job, paying a fixed
per-config cost SMAC's intensifier is already designed to avoid.

Because history now holds individual noisy realisations rather than
pre-averaged results, final config selection in
ask_tell_azure_example.py groups by config and averages across whatever
seeds SMAC ended up requesting for it, rather than trusting any single
realisation's DALYs value directly.

You can still sanity-check this scenario class works standalone via
`tlo scenario-run src/scripts/smac_optimisation/smac_scenario.py` using
its defaults below - useful for catching module/parameter wiring bugs
locally before submitting to batch - but that CLI path is now purely
optional and not what the SMAC loop itself uses.

HYPERPARAMETERS: every tunable knob in this file is marked inline with a
"HYPERPARAMETER" comment - grep for that tag across all three files
(constrained_ei.py, smac_scenario.py, ask_tell_azure_example.py) to find
the complete list in one pass.
"""

from tlo import Date, logging
from tlo.methods import demography, enhanced_lifestyle, healthburden, healthsystem
from tlo.scenario import BaseScenario


class SmacOptimisationScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0  # placeholder - overwritten with SMAC's info.seed
                       # by submit_azure_job() before every real submission
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2011, 1, 1)
        self.pop_size = 1_000  # HYPERPARAMETER: simulation fidelity, not a BO
                                   # hyperparameter, but trades off noise level
                                   # against per-run cost - indirectly relevant
                                   # to min_samples_leaf and max_config_calls in
                                   # the other two files, since a smaller
                                   # population means noisier DALYs/cost per seed
        self.number_of_draws = 1     # one draw == one SMAC configuration
        self.runs_per_draw = 1       # one seed per submission - SMAC's own
                                       # intensifier decides if/when this
                                       # config gets re-evaluated with a
                                       # different seed, not a fixed sweep

        # Defaults so `tlo scenario-run ...` works standalone for local
        # sanity-checking. Overwritten directly by submit_azure_job()
        # before every real submission.
        self.intervention_coverage = 0.5
        self.consumable_stock_target = 0.8

    def log_configuration(self):
        return {
            "filename": "smac_optimisation",
            "directory": "./outputs",
            "custom_levels": {"*": logging.INFO},
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources),
            # ... your real set of disease/intervention modules ...
        ]

    def draw_parameters(self, draw_number, rng):
        # draw_number is ignored - number_of_draws=1, so this always
        # returns the single SMAC config currently set on self.
        return {
            "HealthSystem": {
                "consumable_stock_target": self.consumable_stock_target,
            },
            # map intervention_coverage / other config values onto
            # whichever module parameters they actually control in
            # your real model
        }

