"""This Scenario file generates individual histories for the purpose of generating datasets for emulator training

Still to fix:
- In principle might want to vary parameters that are stored as arrays, issue is that they are not JSON serialisable.
- For now ignoring all 'universal' parameters

"""

from pathlib import Path
from typing import Dict

import pandas as pd

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios, get_filtered_treatment_ids
from tlo.methods import individual_history_tracker, cervical_cancer, demography, enhanced_lifestyle, symptommanager, healthsystem, healthburden, healthseekingbehaviour, epi, hiv, tb, simplified_births
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario
import re
import ast
import numpy as np
from scipy.stats import qmc

module_of_interest ='CervicalCancer'
N_param_combo = 512  # total number of draws across all batches; must be a power of 2 (Sobol design)
# Draws per submission; must be a power of 2 that divides N_param_combo. Each aligned batch of asim Sobol design is
# itself evenly spread over the parameter space. Do not change once batches have been submitted.
BATCH_SIZE = 32
assert N_param_combo % BATCH_SIZE == 0 and (BATCH_SIZE & (BATCH_SIZE - 1)) == 0
# Parameters (by name) to exclude from parameter variation; these keep their resource-file value
parameters_to_ignore = []

def detect_and_convert(value):
    # try float
    try:
        return float(value)
    except:
        pass
    
    # try literal list (e.g. "[1,2,3]")
    try:
        return ast.literal_eval(value)
    except:
        pass
    
    # try semicolon list (e.g. "1;2;3")
    if ";" in value:
        try:
            return [float(x) for x in value.split(";")]
        except:
            pass
    
    # fallback: return as string
    return value

def to_array(x):
    if isinstance(x, str):
        x = ast.literal_eval(x)
    return np.asarray(x, dtype=float)

def parse_value(value):
    # If already not string, return as-is (e.g. native float)
    if not isinstance(value, str):
        return value
    
    v = value.strip()
    
    # 1. FLOAT DETECTION
    try:
        return float(v)
    except:
        pass
    
    # 2. CLEAN-UP STEP FOR LIST-LIKE VALUES
    # detect bracketed patterns: [ ... ]
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1]
        
        # sloppy sanitize:
        # - collapse whitespace
        # - ensure commas are normalized
        inner = re.sub(r"\s+", " ", inner)         # collapse multi-spaces
        inner = re.sub(r"\s*,\s*", ",", inner)     # normalize comma spacing
        
        # try literal_eval first (after cleanup)
        try:
            out = ast.literal_eval(f"[{inner}]")
            if isinstance(out, list):
                # ensure floats inside
                return [float(x) for x in out]
        except:
            pass
        
        # fallback manual parse (sloppy)
        try:
            parts = inner.split(",")
            return [float(p) for p in parts if p.strip() != ""]
        except:
            pass
    
    # 3. FALLBACK TO STRING
    return value


def sample_param_combo(seed):

        # For each N_param_combo, which will constitute a draw, the plan is to create a param combination, that will include "service_availability" to be passed to the HealthSystem and param rescaling to be passed to the module
        # keys: draw number, value = dictionary where key: parameter name, value: new parameter value
        parameter_draws = {}
        
        # First collect all module-specific treatments
        #treatments = get_filtered_treatment_ids(depth=2)
        #treatments_in_module = [item for item in treatments if module_of_interest in item]
            
        # Retreive module parameters which are not scenario or design decisions
        # REVIEW: is this best way to access module parameters?
        p = pd.read_csv('resources/ResourceFile_Cervical_Cancer/parameter_values.csv')
        
        # Select undetermined parameters which were not design decisions
        p = p.loc[p['param_label']=='undetermined']
        p = p.drop(p[p['prior_note'] == 'design decision'].index)
        p = p.reset_index(drop=True)
        # Parse values
        p['value'] = p['value'].apply(parse_value)

        # Sobol design: one row per draw, one column per candidate parameter, values in [0, 1).
        # The design is built over ALL candidate parameters (before applying parameters_to_ignore), so that
        # ignoring a parameter does not change the values sampled for the others.
        # The first 2^k rows of a larger design equal the 2^k-row design, so draws can be extended later.
        n_draws_log2 = int(np.log2(N_param_combo))
        assert 2 ** n_draws_log2 == N_param_combo, "N_param_combo must be a power of 2 for Sobol sampling"
        sobol_design = qmc.Sobol(d=len(p), scramble=True, seed=seed).random_base2(m=n_draws_log2)

        # Exclude parameters explicitly listed to be ignored (their Sobol columns are simply unused)
        p = p.loc[~p['parameter_name'].isin(parameters_to_ignore)]

        # For all param combos/draws, create a dictionary of parameter combinations
        for draw in range(N_param_combo):
            parameter_draws[draw] = {}

            # 1. Select services included with 50/50 probability
            # Create a service availability scenario for this draw. Module treatments are included with a 50/50 probability
            # NOTE: if re-enabled, add extra columns to the Sobol design for this rather than using a separate RNG
            # selected_treatments = [x for x in treatments_in_module if <uniform draw> < 0.5]
            # parameter_draws[draw]['HealthSystem'] = {'Service_Availability': selected_treatments}

            # 2. Resample parameters
            parameter_draws[draw].setdefault(module_of_interest, {})
            for idx, row in p.iterrows():
                # idx is the parameter's column in the Sobol design
                r = sobol_design[draw, idx]

                if isinstance(row['value'], (pd.Series, list, tuple, np.ndarray)):
                    prior_max = to_array(row['prior_max'])
                    prior_min = to_array(row['prior_min'])
                    new_val = prior_min + r*(prior_max - prior_min)
                    new_val = new_val.tolist()
                else:
                    prior_max = float(row['prior_max'])
                    prior_min = float(row['prior_min'])
                    new_val = prior_min + r*(prior_max - prior_min)
                
                parameter_draws[draw][module_of_interest][row['parameter_name']] = new_val

            # 3. Resample consumable availability
            # EXCLUDE FOR NOW

        return parameter_draws


class TrackIndividualHistories(BaseScenario):
    """Submitted in batches: the full Sobol design has N_param_combo draws (global draw numbers
    0 .. N_param_combo-1); each submission runs one batch of BATCH_SIZE consecutive global draws, chosen with
    the command-line argument --batch-number (default 0), e.g.

        tlo batch-submit <this_file> --batch-number 3

    Within a submission TLO numbers draws 0 .. BATCH_SIZE-1 (local draw number); the global draw number is
        global_draw = batch_number * BATCH_SIZE + local_draw
    and is recorded as the draw name (in the draws JSON and in each run's log under 'draw_name').
    Running all batches gives exactly the same parameters and simulation seeds as one 512-draw submission.
    """
    def __init__(self):
        super().__init__()
        # Top-level seed: controls the Sobol design and (via the `seed` property) the simulation seeds
        self.base_seed = 42
        self.batch_size = BATCH_SIZE
        self.batch_number = 0  # overridden by --batch-number
        # Give each draw its own simulation seed (independent population per draw).
        # Requires the opt-in change to SampleRunner.get_sample in src/tlo/scenario.py; ignored otherwise.
        self.unique_seed_per_draw = True
        self.start_date = Date(2010, 1, 1)
        self.end_date = self.start_date + pd.DateOffset(years=1) #20)
        self.pop_size = 500 #6000
        self.number_of_draws = self.batch_size
        self.runs_per_draw = 1
        self.generate_event_chains = True
        self._parameter_draws = None  # full Sobol design, computed lazily (only when draws are generated)

    @property
    def seed(self):
        # Offset the seed by the number of runs in earlier batches. With unique_seed_per_draw, TLO then gives
        # run k of local draw d the seed  hash(base_seed + global_draw * runs_per_draw + k),  i.e. the same seed it
        # would get if all draws were submitted together. Unique across all batches.
        if self.base_seed is None:
            return None
        return self.base_seed + self.batch_number * self.batch_size * self.runs_per_draw

    @seed.setter
    def seed(self, value):
        # BaseScenario.__init__ assigns self.seed; route it to base_seed
        self.base_seed = value

    def add_arguments(self, parser):
        parser.add_argument('--batch-number', type=int, help='Which batch of draws to submit (0-based)')

    def global_draw_number(self, draw_number):
        n_batches = N_param_combo // self.batch_size
        assert 0 <= self.batch_number < n_batches, \
            f"batch_number must be between 0 and {n_batches - 1} (N_param_combo={N_param_combo}, BATCH_SIZE={self.batch_size})"
        return self.batch_number * self.batch_size + draw_number

    def draw_name(self, draw_number):
        return f"global_draw_{self.global_draw_number(draw_number)}"

    def log_configuration(self):
        return {
            'filename': f'track_individual_histories_batch{self.batch_number}',  # batch number appears in job ID
            'directory': Path('./outputs'),  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.events': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
                'tlo.methods.individual_history_tracker': logging.INFO
            }
        }

    def modules(self):
        return (
            #fullmodel() + [individual_history_tracker.IndividualHistoryTracker()]
            [demography.Demography(),
            cervical_cancer.CervicalCancer(),
            enhanced_lifestyle.Lifestyle(),
            healthburden.HealthBurden(),
            healthseekingbehaviour.HealthSeekingBehaviour(),
            symptommanager.SymptomManager(),
            # HealthSystem and the Expanded Programme on Immunizations
            epi.Epi(),
            healthsystem.HealthSystem(),
            hiv.DummyHivModule(),
            #simplified_births.SimplifiedBirths(),
            individual_history_tracker.IndividualHistoryTracker(),
            ]
            )

    def draw_parameters(self, draw_number, rng):
        if draw_number >= self.number_of_draws:
            return
        if self._parameter_draws is None:
            # Sobol design over ALL N_param_combo global draws, seeded with base_seed (same for every batch)
            self._parameter_draws = sample_param_combo(self.base_seed)
        i = self.global_draw_number(draw_number)
        return mix_scenarios(
            self._baseline(),
            {
                #'HealthSystem': self._parameter_draws[i]['HealthSystem'],
                module_of_interest: self._parameter_draws[i][module_of_interest]
            }
        )


    def _baseline(self) -> Dict:
        #Return the Dict with values for the parameter changes that define the baseline scenario.
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {
                "HealthSystem": {
                    "mode_appt_constraints": 1,
                },
                "IndividualHistoryTracker": {
                    "generate_emulator_data": True,
                },
                 "CervicalCancer": {
                    "generate_emulator_data": True,
                }
            },
        )

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
