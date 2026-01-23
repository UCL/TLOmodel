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
import random
import re
import ast
import numpy as np

module_of_interest ='CervicalCancer'
N_param_combo = 2

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


def sample_param_combo():

        # For each N_param_combo, which will constitute a draw, the plan is to create a param combination, that will include "service_availability" to be passed to the HealthSystem and param rescaling to be passed to the module
        # keys: draw number, value = dictionary where key: parameter name, value: new parameter value
        parameter_draws = {}
        
        # First collect all module-specific treatments
        treatments = get_filtered_treatment_ids(depth=2)
        treatments_in_module = [item for item in treatments if module_of_interest in item]
            
        # Retreive module parameters which are not scenario or design decisions
        # REVIEW: is this best way to access module parameters?
        # REVIEW: Ideally unique criterion to filter these
        p = pd.read_csv('resources/ResourceFile_Cervical_Cancer/parameter_values.csv')
        # Drop scenario variables
        p = p.drop(p[p['param_label'] == 'scenario'].index)
        # Drop universal variables, already known
        p = p.drop(p[p['param_label'] == 'universal'].index)
        # Drop design decision
        p = p.drop(p[p['prior_note'] == 'design decision'].index)
        # Parse values
        p['value'] = p['value'].apply(parse_value)
        
        # For all param combos/draws, create a dictionary of parameter combinations
        for draw in range(N_param_combo):
            
            # Create a service availability scenario for this draw. Module treatments are included with a 50/50 probability
            # REVIEW: better handle on random seed here
            selected_treatments = [x for x in treatments_in_module if random.random() < 0.5]
            parameter_draws[draw] = {'HealthSystem': {'Service_Availability': selected_treatments}}

            parameter_draws[draw].setdefault(module_of_interest, {})
            for idx, row in p.iterrows():
                val = row['value']
                # REVIEW: Resampled value should be informed by prior!
                r = random.random()

                if isinstance(val, (pd.Series, list, tuple, np.ndarray)):
                    new_val = np.array(val, dtype=float) * r
                else:
                    new_val = val * r
                    parameter_draws[draw][module_of_interest][row['parameter_name']] = new_val

        return parameter_draws


class TrackIndividualHistories(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 42
        self.start_date = Date(2010, 1, 1)
        self.end_date = self.start_date + pd.DateOffset(years=12)
        self.pop_size = 100
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 1
        self.generate_event_chains = True

    def log_configuration(self):
        return {
            'filename': 'track_individual_histories',
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
            hiv.Hiv(),
            simplified_births.SimplifiedBirths(),
            tb.Tb(),
            individual_history_tracker.IndividualHistoryTracker(),
            ]
            )

    def draw_parameters(self, draw_number, rng):
        if draw_number < self.number_of_draws:
            return list(self._scenarios.values())[draw_number]
        else:
            return

    def _get_scenarios(self) -> Dict[str, Dict]:
        
        module_parameter_and_services_samples = sample_param_combo()
        scenarios = {}
        for i in range(N_param_combo):
            scenarios[str(i)] = mix_scenarios(
                                    self._baseline(),
                                    #{
                                    #    'HealthSystem': module_parameter_and_services_samples[i]['HealthSystem'],
                                    #    module_of_interest: module_parameter_and_services_samples[i][module_of_interest]
                                    #}
                                )

        return scenarios


    def _baseline(self) -> Dict:
        #Return the Dict with values for the parameter changes that define the baseline scenario.
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {
                "HealthSystem": {
                    "mode_appt_constraints": 1,                 # <-- Mode 1 prior to change to preserve calibration
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
