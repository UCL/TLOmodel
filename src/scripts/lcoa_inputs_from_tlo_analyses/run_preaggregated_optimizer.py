"""Generate optimizer CSV inputs and run the preaggregated R optimizer via Rscript."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from scripts.lcoa_inputs_from_tlo_analyses.results_processing_utils import format_scenario_name


OPTIMIZER_HR_COLS = ["hr_clin", "hr_nur", "hr_pharm", "hr_lab", "hr_ment", "hr_nutri"]
REQUIRED_OPT_INPUT_COLS = [
    "code",
    "category",
    "intervention",
    "ce_dalys",
    "ce_cost",
    "pop_size",
    "pop_pin",
    "feascov",
    "conscost",
    *OPTIMIZER_HR_COLS,
]

def _rename_hrh_map(_df):
    """Map officer type labels from model output to optimizer cadre buckets.

    The mapping is deterministic and keyword-based. Unknown officer types are ignored.
    """
    # TODO check with Sakshi
    # This mapping silently ignores 'DCSA', 'Dental' and 'Radiography' cadres
    mapping = (
        {
            'Clinical': 'hr_clin',
            'Laboratory': 'hr_lab',
            'Mental': 'hr_ment',
            'Nursing_and_Midwifery': 'hr_nur',
            'Nutrition': 'hr_nutri',
            'Pharmacy': 'hr_pharm',
        }
    )
    # Rename dataframes indexed by officer type to what they are called in the
    # optimizer
    renamed = _df.rename(index=mapping)
    return renamed


# TODO: Check with Sakshi if we only use the central value.
def _coerce_central_series(df: pd.DataFrame) -> pd.Series:
    out = df["central"].copy()
    out.index = out.index.map(format_scenario_name)
    return out.astype(float)

def _build_optimizer_inputs(results: dict[str, Any]) -> pd.DataFrame:

    dalys_averted = results.get("dalys_averted")
    incremental_cost = results.get("incremental_scenario_cost")
    capacity_used = _rename_hrh_map(results.get("capacity_used_by_cadre"))

    ce_dalys = dalys_averted['central']
    ce_cost = incremental_cost['central']
    hr_needs = capacity_used.xs("central", level="stat", axis=1).T

    interventions = sorted(set(ce_dalys.index).intersection(set(ce_cost.index)))
    if not interventions:
        raise ValueError("No overlapping interventions found between DALYs and costs.")

    opt_df = pd.DataFrame(
        {
            "intcode": range(1, len(interventions) + 1),
            "intervention": interventions,
            "ce_dalys": [float(ce_dalys.loc[i]) for i in interventions],
            "ce_cost": [float(ce_cost.loc[i]) for i in interventions],
            "conscost": [float(ce_cost.loc[i]) for i in interventions],
            "hr_clin": [float(hr_needs.loc[i, "hr_clin"]) for i in interventions],
            "hr_nur": [float(hr_needs.loc[i, "hr_nur"]) for i in interventions],
            "hr_pharm": [float(hr_needs.loc[i, "hr_pharm"]) for i in interventions],
            "hr_lab": [float(hr_needs.loc[i, "hr_lab"]) for i in interventions],
            "hr_ment": [float(hr_needs.loc[i, "hr_ment"]) for i in interventions],
            "hr_nutri": [float(hr_needs.loc[i, "hr_nutri"]) for i in interventions],
        }
    )

    return opt_df

def _build_hr_constraints_from_results(results: dict[str, Any]) -> pd.DataFrame:

    capacity_constraints = _rename_hrh_map(results['annual_capacity_by_cadre'])
    count_constraints = _rename_hrh_map(results['staff_count_by_cadre'])
    combined = (
        {'capacity': capacity_constraints, 'staff_count': count_constraints}
    )
    return pd.DataFrame(combined)



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-results-pkl", type=Path, required=True)
    parser.add_argument(
        "--r-script-path",
        type=Path,
        default=Path("src/scripts/lcoa_inputs_from_tlo_analyses/optimizer_preaggregated.R"),
    )
    parser.add_argument("--rscript-bin", type=str, default="Rscript")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    with open(args.analysis_results_pkl, "rb") as f:
        results = pickle.load(f)

    opt_inputs = _build_optimizer_inputs(results)
    hr_constraints = _build_hr_constraints_from_results(results)

    optimizer_input_csv = Path("src/scripts/lcoa_inputs_from_tlo_analyses/optimizer_inputs.csv")
    hr_constraints_csv = Path("src/scripts/lcoa_inputs_from_tlo_analyses/hr_constraints.csv")

    opt_inputs.to_csv(optimizer_input_csv, index=True)
    hr_constraints.to_csv(hr_constraints_csv, index=True)

    print(f"Wrote optimizer input CSV: {optimizer_input_csv}")
    print(f"Wrote optimizer constraints CSV: {hr_constraints_csv}")


if __name__ == "__main__":
    main()
