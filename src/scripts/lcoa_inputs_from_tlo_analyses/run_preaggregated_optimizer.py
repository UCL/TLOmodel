"""Run the preaggregated R optimizer from Python inputs.

This script:
1. Loads analysis outputs produced by analysis_effect_of_treatment_ids.py.
2. Builds and writes the optimizer intervention input CSV.
3. Loads optimizer constraints from a separate CSV.
4. Invokes optimizer_preaggregated.R::find_optimal_package via rpy2.
5. Writes optimizer outputs to JSON and optional CSV.
"""

from __future__ import annotations

from scripts.lcoa_inputs_from_tlo_analyses.results_processing_utils import format_scenario_name

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OPTIMIZER_HR_COLS = ["hr_clin", "hr_nur", "hr_pharm", "hr_lab", "hr_ment", "hr_nutri"]
REQUIRED_OPT_INPUT_COLS = [
    "intcode",
    "intervention",
    "ce_dalys",
    "conscost",
    "feascov",
    "ce_cost",
    *OPTIMIZER_HR_COLS,
]

GLOBAL_REQUIRED_KEYS = {
    "objective_input",
    "cet_input",
    "drug_budget_input",
    "drug_budget.scale",
    "use_feasiblecov_constraint",
    "feascov_scale",
    "compcov_scale",
    "task_shifting_pharm",
}


def _require_columns(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")

# TODO: Check with Sakshi if we only use the central value.
def _coerce_central_series(df: pd.DataFrame) -> pd.Series:
    out = df["central"].copy()
    out.index = out.index.map(format_scenario_name)
    return out.astype(float)


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


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, pd.Series):
        return {str(k): _jsonify(v) for k, v in value.to_dict().items()}
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, np.ndarray):
        return [_jsonify(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def _run_optimizer_via_rpy2(
    optimizer_inputs: pd.DataFrame,
    constraints: dict[str, Any],
    r_script_path: Path,
) -> dict[str, Any]:
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.vectors import FloatVector, ListVector, StrVector
    except ImportError as exc:
        raise RuntimeError(
            "rpy2 is required but not available. Install rpy2 in your Python environment."
        ) from exc

    if not r_script_path.exists():
        raise FileNotFoundError(f"R script not found: {r_script_path}")

    ro.r["source"](str(r_script_path))
    r_func = ro.globalenv.find("find_optimal_package")

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_inputs = ro.conversion.py2rpy(optimizer_inputs)

    r_compulsory = StrVector([])
    r_subs = ListVector({})

    result_r = r_func(
        r_inputs,
        # whether we are maximizing DALYs or net health
        "dalys",
        # CET; I believe not relevant here but give a value anyway
        600,
        # Drug budget input
        constraints['annual_consumables_budget'],
        # Drug budget scale set to 1
        1,
        # HR constraints; need to be clinical staff, nursing, pharmacy, lab,
        # mental health, nutrition in that order
        FloatVector(constraints["hr_time_constraint"]),
        # HR size; same order as above
        FloatVector(constraints["hr_size"]),
        1,
        # use_feasiblecov_constraint; set to 0 to not use, 1 to use
        0,
        # Feasible coverage scale; set to 1
        1,
        # Compulsory coverage scale; set to 1
        1,
        # Compulsory interventions; pass empty list,
        r_compulsory,
        # substitutes; pass empty list
        r_subs,
        # task_shifting_pharm; set to 0 to not allow, 1 to allow
        0,
    )

    with localconverter(ro.default_converter + pandas2ri.converter):
        result_py = ro.conversion.rpy2py(result_r)

    # rpy2 can return named list-like objects; normalize to dict.
    if isinstance(result_py, dict):
        return {str(k): _jsonify(v) for k, v in result_py.items()}

    if hasattr(result_r, "names"):
        out: dict[str, Any] = {}
        names = list(result_r.names)
        for i, name in enumerate(names):
            out[str(name)] = _jsonify(ro.conversion.rpy2py(result_r[i]))
        return out

    raise RuntimeError("Unexpected optimizer result type from R.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-results-pkl", type=Path, required=True)
    parser.add_argument("--optimizer-output-json", type=Path, required=True)
    parser.add_argument(
        "--r-script-path",
        type=Path,
        default=Path("src/scripts/lcoa_inputs_from_tlo_analyses/optimizer_preaggregated.R"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.analysis_results_pkl.exists():
        raise FileNotFoundError(f"Analysis results pickle not found: {args.analysis_results_pkl}")

    with open(args.analysis_results_pkl, "rb") as f:
        results = pickle.load(f)

    constraints = ({
        'annual_consumables_budget': results.get("annual_consumables_budget"),
        'hr_time_constraint': _rename_hrh_map(results.get("annual_capacity_by_cadre")),
        'hr_size': _rename_hrh_map(results.get("staff_count_by_cadre"))
    })
    optimizer_inputs = _build_optimizer_inputs(results)

    optimizer_output = _run_optimizer_via_rpy2(
        optimizer_inputs=optimizer_inputs,
        constraints=constraints,
        r_script_path=args.r_script_path,
    )

    args.optimizer_output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.optimizer_output_json, "w", encoding="utf-8") as f:
        json.dump(_jsonify(optimizer_output), f, indent=2, sort_keys=True)

if __name__ == "__main__":
    main()
