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


def _coerce_central_series(df: pd.DataFrame, name: str) -> pd.Series:
    if "central" not in df.columns:
        raise ValueError(f"{name} must contain a 'central' column.")
    out = df["central"].copy()
    out.index = out.index.map(format_scenario_name)
    return out.astype(float)


def _build_hr_mapping(officer_types: list[str]) -> dict[str, str]:
    """Map officer type labels from model output to optimizer cadre buckets.

    The mapping is deterministic and keyword-based. Unknown officer types are ignored.
    """
    # TODO check with Sakshi
    # This mapping silently ignores 'DCSA', 'Dental' and 'Radiography' cadres
    mapping = (
        {
            'hr_clin': 'Clinical',
            'hr_lab': 'Laboratory',
            'hr_ment': 'Mental',
            'hr_nur': 'Nursing_and_Midwifery',
            'hr_nutri': 'Nutrition',
            'hr_pharm': 'Pharmacy',
        }
    )
    return mapping


def _aggregate_hr_by_intervention(
    capacity_used_by_cadre: pd.DataFrame,
    interventions: list[str],
) -> pd.DataFrame:
    if "central" not in capacity_used_by_cadre.columns:
        raise ValueError("results['capacity_used_by_cadre'] must contain a 'central' column.")

    officer_types = capacity_used_by_cadre.index.get_level_values(0).astype(str).tolist()

    mapping = _build_hr_mapping(officer_types)
    if not mapping:
        raise ValueError(
            "Could not map any OfficerType values to optimizer HR buckets. "
            "Check results['capacity_used_by_cadre'] index labels."
        )

    total_by_bucket = {k: 0.0 for k in OPTIMIZER_HR_COLS}
    for idx, value in capacity_used_by_cadre["central"].items():
        officer = str(idx[0] if isinstance(idx, tuple) else idx)
        bucket = mapping.get(officer)
        if bucket is not None:
            total_by_bucket[bucket] += float(value)

    rows = []
    for intervention in interventions:
        row = {"intervention": intervention, **total_by_bucket}
        rows.append(row)

    return pd.DataFrame(rows)


def _build_optimizer_inputs(results: dict[str, Any], constraints_df: pd.DataFrame) -> pd.DataFrame:
    dalys_averted = results.get("dalys_averted")
    incremental_cost = results.get("incremental_scenario_cost")
    capacity_used = results.get("capacity_used_by_cadre")

    if dalys_averted is None or incremental_cost is None or capacity_used is None:
        raise ValueError(
            "results pickle must contain 'dalys_averted', 'incremental_scenario_cost', and 'capacity_used_by_cadre'."
        )

    ce_dalys = _coerce_central_series(dalys_averted, "results['dalys_averted']")
    ce_cost = _coerce_central_series(incremental_cost, "results['incremental_scenario_cost']")

    interventions = sorted(set(ce_dalys.index).intersection(set(ce_cost.index)))
    if not interventions:
        raise ValueError("No overlapping interventions found between DALYs and costs.")

    hr_df = _aggregate_hr_by_intervention(capacity_used, interventions)

    opt_df = pd.DataFrame(
        {
            "intcode": interventions,
            "intervention": interventions,
            "ce_dalys": [float(ce_dalys.loc[i]) for i in interventions],
            "conscost": [float(ce_cost.loc[i]) for i in interventions],
            "ce_cost": [float(ce_cost.loc[i]) for i in interventions],
        }
    )

    opt_df = opt_df.merge(hr_df, on="intervention", how="left")
    _require_columns(opt_df, REQUIRED_OPT_INPUT_COLS, "optimizer input dataframe")

    # Ensure numeric columns are numeric.
    numeric_cols = [c for c in REQUIRED_OPT_INPUT_COLS if c not in {"intcode", "intervention"}]
    for col in numeric_cols:
        opt_df[col] = pd.to_numeric(opt_df[col], errors="raise")

    return opt_df[REQUIRED_OPT_INPUT_COLS]


def _parse_constraints(constraints_df: pd.DataFrame, intervention_codes: list[str]) -> dict[str, Any]:
    _require_columns(constraints_df, ["section"], "constraints CSV")
    constraints_df = constraints_df.copy()
    constraints_df["section"] = constraints_df["section"].astype(str).str.strip()

    global_df = constraints_df.loc[constraints_df["section"] == "global"].copy()
    _require_columns(global_df, ["key", "value"], "global section")
    globals_map = (
        global_df.dropna(subset=["key", "value"])
        .drop_duplicates(subset=["key"], keep="last")
        .set_index("key")["value"]
        .to_dict()
    )

    missing_global = sorted(k for k in GLOBAL_REQUIRED_KEYS if k not in globals_map)
    if missing_global:
        raise ValueError(f"Missing required global constraints: {missing_global}")

    def parse_vector(section_name: str) -> list[float]:
        sec = constraints_df.loc[constraints_df["section"] == section_name].copy()
        _require_columns(sec, ["key", "value"], f"{section_name} section")
        order = OPTIMIZER_HR_COLS
        sec = sec.dropna(subset=["key", "value"]).drop_duplicates(subset=["key"], keep="last")
        sec_map = sec.set_index("key")["value"].to_dict()
        missing = [k for k in order if k not in sec_map]
        if missing:
            raise ValueError(f"Missing {section_name} values for keys: {missing}")
        return [float(sec_map[k]) for k in order]

    hr_time_constraint = parse_vector("hr_time_constraint")
    hr_size = parse_vector("hr_size")
    hr_scale = parse_vector("hr_scale")

    compulsory_df = constraints_df.loc[constraints_df["section"] == "compulsory"].copy()
    compulsory_interventions: list[str] = []
    if not compulsory_df.empty:
        _require_columns(compulsory_df, ["intcode"], "compulsory section")
        compulsory_interventions = sorted(
            {
                format_scenario_name(i)
                for i in compulsory_df["intcode"].dropna().astype(str).tolist()
            }
        )

    unknown_compulsory = sorted(set(compulsory_interventions) - set(intervention_codes))
    if unknown_compulsory:
        raise ValueError(f"Compulsory interventions not in optimizer input: {unknown_compulsory}")

    subs_df = constraints_df.loc[constraints_df["section"] == "substitute_group"].copy()
    substitutes: list[list[str]] = []
    if not subs_df.empty:
        _require_columns(subs_df, ["group_id", "intcode"], "substitute_group section")
        subs_df = subs_df.dropna(subset=["group_id", "intcode"]).copy()
        subs_df["intcode"] = subs_df["intcode"].astype(str).map(format_scenario_name)
        for _, grp in subs_df.groupby("group_id"):
            members = sorted(set(grp["intcode"].tolist()))
            if len(members) > 0:
                substitutes.append(members)

    unknown_subs = sorted({x for grp in substitutes for x in grp} - set(intervention_codes))
    if unknown_subs:
        raise ValueError(f"Substitute interventions not in optimizer input: {unknown_subs}")

    return {
        "objective_input": str(globals_map["objective_input"]),
        "cet_input": float(globals_map["cet_input"]),
        "drug_budget_input": float(globals_map["drug_budget_input"]),
        "drug_budget.scale": float(globals_map["drug_budget.scale"]),
        "use_feasiblecov_constraint": int(float(globals_map["use_feasiblecov_constraint"])),
        "feascov_scale": float(globals_map["feascov_scale"]),
        "compcov_scale": float(globals_map["compcov_scale"]),
        "task_shifting_pharm": int(float(globals_map["task_shifting_pharm"])),
        "hr.time.constraint": hr_time_constraint,
        "hr.size": hr_size,
        "hr.scale": hr_scale,
        "compulsory_interventions": compulsory_interventions,
        "substitutes": substitutes,
    }


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


def _flatten_optimizer_output_for_csv(result_obj: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k, v in result_obj.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                rows.append({"metric": str(k), "submetric": str(sk), "value": sv})
        elif isinstance(v, list):
            rows.append({"metric": str(k), "submetric": "", "value": json.dumps(v)})
        else:
            rows.append({"metric": str(k), "submetric": "", "value": v})
    return pd.DataFrame(rows)


def _run_optimizer_via_rpy2(
    optimizer_inputs: pd.DataFrame,
    constraints: dict[str, Any],
    r_script_path: Path,
) -> dict[str, Any]:
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.vectors import FloatVector, IntVector, ListVector, StrVector
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

    r_compulsory = StrVector(constraints["compulsory_interventions"])

    # R code iterates as nested loops over substitutes[i], then k in j; this structure matches list(character vectors).
    r_subs = ListVector(
        {str(i + 1): StrVector(group) for i, group in enumerate(constraints["substitutes"])}
    )

    result_r = r_func(
        r_inputs,
        constraints["objective_input"],
        constraints["cet_input"],
        constraints["drug_budget_input"],
        constraints["drug_budget.scale"],
        FloatVector(constraints["hr.time.constraint"]),
        FloatVector(constraints["hr.size"]),
        FloatVector(constraints["hr.scale"]),
        IntVector([constraints["use_feasiblecov_constraint"]])[0],
        constraints["feascov_scale"],
        constraints["compcov_scale"],
        r_compulsory,
        r_subs,
        IntVector([constraints["task_shifting_pharm"]])[0],
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
    parser.add_argument("--constraints-csv", type=Path, required=True)
    parser.add_argument("--optimizer-input-csv", type=Path, required=True)
    parser.add_argument("--optimizer-output-json", type=Path, required=True)
    parser.add_argument("--optimizer-output-csv", type=Path, required=False, default=None)
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
    if not args.constraints_csv.exists():
        raise FileNotFoundError(f"Constraints CSV not found: {args.constraints_csv}")

    with open(args.analysis_results_pkl, "rb") as f:
        results = pickle.load(f)

    constraints_df = pd.read_csv(args.constraints_csv)
    _require_columns(constraints_df, ["section"], "constraints CSV")

    optimizer_inputs = _build_optimizer_inputs(results, constraints_df)
    args.optimizer_input_csv.parent.mkdir(parents=True, exist_ok=True)
    optimizer_inputs.to_csv(args.optimizer_input_csv, index=False)

    constraints = _parse_constraints(constraints_df, intervention_codes=optimizer_inputs["intcode"].tolist())
    optimizer_output = _run_optimizer_via_rpy2(
        optimizer_inputs=optimizer_inputs,
        constraints=constraints,
        r_script_path=args.r_script_path,
    )

    args.optimizer_output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.optimizer_output_json, "w", encoding="utf-8") as f:
        json.dump(_jsonify(optimizer_output), f, indent=2, sort_keys=True)

    if args.optimizer_output_csv is not None:
        flat_df = _flatten_optimizer_output_for_csv(optimizer_output)
        args.optimizer_output_csv.parent.mkdir(parents=True, exist_ok=True)
        flat_df.to_csv(args.optimizer_output_csv, index=False)

    print(f"Wrote optimizer input CSV: {args.optimizer_input_csv}")
    print(f"Wrote optimizer output JSON: {args.optimizer_output_json}")
    if args.optimizer_output_csv is not None:
        print(f"Wrote optimizer output CSV: {args.optimizer_output_csv}")


if __name__ == "__main__":
    main()
