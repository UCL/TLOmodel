"""
select_representative_models.py

Turn per-MODEL CMIP6 facility panels into per-ROLE files that
model_of_wbgt_dhis2.py reads via WBGT_MODELS = ["lowest", "median", "highest"].

It ranks the ensemble by the overall mean of RANK_COLUMN (default wbgtx_day,
matching WBGT_extreme_indices_projections.py), picks the coolest / middle /
hottest MODEL, and copies that model's file to a role-named file — for every
prefix in PANEL_PREFIXES, so the SAME three models are used whether the
regression toggles to extreme indices or monthly means.

Run this AFTER the per-model panels exist:
  - wbgt_extreme_indices_facility_{model}_{scenario}.csv  (extremes script)
  - wbgt_monthly_mean_facility_{model}_{scenario}.csv     (facility extractor)

Output (per prefix, per role):
  {prefix}{lowest|median|highest}_{scenario}.csv
and a small mapping: representative_models_{scenario}.csv

Re-run this whenever you regenerate a model panel, so the role copies stay in
sync (they are plain copies, not links).
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SCENARIOS = ["ssp126", "ssp245", "ssp585"]
INDICES_DIR = Path(
    "/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/Indices"
)

# Panel prefixes to materialise role copies for. Comment out either line if you
# only use one quantity.
PANEL_PREFIXES = [
    #"wbgt_extreme_indices_facility_",   # WBGTx / WBGT5x panels
    "wbgt_monthly_mean_facility_",      # monthly-mean wbgt_day/night panels
]

# One ranking applied to ALL prefixes, so low/median/high are the same models
# across quantities. wbgtx_day matches the ranking in the extremes script.
RANK_PREFIX = "wbgt_monthly_mean_facility_"#"wbgt_extreme_indices_facility_"
RANK_COLUMN = "wbgt_day"#"wbgtx_day"

ROLES = ["lowest", "median", "highest"]   # exact words model_of_wbgt_dhis2 expects


# ---------------------------------------------------------------------------
def model_files_for(prefix, SCENARIO):
    """{model_id: path} for {prefix}{model}_{scenario}.csv, skipping any files
    already named with a role word (so re-runs don't rank the copies)."""
    tail = f"_{SCENARIO}.csv"
    out = {}
    for p in sorted(INDICES_DIR.glob(f"{prefix}*{tail}")):
        token = p.name[len(prefix):-len(tail)]
        if token in ROLES:
            continue
        out[token] = p
    return out


def rank_models(SCENARIO):
    """Rank models coolest->hottest by mean(RANK_COLUMN). Models with an
    all-NaN column are excluded with a warning."""
    files = model_files_for(RANK_PREFIX, SCENARIO)  # FIXED: Added SCENARIO argument
    if not files:
        raise FileNotFoundError(
            f"No {RANK_PREFIX}*{'_' + SCENARIO}.csv in {INDICES_DIR} — run the "
            f"panel producer first.")
    means = {}
    for model, p in files.items():
        col = pd.read_csv(p, usecols=[RANK_COLUMN])[RANK_COLUMN].to_numpy()
        m = np.nanmean(col) if np.isfinite(col).any() else np.nan
        if np.isnan(m):
            print(f"  ⚠ {model}: all-NaN {RANK_COLUMN} — excluded from ranking")
        else:
            means[model] = float(m)
    ranked = sorted(means, key=means.get)          # coolest -> hottest
    return ranked, means


def pick_roles(ranked):
    """lowest = coolest, highest = hottest, median = lower-middle (matches the
    (n-1)//2 convention in the extremes script for even ensemble sizes)."""
    n = len(ranked)
    if n < 3:
        raise ValueError(f"need >=3 ranked models, have {n}: {ranked}")
    return {"lowest": ranked[0],
            "median": ranked[(n - 1) // 2],
            "highest": ranked[-1]}


def main():
    for SCENARIO in SCENARIOS:
        ranked, means = rank_models(SCENARIO = SCENARIO)
        print(f"Model ranking by mean {RANK_COLUMN} (coolest -> hottest):")
        for m in ranked:
            print(f"  {m:25s} {means[m]:.2f} °C")

        roles = pick_roles(ranked)
        print("\nRepresentative models:")
        for role in ROLES:
            print(f"  {role:8s} -> {roles[role]} ({means[roles[role]]:.2f} °C)")

        # Materialise role-named copies for each panel prefix
        for prefix in PANEL_PREFIXES:
            files = model_files_for(prefix, SCENARIO)  # FIXED: Added SCENARIO argument
            if not files:
                print(f"\n⚠ no files for prefix '{prefix}' — skipping this panel type")
                continue
            print(f"\n{prefix}:")
            for role in ROLES:
                model = roles[role]
                src = files.get(model)
                if src is None:
                    print(f"  ⚠ {role}: model '{model}' has no {prefix} file — skipped")
                    continue
                dst = INDICES_DIR / f"{prefix}{role}_{SCENARIO}.csv"
                shutil.copyfile(src, dst)
                print(f"  {role:8s} <- {src.name}  ->  {dst.name}")

        # Mapping for the record
        map_df = pd.DataFrame(
            {"role": ROLES,
             "model": [roles[r] for r in ROLES],
             f"mean_{RANK_COLUMN}": [means[roles[r]] for r in ROLES]})
        map_path = INDICES_DIR / f"representative_models_{SCENARIO}.csv"
        map_df.to_csv(map_path, index=False)
        print(f"\nSaved mapping -> {map_path.name}")


if __name__ == "__main__":
    main()
