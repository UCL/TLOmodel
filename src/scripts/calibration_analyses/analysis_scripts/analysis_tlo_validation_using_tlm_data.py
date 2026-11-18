"""
Validation of the Thanzi La Onse (TLO) model against the Thanzi La Mawa (TLM) 2024 survey:
patient load per worker-day, patient mix and access to prescribed medicines.

TLO
  - Outputs restricted to the TLM period, districts (central hospitals by region) and facility
    levels 1a, 2 and 3 (level 1b merged into 2, level 4 (ZMH) into 3; levels 0 and 5 and DCSA
    staff excluded).
  - Each measure is calculated per run and by facility level, then reweighted to the TLM
    facility-level composition, using the weights exported from the Stata analysis.
  - Summarised across runs with tlo.analysis.utils.summarize (mean, 95% CI of the mean from the
    standard error, t multiplier with n_runs - 1 df).
TLM
  - Pooled estimates with delete-one-facility jackknife 95% CIs, read from the Stata outputs.

Sensitivity analyses (choose with --analysis; outputs in <output folder>/tlm_validation/<analysis>):
  (1) no_reweighting:     TLO estimates not reweighted to the TLM facility-level composition
  (2) no_blank_footprint: HSIs with a blank appointment footprint dropped (patient load and patient mix)
  (3) outpatient_meds:    medicine access restricted to outpatient service areas
  (4) cadre_weighted_tms: TLM time-and-motion patient load standardised to the TLO cadre mix
  (5) inpatient_days: TLO in-patient days added to patient load and to the inpatient service areas.

The TLO simulation scenario is defined in src/scripts/calibration_analyses/scenarios/scenario_patient_mix.py.
"""
import argparse
import ast
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from tlo import Date
from tlo.analysis.utils import extract_results, summarize

# ----------------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------------
LEVELS = ["1a", "2", "3"]
LEVEL_LABELS = {
    "Overall": "Overall",
    "Level 1a": "Health Centres",
    "Level 2": "Primary/Secondary Hospitals",
    "Level 3": "Tertiary Hospitals",
}
TLM_CI_METHOD = "jackknife"          # main CI method in the TLM results files

# TLO assumes all staff are available every day; scale staff to the expected number on duty
HCW_ON_DUTY_FRACTION = 206.3381 / 365.25

# Main analysis keeps HSIs with a blank appointment footprint and HSIs that did not run due to no mental health staff.
# Sensitivity analysis on the definition of a patient: run with --analysis no_blank_footprint.
KEEP_BLANK_FOOTPRINT_HSI = True
DROP_DID_NOT_RUN_HSI = False  # TLO simulation is in Mode 1, every requested HSI should run

# Stata outputs (results and level weights), expected in the TLM folder
TLM_FILES = {
    "load_tms": "tlm_patient_load_from_tms.csv",
    "load_clinic": "tlm_patient_load_from_clinic.csv",
    "mix": "tlm_patient_mix_from_exit.csv",
    "meds": "tlm_med_access_from_exit.csv",
    "w_tms": "tlm_tms_weights.csv",
    "w_clinic": "tlm_clinic_weights.csv",
    "w_mix": "tlm_pat_mix_weights.csv",
    "w_meds": "tlm_med_access_weights.csv",
    # sensitivity analysis: medicine access among prescribed patients in outpatient areas
    "meds_outpatient": "tlm_med_access_outpatient_from_exit.csv",
    "w_meds_outpatient": "tlm_med_access_outpatient_weights.csv",
    # sensitivity analysis: TLM time-and-motion patient load standardised to the TLO cadre mix
    "load_tms_cadre": "tlm_patient_load_from_tms_cadre_weighted.csv",
}

# Patient mix: TLM variable names -> categories, and harmonised subgroup labels
MIX_VARS = {"sex_n": "Sex", "age_grp": "Age group", "educ_n": "Education",
            "wealth_n": "Wealth quintile", "loc_cat": "Service area"}
SEX_LABELS = {"F": "Female", "M": "Male"}
EDUC_LABELS = {1: "None", 2: "Some/Completed primary", 3: "Some/Completed secondary"}
WEALTH_LABELS = {1: "Q1 (richest)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 (poorest)"}
AGE_TOP_GROUP = 75                    # age groups from 75 upwards merged into "75+"

# Education and wealth: TLM respondents may be companions, so compare adults attending for own care
ADULT_MIN_AGE = 20                                   # must match the threshold used for reason_attend_cat in Stata
ADULT_ONLY_CATS = ("Education", "Wealth quintile")   # categories restricted to adults
ADULT_GROUP = "Adults seeking care"                  # group name in tlm_pat_mix_weights.csv

SERVICE_AREA_ORDER = [
    "Outpatient - General", "Outpatient - Maternity/Gynaecology", "CD clinic",
    "Inpatient - General", "Inpatient - Maternity/Gynaecology", "Emergency / Intensive care",
]
TLO_MEDS_ACCESS_MEASURES = ["complete", "proportional", "any"]

# Districts containing TLM facilities, as named in TLO (TLM "Mzuzu" = TLO "Mzuzu City";
# central hospitals pooled by region).
TLM_DISTRICTS = [
    # Northern
    "Karonga", "Likoma", "Mzuzu City", "Nkhata Bay", "Rumphi", "Central Hospitals (Northern)",
    # Central
    "Dedza", "Lilongwe", "Ntcheu", "Central Hospitals (Central)",
    # Southern
    "Blantyre", "Chiradzulu", "Neno", "Nsanje", "Zomba", "Central Hospitals (Southern)",
]

# Analyses (choose with --analysis); each saves its outputs in <results folder>/tlm_validation/<analysis>
ANALYSES = {
    "main": "Main analysis",
    "no_reweighting": "Sensitivity (1): TLO estimates not reweighted to the TLM facility-level composition",
    "no_blank_footprint": "Sensitivity (2): HSIs with a blank appointment footprint dropped "
                          "(patient load and patient mix)",
    "outpatient_meds": "Sensitivity (3): medicine access restricted to outpatient service areas",
    "cadre_weighted_tms": "Sensitivity (4): TLM time-and-motion patient load standardised to the TLO "
                          "cadre mix within each facility level",
    "inpatient_days": "Sensitivity (5): TLO in-patient days added to patient load and to the inpatient "
                      "service areas in patient mix",
}
MEASURES_BY_ANALYSIS = {
    "main": {"load", "mix", "meds"},
    "no_reweighting": {"load", "mix", "meds"},
    "no_blank_footprint": {"load", "mix"},     # the Consumables log has no appointment footprint
    "outpatient_meds": {"meds"},
    "cadre_weighted_tms": {"load"},
    "inpatient_days": {"load", "mix"},
}
# Outpatient service areas for sensitivity analysis (3); must match the Stata definition exactly
OUTPATIENT_AREAS = ["Outpatient - General", "Outpatient - Maternity/Gynaecology", "CD clinic"]

# Cadre groups for sensitivity analysis (4): must match the cadres in Stata TMS
CADRE_GROUPS = {"Clinical": "Clinical", "Nursing_and_Midwifery": "Nursing and Midwifery",
                "Pharmacy": "Pharmacy", "Laboratory": "Laboratory", "Dental": "Dental",
                "Mental": "Clinical", "Radiography": "Radiography"}

# Plot colours
# (TLM time-and-motion and exit survey blue, TLM clinic records green, TLO orange)
COLOUR_TLM_TIME_MOTION = "tab:blue"      # TLM time-and-motion study (patient load)
COLOUR_TLM_EXIT_SURVEY = "tab:blue"      # TLM patient exit survey (patient mix, medicine access)
COLOUR_TLM_CLINIC_RECORDS = "tab:green"  # TLM clinic operations records (patient load)
COLOUR_TLO = "tab:orange"                # TLO model outputs


# ----------------------------------------------------------------------------------------------
# General helpers
# ----------------------------------------------------------------------------------------------
def age_label(age_range) -> Optional[str]:
    """Map a TLO Age_Range (e.g. '25-29', '100+') to the TLM groups, merging ages >= AGE_TOP_GROUP."""
    try:
        start = int(str(age_range).split("-")[0].rstrip("+"))
    except ValueError:
        return None
    return f"{AGE_TOP_GROUP}+" if start >= AGE_TOP_GROUP else f"{start}-{start + 4}"


def age_order(label) -> int:
    return int(str(label).split("-")[0].rstrip("+"))


def code_label(x, labels: Dict[int, str]) -> Optional[str]:
    v = pd.to_numeric(x, errors="coerce")
    return labels.get(int(v)) if pd.notna(v) else None


def summarise_runs(per_run: pd.DataFrame, prefix: str = "tlo") -> pd.DataFrame:
    """Rows = estimates, columns = runs -> mean and 95% CI of the mean, using the TLO `summarize`
    (mean +/- t x SE across runs)."""
    results = per_run.copy()
    results.columns = pd.MultiIndex.from_product([[0], per_run.columns], names=["draw", "run"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")       # summarize warns that it uses the mean, as intended here
        stats = summarize(results, collapse_columns=True)

    return pd.DataFrame({
        f"{prefix}_mean": stats["mean"],
        f"{prefix}_lower": stats["lower"],
        f"{prefix}_upper": stats["upper"],
    })


def read_tlm(path: Path) -> pd.DataFrame:
    """TLM results (Stata output), main CI method only."""
    d = pd.read_csv(path)
    d = d.loc[d["method"] == TLM_CI_METHOD].copy()
    return d.rename(columns={"est": "tlm_est", "lb": "tlm_lower", "ub": "tlm_upper"})


def read_level_weights(path: Path) -> Dict[str, pd.Series]:
    """Weights by TLO facility level for each group in a Stata weights file."""
    w = pd.read_csv(path, dtype={"fac_lvl": str})
    w["fac_lvl"] = w["fac_lvl"].str.strip()
    out = {}
    for group, d in w.groupby("group"):
        s = d.set_index("fac_lvl")["weight"].reindex(LEVELS, fill_value=0.0)
        assert np.isclose(s.sum(), 1.0), f"Weights for '{group}' in {path.name} do not sum to 1"
        out[group] = s
    return out


def combine_levels(values: pd.Series, weights: pd.Series) -> Tuple[float, bool]:
    """Weighted average across levels. If TLO has no data at a level with a TLM weight,
    the weights are renormalised over the remaining levels and the result is flagged."""
    w = weights.reindex(values.index).fillna(0.0)
    ok = values.notna() & (w > 0)
    if not ok.any():
        return np.nan, True
    renormalised = not np.isclose(w[ok].sum(), w.sum())
    return float((w[ok] * values[ok]).sum() / w[ok].sum()), renormalised


def errors(est, lower, upper):
    """Asymmetric error-bar lengths from CI bounds (never negative)."""
    est, lower, upper = (pd.Series(x, dtype=float) for x in (est, lower, upper))
    return [(est - lower).clip(lower=0).fillna(0).to_numpy(),
            (upper - est).clip(lower=0).fillna(0).to_numpy()]


def style_axis(ax, axis="x"):
    ax.grid(axis=axis, color="#e1e0d9", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ----------------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------------
def forest_layout(sections, label_map=None):
    """Row positions for a forest plot. sections: list of (title or None, category, [subgroups]).
    Returns data rows [(category, subgroup, y)], tick positions, labels, bold flags and total height."""
    label_map = label_map or {}
    rows, ypos, ylab, bold, y = [], [], [], [], 0.0
    for title, cat, subgroups in sections:
        if title:
            ypos.append(y), ylab.append(title), bold.append(True)
            y += 1
        for s in subgroups:
            rows.append((cat, s, y))
            ypos.append(y), ylab.append(label_map.get(s, s)), bold.append(s == "Overall")
            y += 1
        y += 0.3
    return rows, ypos, ylab, bold, y


def draw_forest(ax, cmp: pd.DataFrame, sections, series, xlabel: str, xlim=None, band=None,
                label_map=None):
    """
    Draw a horizontal forest plot on an existing axis.
    cmp:      DataFrame indexed by (category, subgroup).
    sections: list of (title or None, category, [subgroups]).
    series:   list of dicts with est/lo/hi column names, a vertical offset and errorbar style.
    band:     optional dict(lo=col, hi=col, y0=offset, y1=offset, color=...) for a shaded range.
    """
    rows, ypos, ylab, bold, height = forest_layout(sections, label_map)

    if band is not None:
        for cat, s, yy in rows:
            if (cat, s) not in cmp.index:
                continue
            r = cmp.loc[(cat, s)]
            if pd.notna(r[band["lo"]]) and pd.notna(r[band["hi"]]):
                ax.fill_betweenx([yy + band["y0"], yy + band["y1"]], r[band["lo"]], r[band["hi"]],
                                 color=band["color"], alpha=0.15, lw=0)

    for sr in series:
        style = {k: v for k, v in sr.items() if k not in ("est", "lo", "hi", "offset")}
        xs, ys, lo, hi = [], [], [], []
        for cat, s, yy in rows:
            if (cat, s) not in cmp.index:
                continue
            r = cmp.loc[(cat, s)]
            if pd.isna(r[sr["est"]]):
                continue
            xs.append(r[sr["est"]]), ys.append(yy + sr["offset"])
            lo.append(r[sr["lo"]]), hi.append(r[sr["hi"]])
        if xs:
            ax.errorbar(xs, ys, xerr=errors(xs, lo, hi), linestyle="none", ms=6,
                        elinewidth=1.4, capsize=0, **style)

    ax.set_yticks(ypos)
    ax.set_yticklabels(ylab)
    for tick, is_bold in zip(ax.get_yticklabels(), bold):
        tick.set_fontweight("bold" if is_bold else "normal")
    ax.set_ylim(height - 0.3 + 0.2, -0.6)          # top-down order
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    style_axis(ax)
    return height


def forest_plot(cmp: pd.DataFrame, sections, series, xlabel: str, path: Path,
                xlim=None, band=None, label_map=None, width=7.5, legend_cols=2, legend_rows=None):
    """
    Single-panel forest plot saved to `path` (see draw_forest).
    legend_rows: optional list of lists of series positions, one list per legend row,
                 e.g. [[0], [1, 2, 3]] puts series 0 on the first row and series 1-3 on the second.
    """
    height = forest_layout(sections, label_map)[-1]
    fig_height = 0.42 * height + 1.6
    fig, ax = plt.subplots(figsize=(width, fig_height))
    draw_forest(ax, cmp, sections, series, xlabel, xlim=xlim, band=band, label_map=label_map)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    fig.tight_layout()

    # legend below the figure, clear of the x-axis label
    handles, labels = ax.get_legend_handles_labels()
    if legend_rows is None:
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0),
                  bbox_transform=fig.transFigure, ncol=legend_cols, frameon=False, fontsize=9)
    else:
        row_step = 0.3 / fig_height                      # about 0.3 inch per legend row
        for i, row in enumerate(legend_rows):
            leg = fig.legend([handles[j] for j in row], [labels[j] for j in row],
                             loc="upper center", bbox_to_anchor=(0.5, -i * row_step),
                             ncol=len(row), frameon=False, fontsize=9)
            fig.add_artist(leg)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_patient_load(cmp_tms: pd.DataFrame, cmp_clinic: pd.DataFrame, path: Path):
    """Two panels: TLO vs each TLM source (TLO reweighted to each source's level composition)."""
    groups = ["Overall", "Level 1a", "Level 2", "Level 3"]
    # rows: Overall, then a "By facility level" heading followed by the indented levels
    y = np.array([0.0, 2.3, 3.3, 4.3])
    tick_pos = [0.0, 1.3, 2.3, 3.3, 4.3]
    tick_lab = ["Overall", "Facility level"] + ["   " + LEVEL_LABELS[g] for g in groups[1:]]
    tick_bold = [True, True, False, False, False]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True, sharex=True)
    panels = [
        (axes[1], cmp_tms, "(b) TLO vs TLM time-and-motion study", "o", COLOUR_TLM_TIME_MOTION, "TLM time-and-motion"),
        (axes[0], cmp_clinic, "(a) TLO vs TLM clinic records", "s", COLOUR_TLM_CLINIC_RECORDS, "TLM clinic records"),
    ]
    for ax, cmp, title, marker, colour, label in panels:
        d = cmp.set_index("group").reindex(groups)
        ax.errorbar(d["tlm_est"], y - 0.12, xerr=errors(d["tlm_est"], d["tlm_lower"], d["tlm_upper"]),
                    fmt=marker, color=colour, ms=6, elinewidth=1.4, linestyle="none",
                    label=f"{label} (95% CI, jackknife)")
        ax.errorbar(d["tlo_mean"], y + 0.12, xerr=errors(d["tlo_mean"], d["tlo_lower"], d["tlo_upper"]),
                    fmt="D", color=COLOUR_TLO, ms=6, elinewidth=1.4, linestyle="none",
                    label="TLO (95% CI across runs)")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("Number of patients seen per worker-day")
        style_axis(ax)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False, fontsize=9)
    # shared x-axis from 0 to just beyond the largest upper bound in either panel
    xmax = max(pd.concat([c[["tlm_upper", "tlo_upper"]] for c in (cmp_tms, cmp_clinic)]).max().max(), 1)
    axes[0].set_xlim(0, xmax * 1.05)
    axes[0].xaxis.set_major_locator(MultipleLocator(10))
    axes[0].set_yticks(tick_pos)
    axes[0].set_yticklabels(tick_lab)
    for tick, is_bold in zip(axes[0].get_yticklabels(), tick_bold):
        tick.set_fontweight("bold" if is_bold else "normal")
    axes[0].set_ylim(-0.6, 4.9)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_age_profile(ax, cmp_age: pd.DataFrame):
    """Age profile on an existing axis: TLM points with CIs and TLO line with CIs."""
    d = cmp_age.assign(order=cmp_age["subgroup"].map(age_order)).sort_values("order")
    x = np.arange(len(d))
    ax.errorbar(x - 0.1, d["tlm_est"], yerr=errors(d["tlm_est"], d["tlm_lower"], d["tlm_upper"]),
                fmt="o", color=COLOUR_TLM_EXIT_SURVEY, ms=6, elinewidth=1.4, linestyle="none",
                label="TLM patient exit survey (95% CI, jackknife)")
    ax.plot(x + 0.1, d["tlo_mean"], color=COLOUR_TLO, lw=1.5, alpha=0.8)
    ax.errorbar(x + 0.1, d["tlo_mean"], yerr=errors(d["tlo_mean"], d["tlo_lower"], d["tlo_upper"]),
                fmt="D", color=COLOUR_TLO, ms=5, elinewidth=1.4, linestyle="none",
                label="TLO (95% CI across runs)")
    ax.set_xticks(x)
    ax.set_xticklabels(d["subgroup"], rotation=45, ha="right")
    ax.set_xlabel("Age group (years)")
    ax.set_ylabel("Share of patients (%)")
    ax.set_ylim(bottom=0)
    style_axis(ax, axis="y")


def nice_upper(cmp: pd.DataFrame, cols, step=10, cap=100):
    """Axis upper limit: the largest value in `cols`, plus 5%, rounded up to `step` (at most `cap`)."""
    top = np.nanmax(cmp[cols].to_numpy(dtype=float)) * 1.05
    return min(cap, step * np.ceil(top / step))


def plot_patient_mix(cmp_mix: pd.DataFrame, path: Path):
    """
    One patient-mix figure with three panels:
      (a) sex, education and wealth quintile (left, full height)
      (b) age profile (top right)
      (c) service area (bottom right)
    and a single shared legend.
    """
    idx = cmp_mix.set_index(["category", "subgroup"])
    series = [
        dict(est="tlm_est", lo="tlm_lower", hi="tlm_upper", offset=-0.15, fmt="o",
             color=COLOUR_TLM_EXIT_SURVEY, label="TLM patient exit survey (95% CI, jackknife)"),
        dict(est="tlo_mean", lo="tlo_lower", hi="tlo_upper", offset=0.15, fmt="D", color=COLOUR_TLO,
             label="TLO (95% CI across runs)"),
    ]
    sections_a = [("Sex", "Sex", ["Female", "Male"]),
                  ("Education", "Education", list(EDUC_LABELS.values())),
                  ("Wealth quintile", "Wealth quintile", list(WEALTH_LABELS.values()))]
    areas = [a for a in SERVICE_AREA_ORDER if ("Service area", a) in idx.index]
    areas += sorted(s for c, s in idx.index if c == "Service area" and s not in areas)
    sections_b = [(None, "Service area", areas)]

    cols = ["tlm_upper", "tlo_upper"]
    xmax_a = nice_upper(cmp_mix.loc[cmp_mix["category"].isin(["Sex", "Education", "Wealth quintile"])], cols)
    xmax_b = nice_upper(cmp_mix.loc[cmp_mix["category"] == "Service area"], cols)

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1.15, 1], wspace=0.55, hspace=0.45)
    ax_a = fig.add_subplot(gs[:, 0])      # sex, education, wealth
    ax_b = fig.add_subplot(gs[0, 1])      # age profile
    ax_c = fig.add_subplot(gs[1, 1])      # service area

    draw_forest(ax_a, idx, sections_a, series, "Share of patients (%)", xlim=(0, xmax_a))
    ax_a.xaxis.set_major_locator(MultipleLocator(10))
    # wrap long service-area labels onto two lines so they do not run into panel (a)
    wrap_b = {a: a.replace(" - ", " -\n", 1) if len(a) > 22 else a for a in areas}
    draw_age_profile(ax_b, cmp_mix.loc[cmp_mix["category"] == "Age group"])
    draw_forest(ax_c, idx, sections_b, series, "Share of patients (%)", xlim=(0, xmax_b),
                label_map=wrap_b)

    for ax, title in ((ax_a, "(a) Sex, Education and Wealth quintile"), (ax_b, "(b) Age group"),
                      (ax_c, "(c) Service area")):
        ax.set_title(title, loc="left", fontsize=11)

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2,
               frameon=False, fontsize=10)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------------------------
def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path,
          the_target_period: Tuple[Date, Date], tlm_results_folder: Optional[Path] = None,
          analysis: str = "main"):

    assert analysis in ANALYSES, f"Unknown analysis '{analysis}'; choose from {list(ANALYSES)}"
    print(ANALYSES[analysis])
    reweight = analysis != "no_reweighting"
    keep_blank_footprint = KEEP_BLANK_FOOTPRINT_HSI and analysis != "no_blank_footprint"
    outpatient_only = analysis == "outpatient_meds"
    tms_file = TLM_FILES["load_tms_cadre" if analysis == "cadre_weighted_tms" else "load_tms"]
    add_inpatient_days = analysis == "inpatient_days"
    run_measures = MEASURES_BY_ANALYSIS[analysis]

    TARGET_PERIOD = the_target_period
    n_days = (TARGET_PERIOD[1] - TARGET_PERIOD[0]).days + 1
    output_folder = output_folder / "tlm_validation" / analysis
    output_folder.mkdir(parents=True, exist_ok=True)

    path_to_tlm_folder = resourcefilepath / "healthsystem" / "human_resources" / "TLM_2024"
    tlm_folder = tlm_results_folder or path_to_tlm_folder

    mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation"
                      / "ResourceFile_Master_Facilities_List.csv")
    item_is_medicine = pd.read_csv(resourcefilepath / "healthsystem" / "consumables"
                                   / "ResourceFile_Consumables_Item_Designations.csv")
    medicine_lookup = item_is_medicine.set_index("Item_Code")["is_medicine"]
    medicine_lookup.loc[2678] = True  # one missing item (Isoniazid/Rifapentine)

    # TLM districts
    common_districts = sorted(TLM_DISTRICTS)

    # ------------------------------------------------------------------------------------------
    # TLO data preparation helpers
    # ------------------------------------------------------------------------------------------
    def merge_info_from_mfl(_df):
        cols = ["Facility_ID", "District", "Region"]
        if "Facility_Level" not in _df.columns:
            cols.append("Facility_Level")
        return _df.merge(mfl[cols], on="Facility_ID", how="left")

    def restrict_to_tlm(_df):
        """Keep TLM facility levels and districts; central hospitals assigned by region."""
        _df = merge_info_from_mfl(_df)
        _df["Facility_Level"] = _df["Facility_Level"].astype(str)
        _df = _df.loc[~_df["Facility_Level"].isin(["0", "5"])].copy()
        is_zmh = _df["Facility_Level"] == "4"
        _df.loc[is_zmh, "District"] = "Central Hospitals (Southern)"   # ZMH
        _df.loc[is_zmh, "Region"] = "Southern"
        _df.loc[_df["Facility_ID"] == 128, "District"] = "Central Hospitals (Southern)"
        _df.loc[_df["Facility_ID"] == 129, "District"] = "Central Hospitals (Northern)"
        _df.loc[_df["Facility_ID"] == 130, "District"] = "Central Hospitals (Central)"
        _df["Facility_Level"] = _df["Facility_Level"].replace({"4": "3", "1b": "2"})
        return _df.loc[_df["District"].isin(common_districts)].copy()

    def filter_hsi(_df):
        _df = _df.loc[pd.to_datetime(_df["date"]).between(*TARGET_PERIOD)].copy()
        if DROP_DID_NOT_RUN_HSI:
            _df = _df.loc[_df["did_run"].astype(bool)]
        if not keep_blank_footprint:
            _df = _df.loc[_df["Number_By_Appt_Type_Code"].apply(len) > 0]
        return _df

    def add_loc_cat(_df):
        _df["loc_cat"] = pd.MultiIndex.from_frame(_df[["Event_Name", "TREATMENT_ID"]]).map(hsi_loc_cat_map)
        unmapped = set(_df.loc[_df["loc_cat"].isna(), "Event_Name"].unique())
        assert unmapped.issubset({"Inpatient_Care", "HSI_GenericNonEmergencyFirstAppt",
                                  "HSI_GenericEmergencyFirstAppt", "_BaseHSIGenericFirstAppt"}), unmapped
        return _df.dropna(subset=["loc_cat"])

    def extract_draw0(module, key, fn, do_scaling):
        res = extract_results(results_folder, module=module, key=key,
                              custom_generate_series=fn, do_scaling=do_scaling)
        return res.xs(0, axis=1, level="draw")

    # ------------------------------------------------------------------------------------------
    # Map HSIs to TLM service areas
    # ------------------------------------------------------------------------------------------
    def get_all_hsi_events(_df):
        _df = _df.loc[pd.to_datetime(_df["date"]).between(*TARGET_PERIOD), :]
        X = _df[["Event_Name", "Facility_Level", "Number_By_Appt_Type_Code", "TREATMENT_ID"]].copy()
        X["Appt_Type_Code"] = X["Number_By_Appt_Type_Code"].apply(lambda x: ",".join(sorted(map(str, x.keys()))))
        return (
            X.drop_duplicates(subset=["Event_Name", "Facility_Level", "Appt_Type_Code", "TREATMENT_ID"])
            .rename(columns={"Event_Name": "Event", "Facility_Level": "Facility level",
                             "Appt_Type_Code": "Appointment footprint", "TREATMENT_ID": "Treatment"})
            [["Event", "Facility level", "Appointment footprint", "Treatment"]]
            .drop_duplicates()
            .set_index(["Event", "Facility level", "Appointment footprint", "Treatment"])
            .assign(Value=1)["Value"]
        )

    sim_hsi = extract_results(results_folder, module="tlo.methods.healthsystem", key="HSI_Event",
                              custom_generate_series=get_all_hsi_events, do_scaling=False).reset_index()
    sim_hsi = sim_hsi[["Event", "Facility level", "Appointment footprint", "Treatment"]].copy()
    sim_hsi.columns = sim_hsi.columns.droplevel(1)

    web_hsi = pd.read_csv(path_to_tlm_folder / "hsi_web.csv")
    web_hsi["Appointment footprint"] = web_hsi["Appointment footprint"].str.replace(", ", ",", regex=False)
    conflicts = web_hsi.groupby(["Event", "Treatment"], dropna=False)["TLM service area"].nunique(dropna=False)
    assert conflicts[conflicts > 1].empty, f"Event-Treatment pairs with several service areas:\n{conflicts}"

    hsi_list_full = sim_hsi.merge(web_hsi[["Event", "Treatment", "TLM service area"]].drop_duplicates(),
                                  on=["Event", "Treatment"], how="outer", validate="many_to_one")
    fill = [
        (hsi_list_full["Event"] == "HSI_Malaria_rdt_community", "Outpatient - General"), # check
        (hsi_list_full["Event"] == "HSI_Hiv_SelfTest", "CD clinic"),  # check
        (hsi_list_full["Treatment"].isin({"Alri_Pneumonia_Treatment_Inpatient_Followup",
                                          "Alri_Pneumonia_Treatment_Inpatient"}), "Inpatient - General"),
        (hsi_list_full["Event"] == "HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare",
         "Inpatient - Maternity/Gynaecology"),
    ]
    for mask, area in fill:
        hsi_list_full.loc[mask & hsi_list_full["TLM service area"].isna(), "TLM service area"] = area

    hsi_loc_cat_map = hsi_list_full[["Event", "Treatment", "TLM service area"]].drop_duplicates()
    hsi_loc_cat_map["TLM service area"] = hsi_loc_cat_map["TLM service area"].replace(
        {"NCD/Other clinic": "Outpatient - General"})
    hsi_loc_cat_map = (hsi_loc_cat_map
                       .rename(columns={"Event": "Event_Name", "Treatment": "TREATMENT_ID",
                                        "TLM service area": "loc_cat"})
                       .set_index(["Event_Name", "TREATMENT_ID"])["loc_cat"])

    # ------------------------------------------------------------------------------------------
    # Custom series for TLO outputs (one value per run)
    # ------------------------------------------------------------------------------------------
    def get_patient_days_by_level(_df):
        """Sum over days and facilities of unique patients per facility-day, by level."""
        _df = filter_hsi(_df)
        if add_inpatient_days:
            _df = _df.loc[_df["Event_Name"] != "Inpatient_Care"]  # in-patient days are added separately
        daily = _df.groupby(["date", "Facility_ID"])["Person_ID"].nunique().rename("n").reset_index()
        daily = restrict_to_tlm(daily)
        return daily.groupby("Facility_Level")["n"].sum().reindex(LEVELS, fill_value=0)

    def get_staff_by_level(_df):
        """Staff in post by level (latest record up to the end of the target period), excl. DCSA."""
        _df = _df.loc[pd.to_datetime(_df["date"]) <= TARGET_PERIOD[1]]
        staff = (pd.Series(_df["GenericClinic"].iloc[-1], name="Staff_Count")
                 .rename_axis("key").reset_index())
        staff[["Facility_ID", "Officer_Type"]] = staff["key"].str.extract(r"FacilityID_(\d+)_Officer_(.*)")
        staff["Facility_ID"] = staff["Facility_ID"].astype(int)
        staff = restrict_to_tlm(staff.loc[staff["Officer_Type"] != "DCSA"])
        return staff.groupby("Facility_Level")["Staff_Count"].sum().reindex(LEVELS, fill_value=0)

    def get_staff_by_level_cadre(_df):
        """Staff in post by level and cadre group (as in get_staff_by_level), excl. DCSA."""
        _df = _df.loc[pd.to_datetime(_df["date"]) <= TARGET_PERIOD[1]]
        staff = (pd.Series(_df["GenericClinic"].iloc[-1], name="Staff_Count")
                 .rename_axis("key").reset_index())
        staff[["Facility_ID", "Officer_Type"]] = staff["key"].str.extract(r"FacilityID_(\d+)_Officer_(.*)")
        staff["Facility_ID"] = staff["Facility_ID"].astype(int)
        staff = restrict_to_tlm(staff.loc[staff["Officer_Type"] != "DCSA"])
        staff["cadre_grp"] = staff["Officer_Type"].map(CADRE_GROUPS).fillna("Other")
        return staff.groupby(["Facility_Level", "cadre_grp"])["Staff_Count"].sum()

    def get_patient_mix_counts(_df):
        """Patient counts by (category, subgroup, level); one record per person, facility,
        service area and day, consistent with the TLM daily data collection."""
        _df = add_loc_cat(restrict_to_tlm(filter_hsi(_df)))
        _df = _df.drop_duplicates(subset=["date", "Person_ID", "Facility_ID", "loc_cat"])  # check if appropriate
        labels = {
            "Sex": _df["Sex"].map(SEX_LABELS),
            "Age group": _df["Age_Range"].map(age_label),
            "Education": _df["Education"].map(lambda x: code_label(x, EDUC_LABELS)),
            "Wealth quintile": _df["Wealth"].map(lambda x: code_label(x, WEALTH_LABELS)),
            "Service area": _df["loc_cat"],
        }
        # education and wealth: adult patients only (TLM respondents may be companions)
        age_start = pd.to_numeric(_df["Age_Range"].astype(str).str.split("-").str[0].str.rstrip("+"),
                                  errors="coerce")
        adult = age_start >= ADULT_MIN_AGE
        for cat in ADULT_ONLY_CATS:
            labels[cat] = labels[cat].where(adult)

        parts = []
        for cat, lab in labels.items():
            c = (_df.assign(subgroup=lab).dropna(subset=["subgroup"])
                 .groupby(["subgroup", "Facility_Level"]).size())
            parts.append(pd.concat({cat: c}, names=["category"]))
        out = pd.concat(parts)
        out.index = out.index.set_names(["category", "subgroup", "Facility_Level"])
        return out

    def get_meds_access_sums(_df):
        """Per person, facility, service area and day with >= 1 medicine requested: counts and sums
        of the three access measures, by (service area, level)."""
        _df = _df.loc[pd.to_datetime(_df["date"]).between(*TARGET_PERIOD)].copy()
        _df = _df.rename(columns={"event_name": "Event_Name", "facility_id": "Facility_ID",
                                  "person_id": "Person_ID"})
        _df = add_loc_cat(restrict_to_tlm(_df))
        for col in ("Item_NotAvailable", "Item_Available"):
            _df[col] = _df[col].apply(lambda d: list(ast.literal_eval(d).items()))
        _df["Item_Requested"] = _df["Item_Available"] + _df["Item_NotAvailable"]

        missing = {code for items in _df["Item_Requested"] for code, _ in items} - set(medicine_lookup.index)
        assert not missing, f"Missing item codes in item_is_medicine: {missing}"

        _df = (_df.groupby(["date", "Person_ID", "Facility_ID", "Facility_Level", "loc_cat"])
               .agg(Item_Requested=("Item_Requested", lambda x: [i for lst in x for i in lst]),
                    Item_Available=("Item_Available", lambda x: [i for lst in x for i in lst]))
               .reset_index())

        def n_meds(items):
            return sum(bool(medicine_lookup.get(code, False)) for code, _ in items)

        _df["n_req"] = _df["Item_Requested"].apply(n_meds)
        _df["n_av"] = _df["Item_Available"].apply(n_meds)
        _df = _df.loc[_df["n_req"] > 0].copy()
        _df["complete"] = (_df["n_av"] == _df["n_req"]).astype(float)
        _df["proportional"] = _df["n_av"] / _df["n_req"]
        _df["any"] = (_df["n_av"] > 0).astype(float)
        _df["n"] = 1.0
        out = _df.groupby(["loc_cat", "Facility_Level"])[["n"] + TLO_MEDS_ACCESS_MEASURES].sum().stack()
        out.index = out.index.set_names(["loc_cat", "Facility_Level", "stat"])
        return out

    renormalised = set()   # groups where TLO lacked data at a level with TLM weight

    # ------------------------------------------------------------------------------------------
    # Allocation of TLO in-patient days (Inpatient_Care, logged at the regional referral hospital)
    # to TLM districts, levels and inpatient service areas, in proportion to admissions weighted by
    # (requested bed-days - 1); the service area is that of the admitting HSI
    # ------------------------------------------------------------------------------------------
    INPATIENT_AREAS = ["Inpatient - General", "Inpatient - Maternity/Gynaecology"]
    BED_DAYS_OVERRIDE = {
        ("HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare", "DeliveryCare_Comprehensive"): 1,
        ("HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare", "PostnatalCare_Comprehensive"): 1,
        ("HSI_Alri_Treatment", "Alri_Pneumonia_Treatment_Inpatient"): 7,
        ("HSI_Alri_Treatment", "Alri_Pneumonia_Treatment_Inpatient_Followup"): 7,
        ("HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare", "AntenatalCare_Inpatient"): 1,
    }  # (event, treatment); None = all treatments of the event

    # requested bed-days from hsi_web, e.g. "general_bed (15 days)" -> 15
    bd = (web_hsi.set_index(["Event", "Treatment"])["Bed days footprint"].fillna("").astype(str)
          .str.findall(r"\(([\d.]+) days?\)")
          .apply(lambda xs: sum(map(float, xs)) if isinstance(xs, list) and xs else np.nan).dropna())
    bd_trt, bd_evt = bd.groupby(level=[0, 1]).max().to_dict(), bd.groupby(level=0).max().to_dict()
    for (e, t), d in BED_DAYS_OVERRIDE.items():
        if t is None:
            bd_evt[e] = d
        else:
            bd_trt[(e, t)] = d

    def region_level(_df):
        """Region and harmonised level for all districts (as in restrict_to_tlm, without the district filter)."""
        _df = merge_info_from_mfl(_df)
        _df["Facility_Level"] = _df["Facility_Level"].astype(str)
        _df.loc[_df["Facility_Level"] == "4", "Region"] = "Southern"     # ZMH, as in restrict_to_tlm
        _df["Facility_Level"] = _df["Facility_Level"].replace({"4": "3", "1b": "2"})
        return _df.loc[~_df["Facility_Level"].isin(["0", "5"])]

    def get_inpatient_days_allocated(_df):
        """In-patient days allocated to TLM levels and inpatient service areas:
        days(l, a) = sum over regions r of days(r) x weighted admissions(TLM districts in r, l, a)
                                                    / weighted admissions(r)"""
        _df = filter_hsi(_df)
        # actual in-patient days by region
        ip = _df.loc[_df["Event_Name"] == "Inpatient_Care"].copy()
        ip["days"] = ip["Number_By_Appt_Type_Code"].apply(lambda d: d.get("InpatientDays", 0))
        days_r = region_level(ip).groupby("Region")["days"].sum()
        # admissions: HSIs that request bed-days (hsi_web.csv or BED_DAYS_OVERRIDE) or carry IPAdmission
        # (IPAdmission is missing when an HSI returns its own actual footprint, e.g. deliveries)
        keys = pd.MultiIndex.from_frame(_df[["Event_Name", "TREATMENT_ID"]])
        los_all = pd.Series(keys.map(lambda k: bd_trt.get(k, bd_evt.get(k[0], np.nan))),
                            index=_df.index, dtype=float)
        is_adm = (los_all.notna() | _df["Number_By_Appt_Type_Code"].apply(lambda d: "IPAdmission" in d)) \
                 & (_df["Event_Name"] != "Inpatient_Care")
        adm = _df.loc[is_adm].copy()
        los = los_all.loc[adm.index]
        # weight = expected in-patient days (requested bed-days - 1)
        adm["w"] = (los.fillna(los.mean()) - 1).clip(lower=0)
        # inpatient service area of the admitting HSI (any other area is counted as Inpatient - General)
        adm["area"] = pd.MultiIndex.from_frame(adm[["Event_Name", "TREATMENT_ID"]]).map(hsi_loc_cat_map)
        adm.loc[~adm["area"].isin(INPATIENT_AREAS), "area"] = "Inpatient - General"
        w_r = region_level(adm).groupby("Region")["w"].sum()                                    # all districts
        w_rla = restrict_to_tlm(adm).groupby(["Region", "Facility_Level", "area"])["w"].sum()   # TLM districts
        # allocate each region's days to TLM levels and areas, then sum over regions
        share = w_rla.div(w_r, level="Region").fillna(0)
        alloc = share.mul(days_r, level="Region").fillna(0)
        days_la = (alloc.groupby(level=["area", "Facility_Level"]).sum()
                   .reindex(pd.MultiIndex.from_product([INPATIENT_AREAS, LEVELS]), fill_value=0.0))
        days_la.index = days_la.index.set_names(["key1", "key2"])

        def pad(s):   # one-level series -> (key1, "") so all items share the same index depth
            return pd.Series(s.to_numpy(), index=pd.MultiIndex.from_arrays([s.index, [""] * len(s)],
                                                                           names=["key1", "key2"]))
        return pd.concat({
            "inpatient_days": days_la,
            "tlm_share_of_region": pad(share.groupby(level="Region").sum()),     # check: between 0 and 1
            "days_region": pad(days_r.astype(float)),
            "share_admissions_no_bed_days": pad(pd.Series({"all": float(los.isna().mean()) if len(los) else 0.0})),
        }, names=["item"])

    # ------------------------------------------------------------------------------------------
    # 1. Patient load
    # ------------------------------------------------------------------------------------------
    if "load" in run_measures:
        pat_days = extract_draw0("tlo.methods.healthsystem", "HSI_Event",
                                 get_patient_days_by_level, do_scaling=True).reindex(LEVELS).fillna(0.0)
        if add_inpatient_days:
            # sensitivity analysis (5): add in-patient days (both inpatient areas) to patient-days, per run
            ip_load = extract_draw0("tlo.methods.healthsystem", "HSI_Event",
                                    get_inpatient_days_allocated, do_scaling=True)
            ip_load = (ip_load.xs("inpatient_days", level="item").groupby(level="key2").sum()
                       .reindex(LEVELS).fillna(0.0))
            pat_days = pat_days + ip_load[pat_days.columns]
        staff = extract_draw0("tlo.methods.healthsystem.summary", "number_of_hcw_staff",
                              get_staff_by_level, do_scaling=False).reindex(LEVELS)
        # patients per staff on duty per day, by level and run
        load_by_level = pat_days.div(staff[pat_days.columns] * HCW_ON_DUTY_FRACTION * n_days)

        # check: TLO's own level composition = worker share = worker-day share, the implicit weights without reweighting
        tlo_wd = (staff[pat_days.columns] * HCW_ON_DUTY_FRACTION * n_days).mean(axis=1)
        pd.DataFrame({"tlo_worker_days": tlo_wd, "tlo_share": tlo_wd / tlo_wd.sum()}) \
            .rename_axis("level").round(2).to_csv(output_folder / "tlo_level_composition.csv")

        # TLO cadre mix by level (input for the Stata cadre weights, sensitivity analysis (4))
        staff_cadre = extract_draw0("tlo.methods.healthsystem.summary", "number_of_hcw_staff",
                                    get_staff_by_level_cadre, do_scaling=False)
        cadre_mix = staff_cadre.mean(axis=1).rename("n_staff").reset_index()
        cadre_mix = cadre_mix.loc[cadre_mix["Facility_Level"].isin(LEVELS)]
        cadre_mix["tlo_share"] = (cadre_mix["n_staff"]
                                  / cadre_mix.groupby("Facility_Level")["n_staff"].transform("sum"))
        cadre_mix.rename(columns={"Facility_Level": "fac_lvl"}).round(2).to_csv(
            output_folder / "tlo_cadre_mix_by_level.csv", index=False)

        def compare_load(tlm_file, weights_file):
            if reweight:
                w = read_level_weights(tlm_folder / weights_file)["Overall"]
                overall = load_by_level.mul(w.reindex(LEVELS).values, axis=0).sum() / w.sum()
            else:
                # TLO's own level composition: all patients / all staff on duty
                overall = pat_days.sum() / (staff[pat_days.columns].sum() * HCW_ON_DUTY_FRACTION * n_days)
            per_run = pd.concat([overall.to_frame("Overall").T,
                                 load_by_level.rename(index=lambda lvl: f"Level {lvl}")])
            tlo = summarise_runs(per_run).rename_axis("group").reset_index()
            tlm = read_tlm(tlm_folder / tlm_file)
            tlm = tlm.loc[tlm["group"].isin(["Overall"] + [f"Level {lvl}" for lvl in LEVELS])]
            return tlm[["group", "tlm_est", "tlm_lower", "tlm_upper"]].merge(tlo, on="group", how="left")

        cmp_load_tms = compare_load(tms_file, TLM_FILES["w_tms"])
        cmp_load_clinic = compare_load(TLM_FILES["load_clinic"], TLM_FILES["w_clinic"])
        pd.concat([cmp_load_clinic.assign(tlm_source="clinic records"),
                   cmp_load_tms.assign(tlm_source="time-and-motion")]).to_csv(
            output_folder / "comparison_patient_load.csv", index=False)
        plot_patient_load(cmp_load_tms, cmp_load_clinic, output_folder / "fig_patient_load.png")

    # ------------------------------------------------------------------------------------------
    # 2. Patient mix (overall, reweighted by level)
    # ------------------------------------------------------------------------------------------
    if "mix" in run_measures:
        mix_counts = extract_draw0("tlo.methods.healthsystem", "HSI_Event",
                                   get_patient_mix_counts, do_scaling=False).fillna(0)
        mix_counts.index = mix_counts.index.set_names(["category", "subgroup", "Facility_Level"])

        # check: TLO patients by subgroup and level, not reweighted (unscaled counts, mean across runs);
        # shares are within each category and level, and for all levels pooled (TLO's own level mix)
        pooled = mix_counts.groupby(level=["category", "subgroup"]).sum()
        pooled = pd.concat({"All (no reweighting)": pooled}, names=["Facility_Level"]) \
            .reorder_levels(["category", "subgroup", "Facility_Level"])
        counts = pd.concat([mix_counts, pooled])
        chk = counts.mean(axis=1).rename("n_mean").to_frame()
        chk["n_min_run"] = counts.min(axis=1)
        chk["share_pct"] = 100 * chk["n_mean"] / chk.groupby(
            level=["category", "Facility_Level"])["n_mean"].transform("sum")
        chk.sort_index().round(2).to_csv(output_folder / "tlo_patient_mix_by_level.csv")

        w_mix_groups = read_level_weights(tlm_folder / TLM_FILES["w_mix"])
        assert ADULT_GROUP in w_mix_groups, f"'{ADULT_GROUP}' weights missing from {TLM_FILES['w_mix']}"

        # sensitivity analysis (5): in-patient days allocated to levels and inpatient areas, per run
        if add_inpatient_days:
            ip_alloc = extract_draw0("tlo.methods.healthsystem", "HSI_Event",
                                     get_inpatient_days_allocated, do_scaling=False)
            # check two rows: share_admissions_no_bed_days should be about 0,
            # and each tlm_share_of_region should be between 0 and 1.
            ip_alloc.mean(axis=1).rename("mean_across_runs").to_csv(output_folder / "tlo_inpatient_days_allocation.csv")
            ip_days = ip_alloc.xs("inpatient_days", level="item")  # (area, level) x run

        mix_runs = {}
        for run in mix_counts.columns:
            props = []
            for cat, c_cat in mix_counts[run].groupby(level="category"):
                # level weights of the TLM patients compared (adults for education and wealth)
                w_mix = w_mix_groups[ADULT_GROUP if cat in ADULT_ONLY_CATS else "Overall"]
                tab = (c_cat.droplevel("category").unstack("Facility_Level")
                       .reindex(columns=LEVELS).fillna(0))                  # subgroup x level
                if add_inpatient_days and cat == "Service area":
                    # add this run's in-patient days to each inpatient area, by level
                    tab = tab.astype(float)
                    for area in INPATIENT_AREAS:
                        if area not in tab.index:
                            tab.loc[area] = 0.0
                        tab.loc[area] += ip_days[run].xs(area, level="key1").reindex(LEVELS).to_numpy()
                if reweight:
                    share = tab / tab.sum(axis=0)                              # shares within level
                    w = w_mix.where(tab.sum(axis=0) > 0, 0.0)
                    if not np.isclose(w.sum(), w_mix.sum()):
                        renormalised.add(("patient mix", cat))
                    prop = share.fillna(0).mul(w, axis=1).sum(axis=1) / w.sum()
                else:
                    prop = tab.sum(axis=1) / tab.to_numpy().sum()             # TLO's own level mix
                props.append(pd.concat({cat: prop}, names=["category"]))
            mix_runs[run] = pd.concat(props)
        mix_runs = pd.DataFrame(mix_runs) * 100
        mix_runs.index = mix_runs.index.set_names(["category", "subgroup"])
        tlo_mix = summarise_runs(mix_runs).reset_index()

        tlm_mix = read_tlm(tlm_folder / TLM_FILES["mix"])
        tlm_mix["category"] = tlm_mix["variable"].map(MIX_VARS)
        tlm_mix["subgroup"] = tlm_mix["label"].astype(str)
        for cat, labels in (("Education", EDUC_LABELS), ("Wealth quintile", WEALTH_LABELS)):
            is_cat = tlm_mix["category"] == cat
            tlm_mix.loc[is_cat, "subgroup"] = tlm_mix.loc[is_cat, "value"].astype(int).map(labels)

        cmp_mix = tlm_mix[["category", "subgroup", "tlm_est", "tlm_lower", "tlm_upper"]].merge(
            tlo_mix, on=["category", "subgroup"], how="outer", indicator=True)
        unmatched = cmp_mix.loc[cmp_mix["_merge"] != "both", ["category", "subgroup", "_merge"]]
        if not unmatched.empty:
            print("Patient mix subgroups found in only one source (check labels):\n", unmatched)
        cmp_mix = cmp_mix.drop(columns="_merge")
        cmp_mix.to_csv(output_folder / "comparison_patient_mix.csv", index=False)

        plot_patient_mix(cmp_mix, output_folder / "fig_patient_mix.png")

    # ------------------------------------------------------------------------------------------
    # 3. Access to prescribed medicines
    # ------------------------------------------------------------------------------------------
    if "meds" in run_measures:
        meds_sums = extract_draw0("tlo.methods.healthsystem", "Consumables",
                                  get_meds_access_sums, do_scaling=False).fillna(0)
        meds_sums.index = meds_sums.index.set_names(["loc_cat", "Facility_Level", "stat"])
        w_meds = read_level_weights(tlm_folder / TLM_FILES["w_meds_outpatient" if outpatient_only else "w_meds"])

        meds_runs = {}
        for run in meds_sums.columns:
            s = meds_sums[run].unstack("stat").reindex(columns=["n"] + TLO_MEDS_ACCESS_MEASURES).fillna(0)
            if outpatient_only:
                s = s.loc[s.index.get_level_values("loc_cat").isin(OUTPATIENT_AREAS)]
            rec = {}
            # overall (reweighted with overall weights) and by level (no reweighting)
            lvl = s.groupby(level="Facility_Level").sum().reindex(LEVELS)
            lvl_rate = lvl[TLO_MEDS_ACCESS_MEASURES].div(lvl["n"].where(lvl["n"] > 0), axis=0)
            for m in TLO_MEDS_ACCESS_MEASURES:
                if reweight:
                    v, flag = combine_levels(lvl_rate[m], w_meds["Overall"])
                    if flag:
                        renormalised.add(("medicine access", "Overall"))
                else:
                    v = lvl[m].sum() / lvl["n"].sum()                          # TLO's own level mix
                rec[("Overall", "Overall", m)] = v
                for lvl_name in LEVELS:
                    rec[("Facility level", f"Level {lvl_name}", m)] = lvl_rate.loc[lvl_name, m]
            # by service area (reweighted with service-area-specific weights)
            for area, s_area in s.groupby(level="loc_cat"):
                r = s_area.droplevel("loc_cat").reindex(LEVELS)
                rate = r[TLO_MEDS_ACCESS_MEASURES].div(r["n"].where(r["n"] > 0), axis=0)
                for m in TLO_MEDS_ACCESS_MEASURES:
                    if not reweight:
                        v = r[m].sum() / r["n"].sum() if r["n"].sum() > 0 else np.nan
                    elif area in w_meds:
                        v, flag = combine_levels(rate[m], w_meds[area])
                        if flag:
                            renormalised.add(("medicine access", area))
                    else:
                        v = np.nan
                    rec[("Service area", area, m)] = v
            meds_runs[run] = pd.Series(rec)
        meds_runs = pd.DataFrame(meds_runs) * 100
        meds_runs.index = meds_runs.index.set_names(["category", "subgroup", "measure"])

        tlo_meds = summarise_runs(meds_runs).unstack("measure")
        tlo_meds.columns = [col.replace("tlo_", f"tlo_{m}_") for col, m in tlo_meds.columns]
        tlo_meds = tlo_meds.reset_index()

        tlm_meds = read_tlm(tlm_folder / TLM_FILES["meds_outpatient" if outpatient_only else "meds"])
        tlm_meds["category"] = np.select(
            [tlm_meds["group"].eq("Overall"), tlm_meds["group"].str.startswith("Level ")],
            ["Overall", "Facility level"], default="Service area")
        tlm_meds["subgroup"] = tlm_meds["group"]

        cmp_meds = tlm_meds[["category", "subgroup", "tlm_est", "tlm_lower", "tlm_upper"]].merge(
            tlo_meds, on=["category", "subgroup"], how="outer", indicator=True)
        unmatched = cmp_meds.loc[cmp_meds["_merge"] != "both", ["category", "subgroup", "_merge"]]
        if not unmatched.empty:
            print("Medicine access groups found in only one source (check labels):\n", unmatched)
        cmp_meds = cmp_meds.drop(columns="_merge")
        cmp_meds.to_csv(output_folder / "comparison_medicine_access.csv", index=False)

        meds_idx = cmp_meds.set_index(["category", "subgroup"])
        areas = [a for a in SERVICE_AREA_ORDER if ("Service area", a) in meds_idx.index]
        areas += sorted(s for c, s in meds_idx.index if c == "Service area" and s not in areas)
        forest_plot(
            meds_idx,
            [(None, "Overall", ["Overall"]),
             ("Facility level", "Facility level", [f"Level {lvl}" for lvl in LEVELS]),
             ("Service area", "Service area", areas)],
            [
                dict(est="tlm_est", lo="tlm_lower", hi="tlm_upper", offset=-0.27, fmt="o", color=COLOUR_TLM_EXIT_SURVEY,
                     label="TLM patient exit survey: obtained prescribed medicines (95% CI, jackknife)"),
                dict(est="tlo_complete_mean", lo="tlo_complete_lower", hi="tlo_complete_upper",
                     offset=-0.07, fmt="<", color=COLOUR_TLO, label="TLO: all medicines available"),
                dict(est="tlo_proportional_mean", lo="tlo_proportional_lower", hi="tlo_proportional_upper",
                     offset=0.10, fmt="d", color=COLOUR_TLO, label="TLO: proportion of medicine types available"),
                dict(est="tlo_any_mean", lo="tlo_any_lower", hi="tlo_any_upper",
                     offset=0.27, fmt=">", color=COLOUR_TLO, label="TLO: any medicine available"),
            ],
            "Prescribed patients with access to medicines (%)",
            output_folder / "fig_medicine_access.png",
            xlim=(0, 100),
            band=dict(lo="tlo_complete_mean", hi="tlo_any_mean", y0=-0.14, y1=0.34, color=COLOUR_TLO),
            label_map=LEVEL_LABELS,
            width=8.5,
            legend_rows=[[0], [1, 2, 3]]
        )

    # ------------------------------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------------------------------
    if renormalised:
        print("TLO had no data at a level with TLM weight; weights renormalised for:")
        for item in sorted(renormalised):
            print("  ", item)
    print(f"Outputs saved to {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    parser.add_argument("--tlm-results-folder", type=Path, default=None,
                        help="Folder with the Stata output CSVs (default: the TLM_2024 resource folder)")
    parser.add_argument("--analysis", choices=list(ANALYSES), default="main",
                        help="main, no_reweighting, no_blank_footprint, outpatient_meds, cadre_weighted_tms, "
                             "or inpatient_days")
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,  # outputs/bshe@ic.ac.uk/scenario_patient_mix-2026-05-15T125453Z
        output_folder=Path("./outputs"),
        resourcefilepath=Path("./resources"),
        the_target_period=(Date(2024, 1, 1), Date(2024, 5, 31)),
        tlm_results_folder=args.tlm_results_folder,
        analysis=args.analysis,
    )
# outputs/bshe@ic.ac.uk/scenario_patient_mix-2026-05-15T125453Z
# --tlm-results-folder resources/healthsystem/human_resources/TLM_2024
# --analysis no_reweighting | no_blank_footprint | outpatient_meds | cadre_weighted_tms | inpatient_days for the sensitivity analyses
