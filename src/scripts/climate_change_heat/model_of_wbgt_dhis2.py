"""
loop_all_indicators_forest.py

Loops model_of_wbgt_dhis2.py logic over all COUNT indicators,
collects weather-term coefficients, and produces a forest plot.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# ✅ COUNT indicators only — rates/coverages excluded
COUNT_INDICATORS = [
    # From INDICATOR_IDS
    "fp_total_clients",
    "opd_attendance",
    "ipd_total_admissions",
    "vmmc_first_visits",
    # From DATA_ELEMENT_IDS
    "pnc_mother_checked_48h",
    "anc_new_attendees",
    "anc_first_trimester_starts",
    "bcg_under1",
    "penta3_under1",
    "measles1_under1",
    "fully_immunised_under1",
    "pnc_within_2wks",
    "pnc_first_visit_2wks",
    "live_births_total",
    "skilled_deliveries",
]

# Human-readable labels for the forest plot
INDICATOR_LABELS: dict[str, str] = {
    "fp_total_clients":          "FP Total Clients",
    "opd_attendance":            "OPD Attendance",
    "ipd_total_admissions":      "IPD Total Admissions",
    "vmmc_first_visits":         "VMMC First Visits",
    "pnc_mother_checked_48h":    "PNC Mother <48h",
    "anc_new_attendees":         "ANC New Attendees",
    "anc_first_trimester_starts":"ANC 1st Trimester Starts",
    "bcg_under1":                "BCG Under-1",
    "penta3_under1":             "Penta3 Under-1",
    "measles1_under1":           "Measles 1st Dose Under-1",
    "fully_immunised_under1":    "Fully Immunised Under-1",
    "pnc_within_2wks":           "PNC Within 2 Weeks",
    "pnc_first_visit_2wks":      "PNC First Visit <2 Weeks",
    "live_births_total":         "Live Births Total",
    "skilled_deliveries":        "Skilled Deliveries",
}

# Weather / model settings
WEATHER_VARS_LEVEL  = ["wbgt5x_day"]
USE_SQUARE          = True
LAG_MONTHS          = [1, 2, 3, 9]
CENTER              = True
MIN_OBS             = 10
min_year_historical = 2015
max_year_historical = 2025

apply_cap = False

DATA_DIR = "/Users/rachelmurray-watson/Documents/Heat_data"
OUT_DIR  = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers (identical to single-indicator script)
# ---------------------------------------------------------------------------
def add_weather_columns(df, shifts, lag_months=LAG_MONTHS):
    df = df.sort_values(["facility", "date"]).reset_index(drop=True)
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["year_c"] = df["year"] - shifts["year"]
    rhs = ["year_c"]
    for v in WEATHER_VARS_LEVEL:
        c = f"{v}_c"
        df[c] = df[v] - shifts[v]
        rhs.append(c)
        if USE_SQUARE:
            df[f"{c}_sq"] = df[c] ** 2
            rhs.append(f"{c}_sq")
        for lag in lag_months:
            lc = f"{v}_lag{lag}_c"
            df[lc] = df.groupby("facility")[v].shift(lag) - shifts[v]
            rhs.append(lc)
    return df, rhs


def fit_negbin(formula, data, groups, bounds=(1e-3, 10.0)):
    def neg_llf(log_a):
        a = np.exp(log_a)
        return -smf.glm(
            formula, data=data,
            family=sm.families.NegativeBinomial(alpha=a)
        ).fit().llf
    opt = minimize_scalar(
        neg_llf, bounds=np.log(bounds),
        method="bounded", options={"xatol": 1e-2}
    )
    alpha_hat = float(np.exp(opt.x))
    res = smf.glm(
        formula, data=data,
        family=sm.families.NegativeBinomial(alpha=alpha_hat)
    ).fit(cov_type="cluster", cov_kwds={"groups": groups})
    res.alpha_hat = alpha_hat
    return res


def run_indicator(indicator: str) -> pd.DataFrame | None:
    """
    Fit baseline + weather NB for one indicator.
    Returns a DataFrame of weather-term coefficients, or None on failure.
    """
    panel_path = (
        f"{DATA_DIR}/All_predictors_processed/"
        f"regression_panel_{indicator}.csv"
    )
    if not os.path.exists(panel_path):
        print(f"  [{indicator}] Panel file not found — skipping.")
        return None

    # --- Load & tidy -------------------------------------------------------
    long = pd.read_csv(panel_path, parse_dates=["date"])
    long = long.rename(columns={indicator: "y"})

    missing_vars = [v for v in WEATHER_VARS_LEVEL if v not in long.columns]
    if missing_vars:
        print(f"  [{indicator}] Missing weather vars {missing_vars} — skipping.")
        return None

    # Masks
    long.loc[long["date"].between("2020-04-01", "2021-12-01"), "y"] = np.nan
    long.loc[
        (long["date"].between("2023-04-01", "2024-06-01")) &
        (long["facility"] == "Phalombe Health Centre"), "y"
    ] = 0
    long.loc[
        (long["date"].between("2023-03-01", "2024-03-01")) &
        (long["facility"] == "Thumbwe Health Centre"), "y"
    ] = 0

    long["year"]  = long["date"].dt.year
    long["month"] = long["date"].dt.month
    long = long[long["year"].between(min_year_historical, max_year_historical - 1)]

    if apply_cap:
        long.loc[long["y"] > 4e3, "y"] = np.nan

    # Sparsity filter
    obs_per_fac = long.dropna(subset=["y"] + WEATHER_VARS_LEVEL).groupby("facility").size()
    keep_facs   = obs_per_fac[obs_per_fac >= MIN_OBS].index
    long        = long[long["facility"].isin(keep_facs)].copy()

    if long.empty or long["facility"].nunique() < 2:
        print(f"  [{indicator}] Too few facilities after filter — skipping.")
        return None

    # Centring shifts
    shifts = {"year": long["year"].mean() if CENTER else 0.0}
    for v in WEATHER_VARS_LEVEL:
        shifts[v] = long[v].mean() if CENTER else 0.0

    long, weather_rhs = add_weather_columns(long, shifts)

    # NB sample
    nb_cols  = ["y", "facility"] + weather_rhs
    nb_data  = long.dropna(subset=nb_cols).copy()
    nb_data["y_int"] = nb_data["y"].round().clip(lower=0).astype(int)
    obs_nb   = nb_data.groupby("facility").size()
    nb_data  = nb_data[nb_data["facility"].isin(obs_nb[obs_nb >= MIN_OBS].index)].copy()

    if nb_data.empty or nb_data["facility"].nunique() < 2:
        print(f"  [{indicator}] NB sample too small — skipping.")
        return None

    groups = nb_data["facility"]
    FE     = "C(month) + C(facility)"
    f_wx   = f"y_int ~ {' + '.join(weather_rhs)} + {FE}"

    try:
        model_wx = fit_negbin(f_wx, nb_data, groups)
    except Exception as e:
        print(f"  [{indicator}] Model failed: {e} — skipping.")
        return None

    # --- Extract weather-term coefficients ---------------------------------
    ci   = model_wx.conf_int()
    rows = []
    for term in weather_rhs:
        if term == "year_c" or term not in model_wx.params.index:
            continue
        rows.append({
            "indicator":  indicator,
            "label":      INDICATOR_LABELS.get(indicator, indicator),
            "term":       term,
            "coef":       model_wx.params[term],
            "ci_lo":      ci.loc[term, 0],
            "ci_hi":      ci.loc[term, 1],
            "irr":        np.exp(model_wx.params[term]),
            "irr_lo":     np.exp(ci.loc[term, 0]),
            "irr_hi":     np.exp(ci.loc[term, 1]),
            "pval":       model_wx.pvalues[term],
            "alpha_hat":  model_wx.alpha_hat,
            "n_obs":      int(model_wx.nobs),
            "n_fac":      nb_data["facility"].nunique(),
        })

    print(f"  [{indicator}] ✓  n={int(model_wx.nobs):,}, "
          f"facilities={nb_data['facility'].nunique()}, "
          f"alpha={model_wx.alpha_hat:.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
print("=" * 60)
print("Looping over all count indicators")
print("=" * 60)

all_results: list[pd.DataFrame] = []

for ind in COUNT_INDICATORS:
    print(f"\n→ {ind}")
    result = run_indicator(ind)
    if result is not None:
        all_results.append(result)

if not all_results:
    raise RuntimeError("No indicators fitted successfully — check panel paths.")

coef_df = pd.concat(all_results, ignore_index=True)
coef_df.to_csv(f"{OUT_DIR}all_indicators_weather_coefficients.csv", index=False)
print(f"\nCoefficients saved → {OUT_DIR}all_indicators_weather_coefficients.csv")

# ---------------------------------------------------------------------------
# FOREST PLOT — one panel per weather term
# ---------------------------------------------------------------------------
TERM_TITLES = {
    f"{WEATHER_VARS_LEVEL[0]}_c":           "WBGT 5-day Extreme (linear)",
    f"{WEATHER_VARS_LEVEL[0]}_c_sq":        "WBGT 5-day Extreme (quadratic)",
    **{
        f"{WEATHER_VARS_LEVEL[0]}_lag{l}_c": f"WBGT 5-day Extreme (lag {l}m)"
        for l in LAG_MONTHS
    },
}

terms_present = [t for t in TERM_TITLES if t in coef_df["term"].unique()]
n_panels      = len(terms_present)

fig, axes = plt.subplots(
    1, n_panels,
    figsize=(5.5 * n_panels, max(5, len(COUNT_INDICATORS) * 0.55 + 2)),
    sharey=True,
)
if n_panels == 1:
    axes = [axes]

for ax, term in zip(axes, terms_present):
    sub = (
        coef_df[coef_df["term"] == term]
        .sort_values("irr", ascending=True)
        .reset_index(drop=True)
    )

    y_pos  = np.arange(len(sub))
    colors = [
        "#823038" if p < 0.05 else "#888888"   # red = significant
        for p in sub["pval"]
    ]

    # CI lines
    for i, row in sub.iterrows():
        ax.plot(
            [row["irr_lo"], row["irr_hi"]],
            [i, i],
            color=colors[i], linewidth=1.4, zorder=1,
        )

    # Point estimates
    ax.scatter(sub["irr"], y_pos, color=colors, s=55, zorder=2)

    # Reference line
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.9)

    # Shade significant rows
    for i, row in sub.iterrows():
        if row["pval"] < 0.05:
            ax.axhspan(i - 0.4, i + 0.4, color="#f7e0e2", alpha=0.35, zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [INDICATOR_LABELS.get(s, s) for s in sub["indicator"]],
        fontsize=9,
    )
    ax.set_xlabel("IRR (95 % CI)", fontsize=10)
    ax.set_title(TERM_TITLES.get(term, term), fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.5)

# Shared legend
sig_patch  = mpatches.Patch(color="#823038", label="p < 0.05")
ns_patch   = mpatches.Patch(color="#888888", label="p ≥ 0.05")
fig.legend(
    handles=[sig_patch, ns_patch],
    loc="lower center", ncol=2,
    fontsize=9, frameon=False,
    bbox_to_anchor=(0.5, -0.02),
)

fig.suptitle(
    "Weather (WBGT) IRRs across health service indicators\n"
    "Negative Binomial, facility + month FE, clustered SEs",
    fontsize=12, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig(
    f"{OUT_DIR}forest_plot_all_indicators.png",
    dpi=180, bbox_inches="tight",
)
plt.close()
print(f"Forest plot saved → {OUT_DIR}forest_plot_all_indicators.png")
print("\nDone.")
