"""
loop_all_indicators_forest.py

For every count indicator, fit a negative binomial model where the effect of
heat (WBGT) is described by a smooth curve (a natural cubic spline) rather than
a straight line or a polynomial. Then, for each indicator, summarise that curve
as a single number: the cumulative IRR for going from a "cool" WBGT to a "hot"
WBGT. Finally, plot all indicators together in one forest plot.

  1. WBGT enters the model as a spline (smooth, bends where the data bend,
     goes straight at the extremes so projections don't explode).
  2. Report the
     whole-curve effect as one cumulative IRR between two WBGT values.
  3. Apply a
     Benjamini-Hochberg correction because we are testing ~15 indicators.
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
from scipy.optimize import minimize_scalar
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt


# ===========================================================================
# 1. SETTINGS  (everything you might want to change lives here)
# ===========================================================================

# The count indicators to loop over.
COUNT_INDICATORS = [
    #"fp_total_clients",
    #"opd_attendance",
    "ipd_total_admissions",
    #"vmmc_first_visits",
    #"pnc_mother_checked_48h",
    #"anc_new_attendees",
    #"anc_first_trimester_starts",
    #"bcg_under1",
    #"penta3_under1",
    #"measles1_under1",
    #"fully_immunised_under1",
    #"pnc_within_2wks",
    #"pnc_first_visit_2wks",
    #"live_births_total",
    #"skilled_deliveries",
]

# Nice names for the plot.
INDICATOR_LABELS = {
    #"fp_total_clients":           "FP Total Clients",
    #"opd_attendance":             "OPD Attendance",
    "ipd_total_admissions":       "IPD Total Admissions",
    #"vmmc_first_visits":          "VMMC First Visits",
    #"pnc_mother_checked_48h":     "PNC Mother <48h",
    #"anc_new_attendees":          "ANC New Attendees",
    #"anc_first_trimester_starts": "ANC 1st Trimester Starts",
    #"bcg_under1":                 "BCG Under-1",
    #"penta3_under1":              "Penta3 Under-1",
    #"measles1_under1":            "Measles 1st Dose Under-1",
    #"fully_immunised_under1":     "Fully Immunised Under-1",
    #"pnc_within_2wks":            "PNC Within 2 Weeks",
    #"pnc_first_visit_2wks":       "PNC First Visit <2 Weeks",
    #"live_births_total":          "Live Births Total",
    #"skilled_deliveries":         "Skilled Deliveries",
}

# The weather variable and how it enters the model.
WBGT_VAR    = "wbgt5x_day"    # the WBGT column in each panel file
SPLINE_DF   = 4              # flexibility of the WBGT curve (3-4 is standard)
LAG_MONTHS  = [1, 2, 3,4, 5 ,6, 7, 8, 9]   # extra WBGT lags, kept as simple linear terms

# The "cool -> hot" contrast we summarise the curve with.
# Two options; pick one by setting CONTRAST_MODE.
CONTRAST_MODE   = "percentile"   # "percentile" (data-driven) or "fixed" (degrees C)
CONTRAST_PCTL   = (10, 90)       # cool = 10th percentile, hot = 90th percentile
CONTRAST_FIXED  = (28.0, 32.0)   # Gohar moderate -> severe, if CONTRAST_MODE = "fixed"
INCLUDE_LAGS    = False           # True = sustained heat (curve + all lags shifted)

# Years and data-quality settings.
MIN_YEAR   = 2015
MAX_YEAR   = 2025      # uses years < MAX_YEAR
MIN_OBS    = 10        # a facility needs at least this many months to be kept

# Where the data lives and where results go.
DATA_DIR = "/Users/rachelmurray-watson/Documents/Heat_data"
OUT_DIR  = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/"
os.makedirs(OUT_DIR, exist_ok=True)


# ===========================================================================
# 2. FITTING A NEGATIVE BINOMIAL WITH A SEARCHED-FOR DISPERSION
# ===========================================================================
# statsmodels needs the NB dispersion (alpha) supplied. We find the alpha that
# best fits the data, then refit once with clustered standard errors.

def fit_negbin(formula, data, groups):
    """
    Fit a Negative Binomial GLM with:
      - a grid search for alpha (more stable than minimize_scalar on bad data)
      - start_params from a Poisson fit (warm start)
      - capped IRLS iterations and convergence tolerance
      - clustered SEs only on the final, converged fit
    """
    # ---- Step 0: Poisson warm-start to get sensible starting coefficients --
    try:
        poisson_fit = smf.glm(
            formula, data=data,
            family=sm.families.Poisson()
        ).fit(maxiter=100, disp=False)
        start_params = poisson_fit.params.values
    except Exception:
        start_params = None          # fall back to statsmodels default

    # ---- Step 1: Grid search for alpha -------------------------------------
    # A coarse log-spaced grid is much more robust than minimize_scalar when
    # the likelihood surface is flat or has numerical holes.
    best_alpha = 0.5                 # sensible default if search fails
    best_llf   = -np.inf


    for log_a in np.linspace(-1, 1, 10):   # alpha in [0.001, 100]
        alpha = float(np.exp(log_a))
        try:
            family = sm.families.NegativeBinomial(alpha=alpha)
            fit_kwargs = dict(maxiter=60, disp=False)
            if start_params is not None:
                fit_kwargs["start_params"] = start_params
            m = smf.glm(formula, data=data, family=family).fit(**fit_kwargs)
            if np.isfinite(m.llf) and m.llf > best_llf:
                best_llf   = m.llf
                best_alpha = alpha
        except Exception:
            continue

    # ---- Step 2: Final fit with best alpha + clustered SEs -----------------
    family = sm.families.NegativeBinomial(alpha=best_alpha)
    fit_kwargs = dict(
        cov_type  = "cluster",
        cov_kwds  = {"groups": groups},
        maxiter   = 200,
        disp      = False,
    )
    if start_params is not None:
        fit_kwargs["start_params"] = start_params

    result = smf.glm(formula, data=data, family=family).fit(**fit_kwargs)
    result.best_alpha = best_alpha
    return result

# ===========================================================================
# 3. TURNING THE FITTED SPLINE INTO ONE CUMULATIVE IRR
# ===========================================================================
# The spline is spread across several coefficients that mean nothing alone.
# To get a usable number we predict the log-rate at a "hot" WBGT and at a
# "cool" WBGT (holding everything else the same) and take the difference.
# exp(difference) is the cumulative IRR. The facility, month and year terms
# are identical in both rows, so they cancel and only the weather effect is left.

def cool_and_hot_wbgt(data):
    if CONTRAST_MODE == "fixed":
        return CONTRAST_FIXED
    cool = float(np.nanpercentile(data[WBGT_VAR], CONTRAST_PCTL[0]))
    hot  = float(np.nanpercentile(data[WBGT_VAR], CONTRAST_PCTL[1]))
    return cool, hot


def cumulative_irr(model, data, wbgt_mean, cool, hot):
    # Build two identical rows that differ only in WBGT (cool vs hot).
    example_month    = data["month"].iloc[0]
    example_facility = data["facility"].iloc[0]

    row = {
        WBGT_VAR:   [cool, hot],
        "year_c":   [0.0, 0.0],
        "month":    [example_month, example_month],
        "facility": [example_facility, example_facility],
    }
    for lag in LAG_MONTHS:
        lag_col = f"{WBGT_VAR}_lag{lag}_c"
        if INCLUDE_LAGS:
            # Sustained heat: the lagged WBGT shifts too (centred, so subtract mean).
            row[lag_col] = [cool - wbgt_mean, hot - wbgt_mean]
        else:
            # Only today's WBGT changes; lags held at their average (zero centred).
            row[lag_col] = [0.0, 0.0]

    # Re-create the model's design columns for these two rows.
    design_info = model.model.data.design_info
    X = patsy.build_design_matrices(
        [design_info], pd.DataFrame(row), return_type="dataframe"
    )[0]

    # Difference between the hot row and the cool row across all coefficients.
    contrast = (X.iloc[1] - X.iloc[0]).values
    beta     = model.params.values
    cov      = model.cov_params().values

    log_irr = float(contrast @ beta)
    se      = float(np.sqrt(contrast @ cov @ contrast))
    z       = log_irr / se

    return {
        "irr":    np.exp(log_irr),
        "irr_lo": np.exp(log_irr - 1.96 * se),
        "irr_hi": np.exp(log_irr + 1.96 * se),
        "pval":   float(2 * stats.norm.sf(abs(z))),
    }


# ===========================================================================
# 4. "DOES HEAT MATTER AT ALL?" — A CLUSTER-ROBUST WALD TEST
# ===========================================================================
# One joint test over every weather coefficient (spline pieces + lags). Uses
# the clustered covariance, so it is the cluster-robust block test we wanted
# instead of the old likelihood-ratio test.

def weather_block_pvalue(model):
    coef_names = list(model.params.index)
    weather_positions = [
        i for i, name in enumerate(coef_names) if WBGT_VAR in name
    ]

    # Restriction matrix: one row per weather coefficient, testing each = 0.
    restriction = np.zeros((len(weather_positions), len(coef_names)))
    for row_i, coef_i in enumerate(weather_positions):
        restriction[row_i, coef_i] = 1.0

    test = model.wald_test(restriction, scalar=True)
    return float(np.squeeze(test.pvalue))


# ===========================================================================
# 5. RUN ONE INDICATOR
# ===========================================================================

def run_indicator(indicator):
    panel_path = f"{DATA_DIR}/All_predictors_processed/regression_panel_{indicator}.csv"
    if not os.path.exists(panel_path):
        print(f"  [{indicator}] Panel file not found — skipping.")
        return None

    # --- Load the panel and rename the outcome to 'y' ----------------------
    data = pd.read_csv(panel_path, parse_dates=["date"])
    data = data.rename(columns={indicator: "y"})

    if WBGT_VAR not in data.columns:
        print(f"  [{indicator}] WBGT column '{WBGT_VAR}' missing — skipping.")
        return None

    # --- Data corrections --------------------------------------------------
    # COVID period: drop the outcome so it does not distort the fit.
    covid = data["date"].between("2020-04-01", "2021-12-01")
    data.loc[covid, "y"] = np.nan

    # Two facilities closed for a period: their true count is zero, not missing.
    phalombe_closed = (
        data["date"].between("2023-04-01", "2024-06-01")
        & (data["facility"] == "Phalombe Health Centre")
    )
    thumbwe_closed = (
        data["date"].between("2023-03-01", "2024-03-01")
        & (data["facility"] == "Thumbwe Health Centre")
    )
    data.loc[phalombe_closed, "y"] = 0
    data.loc[thumbwe_closed, "y"] = 0

    # --- Keep the study years ---------------------------------------------
    data["year"]  = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data = data[data["year"].between(MIN_YEAR, MAX_YEAR - 1)]

    # --- Centre the year, and remember the WBGT mean for the lags ----------
    year_mean = data["year"].mean()
    wbgt_mean = data[WBGT_VAR].mean()
    data["year_c"] = data["year"] - year_mean

    # --- Build the lagged WBGT columns (centred, linear) -------------------
    data = data.sort_values(["facility", "date"])
    lag_columns = []
    for lag in LAG_MONTHS:
        lag_col = f"{WBGT_VAR}_lag{lag}_c"
        data[lag_col] = data.groupby("facility")[WBGT_VAR].shift(lag) - wbgt_mean
        lag_columns.append(lag_col)

    # --- Drop rows with missing values needed by the model -----------------
    needed = ["y", "facility", "month", "year_c", WBGT_VAR] + lag_columns
    data = data.dropna(subset=needed).copy()
    data["y_int"] = data["y"].round().clip(lower=0).astype(int)

    # --- Keep only facilities with enough months ---------------------------
    months_per_facility = data.groupby("facility").size()
    good_facilities = months_per_facility[months_per_facility >= MIN_OBS].index
    data = data[data["facility"].isin(good_facilities)].copy()

    if data["facility"].nunique() < 2:
        print(f"  [{indicator}] Too few facilities after cleaning — skipping.")
        return None

    # --- Build the formula and fit -----------------------------------------
    # WBGT enters as a spline; lags enter linearly; facility and month are
    # absorbed as fixed effects; year_c is a linear time trend.
    spline = f"cr({WBGT_VAR}, df={SPLINE_DF})"
    formula = (
        "y_int ~ "
        + spline
        + " + " + " + ".join(lag_columns)
        + " + year_c + C(month) + C(facility)"
    )

    try:
        model = fit_negbin(formula, data, groups=data["facility"])
    except Exception as error:
        print(f"  [{indicator}] Model failed: {error} — skipping.")
        return None

    # --- Summarise the curve and test the weather block --------------------
    cool, hot = cool_and_hot_wbgt(data)
    effect    = cumulative_irr(model, data, wbgt_mean, cool, hot)
    block_p   = weather_block_pvalue(model)

    print(
        f"  [{indicator}] ok  "
        f"n={int(model.nobs):,}  facilities={data['facility'].nunique()}  "
        f"IRR({cool:.1f}->{hot:.1f})={effect['irr']:.3f}  "
        f"block p={block_p:.3g}"
    )

    return {
        "indicator":     indicator,
        "label":         INDICATOR_LABELS.get(indicator, indicator),
        "wbgt_cool":     cool,
        "wbgt_hot":      hot,
        "irr":           effect["irr"],
        "irr_lo":        effect["irr_lo"],
        "irr_hi":        effect["irr_hi"],
        "pval_contrast": effect["pval"],
        "pval_block":    block_p,
        "alpha":         model.best_alpha,
        "n_obs":         int(model.nobs),
        "n_facilities":  data["facility"].nunique(),
    }


# ===========================================================================
# 6. LOOP OVER ALL INDICATORS
# ===========================================================================

print("=" * 60)
print("Fitting spline heat model for every count indicator")
print("=" * 60)

results = []
for indicator in COUNT_INDICATORS:
    print(f"\n-> {indicator}")
    one_result = run_indicator(indicator)
    if one_result is not None:
        results.append(one_result)

if not results:
    raise RuntimeError("No indicators fitted successfully — check panel paths.")

results = pd.DataFrame(results)


# ===========================================================================
# 7. BENJAMINI-HOCHBERG CORRECTION
# ===========================================================================
# We ran ~15 weather-block tests, so some will look significant by chance.
# BH controls the false discovery rate across all of them at once.

results["pval_block_bh"] = multipletests(results["pval_block"], method="fdr_bh")[1]
results["significant"]   = results["pval_block_bh"] < 0.05

results = results.sort_values("irr").reset_index(drop=True)
results.to_csv(f"{OUT_DIR}all_indicators_cumulative_irr.csv", index=False)
print(f"\nResults saved -> {OUT_DIR}all_indicators_cumulative_irr.csv")


# ===========================================================================
# 8. FOREST PLOT — ONE CUMULATIVE IRR PER INDICATOR
# ===========================================================================

RED  = "#823038"   # significant after BH
GREY = "#888888"   # not significant

y_positions = np.arange(len(results))
bar_colors  = [RED if sig else GREY for sig in results["significant"]]

fig, ax = plt.subplots(figsize=(7.5, max(4, len(results) * 0.5 + 1.5)))

# Confidence interval line + point estimate for each indicator.
for i in range(len(results)):
    row = results.iloc[i]
    ax.plot([row["irr_lo"], row["irr_hi"]], [i, i],
            color=bar_colors[i], linewidth=1.6, zorder=1)
    if row["significant"]:
        ax.axhspan(i - 0.4, i + 0.4, color="#f7e0e2", alpha=0.35, zorder=0)

ax.scatter(results["irr"], y_positions, color=bar_colors, s=60, zorder=2)

# Reference line at IRR = 1 (no effect).
ax.axvline(1.0, color="black", linestyle="--", linewidth=0.9)

ax.set_yticks(y_positions)
ax.set_yticklabels(results["label"], fontsize=9)

if CONTRAST_MODE == "percentile":
    contrast_text = f"{CONTRAST_PCTL[0]}th to {CONTRAST_PCTL[1]}th percentile WBGT"
else:
    contrast_text = f"{CONTRAST_FIXED[0]:.0f} to {CONTRAST_FIXED[1]:.0f}C WBGT"

ax.set_xlabel(f"Cumulative IRR for {contrast_text} (95% CI)", fontsize=10)
ax.set_title(
    "Cumulative heat effect across health service indicators\n"
    "Negative binomial spline model, facility + month fixed effects,\n"
    "cluster-robust CIs; red = significant weather block after BH correction",
    fontsize=11, fontweight="bold",
)
ax.grid(axis="x", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}forest_plot_cumulative_irr.png", dpi=180, bbox_inches="tight")
plt.close()

print(f"Forest plot saved -> {OUT_DIR}forest_plot_cumulative_irr.png")
print("\nDone.")
