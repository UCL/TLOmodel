import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INDICATOR  = "skilled_deliveries"
service    = "skilled_deliveries"
apply_cap  = False

DATA_DIR  = "/Users/rachelmurray-watson/Documents/Heat_data"
OUT_DIR   = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/"
CMIP6_DIR = "/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/NASA_GDDP_CMIP6_Split/"

MIN_OBS       = 24
WBGT_VARS     = ["wbgt_day", "wbgt_night"]
LAG_MONTHS    = [1, 2, 3, 4, 9]

min_year_historical = 2012
max_year_historical = 2025

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# STEP 0 — Load pre-built long panel + tidy
# ---------------------------------------------------------------------------
print("Loading regression panel...")
long = pd.read_csv(
    f"/Users/rachelmurray-watson/Documents/Heat_data/All_predictors_processed/regression_panel_{INDICATOR}.csv",
    parse_dates=["date"])
long = long.rename(columns={INDICATOR: "y"})

# Date-based masks
long.loc[long["date"].between("2020-04-01", "2021-12-01"), "y"] = np.nan
long.loc[(long["date"].between("2023-04-01", "2024-06-01")) &
         (long["facility"] == "Phalombe Health Centre"), "y"] = 0
long.loc[(long["date"].between("2023-03-01", "2024-03-01")) &
         (long["facility"] == "Thumbwe Health Centre"), "y"] = 0

long["year"]  = long["date"].dt.year
long["month"] = long["date"].dt.month
long = long[long["year"].between(min_year_historical, max_year_historical - 1)]

# Drop sparse facilities
obs_per_fac = long.dropna(subset=["y"] + WBGT_VARS).groupby("facility").size()
keep_facs   = obs_per_fac[obs_per_fac >= MIN_OBS].index
long        = long[long["facility"].isin(keep_facs)].copy()
print(f"Facilities after sparsity filter: {long['facility'].nunique()}")

# Numeric covariates
long["A109__Altitude"]   = pd.to_numeric(long["A109__Altitude"],   errors="coerce")
long["minimum_distance"] = pd.to_numeric(long["minimum_distance"], errors="coerce")
mean_alt = round(long["A109__Altitude"].mean())
med_dist = long["minimum_distance"].median()
long["A109__Altitude"]   = long["A109__Altitude"].fillna(mean_alt).clip(lower=0)
long["minimum_distance"] = long["minimum_distance"].fillna(med_dist)

# Optional cap
if apply_cap:
    long.loc[long["y"] > 4e3, "y"] = np.nan

# ---------------------------------------------------------------------------
# Build lag variables (within-facility, sorted by date)
# ---------------------------------------------------------------------------
long = long.sort_values(["facility", "date"]).reset_index(drop=True)

for var in WBGT_VARS:
    for lag in LAG_MONTHS:
        long[f"{var}_lag{lag}"] = long.groupby("facility")[var].shift(lag)

long["wbgt_day_sq"]      = long["wbgt_day"] ** 2
long["wbgt_day_cu"]      = long["wbgt_day"] ** 3
long["wbgt_night_sq"]    = long["wbgt_night"] ** 2
long["wbgt_night_cu"]    = long["wbgt_night"] ** 3
long["wbgt_day_x_night"] = long["wbgt_day"] * long["wbgt_night"]

lag_cols_month = [f"wbgt_day_lag{l}"   for l in LAG_MONTHS]
lag_cols_night = [f"wbgt_night_lag{l}" for l in LAG_MONTHS]

# ---------------------------------------------------------------------------
# Build NegBin sample — raw integer counts, drop rows with any NaN predictor
# ---------------------------------------------------------------------------
nb_cols = (["y", "wbgt_day", "wbgt_night", "year", "month", "facility",
             "A109__Altitude", "minimum_distance",
             "Zonename", "Dist", "Resid", "A105", "Ftype"]
           + lag_cols_month + lag_cols_night
           + ["wbgt_day_sq", "wbgt_day_cu",
              "wbgt_night_sq", "wbgt_night_cu",
              "wbgt_day_x_night"])

nb_data = long.dropna(subset=nb_cols).copy()
nb_data["year_c"] = nb_data["year"] - nb_data["year"].mean() # centre the year
nb_data["y_int"] = nb_data["y"].round().clip(lower=0).astype(int)

# Re-apply sparsity filter on this subset
obs_nb  = nb_data.groupby("facility").size()
nb_data = nb_data[nb_data["facility"].isin(
    obs_nb[obs_nb >= MIN_OBS].index)].copy()
print(f"NegBin sample: {len(nb_data)} obs, {nb_data['facility'].nunique()} facilities")

# ---------------------------------------------------------------------------
# One NB family; alpha estimated by profile likelihood (IRLS is stable with
# the facility dummies, unlike the discrete joint MLE). No Poisson.
# ---------------------------------------------------------------------------
def fit_negbin(formula, data, groups, bounds=(1e-3, 10.0)):
    """Fit NB2 by GLM/IRLS, choosing alpha to maximise the profile likelihood."""
    def neg_llf(log_a):
        a = np.exp(log_a)
        return -smf.glm(formula, data=data,
                        family=sm.families.NegativeBinomial(alpha=a)).fit().llf
    opt = minimize_scalar(neg_llf, bounds=np.log(bounds),
                          method="bounded", options={"xatol": 1e-2})
    alpha_hat = float(np.exp(opt.x))
    res = smf.glm(formula, data=data,
                  family=sm.families.NegativeBinomial(alpha=alpha_hat)
                  ).fit(cov_type="cluster", cov_kwds={"groups": groups})
    res.alpha_hat = alpha_hat
    return res

groups = nb_data["facility"]
lag_terms  = " + ".join(lag_cols_month + lag_cols_night)
poly_terms = ("wbgt_day_sq + wbgt_day_cu + wbgt_night_sq + wbgt_night_cu"
              " + wbgt_day_x_night")

f_base = "y_int ~ year_c + C(month) + C(facility)"
f_wx   = "y_int ~ wbgt_day + wbgt_night + year_c + C(month) + C(facility)"
f_wx   = "y_int ~ wbgt_day + year_c + C(month) + C(facility)"

f_ext  = (f"y_int ~ wbgt_day + wbgt_night + {poly_terms} + {lag_terms}"
          f" + year_c + C(month) + C(facility)")

print("\nFitting Step 1: baseline NB (no weather)...")
model_base   = fit_negbin(f_base, nb_data, groups)
print(f"  alpha_hat = {model_base.alpha_hat:.4f}")

print("Fitting Step 2: core WBGT NB...")
model_wx     = fit_negbin(f_wx, nb_data, groups)
print(f"  alpha_hat = {model_wx.alpha_hat:.4f}")

print("Fitting Step 3: extended WBGT NB (lags + polynomials)...")
model_wx_ext = fit_negbin(f_ext, nb_data, groups)
print(f"  alpha_hat = {model_wx_ext.alpha_hat:.4f}")
print(model_wx_ext.summary())
# alpha near 0 in any model => overdispersion is mild and Poisson would have sufficed.

# Confirm the three share the same estimation sample before any LR test
assert model_base.nobs == model_wx.nobs == model_wx_ext.nobs, "samples differ!"

def save_irr(model, path):
    ci = model.conf_int()
    pd.DataFrame({
        "coefficient_name": model.params.index,
        "IRR":           np.exp(model.params.values),
        "coefficients":  model.params.values,
        "CI_lower":      ci[0].values,   "CI_upper":     ci[1].values,
        "IRR_CI_lower":  np.exp(ci[0].values),
        "IRR_CI_upper":  np.exp(ci[1].values),
        "p_values":      model.pvalues.values,
    }).to_csv(path, index=False)

save_irr(model_base,   f"{OUT_DIR}results_negbin_baseline_{service}.csv")
save_irr(model_wx_ext, f"{OUT_DIR}results_negbin_weather_extended_{service}.csv")

# ---------------------------------------------------------------------------
# LR tests (all models: same sample, alpha estimated per model, so llf's compare)
# ---------------------------------------------------------------------------
print("\n--- Likelihood Ratio Tests ---")
LR_1 = -2 * (model_base.llf - model_wx.llf)
df_1 = len(model_wx.params) - len(model_base.params)
print(f"Baseline vs WBGT:          LR={LR_1:.2f}, df={df_1}, p={1-stats.chi2.cdf(LR_1, df_1):.4f}")

LR_2 = -2 * (model_wx.llf - model_wx_ext.llf)
df_2 = len(model_wx_ext.params) - len(model_wx.params)
print(f"WBGT vs WBGT + lags/poly:  LR={LR_2:.2f}, df={df_2}, p={1-stats.chi2.cdf(LR_2, df_2):.4f}")

# ---------------------------------------------------------------------------
# WBGT IRRs from the extended model
# ---------------------------------------------------------------------------
print(f"\n{'Variable':<30} {'IRR':>8} {'95% CI IRR':>24} {'p':>8}")
print("-" * 74)
for var in ["wbgt_day", "wbgt_night"] + lag_cols_month + lag_cols_night:
    if var in model_wx_ext.params.index:
        ci = model_wx_ext.conf_int().loc[var]
        print(f"{var:<30} {np.exp(model_wx_ext.params[var]):>8.4f} "
              f"[{np.exp(ci[0]):.4f}, {np.exp(ci[1]):.4f}]  {model_wx_ext.pvalues[var]:>8.4f}")
# ---------------------------------------------------------------------------
# LR tests — all on the same nb_data sample
# ---------------------------------------------------------------------------
print("\n--- Likelihood Ratio Tests ---")

# Test 1: does adding WBGT improve on baseline?
LR_1 = -2 * (model_base.llf - model_wx.llf)
df_1 = len(model_wx.params) - len(model_base.params)
p_1  = 1 - stats.chi2.cdf(LR_1, df_1)
print(f"Baseline vs WBGT:          LR={LR_1:.2f}, df={df_1}, p={p_1:.4f}")

# Test 2: does adding lags + polynomials improve on core WBGT?
LR_2 = -2 * (model_wx.llf - model_wx_ext.llf)
df_2 = len(model_wx_ext.params) - len(model_wx.params)
p_2  = 1 - stats.chi2.cdf(LR_2, df_2)
print(f"WBGT vs WBGT + lags/poly:  LR={LR_2:.2f}, df={df_2}, p={p_2:.4f}")

# ---------------------------------------------------------------------------
# Key results — IRR for WBGT terms from the extended model
# ---------------------------------------------------------------------------
print(f"\n{'Variable':<30} {'IRR':>8} {'95% CI IRR':>24} {'p':>8}")
print("-" * 74)
wbgt_terms = (["wbgt_day", "wbgt_night"]
              + lag_cols_month + lag_cols_night)
for var in wbgt_terms:
    if var in model_wx_ext.params.index:
        coef = model_wx_ext.params[var]
        ci   = model_wx_ext.conf_int().loc[var]
        p    = model_wx_ext.pvalues[var]
        print(f"{var:<30} {np.exp(coef):>8.4f} "
              f"[{np.exp(ci[0]):.4f}, {np.exp(ci[1]):.4f}]  {p:>8.4f}")

# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

nb_data["y_pred_base"] = model_base.fittedvalues
nb_data["y_pred_wx"]   = model_wx_ext.fittedvalues
nb_data["residuals"]   = nb_data["y_int"] - nb_data["y_pred_wx"]
nb_data["difference"]  = nb_data["y_pred_base"] - nb_data["y_pred_wx"]

g = (nb_data.groupby("date")[["y_int", "y_pred_base", "y_pred_wx",
                              "residuals", "difference"]]
     .mean().reset_index().sort_values("date"))

def _year_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for lab in ax.get_xticklabels():
        lab.set_rotation(45); lab.set_ha("right")

# Panels A & B ---------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].scatter(g["date"], g["y_int"],      color="#1C6E8C", alpha=0.7, label="Actual (mean)")
axs[0].scatter(g["date"], g["y_pred_base"],color="grey",    alpha=0.7, label="Baseline NB")
axs[0].scatter(g["date"], g["y_pred_wx"],  color="#9AC4F8", alpha=0.7, label="Weather NB")
axs[0].set_xlabel("Date"); axs[0].set_ylabel(f"Mean {service} visits")
axs[0].set_title(f"A: Actual vs Predicted {service} visits"); axs[0].legend(); _year_axis(axs[0])

axs[1].scatter(g["date"], g["residuals"], color="#823038", alpha=0.7, label="Residuals (mean)")
axs[1].axhline(0, color="black", linestyle="--")
axs[1].set_xlabel("Date"); axs[1].set_ylabel("Residuals")
axs[1].set_title("B: Residuals over time"); axs[1].legend(); _year_axis(axs[1])
plt.tight_layout(); plt.savefig(f"{OUT_DIR}{service}_negbin_model_fit.png"); plt.close()

# Panel C: baseline - weather difference (lollipop) --------------------------
fig, ax = plt.subplots(figsize=(9, 6))
pos = g["difference"] >= 0
ax.vlines(g.loc[pos,  "date"], 0, g.loc[pos,  "difference"], color="#1C6E8C")
ax.scatter(g.loc[pos, "date"],    g.loc[pos,  "difference"], color="#1C6E8C",
           label="More visits (lower WBGT)")
ax.vlines(g.loc[~pos, "date"], 0, g.loc[~pos, "difference"], color="#823038")
ax.scatter(g.loc[~pos,"date"],    g.loc[~pos, "difference"], color="#823038",
           label="Fewer visits (higher WBGT)")
ax.axhline(0, color="black", linestyle="--")
ax.axvspan(pd.Timestamp("2023-02-01"), pd.Timestamp("2023-03-01"),
           color="#B4E33D", alpha=0.4, label="Cyclone Freddy")
ax.set_xlabel("Date"); ax.set_ylabel(f"Difference: baseline − weather predicted {service}")
ax.legend(); _year_axis(ax)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}{service}_negbin_disruptions.png"); plt.close()

# Predictions out ------------------------------------------------------------
nb_data[[
    "year", "month", "facility", "date", "A109__Altitude", "Zonename", "Dist",
    "Resid", "A105", "Ftype", "wbgt_day", "wbgt_night", "y_int",
    "y_pred_base", "y_pred_wx", "residuals", "difference"
] + lag_cols_month + lag_cols_night].to_csv(
    f"{OUT_DIR}results_negbin_predictions_{service}.csv", index=False)

print(f"\nDone. Outputs written to {OUT_DIR}")

# ---------------------------------------------------------------------------
# Save full predictions
# ---------------------------------------------------------------------------
df_out = nb_data[[
    "year", "month", "facility", "date",
    "A109__Altitude", "Zonename", "Dist", "Resid", "A105", "Ftype",
    "wbgt_day", "wbgt_night", "y_int",
    "y_pred_base", "y_pred_wx", "residuals", "difference"
] + lag_cols_month + lag_cols_night].copy()

df_out.to_csv(
    f"{OUT_DIR}results_negbin_predictions_{service}.csv", index=False)

print(f"\nDone. Outputs written to {OUT_DIR}")
