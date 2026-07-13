"""
model_of_wbgt_dhis2.py

Historical fit of a monthly DHIS2 count indicator on WBGT (Negative Binomial,
facility + month fixed effects, clustered SEs, alpha by profile likelihood),
then forward projection under CMIP6.

Stages
------
  1. Fit baseline (no weather) and weather models on the ERA5 historical panel.
  2. "Difference in expected appointments" = baseline prediction − weather
     prediction, per facility-month (the two-model difference).
  3. Project: apply the SAME fitted models to future CMIP6 WBGT, centring with
     the HISTORICAL means (never re-fitting, never re-indexing by position), and
     take the same two-model difference per SSP/model.

Weather terms are parameterised (WEATHER_VARS_LEVEL + squares + lags) so the
identical construction is used for the historical fit and every future frame —
that is what keeps projection columns aligned by NAME via patsy.
"""

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
INDICATOR = "skilled_deliveries"
service   = "skilled_deliveries"
apply_cap = False

# Weather predictor(s). Each LEVEL var optionally gets a square and lags.
# Default is the 5-day extreme, which exists in BOTH the ERA5 historical panel
# and the CMIP6 projection files (so it is actually projectable). Add "wbgt_day"
# only if you also have a CMIP6 monthly-mean file to feed load_future_wbgt().
WEATHER_VARS_LEVEL = ["wbgt5x_day"]
USE_SQUARE = True
LAG_MONTHS = [1, 2, 3, 9]            # [] to disable lags

CENTER = True                        # center continuous predictors (see note)
                                     # Off = raw values; quick checks only —
                                     # with facility dummies + squares this is
                                     # what keeps the fit well-conditioned.

MIN_OBS = 24
min_year_historical = 2012
max_year_historical = 2025
LAST_HIST_YEAR = max_year_historical - 1     # year held here in projection

# Projection
PROJECT = True
PROJECT_HOLD_YEAR = True             # hold year fixed -> climate-only difference
SSP_SCENARIOS = ["ssp126", "ssp245", "ssp585"]
WBGT_MODELS   = ["lowest", "median", "highest"]   # model ids in the file names
min_year_projection = 2025
max_year_projection = 2071

DATA_DIR    = "/Users/rachelmurray-watson/Documents/Heat_data"
OUT_DIR     = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/"
PANEL_PATH  = (f"{DATA_DIR}/All_predictors_processed/"
               f"regression_panel_{INDICATOR}.csv")
INDICES_DIR = f"{DATA_DIR}/Thermofeel_WBGT/Indices/"     # CMIP6 extreme files

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared column construction (used for the historical fit AND every future frame)
# ---------------------------------------------------------------------------
def add_weather_columns(df, shifts, lag_months=LAG_MONTHS):
    """Add model-ready columns to a long (facility x date) frame:
    year_c, month, and for each LEVEL var its centred value {v}_c, optional
    square {v}_c_sq, and centred lags {v}_lag{l}_c. `shifts` holds the constant
    subtracted from each var (historical mean if CENTER, else 0) so future data
    is centred on the SAME origin as the fit. Returns (df, rhs_term_list)."""
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
    """NB2 by GLM/IRLS; alpha chosen by profile likelihood; clustered SEs."""
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


# ---------------------------------------------------------------------------
# STEP 0 — Load historical panel + tidy
# ---------------------------------------------------------------------------
print("Loading regression panel...")
long = pd.read_csv(PANEL_PATH, parse_dates=["date"])
long = long.rename(columns={INDICATOR: "y"})

for v in WEATHER_VARS_LEVEL:
    if v not in long.columns:
        raise KeyError(f"'{v}' not in panel — rerun the panel builder so the "
                       f"ERA5 predictor is written into {os.path.basename(PANEL_PATH)}")

# Date-based masks
long.loc[long["date"].between("2020-04-01", "2021-12-01"), "y"] = np.nan
long.loc[(long["date"].between("2023-04-01", "2024-06-01")) &
         (long["facility"] == "Phalombe Health Centre"), "y"] = 0
long.loc[(long["date"].between("2023-03-01", "2024-03-01")) &
         (long["facility"] == "Thumbwe Health Centre"), "y"] = 0

long["year"]  = long["date"].dt.year
long["month"] = long["date"].dt.month
long = long[long["year"].between(min_year_historical, max_year_historical - 1)]

# Sparse-facility filter (need the weather level vars present)
obs_per_fac = long.dropna(subset=["y"] + WEATHER_VARS_LEVEL).groupby("facility").size()
keep_facs   = obs_per_fac[obs_per_fac >= MIN_OBS].index
long        = long[long["facility"].isin(keep_facs)].copy()
print(f"Facilities after sparsity filter: {long['facility'].nunique()}")

if apply_cap:
    long.loc[long["y"] > 4e3, "y"] = np.nan

# --- Centring shifts (historical means; reused for projection) --------------
SHIFTS = {"year": long["year"].mean() if CENTER else 0.0}
for v in WEATHER_VARS_LEVEL:
    SHIFTS[v] = long[v].mean() if CENTER else 0.0

# Build model columns on the full grid (lags need the complete monthly series)
long, weather_rhs = add_weather_columns(long, SHIFTS)

# ---------------------------------------------------------------------------
# Build the NB estimation sample
# ---------------------------------------------------------------------------
nb_cols = ["y", "facility"] + weather_rhs
nb_data = long.dropna(subset=nb_cols).copy()
nb_data["y_int"] = nb_data["y"].round().clip(lower=0).astype(int)

obs_nb  = nb_data.groupby("facility").size()
nb_data = nb_data[nb_data["facility"].isin(obs_nb[obs_nb >= MIN_OBS].index)].copy()
FITTED_FACILITIES = set(nb_data["facility"].unique())
print(f"NB sample: {len(nb_data)} obs, {len(FITTED_FACILITIES)} facilities "
      f"(CENTER={CENTER})")

# ---------------------------------------------------------------------------
# STEP 1 — Fit baseline + weather; two-model difference
# ---------------------------------------------------------------------------
groups = nb_data["facility"]
FE = "C(month) + C(facility)"
f_base = f"y_int ~ year_c + {FE}"
f_wx   = f"y_int ~ {' + '.join(weather_rhs)} + {FE}"

print("\nFitting baseline NB (no weather)...")
model_base = fit_negbin(f_base, nb_data, groups)
print(f"  alpha_hat = {model_base.alpha_hat:.4f}")

print("Fitting weather NB...")
model_wx = fit_negbin(f_wx, nb_data, groups)
print(f"  alpha_hat = {model_wx.alpha_hat:.4f}")
print(model_wx.summary())

assert model_base.nobs == model_wx.nobs, "baseline and weather samples differ!"

# LR test (same sample, alpha per model)
LR = -2 * (model_base.llf - model_wx.llf)
df = len(model_wx.params) - len(model_base.params)
print(f"\nBaseline vs weather:  LR={LR:.2f}, df={df}, "
      f"p={1 - stats.chi2.cdf(LR, df):.4f}")

# Weather-term effect sizes (per unit AND per SD). NOTE: with squares + lags
# these individual IRRs are collinear — report the reconstructed marginal
# curve / cumulative-lag effect, not this table, in the paper.
print(f"\n{'Variable':<22} {'IRR/unit':>9} {'IRR/SD':>9} {'p':>8}")
print("-" * 52)
for term in weather_rhs:
    if term == "year_c" or term not in model_wx.params.index:
        continue
    coef = model_wx.params[term]
    sd = nb_data[term].std()
    print(f"{term:<22} {np.exp(coef):>9.4f} {np.exp(coef*sd):>9.4f} "
          f"{model_wx.pvalues[term]:>8.4f}")


def save_irr(model, path):
    ci = model.conf_int()
    pd.DataFrame({
        "coefficient_name": model.params.index,
        "IRR":          np.exp(model.params.values),
        "coefficients": model.params.values,
        "CI_lower":     ci[0].values, "CI_upper": ci[1].values,
        "IRR_CI_lower": np.exp(ci[0].values), "IRR_CI_upper": np.exp(ci[1].values),
        "p_values":     model.pvalues.values,
    }).to_csv(path, index=False)

save_irr(model_base, f"{OUT_DIR}results_negbin_baseline_{service}.csv")
save_irr(model_wx,   f"{OUT_DIR}results_negbin_weather_{service}.csv")

# Two-model difference on the historical sample
nb_data["y_pred_base"] = model_base.fittedvalues
nb_data["y_pred_wx"]   = model_wx.fittedvalues
nb_data["residuals"]   = nb_data["y_int"] - nb_data["y_pred_wx"]
nb_data["difference"]  = nb_data["y_pred_base"] - nb_data["y_pred_wx"]

# ---------------------------------------------------------------------------
# Historical diagnostic plots
# ---------------------------------------------------------------------------
def _year_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for lab in ax.get_xticklabels():
        lab.set_rotation(45); lab.set_ha("right")

g = (nb_data.groupby("date")[["y_int", "y_pred_base", "y_pred_wx",
                              "residuals", "difference"]]
     .mean().reset_index().sort_values("date"))

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].scatter(g["date"], g["y_int"],       color="#1C6E8C", alpha=0.7, label="Actual (mean)")
axs[0].scatter(g["date"], g["y_pred_base"], color="grey",    alpha=0.7, label="Baseline NB")
axs[0].scatter(g["date"], g["y_pred_wx"],   color="#9AC4F8", alpha=0.7, label="Weather NB")
axs[0].set_xlabel("Date"); axs[0].set_ylabel(f"Mean {service}")
axs[0].set_title(f"A: Actual vs Predicted {service}"); axs[0].legend(); _year_axis(axs[0])
axs[1].scatter(g["date"], g["residuals"], color="#823038", alpha=0.7, label="Residuals (mean)")
axs[1].axhline(0, color="black", linestyle="--")
axs[1].set_xlabel("Date"); axs[1].set_ylabel("Residuals")
axs[1].set_title("B: Residuals"); axs[1].legend(); _year_axis(axs[1])
plt.tight_layout(); plt.savefig(f"{OUT_DIR}{service}_negbin_model_fit.png"); plt.close()

fig, ax = plt.subplots(figsize=(9, 6))
pos = g["difference"] >= 0
ax.vlines(g.loc[pos,  "date"], 0, g.loc[pos,  "difference"], color="#1C6E8C")
ax.scatter(g.loc[pos, "date"],    g.loc[pos,  "difference"], color="#1C6E8C",
           label="More than weather predicts")
ax.vlines(g.loc[~pos, "date"], 0, g.loc[~pos, "difference"], color="#823038")
ax.scatter(g.loc[~pos,"date"],    g.loc[~pos, "difference"], color="#823038",
           label="Fewer than weather predicts")
ax.axhline(0, color="black", linestyle="--")
ax.axvspan(pd.Timestamp("2023-02-01"), pd.Timestamp("2023-03-01"),
           color="#B4E33D", alpha=0.4, label="Cyclone Freddy")
ax.set_xlabel("Date"); ax.set_ylabel(f"Baseline − weather predicted {service}")
ax.legend(); _year_axis(ax)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}{service}_negbin_disruptions.png"); plt.close()

# Historical predictions out
hist_out = (["year", "month", "facility", "date", "y_int",
             "y_pred_base", "y_pred_wx", "residuals", "difference"]
            + WEATHER_VARS_LEVEL)
nb_data[hist_out].to_csv(
    f"{OUT_DIR}results_negbin_predictions_{service}.csv", index=False)
print(f"\nHistorical outputs written to {OUT_DIR}")

# ---------------------------------------------------------------------------
# STEP 3 — Forward projection under CMIP6
# ---------------------------------------------------------------------------
def load_future_wbgt(scenario, model):
    """Return a long (facility x date) frame with each WEATHER_VARS_LEVEL column
    for this SSP/model. Default reads the CMIP6 extreme-index facility file from
    wbgt_extreme_indices.py. Point this at your own file(s) if named differently
    or if you need a CMIP6 monthly-MEAN column as well."""
    path = f"{INDICES_DIR}wbgt_extreme_indices_facility_{model}_{scenario}.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    if "facility_id" in df.columns:
        df = df.rename(columns={"facility_id": "facility"})
    missing = [v for v in WEATHER_VARS_LEVEL if v not in df.columns]
    if missing:
        raise KeyError(f"{missing} not in {os.path.basename(path)} "
                       f"(have {list(df.columns)}). Supply a file that carries "
                       "these projected predictors.")
    return df[["facility", "date"] + WEATHER_VARS_LEVEL]


def project(future_long):
    """Apply the fitted baseline + weather models to future WBGT and return the
    per-facility-month two-model difference. Centres with the HISTORICAL SHIFTS;
    restricted to facilities the models were fitted on."""
    fut = future_long[future_long["facility"].isin(FITTED_FACILITIES)].copy()
    if fut.empty:
        return fut
    fut = fut[(fut["date"].dt.year >= min_year_projection) &
              (fut["date"].dt.year < max_year_projection)]
    fut, _ = add_weather_columns(fut, SHIFTS)          # SAME construction + shifts
    if PROJECT_HOLD_YEAR:
        fut["year_c"] = LAST_HIST_YEAR - SHIFTS["year"]   # climate-only difference
    fut = fut.dropna(subset=weather_rhs).copy()
    fut["y_pred_base"] = model_base.predict(fut)
    fut["y_pred_wx"]   = model_wx.predict(fut)
    fut["difference"]  = fut["y_pred_base"] - fut["y_pred_wx"]
    return fut


if PROJECT:
    print("\n" + "=" * 60)
    print("FORWARD PROJECTION")
    print("=" * 60)
    for scenario in SSP_SCENARIOS:
        for model in WBGT_MODELS:
            try:
                fut_raw = load_future_wbgt(scenario, model)
            except FileNotFoundError:
                print(f"  {scenario}/{model}: file not found — skipping")
                continue
            proj = project(fut_raw)
            if proj.empty:
                print(f"  {scenario}/{model}: no projectable facilities — skipping")
                continue

            keep = (["year", "month", "facility", "date",
                     "y_pred_base", "y_pred_wx", "difference"] + WEATHER_VARS_LEVEL)
            proj[keep].to_csv(
                f"{OUT_DIR}projection_{scenario}_{model}_{service}.csv", index=False)

            gp = proj.groupby("date")["difference"].mean().reset_index().sort_values("date")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(gp["date"], gp["difference"], color="#823038")
            ax.axhline(0, color="black", linestyle="--")
            ax.set_title(f"Projected mean disruption — {scenario} / {model}")
            ax.set_xlabel("Date"); ax.set_ylabel(f"Baseline − weather predicted {service}")
            _year_axis(ax)
            plt.tight_layout()
            plt.savefig(f"{OUT_DIR}projection_{scenario}_{model}_{service}.png")
            plt.close()

            print(f"  {scenario}/{model}: {proj['facility'].nunique()} facilities, "
                  f"mean diff {proj['difference'].mean():.2f}  -> written")

print("\nDone.")
