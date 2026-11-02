"""
Run the HPV modules
 """

import datetime
import pickle
import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file, extract_results
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    measles,
    simplified_births,
    symptommanager,
    hpv,
    hiv,
    tb
)

results_folder = Path("./outputs")

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = './resources'

# %% Run the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2040, 1, 1)
popsize = 20000


# set up the log config
log_config = {
    "filename": "test_runs",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.hpv": logging.INFO,
    },
}

# Register the appropriate modules
# need to call epi before tb to get bcg vax
# seed = random.randint(0, 50000)
seed = 12345  # set seed for reproducibility

# HPV model labels
AGE_LABELS = ["15_19", "20_24", "25_34", "35_44", "45_54", "55plus"]
HPV_GROUPS = ["hr1", "hr2", "hr3", "hr4", "hr5", "hr6"]
SEXES = ["M", "F"]

# 1. Run simulation
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config,
                 show_progress_bar=True, resourcefilepath=resourcefilepath)
sim.register(
    demography.Demography(),
    simplified_births.SimplifiedBirths(),
    enhanced_lifestyle.Lifestyle(),
    healthsystem.HealthSystem(service_availability=["*"],  # all treatment allowed
        mode_appt_constraints=1,  # mode of constraints to do with officer numbers and time
        cons_availability="default",  # mode for consumable constraints (if ignored, all consumables available)
        ignore_priority=False,  # do not use the priority information in HSI event to schedule
        capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
        use_funded_or_actual_staffing="actual",  # actual: use numbers/distribution of staff available currently
        disable=False,  # disables the healthsystem (no constraints and no logging) and every HSI runs
        disable_and_reject_all=False,  # disable healthsystem and no HSI runs
    ),
    symptommanager.SymptomManager(),
    healthseekingbehaviour.HealthSeekingBehaviour(),
    healthburden.HealthBurden(),
    epi.Epi(),
    hiv.Hiv(),
    measles.Measles(),
    tb.Tb(),
    hpv.HPV(),
)

# # set the scenario
#sim.modules["HPV"].parameters["r_hpv"] = 0.01
#sim.modules["HPV"].parameters["r_hpv_clear"] = 0.6

# # Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)


# 2. Parse and save results
output = parse_log_file(sim.log_filepath)

# save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
with open(outputpath / "default_run.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dict(output), f, pickle.HIGHEST_PROTOCOL)

# load the results
with open(outputpath / "default_run.pickle", "rb") as f:
    output = pickle.load(f)

#Show the results
hpv_outputs = output["tlo.methods.hpv"]["summary"]
print(hpv_outputs)
# proportion_infected = extract_results(
#     results_folder,
#     module="tlo.methods.hpv",
#     key="summary",
#     column="PropInf",
#     do_scaling=False,
# )
#
# number_infected = extract_results(
#     results_folder,
#     module="tlo.methods.hpv",
#     key="summary",
#     column="TotalInf",
#     do_scaling=True,
# )

hpv_outputs = output["tlo.methods.hpv"]["summary"]
hpv_df = pd.DataFrame(hpv_outputs)

print(hpv_df)
print(hpv_df.columns)

hpv_df["Date"] = pd.to_datetime(hpv_df["date"])
hpv_df = hpv_df.sort_values("Date").reset_index(drop=True)

# 4. Helper functions
def compute_group_prev_by_sex(
    df: pd.DataFrame,
    hpv_group: str,
    sex: str,
    age_labels: list[str],
) -> pd.Series:

    inf_cols = [f"{hpv_group}_{sex}_{age}_Inf" for age in age_labels]
    n_cols = [f"Any_{sex}_{age}_N" for age in age_labels]

    missing_inf = [c for c in inf_cols if c not in df.columns]
    missing_n = [c for c in n_cols if c not in df.columns]

    if missing_inf or missing_n:
        print(f"\nCannot compute {hpv_group}_{sex}_TotalPrev.")
        if missing_inf:
            print("Missing infection columns:", missing_inf)
        if missing_n:
            print("Missing denominator columns:", missing_n)

        return pd.Series([float("nan")] * len(df), index=df.index)

    total_inf = df[inf_cols].sum(axis=1)
    total_n = df[n_cols].sum(axis=1)

    return total_inf / total_n.replace(0, pd.NA)

def compute_group_prev_overall(
    df: pd.DataFrame,
    hpv_group: str,
    sexes: list[str],
    age_labels: list[str],
) -> pd.Series:

    inf_cols = []
    n_cols = []

    for sex in sexes:
        inf_cols.extend([f"{hpv_group}_{sex}_{age}_Inf" for age in age_labels])
        n_cols.extend([f"Any_{sex}_{age}_N" for age in age_labels])

    missing_inf = [c for c in inf_cols if c not in df.columns]
    missing_n = [c for c in n_cols if c not in df.columns]

    if missing_inf or missing_n:
        print(f"\nCannot compute {hpv_group}_TotalPrev.")
        if missing_inf:
            print("Missing infection columns:", missing_inf)
        if missing_n:
            print("Missing denominator columns:", missing_n)

        return pd.Series([float("nan")] * len(df), index=df.index)

    total_inf = df[inf_cols].sum(axis=1)
    total_n = df[n_cols].sum(axis=1)

    return total_inf / total_n.replace(0, pd.NA)

for sex in SEXES:
    for group in HPV_GROUPS:
        hpv_df[f"{group}_{sex}_TotalPrev"] = compute_group_prev_by_sex(
            hpv_df,
            hpv_group=group,
            sex=sex,
            age_labels=AGE_LABELS,
        )

for group in HPV_GROUPS:
    hpv_df[f"{group}_TotalPrev"] = compute_group_prev_overall(
        hpv_df,
        hpv_group=group,
        sexes=SEXES,
        age_labels=AGE_LABELS,
    )

def compute_coinfection_prev(df: pd.DataFrame, max_groups: int) -> pd.Series:
    """
    Coinfection prevalence = proportion of eligible individuals
    infected with 2 or more HPV groups.
    """
    multi_cols = [
        f"InfGroup{n}"
        for n in range(2, max_groups + 1)
        if f"InfGroup{n}" in df.columns
    ]

    if "EligibleN" not in df.columns or len(multi_cols) == 0:
        print("Cannot compute CoinfectionPrev: missing EligibleN or InfGroup columns.")
        return pd.Series([float("nan")] * len(df), index=df.index)

    multi_n = df[multi_cols].sum(axis=1)
    return multi_n / df["EligibleN"].replace(0, pd.NA)

hpv_df["CoinfectionPrev"] = compute_coinfection_prev(
    hpv_df,
    max_groups=len(HPV_GROUPS)
)

def compute_hiv_positive_total_prev(df: pd.DataFrame) -> pd.Series:
    hiv_pos_n_cols = [
        "Any_HIVpos_unknown_N",
        "Any_HIVpos_noART_N",
        "Any_HIVpos_unsupp_N",
        "Any_HIVpos_supp_N",
    ]

    hiv_pos_inf_cols = [
        "Any_HIVpos_unknown_Inf",
        "Any_HIVpos_noART_Inf",
        "Any_HIVpos_unsupp_Inf",
        "Any_HIVpos_supp_Inf",
    ]

    if not all(col in df.columns for col in hiv_pos_n_cols + hiv_pos_inf_cols):
        print("Cannot compute Any_HIVpos_Prev: missing HIV-positive columns.")
        return pd.Series([float("nan")] * len(df), index=df.index)

    total_n = df[hiv_pos_n_cols].sum(axis=1)
    total_inf = df[hiv_pos_inf_cols].sum(axis=1)

    return total_inf / total_n.replace(0, pd.NA)

hpv_df["Any_HIVpos_Prev"] = compute_hiv_positive_total_prev(hpv_df)

def compute_binomial_ci(row: pd.Series, z: float = 1.96):

    if pd.isna(row.get("positive")) or pd.isna(row.get("denominator")):
        return None, None

    n = row["denominator"]
    x = row["positive"]

    if n is None or n == 0 or pd.isna(n):
        return None, None

    p = x / n
    se = (p * (1 - p) / n) ** 0.5
    lower = max(0.0, p - z * se)
    upper = min(1.0, p + z * se)
    return lower, upper


def add_observed_points(ax, observed_df: pd.DataFrame, model_color_map: dict | None = None):

    if observed_df is None or len(observed_df) == 0:
        return

    for _, row in observed_df.iterrows():
        start = row["start_date"]
        end = row["end_date"]
        mid = start + (end - start) / 2
        y = row["prevalence"]

        mapped_model = row.get("mapped_model", None)

        if (
            model_color_map is not None
            and isinstance(mapped_model, str)
            and mapped_model in model_color_map
        ):
            color = model_color_map[mapped_model]
        else:
            color = row.get("color", "crimson")

        marker = row.get("marker", "s")
        label = row["label"]

        # horizontal line for study period
        ax.hlines(
            y=y,
            xmin=start,
            xmax=end,
            colors=color,
            linestyles="--",
            linewidth=1.6,
            alpha=0.85,
        )

        # central point
        ax.scatter(
            mid,
            y,
            color=color,
            marker=marker,
            s=70,
            label=label,
            zorder=5,
        )

        # optional vertical CI
        lower, upper = compute_binomial_ci(row)
        if lower is not None and upper is not None:
            ax.vlines(
                mid,
                ymin=lower,
                ymax=upper,
                colors=color,
                linewidth=1.2,
                alpha=0.9,
            )

def plot_model_series(
    ax,
    df: pd.DataFrame,
    col: str,
    label: str,
    model_color_map: dict,
    marker: str = "o",
    **kwargs,
):

    if col not in df.columns:
        print(f"Skipping {label}: missing column {col}")
        return None

    line, = ax.plot(
        df["Date"],
        df[col],
        marker=marker,
        label=label,
        **kwargs,
    )

    model_color_map[col] = line.get_color()
    return line

def deduplicate_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=9)

observed_rows = [
    # -----------------------------------------------------------------
    # HR-HPV
    # -----------------------------------------------------------------
    {
        "type": "HR-HPV",
        "population": "25-59 WLWH",
        "sex": "F",
        "hiv_status": "HIVpos",
        "age_band": "25-59",
        "start_date": pd.Timestamp("2011-11-01"),
        "end_date": pd.Timestamp("2012-04-30"),
        "positive": 114,
        "denominator": 294,
        "prevalence": 114 / 294,
        "source": "Reddy et al., 2015",
        "label": "Reddy 2015 (WLWH 25-59)",
        "color": "crimson",
        "marker": "s",
        "mapped_model": "Calib_Any_F_25_59_HIVpos_Prev",
        "use_for_plot": True,
    },
    {
        "type": "HR-HPV",
        "population": "General female",
        "sex": "F",
        "hiv_status": "All",
        "age_band": "20-60",
        "start_date": pd.Timestamp("2014-01-01"),
        "end_date": pd.Timestamp("2015-04-30"),
        "positive": 149,
        "denominator": 750,
        "prevalence": 149 / 750,
        "source": "Cubie et al., 2017",
        "label": "Cubie 2017 (General female)",
        "color": "crimson",
        "marker": "s",
        "mapped_model": "F_Prev",
        "use_for_plot": True,
    },
    {
        "type": "HR-HPV",
        "population": "25-49 WLWH",
        "sex": "F",
        "hiv_status": "HIVpos",
        "age_band": "25-49",
        "start_date": pd.Timestamp("2019-09-01"),
        "end_date": pd.Timestamp("2020-04-30"),
        "positive": 957,
        "denominator": 2203,
        "prevalence": 957 / 2203,
        "source": "Joseph et al., 2023",
        "label": "Joseph 2023 (WLWH 25-49)",
        "color": "firebrick",
        "marker": "o",
        "mapped_model": "Calib_Any_F_25_59_HIVpos_Prev",
        "use_for_plot": True,
    },
    {
        "type": "HR-HPV",
        "population": "WLWH 25-50",
        "sex": "F",
        "hiv_status": "HIVpos",
        "age_band": "25-50",
        "start_date": pd.Timestamp("2020-06-01"),
        "end_date": pd.Timestamp("2022-02-28"),
        "positive": 295,
        "denominator": 625,
        "prevalence": 295 / 625,
        "source": "Lameck Chinula, 2026",
        "label": "Chinula 2026 (WLWH 25-50)",
        "color": "orangered",
        "marker": "D",
        "mapped_model": "Calib_Any_F_25_59_HIVpos_Prev",
        "use_for_plot": True,
    },
    {
        "type": "HR-HPV",
        "population": "Non-WLWH 25-50",
        "sex": "F",
        "hiv_status": "HIVneg",
        "age_band": "25-50",
        "start_date": pd.Timestamp("2020-06-01"),
        "end_date": pd.Timestamp("2022-02-28"),
        "positive": 181,
        "denominator": 625,
        "prevalence": 181 / 625,
        "source": "Lameck Chinula, 2026",
        "label": "Chinula 2026 (Non-WLWH 25-50)",
        "color": "navy",
        "marker": "D",
        "mapped_model": "Calib_Any_F_25_59_HIVneg_Prev",
        "use_for_plot": True,
    },

    # -----------------------------------------------------------------
    # HPV16 -> hr1
    # -----------------------------------------------------------------
    {
        "type": "HPV16",
        "population": "General",
        "sex": "All",
        "hiv_status": "All",
        "age_band": "All",
        "start_date": pd.Timestamp("2014-01-01"),
        "end_date": pd.Timestamp("2015-04-30"),
        "positive": 36,
        "denominator": 750,
        "prevalence": 36 / 750,
        "source": "Cubie et al., 2017",
        "label": "Cubie 2017 HPV16 (General)",
        "color": "crimson",
        "marker": "s",
        "mapped_model": "hr1_F_TotalPrev",
        "use_for_plot": True,
    },
    {
        "type": "HPV16",
        "population": "25-49 WLWH",
        "sex": "F",
        "hiv_status": "HIVpos",
        "age_band": "25-49",
        "start_date": pd.Timestamp("2019-09-01"),
        "end_date": pd.Timestamp("2020-04-30"),
        "positive": 245,
        "denominator": 2203,
        "prevalence": 245 / 2203,
        "source": "Joseph et al., 2023",
        "label": "Joseph 2023 HPV16 (WLWH 25-49)",
        "color": "firebrick",
        "marker": "o",
        "mapped_model": "Calib_hr1_F_25_59_HIVpos_Prev",
        "use_for_plot": True,
    },
    {
        "type": "HPV16",
        "population": "WLWH 25-50",
        "sex": "F",
        "hiv_status": "HIVpos",
        "age_band": "25-50",
        "start_date": pd.Timestamp("2020-06-01"),
        "end_date": pd.Timestamp("2022-02-28"),
        "positive": 63,
        "denominator": 625,
        "prevalence": 63 / 625,
        "source": "Lameck Chinula, 2026",
        "label": "Chinula 2026 HPV16 (WLWH 25-50)",
        "color": "orangered",
        "marker": "D",
        "mapped_model": "Calib_hr1_F_25_59_HIVpos_Prev",
        "use_for_plot": True,
    },
    {
        "type": "HPV16",
        "population": "Non-WLWH 25-50",
        "sex": "F",
        "hiv_status": "HIVneg",
        "age_band": "25-50",
        "start_date": pd.Timestamp("2020-06-01"),
        "end_date": pd.Timestamp("2022-02-28"),
        "positive": 35,
        "denominator": 625,
        "prevalence": 35 / 625,
        "source": "Lameck Chinula, 2026",
        "label": "Chinula 2026 HPV16 (Non-WLWH 25-50)",
        "color": "navy",
        "marker": "D",
        "mapped_model": "Calib_hr1_F_25_59_HIVneg_Prev",
        "use_for_plot": True,
    },

    # -----------------------------------------------------------------
    # HPV18/45 -> Xpert P2
    # -----------------------------------------------------------------
    {
        "type": "HPV18/45",
        "population": "General",
        "sex": "All",
        "hiv_status": "All",
        "age_band": "All",
        "start_date": pd.Timestamp("2014-01-01"),
        "end_date": pd.Timestamp("2015-04-30"),
        "positive": 36,
        "denominator": 750,
        "prevalence": 36 / 750,
        "source": "Cubie et al., 2017",
        "label": "Cubie 2017 HPV18/45 (General)",
        "color": "crimson",
        "marker": "s",
        "mapped_model": "Xpert_P2_F_Prev",
        "use_for_plot": True,
    },
    {
        "type": "HPV18/45",
        "population": "25-49 WLWH",
        "sex": "F",
        "hiv_status": "HIVpos",
        "age_band": "25-49",
        "start_date": pd.Timestamp("2019-09-01"),
        "end_date": pd.Timestamp("2020-04-30"),
        "positive": 193,
        "denominator": 2203,
        "prevalence": 193 / 2203,
        "source": "Joseph et al., 2023",
        "label": "Joseph 2023 HPV18/45 (WLWH 25-49)",
        "color": "firebrick",
        "marker": "o",
        "mapped_model": "Calib_Xpert_P2_F_25_59_HIVpos_Prev",
        "use_for_plot": True,
    },
    {
        "type": "HPV18/45",
        "population": "WLWH 25-50",
        "sex": "F",
        "hiv_status": "HIVpos",
        "age_band": "25-50",
        "start_date": pd.Timestamp("2020-06-01"),
        "end_date": pd.Timestamp("2022-02-28"),
        "positive": 51,
        "denominator": 625,
        "prevalence": 51 / 625,
        "source": "Lameck Chinula, 2026",
        "label": "Chinula 2026 HPV18/45 (WLWH 25-50)",
        "color": "orangered",
        "marker": "D",
        "mapped_model": "Calib_Xpert_P2_F_25_59_HIVpos_Prev",
        "use_for_plot": True,
    },
    {
        "type": "HPV18/45",
        "population": "Non-WLWH 25-50",
        "sex": "F",
        "hiv_status": "HIVneg",
        "age_band": "25-50",
        "start_date": pd.Timestamp("2020-06-01"),
        "end_date": pd.Timestamp("2022-02-28"),
        "positive": 47,
        "denominator": 625,
        "prevalence": 47 / 625,
        "source": "Lameck Chinula, 2026",
        "label": "Chinula 2026 HPV18/45 (Non-WLWH 25-50)",
        "color": "navy",
        "marker": "D",
        "mapped_model": "Calib_Xpert_P2_F_25_59_HIVneg_Prev",
        "use_for_plot": True,
    },

    # -----------------------------------------------------------------
    # HPV31/33/35/52/58 -> Xpert P3
    # -----------------------------------------------------------------
    {
        "type": "HPV31/33/35/52/58",
        "population": "General",
        "sex": "All",
        "hiv_status": "All",
        "age_band": "All",
        "start_date": pd.Timestamp("2014-01-01"),
        "end_date": pd.Timestamp("2015-04-30"),
        "positive": 61,
        "denominator": 750,
        "prevalence": 61 / 750,
        "source": "Cubie et al., 2017",
        "label": "Cubie 2017 HPV31/33/35/52/58 (General)",
        "color": "crimson",
        "marker": "s",
        "mapped_model": "Xpert_P3_F_Prev",
        "use_for_plot": True,
    },
    {
        "type": "HPV31/33/35/52/58",
        "population": "WLWH 25-50",
        "sex": "F",
        "hiv_status": "HIVpos",
        "age_band": "25-50",
        "start_date": pd.Timestamp("2020-06-01"),
        "end_date": pd.Timestamp("2022-02-28"),
        "positive": 168,
        "denominator": 625,
        "prevalence": 168 / 625,
        "source": "Lameck Chinula, 2026",
        "label": "Chinula 2026 HPV31/33/35/52/58 (WLWH 25-50)",
        "color": "orangered",
        "marker": "D",
        "mapped_model": "Calib_Xpert_P3_F_25_59_HIVpos_Prev",
        "use_for_plot": True,
    },
    {
        "type": "HPV31/33/35/52/58",
        "population": "Non-WLWH 25-50",
        "sex": "F",
        "hiv_status": "HIVneg",
        "age_band": "25-50",
        "start_date": pd.Timestamp("2020-06-01"),
        "end_date": pd.Timestamp("2022-02-28"),
        "positive": 93,
        "denominator": 625,
        "prevalence": 93 / 625,
        "source": "Lameck Chinula, 2026",
        "label": "Chinula 2026 HPV31/33/35/52/58 (Non-WLWH 25-50)",
        "color": "navy",
        "marker": "D",
        "mapped_model": "Calib_Xpert_P3_F_25_59_HIVneg_Prev",
        "use_for_plot": True,
    },

    # -----------------------------------------------------------------
    # Other HR HPV -> Xpert P4
    # -----------------------------------------------------------------
    {
        "type": "Other HR HPV",
        "population": "General",
        "sex": "All",
        "hiv_status": "All",
        "age_band": "All",
        "start_date": pd.Timestamp("2014-01-01"),
        "end_date": pd.Timestamp("2015-04-30"),
        "positive": 95,
        "denominator": 750,
        "prevalence": 95 / 750,
        "source": "Cubie et al., 2017",
        "label": "Cubie 2017 Other HR HPV (General)",
        "color": "crimson",
        "marker": "s",
        "mapped_model": "Xpert_P4_F_Prev",
        "use_for_plot": True,
    },
    {
        "type": "Other HR HPV",
        "population": "25-49 WLWH",
        "sex": "F",
        "hiv_status": "HIVpos",
        "age_band": "25-49",
        "start_date": pd.Timestamp("2019-09-01"),
        "end_date": pd.Timestamp("2020-04-30"),
        "positive": 733,
        "denominator": 2203,
        "prevalence": 733 / 2203,
        "source": "Joseph et al., 2023",
        "label": "Joseph 2023 Other HR HPV (WLWH 25-49)",
        "color": "firebrick",
        "marker": "o",
        "mapped_model": "Calib_Xpert_P4_F_25_59_HIVpos_Prev",
        "use_for_plot": True,
    },

    # -----------------------------------------------------------------
    # Coinfection
    # -----------------------------------------------------------------
    {
        "type": "Coinfection",
        "population": "25-59 WLWH",
        "sex": "F",
        "hiv_status": "HIVpos",
        "age_band": "25-59",
        "start_date": pd.Timestamp("2011-11-01"),
        "end_date": pd.Timestamp("2012-04-30"),
        "positive": 170,
        "denominator": 294,
        "prevalence": 170 / 294,
        "source": "",
        "label": "Coinfection WLWH 25-59",
        "color": "gray",
        "marker": "x",
        "mapped_model": None,   # not directly available in current logging
        "use_for_plot": False,
    },
    {
        "type": "Coinfection",
        "population": "General",
        "sex": "All",
        "hiv_status": "All",
        "age_band": "All",
        "start_date": pd.Timestamp("2014-01-01"),
        "end_date": pd.Timestamp("2015-04-30"),
        "positive": 24,
        "denominator": 750,
        "prevalence": 24 / 750,
        "source": "Cubie et al., 2017",
        "label": "Cubie 2017 Coinfection (General)",
        "color": "crimson",
        "marker": "s",
        "mapped_model": "CoinfectionPrev",
        "use_for_plot": True,
    },
]

observed_df = pd.DataFrame(observed_rows)

# Plot 1: Overall HR-HPV prevalence over time
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    hpv_df["Date"],
    hpv_df["TotalPrev"],
    marker="o",
    label="Model: HR-HPV overall",
)

ax.set_xlabel("Date")
ax.set_ylabel("Total HPV prevalence")
ax.set_title("Overall HR-HPV prevalence over time")
ax.grid(True)
deduplicate_legend(ax)
plt.tight_layout()
plt.savefig(outputpath / "plot1_overall_hrhpv.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# Plot 2: Any HPV prevalence by sex + observed general male data
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

model_color_map = {}

plot_model_series(
    ax,
    hpv_df,
    col="M_Prev",
    label="Model: Male",
    model_color_map=model_color_map,
)

plot_model_series(
    ax,
    hpv_df,
    col="F_Prev",
    label="Model: Female",
    model_color_map=model_color_map,
)

obs_plot2 = observed_df.loc[
    (observed_df["type"] == "HR-HPV") &
    (observed_df["population"] == "General female") &
    (observed_df["use_for_plot"])
]

add_observed_points(ax, obs_plot2, model_color_map=model_color_map)

ax.set_xlabel("Date")
ax.set_ylabel("Any HPV prevalence")
ax.set_title("Any HPV prevalence by sex")
ax.grid(True)
deduplicate_legend(ax)
plt.tight_layout()
plt.savefig(outputpath / "plot2_hrhpv_by_sex_with_observed.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# Plot 3: Female HR-HPV prevalence by HIV status (calibration plot)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

model_color_map = {}

plot_model_series(
    ax,
    hpv_df,
    col="Calib_Any_F_25_59_HIVneg_Prev",
    label="Model: HIV-negative women 25-59",
    model_color_map=model_color_map,
)

plot_model_series(
    ax,
    hpv_df,
    col="Calib_Any_F_25_59_HIVpos_Prev",
    label="Model: WLWH 25-59",
    model_color_map=model_color_map,
)

obs_plot3 = observed_df.loc[
    (observed_df["type"] == "HR-HPV") &
    (observed_df["sex"] == "F") &
    (observed_df["hiv_status"].isin(["HIVpos", "HIVneg"])) &
    (observed_df["use_for_plot"])
]

add_observed_points(ax, obs_plot3, model_color_map=model_color_map)

ax.set_xlabel("Date")
ax.set_ylabel("HR-HPV prevalence")
ax.set_title("Calibration: female HR-HPV prevalence by HIV status")
ax.grid(True)
deduplicate_legend(ax)
plt.tight_layout()
plt.savefig(outputpath / "plot3_female_hrhpv_hiv_calibration.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# Plot 4: HPV16 calibration plot
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

model_color_map = {}

plot_model_series(
    ax,
    hpv_df,
    col="hr1_F_TotalPrev",
    label="Model: HPV16 female overall",
    model_color_map=model_color_map,
)

plot_model_series(
    ax,
    hpv_df,
    col="Calib_hr1_F_25_59_HIVpos_Prev",
    label="Model: HPV16 WLWH 25-59",
    model_color_map=model_color_map,
)

plot_model_series(
    ax,
    hpv_df,
    col="Calib_hr1_F_25_59_HIVneg_Prev",
    label="Model: HPV16 HIV-negative women 25-59",
    model_color_map=model_color_map,
)

obs_plot4 = observed_df.loc[
    (observed_df["type"] == "HPV16") &
    (observed_df["use_for_plot"])
]

add_observed_points(ax, obs_plot4, model_color_map=model_color_map)

ax.set_xlabel("Date")
ax.set_ylabel("Prevalence")
ax.set_title("Calibration: HPV16 prevalence")
ax.grid(True)
deduplicate_legend(ax)
plt.tight_layout()
plt.savefig(outputpath / "plot4_hpv16_calibration.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# Plot 5: HPV18/45 calibration plot (Xpert P2)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

model_color_map = {}

plot_model_series(
    ax,
    hpv_df,
    col="Xpert_P2_F_Prev",
    label="Model: HPV18/45 female overall",
    model_color_map=model_color_map,
)

plot_model_series(
    ax,
    hpv_df,
    col="Calib_Xpert_P2_F_25_59_HIVpos_Prev",
    label="Model: HPV18/45 WLWH 25-59",
    model_color_map=model_color_map,
)

plot_model_series(
    ax,
    hpv_df,
    col="Calib_Xpert_P2_F_25_59_HIVneg_Prev",
    label="Model: HPV18/45 HIV-negative women 25-59",
    model_color_map=model_color_map,
)

obs_plot5 = observed_df.loc[
    (observed_df["type"] == "HPV18/45") &
    (observed_df["use_for_plot"])
]

add_observed_points(ax, obs_plot5, model_color_map=model_color_map)

ax.set_xlabel("Date")
ax.set_ylabel("Prevalence")
ax.set_title("Calibration: HPV18/45 prevalence")
ax.grid(True)
deduplicate_legend(ax)
plt.tight_layout()
plt.savefig(outputpath / "plot5_hpv18_45_calibration.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# Plot 6: HPV31/33/35/52/58 calibration plot (Xpert P3)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

model_color_map = {}

plot_model_series(
    ax,
    hpv_df,
    col="Xpert_P3_F_Prev",
    label="Model: HPV31/33/35/52/58 female overall",
    model_color_map=model_color_map,
)

plot_model_series(
    ax,
    hpv_df,
    col="Calib_Xpert_P3_F_25_59_HIVpos_Prev",
    label="Model: HPV31/33/35/52/58 WLWH 25-59",
    model_color_map=model_color_map,
)

plot_model_series(
    ax,
    hpv_df,
    col="Calib_Xpert_P3_F_25_59_HIVneg_Prev",
    label="Model: HPV31/33/35/52/58 HIV-negative women 25-59",
    model_color_map=model_color_map,
)

obs_plot6 = observed_df.loc[
    (observed_df["type"] == "HPV31/33/35/52/58") &
    (observed_df["use_for_plot"])
]

add_observed_points(ax, obs_plot6, model_color_map=model_color_map)

ax.set_xlabel("Date")
ax.set_ylabel("Prevalence")
ax.set_title("Calibration: HPV31/33/35/52/58 prevalence")
ax.grid(True)
deduplicate_legend(ax)
plt.tight_layout()
plt.savefig(outputpath / "plot6_hpv31_33_35_52_58_calibration.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# Plot 7: Other HR HPV calibration plot (Xpert P4)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

model_color_map = {}

plot_model_series(
    ax,
    hpv_df,
    col="Xpert_P4_F_Prev",
    label="Model: Other HR HPV female overall",
    model_color_map=model_color_map,
)

plot_model_series(
    ax,
    hpv_df,
    col="Calib_Xpert_P4_F_25_59_HIVpos_Prev",
    label="Model: Other HR HPV WLWH 25-59",
    model_color_map=model_color_map,
)

plot_model_series(
    ax,
    hpv_df,
    col="Calib_Xpert_P4_F_25_59_HIVneg_Prev",
    label="Model: Other HR HPV HIV-negative women 25-59",
    model_color_map=model_color_map,
)

obs_plot7 = observed_df.loc[
    (observed_df["type"] == "Other HR HPV") &
    (observed_df["use_for_plot"])
]

add_observed_points(ax, obs_plot7, model_color_map=model_color_map)

ax.set_xlabel("Date")
ax.set_ylabel("Prevalence")
ax.set_title("Calibration: Other HR HPV prevalence")
ax.grid(True)
deduplicate_legend(ax)
plt.tight_layout()
plt.savefig(outputpath / "plot7_other_hr_hpv_calibration.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# Plot 8: Coinfection prevalence overall
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

model_color_map = {}

plot_model_series(
    ax,
    hpv_df,
    col="CoinfectionPrev",
    label="Model: Coinfection overall (>=2 HPV groups)",
    model_color_map=model_color_map,
)

obs_plot8 = observed_df.loc[
    (observed_df["type"] == "Coinfection") &
    (observed_df["population"] == "General") &
    (observed_df["use_for_plot"])
]

add_observed_points(ax, obs_plot8, model_color_map=model_color_map)

ax.set_xlabel("Date")
ax.set_ylabel("Coinfection prevalence")
ax.set_title("Coinfection prevalence over time")
ax.grid(True)
deduplicate_legend(ax)
plt.tight_layout()
plt.savefig(outputpath / "plot8_coinfection_general_calibration.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# Plot 9: Multiplicity of infection over time
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))

for n in range(1, len(HPV_GROUPS) + 1):
    col = f"InfGroup{n}"
    if col in hpv_df.columns:
        plt.plot(hpv_df["Date"], hpv_df[col], marker="o", label=f"{n} HPV group(s)")

plt.xlabel("Date")
plt.ylabel("Number of infected individuals")
plt.title("Multiplicity of HPV infection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(outputpath / "plot9_hpv_multiplicity_over_time.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# Plot 10: Persistent HPV infection prevalence
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))

for group in HPV_GROUPS:
    col = f"{group}_Persistent12_Prev"
    if col in hpv_df.columns:
        plt.plot(
            hpv_df["Date"],
            hpv_df[col],
            marker="o",
            label=group,
        )

plt.xlabel("Date")
plt.ylabel("Persistent infection prevalence")
plt.title("Persistent HPV infection prevalence (>=12 months)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(outputpath / "plot11_hpv_persistent_prevalence.png", dpi=300)
plt.show()


# # ------------------------------------------------------------
# # Plot 11: Persistent HPV infection by sex
# # ------------------------------------------------------------
# for group in HPV_GROUPS:
#     male_col = f"{group}_Persistent12_M_Prev"
#     female_col = f"{group}_Persistent12_F_Prev"
#
#     if not all(col in hpv_df.columns for col in [male_col, female_col]):
#         print(f"Skipping persistent-by-sex plot for {group}: missing columns.")
#         continue
#
#     plt.figure(figsize=(8, 5))
#     plt.plot(hpv_df["Date"], hpv_df[male_col], marker="o", label="Male")
#     plt.plot(hpv_df["Date"], hpv_df[female_col], marker="o", label="Female")
#
#     plt.xlabel("Date")
#     plt.ylabel("Persistent infection prevalence")
#     plt.title(f"{group} persistent infection prevalence by sex")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(outputpath / f"plot12_{group}_persistent_by_sex.png", dpi=300)
#     plt.show()


# ------------------------------------------------------------
# Plot 13: Female any HPV prevalence by age group
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))

for age in AGE_LABELS:
    col = f"Any_F_{age}_Prev"
    if col in hpv_df.columns:
        plt.plot(hpv_df["Date"], hpv_df[col], marker="o", label=age)

plt.xlabel("Date")
plt.ylabel("Any HPV prevalence")
plt.title("Female any HPV prevalence by age group")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(outputpath / "plot13_female_any_hpv_by_age.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# Plot 14: Male any HPV prevalence by age group
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))

for age in AGE_LABELS:
    col = f"Any_M_{age}_Prev"
    if col in hpv_df.columns:
        plt.plot(hpv_df["Date"], hpv_df[col], marker="o", label=age)

plt.xlabel("Date")
plt.ylabel("Any HPV prevalence")
plt.title("Male any HPV prevalence by age group")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(outputpath / "plot14_male_any_hpv_by_age.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# Plot 15: Any HPV prevalence by HIV/ART status
# ------------------------------------------------------------
hiv_prev_cols = [
    "Any_HIVneg_Prev",
    "Any_HIVpos_unknown_Prev",
    "Any_HIVpos_noART_Prev",
    "Any_HIVpos_unsupp_Prev",
    "Any_HIVpos_supp_Prev",
]

available_hiv_cols = [c for c in hiv_prev_cols if c in hpv_df.columns]

if len(available_hiv_cols) > 0:
    plt.figure(figsize=(9, 5))

    for col in available_hiv_cols:
        label = col.replace("Any_", "").replace("_Prev", "")
        plt.plot(hpv_df["Date"], hpv_df[col], marker="o", label=label)

    plt.xlabel("Date")
    plt.ylabel("Any HPV prevalence")
    plt.title("Any HPV prevalence by HIV/ART status")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outputpath / "plot15_hpv_prevalence_by_hiv_art_status.png", dpi=300)
    plt.show()


# ------------------------------------------------------------
# Plot 16: Any HPV prevalence by HIV status (negative vs positive)
# ------------------------------------------------------------
required_cols = ["Any_HIVneg_Prev", "Any_HIVpos_Prev"]

if all(col in hpv_df.columns for col in required_cols):
    plt.figure(figsize=(9, 5))
    plt.plot(hpv_df["Date"], hpv_df["Any_HIVneg_Prev"], marker="o", label="HIV negative")
    plt.plot(hpv_df["Date"], hpv_df["Any_HIVpos_Prev"], marker="o", label="HIV positive")

    plt.xlabel("Date")
    plt.ylabel("Any HPV prevalence")
    plt.title("Any HPV prevalence by HIV status")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outputpath / "plot16_hpv_prevalence_hiv_negative_vs_positive.png", dpi=300)
    plt.show()
else:
    print("Cannot plot HIV negative vs positive prevalence: missing columns.")


# ------------------------------------------------------------
# Plot 17: Female overall HPV vaccine coverage across all ages >=9
# ------------------------------------------------------------
required_cols = [
    "HPVVaccinated_F_9_14_N",
    "HPVVaccinated_F_9_14_Denominator",
    "HPVVaccinated_F_N",
    "F_N",
]

if all(col in hpv_df.columns for col in required_cols):
    hpv_df["HPVVaccinated_F_9+_N"] = (
        hpv_df["HPVVaccinated_F_9_14_N"] + hpv_df["HPVVaccinated_F_N"]
    )
    hpv_df["HPVVaccinated_F_9+_Denominator"] = (
        hpv_df["HPVVaccinated_F_9_14_Denominator"] + hpv_df["F_N"]
    )
    hpv_df["HPVVaccinated_F_9+_Coverage"] = (
        hpv_df["HPVVaccinated_F_9+_N"] /
        hpv_df["HPVVaccinated_F_9+_Denominator"].replace(0, pd.NA)
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        hpv_df["Date"],
        hpv_df["HPVVaccinated_F_9+_Coverage"],
        marker="o",
        label="Females 9+"
    )
    plt.xlabel("Date")
    plt.ylabel("HPV vaccine coverage")
    plt.title("Female HPV vaccine coverage over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outputpath / "plot17_female_hpv_vaccine_coverage_over_time.png", dpi=300)
    plt.show()
else:
    print("Cannot compute female 9+ vaccine coverage: missing columns.")


# ------------------------------------------------------------
# Plot 18: Female HPV vaccine coverage by age group over time
# ------------------------------------------------------------
female_vax_age_cols = ["HPVVaccinated_F_9_14_Coverage"] + [
    f"HPVVaccinated_F_{age}_Coverage" for age in AGE_LABELS
]

available_female_vax_age_cols = [c for c in female_vax_age_cols if c in hpv_df.columns]

if len(available_female_vax_age_cols) > 0:
    plt.figure(figsize=(10, 6))

    for col in available_female_vax_age_cols:
        label = col.replace("HPVVaccinated_F_", "").replace("_Coverage", "")
        plt.plot(hpv_df["Date"], hpv_df[col], marker="o", label=label)

    plt.xlabel("Date")
    plt.ylabel("HPV vaccine coverage")
    plt.title("Female HPV vaccine coverage by age group")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outputpath / "plot18_female_hpv_vaccine_coverage_by_age.png", dpi=300)
    plt.show()
else:
    print("No female age-specific vaccine coverage columns found.")

# ------------------------------------------------------------
# Female 25-59: HIV prevalence and ART/VL suppression trends
# ------------------------------------------------------------

female_hiv_cols = [
    "Calib_Any_F_25_59_HIVneg_N",
    "Calib_Any_F_25_59_HIVpos_N",
    "Calib_Any_F_25_59_HIVpos_supp_N",
    "Calib_Any_F_25_59_HIVpos_unsupp_N",
]

missing_cols = [c for c in female_hiv_cols if c not in hpv_df.columns]

if missing_cols:
    print("Cannot plot female HIV/ART trends. Missing columns:")
    print(missing_cols)
    print("\nAvailable related columns:")
    print([c for c in hpv_df.columns if "Calib_Any_F_25_59_HIV" in c])
else:
    # Total female 25-59 population in this calibration group
    hpv_df["F_25_59_Total_N"] = (
        hpv_df["Calib_Any_F_25_59_HIVneg_N"] +
        hpv_df["Calib_Any_F_25_59_HIVpos_N"]
    )

    # Female 25-59 HIV prevalence
    hpv_df["F_25_59_HIV_Prevalence"] = (
        hpv_df["Calib_Any_F_25_59_HIVpos_N"] /
        hpv_df["F_25_59_Total_N"].replace(0, pd.NA)
    )

    # Among HIV-positive women 25-59:
    # suppressed / unsuppressed
    hpv_df["F_25_59_WLWH_Suppressed_Share"] = (
        hpv_df["Calib_Any_F_25_59_HIVpos_supp_N"] /
        hpv_df["Calib_Any_F_25_59_HIVpos_N"].replace(0, pd.NA)
    )

    hpv_df["F_25_59_WLWH_Unsuppressed_Share"] = (
        hpv_df["Calib_Any_F_25_59_HIVpos_unsupp_N"] /
        hpv_df["Calib_Any_F_25_59_HIVpos_N"].replace(0, pd.NA)
    )

    # Save table
    female_hiv_art_cols = [
        "Date",
        "F_25_59_Total_N",
        "Calib_Any_F_25_59_HIVneg_N",
        "Calib_Any_F_25_59_HIVpos_N",
        "F_25_59_HIV_Prevalence",
        "F_25_59_WLWH_Suppressed_Share",
        "F_25_59_WLWH_Unsuppressed_Share",
    ]

    hpv_df[female_hiv_art_cols].to_csv(
        outputpath / "female_25_59_hiv_art_trends.csv",
        index=False
    )

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        hpv_df["Date"],
        hpv_df["F_25_59_HIV_Prevalence"],
        marker="o",
        label="HIV prevalence among women 25-59"
    )

    # ax.plot(
    #     hpv_df["Date"],
    #     hpv_df["F_25_59_WLWH_Suppressed_Share"],
    #     marker="o",
    #     label="VL suppressed among WLWH 25-59"
    # )
    #
    # ax.plot(
    #     hpv_df["Date"],
    #     hpv_df["F_25_59_WLWH_Unsuppressed_Share"],
    #     marker="o",
    #     label="Unsuppressed/noART among WLWH 25-59"
    # )

    ax.set_xlabel("Date")
    ax.set_ylabel("Proportion")
    ax.set_title("Female 25-59: HIV prevalence and ART/VL status over time")
    ax.grid(True)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outputpath / "female_25_59_hiv_art_trends.png", dpi=300)
    plt.show()
