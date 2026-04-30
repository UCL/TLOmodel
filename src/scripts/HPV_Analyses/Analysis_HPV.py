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
end_date = Date(2020, 1, 1)
popsize = 2000


# set up the log config
log_config = {
    "filename": "test_runs",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.hpv": logging.INFO,
    },
}
#
# # Register the appropriate modules
# # need to call epi before tb to get bcg vax
seed = random.randint(0, 50000)
# # seed = 41728  # set seed for reproducibility

# HPV model labels
AGE_LABELS = ["15_19", "20_24", "25_34", "35_44", "45_54", "55plus"]
HPV_GROUPS = ["hr1", "hr2", "hr3"]
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


# change Year / Month to Date
hpv_df["Year"] = pd.to_numeric(hpv_df["Year"], errors="coerce")
hpv_df["Month"] = pd.to_numeric(hpv_df["Month"], errors="coerce")

hpv_df = hpv_df.dropna(subset=["Year", "Month"]).copy()
hpv_df["Year"] = hpv_df["Year"].astype(int)
hpv_df["Month"] = hpv_df["Month"].astype(int)

hpv_df["Date"] = pd.to_datetime(
    {
        "year": hpv_df["Year"],
        "month": hpv_df["Month"],
        "day": 1,
    }
)
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

for sex in SEXES:
    for group in HPV_GROUPS:
        hpv_df[f"{group}_{sex}_TotalPrev"] = compute_group_prev_by_sex(
            hpv_df,
            hpv_group=group,
            sex=sex,
            age_labels=AGE_LABELS,
        )

# Plot 1: Total infection
plt.figure(figsize=(8, 5))
plt.plot(hpv_df["Date"], hpv_df["TotalPrev"], marker="o")
plt.xlabel("Date")
plt.ylabel("Total HPV prevalence")
plt.title("Total HPV prevalence over time")
plt.grid(True)
plt.tight_layout()
plt.savefig(outputpath / "hpv_total_prevalence.png", dpi=300)
plt.show()

# 2. HPV  prevalence in different gender
plt.figure(figsize=(8, 5))
plt.plot(hpv_df["Date"], hpv_df["M_Prev"], marker="o", label="Male")
plt.plot(hpv_df["Date"], hpv_df["F_Prev"], marker="o", label="Female")
plt.xlabel("Date")
plt.ylabel("Any HPV prevalence")
plt.title("Any HPV prevalence by sex")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(outputpath / "hpv_prevalence_by_sex.png", dpi=300)
plt.show()

# 3. Prevalence of different HPV groups in female
plt.figure(figsize=(8, 5))
for group in HPV_GROUPS:
    plt.plot(
        hpv_df["Date"],
        hpv_df[f"{group}_F_TotalPrev"],
        marker="o",
        label=group,
    )
plt.xlabel("Date")
plt.ylabel("Prevalence")
plt.title("Female HPV prevalence by group")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(outputpath / "female_hpv_group_prevalence.png", dpi=300)
plt.show()

# 4. Prevalence of different HPV groups in male
plt.figure(figsize=(8, 5))
for group in HPV_GROUPS:
    plt.plot(
        hpv_df["Date"],
        hpv_df[f"{group}_M_TotalPrev"],
        marker="o",
        label=group,
    )
plt.xlabel("Date")
plt.ylabel("Prevalence")
plt.title("Male HPV prevalence by group")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(outputpath / "male_hpv_group_prevalence.png", dpi=300)
plt.show()

# Plot 5: Multiplicity of infection
plt.figure(figsize=(8, 5))
plt.plot(hpv_df["Date"], hpv_df["InfGroup1"], marker="o", label="1 HPV group")
plt.plot(hpv_df["Date"], hpv_df["InfGroup2"], marker="o", label="2 HPV groups")
plt.plot(hpv_df["Date"], hpv_df["InfGroup3"], marker="o", label="3 HPV groups")
plt.xlabel("Date")
plt.ylabel("Number of infected individuals")
plt.title("Multiplicity of HPV infection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(outputpath / "hpv_multiplicity_over_time.png", dpi=300)
plt.show()

# Plot 6: Multiplicity of infection by sex
plt.figure(figsize=(9, 5))

plt.plot(hpv_df["Date"], hpv_df["MaleGroup1"], marker="o", label="Male: 1 group")
plt.plot(hpv_df["Date"], hpv_df["MaleGroup2"], marker="o", label="Male: 2 groups")
plt.plot(hpv_df["Date"], hpv_df["MaleGroup3"], marker="o", label="Male: 3 groups")

plt.plot(hpv_df["Date"], hpv_df["FemaleGroup1"], marker="o", label="Female: 1 group")
plt.plot(hpv_df["Date"], hpv_df["FemaleGroup2"], marker="o", label="Female: 2 groups")
plt.plot(hpv_df["Date"], hpv_df["FemaleGroup3"], marker="o", label="Female: 3 groups")

plt.xlabel("Date")
plt.ylabel("Number of infected individuals")
plt.title("Multiplicity of HPV infection by sex")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(outputpath / "hpv_multiplicity_by_sex.png", dpi=300)
plt.show()

# Plot 7: Persistent HPV infection prevalence
plt.figure(figsize=(8, 5))

for group in HPV_GROUPS:
    plt.plot(
        hpv_df["Date"],
        hpv_df[f"{group}_Persistent12_Prev"],
        marker="o",
        label=group,
    )

plt.xlabel("Date")
plt.ylabel("Persistent infection prevalence")
plt.title("Persistent HPV infection prevalence, >=12 months")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(outputpath / "hpv_persistent_prevalence.png", dpi=300)
plt.show()

# Plot 8: Persistent HPV infection by sex
for group in HPV_GROUPS:
    male_col = f"{group}_Persistent12_M_Prev"
    female_col = f"{group}_Persistent12_F_Prev"

    required_cols = ["Date", male_col, female_col]

    plt.figure(figsize=(8, 5))
    plt.plot(hpv_df["Date"], hpv_df[male_col], marker="o", label="Male")
    plt.plot(hpv_df["Date"], hpv_df[female_col], marker="o", label="Female")

    plt.xlabel("Date")
    plt.ylabel("Persistent infection prevalence")
    plt.title(f"{group} persistent infection prevalence by sex")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outputpath / f"{group}_persistent_by_sex.png", dpi=300)
    plt.show()

# Plot 9: Female any HPV prevalence by age group
female_age_cols = [f"Any_F_{age}_Prev" for age in AGE_LABELS]

plt.figure(figsize=(9, 5))

for age in AGE_LABELS:
    plt.plot(
        hpv_df["Date"],
        hpv_df[f"Any_F_{age}_Prev"],
        marker="o",
        label=age,
    )

plt.xlabel("Date")
plt.ylabel("Any HPV prevalence")
plt.title("Female any HPV prevalence by age group")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(outputpath / "female_any_hpv_by_age.png", dpi=300)
plt.show()

# Plot 10: Male any HPV prevalence by age group
male_age_cols = [f"Any_M_{age}_Prev" for age in AGE_LABELS]

plt.figure(figsize=(9, 5))

for age in AGE_LABELS:
    plt.plot(
        hpv_df["Date"],
        hpv_df[f"Any_M_{age}_Prev"],
        marker="o",
        label=age,
    )

plt.xlabel("Date")
plt.ylabel("Any HPV prevalence")
plt.title("Male any HPV prevalence by age group")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(outputpath / "male_any_hpv_by_age.png", dpi=300)
plt.show()

# Plot 11: Any HPV prevalence by HIV/ART status
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
    plt.savefig(outputpath / "hpv_prevalence_by_hiv_status.png", dpi=300)
    plt.show()

