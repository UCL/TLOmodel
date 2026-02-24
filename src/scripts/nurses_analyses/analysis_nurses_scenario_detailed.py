"""
Analysis of nurse expansion scenarios.

Produces:
1. Yearly nurse cadre counts (2010–2034)
2. Minutes used per cadre (focus on nurses)
3. Appointments delivered per year
4. Working time used per cadre (focus on nurses)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from tlo.analysis.utils import (
    extract_results,
    compute_mean_across_runs,
)

START_YEAR = 2010
END_YEAR = 2034


# =============================================================================
# COPIED + ADAPTED FUNCTIONS FROM analysis_hsi_descriptions.py
# =============================================================================

def get_frac_of_hcw_time_used(df):
    """
    Returns minutes of time used per cadre.
    Extracts the time used dictionary stored in HSI_Event output.
    """

    if "Time_Used_By_Cadre" not in df.columns:
        return pd.DataFrame()

    # Expand dict column into columns
    expanded = df["Time_Used_By_Cadre"].apply(pd.Series)

    expanded["date"] = df["date"]

    return expanded


def hcw_time_or_cost_used(df, return_time=True):
    """
    Returns total minutes used per cadre (if return_time=True)
    Otherwise returns cost.
    """

    column = "Time_Used_By_Cadre" if return_time else "Cost_By_Cadre"

    if column not in df.columns:
        return pd.DataFrame()

    expanded = df[column].apply(pd.Series)
    expanded["date"] = df["date"]

    return expanded


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def apply(results_folder: Path, output_folder: Path):

    output_folder.mkdir(exist_ok=True, parents=True)

    # ---------------------------------------------------------------------
    # 1️⃣ CADRE COUNTS
    # ---------------------------------------------------------------------

    cadre_counts = extract_results(
        results_folder,
        module="HealthSystem",
        key="Current_Number_Of_Health_Workers_By_Cadre",
    )

    cadre_counts = compute_mean_across_runs(cadre_counts)

    cadre_counts["year"] = pd.to_datetime(cadre_counts["date"]).dt.year
    cadre_counts = cadre_counts[cadre_counts["year"].between(START_YEAR, END_YEAR)]

    yearly_counts = cadre_counts.groupby("year").mean()

    nurse_cols = [c for c in yearly_counts.columns if "Nurse" in c or "Midwife" in c]

    plt.figure()
    for col in nurse_cols:
        plt.plot(yearly_counts.index, yearly_counts[col], label=col)

    plt.title("Yearly Nurse Cadre Counts (2010–2034)")
    plt.xlabel("Year")
    plt.ylabel("Number of Staff")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_folder / "yearly_nurse_cadre_counts.png")
    plt.close()

    # ---------------------------------------------------------------------
    # 2️⃣ MINUTES USED PER CADRE
    # ---------------------------------------------------------------------

    minutes_used = extract_results(
        results_folder,
        module="HealthSystem",
        key="HSI_Event",
        custom_generate_series=get_frac_of_hcw_time_used
    )

    minutes_used = compute_mean_across_runs(minutes_used)

    minutes_used["year"] = pd.to_datetime(minutes_used["date"]).dt.year
    minutes_used = minutes_used[minutes_used["year"].between(START_YEAR, END_YEAR)]

    yearly_minutes = minutes_used.groupby("year").sum()

    nurse_minutes_cols = [c for c in yearly_minutes.columns if "Nurse" in c]

    plt.figure()
    for col in nurse_minutes_cols:
        plt.plot(yearly_minutes.index, yearly_minutes[col], label=col)

    plt.title("Yearly Minutes Used by Nurse Cadres")
    plt.xlabel("Year")
    plt.ylabel("Minutes Used")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_folder / "yearly_nurse_minutes_used.png")
    plt.close()

    # ---------------------------------------------------------------------
    # 3️⃣ APPOINTMENTS DELIVERED
    # ---------------------------------------------------------------------

    def extract_appointments(df):
        if "Number_By_Appt_Type_Code" not in df.columns:
            return pd.DataFrame()

        expanded = df["Number_By_Appt_Type_Code"].apply(pd.Series)
        expanded["date"] = df["date"]
        return expanded

    appts = extract_results(
        results_folder,
        module="HealthSystem",
        key="HSI_Event",
        custom_generate_series=extract_appointments
    )

    appts = compute_mean_across_runs(appts)

    appts["year"] = pd.to_datetime(appts["date"]).dt.year
    appts = appts[appts["year"].between(START_YEAR, END_YEAR)]

    yearly_appts = appts.groupby("year").sum()

    plt.figure()
    plt.plot(yearly_appts.index, yearly_appts.sum(axis=1))
    plt.title("Total Appointments Delivered per Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Appointments")
    plt.tight_layout()
    plt.savefig(output_folder / "yearly_total_appointments.png")
    plt.close()

    # ---------------------------------------------------------------------
    # 4️⃣ WORKING TIME USED PER CADRE
    # ---------------------------------------------------------------------

    working_time = extract_results(
        results_folder,
        module="HealthSystem",
        key="HSI_Event",
        custom_generate_series=lambda df: hcw_time_or_cost_used(df, return_time=True)
    )

    working_time = compute_mean_across_runs(working_time)

    working_time["year"] = pd.to_datetime(working_time["date"]).dt.year
    working_time = working_time[working_time["year"].between(START_YEAR, END_YEAR)]

    yearly_working_time = working_time.groupby("year").sum()

    nurse_time_cols = [c for c in yearly_working_time.columns if "Nurse" in c]

    plt.figure()
    for col in nurse_time_cols:
        plt.plot(yearly_working_time.index, yearly_working_time[col], label=col)

    plt.title("Working Time Used by Nurse Cadres")
    plt.xlabel("Year")
    plt.ylabel("Minutes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_folder / "yearly_nurse_working_time.png")
    plt.close()

    print("All nurse scenario plots generated successfully.")
