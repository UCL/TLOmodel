import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import get_root_path, parse_log_file
from tlo.methods import (
    cardio_metabolic_disorders,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    diabetic_retinopathy,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    newborn_outcomes,
    oesophagealcancer,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
    tb,
)

outputpath = Path("./outputs/diabetic_retinopathy_outputs")
outputpath.mkdir(parents=True, exist_ok=True)

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

root = get_root_path()
resourcefilepath = root / "resources"

# Set parameters for the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2030, 1, 1)
popsize = 3000
seed = 0
FORECAST_BASELINE_YEAR = 2020
FORECAST_END_YEAR = 2029

CALIBRATION_FILE = Path("./resources/diabetic_retinopathy_calibration_targets.csv")

def run_sim(service_availability, run_name="baseline"):
    # log_config = {
    #     'filename': 'LogFile',
    #     'directory': outputpath
    # }

    # sim = Simulation(start_date=start_date, seed=0,
    #                  log_config={"filename": "LogFile"}, resourcefilepath=resourcefilepath)
    logfile_name = f"LogFile_{run_name}_{datestamp}"
    sim = Simulation(
        start_date=start_date,
        seed=seed,
        log_config={"filename": logfile_name, "directory": outputpath},
        resourcefilepath=resourcefilepath,
    )

    # Register the appropriate modules
    sim.register(care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(),
                 demography.Demography(),
                 contraception.Contraception(),
                 enhanced_lifestyle.Lifestyle(),
                 healthsystem.HealthSystem(service_availability=service_availability),
                 symptommanager.SymptomManager(),
                 healthseekingbehaviour.HealthSeekingBehaviour(),
                 healthburden.HealthBurden(),
                 labour.Labour(),
                 newborn_outcomes.NewbornOutcomes(),
                 pregnancy_supervisor.PregnancySupervisor(),
                 oesophagealcancer.OesophagealCancer(),
                 postnatal_supervisor.PostnatalSupervisor(),
                 diabetic_retinopathy.DiabeticRetinopathy(),
                 hiv.Hiv(),
                 tb.Tb(),
                 epi.Epi(),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(),
                 depression.Depression(),
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    return sim.log_filepath


# Previously
# def get_summary_stats(logfile):
#     output = parse_log_file(logfile)
#     df = output['tlo.methods.diabetic_retinopathy']['summary_stats']
#     # Create a date column for filtering
#     df['date'] = pd.to_datetime(df['date'])
#     df = df.set_index('date')
#     return df

# Alternative
def get_summary_stats(logfile):
    """Read the DR module's monthly summary_stats log."""
    output = parse_log_file(logfile)
    df = output["tlo.methods.diabetic_retinopathy"]["summary_stats"].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def annualise_monthly_results(df):
    """Keep December observations as year-end prevalence/stock estimates."""
    year = df.index.year
    annual = df.groupby(year).last()

    # Incidence columns are monthly flows, so summing them over the year rather
    # than keeping only December's value. Prevalence/count columns remain
    # year-end stocks and therefore we use the last observation in each year.
    incidence_cols = [
        c for c in df.columns
        if c.startswith(("inc_dr_", "inc_dmo_", "inc_vision_"))
    ]
    if incidence_cols:
        annual_incidence = df[incidence_cols].groupby(year).sum()
        annual.loc[:, incidence_cols] = annual_incidence

    annual.index.name = "year"
    return annual


def safe_percent(numerator, denominator):
    """Percentages with missing/zero denominators represented as NA."""
    return (100.0 * numerator / denominator.where(denominator != 0)).astype(float)



def derive_analysis_metrics(monthly):
    """
    Derive measures from existing monthly logs. Vision impairment attribution
    requires the small extra logging addition described in the accompanying
    instructions; without it, do not treat all impairment as DR-attributable.
    """
    df = monthly.copy()
    df["total_dr_any"] = (
        df["total_dr_mild_or_moderate"]
        + df["total_dr_severe"]
        + df["total_dr_proliferative"]
    )
    df["total_dmo_any"] = (
        df["total_dmo_clinically_significant"]
        + df["total_dmo_non_clinically_significant"]
    )
    df["total_vision_impaired"] = (
        df["total_vision_moderate_vision_impairment"]
        + df["total_vision_severe_vision_impairment"]
        + df["total_vision_blindness"]
    )

    # DR among people with diabetes; DMO among
    # people with DR; vision impairment among people with diabetes.
    df["dr_prevalence_among_diabetics_pct"] = safe_percent(
        df["total_dr_any"], df["total_alive_diabetics"]
    )
    df["dmo_prevalence_among_dr_pct"] = safe_percent(
        df["total_dmo_any"], df["total_dr_any"]
    )
    df["vision_impairment_among_diabetics_pct"] = safe_percent(
        df["total_vision_impaired"], df["total_alive_diabetics"]
    )
    df["severe_vision_or_blindness_among_diabetics_pct"] = safe_percent(
        df["total_vision_severe_vision_impairment"] + df["total_vision_blindness"],
        df["total_alive_diabetics"],
    )
    df["blindness_among_diabetics_pct"] = safe_percent(
        df["total_vision_blindness"], df["total_alive_diabetics"]
    )

    # Cumulative incidence is not reconstructed by summing monthly incidence
    # blindly across calendar years; the logged monthly counts are summed below.
    return df

def plot_burden_trends(annual):
    """The core natural-history and possibly future-burden figures."""
    figures = {
        "dr_stages_prevalence": (
            ["total_dr_mild_or_moderate", "total_dr_severe", "total_dr_proliferative"],
            "DR stages: modelled counts",
            "People alive",
        ),
        "vision_status_counts": (
            ["total_vision_moderate_vision_impairment",
             "total_vision_severe_vision_impairment", "total_vision_blindness"],
            "Vision outcomes: modelled counts",
            "People alive",
        ),
        "incidence_dr": (
            ["inc_dr_mild_or_moderate", "inc_dr_severe", "inc_dr_proliferative"],
            "Annual new DR-stage cases",
            "New cases per year",
        ),
        "incidence_vision": (
            ["inc_vision_moderate_vision_impairment",
             "inc_vision_severe_vision_impairment", "inc_vision_blindness"],
            "Annual new vision-impairment transitions",
            "New transitions per year",
        ),
    }

    for filename, (cols, title, ylabel) in figures.items():
        available = [c for c in cols if c in annual.columns]
        if not available:
            continue

        # Grouped bar chart with a wider plotting area
        fig, ax = plt.subplots(figsize=(16, 7))

        annual[available].plot(
            kind="bar",
            ax=ax,
            width=0.85,
        )

        ax.set_title(title, pad=12)
        ax.set_xlabel("Year", labelpad=8)
        ax.set_ylabel(ylabel)

        # Keep year labels horizontal and readable
        ax.tick_params(axis="x", labelrotation=0, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)

        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        # Put the legend below the x-axis label, centred
        ax.legend(
            title="Outcome",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=len(available),
            frameon=False,
        )

        # Reserve room for the legend below the plot
        fig.subplots_adjust(
            left=0.10,
            right=0.98,
            top=0.90,
            bottom=0.27,
        )

        fig.savefig(
            outputpath / f"{filename}_{datestamp}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Percentages use the actual number of alive people with diabetes, not the
    # whole population or the sum of selected categories.
    pct_cols = [
        "dr_prevalence_among_diabetics_pct",
        "vision_impairment_among_diabetics_pct",
        "severe_vision_or_blindness_among_diabetics_pct",
        "blindness_among_diabetics_pct",
    ]

    available = [c for c in pct_cols if c in annual.columns]

    if available:
        fig, ax = plt.subplots(figsize=(16, 7))

        annual[available].plot(
            kind="bar",
            ax=ax,
            width=0.85,
        )

        ax.set_title(
            "DR and vision outcomes among people living with diabetes",
            pad=12,
        )
        ax.set_xlabel("Year", labelpad=8)
        ax.set_ylabel("Percent of people with diabetes")

        ax.tick_params(axis="x", labelrotation=0, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)

        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        ax.legend(
            title="Outcome",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=len(available),
            frameon=False,
        )

        fig.subplots_adjust(
            left=0.10,
            right=0.98,
            top=0.90,
            bottom=0.27,
        )

        fig.savefig(
            outputpath / f"burden_percentages_{datestamp}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def make_pathway_summary(annual):
    """
    Summarise the model's recorded first-diagnosis pathways. These are
    descriptive groups, not randomised/comparable cohorts.
    """
    rows = []
    for pathway in ("screening", "symptomatic"):
        for year in (FORECAST_BASELINE_YEAR, FORECAST_END_YEAR):
            if year not in annual.index:
                continue
            row = {"year": year, "pathway": pathway}
            for col in (
                f"n_diagnosed_{pathway}",
                f"n_on_treatment_{pathway}",
                f"total_dr_mild_or_moderate_{pathway}",
                f"total_dr_severe_{pathway}",
                f"total_dr_proliferative_{pathway}",
                f"total_vision_moderate_vision_impairment_{pathway}",
                f"total_vision_severe_vision_impairment_{pathway}",
                f"total_vision_blindness_{pathway}",
            ):
                row[col] = annual.at[year, col] if col in annual.columns else pd.NA
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(outputpath / f"pathway_summary_{datestamp}.csv", index=False)
    return result


def make_forecast_summary(annual):
    """Compare the (baseline) 2012 model state with 2014 (forecast end year) and quantify absolute changes."""
    requested = [
        "total_alive_diabetics",
        "total_dr_mild_or_moderate",
        "total_dr_severe",
        "total_dr_proliferative",
        "total_dr_any",
        "total_dmo_any",
        "total_vision_moderate_vision_impairment",
        "total_vision_severe_vision_impairment",
        "total_vision_blindness",
        "dr_prevalence_among_diabetics_pct",
        "vision_impairment_among_diabetics_pct",
        "severe_vision_or_blindness_among_diabetics_pct",
        "blindness_among_diabetics_pct",
    ]
    years = [y for y in (FORECAST_BASELINE_YEAR, FORECAST_END_YEAR) if y in annual.index]
    if len(years) < 2:
        print(f"WARNING: Need annual outputs for both {FORECAST_BASELINE_YEAR} and "
              f"{FORECAST_END_YEAR}; found {years}. Check simulation end date.")
        return pd.DataFrame()
    summary = annual.loc[years, [c for c in requested if c in annual.columns]].T
    summary.columns = [f"value_{y}" for y in years]
    summary["absolute_change"] = summary.iloc[:, 1] - summary.iloc[:, 0]
    summary["percent_change"] = (
        summary["absolute_change"] / summary.iloc[:, 0].replace(0, pd.NA) * 100
    )
    summary.to_csv(outputpath / f"forecast_2012_to_2014_{datestamp}.csv")
    return summary


def main():
    logfile = run_sim(service_availability=["*"], run_name="full_healthsystem")
    monthly = get_summary_stats(logfile)
    monthly = derive_analysis_metrics(monthly)
    annual = annualise_monthly_results(monthly)

    monthly.to_csv(outputpath / f"monthly_results_{datestamp}.csv")
    annual.to_csv(outputpath / f"annual_results_{datestamp}.csv")

    plot_burden_trends(annual)
    pathway_summary = make_pathway_summary(annual)
    forecast_summary = make_forecast_summary(annual)
    # check_calibration_targets(annual)

    # Print concise results for quick review; complete results are saved above.
    print("\nForecast summary (2026 model state versus 2040):")
    print(forecast_summary)
    print("\nPathway summary (descriptive; not causal):")
    print(pathway_summary)
    print(f"\nOutputs saved to: {outputpath.resolve()}")


if __name__ == "__main__":
    main()

