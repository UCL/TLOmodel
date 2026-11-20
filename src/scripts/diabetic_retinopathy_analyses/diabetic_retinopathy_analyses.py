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

# Simulation settings
start_date = Date(2010, 1, 1)
end_date = Date(2032, 1, 1)  # Includes December 2041

popsize = 1000
seed = 0

# Reporting years
BASELINE_YEAR = 2026
FORECAST_END_YEAR = 2031 # 2041

SELECTED_YEARS = [
    2026,
    2028, # 2031
    2030, # 2036
    2031, # 2041
]

# Years included in cumulative incident vision-loss analysis
INCIDENCE_START_YEAR = 2027
INCIDENCE_END_YEAR = 2031 # 2041

# Number of simulation replicates
N_REPLICATES = 5

# Percentiles for uncertainty intervals
LOWER_PERCENTILE = 2.5
UPPER_PERCENTILE = 97.5

CALIBRATION_FILE = (
    resourcefilepath
    / "ResourceFile_Diabetic_Retinopathy"
    / "diabetic_retinopathy_calibration_targets.csv"
)

def run_sim(service_availability, run_name="baseline", simulation_seed=seed):
    # log_config = {
    #     'filename': 'LogFile',
    #     'directory': outputpath
    # }

    # sim = Simulation(start_date=start_date, seed=0,
    #                  log_config={"filename": "LogFile"}, resourcefilepath=resourcefilepath)
    logfile_name = f"LogFile_{run_name}_{datestamp}"
    sim = Simulation(
        start_date=start_date,
        seed=simulation_seed,
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



def run_ensemble(service_availability, n_replicates=N_REPLICATES):
    """Run multiple simulations and collect monthly and annual results."""

    monthly_results = []
    annual_results = []

    for replicate in range(n_replicates):

        simulation_seed = seed + replicate

        print(
            f"\nRunning replicate {replicate + 1}/{n_replicates} "
            f"(seed={simulation_seed})"
        )

        logfile = run_sim(
            service_availability=service_availability,
            run_name=f"baseline_rep{replicate + 1}",
            simulation_seed=simulation_seed,
        )

        monthly = get_summary_stats(logfile)
        monthly = derive_analysis_metrics(monthly)

        annual = annualise_monthly_results(monthly)

        monthly_results.append(monthly)
        annual_results.append(annual)

    monthly_ensemble = pd.concat(
        monthly_results,
        keys=range(n_replicates),
        names=["replicate_id", "date"]
    )

    annual_ensemble = pd.concat(
        annual_results,
        keys=range(n_replicates),
        names=["replicate_id", "year"]
    )

    return monthly_ensemble, annual_ensemble


def summarise_ensemble(ensemble):
    """Calculate mean and percentile uncertainty intervals."""

    numeric_cols = ensemble.select_dtypes(include="number").columns.drop("replicate", errors="ignore")

    grouped = ensemble[numeric_cols].groupby(level="year")

    mean = grouped.mean()
    lower = grouped.quantile(LOWER_PERCENTILE / 100)
    upper = grouped.quantile(UPPER_PERCENTILE / 100)

    summary = pd.concat(
        {
            "mean": mean,
            "lower": lower,
            "upper": upper,
        },
        axis=1
    )

    return summary


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
        if c.startswith("inc_")
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

    df["total_vision_impaired"] = df["vision_loss_due_to_dr"]

    df["total_vision_severe_or_blindness"] = (
        df["severe_vision_loss_due_to_dr"]
    )

    df["total_vision_blindness_due_to_dr"] = (
        df["blindness_due_to_dr"]
    )

    # Disaggregate DR-attributable vision loss into mutually exclusive
    # categories. These are counts among people with diabetes.
    df["total_vision_moderate_due_to_dr"] = (
        df["total_vision_impaired"]
        - df["total_vision_severe_or_blindness"]
    )

    df["total_vision_severe_due_to_dr"] = (
        df["total_vision_severe_or_blindness"]
        - df["total_vision_blindness_due_to_dr"]
    )

    # DR among people with diabetes; DMO among
    # people with DR; vision impairment among people with diabetes.
    df["dr_prevalence_among_diabetics_pct"] = safe_percent(
        df["total_dr_any"], df["total_alive_diabetics"]
    )

    # DR stage prevalence among people with diabetes.
    df["dr_mild_or_moderate_among_diabetics_pct"] = safe_percent(
        df["total_dr_mild_or_moderate"],
        df["total_alive_diabetics"],
    )

    df["dr_severe_among_diabetics_pct"] = safe_percent(
        df["total_dr_severe"],
        df["total_alive_diabetics"],
    )

    df["dr_proliferative_among_diabetics_pct"] = safe_percent(
        df["total_dr_proliferative"],
        df["total_alive_diabetics"],
    )

    df["dmo_prevalence_among_dr_pct"] = safe_percent(
        df["total_dmo_any"], df["total_dr_any"]
    )

    # DMO categories among people with DR.
    df["dmo_non_clinically_significant_among_dr_pct"] = safe_percent(
        df["total_dmo_non_clinically_significant"],
        df["total_dr_any"],
    )

    df["dmo_clinically_significant_among_dr_pct"] = safe_percent(
        df["total_dmo_clinically_significant"],
        df["total_dr_any"],
    )


    df["vision_impairment_among_diabetics_pct"] = safe_percent(
        df["total_vision_impaired"], df["total_alive_diabetics"],
    )

    df["severe_vision_or_blindness_among_diabetics_pct"] = safe_percent(
        df["total_vision_severe_or_blindness"], df["total_alive_diabetics"],
    )

    df["blindness_among_diabetics_pct"] = safe_percent(
        df["total_vision_blindness_due_to_dr"], df["total_alive_diabetics"],
    )

    # Mutually exclusive vision outcomes among people with diabetes.
    df["vision_moderate_among_diabetics_pct"] = safe_percent(
        df["total_vision_moderate_due_to_dr"],
        df["total_alive_diabetics"],
    )

    df["vision_severe_among_diabetics_pct"] = safe_percent(
        df["total_vision_severe_due_to_dr"],
        df["total_alive_diabetics"],
    )

    df["total_vision_normal"] = (
        df["total_alive_diabetics"] - df["total_vision_impaired"]
    )

    # Cumulative incidence is not reconstructed by summing monthly incidence
    # blindly across calendar years; the logged monthly counts are summed below.
    return df


def prepare_calibration_data(ensemble_summary):
    """
    Match every published calibration target to the model estimate
    from exactly the same calendar year.

    Model estimates are ensemble means, with percentile intervals
    calculated across simulation replicates.
    """

    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {CALIBRATION_FILE}"
        )

    targets = pd.read_csv(CALIBRATION_FILE)

    required = {
        "metric",
        "label",
        "observed_value",
        "year",
    }

    missing = required - set(targets.columns)

    if missing:
        raise ValueError(
            f"Calibration CSV is missing columns: {missing}"
        )

    targets["year"] = pd.to_numeric(
        targets["year"],
        errors="raise"
    ).astype(int)

    # Ensure all requested model years are available.
    available_years = set(ensemble_summary.index)

    missing_years = sorted(
        set(targets["year"]) - available_years
    )

    if missing_years:
        raise ValueError(
            "The simulation does not contain the following "
            f"calibration years: {missing_years}"
        )

    results = []

    for _, target in targets.iterrows():

        metric = target["metric"]
        year = target["year"]

        if ("mean", metric) not in ensemble_summary.columns:
            raise KeyError(
                f"Metric '{metric}' is not available in "
                "the model summary."
            )

        row = target.to_dict()

        # IMPORTANT: use the observed year's corresponding model year.
        row["model_estimate"] = ensemble_summary.at[
            year, ("mean", metric)
        ]

        row["model_lower"] = ensemble_summary.at[
            year, ("lower", metric)
        ]

        row["model_upper"] = ensemble_summary.at[
            year, ("upper", metric)
        ]

        # Additional model estimate for the baseline year.
        # Panels A-C use this value to compare all observed targets
        # against the model in 2026.
        if BASELINE_YEAR not in ensemble_summary.index:
            raise ValueError(
                f"Baseline year {BASELINE_YEAR} is not available "
                "in the ensemble summary."
            )

        if ("mean", metric) in ensemble_summary.columns:
            row["model_baseline_estimate"] = ensemble_summary.at[
                BASELINE_YEAR, ("mean", metric)
            ]

            row["model_baseline_lower"] = ensemble_summary.at[
                BASELINE_YEAR, ("lower", metric)
            ]

            row["model_baseline_upper"] = ensemble_summary.at[
                BASELINE_YEAR, ("upper", metric)
            ]
        else:
            row["model_baseline_estimate"] = pd.NA
            row["model_baseline_lower"] = pd.NA
            row["model_baseline_upper"] = pd.NA

        row["difference_percentage_points"] = (
            row["model_estimate"] - row["observed_value"]
        )

        results.append(row)

    comparison = pd.DataFrame(results)

    return comparison



def plot_calibration(comparison):
    """
    Figure 3.1:
    A-C: Observed targets versus model estimates in the baseline year.
    D: All calibration targets, retaining the existing point-and-error-bar plot.
    """

    # Metrics used in panels A-C.
    panels = {
        "A. DR prevalence": [
            ("Any DR", "dr_prevalence_among_diabetics_pct"),
            ("Mild/moderate", "dr_mild_or_moderate_among_diabetics_pct"),
            ("Severe", "dr_severe_among_diabetics_pct"),
            ("PDR", "dr_proliferative_among_diabetics_pct"),
        ],

        "B. DMO prevalence": [
            ("Any DMO", "dmo_prevalence_among_dr_pct"),
            ("Non-CS DMO", "dmo_non_clinically_significant_among_dr_pct"),
            ("CS DMO", "dmo_clinically_significant_among_dr_pct"),
        ],

        "C. Vision impairment": [
            ("Moderate", "vision_moderate_among_diabetics_pct"),
            ("Severe", "vision_severe_among_diabetics_pct"),
            ("Blind", "blindness_among_diabetics_pct"),
        ],
    }

    fig, axes = plt.subplots(
        2, 2,
        figsize=(16, 11),
    )

    # ---------------------------------------------------------
    # Panels A-C: grouped bar plots
    # ---------------------------------------------------------

    for ax, (title, categories) in zip(
        [axes[0, 0], axes[0, 1], axes[1, 0]],
        panels.items(),
    ):

        observed_values = []
        model_values = []
        category_labels = []
        observed_lower = []
        observed_upper = []
        model_lower = []
        model_upper = []

        for label, metric in categories:

            rows = comparison.loc[
                comparison["metric"] == metric
            ]

            if rows.empty:
                print(
                    f"WARNING: No calibration target found for "
                    f"'{metric}'. This category will be omitted "
                    f"from panel {title}."
                )
                continue

            row = rows.iloc[0]

            if pd.isna(row.get("model_baseline_estimate")):
                print(
                    f"WARNING: No 2026 model estimate available "
                    f"for '{metric}'."
                )
                continue

            category_labels.append(label)
            observed_values.append(row["observed_value"])
            model_values.append(row["model_baseline_estimate"])

            # Observed uncertainty intervals.
            observed_lower.append(
                row["observed_value"] - row["lower"]
                if pd.notna(row.get("lower"))
                else 0
            )
            observed_upper.append(
                row["upper"] - row["observed_value"]
                if pd.notna(row.get("upper"))
                else 0
            )

            # Model uncertainty intervals.
            model_lower.append(
                row["model_baseline_estimate"] - row["model_baseline_lower"]
                if pd.notna(row.get("model_baseline_lower"))
                else 0
            )
            model_upper.append(
                row["model_baseline_upper"] - row["model_baseline_estimate"]
                if pd.notna(row.get("model_baseline_upper"))
                else 0
            )

        if not category_labels:
            ax.set_visible(False)
            continue

        positions = range(len(category_labels))
        bar_width = 0.36

        # Observed targets.
        ax.bar(
            [p - bar_width / 2 for p in positions],
            observed_values,
            width=bar_width,
            color="lightgrey",
            edgecolor="black",
            linewidth=0.7,
            label="Observed target",
            zorder=3,
        )

        ax.errorbar(
            [p - bar_width / 2 for p in positions],
            observed_values,
            yerr=[observed_lower, observed_upper],
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
            zorder=4,
        )

        # Model estimates in the baseline year.
        ax.bar(
            [p + bar_width / 2 for p in positions],
            model_values,
            width=bar_width,
            color="tab:blue",
            edgecolor="black",
            linewidth=0.7,
            label=f"Model ({BASELINE_YEAR})",
            zorder=3,
        )

        ax.errorbar(
            [p + bar_width / 2 for p in positions],
            model_values,
            yerr=[model_lower, model_upper],
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
            zorder=4,
        )

        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Prevalence (%)")
        ax.set_xticks(list(positions))
        ax.set_xticklabels(category_labels)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)

    # ---------------------------------------------------------
    # Panel D: retain the existing all-targets plot
    # ---------------------------------------------------------

    ax = axes[1, 1]

    subset = comparison[
        comparison["metric"].isin(comparison["metric"].tolist())
    ].copy()

    subset = subset.reset_index(drop=True)

    for i, row in subset.iterrows():

        observed = row["observed_value"]

        if pd.notna(row.get("lower")) and pd.notna(row.get("upper")):
            ax.errorbar(
                observed,
                i,
                xerr=[[
                    observed - row["lower"]
                ], [
                    row["upper"] - observed
                ]],
                fmt="o",
                capsize=4,
                color="black",
                label="Observed" if i == 0 else None,
            )
        else:
            ax.plot(
                observed,
                i,
                "o",
                color="black",
                label="Observed" if i == 0 else None,
            )

        model = row["model_estimate"]
        model_lower = row["model_lower"]
        model_upper = row["model_upper"]

        if pd.notna(model_lower) and pd.notna(model_upper):
            ax.errorbar(
                model,
                i,
                xerr=[[
                    max(0, model - model_lower)
                ], [
                    max(0, model_upper - model)
                ]],
                fmt="s",
                capsize=4,
                color="tab:blue",
                label="Model" if i == 0 else None,
            )
        else:
            ax.plot(
                model,
                i,
                "s",
                color="tab:blue",
                label="Model" if i == 0 else None,
            )

    labels = [
        f"{row['label']} ({row['year']})"
        for _, row in subset.iterrows()
    ]

    ax.set_yticks(range(len(subset)))
    ax.set_yticklabels(labels)
    ax.set_title("D. All calibration targets", fontweight="bold")
    ax.set_xlabel("Prevalence (%)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)

    fig.suptitle(
        "Model estimates compared with published observations",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        outputpath / f"figure_3_1_calibration_{datestamp}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return comparison




def export_analysis_workbook(
    monthly,
    annual,
    calibration=None,
    decomposition=None,
    care_cascade=None,
    vision_loss=None,
    cumulative_vision_loss=None,
    table_3_2=None,
    ensemble_summary=None,
):
    """Export all analysis datasets into one Excel workbook."""

    selected_years = annual.loc[
        annual.index.intersection(SELECTED_YEARS)
    ].copy()

    output_file = outputpath / f"diabetic_retinopathy_analysis_{datestamp}.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

        monthly.to_excel(
            writer,
            sheet_name="Monthly results"
        )

        annual.to_excel(
            writer,
            sheet_name="Annual results"
        )

        selected_years.to_excel(
            writer,
            sheet_name="Selected years"
        )

        if calibration is not None:
            calibration.to_excel(
                writer,
                sheet_name="Calibration",
                index=False
            )

        if decomposition is not None:
            decomposition.to_excel(
                writer,
                sheet_name="Burden decomposition",
                index=False
            )

        if care_cascade is not None:
            care_cascade.to_excel(
                writer,
                sheet_name="Care cascade",
                index=False
            )

        if vision_loss is not None:
            vision_loss.to_excel(
                writer,
                sheet_name="Annual vision loss",
                index=False
            )

        if cumulative_vision_loss is not None:
            cumulative_vision_loss.to_excel(
                writer,
                sheet_name="Cumulative vision loss",
                index=False
            )

        if table_3_2 is not None:
            table_3_2.to_excel(
                writer,
                sheet_name="Table 3.2 data"
            )

        if ensemble_summary is not None:
            ensemble_summary.to_excel(
                writer,
                sheet_name="Uncertainty intervals"
            )

    print(f"\nExcel workbook saved to: {output_file.resolve()}")

    return output_file


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


def plot_projection(ensemble_summary):
    """Generate Figure 3.2: DR, DMO and vision outcomes over time."""

    figures = {
        "A. Diabetic retinopathy": [
            "total_dr_mild_or_moderate",
            "total_dr_severe",
            "total_dr_proliferative",
        ],

        "B. Diabetic macular oedema": [
            "total_dmo_non_clinically_significant",
            "total_dmo_clinically_significant",
        ],

        "C. Vision impairment": [
            "total_vision_moderate_vision_impairment",
            "total_vision_severe_vision_impairment",
            "total_vision_blindness",
        ],

        "D. Prevalence among people with diabetes": [
            "dr_prevalence_among_diabetics_pct",
            "dmo_prevalence_among_dr_pct",
            "vision_impairment_among_diabetics_pct",
        ],
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    for ax, (title, columns) in zip(axes.flatten(), figures.items()):

        for col in columns:

            if ("mean", col) not in ensemble_summary.columns:
                continue

            years = ensemble_summary.index

            mean = ensemble_summary[("mean", col)]
            lower = ensemble_summary[("lower", col)]
            upper = ensemble_summary[("upper", col)]

            ax.plot(
                years,
                mean,
                marker="o",
                markersize=3,
                label=col.replace("total_", "").replace("_", " ").replace("_pct", ""),
            )

            ax.fill_between(
                years,
                lower,
                upper,
                alpha=0.15
            )

        ax.set_title(title)
        ax.set_xlabel("Year")

        if title == "D. Prevalence among people with diabetes":
            ax.set_ylabel("Prevalence (%)")
        else:
            ax.set_ylabel("Number of people")

        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

        ax.set_xlim(BASELINE_YEAR, FORECAST_END_YEAR)

    fig.suptitle(
        "Diabetic retinopathy and vision outcomes: "
        "current burden and 15-year projection",
        fontsize=14
    )

    fig.tight_layout()

    fig.savefig(
        outputpath / f"figure_3_2_projection_{datestamp}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# def make_pathway_summary(annual):
#     """
#     Summarise the model's recorded first-diagnosis pathways. These are
#     descriptive groups, not randomised/comparable cohorts.
#     """
#     rows = []
#     for pathway in ("screening", "symptomatic"):
#         for year in (FORECAST_BASELINE_YEAR, FORECAST_END_YEAR):
#             if year not in annual.index:
#                 continue
#             row = {"year": year, "pathway": pathway}
#             for col in (
#                 f"n_diagnosed_{pathway}",
#                 f"n_on_treatment_{pathway}",
#                 f"total_dr_mild_or_moderate_{pathway}",
#                 f"total_dr_severe_{pathway}",
#                 f"total_dr_proliferative_{pathway}",
#                 f"total_vision_moderate_vision_impairment_{pathway}",
#                 f"total_vision_severe_vision_impairment_{pathway}",
#                 f"total_vision_blindness_{pathway}",
#             ):
#                 row[col] = annual.at[year, col] if col in annual.columns else pd.NA
#             rows.append(row)
#     result = pd.DataFrame(rows)
#     result.to_csv(outputpath / f"pathway_summary_{datestamp}.csv", index=False)
#     return result


# def make_forecast_summary(annual):
#     """Compare the (baseline) 2012 model state with 2014 (forecast end year) and quantify absolute changes."""
#     requested = [
#         "total_alive_diabetics",
#         "total_dr_mild_or_moderate",
#         "total_dr_severe",
#         "total_dr_proliferative",
#         "total_dr_any",
#         "total_dmo_any",
#         "total_vision_moderate_vision_impairment",
#         "total_vision_severe_vision_impairment",
#         "total_vision_blindness",
#         "dr_prevalence_among_diabetics_pct",
#         "vision_impairment_among_diabetics_pct",
#         "severe_vision_or_blindness_among_diabetics_pct",
#         "blindness_among_diabetics_pct",
#     ]
#     years = [y for y in (FORECAST_BASELINE_YEAR, FORECAST_END_YEAR) if y in annual.index]
#     if len(years) < 2:
#         print(f"WARNING: Need annual outputs for both {FORECAST_BASELINE_YEAR} and "
#               f"{FORECAST_END_YEAR}; found {years}. Check simulation end date.")
#         return pd.DataFrame()
#     summary = annual.loc[years, [c for c in requested if c in annual.columns]].T
#     summary.columns = [f"value_{y}" for y in years]
#     summary["absolute_change"] = summary.iloc[:, 1] - summary.iloc[:, 0]
#     summary["percent_change"] = (
#         summary["absolute_change"] / summary.iloc[:, 0].replace(0, pd.NA) * 100
#     )
#     summary.to_csv(outputpath / f"forecast_2012_to_2014_{datestamp}.csv")
#     return summary


def make_table_3_2(annual):
    """Prepare the underlying data for Table 3.2."""

    available_years = [
        year for year in SELECTED_YEARS
        if year in annual.index
    ]

    columns = [
        "total_alive_diabetics",

        # DR
        "total_dr_mild_or_moderate",
        "total_dr_severe",
        "total_dr_proliferative",
        "total_dr_any",
        "dr_prevalence_among_diabetics_pct",

        # DMO
        "total_dmo_non_clinically_significant",
        "total_dmo_clinically_significant",
        "total_dmo_any",
        "dmo_prevalence_among_dr_pct",

        # Vision
        "total_vision_normal",
        "total_vision_moderate_vision_impairment",
        "total_vision_severe_vision_impairment",
        "total_vision_blindness",
        "vision_impairment_among_diabetics_pct",

        # Incidence
        "inc_dr_mild_or_moderate",
        "inc_dr_severe",
        "inc_dr_proliferative",
        "inc_vision_moderate_vision_impairment",
        "inc_vision_severe_vision_impairment",
        "inc_vision_blindness",
    ]

    available_columns = [
        col for col in columns if col in annual.columns
    ]

    table = annual.loc[
        available_years,
        available_columns
    ].T

    table.columns = available_years

    return table




def make_burden_decomposition(annual):
    """Decompose growth in vision impairment between 2026 and 2041."""

    required_years = [BASELINE_YEAR, FORECAST_END_YEAR]

    if not all(year in annual.index for year in required_years):
        print("Missing baseline or forecast-end year.")
        return None

    baseline = annual.loc[BASELINE_YEAR]
    future = annual.loc[FORECAST_END_YEAR]

    N0 = baseline["total_alive_diabetics"]
    N1 = future["total_alive_diabetics"]

    p0 = baseline["vision_impairment_among_diabetics_pct"] / 100
    p1 = future["vision_impairment_among_diabetics_pct"] / 100

    initial_burden = N0 * p0

    diabetes_growth = (N1 - N0) * p0

    prevalence_growth = N1 * (p1 - p0)

    final_burden = N1 * p1

    result = pd.DataFrame({
        "component": [
            "Vision impairment in 2026",
            "Growth due to diabetes population",
            "Growth due to changing impairment prevalence",
            "Vision impairment in 2041",
        ],
        "value": [
            initial_burden,
            diabetes_growth,
            prevalence_growth,
            final_burden,
        ]
    })

    # Verify the decomposition
    assert abs(
        initial_burden + diabetes_growth
        + prevalence_growth - final_burden
    ) < 1e-6

    return result


def plot_burden_decomposition(decomposition):
    """Plot Figure 3.3: decomposition of future vision impairment."""

    if decomposition is None:
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(
        decomposition["component"],
        decomposition["value"],
        width=0.65,
    )

    ax.set_title(
        "Drivers of growth in diabetes-related vision impairment"
    )

    ax.set_ylabel("Number of people")
    ax.set_xlabel("")

    ax.tick_params(
        axis="x",
        labelrotation=15
    )

    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()

    fig.savefig(
        outputpath / f"figure_3_3_decomposition_{datestamp}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)



def make_care_cascade(annual):
    """Prepare the care cascade data."""

    df = annual.copy()

    diagnosed_cols = [
        "n_diagnosed_screening",
        "n_diagnosed_symptomatic",
    ]

    treatment_cols = [
        "n_on_treatment_screening",
        "n_on_treatment_symptomatic",
    ]

    if all(c in df.columns for c in diagnosed_cols):
        df["total_diagnosed_dr"] = df[diagnosed_cols].sum(axis=1)
    elif "total_diagnosed_dr" not in df.columns:
        print("WARNING: Diagnosed DR stock is not available.")

    if all(c in df.columns for c in treatment_cols):
        df["total_on_treatment_dr"] = df[treatment_cols].sum(axis=1)
    elif "total_on_treatment_dr" not in df.columns:
        print("WARNING: Treatment stock is not available.")

    required = [
        "total_alive_diabetics",
        "total_dr_any",
        "total_diagnosed_dr",
        "total_on_treatment_dr",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"Cannot construct complete care cascade. Missing: {missing}")
        return None

    return df[required].copy()



def plot_care_cascade(cascade):
    """
    Figure 3.4:
    Left: care cascade over time.
    Right: baseline versus forecast-year cascade.
    """

    if cascade is None:
        return

    required_years = [BASELINE_YEAR, FORECAST_END_YEAR]

    missing_years = [
        year for year in required_years
        if year not in cascade.index
    ]

    if missing_years:
        print(
            "Cannot create baseline/forecast cascade comparison. "
            f"Missing years: {missing_years}"
        )
        return

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(17, 7),
        gridspec_kw={"width_ratios": [1.15, 1]},
    )

    # ---------------------------------------------------------
    # Panel A: existing care-cascade time series
    # ---------------------------------------------------------

    labels = {
        "total_alive_diabetics": "People with diabetes",
        "total_dr_any": "People with DR",
        "total_diagnosed_dr": "People diagnosed",
        "total_on_treatment_dr": "People on treatment",
    }

    for col, label in labels.items():
        ax1.plot(
            cascade.index,
            cascade[col],
            marker="o",
            markersize=3,
            label=label,
        )

    ax1.set_title("A. Care cascade over time", fontweight="bold")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of people")
    ax1.set_xlim(BASELINE_YEAR, FORECAST_END_YEAR)
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, fontsize=9)

    # ---------------------------------------------------------
    # Panel B: baseline versus forecast-year cascade
    # ---------------------------------------------------------

    stages = [
        "total_alive_diabetics",
        "total_dr_any",
        "total_diagnosed_dr",
        "total_on_treatment_dr",
    ]

    stage_labels = [
        "Diabetes",
        "DR",
        "Diagnosed",
        "On treatment",
    ]

    baseline_values = cascade.loc[BASELINE_YEAR, stages].astype(float).values
    forecast_values = cascade.loc[FORECAST_END_YEAR, stages].astype(float).values

    y = range(len(stages))
    bar_height = 0.34

    ax2.barh(
        [v - bar_height / 2 for v in y],
        baseline_values,
        height=bar_height,
        color="lightgrey",
        edgecolor="black",
        linewidth=0.6,
        label=str(BASELINE_YEAR),
        zorder=3,
    )

    ax2.barh(
        [v + bar_height / 2 for v in y],
        forecast_values,
        height=bar_height,
        color="tab:blue",
        edgecolor="black",
        linewidth=0.6,
        label=str(FORECAST_END_YEAR),
        zorder=3,
    )

    # Calculate the percentage reaching each successive stage.
    def percentage_of_previous_step(values, stage_index):
        if stage_index == 0:
            return None

        denominator = values[stage_index - 1]

        if denominator == 0:
            return float("nan")

        return 100 * values[stage_index] / denominator

    max_value = max(
        max(baseline_values),
        max(forecast_values),
    )

    for values, offset in [
        (baseline_values, -bar_height / 2),
        (forecast_values, bar_height / 2),
    ]:

        for i, value in enumerate(values):

            percentage = percentage_of_previous_step(values, i)

            annotation = f"{value:,.0f}"

            if percentage is not None and pd.notna(percentage):
                annotation += f" ({percentage:.0f}% of prior step)"

            ax2.text(
                value + max_value * 0.015,
                i + offset,
                annotation,
                va="center",
                fontsize=8,
            )

    ax2.set_yticks(list(y))
    ax2.set_yticklabels(stage_labels)
    ax2.invert_yaxis()

    ax2.set_title(
        f"B. Care cascade: {BASELINE_YEAR} versus {FORECAST_END_YEAR}",
        fontweight="bold",
    )
    ax2.set_xlabel("Number of people")

    ax2.grid(axis="x", alpha=0.25)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False)

    # Leave space on the right for the numerical annotations.
    ax2.set_xlim(0, max_value * 1.55)

    fig.suptitle(
        "Diabetic retinopathy care cascade",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        outputpath / f"figure_3_4_care_cascade_{datestamp}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)




def make_annual_vision_loss(annual):
    """Prepare annual new severe vision impairment and blindness cases."""

    required = [
        "inc_vision_loss_severe_diagnosed",
        "inc_vision_loss_severe_undiagnosed",
        "inc_vision_loss_blindness_diagnosed",
        "inc_vision_loss_blindness_undiagnosed",
    ]

    missing = [c for c in required if c not in annual.columns]

    if missing:
        print(
            "Missing vision-loss diagnosis-status logging columns:",
            missing
        )
        return None

    df = annual[required].copy()

    df["severe_total"] = (
        df["inc_vision_loss_severe_diagnosed"]
        + df["inc_vision_loss_severe_undiagnosed"]
    )

    df["blindness_total"] = (
        df["inc_vision_loss_blindness_diagnosed"]
        + df["inc_vision_loss_blindness_undiagnosed"]
    )

    # Combined new severe vision impairment or blindness,
    # classified according to diagnosis status at the time of onset.
    df["new_severe_or_blindness_diagnosed"] = (
        df["inc_vision_loss_severe_diagnosed"]
        + df["inc_vision_loss_blindness_diagnosed"]
    )

    df["new_severe_or_blindness_undiagnosed"] = (
        df["inc_vision_loss_severe_undiagnosed"]
        + df["inc_vision_loss_blindness_undiagnosed"]
    )

    df["all_vision_loss_total"] = (
        df["new_severe_or_blindness_diagnosed"]
        + df["new_severe_or_blindness_undiagnosed"]
    )

    return df



def plot_annual_vision_loss(vision_loss):
    """
    Figure 3.5:
    A. Annual new severe vision impairment or blindness due to DR,
       split by diagnosis status at the time of onset.
    B. Share of new cases by diagnosis status.
    """

    if vision_loss is None:
        return

    required = [
        "new_severe_or_blindness_diagnosed",
        "new_severe_or_blindness_undiagnosed",
    ]

    missing = [
        col for col in required
        if col not in vision_loss.columns
    ]

    if missing:
        print(
            "Cannot create vision-loss figure. Missing columns:",
            missing,
        )
        return

    df = vision_loss.loc[
        INCIDENCE_START_YEAR:INCIDENCE_END_YEAR
    ].copy()

    if df.empty:
        print("No annual vision-loss data available for the selected years.")
        return

    years = df.index.to_numpy()

    diagnosed = df["new_severe_or_blindness_diagnosed"].fillna(0)
    undiagnosed = df["new_severe_or_blindness_undiagnosed"].fillna(0)

    total = diagnosed + undiagnosed

    # Calculate the annual percentage shares.
    diagnosed_share = (
        100 * diagnosed / total.replace(0, pd.NA)
    ).fillna(0)

    undiagnosed_share = (
        100 * undiagnosed / total.replace(0, pd.NA)
    ).fillna(0)

    # ---------------------------------------------------------
    # Create two panels
    # ---------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(16, 7),
        gridspec_kw={"width_ratios": [1.15, 1]},
    )

    # ---------------------------------------------------------
    # Panel A: stacked bars showing annual new cases
    # ---------------------------------------------------------

    ax1.bar(
        years,
        diagnosed,
        color="tab:blue",
        label="Diagnosed at the time",
        width=0.85,
    )

    ax1.bar(
        years,
        undiagnosed,
        bottom=diagnosed,
        color="tab:orange",
        label="Undiagnosed at the time",
        width=0.85,
    )

    ax1.set_title(
        "A. New severe VI or blindness due to DR",
        fontweight="bold",
    )

    ax1.set_xlabel("Year")
    ax1.set_ylabel("New cases per year")

    ax1.set_xlim(
        min(years) - 0.5,
        max(years) + 0.5,
    )

    ax1.grid(axis="y", alpha=0.25)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False)

    # ---------------------------------------------------------
    # Panel B: 100% stacked area showing annual shares
    # ---------------------------------------------------------

    ax2.stackplot(
        years,
        diagnosed_share.to_numpy(),
        undiagnosed_share.to_numpy(),
        labels=[
            "Diagnosed at the time",
            "Undiagnosed at the time",
        ],
        colors=[
            "tab:blue",
            "tab:orange",
        ],
        alpha=0.9,
    )

    ax2.set_title(
        "B. Share of new cases by diagnosed status",
        fontweight="bold",
    )

    ax2.set_xlabel("Year")
    ax2.set_ylabel(
        "% of new severe VI or blindness cases"
    )

    ax2.set_ylim(0, 100)
    ax2.set_xlim(min(years), max(years))

    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.grid(axis="y", alpha=0.25)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, loc="upper right")

    # ---------------------------------------------------------
    # Add labels to the area chart
    # ---------------------------------------------------------

    middle_index = len(years) // 2

    diagnosed_y = diagnosed_share.iloc[middle_index] / 2

    undiagnosed_y = (
        diagnosed_share.iloc[middle_index]
        + undiagnosed_share.iloc[middle_index] / 2
    )

    ax2.text(
        years[middle_index],
        diagnosed_y,
        "Diagnosed at the time",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=9,
    )

    ax2.text(
        years[middle_index],
        undiagnosed_y,
        "Undiagnosed at the time",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=9,
    )

    fig.suptitle(
        "New severe vision impairment or blindness due to diabetic retinopathy",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        outputpath / f"figure_3_5_annual_vision_loss_{datestamp}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def make_cumulative_vision_loss(annual_vision_loss):
    """Calculate cumulative vision-loss cases over 2027–2041."""

    if annual_vision_loss is None:
        return None

    df = annual_vision_loss.loc[
        INCIDENCE_START_YEAR:INCIDENCE_END_YEAR
    ]

    outcomes = {
        "Severe vision impairment": (
            "inc_vision_loss_severe_diagnosed",
            "inc_vision_loss_severe_undiagnosed",
        ),
        "Blindness": (
            "inc_vision_loss_blindness_diagnosed",
            "inc_vision_loss_blindness_undiagnosed",
        ),
    }

    rows = []

    for outcome, (diagnosed_col, undiagnosed_col) in outcomes.items():

        diagnosed = df[diagnosed_col].sum()
        undiagnosed = df[undiagnosed_col].sum()

        total = diagnosed + undiagnosed

        share_undiagnosed = (
            100 * undiagnosed / total if total > 0 else float("nan")
        )

        rows.append({
            "outcome": outcome,
            "total_new_cases_{INCIDENCE_START_YEAR}_{INCIDENCE_END_YEAR}": total,
            "diagnosed_at_time": diagnosed,
            "undiagnosed_at_time": undiagnosed,
            "share_undiagnosed_pct": share_undiagnosed,
        })

    # Combined outcome
    diagnosed_total = df[
        [
            "inc_vision_loss_severe_diagnosed",
            "inc_vision_loss_blindness_diagnosed",
        ]
    ].sum().sum()

    undiagnosed_total = df[
        [
            "inc_vision_loss_severe_undiagnosed",
            "inc_vision_loss_blindness_undiagnosed",
        ]
    ].sum().sum()

    total = diagnosed_total + undiagnosed_total

    rows.append({
        "outcome": "Severe impairment or blindness",
        "total_new_cases_{INCIDENCE_START_YEAR}_{INCIDENCE_END_YEAR}": total,
        "diagnosed_at_time": diagnosed_total,
        "undiagnosed_at_time": undiagnosed_total,
        "share_undiagnosed_pct": (
            100 * undiagnosed_total / total
            if total > 0 else float("nan")
        ),
    })

    return pd.DataFrame(rows)


def main():

    # 1. Run the ensemble
    monthly_ensemble, annual_ensemble = run_ensemble(
        service_availability=["*"],
        n_replicates=N_REPLICATES,
    )

    # 2. Calculate ensemble summaries
    ensemble_summary = summarise_ensemble(annual_ensemble)

    # Mean annual results
    annual = (
        annual_ensemble
        .groupby(level="year")
        .mean(numeric_only=True)
        .drop(columns=["replicate"], errors="ignore")
    )

    # Prepare year-matched calibration comparisons
    calibration = prepare_calibration_data(ensemble_summary)

    # Generate Figure 3.1
    plot_calibration(calibration)

    # 4. Figure 3.2: Burden and projections
    plot_projection(ensemble_summary)

    table_3_2 = make_table_3_2(annual)

    # 5. Figure 3.3: Burden decomposition
    decomposition = make_burden_decomposition(annual)

    plot_burden_decomposition(decomposition)

    # 6. Figure 3.4: Care cascade
    care_cascade = make_care_cascade(annual)

    plot_care_cascade(care_cascade)

    # 7. Figure 3.5: Annual new vision loss
    vision_loss = make_annual_vision_loss(annual)

    plot_annual_vision_loss(vision_loss)

    # 8. Table 3.4: Cumulative vision loss
    cumulative_vision_loss = make_cumulative_vision_loss(
        vision_loss
    )

    # 9. Save results to Excel
    export_analysis_workbook(
        monthly=monthly_ensemble,
        annual=annual,
        calibration=calibration,
        decomposition=decomposition,
        care_cascade=care_cascade,
        vision_loss=vision_loss,
        cumulative_vision_loss=cumulative_vision_loss,
        table_3_2=table_3_2,
        ensemble_summary=ensemble_summary,
    )

    # 10. Save ensemble uncertainty data separately
    ensemble_summary.to_excel(
        outputpath / f"ensemble_uncertainty_{datestamp}.xlsx"
    )

    # 11. Save Table 3.2 separately
    table_3_2.to_excel(
        outputpath / f"table_3_2_{datestamp}.xlsx"
    )

    # 12. Save figures' underlying data
    if decomposition is not None:
        decomposition.to_csv(
            outputpath / f"figure_3_3_data_{datestamp}.csv",
            index=False
        )

    if care_cascade is not None:
        care_cascade.to_csv(
            outputpath / f"figure_3_4_data_{datestamp}.csv"
        )

    if vision_loss is not None:
        vision_loss.to_csv(
            outputpath / f"figure_3_5_data_{datestamp}.csv"
        )

    if cumulative_vision_loss is not None:
        cumulative_vision_loss.to_excel(
            outputpath / f"table_3_4_{datestamp}.xlsx",
            index=False
        )

    print("\nAnalysis completed.")

    print("\nBurden decomposition:")
    print(decomposition)

    print("\nCare cascade:")
    print(care_cascade.tail() if care_cascade is not None else None)

    print("\nCumulative vision loss:")
    print(cumulative_vision_loss)

    print(f"\nAll outputs saved to: {outputpath.resolve()}")


if __name__ == "__main__":
    main()

