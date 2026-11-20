import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import t

from src.scripts.costing.cost_estimation import estimate_mnh_input_cost_of_scenarios, summarize_cost_data
from tlo import Date
from tlo.analysis.utils import (
    bin_hsi_event_details,
    compute_summary_statistics,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
)

from typing import Literal

import calendar

import pandas as pd

from tlo.methods.healthsystem import pool_capabilities_at_levels_1b_and_2

from matplotlib.ticker import FuncFormatter


# Get results file
outputspath = './outputs/sejjj49@ucl.ac.uk/'
resourcefilepath = Path("./resources")

# scenario = 'testing_scenario_73595'
scenario = 'ehp_scale_up_scenario-2026-10-27T154504Z'
results_folder= get_scenario_outputs(scenario, outputspath)[-1]
sim_start_year = 2026
sim_end_year = 2027

# Create a folder to store graphs (if it hasn't already been created when ran previously)
g_path = f'{outputspath}figures_{scenario}'

if not os.path.isdir(g_path):
        os.makedirs(f'{outputspath}figures_{scenario}')

# Get scenario details
info = get_scenario_info(results_folder)
draws = [x for x in range(info['number_of_draws'])]

modelled_pop = 40_000

# We estimate pregnancies in 2026 using proportion of pregnancies ending in live births from Polis et al (2015) and
# UN population props live birth estimates (this is an approximation)
p_scaling_factor =  1 / (modelled_pop / 1_010_000)

int_analysis = ['baseline',
                'abortion',
                'mat_sepsis_cm',
                'pph_cm',
                'ol_cm',
                'spe_ec_cm',
                'cs_surg',
                'neo_sep_cm',
                'preterm_cm',
                'neo_resus',
                'all_mat_cm',
                'all_neo_cm',
                'all_cm'
                ]

scenario_names = ['Status quo',
                  'Post-abortion and ectopic pregnancy case management',
                  'Maternal sepsis case management',
                  'Postpartum haemorrhage case management',
                  'Obstructed labour case management',
                  'Severe pre-eclampsia/eclampsia case management',
                  'Caesarean section/other obstetric surgery',
                  'Neonatal sepsis case management',
                  'Preterm birth case management',
                  'Newborn resuscitation',
                  'All maternal interventions',
                  'All neonatal interventions',
                  'All interventions'
                  ]

draw_labels = {1: 'Abortion CM',
                   2:'Maternal sepsis CM',
                   3: 'Haemorrhage CM',
                   4: 'Obstructed labour CM',
                   5: 'Severe pre-eclampsia CM',
                   6: 'CS/Surgery',
                   7: 'Neonatal sepsis CM',
                   8: 'Preterm birth CM',
                   9: 'Newborn resus.',
                  10: 'Maternal ints.',
                  11: 'Newborn ints.',
                  12: 'All ints.'}

#  ======================================= DEFINE HELPER FUNCTIONS  ===================================================
def summarize_confidence_intervals(results: pd.DataFrame) -> pd.DataFrame:
    """Utility function to compute summary statistics

    Finds mean value and 95% interval across the runs for each draw.
    """

    # Calculate summary statistics
    grouped = results.groupby(axis=1, by='draw', sort=False)
    mean = grouped.mean()
    sem = grouped.sem()  # Standard error of the mean

    # Calculate the critical value for a 95% confidence level
    n = grouped.size().max()  # Assuming the largest group size determines the degrees of freedom
    critical_value = t.ppf(0.975, df=n - 1)  # Two-tailed critical value

    # Compute the margin of error
    margin_of_error = critical_value * sem

    # Compute confidence intervals
    lower = mean - margin_of_error
    upper = mean + margin_of_error

    # Combine into a single DataFrame
    summary = pd.concat({'mean': mean, 'lower': lower, 'upper': upper}, axis=1)

    # Format the DataFrame as in the original code
    summary.columns = summary.columns.swaplevel(1, 0)
    summary.columns.names = ['draw', 'stat']
    summary = summary.sort_index(axis=1)

    return summary

def get_ps_data_frames(key, results_folder):
    def sort_df(_df):
        _x = _df.drop(columns=['date'], inplace=False)
        return _x.iloc[0]

    results_df = extract_results(
                results_folder,
                module="tlo.methods.pregnancy_supervisor",
                key=key,
                custom_generate_series=sort_df,
                do_scaling=False
            )
    results_df_summ = summarize_confidence_intervals(results_df)

    return {'crude':results_df, 'summarised':results_df_summ}

def get_deaths_dalys_demog(group, multiplier):
    direct_deaths = extract_results(
                results_folder,
                module="tlo.methods.demography",
                key="death",
                custom_generate_series=(
                    lambda df: df.loc[(df['label'] == f'{group} Disorders')].assign(
                        year=df['date'].dt.year).groupby(['year'])['year'].count()),
                do_scaling=False)

    br = extract_results(
                results_folder,
                module="tlo.methods.demography",
                key="on_birth",
                custom_generate_series=(
                    lambda df: df.assign(
                        year=df['date'].dt.year).groupby(['year'])['year'].count()),
                do_scaling=False
            )

    dd_sum = summarize_confidence_intervals(direct_deaths)
    dd_mr = (direct_deaths/br) * multiplier
    dd_mr_sum = summarize_confidence_intervals(dd_mr)

    all_dalys_dfs = extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns=['date', 'sex', 'age_range']).groupby(['year']).sum().stack()),
            do_scaling=False)

    disorders_all = all_dalys_dfs.loc[(slice(None), f'{group} Disorders'), :]

    dalys_df = disorders_all.loc[sim_start_year]
    dalys_df_sum = summarize_confidence_intervals(dalys_df)

    return [direct_deaths, dd_sum, dd_mr, dd_mr_sum, dalys_df, dalys_df_sum]

#  ========================================== EXTRACT CORE DATA  ======================================================

results = {k:get_ps_data_frames(k, results_folder) for k in
           ['mat_comp_incidence', 'nb_comp_incidence', 'deaths_and_stillbirths','service_coverage', 'met_need',
            'yearly_mnh_counter_dict', 'intervention_coverage']}

#  ======================================= FIGURE 1 - MET NEED  =====================================================
met_need_df = results['met_need']['summarised']

def produce_fig_1(df):

    scenarios = [
        {
            "name": "Abortion/Ectopic \nCM",
            "draw": 1,
            "items": [
                ("pac_ep", "Post-abortion care")
            ]
        },
        {
            "name": "Maternal sepsis \nCM",
            "draw": 2,
            "items": [
                ("m_sepsis_cm", "Sepsis management")
            ]
        },
        {
            "name": "Maternal haem.\nCM",
            "draw": 3,
            "items": [
                # ("haem_cm_amtsl", "AMTSL")
                ("haem_cm_ut", "Uterotonics"),
                ("haem_cm_mrp", "MRRP"),
                ("haem_cm_blood_pph", "Blood (PPH)†"),
                ("heam_cm_blood_aph", "Blood (APH)")
            ]
        },
        {
            "name": "Obstructed labour\nCM",
            "draw": 4,
            "items": [
                ("ol_cm", "AVD"),
            ]
        },
        {
            "name": "Eclampsia/SPE CM",
            "draw": 5,
            "items": [
                ("ec_cm_mgso4", "Magnesium sulphate (E)"),
                ("spe_cm_mgso4", "Magnesium sulphate (SPE)"),
                ("spe_ec_cm_htns", "Antihypertensives")
            ]
        },
        {
            "name": "CS & Obstetric\nsurgery",
            "draw": 6,
            "items": [
                ("cs_surg_aph", "CS and IP surgery†"),
                ("cs_surg_pph", "PP surgery†")
            ]
        },
        {
            "name": "Neonatal sepsis CM",
            "draw": 7,
            "items": [
                ("n_sepsis_cm", "Sepsis management")
            ]
        },
        {
            "name": "Preterm birth CM",
            "draw": 8,
            "items": [
                ("ptb_cm_resus", "Resuscitation"),
                ("ptb_cm_sepsis", "Sepsis management"),
                ("ptb_cm_kmc", "KMC")
            ]
        },
        {
            "name": "Newborn Resus.",
            "draw": 9,
            "items": [
                ("neo_resus", "Resuscitation")
            ]
        }
    ]

    # ---------------------------------------------------------
    # Plot settings
    # ---------------------------------------------------------

    baseline_colour = "#BDBDBD"
    increased_colour = "#377EB8"

    bar_width = 0.36
    component_spacing = 1.0

    # Slightly increase spacing between packages
    package_gap = 1.05

    # Validate DataFrame structure
    required_stats = {"lower", "mean", "upper"}

    available_draws = set(
        df.columns.get_level_values("draw")
    )

    available_stats = set(
        df.columns.get_level_values("stat")
    )

    if 0 not in available_draws:
        raise KeyError(
            "Baseline draw 0 is not present in the DataFrame."
        )

    if not required_stats.issubset(available_stats):
        raise ValueError(
            f"Missing required statistics: "
            f"{required_stats - available_stats}"
        )

    # ---------------------------------------------------------
    # Extract results and calculate positions
    # ---------------------------------------------------------

    plot_data = []
    package_centres = []
    package_boundaries = []

    current_x = 0.0

    for scenario_number, scenario in enumerate(scenarios):

        draw = scenario["draw"]

        if draw not in available_draws:
            raise KeyError(
                f"Draw {draw} for '{scenario['name']}' is not present."
            )

        package_positions = []

        for component_number, (row_id, item_label) in enumerate(
            scenario["items"]
        ):

            if row_id not in df.index:
                raise KeyError(
                    f"Row '{row_id}' from '{scenario['name']}' "
                    f"is not present in the DataFrame."
                )

            # Baseline
            baseline_mean = df.loc[row_id, (0, "mean")]
            baseline_lower = df.loc[row_id, (0, "lower")]
            baseline_upper = df.loc[row_id, (0, "upper")]

            # Increased coverage
            increased_mean = df.loc[row_id, (draw, "mean")]
            increased_lower = df.loc[row_id, (draw, "lower")]
            increased_upper = df.loc[row_id, (draw, "upper")]

            plot_data.append({
                "package": scenario["name"],
                "component": item_label,
                "x": current_x,

                "baseline_mean": baseline_mean,
                "baseline_lower_error":
                    baseline_mean - baseline_lower,
                "baseline_upper_error":
                    baseline_upper - baseline_mean,

                "increased_mean": increased_mean,
                "increased_lower_error":
                    increased_mean - increased_lower,
                "increased_upper_error":
                    increased_upper - increased_mean
            })

            package_positions.append(current_x)

            current_x += component_spacing

        # Centre package title beneath all its components
        package_centres.append({
            "name": scenario["name"],
            "x": np.mean(package_positions)
        })

        # Boundary between packages
        if scenario_number < len(scenarios) - 1:
            package_boundaries.append(
                current_x
                - component_spacing / 2
                + package_gap / 2
            )

        current_x += package_gap

    # ---------------------------------------------------------
    # Create plot
    # ---------------------------------------------------------

    fig_width = max(
        16,
        len(plot_data) * 1.25
    )

    fig, ax = plt.subplots(
        figsize=(fig_width, 8)
    )

    upper_limits = []

    # ---------------------------------------------------------
    # Plot bars
    # ---------------------------------------------------------

    for item in plot_data:

        x_position = item["x"]

        # -------------------------
        # Baseline: no hatch
        # -------------------------

        baseline_bar = ax.bar(
            x_position - bar_width / 2,
            item["baseline_mean"],
            width=bar_width,
            yerr=np.array([
                [item["baseline_lower_error"]],
                [item["baseline_upper_error"]]
            ]),
            capsize=3,
            color=baseline_colour,
            edgecolor="black",
            linewidth=0.6,
            error_kw={
                "elinewidth": 1
            }
        )

        # -------------------------
        # Increased coverage:
        # dot hatch only
        # -------------------------

        increased_bar = ax.bar(
            x_position + bar_width / 2,
            item["increased_mean"],
            width=bar_width,
            yerr=np.array([
                [item["increased_lower_error"]],
                [item["increased_upper_error"]]
            ]),
            capsize=3,
            color=increased_colour,
            edgecolor="black",
            linewidth=0.6,
            hatch="..",
            error_kw={
                "elinewidth": 1
            }
        )

        # Percentage labels
        ax.bar_label(
            baseline_bar,
            labels=[
                f'{item["baseline_mean"]:.1f}%'
            ],
            padding=7,
            fontsize=8,
            rotation=90
        )

        ax.bar_label(
            increased_bar,
            labels=[
                f'{item["increased_mean"]:.1f}%'
            ],
            padding=7,
            fontsize=8,
            rotation=90
        )

        upper_limits.extend([
            item["baseline_mean"]
            + item["baseline_upper_error"],

            item["increased_mean"]
            + item["increased_upper_error"]
        ])

    # ---------------------------------------------------------
    # Component labels
    # ---------------------------------------------------------

    ax.set_xticks([
        item["x"]
        for item in plot_data
    ])

    ax.set_xticklabels(
        [
            item["component"]
            for item in plot_data
        ],
        rotation=45,
        ha="right",
        fontsize=9
    )

    # Add some space between axis and component labels
    ax.tick_params(
        axis="x",
        pad=4
    )

    # ---------------------------------------------------------
    # Package/group titles
    # ---------------------------------------------------------

    # Move titles further down so they don't overlap
    # the rotated component labels.
    package_label_y = -0.34

    for package in package_centres:

        ax.text(
            package["x"],
            package_label_y,
            package["name"],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            linespacing=1.1,
            clip_on=False
        )

    # ---------------------------------------------------------
    # Package separators
    # ---------------------------------------------------------

    for boundary in package_boundaries:

        ax.axvline(
            boundary,
            color="0.82",
            linewidth=0.9,
            linestyle="--",
            zorder=0
        )

    # ---------------------------------------------------------
    # Legend
    # ---------------------------------------------------------

    legend_handles = [
        Patch(
            facecolor=baseline_colour,
            edgecolor="black",
            label="Baseline coverage"
        ),
        Patch(
            facecolor=increased_colour,
            edgecolor="black",
            hatch="..",
            label="Increased coverage"
        )
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper right"
    )

    # ---------------------------------------------------------
    # General formatting
    # ---------------------------------------------------------

    ax.set_ylabel("Coverage (%)")
    ax.set_xlabel("")

    ax.set_ylim(
        0,
        max(upper_limits) * 1.20
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # More room underneath for both label levels
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.96,
        bottom=0.40
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    fig.text(
        0.07, 0.025,
        "† Denominator approximated: clinical indication cannot be determined "
        "for women who do not seek care.",
        ha="left",
        va="bottom",
        fontsize=9
    )

    plt.savefig(
        f"{g_path}/figure1_met_need.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.show()

produce_fig_1()

#  ======================================= FIGURE 2 - HEALTH IMPACTS  =================================================
# TODO: review if we want to look at all cause maternal deaths and all cause neonatal deaths

TARGET_PERIOD = (Date(sim_start_year, 1, 1), Date(sim_start_year, 12, 31))

# 1.) MATERNAL DEATHS AVERTED
def get_num_deaths_by_cause_label(_df):
    """Return total number of Deaths by label (total by age-group within the TARGET_PERIOD)
    """
    return _df \
        .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
        .groupby(_df['label']) \
        .size()

# 1.) Maternal deaths (direct)
num_deaths_by_cause_label = extract_results(
            results_folder,
            module='tlo.methods.demography',
            key='death',
            custom_generate_series=get_num_deaths_by_cause_label,
            do_scaling=False
        )

num_deaths_by_cause_label = num_deaths_by_cause_label.fillna(0)
num_deaths_by_cause_label = num_deaths_by_cause_label * p_scaling_factor

direct_deaths = num_deaths_by_cause_label.loc['Maternal Disorders'].reindex(num_deaths_by_cause_label.columns).to_frame().T
direct_deaths.rename(index={"Maternal Disorders": sim_start_year}, inplace=True)

baseline = direct_deaths.xs(0, axis=1, level="draw")
direct_mat_deaths_averted_df = baseline.sub(
    direct_deaths,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
direct_mat_deaths_averted_df.rename(index={sim_start_year: "mat_direct_deaths_averted"}, inplace=True)

# 2.) Maternal DALYs (direct)
dalys_by_cause = extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns=['date', 'sex', 'age_range']).groupby(['year']).sum().stack()),
            do_scaling=False)
dalys_by_cause = dalys_by_cause.loc[dalys_by_cause.index.get_level_values(0) != sim_end_year]

mat_dalys = dalys_by_cause.loc[sim_start_year, 'Maternal Disorders'].reindex(dalys_by_cause.columns).to_frame().T
mat_dalys = mat_dalys * p_scaling_factor

baseline = mat_dalys.xs(0, axis=1, level="draw")
mat_dalys_averted_df = baseline.sub(
    mat_dalys,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
mat_dalys_averted_df = mat_dalys_averted_df.droplevel(0, axis=0)
mat_dalys_averted_df.rename(index={'Maternal Disorders': "mat_dalys_averted"}, inplace=True)

# 3.) Neonatal deaths (direct)
direct_neo_deaths = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=(
        lambda df: df.loc[(df['label'] == 'Neonatal Disorders')].assign(
            year=df['date'].dt.year).groupby(['year'])['year'].count()),
    do_scaling=False)

direct_neo_deaths_final = direct_neo_deaths.fillna(0)
direct_neo_deaths_final = direct_neo_deaths_final * p_scaling_factor

baseline = direct_neo_deaths_final.xs(0, axis=1, level="draw")

direct_neo_deaths_averted_df = baseline.sub(
    direct_neo_deaths_final,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")

direct_neo_deaths_averted_df.rename(index={sim_start_year: "direct_neo_deaths_averted"}, inplace=True)

# 4.) Neonatal DALYs (direct)
neo_dalys = dalys_by_cause.loc[sim_start_year, 'Neonatal Disorders'].reindex(dalys_by_cause.columns).to_frame().T
neo_dalys = neo_dalys * p_scaling_factor

baseline = neo_dalys.xs(0, axis=1, level="draw")
neo_dalys_averted_df = baseline.sub(
    neo_dalys,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
neo_dalys_averted_df = neo_dalys_averted_df.droplevel(0, axis=0)
neo_dalys_averted_df.rename(index={'Neonatal Disorders': "neo_dalys_averted"}, inplace=True)

# 5.) Stillbirths
stillbirths = results['deaths_and_stillbirths']['crude'].loc['total_stillbirths'].reindex(
    dalys_by_cause.columns).to_frame().T
stillbirths = stillbirths * p_scaling_factor

baseline = stillbirths.xs(0, axis=1, level="draw")
stillbirths_averted_df = baseline.sub(
    stillbirths,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
stillbirths_averted_df.rename(index={'total_stillbirths': "stillbirths_averted"}, inplace=True)

# 6.) All cause DALYs averted
total_dalys = dalys_by_cause.groupby(['year']).sum()
total_dalys = total_dalys * p_scaling_factor

baseline = total_dalys.xs(0, axis=1, level="draw")
total_dalys_averted_df = baseline.sub(
    total_dalys,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")

total_dalys_averted_df.rename(index={sim_start_year: "total_dalys_averted"}, inplace=True)

# 7.) All cause DALYs averted (inc. stillbirths)
#  TODO: I THINK WE NEED TO WEIGHT THIS BASED ON GESTATIONAL AGE
preg_loss = extract_results(
    results_folder,
    module="tlo.methods.pregnancy_supervisor",
    key="pregnancy_loss",
    custom_generate_series=(
        lambda df: df.loc[(df['gest_age'] > 28)].assign(
            year=df['date'].dt.year).groupby(['year'])['year'].count()),
    do_scaling=False)
preg_loss_yll = preg_loss * 90
preg_loss_yll = preg_loss_yll * p_scaling_factor

adj_dalys = total_dalys + preg_loss_yll

baseline = adj_dalys.xs(0, axis=1, level="draw")
adj_dalys_averted_df = baseline.sub(
    adj_dalys,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")

adj_dalys_averted_df.rename(index={sim_start_year: "adj_dalys_averted"}, inplace=True)

# 8.) Maternal + newborn + stillbirths DALYs averted
mat_neo_dalys_averted = mat_dalys_averted_df.copy()

baseline = preg_loss_yll.xs(0, axis=1, level="draw")
preg_loss_averted_averted_df = baseline.sub(
    preg_loss_yll,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")

mat_neo_dalys_averted.iloc[:, :] = (mat_dalys_averted_df.to_numpy() +
                                    neo_dalys_averted_df.to_numpy() +
                                    preg_loss_averted_averted_df.to_numpy())
mat_neo_dalys_averted.rename(index={'mat_dalys_averted': "mat_neo_dalys_averted"}, inplace=True)

# ---------------------------------------------------------
# Prepare data
# ---------------------------------------------------------

death_plot_df = pd.concat(
    [
        direct_mat_deaths_averted_df,
        direct_neo_deaths_averted_df,
        stillbirths_averted_df,
    ],
    axis=0
)

death_plot_df_summ = summarize_confidence_intervals(death_plot_df)

death_outcome_labels = {
    "mat_direct_deaths_averted": "Direct maternal deaths averted",
    "direct_neo_deaths_averted": "Direct neonatal deaths averted",
    "stillbirths_averted": "Stillbirths averted",
}

dalys_outcomes_df = pd.concat(
    [
        total_dalys_averted_df,
        adj_dalys_averted_df,
        mat_neo_dalys_averted,
    ],
    axis=0
)

dalys_outcomes_df_summ = summarize_confidence_intervals(dalys_outcomes_df)

dalys_outcome_labels = {
    "total_dalys_averted": "All-cause DALYs averted",
    "adj_dalys_averted": "All-cause DALYs averted (inc. pregnancy loss)",
    "mat_neo_dalys_averted": "Maternal and perinatal DALYs averted",
}

panel_colours = [
    [
        "#4C6A92",  # muted blue
        "#8A6F8F",  # muted purple
        "#6F8F7A",  # muted green
    ],
    [
        "#B07A6A",  # muted terracotta
        "#C09A5B",  # muted ochre
        "#6F858F",  # muted slate
    ]
]

def format_bar_value(value):
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K".replace(".0K", "K")
    return f"{value:,.0f}"

def baseline_mean(df):
    """Mean baseline outcome across runs."""
    return df.xs(0, axis=1, level="draw").iloc[0].mean()


baseline_outcomes = {
    "mat_direct_deaths_averted": baseline_mean(direct_deaths),
    "direct_neo_deaths_averted": baseline_mean(
        direct_neo_deaths_final.loc[[sim_start_year]]
    ),
    "stillbirths_averted": baseline_mean(stillbirths),
    "total_dalys_averted": baseline_mean(
        total_dalys.loc[[sim_start_year]]
    ),
    "adj_dalys_averted": baseline_mean(
        adj_dalys.loc[[sim_start_year]]
    ),
    "mat_neo_dalys_averted": (
        baseline_mean(mat_dalys)
        + baseline_mean(neo_dalys)
        + baseline_mean(preg_loss_yll.loc[[sim_start_year]])
    ),
}

def produce_fig_2(
    death_data,
    daly_data,
    death_labels,
    daly_labels,
    save_title,
    baseline_outcomes,

):

    datasets = [
        (death_data, death_labels, "Deaths averted"),
        (daly_data, daly_labels, "DALYs averted"),
    ]

    # Scenario ordering
    draws = sorted(
        death_data.columns.get_level_values("draw").unique()
    )

    scenario_labels = [
        draw_labels.get(draw, f"Scenario {draw}")
        for draw in draws
    ]

    x = np.arange(len(draws))

    # ---------------------------------------------------------
    # Create two vertically stacked panels with shared x-axis
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(max(14, len(draws) * 1.2), 11)
    )

    # ---------------------------------------------------------
    # Plot each panel
    # ---------------------------------------------------------

    for panel_i, (ax, (data, outcome_labels, ylabel)) in enumerate(
        zip(axes, datasets)
    ):

        plot_df = data.copy()

        plot_df.index.name = "outcome"
        plot_df.columns.names = ["draw", "stat"]

        # Convert to long format
        long_df = (
            plot_df
            .stack(level="draw")
            .reset_index()
        )

        outcomes = plot_df.index.tolist()
        n_outcomes = len(outcomes)

        # Width occupied by bars within each intervention
        group_width = 0.8
        bar_width = group_width / n_outcomes

        # -----------------------------------------------------
        # Plot each outcome
        # -----------------------------------------------------

        for i, outcome in enumerate(outcomes):

            outcome_df = (
                long_df
                .loc[long_df["outcome"].eq(outcome)]
                .set_index("draw")
                .reindex(draws)
            )

            means = outcome_df["mean"].to_numpy()
            lower = outcome_df["lower"].to_numpy()
            upper = outcome_df["upper"].to_numpy()

            # Asymmetric 95% confidence intervals
            yerr = np.vstack([
                means - lower,
                upper - means
            ])

            # Centre bars around intervention position
            offset = (
                i - (n_outcomes - 1) / 2
            ) * bar_width

            bars = ax.bar(
                x + offset,
                means,
                width=bar_width * 0.9,
                yerr=yerr,
                capsize=2.5,
                color=panel_colours[panel_i][i],
                error_kw={
                    "elinewidth": 0.8,
                    "capthick": 0.8,
                },
                label=outcome_labels.get(
                    outcome,
                    outcome.replace("_", " ").title()
                )
            )

            for bar, mean, lo, hi in zip(bars, means, lower, upper):
                if not np.isfinite([mean, lo, hi]).all():
                    continue

                # Show the value only if the interval excludes zero.
                if lo > 0:
                    label_y = hi
                    offset_points = 4
                    vertical_alignment = "bottom"
                elif hi < 0:
                    label_y = lo
                    offset_points = -4
                    vertical_alignment = "top"
                else:
                    continue

                baseline_value = baseline_outcomes[outcome]

                if np.isfinite(baseline_value) and baseline_value > 0:
                    pct_averted = 100 * mean / baseline_value
                    annotation = f"{format_bar_value(mean)} ({pct_averted:.1f}%)"
                else:
                    annotation = f"{format_bar_value(mean)} (N/A)"

                ax.annotate(
                    annotation,
                    xy=(bar.get_x() + bar.get_width() / 2, label_y),
                    xytext=(0, offset_points),
                    textcoords="offset points",
                    ha="center",
                    va=vertical_alignment,
                    rotation=90,
                    fontsize=7,
                    clip_on=False,
                )

        # -----------------------------------------------------
        # Panel formatting
        # -----------------------------------------------------

        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=0.8
        )

        ax.set_ylabel(ylabel)

        ax.spines[["top", "right"]].set_visible(False)

        ax.grid(
            axis="y",
            alpha=0.2
        )

        ax.legend(
            frameon=False,
            loc="upper left",
        )

    # ---------------------------------------------------------
    # Panel titles
    # ---------------------------------------------------------

    axes[0].set_title(
        "A. Mortality outcomes",
        loc="left",
        fontweight="bold"
    )

    axes[1].set_title(
        "B. DALY outcomes",
        loc="left",
        fontweight="bold"
    )

    # ---------------------------------------------------------
    # Shared x-axis
    # ---------------------------------------------------------

    axes[1].set_xticks(x)

    axes[1].set_xticklabels(
        scenario_labels,
        rotation=45,
        ha="right"
    )

    axes[1].set_xlabel("Intervention")

    # Remove x tick marks/labels from upper panel
    axes[0].tick_params(
        axis="x",
        which="both",
        bottom=False,
        labelbottom=False
    )

    # ---------------------------------------------------------
    # Final formatting
    # ---------------------------------------------------------

    # Small gap between panels
    fig.subplots_adjust(
        hspace=0.15
    )

    fig.tight_layout()

    plt.savefig(
        f"{g_path}/{save_title}.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.show()

produce_fig_2(
    death_plot_df_summ,
    dalys_outcomes_df_summ,
    death_outcome_labels,
    dalys_outcome_labels,
    "fig_2_health_outcomes",
    baseline_outcomes,
)

#  ============================ FIGURE S1 and S2 - DEATHS AND DALYS BY CAUSE  =========================================
def figure_S1_and_S2_heatmap(
        data,
        title,
        save_title,
        compact_annotations=True,

):
    """
    Outputs a heatmap showing Deaths/DALYS averted by cause.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing mean, lower and upper estimates in the
        'stat' column level.

    compact_annotations : bool, default True
        If True, annotations use compact formatting such as 6.2K.
        If False, annotations show the unscaled value, such as 6,150.
    """

    df = data

    mean_df = df.xs(
        "mean",
        axis=1,
        level="stat"
    )

    lower_df = df.xs(
        "lower",
        axis=1,
        level="stat"
    )

    upper_df = df.xs(
        "upper",
        axis=1,
        level="stat"
    )

    # Make sure all three DataFrames have identical ordering
    lower_df = lower_df.reindex_like(mean_df)
    upper_df = upper_df.reindex_like(mean_df)

    # Remove rows containing only missing values
    mean_df = mean_df.dropna(how="all")

    # Apply the same retained rows to the uncertainty bounds
    lower_df = lower_df.loc[mean_df.index]
    upper_df = upper_df.loc[mean_df.index]

    # Rename scenario columns
    mean_df = mean_df.rename(columns=draw_labels)
    lower_df = lower_df.rename(columns=draw_labels)
    upper_df = upper_df.rename(columns=draw_labels)

    # Remove column-axis names
    mean_df.columns.name = None
    lower_df.columns.name = None
    upper_df.columns.name = None

    # Identify uncertainty intervals that include zero
    includes_zero = (
        lower_df.le(0) &
        upper_df.ge(0)
    )

    def format_dalys(value):
        """
        Format a value according to compact_annotations.

        Examples when compact_annotations=True:
            6150   -> 6.2K
            520    -> 520
            -1730  -> −1.7K

        Examples when compact_annotations=False:
            6150   -> 6,150
            520    -> 520
            -1730  -> −1,730
        """
        if pd.isna(value):
            return ""

        abs_value = abs(value)

        if compact_annotations:
            if abs_value >= 100_000:
                label = f"{abs_value / 1_000:.0f}K"
            elif abs_value >= 1_000:
                label = f"{abs_value / 1_000:.1f}K"
            else:
                label = f"{abs_value:.0f}"
        else:
            label = f"{abs_value:,.0f}"

        # Use a typographic minus sign
        if value < 0:
            label = f"−{label}"

        return label

    # Create annotation labels
    annotations = pd.DataFrame(
        "",
        index=mean_df.index,
        columns=mean_df.columns
    )

    for row in mean_df.index:
        for column in mean_df.columns:

            label = format_dalys(mean_df.loc[row, column])

            # Add dagger when uncertainty interval includes zero
            if label and includes_zero.loc[row, column]:
                label += "†"

            annotations.loc[row, column] = label

    # Use the 95th percentile to prevent extreme values from
    # dominating the colour scale
    max_abs = np.nanquantile(
        np.abs(mean_df.to_numpy()),
        0.95
    )

    norm = colors.TwoSlopeNorm(
        vmin=-max_abs,
        vcenter=0,
        vmax=max_abs
    )

    sns.set_theme(style="white")

    figure_height = max(
        7,
        0.42 * len(mean_df)
    )

    fig, ax = plt.subplots(
        figsize=(12, figure_height)
    )

    sns.heatmap(
        mean_df,
        cmap="RdBu_r",
        norm=norm,
        annot=annotations,
        fmt="",
        linewidths=0.5,
        linecolor="white",
        annot_kws={
            "fontsize": 8,
            "fontweight": "bold"
        },
        cbar_kws={
            "label": "Mean DALYs averted",
            "shrink": 0.8
        },
        ax=ax
    )

    # Make annotations grey and non-bold when the interval includes zero
    includes_zero_flat = includes_zero.to_numpy().flatten()

    for text, interval_includes_zero in zip(
            ax.texts,
            includes_zero_flat
    ):
        if interval_includes_zero:
            text.set_color("dimgray")
            text.set_fontweight("normal")

    ax.set_title(
        title,
        fontsize=14,
        pad=14
    )

    ax.set_xlabel(
        "Intervention scenario",
        fontsize=11,
        labelpad=10
    )

    ax.set_ylabel("")

    ax.tick_params(
        axis="x",
        labelrotation=45,
        labelsize=9
    )

    ax.tick_params(
        axis="y",
        labelrotation=0,
        labelsize=9
    )

    plt.setp(
        ax.get_xticklabels(),
        ha="right",
        rotation_mode="anchor"
    )

    fig.text(
        0.01,
        0.01,
        "Cells show mean difference. "
        "† Uncertainty interval includes zero; these estimates are shown "
        "in grey.",
        ha="left",
        va="bottom",
        fontsize=9
    )

    plt.tight_layout(
        rect=[0, 0.04, 1, 1]
    )

    plt.savefig(
        f"{g_path}/{save_title}p.png",
        bbox_inches="tight"
    )

    plt.show()

# Deaths by cause
deaths_by_cause = extract_results(
            results_folder,
            module="tlo.methods.demography.detail",
            key="properties_of_deceased_persons",
            custom_generate_series=(
                lambda df: df.assign(
                    year=df['date'].dt.year).groupby(['year', 'cause_of_death'])['year'].count()),
            do_scaling=False)
deaths_by_c = deaths_by_cause.fillna(0)
deaths_by_c = deaths_by_c * p_scaling_factor

deaths_by_c = deaths_by_c.loc[deaths_by_c.index.get_level_values(0) != sim_end_year]
deaths_by_c = deaths_by_c.droplevel(0, axis=0)

baseline = deaths_by_c.xs(0, axis=1, level="draw")
deaths_by_c_averted_df = baseline.sub(
    deaths_by_c,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
deaths_by_c_averted_df = summarize_confidence_intervals(deaths_by_c_averted_df)

# DALYs by cause
dalys_by_cause = dalys_by_cause.loc[dalys_by_cause.index.get_level_values(0) != sim_end_year]
dalys_by_cause = dalys_by_cause * p_scaling_factor

dalys_by_cause = dalys_by_cause.droplevel(0, axis=0)

baseline = dalys_by_cause.xs(0, axis=1, level="draw")
dalys_averted_by_cause_df = baseline.sub(
    dalys_by_cause,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
dalys_averted_by_cause_df = summarize_confidence_intervals(dalys_averted_by_cause_df)

figure_S1_and_S2_heatmap(deaths_by_c_averted_df,
                                             'Deaths averted by cause and intervention scenario',
                                             'fig_S1_diff_in_deaths_by_cause_heatmap', True)


figure_S1_and_S2_heatmap(dalys_averted_by_cause_df,
                                             'DALYs averted by cause and intervention scenario',
                                             'fig_S2_diff_in_dalys_by_cause_heatmap', True)



#  ================================ TABLE 3 - COSTS  =================================================================
#  ------------------------------   Helper functions   ---------------------------------------------------------------
def find_difference_relative_to_comparison(_ser: pd.Series,
                                           comparison: str,
                                           scaled: bool = False,
                                           drop_comparison: bool = True,
                                           ):
    """Find the difference in the values in a pd.Series with a multi-index, between the draws (level 0)
    within the runs (level 1), relative to where draw = `comparison`.
    The comparison is `X - COMPARISON`."""
    return _ser \
        .unstack(level=0) \
        .apply(lambda x: (x - x[comparison]) / (x[comparison] if scaled else 1.0), axis=1) \
        .drop(columns=([comparison] if drop_comparison else [])) \
        .stack()

def summarise_cost_data(
    _df: pd.DataFrame,
    _metric: Literal["mean", "median"] = "mean",
    _interval: Literal["ci", "quantiles"] = "quantiles",
    _confidence: float = 0.95,
    _normality_alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Summarize costs across runs for each draw.

    Parameters
    ----------
    _df
        Rows = draws; columns = runs; values = costs.
    _metric
        Central summary statistic: mean or median.
    _interval
        "ci": Student-t confidence interval around the mean.
        "quantiles": central quantile interval across individual runs.
    _confidence
        Interval level; 0.95 gives a 95% CI or 2.5–97.5% quantiles.
    _normality_alpha
        Significance threshold for rejecting normality.

    Returns
    -------
    pd.DataFrame
        One row per draw, containing the central statistic, lower,
        upper, n, normality_test, normality_p and reject_normality.

    Notes
    -----
    Missing values are excluded; infinite values raise an error.
    Normality is checked using Shapiro-Wilk for 3–5,000 observations
    and D'Agostino-Pearson for larger samples. Constant data and
    samples smaller than three are not tested.
    """
    if _metric not in {"mean", "median"}:
        raise ValueError("_metric must be 'mean' or 'median'.")

    if _interval not in {"ci", "quantiles"}:
        raise ValueError("_interval must be 'ci' or 'quantiles'.")

    if _interval == "ci" and _metric != "mean":
        raise ValueError("The Student-t CI requires _metric='mean'.")

    if not 0 < _confidence < 1:
        raise ValueError("_confidence must be between 0 and 1.")

    if not 0 < _normality_alpha < 1:
        raise ValueError("_normality_alpha must be between 0 and 1.")

    tail = (1 - _confidence) / 2

    def summarize_draw(row):
        values = row.dropna().to_numpy(dtype=float)
        n = len(values)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Draw {row.name!r} contains infinite cost values."
            )

        centre = (
            np.mean(values) if _metric == "mean" else np.median(values)
        ) if n else np.nan

        lower = upper = np.nan

        if _interval == "quantiles" and n:
            lower, upper = np.quantile(values, [tail, 1 - tail])

        elif _interval == "ci" and n >= 2:
            margin = (
                stats.t.ppf(1 - tail, df=n - 1)
                * stats.sem(values)
            )
            lower, upper = centre - margin, centre + margin

        p_value = np.nan
        reject_normality = pd.NA

        if n < 3:
            test = "not tested: n < 3"
        elif np.all(values == values[0]):
            test = "not tested: constant data"
        else:
            if n <= 5_000:
                test = "Shapiro-Wilk"
                p_value = stats.shapiro(values).pvalue
            else:
                test = "D'Agostino-Pearson"
                p_value = stats.normaltest(values).pvalue

            reject_normality = bool(p_value < _normality_alpha)

        return pd.Series({
            _metric: centre,
            "lower": lower,
            "upper": upper,
            "n": n,
            "normality_test": test,
            "normality_p": p_value,
            "reject_normality": reject_normality,
        })

    result = _df.apply(summarize_draw, axis=1)
    result["reject_normality"] = result["reject_normality"].astype("boolean")
    return result

def plot_bars_with_uncertainty(df, ylabel="Incremental cost", title=None):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        np.arange(len(df)),
        df["mean"],
        yerr=np.vstack([
            df["mean"] - df["lower"],
            df["upper"] - df["mean"],
        ]),
        color="#4C6A92",
        capsize=4,
        error_kw={"elinewidth": 1},
    )

    ax.set_xticks(np.arange(len(df)), labels=df.index)
    ax.set(xlabel="Scenario", ylabel=ylabel, title=title)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()

    return fig, ax

# ----------------------- Extract initial system level cost data ------------------------------------------------------
list_of_relevant_years_for_costing = list(range(TARGET_PERIOD[0].year, TARGET_PERIOD[-1].year + 1))

# 1.) --------------------------- Estimate incremental costs due to use of MNH consumables ----------------------------
mnh_item_codes, item_audit = get_mnh_consumable_item_codes(
    resourcefilepath,
    include_iptp=True,
    return_audit=True,
)

# Inspect item codes, names and the modules defining them
print(item_audit.to_string(index=False))

# input_costs_df = estimate_mnh_input_cost_of_scenarios(results_folder=results_folder,
#                                      resourcefilepath=resourcefilepath,
#                                      suspended_results_folder=results_folder,
#                                      scaling_factor=p_scaling_factor,
#                                      _draws=draws,
#                                      _years=list_of_relevant_years_for_costing,
#                                      cost_only_used_staff=False,
#                                      consumable_item_codes=mnh_item_codes)
# input_costs_df.to_csv(f'{g_path}/input_costs.csv')

input_costs_df = pd.read_csv(f'{g_path}/input_costs.csv')
input_costs_df = input_costs_df.set_index('Unnamed: 0')

consumable_costs = (input_costs_df.loc[input_costs_df["cost_category"].eq(
    "medical consumables")].groupby(["draw", "run"])["cost"].sum())

total_consumable_costs = consumable_costs.sum()

# 2.) ---------------------- Estimate incremental costs due to use of MNH HCW time ------------------------------------
time_by_hsi, time_differences, time_summary = estimate_mnh_hcw_time(
    results_folder=results_folder,
    resourcefilepath=resourcefilepath,
    years=[2026],  # Replace with your complete analysis years
    baseline_draw=0,
    scaling_factor=p_scaling_factor,
    # Alternatively, omit scaling_factor to use logged scaling factors.
)

# Must match the staffing assumption used in the baseline simulation:
# "actual", "funded" or "funded_plus".
staffing = "actual"

daily_capacity = pd.read_csv(
    resourcefilepath
    / "healthsystem"
    / "human_resources"
    / staffing
    / "ResourceFile_Daily_Capabilities.csv",
    dtype={"Facility_Level": str},
)

# Apply the same pooling of levels 1b and 2 as the model.
daily_capacity = pool_capabilities_at_levels_1b_and_2(daily_capacity)

daily_capacity = (
    daily_capacity.loc[daily_capacity["Facility_Level"].ne("5")]
    .groupby(["Facility_Level", "Officer_Category"], as_index=False)[
        "Total_Mins_Per_Day"
    ]
    .sum()
    .rename(columns={"Officer_Category": "OfficerType"})
)

annual_capacity = pd.concat(
    [
        daily_capacity.assign(
            year=year,
            capacity_minutes=(
                daily_capacity["Total_Mins_Per_Day"]
                * (366 if calendar.isleap(year) else 365)
            ),
        )
        for year in [2026]
    ],
    ignore_index=True,
)[["year", "Facility_Level", "OfficerType", "capacity_minutes"]]

national_hr_costs = input_costs_df.loc[input_costs_df["cost_category"].eq("human resources for health")].copy()

hr_values = value_mnh_hcw_time(
    differences=time_differences,
    national_hr_costs=national_hr_costs,
    annual_capacity=annual_capacity,
    baseline_draw=0,
)

mnh_hr_costs = hr_values.groupby(["draw", "run"])["scenario_mnh_hr_cost"].sum()

# --------------------------------------- 3. Combine HR and cons costs ------------------------------------------------
total_mnh_costs = (consumable_costs + mnh_hr_costs).rename("total_mnh_cost")

# ------------------------------------- 4. Calculate incremental costs ------------------------------------------------
incremental_mnh_costs = (pd.DataFrame(
        find_difference_relative_to_comparison(
            total_mnh_costs,
            comparison=0,
            scaled=False)  # sets the comparator to draw 0 which is the Actual scenario
    ).T.iloc[0].unstack()).T

# --------------------------------------- 5. Add above service level cost ------------------------------------------------
above_service_level_cost_assumption = 1.15

incremental_mnh_costs + incremental_mnh_costs.abs() * (above_service_level_cost_assumption - 1)

incremental_mnh_costs_summ = summarise_cost_data(incremental_mnh_costs, _interval='ci')

fig, ax = plot_bars_with_uncertainty(
    incremental_mnh_costs_summ,
    ylabel="Incremental cost (USD)",
)
plt.show()

# # -------------------------------------  NET MONETARY BENEFIT -------------------------------------------------------
# Net monetary benefit
incremental_mnh_costs.index.name = "draw"
incremental_mnh_costs.columns.name = "run"

cost = incremental_mnh_costs.stack().to_frame().T

# todo: find correct threshold value AND evaluate across a range of thresholds
cost_per_daly_threshold = 63

def get_net_monetary_benefit(outcome_df):
    nmb = (outcome_df.reset_index(drop=True) * cost_per_daly_threshold) - cost.reset_index(drop=True)
    # todo: should potential be quantiles
    nmb_summ = summarize_confidence_intervals(nmb)
    return nmb_summ

nmb_total_daly_averted = get_net_monetary_benefit(total_dalys_averted_df)
nmb_adj_daly_averted = get_net_monetary_benefit(adj_dalys_averted_df)
nmb_mnh_daly_averted = get_net_monetary_benefit(mat_neo_dalys_averted)


def fig_net_mon_benefit():

    summaries = {
        "Total DALYs": nmb_total_daly_averted,
        "Adjusted DALYs": nmb_adj_daly_averted,
        "Maternal & neonatal DALYs": nmb_mnh_daly_averted,
    }

    colors = {
        "Total DALYs": "#52796F",
        "Adjusted DALYs": "#D4A373",
        "Maternal & neonatal DALYs": "#6B7A99",
    }

    def format_value(value):
        sign = "-" if value < 0 else ""
        value = abs(value)

        if value >= 1e9:
            return f"{sign}${value / 1e9:.1f}B"
        elif value >= 1e6:
            return f"{sign}${value / 1e6:.1f}M"
        elif value >= 1e3:
            return f"{sign}${value / 1e3:.0f}K"
        return f"{sign}${value:.0f}"

    # Extract mean NMB for each scenario
    # Extract mean and 95% CI for each scenario
    plot_data = {}
    lower_data = {}
    upper_data = {}

    for label, df in summaries.items():
        means = df.xs("mean", axis=1, level="stat").iloc[0]
        lowers = df.xs("lower", axis=1, level="stat").iloc[0]
        uppers = df.xs("upper", axis=1, level="stat").iloc[0]

        plot_data[label] = means
        lower_data[label] = lowers
        upper_data[label] = uppers

    plot_df = pd.DataFrame(plot_data).sort_index()
    lower_df = pd.DataFrame(lower_data).reindex(plot_df.index)
    upper_df = pd.DataFrame(upper_data).reindex(plot_df.index)

    plot_df = pd.DataFrame(plot_data).sort_index()

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(plot_df))
    bar_width = 0.23

    for i, (label, color) in enumerate(colors.items()):

        positions = x + (i - 1) * bar_width

        values = plot_df[label].to_numpy(dtype=float)
        lowers = lower_df[label].to_numpy(dtype=float)
        uppers = upper_df[label].to_numpy(dtype=float)

        valid = (
            np.isfinite(values) &
            np.isfinite(lowers) &
            np.isfinite(uppers)
        )

        # Asymmetric 95% CI error bars
        yerr = np.vstack([
            values[valid] - lowers[valid],
            uppers[valid] - values[valid]
        ])

        ax.bar(
            positions[valid],
            values[valid],
            width=bar_width,
            color=color,
            alpha=0.9,
            label=label,
            yerr=yerr,
            error_kw={
                "ecolor": "#333333",
                "elinewidth": 1,
                "capsize": 3,
                "capthick": 1
            }
        )

        # Annotate mean NMB values
        # Annotate only where the 95% CI excludes zero
        for pos, value, lower, upper in zip(
            positions[valid],
            values[valid],
            lowers[valid],
            uppers[valid]
        ):

            # Skip annotation if CI includes zero
            if lower <= 0 <= upper:
                continue

            ax.annotate(
                format_value(value),
                xy=(pos, upper if value >= 0 else lower),
                xytext=(0, 5 if value >= 0 else -5),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
                rotation=90
            )

    # Zero reference line
    ax.axhline(0, color="black", linewidth=0.9)

    # Scenario labels
    ax.set_xticks(x)
    ax.set_xticklabels(
        [draw_labels.get(draw, str(draw)) for draw in plot_df.index],
        rotation=45,
        ha="right"
    )

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Net monetary benefit (USD)")
    ax.set_title(
        f"Net monetary benefit at ${cost_per_daly_threshold} per DALY averted"
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: format_value(value))
    )

    ax.legend(frameon=False)

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    # Add headroom above and below bars for annotations
    ymin = min(0, plot_df.min().min())
    ymax = max(0, plot_df.max().max())
    yrange = ymax - ymin

    if yrange == 0:
        yrange = 1

    ax.set_ylim(
        ymin - yrange * 0.15,
        ymax + yrange * 0.15
    )

    plt.tight_layout()
    plt.show()

    return plot_df

fig_net_mon_benefit()

# ----------------------------------------- Generate table -----------------------------------------------------------
# Scenario, Incremental Cost (with uncertainty), Net Monetary Benefit (Total),  Net Monetary Benefit (Adjusted),
# Net Monetary Benefit (MNH)

# TODO: ADD TABLE

# -------------------------------------- MAXIMUM ABILITY TO PAY ------------------------------------------------------
cost_effectiveness_value = 63
# todo: we may need to use the averages an only look at those statistically significant DALYs averted

total_dalys_averted_summ = summarize_confidence_intervals(total_dalys_averted_df)
adj_dalys_averted_summ = summarize_confidence_intervals(adj_dalys_averted_df)
mat_neo_dalys_averted_summ = summarize_confidence_intervals(mat_neo_dalys_averted)


def produce_fig_3(save_title):

    summaries = {
        "Total DALYs": total_dalys_averted_summ,
        "Adjusted DALYs": adj_dalys_averted_summ,
        "Maternal & neonatal DALYs": mat_neo_dalys_averted_summ,
    }

    colors = {
        "Total DALYs": "#52796F",
        "Adjusted DALYs": "#D4A373",
        "Maternal & neonatal DALYs": "#6B7A99",
    }

    plot_data = {}

    for label, df in summaries.items():

        means = df.xs("mean", axis=1, level="stat").iloc[0]
        lowers = df.xs("lower", axis=1, level="stat").iloc[0]
        uppers = df.xs("upper", axis=1, level="stat").iloc[0]

        # Retain only positive estimates with CI entirely above zero
        valid = (means > 0) & (lowers > 0) & (uppers > 0)

        plot_data[label] = (
            means * cost_effectiveness_value
        ).where(valid)

    plot_df = pd.DataFrame(plot_data).sort_index()

    # Remove scenarios with no significant estimates
    plot_df = plot_df.dropna(how="all")

    fig, ax = plt.subplots(figsize=(14, 6))

    bar_width = 0.23
    x = np.arange(len(plot_df))

    # Plot each scenario individually to centre the available bars
    for scenario_idx, (draw, row) in enumerate(plot_df.iterrows()):

        available = row.dropna()
        n_bars = len(available)

        # Symmetric offsets, ensuring no gaps between remaining bars
        offsets = (
            np.arange(n_bars) - (n_bars - 1) / 2
        ) * bar_width

        for offset, (label, value) in zip(offsets, available.items()):

            ax.bar(
                x[scenario_idx] + offset,
                value,
                width=bar_width,
                color=colors[label],
                alpha=0.9,
                label=label if scenario_idx == 0 else None
            )

            # Abbreviate monetary values
            def format_value(value):
                if abs(value) >= 1e9:
                    return f"${value / 1e9:.1f}B"
                elif abs(value) >= 1e6:
                    return f"${value / 1e6:.1f}M"
                elif abs(value) >= 1e3:
                    return f"${value / 1e3:.0f}K"
                return f"${value:.0f}"

            # Add value label above bar
            ax.annotate(
                format_value(value),
                xy=(x[scenario_idx] + offset, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90
            )

    # Use scenario names from draw_labels
    ax.set_xticks(x)
    ax.set_xticklabels(
        [draw_labels.get(draw, str(draw)) for draw in plot_df.index],
        rotation=45,
        ha="right"
    )

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Maximum willingness to pay (USD)")
    ax.set_title("Maximum willingness to pay based on DALYs averted")

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"${value / 1e6:,.1f}M")
    )

    # Construct legend independently so all three categories appear
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=color, label=label)
        for label, color in colors.items()
    ]

    ax.legend(handles=legend_handles, frameon=False)

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    # Provide headroom for bar labels
    ax.set_ylim(0, plot_df.max().max() * 1.18)
    plt.tight_layout()
    plt.savefig(
        f"{g_path}/{save_title}.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.show()

    return plot_df

produce_fig_3('figure_3_max_ab_to_pay')

# -------------------------------------------- END OF FILE -------------------------------------------


# # ------------------------------------ COST PER DALY/DEATH AVERTED --------------------------------------------------
# todo: understand if we should be taking average cost / average effect as opposed to pair-wise cost/effect differences
#
# def get_cost_per_outcome_averted(outcome_df):
#     # Check alignment before removing any labels.
#     print("Cost index:", cost.index[:10])
#     print("Outcome index:", outcome_df.index[:10])
#     if cost.shape != outcome_df.shape:
#         raise ValueError(
#             f"Shape mismatch: cost={cost.shape}, outcomes={outcome_df.shape}"
#         )
#
#     if not cost.columns.equals(outcome_df.columns):
#         raise ValueError("Cost and outcome columns do not match.")
#
#     costs = cost.reset_index(drop=True).astype(float)
#     outcomes = outcome_df.reset_index(drop=True).astype(float)
#
#     costs = cost.astype(float)
#     outcomes = outcome_df.astype(float)
#
#     zero_mask = outcomes.eq(0)
#
#     # Locate zero denominators using positional coordinates so original
#     # row/column labels (including MultiIndex labels) are preserved.
#     rows, cols = np.where(zero_mask.to_numpy())
#
#     zero_details = pd.DataFrame([
#         {
#             "row": outcomes.index[i],
#             "column": outcomes.columns[j],
#             "outcome_averted": outcomes.iat[i, j],
#             "incremental_cost": costs.iat[i, j],
#             "division_type": (
#                 "0 / 0" if costs.iat[i, j] == 0 else "nonzero / 0"
#             ),
#         }
#         for i, j in zip(rows, cols)
#     ])
#
#     print(
#         f"Zero outcomes: {zero_mask.to_numpy().sum()} / {outcomes.size}\n"
#         f"Missing outcomes: {outcomes.isna().to_numpy().sum()}\n"
#         f"Negative outcomes: {outcomes.lt(0).to_numpy().sum()}"
#     )
#
#     if not zero_details.empty:
#         print("\nZero denominators by column:")
#         print(zero_mask.sum().loc[lambda x: x > 0].to_string())
#
#         print("\nFirst 20 zero-denominator locations:")
#         print(zero_details.head(20).to_string(index=False))
#
#     # Undefined ratios remain missing; never replace them with zero.
#     ratios = costs / outcomes.mask(zero_mask)
#     summary = summarize_confidence_intervals(ratios)
#
#     return summary, zero_details
#
#
# def get_cost_per_outcome_averted(outcome_df):
#      cost_per_outcome_averted = cost.reset_index(drop=True) / outcome_df.reset_index(drop=True)
#      cost_per_total_outcome_averted_summ = summarize_confidence_intervals(cost_per_outcome_averted)
#      return cost_per_total_outcome_averted_summ
#
#
# cost_per_total_daly_averted = get_cost_per_outcome_averted(total_dalys_averted_df)
# cost_per_adj_daly_averted = get_cost_per_outcome_averted(adj_dalys_averted_df)
# cost_per_mnh_daly_averted = get_cost_per_outcome_averted(mat_neo_dalys_averted)
#
# # todo: we have division by 0 errors here
# # cost_per_mat_death_averted = get_cost_per_outcome_averted(direct_mat_deaths_averted_df)
# # cost_per_neo_death_averted = get_cost_per_outcome_averted(direct_neo_deaths_averted_df)
# # cost_per_stillbirth_averted = get_cost_per_outcome_averted(stillbirths_averted_df)
#
# def plot_cost_per_outcome(
#     cost,
#     outcomes,
#     draw_labels=None,
#     currency="USD",
# ):
#     """
#     cost: one-row DataFrame with MultiIndex columns ('draw', 'run').
#     outcomes: dict mapping outcome labels to DataFrames of the same structure.
#
#     Bars represent empirical run-level ratio percentiles, not confidence
#     intervals around the central ICER.
#     """
#     draw_labels = {} if draw_labels is None else draw_labels
#
#     def as_series(df):
#         if len(df) != 1:
#             raise ValueError("Each input must have exactly one row.")
#         if not isinstance(df.columns, pd.MultiIndex):
#             raise ValueError("Columns must be a MultiIndex: draw → run.")
#         if list(df.columns.names) != ["draw", "run"]:
#             raise ValueError("Column levels must be named 'draw' and 'run'.")
#         if not df.columns.is_unique:
#             raise ValueError("Each draw/run column must be unique.")
#         return df.iloc[0].astype(float)
#
#     cost_s = as_series(cost)
#     draws = list(cost_s.index.get_level_values("draw").unique())
#
#     fig, axes = plt.subplots(
#         1,
#         len(outcomes),
#         figsize=(5 * len(outcomes), max(5, 0.65 * len(draws))),
#         squeeze=False,
#         constrained_layout=True,
#     )
#
#     records = []
#
#     for ax, (outcome_name, outcome_df) in zip(axes.ravel(), outcomes.items()):
#         effect_s = as_series(outcome_df)
#
#         if (
#             len(effect_s) != len(cost_s)
#             or not cost_s.index.isin(effect_s.index).all()
#         ):
#             raise ValueError(
#                 f"{outcome_name}: draw/run columns do not match costs."
#             )
#
#         effect_s = effect_s.reindex(cost_s.index)
#         labels = []
#
#         for y, draw in enumerate(draws):
#             c = cost_s.xs(draw, level="draw").to_numpy()
#             e = effect_s.xs(draw, level="draw").to_numpy()
#
#             missing = np.isnan(c) | np.isnan(e)
#             valid = np.isfinite(c) & np.isfinite(e)
#             zero_effect = np.isfinite(e) & (e == 0)
#
#             with np.errstate(divide="ignore", invalid="ignore"):
#                 ratios = c / e
#
#             n_undefined = int((~np.isfinite(ratios)).sum())
#             notes = []
#
#             if zero_effect.any():
#                 notes.append(f"zero outcome: {zero_effect.sum()}")
#             if missing.any():
#                 notes.append(f"missing: {missing.sum()}")
#             if n_undefined:
#                 notes.append(f"undefined ratio: {n_undefined}")
#
#             estimate = lower = upper = np.nan
#
#             if valid.any():
#                 mean_effect = e[valid].mean()
#
#                 if mean_effect != 0:
#                     estimate = c[valid].mean() / mean_effect
#
#                 # Ratio intervals are interpretable here only when
#                 # every valid run has strictly positive benefit.
#                 if (e[valid] > 0).all():
#                     lower, upper = np.quantile(
#                         ratios[valid], [0.025, 0.975]
#                     )
#                     ax.hlines(y, lower, upper, color="steelblue", lw=2)
#                 elif (e[valid] < 0).all():
#                     notes.append("all valid runs: worse outcomes")
#                 elif (e[valid] < 0).any() and (e[valid] > 0).any():
#                     notes.append("benefit crosses zero")
#                 else:
#                     notes.append("benefit includes zero")
#
#             if np.isfinite(estimate):
#                 ax.plot(estimate, y, "o", color="black", markersize=5)
#             else:
#                 # x uses axes coordinates; y uses data coordinates.
#                 ax.text(
#                     0.98, y, "undefined",
#                     transform=ax.get_yaxis_transform(),
#                     ha="right", va="center",
#                     color="firebrick", fontsize=9,
#                 )
#
#             label = str(draw_labels.get(draw, f"Scenario {draw}"))
#             if notes:
#                 label += "\n[" + "; ".join(notes) + "]"
#             labels.append(label)
#
#             records.append({
#                 "outcome": outcome_name,
#                 "draw": draw,
#                 "central_icer": estimate,
#                 "run_ratio_p2.5": lower,
#                 "run_ratio_p97.5": upper,
#                 "zero_outcome_runs": int(zero_effect.sum()),
#                 "missing_runs": int(missing.sum()),
#                 "undefined_ratio_runs": n_undefined,
#                 "notes": "; ".join(notes),
#             })
#
#         ax.set(
#             yticks=np.arange(len(draws)),
#             yticklabels=labels,
#             title=outcome_name,
#             xlabel=f"{currency} per outcome averted",
#         )
#         ax.invert_yaxis()
#         ax.axvline(0, color="grey", lw=0.8)
#         ax.grid(axis="x", alpha=0.2)
#         ax.spines[["top", "right"]].set_visible(False)
#         plt.show()
#
#     return fig, pd.DataFrame(records)
#
# outcomes = {
#     "Total DALYs": total_dalys_averted_df,
#     "Adjusted DALYs": adj_dalys_averted_df,
#     "Maternal/newborn DALYs": mat_neo_dalys_averted,
#     # "Maternal deaths": direct_mat_deaths_averted_df,
#     # "Neonatal deaths": direct_neo_deaths_averted_df,
#     # "Stillbirths": stillbirths_averted_df,
# }
#
# fig, plot_summary = plot_cost_per_outcome(
#     cost=cost,
#     outcomes=outcomes,
#     draw_labels=draw_labels,
#     currency="USD",  # Change to your costing currency
# )
# # input_costs_df_not_scaled = estimate_input_cost_of_scenarios(results_folder=results_folder,
# #                                      resourcefilepath=resourcefilepath,
# #                                      suspended_results_folder=results_folder,
# #                                      _draws=draws,
# #                                      _years=list_of_relevant_years_for_costing,
# #                                      cost_only_used_staff=True)
#
# input_costs_df.to_csv(f'{g_path}/input_costs.csv')
# input_costs = pd.read_csv(f'{g_path}/input_costs.csv')
# input_costs = input_costs.set_index('Unnamed: 0')
#
# # --------------------------------- Check difference in input cost without any adjustment -----------------------------
# total_input_cost_no_adj = input_costs.groupby(['draw', 'run'])['cost'].sum()
#
# incremental_scenario_cost_no_adj = (pd.DataFrame(
#         find_difference_relative_to_comparison(
#             total_input_cost_no_adj,
#             comparison=0,
#             scaled=False)  # sets the comparator to draw 0 which is the Actual scenario
#     ).T.iloc[0].unstack()).T
#
# input_costs_no_adj_summ = summarise_cost_data(incremental_scenario_cost_no_adj, _interval='ci')
#
# fig, ax = plot_bars_with_uncertainty(
#     input_costs_no_adj_summ,
#     ylabel="Incremental cost (USD)",
# )
# plt.show()
#
# # -------------------------------------- Adjust costs as required... --------------------------------------------------
# # Output HCW time requirements by appointment type
# appointment_time_table = pd.read_csv(
#     resourcefilepath
#     / 'healthsystem'
#     / 'human_resources'
#     / 'definitions'
#     / 'ResourceFile_Appt_Time_Table.csv',
#     index_col=["Appt_Type_Code", "Facility_Level", "Officer_Category"])
# appt_type_facility_level_officer_category_to_appt_time = (appointment_time_table.Time_Taken_Mins.to_dict())
#
# # List HCW types
# officer_categories = appointment_time_table.index.levels[appointment_time_table.index.names.index(
#     "Officer_Category")].to_list()
#
# # Calculate HCW time use by treatment ID within each scenario
# hcw_time_by_treatment_id = bin_hsi_event_details(
#         results_folder,
#         lambda event_details, count: sum(
#             [
#                 Counter({
#                     (
#                         officer_category,
#                         event_details["treatment_id"]
#                     ):
#                     count
#                     * appt_number
#                     * appt_type_facility_level_officer_category_to_appt_time.get(
#                         (
#                             appt_type,
#                             event_details["facility_level"],
#                             officer_category
#                         ),
#                         0
#                     )
#                     for officer_category in officer_categories
#                 })
#                 for appt_type, appt_number in event_details["appt_footprint"]
#             ],
#             Counter()
#         ),
#         *TARGET_PERIOD,
#         True
#     )
#
# # Read in HCW capabilities data and sum across facility levels etc.
# daily_cap = pd.read_csv('./resources/healthsystem/human_resources/actual/ResourceFile_Daily_Capabilities.csv')
# daily_mins = daily_cap.set_index('Officer_Category')[['Total_Mins_Per_Day']]
# daily_mins = daily_mins.drop('Dental')
# daily_mins = daily_mins.drop('Nutrition')
# daily_mins = daily_mins.groupby(daily_mins.index).sum()
#
# # Next we calculate the average HCW capability
# # TODO: SHOULD WE ADJUST THIS FOR POPULATION GROWTH (SINCE WE'RE LOOKING AT 2026)
# yearly_mins = daily_mins * 365.25
# yearly_mins.rename(columns={'Total_Mins_Per_Day': 'mins_per_year'}, inplace=True)
# yearly_mins_sum = yearly_mins.sum(axis=0).to_frame().T
#
# # Next we calculate the total HCW time use
# hcw_time_by_treatment_id_df = pd.DataFrame.from_dict(hcw_time_by_treatment_id)
# hcw_time_by_treatment_id_df = hcw_time_by_treatment_id_df.fillna(0)
# hcw_time_by_treatment_id_df.index.names = ['first', 'second']
#
# # Now we calculate the demand of HCW time by cadre relative to yearly HCW capabilities
# def get_hcw_time_use_ratios(df):
#     """Get HCW time used/time available - compare that to the status QUO"""
#     def get_hcw_time_fold_diff(df_for_diff):
#         diff = df_for_diff.copy()
#         for col in df_for_diff.columns:
#             if col[0] != 0:
#                 base_col = (0, col[1])
#                 diff[col] = df_for_diff[col] / df_for_diff[base_col]
#             else:
#                 diff[col] = 1
#         return diff
#
#     # Calculate total and annual HCW time used
#     total_hcw_time = df.sum(axis=0).to_frame().T
#
#     # Divide total annual HCW time by total annual HCW time capabilities
#     total_hcw_time_ratio = total_hcw_time.div(yearly_mins_sum.iloc[:, 0], axis=0)
#     total_hcw_time_ratio.columns.names = ['draw', 'run']
#
#     # Find the fold change from the Status Quo scenario
#     fold_change = get_hcw_time_fold_diff(total_hcw_time_ratio)
#     total_hcw_time_ratio_fc = compute_summary_statistics(fold_change, use_standard_error=True)
#
#     annual_hcw_time_by_cadre = df.groupby(level='first').sum()
#
#     # Now we calculate the ratio of time use to time available (by cadre) and summarise it
#     hcw_time_ratio_by_cadre = annual_hcw_time_by_cadre.div(yearly_mins.iloc[:, 0], axis=0)
#     hcw_time_ratio_by_cadre.columns.names = ['draw', 'run']
#
#     # Now we get the diff from the SQ
#     fold_change_cadre = get_hcw_time_fold_diff(hcw_time_ratio_by_cadre)
#     hcw_time_ratio_by_cadre_fc = compute_summary_statistics(fold_change_cadre, use_standard_error=True)
#
#     # We also return the ratios for all scenarios across cadres (not the difference from the SQ)
#     hcw_time_ratio_by_cadre_summ = compute_summary_statistics(hcw_time_ratio_by_cadre, use_standard_error=True)
#
#     return [
#         total_hcw_time_ratio_fc,
#         hcw_time_ratio_by_cadre_fc,
#         hcw_time_ratio_by_cadre_summ,
#         fold_change_cadre,  # rows: officer category; columns: (draw, run)
#     ]
#
# hcw_ratios_unadjusted = get_hcw_time_use_ratios(hcw_time_by_treatment_id_df)
#
# # Now using the relative difference in demand we adjust cost
# # TODO: determine if costs should even be adjusted "down" if less HCW time is used...
# def return_cost_adjusted_for_hcw_growth(cost_data, fold_change_cadre):
#     adjusted = cost_data.copy()
#
#     multipliers = fold_change_cadre.stack(["draw", "run"])
#     multipliers.index.names = ["cost_subgroup", "draw", "run"]
#
#     hrh_rows = adjusted["cost_subgroup"].isin(fold_change_cadre.index)
#
#     row_keys = pd.MultiIndex.from_frame(
#         adjusted.loc[hrh_rows, ["cost_subgroup", "draw", "run"]]
#     )
#     multiplier = multipliers.reindex(row_keys).to_numpy()
#
#     if pd.isna(multiplier).any():
#         unmatched = adjusted.loc[
#             hrh_rows, ["cost_subgroup", "draw", "run"]
#         ].iloc[pd.isna(multiplier)].drop_duplicates()
#         raise ValueError(f"Missing HRH multipliers:\n{unmatched.to_string(index=False)}")
#
#     adjusted.loc[hrh_rows, "cost"] *= multiplier
#
#     total_input_cost = adjusted.groupby(["draw", "run"])["cost"].sum()
#     return [total_input_cost, adjusted]
#
# ratios_for_adjustment = hcw_ratios_unadjusted[3]
# input_costs_adjusted = return_cost_adjusted_for_hcw_growth(input_costs, ratios_for_adjustment)
#
# # -------------------------------------- ABOVE SERVICE-LEVEL COSTS ----------------------------------------------------
# # todo: finalise multiplier
# # todo: supplementary figure showing incremental costs across a range of above service level cost assumptions?
# # todo: i dont think this should be applied to all costs
# #  todo: above service level estimate should account for the "baked in" above service costs we already have
# #   (e.g. supply chain)
# above_service_level_cost_assumption = 1.15
#
#
# final_input_costs_adjusted = input_costs_adjusted[0].copy()
# mask = final_input_costs_adjusted.index.get_level_values("draw") != 0
# final_input_costs_adjusted.loc[mask] *= above_service_level_cost_assumption
#
# # ------------------------------------------ INCREMENTAL COSTS -------------------------------------------------------
# # TODO: summarise with confidence intervals?
# incremental_scenario_adj = (pd.DataFrame(
#         find_difference_relative_to_comparison(
#             final_input_costs_adjusted,
#             comparison=0,
#             scaled=False)  # sets the comparator to draw 0 which is the Actual scenario
#     ).T.iloc[0].unstack()).T
#
# input_costs_adj_summ = summarise_cost_data(incremental_scenario_adj, _interval='ci')
#
# fig, ax = plot_bars_with_uncertainty(
#     input_costs_adj_summ,
#     ylabel="Incremental cost (USD)",
# )
# plt.show()
#
#
# # ------------------------------------ COST PER DALY/DEATH AVERTED ---------------------------------------------------
# # ...
#
# # -------------------------------------- MAXIMUM ABILITY TO PAY -------------------------------------------------------
# # ...
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # def find_cost_diff_from_sq_and_sum(data):
# #     def find_difference_relative_to_comparison(_ser: pd.Series,
# #                                                comparison: str,
# #                                                scaled: bool = False,
# #                                                drop_comparison: bool = True,
# #                                                ):
# #         """Find the difference in the values in a pd.Series with a multi-index, between the draws (level 0)
# #         within the runs (level 1), relative to where draw = `comparison`.
# #         The comparison is `X - COMPARISON`."""
# #         return _ser \
# #             .unstack(level=0) \
# #             .apply(lambda x: (x - x[comparison]) / (x[comparison] if scaled else 1.0), axis=1) \
# #             .drop(columns=([comparison] if drop_comparison else [])) \
# #             .stack()
# #
# #     incremental_scenario_cost_annual = (
# #         find_difference_relative_to_comparison(
# #             data,
# #             comparison=0,
# #         )
# #         .swaplevel(0, 1)
# #         .rename_axis(["draw", "run"])
# #         .sort_index()
# #         .to_frame()
# #         .T
# #     )
# #
# #     incremental_scenario_cost_annual.index = ["incremental_cost"]
# #
# #     incremental_scenario_cost_summarized = summarize_confidence_intervals(incremental_scenario_cost_annual)
# #
# #     return [incremental_scenario_cost_annual, incremental_scenario_cost_summarized]
# #
# # incremental_scenario_costs = find_cost_diff_from_sq_and_sum(input_costs_adjusted_hcw[0])
# #
# # def dummy_fig_incremental_costs(
# #     cost_summary,
# #     draw_labels=draw_labels,
# #     scale=1_000_000,
# #     currency="USD",
# #     save_path=g_path,
# # ):
# #     """
# #     Plot mean incremental costs and their lower–upper intervals.
# #
# #     cost_summary: DataFrame indexed by draw, with mean/lower/upper columns.
# #     draw_labels:  Dictionary mapping draw numbers to scenario names.
# #     scale:        Divide costs by this amount for display (default: millions).
# #     """
# #     draw_labels = draw_labels or {}
# #
# #     data = cost_summary.loc[
# #         cost_summary.index != 0, ["mean", "lower", "upper"]
# #     ].sort_index().copy()
# #
# #     if data.empty:
# #         raise ValueError("No intervention draws found.")
# #
# #     values = data / scale
# #     y = np.arange(len(values))
# #
# #     fig, ax = plt.subplots(
# #         figsize=(10.5, max(5, 0.48 * len(values) + 1.5))
# #     )
# #
# #     ax.axvline(0, color="#667085", linewidth=1.25, zorder=0)
# #
# #     ax.hlines(
# #         y, values["lower"], values["upper"],
# #         color="#8DA7BE", linewidth=2.4, zorder=1
# #     )
# #     ax.plot(
# #         values["lower"], y, "|",
# #         color="#8DA7BE", markersize=10, markeredgewidth=1.5
# #     )
# #     ax.plot(
# #         values["upper"], y, "|",
# #         color="#8DA7BE", markersize=10, markeredgewidth=1.5
# #     )
# #     ax.scatter(
# #         values["mean"], y,
# #         s=75, color="#115E87", edgecolor="white",
# #         linewidth=1, zorder=3
# #     )
# #
# #     ax.set_yticks(
# #         y,
# #         [draw_labels.get(draw, f"Draw {draw}") for draw in values.index],
# #     )
# #     ax.invert_yaxis()
# #
# #     unit = "millions" if scale == 1_000_000 else f"÷ {scale:g}"
# #     ax.set_xlabel(f"Incremental cost ({currency}{unit})")
# #     ax.set_title(
# #         "Incremental cost by scenario",
# #         loc="left", fontsize=16, weight="bold", pad=17
# #     )
# #     ax.text(
# #         0, 1.015,
# #         "Point = mean  •  line = lower–upper interval",
# #         transform=ax.transAxes, fontsize=10, color="#475467"
# #     )
# #
# #     ax.grid(axis="x", color="#E5EAF0", linewidth=0.8)
# #     ax.set_axisbelow(True)
# #     for spine in ax.spines.values():
# #         spine.set_visible(False)
# #     ax.tick_params(axis="both", length=0, pad=8)
# #
# #     fig.tight_layout()
# #     plt.savefig(f'{g_path}/inc_costs.png', dpi=220, bbox_inches="tight")
# #     plt.show()
# #
# # dummy_fig_incremental_costs(
# #     incremental_scenario_costs[1],
# #     draw_labels=draw_labels,
# #     currency="$",  # change or omit as appropriate
# # )
#
# # ------------------------------------ COST PER DALY/DEATH AVERTED ---------------------------------------------------
# # cost = incremental_scenario_costs[0]
# #
# # def get_cost_per_outcome_averted(outcome_df):
# #     cost_per_outcome_averted = cost.reset_index(drop=True) / outcome_df.reset_index(drop=True)
# #     cost_per_total_outcome_averted_summ = summarize_confidence_intervals(cost_per_outcome_averted)
# #     return cost_per_total_outcome_averted_summ
# #
# #
# # cost_per_total_daly_averted = get_cost_per_outcome_averted(total_dalys_averted_df)
# # cost_per_adj_daly_averted = get_cost_per_outcome_averted(adj_dalys_averted_df)
# # cost_per_mnh_daly_averted = get_cost_per_outcome_averted(mat_neo_dalys_averted)
# #
#
# # # TODO - how to deal with no-change in deaths (0 values)
# total_deaths = num_deaths_by_cause_label.sum(axis=0).to_frame().T
# baseline = total_deaths.xs(0, axis=1, level="draw")
# total_deaths_averted_averted_df = baseline.sub(
#     total_deaths,
#     axis="columns",
#     level="run"
# ).drop(columns=0, level="draw")
#
#
# # cost_per_total_deaths_averted = get_cost_per_outcome_averted(total_deaths_averted_averted_df)
# # cost_per_mat_death_averted = get_cost_per_outcome_averted(direct_mat_deaths_averted_df)
# # cost_per_neo_death_averted = get_cost_per_outcome_averted(direct_neo_deaths_averted_df)
# # cost_per_stillbirth_averted = get_cost_per_outcome_averted(stillbirths_averted_df)
#
# def plot_cost_per_outcome(
#     cost,
#     outcomes,
#     draw_labels=None,
#     currency="USD",
# ):
#     """
#     cost: one-row DataFrame with MultiIndex columns ('draw', 'run').
#     outcomes: dict mapping outcome labels to DataFrames of the same structure.
#
#     Bars represent empirical run-level ratio percentiles, not confidence
#     intervals around the central ICER.
#     """
#     draw_labels = {} if draw_labels is None else draw_labels
#
#     def as_series(df):
#         if len(df) != 1:
#             raise ValueError("Each input must have exactly one row.")
#         if not isinstance(df.columns, pd.MultiIndex):
#             raise ValueError("Columns must be a MultiIndex: draw → run.")
#         if list(df.columns.names) != ["draw", "run"]:
#             raise ValueError("Column levels must be named 'draw' and 'run'.")
#         if not df.columns.is_unique:
#             raise ValueError("Each draw/run column must be unique.")
#         return df.iloc[0].astype(float)
#
#     cost_s = as_series(cost)
#     draws = list(cost_s.index.get_level_values("draw").unique())
#
#     fig, axes = plt.subplots(
#         1,
#         len(outcomes),
#         figsize=(5 * len(outcomes), max(5, 0.65 * len(draws))),
#         squeeze=False,
#         constrained_layout=True,
#     )
#
#     records = []
#
#     for ax, (outcome_name, outcome_df) in zip(axes.ravel(), outcomes.items()):
#         effect_s = as_series(outcome_df)
#
#         if (
#             len(effect_s) != len(cost_s)
#             or not cost_s.index.isin(effect_s.index).all()
#         ):
#             raise ValueError(
#                 f"{outcome_name}: draw/run columns do not match costs."
#             )
#
#         effect_s = effect_s.reindex(cost_s.index)
#         labels = []
#
#         for y, draw in enumerate(draws):
#             c = cost_s.xs(draw, level="draw").to_numpy()
#             e = effect_s.xs(draw, level="draw").to_numpy()
#
#             missing = np.isnan(c) | np.isnan(e)
#             valid = np.isfinite(c) & np.isfinite(e)
#             zero_effect = np.isfinite(e) & (e == 0)
#
#             with np.errstate(divide="ignore", invalid="ignore"):
#                 ratios = c / e
#
#             n_undefined = int((~np.isfinite(ratios)).sum())
#             notes = []
#
#             if zero_effect.any():
#                 notes.append(f"zero outcome: {zero_effect.sum()}")
#             if missing.any():
#                 notes.append(f"missing: {missing.sum()}")
#             if n_undefined:
#                 notes.append(f"undefined ratio: {n_undefined}")
#
#             estimate = lower = upper = np.nan
#
#             if valid.any():
#                 mean_effect = e[valid].mean()
#
#                 if mean_effect != 0:
#                     estimate = c[valid].mean() / mean_effect
#
#                 # Ratio intervals are interpretable here only when
#                 # every valid run has strictly positive benefit.
#                 if (e[valid] > 0).all():
#                     lower, upper = np.quantile(
#                         ratios[valid], [0.025, 0.975]
#                     )
#                     ax.hlines(y, lower, upper, color="steelblue", lw=2)
#                 elif (e[valid] < 0).all():
#                     notes.append("all valid runs: worse outcomes")
#                 elif (e[valid] < 0).any() and (e[valid] > 0).any():
#                     notes.append("benefit crosses zero")
#                 else:
#                     notes.append("benefit includes zero")
#
#             if np.isfinite(estimate):
#                 ax.plot(estimate, y, "o", color="black", markersize=5)
#             else:
#                 # x uses axes coordinates; y uses data coordinates.
#                 ax.text(
#                     0.98, y, "undefined",
#                     transform=ax.get_yaxis_transform(),
#                     ha="right", va="center",
#                     color="firebrick", fontsize=9,
#                 )
#
#             label = str(draw_labels.get(draw, f"Scenario {draw}"))
#             if notes:
#                 label += "\n[" + "; ".join(notes) + "]"
#             labels.append(label)
#
#             records.append({
#                 "outcome": outcome_name,
#                 "draw": draw,
#                 "central_icer": estimate,
#                 "run_ratio_p2.5": lower,
#                 "run_ratio_p97.5": upper,
#                 "zero_outcome_runs": int(zero_effect.sum()),
#                 "missing_runs": int(missing.sum()),
#                 "undefined_ratio_runs": n_undefined,
#                 "notes": "; ".join(notes),
#             })
#
#         ax.set(
#             yticks=np.arange(len(draws)),
#             yticklabels=labels,
#             title=outcome_name,
#             xlabel=f"{currency} per outcome averted",
#         )
#         ax.invert_yaxis()
#         ax.axvline(0, color="grey", lw=0.8)
#         ax.grid(axis="x", alpha=0.2)
#         ax.spines[["top", "right"]].set_visible(False)
#         plt.show()
#
#     return fig, pd.DataFrame(records)
#
# outcomes = {
#     "Total DALYs": total_dalys_averted_df,
#     "Adjusted DALYs": adj_dalys_averted_df,
#     "Maternal/newborn DALYs": mat_neo_dalys_averted,
#     "Total deaths": total_deaths_averted_averted_df,
#     "Maternal deaths": direct_mat_deaths_averted_df,
#     "Neonatal deaths": direct_neo_deaths_averted_df,
#     "Stillbirths": stillbirths_averted_df,
# }
#
# fig, plot_summary = plot_cost_per_outcome(
#     cost=incremental_scenario_costs[0],
#     outcomes=outcomes,
#     draw_labels=draw_labels,
#     currency="USD",  # Change to your costing currency
# )
#
#
# # -------------------------------------- MAXIMUM ABILITY TO PAY -------------------------------------------------------
#
#
#
#
# # def figure_3_incremental_costs(
# #     cost_summary,
# #     draw_labels=draw_labels,
# #     scale=1_000_000,
# #     currency="USD",
# #     save_path=g_path,
# # ):
# #     """
# #     Plot mean incremental costs and their lower–upper intervals.
# #
# #     cost_summary: DataFrame indexed by draw, with mean/lower/upper columns.
# #     draw_labels:  Dictionary mapping draw numbers to scenario names.
# #     scale:        Divide costs by this amount for display (default: millions).
# #     """
# #     draw_labels = draw_labels or {}
# #
# #     data = cost_summary.loc[
# #         cost_summary.index != 0, ["mean", "lower", "upper"]
# #     ].sort_index().copy()
# #
# #     if data.empty:
# #         raise ValueError("No intervention draws found.")
# #
# #     values = data / scale
# #     y = np.arange(len(values))
# #
# #     fig, ax = plt.subplots(
# #         figsize=(10.5, max(5, 0.48 * len(values) + 1.5))
# #     )
# #
# #     ax.axvline(0, color="#667085", linewidth=1.25, zorder=0)
# #
# #     ax.hlines(
# #         y, values["lower"], values["upper"],
# #         color="#8DA7BE", linewidth=2.4, zorder=1
# #     )
# #     ax.plot(
# #         values["lower"], y, "|",
# #         color="#8DA7BE", markersize=10, markeredgewidth=1.5
# #     )
# #     ax.plot(
# #         values["upper"], y, "|",
# #         color="#8DA7BE", markersize=10, markeredgewidth=1.5
# #     )
# #     ax.scatter(
# #         values["mean"], y,
# #         s=75, color="#115E87", edgecolor="white",
# #         linewidth=1, zorder=3
# #     )
# #
# #     ax.set_yticks(
# #         y,
# #         [draw_labels.get(draw, f"Draw {draw}") for draw in values.index],
# #     )
# #     ax.invert_yaxis()
# #
# #     unit = "millions" if scale == 1_000_000 else f"÷ {scale:g}"
# #     ax.set_xlabel(f"Incremental cost ({currency}{unit})")
# #     ax.set_title(
# #         "Incremental cost by scenario",
# #         loc="left", fontsize=16, weight="bold", pad=17
# #     )
# #     ax.text(
# #         0, 1.015,
# #         "Point = mean  •  line = lower–upper interval",
# #         transform=ax.transAxes, fontsize=10, color="#475467"
# #     )
# #
# #     ax.grid(axis="x", color="#E5EAF0", linewidth=0.8)
# #     ax.set_axisbelow(True)
# #     for spine in ax.spines.values():
# #         spine.set_visible(False)
# #     ax.tick_params(axis="both", length=0, pad=8)
# #
# #     fig.tight_layout()
# #
# #     if save_path is not None:
# #         fig.savefig(save_path, dpi=220, bbox_inches="tight")
# #
# #     return fig, ax
# #
# # fig, ax = figure_3_incremental_costs(
# #     incremental_scenario_cost_annual_summarized,
# #     draw_labels=draw_labels,
# #     currency="$",  # change or omit as appropriate
# #     save_path=g_path,
# # )
# # plt.show()
#
# # ====================================== DEBUG COST PLOT =============================================================
# cost_by_category = input_costs_adjusted_hcw[1].groupby(['draw', 'run', 'year', 'cost_category'])['cost'].sum()
# reformatted_cost_cat_df = (
#     cost_by_category
#     .reorder_levels(["year", "cost_category", "draw", "run"])
#     .unstack(["draw", "run"])
#     .sort_index()
#     .sort_index(axis=1)
# )
#
# baseline = reformatted_cost_cat_df.xs(0, level="draw", axis=1)
# difference_df = reformatted_cost_cat_df.subtract(
#     baseline,
#     axis="columns",
#     level="run"
# )
#
# summ_cost_cat_diff = summarize_confidence_intervals(difference_df)
#
# def get_cost_diff_by_type_panel_graph():
#     # Select the required year
#
#     year_to_plot = sim_start_year
#
#     # Example: dictionary mapping draw numbers to display labels
#
#     year_df = summ_cost_cat_diff.xs(year_to_plot, level="year")
#
#     # Identify draws, excluding baseline draw 0
#     draws = (
#         year_df.columns
#         .get_level_values("draw")
#         .unique()
#         .drop(0, errors="ignore")
#     )
#
#     cost_categories = year_df.index
#
#     fig, axes = plt.subplots(
#         2,
#         2,
#         figsize=(15, 10),
#         sharex=True,
#         sharey=False
#     )
#
#     axes = axes.flatten()
#
#     for ax, cost_category in zip(axes, cost_categories):
#
#         category_df = year_df.loc[cost_category]
#
#         means = np.array([
#             category_df.loc[(draw, "mean")]
#             for draw in draws
#         ])
#
#         lower = np.array([
#             category_df.loc[(draw, "lower")]
#             for draw in draws
#         ])
#
#         upper = np.array([
#             category_df.loc[(draw, "upper")]
#             for draw in draws
#         ])
#
#         # Matplotlib expects distances from the mean, not CI endpoints
#         yerr = np.vstack([
#             means - lower,
#             upper - means
#         ])
#
#         labels = [
#             draw_labels.get(draw, str(draw))
#             for draw in draws
#         ]
#
#         colours = [
#             "#D55E00" if value > 0 else "#0072B2"
#             for value in means
#         ]
#
#         ax.bar(
#             labels,
#             means,
#             yerr=yerr,
#             capsize=4,
#             color=colours,
#             width=0.75,
#             edgecolor="black",
#             linewidth=0.4,
#             error_kw={
#                 "elinewidth": 1,
#                 "ecolor": "black"
#             }
#         )
#
#         ax.axhline(
#             0,
#             color="black",
#             linewidth=0.8
#         )
#
#         ax.set_title(
#             str(cost_category).replace("_", " ").title()
#         )
#
#         ax.set_ylabel("Difference from baseline")
#
#         ax.yaxis.set_major_formatter(
#             mticker.FuncFormatter(
#                 lambda value, _: f"£{value / 1e6:,.1f}m"
#             )
#         )
#
#         ax.tick_params(
#             axis="x",
#             rotation=45,
#             labelbottom=True
#         )
#
#     # Remove unused panels if there are fewer than four categories
#     for ax in axes[len(cost_categories):]:
#         ax.remove()
#
#     fig.suptitle(
#         f"Difference in costs from baseline, {year_to_plot}",
#         fontsize=15
#     )
#
#     fig.tight_layout()
#     plt.savefig(f'{g_path}/diff_costs_by_category.png', bbox_inches='tight')
#
#     plt.show()
#
# get_cost_diff_by_type_panel_graph()
#
# #  CHECKING HSI COUNT DIFFERENCES
# def get_counts_of_hsi_by_short_treatment_id(_df):
#     """Get the counts of the short TREATMENT_IDs occurring (up to first underscore)"""
#     _counts_by_treatment_id = _df \
#         .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'TREATMENT_ID'] \
#         .apply(pd.Series) \
#         .sum() \
#         .astype(int)
#     # _short_treatment_id = _counts_by_treatment_id.index.map(lambda x: x.split('_')[0] + "*")
#     # return _counts_by_treatment_id.groupby(by=_short_treatment_id).sum()
#     return _counts_by_treatment_id
#
#
# counts_of_hsi_by_short_treatment_id = extract_results(
#     results_folder,
#     module='tlo.methods.healthsystem.summary',
#     key='HSI_Event',
#     custom_generate_series=get_counts_of_hsi_by_short_treatment_id,
#     do_scaling=False)
# counts_of_hsi_by_short_treatment_id.fillna(0)
#
# baseline = counts_of_hsi_by_short_treatment_id.xs(0, level="draw", axis=1)
# hsi_difference_df = counts_of_hsi_by_short_treatment_id.subtract(
#     baseline,
#     axis="columns",
#     level="run"
# )
# summ_hsi_diff = summarize_confidence_intervals(hsi_difference_df)
#
#
# def get_counts_of_appts(_df):
#     """Get the counts of appointments of each type being used."""
#     return _df \
#         .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'Number_By_Appt_Type_Code'] \
#         .apply(pd.Series) \
#         .sum() \
#         .astype(int)
#
# counts_of_appts = extract_results(
#     results_folder,
#     module='tlo.methods.healthsystem.summary',
#     key='HSI_Event',
#     custom_generate_series=get_counts_of_appts,
#     do_scaling=False
# )
#
# baseline = counts_of_appts.xs(0, level="draw", axis=1)
# appt_difference_df = counts_of_appts.subtract(
#     baseline,
#     axis="columns",
#     level="run"
# )
# summ_apt_diff = summarize_confidence_intervals(appt_difference_df)
#
# # figure_S2_heatmap_cause_specific_dalys_averted(summ_apt_diff,
# #                                             'Diff in appt types',
# #                                             'fig_diff_in_apt_types', True)
# # figure_S2_heatmap_cause_specific_dalys_averted(summ_hsi_diff,
# #                                             'Diff in hsi types',
# #                                             'fig_diff_in_hsi', True)
#
# medical_consumables_df = (
#         input_costs[input_costs["cost_category"].eq("medical consumables")]
#           .pivot_table(
#               index="year",
#               columns=["draw", "run"],
#               values="cost",
#               aggfunc="sum"
#           )
#           .sort_index(axis=1))
#
# # TODO make more robust so we dont have to define year
# medical_consumables_summ = summarize_confidence_intervals(medical_consumables_df)
# medical_cons_by_scenario = {k: [medical_consumables_summ.loc[sim_start_year, (d, 'lower')],
#                                 medical_consumables_summ.loc[sim_start_year, (d, 'mean')],
#                                 medical_consumables_summ.loc[sim_start_year, (d, 'upper')]] for k, d in zip (int_analysis, draws)}
#
# def drop_outside_period(_df):
#     """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
#     return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])
#
# def get_counts_of_items_requested(_df):
#     _df = drop_outside_period(_df)
#
#     counts_of_available = defaultdict(int)
#     counts_of_not_available = defaultdict(int)
#
#     for _, row in _df.iterrows():
#         for item, num in row['Item_Used'].items():
#             counts_of_available[item] += num
#         for item, num in row['Item_NotAvailable'].items():  # eval(row['Item_NotAvailable'])
#             counts_of_not_available[item] += num
#
#     return pd.concat(
#         {'Used': pd.Series(counts_of_available), 'Not_Available': pd.Series(counts_of_not_available)},
#         axis=1
#     ).fillna(0).astype(int).stack()
#
# cons_req = extract_results(
#     results_folder,
#     module='tlo.methods.healthsystem.summary',
#     key='Consumables',
#     custom_generate_series=get_counts_of_items_requested,
#     do_scaling=False)  # todo change to False
# cons_req = cons_req * p_scaling_factor
# cons_req.fillna(0, inplace=True)
#
# cons_costs = pd.read_csv(Path("./resources/costing") / 'ResourceFile_Costing_Consumables.csv')
#
# # costed_scenario_cons = cons_req.copy()
# price_lookup = (
#     cons_costs
#     .assign(Item_Code=cons_costs["Item_Code"].astype(str).str.strip())
#     .set_index("Item_Code")["Price_per_unit"])
#
# item_codes = (
#     cons_req.index.get_level_values(0)
#     .astype(str)
#     .str.strip())
# prices = item_codes.map(price_lookup)
# missing_price_mask = prices.isna()
# multipliers = prices.fillna(1)
# costed_scenario_cons = cons_req.mul(multipliers.to_numpy(), axis=0)
#
# diff_results = {}
#
# def get_cost_used_cons(level):
#     return costed_scenario_cons.loc[costed_scenario_cons.index.get_level_values(1) == "Used"].xs(level, level=0,
#                                                                                                  axis=1)
#
# baseline = get_cost_used_cons(0)
# for draw, int in zip(draws, int_analysis):
#     diff_df = get_cost_used_cons(draw) - baseline
#     diff_df.columns = pd.MultiIndex.from_tuples([(draw, v) for v in range(len(diff_df.columns))],
#                                                 names=['draw', 'run'])
#     diff_df.index = diff_df.index.droplevel(1)
#     results_diff = summarize_confidence_intervals(diff_df)
#     results_diff.fillna(0)
#     diff_results.update({int: results_diff})
#
# for k in diff_results.keys():
#
#     if k != "baseline":
#         # Extract values
#         categories = np.array(list(diff_results[k].index))
#
#         mins = np.array([arr[0] for arr in diff_results[k].values])
#         means = np.array([arr[1] for arr in diff_results[k].values])
#         maxs = np.array([arr[2] for arr in diff_results[k].values])
#
#         # Sort by mean difference
#         order = np.argsort(means)
#
#         categories = categories[order]
#         mins = mins[order]
#         means = means[order]
#         maxs = maxs[order]
#
#         y = np.arange(len(categories))
#
#         # Error bars
#         errors = np.vstack([
#             means - mins,
#             maxs - means
#         ])
#
#         # Identify top/bottom 10
#         bottom10 = np.argsort(means)[:10]
#         top10 = np.argsort(means)[-10:]
#
#         # Plot
#         fig, ax = plt.subplots(figsize=(11, 34))
#
#         # All consumables
#         ax.errorbar(
#             means,
#             y,
#             xerr=errors,
#             fmt='o',
#             color='lightgrey',
#             ecolor='lightgrey',
#             markersize=3,
#             capsize=2,
#             elinewidth=0.8,
#             linewidth=0.8,
#             label='Other consumables'
#         )
#
#         # Largest decreases
#         ax.errorbar(
#             means[bottom10],
#             y[bottom10],
#             xerr=errors[:, bottom10],
#             fmt='o',
#             color='red',
#             ecolor='red',
#             markersize=5,
#             capsize=3,
#             elinewidth=1,
#             label='10 largest decreases'
#         )
#
#         # Largest increases
#         ax.errorbar(
#             means[top10],
#             y[top10],
#             xerr=errors[:, top10],
#             fmt='o',
#             color='green',
#             ecolor='green',
#             markersize=5,
#             capsize=3,
#             elinewidth=1,
#             label='10 largest increases'
#         )
#
#         # Baseline reference line
#         ax.axvline(0, color='black', linestyle='--', linewidth=1)
#
#         # Show all consumable IDs
#         ax.set_yticks(y)
#         ax.set_yticklabels(categories, fontsize=7)
#
#         # Labels/title
#         ax.set_xlabel("Crude difference from baseline scenario")
#         ax.set_ylabel("Consumable item")
#         ax.set_title(
#             f"Difference in Cost per Consumable from Baseline Scenario vs {k}"
#         )
#
#         # Grid only on x-axis
#         ax.grid(axis="x", alpha=0.35)
#
#         # Legend
#         ax.legend(loc="best")
#
#         # Save and show
#         fig.tight_layout()
#
#         fig.savefig(
#             f"{g_path}/cons_cost_diff_{k}_horizontal_highlighted.png",
#             dpi=300,
#             bbox_inches="tight"
#         )
#
#         plt.show()
#
#
# # todo: why lower del
