from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
from collections import Counter, defaultdict

import os
from scipy.stats import t

import pandas as pd
from tableone import TableOne

# from scripts.comparison_of_horizontal_and_vertical_programs.economic_analysis_for_manuscript.roi_analysis_horizontal_vs_vertical import \
#     icers_summarized
from tlo import Date
from tlo.analysis.utils import bin_hsi_event_details, extract_results, get_scenario_outputs, get_scenario_info, parse_log_file, compute_summary_statistics
from src.scripts.costing.cost_estimation import (do_stacked_bar_plot_of_cost_by_category,
    estimate_input_cost_of_scenarios, summarize_cost_data
)

# Get results file
outputspath = './outputs/sejjj49@ucl.ac.uk/'
resourcefilepath = Path("./resources")

scenario = 'testing_scenario_172156'
results_folder= get_scenario_outputs(scenario, outputspath)[-1]
sim_start_year = 2025

# Create a folder to store graphs (if it hasn't already been created when ran previously)
g_path = f'{outputspath}figures_{scenario}'

if not os.path.isdir(g_path):
        os.makedirs(f'{outputspath}figures_{scenario}')

# Get scenario details
info = get_scenario_info(results_folder)
draws = [x for x in range(info['number_of_draws'])]

modelled_pop = 40_000
# TODO - find source for predicted pregnancies in 2026
p_scaling_factor = 750_000 / modelled_pop

int_analysis = ['baseline',
                'abortion',
                'mat_sepsis_cm',
                'pph_cm',
                'ol_cm',
                'spe_ec_cm',
                'cs_surg',
                'neo_sep_cm',
                'preterm_cm',
                'neo_resus']

scenario_names = ['Status quo',
                  'Post-abortion and ectopic pregnancy case management',
                  'Maternal sepsis case management',
                  'Postpartum haemorrhage case management',
                  'Severe pre-eclampsia/eclampsia case management',
                  'Caesarean section/other obstetric surgery',
                  'Neonatal sepsis case management',
                  'Preterm birth case management',
                  'Newborn resuscitation']

draw_labels = {1: 'Abortion CM',
                   2:'Maternal sepsis CM',
                   3: 'Haemorrhage CM',
                   4: 'Obstructed labour CM',
                   5: 'Severe pre-eclampsia CM',
                   6: 'CS/Surgery',
                   7: 'Neonatal sepsis CM',
                   8: 'Preterm birth CM',
                   9: 'Newborn resus.'}

#  ======================================= DEFINE HELPER FUNCTIONS  =================================================
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

#  ========================================== EXTRACT CORE DATA  =====================================================

results = {k:get_ps_data_frames(k, results_folder) for k in
           ['mat_comp_incidence', 'nb_comp_incidence', 'deaths_and_stillbirths','service_coverage', 'met_need',
            'yearly_mnh_counter_dict', 'intervention_coverage']}

#  ======================================= FIGURE 1 - MET NEED  =====================================================
met_need_df = results['met_need']['summarised']
df = met_need_df

def produce_fig_1():

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
                ("haem_cm_ut", "Uterotonics"),
                ("haem_cm_mrp", "MRRP"),
                ("haem_cm_blood_pph", "Blood (PPH)"),
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
                ("cs_surg_aph", "CS and IP surgery"),
                ("cs_surg_pph", "PP surgery")
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

    plt.savefig(
        f"{g_path}/figure1_met_need.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.show()

produce_fig_1()

#  ======================================= FIGURE 2 - HEALTH IMPACTS  =================================================
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

num_deaths_by_cause_label.fillna(0)

direct_deaths = num_deaths_by_cause_label.loc['Maternal Disorders'].reindex(num_deaths_by_cause_label.columns).to_frame().T
# direct_deaths = direct_deaths * p_scaling_factor
direct_deaths.rename(index={"Maternal Disorders": sim_start_year}, inplace=True)

baseline = direct_deaths.xs(0, axis=1, level="draw")
direct_mat_deaths_averted_df = baseline.sub(
    direct_deaths,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
direct_mat_deaths_averted_df.rename(index={sim_start_year: "mat_direct_deaths_averted"}, inplace=True)

# todo: we may reinstate this later but currently i think focusing on direct maternal death outcomes here is best
# def extract_indirect_deaths_non_hiv(df):
#     year = pd.to_datetime(df["date"]).dt.year
#
#     pregnant_or_postpartum = (
#         df["is_pregnant"].fillna(False).astype(bool)
#         | df["la_is_postpartum"].fillna(False).astype(bool)
#     )
#
#     relevant_cause = (
#         df["cause_of_death"].str.contains(
#             r"Malaria|Suicide|ever_stroke|diabetes|"
#             r"chronic_ischemic_hd|ever_heart_attack|"
#             r"chronic_kidney_disease",
#             na=False,
#             regex=True,
#         )
#         | df["cause_of_death"].eq("TB")
#     )
#
#     return (
#         year[pregnant_or_postpartum & relevant_cause]
#         .value_counts()
#         .sort_index()
#         .rename_axis("year")
#         .rename("deaths")
#     )
#
#
# indirect_deaths_non_hiv = extract_results(
#     results_folder,
#     module="tlo.methods.demography.detail",
#     key="properties_of_deceased_persons",
#     custom_generate_series=extract_indirect_deaths_non_hiv,
#     do_scaling=False,
# )
# indirect_deaths_non_hiv_final = indirect_deaths_non_hiv.fillna(0)
# # indirect_deaths_non_hiv_final = indirect_deaths_non_hiv_final * p_scaling_factor
#
# # Deaths due to AIDS during/following pregnancy are adjusted in line with UN MMEIG methodology
# hiv_pd = extract_results(
#     results_folder,
#     module="tlo.methods.demography.detail",
#     key="properties_of_deceased_persons",
#     custom_generate_series=lambda df: (
#         df.assign(year=pd.to_datetime(df["date"]).dt.year)
#         .loc[
#             (
#                 df["is_pregnant"].fillna(False).astype(bool)
#                 | df["la_is_postpartum"].fillna(False).astype(bool)
#             )
#             & df["cause_of_death"].str.contains(
#                 r"^(?:AIDS_non_TB|AIDS_TB)$",
#                 na=False,
#                 regex=True,
#             )
#         ]
#         .groupby("year")
#         .size()
#         .rename("deaths")
#     ),
#     do_scaling=False,
# )
# # TODO not sure about this logic...
# hiv_pd = hiv_pd.fillna(0)
# # hiv_pd = hiv_pd * p_scaling_factor
#
# hiv_indirect_maternal_deaths = hiv_pd * 0.3
# hiv_indirect_maternal_deaths = hiv_indirect_maternal_deaths.round(0)
#
# # The MMR is calculated from total deaths extracted above using live births as a denominator
# indirect_deaths_final = indirect_deaths_non_hiv_final + hiv_indirect_maternal_deaths
# total_mat_deaths = direct_deaths + indirect_deaths_final
#
# baseline = total_mat_deaths.xs(0, axis=1, level="draw")
# all_mat_deaths_averted_df = baseline.sub(
#     total_mat_deaths,
#     axis="columns",
#     level="run"
# ).drop(columns=0, level="draw")
#
# all_mat_deaths_averted_df.rename(index={sim_start_year: "all_mat_deaths_averted"}, inplace=True)

# 2.) Maternal DALYs (direct)
dalys_by_cause = extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns=['date', 'sex', 'age_range']).groupby(['year']).sum().stack()),
            do_scaling=False)
dalys_by_cause = dalys_by_cause.loc[dalys_by_cause.index.get_level_values(0) != 2026]

mat_dalys = dalys_by_cause.loc[sim_start_year, 'Maternal Disorders'].reindex(dalys_by_cause.columns).to_frame().T
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
# direct_neo_deaths_final = direct_neo_deaths_final * p_scaling_factor

baseline = direct_neo_deaths_final.xs(0, axis=1, level="draw")

direct_neo_deaths_averted_df = baseline.sub(
    direct_neo_deaths_final,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")

direct_neo_deaths_averted_df.rename(index={sim_start_year: "direct_neo_deaths_averted"}, inplace=True)

# todo: again, at the moment, only looking at direct causes of death
# nd = extract_results(
#     results_folder,
#     module="tlo.methods.demography.detail",
#     key="properties_of_deceased_persons",
#     custom_generate_series=(
#         lambda df: df.loc[(df['age_days'] < 29)].assign(
#             year=df['date'].dt.year).groupby(['year'])['year'].count()),
#     do_scaling=False)
# neo_deaths = nd.fillna(0)
# # neo_deaths = neo_deaths * p_scaling_factor
#
# baseline = neo_deaths.xs(0, axis=1, level="draw")
# all_neo_deaths_averted_df = baseline.sub(
#     neo_deaths,
#     axis="columns",
#     level="run"
# ).drop(columns=0, level="draw")
# all_neo_deaths_averted_df.rename(index={sim_start_year: "all_neo_deaths_averted"}, inplace=True)

# 4.) Neonatal DALYs (direct)
neo_dalys = dalys_by_cause.loc[sim_start_year, 'Neonatal Disorders'].reindex(dalys_by_cause.columns).to_frame().T
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
# stillbirths = stillbirths * p_scaling_factor
baseline = stillbirths.xs(0, axis=1, level="draw")
stillbirths_averted_df = baseline.sub(
    stillbirths,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
stillbirths_averted_df.rename(index={'total_stillbirths': "stillbirths_averted"}, inplace=True)

# 6.) All cause DALYs averted
total_dalys = dalys_by_cause.groupby(['year']).sum()

baseline = total_dalys.xs(0, axis=1, level="draw")
total_dalys_averted_df = baseline.sub(
    total_dalys,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")

total_dalys_averted_df.rename(index={sim_start_year: "total_dalys_averted"}, inplace=True)

# 7.) All cause DALYs averted (inc. stillbirths)
#  TODO: determine GA that whill be included
preg_loss = extract_results(
    results_folder,
    module="tlo.methods.pregnancy_supervisor",
    key="pregnancy_loss",
    custom_generate_series=(
        lambda df: df.loc[(df['gest_age'] > 28)].assign(
            year=df['date'].dt.year).groupby(['year'])['year'].count()),
    do_scaling=False)
preg_loss_yll = preg_loss * 90

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
    "neo_direct_deaths_averted": "Direct neonatal deaths averted",
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

def produce_fig_2(
    death_data,
    daly_data,
    death_labels,
    daly_labels,
    save_title
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

            ax.bar(
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
            title="Outcome",
            frameon=False,
            loc="upper right"
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
    "fig_2_health_outcomes"
)

# Table 1/Figure 3 - Estimating costs

# Extract initial system level cost data
# list_of_relevant_years_for_costing = list(range(TARGET_PERIOD[0].year, TARGET_PERIOD[-1].year + 1))
# input_costs_df = estimate_input_cost_of_scenarios(results_folder=results_folder,
#                                      resourcefilepath=resourcefilepath,
#                                      suspended_results_folder=results_folder,
#                                      _draws=draws,
#                                      _years=list_of_relevant_years_for_costing,
#                                      cost_only_used_staff= True,
#                                      alt_scaling_factor=p_scaling_factor)
#
# input_costs_df.to_csv(f'{g_path}/input_costs.csv')

input_costs = pd.read_csv(f'{g_path}/input_costs.csv')
input_costs = input_costs.set_index('Unnamed: 0')
input_cost_unadjusted = input_costs

# Adjust costs as required...

# 1.) HRH
def get_ratios():

    appointment_time_table = pd.read_csv(
        resourcefilepath
        / 'healthsystem'
        / 'human_resources'
        / 'definitions'
        / 'ResourceFile_Appt_Time_Table.csv',
        index_col=["Appt_Type_Code", "Facility_Level", "Officer_Category"]
    )

    appt_type_facility_level_officer_category_to_appt_time = (
        appointment_time_table.Time_Taken_Mins.to_dict()
    )

    officer_categories = appointment_time_table.index.levels[
        appointment_time_table.index.names.index("Officer_Category")
    ].to_list()

    hcw_time_by_treatment_id = bin_hsi_event_details(
        results_folder,
        lambda event_details, count: sum(
            [
                Counter({
                    (
                        officer_category,
                        event_details["treatment_id"]
                    ):
                        count
                        * appt_number
                        * appt_type_facility_level_officer_category_to_appt_time.get(
                            (
                                appt_type,
                                event_details["facility_level"],
                                officer_category
                            ),
                            0
                        )
                    for officer_category in officer_categories
                })
                for appt_type, appt_number in event_details["appt_footprint"]
            ],
            Counter()
        ),
        *TARGET_PERIOD,
        True
    )

    # Next we calculate the total HCW time use
    hcw_time_by_treatment_id_df = pd.DataFrame.from_dict(hcw_time_by_treatment_id)
    hcw_time_by_treatment_id_df = hcw_time_by_treatment_id_df.fillna(0)
    hcw_time_by_treatment_id_df.index.names = ['first', 'second']
    hcw_time_by_cadre = hcw_time_by_treatment_id_df.groupby(level='first').sum()

    # Read in capabilities data and sum across facility levels etc.
    daily_cap = pd.read_csv('./resources/healthsystem/human_resources/actual/ResourceFile_Daily_Capabilities.csv')
    daily_mins = daily_cap.set_index('Officer_Category')[['Total_Mins_Per_Day']]
    daily_mins = daily_mins.drop('Dental')
    daily_mins = daily_mins.drop('Nutrition')
    daily_mins = daily_mins.groupby(daily_mins.index).sum()

    # Next we calculate the average HCW capabilities assuming capabilities increase yearly in line with population growth
    yrly_hcw_time_cap = daily_mins * 365.25

    # Now we calculate the ratio of time use to time available (by cadre) and summarise it
    hcw_time_ratio_by_cadre = hcw_time_by_cadre.div(yrly_hcw_time_cap.iloc[:, 0], axis=0)
    hcw_time_ratio_by_cadre.columns.names = ['draw', 'run']

    hcw_time_ratio_by_cadre_summ = compute_summary_statistics(hcw_time_ratio_by_cadre, use_standard_error=True)

    return hcw_time_ratio_by_cadre_summ

hcw_ratios = get_ratios()

def return_cost_adjusted_for_hcw_growth(cost_data, hcw_ratios):
    # Multiply the HCW cost estimates by ratios
    central_df = hcw_ratios.xs('central', axis=1, level=1)

    # Function to safely get multiplier
    def get_multiplier(row):
        subgroup = row['cost_subgroup']
        draw = row['draw']
        if subgroup in central_df.index and draw in central_df.columns:
            return central_df.loc[subgroup, draw]
        else:
            return 1.0  # or np.nan, or row['cost'] unmodified depending on your logic

    cost_data['cost'] = cost_data.apply(lambda row: row['cost'] * get_multiplier(row), axis=1)
    total_input_cost = cost_data.groupby(['draw', 'run'])['cost'].sum()

    return total_input_cost

input_cost_unadj = input_cost_unadjusted.groupby(['draw', 'run'])['cost'].sum()

input_costs_adjusted_hcw = return_cost_adjusted_for_hcw_growth(input_costs, hcw_ratios)


# QUESTIONS:
# 1.) Do we need to discount costs/DALYs if time horizon is 1 year
# 2.) Are we only costing HRH that were used OR are do we cost all given we're then going to calculate additional costs

# 1.) CONSUMABLES

# 2.) HRH
# (we can cost only those cadres used in the simulation or not)

# We want different in HCW time use between scenarios

# 3.) SENSITIVITY/ABOVE SERVICE COSTS
