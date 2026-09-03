from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

import matplotlib.colors as colors
import seaborn as sns

import os
from scipy.stats import t

import pandas as pd
from tableone import TableOne

# from scripts.comparison_of_horizontal_and_vertical_programs.economic_analysis_for_manuscript.roi_analysis_horizontal_vs_vertical import \
#     icers_summarized
from tlo import Date
from tlo.analysis.utils import extract_results, get_scenario_outputs, get_scenario_info, parse_log_file
from src.scripts.costing.cost_estimation import (do_stacked_bar_plot_of_cost_by_category,
    estimate_input_cost_of_scenarios, summarize_cost_data
)

outputspath = './outputs/sejjj49@ucl.ac.uk/'
resourcefilepath = Path("./resources")

# create_pickles_locally(results_folder, compressed_file_name_prefix='block_intervention_big_run')

#  ======================================= DEFINE SCENARIO INFORMATION  ===============================================
scenario = 'testing_scenario_172156'
results_folder= get_scenario_outputs(scenario, outputspath)[-1]
sim_start_year = 2025

# Create a folder to store graphs (if it hasn't already been created when ran previously)
g_path = f'{outputspath}graphs_{scenario}'

if not os.path.isdir(g_path):
        os.makedirs(f'{outputspath}graphs_{scenario}')

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

info = get_scenario_info(results_folder)
draws = [x for x in range(info['number_of_draws'])]

# estimate a scaling factor for pregnancies
modelled_pop = 40_000
# TODO - find source for predicted pregnancies in 2026
p_scaling_factor = 750_000 / modelled_pop

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

# Access dataframes generated from pregnancy supervisor
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

mat_deaths_dalys = get_deaths_dalys_demog('Maternal', 100_000)
neo_deaths_dalys = get_deaths_dalys_demog('Neonatal', 1000)

# Get combined dalys
total_dalys = pd.DataFrame(
    mat_deaths_dalys[4].to_numpy() + neo_deaths_dalys[4].to_numpy(),
    index=['total_dalys'],
    columns=mat_deaths_dalys[4].columns)
total_dalys_sum = summarize_confidence_intervals(total_dalys)

# Get still births
stillbirths = results['deaths_and_stillbirths']['crude'].loc[['intrapartum_stillbirths']]

# Add IP stillbirth to DALYs
# TODO finalise YLL for stillbirths
yll_from_sb = stillbirths * 90.0
adjusted_dalys = pd.DataFrame(
    total_dalys.to_numpy() + yll_from_sb.to_numpy(),
    index=['adj_total_dalys'],
    columns=total_dalys.columns)
adj_dalys_summ = summarize_confidence_intervals(adjusted_dalys)

results.update({
                'mat_deaths': {'crude': mat_deaths_dalys[0], 'summarised': mat_deaths_dalys[1]},
                'neo_deaths': {'crude': neo_deaths_dalys[0], 'summarised': neo_deaths_dalys[1]},
                'mmr': {'crude': mat_deaths_dalys[2], 'summarised': mat_deaths_dalys[3]},
                'nmr': {'crude': neo_deaths_dalys[2], 'summarised': neo_deaths_dalys[3]},
                'mat_dalys': {'crude': mat_deaths_dalys[4], 'summarised': mat_deaths_dalys[5]},
                'neo_dalys': {'crude': neo_deaths_dalys[4], 'summarised': neo_deaths_dalys[5]},
                'total_dalys': {'crude': total_dalys, 'summarised': total_dalys_sum},
                'adj_total_dalys': {'crude': adjusted_dalys, 'summarised': adj_dalys_summ},
                })

#  ===================================== EXTRACT AND ADJUST COST DATA  ===============================================
TARGET_PERIOD = (Date(sim_start_year, 1, 1), Date(sim_start_year, 12, 31))
# list_of_relevant_years_for_costing = list(range(TARGET_PERIOD[0].year, TARGET_PERIOD[-1].year + 1))
# #
# input_costs_df = estimate_input_cost_of_scenarios(results_folder=results_folder,
#                                      resourcefilepath=resourcefilepath,
#                                      suspended_results_folder=results_folder,
#                                      _draws=draws,
#                                      _years=list_of_relevant_years_for_costing,
#                                      cost_only_used_staff= True,
#                                      _discount_rate=0.03)
#
# input_costs_df.to_csv(f'{g_path}/input_costs.csv')
#
# # Read in costs (takes a long time to generate)
# # TODO: COST RESULTS ARE SCALED TO POPULATION...
# input_costs = pd.read_csv(f'{g_path}/input_costs.csv')
# input_costs = input_costs.set_index('Unnamed: 0')


# ============================================ FIG 1 - MET NEED =======================================================
met_need_df = results['met_need']['summarised']
df = met_need_df

def produce_fig_1():
    # Define each scenario:
    # - name: package/scenario name
    # - draw: draw in which coverage was increased
    # - items: (DataFrame row ID, readable component label)
    scenarios = [
        {
            "name": "Abortion/Ectopic CM",
            "draw": 1,
            "items": [
                ("pac_ep", "Post-abortion care")
            ]
        },
        {
            "name": "Maternal sepsis CM",
            "draw": 2,
            "items": [
                ("m_sepsis_cm", "Sepsis management")
            ]
        },
        {
            "name": "Maternal haem. CM",
            "draw": 3,
            "items": [
                ("haem_cm_ut", "Uterotonics"),
                ("haem_cm_mrp", "MRRP"),
                ("haem_cm_blood_pph", "Blood (PPH)"),
                ("heam_cm_blood_aph", "Blood (APH)")
            ]
        },
        {
            "name": "Obstructed labour CM",
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
            "name": "CS & Obstetric surgery",
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

    # Plot settings
    baseline_colour = "#BDBDBD"
    increased_colour = "#377EB8"

    bar_width = 0.36
    component_spacing = 1.0
    package_gap = 0.8

    # Hatching distinguishes components within packages
    hatches = ["", "..", "//", "xx", "\\\\", "++", "oo"]

    # Validate DataFrame structure
    required_stats = {"lower", "mean", "upper"}
    available_draws = set(df.columns.get_level_values("draw"))
    available_stats = set(df.columns.get_level_values("stat"))

    if 0 not in available_draws:
        raise KeyError("Baseline draw 0 is not present in the DataFrame.")

    if not required_stats.issubset(available_stats):
        raise ValueError(
            f"Missing required statistics: "
            f"{required_stats - available_stats}"
        )

    # Extract results and calculate positions
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

            baseline_mean = df.loc[row_id, (0, "mean")]
            baseline_lower = df.loc[row_id, (0, "lower")]
            baseline_upper = df.loc[row_id, (0, "upper")]

            increased_mean = df.loc[row_id, (draw, "mean")]
            increased_lower = df.loc[row_id, (draw, "lower")]
            increased_upper = df.loc[row_id, (draw, "upper")]

            plot_data.append({
                "package": scenario["name"],
                "component": item_label,
                "x": current_x,
                "hatch": hatches[
                    component_number % len(hatches)
                ],
                "baseline_mean": baseline_mean,
                "baseline_lower_error": (
                    baseline_mean - baseline_lower
                ),
                "baseline_upper_error": (
                    baseline_upper - baseline_mean
                ),
                "increased_mean": increased_mean,
                "increased_lower_error": (
                    increased_mean - increased_lower
                ),
                "increased_upper_error": (
                    increased_upper - increased_mean
                )
            })

            package_positions.append(current_x)
            current_x += component_spacing

        package_centres.append({
            "name": scenario["name"],
            "x": np.mean(package_positions)
        })

        # Position immediately after the final component in this package
        if scenario_number < len(scenarios) - 1:
            package_boundaries.append(
                current_x - component_spacing / 2 + package_gap / 2
            )

        current_x += package_gap

    # Create plot
    fig_width = max(16, len(plot_data) * 1.25)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    upper_limits = []

    # Plot each package component separately
    for item in plot_data:

        x_position = item["x"]

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
            hatch=item["hatch"],
            error_kw={"elinewidth": 1}
        )

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
            hatch=item["hatch"],
            error_kw={"elinewidth": 1}
        )

        # Percentage labels
        ax.bar_label(
            baseline_bar,
            labels=[f'{item["baseline_mean"]:.1f}%'],
            padding=7,
            fontsize=8,
            rotation=90
        )

        ax.bar_label(
            increased_bar,
            labels=[f'{item["increased_mean"]:.1f}%'],
            padding=7,
            fontsize=8,
            rotation=90
        )

        upper_limits.extend([
            item["baseline_mean"] +
            item["baseline_upper_error"],

            item["increased_mean"] +
            item["increased_upper_error"]
        ])

    # Component labels beneath each baseline/increased pair
    ax.set_xticks([
        item["x"] for item in plot_data
    ])

    ax.set_xticklabels(
        [item["component"] for item in plot_data],
        rotation=45,
        ha="right",
        fontsize=9
    )

    # Package names centred beneath their components
    for package in package_centres:
        ax.text(
            package["x"],
            -0.28,
            package["name"],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            clip_on=False
        )

    # Separators between packages
    for boundary in package_boundaries:
        ax.axvline(
            boundary,
            color="0.82",
            linewidth=0.9,
            linestyle="--",
            zorder=0
        )

    # Coverage legend
    legend_handles = [
        Patch(
            facecolor=baseline_colour,
            edgecolor="black",
            label="Baseline coverage"
        ),
        Patch(
            facecolor=increased_colour,
            edgecolor="black",
            label="Increased coverage"
        )
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper right"
    )

    ax.set_ylabel("Coverage (%)")
    ax.set_xlabel("")

    ax.set_ylim(
        0,
        max(upper_limits) * 1.20
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Make space for component and package labels
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.96,
        bottom=0.32
    )
    plt.savefig(f'{g_path}/figure1_met_need.png', bbox_inches='tight')
    plt.show()
# produce_fig_1()

# ======================================== TEST FIGURE - change in DALYS and incidence =============================================
dalys_by_cause = extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns=['date', 'sex', 'age_range']).groupby(['year']).sum().stack()),
            do_scaling=False)
dalys_by_cause = dalys_by_cause.loc[dalys_by_cause.index.get_level_values(0) != 2026]
# dalys_by_cause = dalys_by_cause * p_scaling_factor
dalys_by_cause = dalys_by_cause.droplevel(0, axis=0)


baseline = dalys_by_cause.xs(0, axis=1, level="draw")
dalys_averted_by_cause_df = baseline.sub(
    dalys_by_cause,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
dalys_averted_by_cause_df = summarize_confidence_intervals(dalys_averted_by_cause_df)


def figure_heatmap_cause_specific_dalys_averted(
        data,
        title,
        save_title,
        compact_annotations=True,

):
    """
    Outputs a heatmap showing DALYs averted by cause.

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

figure_heatmap_cause_specific_dalys_averted(dalys_averted_by_cause_df,
                                            'DALYs averted by cause and intervention scenario',
                                            'diff_in_dalys_by_cause_heatmap', True)

deaths_cause = results['deaths_and_stillbirths']['crude']
baseline = deaths_cause.xs(0, axis=1, level="draw")
change_deaths_cause_df = baseline.sub(
    deaths_cause,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
change_deaths_cause_df = summarize_confidence_intervals(change_deaths_cause_df)

figure_heatmap_cause_specific_dalys_averted(change_deaths_cause_df,
                                            'Changes in key death outcomes',
                                            'diff_in_death_outcomes_cause_heatmap', False)

def get_total_num_death_by_cause(_df):
    """Return the total number of deaths in the TARGET_PERIOD by age-group and cause label."""
    return _df \
        .loc[_df['date'].between(*TARGET_PERIOD)] \
        .groupby(['cause'])['person_id'].size()


total_num_death_by_cause = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=get_total_num_death_by_cause,
    do_scaling=False
)
total_num_death_by_cause.fillna(0, inplace=True)
baseline = total_num_death_by_cause.xs(0, axis=1, level="draw")
change_deaths_cause_demog_df = baseline.sub(
    total_num_death_by_cause,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
change_deaths_cause_demog_df = summarize_confidence_intervals(change_deaths_cause_demog_df)

figure_heatmap_cause_specific_dalys_averted(change_deaths_cause_demog_df,
                                            'Changes in key death outcomes (demog)',
                                            'diff_in_death_outcomes_cause_demog_heatmap', False)

# ======================================= TEST FIGURE - DALYs averted assumptions =====================================
# 1.) Total DALYs averted
dalys_by_cause = extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns=['date', 'sex', 'age_range']).groupby(['year']).sum().stack()),
            do_scaling=False)
dalys_by_cause = dalys_by_cause.loc[dalys_by_cause.index.get_level_values(0) != 2026]
# dalys_by_cause = dalys_by_cause * p_scaling_factor

# all cause DALYs averted
total_dalys = dalys_by_cause.groupby(['year']).sum()
baseline = total_dalys.xs(0, axis=1, level="draw")
total_dalys_averted_df = baseline.sub(
    total_dalys,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
total_dalys_averted_df.rename(index={sim_start_year: "total_dalys_averted"}, inplace=True)

# 2.) Closed cohorts
# For interventions impacting mothers we only look at health impacts in that population and then the same for newborns
dalys_by_cause_and_age = extract_results(
    results_folder,
    module="tlo.methods.healthburden",
    key="dalys_stacked_by_age_and_time",
    custom_generate_series=lambda df: (
        df.drop(columns=["date", "sex"])
          .groupby(["year", "age_range"])
          .sum()
          .stack()
    ),
    do_scaling=False
)

# Select the analysis year directly
dalys_by_cause_and_age = (
    dalys_by_cause_and_age
    .xs(sim_start_year, level="year")
    # .mul(p_scaling_factor)
)

# Map each age group to an outcome category
age_category = (
    dalys_by_cause_and_age.index
    .get_level_values("age_range")
    .to_series(index=dalys_by_cause_and_age.index)
    .map(lambda age: "newborn_dalys" if age == "0-4"
         else "maternal_dalys")
)

# Sum across both ages and causes within each category
total_dalys_by_cause_and_age = (
    dalys_by_cause_and_age
    .groupby(age_category)
    .sum()
)

baseline = total_dalys_by_cause_and_age.xs(0, axis=1, level="draw")
total_dalys_by_cause_and_age_averted = baseline.sub(
    total_dalys_by_cause_and_age,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
total_dalys_by_cause_and_age_averted.rename(index={'newborn_dalys': "newborn_dalys_averted",
                                     'maternal_dalys':'maternal_dalys_averted'}, inplace=True)

maternal_draws = [1, 2, 3, 4, 5, 6]       # replace with your draw numbers
newborn_draws = [7, 8, 9]        # replace with your draw numbers

df = total_dalys_by_cause_and_age_averted

draw_values = df.columns.get_level_values("draw")

# Start with the maternal DALY values
cc_dalys_averted = df.loc[["maternal_dalys_averted"]].copy()

# Replace selected draws with the corresponding newborn DALY values
newborn_columns = draw_values.isin(newborn_draws)

cc_dalys_averted.loc["maternal_dalys_averted", newborn_columns] = (
    df.loc["newborn_dalys_averted", newborn_columns].to_numpy()
)

# Keep only the requested draws, if other draws should be excluded
selected_draws = maternal_draws + newborn_draws
cc_dalys_averted = cc_dalys_averted.loc[
    :,
    draw_values.isin(selected_draws)
]

# Rename the single row
cc_dalys_averted.index = pd.Index(
    ["cc_dalys_averted"],
    name=df.index.name
)

# 3.) DALYs per person
# todo: should this be per birth?
def get_total_population_by_year(_df):
    years_needed = [sim_start_year + 1]  # Malaria scale-up period years
    _df['year'] = pd.to_datetime(_df['date']).dt.year

    # Validate that all necessary years are in the DataFrame
    if not set(years_needed).issubset(_df['year'].unique()):
        raise ValueError("Some years are not recorded in the dataset.")

    # Filter for relevant years and return the total population as a Series
    return \
        _df.loc[_df['year'].between(min(years_needed), max(years_needed)), ['year', 'total']].set_index('year')[
            'total']

total_population_by_year = extract_results(
    results_folder,
    module='tlo.methods.demography',
    key='population',
    custom_generate_series=get_total_population_by_year,
    do_scaling=False
)
total_population_by_year.rename(index={sim_start_year+1: sim_start_year}, inplace=True)
dalys_per_person = (total_population_by_year / total_dalys) * 1000

baseline = dalys_per_person.xs(0, axis=1, level="draw")
person_dalys_averted_df = dalys_per_person.sub(
    baseline,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
person_dalys_averted_df.rename(index={sim_start_year: "person_dalys_averted"}, inplace=True)

# 4.) DALYs accounting for stillbirths

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

# 5.) YLLs attributed to never born newborns
# todo what if there are fewer births in the intervention?
br = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="on_birth",
    custom_generate_series=(
        lambda df: df.assign(
            year=df['date'].dt.year).groupby(['year'])['year'].count()),
    do_scaling=False
)
births_results = br.fillna(0)
# births_results = births_results * p_scaling_factor
baseline = births_results.xs(0, axis=1, level="draw")
birth_diff_df = births_results.sub(
    baseline,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
birth_diff_df.rename(index={sim_start_year: "births"}, inplace=True)


# todo: we only want to add the YLL if theres a change in births

# combine data frames
outcomes_df = pd.concat(
    [total_dalys_averted_df,
          cc_dalys_averted,
          adj_dalys_averted_df,
         person_dalys_averted_df,
        birth_diff_df
     ],
    axis=0
)
outcomes_df_summ = summarize_confidence_intervals(outcomes_df)

def plot_outcomes(data, outcome_labels, save_title):
    plot_df = data

    # Ensure the index has a usable name
    plot_df.index.name = "outcome"
    plot_df.columns.names = ["draw", "stat"]

    long_df = (
        plot_df
        .stack(level="draw")
        .reset_index()
    )

    # Expected columns:
    # outcome | draw | lower | mean | upper

    # Optional readable labels
    scenario_labels = draw_labels

    long_df["scenario"] = long_df["draw"].map(
        lambda x: scenario_labels.get(x, f"Scenario {x}")
    )

    outcomes = plot_df.index.tolist()
    ncols = 2
    nrows = int(np.ceil(len(outcomes) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(13, 2.8 * nrows),
        constrained_layout=True
    )

    axes = np.asarray(axes).flatten()

    for ax, outcome in zip(axes, outcomes):
        data = (
            long_df.loc[long_df["outcome"].eq(outcome)]
            .sort_values("draw")
            .reset_index(drop=True)
        )

        y = np.arange(len(data))

        # Asymmetric confidence intervals
        xerr = np.vstack([
            data["mean"] - data["lower"],
            data["upper"] - data["mean"]
        ])

        ax.errorbar(
            data["mean"],
            y,
            xerr=xerr,
            fmt="o",
            color="#2166AC",
            ecolor="#7F8C8D",
            elinewidth=1.5,
            capsize=3,
            markersize=6
        )

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

        ax.set_yticks(y)
        ax.set_yticklabels(data["scenario"])
        ax.invert_yaxis()

        ax.set_title(
            outcome_labels.get(
                outcome,
                outcome.replace("_", " ").title()
            ),
            loc="left",
            fontweight="bold"
        )

        ax.set_xlabel("Mean (95% CI)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=0.2)

    # Remove any unused panels
    for ax in axes[len(outcomes):]:
        ax.remove()

    fig.suptitle(
        "Estimated outcomes by scenario",
        fontsize=15,
        fontweight="bold"
    )
    plt.savefig(f'{g_path}/{save_title}.png', bbox_inches='tight')
    plt.show()

dalys_outcome_labels = {
    "total_dalys_averted": "All-cause DALYs averted",
    "cc_dalys_averted": "All-cause DALYs averted (closed cohort)",
    "adj_dalys_averted": "All-cause DALYs averted (inc. stillbirth)",
    "person_dalys_averted": "DALYs per 1000 people averted",
    "births": "Additional births compared to baseline"}

plot_outcomes(outcomes_df_summ, dalys_outcome_labels, 'dalys_averted_by_est_method')

# ================================= HEALTH OUTCOME TESTING ======================================
# Maternal DALYs averted
mat_dalys = dalys_by_cause.loc[sim_start_year, 'Maternal Disorders'].reindex(dalys_by_cause.columns).to_frame().T
baseline = mat_dalys.xs(0, axis=1, level="draw")
mat_dalys_averted_df = baseline.sub(
    mat_dalys,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
mat_dalys_averted_df = mat_dalys_averted_df.droplevel(0, axis=0)
mat_dalys_averted_df.rename(index={'Maternal Disorders': "mat_dalys_averted"}, inplace=True)

# Newborn DALYs averted
neo_dalys = dalys_by_cause.loc[sim_start_year, 'Neonatal Disorders'].reindex(dalys_by_cause.columns).to_frame().T
baseline = neo_dalys.xs(0, axis=1, level="draw")
neo_dalys_averted_df = baseline.sub(
    neo_dalys,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
neo_dalys_averted_df = neo_dalys_averted_df.droplevel(0, axis=0)
neo_dalys_averted_df.rename(index={'Neonatal Disorders': "neo_dalys_averted"}, inplace=True)

# Maternal and newborn DALYs averted
mat_neo_dalys_averted = mat_dalys_averted_df.copy()
mat_neo_dalys_averted.iloc[:, :] = mat_dalys_averted_df.to_numpy() + neo_dalys_averted_df.to_numpy()
mat_neo_dalys_averted.rename(index={'mat_dalys_averted': "mat_neo_dalys_averted"}, inplace=True)

# Maternal deaths averted
def get_num_deaths_by_cause_label(_df):
    """Return total number of Deaths by label (total by age-group within the TARGET_PERIOD)
    """
    return _df \
        .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
        .groupby(_df['label']) \
        .size()

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

def extract_indirect_deaths_non_hiv(df):
    year = pd.to_datetime(df["date"]).dt.year

    pregnant_or_postpartum = (
        df["is_pregnant"].fillna(False).astype(bool)
        | df["la_is_postpartum"].fillna(False).astype(bool)
    )

    relevant_cause = (
        df["cause_of_death"].str.contains(
            r"Malaria|Suicide|ever_stroke|diabetes|"
            r"chronic_ischemic_hd|ever_heart_attack|"
            r"chronic_kidney_disease",
            na=False,
            regex=True,
        )
        | df["cause_of_death"].eq("TB")
    )

    return (
        year[pregnant_or_postpartum & relevant_cause]
        .value_counts()
        .sort_index()
        .rename_axis("year")
        .rename("deaths")
    )


indirect_deaths_non_hiv = extract_results(
    results_folder,
    module="tlo.methods.demography.detail",
    key="properties_of_deceased_persons",
    custom_generate_series=extract_indirect_deaths_non_hiv,
    do_scaling=False,
)
indirect_deaths_non_hiv_final = indirect_deaths_non_hiv.fillna(0)
# indirect_deaths_non_hiv_final = indirect_deaths_non_hiv_final * p_scaling_factor

# Deaths due to AIDS during/following pregnancy are adjusted in line with UN MMEIG methodology
hiv_pd = extract_results(
    results_folder,
    module="tlo.methods.demography.detail",
    key="properties_of_deceased_persons",
    custom_generate_series=lambda df: (
        df.assign(year=pd.to_datetime(df["date"]).dt.year)
        .loc[
            (
                df["is_pregnant"].fillna(False).astype(bool)
                | df["la_is_postpartum"].fillna(False).astype(bool)
            )
            & df["cause_of_death"].str.contains(
                r"^(?:AIDS_non_TB|AIDS_TB)$",
                na=False,
                regex=True,
            )
        ]
        .groupby("year")
        .size()
        .rename("deaths")
    ),
    do_scaling=False,
)
# TODO not sure about this logic...
hiv_pd = hiv_pd.fillna(0)
# hiv_pd = hiv_pd * p_scaling_factor

hiv_indirect_maternal_deaths = hiv_pd * 0.3
hiv_indirect_maternal_deaths = hiv_indirect_maternal_deaths.round(0)

# The MMR is calculated from total deaths extracted above using live births as a denominator
indirect_deaths_final = indirect_deaths_non_hiv_final + hiv_indirect_maternal_deaths
total_mat_deaths = direct_deaths + indirect_deaths_final

baseline = total_mat_deaths.xs(0, axis=1, level="draw")
all_mat_deaths_averted_df = baseline.sub(
    total_mat_deaths,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
all_mat_deaths_averted_df.rename(index={sim_start_year: "all_mat_deaths_averted"}, inplace=True)

# Newborn deaths averted
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

nd = extract_results(
    results_folder,
    module="tlo.methods.demography.detail",
    key="properties_of_deceased_persons",
    custom_generate_series=(
        lambda df: df.loc[(df['age_days'] < 29)].assign(
            year=df['date'].dt.year).groupby(['year'])['year'].count()),
    do_scaling=False)
neo_deaths = nd.fillna(0)
# neo_deaths = neo_deaths * p_scaling_factor

baseline = neo_deaths.xs(0, axis=1, level="draw")
all_neo_deaths_averted_df = baseline.sub(
    neo_deaths,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
all_neo_deaths_averted_df.rename(index={sim_start_year: "all_neo_deaths_averted"}, inplace=True)

# Stillbirths averted
stillbirths = results['deaths_and_stillbirths']['crude'].loc['total_stillbirths'].reindex(dalys_by_cause.columns).to_frame().T
# stillbirths = stillbirths * p_scaling_factor
baseline = stillbirths.xs(0, axis=1, level="draw")
stillbirths_averted_df = baseline.sub(
    stillbirths,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
stillbirths_averted_df.rename(index={'total_stillbirths': "stillbirths_averted"}, inplace=True)

# YLDs
yld_by_cause = extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="yld_by_causes_of_disability",
            custom_generate_series=(
                lambda df: df.drop(
                    columns=['date', 'sex']).groupby(['year', 'age_range']).sum().stack()),
            do_scaling=False)
# yld_by_cause = yld_by_cause * p_scaling_factor

ylds_2025 = yld_by_cause.xs(sim_start_year, level=0)

# Reclassify ages and sum across all causes
yld_type = np.where(
    ylds_2025.index.get_level_values('age_range') == '0-4',
    'newborn_ylds',
    'maternal_ylds'
)

total_ylds_2025 = ylds_2025.groupby(yld_type).sum()

# Set index name and order
total_ylds_2025.index.name = 'yld_type'
total_ylds_2025 = total_ylds_2025.reindex([
    'newborn_ylds',
    'maternal_ylds'
])

baseline = total_ylds_2025.xs(0, axis=1, level="draw")
ylds_averted = baseline.sub(
    total_ylds_2025,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
ylds_averted.rename(index={'newborn_ylds': "newborn_ylds_averted",
                                     'maternal_ylds':'maternal_ylds_averted'}, inplace=True)


# combine data frames
health_df = pd.concat(
    [mat_dalys_averted_df,
          neo_dalys_averted_df,
          mat_neo_dalys_averted,
          direct_mat_deaths_averted_df,
          all_mat_deaths_averted_df,
          direct_neo_deaths_averted_df,
          all_neo_deaths_averted_df,
          stillbirths_averted_df,
          ylds_averted,
     ],
    axis=0
)
health_df_summ = summarize_confidence_intervals(health_df)

health_outcome_labels = {
        "mat_dalys_averted": "Maternal Disorders DALYs averted",
        "neo_dalys_averted": "Neonatal Disorders DALYs averted",
        "direct_mat_deaths_averted": "Direct Maternal deaths averted",
        "all_mat_deaths_averted": "Total Maternal deaths averted",
        "direct_neo_deaths_averted": "Direct Neonatal deaths averted",
        "all_neo_deaths_averted": "Total Neonatal deaths averted",
        "stillbirths_averted": "Stillbirths averted",
        "newborn_ylds_averted": "Newborn YLDs averted",
        "maternal_ylds_averted": "Maternal YLDs averted",
    }

plot_outcomes(health_df_summ, health_outcome_labels, 'other_health_outcomes')


def plot_outcomes():
    plot_df = health_df_summ.copy()

    # Ensure the index has a usable name
    plot_df.index.name = "outcome"
    plot_df.columns.names = ["draw", "stat"]

    long_df = (
        plot_df
        .stack(level="draw")
        .reset_index()
    )

    # Expected columns:
    # outcome | draw | lower | mean | upper

    # Optional readable labels
    scenario_labels = draw_labels

    outcome_labels = {
        "total_dalys_averted": "All-cause DALYs averted",
        "mat_dalys_averted": "Maternal Disorders DALYs averted",
        "neo_dalys_averted": "Neonatal Disorders DALYs averted",
        "direct_mat_deaths_averted": "Direct Maternal deaths averted",
        "all_mat_deaths_averted": "Total Maternal deaths averted",
        "direct_neo_deaths_averted": "Direct Neonatal deaths averted",
        "all_neo_deaths_averted": "Total Neonatal deaths averted",
        "stillbirths_averted": "Stillbirths averted",
        "newborn_ylds_averted": "Newborn YLDs averted",
        "maternal_ylds_averted": "Maternal YLDs averted",
        "adj_dalys_averted_df": "Adj. all-cause DALYs averted",
        "person_dalys_averted": "DALYs per person Averted"
    }

    long_df["scenario"] = long_df["draw"].map(
        lambda x: scenario_labels.get(x, f"Scenario {x}")
    )

    outcomes = plot_df.index.tolist()
    ncols = 2
    nrows = int(np.ceil(len(outcomes) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(13, 2.8 * nrows),
        constrained_layout=True
    )

    axes = np.asarray(axes).flatten()

    for ax, outcome in zip(axes, outcomes):
        data = (
            long_df.loc[long_df["outcome"].eq(outcome)]
            .sort_values("draw")
            .reset_index(drop=True)
        )

        y = np.arange(len(data))

        # Asymmetric confidence intervals
        xerr = np.vstack([
            data["mean"] - data["lower"],
            data["upper"] - data["mean"]
        ])

        ax.errorbar(
            data["mean"],
            y,
            xerr=xerr,
            fmt="o",
            color="#2166AC",
            ecolor="#7F8C8D",
            elinewidth=1.5,
            capsize=3,
            markersize=6
        )

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

        ax.set_yticks(y)
        ax.set_yticklabels(data["scenario"])
        ax.invert_yaxis()

        ax.set_title(
            outcome_labels.get(
                outcome,
                outcome.replace("_", " ").title()
            ),
            loc="left",
            fontweight="bold"
        )

        ax.set_xlabel("Mean (95% CI)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=0.2)

    # Remove any unused panels
    for ax in axes[len(outcomes):]:
        ax.remove()

    fig.suptitle(
        "Estimated outcomes by scenario",
        fontsize=15,
        fontweight="bold"
    )
    plt.savefig(f'{g_path}/health_outcome_summary.png', bbox_inches='tight')
    plt.show()

plot_outcomes()

# categories_to_keep = {"Maternal Disorders", "Neonatal Disorders"}
# cause = dalys_by_cause.index.get_level_values(1)
#
# new_category = cause.where(cause.isin(categories_to_keep), "Other")
#
# dalys_main_causes = (
#     dalys_by_cause.assign(_category=new_category)
#       .groupby(
#           [dalys_by_cause.index.get_level_values(0), "_category"],
#           sort=False
#       )
#       .sum()
# )
# dalys_main_causes_scaled = dalys_main_causes * p_scaling_factor
# dalys_main_causes_scaled.index.names = ["year", "cause"]
#
# baseline = dalys_main_causes_scaled.xs(0, axis=1, level="draw")
# dalys_averted_df = baseline.sub(
#     dalys_main_causes_scaled,
#     axis="columns",
#     level="run"
# ).drop(columns=0, level="draw")
#
# def produce_fig_2():
#     year = sim_start_year
#
#     category_order = [
#         "Maternal Disorders",
#         "Neonatal Disorders",
#         "Other"
#     ]
#
#     # Select the year and calculate the mean across runs for each draw
#     plot_data = (
#         dalys_averted_df
#         .xs(year, level="year")
#         .T
#         .groupby(level="draw")
#         .mean()
#         .T
#         .reindex(category_order)
#         .T
#     )
#
#     # Optional: replace draw numbers with intervention labels
#     # draw_labels should map draw numbers to names, e.g. {1: "Sepsis", 2: "PPH"}
#     if "draw_labels" in globals():
#         plot_data.index = [
#             draw_labels.get(draw, draw)
#             for draw in plot_data.index
#         ]
#
#     colours = {
#         "Maternal Disorders": "#D55E00",
#         "Neonatal Disorders": "#0072B2",
#         "Other": "#999999"
#     }
#
#     fig, ax = plt.subplots(figsize=(12, 6))
#
#     plot_data.plot(
#         kind="bar",
#         stacked=True,
#         color=[colours[c] for c in plot_data.columns],
#         width=0.75,
#         ax=ax
#     )
#
#     ax.axhline(0, color="black", linewidth=0.8)
#
#     ax.set_xlabel("")
#     ax.set_ylabel("Difference from baseline")
#     ax.set_title(f"Difference from baseline by cause, {year}")
#
#     ax.tick_params(axis="x", rotation=45)
#     ax.legend(
#         title="Cause",
#         frameon=False,
#         bbox_to_anchor=(1.02, 1),
#         loc="upper left"
#     )
#
#     fig.tight_layout()
#     plt.savefig(f'{g_path}/figure2_diff_in_dalys_by_cause.png', bbox_inches='tight')
#     plt.show()
#
# produce_fig_2()

# ============================================== FIG 3 - ICERs ========================================================
# TODO - DISCOUNT DALYS
# TODO - SORT OUT COST SCALING (CONS NUMBERS ARE SCALED TO THE POPULATION SIZE WHEN EXTRACTING)

# # 1. Get total dalys averted
# TODO i dont know if i should be using all DALYs or just MNH DALYs
total_dalys = dalys_by_cause.groupby(['year']).sum()
baseline = total_dalys.xs(0, axis=1, level="draw")
total_dalys_averted_df = baseline.sub(
    total_dalys,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
total_dalys_averted_df_summarised = summarize_confidence_intervals(total_dalys_averted_df)

# # 2. Get total costs
# cost_by_draw_and_year = input_costs.groupby(['draw', 'run', 'year'])['cost'].sum()
# cost_by_draw_and_year_df = cost_by_draw_and_year.reset_index().pivot(index='year', columns=['draw','run'], values='cost')
# baseline = cost_by_draw_and_year_df.xs(0, axis=1, level="draw")
# incremental_cost =  cost_by_draw_and_year_df.sub(
#     baseline,
#     axis="columns",
#     level="run"
# ).drop(columns=0, level="draw")
# incremental_cost_summarised = summarize_confidence_intervals(incremental_cost)
#
# icers = incremental_cost / total_dalys_averted_df
# icers_summarized = summarize_confidence_intervals(icers)
#


# # ============================================ DEBUGGING PLOTS ========================================================
# cost_by_category = input_costs.groupby(['draw', 'run', 'year', 'cost_category'])['cost'].sum()
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
# cost_by_draw_and_year = input_costs.groupby(['draw', 'run', 'year'])['cost'].sum()
# cost_by_draw_and_year_df = cost_by_draw_and_year.reset_index().pivot(index='year', columns=['draw','run'], values='cost')
# #  todo: above service costs


# Summarised results

def debug_plots():

    def get_data(df, key, draw):
        return (df.loc[key, (draw, 'lower')],
                df.loc[key, (draw, 'mean')],
                df.loc[key, (draw, 'upper')])

    mat_dalys_by_scenario = {k: get_data(results['mat_dalys']['summarised'], 'Maternal Disorders', d) for k, d in zip (
        int_analysis, draws)}
    neo_dalys_by_scenario = {k: get_data(results['neo_dalys']['summarised'], 'Neonatal Disorders', d) for k, d in zip (
        int_analysis, draws)}

    mmr_by_scnario = {k: get_data(results['deaths_and_stillbirths']['summarised'], 'direct_mmr', d) for k, d in zip (
        int_analysis, draws)}
    nmr_by_scnario = {k: get_data(results['deaths_and_stillbirths']['summarised'], 'nmr', d) for k, d in zip (
        int_analysis, draws)}

    mmr_by_scenario_oth_log = {k: get_data(results['mat_deaths']['summarised'], sim_start_year, d) for k, d in zip (
        int_analysis, draws)}
    nmr_by_scenario_oth_log = {k: get_data(results['neo_deaths']['summarised'], sim_start_year, d) for k, d in zip (
        int_analysis, draws)}

    def barcharts(data, y_label, title):

        # Extract means and errors
        labels = data.keys()
        means = [vals[1] for vals in data.values()]
        # lower_errors = [vals[0] for vals in data.values()]
        # upper_errors = [vals[2] for vals in data.values()]

        lower_errors = [vals[1] - vals[0] for vals in data.values()]
        upper_errors = [vals[2] - vals[1] for vals in data.values()]
        errors = [lower_errors, upper_errors]

        # Create bar chart with error bars
        fig, ax = plt.subplots()
        ax.bar(labels, means, yerr=errors, capsize=5, alpha=0.7, ecolor='black')
        ax.set_ylabel(y_label)
        ax.set_title(title)

        # Adjust label size
        plt.xticks(fontsize=8, rotation=90)
        plt.tight_layout()
        plt.savefig(f'{g_path}/{title}.png', bbox_inches='tight')
        plt.show()

    barcharts(mat_dalys_by_scenario, 'DALYs', 'Total Maternal Disorders DALYs by scenario')
    barcharts(neo_dalys_by_scenario, 'DALYs', 'Total Neonatal Disorders DALYs by scenario')

    barcharts(mmr_by_scnario, 'MMR', 'Total MMR by scenario')
    barcharts(nmr_by_scnario, 'MMR', 'Total NMR by scenario')

    # Difference results
    def get_diffs(df_key, result_key, ints, draws):
        diff_results = {}
        baseline = results[df_key]['crude'][0]

        for draw, int in zip(draws, ints):
            diff_df = results[df_key]['crude'][draw] - baseline
            diff_df.columns = pd.MultiIndex.from_tuples([(draw, v) for v in range(len(diff_df.columns))],
                                                        names=['draw', 'run'])
            results_diff = summarize_confidence_intervals(diff_df)
            results_diff.fillna(0)
            diff_results.update({int: results_diff.loc[result_key].values})

        return [diff_results, diff_df]

    def get_diffs_demog_log(mr_df):
        diff_results = {}
        baseline = mr_df['crude'][0]

        for draw, int in zip(draws, int_analysis):
            diff_df = mr_df['crude'][draw] - baseline
            diff_df.columns = pd.MultiIndex.from_tuples([(draw, v) for v in range(len(diff_df.columns))],
                                                            names=['draw', 'run'])
            results_diff = summarize_confidence_intervals(diff_df)
            results_diff.fillna(0)
            diff_results.update({int: results_diff.loc[sim_start_year].values})

        return diff_results


    mat_deaths = get_diffs('deaths_and_stillbirths', 'direct_maternal_deaths', int_analysis, draws)[0]
    neo_deaths = get_diffs('deaths_and_stillbirths', 'neonatal_deaths', int_analysis, draws)[0]

    mmr_diffs = get_diffs('deaths_and_stillbirths', 'direct_mmr', int_analysis, draws)[0]
    nmr_diffs = get_diffs('deaths_and_stillbirths', 'nmr', int_analysis, draws)[0]

    mat_dalys_diffs = get_diffs('mat_dalys', 'Maternal Disorders', int_analysis, draws)[0]
    neo_dalys_diffs = get_diffs('neo_dalys', 'Neonatal Disorders', int_analysis, draws)[0]

    mat_deaths_2 = get_diffs_demog_log(results['mmr'])
    neo_deaths_2 = get_diffs_demog_log(results['nmr'])

    def get_diff_plots(data, outcome):
        categories = list(data.keys())
        mins = [arr[0] for arr in data.values()]
        means = [arr[1] for arr in data.values()]
        maxs = [arr[2] for arr in data.values()]

        # Error bars (top and bottom of the uncertainty interval)
        errors = [(mean - min_val, max_val - mean) for mean, min_val, max_val in zip(means, mins, maxs)]
        errors = np.array(errors).T

        # todo: the error bars are slightly off...

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.errorbar(categories, means, yerr=errors, fmt='o', capsize=5)
        plt.axhline(0, color='gray', linestyle='--')  # Adding a horizontal line at y=0 for reference
        plt.xticks(rotation=90)
        plt.xlabel('Scenarios')
        plt.ylabel('Crude Difference from Baseline Scenario')
        plt.title(f'Difference of {outcome} from Baseline Scenario')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{g_path}/{outcome}.png', bbox_inches='tight')

        plt.show()

    get_diff_plots(mmr_diffs, 'MMR')
    get_diff_plots(nmr_diffs, 'NMR')

    get_diff_plots(mat_deaths, 'Maternal Deaths (crude)')
    get_diff_plots(neo_deaths, 'Neonatal Deaths (crude)')

    get_diff_plots(mat_deaths_2, 'MMR (demog log)')
    get_diff_plots(neo_deaths_2, 'NMR (demog log)')

    get_diff_plots(mat_dalys_diffs, 'Maternal DALYs')
    get_diff_plots(neo_dalys_diffs, 'neonatal DALYs')
    #
    # cost_diff = {}
    # baseline = cost_by_draw_and_year_df[0]
    #
    # for draw, intervention in zip(draws, int_analysis):
    #     diff_df = cost_by_draw_and_year_df[draw] - baseline
    #     diff_df.columns = pd.MultiIndex.from_tuples([(draw, v) for v in range(len(diff_df.columns))],
    #                                                 names=['draw', 'run'])
    #     results_diff = summarize_confidence_intervals(diff_df)
    #     cost_diff.update({intervention: results_diff.values.flatten().tolist()})
    #
    # get_diff_plots(cost_diff, 'Cost')
    #
    # medical_consumables_df = (
    #     input_costs[input_costs["cost_category"].eq("medical consumables")]
    #       .pivot_table(
    #           index="year",
    #           columns=["draw", "run"],
    #           values="cost",
    #           aggfunc="sum"
    #       )
    #       .sort_index(axis=1))
    #
    # # TODO make more robust so we dont have to define year
    # medical_consumables_summ = summarize_confidence_intervals(medical_consumables_df)
    # medical_cons_by_scenario = {k: [medical_consumables_summ.loc[sim_start_year, (d, 'lower')],
    #                                 medical_consumables_summ.loc[sim_start_year, (d, 'mean')],
    #                                 medical_consumables_summ.loc[sim_start_year, (d, 'upper')]] for k, d in zip (int_analysis, draws)}
    #
    # from collections import defaultdict
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
    #     do_scaling=True)  # todo change to False
    #
    # cons_req.fillna(0, inplace=True)
    #
    # cons_costs = pd.read_csv(Path("./resources/costing") / 'ResourceFile_Costing_Consumables.csv')
    #
    # # costed_scenario_cons = cons_req.copy()
    # price_lookup = (
    #     cons_costs
    #     .assign(Item_Code=cons_costs["Item_Code"].astype(str).str.strip())
    #     .set_index("Item_Code")["Price_per_unit"])
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


debug_plots()

# COST




# ======================================== FIGURE 1 - MET NEED =======================================================

#  TODO

# ============================ TABLE 2 - DEATHS, DALYS, TOTAL COSTS BY SCENARIO =======================================
# Get maternal/newborn death data

# COST CALCULATIONS

# TEST COSTS DIFFER FROM BASELINE





# TODO: THIS CODE WILL EXTRACT DIFFERENCE BETWEEN BASELINE AND USED CONS IN CONS UNIT

# def get_tornado_plot(data, outcome):
#     grouped_data = {}
#     data.pop('baseline', None)
#
#     for key in data.keys():
#         base_key = key.rsplit('_', 1)[0] if key.endswith('_max') or key.endswith('_min') else key
#
#         if base_key not in grouped_data:
#             grouped_data[base_key] = {'min': None, 'max': None}
#         if 'min' in key:
#             grouped_data[base_key]['min'] = data[key]
#         elif 'max' in key:
#             grouped_data[base_key]['max'] = data[key]
#
#     # Prepare data for plotting
#     categories = list(grouped_data.keys())
#     min_values = [np.mean(grouped_data[cat]['min']) for cat in categories]
#     max_values = [np.mean(grouped_data[cat]['max']) for cat in categories]
#
#     # Extracting uncertainty intervals (first and third values in each array)
#     min_lower = [grouped_data[cat]['min'][0] for cat in categories]
#     min_upper = [grouped_data[cat]['min'][2] for cat in categories]
#     max_lower = [grouped_data[cat]['max'][0] for cat in categories]
#     max_upper = [grouped_data[cat]['max'][2] for cat in categories]
#
#     # Calculate error bars (distance from mean to bounds)
#     min_errors = [np.abs(np.array(min_values) - np.array(min_lower)),
#                   np.abs(np.array(min_upper) - np.array(min_values))]
#     max_errors = [np.abs(np.array(max_values) - np.array(max_lower)),
#                   np.abs(np.array(max_upper) - np.array(max_values))]
#
#     # Plotting
#     plt.figure(figsize=(10, 6))
#     y_positions = np.arange(len(categories))
#
#     bars_min = plt.barh(y_positions, min_values, color='lightcoral', edgecolor='black', alpha=0.7, label='Min Effect')
#     bars_max = plt.barh(y_positions, max_values, color='skyblue', edgecolor='black', alpha=0.7, label='Max Effect')
#
#     # Add error bars for uncertainty intervals
#     plt.errorbar(min_values, y_positions, xerr=min_errors, fmt='none', ecolor='darkred', capsize=5, alpha=0.9,
#                  label='Uncertainty (Min)')
#     plt.errorbar(max_values, y_positions, xerr=max_errors, fmt='none', ecolor='navy', capsize=5, alpha=0.9,
#                  label='Uncertainty (Max)')
#
#     # Central zero line
#     plt.axvline(0, color='black', linewidth=1, linestyle='--')
#
#     # Add labels
#     plt.yticks(y_positions, categories)
#     plt.xlabel(f'Difference in {outcome} from Status Quo')
#     plt.title(f'Tornado Plot showing current and potential impact of interventions on {outcome}')
#     plt.legend()
#
#     plt.savefig(f'{g_path}/{outcome}_tornado.png', bbox_inches='tight')
#     plt.show()
#
#
# get_tornado_plot(mat_deaths_2, 'MMR')
# get_tornado_plot(neo_deaths_2, 'NMR')
#
# get_tornado_plot(mat_dalys_diffs, 'Maternal DALYs')
# get_tornado_plot(neo_dalys_diffs, 'Neonatal DALYs')

# Table 1
# def get_table_one():
#     columns = ['age_years', 'la_parity', 'region_of_residence', 'li_wealth', 'li_bmi', 'li_mar_stat', 'li_ed_lev',
#                 'li_urban', 'ps_prev_spont_abortion', 'ps_prev_stillbirth', 'ps_prev_pre_eclamp', 'ps_prev_gest_diab']
#     categorical = ['region_of_residence', 'li_wealth', 'li_bmi' ,'li_mar_stat', 'li_ed_lev', 'li_urban',
#                     'ps_prev_spont_abortion', 'ps_prev_stillbirth', 'ps_prev_pre_eclamp', 'ps_prev_gest_diab']
#     continuous = ['age_years', 'la_parity']
#
#     rename = {'age_years': 'Age (years)',
#                'la_parity': 'Parity',
#                'region_of_residence': 'Region',
#                'li_wealth': 'Wealth Quintile',
#                'li_bmi': 'BMI level',
#                'li_mar_stat': 'Marital Status',
#                'li_ed_lev': 'Education Level',
#                'li_urban': 'Urban/Rural',
#                'ps_prev_spont_abortion': 'Previous Miscarriage',
#                'ps_prev_stillbirth': 'Previous Stillbirth',
#                'ps_prev_pre_eclamp': 'Previous Pre-eclampsia',
#                'ps_prev_gest_diab': 'Previous Gestational Diabetes',
#               }
#
#     all_preg_df = pd.read_excel(Path("./resources/ResourceFile_MaternalCohort") /
#                                         'ResourceFile_All2025PregnanciesCohortModel.xlsx')
#     population = 40_000
#
#     # Only select rows equal to the desired population size
#     if population <= len(all_preg_df):
#         preg_pop = all_preg_df.loc[0:population-1]
#     else:
#         # Calculate the number of rows needed to reach the desired length
#         additional_rows = population - len(all_preg_df)
#
#         # Initialize an empty DataFrame for additional rows
#         rows_to_add = pd.DataFrame(columns=all_preg_df.columns)
#
#         # Loop to fill the required additional rows
#         while additional_rows > 0:
#             if additional_rows >= len(all_preg_df):
#                 rows_to_add = pd.concat([rows_to_add, all_preg_df], ignore_index=True)
#                 additional_rows -= len(all_preg_df)
#             else:
#                 rows_to_add = pd.concat([rows_to_add, all_preg_df.iloc[:additional_rows]], ignore_index=True)
#                 additional_rows = 0
#
#         # Concatenate the original DataFrame with the additional rows
#         preg_pop = pd.concat([all_preg_df, rows_to_add], ignore_index=True)
#
#     mytable = TableOne(preg_pop[columns], categorical=categorical,
#                        continuous=continuous, rename=rename, pval=False)
#     print(mytable.tabulate(tablefmt = "fancy_grid"))
#     mytable.to_excel(Path(f"{outputspath}/{scenario}/0/table_one.xlsx") )
