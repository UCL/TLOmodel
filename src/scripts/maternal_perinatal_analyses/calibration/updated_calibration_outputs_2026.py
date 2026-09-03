from pathlib import Path

import os
import pandas as pd
import numpy as np

from tlo.analysis.utils import get_scenario_outputs, extract_results, create_pickles_locally
from scipy.stats import t
import matplotlib.pyplot as plt

outputspath = './outputs/sejjj49@ucl.ac.uk/'
resourcefilepath = Path("./resources")


#  ======================================= DEFINE SCENARIO INFORMATION  ===============================================
scenario = 'calibration_172156'
results_folder= get_scenario_outputs(scenario, outputspath)[-1]
# create_pickles_locally(results_folder)

g_path = f'{outputspath}calibration_{scenario}'

if not os.path.isdir(g_path):
        os.makedirs(f'{outputspath}calibration_{scenario}')

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

#  ========================================== EXTRACT CORE DATA  =====================================================
results = {k:get_ps_data_frames(k, results_folder) for k in
           ['mat_comp_incidence', 'nb_comp_incidence', 'deaths_and_stillbirths','service_coverage', 'met_need',
            'yearly_mnh_counter_dict', 'intervention_coverage']}

calibration_targets = {

    "PROM": {
        "value": 27.0,
        "lower": 19.0,
        "upper": 34.0,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "PROM",
    },

    "an_anaemia": {
        "value": 45.0,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Antenatal anaemia",
    },

    "antepartum_haem": {
        "value": 4.6,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Antepartum haemorrhage",
    },

    "eclampsia": {
        "value": 10.0,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Eclampsia",
    },

    "ectopic_unruptured": {
        "value": 10.0,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Ectopic pregnancy",
    },

    "fistula": {
        "value": 6,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Obstetric Fistula",
    },
    "gest_diab": {
        "value": 16.0,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Gestational diabetes",
    },

    "induced_abortion": {
        "value": 159.0,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Induced abortion",
    },

    "mild_pre_eclamp": {
        "value": 44,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Pre-eclampsia (mild)",
    },

    "mild_gest_htn": {
        "value": 43.8,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Gestational hypertension (mild)",
    },

    "obstructed_labour": {
        "value": 33.7,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Obstructed labour",
    },

    "placenta_praevia": {
        "value": 5.67,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Placenta praevia",
    },

    "pn_anaemia": {
        "value": 45,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Postnatal anaemia",
    },

    "postpartum_haem": {
        "value": 12.8,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Postpartum haemorrhage",
    },

    "preterm_birth": {
        "value": 10.0,
        "lower": 7.4,
        "upper": 14.3,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Preterm birth",
    },

    "sepsis": {
        "value": 1.5,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Maternal sepsis",
    },

    "severe_gest_htn": {
        "value": 5.98,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Gestational hypertension (severe)",
    },

    "severe_pre_eclamp": {
        "value": 22,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Pre-eclampsia (severe)",
    },

    "spontaneous_abortion": {
        "value": 153.0,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Spontaneous abortion",
    },

    "uterine_rupture": {
        "value": 0.8,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Uterine rupture",
    },

    "nb_cba": {
        "value": 20.4,
        "lower": 17,
        "upper": 23.8,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Congential birth anomalies",
    },

    "nb_enceph": {
        "value":18.59,
        "lower": 14.3,
        "upper": 24.9,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Encephalopathy",
    },

    "nb_lbw": {
        "value": 12,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Low birth weight",
    },

    "nb_macrosomia": {
        "value": 5.13,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Macrosomia",
    },

    "nb_rds": {
        "value": 180,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Preterm RDS",
    },

    "nb_resp_diff": {
        "value": 5.7,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Respiratory depression",
    },

    "nb_sepsis": {
        "value": 39.3,
        "lower": 19.4,
        "upper": 78.1,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Neonatal sepsis",
    },

    "nb_sga": {
        "value": 23.2,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Small for GA",
    },

    "twin_birth": {
        "value": 3.9,
        "measure": "Incidence (Rate per 100/0 p/b)",
        "label": "Antepartum haemorrhage",
    },

    "antenatal_sbr": {
        "value": 8,
        "measure": "Incidence (mortality)",
        "label": "Antenatal stillbirth rate",
    },

    "intrapartum_sbr": {
        "value": 8,
        "measure": "Incidence (mortality)",
        "label": "Intrapartum stillbirth rate",
    },

    "sbr": {
        "value": 16,
        "lower": 13,
        "upper": 19,
        "measure": "Incidence (mortality)",
        "label": "Stillbirth rate",
    },

    "nmr": {
        "value": 19,
        "lower": 12,
        "upper": 31,
        "measure": "Incidence (mortality)",
        "label": "Neonatal mortality rate",
    },

    "direct_mmr": {
        "value": 381 *0.7,
        "lower": 269 * 0.7,
        "upper": 543 * 0.7,
        "measure": "Incidence (mortality)",
        "label": "Maternal mortality rate",
    },

    #  TODO: update logging in PS so we outputs match calib targets

    "ectopic_pregnancy_m_death": {
        "value": 3,
        "measure": "Incidence (mortality)",
        "label": "EP MMR",
    },

    "induced_abortion_m_death": {
        "value": 20.9,
        "measure": "Incidence (mortality)",
        "label": "IA MMR",
    },

    "spontaneous_abortion_m_death": {
            "value": 20.9,
            "measure": "Incidence (mortality)",
            "label": "SA MMR",
        },

    "severe_pre_eclampsia_m_death": {
        "value": 55.3,
        "measure": "Incidence (mortality)",
        "label": "SPE MMR",
    },

    "eclampsia_m_death": {
        "value": 55.3,
        "measure": "Incidence (mortality)",
        "label": "Ec MMR",
    },

    "antenatal_sepsis_m_death": {
        "value": 67.6,
        "measure": "Incidence (mortality)",
        "label": "AN SEP MMR",
    },

    "intrapartum_sepsis_m_death": {
        "value": 67.6,
        "measure": "Incidence (mortality)",
        "label": "IP SEP MMR",
    },

    "postpartum_sepsis_m_death": {
        "value": 67.6,
        "measure": "Incidence (mortality)",
        "label": "PN SEP MMR",
    },

    "uterine_rupture_m_death": {
        "value": 43,
        "measure": "Incidence (mortality)",
        "label": "UR MMR",
    },

    "postpartum_heamorrhage_m_death": {
        "value": 95.4,
        "measure": "Incidence (mortality)",
        "label": "PPH MMR",
    },

    "secondary_postpartum_haemorrhage_m_death": {
        "value": 95.4,
        "measure": "Incidence (mortality)",
        "label": "SPPH MMR",
    },

    "antepartum_haemorrhage_m_death": {
        "value": 16.9,
        "measure": "Incidence (mortality)",
        "label": "APH MMR",
    },

    "respiratory_distress_syndrome_n_death": {
        "value": 5.94 ,
        "measure": "Incidence (mortality)",
        "label": "RDS NMR",
    },

    "preterm_other_n_death": {
        "value": 5.94 ,
        "measure": "Incidence (mortality)",
        "label": "PTB oth NMR",
    },

    "encephalopathy_n_death": {
        "value": 5.5,
        "measure": "Incidence (mortality)",
        "label": "ENC NMR",
    },

    "neonatal_respiratory_depression_n_death": {
        "value": 5.5,
        "measure": "Incidence (mortality)",
        "label": "NRD NMR",
    },

    "early_onset_sepsis_n_death": {
        "value": 1.76,
        "measure": "Incidence (mortality)",
        "label": "ESEP NMR",
    },

    "late_onset_sepsis_n_death": {
        "value": 1.76,
        "measure": "Incidence (mortality)",
        "label": "LSEP NMR",
    },

}


model_dfs = [
    results["mat_comp_incidence"]["summarised"],
    results["nb_comp_incidence"]["summarised"],
    results["deaths_and_stillbirths"]["summarised"],
]

def plot_calibration(
    model_dfs,
    calibration_targets,
    draw=0,
    figsize_per_panel=(7, 6),
):
    """
    Plot model estimates against empirical calibration targets.

    Parameters
    ----------
    model_dfs : list of pd.DataFrame
        Summary DataFrames with:
            index = outcome name
            columns = MultiIndex (draw, stat)

        The 'stat' level should contain:
            mean, lower, upper

    calibration_targets : dict
        Dictionary keyed by model DataFrame index.

        Targets may optionally contain uncertainty bounds.

        Example
        -------
        {
            "PROM": {
                "value": 18.0,
                "lower": 15.5,
                "upper": 20.5,
                "measure": "Incidence (%)",
                "label": "PROM"
            },

            "an_anaemia": {
                "value": 41.0,
                "measure": "Incidence (%)",
                "label": "Antenatal anaemia"
            }
        }

    draw : int
        Model draw to plot.

    figsize_per_panel : tuple
        Approximate size of each panel.

    Returns
    -------
    fig
    axes
    plot_df
    """

    rows = []

    # ---------------------------------------------------------
    # Extract model estimates and match to calibration targets
    # ---------------------------------------------------------
    for df in model_dfs:

        draw_df = df.xs(
            draw,
            axis=1,
            level="draw"
        )

        for outcome, values in draw_df.iterrows():

            if outcome not in calibration_targets:
                continue

            target = calibration_targets[outcome]

            rows.append({
                "outcome": outcome,
                "label": target.get("label", outcome),
                "measure": target["measure"],

                "model_mean": values["mean"],
                "model_lower": values["lower"],
                "model_upper": values["upper"],

                "data_value": target["value"],

                # np.nan if empirical uncertainty is unavailable
                "data_lower": target.get("lower", np.nan),
                "data_upper": target.get("upper", np.nan),
            })

    plot_df = pd.DataFrame(rows)

    if plot_df.empty:
        raise ValueError(
            "No matching indices found between model dataframes "
            "and calibration_targets."
        )

    # ---------------------------------------------------------
    # Set up panels
    # ---------------------------------------------------------
    measures = plot_df["measure"].unique()

    n_panels = len(measures)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(
            figsize_per_panel[0] * n_panels,
            figsize_per_panel[1]
        ),
        squeeze=False
    )

    axes = axes.flatten()

    # ---------------------------------------------------------
    # Plot each measure separately
    # ---------------------------------------------------------
    for ax, measure in zip(axes, measures):

        subset = (
            plot_df.loc[plot_df["measure"] == measure]
            .reset_index(drop=True)
        )

        y = np.arange(len(subset))

        # Offset model and empirical estimates slightly
        model_y = y - 0.10
        data_y = y + 0.10

        # -----------------------------------------------------
        # Model estimates
        # -----------------------------------------------------
        model_xerr = np.vstack([
            subset["model_mean"] - subset["model_lower"],
            subset["model_upper"] - subset["model_mean"]
        ])

        ax.errorbar(
            subset["model_mean"],
            model_y,
            xerr=model_xerr,
            fmt="o",
            capsize=3,
            label="Model"
        )

        # -----------------------------------------------------
        # Empirical targets WITHOUT uncertainty
        # -----------------------------------------------------
        has_data_uncertainty = (
            subset["data_lower"].notna()
            & subset["data_upper"].notna()
        )

        no_data_uncertainty = ~has_data_uncertainty

        if no_data_uncertainty.any():

            ax.scatter(
                subset.loc[no_data_uncertainty, "data_value"],
                data_y[no_data_uncertainty],
                marker="x",
                s=60,
                label="Data"
            )

        # -----------------------------------------------------
        # Empirical targets WITH uncertainty
        # -----------------------------------------------------
        if has_data_uncertainty.any():

            data_with_ci = subset.loc[has_data_uncertainty]

            data_xerr = np.vstack([
                (
                    data_with_ci["data_value"]
                    - data_with_ci["data_lower"]
                ),
                (
                    data_with_ci["data_upper"]
                    - data_with_ci["data_value"]
                )
            ])

            ax.errorbar(
                data_with_ci["data_value"],
                data_y[has_data_uncertainty],
                xerr=data_xerr,
                fmt="x",
                capsize=3,
                markersize=7,
                label="Data"
            )

        # -----------------------------------------------------
        # Formatting
        # -----------------------------------------------------
        ax.set_yticks(y)
        ax.set_yticklabels(subset["label"])

        ax.invert_yaxis()

        ax.set_xlabel(measure)
        ax.set_title(measure)

        ax.grid(
            axis="x",
            alpha=0.25
        )

    # ---------------------------------------------------------
    # Remove duplicate legend entries
    # ---------------------------------------------------------
    handles = []
    labels = []

    for ax in axes:

        h, l = ax.get_legend_handles_labels()

        for handle, label in zip(h, l):

            if label not in labels:
                handles.append(handle)
                labels.append(label)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    plt.show()
    plt.savefig(f'{g_path}/calibration.png', bbox_inches='tight')

    return fig, axes, plot_df

plot_calibration(model_dfs, calibration_targets)
