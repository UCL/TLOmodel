import calendar
import os
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import ast
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy import stats
from scipy.stats import t

from tlo import Date
from tlo.analysis.utils import (
    bin_hsi_event_details,
    compute_summary_statistics,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes
)
from tlo.methods.healthsystem import pool_capabilities_at_levels_1b_and_2

from src.scripts.costing.cost_estimation import (
    estimate_mnh_input_cost_of_scenarios,
    summarize_cost_data,
)

# ----------------------------------------- SCENARIO AND OUTPUTS INFORMATION -----------------------------------------
# Define paths for outputs and resource files
resourcefilepath = Path("./resources")
outputspath = './outputs/sejjj49@ucl.ac.uk/'

# Specify scenario details
scenario = 'ehp_scale_up_scenario-2026-10-27T154504Z'
results_folder= get_scenario_outputs(scenario, outputspath)[-1]
sim_start_year = 2026
sim_end_year = 2027

# create folder for scenario outputs
g_path = f'{outputspath}figures_{scenario}'
if not os.path.isdir(g_path):
    os.makedirs(f'{outputspath}figures_{scenario}')

# Specify scenario labels
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

# Get scenario details
info = get_scenario_info(results_folder)
draws = [x for x in range(info['number_of_draws'])]

# Get pregnancy scaling factor (We estimate pregnancies in 2026 using proportion of pregnancies ending in live births
# from Polis et al. (2015) and UN population props live birth estimates (this is an approximation))
modelled_pop = 40_000
p_scaling_factor =  1 / (modelled_pop / 1_010_000)

# Finally, define the analytical target period
TARGET_PERIOD = (Date(sim_start_year, 1, 1), Date(sim_start_year, 12, 31))

# --------------------------------------------- DEFINE HELPER FUNCTIONS -----------------------------------------------
def return_summary_statistics(
    results: pd.DataFrame,
    method: Literal["mean", "median"] = "mean",
) -> pd.DataFrame:
    """Summarise results across runs within each draw.

    Parameters
    ----------
    results
        DataFrame with MultiIndex columns containing a level named 'draw'.
    method
        'mean': mean and 95% Student's t confidence interval.
        'median': median and empirical 2.5th and 97.5th percentiles.

    Returns
    -------
    pd.DataFrame
        Summary with column levels ('draw', 'stat'). Missing values are
        excluded separately for each row and draw.
    """
    if method not in {"mean", "median"}:
        raise ValueError("method must be 'mean' or 'median'.")

    # Transpose to group columns without the deprecated groupby(axis=1).
    # Convert numeric strings/object columns to numeric.
    # Raise an error if any non-numeric values are present.
    results = results.apply(pd.to_numeric, errors="raise")
    grouped = results.T.groupby(level="draw", sort=False)

    if method == "mean":
        central = grouped.mean().T
        sem = grouped.sem().T

        # Use each row/draw's non-missing run count for degrees of freedom.
        n = grouped.count().T
        critical_value = pd.DataFrame(
            t.ppf(0.975, df=n - 1),
            index=n.index,
            columns=n.columns,
        )

        margin_of_error = critical_value * sem
        lower = central - margin_of_error
        upper = central + margin_of_error

    else:
        central = grouped.median().T
        lower = grouped.quantile(0.025).T
        upper = grouped.quantile(0.975).T

    summary = pd.concat(
        {method: central, "lower": lower, "upper": upper},
        axis=1,
    )

    summary.columns = summary.columns.swaplevel(0, 1)
    summary.columns.names = ["draw", "stat"]

    return summary.sort_index(axis=1)

def get_ps_data_frames(key, results_folder, summary_method):
    """Utility function which extracts a log file output from pregnancy_supervisor.py and returns this dataframe in
    addition to summary statistics."""

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
    results_df_summ = return_summary_statistics(results_df, method=summary_method)

    return {'crude':results_df, 'summarised':results_df_summ}

def return_difference_from_baseline(df, index_title):
    """Return baseline minus intervention outcomes, paired by run.

    Positive values indicate outcomes averted.
    Input columns must have levels named 'draw' and 'run'.
    """
    baseline = df.xs(0, axis=1, level="draw")

    difference_df = baseline.sub(
        df,
        axis="columns",
        level="run",
    ).drop(columns=0, level="draw")

    return difference_df.rename(
        index={sim_start_year: index_title}
    )

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

def summarise_cost_data(_df: pd.DataFrame,
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


def get_mnh_consumable_item_codes(resourcefilepath: Path, *, include_iptp: bool = True,
                                  extra_item_names=(), return_audit: bool = False):
    """Return a set[int], or (set[int], audit DataFrame) if return_audit=True.

    resourcefilepath is the same resources directory passed to the costing
    function. extra_item_names supports explicitly selected delegated services.
    IPTp is included by default as an ANC intervention provided by Malaria.
    No simulation needs to be initialised. Exact item names must map uniquely.
    """
    # Define relevant item codes
    MNH_ITEMS_BY_MODULE = \
        {'care_of_women_during_pregnancy': ('Albendazole 200mg_1000_CMST',
                                          'Bag, urine, collecting, 2000 ml',
                                          'Benzathine benzylpenicillin, powder for injection, 2.4 '
                                          'million IU',
                                          'Blood collecting tube, 5 ml',
                                          'Blood glucose level test',
                                          'Blood, one unit',
                                          'Calcium, tablet, 600 mg',
                                          'Cannula iv  (winged with injection pot) 18_each_CMST',
                                          'Complete blood count',
                                          'Dietary supplements (country-specific)',
                                          'Disposables gloves, powder free, 100 pieces per box',
                                          'Ferrous Salt + Folic Acid, tablet, 200 + 0.25 mg',
                                          'Foley catheter',
                                          'Gauze, absorbent 90cm x 40m_each_CMST',
                                          'Gentamycin, injection, 40 mg/ml in 2 ml vial',
                                          'Giving set iv administration + needle 15 drops/ml_each_CMST',
                                          'Glibenclamide 5mg_1000_CMST',
                                          'Haemoglobin test (HB)',
                                          'Halothane (fluothane)_250ml_CMST',
                                          'Hepatitis B test kit-Dertemine_100 tests_CMST',
                                          'Hydralazine, powder for injection, 20 mg ampoule',
                                          'Insecticide-treated net',
                                          'Insulin soluble 100 IU/ml, 10ml_each_CMST',
                                          'Magnesium sulfate, injection, 500 mg/ml in 10-ml ampoule',
                                          'Methyldopa 250mg_1000_CMST',
                                          'Misoprostol, tablet, 200 mcg',
                                          'Nifedipine 10mg_100_CMST',
                                          'Oxygen, 1000 liters, primarily with oxygen cylinders',
                                          'Oxytocin, injection, 10 IU in 1 ml ampoule',
                                          'Paracetamol, tablet, 500 mg',
                                          'Pethidine, 50 mg/ml, 2 ml ampoule',
                                          'Scalpel blade size 22 (individually wrapped)_100_CMST',
                                          'Sodium chloride, injectable solution, 0,9 %, 500 ml',
                                          'Suture pack',
                                          'Test, Rapid plasma reagin (RPR)',
                                          'Urine analysis'),

        'labour': ('Bag, urine, collecting, 2000 ml',
                  'Benzathine benzylpenicillin, powder for injection, 2.4 million IU',
                  'Benzylpenicillin 3g (5MU), PFR_each_CMST',
                  'Blood collecting tube, 5 ml',
                  'Blood, one unit',
                  'Cannula iv  (winged with injection pot) 18_each_CMST',
                  'Ceftriaxone 1g, PFR_each_CMST',
                  'Chlorhexidine 1.5% solution_5_CMST',
                  'Clean delivery kit',
                  'Complete blood count',
                  'Declofenac injection_each_CMST',
                  'Dexamethasone 5mg/ml, 5ml_each_CMST',
                  'Disposables gloves, powder free, 100 pieces per box',
                  'Ferrous Salt + Folic Acid, tablet, 200 + 0.25 mg',
                  'Foley catheter',
                  'Gauze, absorbent 90cm x 40m_each_CMST',
                  'Gentamycin, injection, 40 mg/ml in 2 ml vial',
                  'Giving set iv administration + needle 15 drops/ml_each_CMST',
                  'Haemoglobin test (HB)',
                  'Halothane (fluothane)_250ml_CMST',
                  'Hydralazine, powder for injection, 20 mg ampoule',
                  'Infant resuscitator, clear plastic + mask + bag_each_CMST',
                  'Lidocaine HCl (in dextrose 7.5%), ampoule 2 ml',
                  'Magnesium sulfate, injection, 500 mg/ml in 10-ml ampoule',
                  'Methyldopa 250mg_1000_CMST',
                  'Metronidazole 200mg_1000_CMST',
                  'Misoprostol, tablet, 200 mcg',
                  'Nifedipine 10mg_100_CMST',
                  'Oxygen, 1000 liters, primarily with oxygen cylinders',
                  'Oxytocin, injection, 10 IU in 1 ml ampoule',
                  'Paracetamol, tablet, 500 mg',
                  'Pethidine, 50 mg/ml, 2 ml ampoule',
                  'Scalpel blade size 22 (individually wrapped)_100_CMST',
                  'Sodium chloride, injectable solution, 0,9 %, 500 ml',
                  'Suture pack',
                  'Vacuum, obstetric',
                  "ringer's lactate (Hartmann's solution), 1000 ml_12_IDA"),

        'newborn_outcomes': ('Benzylpenicillin 1g (1MU), PFR_Each_CMST',
                            'Blood collecting tube, 5 ml',
                            'Cannula iv  (winged with injection pot) 18_each_CMST',
                            'Dextrose (glucose) 5%, 1000ml_each_CMST',
                            'Disposables gloves, powder free, 100 pieces per box',
                            'Gentamicin 40mg/ml, 2ml_each_CMST',
                            'Giving set iv administration + needle 15 drops/ml_each_CMST',
                            'Oxygen, 1000 liters, primarily with oxygen cylinders',
                            'Tetracycline eye ointment, 1 %, tube 5 mg',
                            'Tube, feeding CH 8_each_CMST',
                            'vitamin K1  (phytomenadione) 1 mg/ml, 1 ml, inj._100_IDA'),

        'postnatal_supervisor': ('Bag, urine, collecting, 2000 ml',
                                'Cannula iv  (winged with injection pot) 18_each_CMST',
                                'Ceftriaxone 1g, PFR_each_CMST',
                                'Chlorhexidine 1.5% solution_5_CMST',
                                'Declofenac injection_each_CMST',
                                'Foley catheter',
                                'Giving set iv administration + needle 15 drops/ml_each_CMST',
                                'Halothane (fluothane)_250ml_CMST',
                                'Metronidazole 200mg_1000_CMST',
                                'Paracetamol, tablet, 500 mg',
                                'Scalpel blade size 22 (individually wrapped)_100_CMST',
                                'Sodium chloride, injectable solution, 0,9 %, 500 ml',
                                "ringer's lactate (Hartmann's solution), 1000 ml_12_IDA")}

    # Including IPTP
    IPTP_ITEM = 'Fansidar (sulphadoxine / pyrimethamine tab)'

    # Get item codes resource file
    path = (Path(resourcefilepath) / 'healthsystem' / 'consumables' /
            'ResourceFile_Consumables_Items_and_Packages.csv')
    lookup = pd.read_csv(path, encoding='ISO-8859-1', low_memory=False)
    lookup = lookup[['Items', 'Item_Code']].drop_duplicates()
    selections = {module: set(names) for module, names in MNH_ITEMS_BY_MODULE.items()}

    if include_iptp:
        selections['malaria_iptp_delegated_from_anc'] = {IPTP_ITEM}

    selections['additional_explicit_items'] = set(extra_item_names)
    names = set().union(*selections.values())
    rows = []

    for name in sorted(names):
        codes = lookup.loc[lookup['Items'].eq(name), 'Item_Code'].drop_duplicates()

        if len(codes) != 1 or codes.isna().any():
            raise ValueError(f'Expected one item code for {name!r}; found {codes.tolist()}')

        value = float(codes.iloc[0])
        if not value.is_integer():
            raise ValueError(f'Non-integer item code for {name!r}: {value}')

        rows.append({'Item_Code': int(value), 'Items': name,
                     'source_modules': '; '.join(m for m, items in selections.items() if name in items)})

    audit = pd.DataFrame(rows).sort_values(['Item_Code', 'Items']).reset_index(drop=True)
    item_codes = set(audit['Item_Code'].astype(int))

    return (item_codes, audit) if return_audit else item_codes


# --------------------------------------------- EXTRACT INITIAL OUTPUTS ------------------------------------------------
preg_outcomes = {k:get_ps_data_frames(k, results_folder, 'mean') for k in
                   ['mat_comp_incidence', 'nb_comp_incidence', 'deaths_and_stillbirths','service_coverage', 'met_need',
                    'yearly_mnh_counter_dict', 'intervention_coverage']}

#  =================================== FIGURE 1 - EFFECTIVE INTERVENTION COVERAGE  ====================================
#  For the first figure we output the effective coverage of each intervention/intervention package in the baseline
#  scenario and compare this to the effective coverage in the intervention scenario

effective_coverage_df = preg_outcomes['met_need']['summarised']

def produce_fig_1(df, method="mean"):
    """Plot coverage using either mean and 95% CI or median and quantiles.

    Parameters
    ----------
    df : pd.DataFrame
        Summary data indexed by intervention, with MultiIndex columns
        named 'draw' and 'stat'. Statistics must include 'lower', 'upper',
        and either 'mean' or 'median'.
    method : {'mean', 'median'}
        Central statistic to plot. The supplied lower and upper bounds
        should correspond to 95% confidence intervals for 'mean', or
        2.5th and 97.5th percentiles for 'median'.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """
    if method not in {"mean", "median"}:
        raise ValueError("method must be 'mean' or 'median'.")

    scenarios = [
        {
            "name": "Abortion/Ectopic \nCM",
            "draw": 1,
            "items": [
                ("pac_ep", "Post-abortion care"),
            ],
        },
        {
            "name": "Maternal sepsis \nCM",
            "draw": 2,
            "items": [
                ("m_sepsis_cm", "Sepsis management"),
            ],
        },
        {
            "name": "Maternal haem.\nCM",
            "draw": 3,
            "items": [
                ("haem_cm_ut", "Uterotonics"),
                ("haem_cm_mrp", "MRRP"),
                ("haem_cm_blood_pph", "Blood (PPH)†"),
                ("heam_cm_blood_aph", "Blood (APH)"),
            ],
        },
        {
            "name": "Obstructed labour\nCM",
            "draw": 4,
            "items": [
                ("ol_cm", "AVD"),
            ],
        },
        {
            "name": "Eclampsia/SPE CM",
            "draw": 5,
            "items": [
                ("ec_cm_mgso4", "Magnesium sulphate (E)"),
                ("spe_cm_mgso4", "Magnesium sulphate (SPE)"),
                ("spe_ec_cm_htns", "Antihypertensives"),
            ],
        },
        {
            "name": "CS & Obstetric\nsurgery",
            "draw": 6,
            "items": [
                ("cs_surg_aph", "CS and IP surgery†"),
                ("cs_surg_pph", "PP surgery†"),
            ],
        },
        {
            "name": "Neonatal sepsis CM",
            "draw": 7,
            "items": [
                ("n_sepsis_cm", "Sepsis management"),
            ],
        },
        {
            "name": "Preterm birth CM",
            "draw": 8,
            "items": [
                ("ptb_cm_resus", "Resuscitation"),
                ("ptb_cm_sepsis", "Sepsis management"),
                ("ptb_cm_kmc", "KMC"),
            ],
        },
        {
            "name": "Newborn Resus.",
            "draw": 9,
            "items": [
                ("neo_resus", "Resuscitation"),
            ],
        },
    ]

    # Plot settings
    baseline_colour = "#BDBDBD"
    increased_colour = "#377EB8"

    bar_width = 0.36
    component_spacing = 1.0
    package_gap = 1.05

    # Validate DataFrame structure.
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("df must have MultiIndex columns.")

    if df.columns.nlevels != 2 or set(df.columns.names) != {"draw", "stat"}:
        raise ValueError(
            "df must have two column levels named 'draw' and 'stat'."
        )

    # Ensure tuple indexing below uses the order (draw, stat).
    df = df.reorder_levels(["draw", "stat"], axis=1)

    if not df.columns.is_unique or not df.index.is_unique:
        raise ValueError("df must have unique row and column labels.")

    required_stats = {"lower", method, "upper"}
    available_draws = set(df.columns.get_level_values("draw"))
    required_draws = {0} | {scenario["draw"] for scenario in scenarios}

    missing_draws = required_draws - available_draws
    if missing_draws:
        raise KeyError(f"Missing required draws: {sorted(missing_draws)}")

    for draw in sorted(required_draws):
        available_stats = set(df[draw].columns)
        missing_stats = required_stats - available_stats

        if missing_stats:
            raise ValueError(
                f"Draw {draw} is missing statistics: {sorted(missing_stats)}"
            )

    # Extract results and calculate positions.
    plot_data = []
    package_centres = []
    package_boundaries = []

    current_x = 0.0

    for scenario_number, scenario in enumerate(scenarios):
        draw = scenario["draw"]
        package_positions = []

        for row_id, item_label in scenario["items"]:
            if row_id not in df.index:
                raise KeyError(
                    f"Row '{row_id}' from '{scenario['name']}' "
                    "is not present in the DataFrame."
                )

            item = {
                "package": scenario["name"],
                "component": item_label,
                "x": current_x,
            }

            for prefix, selected_draw in [
                ("baseline", 0),
                ("increased", draw),
            ]:
                central = float(df.loc[row_id, (selected_draw, method)])
                lower = float(df.loc[row_id, (selected_draw, "lower")])
                upper = float(df.loc[row_id, (selected_draw, "upper")])

                if not np.isfinite([central, lower, upper]).all():
                    raise ValueError(
                        f"Non-finite summary values for row '{row_id}', "
                        f"draw {selected_draw}."
                    )

                if not lower <= central <= upper:
                    raise ValueError(
                        f"Expected lower <= {method} <= upper for "
                        f"row '{row_id}', draw {selected_draw}; "
                        f"got {lower}, {central}, {upper}."
                    )

                item[f"{prefix}_central"] = central
                item[f"{prefix}_lower_error"] = central - lower
                item[f"{prefix}_upper_error"] = upper - central
                item[f"{prefix}_upper"] = upper

            plot_data.append(item)
            package_positions.append(current_x)
            current_x += component_spacing

        package_centres.append(
            {
                "name": scenario["name"],
                "x": np.mean(package_positions),
            }
        )

        if scenario_number < len(scenarios) - 1:
            package_boundaries.append(
                current_x - component_spacing / 2 + package_gap / 2
            )

        current_x += package_gap

    # Create plot.
    fig_width = max(16, len(plot_data) * 1.25)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    upper_limits = []

    for item in plot_data:
        for prefix, offset, colour, hatch in [
            ("baseline", -bar_width / 2, baseline_colour, None),
            ("increased", bar_width / 2, increased_colour, ".."),
        ]:
            central = item[f"{prefix}_central"]

            bars = ax.bar(
                item["x"] + offset,
                central,
                width=bar_width,
                yerr=np.array(
                    [
                        [item[f"{prefix}_lower_error"]],
                        [item[f"{prefix}_upper_error"]],
                    ]
                ),
                capsize=3,
                color=colour,
                edgecolor="black",
                linewidth=0.6,
                hatch=hatch,
                error_kw={"elinewidth": 1},
            )

            ax.bar_label(
                bars,
                labels=[f"{central:.1f}%"],
                padding=7,
                fontsize=8,
                rotation=90,
            )

            upper_limits.append(item[f"{prefix}_upper"])

    # Component labels.
    ax.set_xticks([item["x"] for item in plot_data])
    ax.set_xticklabels(
        [item["component"] for item in plot_data],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.tick_params(axis="x", pad=4)

    # Package titles beneath the component labels.
    for package in package_centres:
        ax.text(
            package["x"],
            -0.34,
            package["name"],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            linespacing=1.1,
            clip_on=False,
        )

    # Package separators.
    for boundary in package_boundaries:
        ax.axvline(
            boundary,
            color="0.82",
            linewidth=0.9,
            linestyle="--",
            zorder=0,
        )

    # Legend identifies the selected summary method.
    legend_handles = [
        Patch(
            facecolor=baseline_colour,
            edgecolor="black",
            label="Baseline coverage",
        ),
        Patch(
            facecolor=increased_colour,
            edgecolor="black",
            hatch="..",
            label="Increased coverage",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper right",
        title=(
            "Mean (95% CI)"
            if method == "mean"
            else "Median (2.5–97.5% quantiles)"
        ),
        title_fontsize=9,
    )

    # General formatting.
    ax.set_ylabel("Coverage (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, max(1, max(upper_limits) * 1.20))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.96,
        bottom=0.40,
    )

    fig.text(
        0.07,
        0.025,
        "† Denominator approximated: clinical indication cannot be determined "
        "for women who do not seek care.",
        ha="left",
        va="bottom",
        fontsize=9,
    )

    # Uses your existing g_path variable.
    fig.savefig(
        f"{g_path}/figure_1_met_need.png",
        bbox_inches="tight",
        dpi=300,
    )

    plt.show()

    return fig, ax

produce_fig_1(effective_coverage_df, method="mean")

#  ================================= FIGURE 2 - DEATHS, STILLBIRTHS AND DALYS AVERTED  ================================
#  For the second figure we the number of maternal deaths, neonatal deaths and stillbirths averted during each scenario
# alongside DALYS (total, total adjusted to include YLLs from stillbirths and maternal/perinatal only).

# ------------------------------------ Output direct maternal deaths averted ------------------------------------------
#  First get all deaths by cause/label
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
num_deaths_by_cause_label = num_deaths_by_cause_label.fillna(0)

# Isolate maternal deaths and scale
direct_deaths = num_deaths_by_cause_label.loc['Maternal Disorders'].reindex(
    num_deaths_by_cause_label.columns).to_frame().T
direct_deaths = direct_deaths * p_scaling_factor
direct_deaths.rename(index={"Maternal Disorders": sim_start_year}, inplace=True)

# Get difference
direct_mat_deaths_averted_df = return_difference_from_baseline(direct_deaths, 'mat_direct_deaths_averted')

# ------------------------------------ Output direct maternal DALYS averted ------------------------------------------
# Extract DALYs by cause
# todo: we should include 2027 result also
dalys_by_cause = extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns=['date', 'sex', 'age_range']).groupby(['year']).sum().stack()),
            do_scaling=False)
dalys_by_cause = dalys_by_cause.loc[dalys_by_cause.index.get_level_values(0) != sim_end_year]

# Isolate maternal DALYS and scale
mat_dalys = dalys_by_cause.loc[sim_start_year, 'Maternal Disorders'].reindex(dalys_by_cause.columns).to_frame().T
mat_dalys = mat_dalys * p_scaling_factor
mat_dalys.index = mat_dalys.index.droplevel(1)

# Get difference
mat_dalys_averted_df = return_difference_from_baseline(mat_dalys, 'mat_dalys_averted')

# ------------------------------------ Output direct neonatal deaths averted ------------------------------------------
# Extract neonatal deaths and scale them
# todo: same scaling factor?
direct_neo_deaths = num_deaths_by_cause_label.loc['Neonatal Disorders'].reindex(
    num_deaths_by_cause_label.columns).to_frame().T
direct_neo_deaths = direct_neo_deaths * p_scaling_factor
direct_neo_deaths.rename(index={"Neonatal Disorders": sim_start_year}, inplace=True)

# Get difference
direct_neo_deaths_averted_df = return_difference_from_baseline(direct_neo_deaths, 'direct_neo_deaths_averted')

# ------------------------------------ Output direct neonatal DALYS averted ------------------------------------------
# Extract neonatal DALYS and scale them
neo_dalys = dalys_by_cause.loc[sim_start_year, 'Neonatal Disorders'].reindex(dalys_by_cause.columns).to_frame().T
neo_dalys = neo_dalys * p_scaling_factor
neo_dalys.index = neo_dalys.index.droplevel(1)

# Get difference
neo_dalys_averted_df = return_difference_from_baseline(neo_dalys, 'neo_dalys_averted')

# --------------------------------------- Output stillbirths averted -------------------------------------------------
# Extract stillbirths and scale them
stillbirths = preg_outcomes['deaths_and_stillbirths']['crude'].loc['total_stillbirths'].reindex(
    dalys_by_cause.columns).to_frame().T
stillbirths = stillbirths * p_scaling_factor
stillbirths.rename(index={"total_stillbirths": sim_start_year}, inplace=True)

# Get difference
stillbirths_averted_df = return_difference_from_baseline(stillbirths, 'stillbirths_averted')

# ------------------------------------ Output total direct deaths averted --------------------------------------------
# Combine maternal and neonatal deaths
mat_neo_deaths_averted = direct_mat_deaths_averted_df.copy()
mat_neo_deaths_averted.iloc[:, :] = direct_mat_deaths_averted_df.to_numpy() + direct_neo_deaths_averted_df.to_numpy()
mat_neo_deaths_averted.rename(index={'mat_direct_deaths_averted': "mat_neo_deaths_averted"}, inplace=True)

# --------------------------------------- Output total DALYS averted -------------------------------------------------
# Extract DALYS and get difference
total_dalys = dalys_by_cause.groupby(['year']).sum()
total_dalys = total_dalys * p_scaling_factor
total_dalys_averted_df = return_difference_from_baseline(total_dalys, 'total_dalys_averted')

# --------------------------- Output total DALYS (including stillbirth YLL) averted ----------------------------------
# Extract pregnancy losses occurring over 28 weeks
preg_loss = extract_results(
    results_folder,
    module="tlo.methods.pregnancy_supervisor",
    key="pregnancy_loss",
    custom_generate_series=(
        lambda df: df.loc[(df['gest_age'] > 28)].assign(
            year=df['date'].dt.year).groupby(['year'])['year'].count()),
    do_scaling=False)

# Ascribe 90 YLL to each stillbirth and scale results
# todo: consider weighting based on gestational age
preg_loss_yll = preg_loss * 90
preg_loss_yll = preg_loss_yll * p_scaling_factor

# add YLLs from pregnancy loss to DALY total
adj_dalys = total_dalys + preg_loss_yll

# get the difference
adj_dalys_averted_df = return_difference_from_baseline(adj_dalys, 'adj_dalys_averted')

# ------------------------------------ Output maternal/perinatal DALYs averted ---------------------------------------
# Make a copy of maternal DALYs df
mat_neo_dalys_averted = mat_dalys_averted_df.copy()
mat_neo_stillbirth_dalys_averted = mat_dalys_averted_df.copy()

# Get the difference in YLLs due to pregnancy loss
preg_loss_averted_averted_df = return_difference_from_baseline(preg_loss_yll, sim_start_year)

# Combine maternal and neonatal DALYs
mat_neo_dalys_averted.iloc[:, :] = mat_dalys_averted_df.to_numpy() + neo_dalys_averted_df.to_numpy()
mat_neo_dalys_averted.rename(index={'mat_dalys_averted': "mat_neo_dalys_averted"}, inplace=True)

# Combine maternal, neonatal and stillbirth DALYs
mat_neo_stillbirth_dalys_averted.iloc[:, :] = (mat_dalys_averted_df.to_numpy() +
                                               neo_dalys_averted_df.to_numpy() +
                                               preg_loss_averted_averted_df.to_numpy())
mat_neo_stillbirth_dalys_averted.rename(index={'mat_dalys_averted': "mat_neo_stillbirth_dalys_averted"}, inplace=True)

# ------------------------------------ Prepare data for outputting Figure 2 ------------------------------------------
# Combine dataframes detailing deaths averted
death_plot_df = pd.concat([direct_mat_deaths_averted_df, direct_neo_deaths_averted_df, stillbirths_averted_df,
                           mat_neo_deaths_averted], axis=0)
death_plot_df_summ = return_summary_statistics(death_plot_df, 'mean')

# Define labels
death_outcome_labels = {
    "mat_direct_deaths_averted": "Direct maternal deaths averted",
    "direct_neo_deaths_averted": "Direct neonatal deaths averted",
    "stillbirths_averted": "Stillbirths averted",
    "mat_neo_deaths_averted": "Direct maternal and neonatal deaths averted"
}

# Combine dataframes detailing deaths averted
dalys_outcomes_df = pd.concat([
    total_dalys_averted_df, adj_dalys_averted_df, mat_neo_dalys_averted, mat_neo_stillbirth_dalys_averted], axis=0)
dalys_outcomes_df_summ = return_summary_statistics(dalys_outcomes_df, 'mean')

# Define labels
dalys_outcome_labels = {
    "total_dalys_averted": "All-cause DALYs averted",
    "adj_dalys_averted": "All-cause DALYs averted (inc. pregnancy loss)",
    "mat_neo_dalys_averted": "Maternal and neonatal DALYs averted",
    "mat_neo_stillbirth_dalys_averted": "Maternal and perinatal DALYs averted"}

# Define panel colours
panel_colours = [
    [
        "#4C6A92",  # muted blue
        "#8A6F8F",  # muted purple
        "#6F8F7A",  # muted green
        "#327D80",  # deep teal
    ],
    [
        "#B07A6A",  # muted terracotta
        "#C09A5B",  # muted ochre
        "#6F858F",  # muted slate
        "#A65378",  # muted raspberry
    ],
]

def format_bar_value(value):
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K".replace(".0K", "K")
    return f"{value:,.0f}"

# todo: should this be the median?
def baseline_mean(df):
    """Mean baseline outcome across runs."""
    return df.xs(0, axis=1, level="draw").iloc[0].mean()

baseline_outcomes = {
    "mat_direct_deaths_averted": baseline_mean(direct_deaths),
    "direct_neo_deaths_averted": baseline_mean(direct_neo_deaths.loc[[sim_start_year]]),
    "mat_neo_deaths_averted":(baseline_mean(direct_deaths) + baseline_mean(direct_neo_deaths.loc[[sim_start_year]])),
    "stillbirths_averted": baseline_mean(stillbirths),
    "total_dalys_averted": baseline_mean(total_dalys.loc[[sim_start_year]]),
    "adj_dalys_averted": baseline_mean(adj_dalys.loc[[sim_start_year]]),
    "mat_neo_dalys_averted": (baseline_mean(mat_dalys) + baseline_mean(neo_dalys)),
    "mat_neo_stillbirth_dalys_averted": baseline_mean(mat_dalys) + baseline_mean(neo_dalys) + baseline_mean(preg_loss_yll)

}

def produce_fig_2(
    death_data,
    daly_data,
    death_labels,
    daly_labels,
    save_title,
    baseline_outcomes,
    method="mean",
):
    """Plot mean with 95% CIs or median with 2.5–97.5% quantiles.

    Input DataFrames must have outcomes as rows and MultiIndex columns
    named 'draw' and 'stat', including the selected central statistic
    ('mean' or 'median'), 'lower', and 'upper'.

    baseline_outcomes supplies the denominator for percentage annotations.
    """
    if method not in {"mean", "median"}:
        raise ValueError("method must be 'mean' or 'median'.")

    interval_label = (
        "Mean (95% CI)"
        if method == "mean"
        else "Median (2.5–97.5% quantiles)"
    )

    datasets = [
        (death_data, death_labels, "Deaths averted"),
        (daly_data, daly_labels, "DALYs averted"),
    ]

    # Validate the statistics available in each dataset.
    for data, _, ylabel in datasets:
        required_stats = {method, "lower", "upper"}
        available_stats = set(data.columns.get_level_values("stat"))
        missing_stats = required_stats - available_stats

        if missing_stats:
            raise ValueError(
                f"{ylabel} data is missing statistics: "
                f"{sorted(missing_stats)}"
            )

    # Scenario ordering.
    draws = sorted(
        death_data.columns.get_level_values("draw").unique()
    )

    daly_draws = set(daly_data.columns.get_level_values("draw"))
    if set(draws) != daly_draws:
        raise ValueError(
            "death_data and daly_data must contain the same draws."
        )

    scenario_labels = [
        draw_labels.get(draw, f"Scenario {draw}")
        for draw in draws
    ]

    x = np.arange(len(draws))

    # Create two vertically stacked panels with a shared x-axis.
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(max(14, len(draws) * 1.2), 11),
    )

    for panel_i, (ax, (data, outcome_labels, ylabel)) in enumerate(
        zip(axes, datasets)
    ):
        plot_df = data.copy()
        plot_df.index.name = "outcome"

        outcomes = plot_df.index.tolist()
        n_outcomes = len(outcomes)

        group_width = 0.8
        bar_width = group_width / n_outcomes

        # Extract each statistic as an outcome-by-draw DataFrame.
        central_data = plot_df.xs(
            method, level="stat", axis=1
        ).reindex(columns=draws)

        lower_data = plot_df.xs(
            "lower", level="stat", axis=1
        ).reindex(columns=draws)

        upper_data = plot_df.xs(
            "upper", level="stat", axis=1
        ).reindex(columns=draws)

        for i, outcome in enumerate(outcomes):
            central = central_data.loc[outcome].to_numpy(dtype=float)
            lower = lower_data.loc[outcome].to_numpy(dtype=float)
            upper = upper_data.loc[outcome].to_numpy(dtype=float)

            # Asymmetric errors relative to the mean or median.
            yerr = np.vstack([
                central - lower,
                upper - central,
            ])

            # Centre bars around the intervention position.
            offset = (i - (n_outcomes - 1) / 2) * bar_width

            bars = ax.bar(
                x + offset,
                central,
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
                    outcome.replace("_", " ").title(),
                ),
            )

            for bar, value, lo, hi in zip(
                bars, central, lower, upper
            ):
                if not np.isfinite([value, lo, hi]).all():
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
                    pct_averted = 100 * value / baseline_value
                    annotation = (
                        f"{format_bar_value(value)} "
                        f"({pct_averted:.1f}%)"
                    )
                else:
                    annotation = f"{format_bar_value(value)} (N/A)"

                ax.annotate(
                    annotation,
                    xy=(
                        bar.get_x() + bar.get_width() / 2,
                        label_y,
                    ),
                    xytext=(0, offset_points),
                    textcoords="offset points",
                    ha="center",
                    va=vertical_alignment,
                    rotation=90,
                    fontsize=7,
                    clip_on=False,
                )

        # Panel formatting.
        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=0.8,
        )

        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

        ax.legend(
            frameon=False,
            loc="upper left",
            title=interval_label,
            title_fontsize=9,
        )

    axes[0].set_title(
        "A. Mortality outcomes",
        loc="left",
        fontweight="bold",
    )

    axes[1].set_title(
        "B. DALY outcomes",
        loc="left",
        fontweight="bold",
    )

    # Shared x-axis.
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        scenario_labels,
        rotation=45,
        ha="right",
    )
    axes[1].set_xlabel("Intervention")

    axes[0].tick_params(
        axis="x",
        which="both",
        bottom=False,
        labelbottom=False,
    )

    fig.tight_layout(h_pad=1.5)

    fig.savefig(
        f"{g_path}/{save_title}.png",
        bbox_inches="tight",
        dpi=300,
    )

    plt.show()

    return fig, axes

produce_fig_2(
    death_plot_df_summ,
    dalys_outcomes_df_summ,
    death_outcome_labels,
    dalys_outcome_labels,
    "fig_2_health_outcomes",
    baseline_outcomes,
    method='mean'
)

#  ============================ FIGURE S1 and S2 - DEATHS AND DALYS BY CAUSE  =========================================
# Define the function to output heatmaps
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

# ---------------------------------------- DEATHS BY CAUSE -------------------------------------------------------------
# Extract deaths by cause and scale
# todo: this should possibly be limited to direct MNH deaths only
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

# Get difference
baseline = deaths_by_c.xs(0, axis=1, level="draw")
deaths_by_c_averted_df = baseline.sub(
    deaths_by_c,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
deaths_by_c_averted_df = return_summary_statistics(deaths_by_c_averted_df, 'mean')

# ---------------------------------------- DALYS BY CAUSE -------------------------------------------------------------
# TODO: currently this is all-cause DALYS unadjusted and therefore doesnt include stillbirth YLLs. so we may need to do
#  multiple figures
# Extract DALYs by cause
dalys_by_cause = dalys_by_cause.loc[dalys_by_cause.index.get_level_values(0) != sim_end_year]
dalys_by_cause = dalys_by_cause * p_scaling_factor
dalys_by_cause = dalys_by_cause.droplevel(0, axis=0)

# Get difference
baseline = dalys_by_cause.xs(0, axis=1, level="draw")
dalys_averted_by_cause_df = baseline.sub(
    dalys_by_cause,
    axis="columns",
    level="run"
).drop(columns=0, level="draw")
dalys_averted_by_cause_df = return_summary_statistics(dalys_averted_by_cause_df, 'mean')

# ----------------------------------------- output figures S1 & S2 -----------------------------------------------------
figure_S1_and_S2_heatmap(deaths_by_c_averted_df,
                                             'Deaths averted by cause and intervention scenario',
                                             'fig_S1_diff_in_deaths_by_cause_heatmap', True)


figure_S1_and_S2_heatmap(dalys_averted_by_cause_df,
                                             'DALYs averted by cause and intervention scenario',
                                             'fig_S2_diff_in_dalys_by_cause_heatmap', True)

#  =================================== TABLE 3 - ECONOMIC EVALUATION  =================================================
# Here we estimate the incremental cost for each scenario compared to the baseline

# 1.) --------------------------- Estimate incremental costs due to use of MNH consumables ----------------------------
# First we estimate the incremental costs associated with maternal and newborn health consumables

# This function returns the relevant item codes
mnh_item_codes, item_audit = get_mnh_consumable_item_codes(
    resourcefilepath,
    include_iptp=True,
    return_audit=True,
)

# Next we extract system consumables costs via a broader system costing function
list_of_relevant_years_for_costing = list(range(TARGET_PERIOD[0].year, TARGET_PERIOD[-1].year + 1))

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

# These costs are then summed for each scenario
consumable_costs = (input_costs_df.loc[input_costs_df["cost_category"].eq(
    "medical consumables")].groupby(["draw", "run"])["cost"].sum())

total_consumable_costs = consumable_costs.sum()

# 2.) ---------------------- Estimate incremental costs due to use of MNH HCW time ------------------------------------
# Next we estimate the incremental costs associated with increased HCW time use for MNH HSIs.
#
# We start by estimating scenario level differences in HCW time use for relevant HSIs in minutes/hours
MNH_PREFIXES = ('AntenatalCare_', 'DeliveryCare_', 'PostnatalCare_')
KEYS = ['year', 'Facility_Level', 'OfficerType']

def _mapping(value):
    """Accept logged Python literals or already-parsed objects without eval."""
    # literal_eval parses data only; it does not execute arbitrary expressions.
    return ast.literal_eval(value) if isinstance(value, str) else value

def _event_minutes(event, count, time_lookup, prefixes, extra_ids):
    """Return HCW minutes, accounting for appointment quantities."""
    output = Counter()
    treatment = event["treatment_id"]

    # Ignore treatments outside the selected MNH or additional treatment IDs.
    if not (
        treatment.startswith(tuple(prefixes))
        or treatment in extra_ids
    ):
        return output

    level = str(event["facility_level"])

    # Use the recorded actual footprint. Empty footprints contribute no time.
    footprint = _mapping(event["appt_footprint"])
    items = footprint.items() if isinstance(footprint, Mapping) else footprint

    for appt, quantity in items:
        # Each appointment quantity applies to every occurrence of this event.
        # The lookup supplies minutes per appointment for each relevant cadre.
        for officer, minutes in time_lookup[(level, appt)]:
            output[(treatment, level, officer)] += (
                float(count) * float(quantity) * float(minutes)
            )

    return output

def summarize_hcw_time_differences(df, value, group_keys=None):
    """Summarise differences at the supplied grouping level.

    Aggregate within each run FIRST if an overall total is desired, then call
    with group_keys=['draw'] or ['draw', 'year']. Summing group SDs, SEs or
    percentile bounds would ignore covariance and does not give total uncertainty.
    """
    group_keys = ['draw'] + KEYS if group_keys is None else group_keys

    # Assumes one value per run within each summary group. n counts nonmissing
    # observations, not necessarily distinct runs; duplicate rows must be grouped
    # within run first. A single observation yields undefined sample SD/SE.
    result = df.groupby(group_keys)[value].agg(
        n='count', mean='mean', sd='std',
        p025=lambda x: x.quantile(.025), p975=lambda x: x.quantile(.975)).reset_index()

    # SE of the estimated mean from independent simulation repetitions. It
    # decreases with more runs; the distribution's percentile range need not.
    result['mc_se_mean'] = result.sd / np.sqrt(result.n)

    return result

def estimate_mnh_hcw_time(results_folder, resourcefilepath, *, years,
                          baseline_draw=0, draws=None, runs=None,
                          scaling_factor=None, suspended_results_folder=None,
                          treatment_prefixes=MNH_PREFIXES, extra_treatment_ids=(),
                          quantity_mode='model'):
    """Return (time_by_hsi, paired_differences, summary_across_runs).

    years: explicit calendar years covered by the requested count logs.
      Empty period bins count as zero; verify the logs cover these years.
    draws/runs: select outputs; bin_hsi_event_details still reads all runs.
    scaling_factor: positive scalar, or dict keyed by (draw, run). None reads
      Demography.scaling_factor per run. For suspended output, None reads
      suspended_results_folder/0/0, matching the costing function convention.
    quantity_mode: 'model' reproduces this branch's appointment conversion;
      'footprint' instead multiplies by appointment quantities, an alternative
      workload assumption rather than an exact reproduction of model requests.
    paired_differences: scenario minus baseline with matching run number.
      Pairing does not imply identical random streams after scenarios diverge.
    summary: mean difference, SD, MC SE, and empirical 2.5/97.5 percentiles
      of run differences. Percentiles are NOT a CI for the mean difference.
    """

    if quantity_mode not in {'model', 'footprint'}:
        raise ValueError("quantity_mode must be 'model' or 'footprint'")

    # Specify years of interest
    years = sorted(set(int(y) for y in years))
    if not years:
        raise ValueError('Specify at least one complete calendar year')

    # Specify number of draws and nuns
    root = Path(results_folder)
    info = get_scenario_info(root)

    draws = list(range(info['number_of_draws'])) if draws is None else list(draws)
    draws = sorted(set(draws) | {baseline_draw})
    runs = list(range(info['runs_per_draw'])) if runs is None else list(runs)

    if not runs:
        raise ValueError('No runs selected')

    # Read resource file which stores mins per appt footprint
    table = pd.read_csv(Path(resourcefilepath) / 'healthsystem/human_resources/definitions/'
                        'ResourceFile_Appt_Time_Table.csv', dtype={'Facility_Level': str})

    # Store required appointment time by level, appt footprint and cadre
    lookup = {key: list(zip(g.Officer_Category, g.Time_Taken_Mins))
              for key, g in table.groupby(['Facility_Level', 'Appt_Type_Code'])}

    def get_minutes(event, count):
        return _event_minutes(event, count, lookup, treatment_prefixes, extra_treatment_ids)

    # Resolve one population multiplier for all selected outputs. Use provided scaling factor first
    if scaling_factor is None:
        if suspended_results_folder is None:
            sf_root, sf_draw, sf_run = root, baseline_draw, runs[0]
        else:
            sf_root = Path(suspended_results_folder)
            sf_draw, sf_run = 0, 0

        scaling_factor = load_pickled_dataframes(
            sf_root,
            sf_draw,
            sf_run,
            "tlo.methods.demography",
        )["tlo.methods.demography"]["scaling_factor"][
            "scaling_factor"
        ].iloc[0]

    try:
        scaling_factor = float(scaling_factor)

    except (TypeError, ValueError):
        raise ValueError("scaling_factor must be a single positive, finite number") from None

    if not np.isfinite(scaling_factor) or scaling_factor <= 0:
        raise ValueError(f"Invalid scaling factor: {scaling_factor}")

    # Output HSI details including minutes of HCW time used by scenario
    records = []
    for year in years:
        binned = bin_hsi_event_details(
            results_folder=root,
            get_counter_from_event_details=get_minutes,
            start_date=pd.Timestamp(year, 1, 1),
            end_date=pd.Timestamp(year, 12, 31, 23, 59, 59),
            do_scaling=False,
        )
        for draw in draws:
            for run in runs:
                for (treatment, level, officer), minutes in binned[(draw, run)].items():
                    # do_scaling=False above prevents applying the multiplier twice.
                    # Fractional scaled minutes are retained; do not round to staff.
                    records.append((draw, run, year, treatment, level, officer,
                                    minutes * scaling_factor))

    # Concatenate into single record
    by_hsi = pd.DataFrame(records, columns=['draw', 'run', 'year', 'TREATMENT_ID',
                                           'Facility_Level', 'OfficerType', 'minutes'])
    if by_hsi.empty:
        raise ValueError('No matching MNH HSI footprints in selected runs/years')

    # Collapse treatment IDs
    total = by_hsi.groupby(['draw', 'run'] + KEYS)['minutes'].sum()

    # Zero-fill *observed resource combinations* across all selected runs/years,
    # after extraction. Empty bins represent zero matching workload; the utility
    # raises for missing log files/keys rather than silently treating them as zero.

    # Determine which facility level/HCW combos are present in the data. 0 minutes added for missing combos across runs
    combinations = list(by_hsi[['Facility_Level', 'OfficerType']].drop_duplicates().itertuples(index=False, name=None))
    grid = pd.MultiIndex.from_tuples([(d, r, y, l, c) for d in draws for r in runs
                                      for y in years for l, c in combinations],
                                     names=['draw', 'run'] + KEYS)

    # The complete grid ensures a cadre appearing in only one arm is compared
    # with zero in the other arm, rather than lost in an inner alignment.
    # Combinations never seen anywhere are absent from this workload table.
    total = total.reindex(grid, fill_value=0).rename('scenario_minutes').reset_index()

    # Isolate minutes by cadre/level in the baseline scenario
    baseline = total.loc[total.draw.eq(baseline_draw)].drop(columns='draw').rename(
        columns={'scenario_minutes': 'baseline_minutes'})

    # Pair only within the same run, year, level and cadre. Baseline is not averaged across runs. Baseline-draw
    # differences are therefore exactly zero.
    differences = total.merge(baseline, on=['run'] + KEYS, validate='many_to_one')

    # Preserve signed differences: care redesign can reduce appointments o rshift work between cadres/levels. C
    # lipping would remove resource releases.
    differences['delta_minutes'] = differences.scenario_minutes - differences.baseline_minutes
    differences['delta_hours'] = differences.delta_minutes / 60
    summary = summarize_hcw_time_differences(differences, 'delta_minutes')

    return by_hsi, differences, summary

time_by_hsi, time_differences, time_summary = estimate_mnh_hcw_time(
    results_folder=results_folder,
    resourcefilepath=resourcefilepath,
    years=[2026],
    baseline_draw=0,
    scaling_factor=p_scaling_factor,
    # Alternatively, omit scaling_factor to use logged scaling factors.
)

# Next we estimate the cost of incremental differences in HCW time use
# Read in the available HCW time within the system
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

# Sum the available minutes per day by level and officer type
daily_capacity = (daily_capacity.loc[daily_capacity["Facility_Level"].ne("5")]
    .groupby(["Facility_Level", "Officer_Category"], as_index=False)[
        "Total_Mins_Per_Day"
    ]
    .sum()
    .rename(columns={"Officer_Category": "OfficerType"})
)

# Multiply daily capacity by days in the year to get annual capacity
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

# Select HRH costs
national_hr_costs = input_costs_df.loc[input_costs_df["cost_category"].eq("human resources for health")].copy()


def value_mnh_hcw_time(differences, national_hr_costs, annual_capacity,
                        *, baseline_draw=0, cost_subcategories=('salary_for_all_staff',)):
    """Value MNH minutes using baseline national HR costs / national capacity.

    national_hr_costs: UNSUMMARISED, UNDISCOUNTED output from
      estimate_input_cost_of_scenarios(cost_only_used_staff=False).
    annual_capacity: DataFrame with year, Facility_Level, OfficerType,
      capacity_minutes, optionally run. National baseline productive staff
      minutes available over that year, NOT minutes actually used by the cohort.
      It must match the workforce underlying national_hr_costs. If run absent,
      the same capacity is explicitly assumed for every run. Include year-specific
      workforce growth/productivity and leap years where applicable.
    cost_subcategories: salary only by default, avoiding implicit inclusion of
      training/mentorship that may be covered by an above-service adjustment.
    Returns per-run values. Keeps negative costs (released staff time).
    """

    # Select baseline national HR costs (limited currently to salary)
    hr = national_hr_costs.loc[
        national_hr_costs.draw.eq(baseline_draw)
        & national_hr_costs.cost_category.eq('human resources for health')
        & national_hr_costs.cost_subcategory.isin(cost_subcategories)].copy()

    # Check that the requested components exist somewhere in baseline output.
    missing = set(cost_subcategories) - set(hr.cost_subcategory)
    if missing:
        raise ValueError(f'Missing selected HR cost subcategories: {missing}')

    # Formatting
    hr = hr.rename(columns={'cost_subgroup': 'OfficerType'})
    hr['Facility_Level'] = hr.Facility_Level.astype(str)

    # Some cadre/level combinations do not exist (for example Clinical at level 0). Their source cost may be NaN
    # because no salary assumption is applicable.
    hr = (hr.groupby(['run'] + KEYS)['cost'].sum(min_count=1)
          .reset_index().rename(columns={'cost': 'baseline_national_hr_cost'}))

    # Capacity units MUST be staff-person minutes per year at national scale.
    # Include all the productive minutes corresponding to the selected workforce not only MNH minutes or currently
    # used minutes. Using baseline MNH time as denominator would allocate the whole national bill to MNH incorrectly.
    cap = annual_capacity.copy()
    cap['Facility_Level'] = cap.Facility_Level.astype(str)

    # A capacity table without run explicitly assumes the same capacity for every baseline run. Capacity is
    # intentionally not joined by scenario draw: both arms use the baseline national cost-per-minute schedule.
    join_keys = (['run'] if 'run' in cap else []) + KEYS
    if cap.duplicated(join_keys).any():
        raise ValueError('Duplicate annual capacity keys')

    # Left joins preserve every workload row so absent cost/capacity matches
    # remain visible for validation; an inner join could hide uncosted workload.
    result = differences.merge(hr, on=['run'] + KEYS, how='left', validate='many_to_one')
    result = result.merge(cap[join_keys + ['capacity_minutes']], on=join_keys,
                          how='left', validate='many_to_one')
    valid = (np.isfinite(result.baseline_national_hr_cost)
             & np.isfinite(result.capacity_minutes) & result.capacity_minutes.gt(0))

    # Active means time in EITHER arm, including unchanged time (delta=0).
    # Thus both absolute MNH costs must be costable even when their difference
    # is zero. A positive cost with zero capacity cannot define a unit price.
    active = result[['scenario_minutes', 'baseline_minutes']].ne(0).any(axis=1)
    if (active & ~valid).any():
        raise ValueError('Missing HR cost or positive capacity for active MNH resource combinations:\n'
                         + result.loc[active & ~valid, ['run'] + KEYS].to_string(index=False))

    # Combinations with no time in either arm can safely contribute zero.
    # Keep undefined unit prices as NaN on inactive rows for transparency,
    # while assigning those rows zero resource costs. Never replace missing
    # active costs with zero. np.where may evaluate both arithmetic branches;
    # any divide-by-zero warning for an inactive row is not a valid unit price.
    result['cost_per_minute'] = np.where(valid,
        result.baseline_national_hr_cost / result.capacity_minutes, np.nan)

    # Linear opportunity-cost allocation, not an automatic staffing expansion:
    # scenario/baseline costs describe the value of the included MNH time.

    # The signed difference is additional resource use, or released resources.
    # No recruitment thresholds, spare-capacity deduction or scenario-specific
    # salary changes are applied. Values remain undiscounted monetary amounts.

    result['incremental_hr_cost'] = np.where(active,
        result.delta_minutes * result.cost_per_minute, 0.0)

    result['scenario_mnh_hr_cost'] = np.where(active,
        result.scenario_minutes * result.cost_per_minute, 0.0)

    result['baseline_mnh_hr_cost'] = np.where(active,
        result.baseline_minutes * result.cost_per_minute, 0.0)

    return result

# Estimate cost of HCW time use
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
# Now we calculate the difference in costs between baseline and intervention scenarios
incremental_mnh_costs = (pd.DataFrame(
        find_difference_relative_to_comparison(
            total_mnh_costs,
            comparison=0,
            scaled=False)  # sets the comparator to draw 0 which is the Actual scenario
    ).T.iloc[0].unstack()).T

# --------------------------------------- 5. Add above service level cost ------------------------------------------------
# Finally, we adjust the incremental costs to include a 15% cost uplift in line with above service costs which would be
# associated with scale up
above_service_level_cost_assumption = 1.15
incremental_mnh_costs + incremental_mnh_costs.abs() * (above_service_level_cost_assumption - 1)
incremental_mnh_costs_summ = summarise_cost_data(incremental_mnh_costs, _interval='ci')

#  ------------------------------------- COST PER DALY/DEATH AVERTED --------------------------------------------------
#  Next we calculate the cost per DALYs averted across the estimate DALYs
cost = incremental_mnh_costs.stack().to_frame().T



#  --------------------------------------- MAXIMUM ABILITY TO PAY -----------------------------------------------------
#  ------------------------------------------ GENERATE TABLE ----------------------------------------------------------
