'''
Produce plots and estimates for the manuscript "Estimating the health gains and value for money of reducing drug stock-outs in Malawi: an individual-based modelling study"
'''

import datetime
import os
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Literal, Sequence, Optional, Union, List

from tlo import Date
from tlo.analysis.utils import (
    extract_params,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
)

# Define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# Print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# Create folders to store results
resourcefilepath = Path("./resources")
consumable_resourcefilepath = resourcefilepath / "healthsystem/consumables"
simulationfilepath = Path('./outputs/sakshi.mohan@york.ac.uk')
outputfilepath = Path('./outputs/consumables_impact_analysis')
if not os.path.exists(outputfilepath):
    os.makedirs(outputfilepath)

# Utility functions
# Assign item names to item codes
def assign_consumable_names_to_item_codes(df):
   # Create dictionary mapping item_codes to consumables names
    consumables_df = pd.read_csv(consumable_resourcefilepath / "ResourceFile_Consumables_Items_and_Packages.csv")[['Item_Code', 'Items']]
    consumables_df = consumables_df[consumables_df['Item_Code'].notna()]
    consumables_dict = dict(zip(consumables_df['Item_Code'], consumables_df['Items']))

    # Add consumable_name to df
    df = df.copy()
    df['item_name'] = df['item_code'].map(consumables_dict)

    return df

# Prepare availability data
def prepare_availability_dataset_for_plots(
    _df: pd.DataFrame,
    scenario_list: Optional[list[int]] = None,
    scenario_names_dict: Optional[dict[str, str]] = None,
    consumable_resourcefilepath: Path = None,
    resourcefilepath: Path = None
) -> pd.DataFrame:
    """
    Prepares a consumable availability dataset by merging facility and item category data,
    renaming columns for scenarios, and cleaning category names for plotting.
    """
    if scenario_list is None:
        scenario_list = []
    if scenario_names_dict is None:
        scenario_names_dict = {}

    # Load item category mapping
    program_item_mapping = pd.read_csv(
        consumable_resourcefilepath / 'ResourceFile_Consumables_Item_Designations.csv',
        usecols=['Item_Code', 'item_category']
    )
    program_item_mapping = program_item_mapping.rename(columns={'Item_Code': 'item_code'})
    program_item_mapping = program_item_mapping[program_item_mapping['item_category'].notna()]

    # Load facility list
    mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")

    # Merge facility and program info
    _df = _df.merge(
        mfl[['District', 'Facility_Level', 'Facility_ID']],
        on='Facility_ID', how='left'
    )
    _df = _df.merge(program_item_mapping, on='item_code', how='left')

    # Rename scenario columns
    _df = _df.rename(columns=scenario_names_dict)

    # Clean item category names
    clean_category_names = {
        'cancer': 'Cancer',
        'cardiometabolicdisorders': 'Cardiometabolic Disorders',
        'contraception': 'Contraception',
        'general': 'General',
        'hiv': 'HIV',
        'malaria': 'Malaria',
        'ncds': 'Non-communicable Diseases',
        'neonatal_health': 'Neonatal Health',
        'other_childhood_illnesses': 'Other Childhood Illnesses',
        'reproductive_health': 'Reproductive Health',
        'road_traffic_injuries': 'Road Traffic Injuries',
        'tb': 'Tuberculosis',
        'undernutrition': 'Undernutrition',
        'epi': 'Expanded programme on immunization'
    }
    _df['item_category'] = _df['item_category'].map(clean_category_names)

    return _df

# Wrap Labels
def wrap_labels(labels, width=15):
    """Wrap each label to the given character width."""
    return [textwrap.fill(str(lab), width) if lab is not None else "" for lab in labels]

# Generate heatmaps of average availability
def generate_heatmap(
    df: pd.DataFrame,
    include_levels: Optional[List[str]] = None,
    value_col: str = "Actual",
    row: str = "item_category",
    col: str = "Facility_Level",
    row_order: Optional[Sequence[str]] = None,
    col_order: Optional[Sequence[str]] = None,
    figurespath: Optional[Path] = None,
    filename: str = "heatmap_consumable_availability.png",
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "RdYlGn",
    annot: bool = True,
    fmt: Optional[str] = None,              # None -> auto choose
    font_scale: float = 0.9,
    cbar_label: str = "Proportion of days on which consumable is available",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    as_pct: bool = True,                   # format annotations as percentages
    round_decimals: int = 2,
    # option to plot scenarios on the x_axis
    scenario_axis: bool = False,  # if True, columns become scenarios
    scenario_cols: Optional[Sequence[str]] = None,
):
    """
    Build a heatmap either by a single column (e.g., Facility_Level) or across multiple
    scenario columns placed on the x-axis.
    """
    if include_levels is not None:
        df = df[df.Facility_Level.isin(include_levels)]
    if scenario_axis:
        aggregated = (
            df.groupby([row], dropna=True)[scenario_cols]
                         .mean()
                         .reset_index()
        )
        # Add perfect scenario
        aggregated['Perfect'] = 1  # Add a column representing the perfect scenario
        heatmap_df = aggregated.set_index('item_category')
    else:
        # Standard mode: columns = `col`, values = mean(value_col)
        aggregated = (
            df.groupby([row, col], dropna=True)[value_col]
            .mean()
            .reset_index()
        )
        heatmap_df = aggregated.pivot(index=row, columns=col, values=value_col)

    # Optional ordering
    if row_order is not None:
        heatmap_df = heatmap_df.reindex(row_order)
    if col_order is not None:
        heatmap_df = heatmap_df.reindex(columns=col_order)

    # 2) Totals (means across the raw data, not the pivot means)
    if scenario_axis:
        # Means by row across all programs for each scenario
        row_means = heatmap_df.mean(axis=0)  # average per scenario across programs
        avg_row = row_means.copy()
        heatmap_df.loc["Average"] = avg_row
    else:
        # Compute from raw df to avoid double-averaging
        col_means = df.groupby(row, dropna=False)[value_col].mean()
        row_means = df.groupby(col, dropna=False)[value_col].mean()
        overall_mean = df[value_col].mean()

        heatmap_df["Average"] = col_means.reindex(heatmap_df.index)
        avg_row = row_means.reindex(heatmap_df.columns).copy()
        avg_row.loc["Average"] = overall_mean
        heatmap_df.loc["Average"] = avg_row

    # 3) Annotation formatting
    if as_pct:
        # If values are 0–1 proportions, annotate as percentages
        display_df = (heatmap_df * 100).round(round_decimals)
        if fmt is None:
            fmt = ".0f" if round_decimals == 0 else f".{round_decimals}f"
        annot_kws = {"fmt": fmt}
        # Build string labels with % sign
        annot_data = display_df.astype(float)
    else:
        display_df = heatmap_df.round(round_decimals)
        if fmt is None:
            fmt = f".{round_decimals}f"
        annot_kws = {"fmt": fmt}
        annot_data = display_df.astype(float)

    # 4) Plot
    sns.set(font_scale=font_scale)
    fig, ax = plt.subplots(figsize=figsize)
    hm = sns.heatmap(
        annot_data,
        annot=annot,
        cmap=cmap,
        cbar_kws={"label": cbar_label},
        ax=ax,
        fmt=".2f"
    )

    # If percentage labels requested, overwrite text with % suffix
    if annot and as_pct:
        for t in ax.texts:
            t.set_text(f"{t.get_text()}%")

    # 5) Labels & ticks
    xlab = (xlabel or ("Scenario" if scenario_axis else col.replace("_", " ").title()))
    ylab = (ylabel or row.replace("_", " ").title())
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # 6) Save (optional)
    if figurespath is not None:
        figurespath = Path(figurespath)
        figurespath.mkdir(parents=True, exist_ok=True)
        outpath = figurespath / filename
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)
    return fig, ax, heatmap_df

# Generate LaTex-compatible detailed table of availability
def generate_detail_availability_table_by_scenario(
    df: pd.DataFrame,
    groupby_var: str,
    scenario_cols: Sequence[str],
    include_levels: Union[str, Sequence[str]],
    longtable: bool = False,
    outputpath: Path = None,
    decimals: int = 2,
    caption: Optional[str] = None,
    label_prefix: str = "tab:availability_by_",
    col_width_groupbyvar: str = "4cm",
    col_width_scenario: str = "1.8cm",
) -> str:
    """
    Create a LaTeX table of average availability (as percentages) for each consumable across scenarios,
    filtered to selected facility levels.

    Returns the LaTeX string and writes it to figurespath / f"availability_by_{groupby_var}.tex"
    """

    # --- Setup ---
    if outputpath is None:
        outputpath = outputfilepath / "appendix"  # falls back to your existing default
    outputpath.mkdir(parents=True, exist_ok=True)

    # Accept str OR list for include_levels
    if isinstance(include_levels, str):
        include_levels = [include_levels]

    table_df = df.copy()

    # Filter by facility level if the column is present
    if "Facility_Level" in table_df.columns:
        table_df = table_df[table_df["Facility_Level"].isin(include_levels)]

    # Aggregate means per item
    grouped = (
        table_df
        .groupby([groupby_var], dropna=False)[list(scenario_cols)]
        .mean()
        .reset_index()
    )

    # Rename first column to "Consumable"
    grouped.rename(columns={groupby_var: "Consumable"}, inplace=True)

    # Escape LaTeX in Consumable names
    def _latex_escape(s):
        if pd.isna(s):
            return ""
        s = str(s)
        # Order matters for backslashes; escape backslash first
        s = s.replace("\\", r"\\")
        s = s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
        s = s.replace("#", r"\#").replace("{", r"\{").replace("}", r"\}")
        s = s.replace("$", r"\$").replace("^", r"\^{}").replace("~", r"\~{}")
        return s

    grouped["Consumable"] = grouped["Consumable"].map(_latex_escape)

    # Convert proportions -> percentage strings with escaped %
    def pct_format(x: float) -> str:
        if pd.isna(x):
            return ""
        return f"{x * 100:.{decimals}f}\\%"

    for c in scenario_cols:
        grouped[c] = grouped[c].map(pct_format)

    # Build column format dynamically
    # First col wider for names, then one col per scenario
    column_format = (
        f"|R{{{col_width_groupbyvar}}}|"
        + "|".join([f"R{{{col_width_scenario}}}"] * len(scenario_cols))
        + "|"
    )

    # Caption/label
    if caption is None:
        caption = "Summarized availability by consumable"
    label = f"{label_prefix}{groupby_var}_{include_levels}"

    # Export to LaTeX (escape=False since we already escaped)
    latex_table = grouped.to_latex(
        longtable=longtable,
        column_format=column_format,
        caption=caption,
        label=label,
        position="h",
        index=False,
        escape=False,
        header=True,
    )

    # Add \hline after each row
    latex_table = latex_table.replace("\\\\\n", "\\\\ \\hline\n")

    # Save
    outpath = outputpath / f"availability_by_{groupby_var}_{include_levels}.tex"
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(latex_table)

    return latex_table


# Import and clean data files
#**********************************
# Import TLO model availability data
tlo_availability_df = pd.read_csv(consumable_resourcefilepath / "ResourceFile_Consumables_availability_small.csv")
scenario_names_dict={
        'available_prop': 'Actual',
        'available_prop_scenario1': 'Non-therapeutic consumables',
        'available_prop_scenario2': 'Vital medicines',
        'available_prop_scenario3': 'Pharmacist- managed',
        'available_prop_scenario4': 'Level 1b',
        'available_prop_scenario5': 'CHAM',
        'available_prop_scenario6': '75th percentile facility',
        'available_prop_scenario7': '90th percentile facility',
        'available_prop_scenario8': 'Best facility',
        'available_prop_scenario9': 'Best facility (including DHO)',
        'available_prop_scenario10': 'HIV supply chain',
        'available_prop_scenario11': 'EPI supply chain',
        'available_prop_scenario12': 'HIV moved to Govt supply chain (Avg by Level)',
        'available_prop_scenario13': 'HIV moved to Govt supply chain (Avg by Facility_ID)',
        'available_prop_scenario14': 'HIV moved to Govt supply chain (Avg by Facility_ID times 1.25)',
        'available_prop_scenario15': 'HIV moved to Govt supply chain (Avg by Facility_ID times 0.75)'
    }

tlo_availability_df = prepare_availability_dataset_for_plots(
    _df=tlo_availability_df,
    scenario_list=[1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15],
    scenario_names_dict=scenario_names_dict,
    consumable_resourcefilepath=consumable_resourcefilepath,
    resourcefilepath=resourcefilepath
)

# Generate figures for manuscript
#**********************************
# Figure 1: Average probability of consumable availability in public and CHAM health facilities in Malawi
_ = generate_heatmap(
    df=tlo_availability_df,
    value_col="Actual",
    row="item_category",
    col="Facility_Level",
    figurespath = outputfilepath / 'manuscript',
    filename="heatmap_program_and_level_actual.png",
    figsize=(10, 8),
    cmap="RdYlGn",
    round_decimals=4,
    cbar_label="Proportion of days on which consumable is available",
    xlabel="Facility Level",
    ylabel="Program",
)

# Figure 3: Comparison of consumable availability across modelled scenarios
scenario_cols = ['Actual', 'Non-therapeutic consumables', 'Vital medicines', 'Pharmacist- managed','75th percentile facility', '90th percentile facility', 'Best facility']
for level in ['1a', '1b']:
    _ = generate_heatmap(
        df=tlo_availability_df,
        include_levels = [level],
        value_col="Actual",
        row="item_category",
        col="Facility_Level",
        figurespath=outputfilepath / 'manuscript',
        filename=f"scenarios_heatmap_{level}.png",
        figsize=(10, 8),
        cmap="RdYlGn",
        round_decimals=4,
        cbar_label="Proportion of days on which consumable is available",
        xlabel="Facility Level",
        ylabel="Program",
        scenario_axis = True,  # if True, columns become scenarios
        scenario_cols = scenario_cols,
    )


# Figure A.1: Trend in average consumable availability by facility level

# Figure A.2: Comparison of consumable availability as per Open Logistics Management Information System (OpenLMIS), 2018 and Harmonised
# Health Facility Assessment, 2018-19

# Table B.1: Average probability of availability for each consumable under all scenarios (Level 1a)
tlo_availability_df = assign_consumable_names_to_item_codes(tlo_availability_df)

# Table B.1: Average probability of availability for each consumable under all scenarios (Level 1a)
availablity_by_item_1a = generate_detail_availability_table_by_scenario(
    df=tlo_availability_df,
    groupby_var="item_name",
    scenario_cols=scenario_cols,
    include_levels="1a",
    longtable=True,
    outputpath=outputfilepath / "appendix",
    decimals=2,
    caption="Average probability of availability for each consumable under all scenarios (Level 1a)",
)

# Table B.2: Average probability of availability for each consumable under all scenarios (Level 1b)
availablity_by_item_1b = generate_detail_availability_table_by_scenario(
    df=tlo_availability_df,
    groupby_var="item_name",
    scenario_cols=scenario_cols,
    include_levels="1b",
    longtable=True,
    outputpath=outputfilepath / "appendix",
    decimals=2,
    caption="Average probability of availability for each consumable under all scenarios (Level 1b)",
)
