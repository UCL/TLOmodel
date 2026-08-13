"""
Generate the five consumable-redistribution scenarios (national pooling, district pooling,
neighbourhood pooling, and pairwise exchange at 60- and 30-minute radii) and compile the
resulting availability probabilities into the format required by the TLO model.

The optimisation models, travel-time utilities, clustering heuristic, validation checks, and
plotting helpers live in `redistribution_utils.py`; this script handles data preparation,
orchestration, and the final merge into the TLO consumable-availability resource file.

Run `python redistribution_utils.py` (or call run_smoke_tests()) to execute the synthetic-data
tests for both models before regenerating scenarios.
"""
import calendar
import datetime
import pickle
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap

from scripts.costing.cost_estimation import clean_consumable_name

from scripts.data_file_processing.healthsystem.consumables.generating_consumable_scenarios.assess_amc_heterogeneity import (
        compare_pool_scarcity_across_levels)
from scripts.data_file_processing.healthsystem.consumables.generating_consumable_scenarios.redistribution_utils import (
        build_capacity_clusters_all,
        build_edges_within_radius_flat,
        build_time_matrices_by_district,
        do_violin_plot_change_in_p,
        generate_stock_adequacy_heatmap,
        plot_stockout_prevention,
        plot_stockout_prevention_by_month,
        prep_violin_df,
        redistribute_pooling_lp,
        redistribute_radius_lp,
        run_smoke_tests,
        validate_redistribution_output)

# define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# define folder pathways
outputfilepath = Path("./outputs/consumables_impact_analysis")
resourcefilepath = Path("./resources")
path_for_new_resourcefiles = resourcefilepath / "healthsystem/consumables"
# Set local shared drive source
path_to_share = Path(  # <-- point to the shared folder
    '/Users/sm2511/CloudStorage/OneDrive-SharedLibraries-ImperialCollegeLondon/TLOModel - WP - Documents/'
)


def generate_redistribution_scenarios(tlo_availability_df: pd.DataFrame,
                                      scenario_count: int,
                                      outputfilepath: Path = Path("./outputs/consumables_impact_analysis")) -> pd.DataFrame:
    # ----------------------------------------------------------------------------------------------------------------------
    # 1. Import and clean data files
    # ----------------------------------------------------------------------------------------------------------------------
    # Import Cleaned OpenLMIS data from 2018
    lmis = (pd.read_csv(outputfilepath / "ResourceFile_Consumables_availability_and_usage.csv")
            [['district', 'fac_type_tlo', 'fac_name', 'month', 'item_code', 'available_prop',
              'closing_bal', 'amc', 'dispensed', 'received']])

    # Drop duplicated facility, item, month combinations
    print(lmis.shape, "rows before collapsing duplicates")
    key_cols = ["district", "item_code", "fac_name", "month"]  # keys that define a unique record

    # helper to keep one facility level per group (mode -> most common; fallback to first non-null)
    def _mode_or_first(s: pd.Series):
        s = s.dropna()
        if s.empty:
            return np.nan
        m = s.mode()
        return m.iloc[0] if not m.empty else s.iloc[0]

    lmis = (
        lmis
        .groupby(key_cols, as_index=False)
        .agg(
            closing_bal=("closing_bal", "sum"),
            dispensed=("dispensed", "sum"),
            received=("received", "sum"),
            amc=("amc", "sum"),
            available_prop=("available_prop", "mean"),
            fac_type_tlo=("fac_type_tlo", _mode_or_first),
        )
    )

    print(lmis.shape, "rows after collapsing duplicates")

    # Import data on facility location
    location = (pd.read_excel(path_to_share / "07 - Data/Facility_GPS_Coordinates/gis_data_for_openlmis/LMISFacilityLocations_raw.xlsx")
                [['LMIS Facility List', 'LATITUDE', 'LONGITUDE']])
    # Find duplicates in facility names in the location dataset
    duplicates = location[location['LMIS Facility List'].duplicated(keep=False)]
    location = location.drop(duplicates[duplicates['LATITUDE'].isna()].index).reset_index(drop=True)  # Drop those duplicates where location is missing
    # Import ownership data
    ownership = (pd.read_csv(path_to_share / "07 - Data/Consumables data/OpenLMIS/lmis_facility_ownership.csv"))[['fac_name', 'fac_owner']]
    ownership = ownership.drop_duplicates(subset=['fac_name'])

    # Merge OpenLMIS and location and ownership data
    lmis = lmis.merge(location, left_on='fac_name', right_on='LMIS Facility List', how='left', validate='m:1')
    lmis = lmis.merge(ownership, on='fac_name', how='left', validate='m:1')
    lmis.rename(columns={'LATITUDE': 'lat', 'LONGITUDE': 'long', 'fac_type_tlo': 'Facility_Level'}, inplace=True)

    # Cleaning to match date to the same format as consumable availability RF in TLO model
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    lmis["month"] = lmis["month"].map(month_map)
    lmis["Facility_Level"] = lmis["Facility_Level"].str.replace("Facility_level_", "", regex=False)

    # Clean data types before analysis
    # 1) Normalize fac_name
    lmis["fac_name"] = (
        lmis["fac_name"]
        .astype("string")
        .str.normalize("NFKC")
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )

    # 2) Normalize other key columns used in grouping/joins
    lmis["item_code"] = lmis["item_code"].astype("string").str.strip()
    lmis["district"] = lmis["district"].astype("string").str.strip().str.replace(r"\s+", "_", regex=True)
    lmis["Facility_Level"] = lmis["Facility_Level"].astype("string").str.strip()

    # 3) Ensure numeric types (quietly coerce bad strings to NaN)
    lmis["amc"] = pd.to_numeric(lmis["amc"], errors="coerce")
    lmis["closing_bal"] = pd.to_numeric(lmis["closing_bal"], errors="coerce")

    # Keep only those facilities whose location is available
    old_facility_count = lmis.fac_name.nunique()
    lmis = lmis[lmis.lat.notna()]
    new_facility_count = lmis.fac_name.nunique()
    print(f"{old_facility_count - new_facility_count} facilities out of {old_facility_count} in the lmis data dropped due to "
          f"missing location information")

    def compute_opening_balance(df: pd.DataFrame) -> pd.Series:
        """
        Compute opening balance from same-month records.
        Formula: OB = closing_bal - received + dispensed  (equivalent to OB_m = CB_{m-1})
        Any negative OB values are replaced with 0.
        """
        ob = df["closing_bal"] - df["received"] + df["dispensed"]
        return ob.clip(lower=0)

    # Compute opening balance and reconcile with the reported availability:
    # where the mechanistic availability min(1, OB/AMC) is below the reported availability, adjust
    # OB upward so that redistribution never starts from an understated stock position.
    # (This adjustment is applied ONCE here; the redistribution functions treat `opening_bal` as given.)
    lmis["opening_bal"] = compute_opening_balance(lmis).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    amc_safe = np.maximum(1e-6, lmis["amc"].astype(float))
    lmis["p_mech"] = np.clip(lmis["opening_bal"] / amc_safe, 0.0, 1.0)
    mask_inconsistent = lmis["p_mech"] < lmis["available_prop"]
    lmis.loc[mask_inconsistent, "opening_bal"] = (
        lmis.loc[mask_inconsistent, "available_prop"] * lmis.loc[mask_inconsistent, "amc"]
    )
    print(f"Adjusted {mask_inconsistent.sum():,} rows "
          f"({mask_inconsistent.mean()*100:.2f}%) where recorded availability "
          f"exceeded mechanistic availability.")

    lmis.reset_index(inplace=True, drop=True)

    # ----------------------------------------------------------------------------------------------------------------------
    # 2. Travel-time matrices
    # ----------------------------------------------------------------------------------------------------------------------
    # T_car = build_time_matrices_by_district(
    #     lmis[['fac_name', 'district', 'lat', 'long']],
    #     mode="car",
    #     backend="osrm",
    #     osrm_base_url="https://router.project-osrm.org",
    #     max_chunk=50)
    # with open(outputfilepath / "T_car.pkl", "wb") as f:
    #     pickle.dump(T_car, f)
    # -> Commented out because it takes long to run. The result has been stored in pickle format.

    # Load pre-generated dictionary
    with open(outputfilepath / "T_car.pkl", "rb") as f:
        T_car = pickle.load(f)
    # T_car was created after cleaning fac names and getting rid of spaces in the text

    # T_car's district keys were generated from raw district names (with spaces, e.g.
    # 'Mzimba North', 'Nkhata bay', 'Nkhota Kota'), but `lmis["district"]` above has already
    # had whitespace collapsed to underscores (e.g. 'Mzimba_North'). Without reconciling the
    # two, every lookup of T_car by district (in redistribute_radius_lp's per-district travel
    # matrix selection, and in build_capacity_clusters_all's cluster ids) silently misses for
    # any multi-word district, so those districts get zero pairwise redistribution and
    # degenerate singleton clusters -- not because their facilities are actually far apart.
    T_car = {
        re.sub(r"\s+", "_", str(d).strip()): T for d, T in T_car.items()
    }

    # ----------------------------------------------------------------------------------------------------------------------
    # 3. Exploration: stock adequacy and connectivity
    # ----------------------------------------------------------------------------------------------------------------------
    # Attach clean item names to lmis
    consumables_dict = \
        pd.read_csv(resourcefilepath / 'healthsystem' / 'consumables' / 'ResourceFile_Consumables_Items_and_Packages.csv',
                    low_memory=False,
                    encoding="ISO-8859-1")[['Items', 'Item_Code']]
    consumables_dict = dict(zip(consumables_dict['Item_Code'], consumables_dict['Items']))
    lmis['item_name'] = pd.to_numeric(lmis["item_code"], errors="coerce").map(consumables_dict)
    lmis['item_name'] = (
        lmis['item_name']
        .astype(str)
        .apply(clean_consumable_name)
    )

    # Plot stock adequacy by district and month to assess what bounds to set when pooling
    empty_cell_note = "Note: Grey cells in the heatmap indicate missing data."
    generate_stock_adequacy_heatmap(df=lmis, figures_path=outputfilepath,
                                    y_var='district', value_var='item_name',
                                    value_label="% of consumables with Opening Balance ≥ 3 × AMC",
                                    amc_threshold=3, compare="ge",
                                    filename="mth_district_stock_adequacy_3amc.png", figsize=(12, 10))
    generate_stock_adequacy_heatmap(df=lmis, figures_path=outputfilepath,
                                    y_var='district', value_var='item_name',
                                    value_label="% of consumables with Opening Balance ≥ 1.5 × AMC",
                                    amc_threshold=1.5, compare="ge",
                                    filename="mth_district_stock_adequacy_1.5amc.png", figsize=(12, 10))
    generate_stock_adequacy_heatmap(df=lmis, figures_path=outputfilepath,
                                    y_var='district', value_var='item_name',
                                    value_label="% of consumables with Opening Balance <= 1 × AMC",
                                    amc_threshold=1, compare="le",
                                    filename="mth_district_stock_inadequacy_1amc.png", figsize=(12, 10))
    generate_stock_adequacy_heatmap(df=lmis, figures_path=outputfilepath,
                                    y_var='item_name', value_var='fac_name',
                                    value_label="% of facilities with Opening Balance ≥ 3 × AMC",
                                    amc_threshold=3, compare="ge", footnote=empty_cell_note,
                                    filename="mth_item_stock_adequacy_3amc.png")
    generate_stock_adequacy_heatmap(df=lmis, figures_path=outputfilepath,
                                    y_var='item_name', value_var='fac_name',
                                    value_label="% of facilities with Opening Balance ≥ 1.5 × AMC",
                                    amc_threshold=1.5, compare="ge", footnote=empty_cell_note,
                                    filename="mth_item_stock_adequacy_1.5amc.png")
    generate_stock_adequacy_heatmap(df=lmis, figures_path=outputfilepath,
                                    y_var='item_name', value_var='fac_name',
                                    value_label="% of facilities with Opening Balance <= 1 × AMC",
                                    amc_threshold=1, compare="le", footnote=empty_cell_note,
                                    filename="mth_item_stock_inadequacy_1amc.png")

    # Browse the number of eligible neighbours depending on allowable travel time
    results = []
    for mins in [30, 60, 90, 120]:
        edges_flat = build_edges_within_radius_flat(T_car, max_minutes=mins)
        neighbors_count = pd.Series({fac: len(neigh) for fac, neigh in edges_flat.items()})
        results.append({"radius": mins, "mean": neighbors_count.mean(),
                        "ci95": 1.96 * neighbors_count.sem()})
    results_df = pd.DataFrame(results)

    plt.figure(figsize=(6, 4))
    plt.bar(results_df["radius"], results_df["mean"], yerr=results_df["ci95"], capsize=5, color="skyblue")
    plt.xlabel("Travel time radius (minutes)")
    plt.ylabel("Average number of facilities within radius")
    plt.title("Average connectivity of facilities with 95% CI")
    plt.xticks(results_df["radius"])
    plt.savefig(outputfilepath / "neighbour_count_by_max_travel_time")
    plt.close()
    # A manual check shows that for distances greater than 60 minutes ORS underestimates the travel
    # time a little. Ideally google API could have been used but this is not open source so I have retained OSRM.

    # ----------------------------------------------------------------------------------------------------------------------
    # 4. Run the redistribution models
    # ----------------------------------------------------------------------------------------------------------------------
    # Verify both optimisation models on synthetic data before running/reusing real scenarios
    run_smoke_tests()

    # Build clusters (size = 3) from per-district travel-time matrices for neighbourhood pooling
    cluster_size = 3
    cluster_series = build_capacity_clusters_all(T_car, cluster_size=cluster_size)
    # cluster_series is a pd.Series with a (district, fac_name) MultiIndex, value like "District A#C00", ...

    # How often is there actually enough donor surplus to cover eligible facilities' deficits,
    # and how does that change with the size of the pool being coordinated (national vs. district
    # vs. neighbourhood)? Prints a sufficiency summary per level and saves a single comparison
    # figure (analogous to the coverage-ratio panel of assess_pool_scarcity's figure).
    compare_pool_scarcity_across_levels(lmis, cluster_map=cluster_series, figures_path=outputfilepath)

    # a0) National-level pooling
    '''
    # Commented out for quicker runs
    print("Now running Pooled Redistribution at National level")
    start = time.time()
    pooled_national_df, pool_national_moves = redistribute_pooling_lp(
        df=lmis,
        tau_max=3.0,
        tau_donor_keep=1.5,
        pooling_level="national",
        cluster_map=None,
        return_move_log=True,
        floor_to_baseline=True,
    )
    print(pooled_national_df.groupby('Facility_Level')[['available_prop_redis', 'available_prop']].mean())
    pooled_national_df[['district', 'item_code', 'fac_name', 'month', 'amc', 'available_prop', 'Facility_Level',
                        'OB', 'OB_prime', 'available_prop_redis', 'received_from_pool']].to_csv(
                        outputfilepath / 'clustering_national_df.csv', index=False)
    end = time.time()
    print(f"National redistribution completed in {end - start:.3f} seconds")
    # 234.276 seconds
    '''
    pooled_national_df = pd.read_csv(outputfilepath / 'clustering_national_df.csv')
    validate_redistribution_output(pooled_national_df, "National pooling",
                                   group_cols=("month", "item_code"), conservation="leq", strict=False)
    tlo_pooled_national = (
        pooled_national_df
        .groupby(["item_code", "district", "Facility_Level", "month"], as_index=False)
        .agg(available_prop_scenario20=("available_prop_redis", "mean"))
        .sort_values(["item_code", "district", "Facility_Level", "month"])
    )

    # a) District-level pooling
    '''
    # Commented out for quicker runs
    print("Now running Pooled Redistribution at District level")
    start = time.time()
    pooled_district_df, pool_district_moves = redistribute_pooling_lp(
        df=lmis,
        tau_max=3.0,
        tau_donor_keep=1.5,
        pooling_level="district",
        cluster_map=None,
        return_move_log=True,
        floor_to_baseline=True,
    )
    print(pooled_district_df.groupby('Facility_Level')[['available_prop_redis', 'available_prop']].mean())
    pooled_district_df[['district', 'item_code', 'fac_name', 'month', 'amc', 'available_prop', 'Facility_Level',
                        'OB', 'OB_prime', 'available_prop_redis', 'received_from_pool']].to_csv(
                        outputfilepath / 'clustering_district_df.csv', index=False)
    end = time.time()
    print(f"District redistribution completed in {end - start:.3f} seconds")
    # 4972.169 seconds
    '''
    pooled_district_df = pd.read_csv(outputfilepath / 'clustering_district_df.csv')
    validate_redistribution_output(pooled_district_df, "District pooling",
                                   conservation="leq", strict=False)
    tlo_pooled_district = (
        pooled_district_df
        .groupby(["item_code", "district", "Facility_Level", "month"], as_index=False)
        .agg(available_prop_scenario16=("available_prop_redis", "mean"))
        .sort_values(["item_code", "district", "Facility_Level", "month"])
    )

    # b) Neighbourhood pooling (clusters of 3)
    '''
    # Commented out for quicker runs
    print("Now running pooled redistribution at Cluster (Size = 3) level")
    start = time.time()
    pooled_cluster_df, pool_cluster_moves = redistribute_pooling_lp(
        df=lmis,
        tau_max=3.0,
        tau_donor_keep=1.5,
        pooling_level="cluster",
        cluster_map=cluster_series,
        return_move_log=True,
        floor_to_baseline=True,
    )
    print(pooled_cluster_df.groupby('Facility_Level')[['available_prop_redis', 'available_prop']].mean())
    pooled_cluster_df[['district', 'item_code', 'fac_name', 'month', 'amc', 'available_prop', 'Facility_Level',
                       'OB', 'OB_prime', 'available_prop_redis', 'received_from_pool']].to_csv(
                       outputfilepath / 'clustering_n3_df.csv', index=False)
    end = time.time()
    print(f"Cluster redistribution completed in {end - start:.3f} seconds")
    #22414.642 seconds
    '''
    pooled_cluster_df = pd.read_csv(outputfilepath / 'clustering_n3_df.csv')
    validate_redistribution_output(pooled_cluster_df, "Neighbourhood pooling",
                                   conservation="leq", strict=False)
    tlo_pooled_cluster = (
        pooled_cluster_df
        .groupby(["item_code", "district", "Facility_Level", "month"], as_index=False)
        .agg(available_prop_scenario17=("available_prop_redis", "mean"))
        .sort_values(["item_code", "district", "Facility_Level", "month"])
    )

    # c) Pairwise redistribution, 60-minute radius
    '''
    # Commented out for quicker runs
    print("Now running pairwise redistribution with maximum radius 60 minutes")
    start = time.time()
    large_radius_df, large_radius_moves = redistribute_radius_lp(
        df=lmis_test,
        time_matrix=T_car,
        radius_minutes=60,             # facilities within 1 hour by car
        tau_keep=1.5,                  # donor must keep 1.5 x AMC
        tau_tar=1.0,                   # receivers target 1 x AMC
        # K_in/K_out left at their default (None = uncapped): the travel-time radius already
        # bounds the candidate neighbour set and Qmin_proportion already screens out shipments
        # too small to be worth dispatching, so no separate cap on exchange counts is imposed.
        Qmin_proportion=0.25,          # min lot = one week of receiver demand
        eligible_levels=("1a", "1b"),  # only 1a/1b can receive
    )
    print(large_radius_df.groupby('Facility_Level')[['available_prop_redis', 'available_prop']].mean())
    large_radius_df.to_csv(outputfilepath / 'large_radius_df.csv', index=False)
    end = time.time()
    print(f"Large radius exchange distribution completed in {end - start:.3f} seconds")
    '''
    large_radius_df = pd.read_csv(outputfilepath / 'large_radius_df.csv')
    validate_redistribution_output(large_radius_df, "Pairwise exchange (60-min radius)",
                                   tau_max=None, conservation="exact", strict=False)
    tlo_large_radius = (
        large_radius_df
        .groupby(["item_code", "district", "Facility_Level", "month"], as_index=False)
        .agg(available_prop_scenario18=("available_prop_redis", "mean"))
        .sort_values(["item_code", "district", "Facility_Level", "month"])
    )

    # d) Pairwise redistribution, 30-minute radius
    '''
    print("Now running pairwise redistribution with maximum radius 30 minutes")
    start = time.time()
    small_radius_df, small_radius_moves = redistribute_radius_lp(
        df=lmis,
        time_matrix=T_car,
        radius_minutes=30,             # facilities within 30 minutes by car
        tau_keep=1.5,
        tau_tar=1.0,
        # K_in/K_out left at their default (None = uncapped) -- see note above.
        Qmin_proportion=0.25,
        eligible_levels=("1a", "1b"),
    )
    print(small_radius_df.groupby('Facility_Level')[['available_prop_redis', 'available_prop']].mean())
    small_radius_df.to_csv(outputfilepath / 'small_radius_df.csv', index=False)
    end = time.time()
    print(f"Small radius exchange redistribution completed in {end - start:.3f} seconds")
    '''
    small_radius_df = pd.read_csv(outputfilepath / 'small_radius_df.csv')
    validate_redistribution_output(small_radius_df, "Pairwise exchange (30-min radius)",
                                   tau_max=None, conservation="exact", strict=False)
    tlo_small_radius = (
        small_radius_df
        .groupby(["item_code", "district", "Facility_Level", "month"], as_index=False)
        .agg(available_prop_scenario19=("available_prop_redis", "mean"))
        .sort_values(["item_code", "district", "Facility_Level", "month"])
    )

    # ----------------------------------------------------------------------------------------------------------------------
    # 5. Summarise the outcomes of redistribution
    # ----------------------------------------------------------------------------------------------------------------------
    scenario_dfs = {
        "Neighbourhood pooling": pooled_cluster_df,
        "District pooling": pooled_district_df,
        "National pooling": pooled_national_df,
        "Pairwise exchange (Small radius)": small_radius_df,
        "Pairwise exchange (Large radius)": large_radius_df,
    }

    # How much of the anticipated stock-out risk was averted? (headline appendix figures)
    prevention_summary = plot_stockout_prevention(scenario_dfs, figures_path=outputfilepath)
    prevention_summary.to_csv(outputfilepath / "stockout_risk_averted_summary.csv", index=False)
    plot_stockout_prevention_by_month(scenario_dfs, figures_path=outputfilepath)

    # Violin plots of the distribution of changes in availability
    violin_scenario_order = ["Neighbourhood pooling", "District pooling", "National pooling",
                             "Pairwise exchange (Small radius)", "Pairwise exchange (Large radius)"]

    violin_df_all_facs = pd.concat([
        prep_violin_df(pooled_national_df, "National pooling", keep_facs_with_no_change=True),
        prep_violin_df(pooled_district_df, "District pooling", keep_facs_with_no_change=True),
        prep_violin_df(pooled_cluster_df, "Neighbourhood pooling", keep_facs_with_no_change=True),
        prep_violin_df(large_radius_df, "Pairwise exchange (Large radius)", keep_facs_with_no_change=True),
        prep_violin_df(small_radius_df, "Pairwise exchange (Small radius)", keep_facs_with_no_change=True),
    ], ignore_index=True)
    violin_df_only_facs_with_change = pd.concat([
        prep_violin_df(pooled_national_df, "National pooling", keep_facs_with_no_change=False),
        prep_violin_df(pooled_district_df, "District pooling", keep_facs_with_no_change=False),
        prep_violin_df(pooled_cluster_df, "Neighbourhood pooling", keep_facs_with_no_change=False),
        prep_violin_df(large_radius_df, "Pairwise exchange (Large radius)", keep_facs_with_no_change=False),
        prep_violin_df(small_radius_df, "Pairwise exchange (Small radius)", keep_facs_with_no_change=False),
    ], ignore_index=True)

    # Fix the display order (national, district, cluster, 60-min, 30-min) regardless of
    # concatenation order, since seaborn otherwise sorts string categories alphabetically.
    violin_df_all_facs["scenario"] = pd.Categorical(violin_df_all_facs["scenario"],
                                                     categories=violin_scenario_order, ordered=True)
    violin_df_only_facs_with_change["scenario"] = pd.Categorical(
        violin_df_only_facs_with_change["scenario"], categories=violin_scenario_order, ordered=True)

    do_violin_plot_change_in_p(violin_df=violin_df_all_facs,
                               figname="violin_redistribution_national_all_facs.png",
                               figures_path=outputfilepath, legend_location="upper right")
    do_violin_plot_change_in_p(violin_df=violin_df_only_facs_with_change,
                               figname="violin_redistribution_national_only_facs_with_change.png",
                               figures_path=outputfilepath, legend_location="lower right")
    do_violin_plot_change_in_p(violin_df=violin_df_all_facs,
                               figname="violin_by_district_all_facs",
                               figures_path=outputfilepath, by_district=True, ncol=4)
    do_violin_plot_change_in_p(violin_df=violin_df_only_facs_with_change,
                               figname="violin_by_district_only_facs_with_change",
                               figures_path=outputfilepath, by_district=True, ncol=4)

    # ----------------------------------------------------------------------------------------------------------------------
    # 6. Compile updated probabilities and merge with Resourcefile
    # ----------------------------------------------------------------------------------------------------------------------
    tlo_redis = reduce(
        lambda left, right: pd.merge(
            left, right,
            on=["item_code", "district", "Facility_Level", "month"],
            how="outer"
        ),
        [tlo_pooled_national, tlo_pooled_district, tlo_pooled_cluster, tlo_large_radius, tlo_small_radius]
    )

    tlo_redis.to_csv(outputfilepath / 'tlo_redis.csv', index=False)

    # Edit new dataframe to match mfl formatting
    # NOTE: scenario16-19 are the pre-existing district/cluster/60-min/30-min redistribution
    # scenario numbers already referenced elsewhere (e.g. consumables_availability_estimation.py)
    # and are kept unchanged; national pooling is appended as scenario20 rather than
    # renumbering the existing four, so that resource file is not broken by this addition.
    list_of_new_scenario_variables = ['available_prop_scenario16',
                                      'available_prop_scenario17', 'available_prop_scenario18',
                                      'available_prop_scenario19', 'available_prop_scenario20']
    tlo_redis = tlo_redis[['item_code', 'month', 'district', 'Facility_Level'] + list_of_new_scenario_variables].dropna()
    tlo_redis["item_code"] = tlo_redis["item_code"].astype(float).astype(int)

    # Load master facility list
    mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
    mfl["District"] = mfl["District"].astype("string").str.strip().str.replace(r"\s+", "_", regex=True)
    districts = set(mfl[mfl.District.notna()]["District"].unique())
    kch = (mfl.Region == 'Central') & (mfl.Facility_Level == '3')
    qech = (mfl.Region == 'Southern') & (mfl.Facility_Level == '3')
    mch = (mfl.Region == 'Northern') & (mfl.Facility_Level == '3')
    zmh = mfl.Facility_Level == '4'
    mfl.loc[kch, "District"] = "Lilongwe"
    mfl.loc[qech, "District"] = "Blantyre"
    mfl.loc[mch, "District"] = "Mzimba"
    mfl.loc[zmh, "District"] = "Zomba"

    # Do some mapping to make the Districts line-up with the definition of Districts in the model
    rename_and_collapse_to_model_districts = {
        'Nkhota_Kota': 'Nkhotakota',
        'Mzimba_South': 'Mzimba',
        'Mzimba_North': 'Mzimba',
        'Nkhata_bay': 'Nkhata_Bay',
    }

    tlo_redis['district_std'] = tlo_redis['district'].replace(rename_and_collapse_to_model_districts)
    # Take averages (now that 'Mzimba' is mapped-to by both 'Mzimba South' and 'Mzimba North'.)
    tlo_redis = tlo_redis.groupby(by=['district_std', 'Facility_Level', 'month', 'item_code'])[list_of_new_scenario_variables].mean().reset_index()

    # Fill in missing data:
    # 1) Cities to get same results as their respective regions
    copy_source_to_destination = {
        'Mzimba': 'Mzuzu_City',
        'Lilongwe': 'Lilongwe_City',
        'Zomba': 'Zomba_City',
        'Blantyre': 'Blantyre_City'
    }

    for source, destination in copy_source_to_destination.items():
        new_rows = tlo_redis.loc[(tlo_redis.district_std == source) & (tlo_redis.Facility_Level.isin(['1a', '1b', '2']))].copy()
        new_rows.district_std = destination
        tlo_redis = pd.concat([tlo_redis, new_rows], axis=0, ignore_index=True)

    # 2) Fill in Likoma (for which no data) with the means
    means = tlo_redis.loc[tlo_redis.Facility_Level.isin(['1a', '1b', '2'])].groupby(by=['Facility_Level', 'month', 'item_code'])[
        list_of_new_scenario_variables].mean().reset_index()
    new_rows = means.copy()
    new_rows['district_std'] = 'Likoma'
    tlo_redis = pd.concat([tlo_redis, new_rows], axis=0, ignore_index=True)
    assert sorted(set(districts)) == sorted(set(pd.unique(tlo_redis.district_std)))

    # 3) copy the results for 'Mwanza/1b' to be equal to 'Mwanza/1a'.
    mwanza_1b = tlo_redis.loc[(tlo_redis.district_std == 'Mwanza') & (tlo_redis.Facility_Level == '1a')].copy().assign(Facility_Level='1b')
    tlo_redis = pd.concat([tlo_redis, mwanza_1b], axis=0, ignore_index=True)

    # 4) Copy all the results to create a level 0 with an availability equal to half that in the respective 1a
    all_0 = tlo_redis.loc[tlo_redis.Facility_Level == '1a'].copy().assign(Facility_Level='0')
    all_0[list_of_new_scenario_variables] *= 0.5
    tlo_redis = pd.concat([tlo_redis, all_0], axis=0, ignore_index=True)

    # Now, merge-in facility_id
    tlo_redis = tlo_redis.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                                left_on=['district_std', 'Facility_Level'],
                                right_on=['District', 'Facility_Level'], how='left', indicator=True, validate='m:1')
    tlo_redis = tlo_redis[tlo_redis.Facility_ID.notna()].rename(columns={'district_std': 'district'})
    assert sorted(set(mfl.loc[mfl.Facility_Level != '5', 'Facility_ID'].unique())) == sorted(set(pd.unique(tlo_redis.Facility_ID)))

    # Load original availability dataframe
    # ----------------------------------------------------------------------------------------------------------------------
    list_of_old_scenario_variables = [f"available_prop_scenario{i}" for i in range(1, scenario_count + 1)]
    tlo_availability_df = tlo_availability_df[['Facility_ID', 'month', 'item_code', 'available_prop'] + list_of_old_scenario_variables]

    # Attach district, facility level and item_category to this dataset
    program_item_mapping = pd.read_csv(path_for_new_resourcefiles / 'ResourceFile_Consumables_Item_Designations.csv')[['Item_Code', 'item_category']]
    program_item_mapping = program_item_mapping.rename(columns={'Item_Code': 'item_code'})[program_item_mapping.item_category.notna()]
    tlo_availability_df = tlo_availability_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                                                    on=['Facility_ID'], how='left').rename(columns={'District': 'district'})
    tlo_availability_df = tlo_availability_df.merge(program_item_mapping, on=['item_code'], how='left')

    # Because some of the availability data in the original availability comes from data sources other than OpenLMIS, there are
    # more unique item codes in tlo_availability_df than in tlo_redis. For these items, assume that the proportion of 'uplift'
    # is the same as the average 'uplift' experienced across the consumables in tlo_redis disaggregated by district,
    # facility level, and month.

    # First fix any unexpected changes in availability probability
    # Merge the old and new dataframe
    redis_levels = ['1a', '1b']
    tlo_redis = tlo_redis[tlo_redis.Facility_Level.isin(redis_levels)]

    tlo_redis = tlo_redis.merge(
        tlo_availability_df[["district", "Facility_Level", "item_code", "month", "available_prop"]],
        on=["district", "Facility_Level", "item_code", "month"],
        how="left",
        validate="one_to_one"
    )

    for redis_scenario_col in list_of_new_scenario_variables:
        pre = (tlo_redis[redis_scenario_col] < tlo_redis["available_prop"]).mean()
        print(f"Pre-fix {redis_scenario_col}: {pre:.3%}")

        # Enforce no-harm
        tlo_redis[redis_scenario_col] = np.maximum(
            tlo_redis[redis_scenario_col],
            tlo_redis["available_prop"]
        )

        post = (tlo_redis[redis_scenario_col] < tlo_redis["available_prop"]).mean()
        print(f"Post-fix {redis_scenario_col}: {post:.3%}")

    # Next create an uplift dataframe
    modelled_items = tlo_redis["item_code"].unique()
    # Compute uplift once per scenario, store in a dict
    uplift_maps = {}

    for scenario_col in list_of_new_scenario_variables:
        uplift_maps[scenario_col] = (
            tlo_redis.assign(
                uplift=lambda x: np.where(
                    x["available_prop"] > 0,
                    x[scenario_col] / x["available_prop"],
                    np.nan
                )
            )
            .groupby(["district", "Facility_Level", "month"], as_index=False)["uplift"]
            .mean()
            .rename(columns={"uplift": f"uplift_{scenario_col}"})
        )

    # Get baseline rows for missing items
    missing_mask = ~tlo_availability_df["item_code"].isin(modelled_items)

    df_missing = (
        tlo_availability_df[
            (tlo_availability_df["Facility_Level"].isin(redis_levels)) &
            missing_mask
        ]
        .copy()
    )

    # Merge all uplifts horizontally
    for scenario_col, uplift_df in uplift_maps.items():
        df_missing = df_missing.merge(
            uplift_df,
            on=["district", "Facility_Level", "month"],
            how="left"
        )
        df_missing[scenario_col] = np.minimum(
            1.0,
            df_missing["available_prop"] * df_missing[f"uplift_{scenario_col}"]
        )
        df_missing.drop(columns=[f"uplift_{scenario_col}"], inplace=True)

    # Concatenate
    tlo_redis = pd.concat([tlo_redis, df_missing], ignore_index=True)

    dupes = tlo_redis.duplicated(["district", "Facility_Level", "item_code", "month"])
    assert (dupes.sum() == 0)

    for scenario_col in list_of_new_scenario_variables:
        assert ((tlo_redis[scenario_col] < tlo_redis["available_prop"]).sum()) == 0

    tlo_redis = tlo_redis[['Facility_ID', 'month', 'item_code'] + list_of_new_scenario_variables]

    # Interpolate missing values in tlo_redis for all levels except 0
    # ----------------------------------------------------------------------------------------------------------------------
    # Generate the dataframe that has the desired size and shape
    fac_ids = set(mfl.loc[mfl.Facility_Level.isin(redis_levels)].Facility_ID)
    item_codes = set(tlo_availability_df.item_code.unique())
    months = range(1, 13)

    # Create a MultiIndex from the product of fac_ids, months, and item_codes
    index = pd.MultiIndex.from_product([fac_ids, months, item_codes], names=['Facility_ID', 'month', 'item_code'])

    # Initialize a DataFrame with the MultiIndex and columns, filled with NaN
    full_set = pd.DataFrame(index=index, columns=list_of_new_scenario_variables)
    full_set = full_set.astype(float)

    # Insert the data, where it is available.
    full_set = full_set.combine_first(tlo_redis.set_index(['Facility_ID', 'month', 'item_code'])[list_of_new_scenario_variables])

    # Fill in the blanks with rules for interpolation.
    facilities_by_level = defaultdict(set)
    for ix, row in mfl.iterrows():
        facilities_by_level[row['Facility_Level']].add(row['Facility_ID'])

    items_by_category = defaultdict(set)
    for ix, row in program_item_mapping.iterrows():
        items_by_category[row['item_category']].add(row['item_code'])

    def get_other_facilities_of_same_level(_fac_id):
        """Return a set of facility_id for other facilities that are of the same level as that provided."""
        for v in facilities_by_level.values():
            if _fac_id in v:
                return v - {_fac_id}

    def get_other_items_of_same_category(_item_code):
        """Return a set of item_codes for other items that are in the same category/program as that provided."""
        for v in items_by_category.values():
            if _item_code in v:
                return v - {_item_code}

    def interpolate_missing_with_mean(_ser):
        """Return a series in which any values that are null are replaced with the mean of the non-missing."""
        if pd.isnull(_ser).all():
            raise ValueError
        return _ser.fillna(_ser.mean())

    # Create new dataset that includes the interpolations (not done "in place", because the logic is
    # based on what results are missing before the interpolations in other facilities).
    full_set_interpolated = full_set * np.nan
    full_set_interpolated[list_of_new_scenario_variables] = full_set[list_of_new_scenario_variables]

    for fac in fac_ids:
        for item in item_codes:
            for col in list_of_new_scenario_variables:
                print(f"Now doing: fac={fac}, item={item}, column={col}")

                # Get records of the availability of this item in this facility.
                _monthly_records = full_set.loc[(fac, slice(None), item), col].copy()

                if pd.notnull(_monthly_records).any():
                    # If there is at least one record of this item at this facility, then interpolate the missing months
                    # from the months for which there are data on this item in this facility.
                    _monthly_records = interpolate_missing_with_mean(_monthly_records)

                else:
                    # If there is no record of this item at this facility, check other facilities of the same level;
                    # or, failing that, other items of the same category at this facility.
                    facilities = list(get_other_facilities_of_same_level(fac))

                    other_items = get_other_items_of_same_category(item)
                    items = list(other_items) if other_items else other_items

                    recorded_at_other_facilities_of_same_level = pd.notnull(
                        full_set.loc[(facilities, slice(None), item), col]
                    ).any()

                    if not items:
                        category_recorded_at_other_facilities_of_same_level = False
                    else:
                        # Filter only items that exist in the MultiIndex at this facility
                        valid_items = [
                            itm for itm in items
                            if any((fac, m, itm) in full_set.index for m in months)
                        ]

                        category_recorded_at_other_facilities_of_same_level = pd.notnull(
                            full_set.loc[(fac, slice(None), valid_items), col]
                        ).any()

                    if recorded_at_other_facilities_of_same_level:
                        # Use the average availability of the item at other facilities of the same level.
                        print("Data for facility ", fac, " extrapolated from other facilities within level - ", facilities)
                        facilities = list(get_other_facilities_of_same_level(fac))
                        _monthly_records = interpolate_missing_with_mean(
                            full_set.loc[(facilities, slice(None), item), col].groupby(level=1).mean()
                        )

                    elif category_recorded_at_other_facilities_of_same_level and valid_items:
                        # Use the average availability of other items of the same category at this facility.
                        print("Data for item ", item, " extrapolated from other items within category - ", valid_items)
                        _monthly_records = interpolate_missing_with_mean(
                            full_set.loc[(fac, slice(None), valid_items), col].groupby(level=1).mean()
                        )

                    else:
                        # If nothing worked, assume no change.
                        print("No interpolation worked")
                        _monthly_records = _monthly_records.fillna(1.0)

                # Insert values (including corrections) into the resulting dataset.
                full_set_interpolated.loc[(fac, slice(None), item), col] = _monthly_records.values
                assert full_set_interpolated.loc[(fac, slice(None), item), col].mean() >= 0

    # Check that there are no missing values
    assert not pd.isnull(full_set_interpolated).any().any()

    full_set_interpolated = full_set_interpolated.reset_index()

    # Add to this dataset original availability for all the other levels of care
    base_other_levels = tlo_availability_df[
        ~tlo_availability_df["Facility_Level"].isin(redis_levels)
    ].copy()
    for col in list_of_new_scenario_variables:
        base_other_levels[col] = base_other_levels["available_prop"]
    base_other_levels = base_other_levels[['Facility_ID', 'month', 'item_code'] + list_of_new_scenario_variables]
    tlo_redis_final = pd.concat(
        [full_set_interpolated, base_other_levels],
        ignore_index=True,
    )

    # Verify that the shape of this dataframe is identical to the original availability dataframe
    assert sorted(set(tlo_redis_final.Facility_ID)) == sorted(set(pd.unique(tlo_availability_df.Facility_ID)))
    assert sorted(set(tlo_redis_final.month)) == sorted(set(pd.unique(tlo_availability_df.month)))
    assert sorted(set(tlo_redis_final.item_code)) == sorted(set(pd.unique(tlo_availability_df.item_code)))
    assert len(tlo_redis_final) == len(tlo_availability_df.item_code)

    tlo_redis_final = tlo_availability_df.merge(tlo_redis_final, on=['Facility_ID', 'item_code', 'month'],
                                                how='left', validate="1:1")

    return tlo_redis_final


# Plot final availability
def plot_availability_heatmap(
    df: pd.DataFrame,
    y_var: str = None,
    scenario_cols: list[str] = None,
    filter_dict: dict = None,
    cmap: str = "RdYlGn",
    vmin: float = 0,
    vmax: float = 1,
    figsize: tuple = (10, 8),
    annot: bool = True,
    rename_scenarios_dict: dict = None,
    title: str = 'Availability across scenarios',
    figname: Path = None,
):
    """
    Flexible heatmap generator supporting filters and multiple (wide-format) scenario columns.
    """
    if filter_dict:
        for k, v in filter_dict.items():
            if isinstance(v, (list, tuple, set)):
                df = df[df[k].isin(v)]
            else:
                df = df[df[k] == v]

    aggregated_df = df.groupby([y_var])[scenario_cols].mean().reset_index()
    heatmap_data = aggregated_df.set_index(y_var)

    # Calculate aggregate column (true overall mean)
    aggregate_col = df[scenario_cols].mean()
    if rename_scenarios_dict:
        aggregate_col = aggregate_col.rename(index=rename_scenarios_dict)
        heatmap_data = heatmap_data.rename(columns=rename_scenarios_dict)
    heatmap_data.loc['Average'] = aggregate_col

    # Generate the heatmap
    sns.set(font_scale=1)
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        heatmap_data,
        annot=annot,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Proportion of days on which consumable is available'}
    )

    plt.title(title)
    plt.xlabel('Scenarios')
    plt.ylabel(y_var)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(rotation=0, fontsize=11)
    ax.set_xticklabels(
        [textwrap.fill(label.get_text(), 20) for label in ax.get_xticklabels()],
        rotation=90, ha='center'
    )
    ax.set_yticklabels(
        [textwrap.fill(label.get_text(), 25) for label in ax.get_yticklabels()],
        rotation=0, va='center'
    )

    if figname:
        plt.savefig(outputfilepath / figname, dpi=500, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    plt.close()


'''
# Clean item category
clean_category_names = {'cancer': 'Cancer', 'cardiometabolicdisorders': 'Cardiometabolic Disorders',
                        'contraception': 'Contraception', 'general': 'General', 'hiv': 'HIV', 'malaria': 'Malaria',
                        'ncds': 'Non-communicable Diseases', 'neonatal_health': 'Neonatal Health',
                        'other_childhood_illnesses': 'Other Childhood Illnesses', 'reproductive_health': 'Reproductive Health',
                        'road_traffic_injuries': 'Road Traffic Injuries', 'tb': 'Tuberculosis',
                        'undernutrition': 'Undernutrition', 'epi': 'Expanded programme on immunization'}
df_for_plots['item_category_clean'] = df_for_plots['item_category'].map(clean_category_names)

scenario_cols = ['available_prop', 'available_prop_scenario1', 'available_prop_scenario2', 'available_prop_scenario3',
                 'available_prop_scenario6', 'available_prop_scenario7', 'available_prop_scenario8',
                 'available_prop_scenario20', 'available_prop_scenario16', 'available_prop_scenario17',
                 'available_prop_scenario18', 'available_prop_scenario19']
rename_dict = {'available_prop': 'Actual',
               'available_prop_scenario1': 'Non-therapeutic consumables',
               'available_prop_scenario2': 'Vital medicines',
               'available_prop_scenario3': 'Pharmacist-managed',
               'available_prop_scenario6': '75th percentile facility',
               'available_prop_scenario7': '90th percentile facility',
               'available_prop_scenario8': 'Best facility',
               'available_prop_scenario20': 'National Pooling',
               'available_prop_scenario16': 'District Pooling',
               'available_prop_scenario17': 'Cluster Pooling',
               'available_prop_scenario18': 'Pairwise exchange (60-min radius)',
               'available_prop_scenario19': 'Pairwise exchange (30-min radius)'}
scenario_names = list(rename_dict.values())

# Plot heatmap for level 1a
plot_availability_heatmap(
    df=df_for_plots,
    scenario_cols=scenario_cols,
    y_var="item_category_clean",
    filter_dict={"Facility_Level": ["1a"]},
    title="Availability across Scenarios — Level 1a",
    rename_scenarios_dict=rename_dict,
    cmap="RdYlGn",
    figname='availability_1a.png'
)

# Plot heatmap for level 1b
plot_availability_heatmap(
    df=df_for_plots,
    scenario_cols=scenario_cols,
    y_var="item_category_clean",
    filter_dict={"Facility_Level": ["1b"]},
    title="Availability across Scenarios — Level 1b",
    rename_scenarios_dict=rename_dict,
    cmap="RdYlGn",
    figname='availability_1b.png'
)
'''
