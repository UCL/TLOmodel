from pathlib import Path

import pandas as pd


def load_rm_database(path: Path | str | None = None, out_path: Path | str | None = None) -> pd.DataFrame:
    """
    Load the RM_Database sheet from RM Round 7_20220714.ods and save it as out_path (CSV)
     If the CSV already exists at out_path, load it instead.
    """
    if path is None:
            # default to the directory containing this script (same folder as the ods file)
            path = Path(__file__).parent / "RM Round 7_20220714_onelineheader.ods"
    path = Path(path)
    if out_path is None:
            # default to the directory containing this script (same folder as the ods file)
            out_path = Path(__file__).parent / "RM_Database.csv"
    out_path = Path(out_path)

    def try_load_df():
        if not path.exists():
            raise FileNotFoundError(f"`{path}` not found")
        try:
            return pd.read_excel(path, sheet_name="RM_Database", engine="odf")
        except Exception as e:
            raise RuntimeError("Failed to read `RM_Database` from `RM Round 7_20220714.ods`.") from e

    if not out_path.exists():
        df = try_load_df()
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} rows to `{out_path}`")
    else:
        df = pd.read_csv(out_path)
        print(f"Loaded {len(df)} rows from `{out_path}`")
    # print(f"\ncolumns:\n{df.columns}")

    return df

def load_rm_database_nutrition(path: Path | str | None = None, out_path: Path | str | None = None) -> pd.DataFrame:
    if path is None:
        # default to the directory containing this script (same folder as the ods file)
        path = Path(__file__).parent / "RM_Database.csv"
    path = Path(path)
    def try_load_df():
        if not path.exists():
            load_rm_database()
        try:
            return pd.read_csv(path)
        except Exception as e:
            raise RuntimeError("Failed to read `RM_Database` from `RM_Database.csv`.") from e

    if out_path is None:
        # default to the directory containing this script (same folder as the ods file)
        out_path = Path(__file__).parent / "RM_Database_Nutrition.csv"
    out_path = Path(out_path)

    if not out_path.exists():
        df = try_load_df()
        col = "Programmatic Function"
        if col not in df.columns:
            raise KeyError(f"Column `{col}` not found in loaded DataFrame")
        df_nutr = df[df[col] == "Nutrition"].copy()
        # drop the column since it has the same value in all rows
        df_nutr = df_nutr.drop(columns=[col])
        df_nutr.to_csv(out_path, index=False)
        print(f"Saved {len(df_nutr)} rows to `{out_path}`")
    else:
        df_nutr = pd.read_csv(out_path)
        print(f"Loaded {len(df_nutr)} rows from `{out_path}`")
    # print(f"\ncolumns:\n{df_nutr.columns}")

    return df_nutr

def load_rm_database_interv_level(interv_level_name: str, interv_level: str, path: Path | str | None = None) -> pd.DataFrame:
    if path is None:
        # default to the directory containing this script (same folder as the csv)
        path = Path(__file__).parent / "RM_Database_Nutrition.csv"
    path = Path(path)

    def try_load_df_nutr():
        if not path.exists():
            # ensure the nutrition CSV exists
            load_rm_database_nutrition()
        try:
            return pd.read_csv(path)
        except Exception as e:
            raise RuntimeError("Failed to read `RM_Database_Nutrition` from `RM_Database_Nutrition.csv`.") from e

    out_path = Path(__file__).parent / f"RM_Database_{interv_level_name}.csv"
    if not out_path.exists():
        df = try_load_df_nutr()
        col = "Programmatic Intervention Level 1 [NEW]"
        if col not in df.columns:
            raise KeyError(f"Column `{col}` not found in loaded DataFrame")
        df_interv_level = df[df[col] == interv_level].copy()
        # drop the column since it has the same value in all rows
        df_interv_level = df_interv_level.drop(columns=[col])
        df_interv_level.to_csv(out_path, index=False)
        print(f"Saved {len(df_interv_level)} rows to `{out_path}`")
    else:
        df_interv_level = pd.read_csv(out_path)
        print(f"Loaded {len(df_interv_level)} rows from `{out_path}`")
    # print(f"\ncolumns:\n{df_interv_level.columns}")

    return df_interv_level

if __name__ == "__main__":
    # load_rm_database()
    df_nutr = load_rm_database_nutrition()
    df_preven_undernutr = load_rm_database_interv_level(
      interv_level_name="Prevention_of_Undernutrition", interv_level="Prevention of Undernutrition"
    )
    df_behavior_change_nutr = load_rm_database_interv_level(
      interv_level_name="BehaviorChange_Nutri", interv_level="Behavior Change Communication for Nutrition"
    )

    def print_unique_vals(df:pd.DataFrame, df_name:str, col_name:str) -> None:
        if col_name in df.columns:
            vals = df[col_name].dropna().unique()
            vals = sorted(vals, key=lambda x: str(x))
            print(f"\n{df_name}--unique values in {col_name} ({len(vals)}):")
            for v in vals:
                print(v)
        else:
            print(f"\nColumn {col_name} not found in `df_preven_undernutr`. Columns:\n{df.columns}")

    print_unique_vals(df_preven_undernutr, "df_preven_undernutr", "Description of Activity")
    print_unique_vals(df_preven_undernutr, "df_preven_undernutr", "Cost Sub-Type")
    print_unique_vals(df_preven_undernutr, "df_preven_undernutr", "Project Name ")

    print_unique_vals(df_behavior_change_nutr, "df_behavior_change_nutr", "Description of Activity")
    print_unique_vals(df_behavior_change_nutr, "df_behavior_change_nutr", "Cost Sub-Type")
    print_unique_vals(df_behavior_change_nutr, "df_behavior_change_nutr", "Project Name ")

    print_unique_vals(df_nutr, "df_nutr", "Programmatic Intervention Level 1 [NEW]")
    print_unique_vals(df_nutr, "df_nutr", "Cost Sub-Type")
    print_unique_vals(df_nutr, "df_nutr", "Project Name ")

    def print_structure(df: pd.DataFrame, df_name: str) -> None:
        proj_col = "Project Name "
        interv_level_col = "Programmatic Intervention Level 1 [NEW]"
        activ_descrip_col = "Description of Activity"
        cost_col = "Cost Sub-Type"
        expend_col = "FY Ending 2019 EXPENDITURE (USD)(Jul 2018 - Jun 2019)"
        budg_col = "FY Ending 2020 BUDGETS (USD)(Jul 2019 - Jun 2020)"

        for c in (proj_col, interv_level_col, activ_descrip_col, cost_col):
            if c not in df.columns:
                print(f"\nColumn `{c}` not found in `{df_name}`. Available columns:\n{df.columns}")
                return

        projects = sorted(df[proj_col].dropna().unique(), key=lambda x: str(x))
        for proj in projects:
            proj_mask = df[proj_col] == proj
            levels = sorted(df.loc[proj_mask, interv_level_col].dropna().unique(), key=lambda x: str(x))
            print(f"\nProject: {proj} (levels: {len(levels)})")
            for level in levels:
                print(f"  {interv_level_col}: {level}")
                subset = df[proj_mask & (df[interv_level_col] == level)]
                # prepare grouped combos with optional sums
                cols_to_sum = []
                if expend_col in subset.columns:
                    cols_to_sum.append(expend_col)
                if budg_col in subset.columns:
                    cols_to_sum.append(budg_col)

                if cols_to_sum:
                    work = subset[[activ_descrip_col, cost_col] + cols_to_sum].copy()
                    work[[activ_descrip_col, cost_col]] = work[[activ_descrip_col, cost_col]].fillna("")
                    combos = work.groupby([activ_descrip_col, cost_col], as_index=False)[cols_to_sum].sum(numeric_only=True)
                else:
                    combos = subset[[activ_descrip_col, cost_col]].drop_duplicates().fillna("")

                if combos.empty:
                    print("    (no Description/Cost entries)")
                    continue
                for _, row in combos.iterrows():
                    desc = row.get(activ_descrip_col, "") or "(empty)"
                    cost = row.get(cost_col, "") or "(empty)"
                    if cols_to_sum:
                        expend_total = row.get(expend_col)
                        budg_total = row.get(budg_col)
                        expend_str = f"{expend_total:,.2f}" if pd.notna(expend_total) else "(no data)"
                        budg_str = f"{budg_total:,.2f}" if pd.notna(budg_total) else "(no data)"
                        print(f"    - Description: {desc} | Cost Sub-Type: {cost} | FY2019 Exp: {expend_str} | FY2020 Budget: {budg_str}")
                    else:
                        print(f"    - Description: {desc} | Cost Sub-Type: {cost}")

    def print_cost_by_programme(df: pd.DataFrame) -> None:
        proj_col = "Project Name "
        expend_col = "FY Ending 2019 EXPENDITURE (USD)(Jul 2018 - Jun 2019)"
        budg_col = "FY Ending 2020 BUDGETS (USD)(Jul 2019 - Jun 2020)"

        for c in (proj_col, expend_col, budg_col):
            if c not in df.columns:
                print(f"\nColumn `{c}` not found in dataframe. Available columns:\n{df.columns}")
                return

        grouped = df.groupby(proj_col, dropna=False)[[expend_col, budg_col]].sum(numeric_only=True)
        # for proj in sorted(grouped.index, key=lambda x: str(x)):
            # row = grouped.loc[proj]
            # expend_total = row.get(expend_col)
            # budg_total = row.get(budg_col)
            # expend_str = f"{expend_total:,.2f}" if pd.notna(expend_total) else "(no data)"
            # budg_str = f"{budg_total:,.2f}" if pd.notna(budg_total) else "(no data)"
            # print(f"Project: {proj} | FY2018/19 Expenditure Total: {expend_str} | FY2019/20 Budget Total: {budg_str}")

        # summary statistics across projects (use only projects with numeric values)
        expend_series = grouped[expend_col].dropna()
        budg_series = grouped[budg_col].dropna()

        def fmt(v):
            return f"{v:,.0f}" if pd.notna(v) else "(no data)"
        ex_min = expend_series.min() if not expend_series.empty else float("nan")
        ex_max = expend_series.max() if not expend_series.empty else float("nan")
        ex_median = expend_series.median() if not expend_series.empty else float("nan")
        ex_mean = expend_series.mean() if not expend_series.empty else float("nan")
        ex_sum = expend_series.sum() if not expend_series.empty else float("nan")

        bd_min = budg_series.min() if not budg_series.empty else float("nan")
        bd_max = budg_series.max() if not budg_series.empty else float("nan")
        bd_median = budg_series.median() if not budg_series.empty else float("nan")
        bd_mean = budg_series.mean() if not budg_series.empty else float("nan")
        bd_sum = budg_series.sum() if not budg_series.empty else float("nan")

        print("\nAcross-project statistics (per-project totals) 2018 USD:")
        ex_low = expend_series.sort_values().head(5)
        ex_high = expend_series.sort_values().tail(5)
        ex_low_str = ", ".join(fmt(val) for val in ex_low) if not ex_low.empty else "(no data)"
        ex_high_str = ", ".join(fmt(val) for val in ex_high) if not ex_high.empty else "(no data)"
        print(f"FY 2018/19 Expenditure per project — | min: {fmt(ex_min)} | max {fmt(ex_max)} | mean: {fmt(ex_mean)} "
              f"| sum over all projects: {fmt(ex_sum)}\n"
              f"                                   — | lowest 5: {ex_low_str} | highest 5: {ex_high_str} ")
        bd_low = budg_series.sort_values().head(5)
        bd_high = budg_series.sort_values().tail(5)
        bd_low_str = ", ".join(fmt(val) for val in bd_low) if not bd_low.empty else "(no data)"
        bd_high_str = ", ".join(fmt(val) for val in bd_high) if not bd_high.empty else "(no data)"
        print(f"\nFY 2019/20 Budget per project      — | min: {fmt(bd_min)} | max {fmt(bd_max)} | mean: {fmt(bd_mean)} "
              f"| sum over all projects: {fmt(bd_sum)}\n"
              f"                                   — | lowest 5: {bd_low_str} | highest 5: {bd_high_str} ")

        # 2023 USD equivalents (apply multiplier)
        multiplier = 1.0165 * 1.0133 * 1.0457 * 1.0713 * 1.0360
        ex_low_2023 = ex_low * multiplier if not ex_low.empty else ex_low
        ex_high_2023 = ex_high * multiplier if not ex_high.empty else ex_high
        ex_low_2023_str = ", ".join(fmt(val) for val in ex_low_2023) if not ex_low_2023.empty else "(no data)"
        ex_high_2023_str = ", ".join(fmt(val) for val in ex_high_2023) if not ex_high_2023.empty else "(no data)"
        ex_min_2023 = ex_min * multiplier if pd.notna(ex_min) else float("nan")
        ex_max_2023 = ex_max * multiplier if pd.notna(ex_max) else float("nan")
        ex_mean_2023 = ex_mean * multiplier if pd.notna(ex_mean) else float("nan")
        ex_sum_2023 = ex_sum * multiplier if pd.notna(ex_sum) else float("nan")

        bd_low_2023 = bd_low * multiplier if not bd_low.empty else bd_low
        bd_high_2023 = bd_high * multiplier if not bd_high.empty else bd_high
        bd_low_2023_str = ", ".join(fmt(val) for val in bd_low_2023) if not bd_low_2023.empty else "(no data)"
        bd_high_2023_str = ", ".join(fmt(val) for val in bd_high_2023) if not bd_high_2023.empty else "(no data)"
        bd_min_2023 = bd_min * multiplier if pd.notna(bd_min) else float("nan")
        bd_max_2023 = bd_max * multiplier if pd.notna(bd_max) else float("nan")
        bd_mean_2023 = bd_mean * multiplier if pd.notna(bd_mean) else float("nan")
        bd_sum_2023 = bd_sum * multiplier if pd.notna(bd_sum) else float("nan")

        print(f"\nIn 2023 USD (multiplier = {multiplier:.6f}):")
        print(f"FY 2018/19 Expenditure per project — | min: {fmt(ex_min_2023)} | max {fmt(ex_max_2023)} | mean: {fmt(ex_mean_2023)} "
              f"| sum over all projects: {fmt(ex_sum_2023)}\n"
              f"                                   — | lowest 5: {ex_low_2023_str} | highest 5: {ex_high_2023_str} ")
        print(f"\nFY 2019/20 Budget per project      — | min: {fmt(bd_min_2023)} | max {fmt(bd_max_2023)} | mean: {fmt(bd_mean_2023)} "
              f"| sum over all projects: {fmt(bd_sum_2023)}\n"
              f"                                   — | lowest 5: {bd_low_2023_str} | highest 5: {bd_high_2023_str} ")

        # Additional statistics considering only projects with positive (> 0) expenditure/budget
        pos_expend = expend_series[expend_series > 0]
        pos_budg = budg_series[budg_series > 0]

        def make_stats(series):
            if series.empty:
                empty_series = pd.Series(dtype=float)
                return {
                    "min": float("nan"), "max": float("nan"), "mean": float("nan"),
                    "sum": float("nan"), "low5": empty_series, "high5": empty_series
                }
            return {
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "sum": series.sum(),
                "low5": series.sort_values().head(5),
                "high5": series.sort_values().tail(5)
            }

        pos_ex_stats = make_stats(pos_expend)
        pos_bd_stats = make_stats(pos_budg)

        pos_ex_low = pos_ex_stats["low5"]
        pos_ex_high = pos_ex_stats["high5"]
        pos_bd_low = pos_bd_stats["low5"]
        pos_bd_high = pos_bd_stats["high5"]

        pos_ex_low_str = ", ".join(fmt(val) for val in pos_ex_low) if not pos_ex_low.empty else "(no data)"
        pos_ex_high_str = ", ".join(fmt(val) for val in pos_ex_high) if not pos_ex_high.empty else "(no data)"
        pos_bd_low_str = ", ".join(fmt(val) for val in pos_bd_low) if not pos_bd_low.empty else "(no data)"
        pos_bd_high_str = ", ".join(fmt(val) for val in pos_bd_high) if not pos_bd_high.empty else "(no data)"

        print("\nStatistics for projects with positive (>0) values:")
        print(f"FY 2018/19 Expenditure per project (positive only) — | min: {fmt(pos_ex_stats['min'])} | max: {fmt(pos_ex_stats['max'])} | mean: {fmt(pos_ex_stats['mean'])} "
              f"| sum over those projects: {fmt(pos_ex_stats['sum'])}\n"
              f"                                                   — | lowest 5: {pos_ex_low_str} | highest 5: {pos_ex_high_str}")
        print(f"\nFY 2019/20 Budget per project (positive only)      — | min: {fmt(pos_bd_stats['min'])} | max: {fmt(pos_bd_stats['max'])} | mean: {fmt(pos_bd_stats['mean'])} "
              f"| sum over those projects: {fmt(pos_bd_stats['sum'])}\n"
              f"                                                   — | lowest 5: {pos_bd_low_str} | highest 5: {pos_bd_high_str}")

        # 2023 USD equivalents for positive-only stats (apply multiplier)
        # reuse `multiplier` defined earlier
        pos_ex_low_2023 = pos_ex_low * multiplier if not getattr(pos_ex_low, "empty", False) else pos_ex_low
        pos_ex_high_2023 = pos_ex_high * multiplier if not getattr(pos_ex_high, "empty", False) else pos_ex_high
        pos_bd_low_2023 = pos_bd_low * multiplier if not getattr(pos_bd_low, "empty", False) else pos_bd_low
        pos_bd_high_2023 = pos_bd_high * multiplier if not getattr(pos_bd_high, "empty", False) else pos_bd_high

        pos_ex_min_2023 = pos_ex_stats["min"] * multiplier if pd.notna(pos_ex_stats["min"]) else float("nan")
        pos_ex_max_2023 = pos_ex_stats["max"] * multiplier if pd.notna(pos_ex_stats["max"]) else float("nan")
        pos_ex_mean_2023 = pos_ex_stats["mean"] * multiplier if pd.notna(pos_ex_stats["mean"]) else float("nan")
        pos_ex_sum_2023 = pos_ex_stats["sum"] * multiplier if pd.notna(pos_ex_stats["sum"]) else float("nan")

        pos_bd_min_2023 = pos_bd_stats["min"] * multiplier if pd.notna(pos_bd_stats["min"]) else float("nan")
        pos_bd_max_2023 = pos_bd_stats["max"] * multiplier if pd.notna(pos_bd_stats["max"]) else float("nan")
        pos_bd_mean_2023 = pos_bd_stats["mean"] * multiplier if pd.notna(pos_bd_stats["mean"]) else float("nan")
        pos_bd_sum_2023 = pos_bd_stats["sum"] * multiplier if pd.notna(pos_bd_stats["sum"]) else float("nan")

        pos_ex_low_2023_str = ", ".join(fmt(val) for val in pos_ex_low_2023) if not getattr(pos_ex_low_2023, "empty", True) else "(no data)"
        pos_ex_high_2023_str = ", ".join(fmt(val) for val in pos_ex_high_2023) if not getattr(pos_ex_high_2023, "empty", True) else "(no data)"
        pos_bd_low_2023_str = ", ".join(fmt(val) for val in pos_bd_low_2023) if not getattr(pos_bd_low_2023, "empty", True) else "(no data)"
        pos_bd_high_2023_str = ", ".join(fmt(val) for val in pos_bd_high_2023) if not getattr(pos_bd_high_2023, "empty", True) else "(no data)"

        print(f"\nIn 2023 USD (multiplier = {multiplier:.6f}) for positive-only projects:")
        print(f"FY 2018/19 Expenditure per project (positive only) — | min: {fmt(pos_ex_min_2023)} | max {fmt(pos_ex_max_2023)} | mean: {fmt(pos_ex_mean_2023)} "
              f"| sum over those projects: {fmt(pos_ex_sum_2023)}\n"
              f"                                                   — | lowest 5: {pos_ex_low_2023_str} | highest 5: {pos_ex_high_2023_str}")
        print(f"\nFY 2019/20 Budget per project (positive only)      — | min: {fmt(pos_bd_min_2023)} | max {fmt(pos_bd_max_2023)} | mean: {fmt(pos_bd_mean_2023)} "
              f"| sum over those projects: {fmt(pos_bd_sum_2023)}\n"
              f"                                                   — | lowest 5: {pos_bd_low_2023_str} | highest 5: {pos_bd_high_2023_str}")


    print_cost_by_programme(df_nutr)
