"""
Extract DALY totals and DALYs by wealth and cause, alongside population breakdowns,
for TARGET_PERIOD and save them to CSV.
"""

from pathlib import Path
import pandas as pd
import ast

from tlo import Date
from tlo.analysis.utils import extract_draw_names, extract_results, get_scenario_info


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    TARGET_PERIOD = (Date(2025, 1, 1), Date(2035, 12, 31))
    scenario_info = get_scenario_info(results_folder)
    draw_names = extract_draw_names(results_folder)

    def set_draw_names_as_column_index_level_0(_df):
        draws = _df.columns.levels[0]
        missing_draws = set(draws) - draw_names.keys()
        if missing_draws:
            raise ValueError(f"No saved draw names for draws: {sorted(missing_draws)}")
        names = [draw_names[draw] for draw in draws]
        if len(names) != len(set(names)):
            raise ValueError("Saved draw names must be unique to label result columns.")
        _df.columns = _df.columns.set_levels(names, level=0)
        return _df

    def get_year_bounds():
        return TARGET_PERIOD[0].year, TARGET_PERIOD[1].year

    def filter_to_target_period(_df):
        start_year, end_year = get_year_bounds()

        if "year" in _df.columns:
            # Keep same style/logic as your original: year-based filtering for stacked annual logs
            _df = _df.loc[_df["year"].between(start_year, end_year)]
            # Guardrail against crashed/truncated runs
            assert set(range(start_year, end_year + 1)).issubset(set(_df["year"].unique())), \
                "Some years are not recorded in TARGET_PERIOD."
        elif "date" in _df.columns:
            _df = _df.loc[pd.to_datetime(_df["date"]).between(*TARGET_PERIOD)]

        return _df

    def find_cause_columns(_df):
        # Same principle as your original drop(columns=[...]) approach, but dynamic/robust
        non_cause_cols = {
            "date", "year", "sex", "age_range", "age_grp",
            "li_wealth", "li_urban", "region_of_residence"
        }
        return [c for c in _df.columns if c not in non_cause_cols]

    def get_value_column(_df):
        # Population logs may or may not use explicit 'value'
        if "value" in _df.columns:
            return "value"
        numeric_cols = [c for c in _df.columns if pd.api.types.is_numeric_dtype(_df[c])]
        assert len(numeric_cols) > 0, f"No numeric value column found. Columns are: {list(_df.columns)}"
        return numeric_cols[-1]

    # %% DALYs breakdowns
    def get_dalys_per_wealth(_df):
        years_needed = [i.year for i in TARGET_PERIOD]
        assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
        return (
            _df
            .loc[_df.year.between(*years_needed)]
            .drop(columns=['date', 'sex', 'age_range','li_urban', 'region_of_residence', 'year'], errors='ignore')
            .groupby(['li_wealth'])
            .sum()
            .sum(axis=1)
        )

    def get_dalys_per_wealth_cause(_df):
        years_needed = [i.year for i in TARGET_PERIOD]
        assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
        cause_columns = find_cause_columns(_df)
        if not cause_columns:
            raise ValueError("No DALY cause columns found in the healthburden log.")
        return (
            _df.loc[_df.year.between(*years_needed)]
            .groupby('li_wealth')[cause_columns]
            .sum()
            .stack()
            .rename_axis(index=['li_wealth', 'cause'])
        )

    def get_dalys_per_wealth_age(_df):
        years_needed = [i.year for i in TARGET_PERIOD]
        assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
        return (
            _df
            .loc[_df.year.between(*years_needed)]
            .drop(columns=['date','sex', 'li_urban', 'region_of_residence', 'year'], errors='ignore')
            .groupby(['li_wealth', 'age_range'])
            .sum()
            .sum(axis=1)
        )

    def get_dalys_per_wealth_sex(_df):
        years_needed = [i.year for i in TARGET_PERIOD]
        assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
        return (
            _df
            .loc[_df.year.between(*years_needed)]
            .drop(columns=['date', 'age_range', 'li_urban', 'region_of_residence', 'year'], errors='ignore')
            .groupby(['li_wealth', 'sex'])
            .sum()
            .sum(axis=1)
        )

    def get_dalys_per_wealth_urban(_df):
        years_needed = [i.year for i in TARGET_PERIOD]
        assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
        return (
            _df
            .loc[_df.year.between(*years_needed)]
            .drop(columns=['date', 'sex','age_range', 'region_of_residence', 'year'], errors='ignore')
            .groupby(['li_wealth', 'li_urban'])
            .sum()
            .sum(axis=1)
        )

    def get_dalys_per_wealth_region(_df):
        years_needed = [i.year for i in TARGET_PERIOD]
        assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
        return (
            _df
            .loc[_df.year.between(*years_needed)]
            .drop(columns=['date', 'sex', 'age_range','li_urban', 'year'], errors='ignore')
            .groupby(['li_wealth', 'region_of_residence'])
            .sum()
            .sum(axis=1)
        )
    dalys_key = "dalys_by_wealth_urban_region_stacked_by_age_and_time"

    dalys_per_wealth = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key=dalys_key,
        custom_generate_series=get_dalys_per_wealth,
        do_scaling=True
    ).pipe(set_draw_names_as_column_index_level_0)

    dalys_per_wealth_cause = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key=dalys_key,
        custom_generate_series=get_dalys_per_wealth_cause,
        do_scaling=True
    ).pipe(set_draw_names_as_column_index_level_0)

    dalys_per_wealth_age = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key=dalys_key,
        custom_generate_series=get_dalys_per_wealth_age,
        do_scaling=True
    ).pipe(set_draw_names_as_column_index_level_0)

    dalys_per_wealth_sex = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key=dalys_key,
        custom_generate_series=get_dalys_per_wealth_sex,
        do_scaling=True
    ).pipe(set_draw_names_as_column_index_level_0)

    dalys_per_wealth_urban = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key=dalys_key,
        custom_generate_series=get_dalys_per_wealth_urban,
        do_scaling=True
    ).pipe(set_draw_names_as_column_index_level_0)

    dalys_per_wealth_region = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key=dalys_key,
        custom_generate_series=get_dalys_per_wealth_region,
        do_scaling=True
    ).pipe(set_draw_names_as_column_index_level_0)

    # %% Population breakdowns
    def _get_population_long_from_wide(_df):
        # 1) filter to target period (same logic style as get_pop_no)
        _df = _df.loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD)]

        # 2) sum over years for each strata-column
        s = _df.drop(columns=['date'], errors='ignore').sum(axis=0)

        # 3) parse tuple-like column names
        records = []
        for k, v in s.items():
            # k like "('F', 1, False, 'Central')"
            if isinstance(k, str):
                try:
                    sex, age_range, wealth, urban, region = ast.literal_eval(k)
                except Exception:
                    # skip non-strata columns if any
                    continue
            elif isinstance(k, tuple) and len(k) == 4:
                sex, age_range, wealth, urban, region = k
            else:
                continue

            records.append({
                'sex': sex,
                'age_range': age_range,
                'li_wealth': wealth,
                'li_urban': urban,
                'region_of_residence': region,
                'value': v
            })

        return pd.DataFrame.from_records(records)

    def get_pop_per_wealth(_df):
        z = _get_population_long_from_wide(_df)
        return z.groupby(['li_wealth'])['value'].sum()

    def get_pop_per_wealth_age(_df):
        z = _get_population_long_from_wide(_df)
        return z.groupby(['li_wealth', 'age_range'])['value'].sum()

    def get_pop_per_wealth_sex(_df):
        z = _get_population_long_from_wide(_df)
        return z.groupby(['li_wealth', 'sex'])['value'].sum()

    def get_pop_per_wealth_urban(_df):
        z = _get_population_long_from_wide(_df)
        return z.groupby(['li_wealth', 'li_urban'])['value'].sum()

    def get_pop_per_wealth_region(_df):
        z = _get_population_long_from_wide(_df)
        return z.groupby(['li_wealth', 'region_of_residence'])['value'].sum()


    pop_key = "population_by_wealth_age_urban_region"

    pop_per_wealth = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key=pop_key,
        custom_generate_series=get_pop_per_wealth,
        do_scaling=True
    ).pipe(set_draw_names_as_column_index_level_0)

    pop_per_wealth_age = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key=pop_key,
        custom_generate_series=get_pop_per_wealth_age,
        do_scaling=True
    ).pipe(set_draw_names_as_column_index_level_0)

    pop_per_wealth_sex = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key=pop_key,
        custom_generate_series=get_pop_per_wealth_sex,
        do_scaling=True
    ).pipe(set_draw_names_as_column_index_level_0)

    pop_per_wealth_urban = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key=pop_key,
        custom_generate_series=get_pop_per_wealth_urban,
        do_scaling=True
    ).pipe(set_draw_names_as_column_index_level_0)

    pop_per_wealth_region = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key=pop_key,
        custom_generate_series=get_pop_per_wealth_region,
        do_scaling=True
    ).pipe(set_draw_names_as_column_index_level_0)

    # %% Save outputs
    output_folder.mkdir(parents=True, exist_ok=True)

    dalys_per_wealth.to_csv(output_folder / "dalys_per_wealth.csv")
    dalys_per_wealth_cause.to_csv(output_folder / "dalys_per_wealth_cause.csv")
    dalys_per_wealth_age.to_csv(output_folder / "dalys_per_wealth_age.csv")
    dalys_per_wealth_sex.to_csv(output_folder / "dalys_per_wealth_sex.csv")
    dalys_per_wealth_urban.to_csv(output_folder / "dalys_per_wealth_urban.csv")
    dalys_per_wealth_region.to_csv(output_folder / "dalys_per_wealth_region.csv")

    pop_per_wealth.to_csv(output_folder / "population_per_wealth.csv")
    pop_per_wealth_age.to_csv(output_folder / "population_per_wealth_age.csv")
    pop_per_wealth_sex.to_csv(output_folder / "population_per_wealth_sex.csv")
    pop_per_wealth_urban.to_csv(output_folder / "population_per_wealth_urban.csv")
    pop_per_wealth_region.to_csv(output_folder / "population_per_wealth_region.csv")

    print(f"Done. CSVs saved in: {output_folder}")


if __name__ == "__main__":

    apply(
        #results_folder=Path("outputs/n.fuller@ic.ac.uk/htm_with_and_without_hss-2026-08-21T101642Z"),
        #output_folder=Path("outputs/n.fuller@ic.ac.uk/htm_with_and_without_hss-2026-08-21T101642Z"),
        results_folder=Path("outputs/n.fuller@ic.ac.uk/htm_with_and_without_hss-2026-08-07T092059Z"),
        output_folder=Path("outputs/n.fuller@ic.ac.uk/htm_with_and_without_hss-2026-08-07T092059Z"),
        resourcefilepath=Path("./resources")
    )
