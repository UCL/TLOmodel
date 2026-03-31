"""Utilities for extracting and processing results for treatment-id analyses."""

from typing import Tuple

import numpy as np
import pandas as pd

from scripts.lcoa_inputs_from_tlo_analyses.scenario_effect_of_treatment_ids import (
    EffectOfEachTreatment,
)
from tlo import Date
from tlo.analysis.utils import make_age_grp_types, summarize, to_age_group


TARGET_PERIOD = (Date(2025, 1, 1), Date(2041, 1, 1))

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

def get_total_population_by_year(
    _df: pd.DataFrame,
    target_period_tuple: tuple[Date, Date] = TARGET_PERIOD,
) -> pd.Series:
    years_needed = [i.year for i in target_period_tuple]
    _df["year"] = pd.to_datetime(_df["date"]).dt.year
    return _df.loc[_df["year"].between(min(years_needed), max(years_needed)), ["year", "total"]].set_index("year")[
        "total"
    ]


def extract_deaths_total(df: pd.DataFrame) -> pd.Series:
    return pd.Series({"Total": len(df)})


def target_period(target_period_tuple: tuple[Date, Date] = TARGET_PERIOD) -> str:
    """Returns the target period as a string of the form YYYY-YYYY."""
    return "-".join(str(t.year) for t in target_period_tuple)


def get_periods_within_target_period(
    period_length_years: int,
    target_period_tuple: tuple[Date, Date] = TARGET_PERIOD,
) -> list[tuple[str, tuple[int, int]]]:
    """Return chunks within target period as [(label, (start_year, end_year)), ...]."""
    if period_length_years <= 0:
        raise ValueError("period_length_years must be a positive integer.")
    start_year, end_year = target_period_tuple[0].year, target_period_tuple[1].year
    periods = []
    for chunk_start in range(start_year, end_year + 1, period_length_years):
        chunk_end = min(chunk_start + period_length_years - 1, end_year)
        periods.append((f"{chunk_start}-{chunk_end}", (chunk_start, chunk_end)))
    return periods


def get_parameter_names_from_scenario_file() -> Tuple[str]:
    """Get tuple of scenario names from Scenario class used to create results."""
    e = EffectOfEachTreatment()
    excluded = {"Only Hiv_Test_Selftest_*"}
    # I think Hiv_test_Selftest has been added after I had submitted the draws, hence filtering it out.
    return tuple(name for name in e._scenarios.keys() if name not in excluded)


def format_scenario_name(_sn: str) -> str:
    """Return reformatted scenario name ready for plotting."""
    if _sn == "Nothing":
        return "Nothing"
    else:
        return _sn.removeprefix("Only ")


def set_param_names_as_column_index_level_0(_df: pd.DataFrame, param_names: tuple[str, ...]) -> pd.DataFrame:
    """Set columns index level 0 as scenario param names."""

    ordered_param_names_no_prefix = {i: x for i, x in enumerate(param_names)}
    names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
    assert len(names_of_cols_level0) == len(_df.columns.levels[0])

    reformatted_names = map(format_scenario_name, names_of_cols_level0)
    _df.columns = _df.columns.set_levels(reformatted_names, level=0)
    return _df


def find_difference_extra_relative_to_comparison(
    _ser: pd.Series,
    comparison: str,
    scaled: bool = False,
    drop_comparison: bool = True,
):
    """Find run-wise differences relative to comparison in a series with multi-index."""
    return (
        _ser.unstack()
        .apply(lambda x: (x - x[comparison]) / (x[comparison] if scaled else 1.0), axis=0)
        .drop(index=([comparison] if drop_comparison else []))
        .stack()

    )


def find_mean_difference_in_appts_relative_to_comparison(
    _df: pd.DataFrame,
    comparison: str,
    drop_comparison: bool = True,
):
    """Find mean fewer appointments when treatment does not happen relative to comparison."""
    return -summarize(
        pd.concat(
            {
                _idx: find_difference_extra_relative_to_comparison(
                    row, comparison=comparison, drop_comparison=drop_comparison
                )
                for _idx, row in _df.iterrows()
            },
            axis=1,
        ).T,
        only_mean=True,
    )


def find_mean_difference_extra_relative_to_comparison_dataframe(
    _df: pd.DataFrame,
    comparison: str,
    drop_comparison: bool = True,
):
    """Same as find_difference_extra_relative_to_comparison but for dataframe."""
    return summarize(
        pd.concat(
            {
                _idx: find_difference_extra_relative_to_comparison(
                    row, comparison=comparison, drop_comparison=drop_comparison
                )
                for _idx, row in _df.iterrows()
            },
            axis=1,
        ).T,
        only_mean=True,
    )


def get_num_deaths_by_cause_label(_df: pd.DataFrame, target_period_tuple: tuple[Date, Date] = TARGET_PERIOD) -> pd.Series:
    """Return total deaths by label within target period."""
    return _df.loc[pd.to_datetime(_df.date).between(*target_period_tuple)].groupby(_df["label"]).size()


def get_num_dalys_by_cause_label(_df: pd.DataFrame, target_period_tuple: tuple[Date, Date] = TARGET_PERIOD) -> pd.Series:
    """Return total DALYS by label within target period."""
    return (
        _df.loc[_df.year.between(*[i.year for i in target_period_tuple])]
        .drop(columns=["date", "sex", "age_range", "year"])
        .sum()
    )


def make_get_num_deaths_by_cause_label_and_period(
    period_length_years: int,
    target_period_tuple: tuple[Date, Date] = TARGET_PERIOD,
):
    """Create helper that summarizes deaths by cause and period chunks + overall."""
    periods = get_periods_within_target_period(
        period_length_years=period_length_years,
        target_period_tuple=target_period_tuple,
    )
    period_lookup = {
        year: period_label
        for period_label, (start_year, end_year) in periods
        for year in range(start_year, end_year + 1)
    }
    target_period_label = target_period(target_period_tuple)

    def _get_num_deaths_by_cause_label_and_period(_df: pd.DataFrame) -> pd.Series:
        _df_in_target = _df.loc[pd.to_datetime(_df.date).between(*target_period_tuple)].copy()
        _df_in_target["year"] = pd.to_datetime(_df_in_target["date"]).dt.year
        _df_in_target["period"] = _df_in_target["year"].map(period_lookup)

        chunked = _df_in_target.groupby(["label", "period"]).size()
        overall = _df_in_target.groupby("label").size()
        overall.index = pd.MultiIndex.from_arrays(
            [overall.index, np.repeat(target_period_label, len(overall.index))], names=["label", "period"]
        )
        return pd.concat([chunked, overall]).sort_index()

    return _get_num_deaths_by_cause_label_and_period


def make_get_num_dalys_by_cause_label_and_period(
    period_length_years: int,
    target_period_tuple: tuple[Date, Date] = TARGET_PERIOD,
):
    """Create helper that summarizes DALYS by cause and period chunks + overall."""
    periods = get_periods_within_target_period(
        period_length_years=period_length_years,
        target_period_tuple=target_period_tuple,
    )
    period_lookup = {
        year: period_label
        for period_label, (period_start, period_end) in periods
        for year in range(period_start, period_end + 1)
    }
    start_year, end_year = target_period_tuple[0].year, target_period_tuple[1].year
    target_period_label = target_period(target_period_tuple)

    def _get_num_dalys_by_cause_label_and_period(_df: pd.DataFrame) -> pd.Series:
        _df_in_target = _df.loc[_df.year.between(start_year, end_year)].copy()
        _df_in_target["period"] = _df_in_target["year"].map(period_lookup)

        melted = (
            _df_in_target.drop(columns=["date", "sex", "age_range"])
            .melt(id_vars=["year", "period"], var_name="label", value_name="dalys")
        )
        chunked = melted.groupby(["label", "period"])["dalys"].sum()
        overall = melted.groupby("label")["dalys"].sum()
        overall.index = pd.MultiIndex.from_arrays(
            [overall.index, np.repeat(target_period_label, len(overall.index))], names=["label", "period"]
        )
        return pd.concat([chunked, overall]).sort_index()

    return _get_num_dalys_by_cause_label_and_period


def get_num_deaths_by_age_group(
    _df: pd.DataFrame,
    age_grp_lookup: dict,
    target_period_tuple: tuple[Date, Date] = TARGET_PERIOD,
):
    """Return total deaths by age-group in target period."""
    return (
        _df.loc[pd.to_datetime(_df.date).between(*target_period_tuple)]
        .groupby(_df["age"].map(age_grp_lookup).astype(make_age_grp_types()))
        .size()
    )


def get_total_num_death_by_agegrp_and_label(
    _df: pd.DataFrame,
    target_period_tuple: tuple[Date, Date] = TARGET_PERIOD,
) -> pd.Series:
    """Return deaths in target period by age-group and cause label."""
    _df_limited_to_dates = _df.loc[_df["date"].between(*target_period_tuple)]
    age_group = to_age_group(_df_limited_to_dates["age"])
    return _df_limited_to_dates.groupby([age_group, "label"])["person_id"].size()


def get_total_num_dalys_by_agegrp_and_label(
    _df: pd.DataFrame,
    target_period_tuple: tuple[Date, Date] = TARGET_PERIOD,
) -> pd.Series:
    """Return DALYS in target period by age-group and cause label."""
    return (
        _df.loc[_df.year.between(*[i.year for i in target_period_tuple])]
        .assign(age_group=_df["age_range"])
        .drop(columns=["date", "year", "sex", "age_range"])
        .melt(id_vars=["age_group"], var_name="label", value_name="dalys")
        .groupby(by=["age_group", "label"])["dalys"]
        .sum()
    )


def get_counts_of_hsi_by_short_treatment_id(
    _df: pd.DataFrame,
    target_period_tuple: tuple[Date, Date] = TARGET_PERIOD,
) -> pd.Series:
    """Get counts of short treatment ids occurring in target period."""
    mask = pd.to_datetime(_df["date"]).between(*target_period_tuple)
    _counts_by_treatment_id = _df.loc[mask, "TREATMENT_ID"].apply(pd.Series).sum().astype(int)
    return _counts_by_treatment_id


def get_counts_of_appts(_df: pd.DataFrame, target_period_tuple: tuple[Date, Date] = TARGET_PERIOD) -> pd.Series:
    """Get counts of appointments of each type being used in target period."""
    return (
        _df.loc[pd.to_datetime(_df["date"]).between(*target_period_tuple), "Number_By_Appt_Type_Code"]
        .apply(pd.Series)
        .sum()
        .astype(int)
    )


def make_get_counts_of_appts_by_period(
    period_length_years: int,
    target_period_tuple: tuple[Date, Date] = TARGET_PERIOD,
):
    """Create helper that summarizes appointment counts by period chunks + overall."""
    periods = get_periods_within_target_period(
        period_length_years=period_length_years,
        target_period_tuple=target_period_tuple,
    )
    period_lookup = {
        year: period_label
        for period_label, (start_year, end_year) in periods
        for year in range(start_year, end_year + 1)
    }
    target_period_label = target_period(target_period_tuple)

    def _get_counts_of_appts_by_period(_df: pd.DataFrame) -> pd.Series:
        _df_in_target = _df.loc[pd.to_datetime(_df["date"]).between(*target_period_tuple)].copy()
        _df_in_target["year"] = pd.to_datetime(_df_in_target["date"]).dt.year
        _df_in_target["period"] = _df_in_target["year"].map(period_lookup)

        appts = _df_in_target["Number_By_Appt_Type_Code"].apply(pd.Series)
        chunked = appts.groupby(_df_in_target["period"]).sum().T.stack()
        chunked.index = chunked.index.set_names(["appt_type", "period"])

        overall = appts.sum()
        overall.index = pd.MultiIndex.from_arrays(
            [overall.index, np.repeat(target_period_label, len(overall.index))],
            names=["appt_type", "period"],
        )
        return pd.concat([chunked, overall]).astype(int).sort_index()

    return _get_counts_of_appts_by_period


def make_get_counts_of_hsis_by_period(
    period_length_years: int,
    target_period_tuple: tuple[Date, Date] = TARGET_PERIOD,
):
    """Create helper that summarizes appointment counts by period chunks + overall."""
    periods = get_periods_within_target_period(
        period_length_years=period_length_years,
        target_period_tuple=target_period_tuple,
    )
    period_lookup = {
        year: period_label
        for period_label, (start_year, end_year) in periods
        for year in range(start_year, end_year + 1)
    }
    target_period_label = target_period(target_period_tuple)

    def _get_counts_of_hsis_by_period(_df: pd.DataFrame) -> pd.Series:
        _df_in_target = _df.loc[pd.to_datetime(_df["date"]).between(*target_period_tuple)].copy()
        _df_in_target["year"] = pd.to_datetime(_df_in_target["date"]).dt.year
        _df_in_target["period"] = _df_in_target["year"].map(period_lookup)

        hsis = _df_in_target["TREATMENT_ID"].apply(pd.Series)
        chunked = hsis.groupby(_df_in_target["period"]).sum().T.stack()
        chunked.index = chunked.index.set_names(["appt_type", "period"])

        overall = hsis.sum()
        overall.index = pd.MultiIndex.from_arrays(
            [overall.index, np.repeat(target_period_label, len(overall.index))],
            names=["appt_type", "period"],
        )
        return pd.concat([chunked, overall]).astype(int).sort_index()

    return _get_counts_of_hsis_by_period
