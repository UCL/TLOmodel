import datetime
import os
from pathlib import Path

import pandas as pd

from tlo.analysis.life_expectancy import get_life_expectancy_estimates


def test_get_life_expectancy():
    """Use `get_life_expectancy_estimates` to generate estimate of life-expectancy from the dummy simulation data."""

    results_folder_dummy_results = Path(os.path.dirname(__file__)) / 'resources' / 'dummy_simulation_run'

    # Summary measure: Should have row ('M', 'F') and columns ('mean', 'lower', 'upper')
    rtn_summary = get_life_expectancy_estimates(
        results_folder=results_folder_dummy_results,
        target_period=(datetime.date(2010, 1, 1), datetime.date(2010, 12, 31)),
        summary=True,
    )
    assert isinstance(rtn_summary, pd.DataFrame)
    assert sorted(rtn_summary.index.to_list()) == ["F", "M"]
    assert list(rtn_summary.columns.names) == ['draw', 'stat']
    assert rtn_summary.columns.levels[1].to_list() == ["lower", "mean", "upper"]

    # Non-summary measure: Estimate should be for each run/draw
    rtn_full = get_life_expectancy_estimates(
        results_folder=results_folder_dummy_results,
        target_period=(datetime.date(2010, 1, 1), datetime.date(2010, 12, 31)),
        summary=False,
    )
    assert isinstance(rtn_full, pd.DataFrame)
    assert sorted(rtn_full.index.to_list()) == ["F", "M"]
    assert list(rtn_full.columns.names) == ['draw', 'run']
    assert rtn_full.columns.levels[1].to_list() == [0, 1]
