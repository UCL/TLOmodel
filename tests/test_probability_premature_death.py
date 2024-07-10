
import datetime
import os
from pathlib import Path

import pandas as pd

from tlo.analysis.probability_premature_death import get_probability_of_dying_before_70


def test_get_probability_premature_death():
    """Use `get_probability_of_dying_before_70` to generate estimate of probability of preamture death from the dummy simulation data."""

    results_folder_dummy_results = Path(os.path.dirname(__file__)) / 'resources' / 'dummy_simulation_run'

    # Summary measure: Should have row ('M', 'F') and columns ('mean', 'lower', 'upper')
    rtn_summary = get_probability_of_dying_before_70(
        results_folder=results_folder_dummy_results,
        target_period=(datetime.date(2010, 1, 1), datetime.date(2010, 12, 31)),
        summary=True,
    )
    assert isinstance(rtn_summary, pd.DataFrame)
    assert sorted(rtn_summary.index.to_list()) == ["F", "M"]
    assert list(rtn_summary.columns.names) == ['draw', 'stat']
    assert rtn_summary.columns.levels[1].to_list() == ["lower", "mean", "upper"]

    # Non-summary measure: Estimate should be for each run/draw
    rtn_full = get_probability_of_dying_before_70(
        results_folder=results_folder_dummy_results,
        target_period=(datetime.date(2010, 1, 1), datetime.date(2010, 12, 31)),
        summary=False,
    )
    assert isinstance(rtn_full, pd.DataFrame)
    assert sorted(rtn_full.index.to_list()) == ["F", "M"]
    assert list(rtn_full.columns.names) == ['draw', 'run']
    assert rtn_full.columns.levels[1].to_list() == [0, 1]
