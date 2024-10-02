import datetime
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from tlo.analysis.life_expectancy import (
    get_life_expectancy_estimates,
    get_probability_of_premature_death,
)

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


def test_probability_premature_death(tmpdir, age_before_which_death_is_defined_as_premature: int = 70):
    """
    Test the calculation of the probability of premature death from a simulated cohort.

    This function loads results from a dummy cohort (N = 100, with 37 F and 63 M) simulation where all individuals start
    at age 0. The simulation was then run for 70 years (2010 - 2080), during which individuals could die but nobody
    could be born. In this dummy data set, 6 F die and 23 M die prematurely, giving a probability of premature death as
    0.16 and 0.37, respectively. The premature deaths amongst these individuals is then the number that have died
    before the age of 70 (default value).
    This test uses the calculates the probability of premature death separately for males and females using the
    data from this simulated run and the function get_probability_of_premature_death.
    It then compares these simulated probabilities against the total number of deaths before the age of 70 (default)
    that occurred in the simulated cohort.
    """
    # load results from a dummy cohort where everyone starts at age 0.
    target_period = (datetime.date(2010, 1, 1), datetime.date(2080, 12, 31))

    results_folder_dummy_results = Path(os.path.dirname(__file__)) / 'resources' / 'probability_premature_death'
    pickled_file = os.path.join(results_folder_dummy_results, '0', '0', 'tlo.methods.demography.pickle')

    # - Compute 'manually' from raw data
    with open(pickled_file, 'rb') as file:
        demography_data = pickle.load(file)
    initial_popsize = {'F':  demography_data['population']['female'][0], 'M': demography_data['population']['male'][0]}
    deaths_total = demography_data['death'][['sex', 'age']]
    num_premature_deaths = deaths_total.loc[deaths_total['age'] < age_before_which_death_is_defined_as_premature] \
                                       .groupby('sex') \
                                       .size() \
                                       .to_dict()
    prob_premature_death = {s: num_premature_deaths[s] / initial_popsize[s] for s in ("M", "F")}

    # - Compute using utility function
    probability_premature_death_summary = get_probability_of_premature_death(
        results_folder=results_folder_dummy_results,
        target_period=target_period,
        summary=True,
    )

    # Confirm both methods gives the same answer
    # (Absolute tolerance of this test is reasonably large (1%) as small assumptions made in the calculation of the
    # cumulative probability of death in each age-group mean that the manual computation done here and the calculation
    # performed in the utility function are not expected to agree perfectly.)
    assert np.isclose(
        probability_premature_death_summary.loc["F"].loc[(0, 'mean')],
        prob_premature_death['F'],
        atol=0.01
    )
    assert np.isclose(
        probability_premature_death_summary.loc["M"].loc[(0, 'mean')],
        prob_premature_death['M'],
        atol=0.01
    )
