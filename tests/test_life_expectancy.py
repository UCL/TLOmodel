import datetime
import os
import pickle
from pathlib import Path

import pandas as pd

from tlo.analysis.life_expectancy import (
    get_life_expectancy_estimates,
    get_probability_of_premature_death,
)
from tlo.analysis.utils import extract_results


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


def test_probability_premature_death(tmpdir, age_before_which_death_is_defined_as_premature: int =70):
    """
    Test the calculation of the probability of premature death from a simulated cohort.

    This function loads results from a dummy cohort (N = 100, with 37 F and 63 M) simulation where all individuals start at age 0.
    The simulation was then run for 70 years (2010 - 2080), during which individuals could die but nobody could be born.
    In this dummy data set, 7 F die and 24 M die, giving a probability of premature death as 0.189 and 0.381 respectively.
    The premature deaths amongst these individuals is then the number that have died before the age of 70 (default value).
    This test uses the calculates the probability of premature death separately for males and females using the
    data from this simulated run and the function get_probability_of_premature_death.
    It then compares these simulated probabilities against the total number of deaths before the age of 70 (default)
    that occurred in the simulated cohort.

    """
    # load results from a dummy cohort where everyone starts at age 0.
    target_period = (datetime.date(2010, 1, 1), datetime.date(2080, 12, 31))

    results_folder_dummy_results = Path(os.path.dirname(__file__)) / 'resources' / 'probability_premature_death'
    with open('/Users/rem76/PycharmProjects/TLOmodel/tests/resources/probability_premature_death/0/0/tlo.methods.demography.pickle', 'rb') as file:
        demography_data = pickle.load(file)
                # test parsing when log level is INFO
    initial_female = demography_data['population']['female'][0]
    initial_male = demography_data['population']['male'][0]

    death_sex = extract_results(
        results_folder_dummy_results,
        module="tlo.methods.demography",
        key="death",
        column="sex",
        do_scaling=False
    )
    death_age = extract_results(
        results_folder_dummy_results,
        module="tlo.methods.demography",
        key="death",
        column="age",
        do_scaling=False
    )
    deaths_total= pd.DataFrame({
    'Sex': death_sex[0][0],
    'Age': death_age[0][0]}, index = range(len(death_sex)))

    probability_premature_death_sim_F = len(deaths_total[
    (deaths_total['Sex'] == 'F') &
    (deaths_total['Age'].astype(int) <= age_before_which_death_is_defined_as_premature)])/ initial_female
    probability_premature_death_sim_M = len(deaths_total[(deaths_total['Sex'] == 'M') & (
            deaths_total['Age'].astype(int) <= age_before_which_death_is_defined_as_premature)]) / initial_male

    probability_premature_death_summary = get_probability_of_premature_death(
        results_folder=results_folder_dummy_results,
        target_period=target_period,
        summary=True,)


    assert probability_premature_death_summary[0]['lower'][1] < probability_premature_death_sim_M > probability_premature_death_summary[0]['lower'][1]

    assert probability_premature_death_summary[0]['lower'][0] < probability_premature_death_sim_F > probability_premature_death_summary[0]['lower'][0]
    assert probability_premature_death_summary[0]['lower'][1] < probability_premature_death_sim_M > probability_premature_death_summary[0]['lower'][1]


