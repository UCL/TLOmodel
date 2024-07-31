import datetime
import os
from pathlib import Path

from tlo.analysis.life_expectancy import get_probability_of_premature_death
from tlo.analysis.utils import extract_results

target_period=(datetime.date(2010, 1, 1), datetime.date(2080, 12, 31))
AGE_BEFORE_WHICH_DEATH_IS_DEFINED_AS_PREMATURE = 70

def test_probability_premature_death(tmpdir):

    # load results from a dummy cohort where everyone starts at age 0.
    results_folder_dummy_results = Path(os.path.dirname(__file__)) / 'resources' / 'probability_premature_death'

                # test parsing when log level is INFO
    total_female = extract_results(
        results_folder_dummy_results,
        module="tlo.methods.demography",
        key="population",
        column='female',
        do_scaling=False
    )
    deaths = extract_results(
        results_folder_dummy_results,
        module="tlo.methods.demography",
        key="death",
        column="sex",
        do_scaling=False
    )
    total_male = extract_results(
        results_folder_dummy_results,
        module="tlo.methods.demography",
        key="population",
        column='male',
        do_scaling=False
    )

    probability_premature_death_sim_F = len(deaths[deaths[0] == 'F'])/len(total_female)
    probability_premature_death_sim_M = len(deaths[deaths[0] == 'M'])/len(total_male)

     #Summary measure: Should have row ('M', 'F') and columns ('mean', 'lower', 'upper')
    probability_premature_death_summary = get_probability_of_premature_death(
        results_folder=results_folder_dummy_results,
        target_period=target_period,
        summary=True,)

    assert probability_premature_death_summary[0]['lower'][0] < probability_premature_death_sim_M > probability_premature_death_summary[0]['lower'][0]
    assert probability_premature_death_summary[0]['lower'][1] < probability_premature_death_sim_F > probability_premature_death_summary[0]['lower'][1]






