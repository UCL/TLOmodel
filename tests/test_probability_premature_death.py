import datetime
import pickle
import statistics
from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.life_expectancy import get_probability_of_premature_death
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography

target_period=(datetime.date(2010, 1, 1), datetime.date(2080, 12, 31))
AGE_BEFORE_WHICH_DEATH_IS_DEFINED_AS_PREMATURE = 70
resourcefilepath = Path("./resources")
outputpath = Path("/Users/rem76/PycharmProjects/TLOmodel/outputs/probability_premature_death")
logger = logging.getLogger('tlo.methods.demography')
logger.setLevel(logging.INFO)


def test_probability_premature_death(tmpdir):
    # setup a toy simulation to test probability of dying before 70
    seed = 0
    start_date = Date(2010, 1, 1)
    end_date = Date(2080, 1, 1)  # The simulation will stop before reaching this date.
    pop_size = 1000
    number_of_draws = 1
    runs_per_draw = 2
    probability_premature_death_sim_F = []
    probability_premature_death_sim_M = []
    for draw in range(0, number_of_draws):
        for sample in range(0, runs_per_draw):
                draw_dir = outputpath / f"{draw}/{sample}"
                draw_dir.mkdir(parents=True, exist_ok=True)
                # test parsing when log level is INFO
                sim = Simulation(
                    start_date=start_date, seed=seed,
                    log_config={"filename": "dummy_simulation_prob_premature_death", "directory": draw_dir,
                            'custom_levels': {
                            "*": logging.WARNING,
                            "tlo.methods.demography": logging.INFO}})

                sim.register(demography.Demography(resourcefilepath=resourcefilepath))
                sim.modules['Demography'].parameters['max_age_initial'] = 1 # make everyone just born
                sim.make_initial_population(n=pop_size)
                sim.simulate(end_date=end_date)
                output = parse_log_file(sim.log_filepath)
                df = sim.population.props
                deaths = output["tlo.methods.demography"]["death"]
                for key, output in output.items():
                    if key.startswith("tlo."):
                        with open(draw_dir / f"{key}.pickle", "wb") as f:
                            pickle.dump(output, f)
                probability_premature_death_sim_F.append(len(deaths[deaths['sex'] == 'F'])/len(df[df['sex'] == 'F']))
                probability_premature_death_sim_M.append(len(deaths[deaths['sex'] == 'M'])/len(df[df['sex'] == 'M']))

     #Summary measure: Should have row ('M', 'F') and columns ('mean', 'lower', 'upper')
    rtn_summary = get_probability_of_premature_death(
        results_folder=outputpath,
        target_period=target_period,
        summary=True,
    )
    assert rtn_summary[0]['lower'][0] < statistics.mean(probability_premature_death_sim_M) > rtn_summary[0]['lower'][0]
    assert rtn_summary[0]['lower'][1] < statistics.mean(probability_premature_death_sim_F) > rtn_summary[0]['lower'][1]






