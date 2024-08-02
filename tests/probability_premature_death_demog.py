import pickle
from pathlib import Path
import datetime
from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography

resourcefilepath = Path("/Users/rem76/PycharmProjects/TLOmodel/resources")
output_path = Path("/Users/rem76/PycharmProjects/TLOmodel/outputs/probability_premature_death")
logger = logging.getLogger('tlo.methods.demography')
logger.setLevel(logging.INFO)
target_period=(datetime.date(2010, 1, 1), datetime.date(2080, 12, 31))
AGE_BEFORE_WHICH_DEATH_IS_DEFINED_AS_PREMATURE = 70
def test_probability_premature_death(tmpdir):
    # Setup a toy simulation to test probability of dying before 70
    seed = 0
    start_date = Date(2010, 1, 1)
    end_date = Date(2080, 1, 1)  # The simulation will stop before reaching this date.
    pop_size = 100
    number_of_draws = 1
    runs_per_draw = 1


    for draw in range(number_of_draws):
        for sample in range(runs_per_draw):
            draw_dir = output_path / f"{draw}/{sample}"
            draw_dir.mkdir(parents=True, exist_ok=True)

            sim = Simulation(
                start_date=start_date, seed=seed,
                log_config={
                    "filename": "dummy_simulation_prob_premature_death",
                    "directory": draw_dir,
                    'custom_levels': {
                        "*": logging.WARNING,
                        "tlo.methods.demography": logging.INFO
                    }
                }
            )

            sim.register(demography.Demography(resourcefilepath=resourcefilepath))
            sim.modules['Demography'].parameters['max_age_initial'] = 1  # Make everyone just born
            sim.make_initial_population(n=pop_size)
            sim.simulate(end_date=end_date)

            output = parse_log_file(sim.log_filepath)
            df = sim.population.props
            deaths = output["tlo.methods.demography"]["death"]

            for key, output_data in output.items():
                if key.startswith("tlo."):
                    with open(draw_dir / f"{key}.pickle", "wb") as f:
                        pickle.dump(output_data, f)
