import os
from pathlib import Path

import pytest

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file, reconstruct_individual_histories
from tlo.methods import (
    chronicsyndrome,
    individual_history_tracker,
    demography,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthsystem,
    mockitis,
    simplified_births,
    symptommanager,
)

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 200

def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


@pytest.mark.slow
def test_individual_history_tracker(tmpdir, seed):

    # Establish the simulation object
    sim = Simulation(
        start_date=start_date,
        seed=seed,
        log_config={
            "filename": "log",
            "directory": tmpdir,
            "custom_levels": {
                "tlo.methods.healthsystem": logging.DEBUG,
                "tlo.methods.individual_history_tracker": logging.INFO
            }
        }, resourcefilepath=resourcefilepath
    )

    # Register the core modules
    sim.register(demography.Demography(),
                 simplified_births.SimplifiedBirths(),
                 enhanced_lifestyle.Lifestyle(),
                 healthsystem.HealthSystem(),
                 individual_history_tracker.IndividualHistoryTracker(),
                 symptommanager.SymptomManager(),
                 healthseekingbehaviour.HealthSeekingBehaviour(),
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome()
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath, level=logging.DEBUG)
    output_chains = parse_log_file(sim.log_filepath, level=logging.INFO)
    individual_histories = reconstruct_individual_histories(output_chains['tlo.methods.individual_history_tracker']['individual_histories'])
    
    # Check that we have a "StartOfSimulation" event for every individual in the initial population,
    # and that this was logged at the start date
    assert (individual_histories['EventName'] == 'StartOfSimulation').sum() == popsize
    assert (individual_histories.loc[individual_histories['EventName'] == 'StartOfSimulation', 'date'] == start_date).all()
    
    # Check that in the case of birth or start of simulation, all properties were logged
    num_properties = len(sim.population.props.columns)
    mask = individual_histories["EventName"].isin(["Birth", "StartOfSimulation"])
    assert individual_histories.loc[mask, "Info"].apply(len).eq(num_properties).all()
    
    # Assert that all HSI events that occurred were also collected in the event chains
    HSIs_in_individual_histories = individual_histories["EventName"].str.contains('HSI', na=False).sum()
    assert HSIs_in_individual_histories == len(output['tlo.methods.healthsystem']['HSI_Event'])

    # Check that aside from HSIs, StartOfSimulation, and Birth, other events were collected too
    mask = (~individual_histories["EventName"].isin(["StartOfSimulation", "Birth"])) & \
           (~individual_histories["EventName"].str.contains("HSI", na=False))
    count = mask.sum()
    assert count > 0

