import os
from pathlib import Path

import pytest

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file, reconstruct_event_chains
from tlo.methods import (
    chronicsyndrome,
    collect_event_chains,
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
def test_collection_of_event_chains(tmpdir, seed):

    # Establish the simulation object
    sim = Simulation(
        start_date=start_date,
        seed=seed,
        log_config={
            "filename": "log",
            "directory": tmpdir,
            "custom_levels": {
                "tlo.methods.healthsystem": logging.DEBUG,
                "tlo.methods.collect_event_chains": logging.INFO
            }
        }, resourcefilepath=resourcefilepath
    )

    # Register the core modules
    sim.register(demography.Demography(),
                 simplified_births.SimplifiedBirths(),
                 enhanced_lifestyle.Lifestyle(),
                 healthsystem.HealthSystem(),
                 collect_event_chains.CollectEventChains(generate_event_chains=True),
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
    event_chains = reconstruct_event_chains(output_chains['tlo.methods.collect_event_chains']['event_chains'])
    
    # Check that we have a "StartOfSimulation" event for every individual in the initial population,
    # and that this was logged at the start date
    assert (event_chains['EventName'] == 'StartOfSimulation').sum() == popsize
    assert (event_chains.loc[event_chains['EventName'] == 'StartOfSimulation', 'date'] == start_date).all()
    
    # Check that in the case of birth or start of simulation, all properties were logged
    num_properties = len(sim.population.props.columns)
    mask = event_chains["EventName"].isin(["Birth", "StartOfSimulation"])
    assert event_chains.loc[mask, "Info"].apply(len).eq(num_properties).all()
    
    # Assert that all HSI events that occurred were also collected in the event chains
    HSIs_in_event_chains = event_chains["EventName"].str.contains('HSI', na=False).sum()
    assert HSIs_in_event_chains == len(output['tlo.methods.healthsystem']['HSI_Event'])

    # Check that aside from HSIs, StartOfSimulation, and Birth, other events were collected too
    mask = (~event_chains["EventName"].isin(["StartOfSimulation", "Birth"])) & \
           (~event_chains["EventName"].str.contains("HSI", na=False))
    count = mask.sum()
    assert count > 0

