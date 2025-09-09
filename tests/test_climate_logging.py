import os
from pathlib import Path
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.methods import (
    chronicsyndrome,
    demography,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthsystem,
    mockitis,
    simplified_births,
    symptommanager,
)
from tlo.analysis.utils import compare_number_of_deaths, parse_log_file

start_date = Date(2010, 1, 1)
end_date = Date(2029, 1, 12)
popsize = 1000

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
def get_dataframe_of_run_events_count(_sim):
    """Return a dataframe of event counts with info of treatment id, appointment footprint."""
    count_df = pd.DataFrame(index=range(len(_sim.modules['HealthSystem'].hsi_event_counts)))
    count_df['HSI_event'] = [event_details.event_name
                             for event_details in _sim.modules['HealthSystem'].hsi_event_counts.keys()]
    count_df['treatment_id'] = [event_details.treatment_id
                                for event_details in _sim.modules['HealthSystem'].hsi_event_counts.keys()]
    count_df['appt_footprint'] = [event_details.appt_footprint
                                  for event_details in _sim.modules['HealthSystem'].hsi_event_counts.keys()]
    count_df['count'] = [_sim.modules['HealthSystem'].hsi_event_counts[event_details]
                         for event_details in _sim.modules['HealthSystem'].hsi_event_counts.keys()]

    return count_df

def get_dataframe_of_run_events_delayed(_sim):
    """Return a dataframe of event counts with info of treatment id, appointment footprint."""
    count_df = parse_log_file(_sim.log_filepath)['tlo.methods.healthsystem.summary']
    count_df = count_df['weather_delayed_hsi_event_counts']
    total = 0

    for d in count_df["weather_delayed_hsi_event_key_to_counts"]:
        total += sum(d.values())
    return int(total)

def test_number_services(seed, tmpdir):
    """Checks to see if the number of services under the climate disrupted scenario
    are fewer than the no-climate disruption scenario"""

    # 585 highest
    sim = Simulation(start_date=start_date,
                     seed=seed,
                     log_config={
                     "filename": "log",
                     "directory": tmpdir,
                     })
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),

        healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                  climate_ssp = 'ssp585',
                                  climate_model_ensemble_model='mean',
                                  services_affected_precip = 'all'),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        mockitis.Mockitis(),
        chronicsyndrome.ChronicSyndrome(),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    )
    #assert sim.modules['HealthSystem'].parameters['services_affected_precip'] == 'all'

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    #output_climate = parse_log_file(sim.log_filepath)
    hsi_event_count_df_climate = get_dataframe_of_run_events_delayed(sim)

    # 126 lowest
    sim_low = Simulation(start_date=start_date,
                     seed=seed,
                     log_config={
                         "filename": "log",
                         "directory": tmpdir})
    sim_low.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),

        healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                  climate_ssp='ssp126',
                                  climate_model_ensemble_model='lowest',
                                  services_affected_precip='all'),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        mockitis.Mockitis(),
        chronicsyndrome.ChronicSyndrome(),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    )
    sim_low.make_initial_population(n=popsize)
    sim_low.simulate(end_date=end_date)
    #assert sim_low.modules['HealthSystem'].services_affected_precip == 'all'
    #assert sim_low.modules['HealthSystem'].climate_ssp == 'ssp126'

    # output_climate = parse_log_file(sim.log_filepath)
    hsi_event_count_df_climate_126 = get_dataframe_of_run_events_delayed(sim_low)

    # no climate
    sim_no_climate = Simulation(start_date=start_date,
                     seed=seed,
                     log_config={
                     "filename": "log",
                     "directory": tmpdir,
})
    sim_no_climate.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),

        healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                  climate_ssp = 'ssp585',
                                  climate_model_ensemble_model='mean',
                                  services_affected_precip = 'none'),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        mockitis.Mockitis(),
        chronicsyndrome.ChronicSyndrome(),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    )
    sim_no_climate.make_initial_population(n=popsize)
    sim_no_climate.simulate(end_date=end_date)
    #output_no_climate = parse_log_file(sim.log_filepath)
    df = sim.population.props
    #assert sim_no_climate.modules['HealthSystem'].services_affected_precip == 'none'
    hsi_event_count_df_no_climate = get_dataframe_of_run_events_delayed(sim_no_climate)

    #assert 0 > sum(hsi_event_count_df_no_climate['count'])
    print("hsi_event_count_df_climate", hsi_event_count_df_climate)
    print("hsi_event_count_df_no_climate", hsi_event_count_df_no_climate)

    assert (hsi_event_count_df_climate) < (hsi_event_count_df_no_climate)

    assert (hsi_event_count_df_climate) < (hsi_event_count_df_no_climate)
    assert (hsi_event_count_df_climate_126) < (hsi_event_count_df_no_climate)

    assert (hsi_event_count_df_climate) < (hsi_event_count_df_climate_126)
