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
start_date = Date(2026, 1, 1)
end_date = Date(2027, 1, 12)
popsize = 10000

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
# def check_dtypes(simulation):
#     # check types of columns
#     df = simulation.population.props
#     orig = simulation.population.new_row
#     assert (df.dtypes == orig.dtypes).all()
#
#
# def test_setting_climate_disruptions(seed):
#     """Check that the switches for turning on/off climate disruptions to healthcare access work"""
#     sim = Simulation(start_date=start_date, seed=seed)
#     sim.register(demography.Demography(resourcefilepath=resourcefilepath),
#                  contraception.Contraception(resourcefilepath=resourcefilepath),
#                  enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
#                  healthburden.HealthBurden(resourcefilepath=resourcefilepath),
#                  healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
#                                            service_availability=['*'],
#                                            cons_availability='all'),  # went set disable=true, cant check HSI queue,
#                  newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
#                  pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
#                  care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
#                  symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
#                  labour.Labour(resourcefilepath=resourcefilepath),
#                  postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
#                  healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
#
#                  hiv.DummyHivModule())
#
#     sim.make_initial_population(n=popsize)
#     sim.simulate(end_date=end_date)
#     assert sim.modules['HealthSystem'].services_affected_precip == 'none'
#
#
# def test_setting_climate_disruptions(seed):
#     """Check that the switches for turning on/off climate disruptions to healthcare access work"""
#     sim = Simulation(start_date=start_date, seed=seed)
#     sim.register(
#         demography.Demography(resourcefilepath=resourcefilepath),
#         healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
#                                   climate_ssp = 'ssp126',
#                                   climate_model_ensemble_model='mean',
#                                   services_affected_precip = 'ANC')
#     )
#     sim.make_initial_population(n=popsize)
#     sim.simulate(end_date=end_date)
#     assert sim.modules['HealthSystem'].services_affected_precip == 'ANC'


def test_number_services(seed, tmpdir):
    """Checks to see if the number of services under the climate disrupted scenario
    are fewer than the no-climate disruption scenario"""
    sim = Simulation(start_date=start_date,
                     seed=seed,
                     log_config={
                     "filename": "log",
                     "directory": tmpdir,
                     "custom_levels": {
                     "tlo.methods.healthsystem.summary": logging.DEBUG,
            }})
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),

        healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                  climate_ssp = 'ssp585',
                                  climate_model_ensemble_model='mean',
                                  services_affected_precip = 'ANC'),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        mockitis.Mockitis(),
        chronicsyndrome.ChronicSyndrome(),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    )
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    assert sim.modules['HealthSystem'].services_affected_precip == 'ANC'
    #output_climate = parse_log_file(sim.log_filepath)
    hsi_event_count_df_climate = get_dataframe_of_run_events_count(sim)

    sim_no_climate = Simulation(start_date=start_date,
                     seed=seed,
                     log_config={
                     "filename": "log",
                     "directory": tmpdir,
                     "custom_levels": {
                     "tlo.methods.healthsystem": logging.DEBUG,
            }})
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
    assert sim_no_climate.modules['HealthSystem'].services_affected_precip == 'none'
    hsi_event_count_df_no_climate = get_dataframe_of_run_events_count(sim_no_climate)


    assert sum(hsi_event_count_df_climate['count']) < sum(hsi_event_count_df_no_climate['count'])
