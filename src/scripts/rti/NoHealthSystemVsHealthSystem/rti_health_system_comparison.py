from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
    rti,
)
import numpy as np
from matplotlib import pyplot as plt

log_config = {
    "filename": "rti_health_system_comparison",  # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.rti": logging.INFO,
        "tlo.methods.healthsystem": logging.DEBUG
    }
}
# The Resource files [NB. Working directory must be set to the root of TLO: TLOmodel]
resourcefilepath = Path('./resources')
# Establish the simulation object
yearsrun = 2
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
pop_size = 5000

service_availability = ["*"]
sim = Simulation(start_date=start_date)
# We register all modules in a single call to the register method, calling once with multiple
# objects. This is preferred to registering each module in multiple calls because we will be
# able to handle dependencies if modules are registered together
sim.register(
    demography.Demography(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_availability),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    rti.RTI(resourcefilepath=resourcefilepath)
)
logfile = sim.configure_logging(filename="LogFile")
# create and run the simulation
sim.make_initial_population(n=pop_size)
params = sim.modules['RTI'].parameters
params['allowed_interventions'] = []
sim.simulate(end_date=end_date)

# parse the simulation logfile to get the output dataframes
log_df_with_health_system = parse_log_file(logfile)
summary_1m = log_df_with_health_system['tlo.methods.rti']['summary_1m']
plt.plot(summary_1m['date'], summary_1m['number involved in a rti'], label='Number in rti accidents')
plt.plot(summary_1m['date'], summary_1m['number immediate deaths'], label='Number of deaths on scene')
plt.plot(summary_1m['date'], summary_1m['number permanently disabled'], label='Number permanently disabled')
plt.plot(summary_1m['date'], summary_1m['total injuries'], label='Total injuries')
plt.legend()
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.savefig('outputs/HealthSystemComparison/summary_1m_with_health_system', bbox_inches='tight')
plt.clf()
plt.close()
rti_demography = log_df_with_health_system['tlo.methods.rti']['rti_demography']
gender_data_raw = [sum(rti_demography['males_in_rti']), sum(rti_demography['females_in_rti'])]
gender_data_percentage = np.divide(gender_data_raw, gender_data_raw[0] + gender_data_raw[1])
plt.bar(np.arange(2), height=gender_data_percentage, color='lightsteelblue')
plt.xticks(np.arange(2), ['males', 'females'])
plt.ylabel('Percentage of those in road traffic accidents')
plt.savefig('outputs/HealthSystemComparison/demography_gender_in_rti_with_health_system', bbox_inches='tight')
plt.clf()
plt.close()
injury_characteristics = log_df_with_health_system['tlo.methods.rti']['injury_characteristics']
final_characteristics_of_sim = injury_characteristics.iloc[-1]
final_characteristics_of_sim = final_characteristics_of_sim.drop('date')
final_characteristics_of_sim = final_characteristics_of_sim.divide(sum(final_characteristics_of_sim))
plt.bar(np.arange(len(final_characteristics_of_sim)), final_characteristics_of_sim.values, color='lightsteelblue')
plt.xticks(np.arange(len(final_characteristics_of_sim)), final_characteristics_of_sim.keys().tolist(), rotation=45)
plt.savefig('outputs/HealthSystemComparison/injury_characteristics_with_health_system', bbox_inches='tight')
plt.clf()
plt.close()
injury_location_data = log_df_with_health_system['tlo.methods.rti']['injury_location_data']
injury_location_of_sim = injury_location_data.iloc[-1]
injury_location_of_sim = injury_location_of_sim.drop('date')
injury_location_of_sim = injury_location_of_sim.divide(sum(injury_location_of_sim))
plt.bar(np.arange(len(injury_location_of_sim)), injury_location_of_sim.values, color='lightsteelblue')
plt.xticks(np.arange(len(injury_location_of_sim)), injury_location_of_sim.keys().tolist(), rotation=45)
plt.savefig('outputs/HealthSystemComparison/injury_locations_with_health_system', bbox_inches='tight')
plt.clf()
plt.close()
number_of_injuries = log_df_with_health_system['tlo.methods.rti']['number_of_injuries']
final_n_inj_dist = number_of_injuries.iloc[-1]
final_n_inj_dist = final_n_inj_dist.drop('date')
plt.plot(np.arange(len(final_n_inj_dist)) + 1, final_n_inj_dist / sum(final_n_inj_dist))
plt.xlabel('Number of injured body regions')
plt.ylabel('Percent')
plt.savefig('outputs/HealthSystemComparison/number_of_injury_with_health_system', bbox_inches='tight')
plt.clf()
plt.close()
injury_severity = log_df_with_health_system['tlo.methods.rti']['injury_severity']
ISS_distribution = injury_severity['ISS_score']
ISS_distribution = ISS_distribution.iloc[-1]
ISS_distribution = np.unique(ISS_distribution, return_counts=True)
plt.plot(ISS_distribution[0], np.divide(ISS_distribution[1], sum(ISS_distribution[1])))
plt.xlim([0, 75])
plt.xlabel('ISS score')
plt.ylabel('Percentage')
plt.savefig('outputs/HealthSystemComparison/ISS_distribution_with_health_system', bbox_inches='tight')
plt.clf()
plt.close()
burn_location_data = log_df_with_health_system['tlo.methods.rti']['burn_location_data']
laceration_location_data = log_df_with_health_system['tlo.methods.rti']['laceration_location_data']
fracture_location_data = log_df_with_health_system['tlo.methods.rti']['fracture_location_data']
model_progression = log_df_with_health_system['tlo.methods.rti']['model_progression']
# todo: make comparison of total deaths with and without med int
injuries_of_those_who_died = log_df_with_health_system['tlo.methods.rti']['RTI_Death_Injury_Profile']
pain_information = log_df_with_health_system['tlo.methods.rti']['pain_information']
Requested_Pain_Management = log_df_with_health_system['tlo.methods.rti']['Requested_Pain_Management']
Successful_Pain_Management = log_df_with_health_system['tlo.methods.rti']['Successful_Pain_Management']

service_availability = []
sim = Simulation(start_date=start_date)
# We register all modules in a single call to the register method, calling once with multiple
# objects. This is preferred to registering each module in multiple calls because we will be
# able to handle dependencies if modules are registered together
sim.register(
    demography.Demography(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_availability),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    rti.RTI(resourcefilepath=resourcefilepath)
)
logfile = sim.configure_logging(filename="LogFile")
# create and run the simulation
sim.make_initial_population(n=pop_size)
params = sim.modules['RTI'].parameters
params['allowed_interventions'] = []
sim.simulate(end_date=end_date)

# parse the simulation logfile to get the output dataframes
log_df_without_health_system = parse_log_file(logfile)
summary_1m = log_df_without_health_system['tlo.methods.rti']['summary_1m']
plt.plot(summary_1m['date'], summary_1m['number involved in a rti'], label='Number in rti accidents')
plt.plot(summary_1m['date'], summary_1m['number immediate deaths'], label='Number of deaths on scene')
plt.plot(summary_1m['date'], summary_1m['number permanently disabled'], label='Number permanently disabled')
plt.plot(summary_1m['date'], summary_1m['total injuries'], label='Total injuries')
plt.legend()
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.savefig('outputs/HealthSystemComparison/summary_1m_without_health_system', bbox_inches='tight')
plt.clf()
plt.close()
rti_demography = log_df_without_health_system['tlo.methods.rti']['rti_demography']
gender_data_raw = [sum(rti_demography['males_in_rti']), sum(rti_demography['females_in_rti'])]
gender_data_percentage = np.divide(gender_data_raw, gender_data_raw[0] + gender_data_raw[1])
plt.bar(np.arange(2), height=gender_data_percentage, color='lightsteelblue')
plt.xticks(np.arange(2), ['males', 'females'])
plt.ylabel('Percentage of those in road traffic accidents')
plt.savefig('outputs/HealthSystemComparison/demography_gender_in_rti_without_health_system', bbox_inches='tight')
plt.clf()
plt.close()
injury_characteristics = log_df_without_health_system['tlo.methods.rti']['injury_characteristics']
final_characteristics_of_sim = injury_characteristics.iloc[-1]
final_characteristics_of_sim = final_characteristics_of_sim.drop('date')
final_characteristics_of_sim = final_characteristics_of_sim.divide(sum(final_characteristics_of_sim))
plt.bar(np.arange(len(final_characteristics_of_sim)), final_characteristics_of_sim.values, color='lightsteelblue')
plt.xticks(np.arange(len(final_characteristics_of_sim)), final_characteristics_of_sim.keys().tolist(), rotation=45)
plt.savefig('outputs/HealthSystemComparison/injury_characteristics_without_health_system', bbox_inches='tight')
plt.clf()
plt.close()
injury_location_data = log_df_without_health_system['tlo.methods.rti']['injury_location_data']
injury_location_of_sim = injury_location_data.iloc[-1]
injury_location_of_sim = injury_location_of_sim.drop('date')
injury_location_of_sim = injury_location_of_sim.divide(sum(injury_location_of_sim))
plt.bar(np.arange(len(injury_location_of_sim)), injury_location_of_sim.values, color='lightsteelblue')
plt.xticks(np.arange(len(injury_location_of_sim)), injury_location_of_sim.keys().tolist(), rotation=45)
plt.savefig('outputs/HealthSystemComparison/injury_locations_without_health_system', bbox_inches='tight')
plt.clf()
plt.close()
number_of_injuries = log_df_without_health_system['tlo.methods.rti']['number_of_injuries']
final_n_inj_dist = number_of_injuries.iloc[-1]
final_n_inj_dist = final_n_inj_dist.drop('date')
plt.plot(np.arange(len(final_n_inj_dist)) + 1, final_n_inj_dist / sum(final_n_inj_dist))
plt.xlabel('Number of injured body regions')
plt.ylabel('Percent')
plt.savefig('outputs/HealthSystemComparison/number_of_injury_without_health_system', bbox_inches='tight')
plt.clf()
plt.close()
injury_severity = log_df_with_health_system['tlo.methods.rti']['injury_severity']
ISS_distribution = injury_severity['ISS_score']
ISS_distribution = ISS_distribution.iloc[-1]
ISS_distribution = np.unique(ISS_distribution, return_counts=True)
plt.plot(ISS_distribution[0], np.divide(ISS_distribution[1], sum(ISS_distribution[1])))
plt.xlim([0, 75])
plt.xlabel('ISS score')
plt.ylabel('Percentage')
plt.savefig('outputs/HealthSystemComparison/ISS_distribution_without_health_system', bbox_inches='tight')
plt.clf()
plt.close()
burn_location_data = log_df_without_health_system['tlo.methods.rti']['burn_location_data']
laceration_location_data = log_df_without_health_system['tlo.methods.rti']['laceration_location_data']
fracture_location_data = log_df_without_health_system['tlo.methods.rti']['fracture_location_data']
model_progression = log_df_without_health_system['tlo.methods.rti']['model_progression']
pain_information = log_df_without_health_system['tlo.methods.rti']['pain_information']

# Compare deaths
deaths_without_med = log_df_without_health_system['tlo.methods.demography']['death']
tot_death_without_med = len(deaths_without_med.loc[(deaths_without_med['cause'] != 'Other')])
tot_death_on_scene_without_med = len(deaths_without_med.loc[(deaths_without_med['cause'] == 'RTI_imm_death')])
tot_death_due_to_no_health_sys = len(deaths_without_med.loc[(deaths_without_med['cause'] == 'RTI_death_without_med')])
deaths_with_med = log_df_with_health_system['tlo.methods.demography']['death']
tot_death_with_med = len(deaths_with_med.loc[(deaths_with_med['cause'] != 'Other')])
tot_death_on_scene_with_med = len(deaths_with_med.loc[(deaths_with_med['cause'] == 'RTI_imm_death')])
tot_death_after_med = len(deaths_with_med.loc[(deaths_with_med['cause'] == 'RTI_death_with_med')])
tot_death_due_to_delayed_med = len(deaths_with_med.loc[(deaths_with_med['cause'] == 'RTI_unavailable_med')])
barlist = plt.bar([1, 2, 3, 4, 5, 6, 7],
                  [tot_death_without_med, tot_death_on_scene_without_med, tot_death_due_to_no_health_sys,
                   tot_death_with_med, tot_death_on_scene_with_med, tot_death_after_med, tot_death_due_to_delayed_med])
barlist[0].set_color('r')
barlist[1].set_color('r')
barlist[2].set_color('r')
plt.xticks([0, 1, 2, 3, 4, 5, 6], ['Total deaths due to RTI no med', 'Deaths on scene no med',
                                   'Deaths due to no health system',
                                   'Total deaths due to RTI with med', 'Deaths on scene with med',
                                   'Deaths despite care', 'Deaths due to delayed care'], rotation=45)
plt.savefig('outputs/HealthSystemComparison/compare_deaths_with_without_health_sys', bbox_inches='tight')
plt.close()
plt.clf()
