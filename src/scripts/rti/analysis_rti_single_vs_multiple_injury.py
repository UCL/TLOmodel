import numpy as np
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    rti,
    simplified_births,
    symptommanager,
)

# =============================== Analysis description ========================================================
# This analysis file will eventually become what I use to produce the introduction to RTI paper. Here I run the model
# initally only allowing one injury per person, capturing the incidence of RTI and incidence of RTI death, calibrating
# this to the GBD estimates. I then run the model with multiple injuries and compare the outputs, the question being
# asked here is what happens to road traffic injury deaths if we allow multiple injuries to occur

# ============================================== Model run ============================================================
log_config_single = {
    "filename": "rti_single_injury",  # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.rti": logging.INFO,
        "tlo.methods.healthsystem": logging.DEBUG,
        "tlo.methods.healthburden": logging.INFO
    }
}
# The Resource files [NB. Working directory must be set to the root of TLO: TLOmodel]
# resourcefilepath = Path('./resources')
resourcefilepath = './resources'

# Establish the simulation object
yearsrun = 10
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
service_availability = ['*']
pop_size = 10000
seed = 100

# Create the simulation object
sim_single_injury = Simulation(start_date=start_date, seed=seed, log_config=log_config_single,
                               resourcefilepath=resourcefilepath)

# Register the modules
sim_single_injury.register(
    demography.Demography(),
    enhanced_lifestyle.Lifestyle(),
    healthsystem.HealthSystem(service_availability=['*']),
    symptommanager.SymptomManager(),
    healthseekingbehaviour.HealthSeekingBehaviour(),
    healthburden.HealthBurden(),
    rti.RTI(),
    simplified_births.SimplifiedBirths(),
)
# Get the log file

# create and run the simulation
sim_single_injury.make_initial_population(n=pop_size)
# alter the number of injuries given out
# Injury vibes number of GBD injury category distribution:
number_inj_data = [1, 0, 0, 0, 0, 0, 0, 0]
sim_single_injury.modules['RTI'].parameters['number_of_injured_body_regions_distribution'] = \
    [[1, 2, 3, 4, 5, 6, 7, 8], number_inj_data]

# Run the simulation
sim_single_injury.simulate(end_date=end_date)
# Parse the logfile of this simulation
log_df_single = parse_log_file(sim_single_injury.log_filepath)
# Store the incidence of RTI per 100,000 person years in this sim
sing_inj_incidences_of_rti = np.mean(log_df_single['tlo.methods.rti']['summary_1m']
                                     ['incidence of rti per 100,000'].tolist())
# Store the incidence of death due to RTI per 100,000 person years and the sub categories in this sim
sing_inj_incidences_of_death = np.mean(log_df_single['tlo.methods.rti']['summary_1m']
                                       ['incidence of rti death per 100,000'].tolist())
# one injury per person implies above are equivalent
sing_inj_incidences_of_injuries = np.mean(log_df_single['tlo.methods.rti']['summary_1m']
                                          ['incidence of rti per 100,000'].tolist())
sing_DALYs = log_df_single['tlo.methods.healthburden']['dalys']['Transport Injuries'].sum()

# Create the simulation object
log_config_multiple = {
    "filename": "rti_multiple_injury",  # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.rti": logging.INFO,
        "tlo.methods.healthsystem": logging.DEBUG,
        "tlo.methods.healthburden": logging.INFO
    }
}
sim_multiple_injury = Simulation(start_date=start_date, seed=seed, log_config=log_config_multiple,
                                 resourcefilepath=resourcefilepath)
# Register the modules
sim_multiple_injury.register(
    demography.Demography(),
    enhanced_lifestyle.Lifestyle(),
    healthsystem.HealthSystem(service_availability=['*']),
    symptommanager.SymptomManager(),
    healthseekingbehaviour.HealthSeekingBehaviour(),
    healthburden.HealthBurden(),
    rti.RTI(),
    simplified_births.SimplifiedBirths(),
)

# create and run the simulation
sim_multiple_injury.make_initial_population(n=pop_size)
# Run the simulation
sim_multiple_injury.simulate(end_date=end_date)
# Parse the logfile of this simulation
log_df_multiple = parse_log_file(sim_multiple_injury.log_filepath)
# Store the incidence of RTI per 100,000 person years in this sim
mult_inj_incidences_of_rti = np.mean(log_df_multiple['tlo.methods.rti']['summary_1m']
                                     ['incidence of rti per 100,000'].tolist())
# Store the incidence of death due to RTI per 100,000 person years and the sub categories in this sim
mult_inj_incidences_of_death = np.mean(log_df_multiple['tlo.methods.rti']['summary_1m']
                                       ['incidence of rti death per 100,000'].tolist())
print(f"incidence of death for multiple injuries = {mult_inj_incidences_of_death}")
mult_inj_incidences_of_injuries = np.mean(log_df_multiple['tlo.methods.rti']['Inj_category_incidence']
                                          ['tot_inc_injuries'].tolist())

mult_DALYs = log_df_multiple['tlo.methods.healthburden']['dalys']['Transport Injuries'].sum()
# compare the outputs from the single injury model run to the multiple injury model run

# compare incidence of rti, incidence of death and incidence of injuries in the simulation
data_sing = [sing_inj_incidences_of_rti, sing_inj_incidences_of_death, sing_inj_incidences_of_injuries]
data_mult = [mult_inj_incidences_of_rti, mult_inj_incidences_of_death, mult_inj_incidences_of_injuries]
plt.bar(np.arange(len(data_sing)), data_sing, width=0.4, color='lightsteelblue', label='single injury')
plt.bar(np.arange(len(data_mult)) + 0.4, data_mult, width=0.4, color='lightsalmon', label='multiple injury')
plt.ylabel('Incidence per 100,000 person years')
plt.xticks(np.arange(len(data_sing)), ['Incidence\nof\nRTI', 'Incidence\nof\nDeath', 'Incidence\nof\nInjuries'])
plt.legend()
plt.title('Comparing incidence of RTI, RTI death and \nincidence of injuries for a single injury\nvs multiple injury '
          'model run')
plt.show()
