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
import pandas as pd

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
yearsrun = 10
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
pop_size = 5000
nsim = 5
output_for_different_incidence = dict()
service_availability = ["*"]
sim_age_range = []
females = 0
males = 0

for i in range(0, nsim):
    age_range = []
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
    log_df = parse_log_file(logfile)
    demog = log_df['tlo.methods.rti']['rti_demography']
    males += sum(demog['males_in_rti'])
    females += sum(demog['females_in_rti'])
    this_sim_ages = demog['age'].tolist()
    for elem in this_sim_ages:
        for item in elem:
            sim_age_range.append(item)

zero_to_five = len([i for i in sim_age_range if i < 6])
six_to_ten = len([i for i in sim_age_range if 6 <= i < 11])
eleven_to_fifteen = len([i for i in sim_age_range if 11 <= i < 16])
sixteen_to_twenty = len([i for i in sim_age_range if 16 <= i < 21])
twenty1_to_twenty5 = len([i for i in sim_age_range if 21 <= i < 26])
twenty6_to_thirty = len([i for i in sim_age_range if 26 <= i < 31])
thirty1_to_thirty5 = len([i for i in sim_age_range if 31 <= i < 36])
thirty6_to_forty = len([i for i in sim_age_range if 36 <= i < 41])
forty1_to_forty5 = len([i for i in sim_age_range if 41 <= i < 46])
forty6_to_fifty = len([i for i in sim_age_range if 46 <= i < 51])
fifty1_to_fifty5 = len([i for i in sim_age_range if 51 <= i < 56])
fifty6_to_sixty = len([i for i in sim_age_range if 56 <= i < 61])
sixty1_to_sixty5 = len([i for i in sim_age_range if 61 <= i < 66])
sixty6_to_seventy = len([i for i in sim_age_range if 66 <= i < 71])
seventy1_to_seventy5 = len([i for i in sim_age_range if 71 <= i < 76])
seventy6_to_eighty = len([i for i in sim_age_range if 76 <= i < 81])
eighty1_to_eighty5 = len([i for i in sim_age_range if 81 <= i < 86])
eighty6_to_ninety = len([i for i in sim_age_range if 86 <= i < 91])
ninety_plus = len([i for i in sim_age_range if 90 < i])
height_for_bar_plot = [zero_to_five, six_to_ten, eleven_to_fifteen, sixteen_to_twenty, twenty1_to_twenty5,
                       twenty6_to_thirty, thirty1_to_thirty5, thirty6_to_forty, forty1_to_forty5, forty6_to_fifty,
                       fifty1_to_fifty5, fifty6_to_sixty, sixty1_to_sixty5, sixty6_to_seventy, seventy1_to_seventy5,
                       seventy6_to_eighty, eighty1_to_eighty5, eighty6_to_ninety, ninety_plus]
height_for_bar_plot = np.divide(height_for_bar_plot, sum(height_for_bar_plot))
labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40',
          '41-45', '46-50', '51-55', '56-60', '61-65', '66-70', '71-75', '76-80',
          '81-85', '86-90', '90+']
plt.bar(np.arange(len(height_for_bar_plot)), height_for_bar_plot)
plt.xticks(np.arange(len(height_for_bar_plot)), labels, rotation=45)
plt.ylabel('Percentage')
plt.xlabel('Age')
plt.title(f"Age demographics of those with RTIs"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/Demographics_of_RTI/Age_demographics.png', bbox_inches='tight')
plt.clf()
plt.close()

total_injuries = males + females
male_perc = males / total_injuries
femal_perc = females / total_injuries

plt.bar(np.arange(2), [male_perc, femal_perc])
plt.xticks(np.arange(2), ['Males', 'Females'])
plt.ylabel('Percentage')
plt.xlabel('Gender')
plt.title(f"Gender demographics of those with RTIs"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/Demographics_of_RTI/Gender_demographics.png', bbox_inches='tight')
