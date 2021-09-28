from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    dx_algorithm_adult,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    rti,
    symptommanager,
)

# =============================== Analysis description ========================================================
# What I am doing here is artificially reducing the proportion of pre-hospital mortality, increasing the number of
# people funneled into the injured sub-population, who will have to subsequently have to seek health care. At the moment
# I have only included a range of reduction of pre-hospital mortality, but when I get around to focusing on this, I will
# model a reasonable level of pre-hospital mortality reduction.
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
pop_size = 10000
nsim = 2
# Set service availability
service_availability = ["*"]
# create a list to store the incidence of RTIs
list_rti_inc_average = []
# Create a list to store the incidence of RTI death average
list_rti_inc_death_average = []
# create np array for the percentage reduction in prehospital mortality
prehosital_percentage = np.linspace(0, 0.1, 5)
for i in range(0, nsim):
    # create empty lists to store number of deaths and dalys in
    list_inc_deaths = []
    list_inc_rti = []
    for prehosital_percent in prehosital_percentage:
        sim = Simulation(start_date=start_date)
        # register modules
        sim.register(
            demography.Demography(resourcefilepath=resourcefilepath),
            enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
            healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_availability),
            symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
            dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
            dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
            healthburden.HealthBurden(resourcefilepath=resourcefilepath),
            rti.RTI(resourcefilepath=resourcefilepath)
        )
        # name the logfile
        logfile = sim.configure_logging(filename="LogFile")
        # create initial population
        sim.make_initial_population(n=pop_size)
        # reduce prehospital mortality
        params = sim.modules['RTI'].parameters
        params['imm_death_proportion_rti'] = prehosital_percent
        # Run the simulation
        sim.simulate(end_date=end_date)
        # parse the logfile
        log_df = parse_log_file(logfile)
        # get the incidence of road traffic injury's in the sim
        rti_summary = log_df['tlo.methods.rti']['summary_1m']
        rti_inc = rti_summary['incidence of rti per 100,000'].tolist()
        list_inc_rti.append(rti_inc)
        # get the incidence of road traffic injury death in the sim
        rti_inc_death = rti_summary['incidence of rti death per 100,000'].tolist()
        list_inc_deaths.append(rti_inc_death)
    # Store the incidence of deaths and the RTI incidence from the sim
    list_rti_inc_death_average.append(np.mean(rti_inc_death))
    list_rti_inc_average.append(np.mean(list_inc_rti))

# Create a plot of the imm death vs incidences
plt.bar(np.arange(len(prehosital_percentage)), list_rti_inc_death_average, width=0.4, color='lightsteelblue',
        label='Incidence\n of\nRTI death')
plt.bar(np.arange(len(prehosital_percentage)) + 0.4, list_rti_inc_average, width=0.4, color='lightsalmon',
        label='Incidence\n of\nRTI')
xtick_labels = [str(percentage) for percentage in prehosital_percentage]
plt.xticks(np.arange(len(prehosital_percentage)) + 0.2, xtick_labels)
plt.legend()
plt.title(f"The effect of reducing pre-hospital mortality on the incidence of RTI"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")

plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/IncidenceOfRTIDeath/prehospital_vs_inc',
            bbox_inches='tight')
