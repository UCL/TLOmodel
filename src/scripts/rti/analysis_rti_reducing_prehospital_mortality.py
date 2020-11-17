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
pop_size = 500
nsim = 5
output_for_different_incidence = dict()
service_availability = ["*"]
list_deaths_average = []
list_tot_dalys_average = []
prehosital_mortality_reduction = np.linspace(1, 0, 5)
params = pd.read_excel(Path(resourcefilepath) / 'ResourceFile_RTI.xlsx', sheet_name='parameter_values')
orig_prehospital_mortality = float(params.loc[params.parameter_name == 'imm_death_proportion_rti', 'value'].values)
for i in range(0, nsim):
    list_deaths = []
    list_tot_dalys = []
    for reduction in prehosital_mortality_reduction:
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
        # Set the parameters used for this sim
        params['allowed_interventions'] = []
        params['imm_death_proportion_rti'] = orig_prehospital_mortality * reduction
        # Run the simulation
        sim.simulate(end_date=end_date)
        log_df = parse_log_file(logfile)
        # Get the deaths and DALYs from the sim
        deaths_without_med = log_df['tlo.methods.demography']['death']
        tot_death_without_med = len(deaths_without_med.loc[(deaths_without_med['cause'] != 'Other')])
        list_deaths.append(tot_death_without_med)
        # Get the DALYs from the sim
        dalys_df = log_df['tlo.methods.healthburden']['DALYS']
        males_data = dalys_df.loc[dalys_df['sex'] == 'M']
        YLL_males_data = males_data.filter(like='YLL_RTI').columns
        males_dalys = males_data[YLL_males_data].sum(axis=1) + \
                      males_data['YLD_RTI_rt_disability']
        females_data = dalys_df.loc[dalys_df['sex'] == 'F']
        YLL_females_data = females_data.filter(like='YLL_RTI').columns
        females_dalys = females_data[YLL_females_data].sum(axis=1) + \
                        females_data['YLD_RTI_rt_disability']
        tot_dalys = males_dalys.tolist() + females_dalys.tolist()
        list_tot_dalys.append(sum(tot_dalys))
    # Store the deaths and DALYs from the sim
    list_deaths_average.append(list_deaths)
    list_tot_dalys_average.append(list_tot_dalys)

# Get the average deaths per reduction of pre-hospital mortality
average_deaths = [float(sum(col)) / len(col) for col in zip(*list_deaths_average)]
# Get the average DALYs per reduction of pre-hospital mortality
average_tot_dalys = [float(sum(col)) / len(col) for col in zip(*list_tot_dalys_average)]
# Create the xtick labels
xtick_labels = [f"{np.round(1 - reduction, 2)}%" for reduction in prehosital_mortality_reduction]
# Create a dataframe of the results
plt.bar(np.arange(len(average_deaths)), average_deaths, color='lightsteelblue', width=0.25, label='Deaths')
plt.bar(np.arange(len(average_deaths)) + 0.25, average_tot_dalys, color='lightsalmon', width=0.25, label='DALYs')
plt.ylabel('Deaths/DALYs')
plt.xticks(np.arange(len(average_deaths)), xtick_labels, rotation=45)
plt.title(f"The effect of reducing pre-hospital mortality on average Deaths/DALYS"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/PrehospitalMortality/PrehospitalMortality_vs_deaths_DALYS.png', bbox_inches='tight')

