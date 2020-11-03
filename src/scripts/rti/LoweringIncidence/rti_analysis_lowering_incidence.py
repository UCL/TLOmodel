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
pop_size = 50000
nsim = 5
output_for_different_incidence = dict()
service_availability = ["*"]
incidence_average = []
list_deaths_average = []
list_tot_dalys_average = []
incidence_reduction = np.linspace(1, 0, 10)
scenarios = {'None': 1,
             'Speed limit enforcement': (1 - 0.06),  # https://www.bmj.com/content/bmj/344/bmj.e612.full.pdf
             'Drink driving legistlature enforcement': (1 - 0.15),
             }
for i in range(0, nsim):
    incidence = []
    list_deaths = []
    list_tot_dalys = []
    incidences = []
    for scenario_reduction in scenarios.values():
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
        orig_inc = params['base_rate_injrti']
        params['base_rate_injrti'] = orig_inc * scenario_reduction
        incidences.append(params['base_rate_injrti'])
        sim.simulate(end_date=end_date)
        log_df = parse_log_file(logfile)
        summary_1m = log_df['tlo.methods.rti']['summary_1m']
        incidence_in_sim = summary_1m['incidence of rti per 100,000']
        mean_incidence = np.mean(summary_1m['incidence of rti per 100,000'])
        incidence.append(mean_incidence)

        deaths_without_med = log_df['tlo.methods.demography']['death']
        tot_death_without_med = len(deaths_without_med.loc[(deaths_without_med['cause'] != 'Other')])
        list_deaths.append(tot_death_without_med)
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

    incidence_average.append(incidence)
    list_deaths_average.append(list_deaths)
    list_tot_dalys_average.append(list_tot_dalys)

average_incidence = [float(sum(col)) / len(col) for col in zip(*incidence_average)]
average_deaths = [float(sum(col)) / len(col) for col in zip(*list_deaths_average)]
average_tot_dalys = [float(sum(col)) / len(col) for col in zip(*list_tot_dalys_average)]
results_df = pd.DataFrame({
    'incidence per 100,000': average_incidence,
    'average deaths': average_deaths,
    'average total DALYs': average_tot_dalys,
})
ax = results_df[['average deaths', 'average total DALYs']].plot(kind='bar', width=.35)
ax2 = results_df['incidence per 100,000'].plot(secondary_y=True, ax=ax)
ax.set_ylabel('Deaths/DALYs')
ax2.set_ylabel('Incidence per 100,000')
ax.set_xticklabels(labels=scenarios.keys(), rotation=45)
plt.title(f"The effect of reducing incidence on Average Deaths/DALYS"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/ReducingIncidence/incidence_vs_deaths_DALYS.png', bbox_inches='tight')
print(incidences)
