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
list_deaths_average = []
list_tot_dalys_average = []
prehosital_mortality_reduction = np.linspace(1, 0, 5)
for i in range(0, nsim):
    list_deaths = []
    list_tot_dalys = []
    for inc in prehosital_mortality_reduction:
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
        orig_prehospital_mortality = params['imm_death_proportion_rti']
        params['imm_death_proportion_rti'] = orig_prehospital_mortality * inc
        sim.simulate(end_date=end_date)
        log_df = parse_log_file(logfile)
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

    list_deaths_average.append(list_deaths)
    list_tot_dalys_average.append(list_tot_dalys)

average_deaths = [float(sum(col)) / len(col) for col in zip(*list_deaths_average)]
average_tot_dalys = [float(sum(col)) / len(col) for col in zip(*list_tot_dalys_average)]
results_df = pd.DataFrame({
    'average deaths': average_deaths,
    'average total DALYs': average_tot_dalys,
})
ax = results_df[['average deaths', 'average total DALYs']].plot(kind='bar', width=.35)
ax.set_ylabel('Deaths/DALYs')
ax.set_xticklabels(labels=['0% reduction', '25% reduction', '50% reduction', '75% reduction',
                           '100% reduction'], rotation=45)
plt.title(f"The effect of reducing pre-hospital mortality on average Deaths/DALYS"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/PrehospitalMortality/PrehospitalMortality_vs_deaths_DALYS.png', bbox_inches='tight')
