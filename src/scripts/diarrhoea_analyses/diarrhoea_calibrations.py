"""
* Check key outputs for reporting in the calibration table of the write-up
* Produce representative plots for the default parameters
"""
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation
from tlo.analysis.utils import make_age_grp_types, create_age_range_lookup, parse_log_file
from tlo.methods import (
    diarrhoea,
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    pregnancy_supervisor,
    symptommanager, simplified_births, dx_algorithm_child
)

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs
# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")
# The resource files
resourcefilepath = Path("./resources")
# Set parameters for the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 2)
popsize = 100000


def run_sim(service_availability):
    # Establish the simulation object and set the seed
    # seed is not set - each simulation run gets a random seed
    sim = Simulation(start_date=start_date, log_config={"filename": "LogFile"})
    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 # contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 # labour.Labour(resourcefilepath=resourcefilepath),
                 # pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
                 )
    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    return sim.log_filepath


def get_summary_stats(logfile):
    output = parse_log_file(logfile)
    # 3) DALYS wrt age (total over whole simulation)
    dalys = output['tlo.methods.healthburden']['dalys']
    dalys = dalys.groupby(by=['age_range']).sum()
    dalys.index = dalys.index.astype(make_age_grp_types())
    dalys = dalys.sort_index()

    # 4) DEATHS wrt age (total over whole simulation)
    deaths = output['tlo.methods.demography']['death']
    deaths['age_group'] = deaths['age'].map(demography.Demography(resourcefilepath=resourcefilepath).AGE_RANGE_LOOKUP)
    x = deaths.loc[deaths.cause.str.startswith('Diarrhoea')].copy()
    x['age_group'] = x['age_group'].astype(make_age_grp_types())
    diarrhoea_deaths = x.groupby(by=['age_group']).size()

    return {
        'dalys': dalys,
        'deaths': deaths,
        'diarrhoea_deaths': diarrhoea_deaths
    }


# %% Run the simulation with and without interventions being allowed
# With interventions:
logfile_with_healthsystem = run_sim(service_availability=['*'])
results_with_healthsystem = get_summary_stats(logfile_with_healthsystem)
# Without interventions:
logfile_no_healthsystem = run_sim(service_availability=[])
results_no_healthsystem = get_summary_stats(logfile_no_healthsystem)
# %% Produce Summary Graphs:
# Examine DALYS (summed over whole simulation)
results_no_healthsystem['dalys'].plot.bar(
    y=['YLD_Diarrhoea_rotavirus', 'YLD_Diarrhoea_shigella', 'YLD_Diarrhoea_adenovirus',
       'YLD_Diarrhoea_cryptosporidium', 'YLD_Diarrhoea_campylobacter', 'YLD_Diarrhoea_ETEC', 'YLD_Diarrhoea_sapovirus',
       'YLD_Diarrhoea_norovirus', 'YLD_Diarrhoea_astrovirus', 'YLD_Diarrhoea_tEPEC'],
    stacked=True)
plt.xlabel('Age-group')
plt.ylabel('DALYS')
plt.legend()
plt.title("With No Health System")
plt.show()
# Examine Deaths (summed over whole simulation)
deaths = results_no_healthsystem['diarrhoea_deaths']
deaths.index = deaths.index.astype(make_age_grp_types())
# # make a series with the right categories and zero so formats nicely in the grapsh:
agegrps = demography.Demography(resourcefilepath=resourcefilepath).AGE_RANGE_CATEGORIES
totdeaths = pd.Series(index=agegrps, data=np.nan)
totdeaths.index = totdeaths.index.astype(make_age_grp_types())
totdeaths = totdeaths.combine_first(deaths).fillna(0.0)
totdeaths.plot.bar()
plt.title('Deaths due to Diarrhoea')
plt.xlabel('Age-group')
plt.ylabel('Total Deaths During Simulation')
# plt.gca().get_legend().remove()
plt.show()
# Compare Deaths - with and without the healthsystem functioning - sum over age and time
deaths = {
    'No_HealthSystem': sum(results_no_healthsystem['diarrhoea_deaths']),
    'With_HealthSystem': sum(results_with_healthsystem['diarrhoea_deaths'])
}
plt.bar(range(len(deaths)), list(deaths.values()), align='center')
plt.xticks(range(len(deaths)), list(deaths.keys()))
plt.title('Deaths due to Diarrhoea')
plt.xlabel('Scenario')
plt.ylabel('Total Deaths During Simulation')
plt.show()
