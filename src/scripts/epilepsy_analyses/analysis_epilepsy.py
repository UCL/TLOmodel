import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import compare_number_of_deaths, parse_log_file
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epilepsy,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
)

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = Date(2020,  1, 1)
popsize = 200000

# Establish the simulation object
log_config = {
    'filename': 'LogFile',
    'directory': outputpath,
    'custom_levels': {
        '*': logging.CRITICAL,
        'tlo.methods.epilepsy': logging.INFO,
        'tlo.methods.demography': logging.INFO,
        'tlo.methods.healthsystem': logging.WARNING,
        'tlo.methods.healthburden': logging.WARNING,
    }
}

sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

# make a dataframe that contains the switches for which interventions are allowed or not allowed
# during this run. NB. These must use the exact 'registered strings' that the disease modules allow


# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             epilepsy.Epilepsy(resourcefilepath=resourcefilepath),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath)
             )

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)

sim.simulate(end_date=end_date)


# %% read the results
output = parse_log_file(sim.log_filepath)

prop_seiz_stat_1 = pd.Series(
    output['tlo.methods.epilepsy']['epilepsy_logging']['prop_seiz_stat_1'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date']
)
plt.plot(prop_seiz_stat_1, color='lightsteelblue', label='model')
plt.axhline(0.013, color='lightsalmon', label='Ba Diop et al. 2014')
plt.title('Proportion of people with epilepsy but no current seizures')
plt.ylim(0, 0.05)
plt.legend()
plt.tight_layout()
plt.show()
plt.clf()

prop_seiz_stat_2 = pd.Series(
    output['tlo.methods.epilepsy']['epilepsy_logging']['prop_seiz_stat_2'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date']
)
plt.plot(prop_seiz_stat_2, color='lightsteelblue', label='model')
plt.axhline(0.013, color='lightsalmon', label='Ba Diop et al. 2014')
plt.title('Proportion of people with infrequent epilepsy seizures')
plt.ylim(0, 0.02)
plt.legend()
plt.tight_layout()
plt.show()
plt.clf()


prop_seiz_stat_3 = pd.Series(
    output['tlo.methods.epilepsy']['epilepsy_logging']['prop_seiz_stat_3'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date']
)
plt.plot(prop_seiz_stat_3, color='lightsteelblue', label='model')
plt.axhline(0.013, color='lightsalmon', label='Ba Diop et al. 2014')
plt.title('Proportion of people with frequent epilepsy seizures')
plt.ylim(0, 0.015)
plt.legend()
plt.tight_layout()
plt.show()
plt.clf()

mean_proportion_in_sim = [np.mean(prop_seiz_stat_1), np.mean(prop_seiz_stat_2), np.mean(prop_seiz_stat_3)]
plt.bar(np.arange(len(mean_proportion_in_sim)), mean_proportion_in_sim,
        color=['lightsalmon', 'lightsteelblue', 'lemonchiffon'])
plt.axhline(0.013, color='black', linestyle=':', label='Ba Diop et al. 2014')
plt.legend()
plt.title('Average proportion of each seizure status')
plt.xticks(np.arange(len(mean_proportion_in_sim)), ['seizure\nstatus 1', 'seizure\nstatus 2', 'seizure\nstatus 3'])
plt.tight_layout()
plt.show()
plt.clf()

n_seiz_stat_1_3 = pd.Series(
    output['tlo.methods.epilepsy']['epilepsy_logging']['n_seiz_stat_1_3'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date']
)
n_seiz_stat_1_3.plot()
plt.title('Number with epilepsy (past or current)')
plt.ylim(0, 800000)
plt.tight_layout()
plt.show()

n_seiz_stat_2_3 = pd.Series(
    output['tlo.methods.epilepsy']['epilepsy_logging']['n_seiz_stat_2_3'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date']
)
n_seiz_stat_2_3.plot()
plt.title('Number with epilepsy (infrequent or frequent seizures)')
plt.ylim(0, 300000)
plt.tight_layout()
plt.show()
plt.clf()

prop_antiepilep_seiz_stat_1 = pd.Series(
    output['tlo.methods.epilepsy']['epilepsy_logging']['prop_antiepilep_seiz_stat_1'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date']
)
prop_antiepilep_seiz_stat_1.plot()
plt.title('Proportion on antiepileptics\namongst people with epilepsy but no current seizures')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
plt.clf()

prop_antiepilep_seiz_stat_2 = pd.Series(
    output['tlo.methods.epilepsy']['epilepsy_logging']['prop_antiepilep_seiz_stat_2'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date']
)
prop_antiepilep_seiz_stat_2.plot()
plt.title('Proportion on antiepileptics\namongst people with infrequent epilepsy seizures')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
plt.clf()

prop_antiepilep_seiz_stat_3 = pd.Series(
    output['tlo.methods.epilepsy']['epilepsy_logging']['prop_antiepilep_seiz_stat_3'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date']
)
prop_antiepilep_seiz_stat_3.plot()
plt.title('Proportion on antiepileptics\namongst people with frequent epilepsy seizures')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
plt.clf()

n_epi_death = pd.Series(
    output['tlo.methods.epilepsy']['epilepsy_logging']['n_epi_death'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date']
)
n_epi_death.plot()
plt.title('Number of deaths from epilepsy')
plt.ylim(0, 50)
plt.tight_layout()
plt.show()
plt.clf()

n_antiep = pd.Series(
    output['tlo.methods.epilepsy']['epilepsy_logging']['n_antiep'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date']
)
n_antiep.plot()
plt.title('Number of people on antiepileptics')
plt.ylim(0, 50000)
plt.tight_layout()
plt.show()
plt.clf()

epi_death_rate = pd.Series(
    output['tlo.methods.epilepsy']['epilepsy_logging']['epi_death_rate'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date']
)
plt.plot(epi_death_rate, color='lightsteelblue', label='Incidence of\ndeath')
plt.axhline(np.mean(epi_death_rate), label=f"Mean incidence of\ndeath = {np.round(np.mean(epi_death_rate), 3)}")
plt.title('Rate of epilepsy death in people with seizures')
plt.legend()
plt.ylim(0, 20)
plt.tight_layout()
plt.show()
plt.clf()

incidence_epilepsy = pd.Series(
    output['tlo.methods.epilepsy']['inc_epilepsy']['incidence_epilepsy'].values,
    index=output['tlo.methods.epilepsy']['inc_epilepsy']['date']
)
plt.plot(incidence_epilepsy, color='lightsteelblue', label='Incidence of\nepilepsy')
plt.axhline(np.mean(incidence_epilepsy),
            color='steelblue',
            label=f"Mean incidence of\nepilepsy = {np.round(np.mean(incidence_epilepsy), 2)}")
plt.title('Incidence of epilepsy')
plt.legend()
plt.ylim(0, 100)
plt.tight_layout()
plt.show()
plt.clf()


# Compare Deaths due to Epilepsy with GBD data
comparison = compare_number_of_deaths(logfile=sim.log_filepath, resourcefilepath=resourcefilepath)

CAUSE_NAME = 'Epilepsy'

fig, axs = plt.subplots(nrows=2, ncols=1, sharey=True, sharex=True)
for _row, period in enumerate(('2010-2014', '2015-2019')):
    ax = axs[_row]
    comparison.loc[(period, slice(None), slice(None), CAUSE_NAME)]\
              .droplevel([0, 1, 3])\
              .groupby(axis=0, level=0)\
              .sum()\
              .plot(use_index=True, ax=ax)
    ax.set_ylabel('Deaths per year')
    ax.set_title(f"{period}")
    xticks = comparison.index.levels[2]
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks, rotation=90)
fig.tight_layout()
fig.show()


# Compare model outputs to GBD study
gbd_data = pd.read_csv(resourcefilepath / "epilepsy" / "IHME-GBD_2019_DATA-4a82f00e-1.csv")
# get incidence estimates
gbd_inc_data = gbd_data.loc[gbd_data['measure'] == 'Incidence']
# mean incidence of epilepsy
mean_inc_gbd = gbd_inc_data.val.mean()
# get death estimate
gbd_inc_death_data = gbd_data.loc[gbd_data['measure'] == 'Deaths']
# mean incidence of epilepsy
mean_inc_death_gbd = gbd_inc_death_data.val.mean()

plt.bar(np.arange(2), [mean_inc_gbd, mean_inc_death_gbd], width=0.4, color='lightsteelblue', label='GBD')
plt.bar(np.arange(2) + 0.4, [np.mean(incidence_epilepsy), np.mean(epi_death_rate)], width=0.4, color='lightsalmon',
        label='Model')
plt.legend()
plt.xticks(np.arange(2) + 0.2, ['Incidence of epilepsy', 'Incidence of death'])
plt.ylabel('Incidence per 100,000')
plt.title("The epilepsy model's estimated incidence of epilepsy\nand epilepsy death compared to the GBD study")
plt.show()
plt.clf()

plt.bar([1, 2], [mean_inc_death_gbd / mean_inc_gbd, np.mean(epi_death_rate) / np.mean(incidence_epilepsy)],
        color=['lightsteelblue', 'lightsalmon'])
plt.xticks([1, 2], ['GBD', 'Model'])
plt.ylabel('CFR')
plt.title("The epilepsy model's case fatality ratio compared to the\n GBD's estimate")
plt.tight_layout()
plt.show()
plt.clf()

incidence_epilepsy_df = pd.DataFrame(incidence_epilepsy)
incidence_epilepsy_df = incidence_epilepsy_df.rename(columns={0: 'inc'})
incidence_epilepsy_df['year'] = incidence_epilepsy_df.index.year
incidence_epilepsy_df = incidence_epilepsy_df.groupby('year').mean()
gbd_est_inc = gbd_inc_data.loc[gbd_inc_data['year'] <= incidence_epilepsy_df.index.max(), 'val']
plt.plot(np.arange(len(incidence_epilepsy_df.index)), incidence_epilepsy_df['inc'], color='lightsalmon', label='model')
plt.plot(np.arange(len(incidence_epilepsy_df.index)), gbd_est_inc, color='lightsteelblue', label='GBD')
plt.legend()
plt.xticks(np.arange(len(incidence_epilepsy_df.index)), incidence_epilepsy_df.index)
plt.ylabel('Incidence of epilepsy')
plt.title("Comparing the model's estimated incidence\nof epilepsy over time to the GBD estimates")
plt.tight_layout()
plt.show()
plt.clf()

incidence_epilepsy_death_df = pd.DataFrame(epi_death_rate)
incidence_epilepsy_death_df = incidence_epilepsy_death_df.rename(columns={0: 'inc_death'})
incidence_epilepsy_death_df['year'] = incidence_epilepsy_death_df.index.year
incidence_epilepsy_death_df = incidence_epilepsy_death_df.groupby('year').mean()
gbd_est_inc_death = gbd_inc_death_data.loc[gbd_inc_death_data['year'] <= incidence_epilepsy_death_df.index.max(), 'val']
plt.plot(np.arange(len(incidence_epilepsy_death_df.index)), incidence_epilepsy_death_df['inc_death'],
         color='lightsalmon', label='model')
plt.plot(np.arange(len(incidence_epilepsy_death_df.index)), gbd_est_inc_death, color='lightsteelblue', label='GBD')
plt.legend()
plt.xticks(np.arange(len(incidence_epilepsy_death_df.index)), incidence_epilepsy_df.index)
plt.ylabel('Incidence of epilepsy death')
plt.title("Comparing the model's estimated incidence\nof epilepsy death over time to the GBD estimates")
plt.tight_layout()
plt.show()
plt.clf()
