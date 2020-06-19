import datetime
from pathlib import Path

import pandas as pd
from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    parse_log_file,
)
from tlo.methods import demography, contraception, labour, enhanced_lifestyle, newborn_outcomes, healthsystem, \
    pregnancy_supervisor, antenatal_care, \
    healthburden, symptommanager, healthseekingbehaviour

# %%
outputpath = Path("./outputs")
resourcefilepath = Path("./resources")

# Create name for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 10000

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date)

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             contraception.Contraception(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             labour.Labour(resourcefilepath=resourcefilepath),
             newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
             pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
             antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath))

logfile = sim.configure_logging(filename="LogFile")
sim.seed_rngs(1)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Get the output from the logfile
output = parse_log_file(logfile)

stats = output['tlo.methods.antenatal_care']['anc_summary_stats']
stats['date'] = pd.to_datetime(stats['date'])
stats['year'] = stats['date'].dt.year

# Total number of ANC visits for each womans pregnancy
full_anc_output = output['tlo.methods.antenatal_care']['total_anc_per_woman']
full_anc_output['date'] = pd.to_datetime(full_anc_output['date'])
full_anc_output['year'] = full_anc_output['date'].dt.year
total_births = full_anc_output.groupby(['year'])['person_id'].size()

# Subset of women with > 4 ANC visits
women_with_anc4_by_year = full_anc_output.loc[full_anc_output['total_anc'] >= 4]
women_with_anc4_by_year = women_with_anc4_by_year.groupby(['year']).size()

women_with_anc4_by_year = women_with_anc4_by_year.reset_index()
women_with_anc4_by_year.index = women_with_anc4_by_year['year']
women_with_anc4_by_year.drop(columns='year', inplace=True)

#
final_anc4_df = pd.concat([women_with_anc4_by_year, total_births], axis=1)
final_anc4_df.columns = ['women_with_anc4', 'total_births']
final_anc4_df['proportion_of_anc4'] = (final_anc4_df.women_with_anc4 / final_anc4_df.total_births) * 100

# TODO: average number of ANC appoitnments per year
