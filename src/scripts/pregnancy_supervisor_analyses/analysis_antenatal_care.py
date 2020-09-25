import datetime
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
from tlo.lm import LinearModel, LinearModelType
from tlo import Date, Simulation
from tlo.analysis.utils import (
    parse_log_file,
)
from tlo.methods import demography, contraception, labour, enhanced_lifestyle, newborn_outcomes, healthsystem, \
    pregnancy_supervisor, antenatal_care, symptommanager, healthseekingbehaviour, tb, hiv, male_circumcision, \
    postnatal_supervisor

# %%
outputpath = Path("./outputs")
resourcefilepath = Path("./resources")

# Create name for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2016, 1, 1)
popsize = 1000

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date)

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 # healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*']),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 male_circumcision.male_circumcision(resourcefilepath=resourcefilepath),
                 hiv.hiv(resourcefilepath=resourcefilepath),
                 tb.tb(resourcefilepath=resourcefilepath),
                 antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))


logfile = sim.configure_logging(filename="LogFile")
sim.seed_rngs(1)

sim.make_initial_population(n=popsize)

params_preg_sup = sim.modules['PregnancySupervisor'].parameters

#params_preg_sup['ps_linear_equations']['eight_or_more_anc_visits'] = LinearModel(
 #                   LinearModelType.MULTIPLICATIVE, 1)

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

# Subset of women with => 8 ANC visits
women_with_anc8_by_year = full_anc_output.loc[full_anc_output['total_anc'] >= 8]
women_with_anc8_by_year = women_with_anc8_by_year.groupby(['year']).size()

women_with_anc8_by_year = women_with_anc8_by_year.reset_index()
women_with_anc8_by_year.index = women_with_anc8_by_year['year']
women_with_anc8_by_year.drop(columns='year', inplace=True)

final_anc4_df = pd.concat([women_with_anc4_by_year, women_with_anc8_by_year, total_births], axis=1)
final_anc4_df.columns = ['women_with_anc4', 'women_with_anc8', 'total_births']
final_anc4_df['proportion_of_anc4'] = (final_anc4_df.women_with_anc4 / final_anc4_df.total_births) * 100
final_anc4_df['proportion_of_anc8'] = (final_anc4_df.women_with_anc8 / final_anc4_df.total_births) * 100


print(full_anc_output)
print(women_with_anc4_by_year)
print(final_anc4_df)


final_anc4_df.plot.bar(y='proportion_of_anc4', stacked=True)
plt.title("Proportion of women achieving ANC4+ by year")
plt.show()

final_anc4_df.plot.bar(y='proportion_of_anc8', stacked=True)
plt.title("Proportion of women achieving ANC8+ by year")
plt.show()

stats.plot.bar(y='mean_ga_first_anc', stacked=True)
plt.title("Average gestational age of women at ANC1 by year")
plt.show()

stats.plot.bar(y='proportion_anc1_first_trimester', stacked=True)
plt.title("Proportion of ANC1 visits occuring in the first trimester by year")
plt.show()

# TODO: average number of ANC appoitnments per year
