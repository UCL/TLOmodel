"""Show the induced survival distribution without HIV"""


import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    pregnancy_supervisor,
    symptommanager,
)

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
popsize = 10000

# Establish the simulation object
log_config = {
    'filename': 'Logfile',
    'directory': outputpath,
    'custom_levels': {
        '*': logging.WARNING,
        'tlo.methods.hiv': logging.INFO,
    }
}

# Register the appropriate modules
sim = Simulation(start_date=start_date, seed=0, log_config=log_config)
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             contraception.Contraception(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable_and_reject_all=True),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             labour.Labour(resourcefilepath=resourcefilepath),
             pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
             hiv.Hiv(resourcefilepath=resourcefilepath)
             )

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)


# Generate distributions of the induced survival time "manually"

def get_survival_time_in_days(age_years):
    # select an adult who is alive and not currently infected:
    df = sim.population.props
    person_id = df.loc[df.is_alive & ~df.hv_inf & df.age_years.between(15, 80)].index[0]
    # set their age:
    df.at[person_id, 'age_years'] = age_years
    # compute date of death
    date_of_death = \
        sim.date + \
        sim.modules['Hiv'].get_time_from_infection_to_aids(person_id) + \
        sim.modules['Hiv'].get_time_from_aids_to_death()
    # return net survival time in days
    return (date_of_death - sim.date).days


ages = [15, 20, 25, 30, 35, 40, 45, 50]
nreps = 1000

res = []

for age in ages:
    for i in range(nreps):
        res.append({
            'age': age,
            'rep': i,
            'surv': get_survival_time_in_days(age_years=age) / 365.0
        })

res = pd.DataFrame.from_dict(res)

# summarise average survival times (in years):
summary_median = res.groupby(by=['age'])['surv'].median()

# plot distribution:
res.loc[res.age == 15]['surv'].hist(cumulative=True)
plt.show()
