import datetime
from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    hiv,
    male_circumcision,
    tb,
)

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 5000

# Establish the simulation object
log_config = {
    'filename': 'Logfile',
    'directory': outputpath,
    'custom_levels': {
        '*': logging.WARNING,
        'tlo.methods.hiv': logging.INFO,
        'tlo.methods.tb': logging.INFO,
        'tlo.methods.male_circumcision': logging.INFO
    }
}
sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

# ----- Control over the types of intervention that can occur -----
# Make a list that contains the treatment_id that will be allowed. Empty list means nothing allowed.
# '*' means everything. It will allow any treatment_id that begins with a stub (e.g. Mockitis*)
service_availability = ["*"]

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             hiv.hiv(resourcefilepath=resourcefilepath),
             tb.tb(resourcefilepath=resourcefilepath),
             male_circumcision.male_circumcision(resourcefilepath=resourcefilepath))

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# %% read the results

output = parse_log_file(sim.log_filepath)


# outputpath = './src/scripts/outputLogs/'
# TODO: I am removing the redef of outputpath (see above)

# deaths_df = outputs['tlo.methods.demography']['death']
# deaths_df['date'] = pd.to_datetime(deaths_df['date'])
# deaths_df['year'] = deaths_df['date'].dt.year
# death_by_cause = deaths_df.groupby(['year','cause'])['person_id'].size()
# #

# TODO: Maybe add some graphs here to demonstrate the results? For example....
# %% Demonstrate the HIV epidemic and it's impact on the population
