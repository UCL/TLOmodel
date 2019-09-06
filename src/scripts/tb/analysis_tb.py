import datetime
import logging
import os
from pathlib import Path
import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    healthburden,
    healthsystem,
    hiv,
    lifestyle,
    malecircumcision,
    tb
)

# Where will output go
outputpath = './src/scripts/tb/'

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = Date(2025, 12, 31)
popsize = 1000

# Establish the simulation object
sim = Simulation(start_date=start_date)

# Establish the logger
logfile = outputpath + "LogFile" + datestamp + ".log"

if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)

# ----- Control over the types of intervention that can occur -----
# Make a list that contains the treatment_id that will be allowed. Empty list means nothing allowed.
# '*' means everything. It will allow any treatment_id that begins with a stub (e.g. Mockitis*)
service_availability = ["*"]

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(lifestyle.Lifestyle())
sim.register(hiv.Hiv(resourcefilepath=resourcefilepath))
sim.register(tb.Tb(resourcefilepath=resourcefilepath))
sim.register(malecircumcision.MaleCircumcision(resourcefilepath=resourcefilepath))

for name in logging.root.manager.loggerDict:
    if name.startswith("tlo"):
        logging.getLogger(name).setLevel(logging.WARNING)

logging.getLogger('tlo.methods.hiv').setLevel(logging.INFO)
logging.getLogger('tlo.methods.malecircumcision').setLevel(logging.INFO)
logging.getLogger("tlo.methods.tb").setLevel(logging.INFO)
logging.getLogger("tlo.methods.demography").setLevel(logging.INFO)  # to get deaths

# Run the simulation and flush the logger
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
fh.flush()
# fh.close()

# %% read the results
import pandas as pd

outputpath = './src/scripts/tb/'
datestamp = datetime.date.today().strftime("__%Y_%m_%d")
logfile = outputpath + "LogFile" + datestamp + ".log"
output = parse_log_file(logfile)

# output = parse_log_file('./src/scripts/tb/LogFile__2019_09_05FULL_RUN.log')

#
# ## HIV
# inc = output['tlo.methods.hiv']['hiv_infected']
# prev_m = output['tlo.methods.hiv']['hiv_adult_prev_m']
# prev_f = output['tlo.methods.hiv']['hiv_adult_prev_f']
# prev_child = output['tlo.methods.hiv']['hiv_child_prev_m']
# tx = output['tlo.methods.hiv']['hiv_treatment']
# fsw = output['tlo.methods.hiv']['hiv_fsw']
# mort = output['tlo.methods.hiv']['hiv_mortality']
#
#
# inc.to_csv(r'Z:Thanzi la Onse\HIV\Model_original\inc2.csv', header=True)
# prev_m.to_csv(r'Z:Thanzi la Onse\HIV\Model_original\prev_m2.csv', header=True)
# prev_f.to_csv(r'Z:Thanzi la Onse\HIV\Model_original\prev_f2.csv', header=True)
# prev_child.to_csv(r'Z:Thanzi la Onse\HIV\Model_original\prev_child2.csv', header=True)
# tx.to_csv(r'Z:Thanzi la Onse\HIV\Model_original\tx2.csv', header=True)
# fsw.to_csv(r'Z:Thanzi la Onse\HIV\Model_original\fsw2csv', header=True)
# mort.to_csv(r'Z:Thanzi la Onse\HIV\Model_original\mort2.csv', header=True)
#
# #
## TB
# tb_inc = output['tlo.methods.tb']['tb_incidence']
# tb_prev_m = output['tlo.methods.tb']['tb_propActiveTbMale']
# tb_prev_f = output['tlo.methods.tb']['tb_propActiveTbFemale']
# tb_prev = output['tlo.methods.tb']['tb_prevalence']
# tb_mort = output['tlo.methods.tb']['tb_mortality']
#
#
# tb_inc.to_csv(r'Z:Thanzi la Onse\HIV\Model_original\tb_inc2.csv', header=True)
# tb_prev_m.to_csv(r'Z:Thanzi la Onse\HIV\Model_original\tb_prev_m2.csv', header=True)
# tb_prev_f.to_csv(r'Z:Thanzi la Onse\HIV\Model_original\tb_prev_f2.csv', header=True)
# tb_prev.to_csv(r'Z:Thanzi la Onse\HIV\Model_original\tb_prev2.csv', header=True)
# tb_mort.to_csv(r'Z:Thanzi la Onse\HIV\Model_original\tb_mort2.csv', header=True)

#
# deaths_df = output['tlo.methods.demography']['death']
# deaths_df['date'] = pd.to_datetime(deaths_df['date'])
# deaths_df['year'] = deaths_df['date'].dt.year
# d_gp=deaths_df.groupby(['year', 'cause']).size().unstack().fillna(0)
# d_gp.to_csv(r'Z:Thanzi la Onse\HIV\Model_original\deaths2.csv', header=True)
