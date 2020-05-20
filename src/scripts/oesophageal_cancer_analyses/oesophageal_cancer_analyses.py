import datetime
from pathlib import Path

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    oesophagealcancer,
    pregnancy_supervisor, labour, healthseekingbehaviour, symptommanager)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

# Set parameters for the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 10000

# Establish the simulation object and set the seed
sim = Simulation(start_date=start_date)
sim.seed_rngs(0)

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             contraception.Contraception(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             labour.Labour(resourcefilepath=resourcefilepath),
             pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
             oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath)
             )

# Manipulate parameters in order that there is a high burden of oes_cancer in order to do the checking:
# Initial prevalence of cancer:
sim.modules['OesophagealCancer'].parameters['init_prop_oes_cancer_stage'] = [0.2, 0.1, 0.1, 0.05, 0.05, 0.025]

# Rate of cancer onset per 3 months:
sim.modules['OesophagealCancer'].parameters['r_low_grade_dysplasia_none'] = 0.05

# Rates of cancer progression per 3 months:
sim.modules['OesophagealCancer'].parameters['r_high_grade_dysplasia_low_grade_dysp'] *= 5
sim.modules['OesophagealCancer'].parameters['r_stage1_high_grade_dysp'] *= 5
sim.modules['OesophagealCancer'].parameters['r_stage2_stage1'] *= 5
sim.modules['OesophagealCancer'].parameters['r_stage3_stage2'] *= 5
sim.modules['OesophagealCancer'].parameters['r_stage4_stage3'] *= 5

# Establish the logger
logfile = sim.configure_logging(filename="LogFile")

# Run the simulation
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)


# %% TODO: Demonstrate the burden and the interventions
output = parse_log_file(logfile)

# PREVALENCE wrt dx/ treated/ palliative
s = output['tlo.methods.oesophagealcancer']['summary_stats']
s['date'] = pd.to_datetime(s['date'])
s = s.set_index('date', drop=True)

# Total prevalence
s.plot(y=['total_low_grade_dysplasia', 'total_high_grade_dysplasia', 'total_stage1', 'total_stage2', 'total_stage3', 'total_stage4'])
plt.show()

# Numbers diagnosed, treated, palliated

s.plot()
plt.show()

# DALYS wrt age
h = output['tlo.methods.healthburden']['DALYS']
h['date'] = pd.to_datetime(h['date'])
h = h.set_index('date', drop=True)

h.groupby(by=['age_range']).sum().reset_index().plot.bar(x='age_range', y=['YLD_OesophagealCancer_0'], stacked=True)
plt.show()

# DEATHS wrt age and time


# %% TODO: Demonstrate the impact of the interventions

