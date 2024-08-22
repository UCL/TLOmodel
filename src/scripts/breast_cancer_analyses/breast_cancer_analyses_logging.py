import datetime
from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    breast_cancer,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
    hiv,
    tb,
    epi
)

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = './resources'

start_date = Date(2010, 1, 1)
end_date = Date(2011,  1, 1)
popsize = 1450
# 1450000

# Establish the simulation object
log_config = {
    'filename': 'LogFile',
    'directory': outputpath,
    'custom_levels': {
        'tlo.methods.demography': logging.CRITICAL,
        'tlo.methods.contraception': logging.CRITICAL,
        'tlo.methods.healthsystem': logging.CRITICAL,
        'tlo.methods.labour': logging.CRITICAL,
        'tlo.methods.healthburden': logging.CRITICAL,
        'tlo.methods.symptommanager': logging.CRITICAL,
        'tlo.methods.healthseekingbehaviour': logging.CRITICAL,
        'tlo.methods.pregnancy_supervisor': logging.CRITICAL
    }
}
sim = Simulation(start_date=start_date, seed=1, log_config=log_config, resourcefilepath=resourcefilepath)

# make a dataframe that contains the switches for which interventions are allowed or not allowed
# during this run. NB. These must use the exact 'registered strings' that the disease modules allow

# Register the appropriate modules
sim.register(demography.Demography(),
             care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(),
             contraception.Contraception(),
             enhanced_lifestyle.Lifestyle(),
             healthsystem.HealthSystem(),
             symptommanager.SymptomManager(),
             healthseekingbehaviour.HealthSeekingBehaviour(),
             healthburden.HealthBurden(),
             labour.Labour(),
             newborn_outcomes.NewbornOutcomes(),
             postnatal_supervisor.PostnatalSupervisor(),
             pregnancy_supervisor.PregnancySupervisor(),
             breast_cancer.BreastCancer(),
             hiv.Hiv(),
             tb.Tb(),
             epi.Epi()
             )

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# %% read the results
output = parse_log_file(sim.log_filepath)
