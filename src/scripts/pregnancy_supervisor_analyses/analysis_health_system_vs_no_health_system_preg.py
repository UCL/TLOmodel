import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    antenatal_care,
    contraception,
    demography,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthsystem,
    labour,
    newborn_outcomes,
    pregnancy_supervisor,
    symptommanager, male_circumcision, hiv, tb, postnatal_supervisor
)
seed = 567

log_config = {
    "filename": "postnatal_analysis",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.labour": logging.DEBUG,
        "tlo.methods.healthsystem": logging.FATAL,
        "tlo.methods.hiv": logging.FATAL,
        "tlo.methods.newborn_outcomes": logging.DEBUG,
        "tlo.methods.antenatal_care": logging.DEBUG,
        "tlo.methods.pregnancy_supervisor": logging.DEBUG,
        "tlo.methods.postnatal_supervisor": logging.DEBUG,
    }
}
resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# Scenarios Definitions:
# *1: No Treatment
# *2: Some Treatment

scenarios = dict()
scenarios['No_Health_System'] = []
scenarios['Health_System'] = ['*']

# Create dict to capture the outputs
output_files = dict()

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2013, 1, 2)
popsize = 100

for label, service_avail in scenarios.items():
    # add file handler for the purpose of logging
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 # healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_avail),
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

    logfile = log_config["filename"]
    #logfile = sim.configure_logging(filename="LogFile")

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Save the full set of results:
    output_files[label] = logfile


def get_incidence_rate_and_death_numbers_from_logfile(logfile):
    output = parse_log_file(logfile)

    # Calculate the "incidence rate" from the output counts of incidence
    maternal_counts = output['tlo.methods.pregnancy_supervisor']['ps_summary_statistics']
    maternal_counts['year'] = pd.to_datetime(maternal_counts['date']).dt.year
    maternal_counts['year'] = maternal_counts['year'] - 1
    maternal_counts.drop(columns='date', inplace=True)
    maternal_counts.set_index(
        'year',
        drop=True,
        inplace=True
    )

    maternal_counts_lab = output['tlo.methods.labour']['labour_summary_stats_incidence']
    maternal_counts_lab['year'] = pd.to_datetime(maternal_counts_lab['date']).dt.year
    maternal_counts_lab['year'] = maternal_counts_lab['year'] - 1
    maternal_counts_lab.drop(columns='date', inplace=True)
    maternal_counts_lab.set_index(
        'year',
        drop=True,
        inplace=True
    )

    maternal_counts['final_mmr'] = maternal_counts['antenatal_mmr'] + maternal_counts_lab['intrapartum_mmr']
    maternal_counts['final_sbr'] = maternal_counts['antenatal_sbr'] + maternal_counts_lab['sbr']

    mmr = maternal_counts['final_mmr']
    sbr = maternal_counts['final_sbr']
    ar = maternal_counts['anaemia_rate']
    pe = maternal_counts['crude_pe']
    gh = maternal_counts['crude_gest_htn']

    return mmr, sbr, ar, pe, gh


maternal_mortality_ratio = dict()
still_birth_ratio = dict()
anaemia_rate = dict()
crude_pre_eclampsia = dict()
crude_gest_htn = dict()

for label, file in output_files.items():
    maternal_mortality_ratio[label], still_birth_ratio[label], anaemia_rate[label], crude_pre_eclampsia[label], \
    crude_gest_htn[label] = \
        get_incidence_rate_and_death_numbers_from_logfile(file)
data = {}


def generate_graphs(dictionary, title, saved_title):
    for label in dictionary.keys():
        data.update({label: dictionary[label]})
    pd.concat(data, axis=1).plot.bar()
    plt.title(f'{title}')
    plt.savefig(outputpath / (f"{saved_title}" + datestamp + ".pdf"), format='pdf')
    plt.show()


generate_graphs(maternal_mortality_ratio,'Combined antenatal and intrapartum MMR by Year', "mmr_by_scenario")
generate_graphs(still_birth_ratio, 'Combined antenatal and intrapartum SBR by Year', "sbr_by_scenario")
#  generate_graphs(anaemia_rate, 'Anaemia Rate by Year', "ar_by_scenario")
# generate_graphs(crude_pre_eclampsia, 'Crude new onset pre-eclampsia cases', "cpe_by_scenario")
# generate_graphs(crude_gest_htn, 'Crude new onset gestational hypertension cases', "cpe_by_scenario")
