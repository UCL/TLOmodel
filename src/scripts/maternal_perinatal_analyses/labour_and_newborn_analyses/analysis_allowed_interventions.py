import datetime
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

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
    symptommanager, male_circumcision, tb, hiv, postnatal_supervisor
)
seed = 567

log_config = {
    "filename": "postnatal_analysis",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.labour": logging.DEBUG,
        "tlo.methods.newborn_outcomes": logging.DEBUG,
        "tlo.methods.antenatal_care": logging.DEBUG,
        "tlo.methods.pregnancy_supervisor": logging.DEBUG,
        "tlo.methods.postnatal_supervisor": logging.DEBUG,
    }
}

resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# Scenarios Definitions:
# *1: All interventions called within the Labour HSI are able to run
# *2: Selected interventions called within the Labour HSI are able to run, as listed

scenarios = dict()
scenarios['all_interventions'] = ['prophylactic_labour_interventions',
                                  'assessment_and_treatment_of_severe_pre_eclampsia',
                                  'assessment_and_treatment_of_obstructed_labour',
                                  'assessment_and_treatment_of_maternal_sepsis',
                                  'assessment_and_treatment_of_hypertension',
                                  'assessment_and_treatment_of_eclampsia',
                                  'assessment_and_plan_for_referral_antepartum_haemorrhage',
                                  'assessment_and_plan_for_referral_uterine_rupture',
                                  'active_management_of_the_third_stage_of_labour',
                                  'assessment_and_treatment_of_pph_retained_placenta',
                                  'assessment_and_treatment_of_pph_uterine_atony']
#
scenarios['selected_interventions'] = ['prophylactic_labour_interventions',
                                       'assessment_and_treatment_of_severe_pre_eclampsia',
                                       'assessment_and_treatment_of_obstructed_labour',
                                       'assessment_and_treatment_of_maternal_sepsis',
                                       'assessment_and_treatment_of_hypertension',
                                       'assessment_and_treatment_of_eclampsia',
                                       'assessment_and_plan_for_referral_antepartum_haemorrhage',
                                       'active_management_of_the_third_stage_of_labour',
                                       'assessment_and_treatment_of_pph_retained_placenta',
                                       'assessment_and_treatment_of_pph_uterine_atony']

# Create dict to capture the outputs
output_files = dict()

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2013, 1, 2)
popsize = 10000

for label, allowed_interventions in scenarios.items():
    # add file handler for the purpose of logging
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 # healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*']),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 male_circumcision.male_circumcision(resourcefilepath=resourcefilepath),
                 hiv.hiv(resourcefilepath=resourcefilepath),
                 tb.tb(resourcefilepath=resourcefilepath),
                 antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))

    logfile = sim.configure_logging(filename="LogFile")

    sim.make_initial_population(n=popsize)

    params = sim.modules['Labour'].parameters
    params['allowed_interventions'] = allowed_interventions

    sim.simulate(end_date=end_date)

    # Save the full set of results:
    output_files[label] = logfile


def get_incidence_rate_and_death_numbers_from_logfile(log_df):
    output = parse_log_file(log_df)

    # Calculate the "incidence rate" from the output counts of incidence
    maternal_counts = output['tlo.methods.labour']['summary_stats_incidence']
    maternal_counts['year'] = pd.to_datetime(maternal_counts['date']).dt.year
    maternal_counts.drop(columns='date', inplace=True)
    maternal_counts.set_index(
        'year',
        drop=True,
        inplace=True
    )

    mmr = maternal_counts['intrapartum_mmr']

    return mmr


maternal_deaths = dict()

for label, file in output_files.items():
    maternal_deaths[label] = \
        get_incidence_rate_and_death_numbers_from_logfile(file)
data = {}


def generate_graphs(dictionary, title, saved_title):
    for label in dictionary.keys():
        data.update({label: dictionary[label]})
    pd.concat(data, axis=1).plot.bar()
    plt.title(f'{title}')
    plt.savefig(outputpath / (f"{saved_title}" + datestamp + ".pdf"), format='pdf')
    plt.show()


generate_graphs(maternal_deaths, 'Maternal Mortality Ratio by Year', "mmr_selected_interventions_death_by_scenario")
