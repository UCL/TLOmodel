import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tlo import Date, Simulation, logging, Parameter
from tlo.analysis.utils import (
    parse_log_file,
)

from tlo.methods import demography, contraception, labour, enhanced_lifestyle, newborn_outcomes, healthsystem, \
    pregnancy_supervisor, antenatal_care, symptommanager, healthseekingbehaviour

resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

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
    sim = Simulation(start_date=start_date)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']))
    sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
    sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(labour.Labour(resourcefilepath=resourcefilepath))
    sim.register(newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath))
    sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))
    sim.register(antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath))

    logfile = sim.configure_logging(filename="LogFile")
    sim.seed_rngs(0)
    sim.make_initial_population(n=popsize)

    params = sim.modules['Labour'].parameters
    params['allowed_interventions'] = allowed_interventions

    sim.simulate(end_date=end_date)

    # Save the full set of results:
    output_files[label] = logfile


def get_incidence_rate_and_death_numbers_from_logfile(logfile):
    output = parse_log_file(logfile)

    # Calculate the "incidence rate" from the output counts of incidence
    maternal_counts = output['tlo.methods.labour']['summary_stats_incidence']
    maternal_counts['year'] = pd.to_datetime(maternal_counts['date']).dt.year
    maternal_counts.drop(columns='date', inplace=True)
    maternal_counts.set_index(
        'year',
        drop=True,
        inplace=True
    )
    newborn_counts = output['tlo.methods.newborn_outcomes']['summary_stats']
    newborn_counts['year'] = pd.to_datetime(newborn_counts['date']).dt.year
    newborn_counts.drop(columns='date', inplace=True)
    newborn_counts.set_index(
        'year',
        drop=True,
        inplace=True
    )
    maternal_deaths = output['tlo.methods.labour']['summary_stats_death']
    maternal_deaths['year'] = pd.to_datetime(maternal_deaths['date']).dt.year
    maternal_deaths.drop(columns='date', inplace=True)
    maternal_deaths.set_index(
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
