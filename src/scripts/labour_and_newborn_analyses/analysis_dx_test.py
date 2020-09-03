import datetime
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation
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
    symptommanager,
)

resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# Scenarios Definitions:
# *1: All complications occuring in facilities will be identified correctly and prompt treatment administered
# *2: Set dx_test sensitivity is applied

scenarios = dict()
scenarios['dx_test_total_sensitivity'] = [1.0, 0.99]
scenarios['dx_test_standard_sensitivity'] = [0.8, 0.4]

# Create dict to capture the outputs
output_files = dict()

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2013, 1, 2)
popsize = 20000

for label, dx_test_specificity in scenarios.items():
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

    params['sensitivity_of_assessment_of_obstructed_labour_hc'] = dx_test_specificity[1]
    params['sensitivity_of_assessment_of_obstructed_labour_hp'] = dx_test_specificity[0]
    params['sensitivity_of_assessment_of_sepsis_hc'] = dx_test_specificity[1]
    params['sensitivity_of_assessment_of_sepsis_hp'] = dx_test_specificity[0]
    params['sensitivity_of_assessment_of_hypertension_hc'] = dx_test_specificity[1]
    params['sensitivity_of_assessment_of_hypertension_hp'] = dx_test_specificity[0]
    params['sensitivity_of_assessment_of_severe_pe_hc'] = dx_test_specificity[1]
    params['sensitivity_of_assessment_of_severe_pe_hp'] = dx_test_specificity[0]
    params['sensitivity_of_referral_assessment_of_antepartum_haem_hc'] = dx_test_specificity[1]
    params['sensitivity_of_treatment_assessment_of_antepartum_haem_hp'] = dx_test_specificity[0]
    params['sensitivity_of_referral_assessment_of_uterine_rupture_hc'] = dx_test_specificity[1]
    params['sensitivity_of_treatment_assessment_of_uterine_rupture_hp'] = dx_test_specificity[0]

    params_nb = sim.modules['NewbornOutcomes'].parameters

    params['sensitivity_of_assessment_of_neonatal_sepsis_hc'] = dx_test_specificity[1]
    params['sensitivity_of_assessment_of_neonatal_sepsis_hp'] = dx_test_specificity[0]
    params['sensitivity_of_assessment_of_ftt_hc'] = dx_test_specificity[1]
    params['sensitivity_of_assessment_of_ftt_hp'] = dx_test_specificity[0]
    params['sensitivity_of_assessment_of_lbw_hc'] = dx_test_specificity[1]
    params['sensitivity_of_assessment_of_lbw_hp'] = dx_test_specificity[0]

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


generate_graphs(maternal_deaths, 'Maternal Mortality Ratio by Year', "mmr_dx_death_by_scenario")
