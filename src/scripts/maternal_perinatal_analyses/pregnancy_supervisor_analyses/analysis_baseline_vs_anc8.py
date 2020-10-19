import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.lm import LinearModel, LinearModelType
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

resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# Scenarios Definitions:
# *1: Current coverage/access to ANC in Malawi
# *2: 50% of women will attend 3 ANC visits prior to 30 weeks gestation
# *2: All women attend will attend 3 ANC visits prior to 30 weeks gestation

scenarios = dict()

scenarios['90%_early_anc3_coverage'] = [0.9]
#scenarios['50%_early_anc3_coverage'] = [0.5]
#scenarios['status_quo'] = [0.21]

# Create dict to capture the outputs
# output_files = dict()
output_files = {'90%_early_anc3_coverage': r'./outputs/LogFile__2020-09-28T152554.log',
                '50%_early_anc3_coverage': r'./outputs/LogFile__2020-09-28T165027.log',
                'status_quo': r'./outputs/LogFile__2020-09-28T181445.log'}

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2011, 1, 2)
popsize = 100

#for label, parameters in scenarios.items():
#    # add file handler for the purpose of logging
#    sim = Simulation(start_date=start_date)

#    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
#                 contraception.Contraception(resourcefilepath=resourcefilepath),
#                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
#                 # healthburden.HealthBurden(resourcefilepath=resourcefilepath),
#                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
#                                           service_availability=['*']),
#                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
#                 male_circumcision.male_circumcision(resourcefilepath=resourcefilepath),
#                 hiv.hiv(resourcefilepath=resourcefilepath),
#                 tb.tb(resourcefilepath=resourcefilepath),
#                 antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
#                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
#                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
#                 labour.Labour(resourcefilepath=resourcefilepath),
#                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
#                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))

#    logfile = sim.configure_logging(filename="LogFile")
#    sim.seed_rngs(0)
#    sim.make_initial_population(n=popsize)

#    params_preg_sup = sim.modules['PregnancySupervisor'].parameters

#    params_preg_sup['prob_3_early_visits'] = parameters[0]

#    sim.simulate(end_date=end_date)

    # Save the full set of results:
#    output_files[label] = logfile


def get_incidence_rate_and_death_numbers_from_logfile(logfile):
    output = parse_log_file(logfile)

    # Calculate the "incidence rate" from the output counts of incidence
    preg_sup_counts = output['tlo.methods.pregnancy_supervisor']['summary_stats']
    preg_sup_counts['year'] = pd.to_datetime(preg_sup_counts['date']).dt.year
    preg_sup_counts['year'] = preg_sup_counts['year'] - 1
    preg_sup_counts.drop(columns='date', inplace=True)
    preg_sup_counts.set_index(
        'year',
        drop=True,
        inplace=True
    )

    anc_counts = output['tlo.methods.antenatal_care']['anc_summary_stats']
    anc_counts['year'] = pd.to_datetime(anc_counts['date']).dt.year
    anc_counts['year'] = anc_counts['year'] - 1
    anc_counts.drop(columns='date', inplace=True)
    anc_counts.set_index(
        'year',
        drop=True,
        inplace=True
    )

    anc_counts['diet_supps'] = (anc_counts['diet_supps_6_months']/preg_sup_counts['women_month_6']) * 100
    anc_counts['early_anc3_coverage_women_at_6m'] = (anc_counts['early_anc3']/preg_sup_counts['women_month_6']) * 100
    sbr = preg_sup_counts['antenatal_sbr']
    crude_sb = preg_sup_counts['crude_antenatal_sb']
    early_anc3_coverage_births = anc_counts['early_anc3_proportion_of_births']
    early_anc3_coverage_women = anc_counts['early_anc3_coverage_women_at_6m']
    diet_supps = anc_counts['diet_supps']

    #anc_counts.to_csv(r'./outputs/anc_counts.csv', index=False)
    #preg_sup_counts.to_csv(r'./outputs/preg_sup_counts.csv', index=False)

    return sbr, crude_sb, diet_supps, early_anc3_coverage_births, early_anc3_coverage_women


still_birth_ratio = dict()
still_births = dict()
diet_supplements = dict()
eanc3 = dict()
eanc3women = dict()

for label, file in output_files.items():
    still_birth_ratio[label], still_births[label], diet_supplements[label], eanc3[label], eanc3women[label] = \
        get_incidence_rate_and_death_numbers_from_logfile(file)

data = {}


def generate_graphs(dictionary, title, saved_title):
    for label in dictionary.keys():
        data.update({label: dictionary[label]})
    pd.concat(data, axis=1).plot.bar()
    plt.title(f'{title}')
    plt.savefig(outputpath / (f"{saved_title}" + datestamp + ".pdf"), format='pdf')
    plt.show()
    final_results = pd.DataFrame.from_dict(data)

    if saved_title == "eanc3_by_scenario":
        final_results.to_csv(r'./outputs/eanc3_by_scenario.csv', index=False)

    if saved_title == "eanc3_by_scenario_women":
        final_results.to_csv(r'./outputs/eanc3_by_scenario_women.csv', index=False)

    if saved_title == 'sbr_by_scenario':
        final_results.to_csv(r'./outputs/sbr_by_scenario.csv', index=False)

    if saved_title == "sb_by_scenario":
        final_results.to_csv(r'./outputs/sb_by_scenario.csv', index=False)

    if saved_title == "diet_supp_by_scenario":
        final_results.to_csv(r'./outputs/diet_supp_by_scenario.csv', index=False)


generate_graphs(eanc3, 'Coverage of early initiation of ANC1-3 by scenario (birth denom)', "eanc3_by_scenario")
generate_graphs(eanc3women, 'Coverage of early initiation of ANC1-3 by scenario (women denom)',
                "eanc3_by_scenario_women")
generate_graphs(still_birth_ratio, 'Antenatal SBR with current, 50% and 90% coverage of early ANC3', "sbr_by_scenario")
generate_graphs(still_births, 'Crude number of antenatal stillbirths with current, 50% and 90% coverage of early ANC3',
                              'sb_by_scenario')
generate_graphs(diet_supplements, 'Proportion of women reach 30 weeks gestation who are receiving diet supplements '
                                  'at 30 weeks', 'diet_supp_by_scenario')
