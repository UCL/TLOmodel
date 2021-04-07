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
    hiv,
    labour,
    male_circumcision,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
    tb,
)

seed = 563

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
end_date = Date(2015, 1, 2)
popsize = 20000

for label, service_avail in scenarios.items():
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

    # mmr = maternal_counts['intrapartum_mmr']
    # sbr = maternal_counts['sbr']
    # aph_incidence = maternal_counts['aph_incidence']
    # ol_incidence = maternal_counts['ur_incidence']
    # ur_incidence = maternal_counts['ol_incidence']
    # sepsis_incidence = maternal_counts['sep_incidence']
    # eclampsia_incidence = maternal_counts['ec_incidence']
    # pph_incidence = maternal_counts['pph_incidence']
    # home_birth_rate = maternal_counts['home_births_prop']
    # health_centre_rate = maternal_counts['health_centre_births']
    # hospital_rate = maternal_counts['hospital_births']
    # caesarean_section_rate = maternal_counts['cs_delivery_rate']
    # nmr = newborn_counts['nmr_early']
    # neonatal_sepsis  = newborn_counts['nmr_early']
    # ftt  = newborn_counts['nmr_early']
    # encephalopathy  = newborn_counts['nmr_early']

    # sep_deaths = maternal_deaths['sepsis']
    # ur_deaths = maternal_deaths['uterine_rupture']
    # aph_deaths = maternal_deaths['aph']
    # ec_deaths = maternal_deaths['eclampsia']
    # pph_deaths = maternal_deaths['postpartum_haem']

    # sep_deaths = maternal_deaths['sepsis']
    # ur_deaths = maternal_deaths['uterine_rupture']
    # aph_deaths = maternal_deaths['aph']
    # ec_deaths = maternal_deaths['eclampsia']
    # pph_deaths = maternal_deaths['postpartum_haem']

    pt_d = newborn_counts['preterm_birth_death']
    ftt_d = newborn_counts['ftt_death']
    enc_d = newborn_counts['total_enceph_death']
    nb_sep_d = newborn_counts['sepsis_deaths']

    # return mmr, nmr,  sbr, aph_incidence, ol_incidence, ur_incidence, sepsis_incidence, eclampsia_incidence, \
    #       pph_incidence, home_birth_rate, health_centre_rate, hospital_rate, caesarean_section_rate,
    # return home_birth_rate, health_centre_rate, hospital_rate
    # return sep_deaths, ur_deaths, aph_deaths, ec_deaths, pph_deaths
    return pt_d, ftt_d, enc_d, nb_sep_d

# maternal_deaths = dict()
# newborn_deaths = dict()
# still_births = dict()
# antepartum_haem = dict()
# obstructed_labour = dict()
# uterine_rupture = dict()
# maternal_sepsis = dict()
# eclampsia = dict()
# postpartum_haem = dict()
# home_births = dict()
# health_centre_births = dict()
# hospital_births = dict()
# caesarean_births = dict()
# sepsis_deaths = dict()
# uterine_rupture_deaths = dict()
# antepartum_haem_deaths = dict()
# eclampsia_deaths = dict()
# postpartum_haem_deaths = dict()


preterm_birth_deaths = dict()
failure_to_transition_deaths = dict()
encephalopathy_death = dict()
newborn_sepsis_death = dict()


# for label, file in output_files.items():
#    maternal_deaths[label], still_births[label], newborn_deaths[label], antepartum_haem[label], \
#    obstructed_labour[label], uterine_rupture[label], maternal_sepsis[label], eclampsia[label],\
#        postpartum_haem[label], home_births[label], health_centre_births[label], hospital_births[label],\
#        caesarean_births[label] = \
#        get_incidence_rate_and_death_numbers_from_logfile(file)

# for label, file in output_files.items():
#    sepsis_deaths[label], uterine_rupture_deaths[label], antepartum_haem_deaths[label], eclampsia_deaths[label], \
#    postpartum_haem_deaths[label] = get_incidence_rate_and_death_numbers_from_logfile(file)

# for label, file in output_files.items():
#    home_births[label], health_centre_births[label], hospital_births[label] = \
#        get_incidence_rate_and_death_numbers_from_logfile(file)

for label, file in output_files.items():
    preterm_birth_deaths[label], failure_to_transition_deaths[label], encephalopathy_death[label], \
        newborn_sepsis_death[label] = \
        get_incidence_rate_and_death_numbers_from_logfile(file)
data = {}


def generate_graphs(dictionary, title, saved_title):
    for label in dictionary.keys():
        data.update({label: dictionary[label]})
    pd.concat(data, axis=1).plot.bar()
    plt.title(f'{title}')
    plt.savefig(outputpath / (f"{saved_title}" + datestamp + ".pdf"), format='pdf')
    plt.show()


generate_graphs(preterm_birth_deaths, 'Preterm Birth Deaths by Year', "ptb_death_by_scenario")
generate_graphs(failure_to_transition_deaths, 'Failure To Transition Deaths by Year', "ftt_death_by_scenario")
generate_graphs(encephalopathy_death, 'Neonatal Encephalopathy Deaths by Year', "nenc_death_by_scenario")
generate_graphs(newborn_sepsis_death, 'Neonatal Sepsis Deaths by Year', "nsep_death_by_scenario")

# generate_graphs(sepsis_deaths, 'Maternal Sepsis Deaths by Year', "sep_death_by_scenario")
# generate_graphs(uterine_rupture_deaths, 'Uterine Rupture Deaths by Year', "ur_death_by_scenario")
# generate_graphs(antepartum_haem_deaths, 'Antepartum Haemorrhage Deaths by Year', "aph_death_by_scenario")
# generate_graphs(eclampsia_deaths, 'Eclampsia Deaths by Year', "ec_death_by_scenario")
# generate_graphs(postpartum_haem_deaths, 'Postpartum Haem Deaths by Year', "pph_death_by_scenario")
# generate_graphs(maternal_deaths, 'Maternal Mortality Ratio by Year', "MMR_by_scenario")
# generate_graphs(newborn_deaths, 'Early Neonatal Mortality Ratio by Year', "NMR_by_scenario")
# generate_graphs(still_births, 'Intrapartum Stillbirth Rate by Year ', "SBR_by_scenario")
# generate_graphs(antepartum_haem, 'Antepartum Haemorrhage Rate by Year ', "APH_by_scenario")
# generate_graphs(obstructed_labour, 'Obstructed Labour Rate by Year ', "OL_by_scenario")
# generate_graphs(uterine_rupture, 'Uterine Rupture Rate by Year ', "UR_by_scenario")
# generate_graphs(maternal_sepsis, 'Maternal Sepsis Rate by Year ', "MS_by_scenario")
# generate_graphs(eclampsia, 'Eclampsia Rate by Year ', "ER_by_scenario")
# generate_graphs(postpartum_haem, 'Postpartum Haemorrhage Rate by Year ', "PPH_by_scenario")


# data2 = {}
# data3 = {}
# for label in home_births.keys():
#    data.update({label: home_births[label]})
# for label in hospital_births.keys():
#    data2.update({label: hospital_births[label]})
# for label in health_centre_births.keys():
#    data3.update({label: health_centre_births[label]})

# fig, ax = plt.subplots()
# pd.concat(data, axis=1).plot.bar()
# pd.concat(data2, axis=1).plot.bar()
# pd.concat(data3, axis=1).plot.bar()
# plt.title('Births by Setting')
# plt.savefig(outputpath / ("birth_setting" + datestamp + ".pdf"), format='pdf')
# plt.show()
