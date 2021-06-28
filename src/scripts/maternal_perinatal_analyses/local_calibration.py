
from pathlib import Path
import pandas as pd
from tlo.analysis.utils import parse_log_file
from tlo import Date, Simulation, logging
from tlo.methods.labour import LabourOnsetEvent
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    enhanced_lifestyle,
    healthsystem,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    joes_fake_props_module,
    symptommanager
)


# todo: just for reference, can be deleted
antenatal_comps = ['spontaneous_abortion', 'induced_abortion', 'spontaneous_abortion_haemorrhage',
                   'induced_abortion_haemorrhage', 'spontaneous_abortion_sepsis',
                   'induced_abortion_sepsis', 'spontaneous_abortion_injury',
                   'induced_abortion_complication', 'complicated_induced_abortion',
                   'complicated_spontaneous_abortion', 'iron_deficiency', 'folate_deficiency', 'b12_deficiency',
                   'mild_anaemia', 'moderate_anaemia', 'severe_anaemia', 'gest_diab',
                   'mild_pre_eclamp', 'mild_gest_htn', 'severe_pre_eclamp', 'eclampsia', 'severe_gest_htn',
                   'placental_abruption', 'severe_antepartum_haemorrhage', 'mild_mod_antepartum_haemorrhage',
                   'clinical_chorioamnionitis', 'PROM', 'ectopic_unruptured', 'multiple_pregnancy', 'placenta_praevia',
                   'ectopic_ruptured', 'syphilis']

resourcefilepath = Path("./resources")

def register_modules(sim):
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*']),
                 joes_fake_props_module.JoesFakePropsModule(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath))


def set_pregnant_pop(sim, start_date):
    df = sim.population.props

    all = df.loc[df.is_alive]
    df.loc[all.index, 'sex'] = 'F'
    df.loc[all.index, 'is_pregnant'] = True
    df.loc[all.index, 'date_of_last_pregnancy'] = sim.start_date
    for person in all.index:
        age = sim.rng.randint(16, 49)
        df.at[person, 'age_years'] = age
        df.at[person, 'age_exact_years'] = float(age)
        df.at[person, 'age_days'] = age * 365
        df.at[person, 'date_of_birth'] = start_date - pd.DateOffset(days=(age * 365))

        sim.modules['Labour'].set_date_of_labour(person)


def set_pregnant_pop_age_correct(sim):
    df = sim.population.props

    all = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years >14) & (df.age_years < 49)]
    df.loc[all.index, 'is_pregnant'] = True
    df.loc[all.index, 'date_of_last_pregnancy'] = sim.start_date
    for person in all.index:
        sim.modules['Labour'].set_date_of_labour(person)

def set_labour_pop(sim, start_date):
    df = sim.population.props

    all = df.loc[df.is_alive]
    df.loc[all.index, 'sex'] = 'F'
    df.loc[all.index, 'is_pregnant'] = True
    for person in all.index:
        age = sim.rng.randint(16, 49)
        df.at[person, 'age_years'] = age
        df.at[person, 'age_exact_years'] = float(age)
        df.at[person, 'age_days'] = age * 365
        df.at[person, 'date_of_birth'] = start_date - pd.DateOffset(days=(age * 365))

        df.at[person, 'date_of_last_pregnancy'] = sim.start_date - pd.DateOffset(weeks=35)
        df.at[person, 'la_due_date_current_pregnancy'] = sim.start_date + pd.DateOffset(days=1)
        sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(person)

        sim.schedule_event(LabourOnsetEvent(sim.modules['Labour'], person),
                           df.at[person, 'la_due_date_current_pregnancy'])

def set_labour_pop_age_correct(sim):
    df = sim.population.props

    all = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    df.loc[all.index, 'is_pregnant'] = True
    for person in all.index:
        df.at[person, 'date_of_last_pregnancy'] = sim.start_date - pd.DateOffset(weeks=35)
        df.at[person, 'la_due_date_current_pregnancy'] = sim.start_date + pd.DateOffset(days=1)
        sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(person)

        sim.schedule_event(LabourOnsetEvent(sim.modules['Labour'], person), df.at[person,
                                                                                  'la_due_date_current_pregnancy'])


def do_run_pregnancy_only(config_name, start_date, end_date, seed, population, parameters, age_correct):
    log_config = {
        "filename": f"{config_name}_calibration_{seed}",  # The name of the output file (a timestamp will be appended).
        "directory": "./outputs/calibration_files",
        # The default output path is `./outputs`. Change it here, if necessary
        "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
            "*": logging.DEBUG}}

    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)
    register_modules(sim)
    sim.make_initial_population(n=population)
    if not age_correct:
        set_pregnant_pop(sim, start_date)
    else:
        set_pregnant_pop_age_correct(sim)

    if parameters == 2015:
        def switch_parameters(master_params, current_params):
            for key, value in current_params.items():
                current_params[key] = master_params[key][1]

        switch_parameters(sim.modules['PregnancySupervisor'].parameters,
                          sim.modules['PregnancySupervisor'].current_parameters)
        switch_parameters(sim.modules['CareOfWomenDuringPregnancy'].parameters,
                          sim.modules['CareOfWomenDuringPregnancy'].current_parameters)
        switch_parameters(sim.modules['Labour'].parameters,
                          sim.modules['Labour'].current_parameters)
        switch_parameters(sim.modules['NewbornOutcomes'].parameters,
                          sim.modules['NewbornOutcomes'].current_parameters)
        switch_parameters(sim.modules['PostnatalSupervisor'].parameters,
                          sim.modules['PostnatalSupervisor'].current_parameters)

    sim.simulate(end_date=end_date)


def do_labour_run_only(config_name, start_date, end_date, seed, population, parameters):
    log_config = {
        "filename": f"{config_name}_calibration_{seed}",  # The name of the output file (a timestamp will be appended).
        "directory": "./outputs/calibration_files",
        # The default output path is `./outputs`. Change it here, if necessary
        "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
            "*": logging.DEBUG}}

    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)
    register_modules(sim)
    sim.make_initial_population(n=population)
    set_labour_pop_age_correct(sim)

    if parameters == 2015:
        def switch_parameters(master_params, current_params):
            for key, value in current_params.items():
                current_params[key] = master_params[key][1]

        switch_parameters(sim.modules['PregnancySupervisor'].parameters,
                          sim.modules['PregnancySupervisor'].current_parameters)
        switch_parameters(sim.modules['CareOfWomenDuringPregnancy'].parameters,
                          sim.modules['CareOfWomenDuringPregnancy'].current_parameters)
        switch_parameters(sim.modules['Labour'].parameters,
                          sim.modules['Labour'].current_parameters)
        switch_parameters(sim.modules['NewbornOutcomes'].parameters,
                          sim.modules['NewbornOutcomes'].current_parameters)
        switch_parameters(sim.modules['PostnatalSupervisor'].parameters,
                          sim.modules['PostnatalSupervisor'].current_parameters)

    sim.simulate(end_date=end_date)


def do_normal_run_all_pregnant(config_name, start_date, end_date, seed, population, parameters):
    log_config = {
        "filename": f"{config_name}_calibration_{seed}",  # The name of the output file (a timestamp will be appended).
        "directory": "./outputs/calibration_files",
        # The default output path is `./outputs`. Change it here, if necessary
        "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
            "*": logging.DEBUG}}

    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)
    register_modules(sim)
    sim.make_initial_population(n=population)

    df = sim.population.props
    all = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years <50)]
    df.loc[all.index, 'is_pregnant'] = True
    df.loc[all.index, 'date_of_last_pregnancy'] = sim.start_date
    for person in all.index:
        sim.modules['Labour'].set_date_of_labour(person)

    if parameters == 2015:
        def switch_parameters(master_params, current_params):
            for key, value in current_params.items():
                current_params[key] = master_params[key][1]

        switch_parameters(sim.modules['PregnancySupervisor'].parameters,
                          sim.modules['PregnancySupervisor'].current_parameters)
        switch_parameters(sim.modules['CareOfWomenDuringPregnancy'].parameters,
                          sim.modules['CareOfWomenDuringPregnancy'].current_parameters)
        switch_parameters(sim.modules['Labour'].parameters,
                          sim.modules['Labour'].current_parameters)
        switch_parameters(sim.modules['NewbornOutcomes'].parameters,
                          sim.modules['NewbornOutcomes'].current_parameters)
        switch_parameters(sim.modules['PostnatalSupervisor'].parameters,
                          sim.modules['PostnatalSupervisor'].current_parameters)

    sim.simulate(end_date=end_date)

# Get the log

seeds = [110]
for seed in seeds:
    do_run_pregnancy_only(config_name='death_test_2010_more_births', start_date=Date(2010, 1, 1),
                          end_date=Date(2012, 1, 1),
                          seed=seed, population=10000, parameters=2010, age_correct=True)
    do_run_pregnancy_only(config_name='death_test_2015_more_births', start_date=Date(2015, 1, 1),
                          end_date=Date(2017, 1, 1),
                          seed=seed, population=10000, parameters=2015, age_correct=True)
