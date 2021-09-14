
from pathlib import Path
import pandas as pd
import numpy as np
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
    joes_fake_props_module, dummy_contraception,
    symptommanager, malaria, hiv, cardio_metabolic_disorders, depression, dx_algorithm_child, dx_algorithm_adult
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
                 #contraception.Contraception(resourcefilepath=resourcefilepath),
                 dummy_contraception.DummyContraceptionModule(),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           ignore_cons_constraints=True),
                 #joes_fake_props_module.JoesFakePropsModule(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
                 malaria.Malaria(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath),
                 #hiv.DummyHivModule(),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath))


def set_pregnant_pop_at_baseline(sim, start_date):
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

        # Set length of pregnancy
        days_pregnant = sim.rng.randint(0, 245)
        df.at[person, 'date_of_last_pregnancy'] = start_date - pd.DateOffset(days=(days_pregnant))
        sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(person)

        # Schedule labour
        days_till_term = 245 - days_pregnant
        df.at[person, 'la_due_date_current_pregnancy'] = start_date + pd.DateOffset(days=(days_till_term))
        sim.schedule_event(LabourOnsetEvent(sim.modules['Labour'], person),
                           df.at[person, 'la_due_date_current_pregnancy'])


def set_pregnant_pop(sim, start_date):
    df = sim.population.props

    all = df.loc[df.is_alive]
    df.loc[all.index, 'sex'] = 'F'
    df.loc[all.index, 'is_pregnant'] = True
    df.loc[all.index, 'date_of_last_pregnancy'] = sim.start_date


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
    # set_labour_pop_age_correct(sim)
    set_labour_pop(sim, start_date)

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

    sim.modules['Labour'].current_parameters['prob_obstruction_cpd'] = 1

    sim.simulate(end_date=end_date)


def age_corrected_run_with_all_women_pregnant_at_baseline(config_name, start_date, end_date, seed, population,
                                                          parameters):
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
    all = df.loc[df.is_alive]
    df.loc[all.index, 'sex'] = 'F'
    df.loc[all.index, 'is_pregnant'] = True
    df.loc[all.index, 'date_of_last_pregnancy'] = sim.start_date

    init_pop = sim.modules['Demography'].parameters['pop_2010']

    too_young = init_pop.loc[init_pop['Age'] < 15]
    young_adjusted_init_pop = init_pop.drop(index=too_young.index)

    too_old = young_adjusted_init_pop.loc[young_adjusted_init_pop['Age'] > 49]
    old_adjusted_init_pop = young_adjusted_init_pop.drop(index=too_old.index)

    male = old_adjusted_init_pop.loc[old_adjusted_init_pop['Sex'] == 'M']
    final_init_pop = old_adjusted_init_pop.drop(index=male.index)

    final_init_pop['prob'] = final_init_pop['Count'] / final_init_pop['Count'].sum()

    # TODO: note, should we also reassign the region and districts?
    demog_char_to_assign = final_init_pop.loc[sim.rng.choice(final_init_pop.index.values,
                                                             size=len(all),
                                                             replace=True,
                                                             p=final_init_pop.prob)][['District',
                                                                                      'District_Num',
                                                                                      'Region',
                                                                                      'Age']].reset_index(drop=True)
    # make a date of birth that is consistent with the allocated age of each person
    demog_char_to_assign['days_since_last_birthday'] = sim.rng.randint(0, 365, len(demog_char_to_assign))

    demog_char_to_assign['date_of_birth'] = \
        [sim.date - pd.DateOffset(years=int(demog_char_to_assign['Age'][i]),
                                  days=int(demog_char_to_assign['days_since_last_birthday'][i])) for i in
         demog_char_to_assign.index]
    demog_char_to_assign['age_in_days'] = sim.date - demog_char_to_assign['date_of_birth']

    df.loc[all.index, 'date_of_birth'] = demog_char_to_assign['date_of_birth']
    df.loc[all.index, 'age_exact_years'] = demog_char_to_assign['age_in_days'] / np.timedelta64(1, 'Y')
    df.loc[all.index, 'age_years'] = df.loc[all.index, 'age_exact_years'].astype('int64')
    df.loc[all.index, 'age_range'] = df.loc[all.index, 'age_years'].map(sim.modules['Demography'].AGE_RANGE_LOOKUP)
    df.loc[all.index, 'age_days'] = demog_char_to_assign['age_in_days'].dt.days
    df.loc[df.is_alive, 'district_num_of_residence'] = demog_char_to_assign['District_Num'].values[:]
    df.loc[df.is_alive, 'district_of_residence'] = demog_char_to_assign['District'].values[:]
    df.loc[df.is_alive, 'region_of_residence'] = demog_char_to_assign['Region'].values[:]

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


def do_run_using_dummy_contraception(config_name, start_date, end_date, seed, population, parameters):
    log_config = {
        "filename": f"{config_name}_calibration_{seed}",
        "directory": "./outputs/calibration_files",
        "custom_levels": {
            "*": logging.DEBUG}}

    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)
    register_modules(sim)
    sim.make_initial_population(n=population)

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


do_run_using_dummy_contraception(config_name='test_num_pregs', start_date=Date(2010, 1, 1),
                                 end_date=Date(2014, 1, 1), seed=333, population=7500, parameters=2010)

