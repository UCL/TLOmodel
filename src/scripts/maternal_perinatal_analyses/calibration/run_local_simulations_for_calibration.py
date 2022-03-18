
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.methods import (
    cardio_metabolic_disorders,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    enhanced_lifestyle,
    healthsystem,
    hiv,
    labour,
    malaria,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)
from tlo.methods.labour import LabourOnsetEvent

resourcefilepath = Path("./resources")


# HELPER FUNCTIONS
def register_modules(sim):
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           cons_availability='all'),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 malaria.Malaria(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath))


def set_logging(config_name, seed):
    log_config = {
        "filename": f"{config_name}_calibration_{seed}",  # The name of the output file (a timestamp will be appended).
        "directory": "./outputs/calibration_files",
        # The default output path is `./outputs`. Change it here, if necessary
        "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
            "*": logging.DEBUG}}

    return log_config


def make_all_women_of_reproductive_age_pregnant_from_sim_start(sim):
    df = sim.population.props

    all = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 49)]
    df.loc[all.index, 'is_pregnant'] = True
    df.loc[all.index, 'date_of_last_pregnancy'] = sim.start_date
    for person in all.index:
        sim.modules['Labour'].set_date_of_labour(person)


def set_whole_population_as_women_of_reproductive_age_pregnant_and_start_sim_at_labour_onset(sim, start_date):
    """Cant be used with HIV module"""

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


def make_all_women_of_reproductive_age_pregnant_and_sim_start_at_labour_onset(sim):
    df = sim.population.props

    all = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    df.loc[all.index, 'is_pregnant'] = True
    for person in all.index:
        df.at[person, 'date_of_last_pregnancy'] = sim.start_date - pd.DateOffset(weeks=35)
        df.at[person, 'la_due_date_current_pregnancy'] = sim.start_date + pd.DateOffset(days=1)
        sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(person)

        sim.schedule_event(LabourOnsetEvent(sim.modules['Labour'], person), df.at[person, 'la_due_date_current_'
                                                                                          'pregnancy'])


def allow_varying_parameter_sets_to_be_used(parameters, sim):
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


def set_whole_population_as_women_of_reproductive_age_and_pregnant_with_ages_correctly_set(sim):
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


# SIMULATION RUNS

def do_run_pregnancy_only(config_name, start_date, end_date, seed, population, parameters):
    log_config = {
        "filename": f"{config_name}_calibration_{seed}",  # The name of the output file (a timestamp will be appended).
        "directory": "./outputs/calibration_files",
        # The default output path is `./outputs`. Change it here, if necessary
        "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
            "*": logging.DEBUG}}

    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)
    register_modules(sim)
    sim.make_initial_population(n=population)
    make_all_women_of_reproductive_age_pregnant_from_sim_start(sim)
    allow_varying_parameter_sets_to_be_used(parameters, sim)
    sim.simulate(end_date=end_date)


def do_labour_run_only(config_name, start_date, end_date, seed, population, parameters):
    """This wont run with HIV"""
    log_config = set_logging(config_name, seed)

    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)
    register_modules(sim)
    sim.make_initial_population(n=population)
    set_whole_population_as_women_of_reproductive_age_pregnant_and_start_sim_at_labour_onset(sim, start_date)
    allow_varying_parameter_sets_to_be_used(parameters, sim)
    sim.simulate(end_date=end_date)


def age_corrected_run_with_all_women_pregnant_at_baseline(config_name, start_date, end_date, seed, population,
                                                          parameters):
    log_config = set_logging(config_name, seed)

    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)
    register_modules(sim)
    sim.make_initial_population(n=population)
    set_whole_population_as_women_of_reproductive_age_and_pregnant_with_ages_correctly_set(sim)
    allow_varying_parameter_sets_to_be_used(parameters, sim)
    sim.simulate(end_date=end_date)


def do_run_using_dummy_contraception(config_name, start_date, end_date, seed, population, parameters):
    log_config = set_logging(config_name, seed)

    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)
    register_modules(sim)
    sim.make_initial_population(n=population)
    allow_varying_parameter_sets_to_be_used(parameters, sim)
    sim.simulate(end_date=end_date)


def do_run_forcing_complication_pregnancy_to_look_at_cfr(config_name, start_date, end_date, seed, population,
                                                         parameters):
    log_config = set_logging(config_name, seed)
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)
    register_modules(sim)
    sim.make_initial_population(n=population)
    # make_all_women_of_reproductive_age_pregnant_from_sim_start(sim)
    make_all_women_of_reproductive_age_pregnant_and_sim_start_at_labour_onset(sim)
    allow_varying_parameter_sets_to_be_used(parameters, sim)

    sim.modules['Labour'].current_parameters['prob_obstruction_cpd'] = 0
    sim.modules['Labour'].current_parameters['prob_obstruction_malpos_malpres'] = 1
    sim.modules['Labour'].current_parameters['prob_obstruction_other'] = 1

    # sim.modules['PregnancySupervisor'].current_parameters['prob_chorioamnionitis'] = 1
    # sim.modules['PregnancySupervisor'].current_parameters['prob_prom_per_month'] = 0

    sim.simulate(end_date=end_date)


def normal_run(config_name, start_date, end_date, seed, population, parameters):
    log_config = set_logging(config_name, seed)
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)
    register_modules(sim)
    sim.make_initial_population(n=population)
    allow_varying_parameter_sets_to_be_used(parameters, sim)

    sim.simulate(end_date=end_date)


do_run_forcing_complication_pregnancy_to_look_at_cfr('test_avd_is_running', Date(2010, 1, 1),
                                                     Date(2010, 2, 1), 111, 1000, 2010)

# normal_run('test_lbw_logging_less_eptb', Date(2010, 1, 1), Date(2011, 1, 1), 77, 10000, 2010)
# normal_run('anc1_checker_15', Date(2010, 1, 1), Date(2011, 1, 1), 2, 10000, 2015)
