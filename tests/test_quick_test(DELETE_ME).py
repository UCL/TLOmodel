import os
from pathlib import Path
import pandas as pd
from tlo.methods.labour import LabourOnsetEvent

from tlo import Date, Simulation, logging
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    dx_algorithm_adult,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    malaria,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
     joes_fake_props_module, cardio_metabolic_disorders, dummy_contraception
)

seed = 228

# The resource files
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')

start_date = Date(2010, 1, 1)


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def set_all_women_as_pregnant_and_reset_baseline_parity(sim):
    """Force all women of reproductive age to be pregnant at the start of the simulation and overrides parity set at
     initialisation of simulation """
    df = sim.population.props

    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    df.loc[women_repro.index, 'is_pregnant'] = True
    df.loc[women_repro.index, 'date_of_last_pregnancy'] = sim.start_date
    for person in women_repro.index:
        sim.modules['Labour'].set_date_of_labour(person)

    #all_women = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14)]
    #df.loc[all_women.index, 'la_parity'] = 0


def set_all_women_to_go_into_labour(sim):
    df = sim.population.props

    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    df.loc[women_repro.index, 'is_pregnant'] = True

    for person in women_repro.index:
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


def set_population_to_women_and_to_go_into_labour(sim):
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




def register_all_modules():
    """Defines sim variable and registers all modules that can be called when running the full suite of pregnancy
    modules"""

    log_config = {
        "filename": "parity_200k",  # The name of the output file (a timestamp will be appended).
        "directory": "./outputs",
        # The default output path is `./outputs`. Change it here, if necessary
        "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
            "*": logging.DEBUG,  # Asterisk matches all loggers - we set the default level to WARNING
            "tlo.methods.demography": logging.DEBUG,
            "tlo.methods.labour": logging.DEBUG,
            "tlo.methods.healthsystem": logging.DEBUG,
            "tlo.methods.hiv": logging.DEBUG,
            "tlo.methods.newborn_outcomes": logging.DEBUG,
            "tlo.methods.antenatal_care": logging.DEBUG,
            "tlo.methods.pregnancy_supervisor": logging.DEBUG,
            "tlo.methods.postnatal_supervisor": logging.DEBUG,
            "tlo.methods.hsi_generic_first_appts": logging.DEBUG,

        }
    }

    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 #dummy_contraception.DummyContraceptionModule(),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           ignore_cons_constraints=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 malaria.Malaria(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath),
                 #hiv.DummyHivModule(),
                 dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 #joes_fake_props_module.JoesFakePropsModule(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 )

    return sim


def test_run_core_modules_normal_allocation_of_pregnancy():
    """Runs the simulation using only core modules without manipulation of pregnancy rates or parameters and checks
    dtypes at the end"""

    sim = register_all_modules()
    sim.make_initial_population(n=10000)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    sim.simulate(end_date=Date(2011, 1, 1))
    check_dtypes(sim)


def test_run_all_labour():
    """Runs the simulation using only core modules without manipulation of pregnancy rates or parameters and checks
    dtypes at the end"""

    sim = register_all_modules()
    sim.make_initial_population(n=2000)
    set_all_women_to_go_into_labour(sim)
    #sim.modules['Labour'].current_parameters['prob_ip_still_birth'] = 1
    sim.modules['Labour'].current_parameters['odds_will_attend_pnc'] = 100
    sim.modules['Labour'].current_parameters['prob_timings_pnc'] = [1, 0]
    sim.modules['Labour'].current_parameters['prob_pph_other_causes'] = 0

    sim.simulate(end_date=Date(2010, 2, 1))
    check_dtypes(sim)

def test_run_all_pregnant():
    """Runs the simulation using only core modules without manipulation of pregnancy rates or parameters and checks
    dtypes at the end"""

    sim = register_all_modules()
    sim.make_initial_population(n=20000)
    df = sim.population.props
    all = df.loc[df.is_alive]
    df.loc[all.index, 'sex'] = 'F'
    df.loc[all.index, 'is_pregnant'] = True
    df.loc[all.index, 'date_of_last_pregnancy'] = sim.start_date
    for person in all.index:
        sim.modules['Labour'].set_date_of_labour(person)



    sim.simulate(end_date=Date(2011, 1, 1))
    check_dtypes(sim)


#test_run_all_pregnant()
#test_run_core_modules_normal_allocation_of_pregnancy()
test_run_all_labour()
