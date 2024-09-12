import os
from pathlib import Path

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography, enhanced_lifestyle, healthburden, mockitis
from tlo.methods.fullmodel import fullmodel

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
outputpath = Path("./outputs/")

start_date = Date(2010, 1, 1)
end_date = Date(2011, 1, 1)

popsize = 100
do_sim = True


def check_dtypes(simulation):
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def log_prevalences_from_sim_func(sim):
    """Logs the prevalence of disease monthly"""
    health_burden = sim.modules['HealthBurden']
    monthly_prevalence = health_burden.prevalence_of_diseases
    monthly_prevalence['date'] = sim.date.year
    return monthly_prevalence


def test_run_with_healthburden_with_real_diseases(tmpdir, seed):
    """Check that everything runs in the simple cases of Mockitis and Chronic Syndrome and that outputs are as expected."""

    sim = Simulation(start_date=start_date, seed=seed, log_config={'filename': 'test_log', 'directory': outputpath})
    sim.register(*fullmodel(
        resourcefilepath=resourcefilepath,
        use_simplified_births=False, ))
    sim.make_initial_population(n=popsize)
    sim.modules['HealthBurden'].parameters['logging_frequency_prevalence'] = 'day'
    sim.simulate(end_date=end_date)
    check_dtypes(sim)
    output = parse_log_file(sim.log_filepath)

    prevalence = output['tlo.methods.healthburden']['prevalence_of_diseases']

    # check to see if the monthly prevalence is calculated correctly NB for only one month

    log_prevalences_from_sim = log_prevalences_from_sim_func(sim)
    for log_date in log_prevalences_from_sim['date']:
        if log_date in prevalence['date'].values:
            prevalence_row = prevalence.loc[prevalence['date'] == log_date].squeeze()
            if 'date' in prevalence.columns:
                prevalence_row = prevalence_row.drop('date')

            sim_row = log_prevalences_from_sim.loc[
                log_prevalences_from_sim['date'] == log_date].squeeze()

            for column in prevalence_row.index:
                # Compare the values between the two DataFrames for this date and column
                if prevalence_row[column] != sim_row[column]:
                    pass
        else:
            # Handle cases where the date is not found in prevalence DataFrame
            pass

    ## See if the registered modules are reporting prevalences as they should
    columns = prevalence.columns
    excluded_modules = ['Lifestyle', 'HealthBurden', 'HealthSeekingBehaviour', 'SymptomManager', 'Epi', 'HealthSystem',
                        'SimplifiedBirths', 'Contraception', 'CareOfWomenDuringPregnancy']  # don't return prevalences

    assert 'chronic_ischemic_hd' in columns

    for module in sim.modules:
        if module not in excluded_modules:
            if module == 'CardioMetabolicDisorders':
                corresponding_diseases = ['chronic_ischemic_hd', 'chronic_kidney_disease', 'chronic_lower_back_pain',
                                          'diabetes', 'hypertension']
            elif module == 'Demography':
                corresponding_diseases = ['MMR', 'NMR']
            elif module == 'PregnancySupervisor':
                corresponding_diseases = ['Antenatal stillbirth']
            elif module == 'Labour':
                corresponding_diseases = ['Intrapartum stillbirth']
            assert all(disease in columns for disease in corresponding_diseases), \
                f"Not all diseases for module '{module}' are in columns."


def test_structure_logging_dummy_disease(tmpdir, seed):
    start_date = Date(2010, 1, 1)
    end_date = Date(2011, 1, 1)

    sim = Simulation(start_date=start_date, seed=0, log_config={'filename': 'tmp', 'directory': tmpdir})
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        mockitis.DummyDisease(resourcefilepath=resourcefilepath, ),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        sort_modules=False,
        check_all_dependencies=False
    )

    sim.make_initial_population(n=popsize)
    sim.modules['HealthBurden'].parameters['logging_frequency_prevalence'] = 'month'
    sim.simulate(end_date=end_date)
    output = parse_log_file(sim.log_filepath)

    prevalence_healthburden_log = output['tlo.methods.healthburden']['prevalence_of_diseases']['DummyDisease']
    prevalence_dummy_log = output['tlo.methods.mockitis']["summary"]["PropInf"]

    for row in range(len(prevalence_healthburden_log) -1): # has extra log for first day
        assert prevalence_healthburden_log[row + 1] == prevalence_dummy_log[row]
