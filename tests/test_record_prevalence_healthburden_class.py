import os
from pathlib import Path

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    mockitis,
    symptommanager,
)
from tlo.methods.fullmodel import fullmodel

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
outputpath = Path("./outputs/")

start_date = Date(2010, 1, 1)
end_date = Date(2011, 1, 1)

popsize = 1000
do_sim = True
def check_dtypes(simulation):
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def find_closest_recording(prevalence, target_date, log_value, column_name, multiply_by_pop):
    """
    Finds the closest recording log in the prevalence DataFrame based on the target date.

    Parameters:
    prevalence (DataFrame): The DataFrame containing the records.
    target_date (datetime): The date to find the closest log for.
    log_value (any): The value to validate against the closest log.
    column_name (str): The name of the column to compare.
    multiply_by_pop (bool): Whether to multiply by population size, as some regular logs have numbers not prevalences

    Returns:
    None
    """
    closest_recording_log = None
    smallest_diff = float('inf')  # Initialize with a large number

    for i in range(len(prevalence)):
        function_date = prevalence['date'][i]
        date_diff = abs((target_date - function_date).days)

        if date_diff < smallest_diff:
            smallest_diff = date_diff
            if multiply_by_pop:
                closest_recording_log = prevalence[column_name][i] * prevalence['population'][i]
            else:
                closest_recording_log = prevalence[column_name][i]

    if closest_recording_log is not None:
        # Assert statement for validation
        assert log_value == closest_recording_log


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
            use_simplified_births=False,))
    sim.make_initial_population(n=popsize)
    sim.modules['HealthBurden'].parameters['logging_frequency_prevalence'] = 'day'
    sim.simulate(end_date=end_date)
    check_dtypes(sim)
    output = parse_log_file(sim.log_filepath)

    prevalence = output['tlo.methods.healthburden']['prevalence_of_diseases']

    # check to see if the monthly prevalence is calculated correctly NB for only one month

    log_prevalences_from_sim = log_prevalences_from_sim_func(sim)
    for log_date in log_prevalences_from_sim['date']:
        # Check if the current date from log_prevalencesfrom_sim exists in the prevalence DataFrame
        if log_date in prevalence['date'].values:
            # Extract the corresponding row from prevalence
            prevalence_row = prevalence.loc[prevalence['date'] == log_date].squeeze()

            # Remove the date column for comparison (if needed)
            if 'date' in prevalence.columns:
                prevalence_row = prevalence_row.drop('date')

            # Find the corresponding row in log_prevalences_from_sim
            sim_row = log_prevalences_from_sim.loc[
                log_prevalences_from_sim['date'] == log_date].squeeze()

            # Iterate over the columns to compare values
            for column in prevalence_row.index:
                # Compare the values between the two DataFrames for this date and column
                if prevalence_row[column] != sim_row[column]:
                    # Handle mismatches as needed (e.g., logging, storing in a list, etc.)
                    pass
        else:
            # Handle cases where the date is not found in prevalence DataFrame
            pass

    ## See if the registered modules are reporting prevalences as they should
    columns = prevalence.columns
    excluded_modules = ['Lifestyle', 'HealthBurden', 'HealthSeekingBehaviour', 'SymptomManager', 'Epi', 'HealthSystem', 'SimplifiedBirths', 'Contraception', 'CareOfWomenDuringPregnancy'] # don't return prevalences

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

def test_structure_logging_mockitis(tmpdir, seed):

    sim = Simulation(start_date=start_date, seed=seed, log_config={'filename': 'test_log', 'directory': outputpath})
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
                 demography.Demography(resourcefilepath=resourcefilepath),
                 mockitis.Mockitis(),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),)
    sim.make_initial_population(n=popsize)
    sim.modules['HealthBurden'].parameters['logging_frequency_prevalence'] = 'day'
    sim.modules['Mockitis'].parameters['p_cure'] = 0
    sim.modules["Mockitis"].parameters["p_infection"] = 1
    sim.simulate(end_date=end_date)
    check_dtypes(sim)
    output = parse_log_file(sim.log_filepath)

    prevalence = output['tlo.methods.healthburden']['prevalence_of_diseases']
    prevalence_mockitis_log = output['tlo.methods.mockitis']["summary"]

    max_date_in_prevalence = max(prevalence['date'])

    for j in range(len(prevalence_mockitis_log)):
         target_date = prevalence_mockitis_log['date'][j]
         regular_log_value = (
             prevalence_mockitis_log["TotalInf"][j])

         if target_date <= max_date_in_prevalence:
             find_closest_recording(prevalence, target_date, regular_log_value, 'Mockitis', True)


