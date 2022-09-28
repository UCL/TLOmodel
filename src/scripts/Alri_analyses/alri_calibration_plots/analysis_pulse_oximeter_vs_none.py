"""This will demonstrate the effect of pulse oximetry and oxygen being available versus not available, among a
population of children under 5 years old.
* All consumables available
* With forced healthcare seeking
"""

# %% Import Statements and initial declarations
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    alri,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)

resourcefilepath = Path("./resources")
outputpath = Path("./outputs")


def run_scenario(**kwargs):
    """Run the scenario and return the log file"""

    start_date = Date(2010, 1, 1)
    end_date = start_date + pd.DateOffset(years=1)
    popsize = 100_000

    log_config = {
        "filename": "alri",
        "directory": Path("./outputs"),
        "custom_levels": {
            "*": logging.WARNING,
            "tlo.methods.alri": logging.INFO,
            "tlo.methods.demography": logging.INFO,
        }
    }

    sim = Simulation(start_date=start_date, log_config=log_config, show_progress_bar=True)

    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath,
                                                      force_any_symptom_to_lead_to_healthcareseeking=True),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                  disable=True,
                                  cons_availability='all',
                                  ),
        alri.Alri(resourcefilepath=resourcefilepath),
        alri.AlriPropertiesOfOtherModules()
    )

    sim.modules['Demography'].parameters['max_age_initial'] = 5

    if kwargs['do_make_treatment_and_diagnosis_perfect']:
        alri._make_treatment_and_diagnosis_perfect(sim.modules['Alri'])

    if kwargs['pulse_oximeter_and_oxygen_is_available']:
        sim.modules['Alri'].parameters['pulse_oximeter_and_oxygen_is_available'] = 'Yes'
    else:
        sim.modules['Alri'].parameters['pulse_oximeter_and_oxygen_is_available'] = 'No'

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    return sim.log_filepath


def get_death_numbers_from_logfile(logfile):
    """Extract the number of deaths (total and deaths among those with untreated hypoxaemia) to Alri from the logfile,
    over the entire period of the simulation"""
    output = parse_log_file(logfile)
    alri_event_counts = output['tlo.methods.alri']['event_counts'].sum()

    return {
        'deaths': alri_event_counts['deaths'],
        'deaths_among_persons_with_SpO2<90%': alri_event_counts['deaths_among_persons_with_SpO2<90%']
    }


def get_cfr_from_logfile(logfile):
    """Extract the Case Fatality Ratio (Deaths:Cases) to Alri from the logfile, over the entire period of the
    simulation."""
    output = parse_log_file(logfile)
    alri_event_counts = output['tlo.methods.alri']['event_counts'].sum()

    return alri_event_counts['deaths'] / alri_event_counts['incident_cases']


# %% Run the Scenarios
scenarios = {
    # 'No_oximeter/oxygen_Perfect_treatment_effectiveness': {
    #     'pulse_oximeter_and_oxygen_is_available': False,
    #     'do_make_treatment_and_diagnosis_perfect': True,
    # },
    # 'With_oximeter/oxygen_Perfect_treatment_effectiveness': {
    #     'pulse_oximeter_and_oxygen_is_available': True,
    #     'do_make_treatment_and_diagnosis_perfect': True,
    # },
    'No_oximeter/oxygen_Default_treatment_effectiveness': {
        'pulse_oximeter_and_oxygen_is_available': False,
        'do_make_treatment_and_diagnosis_perfect': False,
    },
    'With_oximeter/oxygen_Default_treatment_effectiveness': {
        'pulse_oximeter_and_oxygen_is_available': True,
        'do_make_treatment_and_diagnosis_perfect': False,
    },
}
outputfiles = {_name: run_scenario(**_params) for _name, _params in scenarios.items()}

# %% Extract the number of deaths:
num_deaths = {_name: get_death_numbers_from_logfile(_logfile) for _name, _logfile in outputfiles.items()}
cfr = {_name: get_cfr_from_logfile(_logfile) for _name, _logfile in outputfiles.items()}

# %% Plot results

df_num_deaths = pd.DataFrame(num_deaths)
df_num_deaths.loc['deaths_not_among_persons_with_SpO2<90%'] = \
    df_num_deaths.loc['deaths'] - df_num_deaths.loc['deaths_among_persons_with_SpO2<90%']

fig, ax = plt.subplots()
df_num_deaths.loc[
    ['deaths_among_persons_with_SpO2<90%', 'deaths_not_among_persons_with_SpO2<90%']
].T.plot.barh(ax=ax, stacked=True)
fig.tight_layout()
fig.show()

fig, ax = plt.subplots()
(100_000 * pd.Series(cfr)).T.plot.barh(ax=ax, stacked=True)
ax.set_title('Case:Fatality Ratio')
ax.set_xlabel('Deaths per 100k cases')
fig.tight_layout()
fig.show()
