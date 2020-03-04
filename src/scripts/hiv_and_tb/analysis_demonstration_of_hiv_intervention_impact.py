from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    hiv,
    malecircumcision,
    symptommanager,
    tb,
)

start_date = Date(2010, 1, 1)
end_date = Date(2012, 12, 31)
popsize = 100


# %% Define some helper functions to run and analyse the model

def run_simulation_with_set_intv_parameters(
    params
):
    """
    This helper function will run a simulation with a given set of paramerers for the HIV interventions,
    over-writing the parameters that are normally imported. It returns the path of the logfile for the simulation
    that is created.

    The input is a dict containing values to overwrite the following parameters in the model
    "fsw_prep": Parameter(Types.REAL, "prob of fsw receiving PrEP"): 0.1
    "initial_art_coverage": Parameter(Types.REAL, "coverage of ART at baseline"): <<table of values>>
    "treatment_prob": Parameter(Types.REAL, "probability of requesting ART following positive HIV test"): 0.3
    "hv_behav_mod": Parameter(Types.REAL, "change in force of infection with behaviour modification"): 0.5
    "testing_adj": Parameter(Types.REAL, "additional HIV testing outside generic appts"): 0.05
    """

    # Define paths
    outputpath = Path("./outputs")
    resourcefilepath = Path("./resources")

    # Create simulation and register the appropriate modules
    sim = Simulation(start_date=start_date)
    sim.seed_rngs(0)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            disable=True,
        )
    )
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(hiv.Hiv(resourcefilepath=resourcefilepath))
    sim.register(tb.Tb(resourcefilepath=resourcefilepath))
    sim.register(malecircumcision.MaleCircumcision(resourcefilepath=resourcefilepath))

    # Overwrite the parameters in the modules if they are provided
    if 'fsw_prep' in params:
        sim.modules['Hiv'].parameters['fsw_prep'] = params['fsw_prep']

    if 'initial_art_coverage' in params:
        sim.modules['Hiv'].parameters['initial_art_coverage']['prop_coverage'] = params['initial_art_coverage']

    if 'treatment_prob' in params:
        sim.modules['Hiv'].parameters['treatment_prob'] = params['treatment_prob']

    if 'hv_behav_mod' in params:
        sim.modules['Hiv'].parameters['hv_behav_mod'] = params['hv_behav_mod']

    if 'testing_adj' in params:
        sim.modules['Hiv'].parameters['testing_adj'] = params['testing_adj']

    # Sets all modules to WARNING threshold, then alters hiv and tb to INFO
    custom_levels = {
        "*": logging.WARNING,
        "tlo.methods.hiv": logging.INFO,
        "tlo.methods.tb": logging.INFO,
        "tlo.method.malecircumcision": logging.INFO,
        "tlo.methods.demography": logging.INFO,
    }
    # configure_logging automatically appends datetime
    logfile = sim.configure_logging(filename="LogFile", custom_levels=custom_levels)

    # Run the simulation and flush the logger
    sim.seed_rngs(0)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    return logfile


def get_key_outputs(logfile):
    """
    This helper function accepts the path of a logfile and computes the following key metrics for each year:
    * Number of new HIV infections
    * Numbers of AIDS deaths
    * Coverage of ART among PLHIV
    * Coverage of ever been diagnosed with HIV among PLHIV
    * Coverage of PrEP among FSW
    """
    output = parse_log_file(logfile)
    r = dict()  # processed results

    def make_year_the_index(df):
        df.set_index(pd.to_datetime(df['date']).dt.year, drop=True, inplace=True)
        df.drop(columns='date', inplace=True)

    def sum_15plus(df):
        x = df.copy()
        x = x.transpose()
        x['adult'] = [int(x.split('-')[0]) >= 15 for x in list(df.columns[df.columns != '100+'])] + [True]
        return x.groupby(by='adult').sum().transpose()[True]

    make_year_the_index(output['tlo.methods.hiv']['hiv_treatment'])
    make_year_the_index(output['tlo.methods.hiv']['hiv_epidemiology'])
    make_year_the_index(output['tlo.methods.hiv']['hiv_intvs'])
    make_year_the_index(output['tlo.methods.hiv']['plhiv_m'])
    make_year_the_index(output['tlo.methods.hiv']['plhiv_f'])

    # New infections
    r['num_new_infections_15_to_49'] = output['tlo.methods.hiv']['hiv_epidemiology']['num_new_infections_15_to_49']
    r['num_new_infections_0_to_14'] = output['tlo.methods.hiv']['hiv_epidemiology']['num_new_infections_0_to_14']

    # Adult ART coverage
    adult_art = output['tlo.methods.hiv']['hiv_treatment']['on_art_15plus'].apply(pd.Series)
    adult_art_any = adult_art[[1, 2]].sum(axis=1)  # TODO: check this is valid with diff cats for adherance included
    adult_plhiv = sum_15plus(output['tlo.methods.hiv']['plhiv_m']) \
                  + sum_15plus(output['tlo.methods.hiv']['plhiv_f'])
    r['fraction_of_adult_plhiv_on_art'] = adult_art_any / adult_plhiv

    # Proportion of HIV diagnosed
    r['prop_15plus_diagnosed'] = output['tlo.methods.hiv']['hiv_intvs']['prop_15plus_diagnosed']

    # Proportion exposed to behaviour change
    r['prop_exposed_to_behaviour_change_15plus'] = output['tlo.methods.hiv']['hiv_intvs'][
        'prop_exposed_to_behaviour_change_15plus']

    # Proportion exposed to behaviour change
    r['prop_fsw_on_prep'] = output['tlo.methods.hiv']['hiv_intvs']['prop_fsw_on_prep']

    return r


def batch_run(scenarios):
    """
    :param scenarios: a dict of the form: {name_of_scenario: dict_of_params}
    :return: dict of the form: {key_output: dataframe with rows of simulation year and columsn of name_of_scenario}
    """
    results = dict()
    for name, params in scenarios.items():
        results[name] = get_key_outputs(
            run_simulation_with_set_intv_parameters(
                params
            )
        )

    # turn the outputs into dataframes with a column for each scenario
    outputs = list(list(results.values())[0].keys())
    scenario_names = list(results.keys())

    compiled_results = dict()
    for output in outputs:
        compiled_results[output] = pd.DataFrame()
        for scenario in scenario_names:
            compiled_results[output][scenario] = results[scenario][output]

    return compiled_results


# %% Examine the impact of PrEP
# Scenarios for different levels of PrEP for FSW
scenarios_prep = dict()
scenarios_prep['no_prep'] = {
    'fsw_prep': 0.0,
}
scenarios_prep['fsw_prep_10pc'] = {
    'fsw_prep': 0.1,
}
scenarios_prep['fsw_prep_100pc'] = {
    'fsw_prep': 1.0,
}

batch_run_prep = batch_run(scenarios_prep)
batch_run_prep['prop_fsw_on_prep'].plot()
plt.title('prop_fsw_on_prep')
plt.show()


# %% Scenarios for different amount of ART
scenarios_art = dict()
scenarios_art['no_art'] = {
    'initial_art_coverage': 0.0,
    'treatment_prob': 0.0,
}
scenarios_art['initial_art_no_scale_up'] = {
    'treatment_prob': 0.0,
}
scenarios_art['initial_art_and_scale_up'] = {
}

batch_run_art = batch_run(scenarios_art)
batch_run_art['fraction_of_adult_plhiv_on_art'].plot()
plt.title('fraction_of_adult_plhiv_on_art')
plt.show()

batch_run_art['fraction_of_adult_plhiv_on_art'].plot()
plt.title('fraction_of_adult_plhiv_on_art')
plt.show()

# TODO: add in deaths

# %% Scenarios for different amount of Testing
scenarios_testing = dict()
scenarios_testing['no_adj_testing'] = {
    'testing_adj': 0
}
scenarios_testing['adj_testing_5pc'] = {
    'testing_adj': 0.05
}
scenarios_testing['adj_testing_30pc'] = {
    'testing_adj': 0.30
}

batch_run_testing = batch_run(scenarios_testing)
batch_run_testing['prop_15plus_diagnosed'].plot()
plt.title('prop_15plus_diagnosed')
plt.show()


# %% Scenarios for different amount of Behaviour Change
scenarios_behav_mod = dict()
scenarios_behav_mod['no_behav_chg'] = {
    'hv_behav_mod': 0
}
scenarios_behav_mod['adj_testing_50pc'] = {
    'hv_behav_mod': 0.5
}
scenarios_behav_mod['adj_testing_100pc'] = {
    'hv_behav_mod': 1.00
}

batch_run_behav_mod = batch_run(scenarios_behav_mod)
batch_run_behav_mod['num_new_infections_15_to_49'].plot()
plt.title('num_new_infections_15_to_49')
plt.show()
