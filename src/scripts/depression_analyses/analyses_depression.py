import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    chronicsyndrome,
    contraception,
    demography,
    depression,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    mockitis,
    symptommanager,
)
from tlo.methods.depression import compute_key_outputs_for_last_3_years

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
popsize = 20000


# Establish the simulation object
def run_simulation_with_set_service_coverage_parameter(service_availability, healthsystemdisable):
    """
    This helper function will run a simulation with a given service coverage parameter and return the path of
    the logfile.
    :param service_availability: list indicating which serivces to include (see HealthSystem)
    :param healthsystemdisable: bool to indicate whether or not to disable healthsystem (see HealthSystem)
    :return: logfile name
    """

    sim = Simulation(start_date=start_date)
    sim.seed_rngs(0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
    sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(
        resourcefilepath=resourcefilepath,
        service_availability=service_availability,
        disable=healthsystemdisable
    ))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(depression.Depression(resourcefilepath=resourcefilepath))

    # Establish the logger
    logfile = sim.configure_logging(filename="LogFile", custom_levels={"*": logging.INFO})

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    return logfile


# %%  Run model with all interventions working to check that outputs of depression match thge calibration points

results_health_system_disabled = compute_key_outputs_for_last_3_years(
    parse_log_file(
        run_simulation_with_set_service_coverage_parameter(
            service_availability=['*'],
            healthsystemdisable=True
        )
    )
)

# Add in comparator Data:
calibration_df = pd.DataFrame(data={'Model': results_health_system_disabled})
calibration_df['Data'] = pd.Series(data=np.nan).astype(object)
calibration_df.loc['Current prevalence of depression, aged 15+', 'Data'] = 0.09
calibration_df.loc['Current prevalence of depression, aged 15+ males', 'Data'] = 0.06
calibration_df.at['Current prevalence of depression, aged 15+ females', 'Data'] = [0.10, 0.08]
calibration_df.at['Rate of suicide incidence per 100k persons aged 15+', 'Data'] = [26.1, 8.0, 3.7]

# %% Run a comparison model with the interventions (with all interventions turned off)
results_no_intvs = compute_key_outputs_for_last_3_years(
    parse_log_file(
        run_simulation_with_set_service_coverage_parameter(
            service_availability=[],
            healthsystemdisable=False
        )
    )
)

# Make a table to compare the effects of having vs not having any interventions
effect_of_intvs_df = pd.DataFrame(data={'Intvs_On': results_health_system_disabled, 'Intvs_Off': results_no_intvs})


# %% Run a comparison in which the effectiveness of interventions for depression at turned up to implausible levels
#       in order to check that the effect of the interventions is working. Also with mockitis
#       and chronicsyndrome so allowing further opportunities for diagnosing depression

def run_simulation_with_set_intvs_maximised():
    """
    This helper function will run a simulation with a given service coverage parameter and return the path of
    the logfile.
    :param service_availability: list indicating which serivces to include (see HealthSystem)
    :param healthsystemdisable: bool to indicate whether or not to disable healthsystem (see HealthSystem)
    :return: logfile name
    """

    sim = Simulation(start_date=start_date)
    sim.seed_rngs(0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
    sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(
        resourcefilepath=resourcefilepath,
        service_availability=['*'],
        disable=True
    ))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(depression.Depression(resourcefilepath=resourcefilepath))
    sim.register(mockitis.Mockitis())
    sim.register(chronicsyndrome.ChronicSyndrome())
    sim.register(dx_algorithm_child.DxAlgorithmChild(resourcefilepat=resourcefilepath))

    sim.modules['Depression'].parameters['rr_depr_on_antidepr'] = 50
    sim.modules['Depression'].parameters['rr_resol_depr_on_antidepr'] = 50
    sim.modules['Depression'].parameters['rr_resol_depr_current_talk_ther'] = 50
    sim.modules['Depression'].parameters['sensitivity_of_assessment_of_depression'] = 1.0
    sim.modules['Depression'].parameters['pr_assessed_for_depression_in_generic_appt_level1'] = 1.0

    # Establish the logger
    logfile = sim.configure_logging(filename="LogFile", custom_levels={"*": logging.INFO})

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    return logfile


results_max_intvs = compute_key_outputs_for_last_3_years(
    parse_log_file(
        run_simulation_with_set_intvs_maximised()
    )
)

effect_of_intvs_df['Intvs_Max'] = pd.Series(results_max_intvs)
