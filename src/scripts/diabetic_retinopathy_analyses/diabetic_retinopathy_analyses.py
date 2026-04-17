import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    cardio_metabolic_disorders,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    diabetic_retinopathy,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    newborn_outcomes,
    oesophagealcancer,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
    tb,
)

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

# Alternative from DR calibrarion check
# root = get_root_path()
# resourcefilepath = root / "resources"

# Set parameters for the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2013, 1, 1)
popsize = 2000


def run_sim(service_availability):
    # log_config = {
    #     'filename': 'LogFile',
    #     'directory': outputpath
    # }
    sim = Simulation(start_date=start_date, seed=0,
                     log_config={"filename": "LogFile"}, resourcefilepath=resourcefilepath)

    # Register the appropriate modules
    sim.register(care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(),
                 demography.Demography(),
                 contraception.Contraception(),
                 enhanced_lifestyle.Lifestyle(),
                 healthsystem.HealthSystem(service_availability=service_availability),
                 symptommanager.SymptomManager(),
                 healthseekingbehaviour.HealthSeekingBehaviour(),
                 healthburden.HealthBurden(),
                 labour.Labour(),
                 newborn_outcomes.NewbornOutcomes(),
                 pregnancy_supervisor.PregnancySupervisor(),
                 oesophagealcancer.OesophagealCancer(),
                 postnatal_supervisor.PostnatalSupervisor(),
                 diabetic_retinopathy.DiabeticRetinopathy(),
                 hiv.Hiv(),
                 tb.Tb(),
                 epi.Epi(),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(),
                 depression.Depression(),
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    return sim.log_filepath


def get_summary_stats(logfile):
    output = parse_log_file(logfile)

    df = output['tlo.methods.diabetic_retinopathy']['summary_stats']
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    return df


def plot_prevalence(df, prefix, title):
    cols = [c for c in df.columns if c.startswith(prefix)]

    prev = df[cols].div(df[cols].sum(axis=1), axis=0)

    prev.plot(figsize=(12, 6))
    plt.title(title)
    plt.ylabel("Prevalence")
    plt.xlabel("Time")
    plt.show()


def plot_incidence(df, prefix, title):
    cols = [c for c in df.columns if c.startswith(prefix)]

    df[cols].plot(figsize=(12, 6))
    plt.title(title)
    plt.ylabel("New Cases")
    plt.xlabel("Time")
    plt.show()


try:
    logfile_with_healthsystem = run_sim(service_availability=['*'])
    results_with_healthsystem = get_summary_stats(logfile_with_healthsystem)

    logfile_no_healthsystem = run_sim(service_availability=[])
    results_no_healthsystem = get_summary_stats(logfile_no_healthsystem)

    df_with = get_summary_stats(logfile_with_healthsystem)
    df_without = get_summary_stats(logfile_no_healthsystem)

    # WITH HEALTH SYSTEM
    plot_prevalence(df_with, 'total_dr_', 'DR Prevalence (With Health System)')
    plot_incidence(df_with, 'inc_dr_', 'DR Incidence (With Health System)')

    plot_prevalence(df_with, 'total_dmo_', 'DMO Prevalence (With Health System)')
    plot_incidence(df_with, 'inc_dmo_', 'DMO Incidence (With Health System)')

    plot_prevalence(df_with, 'total_vision_', 'Vision Prevalence (With Health System)')
    plot_incidence(df_with, 'inc_vision_', 'Vision Incidence (With Health System)')

    # WITHOUT HEALTH SYSTEM
    plot_prevalence(df_without, 'total_dr_', 'DR Prevalence (No Health System)')
    plot_incidence(df_without, 'inc_dr_', 'DR Incidence (No Health System)')

    plot_prevalence(df_without, 'total_dmo_', 'DMO Prevalence (No Health System)')
    plot_incidence(df_without, 'inc_dmo_', 'DMO Incidence (No Health System)')

    plot_prevalence(df_without, 'total_vision_', 'Vision Prevalence (No Health System)')
    plot_incidence(df_without, 'inc_vision_', 'Vision Incidence (No Health System)')

except Exception as e:
    print(f"Error running simulation: {str(e)}")
    raise
