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
end_date = Date(2020, 1, 1)
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
    # Create a date column for filtering
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


# def plot_incidence(df, prefix, title):
#     cols = [c for c in df.columns if c.startswith(prefix)]
#
#     df[cols].plot(figsize=(12, 6))
#     plt.title(title)
#     plt.ylabel("New Cases")
#     plt.xlabel("Time")
#     plt.show()


def plot_incidence(df, prefix, title):
    cols = [c for c in df.columns if c.startswith(prefix)]

    fig, ax = plt.subplots(figsize=(12, 6))
    df[cols].plot(ax=ax)

    # Force integer y-axis ticks
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title(title)
    plt.ylabel("Number of New Cases")
    plt.xlabel("Time")
    plt.show()


def plot_dr_status(df, title_prefix):
    """Plot DR status both as counts and as prevalence among diabetics"""

    # Plot 1: Absolute counts
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Absolute counts
    dr_cols = ['total_dr_mild_or_moderate', 'total_dr_severe', 'total_dr_proliferative']
    # for col in dr_cols:
    #     label = col.replace('total_dr_', '').replace('_', ' ').title()
    #     axes[0].plot(df.index, df[col], label=label, linewidth=2)
    # axes[0].set_title(f'{title_prefix} - Absolute Counts')
    # axes[0].set_xlabel('Date')
    # axes[0].set_ylabel('Number of People')
    # axes[0].legend()
    # axes[0].grid(True, alpha=0.3)

    # Prevalence among diabetics
    total_diabetics = df[dr_cols].sum(axis=1) + df['total_dr_none']
    for col in dr_cols:
        label = col.replace('total_dr_', '').replace('_', ' ').title()
        prevalence = (df[col] / total_diabetics * 100).fillna(0)
        axes[1].plot(df.index, prevalence, label=label, linewidth=2)
    axes[1].set_title(f'{title_prefix} - Prevalence Among Diabetics')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Prevalence Among Diabetics (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_dmo_status(df, title_prefix):
    """Plot DMO status among people with DR"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    dmo_cols = ['total_dmo_clinically_significant', 'total_dmo_non_clinically_significant']
    total_with_dr = df[['total_dr_mild_or_moderate', 'total_dr_severe', 'total_dr_proliferative']].sum(axis=1)

    # Absolute counts
    # for col in dmo_cols:
    #     label = col.replace('total_dmo_', '').replace('_', ' ').title()
    #     axes[0].plot(df.index, df[col], label=label, linewidth=2)
    # axes[0].set_title(f'{title_prefix} - Absolute Counts')
    # axes[0].set_xlabel('Date')
    # axes[0].set_ylabel('Number of People')
    # axes[0].legend()
    # axes[0].grid(True, alpha=0.3)

    # Prevalence among those with DR
    for col in dmo_cols:
        label = col.replace('total_dmo_', '').replace('_', ' ').title()
        prevalence = (df[col] / total_with_dr * 100).fillna(0)
        axes[1].plot(df.index, prevalence, label=label, linewidth=2)
    axes[1].set_title(f'{title_prefix} - Prevalence Among People with DR')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Prevalence Among DR Patients (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_dmo_by_dr_stage(df_with):
    """Plot DMO prevalence stratified by DR stage"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    stages = ['mild_or_moderate', 'severe', 'proliferative']
    titles = ['Mild/Moderate DR', 'Severe DR', 'Proliferative DR']

    for idx, (stage, title) in enumerate(zip(stages, titles)):
        # Need to track DMO by DR stage separately
        # This may require additional logging
        axes[idx].set_title(f'DMO in {title}')
        axes[idx].set_ylabel('Prevalence')

    plt.tight_layout()
    plt.show()


def plot_vision_attribution(df_with):
    """Plot what proportion of vision impairment is due to DR vs other causes"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Calculate vision loss due to DR as proportion of total vision impairment
    if 'vision_loss_due_to_dr' in df_with.columns:
        total_vision_impaired = (df_with['total_vision_moderate_vision_impairment'] +
                                 df_with['total_vision_severe_vision_impairment'] +
                                 df_with['total_vision_blindness'])

        dr_attributed = df_with['vision_loss_due_to_dr']

        axes[0].plot(df_with.index, dr_attributed / (total_vision_impaired + 0.01) * 100)
        axes[0].set_title('Vision Impairment Attributed to DR')
        axes[0].set_ylabel('Percentage')

    # Stacked area chart of vision outcomes
    vision_cols = ['total_vision_normal', 'total_vision_moderate_vision_impairment',
                   'total_vision_severe_vision_impairment', 'total_vision_blindness']

    if all(col in df_with.columns for col in vision_cols):
        axes[1].stackplot(df_with.index,
                          df_with[vision_cols[0]] / df_with[vision_cols].sum(axis=1),
                          df_with[vision_cols[1]] / df_with[vision_cols].sum(axis=1),
                          df_with[vision_cols[2]] / df_with[vision_cols].sum(axis=1),
                          df_with[vision_cols[3]] / df_with[vision_cols].sum(axis=1),
                          labels=['Normal', 'Moderate', 'Severe', 'Blindness'],
                          alpha=0.8)
        axes[1].set_title('Vision Status Distribution Over Time')
        axes[1].set_ylabel('Proportion')
        axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()


try:
    logfile_with_healthsystem = run_sim(service_availability=['*'])
    results_with_healthsystem = get_summary_stats(logfile_with_healthsystem)

    logfile_no_healthsystem = run_sim(service_availability=[])
    results_no_healthsystem = get_summary_stats(logfile_no_healthsystem)

    df_with = get_summary_stats(logfile_with_healthsystem)
    df_without = get_summary_stats(logfile_no_healthsystem)

    # WITH HEALTH SYSTEM
    # plot_prevalence(df_with, 'total_dr_', 'DR Prevalence (With Health System)')
    plot_incidence(df_with, 'inc_dr_', 'DR Incidence (With Health System)')

    # plot_prevalence(df_with, 'total_dmo_', 'DMO Prevalence (With Health System)')
    plot_incidence(df_with, 'inc_dmo_', 'DMO Incidence (With Health System)')

    plot_prevalence(df_with, 'total_vision_', 'Vision Prevalence (With Health System)')
    plot_incidence(df_with, 'inc_vision_', 'Vision Incidence (With Health System)')

    plot_dr_status(df_with, 'DR Status (With Health System)')
    plot_dmo_status(df_with, 'DMO Status (With Health System)')

    plot_dmo_by_dr_stage(df_with)
    plot_vision_attribution(df_with)

    # WITHOUT HEALTH SYSTEM
    plot_prevalence(df_without, 'total_dr_', 'DR Prevalence (No Health System)')
    plot_incidence(df_without, 'inc_dr_', 'DR Incidence (No Health System)')

    plot_prevalence(df_without, 'total_dmo_', 'DMO Prevalence (No Health System)')
    plot_incidence(df_without, 'inc_dmo_', 'DMO Incidence (No Health System)')

    plot_prevalence(df_without, 'total_vision_', 'Vision Prevalence (No Health System)')
    plot_incidence(df_without, 'inc_vision_', 'Vision Incidence (No Health System)')

    plot_dr_status(df_without, 'DR Status (Without Health System)')
    plot_dmo_status(df_without, 'DMO Status (Without Health System)')

except Exception as e:
    print(f"Error running simulation: {str(e)}")
    raise
