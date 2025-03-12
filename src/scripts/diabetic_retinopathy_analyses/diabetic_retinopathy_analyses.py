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

# Set parameters for the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 5000

def run_sim(service_availability):
    log_config = {
        'filename': 'LogFile',
        'directory': outputpath
    }
    sim = Simulation(start_date=start_date, seed=0, log_config={"filename": "LogFile"})

    # Register the appropriate modules
    sim.register(care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 diabetic_retinopathy.DiabeticRetinopathy(),
                 hiv.Hiv(resourcefilepath=resourcefilepath),
                 tb.Tb(resourcefilepath=resourcefilepath),
                 epi.Epi(resourcefilepath=resourcefilepath),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    return sim.log_filepath

def get_summary_stats(logfile):
    output = parse_log_file(logfile)
    counts_by_stage = output['tlo.methods.diabetic_retinopathy']['summary_stats']
    counts_by_stage['date'] = pd.to_datetime(counts_by_stage['date'])
    counts_by_stage = counts_by_stage.set_index('date', drop=True)

    # summary = {
    #     'total': counts_by_stage.filter(like='total_').sum(axis=1),
    #     'off': counts_by_stage.filter(like='off_treatment_').sum(axis=1),
    #     'tr': counts_by_stage.filter(like='treatment_').sum(axis=1),
    #     'new_early': counts_by_stage.get('new_early', pd.Series(0, index=counts_by_stage.index)),
    #     'new_late': counts_by_stage.get('new_late', pd.Series(0, index=counts_by_stage.index))
    # }

    # 2) NUMBERS UNDIAGNOSED-DIAGNOSED-TREATED-PALLIATIVE CARE OVER TIME (SUMMED ACROSS TYPES OF CANCER)
    def get_cols_excl_none(allcols, stub):
        # helper function to some columns with a certain prefix stub - excluding the 'none' columns (ie. those
        #  that do not have cancer)
        cols = allcols[allcols.str.startswith(stub)]
        cols_not_none = [s for s in cols if ("none" not in s)]
        return cols_not_none

    summary = {
        'total': counts_by_stage[get_cols_excl_none(counts_by_stage.columns, 'total_')].sum(axis=1),
        'udx': counts_by_stage[get_cols_excl_none(counts_by_stage.columns, 'undiagnosed_')].sum(axis=1),
        'dx': counts_by_stage[get_cols_excl_none(counts_by_stage.columns, 'diagnosed_')].sum(axis=1),
        'tr': counts_by_stage[get_cols_excl_none(counts_by_stage.columns, 'treatment_')].sum(axis=1)
    }

    counts_by_cascade = pd.DataFrame(summary)
    # counts_by_stage['year'] = counts_by_stage.index.year
    # incidence_by_stage = pd.DataFrame({'new_early': summary['new_early'], 'new_late': summary['new_late']})

    # Rates of diagnosis per year:
    counts_by_stage['year'] = counts_by_stage.index.year
    annual_count_of_dxtr = counts_by_stage.groupby(by='year')[['diagnosed_since_last_log',
                                                               'treated_since_last_log']].sum()

    return {
        'total_counts_by_stage_over_time': counts_by_stage,
        'counts_by_cascade': counts_by_cascade,
        'annual_count_of_dxtr': annual_count_of_dxtr
    }


def plot_dr_progression(counts_by_stage, title):
    """Plot stacked bars for early and late diabetic retinopathy over time."""

    # Set the index to be years only
    counts_by_stage = counts_by_stage.copy()
    counts_by_stage.index = counts_by_stage.index.strftime('%Y-%m')

    fig, ax = plt.subplots(figsize=(12, 6))

    counts_by_stage[['total_early', 'total_late']].plot(
        kind='bar', stacked=True, ax=ax, colormap="viridis"
    )

    ax.set_title(title)
    ax.set_xlabel("Time (Months & Years)")
    ax.set_ylabel("Number of People")
    ax.legend(["Early DR", "Late DR"])
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability

    plt.tight_layout()
    plt.show()

def plot_treatment_cascade(counts_by_cascade, title):
    """Plot stacked bar chart for treatment status over time."""

    counts_by_cascade = counts_by_cascade.copy()
    counts_by_cascade.index = counts_by_cascade.index.strftime('%Y-%m')

    fig, ax = plt.subplots(figsize=(12, 6))

    counts_by_cascade[['udx', 'dx', 'tr']].plot(
        kind='bar', stacked=True, ax=ax, colormap="coolwarm"
    )

    ax.set_title(title)
    ax.set_xlabel("Time (Months & Years)")
    ax.set_ylabel("Number of People")
    ax.legend(['Undiagnosed', 'Diagnosed', 'On Treatment'])
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def plot_dr_incidence(incidence_by_stage, title):
    """Plot stacked bars for new cases of early and late DR over time."""
    incidence_by_stage = incidence_by_stage.copy()
    incidence_by_stage.index = incidence_by_stage.index.strftime('%Y-%m')

    print(f'Incidence data for plotting {incidence_by_stage.head()}')

    fig, ax = plt.subplots(figsize=(12, 6))
    incidence_by_stage[['new_early', 'new_late']].plot(kind='bar', stacked=True, ax=ax, colormap="plasma")

    ax.set_title(title)
    ax.set_xlabel("Time (Months & Years)")
    ax.set_ylabel("Number of New Cases")
    ax.legend(["New Early DR Cases", "New Late DR Cases"])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

logfile_with_healthsystem = run_sim(service_availability=['*'])
results_with_healthsystem = get_summary_stats(logfile_with_healthsystem)

logfile_no_healthsystem = run_sim(service_availability=[])
results_no_healthsystem = get_summary_stats(logfile_no_healthsystem)

plot_dr_progression(results_with_healthsystem['total_counts_by_stage_over_time'], "DR Progression Over Time (With Health System)")
plot_dr_progression(results_no_healthsystem['total_counts_by_stage_over_time'], "DR Progression Over Time (No Health System)")
plot_treatment_cascade(results_with_healthsystem['counts_by_cascade'], "Treatment Status Over Time (With Health System)")
plot_treatment_cascade(results_no_healthsystem['counts_by_cascade'], "Treatment Status Over Time (No Health System)")
# plot_dr_incidence(results_with_healthsystem['incidence_by_stage'], "DR Incidence Over Time (With Health System)")
# plot_dr_incidence(results_no_healthsystem['incidence_by_stage'], "DR Incidence Over Time (No Health System)")
