import datetime
from pathlib import Path

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    oesophagealcancer,
    pregnancy_supervisor,
    labour,
    healthseekingbehaviour,
    symptommanager
)

import matplotlib.pyplot as plt
import pandas as pd

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

    # Establish the simulation object and set the seed
    sim = Simulation(start_date=start_date)
    sim.seed_rngs(0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_availability),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath)
                 )

    # Manipulate parameters in order that there is a high burden of oes_cancer in order to do the checking:

    # Set initial prevalence to zero:
    sim.modules['OesophagealCancer'].parameters['init_prop_oes_cancer_stage'] = \
        [0.0] * len(sim.modules['OesophagealCancer'].parameters['init_prop_oes_cancer_stage'])

    # Rate of cancer onset per 3 months:
    sim.modules['OesophagealCancer'].parameters['r_low_grade_dysplasia_none'] = 0.05

    # Rates of cancer progression per 3 months:
    sim.modules['OesophagealCancer'].parameters['r_high_grade_dysplasia_low_grade_dysp'] *= 5
    sim.modules['OesophagealCancer'].parameters['r_stage1_high_grade_dysp'] *= 5
    sim.modules['OesophagealCancer'].parameters['r_stage2_stage1'] *= 5
    sim.modules['OesophagealCancer'].parameters['r_stage3_stage2'] *= 5
    sim.modules['OesophagealCancer'].parameters['r_stage4_stage3'] *= 5

    # Effect of treatment in reducing progression: set so that treatment prevent progression
    sim.modules['OesophagealCancer'].parameters['rr_high_grade_dysp_undergone_curative_treatment'] = 0.0
    sim.modules['OesophagealCancer'].parameters['rr_stage1_undergone_curative_treatment'] = 0.0
    sim.modules['OesophagealCancer'].parameters['rr_stage2_undergone_curative_treatment'] = 0.0
    sim.modules['OesophagealCancer'].parameters['rr_stage3_undergone_curative_treatment'] = 0.0
    sim.modules['OesophagealCancer'].parameters['rr_stage4_undergone_curative_treatment'] = 0.0

    # Establish the logger
    logfile = sim.configure_logging(filename="LogFile")

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    return logfile


def make_set_of_plots(logfile):
    output = parse_log_file(logfile)

    ## TOTAL COUNTS BY STAGE OVER TIME
    s = output['tlo.methods.oesophagealcancer']['summary_stats']
    s['date'] = pd.to_datetime(s['date'])
    s = s.set_index('date', drop=True)

    # Total prevalence over time
    s.plot(y=['total_low_grade_dysplasia', 'total_high_grade_dysplasia', 'total_stage1', 'total_stage2', 'total_stage3',
              'total_stage4'])
    plt.show()


    ## PROGRESSION THROUGH CARE CASCADE OVER TIME (SUMMED ACROSS TYPES OF CANCER)
    def get_cols_excl_none(allcols, stub):
        cols = allcols[allcols.str.startswith(stub)]
        cols_not_none = [s for s in cols if ("none" not in s)]
        return cols_not_none

    summary = {
        'total': s[get_cols_excl_none(s.columns, 'total_')].sum(axis=1),
        'udx': s[get_cols_excl_none(s.columns, 'undiagnosed_')].sum(axis=1),
        'dx': s[get_cols_excl_none(s.columns, 'diagnosed_')].sum(axis=1),
        'tr': s[get_cols_excl_none(s.columns, 'treatment_')].sum(axis=1),
        'pc': s[get_cols_excl_none(s.columns, 'palliative_')].sum(axis=1)
    }
    counts_by_cascade = pd.DataFrame(summary)

    # assert (df['total'] == (df['udx'] + df['dx'])).all()  # todo - fix

    counts_by_cascade.plot.bar(y=['udx', 'dx', 'tr', 'pc'])
    plt.show()

    # DALYS wrt age
    h = output['tlo.methods.healthburden']['DALYS']
    h['date'] = pd.to_datetime(h['date'])
    h = h.set_index('date', drop=True)

    h.groupby(by=['age_range']).sum().reset_index().plot.bar(x='age_range', y=['YLD_OesophagealCancer_0'], stacked=True)
    plt.show()

    # DEATHS wrt age and time
    d = output['tlo.methods.demography']['death']
    d['date'] = pd.to_datetime(d['date'])
    d = d.set_index('date', drop=True)

    d['age_group'] = d['age'].map(demography.Demography(resourcefilepath=resourcefilepath).AGE_RANGE_LOOKUP)
    d['year'] = d.index.year

    oes_cancer_deaths = pd.DataFrame(d.loc[d.cause == 'OesophagealCancer'].groupby(by=['age_group']).size())
    oes_cancer_deaths.plot.bar()
    plt.show()

    return {
        'total_counts_by_stage_over_time': s,
        'counts_by_cascade': counts_by_cascade,
        'oes_cancer_deaths': oes_cancer_deaths
    }


# %% Run the simulation with and without service availabilty

# With:
logfile_with_healthsystem = run_sim(service_availability=['*'])
results_with_healthsystem = make_set_of_plots(logfile_with_healthsystem)

# Without:
logfile_no_healthsystem = run_sim(service_availability=[])
results_no_healthsystem = make_set_of_plots(logfile_no_healthsystem)


# Compare Deaths
deaths = pd.concat({
    'No_HealthSystem': results_no_healthsystem['oes_cancer_deaths'][0],
    'With_HealthSystem': results_with_healthsystem['oes_cancer_deaths'][0]
    }, axis=1)

deaths.plot.bar()
plt.show()

# TODO: TIDY PLOTS
# TODO: CONFIRM RESULTS


# %% Check 5-year survival:
