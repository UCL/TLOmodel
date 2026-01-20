"""
Runs the diabetic retinopathy module and produces the standard compare_number_of_deaths analysis to check
the number of deaths modelled against the GBD data. It also produces plots for diabetes DALYs with and
without the healthsytem
"""

import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    compare_number_of_deaths,
    get_root_path,
    make_age_grp_types,
    parse_log_file,
)
from tlo.methods import (
    cardio_metabolic_disorders,
    demography,
    depression,
    diabetic_retinopathy,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)

# The resource files
root = get_root_path()
resourcefilepath = root / "resources"

log_config = {
    "filename": "diabetic_retinopathy_calibration_check",
    "directory": root / "outputs",
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.demography": logging.INFO,
        "tlo.methods.healthburden": logging.INFO,
    }
}

# Set parameters for the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 5_000


def get_diabetes_dalys(logfile):
    output = parse_log_file(logfile)
    dalys = output['tlo.methods.healthburden']['dalys_stacked']
    # Keep only numeric DALY columns + age_range
    numeric_cols = dalys.select_dtypes(include='number').columns
    dalys = dalys[['age_range', *numeric_cols]]

    # Sum over time
    dalys = (
        dalys
        .groupby('age_range')
        .sum()
        .reindex(make_age_grp_types().categories)
        .fillna(0.0)
    )

    return dalys


def run_sim(allow_hsi: bool):
    sim = Simulation(start_date=start_date, log_config=log_config, resourcefilepath=resourcefilepath)

    sim.register(
        demography.Demography(),
        simplified_births.SimplifiedBirths(),
        enhanced_lifestyle.Lifestyle(),
        healthsystem.HealthSystem(
            disable=(allow_hsi is True),
            disable_and_reject_all=(allow_hsi is False)
        ),
        symptommanager.SymptomManager(),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        healthburden.HealthBurden(),
        cardio_metabolic_disorders.CardioMetabolicDisorders(),
        depression.Depression(),
        diabetic_retinopathy.DiabeticRetinopathy(),
    )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    return sim.log_filepath


# With interventions
logfile_with_healthsystem = run_sim(allow_hsi=True)
dalys_with_hs = get_diabetes_dalys(logfile_with_healthsystem)

# Without interventions
logfile_no_healthsystem = run_sim(allow_hsi=False)
dalys_no_hs = get_diabetes_dalys(logfile_no_healthsystem)

CAUSE_NAME = 'Diabetes'

# Extract Diabetes only
diabetes_dalys_with_hs = dalys_with_hs[CAUSE_NAME].fillna(0.0)
diabetes_dalys_no_hs = dalys_no_hs[CAUSE_NAME].fillna(0.0)

# With healthsystem
comparison = compare_number_of_deaths(
    logfile=logfile_with_healthsystem,
    resourcefilepath=resourcefilepath
).rename(columns={'model': 'model_with_healthsystem'})

# Without healthsystem
x = compare_number_of_deaths(
    logfile=logfile_no_healthsystem,
    resourcefilepath=resourcefilepath
)['model']
x.name = 'model_no_healthsystem'

comparison = pd.concat([comparison, x], axis=1)

comparison = comparison.loc[
    ("2010-2014", slice(None), slice(None), CAUSE_NAME)
].fillna(0.0)

comparison.index = comparison.index.droplevel(
    [name for name in comparison.index.names if name in ('period', 'cause')]
)

print("####################################### Result ###################")
print((comparison["model_with_healthsystem"] - comparison["model_no_healthsystem"]).abs().sum())

fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(8, 6))

for ax, sex in zip(axs, ("M", "F")):
    comparison.loc[sex].plot(ax=ax)
    ax.set_ylabel("Deaths per year")
    ax.set_title(f"Sex: {sex}")

axs[-1].set_xlabel("Age group")

plt.tight_layout()
plt.show()

# Plot for DALYs
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

diabetes_dalys_with_hs.plot.bar(ax=axs[0])
axs[0].set_title("Diabetes DALYs – With Health System")
axs[0].set_xlabel("Age group")
axs[0].set_ylabel("Total DALYs")

diabetes_dalys_no_hs.plot.bar(ax=axs[1])
axs[1].set_title("Diabetes DALYs – No Health System")
axs[1].set_xlabel("Age group")

plt.tight_layout()
plt.show()
