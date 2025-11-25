"""
Run the HIV/TB modules with intervention coverage specified at national level
save outputs for plotting (file: output_plots_tb.py)
 """

import datetime
import pickle
import random
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (  # deviance_measure,
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    simplified_births,
    symptommanager,
    tb,
)

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = './resources'

# %% Run the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2040, 1, 1)
popsize = 10_000

# set up the log config
log_config = {
    "filename": "test_runs",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.hiv": logging.INFO,
        # "tlo.methods.tb": logging.INFO,
        "tlo.methods.demography": logging.INFO,
        # "tlo.methods.healthsystem.summary": logging.INFO,
        # "tlo.methods.healthburden": logging.INFO,
    },
}

# Register the appropriate modules
# need to call epi before tb to get bcg vax
seed = random.randint(0, 50000)
# seed = 41728  # set seed for reproducibility
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config,
                 show_progress_bar=True, resourcefilepath=resourcefilepath)
sim.register(
    demography.Demography(),
    simplified_births.SimplifiedBirths(),
    enhanced_lifestyle.Lifestyle(),
    healthsystem.HealthSystem(service_availability=["*"],  # all treatment allowed
        mode_appt_constraints=1,  # mode of constraints to do with officer numbers and time
        cons_availability="default",  # mode for consumable constraints (if ignored, all consumables available)
        ignore_priority=False,  # do not use the priority information in HSI event to schedule
        capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
        use_funded_or_actual_staffing="actual",  # actual: use numbers/distribution of staff available currently
        disable=False,  # disables the healthsystem (no constraints and no logging) and every HSI runs
        disable_and_reject_all=False,  # disable healthsystem and no HSI runs
    ),
    symptommanager.SymptomManager(),
    healthseekingbehaviour.HealthSeekingBehaviour(),
    healthburden.HealthBurden(),
    epi.Epi(),
    hiv.Hiv(run_with_checks=False),
    tb.Tb(),
    # deviance_measure.Deviance(resourcefilepath=resourcefilepath),
)


# set the scenario
sim.modules["Hiv"].parameters["select_mihpsa_scenario"] = 1
sim.modules["Hiv"].parameters["injectable_prep_allowed"] = False


# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# parse the results
output = parse_log_file(sim.log_filepath)

# save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
with open(outputpath / "default_run.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dict(output), f, pickle.HIGHEST_PROTOCOL)

# load the results
with open(outputpath / "default_run.pickle", "rb") as f:
    output = pickle.load(f)


#%% ---------------------------------------------------------------------------------------





def compare_data_vs_model(data, model_df):
    """
    Compare observed cascade data (rows=years, cols=indicators)
    with model outputs in a single multi-panel figure.
    Also plots UNAIDS series when present: unaids_propdiag, unaids_proptreat, unaids_propsuppressed.
    """
    indicators = ["propdiag", "proptreat", "propsuppressed"]

    # Align on common years (index = years)
    common_years = data.index.intersection(model_df.index)
    if common_years.empty:
        raise ValueError("No overlapping years between observed and model data.")
    data = data.loc[common_years]
    model_df = model_df.loc[common_years]
    years = common_years

    fig, axes = plt.subplots(3, 1, figsize=(7, 14), sharex=True)
    for i, (ax, ind) in enumerate(zip(axes, indicators)):
        # Required columns
        if ind not in data.columns or ind not in model_df.columns:
            print(f"Warning: {ind} missing from one of the datasets, skipping.")
            ax.set_visible(False)
            continue

        # Plot Optima vs Model
        ax.plot(years, data[ind], "o-", label="Optima", linewidth=2)
        ax.plot(years, model_df[ind], "s--", label="TLO", linewidth=2)

        # Plot UNAIDS if present for this indicator
        unaids_col = f"unaids_{ind}"
        if unaids_col in data.columns:
            ax.plot(years, data[unaids_col], "^-", label="UNAIDS", linewidth=2)

        ax.set_title(ind.replace('_', ' ').title())
        ax.set_xlabel("Year")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        # Legend only on the last (bottom) subplot
        if i == len(indicators) - 1:
            ax.legend(loc="lower right")

    axes[0].set_ylabel("Proportion")
    fig.tight_layout()
    plt.show()



# model outputs (you would replace this with your model results)
years = pd.Index(range(start_date.year, end_date.year))

model_num_diagnosed = (
    output["tlo.methods.hiv"]["long_term_mihpsa"]["Diagnosed_15_24_F"]
    + output["tlo.methods.hiv"]["long_term_mihpsa"]["Diagnosed_25_49_F"]
    + output["tlo.methods.hiv"]["long_term_mihpsa"]["Diagnosed_50_UP_F"]
    + output["tlo.methods.hiv"]["long_term_mihpsa"]["Diagnosed_15_24_M"]
    + output["tlo.methods.hiv"]["long_term_mihpsa"]["Diagnosed_25_49_M"]
    + output["tlo.methods.hiv"]["long_term_mihpsa"]["Diagnosed_50_UP_M"]
)

model_plhiv = (
    output["tlo.methods.hiv"]["long_term_mihpsa"]["PLHIV_15_24_F"]
    + output["tlo.methods.hiv"]["long_term_mihpsa"]["PLHIV_25_49_F"]
    + output["tlo.methods.hiv"]["long_term_mihpsa"]["PLHIV_50_UP_F"]
    + output["tlo.methods.hiv"]["long_term_mihpsa"]["PLHIV_15_24_M"]
    + output["tlo.methods.hiv"]["long_term_mihpsa"]["PLHIV_25_49_M"]
    + output["tlo.methods.hiv"]["long_term_mihpsa"]["PLHIV_50_UP_M"]
)

model_prop_dx = model_num_diagnosed / model_plhiv
model_prop_dx.index = years

model_num_tx =  (
    output["tlo.methods.hiv"]["long_term_mihpsa"]["ART_15_24_F"]
    + output["tlo.methods.hiv"]["long_term_mihpsa"]["ART_25_49_F"]
    + output["tlo.methods.hiv"]["long_term_mihpsa"]["ART_50_UP_F"]
    + output["tlo.methods.hiv"]["long_term_mihpsa"]["ART_15_24_M"]
    + output["tlo.methods.hiv"]["long_term_mihpsa"]["ART_25_49_M"]
    + output["tlo.methods.hiv"]["long_term_mihpsa"]["ART_50_UP_M"]
)

model_prop_treated = model_num_tx / model_num_diagnosed
model_prop_treated.index = years

model_num_suppressed = (
    output["tlo.methods.hiv"]["long_term_mihpsa"]["VLS_15_24_F"]
    + output["tlo.methods.hiv"]["long_term_mihpsa"]["VLS_25_49_F"]
    + output["tlo.methods.hiv"]["long_term_mihpsa"]["VLS_50_UP_F"]
    + output["tlo.methods.hiv"]["long_term_mihpsa"]["VLS_15_24_M"]
    + output["tlo.methods.hiv"]["long_term_mihpsa"]["VLS_25_49_M"]
    + output["tlo.methods.hiv"]["long_term_mihpsa"]["VLS_50_UP_M"]
)

model_prop_suppressed = model_num_suppressed / model_num_tx
model_prop_suppressed.index = years


model_outputs = pd.DataFrame({
    "propdiag": model_prop_dx,
    "proptreat": model_prop_treated,
    "propsuppressed": model_prop_suppressed,
})



excel_path= '/Users/tmangal/Documents/MIHPSA/Longterm control/malawi-art-2025-09-29 revised.xlsx'
optima = pd.read_excel(excel_path, sheet_name="cascade", index_col=0)
optima_years = optima.index

# generate plot
compare_data_vs_model(optima, model_outputs)


total_number_receiving_art = (model_num_tx
                              + output["tlo.methods.hiv"]["long_term_mihpsa"]["ART_00_14_M"]
                              + output["tlo.methods.hiv"]["long_term_mihpsa"]["ART_00_14_F"])




adults_num_tx = model_num_tx * output["tlo.methods.population"]["scaling_factor"]["scaling_factor"].values[0]
