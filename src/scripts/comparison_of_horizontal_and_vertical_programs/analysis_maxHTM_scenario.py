"""
This scenario file sets up the scenarios for simulating the effects of scaling up programs

The scenarios are:
*0 baseline mode 1
*1 scale-up HIV program
*2 scale-up TB program
*3 scale-up malaria program
*4 scale-up HIV and Tb and malaria programs

scale-up occurs on the default scale-up start date (01/01/2025: in parameters list of resourcefiles)

For all scenarios, keep all default health system settings

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/comparison_of_horizontal_and_vertical_programs/analysis_maxHTM_scenario.py

Run on the batch system using:
tlo batch-submit src/scripts/comparison_of_horizontal_and_vertical_programs/analysis_maxHTM_scenario.py

or locally using:
tlo scenario-run src/scripts/comparison_of_horizontal_and_vertical_programs/analysis_maxHTM_scenario.py

or execute a single run:
tlo scenario-run src/scripts/comparison_of_horizontal_and_vertical_programs/analysis_maxHTM_scenario.py --draw 1 0

"""

import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from tlo import Date, logging
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    malaria,
    simplified_births,
    symptommanager,
    tb,
)
from tlo.scenario import BaseScenario

from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
)

resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs")
scaleup_start_year = 2012
end_date = Date(2015, 1, 1)


class EffectOfProgrammes(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = end_date
        self.pop_size = 1_000
        self.number_of_draws = 5
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'scaleup_tests',
            'directory': Path('./outputs'),  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.hiv': logging.INFO,
                'tlo.methods.tb': logging.INFO,
                'tlo.methods.malaria': logging.INFO,
                'tlo.methods.demography': logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            tb.Tb(resourcefilepath=self.resources),
            malaria.Malaria(resourcefilepath=self.resources),
        ]

    def draw_parameters(self, draw_number, rng):

        return {
            'Hiv': {
                'type_of_scaleup': ['none', 'max', 'none', 'none', 'max'][draw_number],
                'scaleup_start_year': scaleup_start_year,
            },
            'Tb': {
                'type_of_scaleup': ['none', 'none', 'max', 'none', 'max'][draw_number],
                'scaleup_start_year': scaleup_start_year,
            },
            'Malaria': {
                'type_of_scaleup': ['none', 'none', 'none', 'max', 'max'][draw_number],
                'scaleup_start_year': scaleup_start_year,
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])



# %% Produce some figures and summary info

# # Find results_folder associated with a given batch_file (and get most recent [-1])
# results_folder = get_scenario_outputs("scaleup_tests-", outputspath)[-1]
#
# # get basic information about the results
# info = get_scenario_info(results_folder)
#
# # 1) Extract the parameters that have varied over the set of simulations
# params = extract_params(results_folder)
#
#
# # DEATHS
#
#
# def get_num_deaths_by_cause_label(_df):
#     """Return total number of Deaths by label within the TARGET_PERIOD
#     values are summed for all ages
#     df returned: rows=COD, columns=draw
#     """
#     return _df \
#         .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
#         .groupby(_df['label']) \
#         .size()
#
#
# TARGET_PERIOD = (Date(scaleup_start_year, 1, 1), end_date)
#
# # produce df of total deaths over scale-up period
# num_deaths_by_cause_label = extract_results(
#         results_folder,
#         module='tlo.methods.demography',
#         key='death',
#         custom_generate_series=get_num_deaths_by_cause_label,
#         do_scaling=True
#     )
#
#
# def summarise_deaths_for_one_cause(results_folder, label):
#     """ returns mean deaths for each year of the simulation
#     values are aggregated across the runs of each draw
#     for the specified cause
#     """
#
#     results_deaths = extract_results(
#         results_folder,
#         module="tlo.methods.demography",
#         key="death",
#         custom_generate_series=(
#             lambda df: df.assign(year=df["date"].dt.year).groupby(
#                 ["year", "label"])["person_id"].count()
#         ),
#         do_scaling=True,
#     )
#     # removes multi-index
#     results_deaths = results_deaths.reset_index()
#
#     # select only cause specified
#     tmp = results_deaths.loc[
#         (results_deaths.label == label)
#     ]
#
#     # group deaths by year
#     tmp = pd.DataFrame(tmp.groupby(["year"]).sum())
#
#     # get mean for each draw
#     mean_deaths = pd.concat({'mean': tmp.iloc[:, 1:].groupby(level=0, axis=1).mean()}, axis=1).swaplevel(axis=1)
#
#     return mean_deaths
#
#
# aids_deaths = summarise_deaths_for_one_cause(results_folder, 'AIDS')
# tb_deaths = summarise_deaths_for_one_cause(results_folder, 'TB (non-AIDS)')
# malaria_deaths = summarise_deaths_for_one_cause(results_folder, 'Malaria')
#
#
# draw_labels = ['No scale-up', 'HIV scale-up', 'TB scale-up', 'Malaria scale-up', 'HTM scale-up']
# colours = ['blue', 'green', 'red', 'purple', 'orange']
#
# # Create subplots
# fig, axs = plt.subplots(3, 1, figsize=(10, 10))
# # Plot for df1
# for i, col in enumerate(aids_deaths.columns):
#     axs[0].plot(aids_deaths.index, aids_deaths[col], label=draw_labels[i],
#                 color=colours[i])
# axs[0].set_title('HIV/AIDS')
# axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Legend to the right of the plot
# axs[0].axvline(x=scaleup_start_year, color='gray', linestyle='--')
#
# # Plot for df2
# for i, col in enumerate(tb_deaths.columns):
#     axs[1].plot(tb_deaths.index, tb_deaths[col], color=colours[i])
# axs[1].set_title('TB')
# axs[1].axvline(x=scaleup_start_year, color='gray', linestyle='--')
#
# # Plot for df3
# for i, col in enumerate(malaria_deaths.columns):
#     axs[2].plot(malaria_deaths.index, malaria_deaths[col], color=colours[i])
# axs[2].set_title('Malaria')
# axs[2].axvline(x=scaleup_start_year, color='gray', linestyle='--')
#
# for ax in axs:
#     ax.set_xlabel('Years')
#     ax.set_ylabel('Number deaths')
#
# plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend
# plt.show()
#
