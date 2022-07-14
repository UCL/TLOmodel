# %% Import Statements
import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file, unflatten_flattened_multi_index_in_logging
from tlo.methods import demography, enhanced_lifestyle, simplified_births


class LifeStylePlots:
    """ a class for displaying all Life style plots by both gender and age range """

    def __init__(self, sim_obj, logs):
        # create a list of all dict_keys as defined in models dictionary of LifestyleModels class. Note that we are
        # excluding keys to circumcision and sex workers linear models as these keys are logged differently
        self.en_props: List[str] = [_key for _key in
                                    enhanced_lifestyle.LifestyleModels(sim_obj.modules['Lifestyle']).get_lm_keys()
                                    if _key not in ['li_is_circ', 'li_is_sexworker']]

        # date-stamp to label log files and any other outputs
        self.datestamp: str = datetime.date.today().strftime("__%Y_%m_%d")

        # create a dictionary for all key descriptions
        self.key_des: Dict[str, str] = {'M': 'Males', 'F': 'Females'}

        # map lifestyle properties and their description
        self._prop_des: Dict[str, str] = {'li_ed_lev': 'education level', 'li_mar_stat': 'marital status'}

        # get all logs
        self.all_logs = logs

        # un flatten some properties
        self.dfs = self.construct_dfs(self.all_logs)

        self.outputpath = Path("./outputs")  # folder for convenience of storing outputs

    def construct_dfs(self, lifestyle_log) -> dict:
        """ Create dict of pd.DataFrames containing counts of different lifestyle properties by date, sex and
        age-group """
        return {
            k: unflatten_flattened_multi_index_in_logging(v.set_index('date'))
            for k, v in lifestyle_log.items() if k in self.en_props
        }

    def display_bmi_plots_by_gender(self, _property: str = None):
        """ a function to display bmi categories plots by gender. we are considering bmi categories from 0 - 5 and are
        looking at Male and Female population

        bmi categories:
            category    kg/m3       percentage
            _____________________________________
                1:      (<18)       '80-100%'
                2:      (18-24.9)   '60-79%'
                3:      (25-29.9)   '40-59%'
                4:      (30-34.9)   '20-39%'
                5:      (35+)       '0-19%'

        :param _property: bmi property defined in enhanced lifestyle module """
        get_dates = np.asarray(pd.to_datetime(self.dfs['li_bmi'].index).year)  # get dates from logfile data
        counter: int = 0  # a counter for plot positioning

        for key, desc in self.key_des.items():
            tot_bmi = self.dfs[_property][key].sum(axis=1)
            # get bmi categories per each gender
            cat_0_prp = self.dfs[_property][key]["0"].sum(axis=1) / tot_bmi
            cat_1_prp = self.dfs[_property][key]["1"].sum(axis=1) / tot_bmi
            cat_2_prp = self.dfs[_property][key]["2"].sum(axis=1) / tot_bmi
            cat_3_prp = self.dfs[_property][key]["3"].sum(axis=1) / tot_bmi
            cat_4_prp = self.dfs[_property][key]["4"].sum(axis=1) / tot_bmi
            cat_5_prp = self.dfs[_property][key]["5"].sum(axis=1) / tot_bmi

            # add bmi data to plot
            plt.subplot(131 + counter)
            plt.bar(get_dates, cat_0_prp, color='r')
            plt.bar(get_dates, cat_1_prp, bottom=cat_0_prp, color='b')
            plt.bar(get_dates, cat_2_prp, bottom=cat_0_prp + cat_1_prp, color='y')
            plt.bar(get_dates, cat_3_prp, bottom=cat_0_prp + cat_1_prp + cat_2_prp, color='g')
            plt.bar(get_dates, cat_4_prp, bottom=cat_0_prp + cat_1_prp + cat_2_prp + cat_3_prp, color='c')
            plt.bar(get_dates, cat_5_prp, bottom=cat_0_prp + cat_1_prp + cat_2_prp + cat_3_prp + cat_4_prp, color='m')
            plt.title(f"{desc} bmi categories")
            plt.ylabel("bmi proportions")
            plt.xlabel("Year")
            plt.ylim(0, 1.1)
            plt.legend(['cat_0', 'cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5'])

            # incrementing counter
            counter += 2
        # save and display bmi categories by gender plot  132
        plt.savefig(self.outputpath / (_property + self.datestamp + '.png'), format='png')
        plt.show()

    def display_wealth_plots_by_gender(self, _property: str = None):
        """ a function to display wealth categories plots by gender. we are considering wealth categories of 1 - 5 and are
        looking at Male and Female population

        :param _property: wealth property defined in enhanced lifestyle module """
        get_dates = np.asarray(pd.to_datetime(self.dfs['li_wealth'].index).year)  # get dates from logfile data
        counter: int = 0  # a counter for plot positioning

        for key, desc in self.key_des.items():
            tot_bmi = self.dfs[_property][key].sum(axis=1)
            # get bmi categories per each gender
            cat_1_prp = self.dfs[_property][key]["1"].sum(axis=1) / tot_bmi
            cat_2_prp = self.dfs[_property][key]["2"].sum(axis=1) / tot_bmi
            cat_3_prp = self.dfs[_property][key]["3"].sum(axis=1) / tot_bmi
            cat_4_prp = self.dfs[_property][key]["4"].sum(axis=1) / tot_bmi
            cat_5_prp = self.dfs[_property][key]["5"].sum(axis=1) / tot_bmi

            # add bmi data to plot
            plt.subplot(131 + counter)
            plt.bar(get_dates, cat_1_prp, color='b')
            plt.bar(get_dates, cat_2_prp, bottom=cat_1_prp, color='y')
            plt.bar(get_dates, cat_3_prp, bottom=cat_1_prp + cat_2_prp, color='g')
            plt.bar(get_dates, cat_4_prp, bottom=cat_1_prp + cat_2_prp + cat_3_prp, color='c')
            plt.bar(get_dates, cat_5_prp, bottom=cat_1_prp + cat_2_prp + cat_3_prp + cat_4_prp, color='m')
            plt.title(f"{desc} wealth categories")
            plt.ylabel("wealth proportions")
            plt.xlabel("Year")
            plt.ylim(0, 1.1)
            plt.legend(['cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5'])

            # incrementing counter
            counter += 2
        # save and display bmi categories by gender plot
        plt.savefig(self.outputpath / (_property + self.datestamp + '.png'), format='png')
        plt.show()

    def display_categorical_values_plots_by_gender(self, _property: str = None):
        """ a function to display categorical values plots by gender. Here, we are looking at two enhanced lifestyle
        properties, marital status and education level

        For marital status categories:
            1:  never married
            2:  currently married
            3:  past (widowed or divorced)

        For education levels:
            1:  not in education
            2:  primary education
            3:  secondary education

        :param _property: wealth property defined in enhanced lifestyle module """
        # todo: use this function for all properties with categorical values
        get_dates = np.asarray(pd.to_datetime(self.dfs[_property].index).year)  # get dates from logfile data
        counter: int = 0  # a counter for plot positioning

        for key, desc in self.key_des.items():
            tot_bmi = self.dfs[_property][key].sum(axis=1)
            # get property categories per each gender
            cat_1_prp = self.dfs[_property][key]["1"].sum(axis=1) / tot_bmi
            cat_2_prp = self.dfs[_property][key]["2"].sum(axis=1) / tot_bmi
            cat_3_prp = self.dfs[_property][key]["3"].sum(axis=1) / tot_bmi

            # add property data to plot
            plt.subplot(131 + counter)
            plt.bar(get_dates, cat_1_prp, color='b')
            plt.bar(get_dates, cat_2_prp, bottom=cat_1_prp, color='y')
            plt.bar(get_dates, cat_3_prp, bottom=cat_1_prp + cat_2_prp, color='g')
            plt.title(f"{desc} {self._prop_des[_property]} categories")
            plt.ylabel("proportions")
            plt.xlabel("Year")
            plt.ylim(0, 1.1)
            plt.legend(['cat_1', 'cat_2', 'cat_3'])

            # incrementing counter
            counter += 2
        # save and display bmi categories by gender plot
        plt.savefig(self.outputpath / (_property + self.datestamp + '.png'), format='png')
        plt.show()

    def circumcised_men_plot(self, _property: str = None):
        """ a function to get proportions of men circumcised in enhanced lifestyle logs and do some plotting

        :param _property: circumcision property defined in enhanced lifestyle module """
        # Examine Proportion Men Circumcised:
        circ = extract_formatted_series(self.all_logs[_property])
        circ.plot()
        plt.title('Proportion of Adult Men Circumcised')
        plt.ylim(0, 0.30)
        plt.show()

    def sex_workers_plot(self, _property: str = None):
        """ a function to get proportions of women sex workers in enhanced lifestyle logs and do some plotting

        :param _property: sex workers property defined in enhanced lifestyle module """
        # Examine Proportion Women sex Worker:
        fsw = extract_formatted_series(self.all_logs[_property])
        fsw.plot()
        plt.title('Proportion of 15-49 Women Sex Workers')
        plt.ylim(0, 0.01)
        plt.show()

    def display_plots_by_gender(self):
        """ a function that will display plots of all enhanced lifestyle properties grouped by individual gender """
        # plot graphs of properties defined in enhanced lifestyle module
        for _property in self.en_props:
            if _property in ['li_wealth', 'li_mar_stat', 'li_bmi', 'li_ed_lev']:
                if _property == 'li_bmi':
                    self.display_bmi_plots_by_gender(_property)
                elif _property == 'li_wealth':
                    self.display_wealth_plots_by_gender(_property)
                else:
                    self.display_categorical_values_plots_by_gender(_property)
            else:
                # sum male and female tobacco use
                get_totals = self.dfs[_property].sum(axis=1)

                # get male proportions
                sum_males_df = self.dfs[_property].M.sum(axis=1)
                males_prop = sum_males_df / get_totals

                # get female proportions
                sum_females_df = self.dfs[_property].F.sum(axis=1)
                females_prop = sum_females_df / get_totals

                # Load Model Results
                get_years = pd.to_datetime(self.dfs[_property].index).year

                fig, ax = plt.subplots()
                ax.plot(np.asarray(get_years), males_prop)
                ax.plot(np.asarray(get_years), females_prop)

                plt.title(f"Plotting of property {_property}")
                plt.xlabel("Year")
                plt.ylabel("Male and Females proportions per property")
                plt.ylim(0, 1.0)
                plt.legend(['Male proportions', 'Female Proportions'])
                plt.savefig(self.outputpath / (_property + self.datestamp + '.png'), format='png')
                plt.show()

        # plot male circumcision graph
        self.circumcised_men_plot('li_is_circ')

        # plot women sex workers graph
        self.sex_workers_plot('li_is_sexworker')


def extract_formatted_series(df):
    return pd.Series(index=pd.to_datetime(df['date']), data=df.iloc[:, 1].values)


def run():
    # To reproduce the results, you need to set the seed for the Simulation instance. The Simulation
    # will seed the random number generators for each module when they are registered.
    # If a seed argument is not given, one is generated. It is output in the log and can be
    # used to reproduce results of a run
    seed = 1

    # By default, all output is recorded at the "INFO" level (and up) to standard out. You can
    # configure the behaviour by passing options to the `log_config` argument of
    # Simulation.
    log_config = {
        "filename": "enhanced_lifestyle",  # The prefix for the output file. A timestamp will be added to this.
        "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
            "tlo.methods.demography": logging.WARNING,
            "tlo.methods.enhanced_lifestyle": logging.INFO
        }
    }
    # For default configuration, uncomment the next line
    # log_config = dict()

    # Basic arguments required for the simulation
    start_date = Date(2010, 1, 1)
    end_date = Date(2070, 1, 1)
    pop_size = 20000

    # This creates the Simulation instance for this run. Because we"ve passed the `seed` and
    # `log_config` arguments, these will override the default behaviour.
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    # Path to the resource files used by the disease and intervention methods
    resources = "./resources"

    # We register all modules in a single call to the register method, calling once with multiple
    # objects. This is preferred to registering each module in multiple calls because we will be
    # able to handle dependencies if modules are registered together
    sim.register(
        demography.Demography(resourcefilepath=resources),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resources),
        simplified_births.SimplifiedBirths(resourcefilepath=resources)

    )

    sim.make_initial_population(n=pop_size)
    sim.simulate(end_date=end_date)
    return sim


# %% Run the Simulation
sim = run()

# %% read the results
output = parse_log_file(sim.log_filepath)

# construct a dict of dataframes using lifestyle logs
logs_df = output['tlo.methods.enhanced_lifestyle']


# display plots by gender
g_plots = LifeStylePlots(sim, logs_df)
g_plots.display_plots_by_gender()

