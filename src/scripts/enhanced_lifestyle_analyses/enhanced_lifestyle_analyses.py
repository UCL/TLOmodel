# %% Import Statements
import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file, unflatten_flattened_multi_index_in_logging
from tlo.methods import demography, enhanced_lifestyle, simplified_births


class LifeStylePlots:
    """ a class for for plotting lifestyle properties by both gender and age groups """

    def __init__(self, logs=None):

        # create a dictionary for lifestyle property description to be used as plot descriptors. Here we are
        # excluding two properties circumcision and sex workers as these are logged differently
        self.en_props = {'li_urban': 'currently urban', 'li_wealth': 'wealth level',
                         'li_low_ex': 'currently low exercise', 'li_tob': 'current using tobacco',
                         'li_ex_alc': 'current excess alcohol', 'li_mar_stat': 'marital status',
                         'li_in_ed': 'currently in education', 'li_ed_lev': 'education level',
                         'li_unimproved_sanitation': 'uninproved sanitation',
                         'li_no_clean_drinking_water': 'no clean drinking water',
                         'li_wood_burn_stove': 'wood burn stove', 'li_no_access_handwashing': ' no access hand washing',
                         'li_high_salt': 'high salt', 'li_high_sugar': 'high sugar', 'li_bmi': 'bmi'
                         }

        # date-stamp to label log files and any other outputs
        self.datestamp: str = datetime.date.today().strftime("__%Y_%m_%d")

        # a dictionary for gender descriptions. to be used when plotting by gender
        self.gender_des: Dict[str, str] = {'M': 'Males', 'F': 'Females'}

        # get all logs
        self.all_logs = logs

        # store un flattened logs
        self.dfs = self.construct_dfs(self.all_logs)

        self.outputpath = Path("./outputs")  # folder for convenience of storing outputs

    def construct_dfs(self, lifestyle_log) -> dict:
        """ Create dict of pd.DataFrames containing counts of different lifestyle properties by date, sex and
        age-group """
        return {
            k: unflatten_flattened_multi_index_in_logging(v.set_index('date'))
            for k, v in lifestyle_log.items() if k in self.en_props.keys()
        }

    # 1. GENDER PLOTS
    # --------------------------------------------------------------------------------------------------------
    def plot_categorical_properties_by_gender(self, _property: str, categories: list):
        """ a function to plot all categorical properties of lifestyle module grouped by gender. Available
        categories per property include;

        1. bmi
            bmi is categorised as follows
                  category 1: <18
                  category 2: 18-24.9
                  category 3: 25-29.9
                  category 4: 30-34.9
                  category 5: 35+
            bmi is 0 until age 15

        2. wealth level
            wealth level is categorised as follows as follows;

                    Urban                               |         Rural
                    ------------------------------------|----------------------------------------------
                    level 1 = 75% wealth level           |  level 1 = 11% wealth level
                    level 2 = 16% wealth level          |  level 2 = 21% wealth level
                    level 3 = 5% wealth level           |  level 3 = 23% wealth level
                    level 4 = 2% wealth level           |  level 4 = 23% wealth level
                    level 5 = 2% wealth level           |  level 5 = 23% wealth level

        3. education level
             education level is categorised as follows
                    level 1: not in education
                    level 2: primary education
                    level 3 : secondary+ education )

        4. marital status
            marital status is categorised as follows
                    category 1: never married
                    category 2: married
                    category 3: widowed or divorced

        :param _property: any other categorical property defined in lifestyle module
        :param categories: a list of categories """
        get_dates = np.asarray(pd.to_datetime(self.dfs[_property].index).year)  # get dates from unflattened logfile
        counter: int = 0  # a counter for plot positioning

        # a new dataframe to contain data of property categories grouped by gender
        gc_df = pd.DataFrame()
        for gender, desc in self.gender_des.items():
            for cat in categories:
                gc_df[f'cat_{cat}'] = self.dfs[_property][gender][cat].sum(axis=1)

            # normalise the probabilities
            gc_df = gc_df.apply(lambda row: row / row.sum(), axis=1)

            # plot for bmi
            if _property == 'li_bmi':
                # get bmi categories per each gender
                cat_0 = gc_df.cat_0
                cat_1 = gc_df.cat_1
                cat_2 = gc_df.cat_2
                cat_3 = gc_df.cat_3
                cat_4 = gc_df.cat_4
                cat_5 = gc_df.cat_5

                # add bmi data to plot
                plt.subplot(131 + counter)
                plt.bar(get_dates, cat_0, color='r')
                plt.bar(get_dates, cat_1, bottom=cat_0, color='b')
                plt.bar(get_dates, cat_2, bottom=cat_0 + cat_1, color='y')
                plt.bar(get_dates, cat_3, bottom=cat_0 + cat_1 + cat_2, color='g')
                plt.bar(get_dates, cat_4, bottom=cat_0 + cat_1 + cat_2 + cat_3, color='c')
                plt.bar(get_dates, cat_5, bottom=cat_0 + cat_1 + cat_2 + cat_3 + cat_4, color='m')

            elif _property == 'li_wealth':
                # get wealth categories per each gender
                cat_1 = gc_df.cat_1
                cat_2 = gc_df.cat_2
                cat_3 = gc_df.cat_3
                cat_4 = gc_df.cat_4
                cat_5 = gc_df.cat_5

                # add wealth data to plot
                plt.subplot(131 + counter)
                plt.bar(get_dates, cat_1, color='b')
                plt.bar(get_dates, cat_2, bottom=cat_1, color='y')
                plt.bar(get_dates, cat_3, bottom=cat_1 + cat_2, color='g')
                plt.bar(get_dates, cat_4, bottom=cat_1 + cat_2 + cat_3, color='c')
                plt.bar(get_dates, cat_5, bottom=cat_1 + cat_2 + cat_3 + cat_4, color='m')

            else:
                # get education level and marital status per each gender
                cat_1 = gc_df.cat_1
                cat_2 = gc_df.cat_2
                cat_3 = gc_df.cat_3

                # add education level and marital status data to plot
                plt.subplot(131 + counter)
                plt.bar(get_dates, cat_1, color='b')
                plt.bar(get_dates, cat_2, bottom=cat_1, color='y')
                plt.bar(get_dates, cat_3, bottom=cat_1 + cat_2, color='g')

            plt.title(f"{desc} {self.en_props[_property]}  categories")
            plt.ylabel("proportions")
            plt.xlabel("Year")
            plt.ylim(0, )
            plt.legend([f'cat_{c}' for c in categories])

            # incrementing counter
            counter += 2
        # save and display plots for property categories by gender
        plt.savefig(self.outputpath / (_property + self.datestamp + '.png'), format='png')
        plt.show()

    def plot_non_categorical_properties_by_gender(self, _property):
        """ a function to plot non categorical properties of lifestyle module grouped by gender

         :param _property: any other non categorical property defined in lifestyle module """
        # get simulation years per property
        get_years = pd.to_datetime(self.dfs[_property].index).year

        # sum and store property values
        get_totals = self.dfs[_property].sum(axis=1)

        # get male proportions per each property
        males_prop = self.dfs[_property].M.sum(axis=1) / get_totals

        # get female proportions per each property
        females_prop = self.dfs[_property].F.sum(axis=1) / get_totals

        males_prop = males_prop.fillna(0)
        females_prop = females_prop.fillna(0)
        # add data to plot
        plt.bar(np.asarray(get_years), males_prop, color='b')
        plt.bar(np.asarray(get_years), females_prop, bottom=males_prop, color='y')

        # provide plot descriptors
        plt.title(f"{self.en_props[_property]} by gender")
        plt.xlabel("Year")
        plt.ylabel("Male and Females proportions")
        plt.ylim(0,)
        plt.legend(['Male proportions', 'Female Proportions'])
        # save and display plot
        plt.savefig(self.outputpath / (_property + self.datestamp + '.png'), format='png')
        plt.show()

    def display_all_categorical_and_non_categorical_plots_by_gender(self):
        """ a function to display plots for both categorical and non categorical properties grouped by gender """
        for _property in self.en_props.keys():
            if _property == 'li_bmi':
                self.plot_categorical_properties_by_gender(_property, ['0', '1', '2', '3', '4', '5'])
            elif _property == 'li_wealth':
                self.plot_categorical_properties_by_gender(_property, ['1', '2', '3', '4', '5'])
            elif _property in ['li_mar_stat', 'li_ed_lev']:
                self.plot_categorical_properties_by_gender(_property, ['1', '2', '3'])
            else:
                pass
                self.plot_non_categorical_properties_by_gender(_property)

    # 2. AGE GROUP PLOTS
    # -------------------------------------------------------------------------------------------------------------
    def plot_categorical_properties_by_age_group(self, _property: str, categories: list):
        """ a function to plot all categorical properties of lifestyle module grouped by age group. Available
        categories per property include;

        1. bmi
            bmi is categorised as follows
                  category 1: <18
                  category 2: 18-24.9
                  category 3: 25-29.9
                  category 4: 30-34.9
                  category 5: 35+
            bmi is 0 until age 15

        2. wealth level
            wealth level is categorised as follows as follows;

                    Urban                               |         Rural
                    ------------------------------------|----------------------------------------------
                    level 1 = 75% wealth level           |  level 1 = 11% wealth level
                    level 2 = 16% wealth level          |  level 2 = 21% wealth level
                    level 3 = 5% wealth level           |  level 3 = 23% wealth level
                    level 4 = 2% wealth level           |  level 4 = 23% wealth level
                    level 5 = 2% wealth level           |  level 5 = 23% wealth level

        3. education level
             education level is categorised as follows
                    level 1: not in education
                    level 2: primary education
                    level 3 : secondary+ education )

        4. marital status
            marital status is categorised as follows
                    category 1: never married
                    category 2: married
                    category 3: widowed or divorced

        :param _property: any other categorical property defined in lifestyle module
        :param categories: a list of categories """
        # create a new dataframe to contain data of age groups against categories
        new_df = pd.DataFrame()
        # loop through categories and get data into age groups categories dataframe
        for cat in categories:
            new_df[f'cat_{cat}'] = self.dfs[_property]['M'][cat].sum(axis=0) + self.dfs[_property]['F'][cat].sum(axis=0)

        # convert values to proportions
        new_df = new_df.apply(lambda row: row / row.sum(), axis=1)

        # choose block of code according to the property given to this function
        if _property == 'li_bmi':
            # get bmi categories per each age group
            cat_0 = new_df.cat_0
            cat_1 = new_df.cat_1
            cat_2 = new_df.cat_2
            cat_3 = new_df.cat_3
            cat_4 = new_df.cat_4
            cat_5 = new_df.cat_5

            plt.bar(new_df.index, cat_0, color='r')
            plt.bar(new_df.index, cat_1, bottom=cat_0, color='b')
            plt.bar(new_df.index, cat_2, bottom=cat_0 + cat_1, color='y')
            plt.bar(new_df.index, cat_3, bottom=cat_0 + cat_1 + cat_2, color='g')
            plt.bar(new_df.index, cat_4, bottom=cat_0 + cat_1 + cat_2 + cat_3, color='c')
            plt.bar(new_df.index, cat_5, bottom=cat_0 + cat_1 + cat_2 + cat_3 + cat_4, color='m')

        elif _property == 'li_wealth':
            # get wealth categories per each age group
            cat_1 = new_df.cat_1
            cat_2 = new_df.cat_2
            cat_3 = new_df.cat_3
            cat_4 = new_df.cat_4
            cat_5 = new_df.cat_5

            plt.bar(new_df.index, cat_1, color='b')
            plt.bar(new_df.index, cat_2, bottom=cat_1, color='y')
            plt.bar(new_df.index, cat_3, bottom=cat_1 + cat_2, color='g')
            plt.bar(new_df.index, cat_4, bottom=cat_1 + cat_2 + cat_3, color='c')
            plt.bar(new_df.index, cat_5, bottom=cat_1 + cat_2 + cat_3 + cat_4, color='m')

        else:
            # get education level and marital status per each age group
            cat_1 = new_df.cat_1
            cat_2 = new_df.cat_2
            cat_3 = new_df.cat_3

            plt.bar(new_df.index, cat_1, color='b')
            plt.bar(new_df.index, cat_2, bottom=cat_1, color='y')
            plt.bar(new_df.index, cat_3, bottom=cat_1 + cat_2, color='g')

        plt.title(f"{self.en_props[_property]} category by age group")
        plt.xlabel("Year")
        plt.ylabel("age group proportions")
        plt.ylim(0, )
        plt.legend([f'cat_{c}' for c in categories], loc='upper right')
        plt.savefig(self.outputpath / (_property + 'by_age_range' + self.datestamp + '.png'), format='png')
        plt.show()

    def plot_non_categorical_properties_by_age_group(self, _property: str):
        """ a function to plot non categorical properties of lifestyle module grouped by age group

         :param _property: any other non categorical property defined in lifestyle module """
        # sum whole dataframe per each property
        get_totals = self.dfs[_property].sum(axis=1).sum()

        # age and store age groups from dataframe. age groups are the same for both males and females so choosing
        # either doesn't matter
        get_age_group = self.dfs[_property]["M"].columns

        # loop through age groups and get plot data
        for age_group in get_age_group:
            get_age_group_props = (self.dfs[_property]['M'][age_group] +
                                   self.dfs[_property]['F'][age_group]).sum() / get_totals

            plt.bar(age_group, get_age_group_props, color='darkturquoise')

        plt.title(f"{self.en_props[_property]} by age groups")
        plt.xlabel("age groups")
        plt.ylabel("proportions")
        plt.ylim(0, )
        plt.legend([self.en_props[_property]], loc='upper right')
        plt.savefig(self.outputpath / (_property + 'by_age_group' + self.datestamp + '.png'), format='png')
        plt.show()

    def display_all_categorical_and_non_categorical_plots_by_age_group(self):
        """ a function that will display plots of all enhanced lifestyle properties grouped by age group """
        for _property in self.en_props.keys():
            if _property == 'li_bmi':
                self.plot_categorical_properties_by_age_group(_property, ['0', '1', '2', '3', '4', '5'])
            elif _property == 'li_wealth':
                self.plot_categorical_properties_by_age_group(_property, ['1', '2', '3', '4', '5'])
            elif _property in ['li_mar_stat', 'li_ed_lev']:
                self.plot_categorical_properties_by_age_group(_property, ['1', '2', '3'])
            else:
                self.plot_non_categorical_properties_by_age_group(_property)

    def circumcised_men_plot(self, _property: str = None):
        """ a function to plot for men circumcised

        :param _property: circumcision property defined in enhanced lifestyle module """
        # Examine Proportion Men Circumcised:
        circ = extract_formatted_series(self.all_logs[_property])
        circ.plot()
        plt.title('Proportion of Adult Men Circumcised')
        plt.ylim(0, 0.30)
        plt.ylabel("proportions")
        # save and display circumcision plot
        plt.savefig(self.outputpath / (_property + self.datestamp + '.png'), format='png')
        plt.show()

    def sex_workers_plot(self, _property: str = None):
        """ a function to plot for women sex workers

        :param _property: sex workers property defined in enhanced lifestyle module """
        # Examine Proportion Women sex Worker:
        fsw = extract_formatted_series(self.all_logs[_property])
        fsw.plot()
        plt.title('Proportion of 15-49 Women Sex Workers')
        plt.ylim(0, 0.01)
        plt.ylabel("proportions")
        # save and display circumcision plot
        plt.savefig(self.outputpath / (_property + self.datestamp + '.png'), format='png')
        plt.show()


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
    end_date = Date(2050, 1, 1)
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
# output = parse_log_file(Path("./outputs/enhanced_lifestyle__2022-07-12T132341.log"))

# construct a dict of dataframes using lifestyle logs
logs_df = output['tlo.methods.enhanced_lifestyle']

# initialise LifestylePlots class
g_plots = LifeStylePlots(logs=logs_df)

# plot by gender
g_plots.display_all_categorical_and_non_categorical_plots_by_gender()

# plot by age groups
g_plots.display_all_categorical_and_non_categorical_plots_by_age_group()

# plot male circumcision
g_plots.circumcised_men_plot('li_is_circ')
#
# # plot women sex workers
g_plots.sex_workers_plot('li_is_sexworker')
