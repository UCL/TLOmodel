"""
This is Lifestyle analyses file.  It seeks to show trends in different lifestyle properties. It plots properties by
gender and age groups
"""
# %% Import Statements
import datetime
from pathlib import Path
from typing import Dict, NamedTuple, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file, unflatten_flattened_multi_index_in_logging
from tlo.methods import demography, enhanced_lifestyle, simplified_births


def add_footnote(fig: plt.Figure, footnote: str):
    """ A function that adds a footnote below each plot. Here we are explaining what a denominator for every
    graph is """
    fig.figure.text(0.5, 0.01, footnote, ha="center", fontsize=10,
                    bbox={"facecolor": "gray", "alpha": 0.3, "pad": 5})


class PlotDescriptor(NamedTuple):
    label: str
    footnote_stem: str

    @property
    def per_gender_footnote(self):
        return f"Denominator: {self.footnote_stem} per gender"

    @property
    def per_age_group_footnote(self):
        return f"Denominator: {self.footnote_stem} per each age-group"


class LifeStylePlots:
    """ a class for plotting lifestyle properties by both gender and age groups """

    def __init__(self, logs=None, path: Optional[str] = None):

        # create a dictionary for lifestyle properties and their descriptions, to be used as plot descriptors.
        self.wealth_desc: Dict[str, str] = {'1': "highest wealth level", '5': "lowest wealth level"}
        self.en_props = {
            "li_urban": PlotDescriptor("currently urban", "Sum of all individuals"),
            "li_wealth": PlotDescriptor("wealth level", "Sum of individuals in all wealth levels per urban or rural"),
            "li_low_ex": PlotDescriptor("currently low exercise", "Sum of all individuals aged 15+"),
            "li_tob": PlotDescriptor("current using tobacco", "Sum of all individuals aged 15+"),
            "li_ex_alc": PlotDescriptor("current excess alcohol", "Sum of all individuals aged 15+"),
            "li_mar_stat": PlotDescriptor("marital status", "Sum of individuals aged 15+ in all marital status"),
            "li_in_ed": PlotDescriptor("currently in education", "Sum of individuals aged between 5-19 in education"),
            "li_ed_lev": PlotDescriptor("education level", "Sum of individuals aged 15-49 in all education levels"),
            "li_unimproved_sanitation": PlotDescriptor("unimproved sanitation", "Sum of all individuals "
                                                                                "in urban or rural"),
            "li_no_clean_drinking_water": PlotDescriptor("no clean drinking water", "Sum of all individuals per"
                                                                                    "urban or rural"),

            "li_wood_burn_stove": PlotDescriptor("wood burn stove", "Sum of all individuals per urban or rural"),
            "li_no_access_handwashing": PlotDescriptor("no access hand washing", "Sum of all individuals"),
            "li_high_salt": PlotDescriptor("high salt", "Sum of all individuals"),
            "li_high_sugar": PlotDescriptor("high sugar", "Sum of all individuals"),
            "li_bmi": PlotDescriptor("bmi", "Sum of 15+ individuals in all bmi categories in  individuals "
                                            "per urban or rural"),
            "li_is_circ": PlotDescriptor("Male circumcision", "Sum of all males"),
            "li_is_sexworker": PlotDescriptor("sex workers", "Sum of all females aged between 15-49"),
        }

        # A dictionary to map properties and their description. Useful when setting plot legend
        self.categories_desc: dict = {
            'li_bmi': ["bmi category 1", "bmi category 2", "bmi category 3", "bmi category 4", "bmi category 5"],
            'li_wealth': ['wealth level 1', 'wealth level 2', 'wealth level 3', 'wealth level 4', 'wealth level 5'],
            'li_mar_stat': ['Never Married', 'Married', 'Divorced or Widowed'],
            'li_ed_lev': ['No education', 'Primary edu', 'secondary education']
        }

        # create a dictionary to defining individual's rural or urban state
        self._rural_urban_state = {
            'True': 'Urban',
            'False': 'Rural'
        }

        # define all properties that are categorised by rural or urban in addition to age and sex
        self.cat_by_rural_urban_props = ['li_wealth', 'li_bmi', 'li_low_ex', 'li_ex_alc', 'li_wood_burn_stove',
                                         'li_unimproved_sanitation',
                                         'li_no_clean_drinking_water']

        # date-stamp to label log files and any other outputs
        self.datestamp: str = datetime.date.today().strftime("__%Y_%m_%d")

        # a dictionary for gender descriptions. to be used when plotting by gender
        self.gender_des: Dict[str, str] = {'M': 'Males', 'F': 'Females'}

        # get all logs
        self.all_logs = logs

        # store un flattened logs
        self.dfs = self.construct_dfs(self.all_logs)

        self.outputpath = Path(path)  # folder for convenience of storing outputs

    def construct_dfs(self, lifestyle_log) -> dict:
        """ Create dict of pd.DataFrames containing counts of different lifestyle properties by date, sex and
        age-group """
        return {
            k: unflatten_flattened_multi_index_in_logging(v.set_index('date'))
            for k, v in lifestyle_log.items() if k in self.en_props.keys()
        }

    def custom_axis_formatter(self, df: pd.DataFrame, ax):
        """
        create a custom date formatter since the default pandas date formatter works well with line graphs. see an
        Adapted from https://stackoverflow.com/a/30135182

        :param df: pandas dataframe or series
        :param ax: matplotlib AxesSubplot object
        """
        # make the tick labels empty so the labels don't get too crowded
        tick_labels = [''] * len(df.index)
        # Every 12th tick label includes the year
        tick_labels[::12] = [item.strftime('%Y') for item in df.index[::12]]
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(tick_labels))
        ax.figure.autofmt_xdate()

    # 1. PLOT BY GENDER
    # --------------------------------------------------------------------------------------------------------
    def plot_categorical_properties_by_gender(self, li_property: str):
        """ a function to plot all categorical properties of lifestyle module grouped by gender. Available
        categories per property include;

        1. bmi
            categorised as follows
                  category 1: <18
                  category 2: 18-24.9
                  category 3: 25-29.9
                  category 4: 30-34.9
                  category 5: 35+
            bmi is nan until age 15

        2. wealth level
            categorised as follows;

                    Urban                               |         Rural
                    ------------------------------------|----------------------------------------------
                    level 1 = 75% wealth level          |  level 1 = 11% wealth level
                          2 = 16% wealth level          |        2 = 21% wealth level
                          3 = 5% wealth level           |        3 = 23% wealth level
                          4 = 2% wealth level           |        4 = 23% wealth level
                          5 = 2% wealth level           |        5 = 23% wealth level

        3. education level
             categorised as follows
                    level 1: no education
                          2: primary education
                          3 : secondary+ education

        4. marital status
            categorised as follows
                    category 1: never married
                             2: married
                             3: widowed or divorced

        :param li_property: any other categorical property defined in lifestyle module """

        categories = sorted(set(self.dfs[li_property].columns.get_level_values(li_property)))

        col: int = 0  # counter for indexing purposes
        # create subplots
        fig, axes = plt.subplots(nrows=2 if li_property in self.cat_by_rural_urban_props
                                            or li_property == 'li_ed_lev' else 1, ncols=2, figsize=(10, 5))
        for gender, desc in self.gender_des.items():

            df_dict = dict()
            if li_property == 'li_ed_lev' or li_property in self.cat_by_rural_urban_props:
                _row: int = 0  # row counter
                _rows_counter: int = 0  # a counter for plotting. setting rows
                key_value_desc = self.wealth_desc.items() if li_property == 'li_ed_lev' else \
                    self._rural_urban_state.items()
                for _key, _value in key_value_desc:
                    gc_df = pd.DataFrame()
                    for cat in categories:
                        if li_property == 'li_ed_lev':
                            gc_df[f'cat_{cat}'] = self.dfs[li_property][gender][_key][cat].sum(axis=1)
                        else:
                            gc_df[f'cat_{cat}'] = self.dfs[li_property][_key][gender][cat].sum(axis=1)
                    # add dataframe to dictionary
                    df_dict[f'{gender}_{_value}_{_row}'] = gc_df
                    _row += 1
            else:
                gc_df = pd.DataFrame()
                for cat in categories:
                    gc_df[f'cat_{cat}'] = self.dfs[li_property][gender][cat].sum(axis=1)
                    # normalise the probabilities
                    df_dict[gender] = gc_df
            for _key in df_dict.keys():
                df_dict[_key] = df_dict[_key].apply(lambda row: row / row.sum(), axis=1)
                # do plotting
                ax = df_dict[_key].plot(kind='bar', stacked=True,
                                        ax=axes[int(_key.split("_")[-1]), col] if
                                        li_property in self.cat_by_rural_urban_props or li_property == 'li_ed_lev'
                                        else axes[col],
                                        legend=None,
                                        title=f"{_key.split('_')[1]} {desc} {self.en_props[li_property].label}"
                                              ' categories' if li_property in self.cat_by_rural_urban_props or
                                                               li_property == 'li_ed_lev'
                                        else f"{desc} {self.en_props[li_property].label}",
                                        ylabel=f"{self.en_props[li_property].label} proportions", xlabel="Year"
                                        )
                self.custom_axis_formatter(df_dict[_key], ax)
            # increase counter
            col += 1
        fig.legend(self.categories_desc[li_property], loc='lower left', bbox_to_anchor=(0.8, 0.6))
        add_footnote(fig, f'{self.en_props[li_property].per_gender_footnote}')
        fig.tight_layout()
        plt.savefig(self.outputpath / (li_property + self.datestamp + '.png'), format='png')
        plt.close(fig=fig)  # close figure after saving it to avoid opening multiple figures

    def plot_non_categorical_properties_by_gender(self, li_property: str):
        """ a function to plot non-categorical properties of lifestyle module grouped by gender

        :param li_property: any other non-categorical property defined in lifestyle module """

        # set y-axis limit.
        y_lim: float = 0.8
        if li_property in ['li_no_access_handwashing', 'li_high_salt', 'li_wood_burn_stove', 'li_in_ed']:
            y_lim = 1.0

        if li_property in ['li_tob', 'li_ex_alc']:
            y_lim = 0.3

        # plot for male circumcision and female sex workers
        if li_property in ['li_is_circ', 'li_is_sexworker']:
            self.male_circumcision_and_sex_workers_plot(li_property)

        else:
            col: int = 0  # counter for indexing purposes
            # create subplots
            fig, axes = plt.subplots(nrows=2 if li_property in self.cat_by_rural_urban_props
                                                or li_property == 'li_ed_lev' else 1, ncols=2, figsize=(10, 5))
            for gender, desc in self.gender_des.items():

                df_dict = dict()
                if li_property in self.cat_by_rural_urban_props:
                    _row: int = 0  # row counter
                    _rows_counter: int = 0  # a counter for plotting. setting rows
                    for _key, _value in self._rural_urban_state.items():
                        df_dict[f'{gender}_{_value}_{_row}'] = self.dfs[li_property][_key][gender]["True"].sum(
                                axis=1) / self.dfs[li_property][_key][gender].sum(axis=1)
                        _row += 1

                else:
                    df = self.dfs[li_property].reorder_levels([0, 2, 1, 3], axis=1) if li_property == 'li_in_ed' \
                        else self.dfs[li_property]
                    df_dict[gender] = df[gender]["True"].sum(axis=1) / df[gender].sum(axis=1)
                for _key in df_dict.keys():
                    # do plotting
                    ax = df_dict[_key].plot(kind='bar', stacked=True,
                                            ax=axes[int(_key.split("_")[-1]), col] if
                                            li_property in self.cat_by_rural_urban_props else axes[col],
                                            ylim=(0, y_lim),
                                            legend=None,
                                            color='darkturquoise',
                                            title=f"{_key.split('_')[1]} {desc} {self.en_props[li_property].label}"
                                            if li_property in self.cat_by_rural_urban_props
                                            else f"{desc} {self.en_props[li_property].label}",
                                            ylabel=f"{self.en_props[li_property].label} proportions", xlabel="Year"
                                            )
                    self.custom_axis_formatter(df_dict[_key], ax)
                # increase counter
                col += 1
            fig.legend([self.en_props[li_property].label], loc='lower left', bbox_to_anchor=(0.75, 0.8))
            # save and display plots for property categories by gender
            add_footnote(fig, f'{self.en_props[li_property].per_gender_footnote}')
            fig.tight_layout()
            plt.savefig(self.outputpath / (li_property + self.datestamp + '.png'), format='png')
            plt.close(fig=fig)  # close figure after saving it to avoid opening multiple figures

    def display_all_categorical_and_non_categorical_plots_by_gender(self):
        """ a function to display plots for both categorical and non-categorical properties grouped by gender """
        for _property in self.en_props.keys():
            if _property in ['li_bmi', 'li_wealth', 'li_mar_stat', 'li_ed_lev']:
                self.plot_categorical_properties_by_gender(_property)
            else:
                self.plot_non_categorical_properties_by_gender(_property)

    def plot_categorical_properties_by_age_group(self, li_property: str):
        """ a function to plot all categorical properties of lifestyle module grouped by age group. Available
            categories per property include;

            1. bmi
                categorised as follows
                      category 1: <18
                      category 2: 18-24.9
                      category 3: 25-29.9
                      category 4: 30-34.9
                      category 5: 35+
                bmi is nan until age 15

            2. wealth level
                categorised as follows;

                        Urban                               |         Rural
                        ------------------------------------|----------------------------------------------
                        level 1 = 75% wealth level          |  level 1 = 11% wealth level
                              2 = 16% wealth level          |        2 = 21% wealth level
                              3 = 5% wealth level           |        3 = 23% wealth level
                              4 = 2% wealth level           |        4 = 23% wealth level
                              5 = 2% wealth level           |        5 = 23% wealth level

            3. education level
                 categorised as follows
                        level 1: not in education
                              2: primary education
                              3 : secondary+ education

            4. marital status
                categorised as follows
                        category 1: never married
                                 2: married
                                 3: widowed or divorced

        :param li_property: any other categorical property defined in lifestyle module """
        categories = sorted(set(self.dfs[li_property].columns.get_level_values(li_property)))

        # select logs from the latest year. In this case we are selecting year 2021
        all_logs_df = self.dfs[li_property]
        mask = (all_logs_df.index > pd.to_datetime('2021-01-01')) & (all_logs_df.index <= pd.to_datetime('2022-01-01'))
        self.dfs[li_property] = self.dfs[li_property].loc[mask]

        # create subplots
        fig, axes = plt.subplots(ncols=2 if li_property in self.cat_by_rural_urban_props
                                            or li_property == 'li_ed_lev' else 1, figsize=(10, 5))

        df_dict = dict()
        if li_property == 'li_ed_lev' or li_property in self.cat_by_rural_urban_props:
            _col: int = 0  # column counter
            key_value_desc = self.wealth_desc.items() if li_property == 'li_ed_lev' else \
                self._rural_urban_state.items()
            for _key, _value in key_value_desc:
                gc_df = pd.DataFrame()
                for cat in categories:
                    if li_property == 'li_ed_lev':
                        gc_df[f'cat_{cat}'] = self.dfs[li_property]['M'][_key][cat].sum(axis=0) + \
                                              self.dfs[li_property]['F'][_key][cat].sum(axis=0)
                    else:
                        gc_df[f'cat_{cat}'] = self.dfs[li_property][_key]['M'][cat].sum(axis=0) + \
                                              self.dfs[li_property][_key]['F'][cat].sum(axis=0)
                # add dataframe to dictionary
                df_dict[f'{_value}_{_col}'] = gc_df
                _col += 1

        else:
            gc_df = pd.DataFrame()
            for cat in categories:
                gc_df[f'cat_{cat}'] = self.dfs[li_property]['M'][cat].sum(axis=0) + \
                                      self.dfs[li_property]['F'][cat].sum(axis=0)
                # add to dataframe
                df_dict[f'wealth_cat_{cat}'] = gc_df

        for _key in df_dict.keys():
            df_dict[_key] = df_dict[_key].apply(lambda row: row / row.sum(), axis=1)
            # do plotting
            df_dict[_key].plot(kind='bar', stacked=True,
                               ax=axes[int(_key.split("_")[-1])] if
                               li_property in self.cat_by_rural_urban_props or li_property == 'li_ed_lev'
                               else axes,
                               legend=None,
                               title=f"{_key.split('_')[0]} {self.en_props[li_property].label}"
                                     ' categories by age group in 2021' if li_property in
                                                                           self.cat_by_rural_urban_props or
                                                                           li_property == 'li_ed_lev'
                               else f"{self.en_props[li_property].label}",
                               ylabel=f"{self.en_props[li_property].label} proportions", xlabel="Year"
                               )
            # self.custom_axis_formatter(df_dict[_key], ax)

        fig.legend(self.categories_desc[li_property], loc='lower left', bbox_to_anchor=(0.8, 0.6))
        add_footnote(fig, f'{self.en_props[li_property].per_age_group_footnote}')
        fig.tight_layout()
        plt.savefig(self.outputpath / (li_property + self.datestamp + '.png'), format='png')
        plt.close(fig=fig)  # close figure after saving it to avoid opening multiple figures

    def plot_non_categorical_properties_by_age_group(self, li_property):
        """ plot all non-categorical properties by age group """
        # select logs from the latest year. In this case we are selecting year 2021
        y_lim: float = 1.0
        if li_property in ['li_is_sexworker']:
            y_lim = 0.040

        all_logs_df = self.dfs[li_property]
        mask = (all_logs_df.index > pd.to_datetime('2021-01-01')) & (all_logs_df.index <= pd.to_datetime('2022-01-01'))
        self.dfs[li_property] = self.dfs[li_property].loc[mask]

        # create subplots
        fig, axes = plt.subplots(nrows=2 if li_property in self.cat_by_rural_urban_props or li_property ==
                                            'li_in_ed' else 1,
                                            figsize=(10, 5), sharex=True)

        df_dict = dict()
        if li_property == 'li_in_ed' or li_property in self.cat_by_rural_urban_props:
            _col: int = 0  # column counter
            key_value_desc = self.wealth_desc.items() if li_property == 'li_in_ed' else \
                self._rural_urban_state.items()
            for _key, _value in key_value_desc:
                temp_df = pd.DataFrame()
                for _bool_value in ['True', 'False']:
                    if li_property == 'li_in_ed':
                        temp_df[_bool_value] = self.dfs[li_property]['M'][_key][_bool_value].sum(axis=0) + \
                                               self.dfs[li_property]['F'][_key][_bool_value].sum(axis=0)

                    else:
                        temp_df[_bool_value] = self.dfs[li_property][_key]['M'][_bool_value].sum(axis=0) + \
                                    self.dfs[li_property][_key]['F'][_bool_value].sum(axis=0)

                df_dict[f'{_value}_{_col}'] = temp_df['True'] / temp_df.sum(axis=1)
                _col += 1

        else:
            plot_df = pd.DataFrame()
            for _bool_value in ['True', 'False']:
                plot_df[_bool_value] = self.dfs[li_property]['M'][_bool_value].sum(axis=0) + \
                                  self.dfs[li_property]['F'][_bool_value].sum(axis=0)

            df_dict['non_urban_1'] = plot_df['True'] / plot_df.sum(axis=1)

        for _key in df_dict.keys():
            # do plotting
            df_dict[_key].plot(kind='bar', stacked=True,
                               ax=axes[int(_key.split("_")[-1])] if
                               li_property in self.cat_by_rural_urban_props or li_property == 'li_in_ed' else axes,
                               ylim=(0, y_lim),
                               legend=None,
                               color='darkturquoise',
                               title=f"{self.en_props[li_property].label} by age group in 2021, {_key.split('_')[0]}"
                                     if li_property in self.cat_by_rural_urban_props or li_property == 'li_in_ed' else
                               f"{self.en_props[li_property].label} by age group in 2021",
                               ylabel=f"{self.en_props[li_property].label} proportions", xlabel="Year"
                               )

        fig.legend([self.en_props[li_property].label], loc='lower left', bbox_to_anchor=(0.8, 0.7))
        add_footnote(fig, f'{self.en_props[li_property].per_age_group_footnote}')
        fig.tight_layout()
        plt.savefig(self.outputpath / (li_property + self.datestamp + '.png'), format='png')
        plt.close(fig=fig)  # close figure after saving it to avoid opening multiple figures

    def display_all_categorical_and_non_categorical_plots_by_age_group(self):
        """ a function that will display plots of all enhanced lifestyle properties grouped by age group """
        for _property in self.en_props.keys():
            if _property in ['li_bmi', 'li_wealth', 'li_mar_stat', 'li_ed_lev']:
                self.plot_categorical_properties_by_age_group(_property)
            else:
                self.plot_non_categorical_properties_by_age_group(_property)

    def male_circumcision_and_sex_workers_plot(self, _property: str = None):
        """ a function to plot for men circumcised and female sex workers

        :param _property: circumcision or female sex worker property defined in enhanced lifestyle module """

        # create a dataframe that will hold proportions per each lifestyle property
        gender: str = 'M'  # key in the logs file for men circumcised
        max_ylim = 0.30  # define y limit in plot

        fig, axes = plt.subplots(nrows=1, figsize=(10, 5))
        # check property if not circumcision. if true, update gender and y limit values
        if not _property == 'li_is_circ':
            gender = 'F'
            max_ylim = 0.01

        # get proportions per property
        totals_df = self.dfs[_property][gender]["True"].sum(axis=1) / self.dfs[_property][gender].sum(axis=1)

        ax = totals_df.plot(kind='bar', ylim=(0, max_ylim), ylabel=f'{self.en_props[_property].label} proportions',
                            xlabel="Year",
                            color='darkturquoise', title=f"{self.en_props[_property].label} Percentage")
        # format x-axis
        self.custom_axis_formatter(totals_df, ax)
        # save and display plots
        add_footnote(fig, f'{self.en_props[_property].per_gender_footnote}')
        plt.tight_layout()
        plt.savefig(self.outputpath / (_property + self.datestamp + '.png'), format='png')


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
    end_date = Date(2011, 1, 1)
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


# Run the Simulation
sim = run()

# read the results
output = parse_log_file(sim.log_filepath)

# construct a dict of dataframes using lifestyle logs
logs_df = output['tlo.methods.enhanced_lifestyle']

# initialise LifestylePlots class
g_plots = LifeStylePlots(logs=logs_df, path="./outputs")

# plot by gender
g_plots.display_all_categorical_and_non_categorical_plots_by_gender()

# plot by age groups
g_plots.display_all_categorical_and_non_categorical_plots_by_age_group()
