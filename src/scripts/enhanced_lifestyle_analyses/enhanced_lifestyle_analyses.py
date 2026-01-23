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
import numpy as np


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
            "li_herbal_medication": PlotDescriptor("herbal medication use", "Sum of all individuals per urban or rural"
                                                   ),
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
                                         'li_no_clean_drinking_water', 'li_herbal_medication']

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

    # def plot_non_categorical_properties_by_gender(self, li_property: str):
    #     """ a function to plot non-categorical properties of lifestyle module grouped by gender
    #
    #     :param li_property: any other non-categorical property defined in lifestyle module """
    #
    #     # set y-axis limit.
    #     y_lim: float = 0.8
    #     if li_property in ['li_no_access_handwashing', 'li_high_salt', 'li_wood_burn_stove', 'li_in_ed']:
    #         y_lim = 1.0
    #
    #     if li_property in ['li_tob', 'li_ex_alc']:
    #         y_lim = 0.3
    #
    #     # plot for male circumcision and female sex workers
    #     if li_property in ['li_is_circ', 'li_is_sexworker']:
    #         self.male_circumcision_and_sex_workers_plot(li_property)
    #
    #     else:
    #         col: int = 0  # counter for indexing purposes
    #         # create subplots
    #         fig, axes = plt.subplots(nrows=2 if li_property in self.cat_by_rural_urban_props
    #                                             or li_property == 'li_ed_lev' else 1, ncols=2, figsize=(10, 5))
    #         for gender, desc in self.gender_des.items():
    #
    #             df_dict = dict()
    #             if li_property in self.cat_by_rural_urban_props:
    #                 _row: int = 0  # row counter
    #                 _rows_counter: int = 0  # a counter for plotting. setting rows
    #                 for _key, _value in self._rural_urban_state.items():
    #                     df_dict[f'{gender}_{_value}_{_row}'] = self.dfs[li_property][_key][gender]["True"].sum(
    #                             axis=1) / self.dfs[li_property][_key][gender].sum(axis=1)
    #                     _row += 1
    #
    #             else:
    #                 df = self.dfs[li_property].reorder_levels([0, 2, 1, 3], axis=1) if li_property == 'li_in_ed' \
    #                     else self.dfs[li_property]
    #                 df_dict[gender] = df[gender]["True"].sum(axis=1) / df[gender].sum(axis=1)
    #             for _key in df_dict.keys():
    #                 # do plotting
    #                 ax = df_dict[_key].plot(kind='bar', stacked=True,
    #                                         ax=axes[int(_key.split("_")[-1]), col] if
    #                                         li_property in self.cat_by_rural_urban_props else axes[col],
    #                                         ylim=(0, y_lim),
    #                                         legend=None,
    #                                         color='darkturquoise',
    #                                         title=f"{_key.split('_')[1]} {desc} {self.en_props[li_property].label}"
    #                                         if li_property in self.cat_by_rural_urban_props
    #                                         else f"{desc} {self.en_props[li_property].label}",
    #                                         ylabel=f"{self.en_props[li_property].label} proportions", xlabel="Year"
    #                                         )
    #                 self.custom_axis_formatter(df_dict[_key], ax)
    #             # increase counter
    #             col += 1
    #         fig.legend([self.en_props[li_property].label], loc='lower left', bbox_to_anchor=(0.75, 0.8))
    #         # save and display plots for property categories by gender
    #         add_footnote(fig, f'{self.en_props[li_property].per_gender_footnote}')
    #         fig.tight_layout()
    #         plt.savefig(self.outputpath / (li_property + self.datestamp + '.png'), format='png')
    #         plt.close(fig=fig)  # close figure after saving it to avoid opening multiple figures

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
            return  # Exit early for these properties

        # Check the actual structure of the DataFrame
        df_property = self.dfs[li_property]
        first_level_values = df_property.columns.get_level_values(0).unique()

        # Determine the structure
        has_gender_first = all(v in ['F', 'M'] for v in first_level_values)
        has_urban_rural_first = all(
            str(v).lower() in ['true', 'false'] or v in [True, False] for v in first_level_values)

        print(f"DEBUG for {li_property}:")
        print(f"  First level values: {list(first_level_values)}")
        print(f"  has_gender_first: {has_gender_first}")
        print(f"  has_urban_rural_first: {has_urban_rural_first}")

        # Special handling for li_in_ed which has different structure
        if li_property == 'li_in_ed':
            # li_in_ed has structure [gender][li_wealth][li_in_ed][age_years]
            # We need to aggregate across wealth levels and ages
            col: int = 0
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            axes = np.array([[axes[0], axes[1]]])

            for gender, desc in self.gender_des.items():
                # Get all columns for this gender
                gender_cols = df_property[gender]
                # Sum across all wealth levels, ages, and True/False
                total_in_education = pd.Series(0, index=df_property.index)
                total_population = pd.Series(0, index=df_property.index)

                # Iterate through all columns for this gender
                for col_name in gender_cols.columns:
                    # col_name is a tuple like ('1', 'True', '10')
                    wealth_level, in_ed_status, age = col_name
                    data_series = gender_cols[col_name]

                    total_population += data_series
                    if in_ed_status == 'True' or in_ed_status == True:
                        total_in_education += data_series

                proportion = total_in_education / total_population.replace(0, np.nan)
                proportion = proportion.fillna(0)

                ax = proportion.plot(kind='bar', stacked=True,
                                     ax=axes[0, col],
                                     ylim=(0, y_lim),
                                     legend=None,
                                     color='darkturquoise',
                                     title=f"{desc} {self.en_props[li_property].label}",
                                     ylabel=f"{self.en_props[li_property].label} proportions",
                                     xlabel="Year"
                                     )
                self.custom_axis_formatter(proportion, ax)
                col += 1

            fig.legend([self.en_props[li_property].label], loc='lower left', bbox_to_anchor=(0.75, 0.8))
            add_footnote(fig, f'{self.en_props[li_property].per_gender_footnote}')
            fig.tight_layout()
            plt.savefig(self.outputpath / (li_property + self.datestamp + '.png'), format='png')
            plt.close(fig=fig)
            return

        # For other properties, determine plotting structure
        if has_urban_rural_first:
            # Structure: [urban/rural][gender][property][age_range]
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
            col = 0

            for gender, desc in self.gender_des.items():
                df_dict = {}
                row = 0

                for urban_key, urban_desc in self._rural_urban_state.items():
                    # Convert key to match what's in the DataFrame
                    if urban_key == 'True' and True in first_level_values:
                        df_key = True
                    elif urban_key == 'False' and False in first_level_values:
                        df_key = False
                    elif urban_key in first_level_values:
                        df_key = urban_key
                    elif str(urban_key) in [str(v) for v in first_level_values]:
                        for v in first_level_values:
                            if str(v) == str(urban_key):
                                df_key = v
                                break
                    else:
                        print(f"WARNING: Urban key {urban_key} not found in {li_property}")
                        df_dict[f'{gender}_{urban_desc}_{row}'] = pd.Series(
                            [0] * len(df_property.index), index=df_property.index
                        )
                        row += 1
                        continue

                    # Get data for this urban/rural, gender
                    try:
                        gender_data = df_property[df_key][gender]

                        # Find the True column
                        true_col = None
                        if "True" in gender_data:
                            true_col = gender_data["True"]
                        elif True in gender_data:
                            true_col = gender_data[True]
                        elif 'True' in gender_data:
                            true_col = gender_data['True']

                        if true_col is not None:
                            total = gender_data.sum(axis=1)
                            proportion = true_col.sum(axis=1) / total.replace(0, np.nan)
                            proportion = proportion.fillna(0)
                            df_dict[f'{gender}_{urban_desc}_{row}'] = proportion
                        else:
                            print(f"WARNING: No True column found for {li_property}[{df_key}][{gender}]")
                            df_dict[f'{gender}_{urban_desc}_{row}'] = pd.Series(
                                [0] * len(df_property.index), index=df_property.index
                            )
                    except Exception as e:
                        print(f"ERROR processing {li_property}[{df_key}][{gender}]: {e}")
                        df_dict[f'{gender}_{urban_desc}_{row}'] = pd.Series(
                            [0] * len(df_property.index), index=df_property.index
                        )

                    row += 1

                # Plot for this gender
                for plot_key, plot_data in df_dict.items():
                    plot_row = int(plot_key.split('_')[-1])
                    ax = plot_data.plot(kind='bar', stacked=True,
                                        ax=axes[plot_row, col],
                                        ylim=(0, y_lim),
                                        legend=None,
                                        color='darkturquoise',
                                        title=f"{plot_key.split('_')[1]} {desc} {self.en_props[li_property].label}",
                                        ylabel=f"{self.en_props[li_property].label} proportions",
                                        xlabel="Year"
                                        )
                    self.custom_axis_formatter(plot_data, ax)

                col += 1

        else:
            # Structure: [gender][property][age_range] or other
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            axes = np.array([[axes[0], axes[1]]])
            col = 0

            for gender, desc in self.gender_des.items():
                try:
                    if gender in df_property:
                        gender_data = df_property[gender]

                        # Find the True column
                        true_col = None
                        if "True" in gender_data:
                            true_col = gender_data["True"]
                        elif True in gender_data:
                            true_col = gender_data[True]
                        elif 'True' in gender_data:
                            true_col = gender_data['True']

                        if true_col is not None:
                            total = gender_data.sum(axis=1)
                            proportion = true_col.sum(axis=1) / total.replace(0, np.nan)
                            proportion = proportion.fillna(0)
                        else:
                            # If no True column, check if property values are directly accessible
                            print(f"DEBUG: No True column for {li_property}[{gender}]. Checking structure...")
                            print(f"  Columns: {list(gender_data.columns)[:5]}...")
                            # For properties like li_urban where the property itself is the value
                            # We might need to handle this differently
                            proportion = pd.Series([0] * len(df_property.index), index=df_property.index)
                    else:
                        print(f"WARNING: Gender {gender} not found in {li_property}")
                        proportion = pd.Series([0] * len(df_property.index), index=df_property.index)

                except Exception as e:
                    print(f"ERROR processing {li_property} for {gender}: {e}")
                    proportion = pd.Series([0] * len(df_property.index), index=df_property.index)

                ax = proportion.plot(kind='bar', stacked=True,
                                     ax=axes[0, col],
                                     ylim=(0, y_lim),
                                     legend=None,
                                     color='darkturquoise',
                                     title=f"{desc} {self.en_props[li_property].label}",
                                     ylabel=f"{self.en_props[li_property].label} proportions",
                                     xlabel="Year"
                                     )
                self.custom_axis_formatter(proportion, ax)
                col += 1

        fig.legend([self.en_props[li_property].label], loc='lower left', bbox_to_anchor=(0.75, 0.8))
        add_footnote(fig, f'{self.en_props[li_property].per_gender_footnote}')
        fig.tight_layout()
        plt.savefig(self.outputpath / (li_property + self.datestamp + '.png'), format='png')
        plt.close(fig=fig)

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

    # def plot_non_categorical_properties_by_age_group(self, li_property):
    #     """ plot all non-categorical properties by age group """
    #     # select logs from the latest year. In this case we are selecting year 2021
    #     y_lim: float = 1.0
    #     if li_property in ['li_is_sexworker']:
    #         y_lim = 0.040
    #
    #     all_logs_df = self.dfs[li_property]
    #     mask = (all_logs_df.index > pd.to_datetime('2021-01-01')) & (all_logs_df.index <= pd.to_datetime('2022-01-01'))
    #     self.dfs[li_property] = self.dfs[li_property].loc[mask]
    #
    #     # create subplots
    #     fig, axes = plt.subplots(nrows=2 if li_property in self.cat_by_rural_urban_props or li_property ==
    #                                         'li_in_ed' else 1,
    #                                         figsize=(10, 5), sharex=True)
    #
    #     df_dict = dict()
    #     if li_property == 'li_in_ed' or li_property in self.cat_by_rural_urban_props:
    #         _col: int = 0  # column counter
    #         key_value_desc = self.wealth_desc.items() if li_property == 'li_in_ed' else \
    #             self._rural_urban_state.items()
    #         for _key, _value in key_value_desc:
    #             temp_df = pd.DataFrame()
    #             for _bool_value in ['True', 'False']:
    #                 if li_property == 'li_in_ed':
    #                     temp_df[_bool_value] = self.dfs[li_property]['M'][_key][_bool_value].sum(axis=0) + \
    #                                            self.dfs[li_property]['F'][_key][_bool_value].sum(axis=0)
    #
    #                 else:
    #                     temp_df[_bool_value] = self.dfs[li_property][_key]['M'][_bool_value].sum(axis=0) + \
    #                                 self.dfs[li_property][_key]['F'][_bool_value].sum(axis=0)
    #
    #             df_dict[f'{_value}_{_col}'] = temp_df['True'] / temp_df.sum(axis=1)
    #             _col += 1
    #
    #     else:
    #         plot_df = pd.DataFrame()
    #         for _bool_value in ['True', 'False']:
    #             plot_df[_bool_value] = self.dfs[li_property]['M'][_bool_value].sum(axis=0) + \
    #                               self.dfs[li_property]['F'][_bool_value].sum(axis=0)
    #
    #         df_dict['non_urban_1'] = plot_df['True'] / plot_df.sum(axis=1)
    #
    #     for _key in df_dict.keys():
    #         # do plotting
    #         df_dict[_key].plot(kind='bar', stacked=True,
    #                            ax=axes[int(_key.split("_")[-1])] if
    #                            li_property in self.cat_by_rural_urban_props or li_property == 'li_in_ed' else axes,
    #                            ylim=(0, y_lim),
    #                            legend=None,
    #                            color='darkturquoise',
    #                            title=f"{self.en_props[li_property].label} by age group in 2021, {_key.split('_')[0]}"
    #                                  if li_property in self.cat_by_rural_urban_props or li_property == 'li_in_ed' else
    #                            f"{self.en_props[li_property].label} by age group in 2021",
    #                            ylabel=f"{self.en_props[li_property].label} proportions", xlabel="Year"
    #                            )
    #
    #     fig.legend([self.en_props[li_property].label], loc='lower left', bbox_to_anchor=(0.8, 0.7))
    #     add_footnote(fig, f'{self.en_props[li_property].per_age_group_footnote}')
    #     fig.tight_layout()
    #     plt.savefig(self.outputpath / (li_property + self.datestamp + '.png'), format='png')
    #     plt.close(fig=fig)  # close figure after saving it to avoid opening multiple figures

    def plot_non_categorical_properties_by_age_group(self, li_property):
        """ plot all non-categorical properties by age group """
        # select logs from the latest year. In this case we are selecting year 2021
        y_lim: float = 1.0
        if li_property in ['li_is_sexworker']:
            y_lim = 0.040

        all_logs_df = self.dfs[li_property]
        mask = (all_logs_df.index > pd.to_datetime('2021-01-01')) & (all_logs_df.index <= pd.to_datetime('2022-01-01'))
        self.dfs[li_property] = self.dfs[li_property].loc[mask]

        # Check the actual structure
        df_property = self.dfs[li_property]
        first_level_values = df_property.columns.get_level_values(0).unique()

        # Determine if it has urban/rural structure
        has_urban_rural_structure = False
        for val in first_level_values:
            if str(val).lower() in ['true', 'false'] or val in [True, False]:
                has_urban_rural_structure = True
                break

        # Initialize df_dict
        df_dict = {}

        # Special handling for li_in_ed
        if li_property == 'li_in_ed':
            # li_in_ed has special structure [gender][li_wealth][li_in_ed][age_years]
            fig, axes = plt.subplots(nrows=2, figsize=(10, 5), sharex=True)

            _col = 0
            for wealth_key, wealth_desc in self.wealth_desc.items():
                temp_df = pd.DataFrame()
                for _bool_value in ['True', 'False']:
                    try:
                        m_data = df_property['M'][wealth_key][_bool_value].sum(axis=0)
                        f_data = df_property['F'][wealth_key][_bool_value].sum(axis=0)
                        temp_df[_bool_value] = m_data + f_data
                    except KeyError:
                        # Try with boolean True/False
                        bool_val = True if _bool_value == 'True' else False
                        m_data = df_property['M'][wealth_key][bool_val].sum(axis=0)
                        f_data = df_property['F'][wealth_key][bool_val].sum(axis=0)
                        temp_df[_bool_value] = m_data + f_data

                df_dict[f'{wealth_desc}_{_col}'] = temp_df['True'] / temp_df.sum(axis=1).replace(0, np.nan)
                _col += 1

        elif has_urban_rural_structure and li_property in self.cat_by_rural_urban_props:
            # Has urban/rural structure
            fig, axes = plt.subplots(nrows=2, figsize=(10, 5), sharex=True)

            _col = 0
            for urban_key, urban_desc in self._rural_urban_state.items():
                # Find the matching key in the DataFrame
                df_key = None
                for val in first_level_values:
                    if str(val).lower() == str(urban_key).lower() or \
                        (urban_key == 'True' and val is True) or \
                        (urban_key == 'False' and val is False):
                        df_key = val
                        break

                if df_key is None:
                    print(f"WARNING: Could not find urban key {urban_key} for {li_property}")
                    # Create empty series with correct index (age groups)
                    age_groups = sorted(df_property.columns.get_level_values(-1).unique())
                    df_dict[f'{urban_desc}_{_col}'] = pd.Series([0] * len(age_groups), index=age_groups)
                    _col += 1
                    continue

                temp_df = pd.DataFrame()
                for _bool_value in ['True', 'False']:
                    try:
                        m_data = df_property[df_key]['M'][_bool_value].sum(axis=0)
                        f_data = df_property[df_key]['F'][_bool_value].sum(axis=0)
                        temp_df[_bool_value] = m_data + f_data
                    except KeyError:
                        # Try with boolean True/False
                        bool_val = True if _bool_value == 'True' else False
                        m_data = df_property[df_key]['M'][bool_val].sum(axis=0)
                        f_data = df_property[df_key]['F'][bool_val].sum(axis=0)
                        temp_df[_bool_value] = m_data + f_data

                proportion = temp_df['True'] / temp_df.sum(axis=1).replace(0, np.nan)
                df_dict[f'{urban_desc}_{_col}'] = proportion.fillna(0)
                _col += 1

        else:
            # No urban/rural structure or not in cat_by_rural_urban_props
            fig, axes = plt.subplots(nrows=1, figsize=(10, 5), sharex=True)

            plot_df = pd.DataFrame()
            for _bool_value in ['True', 'False']:
                try:
                    m_data = df_property['M'][_bool_value].sum(axis=0)
                    f_data = df_property['F'][_bool_value].sum(axis=0)
                    plot_df[_bool_value] = m_data + f_data
                except KeyError:
                    # Try with boolean True/False
                    bool_val = True if _bool_value == 'True' else False
                    m_data = df_property['M'][bool_val].sum(axis=0)
                    f_data = df_property['F'][bool_val].sum(axis=0)
                    plot_df[_bool_value] = m_data + f_data

            proportion = plot_df['True'] / plot_df.sum(axis=1).replace(0, np.nan)
            df_dict['all'] = proportion.fillna(0)

        # Plotting - Check if we have data to plot
        if not df_dict:
            print(f"WARNING: No data to plot for {li_property}")
            plt.close(fig=fig)
            return

        # Convert axes to array if needed for consistent indexing
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        # Plot each item in df_dict
        for i, (_key, plot_data) in enumerate(df_dict.items()):
            # Determine which axis to use
            if len(df_dict) > 1 and len(axes) > 1:
                ax = axes[i]
            else:
                ax = axes[0] if isinstance(axes, np.ndarray) else axes

            # Ensure plot_data is a Series with proper index
            if isinstance(plot_data, pd.Series):
                # Sort by index if it's numeric-like
                try:
                    plot_data = plot_data.sort_index(key=lambda x: pd.to_numeric(x, errors='ignore'))
                except:
                    pass
            else:
                # Convert to Series if it's not
                plot_data = pd.Series(plot_data)

            plot_data.plot(kind='bar',
                           ax=ax,
                           ylim=(0, y_lim),
                           legend=None,
                           color='darkturquoise',
                           title=f"{self.en_props[li_property].label} by age group in 2021, {_key.split('_')[0]}"
                           if len(df_dict) > 1 else
                           f"{self.en_props[li_property].label} by age group in 2021",
                           ylabel=f"{self.en_props[li_property].label} proportions",
                           xlabel="Age Group"
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
    end_date = Date(2015, 1, 1)
    pop_size = 5000

    # Path to the resource files used by the disease and intervention methods
    resourcefilepath = './resources'

    # This creates the Simulation instance for this run. Because we"ve passed the `seed` and
    # `log_config` arguments, these will override the default behaviour.
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config, resourcefilepath=resourcefilepath)

    # We register all modules in a single call to the register method, calling once with multiple
    # objects. This is preferred to registering each module in multiple calls because we will be
    # able to handle dependencies if modules are registered together
    sim.register(
        demography.Demography(),
        enhanced_lifestyle.Lifestyle(),
        simplified_births.SimplifiedBirths()

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
