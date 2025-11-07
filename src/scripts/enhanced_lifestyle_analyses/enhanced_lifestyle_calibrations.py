"""
This is lifestyle calibration. it plots lifestyle properties against their observed data.
"""
# %% Import Statements
import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file, unflatten_flattened_multi_index_in_logging
from tlo.methods import demography, enhanced_lifestyle, simplified_births


def add_footnote(footnote: str):
    """ a function that adds a footnote below property plots

    :param footnote: any plot footnote description """
    plt.figtext(0.5, 0.01, footnote, ha="center", fontsize=10,
                bbox={"facecolor": "gray", "alpha": 0.3, "pad": 5})


class LifeStyleCalibration:
    """ a class for calibrating lifestyle properties """

    def __init__(self, logs=None, path: str = None):
        """ initialise variables

        :param logs: all lifestyle logs
        :param path: a path to outputs folder

        """
        # create a dictionary for lifestyle properties and their descriptions to be used as plot descriptors.
        self.en_props = {'li_urban': 'currently urban', 'li_wealth': 'wealth level',
                         'li_low_ex': 'currently low exercise', 'li_tob': 'current using tobacco',
                         'li_ex_alc': 'current excess alcohol', 'li_mar_stat': 'marital status',
                         'li_in_ed': 'currently in education', 'li_ed_lev': 'education level',
                         'li_unimproved_sanitation': 'unimproved sanitation',
                         'li_no_clean_drinking_water': 'no clean drinking water',
                         'li_wood_burn_stove': 'wood burn stove', 'li_no_access_handwashing': ' no access hand washing',
                         'li_high_salt': 'high salt', 'li_high_sugar': 'high sugar', 'li_bmi': 'bmi',
                         'li_is_circ': 'Male circumcision', 'li_is_sexworker': 'sex workers'
                         }

        # ------------------------- OBSERVED DATA PROPORTIONS--------------------------------
        # 1. Lifestyle properties that are logged not by either rural urban or gender
        self.obs_data_prop = {
            'li_urban': {
                'ob_year': 2018,
                'source': '2018 Population census',
                'data': 0.16
            },
            'li_no_access_handwashing': {
                'ob_year': 2015,
                'source': 'Malawi DHS 2015',
                'data': 0.58
            },
            'li_high_salt': {
                'ob_year': 2018,
                'source': 'Price et al 2018; weighting by urban rural',
                'data': 0.27
            },
            'li_high_sugar': {
                'ob_year': 2018,
                'source': 'Price et al 2018; 6 or more sugary drinks per day',
                'data': 0.37
            },
            'li_is_sexworker': {
                'ob_year': 2012,
                'source': 'UNAIDS',
                'data': 0.006
            },
            # currently no data source for individuals in education
        }
        # 2. Lifestyle properties that are logged by rural urban
        self.obs_data_prop_rural_urban = {
            'li_unimproved_sanitation': {
                'ob_year': 2015,
                'source': 'Malawi DHS 2015',
                'data': [0.04, 0.19]
            },
            'li_no_clean_drinking_water': {
                'ob_year': 2015,
                'source': 'Malawi DHS 2015',
                'data': [0.02, 0.15]
            },
            'li_wood_burn_stove': {
                'ob_year': 2015,
                'source': 'Malawi DHS 2015',
                'data': [0.26, 0.94]
            },
            'li_ex_alc': {
                'ob_year': 2014,
                'source': 'WHO 2014',
                'data': [0.15, 0.01]
            }
        }
        # 3. Lifestyle properties that are logged by rural urban and gender
        self.obs_data_prop_rural_urban_cat = {
            'li_wealth': {
                'ob_year': 2015,
                'source': 'Malawi DHS 2015',
                'data': {
                    'urban': [0.75, 0.16, 0.05, 0.02, 0.02],
                    'rural': [0.11, 0.21, 0.22, 0.23, 0.23]
                }
            },

            'li_low_ex': {
                'ob_year': 2011,
                'source': 'Msyamboza et al; 2011; WHO STEPS',
                'data': {
                    'urban': [0.32, 0.18],
                    'rural': [0.11, 0.07]
                }
            },

            'li_bmi': {
                'ob_year': 2018,
                'source': 'Price et al; median age ~ 30',
                'data': {
                    'urban': {
                        'males': [0.04, 0.77, 0.15, 0.04, 0.00],
                        'females': [0.02, 0.53, 0.26, 0.18, 0.01]
                    },
                    'rural': {
                        'males': [0.06, 0.85, 0.08, 0.01, 0.0],
                        'females': [0.05, 0.68, 0.20, 0.08, 0.0]
                    }
                }
            }
        }

        # 4. Lifestyle properties that are logged by gender
        self.other_props = {
            'li_mar_stat': {
                'ob_year': 2015,
                'source': 'Malawi DHS 2015',
                'data': {
                    'males': [0.402, 0.565, 0.033],
                    'females': [0.21, 0.657, 0.133]
                }
            },
            'li_tob': {
                'ob_year': 2015,
                'source': 'Malawi DHS 2015',
                'data': {
                    'males': 0.12,
                    'females': 0.06
                },
            }
        }

        # A dictionary that includes categories description for all categorical properties
        self.categories_desc: dict = {
            'li_bmi': ["bmi category 1", "bmi category 2", "bmi category 3", "bmi category 4", "bmi category 5"],
            'li_wealth': ['wealth level 1', 'wealth level 2', 'wealth level 3', 'wealth level 4', 'wealth level 5'],
            'li_mar_stat': ['Never Married', 'Married', 'Divorced or Widowed'],
            'li_ed_lev': ['Not in education', 'Primary edu', 'secondary education']
        }

        # A dictionary to hold individual's rural or urban status
        self._rural_urban_state = {
            'True': 'urban',
            'False': 'rural'
        }

        # date-stamp to label log files and any other outputs
        self.datestamp: str = datetime.date.today().strftime("__%Y_%m_%d")

        # a dictionary for gender descriptions. to be used when plotting
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

    def custom_axis_formatter(self, df: pd.DataFrame, ax, li_property: str):
        """
        create a custom date formatter since the default pandas date formatter works well with line graphs. see an
        accepted solution to issue https://stackoverflow.com/questions/30133280/pandas-bar-plot-changes-date-format

        :param df: pandas dataframe or series
        :param ax: matplotlib AxesSubplot object
        :param li_property: one of the lifestyle properties
        """
        # make the tick labels empty so the labels don't get too crowded
        tick_labels = [''] * len(df.index)
        # Every 12th tick label includes the year
        tick_labels[::12] = [item.strftime('%Y') for item in df.index[::12]]
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(tick_labels))
        ax.legend(self.categories_desc[li_property] if li_property in self.categories_desc.keys()
                  else [self.en_props[li_property]], loc='upper center')
        plt.gcf().autofmt_xdate()

    def plot_no_cat_gender_age_group_properties(self, li_property: str):
        """ A function to plot all lifestyle properties that are not grouped by categories, gender, age group or
        urban / rural status

         :param li_property: one of the lifestyle properties """

        # a dataframe that will hold plotting data
        total_per_prop = pd.DataFrame()
        # for in education property
        if li_property == 'li_in_ed':
            re_ord_col_df = self.dfs[li_property].reorder_levels(['sex', 'li_in_ed', 'age_years', 'li_wealth'], axis=1)
            for age_years in ['6', '12', '16', '19']:
                pos_to_prop = re_ord_col_df['M']['True'][age_years] + \
                              re_ord_col_df['F']['True'][age_years]

                neg_to_prop = re_ord_col_df['M']['False'][age_years] + re_ord_col_df['F']['False'][age_years]

                pos_to_prop = pos_to_prop.sum(axis=1)
                neg_to_prop = neg_to_prop.sum(axis=1)
                # get population positive to the property
                total_per_prop[f'age_years_{age_years}'] = pos_to_prop / (pos_to_prop + neg_to_prop)
            # get year(to be set as an index and used when plotting)
            total_per_prop['period'] = self.dfs[li_property].index.year
            total_per_prop = total_per_prop.groupby(by='period').mean()

        elif li_property == 'li_tob':
            # get population positive to the property
            total_per_prop['model_output'] = self.dfs[li_property]['M']['True'].sum(axis=1)
        else:
            # get population positive to the property
            total_per_prop['model_output'] = self.dfs[li_property]['M']['True'].sum(axis=1) \
                                             + self.dfs[li_property]['F']['True'].sum(axis=1)
        # plotting
        if li_property == 'li_in_ed':
            _row_counter: int = 0
            _column_counter: int = 0
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
            for age_years in ['6', '12', '16', '19']:
                # create observed data column for comparison with the model runs
                total_per_prop['observed_data'] = np.nan

                # set observed data value
                total_per_prop.loc[
                    (total_per_prop.index == self.obs_data_prop[li_property]['ob_year']), 'observed_data'] = \
                    self.obs_data_prop[li_property]['data'][age_years]

                # do plotting
                ax = total_per_prop.plot.line(y=[f'age_years_{age_years}'], ylabel=f'{self.en_props[li_property]}'
                                                                                   f' proportions', ylim=[0, 1],
                                              ax=axes[_row_counter, _column_counter],
                                              title=f'Individuals aged {age_years} years {self.en_props[li_property]}')
                total_per_prop.plot.line(
                    y=['observed_data'],
                    marker='^',
                    color='red',
                    ax=ax
                )
                # plt.ylim(0, 1)
                ax.legend(['model_output', 'observed_data'])
                if _row_counter < 1:
                    _row_counter += 1
                else:
                    _row_counter = 0
                    if _column_counter < 1:
                        _column_counter += 1

            add_footnote(f"Data source: {self.obs_data_prop[li_property]['source']}")
            plt.tight_layout()
            plt.savefig(self.outputpath / (li_property + self.datestamp + 'fig_.png'), format='png')
            plt.show()

        else:
            y_lim = 1.0

            # get totals male and females per each property
            total_per_prop['total_pop'] = self.dfs[li_property].sum(axis=1)

            if li_property == 'li_is_sexworker':
                y_lim = 0.01
                # get totals male and females per each property
                total_per_prop['total_pop'] = self.dfs[li_property]['F'].sum(axis=1)
            # get year(to be set as an index and used when plotting)
            total_per_prop['period'] = self.dfs[li_property].index.year
            total_per_prop = total_per_prop.groupby(by='period').sum()
            # create observed data column for comparison with the model runs
            total_per_prop['observed_data'] = np.nan
            # set observed data value
            total_per_prop.loc[(total_per_prop.index == self.obs_data_prop[li_property]['ob_year']), 'observed_data']\
                = self.obs_data_prop[li_property]['data'] * \
                total_per_prop.loc[(total_per_prop.index == self.obs_data_prop[li_property]['ob_year']), 'total_pop']
            # convert absolute numbers to proportion
            total_per_prop = total_per_prop.apply(lambda row: row / total_per_prop.total_pop)
            # do plotting
            ax = total_per_prop.plot.line(y=['model_output'], ylabel=f'{self.en_props[li_property]} proportions',
                                          title=f'{self.en_props[li_property]} ')
            total_per_prop.plot.line(
                y=['observed_data'],
                marker='^',
                color='red',
                ax=ax
            )
            plt.ylim(0, y_lim)
            add_footnote(f"Data source: {self.obs_data_prop[li_property]['source']}")
            plt.tight_layout()
            plt.savefig(self.outputpath / (li_property + self.datestamp + 'fig_.png'), format='png')
            plt.show()

    def plot_properties_by_urban_rural(self, li_property: str):
        """ A function to plot all lifestyle properties that are grouped by rural urban

         :param li_property: one of the lifestyle properties """
        ylim = 1.0  # max y-limit
        _rows_counter: int = 0  # a counter to set the number of rows when plotting
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))  # configure plot
        for _value in self._rural_urban_state.keys() if not li_property == 'li_ex_alc' else self.gender_des.keys():
            # a dataframe that will hold plotting data
            total_per_prop = pd.DataFrame()

            if li_property == 'li_ex_alc':
                # get totals male and females
                total_per_prop['total_pop'] = self.dfs[li_property]['True'][_value].sum(axis=1) \
                                              + self.dfs[li_property]['False'][_value].sum(axis=1)

                # get population positive to excess alcohol
                total_per_prop['model_output'] = self.dfs[li_property]['True'][_value]['True'].sum(
                    axis=1) + self.dfs[li_property]['False'][_value]['True'].sum(axis=1)
            else:
                # get totals male and females per each property
                total_per_prop['total_pop'] = self.dfs[li_property][_value]['M'].sum(axis=1) \
                                              + self.dfs[li_property][_value]['F'].sum(axis=1)

                # get population positive to the property
                total_per_prop['model_output'] = self.dfs[li_property][_value]['M']['True'].sum(
                    axis=1) + self.dfs[li_property][_value]['F']['True'].sum(axis=1)

            # get years(to be set as an index and used when plotting)
            total_per_prop['period'] = self.dfs[li_property].index.year
            total_per_prop = total_per_prop.groupby(by='period').sum()

            total_per_prop = total_per_prop.apply(lambda row: row / total_per_prop.total_pop)
            # create observed data column for comparison with the model
            total_per_prop['observed_data'] = np.nan

            # set observed data value
            total_per_prop.loc[(total_per_prop.index == self.obs_data_prop_rural_urban[li_property]['ob_year']),
                               'observed_data'] = self.obs_data_prop_rural_urban[li_property]['data'][_rows_counter]

            total_per_prop.drop('total_pop', inplace=True, axis=1)

            # do plotting
            if not li_property == 'li_ex_alc':
                ax = total_per_prop.plot.line(y=['model_output'], ax=axes[_rows_counter], ylim=(0, ylim),
                                              ylabel=f'{self.en_props[li_property]} proportions',
                                              title=f'{self.en_props[li_property]}'
                                                    f' {self._rural_urban_state[_value]}')
            else:
                ax = total_per_prop.plot.line(y=['model_output'], ax=axes[_rows_counter], ylim=(0, ylim),
                                              ylabel=f'{self.en_props[li_property]} proportions',
                                              title=f'{self.en_props[li_property]}'
                                                    f' {self.gender_des[_value]}')
            total_per_prop.plot.line(
                y=['observed_data'],
                marker='^',
                color='red',
                ax=ax
            )

            _rows_counter += 1
        add_footnote(f"Data source: {self.obs_data_prop_rural_urban[li_property]['source']}")
        plt.tight_layout()
        plt.savefig(self.outputpath / (li_property + self.datestamp + 'fig_.png'), format='png')
        plt.show()

    def plot_properties_by_urban_rural_cat(self, li_property: str, categories: list = None):
        """ A function to plot all Lifestyle properties grouped by rural, urban and categories

         :param li_property: one of the lifestyle properties
         :param categories: a list containing categories """

        # create a dictionary that will hold proportions for individuals in both rural and urban
        _prop_urb_rural = dict()
        ylim = 1.0

        # 1. create model proportions
        for urban_rural in self._rural_urban_state.keys():
            _probs = dict()
            # get years(to be set as an index and used when plotting)
            sim_period = self.dfs[li_property].index.year
            # for bmi
            if li_property == 'li_bmi':
                for gender, gender_desc in self.gender_des.items():
                    total_per_prop = pd.DataFrame()
                    for _category in categories:
                        total_per_prop[f'cat_{_category}'] = self.dfs[li_property][urban_rural][gender][_category].sum(
                            axis=1) + self.dfs[li_property][urban_rural][gender][_category].sum(axis=1)

                    # prepare data for plotting
                    total_per_prop['period'] = sim_period
                    total_per_prop = total_per_prop.groupby(by='period').sum()
                    total_per_prop = total_per_prop.apply(lambda row: row / row.sum(), axis=1)
                    _probs[gender_desc] = total_per_prop
            # for low exercise
            elif li_property == 'li_low_ex':
                total_per_prop = pd.DataFrame()
                for gender, gender_desc in self.gender_des.items():
                    total_per_prop[gender_desc] = self.dfs[li_property][urban_rural][gender]['True'].sum(
                        axis=1)
                    total_per_prop['total'] = self.dfs[li_property][urban_rural][gender].sum(axis=1)

                # prepare data for plotting
                total_per_prop['period'] = sim_period
                total_per_prop = total_per_prop.groupby(by='period').sum()
                total_per_prop = total_per_prop.apply(lambda row: row / total_per_prop.total)
                total_per_prop.drop('total', axis=1, inplace=True)
                _probs = total_per_prop
            # For the remaining properties
            else:
                total_per_prop = pd.DataFrame()
                for _category in categories:
                    total_per_prop[f'cat_{_category}'] = self.dfs[li_property][urban_rural]['M'][_category].sum(
                        axis=1) + self.dfs[li_property][urban_rural]['F'][_category].sum(axis=1)

                # prepare data for plotting
                total_per_prop['period'] = sim_period
                total_per_prop = total_per_prop.groupby(by='period').sum()
                total_per_prop = total_per_prop.apply(lambda row: row / row.sum(), axis=1)
                _probs = total_per_prop

            _prop_urb_rural[urban_rural] = _probs

        # 2. do plotting
        # plot bmi
        if li_property == 'li_bmi':
            for _category in categories:
                _column_counter: int = 0
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
                for urban_rural in self._rural_urban_state.keys():
                    _rows_counter: int = 0
                    for gender_desc in self.gender_des.values():
                        _prop_urb_rural[urban_rural][gender_desc]['observed_data'] = np.nan
                        # set observed data value
                        _prop_urb_rural[urban_rural][gender_desc].loc[
                            (_prop_urb_rural[urban_rural][gender_desc].index ==
                             self.obs_data_prop_rural_urban_cat[li_property]['ob_year']), 'observed_data'] = \
                            self.obs_data_prop_rural_urban_cat[li_property]['data'][
                                self._rural_urban_state[urban_rural]][str.lower(gender_desc)][
                                int(_category) - 1]
                        # do plotting
                        ax = _prop_urb_rural[urban_rural][
                            gender_desc].plot.line(y=[f'cat_{_category}'],
                                                   ylabel=f'{self.en_props[li_property]} proportions',
                                                   ax=axes[_column_counter, _rows_counter], ylim=(0, ylim),
                                                   title=f'{self.en_props[li_property]} category {_category} '
                                                         f'{str.lower(gender_desc)}'
                                                         f' {self._rural_urban_state[urban_rural]}')

                        _prop_urb_rural[urban_rural][gender_desc].plot.line(
                            y=['observed_data'],
                            marker='^',
                            color='red',
                            ax=ax
                        )
                        ax.legend(['model_output', 'observed_data'])
                        _rows_counter += 1
                    _column_counter += 1
                add_footnote(f"Data source: {self.obs_data_prop_rural_urban_cat[li_property]['source']}")
                plt.tight_layout()
                plt.savefig(self.outputpath / (li_property + self.datestamp + _category + 'fig_.png'), format='png')
                plt.show()
        # plot low exercise
        elif li_property == 'li_low_ex':
            _column_counter: int = 0
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
            for urban_rural in self._rural_urban_state.keys():
                _rows_counter: int = 0
                for gender in self.gender_des.values():
                    # set observed data value
                    _prop_urb_rural[urban_rural].loc[(_prop_urb_rural[urban_rural].index ==
                                                      self.obs_data_prop_rural_urban_cat[li_property]['ob_year']),
                                                     'observed_data'] = \
                        self.obs_data_prop_rural_urban_cat[li_property]['data'][self._rural_urban_state[urban_rural]][
                            _rows_counter]

                    # do plotting
                    ax = _prop_urb_rural[urban_rural].plot.line(y=[gender],
                                                                ylabel=f'{self.en_props[li_property]} proportions',
                                                                ax=axes[_column_counter, _rows_counter], ylim=(0, ylim),
                                                                title=f'{self.en_props[li_property]} {gender}'
                                                                      f' {self._rural_urban_state[urban_rural]}')

                    _prop_urb_rural[urban_rural].plot.line(
                        y=['observed_data'],
                        marker='^',
                        color='red',
                        ax=ax
                    )
                    ax.legend(['model_output', 'observed_data'])
                    _rows_counter += 1
                _column_counter += 1
            add_footnote(f"Data source: {self.obs_data_prop_rural_urban_cat[li_property]['source']}")
            plt.tight_layout()
            plt.savefig(self.outputpath / (li_property + self.datestamp + 'fig_.png'), format='png')
            plt.show()

        # plot the remaining properties
        else:
            for _category in categories:
                _rows_counter: int = 0
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
                for urban_rural in self._rural_urban_state.keys():
                    # set observed data value
                    _prop_urb_rural[urban_rural].loc[(_prop_urb_rural[urban_rural].index ==
                                                      self.obs_data_prop_rural_urban_cat[li_property]['ob_year']),
                                                     'observed_data'] = self.obs_data_prop_rural_urban_cat[li_property][
                        'data'][self._rural_urban_state[urban_rural]][int(_category) - 1]

                    # do plotting
                    ax = _prop_urb_rural[urban_rural].plot.line(y=[f'cat_{_category}'],
                                                                ylabel=f'{self.en_props[li_property]} proportions',
                                                                ax=axes[_rows_counter], ylim=(0, ylim),
                                                                title=f'{self.en_props[li_property]} {_category}'
                                                                      f' {self._rural_urban_state[urban_rural]}')

                    _prop_urb_rural[urban_rural].plot.line(
                        y=['observed_data'],
                        marker='^',
                        color='red',
                        ax=ax
                    )
                    ax.legend(['model_output', 'observed_data'])
                    _rows_counter += 1
                add_footnote(f"Data source: {self.obs_data_prop_rural_urban_cat[li_property]['source']}")
                plt.tight_layout()
                plt.savefig(self.outputpath / (li_property + self.datestamp + _category + 'fig_.png'), format='png')
                plt.show()

    def plot_properties_by_gender(self, li_property: str, categories: list = None):
        """ A function to plot Lifestyle properties that are grouped by gender

         :param li_property: one of the lifestyle properties
         :param categories: a list containing categories """
        ylim = 1.0
        _props = dict()  # to hold male female proportions
        # 1. create model proportions
        for gender, gender_desc in self.gender_des.items():
            # get years(to be set as an index and used when plotting)
            sim_period = self.dfs[li_property].index.year
            total_per_prop = pd.DataFrame()
            # get marital status proportions
            if li_property == 'li_mar_stat':
                for _category in categories:
                    total_per_prop[f'cat_{_category}'] = self.dfs[li_property][gender][_category].sum(
                        axis=1) + self.dfs[li_property][gender][_category].sum(axis=1)
            # get tobacco proportions
            else:
                total_per_prop[gender_desc] = self.dfs[li_property][gender]['True'].sum(axis=1)
                total_per_prop['total'] = self.dfs[li_property][gender].sum(axis=1)

            # prepare data for plotting
            total_per_prop['period'] = sim_period
            total_per_prop = total_per_prop.groupby(by='period').sum()
            total_per_prop = total_per_prop.apply(lambda row: row / row.sum(), axis=1)
            _props[gender_desc] = total_per_prop

        # 2. do plotting
        # plot marital status
        if li_property == 'li_mar_stat':
            for _category in categories:
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
                _rows_counter: int = 0
                for gender_desc in self.gender_des.values():
                    _props[gender_desc]['observed_data'] = np.nan
                    # set observed data value
                    _props[gender_desc].loc[
                        (_props[gender_desc].index ==
                         self.other_props[li_property]['ob_year']), 'observed_data'] = \
                        self.other_props[li_property]['data'][str.lower(gender_desc)][int(_category) - 1]
                    # do plotting
                    ax = _props[gender_desc].plot.line(y=[f'cat_{_category}'],
                                                       ylabel=f'{self.en_props[li_property]} proportions',
                                                       ax=axes[_rows_counter], ylim=(0, ylim),
                                                       title=f' {gender_desc} '
                                                             f'{self.categories_desc[li_property][int(_category) - 1]}')

                    _props[gender_desc].plot.line(y=['observed_data'], marker='^', color='red', ax=ax)
                    ax.legend(['model_output', 'observed_data'])
                    _rows_counter += 1
                add_footnote(f"Data source: {self.other_props[li_property]['source']}")
                plt.tight_layout()
                plt.savefig(self.outputpath / (li_property + self.datestamp + _category + 'fig_.png'), format='png')
                plt.show()
        # plot tobacco
        else:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            _rows_counter: int = 0
            for gender_desc in self.gender_des.values():
                _props[gender_desc]['observed_data'] = np.nan
                # set observed data value
                _props[gender_desc].loc[
                    (_props[gender_desc].index ==
                     self.other_props[li_property]['ob_year']), 'observed_data'] = \
                    self.other_props[li_property]['data'][str.lower(gender_desc)]
                # do plotting
                ax = _props[gender_desc].plot.line(y=[gender_desc],
                                                   ylabel=f'{self.en_props[li_property]} proportions',
                                                   ax=axes[_rows_counter], ylim=(0, ylim),
                                                   title=f' {gender_desc} '
                                                         f'{self.en_props[li_property]}')

                _props[gender_desc].plot.line(y=['observed_data'], marker='^', color='red', ax=ax)
                ax.legend(['model_output', 'observed_data'])
                _rows_counter += 1
            add_footnote(f"Data source: {self.other_props[li_property]['source']}")
            plt.tight_layout()
            plt.savefig(self.outputpath / (li_property + self.datestamp + 'fig_.png'), format='png')
            plt.show()

    def display_all_properties_plots(self):
        """ A function to calibrate all lifestyle properties. Here we are looping through a dictionary that contains all
        lifestyle properties and call other functions based on which observed data dictionary the property belongs """
        # loop through all properties dictionary
        for prop in self.en_props:
            # for all properties that are not grouped by rural, urban, gender and have no categories
            if prop in self.obs_data_prop:
                self.plot_no_cat_gender_age_group_properties(prop)
            # for all properties that are grouped by rural, urban status
            elif prop in self.obs_data_prop_rural_urban.keys():
                self.plot_properties_by_urban_rural(prop)
            # for all properties that are grouped by rural, urban, gender and have categories
            elif prop in self.obs_data_prop_rural_urban_cat.keys():
                if prop == 'li_low_ex':
                    self.plot_properties_by_urban_rural_cat(prop)
                else:
                    self.plot_properties_by_urban_rural_cat(prop, ['1', '2', '3', '4', '5'])
            # for all properties that are grouped by gender
            elif prop in self.other_props:
                if prop == 'li_mar_stat':
                    self.plot_properties_by_gender(prop, ['1', '2', '3'])
                else:
                    self.plot_properties_by_gender(prop)
            else:
                pass


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


# uncomment the line below if you want to catch all warnings
# pd.set_option('mode.chained_assignment', 'raise')
# %% Run the Simulation
sim = run()

# %% read the results
output = parse_log_file(sim.log_filepath)
# output = parse_log_file(Path("./outputs/enhanced_lifestyle__2023-01-26T091835.log"))

# construct a dict of dataframes using lifestyle logs
logs_df = output['tlo.methods.enhanced_lifestyle']

# initialise LifeStyleCalibration class
g_plots = LifeStyleCalibration(logs=logs_df, path="./outputs")

# calibrate lifestyle properties
g_plots.display_all_properties_plots()
