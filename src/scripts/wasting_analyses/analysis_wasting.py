"""
An analysis file for the wasting module (so far only for 1 run, 1 draw)
"""
# %% Import statements
import glob
import gzip
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PyPDF2 import PdfReader, PdfWriter

from tlo.analysis.utils import compare_number_of_deaths, get_scenario_outputs, parse_log_file

# start time of the whole analysis
total_time_start = time.time()

# ####### TO SET #######################################################################################################
scenario_filename = 'wasting_analysis__minimal_model'
outputs_path = Path("./outputs/sejjej5@ucl.ac.uk/wasting")
########################################################################################################################


class WastingAnalyses:
    """
    This class looks at plotting all important outputs from the wasting module
    """

    def __init__(self, in_sim_results_folder_path_draw_x_run_0, in_datestamp):
        self.outcomes_path_name = in_sim_results_folder_path_draw_x_run_0
        self.datestamp = in_datestamp
        sim_results_file_name_prefix = scenario_filename
        sim_results_file_name_extension = '.log.gz'
        gz_results_file_path = \
            Path(glob.glob(os.path.join(sim_results_folder_path_draw_x_run_0,
                                        f"{sim_results_file_name_prefix}*{sim_results_file_name_extension}"))[0])

        # Path to the decompressed .log file
        self.__log_file_path = gz_results_file_path.with_suffix('')
        # Decompress the .log.gz file
        with gzip.open(gz_results_file_path, 'rb') as f_in:
            with open(self.__log_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # parse wasting logs
        self.__w_logs_dict = parse_log_file(self.__log_file_path)['tlo.methods.wasting']
        # TODO: Why it prints the messages from parse_log_file() twice?
        # parse scaling factor log
        # self.__scaling_factor = \
        #     parse_log_file(self.__log_file_path)['tlo.methods.population']['scaling_factor'].set_index('date').loc[
        #         '2010-01-01', 'scaling_factor'
        #     ]

        # gender description
        self.__gender_desc = {'M': 'Males',
                              'F': 'Females'}

        # wasting types description
        self.__wasting_types_desc = {'WHZ<-3': 'severe wasting',
                                     '-3<=WHZ<-2': 'moderate wasting',
                                     'WHZ>=-2': 'not undernourished'}

        self.fig_files = []

        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # # define colo(u)rs to use:
        self.__colors_model = {
            'severe wasting': cycle[0],
            'moderate wasting': cycle[1],
            'SAM': cycle[2],
            'MAM': cycle[3],
        }
        self.__colors_data = {
            'severe wasting': '#82C1EC',
            'moderate wasting': '#C71E1E',
        }

    def save_fig__store_pdf_file(self, fig, fig_output_name: str) -> None:
        fig.savefig(self.outcomes_path_name + "/" + fig_output_name + '.png', format='png')
        fig.savefig(self.outcomes_path_name + "/" + fig_output_name + '.pdf', format='pdf')
        self.fig_files.append(fig_output_name + '.pdf')

    def plot_wasting_incidence(self):
        """ plot the incidence of wasting over time """
        w_inc_df = self.__w_logs_dict['wasting_incidence_count']
        w_inc_df = w_inc_df.set_index(w_inc_df.date.dt.year)
        w_inc_df = w_inc_df.drop(columns='date')
        # check no incidence of well-nourished
        all_zeros = w_inc_df['WHZ>=-2'].apply(lambda x: all(value == 0 for value in x.values()))
        assert all(all_zeros)
        w_inc_df = w_inc_df[["WHZ<-3", "-3<=WHZ<-2"]]

        pop_sizes_df = self.__w_logs_dict['pop sizes']
        pop_sizes_df = pop_sizes_df.set_index(pop_sizes_df.date.dt.year)
        pop_sizes_df = pop_sizes_df.drop(columns='date')
        po_sizes_to_keep = [col for col in pop_sizes_df.columns if
                           col.startswith('total__') and col not in ['total__under5']]
        age_gps_total_pop_sizes_df = pop_sizes_df[po_sizes_to_keep].copy()
        age_gps_total_pop_sizes_df['0y']  = \
            age_gps_total_pop_sizes_df['total__0_5mo'] + age_gps_total_pop_sizes_df['total__6_11mo']
        age_gps_total_pop_sizes_df = age_gps_total_pop_sizes_df.drop(columns=['total__0_5mo', 'total__6_11mo'])
        age_gps_total_pop_sizes_df = age_gps_total_pop_sizes_df.rename(columns={
            'total__12_23mo': '1y',
            'total__24_35mo': '2y',
            'total__36_47mo': '3y',
            'total__48_59mo': '4y',
            'total__5y+': '5+y'
        })

        # get age_years, doesn't matter what wasting category you choose,
        # they all have same age groups
        age_years = list(w_inc_df.loc[w_inc_df.index[0], 'WHZ<-3'].keys())
        # age_years.remove('5+y')

        _row_counter = 0
        _col_counter = 0
        # plot setup
        fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(10, 6))
        axes[1, 2].axis('off')  # 5+y has no data (no new cases in 5+y), its space is used to display the label
        for age in age_years:
            plotting = pd.DataFrame()
            for state in w_inc_df.columns:
                plotting[state] = \
                    w_inc_df.apply(lambda row: row[state][age], axis=1)
            # remove sev cases from mod cases (all sev cases went through mod state)
            plotting["-3<=WHZ<-2"] = plotting["-3<=WHZ<-2"] - plotting["WHZ<-3"]
            # calculate props within the age group
            plotting = plotting.div(age_gps_total_pop_sizes_df[age], axis=0)
            plotting = plotting.rename(columns=self.__wasting_types_desc)

            ax = plotting.plot(kind='bar', stacked=True,
                               ax=axes[_row_counter, _col_counter],
                               title=f"incidence of wasting in {age} old")#,
                               #ylim=[0, 1])
            show_legend = (_row_counter == 1 and _col_counter == 2)
            # show_x_axis_label = (_row_counter == 0 and _col_counter == 2)
            if show_legend:
                ax.legend(loc='center')
                ax.set_title('')
            else:
                ax.get_legend().remove()
            # if show_x_axis_label:
            #     ax.set_xlabel('Year')  # TODO: this is not working
            ax.set_xlabel('year')
            ax.set_ylabel('proportion (within age group)')
            # move to another row
            if _col_counter == 2:
                _row_counter += 1
                _col_counter = -1
            _col_counter += 1  # increment column counter
            fig.tight_layout()
        fig_output_name = ('wasting_incidence__' + self.datestamp)
        self.save_fig__store_pdf_file(fig, fig_output_name)
        # plt.show()

    # def plot_wasting_incidence_mod_to_sev_props(self):
    #     """ plot the incidence of wasting over time """
    #     w_inc_df = self.__w_logs_dict['wasting_incidence_count']
    #     w_inc_df = w_inc_df.set_index(w_inc_df.date.dt.year)
    #     w_inc_df = w_inc_df.drop(columns='date')
    #     # check no incidence of well-nourished
    #     all_zeros = w_inc_df['WHZ>=-2'].apply(lambda x: all(value == 0 for value in x.values()))
    #     assert all(all_zeros)
    #     w_inc_df = w_inc_df[["WHZ<-3", "-3<=WHZ<-2"]]
    #     # get age_years, doesn't matter what wasting category you choose,
    #     # they all have same age groups
    #     age_years = list(w_inc_df.loc[w_inc_df.index[0], 'WHZ<-3'].keys(
    #
    #     ))
    #     age_years.remove('5+y')
    #
    #     _row_counter = 0
    #     _col_counter = 0
    #     # plot setup
    #     fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(10, 6))
    #     fig.delaxes(axes[1, 2])
    #     for _age in age_years:
    #         new_df = pd.DataFrame()
    #         for state in w_inc_df.columns:
    #             new_df[state] = \
    #                 w_inc_df.apply(lambda row: row[state][_age], axis=1)
    #         # convert into proportions
    #         new_df = new_df.apply(lambda _row: _row / _row.sum(), axis=1)
    #         plotting = new_df.rename(columns=self.__wasting_types_desc)
    #         ax = plotting.plot(kind='bar', stacked=True,
    #                            ax=axes[_row_counter, _col_counter],
    #                            title=f"incidence of wasting in {_age} old",
    #                            ylim=[0, 1])
    #         ax.legend(loc='lower right')
    #         ax.set_xlabel('year')
    #         ax.set_ylabel('proportion')
    #         # move to another row
    #         if _col_counter == 2:
    #             _row_counter += 1
    #             _col_counter = -1
    #         _col_counter += 1  # increment column counter
    #
    #     handles, labels = axes[1, 1].get_legend_handles_labels()
    #     fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 0.5))
    #     fig_output_name = ('wasting_incidence_mod_to_sev_props__' + self.datestamp)
    #     fig.tight_layout()
    #     self.save_fig__store_pdf_file(fig, fig_output_name)
    #     # plt.show()

    def plot_wasting_length(self):
        """ plot the average length of wasting over time """

        if 'wasting_length_avg' in self.__w_logs_dict:
            w_length_df = self.__w_logs_dict['wasting_length_avg']
            w_length_df = w_length_df.set_index(w_length_df.date.dt.year)
            w_length_df = w_length_df.drop(columns='date')
            # get age_years, doesn't matter from which dict
            age_years = list(w_length_df.loc[w_length_df.index[0], 'mod_MAM_tx_full_recov'].keys())
            # age_years.remove('5+y')
            w_length_df = w_length_df.loc[:, ['mod_nat_recov', 'mod_MAM_tx_full_recov', 'mod_SAM_tx_full_recov',
                                                  'mod_SAM_tx_recov_to_MAM', 'mod_not_yet_recovered',
                                                  'sev_SAM_tx_full_recov', 'sev_SAM_tx_recov_to_MAM',
                                                  'sev_not_yet_recovered']]

            for recov_opt in w_length_df.columns:
                _row_counter = 0
                _col_counter = 0
                # plot setup
                fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(10, 7))
                # axes[1, 2].axis('off')  # 5+y has no data (no new cases in 5+y), its space is used to display the label
                for _age in age_years:
                    plotting = pd.DataFrame()
                    # dict to dataframe
                    plotting[recov_opt] = \
                        w_length_df.apply(lambda row: row[recov_opt][_age], axis=1)

                    if recov_opt.startswith("mod_"):
                        colour_to_use = self.__colors_model['moderate wasting']
                        y_upper_lim = 355
                    else:
                        colour_to_use = self.__colors_model['severe wasting']
                        y_upper_lim = 1000
                    if recov_opt.endswith("not_yet_recovered"):
                        y_upper_lim = 4000

                    ax = plotting.plot(kind='bar', stacked=False,
                                       ax=axes[_row_counter, _col_counter],
                                       title=f"length of wasting in {_age} old",
                                       color=colour_to_use,
                                       ylim=[0, y_upper_lim])
                    # show_legend = (_row_counter == 0 and _col_counter == 0)
                    # # show_x_axis_label = (_row_counter == 0 and _col_counter == 2)
                    # if show_legend:
                    #     ax.legend(loc='upper right', bbox_to_anchor=(0.5, 1.2),
                    #               fancybox=True, shadow=True, ncol=5)
                    # else:
                    ax.get_legend().remove()
                    # if show_x_axis_label:
                    #     ax.set_xlabel('Year')  # TODO: this is not working
                    ax.set_xlabel('year')
                    ax.set_ylabel('avg length of wasting (days)')
                    # move to another row
                    if _col_counter == 2:
                        _row_counter += 1
                        _col_counter = -1
                    _col_counter += 1  # increment column counter

                fig.suptitle(f'{recov_opt}', fontsize=16)
                # Adjust layout to make room for the suptitle
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                fig_output_name = ('wasting_length__' + recov_opt + self.datestamp)
                self.save_fig__store_pdf_file(fig, fig_output_name)
                # plt.show(`)

    def plot_wasting_prevalence_per_year(self):
        """ plot wasting prevalence of all age groups per year. Proportions are obtained by getting a total number of
        children wasted divide by the total number of children less than 5 years"""

        w_prev_calib_data_years_only_df = pd.DataFrame({
            'sev_wast_calib': [0.015, 0.011, 0.006, 0.007],
            'mod_wast_calib': [0.025, 0.027, 0.021, 0.019]
        }, index=[2010, 2014, 2016, 2020])
        date_range = pd.Index(range(2010, 2031), name='date')
        w_prev_calib = pd.DataFrame(index=date_range)
        # filling missing values with 0
        w_prev_calib_df = w_prev_calib.merge(
            w_prev_calib_data_years_only_df, left_index=True, right_index=True, how='left'
        ).fillna(0)

        w_prev_df = self.__w_logs_dict["wasting_prevalence_props"]
        w_prev_df = w_prev_df[['date', 'total_sev_under5_prop', 'total_mod_under5_prop']]
        w_prev_df = w_prev_df.set_index(w_prev_df.date.dt.year)
        w_prev_df = w_prev_df.drop(columns='date')

        w_prev_plot_df = pd.merge(w_prev_df, w_prev_calib_df, on='date')
        columns_to_plot = [['total_sev_under5_prop', 'total_mod_under5_prop'], ['sev_wast_calib', 'mod_wast_calib']]
        colors_to_plot = {
            'total_sev_under5_prop': self.__colors_model['severe wasting'],
            'total_mod_under5_prop': self.__colors_model['moderate wasting'],
            'sev_wast_calib': self.__colors_data['severe wasting'],
            'mod_wast_calib': self.__colors_data['moderate wasting']
        }
        labels_to_plot = {
            'total_sev_under5_prop': 'severe wasting (model)',
            'total_mod_under5_prop': 'moderate wasting (model)',
            'sev_wast_calib': 'severe wasting (data)',
            'mod_wast_calib': 'moderate wasting (data)'
        }

        fig, ax = plt.subplots()
        bar_spots = len(columns_to_plot)
        bar_width = 0.8 / bar_spots
        pos = np.arange(len(w_prev_plot_df))
        dodge_offsets = np.linspace(-bar_spots * bar_width / 2, bar_spots * bar_width / 2, bar_spots, endpoint=False)
        for columns, offset in zip(columns_to_plot, dodge_offsets):
            bottom = 0
            for col in ([columns] if isinstance(columns, str) else columns):
                ax.bar(pos + offset, w_prev_plot_df[col], bottom=bottom, width=bar_width, align='edge',
                       label=labels_to_plot[col], color=colors_to_plot[col])
                bottom += w_prev_plot_df[col]
        ax.set_xticks(pos)
        ax.set_xticklabels(w_prev_plot_df.index, rotation=90)
        ax.set_title("Wasting prevalence in children 0-59 months per year")
        ax.set_ylabel('proportion of wasted children in the year')
        ax.set_xlabel('year')
        ax.legend()
        plt.tight_layout()
        fig_output_name = ('wasting_prevalence_per_year__' + self.datestamp)
        self.save_fig__store_pdf_file(fig, fig_output_name)
        # plt.show()

    def plot_wasting_prevalence_by_age_group(self):
        """ Plot wasting prevalence per each age group. Proportions are obtained by getting a total number of
        children wasted in a particular age-group divided by the total number of children per that age-group"""

        age_groups = ['0_5mo', '6_11mo', '12_23mo', '24_35mo', '36_47mo', '48_59mo', '5y+']
        columns = [f'mod__{age}' for age in age_groups] + [f'sev__{age}' for age in age_groups]
        # data in percent (0% to 100%)
        data = {
            2010: {
                'wasted_calib': [7.0, 13.0, 12.7, 2.4, 2.7, 1.9, 0.0],
                'sev_wast_calib': [2.1, 7.1, 4.7, 0.9, 0.7, 0.6, 0.0]
            },
            2014: {
                'wasted_calib': [5.8, 5.8, 5.4, 3.9, 2.2, 2.0, 0.0],
                'sev_wast_calib': [2.6, 2.5, 1.1, 0.8, 0.7, 0.3, 0.0]
            },
            2016: {
                'wasted_calib': [3.7, 7.7, 6.5, 2.2, 1.9, 2.6, 0.0],
                'sev_wast_calib': [1.1, 1.0, 0.7, 1.0, 0.1, 0.5, 0.0]
            },
            2020: {
                'wasted_calib': [2.5, 2.6, 9.1, 2.0, 1.8, 1.8, 0.0],
                'sev_wast_calib': [1.0, 1.0, 2.7, 0.8, 0.2, 0.3, 0.0]
            }
        }
        # recalculate data to proportions (0 to 1) and separate mod wast as (wasted - sev wast)
        for year in data:
            data[year]['mod_wast_calib'] = \
                [(w - s)/100 for w, s in zip(data[year]['wasted_calib'], data[year]['sev_wast_calib'])]
            data[year]['sev_wast_calib'] = \
                [s/100 for s in data[year]['sev_wast_calib']]
        data_list = []
        for year in data:
            values = data[year]['mod_wast_calib'] + data[year]['sev_wast_calib']
            data_list.append(values)
        w_prev_calib_data_df = pd.DataFrame(data_list, columns=columns, index=data.keys())

        w_prev_model_df = self.__w_logs_dict["wasting_prevalence_props"]
        w_prev_model_df = w_prev_model_df.drop(columns={'total_mod_under5_prop', 'total_sev_under5_prop'})
        w_prev_model_df = w_prev_model_df.set_index(w_prev_model_df.date.dt.year)
        w_prev_model_df = w_prev_model_df.drop(columns='date')

        for year_calib in w_prev_calib_data_df.index:
            w_prev_calib_data_year_df = w_prev_calib_data_df.loc[w_prev_calib_data_df.index == year_calib]
            w_prev_model_year_df = w_prev_model_df.loc[w_prev_model_df.index == year_calib]
            order_x_axis = ['0_5mo', '6_11mo', '12_23mo', '24_35mo', '36_47mo', '48_59mo', '5y+']

            def create_plotting_data(df, df_name):
                plotting = {'severe wasting': {}, 'moderate wasting': {}}
                for col in df.columns:
                    prefix, age_group = col.split('__')
                    if prefix == 'sev':
                        plotting['severe wasting'][age_group] = df[col].values[0]
                    elif prefix == 'mod':
                        plotting['moderate wasting'][age_group] = df[col].values[0]
                plotting_df = pd.DataFrame(plotting)
                assert set(plotting_df.index) == set(
                    order_x_axis), f"age groups in {w_prev_calib_data_year_df} are not in line with the order_x_axis."
                plotting_df = plotting_df.reindex(order_x_axis)
                return plotting_df

            # Create plotting data for both dataframes
            plotting_model = create_plotting_data(w_prev_model_year_df, 'w_prev_model_year_df')
            plotting_calib = create_plotting_data(w_prev_calib_data_year_df, 'w_prev_calib_data_year_df')

            # Plot wasting prevalence
            fig, ax = plt.subplots(figsize=(10, 6))
            bar_width = 0.35
            # Set positions of bars on x-axis
            r1 = range(len(plotting_model))
            r2 = [x + bar_width for x in r1]

            # Plot the first set of bars (model data)
            ax.bar(r1, plotting_model['severe wasting'],
                   color=self.__colors_model['severe wasting'], width=bar_width,
                   label='severe wasting (model)')
            ax.bar(r1, plotting_model['moderate wasting'], bottom=plotting_model['severe wasting'],
                   color=self.__colors_model['moderate wasting'], width=bar_width,
                   label='moderate wasting (model)')

            # Plot the second set of bars (calibration data)
            ax.bar(r2, plotting_calib['severe wasting'],
                   color=self.__colors_data['severe wasting'], width=bar_width,
                   label='severe wasting (data)')
            ax.bar(r2, plotting_calib['moderate wasting'], bottom=plotting_calib['severe wasting'],
                   color=self.__colors_data['moderate wasting'], width=bar_width,
                   label='moderate wasting (data)')

            ax.set_xlabel('age group')
            ax.set_ylabel('proportion')
            ax.set_title(f"Wasting prevalence in children 0-59 months per each age group in {year_calib}")
            ax.set_xticks([r + bar_width / 2 for r in range(len(plotting_model))])
            ax.set_xticklabels(order_x_axis)
            ax.set_ylim([0, 0.14])
            ax.legend()

            # Adjust the layout to make space for the footnote
            plt.subplots_adjust(top=0.15)  # Adjust the bottom margin
            # Add footnote
            fig.figure.text(0.43, 0.95,
                            "proportion = number of wasted children in the age group "
                            "/ total number of children in the age group",
                            ha="center", fontsize=10, bbox={"facecolor": "gray", "alpha": 0.3, "pad": 5})

            plt.tight_layout()
            fig_output_name = (f'wasting_prevalence_per_each_age_group_{year_calib}__' + self.datestamp)
            self.save_fig__store_pdf_file(fig, fig_output_name)
            # plt.show()

    def plot_wasting_initial_prevalence_by_age_group(self):
        """ Plot wasting prevalence per each age group. Proportions are obtained by getting a total number of
        children wasted in a particular age-group divided by the total number of children per that age-group"""
        w_prev_df = self.__w_logs_dict["wasting_init_prevalence_props"]
        w_prev_df = w_prev_df.drop(columns={'total_mod_under5_prop', 'total_sev_under5_prop'})
        w_prev_df = w_prev_df.set_index(w_prev_df.date.dt.year)
        w_prev_df = w_prev_df.drop(columns='date')
        plotting = {'severe wasting': {}, 'moderate wasting': {}}
        for col in w_prev_df.columns:
            prefix, age_group = col.split('__')
            if prefix == 'sev':
                plotting['severe wasting'][age_group] = w_prev_df[col].values[0]
            elif prefix == 'mod':
                plotting['moderate wasting'][age_group] = w_prev_df[col].values[0]
        plotting = pd.DataFrame(plotting)
        order_x_axis = ['0_5mo', '6_11mo', '12_23mo', '24_35mo', '36_47mo', '48_59mo', '5y+']
        # Assert all age groups are included
        assert set(plotting.index) == set(order_x_axis), "age groups are not in line with the order_x_axis."
        plotting = plotting.reindex(order_x_axis)

        # Plot wasting prevalence
        fig, ax = plt.subplots(figsize=(10, 6))
        plotting.squeeze().plot(kind='bar', stacked=True,
                                ax=ax,
                                ylabel='proportion',
                                xlabel='age group',
                                ylim=[0, 0.14])
        ax.set_title(r"Wasting prevalence in children 0-59 months per each age group $\bf{at}$ $\bf{initiation}$")
        # Adjust the layout to make space for the footnote
        plt.subplots_adjust(top=0.35)  # Adjust the bottom margin
        # Add footnote
        fig.figure.text(0.44, 0.9,
                        "proportion = number of wasted children in the age group "
                        "/ total number of children in the age group",
                        ha="center", fontsize=10, bbox={"facecolor": "gray", "alpha": 0.3, "pad": 5})
        plt.tight_layout()
        fig_output_name = ('wasting_initial_prevalence_per_each_age_group__' + self.datestamp)
        self.save_fig__store_pdf_file(fig, fig_output_name)
        # plt.show()

    def add_wasting_initial_prevalence_by_age_group(self):
        self.fig_files.append('wasting_initial_prevalence_per_each_age_group__' + self.datestamp + '.pdf')

    def plot_modal_gbd_deaths_by_gender(self):
        """ compare modal and GBD deaths by gender """
        death_compare = \
            compare_number_of_deaths(self.__log_file_path, resources_path)
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_df = death_compare.loc[(['2010-2014', '2015-2019'],
                                     slice(None), slice(None), 'Childhood Undernutrition'
                                     )].groupby('period').sum()
        plotting = plot_df.loc[['2010-2014', '2015-2019']]
        ax = plotting['model'].plot.bar(label='Model', ax=ax, rot=0)
        ax.errorbar(x=plotting['model'].index, y=plotting.GBD_mean,
                    yerr=[plotting.GBD_lower, plotting.GBD_upper],
                    fmt='o', color='#000', label="GBD")
        ax.set_title('Direct deaths due to severe acute malnutrition')
        ax.set_xlabel("time period")
        ax.set_ylabel("number of deaths")
        ax.legend(loc=2)
        fig.tight_layout()
        # Adjust the layout to make space for the footnote
        plt.subplots_adjust(bottom=0.15)  # Adjust the bottom margin
        # Add footnote
        fig.figure.text(0.5, 0.02,
                        "Model output against Global Burden of Diseases (GBD) study data",
                        ha="center", fontsize=10, bbox={"facecolor": "gray", "alpha": 0.3, "pad": 5})
        fig_output_name = ('modal_gbd_deaths_by_gender__' + self.datestamp)
        self.save_fig__store_pdf_file(fig, fig_output_name)
        # plt.show()

    def plot_all_figs_in_one_pdf(self):

        output_file_path = Path(self.outcomes_path_name + '/wasting_all_figures__' + self.datestamp + '.pdf')
        # Remove the existing output file if it exists to ensure a clean start
        if os.path.exists(output_file_path):
            os.remove(output_file_path)

        # Assert that the file doesn't exist anymore after removal
        assert not os.path.exists(output_file_path), "The file was not successfully removed."

        # Merge the PDF files
        # Create a PDF writer object
        pdf_writer = PdfWriter()

        # Iterate through the figure files and add each to the writer
        for fig_file in self.fig_files:
            pdf_reader = PdfReader(self.outcomes_path_name + "/" + fig_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                pdf_writer.add_page(page)

        # Write the merged PDF to a file
        with open(output_file_path, 'wb') as out_file:
            pdf_writer.write(out_file)


if __name__ == "__main__":

    # Path to the resource files used by the disease and intervention methods
    resources_path = Path("./resources")

    # Find sim_results_folder associated with a given batch_file (and get most recent [-1])
    sim_results_folder = get_scenario_outputs(scenario_filename, outputs_path)[-1]
    sim_results_parent_folder_name = str(sim_results_folder.parent)
    sim_results_folder_name = sim_results_folder.name
    # Get the datestamp
    assert sim_results_folder_name.startswith(scenario_filename + '-'),\
        "The scenario output name does not correspond with the set scenario_filename."
    datestamp = sim_results_folder_name[(len(scenario_filename) + 1):]

    # Path to the results folder
    sim_results_folder_path =  sim_results_parent_folder_name + '/' + sim_results_folder_name
    print(f"{sim_results_folder_path=}")
    folders = [name for name in os.listdir(sim_results_folder_path) if \
               os.path.isdir(os.path.join(sim_results_folder_path, name))]

    # Analyse each draw
    for draw_nmb in range(len(folders)):
        print(f"Analysing {draw_nmb=} ...")
        time_start = time.time()
        # Path to the draw folder
        sim_results_folder_path_draw_x_run_0 = \
            sim_results_parent_folder_name + '/' + sim_results_folder_name + f'/{draw_nmb}/0/'

        # initialise the wasting class
        wasting_analyses = WastingAnalyses(sim_results_folder_path_draw_x_run_0, datestamp)

        # plot wasting incidence
        wasting_analyses.plot_wasting_incidence()

        # plot wasting incidence mod:sev proportions
        # wasting_analyses.plot_wasting_incidence_mod_to_sev_props()

        # plot wasting length
        wasting_analyses.plot_wasting_length()

        # plot wasting prevalence
        wasting_analyses.plot_wasting_prevalence_per_year()

        # plot wasting initial prevalence by age group
        wasting_analyses.plot_wasting_initial_prevalence_by_age_group()

        # plot wasting prevalence by age group
        wasting_analyses.plot_wasting_prevalence_by_age_group()

        # plot wasting deaths by gender as compared to GBD deaths
        wasting_analyses.plot_modal_gbd_deaths_by_gender()

        # save all figures in one pdf
        wasting_analyses.plot_all_figs_in_one_pdf()

        time_end = time.time()
        print(f"... finished in (s): {(time_end - time_start)}")

    total_time_end = time.time()
    print(f"total running time (s): {(total_time_end - total_time_start)}")


