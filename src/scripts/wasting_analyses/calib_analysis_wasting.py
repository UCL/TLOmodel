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
from scipy import stats

from tlo.analysis.utils import compare_number_of_deaths, get_scenario_outputs, parse_log_file

# start time of the whole analysis
total_time_start = time.time()

# ####### TO SET #######################################################################################################
scenario_filename = 'wasting_analysis__minimal_model'
outputs_path = Path("./outputs/sejjej5@ucl.ac.uk/wasting")
legend_fontsize = 12
title_fontsize = 16
########################################################################################################################

def create_calib_outcome_csv(sim_results_folder_path_str):
    """
    Creates a new empty csv file with the header if it doesn't exist yet.
    :return:
    """
    csv_file_name = str(sim_results_folder_path_str).replace(str(outputs_path), '').lstrip('/') + \
                    "_model_calib-data_intersect_bool"
    csv_file_path = sim_results_folder_path_str / f"{csv_file_name}.csv"

    if not csv_file_path.exists():
        age_groups = [(0, 5), (6, 11), (12, 23), (24, 35), (36, 47), (48, 59)]
        calib_ys = [2015, 2019]
        wast_type_agegp = [f'{wast_type}_wast__{low_bound}_{high_bound}mo' for wast_type in ['any', 'sev'] for
                           low_bound, high_bound in age_groups]
        year_wast_age_grps = [f'{year}__{wast_age_grp}' for year in calib_ys for wast_age_grp in wast_type_agegp]
        sum_year_prev_calib_points = [f'{year}__sum_prev_calib_points' for year in calib_ys]

        with open(csv_file_path, 'w') as csv_file:
            csv_file.write(
                'draw,run,' + ','.join(year_wast_age_grps) + ',deaths_2010_2014,deaths_2015_2019,' +
                ','.join(sum_year_prev_calib_points) + ',sum_prev_calib_points,sum_all_calib_points\n'
            )

class WastingAnalyses:
    """
    This class looks at plotting all important outputs from the wasting module
    """

    def __init__(self, sim_results_folder_path_str, in_datestamp, in_draw_nmb, in_run_nmb, in_png=False):
        self.outcomes_folder_path = sim_results_folder_path_str
        self.datestamp = in_datestamp
        self.draw_nmb = in_draw_nmb
        self.run_nmb = in_run_nmb
        self.png = in_png, """bool indicating whether we want to save all figures not only as pdf, but also as png"""

        sim_results_folder_draw_x_run_0_path_str = sim_results_folder_path_str + f'/{draw_nmb}/{run_nmb}/'
        sim_results_file_name_prefix = scenario_filename
        sim_results_file_name_extension = '.log.gz'
        gz_results_file_path = \
            Path(glob.glob(os.path.join(sim_results_folder_draw_x_run_0_path_str,
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

        # wasting types description
        self.__wasting_types_desc = {'WHZ<-3': 'severe wasting',
                                     '-3<=WHZ<-2': 'moderate wasting',
                                     'WHZ>=-2': 'no wasting'}

        self.fig_files = []

        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # # define colo(u)rs to use:
        self.__colors_model = {
            'severe wasting': cycle[0],  # #1f77b4
            'moderate wasting': cycle[1],  # #ff7f0e
            'SAM': '#B372B7',
            'MAM': '#D1BCD2',
        }
        self.__colors_data = {
            'severe wasting': '#82C1EC',
            'moderate wasting': '#C71E1E',
        }
        self.__colors_init_data = {
            'severe wasting': '#0E53EA',
            'moderate wasting': '#FFA783',
        }

    def save_fig__store_pdf_file(self, fig, fig_output_name: str) -> None:
        full_path_and_file_name = self.outcomes_folder_path + f'/{self.draw_nmb}/{self.run_nmb}/' + fig_output_name + \
                    f'_{self.draw_nmb}_{self.run_nmb}'
        if self.png: #TODO: doesn't seem to be working
            fig.savefig(full_path_and_file_name + '.png', format='png')
        fig.savefig(full_path_and_file_name + '.pdf', format='pdf')
        self.fig_files.append(full_path_and_file_name + '.pdf')

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
            plotting["-3<=WHZ<-2"] = plotting.apply(lambda row: max(row["-3<=WHZ<-2"] - row["WHZ<-3"], 0), axis=1)
            # calculate props within the age group
            plotting = plotting.div(age_gps_total_pop_sizes_df[age], axis=0)
            plotting = plotting.rename(columns=self.__wasting_types_desc)
            # filter data to include only years from 2015 onwards
            plotting = plotting.loc[plotting.index >= 2015]
            # check for invalid values
            if (plotting < 0).any().any() or (plotting > 1).any().any():
                print(f"Warning plot_wasting_incidence: Invalid values detected in plotting data for age group {age}:")
                print(plotting)

            ax = plotting.plot(kind='bar', stacked=True,
                               ax=axes[_row_counter, _col_counter],
                               title=f"{age} old")#,
                               #ylim=[0, 1])
            show_legend = (_row_counter == 1 and _col_counter == 2)
            # show_x_axis_label = (_row_counter == 0 and _col_counter == 2)
            if show_legend:
                ax.legend(loc='center', fontsize=legend_fontsize)
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
            fig.suptitle('Annual incidence of wasting among the age group', fontsize=title_fontsize) #, weight='bold')
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
            w_length_df = w_length_df.loc[:, ['mod_MAM_nat_full_recov',
                                 'mod_SAM_nat_full_recov', 'mod_SAM_nat_recov_to_MAM',
                                 'sev_SAM_nat_full_recov', 'sev_SAM_nat_recov_to_MAM',
                                 'mod_MAM_tx/nat_full_recov',
                                 'mod_SAM_tx_full_recov', 'mod_SAM_tx/nat_recov_to_MAM',
                                 'sev_SAM_tx_full_recov', 'sev_SAM_tx/nat_recov_to_MAM',
                                 'mod_not_yet_recovered',
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

    def plot_wasting_initial_overall_prevalence(self):
        """ plot wasting prevalence of all age groups for the year 2010. Proportions are obtained by getting a total
        number of children wasted (moderately and severely) divided by the total number of children less than 5 years"""

        ## Prevalence at 2010, ie data from the same source used to draw initial prevalence by age group
        w_prev_calib_data_years_only_df = pd.DataFrame({
            'sev_wast_calib': [0.015],
            'mod_wast_calib': [0.025]
        }, index=[2010])
        date_range = pd.Index([2010], name='date')
        w_prev_calib = pd.DataFrame(index=date_range)
        # filling missing values with 0
        w_prev_calib_df = w_prev_calib.merge(
            w_prev_calib_data_years_only_df, left_index=True, right_index=True, how='left'
        ).fillna(0)

        ## Initial prevalence at the beginning of 2010 - model
        init_w_prev_2010_only_df = self.__w_logs_dict["wasting_init_prevalence_props"]
        init_w_prev_2010_only_df = init_w_prev_2010_only_df[['date', 'total_sev_under5_prop', 'total_mod_under5_prop']].rename(
            columns={'total_sev_under5_prop': 'total_init_sev_under5_prop', 'total_mod_under5_prop': 'total_init_mod_under5_prop'}
        )
        init_w_prev_2010_only_df = init_w_prev_2010_only_df.set_index(init_w_prev_2010_only_df.date.dt.year)
        init_w_prev_2010_only_df = init_w_prev_2010_only_df.drop(columns='date')
        init_w_prev_2010_only_df = init_w_prev_2010_only_df.loc[[2010]]
        init_w_prev_df = pd.DataFrame(index=date_range)
        # filling missing values with 0
        init_w_prev_df = init_w_prev_df.merge(
            init_w_prev_2010_only_df, left_index=True, right_index=True, how='left'
        ).fillna(0)

        w_prev_calib_and_init_df = pd.merge(init_w_prev_df, w_prev_calib_df, on='date')
        columns_to_plot = [
            ['total_init_sev_under5_prop', 'total_init_mod_under5_prop'],
            ['sev_wast_calib', 'mod_wast_calib'],
        ]
        colors_to_plot = {
            'total_init_sev_under5_prop': self.__colors_init_data['severe wasting'],
            'total_init_mod_under5_prop': self.__colors_init_data['moderate wasting'],
            'sev_wast_calib': self.__colors_data['severe wasting'],
            'mod_wast_calib': self.__colors_data['moderate wasting'],
        }
        labels_to_plot = {
            'total_init_sev_under5_prop': 'severe wasting (initial)',
            'total_init_mod_under5_prop': 'moderate wasting (initial)',
            'sev_wast_calib': 'severe wasting (data)',
            'mod_wast_calib': 'moderate wasting (data)',
        }

        fig, ax = plt.subplots()
        bar_spots = len(columns_to_plot)
        bar_width = 0.3 / bar_spots
        pos = np.arange(len(w_prev_calib_and_init_df))
        dodge_offsets = np.linspace(-bar_spots * bar_width / 2, bar_spots * bar_width / 2, bar_spots, endpoint=False)
        for columns, offset in zip(columns_to_plot, dodge_offsets):
            bottom = 0
            for col in ([columns] if isinstance(columns, str) else columns):
                ax.bar(pos + offset, w_prev_calib_and_init_df[col], bottom=bottom, width=bar_width, align='edge',
                       label=labels_to_plot[col], color=colors_to_plot[col])
                bottom += w_prev_calib_and_init_df[col]
        ax.set_xticks(pos)
        ax.set_xticklabels(w_prev_calib_and_init_df.index, rotation=90)
        ax.set_title(r"Overall wasting prevalence $\bf{at}$ $\bf{initiation}$ (2010)", fontsize=title_fontsize-1)
        ax.set_ylabel('proportion of wasted children in the year')
        ax.set_xlabel('year')
        ax.set_ylim([0, 0.131])
        ax.legend(fontsize=legend_fontsize)
        plt.tight_layout()
        fig_output_name = ('wasting_initial_overall_prevalence__' + self.datestamp)
        self.save_fig__store_pdf_file(fig, fig_output_name)
        # plt.show()

    def plot_wasting_initial_prevalence_by_age_group(self):
        """ Plot wasting prevalence per each age group. Proportions are obtained by getting a total number of
        children wasted in a particular age-group divided by the total number of children per that age-group"""

        # Initial prevalence at the beginning of 2010 - model
        w_prev_df = self.__w_logs_dict["wasting_init_prevalence_props"]
        w_prev_df = w_prev_df.drop(columns={'total_mod_under5_prop', 'total_sev_under5_prop'})
        w_prev_df = w_prev_df.set_index(w_prev_df.date.dt.year)
        w_prev_df = w_prev_df.drop(columns='date')

        # 2010 prevalence calibration data
        # TODO: load data_2010 from the resource file:
        #  resources_path / 'ResourceFile_Wasting/wasting_prevalence_and_sample_size.csv'
        data_2010 = {
            'wasted_calib': [7.0, 13.0, 12.7, 2.4, 2.7, 1.9, 0.0],
            'sev_wast_calib': [2.1, 7.1, 4.7, 0.9, 0.7, 0.6, 0.0]
        }
        data_2010['mod_wast_calib'] = \
            [(w - s)/100 for w, s in zip(data_2010['wasted_calib'], data_2010['sev_wast_calib'])]
        data_2010['sev_wast_calib'] = \
            [s/100 for s in data_2010['sev_wast_calib']]

        # Prepare plotting data
        plotting_model = {'severe wasting': {}, 'moderate wasting': {}}
        for col in w_prev_df.columns:
            prefix, age_group = col.split('__')
            if prefix == 'sev':
                plotting_model['severe wasting'][age_group] = w_prev_df[col].values[0]
            elif prefix == 'mod':
                plotting_model['moderate wasting'][age_group] = w_prev_df[col].values[0]
        plotting_model = pd.DataFrame(plotting_model)

        plotting_calib = {'severe wasting': {}, 'moderate wasting': {}}
        age_groups = ['0_5mo', '6_11mo', '12_23mo', '24_35mo', '36_47mo', '48_59mo', '5y+']
        for i, age_group in enumerate(age_groups):
            plotting_calib['severe wasting'][age_group] = data_2010['sev_wast_calib'][i]
            plotting_calib['moderate wasting'][age_group] = data_2010['mod_wast_calib'][i]
        plotting_calib = pd.DataFrame(plotting_calib)

        plotting_model = plotting_model.reindex(age_groups)
        plotting_calib = plotting_calib.reindex(age_groups)

        # Plot wasting prevalence
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.35
        # Set positions of bars on x-axis
        r1 = range(len(plotting_model))
        r2 = [x + bar_width for x in r1]

        # Plot the first set of bars (model data)
        ax.bar(r1, plotting_model['severe wasting'],
               color=self.__colors_init_data['severe wasting'], width=bar_width,
               label='severe wasting (initial)')
        ax.bar(r1, plotting_model['moderate wasting'], bottom=plotting_model['severe wasting'],
               color=self.__colors_init_data['moderate wasting'], width=bar_width,
               label='moderate wasting (initial)')

        # Plot the second set of bars (calibration data)
        ax.bar(r2, plotting_calib['severe wasting'],
               color=self.__colors_data['severe wasting'], width=bar_width,
               label='severe wasting (data)')
        ax.bar(r2, plotting_calib['moderate wasting'], bottom=plotting_calib['severe wasting'],
               color=self.__colors_data['moderate wasting'], width=bar_width,
               label='moderate wasting (data)')

        ax.set_xlabel('age group')
        ax.set_ylabel('proportion')
        ax.set_title(
            r"Wasting prevalence in children 0-59 months per each age group $\bf{at}$ $\bf{initiation}$ (2010)",
            fontsize=title_fontsize-1)
        ax.set_xticks([r + bar_width / 2 for r in range(len(plotting_model))])
        ax.set_xticklabels(age_groups)
        ax.set_ylim([0, 0.16])
        ax.legend(fontsize=legend_fontsize)

        # Adjust the layout to make space for the footnote
        plt.subplots_adjust(top=0.15) # Adjust the top margin
        # Add footnote
        fig.figure.text(0.43, 0.95,
                        "proportion = number of wasted children in the age group "
                        "/ total number of children in the age group",
                        ha="center", fontsize=10, bbox={"facecolor": "gray", "alpha": 0.3, "pad": 5})
        plt.tight_layout()
        fig_output_name = ('wasting_initial_prevalence_per_each_age_group__' + self.datestamp)
        self.save_fig__store_pdf_file(fig, fig_output_name)
        # plt.show()

    def plot_wasting_prevalence_per_year(self):
        """ plot wasting prevalence of all age groups per year. Proportions are obtained by getting a total number of
        children wasted divide by the total number of children less than 5 years"""

        ## Prevalence at some years - data (2010 are the data used to draw initial prevalence)
        # TODO: add calibration data into the resource file:
        #  resources_path / 'ResourceFile_Wasting/wasting_prevalence_and_sample_size.csv'
        #  and load here and for initial overall prev from the RF
        w_prev_calib_data_years_only_df = pd.DataFrame({
            'sev_wast_calib': [0.015, 0.011, 0.006, 0.007],
            'mod_wast_calib': [0.025, 0.027, 0.021, 0.019]
        }, index=[2010, 2013, 2015, 2019])
        date_range = pd.Index(range(2010, 2031), name='date')
        w_prev_calib = pd.DataFrame(index=date_range)
        # filling missing values with 0
        w_prev_calib_df = w_prev_calib.merge(
            w_prev_calib_data_years_only_df, left_index=True, right_index=True, how='left'
        ).fillna(0)

        ## Prevalence at the end of years - model
        w_prev_df = self.__w_logs_dict["wasting_prevalence_props"]
        w_prev_df = w_prev_df[['date', 'total_sev_under5_prop', 'total_mod_under5_prop']]
        w_prev_df = w_prev_df.set_index(w_prev_df.date.dt.year)
        w_prev_df = w_prev_df.drop(columns='date')

        ## Initial prevalence at the beginning of 2010 - model
        init_w_prev_2010_only_df = self.__w_logs_dict["wasting_init_prevalence_props"]
        init_w_prev_2010_only_df = init_w_prev_2010_only_df[['date', 'total_sev_under5_prop', 'total_mod_under5_prop']].rename(
            columns={'total_sev_under5_prop': 'total_init_sev_under5_prop', 'total_mod_under5_prop': 'total_init_mod_under5_prop'}
        )
        init_w_prev_2010_only_df = init_w_prev_2010_only_df.set_index(init_w_prev_2010_only_df.date.dt.year)
        init_w_prev_2010_only_df = init_w_prev_2010_only_df.drop(columns='date')
        init_w_prev_2010_only_df = init_w_prev_2010_only_df.loc[[2010]]
        init_w_prev_df = pd.DataFrame(index=date_range)
        # filling missing values with 0
        init_w_prev_df = init_w_prev_df.merge(
            init_w_prev_2010_only_df, left_index=True, right_index=True, how='left'
        ).fillna(0)

        w_prev_calib_and_init_df = pd.merge(init_w_prev_df, w_prev_calib_df, on='date')
        w_prev_plot_df = pd.merge(w_prev_df, w_prev_calib_and_init_df, on='date').loc[lambda df: df.index >= 2015]
        columns_to_plot = [
            ['total_init_sev_under5_prop', 'total_init_mod_under5_prop'],
            ['total_sev_under5_prop', 'total_mod_under5_prop'],
            ['sev_wast_calib', 'mod_wast_calib'],
            ]
        colors_to_plot = {
            'total_sev_under5_prop': self.__colors_model['severe wasting'],
            'total_mod_under5_prop': self.__colors_model['moderate wasting'],
            'sev_wast_calib': self.__colors_data['severe wasting'],
            'mod_wast_calib': self.__colors_data['moderate wasting'],
            'total_init_sev_under5_prop': self.__colors_init_data['severe wasting'],
            'total_init_mod_under5_prop': self.__colors_init_data['moderate wasting']

        }
        labels_to_plot = {
            'total_sev_under5_prop': 'severe wasting (model)',
            'total_mod_under5_prop': 'moderate wasting (model)',
            'sev_wast_calib': 'severe wasting (data)',
            'mod_wast_calib': 'moderate wasting (data)',
            'total_init_sev_under5_prop': 'severe wasting (initial)',
            'total_init_mod_under5_prop': 'moderate wasting (initial)'
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
        ax.set_title("Wasting prevalence in children 0-59 months per year", fontsize=title_fontsize-6)
        ax.set_ylabel('proportion of wasted children in the year')
        ax.set_xlabel('year')
        ax.set_ylim([0, 0.06])
        ax.legend(fontsize=legend_fontsize-4)
        plt.tight_layout()
        fig_output_name = ('wasting_prevalence_per_year__' + self.datestamp)
        self.save_fig__store_pdf_file(fig, fig_output_name)
        # plt.show()

    def plot_wasting_prevalence_by_age_group(self):
        """ Plot wasting prevalence per each age group. Proportions are obtained by getting a total number of
        children wasted in a particular age-group divided by the total number of children per that age-group"""

        age_groups = ['0_5mo', '6_11mo', '12_23mo', '24_35mo', '36_47mo', '48_59mo', '5y+']

        # ### Calibration Data
        # Load calibration data from CSV file
        wasting_calib_data_path = resources_path / 'ResourceFile_Wasting/wasting_prevalence_and_sample_size.csv'
        wasting_calib_data_df = pd.read_csv(wasting_calib_data_path, index_col='year')

        # Recalculate data to proportions (0 to 1) and separate mod wast as (wasted - sev wast)
        wasting_calib_data_df['mod_wast_calib'] = \
            (wasting_calib_data_df['prev any wast (%)'] - wasting_calib_data_df['prev severe wast (%)']) / 100
        wasting_calib_data_df['sev_wast_calib'] = wasting_calib_data_df['prev severe wast (%)'] / 100

        # Pivot the data to get the required format
        w_prev_calib_data_df = wasting_calib_data_df.pivot(columns='age_group (months)',
                                                     values=['mod_wast_calib', 'sev_wast_calib'])
        w_prev_calib_data_df.columns = [f'{col[0][:3]}__{col[1]}' for col in w_prev_calib_data_df.columns]

        # Load calibration sample sizes from CSV file
        sample_sizes_calib_data_df = wasting_calib_data_df.pivot(columns='age_group (months)', values='sample_size')
        sample_sizes_calib_data_df = sample_sizes_calib_data_df.reindex(columns=age_groups)

        # ### Model Outcomes
        # Load modelled prevalence proportions
        w_prev_model_df = self.__w_logs_dict["wasting_prevalence_props"]
        w_prev_model_df = w_prev_model_df.drop(columns={'total_mod_under5_prop', 'total_sev_under5_prop'})
        w_prev_model_df = w_prev_model_df.set_index(w_prev_model_df.date.dt.year)
        w_prev_model_df = w_prev_model_df.drop(columns='date')

        # Load modelled population sizes
        pop_sizes_model_df = self.__w_logs_dict['pop sizes']
        pop_sizes_model_df = pop_sizes_model_df.set_index(pop_sizes_model_df.date.dt.year).rename_axis('year')
        pop_sizes_model_df = pop_sizes_model_df.drop(columns='date')
        pop_sizes_model_df = pop_sizes_model_df.filter(like='total__').rename(
            lambda x: x.replace('total__', ''), axis=1
        )[age_groups]

        for year_calib in w_prev_calib_data_df.index:
            w_prev_calib_data_year_df = w_prev_calib_data_df.loc[w_prev_calib_data_df.index == year_calib]
            w_prev_model_year_df = w_prev_model_df.loc[w_prev_model_df.index == year_calib]

            def create_plotting_data(df, df_name):
                plotting = {'severe wasting': {}, 'moderate wasting': {}, 'any wasting': {}}
                for col in df.columns:
                    prefix, agegp = col.split('__')
                    if prefix == 'sev':
                        plotting['severe wasting'][agegp] = df[col].values[0]
                    elif prefix == 'mod':
                        plotting['moderate wasting'][agegp] = df[col].values[0]
                        plotting['any wasting'][agegp] = df[col].values[0] + df[f'sev__{agegp}'].values[0]
                plotting_df = pd.DataFrame(plotting)
                assert set(plotting_df.index) == set(age_groups),\
                    f"age groups in {df_name} are not in line with the age_groups."
                plotting_df = plotting_df.reindex(age_groups)
                return plotting_df

            # Create plotting data for both dataframes
            plotting_model = create_plotting_data(w_prev_model_year_df, 'w_prev_model_year_df')
            plotting_calib = create_plotting_data(w_prev_calib_data_year_df, 'w_prev_calib_data_year_df')

            # Calculate 95% confidence intervals for both
            sample_sizes_calib_data_year = sample_sizes_calib_data_df.loc[year_calib, :]
            sample_sizes_model_year = pop_sizes_model_df.loc[year_calib, :]

            confidence_level = 0.95
            z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)

            calib_data_margin_of_error_any_wast = []
            calib_data_margin_of_error_sev_wast = []
            for p, n in zip(plotting_calib['any wasting'].reindex(age_groups[:-1]), sample_sizes_calib_data_year[:-1]):
                calib_data_margin_of_error_any_wast.append(z_score * np.sqrt((p * (1 - p)) / n))
            for p, n in \
                zip(plotting_calib['severe wasting'].reindex(age_groups[:-1]),sample_sizes_calib_data_year[:-1]):
                calib_data_margin_of_error_sev_wast.append(z_score * np.sqrt((p * (1 - p)) / n))
            model_margin_of_error_any_wast = []
            model_margin_of_error_sev_wast = []
            for p, n in zip(plotting_model['any wasting'].reindex(age_groups[:-1]), sample_sizes_model_year[:-1]):
                model_margin_of_error_any_wast.append(z_score * np.sqrt((p * (1 - p)) / n))
            for p, n in zip(plotting_model['severe wasting'].reindex(age_groups[:-1]), sample_sizes_model_year[:-1]):
                model_margin_of_error_sev_wast.append(z_score * np.sqrt((p * (1 - p)) / n))

            # #####
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

            # Add the confidence intervals
            for i, age_group in enumerate(age_groups[0:len(age_groups)-1]):
                ax.errorbar(r1[i], plotting_model['any wasting'][age_group],
                            yerr=[model_margin_of_error_any_wast[i]],
                            capsize=5, fmt='none', color='black')
                ax.errorbar(r1[i], plotting_model['severe wasting'][age_group],
                            yerr=[model_margin_of_error_sev_wast[i]],
                            capsize=5, fmt='none', color='white')

            # Plot the second set of bars (calibration data)
            ax.bar(r2, plotting_calib['severe wasting'],
                   color=self.__colors_data['severe wasting'], width=bar_width,
                   label='severe wasting (data)')
            ax.bar(r2, plotting_calib['moderate wasting'], bottom=plotting_calib['severe wasting'],
                   color=self.__colors_data['moderate wasting'], width=bar_width,
                   label='moderate wasting (data)')

            # Add the confidence intervals
            for i, age_group in enumerate(age_groups[0:len(age_groups)-1]):
                ax.errorbar(r2[i], plotting_calib['any wasting'][age_group],
                            yerr=[calib_data_margin_of_error_any_wast[i]],
                            capsize=5, fmt='none', color='black')
                ax.errorbar(r2[i], plotting_calib['severe wasting'][age_group],
                            yerr=[calib_data_margin_of_error_sev_wast[i]],
                            capsize=5, fmt='none', color='white')

            ax.set_xlabel('age group')
            ax.set_ylabel('proportion')
            ax.set_title(f"Wasting prevalence in children 0-59 months per each age group in {year_calib}",
                         fontsize=title_fontsize-1)
            ax.set_xticks([r + bar_width / 2 for r in range(len(plotting_model))])
            ax.set_xticklabels(age_groups)
            ax.set_ylim([0, 0.12])
            ax.legend(fontsize=legend_fontsize)

            # Adjust the layout to make space for the footnote
            plt.subplots_adjust(top=0.15)  # Adjust the bottom margin
            # Add footnote
            fig.figure.text(0.43, 0.95,
                            "proportion = number of wasted children in the age group "
                            "/ total number of children in the age group",
                            ha="center", fontsize=10, bbox={"facecolor": "gray", "alpha": 0.3, "pad": 5})

            plt.tight_layout()
            fig_output_name = (f'wasting_prevalence_per_each_age_group_{year_calib}__' + self.datestamp)
            if year_calib in [2015, 2019]:
                self.save_fig__store_pdf_file(fig, fig_output_name)
            # plt.show()

    def plot_model_gbd_deaths_incl_burnin_period(self):
        """ compare model and GBD deaths 2010-2014 & 2015-2019 """
        death_compare = \
            compare_number_of_deaths(self.__log_file_path, resources_path)
        fig, ax = plt.subplots(figsize=(10, 6))
        # cause of death as of GBD 2019 'Protein-energy malnutrition' was labeled as 'Childhood Undernutrition' in
        # wasting module
        plot_df = death_compare.loc[(['2010-2014', '2015-2019'],
                                     slice(None), ['0-4'], 'Childhood Undernutrition'
                                     )].groupby('period').sum()
        plotting = plot_df.loc[['2010-2014', '2015-2019']]
        ax = plotting['model'].plot.bar(label='Model', ax=ax, rot=0)
        ax.errorbar(x=plotting['model'].index, y=plotting.GBD_mean,
                    yerr=[plotting.GBD_lower, plotting.GBD_upper],
                    fmt='o', color='#000', label="GBD")

        ax.set_title('Average direct deaths per year due to severe acute malnutrition in children under 5', fontsize=title_fontsize-1)
        ax.set_xlabel("time period")
        ax.set_ylabel("number of deaths")
        ax.legend(loc='upper center', fontsize=legend_fontsize)
        fig.tight_layout()
        # Adjust the layout to make space for the footnote
        plt.subplots_adjust(bottom=0.15)  # Adjust the bottom margin
        # Add footnote
        fig.figure.text(0.5, 0.02,
                        "Model output against Global Burden of Diseases (GBD) study data",
                        ha="center", fontsize=10, bbox={"facecolor": "gray", "alpha": 0.3, "pad": 5})
        fig_output_name = ('model_gbd_deaths_incl_burnin__' + self.datestamp)
        self.save_fig__store_pdf_file(fig, fig_output_name)
        # plt.show()

    def plot_model_gbd_deaths_excl_burnin_period(self):
        """ compare model and GBD deaths 2015-2019 """
        death_compare = \
            compare_number_of_deaths(self.__log_file_path, resources_path)
        fig, ax = plt.subplots(figsize=(10, 6))
        # cause of death as of GBD 2019 'Protein-energy malnutrition' was labeled as 'Childhood Undernutrition' in
        # wasting module
        plot_df = death_compare.loc[(['2015-2019'],
                                     slice(None), ['0-4'], 'Childhood Undernutrition'
                                     )].groupby('period').sum()
        plotting = plot_df.loc[['2015-2019']]
        ax = plotting['model'].plot.bar(label='Model', ax=ax, rot=0)
        ax.errorbar(x=plotting['model'].index, y=plotting.GBD_mean,
                    yerr=[plotting.GBD_lower, plotting.GBD_upper],
                    fmt='o', color='#000', label="GBD")

        ax.set_title('Average direct deaths per year due to severe acute malnutrition in children under 5',
                     fontsize=title_fontsize - 1)
        ax.set_xlabel("time period")
        ax.set_ylabel("number of deaths")
        ax.legend(loc='upper right', fontsize=legend_fontsize)
        fig.tight_layout()
        # Adjust the layout to make space for the footnote
        plt.subplots_adjust(bottom=0.15)  # Adjust the bottom margin
        # Add footnote
        fig.figure.text(0.5, 0.02,
                        "Model output against Global Burden of Diseases (GBD) study data",
                        ha="center", fontsize=10, bbox={"facecolor": "gray", "alpha": 0.3, "pad": 5})
        fig_output_name = ('model_gbd_deaths_excl_burnin__' + self.datestamp)
        self.save_fig__store_pdf_file(fig, fig_output_name)
        # plt.show()

    def plot_all_figs_in_one_pdf(self, in_outcome_figs_folder):

        output_file_path = \
            in_outcome_figs_folder / f'wasting_all_figures__{self.datestamp}_{self.draw_nmb}_{self.run_nmb}.pdf'
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
            pdf_reader = PdfReader(fig_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                pdf_writer.add_page(page)

        # Write the merged PDF to a file
        with open(output_file_path, 'wb') as out_file:
            pdf_writer.write(out_file)


if __name__ == "__main__":

    # Path to the resource files used by the disease and intervention methods
    resources_path = Path("./resources")

    # Find sim_results_folder_path associated with a given batch_file (and get most recent [-1])
    sim_results_folder_path = get_scenario_outputs(scenario_filename, outputs_path)[-1]
    sim_results_folder_name = sim_results_folder_path.name
    # Get the datestamp
    assert sim_results_folder_name.startswith(scenario_filename + '-'),\
        "The scenario output name does not correspond with the set scenario_filename."
    datestamp = sim_results_folder_name[(len(scenario_filename) + 1):]

    folders = [name for name in os.listdir(sim_results_folder_path) if \
    os.path.isdir(os.path.join(sim_results_folder_path, name)) and name.isdigit()]

    # Create a csv to write down calibration outputs
    #  as bool values indicating whether model outcomes and calibration data intersect
    create_calib_outcome_csv(sim_results_folder_path)

    # Analyse each draw
    # for now, we always have just one run, run 0
    run_nmb = 0
    for draw_nmb in range(0, len(folders)):
        print(f"Analysing {draw_nmb=} ...")
        time_start = time.time()

        # initialise the wasting class
        wasting_analyses = WastingAnalyses(str(sim_results_folder_path), datestamp, draw_nmb, run_nmb)

        # plot wasting incidence
        wasting_analyses.plot_wasting_incidence()

        # plot wasting incidence mod:sev proportions
        # wasting_analyses.plot_wasting_incidence_mod_to_sev_props()

        # plot wasting length
        # wasting_analyses.plot_wasting_length()

        # plot initial wasting prevalence
        wasting_analyses.plot_wasting_initial_overall_prevalence()
        wasting_analyses.plot_wasting_initial_prevalence_by_age_group()

        # plot prevalence through simulation
        wasting_analyses.plot_wasting_prevalence_per_year()
        wasting_analyses.plot_wasting_prevalence_by_age_group()

        # plot wasting deaths as compared to GBD deaths
        # wasting_analyses.plot_model_gbd_deaths_incl_burnin_period()
        wasting_analyses.plot_model_gbd_deaths_excl_burnin_period()

        # ### Save all figures in one pdf
        outcome_figs_folder = sim_results_folder_path / '_outcome_figures'
        outcome_figs_folder.mkdir(parents=True, exist_ok=True)
        wasting_analyses.plot_all_figs_in_one_pdf(outcome_figs_folder)

        time_end = time.time()
        print(f"... finished in (s): {(time_end - time_start)}")

    total_time_end = time.time()
    print(f"total running time (s): {(total_time_end - total_time_start)}")


