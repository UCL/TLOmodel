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

# start time of the analysis
time_start = time.time()

# ####### TO SET #######################################################################################################
scenario_filename = 'wasting_analysis__minimal_model'
outputs_path = Path("./outputs/sejjej5@ucl.ac.uk/wasting")
########################################################################################################################


class WastingAnalyses:
    """
    This class looks at plotting all important outputs from the wasting module
    """

    def __init__(self, in_scenario_filename, in_outputs_path):

        # Find sim_results_folder associated with a given batch_file (and get most recent [-1])
        sim_results_folder = get_scenario_outputs(in_scenario_filename, in_outputs_path)[-1]
        sim_results_parent_folder_name = str(sim_results_folder.parent)
        sim_results_folder_name = sim_results_folder.name
        self.outcomes_path_name = str(in_outputs_path) + "/" + sim_results_folder_name
        # Get the datestamp
        if sim_results_folder_name.startswith(scenario_filename + '-'):
            self.datestamp = sim_results_folder_name[(len(scenario_filename)+1):]
        else:
            print("The scenario output name does not correspond with the set scenario_filename.")

        # Path to the .log.gz file
        sim_results_folder_path_run0_draw0 = sim_results_parent_folder_name + '/' + sim_results_folder_name + '/0/0/'
        sim_results_file_name_prefix = scenario_filename
        sim_results_file_name_extension = '.log.gz'
        gz_results_file_path = \
            Path(glob.glob(os.path.join(sim_results_folder_path_run0_draw0,
                                        f"{sim_results_file_name_prefix}*{sim_results_file_name_extension}"))[0])

        # Path to the decompressed .log file
        log_results_file_path = gz_results_file_path.with_suffix('')

        # Decompress the .log.gz file
        with gzip.open(gz_results_file_path, 'rb') as f_in:
            with open(log_results_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        self.__log_file_path = log_results_file_path
        # parse wasting logs
        self.__w_logs_dict = parse_log_file(self.__log_file_path)['tlo.methods.wasting']
        # parse scaling factor log
        self.__scaling_factor = \
            parse_log_file(self.__log_file_path)['tlo.methods.population']['scaling_factor'].set_index('date').loc[
                '2010-01-01', 'scaling_factor'
            ]

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
        # get age_years, doesn't matter what wasting category you choose,
        # they all have same age groups
        age_years = list(w_inc_df.loc[w_inc_df.index[0], 'WHZ<-3'].keys(

        ))
        # age_years.remove('5+y')

        _row_counter = 0
        _col_counter = 0
        # plot setup
        fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(10, 6))
        axes[1, 2].axis('off')  # 5+y has no data (no new cases in 5+y), its space is used to display the label
        for _age in age_years:
            plotting = pd.DataFrame()
            for state in w_inc_df.columns:
                plotting[state] = \
                    w_inc_df.apply(lambda row: row[state][_age], axis=1)
            # remove sev cases from mod cases (all sev cases went through mod state)
            plotting["-3<=WHZ<-2"] = plotting["-3<=WHZ<-2"] - plotting["WHZ<-3"]
            # rescale nmbs from simulated pop_size to pop size of Malawi
            plotting = plotting * self.__scaling_factor
            plotting = plotting.rename(columns=self.__wasting_types_desc)

            ax = plotting.plot(kind='bar', stacked=True,
                               ax=axes[_row_counter, _col_counter],
                               title=f"incidence of wasting in {_age} old")#,
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
            ax.set_ylabel('number of incidence cases')
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
        w_prev_df = self.__w_logs_dict["wasting_prevalence_props"]
        w_prev_df = w_prev_df.drop(columns={'total_mod_under5_prop', 'total_sev_under5_prop'})
        w_prev_df = w_prev_df.set_index(w_prev_df.date.dt.year)
        w_prev_df = w_prev_df.loc[w_prev_df.index == 2020]
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
                                title="Wasting prevalence in children 0-59 months per each age group in 2020",
                                ylabel='proportion',
                                xlabel='age group',
                                ylim=[0, 0.2])
        # Adjust the layout to make space for the footnote
        plt.subplots_adjust(bottom=0.85)  # Adjust the bottom margin
        # Add footnote
        fig.figure.text(0.45, 0.88,
                        "proportion = number of wasted children in the age group "
                        "/ total number of children in the age group",
                        ha="center", fontsize=10, bbox={"facecolor": "gray", "alpha": 0.3, "pad": 5})
        plt.tight_layout()
        fig_output_name = ('wasting_prevalence_per_each_age_group__' + self.datestamp)
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
                                ylim=[0, 0.2])
        ax.set_title(r"Wasting prevalence in children 0-59 months per each age group $\bf{at}$ $\bf{initiation}$")
        # Adjust the layout to make space for the footnote
        plt.subplots_adjust(bottom=0.85)  # Adjust the bottom margin
        # Add footnote
        fig.figure.text(0.45, 0.88,
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

    # initialise the wasting class
    wasting_analyses = WastingAnalyses(scenario_filename, outputs_path)

    # plot wasting incidence
    wasting_analyses.plot_wasting_incidence()

    # plot wasting incidence mod:sev proportions
    # wasting_analyses.plot_wasting_incidence_mod_to_sev_props()

    # plot wasting length
    wasting_analyses.plot_wasting_length()

    # plot wasting prevalence
    wasting_analyses.plot_wasting_prevalence_per_year()

    # plot wasting prevalence by age group
    wasting_analyses.plot_wasting_prevalence_by_age_group()

    # plot wasting initial prevalence by age group
    wasting_analyses.plot_wasting_initial_prevalence_by_age_group()

    # plot wasting deaths by gender as compared to GBD deaths
    wasting_analyses.plot_modal_gbd_deaths_by_gender()

    # save all figures in one pdf
    wasting_analyses.plot_all_figs_in_one_pdf()
