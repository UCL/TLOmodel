"""
An analysis file for the wasting module (so far only for 1 run, 1 draw)
"""
# %% Import statements
import glob
import gzip
import os
import PyPDF2
import shutil
import time

from fpdf import FPDF
from pathlib import Path
from PIL import Image

import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    compare_number_of_deaths,
    get_scenario_outputs,
    load_pickled_dataframes,
    parse_log_file
)

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

        self.__scenario_filename = in_scenario_filename
        self.__outputs_path = in_outputs_path

        # Find results_folder associated with a given batch_file (and get most recent [-1])
        results_folder = get_scenario_outputs(self.__scenario_filename, self.__outputs_path)[-1]
        results_parent_folder_name = str(results_folder.parent)
        results_folder_name = results_folder.name
        # Get the datestamp
        if results_folder_name.startswith(scenario_filename + '-'):
            self.datestamp = results_folder_name[(len(scenario_filename)+1):]
        else:
            print("The scenario output name does not correspond with the set scenario_filename.")

        # Path to the .log.gz file
        results_folder_path_run0_draw0 = results_parent_folder_name + '/' + results_folder_name + '/0/0/'
        results_file_name_prefix = scenario_filename
        results_file_name_extension = '.log.gz'
        gz_results_file_path = Path(glob.glob(os.path.join(results_folder_path_run0_draw0,
                                                           f"{results_file_name_prefix}*{results_file_name_extension}"))[0])

        # Path to the decompressed .log file
        log_results_file_path = gz_results_file_path.with_suffix('')

        # Decompress the .log.gz file
        with gzip.open(gz_results_file_path, 'rb') as f_in:
            with open(log_results_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        self.__log_file_path = log_results_file_path
        # parse wasting logs
        self.__logs_dict = \
            parse_log_file(self.__log_file_path)['tlo.methods.wasting']

        # gender description
        self.__gender_desc = {'M': 'Males',
                              'F': 'Females'}

        # wasting types description
        self.__wasting_types_desc = {'WHZ<-3': 'severe wasting',
                                     '-3<=WHZ<-2': 'moderate wasting',
                                     'WHZ>=-2': 'not undernourished'}

        self.fig_files = []
        self.type_of_individual_figs = 'png'

    def plot_wasting_incidence(self):
        """ plot the incidence of wasting over time """
        w_inc_df = self.__logs_dict['wasting_incidence_count']
        w_inc_df.set_index(w_inc_df.date.dt.year, inplace=True)
        w_inc_df.drop(columns='date', inplace=True)
        # get age year. doesn't matter what wasting category you choose for
        # they all have same age groups
        age_years = list(w_inc_df.loc[w_inc_df.index[0], 'WHZ<-3'].keys(

        ))
        age_years.remove('5+y')

        _row_counter = 0
        _col_counter = 0
        # plot setup
        fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(10, 6))
        for _age in age_years:
            new_df = pd.DataFrame()
            for state in w_inc_df.columns:
                new_df[state] = \
                    w_inc_df.apply(lambda row: row[state][_age], axis=1)

            new_df = new_df.apply(lambda _row: _row / _row.sum(), axis=1)
            plotting = new_df[["WHZ<-3", "-3<=WHZ<-2"]]
            # convert into proportions
            ax = plotting.plot(kind='bar', stacked=True,
                               ax=axes[_row_counter, _col_counter],
                               title=f"incidence of wasting in {_age} old",
                               ylim=[0, 1])
            ax.legend(self.__wasting_types_desc.values(), loc='lower right')
            ax.set_xlabel('year')
            ax.set_ylabel('proportion')
            # move to another row
            if _col_counter == 2:
                _row_counter += 1
                _col_counter = -1
            _col_counter += 1  # increment column counter
            fig.tight_layout()
        fig_output_name = (str(outputs_path) + '/wasting_incidence__' + self.datestamp + '.' +
                           self.type_of_individual_figs)
        fig.savefig(fig_output_name, format=self.type_of_individual_figs)
        self.fig_files.append(fig_output_name)
        plt.show()

    def plot_wasting_prevalence_per_year(self):
        """ plot wasting prevalence of all age groups per year. Proportions are obtained by getting a total number of
        children wasted divide by the total number of children less than 5 years"""
        w_prev_df = self.__logs_dict["wasting_prevalence_props"]
        w_prev_df = w_prev_df[['date', 'total_under5_prop']]
        w_prev_df = w_prev_df.set_index(w_prev_df.date.dt.year)
        w_prev_df = w_prev_df.drop(columns='date')
        fig, ax = plt.subplots()
        w_prev_df["total_under5_prop"].plot(kind='bar', stacked=True,
                                            ax=ax,
                                            title="Wasting prevalence in children 0-59 months per year",
                                            ylabel='proportion of wasted children in the year',
                                            xlabel='year',
                                            ylim=[0, 0.15])
        # add_footnote(fig, "proportion of wasted children within each age-group")
        plt.tight_layout()
        fig_output_name = (str(outputs_path) + '/wasting_prevalence_per_year__' + self.datestamp + '.' +
                           self.type_of_individual_figs)
        fig.savefig(fig_output_name, format=self.type_of_individual_figs)
        self.fig_files.append(fig_output_name)
        plt.show()

    def plot_wasting_prevalence_by_age_group(self):
        """ plot wasting prevalence per each age group. Proportions are obtained by getting a total number of
        children wasted in a particular age-group divide by the total number of children per that age-group"""
        w_prev_df = self.__logs_dict["wasting_prevalence_props"]
        w_prev_df = w_prev_df.drop(columns={'total_under5_prop'})
        w_prev_df = w_prev_df.set_index(w_prev_df.date.dt.year)
        w_prev_df = w_prev_df.loc[w_prev_df.index == 2023]
        w_prev_df = w_prev_df.drop(columns='date')
        print(f"{w_prev_df=}")
        order_x_axis = ['0_5mo', '6_11mo', '12_23mo', '24_35mo', '36_47mo', '48_59mo']
        # Assert that all columns are included
        assert set(w_prev_df.columns) == set(order_x_axis), "Not all columns are included in the order_x_axis."
        w_prev_df = w_prev_df[order_x_axis]

        fig, ax = plt.subplots(figsize=(10, 6))
        # plot wasting prevalence
        w_prev_df.squeeze().plot(kind='bar', stacked=False,
                                 ax=ax,
                                 title="Wasting prevalence in children 0-59 months per each age group in 2023",
                                 ylabel='proportion',
                                 xlabel='age group',
                                 ylim=[0, 0.3])
        # Adjust the layout to make space for the footnote
        plt.subplots_adjust(bottom=0.85)  # Adjust the bottom margin
        # Add footnote
        fig.figure.text(0.45, 0.88,
                        "proportion = number of wasted children in the age group "
                        "/ total number of children in the age group",
                        ha="center", fontsize=10, bbox={"facecolor": "gray", "alpha": 0.3, "pad": 5})
        plt.tight_layout()
        fig_output_name = (str(outputs_path) + '/wasting_prevalence_per_each_age_group__' + self.datestamp + '.'
                           + self.type_of_individual_figs)
        fig.savefig(fig_output_name, format=self.type_of_individual_figs)
        self.fig_files.append(fig_output_name)
        plt.show()

    def plot_modal_gbd_deaths_by_gender(self):
        """ compare modal and GBD deaths by gender """
        death_compare = \
            compare_number_of_deaths(self.__log_file_path, resources_path)
        fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(10, 6))
        for _col, sex in enumerate(('M', 'F')):
            plot_df = death_compare.loc[(['2010-2014', '2015-2019'],
                                         sex, slice(None), 'Childhood Undernutrition'
                                         )].groupby('period').sum()
            plotting = plot_df.loc[['2010-2014', '2015-2019']]
            ax = plotting['model'].plot.bar(label='Model', ax=axs[_col], rot=0)
            ax.errorbar(x=plotting['model'].index, y=plotting.GBD_mean,
                        yerr=[plotting.GBD_lower, plotting.GBD_upper],
                        fmt='o', color='#000', label="GBD")
            ax.set_title(f'{self.__gender_desc[sex]} '
                         f'wasting deaths, 2010-2014')
            ax.set_xlabel("Time period")
            ax.set_ylabel("Number of deaths")
            ax.legend(loc=2)
        fig.tight_layout()
        # Adjust the layout to make space for the footnote
        plt.subplots_adjust(bottom=0.15)  # Adjust the bottom margin
        # Add footnote
        fig.figure.text(0.5, 0.02,
                        "Model output against Global Burden of Diseases (GDB) study data",
                        ha="center", fontsize=10, bbox={"facecolor": "gray", "alpha": 0.3, "pad": 5})
        fig_output_name = (str(outputs_path) + '/modal_gbd_deaths_by_gender__' + self.datestamp + '.' +
                           self.type_of_individual_figs)
        fig.savefig(fig_output_name, format=self.type_of_individual_figs)
        self.fig_files.append(fig_output_name)
        plt.show()

    def plot_all_figs_in_one_pdf(self):

        output_file_path = str(self.__outputs_path) + '/wasting_all_figures__' + self.datestamp + '.pdf'
        # Remove the existing output file if it exists to ensure a clean start
        if os.path.exists(output_file_path):
            os.remove(output_file_path)

        # Assert that the file doesn't exist anymore after removal
        assert not os.path.exists(output_file_path), "The file was not successfully removed."

        # Create instance of FPDF class
        pdf = FPDF()

        # Standard A4 page size in millimeters
        a4_width_mm = 210
        a4_height_mm = 297

        # Iterate through the figure files and add each as a new page
        for figure_file in self.fig_files:
            # Open the figure file
            figure = Image.open(figure_file)

            # Convert the figure to RGB mode if it's not already
            if figure.mode != 'RGB':
                figure = figure.convert('RGB')

            # Get the size of the figure
            width, height = figure.size

            # Convert pixels to millimeters (1 pixel = 0.264583 mm)
            width_mm = width * 0.264583
            height_mm = height * 0.264583

            # Calculate the scaling factor to fit the figure within A4 dimensions
            scale_factor = min(a4_width_mm / width_mm, a4_height_mm / height_mm)

            # Calculate the new dimensions of the figure
            new_width_mm = width_mm * scale_factor
            new_height_mm = height_mm * scale_factor

            # Add a new page to the PDF
            pdf.add_page()

            # Center the figure on the page
            x_offset = (a4_width_mm - new_width_mm) / 2
            y_offset = (a4_height_mm - new_height_mm) / 2

            # Add the figure to the page
            pdf.image(figure_file, x=x_offset, y=y_offset, w=new_width_mm, h=new_height_mm)

        # Save the PDF to a file
        pdf.output(output_file_path)


if __name__ == "__main__":

    # Path to the resource files used by the disease and intervention methods
    resources_path = Path("./resources")

    # initialise the wasting class
    wasting_analyses = WastingAnalyses(scenario_filename, outputs_path)

    # plot wasting incidence
    wasting_analyses.plot_wasting_incidence()

    # plot wasting prevalence
    wasting_analyses.plot_wasting_prevalence_per_year()

    # plot wasting prevalence by age group
    wasting_analyses.plot_wasting_prevalence_by_age_group()

    # plot wasting deaths by gender as compared to GBD deaths
    wasting_analyses.plot_modal_gbd_deaths_by_gender()

    # save all figures in one pdf
    wasting_analyses.plot_all_figs_in_one_pdf()
