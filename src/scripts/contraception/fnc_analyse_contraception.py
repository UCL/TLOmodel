"""
a function 'analyse_contraception' defined to be used for pre-simulated data (using the scenario file
run_analysis_contraception_no_diseases.py) called from the script analysis_contraception_plot_table.py, plots use of
contraception over time, use of contraception methods over time, pregnancies over time, dependency ratio over time,
and/or calculates data for a table of use and costs of contraception methods and interventions (whichever are required
according to the setting in the analysis_contraception_plot_table.py script).
"""
import logging
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import extract_results, parse_log_file, summarize


def analyse_contraception(in_id: str, in_log_file: str, in_suffix: str,
                          in_plot_use_time_bool: bool = False,
                          in_plot_use_time_method_bool: bool = False,
                          in_plot_pregnancies_bool: bool = False,
                          in_plot_depend_ratio_bool: bool = False,
                          in_set_ylims_bool: bool = False, in_ylims_l: list = [],
                          in_contraceptives_order: list = ['pill', 'IUD', 'injections', 'implant', 'male_condom',
                                                           'female_sterilization', 'other_modern'],
                          in_calc_use_costs_bool: bool = False, in_required_time_period_starts: list = [],
                          in_calc_intervention_costs_bool: bool = False,
                          in_use_output: str = "mean"
                          ):
    """
    Performs analysis of contraception for pre-simulated data (data given by
    'in_log_file'), saves figure(s) and/or calculates contraception use and
    costs to be included in a table, according to what is (not) required
    (requirements set by inputs in_xx_bool). The name of output figs includes
     'in_id' to be assigned to correct simulations.

    :param in_id: simulation id to be included in output files names
    :param in_log_file: log file from which the simulations logging is
        downloaded
    :param in_suffix: A suffix added to the end of Figs output filenames.
    :param in_plot_use_time_bool: True if we want to plot use of any
        contraception over time (default: False)
    :param in_plot_use_time_method_bool: True if we want to plot use of
        individual contraception methods over time (default: False)
    :param in_plot_pregnancies_bool: True if we want to plot pregnancies over
        time (default: False)
    :param in_plot_depend_ratio_bool: True if we want to plot dependency ratio
        over time (default: False)
    :param in_set_ylims_bool: True if we want to set upper limits for the y-axes
        for the 3 plots. (default: False)
    :param in_ylims_l: list of the upper limits for y-axes of the figures in the
        order [Use, Use By Method, Pregnancies] (default: [] -- as we don't need
        it if 'in_set_ylims_bool' is False)
    :param in_contraceptives_order: list of modern contraceptives ordered as we
        want them to appear in the plots and tables
    :param in_calc_use_costs_bool: True if we want to calculate use and costs of
        contraception methods in time periods (time periods
        'in_required_time_period_starts' needs to be given as input)
        (default: False)
    :param in_required_time_period_starts: a list of years determining the time
        periods for which we require the calculations, first year inc.,
        last year excl. (default: [] -- as we don't need it if
        'in_calc_use_costs_bool' is False)
    :param in_calc_intervention_costs_bool: True if we want to calculate
        contraception Pop and PPFP intervention costs over time (default: False)
    :param in_use_output: "mean" or "max", according to which output of numbers,
        and percentage of women using contraception methods we want to display
        in the table (default: "mean")


    :return: Four data frames by time periods:
        * number of women using contraception methods,
        * percentage of women using contraception methods,
        * costs of contraception methods,
        (if 'in_calc_use_costs_bool' is False, returns 3 empty lists for the above)
        * costs of contraception interventions (Pop, PPFP, Pop+PPFP)
        (if 'in_calc_use_costs_bool' or 'in_calc_intervention_costs_bool' is False,
        returns an empty DataFrame for the above)
    """

    # Where will outputs go - by default, wherever this script is run
    outputpath = Path("./outputs")  # folder for convenience of storing outputs

    # Load without simulating again - parse the simulation logfile to get the
    # output dataframes
    log_df = parse_log_file('outputs/' + in_log_file, level=logging.DEBUG)
    # last year simulated
    co_sum_df = log_df['tlo.methods.contraception']['contraception_use_summary'].copy()
    co_sum_df['year'] = co_sum_df['date'].dt.year
    last_year_simulated = co_sum_df.loc[co_sum_df.shape[0] - 1, 'year']
    last_day_simulated = co_sum_df.loc[co_sum_df.shape[0] - 1, 'date']
    # Load scaling factor to rescale nmbs from simulated pop_size to pop size of Malawi
    df_scale = log_df['tlo.methods.population']['scaling_factor'].set_index('date').copy()
    scaling_factor = df_scale.loc['2010-01-01', 'scaling_factor']

    # Define line styles
    if not in_calc_intervention_costs_bool:  # used as approximation of sim without interv
        # => TODO: fix for an option of when we analyse sim with intervention without this calculation
        line_style = '-'
    else:
        line_style = '--'
    ls_start_interv = ':'

    def rgb_perc(nmb_r, nmb_g, nmb_b):
        return nmb_r/255, nmb_g/255, nmb_b/255
        """
        Converts 0-255 RGB colors to 0-1 RGB.
        """

    # %% Plot Contraception Use (By Method) Over Time?
    if in_plot_use_time_bool or in_plot_use_time_method_bool or in_plot_pregnancies_bool:

        # Load Model Results
        co_df = log_df['tlo.methods.contraception']['contraception_use_summary'].set_index('date').copy()
        model_months = pd.to_datetime(co_df.index)
        # Keep only data up to 2050
        if (model_months.year[-1]) > 2050:
            plot_months = model_months[model_months.year <= 2050]
        else:
            plot_months = model_months
            if (model_months.year[-1]) < 2050:
                # warn that the sim ended before 2050, hence the plots will be prepared till then only
                warnings.warn(
                    '\nWarning: The simulation ended before the year 2050, specifically in ' +
                    str(model_months.year[-1]) + ', hence all the figs are till then only.')

        # %% Plot Contraception Use Over time:
        if in_plot_use_time_bool:

            # Load Model Results
            women1549_total = co_df.sum(axis=1)[0:len(plot_months)]
            women_not_using = co_df.not_using[0:len(plot_months)]
            women_using = women1549_total - women_not_using

            # Plot total values
            fig, ax = plt.subplots()
            ax.plot(np.asarray(plot_months), women1549_total * scaling_factor, ls=line_style)
            ax.plot(np.asarray(plot_months), women_not_using * scaling_factor, ls=line_style)
            ax.plot(np.asarray(plot_months), women_using * scaling_factor, color=(rgb_perc(51, 160, 44)), ls=line_style)
            plt.axvline(x=Date(2023, 1, 1), ls=ls_start_interv, color='gray', label='interventions start')
            if in_set_ylims_bool:
                ax.set_ylim([0, in_ylims_l[0]])
            plt.title("Contraception Use")
            plt.xlabel("Year")
            plt.ylabel("Number of women")
            # plt.gca().set_xlim(Date(2010, 1, 1), Date(2023, 1, 1)) to see only 2010-2023 (excl)
            plt.legend(['Total women age 15-49 years', 'Not Using Contraception', 'Using Contraception'])
            plt.savefig(outputpath / ('Contraception Use ' + in_id + "_UpTo" + str(plot_months.year[-1]) + in_suffix
                                      + '.png'), format='png')

            # Plot proportions within 15-49 population
            fig, ax = plt.subplots()
            # # Since when (incl) are more than 50% women using
            # women_using_prop = women_using / women1549_total
            # women_using_prop_gt_half = women_using_prop[women_using_prop.gt(0.5)].index[0]
            # print("Since when (incl) are more than 50% women using")
            # print(women_using_prop_gt_half)
            ax.plot(np.asarray(plot_months), women_using / women1549_total, ls=line_style)
            plt.axvline(x=Date(2023, 1, 1), ls=ls_start_interv, color='gray', label='interventions start')
            if in_set_ylims_bool:
                ax.set_ylim([0, in_ylims_l[1]])
            plt.title("Proportion of Females 15-49 Using Contraceptive over Time")
            plt.xlabel("Year")
            plt.ylabel('Proportion')
            plt.savefig(outputpath / ('Prop Fem1549 Using Contraceptive Over Time ' + in_id +
                                      "_UpTo" + str(plot_months.year[-1]) + in_suffix + '.png'), format='png')

            print("Figs: Contraception Use Over time saved.")

        # %% Plot Contraception Use By Method Over time:
        if in_plot_use_time_method_bool:

            # Load Model Results
            Model_pill = co_df.pill[0:len(plot_months)]
            Model_IUD = co_df.IUD[0:len(plot_months)]
            Model_injections = co_df.injections[0:len(plot_months)]
            Model_implant = co_df.implant[0:len(plot_months)]
            Model_male_condom = co_df.male_condom[0:len(plot_months)]
            Model_female_sterilization = co_df.female_sterilization[0:len(plot_months)]
            Model_other_modern = co_df.other_modern[0:len(plot_months)]
            Model_periodic_abstinence = co_df.periodic_abstinence[0:len(plot_months)]
            Model_withdrawal = co_df.withdrawal[0:len(plot_months)]
            Model_other_traditional = co_df.other_traditional[0:len(plot_months)]

            # TODO: add comments with names of the colours
            # Define colours for all contraception methods
            colours_all_meths = [rgb_perc(166, 206, 227),
                                 rgb_perc(227, 26, 28),
                                 rgb_perc(51, 160, 44),
                                 rgb_perc(253, 191, 111),
                                 rgb_perc(31, 120, 180),
                                 rgb_perc(255, 127, 0),
                                 rgb_perc(178, 223, 138),
                                 rgb_perc(251, 154, 153),
                                 rgb_perc(202, 178, 214),
                                 rgb_perc(106, 61, 154)]
            # TODO: Find better way to use own colour palette.

            # Plot absolut values
            fig, ax = plt.subplots()
            ax.plot(np.asarray(plot_months), Model_pill * scaling_factor, color=colours_all_meths[0],
                    ls=line_style)
            ax.plot(np.asarray(plot_months), Model_IUD * scaling_factor, color=colours_all_meths[1],
                    ls=line_style)
            ax.plot(np.asarray(plot_months), Model_injections * scaling_factor, color=colours_all_meths[2],
                    ls=line_style)
            ax.plot(np.asarray(plot_months), Model_implant * scaling_factor, color=colours_all_meths[3],
                    ls=line_style)
            ax.plot(np.asarray(plot_months), Model_male_condom * scaling_factor, color=colours_all_meths[4],
                    ls=line_style)
            ax.plot(np.asarray(plot_months), Model_female_sterilization * scaling_factor, color=colours_all_meths[5],
                    ls=line_style)
            ax.plot(np.asarray(plot_months), Model_other_modern * scaling_factor, color=colours_all_meths[6],
                    ls=line_style)
            ax.plot(np.asarray(plot_months), Model_periodic_abstinence * scaling_factor, color=colours_all_meths[7],
                    ls=line_style)
            ax.plot(np.asarray(plot_months), Model_withdrawal * scaling_factor, color=colours_all_meths[8],
                    ls=line_style)
            ax.plot(np.asarray(plot_months), Model_other_traditional * scaling_factor, color=colours_all_meths[9],
                    ls=line_style)
            plt.axvline(x=Date(2023, 1, 1), ls=ls_start_interv, color='gray', label='interventions start')
            if in_set_ylims_bool:
                ax.set_ylim([0, in_ylims_l[2]])
            contraceptives_order_all_meths = in_contraceptives_order +\
                ['periodic_abstinence', 'withdrawal', 'other_traditional']
            # TODO: make the order of non-modern methods as input parameter
            #  (then join ordered modern & non-modern methods)
            plt.title("Contraception Use By Method")
            plt.xlabel("Year")
            plt.ylabel("Number of women using method")
            plt.legend(contraceptives_order_all_meths)
            plt.savefig(outputpath / ('Contraception Use By Method ' + in_id +
                                      "_UpTo" + str(plot_months.year[-1]) + in_suffix + '.png'), format='png')

            # Plot proportions within 15-49 population
            def get_annual_mean_usage(_df):
                _x = _df \
                    .assign(year=_df['date'].dt.year) \
                    .set_index('year') \
                    .drop(columns=['date']) \
                    .apply(lambda row: row / row.sum(),
                           axis=1
                           )
                return _x.groupby(_x.index).mean().stack()

            if in_log_file == 'run_analysis_contraception_no_diseases__2023-05-06T170512.log':
                # without interv, 250K till 2050; final costs update EHP & OHT + rebased on master
                # + pregn test corrected
                results_folder_name = 'run_analysis_contraception_no_diseases-2023-05-06T170253Z'
            elif in_log_file == 'run_analysis_contraception_no_diseases__2023-05-06T170612.log':
                # with interv, 250K till 2050; final costs update EHP & OHT + rebased on master + pregn test corrected
                results_folder_name = 'run_analysis_contraception_no_diseases-2023-05-06T170359Z'
            elif in_log_file == 'run_analysis_contraception_no_diseases__2023-04-26T141435.log':
                # without interv, 2K till 2099, final costs update EHP & OHT + pregn test to initiate co
                results_folder_name = 'run_analysis_contraception_no_diseases-2023-04-26T141159Z'
            elif in_log_file == 'run_analysis_contraception_no_diseases__2023-04-26T141545.log':
                # with interv, 2K till 2099, final costs update EHP & OHT + pregn test to initiate co
                results_folder_name = 'run_analysis_contraception_no_diseases-2023-04-26T141321ZZ'
            else:
                raise ValueError(
                    "Unknown results_folder_name for the log file " + str(in_log_file) +
                    ". Needs to be defined in the code for Figs: Contraception Use By Method Over time."
                )
            results_folder = Path('./outputs/sejjej5@ucl.ac.uk/' + results_folder_name)
            # TODO: make the whole analysis to take the results from the folder
            #  (hence no need of extracting the log file)

            mean_usage = summarize(extract_results(results_folder,
                                                   module="tlo.methods.contraception",
                                                   key="contraception_use_summary",
                                                   custom_generate_series=get_annual_mean_usage,
                                                   do_scaling=False),
                                   collapse_columns=True,
                                   only_mean=True
                                   ).unstack()

            # print("Mean usage of injections")
            # print("by 2030: " + str(mean_usage.loc[2030, 'injections']) + "; by 2050: "
            #       + str(mean_usage.loc[2050, 'injections']))

            contraceptives_order_notusing_all_meths = ["not_using"]
            contraceptives_order_notusing_all_meths.extend(contraceptives_order_all_meths)
            # Define a colour for not using
            colours_notusing_all_meths = [rgb_perc(255, 255, 153)]  # pale canary yellow green ~ ie light yellow
            colours_notusing_all_meths.extend(colours_all_meths)

            # Keep only data up to 2050
            mean_usage = mean_usage[0:(2050-2010+1)]
            # Reverse methods so the last method is plotted lowest
            mean_usage = mean_usage.loc[:, reversed(contraceptives_order_notusing_all_meths)]
            mean_usage = mean_usage.loc[:, reversed(contraceptives_order_notusing_all_meths)]

            fig, ax = plt.subplots()
            # colours defined in the same order as methods, hence need to be reversed too
            mean_usage.plot.area(stacked=True, ax=ax, legend=False, color=list(reversed(colours_notusing_all_meths)))
            plt.axvline(x=2023, ls=ls_start_interv, color='white')
            plt.title('Proportions of Females 15-49 Using Contraception Methods', x=0.7)
            plt.xlabel('Year')
            plt.ylabel('Proportion')
            # Move the fig title, so it fits with others in the panel
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles[::-1], labels[::-1], title='Contraception Method', loc=7)
            fig.subplots_adjust(right=0.65)
            plt.savefig(outputpath / ('Prop Fem1549 Using Method ' + in_id +
                                      "_UpTo" + str(plot_months.year[-1]) + in_suffix + '.png'), format='png')

            print("Figs: Contraception Use By Method Over time saved.")

        # %% Plot Pregnancies Over time:
        if in_plot_pregnancies_bool:

            # Load Model Results by Months up to 2050
            women1549_total = co_df.sum(axis=1)[0:len(plot_months)]
            preg_df_by_months = log_df['tlo.methods.contraception']['pregnancy'].set_index('date').copy()
            if preg_df_by_months.index.year[-1] > 2050:
                preg_df_by_months = preg_df_by_months[preg_df_by_months.index.year <= 2050]
            # If not warn yet, warn that the sim ended before 2050, hence the plots will be prepared till then only
            elif preg_df_by_months.index.year[-1] < 2050\
                    and not in_plot_use_time_bool and not in_plot_use_time_method_bool:
                warnings.warn(
                    '\nWarning: The simulation ended before the year 2050, specifically in ' +
                    str(preg_df_by_months.index.year[-1]) + ', hence all the figs are till then only.')
            # Create Data by Years (NB. Figs by Months are too noisy.)
            preg_df_by_years = preg_df_by_months.copy()
            preg_df_by_years.index = pd.to_datetime(preg_df_by_years.index).year
            num_pregs_by_year = preg_df_by_years.groupby(by=preg_df_by_years.index).size()
            plot_years = num_pregs_by_year.index
            pregnancy_by_years = num_pregs_by_year.values

            # Plot total pregnancies per Year
            fig, ax = plt.subplots()
            ax.plot(np.asarray(plot_years), pregnancy_by_years * scaling_factor, ls=line_style)
            plt.axvline(x=2023, ls=ls_start_interv, color='gray', label='interventions start')
            if in_set_ylims_bool:
                ax.set_ylim([0, in_ylims_l[3]])
            plt.title("Pregnancies per Year")
            plt.xlabel("Year")
            plt.ylabel("Number of pregnancies")
            plt.savefig(outputpath / ('Pregnancies per Year ' + in_id +
                                      "_UpTo" + str(plot_years[-1]) + in_suffix + '.png'), format='png')

            # Calculate Means of Pregnancies Proportions within Women 15-49 per Year
            # (women1549_total are monthly data, hence pregnancy monthly data used to calculate the means )
            num_pregs_by_months = preg_df_by_months.groupby(by=preg_df_by_months.index).size()
            model_pregnancy_by_month = num_pregs_by_months.values
            preg_props = model_pregnancy_by_month / women1549_total
            preg_props.index = pd.to_datetime(preg_props.index).year
            mean_preg_props_by_year = preg_props.groupby(by=preg_props.index).mean()

            # Plot mean proportion of pregnancies in 15-49 women pop per Years
            fig, ax = plt.subplots()
            ax.plot(np.asarray(plot_years), mean_preg_props_by_year, ls=line_style)
            plt.axvline(x=2023, ls=ls_start_interv, color='gray', label='interventions start')
            if in_set_ylims_bool:
                ax.set_ylim([0, in_ylims_l[4]])
            plt.title("Mean Proportion of Females 15-49 Becoming Pregnant per Year")
            plt.xlabel("Year")
            plt.ylabel("Mean proportion of pregnancies")
            plt.savefig(outputpath / ('Mean Prop of Fem15-49 Becom Preg per Year ' + in_id +
                                      "_UpTo" + str(plot_years[-1]) + in_suffix + '.png'), format='png')

            print("Figs: Pregnancies Over time saved.")

    # %% Plot Dependency Ratio Over time:
    if in_plot_depend_ratio_bool:

        # Load Demographic Results by Age groups by Years up to 2050
        demog_df_f_by_years = log_df['tlo.methods.demography']['age_range_f'].set_index('date').copy()
        demog_df_m_by_years = log_df['tlo.methods.demography']['age_range_m'].set_index('date').copy()

        def demog_gend_labor(in_demog_gend_df_by_years):
            """
            :param in_demog_gend_df_by_years: A dataframe with demographic data for a gender (age groups:
                0-4  5-9  10-14  15-19  20-24  25-29  30-34  35-39  40-44  45-49  50-54  55-59  60-64  65-69  70-74
                75-79 80-84  85-89
            :return: The dataframe with two new columns: 'labor_force' = nmb of individuals of the gender in the labor
                force (age of 15-64), and 'non-labor_force' = the nmb of individuals who are not typically in the labor
                force (age of 0-14 or 65+).
            """
            out_demog_gend_df_by_years = in_demog_gend_df_by_years.copy()
            if out_demog_gend_df_by_years.index.year[-1] > 2050:
                out_demog_gend_df_by_years = out_demog_gend_df_by_years[out_demog_gend_df_by_years.index.year <= 2050]
            out_demog_gend_df_by_years['year'] = pd.to_datetime(out_demog_gend_df_by_years.index).year
            out_demog_gend_df_by_years.index = pd.to_datetime(out_demog_gend_df_by_years.index).year
            out_demog_gend_df_by_years.index.name = 'year'
            out_demog_gend_df_by_years['labor_force'] =\
                out_demog_gend_df_by_years.loc[:, ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49',
                                                   '50-54', '55-59', '60-64']].sum(axis=1)
            out_demog_gend_df_by_years['non-labor_force'] =\
                out_demog_gend_df_by_years.loc[:, ['0-4', '5-9', '10-14', '65-69', '70-74', '75-79', '80-84', '85-89']]\
                .sum(axis=1)
            return out_demog_gend_df_by_years

        # Add total females & males labor force & non-labor force
        demog_df_f_by_years = demog_gend_labor(demog_df_f_by_years)
        demog_df_m_by_years = demog_gend_labor(demog_df_m_by_years)

        # Calculate dependency ratio by Years
        popsize_labor_force = demog_df_f_by_years['labor_force'] + demog_df_m_by_years['labor_force']
        popsize_nonlabor_force = demog_df_f_by_years['non-labor_force'] + demog_df_m_by_years['non-labor_force']
        # total females & males non-labor force:
        depend_ratio_df = pd.DataFrame({"Dependency ratio": popsize_nonlabor_force/popsize_labor_force})
        plot_years = depend_ratio_df.index

        # Plot dependency ratio by Years
        fig, ax = plt.subplots()
        ax.plot(np.asarray(plot_years), depend_ratio_df, ls=line_style)
        plt.axvline(x=2023, ls=ls_start_interv, color='gray', label='interventions start')
        if in_set_ylims_bool:
            ax.set_ylim([0, in_ylims_l[5]])
        plt.title("Dependency Ratio per Year")
        plt.xlabel("Year")
        plt.ylabel("Dependency ratio")
        plt.savefig(outputpath / ('Dependency Ratio over Time ' + in_id +
                                  "_UpTo" + str(plot_years[-1]) + in_suffix + '.png'), format='png')

        print("Fig: Dependency Ratio Over time saved.")

    # TODO: create the multipanel figs (Fig 3, Fig A6.1)

    # %% Calculate Use and Consumables Costs of Contraception methods within
    # some time periods:
    if in_calc_use_costs_bool:
        # time period starts should be given as input
        assert in_required_time_period_starts != [], \
            "The calculations of use and costs are requested (ie input 'in_calc_use_costs_bool' set as True, " +\
            "but no periods starts are provided (ie input 'TimePeriods_starts' is empty)."

        # this input needs to include at least 3 values
        assert len([y for y in in_required_time_period_starts if y <= last_year_simulated + 1]) > 2, \
            "The input 'TimePeriods_starts' needs to include at least 3 years within simulated range + 1 year, ie up " \
            "to " + str(last_year_simulated + 1) + ", to define at least 2 time periods."
        # time period starts should be ordered
        assert all(in_required_time_period_starts[i] <= in_required_time_period_starts[i + 1]
                   for i in range(len(in_required_time_period_starts) - 1)), \
            "The 'TimePeriods_starts' needs to be in chronological order."
#  ###### USE ##################################################################
        # Load Contraception Use Results
        # ['date', 'IUD', 'female_sterilization', 'implant', 'injections',
        # 'male_condom', 'not_using', 'other_modern', 'other_traditional',
        # 'periodic_abstinence', 'pill', 'withdrawal']:
        co_use_df = log_df['tlo.methods.contraception']['contraception_use_summary'].copy()
        co_use_df['women_total'] = co_use_df.sum(axis=1)
        cols_to_keep = in_contraceptives_order.copy()
        cols_to_keep.append('date')
        cols_to_keep.append('women_total')
        co_use_modern_df = co_use_df.loc[:, cols_to_keep].copy()
        co_use_modern_df['co_modern_total'] = co_use_modern_df.loc[:, in_contraceptives_order].sum(axis=1)
        co_use_modern_df['year'] = co_use_modern_df['date'].dt.year
        print("Nmb of women at the initiation:", co_use_df['women_total'][0])
        print("Years simulated:",
              co_use_modern_df.loc[1, 'year'], "-", last_year_simulated)
        print()

        # Keep only records within required time periods and assign the time
        # periods they belong to
        def assign_time_period(in_year, in_l_time_period_start):
            """
            For an input year returns a time period to which the year belongs
            according to the list of time period starts (in_l_time_period_start).
            The input year should be within these time periods.

            :param in_year: a year
            :param in_l_time_period_start: list of time period starts, dividing
                individual time periods

            :return: A time period to which the input year belongs.
            """

            time_period_pos = next(i for i, v in enumerate(in_l_time_period_start) if v > in_year) - 1
            return str(in_l_time_period_start[time_period_pos]) + "-"\
                + str(in_l_time_period_start[time_period_pos+1] - 1)

        def create_time_period_data(in_l_time_period_start, in_df):
            """
            Keeps only data within the required time periods and then assigns
            each record the time period it belongs to, based on the 'year' the
            record was performed.

            :param in_l_time_period_start: a list of starts of the time periods,
                first year incl., last year excl.,
            :param in_df: the data with records that include the column 'year'
                when the record was performed

            :return: A new data frame that includes only records from the required
                time periods, and includes a new column 'Time_Period'.
            """
            # Keep only the data from the required time periods
            tp_df = in_df.loc[(in_l_time_period_start[0] <= in_df['year']) &
                              (in_df['year'] < in_l_time_period_start[-1])].copy()
            tp_df['Time_Period'] = \
                tp_df['year'].apply(assign_time_period,
                                    in_l_time_period_start=in_l_time_period_start)
            return tp_df

        co_use_modern_tp_df = \
            create_time_period_data(in_required_time_period_starts,
                                    co_use_modern_df)

        co_use_modern_tp_df = \
            co_use_modern_tp_df.loc[
                :, ((co_use_modern_tp_df.columns != 'date') & (co_use_modern_tp_df.columns != 'year'))
            ]

        def create_percentage_use_df(in_df_use_incl_women_total):
            """
            Based on mean/max use creates a data frame with mean/max percentage
            use.

            :param in_df_use_incl_women_total: data frame of contraceptive use,
                grouped by 'Time_Period', for all modern contraception methods,
                including a column 'women_total'

            :return: Data frame grouped by 'Time_Period' of percentage use for
                all modern contraception methods, including a column
                'women_total' (ie 100%).
            """
            df_percentage_use = in_df_use_incl_women_total.copy()
            df_percentage_use.iloc[:, :-1] = df_percentage_use.iloc[:, :-1]\
                .div(df_percentage_use['women_total'], axis=0).mul(100, axis=0)
            # we can exclude the column with women_total as it is no more
            # needed
            return df_percentage_use.loc[:, df_percentage_use.columns != 'women_total']

        co_percentage_use_df = create_percentage_use_df(co_use_modern_tp_df)
        # Remove women_total from co_use_modern_tp_df too
        co_use_modern_tp_df =\
            co_use_modern_tp_df.loc[:, co_use_modern_tp_df.columns != 'women_total']

        # Rescale the numbers of contraception use to the population size of Malawi
        # (from the nmbs for simulation pop_size)
        co_use_modern_tp_df.loc[:, co_use_modern_tp_df.columns != 'Time_Period'] =\
            co_use_modern_tp_df.loc[:, co_use_modern_tp_df.columns != 'Time_Period'] * scaling_factor

        def sum_use_all_times(in_df_use_by_tp, in_output_type):
            """
            Adds a row with mean/max use in all time periods.
            :param in_df_use_by_tp: 'Time_Period' as index, mean/max use of
                contraception methods in columns
            :param in_output_type: The type of output, "mean" or "max".

            :return: The sum row to append to the input df.
            """
            # tp of all times, ie very first to very last year of time periods
            y_first = in_df_use_by_tp.index[0].split("-")[0]
            y_last = in_df_use_by_tp.index[len(in_df_use_by_tp) - 1].split("-")[1]
            sum_tp = (str(y_first) + "-" + str(y_last))
            l_summation = []
            for c in in_df_use_by_tp:
                # outputs (min/max) for contraceptives within all times
                if in_output_type == "mean":
                    l_summation.append(in_df_use_by_tp[c].mean())
                if in_output_type == "max":
                    l_summation.append(in_df_use_by_tp[c].max())
            return pd.DataFrame([l_summation], columns=list(in_df_use_by_tp.columns), index=[sum_tp])

        # Output (default: "mean", may be changed to "max") of contraception use
        # within the tp (= time period)
        if in_use_output == "mean":
            co_output_use_modern_tp_df = \
                co_use_modern_tp_df.groupby('Time_Period').mean()
            # Store copy as mean_use to work with it separately
            mean_use_df = co_output_use_modern_tp_df.copy()
            # Include the output summation for all time periods:
            co_output_use_modern_tp_df = \
                co_output_use_modern_tp_df \
                .append(sum_use_all_times(co_output_use_modern_tp_df, in_use_output))

            co_output_percentage_use_df =\
                co_percentage_use_df.groupby('Time_Period').mean()
            # Include the output summation for all time periods:
            co_output_percentage_use_df = \
                co_output_percentage_use_df\
                .append(sum_use_all_times(co_output_percentage_use_df, in_use_output))
        elif in_use_output == "max":
            co_output_use_modern_tp_df = \
                co_use_modern_tp_df.groupby('Time_Period').max()
            co_output_use_modern_tp_df = \
                co_output_use_modern_tp_df \
                .append(sum_use_all_times(co_output_use_modern_tp_df, in_use_output))
            co_output_percentage_use_df =\
                co_percentage_use_df.groupby('Time_Period').max()
            co_output_percentage_use_df = \
                co_output_percentage_use_df\
                .append(sum_use_all_times(co_output_percentage_use_df, in_use_output))
            # we still need the mean use to calculate costs of condoms
            mean_use_df = \
                co_use_modern_tp_df.groupby('Time_Period').mean()
        else:
            raise ValueError(
                "Unrecognised use output:" + str(in_use_output) +
                ". The type of use output ('in_use_output') can only be 'mean' or 'max'."
            )

        co_output_use_modern_tp_df.index.name =\
            co_output_percentage_use_df.index.name = mean_use_df.index.name =\
            'Time_Period'

        print("Calculations of Contraception Methods Use finished.")

#  ###### CONSUMABLES COSTS ##########################################################
        # Add a column with the nmb of years within the time periods to mean_use_df
        def calculate_tp_len(in_tp_as_string):
            l_start_end_tp = [int(x) for x in in_tp_as_string.split("-")]
            return l_start_end_tp[1] - l_start_end_tp[0] + 1

        # Add length of time periods to mean_use_df
        mean_use_df['tp_len'] = mean_use_df.index.map(calculate_tp_len)

        # Load Consumables results
        cons_df = log_df['tlo.methods.contraception']['Contraception_consumables'].copy()
        cons_df['date'] = pd.to_datetime(cons_df['date'])
        cons_df['year'] = cons_df['date'].dt.year

        # Drop any entry that is not related to Contraception
        cons_df =\
            cons_df.loc[cons_df.TREATMENT_ID.str.startswith('Contraception')]

        # Get individual requests from which some items were Available,
        # some NotAvailable.
        def merge_dicts(in_l_dicts):
            """
            Merges a list of dictionaries into one dictionary.

            :param in_l_dicts: list of dictionaries
            :return: One dictionary.
            """
            c = Counter()
            for d in in_l_dicts:
                c.update(d)
            return dict(c)

        def join_avail_notavail_items(in_df):
            l_requests = []
            for i in range(len(in_df.index)):
                l_requests.append(merge_dicts([
                    eval(in_df.loc[i]['Item_Available']),
                    eval(in_df.loc[i]['Item_NotAvailable'])
                    ])
                )
            return l_requests

        cons_df['Request'] = join_avail_notavail_items(cons_df)

        # Limit consumables data to those which were processed (contraception
        # was given to a woman as all items were available)
        # TODO: make it to work with essential and optional items
        #  (here we assume that all requested items are essential)
        cons_processed_df = cons_df.loc[cons_df['Item_NotAvailable'] == "{}"].copy()

        # Assign a contraceptive method to each record according to the request.
        resource_items_pkgs_df = pd.read_csv(
            'resources/healthsystem/consumables/ResourceFile_Consumables_Items_and_Packages.csv'
        )

        def get_contraceptive_method_for_request(in_d):
            """
            Based on a dictionary of requested items returns what contraception
            method was requested.

            :param in_d: a dictionary of requested items

            :return: Contraception method as string.
            """
        # TODO: soft code this (use resource_items_pkgs_df)
        #  (note: similar thing done in co_test in analyses combined branch)
        # TODO: remove zeros from logging
            if dict({1: 3.75}).items() <= in_d.items():
                return 'pill'
            if in_d == dict({2: 30}):
                return 'male_condom'
            if dict({3: 1, 1933: 1, 98: 1, 5: 1, 75: 1}).items() <= in_d.items():
                return 'injections'
            if dict({1933: 2, 7: 1}).items() <= in_d.items():
                return 'IUD'
            if dict({1933: 3, 8: 2, 5: 1, 9: 2, 10: 0.1, 11: 1, 12: 1, 75: 1}).items() <= in_d.items():
                return 'implant'
            if dict({8: 1, 307: 0.5, 15: 1, 1960: 3, 75: 2, 2676: 3, 2677: 3, 21: 0.25, 112: 2, 23: 8, 5: 2,
                     49: 0.2}).items() <= in_d.items():
                return 'female_sterilization'
            if in_d == dict({25: 30}):
                return 'other_modern'
            else:
                raise ValueError(
                    "There is an unrecognised request: " + str(in_d) + "."
                )

        cons_processed_df['Contraceptive_Method'] = \
            cons_processed_df['Request'].apply(get_contraceptive_method_for_request)

        cons_time_and_method_df =\
            create_time_period_data(in_required_time_period_starts, cons_processed_df)

        # Group consumables data by time period and contraceptive method:
        cons_time_and_method_df =\
            cons_time_and_method_df.set_index(['Time_Period', 'Contraceptive_Method'])

        cons_avail_grouped_by_time_and_method_df =\
            cons_time_and_method_df.loc[:, 'Item_Available']\
            .dropna().groupby(['Time_Period', 'Contraceptive_Method']).agg(lambda x: list(x)).copy().to_frame()

        # Sum the counts of all item types that were actually used
        # (i.e. were available when requested) per time period per method.

        def string_to_dict(in_l_d_as_string):
            """
            Transforms a dictionary written as string to actual dictionaries.

            :param in_l_d_as_string: a list of dictionaries written as string

            :return: A list of dictionaries as dictionaries.
            """
            l_dicts = []
            for d_as_string in in_l_d_as_string:
                l_dicts.append(eval(d_as_string))
            return l_dicts

        cons_avail_grouped_by_time_and_method_df['Item_Available_summation'] =\
            cons_avail_grouped_by_time_and_method_df['Item_Available'].apply(string_to_dict).apply(merge_dicts)

        def get_intervention_pkg_name(in_co_meth_name):
            """
            Returns Intervention_Pkg name used in a ResourceFile for the input co. method name.

            :param in_co_meth_name: name of the contraception method

            :return: Intervention_Pkg name used in a ResourceFile.
            """
            if in_co_meth_name == 'pill':
                return 'Pill'
            if in_co_meth_name == 'male_condom':
                return 'Male condom'
            if in_co_meth_name == 'other_modern':
                return 'Female Condom'
            if in_co_meth_name == 'IUD':
                return 'IUD'
            if in_co_meth_name == 'injections':
                return 'Injectable'
            if in_co_meth_name == 'implant':
                return 'Implant'
            if in_co_meth_name == 'female_sterilization':
                return 'Female sterilization'
            else:
                raise ValueError(
                    "There is an unrecognised co. method name: " + str(in_co_meth_name) + "."
                )

        # Calculate the costs of available items for all except male_condom
        # & other_modern (= female condom only currently)
        # TODO in future?: change if Male sterilization is modelled as other_modern as well
        def calculate_costs(in_df_resource_items_pkgs,
                            in_df_cons_avail_by_time_and_method,
                            in_df_mean_use):
            """
            Calculates costs of available items per time period per method rescaled (from the nmbs for simulated
            pop_size) to the population size of Malawi.

            :param in_df_resource_items_pkgs: resource data frame with
                information about items and pkgs for contraception methods only
                (such as 'Contraceptive_Method', 'Item_Code',
                'Expected_Units_Per_Case', 'Unit_Cost', etc.)
            :param in_df_cons_avail_by_time_and_method: data frame grouped by
                'Time_Period' and 'Contraceptive_Method' including available
                items as individual records 'Item_Available' and as one
                summation record 'Item_Available_summation'
            :param in_df_mean_use: data frame with mean numbers of women using
                the contraceptives within the 'Time_Period' and including also
                the length of the time period 'tp_len'

             :return: List of costs per time period per contraceptive method.
            """
            l_costs = []
            for i in in_df_cons_avail_by_time_and_method.index:
                costs = 0
                # Calculate costs from the logs and rescale to the pop. size of Malawi
                item_avail_dict = in_df_cons_avail_by_time_and_method.loc[
                    i, 'Item_Available_summation'
                ]
                for time_method_key in list(item_avail_dict.keys()):
                    # TODO: change the below to be any of items form the 'Contraception initiation' pkg
                    if time_method_key == 2019:
                        unit_cost = float(in_df_resource_items_pkgs['Unit_Cost'].loc[
                                (in_df_resource_items_pkgs['Intervention_Pkg'] == 'Contraception initiation')
                                & (in_df_resource_items_pkgs['Item_Code'] == time_method_key)])
                        costs = costs + (unit_cost * item_avail_dict[time_method_key])
                    else:
                        unit_cost = float(in_df_resource_items_pkgs['Unit_Cost'].loc[
                                (in_df_resource_items_pkgs['Intervention_Pkg'] == get_intervention_pkg_name(i[1]))
                                & (in_df_resource_items_pkgs['Item_Code'] == time_method_key)])
                        costs = costs + (unit_cost * item_avail_dict[time_method_key])
                costs = costs * scaling_factor
                l_costs.append(costs)
            return l_costs

        cons_avail_grouped_by_time_and_method_df['Costs'] =\
            calculate_costs(resource_items_pkgs_df,
                            cons_avail_grouped_by_time_and_method_df,
                            mean_use_df)

        def sum_costs_all_times(in_df_costs_by_tp):
            """
            Adds a row with sum of costs in all time periods.
            :param in_df_costs_by_tp: 'Time_Period' and 'Contraceptive_Method'
                as index, 'Costs' as a column.

            :return: The sum rows for each contraceptive method to append to the
                input df.
            """
            # tp of all times, ie very first to very last year of time periods
            y_first = in_df_costs_by_tp.index[0][0].split("-")[0]
            y_last = in_df_costs_by_tp.index[len(in_df_costs_by_tp) - 1][0].split("-")[1]
            sum_tp = (str(y_first) + "-" + str(y_last))
            # Sum the costs in all time periods for each contraceptive method
            sum_costs = in_df_costs_by_tp.groupby(level=[1]).sum()
            return pd.DataFrame(list(sum_costs.loc[:, 'Costs']),
                                columns=['Costs'],
                                index=[[sum_tp] * len(sum_costs.index),
                                       sum_costs.index]
                                )

        cons_costs_by_time_and_method_df =\
            cons_avail_grouped_by_time_and_method_df.loc[:, 'Costs'].copy().to_frame()
        cons_costs_by_time_and_method_df =\
            cons_costs_by_time_and_method_df\
            .append(sum_costs_all_times(cons_costs_by_time_and_method_df))

        print("Calculations of Consumables Costs finished.")

        # If calculation of intervention costs requested,
        # warn if simulation ends before the interventions are implemented
        calc_intervention_costs_bool = in_calc_intervention_costs_bool
        if in_calc_intervention_costs_bool:
            interv_info_df =\
                log_df['tlo.methods.contraception']['interventions_start_date'].set_index('date').copy()
            interv_implem_date = Date(interv_info_df.loc['2010-01-01', 'date_co_interv_implemented'])
            if Date(last_day_simulated) < interv_implem_date:
                warnings.warn(
                    '\nWarning: Calculations of intervention costs are not provided as the simulation ends before'
                    ' interventions are introduced.'
                )
                calc_intervention_costs_bool = False
        # %% Calculate annual Pop and PPFP intervention costs:
        if calc_intervention_costs_bool:
            # @@ Load Population Totals (Demography Model Results)
            # females 15-49 by year
            demog_df_f = log_df['tlo.methods.demography']['age_range_f'].set_index('date').copy()
            demog_df_f['year'] = pd.to_datetime(demog_df_f.index).year
            demog_df_f.index = pd.to_datetime(demog_df_f.index).year
            demog_df_f.index.name = 'year'
            demog_df_f['15-49'] =\
                demog_df_f.loc[:, ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']].sum(axis=1)
            # males 15-49 by year
            demog_df_m = log_df['tlo.methods.demography']['age_range_m'].set_index('date').copy()
            demog_df_m['year'] = pd.to_datetime(demog_df_m.index).year
            demog_df_m.index = pd.to_datetime(demog_df_m.index).year
            demog_df_m.index.name = 'year'
            demog_df_m['15-49'] =\
                demog_df_m.loc[:, ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']].sum(axis=1)
            # total females & males 15-49 as both targeted by Pop and PPFP interventions, by year
            popsize1549 = demog_df_f['15-49'] + demog_df_m['15-49']
            popsize1549 = pd.DataFrame(popsize1549)
            popsize1549['year'] = demog_df_f['year']
            # Calculate ratio of population of reproductive age compared to 2016 as base year (a year for which Pop and
            # PPFP intervention implementation costs are estimated)
            popsize1549['ratio'] = popsize1549.loc[:, '15-49'] / popsize1549.loc[2016, '15-49']
            # Multiply Pop and PPFP intervention costs by this ratio for each year:
            # cost of Pop and PPFP intervention implementations for whole population of Malawi at the end of 2015
            # (MWK - Malawi Kwacha) with inflation rate of 81% to 2020
            # TODO?: these 2 parameters are approximated from costs for 2016-2020 - ie not approximation for pop of 2016
            #  as we use it, hence may be used rather as approximation for average pop of 2016-2020 if decided to
            co_analysis_params = pd.read_csv('resources/ResourceFile_ContraceptionAnalysisParams.csv')
            pop_interv_cost_2016 = float(co_analysis_params['value'].loc[co_analysis_params['parameter_name'] ==
                                                                         'pop_intervention_cost']) * 1.81
            ppfp_interv_cost_2016 = float(co_analysis_params['value'].loc[co_analysis_params['parameter_name'] ==
                                                                          'ppfp_intervention_cost']) * 1.81
            # Calculate interventions costs for each year (proportional to popsize of reproductive age)
            popsize1549['pop_intervention_cost'] = popsize1549['ratio'] * pop_interv_cost_2016
            popsize1549['ppfp_intervention_cost'] = popsize1549['ratio'] * ppfp_interv_cost_2016
            popsize1549['interventions_total'] =\
                popsize1549['pop_intervention_cost'] + popsize1549['ppfp_intervention_cost']
            # interventions costs before implementation = 0
            popsize1549.loc[range(2010, interv_implem_date.year),
                            ['pop_intervention_cost', 'ppfp_intervention_cost', 'interventions_total']] = 0
            # Assign time_periods to the data
            co_interv_costs_tp_df = \
                create_time_period_data(in_required_time_period_starts,
                                        popsize1549)
            # Group intervention costs by time period
            co_interv_costs_tp_df.index = co_interv_costs_tp_df['Time_Period']
            co_interv_costs_sum_by_tp_df =\
                co_interv_costs_tp_df.loc[
                    :, ['pop_intervention_cost', 'ppfp_intervention_cost', 'interventions_total']
                ].dropna().groupby(['Time_Period']).sum()

            def sum_interv_costs_all_times(in_df_interv_costs_by_tp):
                """
                Adds a row with sum of intervention costs in all time periods.
                :param in_df_interv_costs_by_tp: 'Time_Period' as index;
                    'pop_intervention_cost', 'ppfp_intervention_cost', 'interventions_total' as columns.
                :return: The sum row with interventions costs in all time periods.
                """
                # tp of all times, ie very first to very last year of time periods
                y_first = in_df_interv_costs_by_tp.index[0].split("-")[0]
                y_last = in_df_interv_costs_by_tp.index[len(in_df_interv_costs_by_tp) - 1].split("-")[1]
                sum_tp = (str(y_first) + "-" + str(y_last))
                # Sum the costs in all time periods
                sum_costs = in_df_interv_costs_by_tp.sum(axis=0).to_frame().transpose()
                sum_costs.index = [sum_tp]
                return sum_costs

            co_interv_costs_sum_by_tp_df = co_interv_costs_sum_by_tp_df.append(
                sum_interv_costs_all_times(co_interv_costs_sum_by_tp_df)
            )

            print("Calculations of Intervention Costs finished.")

        # If calculation of intervention costs is not required:
        else:
            co_interv_costs_sum_by_tp_df =\
                pd.DataFrame({'pop_intervention_cost': [], 'ppfp_intervention_cost': [], 'interventions_total': []})

    # If calculation of Use and Consumables Costs of Contraception methods are not required,
    # and hence also no intervention costs calculated:
    else:
        co_output_use_modern_tp_df, co_output_percentage_use_df, cons_costs_by_time_and_method_df = [], [], []
        co_interv_costs_sum_by_tp_df = \
            pd.DataFrame({'pop_intervention_cost': [], 'ppfp_intervention_cost': [], 'interventions_total': []})

    return co_output_use_modern_tp_df, co_output_percentage_use_df, \
        cons_costs_by_time_and_method_df, co_interv_costs_sum_by_tp_df, \
        scaling_factor
