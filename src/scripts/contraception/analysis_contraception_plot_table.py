"""
To be able to run this script to save figs & tables, the log file has to be exported to the folder 'outputs'.
# TODO: Improve the code so this doesn't have to be done.

This script can be used to plot:
 * use of (any) contraception over time,
 * use of all contraception methods over time,
 * pregnancies over time, and
 * dependency ratio
up to the year 2050 (or less if less years simulated), the first 3 are plotted in both variants, as total numbers of
women and as the proportions among women of reproductive age (15-49 years).

All plots can be prepared for both, simulation without and with interventions. The y-axis limits are pre-set, so they
are the same for both (without/with) to allow easy comparison.
NB. If any line is out of the plot, needs to be run with the parameter 'plot_depend_ratio_bool=False' to see the maximum
and subsequently the parameter 'ylims_l' needs to be adjusted.

* The table of consumables is always prepared (as inexpensive for time).

* The table of use and consumables costs (in MWK & USD) can be prepared. If requested, the time periods for the table
need to be specified, the default 'use_output' for the table is 'mean', but can be changed to 'max'.
* Figure with the total costs per periods from the table, showing consumables and interventions implementation costs,
along with a fig of only totals in all intervention time (for presentations) can be prepared.
NB. To work on the table of use and costs (prepared if 'table_use_costs_bool==True'), it can be run for the first time
with 'run_analysis==True', to calculate the use and costs as it takes the most of the running time (~37-40 min for 250K
pop size simulated to 2050) and store them. But the follow-up runs to see the progress of the work on the table, can be
run with 'run_analysis==False' (~1.7 min for 250K pop size simulated to 2050) when all the pre-calculated values of the
use and costs for the table are imported. Also only one of the analyses (without or with interventions) can be
performed, the other one will be then filled with the numbers from the same analysis. Figures are prepared only if the
the table is prepared,and if requested.

All the options can be set in the # TO SET # section below.
"""

import time
from pathlib import Path

import fnc_analyse_contraception as a_co
import numpy as np
import pandas as pd
import tables

time_start = time.time()
# ####### TO SET #######################################################################################################
# TODO: estimate the pop_size_simulated from scaling_factor (and if not same for both sims, add them to IDs instead to
#  suffix) & return last year of sims (the same for that) // separate them as pop_size_simulated & last_year_simulated
# pop_size_simulated = "2K"
pop_size_simulated = "250K"
branch_name = 'co_final'
# which results to use
# - Without interv
# datestamp_without_log = '2023-04-26T141435'
# 2K till 2099, final costs update EHP & OHT + pregn test to initiate co: '2023-04-26T141435' from 2023-04-26T141159Z
datestamp_without_log = '2023-05-06T170512'
# 250K till 2050; final costs update EHP & OHT + rebased on master + pregn test corrected: '2023-05-06T170512'
#    from 2023-05-06T170253Z
# # - With interv
# datestamp_with_log = '2023-04-26T141545'
# 2K till 2099, final costs update EHP & OHT + pregn test to initiate co: '2023-04-26T141545' from 2023-04-26T141321Z
datestamp_with_log = '2023-05-06T170612'
# 250K till 2050; final costs update EHP & OHT + rebased on master + pregn test corrected: '2023-05-06T170612'
#    from 2023-05-06T170359Z
logFile_without = 'run_analysis_contraception_no_diseases__' + datestamp_without_log + '.log'
logFile_with = 'run_analysis_contraception_no_diseases__' + datestamp_with_log + '.log'
##
# OUTPUT REQUIREMENTS
# %%%% plots up to the year 2050 (regarding of how long are the simulations)
# TODO: have the last year included in figs as input parameter?
# %% Plot Contraception Use Over time?
# plot_use_time_bool = False
plot_use_time_bool = True
# %% Plot Contraception Use By Method Over time?
# plot_use_time_method_bool = False
plot_use_time_method_bool = True
# %% Plot Pregnancies Over time?
# plot_pregnancies_bool = False
plot_pregnancies_bool = True
# %% Plot Dependency Ratio Over time?
# plot_depend_ratio_bool = False
plot_depend_ratio_bool = True
# %% Do you want to set the upper limits for the y-axes for the 3 plots above?
set_ylims_bool = True
# If the above is True (otherwise it doesn't matter),
# upper limits for the figures (in the order [Use, Props of Use, Use By Method, Pregnancies, Props of Pregnancies ]
ylims_l = [1.08e7, 0.88, 3.6e6, 1.37e6, 0.019, 1]
#
# %%%% table
# %% Run analysis? If the dataframes from the analysis are not prepared yet, then run the analysis.
# Otherwise, the saved dataframes will be used to create table and costs fig.
# TODO: Later also the other figs can be prepared outside the analysis script
# run_analysis = False
run_analysis = True
# %% Table the Use and Costs (By Method) Over time?
# table_use_costs_bool = False
table_use_costs_bool = True
# TODO: if there is no debug (ie lookup the key) logging - set the table_use_costs_bool = False and display a Warning
# If the above is True (otherwise all the table inputs below doesn't matter),
# years to summarise in the table of use and costs (totals for time periods between each 2 consecutive years;
# first year included, last year excluded)
TimePeriods_starts = [2023, 2031, 2041, 2051]
# The use & cost values within the time periods in table can be "mean" (default) or can be changed to "max"
# use_output = "max"  # TODO: test whether it still works
# Order of modern contraception methods in which they should appear in figs and tables
contraceptives_order = ['pill', 'IUD', 'injections', 'implant', 'male_condom',
                        'female_sterilization', 'other_modern']
# MWK to USD exchange rate (1 MWK = mwk_to_usd_exchange_rate USD)
mwk_to_usd_exchange_rate = 1/790
# %% Calculate Contraception Pop and PPFP Intervention Costs over time?
# calc_intervention_costs_bool = False
calc_intervention_costs_bool = True
# %% Round the number of women using contraception? No => None, Yes => set to nearest what to round them
# (e.g. to nearest thousands => 1e3).
# TODO: test whether None in following 3 pars works
rounding_use_to = 1e3
# %% Round the costs in MWK? No => None, Yes => set to nearest what to round them.
rounding_costs_mwk_to = 1e6
# %% Round the costs in USD? No => None, Yes => set to nearest what to round them.
rounding_costs_usd_to = rounding_costs_mwk_to / 1000
#
# %%%% Parameters only for test runs (for final runs set them as True-True-False-True)
# # Do you want to do both analysis? If not (set one of the below to False). The analysis won't be done and the outputs
# from the other analysis (set to True below) will be used instead.
do_no_interv_analysis = True
do_interv_analysis = True
# %% Plot Consumables & Intervention Costs Over Time from the Table?
# plot_costs = False
plot_costs = True
########################################################################################################################
# Prepare the table of consumables (no sim is needed)
tables.table_cons(mwk_to_usd_exchange_rate, contraceptives_order)

# Actually run analysis for the table, only if you require the table. ;)
run_analysis = run_analysis and table_use_costs_bool

# suffix for the output figure(s) and/or table
if set_ylims_bool:
    branch_name = branch_name + '_yaxis-lims-united'
suffix = '_' + branch_name + '_' + 'useTo_' + str(rounding_use_to) + '_MWKcostsTo_' + str(rounding_costs_mwk_to) + \
         '_' + pop_size_simulated
###
if run_analysis and table_use_costs_bool:
    assert do_no_interv_analysis | do_interv_analysis, \
        "If you request to create a table of use & costs, at least one analysis needs to be done, ie " + \
        "'do_no_interv_analysis' or 'do_interv_analysis' needs to be True.\n" + \
        "Otherwise, do not request the analysis, ie set 'table_use_costs_bool' to False."

if not ('ylims_l' in locals() or 'ylims_l' in globals()):
    ylims_l = []
if not ('TimePeriods_starts' in locals() or 'TimePeriods_starts' in globals()):
    TimePeriods_starts = []
    # TODO: test whether this works
if not ('use_output' in locals() or 'use_output' in globals()):
    use_output = "mean"

dataframe_folder = 'outputs/dataframes'
# create folder to save dataframes if it doesn't exist yet.
dataframe_path = Path(dataframe_folder)
dataframe_path.mkdir(parents=True, exist_ok=True)


def save_csv(in_to_save, in_file_name, in_datestamp_log):
    in_to_save.to_csv(Path(dataframe_folder + '/' + in_file_name + '_' + in_datestamp_log + '.csv'))
    return 0


# Use and Consumables Costs of Contraception methods Over time
def do_analysis(ID, logFile, in_calc_intervention_costs_bool):
    use_df, percentage_use_df, costs_df, interv_costs_df, scaling_factor_out = \
        a_co.analyse_contraception(
            ID, logFile, suffix,
            # %% Plot Contraception Use Over time?
            plot_use_time_bool,
            # %% Plot Contraception Use By Method Over time?
            plot_use_time_method_bool,
            # %% Plot Pregnancies Over time?
            plot_pregnancies_bool,
            # %% Plot Dependency Ratio Over time?
            plot_depend_ratio_bool,
            # %% Do you want to set the upper limits for the y-axes?
            # If so, order them as [Use, Use By Method, Pregnancies] within ylims_l.
            set_ylims_bool, ylims_l,
            # List of modern methods in order in which they should appear in plots and tables
            contraceptives_order,
            # %% Calculate Use and Consumables Costs of Contraception methods within
            # some time periods?
            run_analysis, TimePeriods_starts,
            # %% Calculate Contraception Pop and PPFP Intervention Costs over time?
            in_calc_intervention_costs_bool
            # and default: in_use_output="mean"
        )
    return use_df, percentage_use_df, costs_df, interv_costs_df, scaling_factor_out


def load_analysis_out(in_analysis_type, in_datestamp_log):
    use_with_df_loaded = \
        pd.read_csv(Path(dataframe_folder + '/use_' + in_analysis_type + '_df_' + in_datestamp_log + '.csv'),
                    index_col=[0])
    percentage_use_with_df_loaded = \
        pd.read_csv(Path(dataframe_folder + '/percentage_use_' + in_analysis_type + '_df_' + in_datestamp_log + '.csv'),
                    index_col=[0])
    costs_with_df_loaded = \
        pd.read_csv(Path(dataframe_folder + '/costs_' + in_analysis_type + '_df_' + in_datestamp_log + '.csv'),
                    index_col=[0, 1])
    interv_costs_with_df_loaded = \
        pd.read_csv(Path(dataframe_folder + '/interv_costs_' + in_analysis_type + '_df_' + in_datestamp_log + '.csv'),
                    index_col=[0])
    scaling_factor_with_loaded = \
        pd.read_csv(Path(dataframe_folder + '/scaling_factor_' + in_analysis_type + '_' + in_datestamp_log + '.npy'))
    return use_with_df_loaded, percentage_use_with_df_loaded, costs_with_df_loaded, interv_costs_with_df_loaded, \
        scaling_factor_with_loaded


if do_no_interv_analysis:
    # WITHOUT interventions:
    print()
    print("analysis without interventions in progress")
    print('--------------------')
    ID_without = datestamp_without_log + "_without"
    use_without_df, percentage_use_without_df, costs_without_df, interv_costs_without_df, scaling_factor_without = \
        do_analysis(ID_without, logFile_without, False)  # no calc of intervention costs for sim without interv
    if not run_analysis:
        use_without_df, percentage_use_without_df, costs_without_df, interv_costs_without_df, scaling_factor_without =\
            load_analysis_out('without', datestamp_without_log)
    else:
        # save dataframes
        for (to_save, file_name) in \
            ((use_without_df, "use_without_df"), (percentage_use_without_df, "percentage_use_without_df"),
             (costs_without_df, "costs_without_df"), (interv_costs_without_df, "interv_costs_without_df")):
            save_csv(to_save, file_name, datestamp_without_log)
        # save scaling factor (numpy float64)
        np.save(Path(dataframe_folder + '/scaling_factor_without_' + datestamp_without_log + '.npy'),
                scaling_factor_without)
        print("Dataframes and scaling factor saved.\n")

    if do_interv_analysis:
        # WITH interventions:
        print()
        print("analysis with interventions in progress")
        print('--------------------')
        ID_with = datestamp_with_log + "_with"
        use_with_df, percentage_use_with_df, costs_with_df, interv_costs_with_df, scaling_factor_with = \
            do_analysis(ID_with, logFile_with, calc_intervention_costs_bool)
        if not run_analysis:
            # load dataframes & scaling factor
            use_with_df, percentage_use_with_df, costs_with_df, interv_costs_with_df, scaling_factor_with = \
                load_analysis_out('with', datestamp_with_log)
        else:
            # save dataframes
            for (to_save, file_name) in \
                ((use_with_df, "use_with_df"), (percentage_use_with_df, "percentage_use_with_df"),
                 (costs_with_df, "costs_with_df"), (interv_costs_with_df, "interv_costs_with_df")):
                save_csv(to_save, file_name, datestamp_with_log)
            # save scaling factor (numpy float64)
            np.save(Path(dataframe_folder + '/scaling_factor_with_' + datestamp_with_log + '.npy'),
                    scaling_factor_with)
            print("Dataframes and scaling factor saved.\n")
    else:
        # use as WITH interventions outputs from the sim WITHOUT interventions
        ID_with = ID_without + "-again"
        use_with_df = use_without_df
        percentage_use_with_df = percentage_use_without_df
        costs_with_df = costs_without_df
        interv_costs_with_df = interv_costs_without_df
        scaling_factor_with = scaling_factor_without

else:
    # WITH interventions:
    print()
    print("analysis with interventions in progress")
    print('--------------------')
    ID_with = datestamp_with_log + "_with"
    use_with_df, percentage_use_with_df, costs_with_df, interv_costs_with_df, scaling_factor_with = \
        do_analysis(ID_with, logFile_with, calc_intervention_costs_bool)
    if not run_analysis:
        # load dataframes & scaling factor
        use_with_df, percentage_use_with_df, costs_with_df, interv_costs_with_df, scaling_factor_with = \
            load_analysis_out('with', datestamp_with_log)
    else:
        # save dataframes
        for (to_save, file_name) in \
            ((use_with_df, "use_witht_df"), (percentage_use_with_df, "percentage_use_with_df"),
             (costs_with_df, "costs_with_df"), (interv_costs_with_df, "interv_costs_with_df")):
            save_csv(to_save, file_name, datestamp_with_log)
        # save scaling factor (numpy float64)
        np.save(Path(dataframe_folder + '/scaling_factor_with_' + datestamp_with_log + '.npy'),
                scaling_factor_with)
        print("Dataframes and scaling factor saved.\n")

    # use as WITHOUT interventions outputs from the sim WITH interventions
    ID_without = ID_with + "-again"
    use_without_df = use_with_df
    percentage_use_without_df = percentage_use_with_df
    costs_without_df = costs_with_df
    interv_costs_without_df = interv_costs_with_df
    scaling_factor_without = scaling_factor_with

# %% Check both sims done with the same population size - if not Warning
# TODO: if both analyses done, check whether both were simulated for the same pop_size (ie scaling factors
#  scaling_factor_without and scaling_factor_with are equal)
#  if not => Warning (just to inform about it)

# %% Table Use and Consumables Costs of Contraception methods Over time
# with and without intervention:
if table_use_costs_bool:

    if not ('use_output' in locals() or 'use_output' in globals()):
        use_output = "mean"

    tables.table_use_costs__plot_costs(use_output, use_without_df, percentage_use_without_df,
                                       use_with_df, percentage_use_with_df, rounding_use_to,
                                       costs_without_df, costs_with_df,
                                       interv_costs_without_df, interv_costs_with_df,
                                       mwk_to_usd_exchange_rate, rounding_costs_mwk_to, rounding_costs_usd_to,
                                       plot_costs, datestamp_without_log, datestamp_with_log, suffix,
                                       ID_without, ID_with)

    to_print = "Tab: Contraception Use (total & percentage) & Costs within Time Periods "
    if do_no_interv_analysis:
        if do_interv_analysis:
            to_print += "Without and With Interventions saved."
        else:
            to_print += "Without and Without-again Interventions saved."
    else:
        to_print += "With and With-again Interventions saved."
    print(to_print)

time_end = time.time()

print("\n")
print("running time (s):")
print(time_end - time_start)
