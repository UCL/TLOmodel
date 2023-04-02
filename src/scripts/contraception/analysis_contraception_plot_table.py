import math
import time
from pathlib import Path

import bar_chart_costs
import fnc_analyse_contraception as a_co
import numpy as np
import pandas as pd

time_start = time.time()
# running time - both analysis all figs & tab for 250K pop till 2050:
# running 1st time (ie run_analysis = True) ~ 34 mins
# running 2nd time (ie run_analysis = False) ~ 1.5 min
########################################################################################################################
# TODO: estimate the pop_size_simulated from scaling_factor (and if not same for both sims, add them to IDs instead to
#  suffix) & return last year of sims (the same for that) // separate them as pop_size_simulated & last_year_simulated
pop_size_simulated = "2K"
# pop_size_simulated = "250K_till2050"
branch_name = 'co_2023_02_inclPR807-CostsUpdate'
# which results to use
# - Without interv
datestamp_without_log = '2023-03-25T113153'
# 2K till 2099, with new days_betw_appts: '2023-03-25T113153' from 2023-03-25T112934Z
# datestamp_without_log = '2023-01-20T185253'
# 2K till 2099: '2023-01-20T185253' from 2023-01-20T185037Z
# datestamp_without_log = '2023-02-02T194158'
# 250K till 2050; enhanced_lifestyle, healthseekingbehaviour, symptommanager excluded:
#     '2023-02-02T194158' from 2023-02-02T193933Z
# # - With interv
datestamp_with_log = '2023-03-25T115607'
# 2K till 2099, with new days_betw_appts: '2023-03-25T115607' from 2023-03-25T115340Z
# datestamp_with_log = '2023-01-20T185048'
# 2K till 2050: '2023-01-20T185048' from 2023-01-20T184840Z
# datestamp_with_log = '2023-02-02T194458'
# 250K till 2050; enhanced_lifestyle, healthseekingbehaviour, symptommanager excluded:
#     '2023-02-02T194458' from 2023-02-02T194247Z
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
# %% Do you want to set the upper limits for the y-axes for the 3 plots above?
set_ylims_bool = True
# If the above is True (otherwise it doesn't matter),
# upper limits for the figures (in the order [Use, Props of Use, Use By Method, Pregnancies, Props of Pregnancies ]
ylims_l = [1.08e7, 0.88, 3.6e6, 1.37e6, 0.019]
#
# %%%% table
# %% Run analysis? If the dataframes from the analysis are not prepared yet, then run the analysis.
# Otherwise the saved dataframes will be used to create table and costs fig.
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
# Order of contraceptives for a fig and a table
contraceptives_order = ['pill', 'IUD', 'injections', 'implant', 'male_condom',
                        'female_sterilization', 'other_modern']
# MWK to USD exchange rate (1 MWK = mwk_to_usd_exchange_rate USD)
mwk_to_usd_exchange_rate = 1/790
# %% Calculate Contraception Pop and PPFP Intervention Costs over time?
# calc_intervention_costs_bool = False
calc_intervention_costs_bool = True
# %% Round the number of women using contraception? No => None, Yes => set to nearest what to round them
# (e.g. to nearest thousands => 1e3).
rounding_use_to = 1e3
# %% Round the costs? No => None, Yes => set to nearest what to round them.
rounding_costs_to = 1e6
#
# %%%% Parameters only for test runs (for final runs set them as True-True-False-True)
# # Do you want to do both analysis? If not (set one of the below to False). The analysis won't be done and the outputs
# from the other analysis (set to True below) will be used instead.
do_no_interv_analysis = True
do_interv_analysis = True
# Do you want prints to see costs, use, percentage use and table?
# If False, no output is printed, but the output table is still saved in the 'outputs' folder.
print_bool = False
# print_bool = True
# %% Plot Consumables & Intervention Costs Over Time from the Table?
# plot_costs = False
plot_costs = True
########################################################################################################################
# Actually run analysis for the table, only if you require the table. ;)
run_analysis = run_analysis and table_use_costs_bool

# suffix for the output figure(s) and/or table
if set_ylims_bool:
    branch_name = branch_name + '_yaxis-lims-united'
suffix = '_' + branch_name + '_' + 'useTo_' + str(rounding_use_to) + '_costsTo_' + str(rounding_costs_to) + \
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


def fullprint(in_to_print):
    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None):
        print(in_to_print)


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
            # %% Do you want to set the upper limits for the y-axes?
            # If so, order them as [Use, Use By Method, Pregnancies] within ylims_l.
            set_ylims_bool, ylims_l,
            # %% Calculate Use and Consumables Costs of Contraception methods within
            # some time periods?
            run_analysis, TimePeriods_starts,
            # List of modern methods in order in which they should appear in table
            contraceptives_order,
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
    return use_with_df_loaded, percentage_use_with_df_loaded, costs_with_df_loaded, interv_costs_with_df_loaded,\
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

# prints
if print_bool:
    if do_no_interv_analysis:
        print("\n")
        print("COSTS WITHOUT")
        print(costs_without_df)
        print()
        print("\n")
        print("INTERVENTION COSTS WITHOUT")
        print(interv_costs_without_df)
        print()
    if do_interv_analysis:
        print("\n")
        print("COSTS WITH")
        print(costs_with_df)
        print("\n")
        print("INTERVENTION COSTS WITH")
        print(interv_costs_with_df)
        print()
    if do_no_interv_analysis:
        print("\n")
        print("MEAN USE WITHOUT")
        fullprint(use_without_df)
        print(list(use_without_df.columns))
        #
        print("\n")
        print("MEAN PERCENTAGE USE WITHOUT")
        fullprint(percentage_use_without_df)
        print(list(percentage_use_without_df.columns))

# %% Check both sims done with the same population size - if not Warning
# TODO: if both analyses done, check whether both were simulated for the same pop_size (ie scaling factors
#  scaling_factor_without and scaling_factor_with are equal)
#  if not => Warning (just to inform about it)

# %% Table Use and Consumables Costs of Contraception methods Over time
# with and without intervention:
if table_use_costs_bool:
    if not ('use_output' in locals() or 'use_output' in globals()):
        use_output = "mean"

    def round_format(in_df, in_rounding_to):
        if in_rounding_to:
            round_index = math.log10(in_rounding_to)  # TODO: fix the round_index warning
        else:
            round_index = math.log10(1)
        df_rounded = (round(in_df, -int(round_index)) / in_rounding_to).astype(int)
        df_formatted = pd.DataFrame()
        for col_name in df_rounded.columns:
            df_formatted[col_name] = df_rounded[col_name].map('{:,.0f}'.format)
        return df_formatted

    def use_perc_val_df_round_format(use_df, percentage_use_df):
        # %% Round the nmb of women using?
        if rounding_use_to:
            round_use_index = math.log10(rounding_use_to)
        else:
            round_use_index = math.log10(1)
        use_df_rounded = (round(use_df, -int(round_use_index)) / rounding_use_to).astype(int)
        use_df_formatted = pd.DataFrame()
        for col_name in use_df_rounded.columns:
            use_df_formatted[col_name] = use_df_rounded[col_name].map('{:,.0f}'.format)
        return percentage_use_df.round(1).astype(str) + '%' + " (" + use_df_formatted.astype(str) + ")"

    # %% Join & Round percentages and total values of use:
    use_without_perc_val_df = percentage_use_without_df.round(1).astype(str) + '%' + " (" + \
        round_format(use_without_df, rounding_use_to).astype(str) + ")"
    use_with_perc_val_df = percentage_use_with_df.round(1).astype(str) + '%' + " (" + \
        round_format(use_with_df, rounding_use_to).astype(str) + ")"

    # %% Round the costs:
    if rounding_costs_to:
        round_index = math.log10(rounding_costs_to)
        costs_without_df = round(costs_without_df, -int(round_index)) / rounding_costs_to
        costs_with_df = round(costs_with_df, -int(round_index)) / rounding_costs_to
        if do_interv_analysis:
            interv_costs_with_df = round(interv_costs_with_df, -int(round_index)) / rounding_costs_to
        if not do_no_interv_analysis:
            interv_costs_without_df = interv_costs_with_df

        if print_bool:
            if rounding_costs_to:
                if do_no_interv_analysis:
                    print("\n")
                    print("COSTS WITHOUT rounded")
                    print(costs_without_df)
                if do_interv_analysis:
                    print("\n")
                    print("COSTS WITH rounded")
                    print(costs_with_df)
            if calc_intervention_costs_bool:
                if do_no_interv_analysis:
                    print("\n")
                    print("INTERVENTION COSTS WITHOUT")
                    print(interv_costs_without_df)
                    print()
                if do_interv_analysis:
                    print("\n")
                    print("INTERVENTION COSTS WITH rounded")
                    print(interv_costs_with_df)
                    print()

    # %% Plot Consumables & Intervention Costs Over Time from the Table:
    if plot_costs:
        # group consumables costs by time periods
        cons_costs_without_tp_l = costs_without_df.groupby(level=[0], sort=False).sum()['Costs'].tolist()
        cons_costs_with_tp_l = costs_with_df.groupby(level=[0], sort=False).sum()['Costs'].tolist()
        # create lists with interv costs
        pop_interv_costs_with_tp_l = interv_costs_with_df['pop_intervention_cost'].tolist()
        ppfp_interv_costs_with_tp_l = interv_costs_with_df['ppfp_intervention_cost'].tolist()
        bar_chart_costs.plot_costs(
            [datestamp_without_log, datestamp_with_log], suffix, list(interv_costs_with_df.index),
            cons_costs_without_tp_l, cons_costs_with_tp_l, pop_interv_costs_with_tp_l, ppfp_interv_costs_with_tp_l,
            mwk_to_usd_exchange_rate  # & default in_reduce_magnitude=1e3
        )

    # TODO: move the creation of the table (bellow) to a separate .py file
    def combine_use_costs_with_without_interv(
        in_df_use_without, in_df_use_perc_without, in_df_costs_without, in_df_interv_costs_without,
            in_df_use_with, in_df_use_perc_with, in_df_costs_with, in_df_interv_costs_with):
        data = []
        for tp in in_df_use_without.index:
            for in_df_use, in_df_use_perc, in_df_costs, in_df_interv_costs in \
                [(in_df_use_without, in_df_use_perc_without, in_df_costs_without, in_df_interv_costs_without),
                 (in_df_use_with, in_df_use_perc_with, in_df_costs_with, in_df_interv_costs_with)]:
                # use nmb (perc)
                l_tp_use = list(in_df_use_perc.loc[tp, :])
                for i in range(4):
                    l_tp_use.append('--')
                data.append(l_tp_use)
                # costs
                l_tp_costs = []
                co_modern_total = "NA"
                for meth in in_df_use.columns:
                    if meth in list(in_df_costs.loc[tp].index.get_level_values('Contraceptive_Method')):
                        l_tp_costs.append(round(float(in_df_costs.loc[(tp, meth), :]), 2))
                    else:
                        if meth == 'co_modern_total':
                            co_modern_total = sum(l_tp_costs)
                            l_tp_costs.append(round(co_modern_total, 2))
                        else:
                            l_tp_costs.append(0)
                if in_df_interv_costs.empty:
                    for i in range(3):
                        l_tp_costs.append(0)
                    l_tp_costs.append(co_modern_total)
                else:
                    l_tp_costs.append(round(in_df_interv_costs.loc[tp, 'pop_intervention_cost'], 2))
                    l_tp_costs.append(round(in_df_interv_costs.loc[tp, 'ppfp_intervention_cost'], 2))
                    l_tp_costs.append(round(in_df_interv_costs.loc[tp, 'interventions_total'], 2))
                    l_tp_costs.append(round(in_df_interv_costs.loc[tp, 'interventions_total'] +
                                            co_modern_total, 2))
                data.append(l_tp_costs)
        table_cols = list(in_df_use_without.columns)
        table_cols.append('pop_interv')
        table_cols.append('ppfp_interv')
        table_cols.append('pop_ppfp_interv')
        table_cols.append('co_modern_all_interv_total')

        def rounding_name(in_rounding_scale):
            if in_rounding_scale:
                if in_rounding_scale == 1e9:
                    return "billions "
                elif in_rounding_scale == 1e6:
                    return "millions "
                elif in_rounding_scale == 1e5:
                    return "hundreds of thousands "
                elif in_rounding_scale == 1e4:
                    return "tens of thousands "
                elif in_rounding_scale == 1e3:
                    return "thousands "
                elif in_rounding_scale == 100:
                    return "hundreds "
                elif in_rounding_scale == 10:
                    return "tens "
                else:
                    return str(in_rounding_scale) + " "
            else:
                return ""

        df_combined = pd.DataFrame(data[:],
                                   columns=pd.Index(table_cols,
                                                    name='Contraception method'),  # TODO: maybe remove?
                                   index=pd.MultiIndex.from_product([
                                       in_df_use_without.index,
                                       ['Without interventions', 'With Pop and PPFP interventions'],
                                       [str(use_output).capitalize() + ' % of\n women using\n (' +
                                        rounding_name(rounding_use_to) + ' users)',
                                        'Costs\n (' + rounding_name(rounding_costs_to) + ' MWK ~\n '
                                        + rounding_name(rounding_costs_to / 1000) + 'USD)']
                                   ]))
        return df_combined.loc[:, :].transpose()

    use_costs_table_df = combine_use_costs_with_without_interv(
        use_without_df, use_without_perc_val_df, costs_without_df, interv_costs_without_df,
        use_with_df, use_with_perc_val_df, costs_with_df, interv_costs_with_df
    )

    # Change the names of totals
    use_costs_table_df = use_costs_table_df.rename(
        index={'co_modern_total': 'modern contraceptives\n TOTAL',
               'pop_interv': 'Pop intervention',
               'ppfp_interv': 'PPFP intervention',
               'pop_ppfp_interv': 'Pop & PPFP intervention',
               'co_modern_all_interv_total': 'modern contraceptives\n & interventions TOTAL'
               }
    )
    # Remove the underscores from the names of contraception methods
    use_costs_table_df.index = use_costs_table_df.index.map(lambda s: s.replace("_", " "))

    if print_bool:
        print("\n")
        print("TABLE")
        fullprint(use_costs_table_df)

    output_table_file = r"outputs/output_table_" + use_output + "-use_costs" + "__" + ID_without + "_" + ID_with + \
                        suffix + ".xlsx"
    writer = pd.ExcelWriter(output_table_file)
    # Mean use rounded to two decimals for the table
    use_costs_table_df.to_excel(writer, index_label=use_costs_table_df.columns.name)

    writer.save()
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
