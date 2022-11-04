import numpy as np

import fnc_analyse_contraception as a_co
import pandas as pd
# import collections
import timeit
import time

time_start = time.time()

################################################################################
# TO SET:  # TODO: update with final runs
# for the output figures
datestamp_without = '2022_11_03'
datestamp_with = '2022_11_03'  # TODO: need data
datestamp_without_log = '2022-11-03T141744'
datestamp_with_log = '2022-11-03T141744'  # TODO: need data
logFile_without = 'run_analysis_contraception__' + datestamp_without_log + '.log'
logFile_with = 'run_analysis_contraception__' + datestamp_with_log + '.log'
# which years we want to summarise for the table of use and costs
TimePeriods_starts = [2022, 2031, 2041, 2051]
# order of contraceptives for the table
contraceptives_order = ['pill', 'IUD', 'injections', 'implant', 'male_condom',
                        'female_sterilization', 'other_modern']
################################################################################
# ##### '2022-11-03T141744'
# => cons_availability = "all"
# => start_date = Date(2010, 1, 1); end_date = Date(2099, 12, 31)
# seed = 2022
# and pop_size = 50000
# after rounding up the numbers of items removed; no intervention;
# f. steril only in women 30+ (except during pop initiation, 20+ then)
# ##### '2022-11-03T115327'
# => cons_availability = "all"
# => start_date = Date(2010, 1, 1); end_date = Date(2099, 12, 31)
# seed = 2022
# and pop_size = 50000
# after rounding up the numbers of items removed; no intervention;
# f. steril only in women 30+
# ##### '2022-11-03T114819'
# => cons_availability = "all"
# => start_date = Date(2010, 1, 1); end_date = Date(2099, 12, 31)
# seed = 2022
# and pop_size = 50000
# after rounding up the numbers of items removed; interventions since 2023;
# f. steril only in women 30+
# ##### '2022-10-13T165132'
# => cons_availability = "all"
# => start_date = Date(2010, 1, 1); end_date = Date(2050, 12, 31)
# seed = 2022
# and pop_size = 20 => only table running time: ~6.688109636306763 s
# after rounding up the numbers of items removed
# ##### '2022-10-13T105006'
# => cons_availability = "all"
# => start_date = Date(2010, 1, 1); end_date = Date(2099, 12, 31)
# seed = 2022
# and pop_size = 50
# before rounding up the numbers of items removed
################################################################################


def fullprint(in_to_print):  # TODO: remove
    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None):
        print(in_to_print)


def timeitprint(in_what_measures, in_fnc, in_timeit_rep_nmb=1):  # TODO: remove
    if in_timeit_rep_nmb > 1:
        print("time (s) of " + in_what_measures +
              " (" + str(in_timeit_rep_nmb) + " repetitions):")
    else:
        print("time (s) of " + in_what_measures + ":")
    print(timeit.timeit(in_fnc, number=in_timeit_rep_nmb))


# Use and Consumables Costs of Contraception methods Over time
# WITHOUT interventions:
def do_without_analysis():  # TODO: temporarily as a function (to allow better measurement of running time)
    out_use_without_df, out_percentage_use_without_df, out_costs_without_df =\
        a_co.analyse_contraception(
            datestamp_without, logFile_without,
            # %% Plot Contraception Use Over time?
            False,
            # %% Plot Contraception Use By Method Over time?
            False,
            # %% Plot Pregnancies Over time?
            False,
            # List of modern methods
            contraceptives_order,
            # Calculate Use and Consumables Costs of Contraception methods within
            # some time periods?
            True, TimePeriods_starts
            # and default: in_use_output="mean"
        )
    return out_use_without_df, out_percentage_use_without_df, out_costs_without_df


use_without_df, percentage_use_without_df, costs_without_df = do_without_analysis()
# timeitprint("one analysis performance only", do_without_analysis)
# 11.47628352700849 for test data ('2022-09-10T181844')

print("\n")
print("COSTS")
print(costs_without_df)
#
print("\n")
print("MEAN USE")
fullprint(use_without_df)
print(list(use_without_df.columns))
#
print("\n")
print("MEAN PERCENTAGE USE")
fullprint(percentage_use_without_df)
print(list(percentage_use_without_df.columns))

# # Use and Consumables Costs of Contraception methods Over time
# WITH interventions:  # TODO: need the data
use_with_df, percentage_use_with_df, costs_with_df =\
    a_co.analyse_contraception(
        datestamp_with, logFile_with,
        # %% Plot Contraception Use Over time?
        False,
        # %% Plot Contraception Use By Method Over time?
        False,
        # %% Plot Pregnancies Over time?
        False,
        # Calculate Use and Consumables Costs of Contraception methods within
        # some time periods?
        True, TimePeriods_starts
    )
# use_with_df = use_without_df
# percentage_use_with_df = percentage_use_without_df
# costs_with_df = costs_without_df

# %% Plot Use and Consumables Costs of Contraception methods Over time
# with and without intervention:
if not ('use_output' in locals() or 'use_output' in globals()):
    use_output = "mean"

# TODO: finish

# TODO: Footnote marks manually into the table in the manuscript, at least so
#  far.

# DF.rename_axis('Contraception method', axis=1)

use_without_val_perc_df =\
    use_without_df.round(2).astype(str) +\
    " (" + percentage_use_without_df.round(2).astype(str) + ")"
use_with_val_perc_df =\
    use_with_df.round(2).astype(str) +\
    " (" + percentage_use_with_df.round(2).astype(str) + ")"


def combine_use_costs_with_without_interv(
    in_df_use_without, in_df_use_perc_without, in_df_costs_without,
    in_df_use_with, in_df_use_perc_with, in_df_costs_with,
    in_co_order):
    # assert collections.Counter(in_df1_use) == collections.Counter(in_df2_use)
    # assert collections.Counter(in_df1_costs) == collections.Counter(in_df2_costs)
    # assert all(in_df1_use.index == in_df2_use.index)
    # assert all(in_df1_costs.index == in_df2_costs.index)
    data = []
    for tp in in_df_use_without.index:
        for in_df_use, in_df_use_perc, in_df_costs in \
            [(in_df_use_without, in_df_use_perc_without, in_df_costs_without),
             (in_df_use_with, in_df_use_perc_with, in_df_costs_with)]:
            # use nmb (perc)
            l_tp_use = list(in_df_use_perc.loc[tp, :])
            l_tp_use.append('--')
            data.append(l_tp_use)
            # costs
            l_tp_costs = []
            for meth in in_df_use.columns:
                if meth in list(in_df_costs.loc[tp].index.get_level_values('Contraceptive_Method')):
                    l_tp_costs.append(round(float(in_df_costs.loc[(tp, meth), :]), 2))
                else:
                    if meth == 'co_modern_total':
                        l_tp_costs.append(round(sum(l_tp_costs), 2))
                    else:
                        l_tp_costs.append(0)
            l_tp_costs.append('-tba-')  # TODO: to be added -> sum total costs and interv
            data.append(l_tp_costs)
    table_cols = list(in_df_use_without.columns)
    table_cols.append('co_modern_interv_total')
    df_combined = pd.DataFrame(data[:],
                               columns=pd.Index(table_cols,
                                                name='Contraception method'),  # TODO: maybe remove?
                               index=pd.MultiIndex.from_product([
                                   in_df_use_without.index,
                                   ['Without interventions', 'With Pop and PPFP interventions'],
                                   [str(use_output).capitalize() + ' number of women using (%)', 'Costs']
                               ]))
    in_co_order.append('co_modern_total')
    in_co_order.append('co_modern_interv_total')
    df_combined = df_combined.loc[:, in_co_order].transpose()
    return df_combined


use_costs_table_df = combine_use_costs_with_without_interv(
    use_without_df, use_without_val_perc_df, costs_without_df,
    use_with_df, use_with_val_perc_df, costs_with_df,
    contraceptives_order
)


print("\n")
print("TABLE")
fullprint(use_costs_table_df)

output_table_file = r"outputs/output_table_" + use_output + "-use_costs" +\
                    "__" + datestamp_without_log + "_" + datestamp_with_log +\
                    ".xlsx"
writer = pd.ExcelWriter(output_table_file)
# Mean use rounded to two decimals for the table
use_costs_table_df.to_excel(writer, index_label=use_costs_table_df.columns.name)

# reverse_mean_percentage_use.round(2)
writer.save()

time_end = time.time()

print("\n")
print("total time incl. two analysis, all plots for both data & creating the table:")  #TODO: rewrite to be true
print(time_end - time_start)
