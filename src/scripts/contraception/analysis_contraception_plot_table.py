import numpy as np

import fnc_analyse_contraception as a_co
import pandas as pd
import collections

################################################################################
# TO SET:
# for the output figures
datestamp_without = '2022_09_10'
datestamp_with = '2022_09_10'  # TODO: need data
# datestamp1_log = '2022-09-14T105623'
datestamp_without_log = '2022-09-10T181844'
datestamp_with_log = '2022-09-10T181844'  # TODO: need data
logFile_without = 'contraception_analysis__' + datestamp_without_log + '.log'
logFile_with = 'contraception_analysis__' + datestamp_with_log + '.log'
# which years we want to summarise for the table of use and costs
TimePeriods_starts = [2022, 2031, 2041, 2051]
# order of contraceptives for the table
contraceptives_order = ['pill', 'IUD', 'injections', 'implant', 'male_condom',
                        'female_sterilization', 'other_modern']
# ##### '2022-09-14T105623' or '2022-09-14T105623' (datestamp1 = .._v1 or ..v2)
# outputFile = 'run_analysis_contraception__' + datestamp1_log + '.log'
# => cons_availability = "all"
# => start_date = Date(2010, 1, 1); end_date = Date(2099, 12, 31)
# seed = 2022
# and pop_size = 50000 => running time: cca 11:02-
# ##### '2022-09-10T181844'
# outputFile = 'contraception_analysis__' + datestamp1_log + '.log'
# => cons_availability = "all"
# => start_date = Date(2010, 1, 1); end_date = Date(2099, 12, 31)
# seed = 0
# and pop_size = 20 => running time: ~17s
# ##### '2022-08-29T160006'
# outputFile = 'contraception_analysis__' + datestamp1_log + '.log'
# => cons_availability = "default"
# => start_date = Date(2010, 1, 1); end_date = Date(2099, 12, 31)
# it is prepared so it can be run max to Date(2099, 12, 31), no longer
# seed = 0
# and pop_size = 20  # TODO: need data simulated with bigger population, 50000 or more?
################################################################################


def fullprint(in_to_print):  # TODO: remove
    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None):
        print(in_to_print)


# Use and Consumables Costs of Contraception methods Over time
# WITHOUT interventions:
use_without_df, percentage_use_without_df, costs_without_df =\
    a_co.analyse_contraception(
        datestamp_without, logFile_without,
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

print("COSTS")
print(costs_without_df)
print(list(costs_without_df.columns))
#
print("MEAN USE")
fullprint(use_without_df)
print(list(use_without_df.columns))

print("MEAN PERCENTAGE USE")
fullprint(percentage_use_without_df)
print(list(percentage_use_without_df.columns))

# # Use and Consumables Costs of Contraception methods Over time
# # WITH interventions:  # TODO: need the data
# use_with_df, percentage_use_with_df, costs_with_df =\
#     a_co.analyse_contraception(
#         datestamp_with, outputFile_with,
#         # %% Plot Contraception Use Over time?
#         False,
#         # %% Plot Contraception Use By Method Over time?
#         False,
#         # %% Plot Pregnancies Over time?
#         False,
#         # Calculate Use and Consumables Costs of Contraception methods within
#         # some time periods?
#         True, TimePeriods_starts
#     )
use_with_df = use_without_df
percentage_use_with_df = percentage_use_without_df
costs_with_df = costs_without_df

# %% Plot Use and Consumables Costs of Contraception methods Over time
# with and without intervention:
if not ('use_output' in locals() or 'use_output' in globals()):
    use_output = "mean"

# TODO: finish

# TODO: Methods in a particular order.

# TODO: Remove the underscores from the names of contraception methods

# TODO: Footnote marks manually into the table in the manuscript, at least so
#  far.

# DF.rename_axis('Contraception method', axis=1)


def join_two_df_second_in_brackets(in_df1, in_df2):
    """
    Creates a data frame where are values from in_df1 followed by values from
    in_df2 in brackets.
    :param in_df1: a data frame - values added before the bracket,
    :param in_df2: a data frame - values added inside the bracket.

    :return: A data frame with joined values from both inputs, second of them
        in brackets.
    """
    assert collections.Counter(in_df1) == collections.Counter(in_df2)  # TODO: try to break it, whether it will warn me
    assert all(in_df1.index == in_df2.index)  # TODO: try to break it, whether it will warn me
    data = []
    for c in in_df1.columns:
        data_c = []
        for r in in_df1.index:
            data_c.append(str(str(in_df1.loc[r, c]) + " (" +
                              str(in_df2.loc[r, c]) + ")"))
            # print(data_c[len(data_c) - 1])
        data.append(data_c)
    joined_df = pd.DataFrame(data, index=in_df1.columns, columns=in_df1.index).transpose()
    return joined_df


# TODO: remove the below
use_without_val_perc_df =\
    join_two_df_second_in_brackets(use_without_df.round(2),
                                   percentage_use_without_df.round(2))

use_with_val_perc_df = join_two_df_second_in_brackets(use_with_df.round(2),
                                                      percentage_use_with_df.round(2))


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
            data.append(list(in_df_use_perc.loc[tp, :]))
            # costs
            l_tp_costs = []
            for meth in in_df_use.columns:
                if meth in list(in_df_costs.loc[tp].index.get_level_values('Contraceptive_Method')):
                    l_tp_costs.append(round(float(in_df_costs.loc[(tp, meth), :]), 2))
                else:
                    if meth == 'women_total':
                        l_tp_costs.append(round(sum(l_tp_costs), 2))
                    else:
                        l_tp_costs.append(0)
            data.append(l_tp_costs)
    df_combined = pd.DataFrame(data[:],
                               columns=pd.Index(in_df_use_without.columns,
                                                name='Contraception method'),
                               index=pd.MultiIndex.from_product([
                                   in_df_use_without.index,
                                   ['Without interventions', 'With Pop and PPFP interventions'],
                                   [str(use_output).capitalize() + ' number of women using (%)', 'Costs']
                               ]))
    in_co_order.append('women_total')
    df_combined = df_combined.loc[:, in_co_order].transpose()
    return df_combined


use_costs_table_df = combine_use_costs_with_without_interv(
    use_without_df, use_without_val_perc_df, costs_without_df,
    use_with_df, use_with_val_perc_df, costs_with_df,
    contraceptives_order
)


# use_costs_table_df = use_without_val_perc_df  # TODO: remove


print("TABLE")
fullprint(use_costs_table_df)

output_file = r"outputs/output_table_" + use_output + "-use_costs_" + "test" + ".xlsx"
writer = pd.ExcelWriter(output_file)
# Mean use rounded to two decimals for the table
use_costs_table_df.to_excel(writer, index_label=use_costs_table_df.columns.name)

# reverse_mean_percentage_use.round(2)
writer.save()
