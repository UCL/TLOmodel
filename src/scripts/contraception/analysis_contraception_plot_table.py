import fnc_analyse_contraception as a_co
import pandas as pd
# import collections
import timeit
import time

time_start = time.time()

################################################################################
# TO SET:  # TODO: update with final sims
# for the output figures
suffix = "20K_TimsCode" # "50K_TimsCode"
datestamp_without_log = '2022-11-07T144634' #50K: '2022-11-08T165333'  # TODO: update with final sim
datestamp_with_log = '2022-11-04T175536' #50K: '2022-11-13T180430'  # TODO: update with final sim
logFile_without = 'run_analysis_contraception__' + datestamp_without_log + '.log'
logFile_with = 'run_analysis_contraception__' + datestamp_with_log + '.log'
# which years we want to summarise for the table of use and costs
TimePeriods_starts = [2023, 2031, 2041, 2051]
# order of contraceptives for the table
contraceptives_order = ['pill', 'IUD', 'injections', 'implant', 'male_condom',
                        'female_sterilization', 'other_modern']
# Do you want prints to see costs, use, percentage use and table?
# If False, no output is printed, but the output table is still saved in the 'outputs' folder.
print_bool = False
# parameter only for test runs (if False, skips the second analysis and uses the outputs from the 1st analysis instead)
# needs to be True for the final run
do_interv_analysis = True
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
print()
print("analysis without interventions in progress")
print('--------------------')
ID_without = datestamp_without_log + "_without" + suffix
use_without_df, percentage_use_without_df, costs_without_df =\
    a_co.analyse_contraception(
        ID_without, logFile_without,
        # %% Plot Contraception Pop and PPFP Intervention Costs over time?
        True,
        # %% Plot Contraception Use Over time?
        True,
        # %% Plot Contraception Use By Method Over time?
        True,
        # %% Plot Pregnancies Over time?
        True,
        # List of modern methods
        contraceptives_order,
        # Calculate Use and Consumables Costs of Contraception methods within
        # some time periods?
        True, TimePeriods_starts
        # and default: in_use_output="mean"
    )

if do_interv_analysis:
    # Use and Consumables Costs of Contraception methods Over time
    # WITH interventions:
    print()
    print("analysis with interventions in progress")
    print('--------------------')
    ID_with = datestamp_with_log + "_with" + suffix
    use_with_df, percentage_use_with_df, costs_with_df =\
        a_co.analyse_contraception(
            datestamp_with_log, logFile_with,
            # %% Plot Contraception Pop and PPFP Intervention Costs over time?
            True,
            # %% Plot Contraception Use Over time?
            True,
            # %% Plot Contraception Use By Method Over time?
            True,
            # %% Plot Pregnancies Over time?
            True,
            # List of modern methods
            contraceptives_order,
            # Calculate Use and Consumables Costs of Contraception methods within
            # some time periods?
            True, TimePeriods_starts
        )
else:
    use_with_df = use_without_df
    percentage_use_with_df = percentage_use_without_df
    costs_with_df = costs_without_df
    ID_with = ID_without + "_again"
if print_bool:
    print("\n")
    print("COSTS WITHOUT")
    print(costs_without_df)
    print()
    print("\n")
    print("COSTS WITH")
    print(costs_with_df)
    #
    print("\n")
    print("MEAN USE WITHOUT")
    fullprint(use_without_df)
    print(list(use_without_df.columns))
    #
    print("\n")
    print("MEAN PERCENTAGE USE WITHOUT")
    fullprint(percentage_use_without_df)
    print(list(percentage_use_without_df.columns))

# %% Plot Use and Consumables Costs of Contraception methods Over time
# with and without intervention:
if not ('use_output' in locals() or 'use_output' in globals()):
    use_output = "mean"

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
    return df_combined.loc[:, :].transpose()


use_costs_table_df = combine_use_costs_with_without_interv(
    use_without_df, use_without_val_perc_df, costs_without_df,
    use_with_df, use_with_val_perc_df, costs_with_df,
    contraceptives_order
)

# Change the names of totals
use_costs_table_df = use_costs_table_df.rename(
    index={'co_modern_total': 'modern contraceptives total',
           'co_modern_interv_total': 'modern contraceptives & interventions total'}
)
# Remove the underscores from the names of contraception methods
use_costs_table_df.index = use_costs_table_df.index.map(lambda s: s.replace("_", " "))

if print_bool:
    print("\n")
    print("TABLE")
    fullprint(use_costs_table_df)

output_table_file = r"outputs/output_table_" + use_output + "-use_costs" +\
                    "__" + ID_without + "_" + ID_with +\
                    ".xlsx"
writer = pd.ExcelWriter(output_table_file)
# Mean use rounded to two decimals for the table
use_costs_table_df.to_excel(writer, index_label=use_costs_table_df.columns.name)

writer.save()
print("\n")
print("Tab: Contraception Use (total & percentage) & Costs within Time Periods With and Without Interventions saved.")

time_end = time.time()

print("\n")
print("running time (s):")
print(time_end - time_start)
