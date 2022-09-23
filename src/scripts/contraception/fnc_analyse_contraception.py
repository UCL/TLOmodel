"""inspired by analysis_contraception.py

a function 'analyse contraception' defined to be used for pre-simulated data
(using a scenario file run_analysis_contraception.py) by another script
(analysis_contraception_plot_table.py) to plot use of contraception over time,
use of contraception methods over time, pregnancies over time, and/or calculate
data for a table of use and costs of contraception methods (if required)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from collections import Counter
from tlo.analysis.utils import parse_log_file


def analyse_contraception(in_datestamp, in_log_file,
                          in_plot_use_time_bool,
                          in_plot_use_time_method_bool,
                          in_plot_pregnancies_bool,
                          in_calc_use_costs_bool,
                          in_required_time_period_starts=[],
                          in_use_output="mean"):
    """

    :param in_datestamp: datestamp to be included in output files names
    :param in_log_file: log file from which the simulations logging is downloaded
    :param in_plot_use_time_bool: True if we want to plot use of contraception
        over time
    :param in_plot_use_time_method_bool: True if we want to plot use of
        contraception methods over time
    :param in_plot_pregnancies_bool: True if we want to plot pregnancies over
        time
    :param in_calc_use_costs_bool: True if we want to calculate use and costs of
        contraception methods in time periods
    :param in_required_time_period_starts: time periods specified if
        'in_calc_use_costs_bool' is True (default: [] -- as we don't need it if
         'in_calc_use_costs_bool' is False)
    :param in_use_output: "mean" or "max", according to which output of numbers,
        and percentage of women using contraception methods we want to display
        in the table (default: "mean")

    :return: Three data frames by time periods:
        number of women using contraception methods,
        percentage of women using contraception methods,
        costs of contraception methods
    """

    def fullprint(in_to_print):  # TODO: remove
        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):
            print(in_to_print)

    # Where will outputs go - by default, wherever this script is run
    outputpath = Path("./outputs")  # folder for convenience of storing outputs

    # Load without simulating again - parse the simulation logfile to get the
    # output dataframes
    log_df = parse_log_file('outputs/' + in_log_file)

    # %% Plot Contraception Use Over time:
    if in_plot_use_time_bool:
        years = mdates.YearLocator()  # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')

        # Load Model Results
        co_df = log_df['tlo.methods.contraception']['contraception_use_summary'].set_index('date')
        Model_Years = pd.to_datetime(co_df.index)
        Model_total = co_df.sum(axis=1)
        Model_not_using = co_df.not_using
        Model_using = Model_total - Model_not_using

        fig, ax = plt.subplots()
        ax.plot(np.asarray(Model_Years), Model_total)
        ax.plot(np.asarray(Model_Years), Model_not_using)
        ax.plot(np.asarray(Model_Years), Model_using)
        # plt.plot(Data_Years, Data_Pop_Normalised)

        # format the ticks
        # ax.xaxis.set_major_locator(years)
        # ax.xaxis.set_major_formatter(years_fmt)

        plt.title("Contraception Use")
        plt.xlabel("Year")
        plt.ylabel("Number of women")
        # plt.gca().set_xlim(Date(2010, 1, 1), Date(2013, 1, 1))
        plt.legend(['Total women age 15-49 years', 'Not Using Contraception', 'Using Contraception'])
        plt.savefig(outputpath / ('Contraception Use' + in_datestamp + '.png'), format='png')
        plt.show()

    # %% Plot Contraception Use By Method Over time:
    if in_plot_use_time_method_bool:
        years = mdates.YearLocator()  # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')

        # Load Model Results
        com_df = log_df['tlo.methods.contraception']['contraception_use_summary']
        Model_Years = pd.to_datetime(com_df.date)  # TODO: confusing name, as these are dates not just years
        Model_pill = com_df.pill
        Model_IUD = com_df.IUD
        Model_injections = com_df.injections
        Model_implant = com_df.implant
        Model_male_condom = com_df.male_condom
        Model_female_sterilization = com_df.female_sterilization
        Model_other_modern = com_df.other_modern
        Model_periodic_abstinence = com_df.periodic_abstinence
        Model_withdrawal = com_df.withdrawal
        Model_other_traditional = com_df.other_traditional

        fig, ax = plt.subplots()
        ax.plot(np.asarray(Model_Years), Model_pill)
        ax.plot(np.asarray(Model_Years), Model_IUD)
        ax.plot(np.asarray(Model_Years), Model_injections)
        ax.plot(np.asarray(Model_Years), Model_implant)
        ax.plot(np.asarray(Model_Years), Model_male_condom)
        ax.plot(np.asarray(Model_Years), Model_female_sterilization)
        ax.plot(np.asarray(Model_Years), Model_other_modern)
        ax.plot(np.asarray(Model_Years), Model_periodic_abstinence)
        ax.plot(np.asarray(Model_Years), Model_withdrawal)
        ax.plot(np.asarray(Model_Years), Model_other_traditional)

        # format the ticks
        # ax.xaxis.set_major_locator(years)
        # ax.xaxis.set_major_formatter(years_fmt)

        plt.title("Contraception Use By Method")
        plt.xlabel("Year")
        plt.ylabel("Number using method")
        # plt.gca().set_ylim(0, 50)
        # plt.gca().set_xlim(Date(2010, 1, 1), Date(2013, 1, 1))
        plt.legend(['pill', 'IUD', 'injections', 'implant', 'male_condom', 'female_sterilization',
                    'other_modern', 'periodic_abstinence', 'withdrawal', 'other_traditional'])
        plt.savefig(outputpath / ('Contraception Use By Method' + in_datestamp + '.png'), format='png')
        plt.show()

    # %% Plot Pregnancies Over time:
    if in_plot_pregnancies_bool:
        years = mdates.YearLocator()  # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')

        # Load Model Results
        preg_df = log_df['tlo.methods.contraception']['pregnancy'].set_index('date')
        preg_df.index = pd.to_datetime(preg_df.index).year
        num_pregs_by_year = preg_df.groupby(by=preg_df.index).size()
        Model_Years = num_pregs_by_year.index
        Model_pregnancy = num_pregs_by_year.values


        fig, ax = plt.subplots()
        ax.plot(np.asarray(Model_Years), Model_pregnancy)


        # format the ticks
        # ax.xaxis.set_major_locator(years)
        # ax.xaxis.set_major_formatter(years_fmt)

        plt.title("Pregnancies Over Time")
        plt.xlabel("Year")
        plt.ylabel("Number of pregnancies")
        # plt.gca().set_ylim(0, 50)
        # plt.gca().set_xlim(Date(2010, 1, 1), Date(2013, 1, 1))
        plt.legend(['total', 'pregnant', 'not_pregnant'])
        plt.savefig(outputpath / ('Pregnancies Over Time' + in_datestamp + '.png'), format='png')
        plt.show()

    # %% Calculate Use and Consumables Costs of Contraception methods within
    # some time periods:
    if in_calc_use_costs_bool:

        assert in_required_time_period_starts != []

#  ###### USE ##################################################################
        # Load Contraception Use Results
        # ['date', 'IUD', 'female_sterilization', 'implant', 'injections',
        # 'male_condom', 'not_using', 'other_modern', 'other_traditional',
        # 'periodic_abstinence', 'pill', 'withdrawal']:
        co_use_df = log_df['tlo.methods.contraception']['contraception_use_summary']
        co_use_df['women_total'] = co_use_df.sum(axis=1)
        co_use_modern_df = co_use_df.loc[:,
                           ['date', 'IUD', 'female_sterilization', 'implant',
                            'injections', 'male_condom', 'other_modern', 'pill',
                            'women_total']
                           ].copy()
        co_use_modern_df['year'] = co_use_modern_df['date'].dt.year

        # Keep only records within required time periods and assign the time
        # periods they belong to
        def keep_data_required_time_period(in_l_time_period_start, in_df_data):
            """
            Only keeps the data in required time periods, ie between first
            (inclusive) and last (exclusive) year from in_l_time_period_start.

            :param in_l_time_period_start: list of time period starts, dividing
                individual time periods
            :param in_df_data: all data, which we want to limit to required time
                periods
            :return: Only those data within required time periods.
            """
            return in_df_data.loc[
                (in_l_time_period_start[0] <= in_df_data['year']) &
                (in_df_data['year'] < in_l_time_period_start[-1])
            ].copy()

        def assign_time_period(in_l_time_period_start, in_l_year):
            """
            Assigns a time period to each individual record, according to the
            list of time period starts (in_l_time_period_start) and the years
            when the records were performed (in_l_year). All records in input
            should be within these years.

            :param in_l_time_period_start: list of time period starts, dividing
               individual time periods
            :param in_l_year: list of years when the records were performed
            :return: List of time periods to which the records belong.
            """

            in_l_year = list(map(int, in_l_year))
            # time period starts should be ordered
            assert all(in_l_time_period_start[i] <= in_l_time_period_start[i+1]
                       for i in range(len(in_l_time_period_start) - 1))
            # all records in input should be from the required time periods
            assert all((in_l_time_period_start[0] <= in_l_year[j]) &
                       (in_l_year[j] < in_l_time_period_start[-1])
                       for j in range(len(in_l_year))
                       )
            l_time_period = []
            for y in in_l_year:
                time_period_i = next(i for i, v in enumerate(in_l_time_period_start) if v > y)
                l_time_period.append(
                    str(in_l_time_period_start[time_period_i-1]) +
                    "-" + str(in_l_time_period_start[time_period_i]-1)
                )
            return l_time_period

        def create_time_period_data(in_l_time_period_start, in_df):
            tp_df =\
                keep_data_required_time_period(in_l_time_period_start, in_df)

            tp_df['Time_Period'] =\
                assign_time_period(in_l_time_period_start, tp_df.loc[:, 'year'])

            return tp_df

        co_use_modern_tp_df = \
            create_time_period_data(in_required_time_period_starts,
                                    co_use_modern_df)

        co_use_modern_tp_df = \
            co_use_modern_tp_df.loc[:,
            ((co_use_modern_tp_df.columns != 'date') &
             (co_use_modern_tp_df.columns != 'year'))]

        def create_percentage_use_df(in_df_use_incl_women_total):
            df_percentage_use = in_df_use_incl_women_total.copy()
            df_percentage_use.iloc[:, :-1] = df_percentage_use.iloc[:, :-1]\
                .div(df_percentage_use['women_total'], axis=0).mul(100, axis=0)
            # we can exclude the column with women_total as it is no more
            # needed, but it's nicely seen in it, that it does work
            return df_percentage_use

        co_percentage_use_df = create_percentage_use_df(co_use_modern_tp_df)

        def sum_use_all_times(in_df_use_by_tp, in_output_type):
            """
            Adds a row with mean/max use in all time periods.
            :param in_df_use_by_tp: 'Time_Period' as index, mean/max use of
                contraception methods in columns
            :param in_output_type: The type of output, "mean" or "max".

            :return: The sum row to append to the input df.
            """
            # tp of all times, ie very first to very last year of time periods
            for r, i in enumerate(in_df_use_by_tp.index):
                if r == 0:
                    y_first = i.split("-")[0]
                if r == (len(in_df_use_by_tp) - 1):
                    y_last = i.split("-")[1]
            sum_tp = (str(y_first) + "-" + str(y_last))
            l_summation = []
            for c in in_df_use_by_tp:
                # outputs (min/max) for contraceptives and women_total within all times
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
            # store copy as mean_use to work with it separately
            mean_use_df = co_output_use_modern_tp_df.copy()
            # Include the output summation for all time periods:
            co_output_use_modern_tp_df = \
                co_output_use_modern_tp_df \
                    .append(sum_use_all_times(co_output_use_modern_tp_df, in_use_output))
            # print("OUTPUT MEAN/MAX USE WITH ALL TIMES SUM")
            # fullprint(co_output_use_modern_tp_df
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

        # Add a column with the nmb of years within the time periods
        def calculate_tp_len(in_a_tp_as_string):
            l_tp_len = []
            for tp in in_a_tp_as_string:
                l_first_last_y = [int(x) for x in tp.split("-")]
                l_tp_len.append(l_first_last_y[1] - l_first_last_y[0] + 1)
            return l_tp_len

        # Add length of time periods to mean_use_df
        mean_use_df['tp_len'] = calculate_tp_len(mean_use_df.index)

#  ###### CONSUMABLES ##########################################################
        # Load Consumables results
        cons_df = log_df['tlo.methods.healthsystem']['Consumables'].copy()
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

        # # All individual requests
        # print(cons_df['Request'].value_counts(dropna=False)) # TODO: remove
        # # All records of Item_Available
        # fullprint(cons_df['Item_Available'].value_counts(dropna=False)) # TODO: remove

        # Limit consumables data to those which were processed (contraception
        # was given to a woman as all items were available)
        cons_processed_df = cons_df.loc[cons_df['Item_NotAvailable'] == "{}"].copy()

        # Assign a contraceptive method to each record according to the request.
        resource_items_pkgs_contraception_only_df = pd.read_csv(
            'src/scripts/contraception/Consumables_Items_and_Packages_contraceptionOnly_MadeAccordingToModel.csv'
        )  # TODO: Use this in the function below.

        def get_contraceptive_method_for_request(in_d):
        # TODO: any chance to take these from the model, or make a file from
        #  which both model and visualisation will take it, so if any changes
        #  are done in future, it will be done only once.
        # TODO: for now create dictionaries for contraceptives methods from the
        #  CSV I created - resources_df
            if in_d == dict({0: 8, 1: 8}):  # TODO: change it in the table to 8s and remove the rounding up
                return 'pill'
            if in_d == dict({2: 120}):
                return 'male_condom'
            if in_d == dict({25: 120}):
                return 'other_modern'
            if in_d == dict({7: 1}):
                return 'IUD'
            if in_d == dict({3: 1, 5: 1, 6: 1}):
                return 'injections'
            if in_d == dict({5: 1, 8: 1, 12: 1, 13: 1, 247: 1}):
                return 'implant'
            if in_d == dict({5: 1, 14: 1, 15: 1, 16: 1, 17: 1, 21: 1, 23: 1, 101: 1, 247: 1}):
                return 'female_sterilization'
            else:
                raise ValueError(
                    "There is an unrecognised request: " + str(in_d) + "."
                )
            # TODO: in resource file, there are expected units 7.5 for both, but
            #  when getting them from the file in the model, they are rounded up
            #  and their type changed to int (but for some reason it looks that
            #  they are rounded up indeed, but their type is not integer as for
            #  male_condom, I see 120.0 units to be logged and for pills 8.0)
            #  SEE consumables_L208-210:
            #  ser = lookup_df.loc[
            #     lookup_df['Intervention_Pkg'] == package, ['Item_Code', 'Expected_Units_Per_Case']].set_index(
            #     'Item_Code')['Expected_Units_Per_Case'].apply(np.ceil).astype(int)

        def get_list_contraceptive_methods(in_l_requests):
            l_contraceptive_methods = []
            for d in in_l_requests:
                l_contraceptive_methods.append(
                    get_contraceptive_method_for_request(d)
                )
            return l_contraceptive_methods

        cons_processed_df['Contraceptive_Method'] =\
            get_list_contraceptive_methods(cons_processed_df['Request'])

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
        def sum_dict_available(in_a_l_d_as_string):
            """
            Transforms multiple dictionaries per method per time period to one
            joint dictionary.

            :param in_a_l_d_as_string: array with indexes ('Time_Period' and
                'Contraceptive_Method') and lists of dictionaries written as
                string ('Item_Available')
            :return: Data frame with 'Time_Period' and 'Contraceptive_Method'
                and  one dictionary per method per time period written as dict
                'Items_Available_forMethod_inTimePeriod'.
            """
            df_item_avail = pd.DataFrame(in_a_l_d_as_string)
            df_item_avail['Items_Available_dicts'] = ""
            # type(y) = <class 'int'>
            for y, l_d_as_string in df_item_avail['Item_Available'].items():
                l_dicts = []
                for d_as_string in l_d_as_string:
                    l_dicts.append(eval(d_as_string))
                df_item_avail['Items_Available_dicts'][y] = l_dicts
            df_item_avail['Items_Available_inTimePeriod'] = df_item_avail['Items_Available_dicts'].apply(merge_dicts)
            return df_item_avail['Items_Available_inTimePeriod']

        cons_avail_grouped_by_time_and_method_df['Item_Available_summation'] =\
            sum_dict_available(cons_avail_grouped_by_time_and_method_df .loc[:, 'Item_Available'])

        # Calculate the costs of available items for all except male_condom
        # & other_modern (= female condom only currently)
        # TODO: change if Male sterilization is modelled as other_modern as well
        def calculate_costs(in_df_resource_items_pkgs,
                            in_df_cons_avail_by_time_and_method,
                            in_df_mean_use):
            """
            Calculates costs of available items per time period per method.

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
                # if the method == males or females condoms calculate from the
                # mean numbers of women using them (there is only one item for
                # condoms)
                if (i[1] == "male_condom") | (i[1] == "other_modern"):
                    unit_cost = float(in_df_resource_items_pkgs['Unit_Cost']
                                      .loc[in_df_resource_items_pkgs['Contraceptive_Method']
                                           == i[1]])
                    # costs = unit_cost *
                    # nmb of years within the time period (tp_len) *
                    # 2/3 of 365.25 days as approximation of number of condom used per year *
                    # mean nmb of women using
                    costs = unit_cost *\
                            int(in_df_mean_use['tp_len'].loc[in_df_mean_use.index == i[0]]) *\
                            2 / 3 * 365.25 *\
                            float(in_df_mean_use[i[1]].loc[in_df_mean_use.index == i[0]])
                # otherwise calculate from the logs
                elif i[1] == "pill":
                    item_avail_dict = in_df_cons_avail_by_time_and_method.loc[
                        i, 'Item_Available_summation'
                    ]
                # TODO: this version is only for sims where all resources are
                #  always available (ie health.system with
                #  cons_availability="all"), and with a knowledge that there are
                #  items 0 or 1 assigned with contraception method 'pill', where
                #  item 1 (=combined pills) is used in main cases, and item 0
                #  (progestin-only pills) only in a few special cases. Better,
                #  in future when pills required, choose which one is used in a
                #  probabilistic manner, then check the availability and then do
                #  the logging if available.
                    # 2.8% of pills users using progestin-only pills
                    # [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3440515]
                    # - study from United States from 2012)
                    # TODO: get real data on this or find a better approximation
                    costs = 0
                    prob_progestin_only = 0.028
                    for time_method_key in list(item_avail_dict.keys()):
                        # costs of pills = prob_progestin_only of the price of
                        # progestin-only pills + (1 - prob_progestin_only) of
                        # the price of combined pills
                        unit_cost = (prob_progestin_only * (1-time_method_key) +\
                                     (1 - prob_progestin_only) * time_method_key) *\
                                    float(in_df_resource_items_pkgs['Unit_Cost'].loc[
                                              (in_df_resource_items_pkgs['Contraceptive_Method'] == i[1]) &
                                              (in_df_resource_items_pkgs[ 'Item_Code'] == time_method_key)])
                        costs = costs + (unit_cost * item_avail_dict[time_method_key])
                else:
                    item_avail_dict = in_df_cons_avail_by_time_and_method.loc[
                        i, 'Item_Available_summation'
                    ]
                    costs = 0
                    for time_method_key in list(item_avail_dict.keys()):
                        unit_cost = float(in_df_resource_items_pkgs['Unit_Cost'].loc[
                                (in_df_resource_items_pkgs['Contraceptive_Method'] == i[1]) &
                                (in_df_resource_items_pkgs['Item_Code'] == time_method_key)])
                        costs = costs + (unit_cost * item_avail_dict[time_method_key])
                l_costs.append(costs)
            return l_costs

        cons_avail_grouped_by_time_and_method_df['Costs'] =\
            calculate_costs(resource_items_pkgs_contraception_only_df,
                            cons_avail_grouped_by_time_and_method_df, mean_use_df)

        def sum_costs_all_times(in_df_costs_by_tp):
            """
            Adds a row with sum of costs in all time periods.
            :param in_df_costs_by_tp: 'Time_Period' and 'Contraceptive_Method'
                as index, 'Costs' as a column.

            :return: The sum rows for each contraceptive method to append to the
                input df.
            """
            # tp of all times, ie very first to very last year of time periods
            for r, i in enumerate(in_df_costs_by_tp.index):
                if r == 0:
                    y_first = i[0].split("-")[0]
                if r == (len(in_df_costs_by_tp) - 1):
                    y_last = i[0].split("-")[1]
            sum_tp = (str(y_first) + "-" + str(y_last))
            # sum the costs in all time periods for each contraceptive method
            sum_costs = in_df_costs_by_tp.groupby(level=[1]).sum()
            return pd.DataFrame(list(sum_costs.loc[:, 'Costs']),
                                columns=['Costs'],
                                index=[[sum_tp] * len(sum_costs.index),
                                       sum_costs.index]
                                )

        cons_costs_by_time_and_method_df =\
            cons_avail_grouped_by_time_and_method_df.loc[:, 'Costs'].copy().to_frame()
        print(type(cons_costs_by_time_and_method_df))
        cons_costs_by_time_and_method_df =\
            cons_costs_by_time_and_method_df\
                .append(sum_costs_all_times(cons_costs_by_time_and_method_df))

    return co_output_use_modern_tp_df, co_output_percentage_use_df,\
           cons_costs_by_time_and_method_df

    ############################################################################
    # What follows is TimH's code for this section:
    # def unpack(in_dict_as_string):
    #     in_dict = eval(in_dict_as_string)
    #     l = list()
    #     for k, v in in_dict.items():
    #         for _v in range(v):
    #             l.append(k)
    #     return l
    #
    # pkg_counts = cons['Package_Available'].apply(unpack).apply(pd.Series).dropna().astype(int)[0].value_counts()

    # What follows is TimC's original code for this section:
    # ...
    # ...
    # years = mdates.YearLocator()  # every year
    # months = mdates.MonthLocator()  # every month
    # years_fmt = mdates.DateFormatter('%Y')
    #
    # # Load Model Results
    # com_df = log_df['tlo.methods.contraception']['contraception_use_yearly_summary']
    # Model_Years = pd.to_datetime(com_df.date)
    # Model_pill = com_df.pill
    # Model_IUD = com_df.IUD
    # Model_injections = com_df.injections
    # Model_implant = com_df.implant
    # Model_male_condom = com_df.male_condom
    # Model_female_sterilization = com_df.female_sterilization
    # Model_other_modern = com_df.other_modern
    #
    # fig, ax = plt.subplots()
    # ax.plot(np.asarray(Model_Years), Model_pill)
    # ax.plot(np.asarray(Model_Years), Model_IUD)
    # ax.plot(np.asarray(Model_Years), Model_injections)
    # ax.plot(np.asarray(Model_Years), Model_implant)
    # ax.plot(np.asarray(Model_Years), Model_male_condom)
    # ax.plot(np.asarray(Model_Years), Model_female_sterilization)
    # ax.plot(np.asarray(Model_Years), Model_other_modern)
    #
    # # format the ticks
    # # ax.xaxis.set_major_locator(years)
    # # ax.xaxis.set_major_formatter(years_fmt)
    #
    # plt.title("Contraception Consumables By Method")
    # plt.xlabel("Year")
    # plt.ylabel("Consumables used (number using method")
    # # plt.gca().set_xlim(Date(2010, 1, 1), Date(2013, 1, 1))
    # plt.legend(['pills', 'IUDs', 'injections', 'implants', 'male_condoms', 'female_sterilizations',
    #             'other modern'])
    # plt.savefig(outputpath / ('Contraception Consumables By Method' + in_datestamp + '.png'), format='png')
    # plt.show()
    #
    # # %% Plot Consumable Costs Over time:
    #
    # years = mdates.YearLocator()  # every year
    # months = mdates.MonthLocator()  # every month
    # years_fmt = mdates.DateFormatter('%Y')
    #
    # # Load Model Results
    # com_df = log_df['tlo.methods.contraception']['contraception_costs_yearly_summary']  #TODO: I didn't find any costs there, maybe this was in older version?
    # Model_Years = pd.to_datetime(com_df.date)
    # Model_pill = com_df.pill_annual_cost
    # Model_IUD = com_df.IUD_annual_cost
    # Model_injections = com_df.injections_annual_cost
    # Model_implant = com_df.implant_annual_cost
    # Model_male_condom = com_df.male_condom_annual_cost
    # Model_female_sterilization = com_df.female_sterilization_annual_cost
    # Model_other_modern = com_df.other_modern_annual_cost
    #
    # fig, ax = plt.subplots()
    # ax.plot(np.asarray(Model_Years), Model_pill)
    # ax.plot(np.asarray(Model_Years), Model_IUD)
    # ax.plot(np.asarray(Model_Years), Model_injections)
    # ax.plot(np.asarray(Model_Years), Model_implant)
    # ax.plot(np.asarray(Model_Years), Model_male_condom)
    # ax.plot(np.asarray(Model_Years), Model_female_sterilization)
    # ax.plot(np.asarray(Model_Years), Model_other_modern)
    #
    # # format the ticks
    # # ax.xaxis.set_major_locator(years)
    # # ax.xaxis.set_major_formatter(years_fmt)
    #
    # plt.title("Contraception Consumable Costs By Method")
    # plt.xlabel("Year")
    # plt.ylabel("Consumable Costs (Cumulative)")
    # # plt.gca().set_xlim(Date(2010, 1, 1), Date(2013, 1, 1))
    # plt.legend(['pill costs', 'IUD costs', 'injection costs', 'implant costs', 'male condom costs',
    #             'female sterilization costs', 'other modern method costs'])
    # plt.savefig(outputpath / ('Contraception Consumable Costs By Method' + in_datestamp + '.png'), format='png')
    # plt.show()
    #
    # # %% Plot Public Health Costs Over time:
    #
    # years = mdates.YearLocator()  # every year
    # years_fmt = mdates.DateFormatter('%Y')
    #
    # # Load Model Results
    # com_df = log_df['tlo.methods.contraception']['contraception_costs_yearly_summary']
    # Model_Years = pd.to_datetime(com_df.date)
    # Model_public_health_costs1 = com_df.public_health_costs1
    # Model_public_health_costs2 = com_df.public_health_costs2
    #
    # fig, ax = plt.subplots()
    # ax.plot(np.asarray(Model_Years), Model_public_health_costs1)
    # ax.plot(np.asarray(Model_Years), Model_public_health_costs2)
    #
    # # format the ticks
    # # ax.xaxis.set_major_locator(years)
    # # ax.xaxis.set_major_formatter(years_fmt)
    #
    # plt.title("Public Health Costs for Contraception uptake")
    # plt.xlabel("Year")
    # plt.ylabel("Cost")
    # # plt.gca().set_xlim(Date(2010, 1, 1), Date(2013, 1, 1))
    # plt.legend(['population scope campaign to increase contraception initiation',
    #             'post partum family planning (PPFP) campaign'])
    # plt.savefig(outputpath / ('Public Health Costs' + in_datestamp + '.png'), format='png')
    # plt.show()
######################


if __name__ == '__main__':
    analyse_contraception(in_datestamp, in_output_file,
                          in_plot_use_time_bool, in_plot_use_time_method_bool,
                          in_plot_pregnancies_bool,
                          in_calc_use_costs_bool,
                          in_required_time_period_starts)
