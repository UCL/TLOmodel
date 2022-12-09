"""inspired by analysis_contraception.py

a function 'analyse contraception' defined to be used for pre-simulated data
(using a scenario file run_analysis_contraception.py) by another script
(analysis_contraception_plot_table.py) to plot use of contraception over time,
use of contraception methods over time, pregnancies over time, and/or calculate
data for a table of use and costs of contraception methods (if required)
"""
import logging
import timeit
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from collections import Counter
from tlo.analysis.utils import parse_log_file
import functools


def analyse_contraception(in_datestamp: str, in_log_file: str,
                          in_pop_size_multiplier: float,
                          in_plot_use_time_bool: bool = False,
                          in_plot_use_time_method_bool: bool = False,
                          in_plot_pregnancies_bool: bool = False,
                          in_set_ylims_bool: bool = False, in_ylims_l: list = [],
                          in_calc_use_costs_bool: bool = False, in_required_time_period_starts: list = [],
                          in_contraceptives_order: list = ['pill', 'IUD', 'injections', 'implant','male_condom',
                                                           'female_sterilization', 'other_modern'],
                          in_calc_intervention_costs_bool: bool = False,
                          in_use_output: str = "mean"
                          ):
    """
    Performs analysis of contraception for pre-simulated data (data given by
    'in_log_file'), saves figure(s) and/or calculates contraception use and
    costs to be included in a table, according to what is (not) required
    (requirements set by inputs in_xx_bool). The name of output figs includes
     'in_datestamp' to be assigned to correct simulations.

    :param in_datestamp: datestamp to be included in output files names
    :param in_log_file: log file from which the simulations logging is
        downloaded
    :param in_pop_size_multiplier:
    :param in_plot_use_time_bool: True if we want to plot use of any
        contraception over time (default: False)
    :param in_plot_use_time_method_bool: True if we want to plot use of
        individual contraception methods over time (default: False)
    :param in_plot_pregnancies_bool: True if we want to plot pregnancies over
        time (default: False)
    :param in_set_ylims_bool: True if we want to set upper limits for the y-axes
        for the 3 plots. (default: False)
    :param in_ylims_l: list of the upper limits for y-axes of the figures in the
        order [Use, Use By Method, Pregnancies] (default: [] -- as we don't need
        it if 'in_set_ylims_bool' is False)
    :param in_calc_use_costs_bool: True if we want to calculate use and costs of
        contraception methods in time periods (time periods
        'in_required_time_period_starts' needs to be given as input)
        (default: False)
    :param in_required_time_period_starts: a list of years determining the time
        periods for which we require the calculations, first year inc.,
        last year excl. (default: [] -- as we don't need it if
        'in_calc_use_costs_bool' is False)
    :param in_contraceptives_order: list of modern contraceptives ordered as we
        want them to appear in the table
    :param in_calc_intervention_costs_bool: True if we want to calculate
        contraception Pop and PPFP intervention costs over time (default: False)
    :param in_use_output: "mean" or "max", according to which output of numbers,
        and percentage of women using contraception methods we want to display
        in the table (default: "mean")


    :return: Three data frames by time periods:
        number of women using contraception methods,
        percentage of women using contraception methods,
        costs of contraception methods
        (if 'in_calc_use_costs_bool' is False, returns 3 empty lists)
    """

    def fullprint(in_to_print):  # TODO: remove
        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):
            print(in_to_print)

    timeit_rep_nmb = 600

    def timeitprint(in_what_measures: str, in_fnc, in_timeit_rep_nmb=1):  # TODO: remove
        if in_timeit_rep_nmb > 1:
            print("time (s) of " + in_what_measures +
                  " (" + str(in_timeit_rep_nmb) + " repetitions):")
        else:
            print("time (s) of " + in_what_measures + ":")
        print(timeit.timeit(in_fnc, number=in_timeit_rep_nmb))

    # Where will outputs go - by default, wherever this script is run
    outputpath = Path("./outputs")  # folder for convenience of storing outputs

    # Load without simulating again - parse the simulation logfile to get the
    # output dataframes
    log_df = parse_log_file('outputs/' + in_log_file, level=logging.DEBUG)

    # %% Calculate annual Pop and PPFP intervention costs:
    if in_calc_intervention_costs_bool:
        # @@ Load Population Totals (Demography Model Results)
        # females 15-49 by year:
        demog_df_f = log_df['tlo.methods.demography']['age_range_f'].set_index('date')
        demog_df_f['15-49'] =\
            demog_df_f.loc[:, ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']].sum(axis=1)
        # print("demog_df_f.loc[:, ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '15-49']]")
        # print(demog_df_f.loc[:, ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '15-49']])
        # males 15-49 by year:
        demog_df_m = log_df['tlo.methods.demography']['age_range_m'].set_index('date')
        demog_df_m['15-49'] =\
            demog_df_m.loc[:, ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']].sum(axis=1)
        # print("demog_df_m.loc[:, ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '15-49']]")
        # print(demog_df_m.loc[:, ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '15-49']])
        # total females & males 15-49 as both targeted by Pop and PPFP interventions, by year:
        popsize1549 = demog_df_f['15-49'] + demog_df_m['15-49']
        popsize1549 = pd.DataFrame(popsize1549)
        # print("popsize1549")
        # print(popsize1549)
        # Calculate ratio of population compared to 2016 as base year (when PPFP and Pop interventions start):
        popsize1549['ratio'] = popsize1549.loc[:, '15-49'] / popsize1549.loc['2016-01-01 00:00:00', '15-49']
        # print("popsize1549")
        # print(popsize1549)
        # Mulitply PPFP and Pop intervention costs by this ratio for each year:
        # TODO: was trying to pull these parameters from contracption.py but not managing too
        #  - perhaps not worth it and just hard code for now:
        # ppfp_intervention_cost = tlo.methods.contraception.Contraception.process_params(['ppfp_intervention_cost'])
        # cost of PPFP intervention for whole population of Malawi in 2016 (MWK - Malawi Kwacha)
        ppfp_interv_cost_2016 = 146000000
        # cost of Pop intervention for whole population of Malawi in 2016 (MWK - Malawi Kwacha)
        pop_interv_cost_2016 = 1300000000
        # print("ppfp_interv_cost_2016")
        # print(ppfp_interv_cost_2016)
        # print("pop_interv_cost_2016")
        # print(pop_interv_cost_2016)
        popsize1549['ppfp_intervention_cost'] = popsize1549['ratio'] * ppfp_interv_cost_2016
        popsize1549['pop_intervention_cost'] = popsize1549['ratio'] * pop_interv_cost_2016
        popsize1549['interventions_total'] =\
            popsize1549['ppfp_intervention_cost'] + popsize1549['pop_intervention_cost']
        # print("popsize1549.loc[:, ['ratio', 'ppfp_intervention_cost', 'pop_intervention_cost', 'interventions_total']]")
        # with pd.option_context('display.max_columns', None):
        #     print(
        #         popsize1549.loc[:, ['ratio', 'ppfp_intervention_cost', 'pop_intervention_cost', 'interventions_total']]
        #     )
        # print("popsize1549['ppfp_intervention_cost']")
        # print(popsize1549['ppfp_intervention_cost'])
        # print("popsize1549['pop_intervention_cost']")
        # print(popsize1549['pop_intervention_cost'])
        # print("popsize1549['interventions_total']")
        # print(popsize1549['interventions_total'])

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
        ax.plot(np.asarray(Model_Years), Model_total * in_pop_size_multiplier)
        ax.plot(np.asarray(Model_Years), Model_not_using * in_pop_size_multiplier)
        ax.plot(np.asarray(Model_Years), Model_using * in_pop_size_multiplier)
        if in_set_ylims_bool:
            ax.set_ylim([0, in_ylims_l[0]])
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
        # plt.show()
        print("Fig: Contraception Use Over time saved.")

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
        ax.plot(np.asarray(Model_Years), Model_pill * in_pop_size_multiplier)
        ax.plot(np.asarray(Model_Years), Model_IUD * in_pop_size_multiplier)
        ax.plot(np.asarray(Model_Years), Model_injections * in_pop_size_multiplier)
        ax.plot(np.asarray(Model_Years), Model_implant * in_pop_size_multiplier)
        ax.plot(np.asarray(Model_Years), Model_male_condom * in_pop_size_multiplier)
        ax.plot(np.asarray(Model_Years), Model_female_sterilization * in_pop_size_multiplier)
        ax.plot(np.asarray(Model_Years), Model_other_modern * in_pop_size_multiplier)
        ax.plot(np.asarray(Model_Years), Model_periodic_abstinence * in_pop_size_multiplier)
        ax.plot(np.asarray(Model_Years), Model_withdrawal * in_pop_size_multiplier)
        ax.plot(np.asarray(Model_Years), Model_other_traditional * in_pop_size_multiplier)
        if in_set_ylims_bool:
            ax.set_ylim([0, in_ylims_l[1]])

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
        # plt.show()
        print("Fig: Contraception Use By Method Over time saved.")

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
        ax.plot(np.asarray(Model_Years), Model_pregnancy * in_pop_size_multiplier)
        if in_set_ylims_bool:
            ax.set_ylim([0, in_ylims_l[2]])

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
        # plt.show()
        print("Fig: Pregnancies Over time saved.")

    # %% Calculate Use and Consumables Costs of Contraception methods within
    # some time periods:
    if in_calc_use_costs_bool:  # TODO: add population scaling to use and consts calculations
        # time period starts should be given as input
        assert in_required_time_period_starts != [],\
            "The calculations of use and costs are requested (ie input 'in_calc_use_costs_bool' set as True, " +\
            "but no periods starts are provided (ie input 'TimePeriods_starts' is empty)."

        # this input needs to include at least 3 values
        assert len(in_required_time_period_starts) > 2,\
            "The input 'TimePeriods_starts' needs to include at least 3 years to define at least 2 time periods."
        # time period starts should be ordered
        assert all(in_required_time_period_starts[i] <= in_required_time_period_starts[i + 1]
                   for i in range(len(in_required_time_period_starts) - 1)),\
            "The 'TimePeriods_starts' needs to be in chronological order."

#  ###### USE ##################################################################
        # Load Contraception Use Results
        # ['date', 'IUD', 'female_sterilization', 'implant', 'injections',
        # 'male_condom', 'not_using', 'other_modern', 'other_traditional',
        # 'periodic_abstinence', 'pill', 'withdrawal']:
        co_use_df = log_df['tlo.methods.contraception']['contraception_use_summary']
        co_use_df['women_total'] = co_use_df.sum(axis=1)
        cols_to_keep = in_contraceptives_order.copy()
        cols_to_keep.append('date')
        cols_to_keep.append('women_total')
        co_use_modern_df = co_use_df.loc[:, cols_to_keep].copy()
        co_use_modern_df['co_modern_total'] = co_use_modern_df.loc[:, in_contraceptives_order].sum(axis=1)
        co_use_modern_df['year'] = co_use_modern_df['date'].dt.year
        print("Nmb of women at the initiation:", co_use_df['women_total'][0])
        print("Years simulated:",
              co_use_modern_df.loc[1, 'year'], "-", co_use_modern_df.loc[co_use_modern_df.shape[0]-1, 'year'])
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
            # # the input year should be from the required time periods
            # assert (in_l_time_period_start[0] <= in_year) & (in_year < in_l_time_period_start[-1])  # TODO: give it back? (how much time it takes?)

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
            :param in_df: the data with records that include the 'year' when the
                record was performed

            :return: A new data frame that includes only records from the required
                time periods, and includes a new column 'Time_Period'.
            """
            # keep data only from the required time periods
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
            co_use_modern_tp_df.loc[:,
            ((co_use_modern_tp_df.columns != 'date') &
             (co_use_modern_tp_df.columns != 'year'))]

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
            # store copy as mean_use to work with it separately
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

#  ###### CONSUMABLES ##########################################################
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

        # # All individual requests
        # print(cons_df['Request'].value_counts(dropna=False)) # TODO: remove
        # # All records of Item_Available
        # fullprint(cons_df['Item_Available'].value_counts(dropna=False)) # TODO: remove

        # Limit consumables data to those which were processed (contraception
        # was given to a woman as all items were available)
        # TODO: make it to work with essential and optional items
        #  (here we assume that all requested items are essential)
        cons_processed_df = cons_df.loc[cons_df['Item_NotAvailable'] == "{}"].copy()

        # Assign a contraceptive method to each record according to the request.
        resource_items_pkgs_df = pd.read_csv(
            'resources/healthsystem/consumables/ResourceFile_Consumables_Items_and_Packages.csv'
        )  # TODO: Use this in the function below.

        def get_contraceptive_method_for_request(in_d):
            """
            Based on a dictionary of requested items returns what contraception
            method was requested.

            :param in_d: a dictionary of requested items

            :return: Contraception method as string.
            """
        # TODO: Create dictionaries for contraception methods from the ResourceFile (resource_items_pkgs_df)
            if in_d == dict({1: 8}):
                return 'pill'
            if in_d == dict({2: 120}):
                return 'male_condom'
            if in_d == dict({25: 120}):
                return 'other_modern'
            if in_d == dict({4: 3, 7: 1}):
                return 'IUD'
            if in_d == dict({3: 1, 4: 1, 5: 1, 6: 1}):
                return 'injections'
            if in_d == dict({4: 2, 5: 2, 8: 2, 9: 2, 10: 0.1, 11: 0.2, 12: 0.5}):
                return 'implant'
            if in_d == dict(
                {5: 2, 9: 3, 14: 1, 15: 1, 16: 1, 17: 0.02, 18: 2, 19: 3, 20: 3, 21: 1, 22: 2, 23: 8, 24: 2}
            ):
                return 'female_sterilization'
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
                costs = 0
                if (i[1] == "male_condom") | (i[1] == "other_modern"):
                    unit_cost = float(in_df_resource_items_pkgs['Unit_Cost'].loc[
                                          in_df_resource_items_pkgs['Intervention_Pkg']
                                          == get_intervention_pkg_name(i[1])])
                    # costs = unit_cost *
                    # nmb of years within the time period (tp_len) *
                    # 2/3 of 365.25 days as approximation of number of condom used per year *
                    # mean nmb of women using
                    costs = unit_cost *\
                            int(in_df_mean_use['tp_len'].loc[in_df_mean_use.index == i[0]]) *\
                            2 / 3 * 365.25 *\
                            float(in_df_mean_use[i[1]].loc[in_df_mean_use.index == i[0]])
                # otherwise calculate from the logs
                else:
                    item_avail_dict = in_df_cons_avail_by_time_and_method.loc[
                        i, 'Item_Available_summation'
                    ]
                    for time_method_key in list(item_avail_dict.keys()):
                        unit_cost = float(in_df_resource_items_pkgs['Unit_Cost'].loc[
                                (in_df_resource_items_pkgs['Intervention_Pkg'] == get_intervention_pkg_name(i[1]))
                                & (in_df_resource_items_pkgs['Item_Code'] == time_method_key)])
                        costs = costs + (unit_cost * item_avail_dict[time_method_key])
                l_costs.append(costs)
            return l_costs

        # timeitprint("calc costs",
        #             functools.partial(calculate_costs,
        #                               resource_items_pkgs_df,
        #                               cons_avail_grouped_by_time_and_method_df,
        #                               mean_use_df),
        #             6000)
        # 26.302437201999055 s for 6000 repetitions
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
            # sum the costs in all time periods for each contraceptive method
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
    # %% If computations are not required:
    else:
        co_output_use_modern_tp_df, co_output_percentage_use_df, cons_costs_by_time_and_method_df = [], [], []

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
    analyse_contraception(in_datestamp, in_log_file,
                          in_pop_size_multiplier,
                          in_plot_use_time_bool,
                          in_plot_use_time_method_bool,
                          in_plot_pregnancies_bool,
                          in_set_ylims_bool, in_ylims_l,
                          in_calc_use_costs_bool, in_required_time_period_starts,
                          in_contraceptives_order,
                          in_calc_intervention_costs_bool,
                          in_use_output)
