"""
A script called from analysis_contraception_plot_table.py, containing functions
* table_use_costs__plot_costs() which creates the table of use and costs and plots the costs, and
* table_cons() which creates the table of consumables.
"""
import math
from datetime import date

import bar_chart_costs
import numpy as np
import pandas as pd


# %% TABLE USE & COSTS + FIGs COSTS ....................................................................................
def table_use_costs__plot_costs(in_use_output, in_use_without_df, in_percentage_use_without_df,
                                in_use_with_df, in_percentage_use_with_df, in_rounding_use_to,
                                in_costs_without_df, in_costs_with_df,
                                in_interv_costs_without_df, in_interv_costs_with_df,
                                in_mwk_to_usd_exchange_rate, in_rounding_costs_mwk_to, in_rounding_costs_usd_to,
                                in_plot_costs, in_datestamp_without_log, in_datestamp_with_log, in_suffix,
                                in_ID_without, in_ID_with):

    def round_df(in_df, in_rounding_to):
        if in_rounding_to:
            round_index = math.log10(in_rounding_to)  # TODO: fix the round_index warning
        else:
            round_index = math.log10(1)
        return (round(in_df, -int(round_index)) / in_rounding_to).astype(int)

    def format_df(in_df_rounded):
        df_formatted = pd.DataFrame()
        for col_name in in_df_rounded.columns:
            df_formatted[col_name] = in_df_rounded[col_name].map('{:,.0f}'.format)
        return df_formatted

    # TODO: it formats df even if not rounded
    def round_format(in_df, in_rounding_to):
        df_rounded = round_df(in_df, in_rounding_to)
        return format_df(df_rounded)

    # %% Join, Round & Format percentages and total values of use:
    use_without_perc_val_df = in_percentage_use_without_df.round(1).astype(str) + '%' + " (" + \
        round_format(in_use_without_df, in_rounding_use_to).astype(str) + ")"
    use_with_perc_val_df = in_percentage_use_with_df.round(1).astype(str) + '%' + " (" + \
        round_format(in_use_with_df, in_rounding_use_to).astype(str) + ")"

    # %% For each time period sum costs for all modern methods:
    def add_total_cons_costs_tp(in_costs_df):
        total_costs_without_tp = in_costs_df.groupby(level=0).sum()
        total_costs_without_tp.index =\
            pd.MultiIndex.from_tuples([(x, 'co_modern_total') for x in total_costs_without_tp.index])
        out_costs_df =\
            pd.concat([in_costs_df, total_costs_without_tp]).sort_index(level=[0, 1], sort_remaining=False)
        tp_order = pd.unique(in_use_without_df.index.get_level_values(0))
        # create a new dataframe with the new order of level 0 index
        out_costs_df_reordered = pd.DataFrame(index=pd.MultiIndex.from_product(
            [tp_order, out_costs_df.index.levels[1]]
        ), columns=out_costs_df.columns)
        # fill the new dataframe with the values from the original dataframe
        out_costs_df_reordered.loc[out_costs_df.index, :] = out_costs_df.values
        # drop rows with all NaN values
        out_costs_df_reordered.dropna(how='all', inplace=True)
        return out_costs_df_reordered

    in_costs_without_df = add_total_cons_costs_tp(in_costs_without_df)
    in_costs_with_df = add_total_cons_costs_tp(in_costs_with_df)

    # %% Sum all modern contraceptives + both interventions implementation for results with interventions
    interv_cons_total = []
    for tp in in_interv_costs_with_df.index:
        interv_cons_total.append(
            in_interv_costs_with_df.loc[tp, 'interventions_total'] + in_costs_with_df.loc[(tp, 'co_modern_total'),
                                                                                          'Costs']
        )
    in_interv_costs_with_df['interv_cons_total'] = interv_cons_total

    # %% Calculate costs in USD:
    costs_usd_without_df = in_costs_without_df * in_mwk_to_usd_exchange_rate
    costs_usd_with_df = in_costs_with_df * in_mwk_to_usd_exchange_rate
    interv_costs_usd_without_df = in_interv_costs_without_df * in_mwk_to_usd_exchange_rate
    interv_costs_usd_with_df = in_interv_costs_with_df * in_mwk_to_usd_exchange_rate

    # %% Round the cons & interv costs:
    if in_rounding_costs_mwk_to:
        in_costs_without_df = round_df(in_costs_without_df, in_rounding_costs_mwk_to)
        in_costs_with_df = round_df(in_costs_with_df, in_rounding_costs_mwk_to)
        in_interv_costs_without_df = round_df(in_interv_costs_without_df, in_rounding_costs_mwk_to)
        in_interv_costs_with_df = round_df(in_interv_costs_with_df, in_rounding_costs_mwk_to)
    if in_rounding_costs_usd_to:
        costs_usd_without_df = round_df(costs_usd_without_df, in_rounding_costs_usd_to)
        costs_usd_with_df = round_df(costs_usd_with_df, in_rounding_costs_usd_to)
        interv_costs_usd_without_df = round_df(interv_costs_usd_without_df, in_rounding_costs_usd_to)
        interv_costs_usd_with_df = round_df(interv_costs_usd_with_df, in_rounding_costs_usd_to)

    # %% Plot Consumables & Intervention Costs Over Time from the Table:
    if in_plot_costs:
        # all consumable costs by time periods
        all_cons_costs_without_tp =\
            in_costs_without_df.loc[(slice(None), 'co_modern_total'), 'Costs'].reset_index(level=1, drop=True).tolist()
        all_cons_costs_with_tp =\
            in_costs_with_df.loc[(slice(None), 'co_modern_total'), 'Costs'].reset_index(level=1, drop=True).tolist()
        # create lists with interv costs
        pop_interv_costs_with_tp_l = in_interv_costs_with_df['pop_intervention_cost'].tolist()
        ppfp_interv_costs_with_tp_l = in_interv_costs_with_df['ppfp_intervention_cost'].tolist()
        bar_chart_costs.plot_costs(
            [in_datestamp_without_log, in_datestamp_with_log], in_suffix, list(in_interv_costs_with_df.index),
            all_cons_costs_without_tp, all_cons_costs_with_tp, pop_interv_costs_with_tp_l, ppfp_interv_costs_with_tp_l,
            in_mwk_to_usd_exchange_rate  # & default in_reduce_magnitude=1e3
        )

    # %% Format cons & interv costs:
    in_costs_without_df = format_df(in_costs_without_df)
    in_costs_with_df = format_df(in_costs_with_df)
    in_interv_costs_without_df = format_df(in_interv_costs_without_df)
    in_interv_costs_with_df = format_df(in_interv_costs_with_df)
    costs_usd_without_df = format_df(costs_usd_without_df)
    costs_usd_with_df = format_df(costs_usd_with_df)
    interv_costs_usd_without_df = format_df(interv_costs_usd_without_df)
    interv_costs_usd_with_df = format_df(interv_costs_usd_with_df)

    # %% Join costs in MWK (and in USD):
    costs_mwk_usd_without_df = in_costs_without_df.astype(str) + " (" + costs_usd_without_df.astype(str) + ")"
    costs_mwk_usd_with_df = in_costs_with_df.astype(str) + " (" + costs_usd_with_df.astype(str) + ")"
    interv_costs_mwk_usd_without_df = \
        in_interv_costs_without_df.astype(str) + " (" + interv_costs_usd_without_df.astype(str) + ")"
    interv_costs_mwk_usd_with_df =\
        in_interv_costs_with_df.astype(str) + " (" + interv_costs_usd_with_df.astype(str) + ")"

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
                # consumable costs in MWK (in USD)
                l_tp_costs = []
                for meth in in_df_use.columns:
                    if meth in list(in_df_costs.loc[tp].index.get_level_values('Contraceptive_Method')):
                        # add costs for each meth & 'co_modern_total'
                        l_tp_costs.append(in_df_costs.loc[(tp, meth), 'Costs'])
                    else:
                        l_tp_costs.append('0 (0)')
                # intervention implementation costs & all costs in MWK (in USD)
                if in_df_interv_costs.empty:
                    for i in range(3):
                        l_tp_costs.append('0 (0)')
                    l_tp_costs.append(in_df_costs.loc[(tp, 'co_modern_total'), 'Costs'])
                else:
                    l_tp_costs.append(in_df_interv_costs.loc[tp, 'pop_intervention_cost'])
                    l_tp_costs.append(in_df_interv_costs.loc[tp, 'ppfp_intervention_cost'])
                    l_tp_costs.append(in_df_interv_costs.loc[tp, 'interventions_total'])
                    l_tp_costs.append(in_df_interv_costs.loc[tp, 'interv_cons_total'])
                data.append(l_tp_costs)
        table_cols = list(in_df_use_without.columns)
        table_cols.append('pop_interv')
        table_cols.append('ppfp_interv')
        table_cols.append('pop_ppfp_interv')
        table_cols.append('co_modern_all_interv_total')

        def rounding_name(in_rounding_scale):
            if in_rounding_scale:
                if in_rounding_scale == 1e9:
                    return "billions"
                elif in_rounding_scale == 1e6:
                    return "millions"
                elif in_rounding_scale == 1e5:
                    return "hundreds of thousands"
                elif in_rounding_scale == 1e4:
                    return "tens of thousands"
                elif in_rounding_scale == 1e3:
                    return "thousands"
                elif in_rounding_scale == 100:
                    return "hundreds"
                elif in_rounding_scale == 10:
                    return "tens"
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
                                       [str(in_use_output).capitalize() + ' % of\nwomen using\n(' +
                                        rounding_name(in_rounding_use_to) + ' users)',
                                        'Costs in\n' + rounding_name(in_rounding_costs_mwk_to) + ' MWK\n'
                                        + "(" + rounding_name(in_rounding_costs_mwk_to / 1000) + ' USD)']
                                   ]))
        return df_combined.loc[:, :].transpose()

    use_costs_table_df = combine_use_costs_with_without_interv(
        in_use_without_df, use_without_perc_val_df, costs_mwk_usd_without_df, interv_costs_mwk_usd_without_df,
        in_use_with_df, use_with_perc_val_df, costs_mwk_usd_with_df, interv_costs_mwk_usd_with_df
    )

    # Change the names of totals
    use_costs_table_df = use_costs_table_df.rename(
        index={'co_modern_total': 'modern contraceptives\nTOTAL',
               'pop_interv': 'Pop implementation',
               'ppfp_interv': 'PPFP implementation',
               'pop_ppfp_interv': 'Pop & PPFP implementation',
               'co_modern_all_interv_total': 'modern contraceptives &\ninterventions implementation\nTOTAL'
               }
    )
    # Remove the underscores from the names of contraception methods
    use_costs_table_df.index = use_costs_table_df.index.map(lambda s: s.replace("_", " "))

    output_table_file = r"outputs/output_table_" + in_use_output + "-use_costs" + "__" +\
                        in_ID_without + "_" + in_ID_with + in_suffix + ".xlsx"
    writer = pd.ExcelWriter(output_table_file)
    use_costs_table_df.to_excel(writer, index_label=use_costs_table_df.columns.name)

    writer.save()


# %% TABLE CONSUMABLES .................................................................................................
def table_cons(in_mwk_to_usd_exchange_rate,
               in_contraceptives_order: list = ['pill', 'IUD', 'injections', 'implant', 'male_condom',
                                                'female_sterilization', 'other_modern'],):

    resource_items_pkgs_df = pd.read_csv(
        'resources/healthsystem/consumables/ResourceFile_Consumables_Items_and_Packages.csv'
    )

    last_line_interv_pkg_name = 'Female Condom'
    assert np.count_nonzero(resource_items_pkgs_df['Intervention_Pkg'] == last_line_interv_pkg_name) == 1
    # TODO: This works only if there is only one item for the last pkg. Needs to be improved to work with a pkg with
    #  more items.
    last_line_nmb_data =\
        resource_items_pkgs_df.loc[resource_items_pkgs_df['Intervention_Pkg'] == last_line_interv_pkg_name].index[0]
    co_pkgs_df = resource_items_pkgs_df.loc[0:last_line_nmb_data,
                                            ['Intervention_Pkg', 'Items', 'Expected_Units_Per_Case', 'Unit_Cost']]
    # Remove Male sterilization if not among contraceptives we consider
    if 'male_sterilization' not in in_contraceptives_order:
        co_pkgs_df = co_pkgs_df.loc[co_pkgs_df['Intervention_Pkg'] != 'Male sterilization', :]
    # Rename the columns
    co_pkgs_df.columns =\
        ['\nContraception\npackage', '\n\nItem', 'Expected\nUnits\nPer Case', 'Unit Cost\n2021 price\nin MWK']

    def mwk_to_usd(in_mwk_price):
        return in_mwk_price * in_mwk_to_usd_exchange_rate

    # Calculate the costs in USD and round to 2 decimals
    co_pkgs_df['Unit Cost\n2021 price\nin USD'] =\
        round(co_pkgs_df['Unit Cost\n2021 price\nin MWK'].apply(mwk_to_usd), 2)
    # Round costs in MWK to 0 decimals
    co_pkgs_df['Unit Cost\n2021 price\nin MWK'] = round(co_pkgs_df['Unit Cost\n2021 price\nin MWK'])

    # Rename the pkgs to be consistent with methods names elsewhere and order the dataframe by contraceptives_order
    def pkg_name_to_method(in_co_pkg_name):
        """
        Returns contraception method name based on input Intervention_Pkg name from the RF.

        :param in_co_pkg_name: Intervention_Pkg name used in a ResourceFile.

        :return: Name of the contraception method.
        """
        if in_co_pkg_name == 'Pill':
            return 'pill'
        if in_co_pkg_name == 'Male condom':
            return 'male_condom'
        if in_co_pkg_name == 'Female Condom':
            return 'other_modern'
        if in_co_pkg_name == 'IUD':
            return 'IUD'
        if in_co_pkg_name == 'Injectable':
            return 'injections'
        if in_co_pkg_name == 'Implant':
            return 'implant'
        if in_co_pkg_name == 'Female sterilization':
            return 'female_sterilization'
        if in_co_pkg_name == 'Contraception initiation':
            return 'contraception initiation'
        else:
            raise ValueError(
                "There is an unrecognised co. Intervention_Pkg name: " + str(in_co_pkg_name) + "."
            )

    contraception_pkgs_order = ['contraception initiation'] + in_contraceptives_order
    co_pkgs_df['\nContraception\npackage'] = co_pkgs_df['\nContraception\npackage'].apply(pkg_name_to_method)
    co_pkgs_df = co_pkgs_df.sort_values(by='\nContraception\npackage',
                                        key=lambda x: x.map(lambda v: contraception_pkgs_order.index(v)))

    output_table_file = r"outputs/output_table_cons" "__" + str(date.today()) + ".xlsx"
    writer = pd.ExcelWriter(output_table_file)
    co_pkgs_df.to_excel(writer, index=False)

    writer.save()
    # TODO: finish? (the boarders, header align left, make the expected units to be nmbs so they are aligned right,
    #  write down the unique contraception package just once)
