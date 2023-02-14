import os

import analysis_utility_functions
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import extract_results, get_scenario_outputs

plt.style.use('seaborn-darkgrid')

def run_maternal_newborn_health_analysis(scenario_file_dict, outputspath, sim_years,
                                         intervention_years, service_of_interest,
                                         show_all_results, scen_colours):
    """
    This function can be used to output primary and secondary outcomes from a dictionary of scenario files. The type
    of outputs is dependent on the 'intervention' of interest (i.e. ANC/SBA/PNC) and can be amended accordingly.
    :param scenario_file_dict: dict containing names of python scripts for each scenario of interest
    :param outputspath: directory for graphs to be saved
    :param intervention_years: years of interest for the analysis
    :param service_of_interest: ANC/SBA/PNC
    :param show_all_results: bool - whether to output all results
    """

    # Create dictionary containing the results folder for each scenario
    results_folders = {k: get_scenario_outputs(scenario_file_dict[k], outputspath)[-1] for k in scenario_file_dict}
    output_df = pd.DataFrame(columns=list(scenario_file_dict.keys()))

    # Create folder to store graphs (if it hasnt already been created when ran previously)
    path = f'{outputspath}/{service_of_interest}_analysis_output_graphs_{results_folders["Status Quo"].name}'
    if not os.path.isdir(path):
        os.makedirs(f'{outputspath}/{service_of_interest}_analysis_output_graphs_{results_folders["Status Quo"].name}')

    # Save the file path
    plot_destination_folder = path

    #  BIRTHs PER SCENARIO...
    # Access birth data for each scenario (used as a denominator in some parts of the script)
    births_dict = analysis_utility_functions.return_birth_data_from_multiple_scenarios(results_folders,
                                                                                       sim_years, intervention_years)

    #  CHECKING INTERVENTION COVERAGE IS AS EXPECTED...
    # Before outputting the results for a given set of scenarios we check the intervention coverage for the core
    # interventions
    if service_of_interest == 'anc' or show_all_results:
        def get_anc_coverage(folder, service_structure):
            """ Returns the mean, lower quantile, upper quantile proportion of women who gave birth per year who
            received
             4/8 ANC visits during their pregnancy by scenario
            :param folder: results folder for scenario
            :param service_structure: 4/8
            :return: mean, lower quant, upper quant of coverage
            """

            # Get DF with ANC counts of all women who have delivered
            anc_coverage = extract_results(
                folder,
                module="tlo.methods.care_of_women_during_pregnancy",
                key="anc_count_on_birth",
                custom_generate_series=(
                    lambda df: df.assign(
                        year=df['date'].dt.year).groupby(['year'])['person_id'].count()),
                do_scaling=True
            )

            # Next get a version of that DF with women who attended >= 4/8 visits by birth
            an = extract_results(
                folder,
                module="tlo.methods.care_of_women_during_pregnancy",
                key="anc_count_on_birth",
                custom_generate_series=(
                    lambda df: df.loc[df['total_anc'] >= service_structure].assign(
                        year=df['date'].dt.year).groupby(['year'])['person_id'].count()),
                do_scaling=True
            )
            anc_cov_of_interest = an.fillna(0)

            cd = (anc_cov_of_interest/anc_coverage) * 100
            coverage_df = cd.fillna(0)

            mean_coverage_across_runs = analysis_utility_functions.get_mean_and_quants(coverage_df, sim_years)

            return mean_coverage_across_runs

        cov_data_4 = {k: get_anc_coverage(results_folders[k], 4) for k in results_folders}
        cov_data_8 = {k: get_anc_coverage(results_folders[k], 8) for k in results_folders}

        # output graphs
        for service_structure, cov_data in zip([4, 8], [cov_data_4, cov_data_8]):
            analysis_utility_functions.comparison_graph_multiple_scenarios(
                scen_colours, sim_years, cov_data, '% Births',
                f'Proportion of women receiving {service_structure} (or more) ANC visits at birth',
                plot_destination_folder, f'anc{service_structure}_cov')

    if service_of_interest == 'sba' or show_all_results:
        def get_delivery_place_info(folder, sim_years):
            all_deliveries = extract_results(
                folder,
                module="tlo.methods.labour",
                key="delivery_setting_and_mode",
                custom_generate_series=(
                    lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year'])[
                        'mother'].count()),
                do_scaling=True
            )

            results = dict()

            for facility_type in ['home_birth', 'hospital', 'health_centre', 'facility']:

                if facility_type == 'facility':
                    deliver_setting_results = extract_results(
                        folder,
                        module="tlo.methods.labour",
                        key="delivery_setting_and_mode",
                        custom_generate_series=(
                            lambda df: df.loc[df['facility_type'] != 'home_birth'].assign(
                                year=df['date'].dt.year).groupby(['year'])[
                                'mother'].count()),
                        do_scaling=True
                    )

                else:
                    deliver_setting_results = extract_results(
                        folder,
                        module="tlo.methods.labour",
                        key="delivery_setting_and_mode",
                        custom_generate_series=(
                            lambda df: df.loc[df['facility_type'] == facility_type].assign(
                                year=df['date'].dt.year).groupby(['year'])[
                                'mother'].count()),
                        do_scaling=True
                    )
                rate_df = (deliver_setting_results/all_deliveries) * 100
                results.update({facility_type: analysis_utility_functions.get_mean_and_quants(rate_df, sim_years)})

            return results

        delivery_data = {k: get_delivery_place_info(results_folders[k], sim_years) for k in results_folders}

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, delivery_data, 'facility',
            '% Total Births',
            'Facility Delivery Rate per Year Per Scenario',
            plot_destination_folder, 'fd_rate')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, delivery_data, 'home_birth',
            '% Total Births',
            'Home birth Rate per Year Per Scenario',
            plot_destination_folder, 'hb_rate')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, delivery_data, 'hospital',
            '% Total Births',
            'Hospital birth Rate per Year Per Scenario',
            plot_destination_folder, 'hp_rate')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, delivery_data, 'health_centre',
            '% Total Births',
            'Health Centre Birth Rate per Year Per Scenario',
            plot_destination_folder, 'hc_rate')

    if service_of_interest == 'pnc' or show_all_results:
        def get_pnc_coverage(folder, births_df):
            """
            Returns the mean, lower quantile, upper quantile proportion of women and neonates who received at least 1
            postnatal care visit after birth
            :param folder: results folder for scenario
            :param birth_data: dictionary containing mean/quantiles of births per year
            :return: mean, lower quantile, upper quantil of coverage
            """

            all_surviving_mothers = extract_results(
                folder,
                module="tlo.methods.postnatal_supervisor",
                key="total_mat_pnc_visits",
                custom_generate_series=(
                    lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])[
                        'mother'].count()),
                do_scaling=True
            )

            # Extract data on all women with 1+ PNC visits
            pnc_results_maternal = extract_results(
                folder,
                module="tlo.methods.postnatal_supervisor",
                key="total_mat_pnc_visits",
                custom_generate_series=(
                    lambda df: df.loc[df['visits'] > 0].assign(year=df['date'].dt.year).groupby(['year'])[
                        'mother'].count()),
                do_scaling=True
            )

            # Followed by newborns...
            all_surviving_newborns = extract_results(
                folder,
                module="tlo.methods.postnatal_supervisor",
                key="total_neo_pnc_visits",
                custom_generate_series=(
                    lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])[
                        'child'].count()),
                do_scaling=True
            )

            pnc_results_newborn = extract_results(
                folder,
                module="tlo.methods.postnatal_supervisor",
                key="total_neo_pnc_visits",
                custom_generate_series=(
                    lambda df: df.loc[df['visits'] > 0].assign(year=df['date'].dt.year).groupby(['year'])[
                        'child'].count()),
                do_scaling=True
            )

            cov_mat_birth_df = (pnc_results_maternal / births_df) * 100
            cov_mat_surv_df = (pnc_results_maternal / all_surviving_mothers) * 100
            cov_neo_birth_df = (pnc_results_newborn / births_df) * 100
            cov_neo_surv_df = (pnc_results_newborn / all_surviving_newborns) * 100

            return {'maternal_pnc_births': analysis_utility_functions.get_mean_and_quants(cov_mat_birth_df,
                                                                                          sim_years),
                    'maternal_pnc_survivors': analysis_utility_functions.get_mean_and_quants(cov_mat_surv_df,
                                                                                             sim_years),
                    'neonatal_pnc_births': analysis_utility_functions.get_mean_and_quants(cov_neo_birth_df,
                                                                                          sim_years),
                    'neonatal_pnc_survivors': analysis_utility_functions.get_mean_and_quants(cov_neo_surv_df,
                                                                                             sim_years)}

        coverage_data = {k: get_pnc_coverage(results_folders[k], births_dict[k]['births_data_frame']) for k in
                         results_folders}
        output_df = output_df.append(pd.DataFrame.from_dict(coverage_data))

        # generate plots showing coverage of ANC intervention in the baseline and intervention scenarios
        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, coverage_data, 'maternal_pnc_births',
            '% Total Births',
            'Maternal PNC Coverage as Proportion of Total Births',
            plot_destination_folder, 'mat_pnc_coverage_births')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, coverage_data, 'maternal_pnc_survivors',
            '% Total Survivors at Day 42',
            'Maternal PNC Coverage as Proportion of Postnatal Survivors',
            plot_destination_folder, 'mat_pnc_coverage_survivors')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, coverage_data, 'neonatal_pnc_births',
            '% Total Births',
            'Neonatal PNC Coverage as Proportion of Total Births',
            plot_destination_folder, 'neo_pnc_coverage_births')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, coverage_data, 'neonatal_pnc_survivors',
            '% Total Survivors at Day 28',
            'Neonatal PNC Coverage as Proportion of Neonatal Survivors',
            plot_destination_folder, 'neo_pnc_coverage_survivors')

    # --------------------------------------------PRIMARY OUTCOMES ----------------------------------------------------
    # ===================================== MATERNAL/NEONATAL DEATHS ==================================================
    # 1.) AGGREGATE DEATHS BY SCENARIO
    death_data = analysis_utility_functions.return_death_data_from_multiple_scenarios(results_folders,
                                                                                      births_dict,
                                                                                      sim_years,
                                                                                      intervention_years)
    output_df = output_df.append(pd.DataFrame.from_dict(death_data))

    for data, title, y_lable in \
        zip(['agg_dir_m_deaths',
             'agg_dir_mr',
             'agg_ind_m_deaths',
             'agg_ind_mr',
             'agg_total',
             'agg_total_mr',
             'agg_n_deaths',
             'agg_nmr'],
            ['Total Direct Maternal Deaths By Scenario',
             'Average Direct MMR by Scenario',
             'Total Indirect Maternal Deaths By Scenario',
             'Average Indirect MMR by Scenario',
             'Total Maternal Deaths By Scenario',
             'Average MMR by Scenario',
             'Total Neonatal Deaths By Scenario',
             'Average NMR by Scenario'],
            ['Total Direct Maternal Deaths',
             'Average MMR',
             'Total Indirect Maternal Deaths',
             'Average MMR',
             'Total Maternal Deaths',
             'Average MMR',
             'Total Neonatal Deaths',
             'Average NMR']):

        labels = results_folders.keys()

        mean_vals = list()
        lq_vals = list()
        uq_vals = list()
        for k in death_data:
            mean_vals.append(death_data[k][data][0])
            lq_vals.append(death_data[k][data][1])
            uq_vals.append(death_data[k][data][2])

        width = 0.55  # the width of the bars: can also be len(x) sequence
        fig, ax = plt.subplots()

        ci = [(x - y) / 2 for x, y in zip(uq_vals, lq_vals)]
        ax.bar(labels, mean_vals, color=scen_colours, width=width, yerr=ci)
        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.set_ylabel(y_lable)
        ax.set_xlabel('Scenario')
        ax.set_title(title)
        plt.savefig(f'{plot_destination_folder}/{data}.png')
        plt.show()

    # 2.) TRENDS IN DEATHS
    # Output and save the relevant graphs
    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, death_data, 'direct_mmr',
        'Deaths per 100,000 live births',
        'MMR per Year at Baseline and Under Intervention (Direct only)', plot_destination_folder,
        'maternal_mr_direct')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours,sim_years, death_data, 'total_mmr',
        'Deaths per 100,000 live births',
        'MMR per Year at Baseline and Under Intervention (Total)',
        plot_destination_folder, 'maternal_mr_total')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, death_data, 'nmr',
        'Total Deaths per 1000 live births',
        'Neonatal Mortality Ratio per Year at Baseline and Under Intervention',
        plot_destination_folder, 'neonatal_mr_int')

    for group, l in zip(['Maternal', 'Neonatal'], ['dir_m', 'n']):
        analysis_utility_functions.comparison_bar_chart_multiple_bars(
            death_data, f'crude_{l}_deaths', sim_years, scen_colours,
            f'Total {group} Deaths (scaled)', f'Yearly Baseline {group} Deaths Compared to Intervention',
            plot_destination_folder, f'{group}_crude_deaths_comparison.png')


    # 4.) DEATHS BY CAUSE
    # 4.a) Direct vs Indirect
    # can add uncertainty see - https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_stacked.html
    labels = results_folders.keys()
    dr_mr = list()
    ind_mr = list()
    for k in death_data:
        dr_mr.append(death_data[k]['agg_dir_mr'][0])
        ind_mr.append(death_data[k]['agg_ind_mr'][0])

    width = 0.35  # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots()

    ax.bar(labels, dr_mr, width,  label='Direct MMR', color='lightcoral')
    ax.bar(labels, ind_mr, width, bottom=dr_mr, label='Indirect MMR', color='firebrick')
    ax.tick_params(axis='x', which='major', labelsize=8)

    ax.set_ylabel('MMR')
    ax.set_xlabel('Scenario')
    ax.set_title('Aggregate Total MMR By Cause By Scenario')
    ax.legend()
    plt.savefig(f'{plot_destination_folder}/mmr_by_dir_v_ind.png')
    plt.show()

    # 4.b) Total deaths by cause (aggregate)
    def extract_deaths_by_cause(results_folder, births_df, intervention_years, scenario_name, final_df):

        dd = extract_results(
            results_folder,
            module="tlo.methods.demography.detail",
            key="properties_of_deceased_persons",
            custom_generate_series=(
                lambda df: df.loc[(df['date'].dt.year >= intervention_years[0]) &
                                  (df['date'].dt.year <= intervention_years[-1]) &
                                  df['cause_of_death'].str.contains(
                                      'ectopic_pregnancy|spontaneous_abortion|induced_abortion|'
                                      'severe_gestational_hypertension|severe_pre_eclampsia|eclampsia|antenatal_sepsis|'
                                      'uterine_rupture|intrapartum_sepsis|postpartum_sepsis|postpartum_haemorrhage|'
                                      'secondary_postpartum_haemorrhage|antepartum_haemorrhage')].assign(
                    year=df['date'].dt.year).groupby(['cause_of_death'])['year'].count()),
            do_scaling=True)
        direct_deaths = dd.fillna(0)

        id = extract_results(
            results_folder,
            module="tlo.methods.demography.detail",
            key="properties_of_deceased_persons",
            custom_generate_series=(
                lambda df: df.loc[(df['date'].dt.year >= intervention_years[0]) &
                                  (df['date'].dt.year <= intervention_years[-1]) &
                                  (df['is_pregnant'] | df['la_is_postpartum']) &
                                  df['cause_of_death'].str.contains(
                                      'AIDS_non_TB|AIDS_TB|TB|Malaria|Suicide|ever_stroke|diabetes|chronic_ischemic_hd|'
                                      'ever_heart_attack|chronic_kidney_disease')].assign(
                    year=df['date'].dt.year).groupby(['cause_of_death'])['year'].count()),
            do_scaling=True)

        indirect_deaths = id.fillna(0)

        total = direct_deaths.append(indirect_deaths)
        df_index = list(range(0, len(total.index)))
        df = final_df.reindex(df_index)

        births_int = births_df.loc[intervention_years[0]: intervention_years[-1]]
        sum_births_int_by_run = analysis_utility_functions.get_mean_from_columns(births_int, 'sum')
        births_agg_data = analysis_utility_functions.get_mean_and_quants_from_list(sum_births_int_by_run)

        for comp, v in zip(total.index, df_index):
            df.update(
                pd.DataFrame({'Scenario': scenario_name,
                              'Complication/Disease': comp,
                              'MMR': (total.loc[comp].mean()/births_agg_data[0]) * 100_000},
                             index=[v]))

        return df

    cause_df = pd.DataFrame(columns=['Scenario', 'Complication/Disease', 'MMR'])
    scenario_titles = list(results_folders.keys())
    t = []
    for k in scenario_titles:
        sq_df = extract_deaths_by_cause(results_folders[k], births_dict[k]['births_data_frame'],
                                        intervention_years, k,
                                        cause_df)
        t.append(sq_df)

    final_df = pd.concat(t)

    import seaborn as sns

    g = sns.catplot(kind='bar', data=final_df, col='Scenario', x='Complication/Disease', y='MMR')
    g.set_xticklabels(rotation=75, fontdict={'fontsize': 7}, horizontalalignment='right')
    plt.savefig(f'{plot_destination_folder}/deaths_by_cause_by_scenario.png', bbox_inches='tight')
    plt.show()

    final_df.to_csv(f'{plot_destination_folder}/mmrs_by_cause.csv')

    # NEONATAL DEATH BY CAUSE
    def extract_neo_deaths_by_cause(results_folder, births_df, intervention_years, scenario_name, final_df):

        nd = extract_results(
            results_folder,
            module="tlo.methods.demography.detail",
            key="properties_of_deceased_persons",
            custom_generate_series=(
                lambda df: df.loc[(df['date'].dt.year >= intervention_years[0]) &
                                  (df['date'].dt.year <= intervention_years[-1]) &
                                  (df['age_days'] < 29)].assign(
                    year=df['date'].dt.year).groupby(['cause_of_death'])['year'].count()),
            do_scaling=True)
        neo_deaths = nd.fillna(0)

        births_int = births_df.loc[intervention_years[0]: intervention_years[-1]]
        sum_births_int_by_run = analysis_utility_functions.get_mean_from_columns(births_int, 'sum')
        births_agg_data = analysis_utility_functions.get_mean_and_quants_from_list(sum_births_int_by_run)

        df_index = list(range(0, len(neo_deaths.index)))
        df = final_df.reindex(df_index)

        for comp, v in zip(neo_deaths.index, df_index):
            df.update(
                pd.DataFrame({'Scenario': scenario_name,
                              'Complication/Disease': comp,
                              'NMR': (neo_deaths.loc[comp].mean()/births_agg_data[0]) * 1000},
                             index=[v]))

        return df

    ncause_df = pd.DataFrame(columns=['Scenario', 'Complication/Disease', 'NMR'])
    t = []
    for k in scenario_titles:
        sq_df = extract_neo_deaths_by_cause(results_folders[k], births_dict[k]['births_data_frame'],
                                            intervention_years, k, ncause_df)
        t.append(sq_df)
    final_df_n = pd.concat(t)

    g = sns.catplot(kind='bar', data=final_df_n, col='Scenario', x='Complication/Disease', y='NMR')
    g.set_xticklabels(rotation=75, fontdict={'fontsize': 6}, horizontalalignment='right')
    plt.savefig(f'{plot_destination_folder}/neo_deaths_by_cause_by_scenario.png', bbox_inches='tight')
    plt.show()

    final_df_n.to_csv(f'{plot_destination_folder}/nmrs_by_cause.csv')

    # ===================================== MATERNAL/NEONATAL DALYS ===================================================
    # =================================================== DALYS =======================================================
    # Here we extract maternal and neonatal DALYs from each scenario to allow for comparison
    dalys_data = analysis_utility_functions.return_dalys_from_multiple_scenarios(results_folders, sim_years,
                                                                                 intervention_years)

    output_df = output_df.append(pd.DataFrame.from_dict(dalys_data))

    for data, title, y_lable in \
        zip(['agg_mat_dalys_stacked',
             'agg_neo_dalys_stacked',
             'avg_mat_dalys_rate_stacked',
             'avg_neo_dalys_rate_stacked'],
            ['Average Total Maternal DALYs (stacked) by Scenario',
             'Average Total Neonatal DALYs (stacked) by Scenario',
             'Average Total Maternal DALYs per 100k PY by Scenario',
             'Average Total Neonatal DALYs per 100k PY by Scenario'],
            ['DALYs',
             'DALYs',
             'DALYs per 100k PY',
             'DALYs per 100k PY']):
        labels = results_folders.keys()

        mean_vals = list()
        lq_vals = list()
        uq_vals = list()
        for k in dalys_data:
            mean_vals.append(dalys_data[k][data][0])
            lq_vals.append(dalys_data[k][data][1])
            uq_vals.append(dalys_data[k][data][2])

        width = 0.55  # the width of the bars: can also be len(x) sequence
        fig, ax = plt.subplots()

        ci = [(x - y) / 2 for x, y in zip(uq_vals, lq_vals)]
        ax.bar(labels, mean_vals, color=scen_colours, width=width, yerr=ci)
        ax.tick_params(axis='x', which='major', labelsize=8)

        ax.set_ylabel(y_lable)
        ax.set_xlabel('Scenario')
        ax.set_title(title)
        ax.legend()
        plt.savefig(f'{plot_destination_folder}/{data}.png')
        plt.show()

    for dict_key, axis, title, save_name in zip(['maternal_dalys_crude_stacked', 'maternal_dalys_rate_stacked',
                                                 'maternal_yll_crude_stacked', 'maternal_yll_rate_stacked',
                                                 'maternal_yld_crude_unstacked', 'maternal_yld_rate_unstacked',

                                                 'neonatal_dalys_crude_stacked', 'neonatal_dalys_rate_stacked',
                                                 'neonatal_yll_crude_stacked', 'neonatal_yll_rate_stacked',
                                                 'neonatal_yld_crude_unstacked', 'neonatal_yld_rate_unstacked'],

                                                ['DALYs', 'DALYs per 100k Person-Years', 'YLL',
                                                 'YLL per 100k Person-Years', 'YLD', 'YLD per 100k Person-Years',

                                                 'DALYs', 'DALYs per 100k Person-Years', 'YLL',
                                                 'YLL per 100k Person-Years', 'YLD', 'YLD per 100k Person-Years'],

                                                ['Crude Total DALYs per Year Attributable to Maternal disorders',
                                                 'DALYs per 100k Person-Years Attributable to Maternal disorders',
                                                 'Crude Total YLL per Year Attributable to Maternal disorders',
                                                 'YLL per 100k Person-Years Attributable to Maternal disorders',
                                                 'Crude Total YLD per Year Attributable to Maternal disorders',
                                                 'YLD per 100k Person-Years Attributable to Maternal disorders',

                                                 'Crude Total DALYs per Year Attributable to Neonatal disorders',
                                                 'DALYs per 100k Person-Years Attributable to Neonatal disorders',
                                                 'Crude Total YLL per Year Attributable to Neonatal disorders',
                                                 'YLL per 100k Person-Years Attributable to Neonatal disorders',
                                                 'Crude Total YLD per Year Attributable to Neonatal disorders',
                                                 'YLD per 100k Person-Years Attributable to Neonatal disorders'],

                                                ['maternal_dalys_stacked', 'maternal_dalys_rate',
                                                 'maternal_yll', 'maternal_yll_rate',
                                                 'maternal_yld', 'maternal_yld_rate',
                                                 'neonatal_dalys_stacked', 'neonatal_dalys_rate',
                                                 'neonatal_yll', 'neonatal_yll_rate',
                                                 'neonatal_yld', 'neonatal_yld_rate']):

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, dalys_data, dict_key, axis, title, plot_destination_folder, save_name)

    #  ================================== STILLBIRTH  ================================================================
    # For interventions that may impact rates of stillbirth we output that information here
    if (service_of_interest != 'pnc') or show_all_results:
        sbr_data = analysis_utility_functions.return_stillbirth_data_from_multiple_scenarios(
            results_folders, births_dict, sim_years, intervention_years)

        output_df = output_df.append(pd.DataFrame.from_dict(sbr_data))

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, sbr_data, 'an_sbr',
            'Antenatal stillbirths per 1000 births',
            'Antenatal stillbirth Rate per Year at Baseline and Under Intervention',
            plot_destination_folder, 'an_sbr_int')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, sbr_data, 'ip_sbr',
            'Intrapartum stillbirths per 1000 births',
            'Intrapartum stillbirth Rate per Year at Baseline and Under Intervention',
            plot_destination_folder, 'ip_sbr_int')

        # Output SBR per year for scenario vs intervention
        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, sbr_data, 'sbr',
            'Stillbirths per 1000 births',
            'Stillbirth Rate per Year at Baseline and Under Intervention',
            plot_destination_folder, 'sbr_int')

        analysis_utility_functions.comparison_bar_chart_multiple_bars(
            sbr_data, 'crude_sb', sim_years, scen_colours,
            'Total Stillbirths (scaled)', 'Yearly Baseline Stillbirths Compared to Intervention',
            plot_destination_folder, 'crude_stillbirths_comparison.png')

        for data, title, y_lable in \
            zip(['avg_sbr',
                 'avg_i_sbr',
                 'avg_a_sbr'],
                ['Average Total Stillbirth Rate during the Intervention Period',
                 'Average Intrapartum Stillbirth Rate during the Intervention Period',
                 'Average Antenatal Stillbirth Rate during the Intervention Period'],
                ['Stillbirths per 1000 births',
                 'Stillbirths per 1000 births',
                 'Stillbirths per 1000 births']):

            labels = results_folders.keys()

            mean_vals = list()
            lq_vals = list()
            uq_vals = list()
            for k in sbr_data:
                mean_vals.append(sbr_data[k][data][0])
                lq_vals.append(sbr_data[k][data][1])
                uq_vals.append(sbr_data[k][data][2])

            width = 0.55  # the width of the bars: can also be len(x) sequence
            fig, ax = plt.subplots()

            ci = [(x - y) / 2 for x, y in zip(uq_vals, lq_vals)]
            ax.bar(labels, mean_vals, color=scen_colours, width=width, yerr=ci)
            ax.tick_params(axis='x', which='major', labelsize=8)

            ax.set_ylabel(y_lable)
            ax.set_xlabel('Scenario')
            ax.set_title(title)
            ax.legend()
            plt.savefig(f'{plot_destination_folder}/{data}.png')
            plt.show()

    # ------------------------------------------- SECONDARY OUTCOMES --------------------------------------------------
    # ========================================= HEALTH SYSTEM OUTCOMES ================================================

    def plot_agg_graph(data, key, y_label, title, save_name):
        labels = results_folders.keys()
        mean_vals = list()
        lq_vals = list()
        uq_vals = list()
        for k in data:
            mean_vals.append(data[k][key][0])
            lq_vals.append(data[k][key][1])
            uq_vals.append(data[k][key][2])

        width = 0.55  # the width of the bars: can also be len(x) sequence
        fig, ax = plt.subplots()
        ax.tick_params(axis='x', which='major', labelsize=8)

        ci = [(x - y) / 2 for x, y in zip(uq_vals, lq_vals)]
        ax.bar(labels, mean_vals, color=scen_colours, width=width, yerr=ci)
        ax.set_ylabel(y_label)
        ax.set_xlabel('Scenario')
        ax.set_title(title)
        plt.savefig(f'{plot_destination_folder}/{save_name}.png')
        plt.show()

    # Next we output certain health system outcomes such as number of HSIs by scenario
    # 1.) Additional HSI numbers
    if service_of_interest == 'anc' or show_all_results:
        def get_hsi_counts_from_cowdp_logger(folder, sim_years, intervention_years):

            def get_counts_of_hsi_by_treatment_id(_df):
                new_d = _df.assign(year=_df['date'].dt.year).drop(['date'], axis=1).set_index(['year'])
                new_d['total'] = new_d[list(new_d.columns)].sum(axis=1)
                return new_d['total']

            hsi = extract_results(
                folder,
                module="tlo.methods.care_of_women_during_pregnancy",
                key="anc_visits_which_ran",
                custom_generate_series=get_counts_of_hsi_by_treatment_id,
                do_scaling=True)

            hsi_data = analysis_utility_functions.get_mean_and_quants(hsi, sim_years)
            hsi_data_int = analysis_utility_functions.get_mean_and_quants(
                hsi.loc[intervention_years[0]:intervention_years[-1]], intervention_years)

            agg_hsi_data = analysis_utility_functions.get_mean_from_columns(hsi_data_int, 'sum')

            agg = [np.mean(agg_hsi_data),
                   np.quantile(agg_hsi_data, 0.025),
                   np.quantile(agg_hsi_data, 0.975)]

            return {'anc_contacts_trend': hsi_data,
                    'agg_anc_contacts': agg}

        hs_data = {k: get_hsi_counts_from_cowdp_logger(results_folders[k], sim_years, intervention_years) for k in
                   results_folders}

        output_df = output_df.append(pd.DataFrame.from_dict(hs_data))

        # todo: Better as a rate?
        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, hs_data, 'anc_contacts_trend',
            'Number of Visits',
            'Total Number of Antenatal Care Visits per Year Per Scenario',
            plot_destination_folder, 'anc_visits_crude_rate')

        plot_agg_graph(hs_data, 'agg_anc_contacts', 'Total ANC contacts', 'Total Number of ANC visits per Scenario',
                       'agg_anc_contacts')

    # TODO: FIX
    if service_of_interest == 'pnc' or show_all_results:
        def get_hsi_counts_from_summary_logger(folder, sim_years, intervention_years):

            # TODO: this is hacky - have to check that there actually are no visits
            # if 'min_pnc' in folder.name:
            #     empty = [[0 for y in intervention_years],
            #              [0 for y in intervention_years],
            #              [0 for y in intervention_years]]
            #     hsi_data = empty
            #     mat_agg = [0, 0, 0]
            #     hsi_data_neo = empty
            #     neo_agg = [0, 0,0]
            # else:
            hsi = extract_results(
                    folder,
                    module="tlo.methods.healthsystem.summary",
                    key="HSI_Event",
                    custom_generate_series=(
                        lambda df: pd.concat([df, df['TREATMENT_ID'].apply(pd.Series)], axis=1).assign(
                            year=df['date'].dt.year).groupby(['year'])['PostnatalCare_Maternal'].sum()),
                    do_scaling=True)

            hsi_n = extract_results(
                    folder,
                    module="tlo.methods.healthsystem.summary",
                    key="HSI_Event",
                    custom_generate_series=(
                        lambda df: pd.concat([df, df['TREATMENT_ID'].apply(pd.Series)], axis=1).assign(
                            year=df['date'].dt.year).groupby(['year'])['PostnatalCare_Neonatal'].sum()),
                    do_scaling=True)

            if 'min_pnc' in folder.name:
                pre_int_years = list(range(2010, intervention_years[0]))
                hsi_data = analysis_utility_functions.get_mean_and_quants(hsi, pre_int_years)
                hsi_data_neo = analysis_utility_functions.get_mean_and_quants(hsi_n, pre_int_years)

                for data in [hsi_data, hsi_data_neo]:
                    for l in data:
                        for y in intervention_years:
                            l.append(0)
                mat_agg = [0, 0, 0]
                neo_agg = [0, 0, 0]

            else:

                hsi_data = analysis_utility_functions.get_mean_and_quants(hsi, sim_years)
                hsi_data_int = analysis_utility_functions.get_mean_and_quants(
                        hsi.loc[intervention_years[0]:intervention_years[-1]], intervention_years)

                mat_agg_data = analysis_utility_functions.get_mean_from_columns(hsi_data_int, 'sum')

                mat_agg = [np.mean(mat_agg_data),
                           np.quantile(mat_agg_data, 0.025),
                           np.quantile(mat_agg_data, 0.975)]

                hsi_data_neo = analysis_utility_functions.get_mean_and_quants(hsi_n, sim_years)
                hsi_data_neo_int = analysis_utility_functions.get_mean_and_quants(
                        hsi_n.loc[intervention_years[0]:intervention_years[-1]], intervention_years)

                neo_agg_data = analysis_utility_functions.get_mean_from_columns(hsi_data_neo_int, 'sum')

                neo_agg = [np.mean(neo_agg_data),
                           np.quantile(neo_agg_data, 0.025),
                           np.quantile(neo_agg_data, 0.975)]

            return {'pnc_visits_mat_trend': hsi_data,
                    'pnc_visits_mat_agg': mat_agg,
                    'pnc_visits_neo_trend': hsi_data_neo,
                    'pnc_visits_neo_agg': neo_agg}

        hs_data = {k: get_hsi_counts_from_summary_logger(results_folders[k], sim_years, intervention_years) for k in
                   results_folders}

        output_df = output_df.append(pd.DataFrame.from_dict(hs_data))

        # todo: Better as a rate?
        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, hs_data, 'pnc_visits_mat_trend',
            'Crude Number',
            'Total Number of Maternal Postnatal Care Visits per Year Per Scenario',
            plot_destination_folder, 'pnc_mat_visits')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, hs_data, 'pnc_visits_neo_trend',
            'Crude Number',
            'Total Number of Neonatal Postnatal Care Visits per Year Per Scenario',
            plot_destination_folder, 'pnc_neo_visits')

        for group, title in zip(['mat', 'neo'], ['Maternal', 'Neonatal']):
            plot_agg_graph(hs_data, f'pnc_visits_{group}_agg', 'Total PNC Visist',
                           f'Total Number of {title} PNC visits per Scenario',
                           f'agg_{group}_pnc_visits')

    # ------------------------------------------------ MALARIA ------------------------------------------------------
    # Output malaria total incidence and clinical cases  during pregnancy

    if service_of_interest == 'anc' or show_all_results:
        def get_malaria_incidence_in_pregnancy(folder):
            preg_clin_counter_dates = extract_results(
                folder,
                module="tlo.methods.malaria",
                key="incidence",
                column='clinical_preg_counter',
                index='date',
                do_scaling=True
            )

            years = preg_clin_counter_dates.index.year
            preg_clin_counter_years = preg_clin_counter_dates.set_index(years)
            preg_clinical_counter = analysis_utility_functions.get_mean_and_quants(preg_clin_counter_years,
                                                                                   sim_years)
            preg_clinical_counter_int = analysis_utility_functions.get_mean_and_quants(
                preg_clin_counter_years.loc[intervention_years[0]:intervention_years[-1]], intervention_years)

            preg_clin_counter_agg_data = analysis_utility_functions.get_mean_from_columns(preg_clinical_counter_int,
                                                                                          'sum')

            preg_clin_counter_agg = [np.mean(preg_clin_counter_agg_data),
                                     np.quantile(preg_clin_counter_agg_data, 0.025),
                                     np.quantile(preg_clin_counter_agg_data, 0.975)]

            incidence_dates = extract_results(
                folder,
                module="tlo.methods.malaria",
                key="incidence",
                column='inc_1000py',
                index='date',
                do_scaling=True
            )

            years = incidence_dates.index.year
            incidence_years = incidence_dates.set_index(years)
            incidence = analysis_utility_functions.get_mean_and_quants(incidence_years, sim_years)

            return {'mal_clin_counter': preg_clinical_counter,
                    'mal_clin_counter_agg': preg_clin_counter_agg,
                    'mal_incidence': incidence}

        mal_data = {k: get_malaria_incidence_in_pregnancy(results_folders[k]) for k in results_folders}

        output_df = output_df.append(pd.DataFrame.from_dict(mal_data))

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, mal_data, 'mal_clin_counter',
            'Num. Clinical Cases',
            'Number of Clinical Cases of Malaria During Pregnancy Per Year Per Scenario',
            plot_destination_folder, 'mal_clinical_cases')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, mal_data, 'mal_incidence',
            'Incidence per 1000 person years',
            'Incidence of Malaria Per Year Per Scenario',
            plot_destination_folder, 'mal_incidence')

        plot_agg_graph(mal_data, 'mal_clin_counter_agg', 'Number of Clinical Cases',
                       'Total Clinical Cases of Malaria During Pregnancy Per Scenario', 'mal_agg_clin_cases')

        # ------------------------------------------------ TB ------------------------------------------------------
        # Output total new Tb diagnoses per year (all people) and then the coverage of treatment
        def get_tb_info_in_pregnancy(folder):
            # New Tb diagnoses per year
            tb_new_diag_dates = extract_results(
                folder,
                module="tlo.methods.tb",
                key="tb_treatment",
                column='tbNewDiagnosis',
                index='date',
                do_scaling=True
            )

            years = tb_new_diag_dates.index.year
            tb_new_diag_years = tb_new_diag_dates.set_index(years)
            tb_diagnosis = analysis_utility_functions.get_mean_and_quants(tb_new_diag_years,
                                                                          sim_years)

            tb_diagnosis_int = analysis_utility_functions.get_mean_and_quants(
                tb_new_diag_years.loc[intervention_years[0]: intervention_years[-1]], intervention_years)

            tb_diagnosis_agg_data = analysis_utility_functions.get_mean_from_columns(tb_diagnosis_int, 'sum')

            tb_diagnosis_agg = [np.mean(tb_diagnosis_agg_data),
                                np.quantile(tb_diagnosis_agg_data, 0.025),
                                np.quantile(tb_diagnosis_agg_data, 0.975)]

            # Treatment coverage
            tb_treatment_dates = extract_results(
                folder,
                module="tlo.methods.tb",
                key="tb_treatment",
                column='tbTreatmentCoverage',
                index='date',
            )

            years = tb_treatment_dates.index.year
            tb_treatment_years = tb_treatment_dates.set_index(years)
            tb_treatment = analysis_utility_functions.get_mean_and_quants(tb_treatment_years, sim_years)

            return {'tb_diagnosis': tb_diagnosis,
                    'tb_diagnosis_agg': tb_diagnosis_agg,
                    'tb_treatment': tb_treatment}

        tb_data = {k: get_tb_info_in_pregnancy(results_folders[k]) for k in results_folders}

        output_df = output_df.append(pd.DataFrame.from_dict(tb_data))

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, tb_data, 'tb_diagnosis',
            'Number of Tb Diagnoses',
            'Number of New Tb Diagnoses Per Year Per Scenario',
            plot_destination_folder, 'tb_diagnoses')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, tb_data, 'tb_treatment',
            '% New Tb Cases Treated',
            'Proportion of New Cases of Tb Treated Per Year Per Scenario',
            plot_destination_folder, 'tb_treatment')

        plot_agg_graph(tb_data, 'tb_diagnosis_agg', 'Number of Tb Diagnoses',
                       'Total New Tb Cases Diagnosed per Scenario', 'tb_diagnoses_agg')

    # ------------------------------------------------ HIV ------------------------------------------------------
    # Output the proportion of all women per year that are tested for HIV, the per-capita testing rate and the
    # number of women on ART per year

    if (service_of_interest != 'sba') or show_all_results:
        def get_hiv_information(folder):
            hiv_tests_dates = extract_results(
                folder,
                module="tlo.methods.hiv",
                key="hiv_program_coverage",
                column='prop_tested_adult_female',
                index='date',
            )

            years = hiv_tests_dates.index.year
            hiv_tests_years = hiv_tests_dates.set_index(years)
            hiv_tests = analysis_utility_functions.get_mean_and_quants(hiv_tests_years,sim_years)
            hiv_test_int = analysis_utility_functions.get_mean_and_quants(
                hiv_tests_years.loc[intervention_years[0]:intervention_years[-1]], intervention_years)

            avg_test_prop_data = analysis_utility_functions.get_mean_from_columns(hiv_test_int, 'avg')

            avg_test_prop = [np.mean(avg_test_prop_data),
                             np.quantile(avg_test_prop_data, 0.025),
                             np.quantile(avg_test_prop_data, 0.975)]

            # Per-capita testing rate
            hiv_tests_rate_dates = extract_results(
                folder,
                module="tlo.methods.hiv",
                key="hiv_program_coverage",
                column='per_capita_testing_rate',
                index='date',
            )

            years = hiv_tests_rate_dates.index.year
            hiv_tests_rate_years = hiv_tests_rate_dates.set_index(years)
            hiv_test_rate = analysis_utility_functions.get_mean_and_quants(hiv_tests_rate_years,
                                                                           sim_years)

            # Number of women on ART
            art_dates = extract_results(
                folder,
                module="tlo.methods.hiv",
                key="hiv_program_coverage",
                column='n_on_art_female_15plus',
                index='date',
                do_scaling=True
            )

            years = art_dates.index.year
            art_years = art_dates.set_index(years)
            art = analysis_utility_functions.get_mean_and_quants(art_years, sim_years)
            art_int = analysis_utility_functions.get_mean_and_quants(art_years.loc[intervention_years[0]:
                                                                                   intervention_years[-1]],
                                                                     intervention_years)
            art_agg_data = analysis_utility_functions.get_mean_from_columns(art_int, 'sum')

            art_agg = [np.mean(art_agg_data),
                       np.quantile(art_agg_data, 0.025),
                       np.quantile(art_agg_data, 0.975)]

            return {'hiv_testing_prop': hiv_tests,
                    'avg_test_prop': avg_test_prop,
                    'hiv_testing_rate': hiv_test_rate,
                    'art_number': art,
                    'art_number_agg': art_agg}

        hiv_data = {k: get_hiv_information(results_folders[k]) for k in results_folders}
        output_df = output_df.append(pd.DataFrame.from_dict(hiv_data))

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, hiv_data, 'hiv_testing_prop',
            '% Total Female Pop.',
            'Proportion of Female Population Who Received HIV test Per Year Per Scenario',
            plot_destination_folder, 'hiv_fem_testing_prop')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, hiv_data, 'hiv_testing_rate',
            'Per Captia Rate',
            'Rate of HIV testing per capita per year per scenario',
            plot_destination_folder, 'hiv_pop_testing_rate')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, hiv_data, 'art_number',
            'Women',
            'Number of Women Receiving ART per Year Per Scenario',
            plot_destination_folder, 'hiv_women_art')

        plot_agg_graph(hiv_data, 'art_number_agg', 'Number of Women',
                       'Total Number of Women Receiving ART per Per Scenario', 'hiv_women_art_agg')

        plot_agg_graph(hiv_data, 'avg_test_prop', '% Total Female Pop.',
                       'Average % of Total Female Population Tested for HIV during Intervention Period',
                       'avg_test_prop')

    # ------------------------------------------- Depression ---------------------------------------------------------
    # For depression we output diagnosis of ever depressed people, proportion of depressed people started on
    # antidepressants and proportion of depressed people started on talking therapy

    if service_of_interest != 'sba' or show_all_results:
        def get_depression_info_in_pregnancy(folder, sim_years):

            # Diagnosis of depression in ever depressed people
            diag_prop = extract_results(
                folder,
                module="tlo.methods.depression",
                key="summary_stats",
                column='p_ever_diagnosed_depression_if_ever_depressed',
                index='date',
            )
            diag_reindexed = diag_prop.set_index(diag_prop.index.year)
            diag_data = diag_reindexed.groupby(diag_reindexed.index).mean()
            diag_final = analysis_utility_functions.get_mean_and_quants(diag_data, sim_years)

            anti_depress = extract_results(
                folder,
                module="tlo.methods.depression",
                key="summary_stats",
                column='prop_antidepr_if_ever_depr',
                index='date',
            )
            anti_depress_reindexed = anti_depress.set_index(anti_depress.index.year)
            ad_data = anti_depress_reindexed.groupby(anti_depress_reindexed.index).mean()
            ad_final = analysis_utility_functions.get_mean_and_quants(ad_data, sim_years)

            tt = extract_results(
                folder,
                module="tlo.methods.depression",
                key="summary_stats",
                column='prop_ever_talk_ther_if_ever_depr',
                index='date',
            )

            tt_reindexed = tt.set_index(tt.index.year)
            tt_data = tt_reindexed.groupby(tt_reindexed.index).mean()
            tt_final = analysis_utility_functions.get_mean_and_quants(tt_data, sim_years)

            return {'dep_diag': diag_final,
                    'dep_anti_d': ad_final,
                    'dep_talking_t': tt_final}

        depression_data = {k: get_depression_info_in_pregnancy(results_folders[k], sim_years) for k in
                           results_folders}

        output_df = output_df.append(pd.DataFrame.from_dict(depression_data))

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, depression_data, 'dep_diag',
            'Proportion (%)',
            'Proportion of Ever Depressed Individuals Diagnosed with Depression',
            plot_destination_folder, 'depression_diag')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, depression_data, 'dep_anti_d',
            'Proportion (%)',
            'Proportion of Ever Depressed Individuals Started on Antidepressants',
            plot_destination_folder, 'depression_ad')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, depression_data, 'dep_talking_t',
            'Proportion (%)',
            'Proportion of Ever Depressed Individuals Started on Talking Therapy',
            plot_destination_folder, 'depression_tt')

    output_df.to_csv(f'{plot_destination_folder}/outputs.csv')
