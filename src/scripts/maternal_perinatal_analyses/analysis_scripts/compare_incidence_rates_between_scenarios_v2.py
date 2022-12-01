import os
import pandas as pd
import numpy as np

import analysis_utility_functions
from matplotlib import pyplot as plt

from tlo.analysis.utils import extract_results, get_scenario_outputs


def compare_key_rates_between_multiple_scenarios(scenario_file_dict, service_of_interest, outputspath, sim_years,
                                                 intervention_years, scen_colours):
    """
    This function will output and store plots of key outcomes (incidence, care seeking etc) across multiple
    scenarios. This can largely be used as a sense check or in some instances will contain secondary outcomes
    for some analyses.
    :param scenario_file_dict: dictionary containing file names for results folders for a selection of scenarios
    :param service_of_interest:  string variable (anc/sba/pnc)
    :param outputspath: directory for graphs to be saved
    :param intervention_years: years of interest for the analysis
    :return:
    """
    output_df = pd.DataFrame(columns=list(scenario_file_dict.keys()))

    # HELPER FUNCTIONS (TODO: MOVE/COMBINE WITH ONE USED IN OTHER FILE)
    def bar_chart_from_dict(dict, y_title, title,plot_destination_folder, file_name):
        labels = dict.keys()
        mean_vals = list()
        lq_vals = list()
        uq_vals = list()
        for k in dict:
            mean_vals.append(dict[k][0])
            lq_vals.append(dict[k][1])
            uq_vals.append(dict[k][2])

        width = 0.55  # the width of the bars: can also be len(x) sequence
        fig, ax = plt.subplots()

        ci = [(x - y) / 2 for x, y in zip(uq_vals, lq_vals)]
        ax.bar(labels, mean_vals, color=scen_colours, width=width, yerr=ci)
        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.set_ylabel(y_title)
        ax.set_xlabel('Scenario')
        ax.set_title(title)
        plt.savefig(f'{plot_destination_folder}/{file_name}.png')
        plt.show()

    def get_agg_values(yr_dict, key):
        agg_dict = dict()

        for k in yr_dict:
            agg_dict.update({k: [sum(yr_dict[k][key][0]), sum(yr_dict[k][key][1]), sum(yr_dict[k][key][2])]})

        return agg_dict

    def get_avg_rate_per_scenario(rate_dict, multi_level,  *key):
        avg_dict = dict()
        for k in rate_dict:
            if not multi_level:
                avg_dict.update({k: [sum(rate_dict[k][0]) / len(rate_dict[k][0]),
                                     sum(rate_dict[k][1]) / len(rate_dict[k][0]),
                                     sum(rate_dict[k][2]) / len(rate_dict[k][0])]})
            else:
                avg_dict.update({k: [sum(rate_dict[k][key[0]][0]) / len(rate_dict[k][key[0]][0]),
                                     sum(rate_dict[k][key[0]][1]) / len(rate_dict[k][key[0]][0]),
                                     sum(rate_dict[k][key[0]][2]) / len(rate_dict[k][key[0]][0])]})
        return avg_dict

    # Find results folder (most recent run generated using that scenario_filename)
    results_folders = {k: get_scenario_outputs(scenario_file_dict[k], outputspath)[-1] for k in scenario_file_dict}

    path = f'{outputspath}/analysis_comparing_incidence_{results_folders["Status Quo"].name}_{service_of_interest}'
    if not os.path.isdir(path):
        os.makedirs(
            f'{outputspath}/analysis_comparing_incidence_{results_folders["Status Quo"].name}_{service_of_interest}')

    plot_destination_folder = path

    # GET COMPLICATION DATAFRAMES...
    comp_dfs = {k: analysis_utility_functions.get_modules_maternal_complication_dataframes(results_folders[k])
                for k in results_folders}

    def get_neonatal_comp_dfs(results_folder):
        nb_comp_dfs = dict()
        nb_df = extract_results(
                results_folder,
                module="tlo.methods.newborn_outcomes",
                key="newborn_complication",
                custom_generate_series=(
                    lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['newborn'].count()),
                do_scaling=True
            )
        nb_comp_dfs['newborn_outcomes'] = nb_df.fillna(0)

        nb_pn_df = extract_results(
                results_folder,
                module="tlo.methods.postnatal_supervisor",
                key="newborn_complication",
                custom_generate_series=(
                    lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['newborn'].count()),
                do_scaling=True
            )
        nb_comp_dfs['newborn_postnatal'] = nb_pn_df.fillna(0)

        return nb_comp_dfs

    neo_comp_dfs = {k: get_neonatal_comp_dfs(results_folders[k]) for k in results_folders}

    # ------------------------------------ PREGNANCIES AND BIRTHS ----------------------------------------------------
    # Extract and plot pregnancies and births across scenarios
    preg_dict = analysis_utility_functions.return_pregnancy_data_from_multiple_scenarios(results_folders,
                                                                                         sim_years, intervention_years)
    agg_pregs = get_agg_values(preg_dict, 'int_preg')
    bar_chart_from_dict(agg_pregs, 'Pregnancies', 'Total Pregnancies by Scenario', plot_destination_folder, 'agg_preg')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, preg_dict, 'total_preg',
        'Total Pregnancies',
        'Total Number of Pregnancies Per Year By Scenario',
        plot_destination_folder, 'preg')

    births_dict = analysis_utility_functions.return_birth_data_from_multiple_scenarios(results_folders,
                                                                                       sim_years, intervention_years)
    agg_births = get_agg_values(births_dict,  'int_births')

    bar_chart_from_dict(agg_births, 'Births', 'Total Births by Scenario', plot_destination_folder, 'agg_births')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, births_dict, 'total_births',
        'Total Births',
        'Total Number of Births Per Year By Scenario',
        plot_destination_folder, 'births')

    # TOTAL COMPLETED PREGNANCIES
    def get_completed_pregnancies(comps_df, birth_dict, results_folder):
        """Sums the number of pregnancies that have ended in a given year including ectopic pregnancies,
        abortions, stillbirths and births"""

        def get_comps_p_all_or_int(years, birth_key):
            ectopic_mean_numbers_per_year = analysis_utility_functions.get_mean_and_quants_from_str_df(
                comps_df['pregnancy_supervisor'], 'ectopic_unruptured', years)[0]

            ia_mean_numbers_per_year = analysis_utility_functions.get_mean_and_quants_from_str_df(
                comps_df['pregnancy_supervisor'], 'induced_abortion', years)[0]

            sa_mean_numbers_per_year = analysis_utility_functions.get_mean_and_quants_from_str_df(
                comps_df['pregnancy_supervisor'], 'spontaneous_abortion', years)[0]

            ansb_df = extract_results(
                results_folder,
                module="tlo.methods.pregnancy_supervisor",
                key="antenatal_stillbirth",
                custom_generate_series=(
                    lambda df: df.loc[df['date'].dt.year > years[0]].assign(
                        year=df['date'].dt.year).groupby(['year'])['year'].count()),
                do_scaling=True
            )
            an_stillbirth_results = ansb_df.fillna(0)

            an_still_birth_data = analysis_utility_functions.get_mean_and_quants(an_stillbirth_results, years)

            total_completed_pregnancies_per_year = [a + b + c + d + e for a, b, c, d, e in
                                                    zip(birth_dict[birth_key][0], ectopic_mean_numbers_per_year,
                                                        ia_mean_numbers_per_year, sa_mean_numbers_per_year,
                                                        an_still_birth_data[0])]
            return total_completed_pregnancies_per_year

        tcp_all = get_comps_p_all_or_int(sim_years, 'total_births')
        tcp_int = get_comps_p_all_or_int(intervention_years, 'int_births')

        return {'total_cp': tcp_all,
                'int_cp': tcp_int}

    comp_pregs = {k: get_completed_pregnancies(comp_dfs[k], births_dict[k], results_folders[k]) for k in
                  results_folders}
    agg_c_pregs = dict()
    for k in comp_pregs:
        agg_c_pregs.update({k: sum(comp_pregs[k]['int_cp'])})

    # TWINS....
    # Extract and plot the twin birth rate across scenarios
    def get_twin_data(results_folder, birth_dict):
        t_df = extract_results(
            results_folder,
            module="tlo.methods.newborn_outcomes",
            key="twin_birth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
        )
        twins_results = t_df.fillna(0)
        results = dict()
        twin_data = analysis_utility_functions.get_mean_and_quants(twins_results, sim_years)
        twin_data_int = analysis_utility_functions.get_mean_and_quants(
            twins_results.loc[intervention_years[0]: intervention_years[-1]], intervention_years)

        for data, key in zip([twin_data, twin_data_int], ['total_births', 'int_births']):
            total_deliveries = [x - y for x, y in zip(birth_dict[key][0], data[0])]
            final_twining_rate = [(x / y) * 100 for x, y in zip(data[0], total_deliveries)]
            lq_rate = [(x / y) * 100 for x, y in zip(data[1], total_deliveries)]
            uq_rate = [(x / y) * 100 for x, y in zip(data[2], total_deliveries)]

            if 'total_births' in key:
                results.update({'twins_total': [final_twining_rate, lq_rate, uq_rate]})
            else:
                results.update({'twins_int': [final_twining_rate, lq_rate, uq_rate]})

        return results

    twin_data = {k: get_twin_data(results_folders[k], births_dict[k]) for k in results_folders}
    avg_twin_data = get_avg_rate_per_scenario(twin_data, True, 'twins_int')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, twin_data, 'twins_total',
        '% Total Births',        'Twin births as a Proportion of Total Births',
        plot_destination_folder, 'twins')

    bar_chart_from_dict(avg_twin_data, 'Proportion Total Births', 'Average Twin Birth Rate by Scenario',
                        plot_destination_folder, 'avg_twins')

    # ---------------------------------------- COMPLICATIONS... ----------------------------------------------
    for type, years in zip(['total', 'int'], [sim_years, intervention_years]):

        # Early Pregnancy loss...
        # Ectopic...
        ep_data = {k: analysis_utility_functions.get_comp_mean_and_rate(
            'ectopic_unruptured', preg_dict[k][f'{type}_preg'][0], comp_dfs[k]['pregnancy_supervisor'], 1000, years)
            for k in results_folders}

        # Miscarriage...
        sa_data = {k: analysis_utility_functions.get_comp_mean_and_rate(
            'spontaneous_abortion', comp_pregs[k][f'{type}_cp'], comp_dfs[k]['pregnancy_supervisor'], 1000, years)
            for k in results_folders}

        # Abortion...
        ia_data = {k: analysis_utility_functions.get_comp_mean_and_rate(
            'induced_abortion', comp_pregs[k][f'{type}_cp'], comp_dfs[k]['pregnancy_supervisor'], 1000, years)
            for k in results_folders}

        # Syphilis...
        syph_data = {k: analysis_utility_functions.get_comp_mean_and_rate(
            'syphilis', comp_pregs[k][f'{type}_cp'], comp_dfs[k]['pregnancy_supervisor'], 1000, years)
            for k in results_folders}

        # Gestational Diabetes...
        gdm_data = {k: analysis_utility_functions.get_comp_mean_and_rate(
            'gest_diab', comp_pregs[k][f'{type}_cp'], comp_dfs[k]['pregnancy_supervisor'], 1000, years)
            for k in results_folders}

        # PROM...
        prom_data = {k: analysis_utility_functions.get_comp_mean_and_rate(
            'PROM', births_dict[k][f'{type}_births'][0], comp_dfs[k]['pregnancy_supervisor'], 1000, years)
            for k in results_folders}

        # Placenta Praevia...
        praevia_data = {k: analysis_utility_functions.get_comp_mean_and_rate(
            'placenta_praevia', preg_dict[k][f'{type}_preg'][0], comp_dfs[k]['pregnancy_supervisor'], 1000, years)
            for k in results_folders}

        # Placental abruption...
        abruption = {k: analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
            'placental_abruption', births_dict[k][f'{type}_births'][0], 1000,
            [comp_dfs[k]['pregnancy_supervisor'], comp_dfs[k]['labour']],
            years)
            for k in results_folders}

        # Post term Labour...
        potl = {k: analysis_utility_functions.get_comp_mean_and_rate(
            'post_term_labour', preg_dict[k][f'{type}_preg'][0], comp_dfs[k]['labour'], 100, years)
            for k in results_folders}

        # Low Birth Weight...
        lbw = {k: analysis_utility_functions.get_comp_mean_and_rate(
            'low_birth_weight', births_dict[k][f'{type}_births'][0], neo_comp_dfs[k]['newborn_outcomes'], 100, years)
            for k in results_folders}

        # Macrosomia...
        macro = {k: analysis_utility_functions.get_comp_mean_and_rate(
            'macrosomia', births_dict[k][f'{type}_births'][0], neo_comp_dfs[k]['newborn_outcomes'], 100, years)
            for k in results_folders}

        # Small for gestational age...
        sga = {k: analysis_utility_functions.get_comp_mean_and_rate(
            'small_for_gestational_age', births_dict[k][f'{type}_births'][0], neo_comp_dfs[k]['newborn_outcomes'], 100,
            years)
            for k in results_folders}

        # Obstructed Labour...
        ol = {k: analysis_utility_functions.get_comp_mean_and_rate(
            'obstructed_labour', births_dict[k][f'{type}_births'][0], comp_dfs[k]['labour'], 1000, years)
            for k in results_folders}

        # Uterine Rupture...
        ur = {k: analysis_utility_functions.get_comp_mean_and_rate(
            'uterine_rupture', births_dict[k][f'{type}_births'][0], comp_dfs[k]['labour'], 1000, years)
            for k in results_folders}

        # Respiratory Depression...
        rd = {k: analysis_utility_functions.get_comp_mean_and_rate(
            'not_breathing_at_birth', births_dict[k][f'{type}_births'][0], neo_comp_dfs[k]['newborn_outcomes'], 1000,
            years) for k in results_folders}

        # Respiratory Distress Syndrome...
        rds_data = {k: analysis_utility_functions.get_comp_mean_and_rate(
            'respiratory_distress_syndrome', births_dict[k][f'{type}_births'][0], neo_comp_dfs[k]['newborn_outcomes'],
            1000, years) for k in results_folders}

        if type == 'total':
            analysis_utility_functions.comparison_graph_multiple_scenarios(
                scen_colours, years, ep_data, 'Rate per 1000 Completed Pregnancies',
                'Ectopic Pregnancy Rate Per Year By Scenario',
                plot_destination_folder, 'ectopic')

            analysis_utility_functions.comparison_graph_multiple_scenarios(
                scen_colours, years, sa_data, 'Rate per 1000 Completed Pregnancies',
                'Miscarriage Rate Per Year By Scenario',
                plot_destination_folder, 'miscarriage')

            analysis_utility_functions.comparison_graph_multiple_scenarios(
                scen_colours, years, ia_data, 'Rate per 1000 Completed Pregnancies',
                'Abortion Rate Per Year By Scenario',
                plot_destination_folder, 'abortion')

            analysis_utility_functions.comparison_graph_multiple_scenarios(
                scen_colours, years, syph_data, 'Rate per 1000 Completed Pregnancies',
                'Maternal Syphilis Rate Per Year By Scenario',
                plot_destination_folder, 'syphilis')

            analysis_utility_functions.comparison_graph_multiple_scenarios(
                scen_colours, years, gdm_data, 'Rate per 1000 Completed Pregnancies',
                'Gestational Diabetes Rate Per Year By Scenario',
                plot_destination_folder, 'gdm')

            analysis_utility_functions.comparison_graph_multiple_scenarios(
                scen_colours, years, prom_data, 'Rate per 1000 Births',
                'Premature Rupture of Membranes Rate Per Year By Scenario',
                plot_destination_folder, 'prom')

            analysis_utility_functions.comparison_graph_multiple_scenarios(
                scen_colours, years, praevia_data, 'Rate per 1000 Pregnancies',
                'Rate of Placenta Praevia Per Year Per Scenario',
                plot_destination_folder, 'praevia')

            analysis_utility_functions.comparison_graph_multiple_scenarios(
                scen_colours, years, abruption, 'Rate per 1000 Births',
                'Rate of Placental Abruption Per Year Per Scenario',
                plot_destination_folder, 'abruption')

            analysis_utility_functions.comparison_graph_multiple_scenarios(
                scen_colours, years, potl, 'Rate per 100 Births',
                'Rate of Post Term Labour Per Year Per Scenario',
                plot_destination_folder, 'potl')

            analysis_utility_functions.comparison_graph_multiple_scenarios(
                scen_colours, years, lbw, 'Rate per 100 Births',
                'Rate of Low Birth Weight Per Year Per Scenario',
                plot_destination_folder, 'lbw')

            analysis_utility_functions.comparison_graph_multiple_scenarios(
                scen_colours, years, macro, 'Rate per 100 Births',
                'Rate of Macrosomia Per Year Per Scenario',
                plot_destination_folder, 'macro')

            analysis_utility_functions.comparison_graph_multiple_scenarios(
                scen_colours, years, sga, 'Rate per 100 Births',
                'Rate of Small For Gestational Age Per Year Per Scenario',
                plot_destination_folder, 'sga')

            analysis_utility_functions.comparison_graph_multiple_scenarios(
                scen_colours, years, ol, 'Rate per 1000 Births',
                'Rate of Obstructed Labour Per Year Per Scenario',
                plot_destination_folder, 'ol')

            analysis_utility_functions.comparison_graph_multiple_scenarios(
                scen_colours, years, ur, 'Rate per 1000 Births',
                'Rate of Uterine Rupture Per Year Per Scenario',
                plot_destination_folder, 'ur')

            analysis_utility_functions.comparison_graph_multiple_scenarios(
                scen_colours, years, rd, 'Rate per 1000 Births',
                'Rate of Respiratory Depression Per Year Per Scenario',
                plot_destination_folder, 'rd')

            analysis_utility_functions.comparison_graph_multiple_scenarios(
                scen_colours, years, rds_data, 'Rate per 1000 Births',
                'Rate of Preterm Respiratory Distress Syndrome Per Year Per Scenario',
                plot_destination_folder, 'rds')

        else:
            avg_data = get_avg_rate_per_scenario(ep_data, False)
            avg_sa_data = get_avg_rate_per_scenario(sa_data, False)
            avg_ia_data = get_avg_rate_per_scenario(ia_data, False)
            avg_syph_data = get_avg_rate_per_scenario(syph_data, False)
            avg_gdm_data = get_avg_rate_per_scenario(gdm_data, False)
            avg_prom_data = get_avg_rate_per_scenario(prom_data, False)
            avg_praevia_data = get_avg_rate_per_scenario(praevia_data, False)
            avg_abruption = get_avg_rate_per_scenario(abruption, False)
            avg_potl = get_avg_rate_per_scenario(potl, False)
            avg_lbw = get_avg_rate_per_scenario(lbw, False)
            avg_macro = get_avg_rate_per_scenario(macro, False)
            avg_sga = get_avg_rate_per_scenario(sga, False)
            avg_ol = get_avg_rate_per_scenario(ol, False)
            avg_ur = get_avg_rate_per_scenario(ur, False)
            avg_rd = get_avg_rate_per_scenario(rd, False)
            avg_rds_data = get_avg_rate_per_scenario(rds_data, False)

            bar_chart_from_dict(avg_data, 'Avg. Rate per 1000 Completed Pregnancies',
                                'Average Ectopic Pregnancy Rate by Scenario',
                                plot_destination_folder, 'avg_ectopic')

            bar_chart_from_dict(avg_sa_data, 'Avg. Rate per 1000 Completed Pregnancies',
                                'Average Spontaneous Abortion Rate by Scenario',
                                plot_destination_folder, 'avg_miscarriage')

            bar_chart_from_dict(avg_ia_data, 'Avg. Rate per 1000 Completed Pregnancies',
                                'Average Induced Abortion Rate by Scenario',
                                plot_destination_folder, 'avg_abortion')

            bar_chart_from_dict(avg_syph_data, 'Avg. Rate per 1000 Completed Pregnancies',
                                'Average Maternal Syphilis Rate by Scenario',
                                plot_destination_folder, 'avg_syphilis')

            bar_chart_from_dict(avg_gdm_data, 'Avg. Rate per 1000 Completed Pregnancies',
                                'Average Gestational Diabetes Rate by Scenario',
                                plot_destination_folder, 'avg_gdm')

            bar_chart_from_dict(avg_prom_data, 'Avg. Rate per 1000 Births',
                                'Average Premature Rupture of Membranes Rate by Scenario',
                                plot_destination_folder, 'avg_prom')

            bar_chart_from_dict(avg_praevia_data, 'Avg. Rate per 1000 Pregnancies',
                                'Average Rate of Placenta Praevia by Scenario',
                                plot_destination_folder, 'avg_praevia')

            bar_chart_from_dict(avg_abruption, 'Avg. Rate per 1000 Births',
                                'Average Rate of Placental Abruption by Scenario',
                                plot_destination_folder, 'avg_abruption')

            bar_chart_from_dict(avg_potl, 'Avg. Rate per 100 Births',
                                'Average Rate of Post Term Labour by Scenario',
                                plot_destination_folder, 'avg_potl')

            bar_chart_from_dict(avg_lbw, 'Avg. Rate per 100 Births',
                                'Average Rate of Low Birth Weight by Scenario',
                                plot_destination_folder, 'avg_lbw')

            bar_chart_from_dict(avg_macro, 'Avg. Rate per 100 Births',
                                'Average Rate of Macrosomia by Scenario',
                                plot_destination_folder, 'avg_macro')

            bar_chart_from_dict(avg_sga, 'Avg. Rate per 100 Births',
                                'Average Rate of Small For Gestational Age by Scenario',
                                plot_destination_folder, 'avg_sga')

            bar_chart_from_dict(avg_ol, 'Avg. Rate per 1000 Births',
                                'Average Rate of Obstructed Labour by Scenario',
                                plot_destination_folder, 'avg_ol')

            bar_chart_from_dict(avg_ur, 'Avg. Rate per 1000 Births',
                                'Average Rate of Uterine Rupture by Scenario',
                                plot_destination_folder, 'avg_ur')

            bar_chart_from_dict(avg_rd, 'Avg. Rate per 1000 Births',
                                'Average Rate of Respiratory Depression by Scenario',
                                plot_destination_folder, 'avg_rd')

            bar_chart_from_dict(avg_rds_data, 'Avg. Rate per 1000 Births',
                                'Average Rate of Preterm Respiratory Distress Syndrome by Scenario',
                                plot_destination_folder, 'avg_rds')

    # OTHER COMPLICATIONS

    # Anaemia...
    def get_anaemia_output_at_birth(results_folder, total_births_per_year, years):
        anaemia_results = extract_results(
            results_folder,
            module="tlo.methods.pregnancy_supervisor",
            key="anaemia_on_birth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'anaemia_status'])['year'].count()),
            do_scaling=True
        )

        no_anaemia_data = analysis_utility_functions.get_mean_and_quants_from_str_df(anaemia_results, 'none',
                                                                                     years)

        prevalence_of_anaemia_per_year = [100 - ((x / y) * 100) for x, y in zip(no_anaemia_data[0],
                                                                                total_births_per_year)]
        no_anaemia_lqs = [100 - ((x / y) * 100) for x, y in zip(no_anaemia_data[1], total_births_per_year)]
        no_anaemia_uqs = [100 - ((x / y) * 100) for x, y in zip(no_anaemia_data[2], total_births_per_year)]

        return [prevalence_of_anaemia_per_year, no_anaemia_lqs, no_anaemia_uqs]

    anaemia_birth_data = {k: get_anaemia_output_at_birth(results_folders[k], births_dict[k]['total_births'][0],
                                                         sim_years) for k in
                          results_folders}
    anaemia_birth_data_int = {k: get_anaemia_output_at_birth(results_folders[k], births_dict[k]['int_births'][0],
                                                             intervention_years) for k in results_folders}

    avg_anaemia_birth_data = get_avg_rate_per_scenario(anaemia_birth_data_int, False)

    # Repeat this process looking at anaemia at the time of delivery
    def get_anaemia_output_at_delivery(results_folder, total_births_per_year, years):
        pnc_anaemia = extract_results(
            results_folder,
            module="tlo.methods.postnatal_supervisor",
            key="total_mat_pnc_visits",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'anaemia'])['mother'].count()),
            do_scaling=True
        )

        no_anaemia_data = analysis_utility_functions.get_mean_and_quants_from_str_df(pnc_anaemia, 'none',
                                                                                     years)
        prevalence_of_anaemia_per_year = [100 - ((x / y) * 100) for x, y in zip(no_anaemia_data[0],
                                                                                total_births_per_year)]
        no_anaemia_lqs = [100 - ((x / y) * 100) for x, y in zip(no_anaemia_data[1], total_births_per_year)]
        no_anaemia_uqs = [100 - ((x / y) * 100) for x, y in zip(no_anaemia_data[2], total_births_per_year)]

        return [prevalence_of_anaemia_per_year, no_anaemia_lqs, no_anaemia_uqs]

    anaemia_delivery_data = {k: get_anaemia_output_at_delivery(
        results_folders[k], births_dict[k]['total_births'][0], sim_years) for k in results_folders}

    anaemia_delivery_data_int = {k: get_anaemia_output_at_delivery(
        results_folders[k], births_dict[k]['int_births'][0], intervention_years) for k in results_folders}

    avg_anaemia_delivery_data = get_avg_rate_per_scenario(anaemia_delivery_data_int, False)

    analysis_utility_functions.comparison_graph_multiple_scenarios(
        scen_colours, sim_years, anaemia_birth_data, '% Total Births',
        'Prevalence of Anaemia at Birth Per Year By Scenario',
        plot_destination_folder, 'anaemia_birth')

    analysis_utility_functions.comparison_graph_multiple_scenarios(
        scen_colours, sim_years, anaemia_delivery_data, '% Total Births',
        'Prevalence of Anaemia at End of Postnatal Period Per Year By Scenario',
        plot_destination_folder, 'anaemia_pn')

    bar_chart_from_dict(avg_anaemia_birth_data, 'Avg. % of Total Births',
                        'Prevalence of Anaemia at Birth by Scenario',
                        plot_destination_folder, 'avg_anaemia_birth')

    bar_chart_from_dict(avg_anaemia_delivery_data, 'Avg. % of Total Births',
                        'Prevalence of Anaemia at End of Postnatal Period by Scenario',
                        plot_destination_folder, 'avg_anaemia_pn')

    # Hypertensive disorders ...
    def get_htn_disorders_outputs(comps_df, total_births_per_year, years):
        """Extracts rates of HDPs across the modules"""
        output = dict()

        # Here total rates of each HDP is provided (summing cases across time periods)
        output['gh'] = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
            'mild_gest_htn', total_births_per_year, 1000, [comps_df['pregnancy_supervisor'],
                                                           comps_df['postnatal_supervisor']], years)

        output['sgh'] = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
            'severe_gest_htn', total_births_per_year, 1000, [comps_df['pregnancy_supervisor'], comps_df['labour'],
                                                             comps_df['postnatal_supervisor']], years)

        output['mpe'] = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
            'mild_pre_eclamp', total_births_per_year, 1000, [comps_df['pregnancy_supervisor'],
                                                             comps_df['postnatal_supervisor']], years)

        output['spe'] = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
            'severe_pre_eclamp', total_births_per_year, 1000, [comps_df['pregnancy_supervisor'], comps_df['labour'],
                                                               comps_df['postnatal_supervisor']], years)

        output['ec'] = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
            'eclampsia', total_births_per_year, 1000, [comps_df['pregnancy_supervisor'], comps_df['labour'],
                                                       comps_df['postnatal_supervisor']], years)
        return output

    htn_data = {k: get_htn_disorders_outputs(comp_dfs[k], births_dict[k]['total_births'][0], sim_years)
                for k in results_folders}
    htn_data_int = {k: get_htn_disorders_outputs(comp_dfs[k], births_dict[k]['int_births'][0], intervention_years)
                    for k in results_folders}

    for key, condition in zip(['gh', 'sgh', 'mpe', 'spe', 'ec'],
                              ['Gestational Hypertension', 'Severe Gestational Hypertension',
                                'Mild Pre-eclampsia', 'Severe pre-eclampsia', 'Eclampsia']):

        avg_data = get_avg_rate_per_scenario(htn_data_int, True, key)
        d = {key: avg_data}
        output_df = output_df.append(pd.DataFrame.from_dict(d, orient='index'))

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, htn_data, key,
            'Rate per 1000 Births',
            f'{condition} Rate Per Year Per Scenario',
            plot_destination_folder, key)

        bar_chart_from_dict(avg_data, 'Avg. Rate per 1000 Births',
                            f'Average Rate of {condition} per Scenario',
                            plot_destination_folder, f'avg_{key}')

    #  Antepartum Haemorrhage...

    def get_aph_data(comps_df, total_births_per_year, years):
        """Extract incidence of mild/moderate and severe antepartum haemorrhage across the modules"""

        mm_aph_data = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
            'mild_mod_antepartum_haemorrhage', total_births_per_year, 1000,
            [comps_df['pregnancy_supervisor'], comps_df['labour']], years)

        s_aph_data = analysis_utility_functions. get_comp_mean_and_rate_across_multiple_dataframes(
            'severe_antepartum_haemorrhage', total_births_per_year, 1000,
            [comps_df['pregnancy_supervisor'], comps_df['labour']], years)

        total_aph_rates = [x + y for x, y in zip(mm_aph_data[0], s_aph_data[0])]
        aph_lqs = [x + y for x, y in zip(mm_aph_data[1], s_aph_data[1])]
        aph_uqs = [x + y for x, y in zip(mm_aph_data[2], s_aph_data[2])]

        return [total_aph_rates, aph_lqs, aph_uqs]

    aph_data = {k: get_aph_data(comp_dfs[k], births_dict[k]['total_births'][0], sim_years) for k in
                results_folders}
    aph_data_int = {k: get_aph_data(comp_dfs[k], births_dict[k]['int_births'][0], intervention_years) for k in
                    results_folders}

    avg_aph_data = get_avg_rate_per_scenario(aph_data_int, False)

    analysis_utility_functions.comparison_graph_multiple_scenarios(
        scen_colours, sim_years, aph_data, 'Rate per 1000 Births',
        'Rate of Antepartum Haemorrhage Per Year Per Scenario',
        plot_destination_folder, 'aph')

    bar_chart_from_dict(avg_aph_data, 'Avg. Rate per 1000 Births',
                        'Average Rate of Antepartum Haemorrhage by Scenario',
                        plot_destination_folder, 'avg_aph')

    # --------------------------------------------- Preterm birth ... ------------------------------------------------
    # Extract and plot rate of preterm birth rates across scenarios
    def get_ptl_data(total_births_per_year, comps_df, years):
        early_ptl_data = analysis_utility_functions.get_comp_mean_and_rate(
            'early_preterm_labour', total_births_per_year, comps_df['labour'], 100, years)
        late_ptl_data = analysis_utility_functions.get_comp_mean_and_rate(
            'late_preterm_labour', total_births_per_year, comps_df['labour'], 100, years)

        total_ptl_rates = [x + y for x, y in zip(early_ptl_data[0], late_ptl_data[0])]
        ptl_lqs = [x + y for x, y in zip(early_ptl_data[1], late_ptl_data[1])]
        ltl_uqs = [x + y for x, y in zip(early_ptl_data[2], late_ptl_data[2])]

        return [total_ptl_rates, ptl_lqs, ltl_uqs]

    ptl_data = {k: get_ptl_data(births_dict[k]['total_births'][0], comp_dfs[k], sim_years)
                for k in results_folders}
    ptl_data_int = {k: get_ptl_data(births_dict[k]['int_births'][0], comp_dfs[k], intervention_years)
                    for k in results_folders}

    avg_ptl_data = get_avg_rate_per_scenario(ptl_data_int, False)

    analysis_utility_functions.comparison_graph_multiple_scenarios(
        scen_colours, sim_years, ptl_data, 'Rate per 100 Births',
        'Rate of Preterm Labour Per Year Per Scenario',
        plot_destination_folder, 'ptl')

    bar_chart_from_dict(avg_ptl_data, 'Avg. Rate per 100 Births',
                        'Average Rate of Preterm Labour by Scenario',
                        plot_destination_folder, 'avg_ptl')

    # Caesarean Section Rate & Assisted Vaginal Delivery Rate...
    def get_delivery_data(results_folder, total_births_per_year, years):
        delivery_mode = extract_results(
            results_folder,
            module="tlo.methods.labour",
            key="delivery_setting_and_mode",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'mode'])['mother'].count()),
            do_scaling=True
        )

        cs_data = analysis_utility_functions.get_comp_mean_and_rate(
            'caesarean_section', total_births_per_year, delivery_mode, 100, years)
        avd_data = analysis_utility_functions.get_comp_mean_and_rate(
            'instrumental', total_births_per_year, delivery_mode, 100, years)

        return {'cs': cs_data,
                'avd': avd_data}

    delivery_data = {k: get_delivery_data(results_folders[k], births_dict[k]['total_births'][0], sim_years)
                     for k in results_folders}
    delivery_data_int = {k: get_delivery_data(results_folders[k], births_dict[k]['int_births'][0], intervention_years)
                         for k in results_folders}

    for key, intervention in zip(['cs', 'avd'], ['Caesarean Section', 'Assisted Vaginal Delivery']):

        avg = get_avg_rate_per_scenario(delivery_data_int, True, key)
        d = {key: avg}
        output_df = output_df.append(pd.DataFrame.from_dict(d, orient='index'))

        bar_chart_from_dict(avg, 'Avg. % of Total Births',
                            f'Average Rate of Delivery via {intervention} by Scenario',
                            plot_destination_folder, f'avg_{key}')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, delivery_data, key,
            '% Total Births',
            f'{intervention} Rate Per Year Per Scenario',
            plot_destination_folder, key)

    #  Maternal Sepsis Rate...
    def get_total_sepsis_rates(total_births_per_year, comps_df, years):
        an_sep_data = analysis_utility_functions.get_comp_mean_and_rate(
            'clinical_chorioamnionitis', total_births_per_year, comps_df['pregnancy_supervisor'], 1000,
            years)
        an_number = analysis_utility_functions.get_mean_and_quants_from_str_df(comps_df['pregnancy_supervisor'],
                                                                               'clinical_chorioamnionitis',
                                                                               years)

        la_sep_data = analysis_utility_functions.get_comp_mean_and_rate(
            'sepsis', total_births_per_year, comps_df['labour'], 1000, years)
        la_number = analysis_utility_functions.get_mean_and_quants_from_str_df(comps_df['labour'],
                                                                               'sepsis',
                                                                               years)

        pn_la_sep_data = analysis_utility_functions.get_comp_mean_and_rate(
            'sepsis_postnatal', total_births_per_year, comps_df['labour'], 1000, years)
        pn_la_number = analysis_utility_functions.get_mean_and_quants_from_str_df(
            comps_df['labour'], 'sepsis_postnatal', years)

        pn_sep_data = analysis_utility_functions.get_comp_mean_and_rate(
            'sepsis', total_births_per_year, comps_df['postnatal_supervisor'], 1000, years)
        pn_number = analysis_utility_functions.get_mean_and_quants_from_str_df(comps_df['postnatal_supervisor'],
                                                                               'sepsis', years)

        complete_pn_sep_data = [x + y for x, y in zip(pn_la_sep_data[0], pn_sep_data[0])]
        complete_pn_sep_lq = [x + y for x, y in zip(pn_la_sep_data[1], pn_sep_data[1])]
        complete_pn_sep_up = [x + y for x, y in zip(pn_la_sep_data[2], pn_sep_data[2])]

        total_sep_rates = [x + y + z for x, y, z in zip(an_sep_data[0], la_sep_data[0], complete_pn_sep_data)]
        sep_lq = [x + y + z for x, y, z in zip(an_sep_data[1], la_sep_data[1], complete_pn_sep_lq)]
        sep_uq = [x + y + z for x, y, z in zip(an_sep_data[2], la_sep_data[2], complete_pn_sep_up)]

        return {'total_sep': [total_sep_rates, sep_lq, sep_uq],
                'an_sep': an_sep_data,
                'la_sep': la_sep_data,
                'pn_sep': [complete_pn_sep_data, complete_pn_sep_lq, complete_pn_sep_up],
                'an_number': an_number,
                'la_number': la_number,
                'pn_la_number': pn_la_number,
                'pn_number': pn_number
                }

    sep_data = {k: get_total_sepsis_rates(births_dict[k]['total_births'][0], comp_dfs[k], sim_years)
                for k in results_folders}
    sep_data_int = {k: get_total_sepsis_rates(births_dict[k]['int_births'][0], comp_dfs[k], intervention_years)
                    for k in results_folders}

    for key, condition in zip(['total_sep', 'an_sep', 'la_sep', 'pn_sep'],
                              ['Maternal Sepsis', 'Antenatal Sepsis', 'Intrapartum Sepsis', 'Postnatal Sepsis']):
        avg = get_avg_rate_per_scenario(sep_data_int, True, key)

        d = {key: avg}
        output_df = output_df.append(pd.DataFrame.from_dict(d, orient='index'))

        bar_chart_from_dict(avg, 'Avg. Rate per 1000 Births',
                            f'Average Rate of {condition} by Scenario',
                            plot_destination_folder, f'avg_{key}')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, sep_data, key,
            'Rate per 1000 Births',
            f'{condition} Rate Per Year Per Scenario',
            plot_destination_folder, key)

    #  Postpartum Haemorrhage...
    def get_pph_data(total_births_per_year, comps_df, years):
        la_pph_data = analysis_utility_functions.get_comp_mean_and_rate(
            'primary_postpartum_haemorrhage', total_births_per_year, comps_df['postnatal_supervisor'], 1000, years)
        pn_pph_data = analysis_utility_functions.get_comp_mean_and_rate(
            'secondary_postpartum_haemorrhage', total_births_per_year, comps_df['postnatal_supervisor'],
            1000, years)

        total_pph_rates = [x + y for x, y in zip(la_pph_data[0], pn_pph_data[0])]
        pph_lq = [x + y for x, y in zip(la_pph_data[1], pn_pph_data[1])]
        pph_uq = [x + y for x, y in zip(la_pph_data[2], pn_pph_data[2])]

        return [total_pph_rates, pph_lq, pph_uq]

    pph_data = {k: get_pph_data(births_dict[k]['total_births'][0], comp_dfs[k], sim_years) for k in results_folders}
    pph_data_int = {k: get_pph_data(births_dict[k]['int_births'][0], comp_dfs[k], intervention_years)
                    for k in results_folders}

    avg_pph = get_avg_rate_per_scenario(pph_data_int, False)

    analysis_utility_functions.comparison_graph_multiple_scenarios(
        scen_colours, sim_years, pph_data, 'Rate per 1000 Births',
        'Rate of Postpartum Haemorrhage Per Year Per Scenario',
        plot_destination_folder, 'pph')

    bar_chart_from_dict(avg_pph, 'Avg. Rate per 1000 Births',
                        'Average Rate of Postpartum Haemorrhage by Scenario',
                        plot_destination_folder, 'avg_pph')

    #  Neonatal sepsis (labour & postnatal)
    def get_neonatal_sepsis(total_births_per_year, nb_comp_dfs, years):
        early_ns_data = analysis_utility_functions.get_comp_mean_and_rate(
            'early_onset_sepsis', total_births_per_year, nb_comp_dfs['newborn_outcomes'], 1000, years)

        early_ns_pn = analysis_utility_functions.get_comp_mean_and_rate(
            'early_onset_sepsis', total_births_per_year, nb_comp_dfs['newborn_postnatal'], 1000, years)

        late_ns_data = analysis_utility_functions.get_comp_mean_and_rate(
            'late_onset_sepsis', total_births_per_year, nb_comp_dfs['newborn_postnatal'], 1000, years)

        total_ns_rates = [x + y + z for x, y, z in zip(early_ns_data[0], early_ns_pn[0], late_ns_data[0])]
        ns_lqs = [x + y + z for x, y, z in zip(early_ns_data[1], early_ns_pn[1], late_ns_data[1])]
        ns_uqs = [x + y + z for x, y, z in zip(early_ns_data[2], early_ns_pn[2], late_ns_data[2])]

        return [total_ns_rates, ns_lqs, ns_uqs]

    neo_sep_data = {k: get_neonatal_sepsis(births_dict[k]['total_births'][0], neo_comp_dfs[k], sim_years)
                    for k in results_folders}
    neo_sep_data_int = {k: get_neonatal_sepsis(births_dict[k]['int_births'][0], neo_comp_dfs[k], intervention_years)
                        for k in results_folders}

    avg_neo_sep_data = get_avg_rate_per_scenario(neo_sep_data_int, False)

    analysis_utility_functions.comparison_graph_multiple_scenarios(
        scen_colours, sim_years, neo_sep_data, 'Rate per 1000 Births',
        'Rate of Neonatal Sepsis Per Year Per Scenario',
        plot_destination_folder, 'neo_sep')

    bar_chart_from_dict(avg_neo_sep_data, 'Avg. Rate per 1000 Births',
                        'Average Rate of Neonatal Sepsis by Scenario',
                        plot_destination_folder, 'avg_neo_sep')

    #   Neonatal encephalopathy ...
    def get_neonatal_encephalopathy(total_births_per_year, nb_comp_dfs, years):
        mild_data = analysis_utility_functions.get_comp_mean_and_rate(
            'mild_enceph', total_births_per_year,  nb_comp_dfs['newborn_outcomes'], 1000, years)
        mod_data = analysis_utility_functions.get_comp_mean_and_rate(
            'moderate_enceph', total_births_per_year,  nb_comp_dfs['newborn_outcomes'], 1000, years)
        sev_data = analysis_utility_functions.get_comp_mean_and_rate(
            'severe_enceph', total_births_per_year,  nb_comp_dfs['newborn_outcomes'], 1000, years)

        total_enceph_rates = [x + y + z for x, y, z in zip(mild_data[0], mod_data[0], sev_data[0])]
        enceph_lq = [x + y + z for x, y, z in zip(mild_data[1], mod_data[1], sev_data[1])]
        enceph_uq = [x + y + z for x, y, z in zip(mild_data[2], mod_data[2], sev_data[2])]

        return [total_enceph_rates, enceph_lq, enceph_uq]

    neo_enceph_data = {k: get_neonatal_encephalopathy(births_dict[k]['total_births'][0], neo_comp_dfs[k],
                                                      sim_years) for k in results_folders}
    neo_enceph_data_int = {k: get_neonatal_encephalopathy(births_dict[k]['int_births'][0], neo_comp_dfs[k],
                                                          intervention_years) for k in results_folders}

    avg_neo_enceph_data = get_avg_rate_per_scenario(neo_enceph_data_int, False)

    analysis_utility_functions.comparison_graph_multiple_scenarios(
        scen_colours, sim_years, neo_enceph_data, 'Rate per 1000 Births',
        'Rate of Neonatal Encephalopathy Per Year Per Scenario',
        plot_destination_folder, 'neo_enceph')

    bar_chart_from_dict(avg_neo_enceph_data, 'Avg. Rate per 1000 Births',
                        'Average Rate of Neonatal Encephalopathy by Scenario',
                        plot_destination_folder, 'avg_neo_enceph')

    # ===================================== COMPARING COMPLICATION LEVEL MMR ==========================================
    simplified_causes = ['ectopic_pregnancy', 'abortion', 'severe_pre_eclampsia', 'sepsis', 'uterine_rupture',
                         'postpartum_haemorrhage', 'antepartum_haemorrhage']

    def get_death_data(results_folder, total_births):

        dd_df = extract_results(
            results_folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'cause'])['year'].count()
            ),
            do_scaling=True
        )
        direct_death_results = dd_df.fillna(0)
        mmr_dict = dict()
        crude_deaths = dict()

        for cause in simplified_causes:
            if (cause == 'ectopic_pregnancy') or (cause == 'antepartum_haemorrhage') or (cause == 'uterine_rupture'):
                deaths_data = analysis_utility_functions.get_mean_and_quants_from_str_df(
                    direct_death_results, cause, sim_years)
                crude_deaths.update({cause: deaths_data[0]})

            elif cause == 'abortion':
                def get_ab_mmr(death_results):
                    ia_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                        death_results, 'induced_abortion', sim_years)
                    sa_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                        death_results, 'spontaneous_abortion', sim_years)
                    mean_deaths = [x + y for x, y in zip(ia_deaths[0], sa_deaths[0])]
                    lq_deaths = [x + y for x, y in zip(ia_deaths[1], sa_deaths[1])]
                    uq_deaths = [x + y for x, y in zip(ia_deaths[2], sa_deaths[2])]

                    return [mean_deaths, lq_deaths, uq_deaths]
                deaths_data = get_ab_mmr(direct_death_results)
                crude_deaths.update({cause: deaths_data[0]})

            elif cause == 'severe_pre_eclampsia':
                def get_htn_mmr(death_results):
                    spe_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                        death_results, 'severe_pre_eclampsia', sim_years)
                    ec_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                        death_results, 'eclampsia', sim_years)
                    sgh_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                        death_results, 'severe_gestational_hypertension', sim_years)

                    deaths = [x + y + z for x, y, z in zip(spe_deaths[0], ec_deaths[0], sgh_deaths[0])]
                    lq = [x + y + z for x, y, z in zip(spe_deaths[1], ec_deaths[1], sgh_deaths[1])]
                    uq = [x + y + z for x, y, z in zip(spe_deaths[2], ec_deaths[2], sgh_deaths[2])]

                    return [deaths, lq, uq]

                deaths_data = get_htn_mmr(direct_death_results)
                crude_deaths.update({cause: deaths_data[0]})

            elif cause == 'postpartum_haemorrhage':
                def get_pph_mmr(death_results):
                    p_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                        death_results, 'postpartum_haemorrhage', sim_years)
                    s_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                        death_results, 'secondary_postpartum_haemorrhage', sim_years)
                    deaths = [x + y for x, y in zip(p_deaths[0], s_deaths[0])]
                    lq = [x + y for x, y in zip(p_deaths[1], s_deaths[1])]
                    uq = [x + y for x, y in zip(p_deaths[2], s_deaths[2])]

                    return [deaths, lq, uq]
                deaths_data = get_pph_mmr(direct_death_results)
                crude_deaths.update({cause: deaths_data[0]})

            elif cause == 'sepsis':
                def get_sep_mmr(death_results):
                    a_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                        death_results, 'antenatal_sepsis', sim_years)
                    i_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                        death_results, 'intrapartum_sepsis', sim_years)
                    p_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                        death_results, 'postpartum_sepsis', sim_years)

                    deaths = [x + y + z for x, y, z in zip(a_deaths[0], i_deaths[0], p_deaths[0])]
                    lq = [x + y + z for x, y, z in zip(a_deaths[1], i_deaths[1], p_deaths[1])]
                    uq = [x + y + z for x, y, z in zip(a_deaths[2], i_deaths[2], p_deaths[2])]

                    return [deaths, lq, uq]

                deaths_data = get_sep_mmr(direct_death_results)
                crude_deaths.update({cause: deaths_data[0]})

            print(cause, deaths_data)

            mmr_mean = [(x / y) * 100000 for x, y in zip(deaths_data[0], total_births[0])]
            mmr_lq = [(x / y) * 100000 for x, y in zip(deaths_data[1], total_births[1])]
            mmr_uq = [(x / y) * 100000 for x, y in zip(deaths_data[2], total_births[2])]

            mmr_dict.update({cause: [mmr_mean, mmr_lq, mmr_uq]})

        # todo: stacked area chart

        return mmr_dict

    comp_mmrs = {k: get_death_data(results_folders[k], births_dict[k]['total_births']) for k in results_folders}

    path = f'{plot_destination_folder}/comp_mmr'
    if not os.path.isdir(path):
        os.makedirs(f'{plot_destination_folder}/comp_mmr')

    mmr_destination = path

    for cause, condition in zip(simplified_causes, ['Ectopic Pregnancy', 'Abortion', 'Severe Pre-eclampsia/Eclampsia',
                                                    'Maternal Sepsis', 'Uterine Rupture',  'Postpartum Haemorrhage',
                                                    'Antepartum Haemorrhage']):
        avg = get_avg_rate_per_scenario(comp_mmrs, True, cause)
        bar_chart_from_dict(avg, 'Avg. MMR',
                            f'Average MMR due to {condition} by Scenario',
                            mmr_destination, f'avg_{cause}_mmr')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, comp_mmrs, cause,
            'Rate per 1000 Births',
            f'MMR due to {condition} Per Year Per Scenario',
            mmr_destination, f'{cause}_mmr')

    # ===================================== COMPARING COMPLICATION LEVEL NMR ========================================
    simplified_causes_neo = ['prematurity', 'encephalopathy', 'neonatal_sepsis', 'neonatal_respiratory_depression']

    def get_neo_death_data(results_folder, births):

        dd_df = extract_results(
            results_folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'cause'])['year'].count()
            ),
            do_scaling=True
        )
        direct_death_results = dd_df.fillna(0)

        nmr_dict = dict()

        for cause in simplified_causes_neo:
            if (cause == 'encephalopathy') or (cause == 'neonatal_respiratory_depression'):
                deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(direct_death_results, cause,
                                                                                    sim_years)

            elif cause == 'neonatal_sepsis':
                def get_neo_sep_deaths(death_results):
                    early1 = analysis_utility_functions.get_mean_and_quants_from_str_df(
                        death_results, 'early_onset_neonatal_sepsis', sim_years)
                    early2 = analysis_utility_functions.get_mean_and_quants_from_str_df(
                        death_results, 'early_onset_sepsis', sim_years)
                    late = analysis_utility_functions.get_mean_and_quants_from_str_df(
                        death_results, 'late_onset_sepsis', sim_years)

                    deaths = [x + y + z for x, y, z in zip(early1[0], early2[0], late[0])]
                    lq = [x + y + z for x, y, z in zip(early1[1], early2[1], late[1])]
                    uq = [x + y + z for x, y, z in zip(early1[2], early2[2], late[2])]

                    return [deaths, lq, uq]

                deaths = get_neo_sep_deaths(direct_death_results)

            elif cause == 'prematurity':
                def get_pt_deaths(death_results):
                    rds_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                        death_results, 'respiratory_distress_syndrome', sim_years)
                    other_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                        death_results, 'preterm_other', sim_years)
                    deaths = [x + y for x, y in zip(rds_deaths[0], other_deaths[0])]
                    lq = [x + y for x, y in zip(rds_deaths[1], other_deaths[1])]
                    uq = [x + y for x, y in zip(rds_deaths[2], other_deaths[2])]

                    return [deaths, lq, uq]

                deaths = get_pt_deaths(direct_death_results)

            nmr_mean = [(x / y) * 1000 for x, y in zip(deaths[0], births[0])]
            nmr_lq = [(x / y) * 1000 for x, y in zip(deaths[1], births[1])]
            nmr_uq = [(x / y) * 1000 for x, y in zip(deaths[2], births[2])]

            nmr_dict.update({cause: [nmr_mean, nmr_lq, nmr_uq]})

        return  nmr_dict

    nmr_data = {k: get_neo_death_data(results_folders[k], births_dict[k]['total_births']) for k in results_folders}

    path = f'{plot_destination_folder}/comp_nmr'
    if not os.path.isdir(path):
        os.makedirs(f'{plot_destination_folder}/comp_nmr')

    nmr_destination = path

    for cause, condition in zip(simplified_causes_neo, ['Prematurity', 'Neonatal Encephalopathy', 'Neonatal Sepsis',
                                                        'Neonatal Respiratory Depression']):
        avg = get_avg_rate_per_scenario(nmr_data, True, cause)
        bar_chart_from_dict(avg, 'Avg. NMR',
                            f'Average NMR due to {condition} by Scenario',
                            nmr_destination, f'avg_{cause}_nmr')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, nmr_data, cause,
            'NMR',
            f'NMR due to {condition} Per Year Per Scenario',
            nmr_destination, f'{cause}_nmr')

    for data, key in zip([avg_abruption, avg_anaemia_birth_data, avg_anaemia_delivery_data, avg_aph_data,
                          avg_gdm_data, avg_ia_data, avg_lbw, avg_macro, avg_neo_enceph_data, avg_neo_sep_data,
                          avg_ol, avg_potl, avg_pph, avg_praevia_data, avg_prom_data, avg_ptl_data, avg_rd, avg_rds_data,
                          avg_sa_data, avg_sga, avg_syph_data, avg_twin_data, avg_ur],
                         ['abruption', 'anaemia_birth', 'anaemia_delivery', 'aph', 'gdm', 'abortion', 'lbw',
                          'macrosomia', 'enceph', 'neo_sepsis', 'ol', 'potl', 'pph', 'praevia', 'prom', 'ptl',
                          'resp_d', 'rds', 'miscarriage', 'sga', 'syphilis', 'twins', 'ur']):
        d = {key: data}
        output_df = output_df.append(pd.DataFrame.from_dict(d, orient='index'))

    output_df.to_csv(f'{plot_destination_folder}/avg_inc_outputs.csv')
