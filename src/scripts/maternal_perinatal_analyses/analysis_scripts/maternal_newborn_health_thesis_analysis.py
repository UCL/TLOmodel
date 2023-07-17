import os

import analysis_utility_functions
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import extract_results, get_scenario_outputs
from analysis_utility_functions import get_mean_from_columns, get_mean_95_CI_from_list, return_95_CI_across_runs
import scipy.stats

plt.style.use('seaborn-darkgrid')

def run_maternal_newborn_health_thesis_analysis(scenario_file_dict, outputspath, sim_years,
                                         intervention_years, service_of_interest, scen_colours):
    """
    """
    results_folders = {k: get_scenario_outputs(scenario_file_dict[k], outputspath)[-1] for k in scenario_file_dict}
    scenario_names = list(results_folders.keys())

    # Create folder to store graphs (if it hasn't already been created when ran previously)
    path = f'{outputspath}/{service_of_interest}_thesis_analysis_graphs_and_results_' \
           f'{results_folders[scenario_names[0]].name}'

    if not os.path.isdir(path):
        os.makedirs(f'{outputspath}/{service_of_interest}_thesis_analysis_graphs_and_results_'
                    f'{results_folders[scenario_names[0]].name}')

    primary_oc_path = f'{path}/primary_outcomes'
    if not os.path.isdir(primary_oc_path):
        os.makedirs(f'{path}/primary_outcomes')

    comp_incidence_path = f'{path}/comp_incidence'
    if not os.path.isdir(comp_incidence_path):
        os.makedirs(f'{path}/comp_incidence')

    secondary_oc_path = f'{path}/secondary_outcomes'
    if not os.path.isdir(secondary_oc_path):
        os.makedirs(f'{path}/secondary_outcomes')

    scenario_titles = list(results_folders.keys())

    output_df = pd.DataFrame(
        columns=['scenario',
                 'output',
                 'mean_95%CI_value_for_int_period',
                 'skew_for_int_data',
                 'mean_95%CI_diff_outcome_int_period',
                 'skew_for_diff_data',
                 'median_diff_outcome_int_period'])

    # DEFINE HELPER FUNCTIONs....
    def plot_agg_graph(data, key, y_label, title, save_name, save_location):
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

        ci = [(x - y) / 2 for x, y in zip(uq_vals, lq_vals)]
        ax.bar(labels, mean_vals, color=scen_colours, width=width, yerr=ci)
        ax.tick_params(axis='x', which='major', labelsize=8)

        ax.set_ylabel(y_label)
        ax.set_xlabel('Scenario')
        ax.set_title(title)
        plt.savefig(f'{save_location}/{save_name}.png')
        plt.show()

    def get_med_or_mean_from_columns(df, mean_or_med):
        values = list()
        for col in df:
            if mean_or_med == 'mean':
                values.append(np.mean(df[col]))
            elif mean_or_med == 'median':
                values.append(np.median(df[col]))
            else:
                values.append(sum(df[col]))
        return values

    def get_diff_between_runs(dfs, baseline, intervention, keys, intervention_years, output_df):

        def get_mean_and_confidence_interval(data, confidence=0.95):
            a = 1.0 * np.array(data)
            n = len(a)
            m, se = np.mean(a), scipy.stats.sem(a)
            h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)

            round(m, 3),
            round(h, 2)

            return m, m - h, m + h

        # TODO: replace with below, neater
        # st.t.interval(0.95, len(mean_diff_list) - 1, loc=np.mean(mean_diff_list), scale=st.sem(mean_diff_list))

        for k in keys:
            # Get DF which gives difference between outcomes for each run
            diff = dfs[baseline][k] - dfs[intervention][k]
            int_diff = diff.loc[intervention_years[0]: intervention_years[-1]]

            # Calculate, skew, mean and 95% CI for outcome in intervention
            if 'total' in k:
                operation = 'agg'
            else:
                operation = 'mean'

            mean_outcome_list_int = get_med_or_mean_from_columns(dfs[intervention][k].loc[intervention_years[0]:
                                                                                          intervention_years[-1]],
                                                                 operation)
            skew_mean_outcome_list_int = scipy.stats.skew(mean_outcome_list_int)
            mean_outcome_value_int = get_mean_and_confidence_interval(mean_outcome_list_int)

            # Calculate mean difference between outcome by run for intervention period, check skew and
            # calculate mean/95 % CI
            mean_diff_list = get_med_or_mean_from_columns(int_diff.loc[intervention_years[0]:
                                                                       intervention_years[-1]], operation)

            skew_diff_list = scipy.stats.skew(mean_diff_list)
            mean_outcome_diff = get_mean_and_confidence_interval(mean_diff_list)
            median_outcome_diff = [round(np.median(mean_diff_list), 2),
                                   round(np.quantile(mean_diff_list, 0.025), 2),
                                   round(np.quantile(mean_diff_list, 0.975), 2)]

            res_df = pd.DataFrame([(intervention,
                                    k,
                                    [round(x, 2) for x in mean_outcome_value_int],
                                    round(skew_mean_outcome_list_int, 2),
                                    [round(x, 2) for x in mean_outcome_diff],
                                    skew_diff_list,
                                    median_outcome_diff
                                    )],
                                  columns=['scenario',
                                           'output',
                                           'mean_95%CI_value_for_int_period',
                                           'skew_for_int_data',
                                           'mean_95%CI_diff_outcome_int_period',
                                           'skew_for_diff_data',
                                           'median_diff_outcome_int_period'])

            output_df = output_df.append(res_df)

        return output_df

    def save_outputs(folder, keys, save_name, save_folder):
        dfs = []
        for k in scenario_names:
            scen_df = get_diff_between_runs(folder, scenario_names[0], k, keys, intervention_years, output_df)
            dfs.append(scen_df)

        final_df = pd.concat(dfs)
        final_df.to_csv(f'{save_folder}/{save_name}.csv')

    # Get denominator and complications folders
    births_dict = analysis_utility_functions.return_birth_data_from_multiple_scenarios(
        results_folders,  sim_years, intervention_years)

    preg_dict = analysis_utility_functions.return_pregnancy_data_from_multiple_scenarios(results_folders,
                                                                                         sim_years, intervention_years)

    comps_dfs = {k: analysis_utility_functions.get_modules_maternal_complication_dataframes(results_folders[k])
                 for k in results_folders}

    neo_comps_dfs = {k: analysis_utility_functions.get_modules_neonatal_complication_dataframes(results_folders[k])
                     for k in results_folders}

    comp_pregs_dict = {k: analysis_utility_functions.get_completed_pregnancies_from_multiple_scenarios(
        comps_dfs[k], births_dict[k], results_folders[k], sim_years, intervention_years) for k in results_folders}

    # --------------------------------------- DENOMINATOR OUTCOMES --------------------------------------------------
    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, preg_dict, 'total_preg',
        'Total Pregnancies',
        'Total Number of Pregnancies Per Year By Scenario',
        primary_oc_path, 'preg')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, births_dict, 'total_births',
        'Total Births',
        'Total Number of Births Per Year By Scenario',
        primary_oc_path, 'births')

    def bar_chart_from_dict(dict, y_title, title,plot_destination_folder, file_name):
        labels = dict.keys()
        mean_vals = list()
        lq_vals = list()
        uq_vals = list()
        for k in dict:
            mean_vals.append(dict[k][file_name][0])
            lq_vals.append(dict[k][file_name][1])
            uq_vals.append(dict[k][file_name][2])

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

    bar_chart_from_dict(preg_dict, 'Pregnancies', 'Total Pregnancies by Scenario', primary_oc_path, 'agg_preg')
    bar_chart_from_dict(births_dict, 'Births', 'Total Births by Scenario', primary_oc_path, 'agg_births')

    # ------------------------------------ PRIMARY OUTCOMES... -------------------------------------------------------
    def extract_death_and_stillbirth_data_frames_and_summ_outcomes(folder, birth_df):
        # MATERNAL

        direct_deaths = extract_results(
            folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=(
                lambda df: df.loc[(df['label'] == 'Maternal Disorders')].assign(
                    year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True)
        direct_deaths_final = direct_deaths.fillna(0)

        indirect_deaths_non_hiv = extract_results(
            folder,
            module="tlo.methods.demography.detail",
            key="properties_of_deceased_persons",
            custom_generate_series=(
                lambda df: df.loc[(df['is_pregnant'] | df['la_is_postpartum']) &
                                  (df['cause_of_death'].str.contains('Malaria|Suicide|ever_stroke|diabetes|'
                                                                     'chronic_ischemic_hd|ever_heart_attack|'
                                                                     'chronic_kidney_disease') |
                                   (df['cause_of_death'] == 'TB'))].assign(
                    year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True)
        indirect_deaths_non_hiv_final = indirect_deaths_non_hiv.fillna(0)

        hiv_pd = extract_results(
            folder,
            module="tlo.methods.demography.detail",
            key="properties_of_deceased_persons",
            custom_generate_series=(
                lambda df: df.loc[(df['is_pregnant'] | df['la_is_postpartum']) &
                                  (df['cause_of_death'].str.contains('AIDS_non_TB|AIDS_TB'))].assign(
                    year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True)

        hiv_pd_fill = hiv_pd.fillna(0)
        hiv_indirect = hiv_pd_fill * 0.3
        hiv_indirect_maternal_deaths = hiv_indirect.round(0)

        indirect_deaths_final = indirect_deaths_non_hiv_final + hiv_indirect_maternal_deaths

        total_deaths = direct_deaths_final + indirect_deaths_final


        mmr = (total_deaths / birth_df) * 100_000
        total_mmr_by_year = return_95_CI_across_runs(mmr, sim_years)

        # TOTAL AVERAGE MMR DURING INTERVENTION
        mmr_df_int = mmr.loc[intervention_years[0]: intervention_years[-1]]
        mean_mmr_by_year_int = get_mean_from_columns(mmr_df_int, 'avg')
        total_mmr_aggregated = get_mean_95_CI_from_list(mean_mmr_by_year_int)

        # TOTAL MATERNAL DEATHS BY YEAR
        td_int = total_deaths.loc[intervention_years[0]: intervention_years[-1]]
        mean_total_deaths_by_year = return_95_CI_across_runs(total_deaths, sim_years)

        # TOTAL MATERNAL DEATHS DURING INTERVENTION PERIOD
        sum_mat_death_int_by_run = get_mean_from_columns(td_int, 'sum')
        total_deaths_by_scenario = get_mean_95_CI_from_list(sum_mat_death_int_by_run)

        # DIRECT MATERNAL DEATHS PER YEAR
        mean_direct_deaths_by_year = return_95_CI_across_runs(direct_deaths_final, sim_years)
        mean_direct_deaths_by_year_int = return_95_CI_across_runs(direct_deaths_final, intervention_years)

        # TOTAL DIRECT MATERNAL DEATHS DURING INTERVENTION
        dd_int = direct_deaths_final.loc[intervention_years[0]: intervention_years[-1]]
        sum_d_mat_death_int_by_run = get_mean_from_columns(dd_int, 'sum')
        total_direct_deaths_by_scenario = get_mean_95_CI_from_list(sum_d_mat_death_int_by_run)

        # DIRECT MMR BY YEAR
        d_mmr_df = (direct_deaths / birth_df) * 100_000
        total_direct_mmr_by_year = return_95_CI_across_runs(d_mmr_df, sim_years)

        # AVERAGE DIRECT MMR DURING INTERVENTION
        d_mmr_df_int = d_mmr_df.loc[intervention_years[0]: intervention_years[-1]]
        mean_d_mmr_by_year_int = get_mean_from_columns(d_mmr_df_int, 'avg')
        total_direct_mmr_aggregated = get_mean_95_CI_from_list(mean_d_mmr_by_year_int)

        # INDIRECT MATERNAL DEATHS PER YEAR
        mean_indirect_deaths_by_year = return_95_CI_across_runs(indirect_deaths_final, sim_years)
        mean_indirect_deaths_by_year_int = return_95_CI_across_runs(indirect_deaths_final, intervention_years)

        # TOTAL INDIRECT MATERNAL DEATHS DURING INTERVENTION
        in_int = indirect_deaths_final.loc[intervention_years[0]: intervention_years[-1]]
        sum_in_mat_death_int_by_run = get_mean_from_columns(in_int, 'sum')
        total_indirect_deaths_by_scenario = get_mean_95_CI_from_list(sum_in_mat_death_int_by_run)

        # INDIRECT MMR BY YEAR
        in_mmr_df = (indirect_deaths_final / birth_df) * 100_000
        total_indirect_mmr_by_year = return_95_CI_across_runs(in_mmr_df, sim_years)

        # AVERAGE INDIRECT MMR DURING INTERVENTION
        in_mmr_df_int = in_mmr_df.loc[intervention_years[0]: intervention_years[-1]]
        mean_in_mmr_by_year_int = get_mean_from_columns(in_mmr_df_int, 'avg')
        total_indirect_mmr_aggregated = get_mean_95_CI_from_list(mean_in_mmr_by_year_int)

        # NEONATAL
        nd = extract_results(
            folder,
            module="tlo.methods.demography.detail",
            key="properties_of_deceased_persons",
            custom_generate_series=(
                lambda df: df.loc[(df['age_days'] < 29)].assign(
                    year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True)
        neo_deaths = nd.fillna(0)
        neo_deaths_int = neo_deaths.loc[intervention_years[0]: intervention_years[-1]]

        nmr = (neo_deaths / birth_df) * 1000
        total_nmr_by_year = return_95_CI_across_runs(nmr, sim_years)

        # AVERAGE NMR DURING INTERVENTION PERIOD
        nmr_df = nmr.loc[intervention_years[0]: intervention_years[-1]]
        mean_nmr_by_year_int = get_mean_from_columns(nmr_df, 'avg')
        total_nmr_aggregated = get_mean_95_CI_from_list(mean_nmr_by_year_int)

        # NEONATAL DEATHS PER YEAR
        mean_neonatal_deaths_by_year = return_95_CI_across_runs(neo_deaths, sim_years)
        mean_neonatal_deaths_by_year_int = return_95_CI_across_runs(neo_deaths_int, intervention_years)

        # TOTAL NEONATAL DEATHS DURING INTERVENTION PERIOD
        sum_neo_death_int_by_run = get_mean_from_columns(neo_deaths_int, 'sum')
        total_neonatal_deaths_by_scenario = get_mean_95_CI_from_list(sum_neo_death_int_by_run)

        # STILLBIRTH
        an_stillbirth_results = extract_results(
            folder,
            module="tlo.methods.pregnancy_supervisor",
            key="antenatal_stillbirth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
        )
        an_stillbirth_results = an_stillbirth_results.fillna(0)

        ip_stillbirth_results = extract_results(
            folder,
            module="tlo.methods.labour",
            key="intrapartum_stillbirth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
        )
        ip_stillbirth_results = ip_stillbirth_results.fillna(0)
        all_sb = an_stillbirth_results + ip_stillbirth_results

        an_sbr_df = (an_stillbirth_results / birth_df) * 1000
        ip_sbr_df = (ip_stillbirth_results / birth_df) * 1000
        sbr_df = (all_sb / birth_df) * 1000

        all_sb = an_stillbirth_results + ip_stillbirth_results
        all_sb_int = all_sb.loc[intervention_years[0]: intervention_years[-1]]
        # Store mean number of stillbirths, LQ, UQ
        crude_sb = return_95_CI_across_runs(all_sb, sim_years)

        def get_sbr(df):
            sbr_df = (df / birth_df) * 1000
            sbr_mean_quants = return_95_CI_across_runs(sbr_df, sim_years)
            return sbr_mean_quants

        an_sbr = get_sbr(an_stillbirth_results)
        ip_sbr = get_sbr(ip_stillbirth_results)
        total_sbr = get_sbr(all_sb)

        avg_sbr_df = (all_sb_int / birth_df.loc[intervention_years[0]:intervention_years[-1]]) * 1000
        avg_sbr_means = get_mean_from_columns(avg_sbr_df, 'avg')

        avg_isbr_df = (ip_stillbirth_results.loc[intervention_years[0]:intervention_years[-1]] /
                       birth_df.loc[intervention_years[0]:intervention_years[-1]]) * 1000
        ip_sbr_means = get_mean_from_columns(avg_isbr_df, 'avg')

        avg_asbr_df = (an_stillbirth_results.loc[intervention_years[0]:intervention_years[-1]] /
                       birth_df.loc[intervention_years[0]:intervention_years[-1]]) * 1000
        ap_sbr_means = get_mean_from_columns(avg_asbr_df, 'avg')

        avg_sbr = get_mean_95_CI_from_list(avg_sbr_means)
        avg_i_sbr = get_mean_95_CI_from_list(ip_sbr_means)
        avg_a_sbr = get_mean_95_CI_from_list(ap_sbr_means)

        return {'mmr_df': mmr,
                'mat_deaths_total_df': total_deaths,
                'nmr_df': nmr,
                'neo_deaths_total_df': neo_deaths,
                'sbr_df': sbr_df,
                'ip_sbr_df': ip_sbr_df,
                'an_sbr_df': an_sbr_df,
                'stillbirths_total_df': all_sb,

                'crude_t_deaths': mean_total_deaths_by_year,
                'agg_total': total_deaths_by_scenario,
                'total_mmr': total_mmr_by_year,
                'agg_total_mr': total_mmr_aggregated,

                'crude_dir_m_deaths': mean_direct_deaths_by_year,
                'agg_dir_m_deaths': total_direct_deaths_by_scenario,
                'direct_mmr': total_direct_mmr_by_year,
                'agg_dir_mr': total_direct_mmr_aggregated,

                'crude_ind_m_deaths': mean_indirect_deaths_by_year,
                'agg_ind_m_deaths': total_indirect_deaths_by_scenario,
                'indirect_mmr': total_indirect_mmr_by_year,
                'agg_ind_mr': total_indirect_mmr_aggregated,

                'crude_n_deaths': mean_neonatal_deaths_by_year,
                'agg_n_deaths': total_neonatal_deaths_by_scenario,
                'nmr': total_nmr_by_year,
                'agg_nmr': total_nmr_aggregated,

                'an_sbr': an_sbr,
                'ip_sbr': ip_sbr,
                'sbr': total_sbr,
                'crude_sb': crude_sb,
                'avg_sbr': avg_sbr,
                'avg_i_sbr': avg_i_sbr,
                'avg_a_sbr': avg_a_sbr}

    # Extract data from scenarios
    death_data = {k: extract_death_and_stillbirth_data_frames_and_summ_outcomes(results_folders[k],
                                                                                births_dict[k]['births_data_frame'])
                     for k in results_folders}

    #  ---------------- MMR/NMR GRAPHS ---------------
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
        plot_agg_graph(death_data, data, y_lable, title, data, primary_oc_path)

    # 2.) TRENDS IN DEATHS
    # Output and save the relevant graphs
    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, death_data, 'direct_mmr',
        'Deaths per 100,000 live births',
        'MMR per Year at Baseline and Under Intervention (Direct only)', primary_oc_path,
        'maternal_mr_direct')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours,sim_years, death_data, 'total_mmr',
        'Deaths per 100,000 live births',
        'MMR per Year at Baseline and Under Intervention (Total)',
        primary_oc_path, 'maternal_mr_total')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, death_data, 'nmr',
        'Total Deaths per 1000 live births',
        'Neonatal Mortality Ratio per Year at Baseline and Under Intervention',
        primary_oc_path, 'neonatal_mr_int')

    for group, l in zip(['Maternal', 'Neonatal'], ['dir_m', 'n']):
        analysis_utility_functions.comparison_bar_chart_multiple_bars(
            death_data, f'crude_{l}_deaths', sim_years, scen_colours,
            f'Total {group} Deaths (scaled)', f'Yearly Baseline {group} Deaths Compared to Intervention',
            primary_oc_path, f'{group}_crude_deaths_comparison.png')

    def extract_deaths_by_cause(results_folder, births_df, intervention_years):

        d_causes = ['ectopic_pregnancy', 'spontaneous_abortion', 'induced_abortion', 'severe_gestational_hypertension',
                    'severe_pre_eclampsia', 'eclampsia', 'antenatal_sepsis', 'uterine_rupture',
                    'intrapartum_sepsis', 'postpartum_sepsis','postpartum_haemorrhage',
                    'secondary_postpartum_haemorrhage','antepartum_haemorrhage']

        ind_causes = ['AIDS_non_TB', 'AIDS_TB', 'TB', 'Malaria', 'Suicide', 'ever_stroke', 'diabetes',
                      'chronic_ischemic_hd', 'ever_heart_attack',
                      'chronic_kidney_disease']

        def update_dfs_to_replace_missing_causes(df, causes):
            t = []
            for year in sim_years:
                for cause in causes:
                    if cause not in df.loc[year].index:
                        index = pd.MultiIndex.from_tuples([(year, cause)], names=["year", "cause_of_death"])
                        new_row = pd.DataFrame(columns=df.columns, index=index)
                        f_df = new_row.fillna(0.0)
                        t.append(f_df)

            causes_df = pd.concat(t)
            updated_df = df.append(causes_df)

            return updated_df

        dd = extract_results(
            results_folder,
            module="tlo.methods.demography.detail",
            key="properties_of_deceased_persons",
            custom_generate_series=(
                lambda df: df.assign(
                    year=df['date'].dt.year).groupby(['year', 'cause_of_death'])['year'].count()),
            do_scaling=True)
        direct_deaths = dd.fillna(0)

        id = extract_results(
            results_folder,
            module="tlo.methods.demography.detail",
            key="properties_of_deceased_persons",
            custom_generate_series=(
                lambda df: df.loc[(df['is_pregnant'] | df['la_is_postpartum'])].assign(
                    year=df['date'].dt.year).groupby(['year', 'cause_of_death'])['year'].count()),
            do_scaling=True)
        indirect_deaths = id.fillna(0)

        updated_dd = update_dfs_to_replace_missing_causes(direct_deaths, d_causes)
        updated_ind = update_dfs_to_replace_missing_causes(indirect_deaths, ind_causes)

        results = dict()

        def extract_mmr_data(cause, df):
            death_df = df.loc[(slice(None), cause), slice(None)].droplevel(1)

            if 'AIDS' in cause:
                death_df = death_df * 0.3

            mmr_df = (death_df / births_df) * 100_000
            results.update({f'{cause}_mmr_df': mmr_df})
            mmr_df_int = mmr_df.loc[intervention_years[0]:intervention_years[-1]]
            list_mmr = get_mean_from_columns(mmr_df_int, 'avg')
            results.update({f'{cause}_mmr_avg': get_mean_95_CI_from_list(list_mmr)})

        for cause in d_causes:
            extract_mmr_data(cause, updated_dd)

        for cause in ind_causes:
            extract_mmr_data(cause, updated_ind)

        return results

    def extract_neonatal_deaths_by_cause(results_folder, births_df, intervention_years):

        nd = extract_results(
            results_folder,
            module="tlo.methods.demography.detail",
            key="properties_of_deceased_persons",
            custom_generate_series=(
                lambda df: df.loc[(df['age_days'] < 29)].assign(
                    year=df['date'].dt.year).groupby(['year', 'cause_of_death'])['year'].count()),
            do_scaling=True)
        neo_deaths = nd.fillna(0)

        results = dict()
        n_causes = list(neo_deaths.loc[2010].index)

        for cause in n_causes:
            death_df = neo_deaths.loc[(slice(None), cause), slice(None)].droplevel(1)
            nmr_df = (death_df / births_df) * 1000
            nmr_df_final = nmr_df.fillna(0)
            results.update({f'{cause}_nmr_df': nmr_df_final})
            nmr_df_int = nmr_df_final.loc[intervention_years[0]:intervention_years[-1]]
            list_nmr = get_mean_from_columns(nmr_df_int, 'avg')
            results.update({f'{cause}_nmr_avg': get_mean_95_CI_from_list(list_nmr)})

        return results

    cod_data = {k: extract_deaths_by_cause(results_folders[k], births_dict[k]['births_data_frame'],
                                                intervention_years) for k in results_folders}

    cod_neo_data = {k: extract_neonatal_deaths_by_cause(results_folders[k], births_dict[k]['births_data_frame'],
                                                        intervention_years) for k in results_folders}
    def save_mr_by_cause_data_and_output_graphs(group, cause_d):
        if group == 'mat':
            d = ['m', 'MMR']
        else:
            d = ['n', 'NMR']

        cod_keys = list()
        for k in cause_d[scenario_titles[0]].keys():
            if 'df' in k:
                cod_keys.append(k)

        save_outputs(cause_d, cod_keys, f'diff_in_cause_specific_{d[0]}mr', primary_oc_path)

        labels = [l.replace(f'_{d[0]}mr_df', '') for l in cod_keys]

        for k, colour in zip(cause_d, scen_colours):
            mean_vals = list()
            lq_vals = list()
            uq_vals = list()
            for key in cause_d[k]:
                if 'avg' in key:
                    mean_vals.append(cause_d[k][key][0])
                    lq_vals.append(cause_d[k][key][1])
                    uq_vals.append(cause_d[k][key][2])

            width = 0.55  # the width of the bars: can also be len(x) sequence
            fig, ax = plt.subplots()

            ci = [(x - y) / 2 for x, y in zip(uq_vals, lq_vals)]
            ax.bar(labels, mean_vals, color=colour, width=width, yerr=ci)
            ax.tick_params(axis='x', which='major', labelsize=8, labelrotation=90)
            if group == 'mat':
                plt.gca().set_ylim(bottom=0, top=70)
            else:
                plt.gca().set_ylim(bottom=0, top=7)

            ax.set_ylabel(d[1])
            ax.set_xlabel('Complication')
            ax.set_title(f'Cause specific {d[1]} for {k} scenario')
            plt.savefig(f'{primary_oc_path}/{k}_{d[0]}mr_by_cause.png', bbox_inches='tight')
            plt.show()
    save_mr_by_cause_data_and_output_graphs('mat', cod_data)
    save_mr_by_cause_data_and_output_graphs('neo', cod_neo_data)

    def get_cause_spef_mmrs_on_same_graph(group, cod_data):
        csmmr_dict = dict()
        for k in cod_data:
            csmmr_dict.update({k: []})
            mean_vals = list()
            lq_vals = list()
            uq_vals = list()
            for key in cod_data[k]:
                if 'avg' in key:
                    mean_vals.append(cod_data[k][key][0])
                    lq_vals.append(cod_data[k][key][1])
                    uq_vals.append(cod_data[k][key][2])
            csmmr_dict[k] = [mean_vals, lq_vals, uq_vals]
        N = len(csmmr_dict['Status Quo'][0])
        ind = np.arange(N)
        if len(csmmr_dict.keys()) > 3:
            width = 0.15
        else:
            width = 0.35
        x_ticks = list()
        for x in range(len(csmmr_dict['Status Quo'][0])):
            x_ticks.append(x)
        for k, position, colour in zip(csmmr_dict, [ind - width, ind, ind + width, ind + width * 2, ind + width * 3],
                                       scen_colours):
            ci = [(x - y) / 2 for x, y in zip(csmmr_dict[k][2], csmmr_dict[k][1])]
            plt.bar(position, csmmr_dict[k][0], width, label=k, yerr=ci, color=colour)
        cod_keys = list()
        for k in cod_data[scenario_titles[0]].keys():
            if 'df' in k:
                cod_keys.append(k)
        labels = [l.replace('_mmr_df', '') for l in cod_keys]
        plt.gca().set_ylim(bottom=0, top=45)
        plt.ylabel('Average deaths per 100,000 live births')
        plt.xlabel('Cause of death')
        plt.title('Cause specific MMR by scenario')
        plt.legend(loc='best')
        plt.xticks(x_ticks, labels=labels, rotation=90, size=7)
        plt.savefig(f'{primary_oc_path}/cs_mmr_one_graph.png', bbox_inches='tight')
        plt.show()
        pass

    #  ---------------- STILLBIRTH GRAPHS ---------------
    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, death_data, 'an_sbr',
        'Antenatal stillbirths per 1000 births',
        'Antenatal stillbirth Rate per Year at Baseline and Under Intervention',
        primary_oc_path, 'an_sbr_int')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, death_data, 'ip_sbr',
        'Intrapartum stillbirths per 1000 births',
        'Intrapartum stillbirth Rate per Year at Baseline and Under Intervention',
        primary_oc_path, 'ip_sbr_int')

    # Output SBR per year for scenario vs intervention
    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, death_data, 'sbr',
        'Stillbirths per 1000 births',
        'Stillbirth Rate per Year at Baseline and Under Intervention',
        primary_oc_path, 'sbr_int')

    analysis_utility_functions.comparison_bar_chart_multiple_bars(
        death_data, 'crude_sb', sim_years, scen_colours,
        'Total Stillbirths (scaled)', 'Yearly Baseline Stillbirths Compared to Intervention',
        primary_oc_path, 'crude_stillbirths_comparison.png')

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
        plt.savefig(f'{primary_oc_path}/{data}.png')
        plt.show()

    keys = ['mmr_df', 'nmr_df', 'sbr_df', 'an_sbr_df', 'ip_sbr_df', 'mat_deaths_total_df', 'neo_deaths_total_df',
            'stillbirths_total_df']
    save_outputs(death_data, keys, 'diff_in_mortality_outcomes', primary_oc_path)

    def extract_dalys(folder):
        results_dict = dict()

        # Get DALY df
        dalys = extract_results(
            folder,
            module="tlo.methods.healthburden",
            key="dalys",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year']).sum().stack()),
            do_scaling=True)

        dalys_stacked = extract_results(
            folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year']).sum().stack()),
            do_scaling=True)

        # todo - should this just be at risk or total (gbd suggests total, which this calibrates well with)
        person_years_total = extract_results(
            folder,
            module="tlo.methods.demography",
            key="person_years",
            custom_generate_series=(
                lambda df: df.assign(total=(df['M'].apply(lambda x: sum(x.values()))) + df['F'].apply(
                    lambda x: sum(x.values()))).assign(
                    year=df['date'].dt.year).groupby(['year'])['total'].sum()),
            do_scaling=True)

        for type, d in zip(['stacked', 'unstacked'], [dalys_stacked, dalys]):
            md = d.loc[(slice(None), 'Maternal Disorders'), slice(None)].droplevel(1)
            nd = d.loc[(slice(None), 'Neonatal Disorders'), slice(None)].droplevel(1)

            results_dict.update({f'maternal_total_dalys_df_{type}': md})
            results_dict.update({f'neonatal_total_dalys_df_{type}': nd})

            m_d_rate_df = (md / person_years_total) * 100_000
            n_d_rate_df = (nd / person_years_total) * 100_000

            results_dict.update({f'maternal_dalys_rate_df_{type}': m_d_rate_df})
            results_dict.update({f'neonatal_dalys_rate_df_{type}': n_d_rate_df})

            results_dict.update({f'maternal_dalys_rate_{type}': return_95_CI_across_runs(m_d_rate_df, sim_years)})
            results_dict.update({f'neonatal_dalys_rate_{type}': return_95_CI_across_runs(n_d_rate_df, sim_years)})

            m_int_period = m_d_rate_df.loc[intervention_years[0]: intervention_years[-1]]
            n_int_period = n_d_rate_df.loc[intervention_years[0]: intervention_years[-1]]

            m_int_means = get_mean_from_columns(m_int_period, 'avg')
            n_int_means = get_mean_from_columns(n_int_period, 'avg')

            results_dict.update({f'avg_mat_dalys_rate_{type}':  get_mean_95_CI_from_list(m_int_means)})
            results_dict.update({f'avg_neo_dalys_rate_{type}':  get_mean_95_CI_from_list(n_int_means)})

            # Get averages/sums
            results_dict.update({f'maternal_dalys_crude_{type}': return_95_CI_across_runs(md, sim_years)})
            results_dict.update({f'neonatal_dalys_crude_{type}': return_95_CI_across_runs(nd, sim_years)})

            m_int_agg = get_mean_from_columns(m_int_period, 'sum')
            n_int_agg = get_mean_from_columns(n_int_period, 'sum')

            results_dict.update({f'agg_mat_dalys_{type}': get_mean_95_CI_from_list(m_int_agg)})
            results_dict.update({f'agg_neo_dalys_{type}': get_mean_95_CI_from_list(n_int_agg)})

        mat_causes_death = ['ectopic_pregnancy',
                            'spontaneous_abortion',
                            'induced_abortion',
                            'severe_gestational_hypertension',
                            'severe_pre_eclampsia',
                            'eclampsia',
                            'antenatal_sepsis',
                            'uterine_rupture',
                            'intrapartum_sepsis',
                            'postpartum_sepsis',
                            'postpartum_haemorrhage',
                            'secondary_postpartum_haemorrhage',
                            'antepartum_haemorrhage']

        mat_causes_disab = ['maternal']

        neo_causes_death = ['early_onset_sepsis', 'late_onset_sepsis', 'encephalopathy', 'preterm_other',
                            'respiratory_distress_syndrome', 'neonatal_respiratory_depression']

        neo_causes_disab = ['Retinopathy of Prematurity', 'Neonatal Encephalopathy',
                            'Neonatal Sepsis Long term Disability', 'Preterm Birth Disability']

        yll = extract_results(
            folder,
            module="tlo.methods.healthburden",
            key="yll_by_causes_of_death",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year']).sum().stack()),
            do_scaling=True)
        yll_final = yll.fillna(0)

        yll_stacked = extract_results(
            folder,
            module="tlo.methods.healthburden",
            key="yll_by_causes_of_death_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year']).sum().stack()),
            do_scaling=True)
        yll_stacked_final = yll_stacked.fillna(0)

        yld = extract_results(
            folder,
            module="tlo.methods.healthburden",
            key="yld_by_causes_of_disability",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year']).sum().stack()),
            do_scaling=True)
        yld_final = yld.fillna(0)

        def get_total_dfs(df, causes):
            dfs = []
            for k in causes:
                scen_df = df.loc[(slice(None), k), slice(None)].droplevel(1)
                dfs.append(scen_df)

            final_df = sum(dfs)
            return final_df

        neo_yll_df = get_total_dfs(yll_final, neo_causes_death)
        mat_yll_df = get_total_dfs(yll_final, mat_causes_death)
        neo_yll_s_df = get_total_dfs(yll_stacked_final, neo_causes_death)
        mat_yll_s_df = get_total_dfs(yll_stacked_final, mat_causes_death)
        neo_yld_df = get_total_dfs(yld_final, neo_causes_disab)
        mat_yld_df = get_total_dfs(yld_final, mat_causes_disab)

        results_dict.update({'maternal_yll_crude_unstacked': return_95_CI_across_runs(mat_yll_df, sim_years)})
        results_dict.update({'maternal_yll_crude_stacked': return_95_CI_across_runs(mat_yll_s_df, sim_years)})

        mat_yll_df_rate = (mat_yll_df / person_years_total) * 100_000
        mat_yll_s_df_rate = (mat_yll_s_df / person_years_total) * 100_000

        results_dict.update({'maternal_yll_rate_unstacked': return_95_CI_across_runs(mat_yll_df_rate, sim_years)})
        results_dict.update({'maternal_yll_rate_stacked': return_95_CI_across_runs(mat_yll_s_df_rate, sim_years)})

        results_dict.update({'maternal_yld_crude_unstacked': return_95_CI_across_runs(mat_yld_df, sim_years)})

        mat_yld_df_rate = (mat_yld_df / person_years_total) * 100_000
        results_dict.update({'maternal_yld_rate_unstacked': return_95_CI_across_runs(mat_yld_df_rate, sim_years)})

        results_dict.update({'neonatal_yll_crude_unstacked': return_95_CI_across_runs(neo_yll_df, sim_years)})
        results_dict.update({'neonatal_yll_crude_stacked': return_95_CI_across_runs(neo_yll_s_df, sim_years)})

        neo_yll_df_rate = (neo_yll_df / person_years_total) * 100_000
        neo_yll_s_df_rate = (neo_yll_s_df / person_years_total) * 100_000

        results_dict.update({'neonatal_yll_rate_unstacked': return_95_CI_across_runs(neo_yll_df_rate, sim_years)})
        results_dict.update({'neonatal_yll_rate_stacked': return_95_CI_across_runs(neo_yll_s_df_rate, sim_years)})

        results_dict.update({'neonatal_yld_crude_unstacked': return_95_CI_across_runs(neo_yld_df, sim_years)})

        neo_yld_df_rate = (neo_yld_df / person_years_total) * 100_000
        results_dict.update({'neonatal_yld_rate_unstacked': return_95_CI_across_runs(neo_yld_df_rate, sim_years)})

        return results_dict

    dalys_folders = {k: extract_dalys(results_folders[k]) for k in results_folders}

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
        for k in dalys_folders:
            mean_vals.append(dalys_folders[k][data][0])
            lq_vals.append(dalys_folders[k][data][1])
            uq_vals.append(dalys_folders[k][data][2])

        width = 0.55  # the width of the bars: can also be len(x) sequence
        fig, ax = plt.subplots()

        ci = [(x - y) / 2 for x, y in zip(uq_vals, lq_vals)]
        ax.bar(labels, mean_vals, color=scen_colours, width=width, yerr=ci)
        ax.tick_params(axis='x', which='major', labelsize=8)

        ax.set_ylabel(y_lable)
        ax.set_xlabel('Scenario')
        ax.set_title(title)
        plt.savefig(f'{primary_oc_path}/{data}.png')
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
            scen_colours, sim_years, dalys_folders, dict_key, axis, title, primary_oc_path, save_name)

    keys = ['maternal_total_dalys_df_stacked', 'neonatal_total_dalys_df_stacked',
            'maternal_dalys_rate_df_stacked', 'neonatal_dalys_rate_df_stacked']
    save_outputs(dalys_folders, keys, 'diff_in_daly_outcomes', primary_oc_path)

    # ----------------------------------------- SECONDARY OUTCOMES --------------------------------------------------
    def get_coverage_of_key_maternity_services(folder, births_df):

        results = dict()

        # --- ANC ---
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
                lambda df: df.loc[df['total_anc'] >= 4].assign(
                    year=df['date'].dt.year).groupby(['year'])['person_id'].count()),
            do_scaling=True
        )
        anc_cov_of_interest = an.fillna(0)

        cd = (anc_cov_of_interest / anc_coverage) * 100
        coverage_df = cd.fillna(0)

        results.update({'anc_cov_df': coverage_df})
        results.update({'anc_cov_rate': return_95_CI_across_runs(coverage_df, sim_years)})

        # ---SBA--
        all_deliveries = extract_results(
            folder,
            module="tlo.methods.labour",
            key="delivery_setting_and_mode",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year'])[
                    'mother'].count()),
            do_scaling=True
        )

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
            rate_df = (deliver_setting_results / all_deliveries) * 100
            results.update({f'{facility_type}_df': rate_df})
            results.update({f'{facility_type}_rate': return_95_CI_across_runs(rate_df, sim_years)})

        # --- PNC ---
        all_surviving_mothers = extract_results(
            folder,
            module="tlo.methods.postnatal_supervisor",
            key="total_mat_pnc_visits",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['mother'].count()),
            do_scaling=True)

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

        results.update({'pnc_mat_cov_birth_df': cov_mat_birth_df})
        results.update({'pnc_mat_cov_birth_rate': return_95_CI_across_runs(cov_mat_birth_df, sim_years)})

        results.update({'pnc_mat_cov_surv_df': cov_mat_surv_df})
        results.update({'pnc_mat_cov_surv_rate': return_95_CI_across_runs(cov_mat_surv_df, sim_years)})

        results.update({'pnc_neo_cov_birth_df': cov_neo_birth_df})
        results.update({'pnc_neo_cov_birth_rate': return_95_CI_across_runs(cov_neo_birth_df, sim_years)})

        results.update({'pnc_neo_cov_surv_df': cov_neo_surv_df})
        results.update({'pnc_neo_cov_surv_rate': return_95_CI_across_runs(cov_mat_surv_df, sim_years)})

        return results

    cov_data = {k: get_coverage_of_key_maternity_services(results_folders[k], births_dict[k]['births_data_frame'])
                for k in results_folders}

    cov_keys = list()
    for k in cov_data[scenario_titles[0]].keys():
        if 'df' in k:
            cov_keys.append(k)

    save_outputs(cov_data, cov_keys, 'diff_in_mat_service_coverage', secondary_oc_path)

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, cov_data, 'anc_cov_rate',
        '% Total Births',
        'Proportion of women receiving four (or more) ANC visits at birth',
        secondary_oc_path, 'anc4_cov')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, cov_data, 'facility_rate',
        '% Total Births',
        'Facility Delivery Rate per Year Per Scenario',
        secondary_oc_path, 'fd_rate')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, cov_data, 'home_birth_rate',
        '% Total Births',
        'Home birth Rate per Year Per Scenario',
        secondary_oc_path, 'hb_rate')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, cov_data, 'hospital_rate',
        '% Total Births',
        'Hospital birth Rate per Year Per Scenario',
        secondary_oc_path, 'hp_rate')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, cov_data, 'health_centre_rate',
        '% Total Births',
        'Health Centre Birth Rate per Year Per Scenario',
        secondary_oc_path, 'hc_rate')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, cov_data, 'pnc_mat_cov_birth_rate',
        '% Total Births',
        'Maternal PNC Coverage as Proportion of Total Births',
        secondary_oc_path, 'mat_pnc_coverage_births')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, cov_data, 'pnc_mat_cov_surv_rate',
        '% Total Survivors at Day 42',
        'Maternal PNC Coverage as Proportion of Postnatal Survivors',
        secondary_oc_path, 'mat_pnc_coverage_survivors')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, cov_data, 'pnc_neo_cov_birth_rate',
        '% Total Births',
        'Neonatal PNC Coverage as Proportion of Total Births',
        secondary_oc_path, 'neo_pnc_coverage_births')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, cov_data, 'pnc_neo_cov_surv_rate',
        '% Total Survivors at Day 28',
        'Neonatal PNC Coverage as Proportion of Neonatal Survivors',
        secondary_oc_path, 'neo_pnc_coverage_survivors')

    # COMPLICATION INCIDCENE
    def extract_comp_inc_folders(folder, comps_df, neo_comps_df, pregnancy_df, births_df, comp_preg_df):
        def get_rate_df(comp_df, denom_df, denom):
            rate = (comp_df / denom_df) * denom
            return rate

        def get_avg_result(df):
            int_df = df.loc[intervention_years[0]:intervention_years[-1]]
            means = get_mean_from_columns(int_df, 'avg')
            return get_mean_95_CI_from_list(means)

        results = dict()

        # MATERNAL

        # TODO: FIX AS PER THIS EXAMPLE FOR ALL
        eu = comps_df['pregnancy_supervisor'].loc[(slice(None), 'ectopic_unruptured'), slice(None)].droplevel(1)
        eu_rate_df = get_rate_df(eu, pregnancy_df, 1000)
        results.update({'eu_rate_df': eu_rate_df})
        results.update({'eu_rate_per_year': return_95_CI_across_runs(eu_rate_df, sim_years)})
        results.update({'eu_rate_avg': get_avg_result(eu_rate_df)})

        sa = comps_df['pregnancy_supervisor'].loc[(slice(None), 'spontaneous_abortion'), slice(None)].droplevel(1)
        sa_rate_df = get_rate_df(sa, comp_preg_df, 1000)
        results.update({'sa_rate_df': sa_rate_df})
        results.update({'sa_rate_per_year': return_95_CI_across_runs(sa_rate_df, sim_years)})
        results.update({'sa_rate_avg': get_avg_result(sa_rate_df)})

        ia = comps_df['pregnancy_supervisor'].loc[(slice(None), 'induced_abortion'), slice(None)].droplevel(1)
        ia_rate_df = get_rate_df(ia, comp_preg_df, 1000)
        results.update({'ia_rate_df': ia_rate_df})
        results.update({'ia_rate_per_year': return_95_CI_across_runs(ia_rate_df, sim_years)})
        results.update({'ia_rate_avg': get_avg_result(ia_rate_df)})

        syph = comps_df['pregnancy_supervisor'].loc[(slice(None), 'syphilis'), slice(None)].droplevel(1)
        syph_rate_df = get_rate_df(syph, comp_preg_df, 1000)
        results.update({'syph_rate_df': syph_rate_df})
        results.update({'syph_rate_per_year': return_95_CI_across_runs(syph_rate_df, sim_years)})
        results.update({'syph_rate_avg': get_avg_result(syph_rate_df)})

        gdm = comps_df['pregnancy_supervisor'].loc[(slice(None), 'gest_diab'), slice(None)].droplevel(1)
        gdm_rate_df = get_rate_df(gdm, comp_preg_df, 1000)
        results.update({'gdm_rate_df': gdm_rate_df})
        results.update({'gdm_rate_per_year': return_95_CI_across_runs(gdm_rate_df, sim_years)})
        results.update({'gdm_rate_avg': get_avg_result(gdm_rate_df)})

        prom = comps_df['pregnancy_supervisor'].loc[(slice(None), 'PROM'), slice(None)].droplevel(1)
        prom_rate_df = get_rate_df(prom, comp_preg_df, 1000)
        results.update({'prom_rate_df': prom_rate_df})
        results.update({'prom_rate_per_year': return_95_CI_across_runs(prom_rate_df, sim_years)})
        results.update({'prom_rate_avg': get_avg_result(prom_rate_df)})

        praevia = comps_df['pregnancy_supervisor'].loc[(slice(None), 'placenta_praevia'), slice(None)].droplevel(1)
        pravia_rate_df = get_rate_df(praevia, pregnancy_df, 1000)
        results.update({'praevia_rate_df': get_rate_df(pravia_rate_df, pregnancy_df, 1000)})
        results.update({'praevia_rate_per_year': return_95_CI_across_runs(pravia_rate_df, sim_years)})
        results.update({'praevia_rate_avg': get_avg_result(pravia_rate_df)})

        abruption = comps_df['pregnancy_supervisor'].loc[(slice(None), 'placental_abruption'), slice(None)].droplevel(1)
        abruptio_rate_df = get_rate_df(abruption, births_df, 1000)
        results.update({'abruption_rate_df': abruptio_rate_df})
        results.update({'abruption_rate_per_year': return_95_CI_across_runs(abruptio_rate_df, sim_years)})
        results.update({'abruption_rate_avg': get_avg_result(abruptio_rate_df)})

        potl = comps_df['labour'].loc[(slice(None), 'post_term_labour'), slice(None)].droplevel(1)
        potl_rate_df = get_rate_df(potl, pregnancy_df, 100)
        results.update({'potl_rate_df': potl_rate_df})
        results.update({'potl_rate_per_year': return_95_CI_across_runs(potl_rate_df, sim_years)})
        results.update({'potl_rate_avg': get_avg_result(potl_rate_df)})

        ol = comps_df['labour'].loc[(slice(None), 'obstructed_labour'), slice(None)].droplevel(1)
        ol_rate_df = get_rate_df(ol, births_df, 1000)
        results.update({'ol_rate_df': ol_rate_df})
        results.update({'ol_rate_per_year': return_95_CI_across_runs(ol_rate_df, sim_years)})
        results.update({'ol_rate_avg': get_avg_result(ol_rate_df)})

        ur = comps_df['labour'].loc[(slice(None), 'uterine_rupture'), slice(None)].droplevel(1)
        ur_rate_df = get_rate_df(ur, births_df, 1000)
        results.update({'ur_rate_df': ur_rate_df})
        results.update({'ur_rate_per_year': return_95_CI_across_runs(ur_rate_df, sim_years)})
        results.update({'ur_rate_avg': get_avg_result(ur_rate_df)})

        gh_an = comps_df['pregnancy_supervisor'].loc[(slice(None), 'mild_gest_htn'), slice(None)].droplevel(1)
        gh_pn = comps_df['postnatal_supervisor'].loc[(slice(None), 'mild_gest_htn'), slice(None)].droplevel(1)
        gh_rate_df = get_rate_df((gh_an + gh_pn), births_df, 1000)
        results.update({'gh_rate_df': gh_rate_df})
        results.update({'gh_rate_per_year': return_95_CI_across_runs(gh_rate_df, sim_years)})
        results.update({'gh_rate_avg': get_avg_result(gh_rate_df)})

        mpe_an = comps_df['pregnancy_supervisor'].loc[(slice(None), 'mild_pre_eclamp'), slice(None)].droplevel(1)
        mpe_pn = comps_df['postnatal_supervisor'].loc[(slice(None), 'mild_pre_eclamp'), slice(None)].droplevel(1)
        mpe_rate_df = get_rate_df((mpe_an + mpe_pn), births_df, 1000)
        results.update({'mpe_rate_df': mpe_rate_df})
        results.update({'mpe_rate_per_year': return_95_CI_across_runs(mpe_rate_df, sim_years)})
        results.update({'mpe_rate_avg': get_avg_result(mpe_rate_df)})

        sgh_an = comps_df['pregnancy_supervisor'].loc[(slice(None), 'severe_gest_htn'), slice(None)].droplevel(1)
        sgh_la = comps_df['labour'].loc[(slice(None), 'severe_gest_htn'), slice(None)].droplevel(1)
        sgh_pn = comps_df['postnatal_supervisor'].loc[(slice(None), 'severe_gest_htn'), slice(None)].droplevel(1)
        sgh_rate_df = get_rate_df((sgh_an + sgh_la + sgh_pn), births_df, 1000)
        results.update({'sgh_rate_df': sgh_rate_df})
        results.update({'sgh_rate_per_year': return_95_CI_across_runs(sgh_rate_df, sim_years)})
        results.update({'sgh_rate_avg': get_avg_result(sgh_rate_df)})

        spe_an = comps_df['pregnancy_supervisor'].loc[(slice(None), 'severe_pre_eclamp'), slice(None)].droplevel(1)
        spe_la = comps_df['labour'].loc[(slice(None), 'severe_pre_eclamp'), slice(None)].droplevel(1)
        spe_pn = comps_df['postnatal_supervisor'].loc[(slice(None), 'severe_pre_eclamp'), slice(None)].droplevel(1)
        spe_rate_df = get_rate_df((spe_an + spe_la + spe_pn), births_df, 1000)
        results.update({'spe_rate_df': spe_rate_df})
        results.update({'spe_rate_per_year': return_95_CI_across_runs(spe_rate_df, sim_years)})
        results.update({'spe_rate_avg': get_avg_result(spe_rate_df)})

        ec_an = comps_df['pregnancy_supervisor'].loc[(slice(None), 'eclampsia'), slice(None)].droplevel(1)
        ec_la = comps_df['labour'].loc[(slice(None), 'eclampsia'), slice(None)].droplevel(1)
        ec_pn = comps_df['postnatal_supervisor'].loc[(slice(None), 'eclampsia'), slice(None)].droplevel(1)
        ec_rate_df = get_rate_df((ec_an + ec_la + ec_pn), births_df, 1000)
        results.update({'ec_rate_df': ec_rate_df})
        results.update({'ec_rate_per_year': return_95_CI_across_runs(ec_rate_df, sim_years)})
        results.update({'ec_rate_avg': get_avg_result(ec_rate_df)})

        m_aph_ps = comps_df['pregnancy_supervisor'].loc[(slice(None), 'mild_mod_antepartum_haemorrhage'
                                                         ), slice(None)].droplevel(1)
        m_aph_la = comps_df['labour'].loc[(slice(None), 'mild_mod_antepartum_haemorrhage'), slice(None)].droplevel(1)
        s_aph_ps = comps_df['pregnancy_supervisor'].loc[
            (slice(None), 'severe_antepartum_haemorrhage'), slice(None)].droplevel(1)
        s_aph_la = comps_df['labour'].loc[
            (slice(None), 'severe_antepartum_haemorrhage'), slice(None)].droplevel(1)

        aph_rate_df = get_rate_df((m_aph_ps + m_aph_la + s_aph_la + s_aph_ps), births_df, 1000)
        results.update({'aph_rate_df': aph_rate_df})
        results.update({'aph_rate_per_year': return_95_CI_across_runs(aph_rate_df, sim_years)})
        results.update({'aph_rate_avg': get_avg_result(aph_rate_df)})

        e_ptl = comps_df['labour'].loc[(slice(None), 'early_preterm_labour'), slice(None)].droplevel(1)
        l_ptl = comps_df['labour'].loc[(slice(None), 'late_preterm_labour'), slice(None)].droplevel(1)
        ptl_rate_df = get_rate_df((e_ptl + l_ptl), births_df, 100)
        results.update({'ptl_rate_df':ptl_rate_df })
        results.update({'ptl_rate_per_year': return_95_CI_across_runs(ptl_rate_df, sim_years)})
        results.update({'ptl_rate_avg': get_avg_result(ptl_rate_df)})

        an_sep = comps_df['pregnancy_supervisor'].loc[
            (slice(None), 'clinical_chorioamnionitis'), slice(None)].droplevel(1)
        la_sep = comps_df['labour'].loc[(slice(None), 'sepsis'), slice(None)].droplevel(1)
        pn_la_sep = comps_df['postnatal_supervisor'].loc[(slice(None), 'sepsis_postnatal'), slice(None)].droplevel(1)
        pn_sep = comps_df['postnatal_supervisor'].loc[(slice(None), 'sepsis'), slice(None)].droplevel(1)

        sep_rate_df = get_rate_df((an_sep + la_sep + pn_la_sep + pn_sep), births_df, 1000)
        results.update({'sep_rate_df': sep_rate_df})
        results.update({'sep_rate_per_year': return_95_CI_across_runs(sep_rate_df,sim_years)})
        results.update({'sep_rate_avg': get_avg_result(sep_rate_df)})

        an_sep_rate_df = get_rate_df(an_sep, births_df, 1000)
        results.update({'an_sep_rate_df': an_sep_rate_df})
        results.update({'an_sep_rate_per_year': return_95_CI_across_runs(an_sep_rate_df, sim_years)})
        results.update({'an_sep_rate_avg': get_avg_result(an_sep_rate_df)})

        la_sep_rate_df = get_rate_df(la_sep, births_df, 1000)
        results.update({'la_sep_rate_df': la_sep_rate_df})
        results.update({'la_sep_rate_per_year': return_95_CI_across_runs(la_sep_rate_df, sim_years)})
        results.update({'la_sep_rate_avg': get_avg_result(la_sep_rate_df)})

        pn_sep_rate_df = get_rate_df((pn_la_sep + pn_sep), births_df, 1000)
        results.update({'pn_sep_rate_df': pn_sep_rate_df})
        results.update({'pn_sep_rate_per_year': return_95_CI_across_runs(pn_sep_rate_df, sim_years)})
        results.update({'pn_sep_rate_avg': get_avg_result(pn_sep_rate_df)})

        l_pph = comps_df['postnatal_supervisor'].loc[(slice(None), 'primary_postpartum_haemorrhage'),
        slice(None)].droplevel(1)
        p_pph = comps_df['postnatal_supervisor'].loc[(slice(None), 'secondary_postpartum_haemorrhage'),
        slice(None)].droplevel(1)
        pph_rate_df = get_rate_df((l_pph + p_pph), births_df, 1000)
        results.update({'pph_rate_df': pph_rate_df})
        results.update({'pph_rate_per_year': return_95_CI_across_runs(pph_rate_df, sim_years)})
        results.update({'pph_rate_avg': get_avg_result(pph_rate_df)})

        p_pph_rate_df = get_rate_df(l_pph, births_df, 1000)
        results.update({'p_pph_rate_df': p_pph_rate_df})
        results.update({'p_pph_rate_per_year': return_95_CI_across_runs(p_pph_rate_df, sim_years)})
        results.update({'p_pph_rate_avg': get_avg_result(p_pph_rate_df)})

        s_pph_rate_df = get_rate_df(p_pph, births_df, 1000)
        results.update({'s_pph_rate_df': s_pph_rate_df})
        results.update({'s_pph_rate_per_year': return_95_CI_across_runs(s_pph_rate_df, sim_years)})
        results.update({'s_pph_rate_avg': get_avg_result(s_pph_rate_df)})

        anaemia_results = extract_results(
            folder,
            module="tlo.methods.pregnancy_supervisor",
            key="conditions_on_birth",
            custom_generate_series=(
                lambda df: df.loc[df['anaemia_status'] != 'none'].assign(year=df['date'].dt.year
                                                                         ).groupby(['year'])['year'].count()),
            do_scaling=True
        )
        an_ps_prev_df = get_rate_df(anaemia_results, births_df, 100)
        results.update({'an_ps_prev_df': an_ps_prev_df})
        results.update({'an_ps_prev_per_year': return_95_CI_across_runs(an_ps_prev_df, sim_years)})
        results.update({'an_ps_prev_avg': get_avg_result(an_ps_prev_df)})

        pnc_anaemia = extract_results(
            folder,
            module="tlo.methods.postnatal_supervisor",
            key="total_mat_pnc_visits",
            custom_generate_series=(
                lambda df_: df_.loc[df_['anaemia'] != 'none'].assign(year=df_['date'].dt.year
                                                                     ).groupby(['year'])['mother'].count()),
            do_scaling=True
        )
        an_pn_prev_df = get_rate_df(pnc_anaemia, births_df, 100)
        results.update({'an_pn_prev_df': an_pn_prev_df})
        results.update({'an_pn_prev_per_year': return_95_CI_across_runs(an_pn_prev_df, sim_years)})
        results.update({'an_pn_prev_avg': get_avg_result(an_pn_prev_df)})

        # NEWBORNS
        macro = neo_comps_df['newborn_outcomes'].loc[(slice(None), 'macrosomia'), slice(None)].droplevel(1)
        macro_df = get_rate_df(macro, births_df, 100)
        results.update({'macro_rate_df': macro_df})
        results.update({'macro_rate_per_year': return_95_CI_across_runs(macro_df, sim_years)})
        results.update({'macro_rate_avg': get_avg_result(macro_df)})

        sga = neo_comps_df['newborn_outcomes'].loc[(slice(None), 'small_for_gestational_age'), slice(None)].droplevel(1)
        sga_df = get_rate_df(sga, births_df, 100)
        results.update({'sga_rate_df': sga_df})
        results.update({'sga_rate_per_year': return_95_CI_across_runs(sga_df, sim_years)})
        results.update({'sga_rate_avg': get_avg_result(sga_df)})

        lbw = neo_comps_df['newborn_outcomes'].loc[(slice(None), 'low_birth_weight'), slice(None)].droplevel(1)
        lbw_df = get_rate_df(lbw, births_df, 100)
        results.update({'lbw_rate_df': lbw_df})
        results.update({'lbw_rate_per_year': return_95_CI_across_runs(lbw_df, sim_years)})
        results.update({'lbw_rate_avg': get_avg_result(lbw_df)})

        resp_distress = neo_comps_df['newborn_outcomes'].loc[(slice(None), 'not_breathing_at_birth'),
        slice(None)].droplevel(1)
        rd_df = get_rate_df(resp_distress, births_df, 1000)
        results.update({'rd_rate_df': rd_df})
        results.update({'rd_rate_per_year': return_95_CI_across_runs(rd_df, sim_years)})
        results.update({'rd_rate_avg': get_avg_result(rd_df)})

        rds = neo_comps_df['newborn_outcomes'].loc[(slice(None), 'respiratory_distress_syndrome'),
        slice(None)].droplevel(1)
        rds_df = get_rate_df(rds, births_df, 1000)
        results.update({'rds_rate_df': rds_df})
        results.update({'rds_rate_per_year': return_95_CI_across_runs(rds_df, sim_years)})
        results.update({'rds_rate_avg': get_avg_result(rds_df)})

        eons_n = neo_comps_df['newborn_outcomes'].loc[(slice(None), 'early_onset_sepsis'), slice(None)].droplevel(1)
        eons_pn = neo_comps_df['postnatal_supervisor'].loc[(slice(None), 'early_onset_sepsis'), slice(None)].droplevel(
            1)
        lons = neo_comps_df['postnatal_supervisor'].loc[(slice(None), 'late_onset_sepsis'), slice(None)].droplevel(1)
        neo_sep_df = get_rate_df((eons_n + eons_pn + lons), births_df, 1000)
        results.update({'neo_sep_rate_df': neo_sep_df})
        results.update({'neo_sep_rate_per_year': return_95_CI_across_runs(neo_sep_df, sim_years)})
        results.update({'neo_sep_rate_avg': get_avg_result(neo_sep_df)})

        m_enc = neo_comps_df['newborn_outcomes'].loc[(slice(None), 'mild_enceph'), slice(None)].droplevel(1)
        mo_enc = neo_comps_df['newborn_outcomes'].loc[(slice(None), 'moderate_enceph'), slice(None)].droplevel(1)
        s_enc = neo_comps_df['newborn_outcomes'].loc[(slice(None), 'severe_enceph'), slice(None)].droplevel(1)
        neo_enc_df = get_rate_df((m_enc + mo_enc + s_enc), births_df, 1000)
        results.update({'enc_rate_df': neo_enc_df})
        results.update({'enc_rate_per_year': return_95_CI_across_runs(neo_enc_df, sim_years)})
        results.update({'enc_rate_avg': get_avg_result(neo_enc_df)})

        return results

    comp_inc_folders = {k: extract_comp_inc_folders(
        results_folders[k], comps_dfs[k], neo_comps_dfs[k], preg_dict[k]['preg_data_frame'],
        births_dict[k]['births_data_frame'], comp_pregs_dict[k]['comp_preg_data_frame']) for k in results_folders}

    rate_keys = list()
    for k in comp_inc_folders[scenario_titles[0]].keys():
        if 'per_year' in k:
            rate_keys.append(k)

    for dict_key, axis, title, save_name in \
        zip(rate_keys,
            ['Rate per 1000 pregnancies',
             'Rate per 1000 pregnancies',
             'Rate per 1000 pregnancies',
             'Rate per 1000 pregnancies',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 100 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 100 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Prevalence at birth',
             'Prevalence following birth',
             'Rate per 100 Births',
             'Rate per 100 Births',
             'Rate per 100 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births'],

            ['Mean Rate of Ectopic Pregnancy per Year by Scenario',
             'Mean Rate of Spontaneous Abortion per Year by Scenario',
             'Mean Rate of Induced Abortion per Year by Scenario',
             'Mean Rate of Syphilis per Year by Scenario',
             'Mean Rate of Gestational Diabetes per Year by Scenario',
             'Mean Rate of PROM per Year by Scenario',
             'Mean Rate of Placenta Praevia per Year by Scenario',
             'Mean Rate of Placental Abruption per Year by Scenario',
             'Mean Rate of Post term labour per Year by Scenario',
             'Mean Rate of Obstructed Labour per Year by Scenario',
             'Mean Rate of Uterine Rupture per Year by Scenario',
             'Mean Rate of Gestational Hypertension per Year by Scenario',
             'Mean Rate of Mild Pre-eclampsia per Year by Scenario',
             'Mean Rate of Severe Gestational Hypertension per Year by Scenario',
             'Mean Rate of Severe Pre-eclampsia per Year by Scenario',
             'Mean Rate of Eclampsia per Year by Scenario',
             'Mean Rate of Antepartum Haemorrhage per Year by Scenario',
             'Mean Rate of Preterm Labour per Year by Scenario',
             'Mean Rate of Maternal Sepsis per Year by Scenario',
             'Mean Rate of Antenatal Maternal Sepsis per Year by Scenario',
             'Mean Rate of Intrapartum Maternal Sepsis per Year by Scenario',
             'Mean Rate of Postpartum Maternal Sepsis per Year by Scenario',
             'Mean Rate of Postpartum Haemorrhage per Year by Scenario',
             'Mean Rate of Primary Postpartum Haemorrhage per Year by Scenario',
             'Mean Rate of Secondary Postpartum Haemorrhage per Year by Scenario',
             'Mean Prevalence of Anaemia at birth per Year by Scenario',
             'Mean Prevalence of Anaemia following birth per Year by Scenario',
             'Mean Rate of Macrosomia per Year by Scenario',
             'Mean Rate of Small for Gestational Age per Year by Scenario',
             'Mean Rate of Low Birth Rate per Year by Scenario',
             'Mean Rate of  Newborn Respiratory Depression per Year by Scenario',
             'Mean Rate of Preterm Respiratory Distress Syndrome per Year by Scenario',
             'Mean Rate of Neonatal Sepsis per Year by Scenario',
             'Mean Rate of Neonatal Encephalopathy per Year by Scenario'],
            rate_keys):
        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            scen_colours, sim_years, comp_inc_folders, dict_key, axis, title, comp_incidence_path, save_name)

    avg_keys = list()
    for k in comp_inc_folders[scenario_titles[0]].keys():
        if 'avg' in k:
            avg_keys.append(k)

    for dict_key, axis, title, save_name in \
        zip(avg_keys,
            ['Rate per 1000 pregnancies',
             'Rate per 1000 pregnancies',
             'Rate per 1000 pregnancies',
             'Rate per 1000 pregnancies',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 100 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 100 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Prevalence at birth',
             'Prevalence following birth',
             'Rate per 100 Births',
             'Rate per 100 Births',
             'Rate per 100 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births',
             'Rate per 1000 Births'],

            ['Mean Rate of Ectopic Pregnancy During Intervention by Scenario',
             'Mean Rate of Spontaneous Abortion During Intervention by Scenario',
             'Mean Rate of Induced Abortion During Intervention by Scenario',
             'Mean Rate of Syphilis During Intervention by Scenario',
             'Mean Rate of Gestational Diabetes During Intervention by Scenario',
             'Mean Rate of PROM During Intervention by Scenario',
             'Mean Rate of Placenta Praevia During Intervention by Scenario',
             'Mean Rate of Placental Abruption During Intervention by Scenario',
             'Mean Rate of Post term labour During Intervention by Scenario',
             'Mean Rate of Obstructed Labour During Intervention by Scenario',
             'Mean Rate of Uterine Rupture During Intervention by Scenario',
             'Mean Rate of Gestational Hypertension During Intervention by Scenario',
             'Mean Rate of Mild Pre-eclampsia During Intervention by Scenario',
             'Mean Rate of Severe Gestational Hypertension During Intervention by Scenario',
             'Mean Rate of Severe Pre-eclampsia During Intervention by Scenario',
             'Mean Rate of Eclampsia During Intervention by Scenario',
             'Mean Rate of Antepartum Haemorrhage During Intervention by Scenario',
             'Mean Rate of Preterm Labour During Intervention by Scenario',
             'Mean Rate of Maternal Sepsis During Intervention by Scenario',
             'Mean Rate of Antenatal Maternal Sepsis During Intervention by Scenario',
             'Mean Rate of Intrapartum Maternal Sepsis During Intervention by Scenario',
             'Mean Rate of Postpartum Maternal Sepsis During Intervention by Scenario',
             'Mean Rate of Postpartum Haemorrhage During Intervention by Scenario',
             'Mean Rate of Primary Postpartum Haemorrhage During Intervention by Scenario',
             'Mean Rate of Secondary Postpartum Haemorrhage During Intervention by Scenario',
             'Mean Prevalence of Anaemia at birth During Intervention by Scenario',
             'Mean Prevalence of Anaemia following birth During Intervention by Scenario',
             'Mean Rate of Macrosomia During Intervention by Scenario',
             'Mean Rate of Small for Gestational Age During Intervention by Scenario',
             'Mean Rate of Low Birth Weight During Intervention by Scenario',
             'Mean Rate of  Newborn Respiratory Depression During Intervention by Scenario',
             'Mean Rate of Preterm Respiratory Distress Syndrome During Intervention by Scenario',
             'Mean Rate of Neonatal Sepsis During Intervention by Scenario',
             'Mean Rate of Neonatal Encephalopathy During Intervention by Scenario'],
            avg_keys):
        plot_agg_graph(comp_inc_folders, dict_key, axis, title, save_name, comp_incidence_path)

    rate_df_keys = list()
    for k in comp_inc_folders[scenario_titles[0]].keys():
        if 'df' in k:
            rate_df_keys.append(k)

    save_outputs(comp_inc_folders, rate_df_keys, 'diff_in_incidence_outcomes', comp_incidence_path)

    # OTHERS
    def extract_secondary_outcomes_dfs(folder, births):

        results = dict()

        # ANC HSI numbers
        def get_counts_of_hsi_by_treatment_id(_df):
            new_d = _df.assign(year=_df['date'].dt.year).drop(['date'], axis=1).set_index(['year'])
            new_d['total'] = new_d[list(new_d.columns)].sum(axis=1)
            return new_d['total']

        anc_count = extract_results(
            folder,
            module="tlo.methods.care_of_women_during_pregnancy",
            key="anc_visits_which_ran",
            custom_generate_series=get_counts_of_hsi_by_treatment_id,
            do_scaling=True)

        hsi_data = analysis_utility_functions.get_mean_and_quants(anc_count, sim_years)
        hsi_data_int = anc_count.loc[intervention_years[0]:intervention_years[-1]]

        agg_hsi_data = get_mean_from_columns(hsi_data_int, 'sum')

        agg = get_mean_95_CI_from_list(agg_hsi_data)

        results.update({'total_anc_df': anc_count,
                        'anc_contacts_trend': hsi_data,
                        'agg_anc_contacts': agg})

        # PNC HSI numbers
        # todo: this might crash on min pnc
        pnc_mat_count = extract_results(
            folder,
            module="tlo.methods.healthsystem.summary",
            key="HSI_Event",
            custom_generate_series=(
                lambda df: pd.concat([df, df['TREATMENT_ID'].apply(pd.Series)], axis=1).assign(
                    year=df['date'].dt.year).groupby(['year'])['PostnatalCare_Maternal'].sum()),
            do_scaling=True)

        pnc_neo_count = extract_results(
            folder,
            module="tlo.methods.healthsystem.summary",
            key="HSI_Event",
            custom_generate_series=(
                lambda df: pd.concat([df, df['TREATMENT_ID'].apply(pd.Series)], axis=1).assign(
                    year=df['date'].dt.year).groupby(['year'])['PostnatalCare_Neonatal'].sum()),
            do_scaling=True)

        if 'min_pnc' in folder.name:
            pre_int_years = list(range(2010, intervention_years[0]))
            pnc_data_mat_data = return_95_CI_across_runs(pnc_mat_count, pre_int_years)
            pnc_data_neo_data = return_95_CI_across_runs(pnc_neo_count, pre_int_years)

            for data in [pnc_data_mat_data, pnc_data_neo_data]:
                for l in data:
                    for y in intervention_years:
                        l.append(0)
            mat_agg = [0, 0, 0]
            neo_agg = [0, 0, 0]

        else:

            pnc_data_mat_data = return_95_CI_across_runs(pnc_mat_count, sim_years)
            pnc_mat_data_int = pnc_mat_count.loc[intervention_years[0]:intervention_years[-1]]
            mat_agg_data = get_mean_from_columns(pnc_mat_data_int, 'sum')
            mat_agg = get_mean_95_CI_from_list(mat_agg_data)

            pnc_data_neo_data = return_95_CI_across_runs(pnc_neo_count, sim_years)
            pnc_neo_data_int = pnc_neo_count.loc[intervention_years[0]:intervention_years[-1]]
            neo_agg_data = get_mean_from_columns(pnc_neo_data_int, 'sum')
            neo_agg = get_mean_95_CI_from_list(neo_agg_data)

        results.update({'total_mat_pnc_count_df': pnc_mat_count,
                        'total_neo_pnc_count_df': pnc_neo_count,
                        'pnc_visits_mat_trend': pnc_data_mat_data,
                        'pnc_visits_mat_agg': mat_agg,
                        'pnc_visits_neo_trend': pnc_data_neo_data,
                        'pnc_visits_neo_agg': neo_agg})

        # MALARIA
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
        preg_clinical_counter = return_95_CI_across_runs(preg_clin_counter_years, sim_years)
        preg_clinical_counter_int = preg_clin_counter_years.loc[intervention_years[0]:intervention_years[-1]]
        preg_clin_counter_agg_data = get_mean_from_columns(preg_clinical_counter_int, 'sum')
        preg_clin_counter_agg = get_mean_95_CI_from_list(preg_clin_counter_agg_data)

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

        incidence = return_95_CI_across_runs(incidence_years, sim_years)

        results.update({'mal_total_clinical_df': preg_clin_counter_years,
                        'mal_incidence_df': incidence_years,
                        'mal_clin_counter': preg_clinical_counter,
                        'mal_clin_counter_agg': preg_clin_counter_agg,
                        'mal_incidence': incidence})

        # TB
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
        results.update({'tb_diagnosis_df': tb_new_diag_years})

        tb_treatment_dates = extract_results(
            folder,
            module="tlo.methods.tb",
            key="tb_treatment",
            column='tbTreatmentCoverage',
            index='date',
        )

        years = tb_treatment_dates.index.year
        tb_treatment_years = tb_treatment_dates.set_index(years)

        tb_diagnosis_int = tb_new_diag_years.loc[intervention_years[0]: intervention_years[-1]]
        tb_diagnosis = return_95_CI_across_runs(tb_new_diag_years, sim_years)
        tb_diagnosis_agg_data = get_mean_from_columns(tb_diagnosis_int, 'sum')
        tb_diagnosis_agg = get_mean_95_CI_from_list(tb_diagnosis_agg_data)
        tb_treatment = return_95_CI_across_runs(tb_treatment_years, sim_years)

        results.update({'tb_treatment_df': tb_treatment_years,
                        'tb_diagnosis': tb_diagnosis,
                        'tb_diagnosis_agg': tb_diagnosis_agg,
                        'tb_treatment': tb_treatment
                        })

        # HIV
        hiv_tests_dates = extract_results(
            folder,
            module="tlo.methods.hiv",
            key="hiv_program_coverage",
            column='prop_tested_adult_female',
            index='date',
        )

        years = hiv_tests_dates.index.year
        hiv_tests_years = hiv_tests_dates.set_index(years)

        hiv_tests_rate_dates = extract_results(
            folder,
            module="tlo.methods.hiv",
            key="hiv_program_coverage",
            column='per_capita_testing_rate',
            index='date',
        )

        years = hiv_tests_rate_dates.index.year
        hiv_tests_rate_years = hiv_tests_rate_dates.set_index(years)

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

        hiv_tests = return_95_CI_across_runs(hiv_tests_years, sim_years)
        hiv_test_int = hiv_tests_years.loc[intervention_years[0]:intervention_years[-1]]
        avg_test_prop_data = get_mean_from_columns(hiv_test_int, 'avg')
        avg_test_prop = get_mean_95_CI_from_list(avg_test_prop_data)

        hiv_test_rate = return_95_CI_across_runs(hiv_tests_rate_years, sim_years)
        art = return_95_CI_across_runs(art_years, sim_years)
        art_int = art_years.loc[intervention_years[0]: intervention_years[-1]]
        art_agg_data = get_mean_from_columns(art_int, 'sum')
        art_agg = get_mean_95_CI_from_list(art_agg_data)

        results.update({'hiv_prog_cov_df': hiv_tests_years,
                        'hiv_total_women_on_art_df': art_years,
                        'hiv_per_cap_testing_df': hiv_tests_rate_years,
                        'hiv_testing_prop': hiv_tests,
                        'avg_test_prop': avg_test_prop,
                        'hiv_testing_rate': hiv_test_rate,
                        'art_number': art,
                        'art_number_agg': art_agg
                        })

        # DEPRESSION
        diag_prop = extract_results(
            folder,
            module="tlo.methods.depression",
            key="summary_stats",
            column='p_ever_diagnosed_depression_if_ever_depressed',
            index='date',
        )
        diag_reindexed = diag_prop.set_index(diag_prop.index.year)
        diag_data = diag_reindexed.groupby(diag_reindexed.index).mean()
        diag_final = return_95_CI_across_runs(diag_data, sim_years)

        anti_depress = extract_results(
            folder,
            module="tlo.methods.depression",
            key="summary_stats",
            column='prop_antidepr_if_ever_depr',
            index='date',
        )
        anti_depress_reindexed = anti_depress.set_index(anti_depress.index.year)
        ad_data = anti_depress_reindexed.groupby(anti_depress_reindexed.index).mean()
        ad_final = return_95_CI_across_runs(ad_data, sim_years)

        tt = extract_results(
            folder,
            module="tlo.methods.depression",
            key="summary_stats",
            column='prop_ever_talk_ther_if_ever_depr',
            index='date',
        )

        tt_reindexed = tt.set_index(tt.index.year)
        tt_data = tt_reindexed.groupby(tt_reindexed.index).mean()
        tt_final = return_95_CI_across_runs(tt_data, sim_years)

        results.update({'depression_treatment_tt_df': tt_data,
                        'depression_diag_df': diag_data,
                        'depression_treatment_med_df': ad_data,
                        'dep_diag': diag_final,
                        'dep_anti_d': ad_final,
                        'dep_talking_t': tt_final
                        })

        # CS RATE
        cs_delivery = extract_results(
            folder,
            module="tlo.methods.labour",
            key="delivery_setting_and_mode",
            custom_generate_series=(
                lambda df_: df_.loc[df_['mode'] == 'caesarean_section'].assign(year=df_['date'].dt.year
                                                                               ).groupby(['year'])['mother'].count()),
            do_scaling=True
        )

        cs_rate_df = (cs_delivery/births) * 100
        cs_rate_per_year = return_95_CI_across_runs(cs_rate_df, sim_years)
        cs_int = cs_rate_df.loc[intervention_years[0]:intervention_years[-1]]
        cs_avg_rate = get_mean_from_columns(cs_int, 'avg')
        avg_cs = get_mean_95_CI_from_list(cs_avg_rate)

        results.update({'cs_rate_df': cs_rate_df,
                        'cs_rate_per_year': cs_rate_per_year,
                        'avg_cs_rate_int': avg_cs})

        # AVD RATE
        avd_delivery = extract_results(
            folder,
            module="tlo.methods.labour",
            key="delivery_setting_and_mode",
            custom_generate_series=(
                lambda df_:
                df_.loc[df_['mode'] == 'instrumental'].assign(year=df_['date'].dt.year
                                                              ).groupby(['year'])[
                    'mother'].count()),
            do_scaling=True )

        avd_rate_df = (avd_delivery / births) * 100
        avd_rate = return_95_CI_across_runs(avd_rate_df, sim_years)
        avd_int = avd_rate_df.loc[intervention_years[0]:intervention_years[-1]]
        avd_avg_rate = get_mean_from_columns(avd_int, 'avg')
        avg_avd = get_mean_95_CI_from_list(avd_avg_rate)

        results.update({'avd_rate_df': avd_rate_df,
                        'avd_rate_per_year': avd_rate,
                        'avg_avd_rate_int': avg_avd})

        return results

    sec_outcomes_df = {k: extract_secondary_outcomes_dfs(results_folders[k],
                                                         births_dict[k]['births_data_frame']) for k in results_folders}

    so_keys = list()
    for k in sec_outcomes_df[scenario_titles[0]].keys():
        if 'df' in k:
            so_keys.append(k)

    save_outputs(sec_outcomes_df, so_keys, 'diff_in_secondary_outcomes', secondary_oc_path)

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, sec_outcomes_df, 'avd_rate_per_year',
        '% Total Births',
        'Proportion of total births which occur via Assisted Vaginal Delivery',
        secondary_oc_path, 'avd_trend')

    plot_agg_graph(sec_outcomes_df, 'avg_avd_rate_int', '% Total Births',
                   'Average proportion of total births occuring via Assisted Vaginal Delivery',
                   'avg_avd_rate', secondary_oc_path)

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, sec_outcomes_df, 'cs_rate_per_year',
        '% Total Births',
        'Proportion of total births which occur via Caesarean Section',
        secondary_oc_path, 'cs_trend')

    plot_agg_graph(sec_outcomes_df, 'avg_cs_rate_int', '% Total Births',
                   'Average proportion of total births occurring via  Caesarean Section',
                   'avg_cs_rate', secondary_oc_path)

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, sec_outcomes_df, 'anc_contacts_trend',
        'Number of Visits',
        'Total Number of Antenatal Care Visits per Year Per Scenario',
        secondary_oc_path, 'anc_visits_crude_rate')

    plot_agg_graph(sec_outcomes_df, 'agg_anc_contacts', 'Total ANC contacts', 'Total Number of ANC visits per Scenario',
                   'agg_anc_contacts', secondary_oc_path)

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, sec_outcomes_df, 'pnc_visits_mat_trend',
        'Crude Number',
        'Total Number of Maternal Postnatal Care Visits per Year Per Scenario',
        secondary_oc_path, 'pnc_mat_visits')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, sec_outcomes_df, 'pnc_visits_neo_trend',
        'Crude Number',
        'Total Number of Neonatal Postnatal Care Visits per Year Per Scenario',
        secondary_oc_path, 'pnc_neo_visits')

    for group, title in zip(['mat', 'neo'], ['Maternal', 'Neonatal']):
        plot_agg_graph(sec_outcomes_df, f'pnc_visits_{group}_agg', 'Total PNC Visist',
                       f'Total Number of {title} PNC visits per Scenario',
                       f'agg_{group}_pnc_visits', secondary_oc_path)

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, sec_outcomes_df, 'mal_clin_counter',
        'Num. Clinical Cases',
        'Number of Clinical Cases of Malaria During Pregnancy Per Year Per Scenario',
        secondary_oc_path, 'mal_clinical_cases')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, sec_outcomes_df, 'mal_incidence',
        'Incidence per 1000 person years',
        'Incidence of Malaria Per Year Per Scenario',
        secondary_oc_path, 'mal_incidence')

    plot_agg_graph(sec_outcomes_df, 'mal_clin_counter_agg', 'Number of Clinical Cases',
                   'Total Clinical Cases of Malaria During Pregnancy Per Scenario', 'mal_agg_clin_cases',
                   secondary_oc_path)

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, sec_outcomes_df, 'tb_diagnosis',
        'Number of Tb Diagnoses',
        'Number of New Tb Diagnoses Per Year Per Scenario',
        secondary_oc_path, 'tb_diagnosis')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, sec_outcomes_df, 'tb_treatment',
        '% New Tb Cases Treated',
        'Proportion of New Cases of Tb Treated Per Year Per Scenario',
        secondary_oc_path, 'tb_treatment')

    plot_agg_graph(sec_outcomes_df, 'tb_diagnosis_agg', 'Number of Tb Diagnoses',
                   'Total New Tb Cases Diagnosed per Scenario', 'tb_diagnoses_agg', secondary_oc_path)

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, sec_outcomes_df, 'hiv_testing_prop',
        '% Total Female Pop.',
        'Proportion of Female Population Who Received HIV test Per Year Per Scenario',
        secondary_oc_path, 'hiv_fem_testing_prop')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, sec_outcomes_df, 'hiv_testing_rate',
        'Per Captia Rate',
        'Rate of HIV testing per capita per year per scenario',
        secondary_oc_path, 'hiv_pop_testing_rate')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, sec_outcomes_df, 'art_number',
        'Women',
        'Number of Women Receiving ART per Year Per Scenario',
        secondary_oc_path, 'hiv_women_art')

    plot_agg_graph(sec_outcomes_df, 'art_number_agg', 'Number of Women',
                   'Total Number of Women Receiving ART per Per Scenario', 'hiv_women_art_agg', secondary_oc_path)

    plot_agg_graph(sec_outcomes_df, 'avg_test_prop', '% Total Female Pop.',
                   'Average % of Total Female Population Tested for HIV during Intervention Period',
                   'avg_test_prop', secondary_oc_path)

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, sec_outcomes_df, 'dep_diag',
        'Proportion (%)',
        'Proportion of Ever Depressed Individuals Diagnosed with Depression',
        secondary_oc_path, 'depression_diag')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, sec_outcomes_df, 'dep_anti_d',
        'Proportion (%)',
        'Proportion of Ever Depressed Individuals Started on Antidepressants',
        secondary_oc_path, 'depression_ad')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, sec_outcomes_df, 'dep_talking_t',
        'Proportion (%)',
        'Proportion of Ever Depressed Individuals Started on Talking Therapy',
        secondary_oc_path, 'depression_tt')


    # TODO: diff between MMR by cause


    # Create dictionary containing the results folder for each scenario

    # output_df = pd.DataFrame(columns=list(scenario_file_dict.keys()))

