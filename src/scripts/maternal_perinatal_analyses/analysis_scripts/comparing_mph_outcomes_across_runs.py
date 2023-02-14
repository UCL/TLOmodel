import os

import analysis_utility_functions
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats

from tlo.analysis.utils import extract_results, get_scenario_outputs

plt.style.use('seaborn-darkgrid')


def compare_outcomes_across_runs(scenario_file_dict, outputspath, sim_years, intervention_years, service_of_interest):
    """
    """
    # Create dictionary containing the results folder for each scenario
    results_folders = {k: get_scenario_outputs(scenario_file_dict[k], outputspath)[-1] for k in scenario_file_dict}


    # Create folder to store graphs (if it hasn't already been created when ran previously)
    path = f'{outputspath}/compare_runs_graphs_{service_of_interest}_and_{results_folders["Status Quo"].name}'
    if not os.path.isdir(path):
        os.makedirs(f'{outputspath}/compare_runs_graphs_{service_of_interest}_and_{results_folders["Status Quo"].name}')

    plot_destination_folder = path
    # output_df = pd.DataFrame(columns=['scenario', 'output', 'mean_value_for_int_period', 'mean_diff_outcome_per_year',
    #                                   'avg_diff_outcome_int_period', 'percent_diff_outcome_int_period',
    #                                   'prop_reduction'])

    output_df = pd.DataFrame(
                   columns=['scenario',
                            'output',
                            'mean_95%CI_value_for_int_period',
                            'skew_for_int_data',
                            'mean_95%CI_diff_outcome_int_period',
                            'skew_for_diff_data',
                            'median_diff_outcome_int_period'])


     # GET DENOMINATOR AND COMPLICATION DATA FRAMES
    births_dict = analysis_utility_functions.return_birth_data_from_multiple_scenarios(results_folders,
                                                                                       sim_years, intervention_years)

    preg_dict = analysis_utility_functions.return_pregnancy_data_from_multiple_scenarios(results_folders,
                                                                                         sim_years, intervention_years)

    comps_dfs = {k: analysis_utility_functions.get_modules_maternal_complication_dataframes(results_folders[k])
                 for k in results_folders}

    neo_comps_dfs = {k: analysis_utility_functions.get_modules_neonatal_complication_dataframes(results_folders[k])
                     for k in results_folders}

    comp_pregs_dict = {k: analysis_utility_functions.get_completed_pregnancies_from_multiple_scenarios(
        comps_dfs[k], births_dict[k], results_folders[k], sim_years, intervention_years) for k in results_folders}

    def extract_deaths_and_stillbirths(folder, birth_df):
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

        indirect_deaths = extract_results(
            folder,
            module="tlo.methods.demography.detail",
            key="properties_of_deceased_persons",
            custom_generate_series=(
                lambda df: df.loc[(df['is_pregnant'] | df['la_is_postpartum']) &
                                      df['cause_of_death'].str.contains(
                                          'AIDS_non_TB|AIDS_TB|TB|Malaria|Suicide|ever_stroke|diabetes|'
                                          'chronic_ischemic_hd|ever_heart_attack|chronic_kidney_disease')].assign(
                    year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
            )
        indirect_deaths_final = indirect_deaths.fillna(0)
        total_deaths = direct_deaths_final + indirect_deaths_final

        mmr = (total_deaths/birth_df) * 100_000

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

        nmr = (neo_deaths/birth_df) * 1000

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

        an_sbr = (an_stillbirth_results/birth_df) * 1000
        ip_sbr = (ip_stillbirth_results/birth_df) * 1000
        sbr = (all_sb/birth_df) * 1000

        # TOTAL MATERNAL DEATHS
        return {'mmr': mmr,
                'nmr': nmr,
                'sbr': sbr,
                'ip_sbr': ip_sbr,
                'an_sbr': an_sbr,
                }

    # Extract data from scenarios
    death_folders = {k: extract_deaths_and_stillbirths(results_folders[k], births_dict[k]['births_data_frame'])
                     for k in results_folders}

    def extract_dalys(folder):
        py = extract_results(
            folder,
            module="tlo.methods.demography",
            key="person_years",
            custom_generate_series=(
                lambda df: df.assign(total=(df['M'].apply(lambda x: sum(x.values()))) + df['F'].apply(
                    lambda x: sum(x.values()))).assign(
                    year=df['date'].dt.year).groupby(['year'])['total'].sum()),
            do_scaling=True)
        person_years = py.fillna(0)

        m_dalys_s = extract_results(
            folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year'])['Maternal Disorders'].sum()),
            do_scaling=True)
        m_dalys_stacked = m_dalys_s.fillna(0)

        n_dalys_s = extract_results(
            folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year'])['Neonatal Disorders'].sum()),
            do_scaling=True)
        n_dalys_stacked = n_dalys_s.fillna(0)

        return {'mat_daly_rate': (m_dalys_stacked/person_years) * 100_000,
                'neo_daly_rate': (n_dalys_stacked/person_years) * 100_000}

    dalys_folders = {k: extract_dalys(results_folders[k]) for k in results_folders}

    comp_inc_folders = {k: analysis_utility_functions.extract_comp_inc_folders(
        results_folders[k], comps_dfs[k], neo_comps_dfs[k], preg_dict[k]['preg_data_frame'],
        births_dict[k]['births_data_frame'],  comp_pregs_dict[k]['comp_preg_data_frame']) for k in results_folders}

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

        results.update({'total_anc':anc_count})

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

        results.update({'mat_pnc_count': pnc_mat_count})
        results.update({'neo_pnc_count': pnc_neo_count})


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
        results.update({'mal_clinical': preg_clin_counter_years})

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
        results.update({'mal_incidence': incidence_years})

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
        results.update({'tb_diagnosis': tb_new_diag_years})

        tb_treatment_dates = extract_results(
            folder,
            module="tlo.methods.tb",
            key="tb_treatment",
            column='tbTreatmentCoverage',
            index='date',
        )

        years = tb_treatment_dates.index.year
        tb_treatment_years = tb_treatment_dates.set_index(years)
        results.update({'tb_treatment': tb_treatment_years})

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
        results.update({'hiv_prog_cov': hiv_tests_years})

        hiv_tests_rate_dates = extract_results(
            folder,
            module="tlo.methods.hiv",
            key="hiv_program_coverage",
            column='per_capita_testing_rate',
            index='date',
        )

        years = hiv_tests_rate_dates.index.year
        hiv_tests_rate_years = hiv_tests_rate_dates.set_index(years)
        results.update({'hiv_per_cap_testing': hiv_tests_rate_years})

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
        results.update({'hiv_women_on_art': art_years})

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

        results.update({'depression_diag': diag_data})

        anti_depress = extract_results(
            folder,
            module="tlo.methods.depression",
            key="summary_stats",
            column='prop_antidepr_if_ever_depr',
            index='date',
        )
        anti_depress_reindexed = anti_depress.set_index(anti_depress.index.year)
        ad_data = anti_depress_reindexed.groupby(anti_depress_reindexed.index).mean()
        results.update({'depression_treatment_med': ad_data})

        tt = extract_results(
            folder,
            module="tlo.methods.depression",
            key="summary_stats",
            column='prop_ever_talk_ther_if_ever_depr',
            index='date',
        )

        tt_reindexed = tt.set_index(tt.index.year)
        tt_data = tt_reindexed.groupby(tt_reindexed.index).mean()
        results.update({'depression_treatment_tt': tt_data})

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

        results.update({'cs_rate': (cs_delivery/births) * 100})

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

        results.update({'avd_rate': (avd_delivery/births) * 100})

        return results

    sec_outcomes_df = {k: extract_secondary_outcomes_dfs(results_folders[k],
                                                         births_dict[k]['births_data_frame']) for k in results_folders}

    def get_diff_between_runs(dfs, baseline, intervention, keys, intervention_years, output_df):

        def get_med_or_mean_from_columns(df, mean_or_med):
            values = list()
            for col in df:
                if mean_or_med == 'mean':
                    values.append(np.mean(df[col]))
                else:
                    values.append(np.median(df[col]))
            return values

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
            mean_outcome_list_int = get_med_or_mean_from_columns(dfs[intervention][k].loc[intervention_years[0]:
                                                                                          intervention_years[-1]],
                                                                 'mean')
            skew_mean_outcome_list_int = scipy.stats.skew(mean_outcome_list_int)
            mean_outcome_value_int = get_mean_and_confidence_interval(mean_outcome_list_int)

            # Calculate mean difference between outcome by run for intervention period, check skew and
            # calculate mean/95 % CI
            mean_diff_list = get_med_or_mean_from_columns(int_diff.loc[intervention_years[0]:
                                                                       intervention_years[-1]], 'mean')

            skew_diff_list = scipy.stats.skew(mean_diff_list)
            mean_outcome_diff = get_mean_and_confidence_interval(mean_diff_list)
            median_outcome_diff = [round(np.median(mean_diff_list), 2),
                                   round(np.quantile(mean_diff_list, 0.025), 2),
                                   round(np.quantile(mean_diff_list, 0.975), 2)]

            res_df = pd.DataFrame([(intervention,
                                    k,
                                    mean_outcome_value_int,
                                    skew_mean_outcome_list_int,
                                    mean_outcome_diff,
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

            # # MEDIAN CALCULATIONS
            # int_med_value = get_med_or_mean_from_columns(
            #     dfs[intervention][k].loc[intervention_years[0]: intervention_years[-1]], 'median')
            # total_median = [round(np.median(int_mean_value), 2),
            #                 round(np.quantile(int_mean_value, 0.025), 2),
            #                 round(np.quantile(int_mean_value, 0.975), 2)]
            #
            # int_run_median = get_med_or_mean_from_columns(int_diff, 'median')
            #
            # total_median_diff = [round(np.median(int_run_means), 2),
            #                      round(np.quantile(int_run_means, 0.025), 2),
            #                      round(np.quantile(int_run_means, 0.975), 2)]
            #
            # # MEAN CALCULATIONS
            # int_mean_value = get_med_or_mean_from_columns(dfs[intervention][k].loc[intervention_years[0]:
            #                                                                 intervention_years[-1]], 'mean')
            # total_mean = [round(np.mean(int_mean_value), 2),
            #               round(np.quantile(int_mean_value, 0.025), 2),
            #               round(np.quantile(int_mean_value, 0.975), 2)]
            #
            # mean_diff_in_outcome_per_year = analysis_utility_functions.get_mean_and_quants(int_diff, intervention_years)
            # int_run_means = get_med_or_mean_from_columns(int_diff, 'mean')
            #
            # pos, neg = 0, 0
            # for num in int_run_means:
            #     if num >= 0:
            #         pos += 1
            #     else:
            #         neg += 1
            # prop_pos = (pos / (pos + neg)) * 100
            #
            # import numpy as np
            # import scipy.stats
            #
            # def mean_confidence_interval(data, confidence=0.95):
            #     a = 1.0 * np.array(data)
            #     n = len(a)
            #     m, se = np.mean(a), scipy.stats.sem(a)
            #     h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
            #     return m, m - h, m + h
            #
            # # PERCENTAGE DIFFERENCE
            # total_diff = [round(np.mean(int_run_means), 2),
            #               round(np.quantile(int_run_means, 0.025), 2),
            #               round(np.quantile(int_run_means, 0.975), 2)]
            #
            #
            # p_diff = ((dfs[intervention][k] - dfs[baseline][k]) / dfs[baseline][k]) * 100
            # int_p_diff = p_diff.loc[intervention_years[0]: intervention_years[-1]]
            # pdiff_run_means = get_med_or_mean_from_columns(int_p_diff, 'mean')
            # total_p_diff = [round(np.mean(pdiff_run_means), 2),
            #                 round(np.quantile(pdiff_run_means, 0.025), 2),
            #                 round(np.quantile(pdiff_run_means, 0.975), 2)]
            #
            #
            # int_df = dfs[intervention][k].loc[intervention_years[0]: intervention_years[-1]]
            # bl_df =dfs[baseline][k].loc[intervention_years[0]: intervention_years[-1]]
            #
            # mean_int_mmr_by_scen = get_med_or_mean_from_columns(int_df, 'mean')
            # mean_bl_mmr_by_scen = get_med_or_mean_from_columns(bl_df, 'mean')
            #
            # res_df = pd.DataFrame([(intervention,
            #                         k,
            #                         total_mean,
            #                         mean_diff_in_outcome_per_year,
            #                         total_diff,
            #                         total_p_diff,
            #                         prop_pos)],
            #                       columns=['scenario',
            #                                'output',
            #                                'mean_value_for_int_period',
            #                                'mean_diff_outcome_per_year',
            #                                'avg_diff_outcome_int_period',
            #                                'percent_diff_outcome_int_period',
            #                                'prop_reduction'])

            # output_df = output_df.append(res_df)

        return output_df

    s_names = list(results_folders.keys())

    def save_outputs(folder, save_name):
        keys = list(folder[s_names[0]].keys())
        dfs = []
        for k in s_names:
            scen_df = get_diff_between_runs(folder, 'Status Quo', k, keys, intervention_years, output_df)
            dfs.append(scen_df)

        final_df = pd.concat(dfs)
        final_df.to_csv(f'{plot_destination_folder}/{save_name}.csv')

    save_outputs(death_folders, 'diff_in_mortality_outcomes')
    save_outputs(dalys_folders, 'diff_in_daly_outcomes')
    save_outputs(comp_inc_folders, 'diff_incidence_outcomes')
    save_outputs(sec_outcomes_df, 'diff_secondary_outcomes')




