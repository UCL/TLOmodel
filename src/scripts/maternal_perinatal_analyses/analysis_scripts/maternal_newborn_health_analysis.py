import os

import analysis_utility_functions
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import extract_results, get_scenario_outputs

plt.style.use('seaborn-darkgrid')


def run_maternal_newborn_health_analysis(scenario_file_dict, outputspath, intervention_years, service_of_interest,
                                         show_all_results):
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

    # Create folder to store graphs (if it hasnt already been created when ran previously)
    path = f'{outputspath}/{service_of_interest}_analysis_output_graphs_{results_folders["Status Quo"].name}'
    if not os.path.isdir(path):
        os.makedirs(f'{outputspath}/{service_of_interest}_analysis_output_graphs_{results_folders["Status Quo"].name}')

    # Save the file path
    plot_destination_folder = path

    # ========================================= BIRTHs PER SCENARIO ==================================================
    # Access birth data for each scenario (used as a denominator in some parts of the script)
    births_dict = analysis_utility_functions.return_birth_data_from_multiple_scenarios(results_folders,
                                                                                       intervention_years)

    # ===================================== INTERVENTION COVERAGE ====================================================
    # Depending on the service of interest for the scenarios being analysed, here we output the coverage of
    # relevant interventions

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
            anc_cov_of_interest = extract_results(
                folder,
                module="tlo.methods.care_of_women_during_pregnancy",
                key="anc_count_on_birth",
                custom_generate_series=(
                    lambda df: df.loc[df['total_anc'] >= service_structure].assign(
                        year=df['date'].dt.year).groupby(['year'])['person_id'].count()),
                do_scaling=True
            )

            # Get the mean and quantiles for both DFs
            mean_total_anc = analysis_utility_functions.get_mean_and_quants(anc_coverage, intervention_years)
            mean_cov_anc = analysis_utility_functions.get_mean_and_quants(anc_cov_of_interest, intervention_years)

            # The calculate the mean proportion of women receiving the ANC coverage of interest
            result_m = [(x / y) * 100 for x, y in zip(mean_cov_anc[0], mean_total_anc[0])]
            result_lq = [(x / y) * 100 for x, y in zip(mean_cov_anc[1], mean_total_anc[1])]
            result_uq = [(x / y) * 100 for x, y in zip(mean_cov_anc[2], mean_total_anc[2])]

            return [result_m, result_lq, result_uq]

        cov_data_4 = {k: get_anc_coverage(results_folders[k], 4) for k in results_folders}
        cov_data_8 = {k: get_anc_coverage(results_folders[k], 8) for k in results_folders}

        # output graphs
        for service_structure, cov_data in zip([4, 8], [cov_data_4, cov_data_8]):
            analysis_utility_functions.comparison_graph_multiple_scenarios(
                intervention_years, cov_data, '% Births',
                f'Proportion of women receiving {service_structure} (or more) ANC visits at birth',
                plot_destination_folder, f'anc{service_structure}_cov')

    if service_of_interest == 'sba' or show_all_results:
        def get_delivery_place_info(folder, intervention_years):
            deliver_setting_results = extract_results(
                folder,
                module="tlo.methods.labour",
                key="delivery_setting_and_mode",
                custom_generate_series=(
                    lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'facility_type'])[
                        'mother'].count()),
                do_scaling=True
            )

            hb_data = analysis_utility_functions.get_mean_and_quants_from_str_df(
                deliver_setting_results, 'home_birth', intervention_years)
            hp_data = analysis_utility_functions.get_mean_and_quants_from_str_df(
                deliver_setting_results, 'hospital', intervention_years)
            hc_data = analysis_utility_functions.get_mean_and_quants_from_str_df(
                deliver_setting_results, 'health_centre', intervention_years)

            mean_total_deliveries = [x + y + z for x, y, z in zip(hb_data[0], hc_data[0], hp_data[0])]

            home_birth_rate = [(x / y) * 100 for x, y in zip(hb_data[0], mean_total_deliveries)]
            home_birth_lq = [(x / y) * 100 for x, y in zip(hb_data[1], mean_total_deliveries)]
            home_birth_uq = [(x / y) * 100 for x, y in zip(hb_data[2], mean_total_deliveries)]

            health_centre_rate = [(x / y) * 100 for x, y in zip(hc_data[0], mean_total_deliveries)]
            health_centre_lq = [(x / y) * 100 for x, y in zip(hc_data[1], mean_total_deliveries)]
            health_centre_uq = [(x / y) * 100 for x, y in zip(hc_data[2], mean_total_deliveries)]

            hospital_rate = [(x / y) * 100 for x, y in zip(hp_data[0], mean_total_deliveries)]
            hospital_lq = [(x / y) * 100 for x, y in zip(hp_data[1], mean_total_deliveries)]
            hospital_uq = [(x / y) * 100 for x, y in zip(hp_data[2], mean_total_deliveries)]

            total_fd_rate = [x + y for x, y in zip(health_centre_rate, hospital_rate)]
            fd_lqs = [x + y for x, y in zip(health_centre_lq, hospital_lq)]
            fd_uqs = [x + y for x, y in zip(health_centre_uq, hospital_uq)]

            return {'hb': [home_birth_rate, home_birth_lq, home_birth_uq],
                    'hc': [health_centre_rate, health_centre_lq, health_centre_uq],
                    'hp': [hospital_rate, hospital_lq, home_birth_uq],
                    'fd': [total_fd_rate, fd_lqs, fd_uqs]}

        delivery_data = {k: get_delivery_place_info(results_folders[k], intervention_years) for k in results_folders}

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, delivery_data, 'fd',
            '% Total Births',
            'Facility Delivery Rate per Year Per Scenario',
            plot_destination_folder, 'fd_rate')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, delivery_data, 'hb',
            '% Total Births',
            'Home birth Rate per Year Per Scenario',
            plot_destination_folder, 'hb_rate')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, delivery_data, 'hp',
            '% Total Births',
            'Hospital birth Rate per Year Per Scenario',
            plot_destination_folder, 'hp_rate')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, delivery_data, 'hc',
            '% Total Births',
            'Health Centre Birth Rate per Year Per Scenario',
            plot_destination_folder, 'hc_rate')

    if service_of_interest == 'pnc' or show_all_results:
        def get_pnc_coverage(folder, birth_data):
            """
            Returns the mean, lower quantile, upper quantile proportion of women and neonates who received at least 1
            postnatal care visit after birth
            :param folder: results folder for scenario
            :param birth_data: dictionary containing mean/quantiles of births per year
            :return: mean, lower quantile, upper quantil of coverage
            """

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
            pnc_results_newborn = extract_results(
                folder,
                module="tlo.methods.postnatal_supervisor",
                key="total_neo_pnc_visits",
                custom_generate_series=(
                    lambda df: df.loc[df['visits'] > 0].assign(year=df['date'].dt.year).groupby(['year'])[
                        'child'].count()),
                do_scaling=True
            )

            # Get mean/quantiles
            pn_mat_data = analysis_utility_functions.get_mean_and_quants(pnc_results_maternal, intervention_years)
            pn_neo_data = analysis_utility_functions.get_mean_and_quants(pnc_results_newborn, intervention_years)

            # Use birth data to calculate coverage as a proportion of total births
            pnc_1_plus_rate_mat = [(x / y) * 100 for x, y in zip(pn_mat_data[0], birth_data[0])]
            pnc_mat_lqs = [(x / y) * 100 for x, y in zip(pn_mat_data[1], birth_data[1])]
            pnc_mat_uqs = [(x / y) * 100 for x, y in zip(pn_mat_data[2], birth_data[2])]

            pnc1_plus_rate_neo = [(x / y) * 100 for x, y in zip(pn_neo_data[0], birth_data[0])]
            pnc_neo_lqs = [(x / y) * 100 for x, y in zip(pn_neo_data[1], birth_data[1])]
            pnc_neo_uqs = [(x / y) * 100 for x, y in zip(pn_neo_data[2], birth_data[2])]

            return {'maternal': [pnc_1_plus_rate_mat, pnc_mat_lqs, pnc_mat_uqs],
                    'neonatal': [pnc1_plus_rate_neo, pnc_neo_lqs, pnc_neo_uqs]}

        coverage_data = {k: get_pnc_coverage(results_folders[k], births_dict[k]) for k in results_folders}

        # generate plots showing coverage of ANC intervention in the baseline and intervention scenarios
        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, coverage_data, 'maternal',
            '% Total Births',
            'Proportion of Mothers Receiving PNC following Birth',
            plot_destination_folder, 'mat_pnc_coverage')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, coverage_data, 'neonatal',
            '% Total Births',
            'Proportion of Neonates Receiving PNC following Birth',
            plot_destination_folder, 'neo_pnc_coverage')

    # ===================================== MATERNAL/NEONATAL DEATH ====================================================
    #  Extract data on direct and indirect maternal deaths from the demography logger
    death_data = analysis_utility_functions.return_death_data_from_multiple_scenarios(results_folders, births_dict,
                                                                                      intervention_years,
                                                                                      detailed_log=True)

    # Output and save the relevant graphs
    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        intervention_years, death_data, 'direct_mmr',
        'Deaths per 100,000 live births',
        'MMR per Year at Baseline and Under Intervention (Direct only)', plot_destination_folder,
        'maternal_mr_direct')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        intervention_years, death_data, 'total_mmr',
        'Deaths per 100,000 live births',
        'MMR per Year at Baseline and Under Intervention (Total)',
        plot_destination_folder, 'maternal_mr_total')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        intervention_years, death_data, 'nmr',
        'Total Deaths per 1000 live births',
        'Neonatal Mortality Ratio per Year at Baseline and Under Intervention',
        plot_destination_folder, 'neonatal_mr_int')

    for group, group_label in zip(['Maternal', 'Neonatal'], ['m', 'n']):
        analysis_utility_functions.comparison_bar_chart_multiple_bars(
            death_data, f'crude_{group_label}_deaths', intervention_years,
            f'Total {group} Deaths (scaled)', f'Yearly Baseline {group} Deaths Compared to Intervention',
            plot_destination_folder, f'{group}_crude_deaths_comparison.png')

    #  ================================== STILLBIRTH  ================================================================
    # For interventions that may impact rates of stillbirth we output that information here
    if (service_of_interest != 'pnc') or show_all_results:
        sbr_data = analysis_utility_functions.return_stillbirth_data_from_multiple_scenarios(
            results_folders, births_dict, intervention_years)

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, sbr_data, 'an_sbr',
            'Antenatal stillbirths per 1000 births',
            'Antenatal stillbirth Rate per Year at Baseline and Under Intervention',
            plot_destination_folder, 'an_sbr_int')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, sbr_data, 'ip_sbr',
            'Intrapartum stillbirths per 1000 births',
            'Intrapartum stillbirth Rate per Year at Baseline and Under Intervention',
            plot_destination_folder, 'ip_sbr_int')

        # Output SBR per year for scenario vs intervention
        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, sbr_data, 'sbr',
            'Stillbirths per 1000 births',
            'Stillbirth Rate per Year at Baseline and Under Intervention',
            plot_destination_folder, 'sbr_int')

        analysis_utility_functions.comparison_bar_chart_multiple_bars(
            sbr_data, 'crude_sb', intervention_years,
            'Total Stillbirths (scaled)', 'Yearly Baseline Stillbirths Compared to Intervention',
            plot_destination_folder, 'crude_stillbirths_comparison.png')

    # =================================================== DALYS =======================================================
    # Here we extract maternal and neonatal DALYs from each scenario to allow for comparison
    dalys_data = analysis_utility_functions.return_dalys_from_multiple_scenarios(results_folders, intervention_years)
    for dict_key, axis, title, save_name in zip(['maternal_dalys_crude', 'maternal_dalys_rate', 'maternal_yll_crude',
                                                 'maternal_yll_rate', 'maternal_yld_crude', 'maternal_yld_rate',

                                                 'neonatal_dalys_crude', 'neonatal_dalys_rate', 'neonatal_yll_crude',
                                                 'neonatal_yll_rate', 'neonatal_yld_crude', 'neonatal_yld_rate'],

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
            intervention_years, dalys_data, dict_key, axis, title, plot_destination_folder, save_name)

    # ========================================= DIFFERENCE IN OUTCOMES ===============================================
    # Next we calculate the crude and percentage difference between scenarios for each year
    if service_of_interest == 'pnc':
        output_list = ['direct_mmr', 'nmr']
        data_list = [death_data, death_data]
    else:
        output_list = ['direct_mmr', 'sbr', 'nmr']
        data_list = [death_data, sbr_data, death_data]

    keys = results_folders.keys()
    keys_list = list(keys)
    keys_list.remove('Status Quo')

    for output, data in zip(output_list, data_list):
        diff_data = {k: analysis_utility_functions.get_differences_between_two_outcomes(
            data['Status Quo'][output], data[k][output]) for k
            in keys_list}  # todo: not ideal to have to change key names

        N = len(intervention_years)
        ind = np.arange(N)
        width = 0.2

        x_ticks = list()
        for x in range(len(intervention_years)):
            x_ticks.append(x)

        for dict_name in ['crude', 'percentage']:
            for k, position, colour in zip(diff_data, [ind, ind + width], ['bisque', 'powderblue']):
                plt.bar(position, diff_data[k][dict_name], width, label=k, color=colour)

            plt.ylabel('Reduction From Baseline')
            plt.xlabel('Years')
            plt.title(f'Mean Reduction of {output} from Baseline per Year ({dict_name})')
            plt.legend(loc='best')
            plt.xticks(x_ticks, labels=intervention_years)
            plt.savefig(f'{plot_destination_folder}/{dict_name}_diff_{output}.png')
            plt.show()

    # ========================================= HEALTH SYSTEM OUTCOMES ================================================
    # Next we output certain health system outcomes such as number of HSIs by scenario

    if service_of_interest == 'anc' or show_all_results:
        def get_hsi_counts_from_cowdp_logger(folder, intervention_years):

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

            hsi_data = analysis_utility_functions.get_mean_and_quants(hsi, intervention_years)

            return hsi_data

        hs_data = {k: get_hsi_counts_from_cowdp_logger(results_folders[k], intervention_years) for k in
                   results_folders}

        # todo: Better as a rate?
        analysis_utility_functions.comparison_graph_multiple_scenarios(
            intervention_years, hs_data, 'Crude Number',
            'Total Number of Antenatal Care Visits per Year Per Scenario',
            plot_destination_folder, f'{service_of_interest}_visits')

    if service_of_interest == 'pnc' or show_all_results:
        def get_hsi_counts_from_summary_logger(folder, intervention_years):
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

            hsi_data = analysis_utility_functions.get_mean_and_quants(hsi, intervention_years)
            hsi_data_neo = analysis_utility_functions.get_mean_and_quants(hsi_n, intervention_years)

            return {'mat': hsi_data,
                    'neo': hsi_data_neo}

        hs_data = {k: get_hsi_counts_from_summary_logger(results_folders[k], intervention_years) for k in
                   results_folders}

        # todo: Better as a rate?
        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, hs_data, 'mat',
            'Crude Number',
            'Total Number of Maternal Postnatal Care Visits per Year Per Scenario',
            plot_destination_folder, f'{service_of_interest}_mat_visits')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, hs_data, 'neo',
            'Crude Number',
            'Total Number of Neonatal Postnatal Care Visits per Year Per Scenario',
            plot_destination_folder, f'{service_of_interest}_neo_visits')

    # =========================================== ADDITIONAL OUTCOMES ================================================
    # Finally we output secondary outcomes related to other disease in the model which are related to/affected by
    # the pregnancy modules

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
                                                                                   intervention_years)

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
            incidence = analysis_utility_functions.get_mean_and_quants(incidence_years, intervention_years)

            return {'clin_counter': preg_clinical_counter,
                    'incidence': incidence}

        mal_data = {k: get_malaria_incidence_in_pregnancy(results_folders[k]) for k in results_folders}

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, mal_data, 'clin_counter',
            'Num. Clinical Cases',
            'Number of Clinical Cases of Malaria During Pregnancy Per Year Per Scenario',
            plot_destination_folder, 'mal_clinical_cases')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, mal_data, 'incidence',
            'Incidence per 1000 person years',
            'Incidence of Malaria Per Year Per Scenario',
            plot_destination_folder, 'mal_incidence')

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
                                                                          intervention_years)
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
            tb_treatment = analysis_utility_functions.get_mean_and_quants(tb_treatment_years, intervention_years)

            return {'diagnosis': tb_diagnosis,
                    'treatment': tb_treatment}

        tb_data = {k: get_tb_info_in_pregnancy(results_folders[k]) for k in results_folders}

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, tb_data, 'diagnosis',
            'Number of Tb Diagnoses',
            'Number of New Tb Diagnoses Per Year Per Scenario',
            plot_destination_folder, 'tb_diagnoses')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, tb_data, 'treatment',
            '% New Tb Cases Treated',
            'Proportion of New Cases of Tb Treated Per Year Per Scenario',
            plot_destination_folder, 'tb_treatment')

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
            hiv_tests = analysis_utility_functions.get_mean_and_quants(hiv_tests_years,
                                                                       intervention_years)
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
                                                                           intervention_years)

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
            art = analysis_utility_functions.get_mean_and_quants(art_years, intervention_years)

            return {'testing_prop': hiv_tests,
                    'testing_rate': hiv_test_rate,
                    'art_number': art}

        hiv_data = {k: get_hiv_information(results_folders[k]) for k in results_folders}

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, hiv_data, 'testing_prop',
            '% Total Female Pop.',
            'Proportion of Female Population Who Received HIV test Per Year Per Scenario',
            plot_destination_folder, 'hiv_fem_testing_prop')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, hiv_data, 'testing_rate',
            'Per Captia Rate',
            'Rate of HIV testing per capita per year per scenario',
            plot_destination_folder, 'hiv_pop_testing_rate')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, hiv_data, 'art_number',
            'Women',
            'Number of Women Receiving ART per Year Per Scenario',
            plot_destination_folder, 'hiv_women_art')

    # ------------------------------------------- Depression ---------------------------------------------------------
    # For depression we output diagnosis of ever depressed people, proportion of depressed people started on
    # antidepressants and proportion of depressed people started on talking therapy

    if service_of_interest != 'sba' or show_all_results:
        def get_depression_info_in_pregnancy(folder, intervention_years):

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
            diag_final = analysis_utility_functions.get_mean_and_quants(diag_data, intervention_years)

            anti_depress = extract_results(
                folder,
                module="tlo.methods.depression",
                key="summary_stats",
                column='prop_antidepr_if_ever_depr',
                index='date',
            )
            anti_depress_reindexed = anti_depress.set_index(anti_depress.index.year)
            ad_data = anti_depress_reindexed.groupby(anti_depress_reindexed.index).mean()
            ad_final = analysis_utility_functions.get_mean_and_quants(ad_data, intervention_years)

            tt = extract_results(
                folder,
                module="tlo.methods.depression",
                key="summary_stats",
                column='prop_ever_talk_ther_if_ever_depr',
                index='date',
            )

            tt_reindexed = tt.set_index(tt.index.year)
            tt_data = tt_reindexed.groupby(tt_reindexed.index).mean()
            tt_final = analysis_utility_functions.get_mean_and_quants(tt_data, intervention_years)

            return {'diag': diag_final,
                    'ad': ad_final,
                    'tt': tt_final}

        depression_data = {k: get_depression_info_in_pregnancy(results_folders[k], intervention_years) for k in
                           results_folders}

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, depression_data, 'diag',
            'Proportion (%)',
            'Proportion of Ever Depressed Individuals Diagnosed with Depression',
            plot_destination_folder, 'depression_diag')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, depression_data, 'ad',
            'Proportion (%)',
            'Proportion of Ever Depressed Individuals Started on Antidepressants',
            plot_destination_folder, 'depression_ad')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, depression_data, 'tt',
            'Proportion (%)',
            'Proportion of Ever Depressed Individuals Started on Talking Therapy',
            plot_destination_folder, 'depression_tt')
