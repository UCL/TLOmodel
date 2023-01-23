import os

import analysis_utility_functions
from matplotlib import pyplot as plt

from tlo.analysis.utils import extract_results, get_scenario_outputs


def met_need_and_contributing_factors_for_deaths(scenario_file_dict, outputspath, intervention_years,
                                                 service_of_interest):
    """
    When called this function will extract and output results relating to met need for the specific interventions
    delivered during the pregnancy modules. (met need here meaning (total cases/total treatments) for each
    'death causing' complication in the model. In addition the function will output the proportion of women
    who have died that have experienced factors impacting treatment such as delays, not seeking care, no consumables
    etc etc)
    :param scenario_file_dict: dict containing names of python scripts for each scenario of interest
    :param outputspath: directory for graphs to be saved
    :param intervention_years: years of interest for the analysis
    :param service_of_interest: ANC/SBA/PNC
    """
    # TODO: this is still a bit of a work in process and should be refined later (i.e. errors with uncertainty
    #  intervals and some estimates exceed 100 in certain runs)

    # Find results folder (most recent run generated using that scenario_filename)
    results_folders = {k: get_scenario_outputs(scenario_file_dict[k], outputspath)[-1] for k in scenario_file_dict}

    path = f'{outputspath}/met_need_{results_folders["Status Quo"].name}_{service_of_interest}'
    if not os.path.isdir(path):
        os.makedirs(
            f'{outputspath}/met_need_{results_folders["Status Quo"].name}_{service_of_interest}')

    plot_destination_folder = path

    # Get complication dataframes
    comp_dfs = {k: analysis_utility_functions.get_modules_maternal_complication_dataframes(results_folders[k]) for
                k in results_folders}
    neo_comp_dfs = {k: analysis_utility_functions.get_modules_neonatal_complication_dataframes(results_folders[k]) for
                    k in results_folders}

    treatments = ['pac',
                  'ep_case_mang',
                  'abx_an_sepsis',
                  'abx_pn_sepsis',
                  'uterotonics',
                  'man_r_placenta',
                  'ur_surg',
                  'blood_tran_ur',
                  'mag_sulph_an_severe_pre_eclamp',
                  'iv_htns_an_severe_pre_eclamp',
                  'mag_sulph_an_eclampsia',
                  'iv_htns_an_eclampsia',
                  'mag_sulph_la_severe_pre_eclamp',
                  'iv_htns_la_severe_pre_eclamp',
                  'mag_sulph_la_eclampsia',
                  'iv_htns_la_eclampsia',
                  'mag_sulph_pn_severe_pre_eclamp',
                  'iv_htns_pn_severe_pre_eclamp',
                  'mag_sulph_pn_eclampsia',
                  'iv_htns_pn_eclampsia',
                  'iv_htns_an_severe_gest_htn',
                  'iv_htns_la_severe_gest_htn',
                  'iv_htns_pn_severe_gest_htn',
                  'blood_tran_aph',
                  'blood_tran_pph',
                  'pph_surg',
                  'avd_ol',
                  'neo_resus',
                  'neo_sep_treat']

    # ============================================ MET NEED ==========================================================
    def get_total_interventions_delivered(results_folder, interventions, intervention_years):

        int_df = extract_results(
            results_folder,
            module="tlo.methods.labour.detail",
            key="intervention",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'int'])['year'].count()),
            do_scaling=True
        )

        intervention_results = int_df.fillna(0)
        treatment_dict = dict()

        for treatment in interventions:
            if treatment == 'neo_sep_treat':
                abx = analysis_utility_functions.get_mean_and_quants_from_str_df(
                    intervention_results, 'neo_sep_abx', intervention_years)
                supp = analysis_utility_functions.get_mean_and_quants_from_str_df(
                    intervention_results, 'neo_sep_supportive_care', intervention_years)
                treatment_dict.update({treatment: []})
                for lp in [0, 1, 2]:
                    data = [abx[lp], supp[lp]]
                    treatment_dict[treatment].append(list(map(sum, zip(*data))))
            else:
                treatment_dict.update({treatment: analysis_utility_functions.get_mean_and_quants_from_str_df(
                    intervention_results, treatment, intervention_years)})

        return treatment_dict

    ints = {k: get_total_interventions_delivered(results_folders[k], treatments, intervention_years) for k in
            results_folders}

    def get_crude_complication_numbers(mat_comps, neo_comps, intervention_years):
        crude_comps = dict()

        def sum_lists(list1, list2):
            mean = [x + y for x, y in zip(list1[0], list2[0])]
            lq = [x + y for x, y in zip(list1[1], list2[1])]
            uq = [x + y for x, y in zip(list1[2], list2[2])]

            return [mean, lq, uq]

        # Ectopic
        crude_comps.update({'ectopic': analysis_utility_functions.get_mean_and_quants_from_str_df(
                    mat_comps['pregnancy_supervisor'], 'ectopic_unruptured', intervention_years)})

        # Complicated abortion
        incidence_compsa = analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['pregnancy_supervisor'], 'complicated_spontaneous_abortion', intervention_years)
        incidence_compia = analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['pregnancy_supervisor'], 'complicated_induced_abortion', intervention_years)
        crude_comps.update({'abortion': sum_lists(incidence_compia, incidence_compsa)})

        # Antenatal/Intrapartum Sepsis
        incidence_an_sep = analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['pregnancy_supervisor'], 'clinical_chorioamnionitis', intervention_years)
        incidence_la_sep = analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['labour'], 'sepsis', intervention_years)
        crude_comps.update({'an_ip_sepsis': sum_lists(incidence_an_sep, incidence_la_sep)})

        # Antenatal/Intrapartum Haemorrhage
        incidence_an_haem_mm = analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['pregnancy_supervisor'], 'mild_mod_antepartum_haemorrhage', intervention_years)
        incidence_an_haem_s = analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['pregnancy_supervisor'], 'severe_antepartum_haemorrhage', intervention_years)
        incidence_la_haem_mm = analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['labour'], 'mild_mod_antepartum_haemorrhage', intervention_years)
        incidence_la_haem_s = analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['labour'], 'severe_antepartum_haemorrhage', intervention_years)

        crude_comps.update({'an_ip_haem': []})
        for lp in [0, 1, 2]:
            data = [incidence_an_haem_mm[lp], incidence_an_haem_s[lp], incidence_la_haem_mm[lp],
                    incidence_la_haem_s[lp]]
            crude_comps['an_ip_haem'].append(list(map(sum, zip(*data))))

        # Postpartum Sepsis
        incidence_pn_l_sep = analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['labour'], 'sepsis_postnatal', intervention_years)
        incidence_pn_p_sep = analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['postnatal_supervisor'], 'sepsis', intervention_years)
        crude_comps.update({'pp_sepsis': sum_lists(incidence_pn_l_sep, incidence_pn_p_sep)})

        # PPH - uterine atony
        incidence_ua_pph = analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['postnatal_supervisor'], 'pph_uterine_atony', intervention_years)
        incidence_oth_pph = analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['postnatal_supervisor'], 'pph_other', intervention_years)
        crude_comps.update({'pph_uterine_atony': sum_lists(incidence_ua_pph, incidence_oth_pph)})

        # PPH - retained placenta
        incidence_p_rp = analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['labour'], 'pph_retained_placenta', intervention_years)
        incidence_s_rp = analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['postnatal_supervisor'], 'secondary_postpartum_haemorrhage', intervention_years)
        crude_comps.update({'pph_retained_p': sum_lists(incidence_p_rp, incidence_s_rp)})

        # PPH - requring surgery
        # 43% of uterine atony
        surg_data = list()
        surg_data_rp = list()
        for list_pos in [0, 1, 2]:
            surg_data.append([x * 0.43 for x in crude_comps['pph_uterine_atony'][list_pos]])
            surg_data_rp.append([x * 0.3 for x in crude_comps['pph_retained_p'][list_pos]])

        crude_comps.update({'pph_surg_cases': sum_lists(surg_data, surg_data_rp)})

        # Uterine rupture
        crude_comps.update({'uterine_rupture': analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['labour'], 'uterine_rupture', intervention_years)})

        # Severe pre-eclampsia - antenatal
        crude_comps.update({'spe_an': analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['pregnancy_supervisor'], 'severe_pre_eclamp', intervention_years)})

        # Severe pre-eclampsia - intrapartum
        crude_comps.update({'spe_la': analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['labour'], 'severe_pre_eclamp', intervention_years)})

        # Severe pre-eclampsia - postnatal
        crude_comps.update({'spe_pn': analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['postnatal_supervisor'], 'severe_pre_eclamp', intervention_years)})

        # Severe gestational hypertension - antenatal
        crude_comps.update({'sgh_an': analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['pregnancy_supervisor'], 'severe_gest_htn', intervention_years)})

        # Severe gestational hypertension - intrapartum
        crude_comps.update({'sgh_la': analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['labour'], 'severe_gest_htn', intervention_years)})

        # Severe gestational hypertension - postnatal
        crude_comps.update({'sgh_pn': analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['postnatal_supervisor'], 'severe_gest_htn', intervention_years)})

        # Eclampsia - antenatal
        # Severe pre-eclampsia - antenatal
        crude_comps.update({'ec_an': analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['pregnancy_supervisor'], 'eclampsia', intervention_years)})

        # Severe pre-eclampsia - intrapartum
        crude_comps.update({'ec_la': analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['labour'], 'eclampsia', intervention_years)})

        # Eclampsia - postnatal
        crude_comps.update({'ec_pn': analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['postnatal_supervisor'], 'eclampsia', intervention_years)})

        # Obstructed labour
        crude_comps.update({'obs_labour': analysis_utility_functions.get_mean_and_quants_from_str_df(
            mat_comps['labour'], 'obstructed_labour', intervention_years)})

        ol_cpd = list()
        old_oth = list()
        for list_pos in [0, 1, 2]:
            ol_cpd.append([x * 0.7 for x in crude_comps['obs_labour'][list_pos]])
            old_oth.append([x * 0.3 for x in crude_comps['obs_labour'][list_pos]])
        crude_comps.update({'obs_labour_cpd': ol_cpd})
        crude_comps.update({'obs_labour_other': old_oth})

        # Neonatal sepsis
        incidence_eons_nb = analysis_utility_functions.get_mean_and_quants_from_str_df(
            neo_comps['newborn_outcomes'], 'early_onset_sepsis', intervention_years)
        incidence_eons_pn = analysis_utility_functions.get_mean_and_quants_from_str_df(
            neo_comps['postnatal_supervisor'], 'early_onset_sepsis', intervention_years)
        incidence_lons_pn = analysis_utility_functions.get_mean_and_quants_from_str_df(
            neo_comps['postnatal_supervisor'], 'late_onset_sepsis', intervention_years)

        crude_comps.update({'neo_sepsis': []})
        for lp in [0, 1, 2]:
            data = [incidence_eons_nb[lp], incidence_eons_pn[lp], incidence_lons_pn[lp]]
            crude_comps['neo_sepsis'].append(list(map(sum, zip(*data))))

        # Neonatal respiratory distress
        incidence_prds = analysis_utility_functions.get_mean_and_quants_from_str_df(
            neo_comps['newborn_outcomes'], 'respiratory_distress_syndrome', intervention_years)
        incidence_nbab = analysis_utility_functions.get_mean_and_quants_from_str_df(
            neo_comps['newborn_outcomes'], 'not_breathing_at_birth', intervention_years)

        incidence_enceph_mi = analysis_utility_functions.get_mean_and_quants_from_str_df(
            neo_comps['newborn_outcomes'], 'mild_enceph', intervention_years)
        incidence_enceph_mo = analysis_utility_functions.get_mean_and_quants_from_str_df(
            neo_comps['newborn_outcomes'], 'moderate_enceph', intervention_years)
        incidence_enceph_se = analysis_utility_functions.get_mean_and_quants_from_str_df(
            neo_comps['newborn_outcomes'], 'severe_enceph', intervention_years)

        crude_comps.update({'neo_resp_distress': []})
        for lp in [0, 1, 2]:
            data = [incidence_prds[lp], incidence_nbab[lp], incidence_enceph_mi[lp], incidence_enceph_mo[lp],
                    incidence_enceph_se[lp]]
            crude_comps['neo_resp_distress']. append(list(map(sum, zip(*data))))

        return crude_comps

    comp_numbers = {k: get_crude_complication_numbers(comp_dfs[k], neo_comp_dfs[k],
                                                      intervention_years) for k in results_folders}

    def get_cs_indication_counts(folder):

        cs_df = extract_results(
           folder,
           module="tlo.methods.labour",
           key="cs_indications",
           custom_generate_series=(
               lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'indication'])['id'].count()),
           do_scaling=True)
        cs_results = cs_df.fillna(0)

        cs_id_counts = dict()
        if 'min_sba' in folder.name and (intervention_years[0] not in cs_results.index):
            l = [0 for y in intervention_years]
            blank_data = [l, l, l]
            treatments.append('ur_cs')
            treatments.append('ol_cs')
            treatments.append('aph_cs')

            cs_id_counts.update({'ol_cs':blank_data,
                                 'ur_cs' :blank_data,
                                 'aph_cs':blank_data})

        else:
            for indication in ['ol', 'ur']:  # 'spe_ec', 'other', 'previous_scar'
                cs_id_counts.update({f'{indication}_cs': analysis_utility_functions.get_mean_and_quants_from_str_df(
                    cs_results, indication, intervention_years)})
                treatments.append(f'{indication}_cs')

            pa_cs = analysis_utility_functions.get_mean_and_quants_from_str_df(cs_results, 'an_aph_pa', intervention_years)
            pp_cs = analysis_utility_functions.get_mean_and_quants_from_str_df(cs_results, 'an_aph_pp', intervention_years)
            la_aph_cs = analysis_utility_functions.get_mean_and_quants_from_str_df(cs_results, 'la_aph', intervention_years)

            mean = [a + b + c for a, b, c in zip(pa_cs[0], pp_cs[0], la_aph_cs[0])]
            lq = [a + b + c for a, b, c, in zip(pa_cs[1], pp_cs[1], la_aph_cs[1])]
            uq = [a + b + c for a, b, c in zip(pa_cs[2], pp_cs[2], la_aph_cs[2],)]

            cs_id_counts.update({'aph_cs': [mean, lq, uq]})
            treatments.append('aph_cs')

        return cs_id_counts

    cs_indication = {k: get_cs_indication_counts(results_folders[k]) for k in results_folders}

    def get_met_need(ints, cs_data, crude_comps):

        def update_met_need_dict(comp, treatment, cs):
            if cs:
                iv = cs_data
            else:
                iv = ints

            mean_met_need = [(x / y) * 100 for x, y in zip(iv[treatment][0], crude_comps[comp][0])]
            # Quantiles not used due to lots of 0 values for some treatments
            # lq_mn = [(x / y) * 100 for x, y in zip(iv[treatment][1], crude_comps[comp][1])]
            # uq_mn = [(x / y) * 100 for x, y in zip(iv[treatment][2], crude_comps[comp][2])]
            met_need_dict.update({treatment: mean_met_need})

        met_need_dict = dict()
        comp_and_treatment = {'ectopic': 'ep_case_mang',
                              'abortion': 'pac',
                              'an_ip_sepsis': 'abx_an_sepsis',
                              'uterine_rupture': ['ur_surg', 'blood_tran_ur'],
                              'an_ip_haem': 'blood_tran_aph',
                              'pph_uterine_atony': 'uterotonics',
                              'pph_retained_p': 'man_r_placenta',
                              'pph_surg_cases': ['pph_surg', 'blood_tran_pph'],
                              'pp_sepsis': 'abx_pn_sepsis',
                              'spe_an': ['mag_sulph_an_severe_pre_eclamp', 'iv_htns_an_severe_pre_eclamp'],
                              'spe_la': ['mag_sulph_la_severe_pre_eclamp', 'iv_htns_la_severe_pre_eclamp'],
                              'spe_pn': ['iv_htns_pn_severe_pre_eclamp', 'mag_sulph_pn_severe_pre_eclamp'],
                              'sgh_an': 'iv_htns_an_severe_gest_htn',
                              'sgh_la': 'iv_htns_la_severe_gest_htn',
                              'sgh_pn': 'iv_htns_pn_severe_gest_htn',
                              'ec_an': ['iv_htns_an_eclampsia', 'mag_sulph_an_eclampsia'],
                              'ec_la': ['iv_htns_la_eclampsia', 'mag_sulph_la_eclampsia'],
                              'ec_pn': ['iv_htns_pn_eclampsia', 'mag_sulph_pn_eclampsia'],
                              'obs_labour_other': 'avd_ol',
                              'neo_resp_distress': 'neo_resus',
                              'neo_sepsis': 'neo_sep_treat'}

        for k in comp_and_treatment:
            if not isinstance(comp_and_treatment[k], list):
                update_met_need_dict(k, comp_and_treatment[k], False)
            else:
                update_met_need_dict(k, comp_and_treatment[k][0], False)
                update_met_need_dict(k, comp_and_treatment[k][1], False)

        cs_comp = {'an_ip_haem': 'aph_cs',
                   'uterine_rupture': 'ur_cs',
                   'obs_labour': 'ol_cs'}
        for k in cs_comp:
            update_met_need_dict(k, cs_comp[k], True)

        return met_need_dict

    met_need = {k: get_met_need(ints[k], cs_indication[k], comp_numbers[k]) for k in ints}

    for t in treatments:
        fig, ax = plt.subplots()
        for k, colour in zip(met_need, ['deepskyblue', 'olivedrab', 'darksalmon', 'darkviolet']):
            ax.plot(intervention_years, met_need[k][t], label=k, color=colour)

        plt.ylabel('% of Cases Receiving Treatment')
        plt.xlabel('Year')
        plt.title(f'Met need for {t} Per Year by Scenario')
        plt.gca().set_ylim(bottom=0)
        plt.legend()
        plt.savefig(f'{plot_destination_folder}/{t}.png')
        plt.show()

    met_need_avg = dict()
    for k in met_need:
        met_need_avg.update({k:{}})
        for v in met_need[k]:
            met_need_avg[k][v] = sum(met_need[k][v]) / len(intervention_years)

    labels = results_folders.keys()

    for v in met_need_avg[list(labels)[0]]:

        mean_vals = list()
        for k in met_need_avg:
            mean_vals.append(met_need_avg[k][v])

        width = 0.55  # the width of the bars: can also be len(x) sequence
        fig, ax = plt.subplots()
        ax.bar(labels, mean_vals, width=width)
        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.set_ylabel('Met need%')
        ax.set_xlabel('Scenario')
        ax.set_title(f'Average Met Need for {v}')
        plt.savefig(f'{plot_destination_folder}/{v}_avg.png')
        plt.show()


# ===================================== CONTRIBUTION TO DEATH ========================================================
    factors = ['delay_one_two', 'delay_three', 'didnt_seek_care', 'cons_not_avail', 'comp_not_avail',
               'hcw_not_avail']

    def get_factors_impacting_death(results_folder, factors, intervention_years):
        total_deaths = extract_results(
            results_folder,
            module="tlo.methods.labour.detail",
            key="death_mni",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
        )

        deaths = analysis_utility_functions.get_mean_and_quants(total_deaths, intervention_years)

        factors_prop = dict()
        for factor in factors:
            factor_df = extract_results(
                results_folder,
                module="tlo.methods.labour.detail",
                key="death_mni",
                custom_generate_series=(
                    lambda df: df.loc[df[factor]].assign(
                        year=df['date'].dt.year).groupby(['year', factor])['year'].count()),
                do_scaling=True
            )

            death_factors = factor_df.fillna(0)

            year_means = [death_factors.loc[year, True].mean() for year in intervention_years if year in
                          death_factors.index]
            lower_quantiles = [death_factors.loc[year, True].quantile(0.025) for year in intervention_years if year in
                               death_factors.index]
            upper_quantiles = [death_factors.loc[year, True].quantile(0.925) for year in intervention_years if year in
                               death_factors.index]

            factor_data = [year_means, lower_quantiles, upper_quantiles]

            mean = [(x / y) * 100 for x, y in zip(factor_data[0], deaths[0])]
            lq = [(x / y) * 100 for x, y in zip(factor_data[1], deaths[1])]
            uq = [(x / y) * 100 for x, y in zip(factor_data[2], deaths[2])]

            factors_prop.update({factor: [mean, lq, uq]})

        return factors_prop

    death_causes = {k: get_factors_impacting_death(results_folders[k], factors, intervention_years) for k in
                    results_folders}

    for f in factors:
        fig, ax = plt.subplots()
        for k, colour in zip(death_causes, ['deepskyblue', 'olivedrab', 'darksalmon', 'darkviolet']):
            ax.plot(intervention_years, death_causes[k][f][0], label=k, color=colour)
            ax.fill_between(intervention_years, death_causes[k][f][1], death_causes[k][f][2], color=colour, alpha=.1)

        plt.ylabel('% of total deaths')
        plt.xlabel('Year')
        plt.title(f'Proportion of Total Deaths in which {f} Per Year by Scenario')
        plt.gca().set_ylim(bottom=0)
        plt.legend()
        plt.savefig(f'{plot_destination_folder}/{f}_factor_in_death.png')
        plt.show()

    # todo: proportion of women who died with 0, 1, 2, 3 factors etc?
