import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt

from tlo.analysis.utils import extract_results

plt.style.use('seaborn-darkgrid')

"""This file contains functions used through maternal/perinatal analysis and calibration scripts to extract results,
derive results and generate plots"""


# =========================================== FUNCTIONS TO EXTRACT RATES  ============================================
def return_95_CI_across_runs(df, sim_years):

    year_means = list()
    lower_CI = list()
    upper_CI = list()

    for year in sim_years:
        if year in df.index:
            row = df.loc[year]
            year_means.append(row.mean())
            ci = st.t.interval(0.95, len(row) - 1, loc=np.mean(row), scale=st.sem(row))
            lower_CI.append(ci[0])
            upper_CI.append(ci[1])
        else:
            year_means.append(0)
            lower_CI.append(0)
            upper_CI.append(0)

    return [year_means, lower_CI, upper_CI]


def get_mean_and_quants_from_str_df(df, complication, sim_years):
    yearly_mean_number = list()
    yearly_lq = list()
    yearly_uq = list()
    for year in sim_years:
        if complication in df.loc[year].index:
            yearly_mean_number.append(df.loc[year, complication].mean())
            yearly_lq.append(df.loc[year, complication].quantile(0.025))
            yearly_uq.append(df.loc[year, complication].quantile(0.975))
        else:
            yearly_mean_number.append(0)
            yearly_lq.append(0)
            yearly_uq.append(0)

    return [yearly_mean_number, yearly_lq, yearly_uq]


def get_mean_and_quants_from_list(list_item):
    result = [np.mean(list_item),
              np.quantile(list_item, 0.025),
              np.quantile(list_item, 0.925)]
    return result


def get_mean_95_CI_from_list(list_item):
    ci = st.t.interval(0.95, len(list_item) - 1, loc=np.mean(list_item), scale=st.sem(list_item))
    result = [np.mean(list_item), ci[0], ci[1]]
    return result


def get_mean_from_columns(df, function):
    values = list()
    for col in df:
        if function == 'avg':
            values.append(np.mean(df[col]))
        else:
            values.append(sum(df[col]))
    return values


def get_comp_mean_and_rate(complication, denominator_list, df, rate, years):
    yearly_means = get_mean_and_quants_from_str_df(df, complication, years)[0]
    yearly_lq = get_mean_and_quants_from_str_df(df, complication, years)[1]
    yearly_uq = get_mean_and_quants_from_str_df(df, complication, years)[2]

    yearly_mean_rate = [(x / y) * rate for x, y in zip(yearly_means, denominator_list)]
    yearly_lq_rate = [(x / y) * rate for x, y in zip(yearly_lq, denominator_list)]
    yearly_uq_rate = [(x / y) * rate for x, y in zip(yearly_uq, denominator_list)]

    return [yearly_mean_rate, yearly_lq_rate, yearly_uq_rate]


def get_mean_and_quants(df, sim_years):
    year_means = list()
    lower_quantiles = list()
    upper_quantiles = list()

    for year in sim_years:
        if year in df.index:
            year_means.append(df.loc[year].mean())
            lower_quantiles.append(df.loc[year].quantile(0.025))
            upper_quantiles.append(df.loc[year].quantile(0.975))
        else:
            year_means.append(0)
            lower_quantiles.append(0)
            upper_quantiles.append(0)

    return [year_means, lower_quantiles, upper_quantiles]


def get_comp_mean_and_rate_across_multiple_dataframes(complication, denominators, rate, dataframes, sim_years):
    def get_list_of_rates_and_quants(df):
        rates_per_year = list()
        lq_per_year = list()
        uq_per_year = list()
        for year, denominator in zip(sim_years, denominators):
            if year in df.index:
                if complication in df.loc[year].index:
                    rates = (df.loc[year, complication].mean() / denominator) * rate
                    lq = (df.loc[year, complication].quantile(0.025) / denominator) * rate
                    uq = (df.loc[year, complication].quantile(0.975) / denominator) * rate
                    rates_per_year.append(rates)
                    lq_per_year.append(lq)
                    uq_per_year.append(uq)

                else:
                    rates_per_year.append(0)
                    lq_per_year.append(0)
                    uq_per_year.append(0)
            else:
                rates_per_year.append(0)
                lq_per_year.append(0)
                uq_per_year.append(0)

        return [rates_per_year, lq_per_year, uq_per_year]

    if len(dataframes) == 2:
        df_1_data = get_list_of_rates_and_quants(dataframes[0])
        df_2_data = get_list_of_rates_and_quants(dataframes[1])

        total_rates = [x + y for x, y in zip(df_1_data[0], df_2_data[0])]
        total_lq = [x + y for x, y in zip(df_1_data[1], df_2_data[1])]
        total_uq = [x + y for x, y in zip(df_1_data[2], df_2_data[2])]

    else:
        df_1_data = get_list_of_rates_and_quants(dataframes[0])
        df_2_data = get_list_of_rates_and_quants(dataframes[1])
        df_3_data = get_list_of_rates_and_quants(dataframes[2])

        total_rates = [x + y + z for x, y, z in zip(df_1_data[0], df_2_data[0], df_3_data[0])]
        total_lq = [x + y + z for x, y, z in zip(df_1_data[1], df_2_data[1], df_3_data[1])]
        total_uq = [x + y + z for x, y, z in zip(df_1_data[2], df_2_data[2], df_3_data[2])]

    return [total_rates, total_lq, total_uq]


def get_modules_maternal_complication_dataframes(results_folder):
    comp_dfs = dict()

    for module in ['pregnancy_supervisor', 'labour', 'postnatal_supervisor']:
        c_df = extract_results(
            results_folder,
            module=f"tlo.methods.{module}",
            key="maternal_complication",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['person'].count()),
            do_scaling=True
        )
        complications_df = c_df.fillna(0)

        comp_dfs[module] = complications_df

    return comp_dfs


def get_modules_neonatal_complication_dataframes(results_folder):
    comp_dfs = dict()

    for module in ['newborn_outcomes', 'postnatal_supervisor']:
        n_df = extract_results(
            results_folder,
            module=f"tlo.methods.{module}",
            key="newborn_complication",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['newborn'].count()),
            do_scaling=True
        )
        complications_df = n_df.fillna(0)

        comp_dfs[module] = complications_df

    return comp_dfs


def extract_comp_inc_folders(folder, comps_df, neo_comps_df, pregnancy_df, births_df, comp_preg_df):
    def get_rate_df(comp_df, denom_df, denom):
        rate = (comp_df / denom_df) * denom
        return rate

    results = dict()

    # MATERNAL
    eu = comps_df['pregnancy_supervisor'].loc[(slice(None), 'ectopic_unruptured'), slice(None)].droplevel(1)
    results.update({'eu_rate': get_rate_df(eu, pregnancy_df, 1000)})

    sa = comps_df['pregnancy_supervisor'].loc[(slice(None), 'spontaneous_abortion'), slice(None)].droplevel(1)
    results.update({'sa_rate': get_rate_df(sa, comp_preg_df, 1000)})

    ia = comps_df['pregnancy_supervisor'].loc[(slice(None), 'induced_abortion'), slice(None)].droplevel(1)
    results.update({'ia_rate': get_rate_df(ia, comp_preg_df, 1000)})

    syph = comps_df['pregnancy_supervisor'].loc[(slice(None), 'syphilis'), slice(None)].droplevel(1)
    results.update({'syph_rate': get_rate_df(syph, comp_preg_df, 1000)})

    gdm = comps_df['pregnancy_supervisor'].loc[(slice(None), 'gest_diab'), slice(None)].droplevel(1)
    results.update({'gdm_rate': get_rate_df(gdm, comp_preg_df, 1000)})

    prom = comps_df['pregnancy_supervisor'].loc[(slice(None), 'PROM'), slice(None)].droplevel(1)
    results.update({'prom_rate': get_rate_df(prom, births_df, 1000)})

    praevia = comps_df['pregnancy_supervisor'].loc[(slice(None), 'placenta_praevia'), slice(None)].droplevel(1)
    results.update({'praevia_rate': get_rate_df(praevia, pregnancy_df, 1000)})

    abruption = comps_df['pregnancy_supervisor'].loc[(slice(None), 'placental_abruption'), slice(None)].droplevel(1)
    results.update({'abruption_rate': get_rate_df(abruption, births_df, 1000)})

    potl = comps_df['labour'].loc[(slice(None), 'post_term_labour'), slice(None)].droplevel(1)
    results.update({'potl_rate': get_rate_df(potl, pregnancy_df, 100)})

    ol = comps_df['labour'].loc[(slice(None), 'obstructed_labour'), slice(None)].droplevel(1)
    results.update({'ol_rate': get_rate_df(ol, births_df, 1000)})

    ur = comps_df['labour'].loc[(slice(None), 'uterine_rupture'), slice(None)].droplevel(1)
    results.update({'ur_rate': get_rate_df(ur, births_df, 1000)})

    gh_an = comps_df['pregnancy_supervisor'].loc[(slice(None), 'mild_gest_htn'), slice(None)].droplevel(1)
    gh_pn = comps_df['postnatal_supervisor'].loc[(slice(None), 'mild_gest_htn'), slice(None)].droplevel(1)
    results.update({'gh_rate': get_rate_df((gh_an + gh_pn), births_df, 1000)})

    mpe_an = comps_df['pregnancy_supervisor'].loc[(slice(None), 'mild_pre_eclamp'), slice(None)].droplevel(1)
    mpe_pn = comps_df['postnatal_supervisor'].loc[(slice(None), 'mild_pre_eclamp'), slice(None)].droplevel(1)
    results.update({'mpe_rate': get_rate_df((mpe_an + mpe_pn), births_df, 1000)})

    sgh_an = comps_df['pregnancy_supervisor'].loc[(slice(None), 'severe_gest_htn'), slice(None)].droplevel(1)
    sgh_la = comps_df['labour'].loc[(slice(None), 'severe_gest_htn'), slice(None)].droplevel(1)
    sgh_pn = comps_df['postnatal_supervisor'].loc[(slice(None), 'severe_gest_htn'), slice(None)].droplevel(1)
    results.update({'sgh_rate': get_rate_df((sgh_an + sgh_la + sgh_pn), births_df, 1000)})

    spe_an = comps_df['pregnancy_supervisor'].loc[(slice(None), 'severe_pre_eclamp'), slice(None)].droplevel(1)
    spe_la = comps_df['labour'].loc[(slice(None), 'severe_pre_eclamp'), slice(None)].droplevel(1)
    spe_pn = comps_df['postnatal_supervisor'].loc[(slice(None), 'severe_pre_eclamp'), slice(None)].droplevel(1)
    results.update({'spe_rate': get_rate_df((spe_an + spe_la + spe_pn), births_df, 1000)})

    ec_an = comps_df['pregnancy_supervisor'].loc[(slice(None), 'eclampsia'), slice(None)].droplevel(1)
    ec_la = comps_df['labour'].loc[(slice(None), 'eclampsia'), slice(None)].droplevel(1)
    ec_pn = comps_df['postnatal_supervisor'].loc[(slice(None), 'eclampsia'), slice(None)].droplevel(1)
    results.update({'ec_rate': get_rate_df((ec_an + ec_la + ec_pn), births_df, 1000)})

    m_aph_ps = comps_df['pregnancy_supervisor'].loc[(slice(None), 'mild_mod_antepartum_haemorrhage'
                                                     ), slice(None)].droplevel(1)
    m_aph_la = comps_df['labour'].loc[(slice(None), 'mild_mod_antepartum_haemorrhage'), slice(None)].droplevel(1)
    s_aph_ps = comps_df['pregnancy_supervisor'].loc[
        (slice(None), 'severe_antepartum_haemorrhage'), slice(None)].droplevel(1)
    s_aph_la = comps_df['labour'].loc[
        (slice(None), 'severe_antepartum_haemorrhage'), slice(None)].droplevel(1)
    results.update({'aph_rate': get_rate_df((m_aph_ps + m_aph_la + s_aph_la + s_aph_ps), births_df, 1000)})

    e_ptl = comps_df['labour'].loc[(slice(None), 'early_preterm_labour'), slice(None)].droplevel(1)
    l_ptl = comps_df['labour'].loc[(slice(None), 'late_preterm_labour'), slice(None)].droplevel(1)
    results.update({'ptl_rate': get_rate_df((e_ptl + l_ptl), births_df, 100)})

    an_sep = comps_df['pregnancy_supervisor'].loc[(slice(None), 'clinical_chorioamnionitis'), slice(None)].droplevel(1)
    la_sep = comps_df['labour'].loc[(slice(None), 'sepsis'), slice(None)].droplevel(1)
    pn_la_sep = comps_df['postnatal_supervisor'].loc[(slice(None), 'sepsis_postnatal'), slice(None)].droplevel(1)
    pn_sep = comps_df['postnatal_supervisor'].loc[(slice(None), 'sepsis'), slice(None)].droplevel(1)
    results.update({'sep_rate': get_rate_df((an_sep + la_sep + pn_la_sep + pn_sep), births_df, 1000)})

    l_pph = comps_df['postnatal_supervisor'].loc[(slice(None), 'primary_postpartum_haemorrhage'),
    slice(None)].droplevel(1)
    p_pph = comps_df['postnatal_supervisor'].loc[(slice(None), 'secondary_postpartum_haemorrhage'),
    slice(None)].droplevel(1)
    results.update({'pph_rate': get_rate_df((l_pph + p_pph), births_df, 1000)})

    anaemia_results = extract_results(
        folder,
        module="tlo.methods.pregnancy_supervisor",
        key="conditions_on_birth",
        custom_generate_series=(
            lambda df: df.loc[df['anaemia_status'] != 'none'].assign(year=df['date'].dt.year
                                                                     ).groupby(['year'])['year'].count()),
        do_scaling=True
    )
    results.update({'an_ps_prev': get_rate_df(anaemia_results, births_df, 100)})

    pnc_anaemia = extract_results(
        folder,
        module="tlo.methods.postnatal_supervisor",
        key="total_mat_pnc_visits",
        custom_generate_series=(
            lambda df_: df_.loc[df_['anaemia'] != 'none'].assign(year=df_['date'].dt.year
                                                                 ).groupby(['year'])['mother'].count()),
        do_scaling=True
    )
    results.update({'an_pn_prev': get_rate_df(pnc_anaemia, births_df, 100)})

    # NEWBORNS
    macro = neo_comps_df['newborn_outcomes'].loc[(slice(None), 'macrosomia'), slice(None)].droplevel(1)
    results.update({'macro_rate': get_rate_df(macro, births_df, 100)})

    sga = neo_comps_df['newborn_outcomes'].loc[(slice(None), 'small_for_gestational_age'), slice(None)].droplevel(1)
    results.update({'sga_rate': get_rate_df(sga, births_df, 100)})

    resp_distress = neo_comps_df['newborn_outcomes'].loc[(slice(None), 'not_breathing_at_birth'),
    slice(None)].droplevel(1)
    results.update({'rd_rate': get_rate_df(resp_distress, births_df, 1000)})

    rds = neo_comps_df['newborn_outcomes'].loc[(slice(None), 'respiratory_distress_syndrome'),
    slice(None)].droplevel(1)
    results.update({'rds_rate': get_rate_df(rds, births_df, 1000)})

    eons_n = neo_comps_df['newborn_outcomes'].loc[(slice(None), 'early_onset_sepsis'), slice(None)].droplevel(1)
    eons_pn = neo_comps_df['postnatal_supervisor'].loc[(slice(None), 'early_onset_sepsis'), slice(None)].droplevel(1)
    lons = neo_comps_df['postnatal_supervisor'].loc[(slice(None), 'late_onset_sepsis'), slice(None)].droplevel(1)
    results.update({'neo_sep_rate': get_rate_df((eons_n + eons_pn + lons), births_df, 1000)})

    m_enc = neo_comps_df['newborn_outcomes'].loc[(slice(None), 'mild_enceph'), slice(None)].droplevel(1)
    mo_enc = neo_comps_df['newborn_outcomes'].loc[(slice(None), 'moderate_enceph'), slice(None)].droplevel(1)
    s_enc = neo_comps_df['newborn_outcomes'].loc[(slice(None), 'severe_enceph'), slice(None)].droplevel(1)
    results.update({'enc_rate': get_rate_df((m_enc + mo_enc + s_enc), births_df, 1000)})

    return results


def line_graph_with_ci_and_target_rate(sim_years, data, target_data_dict, y_label, title,
                                       graph_location, file_name):
    fig, ax = plt.subplots()
    ax.plot(sim_years, data[0], label="Model", color='deepskyblue')
    ax.fill_between(sim_years, data[1], data[2], label="95% CI", color='b', alpha=.1)

    if target_data_dict['double']:
        plt.errorbar(target_data_dict['first']['year'], target_data_dict['first']['value'],
                     label=target_data_dict['first']['label'], yerr=target_data_dict['first']['ci'],
                     fmt='o', color='darkseagreen', ecolor='green', elinewidth=3, capsize=0)
        plt.errorbar(target_data_dict['second']['year'], target_data_dict['second']['value'],
                     label=target_data_dict['second']['label'], yerr=target_data_dict['second']['ci'],
                     fmt='o', color='darkseagreen', ecolor='green', elinewidth=3, capsize=0)

    elif not target_data_dict['double']:
        plt.errorbar(target_data_dict['first']['year'], target_data_dict['first']['value'],
                     label=target_data_dict['first']['label'], yerr=target_data_dict['first']['ci'],
                     fmt='o', color='darkseagreen', ecolor='green', elinewidth=3, capsize=0)

    plt.xlabel('Year')
    plt.ylabel(y_label)
    plt.title(title)

    if ('anc_prop' in file_name) or ('sba' in file_name) or ('pnc' in file_name):
        ax.set(ylim=(0, 100))
    if 'caesarean' in file_name:
        ax.set(ylim=(0, 10))

    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.savefig(f'{graph_location}/{file_name}.png')
    plt.show()


def basic_comparison_graph(intervention_years, bdata, idata, x_label, title, graph_location, save_name):
    fig, ax = plt.subplots()
    ax.plot(intervention_years, bdata[0], label="Baseline (mean)", color='deepskyblue')
    ax.fill_between(intervention_years, bdata[1], bdata[2], color='b', alpha=.1, label="UI (2.5-97.5)")
    ax.plot(intervention_years, idata[0], label="Intervention (mean)", color='olivedrab')
    ax.fill_between(intervention_years, idata[1], idata[2], color='g', alpha=.1, label="UI (2.5-97.5)")
    plt.ylabel('Year')
    plt.xlabel(x_label)
    plt.title(title)
    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.savefig(f'{graph_location}/{save_name}.png')
    plt.show()


def simple_line_chart(sim_years, model_rate, y_title, title, file_name, graph_location):
    plt.plot(sim_years, model_rate, 'o-g', label="Model", color='deepskyblue')
    plt.ylabel(y_title)
    plt.xlabel('Year')
    plt.title(title)
    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.savefig(f'{graph_location}/{file_name}.png')
    plt.show()


def simple_line_chart_with_target(sim_years, model_rate, target_rate, y_title, title, file_name, graph_location):
    plt.plot(sim_years, model_rate, 'o-g', label="Model", color='deepskyblue')
    plt.plot(sim_years, target_rate, 'o-g', label="Target rate", color='darkseagreen')
    plt.ylabel(y_title)
    plt.xlabel('Year')
    plt.title(title)
    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.savefig(f'{graph_location}/{file_name}.png')
    plt.show()


def simple_line_chart_with_ci(sim_years, data, y_title, title, file_name, graph_location):
    fig, ax = plt.subplots()
    ax.plot(sim_years, data[0], label="Model (mean)", color='deepskyblue')
    ax.fill_between(sim_years, data[1], data[2], color='b', alpha=.1, label="95% CI")
    plt.ylabel(y_title)
    plt.xlabel('Year')
    plt.title(title)
    plt.legend()
    plt.gca().set_ylim(bottom=0)
    plt.grid(True)
    plt.savefig(f'{graph_location}/{file_name}.png')
    plt.show()


def simple_bar_chart(model_rates, x_title, y_title, title, file_name, sim_years, graph_location):
    bars = sim_years
    x_pos = np.arange(len(bars))
    plt.bar(x_pos, model_rates, label="Model", color='thistle')
    plt.xticks(x_pos, bars, rotation=90)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.legend()
    plt.savefig(f'{graph_location}/{file_name}.png')
    plt.show()


def return_median_and_mean_squeeze_factor_for_hsi(folder, hsi_string, sim_years, graph_location):
    hsi_med = extract_results(
        folder,
        module="tlo.methods.healthsystem",
        key="HSI_Event",
        custom_generate_series=(
            lambda df: df.loc[df['TREATMENT_ID'].str.contains(hsi_string) & df['did_run']].assign(
                year=df['date'].dt.year).groupby(['year'])['Squeeze_Factor'].median()))

    hsi_mean = extract_results(
        folder,
        module="tlo.methods.healthsystem",
        key="HSI_Event",
        custom_generate_series=(
            lambda df: df.loc[df['TREATMENT_ID'].str.contains(hsi_string) & df['did_run']].assign(
                year=df['date'].dt.year).groupby(['year'])['Squeeze_Factor'].mean()))

    median = [hsi_med.loc[year].median() for year in sim_years]
    lq = [hsi_med.loc[year].quantile(0.025) for year in sim_years]
    uq = [hsi_med.loc[year].quantile(0.975) for year in sim_years]
    data = [median, lq, uq]

    simple_line_chart_with_ci(sim_years, data, 'Median Squeeze Factor', f'Median Yearly Squeeze for HSI {hsi_string}',
                              f'median_sf_{hsi_string}', graph_location)

    mean = [hsi_mean.loc[year].mean() for year in sim_years]
    lq = [hsi_mean.loc[year].quantile(0.025) for year in sim_years]
    uq = [hsi_mean.loc[year].quantile(0.975) for year in sim_years]
    data = [mean, lq, uq]

    simple_line_chart_with_ci(sim_years, data, 'Mean Squeeze Factor', f'Mean Yearly Squeeze for HSI {hsi_string}',
                              f'mean_sf_{hsi_string}', graph_location)


def return_squeeze_plots_for_hsi(folder, hsi_string, sim_years, graph_location):
    hsi = extract_results(
        folder,
        module="tlo.methods.healthsystem",
        key="HSI_Event",
        custom_generate_series=(
            lambda df: df.loc[df['TREATMENT_ID'].str.contains(hsi_string) & df['did_run']].assign(
                year=df['date'].dt.year).groupby(['year'])['Squeeze_Factor'].mean()))

    mean_squeeze_per_year = [hsi.loc[year].to_numpy().mean() for year in sim_years]
    lq_squeeze_per_year = [np.percentile(hsi.loc[year].to_numpy(), 2.5) for year in sim_years]
    uq_squeeze_per_year = [np.percentile(hsi.loc[year].to_numpy(), 97.5) for year in sim_years]
    mean_data = [mean_squeeze_per_year, lq_squeeze_per_year, uq_squeeze_per_year]

    hsi_med = extract_results(
        folder,
        module="tlo.methods.healthsystem",
        key="HSI_Event",
        custom_generate_series=(
            lambda df: df.loc[df['TREATMENT_ID'].str.contains(hsi_string) & df['did_run']].assign(
                year=df['date'].dt.year).groupby(['year'])['Squeeze_Factor'].median()))

    median = [hsi_med.loc[year].median() for year in sim_years]

    hsi_count = extract_results(
        folder,
        module="tlo.methods.healthsystem",
        key="HSI_Event",
        custom_generate_series=(
            lambda df: df.loc[df['TREATMENT_ID'].str.contains(hsi_string) & df['did_run']].assign(
                year=df['date'].dt.year).groupby(['year'])['year'].count()))

    hsi_squeeze = extract_results(
        folder,
        module="tlo.methods.healthsystem",
        key="HSI_Event",
        custom_generate_series=(
            lambda df:
            df.loc[(df['TREATMENT_ID'].str.contains(hsi_string)) & df['did_run'] & (df['Squeeze_Factor'] > 0)
                   ].assign(year=df['date'].dt.year).groupby(['year'])['year'].count()))

    prop_squeeze_year = [(hsi_squeeze.loc[year].to_numpy().mean() / hsi_count.loc[year].to_numpy().mean()) * 100
                         for year in sim_years]
    prop_squeeze_lq = [
        (np.percentile(hsi_squeeze.loc[year].to_numpy(), 2.5) /
         np.percentile(hsi_count.loc[year].to_numpy(), 2.5)) * 100 for year in sim_years]

    prop_squeeze_uq = [
        (np.percentile(hsi_squeeze.loc[year].to_numpy(), 97.5) /
         np.percentile(hsi_count.loc[year].to_numpy(), 97.5)) * 100 for year in sim_years]

    prop_data = [prop_squeeze_year, prop_squeeze_lq, prop_squeeze_uq]

    simple_line_chart_with_ci(sim_years, mean_data, 'Mean Squeeze Factor', f'Mean Yearly Squeeze for HSI {hsi_string}',
                              f'mean_sf_{hsi_string}', graph_location)
    simple_line_chart(sim_years, median, 'Median Squeeze Factor', f'Median Yearly Squeeze for HSI {hsi_string}',
                      f'med_sf_{hsi_string}', graph_location)
    simple_line_chart_with_ci(sim_years, prop_data, '% HSIs', f'Proportion of HSI {hsi_string} where squeeze > 0',
                              f'prop_sf_{hsi_string}', graph_location)


def comparison_graph_multiple_scenarios(colours, intervention_years, data_dict, y_label, title, graph_location,
                                        save_name):
    fig, ax = plt.subplots()

    for k, colour in zip(data_dict, colours):
        ax.plot(intervention_years, data_dict[k][0], label=k, color=colour)
        ax.fill_between(intervention_years, data_dict[k][1], data_dict[k][2], color=colour, alpha=.1)

    plt.ylabel(y_label)
    plt.xlabel('Year')
    plt.title(title)
    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.savefig(f'./{graph_location}/{save_name}.png')
    plt.show()


def comparison_graph_multiple_scenarios_multi_level_dict(colours, intervention_years, data_dict, key, y_label, title,
                                                         graph_location, save_name):
    fig, ax = plt.subplots()

    for k, colour in zip(data_dict, colours):
        ax.plot(intervention_years, data_dict[k][key][0], label=k, color=colour)
        ax.fill_between(intervention_years, data_dict[k][key][1], data_dict[k][key][2], color=colour, alpha=.1)

    plt.ylabel(y_label)
    plt.xlabel('Year')
    plt.title(title)
    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.savefig(f'./{graph_location}/{save_name}.png')
    plt.show()


def comparison_bar_chart_multiple_bars(data, dict_name, intervention_years, colours, y_title, title,
                                       plot_destination_folder, save_name):
    N = len(intervention_years)
    ind = np.arange(N)
    if len(data.keys()) > 3:
        width = 0.15
    else:
        width = 0.2

    x_ticks = list()
    for x in range(len(intervention_years)):
        x_ticks.append(x)

    for k, position, colour in zip(data, [ind - width, ind, ind + width, ind + width * 2, ind + width * 3],
                                   colours):
        ci = [(x - y) / 2 for x, y in zip(data[k][dict_name][2], data[k][dict_name][1])]
        plt.bar(position, data[k][dict_name][0], width, label=k, yerr=ci, color=colour)

    plt.ylabel(y_title)
    plt.xlabel('Years')
    plt.title(title)
    plt.legend(loc='best')
    plt.xticks(x_ticks, labels=intervention_years)
    plt.savefig(f'{plot_destination_folder}/{save_name}.png')
    plt.show()


# =========================== FUNCTIONS RETURNING DATA FROM MULTIPLE SCENARIOS =======================================
def return_birth_data_from_multiple_scenarios(results_folders, sim_years, intervention_years):
    """
    Extract mean, lower and upper quantile births per year for a given scenario
    :param folder: results folder for scenario
    :return: list of total births per year of pre defined intervention period (i.e. 2020-2030)
    """

    def extract_births(folder):
        br = extract_results(
            folder,
            module="tlo.methods.demography",
            key="on_birth",
            custom_generate_series=(
                lambda df: df.assign(
                    year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
        )
        births_results = br.fillna(0)
        # total_births_per_year = get_mean_and_quants(births_results, sim_years)
        total_births_per_year = return_95_CI_across_runs(births_results, sim_years)

        int_df = births_results.loc[intervention_years[0]: intervention_years[-1]]
        int_births_per_year = get_mean_and_quants(int_df, intervention_years)

        agg_births_data = get_mean_from_columns(births_results.loc[intervention_years[0]: intervention_years[-1]],
                                                'agg')
        agg_births = get_mean_95_CI_from_list(agg_births_data)

        # agg_births = [np.mean(agg_births_data),
        #               np.quantile(agg_births_data, 0.025),
        #               np.quantile(agg_births_data, 0.975)]

        return {'total_births': total_births_per_year,
                'int_births': int_births_per_year,
                'agg_births': agg_births,
                'births_data_frame': births_results}

    return {k: extract_births(results_folders[k]) for k in results_folders}


def return_pregnancy_data_from_multiple_scenarios(results_folders, sim_years, intervention_years):
    """
    """

    def extract_pregnancies(folder):
        pr = extract_results(
            folder,
            module="tlo.methods.contraception",
            key="pregnancy",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
        )
        preg_results = pr.fillna(0)
        # total_pregnancies_per_year = get_mean_and_quants(preg_results, sim_years)
        # total_pregnancies_per_year_int = get_mean_and_quants(
        #     preg_results.loc[intervention_years[0]: intervention_years[-1]], intervention_years)

        total_pregnancies_per_year = return_95_CI_across_runs(preg_results, sim_years)
        total_pregnancies_per_year_int = return_95_CI_across_runs(
             preg_results.loc[intervention_years[0]: intervention_years[-1]], intervention_years)

        agg_preg_data = get_mean_from_columns(preg_results.loc[intervention_years[0]: intervention_years[-1]], 'agg')
        agg_preg = get_mean_95_CI_from_list(agg_preg_data)

        # agg_preg = [np.mean(agg_preg_data),
        #             np.quantile(agg_preg_data, 0.025),
        #             np.quantile(agg_preg_data, 0.975)]

        return {'total_preg': total_pregnancies_per_year,
                'int_preg': total_pregnancies_per_year_int,
                'agg_preg': agg_preg,
                'preg_data_frame': preg_results}

    return {k: extract_pregnancies(results_folders[k]) for k in results_folders}


def get_completed_pregnancies_from_multiple_scenarios(comps_df, birth_dict, results_folder, sim_years,
                                                      intervention_years):
    """Sums the number of pregnancies that have ended in a given year including ectopic pregnancies,
    abortions, stillbirths and births"""

    def get_comps_p_all_or_int(years, birth_key):
        ectopic_mean_numbers_per_year = get_mean_and_quants_from_str_df(
            comps_df['pregnancy_supervisor'], 'ectopic_unruptured', years)[0]

        ia_mean_numbers_per_year = get_mean_and_quants_from_str_df(
            comps_df['pregnancy_supervisor'], 'induced_abortion', years)[0]

        sa_mean_numbers_per_year = get_mean_and_quants_from_str_df(
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

        an_still_birth_data = get_mean_and_quants(an_stillbirth_results, years)

        total_completed_pregnancies_per_year = [a + b + c + d + e for a, b, c, d, e in
                                                zip(birth_dict[birth_key][0], ectopic_mean_numbers_per_year,
                                                    ia_mean_numbers_per_year, sa_mean_numbers_per_year,
                                                    an_still_birth_data[0])]
        return total_completed_pregnancies_per_year

    tcp_all = get_comps_p_all_or_int(sim_years, 'total_births')
    tcp_int = get_comps_p_all_or_int(intervention_years, 'int_births')

    ansb_df = extract_results(
        results_folder,
        module="tlo.methods.pregnancy_supervisor",
        key="antenatal_stillbirth",
        custom_generate_series=(
            lambda df: df.assign(
                year=df['date'].dt.year).groupby(['year'])['year'].count()),
        do_scaling=True
    )

    eu = comps_df['pregnancy_supervisor'].loc[(slice(None), 'ectopic_unruptured'), slice(None)]
    eu_f = eu.droplevel(1)

    ia = comps_df['pregnancy_supervisor'].loc[(slice(None), 'induced_abortion'), slice(None)]
    ia_f = ia.droplevel(1)

    sa = comps_df['pregnancy_supervisor'].loc[(slice(None), 'spontaneous_abortion'), slice(None)]
    sa_f = sa.droplevel(1)

    comp_preg_data_frame = eu_f + ia_f + sa_f + ansb_df + birth_dict['births_data_frame']

    return {'total_cp': tcp_all,
            'int_cp': tcp_int,
            'comp_preg_data_frame': comp_preg_data_frame}


def return_death_data_from_multiple_scenarios(results_folders, births_dict, sim_years, intervention_years):
    """
    Extract mean, lower and upper quantile maternal mortality ratio, neonatal mortality ratio, crude maternal
    deaths and crude neonatal deaths per year for a given scenario
    :param folder: results folder for scenario
    :param births: list. mean number of births per year for a scenario (used as a denominator)
    :return: dict containing mean, LQ, UQ for MMR, NMR, maternal deaths and neonatal deaths
    """

    def extract_deaths(folder, births_df):
        # Get full death dataframe
        dd = extract_results(
            folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=(
                lambda df: df.loc[(df['label'] == 'Maternal Disorders')].assign(
                    year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True)
        direct_deaths = dd.fillna(0)

        ind_d = extract_results(
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
        indirect_deaths = ind_d.fillna(0)

        # TOTAL MMR BY YEAR
        total_deaths = direct_deaths + indirect_deaths
        mmr_df = (total_deaths / births_df) * 100_000
        total_mmr_by_year = get_mean_and_quants(mmr_df, sim_years)

        # TOTAL AVERAGE MMR DURING INTERVENTION
        mmr_df_int = mmr_df.loc[intervention_years[0]: intervention_years[-1]]
        mean_mmr_by_year_int = get_mean_from_columns(mmr_df_int, 'avg')
        total_mmr_aggregated = get_mean_and_quants_from_list(mean_mmr_by_year_int)

        # TOTAL MATERNAL DEATHS BY YEAR
        td_int = total_deaths.loc[intervention_years[0]: intervention_years[-1]]
        mean_total_deaths_by_year = get_mean_and_quants(total_deaths, sim_years)
        mean_total_deaths_by_year_int = get_mean_and_quants(td_int, intervention_years)

        # TOTAL MATERNAL DEATHS DURING INTERVENTION PERIOD
        sum_mat_death_int_by_run = get_mean_from_columns(td_int, 'sum')
        total_deaths_by_scenario = get_mean_and_quants_from_list(sum_mat_death_int_by_run)

        # DIRECT MATERNAL DEATHS PER YEAR
        mean_direct_deaths_by_year = get_mean_and_quants(direct_deaths, sim_years)
        mean_direct_deaths_by_year_int = get_mean_and_quants(direct_deaths, intervention_years)

        # TOTAL DIRECT MATERNAL DEATHS DURING INTERVENTION
        dd_int = direct_deaths.loc[intervention_years[0]: intervention_years[-1]]
        sum_d_mat_death_int_by_run = get_mean_from_columns(dd_int, 'sum')
        total_direct_deaths_by_scenario = get_mean_and_quants_from_list(sum_d_mat_death_int_by_run)

        # DIRECT MMR BY YEAR
        d_mmr_df = (direct_deaths / births_df) * 100_000
        total_direct_mmr_by_year = get_mean_and_quants(d_mmr_df, sim_years)

        # AVERAGE DIRECT MMR DURING INTERVENTION
        d_mmr_df_int = d_mmr_df.loc[intervention_years[0]: intervention_years[-1]]
        mean_d_mmr_by_year_int = get_mean_from_columns(d_mmr_df_int, 'avg')
        total_direct_mmr_aggregated = get_mean_and_quants_from_list(mean_d_mmr_by_year_int)

        # INDIRECT MATERNAL DEATHS PER YEAR
        mean_indirect_deaths_by_year = get_mean_and_quants(indirect_deaths, sim_years)
        mean_indirect_deaths_by_year_int = get_mean_and_quants(indirect_deaths, intervention_years)

        # TOTAL INDIRECT MATERNAL DEATHS DURING INTERVENTION
        in_int = indirect_deaths.loc[intervention_years[0]: intervention_years[-1]]
        sum_in_mat_death_int_by_run = get_mean_from_columns(in_int, 'sum')
        total_indirect_deaths_by_scenario = get_mean_and_quants_from_list(sum_in_mat_death_int_by_run)

        # INDIRECT MMR BY YEAR
        in_mmr_df = (indirect_deaths / births_df) * 100_000
        total_indirect_mmr_by_year = get_mean_and_quants(in_mmr_df, sim_years)

        # AVERAGE INDIRECT MMR DURING INTERVENTION
        in_mmr_df_int = in_mmr_df.loc[intervention_years[0]: intervention_years[-1]]
        mean_in_mmr_by_year_int = get_mean_from_columns(in_mmr_df_int, 'avg')
        total_indirect_mmr_aggregated = get_mean_and_quants_from_list(mean_in_mmr_by_year_int)

        # NEONATAL DEATHS
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

        # TOTAL NMR PER YEAR
        nmr_df = (neo_deaths / births_df) * 1000
        total_nmr_by_year = get_mean_and_quants(nmr_df, sim_years)

        # AVERAGE NMR DURING INTERVENTION PERIOD
        nmr_df = nmr_df.loc[intervention_years[0]: intervention_years[-1]]
        mean_nmr_by_year_int = get_mean_from_columns(nmr_df, 'avg')
        total_nmr_aggregated = get_mean_and_quants_from_list(mean_nmr_by_year_int)

        # NEONATAL DEATHS PER YEAR
        mean_neonatal_deaths_by_year = get_mean_and_quants(neo_deaths, sim_years)
        mean_neonatal_deaths_by_year_int = get_mean_and_quants(neo_deaths_int, intervention_years)

        # TOTAL NEONATAL DEATHS DURING INTERVENTION PERIOD
        sum_neo_death_int_by_run = get_mean_from_columns(neo_deaths_int, 'sum')
        total_neonatal_deaths_by_scenario = get_mean_and_quants_from_list(sum_neo_death_int_by_run)

        return {'crude_t_deaths': mean_total_deaths_by_year,
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
                'agg_nmr': total_nmr_aggregated}

    # Extract data from scenarios
    return {k: extract_deaths(results_folders[k], births_dict[k]['births_data_frame']) for k in results_folders}


def get_differences_between_two_outcomes(baseline_data, comparator):
    crude_diff = [x - y for x, y in zip(baseline_data[0], comparator[0])]
    avg_crude_diff = sum(crude_diff) / len(crude_diff)
    percentage_diff = [100 - ((x / y) * 100) for x, y in zip(comparator[0], baseline_data[0])]
    avg_percentage_diff = sum(percentage_diff) / len(percentage_diff)

    return {'crude': crude_diff,
            'crude_avg': avg_crude_diff,
            'percentage': percentage_diff,
            'percentage_avf': avg_percentage_diff}


def return_stillbirth_data_from_multiple_scenarios(results_folders, births_dict, sim_years, intervention_years):
    """
    Extract antenatal and intrapartum stillbirths from a scenario and return crude numbers and stillbirth rate per
    year
    :param folder: results folder for scenario
    :param births: list. mean number of births per year for a scenario (used as a denominator)
    """

    # TODO: should we report total SBR even though only ISBR will really have been effected?

    def extract_stillbirths(folder, births):
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
        all_sb_int = all_sb.loc[intervention_years[0]: intervention_years[-1]]
        # Store mean number of stillbirths, LQ, UQ
        crude_sb = get_mean_and_quants(all_sb, sim_years)

        def get_sbr(df):
            sbr_df = (df / births['births_data_frame']) * 1000
            sbr_mean_quants = get_mean_and_quants(sbr_df, sim_years)
            return sbr_mean_quants

        an_sbr = get_sbr(an_stillbirth_results)
        ip_sbr = get_sbr(ip_stillbirth_results)
        total_sbr = get_sbr(all_sb)

        avg_sbr_df = (all_sb_int / births['births_data_frame'].loc[intervention_years[0]:intervention_years[-1]]) * 1000
        avg_sbr_means = get_mean_from_columns(avg_sbr_df, 'avg')

        avg_isbr_df = (ip_stillbirth_results.loc[intervention_years[0]:intervention_years[-1]] /
                       births['births_data_frame'].loc[intervention_years[0]:intervention_years[-1]]) * 1000
        ip_sbr_means = get_mean_from_columns(avg_isbr_df, 'avg')

        avg_asbr_df = (an_stillbirth_results.loc[intervention_years[0]:intervention_years[-1]] /
                       births['births_data_frame'].loc[intervention_years[0]:intervention_years[-1]]) * 1000
        ap_sbr_means = get_mean_from_columns(avg_asbr_df, 'avg')

        # Return as dict for graphs
        return {'an_sbr': an_sbr,
                'ip_sbr': ip_sbr,
                'sbr': total_sbr,
                'crude_sb': crude_sb,
                'avg_sbr': [np.mean(avg_sbr_means), np.quantile(avg_sbr_means, 0.025),
                            np.quantile(avg_sbr_means, 0.975)],
                'avg_i_sbr': [np.mean(ip_sbr_means), np.quantile(ip_sbr_means, 0.025),
                              np.quantile(ip_sbr_means, 0.975)],
                'avg_a_sbr': [np.mean(ap_sbr_means), np.quantile(ap_sbr_means, 0.025),
                              np.quantile(ap_sbr_means, 0.975)],
                }

    return {k: extract_stillbirths(results_folders[k], births_dict[k]) for k in results_folders}


def return_dalys_from_multiple_scenarios(results_folders, sim_years, intervention_years):
    def get_dalys_from_scenario(results_folder):
        """
        Extracted stacked DALYs from logger for maternal and neonatal disorders
        :param results_folder: results folder for scenario
        :return: Maternal and neonatal dalys [Mean, LQ, UQ]
        """

        results_dict = dict()

        # Get DALY df
        dalys = extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year']).sum().stack()),
            do_scaling=True)

        dalys_stacked = extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year']).sum().stack()),
            do_scaling=True)

        # todo - should this just be at risk or total (gbd suggests total, which this calibrates well with)
        person_years_total = extract_results(
            results_folder,
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

            m_d_rate_df = (md / person_years_total) * 100_000
            n_d_rate_df = (nd / person_years_total) * 100_000

            results_dict.update({f'maternal_dalys_rate_{type}': get_mean_and_quants(m_d_rate_df, sim_years)})
            results_dict.update({f'neonatal_dalys_rate_{type}': get_mean_and_quants(n_d_rate_df, sim_years)})

            m_int_period = m_d_rate_df.loc[intervention_years[0]: intervention_years[-1]]
            n_int_period = n_d_rate_df.loc[intervention_years[0]: intervention_years[-1]]

            m_int_means = get_mean_from_columns(m_int_period, 'avg')
            n_int_means = get_mean_from_columns(n_int_period, 'avg')

            results_dict.update({f'avg_mat_dalys_rate_{type}': [np.mean(m_int_means), np.quantile(m_int_means, 0.025),
                                                                np.quantile(m_int_means, 0.975)]})
            results_dict.update({f'avg_neo_dalys_rate_{type}': [np.mean(n_int_means), np.quantile(n_int_means, 0.025),
                                                                np.quantile(n_int_means, 0.975)]})

            # Get averages/sums
            results_dict.update({f'maternal_dalys_crude_{type}': get_mean_and_quants(md, sim_years)})
            results_dict.update({f'neonatal_dalys_crude_{type}': get_mean_and_quants(nd, sim_years)})

            m_int_agg = get_mean_from_columns(m_int_period, 'sum')
            n_int_agg = get_mean_from_columns(n_int_period, 'sum')

            results_dict.update({f'agg_mat_dalys_{type}': [np.mean(m_int_agg), np.quantile(m_int_agg, 0.025),
                                                           np.quantile(m_int_agg, 0.975)]})

            results_dict.update({f'agg_neo_dalys_{type}': [np.mean(n_int_agg), np.quantile(n_int_agg, 0.025),
                                                           np.quantile(n_int_agg, 0.975)]})

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
            results_folder,
            module="tlo.methods.healthburden",
            key="yll_by_causes_of_death",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year']).sum().stack()),
            do_scaling=True)
        yll_final = yll.fillna(0)

        yll_stacked = extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="yll_by_causes_of_death_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year']).sum().stack()),
            do_scaling=True)
        yll_stacked_final = yll_stacked.fillna(0)

        yld = extract_results(
            results_folder,
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

        results_dict.update({'maternal_yll_crude_unstacked': get_mean_and_quants(mat_yll_df, sim_years)})
        results_dict.update({'maternal_yll_crude_stacked': get_mean_and_quants(mat_yll_s_df, sim_years)})

        mat_yll_df_rate = (mat_yll_df / person_years_total) * 100_000
        mat_yll_s_df_rate = (mat_yll_s_df / person_years_total) * 100_000

        results_dict.update({'maternal_yll_rate_unstacked': get_mean_and_quants(mat_yll_df_rate, sim_years)})
        results_dict.update({'maternal_yll_rate_stacked': get_mean_and_quants(mat_yll_s_df_rate, sim_years)})

        results_dict.update({'maternal_yld_crude_unstacked': get_mean_and_quants(mat_yld_df, sim_years)})

        mat_yld_df_rate = (mat_yld_df / person_years_total) * 100_000
        results_dict.update({'maternal_yld_rate_unstacked': get_mean_and_quants(mat_yld_df_rate, sim_years)})

        results_dict.update({'neonatal_yll_crude_unstacked': get_mean_and_quants(neo_yll_df, sim_years)})
        results_dict.update({'neonatal_yll_crude_stacked': get_mean_and_quants(neo_yll_s_df, sim_years)})

        neo_yll_df_rate = (neo_yll_df / person_years_total) * 100_000
        neo_yll_s_df_rate = (neo_yll_s_df / person_years_total) * 100_000

        results_dict.update({'neonatal_yll_rate_unstacked': get_mean_and_quants(neo_yll_df_rate, sim_years)})
        results_dict.update({'neonatal_yll_rate_stacked': get_mean_and_quants(neo_yll_s_df_rate, sim_years)})

        results_dict.update({'neonatal_yld_crude_unstacked': get_mean_and_quants(neo_yld_df, sim_years)})

        neo_yld_df_rate = (neo_yld_df / person_years_total) * 100_000
        results_dict.update({'neonatal_yld_rate_unstacked': get_mean_and_quants(neo_yld_df_rate, sim_years)})

        return results_dict

    # Store DALYs data for baseline and intervention
    return {k: get_dalys_from_scenario(results_folders[k]) for k in results_folders}
