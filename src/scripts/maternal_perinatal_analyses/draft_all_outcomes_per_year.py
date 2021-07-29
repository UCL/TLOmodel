from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import (
    extract_results, extract_str_results,
    get_scenario_outputs, create_pickles_locally
)

# %% Declare the name of the file that specified the scenarios used in this run.
scenario_filename = 'multi_run_calibration.py'  # <-- update this to look at other results

# %% Declare usual paths:
outputspath = Path('./outputs/sejjj49@ucl.ac.uk/')
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]
#create_pickles_locally(results_folder)  # if not created via batch

# Enter the years the simulation has ran for here?
sim_years = [2010, 2011]
# todo: replace with something more clever at some point


# ========================================== INTERVENTION COVERAGE... =================================================

# 1.) Antenatal Care...
# Mean proportion of women (across draws) who have given birth that have attended ANC1, ANC4+ and ANC8+ per year...
results = extract_results(
    results_folder,
    module="tlo.methods.care_of_women_during_pregnancy",
    key="anc_count_on_birth",
    custom_generate_series=(
        lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'total_anc'])['person_id'].count()),
    do_scaling=False
)
results.columns = results.columns.get_level_values(0)
anc_count_df = pd.DataFrame(columns=sim_years, index=[0, 1, 2, 3, 4, 5, 6, 7, 8])

# get yearly outputs
for year in sim_years:
    for row in anc_count_df.index:
        x = results.loc[year, row]
        mean = x.mean()
        anc_count_df.at[row, year] = mean

yearly_anc1_rates = list()
yearly_anc4_rates = list()
yearly_anc8_rates = list()

for year in sim_years:
    yearly_anc1_rates.append(100 - ((anc_count_df.at[0, year] / anc_count_df[year].sum()) * 100))

    four_or_more_visits = anc_count_df.loc[anc_count_df.index > 3]
    yearly_anc4_rates.append((four_or_more_visits[year].sum() / anc_count_df[year].sum()) * 100)

    eight_or_more_visits = anc_count_df.loc[anc_count_df.index > 7]
    yearly_anc8_rates.append((eight_or_more_visits[year].sum() / anc_count_df[year].sum()) * 100)


plt.plot(sim_years, yearly_anc1_rates)
plt.title('Proportion of women attending > 1 ANC contact per year')
plt.xlabel('Year')
plt.ylabel('% attended > ANC1')
#plt.show()

plt.plot(sim_years, yearly_anc1_rates, label="anc1")
plt.plot(sim_years, yearly_anc4_rates, label="anc4+")
plt.plot(sim_years, yearly_anc8_rates, label="anc8+")

plt.xlabel('Year')
plt.ylabel('Proportion of women attending ANC1, AN4, ANC8')
plt.title('Two or more lines on same plot with suitable legends ')
plt.legend()
#plt.show()
# todo: credible intervals

# Mean proportion of women who attended at least one ANC visit that attended at < 4, 4-5, 6-7 and > 8 months
# gestation...

anc_ga_first_visit = extract_results(
    results_folder,
    module="tlo.methods.care_of_women_during_pregnancy",
    key="anc_ga_first_visit",
    custom_generate_series=(
        lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'ga_anc_one'])['mother'].count()),
    do_scaling=False
)
anc_ga_first_visit.columns = anc_ga_first_visit.columns.get_level_values(0)

anc_before_four_months = list()
anc_before_four_five_months = list()
anc_before_six_seven_months = list()
anc_before_eight_plus_months = list()

total_women = dict()

for year in sim_years:
    year_df = anc_ga_first_visit.loc[year]
    total_women_that_year = 0
    for index in year_df.index:
        total_women_that_year += year_df.loc[index].mean()

    total_women.update({year: total_women_that_year})

# get yearly outputs

def get_means(df):
    mean_list = 0
    for index in df.index:
        mean_list += df.loc[index].mean()
    return mean_list

for year in sim_years:
    year_series = anc_ga_first_visit.loc[year]

    early_anc = year_series.loc[year_series.index <= 13]
    four_to_five = year_series.loc[(year_series.index > 13) & (year_series.index <= 22)]
    six_to_seven = year_series.loc[(year_series.index > 22) & (year_series.index <= 32)]
    eight_plus = year_series.loc[(year_series.index > 31)]

    sum_means_early = get_means(early_anc)
    sum_four = get_means(four_to_five)
    sum_six = get_means(six_to_seven)
    sum_eight = get_means(eight_plus)

    anc_before_four_months.append((sum_means_early/total_women[year]) * 100)
    anc_before_four_five_months.append((sum_four/total_women[year]) * 100)
    anc_before_six_seven_months.append((sum_six/total_women[year]) * 100)
    anc_before_eight_plus_months.append((sum_eight/total_women[year]) * 100)

plt.plot(sim_years, anc_before_four_months, label="<4")
plt.plot(sim_years, anc_before_four_five_months, label="4-5+")
plt.plot(sim_years, anc_before_six_seven_months, label="6-7+")
plt.plot(sim_years, anc_before_eight_plus_months, label="8+")

plt.xlabel('Year')
plt.ylabel('Gestational Age at Presentation to ANC1 per year')
plt.title('s ')
plt.legend()
#plt.show()


# TODO: quartiles, median month ANC1, total visits at pregnancy end

# 2.) Facility delivery
# Total FDR per year (denominator - total births)

# % home births per year (denominator - total births)
# % hospital deliveries per year (denominator - total births)
# % health centre deliveries per year (denominator - total births)

# 3.) Postnatal Care
# PNC mothers (denominator - total births)
# % PNC early mothers (denominator - total mothers PNC > 0 )
# PNC newborns (denominator - total (live?) births)
# % PNC early newborns (denominator - total newborns PNC > 0 )

# ========================================== COMPLICATION/DISEASE RATES.... ===========================================

# HELPER FUNCTIONS...
def get_modules_maternal_complication_dataframes(module, id_name):
    complications_df = extract_results(
        results_folder,
        module=f"tlo.methods.{module}",
        key="maternal_complication",
        custom_generate_series=(
            lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])[f'{id_name}'].count()),
        do_scaling=False
    )

    return complications_df


def get_mean(df):
    year_means = list()
    for year in sim_years:
        if year in df.index:
            year_means.append(df.loc[year].mean())
        else:
            year_means.append(0)

    return year_means


def get_mean_number_of_some_complication(df, complication):
    yearly_mean_number = list()
    for year in sim_years:
        if complication in df.loc[year].index:
            yearly_mean_number.append(df.loc[year, complication].mean())
        else:
            yearly_mean_number.append(0)
    return yearly_mean_number


def get_comp_mean_and_rate(complication, denominators, df, rate):
    yearly_comp_rate = list()
    for year, denominator in zip(sim_years, denominators):
        if complication in df.loc[year].index:
            mean_over_draws = df.loc[year, complication].mean()
            rate = (mean_over_draws / denominator) * rate
            yearly_comp_rate.append(rate)
        else:
            yearly_comp_rate.append(0)

    return yearly_comp_rate


def get_comp_mean_and_rate_across_multiple_dataframes(complication, denominators, rate, dataframes):

    def get_list_of_rates(df):
        rates_per_year = list()
        for year, denominator in zip(sim_years, denominators):
            if year in df.index:
                if complication in df.loc[year].index:
                    rates = (df.loc[year, complication].mean() / denominator) * rate
                    rates_per_year.append(rates)
                else:
                    rates_per_year.append(0)
            else:
                rates_per_year.append(0)

        return rates_per_year

    if len(dataframes) == 2:
        df_1_list = get_list_of_rates(dataframes[0])
        df_2_list = get_list_of_rates(dataframes[1])

        total_rates = [x + y for x, y in zip(df_1_list, df_2_list)]

    else:
        df_1_list = get_list_of_rates(dataframes[0])
        df_2_list = get_list_of_rates(dataframes[1])
        df_3_list = get_list_of_rates(dataframes[2])

        total_rates = [x + y + z for x, y, z in zip(df_1_list, df_2_list, df_3_list)]

    return total_rates

# ---------------------------------------------- DENOMINATORS... -----------------------------------------------------

# ________________________________________ COMPLICATION DATA FRAMES....  _____________________________________________
an_comps = get_modules_maternal_complication_dataframes('pregnancy_supervisor', 'person')
an_comps.columns = an_comps.columns.get_level_values(0)
la_comps = get_modules_maternal_complication_dataframes('labour', 'person')
pn_comps = get_modules_maternal_complication_dataframes('postnatal_supervisor', 'mother')

# -------------------------Total_pregnancies...--------------------
pregnancy_poll_results = extract_results(
    results_folder,
    module="tlo.methods.contraception",
    key="pregnant_at_age",
    custom_generate_series=(
        lambda df: df.assign(year=pd.to_datetime(df['date']).dt.year).groupby(['year'])['year'].count()
    ),
    do_scaling=True)

contraception_failure = extract_results(
    results_folder,
    module="tlo.methods.contraception",
    key="fail_contraception",
    custom_generate_series=(
        lambda df: df.assign(year=pd.to_datetime(df['date']).dt.year).groupby(['year'])['year'].count()
    ),
    do_scaling=True)

mean_pp_pregs = get_mean(pregnancy_poll_results)
mean_cf_pregs = get_mean(contraception_failure)
total_pregnancies_per_year = [x + y for x, y in zip(mean_pp_pregs, mean_cf_pregs)]

# ---------------------Total births...----------------------------
births_results = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="on_birth",
    custom_generate_series=(
        lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
    ),
    do_scaling=True
)

total_births_per_year = get_mean(births_results)

# ------------------Completed pregnancies...-----------------------
ectopic_mean_numbers_per_year = get_mean_number_of_some_complication(an_comps, 'ectopic_unruptured')
ia_mean_numbers_per_year = get_mean_number_of_some_complication(an_comps, 'induced_abortion')
sa_mean_numbers_per_year = get_mean_number_of_some_complication(an_comps, 'spontaneous_abortion')

an_stillbirth_results = extract_results(
    results_folder,
    module="tlo.methods.pregnancy_supervisor",
    key="antenatal_stillbirth",
    custom_generate_series=(
        lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
    ),
    do_scaling=True
)
total_an_stillbirths_per_year = get_mean(an_stillbirth_results)

total_completed_pregnancies_per_year = [a + b + c + d + e for a, b, c, d, e in zip(total_births_per_year,
                                                                                   ectopic_mean_numbers_per_year,
                                                                                   ia_mean_numbers_per_year,
                                                                                   sa_mean_numbers_per_year,
                                                                                   total_an_stillbirths_per_year)]

# ---------------------------------------- Twinning Rate... -----------------------------------------------------------
# % Twin births/Total Births per year
#twins_results = extract_results(
#    results_folder,
#    module="tlo.methods.newborn_outcomes",
#    key="twin_birth",
#    custom_generate_series=(
#        lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
#    ),
#    do_scaling=True
#)
# todo: possibility for /0 crash

# ---------------------------------------- Early Pregnancy Loss... ----------------------------------------------------
# Ectopic pregnancies/Total pregnancies
rate_of_ectopic_pregnancy_per_year = get_comp_mean_and_rate('ectopic_unruptured', total_pregnancies_per_year, an_comps,
                                                            1000)

# Ruptured ectopic pregnancies / Total Pregnancies
mean_unrup_ectopics = get_mean_number_of_some_complication(an_comps, 'ectopic_unruptured')
proportion_of_ectopics_that_rupture_per_year = get_comp_mean_and_rate('ectopic_ruptured', mean_unrup_ectopics, an_comps,
                                                                      100)

# todo: plot...

# Spontaneous Abortions....
rate_of_spontaneous_abortion_per_year = get_comp_mean_and_rate('spontaneous_abortion',
                                                               total_completed_pregnancies_per_year, an_comps, 1000)

# todo: plot...
# Complicated SA / Total SA
mean_complicated_sa = get_mean_number_of_some_complication(an_comps, 'spontaneous_abortion')
proportion_of_complicated_sa_per_year = get_comp_mean_and_rate('complicated_spontaneous_abortion',
                                                               mean_complicated_sa, an_comps, 100)

# Induced Abortions...
rate_of_induced_abortion_per_year = get_comp_mean_and_rate('induced_abortion',
                                                           total_completed_pregnancies_per_year, an_comps, 1000)

# todo: plot...

# Complicated IA / Total IA
mean_complicated_ia = get_mean_number_of_some_complication(an_comps, 'induced_abortion')
proportion_of_complicated_ia_per_year = get_comp_mean_and_rate('complicated_induced_abortion',
                                                               mean_complicated_ia, an_comps, 100)

# --------------------------------------------------- Syphilis Rate... ------------------------------------------------
rate_of_syphilis_per_year = get_comp_mean_and_rate('syphilis', total_completed_pregnancies_per_year, an_comps,
                                                   1000)
# todo: plot...

# ------------------------------------------------ Gestational Diabetes... -------------------------------------------
rate_of_gdm_per_year = get_comp_mean_and_rate('gest_diab', total_completed_pregnancies_per_year, an_comps,
                                              1000)

# ------------------------------------------------ PROM... -----------------------------------------------------------
rate_of_prom_per_year = get_comp_mean_and_rate('PROM', total_births_per_year, an_comps, 1000)


# ---------------------------------------------- Anaemia... ----------------------------------------------------------
# Total prevalence of Anaemia at birth (total cases of anaemia at birth/ total births per year) and by severity
anaemia_results = extract_results(
    results_folder,
    module="tlo.methods.pregnancy_supervisor",
    key="anaemia_on_birth",
    custom_generate_series=(
        lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'anaemia_status'])['year'].count()
    ),
    do_scaling=True
)

no_anaemia = get_mean_number_of_some_complication(anaemia_results, 'none')
prevalence_of_anaemia_per_year = [100 - ((x/y) * 100) for x, y in zip(no_anaemia, total_births_per_year)]

# todo: plot

# Prevalence of Anaemia at the end of the postnatal period (total cases of anaemia at 6 weeks PN/ total women at 6 weeks
# PN?) and by severity

mild_anaemia_at_birth = get_mean_number_of_some_complication(anaemia_results, 'mild')
prevalence_of_mild_anaemia_per_year = [(x/y) * 100 for x, y in zip(mild_anaemia_at_birth, total_births_per_year)]

moderate_anaemia_at_birth = get_mean_number_of_some_complication(anaemia_results, 'moderate')
prevalence_of_mod_anaemia_per_year = [(x/y) * 100 for x, y in zip(moderate_anaemia_at_birth, total_births_per_year)]

severe_anaemia_at_birth = get_mean_number_of_some_complication(anaemia_results, 'severe')
prevalence_of_sev_anaemia_per_year = [(x/y) * 100 for x, y in zip(severe_anaemia_at_birth, total_births_per_year)]


# todo: log anaemia status at end of PN...??
# todo: plot

# ------------------------------------------- Hypertensive disorders -------------------------------------------------
rate_of_gh_per_year = get_comp_mean_and_rate_across_multiple_dataframes('mild_gest_htn', total_births_per_year, 1000,
                                                                        [an_comps, pn_comps])
rate_of_sgh_per_year = get_comp_mean_and_rate_across_multiple_dataframes('severe_gest_htn', total_births_per_year, 1000,
                                                                         [an_comps, la_comps, pn_comps])
rate_of_mpe_per_year = get_comp_mean_and_rate_across_multiple_dataframes('mild_pre_eclamp', total_births_per_year, 1000,
                                                                         [an_comps, pn_comps])
rate_of_spe_per_year = get_comp_mean_and_rate_across_multiple_dataframes('severe_pre_eclamp', total_births_per_year,
                                                                         1000, [an_comps, la_comps, pn_comps])
rate_of_ec_per_year = get_comp_mean_and_rate_across_multiple_dataframes('eclampsia', total_births_per_year, 1000,
                                                                        [an_comps, la_comps, pn_comps])
# todo: plot

#  ---------------------------------------------Placenta praevia... -------------------------------------------------
rate_of_praevia_per_year = get_comp_mean_and_rate('placenta_praevia', total_pregnancies_per_year, an_comps, 1000)
# todo: plot

#  ---------------------------------------------Placental abruption... -------------------------------------------------
rate_of_abruption_per_year = get_comp_mean_and_rate_across_multiple_dataframes('placental_abruption',
                                                                               total_births_per_year, 1000,
                                                                               [an_comps, la_comps])
# todo: plot

# --------------------------------------------- Antepartum Haemorrhage... ---------------------------------------------
# Rate of APH/total births (antenatal and labour)
rate_of_mm_aph_per_year = get_comp_mean_and_rate_across_multiple_dataframes('mild_mod_antepartum_haemorrhage',
                                                                             total_births_per_year, 1000,
                                                                             [an_comps, la_comps])

rate_of_s_aph_per_year = get_comp_mean_and_rate_across_multiple_dataframes('severe_antepartum_haemorrhage',
                                                                           total_births_per_year, 1000,
                                                                           [an_comps, la_comps])

total_aph_rates = [x + y for x, y in zip(rate_of_mm_aph_per_year, rate_of_s_aph_per_year)]
# todo: plot

# --------------------------------------------- Preterm birth ... ---------------------------------------------

# 11.) Preterm birth...
# Rate of preterm birth/total births
# (separated by early and late)

# 12.) Post term birth...
# Rate of post term birth/ total births
# (Separated by early and late)

# 13.) Antenatal Stillbirth...
# Total antenatal stillbirths / total births

# 14) Birth weight...
# Prevalence of low birth weight/ total births
# Prevalence of macrosomia/ total births
# todo: check with Ines r.e. SGA and the impact on her modules

"""Mothers…
18.) Obstructed Labour
19.) Uterine rupture
20.) Caesarean section rate
21.) Assisted Vaginal Delivery rate
22.) Sepsis (labour & postnatal)
23.) Postpartum haemorrhage (labour & postnatal)
24.) Intrapartum stillbirth
26.) Fistula
27.) Maternal Death

Newborns…
1.) Neonatal sepsis (labour & postnatal)
	1.) Early onset (pre & post discharge)
	2.) Late onset
2.) Neonatal Encephalopathy
3.) Preterm RDS
4.) Neonatal respiratory depression
5.) Congenital anomalies
	1.) Heart
	2.) Limb
	3.) Urogenital
	5.) Digestive
	6.) Other
5.) Breastfeeding
6.) Neonatal Death
"""
