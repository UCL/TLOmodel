import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tlo.analysis.utils import parse_log_file

logs_dict = dict()
files = ['multi_run_calib_1', 'multi_run_calib_2', 'multi_run_calib_3',  'multi_run_calib_4', 'multi_run_calib_5',
                              'multi_run_calib_6', 'multi_run_calib_7', 'multi_run_calib_8', 'multi_run_calib_9',
                              'multi_run_calib_10']

for file in files:
    new_parse_log = {file: parse_log_file(filepath=f"./outputs/sejjj49@ucl.ac.uk/"
                                                   f"multi_run_calibration-2021-06-02T192336Z/logfiles/{file}.log")}
    logs_dict.update(new_parse_log)


# ========================================== HELPER FUNCTIONS =========================================================
def get_deaths(module):
    if 'direct_maternal_death' in logs_dict[file][f'tlo.methods.{module}']:
        direct_deaths = logs_dict[file][f'tlo.methods.{module}']['direct_maternal_death']
        direct_deaths['date'] = pd.to_datetime(direct_deaths['date'])
        direct_deaths['year'] = direct_deaths['date'].dt.year
        return len(direct_deaths.loc[direct_deaths['year'] == 2011])


def get_live_births(module):
    if 'live_birth' in logs_dict[file][f'tlo.methods.{module}']:
        live_births_df = logs_dict[file][f'tlo.methods.{module}']['live_birth']
        live_births_df['date'] = pd.to_datetime(live_births_df['date'])
        live_births_df['year'] = live_births_df['date'].dt.year
        return len(live_births_df.loc[live_births_df['year'] == 2011])
    else:
        return 0


def get_total_births():
    if 'on_birth' in logs_dict[file][f'tlo.methods.demography']:
        all_births = logs_dict[file][f'tlo.methods.demography']['on_birth']
        all_births['date'] = pd.to_datetime(all_births['date'])
        all_births['year'] = all_births['date'].dt.year
        return len(all_births.loc[all_births['year'] == 2011])
    else:
        return 0


def get_incidence(module, complications):
    per_pregnancy_incidence = dict()
    for complication in complications:
        row = {complication: 0}
        per_pregnancy_incidence.update(row)

    for file in files:
        if 'maternal_complication' in logs_dict[file][f'tlo.methods.{module}']:
            comps = logs_dict[file][f'tlo.methods.{module}']['maternal_complication']
            comps['date'] = pd.to_datetime(comps['date'])
            comps['year'] = comps['date'].dt.year
            for complication in per_pregnancy_incidence:
                number_of_comps = len(comps.loc[(comps['type'] == f'{complication}') &
                                                (comps['year'] == 2011)])

                per_pregnancy_incidence[complication] += number_of_comps
    print(per_pregnancy_incidence)
    return per_pregnancy_incidence

# ========================================== DENOMINATORS =============================================================

total_pregnancies = 0
for file in files:
    if 'pregnant_at_age' in logs_dict[file][f'tlo.methods.contraception']:
        pregnancies = logs_dict[file]['tlo.methods.contraception']['pregnant_at_age']
        pregnancies['date'] = pd.to_datetime(pregnancies['date'])
        pregnancies['year'] = pregnancies['date'].dt.year
        total_pregnancies += len(pregnancies.loc[pregnancies['year'] == 2011])

live_births = 0  # TODO: CHECK STILLBIRTHS ARENT LOGGED HERE
for file in files:
    live_births += get_live_births('labour')
    live_births += get_live_births('newborn_outcomes')

total_births = 0
for file in files:
    total_births += get_total_births()


# ============================================= DEATH =================================================================
total_direct_death = 0
total_indirect_death = 0

for file in files:
    total_direct_death += get_deaths('labour')
    total_direct_death += get_deaths('pregnancy_supervisor')
    total_direct_death += get_deaths('postnatal_supervisor')

for file in files:
    if 'death' in logs_dict[file]['tlo.methods.demography']:
        total_deaths = logs_dict[file]['tlo.methods.demography']['death']
        total_deaths['date'] = pd.to_datetime(total_deaths['date'])
        total_deaths['year'] = total_deaths['date'].dt.year

        deaths = total_deaths.loc[total_deaths['pregnancy'] &
                                  (total_deaths['cause'].str.contains('AIDS|severe_malaria|Suicide')) &
                                  (total_deaths['year'] == 2011)]
        indirect_deaths_preg_2011 = len(deaths)

        # todo: only aids, malaria, tb (? NCDs?)
        indirect_deaths_postnatal_2011 = len(
            total_deaths.loc[total_deaths['postnatal'] &
                             (total_deaths['cause'].str.contains('AIDS|severe_malaria|Suicide')) &
                             (total_deaths['year'] == 2011)])

        total_indirect_death += indirect_deaths_preg_2011
        total_indirect_death += indirect_deaths_postnatal_2011

maternal_deaths = total_direct_death + total_indirect_death
mean_maternal_deaths = maternal_deaths / len(files)

mean_livebirths = live_births / len(files)
mean_mmr = (mean_maternal_deaths / mean_livebirths) * 100000
total_mmr = (maternal_deaths / live_births) * 100000

prop_indirect_deaths = (total_indirect_death / maternal_deaths) * 100
indirect_mmr = (total_indirect_death / live_births) * 100000
direct_mmr = (total_direct_death / live_births) * 100000

print('total', total_mmr)
print('prop_indirect', prop_indirect_deaths)

# PLOT ...
labels = ['2011', 'Target']
direct_deaths = [direct_mmr, 472]
indirect_deaths = [indirect_mmr, 203]
width = 0.35
fig, ax = plt.subplots()
ax.bar(labels, direct_deaths, width, label='Direct Deaths')
ax.bar(labels, indirect_deaths, width, bottom=direct_deaths,
       label='Indirect Deaths')

ax.set_ylabel('Maternal Deaths per 100,000 live births')
ax.set_title('Maternal Mortality Ratio Calibration')
ax.legend()
plt.show()

# TODO: by time point
# --------------------------------------- PROPORTION OF DEATHS PER COMPLICATION --------------------------------------
direct_causes = ['ectopic_pregnancy', 'spontaneous_abortion', 'induced_abortion',
                 'severe_gestational_hypertension', 'severe_pre_eclampsia', 'eclampsia', 'antenatal_sepsis',
                 'uterine_rupture', 'intrapartum_sepsis', 'postpartum_sepsis', 'postpartum_haemorrhage',
                 'secondary_postpartum_haemorrhage', 'antepartum_haemorrhage']

crude_deaths= dict()
proportions = dict()

for cause in direct_causes:
    row = {cause: 0}
    proportions.update(row)
    crude_deaths.update(row)

for file in files:
    if 'death' in logs_dict[file]['tlo.methods.demography']:
        total_deaths = logs_dict[file]['tlo.methods.demography']['death']
        total_deaths['date'] = pd.to_datetime(total_deaths['date'])
        total_deaths['year'] = total_deaths['date'].dt.year
        for cause in proportions:
            number_of_deaths = len(total_deaths.loc[(total_deaths['cause'] == f'{cause}') &
                                               (total_deaths['year'] == 2011)])

            crude_deaths[cause] += number_of_deaths

for cause in proportions:
    proportions[cause] = (crude_deaths[cause] / total_direct_death) * 100

# PLOT...
plt.rcParams['font.size'] = 9.0
labels = proportions.keys()
sizes = list(proportions.values())
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Proportion of total maternal deaths by cause ')
plt.show()
print(proportions)

x = labels
sizes = list(proportions.values())
x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, sizes, color='green', width=0.9)
plt.xlabel("Energy Source")
plt.ylabel("% of Total Direct Deaths")
plt.title("% of Direct Maternal Deaths Attributed to each cause")

plt.xticks(x_pos, x, rotation=90)
plt.show()


# --------------------------------------- CASE FATALITIES -------------------------------------------------------------
antenatal_comps = ['spontaneous_abortion', 'induced_abortion', 'spontaneous_abortion_haemorrhage',
                   'induced_abortion_haemorrhage', 'spontaneous_abortion_sepsis',
                   'induced_abortion_sepsis', 'spontaneous_abortion_injury',
                   'induced_abortion_complication', 'complicated_induced_abortion',
                   'complicated_spontaneous_abortion', 'iron_deficiency', 'folate_deficiency', 'b12_deficiency',
                   'mild_anaemia', 'moderate_anaemia', 'severe_anaemia', 'gest_diab',
                   'mild_pre_eclamp', 'mild_gest_htn', 'severe_pre_eclamp', 'eclampsia', 'severe_gest_htn',
                   'placental_abruption', 'severe_antepartum_haemorrhage', 'mild_mod_antepartum_haemorrhage',
                   'clinical_chorioamnionitis', 'PROM', 'ectopic_unruptured', 'multiple_pregnancy', 'placenta_praevia',
                   'ectopic_ruptured', 'syphilis']

intrapartum_comps = ['placental_abruption', 'mild_mod_antepartum_haemorrhage', 'severe_antepartum_haemorrhage',
                     'sepsis', 'uterine_rupture', 'eclampsia', 'severe_gest_htn', 'severe_pre_eclamp',
                     'early_preterm_labour', 'late_preterm_labour', 'post_term_labour', 'obstructed_labour',
                     'primary_postpartum_haemorrhage']

postnatal_comps = ['vesicovaginal_fistula', 'rectovaginal_fistula', 'sepsis', 'secondary_postpartum_haemorrhage',
                   'iron_deficiency', 'folate_deficiency', 'b12_deficiency', 'mild_anaemia', 'moderate_anaemia',
                   'severe_anaemia',  'mild_pre_eclamp', 'mild_gest_htn', 'severe_pre_eclamp', 'eclampsia',
                   'severe_gest_htn']

ps_incidence_dict = get_incidence('pregnancy_supervisor', antenatal_comps)
lab_incidence_dict = get_incidence('labour', intrapartum_comps)
pn_incidence_dict = get_incidence('postnatal_supervisor', postnatal_comps)

cfrs = dict()
for i in direct_causes:
    if i == 'ectopic_pregnancy':
        cfr = (crude_deaths['ectopic_pregnancy'] / ps_incidence_dict['ectopic_unruptured']) * 100
        new_row = {'ectopic_cfr': cfr}
        cfrs.update(new_row)

    if i == 'spontaneous_abortion':
        cfr = (crude_deaths['spontaneous_abortion'] / ps_incidence_dict['complicated_spontaneous_abortion']) * 100
        new_row = {'spontaneous_abortion_cfr': cfr}
        cfrs.update(new_row)

    if i == 'induced_abortion':
        cfr = (crude_deaths['induced_abortion'] / ps_incidence_dict['complicated_induced_abortion']) * 100
        new_row = {'induced_abortion_cfr': cfr}
        cfrs.update(new_row)

    if i == 'severe_gestational_hypertension':
        total_cases = ps_incidence_dict['severe_gest_htn'] + lab_incidence_dict['severe_gest_htn'] + \
                      pn_incidence_dict['severe_gest_htn']
        if total_cases > 0:
            cfr = (crude_deaths['severe_gestational_hypertension'] / total_cases) * 100
            new_row = {'severe_gest_htn_cfr': cfr}

        else:
            new_row = {'severe_gest_htn_cfr': 0}
        cfrs.update(new_row)

    if i == 'severe_pre_eclampsia':
        total_cases = ps_incidence_dict['severe_pre_eclamp'] + lab_incidence_dict['severe_pre_eclamp'] + \
                      pn_incidence_dict['severe_pre_eclamp']
        if total_cases > 0:
            cfr = (crude_deaths['severe_pre_eclampsia'] / total_cases) * 100
            new_row = {'severe_pre_eclamp_cfr': cfr}

        else:
            new_row = {'severe_pre_eclamp_cfr': 0}
        cfrs.update(new_row)

    if i == 'eclampsia':
        total_cases = ps_incidence_dict['eclampsia'] + lab_incidence_dict['eclampsia'] + \
                      pn_incidence_dict['eclampsia']
        if total_cases > 0:
            cfr = (crude_deaths['eclampsia'] / total_cases) * 100
            new_row = {'eclampsia_cfr': cfr}

        else:
            new_row = {'eclampsia_cfr': 0}
        cfrs.update(new_row)

    if i == 'antenatal_sepsis':  # todo: this is quite right as labour has a mix of ip/pn. so maybe just check total
        if ps_incidence_dict['clinical_chorioamnionitis'] == 0:
            cfr = 0
        else:
            cfr = (crude_deaths['antenatal_sepsis'] / ps_incidence_dict['clinical_chorioamnionitis']) * 100
        new_row = {'antenatal_sepsis_cfr': cfr}
        cfrs.update(new_row)
    if i == 'intrapartum_sepsis':
        cfr = (crude_deaths['intrapartum_sepsis'] / lab_incidence_dict['sepsis']) * 100
        new_row = {'intrapartum_sepsis_cfr': cfr}
        cfrs.update(new_row)
    if i == 'postpartum_sepsis':
        cfr = (crude_deaths['postpartum_sepsis'] / pn_incidence_dict['sepsis']) * 100
        new_row = {'postpartum_sepsis_cfr': cfr}
        cfrs.update(new_row)

        total_sepsis = ps_incidence_dict['clinical_chorioamnionitis'] + lab_incidence_dict['sepsis'] + \
                       pn_incidence_dict['sepsis']
        total_sepsis_deaths = crude_deaths['antenatal_sepsis'] + crude_deaths['intrapartum_sepsis'] + \
                              crude_deaths['postpartum_sepsis']

        total_sepsis_cfr = (total_sepsis_deaths/total_sepsis) * 100
        new_row = {'total_sepsis_cfr': total_sepsis_cfr}
        cfrs.update(new_row)

    if i == 'uterine_rupture':
        cfr = (crude_deaths['uterine_rupture'] / lab_incidence_dict['uterine_rupture']) * 100
        new_row = {'uterine_rupture_cfr': cfr}
        cfrs.update(new_row)

    if i == 'postpartum_haemorrhage':
        cfr = (crude_deaths['postpartum_haemorrhage'] / lab_incidence_dict['primary_postpartum_haemorrhage']) * 100
        new_row = {'primary_pph_cfr': cfr}
        cfrs.update(new_row)
    if i == 'secondary_postpartum_haemorrhage':
        cfr = (crude_deaths['secondary_postpartum_haemorrhage'] /
               pn_incidence_dict['secondary_postpartum_haemorrhage']) * 100
        new_row = {'secondary_pph_cfr': cfr}
        cfrs.update(new_row)

    if i == 'antepartum_haemorrhage':
        total_cases = ps_incidence_dict['mild_mod_antepartum_haemorrhage'] +\
                      ps_incidence_dict['severe_antepartum_haemorrhage'] + \
                      lab_incidence_dict['mild_mod_antepartum_haemorrhage'] + \
                      lab_incidence_dict['severe_antepartum_haemorrhage']
        if (total_cases > 0) and (crude_deaths['antepartum_haemorrhage'] <= total_cases):
            cfr = (crude_deaths['antepartum_haemorrhage'] / total_cases) * 100
            new_row = {'antepartum_haemorrhage_cfr': cfr}
        else:
            new_row = {'antepartum_haemorrhage_cfr': cfr}
        cfrs.update(new_row)

# PLOT
N = 14
model_cfrs = cfrs.values()
calibration_cfrs = (2.38, 1.21, 1.21, 0, 1.82, 1.82, 0, 7.87, 0, 0, 6.85, 3.86, 3.86, 1.73)
ind = np.arange(N)
width = 0.35
plt.bar(ind, model_cfrs, width, label='Model')
plt.bar(ind + width, calibration_cfrs, width,
    label='Calibration')
plt.ylabel('CRF')
plt.title('CFRs of pregnancy complications')
plt.xticks(ind + width / 2, ('ECT', 'SA', 'IA', 'SGH', 'SPE', 'EC', 'ASEP', 'UR', 'ISEP', 'PSEP', 'SEP', 'PPPH',
                             'SPPH', 'APH'))
plt.legend(loc='best')
plt.show()

# =============================================== INCIDENCE ===========================================================

# check these denominators

for complication in ps_incidence_dict:
       ps_incidence_dict[complication] = (ps_incidence_dict[complication] / total_pregnancies) * 100

x = ['MP', 'EP', 'SA', 'IA', 'AN', 'GD', 'MPE', 'MGH', 'SPE', 'SGH', 'EC', 'APH', 'SYPH']
sizes= [ps_incidence_dict['multiple_pregnancy'], ps_incidence_dict['ectopic_unruptured'],
        ps_incidence_dict['spontaneous_abortion'],
        ps_incidence_dict['induced_abortion'], (ps_incidence_dict['mild_anaemia'] +
                                                ps_incidence_dict['moderate_anaemia'] +
                                                ps_incidence_dict['severe_anaemia']),
        ps_incidence_dict['gest_diab'],
        ps_incidence_dict['mild_pre_eclamp'], ps_incidence_dict['mild_gest_htn'], ps_incidence_dict['severe_pre_eclamp'],
        ps_incidence_dict['severe_gest_htn'], ps_incidence_dict['eclampsia'],
        (ps_incidence_dict['mild_mod_antepartum_haemorrhage'] + ps_incidence_dict['severe_antepartum_haemorrhage']),
        ps_incidence_dict['syphilis']]

plt.rcParams['font.size'] = 0.9
x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, sizes, color='green', width=0.9)
plt.xlabel("Complication")
plt.ylabel("Incidence (cases per 100 pregnancies)")
plt.title("Incidence of complications during antenatal period ")

plt.xticks(x_pos, x, rotation=75)
plt.show()

for complication in lab_incidence_dict:
       lab_incidence_dict[complication] = (lab_incidence_dict[complication] / total_births) * 100

x = ['OL', 'APH', 'UR', 'SGH', 'SPE', 'EC', 'SEP', 'PPH']
sizes = [lab_incidence_dict['obstructed_labour'], (lab_incidence_dict['mild_mod_antepartum_haemorrhage'] +
                                                   lab_incidence_dict['severe_antepartum_haemorrhage']),
         lab_incidence_dict['uterine_rupture'], lab_incidence_dict['severe_gest_htn'],
         lab_incidence_dict['severe_pre_eclamp'],  lab_incidence_dict['eclampsia'],
         lab_incidence_dict['sepsis'], lab_incidence_dict['primary_postpartum_haemorrhage']]

x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, sizes, color='green', width=0.9)
plt.xlabel("Complication")
plt.ylabel("Incidence (cases per 100 deliveries)")
plt.title("Incidence of complications during labour")

plt.xticks(x_pos, x, rotation=75)
plt.show()

for complication in pn_incidence_dict:     # TODO: MAY NOT BE THE RIGHT DENOMINATOR
       pn_incidence_dict[complication] = (pn_incidence_dict[complication] / total_births) * 100

x = ['VVF', 'RVF', 'SEP', 'SPPH', 'AN', 'MPE', 'MGH', 'SGH', 'SPE', 'EC']
sizes = [pn_incidence_dict['vesicovaginal_fistula'], pn_incidence_dict['rectovaginal_fistula'],
         pn_incidence_dict['sepsis'], pn_incidence_dict['secondary_postpartum_haemorrhage'],
         (pn_incidence_dict['mild_anaemia'] + pn_incidence_dict['moderate_anaemia'] +
          pn_incidence_dict['severe_anaemia']),   pn_incidence_dict['mild_pre_eclamp'],
         pn_incidence_dict['mild_gest_htn'], ps_incidence_dict['severe_gest_htn'],
         pn_incidence_dict['severe_pre_eclamp'], pn_incidence_dict['eclampsia']]

x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, sizes, color='green', width=0.9)
plt.xlabel("Complication")
plt.ylabel("Incidence (cases per 100 deliveries)")
plt.title("Incidence of complications in the postnatal period")

plt.xticks(x_pos, x, rotation=75)
plt.show()

# ==========================================HEALTH SYSTEM==============================================================
# FACILITY DELIVERIES...
hospital_deliveries = 0
health_centre_deliveries = 0

for file in files:
    if 'delivery_setting' in logs_dict[file]['tlo.methods.labour']:
        facility_deliveries = logs_dict[file][f'tlo.methods.labour']['delivery_setting']
        facility_deliveries['date'] = pd.to_datetime(facility_deliveries['date'])
        facility_deliveries['year'] = facility_deliveries['date'].dt.year

        hospital_deliveries += len(facility_deliveries.loc[(facility_deliveries['facility_type'] == 'hospital') &
                                                           (facility_deliveries['year'] == 2011)])
        health_centre_deliveries += len(facility_deliveries.loc[(facility_deliveries['facility_type'] ==
                                                                 'health_centre') &
                                                                (facility_deliveries['year'] == 2011)])


hpd_rate_2010 = (hospital_deliveries / total_births) * 100
hcd_rate_2010 = (health_centre_deliveries / total_births) * 100
fd_rate_2010 = ((hospital_deliveries + health_centre_deliveries) / total_births) * 100

# tofo this is wrong...

objects = ('Total FDR', 'Hospital DR', 'Health Centre DR', 'Calibration')
y_pos = np.arange(len(objects))
plt.bar(y_pos, [fd_rate_2010, hpd_rate_2010, hcd_rate_2010, 73], align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Facility Deliveries/ Total births')
plt.title('Facility Delivery Rate 2010')
plt.show()

# ANC ...
total_anc1 = 0
early_anc1 = 0
total_anc4 = 0
total_anc8 = 0
anc1_months_5 = 0
anc1_months_6_7 = 0
anc1_months_8 = 0
total_anc_num = 0

for file in files:
    anc1 = logs_dict[file]['tlo.methods.care_of_women_during_pregnancy']['anc1']
    anc1['date'] = pd.to_datetime(anc1['date'])
    anc1['year'] = anc1['date'].dt.year

    total_anc = logs_dict[file]['tlo.methods.care_of_women_during_pregnancy']['anc_count_on_birth']
    total_anc['date'] = pd.to_datetime(total_anc['date'])
    total_anc['year'] = total_anc['date'].dt.year

    total_anc_num += len(total_anc.loc[total_anc['year'] == 2011])

    total_anc1 += len(total_anc.loc[(total_anc['total_anc'] > 0) & (total_anc['year'] == 2011)])
    early_anc1 += len(anc1.loc[(anc1['year'] == 2011) & (anc1['gestation'] <= 17)])
    anc1_months_5 += len(anc1.loc[(anc1['year'] == 2011) & (anc1['gestation'] > 17) & (anc1['gestation'] < 22)])
    # calibrating to months 5-6
    anc1_months_6_7 += len(anc1.loc[(anc1['year'] == 2011) & (anc1['gestation'] > 22) & (anc1['gestation'] < 35)])
    # calibrating to months 5-6
    anc1_months_8 += len(anc1.loc[(anc1['year'] == 2011) & (anc1['gestation'] > 34)])

    total_anc4 += len(total_anc.loc[total_anc['total_anc'] > 3])
    total_anc8 += len(total_anc.loc[total_anc['total_anc'] > 7])


# 2.) calculate the rates
anc1_rate = (total_anc1 / total_anc_num) * 100
anc4_rate = (total_anc4 / total_anc_num) * 100
anc8_rate = (total_anc8 / total_anc_num) * 100

objects = ('ANC 1', 'Calibration', 'ANC4+', 'Calibration', 'ANC8+')
y_pos = np.arange(len(objects))
plt.bar(y_pos, [anc1_rate, 94.7, anc4_rate, 46, anc8_rate], align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Antenatal Care / Live births')
plt.title('Antenatal Care Coverage Rate 2010')
plt.show()

# todo: create a DF and output to excel?
