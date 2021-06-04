import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tlo.analysis.utils import parse_log_file

plt.rcParams.update({'font.size': 9})


logs_dict = dict()
files = ['multi_run_calib_1', 'multi_run_calib_2', 'multi_run_calib_3',  'multi_run_calib_4', 'multi_run_calib_5',
                              'multi_run_calib_6', 'multi_run_calib_7', 'multi_run_calib_8', 'multi_run_calib_9',
                              'multi_run_calib_10']

for file in files:
    new_parse_log = {file: parse_log_file(filepath=f"./outputs/sejjj49@ucl.ac.uk/"
                                                   f"multi_run_calibration-2021-06-03T173836Z/logfiles/{file}.log")}
    logs_dict.update(new_parse_log)

x='y'


# ========================================== HELPER FUNCTIONS =========================================================
def get_deaths(module):
    if 'direct_maternal_death' in logs_dict[file][f'tlo.methods.{module}']:
        direct_deaths = logs_dict[file][f'tlo.methods.{module}']['direct_maternal_death']
        direct_deaths['date'] = pd.to_datetime(direct_deaths['date'])
        direct_deaths['year'] = direct_deaths['date'].dt.year
        return len(direct_deaths.loc[direct_deaths['year'] == 2011])
    else:
        return 0


def get_direct_deaths_from_demography(list):
    total_direct_deaths = 0
    for file in files:
        if 'death' in logs_dict[file]['tlo.methods.demography']:
            total_deaths = logs_dict[file]['tlo.methods.demography']['death']
            total_deaths['date'] = pd.to_datetime(total_deaths['date'])
            total_deaths['year'] = total_deaths['date'].dt.year
            for cause in list:
                number_of_deaths = len(total_deaths.loc[(total_deaths['cause'] == f'{cause}') &
                                                        (total_deaths['year'] == 2011)])

                total_direct_deaths += number_of_deaths

    return total_direct_deaths


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


def get_stillbirths(module, pregnancy_period):
    stillbirths_time_period = 0
    for file in files:
        if f'{pregnancy_period}_stillbirth' in logs_dict[file][f'tlo.methods.{module}']:
            stillbirths = logs_dict[file][f'tlo.methods.{module}'][f'{pregnancy_period}_stillbirth']
            stillbirths['date'] = pd.to_datetime(stillbirths['date'])
            stillbirths['year'] = stillbirths['date'].dt.year

            stillbirths_time_period += len(stillbirths.loc[stillbirths['year'] == 2011])

    return stillbirths_time_period

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

# =====================================================================================================================
# =========================================== MATERNAL OUTCOMES =======================================================
# =====================================================================================================================

# MATERNAL DEATH
total_indirect_death = 0

direct_causes = ['ectopic_pregnancy', 'spontaneous_abortion', 'induced_abortion',
                 'severe_gestational_hypertension', 'severe_pre_eclampsia', 'eclampsia', 'antenatal_sepsis',
                 'uterine_rupture', 'intrapartum_sepsis', 'postpartum_sepsis', 'postpartum_haemorrhage',
                 'secondary_postpartum_haemorrhage', 'antepartum_haemorrhage']

total_direct_death = get_direct_deaths_from_demography(direct_causes)


# todo: add TB to indirect deaths
for file in files:
    if 'death' in logs_dict[file]['tlo.methods.demography']:
        total_deaths = logs_dict[file]['tlo.methods.demography']['death']
        total_deaths['date'] = pd.to_datetime(total_deaths['date'])
        total_deaths['year'] = total_deaths['date'].dt.year

        deaths = total_deaths.loc[total_deaths['pregnancy'] &
                                  (total_deaths['cause'].str.contains('AIDS|severe_malaria|Suicide|diabetes|'
                                                                      'chronic_kidney_disease|chronic_ischemic_hd'))
                                  & (total_deaths['year'] == 2011)]
        indirect_deaths_preg_2011 = len(deaths)

        indirect_deaths_postnatal_2011 = len(
            total_deaths.loc[total_deaths['postnatal'] &
                             (total_deaths['cause'].str.contains('AIDS|severe_malaria|Suicide|diabetes|'
                                                                 'chronic_kidney_disease|chronic_ischemic_hd')) &
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
indirect_causes = ['AIDS', 'severe_malaria', 'Suicide', 'diabetes', 'chronic_kidney_disease', 'chronic_ischemic_hd']

direct_crude_deaths = dict()
direct_proportions = dict()

for cause in direct_causes:
    row = {cause: 0}
    direct_proportions.update(row)
    direct_crude_deaths.update(row)

for file in files:
    if 'death' in logs_dict[file]['tlo.methods.demography']:
        total_deaths = logs_dict[file]['tlo.methods.demography']['death']
        total_deaths['date'] = pd.to_datetime(total_deaths['date'])
        total_deaths['year'] = total_deaths['date'].dt.year
        for cause in direct_proportions:
            number_of_deaths = len(total_deaths.loc[(total_deaths['cause'] == f'{cause}') &
                                                    (total_deaths['year'] == 2011)])


            direct_crude_deaths[cause] += number_of_deaths

for cause in direct_proportions:
    direct_proportions[cause] = (direct_crude_deaths[cause] / total_direct_death) * 100

indirect_crude_deaths = dict()
indirect_proportions = dict()

for cause in indirect_causes:
    row = {cause: 0}
    indirect_crude_deaths.update(row)
    indirect_proportions.update(row)

for file in files:
    if 'death' in logs_dict[file]['tlo.methods.demography']:
        total_deaths = logs_dict[file]['tlo.methods.demography']['death']
        total_deaths['date'] = pd.to_datetime(total_deaths['date'])
        total_deaths['year'] = total_deaths['date'].dt.year
        for cause in indirect_proportions:
            number_of_an_deaths = len(total_deaths.loc[(total_deaths['cause'] == f'{cause}') &
                                                       (total_deaths['year'] == 2011) &
                                                       (total_deaths['pregnancy'])])

            number_of_pn_deaths = len(total_deaths.loc[(total_deaths['cause'] == f'{cause}') &
                                                       (total_deaths['year'] == 2011) &
                                                       (total_deaths['postnatal'])])

            indirect_crude_deaths[cause] += (number_of_an_deaths + number_of_pn_deaths)

for cause in indirect_proportions:
    indirect_proportions[cause] = (indirect_crude_deaths[cause] / total_indirect_death) * 100


# PLOT...
labels = ['EP', 'SA', 'IA', 'SGH', 'SPE', 'EC', 'ASEP', 'UR', 'ISEP', 'PSEP', 'PPH', 'SPPH', 'APH']
sizes = list(direct_proportions.values())
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Proportion of total maternal deaths by cause ')
plt.show()
print(direct_proportions)

x = labels
sizes = list(direct_proportions.values())
x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, sizes, color='green', width=0.9)
plt.xlabel("Cause of death")
plt.ylabel("% of Total Direct Deaths")
plt.title("% of Direct Maternal Deaths Attributed to each cause")
plt.xticks(x_pos, x)
plt.show()

x = ['AIDS', 'Malaria', 'Suicide', 'Diabetes', 'CKD', 'CHD']
sizes = list(indirect_proportions.values())
x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, sizes, color='pink', width=0.9)
plt.xlabel("Cause of death")
plt.ylabel("% of Total Indirect Deaths")
plt.title("% of Indirect Maternal Deaths Attributed to each cause")
plt.xticks(x_pos, x)
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
        cfr = (direct_crude_deaths['ectopic_pregnancy'] / ps_incidence_dict['ectopic_unruptured']) * 100
        new_row = {'ectopic_cfr': cfr}
        cfrs.update(new_row)

    if i == 'spontaneous_abortion':
        cfr = (direct_crude_deaths['spontaneous_abortion'] / ps_incidence_dict['complicated_spontaneous_abortion']) * 100
        new_row = {'spontaneous_abortion_cfr': cfr}
        cfrs.update(new_row)

    if i == 'induced_abortion':
        cfr = (direct_crude_deaths['induced_abortion'] / ps_incidence_dict['complicated_induced_abortion']) * 100
        new_row = {'induced_abortion_cfr': cfr}
        cfrs.update(new_row)

    if i == 'severe_gestational_hypertension':
        total_cases = ps_incidence_dict['severe_gest_htn'] + lab_incidence_dict['severe_gest_htn'] + \
                      pn_incidence_dict['severe_gest_htn']
        if total_cases > 0:
            cfr = (direct_crude_deaths['severe_gestational_hypertension'] / total_cases) * 100
            new_row = {'severe_gest_htn_cfr': cfr}

        else:
            new_row = {'severe_gest_htn_cfr': 0}
        cfrs.update(new_row)

    if i == 'severe_pre_eclampsia':
        total_cases = ps_incidence_dict['severe_pre_eclamp'] + lab_incidence_dict['severe_pre_eclamp'] + \
                      pn_incidence_dict['severe_pre_eclamp']
        if total_cases > 0:
            cfr = (direct_crude_deaths['severe_pre_eclampsia'] / total_cases) * 100
            new_row = {'severe_pre_eclamp_cfr': cfr}

        else:
            new_row = {'severe_pre_eclamp_cfr': 0}
        cfrs.update(new_row)

    if i == 'eclampsia':
        total_cases = ps_incidence_dict['eclampsia'] + lab_incidence_dict['eclampsia'] + \
                      pn_incidence_dict['eclampsia']
        if total_cases > 0:
            cfr = (direct_crude_deaths['eclampsia'] / total_cases) * 100
            new_row = {'eclampsia_cfr': cfr}

        else:
            new_row = {'eclampsia_cfr': 0}
        cfrs.update(new_row)

    if i == 'antenatal_sepsis':  # todo: this is quite right as labour has a mix of ip/pn. so maybe just check total
        if ps_incidence_dict['clinical_chorioamnionitis'] == 0:
            cfr = 0
        else:
            cfr = (direct_crude_deaths['antenatal_sepsis'] / ps_incidence_dict['clinical_chorioamnionitis']) * 100
        new_row = {'antenatal_sepsis_cfr': cfr}
        cfrs.update(new_row)
    if i == 'intrapartum_sepsis':
        cfr = (direct_crude_deaths['intrapartum_sepsis'] / lab_incidence_dict['sepsis']) * 100
        new_row = {'intrapartum_sepsis_cfr': cfr}
        cfrs.update(new_row)
    if i == 'postpartum_sepsis':
        cfr = (direct_crude_deaths['postpartum_sepsis'] / pn_incidence_dict['sepsis']) * 100
        new_row = {'postpartum_sepsis_cfr': cfr}
        cfrs.update(new_row)

        total_sepsis = ps_incidence_dict['clinical_chorioamnionitis'] + lab_incidence_dict['sepsis'] + \
                       pn_incidence_dict['sepsis']
        total_sepsis_deaths = direct_crude_deaths['antenatal_sepsis'] + direct_crude_deaths['intrapartum_sepsis'] + \
                              direct_crude_deaths['postpartum_sepsis']

        total_sepsis_cfr = (total_sepsis_deaths/total_sepsis) * 100
        new_row = {'total_sepsis_cfr': total_sepsis_cfr}
        cfrs.update(new_row)

    if i == 'uterine_rupture':
        cfr = (direct_crude_deaths['uterine_rupture'] / lab_incidence_dict['uterine_rupture']) * 100
        new_row = {'uterine_rupture_cfr': cfr}
        cfrs.update(new_row)

    if i == 'postpartum_haemorrhage':
        cfr = (direct_crude_deaths['postpartum_haemorrhage'] / lab_incidence_dict['primary_postpartum_haemorrhage']) * 100
        new_row = {'primary_pph_cfr': cfr}
        cfrs.update(new_row)
    if i == 'secondary_postpartum_haemorrhage':
        cfr = (direct_crude_deaths['secondary_postpartum_haemorrhage'] /
               pn_incidence_dict['secondary_postpartum_haemorrhage']) * 100
        new_row = {'secondary_pph_cfr': cfr}
        cfrs.update(new_row)

    if i == 'antepartum_haemorrhage':
        total_cases = ps_incidence_dict['mild_mod_antepartum_haemorrhage'] +\
                      ps_incidence_dict['severe_antepartum_haemorrhage'] + \
                      lab_incidence_dict['mild_mod_antepartum_haemorrhage'] + \
                      lab_incidence_dict['severe_antepartum_haemorrhage']
        if (total_cases > 0) and (direct_crude_deaths['antepartum_haemorrhage'] <= total_cases):
            cfr = (direct_crude_deaths['antepartum_haemorrhage'] / total_cases) * 100
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
       ps_incidence_dict[complication] = (ps_incidence_dict[complication] / total_pregnancies) * 1000

x = ['MP', 'EP', 'SA', 'IA', 'AN', 'GD', 'MPE', 'MGH', 'SPE', 'SGH', 'EC', 'APH', 'SYPH']
sizes = [ps_incidence_dict['multiple_pregnancy'], ps_incidence_dict['ectopic_unruptured'],
        ps_incidence_dict['spontaneous_abortion'],
        ps_incidence_dict['induced_abortion'], (ps_incidence_dict['mild_anaemia'] +
                                                ps_incidence_dict['moderate_anaemia'] +
                                                ps_incidence_dict['severe_anaemia']),
        ps_incidence_dict['gest_diab'],
        ps_incidence_dict['mild_pre_eclamp'], ps_incidence_dict['mild_gest_htn'], ps_incidence_dict['severe_pre_eclamp'],
        ps_incidence_dict['severe_gest_htn'], ps_incidence_dict['eclampsia'],
        (ps_incidence_dict['mild_mod_antepartum_haemorrhage'] + ps_incidence_dict['severe_antepartum_haemorrhage']),
        ps_incidence_dict['syphilis']]

x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, sizes, color='green', width=0.9)
plt.xlabel("Complication")
plt.ylabel("Incidence (cases per 1000 pregnancies)")
plt.title("Incidence of complications during antenatal period ")
plt.xticks(x_pos, x, rotation=75)
plt.show()

for complication in lab_incidence_dict:
       lab_incidence_dict[complication] = (lab_incidence_dict[complication] / total_births) * 1000

x = ['OL', 'APH', 'UR', 'SGH', 'SPE', 'EC', 'SEP', 'PPH']
sizes = [lab_incidence_dict['obstructed_labour'], (lab_incidence_dict['mild_mod_antepartum_haemorrhage'] +
                                                   lab_incidence_dict['severe_antepartum_haemorrhage']),
         lab_incidence_dict['uterine_rupture'], lab_incidence_dict['severe_gest_htn'],
         lab_incidence_dict['severe_pre_eclamp'],  lab_incidence_dict['eclampsia'],
         lab_incidence_dict['sepsis'], lab_incidence_dict['primary_postpartum_haemorrhage']]

x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, sizes, color='red', width=0.9)
plt.xlabel("Complication")
plt.ylabel("Incidence (cases per 1000 deliveries)")
plt.title("Incidence of complications during labour")

plt.xticks(x_pos, x, rotation=75)
plt.show()

for complication in pn_incidence_dict:     # TODO: MAY NOT BE THE RIGHT DENOMINATOR
       pn_incidence_dict[complication] = (pn_incidence_dict[complication] / total_births) * 1000

x = ['VVF', 'RVF', 'SEP', 'SPPH', 'AN', 'MPE', 'MGH', 'SGH', 'SPE', 'EC']
sizes = [pn_incidence_dict['vesicovaginal_fistula'], pn_incidence_dict['rectovaginal_fistula'],
         pn_incidence_dict['sepsis'], pn_incidence_dict['secondary_postpartum_haemorrhage'],
         (pn_incidence_dict['mild_anaemia'] + pn_incidence_dict['moderate_anaemia'] +
          pn_incidence_dict['severe_anaemia']),   pn_incidence_dict['mild_pre_eclamp'],
         pn_incidence_dict['mild_gest_htn'], ps_incidence_dict['severe_gest_htn'],
         pn_incidence_dict['severe_pre_eclamp'], pn_incidence_dict['eclampsia']]

x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, sizes, color='orange', width=0.9)
plt.xlabel("Complication")
plt.ylabel("Incidence (cases per 1000 deliveries)")
plt.title("Incidence of complications in the postnatal period")

plt.xticks(x_pos, x, rotation=75)
plt.show()

intrapartum_comps = ['placental_abruption', 'mild_mod_antepartum_haemorrhage', 'severe_antepartum_haemorrhage',
                     'sepsis', 'uterine_rupture', 'eclampsia', 'severe_gest_htn', 'severe_pre_eclamp',
                     'early_preterm_labour', 'late_preterm_labour', 'post_term_labour', 'obstructed_labour',
                     'primary_postpartum_haemorrhage']
#  ============================================== BIRTH TERM ==========================================================

x = ['Term', 'Preterm', 'Early', 'Late', 'Post-term']
term = 100 - (lab_incidence_dict['early_preterm_labour'] +
            lab_incidence_dict['late_preterm_labour'] +
            lab_incidence_dict['post_term_labour'])

sizes = [term, (lab_incidence_dict['early_preterm_labour'] + lab_incidence_dict['late_preterm_labour']),
         lab_incidence_dict['early_preterm_labour'], lab_incidence_dict['late_preterm_labour'],
         lab_incidence_dict['post_term_labour']]

x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, sizes, color='yellowgreen', width=0.9)
plt.xlabel("Gestation at birth")
plt.ylabel("Incidence (cases per 100 deliveries)")
plt.title("Gestation at birth")

plt.xticks(x_pos, x)
plt.show()

# =====================================================================================================================
# =============================================== STILL BIRTHS ========================================================
# =====================================================================================================================

antenatal_stillbirths = get_stillbirths('pregnancy_supervisor', 'antenatal')
intrapartum_stillbirths = get_stillbirths('labour', 'intrapartum')
intrapartum_stillbirths += get_stillbirths('newborn_outcomes', 'intrapartum')

# 2.) Add livebirths to stillbirth to get stillbirth rate
total_births_2011 = (antenatal_stillbirths + intrapartum_stillbirths) + live_births

# 4.) calculate SBR and Plot
asbr = (antenatal_stillbirths / total_births_2011) * 1000
isbr = (intrapartum_stillbirths / total_births_2011) * 1000
sbr_2010 = ((antenatal_stillbirths + intrapartum_stillbirths) / total_births_2011) * 1000

objects = ('Total Model SBR', 'Calibration Target', 'Antenatal SBR', 'Intrapartum SBR')
y_pos = np.arange(len(objects))
plt.bar(y_pos, [sbr_2010, 20, asbr, isbr], align='center', alpha=0.5, color='grey')
plt.xticks(y_pos, objects)
plt.ylabel('Stillbirths/1000 live and still births')
plt.title('Stillbirth rate in 2011')
plt.show()

# =====================================================================================================================
# =============================================== NEWBORN OUTCOMES ====================================================
# =====================================================================================================================

direct_neonatal_causes = ['early_onset_neonatal_sepsis', 'encephalopathy', 'preterm_other',
                          'respiratory_distress_syndrome', 'neonatal_respiratory_depression',
                          'congenital_heart_anomaly', 'limb_or_musculoskeletal_anomaly', 'urogenital_anomaly',
                          'digestive_anomaly', 'other_anomaly', 'neonatal']

neonatal_deaths = dict()
neonatal_death_proportions = dict()

for cause in direct_neonatal_causes:
    row = {cause: 0}
    neonatal_deaths.update(row)
    neonatal_death_proportions.update(row)

for file in files:
    if 'death' in logs_dict[file]['tlo.methods.demography']:
        total_deaths = logs_dict[file]['tlo.methods.demography']['death']
        total_deaths['date'] = pd.to_datetime(total_deaths['date'])
        total_deaths['year'] = total_deaths['date'].dt.year
        for cause in neonatal_deaths:
            number_of_newborn_deaths = len(total_deaths.loc[(total_deaths['cause'] == f'{cause}') &
                                                            (total_deaths['year'] == 2011)])

            neonatal_deaths[cause] += number_of_newborn_deaths


neonatal_deaths_2011 = sum(neonatal_deaths.values())
nmr_2011 = (neonatal_deaths_2011/live_births) * 1000

objects = ('Total NMR', 'Calibration Target')
y_pos = np.arange(len(objects))
plt.bar(y_pos, [nmr_2011, 31], align='center', alpha=0.5, color='yellow')
plt.xticks(y_pos, objects)
plt.ylabel('Neonatal Deaths / 1000 live births')
plt.title('Neonatal Mortality Ratio in 2011')
plt.show()

for cause in neonatal_death_proportions.keys():
    neonatal_death_proportions[cause] = (neonatal_deaths[cause] / neonatal_deaths_2011) * 100

x = ['EONS', 'ENC', 'PTB', 'RDS', 'NRD', 'CHA', 'LA', ' UA', 'DA', 'OA', 'SEP']
sizes = list(neonatal_death_proportions.values())
x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, sizes, color='purple', width=0.9)
plt.xlabel("Cause of death")
plt.ylabel("% of Total Neonatal Deaths")
plt.title("% of Neonatal Deaths Attributed to each cause")
plt.xticks(x_pos, x)
plt.show()

labels = x
sizes = list(neonatal_death_proportions.values())
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("% of Neonatal Deaths Attributed to each cause")
plt.show()

# todo: cfrs
# =====================================================================================================================
# =============================================== HEALTH SYSTEM =======================================================
# =====================================================================================================================
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

# PNC
total_women_pnc1 = 0
total_newborns_pnc1 = 0

for file in files:
    if 'pnc_mother' in logs_dict[file]['tlo.methods.postnatal_supervisor']:
        pnc_m = logs_dict[file]['tlo.methods.postnatal_supervisor']['pnc_mother']
        pnc_m['date'] = pd.to_datetime(pnc_m['date'])
        pnc_m['year'] = pnc_m['date'].dt.year
        total_women_pnc1 += len(pnc_m.loc[(pnc_m['year'] == 2011) & (pnc_m['total_visits'] > 0)])

    if 'pnc_child' in logs_dict[file]['tlo.methods.postnatal_supervisor']:
        pnc_c = logs_dict[file]['tlo.methods.postnatal_supervisor']['pnc_child']
        pnc_c['date'] = pd.to_datetime(pnc_c['date'])
        pnc_c['year'] = pnc_c['date'].dt.year
        total_newborns_pnc1 += len(pnc_c.loc[(pnc_c['year'] == 2011) & (pnc_c['total_visits'] > 0)])

total_pnc1 = (total_women_pnc1 / total_births) * 100
total_npnc1 = (total_newborns_pnc1 / total_births) * 100

N = 2
model = [total_pnc1, total_npnc1]
calibration = (52, 52)
ind = np.arange(N)
width = 0.35
plt.bar(ind, model, width, label='Model')
plt.bar(ind + width, calibration, width,
    label='Calibration')
plt.ylabel('PNC rate')
plt.title('% of total births in which PNC is sought')
plt.xticks(ind + width / 2, ('Mother', 'Newborns'))
plt.legend(loc='best')
plt.show()


# CAESAREAN SECTION
total_cs = 0
for file in files:
    if 'caesarean_delivery' in logs_dict[file]['tlo.methods.labour']:
        cs = logs_dict[file]['tlo.methods.labour']['caesarean_delivery']
        cs['date'] = pd.to_datetime(cs['date'])
        cs['year'] = cs['date'].dt.year

        total_cs += len(cs.loc[cs['year'] == 2011])

cs_rate = (total_cs/total_births) * 100

objects = ('Model CS rate', 'Target CS rate')
y_pos = np.arange(len(objects))
plt.bar(y_pos, [cs_rate, 3.8], align='center', alpha=0.5, color='lightcoral')
plt.xticks(y_pos, objects)
plt.ylabel('Caesarean Deliveries / Total births')
plt.title('Caesarean Section Rate')
plt.show()



# todo: create a DF and output to excel?
