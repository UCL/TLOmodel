import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tlo.analysis.utils import parse_log_file

# ================================= Maternal Mortality Ratio =========================================================
log_df = parse_log_file(filepath="./outputs/sejjj49@ucl.ac.uk/long_run_2010_calib-2021-05-24T133049Z/0/0/"
                                 "long_run_2010__2021-05-24T133357.log")

# define the log DFs required
# 1.) Calculate indirect deaths
total_deaths = log_df['tlo.methods.demography']['death']
total_deaths['date'] = pd.to_datetime(total_deaths['date'])
total_deaths['year'] = total_deaths['date'].dt.year

indirect_deaths_preg_2010 = len(total_deaths.loc[total_deaths['pregnancy'] & (total_deaths['cause'] != 'maternal') &
                                             (total_deaths['year'] == 2010)])
indirect_deaths_postnatal_2010 = len(total_deaths.loc[total_deaths['postnatal'] & (total_deaths['cause'] != 'maternal') &
                                            (total_deaths['year'] == 2010)])
# todo: specificy what causes are acceptable to be defined as indirect


# 2.) calculate direct deaths
def get_direct_deaths(module):
    if f'tlo.methods.{module}' in log_df:
        if 'direct_maternal_death' in log_df[f'tlo.methods.{module}']:
            direct_deaths = len(log_df[f'tlo.methods.{module}']['direct_maternal_death'])
            # direct_deaths['date'] = pd.to_datetime(direct_deaths['date'])
            # direct_deaths['year'] = direct_deaths['date'].dt.year

            return direct_deaths
        else:
            return 0
    else:
        return 0

direct_deaths_2010 = 0
direct_deaths_2010 += get_direct_deaths('labour')
direct_deaths_2010 += get_direct_deaths('pregnancy_supervisor')
direct_deaths_2010 += get_direct_deaths('postnatal_supervisor')

total_maternal_deaths_2010 = indirect_deaths_preg_2010 + indirect_deaths_postnatal_2010 + direct_deaths_2010


# 3.) calculate live births

def get_live_births(module):
    if f'tlo.methods.{module}' in log_df:
        if 'live_birth' in log_df[f'tlo.methods.{module}']:
            live_births = log_df[f'tlo.methods.{module}']['live_birth']
            live_births['date'] = pd.to_datetime(live_births['date'])
            live_births['year'] = live_births['date'].dt.year

            return len(live_births.loc[live_births['year'] == 2010])
        else:
            return 0
    else:
        return 0

# TODO: THIS OUTPUTS MORE BIRTHS THAN ON_BIRTH LOGGING FROM DEMOGRAPHY
live_births_2010 = 0
live_births_2010 += get_live_births('labour')
live_births_2010 += get_live_births('newborn_outcomes')

# 4.) calculate MMR and Plot
mmr_2010 = (total_maternal_deaths_2010 / live_births_2010) * 100000

objects = ('Model MMR', 'Calibration')
y_pos = np.arange(len(objects))
plt.bar(y_pos, [mmr_2010, 675], align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Maternal deaths/100,000 live births')
plt.title('Maternal mortality rate in 2010')
plt.show()

# Alternative plot with enumeration (but sideways, annoyingly)
x = [u'MMR (model)', u'MMR Calib.']
y = [mmr_2010, 675]
fig, ax = plt.subplots()
width = 0.75 # the width of the bars
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color="green")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
plt.title('Maternal Mortality Ratio 2010')
plt.xlabel('Maternal Deaths per 100,000 live births')
plt.ylabel('y')
for i, v in enumerate(y):
    ax.text(v + 3, i + .25, str(v), color='black', fontweight='bold')
plt.show()

# ========================================= Still Birth Rate =========================================================

# 1.) Calculate the number of stillbirths
def get_stillbirths(module, pregnancy_period):
    if f'tlo.methods.{module}' in log_df:
        if f'{pregnancy_period}_stillbirth' in log_df[f'tlo.methods.{module}']:
            stillbirths = log_df[f'tlo.methods.{module}'][f'{pregnancy_period}_stillbirth']
            stillbirths['date'] = pd.to_datetime(stillbirths['date'])
            stillbirths['year'] = stillbirths['date'].dt.year

            return len(stillbirths.loc[stillbirths['year'] == 2010])
        else:
            return 0
    else:
        return 0


antenatal_stillbirths = get_stillbirths('pregnancy_supervisor', 'antenatal')
intrapartum_stillbirths = get_stillbirths('labour', 'intrapartum')
intrapartum_stillbirths += get_stillbirths('newborn_outcomes', 'intrapartum')

# 2.) Add livebirths to stillbirth to get stillbirth rate
total_births_2010 = (antenatal_stillbirths + intrapartum_stillbirths) + live_births_2010

# 4.) calculate SBR and Plot
asbr = (antenatal_stillbirths / total_births_2010) * 1000
isbr = (intrapartum_stillbirths / total_births_2010) * 1000
sbr_2010 = ((antenatal_stillbirths + intrapartum_stillbirths) / total_births_2010) * 1000

objects = ('Total Model SBR', 'Antenatal SBR', 'Intrapartum SBR', 'Calibration')
y_pos = np.arange(len(objects))
plt.bar(y_pos, [sbr_2010, asbr, isbr, 20], align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Stillbirths/1000 live and still births')
plt.title('Stillbirth rate in 2010')
plt.show()

# ================================= Neonatal Mortality Ratio =========================================================
# 1.) Select deaths of neonates
neonatal_deaths_2010 = len(total_deaths.loc[total_deaths['cause'] == 'neonatal'])
# TODO: CAPTURE ANY OTHER DEATHS IN THE NEONATAL PERIOD

early_neonatal_deaths = 0

# 2.) Calculate NMR
enmr_2010 = (early_neonatal_deaths / live_births_2010) * 1000
nmr_2010 = (neonatal_deaths_2010 / live_births_2010) * 1000

# 3.) PLOT
# TODO: plot early NMR

objects = ('Total NMR', 'Calibration Total NMR')
y_pos = np.arange(len(objects))
plt.bar(y_pos, [nmr_2010, 31], align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Neonatal Deaths / 1000 live births')
plt.title('Neonatal Mortality Ratio in 2010')
plt.show()

# ================================= Facility Delivery  =========================================================
# 1.) Calculate the number of facility deliveries
hospital_deliveries = 0
health_centre_deliveries = 0

if f'tlo.methods.labour' in log_df:
    if 'facility_delivery' in log_df['tlo.methods.labour']:

        facility_deliveries = log_df[f'tlo.methods.labour']['facility_delivery']
        facility_deliveries['date'] = pd.to_datetime(facility_deliveries['date'])
        facility_deliveries['year'] = facility_deliveries['date'].dt.year

        hospital_deliveries = len(facility_deliveries.loc[facility_deliveries['facility_type'] == 'hospital'])
        health_centre_deliveries = len(facility_deliveries.loc[facility_deliveries['facility_type'] == 'health_centre'])

# 2.) calculate rates
total_deliveries = live_births_2010 + intrapartum_stillbirths

hpd_rate_2010 = (hospital_deliveries / total_deliveries) * 100
hcd_rate_2010 = (health_centre_deliveries / total_deliveries) * 100
fd_rate_2010 = ((hospital_deliveries + health_centre_deliveries) / total_deliveries) * 100

# 3) plot
objects = ('Total FDR', 'Hospital DR', 'Health Centre DR', 'Calibration')
y_pos = np.arange(len(objects))
plt.bar(y_pos, [fd_rate_2010, hpd_rate_2010, hcd_rate_2010, 73], align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Facility Deliveries/ Total births')
plt.title('Facility Delivery Rate 2010')
plt.show()

# =============================================== Antenatal Care =====================================================
# 1.) Calculate rates of ANC1, 4+, EARLY
total_anc1 = 0
early_anc1 = 0
total_anc4 = 0
total_anc8 = 0

anc1 = log_df['tlo.methods.care_of_women_during_pregnancy']['anc1']
anc1['date'] = pd.to_datetime(anc1['date'])
anc1['year'] = anc1['date'].dt.year

total_anc = log_df['tlo.methods.care_of_women_during_pregnancy']['anc_count_on_birth']
total_anc['date'] = pd.to_datetime(total_anc['date'])
total_anc['year'] = total_anc['date'].dt.year

total_anc1 = len(total_anc.loc[total_anc['total_anc'] > 0])
early_anc1 = len(anc1.loc[(anc1['year'] == 2010) & (anc1['gestation'] <= 17)])
anc1_months_5 = len(anc1.loc[(anc1['year'] == 2010) & (anc1['gestation'] > 17) & (anc1['gestation'] < 22)]) #calibrating to months 5-6
anc1_months_6_7 = len(anc1.loc[(anc1['year'] == 2010) & (anc1['gestation'] > 22) & (anc1['gestation'] < 35)]) #calibrating to months 5-6
anc1_months_8 = len(anc1.loc[(anc1['year'] == 2010) & (anc1['gestation'] > 34)])

total_anc4 = len(total_anc.loc[total_anc['total_anc'] > 3])
total_anc8 = len(total_anc.loc[total_anc['total_anc'] > 7])


# 2.) calculate the rates
anc1_rate = (total_anc1 / len(total_anc)) * 100
anc4_rate = (total_anc4 / len(total_anc)) * 100
anc8_rate = (total_anc8 / len(total_anc)) * 100

objects = ('ANC 1', 'Calibration', 'ANC4+', 'Calibration', 'ANC8+')
y_pos = np.arange(len(objects))
plt.bar(y_pos, [anc1_rate, 94.7, anc4_rate, 46, anc8_rate], align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Antenatal Care / Live births')
plt.title('Antenatal Care Coverage Rate 2010')
plt.show()

# 3.)Plot



# TODO: ANC MEDIAN FIRST VISIT MONTH,
# TODO: PNC1 MATERNAL, PNC1 NEWBORNS
# TODO: CAESAREAN SECTION RATE
# TODO: DOCFR
# TODO: CFR PPH
# TODO: % DEATHS DUE TO PPH
# TODO: CFR SEPSIS
# TODO: % DEATHS DUE TO SEPSIS
# TODO: CFR SPE/EC
# TODO: % DEATHS DUE TO SPE/EC
# TODO: % DEATHS DUE TO INDIRECT CAUSES


