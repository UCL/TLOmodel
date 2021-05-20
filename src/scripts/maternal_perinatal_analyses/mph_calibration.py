import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tlo.analysis.utils import parse_log_file

# ================================= Maternal Mortality Ratio =========================================================
log_df = parse_log_file(filepath="./outputs/joes_log_file.log")

# define the log DFs required
# 1.) Calculate indirect deaths
total_deaths = log_df['tlo.methods.demography']['death']
total_deaths['date'] = pd.to_datetime(total_deaths['date'])
total_deaths['year'] = total_deaths['date'].dt.year

indirect_deaths_2010 = len(total_deaths.loc[total_deaths['pregnancy'] & (total_deaths['cause'] != 'maternal') &
                                            (total_deaths['year'] == 2010)])

# todo: specificy what causes are acceptable to be defined as indirect


# 2.) calculate direct deaths
def get_direct_deaths(module):
    if f'tlo.methods.{module}' in log_df:
        if 'direct_maternal_death' in log_df[f'tlo.methods.{module}']:
            direct_deaths = log_df[f'tlo.methods.{module}']['direct_maternal_death']
            direct_deaths['date'] = pd.to_datetime(direct_deaths['date'])
            direct_deaths['year'] = direct_deaths['date'].dt.year

            return len(direct_deaths.loc[direct_deaths['year'] == 2010])
        else:
            return 0
    else:
        return 0

direct_deaths_2010 = 0
direct_deaths_2010 += get_direct_deaths('labour')
direct_deaths_2010 += get_direct_deaths('pregnancy_supervisor')
direct_deaths_2010 += get_direct_deaths('postnatal_supervisor')

total_maternal_deaths_2010 = indirect_deaths_2010 + direct_deaths_2010


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


live_births_2010 = 0
live_births_2010 += get_live_births('labour')
live_births_2010 += get_live_births('newborn_outcomes')

# 4.) calculate MMR and Plot
mmr_2010 = (total_maternal_deaths_2010 / live_births_2010) * 100000

objects = ('Model MMR', 'Calibration')
y_pos = np.arange(len(objects))
plt.bar(y_pos, [mmr_2010, 675], align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Maternal deaths/100,000 births')
plt.title('Maternal mortality rate in 2010')
plt.show()

# ========================================= Still Birth Rate =========================================================
# 1.) Calculate the number of stillbirths







# ================================= Neonatal Mortality Ratio =========================================================






# TODO: DOCFR
# TODO: CFR PPH
# TODO: % DEATHS DUE TO PPH
# TODO: CFR SEPSIS
# TODO: % DEATHS DUE TO SEPSIS
# TODO: CFR SPE/EC
# TODO: % DEATHS DUE TO SPE/EC
# TODO: % DEATHS DUE TO INDIRECT CAUSES

#
#
# TODO: SBR (AN/IP)
# TODO: NMR
# TODO : FACILITY DELIVERY RATE
# TODO: HOSPITAL VS HEALTH CENTRE
# TODO: ANC1, ANC4+, MEDIAN FIRST VISIT MONTH, EARLY ANC1
# TODO: PNC1 MATERNAL, PNC1 NEWBORNS
# TODO: CAESAREAN SECTION RATE

