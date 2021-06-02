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
                                                       f"multi_run_calibration-2021-05-28T130406Z/"
                                                       f"logfiles/{file}.log")}
    logs_dict.update(new_parse_log)

# DEATH
total_direct_death = 0

for file in files:
    if 'direct_maternal_death' in logs_dict[file]['tlo.methods.labour']:
        direct_deaths_lab = logs_dict[file]['tlo.methods.labour']['direct_maternal_death']
        total_direct_death += len(direct_deaths_lab)

    if 'direct_maternal_death' in logs_dict[file]['tlo.methods.pregnancy_supervisor']:
        direct_deaths_ps = logs_dict[file]['tlo.methods.pregnancy_supervisor']['direct_maternal_death']
        total_direct_death += len(direct_deaths_ps)

    #if 'direct_maternal_death' in logs_dict[file]['tlo.methods.postnatal_supervisor']:
    #    direct_deaths_pn = logs_dict[file]['tlo.methods.postnatal_supervisor']['direct_maternal_death']
    #    total_direct_death += len(direct_deaths_pn)

        # todo: would you take the mean here?
