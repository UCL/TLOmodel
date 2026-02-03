"""
Visualise the calibration outcomes (i.e. comparison of modelled outcomes to data):
    * prevalence of moderate and severe wasting among age groups in 2016 & 2020,
    * average annual direct deaths due to SAM
"""

from pathlib import Path

import analysis_utility_functions_wast

# ####### TO SET #######################################################################################################
# Where to save the outcomes
calib_outputs_path = Path("./outputs/sejjej5@ucl.ac.uk/wasting/scenarios/_outcomes/calibration")
########################################################################################################################

analysis_utility_functions_wast.plot_calibration_outputs(calib_outputs_path)
