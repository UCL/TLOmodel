"""
An analysis file for the wasting module to visualise availability of:
    * essential consumables,
    * treatments (i.e., probability of all consumables essential for the treatment being available)
"""

from pathlib import Path

import analysis_utility_functions_wast

# ####### TO SET #######################################################################################################
# Where to save the outcomes
outputs_path = Path("./outputs/sejjej5@ucl.ac.uk/wasting/scenarios/_outcomes")
########################################################################################################################

analysis_utility_functions_wast.plot_availability_heatmaps(outputs_path)
