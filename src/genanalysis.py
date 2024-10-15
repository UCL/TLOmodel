import sys
# from scripts.enhanced_lifestyle_analyses.enhanced_lifestyle_analyses import LifeStylePlots, run
from scripts.enhanced_lifestyle_analyses.enhanced_lifestyle_calibrations import LifeStyleCalibration, run
# from scripts.enhanced_lifestyle_analyses.enhanced_lifestyle_calibrations import run
from tlo.analysis.utils import parse_log_file

def run_lifestyle_calibration():
    sim = run()

    # %% read the results
    output = parse_log_file(sim.log_filepath)

    # output = parse_log_file(Path("./outputs/enhanced_lifestyle__2023-01-26T091835.log"))

    # construct a dict of dataframes using lifestyle logs
    logs_df = output['tlo.methods.enhanced_lifestyle']

    # initialise LifeStyleCalibration class
    g_plots = LifeStyleCalibration(logs=logs_df, path="./outputs")

    # calibrate lifestyle properties
    g_plots.display_all_properties_plots()


