# This script sources other scripts needed for the regression analysis and controls the workflow.

# Load Libraries and Functions
source(paste0(path_to_scripts, "load_packages_and_functions.R"))

# Load Data
source(paste0(path_to_scripts, "data_setup.R"))

# Data Preparation
source(paste0(path_to_scripts, "feature_manipulation.R"))

# Regression analysis
source(paste0(path_to_scripts, "regression_analysis.R"))

# Prediction
