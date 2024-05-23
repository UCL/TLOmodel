# This script sources other scripts needed for the regression analysis and controls the workflow.

# Set file paths
path_to_local_repo <- "/Users/sm2511/PycharmProjects/TLOmodel/" # Change this if different user
path_to_dropbox <- "/Users/sm2511/Dropbox/Thanzi la Onse/" # Change this if different user
path_to_files_in_dropbox <- paste0(path_to_dropbox, "05 - Resources/Module-healthsystem/consumables raw files/")
path_to_data <- paste0(path_to_dropbox, "07 - Data/HHFA_2018-19/2 clean/")
path_to_scripts <-  paste0(path_to_local_repo, "src/scripts/data_file_processing/healthsystem/consumables/consumable_resource_analyses_with_hhfa/regression_analysis/")

# Load Libraries and Functions
source(paste0(path_to_scripts, "load_packages_and_functions.R"))

# Load Data
source(paste0(path_to_scripts, "data_setup.R"))

# Data Preparation
source(paste0(path_to_scripts, "feature_manipulation.R"))

# Regression analysis
source(paste0(path_to_scripts, "regression_analysis.R"))

# Prediction
