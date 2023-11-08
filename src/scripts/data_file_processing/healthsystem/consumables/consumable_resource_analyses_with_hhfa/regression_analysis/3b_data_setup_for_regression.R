# This script creates dataframes for various regression models. The list of control variables is chosen 
# based on results from  3_pre_regression_analysis.R - we don't need to run this script again because the output has been
# copied into this script. 

# setwd() # use if running locally

###########################################################
# 1. Data setup
###########################################################
# 1.1 Run previous setup files
#---------------------------------
source(paste0(path_to_scripts, "0_packages_and_functions.R"))
source(paste0(path_to_scripts, "1_data_setup.R"))
source(paste0(path_to_scripts, "2_feature_manipulation.R"))

# The script below runs the stepwise algorithm for the selection of control variables. This takes 5-6 hours to run so it has been replaced by the output below
#source("0 scripts/3_pre_regression_analysis.R") 

# Output of 3_pre_regression_analysis.R
chosen_varlist_orig = c( "available",                            "fac_urban",                           
                         "fac_type" ,                            "functional_computer",                 
                         "incharge_drug_orders",                 "functional_emergency_vehicle",        
                         "dist_todh",                            "dist_torms",                          
                         "drug_order_fulfilment_freq_last_3mts", "service_diagnostic",                  
                         "item",                                 "functional_refrigerator",             
                         "fac_owner",                            "service_fp",                          
                         "service_othersti",                     "functional_handwashing_facility",     
                         "service_hiv",                          "service_malaria",                     
                         "source_drugs_ngo",                     "service_cvd",                         
                         "service_tb",                           "bed_count",                           
                         "outpatient_only",                      "water_source_main",                   
                         "functional_toilet",                    "functional_landline",                 
                         "fuctional_mobile",                     "service_imci",                        
                         "drug_transport_self",                  "source_drugs_pvt",
                         "rms",                                  "item_drug",
                         "eml_priority_v")

# 1.2 Setting up categorical variables for regression analysis
#--------------------------------------------------------------
# The categorical version of continuous variables is used in the first instance.
# In robustness checks, this will be replaced by the continuous version
continuous_variables = c("dist_todh", "dist_torms", "drug_order_fulfilment_freq_last_3mts")
continuous_variables_cat_version = c("dist_todh_cat", "dist_torms_cat", 
                                     "drug_order_fulfilment_freq_last_3mts_cat")

chosen_varlist = chosen_varlist_orig[!(chosen_varlist_orig %in% continuous_variables)]
chosen_varlist = unlist(c(chosen_varlist,continuous_variables_cat_version ))

# 1.3 Clean chosen independent variables list to ensure convergence of random effects model
# - remove highly correlated variables
# - remove variables with little variability
#-------------------------------------------------------------------------------------------
# Set up data for regression analysis
full_df <- df[chosen_varlist]
full_df <- na.omit(full_df)

# Check correlation between independent variables 
varlist_check_corr = chosen_varlist[!(chosen_varlist %in% c('district', 'fac_type', 'incharge_drug_orders', 
                                                            'fac_owner', 'water_source_main', 'item', 'rms', 
                                                            continuous_variables_cat_version))] # 'program', 'mode_administration',
correlation_final_varlist <- cor(full_df[varlist_check_corr])
corrplot_final_varlist <- ggcorrplot(correlation_final_varlist, lab_size = 1.5, p.mat = NULL, 
                                     insig = c("pch", "blank"), pch = 1, pch.col = "black", pch.cex =1,
                                     tl.cex =5.5, lab = TRUE)
ggsave(plot = corrplot_final_varlist, filename = paste0(path_to_outputs, "figures/correlation_final_varlist.png"))

# Based on the above results, drop variables on account of reasons listed below
vars_low_variation <- c('service_cvd') # %% service_diagnostic was previously dropped %%
vars_highly_correlated <- c('bed_count') # highly correlated with outpatient_only
chosen_varlist = chosen_varlist[!(chosen_varlist %in% vars_low_variation)] 
chosen_varlist = chosen_varlist[!(chosen_varlist %in% vars_highly_correlated)] 

# bed_count is highly correlated with outpatient_only
# very few instances of no lab facilities/no diagnostic services

# Update correlation matrix after dropping the above variables
varlist_check_corr_post = chosen_varlist[!(chosen_varlist %in% c('district', 'fac_type', 'incharge_drug_orders', 
                                                                 'fac_owner', 'water_source_main', 'item','rms',
                                                                 continuous_variables_cat_version))] # 'program', 'mode_administration',
correlation_final_varlist_post <- cor(full_df[varlist_check_corr_post])
corrplot_final_varlist_post <- ggcorrplot(correlation_final_varlist_post, lab_size = 1.5, p.mat = NULL, 
                                          insig = c("pch", "blank"), pch = 1, pch.col = "black", pch.cex =1,
                                          tl.cex =5.5, lab = TRUE)

ggsave(plot = corrplot_final_varlist_post, filename = paste0(path_to_outputs, "figures/correlation_final_varlist_post.png"))


# List of binary variables
bin_exp_vars <- c('fac_urban', 'functional_emergency_vehicle' ,'functional_computer', 
                  'service_diagnostic' , 'functional_refrigerator', 'functional_landline',
                  'fuctional_mobile', 'functional_toilet', 'functional_handwashing_facility', 
                  'outpatient_only', 'service_hiv', 'service_othersti', 'service_malaria',
                  'service_tb', 'service_fp', 'service_imci', 'source_drugs_ngo', 
                  'source_drugs_pvt', 'drug_transport_self', 'item_drug', "eml_priority_v")


#######################################################################
# 2. Set up Dataframes for Regression analysis
#######################################################################
# Drop items and facilities with too few facilities reporting/items reported respectively
# Drop items and facs with very few observations
df_not_na_by_fac <- df %>% 
  group_by(fac_code) %>% 
  summarise(available_count = sum(!is.na(available))) %>%
  arrange(available_count)

df_not_na_by_item <- df %>% 
  group_by(item) %>% 
  summarise(available_count = sum(!is.na(available))) %>%
  arrange(available_count)

# Make a list of items with less than 10% facilities reporting (these are the consumables relevant to higher level RMNCH services for which
# only 64 facilities submitted reports)
items_with_too_few_obs <- subset(df_not_na_by_item, df_not_na_by_item$available_count < 100)['item']
items_with_too_few_obs <- as.list(items_with_too_few_obs)

# Make a list of facilities with less than 10% items reported 
print(max(df_not_na_by_fac$available_count)) 
facs_with_too_few_obs <- subset(df_not_na_by_fac, df_not_na_by_fac$available_count <= 0.1*max(df_not_na_by_fac$available_count))['fac_code']
facs_with_too_few_obs <- as.list(facs_with_too_few_obs)

df_for_regs <- subset(df, !(fac_code %in% facs_with_too_few_obs$fac_code))
df_for_regs <- subset(df_for_regs, !(item %in% items_with_too_few_obs$item))

print(paste(length(facs_with_too_few_obs$fac_code), " facilities dropped."))
print(paste(length(items_with_too_few_obs$item), " items dropped."))

# 2.1 Logistic regression model with independent vars based on literature (Model 1)
#--------------------------------------------------------------------------------
# Set up data for regression analysis
stockout_factors_from_lit <- stockout_factors_from_lit[!(stockout_factors_from_lit %in%
                                                           vars_low_variation)]
stockout_factors_from_lit <- stockout_factors_from_lit[!(stockout_factors_from_lit %in%
                                                           vars_highly_correlated)]

stockout_factors_from_lit = stockout_factors_from_lit[!(stockout_factors_from_lit %in% continuous_variables)]
stockout_factors_from_lit = unlist(c(stockout_factors_from_lit,continuous_variables_cat_version ))

df_for_lit <- df_for_regs[unlist(c(stockout_factors_from_lit, 'available'))]
df_for_lit <- na.omit(df_for_lit)

# 2.2 Basic Logistic regression model with facility-features from stepwise  (Model 2)
#------------------------------------------------------------------------------------------
# Set up data for regression analysis
chosen_varlist_for_base <-  chosen_varlist[!(chosen_varlist %in% c('item'))] 
# add program and district fixed effects
chosen_varlist_for_base <- unlist(c(chosen_varlist_for_base, 'program', 'district'))

df_for_base <- df_for_regs[chosen_varlist_for_base]
df_for_base <- na.omit(df_for_base)

# 2.3 Model with facility random errors (Model 3)
#------------------------------------------------------
# Set up data for regression analysis
chosen_varlist_for_fac_re <-  chosen_varlist[!(chosen_varlist %in% c('item'))] 
chosen_varlist_for_fac_re <- unlist(c(chosen_varlist_for_fac_re, 'program', 'fac_code', 'item', 'district', 'item_type'))

df_for_fac_re <- df_for_regs[chosen_varlist_for_fac_re]
df_for_fac_re <- na.omit(df_for_fac_re)

# Sort by fac_code and item
df_for_fac_re_sorted <-  df_for_fac_re[order(df_for_fac_re$fac_code),]
# Create an numeric value for fac_code (clustering variable)
df_for_fac_re_sorted$fac_id <- as.integer(factor(df_for_fac_re_sorted$fac_code,levels=unique(df_for_fac_re_sorted$fac_code)))
df_for_fac_re_sorted <- as.data.frame(df_for_fac_re_sorted)

# 2.4 Model with item and facility random errors  (Model 4)
#------------------------------------------------------------------
df_for_fac_item_re_sorted <- df_for_fac_re_sorted
