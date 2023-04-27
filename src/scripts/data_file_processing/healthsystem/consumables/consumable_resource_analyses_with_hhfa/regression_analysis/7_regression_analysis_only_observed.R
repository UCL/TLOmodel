# This script performs the regression analysis based on the model suggested by 
# 3_pre_regression_analysis.R

# setwd() # use if running locally

# The script below runs the stepwise algorithm for the selection of control variables. This takes 5-6 hours to run so it has been replaced by the output below
#source("0 scripts/3_pre_regression_analysis.R") 

###########################################################
# 1. Data setup
###########################################################
# 1.1 Run previous setup files
#---------------------------------
source("0 scripts/0_packages_and_functions.R")
source("0 scripts/1b_data_setup_only_observed.R")
source("0 scripts/2_feature_manipulation.R")
#source("0 scripts/3_pre_regression_analysis.R")

# Output of 3_pre_regression_analysis.R
chosen_varlist_orig = c( "available",                       "fac_urban",                           
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
                         "rms",                                  "item_drug")
# rms added on 23 Sep 2022
# item_drug added on 17 oct 2022

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

#######################################################################
# 2. Regression analysis
#######################################################################
# 2.3 Model with facility random errors
#--------------------------------------
# Set up data for regression analysis
chosen_varlist_for_fac_re <-  chosen_varlist[!(chosen_varlist %in% c('item'))] 
chosen_varlist_for_fac_re <- unlist(c(chosen_varlist_for_fac_re, 'program', 'fac_code', 'item', 'district', 'item_type'))

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

# Make a list of facilities with less than 10% facilities reporting 
print(max(df_not_na_by_fac$available_count)) 
facs_with_too_few_obs <- subset(df_not_na_by_fac, df_not_na_by_fac$available_count <= 0.1*max(df_not_na_by_fac$available_count))['fac_code']
facs_with_too_few_obs <- as.list(facs_with_too_few_obs)
# %% the aobve was previously 90 %%

df_for_fac_re <- df[chosen_varlist_for_fac_re]
df_for_fac_re <- na.omit(df_for_fac_re)

# Drop above facitlities from the dataset
df_for_fac_re <- subset(df_for_fac_re, !(fac_code %in% facs_with_too_few_obs$fac_code))
df_for_fac_re <- subset(df_for_fac_re, !(item %in% items_with_too_few_obs$item))
print(paste(length(facs_with_too_few_obs$fac_code), " facilities dropped."))
print(paste(length(items_with_too_few_obs$item), " items dropped."))

# Sort by fac_code and item
df_for_fac_re_sorted <-  df_for_fac_re[order(df_for_fac_re$fac_code),]
# Create an numeric value for fac_code (clustering variable)
df_for_fac_re_sorted$fac_id <- as.integer(factor(df_for_fac_re_sorted$fac_code,levels=unique(df_for_fac_re_sorted$fac_code)))
df_for_fac_re_sorted <- as.data.frame(df_for_fac_re_sorted)

# 2.4 Model with item and facility random errors
#--------------------------------------------------
# Drop above items from the dataset (# already dropped)
df_for_fac_item_re_sorted <- subset(df_for_fac_re_sorted, !(item %in% items_with_too_few_obs$item))

# The model below works but does not have service_diagnostic
model_fac_item_re_only_observed <- glmer(available ~ fac_type + fac_owner + fac_urban + functional_computer +
                             functional_emergency_vehicle + service_diagnostic +
                             incharge_drug_orders +
                             dist_todh_cat + dist_torms_cat +
                             drug_order_fulfilment_freq_last_3mts_cat +  rms +
                             functional_refrigerator +  functional_landline + fuctional_mobile +
                             functional_toilet +  functional_handwashing_facility +
                             water_source_main +
                             outpatient_only + 
                             service_hiv + service_othersti + 
                             service_malaria + service_tb + 
                             service_fp + service_imci +  
                             source_drugs_ngo +  source_drugs_pvt + 
                             drug_transport_self + item_drug +
                             (1|district/fac_code) + (1|program/item), 
                           family = binomial(logit),
                           data = df_for_fac_item_re_sorted, 
                           control = glmerControl(optimizer = "bobyqa",
                                                  optCtrl=list(maxfun=1e5),
                                                  calc.derivs = TRUE)
) 

# 3. Save regression results
###########################
save(model_fac_item_re_only_observed, file = "2 outputs/regression_results/model_fac_item_re_only_observed.rdta")

# 4. Summarise results in a table
##################################
t_only_observed <- tbl_regression(model_fac_item_re_only_observed, exponentiate = TRUE, conf.int = TRUE)

