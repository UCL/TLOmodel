# This script performs the regression analyses
# Step 1: Chose subset of variables which will be included in the regression model - The Lancet Global Health Paper describes in
# detail how this was arrived at, therefore, the script to produce chosen_varlist is not included in master
# Link to paper - https://doi.org/10.1016/S2214-109X(24)00095-0
# Link to branch which contains the full script - https://github.com/UCL/TLOmodel/tree/consumable_scenarios/src/scripts/data_file_processing/healthsystem/consumables/consumable_resource_analyses_with_hhfa/regression_analysis

# Step 2: Run chosen regression model. See links above for a description of how the facility and item random effects
# model was chosen for the main result

####################################################################
# 1. Set up list of variables to be included in the regression model
####################################################################
# 1.1 Choose variable list for regression model
#-----------------------------------------------
chosen_varlist_orig <- c( "available",                            "fac_urban",
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
continuous_variables <- c("dist_todh", "dist_torms", "drug_order_fulfilment_freq_last_3mts")
continuous_variables_cat_version <- c("dist_todh_cat", "dist_torms_cat",
                                     "drug_order_fulfilment_freq_last_3mts_cat")

chosen_varlist <- chosen_varlist_orig[!(chosen_varlist_orig %in% continuous_variables)]
chosen_varlist <- unlist(c(chosen_varlist,continuous_variables_cat_version ))

# 1.3 Clean chosen independent variables list to ensure convergence of random effects model
# - remove highly correlated variables
# - remove variables with little variability
#-------------------------------------------------------------------------------------------
# Set up data for regression analysis
full_df <- df[chosen_varlist]
full_df <- na.omit(full_df)

# Check correlation between independent variables
varlist_check_corr <- chosen_varlist[!(chosen_varlist %in% c('district', 'fac_type', 'incharge_drug_orders',
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
chosen_varlist <- chosen_varlist[!(chosen_varlist %in% vars_low_variation)]
chosen_varlist <- chosen_varlist[!(chosen_varlist %in% vars_highly_correlated)]

# bed_count is highly correlated with outpatient_only
# very few instances of no lab facilities/no diagnostic services

# Update correlation matrix after dropping the above variables
varlist_check_corr_post <- chosen_varlist[!(chosen_varlist %in% c('district', 'fac_type', 'incharge_drug_orders',
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
# 2. Set up Dataframe for Regression analysis
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

# Set up dataframe for facility and item random effects model
#------------------------------------------------------------
# Note that this is model 4 in the Lancet GH paper. We do not include models 1-3 because these are not used for analysis on the impact of consumable availability scenarios
# add variables for random effects regression (re_reg)
chosen_varlist_for_re_reg <- unlist(c(chosen_varlist, 'program', 'fac_code', 'item', 'district', 'item_type'))

df_for_re_reg <- df_for_regs[chosen_varlist_for_re_reg]
df_for_re_reg <- na.omit(df_for_re_reg)

# Sort by fac_code
df_for_re_reg_sorted <-  df_for_re_reg[order(df_for_re_reg$fac_code, df_for_re_reg$item),]
# Create an numeric value for fac_code (clustering variable)
df_for_re_reg_sorted$fac_id <- as.integer(factor(df_for_re_reg_sorted$fac_code,levels=unique(df_for_re_reg_sorted$fac_code)))
df_for_re_reg_sorted <- as.data.frame(df_for_re_reg_sorted)

#######################################################################
# 3. Run Regression Model
#######################################################################
# The model below works but does not have service_diagnostic
model_fac_item_re <- glmer(available ~ fac_type + fac_owner + fac_urban + functional_computer +
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
                             drug_transport_self + item_drug + eml_priority_v +
                             (1|district/fac_code) + (1|program/item),
                           family = binomial(logit),
                           data = df_for_re_reg_sorted,
                           control = glmerControl(optimizer = "bobyqa",
                                                  optCtrl=list(maxfun=1e5),
                                                  calc.derivs = TRUE)
)

# Calculate the Intra-class correlation
icc_between_model_fac_item_re <- performance::icc(model_fac_item_re, by_group = TRUE)

# Save regression results
#-------------------------
save(model_fac_item_re, file = paste0(path_to_outputs, "regression_results/model_fac_item_re.rdta"))

# Summarise results in a table
#--------------------------------
t <- tbl_regression(model_fac_item_re, exponentiate = TRUE, conf.int = TRUE, pvalue_fun = ~style_sigfig(., digits = 4))

tbl_merge <-
  tbl_merge(
    tbls = list(t),
    tab_spanner = c("**Facility and Item RE**") #
  )  %>%    # build gtsummary table
  as_gt() # %>%             # convert to gt table
#  gt::gtsave(             # save table as image
#    filename = reg_results1
#  )
