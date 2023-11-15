# This script runs the main regression model for the version of the HHFA data with available not observed consumables code as NOT available

###########################################################
# 1. Data setup
###########################################################
# 1.1 Run previous setup files
#---------------------------------
source(paste0(path_to_scripts, "0_packages_and_functions.R"))
source(paste0(path_to_scripts, "1_data_setup.R"))
source(paste0(path_to_scripts, "2_feature_manipulation.R"))
source(paste0(path_to_scripts, "3b_data_setup_for_regression"))

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

# 2.4 Model with item and facility random errors  (Model 4)
#------------------------------------------------------------------
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
                             drug_transport_self + item_drug + eml_priority_v + 
                             (1|district/fac_code) + (1|program/item), 
                           family = binomial(logit),
                           data = df_for_fac_item_re_sorted, 
                           control = glmerControl(optimizer = "bobyqa",
                                                  optCtrl=list(maxfun=1e5),
                                                  calc.derivs = TRUE)
) 

# Calculate the Intra-class correlation
icc_between_model_fac_item_re = performance::icc(model_fac_item_re, by_group = TRUE)

# 3. Save regression results
###########################
save(model_fac_item_re_only_observed, file = paste0(path_to_outputs, "regression_results/model_fac_item_re_only_observed.rdta"))

# 4. Summarise results in a table
##################################
t_only_observed <- tbl_regression(model_fac_item_re_only_observed, exponentiate = TRUE, conf.int = TRUE)


# Create the combined plot for manuscript
#-----------------------------------------------------------
load(paste0(path_to_outputs, "regression_results/model_fac_item_re_only_observed.rdta"))
filename =  paste0(path_to_outputs, "figures/regression_plot_only_observed.png")
custom_forest_plot(model = model_fac_item_re) #, xlab = "Odds ratio with 95% Confidence Interval"
png(filename, units="in", width=8, height=5, res=300)
cowplot::plot_grid(data_table,p, align = "h", rel_widths = c(0.65, 0.35))
dev.off()
