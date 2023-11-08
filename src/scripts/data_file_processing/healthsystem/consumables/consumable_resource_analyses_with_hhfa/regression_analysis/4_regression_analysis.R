# This script performs the regression analysis based on the model suggested by 
# 3_pre_regression_analysis.R - we don't need to run this script again because the output has been
# copied into this script. 

# setwd() # use if running locally


###########################################################
# 1. Data setup
###########################################################
# 1.1 Run setup script
#---------------------------------
source(paste0(path_to_scripts, "3b_data_setup_for_regression.R"))

#######################################################################
# 2. Regression analysis
#######################################################################
# 2.1 Logistic regression model with independent vars based on literature (Model 1)
#--------------------------------------------------------------------------------
model_lit <- glm(available ~ .,
                 data = df_for_lit,
                 family = "binomial")

# 2.2 Basic Logistic regression model with facility-features from stepwise  (Model 2)
#------------------------------------------------------------------------------------------
model_base <- glm(available ~ .,
                  data = df_for_base,
                  family = "binomial")

# 2.3 Model with facility random errors (Model 3)
#------------------------------------------------------
model_fac_re <- glmer(available ~ fac_type + fac_owner + fac_urban + functional_computer +
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
                        drug_transport_self + program + item_drug + eml_priority_v +
                        (1|district/fac_code), 
                      family = binomial(logit),
                      data = df_for_fac_re_sorted,
                      control = glmerControl(optimizer = "bobyqa",
                                             optCtrl=list(maxfun=1e5),
                                             calc.derivs = TRUE)) 

# Add option if the above model does not converge - optCtrl=list(maxfun=10000) 

# dropped service_cvd since nearly fully explained by fac_type; 
# calc.derivs = FALSE - turns off the series of finite-difference calculations to estimate the 
# gradient and Hessian at the MLE - these are used to establish whether the model has converged reliably

# Calculate the Intra-class correlation
# Compute the intra-class correlation (ICC) as the ratio of the random intercept variance 
# (between-person) to the total variance, defined as the sum of the random intercept variance 
# and residual variance (between + within)

random_effects_model_fac_re <- performance::icc(model_fac_re)

# From the unconditional means model, the ICC was calculated, which indicated that of the total variance
# in consumable availability, 2.6% is attributable to between-facility variation whereas 
# 97.4% is attributatable to within-facility variation.

# 2.4 Model with item and facility random errors  (Model 4)
#------------------------------------------------------------------
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
                           data = df_for_fac_item_re_sorted, 
                           control = glmerControl(optimizer = "bobyqa",
                                                  optCtrl=list(maxfun=1e5),
                                                  calc.derivs = TRUE)
) 

# Calculate the Intra-class correlation
icc_between_model_fac_item_re = performance::icc(model_fac_item_re, by_group = TRUE)

#From the unconditional means model, the ICC was calculated, which indicated that of the total variance
# in consumable availability, X% is attributable to between-facility variation whereas 
# 1-x is attributatable to within-facility variation.

# For the main regression model, calculate average marginal effects for the manuscript text
varlist_margins <- c("fac_type", "fac_owner", "fac_urban", "functional_computer", "functional_emergency_vehicle",
                     "service_diagnostic", "incharge_drug_orders", "item_drug", "dist_todh_cat", "dist_torms_cat",
                     "drug_order_fulfilment_freq_last_3mts_cat")
margins_fac_item_re <- margins(model_fac_item_re, type = "response", variables = varlist_margins)
save(margins_fac_item_re, file = "2 outputs/regression_results/margins_fac_item_re.rdta")
write_xlsx(margins_fac_item_re,"2 outputs/tables/margins_fac_item_re.xlsx")

# Save regression results
###########################
save(model_lit, file = paste0(path_to_outputs, "regression_results/model_lit.rdta"))
save(model_base, file = paste0(path_to_outputs, "regression_results/model_base.rdta"))
save(model_fac_re, file = paste0(path_to_outputs, "regression_results/model_fac_re.rdta"))
save(model_fac_item_re, file = paste0(path_to_outputs, "regression_results/model_fac_item_re.rdta"))

# 3. Summarise results in a table
##################################
t1 <- tbl_regression(model_lit, exponentiate = TRUE, conf.int = TRUE, pvalue_fun = ~style_sigfig(., digits = 4))
t2 <- tbl_regression(model_base, exponentiate = TRUE, conf.int = TRUE, pvalue_fun = ~style_sigfig(., digits = 4)) # R often hangs while running this line. If this happens, run this line separately. 
t3 <- tbl_regression(model_fac_re, exponentiate = TRUE, conf.int = TRUE, pvalue_fun = ~style_sigfig(., digits = 4))
t4 <- tbl_regression(model_fac_item_re, exponentiate = TRUE, conf.int = TRUE, pvalue_fun = ~style_sigfig(., digits = 4))

tbl_merge <-
  tbl_merge(
    tbls = list(t1, t2, t3, t4),
    tab_spanner = c("**Model based on literature**", "**Base model**", 
                    "**Facility RE**", "**Facility and Item RE**") #
  )  %>%    # build gtsummary table
  as_gt() # %>%             # convert to gt table
#  gt::gtsave(             # save table as image
#    filename = reg_results1
#  )



# 4. Sub-group analyses
##############################

# 4.1.1 Regression by program (consumable random effects)
#----------------------------------------------------------
group_by_program <- 
  df_for_fac_item_re_sorted %>%                             
  group_by(program) %>%
  summarise(unique_items = n_distinct(item))

# run regression for programs which have sufficient items
programs_for_reg <- c('general', 'hiv', 'malaria', 'tb', 'obstetric and newborn care',
                      'contraception','child health', 'epi','acute lower respiratory infections',
                      'ncds',  'surgical')

chosen_varlist_for_prog <- chosen_varlist_for_base[!(chosen_varlist_for_base %in% c('program'))]
i = 1
model_prog_summaries <- list()      # Create empty list to store model outputs
model_prog_ci <- list() # Create empty list to store confidence intervals
prog_regs_stats <- matrix(NaN, nrow = length(programs_for_reg) , ncol = 4) # empty list for data counts

j = 1

for (prog in programs_for_reg){
  print(paste("running program", prog))
  df_for_fac_item_re_sorted_prog <- df_for_fac_item_re_sorted[which(df_for_fac_item_re_sorted$program == prog),]
  # Regression
  model_prog_summaries[[i]] <-glmer(available ~ fac_type + fac_owner + fac_urban + functional_computer +
                                      functional_emergency_vehicle + service_diagnostic +
                                      incharge_drug_orders +
                                      dist_todh_cat + dist_torms_cat +
                                      drug_order_fulfilment_freq_last_3mts_cat + rms +
                                      functional_refrigerator +  functional_landline + fuctional_mobile +
                                      functional_toilet +  functional_handwashing_facility +
                                      water_source_main +
                                      outpatient_only + 
                                      service_hiv + service_othersti + 
                                      service_malaria + service_tb + 
                                      service_fp + service_imci +  
                                      source_drugs_ngo +  source_drugs_pvt + 
                                      drug_transport_self +
                                      (1|fac_code) + (1|item), 
                                    family = binomial(logit),
                                    data = df_for_fac_item_re_sorted_prog, 
                                    control = glmerControl(optimizer = "bobyqa",
                                                           optCtrl=list(maxfun=1e5),
                                                           calc.derivs = TRUE))  #  item_drug + removed on 14 Nov 2022 - could be reverted
  
  # Confidence intervals
  model_prog_ci[[i]]<- tbl_regression(model_prog_summaries[[i]], exponentiate = TRUE, conf.int = TRUE, pvalue_fun = ~style_sigfig(., digits = 4))
  
  # Data count
  prog_regs_stats[j,1] <- prog
  prog_regs_stats[j,2] <- dim(df_for_fac_item_re_sorted_prog)[1]
  prog_regs_stats[j,3] <- length(unique(df_for_fac_item_re_sorted_prog$fac_code))
  prog_regs_stats[j,4] <- length(unique(df_for_fac_item_re_sorted_prog$item))
  
  i = i+1
  j = j + 1
}

# Store regression results
program_tbl_merge <-
  tbl_merge(
    tbls = list(model_prog_ci[[1]],model_prog_ci[[2]],model_prog_ci[[3]],
                model_prog_ci[[4]],model_prog_ci[[5]],model_prog_ci[[6]],
                model_prog_ci[[7]],model_prog_ci[[8]],model_prog_ci[[9]],
                model_prog_ci[[10]],model_prog_ci[[11]]),
    tab_spanner = c("**General**", "**HIV**", "**Malaria**", "**Tuberculosis**",
                    "**Obstetric and newborn care**", "**Contraception**", "**Child health**",
                    "**EPI**", "**ALRI**","**NCDs**", "**Surgery**")
  )  %>%    # build gtsummary table
  as_gt() 

# 4.1.2 Regression by program (consumable fixed effects)
#-------------------------------------------------------
i = 1
model_prog_summaries_itemfe <- list()      # Create empty list to store model outputs
model_prog_ci_itemfe <- list() # Create empty list to store confidence intervals
prog_regs_stats_itemfe <- matrix(NaN, nrow = length(programs_for_reg) , ncol = 4) # empty list for data counts

j = 1

for (prog in programs_for_reg){
  print(paste("running program", prog))
  df_for_fac_item_re_sorted_prog <- df_for_fac_item_re_sorted[which(df_for_fac_item_re_sorted$program == prog),]
  # Regression
  model_prog_summaries_itemfe[[i]] <-glmer(available ~ fac_type + fac_owner + fac_urban + functional_computer +
                                             functional_emergency_vehicle + service_diagnostic +
                                             incharge_drug_orders +
                                             dist_todh_cat + dist_torms_cat +
                                             drug_order_fulfilment_freq_last_3mts_cat + rms +
                                             functional_refrigerator +  functional_landline + fuctional_mobile +
                                             functional_toilet +  functional_handwashing_facility +
                                             water_source_main +
                                             outpatient_only + 
                                             service_hiv + service_othersti + 
                                             service_malaria + service_tb + 
                                             service_fp + service_imci +  
                                             source_drugs_ngo +  source_drugs_pvt + 
                                             drug_transport_self + item +
                                             (1|fac_code), 
                                           family = binomial(logit),
                                           data = df_for_fac_item_re_sorted_prog, 
                                           control = glmerControl(optimizer = "bobyqa",
                                                                  optCtrl=list(maxfun=1e5),
                                                                  calc.derivs = TRUE))  
  
  # Confidence intervals
  model_prog_ci_itemfe[[i]]<- tbl_regression(model_prog_summaries_itemfe[[i]], exponentiate = TRUE, conf.int = TRUE, pvalue_fun = ~style_sigfig(., digits = 4))
  
  # Data count
  prog_regs_stats_itemfe[j,1] <- prog
  prog_regs_stats_itemfe[j,2] <- dim(df_for_fac_item_re_sorted_prog)[1]
  prog_regs_stats_itemfe[j,3] <- length(unique(df_for_fac_item_re_sorted_prog$fac_code))
  prog_regs_stats_itemfe[j,4] <- length(unique(df_for_fac_item_re_sorted_prog$item))
  
  i = i+1
  j = j + 1
}

# Store regression results
program_tbl_merge_itemfe <-
  tbl_merge(
    tbls = list(model_prog_ci_itemfe[[1]],model_prog_ci_itemfe[[2]],model_prog_ci_itemfe[[3]],
                model_prog_ci_itemfe[[4]],model_prog_ci_itemfe[[5]],model_prog_ci_itemfe[[6]],
                model_prog_ci_itemfe[[7]],model_prog_ci_itemfe[[8]],model_prog_ci_itemfe[[9]],
                model_prog_ci_itemfe[[10]],model_prog_ci_itemfe[[11]]),
    tab_spanner = c("**General**", "**HIV**", "**Malaria**", "**Tuberculosis**",
                    "**Obstetric and newborn care**", "**Contraception**", "**Child health**",
                    "**EPI**", "**ALRI**","**NCDs**", "**Surgery**")
  )  %>%    # build gtsummary table
  as_gt() 


# 4.2 Regression by fac_type
#----------------------------
group_by_fac_type <- 
  df_orig %>%                             #Applying group_by &summarise
  group_by(fac_type) %>%
  summarise(unique_facs = n_distinct(fac_code))

fac_types_for_reg <- c('Facility_level_1a', 'Facility_level_1b')
i = 1
model_level_summaries <- list()      # Create empty list to store model outputs
model_level_ci <- list() # Create empty list to store confidence intervals
fac_regs_stats <- matrix(NaN, nrow = length(unique(df_for_fac_item_re_sorted$fac_type)) , ncol = 4) # list to store data counts
j = 1

for (level in fac_types_for_reg){
  print(paste("running level", level))
  df_for_fac_item_re_sorted_level <- df_for_fac_item_re_sorted[which(df_for_fac_item_re_sorted$fac_type == level),]
  
  # Regression
  model_level_summaries[[i]] <-glmer(available ~ fac_owner + fac_urban + functional_computer +
                                       functional_emergency_vehicle + service_diagnostic +
                                       incharge_drug_orders +
                                       dist_todh_cat + dist_torms_cat +
                                       drug_order_fulfilment_freq_last_3mts_cat + rms +
                                       functional_refrigerator +  functional_landline + fuctional_mobile +
                                       functional_toilet +  functional_handwashing_facility +
                                       water_source_main +
                                       outpatient_only + 
                                       service_hiv + service_othersti + 
                                       service_malaria + service_tb + 
                                       service_fp + service_imci +  
                                       source_drugs_ngo +  source_drugs_pvt + 
                                       drug_transport_self + item_drug + eml_priority_v +
                                       (1|fac_code) + (1|program/item), 
                                     family = binomial(logit),
                                     data = df_for_fac_item_re_sorted_level, 
                                     control = glmerControl(optimizer = "bobyqa",
                                                            optCtrl=list(maxfun=1e5),
                                                            calc.derivs = TRUE))  
  # Confidence intervals
  model_level_ci[[i]]<- tbl_regression(model_level_summaries[[i]], exponentiate = TRUE, conf.int = TRUE, pvalue_fun = ~style_sigfig(., digits = 4))
  
  # Data count
  fac_regs_stats[j,1] <- level
  fac_regs_stats[j,2] <- dim(df_for_fac_item_re_sorted_level)[1]
  fac_regs_stats[j,3] <- length(unique(df_for_fac_item_re_sorted_level$fac_code))
  fac_regs_stats[j,4] <- length(unique(df_for_fac_item_re_sorted_level$item))
  
  i = i+1
  j = j + 1
}

level_tbl_merge <-
  tbl_merge(
    tbls = list(model_level_ci[[1]],model_level_ci[[2]]),
    tab_spanner = c("**Primary level**", "**Secondary level**")
  )  %>%    # build gtsummary table
  as_gt() 

# 4.3 Regression by fac_owner
#----------------------------
group_by_fac_owner <- 
  df_orig %>%                             #Applying group_by &summarise
  group_by(fac_owner) %>%
  summarise(unique_facs = n_distinct(fac_code))

fac_owners_for_reg <- c('Government', 'CHAM', 'NGO', 'Private for profit')
i = 1
model_owner_summaries <- list()      # Create empty list to store model outputs
model_owner_ci <- list() # Create empty list to store confidence intervals
owner_regs_stats <- matrix(NaN, nrow = length(unique(df_for_fac_item_re_sorted$fac_owner)) , ncol = 4) # list to store data counts
j = 1

for (owner in fac_owners_for_reg){
  print(paste("running owner", owner))
  df_for_fac_item_re_sorted_owner <- df_for_fac_item_re_sorted[which(df_for_fac_item_re_sorted$fac_owner == owner),]
  
  # Regression
  model_owner_summaries[[i]] <-glmer(available ~ fac_type + fac_urban + functional_computer +
                                       functional_emergency_vehicle + service_diagnostic +
                                       incharge_drug_orders +
                                       dist_todh_cat + dist_torms_cat +
                                       drug_order_fulfilment_freq_last_3mts_cat + rms +
                                       functional_refrigerator +  functional_landline + fuctional_mobile +
                                       functional_toilet +  functional_handwashing_facility +
                                       water_source_main +
                                       outpatient_only + 
                                       service_hiv + service_othersti + 
                                       service_malaria + service_tb + 
                                       service_fp + service_imci +  
                                       source_drugs_ngo +  source_drugs_pvt + 
                                       drug_transport_self + item_drug + eml_priority_v +
                                       (1|fac_code) + (1|program/item), 
                                     family = binomial(logit),
                                     data = df_for_fac_item_re_sorted_owner, 
                                     control = glmerControl(optimizer = "bobyqa",
                                                            optCtrl=list(maxfun=1e5),
                                                            calc.derivs = TRUE))  
  # Confidence intervals
  model_owner_ci[[i]]<- tbl_regression(model_owner_summaries[[i]], exponentiate = TRUE, conf.int = TRUE, pvalue_fun = ~style_sigfig(., digits = 4))
  
  # Data count
  owner_regs_stats[j,1] <- owner
  owner_regs_stats[j,2] <- dim(df_for_fac_item_re_sorted_owner)[1]
  owner_regs_stats[j,3] <- length(unique(df_for_fac_item_re_sorted_owner$fac_code))
  owner_regs_stats[j,4] <- length(unique(df_for_fac_item_re_sorted_owner$item))
  
  i = i+1
  j = j + 1
}

owner_tbl_merge <-
  tbl_merge(
    tbls = list(model_owner_ci[[1]],model_owner_ci[[2]],model_owner_ci[[3]],model_owner_ci[[4]]),
    tab_spanner = c("**Government**", "**CHAM**", "**NGO**", "**Private for profit**")
  )  %>%    # build gtsummary table
  as_gt() 

# 4.4 Regression by consumable type
#-----------------------------------
group_by_item_type <- 
  df_orig %>%                             #Applying group_by &summarise
  group_by(item_type) %>%
  summarise(unique_facs = n_distinct(fac_code))

item_types_for_reg <- c('drug', 'consumable')
i = 1
model_item_type_summaries <- list()      # Create empty list to store model outputs
model_item_type_ci <- list() # Create empty list to store confidence intervals
item_type_regs_stats <- matrix(NaN, nrow = length(unique(df_for_fac_item_re_sorted$item_type)) , ncol = 4) # list to store data counts
j = 1

for (type in item_types_for_reg){
  print(paste("running item_type", type))
  df_for_fac_item_re_sorted_type <- df_for_fac_item_re_sorted[which(df_for_fac_item_re_sorted$item_type == type),]
  
  # Regression
  model_item_type_summaries[[i]] <-glmer(available ~ fac_type + fac_owner + fac_urban + functional_computer +
                                           functional_emergency_vehicle + service_diagnostic +
                                           incharge_drug_orders +
                                           dist_todh_cat + dist_torms_cat +
                                           drug_order_fulfilment_freq_last_3mts_cat + rms +
                                           functional_refrigerator +  functional_landline + fuctional_mobile +
                                           functional_toilet +  functional_handwashing_facility +
                                           water_source_main +
                                           outpatient_only + 
                                           service_hiv + service_othersti + 
                                           service_malaria + service_tb + 
                                           service_fp + service_imci +  
                                           source_drugs_ngo +  source_drugs_pvt + 
                                           drug_transport_self +  eml_priority_v +
                                           (1|fac_code) + (1|program/item), 
                                         family = binomial(logit),
                                         data = df_for_fac_item_re_sorted_type, 
                                         control = glmerControl(optimizer = "bobyqa",
                                                                optCtrl=list(maxfun=1e5),
                                                                calc.derivs = TRUE))  
  # Confidence intervals
  model_item_type_ci[[i]]<- tbl_regression(model_item_type_summaries[[i]], exponentiate = TRUE, conf.int = TRUE, pvalue_fun = ~style_sigfig(., digits = 4))
  
  # Data count
  item_type_regs_stats[j,1] <- type
  item_type_regs_stats[j,2] <- dim(df_for_fac_item_re_sorted_type)[1]
  item_type_regs_stats[j,3] <- length(unique(df_for_fac_item_re_sorted_type$fac_code))
  item_type_regs_stats[j,4] <- length(unique(df_for_fac_item_re_sorted_type$item))
  
  i = i+1
  j = j + 1
}

item_type_tbl_merge <-
  tbl_merge(
    tbls = list(model_item_type_ci[[1]],model_item_type_ci[[2]]),
    tab_spanner = c("**Drugs**", "**Other consumables**")
  )  %>%    # build gtsummary table
  as_gt() 

# 4.5 Regression by essential medicine list category
#---------------------------------------------------
group_by_eml_priority <- 
  df %>%                             #Applying group_by &summarise
  group_by(eml_priority_v) %>%
  summarise(unique_items = n_distinct(item))

item_priority_for_reg <- c(1,0)
i = 1
model_item_priority_summaries <- list()      # Create empty list to store model outputs
model_item_priority_ci <- list() # Create empty list to store confidence intervals
item_priority_regs_stats <- matrix(NaN, nrow = length(unique(df_for_fac_item_re_sorted$eml_priority_v)) , ncol = 4) # list to store data counts
j = 1

for (category in item_priority_for_reg){
  print(paste("running eml_priority_v", category))
  df_for_fac_item_re_sorted_category <- df_for_fac_item_re_sorted[which(df_for_fac_item_re_sorted$eml_priority_v == category),]
  
  # Regression
  model_item_priority_summaries[[i]] <-glmer(available ~ fac_type + fac_owner + fac_urban + functional_computer +
                                               functional_emergency_vehicle + service_diagnostic +
                                               incharge_drug_orders +
                                               dist_todh_cat + dist_torms_cat +
                                               drug_order_fulfilment_freq_last_3mts_cat + rms +
                                               functional_refrigerator +  functional_landline + fuctional_mobile +
                                               functional_toilet +  functional_handwashing_facility +
                                               water_source_main +
                                               outpatient_only + 
                                               service_hiv + service_othersti + 
                                               service_malaria + service_tb + 
                                               service_fp + service_imci +  
                                               source_drugs_ngo +  source_drugs_pvt + 
                                               drug_transport_self + item_drug +
                                               (1|fac_code) + (1|program/item), 
                                             family = binomial(logit),
                                             data = df_for_fac_item_re_sorted_category, 
                                             control = glmerControl(optimizer = "bobyqa",
                                                                    optCtrl=list(maxfun=1e5),
                                                                    calc.derivs = TRUE))  
  # Confidence intervals
  model_item_priority_ci[[i]]<- tbl_regression(model_item_priority_summaries[[i]], exponentiate = TRUE, conf.int = TRUE, pvalue_fun = ~style_sigfig(., digits = 4))
  
  # Data count
  item_priority_regs_stats[j,1] <- type
  item_priority_regs_stats[j,2] <- dim(df_for_fac_item_re_sorted_category)[1]
  item_priority_regs_stats[j,3] <- length(unique(df_for_fac_item_re_sorted_category$fac_code))
  item_priority_regs_stats[j,4] <- length(unique(df_for_fac_item_re_sorted_category$item))
  
  i = i+1
  j = j + 1
}

item_priority_tbl_merge <-
  tbl_merge(
    tbls = list(model_item_priority_ci[[1]],model_item_priority_ci[[2]]),
    tab_spanner = c("**Vital consumables**", "**Other consumables**")
  )  %>%    # build gtsummary table
  as_gt()

# 4.6 Regression by national consumable availability
#---------------------------------------------------
# Make a list of items with less than 10% availability across all facilities
df_by_item <- df %>% 
  group_by(item) %>% 
  summarise(available_average = mean(available, na.rm = TRUE)) %>%
  arrange(available_average)

items_with_low_national_availability <- subset(df_by_item, df_by_item$available_average < 0.1)['item']
items_with_low_national_availability <- as.list(items_with_low_national_availability)
items_with_high_national_availability <- subset(df_by_item, df_by_item$available_average >= 0.1)['item']
items_with_high_national_availability <- as.list(items_with_high_national_availability)

item_availability_groups_for_reg <- c(items_with_low_national_availability, items_with_high_national_availability)

i = 1
model_item_availability_group_summaries <- list()      # Create empty list to store model outputs
model_item_availability_group_ci <- list() # Create empty list to store confidence intervals
item_availability_group_regs_stats <- matrix(NaN, nrow = 2 , ncol = 4) # list to store data counts
j = 1

for (group in item_availability_groups_for_reg){
  print(paste("running availability group"))
  df_for_fac_item_re_sorted_group <- df_for_fac_item_re_sorted %>% filter(!(item %in% group))
  
  # Regression
  model_item_availability_group_summaries[[i]] <-glmer(available ~ fac_type + fac_owner + fac_urban + functional_computer +
                                                         functional_emergency_vehicle + service_diagnostic +
                                                         incharge_drug_orders +
                                                         dist_todh_cat + dist_torms_cat +
                                                         drug_order_fulfilment_freq_last_3mts_cat + rms +
                                                         functional_refrigerator +  functional_landline + fuctional_mobile +
                                                         functional_toilet +  functional_handwashing_facility +
                                                         water_source_main +
                                                         outpatient_only + 
                                                         service_hiv + service_othersti + 
                                                         service_malaria + service_tb + 
                                                         service_fp + service_imci +  
                                                         source_drugs_ngo +  source_drugs_pvt + 
                                                         drug_transport_self + item_drug + eml_priority_v + 
                                                         (1|fac_code) + (1|program/item), 
                                                       family = binomial(logit),
                                                       data = df_for_fac_item_re_sorted_group, 
                                                       control = glmerControl(optimizer = "bobyqa",
                                                                              optCtrl=list(maxfun=1e5),
                                                                              calc.derivs = TRUE))  
  # Confidence intervals
  model_item_availability_group_ci[[i]]<- tbl_regression(model_item_availability_group_summaries[[i]], exponentiate = TRUE, conf.int = TRUE, pvalue_fun = ~style_sigfig(., digits = 4))
  
  # Data count
  item_availability_group_regs_stats[j,1] <- group
  item_availability_group_regs_stats[j,2] <- dim(df_for_fac_item_re_sorted_group)[1]
  item_availability_group_regs_stats[j,3] <- length(unique(df_for_fac_item_re_sorted_group$fac_code))
  item_availability_group_regs_stats[j,4] <- length(unique(df_for_fac_item_re_sorted_group$item))
  
  i = i+1
  j = j + 1
}

item_availability_group_tbl_merge <-
  tbl_merge(
    tbls = list(model_item_availability_group_ci[[1]],model_item_availability_group_ci[[2]]),
    tab_spanner = c("**Availability >= 10%**", "**Availability < 10%**")
  )  %>%    # build gtsummary table
  as_gt()

# Save subgroup regression results
###########################
save(model_owner_summaries, file = paste0(path_to_outputs, "regression_results/sub-group/model_owner.rdta"))
save(model_item_type_summaries, file = paste0(path_to_outputs, "regression_results/sub-group/model_item_type.rdta"))
save(model_item_priority_summaries, file = paste0(path_to_outputs, "regression_results/sub-group/model_item_priority.rdta"))
save(model_item_availability_group_summaries, file = paste0(path_to_outputs, "regression_results/sub-group/model_item_availability_group.rdta"))
save(model_level_summaries, file = paste0(path_to_outputs, "regression_results/sub-group/model_level.rdta"))
save(model_prog_summaries, file = paste0(path_to_outputs, "regression_results/sub-group/model_program.rdta"))
save(model_prog_summaries_itemfe, file = paste0(path_to_outputs, "regression_results/sub-group/model_program_itemfe.rdta"))

save(model_owner_ci, file = paste0(path_to_outputs, "regression_results/sub-group/model_owner_ci.rdta"))
save(model_item_type_ci, file = paste0(path_to_outputs, "regression_results/sub-group/model_item_type_ci.rdta"))
save(model_item_priority_ci, file = paste0(path_to_outputs, "regression_results/sub-group/model_item_priority_ci.rdta"))
save(model_item_availability_group_ci, file = paste0(path_to_outputs, "regression_results/sub-group/model_item_availability_group_ci.rdta"))
save(model_level_ci, file = paste0(path_to_outputs, "regression_results/sub-group/model_level_ci.rdta"))
save(model_prog_ci, file = paste0(path_to_outputs, "regression_results/sub-group/model_program_ci.rdta"))
save(model_prog_ci_itemfe, file = paste0(path_to_outputs, "regression_results/sub-group/model_program_itemfe_ci.rdta"))
