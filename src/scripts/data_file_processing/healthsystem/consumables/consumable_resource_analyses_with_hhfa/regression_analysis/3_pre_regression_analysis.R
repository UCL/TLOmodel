# This script performs the analysis to choose the regression model
# Since this script can take several hours to run, the output of this script has been copied directly into 
# 4_regression_analysis.R rather that running this each time. 

# Run setup files
source(paste0(path_to_scripts, "0_packages_and_functions.R"))
source(paste0(path_to_scripts, "1_data_setup.R"))
source(paste0(path_to_scripts, "2_feature_manipulation.R"))

# 1. Choose variables to be included in the backward elimination analysis #
###########################################################################

# Create varlist to record the outcome of the stepwise procedure
regressor_list_initial <- fac_exp_vars[!(fac_exp_vars %in% stockout_factors_from_lit)]

# 1.1 Remove highly correlated variables
#----------------------------------------
#----------------------------------------
# 1.1.1 Numeric variables
#---------------------
correlation_numeric_vars <- cor(fac_reg_df[fac_vars_numeric])
#print(correlation_numeric_vars)
fac_vars_numeric_corrplot <- ggcorrplot(correlation_numeric_vars, lab_size = 1.5, p.mat = NULL, 
                                        insig = c("pch", "blank"), pch = 1, pch.col = "black", pch.cex =1,
                                        tl.cex =5.5, lab = TRUE)
ggsave(plot = fac_vars_numeric_corrplot, filename = paste0(path_to_outputs, "figures/fac_vars_numeric_corrplot.png"))

highly_correlated_numeric_vars <- c('drivetime_todh', 'drivetime_torms', 
                                    'inpatient_visit_count', 'inpatient_days_count') #'functional_computer_no', 'functional_car_no', 'functional_motor_cycle_no'
fac_exp_vars <- fac_exp_vars[!(fac_exp_vars %in% highly_correlated_numeric_vars)]
fac_vars_numeric <- fac_vars_numeric[!(fac_vars_numeric %in% highly_correlated_numeric_vars)]

# Plot after highly correlated variables are dropped
correlation_numeric_vars <- cor(fac_reg_df[fac_vars_numeric])
fac_vars_numeric_corrplot_post <- ggcorrplot(correlation_numeric_vars, lab_size = 1.8, p.mat = NULL, 
                                             insig = c("pch", "blank"), pch = 1, pch.col = "black", pch.cex =1,
                                             tl.cex =5.5, lab = TRUE)
ggsave(plot = fac_vars_numeric_corrplot_post, filename = paste0(path_to_outputs, "figures/fac_vars_numeric_corrplot_post.png"))

# 1.1.2 Binary variables
#---------------------
correlation_binary_vars <- cor(fac_reg_df[fac_vars_binary])
#print(correlation_binary_vars)
fac_vars_binary_corrplot <- ggcorrplot(correlation_binary_vars, lab_size = 0.8, p.mat = NULL, 
                                       insig = c("pch", "blank"), pch = 1, pch.col = "black", pch.cex =1,
                                       tl.cex =3.5, lab = TRUE)
ggsave(plot = fac_vars_binary_corrplot, filename = paste0(path_to_outputs, "figures/fac_vars_binary_corrplot.png"))

highly_correlated_binary_vars <- c('service_delivery', 'service_pmtct', 'service_pnc',
                                   'functional_ambulance', 'vaccine_storage', 'service_anc',
                                   'service_blood_transfusion')
fac_exp_vars <- fac_exp_vars[!(fac_exp_vars %in% highly_correlated_binary_vars)]
fac_vars_binary <- fac_vars_binary[!(fac_vars_binary %in% highly_correlated_binary_vars)]

# Plot after highly correlated variables are dropped
correlation_binary_vars <- cor(fac_reg_df[fac_vars_binary])
fac_vars_binary_corrplot <- ggcorrplot(correlation_binary_vars, lab_size = 1, p.mat = NULL, 
                                       insig = c("pch", "blank"), pch = 1, pch.col = "black", pch.cex =1,
                                       tl.cex =3.5, lab = TRUE)
ggsave(plot = fac_vars_binary_corrplot, filename = paste0(path_to_outputs, "figures/fac_vars_binary_corrplot_post.png"))

# Create varlist to record the outcome of the stepwise procedure
regressor_list_postcorrelationanalysis <- fac_exp_vars

# 1.2 Remove variables for other reasons
#----------------------------------------
#----------------------------------------
other_vars_to_be_dropped <- c('travel_time_to_district_hq', # accounted for by googlemaps distance calculation
                              'functional_bike_ambulance_no', # only 3% facilities have these
                              'functional_computer_no', # not enough variation
                              'functional_motor_cycle_no', # not enough variation
                              'functional_ambulance_no', # not enough variation
                              'functional_bicycle_no') # not enough variation

fac_exp_vars <- fac_exp_vars[!(fac_exp_vars %in% other_vars_to_be_dropped)]
fac_vars_numeric <- fac_vars_numeric[!(fac_vars_numeric %in% other_vars_to_be_dropped)]

# Create varlist to record the outcome of the stepwise procedure
regressor_list_postotherreasons <- fac_exp_vars

# 2. Subset dataframes with the chosen variables #
##################################################
# Define facilty-level regression dataframe
fac_reg_df <- fac_reg_df[fac_exp_vars]
fac_reg_df <- na.omit(fac_reg_df) 

# Edit long data based on the above removal of explanatory variables
# Define explanatory variables list
exp_vars <- list(fac_exp_vars, 'available', 'item', 'rms', item_vars_binary) # 'drug_class_rx_list', 'fac_code', 'program', 'mode_administration'
exp_vars <- unlist(exp_vars, recursive=FALSE)


# Define detailed regression dataframe
reg_df <- df[exp_vars]
reg_df <- na.omit(reg_df) 

# 3. Feature selection through bi-directional selection #
#########################################################
# 2a.Apply logistic model to item-level data
logistic_model_allvars <- glm(available ~ .,
                              data = reg_df, family = binomial("logit"))
logistic_model_litvars <- glm(available ~  fac_urban + fac_type +
                                functional_computer + incharge_drug_orders +
                                functional_emergency_vehicle + 
                                dist_todh + dist_torms + drug_order_fulfilment_freq_last_3mts + 
                                service_diagnostic + item_drug + eml_priority_v,
                              data = reg_df, family = binomial("logit")) # quasibinomial does not work because no AIC for backward elimination
# %% drug_resupply_calculation_system was dropped because this seems to be incorrectly recorded %%
# %% district, fac_type were dropped because they lead to convergence issues when accounting for random effects %%

# Run bidirectional stepwise to determine best parsimonious model (based on BIC)
stepwise_logistic_model <- stepAIC(logistic_model_litvars, k= log(nrow(reg_df)), direction = "both",
                                   scope = list(lower = ~  fac_urban + fac_type +
                                                  functional_computer + incharge_drug_orders +
                                                  functional_emergency_vehicle + 
                                                  dist_todh + dist_torms + drug_order_fulfilment_freq_last_3mts + 
                                                  service_diagnostic + eml_priority_v,
                                                upper = logistic_model_allvars))

model_base <- summary(stepwise_logistic_model)
stepwise_logistic_model$call

# 4. Model diagnostic #
#######################
png(file = paste0(path_to_outputs, "figures/logistic_model_diagnostic.png"))
autoplot(stepwise_logistic_model)
dev.off()

set.seed(81)
sample_split <- sample.split(Y = reg_df$available, SplitRatio = 0.7) # change this back to 0.7
train_set <- subset(x = reg_df, sample_split == TRUE)
test_set <- subset(x = reg_df, sample_split == FALSE)

independent_varlist_logistic <- unlist(c('available', attr(stepwise_logistic_model$terms , "term.labels")))
train_set_reg = train_set[,dput(as.character(independent_varlist_logistic))] 
model_base_train <- glm(available ~ .,
                        data = train_set_reg,
                        family = "binomial")

# Generating predictions to test model accuracy
test_set_reg <- test_set[,dput(as.character(independent_varlist_logistic))] 
probs_model <- predict(model_base_train, newdata = test_set_reg, type = "response")
pred_model <- ifelse(probs_model > 0.5, 1, 0)
cm <- confusionMatrix(factor(pred_model), factor(test_set_reg$available), positive = as.character(1))

chosen_model <- formula(model_base_train)
chosen_varlist <- independent_varlist_logistic
print(chosen_model)
print(chosen_varlist)
print(paste("accuracy = ", cm$overall[1]*100, "%"))

# Export relevant regressor lists from various model selection steps
sheet_lst <- list('regressor_list_initial' = regressor_list_initial, 
                  'regressor_list_postcorr' = regressor_list_postcorrelationanalysis,
                  'regressor_list_postother' = regressor_list_postotherreasons, 
                  'chosen_varlist'= chosen_varlist)

write.xlsx(sheet_lst, file = paste0(path_to_outputs, "tables/regressor_list.xlsx"))
