# This script loads the outputs of the final regression model
# and generates predictions for consumable availability scenarios

###########################################################
# 1. Set up
###########################################################
# 1.1 Run setup script
#---------------------------------
path_to_scripts = "C:/Users/sm2511/PycharmProjects/TLOmodel/src/scripts/data_file_processing/healthsystem/consumables/consumable_resource_analyses_with_hhfa/regression_analysis/"
source(paste0(path_to_scripts, "3b_data_setup_for_regression.R"))

###########################################################
# 2. Load model outputs
###########################################################
load(paste0(path_to_outputs, "regression_results/model_fac_item_re.rdta"))

###########################################################
# 3. Run predictions for policy evaluation
###########################################################
# Replace program string which shorter text for plotting
df_regress <- df_for_fac_re_sorted
df_regress$program_plot <- df_regress$program
rep_str = c('acute lower respiratory infections'='alri','obstetric and newborn care'='obs&newb',
            'other - infection prevention'='infection_prev', 'child health' = 'child')
df_regress$program_plot <- str_replace_all(df_regress$program_plot, rep_str)
df_regress <- df_regress[unlist(c(chosen_varlist_for_fac_re, 'program_plot'))]

# Drop one item which seems to be causing a but in the prediction
drop_items = c("Dipsticks for urine ketone bodies for rapid diagnostic ")
df_regress <- df_regress %>%
  filter(!(item %in% drop_items) # Second line ARVs
  )

# Generate a column with regression based predictions
df_regress <- df_regress %>% 
  mutate(available_prob_predicted = predict(model_fac_item_re,newdata=df_regress, type = "response"))
df_regress$fac_type_original = df_regress$fac_type # because predictions are run on fac_type and then data collapsed on the basis of this variable

# 3.1 Run predictions on individual characteristics
##########################################################

ideal_features_for_cons_availability <- list('fac_type' = 'Facility_level_1b',
                                         'fac_owner' = 'CHAM', 
                                         'functional_computer' = 1, 
                                         'incharge_drug_orders' = 'pharmacist or pharmacy technician',
                                         'functional_emergency_vehicle' = 1, 
                                         'service_diagnostic' = 1,
                                         'dist_todh_cat' = '0-10 kms', 
                                         'dist_torms_cat' = '0-10 kms', 
                                         'drug_order_fulfilment_freq_last_3mts_cat' = '3')
#, #urban

i = 1
for (feature in names(ideal_features_for_cons_availability)){
  print(paste0("Running predictions for ", feature, " = ", ideal_features_for_cons_availability[feature]))
  df <- df_regress
  df[feature] = ideal_features_for_cons_availability[feature]
  
  new_prediction = predict(model_fac_item_re,newdata=df, type = "response") # Predict availability when all facilities have pharmacist managing drug orders
  df$available_prob_new_prediction <- new_prediction 
  
  # Check that the prediction improves availability from the base case
  stopifnot(mean(df$available_prob_predicted, na.rm = TRUE) < 
              mean(df$available_prob_new_prediction, na.rm = TRUE))
  
  # Calculate proportional change which can be applied to LMIS data
  df$availability_change_prop = df$available_prob_new_prediction/df$available_prob_predicted
  
  # Collapse data
  summary_pred <- df %>% 
    group_by(district, fac_type_original, program_plot, item) %>%
    summarise_at(vars(availability_change_prop), list(mean))
  
  pred_col_number = length(colnames(summary_pred))
  colnames(summary_pred)[pred_col_number] <- paste0('change_proportion_', names(ideal_features_for_cons_availability)[i])
  
  if (i == 1){
    all_predictions_df = summary_pred
  } else{
    all_predictions_df = merge(all_predictions_df, summary_pred, by = c('district', 'fac_type_original', 'program_plot', 'item'))
  }
  i = i + 1
}

# 3.2 Run predictions on all characteristics being changed simulataneously
# to arrive at a theoretical maximum
#############################################################################
df <- df_regress
for (feature in names(ideal_features_for_cons_availability)){
  print(paste0("Editing ", feature, " = ", ideal_features_for_cons_availability[feature]))
  df[feature] = ideal_features_for_cons_availability[feature]
}

new_prediction = predict(model_fac_item_re,newdata=df, type = "response") # Predict availability when all facilities have pharmacist managing drug orders
df$available_prob_new_prediction <- new_prediction 

# Check that the prediction improves availability from the base case
stopifnot(mean(df$available_prob_predicted, na.rm = TRUE) < 
            mean(df$available_prob_new_prediction, na.rm = TRUE))

# Calculate proportional change which can be applied to LMIS data
df$change_proportion_all_features = df$available_prob_new_prediction/df$available_prob_predicted

# Collapse data
summary_pred <- df %>% 
  group_by(district, fac_type_original, program_plot, item) %>%
  summarise_at(vars(change_proportion_all_features), list(mean))

# Extract .csv for model simulation
all_predictions_df = merge(all_predictions_df, summary_pred, by = c('district', 'fac_type_original', 'program_plot', 'item'))
all_predictions_df = all_predictions_df %>% rename(fac_type = fac_type_original)

#i = 1
#for (feature in names(ideal_features_for_cons_availability)){
#  print(paste0("Testing ", feature, " = ", ideal_features_for_cons_availability[feature]))
#  var = paste0('change_proportion_', names(ideal_features_for_cons_availability)[i])
#  stopifnot(all_predictions_df['change_proportion_all_features'] >= all_predictions_df[var])
#  i = i + 1
#}

for (i in length(colnames(summary_pred))+1:length(colnames(all_predictions_df))-1){
  print(paste0("Testing ", colnames(all_predictions_df)[i]))
  stopifnot(all_predictions_df['change_proportion_all_features'] >= all_predictions_df[,i])
}
stopifnot(all_predictions_df['change_proportion_all_features'] >= all_predictions_df[,13])

subset_df <- all_predictions_df[all_predictions_df['change_proportion_all_features'] < all_predictions_df[,13], ]
# this is true in only one instance so we can ignore it?

write.csv(all_predictions_df,paste0(path_to_outputs, "predictions/predicted_consumable_availability_regression_scenarios.csv"), row.names = TRUE)


#test = merge(all_predictions_df, summary_pred_computer, by = c('district', 'fac_type', 'program_plot', 'item'))
#stopifnot(test$change_proportion_functional_computer == test$availability_change_prop)

# Make sure the prediction on fac_type makes sense (no change should be observed for level 1b)
stopifnot(mean(all_predictions_df[which(all_predictions_df$fac_type == 'Facility_level_1b'),]$change_proportion_fac_type) == 1)

# Another prediction could be increasing the availability by each fac_type and district 
# to what the average availability is for programs served by parallel supply chains

# 


#################################
# 4. Plot predicted availability
###############################
# 4.1 Computer
###############################
# Plot original values
p_original <- ggplot(summary_pred_computer, aes(item, district,  fill= available)) + 
  geom_tile() +
  facet_wrap(~fac_type) +
  scale_fill_viridis(discrete = FALSE, direction = -1) +
  theme(axis.text.x = element_text(angle = 45 , vjust = 0.7, size = 1),
        axis.title.x = element_text(size = 8), axis.title.y = element_text(size = 8), 
        axis.text.y = element_text(size = 4),
        legend.position = 'none',
        plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5)) +
  labs(title = "Probability of consumable availability - actual", 
       subtitle =paste0("Global average = ", round(mean(summary_pred_computer$available) *100, 2),"%")) +
  xlab("consumable")

# Plot predicted values
p_predict <- ggplot(summary_pred_computer, aes(item, district,  fill= available_predict)) + 
  geom_tile() +
  facet_wrap(~fac_type) +
  scale_fill_viridis(discrete = FALSE, direction = -1) +
  theme(axis.text.x = element_text(angle = 45 , vjust = 0.7, size = 1),
        axis.title.x = element_text(size = 8), axis.title.y = element_text(size = 8), 
        axis.text.y = element_text(size = 4),
        legend.position = 'none',
        plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5))  +
  labs(title = "Probability of consumable availability - predicted \n (all facilities have computers)", 
       subtitle =paste0("Global average = ", round(mean(summary_pred_computer$available_predict) *100, 2),"%")) +
  xlab("consumable")



figure <- ggpubr::ggarrange(p_original, p_predict, # list of plots
                  labels = "AUTO", # labels
                  common.legend = T, # COMMON LEGEND
                  legend = "bottom", # legend position
                  align = "hv", # Align them both, horizontal and vertical
                  nrow = 2)  %>% # number of rows
  ggexport(filename = paste0(path_to_outputs, "predictions/figures/pred_computer.pdf"))

# 4.2 Person in-charge of drug orders
#####################################
# Plot original values
p_original <- ggplot(summary_pred_pharma, aes(item, district,  fill= available)) + 
  geom_tile() +
  facet_wrap(~fac_type) +
  scale_fill_viridis(discrete = FALSE, direction = -1) +
  theme(axis.text.x = element_text(angle = 45 , vjust = 0.7, size = 1),
        axis.title.x = element_blank(), axis.title.y = element_blank(), 
        axis.text.y = element_text(size = 4),
        legend.position = 'none',
        plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5)) +
  labs(title = "Probability of consumable availability - actual", 
       subtitle =paste0("Global average = ", round(mean(summary_pred_pharma$available) *100, 2),"%")) +
  xlab("consumable")

# Plot predicted values
p_predict <- ggplot(summary_pred_pharma, aes(item, district,  fill= available_predict)) + 
  geom_tile() +
  facet_wrap(~fac_type) +
  scale_fill_viridis(discrete = FALSE, direction = -1) +
  theme(axis.text.x = element_text(angle = 45 , vjust = 0.7, size = 1),
        axis.title.x = element_blank(), axis.title.y = element_blank(), 
        axis.text.y = element_text(size = 4),
        legend.position = 'none',
        plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5))  +
  labs(title = "Probability of consumable availability - predicted \n (all facilities have pharmacists for drug stock management)", 
       subtitle =paste0("Global average = ", round(mean(summary_pred_pharma$available_predict) *100, 2),"%")) +
  xlab("consumable")


ggpubr::ggarrange(p_original, p_predict, # list of plots
                  labels = "AUTO", # labels
                  common.legend = T, # COMMON LEGEND
                  legend = "bottom", # legend position
                  align = "hv", # Align them both, horizontal and vertical
                  nrow = 2)  %>% # number of rows
  ggexport(filename = paste0(path_to_outputs, "predictions/figures/pred_pharma.pdf"))

# 4.3 Increase availability of HIV, TB, Malaria drugs by 10%
############################################################
global_fund_programs = c('hiv', 'tb', 'malaria')
summary_pred_pharma %>% filter(program_plot %in% global_fund_programs) %>%
  group_by(program_plot) %>%
  summarise_at(vars(available_predict, available), list(mean))

df_pred_globalfund_target <- df_regress

summary_pred_globalfund <- df_pred_globalfund_target %>% 
  group_by(district, fac_type, program_plot, item) %>%
  summarise_at(vars(available), list(mean))

summary_pred_globalfund$available_predict <- summary_pred_globalfund$available
summary_pred_globalfund[which(summary_pred_globalfund$program_plot %in% global_fund_programs),]$available_predict <- 
  summary_pred_globalfund[which(summary_pred_globalfund$program_plot %in% global_fund_programs),]$available*1.1
summary_pred_globalfund[which(summary_pred_globalfund$available_predict > 1),]$available_predict = 1


stopifnot(mean(summary_pred_globalfund$available, na.rm = TRUE) < 
  mean(summary_pred_globalfund$available_predict, na.rm = TRUE))

p_predict <- ggplot(summary_pred_globalfund, aes(item, district,  fill= available_predict)) + 
  geom_tile() +
  facet_wrap(~fac_type) +
  scale_fill_viridis(discrete = FALSE, direction = -1) +
  theme(axis.text.x = element_text(angle = 45 , vjust = 0.7, size = 1),
        axis.title.x = element_blank(), axis.title.y = element_blank(), 
        axis.text.y = element_text(size = 4),
        legend.position = 'none',
        plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5))  +
  labs(title = "Probability of consumable availability - predicted \n (Availability of HIV, TB, Malaria drugs increased by 10%)", 
       subtitle =paste0("Global average = ", round(mean(summary_pred_globalfund$available_predict) *100, 2),"%")) +
  xlab("consumable")


ggpubr::ggarrange(p_original, p_predict, # list of plots
                  labels = "AUTO", # labels
                  common.legend = T, # COMMON LEGEND
                  legend = "bottom", # legend position
                  align = "hv", # Align them both, horizontal and vertical
                  nrow = 2)  %>% # number of rows
  ggexport(filename = paste0(path_to_outputs, "predictions/figures/pred_global_fund_target.pdf"))

