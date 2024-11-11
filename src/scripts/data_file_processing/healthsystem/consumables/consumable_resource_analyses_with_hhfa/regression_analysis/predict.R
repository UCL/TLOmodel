# This script loads the outputs of the final regression model
# and generates predictions for consumable availability scenarios

###########################################################
# 1. Load regression model output
###########################################################
load(paste0(path_to_outputs, "regression_results/model_fac_item_re.rdta"))
#TODO CHange above path

###########################################################
# 2. Run predictions for policy evaluation
###########################################################
# 2.1 Setup
#--------------------------------------------------------------------------------------------------------------------------------
# Replace program string which shorter text for plotting
df_regress <- df_for_re_reg_sorted
df_regress$program_plot <- df_regress$program
rep_str <- c('acute lower respiratory infections'='alri','obstetric and newborn care'='obs&newb',
            'other - infection prevention'='infection_prev', 'child health' = 'child')
df_regress$program_plot <- str_replace_all(df_regress$program_plot, rep_str)
df_regress <- df_regress[unlist(c(chosen_varlist_for_re_reg, 'program_plot'))]

# Drop one item which seems to be causing an issue in the prediction
drop_items <- c("Dipsticks for urine ketone bodies for rapid diagnostic ")
df_regress <- df_regress %>%
  filter(!(item %in% drop_items) # Second line ARVs
  )
# TODO check why this item needs to be dropped
# TODO check why there is an extra column item.1 in the dataframe

# Generate a column with regression based predictions
df_regress <- df_regress %>%
  mutate(available_prob_predicted = predict(model_fac_item_re,newdata=df_regress, type = "response"))
df_regress$fac_type_original <- df_regress$fac_type # because predictions are run on fac_type and then data collapsed on the basis of this variable

# 2.2 Run predictions by changing the 5 characteristics which have the largest association with consumable availability
#--------------------------------------------------------------------------------------------------------------------------------
top_5_features_for_cons_availability <- list('item_drug' = 0,
                                             'eml_priority_v' = 1,
                                             'incharge_drug_orders' = 'pharmacist or pharmacy technician',
                                             'fac_type' = 'Facility_level_1b',
                                             'fac_owner' = 'CHAM')
top_5_features_for_cons_availability_cumulative_names <- list('item_drug' = 'scen1',
                                             'eml_priority_v' = 1,
                                             'incharge_drug_orders' = 'pharmacist or pharmacy technician',
                                             'fac_type' = 'Facility_level_1b',
                                             'fac_owner' = 'CHAM') # this is for naming

i <- 0
j <- 0
for (feature in names(top_5_features_for_cons_availability)){
    if (i == 0){
        print(paste0("Running predictions for ", feature, " = ", top_5_features_for_cons_availability[feature]))
        df <- df_regress
        df[feature] <- top_5_features_for_cons_availability[feature] # cumulatively change facility and consumable features
        old_prediction <- df$available_prob_predicted # store prediction from the previous run
        i <- 1
    } else{
        print(paste0("Running predictions for ", feature, " = ", top_5_features_for_cons_availability[feature], " (cumulative, i.e. in addition to previous update)"))
        df[feature] <- top_5_features_for_cons_availability[feature] # cumulatively change facility and consumable features
        old_prediction <- df$available_prob_new_prediction # store prediction from the previous run
    }
    #unique_values_of_chosen_features <- df %>%  summarise_at(vars(eml_priority_v, item_drug, incharge_drug_orders, fac_type, fac_owner), ~list(unique(.)))

    # Run prediction with update set of features
    new_prediction <- predict(model_fac_item_re,newdata=df, type = "response") # Predict availability when all facilities have pharmacist managing drug orders
    df$available_prob_new_prediction <- new_prediction

    print(paste0("New mean probability of availability is ",mean(df$available_prob_new_prediction, na.rm = TRUE) * 100, "%"))

    # Check that the prediction improves availability from the previous case
    stopifnot(mean(old_prediction, na.rm = TRUE) <
                  mean(new_prediction, na.rm = TRUE))

    # Calculate proportional change in availability from baseline prediction which can be applied to LMIS data
      df$availability_change_prop <- df$available_prob_new_prediction/df$available_prob_predicted

      if (j == 0){
        # Collapse data
        summary_pred <- df %>%
          group_by(district, fac_type_original, program_plot, item) %>%
          summarise_at(vars(available, available_prob_predicted, availability_change_prop), list(mean))

        pred_col_number <- length(colnames(summary_pred))
        colnames(summary_pred)[pred_col_number] <- paste0('change_proportion_scenario', j+1)
        all_predictions_df <- summary_pred
      } else{
        # Collapse data
        summary_pred <- df %>%
          group_by(district, fac_type_original, program_plot, item) %>%
          summarise_at(vars(availability_change_prop), list(mean))

        pred_col_number <- length(colnames(summary_pred))
        colnames(summary_pred)[pred_col_number] <- paste0('change_proportion_scenario', j+1) # names(top_5_features_for_cons_availability)[j]

        all_predictions_df <- merge(all_predictions_df, summary_pred, by = c('district', 'fac_type_original', 'program_plot', 'item'))
      }
      j <- j + 1
}

###########################################################
# 3. Export predictions
###########################################################
write.csv(all_predictions_df,paste0(path_to_outputs, "predictions/predicted_consumable_availability_regression_scenarios.csv"), row.names = TRUE)

##################################################################
# 4. Plot predicted availability under 5 scenarios
################################################################
# Plot original values
p_original <- ggplot(all_predictions_df, aes(item, district,  fill= available_prob_predicted)) +
  geom_tile() +
  facet_wrap(~fac_type_original) +
  scale_fill_viridis_c(limits = c(0, 1), option = "viridis", direction = -1) +
  theme(axis.text.x = element_text(angle = 45 , vjust = 0.7, size = 1),
        axis.title.x = element_text(size = 8), axis.title.y = element_text(size = 8),
        axis.text.y = element_text(size = 4),
        legend.position = 'none',
        plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5)) +
  labs(title = "Probability of consumable availability - actual",
       subtitle =paste0("Global average = ", round(mean(all_predictions_df$available_prob_predicted) *100, 2),"%")) +
  xlab("consumable")

# Plot predicted values
p_predict <- ggplot(all_predictions_df, aes(item, district,  fill= available_prob_predicted * change_proportion_scenario1)) +
  geom_tile() +
  facet_wrap(~fac_type_original) +
  scale_fill_viridis_c(limits = c(0, 1), option = "viridis", direction = -1) +
  theme(axis.text.x = element_text(angle = 45 , vjust = 0.7, size = 1),
        axis.title.x = element_text(size = 8), axis.title.y = element_text(size = 8),
        axis.text.y = element_text(size = 4),
        legend.position = 'none',
        plot.title = element_text(color="black", size=14, face="bold", hjust = 0.5))  +
  labs(title = "Probability of consumable availability - predicted \n (all items are consumables rather than drugs)",
       subtitle =paste0("Global average = ", round(mean(all_predictions_df$available_prob_predicted * all_predictions_df$change_proportion_scenario1) *100, 2),"%")) +
  xlab("consumable")

figure <- ggpubr::ggarrange(p_original, p_predict, # list of plots
                  labels = "AUTO", # labels
                  common.legend = T, # COMMON LEGEND
                  legend = "bottom", # legend position
                  align = "hv", # Align them both, horizontal and vertical
                  nrow = 2)  %>% # number of rows
  ggexport(filename = paste0(path_to_outputs, "predictions/figures/pred_item_is_consumable_other_than_drug.pdf"))

# TODO Update this figure to include all scenarios
