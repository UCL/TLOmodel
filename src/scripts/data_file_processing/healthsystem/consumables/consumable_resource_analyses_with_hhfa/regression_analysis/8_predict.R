# This script loads the outputs of the final regression model
# and generates predictions for consumable availability scenarios

###########################################################
# 1. Set up
###########################################################

# 1.1 Run setup script
#---------------------------------
source(paste0(path_to_scripts, "3b_data_setup_for_regression.R"))

###########################################################
# 2. Load model outputs
###########################################################
load(paste0(path_to_outputs, "regression_results/model_lit.rdta"))
load(paste0(path_to_outputs, "regression_results/model_base.rdta"))
load(paste0(path_to_outputs, "regression_results/model_fac_item_re.rdta"))

# Run new model which can be used in the LinearModel functionality of the TLO model
model_tlo_lm <- glm(available ~ fac_type + fac_owner + fac_urban + functional_computer +
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
                             drug_transport_self  + item + district, 
                           family = binomial(logit),
                           data = df_for_fac_item_re_sorted) 
save(model_tlo_lm, file = "2 outputs/regression_results/model_tlo_lm.rdta")

# Test the accuracy of this model
k = 5
kfold_cross_validation(model_tlo_lm, df_for_fac_item_re_sorted, k)
accuracy_tlo_lm <- accuracy
accuracy_tlo_lm_sd <- accuracy_sd
accuracy_tlo_lm_se <- accuracy_tlo_lm_sd/sqrt(length(fitted(model_tlo_lm)))
# The accuracy of the model with item fixed effects is 79.3% (same with district fixed effects) and without item fixed effects it’s only 57.3% 

# Extract consumable list is the original data
extract <- df_orig[which(!is.na(df_orig$available)),] %>% 
  group_by(program, item) %>%
  summarise_at(vars(available), list(mean))

write.csv(extract, paste0(path_to_outputs, "tables/full_item_list_hhfa.csv"), row.names = TRUE)


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

# 3.1 All facilities have computers
####################################
df_pred_computer <- df_regress
df_pred_computer['functional_computer'] = 1 # Set computer availability
df_pred_computer$available_prob <- rep(NA,nrow(df_pred_computer)) # empty column

newpred_computer <- predict(model_fac_item_re,newdata=df_pred_computer, type = "response") # Predict availability
save(newpred_computer, file = "2 outputs/predictions/pred_computer_all.rdta")
df_pred_computer$available_prob <- newpred_computer 

# Update probability values to binary
df_pred_computer$available_predict <- rep(0,nrow(df_pred_computer))
df_pred_computer$available_predict[df_pred_computer$available_prob <0.5] <- 0
df_pred_computer$available_predict[df_pred_computer$available_prob >0.5] <- 1

# Check that the prediction improves availability from the base case
stopifnot(mean(df_pred_computer$available, na.rm = TRUE) < 
            mean(df_pred_computer$available_predict, na.rm = TRUE))

# Count the number of instances where the availability reduces
df_pred_computer$functional_computer_orig <- df_regress$functional_computer

length(unique(df_regress[which(df_regress$functional_computer == 1),]$fac_code))

a <- sum((df_pred_computer$available_predict > df_pred_computer$available) &
      (df_pred_computer$functional_computer_orig < df_pred_computer$functional_computer))/
  sum((df_pred_computer$functional_computer_orig < df_pred_computer$functional_computer))
b <- sum((df_pred_computer$available_predict < df_pred_computer$available) &
      (df_pred_computer$functional_computer_orig < df_pred_computer$functional_computer))/
  sum((df_pred_computer$functional_computer_orig < df_pred_computer$functional_computer))
c <- length(unique(df_regress[which(df_regress$functional_computer == 0),]$fac_code))

print(paste0("Among ",c,  " facilities which previously did not have computers, availability changed from 0 to 1 in ",
             round(a*100,2), " % of instances, and changed from 1 to 0 in ", round(b*100,2), " % of the instances."))

# Extract .csv for model simulation
write.csv(summary_pred_computer,paste0(path_to_outputs, "predictions/summary_pred_computer.csv"), row.names = TRUE)

# 3.2 All facilities have pharmacists managing drug orders
##########################################################
df_pred_pharma <- df_regress
df_pred_pharma['incharge_drug_orders'] = 'pharmacist or pharmacy technician' 
df_pred_pharma$available_prob <- rep(NA,nrow(df_pred_pharma)) # empty column

newpred_pharma <- predict(model_fac_item_re,newdata=df_pred_pharma, type = "response") # Predict availability
save(newpred_pharma, file = "2 outputs/predictions/pred_pharma_all.rdta")
df_pred_pharma$available_prob <- newpred_pharma 

# Update probability values to binary
df_pred_pharma$available_predict <- rep(0,nrow(df_pred_pharma))
df_pred_pharma$available_predict[df_pred_pharma$available_prob <0.5] <- 0
df_pred_pharma$available_predict[df_pred_pharma$available_prob >0.5] <- 1

# Check that the prediction improves availability from the base case
stopifnot(mean(df_pred_pharma$available, na.rm = TRUE) < 
            mean(df_pred_pharma$available_predict, na.rm = TRUE))

#################################
# 4. Plot predicted availability
###############################
# 4.1 Computer
###############################
# Plot predicted values
# Collapse data
summary_pred_computer <- df_pred_computer %>% 
  group_by(district, fac_type, program_plot, item) %>%
  summarise_at(vars(available_predict, available), list(mean))

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

# If we want common axis titles
#annotate_figure(figure, left = textGrob("District", rot = 90, vjust = 1, gp = gpar(cex = 1.3)),
#                bottom = textGrob("Consumable", gp = gpar(cex = 1.3))) %>% # 
#  ggexport(filename = "2 outputs/figures/prediction/pred_computer.pdf") 

# 4.2 Person in-charge of drug orders
#####################################
# Plot predicted values
# Collapse data
summary_pred_pharma <- df_pred_pharma %>% 
  group_by(district, fac_type, program_plot, item) %>%
  summarise_at(vars(available_predict, available), list(mean))

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

