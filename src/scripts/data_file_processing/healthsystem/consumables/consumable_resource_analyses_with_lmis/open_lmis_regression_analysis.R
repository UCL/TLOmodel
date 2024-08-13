# 1. Load libraries #
#####################
install.packages("pacman")
pacman::p_load(magrittr, # for %>% to work
               dplyr,
               modeest,
               broom.mixed, # for tidy to work to generate conf int table

               #Excel packages
               readxl,
               writexl,
               readr,

               # Regression packages
               nlme, # random effects regression - lme
               lmerTest, # random effects regression - lmer
               ggfortify, # for diagnostic plots
               glmmTMB, # Multilevel regression model
               MASS, # to run stepAIC with BIC criterion
               fixest, # clustered standard errors

               # visualisation packages
               jtools, # for regression visualisations
               sjPlot, # for plot_model
               sjmisc, # for plot_model
               viridis,
               ggpubr,
               ggplot2,
               cowplot, # for plot_grid (combine multiple graphs)
               gridExtra, # for combining forest plots (text grob)
               grid,

               # packages for tables
               gtsummary, # to get p-values in summary tables
               huxtable,  # to export regression summary tables using export_summs
               margins, # to calculate average marginal effects from logistic regression results
               janitor, # for tabyl command - get summary of categorical variable
               aod, # wald.test

               MESS, # for model comparison
               car, # to run Wald tests on regression outputs (Anova function)
               effects, # to allow for type= "eff" in plot_model
               cvTools, # for k-fold cross validation
               fastDummies # to create dummies from categorical variable
               )

# Set paths to store outputs
path_to_local_repo <- "/Users/sm2511/PycharmProjects/TLOmodel/" # Change this if different user
path_to_data <- paste0(path_to_local_repo, "outputs/", "openlmis_data/")
dir.create(file.path(path_to_data, "regression_analysis"))

# Load data
df <- read.csv(paste0(path_to_data, "regression_subset_df.csv"))

#######################################################################
# 2. Set up Dataframe for Regression analysis
#######################################################################
# Drop items and facilities with too few facilities reporting/items reported respectively
df_not_na_by_fac <- df %>%
  group_by(fac_name) %>%
  summarise(available_count = sum(!is.na(available_prob))) %>%
  arrange(available_count)

df_not_na_by_item <- df %>%
  group_by(item_code) %>%
  summarise(available_count = sum(!is.na(available_prob))) %>%
  arrange(available_count)

# Make a list of items with less than 10% facilities reporting (these are the consumables relevant to higher level RMNCH services for which
# only 64 facilities submitted reports)
items_with_too_few_obs <- subset(df_not_na_by_item, df_not_na_by_item$available_count < 1000)['item_code']
items_with_too_few_obs <- as.list(items_with_too_few_obs)

# Make a list of facilities with less than 10% items reported
print(max(df_not_na_by_fac$available_count))
facs_with_too_few_obs <- subset(df_not_na_by_fac, df_not_na_by_fac$available_count <= 0.1*max(df_not_na_by_fac$available_count))['fac_name']
facs_with_too_few_obs <- as.list(facs_with_too_few_obs)

df_for_regs <- subset(df, !(fac_name %in% facs_with_too_few_obs$fac_name))
df_for_regs <- subset(df_for_regs, !(item_code %in% items_with_too_few_obs$item_code))

print(paste(length(facs_with_too_few_obs$fac_name), " facilities dropped."))
print(paste(length(items_with_too_few_obs$item_code), " items dropped."))

# Set up dataframe for facility and item random effects model
#------------------------------------------------------------
# Note that this is model 4 in the Lancet GH paper. We do not include models 1-3 because these are not used for analysis on the impact of consumable availability scenarios
# add variables for random effects regression (re_reg)
chosen_varlist_for_re_reg <- c("available_prob", "year", "category", "is_vital", "fac_level", "district", "item_code", "month", "fac_name")
df_for_re_reg <- df_for_regs[chosen_varlist_for_re_reg]
df_for_re_reg <- na.omit(df_for_re_reg)

# Sort by fac_code
df_for_re_reg_sorted <-  df_for_re_reg[order(df_for_re_reg$fac_name, df_for_re_reg$item_code),]
# Create an numeric value for fac_code (clustering variable)
df_for_re_reg_sorted$fac_id <- as.integer(factor(df_for_re_reg_sorted$fac_name,levels=unique(df_for_re_reg_sorted$fac_name)))
df_for_re_reg_sorted <- as.data.frame(df_for_re_reg_sorted)

# Clean columns for regression analysis
df_for_re_reg_sorted$item_code <- factor(df_for_re_reg_sorted$item_code)
df_for_re_reg_sorted$is_vital[df_for_re_reg_sorted$is_vital == ""] <- "False"
df_for_re_reg_sorted_months_collapsed <- df_for_re_reg_sorted %>%
  group_by(across(c("year", "category", "is_vital", "fac_level", "district", "item_code", "fac_name"))) %>%
  summarise(mean_available_prob = mean(available_prob, na.rm = TRUE), .groups = 'drop')

df_for_re_reg_sorted_months_collapsed_without_2023 <- df_for_re_reg_sorted_months_collapsed %>% filter(year != 2023)

#######################################################################
# 3. Run Regression Model
#######################################################################
# Fixed effects model
model_fe_yearly_data <- glm(mean_available_prob ~ fac_level + district + category + is_vital + year +
                             item_code + year*category + year*fac_level + year*is_vital,
                           family = binomial(logit), #gaussian(link = "identity")
                           data = df_for_re_reg_sorted_months_collapsed)
summary(model_fe_yearly_data)

model_fe_without_2023_yearly_data <- glm(mean_available_prob ~ fac_level + district + category + is_vital + year +
                             item_code + year*category + year*fac_level + year*is_vital,
                           family = binomial(logit), #gaussian(link = "identity")
                           data = df_for_re_reg_sorted_months_collapsed_without_2023)
summary(model_fe_without_2023_yearly_data)

model_cse_item_yearly_data <- feglm(mean_available_prob ~ fac_level + district + category + is_vital + year +
                               year*category + year*fac_level + year*is_vital | item_code,
                              family = binomial(link = "logit"),
                              data = df_for_re_reg_sorted_months_collapsed)
summary(model_cse_item_yearly_data)

model_cse_item_without_2023_yearly_data <- feglm(mean_available_prob ~ fac_level + district + category + is_vital + year +
                              year*category + year*fac_level + year*is_vital | item_code,
                           family = binomial(logit), #gaussian(link = "identity")
                           data = df_for_re_reg_sorted_months_collapsed_without_2023)
summary(model_cse_item_without_2023_yearly_data)

model_cse_item_and_fac_yearly_data <- feglm(mean_available_prob ~ year + fac_level + district + category + is_vital +
                               year*category + year*fac_level + year*is_vital + year*district,
                              family = binomial(link = "logit"),
                              data = df_for_re_reg_sorted_months_collapsed, clustervar = c("item_code", "fac_name"))
summary(model_cse_item_and_fac_yearly_data)

linmodel_cse_item_and_fac_yearly_data <- feols(mean_available_prob ~ year + fac_level + district + category + is_vital +
                  year * category + year * fac_level + year * is_vital + year*district ,
                  data = df_for_re_reg_sorted_months_collapsed,
                  vcov = ~ item_code + fac_name)
summary(linmodel_cse_item_and_fac_yearly_data)

linmodel_cse_item_and_fac_yearly_data_without_2023 <- feols(mean_available_prob ~ year + fac_level + district + category + is_vital +
                  year * category + year * fac_level + year * is_vital + year*district ,
                  data = df_for_re_reg_sorted_months_collapsed_without_2023,
                  vcov = ~ item_code + fac_name)
summary(linmodel_cse_item_and_fac_yearly_data_without_2023)

model_cse_item_and_fac_yearly_data_nointeraction <- feglm(mean_available_prob ~ year + fac_level + district + category + is_vital,
                              family = binomial(link = "probit"),
                              data = df_for_re_reg_sorted_months_collapsed, clustervar = c("item_code", "fac_name"))
summary(model_cse_item_and_fac_yearly_data_nointeraction)

model_cse_item_and_fac_without_2023_yearly_data <- feglm(mean_available_prob ~ year + fac_level + district + category + is_vital +
                               year*category + year*fac_level + year*is_vital + year*district,
                           family = binomial(logit), #gaussian(link = "identity")
                           data = df_for_re_reg_sorted_months_collapsed_without_2023,
                                                         clustervar = c("item_code", "fac_name"))
summary(model_cse_item_and_fac_without_2023_yearly_data)

model_fe_monthly_data <- glm(available_prob ~ fac_level + district + category + is_vital + year +
                             item_code + year*category + year*fac_level + year*is_vital,
                           family = binomial(logit), #gaussian(link = "identity")
                           data = df_for_re_reg_sorted)
summary(model_fe_monthly_data)

model_fe_without_2023_monthly_data <- glm(available_prob ~ fac_level + district + category + is_vital + year +
                             item_code + year*category + year*fac_level + year*is_vital,
                           family = binomial(logit), #gaussian(link = "identity")
                           data = df_for_re_reg_sorted_without_2023)
summary(model_fe_without_2023_monthly_data)

# Random effects model (takes too loong to converge)
model_item_re <- glmer(mean_available_prob ~ fac_level + district + category + is_vital + year +
                             (1|item_code),
                           family = binomial(logit), #gaussian(link = "identity")
                           data = df_for_re_reg_sorted_months_collapsed,
                           control = glmerControl(optimizer = "bobyqa",
                                                  optCtrl=list(maxfun=1e5),
                                                  calc.derivs = TRUE)) #  + (1|fac_name)

# Calculate the Intra-class correlation
icc_between_model_fac_item_re <- performance::icc(model_fac_item_re, by_group = TRUE)

# Save regression results
#-------------------------
save(model_fe_monthly_data, file = paste0(path_to_data, "regression_analysis/model_fe_monthly_data.rdta"))
save(model_fe_without_2023_monthly_data, file = paste0(path_to_data, "regression_analysis/model_fe_without_2023_monthly_data.rdta"))
save(model_fe_yearly_data, file = paste0(path_to_data, "regression_analysis/model_fe_yearly_data.rdta"))
save(model_fe_without_2023_yearly_data, file = paste0(path_to_data, "regression_analysis/model_fe_without_2023_yearly_data.rdta"))
save(model_cse_item_yearly_data, file = paste0(path_to_data, "regression_analysis/model_cse_item_yearly_data.rdta"))
save(model_cse_item_and_fac_yearly_data, file = paste0(path_to_data, "regression_analysis/model_cse_item_and_fac_yearly_data.rdta"))
save(linmodel_cse_item_and_fac_yearly_data, file = paste0(path_to_data, "regression_analysis/linmodel_cse_item_and_fac_yearly_data.rdta"))

# Extract results to excel file
linear_model <- tidy(linmodel_cse_item_and_fac_yearly_data, conf.int = TRUE)
linear_model_without_2023 <- tidy(linmodel_cse_item_and_fac_yearly_data_without_2023, conf.int = TRUE)
logit_model <- tidy(model_cse_item_and_fac_yearly_data, conf.int = TRUE)
logit_model_without_2023 <- tidy(model_cse_item_and_fac_without_2023_yearly_data, conf.int = TRUE)

write.csv(linear_model, file = paste0(path_to_data, "regression_analysis/linear_regression_model.csv"))
write.csv(linear_model_without_2023, file = paste0(path_to_data, "regression_analysis/linear_regression_model_without_2023.csv"))
write.csv(logit_model, file = paste0(path_to_data, "regression_analysis/logit_regression_model.csv"))
write.csv(logit_model_without_2023, file = paste0(path_to_data, "regression_analysis/logit_regression_model_without_2023.csv"))
logit_model$estimate <- exp(logit_model$estimate)
logit_model$conf.low <- exp(logit_model$conf.low)
logit_model$conf.high <- exp(logit_model$conf.high)
write.csv(logit_model, file = paste0(path_to_data, "regression_analysis/logit_regression_model_exponentiated.csv"))
logit_model_without_2023$estimate <- exp(logit_model_without_2023$estimate)
logit_model_without_2023$conf.low <- exp(logit_model_without_2023$conf.low)
logit_model_without_2023$conf.high <- exp(logit_model_without_2023$conf.high)
write.csv(logit_model_without_2023, file = paste0(path_to_data, "regression_analysis/logit_regression_model_without_2023_exponentiated.csv"))


varlist_margins <- c('fac_level', 'category', 'is_vital', 'year')
chosen_model <- model_cse_item_and_fac_yearly_data
chosen_model_without_2023 <- model_cse_item_and_fac_without_2023_yearly_data
# For the main regression model, calculate average marginal effects for the manuscript text
margin_chosen_model <- marginaleffects(chosen_model, variables = varlist_margins)

margin_fe_yearly_data <- margins(model_fe_yearly_data, type = "response", variables = varlist_margins)
write_xlsx(margin_fe_yearly_data,paste0(path_to_data, "regression_analysis/margin_fe_yearly_data.xlsx"))

# All effects
library(effects)
all_effects <- allEffects(chosen_model)
plot(all_effects)
Effect("year", chosen_model)


library(broom.helpers)
table <- chosen_model %>%
  tbl_regression(
    tidy_fun = tidy_all_effects,
    estimate_fun = scales::label_percent(accuracy = .1)
  ) %>%
  bold_labels()
table

png(paste0(path_to_data, "regression_analysis/predicted_values_by_fac_level_without_2023.png"), width = 800, height = 600)
plot_model(chosen_model_without_2023, type = "pred", terms = c( "year", "fac_level"))
dev.off()
png(paste0(path_to_data, "regression_analysis/predicted_values_by_eml_category_without_2023.png"), width = 800, height = 600)
plot_model(chosen_model_without_2023, type = "pred", terms = c( "year", "is_vital"))
dev.off()
png(paste0(path_to_data, "regression_analysis/predicted_values_by_programmatic_category_without_2023.png"), width = 800, height = 600)
plot_model(chosen_model_without_2023, type = "pred", terms = c( "year", "category"))
dev.off()
png(paste0(path_to_data, "regression_analysis/predicted_values_by_district_without_2023.png"), width = 800, height = 600)
plot_model(chosen_model_without_2023, type = "pred", terms = c( "year", "district"))
dev.off()

png(paste0(path_to_data, "regression_analysis/predicted_values_by_fac_level.png"), width = 800, height = 600)
plot_model(chosen_model, type = "pred", terms = c( "year", "fac_level"))
dev.off()
png(paste0(path_to_data, "regression_analysis/predicted_values_by_eml_category.png"), width = 800, height = 600)
plot_model(chosen_model, type = "pred", terms = c( "year", "is_vital"))
dev.off()
png(paste0(path_to_data, "regression_analysis/predicted_values_by_programmatic_category.png"), width = 800, height = 600)
plot_model(chosen_model, type = "pred", terms = c( "year", "category"))
dev.off()
png(paste0(path_to_data, "regression_analysis/predicted_values_by_district.png"), width = 800, height = 600)
plot_model(chosen_model, type = "pred", terms = c( "year", "district"))
dev.off()
plot_model(chosen_model, type = "pred", terms = c("fac_level"))
png(paste0(path_to_data, "regression_analysis/predicted_values_by_programmatic_category_static.png"), width = 800, height = 600)
plot_model(chosen_model, type = "pred", terms = c("category"))
dev.off()
plot_model(chosen_model, type = "pred", terms = c("is_vital"))
plot_model(chosen_model, type = "pred", terms = c("year", "category ['hiv','tb','malaria']", "fac_level"))

# Summarise results in a table/figure
#--------------------------------
t <- tbl_regression(chosen_model, exponentiate = TRUE, conf.int = TRUE, pvalue_fun = ~style_sigfig(., digits = 4))

tbl_merge <-
  tbl_merge(
    tbls = list(t),
    tab_spanner = c("**Fixed effects**") #
  )  %>%    # build gtsummary table
  as_gt() # %>%             # convert to gt table
  gt::gtsave(             # save table as image
    filename = reg_results
  )


# Based on the chosen regression models, predict the probability of consumable availability
explanatory_df <-
