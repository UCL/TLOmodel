# This file generates confusion matrices for the 4 regression models in 4_regression_analysis.R to compare
# model accuracy
# This script can be run independently OR after running the following code. 
# source("0 scripts/4_regression_analysis.R")

###########################################################
# 1. Data setup
###########################################################
# 1.1 Run previous setup files
#---------------------------------
source("0 scripts/0_packages_and_functions.R")
source("0 scripts/1_data_setup.R")
source("0 scripts/2_feature_manipulation.R")

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
ggsave(plot = corrplot_final_varlist, filename = "2 outputs/figures/correlation_final_varlist.png")

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

ggsave(plot = corrplot_final_varlist_post, filename = "2 outputs/figures/correlation_final_varlist_post.png")


# Convert binary variables to factors
bin_exp_vars <- c('fac_urban', 'functional_emergency_vehicle' ,'functional_computer', 
                  'service_diagnostic' , 'functional_refrigerator', 'functional_landline',
                  'fuctional_mobile', 'functional_toilet', 'functional_handwashing_facility', 
                  'outpatient_only', 'service_hiv', 'service_othersti', 'service_malaria',
                  'service_tb', 'service_fp', 'service_imci', 'source_drugs_ngo', 
                  'source_drugs_pvt', 'drug_transport_self', 'item_drug')

#######################################################################
# 2. CREATE NECESSARY DATAFRAMES FOR ACCURACY ANALYSIS
#######################################################################
# 2.1 Logistic regression model with independent vars based on literature (Model 1)
#--------------------------------------------------------------------------------
# Set up data for regression analysis
stockout_factors_from_lit <- stockout_factors_from_lit[!(stockout_factors_from_lit %in%
                                                           vars_low_variation)]
stockout_factors_from_lit <- stockout_factors_from_lit[!(stockout_factors_from_lit %in%
                                                           vars_highly_correlated)]

stockout_factors_from_lit = stockout_factors_from_lit[!(stockout_factors_from_lit %in% continuous_variables)]
stockout_factors_from_lit = unlist(c(stockout_factors_from_lit,continuous_variables_cat_version ))

df_for_lit <- df[unlist(c(stockout_factors_from_lit, 'available'))]
df_for_lit <- na.omit(df_for_lit)

# 2.2 Basic Logistic regression model with facility-features from stepwise  (Model 2)
#------------------------------------------------------------------------------------------
# Set up data for regression analysis
chosen_varlist_for_base <-  chosen_varlist[!(chosen_varlist %in% c('item'))] 
# add program and district fixed effects
chosen_varlist_for_base <- unlist(c(chosen_varlist_for_base, 'program', 'district'))

df_for_base <- df[chosen_varlist_for_base]
df_for_base <- na.omit(df_for_base)

# 2.3 Model with facility random errors (Model 3)
#------------------------------------------------------
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

# 2.4 Model with item and facility random errors  (Model 4)
#------------------------------------------------------------------
# Drop above items from the dataset (# already dropped)
df_for_fac_item_re_sorted <- subset(df_for_fac_re_sorted, !(item %in% items_with_too_few_obs$item))

###########################################################
# 2. Load model outputs
###########################################################
load("2 outputs/regression_results/model_lit.rdta")
load("2 outputs/regression_results/model_base.rdta")
load("2 outputs/regression_results/model_fac_re.rdta")
load("2 outputs/regression_results/model_fac_item_re.rdta")

###########################################################
# 3. Function to run k-fold cross validation
###########################################################
kfold_cross_validation <- function(model, dataset, k){
  # Split data for k-fold cross-validation
  folds <- cvFolds(NROW(dataset), K=k)
  dataset$holdoutpred <- rep(0,nrow(dataset))
  
  # Run model on 5 sections of the data
  for(i in 1:k){
    print(paste("running fold ", i))
    train <- dataset[folds$subsets[folds$which != i], ] #Set the training set
    validation <- dataset[folds$subsets[folds$which == i], ] #Set the validation set
 
    model <<- update(model, data = train)
      
    newpred <- predict(model,newdata=validation, type="response") # Get the predicitons for the validation set (from the model just fit on the train data)
    
    dataset[folds$subsets[folds$which == i], ]$holdoutpred <- newpred #Put the hold out prediction in the data set for later use
  }
  
  # Create a column with predicted probabilities
  dataset$prediction <- rep(0,nrow(dataset))
  dataset$prediction[dataset$holdoutpred <0.5] <- 0
  dataset$prediction[dataset$holdoutpred >0.5] <- 1
  
  # Create a column recording accuracy of prediction
  dataset$correct_prediction <- rep(0,nrow(dataset))
  dataset$correct_prediction[dataset$prediction == dataset$available] <- 1
  dataset$correct_prediction[dataset$prediction != dataset$available] <- 0
  
  # Calculate model accuracy
  accuracy <<- sum(dataset$correct_prediction)/length(dataset$correct_prediction)
  accuracy_sd <<- sd(dataset$correct_prediction)
  #matrix <<- dataset$correct_prediction
  
}

# 2. Train models and generate confusion matrices #
###################################################
k = 10 # choose number for k-fold accuracy test
# 1. Run chosen regression model
####################################
# 1.1 Basic Logistic regression model
#--------------------------------------
# Set up data for regression analysis
stockout_factors_from_lit <- stockout_factors_from_lit[!(stockout_factors_from_lit %in%
                                                           vars_low_variation)]
stockout_factors_from_lit <- stockout_factors_from_lit[!(stockout_factors_from_lit %in%
                                                           vars_highly_correlated)]
df_for_lit <- df[unlist(c(stockout_factors_from_lit, 'available'))]
df_for_lit <- na.omit(df_for_lit)

kfold_cross_validation(model_lit, df_for_lit, k)
accuracy_lit <- accuracy
accuracy_lit_sd <- accuracy_sd
accuracy_lit_se <- accuracy_lit_sd/sqrt(length(fitted(model_lit)))


# 2.2 Basic Logistic regression model with facility-features from stepwise
#--------------------------------------------------------------------------
# Set up data for regression analysis
chosen_varlist_for_base <-  chosen_varlist[!(chosen_varlist %in% c('item'))] 
# add program and district fixed effects
chosen_varlist_for_base <- unlist(c(chosen_varlist_for_base, 'program', 'district'))

df_for_base <- df[chosen_varlist_for_base]
df_for_base <- na.omit(df_for_base)

kfold_cross_validation(model_base, df_for_base, k)
accuracy_base <- accuracy
accuracy_base_sd <- accuracy_sd
accuracy_base_se <- accuracy_base_sd/sqrt(length(fitted(model_base)))

# 2.3 Model with facility random errors
#--------------------------------------

kfold_cross_validation(model_fac_re, df_for_fac_re_sorted, k)
accuracy_fac_re <- accuracy
accuracy_fac_re_sd <- accuracy_sd
accuracy_fac_re_se <- accuracy_fac_re_sd/sqrt(length(fitted(model_fac_re)))

# 2.4 Model with item and facility random errors
#--------------------------------------------------
kfold_cross_validation(model_fac_item_re, df_for_fac_item_re_sorted, k)
accuracy_fac_item_re <- accuracy
accuracy_fac_item_re_sd <- accuracy_sd
accuracy_fac_item_re_se <- accuracy_fac_item_re_sd/sqrt(length(fitted(model_fac_item_re)))

# Extract above results in a .csv
##################################
a <- matrix(NaN, nrow = 4 , ncol = 4)
a[1,1] = accuracy_lit
a[2,1] = accuracy_base
a[3,1] = accuracy_fac_re
a[4,1] = accuracy_fac_item_re
a[1,2] = length(fitted(model_lit))
a[2,2] = length(fitted(model_base))
a[3,2] = length(fitted(model_fac_re))
a[4,2] = length(fitted(model_fac_item_re))
a[1,3] = accuracy_lit_sd
a[2,3] = accuracy_base_sd
a[3,3] = accuracy_fac_re_sd
a[4,3] = accuracy_fac_item_re_sd
a[1,4] = accuracy_lit_se
a[2,4] = accuracy_base_se
a[3,4] = accuracy_fac_re_se
a[4,4] = accuracy_fac_item_re_se

kfold_cross_validation_results <- as.data.frame(a)
rownames(kfold_cross_validation_results) <- c("Model based on literature", 
                                              "Base model", "Model with facilty RE", 
                                              "Model with facility and item RE")
colnames(kfold_cross_validation_results) <- c("Mean (accuracy)", "N", "SD(accuracy)", "SE(accuracy)")
write.csv(kfold_cross_validation_results,"2 outputs/regression_results/kfold_cross_validation_results.csv")


