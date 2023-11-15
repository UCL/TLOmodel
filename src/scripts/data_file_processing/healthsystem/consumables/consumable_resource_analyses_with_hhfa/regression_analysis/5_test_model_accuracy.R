# This file generates confusion matrices for the 4 regression models in 4_regression_analysis.R to compare model accuracy
# This script can be run independently (if regression model outputs from 4_regression_analysis.R are already saved) OR after running the following code. 
# source(paste0(path_to_scripts, "4_regression_analysis.R"))

###########################################################
# 1. Data setup
###########################################################
# 1.1 Run previous setup files
#---------------------------------
#source(paste0(path_to_scripts, "3b_data_setup_for_regression.R"))

###########################################################
# 2. Load model outputs
###########################################################
load(paste0(path_to_outputs, "regression_results/model_lit.rdta"))
load(paste0(path_to_outputs, "regression_results/model_base.rdta"))
load(paste0(path_to_outputs, "regression_results/model_fac_re.rdta"))
load(paste0(path_to_outputs, "regression_results/model_fac_item_re.rdta"))

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
kfold_cross_validation(model_lit, df_for_lit, k)
accuracy_lit <- accuracy
accuracy_lit_sd <- accuracy_sd
accuracy_lit_se <- accuracy_lit_sd/sqrt(length(fitted(model_lit)))

# 2.2 Basic Logistic regression model with facility-features from stepwise
#--------------------------------------------------------------------------
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
write.csv(kfold_cross_validation_results,paste0(path_to_outputs, "regression_results/kfold_cross_validation_results.csv"))


