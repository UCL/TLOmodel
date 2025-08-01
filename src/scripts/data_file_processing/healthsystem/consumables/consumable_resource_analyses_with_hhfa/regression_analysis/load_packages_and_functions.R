# This script loads the necessary packages for the regression analysis
# and creates functions used in the feature manipulation and regression analysis scripts.

# 1. Load libraries #
#####################
install.packages("pacman")
pacman::p_load(magrittr, # for %>% to work
               estimatr, # for lm_robust to work (clustered SE)

               dplyr,
               modeest,
               broom.mixed, # for tidy to work to generate conf int table
               stringr,

               #Excel packages
               caTools,
               readxl,
               writexl,
               openxlsx, #for write.xlsx
               readr,
               ggcorrplot,

               # Regression packages
               nlme, # random effects regression - lme
               lmerTest, # random effects regression - lmer
               ggfortify, # for diagnostic plots
               glmmTMB, # Multilevel regression model
               MASS, # to run stepAIC with BIC criterion

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
               caret, # to run k-fold cross validation
               effects, # to allow for type= "eff" in plot_model
               cvTools, # for k-fold cross validation
               fastDummies # to create dummies from categorical variable
               )

# Set paths to store outputs
dir.create(file.path(path_to_local_repo, "outputs/", "regression_analysis"))
path_to_outputs <- paste0(path_to_local_repo, "outputs/", "regression_analysis/")
dir.create(file.path(path_to_outputs, "regression_results"))
dir.create(file.path(path_to_outputs, "predictions"))
dir.create(file.path(paste0(path_to_outputs, "predictions/", "figures")))
dir.create(file.path(path_to_outputs, "tables"))
dir.create(file.path(path_to_outputs, "figures"))

# 2. Create functions #
#######################
# 2.1 Cap variables (or remove outliers)
#---------------------------------------
edit_outliers <- function(varname, cap_direction){
  caps <- quantile(varname, probs=c(.05, .99), na.rm = T)
  if (cap_direction == "upper"){
    varname[varname > caps[2]] <- caps[2]
  }
  if (cap_direction == "lower"){
    varname[varname < caps[1]] <- caps[1]
  }
  else{
    varname[varname < caps[1]] <- caps[1]
    varname[varname > caps[2]] <- caps[2]
  }
  return(varname)
}

# 2.2 Standardise variable
#------------------------------
standardise <- function(varname){
  varname = (varname - mean(varname, na.rm = TRUE))/sd(varname, na.rm = TRUE)
  return(varname)
}

# 2.3 Function for captioning and referencing images and RMD
#--------------------------------------------------------------
fig <- local({
  i <- 0
  ref <- list()
  list(
    cap=function(refName, text) {
      i <<- i + 1
      ref[[refName]] <<- i
      paste("Figure ", i, ": ", text, sep="")
    },
    ref=function(refName) {
      ref[[refName]]
    })
})

# 2.4 Function to insert row into a dataframe
#---------------------------------------------
insertRow <- function(existingDF, newrow, r) {
  existingDF[seq(r+1,nrow(existingDF)+1),] <- existingDF[seq(r,nrow(existingDF)),]
  existingDF[r,] <- newrow
  return(existingDF)
}

# 2.5 Create forest plot of model results - main regression
#-------------------------------------------------------------------------------------------
custom_forest_plot <- function(model, ylimit= 4){
  # Choose model for graphing
  model_forest <- model

  # Cleaned row names for dataframes
  vars_of_interest <- c("fac_typeFacility_level_1b", "fac_ownerCHAM", "fac_ownerNGO","fac_ownerPrivate for profit",
                        "fac_urban",
                        "functional_computer",
                        "functional_emergency_vehicle",
                        "service_diagnostic",
                        "incharge_drug_orderscenter manager/owner","incharge_drug_ordersclinical officer",
                        "incharge_drug_ordershsa/shsa", "incharge_drug_ordersmedical assistant",
                        "incharge_drug_ordersnurse", "incharge_drug_ordersother", "incharge_drug_orderspharmacist or pharmacy technician",
                        "incharge_drug_orderspharmacy assistant",
                        "dist_todh_cat10-25 kms", "dist_todh_cat25-50 kms", "dist_todh_cat50-75 kms", "dist_todh_cat> 75 kms",
                        "dist_torms_cat10-50 kms", "dist_torms_cat50-100 kms", "dist_torms_cat100-200 kms", "dist_torms_cat> 200 kms",
                        "drug_order_fulfilment_freq_last_3mts_cat2", "drug_order_fulfilment_freq_last_3mts_cat3", "drug_order_fulfilment_freq_last_3mts_cat>= 4",
                        "item_drug",
                        "eml_priority_v")
  reg_table_rownames <- data.frame(term = vars_of_interest,
                                   newnames = c("...Level 1b",
                                                "...Christian Health Association of Malawi (CHAM)", "...Non-governmental Organisation (NGO)", "...Private for profit",
                                                "Facility is urban",
                                                "Functional computer available",
                                                "Emergency vehicle available",
                                                "Diagnostic services available",
                                                "...Center manager/owner", "...Clinical officer",
                                                "...Health Surveillance Assistant (HSA)/Senior HSA", "...Medical assistant",
                                                "...Nurse", "...Other",
                                                "...Pharmacist or Pharmacy technician", "...Pharmacist assistant",
                                                "...10-25 kms", "...25-50 kms", "...50-75 kms", "...> 75 kms",
                                                "...10-50 kms", "...50-100 kms", "...100-200 kms", "...> 200 kms",
                                                "...2", "...3", "...>=4",
                                                "Consumable is a drug",
                                                "Consumable is classified as vital"))


  # Create the dataframe "forestdf" for the forest plot (Odds ratio, and upper/lower bounds)
  #-----------------------------------------------------------------------------------------
  forest_matrix <- tidy(model_forest,conf.int=TRUE,exponentiate=TRUE)
  forest_matrix <- forest_matrix[which(forest_matrix$term %in% vars_of_interest),]
  forest_matrix[c('estimate', 'conf.low', 'conf.high')] <- lapply(forest_matrix[c('estimate', 'conf.low', 'conf.high')], round, 2)
  forest_matrix[c('p.value')] <- sprintf("%.4f",unlist(lapply(forest_matrix[c('p.value')], round, 4)))
  forest_matrix[which(forest_matrix[c('p.value')] == "0.0000"),][c('p.value')] <- "<0.0001" # added on 11 March 2023

  # Change rownames
  forest_matrix$order <- 1:length(vars_of_interest)
  forest_matrix <- merge(reg_table_rownames,forest_matrix,by="term")
  forest_matrix$term <- forest_matrix$newnames
  forest_matrix <-  forest_matrix[order(forest_matrix$order),]

  forestdf <- structure(list(labels = structure(1:length(vars_of_interest), .Label = forest_matrix$term, class = "factor"),
                             rr = forest_matrix$estimate, rrhigh = forest_matrix$conf.high, rrlow = forest_matrix$conf.low),
                        class = "data.frame", row.names = c(NA, -29L)) # changes from factor to character


  # Create the dataframe "fpplot" for the data table
  #-----------------------------------------------------------
  fplottable <- structure(list(labels = structure(1:length(vars_of_interest), .Label = forest_matrix$term, class = "factor"),
                               ci = paste0(forest_matrix$estimate, " (", forest_matrix$conf.low, " - ",forest_matrix$conf.high, ")"),
                               p = forest_matrix$p.value),
                          class = "data.frame", row.names = c(NA,-29L))

  # Add reference level rows to above dataframes
  #----------------------------------------------
  for (df_name in c('fplottable', 'forestdf')){
    df_results <- get(df_name)
    df_results$labels <- as.character(df_results$labels) # change format to character in order to add new rows
    r_fac_type <- c("Facility level (Ref: Level 1a)", rep(NA, dim(df_results)[2]-1))
    r_fac_owner <- c("Facility owner (Ref: Government)", rep(NA, dim(df_results)[2]-1))
    r_incharge_drug_orders <- c("Person in charge of drug orders (Ref: Drug store clerk)", rep(NA, dim(df_results)[2]-1))
    r_dist_todh <- c("Distance from DHO (Ref: 0-10kms)", rep(NA, dim(df_results)[2]-1))
    r_dist_torms <- c("Distance from RMS (Ref: 0-10kms)", rep(NA, dim(df_results)[2]-1))
    r_drug_order_fulfilment_freq_last_3mts <- c("Quarterly drug order fulfillment frequency (Ref: 1)", rep(NA, dim(df_results)[2]-1))
    regressor_headings = c("Facility level (Ref: Level 1a)", "Facility owner (Ref: Government)",
                           "Person in charge of drug orders (Ref: Drug store clerk)", "Distance from DHO (Ref: 0-10kms)",
                           "Distance from RMS (Ref: 0-10kms)", "Quarterly drug order fulfillment frequency (Ref: 1)")

    a <- which(df_results$labels == "...Level 1b")
    df_results <- insertRow(df_results , r_fac_type, a)
    b <- which(df_results$labels == "...Christian Health Association of Malawi (CHAM)")
    df_results <- insertRow(df_results , r_fac_owner, b)
    c <- which(df_results$labels == "...Center manager/owner")
    df_results <- insertRow(df_results , r_incharge_drug_orders, c)
    d <- which(df_results$labels == "...10-25 kms")
    df_results <- insertRow(df_results , r_dist_todh, d)
    e <- which(df_results$labels == "...10-50 kms")
    df_results <- insertRow(df_results , r_dist_torms, e)
    f <- which(df_results$labels == "...2")
    df_results <- insertRow(df_results , r_drug_order_fulfilment_freq_last_3mts, f)

    # Add alternating color scheme
    if((dim(df_results)[1] %% 2) == 0){
      df_results$colour <- rep(c("white", "gray"), dim(df_results)[1]/2)
    } else {
      df_results$colour <- c(rep(c("white", "gray"), dim(df_results)[1]/2), "white")
    }

    assign(df_name, df_results)
  }

  column_headers_space1 <- c("", NA, NA, NA, "white")
  forestdf <- insertRow(forestdf , column_headers_space1, 1)
  forestdf <<- insertRow(forestdf , column_headers_space1, 1)

  column_headers <- c("", "Odds ratio (95% CI)", "p-value", "white")
  column_headers_space2 <- c("", "__________________", "________", "white")
  fplottable <- insertRow(fplottable , column_headers, 1)
  fplottable <<- insertRow(fplottable , column_headers_space2, 2)


  # Create data table for plot
  #-------------------------------
  # Ensure that the order of labels does not change in the table
  fplottable$labels <- factor(fplottable$labels, levels = rev(unique(fplottable$labels)))

  data_table <<- ggplot(data = fplottable, aes(y = labels, fontface = ifelse(forestdf$labels %in% regressor_headings, "italic", "plain")),
                        family = "Times") +
    geom_hline(aes(yintercept = labels, colour = colour), size = 3) +
    geom_text(aes(x = 0, label = labels), hjust = 0, size = 2.5) +
    geom_text(aes(x = 5, label = ci), size = 2.5) +
    geom_text(aes(x = 7, label = p), hjust = 1, size = 2.5) +
    scale_colour_identity() +
    theme_void() +
    theme(plot.margin = ggplot2::margin(5, 0, 32, 0))

  # Create forest plot for plot
  #-------------------------------
  forestdf[c('rr', 'rrhigh', 'rrlow')] <- sapply(forestdf[c('rr', 'rrhigh', 'rrlow')],as.numeric)

  p <<- ggplot(forestdf, aes(x = rr, y = labels, xmin = rrlow, xmax = rrhigh)) +
    geom_hline(aes(yintercept = labels, colour = colour), size = 3) + # creates the grid #4.5
    geom_pointrange(shape = 20, fill = "black") + # previous 22
    geom_vline(xintercept = 1, linetype = 3) + # vertical line at x = 1
    xlab("Odds Ratio with 95% Confidence Interval") +
    ylab("") +
    theme_classic() +
    scale_colour_identity() +
    scale_y_discrete(limits = rev(forestdf$labels)) +
    scale_x_log10(limits = c(0.20, ylimit*(1.15)),
                  breaks =  0.25 * 2^(seq(0,log(4*ylimit)/log(2),1)),
                  labels = as.character(0.25 * 2^(seq(0,log(4*ylimit)/log(2),1))), expand = c(0,0),
                  oob=scales::squish) +
    theme(axis.text.y = element_blank(), axis.title.y = element_blank()) +
    theme(plot.margin = ggplot2::margin(5, 0, 5, 0)) +
    theme(axis.title=element_text(size=6, face = "bold"))
}
