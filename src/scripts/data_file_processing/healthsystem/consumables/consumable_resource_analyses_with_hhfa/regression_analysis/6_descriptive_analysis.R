# This script extracts descriptive tables for the manuscript. This script can be run independently.

# Run setup files
source(paste0(path_to_scripts, "0_packages_and_functions.R"))
source(paste0(path_to_scripts, "1_data_setup.R"))
source(paste0(path_to_scripts, "2_feature_manipulation.R"))

# 1. Set up variable lists #
###########################
desc_table_varlist_lit_bin <- c('fac_type', 'fac_owner', 'fac_urban',
                                'functional_computer', 'incharge_drug_orders',
                                'functional_emergency_vehicle', 'service_diagnostic',
                                'dist_todh_cat', 'dist_torms_cat',  'drug_order_fulfilment_freq_last_3mts_cat' )
desc_table_varlist_lit_cont <- c('dist_todh', 'drug_order_fulfilment_freq_last_3mts')
desc_table_varlist_full_bin <- unlist(c(desc_table_varlist_lit_bin,'rms', 
                                        'functional_refrigerator',  
                                        'functional_landline', 'fuctional_mobile',
                                        'functional_toilet', 'functional_handwashing_facility',
                                        'water_source_main',
                                        'outpatient_only',
                                        'service_hiv','service_othersti', 
                                        'service_malaria', 'service_tb', 
                                        'service_fp', 'service_imci',
                                        'source_drugs_ngo', 'source_drugs_pvt', 'district'))
# 'drug_transport_self' # these variables are failing
desc_table_varlist_full_cont <- unlist(c(desc_table_varlist_lit_cont))

# redefine fac_reg_df without dropping na
fac_features <- aggregate(df_for_fac_item_re_sorted[unlist(c(desc_table_varlist_full_bin))], by = list(fac_code = df_for_fac_item_re_sorted$fac_code), FUN = head, 1)
availability_by_facility <- aggregate( df_for_fac_item_re_sorted[,'available'], list(fac_code = df_for_fac_item_re_sorted$fac_code), 
                                       FUN = mean, na.rm = TRUE)
fac_df <- merge(fac_features,availability_by_facility,by="fac_code")

# 2.1 Extract descriptive table for primary analysis dataset #
#############################################################
i = 0
for (var in desc_table_varlist_full_bin){
  #m1 <- lm(available ~ fac_type, data = df)
  print(paste("processing", var))
  tempSubset <- df_for_fac_item_re_sorted[,c("available", var)]
  tempSubset <- tempSubset[complete.cases(tempSubset),]
  
  fac_df_subset <- fac_df[,c("fac_code", var)]
  fac_df_subset <- fac_df_subset[complete.cases(fac_df_subset),]
  
  # Convert binary variable to categorical
  if (is.numeric(tempSubset[[var]]) && all(unique(is.numeric(tempSubset[[var]])) %in% c(0, 1, NA, NaN))){
    tempSubset[[var]] <- ifelse(tempSubset[[var]]== 0, "No", ifelse(tempSubset[[var]] == 1, "Yes", tempSubset[[var]]))
  }
  
  m1 <- glm(available ~ ., data = tempSubset, family = binomial) # for logistic reg results
  m1a <- fac_df_subset %>% tabyl(var) %>% filter(n > 0) # for frequency stats
  m1b <- tidy(lm(available ~ . - 1, data = tempSubset)) # for mean availability by category
  m1c <- tbl_regression(m1, exponentiate = TRUE, conf.int = TRUE) # for OR and CI from logistic reg results
  
  desc_table_var <- cbind(m1a[,1:3], m1b[1:dim(m1a),2], rbind(
    matrix("Ref", nrow = 1, ncol = 4),
    round(cbind(m1c$table_body$estimate, m1c$table_body$conf.low,
                m1c$table_body$conf.high, m1c$table_body$p.value)[2:dim(m1a)[1]+1,],2))
  )
  # Note that m1b[1:dim(m1a),2] contains dim(m1a) becauase otherwise m1b includes a 
  # row for NaN
  
  names(desc_table_var)[1] <- "variable" 
  
  if (i == 0){
    desc_table <- desc_table_var
  }
  else{
    desc_table <- rbind(desc_table, desc_table_var)
  }
  i = 1
}


# Display and extract descriptive table
desc_table
write_xlsx(desc_table,paste0(path_to_outputs, "tables/desc_table.xlsx"))

# 2.1.a Separately extract data for higher levels of care not included in the main analysis
#############################################################################################
i = 0
for (var in c("fac_type")){
  print(paste("processing", var))
  tempSubset <- df_all_levels[,c("available", var)]
  tempSubset <- tempSubset[complete.cases(tempSubset),]
  
  # Convert binary variable to categorical
  if (is.numeric(tempSubset[[var]]) && all(unique(is.numeric(tempSubset[[var]])) %in% c(0, 1, NA, NaN))){
    tempSubset[[var]] <- ifelse(tempSubset[[var]]== 0, "No", ifelse(tempSubset[[var]] == 1, "Yes", tempSubset[[var]]))
  }
  
  m1 <- glm(available ~ ., data = tempSubset, family = binomial) # for logistic reg results
  m1a <- fac_reg_df_all_levels %>% tabyl(var) # for frequency stats
  m1b <- tidy(lm(available ~ . - 1, data = tempSubset)) # for mean availability by category
  m1c <- tbl_regression(m1, exponentiate = TRUE, conf.int = TRUE) # for OR and CI from logistic reg results
  
  desc_table_var <- cbind(m1a[,1:3], m1b[1:dim(m1a),2], rbind(
    matrix("Ref", nrow = 1, ncol = 4),
    round(cbind(m1c$table_body$estimate, m1c$table_body$conf.low,
                m1c$table_body$conf.high, m1c$table_body$p.value)[2:dim(m1a)[1]+1,],2))
  )
  # Note that m1b[1:dim(m1a),2] contains dim(m1a) becauase otherwise m1b includes a 
  # row for NaN
  
  names(desc_table_var)[1] <- "variable" 
  
  if (i == 0){
    desc_table_all_levels <- desc_table_var
  }
  else{
    desc_table_all_levels <- rbind(desc_table_all_levels, desc_table_var)
  }
  i = 1
}


# Display and extract descriptive table
desc_table_all_levels
write_xlsx(desc_table,paste0(path_to_outputs, "tables/desc_table_alllevels.xlsx"))

# 2.2 Extract descriptive table for program and item   #
#--------------------------------------------------#
# Choose which dataframe is used to draw descriptive stats (M4 or M1 df)
item_desc_df <- df

# Create collapsed dataframe by item
item_exp_vars <- c('program', 'item_drug', 'eml_priority_v')
item_features <- aggregate(item_desc_df[item_exp_vars], by = list(item = item_desc_df[,'item']), FUN = head, 1)
availability_by_item<- aggregate( item_desc_df[,'available'], by = list(item = item_desc_df[,'item']), 
                                  FUN = mean, na.rm = TRUE)
desc_table_item_pt1 <- merge(item_features,availability_by_item,by="item")
desc_table_item_pt1 <- na.omit(desc_table_item_pt1)

# Get odds ratios and 95% CI for availability by item
m1 <- glm(available ~ item, data = item_desc_df, family = binomial) # for logistic reg results
m1c <- tbl_regression(m1, exponentiate = TRUE, conf.int = TRUE) # for OR and CI from logistic reg results

desc_table_item_pt2 <- cbind(m1c$table_body$label[2:dim(m1c$table_body)[1]], rbind(matrix("Ref", nrow = 1, ncol = 4),
                                                                                   round(cbind(m1c$table_body$estimate, m1c$table_body$conf.low,
                                                                                               m1c$table_body$conf.high, m1c$table_body$p.value)[3:dim(m1c$table_body)[1],],3))
)
colnames(desc_table_item_pt2) = c("item", "or", "ci.lower", "ci.upper", "p.value")
item_summary <- merge(desc_table_item_pt1,desc_table_item_pt2,by="item")

# Display and extract descriptive table
item_summary
write_xlsx(item_summary,paste0(path_to_outputs, "tables/desc_table_item.xlsx"))


# Get descriptve table by program

# Get odds ratios and 95% CI for availability by item
m1 <- glm(available ~ program, data = item_desc_df, family = binomial) # for logistic reg results
m1c <- tbl_regression(m1, exponentiate = TRUE, conf.int = TRUE) # for OR and CI from logistic reg results
m1a <- desc_table_item_pt1 %>% tabyl('program') # for frequency stats

desc_table_program <- cbind(m1a[,1:3], m1b[1:dim(m1a),2], rbind(
  matrix("Ref", nrow = 1, ncol = 4),
  round(cbind(m1c$table_body$estimate, m1c$table_body$conf.low,
              m1c$table_body$conf.high, m1c$table_body$p.value)[2:dim(m1a)[1]+1,],2))
)

colnames(desc_table_program) = c("program", "n", "percent", "mean", "or", "ci.lower", "ci.upper", "p.value")

# Display and extract descriptive table
desc_table_program
write_xlsx(desc_table_program,paste0(path_to_outputs, "tables/desc_table_program.xlsx"))

# Get descriptve table by type of consumable
item_desc_df$item_drug <- relevel(factor(item_desc_df$item_drug), ref="0")
m1 <- glm(available ~ item_drug, data = item_desc_df, family = binomial) # for logistic reg results
m1c <- tbl_regression(m1, exponentiate = TRUE, conf.int = TRUE) # for OR and CI from logistic reg results
m1a <- desc_table_item_pt1 %>% tabyl('item_drug') # for frequency stats
m1b <- tidy(lm(available ~ item_drug - 1, data = item_desc_df))

desc_table_item_type <- cbind(m1a[,1:3], m1b[1:2,2], rbind(
  matrix("Ref", nrow = 1, ncol = 4),
  round(cbind(m1c$table_body$estimate, m1c$table_body$conf.low,
              m1c$table_body$conf.high, m1c$table_body$p.value),3))
)

colnames(desc_table_item_type) = c("consumable type", "n", "percent", "mean", "or", "ci.lower", "ci.upper", "p.value")

# Display and extract descriptive table
desc_table_item_type
write_xlsx(desc_table_item_type,paste0(path_to_outputs, "tables/desc_table_item_type.xlsx"))

# Get descriptive table by EML prioritization
item_desc_df$eml_priority_v <- relevel(factor(item_desc_df$eml_priority_v), ref="0")
m1 <- glm(available ~ eml_priority_v, data = item_desc_df, family = binomial) # for logistic reg results
m1c <- tbl_regression(m1, exponentiate = TRUE, conf.int = TRUE) # for OR and CI from logistic reg results
m1a <- desc_table_item_pt1 %>% tabyl('eml_priority_v') # for frequency stats
m1b <- tidy(lm(available ~ eml_priority_v - 1, data = item_desc_df))

desc_table_eml_priority <- cbind(m1a[,1:3], m1b[1:2,2], rbind(
  matrix("Ref", nrow = 1, ncol = 4),
  round(cbind(m1c$table_body$estimate, m1c$table_body$conf.low,
              m1c$table_body$conf.high, m1c$table_body$p.value),3))
)

colnames(desc_table_eml_priority) = c("EML classification", "n", "percent", "mean", "or", "ci.lower", "ci.upper", "p.value")

# Display and extract descriptive table
desc_table_eml_priority
write_xlsx(desc_table_eml_priority,paste0(path_to_outputs, "tables/desc_table_eml_priority.xlsx"))


# 3. Extract descriptive table by level of care for secondary analysis #
########################################################################
varlist_for_desc_by_level <- desc_table_varlist_full_bin[
  !(desc_table_varlist_full_bin %in% c('fac_type'))
]

datasets_by_level <- list(df_level1a, df_level1b, df_level2, df_level0)

i=0
for (level in 1:length(datasets_by_level)){
  print(paste("processing", datasets_by_level[[level]]$fac_type[1]))
  data <- datasets_by_level[[level]]
  # Create collapsed facility level dataset
  fac_features <- aggregate(data[unlist(c(fac_exp_vars, new_vars))], data[,'fac_code'], FUN = head, 1)
  availability_by_facility <- aggregate(data[,'available'], data[,'fac_code'], 
                                        FUN = mean, na.rm = TRUE)
  data_fac <- merge(fac_features,availability_by_facility,by="fac_code")
  data_fac <- na.omit(data_fac)
  
  print(paste("processing", datasets_by_level[[level]]$fac_type[1]))
  for (var in varlist_for_desc_by_level){
    #m1 <- lm(available ~ fac_type, data = df)
    print(paste("processing", var))
    
    # Create temporary dataset with just one variable at a time
    tempSubset <- data[,c("available", var)]
    tempSubset <- tempSubset[complete.cases(tempSubset),]
    
    # Convert binary variable to categorical
    if (is.numeric(tempSubset[[var]]) && all(unique(is.numeric(tempSubset[[var]])) %in% c(0, 1, NA, NaN))){
      tempSubset[[var]] <- ifelse(tempSubset[[var]]== 0, "No", ifelse(tempSubset[[var]] == 1, "Yes", tempSubset[[var]]))
    }
    
    m1 <- lm(available ~ ., data = tempSubset)
    
    m1a <- data_fac %>% tabyl(var)
    #m1b <- tidy(lm(available ~ var - 1, data = df))
    m1b <- tidy(lm(available ~ . - 1, data = tempSubset))
    
    m1c <- wald.test(Sigma = vcov(m1), b = coef(m1), Terms = 2:dim(m1a)[1])
    
    desc_table_var <- cbind(m1a[,1:3], m1b[1:dim(m1a),2], rbind(
      matrix("", nrow = dim(m1a)[1]-1 , ncol = 3),
      round(m1c$result$chi2,3)))
    # Note that m1b[1:dim(m1a),2] contains dim(m1a) becauase otherwise m1b includes a 
    # row for NaN
    
    names(desc_table_var)[1] <- "variable" 
    
    if (i == 0){
      desc_table <- desc_table_var
    }
    else{
      desc_table <- rbind(desc_table, desc_table_var)
    }
    i = 1
  }
  # Display and extract descriptive table
  desc_file = paste0(path_to_outputs, "tables/desc_table_", datasets_by_level[[i]]$fac_type[1] ,".txt")
  write.table(desc_table, file = desc_file, sep = ",", quote = FALSE, row.names = F)
}
pvals <- m1c$result$chi2[3]
print(m1c$result$chi2[3],digits=3.3)

# 4. Other descriptive tables for manuscript
##############################################
# Table of districts
district_table_all <-  df %>% 
  group_by(district) %>% 
  summarise(facilities = n_distinct(fac_code))
district_table_reg <-  df_for_fac_item_re_sorted %>% 
  group_by(district) %>% 
  summarise(facilities = n_distinct(fac_code))
district_table <- merge(district_table_all,district_table_reg,by="district")
write.csv(district_table,paste0(path_to_outputs, "tables/district_table.csv"), row.names = TRUE)

# Table of items
item_table <- df[c('program', 'item')]
item_table <- item_table[!duplicated(df$item),]
write.csv(item_table,paste0(path_to_outputs, "tables/item_table.csv"), row.names = TRUE)