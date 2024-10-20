# This script performs feature manipulation on the cleaned HHFA dataset

# 1. Feature manipulation #
##########################################
# 1.0 Clean binary variables
#---------------------------------------
# Convert item_type variable to binary
df["item_drug"] <- 0
df[which(df$item_type == "drug"),]$item_drug <- 1

# 1.1 Cap variables/remove outliers
#----------------------------------
df$bed_count <- edit_outliers(df$bed_count, "upper")
df$inpatient_visit_count <- edit_outliers(df$inpatient_visit_count, "upper")
df$drug_order_fulfilment_delay <- edit_outliers(df$drug_order_fulfilment_delay, "upper")
df$drug_order_fulfilment_freq_last_3mts <- edit_outliers(df$drug_order_fulfilment_freq_last_3mts, "upper")
df$water_disruption_duration <- edit_outliers(df$water_disruption_duration, "upper")
df$dist_todh <- edit_outliers(df$dist_todh, "upper")
df$dist_torms <- edit_outliers(df$dist_torms, "upper")

df$functional_ambulance_no <- edit_outliers(df$functional_ambulance_no, "upper")
df$functional_car_no <- edit_outliers(df$functional_car_no, "upper")
df$functional_bicycle_no <- edit_outliers(df$functional_bicycle_no, "upper")
df$functional_motor_cycle_no <- edit_outliers(df$functional_motor_cycle_no, "upper")


# 1.2 Rescale continuous variables
#----------------------------------
df <- df %>% dplyr::rename(
  dist_todh_orig = dist_todh,
  dist_torms_orig = dist_torms,
  fac_daily_opening_hours_orig = fac_daily_opening_hours,
  bed_count_orig = bed_count,
  inpatient_visit_count_orig = inpatient_visit_count,
  inpatient_days_count_orig = inpatient_days_count,
  drug_order_fulfilment_delay_orig = drug_order_fulfilment_delay,
  drug_order_fulfilment_freq_last_3mts_orig = drug_order_fulfilment_freq_last_3mts,
  water_disruption_duration_orig = water_disruption_duration,
  functional_bicycle_no_orig = functional_bicycle_no,
  functional_motor_cycle_no_orig = functional_motor_cycle_no
)

df$dist_todh <- log(df$dist_todh_orig + 1)
df$dist_torms <- log(df$dist_torms_orig + 1)
df$fac_daily_opening_hours <- log(df$fac_daily_opening_hours_orig)
df$bed_count <- log(df$bed_count_orig + 1)
df$inpatient_visit_count <- log(df$inpatient_visit_count_orig + 1)
df$inpatient_days_count <- log(df$inpatient_days_count_orig + 1)
df$drug_order_fulfilment_delay <- log(df$drug_order_fulfilment_delay_orig + 1)
df$drug_order_fulfilment_freq_last_3mts <- log(df$drug_order_fulfilment_freq_last_3mts_orig +1)
df$water_disruption_duration <- log(df$water_disruption_duration_orig +1)

df$functional_bicycle_no <- log(df$functional_bicycle_no_orig +1)
df$functional_motor_cycle_no <- log(df$functional_motor_cycle_no_orig +1)


# 1.3 Streamline categorical variables
#-------------------------------------
# Incharge of drug orders
df[which(df$incharge_drug_orders == "pharmacy technician"|df$incharge_drug_orders == "pharmacist"),]$incharge_drug_orders <-
  'pharmacist or pharmacy technician'

# Water source
df[which(df$water_source_main == "other"|df$water_source_main == "no water source"),]$water_source_main <-
  'no convenient water source'

# Generate categorical version of dist_rorms
df$dist_torms_cat <- ""
df$dist_torms_cat <- ifelse((df$dist_torms_orig > 10000) & (df$dist_torms_orig <= 50000), "10-50 kms",
                            ifelse((df$dist_torms_orig > 50000) & (df$dist_torms_orig < 100000),"50-100 kms",
                                   ifelse((df$dist_torms_orig > 100000) & (df$dist_torms_orig < 200000),"100-200 kms",
                                          ifelse((df$dist_torms_orig < 10000), "0-10 kms","> 200 kms"))
                            ))

df$dist_torms_cat <- ifelse((is.na(df$dist_torms_orig)), NaN, df$dist_torms_cat)
df$dist_torms_cat <- factor(df$dist_torms_cat, levels = c("0-10 kms", "10-50 kms", "50-100 kms", "100-200 kms", "> 200 kms")) # specify order

# Generate categorical version of dist_todh
df$dist_todh_cat <- ifelse((df$dist_todh_orig > 10000) & (df$dist_todh_orig <= 25000), "10-25 kms",
                           ifelse((df$dist_todh_orig > 25000) & (df$dist_todh_orig < 50000),"25-50 kms",
                                  ifelse((df$dist_todh_orig > 50000) & (df$dist_todh_orig < 75000),"50-75 kms",
                                         ifelse((df$dist_todh_orig < 10000), "0-10 kms","> 75 kms"))
                           ))

df$dist_todh_cat <- ifelse((is.na(df$dist_todh_orig)), NaN, df$dist_todh_cat)
df$dist_todh_cat <- factor(df$dist_todh_cat, levels = c("0-10 kms", "10-25 kms", "25-50 kms", "50-75 kms", "> 75 kms")) # specify order

# Generate categorical version of drug_order_fulfilment_freq_last_3mts
df$drug_order_fulfilment_freq_last_3mts_cat <- as.character(df$drug_order_fulfilment_freq_last_3mts_orig)
df$drug_order_fulfilment_freq_last_3mts_cat <- ifelse((df$drug_order_fulfilment_freq_last_3mts_orig > 3), ">= 4",
                                                      df$drug_order_fulfilment_freq_last_3mts_cat)

df$drug_order_fulfilment_freq_last_3mts_cat <- factor(df$drug_order_fulfilment_freq_last_3mts_cat, levels = c("1", "2", "3", ">= 4")) # specify order


# Drug transport
df$drug_transport_self <- ifelse(is.na(df$drug_transport_self), NaN, df$drug_transport_self)
df$drug_transport_higher_level_supplier <- ifelse(is.na(df$drug_transport_higher_level_supplier), NaN, df$drug_transport_higher_level_supplier)
df$drug_transport_local_supplier <- ifelse(is.na(df$drug_transport_local_supplier), NaN, df$drug_transport_local_supplier)
df$drug_transport_other <- ifelse(is.na(df$drug_transport_other), NaN, df$drug_transport_other)


# 1.4 Create a joint drug storage practice variable from individual ones
# for oxytocin and amoxicillin
#------------------------------------------------------------------------
# Clean variable
df$label_and_expdate_visible <- ifelse((df$mathealth_label_and_expdate_visible == 1) &
                                         (df$childhealth_label_and_expdate_visible == 1), 1,
                                       ifelse((df$mathealth_label_and_expdate_visible == 0) &
                                                (df$childhealth_label_and_expdate_visible == 0),3,2))

df$expdate_fefo <- ifelse((df$mathealth_expdate_fefo == 1) &
                            (df$childhealth_expdate_fefo == 1), 1,
                          ifelse((df$mathealth_expdate_fefo == 0) &
                                   (df$childhealth_expdate_fefo == 0),3,2))

df$label_and_expdate_visible <- factor(df$label_and_expdate_visible)
df$expdate_fefo <- factor(df$expdate_fefo)

# 1.5 Clean classification of programs into items
#------------------------------------------------
df[which(df$item == "Oral hypoglycaemics"),]$program <- "ncds"
df[which(df$program == "hygiene/antiseptic"),]$program <- "other - infection prevention"
df[which(df$item == "Disposable latex gloves"),]$program <- "other - infection prevention"
df[which(df$item == "Medical (surgical or procedural) masks"),]$program <- "other - infection prevention"
df[which(df$item == "Eye protection (goggles, face shields)"),]$program <- "other - infection prevention"
df[which(df$item == "Gowns"),]$program <- "other - infection prevention"
df[which(df$program == "other"),]$program <- "general"
df[which(df$item == "Simvastatin tablet or other statin"),]$program <- "ncds"
df[which(df$item == "Cryptococcal antigen"),]$program <- "hiv"
df[which(df$item == "Ephedrine (injection)"),]$program <- "surgical"
df[which(df$item == "Fluconazole"),]$program <- "hiv"

# 1.6 Clean consumable names for manuscript
#-------------------------------------------------------------
df[which(df$item == "slides and cover slips"),]$item <- "Slides and cover slips"
df[which(df$item == "art_component_1"),]$item <- "Antiretroviral treatment (ART) component 1 (ZDV/AZT/TDF/D4T)"
df[which(df$item == "art_component_2"),]$item <- "Antiretroviral treatment (ART) component 2 (3TC/FTC)"
df[which(df$item == "art_component_3"),]$item <- "Antiretroviral treatment (ART) component 3 (Protease inhibitor)"
df[which(df$item == "ACT"),]$item <- "Artemisinin-based combination therapy (ACT)"
df[which(df$item == "MRDT"),]$item <- "Malaria Rapid Diagnostic Test (MRDT)"
df[which(df$item == "SP"),]$item <- "Sulfadoxine/pyrimethamine"
df[which(df$item == "glucose inhibitors"),]$item <- "Glucose inhibitor"
df[which(df$item == "ACE inhibitor"),]$item <- "Angiotensin-converting enzyme(ACE) inhibitor"

# 1.7 Clean level of care
#-------------------------------------------
#df[which(df$item == "Facility_level_1a"),]$item = "Primary level"
#df[which(df$item == "Facility_level_1b"),]$item = "Secondary level"


# 2. Create final dataframes for analysis
##############################################################
# Dataframes for sub-level analysis
df_level0 <- df[df$fac_type == 'Facility_level_0',]
df_level1a <- df[df$fac_type == 'Facility_level_1a',]
df_level1b <- df[df$fac_type == 'Facility_level_1b',]
df_level2 <- df[df$fac_type == 'Facility_level_2',]
df_level3 <- df[df$fac_type == 'Facility_level_2',]


# Set reference level for categorical variables
df$incharge_drug_orders <- relevel(factor(df$incharge_drug_orders), ref="drug store clerk")
df$district <- relevel(factor(df$district), ref="Lilongwe")
df$item <- relevel(factor(df$item), ref="Paracetamol cap/tab")
df$fac_owner <- relevel(factor(df$fac_owner), ref="Government")
df$fac_type <- relevel(factor(df$fac_type), ref="Facility_level_1a")
df$water_source_main <- relevel(factor(df$water_source_main), ref="piped into facility")
df$program <- relevel(factor(df$program), ref="general")
df$dist_torms_cat <- relevel(factor(df$dist_torms_cat), ref = "0-10 kms")
df$dist_todh_cat <- relevel(factor(df$dist_todh_cat), ref = "0-10 kms")
df$drug_order_fulfilment_freq_last_3mts_cat <- relevel(factor(df$drug_order_fulfilment_freq_last_3mts_cat), ref = "1")

df_all_levels <- df

# Dataframe for primary analysis - only levels 1a and 1b
df <- df %>%
  filter(fac_type != 'Facility_level_4', # Zomba Mental Hospital
         fac_type != 'Facility_level_3', # Central Hospitals
         fac_type != 'Facility_level_2', # District Hospitals
         fac_type != 'Facility_level_0', # Health posts
  )
# fac_types 2 and 3 are always urban and always government owned
# fac_type 0 has only 3 instances of non-government ownership, never provides cvd,
# never has a vehicle

# 3 Create facility-level dataframe with the above adjustments
###############################################################
# For the dataframe with facilities in levels 1a and 1b
new_vars <- c('dist_torms_cat', 'dist_todh_cat', 'drug_order_fulfilment_freq_last_3mts_cat' )
fac_features <- aggregate(df[unlist(c(fac_exp_vars, new_vars))], by = list(fac_code = df$fac_code), FUN = head, 1)
availability_by_facility <- aggregate( df[,'available'], list(fac_code = df$fac_code),
                                       FUN = mean, na.rm = TRUE)
fac_reg_df <- merge(fac_features,availability_by_facility,by="fac_code")
fac_reg_df <- na.omit(fac_reg_df)

# For the dataframe with all facilities
fac_features_all_levels <- aggregate(df_all_levels[unlist(c(fac_exp_vars, new_vars))], list(fac_code = df_all_levels$fac_code),  FUN = head, 1)
availability_by_facility_all_levels <- aggregate( df_all_levels[,'available'], list(fac_code = df_all_levels$fac_code),
                                                  FUN = mean, na.rm = TRUE)
fac_reg_df_all_levels <- merge(fac_features_all_levels,availability_by_facility_all_levels,by="fac_code")
fac_reg_df_all_levels <- na.omit(fac_reg_df_all_levels)
