# This script loads the data for descriptive and inferential analysis

# 1. Load HHFA data
####################
df_orig <- read.csv(paste0(path_to_data, "cleaned_hhfa_2019.csv"))

# 1.1 Assign a code to items
#----------------------------
df_orig <- df_orig %>%
  group_by(item) %>%
  mutate(item_code = cur_group_id()) %>%
  ungroup()

# 2. Filter data to be ignored
###############################
second_line_arvs <- c("Didanosine (DDI) (ARVs)",
                     "Enfuvirtide (T-20) (ARVs)",
                     "Delavirdine (DLV) (ARVs)")
df <- df_orig %>%
  filter(service_consumable_stock == 1, # Facilities which do not store consumables
         program != 'mental health/epilepsy', # mental health commodities are rare at this level of care
         fac_daily_opening_hours != 0, # Facilities which reporting 0 opening hours per day
         !(item %in% second_line_arvs) # Second line ARVs
  )

# Formatting
df$fac_code <- as.character(df$fac_code)

# 3. Add consumable classification in the Essential Medicines List #
#####################################################################
essential_med_mapping <- read.csv(paste0(path_to_data, "essential_medicine_list_mapping.csv"))
names(essential_med_mapping)[names(essential_med_mapping) == 'Consumable'] <- 'item'
names(essential_med_mapping)[names(essential_med_mapping) == 'Therapeutic.priority'] <- 'eml_therapeutic_priority'

df <- merge(df, essential_med_mapping[c('item','eml_therapeutic_priority')], by = 'item')
# Generate binary variable
df$eml_priority_v <- 0
df[which(df$eml_therapeutic_priority == "V"),]$eml_priority_v <- 1
df$eml_priority_e <- 0
df[which(df$eml_therapeutic_priority == "E"),]$eml_priority_e <- 1

# 4. Define variable lists for analysis #
#########################################
fac_exp_vars <- c(# Main characteristics
  'district', 'fac_type','fac_owner', 'fac_urban', 'rms', 'item_drug',

  # types of services offered
  'outpatient_only','bed_count', 'inpatient_visit_count','inpatient_days_count',
  'service_fp','service_anc','service_pmtct','service_delivery','service_pnc','service_epi','service_imci',
  'service_hiv','service_tb','service_othersti','service_malaria','service_blood_transfusion',
  'service_cvd', 'service_diagnostic',

  # operation frequency
  'fac_weekly_opening_days','fac_daily_opening_hours',

  # utilities/facilities available
  'functional_landline','fuctional_mobile','functional_radio','functional_computer','functional_computer_no',
  'internet_access_today',
  'electricity', 'water_source_main','water_source_main_within_500m', # this is in minutes
  'functional_toilet','functional_handwashing_facility',
  'water_disruption_last_3mts','water_disruption_duration', # converted to days

  # vehicles available
  'functional_emergency_vehicle','accessible_emergency_vehicle','purpose_last_vehicle_trip',

  'functional_ambulance', 'functional_ambulance_no', # Keep one of the two variables
  'functional_car', 'functional_car_no',
  'functional_motor_cycle', 'functional_motor_cycle_no',
  'functional_bike_ambulance', 'functional_bike_ambulance_no',
  'functional_bicycle', 'functional_bicycle_no',
  'vaccine_storage','functional_refrigerator',

  # Drug ordering process
  'incharge_drug_orders','drug_resupply_calculation_system','drug_resupply_calculation_method','source_drugs_cmst',
  'source_drugs_local_warehouse','source_drugs_ngo','source_drugs_donor','source_drugs_pvt',

  'drug_transport_local_supplier','drug_transport_higher_level_supplier','drug_transport_self','drug_transport_other',

  'drug_order_fulfilment_delay','drug_order_fulfilment_freq_last_3mts',
  'transport_to_district_hq','travel_time_to_district_hq',

  # referral system
  'referral_system_from_community','referrals_to_other_facs',

  # location w.r.t reference points
  'dist_todh', 'dist_torms', 'drivetime_todh', 'drivetime_torms')

# Continuous variables
fac_vars_numeric <- c('bed_count', 'inpatient_visit_count','inpatient_days_count', 'functional_computer_no',
                     'water_disruption_duration',
                     'functional_ambulance_no', 'functional_car_no', 'functional_motor_cycle_no', 'functional_bike_ambulance_no',
                     'functional_bicycle_no',
                     'fac_weekly_opening_days','fac_daily_opening_hours',
                     'drug_order_fulfilment_delay','drug_order_fulfilment_freq_last_3mts',
                     'travel_time_to_district_hq', 'dist_todh', 'dist_torms', 'drivetime_todh', 'drivetime_torms')

# Binary variables
fac_vars_binary <- c('outpatient_only', 'fac_urban',
                    'service_fp','service_anc','service_pmtct','service_delivery','service_pnc','service_epi','service_imci',
                    'service_hiv','service_tb','service_othersti','service_malaria','service_blood_transfusion',
                    'service_cvd', 'service_diagnostic',
                    'functional_landline','fuctional_mobile','functional_radio','functional_computer','internet_access_today',
                    'electricity', 'functional_toilet','functional_handwashing_facility',
                    'water_disruption_last_3mts',
                    'functional_emergency_vehicle','accessible_emergency_vehicle',
                    'functional_ambulance',
                    'functional_car',
                    'functional_motor_cycle',
                    'functional_bike_ambulance',
                    'functional_bicycle',
                    'vaccine_storage','functional_refrigerator',
                    'source_drugs_cmst','source_drugs_local_warehouse','source_drugs_ngo','source_drugs_donor','source_drugs_pvt',
                    'drug_transport_local_supplier','drug_transport_higher_level_supplier','drug_transport_self','drug_transport_other',
                    'referral_system_from_community','referrals_to_other_facs')

item_vars_binary <- c('item_drug', 'eml_priority_v', 'eml_priority_e')

# Determinants as per literature
stockout_factors_from_lit <- c('fac_type', 'fac_owner', 'fac_urban','functional_computer',
                               'functional_emergency_vehicle', 'incharge_drug_orders',
                               'dist_todh','dist_torms','drug_order_fulfilment_freq_last_3mts',
                               'service_diagnostic', 'item_drug','eml_priority_v')
# drug_resupply_calculation_system - this variable has been dropped since it's not clear whether this was  accurately reported

# Look at data by item
df_not_na <- subset(df, !is.na(df$available))
df_by_item <- df %>%
  group_by(item) %>%
  summarise(available = mean(available))

df_by_fac <- df_not_na %>%
  group_by(fac_code) %>%
  summarise(available = mean(available),
            drug_resupply_calculation_system = first(na.omit(drug_resupply_calculation_system)),
            fac_owner = first(na.omit(fac_owner)),
            functional_bike_ambulance_no = first(na.omit(functional_bike_ambulance_no)))
