# This script generates tables and graphs based on the results of 
# 4_regression_analysis.R
# In the future, we might want to load the regression outputs instead of re-running 
# the regression analysis script. 

###########################################################
# 1. Data setup
###########################################################
# 1.1 Run previous setup files
#---------------------------------
source(paste0(path_to_scripts, "4_regression_analysis.R"))

###########################################################
# 2. Tables and Graphs
###########################################################
# 2.1 Table 3: Forest plot of main regression results
#######################################################
# Create the combined plot for manuscript
#-----------------------------------------------------------
load(paste0(path_to_outputs, "regression_results/model_fac_item_re.rdta"))
filename =  paste0(path_to_outputs, "figures/main_regression_plot.png")
custom_forest_plot(model = model_fac_item_re) #, xlab = "Odds ratio with 95% Confidence Interval"
png(filename, units="in", width=8, height=5, res=300)
cowplot::plot_grid(data_table,p, align = "h", rel_widths = c(0.65, 0.35))
dev.off()

# 2.2 Table 4: Forest plot of subgroup analysis
#######################################################
# 2.2.1 Sub-group analysis by level
#----------------------------------------
load(paste0(path_to_outputs, "regression_results/sub-group/model_level.rdta"))
fac_types_for_reg <- c('Level 1a', 'Level 1b')
custom_forest_plot_by_level(model = model_level_summaries[[1]], xlab = fac_types_for_reg[1], ylimit = 16)
p_level1 <- p1
custom_forest_plot_by_level(model = model_level_summaries[[2]], xlab = fac_types_for_reg[2], ylimit = 16)
p_level2 <- p

filename =  paste0(path_to_outputs, "figures/regression_plot_bylevel.png")
png(filename, units="in", width=8, height=5, res=300)
plot <- cowplot::plot_grid(p_level1,p_level2, align = "h", nrow = 1, rel_widths = c(0.68, 0.32))
x.grob <- textGrob("                                                                              Odds Ratio with 95% Confidence Interval", 
                   gp=gpar(fontface="bold", col="black", fontsize=8))
grid.arrange(arrangeGrob(plot, bottom = x.grob))
dev.off()

# 2.2.2.1. Sub-group analysis by program (Consumable random effects)
#---------------------------------------------------------------------
load(paste0(path_to_outputs, "regression_results/sub-group/model_program.rdta"))

programs_for_reg <- c('General', 'HIV', 'Malaria', 'Tuberculosis', 'Obstetric and\n Newborn care',
                      'Contraception','Child Health', 'EPI','ALRI',
                      'NCD',  'Surgical')
custom_forest_plot_by_level(model = model_prog_summaries[[1]], xlab = programs_for_reg[1], ylimit = 8)
p_prog1 <- p1
for (i in 2:10){
  print(programs_for_reg[i])
  custom_forest_plot_by_level(model = model_prog_summaries[[i]], xlab = programs_for_reg[i], ylimit = 8)
  name <- paste("p_prog", i, sep = "")
  assign(name, p)
}
custom_forest_plot_by_level(model = model_prog_summaries[[6]], xlab = programs_for_reg[6], ylimit = 8)
p_prog6 <- p1

filename =  paste0(path_to_outputs, "figures/regression_plot_byprog1.png")
png(filename, units="in", width=8, height=5, res=300)
plot <- cowplot::plot_grid(p_prog1,p_prog2,p_prog3,p_prog4, p_prog5,
                           align = "h", nrow = 1, rel_widths = c(0.45, 0.125,0.125, 0.125, 0.125))
x.grob <- textGrob("                                                                              Odds Ratio with 95% Confidence Interval", 
                   gp=gpar(fontface="bold", col="black", fontsize=8))
grid.arrange(arrangeGrob(plot, bottom = x.grob))
dev.off()

filename =  paste0(path_to_outputs, "figures/regression_plot_byprog2.png")
png(filename, units="in", width=8, height=5, res=300)
plot <- cowplot::plot_grid(p_prog6,p_prog7,p_prog8,p_prog9, p_prog10,
                           align = "h", nrow = 1, rel_widths = c(0.45, 0.125,0.125, 0.125, 0.125))
x.grob <- textGrob("                                                                              Odds Ratio with 95% Confidence Interval", 
                   gp=gpar(fontface="bold", col="black", fontsize=8))
grid.arrange(arrangeGrob(plot, bottom = x.grob))
dev.off()

# 2.2.2.2 Sub-group analysis by program (Consumable fixed effects)
#------------------------------------------------------------------
load(paste0(path_to_outputs, "regression_results/sub-group/model_program_itemfe.rdta"))

programs_for_reg <- c('General', 'HIV', 
                      'Malaria', 'Tuberculosis', 
                      'Obstetric and Newborn care','Contraception',
                      'Child Health', 'EPI',
                      'ALRI','NCD',
                      'Surgical')

ref_item_list <- c('Paracetamol cap/tab','Antiretroviral treatment (ART) component 1 (ZDV/AZT/TDF/D4T)',
                   'Artemisinin monotherapy (oral)', 'Ethambutol',
                   'Albendazole or Mebendazole', 'Atropine (injection)',
                   'Ciprofloxacin', 'BCG vaccine',
                   'Amoxicillin', 'Angiotensin-converting enzyme(ACE) inhibitor',
                   'Absorbable suture material')

for (i in 1:10){
  print(programs_for_reg[i])
  custom_forest_plot_by_level_itemfe(model = model_prog_summaries_itemfe[[i]], ylimit = 16,
                                     ref_item = ref_item_list[i])
  
  filename <<- paste( paste0(path_to_outputs, "figures/regression_plot_prog_"),programs_for_reg[i], ".png")
  png(filename, units="in", width=8, height=5, res=300)
  prog_plot <- cowplot::plot_grid(data_table,p, align = "h", rel_widths = c(0.65, 0.35))
  print(prog_plot)
  dev.off()
}

# 2.2.3 Sub-group analysis by owner
#----------------------------------------
load(paste0(path_to_outputs, "regression_results/sub-group/model_owner.rdta"))

fac_owners_for_reg <- c('Government', 'CHAM', 'NGO', 'Private for profit')
custom_forest_plot_by_level(model = model_owner_summaries[[1]], xlab = fac_owners_for_reg[1])
p_owner1 <- p1
for (i in 2:4){
  print(fac_owners_for_reg[i])
  custom_forest_plot_by_level(model = model_owner_summaries[[i]], xlab = fac_owners_for_reg[i])
  name <- paste("p_owner", i, sep = "")
  assign(name, p)
}

filename =  paste0(path_to_outputs, "figures/regression_plot_byowner.png")
png(filename, units="in", width=8, height=5, res=300)
plot <- cowplot::plot_grid(p_owner1,p_owner2, p_owner3, p_owner4, 
                           align = "h", nrow = 1, rel_widths = c(0.52, 0.16, 0.16, 0.16))
x.grob <- textGrob("                                                                              Odds Ratio with 95% Confidence Interval", 
                   gp=gpar(fontface="bold", col="black", fontsize=8))
grid.arrange(arrangeGrob(plot, bottom = x.grob))
dev.off()

# 2.2.4 Sub-group analysis by consumable type
#----------------------------------------
load(paste0(path_to_outputs, "regression_results/sub-group/model_item_type.rdta"))

item_types_for_reg <- c('Drugs', 'Other consumables')
custom_forest_plot_by_level(model = model_item_type_summaries[[1]], xlab = item_types_for_reg[1])
p_item_type1 <- p1
custom_forest_plot_by_level(model = model_item_type_summaries[[2]], xlab = item_types_for_reg[2])
p_item_type2 <- p


filename =  paste0(path_to_outputs, "figures/regression_plot_byitem_type.png")
png(filename, units="in", width=8, height=5, res=300)
plot <- cowplot::plot_grid(p_item_type1,p_item_type2, align = "h", nrow = 1, rel_widths = c(0.68,0.32))
x.grob <- textGrob("                                                                              Odds Ratio with 95% Confidence Interval", 
                   gp=gpar(fontface="bold", col="black", fontsize=8))
grid.arrange(arrangeGrob(plot, bottom = x.grob))
dev.off()

# 2.2.4 Sub-group analysis by consumable priority
#----------------------------------------
load(paste0(path_to_outputs, "regression_results/sub-group/model_item_priority.rdta"))

item_priority_for_reg <- c('Vital', 'Other')
custom_forest_plot_by_level(model = model_item_priority_summaries[[1]], xlab = item_priority_for_reg[1])
p_item_priority1 <- p1
custom_forest_plot_by_level(model = model_item_priority_summaries[[2]], xlab = item_priority_for_reg[2])
p_item_priority2 <- p


filename =  paste0(path_to_outputs, "figures/regression_plot_by_item_priority.png")
png(filename, units="in", width=8, height=5, res=300)
plot <- cowplot::plot_grid(p_item_priority1,p_item_priority2, align = "h", nrow = 1, rel_widths = c(0.68,0.32))
x.grob <- textGrob("                                                                              Odds Ratio with 95% Confidence Interval", 
                   gp=gpar(fontface="bold", col="black", fontsize=8))
grid.arrange(arrangeGrob(plot, bottom = x.grob))
dev.off()

# 2.2.5 Sub-group analysis by consumable availability group
#-----------------------------------------------------------
load(paste0(path_to_outputs, "regression_results/sub-group/model_item_availability_group.rdta"))

item_availability_groups_for_reg <- c('Availability >= 10%', 'Availability < 10%')
custom_forest_plot_by_level(model = model_item_availability_group_summaries[[1]], xlab = item_availability_groups_for_reg[1])
p_item_availability1 <- p1
custom_forest_plot_by_level(model = model_item_availability_group_summaries[[2]], xlab = item_availability_groups_for_reg[2])
p_item_availability2 <- p


filename =  paste0(path_to_outputs, "figures/regression_plot_by_item_availability_group.png")
png(filename, units="in", width=8, height=5, res=300)
plot <- cowplot::plot_grid(p_item_availability1,p_item_availability2, align = "h", nrow = 1, rel_widths = c(0.68,0.32))
x.grob <- textGrob("                                                                              Odds Ratio with 95% Confidence Interval", 
                   gp=gpar(fontface="bold", col="black", fontsize=8))
grid.arrange(arrangeGrob(plot, bottom = x.grob))
dev.off()