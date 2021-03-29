from pathlib import Path

import numpy as np

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    antenatal_care,
    contraception,
    demography,
    dx_algorithm_adult,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    newborn_outcomes,
    pregnancy_supervisor,
    rti,
    symptommanager,
)

# =============================== Analysis description ========================================================
# This analysis file has essentially become the model fitting analysis, seeing what happens when we run the model
# and whether an ordinary model run will behave how we would expect it to, hitting the right demographics, producing
# the right injuries, measuring the percent of crashes involving alcohol

# ============================================== Model run ============================================================
log_config = {
    "filename": "rti_health_system_comparison",  # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.rti": logging.INFO,
        "tlo.methods.healthsystem": logging.DEBUG,
        "tlo.methods.labour": logging.disable(logging.DEBUG)
    }
}
# The Resource files [NB. Working directory must be set to the root of TLO: TLOmodel]
resourcefilepath = Path('./resources')
# Establish the simulation object
yearsrun = 2
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
service_availability = ['*']
pop_size = 50000
nsim = 2
# Create a variable whether to save figures or not (used in debugging)
save_figures = True
# Prior to using Azure batches I have used for loops to handle longer model runs. To get the information I need
# from the simulations, I store results in lists. As this anaylsis file is essentially handling the model fitting so
# with my current system I have many lists, there is probably a better way to do this.
# Create lists to store information from each simulation in
# Age demographics
sim_age_range = []
sim_male_age_range = []
sim_female_age_range = []
# Gender demographics
females = []
males = []
# Alcohol consumption demographics
percents_attributable_to_alcohol = []
# Health outcomes
list_number_of_crashes = []
list_number_of_disabilities = []
number_of_deaths_pre_hospital = []
number_of_deaths_in_hospital = []
number_of_deaths_no_med = []
number_of_deaths_unavailable_med = []
# Incidences
incidences_of_rti = []
incidences_of_rti_in_children = []
incidences_of_death = []
incidences_of_death_pre_hospital = []
incidences_of_death_post_med = []
incidences_of_death_no_med = []
incidences_of_death_unavailable_med = []
incidences_of_rti_yearly_average = []
incidences_of_death_yearly_average = []
incidences_of_injuries = []
inc_amputations = []
inc_burns = []
inc_fractures = []
inc_tbi = []
inc_sci = []
inc_minor = []
inc_other = []
tot_inc_injuries = []
# Percentages of the context of deaths
ps_of_imm_death = []
ps_of_death_post_med = []
ps_of_death_without_med = []
ps_of_death_unavailable_med = []
percent_died_after_med = []
# Overall percentage of fatal crashes
percent_of_fatal_crashes = []
# Injury severity
perc_mild = []
perc_severe = []
iss_scores = []
# Injury level data
number_of_injured_body_locations = []
inj_loc_data = []
inj_cat_data = []
per_injury_fatal = []
number_of_injuries_per_sim = []
# Flows between model states
rti_model_flow_summary = []
# Health seeking behaviour
percent_sought_healthcare = []
# Admitted to ICU or HDU
percent_admitted_to_icu_or_hdu = []
# Inpatient days
all_sim_inpatient_days = []
# Per sim inpatient days
per_sim_inpatient_days = []
# Number of consumables used
list_consumables_dict = []
# Overall healthsystem time usage
health_system_time_usage = []
# number of major surgeries
per_sim_major_surg = []
# number of minor surgeries
per_sim_minor_surg = []
# number of fractures cast
per_sim_frac_cast = []
# number of lacerations stitched
per_sim_laceration = []
# number of burns managed
per_sim_burn_treated = []
# number of tetanus vaccine administered
per_sim_tetanus = []
# number of pain medicine requested
per_sim_pain_med = []
# number of open fracture appointments used
per_sim_open_frac = []
# Deaths in 2010
deaths_2010 = []
# Injuries in 2010
injuries_in_2010 = []
# Number of injuries per year
injuries_per_year = []
# Number of prehospital deaths in 2010
number_of_prehospital_deaths_2010 = []
# ICU injury characteristics
ICU_frac = []
ICU_dis = []
ICU_tbi = []
ICU_soft = []
ICU_int_o = []
ICU_int_b = []
ICU_sci = []
ICU_amp = []
ICU_eye = []
ICU_lac = []
ICU_burn = []
# injury severity of rural vs urban injuries
per_sim_rural_severe = []
per_sim_urban_severe = []
# proportion of lower extremity fractures that are open
per_sim_average_percentage_lx_open = []

# Iterate over the number of simulations nsim
for i in range(0, nsim):
    # Create the simulation object
    sim = Simulation(start_date=start_date)
    # Register the modules
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
        dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath),
        contraception.Contraception(resourcefilepath=resourcefilepath),
        labour.Labour(resourcefilepath=resourcefilepath),
        newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
        pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
        antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
    )
    # Get the log file
    logfile = sim.configure_logging(filename="LogFile")
    # create and run the simulation
    sim.make_initial_population(n=pop_size)
    # alter the number of injuries given out
    sim.modules['RTI'].parameters['number_of_injured_body_regions_distribution'] = [
        [1, 2, 3, 4, 5, 6, 7, 8], [1, 0.0, 0.0, 0.0, 0.00, 0.0, 0.0, 0.0]
    ]
    sim.modules['RTI'].parameters['base_rate_injrti'] = \
        sim.modules['RTI'].parameters['base_rate_injrti'] * 6.9
    sim.modules['RTI'].parameters['imm_death_proportion_rti'] = \
        sim.modules['RTI'].parameters['imm_death_proportion_rti'] * 0.1
    # Run the simulation
    sim.simulate(end_date=end_date)
    # Parse the logfile of this simulation
    log_df = parse_log_file(logfile)
    # Store the incidence of RTI per 100,000 person years in this sim
    incidences_of_rti.append(log_df['tlo.methods.rti']['summary_1m']['incidence of rti per 100,000'].tolist())
    # Store the incidence of death due to RTI per 100,000 person years and the sub categories in this sim
    incidences_of_death.append(log_df['tlo.methods.rti']['summary_1m']['incidence of rti death per 100,000'].tolist())
    incidences_of_death_pre_hospital.append(
        log_df['tlo.methods.rti']['summary_1m']['incidence of prehospital death per 100,000'].tolist()
    )
print(f"Mean incidence of rti = {np.mean(incidences_of_rti)}")
print(f"Mean incidence of rti death = {np.mean(incidences_of_death)}")
print(f"Mean incidence of pre-hospital mortality = {np.mean(incidences_of_death_pre_hospital)}")
gbd2019nRTI = 180632.22
gbd2019nRTIDeath = 2077.15
