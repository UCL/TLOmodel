from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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
    postnatal_supervisor,
    rti,
    symptommanager,
)

# =============================== Analysis description ========================================================
# This script looks at the effect of reducing the number of lower extremity fractures that are treated
# using skeletal traction. Initially we look at inpatient days consumed in each simulation but eventually I want
# to expand this to the health benefits of not using skeletal traction to treat fractures

log_config = {
    "filename": "rti_health_system_comparison",  # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.rti": logging.INFO,
        "tlo.methods.healthsystem": logging.DEBUG
    }
}
# The Resource files [NB. Working directory must be set to the root of TLO: TLOmodel]
resourcefilepath = Path('./resources')
# Establish the simulation object
yearsrun = 2
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
pop_size = 50000

nsim = 2
scenarios = {'Current': 1,
             '25%': 0.75,
             '50%': 0.5,
             '75%': 0.25,
             '100%': 0
             }
# Get the parameters
params = pd.read_excel(Path(resourcefilepath) / 'ResourceFile_RTI.xlsx', sheet_name='parameter_values')
# Get the origional proportions of injuries treated with skeletal traction and other methods
# ======== tibia/fibula fracture treatment plan ====================================
orig_tib_fib_frac_cast = float(params.loc[params.parameter_name == 'prob_tib_fib_frac_require_cast', 'value'].values)
orig_tib_fib_maj_surg = float(params.loc[params.parameter_name == 'prob_tib_fib_frac_require_maj_surg', 'value'].values)
orig_tib_fib_min_surg = float(params.loc[params.parameter_name == 'prob_tib_fib_frac_require_min_surg', 'value'].values)
orig_tib_fib_amp = float(params.loc[params.parameter_name == 'prob_tib_fib_frac_require_amp', 'value'].values)
orig_tib_fib_traction = float(params.loc[params.parameter_name == 'prob_tib_fib_frac_require_traction', 'value'].values)
orig_tib_fib_other_options = 1 - orig_tib_fib_traction

# ============ Femur fracture treatment plan ========================================
orig_femur_frac_cast = float(params.loc[params.parameter_name == 'prob_femural_fracture_require_cast', 'value'].values)
orig_femur_maj_surg = \
    float(params.loc[params.parameter_name == 'prob_femural_fracture_require_major_surgery', 'value'].values)
orig_femur_min_surg = \
    float(params.loc[params.parameter_name == 'prob_femural_fracture_require_minor_surgery', 'value'].values)
orig_femur_amp = float(params.loc[params.parameter_name == 'prob_femural_fracture_require_amputation', 'value'].values)
orig_femur_traction = \
    float(params.loc[params.parameter_name == 'prob_femural_fracture_require_traction', 'value'].values)
orig_femur_other_options = 1 - orig_femur_traction
# =============== Pelvis fracture treatment plan ===================================
orig_pelvis_maj_surg = \
    float(params.loc[params.parameter_name == 'prob_pelvis_frac_major_surgery', 'value'].values)
orig_pelvis_min_surg = \
    float(params.loc[params.parameter_name == 'prob_pelvis_frac_minor_surgery', 'value'].values)
orig_pelvis_traction = \
    float(params.loc[params.parameter_name == 'prob_pelvis_fracture_traction', 'value'].values)
orig_pelvis_other_options = 1 - orig_pelvis_traction
orig_pelvis_cast = \
    float(params.loc[params.parameter_name == 'prob_pelvis_frac_cast', 'value'].values)
# =============== dislocated hip treatment plan ===============================================
orig_dis_hip_major_surgery = \
    float(params.loc[params.parameter_name == 'prob_dis_hip_require_maj_surg', 'value'].values)
orig_dis_hip_cast = \
    float(params.loc[params.parameter_name == 'prob_dis_hip_require_cast', 'value'].values)
orig_dis_hip_traction = \
    float(params.loc[params.parameter_name == 'prob_hip_dis_require_traction', 'value'].values)
orig_hip_dis_other_options = 1 - orig_dis_hip_traction
# Create empty lists to store the deaths, DALYs and inpatient days in each simulation
per_scenario_death = []
per_scenario_dalys = []
per_scenario_inpatient_days = []
for scenario_reduction in scenarios.values():
    # set up parameters for tibia fibula fracture treatment, what i'm doing here is getting the origional
    # proportion of tib/fib fractures treated by skeletal traction and reducing that proportion, and then
    # increasing the proportion of tib/fib fractures treated by other methods
    scenario_tib_fib_skeletal_traction = orig_tib_fib_traction * scenario_reduction
    scaler_for_rest_of_tib_fib_options = (1 / orig_tib_fib_other_options) * (1 - scenario_tib_fib_skeletal_traction)
    scenario_tib_fib_other = scaler_for_rest_of_tib_fib_options * orig_tib_fib_other_options
    # make sure that every tibia/fibula fracture has a treatment plan assigned to it
    assert np.round(scenario_tib_fib_skeletal_traction + scenario_tib_fib_other, 6) == 1, 'scaling did not work'
    # set up parameters for femur fractures, what i'm doing here is getting the origional
    # proportion of femur fractures treated by skeletal traction and reducing that proportion, and then
    # increasing the proportion of femur fractures treated by other methods
    scenario_femur_skeletal_traction = orig_femur_traction * scenario_reduction
    scaler_for_rest_of_femur_options = (1 / orig_femur_other_options) * (1 - scenario_femur_skeletal_traction)
    scenario_femur_other = scaler_for_rest_of_femur_options * orig_femur_other_options
    # make sure that every femur fracture has a treatment plan assigned to it
    assert np.round(scenario_femur_skeletal_traction + scenario_femur_other, 6) == 1, 'scaling did not work'
    # set up parameters for pelvis fractures, what i'm doing here is getting the origional
    # proportion of pelvis fractures treated by skeletal traction and reducing that proportion, and then
    # increasing the proportion of pelvis fractures treated by other methods
    scenario_pelvis_skeletal_traction = orig_pelvis_traction * scenario_reduction
    scaler_for_rest_of_pelvis_options = (1 / orig_pelvis_other_options) * (1 - scenario_pelvis_skeletal_traction)
    scenario_pelvis_other = scaler_for_rest_of_pelvis_options * orig_pelvis_other_options
    # make sure that every pelvis fracture has a treatment plan assigned to it
    assert np.round(scenario_pelvis_skeletal_traction + scenario_pelvis_other, 6) == 1, 'scaling did not work'
    # set up parameters for hip dislocations fractures, what i'm doing here is getting the origional
    # proportion of hip dislocationstreated by skeletal traction and reducing that proportion, and then
    # increasing the proportion of hip dislocations treated by other methods
    scenario_hip_dis_skeletal_traction = orig_dis_hip_traction * scenario_reduction
    scaler_for_rest_of_hip_dis_options = (1 / orig_hip_dis_other_options) * (1 - scenario_hip_dis_skeletal_traction)
    scenario_hip_dis_other = scaler_for_rest_of_hip_dis_options * orig_hip_dis_other_options
    # make sure that every hip dislocations has a treatment plan assigned to it
    assert np.round(scenario_hip_dis_skeletal_traction + scenario_hip_dis_other, 6) == 1, 'scaling did not work'
    # create empty lists to store the deaths, dalys and inpatient days found in this scenario
    in_scenario_deaths = []
    in_scenario_dalys = []
    average_inpatient_days_from_scenario = []
    for i in range(0, nsim):
        # create variable to count inpatient days used in sim
        total_inpatient_days_this_sim = 0
        # create the simulation object
        sim = Simulation(start_date=start_date)
        # We register all modules in a single call to the register method, calling once with multiple
        # objects. This is preferred to registering each module in multiple calls because we will be
        # able to handle dependencies if modules are registered together
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
            postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
            antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
        )
        # name the logfile
        logfile = sim.configure_logging(filename="LogFile")
        # make initial population
        sim.make_initial_population(n=pop_size)
        # Reduce the percentage of tibia fibula fractures treated with skeletal traction
        params['prob_tib_fib_frac_require_traction'] = scenario_tib_fib_skeletal_traction
        # Increase the percentage of tibia/fibula fractures treated with other methods
        params['prob_tib_fib_frac_require_maj_surg'] = orig_tib_fib_maj_surg * scaler_for_rest_of_tib_fib_options
        params['prob_tib_fib_frac_require_min_surg'] = orig_tib_fib_min_surg * scaler_for_rest_of_tib_fib_options
        params['prob_tib_fib_frac_require_amp'] = orig_tib_fib_amp * scaler_for_rest_of_tib_fib_options
        # Reduce the percentage of femur fractures treated with skeletal traction
        params['prob_femural_fracture_require_traction'] = scenario_femur_skeletal_traction
        # Increase the percentage of femur fractures treated with other methods
        params['prob_femural_fracture_require_cast'] = orig_femur_frac_cast * scaler_for_rest_of_femur_options
        params['prob_femural_fracture_require_major_surgery'] = orig_femur_maj_surg * scaler_for_rest_of_femur_options
        params['prob_femural_fracture_require_minor_surgery'] = orig_femur_min_surg * scaler_for_rest_of_femur_options
        params['prob_femural_fracture_require_amputation'] = orig_femur_amp * scaler_for_rest_of_femur_options
        # Reduce the percentage of pelvis fractures treated with skeletal traction
        params['prob_pelvis_fracture_traction'] = scenario_pelvis_skeletal_traction
        # Increase the percentage of pelvis fracture treated with other methods
        params['prob_pelvis_frac_major_surgery'] = orig_pelvis_maj_surg * scaler_for_rest_of_pelvis_options
        params['prob_pelvis_frac_minor_surgery'] = orig_pelvis_min_surg * scaler_for_rest_of_pelvis_options
        params['prob_pelvis_frac_cast'] = orig_pelvis_cast * scaler_for_rest_of_pelvis_options
        # Reduce the percentage of hip dislocations treated with skeletal traction
        params['prob_hip_dis_require_traction'] = scenario_hip_dis_skeletal_traction
        # increase the percentage of hip dislocations treated with other methods
        params['prob_dis_hip_require_cast'] = orig_dis_hip_cast * scaler_for_rest_of_hip_dis_options
        params['prob_dis_hip_require_maj_surg'] = orig_dis_hip_major_surgery * scaler_for_rest_of_hip_dis_options

        # Run the simulation
        sim.simulate(end_date=end_date)
        # parse the logfile
        log_df = parse_log_file(logfile)
        # get the number of road traffic injury related deaths from the sim
        rti_deaths = log_df['tlo.methods.demography']['death']
        rti_causes_of_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        # calculate the total number of rti related deaths
        tot_rti_deaths = len(rti_deaths.loc[rti_deaths['cause'].isin(rti_causes_of_deaths)])
        # store the number of rti related deaths
        in_scenario_deaths.append(tot_rti_deaths)
        # get the dalys produced from the sim
        dalys_df = log_df['tlo.methods.healthburden']['dalys']
        # get male daly data
        males_data = dalys_df.loc[dalys_df['sex'] == 'M']
        # get yll in males
        YLL_males_data = males_data.filter(like='YLL_RTI').columns
        # calculate male dalys
        males_dalys = males_data[YLL_males_data].sum(axis=1) + males_data['YLD_RTI_rt_disability']
        # get female daly data
        females_data = dalys_df.loc[dalys_df['sex'] == 'F']
        # get female YLL
        YLL_females_data = females_data.filter(like='YLL_RTI').columns
        # calculate female dalys
        females_dalys = females_data[YLL_females_data].sum(axis=1) + females_data['YLD_RTI_rt_disability']
        # calculate total dalys
        tot_dalys = males_dalys.tolist() + females_dalys.tolist()
        # store total dalys
        in_scenario_dalys.append(sum(tot_dalys))
        # get health system usage df
        inpatient_day_df = log_df['tlo.methods.healthsystem']['HSI_Event'].loc[
            log_df['tlo.methods.healthsystem']['HSI_Event']['TREATMENT_ID'] == 'RTI_MedicalIntervention']
        for person in inpatient_day_df.index:
            # Get the number of inpatient days per person, if there is a key error when trying to access inpatient days
            # it means that this patient didn't require any so append (0)
            try:
                total_inpatient_days_this_sim += \
                    inpatient_day_df.loc[person, 'Number_By_Appt_Type_Code']['InpatientDays']
            except KeyError:
                total_inpatient_days_this_sim += 0
        # Get the number of consumables used in this sim
        average_inpatient_days_from_scenario.append(total_inpatient_days_this_sim)
    # store the deaths, dalys and inpatient day usage in this scenario run
    per_scenario_death.append(in_scenario_deaths)
    per_scenario_dalys.append(in_scenario_dalys)
    per_scenario_inpatient_days.append(np.mean(average_inpatient_days_from_scenario))
# calculate the average number of deaths in each scenario
average_deaths = [np.mean(death_list) for death_list in per_scenario_death]
# calculate the average number of dalys in each scenario
average_tot_dalys = [np.mean(daly_list) for daly_list in per_scenario_dalys]
# calculate the average total inpatient days in each scenario
average_tot_inpatient_days = [np.mean(day_list) for day_list in per_scenario_inpatient_days]
# calculate the percentage reduction in deaths in each scenario
percent_deaths_reduction = \
    [((deaths - average_deaths[0]) / average_deaths[0]) * 100 for deaths in average_deaths if average_deaths[0] != 0]
# calcualte the percentage reduction in dalys in each scenario
percent_dalys_reduction = [((daly - average_tot_dalys[0]) / average_tot_dalys[0]) * 100 for daly in average_tot_dalys]
# calculate the percentage reduction in inpatient day usage
percent_inpatient_day_reduction = \
    [((days - average_tot_inpatient_days[0]) / average_tot_inpatient_days[0]) * 100 for days in
     average_tot_inpatient_days]
w = 0.8 / len(scenarios)
# plot the resulting change in deaths, dalyus and inpatient days in a bar chart
plt.bar(np.arange(len(scenarios)), percent_deaths_reduction, color='lightsteelblue', width=w,
        label='% change in deaths')
plt.bar(np.arange(len(scenarios)) + w, percent_dalys_reduction, color='lightsalmon', width=w,
        label='% change in dalys')
plt.bar(np.arange(len(scenarios)) + 2 * w, percent_inpatient_day_reduction, color='olive', width=w,
        label='% change in inpatient days')
plt.xticks(np.arange(len(scenarios)) + w, scenarios.keys())
plt.legend()
plt.ylabel('Percent')
plt.xlabel('Reduction')
plt.title(f"The percent change of average deaths, DALYS and inpatient days"
          f"\n"
          f"due to reduced use of skeletal traction"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/ReducingSkeletalTraction/ReducingSkeletalTraction.png', bbox_inches='tight')
