from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    dx_algorithm_adult,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    rti,
    symptommanager,
)

# =============================== Analysis description ========================================================
# What I am doing here is artificially reducing the proportion of pre-hospital mortality, increasing the number of
# people funneled into the injured sub-population, who will have to subsequently have to seek health care. At the moment
# I have only included a range of reduction of pre-hospital mortality, but when I get around to focusing on this, I will
# model a reasonable level of pre-hospital mortality reduction.
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
yearsrun = 10
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
pop_size = 20000
nsim = 2
# Set service availability
service_availability = ["*"]
# create lists to store deaths and dalys for each level of reduction in prehospital mortality
list_deaths_average = []
list_tot_dalys_average = []
# create np array for the percentage reduction in prehospital mortality
prehosital_mortality_reduction = np.linspace(1, 0.8, 3)
# get the parameters
params = pd.read_excel(Path(resourcefilepath) / 'ResourceFile_RTI.xlsx', sheet_name='parameter_values')
# get origional value for prehospital mortality
orig_prehospital_mortality = float(params.loc[params.parameter_name == 'imm_death_proportion_rti', 'value'].values)
extrapolated_deaths = []
extrapolated_dalys = []
for i in range(0, nsim):
    # create empty lists to store number of deaths and dalys in
    list_deaths = []
    list_tot_dalys = []
    list_extrapolated_deaths = []
    list_extrapolated_dalys = []
    for reduction in prehosital_mortality_reduction:
        sim = Simulation(start_date=start_date)
        # register modules
        sim.register(
            demography.Demography(resourcefilepath=resourcefilepath),
            enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
            healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_availability),
            symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
            dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
            dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
            healthburden.HealthBurden(resourcefilepath=resourcefilepath),
            rti.RTI(resourcefilepath=resourcefilepath)
        )
        # name the logfile
        logfile = sim.configure_logging(filename="LogFile_Reducing_Prehospital_Mortality",
                                        directory="./outputs/reducing_prehospital_mortality")
        # create initial population
        sim.make_initial_population(n=pop_size)
        # reduce prehospital mortality
        params = sim.modules['RTI'].parameters
        params['imm_death_proportion_rti'] = orig_prehospital_mortality * reduction
        # Run the simulation
        sim.simulate(end_date=end_date)
        # parse the logfile
        log_df = parse_log_file(logfile)
        # get the number of road traffic injury related deaths from the sim
        Data = pd.read_excel(
            Path(resourcefilepath) / "ResourceFile_DemographicData.xlsx",
            sheet_name="Interpolated Pop Structure",
        )
        sim_start_year = sim.start_date.year
        sim_end_year = sim.date.year
        sim_year_range = pd.Index(np.arange(sim_start_year, sim_end_year))
        Data_Pop = Data.groupby(by="year")["value"].sum()
        Data_Pop = Data_Pop.loc[sim_year_range]
        log_df['tlo.methods.demography']['population'].index = log_df['tlo.methods.demography']['population']['date']
        log_df['tlo.methods.demography']['population'].index = log_df['tlo.methods.demography']['population'].index.year
        model_pop_size = log_df['tlo.methods.demography']['population']['total']
        scaling_df = pd.DataFrame(model_pop_size)
        scaling_df['pred_pop_size'] = Data_Pop
        scaling_df['scale_for_each_year'] = scaling_df['pred_pop_size'] / scaling_df['total']
        rti_deaths = log_df['tlo.methods.demography']['death']
        rti_causes_of_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        # calculate the total number of rti related deaths
        # find deaths caused by RTI
        rti_deaths = rti_deaths.loc[rti_deaths['cause'].isin(rti_causes_of_deaths)]
        # create a column to show the year deaths occurred in
        rti_deaths['year'] = rti_deaths['date'].dt.year.to_list()
        # group by the year and count how many deaths ocurred
        rti_deaths = rti_deaths.groupby('year').count()
        # calculate extrapolated number of deaths
        rti_deaths['estimated_n_deaths'] = rti_deaths['cause'] * scaling_df['scale_for_each_year']
        # store the extrapolated number of deaths over the course of the sim
        list_extrapolated_deaths.append(rti_deaths['estimated_n_deaths'].sum())
        # calculate the total number of rti deaths that occurred in the sim
        tot_rti_deaths = rti_deaths['cause'].sum()
        # store the number of deaths in this sim
        list_deaths.append(tot_rti_deaths)
        # Get the DALYs from the sim
        dalys_df = log_df['tlo.methods.healthburden']['dalys']
        # group the dalys by year
        dalys_df = dalys_df.groupby('year').sum()
        # get the YLL caused by RTI
        dalys_df_RTI_YLL = dalys_df.filter(like='YLL_RTI').columns
        # calculate dalys caused by rti
        dalys_df['dalys'] = dalys_df[dalys_df_RTI_YLL].sum(axis=1) + dalys_df['YLD_RTI_rt_disability']
        # get the dalys occurring in each full year of the simulation
        dalys_df = dalys_df.loc[scaling_df.index]
        # extrapolate the total number of dalys caused in the model
        dalys_df['estimated_n_dalys'] = dalys_df['dalys'] * scaling_df['scale_for_each_year']
        # store the extrapolated number of dalys
        list_extrapolated_dalys.append(dalys_df['estimated_n_dalys'].sum())
        # calculate total dalys that occurred in the simulation
        tot_dalys = dalys_df['dalys'].sum()
        # store total dalys in scenario
        list_tot_dalys.append(tot_dalys)
    # Store the deaths and DALYs from the sim
    list_deaths_average.append(list_deaths)
    list_tot_dalys_average.append(list_tot_dalys)
    extrapolated_dalys.append(list_extrapolated_dalys)
    extrapolated_deaths.append(list_extrapolated_deaths)

# Get the average deaths per reduction of pre-hospital mortality
average_deaths = [float(sum(col)) / len(col) for col in zip(*list_deaths_average)]
# get the average extrapolated deaths per reduction of prehospital mortality
average_extrapolated_deaths = [float(sum(col)) / len(col) for col in zip(*extrapolated_deaths)]
# Get the average DALYs per reduction of pre-hospital mortality
average_tot_dalys = [float(sum(col)) / len(col) for col in zip(*list_tot_dalys_average)]
# get the average extrapolated dalys per reduction of prehospital mortality
average_extrapolated_dalys = [float(sum(col)) / len(col) for col in zip(*extrapolated_dalys)]
# Create the percentage reduction in deaths
mortality_reduction = [1 - deaths / average_deaths[0] for deaths in average_deaths]
mortality_reduction = np.multiply(mortality_reduction, 100)
# Create the percentage reduction in dalys
disability_reduction =  [1 - dalys / average_tot_dalys[0] for dalys in average_tot_dalys]
disability_reduction = np.multiply(disability_reduction, 100)

# Create the xtick labels
xtick_labels = [f"{np.round(1 - reduction, 2) * 100}%" for reduction in prehosital_mortality_reduction]
# plot sim data in a bar chart
plt.bar(np.arange(len(average_deaths)), average_deaths, color='lightsteelblue', width=0.25, label='Deaths')
plt.bar(np.arange(len(average_deaths)) + 0.25, average_tot_dalys, color='lightsalmon', width=0.25, label='DALYs')
plt.ylabel('Deaths/DALYs')
plt.xticks(np.arange(len(average_deaths)), xtick_labels, rotation=45)
plt.legend()
plt.title(f"The effect of reducing pre-hospital mortality on average Deaths/DALYS"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/PrehospitalMortality/'
            'PrehospitalMortality_vs_deaths_DALYS_in_sim.png', bbox_inches='tight')
plt.clf()
plt.bar(np.arange(len(mortality_reduction)), mortality_reduction, color='lightsteelblue', width=0.25, label='Deaths')
plt.bar(np.arange(len(disability_reduction)) + 0.25, disability_reduction, color='lightsalmon', width=0.25,
        label='DALYs')
plt.ylabel('Percent Change in deaths and DALYs')
plt.xticks(np.arange(len(average_deaths)), xtick_labels, rotation=45)
plt.legend()
plt.title(f"The percent reduction of average Deaths/DALYS when reduction pre-hospital mortality"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/PrehospitalMortality/'
            'PrehospitalMortality_vs_deaths_DALYS_in_sim_percent_reduction.png', bbox_inches='tight')
plt.clf()
plt.bar(np.arange(len(average_extrapolated_dalys)), average_extrapolated_deaths, color='lightsteelblue',
        width=0.25, label='Deaths')
plt.bar(np.arange(len(average_extrapolated_dalys)) + 0.25, average_extrapolated_dalys, color='lightsalmon', width=0.25,
        label='DALYs')
plt.ylabel('Deaths and DALYs')
plt.xticks(np.arange(len(average_deaths)), xtick_labels, rotation=45)
plt.legend()
plt.title(f"The percent reduction of average Deaths/DALYS when reduction pre-hospital mortality"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/PrehospitalMortality/'
            'PrehospitalMortality_vs_deaths_DALYS_in_sim_extrapolated.png', bbox_inches='tight')
