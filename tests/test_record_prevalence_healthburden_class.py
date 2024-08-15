import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from tlo import Date, Simulation
from tlo.analysis.utils import extract_results, parse_log_file
from tlo.methods.fullmodel import fullmodel

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
outputpath = Path("./outputs")

start_date = Date(2010, 1, 1)
end_date = Date(2011, 1, 12)
popsize = 1000
seed = 42

def check_dtypes(simulation):
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()

def test_run_with_healthburden_with_dummy_diseases(tmpdir, seed):
    """Check that everything runs in the simple cases of Mockitis and Chronic Syndrome and that outputs are as expected."""

    sim = Simulation(start_date=start_date, seed=seed, log_config={'filename': 'test_log', 'directory': outputpath})

    sim.register(*fullmodel(
        resourcefilepath=resourcefilepath,
        use_simplified_births=False,
    ))

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    output = parse_log_file(sim.log_filepath)
    prevalence = output['tlo.methods.healthburden']['prevalence_of_diseases']


    ## HIV
    prevalence_HIV_function = prevalence['Hiv']
    prevalence_HIV_log = output['tlo.methods.hiv']["summary_inc_and_prev_for_adults_and_children_and_fsw"]["total_plhiv"]/output['tlo.methods.hiv']["summary_inc_and_prev_for_adults_and_children_and_fsw"]["pop_total"]

    assert prevalence_HIV_function[1] == prevalence_HIV_log[0] # the first entry in HIV function is before the regular logger has recorded anything
    ## TB
    prevalence_tb_function = prevalence['Tb']
    prevalence_tb_log = output['tlo.methods.tb']["tb_prevalence"]["tbPrevActive"] + \
                        output['tlo.methods.tb']["tb_prevalence"]["tbPrevLatent"]
    assert prevalence_tb_function[1] == prevalence_tb_log[0] # the first entry in TB function is before the regular logger has recorded anything

    ## Malaria - only clinical prevalence
    prevalence_malaria_function = prevalence['Malaria']
    prevalence_malaria_log = output['tlo.methods.malaria']["prevalence"]["clinical_prev"]
    assert prevalence_malaria_function[1] != prevalence_malaria_log[0] # the first entry in malaria function is before the regular logger has recorded anything

    ## Maternal deaths
    maternal_deaths_function = prevalence['maternal_deaths']
    death_df = output['tlo.methods.demography']['death']

    direct_deaths = len(death_df[death_df['cause'] == 'Maternal Disorders'])
    properties_deceased = output['tlo.methods.demography.detail']["properties_of_deceased_persons"]
    indirect_deaths_non_hiv = len(properties_deceased.loc[(properties_deceased['is_pregnant'] | properties_deceased['la_is_postpartum']) &
                              (properties_deceased['cause_of_death'].str.contains('Malaria|Suicide|ever_stroke|diabetes|'
                                                                 'chronic_ischemic_hd|ever_heart_attack|'
                                                                 'chronic_kidney_disease') |
                               (properties_deceased['cause_of_death'] == 'TB'))])
    hiv_pd = len(properties_deceased.loc[(properties_deceased['is_pregnant'] | properties_deceased['la_is_postpartum']) &
                              (properties_deceased['cause_of_death'].str.contains('AIDS_non_TB|AIDS_TB'))])

    hiv_indirect_maternal_deaths = hiv_pd * 0.3

    maternal_deaths_log = direct_deaths + indirect_deaths_non_hiv + hiv_indirect_maternal_deaths

    prevalence_newborn_deaths_function = prevalence['newborn_deaths']
    prevalence_newborn_deaths_log = (
        properties_deceased[
            (properties_deceased['age_days'] < 29) &
            (properties_deceased['age_years'] == 0) &
            (~properties_deceased['is_alive'])
            ]
        .assign(year=properties_deceased['date'].dt.month)
        .groupby('year')
        .size()
    )

    return prevalence_newborn_deaths_log, prevalence_newborn_deaths_function, prevalence, prevalence_tb_log, prevalence_HIV_function, prevalence_HIV_log, prevalence_malaria_function, prevalence_malaria_log

prevalence_newborn_deaths_log, prevalence_newborn_deaths_function, prevalence, prevalence_tb_log, prevalence_HIV_function, prevalence_HIV_log, prevalence_malaria_function, prevalence_malaria_log = test_run_with_healthburden_with_dummy_diseases(outputpath, seed)
print("prevalence_newborn_deaths_log", prevalence_newborn_deaths_log)
print("prevalence_newborn_deaths_function", prevalence_newborn_deaths_function)
print("prevalence", prevalence)
print("prevalence_tb_function", prevalence['Tb'])
print("prevalence_tb_log", prevalence_tb_log)
print("prevalence_HIV_function", prevalence_HIV_function)
print("prevalence_HIV_log", prevalence_HIV_log)
print("prevalence_malaria_function", prevalence_malaria_function)
print("prevalence_malaria_log", prevalence_malaria_log)

