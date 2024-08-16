import os
from pathlib import Path

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
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

    # HIV
    prevalence_HIV_function = prevalence['Hiv']
    prevalence_HIV_log = output['tlo.methods.hiv']["summary_inc_and_prev_for_adults_and_children_and_fsw"]["total_plhiv"] / \
                         output['tlo.methods.hiv']["summary_inc_and_prev_for_adults_and_children_and_fsw"]["pop_total"]

    # TB
    prevalence_tb_function = prevalence['Tb']
    prevalence_tb_log = output['tlo.methods.tb']["tb_prevalence"]["tbPrevActive"] + \
                        output['tlo.methods.tb']["tb_prevalence"]["tbPrevLatent"]

    # Malaria - only clinical prevalence
    prevalence_malaria_function = prevalence['Malaria']
    prevalence_malaria_log = output['tlo.methods.malaria']["prevalence"]["clinical_prev"]

    # Maternal deaths
    #maternal_deaths_function = prevalence['maternal_deaths']
    death_df = output['tlo.methods.demography']['death']
    #properties_deceased = output['tlo.methods.demography.detail']["properties_of_deceased_persons"]

   #direct_deaths = len(death_df[death_df['cause'] == 'Maternal Disorders'])
    #indirect_deaths_non_hiv = len(properties_deceased.loc[
    #    (properties_deceased['is_pregnant'] | properties_deceased['la_is_postpartum']) &
    #    (properties_deceased['cause_of_death'].str.contains('Malaria|Suicide|ever_stroke|diabetes|chronic_ischemic_hd|ever_heart_attack|chronic_kidney_disease') |
    #     (properties_deceased['cause_of_death'] == 'TB'))])
    #hiv_pd = len(properties_deceased.loc[
    #    (properties_deceased['is_pregnant'] | properties_deceased['la_is_postpartum']) &
    #    (properties_deceased['cause_of_death'].str.contains('AIDS_non_TB|AIDS_TB'))])

    #hiv_indirect_maternal_deaths = hiv_pd * 0.3
    #maternal_deaths_log = direct_deaths + indirect_deaths_non_hiv + hiv_indirect_maternal_deaths

    # Newborn deaths
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

    # Stillbirths
    intrapartum_stillbirths_function = prevalence['Intrapartum stillbirth']
    antenatal_stillbirths_function = prevalence['Antenatal stillbirth']

    #intrapartum_stillbirths_log = (
    #    output['tlo.methods.labour']['intrapartum_stillbirth']
    #    .assign(year=output['date'].dt.year)
    #    .groupby(['year'])#['year']
    #    .count()
    #)
    #antenatal_stillbirths_log = (
    #    output['tlo.methods.pregnancy_supervisor']['antenatal_stillbirth']
    #    .assign(year=output['date'].dt.year)
    #    .groupby(['year'])#['year']
    #    .count()
    #)

    # Cardiometablis disorders
    diabetes_prevalence_function = prevalence['diabetes']
    diabetes_prevalence_log = output['tlo.methods.cardio_metabolic_disorders']["diabetes_prevalence"]

    hypertension_prevalence_function = prevalence['hypertension']
    hypertension_prevalence_log = output['tlo.methods.cardio_metabolic_disorders']["hypertension_prevalence"]
    results = {
        'prevalence_newborn_deaths_log': prevalence_newborn_deaths_log,
        'prevalence_newborn_deaths_function': prevalence_newborn_deaths_function,
        'prevalence': prevalence,
        'prevalence_tb_function': prevalence_tb_function,
        'prevalence_tb_log': prevalence_tb_log,
        'prevalence_HIV_function': prevalence_HIV_function,
        'prevalence_HIV_log': prevalence_HIV_log,
        'prevalence_malaria_function': prevalence_malaria_function,
        'prevalence_malaria_log': prevalence_malaria_log,
        'intrapartum_stillbirths_function': intrapartum_stillbirths_function,
        #'intrapartum_stillbirths_log': intrapartum_stillbirths_log,
        'antenatal_stillbirths_function': antenatal_stillbirths_function,
        #'antenatal_stillbirths_log': antenatal_stillbirths_log,
        'maternal_deaths_log': maternal_deaths_log,
        'maternal_deaths_function': maternal_deaths_function
    }

    return results

results = test_run_with_healthburden_with_dummy_diseases(outputpath, seed)

for key, value in results.items():
    print(f"{key}:")
    print(value)

print("COPD", results['prevalence']["Copd"])
