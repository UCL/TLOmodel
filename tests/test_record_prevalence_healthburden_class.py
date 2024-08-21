import os
from pathlib import Path
import datetime
from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods.fullmodel import fullmodel

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
outputpath = Path("./outputs")

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 12)

popsize = 500
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
    max_date_in_prevalence = max(prevalence['date'])
    # HIV
    prevalence_HIV_function = prevalence['Hiv']
    prevalence_HIV_log = output['tlo.methods.hiv']["summary_inc_and_prev_for_adults_and_children_and_fsw"]["total_plhiv"] / \
                         output['tlo.methods.hiv']["summary_inc_and_prev_for_adults_and_children_and_fsw"]["pop_total"]

    # TB
    prevalence_tb_function = prevalence['Tb']
    print(prevalence_tb_function)
    prevalence_tb_log = output['tlo.methods.tb']["tb_prevalence"]

    closest_recording_log = None
    smallest_diff = None
    for j in range(len(prevalence_tb_log)):
        target_date = prevalence_tb_log['date'][j]
        regular_log_value = prevalence_tb_log["tbPrevActive"][j] + prevalence_tb_log["tbPrevLatent"][j]
        if target_date > max_date_in_prevalence:
            continue
        else:
            for i in range(len([prevalence])):
                function_date = prevalence['date'][i]
                date_diff = abs((target_date - function_date).days)
                if smallest_diff is None or date_diff < smallest_diff:
                    print("TB function", function_date)
                    smallest_diff = date_diff
                    closest_recording_log = prevalence['Tb'][i]
        print(closest_recording_log)
        print(regular_log_value)
        # assert regular_log_value == closest_recording_log

    # Oesophageal Cancer
    closest_recording_log = None
    smallest_diff = None
    prevalence_oesophageal_cancer_log = output['tlo.methods.oesophagealcancer']["summary_stats"]

    for j in range(len(prevalence_oesophageal_cancer_log)):
        target_date = prevalence_oesophageal_cancer_log['date'][j]
        regular_log_value = (
            prevalence_oesophageal_cancer_log["total_low_grade_dysplasia"][j] +
            prevalence_oesophageal_cancer_log["total_high_grade_dysplasia"][j] +
            prevalence_oesophageal_cancer_log["total_stage1"][j] +
            prevalence_oesophageal_cancer_log["total_stage2"][j] +
            prevalence_oesophageal_cancer_log["total_stage3"][j] +
            prevalence_oesophageal_cancer_log["total_stage4"][j]
        )
        if target_date > max_date_in_prevalence:
            continue
        else:
            for i in range(len([prevalence])):
                function_date = prevalence['date'][i]
                date_diff = abs((target_date - function_date).days)
                if smallest_diff is None or date_diff < smallest_diff:
                    print("OC function", function_date)
                    smallest_diff = date_diff
                    closest_recording_log = prevalence['OesophagealCancer'][i] * prevalence['population'][i]  # only record totals

        print("Log OC:", regular_log_value)
        print("Function OC:", closest_recording_log)
        assert regular_log_value == closest_recording_log

    # Other Adult Cancer
    closest_recording_log = None
    smallest_diff = None
    prevalence_oac_cancer_log = output['tlo.methods.other_adult_cancers']["summary_stats"]

    for j in range(len(prevalence_oac_cancer_log)):
        target_date = prevalence_oac_cancer_log['date'][j]
        regular_log_value = (
            prevalence_oac_cancer_log["total_site_confined"][j] +
            prevalence_oac_cancer_log["total_local_ln"][j] +
            prevalence_oac_cancer_log["total_metastatic"][j]
        )
        if target_date > max_date_in_prevalence:
            continue
        else:
            for i in range(len([prevalence])):
                function_date = prevalence['date'][i]
                date_diff = abs((target_date - function_date).days)
                if smallest_diff is None or date_diff < smallest_diff:
                    print("AOC function", function_date)
                    smallest_diff = date_diff
                    closest_recording_log = prevalence['OtherAdultCancer'][i] * prevalence['population'][i]  # only record totals

        print("Log OAC:", regular_log_value)
        print("Function OAC:", round(closest_recording_log,7))
        assert regular_log_value == closest_recording_log

    # Bladder Cancer
    closest_recording_log = None
    smallest_diff = None
    prevalence_bladder_cancer_log = output['tlo.methods.bladder_cancer']["summary_stats"]

    for j in range(len(prevalence_bladder_cancer_log)):
        target_date = prevalence_bladder_cancer_log['date'][j]
        regular_log_value = (
            prevalence_bladder_cancer_log["total_tis_t1"][j] +
            prevalence_bladder_cancer_log["total_t2p"][j] +
            prevalence_bladder_cancer_log["total_metastatic"][j]
        )
        if target_date > max_date_in_prevalence:
            continue
        else:
            for i in range(len([prevalence])):
                function_date = prevalence['date'][i]
                date_diff = abs((target_date - function_date).days)
                if smallest_diff is None or date_diff < smallest_diff:
                    print("Bladder function", function_date)
                    smallest_diff = date_diff
                    closest_recording_log = prevalence['BladderCancer'][i] * prevalence['population'][i]  # only record totals

        print("Log Bladder:", regular_log_value)
        print("Function Bladder:", round(closest_recording_log,7))
        assert regular_log_value == round(closest_recording_log,7)

    # Breast Cancer
    prevalence_breast_cancer_log = output['tlo.methods.breast_cancer']["summary_stats"]
    max_date_in_prevalence = max(prevalence['date'])  # Find the maximum date in prevalence['date']

    for j in range(len(prevalence_breast_cancer_log)):
        target_date = prevalence_breast_cancer_log['date'][j]
        regular_log_value = (
            prevalence_breast_cancer_log["total_stage1"][j] +
            prevalence_breast_cancer_log["total_stage2"][j] +
            prevalence_breast_cancer_log["total_stage3"][j] +
            prevalence_breast_cancer_log["total_stage4"][j]
        )

        if target_date > max_date_in_prevalence:
            continue
        else:
        # Reset smallest_diff and closest_recording_log for each target_date
            smallest_diff = None
            closest_recording_log = None
            for i in range(len(prevalence)):
                function_date = prevalence['date'][i]
                date_diff = abs((target_date - function_date).days)
                if smallest_diff is None or date_diff < smallest_diff:
                    smallest_diff = date_diff
                    closest_recording_log = prevalence['BreastCancer'][i] * prevalence['population'][i]
        print("Breast log date:", target_date)
        print("Closest function date:", function_date)
        print("Log Breast:", regular_log_value)
        assert regular_log_value == closest_recording_log

    # Prostate Cancer
    closest_recording_log = None
    smallest_diff = None
    prevalence_prostate_cancer_log = output['tlo.methods.prostate_cancer']["summary_stats"]

    for j in range(len(prevalence_prostate_cancer_log)):
        target_date = prevalence_prostate_cancer_log['date'][j]
        regular_log_value = (
            prevalence_prostate_cancer_log["total_prostate_confined"][j] +
            prevalence_prostate_cancer_log["total_local_ln"][j] +
            prevalence_prostate_cancer_log["total_metastatic"][j]
        )
        if target_date > max_date_in_prevalence:
            continue
        else:
            for i in range(len([prevalence])):
                function_date = prevalence['date'][i]
                date_diff = abs((target_date - function_date).days)
                if smallest_diff is None or date_diff < smallest_diff:
                    print("Prostate function", function_date)
                    smallest_diff = date_diff
                    closest_recording_log = prevalence['ProstateCancer'][i] * prevalence['population'][i]  # only record totals

        print("Log Prostate:", regular_log_value)
        print("Function Prostate:", round(closest_recording_log,7))
        assert regular_log_value == round(closest_recording_log,7)

    # Malaria - only clinical prevalence

    closest_recording_log = None
    smallest_diff = None
    prevalence_malaria_log = output['tlo.methods.malaria']["prevalence"]

    for j in range(len(prevalence_malaria_log)):
        target_date = prevalence_malaria_log['date'][j]
        regular_log_value = (
            prevalence_malaria_log["clinical_prev"][j]  # only records clinical prevalence
        )
        if target_date > max_date_in_prevalence:
            continue
        else:
            for i in range(len([prevalence])):
                function_date = prevalence['date'][i]
                date_diff = abs((target_date - function_date).days)
                if smallest_diff is None or date_diff < smallest_diff:
                    print("Malaria function", function_date)
                    smallest_diff = date_diff
                    closest_recording_log = prevalence['Malaria'][i] * prevalence['population'][i]  # only record totals

        print("Log Malaria:", regular_log_value)
        print("Function Malaria:", round(closest_recording_log,7))
        assert regular_log_value < round(closest_recording_log,7)

    # Maternal deaths
    maternal_deaths_function = prevalence['maternal_deaths']
    death_df = output['tlo.methods.demography']['death']
    properties_deceased = output['tlo.methods.demography.detail']["properties_of_deceased_persons"]
    direct_deaths = len(death_df[death_df['cause'] == 'Maternal Disorders'])
    indirect_deaths_non_hiv = len(properties_deceased.loc[
        (properties_deceased['is_pregnant'] | properties_deceased['la_is_postpartum']) &
        (properties_deceased['cause_of_death'].str.contains('Malaria|Suicide|ever_stroke|diabetes|chronic_ischemic_hd|ever_heart_attack|chronic_kidney_disease') |
         (properties_deceased['cause_of_death'] == 'TB'))])
    hiv_pd = len(properties_deceased.loc[
        (properties_deceased['is_pregnant'] | properties_deceased['la_is_postpartum']) &
        (properties_deceased['cause_of_death'].str.contains('AIDS_non_TB|AIDS_TB'))])

    hiv_indirect_maternal_deaths = hiv_pd * 0.3
    maternal_deaths_log = direct_deaths + indirect_deaths_non_hiv + hiv_indirect_maternal_deaths

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

    # Cardiometabolic disorders
    conditions = ['diabetes', 'hypertension', 'chronic_kidney_disease', 'chronic_lower_back_pain', 'chronic_ischemic_hd'] # logged only in adult population
    results = {}

    for condition in conditions:
        prevalence_function = prevalence[condition]
        closest_recording_log = None
        smallest_diff = None
        prevalence_log = output['tlo.methods.cardio_metabolic_disorders'][f"{condition}_prevalence"]
        if target_date > max_date_in_prevalence:
            continue
        else:
            for j in range(len(prevalence_log)):
                target_date = prevalence_breast_cancer_log['date'][j]
                regular_log_value = output['tlo.methods.cardio_metabolic_disorders'][f"{condition}_prevalence"]
                for i in range(len([prevalence])):
                    function_date = prevalence['date'][i]
                    date_diff = abs((target_date - function_date).days)
                    if smallest_diff is None or date_diff < smallest_diff:
                        print(f"{condition} function", function_date)
                        smallest_diff = date_diff
                        closest_recording_log = prevalence[condition][i] * prevalence['population'][
                            i]  # only record totals

            print(f"Log {condition}:", regular_log_value)
            print(f"Function {condition}:", closest_recording_log)
            assert regular_log_value == closest_recording_log

        results[f'{condition}_prevalence_function'] = prevalence_function
        results[f'{condition}_prevalence_log'] = prevalence_log

    results.update({
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
    })
    return results

results = test_run_with_healthburden_with_dummy_diseases(outputpath, seed)

for key, value in results.items():
    print(f"{key}:")
    print(value)

print("COPD", results['prevalence']["Copd"])
