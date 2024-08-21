import os
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import create_pickles_locally, extract_results, parse_log_file
from tlo.methods.fullmodel import fullmodel

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
outputpath = Path("./outputs/test_record_prevalence/0/0")

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 12)

popsize = 1000
seed = 42
tolerance_percentage = 0.15 # attempt to account for differences in recording times
do_sim = True
tolerance_days =  10
def check_dtypes(simulation):
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()

def test_run_with_healthburden_with_dummy_diseases(tmpdir, seed):
    """Check that everything runs in the simple cases of Mockitis and Chronic Syndrome and that outputs are as expected."""

    sim = Simulation(start_date=start_date, seed=seed, log_config={'filename': 'test_log', 'directory': outputpath})
    sim.register(*fullmodel(
            resourcefilepath=resourcefilepath,
            use_simplified_births=False,))
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)
    output = parse_log_file(sim.log_filepath)

    prevalence = output['tlo.methods.healthburden']['prevalence_of_diseases']
    max_date_in_prevalence = max(prevalence['date'])
    # HIV
    # HIV
    prevalence_HIV_log = output['tlo.methods.hiv']['summary_inc_and_prev_for_adults_and_children_and_fsw']

    for j in range(len(prevalence_HIV_log)):
        target_date = prevalence_HIV_log['date'][j]
        regular_log_value = prevalence_HIV_log["total_plhiv"][j] / prevalence_HIV_log["pop_total"][j]

        if target_date > max_date_in_prevalence:
            continue
        else:
            closest_recording_log = None
            smallest_diff = float('inf')  # Initialize with a large number

            for i in range(len(prevalence)):
                function_date = prevalence['date'][i]
                date_diff = abs((target_date - function_date).days)

                if date_diff < smallest_diff and date_diff < tolerance_days:
                    smallest_diff = date_diff
                    closest_recording_log = prevalence['Hiv'][i]

            if smallest_diff < tolerance_days and closest_recording_log is not None:
                # Assert statement for validation
                assert abs(regular_log_value - closest_recording_log) < tolerance_percentage * regular_log_value

    # TB

    prevalence_tb_log = output['tlo.methods.tb']["tb_prevalence"]

    for j in range(len(prevalence_tb_log)):
        target_date = prevalence_tb_log['date'][j]
        regular_log_value = prevalence_tb_log["tbPrevActive"][j] + prevalence_tb_log["tbPrevLatent"][j]

        if target_date > max_date_in_prevalence:
            continue
        else:
            closest_recording_log = None
            smallest_diff = float('inf')  # Initialize with a large number

            for i in range(len(prevalence)):
                function_date = prevalence['date'][i]
                date_diff = abs((target_date - function_date).days)

                if date_diff < smallest_diff:
                    smallest_diff = date_diff
                    closest_recording_log = prevalence['Tb'][i]
                    closest_date = function_date
            # Check if the closest date difference is within 20 days
            if smallest_diff < tolerance_days and closest_recording_log is not None:
                print(closest_date)
                print(target_date)
                # Assert statement for validation

                #assert abs(regular_log_value - closest_recording_log) < tolerance_percentage * regular_log_value

    # Oesophageal Cancer
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
            closest_recording_log = None
            smallest_diff = float('inf')  # Initialize with a large number

            for i in range(len(prevalence)):
                function_date = prevalence['date'][i]
                date_diff = abs((target_date - function_date).days)

                if date_diff < smallest_diff and date_diff < tolerance_days:
                    smallest_diff = date_diff
                    closest_recording_log = prevalence['OesophagealCancer'][i] * prevalence['population'][
                        i]  # Record totals only

            # Check if the closest date difference is within 20 days
            if smallest_diff < tolerance_days and closest_recording_log is not None:
                # Assert statement for validation
                assert abs(regular_log_value - closest_recording_log) < tolerance_percentage * regular_log_value

    # Other Adult Cancers (OAC)
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
            closest_recording_log = None
            smallest_diff = float('inf')  # Initialize with a large number

            for i in range(len(prevalence)):
                function_date = prevalence['date'][i]
                date_diff = abs((target_date - function_date).days)

                if date_diff < smallest_diff and date_diff < tolerance_days:
                    smallest_diff = date_diff
                    closest_recording_log = prevalence['OtherAdultCancer'][i] * prevalence['population'][i]  # Record totals only

            if date_diff < smallest_diff and date_diff < tolerance_days:
                # Handle the case where closest_recording_log is zero
                if closest_recording_log == 0:
                    # Special case: if both values are zero, the assertion is considered true
                    assert regular_log_value == 0
                else:
                    # Normal case: check if the difference is within tolerance
                    assert abs(regular_log_value - closest_recording_log) < tolerance_percentage * regular_log_value

    # Bladder Cancer
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

        closest_recording_log = None
        smallest_diff = float('inf')  # Initialize with a large number

        for i in range(len(prevalence)):
            function_date = prevalence['date'][i]
            date_diff = abs((target_date - function_date).days)

            if date_diff < smallest_diff and date_diff < tolerance_days:
                smallest_diff = date_diff
                closest_recording_log = prevalence['BladderCancer'][i] * prevalence['population'][
                    i]  # Record totals only

        if smallest_diff < tolerance_days and closest_recording_log is not None:

            # Handle the case where closest_recording_log is zero
            if closest_recording_log == 0:
                assert regular_log_value == 0
            else:
                assert abs(regular_log_value - closest_recording_log) < tolerance_percentage * regular_log_value

    # Breast Cancer
    prevalence_breast_cancer_log = output['tlo.methods.breast_cancer']["summary_stats"]
    max_date_in_prevalence = max(prevalence['date'])  # Ensure this is set appropriately

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
            closest_recording_log = None
            smallest_diff = float('inf')  # Initialize with a large number

            for i in range(len(prevalence)):
                function_date = prevalence['date'][i]
                date_diff = abs((target_date - function_date).days)

                if date_diff < smallest_diff and date_diff < tolerance_days:
                    smallest_diff = date_diff
                    closest_recording_log = prevalence['BreastCancer'][i] * prevalence['population'][i]

            if date_diff < smallest_diff and date_diff < tolerance_days:

                    assert abs(regular_log_value - closest_recording_log) < tolerance_percentage * regular_log_value

    # Prostate Cancer
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
            closest_recording_log = None
            smallest_diff = float('inf')  # Initialize with a large number

            for i in range(len(prevalence)):
                function_date = prevalence['date'][i]
                date_diff = abs((target_date - function_date).days)

                if date_diff < smallest_diff and date_diff < tolerance_days:
                    smallest_diff = date_diff
                    closest_recording_log = prevalence['ProstateCancer'][i] * prevalence['population'][
                        i]  # Record totals only

            if smallest_diff < tolerance_days and closest_recording_log is not None:

                # Handle the case where closest_recording_log is zero
                if closest_recording_log == 0:
                    assert regular_log_value == 0
                else:
                    assert abs(regular_log_value - closest_recording_log) < tolerance_percentage * regular_log_value

    # Malaria - only clinical prevalence
    prevalence_malaria_log = output['tlo.methods.malaria']["prevalence"]

    for j in range(len(prevalence_malaria_log)):
        target_date = prevalence_malaria_log['date'][j]
        regular_log_value = prevalence_malaria_log["clinical_prev"][j]  # only records clinical prevalence

        if target_date > max_date_in_prevalence:
            continue
        else:
            closest_recording_log = None
            smallest_diff = float('inf')  # Initialize with a large number

            for i in range(len(prevalence)):
                function_date = prevalence['date'][i]
                date_diff = abs((target_date - function_date).days)

                if date_diff < smallest_diff and date_diff < tolerance_days:
                    smallest_diff = date_diff
                    closest_recording_log = prevalence['Malaria'][i]

            if smallest_diff < tolerance_days and closest_recording_log is not None:

                # Handle the case where closest_recording_log is zero
                if closest_recording_log == 0:
                    assert regular_log_value == 0
                else:
                    assert regular_log_value != closest_recording_log # the regular log only records the clinical prevalence so can expect to be different


    # Cardiometabolic disorders
    conditions =  ['diabetes','hypertension', 'chronic_kidney_disease', 'chronic_lower_back_pain',
                  'chronic_ischemic_hd']  # logged only in adult population

    for condition in conditions:
        prevalence_log = output['tlo.methods.cardio_metabolic_disorders'][f"{condition}_prevalence"]

        for j in range(len(prevalence_log)):
            target_date = prevalence_log['date'][j]
            regular_log_value = prevalence_log['prevalence'][j]  # Fixed to access the correct value

            if target_date > max_date_in_prevalence:
                continue
            else:
                closest_recording_log = None
                smallest_diff = float('inf')  # Initialize with a large number

                for i in range(len(prevalence)):
                    function_date = prevalence['date'][i]
                    date_diff = abs((target_date - function_date).days)

                    if date_diff < smallest_diff and date_diff < tolerance_days:
                        smallest_diff = date_diff
                        closest_recording_log = prevalence[condition][i]

                if smallest_diff < tolerance_days and closest_recording_log is not None:

                    # Handle the case where closest_recording_log is zero
                    if closest_recording_log == 0:
                        assert regular_log_value == 0
                    else:
                        assert regular_log_value > closest_recording_log # log only records adult population


    # Neonatal deaths

    properties_dead = output['tlo.methods.demography.detail']['properties_of_deceased_persons']

    neonatal_deaths = properties_dead[properties_dead['age_days'] < 29]

    neonatal_deaths['year_month'] = neonatal_deaths['date'].dt.to_period('M')
    neonatal_deaths = neonatal_deaths.groupby('year_month').size().reset_index(name='count')
    neonatal_deaths['date'] = neonatal_deaths['year_month'].dt.strftime('%Y-%m-%d')


    for j in range(len(neonatal_deaths)):
        target_date =  pd.to_datetime(neonatal_deaths['date'][j])
        regular_log_value = neonatal_deaths["count"][j]  # only records clinical prevalence

        if target_date > max_date_in_prevalence:
            continue

        else:
            closest_recording_log = None
            smallest_diff = float('inf')  # Initialize with a large number

            for i in range(len(prevalence)):
                function_date = prevalence['date'][i]
                date_diff = abs((target_date - function_date).days)

                if date_diff < smallest_diff and date_diff < tolerance_days:
                    smallest_diff = date_diff
                    closest_recording_log = prevalence['NMR'][i] * prevalence['live_births'][i]

            if smallest_diff < tolerance_days and closest_recording_log is not None:

                # Handle the case where closest_recording_log is zero
                if closest_recording_log == 0:
                    assert regular_log_value == 0
                else:
                    assert regular_log_value != closest_recording_log # the regular log only records the clinical prevalence so can expect to be different


       # maternal deaths
        properties_dead = output['tlo.methods.demography.detail']['properties_of_deceased_persons']

        # Maternal_Disorders processing
        Maternal_Disorders = output['tlo.methods.demography']['death']
        Maternal_Disorders = Maternal_Disorders[Maternal_Disorders['label'] == 'Maternal_Disorders']
        Maternal_Disorders['year_month'] = Maternal_Disorders['date'].dt.to_period('M')
        Maternal_Disorders = Maternal_Disorders.groupby('year_month').size().reset_index(
            name='Maternal_Disorders_count')
        Maternal_Disorders['year_month'] = Maternal_Disorders['year_month'].dt.strftime('%Y-%m')

        # Indirect deaths non-HIV processing
        indirect_deaths_non_HIV = properties_dead[
            (properties_dead['is_pregnant'] | properties_dead['la_is_postpartum']) &
            (properties_dead['cause_of_death'].str.contains('Malaria|Suicide|ever_stroke|diabetes|'
                                                            'chronic_ischemic_hd|ever_heart_attack|'
                                                            'chronic_kidney_disease') |
             (properties_dead['cause_of_death'] == 'TB'))
            ]
        indirect_deaths_non_HIV['year_month'] = indirect_deaths_non_HIV['date'].dt.to_period('M')
        indirect_deaths_non_HIV = indirect_deaths_non_HIV.groupby('year_month').size().reset_index(
            name='indirect_deaths_non_HIV_count')
        indirect_deaths_non_HIV['year_month'] = indirect_deaths_non_HIV['year_month'].dt.strftime('%Y-%m')

        # Direct deaths non-HIV processing
        direct_deaths_non_HIV = properties_dead[
            (properties_dead['is_pregnant'] | properties_dead['la_is_postpartum']) &
            (properties_dead['cause_of_death'].str.contains('AIDS_non_TB|AIDS_TB'))
            ]
        direct_deaths_non_HIV['year_month'] = direct_deaths_non_HIV['date'].dt.to_period('M')
        direct_deaths_non_HIV = direct_deaths_non_HIV.groupby('year_month').size().reset_index(
            name='direct_deaths_non_HIV_count')
        direct_deaths_non_HIV['direct_deaths_non_HIV_count'] *= 0.3
        direct_deaths_non_HIV['year_month'] = direct_deaths_non_HIV['year_month'].dt.strftime('%Y-%m')

        # Merging the DataFrames
        combined_df = pd.merge(Maternal_Disorders, indirect_deaths_non_HIV, on='year_month', how='outer')
        combined_df = pd.merge(combined_df, direct_deaths_non_HIV, on='year_month', how='outer')

        # Fill NaN values with 0
        combined_df.fillna(0, inplace=True)

        combined_df.sort_values(by='year_month', inplace=True)
        combined_df.rename(columns={'year_month': 'date'}, inplace=True)
        combined_df['count'] = combined_df[
            ['Maternal_Disorders_count', 'indirect_deaths_non_HIV_count', 'direct_deaths_non_HIV_count']].sum(axis=1)
    for j in range(len(combined_df)):
        target_date = pd.to_datetime(combined_df['date'][j])
        regular_log_value = combined_df["count"][j]  # only records clinical prevalence

        if target_date > max_date_in_prevalence:
            continue
        else:
            closest_recording_log = None
            smallest_diff = float('inf')  # Initialize with a large number

            for i in range(len(prevalence)):
                function_date = prevalence['date'][i]
                date_diff = abs((target_date - function_date).days)

                if date_diff < smallest_diff and date_diff < tolerance_days:
                    smallest_diff = date_diff
                    closest_recording_log = prevalence['MMR'][i] * prevalence['live_births'][i]

            if smallest_diff < tolerance_days and closest_recording_log is not None:
                    assert regular_log_value != closest_recording_log # the regular log only records the clinical prevalence so can expect to be different
    # Antenatal still births
    if 'antenatal_stillbirth' in output.get('tlo.methods.pregnancy_supervisor', {}):
        antenatal_stillbirths = output['tlo.methods.pregnancy_supervisor']['antenatal_stillbirth']

        antenatal_stillbirths['year_month'] = antenatal_stillbirths['date'].dt.to_period('M')
        antenatal_stillbirths = antenatal_stillbirths.groupby('year_month').size().reset_index(name='count')
        antenatal_stillbirths['date'] = antenatal_stillbirths['year_month'].dt.strftime('%Y-%m-%d')

        for j in range(len(antenatal_stillbirths)):
            target_date = pd.to_datetime(antenatal_stillbirths['date'][j])
            regular_log_value = antenatal_stillbirths["count"][j]  # only records clinical prevalence

            if target_date > max_date_in_prevalence:
                continue
            else:
                closest_recording_log = None
                smallest_diff = float('inf')  # Initialize with a large number

                for i in range(len(prevalence)):
                    function_date = prevalence['date'][i]
                    date_diff = abs((target_date - function_date).days)

                    if date_diff < smallest_diff and date_diff < tolerance_days:
                        smallest_diff = date_diff
                        closest_recording_log = prevalence['Antenatal stillbirth'][i]

                if smallest_diff < tolerance_days and closest_recording_log is not None:
                    assert regular_log_value != closest_recording_log  # the regular log only records the clinical prevalence so can expect to be different

    # Intrapartum still births
    if 'intrapartum_stillbirth' in output.get('tlo.methods.labour', {}):
        intrapartum_stillbirths = output['tlo.methods.labour']['intrapartum_stillbirth']

        intrapartum_stillbirths['year_month'] = intrapartum_stillbirths['date'].dt.to_period('M')
        intrapartum_stillbirths = intrapartum_stillbirths.groupby('year_month').size().reset_index(name='count')
        intrapartum_stillbirths['date'] = intrapartum_stillbirths['year_month'].dt.strftime('%Y-%m-%d')

        for j in range(len(intrapartum_stillbirths)):
            target_date = pd.to_datetime(intrapartum_stillbirths['date'][j])
            regular_log_value = intrapartum_stillbirths["count"][j]  # only records clinical prevalence

            if target_date > max_date_in_prevalence:
                continue
            else:
                closest_recording_log = None
                smallest_diff = float('inf')  # Initialize with a large number

                for i in range(len(prevalence)):
                    function_date = prevalence['date'][i]
                    date_diff = abs((target_date - function_date).days)

                    if date_diff < smallest_diff and date_diff < tolerance_days:
                        smallest_diff = date_diff
                        closest_recording_log = prevalence['Intrapartum stillbirth'][i]

                if smallest_diff < tolerance_days and closest_recording_log is not None:
                    assert regular_log_value != closest_recording_log  # the regular log only records the clinical prevalence so can expect to be different


tmpdir = 'outputs/'
test_run_with_healthburden_with_dummy_diseases(tmpdir, seed)
