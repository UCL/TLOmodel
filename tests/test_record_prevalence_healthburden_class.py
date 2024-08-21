import datetime
import os
import pickle
from pathlib import Path

from tlo import Date, Simulation
from tlo.analysis.utils import create_pickles_locally, parse_log_file, extract_results
from tlo.methods.fullmodel import fullmodel

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
outputpath = Path("./outputs/test_record_prevalence/0/0")

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 12)

popsize = 500
seed = 42
tolerance_percentage = 0.1 # attempt to account for differences in recording times
def check_dtypes(simulation):
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()

def test_run_with_healthburden_with_dummy_diseases(tmpdir, seed):
    """Check that everything runs in the simple cases of Mockitis and Chronic Syndrome and that outputs are as expected."""

    #sim = Simulation(start_date=start_date, seed=seed, log_config={'filename': 'test_log', 'directory': outputpath})

    #sim.register(*fullmodel(
    #    resourcefilepath=resourcefilepath,
    #    use_simplified_births=False,
    #))

    #sim.make_initial_population(n=popsize)
    #sim.simulate(end_date=end_date)
    #check_dtypes(sim)
    #print(sim.logfilepath)
    log_filepath = 'outputs/test_record_prevalence/0/0/test_log__2024-08-21T110023.log'
    output = parse_log_file(log_filepath)
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

        closest_recording_log = None
        smallest_diff = float('inf')  # Initialize with a large number

        for i in range(len(prevalence)):
            function_date = prevalence['date'][i]
            date_diff = abs((target_date - function_date).days)

            if date_diff < smallest_diff:
                smallest_diff = date_diff
                closest_recording_log = prevalence['Hiv'][i]

        # Check if the closest date difference is within 20 days
        if smallest_diff < 20 and closest_recording_log is not None:
            # Assert statement for validation
            assert abs(regular_log_value - closest_recording_log) < tolerance_percentage * closest_recording_log
        else:
            # Optionally handle the case where no close enough date was found
            print(f"No close enough date found for HIV log date: {target_date}")

    # TB

    # TB
    prevalence_tb_log = output['tlo.methods.tb']["tb_prevalence"]

    for j in range(len(prevalence_tb_log)):
        target_date = prevalence_tb_log['date'][j]
        regular_log_value = prevalence_tb_log["tbPrevActive"][j] + prevalence_tb_log["tbPrevLatent"][j]

        if target_date > max_date_in_prevalence:
            continue

        closest_recording_log = None
        smallest_diff = float('inf')  # Initialize with a large number

        for i in range(len(prevalence)):
            function_date = prevalence['date'][i]
            date_diff = abs((target_date - function_date).days)

            if date_diff < smallest_diff:
                smallest_diff = date_diff
                closest_recording_log = prevalence['Tb'][i]

        # Check if the closest date difference is within 20 days
        if smallest_diff < 20 and closest_recording_log is not None:
            print("TB log date:", target_date)
            print("Closest function date:", function_date)
            print("Closest recording log value:", closest_recording_log)
            print("Regular log value:", regular_log_value)

            # Assert statement for validation
            assert abs(regular_log_value - closest_recording_log) < tolerance_percentage * closest_recording_log
        else:
            # Optionally handle the case where no close enough date was found
            print(f"No close enough date found for TB log date: {target_date}")

    # Oesophageal Cancer
    # Esophageal Cancer
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

        closest_recording_log = None
        smallest_diff = float('inf')  # Initialize with a large number

        for i in range(len(prevalence)):
            function_date = prevalence['date'][i]
            date_diff = abs((target_date - function_date).days)

            if date_diff < smallest_diff:
                smallest_diff = date_diff
                closest_recording_log = prevalence['OesophagealCancer'][i] * prevalence['population'][
                    i]  # Record totals only

        # Check if the closest date difference is within 20 days
        if smallest_diff < 20 and closest_recording_log is not None:
            print("OC log date:", target_date)
            print("Closest function date:", function_date)
            print("Function OC:", closest_recording_log)
            print("Log OC:", regular_log_value)

            # Assert statement for validation
            assert abs(regular_log_value - closest_recording_log) < tolerance_percentage * closest_recording_log
        else:
            # Optionally handle the case where no close enough date was found
            print(f"No close enough date found for OC log date: {target_date}")

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

        closest_recording_log = None
        smallest_diff = float('inf')  # Initialize with a large number

        for i in range(len(prevalence)):
            function_date = prevalence['date'][i]
            date_diff = abs((target_date - function_date).days)

            if date_diff < smallest_diff:
                smallest_diff = date_diff
                closest_recording_log = prevalence['OtherAdultCancer'][i] * prevalence['population'][
                    i]  # Record totals only

        # Check if the closest date difference is within 20 days
        if smallest_diff < 20 and closest_recording_log is not None:
            print("OAC log date:", target_date)
            print("Closest function date:", function_date)
            print("Log OAC:", regular_log_value)
            print("Function OAC:", round(closest_recording_log, 7))

            # Handle the case where closest_recording_log is zero
            if closest_recording_log == 0:
                # Special case: if both values are zero, the assertion is considered true
                assert regular_log_value == 0
            else:
                # Normal case: check if the difference is within tolerance
                assert abs(regular_log_value - closest_recording_log) < tolerance_percentage * closest_recording_log
        else:
            # Optionally handle the case where no close enough date was found
            print(f"No close enough date found for OAC log date: {target_date}")

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

            if date_diff < smallest_diff:
                smallest_diff = date_diff
                closest_recording_log = prevalence['BladderCancer'][i] * prevalence['population'][
                    i]  # Record totals only

        # Check if the closest date difference is within 20 days
        if smallest_diff < 20 and closest_recording_log is not None:
            print("Bladder log date:", target_date)
            print("Closest function date:", function_date)
            print("Log Bladder:", regular_log_value)
            print("Function Bladder:", round(closest_recording_log, 7))

            # Handle the case where closest_recording_log is zero
            if closest_recording_log == 0:
                assert regular_log_value == 0
            else:
                assert abs(regular_log_value - closest_recording_log) < tolerance_percentage * closest_recording_log
        else:
            # Optionally handle the case where no close enough date was found
            print(f"No close enough date found for Bladder log date: {target_date}")

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

        closest_recording_log = None
        smallest_diff = float('inf')  # Initialize with a large number

        for i in range(len(prevalence)):
            function_date = prevalence['date'][i]
            date_diff = abs((target_date - function_date).days)

            if date_diff < smallest_diff and date_diff < 20:
                smallest_diff = date_diff
                closest_recording_log = prevalence['BreastCancer'][i] * prevalence['population'][i]

        # Check if the closest date difference is within 20 days
        if smallest_diff < 20 and closest_recording_log is not None:
            print("Breast log date:", target_date)
            print("Closest function date:", function_date)
            print("Log Breast:", regular_log_value)
            print("Function Breast:", round(closest_recording_log, 7))

            # Handle the case where closest_recording_log is zero
            if closest_recording_log == 0:
                assert regular_log_value == 0
            else:
                assert abs(regular_log_value - closest_recording_log) < tolerance_percentage * closest_recording_log
        else:
            # Optionally handle the case where no close enough date was found
            print(f"No close enough date found for Breast log date: {target_date}")

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

        closest_recording_log = None
        smallest_diff = float('inf')  # Initialize with a large number

        for i in range(len(prevalence)):
            function_date = prevalence['date'][i]
            date_diff = abs((target_date - function_date).days)

            if date_diff < smallest_diff:
                smallest_diff = date_diff
                closest_recording_log = prevalence['ProstateCancer'][i] * prevalence['population'][
                    i]  # Record totals only

        # Check if the closest date difference is within 20 days
        if smallest_diff < 20 and closest_recording_log is not None:
            print("Prostate log date:", target_date)
            print("Closest function date:", function_date)
            print("Log Prostate:", regular_log_value)
            print("Function Prostate:", round(closest_recording_log, 7))

            # Handle the case where closest_recording_log is zero
            if closest_recording_log == 0:
                assert regular_log_value == 0
            else:
                assert abs(regular_log_value - closest_recording_log) < tolerance_percentage * closest_recording_log
        else:
            # Optionally handle the case where no close enough date was found
            print(f"No close enough date found for Prostate log date: {target_date}")

    # Malaria - only clinical prevalence
    prevalence_malaria_log = output['tlo.methods.malaria']["prevalence"]

    for j in range(len(prevalence_malaria_log)):
        target_date = prevalence_malaria_log['date'][j]
        regular_log_value = prevalence_malaria_log["clinical_prev"][j]  # only records clinical prevalence

        if target_date > max_date_in_prevalence:
            continue

        closest_recording_log = None
        smallest_diff = float('inf')  # Initialize with a large number

        for i in range(len(prevalence)):
            function_date = prevalence['date'][i]
            date_diff = abs((target_date - function_date).days)

            if date_diff < smallest_diff:
                smallest_diff = date_diff
                closest_recording_log = prevalence['Malaria'][i]

        # Check if the closest date difference is within 20 days
        if smallest_diff < 20 and closest_recording_log is not None:
            print("Malaria log date:", target_date)
            print("Closest function date:", function_date)
            print("Log Malaria:", regular_log_value)
            print("Function Malaria:", round(closest_recording_log, 7))

            # Handle the case where closest_recording_log is zero
            if closest_recording_log == 0:
                assert regular_log_value == 0
            else:
                assert regular_log_value != closest_recording_log # the regular log only records the clinical prevalence so can expect to be different
        else:
            # Optionally handle the case where no close enough date was found
            print(f"No close enough date found for Malaria log date: {target_date}")


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

            closest_recording_log = None
            smallest_diff = float('inf')  # Initialize with a large number

            for i in range(len(prevalence)):
                function_date = prevalence['date'][i]
                date_diff = abs((target_date - function_date).days)

                if date_diff < smallest_diff:
                    smallest_diff = date_diff
                    closest_recording_log = prevalence[condition][i]
                    closest_date = function_date

            # Check if the closest date difference is within 20 days
            if smallest_diff < 20 and closest_recording_log is not None:
                print(f"{condition} log date:", target_date)
                print(f"Closest function date:", closest_date)
                print(f"Log {condition}:", regular_log_value)
                print(f"Function {condition}:", round(closest_recording_log, 7))

                # Handle the case where closest_recording_log is zero
                if closest_recording_log == 0:
                    assert regular_log_value == 0
                else:
                    assert regular_log_value > closest_recording_log # log only records adult population
            else:
                # Optionally handle the case where no close enough date was found
                print(f"No close enough date found for {condition} log date: {target_date}")
                print(f"Smallest date difference: {smallest_diff} days")
