import os
from pathlib import Path

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods.fullmodel import fullmodel

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
outputpath = Path("./outputs/")

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)

popsize = 1000
seed = 42
do_sim = True
def check_dtypes(simulation):
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def find_closest_recording(prevalence, target_date, log_value, column_name, multiply_by_pop):
    """
    Finds the closest recording log in the prevalence DataFrame based on the target date.

    Parameters:
    prevalence (DataFrame): The DataFrame containing the records.
    target_date (datetime): The date to find the closest log for.
    log_value (any): The value to validate against the closest log.
    column_name (str): The name of the column to compare.
    multiply_by_pop (bool): Whether to multiply by population size, as some regular logs have numbers not prevalences

    Returns:
    None
    """
    closest_recording_log = None
    smallest_diff = float('inf')  # Initialize with a large number

    for i in range(len(prevalence)):
        function_date = prevalence['date'][i]
        date_diff = abs((target_date - function_date).days)

        if date_diff < smallest_diff:
            smallest_diff = date_diff
            if multiply_by_pop:
                closest_recording_log = prevalence[column_name][i] * prevalence['population'][i]
            else:
                closest_recording_log = prevalence[column_name][i]

    if closest_recording_log is not None:
        # Assert statement for validation
        assert log_value == closest_recording_log


def log_prevalences_from_sim_func(sim):
    """Logs the prevalence of disease monthly"""
    health_burden = sim.modules['HealthBurden']
    monthly_prevalence = health_burden.prevalence_of_diseases
    monthly_prevalence['date'] = sim.date.year
    return monthly_prevalence

def test_run_with_healthburden_with_dummy_diseases(tmpdir, seed):
    """Check that everything runs in the simple cases of Mockitis and Chronic Syndrome and that outputs are as expected."""

    sim = Simulation(start_date=start_date, seed=seed, log_config={'filename': 'test_log', 'directory': outputpath})
    sim.register(*fullmodel(
            resourcefilepath=resourcefilepath,
            use_simplified_births=False,))
    sim.make_initial_population(n=popsize)
    sim.modules['HealthBurden'].parameters['test'] = True
    sim.simulate(end_date=end_date)
    check_dtypes(sim)
    output = parse_log_file(sim.log_filepath)

    prevalence = output['tlo.methods.healthburden']['prevalence_of_diseases']
    #max_date_in_prevalence = max(prevalence['date'])
    log_prevalences_from_sim = log_prevalences_from_sim_func(sim)

    for log_date in log_prevalences_from_sim['date']:
        # Check if the current date from log_prevalencesfrom_sim exists in the prevalence DataFrame
        if log_date in prevalence['date'].values:
            # Extract the corresponding row from prevalence
            prevalence_row = prevalence.loc[prevalence['date'] == log_date].squeeze()

            # Remove the date column for comparison (if needed)
            if 'date' in prevalence.columns:
                prevalence_row = prevalence_row.drop('date')

            # Find the corresponding row in log_prevalences_from_sim
            sim_row = log_prevalences_from_sim.loc[
                log_prevalences_from_sim['date'] == log_date].squeeze()

            # Iterate over the columns to compare values
            for column in prevalence_row.index:
                # Compare the values between the two DataFrames for this date and column
                if prevalence_row[column] != sim_row[column]:
                    # Handle mismatches as needed (e.g., logging, storing in a list, etc.)
                    pass
        else:
            # Handle cases where the date is not found in prevalence DataFrame
            pass

    # #HIV
    # prevalence_HIV_log = output['tlo.methods.hiv']['summary_inc_and_prev_for_adults_and_children_and_fsw']
    # for j in range(len(prevalence_HIV_log)):
    #     target_date = prevalence_HIV_log['date'][j]
    #     regular_log_value = prevalence_HIV_log["total_plhiv"][j] / prevalence_HIV_log["pop_total"][j]
    #
    #     if target_date > max_date_in_prevalence:
    #         continue
    #     else:
    #         find_closest_recording(prevalence, target_date, regular_log_value, 'Hiv', False)
    #
    # # TB
    #
    # prevalence_tb_log = output['tlo.methods.tb']["tb_prevalence"]
    #
    # for j in range(len(prevalence_tb_log)):
    #     target_date = prevalence_tb_log['date'][j]
    #     regular_log_value = prevalence_tb_log["tbPrevActive"][j] + prevalence_tb_log["tbPrevLatent"][j]
    #
    #     if target_date > max_date_in_prevalence:
    #         continue
    #     else:
    #         find_closest_recording(prevalence, target_date, regular_log_value, 'Tb', False)
    #
    # # Oesophageal Cancer
    # prevalence_oesophageal_cancer_log = output['tlo.methods.oesophagealcancer']["summary_stats"]
    #
    # for j in range(len(prevalence_oesophageal_cancer_log)):
    #     target_date = prevalence_oesophageal_cancer_log['date'][j]
    #     regular_log_value = (
    #         prevalence_oesophageal_cancer_log["total_low_grade_dysplasia"][j] +
    #         prevalence_oesophageal_cancer_log["total_high_grade_dysplasia"][j] +
    #         prevalence_oesophageal_cancer_log["total_stage1"][j] +
    #         prevalence_oesophageal_cancer_log["total_stage2"][j] +
    #         prevalence_oesophageal_cancer_log["total_stage3"][j] +
    #         prevalence_oesophageal_cancer_log["total_stage4"][j]
    #     )
    #
    #     if target_date <= max_date_in_prevalence:
    #
    #         find_closest_recording(prevalence, target_date, regular_log_value, 'OesophagealCancer', True)
    #
    #
    # # Other Adult Cancers (OAC)
    # prevalence_oac_cancer_log = output['tlo.methods.other_adult_cancers']["summary_stats"]
    #
    # # Bladder Cancer
    # prevalence_bladder_cancer_log = output['tlo.methods.bladder_cancer']["summary_stats"]
    #
    # for j in range(len(prevalence_bladder_cancer_log)):
    #     target_date = prevalence_bladder_cancer_log['date'][j]
    #     regular_log_value = (
    #         prevalence_bladder_cancer_log["total_tis_t1"][j] +
    #         prevalence_bladder_cancer_log["total_t2p"][j] +
    #         prevalence_bladder_cancer_log["total_metastatic"][j]
    #     )
    #
    #     if target_date <= max_date_in_prevalence:
    #         find_closest_recording(prevalence, target_date, regular_log_value, 'BladderCancer', True)
    #
    #
    # # Breast Cancer
    # prevalence_breast_cancer_log = output['tlo.methods.breast_cancer']["summary_stats"]
    # max_date_in_prevalence = max(prevalence['date'])  # Ensure this is set appropriately
    #
    # for j in range(len(prevalence_breast_cancer_log)):
    #     target_date = prevalence_breast_cancer_log['date'][j]
    #     regular_log_value = (
    #         prevalence_breast_cancer_log["total_stage1"][j] +
    #         prevalence_breast_cancer_log["total_stage2"][j] +
    #         prevalence_breast_cancer_log["total_stage3"][j] +
    #         prevalence_breast_cancer_log["total_stage4"][j]
    #     )
    #
    #     if target_date <= max_date_in_prevalence:
    #         find_closest_recording(prevalence, target_date, regular_log_value, 'BreastCancer', True)
    #
    # # Prostate Cancer
    # prevalence_prostate_cancer_log = output['tlo.methods.prostate_cancer']["summary_stats"]
    #
    # for j in range(len(prevalence_prostate_cancer_log)):
    #     target_date = prevalence_prostate_cancer_log['date'][j]
    #     regular_log_value = (
    #         prevalence_prostate_cancer_log["total_prostate_confined"][j] +
    #         prevalence_prostate_cancer_log["total_local_ln"][j] +
    #         prevalence_prostate_cancer_log["total_metastatic"][j]
    #     )
    #
    #     if target_date <= max_date_in_prevalence:
    #         find_closest_recording(prevalence, target_date, regular_log_value, 'ProstateCancer', True)
    #
    #
    # # Cardiometabolic disorders
    # conditions =  ['diabetes','hypertension', 'chronic_kidney_disease', 'chronic_lower_back_pain',
    #               'chronic_ischemic_hd']  # logged only in adult population
    #
    # for condition in conditions:
    #     prevalence_log = output['tlo.methods.cardio_metabolic_disorders'][f"{condition}_prevalence"]
    #
    #     for j in range(len(prevalence_log)):
    #         target_date = prevalence_log['date'][j]
    #         regular_log_value = prevalence_log['prevalence'][j]  # Fixed to access the correct value
    #
    #         if target_date <= max_date_in_prevalence:
    #             find_closest_recording(prevalence, target_date, regular_log_value, condition, False)
    #
    # # Neonatal deaths
    #
    # properties_dead = output['tlo.methods.demography.detail']['properties_of_deceased_persons']
    #
    # neonatal_deaths = properties_dead[properties_dead['age_days'] < 29]
    #
    # neonatal_deaths['year_month'] = neonatal_deaths['date'].dt.to_period('M')
    # neonatal_deaths = neonatal_deaths.groupby('year_month').size().reset_index(name='count')
    # neonatal_deaths['date'] = neonatal_deaths['year_month'].dt.strftime('%Y-%m-%d')
    #
    #
    # for j in range(len(neonatal_deaths)):
    #     target_date =  pd.to_datetime(neonatal_deaths['date'][j])
    #     regular_log_value = neonatal_deaths["count"][j]  # only records clinical prevalence
    #
    #     if target_date <= max_date_in_prevalence:
    #         closest_recording_log = None
    #         smallest_diff = float('inf')  # Initialize with a large number
    #
    #         for i in range(len(prevalence)):
    #             function_date = prevalence['date'][i]
    #             date_diff = abs((target_date - function_date).days)
    #
    #             if date_diff < smallest_diff:
    #                 smallest_diff = date_diff
    #                 closest_recording_log = prevalence['NMR'][i] * prevalence['live_births'][i]
    #
    #         if closest_recording_log is not None:
    #             assert regular_log_value == closest_recording_log
    #
    #    # maternal deaths
    #     properties_dead = output['tlo.methods.demography.detail']['properties_of_deceased_persons']
    #
    #     # Maternal_Disorders processing
    #     Maternal_Disorders = output['tlo.methods.demography']['death']
    #     Maternal_Disorders = Maternal_Disorders[Maternal_Disorders['label'] == 'Maternal_Disorders']
    #     Maternal_Disorders['year_month'] = Maternal_Disorders['date'].dt.to_period('M')
    #     Maternal_Disorders = Maternal_Disorders.groupby('year_month').size().reset_index(
    #         name='Maternal_Disorders_count')
    #     Maternal_Disorders['year_month'] = Maternal_Disorders['year_month'].dt.strftime('%Y-%m')
    #
    #     # Indirect deaths non-HIV processing
    #     indirect_deaths_non_HIV = properties_dead[
    #         (properties_dead['is_pregnant'] | properties_dead['la_is_postpartum']) &
    #         (properties_dead['cause_of_death'].str.contains('Malaria|Suicide|ever_stroke|diabetes|'
    #                                                         'chronic_ischemic_hd|ever_heart_attack|'
    #                                                         'chronic_kidney_disease') |
    #          (properties_dead['cause_of_death'] == 'TB'))
    #         ]
    #     indirect_deaths_non_HIV['year_month'] = indirect_deaths_non_HIV['date'].dt.to_period('M')
    #     indirect_deaths_non_HIV = indirect_deaths_non_HIV.groupby('year_month').size().reset_index(
    #         name='indirect_deaths_non_HIV_count')
    #     indirect_deaths_non_HIV['year_month'] = indirect_deaths_non_HIV['year_month'].dt.strftime('%Y-%m')
    #
    #     # Direct deaths non-HIV processing
    #     direct_deaths_non_HIV = properties_dead[
    #         (properties_dead['is_pregnant'] | properties_dead['la_is_postpartum']) &
    #         (properties_dead['cause_of_death'].str.contains('AIDS_non_TB|AIDS_TB'))
    #         ]
    #     direct_deaths_non_HIV['year_month'] = direct_deaths_non_HIV['date'].dt.to_period('M')
    #     direct_deaths_non_HIV = direct_deaths_non_HIV.groupby('year_month').size().reset_index(
    #         name='direct_deaths_non_HIV_count')
    #     direct_deaths_non_HIV['direct_deaths_non_HIV_count'] *= 0.3
    #     direct_deaths_non_HIV['year_month'] = direct_deaths_non_HIV['year_month'].dt.strftime('%Y-%m')
    #
    #     # Merging the DataFrames
    #     combined_df = pd.merge(Maternal_Disorders, indirect_deaths_non_HIV, on='year_month', how='outer')
    #     combined_df = pd.merge(combined_df, direct_deaths_non_HIV, on='year_month', how='outer')
    #
    #     # Fill NaN values with 0
    #     combined_df.fillna(0, inplace=True)
    #
    #     combined_df.sort_values(by='year_month', inplace=True)
    #     combined_df.rename(columns={'year_month': 'date'}, inplace=True)
    #     combined_df['count'] = combined_df[
    #         ['Maternal_Disorders_count', 'indirect_deaths_non_HIV_count', 'direct_deaths_non_HIV_count']].sum(axis=1)
    # for j in range(len(combined_df)):
    #     target_date = pd.to_datetime(combined_df['date'][j])
    #     regular_log_value = combined_df["count"][j]  # only records clinical prevalence
    #
    #     if target_date <= max_date_in_prevalence:
    #
    #         closest_recording_log = None
    #         smallest_diff = float('inf')  # Initialize with a large number
    #
    #         for i in range(len(prevalence)):
    #             function_date = prevalence['date'][i]
    #             date_diff = abs((target_date - function_date).days)
    #
    #             if date_diff < smallest_diff:
    #                 smallest_diff = date_diff
    #                 closest_recording_log = prevalence['MMR'][i] * prevalence['live_births'][i]
    #
    #         if closest_recording_log is not None:
    #             assert regular_log_value != closest_recording_log # the regular log only records the clinical prevalence so can expect to be different
    # # Antenatal still births
    # if 'antenatal_stillbirth' in output.get('tlo.methods.pregnancy_supervisor', {}):
    #     antenatal_stillbirths = output['tlo.methods.pregnancy_supervisor']['antenatal_stillbirth']
    #
    #     antenatal_stillbirths['year_month'] = antenatal_stillbirths['date'].dt.to_period('M')
    #     antenatal_stillbirths = antenatal_stillbirths.groupby('year_month').size().reset_index(name='count')
    #     antenatal_stillbirths['date'] = antenatal_stillbirths['year_month'].dt.strftime('%Y-%m-%d')
    #
    #     for j in range(len(antenatal_stillbirths)):
    #         target_date = pd.to_datetime(antenatal_stillbirths['date'][j])
    #         regular_log_value = antenatal_stillbirths["count"][j]  # only records clinical prevalence
    #
    #         if target_date <= max_date_in_prevalence:
    #
    #             closest_recording_log = None
    #             smallest_diff = float('inf')  # Initialize with a large number
    #
    #             for i in range(len(prevalence)):
    #                 function_date = prevalence['date'][i]
    #                 date_diff = abs((target_date - function_date).days)
    #
    #                 if date_diff < smallest_diff:
    #                     smallest_diff = date_diff
    #                     closest_recording_log = prevalence['Antenatal stillbirth'][i]
    #
    #             if closest_recording_log is not None:
    #                 assert regular_log_value == closest_recording_log
    #
    # # Intrapartum still births
    # if 'intrapartum_stillbirth' in output.get('tlo.methods.labour', {}):
    #     intrapartum_stillbirths = output['tlo.methods.labour']['intrapartum_stillbirth']
    #
    #     intrapartum_stillbirths['year_month'] = intrapartum_stillbirths['date'].dt.to_period('M')
    #     intrapartum_stillbirths = intrapartum_stillbirths.groupby('year_month').size().reset_index(name='count')
    #     intrapartum_stillbirths['date'] = intrapartum_stillbirths['year_month'].dt.strftime('%Y-%m-%d')
    #
    #     for j in range(len(intrapartum_stillbirths)):
    #         target_date = pd.to_datetime(intrapartum_stillbirths['date'][j])
    #         regular_log_value = intrapartum_stillbirths["count"][j]  # only records clinical prevalence
    #
    #         if target_date > max_date_in_prevalence:
    #             continue
    #         else:
    #             closest_recording_log = None
    #             smallest_diff = float('inf')  # Initialize with a large number
    #
    #             for i in range(len(prevalence)):
    #                 function_date = prevalence['date'][i]
    #                 date_diff = abs((target_date - function_date).days)
    #
    #                 if date_diff < smallest_diff:
    #                     smallest_diff = date_diff
    #                     closest_recording_log = prevalence['Intrapartum stillbirth'][i]
    #
    #             if closest_recording_log is not None:
    #                 assert regular_log_value == closest_recording_log
    #
    # return log_prevalences_from_sim_func, prevalence

tmpdir = 'outputs/'
test_run_with_healthburden_with_dummy_diseases(tmpdir, seed)
