import pandas as pd

# Data accessed from https://dhis2.health.gov.mw/dhis-web-data-visualizer/#/YiQK65skxjz
# Reporting rate is expected reporting vs actual reporting
reporting_data = pd.read_csv('/Users/rem76/Desktop/Climate_change_health/Data/Reporting_Rate/Reporting_Rate_Central_Hospital_2000_2024.csv')

# Divide dataset based on what is being reported
# get metrics recorded
all_columns = reporting_data.columns
metrics = set([col.split(" - Reporting rate")[0] for col in all_columns])
metrics = {metric for metric in metrics if not metric.startswith("organisation")} # inlcude only reporting data


monthly_reporting_data_by_metric =  {}

for metric in metrics:
    columns_of_interest = [reporting_data.columns[1]] + reporting_data.columns[reporting_data.columns.str.startswith(metric)].tolist()
    data_of_interest = reporting_data[columns_of_interest]
    data_of_interest.columns = [col.replace(metric, "") for col in data_of_interest.columns]
    data_of_interest.columns = [col.replace(" - Reporting rate ", "") for col in data_of_interest.columns]
    monthly_reporting_data_by_metric[metric] = data_of_interest

