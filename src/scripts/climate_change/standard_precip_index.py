import pandas as pd
import numpy as np
from standard_precip.spi import SPI
from standard_precip.utils import plot_index


spi = SPI()
service = "Inpatient" # ANC

rainfall_data =  pd.read_csv(
                        f"/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facilities_with_{service}_lm_baseline_all_years.csv",
                        index_col=0)

n_rows = len(rainfall_data)
dates = pd.date_range(start="1940-01-01", periods=n_rows, freq="MS")
rainfall_data['date'] = dates
# empty data frame for storage
spi_time_facility = pd.DataFrame(np.nan, index=rainfall_data.index, columns=rainfall_data.columns)
spi_time_facility['date'] = dates

for facility in rainfall_data.columns[0:len(rainfall_data.columns) -1]: #unsure why phantom column is appearing
    rainfall_for_facility = pd.DataFrame(rainfall_data.loc[:, ['date', facility]], index=rainfall_data.index)
    rainfall_for_facility.rename(columns={facility:'precip'}, inplace=True)
    df_spi = spi.calculate(
        rainfall_for_facility,
        'date',
        'precip',
        freq="M",
        scale=1,
        fit_type="lmom",
        dist_type="gam"
    )
    spi_time_facility.loc[:,facility] = df_spi.iloc[:,-1].values
    spi_time_facility.rename(columns={df_spi.columns[-1]: facility}, inplace=True)

spi_time_facility.drop(columns='date', inplace=True)
# remove before 2011
spi_time_facility = spi_time_facility.iloc[(12 * (2011 - 1940)):,]
spi_time_facility.to_csv(f"/Users/rem76/Desktop/Climate_change_health/Data/Drought_data/historical_drought_data_2010_2024_{service}.csv")

