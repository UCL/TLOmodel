import pandas as pd
import numpy as np
from standard_precip.spi import SPI
from standard_precip.utils import plot_index


spi = SPI()
rainfall_data =  pd.read_csv(
                    "/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facilities_with_ANC_lm.csv",
                    index_col=0)
n_rows = len(rainfall_data)

dates = pd.date_range(start="2010-01-01", periods=n_rows, freq="MS")

rainfall_data['date'] = dates

# empty data frame for storage
spi_time_facility = pd.DataFrame(np.nan, index=rainfall_data.index, columns=rainfall_data.columns)


for facility in rainfall_data.columns[0:len(rainfall_data.columns) -1]: #unsure why phantom column is appearing
    rainfall_for_facility = pd.DataFrame(rainfall_data.loc[:, ['date', facility]], index=rainfall_data.index)
    rainfall_for_facility.rename(columns={facility:'precip'}, inplace=True)
    df_spi = spi.calculate(
        rainfall_for_facility,
        'date',
        'precip',
        freq="M",
        scale=3,
        fit_type="lmom",
        dist_type="gam"
    )
    spi_time_facility.loc[:,facility] = df_spi.loc[:,'precip_scale_3_calculated_index']
    spi_time_facility.rename(columns={'precip_scale_3_calculated_index':facility}, inplace=True)

spi_time_facility.drop(columns='date', inplace=True)
print(spi_time_facility)


percent_below_1 = spi_time_facility.apply(lambda col: (col < -1).sum() / len(col) * 100)
print(percent_below_1)
