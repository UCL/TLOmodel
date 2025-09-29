import itertools
from pathlib import Path

import pandas as pd

RFs_consumable_path = Path('./resources') / 'healthsystem' / 'consumables'
cons_avail_df = pd.read_csv(RFs_consumable_path / 'ResourceFile_Consumables_availability_small.csv')

unique_facilities = cons_avail_df['Facility_ID'].unique()
unique_months = cons_avail_df['month'].unique()

new_rows = pd.DataFrame(
    [
        dict({'Facility_ID': f, 'month': m, 'item_code': 208},
             **{col: 0.0 for col in cons_avail_df.columns if col not in ['Facility_ID', 'month', 'item_code']})
        for f, m in itertools.product(unique_facilities, unique_months)
    ]
)
cons_avail_FSinterv_df = pd.concat([cons_avail_df, new_rows], ignore_index=True)

cons_avail_FSinterv_df.to_csv(RFs_consumable_path / 'ResourceFile_Consumables_availability_small_FSinterv.csv', index=False)
