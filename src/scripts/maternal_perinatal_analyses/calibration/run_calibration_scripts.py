from ..calibration import output_all_death_calibration_per_year
from ..calibration import output_all_key_outcomes_per_year

# pass file name to provide outputs
for file in ['standard_mph_calibration']:
    output_all_key_outcomes_per_year.output_incidence_for_calibration(
        scenario_filename=f'{file}.py',
        pop_size='50k',
        outputspath='./outputs/sejjj49@ucl.ac.uk/',
        sim_years=list(range(2010, 2021)),
        show_and_store_graphs=True)

    output_all_death_calibration_per_year.output_all_death_calibrations(
        scenario_filename=f'{file}.py',
        outputspath='./outputs/sejjj49@ucl.ac.uk/',
        pop_size='50k',
        sim_years=list(range(2010, 2021)),
        daly_years=list(range(2010, 2020)))
