from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from tlo import Date, DateOffset, Module, Parameter, Property, Types, logging
from tlo.analysis.utils import (
    flatten_multi_index_series_into_dict_for_logging,
    get_counts_by_sex_and_age_group,
)
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.dxmanager import DxTest
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.hsi_generic_first_appts import GenericFirstAppointmentsMixin
from tlo.methods.symptommanager import Symptom
from tlo.util import read_csv_files

if TYPE_CHECKING:
    from tlo.methods.hsi_generic_first_appts import HSIEventScheduler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Schisto(Module, GenericFirstAppointmentsMixin):
    """Schistosomiasis module.
    Two species of worm that cause Schistosomiasis are modelled independently. Worms are acquired by persons via the
     environment. There is a delay between the acquisition of worms and the maturation to 'adults' worms; and a long
     period before the adult worms die. The number of worms in a person (whether a heavy-intensity infection or not)
     determines the symptoms they experience. These symptoms are associated with disability weights. There is no risk
     of death. Treatment can be provided to persons who present following the onset of symptoms. Mass Drug
     Administrations also give treatment to the general population, which clears any worm burden they have.

    N.B. Formal fitting has only been undertaken for: ('Blantyre', 'Chiradzulu', 'Mulanje', 'Nsanje', 'Nkhotakota',
    'Phalombe')."""

    INIT_DEPENDENCIES = {'Demography', 'SymptomManager'}

    OPTIONAL_INIT_DEPENDENCIES = {'HealthSystem', 'HealthBurden'}

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN,
        Metadata.REPORTS_DISEASE_NUMBERS
    }

    CAUSES_OF_DEATH = {}

    CAUSES_OF_DISABILITY = {
        'Schistosomiasis': Cause(gbd_causes='Schistosomiasis', label='Schistosomiasis'),
    }

    module_prefix = 'ss'

    PROPERTIES = {
        f'{module_prefix}_MDA_treatment_counter': Property(Types.INT,
                                                           'Counter for number of MDA treatments received '
                                                           'in logging interval')
    }

    PARAMETERS = {
        # these values do not vary between species
        'delay_till_hsi_a_repeated': Parameter(Types.REAL,
                                               'Time till seeking healthcare again after not being sent to '
                                               'schisto test: start'),
        'delay_till_hsi_b_repeated': Parameter(Types.REAL,
                                               'Time till seeking healthcare again after not being sent to '
                                               'schisto test: end'),
        'rr_WASH': Parameter(Types.REAL, 'proportional reduction in population susceptible to schistosoma '
                                         'infection with improved WASH'),
        'calibration_scenario': Parameter(Types.REAL,
                                          'Scenario used to reset parameters to run calibration sims'),
        'urine_filtration_sensitivity_noneWB': Parameter(Types.REAL,
                                                         'Sensitivity of urine filtration test for non-infected cases'),
        'urine_filtration_sensitivity_lowWB': Parameter(Types.REAL,
                                                        'Sensitivity of UF in detecting low WB'),
        'urine_filtration_sensitivity_moderateWB': Parameter(Types.REAL,
                                                             'Sensitivity of UF in detecting moderate WB'),
        'urine_filtration_sensitivity_heavyWB': Parameter(Types.REAL,
                                                         'Sensitivity of UF in detecting heavy WB'),
        'urine_filtration_specificity_noneWB': Parameter(Types.REAL,
                                                         'Specificity of urine filtration test'),
        'urine_filtration_specificity_lowWB': Parameter(Types.REAL,
                                                        'Specificity of UF in detecting low WB'),
        'urine_filtration_specificity_moderateWB': Parameter(Types.REAL,
                                                             'Specificity of UF in detecting moderate WB'),
        'urine_filtration_specificity_heavyWB': Parameter(Types.REAL,
                                                          'Specificity of UF in detecting heavy WB'),
        'kato_katz_sensitivity_lowWB': Parameter(Types.REAL,
                                                 'Sensitivity of Kato-Katz test for non-infected cases'),
        'kato_katz_sensitivity_moderateWB': Parameter(Types.REAL,
                                                      'Sensitivity of KK in detecting moderate WB'),
        'kato_katz_sensitivity_heavyWB': Parameter(Types.REAL,
                                                   'Sensitivity of KK in detecting heavy WB'),
        'kato_katz_specificity_lowWB': Parameter(Types.REAL,
                                                 'Sensitivity of Kato-Katz test for non-infected cases'),
        'kato_katz_specificity_moderateWB': Parameter(Types.REAL,
                                                      'Specificity of KK in detecting moderate WB'),
        'kato_katz_specificity_heavyWB': Parameter(Types.REAL,
                                                      'Specificity of KK in detecting moderate WB'),
        'scaleup_WASH': Parameter(Types.STRING,
                                  'Whether to scale-up WASH during simulation, pause fixes values at 2024 '
                                  'levels with no further improvement, continue allows historical trends to continue, '
                                  'scaleup switches everyone to having access to WASH'),
        'scaleup_WASH_start_year': Parameter(Types.INT,
                                             'Start date to scale-up WASH, years after sim start date'),
        'mda_coverage': Parameter(Types.REAL,
                                  'Coverage of future MDA activities, consistent across all'
                                  'target groups'),
        'mda_target_group': Parameter(Types.STRING,
                                      'Target group for future MDA activities, '
                                      'one of [PSAC_SAC, SAC, ALL]'),
        'mda_frequency_months': Parameter(Types.REAL,
                                          'Number of months between MDA activities'),
        'scaling_factor_baseline_risk': Parameter(Types.REAL,
                                                  'scaling factor controls how the background risk of '
                                                  'infection is adjusted based on the deviation of current prevalence '
                                                  'from baseline prevalence'),
        'baseline_risk': Parameter(Types.REAL,
                                   'number of worms applied as a baseline risk across districts to prevent '
                                   'fadeout, number is scaled by scaling_factor_baseline_risk'),
        'background_gamma': Parameter(Types.REAL,
                                      'controls how fast the cushion shrinks as national prevalence falls; '
                                      'larger values (e.g. 2.5–3.0) make it disappear sooner'),
        'background_rel': Parameter(Types.REAL,
                                    'a small proportional cushion (1% at baseline) applied to the human '
                                    'reservoir; keeps early declines smooth without creating an artificial floor'),
        'daly_weight_mild_schistosomiasis': Parameter(Types.REAL, 'daly weight assigned to mild '
                                                                  'schistosomiasis, both species'),
        'daly_weight_moderate_s_mansoni': Parameter(Types.REAL, 'daly weight assigned to moderate '
                                                                'S. mansoni'),
        'daly_weight_heavy_s_mansoni': Parameter(Types.REAL, 'daly weight assigned to heavy S. mansoni'),
        'daly_weight_moderate_s_haematobium': Parameter(Types.REAL, 'daly weight assigned to moderate '
                                                                    'S. haematobium'),
        'daly_weight_heavy_s_haematobium': Parameter(Types.REAL, 'daly weight assigned to heavy '
                                                                 'S. haematobium'),
        'MDA_coverage_historical': Parameter(Types.DATA_FRAME,
                                             'Probability of getting PZQ in the MDA for PSAC, SAC and Adults '
                                             'in historic rounds'),
        'odds_ratio_health_seeking_children_schisto_low': Parameter(
            Types.REAL, 'Odds ratio for health seeking in children with schisto low symptoms'),
        'odds_ratio_health_seeking_adults_schisto_low': Parameter(
            Types.REAL, 'Odds ratio for health seeking in adults with schisto low symptoms'),
        'single_district_calibration_number': Parameter(Types.INT,
                                                       'District number for single district calibration runs'),
        'single_district_calibration_name': Parameter(Types.STRING,
                                                        'District name for single district calibration runs'),
        'single_district_calibration_region': Parameter(Types.STRING,
                                                        'District region for single district calibration runs'),
        'mda_schedule_month': Parameter(Types.INT,
                                       'Month for scheduling MDA events'),
        'mda_schedule_day': Parameter(Types.INT,
                                     'Day for scheduling MDA events'),
        'minimum_baseline_prevalence': Parameter(Types.REAL,
                                                'Minimum baseline prevalence to prevent division by zero'),
        'prevalence_lower_bound': Parameter(Types.REAL,
                                           'Lower bound for prevalence clamping'),
        'prevalence_upper_bound': Parameter(Types.REAL,
                                           'Upper bound for prevalence clamping'),
        'infant_min_age': Parameter(Types.INT, 'Minimum age for Infant group'),
        'infant_max_age': Parameter(Types.INT, 'Maximum age for Infant group'),
        'psac_min_age': Parameter(Types.INT, 'Minimum age for PSAC group'),
        'psac_max_age': Parameter(Types.INT, 'Maximum age for PSAC group'),
        'sac_min_age': Parameter(Types.INT, 'Minimum age for SAC group'),
        'sac_max_age': Parameter(Types.INT, 'Maximum age for SAC group'),
        'adults_min_age': Parameter(Types.INT, 'Minimum age for Adults group'),
        'adults_max_age': Parameter(Types.INT, 'Maximum age for Adults group'),
        'mda_execute': Parameter(Types.BOOL, 'Whether to execute MDA events'),
        'single_district': Parameter(Types.BOOL, 'Whether to run simulation for a single district only'),
        'main_polling_frequency': Parameter(Types.INT, 'Polling freq main schisto event in months'),
        'worm_maturation_period_months': Parameter(Types.INT,
            'Time in months for juvenile worms to mature into adult worms'),
        'worm_death_check_frequency_years': Parameter(Types.INT,
            'Frequency in years to check for worm deaths'),
        'mda_appointment_window_months': Parameter(Types.INT,
            'Time window in months for MDA appointment scheduling'),
        'wash_scaleup_frequency_years': Parameter(Types.INT,
            'Frequency in years to check and apply WASH scale-up'),
        'recent_sanitation_window_years': Parameter(Types.INT,
            'Time window in years to consider sanitation acquisition as recent'),
        'avg_weight_psac_kg': Parameter(Types.REAL,
            'Average weight in kg for PSAC age group for drug dosing'),
        'avg_weight_sac_kg': Parameter(Types.REAL,
            'Average weight in kg for SAC age group for drug dosing'),
        'avg_weight_adult_kg': Parameter(Types.REAL,
            'Average weight in kg for adults for drug dosing'),
    }

    def __init__(self, name=None):
        super().__init__(name)

        # Create pointer that will be to dict of disability weights
        self.disability_weights = None

        # Create pointer that will be to the item_code for praziquantel
        self.item_codes_for_consumables_required = dict()

        # Create the instances of `SchistoSpecies` that will represent the two species being considered
        self.species = {_name: SchistoSpecies(self, name=_name) for _name in ('mansoni', 'haematobium')}

        # Add properties and parameters declared by each species:
        for _spec in self.species.values():
            self.PROPERTIES.update(_spec.get_properties())
            self.PARAMETERS.update(_spec.get_parameters())

        # Property names for infection_status of all species
        self.cols_of_infection_status = [_spec.infection_status_property for _spec in self.species.values()]

        self.districts = None

        # create future mda strategy
        self.prognosed_mda = None

        # Age-group mapper will be created in read_parameters after parameters are loaded
        self.age_group_mapper = None

    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        """Read parameters and register symptoms."""

        # Define districts that this module will operate in:
        self.districts = self.sim.modules['Demography'].districts  # <- all districts

        # Load parameters
        workbook = read_csv_files(Path(resourcefilepath) / 'ResourceFile_Schisto', files=None)
        self.parameters = self._load_parameters_from_workbook(workbook)

        # Create age-group mapper now that parameters are loaded
        s = pd.Series(index=range(1 + int(self.parameters['adults_max_age'])), data='object')
        age_groups = self._get_age_groups()
        for name, (low_limit, heavy_limit) in age_groups.items():
            if name != 'All':
                s.loc[(s.index >= low_limit) & (s.index <= heavy_limit)] = name
        self.age_group_mapper = s.to_dict()

        # check WASH scaleup specified correctly
        assert self.parameters['scaleup_WASH'] in ['pause', 'continue', 'scaleup']

        # load species-specific parameters
        for _spec in self.species.values():
            self.parameters.update(_spec.load_parameters_from_workbook(workbook))

        # Register symptoms
        self._register_symptoms()

        # create container for logging person-days infected
        index = pd.MultiIndex.from_product(
            [
                ['mansoni', 'haematobium'],  # species
                ['Infant', 'PSAC', 'SAC', 'Adults'],  # age_group
                ['Low-infection', 'Moderate-infection', 'Heavy-infection'],  # infection_level
                self.districts  # district
            ],
            names=['species', 'age_group', 'infection_level', 'district']
        )
        self.log_person_days = pd.DataFrame(0, index=index, columns=['person_days']).sort_index()

    def pre_initialise_population(self):
        """Do things before generating the population (but after read_parameters and any parameter updating)."""

        # Call `pre_initialise_population` for each `SchistoSpecies` helper module.
        for _spec in self.species.values():
            _spec.update_parameters_from_schisto_module()

    def initialise_population(self, population):
        """Set the property values for the initial population."""

        df = population.props
        p = self.parameters
        df.loc[df.is_alive, f'{self.module_prefix}_MDA_treatment_counter'] = 0

        # reset all to one district if doing calibration or test runs
        # choose district based on parameter (in Malawi, default Zomba district 19) as it has ~10% prev of both species
        if p['single_district']:
            district_num = int(p['single_district_calibration_number'])
            df['district_num_of_residence'] = pd.Categorical([district_num] * len(df),
                                                             categories=df['district_num_of_residence'].cat.categories)

            df['district_of_residence'] = pd.Categorical([p['single_district_calibration_name']] * len(df),
                                                         categories=df['district_of_residence'].cat.categories)
            df['region_of_residence'] = pd.Categorical([p['single_district_calibration_region']] * len(df),
                                                       categories=df['region_of_residence'].cat.categories)
        for _spec in self.species.values():
            _spec.initialise_population(population)

    def initialise_simulation(self, sim):
        """Get ready for simulation start."""
        p = self.parameters
        # Look-up DALY weights
        if 'HealthBurden' in self.sim.modules:
            self.disability_weights = self._get_disability_weight()

        # Look-up item codes for Praziquantel
        self.item_code_for_praziquantel = self._get_item_code_for_praziquantel(MDA=False)
        self.item_code_for_praziquantel_MDA = self._get_item_code_for_praziquantel(MDA=True)

        # define the Dx tests and consumables required
        self._get_consumables_for_dx()

        # schedule regular events
        sim.schedule_event(SchistoMatureJuvenileWormsEvent(self),
                          sim.date + pd.DateOffset(months=p['worm_maturation_period_months']))
        sim.schedule_event(SchistoWormDeathEvent(self),
                          sim.date + pd.DateOffset(years=p['worm_death_check_frequency_years']))

        # Initialise the simulation for each species
        for _spec in self.species.values():
            _spec.initialise_simulation(sim)

        # Schedule the logging event, annual, by district, age-group
        sim.schedule_event(SchistoLoggingEvent(self), sim.date + pd.DateOffset(years=1))
        sim.schedule_event(SchistoPersonDaysLoggingEvent(self), sim.date)

        # over-ride availability of PZQ for MDA, MDA cons is optional in HSI so will always run
        # self.sim.modules['HealthSystem'].override_availability_of_consumables(
        #     {1735: 1.0})  # this is the donated PZQ not currently in consumables availability worksheet
        # this is the tx PZQ
        # self.sim.modules['HealthSystem'].override_availability_of_consumables(
        #     {286: 1.0})

        # Schedule MDA events
        if p['mda_execute']:
            # update future mda strategy from default values
            self.prognosed_mda = self._create_mda_strategy()

            self._schedule_mda_events()

        # schedule WASH scale-up
        sim.schedule_event(SchistoWashScaleUp(self),
                          sim.date + pd.DateOffset(years=p['wash_scaleup_frequency_years']))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        All children are born without an infection, even if the mother is infected.

        :param mother_id: the ID for the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props
        df.at[child_id, f'{self.module_prefix}_MDA_treatment_counter'] = 0

        # WASH in action, update property li_unimproved_sanitation=False for all new births
        if (self.parameters['scaleup_WASH'] == 'scaleup') and (
                self.sim.date >= Date(int(self.parameters['scaleup_WASH_start_year']), 1, 1)):
            df.at[child_id, 'li_unimproved_sanitation'] = False
            df.at[child_id, 'li_no_clean_drinking_water'] = False
            df.at[child_id, 'li_no_access_handwashing'] = False

        for _spec in self.species.values():
            # this assigns infection_status, aggregate_worm_burden, harbouring_rate, susceptibility
            # if li_unimproved_sanitation=False, child susceptibility determined on current prop susceptible in district
            _spec.on_birth(mother_id, child_id)

    def report_daly_values(self):
        """Report the daly values, as the sum of the disability weight associated with each symptom caused by this
        module."""

        # Get the total weights for all those that have symptoms caused by this module.
        symptoms_being_caused = self.sim.modules['SymptomManager'].caused_by(self)

        def get_total_disability_weight(list_of_symptoms: list) -> float:
            """Returns the maximum disability weight from the list of schisto symptoms
            this avoids having unreasonably heavy DW if co-infected with heavy mansoni+haem infections """
            if not list_of_symptoms:
                return 0.0

            return max(
                (self.disability_weights.get(symptom, 0.0) for symptom in list_of_symptoms),
                default=0.0
            )

        disability_weights_for_each_person_with_symptoms = pd.Series(symptoms_being_caused).apply(
            get_total_disability_weight
        )

        # Return pd.Series that includes entries for all alive persons (filling 0.0 where they do not have any symptoms)
        df = self.sim.population.props
        return pd.Series(index=df.index[df.is_alive], data=0.0).add(
            disability_weights_for_each_person_with_symptoms, fill_value=0.0
        )

    def report_summary_stats(self):
        """Returns disease numbers by sex, age, species, and infection status."""
        return {
            f'number_with_any_infection_with_{spec_name}': get_counts_by_sex_and_age_group(
                self.sim.population.props,
                property=spec_obj.infection_status_property,
                targets=('Low-infection', 'High-infection')
            )
            for spec_name, spec_obj in self.species.items()
        }

    def do_effect_of_treatment(self, person_id: Union[int, Sequence[int]], mda=False) -> None:
        """Do the effects of a treatment administered to a person or persons. This can be called for a person who is
        infected and receiving treatment following a diagnosis, or for a person who is receiving treatment as part of a
         Mass Drug Administration. The burden and effects of any species are alleviated by a successful treatment."""

        p = self.parameters

        df = self.sim.population.props

        # Ensure person_id is treated as an iterable (e.g., list) even if it's a single integer
        if isinstance(person_id, int):
            person_id = [person_id]

        # Record the treatment
        if mda:
            df.loc[person_id, 'ss_MDA_treatment_counter'] += 1

        # Update properties after PZQ treatment
        for spec_prefix in [_spec.prefix for _spec in self.species.values()]:
            pzq_efficacy = p[f'{spec_prefix}_PZQ_efficacy']
            worm_burden_col = f'{self.module_prefix}_{spec_prefix}_aggregate_worm_burden'

            # reduce the worm burden
            df.loc[person_id, worm_burden_col] = (
                (df.loc[person_id, worm_burden_col] * (1 - pzq_efficacy))
                .clip(lower=0)
                .round()
                .astype(int)
            )

            self.update_infection_symptoms(df, worm_burden_col, f'{self.module_prefix}_{spec_prefix}')

    def _load_parameters_from_workbook(self, workbook) -> dict:
        """Load parameters from ResourceFile (loaded by pd.read_excel as `workbook`) that are general (i.e., not
        specific to a particular species)."""

        parameters = dict()

        # HSI and treatment params:
        param_list = workbook['parameter_values'].set_index("parameter_name")['value']

        def cast_param(v: str):
            """Try to cast a string value to the appropriate native type (int, float or bool).
            Return: the original value is not a string or if casting fails."""

            if not isinstance(v, str):
                return v

            s_lower = v.strip().lower()

            # intended to be a bool
            if s_lower == "true":
                return True
            elif s_lower == "false":
                return False

            # intended to be an int (handles + / -)
            elif s_lower.lstrip("+-").isdigit():
                return int(s_lower)

            # intended to be a float
            else:
                # try casting
                try:
                    return float(s_lower)
                except ValueError:
                    # return original value if casting to float fails
                    return v

        # parameters are all converted to strings if any strings are present
        for _param_name in (
            'delay_till_hsi_a_repeated',
            'delay_till_hsi_b_repeated',
            'rr_WASH',
            'calibration_scenario',
            'urine_filtration_sensitivity_lowWB',
            'urine_filtration_sensitivity_moderateWB',
            'urine_filtration_sensitivity_heavyWB',
            'urine_filtration_specificity_lowWB',
            'urine_filtration_specificity_moderateWB',
            'urine_filtration_specificity_heavyWB',
            'kato_katz_sensitivity_moderateWB',
            'kato_katz_specificity_moderateWB',
            'kato_katz_sensitivity_heavyWB',
            'kato_katz_specificity_heavyWB',
            'scaleup_WASH',  # Needs to be included
            'scaleup_WASH_start_year',
            'mda_coverage',
            'mda_target_group',  # Needs to be included
            'mda_frequency_months',
            'scaling_factor_baseline_risk',
            'baseline_risk',
            'background_gamma',
            'background_rel',
            'daly_weight_mild_schistosomiasis',
            'daly_weight_moderate_s_mansoni',
            'daly_weight_heavy_s_mansoni',
            'daly_weight_moderate_s_haematobium',
            'daly_weight_heavy_s_haematobium',
            'odds_ratio_health_seeking_children_schisto_low',
            'odds_ratio_health_seeking_adults_schisto_low',
            'kato_katz_sensitivity_lowWB',
            'kato_katz_specificity_lowWB',
            'urine_filtration_sensitivity_noneWB',
            'urine_filtration_specificity_noneWB',
            'single_district_calibration_number',
            'single_district_calibration_name',
            'single_district_calibration_region',
            'mda_schedule_month',
            'mda_schedule_day',
            'minimum_baseline_prevalence',
            'prevalence_lower_bound',
            'prevalence_upper_bound',
            'infant_min_age',
            'infant_max_age',
            'psac_min_age',
            'psac_max_age',
            'sac_min_age',
            'sac_max_age',
            'adults_min_age',
            'adults_max_age',
            'mda_execute',
            'single_district',
            'main_polling_frequency',
            'worm_maturation_period_months',
            'worm_death_check_frequency_years',
            'mda_appointment_window_months',
            'wash_scaleup_frequency_years',
            'recent_sanitation_window_years',
            'avg_weight_psac_kg',
            'avg_weight_sac_kg',
            'avg_weight_adult_kg'
        ):
            value = param_list[_param_name]
            parameters[_param_name] = value if isinstance(value, bool) else cast_param(value)

        # MDA coverage - historic
        # this is updated now with the EPSEN data
        historical_mda = workbook['ESPEN_MDA'].set_index(['District', 'Year'])[
            ['EpiCov_PSAC', 'EpiCov_SAC', 'EpiCov_Adults']]
        historical_mda.columns = historical_mda.columns.str.replace('EpiCov_', '')

        parameters['MDA_coverage_historical'] = historical_mda.astype(float)
        # clip upper limit of MDA coverage at the specified limit
        parameters['MDA_coverage_historical'] = parameters['MDA_coverage_historical'].clip(
            upper=0.99)

        return parameters

    def _get_age_groups(self) -> dict:
        """Create age groups dictionary from parameters."""
        p = self.parameters
        return {
            'Infant': (int(p['infant_min_age']), int(p['infant_max_age'])),
            'PSAC': (int(p['psac_min_age']), int(p['psac_max_age'])),
            'SAC': (int(p['sac_min_age']), int(p['sac_max_age'])),
            'Adults': (int(p['adults_min_age']), int(p['adults_max_age'])),
            'All': (int(p['infant_min_age']), int(p['adults_max_age']))
        }

    def _create_mda_strategy(self) -> pd.DataFrame:
        """ this uses the parameters set in the module to create a pd.DataFrame that contains the MDA strategy for
        future MDA activities. This will take effect the year after the last entry in
        parameters['MDA_coverage_historical']"""

        params = self.parameters
        coverage = params['mda_coverage']
        target = params['mda_target_group']
        frequency = params['mda_frequency_months']
        districts = self.districts

        # Create a new DataFrame with districts and schedule
        prognosed_mda = pd.DataFrame(
            index=pd.MultiIndex.from_product([districts, [frequency]], names=['District', 'Frequency_months']))

        # Initialise columns with default values
        prognosed_mda[['PSAC', 'SAC', 'Adults']] = 0

        # Define default values for each column
        default_values = {
            'PSAC': 0,
            'SAC': 0,
            'Adults': 0
        }

        # Define the updates based on the target
        updates = {
            'PSAC_SAC': {
                'PSAC': coverage,
                'SAC': coverage,
                'Adults': default_values['Adults']
            },
            'SAC': {
                'SAC': coverage,
                'PSAC': default_values['PSAC'],
                'Adults': default_values['Adults']
            },
            'ALL': {
                'PSAC': coverage,
                'SAC': coverage,
                'Adults': coverage
            }
        }

        # Get the appropriate update based on the target
        if target in updates:
            prognosed_mda.update(pd.DataFrame([updates[target]], index=prognosed_mda.index))

        return prognosed_mda

    def _register_symptoms(self) -> None:
        """Register the symptoms with the `SymptomManager`.
        :params symptoms: The symptoms that are used by this module in a dictionary of the form, {<symptom>:
        <generic_symptom_similar>}. Each symptom is associated with the average healthcare seeking behaviour
        unless otherwise specified."""

        # Declare symptoms that this module will cause and which are not included in the generic symptoms:
        p = self.parameters
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='schisto_low',
                    odds_ratio_health_seeking_in_children=p['odds_ratio_health_seeking_children_schisto_low'],
                    odds_ratio_health_seeking_in_adults=p['odds_ratio_health_seeking_adults_schisto_low'])
        )

        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='ss_sm_moderate'))

        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='ss_sm_heavy'))

        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='ss_sh_moderate'))

        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='ss_sh_heavy'))

    def _get_disability_weight(self) -> dict:
        """Return dict containing the disability weight (value) of each symptom (key)."""

        p = self.parameters

        return {
            "schisto_low": p['daly_weight_mild_schistosomiasis'],
            "ss_sm_moderate": p['daly_weight_moderate_s_mansoni'],
            "ss_sm_heavy": p['daly_weight_heavy_s_mansoni'],
            "ss_sh_moderate": p['daly_weight_moderate_s_haematobium'],
            "ss_sh_heavy": p['daly_weight_heavy_s_haematobium'],
        }

    def _get_item_code_for_praziquantel(self, MDA=False) -> int:
        """Look-up the item code for Praziquantel"""

        if MDA:
            # todo donated PZQ not currently in consumables availability sheet
            # return self.sim.modules['HealthSystem'].get_item_code_from_item_name("Praziquantel, 600 mg (donated)")
            return self.sim.modules['HealthSystem'].get_item_code_from_item_name("Praziquantel 600mg_1000_CMST")
        else:
            return self.sim.modules['HealthSystem'].get_item_code_from_item_name("Praziquantel 600mg_1000_CMST")

    def calculate_praziquantel_dosage(self, person_id):
        age = self.sim.population.props.at[person_id, "age_years"]

        # Dosing based on age group
        # 40mg per kg, as single dose (MSTG)
        # in children <4 years, 20mg/kg
        p = self.parameters
        if age <= p['psac_max_age']:
            dose = 20 * p['avg_weight_psac_kg']
        elif age >= p['sac_min_age'] and age <= p['sac_max_age']:
            dose = 40 * p['avg_weight_sac_kg']
        else:  # adults
            dose = 40 * p['avg_weight_adult_kg']

        return int(dose)

    def _get_consumables_for_dx(self):
        p = self.parameters
        hs = self.sim.modules["HealthSystem"]

        # diagnostic test consumables
        self.item_codes_for_consumables_required['malachite_stain'] = hs.get_item_code_from_item_name(
            "Malachite green oxalate")

        self.item_codes_for_consumables_required['iodine_stain'] = hs.get_item_code_from_item_name(
            "Iodine strong 10% solution_500ml_CMST")

        self.item_codes_for_consumables_required['microscope_slide'] = hs.get_item_code_from_item_name(
            "Microscope slides-frosted end(Tropical packaging)_50_CMST")

        self.item_codes_for_consumables_required['filter_paper'] = hs.get_item_code_from_item_name(
            "Paper filter Whatman no.1 size 10cm_100_CMST")

        # KATO-KATZ TEST
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            KK_schisto_test_lowWB=DxTest(
                property='ss_sm_infection_status',
                target_categories=["Non-infected", "Low-infection"],
                sensitivity=p['kato_katz_sensitivity_lowWB'],
                specificity=p['kato_katz_specificity_lowWB'],
                item_codes={self.item_codes_for_consumables_required['microscope_slide']: 2},
                optional_item_codes=[
                    self.item_codes_for_consumables_required['malachite_stain'],
                    self.item_codes_for_consumables_required['iodine_stain']]

            )
        )

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            KK_schisto_test_moderateWB=DxTest(
                property='ss_sm_infection_status',
                target_categories=["Moderate-infection"],
                sensitivity=p["kato_katz_sensitivity_moderateWB"],
                specificity=p['kato_katz_specificity_moderateWB'],
                item_codes={self.item_codes_for_consumables_required['microscope_slide']: 2},
                optional_item_codes=[
                    self.item_codes_for_consumables_required['malachite_stain'],
                    self.item_codes_for_consumables_required['iodine_stain']]
            )
        )

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            KK_schisto_test_heavyWB=DxTest(
                property='ss_sm_infection_status',
                target_categories=["Heavy-infection"],
                sensitivity=p["kato_katz_sensitivity_heavyWB"],
                specificity=p['kato_katz_specificity_heavyWB'],
                item_codes={self.item_codes_for_consumables_required['microscope_slide']: 2},
                optional_item_codes=[
                    self.item_codes_for_consumables_required['malachite_stain'],
                    self.item_codes_for_consumables_required['iodine_stain']]
            )
        )

        # URINE FILTRATION
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            UF_schisto_test_noWB=DxTest(
                property='ss_sh_infection_status',
                target_categories=["Non-infected"],
                sensitivity=p['urine_filtration_sensitivity_noneWB'],
                specificity=p['urine_filtration_specificity_noneWB'],
                item_codes={self.item_codes_for_consumables_required['microscope_slide']: 2},
                optional_item_codes=[
                    self.item_codes_for_consumables_required['filter_paper'],
                    self.item_codes_for_consumables_required['iodine_stain']]

            )
        )
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            UF_schisto_test_lowWB=DxTest(
                property='ss_sh_infection_status',
                target_categories=["Low-infection"],
                sensitivity=p["urine_filtration_sensitivity_lowWB"],
                specificity=p['urine_filtration_specificity_lowWB'],
                item_codes={self.item_codes_for_consumables_required['microscope_slide']: 2},
                optional_item_codes=[
                    self.item_codes_for_consumables_required['filter_paper'],
                    self.item_codes_for_consumables_required['iodine_stain']]

            )
        )

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            UF_schisto_test_moderateWB=DxTest(
                property='ss_sh_infection_status',
                target_categories=["Moderate-infection"],
                sensitivity=p["urine_filtration_sensitivity_moderateWB"],
                specificity=p['urine_filtration_specificity_moderateWB'],
                item_codes={self.item_codes_for_consumables_required['microscope_slide']: 2},
                optional_item_codes=[
                    self.item_codes_for_consumables_required['filter_paper'],
                    self.item_codes_for_consumables_required['iodine_stain']]
            )
        )

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            UF_schisto_test_heavyWB=DxTest(
                property='ss_sh_infection_status',
                target_categories=["Heavy-infection"],
                sensitivity=p["urine_filtration_sensitivity_heavyWB"],
                specificity=p['urine_filtration_specificity_heavyWB'],
                item_codes={self.item_codes_for_consumables_required['microscope_slide']: 2},
                optional_item_codes=[
                    self.item_codes_for_consumables_required['filter_paper'],
                    self.item_codes_for_consumables_required['iodine_stain']]
            )
        )

    def _schedule_mda_events(self) -> None:
        """Schedule MDA events, historical and prognosed."""

        # Schedule the  MDA that have occurred, in each district and in each year:
        for (district, year), cov in self.parameters['MDA_coverage_historical'].iterrows():
            assert district in self.sim.modules['Demography'].districts, f'District {district} is not recognised.'
            self.sim.schedule_event(
                SchistoMDAEvent(self,
                                district=district,
                                coverage=cov.to_dict(),
                                months_between_repeats=None),
                Date(year=year,
                     month=int(self.parameters['mda_schedule_month']),
                     day=int(self.parameters['mda_schedule_day']))
            )

        # Schedule the first occurrence of a future MDA in each district. It will occur after the last historical MDA.
        # The event that will schedule further instances of itself.
        year_last_historical_mda = self.parameters['MDA_coverage_historical'].reset_index().Year.max()
        year_first_simulated_mda = year_last_historical_mda + 1

        for (district, frequency_in_months), cov in self.prognosed_mda.iterrows():
            assert district in self.sim.modules['Demography'].districts, f'District {district} is not recognised.'
            self.sim.schedule_event(
                SchistoMDAEvent(self,
                                district=district,
                                coverage=cov.to_dict(),
                                months_between_repeats=frequency_in_months if frequency_in_months > 0 else None),
                Date(year=year_first_simulated_mda,
                     month=int(self.parameters['mda_schedule_month']),
                     day=int(self.parameters['mda_schedule_day']))
            )

    def get_infection_status(self, age, aggregate_worm_burden, species_prefix):
        """Find the correct infection intensity status given the age and aggregate worm burden """

        if species_prefix == 'ss_sm':
            params = self.sim.modules['Schisto'].species['mansoni'].params
        else:
            params = self.sim.modules['Schisto'].species['haematobium'].params

        p = self.parameters

        status = pd.Series("Non-infected", index=age.index, dtype="object")

        heavy_group = ((age <= p["psac_max_age"]) & (aggregate_worm_burden >=
            params["heavy_intensity_threshold_PSAC"])) | (aggregate_worm_burden >= params["heavy_intensity_threshold"])
        moderate_group = ~heavy_group & (aggregate_worm_burden >= params["low_intensity_threshold"])
        low_group = (aggregate_worm_burden < params["low_intensity_threshold"]) & (aggregate_worm_burden > 0)

        # assign status
        status[heavy_group] = "Heavy-infection"
        status[moderate_group] = "Moderate-infection"
        status[low_group] = "Low-infection"

        # same index as the original DataFrame
        return pd.Series(status)

    def update_infection_symptoms(self, df: pd.DataFrame, species_column_aggregate: str, species_prefix: str) -> None:

        correct_status = self.get_infection_status(df['age_years'], df[species_column_aggregate], species_prefix)
        original_status = df[f"{species_prefix}_infection_status"]
        is_alive = df['is_alive']
        symptom_list = ['schisto_low', 'ss_sm_moderate', 'ss_sm_heavy', 'ss_sh_moderate', 'ss_sh_heavy']

        # Clear symptoms for newly non-infected
        newly_non_infected = (correct_status == 'Non-infected') & (original_status != 'Non-infected') & is_alive
        if sum(newly_non_infected) > 0:
            idx_clear = df.index[newly_non_infected]
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=idx_clear,
                symptom_string=symptom_list,
                add_or_remove='-',
                disease_module=self.sim.modules['Schisto'])

        # Filter those with changed infection status and alive, excluding non-infected
        changed_mask = (correct_status != original_status) & (correct_status != 'Non-infected') & is_alive

        if sum(changed_mask) > 0:
            changed_idx = df.index[changed_mask]
            old_statuses = original_status[changed_idx]
            new_statuses = correct_status[changed_idx]

            # Map infection status to symptom string for additions
            new_symptom_map = {
                'Low-infection': "schisto_low",
                'Moderate-infection': f"{species_prefix}_moderate",
                'Heavy-infection': f"{species_prefix}_heavy"
            }

            # Build dicts for batch removal and addition
            remove_symptoms = {}
            add_symptoms = {}

            for person_id, old_status, new_status in zip(changed_idx, old_statuses, new_statuses):
                # Remove old symptom if previously infected
                if old_status in new_symptom_map:
                    old_symptom = new_symptom_map[old_status]
                    remove_symptoms.setdefault(old_symptom, []).append(person_id)

                # Add new symptom if newly infected / reinfected
                if new_status in new_symptom_map:
                    new_symptom = new_symptom_map[new_status]
                    add_symptoms.setdefault(new_symptom, []).append(person_id)

            # Batch remove old symptoms
            for symptom, persons in remove_symptoms.items():
                self.sim.modules['SymptomManager'].change_symptom(
                    person_id=persons,
                    symptom_string=symptom,
                    add_or_remove='-',
                    disease_module=self.sim.modules['Schisto']
                )

            # Batch add new symptoms
            for symptom, persons in add_symptoms.items():
                self.sim.modules['SymptomManager'].change_symptom(
                    person_id=persons,
                    symptom_string=symptom,
                    add_or_remove='+',
                    disease_module=self.sim.modules['Schisto']
                )

        # Update infection status column
        # this can sometimes return an object due to mixed types, eg string and None
        df[f"{species_prefix}_infection_status"] = pd.Categorical(correct_status,
                                                              dtype=df[f"{species_prefix}_infection_status"].dtype)

    def do_at_generic_first_appt(
        self,
        person_id: int,
        symptoms: List[str],
        schedule_hsi_event: HSIEventScheduler,
        **kwargs,
    ) -> None:
        # Do when person presents to the GenericFirstAppt.
        # If the person has certain set of symptoms, refer ta HSI for testing.
        # don't include low infection as symptoms likely very mild
        set_of_symptoms_indicative_of_schisto = {'ss_sm_moderate',
                                                 'ss_sm_heavy',
                                                 'ss_sh_moderate',
                                                 'ss_sm_heavy'}

        if set_of_symptoms_indicative_of_schisto.issubset(symptoms):
            event = HSI_Schisto_TestingFollowingSymptoms(
                module=self, person_id=person_id
            )
            schedule_hsi_event(event, priority=0, topen=self.sim.date)

    def select_test(self, person_id):
        """ choose the most likely test administered given the prevalent symptoms
        of this person"""

        # choose test
        persons_symptoms = self.sim.modules["SymptomManager"].has_what(person_id)
        if {'ss_sh_moderate', 'ss_sh_heavy'} & set(persons_symptoms):
            test = 'urine_filtration_test'
        else:
            test = 'kato-katz'

        return test

    def reduce_susceptibility(self, df, species_column):
        """
        Updates individual susceptibility to Schistosoma species following changes in WASH-related attributes,
        either through enhanced lifestyle interventions or WASH scale-up in the schisto module.

        Individuals newly gaining access to WASH experience a 40% reduction in susceptibility,
        such that 40% of them are no longer considered susceptible to schistosomiasis.
        """
        p = self.parameters

        # Find the number of individuals with currently susceptible to species and no sanitation
        # susceptible_no_sanitation = df.query(f"{species_column} == 1 and li_unimproved_sanitation").index
        recent_sanitation = df['li_date_acquire_improved_sanitation'] >= (
            self.sim.date - pd.DateOffset(years=p['recent_sanitation_window_years']))

        # Restrict to those who are susceptible, had no sanitation before, and recently acquired sanitation
        condition = (df[species_column] == 1) & df['li_unimproved_sanitation'] & recent_sanitation
        susceptible_no_sanitation = df.index[condition]

        # Calculate the number to be reduced
        n_to_reduce = int(p['rr_WASH'] * len(susceptible_no_sanitation))

        if n_to_reduce > 0:
            selected_change_susceptibility = np.random.choice(susceptible_no_sanitation, size=n_to_reduce,
                                                              replace=False)
            df.loc[selected_change_susceptibility, species_column] = 0


class SchistoSpecies:
    """Helper Class to hold the information specific to a particular species (either S. mansoni or S. haematobium)."""

    def __init__(self, schisto_module, name):
        self.schisto_module = schisto_module
        assert name in ('mansoni', 'haematobium')
        self.name = name.lower()

        # Store prefix for this species
        self.prefix = 's' + self.name[0]

        # Store parameters specific to this species (for ease of access)
        self.params = dict()

    def get_parameters(self):
        """The species-specific parameters for this species."""
        params = {
            'R0': Parameter(Types.REAL, 'R0 of species'),
            'beta_PSAC': Parameter(Types.REAL, 'Contact/exposure rate of PSAC'),
            'beta_SAC': Parameter(Types.REAL, 'Contact/exposure rate of SAC'),
            'beta_Adults': Parameter(Types.REAL, 'Contact/exposure rate of Adults'),
            'worms_fecundity': Parameter(Types.REAL, 'Fecundity parameter, driving density-dependent reproduction'),
            'worm_lifespan': Parameter(Types.REAL, 'Lifespan of the worm in human host given in years'),
            'heavy_intensity_threshold': Parameter(Types.REAL,
                                                  'Threshold of worm burden indicating heavy intensity infection'),
            'low_intensity_threshold': Parameter(Types.REAL,
                                                 'Threshold of worm burden indicating low intensity infection'),
            'heavy_intensity_threshold_PSAC': Parameter(Types.REAL,
                                                       'Worm burden threshold for heavy intensity infection in PSAC'),
            'PZQ_efficacy': Parameter(Types.REAL,
                                      'Efficacy of praziquantel in reducing worm burden'),
            'baseline_prevalence': Parameter(Types.REAL,
                                             'Baseline prevalence of species across all districts in 2010'),
            'mean_worm_burden2010': Parameter(Types.DATA_FRAME,
                                              'Mean worm burden per infected person per district in 2010'),
            'prevalence_2010': Parameter(Types.DATA_FRAME,
                                         'Prevalence per district in 2010'),
            'prop_susceptible': Parameter(Types.DATA_FRAME,
                                          'Proportion of population in each district susceptible to schisto infection'),
            'gamma_alpha': Parameter(Types.DATA_FRAME, 'Parameter alpha for Gamma distribution for harbouring rates'),
        }
        return {self._prefix_species_parameter(k): v for k, v in params.items()}

    def get_properties(self):
        """The species-specific properties for this species."""
        properties = {
            'infection_status': Property(
                Types.CATEGORICAL, 'Current status of schistosomiasis infection for this species',
                categories=['Non-infected', 'Low-infection', 'Moderate-infection', 'Heavy-infection']),
            'aggregate_worm_burden': Property(
                Types.INT, 'Number of mature worms of this species in the individual'),
            'juvenile_worm_burden': Property(
                Types.INT, 'Number of juvenile worms of this species in the individual'),
            'juvenile_worm_infection_date': Property(
                Types.DATE, 'Date at which infection with juvenile worms occurred'),
            'susceptibility': Property(
                Types.INT, 'Binary value 0,1 denoting whether person is susceptible or not'),
            'harbouring_rate': Property(
                Types.REAL, 'Rate of harbouring new worms of this species (Poisson), drawn from gamma distribution'),
        }
        return {self.prefix_species_property(k): v for k, v in properties.items()}

    def prefix_species_property(self, generic_property_name: str) -> str:
        """Add the prefix to a `generic_property_name` to get the name of the species-specific property for this
        species."""
        return f"{self.schisto_module.module_prefix}_{self.prefix}_{generic_property_name}"

    def _prefix_species_parameter(self, generic_parameter_name: str) -> str:
        """Add the prefix to a `generic_parameter_name` to get the name of the species-specific parameter for this
        species."""
        return f"{self.prefix}_{generic_parameter_name}"

    @property
    def infection_status_property(self):
        """Return the property that identifies the infection_status of the person with respect to this species."""
        return self.prefix_species_property('infection_status')

    def load_parameters_from_workbook(self, workbook) -> dict:
        """Load parameters from ResourceFile (loaded by pd.read_excel as `workbook`) that are specific to this
        species."""
        parameters = dict()

        # Natural history params
        param_list = workbook['parameter_values'].set_index("parameter_name")['value']
        for _param_name in ('R0',
                            'beta_PSAC',
                            'beta_SAC',
                            'beta_Adults',
                            'worm_lifespan',
                            'worms_fecundity',
                            'heavy_intensity_threshold',
                            'low_intensity_threshold',
                            'heavy_intensity_threshold_PSAC',
                            'PZQ_efficacy',
                            'baseline_prevalence',
                            ):
            parameters[_param_name] = float(param_list[f'{_param_name}_{self.name}'])

        # Baseline reservoir size and other district-related params (R0, proportion susceptible)
        schisto_initial_reservoir = workbook[f'LatestData_{self.name}'].set_index("District")
        parameters['mean_worm_burden2010'] = schisto_initial_reservoir['mean_worm_burden2022']
        parameters['prevalence_2010'] = schisto_initial_reservoir['mean_prevalence2010']
        parameters['gamma_alpha'] = schisto_initial_reservoir['gamma_alpha']
        parameters['prop_susceptible'] = schisto_initial_reservoir['prop_susceptible']

        return {self._prefix_species_parameter(k): v for k, v in parameters.items()}

    def pre_initialise_population(self):
        """Do things before generating the population (but after read_parameters) and any parameter updating."""

        # Save species-specific parameter in this class, copying from the `Schisto` module. (We have to do this step
        # because the module may have updated the parameters).
        self.update_parameters_from_schisto_module()

    def initialise_population(self, population):
        """Set species-specific property values for the initial population."""

        df = population.props
        prop = self.prefix_species_property

        # assign aggregate_worm_burden (zero for everyone initially)
        df.loc[df.is_alive, prop('aggregate_worm_burden')] = 0
        df.loc[df.is_alive, prop('juvenile_worm_burden')] = 0
        df.loc[df.is_alive, prop('juvenile_worm_infection_date')] = pd.NaT
        df.loc[df.is_alive, prop('infection_status')] = 'Non-infected'

        # assign a harbouring rate
        self._assign_initial_properties(population)

        # assign initial worm burden
        self._assign_initial_worm_burden(population)

        # update infection status and symptoms
        self.schisto_module.update_infection_symptoms(df, prop('aggregate_worm_burden'), f'ss_{self.prefix}')

    def initialise_simulation(self, sim):
        """
        * Schedule natural history events for those with worm burden initially.
        * Schedule the WormBurdenEvent for this species. (A recurring instance of this event will be scheduled for
        each species independently.)"""

        p = self.schisto_module.parameters
        sim.schedule_event(
            SchistoInfectionWormBurdenEvent(
                module=self.schisto_module,
                species=self),
            sim.date + DateOffset(months=p['main_polling_frequency'])
        )

    def on_birth(self, mother_id, child_id):
        """Initialise the species-specific properties for a newborn individual.
        :param mother_id: the ID for the mother for this child
        :param child_id: the new child"""

        df = self.schisto_module.sim.population.props
        prop = self.prefix_species_property
        params = self.params
        global_params = self.schisto_module.parameters
        rng = self.schisto_module.rng

        # Assign the default for a newly born child
        df.at[child_id, prop('infection_status')] = 'Non-infected'
        df.at[child_id, prop('aggregate_worm_burden')] = 0
        df.at[child_id, prop('juvenile_worm_burden')] = 0
        df.at[child_id, prop('juvenile_worm_infection_date')] = pd.NaT

        # Generate the harbouring rate depending on a district of residence.
        district = df.at[child_id, 'district_of_residence']
        df.at[child_id, prop('harbouring_rate')] = rng.gamma(params['gamma_alpha'][district], size=1)

        # Determine if individual should be susceptible to each species
        # susceptibility depends on district
        # Get base susceptibility for the child's district
        prop_susceptible = params['prop_susceptible'][df.at[child_id, 'district_of_residence']]

        # Adjust for sanitation status
        if df.at[child_id, 'li_unimproved_sanitation']:  # if true, child has full risk of susceptibility
            susceptibility_probability = prop_susceptible
        else:  # if false, child has access to improved sanitation so reduce risk
            susceptibility_probability = prop_susceptible * global_params['rr_WASH']

        # Draw from Bernoulli to determine susceptibility
        df.at[child_id, prop('susceptibility')] = 1 if rng.random_sample() < susceptibility_probability else 0

    def update_parameters_from_schisto_module(self) -> None:
        """Update the internally-held parameters from the `Schisto` module that are specific to this species."""

        def remove_prefix(text, prefix):
            return text[text.startswith(prefix) and len(prefix):]

        self.params = {
            remove_prefix(k, f"{self.prefix}_"): v for k, v in self.schisto_module.parameters.items()
            if k.startswith(self.prefix)
        }

    def _assign_initial_properties(self, population) -> None:
        """Assign a harbouring rate and susceptibility to every individual in the initial population
        (based on their district of residence)."""
        df = population.props
        prop = self.prefix_species_property
        params = self.params
        districts = self.schisto_module.districts
        rng = self.schisto_module.rng

        for district in districts:
            # in_the_district = df.index[df['district_of_residence'] == district]  # people in the district
            district_mask = df['district_of_residence'] == district

            # num_in_district = len(in_the_district)  # population size in district
            num_in_district = district_mask.sum()

            # HARBOURING RATE
            hr = params['gamma_alpha'][district]

            df.loc[district_mask, prop('harbouring_rate')] = rng.gamma(hr, size=district_mask.sum())

            # SUSCEPTIBILITY
            # Calculate the number of people that need to be susceptible
            prop_susceptible = params['prop_susceptible'][district]

            # the total number needed to fill the proportion susceptible in district
            n_susceptible = int(np.ceil(prop_susceptible * num_in_district))

            # Select people with li_unimproved_sanitation=True
            no_sanitation = df.loc[(df['district_of_residence'] == district) & df['li_unimproved_sanitation']]

            # Determine the number of people to select from those with no sanitation
            n_no_sanitation = min(n_susceptible, len(no_sanitation))

            # Assign susceptibility=1 to the selected people with no sanitation
            susceptible_idx = rng.choice(no_sanitation.index, size=n_no_sanitation, replace=False)
            df.loc[susceptible_idx, prop('susceptibility')] = 1

            # Update the number of susceptible people still needed
            n_susceptible_remaining = n_susceptible - n_no_sanitation

            if n_susceptible_remaining > 0:
                # Select additional people from those with li_unimproved_sanitation=False if needed
                with_sanitation = df.loc[
                    (df['district_of_residence'] == district) & ~df['li_unimproved_sanitation']]
                susceptible_additional_idx = rng.choice(with_sanitation.index, size=n_susceptible_remaining,
                                                        replace=False)
                df.loc[susceptible_additional_idx, prop('susceptibility')] = 1

    def _assign_initial_worm_burden(self, population) -> None:
        """Assign initial distribution of worms to each person (based on district and age-group)."""
        df = population.props
        prop = self.prefix_species_property
        params = self.params  # these are species-specific
        districts = self.schisto_module.districts
        rng = self.schisto_module.rng

        for district in districts:
            in_the_district = df.index[df['district_of_residence'] == district]

            # get reservoir in district
            reservoir = int(len(in_the_district) * params['mean_worm_burden2010'][district])

            # Determine a 'contact rate' for each person
            contact_and_susceptibility = df.loc[in_the_district, prop('susceptibility')]

            age_groups = self.schisto_module._get_age_groups()
            for age_group in ['PSAC', 'SAC', 'Adults']:
                age_range = age_groups[age_group]
                in_the_district_and_age_group = \
                    df.index[(df['district_of_residence'] == district) &
                             (df['age_years'].between(age_range[0], age_range[1]))]
                contact_and_susceptibility.loc[in_the_district_and_age_group] *= params[f"beta_{age_group}"]

            if len(in_the_district):
                harbouring_rates = df.loc[in_the_district, prop('harbouring_rate')].values
                rates = np.multiply(harbouring_rates, contact_and_susceptibility)

                # Distribute a worm burden among persons, according to their 'contact rate'
                if (reservoir > 0) and (rates.sum() > 0):
                    chosen = rng.choice(in_the_district, reservoir, p=rates / rates.sum())
                    unique, counts = np.unique(chosen, return_counts=True)
                    worms_per_idx = dict(zip(unique, counts))
                    df[prop('aggregate_worm_burden')].update(pd.Series(worms_per_idx))

    def log_infection_status(self) -> None:
        """Log the number of persons in each infection status for this species,
        by age-group, infection level and district."""

        df = self.schisto_module.sim.population.props

        # Directly filter and map values in one go without creating intermediate DataFrames
        age_grp = df.loc[df.is_alive, 'age_years'].map(self.schisto_module.age_group_mapper)

        # Perform the grouping and size computation in a single step
        data = df.query('is_alive').groupby(
            by=[
                df[self.infection_status_property],
                df['district_of_residence'],
                age_grp
            ],
            observed=False
        ).size()

        # Rename the index after grouping
        data.index.rename('infection_status', level=0, inplace=True)

        logger.info(
            key=f'infection_status_{self.name}',
            data=flatten_multi_index_series_into_dict_for_logging(data),
            description='Counts of infection status with this species by age-group and district.'
        )

        #  Susceptibility
        # reinstate if wanting to check susceptibility across districts
        # Directly filter and group in one step to avoid intermediate DataFrames
        # grouped_data = df[df.is_alive].groupby('district_of_residence')[prop('susceptibility')].agg(
        #     total_count='count',
        #     susceptible_count=lambda x: (x == 1).sum()
        # )

        # Calculate the proportion of susceptible individuals in each district
        # susceptibility_proportion = pd.Series(grouped_data['susceptible_count'] / grouped_data['total_count'])

        # logger.info(
        #     key=f'susceptibility_{self.name}',
        #     data=flatten_multi_index_series_into_dict_for_logging(susceptibility_proportion),
        #     description='Proportion of people susceptible to this species in district.'
        # )

    def log_mean_worm_burden(self) -> None:
        """Log the mean worm burden across the population for this species, by age-group and district."""

        df = self.schisto_module.sim.population.props

        prop = self.prefix_species_property

        # Apply the transformation and aggregation without unnecessary intermediate DataFrames
        data = (
            df.loc[df.is_alive, ['age_years', 'district_of_residence', prop('aggregate_worm_burden')]]
            .assign(age_group=df.loc[df.is_alive, 'age_years'].map(self.schisto_module.age_group_mapper))
            .groupby(['district_of_residence', 'age_group'], observed=False)[prop('aggregate_worm_burden')]
            .mean()
        )

        logger.info(
            key=f'mean_worm_burden_by_age_{self.name}',
            data=flatten_multi_index_series_into_dict_for_logging(data),
            description='Mean worm burden of this species by age-group and district.'
        )

        overall_mean = df.loc[df.is_alive].groupby(
            'district_of_residence'
        )[prop('aggregate_worm_burden')].mean()

        logger.info(
            key=f'mean_worm_burden_by_district_{self.name}',
            data=flatten_multi_index_series_into_dict_for_logging(overall_mean),
            description='Mean worm burden of this species by age-group and district.'
        )


class SchistoInfectionWormBurdenEvent(RegularEvent, PopulationScopeEventMixin):
    """A recurring event that causes infection of people with this species.
     * Determines who becomes infected using worm burden and reservoir of infectious material.
    """

    def __init__(self, module: Module, species: SchistoSpecies):
        p = module.parameters
        super().__init__(module, frequency=DateOffset(months=p['main_polling_frequency']))
        self.species = species

    def apply(self, population):
        df = population.props
        params = self.species.params
        p = self.module.parameters
        global_params = self.module.parameters
        rng = self.module.rng
        # prop calls the property starting with the prefix species property, i.e. ss_sm or ss_sh
        prop = self.species.prefix_species_property

        # --------------------- get exposure rates for each age-group ---------------------

        # betas (exposure rates) are fixed for each age-group
        # exposure rates determine contribution to transmission and acquisition risk
        betas = [params['beta_PSAC'], params['beta_SAC'], params['beta_Adults']]
        # R0 is district-specific and fixed
        R0 = params['R0']

        where = df.is_alive
        age_group = pd.cut(df.loc[where, 'age_years'],
                           [0, p['psac_max_age'], p['sac_max_age'], p['adults_max_age']],
                           labels=['PSAC', 'SAC', 'Adults'],
                           include_lowest=True)
        age_group = age_group.astype('category')  # Convert to a categorical type for memory efficiency
        age_group.name = 'age_group'

        beta_by_age_group = pd.Series(betas, index=['PSAC', 'SAC', 'Adults'])
        beta_by_age_group.index.name = 'age_group'

        # --------------------- get the size of reservoir per district ---------------------
        # returns the mean worm burden and the total worm burden by age and district
        mean_count_burden_district_age_group = \
            df.loc[where, [prop('aggregate_worm_burden'), 'district_of_residence']].groupby(
                [df.loc[where, 'district_of_residence'], age_group], observed=False
            )[prop('aggregate_worm_burden')].agg(['mean', 'size'])

        # get population size by district
        district_count = df[where].groupby('district_of_residence').size()

        # mean worm burden by age multiplied by exposure rates of each age-gp
        beta_contribution_to_reservoir = mean_count_burden_district_age_group['mean'] * beta_by_age_group
        # weighted mean of the worm burden, considering the size of each age-gp in district
        to_get_weighted_mean = mean_count_burden_district_age_group['size'] / district_count

        # weighted mean worm burden * exposure rates -> age-specific contribution to reservoir
        age_worm_burden = beta_contribution_to_reservoir * to_get_weighted_mean
        # sum all contributions to district reservoir of infection
        reservoir = age_worm_burden.groupby(['district_of_residence'], observed=False).sum()

        # --------------------- reservoir-stage background (prevalence-based, fading) ---------------------
        prevalence_now = (df.loc[where, prop('aggregate_worm_burden')] > 0).mean()  # prevalence (Bool True/# entries)

        # Clamp to safe bounds
        baseline_prevalence = max(params['baseline_prevalence'],
                                global_params['minimum_baseline_prevalence'])  # fixed reference prevalence
        prevalence_now = min(max(prevalence_now, global_params['prevalence_lower_bound']),
                           global_params['prevalence_upper_bound'])

        # Scale in [0,1]: 1 at baseline; → 0 as national prevalence → 0
        scale = min(1.0, (prevalence_now / baseline_prevalence) ** global_params['background_gamma'])

        # Apply proportional background; disappears as scale→0 (no hard floor)
        reservoir *= (1.0 + global_params['background_rel'] * scale)

        # --------------------- harbouring new worms ---------------------

        # the harbouring rates are randomly assigned to each individual
        # using a gamma distribution to reflect clustering of worms in high-risk people
        # this is not age-specific
        contact_rates = age_group.map(beta_by_age_group).astype(float)
        # multiply by susceptibility
        contact_rates = contact_rates * df.loc[where, prop('susceptibility')]

        harbouring_rates = df.loc[where, prop('harbouring_rate')]
        rates = harbouring_rates * contact_rates
        worms_total = reservoir * R0

        draw_worms = pd.Series(
            rng.poisson(
                (
                    df.loc[where, 'district_of_residence']
                    .map(worms_total)
                    .astype(float)  # Ensure compatibility for multiplication
                    * rates
                ).fillna(0.0)
            ),
            index=df.index[where]
        )

        # density dependent establishment of new worms
        # establishment of new worms dependent on number of worms currently in host * worm fecundity
        # limits numbers of worms harboured by each individual
        param_worm_fecundity = params['worms_fecundity']
        established = self.module.rng.random_sample(size=sum(where)) < np.exp(
            df.loc[where, prop('aggregate_worm_burden')] * -param_worm_fecundity
        )
        to_establish = draw_worms[(draw_worms > 0) & established].to_dict()

        df.loc[to_establish.keys(), prop('juvenile_worm_burden')] = pd.Series(to_establish)
        df.loc[to_establish.keys(), prop('juvenile_worm_infection_date')] = self.sim.date


class SchistoMatureJuvenileWormsEvent(RegularEvent, PopulationScopeEventMixin):
    """A recurring event that:
     * Matures the juvenile worms into adult worms
     """
    def __init__(self, module):
        p = module.parameters
        super().__init__(
            module, frequency=DateOffset(months=p['worm_maturation_period_months'])
        )

    def apply(self, population):

        df = population.props

        def juvenile_worms_to_adults(df, species_column_juvenile, species_column_aggregate,
                                     juvenile_infection_date, species_prefix):
            """
            moves the juveniles worms into the aggregate_worm_burden property
            indicating that they are now mature and will contribute to the disability threshold
            then clears the column containing juvenile worm numbers ready for next infection event
            this is called separately for each species
            """
            # all new juvenile infections will have same infection date
            p = self.module.parameters
            if (df[juvenile_infection_date] <= self.sim.date -
                    pd.DateOffset(months=p['worm_maturation_period_months'])).any():
                df[species_column_aggregate] += df[species_column_juvenile]

                # Set 'juvenile' column to zeros
                df[species_column_juvenile] = 0
                df[juvenile_infection_date] = pd.NaT  # clear the infection date

                self.module.update_infection_symptoms(df, species_column_aggregate, species_prefix)

                # update_infectious_status_and_symptoms(df.index[df.is_alive], species=species)

        juvenile_worms_to_adults(df, 'ss_sm_juvenile_worm_burden',
                                 'ss_sm_aggregate_worm_burden',
                                 'ss_sm_juvenile_worm_infection_date',
                                 species_prefix='ss_sm')
        juvenile_worms_to_adults(df, 'ss_sh_juvenile_worm_burden',
                                 'ss_sh_aggregate_worm_burden',
                                 'ss_sh_juvenile_worm_infection_date',
                                 species_prefix='ss_sh')


class SchistoWormDeathEvent(RegularEvent, PopulationScopeEventMixin):
    """A recurring event that:
     * Kills any adult worms according to species-specific lifespan
     """

    def __init__(self, module):
        p = module.parameters
        super().__init__(
            module, frequency=DateOffset(years=p['worm_death_check_frequency_years'])
        )

    def apply(self, population):

        df = population.props
        mansoni_params = self.sim.modules['Schisto'].species['mansoni'].params
        haematobium_params = self.sim.modules['Schisto'].species['haematobium'].params

        def update_symptoms_after_worm_death(df, species_column_aggregate, worm_lifespan, species_prefix):
            """
            Kills a proportion of adult worms and updates symptoms based on new infection intensity.
            Clears symptoms only if person is now non-infected.
            """

            # Kill proportion of adult worms
            decay_fraction = 1 - 1 / worm_lifespan

            df[species_column_aggregate] = (
                (df[species_column_aggregate] * decay_fraction)
                .clip(lower=0)
                .round()
                .astype(int)
            )

            self.module.update_infection_symptoms(df, species_column_aggregate, species_prefix)

        update_symptoms_after_worm_death(df, 'ss_sm_aggregate_worm_burden',
                                         mansoni_params['worm_lifespan'], 'ss_sm')
        update_symptoms_after_worm_death(df, 'ss_sh_aggregate_worm_burden',
                                         haematobium_params['worm_lifespan'],'ss_sh')


class SchistoMDAEvent(Event, PopulationScopeEventMixin):
    """Mass-Drug administration scheduled for the population. This event schedules the occurrence of the
    individual-level HSIs for the administration of drugs to each individual.
    :param district: The district in which the MDA occurs.
    :param coverage: A dictionary of the form {<age_group>: <coverage>}, where <age_group> is one of ('PSAC', 'SAC',
     'Adults').
    :params months_between_repeat: The number of months between repeated occurrences of this event. (None for no
     repeats)."""

    def __init__(self, module: Module, district: str, coverage: dict, months_between_repeats: Optional[int]):
        super().__init__(module)
        assert isinstance(module, Schisto)

        self.district = district
        self.coverage = coverage
        self.months_between_repeats = months_between_repeats

    def apply(self, population):
        """ Represents the occurence of an MDA, in a particular year and district, which achieves a particular coverage
         (by age-group).
         * Schedules the MDA HSI for each person that is reached in the MDA.
         * Schedules the recurrence of this event, if the MDA is to be repeated in the future."""

        # Determine who receives the MDA
        age_included = [key for key, value in self.coverage.items() if value != 0]
        idx_to_receive_mda = []
        for age_group, cov in self.coverage.items():
            idx_to_receive_mda.extend(
                self._select_recipients(district=self.district, age_group=age_group, coverage=cov))

        # Schedule the MDA HSI. This HSI will do the work for all the `person_id`s in `idx_to_receive_mda`, but
        # the HSI's argument `person_id` is attached only to the one of these people. This is to avoid the inefficiency
        # of multiple individual HSI being created that do the same thing and occur on the same day and in the same
        # facility. The limitation is that if this person dies then no one gets the HSI.
        # This is discussed in https://github.com/UCL/TLOmodel/issues/531
        if idx_to_receive_mda:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_Schisto_MDA(
                    self.module,
                    person_id=idx_to_receive_mda[0],
                    beneficiaries_ids=idx_to_receive_mda,
                    age_group_included=age_included,
                ),
                topen=self.sim.date,
                tclose=self.sim.date + pd.DateOffset(months=self.module.parameters['mda_appointment_window_months']),
                priority=2
                # A long time-window of operation and a low priority is used for this MDA Appointment, to represent
                # that the MDA would not take a priority over other appointments.
            )

        # Schedule the recurrence of this event, if the MDA is to be repeated in the future.
        if self.months_between_repeats is not None:
            self.sim.schedule_event(self, self.sim.date + pd.DateOffset(months=self.months_between_repeats))

    def _select_recipients(self, district, age_group, coverage) -> list:
        """Determine persons to receive MDA, based on a specified target age-group and coverage."""

        assert 0.0 <= coverage <= 1.0, f'Value of coverage {coverage} is out of bounds.'
        assert age_group in ('PSAC', 'SAC', 'Adults')

        df = self.sim.population.props
        rng = self.module.rng

        age_groups = self.module._get_age_groups()
        age_range = age_groups[age_group]  # returns a tuple (a,b) a <= age_group <= b

        eligible = df.index[
            df['is_alive']
            & (df['district_of_residence'] == district)
            & df['age_years'].between(age_range[0], age_range[1])
            ]

        return eligible[rng.random_sample(len(eligible)) < coverage].to_list()


class SchistoWashScaleUp(RegularEvent, PopulationScopeEventMixin):
    """
    This has two functions:
    *1 update the susceptibility of individuals if their WASH properties have changed
    *2 change WASH properties if scale-up WASH scenario is implemented

    When WASH is implemented, two processes will occur:
    *1 scale the proportion of the population susceptible to schisto infection
    assuming that WASH reduces individual risk of infection by 0.6
    *2 increase the proportion of the population who have access to
    sanitation and clean drinking water

    Event is initially scheduled by initialise_simulation on specified date
    """

    def __init__(self, module):
        p = module.parameters
        super().__init__(
            module, frequency=DateOffset(years=p['wash_scaleup_frequency_years'])
        )

    def apply(self, population):
        df = population.props
        p = self.module.parameters

        if (p['scaleup_WASH'] == 'scaleup') & (self.sim.date.year == p['scaleup_WASH_start_year']):

            # scale-up properties related to WASH
            # set the properties to False for everyone
            df['li_unimproved_sanitation'] = False
            df['li_no_clean_drinking_water'] = False
            df['li_no_access_handwashing'] = False
            df['li_date_acquire_improved_sanitation'] = self.sim.date
            df['li_date_acquire_access_handwashing'] = self.sim.date
            df['li_date_acquire_clean_drinking_water'] = self.sim.date

        # wash improvements being paused
        if (p['scaleup_WASH'] == 'pause') & (self.sim.date.year >= p['scaleup_WASH_start_year']):
            self.sim.modules['Lifestyle'].parameters['r_improved_sanitation'] = 0
            self.sim.modules['Lifestyle'].parameters['r_clean_drinking_water'] = 0
            self.sim.modules['Lifestyle'].parameters['r_access_handwashing'] = 0

        # access to WASH constantly changing through lifestyle module
        # need to reduce proportion susceptible by 60% for both species
        # Reduce susceptibility for mansoni
        self.module.reduce_susceptibility(df, species_column='ss_sm_susceptibility')

        # Reduce susceptibility for haematobium
        self.module.reduce_susceptibility(df, species_column='ss_sh_susceptibility')


class HSI_Schisto_TestingFollowingSymptoms(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event for a person with symptoms who has been referred from the FirstAppt
    for testing at the clinic."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Schisto)

        self.TREATMENT_ID = 'Schisto_Test'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'LabParasit': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self._num_occurrences = 0

    def apply(self, person_id, squeeze_factor):
        self._num_occurrences += 1

        df = self.sim.population.props
        params = self.module.parameters

        # select and perform the appropriate diagnostic test
        test = self.module.select_test(person_id)
        test_result = None

        # perform the test
        if test == 'urine_filtration_test':

            # sensitivity of test depends worm burden
            if df.at[person_id, 'ss_sh_infection_status'] == 'Non-infected':
                test_result = self.sim.modules[
                    "HealthSystem"
                ].dx_manager.run_dx_test(
                    dx_tests_to_run="UF_schisto_test_noWB",
                    hsi_event=self
                )

            elif df.at[person_id, 'ss_sh_infection_status'] == 'Low-infection':
                test_result = self.sim.modules[
                    "HealthSystem"
                ].dx_manager.run_dx_test(
                    dx_tests_to_run="UF_schisto_test_lowWB", hsi_event=self
                )

            elif df.at[person_id, 'ss_sh_infection_status'] == 'Moderate-infection':
                test_result = self.sim.modules[
                    "HealthSystem"
                ].dx_manager.run_dx_test(
                    dx_tests_to_run="UF_schisto_test_moderateWB", hsi_event=self
                )

            else:
                test_result = self.sim.modules[
                    "HealthSystem"
                ].dx_manager.run_dx_test(
                    dx_tests_to_run="UF_schisto_test_heavyWB", hsi_event=self
                )

        # otherwise perform Kato-Katz
        else:
            # sensitivity of test depends worm burden
            if df.at[person_id, 'ss_sm_infection_status'] in ['Non-infected', 'Low-infection']:

                test_result = self.sim.modules[
                    "HealthSystem"
                ].dx_manager.run_dx_test(
                    dx_tests_to_run="KK_schisto_test_lowWB", hsi_event=self
                )
            elif df.at[person_id, 'ss_sm_infection_status'] == 'Moderate-infection':
                test_result = self.sim.modules[
                    "HealthSystem"
                ].dx_manager.run_dx_test(
                    dx_tests_to_run="KK_schisto_test_moderateWB", hsi_event=self
                )
            else:
                test_result = self.sim.modules[
                    "HealthSystem"
                ].dx_manager.run_dx_test(
                    dx_tests_to_run="KK_schisto_test_heavyWB", hsi_event=self
                )

        # add equipment
        if test_result is not None:
            self.add_equipment({'Ordinary Microscope'})

        if test_result:
            self.module.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Schisto_TreatmentFollowingDiagnosis(
                    module=self.module,
                    person_id=person_id),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )

        # if test negative or test not available, second testing appt is scheduled
        else:
            # The person will not test now. If this is the "first attempt", re-schedule this HSI to occur after a delay,
            if self._num_occurrences <= 1:
                next_occurence = self.sim.date + pd.DateOffset(days=int(
                    self.module.rng.uniform(params['delay_till_hsi_a_repeated'], params['delay_till_hsi_b_repeated'])
                ))

                self.module.sim.modules['HealthSystem'].schedule_hsi_event(
                    self,
                    topen=next_occurence,
                    tclose=None,
                    priority=0
                )


class HSI_Schisto_TreatmentFollowingDiagnosis(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event for a person being provided with PZQ treatment after having been
    diagnosed."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Schisto)

        under_5 = self.sim.population.props.at[person_id, 'age_years'] <= 5
        self.TREATMENT_ID = 'Schisto_Treatment'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Under5OPD' if under_5 else 'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        """Do the treatment for this person."""
        dosage = self.module.calculate_praziquantel_dosage(person_id)

        if self.get_consumables(item_codes={self.module.item_code_for_praziquantel: dosage}):
            self.module.do_effect_of_treatment(person_id=person_id, mda=False)


class HSI_Schisto_MDA(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event for providing one or more persons with PZQ as part of a Mass Drug
    Administration (MDA). Note that the `person_id` declared as the `target` of this `HSI_Event` is only one of the
    beneficiaries. This is in, effect, a "batch job" of individual HSI being handled within one HSI, for the sake of
    computational efficiency.

    This is repeated for each district every time MDA is scheduled, allowing variable coverage by district each year
    """

    def __init__(self, module, person_id, beneficiaries_ids: Optional[Sequence] = None,
                 age_group_included: Optional[Sequence] = None):

        super().__init__(module, person_id=person_id)
        assert isinstance(module, Schisto)
        self.beneficiaries_ids = beneficiaries_ids
        self.age_group_included = age_group_included

        self.TREATMENT_ID = 'Schisto_MDA'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({
            'ConWithDCSA': len(beneficiaries_ids) * 0.5 if beneficiaries_ids else 0.5})
        self.ACCEPTED_FACILITY_LEVEL = '0'

    def apply(self, person_id, squeeze_factor):
        """Provide the treatment to the beneficiaries of this HSI."""

        # Find which of the beneficiaries are still alive
        beneficiaries_still_alive = list(set(self.beneficiaries_ids).intersection(
            self.sim.population.props.index[self.sim.population.props.is_alive]
        ))

        # Calculate total dosage required for all beneficiaries still alive
        # if MDA includes adults, return average dose for adults for all beneficiaries
        # if MDA includes SAC/PSAC, return average dose for SAC for all beneficiaries
        # using total_dosage = sum(self.module.calculate_praziquantel_dosage(pid) for pid in beneficiaries_still_alive)
        # is very slow
        p = self.module.parameters
        if 'Adults' in self.age_group_included:
            # adult dosing
            total_dosage = 40 * p['avg_weight_adult_kg'] * len(beneficiaries_still_alive)
        else:
            # child (SAC/PSAC) dosing
            total_dosage = 40 * p['avg_weight_sac_kg'] * len(beneficiaries_still_alive)

        # Let the key consumable be "optional" in order that provision of the treatment is NOT conditional on the drugs
        # being available.This is because we expect that special planning would be undertaken in order to ensure the
        # availability of the drugs on the day(s) when the MDA is planned.
        if self.get_consumables(
            optional_item_codes={self.module.item_code_for_praziquantel_MDA: total_dosage}
        ):
            self.module.do_effect_of_treatment(person_id=beneficiaries_still_alive, mda=True)

        # Return the update appointment that reflects the actual number of beneficiaries.
        return self.make_appt_footprint({'ConWithDCSA': len(beneficiaries_still_alive) * 0.5})


class SchistoPersonDaysLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """This is a regular event (every day) that logs the person-days infected """
        super().__init__(module, frequency=DateOffset(days=1))
        assert isinstance(module, Schisto)

    def apply(self, population):
        """
        Log the numbers of people infected with any species of schisto and heavy-burden infections
        sum these in SchistoLoggingEvent each year to get person-years infected
        """

        # Precompute categories
        df = self.sim.population.props

        # get infection counts for both 'sh' and 'sm'
        def count_infection_status(df, species_prefix, infection_status):
            # Filter by infection status and alive status
            mask = (df[f"ss_{species_prefix}_infection_status"] == infection_status) & df.is_alive

            # Use 'map' if age_group_mapper is a dictionary (assuming it's defined as a dictionary)
            age_groups = df.loc[mask, 'age_years'].map(self.module.age_group_mapper)

            # Group by the mapped age groups and district directly, without creating a DataFrame
            grouped = pd.Series(1, index=[age_groups, df.loc[mask, 'district_of_residence']]).groupby(
                level=[0, 1]).sum()

            return grouped

        # Count people based on ss_sh_infection_status
        sh_counts = {
            'Low-infection': count_infection_status(df, 'sh', 'Low-infection'),
            'Moderate-infection': count_infection_status(df, 'sh', 'Moderate-infection'),
            'Heavy-infection': count_infection_status(df, 'sh', 'Heavy-infection')
        }

        # Count people based on ss_sm_infection_status
        sm_counts = {
            'Low-infection': count_infection_status(df, 'sm', 'Low-infection'),
            'Moderate-infection': count_infection_status(df, 'sm', 'Moderate-infection'),
            'Heavy-infection': count_infection_status(df, 'sm', 'Heavy-infection')
        }

        # Update the log_person_days DataFrame
        for infection_status, counts in sh_counts.items():
            for (age_group, district), count in counts.items():
                self.module.log_person_days.loc[
                    ('haematobium', age_group, infection_status, district), 'person_days'
                ] += count

        for infection_status, counts in sm_counts.items():
            for (age_group, district), count in counts.items():
                self.module.log_person_days.loc[
                    ('mansoni', age_group, infection_status, district), 'person_days'
                ] += count


class SchistoLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """This is a regular event (every year) that causes the logging for each species."""
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, Schisto)

    def apply(self, population):
        """
        Call `log_infection_status` and 'log_mean_worm_burden' for each species
        """
        for _spec in self.module.species.values():
            _spec.log_infection_status()
            # _spec.log_mean_worm_burden()  # revert this if needed

        #PZQ MDA episodes
        df = population.props

        # this is logging MDA only
        new_mda = df[
            'ss_MDA_treatment_counter'
        ].sum()

        mda_episodes = {
            'mda_episodes': new_mda,
        }
        logger.info(
            key='schisto_mda_episodes',
            data=mda_episodes,
            description='Counts of mda occurring in timeperiod'
        )

        mda_by_district = df.groupby('district_of_residence', observed=False).agg({'ss_MDA_treatment_counter': 'sum'})

        mda_episodes_district = {
            'mda_episodes_district': mda_by_district.to_dict()
        }

        logger.info(
            key='schisto_mda_episodes_by_district',
            data=mda_episodes_district,
            description='Counts of mda occurring in timeperiod by district'
        )

        # reset the counter
        df['ss_MDA_treatment_counter'] = 0


        # PERSON-DAYS OF INFECTION
        # log person-days of infection by low, moderate and heavy for all, SAC and PSAC separately
        logger.info(
            key='Schisto_person_days_infected',
            data=flatten_multi_index_series_into_dict_for_logging(self.module.log_person_days['person_days']),
            description='Counts of person-days infected by species'
        )
        # Reset the daily counts for the next month
        self.module.log_person_days.loc[:, 'person_days'] = 0


        # NUMBERS INFECTED
        # extract and map age groups for those alive
        age_grp = df['age_years'].map(self.module.age_group_mapper)

        # Create the infection mask without creating new DataFrames
        infection_mask = (df['ss_sm_infection_status'] != 'Non-infected') | (
                df['ss_sh_infection_status'] != 'Non-infected')
        alive_mask = df['is_alive']  # Mask for people who are alive

        # Apply mask directly in groupby and calculate infected count
        infected = df[alive_mask & infection_mask].groupby(
            by=['district_of_residence', age_grp],
            observed=False
        ).size()

        # Rename index directly
        infected.index.rename(['district_of_residence', 'age_group'], inplace=True)

        # repeat above but for heavy-intensity infections only
        heavy_infection_mask = (df['ss_sm_infection_status'] == 'Heavy-infection') | (
                df['ss_sh_infection_status'] == 'Heavy-infection')

        heavy_infected = df[alive_mask & heavy_infection_mask].groupby(
            by=['district_of_residence', age_grp],
            observed=False
        ).size()
        heavy_infected.index.rename(['district_of_residence', 'age_group'], inplace=True)

        # Apply mask for just the alive individuals and calculate alive count
        alive = df[alive_mask].groupby(
            by=['district_of_residence', age_grp],
            observed=False
        ).size()

        # Rename index directly
        alive.index.rename(['district_of_residence', 'age_group'], inplace=True)

        logger.info(
            key='number_infected_any_species',
            description='Counts of infection status by age-group and district.',
            data={
                'number_infected': flatten_multi_index_series_into_dict_for_logging(infected),
            },
        )
        logger.info(
            key='number_heavy_infected_any_species',
            description='Counts of heavy infection status by age-group and district.',
            data={
                'number_heavy_infected': flatten_multi_index_series_into_dict_for_logging(heavy_infected),
            },
        )
        logger.info(
            key='number_in_subgroup',
            description='Counts of infection status with this species by age-group and district.',
            data={
                'number_alive': flatten_multi_index_series_into_dict_for_logging(alive),
            },
        )

        # WASH properties
        unimproved_sanitation = sum(df.li_unimproved_sanitation & df.is_alive) / sum(df.is_alive) if sum(
            df.is_alive) else 0

        no_access_handwashing = sum(df.li_no_access_handwashing & df.is_alive) / sum(df.is_alive) if sum(
            df.is_alive) else 0

        no_clean_drinking_water = sum(df.li_no_clean_drinking_water & df.is_alive) / sum(df.is_alive) if sum(
            df.is_alive) else 0

        wash = {
            'unimproved_sanitation': unimproved_sanitation,
            'no_access_handwashing': no_access_handwashing,
            'no_clean_drinking_water': no_clean_drinking_water,
        }

        logger.info(
            key='Schisto_wash_properties',
            data=wash,
            description='Proportion of population with each wash-related property'
        )

        # General function to calculate the proportion of any property for people who are alive
        def calculate_wash_proportion(group, property_column):
            total_alive = group['is_alive'].sum()

            if total_alive > 0:
                # Calculate the sum of the property (e.g., li_unimproved_sanitation or li_access_water) for alive people
                property_sum = (group[property_column] * group['is_alive']).sum()
                property_proportion = property_sum / total_alive
            else:
                property_proportion = 0

            return property_proportion

        # For 'li_unimproved_sanitation'
        unimproved_sanitation_by_district = df.groupby('district_of_residence').apply(calculate_wash_proportion,
                                                                                      property_column='li_unimproved_sanitation')

        no_access_handwashing_by_district = df.groupby('district_of_residence').apply(calculate_wash_proportion,
                                                                             property_column='li_no_access_handwashing')

        no_clean_drinking_water_by_district = df.groupby('district_of_residence').apply(calculate_wash_proportion,
                                                                             property_column='li_no_clean_drinking_water')

        # Convert the results into dictionaries
        unimproved_sanitation_by_district = unimproved_sanitation_by_district.to_dict()
        no_access_handwashing_by_district = no_access_handwashing_by_district.to_dict()
        no_clean_drinking_water_by_district = no_clean_drinking_water_by_district.to_dict()

        wash_district = {
            'unimproved_sanitation_district': unimproved_sanitation_by_district,
            'no_access_handwashing_district': no_access_handwashing_by_district,
            'no_clean_drinking_water_district': no_clean_drinking_water_by_district,
        }

        logger.info(
            key='Schisto_wash_properties_by_district',
            data=wash_district,
            description='Proportion of population with each wash-related property by district'
        )
