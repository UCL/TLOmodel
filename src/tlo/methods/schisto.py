from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence, Optional, Union

import numpy as np
import pandas as pd
from itertools import product

from tlo import Date, DateOffset, Module, Parameter, Property, Types, logging
from tlo.analysis.utils import flatten_multi_index_series_into_dict_for_logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.hsi_generic_first_appts import GenericFirstAppointmentsMixin
from tlo.methods.symptommanager import Symptom
from tlo.util import random_date, read_csv_files
from tlo.methods.dxmanager import DxTest

if TYPE_CHECKING:
    from tlo.methods.hsi_generic_first_appts import HSIEventScheduler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Definition of the age-groups used in the module, as a tuple of two integers (a,b) such that the given age group
#  is in range a <= group <= b. i.e.,
#     2 <= PSAC <= 4
#     5 <= SAC <= 14
#     15 <= Adults
#     0 <= All
_AGE_GROUPS = {'Infant': (0, 1), 'PSAC': (2, 4), 'SAC': (5, 14), 'Adults': (15, 120), 'All': (0, 120)}


class Schisto(Module, GenericFirstAppointmentsMixin):
    """Schistosomiasis module.
    Two species of worm that cause Schistosomiasis are modelled independently. Worms are acquired by persons via the
     environment. There is a delay between the acquisition of worms and the maturation to 'adults' worms; and a long
     period before the adult worms die. The number of worms in a person (whether a high-intensity infection or not)
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
        Metadata.USES_HEALTHBURDEN
    }

    CAUSES_OF_DEATH = {}

    CAUSES_OF_DISABILITY = {
        'Schistosomiasis': Cause(gbd_causes='Schistosomiasis', label='Schistosomiasis'),
    }

    module_prefix = 'ss'

    PROPERTIES = {
        f'{module_prefix}_last_PZQ_date': Property(Types.DATE, 'Day of the most recent treatment with PZQ')
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
        'urine_filtration_sensitivity_lowWB': Parameter(Types.REAL,
                                                        'Sensitivity of UF in detecting low WB'),
        'urine_filtration_sensitivity_moderateWB': Parameter(Types.REAL,
                                                             'Sensitivity of UF in detecting moderate WB'),
        'urine_filtration_sensitivity_highWB': Parameter(Types.REAL,
                                                         'Sensitivity of UF in detecting high WB'),
        'kato_katz_sensitivity_moderateWB': Parameter(Types.REAL,
                                                      'Sensitivity of KK in detecting moderate WB'),
        'kato_katz_sensitivity_highWB': Parameter(Types.REAL,
                                                  'Sensitivity of KK in detecting high WB'),
        'scaleup_WASH': Parameter(Types.INT,
                                  'Boolean whether to scale-up WASH during simulation'),
        'scaleup_WASH_start_year': Parameter(Types.INT,
                                             'Start date to scale-up WASH, years after sim start date'),
        'mda_coverage': Parameter(Types.REAL,
                                  'Coverage of future MDA activities, consistent across all'
                                  'target groups'),
        'mda_target_group': Parameter(Types.STRING,
                                      'Target group for future MDA activities, '
                                      'one of [PSAC_SAC, SAC, ALL]'),
        'mda_frequency_months': Parameter(Types.INT,
                                          'Number of months between MDA activities'),
        'scaling_factor_baseline_risk': Parameter(Types.REAL,
                                                  'scaling factor controls how the background risk of '
                                                  'infection is adjusted based on the deviation of current prevalence '
                                                  'from baseline prevalence'),
        'baseline_risk': Parameter(Types.REAL,
                                  'number of worms applied as a baseline risk across districts to prevent '
                                  'fadeout, number is scaled by scaling_factor_baseline_risk'),
        'MDA_coverage_historical': Parameter(Types.DATA_FRAME,
                                             'Probability of getting PZQ in the MDA for PSAC, SAC and Adults '
                                             'in historic rounds'),
    }

    def __init__(self, name=None, resourcefilepath=None, mda_execute=True, single_district=False):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.mda_execute = mda_execute
        self.single_district = single_district

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

        # Age-group mapper
        s = pd.Series(index=range(1 + 120), data='object')
        for name, (low_limit, high_limit) in _AGE_GROUPS.items():
            if name != 'All':
                s.loc[(s.index >= low_limit) & (s.index <= high_limit)] = name
        self.age_group_mapper = s.to_dict()

    def read_parameters(self, data_folder):
        """Read parameters and register symptoms."""

        # Define districts that this module will operate in:
        self.districts = self.sim.modules['Demography'].districts  # <- all districts

        workbook = read_csv_files(Path(self.resourcefilepath) / 'ResourceFile_Schisto', files=None)
        self.parameters = self._load_parameters_from_workbook(workbook)

        # load species-specific parameters
        for _spec in self.species.values():
            self.parameters.update(_spec.load_parameters_from_workbook(workbook))

        # Register symptoms
        symptoms_df = workbook['Symptoms']
        # self._register_symptoms(symptoms_df.set_index('Symptom')['HSB_mapped_symptom'].to_dict())
        self._register_symptoms(symptoms_df)

        # create container for logging person-days infected
        index = pd.MultiIndex.from_product(
            [
                ['mansoni', 'haematobium'],  # species
                ['Infant', 'PSAC', 'SAC', 'Adults'],  # age_group
                ['Low-infection', 'Moderate-infection', 'High-infection'],  # infection_level
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
        df.loc[df.is_alive, f'{self.module_prefix}_last_PZQ_date'] = pd.NaT

        # reset all to one district if doing calibration or test runs
        # choose Zomba as it has ~10% prev of both species
        if self.single_district:
            df['district_num_of_residence'] = 19
            df['district_of_residence'] = pd.Categorical(['Zomba'] * len(df),
                                                         categories=df['district_of_residence'].cat.categories)
            df['region_of_residence'] = pd.Categorical(['Southern'] * len(df),
                                                       categories=df['region_of_residence'].cat.categories)
        for _spec in self.species.values():
            _spec.initialise_population(population)

    def initialise_simulation(self, sim):
        """Get ready for simulation start."""

        # Initialise the simulation for each species
        for _spec in self.species.values():
            _spec.initialise_simulation(sim)

        # Look-up DALY weights
        if 'HealthBurden' in self.sim.modules:
            self.disability_weights = self._get_disability_weight()

        # Look-up item codes for Praziquantel
        if 'HealthSystem' in self.sim.modules:
            self.item_code_for_praziquantel = self._get_item_code_for_praziquantel(MDA=False)
            self.item_code_for_praziquantel_MDA = self._get_item_code_for_praziquantel(MDA=True)

        # define the Dx tests and consumables required
        self._get_consumables_for_dx()

        # Schedule the logging event
        sim.schedule_event(SchistoLoggingEvent(self), sim.date)  # monthly, by district, age-group
        sim.schedule_event(SchistoPersonDaysLoggingEvent(self), sim.date)

        # over-ride availability of PZQ for MDA
        # self.sim.modules['HealthSystem'].override_availability_of_consumables(
        #     {1735: 1.0})  # this is the donated PZQ not currently in consumables availability worksheet
        self.sim.modules['HealthSystem'].override_availability_of_consumables(
            {286: 1.0})

        # Schedule MDA events
        if self.mda_execute:
            # update future mda strategy from default values
            self.prognosed_mda = self._create_mda_strategy()

            self._schedule_mda_events()

        # schedule WASH scale-up
        if self.parameters['scaleup_WASH']:
            sim.schedule_event(SchistoWashScaleUp(self),
                               Date(int(self.parameters['scaleup_WASH_start_year']), 1, 1))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        All children are born without an infection, even if the mother is infected.

        :param mother_id: the ID for the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props
        df.at[child_id, f'{self.module_prefix}_last_PZQ_date'] = pd.NaT

        # if WASH in action, update property li_unimproved_sanitation=False for all new births
        if self.parameters['scaleup_WASH'] and (
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
            """Returns the sum of the disability weights from a list of symptoms, capping at 1.0"""
            dw = 0.0
            if not list_of_symptoms:
                return dw

            for symptom in list_of_symptoms:
                if symptom in self.disability_weights:
                    dw += self.disability_weights.get(symptom)

            return min(1.0, dw)

        disability_weights_for_each_person_with_symptoms = pd.Series(symptoms_being_caused).apply(
            get_total_disability_weight)

        # Return pd.Series that include entries for all alive persons (filling 0.0 where they do not have any symptoms)
        df = self.sim.population.props
        return pd.Series(index=df.index[df.is_alive], data=0.0).add(disability_weights_for_each_person_with_symptoms,
                                                                    fill_value=0.0)

    def do_effect_of_treatment(self, person_id: Union[int, Sequence[int]]) -> None:
        """Do the effects of a treatment administered to a person or persons. This can be called for a person who is
        infected and receiving treatment following a diagnosis, or for a person who is receiving treatment as part of a
         Mass Drug Administration. The burden and effects of any species are alleviated by a successful treatment."""

        df = self.sim.population.props

        # Ensure person_id is treated as an iterable (e.g., list) even if it's a single integer
        if isinstance(person_id, int):
            person_id = [person_id]

        # Clear any symptoms caused by this module (i.e., Schisto of any species)
        self.sim.modules['SymptomManager'].clear_symptoms(person_id=person_id, disease_module=self)

        # Record the date of last treatment
        df.loc[person_id, 'ss_last_PZQ_date'] = self.sim.date

        # Update properties after PZQ treatment
        for spec_prefix in [_spec.prefix for _spec in self.species.values()]:
            pzq_efficacy = self.parameters[f'{spec_prefix}_PZQ_efficacy']

            df.loc[person_id, f'{self.module_prefix}_{spec_prefix}_aggregate_worm_burden'] = df.loc[
                                                                                                 person_id, f'{self.module_prefix}_{spec_prefix}_aggregate_worm_burden'] * (
                                                                                                 1 - pzq_efficacy)
            # if worm burden >=1, still infected
            mask = df.loc[person_id, f'{self.module_prefix}_{spec_prefix}_aggregate_worm_burden'] < 1
            df.loc[mask.index[mask], f'{self.module_prefix}_{spec_prefix}_infection_status'] = 'Non-infected'

    def _load_parameters_from_workbook(self, workbook) -> dict:
        """Load parameters from ResourceFile (loaded by pd.read_excel as `workbook`) that are general (i.e., not
        specific to a particular species)."""

        parameters = dict()

        # HSI and treatment params:
        param_list = workbook['Parameters'].set_index("Parameter")['Value']

        for _param_name in (
            'delay_till_hsi_a_repeated',
            'delay_till_hsi_b_repeated',
            'rr_WASH',
            'calibration_scenario',
            'urine_filtration_sensitivity_lowWB',
            'urine_filtration_sensitivity_moderateWB',
            'urine_filtration_sensitivity_highWB',
            'kato_katz_sensitivity_moderateWB',
            'kato_katz_sensitivity_highWB',
            'scaleup_WASH',  # Needs to be included
            'scaleup_WASH_start_year',
            'mda_coverage',
            'mda_target_group',  # Needs to be included
            'mda_frequency_months',
            'scaling_factor_baseline_risk',
            'baseline_risk',
        ):
            value = param_list[_param_name]

            # Convert to float if possible, otherwise store as is
            try:
                parameters[_param_name] = float(value)
            except ValueError:
                parameters[_param_name] = value

        # MDA coverage - historic ESPEN data
        historical_mda = workbook['ESPEN_MDA'].set_index(['District', 'Year'])[
            ['EpiCov_PSAC', 'EpiCov_SAC', 'EpiCov_Adults']]
        historical_mda.columns = historical_mda.columns.str.replace('EpiCov_', '')

        parameters['MDA_coverage_historical'] = historical_mda.astype(float)
        # clip upper limit of MDA coverage at 99%
        parameters['MDA_coverage_historical'] = parameters['MDA_coverage_historical'].clip(upper=0.99)

        return parameters

    def _create_mda_strategy(self) -> pd.DataFrame:

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

    def _register_symptoms(self, symptoms: dict) -> None:
        """Register the symptoms with the `SymptomManager`.
        :params symptoms: The symptoms that are used by this module in a dictionary of the form, {<symptom>:
        <generic_symptom_similar>}. Each symptom is associated with the average healthcare seeking behaviour
        unless otherwise specified."""
        generic_symptoms = self.sim.modules['SymptomManager'].generic_symptoms

        # Iterate through DataFrame rows and register each symptom
        for _, row in symptoms.iterrows():
            if row['Symptom'] not in generic_symptoms:
                symptom_kwargs = {
                    'name': row['Symptom'],
                    'odds_ratio_health_seeking_in_children': row.get('odds_ratio_health_seeking_in_children', None),
                    'odds_ratio_health_seeking_in_adults': row.get('odds_ratio_health_seeking_in_adults', None),
                    'prob_seeks_emergency_appt_in_children': row.get('prob_seeks_emergency_appt_in_children', None),
                    'prob_seeks_emergency_appt_in_adults': row.get('prob_seeks_emergency_appt_in_adults', None),
                }
                # Remove None values to avoid passing unnecessary arguments
                symptom_kwargs = {k: v for k, v in symptom_kwargs.items() if v is not None}

                self.sim.modules['SymptomManager'].register_symptom(Symptom(**symptom_kwargs))

    def _get_disability_weight(self) -> dict:
        """Return dict containing the disability weight (value) of each symptom (key)."""

        symptoms_to_disability_weight_mapping = {
            # These mapping are justified in the 'DALYS' worksheet of the ResourceFile.
            'anemia': 258,
            'fever': 262,
            'hydronephrosis': 260,
            'dysuria': 263,
            'bladder_pathology': 264,
            'diarrhoea': 259,
            'vomiting': 254,
            'ascites': 261,
            'hepatomegaly': 257,
            'haematuria': None  # That's a very common symptom but no official DALY weight yet defined.
        }
        get_daly_weight = lambda _code: self.sim.modules['HealthBurden'].get_daly_weight(  # noqa: E731
            _code) if _code is not None else 0.0

        return {
            symptom: get_daly_weight(dw_code) for symptom, dw_code in symptoms_to_disability_weight_mapping.items()
        }

    def _get_item_code_for_praziquantel(self, MDA=False) -> int:
        """Look-up the item code for Praziquantel"""

        if MDA:
            # todo donated PZQ not currently in consumables availability sheet
            # return self.sim.modules['HealthSystem'].get_item_code_from_item_name("Praziquantel, 600 mg (donated)")
            return self.sim.modules['HealthSystem'].get_item_code_from_item_name("Praziquantel 600mg_1000_CMST")
        else:
            return self.sim.modules['HealthSystem'].get_item_code_from_item_name("Praziquantel 600mg_1000_CMST")

    def _calculate_praziquantel_dosage(self, person_id):
        age = self.sim.population.props.at[person_id, "age_years"]

        # 40mg per kg, as single dose (MSTG)
        # in children <4 years, 20mg/kg
        # assume child 0-5, maximum weight 17.5kg (WHO- average between girls/boys)
        if age < 5:
            dose = 20 * 17.5
        # child aged 5-15, 40mg/kg, use mid-point age 10: 50th percentile weight=30kg
        elif age >= 5 and age < 15:
            dose = 40 * 30
        # adult, 40mg/kg, average weight 62kg
        else:
            dose = 40 * 62

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
        # todo update quantities
        # e.g. item_codes = {self.module.item_codes_for_consumables_required['item']: 63}
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            KK_schisto_test_lowWB=DxTest(
                property='ss_sm_infection_status',
                target_categories=["Non-infected", "Low-infection"],
                sensitivity=0.0,
                specificity=0.0,
                item_codes=self.item_codes_for_consumables_required['microscope_slide'],
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
                specificity=1.0,
                item_codes=self.item_codes_for_consumables_required['microscope_slide'],
                optional_item_codes=[
                    self.item_codes_for_consumables_required['malachite_stain'],
                    self.item_codes_for_consumables_required['iodine_stain']]
            )
        )

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            KK_schisto_test_highWB=DxTest(
                property='ss_sm_infection_status',
                target_categories=["High-infection"],
                sensitivity=p["kato_katz_sensitivity_highWB"],
                specificity=1.0,
                item_codes=self.item_codes_for_consumables_required['microscope_slide'],
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
                sensitivity=0.0,
                specificity=0.0,
                item_codes=self.item_codes_for_consumables_required['microscope_slide'],
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
                specificity=1.0,
                item_codes=self.item_codes_for_consumables_required['microscope_slide'],
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
                specificity=1.0,
                item_codes=self.item_codes_for_consumables_required['microscope_slide'],
                optional_item_codes=[
                    self.item_codes_for_consumables_required['filter_paper'],
                    self.item_codes_for_consumables_required['iodine_stain']]
            )
        )

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            UF_schisto_test_highWB=DxTest(
                property='ss_sh_infection_status',
                target_categories=["High-infection"],
                sensitivity=p["urine_filtration_sensitivity_highWB"],
                specificity=1.0,
                item_codes=self.item_codes_for_consumables_required['microscope_slide'],
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
                Date(year=year, month=7, day=1)
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
                Date(year=year_first_simulated_mda, month=7, day=1)
            )

    def do_at_generic_first_appt(
        self,
        person_id: int,
        symptoms: List[str],
        schedule_hsi_event: HSIEventScheduler,
        **kwargs,
    ) -> None:
        # Do when person presents to the GenericFirstAppt.
        # If the person has certain set of symptoms, refer ta HSI for testing.
        set_of_symptoms_indicative_of_schisto = {'anemia',
                                                 'haematuria',
                                                 'bladder_pathology',
                                                 'fever',
                                                 'ascites',
                                                 'diarrhoea',
                                                 'vomiting',
                                                 'hepatomegaly',
                                                 'dysuria'}

        if any(symptom in set_of_symptoms_indicative_of_schisto for symptom in symptoms):
            event = HSI_Schisto_TestingFollowingSymptoms(
                module=self, person_id=person_id
            )
            schedule_hsi_event(event, priority=0, topen=self.sim.date)

    def select_test(self, person_id):

        # choose test
        persons_symptoms = self.sim.modules["SymptomManager"].has_what(person_id)
        if any(symptom in ['haematuria', 'bladder_pathology',] for symptom in persons_symptoms):
            test = 'urine_filtration_test'
        else:
            test = 'kato-katz'

        return test

    def reduce_susceptibility(self, df, species_column):
        """
        Reduce the proportion of individuals susceptible to each species by a specified percentage
        applied
        """
        p = self.parameters

        # Find the number of individuals with currently susceptible to species and no sanitation
        susceptible_no_sanitation = df.index[(df[species_column] == 1) & (df['li_unimproved_sanitation'] == True)]

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
            # 'delay_a': Parameter(Types.REAL, 'End of the latent period in days, start'),
            # 'delay_b': Parameter(Types.REAL, 'End of the latent period in days, end'),
            'symptoms': Parameter(Types.DICT, 'Symptoms of the schistosomiasis infection, dependent on the module'),
            'R0': Parameter(Types.REAL, 'R0 of species'),
            'beta_PSAC': Parameter(Types.REAL, 'Contact/exposure rate of PSAC'),
            'beta_SAC': Parameter(Types.REAL, 'Contact/exposure rate of SAC'),
            'beta_Adults': Parameter(Types.REAL, 'Contact/exposure rate of Adults'),
            'worms_fecundity': Parameter(Types.REAL, 'Fecundity parameter, driving density-dependent reproduction'),
            'worm_lifespan': Parameter(Types.REAL, 'Lifespan of the worm in human host given in years'),
            'high_intensity_threshold': Parameter(Types.REAL,
                                                  'Threshold of worm burden indicating high intensity infection'),
            'low_intensity_threshold': Parameter(Types.REAL,
                                                 'Threshold of worm burden indicating low intensity infection'),
            'high_intensity_threshold_PSAC': Parameter(Types.REAL,
                                                       'Worm burden threshold for high intensity infection in PSAC'),
            'PZQ_efficacy': Parameter(Types.REAL,
                                      'Efficacy of praziquantel in reducing worm burden'),
            'baseline_prevalence': Parameter(Types.REAL,
                                      'Baseline prevalence of species across all districts in 2010'),
            'mean_worm_burden2010': Parameter(Types.DATA_FRAME,
                                              'Mean worm burden per infected person per district in 2010'),
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
                categories=['Non-infected', 'Low-infection', 'Moderate-infection', 'High-infection']),
            'aggregate_worm_burden': Property(
                Types.INT, 'Number of mature worms of this species in the individual'),
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
        param_list = workbook['Parameters'].set_index("Parameter")['Value']
        for _param_name in ('R0',
                            'beta_PSAC',
                            'beta_SAC',
                            'beta_Adults',
                            'worm_lifespan',
                            'worms_fecundity',
                            'high_intensity_threshold',
                            'low_intensity_threshold',
                            'high_intensity_threshold_PSAC',
                            'PZQ_efficacy',
                            'baseline_prevalence',
                            ):
            parameters[_param_name] = float(param_list[f'{_param_name}_{self.name}'])

        # this is the updated (calibrated) data
        schisto_initial_reservoir = workbook[f'LatestData_{self.name}'].set_index("District")
        parameters['mean_worm_burden2010'] = schisto_initial_reservoir['Mean_worm_burden']
        parameters['gamma_alpha'] = schisto_initial_reservoir['gamma_alpha']
        parameters['prop_susceptible'] = schisto_initial_reservoir['prop_susceptible']

        # Symptoms (prevalence of each type of symptom)
        symptoms_df = workbook['Symptoms']
        parameters['symptoms'] = \
            symptoms_df.loc[symptoms_df['Infection_type'].isin(['both', self.name])].set_index('Symptom')[
                'Prevalence'].to_dict()

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

        # assign a harbouring rate
        self._assign_initial_properties(population)

        # assign initial worm burden
        self._assign_initial_worm_burden(population)

    def initialise_simulation(self, sim):
        """
        * Schedule natural history events for those with worm burden initially.
        * Schedule the WormBurdenEvent for this species. (A recurring instance of this event will be scheduled for
        each species independently.)"""

        df = sim.population.props

        # Assign infection statuses and symptoms and schedule natural history events for those with worm burden
        # initially.
        self.update_infectious_status_and_symptoms(df.index[df.is_alive])
        self._schedule_death_of_worms_in_initial_population()

        sim.schedule_event(
            SchistoInfectionWormBurdenEvent(
                module=self.schisto_module,
                species=self),
            sim.date + DateOffset(months=1)
        )

    def on_birth(self, mother_id, child_id):
        """Initialise the species-specific properties for a newborn individual.
        :param mother_id: the ID for the mother for this child (redundant)
        :param child_id: the new child"""

        df = self.schisto_module.sim.population.props
        prop = self.prefix_species_property
        params = self.params
        global_params = self.schisto_module.parameters
        rng = self.schisto_module.rng

        # Assign the default for a newly born child
        df.at[child_id, prop('infection_status')] = 'Non-infected'
        df.at[child_id, prop('aggregate_worm_burden')] = 0

        # Generate the harbouring rate depending on a district of residence.
        district = df.at[child_id, 'district_of_residence']
        df.at[child_id, prop('harbouring_rate')] = rng.gamma(params['gamma_alpha'][district], size=1)

        # Determine if individual should be susceptible to each species
        # susceptibility depends on district
        prop_susceptible = params['prop_susceptible'][district]
        df.at[child_id, prop('susceptibility')] = 0  # Default to not susceptible

        # WASH in action
        if global_params['scaleup_WASH'] and (
            self.schisto_module.sim.date >= Date(int(global_params['scaleup_WASH_start_year']), 1, 1)):

            # if the child has sanitation, apply risk mitigated by rr_WASH
            if not df.at[child_id, 'li_unimproved_sanitation']:
                if rng.random_sample() < (prop_susceptible * (1 - global_params['rr_WASH'])):
                    df.at[child_id, prop('susceptibility')] = 1
            # if no sanitation, apply full risk
            elif df.at[child_id, 'li_unimproved_sanitation']:
                if rng.random_sample() < prop_susceptible:
                    df.at[child_id, prop('susceptibility')] = 1

        else:
            # WASH not implemented
            prop_population_without_sanitation = 0.11
            p_no_san = prop_susceptible / (prop_population_without_sanitation + (
                1 - prop_population_without_sanitation) * global_params['rr_WASH'])
            p_with_san = p_no_san * global_params['rr_WASH']

            # Determine the probability based on sanitation status
            # property li_unimproved_sanitation if False, person HAS improved sanitation
            if not df.at[child_id, 'li_unimproved_sanitation']:
                susceptibility_probability = p_with_san
            else:
                susceptibility_probability = p_no_san
            df.at[child_id, prop('susceptibility')] = 1 if rng.random_sample() < susceptibility_probability else 0

    def update_infectious_status_and_symptoms(self, idx: pd.Index) -> None:
        """Updates the infection status and symptoms based on the current aggregate worm burden of this species.
         * Assigns the 'infection status' (High-infection, Moderate-infection, Low-infection, Non-infected) to the persons with ids given
         in the `idx` argument, according their age (in years) and their aggregate worm burden (of worms of this
         species).
         * Causes the onset symptoms to the persons newly with high intensity infection.
         * Removes the symptoms if a person no longer has a high intensity infection (from any species)"""

        schisto_module = self.schisto_module
        df = schisto_module.sim.population.props
        prop = self.prefix_species_property
        params = self.params
        rng = schisto_module.rng
        possible_symptoms = params["symptoms"]
        sm = schisto_module.sim.modules['SymptomManager']
        cols_of_infection_status = schisto_module.cols_of_infection_status

        if not len(idx) > 0:
            return

        def _get_infection_status(population: pd.DataFrame) -> pd.Series:
            age = population["age_years"]
            agg_wb = population[prop("aggregate_worm_burden")]
            status = pd.Series(
                "Non-infected",
                index=population.index,
                dtype=population[prop("infection_status")].dtype
            )
            high_group = (
                             (age < 5) & (agg_wb >= params["high_intensity_threshold_PSAC"])
                         ) | (agg_wb >= params["high_intensity_threshold"])
            moderate_group = ~high_group & (agg_wb >= params["low_intensity_threshold"])
            low_group = (agg_wb < params["low_intensity_threshold"]) & (agg_wb > 0)
            status[high_group] = "High-infection"
            status[moderate_group] = "Moderate-infection"
            status[low_group] = "Low-infection"
            return status

        def _impose_symptoms_of_high_intensity_infection(idx: pd.Index) -> None:
            """Assign symptoms to the person with high intensity infection.
            :param idx: indices of individuals"""
            if not len(idx) > 0:
                return

            for symptom, prev in possible_symptoms.items():
                will_onset_this_symptom = idx[rng.random_sample(len(idx)) < prev]
                if not will_onset_this_symptom.empty:
                    sm.change_symptom(
                        person_id=will_onset_this_symptom,
                        symptom_string=symptom,
                        add_or_remove='+',
                        disease_module=schisto_module
                    )

        correct_status = _get_infection_status(df.loc[idx])
        original_status = df.loc[idx, prop('infection_status')]

        # Impose symptoms for those newly having 'High-infection' or 'Moderate-infection' status
        newly_have_high_infection = (original_status != 'High-infection') & (correct_status == 'High-infection')
        newly_have_moderate_infection = (original_status != 'Moderate-infection') & (correct_status == 'Moderate-infection')
        idx_newly_high_or_moderate_infection = newly_have_high_infection.index[
            newly_have_high_infection | newly_have_moderate_infection]
        _impose_symptoms_of_high_intensity_infection(idx=idx_newly_high_or_moderate_infection)

        # Update status for those whose status is changing
        idx_changing = correct_status.index[original_status != correct_status]
        df.loc[idx_changing, prop('infection_status')] = correct_status.loc[idx_changing]

        # Remove symptoms if there is no cause High-infection status caused by either species
        cols_of_infection_status_for_other_species = [
            col for col in cols_of_infection_status if col != prop('infection_status')
        ]
        high_infection_any_other_species = (
            df.loc[idx, cols_of_infection_status_for_other_species] == 'High-infection').any(axis=1)
        no_longer_high_infection = idx[
            (original_status == 'High-infection') & (
                correct_status != 'High-infection') & ~high_infection_any_other_species
            ]
        sm.clear_symptoms(person_id=no_longer_high_infection, disease_module=schisto_module)

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
            in_the_district = df.index[df['district_of_residence'] == district]  # people in the district
            num_in_district = len(in_the_district)  # population size in district

            # HARBOURING RATE
            hr = params['gamma_alpha'][district]
            df.loc[in_the_district, prop('harbouring_rate')] = rng.gamma(hr, size=num_in_district)

            # SUSCEPTIBILITY
            prop_susceptible = params['prop_susceptible'][district]

            # the total number needed to fill the proportion susceptible in district
            n_susceptible = int(np.ceil(prop_susceptible * num_in_district))

            # Select people with li_unimproved_sanitation=True
            no_sanitation = df.loc[(df['district_of_residence'] == district) & (df['li_unimproved_sanitation'] == True)]

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
                    (df['district_of_residence'] == district) & (df['li_unimproved_sanitation'] == False)]
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
            reservoir = int(len(in_the_district) * params['mean_worm_burden2010'][district])

            # Determine a 'contact rate' for each person
            contact_rates = pd.Series(1, index=in_the_district, dtype=float)

            # multiply by susceptibility (0 or 1)
            contact_and_susceptibility = contact_rates * df.loc[in_the_district, prop('susceptibility')]

            for age_group in ['PSAC', 'SAC', 'Adults']:
                age_range = _AGE_GROUPS[age_group]
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

    def _schedule_death_of_worms_in_initial_population(self) -> None:
        """Schedule death of worms assigned to the initial population"""
        df = self.schisto_module.sim.population.props
        prop = self.prefix_species_property
        params = self.params
        rng = self.schisto_module.rng
        date = self.schisto_module.sim.date

        people_with_worms = df.index[df[prop('aggregate_worm_burden')] > 0]
        for person_id in people_with_worms:
            months_till_death = int(rng.uniform(1, params['worm_lifespan'] * 12 / 2))
            self.schisto_module.sim.schedule_event(
                SchistoWormsNatDeath(module=self.schisto_module,
                                     species=self,
                                     person_id=person_id,
                                     number_of_worms_that_die=df.at[person_id, prop('aggregate_worm_burden')]),
                date + DateOffset(months=months_till_death)
            )

    def log_infection_status(self) -> None:
        """Log the number of persons in each infection status for this species, by age-group and district."""

        df = self.schisto_module.sim.population.props
        prop = self.prefix_species_property

        age_grp = df.loc[df.is_alive].age_years.map(self.schisto_module.age_group_mapper)

        data = df.loc[df.is_alive].groupby(by=[
            df.loc[df.is_alive, self.infection_status_property],
            df.loc[df.is_alive, 'district_of_residence'],
            age_grp
        ], observed=False).size()
        data.index.rename('infection_status', level=0, inplace=True)

        logger.info(
            key=f'infection_status_{self.name}',
            data=flatten_multi_index_series_into_dict_for_logging(data),
            description='Counts of infection status with this species by age-group and district.'
        )

        #  Group by district and calculate counts
        grouped_data = df.loc[df.is_alive].groupby('district_of_residence')[prop('susceptibility')].agg(
            total_count='count',
            susceptible_count=lambda x: (x == 1).sum()
        )

        # reinstate if need to output prop susceptible
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

        age_grp = df.loc[df.is_alive].age_years.map(self.schisto_module.age_group_mapper)

        data = df.loc[df.is_alive].groupby(by=[
            df.loc[df.is_alive, 'district_of_residence'],
            age_grp
        ], observed=False)[prop('aggregate_worm_burden')].mean()

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
            description='Mean worm burden of this species by district.'
        )


class SchistoInfectionWormBurdenEvent(RegularEvent, PopulationScopeEventMixin):
    """A recurring event that causes infection of people with this species.
     * Determines who becomes infected using worm burden and reservoir of infectious material.
     * Schedules `SchistoMatureWorms` for when the worms mature to adult worms."""

    def __init__(self, module: Module, species: SchistoSpecies):
        super().__init__(module, frequency=DateOffset(months=1))
        self.species = species

    def apply(self, population):
        df = population.props
        params = self.species.params
        global_params = self.module.parameters
        rng = self.module.rng
        # prop calls the property starting with the prefix species property, i.e. ss_sm or ss_sh
        prop = self.species.prefix_species_property

        # betas (exposure rates) are fixed for each age-group
        # exposure rates determine contribution to transmission and acquisition risk
        betas = [params['beta_PSAC'], params['beta_SAC'], params['beta_Adults']]
        # R0 is district-specific and fixed
        R0 = params['R0']

        where = df.is_alive
        age_group = pd.cut(df.loc[where, 'age_years'], [0, 4, 14, 120], labels=['PSAC', 'SAC', 'Adults'],
                           include_lowest=True)
        age_group.name = 'age_group'
        beta_by_age_group = pd.Series(betas, index=['PSAC', 'SAC', 'Adults'])
        beta_by_age_group.index.name = 'age_group'

        # --------------------- get the size of reservoir per district ---------------------
        # returns the mean worm burden and the total worm burden by age and district
        mean_count_burden_district_age_group = \
            df.loc[where].groupby(['district_of_residence', age_group], observed=False)[
                prop('aggregate_worm_burden')].agg(['mean', 'size'])

        # get population size by district
        district_count = df.loc[where].groupby(by='district_of_residence', observed=False)[
            'district_of_residence'].count()
        # mean worm burden by age multiplied by exposure rates of each age-gp
        beta_contribution_to_reservoir = mean_count_burden_district_age_group['mean'] * beta_by_age_group
        # weighted mean of the worm burden, considering the size of each age-gp in district
        to_get_weighted_mean = mean_count_burden_district_age_group['size'] / district_count

        # weighted mean worm burden * exposure rates -> age-specific contribution to reservoir
        age_worm_burden = beta_contribution_to_reservoir * to_get_weighted_mean
        # sum all contributions to district reservoir of infection
        reservoir = age_worm_burden.groupby(['district_of_residence'], observed=False).sum()

        # --------------------- estimate background risk of infection ---------------------

        current_prevalence = len(df[df['is_alive'] & (df[prop('infection_status')] != 'Non-infected')]
                         ) / len(df[df.is_alive])
        baseline_prevalence = params['baseline_prevalence']  # baseline prevalence for species in 2010

        # this returns positive value if current_prevalence lower than baseline_prevalence and
        # increases baseline_risk value
        # if current_prevalence > baseline_prevalence, value returned is 0 and no additional risk applied
        background_risk = max(0, global_params['baseline_risk'] * (1 + global_params['scaling_factor_baseline_risk'] *
                                                           (current_prevalence - baseline_prevalence)))

        reservoir += background_risk  # add the background reservoir to every district

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

        # schedule maturation of the established worms
        # at this point, the person has become infected but this is not recorded until the worms mature
        # no indicator for person harbouring only juvenile worms
        for person_id, num_new_worms in to_establish.items():
            date_of_maturation = random_date(self.sim.date + pd.DateOffset(days=30),
                                             self.sim.date + pd.DateOffset(days=55), rng)
            self.sim.schedule_event(
                SchistoMatureWorms(
                    module=self.module,
                    species=self.species,
                    person_id=person_id,
                    number_of_worms_that_mature=num_new_worms,
                ),
                date_of_maturation
            )


class SchistoMatureWorms(Event, IndividualScopeEventMixin):
    """Represents the maturation of worms to adult worms.
    * Increases the aggregate worm burden of an individual upon maturation of the worms
    * Schedules the natural death of worms and symptoms development if High-infection
    * Updates the infection status and symptoms of the person accordingly."""

    def __init__(self, module: Module, species: SchistoSpecies, person_id: int, number_of_worms_that_mature: int):
        super().__init__(module, person_id=person_id)
        self.species = species
        self.number_of_worms_that_mature = number_of_worms_that_mature

    def apply(self, person_id):
        df = self.sim.population.props
        prop = self.species.prefix_species_property
        params = self.species.params

        person = df.loc[person_id]

        if not person.is_alive:
            return

        # increase worm burden
        df.loc[person_id, prop('aggregate_worm_burden')] += self.number_of_worms_that_mature

        # schedule the natural death of the worms
        self.sim.schedule_event(
            SchistoWormsNatDeath(module=self.module,
                                 person_id=person_id,
                                 number_of_worms_that_die=self.number_of_worms_that_mature,
                                 species=self.species),
            self.sim.date + DateOffset(years=params['worm_lifespan'])
        )

        self.species.update_infectious_status_and_symptoms(idx=pd.Index([person_id]))


class SchistoWormsNatDeath(Event, IndividualScopeEventMixin):
    """Represents the death of adult worms.
     * Decreases the aggregate worm burden of an individual upon natural death of the adult worm.
     * Updates the infection status and the symtoms of the person accordingly.
    Nb. This event checks the last day of PZQ treatment and if has been less than the lifespan of the worm it doesn't
    do anything (because the worms for which this event was raised will since have been killed by the PZQ)."""

    def __init__(self, module: Module, species: SchistoSpecies, person_id: int, number_of_worms_that_die: int):
        super().__init__(module, person_id=person_id)
        self.species = species
        self.number_of_worms_that_die = number_of_worms_that_die

    def apply(self, person_id):
        df = self.sim.population.props
        prop = self.species.prefix_species_property
        params = self.species.params
        person = df.loc[person_id]

        if not person.is_alive:
            return

        worms_now = person[prop('aggregate_worm_burden')]
        date_last_pzq = person[f'{self.module.module_prefix}_last_PZQ_date']
        date_worm_acquisition = self.sim.date - pd.DateOffset(years=params['worm_lifespan'])
        has_had_treatment_since_worm_acquisition = date_last_pzq >= date_worm_acquisition

        if worms_now == 0:
            return  # Do nothing if there are currently no worms

        if not has_had_treatment_since_worm_acquisition:
            # This event is for worms that have matured since the last treatment.
            df.loc[person_id, prop('aggregate_worm_burden')] = max(0, worms_now - self.number_of_worms_that_die)
            self.species.update_infectious_status_and_symptoms(idx=pd.Index([person_id]))


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
                    beneficiaries_ids=idx_to_receive_mda
                ),
                topen=self.sim.date,
                tclose=self.sim.date + pd.DateOffset(months=1),
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

        age_range = _AGE_GROUPS[age_group]  # returns a tuple (a,b) a <= age_group <= b

        eligible = df.index[
            df['is_alive']
            & (df['district_of_residence'] == district)
            & df['age_years'].between(age_range[0], age_range[1])
            ]

        return eligible[rng.random_sample(len(eligible)) < coverage].to_list()


class SchistoWashScaleUp(RegularEvent, PopulationScopeEventMixin):
    """
    When WASH is implemented, two processes will occur:
    *1 scale the proportion of the population susceptible to schisto infection
    assuming that WASH reduces individual risk of infection by 0.6
    *2 increase the proportion of the population who have access to
    sanitation and clean drinking water

    Event is initially scheduled by initialise_simulation on specified date
    This is a one-off event
    """

    def __init__(self, module):
        super().__init__(
            module, frequency=DateOffset(years=100)
        )

    def apply(self, population):
        df = population.props

        # need to reduce proportion susceptible by 60% for both species

        # Reduce susceptibility for mansoni
        self.module.reduce_susceptibility(df, species_column='ss_sm_susceptibility')

        # Reduce susceptibility for haematobium
        self.module.reduce_susceptibility(df, species_column='ss_sh_susceptibility')

        # scale-up properties related to WASH
        # set the properties to False for everyone
        df['li_unimproved_sanitation'] = False
        df['li_no_clean_drinking_water'] = False
        df['li_no_access_handwashing'] = False
        df['li_date_acquire_improved_sanitation'] = self.sim.date
        df['li_date_acquire_access_handwashing'] = self.sim.date
        df['li_date_acquire_clean_drinking_water'] = self.sim.date


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
                    dx_tests_to_run="UF_schisto_test_highWB", hsi_event=self
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
                    dx_tests_to_run="KK_schisto_test_highWB", hsi_event=self
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
        dosage = self.module._calculate_praziquantel_dosage(person_id)

        if self.get_consumables(item_codes={self.module.item_code_for_praziquantel: dosage}):
            self.module.do_effect_of_treatment(person_id=person_id)


class HSI_Schisto_MDA(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event for providing one or more persons with PZQ as part of a Mass Drug
    Administration (MDA). Note that the `person_id` declared as the `target` of this `HSI_Event` is only one of the
    beneficiaries. This is in, effect, a "batch job" of individual HSI being handled within one HSI, for the sake of
    computational efficiency."""

    def __init__(self, module, person_id, beneficiaries_ids: Optional[Sequence] = None):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Schisto)
        self.beneficiaries_ids = beneficiaries_ids

        self.TREATMENT_ID = 'Schisto_MDA'

        # self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({
        #     'EPI': len(beneficiaries_ids) if beneficiaries_ids else 1})
        # self.ACCEPTED_FACILITY_LEVEL = '1a'
        # The `EPI` appointment is appropriate because it's a very small appointment, and we note that this is used in
        # the coding for 'de-worming'-type activities in the DHIS2 data. We show that expect there will be one of these
        # appointments for each of the beneficiaries, whereas, in fact, it may be more realistic to consider that the
        # real requirement is fewer than that.
        # This class is created when running `tlo_hsi_event.py`, which doesn't provide the argument `beneficiaries_ids`
        # but does require that `self.EXPECTED_APPT_FOOTPRINT` is valid. So, in this case, we let
        # `self.EXPECTED_APPT_FOOTPRINT` show that this requires 1 * that appointment type.
        # EPI footprint at level 1a = Clinical 0.06, Nursing_and_Midwifery 4.0, Pharmacy 1.68

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
        total_dosage = sum(self.module._calculate_praziquantel_dosage(pid) for pid in beneficiaries_still_alive)

        # Let the key consumable be "optional" in order that provision of the treatment is NOT conditional on the drugs
        # being available.This is because we expect that special planning would be undertaken in order to ensure the
        # availability of the drugs on the day(s) when the MDA is planned.
        if self.get_consumables(
            optional_item_codes={self.module.item_code_for_praziquantel_MDA: total_dosage}
        ):
            self.module.do_effect_of_treatment(person_id=beneficiaries_still_alive)

        # Return the update appointment that reflects the actual number of beneficiaries.
        return self.make_appt_footprint({'ConWithDCSA': len(beneficiaries_still_alive) * 0.5})


class SchistoPersonDaysLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """This is a regular event (every day) that logs the person-days infected """
        super().__init__(module, frequency=DateOffset(days=1))
        assert isinstance(module, Schisto)

    def apply(self, population):
        """
        Log the numbers of people infected with any species of schisto and high-burden infections
        sum these in SchistoLoggingEvent each year to get person-years infected
        """
        df = self.sim.population.props
        df_alive = df.loc[df.is_alive].copy()

        # Precompute categories
        df_alive['age_group'] = df_alive.age_years.map(self.module.age_group_mapper)
        df_alive['species_prefix'] = df_alive.apply(
            lambda row: 'sm' if row['ss_sm_infection_status'] != 'None' else 'sh', axis=1
        )
        df_alive['infection_level'] = df_alive.apply(
            lambda row: row[f"ss_{row['species_prefix']}_infection_status"], axis=1
        )

        # Exclude non-infected individuals
        df_alive = df_alive[df_alive['infection_level'] != 'Non-infected']

        # Group by species, age group, infection level, and district
        grouped_counts = df_alive.groupby(
            ['species_prefix', 'age_group', 'infection_level', 'district_of_residence']
        ).size()

        # Update the log_person_days DataFrame
        for idx, count in grouped_counts.items():
            species = 'mansoni' if idx[0] == 'sm' else 'haematobium'
            self.module.log_person_days.loc[
                (species, idx[1], idx[2], idx[3]), 'person_days'
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
            # _spec.log_mean_worm_burden()  # todo revert this if needed

        # PZQ treatment episodes
        df = population.props
        now = self.sim.date

        # todo this is logging MDA as well as treatment
        new_tx = len(
            df[
                (df.ss_last_PZQ_date >= (now - DateOffset(months=self.repeat)))
                & (df.ss_last_PZQ_date >= (now - DateOffset(months=self.repeat)))
                ]
        )
        # treatment logger
        treatment_episodes = {
            'treatment_episodes': new_tx,
        }
        logger.info(
            key='schisto_treatment_episodes',
            data=treatment_episodes,
            description='Counts of treatment occurring in timeperiod'
        )

        # log person-days of infection by low, moderate and high for all, SAC and PSAC separately
        logger.info(
            key='Schisto_person_days_infected',
            data=flatten_multi_index_series_into_dict_for_logging(self.module.log_person_days['person_days']),
            description='Counts of person-days infected by species'
        )
        # Reset the daily counts for the next month
        self.module.log_person_days.loc[:, 'person_days'] = 0

        # extract prevalence of either infection by age-group and district
        age_grp = df.loc[df['is_alive'], 'age_years'].map(self.module.age_group_mapper)

        infection_mask = (df['ss_sm_infection_status'] != 'Non-infected') | (
                df['ss_sh_infection_status'] != 'Non-infected')

        # Step 2: Filter the DataFrame to include only those who are alive and infected
        infected = df.loc[df['is_alive'] & infection_mask].groupby(
            by=[
                df.loc[df['is_alive'] & infection_mask, 'district_of_residence'],
                age_grp
            ],
            observed=False
        ).size()
        infected.index.rename(['district_of_residence', 'age_group'], inplace=True)

        alive = df.loc[df['is_alive']].groupby(
            by=[
                df.loc[df['is_alive'], 'district_of_residence'],
                age_grp
            ],
            observed=False
        ).size()
        alive.index.rename(['district_of_residence', 'age_group'], inplace=True)

        logger.info(
            key=f'number_infected_any_species',
            description='Counts of infection status with this species by age-group and district.',
            data={
                'number_infected': flatten_multi_index_series_into_dict_for_logging(infected),
            },
        )
        logger.info(
            key=f'number_in_subgroup',
            description='Counts of infection status with this species by age-group and district.',
            data={
                'number_alive': flatten_multi_index_series_into_dict_for_logging(alive),
            },
        )

        # WASH properties
        unimproved_sanitation = len(
            df[df.li_unimproved_sanitation & df.is_alive]
        ) / len(df[df.is_alive]) if len(df[df.is_alive]) else 0

        no_access_handwashing = len(
            df[df.li_no_access_handwashing & df.is_alive]
        ) / len(df[df.is_alive]) if len(df[df.is_alive]) else 0

        no_clean_drinking_water = len(
            df[df.li_no_clean_drinking_water & df.is_alive]
        ) / len(df[df.is_alive]) if len(df[df.is_alive]) else 0

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
