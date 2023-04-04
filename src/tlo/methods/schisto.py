from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from tlo import Date, DateOffset, Module, Parameter, Property, Types, logging
from tlo.analysis.utils import flatten_multi_index_series_into_dict_for_logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom
from tlo.util import random_date

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Definition of the age-groups used in the module, as a tuple of two integers (a,b) such that the given age group
#  is in range a <= group <= b. i.e.,
#     0 <= PSAC <= 4
#     5 <= SAC <= 14
#     15 <= Adults
#     0 <= All
_AGE_GROUPS = {'PSAC': (0, 4), 'SAC': (5, 14), 'Adults': (15, 120), 'All': (0, 120)}


class Schisto(Module):
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
        'prob_sent_to_lab_test_children': Parameter(Types.REAL,
                                                    'Probability that infected child gets sent to lab test'),
        'prob_sent_to_lab_test_adults': Parameter(Types.REAL,
                                                  'Probability that an infected adults gets sent to lab test'),
        'delay_till_hsi_a_repeated': Parameter(Types.REAL,
                                               'Time till seeking healthcare again after not being sent to schisto '
                                               'test: start'),
        'delay_till_hsi_b_repeated': Parameter(Types.REAL,
                                               'Time till seeking healthcare again after not being sent to schisto '
                                               'test: end'),
        'PZQ_efficacy': Parameter(Types.REAL,
                                  'The efficacy of Praziquantel in clearing burden of any Schistosomiasis worm '
                                  'species'),
        'MDA_coverage_historical': Parameter(Types.DATA_FRAME,
                                             'Probability of getting PZQ in the MDA for PSAC, SAC and Adults in '
                                             'historic rounds'),
        'MDA_coverage_prognosed': Parameter(Types.DATA_FRAME,
                                            'Probability of getting PZQ in the MDA for PSAC, SAC and Adults in future '
                                            'rounds, with the frequency given in months'),
    }

    def __init__(self, name=None, resourcefilepath=None, mda_execute=True):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.mda_execute = mda_execute

        # Create pointer that will be to dict of disability weights
        self.disability_weights = None

        # Create pointer that will be to the item_code for praziquantel
        self.item_code_for_praziquantel = None

        # Create the instances of `SchistoSpecies` that will represent the two species being considered
        self.species = {_name: SchistoSpecies(self, name=_name) for _name in ('mansoni', 'haematobium')}

        # Add properties and parameters declared by each species:
        for _spec in self.species.values():
            self.PROPERTIES.update(_spec.get_properties())
            self.PARAMETERS.update(_spec.get_parameters())

        # Property names for infection_status of all species
        self.cols_of_infection_status = [_spec.infection_status_property for _spec in self.species.values()]

        self.districts = None

        # Age-group mapper
        s = pd.Series(index=range(1 + 120), data='object')
        for name, (low_limit, high_limit) in _AGE_GROUPS.items():
            if name != 'All':
                s.loc[(s.index >= low_limit) & (s.index <= high_limit)] = name
        self.age_group_mapper = s.to_dict()

    def read_parameters(self, data_folder):
        """Read parameters and register symptoms."""

        # Load parameters
        workbook = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Schisto.xlsx', sheet_name=None)
        self.parameters = self._load_parameters_from_workbook(workbook)
        for _spec in self.species.values():
            self.parameters.update(_spec.load_parameters_from_workbook(workbook))

        # Register symptoms
        symptoms_df = workbook['Symptoms']
        self._register_symptoms(symptoms_df.set_index('Symptom')['HSB_mapped_symptom'].to_dict())

    def pre_initialise_population(self):
        """Do things before generating the population (but after read_parameters and any parameter updating)."""

        # Define districts that this module will operate in:
        self.districts = self.sim.modules['Demography'].districts  # <- all districts

        # Call `pre_initialise_population` for each `SchistoSpecies` helper module.
        for _spec in self.species.values():
            _spec.update_parameters_from_schisto_module()

    def initialise_population(self, population):
        """Set the property values for the initial population."""

        df = population.props
        df.loc[df.is_alive, f'{self.module_prefix}_last_PZQ_date'] = pd.NaT

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

        # Look-up item code for Praziquantel
        if 'HealthSystem' in self.sim.modules:
            self.item_code_for_praziquantel = self._get_item_code_for_praziquantel()

        # Schedule the logging event
        sim.schedule_event(SchistoLoggingEvent(self), sim.date)

        # Schedule MDA events
        if self.mda_execute:
            self._schedule_mda_events()

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        All children are born without an infection, even if the mother is infected.

        :param mother_id: the ID for the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props
        df.at[child_id, f'{self.module_prefix}_last_PZQ_date'] = pd.NaT
        # df.at[child_id, 'ss_scheduled_hsi_date'] = pd.NaT

        for _spec in self.species.values():
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

    def do_on_presentation_with_symptoms(self, person_id: int, symptoms: Union[list, set, tuple]) -> None:
        """Do when person presents to the GenericFirstAppt. If the person has certain set of symptoms, refer ta HSI for
         testing."""

        set_of_symptoms_indicative_of_schisto = {'anemia', 'haematuria', 'bladder_pathology'}

        if set_of_symptoms_indicative_of_schisto.issubset(symptoms):
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Schisto_TestingFollowingSymptoms(
                    module=self,
                    person_id=person_id),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )

    def do_effect_of_treatment(self, person_id: Union[int, Sequence[int]]) -> None:
        """Do the effects of a treatment administered to a person or persons. This can be called for a person who is
        infected and receiving treatment following a diagnosis, or for a person who is receiving treatment as part of a
         Mass Drug Administration. The burden and effects of any species are alleviated by a successful treatment."""

        df = self.sim.population.props

        # Clear any symptoms caused by this module (i.e., Schisto of any species)
        self.sim.modules['SymptomManager'].clear_symptoms(person_id=person_id, disease_module=self)

        # Record in the property the date of last treatment
        df.loc[person_id, 'ss_last_PZQ_date'] = self.sim.date

        # Set properties to be not-infected (any species), and zero-out all worm burden information.
        for spec_prefix in [_spec.prefix for _spec in self.species.values()]:
            df.loc[person_id, f'{self.module_prefix}_{spec_prefix}_aggregate_worm_burden'] = 0
            df.loc[person_id, f'{self.module_prefix}_{spec_prefix}_infection_status'] = 'Non-infected'
            # df.loc[person_id, f'{self.module_prefix}_{spec_prefix}_start_of_prevalent_period'] = pd.NaT
            # df.loc[person_id, f'{self.module_prefix}_{spec_prefix}_start_of_high_infection'] = pd.NaT

    def _load_parameters_from_workbook(self, workbook) -> dict:
        """Load parameters from ResourceFile (loaded by pd.read_excel as `workbook`) that are general (i.e., not
        specific to a particular species)."""

        parameters = dict()

        # HSI and treatment params:
        param_list = workbook['Parameters'].set_index("Parameter")['Value']
        for _param_name in ('delay_till_hsi_a_repeated',
                            'delay_till_hsi_b_repeated',
                            'prob_sent_to_lab_test_children',
                            'prob_sent_to_lab_test_adults',
                            'PZQ_efficacy',
                            ):
            parameters[_param_name] = param_list[_param_name]

        # MDA coverage - historic
        historical_mda = workbook['MDA_historical_Coverage'].set_index(['District', 'Year'])[
            ['Coverage PSAC', 'Coverage SAC', 'Coverage Adults']]
        historical_mda.columns = historical_mda.columns.str.replace('Coverage ', '')
        parameters['MDA_coverage_historical'] = historical_mda

        # MDA coverage - prognosed
        prognosed_mda = workbook['MDA_prognosed_Coverage'].set_index(['District', 'Frequency'])[
            ['Coverage PSAC', 'Coverage SAC', 'Coverage Adults']]
        prognosed_mda.columns = prognosed_mda.columns.str.replace('Coverage ', '')
        parameters['MDA_coverage_prognosed'] = prognosed_mda

        return parameters

    def _register_symptoms(self, symptoms: dict) -> None:
        """Register the symptoms with the `SymptomManager`.
        :params symptoms: The symptoms that are used by this module in a dictionary of the form, {<symptom>:
        <generic_symptom_similar>}. Each symptom is associated with the average healthcare seeking behaviour."""
        generic_symptoms = self.sim.modules['SymptomManager'].generic_symptoms
        self.sim.modules['SymptomManager'].register_symptom(*[
            Symptom(name=_symp) for _symp in symptoms if _symp not in generic_symptoms
        ])

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

    def _get_item_code_for_praziquantel(self) -> int:
        """Look-up the item code for Praziquantel"""
        return self.sim.modules['HealthSystem'].get_item_code_from_item_name("Praziquantel, 600 mg (donated)")

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

        for (district, frequency_in_months), cov in self.parameters['MDA_coverage_prognosed'].iterrows():
            assert district in self.sim.modules['Demography'].districts, f'District {district} is not recognised.'
            self.sim.schedule_event(
                SchistoMDAEvent(self,
                                district=district,
                                coverage=cov.to_dict(),
                                months_between_repeats=frequency_in_months if frequency_in_months > 0 else None),
                Date(year=year_first_simulated_mda, month=7, day=1)
            )


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
            # 'symptoms': Parameter(Types.LIST, 'Symptoms of the schistosomiasis infection, dependent on the module'),
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
            'reservoir_2010': Parameter(Types.DATA_FRAME,
                                        'Initial reservoir of infectious material per district in 2010'),
            'gamma_alpha': Parameter(Types.DATA_FRAME, 'Parameter alpha for Gamma distribution for harbouring rates'),
            'R0': Parameter(Types.DATA_FRAME, 'Effective reproduction number, for the FOI'),
        }
        return {self._prefix_species_parameter(k): v for k, v in params.items()}

    def get_properties(self):
        """The species-specific properties for this species."""
        properties = {
            'infection_status': Property(
                Types.CATEGORICAL, 'Current status of schistosomiasis infection for this species',
                categories=['Non-infected', 'Low-infection', 'High-infection']),
            'aggregate_worm_burden': Property(
                Types.INT, 'Number of mature worms of this species in the individual'),
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
        for _param_name in ('beta_PSAC',
                            'beta_SAC',
                            'beta_Adults',
                            'worm_lifespan',
                            'worms_fecundity',
                            'high_intensity_threshold',
                            'low_intensity_threshold',
                            'high_intensity_threshold_PSAC'
                            ):
            parameters[_param_name] = param_list[f'{_param_name}_{self.name}']

        # Baseline reservoir size and other district-related params (alpha and R0)
        schisto_initial_reservoir = workbook[f'District_Params_{self.name}'].set_index("District")
        parameters['reservoir_2010'] = schisto_initial_reservoir['Reservoir']
        parameters['gamma_alpha'] = schisto_initial_reservoir['alpha_value']
        parameters['R0'] = schisto_initial_reservoir['R0_value']

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
        self._assign_initial_harbouring_rate(population)

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
        rng = self.schisto_module.rng

        # Assign the default for a newly born child
        df.at[child_id, prop('infection_status')] = 'Non-infected'
        df.at[child_id, prop('aggregate_worm_burden')] = 0

        # Generate the harbouring rate depending on a district of residence.
        district = df.loc[child_id, 'district_of_residence']
        df.at[child_id, prop('harbouring_rate')] = rng.gamma(params['gamma_alpha'][district], size=1)

    def update_infectious_status_and_symptoms(self, idx: pd.Index) -> None:
        """Updates the infection status and symptoms based on the current aggregate worm burden of this species.
         * Assigns the 'infection status' (High-infection, Low-infection, Non-infected) to the persons with ids given
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
            low_group = ~high_group & (agg_wb >= params["low_intensity_threshold"])
            status[high_group] = "High-infection"
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

        # Impose symptoms for those newly having 'High-infection' status
        newly_have_high_infection = (original_status != 'High-infection') & (correct_status == 'High-infection')
        idx_newly_have_high_infection = newly_have_high_infection.index[newly_have_high_infection]
        _impose_symptoms_of_high_intensity_infection(idx=idx_newly_have_high_infection)

        # Update status for those whose status is changing
        idx_changing = correct_status.index[original_status != correct_status]
        df.loc[idx_changing, prop('infection_status')] = correct_status.loc[idx_changing]

        # Remove symptoms if there is no cause High-infection status caused by either species
        #  NB. This is a limitation because there is no possibility of species-specific removal of symptoms. So, if a
        #  person has two infections causing 'High-infection' and one is reduced below 'High-infection', symptoms will
        #  persist as if the person still had two causes of 'High-infection'. The symptoms would not be removed until
        #  both the aggregate worm burden of both species is reduced.
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

    def _assign_initial_harbouring_rate(self, population) -> None:
        """Assign a harbouring rate to every individual in the initial populattion (based on their district of
        residence)."""
        df = population.props
        prop = self.prefix_species_property
        params = self.params
        districts = self.schisto_module.districts
        rng = self.schisto_module.rng

        for district in districts:
            in_the_district = df.index[df['district_of_residence'] == district]
            hr = params['gamma_alpha'][district]
            df.loc[in_the_district, prop('harbouring_rate')] = rng.gamma(hr, size=len(in_the_district))

    def _assign_initial_worm_burden(self, population) -> None:
        """Assign initial distribution of worms to each person (based on district and age-group)."""
        df = population.props
        prop = self.prefix_species_property
        params = self.params
        districts = self.schisto_module.districts
        rng = self.schisto_module.rng

        reservoir = params['reservoir_2010']

        for district in districts:
            # Determine a 'contact rate' for each person
            in_the_district = df.index[df['district_of_residence'] == district]
            contact_rates = pd.Series(1, index=in_the_district)
            for age_group in ['PSAC', 'SAC', 'Adults']:
                age_range = _AGE_GROUPS[age_group]
                in_the_district_and_age_group = \
                    df.index[(df['district_of_residence'] == district) &
                             (df['age_years'].between(age_range[0], age_range[1]))]
                contact_rates.loc[in_the_district_and_age_group] *= params[f"beta_{age_group}"]

            if len(in_the_district):
                harbouring_rates = df.loc[in_the_district, prop('harbouring_rate')].values
                rates = np.multiply(harbouring_rates, contact_rates)
                reservoir_distr = int(reservoir[district] * len(in_the_district))

                # Distribute a worm burden among persons, according to their 'contact rate'
                if (reservoir_distr > 0) and (rates.sum() > 0):
                    chosen = rng.choice(in_the_district, reservoir_distr, p=rates / rates.sum())
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

        age_grp = df.loc[df.is_alive].age_years.map(self.schisto_module.age_group_mapper)

        data = df.loc[df.is_alive].groupby(by=[
            df.loc[df.is_alive, self.infection_status_property],
            df.loc[df.is_alive, 'district_of_residence'],
            age_grp
        ]).size()
        data.index.rename('infection_status', level=0, inplace=True)

        logger.info(
            key=f'infection_status_{self.name}',
            data=flatten_multi_index_series_into_dict_for_logging(data),
            description='Counts of infection status with this species by age-group and district.'
        )


class SchistoInfectionWormBurdenEvent(RegularEvent, PopulationScopeEventMixin):
    """A recurring event that causes infection of people with this species.
     * Determines who becomes infected (using worm burden and reservoir of infectious material.
     * Schedules `SchistoMatureWorms` for when the worms mature to adult worms."""

    def __init__(self, module: Module, species: SchistoSpecies):
        super().__init__(module, frequency=DateOffset(months=1))
        self.species = species

    def apply(self, population):
        df = population.props
        params = self.species.params
        rng = self.module.rng
        prop = self.species.prefix_species_property

        betas = [params['beta_PSAC'], params['beta_SAC'], params['beta_Adults']]
        R0 = params['R0']

        where = df.is_alive
        age_group = pd.cut(df.loc[where, 'age_years'], [0, 4, 14, 120], labels=['PSAC', 'SAC', 'Adults'],
                           include_lowest=True)
        age_group.name = 'age_group'
        beta_by_age_group = pd.Series(betas, index=['PSAC', 'SAC', 'Adults'])
        beta_by_age_group.index.name = 'age_group'

        # get the size of reservoir per district
        mean_count_burden_district_age_group = df.loc[where].groupby(['district_of_residence', age_group])[
            prop('aggregate_worm_burden')].agg([np.mean, np.size])
        district_count = df.loc[where].groupby(by='district_of_residence')['district_of_residence'].count()
        beta_contribution_to_reservoir = mean_count_burden_district_age_group['mean'] * beta_by_age_group
        to_get_weighted_mean = mean_count_burden_district_age_group['size'] / district_count
        age_worm_burden = beta_contribution_to_reservoir * to_get_weighted_mean
        reservoir = age_worm_burden.groupby(['district_of_residence']).sum()

        # harbouring new worms
        contact_rates = age_group.map(beta_by_age_group).astype(float)
        harbouring_rates = df.loc[where, prop('harbouring_rate')]
        rates = harbouring_rates * contact_rates
        worms_total = reservoir * R0
        draw_worms = pd.Series(
            rng.poisson(
                (df.loc[where, 'district_of_residence'].map(worms_total) * rates).fillna(0.0)
            ),
            index=df.index[where]
        )

        # density dependent establishment
        param_worm_fecundity = params['worms_fecundity']
        established = self.module.rng.random_sample(size=sum(where)) < np.exp(
            df.loc[where, prop('aggregate_worm_burden')] * -param_worm_fecundity
        )
        to_establish = draw_worms[(draw_worms > 0) & established].to_dict()

        # schedule maturation of the established worms
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


class HSI_Schisto_TestingFollowingSymptoms(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event for a person with symptoms who has been referred from the FirstAppt
    for testing at the clinic."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Schisto)

        under_5 = self.sim.population.props.at[person_id, 'age_years'] <= 5
        self.TREATMENT_ID = 'Schisto_Treatment'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Under5OPD' if under_5 else 'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self._num_occurrences = 0

    def apply(self, person_id, squeeze_factor):
        self._num_occurrences += 1

        df = self.sim.population.props
        person = df.loc[person_id]
        params = self.module.parameters
        cols_of_infection_status = self.module.cols_of_infection_status

        # Determine if the person will be tested now
        under_15 = person.age_years <= 15
        will_test = self.module.rng.random_sample() < (
            params['prob_sent_to_lab_test_children'] if under_15 else params['prob_sent_to_lab_test_adults']
        )

        if will_test:
            # Determine if they truly are infected (with any of the species)
            is_infected = (person.loc[cols_of_infection_status] != 'Non-infected').any()

            if is_infected & will_test:
                # If they are infected and will test, schedule a treatment HSI:
                self.module.sim.modules['HealthSystem'].schedule_hsi_event(
                    HSI_Schisto_TreatmentFollowingDiagnosis(
                        module=self.module,
                        person_id=person_id),
                    topen=self.sim.date,
                    tclose=None,
                    priority=0
                )

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
        if self.get_consumables(item_codes=self.module.item_code_for_praziquantel):
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
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({
            'EPI': len(beneficiaries_ids) if beneficiaries_ids else 1})
        # The `EPI` appointment is appropriate because it's a very small appointment, and we note that this is used in
        # the coding for 'de-worming'-type activities in the DHIS2 data. We show that expect there will be one of these
        # appointments for each of the beneficiaries, whereas, in fact, it may be more realistic to consider that the
        # real requirement is fewer than that.
        # This class is created when running `tlo_hsi_event.py`, which doesn't provide the argument `beneficiaries_ids`
        # but does require that `self.EXPECTED_APPT_FOOTPRINT` is valid. So, in this case, we let
        # `self.EXPECTED_APPT_FOOTPRINT` show that this requires 1 * that appointment type.

        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        """Provide the treatment to the beneficiaries of this HSI."""

        # Find which of the beneficiaries are still alive
        beneficiaries_still_alive = list(set(self.beneficiaries_ids).intersection(
            self.sim.population.props.index[self.sim.population.props.is_alive]
        ))

        # Let the key consumable be "optional" in order that provision of the treatment is NOT conditional on the drugs
        # being available.This is because we expect that special planning would be undertaken in order to ensure the
        # availability of the drugs on the day(s) when the MDA is planned.
        if self.get_consumables(
            optional_item_codes={self.module.item_code_for_praziquantel: len(beneficiaries_still_alive)}
        ):
            self.module.do_effect_of_treatment(person_id=beneficiaries_still_alive)

        # Return the update appointment that reflects the actual number of beneficiaries.
        return self.make_appt_footprint({'EPI': len(beneficiaries_still_alive)})


class SchistoLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """This is a regular event (every month) that causes the logging for each species."""
        super().__init__(module, frequency=DateOffset(months=1))
        assert isinstance(module, Schisto)

    def apply(self, population):
        """Call `log_infection_status` for each species."""
        for _spec in self.module.species.values():
            _spec.log_infection_status()
