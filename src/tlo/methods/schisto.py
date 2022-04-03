import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom
from tlo.util import random_date
from tlo import Date

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def map_age_groups(age_group):
    """Helper function for obtaining the age range for each age_group
    It returns a tuple of two integers (a,b) such that the given age group is in range a <= group <= b,i.e.:
        0 <= PSAC <= 4
        5 <= SAC <= 14
        15 <= Adults
        0 <= All

    this will cover all ages because we look at the int variable of age
    :param age_group: 'SAC', 'PSAC', 'Adults', 'All'
    """
    assert age_group in ('SAC', 'PSAC', 'Adults', 'All'), "Incorrect age group"
    return {'PSAC': (0, 4), 'SAC': (5, 14), 'Adults': (15, 120), 'All': (0, 120)}[age_group]


class Schisto(Module):
    """Schistosomiasis module"""

    INIT_DEPENDENCIES = {'Demography', 'SymptomManager'}

    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden'}

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {}

    # Declare Causes of Disability
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
                                               'Time till seeking healthcare again '
                                               'after not being sent to schisto test, start'),
        'delay_till_hsi_b_repeated': Parameter(Types.REAL,
                                               'Time till seeking healthcare again '
                                               'after not being sent to schisto test, end'),

        'PZQ_efficacy': Parameter(Types.REAL,
                                  'The efficacy of Praziquantel in clearing burden of any Schistosomiasis worm specieis'),

        # MDA parameters
        # 'years_till_first_MDA': Parameter(Types.REAL, 'Years till the first historical MDA'),  # todo - remove from excel?

        'MDA_coverage_historical': Parameter(Types.DATA_FRAME,
                                             'Probability of getting PZQ in the MDA for PSAC, SAC and Adults in historic rounds'),
        'MDA_coverage_prognosed': Parameter(Types.DATA_FRAME,
                                             'Probability of getting PZQ in the MDA for PSAC, SAC and Adults in future rounds, with the frequency given in months'),
    }

    def __init__(self, name=None, resourcefilepath=None, mda_execute=True):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.mda_execute = mda_execute

        # self.districts = sim.modules['Demography'].districts  # if we want it to be all districts:
        # Iwona used in her thesis (high burden districts):
        # todo - make it work for all districts
        self.districts = ['Blantyre', 'Chiradzulu', 'Mulanje', 'Nsanje', 'Nkhotakota', 'Phalombe']

        # Create pointer that will be to dict of disability weights
        self.disability_weights = None

        # Create pointer that will be to the item_code for praziquantel
        self.item_code_for_praziquantel = None

        # Create the instances of `SchistoSpecies` that will represent the two species being considered
        self.species = {_name: SchistoSpecies(self, name=_name) for _name in ('mansoni', 'haematobium')}

        # Add properties and parameters declared by each species:
        for _spec in self.species.values():
            self.PROPERTIES.update(_spec.PROPERTIES)
            self.PARAMETERS.update(_spec.PARAMETERS)

        # Property names for infection_status of all species
        self.cols_of_infection_status = [_p for _p in self.PROPERTIES
                                         if (_p.startswith(self.module_prefix) & _p.endswith('_infection_status'))]

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

        # Call `pre_initialise_population` for each `SchistoSpecies` helper module.
        for _spec in self.species.values():
            _spec._update_parameters_from_schisto_module()

    def initialise_population(self, population):
        """Set the property values for the initial population."""
        df = population.props
        df.loc[df.is_alive, f'{self.module_prefix}_last_PZQ_date'] = pd.Timestamp(year=1900, month=1,
                                                                                  day=1)  # for simplicity to avoid NaT
        # df.loc[df.is_alive, 'ss_scheduled_hsi_date'] = pd.NaT

        for _spec in self.species.values():
            _spec.initialise_population(population)

    def initialise_simulation(self, sim):
        """Get ready for simulation start."""

        for _spec in self.species.values():
            _spec.initialise_simulation(sim)

        # DALY weights
        if 'HealthBurden' in self.sim.modules:
            self.disability_weights = self._get_disability_weight()

        # Look-up item code for Praziquantel
        self.item_code_for_praziquantel = self._get_item_code_for_praziquantel()

        if self.mda_execute:
            self._schedule_mda_events()

        # todo switch to turn these logging events on/ off, and consolidate these logging events / remove them
        # Schedule logging events:
        sim.schedule_event(SchistoLoggingTotalEvent(self), sim.date + DateOffset(months=0))
        # sim.schedule_event(SchistoPrevalentDaysLoggingEvent(self), pd.Timestamp(year=sim.date.year, month=12, day=31, hour=23))
        # sim.schedule_event(SchistoLoggingEvent(self.schisto_module), sim.date + DateOffset(months=0))
        # sim.schedule_event(SchistoLoggingPrevDistrictEvent(self.schisto_module), sim.date + DateOffset(years=70))
        # sim.schedule_event(SchistoParamFittingLogging(self), sim.date + DateOffset(years=15))

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
        dw = pd.Series(symptoms_being_caused).apply(pd.Series).replace(self.disability_weights).fillna(0).sum(
            axis=1).clip(upper=1.0)

        # Return pd.Series that include entries for all alive persons (filling 0.0 where they do not have any symptoms)
        df = self.sim.population.props
        return pd.Series(index=df.index[df.is_alive], data=0.0).add(dw, fill_value=0.0)

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

    def do_effect_of_treatment(self, person_id: int) -> None:
        """Do the effects of a treatment administered to a person. This can be called for a person who is infected
        and receiving treatment, or not infected and receiving treatment as part of a Mass Drug Administration.
        The burden and effects of any species are aleviated by a succesful treatment."""
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
                            'years_till_first_MDA'
                            ):
            parameters[_param_name] = param_list[_param_name]

        # MDA coverage - historic
        historical_mda = workbook['MDA_historical_Coverage'].set_index(['District', 'Year'])[['Coverage PSAC', 'Coverage SAC', 'Coverage Adults']]
        historical_mda.columns = historical_mda.columns.str.replace('Coverage ', '')
        parameters['MDA_coverage_historical'] = historical_mda

        # MDA coverage - prognosed
        prognosed_mda = workbook['MDA_prognosed_Coverage'].set_index(['District', 'Frequency'])[['Coverage PSAC', 'Coverage SAC', 'Coverage Adults']]
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
        get_daly_weight = lambda _code: self.sim.modules['HealthBurden'].get_daly_weight(
            _code) if _code is not None else 0.0

        return {
            symptom: get_daly_weight(dw_code) for symptom, dw_code in symptoms_to_disability_weight_mapping.items()
        }

    def _get_item_code_for_praziquantel(self) -> None:
        """Look-up the item code for Praziquantel"""
        return self.sim.modules['HealthSystem'].get_item_code_from_item_name("Praziquantel, 600 mg (donated)")

    def _schedule_mda_events(self):
        """Schedule MDA events, historical and prognosed."""

        # Schedule the district-specific MDA that have occurred:
        for (district, year), cov in self.parameters['MDA_coverage_historical'].iterrows():
            assert district in self.sim.modules['Demography'].districts, f'District {district} is not recognised.'
            self.sim.schedule_event(
                SchistoMDAEvent(self,
                                district=district,
                                coverage=cov.to_dict(),
                                months_between_repeats=None),    # todo - check that this causes it to occur once only.
                Date(year=year, month=7, day=1)
            )

        # Schedule the first occurrence (of a repeating event) of the MDA that occur after the last historical MDA
        year_last_historical_mda = self.parameters['MDA_coverage_historical'].reset_index().Year.max()
        year_first_simulated_mda = year_last_historical_mda + 1

        for (district, frequency_in_months) in self.parameters['MDA_coverage_prognosed'].iterrows():
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

    @property
    def PARAMETERS(self):
        """The species-specific parameters for this species."""
        return {self._prefix_species_parameter(k): v for k, v in self._parameters.items()}

    @property
    def PROPERTIES(self):
        """The species-specific properties for this species."""
        return {self._prefix_species_property(k): v for k, v in self._properties.items()}

    @property
    def _parameters(self):
        return {
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

    @property
    def _properties(self):
        return {
            'infection_status': Property(
                Types.CATEGORICAL, 'Current status of schistosomiasis infection for this species',
                categories=['Non-infected', 'Low-infection', 'High-infection']),
            'aggregate_worm_burden': Property(
                Types.INT, 'Number of mature worms of this species in the individual'),
            'harbouring_rate': Property(
                Types.REAL, 'Rate of harbouring new worms of this species (Poisson), drawn from gamma distribution'),

            # 'start_of_prevalent_period': Property(Types.DATE, 'Date of going from Non-infected to Infected'),
            # 'start_of_high_infection': Property(Types.DATE, 'Date of going from entering state High-inf'),
            # 'prevalent_days_this_year': Property(Types.INT, 'Cumulative days with infection in current year'),
            # 'high_inf_days_this_year': Property(Types.INT, 'Cumulative days with high-intensity infection in current year')
        }

    def _prefix_species_property(self, generic_property_name: str) -> str:
        """Add the prefix to a `generic_property_name` to get the name of the species-specific property for this
        species."""
        return f"{self.schisto_module.module_prefix}_{self.prefix}_{generic_property_name}"

    def _prefix_species_parameter(self, generic_parameter_name: str) -> str:
        """Add the prefix to a `generic_parameter_name` to get the name of the species-specific parameter for this
        species."""
        return f"{self.prefix}_{generic_parameter_name}"

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

    def pre_initialiase_population(self):
        """Do things before generating the population (but after read_parameters) and any parameter updating."""

        # Save species-specific parameter in this class, copying from the `Schisto` module. (We have to do this step
        # because the module may have updated the parameters).
        self._update_parameters_from_schisto_module()

    def initialise_population(self, population):
        """Set species-specific property values for the initial population."""

        df = population.props
        date = self.schisto_module.sim.date
        prop = self._prefix_species_property

        df.loc[df.is_alive, prop('aggregate_worm_burden')] = 0
        # df.loc[df.is_alive, prop('prevalent_days_this_year')] = 0
        # df.loc[df.is_alive, prop('start_of_prevalent_period')] = pd.NaT  # todo - not sure if needed?
        # df.loc[df.is_alive, prop('start_of_high_infection')] = pd.NaT  # todo - not sure if needed?
        # df.loc[df.is_alive, prop('high_inf_days_this_year')] = 0  # todo - not sure if needed?

        # assign a harbouring rate
        self._assign_initial_harbouring_rate(population)

        # assign initial worm burden
        self._assign_initial_worm_burden(population)

        # assign the properties recording start of the prevalent period & start of high-intensity infections
        # todo - needed?????
        # df.loc[df.is_alive & (df[prop('infection_status')] != 'Non-infected'), prop('start_of_prevalent_period')] = date
        # high_infected_idx = df.index[df[prop('infection_status')] == 'High-infection']
        # df.loc[high_infected_idx, prop('start_of_high_infection')] = date

    def initialise_simulation(self, sim):
        """
        * Schedule natural history events for those with worm burden initially.
        * Schedule the WormBurdenEvent for this species. (A recurring instance of this event will be scheduled for
        each species independently.)"
        """
        df = sim.population.props

        # Assign infection statuses and symptoms and schedule natural history events for those with worm burden initially.
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
        :param child_id: the new child
        """

        df = self.schisto_module.sim.population.props
        prop = self._prefix_species_property
        params = self.params
        rng = self.schisto_module.rng

        # Assign the default for a newly born child
        df.at[child_id, prop('infection_status')] = 'Non-infected'
        df.at[child_id, prop('aggregate_worm_burden')] = 0

        # df.at[child_id, prop('prevalent_days_this_year')] = 0
        # df.at[child_id, prop('start_of_prevalent_period')] = pd.NaT
        # df.at[child_id, prop('start_of_high_infection')] = pd.NaT
        # df.at[child_id, prop('high_inf_days_this_year')] = 0

        # Generate the harbouring rate depending on a district of residence.
        district = df.loc[child_id, 'district_of_residence']
        df.at[child_id, prop('harbouring_rate')] = rng.gamma(params['gamma_alpha'][district], size=1)

    def update_infectious_status_and_symptoms(self, idx: pd.Index) -> None:
        """
         * Assigns the 'infection status' (High-infection, Low-infection, Non-infected) to the persons with ids given
         in the `idx` argument, according their age (in years) of the person and their aggregate worm burden (of worms
         of this species).
         * Causes the onset symptoms to the persons newly with high intensity infection.
         * Removes the symptoms if a person no longer has a high intensity infection (from any species)
         """
        schisto_module = self.schisto_module
        df = schisto_module.sim.population.props
        prop = self._prefix_species_property
        params = self.params
        params = self.params
        rng = schisto_module.rng
        possible_symptoms = params["symptoms"]
        sm = self.schisto_module.sim.modules['SymptomManager']
        cols_of_infection_status = self.schisto_module.cols_of_infection_status

        if not len(idx):
            return

        def _inf_status(age: int, agg_wb: int) -> str:
            if age < 5:
                if agg_wb >= params['high_intensity_threshold_PSAC']:
                    return 'High-infection'

            if agg_wb >= params['high_intensity_threshold']:
                return 'High-infection'

            if agg_wb >= params['low_intensity_threshold']:
                return 'Low-infection'

            return 'Non-infected'

        def _impose_symptoms_of_high_intensity_infection(idx: pd.Index) -> None:
            """Assign symptoms to the person with high intensity infection.
            :param idx: indices of individuals
            """
            if not len(idx):
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

        correct_status = df.loc[idx].apply(
            lambda x: _inf_status(x['age_years'], x[prop('aggregate_worm_burden')]),
            axis=1
        )

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
        cols_of_infection_status_for_other_species = set(cols_of_infection_status) - set([prop('infection_status')])
        high_infection_any_other_species = (
                df.loc[idx, cols_of_infection_status_for_other_species] == 'High-infection').any(axis=1)
        no_longer_high_infection = idx[
            (original_status == 'High-infection') & (
                    correct_status != 'High-infection') & ~high_infection_any_other_species
            ]
        sm.clear_symptoms(person_id=no_longer_high_infection, disease_module=schisto_module)

    def log_counts_by_status_and_age_group(self) -> None:
        """Write to the log the counts by persons in each status by each age-group"""
        # todo this section could be simplified with groupby
        for age in ['PSAC', 'SAC', 'Adults', 'All']:
            self._write_to_log_count_of_states_for_age_group(age)

    def _update_parameters_from_schisto_module(self) -> None:
        """Update the internally-held parameters from the `Schisto` module that are specific to this species."""

        def remove_prefix(text, prefix):
            return text[text.startswith(prefix) and len(prefix):]

        self.params = {
            remove_prefix(k, f"{self.prefix}_"): v for k, v in self.schisto_module.parameters.items()
            if k.startswith(self.prefix)
        }

    def _assign_initial_harbouring_rate(self, population) -> None:
        """Assign a harbouring rate to every individual in the initial populattion (based on their district of  residence)."""
        df = population.props
        prop = self._prefix_species_property
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
        prop = self._prefix_species_property
        params = self.params
        districts = self.schisto_module.districts
        rng = self.schisto_module.rng

        reservoir = params['reservoir_2010']

        for district in districts:
            # Determine a 'contact rate' for each person
            in_the_district = df.index[df['district_of_residence'] == district]
            contact_rates = pd.Series(1, index=in_the_district)
            for age_group in ['PSAC', 'SAC', 'Adults']:
                age_range = map_age_groups(age_group)
                in_the_district_and_age_group = \
                    df.index[(df['district_of_residence'] == district) &
                             (df['age_years'].between(age_range[0], age_range[1]))]
                contact_rates.loc[in_the_district_and_age_group] *= params[f"beta_{age_group}"]

            if len(in_the_district):
                harbouring_rates = df.loc[in_the_district, prop('harbouring_rate')].values
                rates = np.multiply(harbouring_rates, contact_rates)
                reservoir_distr = int(reservoir[district] * len(in_the_district))

                # Distribute a worm burden among persons, according to their 'contact rate'
                chosen = rng.choice(in_the_district, reservoir_distr, p=rates / rates.sum())
                unique, counts = np.unique(chosen, return_counts=True)
                worms_per_idx = dict(zip(unique, counts))
                df[prop('aggregate_worm_burden')].update(pd.Series(worms_per_idx))

    def _schedule_death_of_worms_in_initial_population(self) -> None:
        """Schedule death of worms assigned to the initial population"""
        df = self.schisto_module.sim.population.props
        prop = self._prefix_species_property
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

    def _write_to_log_count_of_states_for_age_group(self, age_group: str) -> None:
        """Write to the log for this species the count of persons by status within a particular age-group."""

        prefix = self.name
        df = self.schisto_module.sim.population.props
        count_states = self._count_age_group_states_in_age_group(df, age_group)
        logger.info(key=f"{age_group}_{prefix}",
                    data={
                        'Non_infected': count_states['Non-infected'],
                        'Low_infections': count_states['Low-infection'],
                        'High_infections': count_states['High-infection'],
                        'Infected': count_states['infected_any'],
                        'Prevalence': count_states['Prevalence'],
                        'High-inf_Prevalence': count_states['High-inf_Prevalence'],
                        'MeanWormBurden': count_states['MeanWormBurden']
                    })

    def _count_age_group_states_in_age_group(self, age_group: str) -> dict:
        """
        :param age_group: The age-group in which to count the stages
        :return: count_states: a dictionary of counts of individuals in age_group in different states on infection
        """
        districts = self.schisto_module.districts
        df = self.schisto_module.sim.population.props

        age_range = map_age_groups(age_group)  # returns a tuple

        where = df.is_alive & df.age_years.between(*age_range) & df.district_of_residence.isin(districts)

        count = {'Non-infected': 0, 'Low-infection': 0, 'High-infection': 0}
        count.update(df.loc[where, f'{self.module.prefix}_infection_status'].value_counts().to_dict())

        count['infected_any'] = count['Low-infection'] + count['High-infection']
        count['total_pop_alive'] = count['infected_any'] + count['Non-infected']
        count['Prevalence'] = count['infected_any'] / count['total_pop_alive']
        count['High-inf_Prevalence'] = count['High-infection'] / count['total_pop_alive']
        count['MeanWormBurden'] = df.loc[where, f'{self.module.prefix}_aggregate_worm_burden'].mean()
        return count

    #
    # def _draw_worms(self, worms_total, rates, district):
    #     """ This function generates random number of new worms drawn from Poisson distribution multiplied by
    #     a product of harbouring rate and exposure rate
    #     :param district: district name
    #     :param rates: harbouring rates used for Poisson distribution, drawn from Gamma,
    #     multiplied by contact rate per age group
    #     :param worms_total: total size of the reservoir of infectious material
    #     :return harboured_worms: array of numbers of new worms for each of the persons (len = len(rates))
    #     """
    #     module = self.schisto_module
    #     params = module.parameters
    #     if worms_total == 0:
    #         return np.zeros(len(rates))
    #     rates = list(rates)
    #     R0 = params['R0'][district]
    #     worms_total *= R0
    #     harboured_worms = np.asarray([module.rng.poisson(x * worms_total, 1)[0] for x in rates]).astype(int)
    #     return harboured_worms
    #

    # def _assign_hsi_dates_initial(self, population, symptomatic_idx):
    #     """
    #     Schedules the treatment seeking to the initial population (only the clinical cases)
    #     :param population:
    #     :param symptomatic_idx: indices of people with symptoms
    #     """

    # This is not needed because the symptoms are onset and so there will be seekng from the gneric HSI automatically

    #
    # df = population.props
    # module = self.schisto_module
    # params = self.schisto_module.parameters
    # healthsystem = module.sim.modules['HealthSystem']
    #
    # for person_id in symptomatic_idx:
    #     will_seek_treatment = module.rng.rand() < params['prob_seeking_healthcare']
    #     # will_seek_treatment = self.rng.choice(['True', 'False'], size=1, p=[p, 1 - p])
    #     if will_seek_treatment:
    #         seeking_treatment_ahead = int(module.rng.uniform(params['delay_till_hsi_a'],
    #                                                          params['delay_till_hsi_b'],
    #                                                          size=1))
    #         df.at[person_id, 'ss_scheduled_hsi_date'] = module.sim.date + DateOffset(days=seeking_treatment_ahead)
    #         healthsystem.schedule_hsi_event(HSI_SchistoSeekTreatment(module.sim.modules['Schisto'], person_id=person_id),
    #                                         priority=1,
    #                                         topen=df.at[person_id, 'ss_scheduled_hsi_date'],
    #                                         tclose=df.at[person_id, 'ss_scheduled_hsi_date'] + DateOffset(weeks=502))

    # def _add_DALYs_from_symptoms(self, symptoms) -> None:
    #     raise NotImplementedError
    #
    #     # params = self.schisto_module.parameters
    #     # # todo factorize -- also, it's not in params, so remove??!!?
    #     # dalys_map = {
    #     #     'anemia': params['daly_wt_anemia'],
    #     #     'fever': params['daly_wt_fever'],
    #     #     'haematuria': params['daly_wt_haematuria'],
    #     #     'hydronephrosis': params['daly_wt_hydronephrosis'],
    #     #     'dysuria': params['daly_wt_dysuria'],
    #     #     'bladder_pathology': params['daly_wt_bladder_pathology'],
    #     #     'diarrhoea': params['daly_wt_diarrhoea'],
    #     #     'vomit': params['daly_wt_vomit'],
    #     #     'ascites': params['daly_wt_ascites'],
    #     #     'hepatomegaly': params['daly_wt_hepatomegaly']
    #     # }
    #     #
    #     # if isinstance(symptoms, list):
    #     #     symptoms = [dalys_map[s] for s in symptoms]
    #     #     return sum(symptoms)
    #     # else:
    #     #     return 0


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
        prop = self.species._prefix_species_property

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
    * Updates the infection status and symptoms of the person accordingly.
    """

    def __init__(self, module: Module, species: SchistoSpecies, person_id: int, number_of_worms_that_mature: int):
        super().__init__(module, person_id=person_id)
        self.species = species
        self.number_of_worms_that_mature = number_of_worms_that_mature

    def apply(self, person_id):
        df = self.sim.population.props
        prop = self.species._prefix_species_property
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

        #
        # if df.loc[person_id, prop('infection_status')] != 'High-infection':
        #     if df.loc[person_id, 'age_years'] < 5:
        #         threshold = params['high_intensity_threshold_PSAC']
        #     else:
        #         threshold = params['high_intensity_threshold']
        #     if df.loc[person_id, prop('aggregate_worm_burden')] >= threshold:
        #         df.loc[person_id, prop('infection_status')] = 'High-infection'
        #         df.loc[person_id, prop('start_of_high_infection')] = self.sim.date
        #
        #         # develop symptoms immediately of infection status is 'High-infection'
        #         self.species.impose_symptoms(idx=pd.Index([person_id]))
        #
        #     elif df.loc[person_id, prop('aggregate_worm_burden')] >= params['low_intensity_threshold']:
        #         if df.loc[person_id, prop('infection_status')] == 'Non-infected':
        #             df.loc[person_id, prop('infection_status')] = 'Low-infection'
        #
        # if \
        #     (df.loc[person_id, prop('infection_status')] != 'Non-infected') & \
        #         (pd.isna(df.loc[person_id, prop('start_of_prevalent_period')])):
        #     df.loc[person_id, prop('start_of_prevalent_period')] = self.sim.date
        #
        #


class SchistoWormsNatDeath(Event, IndividualScopeEventMixin):
    """Represents the death of adult worms.
     * Decreases the aggregate worm burden of an individual upon natural death of the adult worm.
     * Updates the infection status of the person accordingly. todo - and symptoms?
    Nb. This event checks the last day of PZQ treatment and if has been less than the lifespan of the worm it doesn't
    do anything (because the worms for which this event was raised will since have been killed by the PZQ)."""

    def __init__(self, module: Module, species: SchistoSpecies, person_id: int, number_of_worms_that_die: int):
        super().__init__(module, person_id=person_id)
        self.species = species
        self.number_of_worms_that_die = number_of_worms_that_die

    def apply(self, person_id):
        df = self.sim.population.props
        prop = self.species._prefix_species_property
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

        #
        # worms_now = df.loc[person_id, prop('aggregate_worm_burden')]
        # days_since_last_treatment = self.sim.date - df.loc[person_id, 'ss_last_PZQ_date']
        # days_since_last_treatment = int(days_since_last_treatment / np.timedelta64(1, 'Y'))
        #
        # if days_since_last_treatment > params['worm_lifespan']:
        #     df.loc[person_id, prop('aggregate_worm_burden')] = worms_now - self.number_of_worms_that_die
        #
        #     # clearance of the worms
        #     if df.loc[person_id, prop('aggregate_worm_burden')] < params['low_intensity_threshold']:
        #         df.loc[person_id, prop('infection_status')] = 'Non-infected'
        #         # does not matter if low or high int infection
        #         if df.loc[person_id, prop('infection_status')] != 'Non-infected':
        #             # calculate prevalent period
        #             prevalent_duration = self.sim.date - df.loc[person_id, prop('start_of_prevalent_period')]
        #             prevalent_duration = int(prevalent_duration / np.timedelta64(1, 'D')) % 365
        #             df.loc[person_id, prop('prevalent_days_this_year')] += prevalent_duration
        #             df.loc[person_id, prop('start_of_prevalent_period')] = pd.NaT
        #             df.loc[person_id, prop('start_of_high_infection')] = pd.NaT
        #     else:
        #         if df.loc[person_id, prop('infection_status')] == 'High-infection':
        #             if df.loc[person_id, 'age_years'] < 5:
        #                 threshold = params['high_intensity_threshold_PSAC']
        #             else:
        #                 threshold = params['high_intensity_threshold']
        #             if df.loc[person_id, prop('aggregate_worm_burden')] < threshold:
        #                 df.loc[person_id, prop('infection_status')] = 'Low-infection'
        #                 high_inf_duration = self.sim.date - df.loc[person_id, prop('start_of_high_infection')]
        #                 high_inf_duration = int(high_inf_duration / np.timedelta64(1, 'D')) % 365
        #                 df.loc[person_id, prop('high_inf_days_this_year')] += high_inf_duration
        #                 df.loc[person_id, prop('start_of_high_infection')] = pd.NaT


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
    """This is a Health System Interaction Event for a person being provided with PZQ as part of a Mass Drug
    Administration."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Schisto)

        self.TREATMENT_ID = 'Schisto_MDA'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ConWithDCSA': 1})
        self.ACCEPTED_FACILITY_LEVEL = '0'

    def apply(self, person_id, squeeze_factor):
        """Do the treatment for this person.
        N.B. As written, the provision of the PZQ is conditional on its availability. This is a limitation because, in
        practise, the MDA would occur once sufficient stocks are available."""
        if self.get_consumables(item_codes=self.module.item_code_for_praziquantel):
            self.module.do_effect_of_treatment(person_id=person_id)


class SchistoMDAEvent(RegularEvent, PopulationScopeEventMixin):
    """Mass-Drug administration scheduled for the population. This event schedule the occurrence of the individual-level
    HSI for the administration of drugs to each individual.
    :param district: The district in which the MDA occurs.
    :param coverage: A dictionary of the form {<age_group>: <coverage>}, where <age_group> is one of 'PSAC', 'SAC', 'Adults'
    :params months_between_repeat: The number of months between repeated occurrences of this event. (None for no repeats).
    """

    def __init__(self,
                 module: Module,
                 district: str,
                 coverage: dict,
                 months_between_repeats: int
                 ):
        super().__init__(module, frequency=DateOffset(
            months=months_between_repeats if months_between_repeats is not None else 10_000)   # todo - neater way to prevent repeating in a repeating type? (done to keep the logic in all one place)
                         )
        assert isinstance(module, Schisto)

        self.district = district
        self.coverage = coverage

    def apply(self, population):
        """Schedule the MDA HSI for each person that is reached in the MDA."""

        for age_group, cov in self.coverage.items():
            for person_id in self._select_recipients(district=self.district, age_group=age_group, coverage=cov):
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    HSI_Schisto_MDA(self.module, person_id),
                    self.sim.date)

    def _select_recipients(self, district, age_group, coverage) -> list:
        """Determine persons to receive MDA, based on a specified target age-group and coverage."""

        assert 0.0 <= coverage <= 1.0, f'Value of coverage {coverage} is out of bounds.'
        assert age_group in ('PSAC', 'SAC', 'Adults')

        df = self.sim.population.props
        rng = self.module.rng

        age_range = map_age_groups(age_group)  # returns a tuple (a,b) a <= age_group <= b

        eligible = df.index[
            df['is_alive']
            & (df['district_of_residence'] == district)
            & df['age_years'].between(age_range[0], age_range[1])
        ]

        if len(eligible):
            return eligible.index[rng.random_sample(len(eligible)) < coverage].to_list()
        else:
            # todo - need this clause?
            return []


# class SchistoPrognosedMDAEvent(RegularEvent, PopulationScopeEventMixin):
#     """Mass-Drug administration scheduled for the population
#     Using the proposed MDA coverage
#     """
#
#     def __init__(self, module, freq, district):
#         super().__init__(module, frequency=DateOffset(months=freq))
#         self.district = district
#         assert isinstance(module, Schisto)
#
#     def apply(self, population):
#         print("Prognosed MDA is happening now!")
#         district = self.district
#
#         treated_idx_PSAC = self.assign_prognosed_MDA_coverage(population, district, 'PSAC')
#         treated_idx_SAC = self.assign_prognosed_MDA_coverage(population, district, 'SAC')
#         treated_idx_Adults = self.assign_prognosed_MDA_coverage(population, district, 'Adults')
#
#         treated_idx = treated_idx_PSAC + treated_idx_SAC + treated_idx_Adults
#         # all treated people will have worm burden decreased, and we already have chosen only alive people
#         for person_id in treated_idx:
#             self.sim.schedule_hsi_event(HSI_Schisto_MDA(self.module, person_id), self.sim.date)
#
#     def assign_prognosed_MDA_coverage(self, population, district, age_group):
#         """Assign coverage of MDA program to chosen age_group. The same coverage for every district.
#
#           :param district: district for which the MDA coverage is required
#           :param population: population
#           :param age_group: 'SAC', 'PSAC', 'Adults'
#           :returns MDA_idx: indices of people that will be administered PZQ in the MDA program
#           """
#
#         df = population.props
#         params = self.module.parameters
#         age_range = map_age_groups(age_group)  # returns a tuple (a,b) a <= age_group <= b
#         param_str = 'MDA_prognosed_' + age_group
#
#         coverage = params[param_str]
#         coverage_distr = coverage[district]
#
#         eligible = df.index[(df.is_alive) & (df['district_of_residence'] == district)
#                             & (df['age_years'].between(age_range[0], age_range[1]))]
#         MDA_idx = []
#         if len(eligible):
#             MDA_idx = self.module.rng.choice(eligible,
#                                              size=int(coverage_distr * (len(eligible))),
#                                              replace=False)
#
#         return MDA_idx



class SchistoLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """This is a regular event (every month) that causes the logging for each species."""
        super().__init__(module, frequency=DateOffset(months=1))
        assert isinstance(module, Schisto)

    def apply(self, population):
        """Write to log, for each species."""
        for _spec in self.module.species.values():
            _spec.log_counts_by_status_and_age_group()


class SchistoLoggingTotalEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """This logger logs the prevalence of ANY type of schistosomiasis
        If both schisto modules are registered it will count people with s.haematobium OR s.mansoni
        If only one of the schisto modules registered, it will work as the normal logger, i.e.
        count people with the specific infection type
        """
        # run this event every month
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, Schisto)

    def write_to_log(self, population, age_group):
        count_states = self.count_age_group_states(population, age_group)
        logger.info(key=f'{age_group}_Total',
                    data={
                        'Prevalence': count_states['Prevalence'],
                        'High_infections_Prevalence': count_states['High-inf_Prevalence'],
                    })

    def count_age_group_states(self, population, age_group):
        """
        :param population:
        :param age_group:
        :return: count_states: a dictionary of counts of individuals in age_group in different states on infection
        """
        df = population.props
        age_range = map_age_groups(age_group)  # returns a tuple

        where = df.is_alive & (df.age_years.between(age_range[0], age_range[1]))

        inf = self.count_status(population, where, ['Low-infection', 'High-infection'])
        high_inf = self.count_status(population, where, ['High-infection'])
        total_pop_size = sum(where)

        count_states = {}
        count_states.update({'Prevalence': inf / total_pop_size})
        count_states.update({'High-inf_Prevalence': high_inf / total_pop_size})

        return count_states

    def count_status(self, population, mask, status):
        module_prefix = self.module.module_prefix
        prefixes_for_each_species = [f"{module_prefix}_{_spec.prefix}" for _spec in self.module.species.values()]

        df = population.props
        count = sum(
            mask &
            df[[f'{_prefix}_infection_status' for _prefix in prefixes_for_each_species]].isin(status).any(axis=1)
        )

        return count

    def apply(self, population):
        for age in ['PSAC', 'SAC', 'Adults', 'All']:
            self.write_to_log(population, age)



# def count_days_this_year(date_end, date_start):
#     """Used for calculating PrevalentYears this year (that is the year of date_end)
#     If the start_date is in the previous years only gives the number of days from
#      the beginning of the year till date_end, if it start_date is the same year then just gives the time elapsed"""
#     year = date_end.year
#     if date_start.year < year:
#         date_start = pd.Timestamp(year=year, month=1, day=1)
#     duration = date_end - date_start
#     duration = int(duration / np.timedelta64(1, 'D'))
#     assert duration >= 0, "Duration is negative!"
#     return duration


# def add_elements(el1, el2):
#     """Helper function for multiple symptoms assignments
#
#     :param el1: np.nan or a list
#     :param el2: list containing a single string with a symptom
#     :return: either a sum of two lists or the list el2
#     """
#     if isinstance(el1, list):
#         return el1 + el2
#     return el2


# class SchistoParamFittingLogging(Event, PopulationScopeEventMixin):
#     """This Logging event should only be used for creating a lookup table for parameter fitting
#     Otherwise should not be scheduled"""
#     def __init__(self, module):
#         super().__init__(module)
#         assert isinstance(module, Schisto_Haematobium) or isinstance(module, Schisto_Mansoni)
#
#     def create_logger(self, population):
#
#         if isinstance(self.module, Schisto_Haematobium):
#             inf_type = 'Haematobium'
#             prefix = 'sh'
#         else:
#             inf_type = 'Mansoni'
#             prefix = 'sm'
#
#         df = population.props
#         df_alive = df[df.is_alive].copy()
#         all_infected = len(df_alive.index[df_alive[f'{prefix}_aggregate_worm_burden'] > 1])
#         total_pop = len(df_alive.index)
#         prevalence = all_infected / total_pop
#         mwb = df_alive[f'{prefix}_aggregate_worm_burden'].mean()
#
#         logger.info(key=f"{inf_type}",
#                     data={
#                         'Prevalence': prevalence,
#                         'MWB': mwb,
#                         'alpha': self.module.alpha,
#                         'R0': self.module.r0
#                     })
#
#     def apply(self, population):
#         self.create_logger(population)
#
#
# class SchistoPrevalentDaysLoggingEvent(RegularEvent, PopulationScopeEventMixin):
#     def __init__(self, module):
#         """Produce a summary of the prevalent days & high infection days per year
#         It is best to use it only with one type of infection, as the logger does not contain info on the infection type
#         """
#         # run this event every year
#         super().__init__(module, frequency=DateOffset(months=12))
#         assert isinstance(module, Schisto)
#
#     def create_logger(self, population):
#         df = population.props
#         # adding unfinished infections
#         prefixes = []
#         if 'Schisto_Haematobium' in self.sim.modules:
#             prefixes.append('sh')
#         if 'Schisto_Mansoni' in self.sim.modules:
#             prefixes.append('sm')
#
#         for prefix in prefixes:
#             # still infected
#             condition, duration = self.days_in_year(df, f'{prefix}_start_of_prevalent_period', self.sim.date)
#             df.loc[condition, f'{prefix}_prevalent_days_this_year'] += duration
#
#             # still high infected
#             condition, duration = self.days_in_year(df, f'{prefix}_start_of_high_infection', self.sim.date)
#             df.loc[condition, f'{prefix}_high_inf_days_this_year'] += duration
#
#             for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
#                 count_states = self.count_prev_years_age(prefix, df, age_group)
#                 logger.info(key=f"{age_group}_PrevalentYears",
#                             data={
#                                 'Prevalent_years_this_year_total': count_states['Prevalent_years_this_year_total'],
#                                 'Prevalent_years_per_100': count_states['Prevalent_years_per_100'],
#                                 'High_infection_years_this_year_total':
#                                     count_states['High_infection_years_this_year_total'],
#                                 'High_infection_years_per_100': count_states['High_infection_years_per_100'],
#                                 'Total_pop_alive': count_states['Tot_pop_alive']
#                             })
#
#             # clear so that it's ready for next year
#             df.loc[df.is_alive, f'{prefix}_prevalent_days_this_year'] = 0
#             df.loc[df.is_alive, f'{prefix}_high_inf_days_this_year'] = 0
#
#     def days_in_year(self, df, dt_column, end_date):
#         condition = df.is_alive & (~df[dt_column].isna())
#         duration_start = df.loc[condition, dt_column].where(df[dt_column].dt.year == end_date.year,
#                                                             Date(end_date.year, 1, 1))
#         duration = end_date - duration_start
#         return condition, duration.dt.days
#
#     def count_prev_years_age(self, prefix, df, age):
#         # sum up for each age_group
#         age_range = map_age_groups(age)  # returns a tuple
#         cond = df.is_alive & df['age_years'].between(age_range[0], age_range[1])
#         Tot_pop_alive = sum(cond)
#         Prevalent_years_this_year_total = df.loc[cond, f'{prefix}_prevalent_days_this_year'].values.sum() / 365
#         High_infection_years_this_year_total = df.loc[cond, f'{prefix}_high_inf_days_this_year'].values.sum() / 365
#         Prevalent_years_per_100 = Prevalent_years_this_year_total * 100 / Tot_pop_alive
#         High_infection_years_per_100 = High_infection_years_this_year_total * 100 / Tot_pop_alive
#         count_states = {'Prevalent_years_this_year_total': Prevalent_years_this_year_total,
#                         'Prevalent_years_per_100': Prevalent_years_per_100,
#                         'High_infection_years_this_year_total': High_infection_years_this_year_total,
#                         'High_infection_years_per_100': High_infection_years_per_100,
#                         'Tot_pop_alive': Tot_pop_alive
#                         }
#         return count_states
#
#     def apply(self, population):
#         self.create_logger(population)
#
#
# class SchistoLoggingPrevDistrictEvent(Event, PopulationScopeEventMixin):
#     """Produces a log of prevalence and MWB in every district used for validating the parameters fit"""
#     def __init__(self, module):
#         super().__init__(module)
#         assert isinstance(module, Schisto)
#
#     def create_logger(self, population, district):
#         if isinstance(self.module, Schisto_Haematobium):
#             inf_type = 'Haematobium'
#         else:
#             inf_type = 'Mansoni'
#
#         count_states = self.count_district_states(population, district)
#         logger.info(key=f"{district}_{inf_type}",
#                     data={
#                         'Non_infected': count_states['Non-infected'],
#                         'Low_infections': count_states['Low-infection'],
#                         'High_infections': count_states['High-infection'],
#                         'Infected': count_states['infected_any'],
#                         'Prevalence': count_states['Prevalence'],
#                         'MeanWormBurden': count_states['MeanWormBurden']
#                     })
#
#     def count_district_states(self, population, district):
#         """
#         :param population:
#         :param district:
#         :return: count_states: a dictionary of counts of individuals in district in different states on infection
#         """
#         df = population.props
#
#         where = df.is_alive & (df.district_of_residence == district)
#
#         counts = {'Non-infected': 0, 'Low-infection': 0, 'High-infection': 0}
#         counts.update(df.loc[where, f'{self.module.prefix}_infection_status'].value_counts().to_dict())
#
#         counts['infected_any'] = counts['Low-infection'] + counts['High-infection']
#         counts['total_pop_alive'] = counts['infected_any'] + counts['Non-infected']
#
#         if counts['total_pop_alive'] > 0:
#             counts['Prevalence'] = counts['infected_any'] / counts['total_pop_alive']
#         else:
#             counts['Prevalence'] = 0.0
#
#         counts['MeanWormBurden'] = df.loc[where, f'{self.module.prefix}_aggregate_worm_burden'].mean()
#         return counts
#
#     def apply(self, population):
#         districts = self.module.districts
#         for distr in districts:
#             self.create_logger(population, distr)


# # TODO: Should this be a function or an HSI?
# class SchistoTreatmentEvent(Event, IndividualScopeEventMixin):
#     """Cured upon PZQ treatment through HSI or MDA (Infected -> Non-infected)
#     PZQ treats both types of infections, so affect symptoms and worm burden of any infection type registered
#     """
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#         assert isinstance(module, Schisto)
#
#     def apply(self, person_id):
#         df = self.sim.population.props
#
#         prefixes = []
#         if 'Schisto_Haematobium' in self.sim.modules.keys():
#             prefixes.append('sh')
#         if 'Schisto_Mansoni' in self.sim.modules.keys():
#             prefixes.append('sm')
#
#         if not df.loc[person_id, 'is_alive']:
#             return
#
#         for prefix in prefixes:
#             if df.loc[person_id, f'{prefix}_infection_status'] != 'Non-infected':
#
#                 # check if they experienced symptoms, and if yes, treat them
#                 df.loc[person_id, f'{prefix}_symptoms'] = np.nan
#                 # if isinstance(df.loc[person_id, prefix + '_symptoms'], list):
#                 #     df.loc[person_id, prefix + '_symptoms'] = np.nan
#
#                 # calculate the duration of the prevalent period
#                 prevalent_duration = count_days_this_year(self.sim.date, df.loc[
#                     person_id, f'{prefix}_start_of_prevalent_period'])
#                 df.loc[person_id, f'{prefix}_prevalent_days_this_year'] += prevalent_duration
#                 df.loc[person_id, f'{prefix}_start_of_prevalent_period'] = pd.NaT
#
#                 # calculate the duration of the high-intensity infection
#                 if df.loc[person_id, f'{prefix}_infection_status'] == 'High-infection':
#                     high_infection_duration = count_days_this_year(self.sim.date, df.loc[
#                         person_id, f'{prefix}_start_of_high_infection'])
#                     df.loc[person_id, f'{prefix}_high_inf_days_this_year'] += high_infection_duration
#                     df.loc[person_id, f'{prefix}_start_of_high_infection'] = pd.NaT
#
#                 df.loc[person_id, f'{prefix}_aggregate_worm_burden'] = 0  # PZQ_efficacy = 100% for now
#                 df.loc[person_id, f'{prefix}_start_of_prevalent_period'] = pd.NaT
#                 df.loc[person_id, f'{prefix}_start_of_high_infection'] = pd.NaT
#                 df.loc[person_id, f'{prefix}_infection_status'] = 'Non-infected'
#
#         # the general Schisto module properties
#         df.loc[person_id, 'ss_scheduled_hsi_date'] = pd.NaT
#         df.loc[person_id, 'ss_last_PZQ_date'] = self.sim.date
#
#
# #TODO: Consider if this should be the generic HSI
# class HSI_SchistoSeekTreatment(HSI_Event, IndividualScopeEventMixin):
#     """This is a Health System Interaction Event of seeking treatment for a person with symptoms"""
#     # todo should this be handled with generic appointments?
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#         assert isinstance(module, Schisto)
#
#         under_5 = self.sim.population.props.at[person_id, 'age_years'] <= 5
#         self.TREATMENT_ID = 'Schisto_Treatment_seeking'
#         self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Under5OPD' if under_5 else 'Over5OPD': 1})
#         self.ACCEPTED_FACILITY_LEVEL = '1a'
#         self.ALERT_OTHER_DISEASES = []
#
#     def apply(self, person_id, squeeze_factor):
#         df = self.sim.population.props
#         params = self.module.parameters
#
#         prefixes = []
#         if 'Schisto_Haematobium' in self.sim.modules.keys():
#             prefixes.append('sh')
#         if 'Schisto_Mansoni' in self.sim.modules.keys():
#             prefixes.append('sm')
#         is_infected = False
#         for pref in prefixes:
#             if df.loc[person_id, f'{pref}_infection_status'] != 'Non-infected':
#                 is_infected = True
#
#         # appt are scheduled and cannot be cancelled in the following situations:
#         #   a) person has died
#         #   b) the infection has been treated in MDA or by treating symptoms from
#         #   other schisto infection before the appt happened
#         if (df.loc[person_id, 'is_alive'] & is_infected):  # &
#             # (df.loc[person_id, 'ss_scheduled_hsi_date'] <= self.sim.date)):
#             # check if a person is a child or an adult and assign prob of being sent to schisto test (hence being cured)
#             if df.loc[person_id, 'age_years'] <= 15:
#                 prob_test = params['prob_sent_to_lab_test_children']
#             else:
#                 prob_test = params['prob_sent_to_lab_test_adults']
#
#             sent_to_test = self.module.rng.rand() < prob_test
#             # sent_to_test = self.module.rng.choice([True, False], p=[prob_test, 1-prob_test])
#             # use this is you don't care about whether PZQ is available or not
#             # if sent_to_test:
#             #     self.sim.schedule_event(SchistoTreatmentEvent(self.module, person_id), self.sim.date)
#             if sent_to_test:
#                 # request the consumable
#                 consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
#                 items_code1 = \
#                     pd.unique(
#                         consumables.loc[
#                             consumables['Items'] == "Praziquantel, 600 mg (donated)", 'Item_Code'])[0]
#                 the_cons_footprint = {'Intervention_Package_Code': {}, 'Item_Code': {items_code1: 1}}
#                 outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
#                     hsi_event=self, cons_req_as_footprint=the_cons_footprint, to_log=False
#                 )
#
#                 # give the PZQ to the patient
#                 if outcome_of_request_for_consumables['Item_Code'][items_code1]:
#                     self.sim.modules['HealthSystem'].request_consumables(
#                         hsi_event=self, cons_req_as_footprint=the_cons_footprint, to_log=True
#                     )
#                     # patient is cured
#                     self.sim.schedule_event(SchistoTreatmentEvent(self.module, person_id), self.sim.date)
#
#             else:  # person seeked treatment but was not sent to test; visit is reschedulled
#                 # schedule another Seeking Treatment event for that person
#                 seeking_treatment_ahead_repeated = \
#                     int(self.module.rng.uniform(params['delay_till_hsi_a_repeated'],
#                                                 params['delay_till_hsi_b_repeated']))
#                 seeking_treatment_ahead_repeated = pd.to_timedelta(seeking_treatment_ahead_repeated, unit='D')
#                 df.loc[person_id, 'ss_scheduled_hsi_date'] = self.sim.date + seeking_treatment_ahead_repeated
#
#                 seek_treatment_repeated = HSI_SchistoSeekTreatment(self.module, person_id)
#                 self.sim.modules['HealthSystem'].schedule_hsi_event(seek_treatment_repeated,
#                                                                     priority=1,
#                                                                     topen=df.loc[person_id, 'ss_scheduled_hsi_date'],
#                                                                     tclose=df.loc[person_id, 'ss_scheduled_hsi_date']
#                                                                     + DateOffset(weeks=500))
#
#     def did_not_run(self):
#         return True
#
#
#
