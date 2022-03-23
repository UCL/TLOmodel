from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from tlo import Date, DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------------------------------------
#   Utility Functions
# ---------------------------------------------------------------------------------------------------------

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
    assert age_group in ['SAC', 'PSAC', 'Adults', 'All'], "Incorrect age group"
    age_map = {'PSAC': (0, 4), 'SAC': (5, 14), 'Adults': (15, 120), 'All': (0, 120)}
    return age_map[age_group]


def count_days_this_year(date_end, date_start):
    """Used for calculating PrevalentYears this year (that is the year of date_end)
    If the start_date is in the previous years only gives the number of days from
     the beginning of the year till date_end, if it start_date is the same year then just gives the time elapsed"""
    year = date_end.year
    if date_start.year < year:
        date_start = pd.Timestamp(year=year, month=1, day=1)
    duration = date_end - date_start
    duration = int(duration / np.timedelta64(1, 'D'))
    assert duration >= 0, "Duration is negative!"
    return duration


def add_elements(el1, el2):
    """Helper function for multiple symptoms assignments

    :param el1: np.nan or a list
    :param el2: list containing a single string with a symptom
    :return: either a sum of two lists or the list el2
    """
    if isinstance(el1, list):
        return el1 + el2
    return el2


# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITION
# ---------------------------------------------------------------------------------------------------------

class Schisto(Module):
    """
    Schistosomiasis module
    Required for the Schisto_Haematobium and Schisto_Mansoni modules.
    Governs the properties that are common to both of the Schisto modules.
    Also schedules HSI events, Treatment and MDA event.
    attribute mda_execute: TRUE (default) /FALSE determines whether MDA events should be scheduled or not
    """

    INIT_DEPENDENCIES = {'Demography', 'HealthSystem', 'SymptomManager'}

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
        f'{module_prefix}_scheduled_hsi_date': Property(Types.DATE, 'Date of scheduled seeking healthcare'),   # todo - needed?
        f'{module_prefix}_last_PZQ_date': Property(Types.DATE, 'Day of the most recent treatment with PZQ')    # todo - needed?
    }

    PARAMETERS = {
        # NOT USED PARAMETERS:
        # health system interaction
        # 'delay_till_hsi_a': Parameter(Types.REAL, 'Time till seeking healthcare since the onset of symptoms, start'),
        # 'delay_till_hsi_b': Parameter(Types.REAL, 'Time till seeking healthcare since the onset of symptoms, end'),
        #
        #
        # 'prob_seeking_healthcare': Parameter(Types.REAL,
        #                                      'Probability that an infected individual visits a healthcare facility'),

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


        # # symptoms prevalence
        # 'symptoms_prevalence': Parameter(Types.DICT, 'Prevalence of a symptom occurring given an infection'),
        # 'symptoms_mapping': Parameter(Types.DICT, 'Schisto symptom mapped to HSB symptom'),

        # MDA parameters
        'years_till_first_MDA': Parameter(Types.REAL, 'Years till the first historical MDA'),

        'MDA_coverage_PSAC': Parameter(Types.DATA_FRAME, 'Probability of getting PZQ in the MDA for PSAC'),
        'MDA_coverage_SAC': Parameter(Types.DATA_FRAME, 'Probability of getting PZQ in the MDA for SAC'),
        'MDA_coverage_Adults': Parameter(Types.DATA_FRAME, 'Probability of getting PZQ in the MDA for Adults'),

        'MDA_prognosed_freq': Parameter(Types.DATA_FRAME, 'Prognosed MDA frequency in months'),
        'MDA_prognosed_PSAC': Parameter(Types.DATA_FRAME, 'Prognosed coverage of MDA in PSAC'),
        'MDA_prognosed_SAC': Parameter(Types.DATA_FRAME, 'Prognosed coverage of MDA in SAC'),
        'MDA_prognosed_Adults': Parameter(Types.DATA_FRAME, 'Prognosed coverage of MDA in Adults'),
    }

    def __init__(self, name=None, resourcefilepath=None, mda_execute=True):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.mda_execute = mda_execute

        # self.districts = sim.modules['Demography'].districts  # if we want it to be all districts:
        # Iwona used in her thesis:
        self.districts = ['Blantyre', 'Chiradzulu', 'Mulanje', 'Nsanje', 'Nkhotakota', 'Phalombe']

        # Create the instances of `SchistoSpecies` that will represent the two species being considered
        self.species = {_name: SchistoSpecies(self, name=_name) for _name in ['mansoni', 'haematobium']}

        # Add properties and parameters declared by each species:
        for _spec in self.species.values():
            self.PROPERTIES.update(_spec.PROPERTIES)
            self.PARAMETERS.update(_spec.PARAMETERS)

        # Create pointer that will be to dict of disability weights
        self.disability_weights = None

    def read_parameters(self, data_folder):
        """Read parameters and load into `self.parameters` dictionary."""

        workbook = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Schisto.xlsx', sheet_name=None)

        self.parameters = self._load_parameters_from_workbook(workbook)
        for _spec in self.species.values():
            self.parameters.update(_spec.load_parameters_from_workbook(workbook))

        # Register symptoms
        symptoms_df = workbook['Symptoms']
        self._register_symptoms(symptoms_df.set_index('Symptom')['HSB_mapped_symptom'].to_dict())

    def initialise_population(self, population):
        """Set the property values for the initial population."""
        df = population.props
        df.loc[df.is_alive, 'ss_scheduled_hsi_date'] = pd.NaT
        df.loc[df.is_alive, 'ss_last_PZQ_date'] = pd.Timestamp(year=1900, month=1, day=1)  # for simplicity to avoid NaT

        for _spec in self.species.values():
            _spec.initialise_population(population)

    def initialise_simulation(self, sim):
        """Get ready for simulation start."""

        for _spec in self.species.values():
            _spec.initialise_simulation(sim)

        # DALY weights
        if 'HealthBurden' in self.sim.modules:
            self.disability_weights = self._get_disability_weight()

        if self.mda_execute:
            # schedule historical MDA to happen once per year in July (4 events)
            y0 = self.parameters['years_till_first_MDA']
            years_of_historical_mda = [int(y0 + ii) for ii in range(4)]  # 4 consecutive years
            years_of_historical_mda = [sim.date.year + el for el in years_of_historical_mda]  # create dates
            for historical_mda_year in years_of_historical_mda:
                # todo - to be an HSI event
                sim.schedule_event(SchistoHistoricalMDAEvent(self),
                                   pd.Timestamp(year=historical_mda_year, month=7, day=1, hour=23))
            year_first_simulated_mda = years_of_historical_mda[-1] + 1

            # schedule prognosed MDA programmes for every district
            # todo - be an HSI event
            for district in sim.population.props.district_of_residence.unique():
                freq = self.parameters['MDA_frequency_prognosed'][district]
                if freq > 0:  # frequency 0 means no need for MDA, because prevalence there is always 0
                    sim.schedule_event(SchistoPrognosedMDAEvent(self, freq, district),
                                       pd.Timestamp(year=year_first_simulated_mda, month=6, day=1, hour=12))


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
        df.at[child_id, 'ss_scheduled_hsi_date'] = pd.NaT
        df.at[child_id, 'ss_last_PZQ_date'] = pd.Timestamp(year=1900, month=1, day=1)

        for _spec in self.species.values():
            _spec.on_birth(mother_id, child_id)

    def report_daly_values(self):
        """Report the daly values, as the sum of the disability weight associated with each symptom caused by this
        module."""
        # Get the total weights for all those that have symptoms caused by this module.
        symptoms_being_caused = self.sim.modules['SymptomManager'].caused_by(self)
        dw = pd.Series(symptoms_being_caused).apply(pd.Series).replace(self.disability_weights).fillna(0).sum(axis=1).clip(upper=1.0)

        # Return series that include entries for all alive persons (filling 0.0) where they do not have any disability
        df = self.sim.population.props
        return pd.Series(index=df.index[df.is_alive], data=0.0).add(dw, fill_value=0.0)

    def _load_parameters_from_workbook(self, workbook) -> dict:
        """Load parameters from workboook that are general."""
        parameters = dict()

        # HSI and treatment params
        param_list = workbook['Parameters'].set_index("Parameter")
        # parameters['delay_till_hsi_a'] = param_list.loc['delay_till_hsi_a', 'Value']    todo- remove!?!
        # parameters['delay_till_hsi_b'] = param_list.loc['delay_till_hsi_b', 'Value']
        # parameters['prob_seeking_healthcare'] = param_list.loc['prob_seeking_healthcare', 'Value']

        # todo - use helper function!?

        parameters['delay_till_hsi_a_repeated'] = param_list.loc['delay_till_hsi_a_repeated', 'Value']
        parameters['delay_till_hsi_b_repeated'] = param_list.loc['delay_till_hsi_b_repeated', 'Value']
        parameters['prob_sent_to_lab_test_children'] = param_list.loc['prob_sent_to_lab_test_children', 'Value']
        parameters['prob_sent_to_lab_test_adults'] = param_list.loc['prob_sent_to_lab_test_adults', 'Value']
        parameters['PZQ_efficacy'] = param_list.loc['PZQ_efficacy', 'Value']

        # todo - need to load beta_PSAC / beta_SAC / beta_Adults are are these not used?

        # MDA coverage historical
        parameters['years_till_first_MDA'] = param_list.loc['years_till_first_MDA', 'Value']

        mda_historical_coverage = workbook['MDA_historical_Coverage'].set_index(['District', 'Year'])
        parameters['MDA_coverage_PSAC'] = mda_historical_coverage.loc[:, 'Coverage PSAC']
        parameters['MDA_coverage_SAC'] = mda_historical_coverage.loc[:, 'Coverage SAC']
        parameters['MDA_coverage_Adults'] = mda_historical_coverage.loc[:, 'Coverage Adults']

        # MDA coverage prognosed
        mda_prognosed_coverage = workbook['MDA_prognosed_Coverage'].set_index(['District'])
        parameters['MDA_frequency_prognosed'] = mda_prognosed_coverage.loc[:, 'Frequency']
        parameters['MDA_prognosed_PSAC'] = mda_prognosed_coverage.loc[:, 'Coverage PSAC']
        parameters['MDA_prognosed_SAC'] = mda_prognosed_coverage.loc[:, 'Coverage SAC']
        parameters['MDA_prognosed_Adults'] = mda_prognosed_coverage.loc[:, 'Coverage Adults']

        return parameters

    def _register_symptoms(self, symptoms: dict):
        """Symptoms that are used by this module in a dictionary of the form, {<symptom>: <generic_symptom_similar>}"""
        # todo currently ignore that these should have the HSB of a particular generic symptom, for now - all generic
        # todo check names are not close to but not equal to generic names for symptpoms
        generic_symptoms = self.sim.modules['SymptomManager'].generic_symptoms
        self.sim.modules['SymptomManager'].register_symptom(*[
            Symptom(name=_symp) for _symp in symptoms if _symp not in generic_symptoms
        ])

    def _get_disability_weight(self):
        """Return dict containing the disability weight (value) of each symptom (key)."""
        symptoms_to_disability_weight_mapping = {
            # These mapping are justified in the 'DALYS' worksheet of the ResourceFile.
            'anemia': 258,
            'fever': 262,
            'hydronephrosis': 260,
            'dysuria': 263,
            'bladder_pathology': 264,
            'diarrhoea': 259,
            'vomit': 254,
            'ascites': 261,
            'hepatomegaly': 257,
            'haematuria': None  # That's a very common symptom but no official DALY weight yet defined.
        }
        get_daly_weight = lambda _code: self.sim.modules['HealthBurden'].get_daly_weight(_code) if _code is not None else 0.0

        return {
            symptom: get_daly_weight(dw_code) for symptom, dw_code in symptoms_to_disability_weight_mapping.items()
        }


class SchistoSpecies:
    """Helper Class to hold the information specific to either S. mansoni or S. haematobium."""

    def __init__(self, schisto_module, name):
        self.schisto_module = schisto_module
        assert name in ('mansoni', 'haematobium')
        self.name = name

        # Store prefix
        self.prefix = 's' + self.name[0]

        # Store params (general and those specific to this species)
        self.params = dict()

    @property
    def PARAMETERS(self):
        return {self._species_specific_parameter(k): v for k, v in self._define_parameters.items()}

    @property
    def PROPERTIES(self):
        return {self._species_specific_property(k): v for k, v in self._define_properties.items()}

    @property
    def _prefix_on_property(self):
        return f"{self.schisto_module.module_prefix}_{self.prefix}"

    @property
    def _prefix_on_parameters(self):
        return f"{self.prefix}"

    def _species_specific_property(self, generic_property_name):
        """Append the prefix to a `generic_property_name` to get the name of the Property that is defined by the
        module."""
        return f"{self._prefix_on_property}_{generic_property_name}"

    def _species_specific_parameter(self, generic_parameter_name):
        """Append the prefix to a `generic_property_name` to get the name of the Property that is defined by the
        module."""
        return f"{self._prefix_on_parameters}_{generic_parameter_name}"

    def load_parameters_from_workbook(self, workbook):
        parameters = dict()

        # natural history params
        param_list = workbook['Parameters'].set_index("Parameter")
        # parameters['delay_a'] = param_list.loc['delay_a', 'Value']
        # parameters['delay_b'] = param_list.loc['delay_b', 'Value']
        parameters['beta_PSAC'] = param_list.loc['beta_PSAC', 'Value']  # todo don't load - not specific --- load in module
        parameters['beta_SAC'] = param_list.loc['beta_SAC', 'Value']  # todo don't load- not specific
        parameters['beta_Adults'] = param_list.loc['beta_Adults', 'Value']  # todo don't load- not specific

        parameters['worms_fecundity'] = param_list.loc[f'worms_fecundity_{self.name.lower()}', 'Value']
        parameters['worm_lifespan'] = param_list.loc[f'lifespan_{self.name.lower()}', 'Value']
        parameters['high_intensity_threshold'] = param_list.loc[f'high_intensity_threshold_{self.name.lower()}', 'Value']
        parameters['low_intensity_threshold'] = param_list.loc[f'low_intensity_threshold_{self.name.lower()}', 'Value']
        parameters['high_intensity_threshold_PSAC'] = param_list.loc[f'high_intensity_threshold_{self.name.lower()}_PSAC', 'Value']

        # baseline reservoir size and other district-related params (alpha and R0)
        schisto_initial_reservoir = workbook[f'District_Params_{self.name.lower()}'].set_index("District")
        parameters['reservoir_2010'] = schisto_initial_reservoir.loc[:, 'Reservoir']
        parameters['gamma_alpha'] = schisto_initial_reservoir.loc[:, 'alpha_value']
        parameters['R0'] = schisto_initial_reservoir.loc[:, 'R0_value']

        # symptoms (prevalence of each type of symptom)
        symptoms_df = workbook['Symptoms']
        parameters['symptoms'] = symptoms_df.loc[symptoms_df['Infection_type'].isin(['both', self.name])].set_index('Symptom')['Prevalence'].to_dict()

        return {self._species_specific_parameter(k): v for k, v in parameters.items()}

    def initialise_population(self, population):
        """Set our property values for the initial population, using a top-level function"""
        self._update_parameters_from_module()

        df = population.props
        date = self.schisto_module.sim.date
        prop = self._species_specific_property

        df.loc[df.is_alive, prop('aggregate_worm_burden')] = 0
        df.loc[df.is_alive, prop('prevalent_days_this_year')] = 0
        df.loc[df.is_alive, prop('start_of_prevalent_period')] = pd.NaT
        df.loc[df.is_alive, prop('start_of_high_infection')] = pd.NaT
        df.loc[df.is_alive, prop('high_inf_days_this_year')] = 0

        # assign a harbouring rate
        self._assign_harbouring_rate(population)

        # assign initial worm burden
        self._assign_initial_worm_burden(population)

        # assign infection statuses
        df.loc[df.is_alive, prop('infection_status')] = df.loc[df.is_alive].apply(
            lambda x: self._intensity_of_infection(x['age_years'], x[prop('aggregate_worm_burden')]),
            axis=1
        )
        #  start of the prevalent period & start of high-intensity infections
        df.loc[df.is_alive & (df[prop('infection_status')] != 'Non-infected'), prop('start_of_prevalent_period')] = date
        high_infected_idx = df.index[df[prop('infection_status')] == 'High-infection']
        df.loc[high_infected_idx, prop('start_of_high_infection')] = date

        # assign initial symptoms
        self._impose_symptoms(high_infected_idx)

        # TIM: No healthcare seeking is needed and this is handled now automatically through the SymptomManager --> GenericHSI
        # # assign initial dates of seeking healthcare for people with symptoms
        # # todo ---- symptons and HSI stufff!?!?!?!
        # # todo - as the symptoms have beem imposed, Generic Healthcare Seeking will happen automatically......
        # symptomatic_idx = df.index[~df[prop('symptoms')].isna()]
        # self._assign_hsi_dates_initial(population, symptomatic_idx)

    def _update_parameters_from_module(self):
        """Update the internally-held parameters from that Module that are specific to this species.
        (NB. We have to do this step because the module may have update the parameters.)"""
        def remove_prefix(text, prefix):
            return text[text.startswith(prefix) and len(prefix):]

        self.params = {
            remove_prefix(k, f"{self.prefix}_"): v for k, v in self.schisto_module.parameters.items()
            if k.startswith(self.prefix)
        }

    def initialise_simulation(self, sim):
        """Initialise the WormBurdenEvent, with a pointer this species"""
        sim.schedule_event(
            SchistoInfectionWormBurdenEvent(
                module=self.schisto_module,
                species=self
            ), sim.date + DateOffset(months=1)
        )

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual, using a top-level function

        :param mother_id: the ID for the mother for this child (redundant)
        :param child_id: the new child
        """
        """Initialise our properties for a newborn individual.
        All children are born without an infection, even if the mother is infected.

        :param mother_id: the ID for the mother for this child (redundant)
        :param child_id: the new child
        """

        df = self.schisto_module.sim.population.props
        prop = self._species_specific_property
        params = self.params
        rng = self.schisto_module.rng

        # Assign the default for a newly born child
        df.at[child_id, prop('infection_status')] = 'Non-infected'
        df.at[child_id, prop('aggregate_worm_burden')] = 0
        df.at[child_id, prop('prevalent_days_this_year')] = 0
        df.at[child_id, prop('start_of_prevalent_period')] = pd.NaT
        df.at[child_id, prop('start_of_high_infection')] = pd.NaT
        df.at[child_id, prop('high_inf_days_this_year')] = 0

        # generate the harbouring rate depending on a district of mother / person inheriting from
        # todo - allow this to cope with mother_id = -1
        district = df.loc[mother_id, 'district_of_residence']
        df.at[child_id, prop('harbouring_rate')] = rng.gamma(params['gamma_alpha'][district], size=1)

    @property
    def _define_parameters(self):
        return {
            'reservoir_2010': Parameter(Types.REAL, 'Initial reservoir of infectious material per district in 2010'),
            # 'delay_a': Parameter(Types.REAL, 'End of the latent period in days, start'),
            # 'delay_b': Parameter(Types.REAL, 'End of the latent period in days, end'),
            'symptoms': Parameter(Types.LIST, 'Symptoms of the schistosomiasis infection, dependent on the module'),
            'gamma_alpha': Parameter(Types.REAL, 'Parameter alpha for Gamma distribution for harbouring rates'),
            'beta_PSAC': Parameter(Types.REAL, 'Contact/exposure rate of PSAC'),
            'beta_SAC': Parameter(Types.REAL, 'Contact/exposure rate of SAC'),
            'beta_Adults': Parameter(Types.REAL, 'Contact/exposure rate of Adults'),
            'worms_fecundity': Parameter(Types.REAL, 'Fecundity parameter, driving density-dependent reproduction'),
            'R0': Parameter(Types.REAL, 'Effective reproduction number, for the FOI'),
            'worm_lifespan': Parameter(Types.REAL, 'Lifespan of the worm in human host given in years'),
            'high_intensity_threshold': Parameter(Types.REAL,
                                                  'Threshold of worm burden indicating high intensity infection'),
            'low_intensity_threshold': Parameter(Types.REAL, 'Threshold of worm burden indicating low intensity infection'),
            'high_intensity_threshold_PSAC': Parameter(Types.REAL,
                                                       'Worm burden threshold for high intensity infection in PSAC'),
        }

    @property
    def _define_properties(self):
        return {
            'infection_status': Property(
                Types.CATEGORICAL, 'Current status of schistosomiasis infection',
                categories=['Non-infected', 'Low-infection', 'High-infection']),
            'aggregate_worm_burden': Property(
                Types.INT, 'Number of mature worms in the individual'),
            'start_of_prevalent_period': Property(Types.DATE, 'Date of going from Non-infected to Infected'),
            'start_of_high_infection': Property(Types.DATE, 'Date of going from entering state High-inf'),
            'harbouring_rate': Property(Types.REAL,
                                                  'Rate of harbouring new worms (Poisson), drawn from gamma distribution'),
            'prevalent_days_this_year': Property(Types.INT, 'Cumulative days with infection in current year'),
            'high_inf_days_this_year': Property(Types.INT,
                                                          'Cumulative days with high-intensity infection in current year')
        }

    def _assign_harbouring_rate(self, population):
        """Assign a harbouring rate to every individual, this happens with a district-related param"""
        df = population.props
        prop = self._species_specific_property
        params = self.params
        districts = self.schisto_module.districts
        rng = self.schisto_module.rng

        for district in districts:
            eligible = df.index[df['district_of_residence'] == district]
            hr = params['gamma_alpha'][district]
            df.loc[eligible, prop('harbouring_rate')] = rng.gamma(hr, size=len(eligible))

    def _assign_initial_worm_burden(self, population):
        """Assign initial 2010 prevalence of schistosomiasis infections
        This will depend on a district and age group.
        """
        df = population.props
        prop = self._species_specific_property
        params = self.params
        districts = self.schisto_module.districts
        rng = self.schisto_module.rng

        reservoir = params['reservoir_2010']

        for distr in districts:
            eligible = df.index[df['district_of_residence'] == distr]
            contact_rates = pd.Series(1, index=eligible)
            for age_group in ['PSAC', 'SAC', 'Adults']:
                age_range = map_age_groups(age_group)
                in_the_age_group = \
                    df.index[(df['district_of_residence'] == distr) &
                             (df['age_years'].between(age_range[0], age_range[1]))]
                contact_rates.loc[in_the_age_group] *= params[f"beta_{age_group}"]  # Beta(age_group) and species

            if len(eligible):
                harbouring_rates = df.loc[eligible, prop('harbouring_rate')].values
                rates = np.multiply(harbouring_rates, contact_rates)
                reservoir_distr = int(reservoir[distr] * len(eligible))
                # distribute a worm burden
                chosen = rng.choice(eligible, reservoir_distr, p=rates / rates.sum())
                unique, counts = np.unique(chosen, return_counts=True)
                worms_per_idx = dict(zip(unique, counts))
                df[prop('aggregate_worm_burden')].update(pd.Series(worms_per_idx))

        # schedule death of worms
        people_with_worms = df.index[df[prop('aggregate_worm_burden')] > 0]
        for person_id in people_with_worms:
            worms = df.loc[person_id, prop('aggregate_worm_burden')]
            months_till_death = int(rng.uniform(1, params['worm_lifespan'] * 12 / 2))
            self.schisto_module.sim.schedule_event(
                SchistoWormsNatDeath(module=self.schisto_module,
                                     species=self,
                                     person_id=person_id,
                                     number_of_worms=worms),
                self.schisto_module.sim.date + DateOffset(months=months_till_death)
            )

    def _draw_worms(self, worms_total, rates, district):
        """ This function generates random number of new worms drawn from Poisson distribution multiplied by
        a product of harbouring rate and exposure rate
        :param district: district name
        :param rates: harbouring rates used for Poisson distribution, drawn from Gamma,
        multiplied by contact rate per age group
        :param worms_total: total size of the reservoir of infectious material
        :return harboured_worms: array of numbers of new worms for each of the persons (len = len(rates))
        """
        module = self.schisto_module
        params = module.parameters
        if worms_total == 0:
            return np.zeros(len(rates))
        rates = list(rates)
        R0 = params['R0'][district]
        worms_total *= R0
        harboured_worms = np.asarray([module.rng.poisson(x * worms_total, 1)[0] for x in rates]).astype(int)
        return harboured_worms

    def _intensity_of_infection(self, age, agg_wb):
        params = self.params

        if age < 5:
            if agg_wb >= params['high_intensity_threshold_PSAC']:
                return 'High-infection'
        if agg_wb >= params['high_intensity_threshold']:
            return 'High-infection'
        if agg_wb >= params['low_intensity_threshold']:
            return 'Low-infection'
        return 'Non-infected'

    def _impose_symptoms(self, eligible_idx: Union[list, pd.Index]):
        """Assign symptoms to the person with high intensity infection.
        :param eligible_idx: indices of individuals
        """
        if not len(eligible_idx):
            return

        if not isinstance(eligible_idx, pd.Index):
            eligible_idx = pd.Index(eligible_idx)

        module = self.schisto_module
        params = self.params
        possible_symptoms = params["symptoms"]
        rng = module.rng
        sm = self.schisto_module.sim.modules['SymptomManager']

        for symptom, prev in possible_symptoms.items():
            will_onset = eligible_idx[rng.random_sample(len(eligible_idx)) < prev]
            if not will_onset.empty:
                sm.change_symptom(
                    person_id=will_onset,
                    symptom_string=symptom,
                    add_or_remove='+',
                    disease_module=module
                )

        # def assign_symptoms(module_prefix):
        #     """
        #     Assign symptoms to the person with high intensity infection.
        #
        #     :param module_prefix: indicates type of infection, haematobium or mansoni
        #     :return symptoms: np.nan if no symptom or a list of symptoms
        #     """
        #     assert module_prefix in ['sm', 'sh'], "Incorrect infection type. Can't assign symptoms."
        #     params = self.module.parameters
        #     symptoms_possible = params['symptoms']
        #     all_symptoms_dict = self.sim.modules['Schisto'].parameters['symptoms_prevalence']
        #     # get the prevalence of the possible symptoms
        #     symptoms_dict = {k: all_symptoms_dict[k] for k in symptoms_possible}
        #     symptoms_exp = []
        #     for symptom in symptoms_dict.keys():
        #         prev = symptoms_dict[symptom]  # get the prevalence of the symptom among the infected population
        #         is_experienced = self.module.rng.rand() < prev
        #         # is_experienced = self.module.rng.choice([True, False], 1, p=[prev, 1-prev])
        #         if is_experienced:
        #             symptoms_exp.append(symptom)
        #     if len(symptoms_exp):
        #         return symptoms_exp
        #     return np.nan

        # # assign symptoms
        # symptoms = assign_symptoms(self.prefix)

        # THIS IS NOT NEEDED ANY LONGER BECAUSE WITH THE SYMPTOM BEING ONSET THERE WILL AUTOMATICALLY BE GENERIC HEALTHCARE SEEKING
        # if isinstance(symptoms, list):
        #     df.at[person_id, f'{self.module.prefix}_symptoms'] = symptoms
        #     # schedule Healthcare Seeking
        #     p = params['prob_seeking_healthcare']
        #     will_seek_treatment = self.module.rng.rand() < p
        #     # will_seek_treatment = self.module.rng.choice(['True', 'False'], size=1, p=[p, 1-p])
        #     if will_seek_treatment:
        #         seeking_treatment_ahead = int(self.module.rng.uniform(params['delay_till_hsi_a'],
        #                                                               params['delay_till_hsi_b'], size=1))
        #         # seeking_treatment_ahead = pd.to_timedelta(seeking_treatment_ahead, unit='D')
        #         df.loc[person_id, 'ss_scheduled_hsi_date'] = self.sim.date + DateOffset(days=seeking_treatment_ahead)
        #         seek_treatment_event = HSI_SchistoSeekTreatment(self.sim.modules['Schisto'], person_id=person_id)
        #         self.sim.modules['HealthSystem'].schedule_hsi_event(seek_treatment_event,
        #                                                             priority=1,
        #                                                             topen=df.loc[person_id, 'ss_scheduled_hsi_date'],
        #                                                             tclose=df.loc[person_id, 'ss_scheduled_hsi_date']
        #                                                             + DateOffset(weeks=502))

    def _assign_hsi_dates_initial(self, population, symptomatic_idx):
        """
        Schedules the treatment seeking to the initial population (only the clinical cases)
        :param population:
        :param symptomatic_idx: indices of people with symptoms
        """

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

    def log_counts_by_status_and_age_group(self):
        """Write to the log the counts by persons in each status by each age-group"""
        # todo this section could be simplified with groupby
        for age in ['PSAC', 'SAC', 'Adults', 'All']:
            self._write_to_log_count_of_states_for_age_group(age)

    def _write_to_log_count_of_states_for_age_group(self, age_group):
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

    def _add_DALYs_from_symptoms(self, symptoms):
        raise NotImplementedError

        # params = self.schisto_module.parameters
        # # todo factorize -- also, it's not in params, so remove??!!?
        # dalys_map = {
        #     'anemia': params['daly_wt_anemia'],
        #     'fever': params['daly_wt_fever'],
        #     'haematuria': params['daly_wt_haematuria'],
        #     'hydronephrosis': params['daly_wt_hydronephrosis'],
        #     'dysuria': params['daly_wt_dysuria'],
        #     'bladder_pathology': params['daly_wt_bladder_pathology'],
        #     'diarrhoea': params['daly_wt_diarrhoea'],
        #     'vomit': params['daly_wt_vomit'],
        #     'ascites': params['daly_wt_ascites'],
        #     'hepatomegaly': params['daly_wt_hepatomegaly']
        # }
        #
        # if isinstance(symptoms, list):
        #     symptoms = [dalys_map[s] for s in symptoms]
        #     return sum(symptoms)
        # else:
        #     return 0


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------


class SchistoInfectionWormBurdenEvent(RegularEvent, PopulationScopeEventMixin):
    """An event of infecting people with Schistosomiasis
    Using Worm Burden and Reservoir of Infectious Material - see write up
    """

    def __init__(self, module: Module, species: SchistoSpecies):
        """
        :param module: the module that created this event,
        must be either Schisto_Haematobium or Schisto_Mansoni
        """
        super().__init__(module, frequency=DateOffset(months=1))
        self.species = species

    def apply(self, population):
        params = self.species.params
        rng = self.module.rng
        prop = self.species._species_specific_property

        betas = [params['beta_PSAC'], params['beta_SAC'], params['beta_Adults']]
        R0 = params['R0']

        df = population.props
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
        contact_rates = age_group.map(beta_by_age_group)
        harbouring_rates = df.loc[where, prop('harbouring_rate')]
        rates = harbouring_rates * contact_rates.astype(float)
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
        to_establish = pd.DataFrame({'new_worms': draw_worms[(draw_worms > 0) & established]})

        # schedule maturation of the established worms
        to_establish['date_maturation'] = \
            self.sim.date + pd.to_timedelta(self.module.rng.randint(30, 55, size=len(to_establish)), unit='D')
        for index, row in to_establish.iterrows():
            self.sim.schedule_event(
                SchistoMatureWorms(
                    module=self.module,
                    species=self.species,
                    person_id=index,
                    new_worms=row.new_worms,
                ),
                row.date_maturation
            )

class SchistoMatureWorms(Event, IndividualScopeEventMixin):
    """Increases the aggregate worm burden of an individual upon maturation of the worms
    Changes the infection status accordingly
    Schedules the natural death of worms and symptoms development if High-infection
    """
    def __init__(self, module: Module, species: SchistoSpecies, person_id: int, new_worms: int):
        super().__init__(module, person_id=person_id)
        self.species = species
        self.new_worms = new_worms

    def apply(self, person_id):
        df = self.sim.population.props
        prop = self.species._species_specific_property
        params = self.species.params

        if df.loc[person_id, 'is_alive']:
            # increase worm burden
            df.loc[person_id, prop('aggregate_worm_burden')] += self.new_worms

            # schedule the natural death of the worms
            self.sim.schedule_event(
                SchistoWormsNatDeath(module=self.module,
                                     person_id=person_id,
                                     number_of_worms=self.new_worms,
                                     species=self.species),
                self.sim.date + DateOffset(years=params['worm_lifespan'])
            )

            if df.loc[person_id, prop('infection_status')] != 'High-infection':
                if df.loc[person_id, 'age_years'] < 5:
                    threshold = params['high_intensity_threshold_PSAC']
                else:
                    threshold = params['high_intensity_threshold']
                if df.loc[person_id, prop('aggregate_worm_burden')] >= threshold:
                    df.loc[person_id, prop('infection_status')] = 'High-infection'
                    df.loc[person_id, prop('start_of_high_infection')] = self.sim.date

                    # develop symptoms immediately
                    self.species._impose_symptoms(eligible_idx=[person_id])

                elif df.loc[person_id, prop('aggregate_worm_burden')] >= params['low_intensity_threshold']:
                    if df.loc[person_id, prop('infection_status')] == 'Non-infected':
                        df.loc[person_id, prop('infection_status')] = 'Low-infection'

            if \
                (df.loc[person_id, prop('infection_status')] != 'Non-infected') &\
                    (pd.isna(df.loc[person_id, prop('start_of_prevalent_period')])):
                df.loc[person_id, prop('start_of_prevalent_period')] = self.sim.date


class SchistoWormsNatDeath(Event, IndividualScopeEventMixin):
    """Decreases the aggregate worm burden of an individual upon natural death of the adult worms.
    This event checks the last day of PZQ treatment and if has been less than the lifespan of the worm
    it doesn't do anything (because the worms have been killed by the PZQ by now)."""

    def __init__(self, module: Module, species: SchistoSpecies, person_id: int, number_of_worms: int):
        super().__init__(module, person_id=person_id)
        self.species = species
        self.number_of_worms = number_of_worms

    def apply(self, person_id):
        df = self.sim.population.props
        prop = self.species._species_specific_property
        params = self.species.params

        worms_now = df.loc[person_id, prop('aggregate_worm_burden')]
        days_since_last_treatment = self.sim.date - df.loc[person_id, 'ss_last_PZQ_date']
        days_since_last_treatment = int(days_since_last_treatment / np.timedelta64(1, 'Y'))
        if days_since_last_treatment > params['worm_lifespan']:
            df.loc[person_id, prop('aggregate_worm_burden')] = worms_now - self.number_of_worms
            # clearance of the worms
            if df.loc[person_id, prop('aggregate_worm_burden')] < params['low_intensity_threshold']:
                df.loc[person_id, prop('infection_status')] = 'Non-infected'
                # does not matter if low or high int infection
                if df.loc[person_id, prop('infection_status')] != 'Non-infected':
                    # calculate prevalent period
                    prevalent_duration = self.sim.date - df.loc[person_id, prop('start_of_prevalent_period')]
                    prevalent_duration = int(prevalent_duration / np.timedelta64(1, 'D')) % 365
                    df.loc[person_id, prop('prevalent_days_this_year')] += prevalent_duration
                    df.loc[person_id, prop('start_of_prevalent_period')] = pd.NaT
                    df.loc[person_id, prop('start_of_high_infection')] = pd.NaT
            else:
                if df.loc[person_id, prop('infection_status')] == 'High-infection':
                    if df.loc[person_id, 'age_years'] < 5:
                        threshold = params['high_intensity_threshold_PSAC']
                    else:
                        threshold = params['high_intensity_threshold']
                    if df.loc[person_id, prop('aggregate_worm_burden')] < threshold:
                        df.loc[person_id, prop('infection_status')] = 'Low-infection'
                        high_inf_duration = self.sim.date - df.loc[person_id, prop('start_of_high_infection')]
                        high_inf_duration = int(high_inf_duration / np.timedelta64(1, 'D')) % 365
                        df.loc[person_id, prop('high_inf_days_this_year')] += high_inf_duration
                        df.loc[person_id, prop('start_of_high_infection')] = pd.NaT


# TODO: Should this be a function or an HSI?
class SchistoTreatmentEvent(Event, IndividualScopeEventMixin):
    """Cured upon PZQ treatment through HSI or MDA (Infected -> Non-infected)
    PZQ treats both types of infections, so affect symptoms and worm burden of any infection type registered
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Schisto)

    def apply(self, person_id):
        df = self.sim.population.props

        prefixes = []
        if 'Schisto_Haematobium' in self.sim.modules.keys():
            prefixes.append('sh')
        if 'Schisto_Mansoni' in self.sim.modules.keys():
            prefixes.append('sm')

        if not df.loc[person_id, 'is_alive']:
            return

        for prefix in prefixes:
            if df.loc[person_id, f'{prefix}_infection_status'] != 'Non-infected':

                # check if they experienced symptoms, and if yes, treat them
                df.loc[person_id, f'{prefix}_symptoms'] = np.nan
                # if isinstance(df.loc[person_id, prefix + '_symptoms'], list):
                #     df.loc[person_id, prefix + '_symptoms'] = np.nan

                # calculate the duration of the prevalent period
                prevalent_duration = count_days_this_year(self.sim.date, df.loc[
                    person_id, f'{prefix}_start_of_prevalent_period'])
                df.loc[person_id, f'{prefix}_prevalent_days_this_year'] += prevalent_duration
                df.loc[person_id, f'{prefix}_start_of_prevalent_period'] = pd.NaT

                # calculate the duration of the high-intensity infection
                if df.loc[person_id, f'{prefix}_infection_status'] == 'High-infection':
                    high_infection_duration = count_days_this_year(self.sim.date, df.loc[
                        person_id, f'{prefix}_start_of_high_infection'])
                    df.loc[person_id, f'{prefix}_high_inf_days_this_year'] += high_infection_duration
                    df.loc[person_id, f'{prefix}_start_of_high_infection'] = pd.NaT

                df.loc[person_id, f'{prefix}_aggregate_worm_burden'] = 0  # PZQ_efficacy = 100% for now
                df.loc[person_id, f'{prefix}_start_of_prevalent_period'] = pd.NaT
                df.loc[person_id, f'{prefix}_start_of_high_infection'] = pd.NaT
                df.loc[person_id, f'{prefix}_infection_status'] = 'Non-infected'

        # the general Schisto module properties
        df.loc[person_id, 'ss_scheduled_hsi_date'] = pd.NaT
        df.loc[person_id, 'ss_last_PZQ_date'] = self.sim.date

# ---------------------------------------------------------------------------------------------------------
#   HSI EVENTS
# ---------------------------------------------------------------------------------------------------------

#TODO: Consider if this should be the generic HSI
class HSI_SchistoSeekTreatment(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event of seeking treatment for a person with symptoms"""
    # todo should this be handled with generic appointments?
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Schisto)

        under_5 = self.sim.population.props.at[person_id, 'age_years'] <= 5
        self.TREATMENT_ID = 'Schisto_Treatment_seeking'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Under5OPD' if under_5 else 'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        params = self.module.parameters

        prefixes = []
        if 'Schisto_Haematobium' in self.sim.modules.keys():
            prefixes.append('sh')
        if 'Schisto_Mansoni' in self.sim.modules.keys():
            prefixes.append('sm')
        is_infected = False
        for pref in prefixes:
            if df.loc[person_id, f'{pref}_infection_status'] != 'Non-infected':
                is_infected = True

        # appt are scheduled and cannot be cancelled in the following situations:
        #   a) person has died
        #   b) the infection has been treated in MDA or by treating symptoms from
        #   other schisto infection before the appt happened
        if (df.loc[person_id, 'is_alive'] & is_infected):  # &
            # (df.loc[person_id, 'ss_scheduled_hsi_date'] <= self.sim.date)):
            # check if a person is a child or an adult and assign prob of being sent to schisto test (hence being cured)
            if df.loc[person_id, 'age_years'] <= 15:
                prob_test = params['prob_sent_to_lab_test_children']
            else:
                prob_test = params['prob_sent_to_lab_test_adults']

            sent_to_test = self.module.rng.rand() < prob_test
            # sent_to_test = self.module.rng.choice([True, False], p=[prob_test, 1-prob_test])
            # use this is you don't care about whether PZQ is available or not
            # if sent_to_test:
            #     self.sim.schedule_event(SchistoTreatmentEvent(self.module, person_id), self.sim.date)
            if sent_to_test:
                # request the consumable
                consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
                items_code1 = \
                    pd.unique(
                        consumables.loc[
                            consumables['Items'] == "Praziquantel, 600 mg (donated)", 'Item_Code'])[0]
                the_cons_footprint = {'Intervention_Package_Code': {}, 'Item_Code': {items_code1: 1}}
                outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=self, cons_req_as_footprint=the_cons_footprint, to_log=False
                )

                # give the PZQ to the patient
                if outcome_of_request_for_consumables['Item_Code'][items_code1]:
                    self.sim.modules['HealthSystem'].request_consumables(
                        hsi_event=self, cons_req_as_footprint=the_cons_footprint, to_log=True
                    )
                    # patient is cured
                    self.sim.schedule_event(SchistoTreatmentEvent(self.module, person_id), self.sim.date)

            else:  # person seeked treatment but was not sent to test; visit is reschedulled
                # schedule another Seeking Treatment event for that person
                seeking_treatment_ahead_repeated = \
                    int(self.module.rng.uniform(params['delay_till_hsi_a_repeated'],
                                                params['delay_till_hsi_b_repeated']))
                seeking_treatment_ahead_repeated = pd.to_timedelta(seeking_treatment_ahead_repeated, unit='D')
                df.loc[person_id, 'ss_scheduled_hsi_date'] = self.sim.date + seeking_treatment_ahead_repeated

                seek_treatment_repeated = HSI_SchistoSeekTreatment(self.module, person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(seek_treatment_repeated,
                                                                    priority=1,
                                                                    topen=df.loc[person_id, 'ss_scheduled_hsi_date'],
                                                                    tclose=df.loc[person_id, 'ss_scheduled_hsi_date']
                                                                    + DateOffset(weeks=500))

    def did_not_run(self):
        return True


# ---------------------------------------------------------------------------------------------------------
#   MASS-DRUG ADMINISTRATION EVENTS
# ---------------------------------------------------------------------------------------------------------
# TODO: Check function?
class SchistoHistoricalMDAEvent(Event, PopulationScopeEventMixin):
    """Mass-Drug administration scheduled for the population
    Using the historical MDA coverage
    """
    def __init__(self, module):
        super().__init__(module)
        assert isinstance(module, Schisto)

    def apply(self, population):
        print("Historical MDA is happening now!")
        year = self.sim.date.year
        # this might look hacky, but it's done because the data for the MDA is for only these specific years
        # and the simulation might have to run longer to equilibrate
        year = year - self.sim.start_date.year + 2015
        assert year in [2015, 2016, 2017, 2018], "No historical coverage data for this year"

        treated_idx_PSAC = self.assign_historical_MDA_coverage(population, year, 'PSAC')
        treated_idx_SAC = self.assign_historical_MDA_coverage(population, year, 'SAC')
        treated_idx_Adults = self.assign_historical_MDA_coverage(population, year, 'Adults')

        treated_idx = treated_idx_PSAC + treated_idx_SAC + treated_idx_Adults
        # all treated people will have worm burden decreased, and we already have chosen only alive people
        for person_id in treated_idx:
            self.sim.schedule_event(SchistoTreatmentEvent(self.module, person_id), self.sim.date)

    def assign_historical_MDA_coverage(self, population, year, age_group):
        """Assign coverage of MDA program to chosen age_group.

          :param population: population
          :param year: current year. used to find the coverage
          :param age_group: 'SAC', 'PSAC', 'Adults'
          """
        assert year in [2015, 2016, 2017, 2018], "No data for requested MDA coverage"

        df = population.props
        params = self.module.parameters
        districts = df.district_of_residence.unique()

        age_range = map_age_groups(age_group)  # returns a tuple (a,b) a <= age_group <= b

        param_str = 'MDA_coverage_' + age_group
        coverage = params[param_str]  # this is a pd.Series not a single value
        coverage = coverage[:, year]
        MDA_idx = []  # store indices of treated individuals

        for distr in districts:
            coverage_distr = coverage[distr]  # get a correct value from the pd.Series
            eligible = df.index[(df['is_alive']) & (df['district_of_residence'] == distr) &
                                (df['age_years'].between(age_range[0], age_range[1]))]
            if len(eligible):
                MDA_idx_distr = self.module.rng.choice(eligible,
                                                       size=int(coverage_distr * (len(eligible))), replace=False)
                MDA_idx = MDA_idx + MDA_idx_distr

        return MDA_idx

# TODO: Make this into HSI (Population Level)
class SchistoPrognosedMDAEvent(RegularEvent, PopulationScopeEventMixin):
    """Mass-Drug administration scheduled for the population
    Using the proposed MDA coverage
    """
    def __init__(self, module, freq, district):
        super().__init__(module, frequency=DateOffset(months=freq))
        self.district = district
        assert isinstance(module, Schisto)

    def apply(self, population):
        print("Prognosed MDA is happening now!")
        district = self.district

        treated_idx_PSAC = self.assign_prognosed_MDA_coverage(population, district, 'PSAC')
        treated_idx_SAC = self.assign_prognosed_MDA_coverage(population, district, 'SAC')
        treated_idx_Adults = self.assign_prognosed_MDA_coverage(population, district, 'Adults')

        treated_idx = treated_idx_PSAC + treated_idx_SAC + treated_idx_Adults
        # all treated people will have worm burden decreased, and we already have chosen only alive people
        for person_id in treated_idx:
            self.sim.schedule_event(SchistoTreatmentEvent(self.module, person_id), self.sim.date)

    def assign_prognosed_MDA_coverage(self, population, district, age_group):
        """Assign coverage of MDA program to chosen age_group. The same coverage for every district.

          :param district: district for which the MDA coverage is required
          :param population: population
          :param age_group: 'SAC', 'PSAC', 'Adults'
          :returns MDA_idx: indices of people that will be administered PZQ in the MDA program
          """

        df = population.props
        params = self.module.parameters
        age_range = map_age_groups(age_group)  # returns a tuple (a,b) a <= age_group <= b
        param_str = 'MDA_prognosed_' + age_group

        coverage = params[param_str]
        coverage_distr = coverage[district]

        eligible = df.index[(df.is_alive) & (df['district_of_residence'] == district)
                            & (df['age_years'].between(age_range[0], age_range[1]))]
        MDA_idx = []
        if len(eligible):
            MDA_idx = self.module.rng.choice(eligible,
                                             size=int(coverage_distr * (len(eligible))),
                                             replace=False)

        return MDA_idx


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
# ---------------------------------------------------------------------------------------------------------


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
        count_states.update({'Prevalence': inf/total_pop_size})
        count_states.update({'High-inf_Prevalence': high_inf/total_pop_size})

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
