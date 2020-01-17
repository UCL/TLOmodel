"""
Childhood malnutrition module
Documentation: 04 - Methods Repository/Method_Child_EntericInfection.xlsx
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Malnutrition(Module):

    PARAMETERS = {
        #
        # 'eq_for_alloc_shigella': Parameter(Types.Eq, 'the e.... '),
        # 'eq_for_alloc_rota': Parameter(Types.Eq), 'the XX...')

        'base_incidence_diarrhoea_by_rotavirus':
            Parameter(Types.LIST, 'incidence of diarrhoea caused by rotavirus in age groups 0-11, 12-23, 24-59 months '
                      ),
        'base_incidence_diarrhoea_by_shigella':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by shigella spp in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_adenovirus':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by adenovirus 40/41 in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_crypto':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by cryptosporidium in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_campylo':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by campylobacter spp in age groups 0-11, 12-23, 24-59 months'
                      ),
    }

    PROPERTIES = {
        'gi_diarrhoea_status': Property(Types.BOOL, 'symptomatic infection - diarrhoea disease'),
        'gi_diarrhoea_pathogen': Property(Types.CATEGORICAL, 'attributable pathogen for diarrhoea',
                                          categories=['rotavirus', 'shigella', 'adenovirus', 'cryptosporidium',
                                                      'campylobacter', 'ST-ETEC', 'sapovirus', 'norovirus',
                                                      'astrovirus', 'tEPEC']),
        'gi_diarrhoea_type': Property(Types.CATEGORICAL, 'progression of diarrhoea type',
                                      categories=['acute', 'prolonged', 'persistent']),
        'gi_diarrhoea_acute_type': Property(Types.CATEGORICAL, 'clinical acute diarrhoea type',
                                            categories=['dysentery', 'acute watery diarrhoea']),
        'gi_dehydration_status': Property(Types.CATEGORICAL, 'dehydration status',
                                          categories=['no dehydration', 'some dehydration', 'severe dehydration']),
        'gi_persistent_diarrhoea': Property(Types.CATEGORICAL,
                                            'diarrhoea episode longer than 14 days with or without dehydration',
                                            categories=['persistent diarrhoea', 'severe persistent diarrhoea']),
        'gi_diarrhoea_death': Property(Types.BOOL, 'death caused by diarrhoea'),
    }

    # Declares symptoms
    SYMPTOMS = {'watery diarrhoea', 'bloody diarrhoea', 'fever', 'vomiting', 'dehydration', 'persistent diarrhoea'}

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module """

        p = self.parameters
        m = self
        dfd = pd.read_excel(
            Path(self.resourcefilepath) / 'ResourceFile_Childhood_Diarrhoea.xlsx', sheet_name='Parameter_values')
        dfd.set_index("Parameter_name", inplace=True)
        # self.load_parameters_from_dataframe(dfd)

        # all diarrhoea prevalence values
        p['rp_acute_diarr_age12to23mo'] = dfd.loc['rp_acute_diarr_age12to23mo', 'value1']
        p['rp_acute_diarr_age24to59mo'] = dfd.loc['rp_acute_diarr_age24to59mo', 'value1']
        p['rp_acute_diarr_HIV'] = dfd.loc['rp_acute_diarr_HIV', 'value1']
        p['rp_acute_diarr_SAM'] = dfd.loc['rp_acute_diarr_SAM', 'value1']
        p['rp_acute_diarr_excl_breast'] = dfd.loc['rp_acute_diarr_excl_breast', 'value1']
        p['rp_acute_diarr_cont_breast'] = dfd.loc['rp_acute_diarr_cont_breast', 'value1']
        p['rp_acute_diarr_HHhandwashing'] = dfd.loc['rp_acute_diarr_HHhandwashing', 'value1']
        p['rp_acute_diarr_clean_water'] = dfd.loc['rp_acute_diarr_clean_water', 'value1']

        # DALY weights
        if 'HealthBurden' in self.sim.modules.keys():
            p['daly_mild_malnutrition'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=32)
            p['daly_moderate_malnutrition'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=35)
            p['daly_severe_malnutrition'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=34)

    def initialise_population(self, population):
        """Set our property values for the initial population.
        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.
        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals
        rng = self.rng

        # DEFAULTS
        df['gi_diarrhoea_status'] = False
        df['gi_diarrhoea_acute_type'] = np.nan
        df['gi_diarrhoea_pathogen'] = np.nan
        df['gi_diarrhoea_type'] = np.nan
        df['gi_persistent_diarrhoea'] = np.nan
        df['gi_dehydration_status'] = 'no dehydration'
        df['date_of_onset_diarrhoea'] = pd.NaT
        df['gi_recovered_date'] = pd.NaT

    def initialise_simulation(self, sim):
        """
        Get ready for simulation start.
        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event for acute diarrhoea ---------------------------------------------------
        sim.schedule_event(AcuteMalnutritionEvent(self), sim.date + DateOffset(months=0))

        # add an event to log to screen
        sim.schedule_event(MalnutritionLoggingEvent(self), sim.date + DateOffset(months=1))

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props

        df.at[child_id, 'gi_recovered_date'] = pd.NaT
        df.at[child_id, 'gi_diarrhoea_status'] = False
        df.at[child_id, 'gi_diarrhoea_acute_type'] = np.nan
        df.at[child_id, 'gi_diarrhoea_type'] = np.nan
        df.at[child_id, 'gi_persistent_diarrhoea'] = np.nan
        df.at[child_id, 'gi_dehydration_status'] = 'no dehydration'
        df.at[child_id, 'date_of_onset_diarrhoea'] = pd.NaT

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is Diarrhoea, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)
        pass

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug('This is diarrhoea reporting my health values')

        df = self.sim.population.props
        p = self.parameters

        health_values = df.loc[df.is_alive, 'gi_dehydration_status'].map({
            'none': 0,
            'no dehydration': p['daly_mild_diarrhoea'],     # TODO; maybe rename and checkdaly_mild_dehydration_due_to_diarrahea
            'some dehydration': p['daly_moderate_diarrhoea'],
            'severe dehydration': p['daly_severe_diarrhoea']
        })
        health_values.name = 'dehydration'    # label the cause of this disability

        #TODO: is it right that the only thing causing lays from diarrhoa is the dehydration
        #TODO: are these dalys for the episode of diarrhoa of for an amount of time?
        #TODO; nb that this will change when symtoms tracked in SymptomManager

        return health_values.loc[df.is_alive]   # returns the series


class AcuteDiarrhoeaEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=3))

    def apply(self, population):
        # TODO: Say what this event is for and what it will do.
        """Apply this event to the population.
        :param population: the current population
        """

        df = population.props
        m = self.module
        rng = m.rng
        now = self.sim.date

