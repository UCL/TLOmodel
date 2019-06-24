"""
Male circumcision
"""
import os
import logging

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class male_circumcision(Module):
    """
    male circumcision, without health system links
    """

    def __init__(self, name=None, resourcefilepath=None, par_est5=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    PARAMETERS = {
        'initial_circumcision': Parameter(Types.REAL, 'Prevalence of circumcision in the population at baseline'),
        'prob_circumcision': Parameter(Types.REAL, 'probability of circumcision in the eligible population'),
        'prop_muslim': Parameter(Types.REAL, 'proportion of population muslim'),
        'daly_wt_circumcision': Parameter(
            Types.REAL, 'DALY weight for circumcision'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'mc_is_circumcised': Property(Types.BOOL, 'individual is circumcised'),
        'mc_date_circumcised': Property(Types.DATE, 'Date of circumcision'),
        'mc_specific_symptoms': Property(
            Types.CATEGORICAL, 'Level of symptoms for circumcision specifically',
            categories=['none']),
        'mc_unified_symptom_code': Property(
            Types.CATEGORICAL,
            'Level of symptoms on the standardised scale (governing health-care seeking): '
            '0=None; 1=Mild; 2=Moderate; 3=Severe; 4=Extreme_Emergency',
            categories=[0, 1, 2, 3, 4]),

    }

    def read_parameters(self, data_folder):
        workbook = pd.read_excel(os.path.join(self.resourcefilepath,
                                              'ResourceFile_HIV.xlsx'), sheet_name=None)

        params = self.parameters
        params['circ_coverage'] = workbook['circumcision']

        params['param_list'] = workbook['parameters']
        self.param_list.set_index("Parameter", inplace=True)
        params['prop_muslim'] = self.param_list.loc['prop_muslim', 'Value1']

        params['daly_wt_circumcision'] = 0

    def initialise_population(self, population):
        df = population.props
        now = self.sim.date.year

        df['mc_is_circumcised'] = False  # default: no individuals circumcised
        df['mc_date_circumcised'] = pd.NaT  # default: not a time
        df['mc_specific_symptoms'] = 'none'
        df['mc_unified_symptom_code'] = 0

        init_circumcision = self.circ_coverage.loc[self.circ_coverage.year == now, 'coverage'].values[0]
        # print('initial_circumcision', self.parameters['initial_circumcision'])

        # select all eligible uncircumcised men
        uncircum = df.index[df.is_alive & (df.age_years >= 15) & ~df.mc_is_circumcised & (df.sex == 'M')]
        # print('uncircum', len(uncircum))

        # 2. baseline prevalence of circumcisions
        circum = self.rng.choice([True, False], size=len(uncircum),
                                  p=[init_circumcision,
                                     1 - init_circumcision])

        # print('circum', circum.sum())

        # if any are circumcised
        if circum.sum():
            circum_idx = uncircum[circum]
            # print('circum_idx', len(circum_idx))

            df.loc[circum_idx, 'mc_is_circumcised'] = True
            df.loc[circum_idx, 'mc_date_circumcised'] = self.sim.date

    def initialise_simulation(self, sim):

        sim.schedule_event(CircumcisionEvent(self), sim.date + DateOffset(months=12))
        sim.schedule_event(CircumcisionLoggingEvent(self), sim.date + DateOffset(months=1))

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        """
        df = self.sim.population.props
        # TODO: apply circumcision property to 15% of male infants

        df.at[child_id, 'mc_is_circumcised'] = False
        df.at[child_id, 'mc_date_circumcised'] = pd.NaT
        df.at[child_id, 'mc_specific_symptoms'] = 'none'
        df.at[child_id, 'mc_unified_symptom_code'] = 0

        # select 15% of male births for circumcision
        if (df.at[child_id, 'sex'] == 'M') & (self.sim.rng.random_sample(size=1) < self.parameters['prop_muslim']):
            df.at[child_id, 'mc_is_circumcised'] = True
            df.at[child_id, 'mc_date_circumcised'] = self.sim.date

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        logger.debug('This is circumcision, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug('This is circumcision reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe

        health_values = df.loc[df.is_alive, 'mc_specific_symptoms'].map({
            'none': 0
        })
        health_values.name = 'circumcision symptoms'    # label the cause of this disability

        return health_values.loc[df.is_alive]


class CircumcisionEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))

    def apply(self, population):
        now = self.sim.date
        df = population.props
        params = self.module.parameters

        # Determine if anyone will request circumcision
        eligible = (df['is_alive']) & (~df['mc_is_circumcised']) & (df['age_years'] >= 10) & (
            df['age_years'] < 35) & (df['sex'] == 'M')

        prob_circum = params['circ_coverage'].loc[params['circ_coverage'].year == now.year, 'prob_circum'].values[0]

        seeks_care = pd.Series(data=False, index=df.loc[eligible].index)
        for i in df.index[eligible]:
            # this doesn't go through the health_system 'get_prob_seeks_care', uses gov targets instead
            seeks_care[i] = self.module.rng.rand() < prob_circum

        if seeks_care.sum() > 0:
            for person_index in seeks_care.index[seeks_care == True]:
                logger.debug(
                    'This is CircumcisionEvent, scheduling Circumcision_PresentsForCare for person %d',
                    person_index)
                event = HSI_Circumcision_PresentsForCare(self.module, person_id=person_index)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=1,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(weeks=4)
                                                                )
        else:
            logger.debug(
                'This is CircumcisionEvent, There is  no one with new severe symptoms so no new healthcare seeking')


# ---------------------------------------------------------------------------
#   Health system interactions
# ---------------------------------------------------------------------------


class HSI_Circumcision_PresentsForCare(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['MinorSurg'] = 1  # This requires one out patient

        # Get the consumables required
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = \
            pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Male circumcision ', 'Intervention_Pkg_Code'])[
                0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Circumcision'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1, 2, 3]
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug('This is HSI_Circumcision_PresentsForCare, a first appointment for person %d', person_id)

        df = self.sim.population.props  # shortcut to the dataframe

        df.at[person_id, 'mc_is_circumcised'] = True
        df.at[person_id, 'mc_date_circumcised'] = self.sim.date


class CircumcisionLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        self.repeat = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):

        df = population.props

        circumcised_total = len(df.index[df.is_alive & (df.age_years >= 15) & df.mc_is_circumcised & (df.sex == 'M')])
        proportion_circumcised = circumcised_total / len(
            df.index[df.is_alive & (df.age_years >= 15) & (df.sex == 'M')])

        mask = (df['mc_date_circumcised'] > self.sim.date - DateOffset(months=self.repeat))

        logger.info('%s|summary|%s', self.sim.date,
                    {
                        'TotalCircumcised': circumcised_total,
                        'PropCircumcised': proportion_circumcised,
                    })
