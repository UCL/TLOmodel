"""
TB infections
"""

import logging
import os

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent, IndividualScopeEventMixin
from tlo.methods import healthsystem, demography

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO: add logging

# TODO: remove store
class tb(Module):
    """ Set up the baseline population with TB prevalence
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.store = {'Time': [], 'Total_active_tb': [], 'Total_active_tb_mdr': [], 'Total_co-infected': [],
                      'TB_deaths': [],
                      'Time_death_TB': []}

    PARAMETERS = {
        'prop_fast_progressor': Parameter(Types.REAL,
                                          'Proportion of infections that progress directly to active stage'),
        'transmission_rate': Parameter(Types.REAL, 'TB transmission rate, estimated by Juan'),
        'monthly_prob_progr_active': Parameter(Types.REAL,
                                               'Combined rate of progression/reinfection/relapse from Juan'),
        'rr_tb_hiv_stages': Parameter(Types.REAL, 'relative risk of tb in hiv+ compared with hiv- by cd4 stage'),
        'rr_tb_art': Parameter(Types.REAL, 'relative risk of tb in hiv+ on art'),
        'rr_tb_ipt': Parameter(Types.REAL, 'relative risk of tb on ipt'),
        'rr_tb_malnourished': Parameter(Types.REAL, 'relative risk of tb with malnourishment'),
        'rr_tb_diabetes1': Parameter(Types.REAL, 'relative risk of tb with diabetes type 1'),
        'rr_tb_alcohol': Parameter(Types.REAL, 'relative risk of tb with heavy alcohol use'),
        'rr_tb_smoking': Parameter(Types.REAL, 'relative risk of tb with smoking'),
        'rr_tb_pollution': Parameter(Types.REAL, 'relative risk of tb with indoor air pollution'),
        'rel_infectiousness_hiv': Parameter(Types.REAL, 'relative infectiousness of tb in hiv+ compared with hiv-'),
        'monthly_prob_self_cure': Parameter(Types.REAL, 'annual probability of self-cure'),
        'monthly_prob_tb_mortality': Parameter(Types.REAL, 'mortality rate with active tb'),
        'monthly_prob_tb_mortality_hiv': Parameter(Types.REAL, 'mortality from tb with concurrent HIV'),
        'monthly_prob_relapse_tx_complete': Parameter(Types.REAL,
                                                      'annual probability of relapse once treatment complete'),
        'monthly_prob_relapse_tx_incomplete': Parameter(Types.REAL,
                                                        'annual probability of relapse if treatment incomplete'),
        'monthly_prob_relapse_2yrs': Parameter(Types.REAL,
                                               'annual probability of relapse 2 years after treatment complete'),
        'prob_treatment_success': Parameter(Types.REAL,
                                            'probability of treatment completion'),

        'prop_mdr2010': Parameter(Types.REAL, 'prevalence of mdr in TB cases 2010'),
        'prop_mdr_new': Parameter(Types.REAL, 'prevalence of mdr in new tb cases'),
        'prop_mdr_retreated': Parameter(Types.REAL, 'prevalence of mdr in previously treated cases'),
        'tb_testing_coverage': Parameter(Types.REAL, 'proportion of population tested'),
        'prop_smear_positive': Parameter(Types.REAL, 'proportion of TB cases smear positive'),
        'prop_smear_positive_hiv': Parameter(Types.REAL, 'proportion of HIV/TB cases smear positive'),
        'testing_prob_xpert': Parameter(Types.REAL, 'probability of individual receiving xpert test'),
        'prop_xpert_positive': Parameter(Types.REAL,
                                         'proportion active tb cases tested with xpert with positive results'),
        'prob_tb_treatment': Parameter(Types.REAL, 'probability of individual starting treatment'),
        'prob_mdr': Parameter(Types.REAL, 'probability tb case is mdr'),
        'prob_tb_mdr_treatment': Parameter(Types.REAL, 'probability of individual starting mdr treatment'),

        'qalywt_latent':
            Parameter(Types.REAL, 'QALY weighting for latent tb'),
        'qalywt_active':
            Parameter(Types.REAL, 'QALY weighting for active tb'),
    }

    PROPERTIES = {
        'tb_inf': Property(Types.CATEGORICAL,
                           categories=['uninfected', 'latent_susc_primary', 'active_susc_primary', 'latent_mdr_primary',
                                       'active_mdr_primary',
                                       'latent_susc_secondary', 'latent_mdr_secondary', 'active_susc_secondary',
                                       'active_mdr_secondary'],
                           description='tb status'),
        'tb_date_active': Property(Types.DATE, 'Date active tb started'),
        'tb_date_latent': Property(Types.DATE, 'Date acquired tb infection (latent stage)'),
        'tb_date_death': Property(Types.DATE, 'Projected time of tb death if untreated'),
        'tb_ever_tb': Property(Types.BOOL, 'if ever had active drug-susceptible tb'),
        'tb_ever_tb_mdr': Property(Types.BOOL, 'if ever had active multi-drug resistant tb'),
        'tb_specific_symptoms': Property(Types.CATEGORICAL, 'Level of symptoms for tb',
                                         categories=['none', 'latent', 'active']),
        'tb_unified_symptom_code': Property(Types.CATEGORICAL, 'level of symptoms on the standardised scale, 0-4',
                                            categories=[0, 1, 2, 3, 4]),
        'tb_ever_tested': Property(Types.BOOL, 'ever had a tb test'),
        'tb_smear_test': Property(Types.BOOL, 'ever had a tb smear test'),
        'tb_result_smear_test': Property(Types.BOOL, 'result from tb smear test'),
        'tb_date_smear_test': Property(Types.DATE, 'date of tb smear test'),
        'tb_xpert_test': Property(Types.BOOL, 'ever had a tb Xpert test'),
        'tb_result_xpert_test': Property(Types.BOOL, 'result from tb Xpert test'),
        'tb_date_xpert_test': Property(Types.DATE, 'date of tb Xpert test'),
        'tb_diagnosed': Property(Types.BOOL, 'current diagnosis of active tb'),
        'tb_mdr_diagnosed': Property(Types.BOOL, 'current diagnosis of active tb_mdr'),
        'tb_treated': Property(Types.BOOL, 'on tb treatment regimen'),
        'tb_date_treated': Property(Types.DATE, 'date tb treatment started'),
        'tb_treatment_failure': Property(Types.BOOL, 'failed first line tb treatment'),
        'tb_treated_mdr': Property(Types.BOOL, 'on tb treatment MDR regimen'),
        'tb_date_treated_mdr': Property(Types.DATE, 'date tb MDR treatment started'),
        'request_mdr_regimen': Property(Types.BOOL, 'request for mdr treatment'),
    }

    TREATMENT_ID = 'tb_treatment'
    MDR_TREATMENT_ID = 'tb_mdr_treatment'
    SPUTUM_TEST_ID = 'tb_sputum_test'
    XPERT_TEST_ID = 'tb_xpert_test'
    FOLLOWUP_ID = 'tb_followup'

    def read_parameters(self, data_folder):

        workbook = pd.read_excel(os.path.join(self.resourcefilepath,
                                              'Method_TB.xlsx'), sheet_name=None)

        params = self.parameters
        params['param_list'] = workbook['parameters']
        self.param_list.set_index("parameter", inplace=True)

        params['prop_fast_progressor'] = self.param_list.loc['prop_fast_progressor', 'value1']
        params['transmission_rate'] = self.param_list.loc['transmission_rate', 'value1']
        params['monthly_prob_progr_active'] = self.param_list.loc['monthly_prob_progr_active', 'value1']

        params['rr_tb_with_hiv_stages'] = self.param_list.loc['transmission_rate'].values
        params['rr_tb_art'] = self.param_list.loc['rr_tb_art', 'value1']
        params['rr_tb_ipt'] = self.param_list.loc['rr_tb_ipt', 'value1']
        params['rr_tb_malnourished'] = self.param_list.loc['rr_tb_malnourished', 'value1']
        params['rr_tb_diabetes1'] = self.param_list.loc['rr_tb_diabetes1', 'value1']
        params['rr_tb_alcohol'] = self.param_list.loc['rr_tb_alcohol', 'value1']
        params['rr_tb_smoking'] = self.param_list.loc['rr_tb_smoking', 'value1']
        params['rr_tb_pollution'] = self.param_list.loc['rr_tb_pollution', 'value1']
        params['rel_infectiousness_hiv'] = self.param_list.loc['rel_infectiousness_hiv', 'value1']
        params['monthly_prob_self_cure'] = self.param_list.loc['monthly_prob_self_cure', 'value1']
        params['monthly_prob_tb_mortality'] = self.param_list.loc['ann_prob_tb_mortality', 'value1']
        params['monthly_prob_tb_mortality_hiv'] = self.param_list.loc['ann_prob_tb_mortality_hiv', 'value1']
        params['prop_mdr2010'] = self.param_list.loc['prop_mdr2010', 'value1']
        params['prop_mdr_new'] = self.param_list.loc['prop_mdr_new', 'value1']
        params['prop_mdr_retreated'] = self.param_list.loc['prop_mdr_retreated', 'value1']
        params['monthly_prob_relapse_tx_complete'] = self.param_list.loc['monthly_prob_relapse_tx_complete', 'value1']
        params['monthly_prob_relapse_tx_incomplete'] = self.param_list.loc[
            'monthly_prob_relapse_tx_incomplete', 'value1']
        params['monthly_prob_relapse_2yrs'] = self.param_list.loc['monthly_prob_relapse_2yrs', 'value1']
        params['prob_treatment_success'] = self.param_list.loc['prob_treatment_sucess', 'value1']
        params['Active_tb_prob'], params['Latent_tb_prob'] = workbook['Active_TB_prob'], \
                                                             workbook['Latent_TB_prob']

        params['tb_testing_coverage'] = 0.1  # dummy value
        params['prop_smear_positive'] = 0.8
        params['prop_smear_positive_hiv'] = 0.5
        params['testing_prob_xpert'] = 0.7
        params['prop_xpert_positive'] = 0.5
        params['prob_tb_treatment'] = 0.75
        params['prob_mdr'] = 0.05
        params['prob_tb_mdr_treatment'] = 0.8

        params['qalywt_latent'] = self.sim.modules['QALY'].get_qaly_weight(0)
        params['qalywt_active'] = self.sim.modules['QALY'].get_qaly_weight(1)
        # Drug-susceptible, Multidrug-resistant and Extensively drug-resistant tb all have the same DALY weights
        # TODO: co-infected hiv-tb need different DALY weights

    def initialise_population(self, population):
        """Set our property values for the initial population.
        """
        df = population.props
        now = self.sim.date

        # set-up baseline population
        df['tb_inf'].values[:] = 'uninfected'
        df['tb_date_active'] = pd.NaT
        df['tb_date_latent'] = pd.NaT
        df['tb_date_death'] = pd.NaT

        df['tb_ever_tb'] = False
        df['tb_ever_tb_mdr'] = False

        df['tb_specific_symptoms'].values[:] = 'none'
        df['tb_unified_symptom_code'].values[:] = 0

        df['tb_ever_tested'] = False  # default: no individuals tested
        df['tb_smear_test'] = False
        df['tb_result_smear_test'] = False
        df['tb_date_smear_test'] = pd.NaT
        df['tb_xpert_test'] = False
        df['tb_result_xpert_test'] = False
        df['tb_date_xpert_test'] = pd.NaT
        df['tb_diagnosed'] = False
        df['tb_mdr_diagnosed'] = False
        df['tb_treated'] = False
        df['tb_date_treated'] = pd.NaT
        df['tb_treatment_failure'] = False
        df['tb_treatedMDR'] = False
        df['tb_date_treatedMDR'] = pd.NaT
        df['request_mdr_regimen'] = False

        # TB infections - active / latent
        # baseline infections not weighted by RR, randomly assigned
        active_tb_data = self.parameters['Active_tb_prob']
        latent_tb_data = self.parameters['Latent_tb_prob']

        active_tb_prob_year = active_tb_data.loc[
            active_tb_data.Year == now.year, ['ages', 'Sex', 'incidence_per_capita']]

        # TODO: condense this with a merge function and remove if statements
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ LATENT ~~~~~~~~~~~~~~~~~~~~~~~~~~

        for i in range(0, 81):
            idx = (df.age_years == i) & (df.sex == 'M') & (df.tb_inf == 'uninfected') & df.is_alive

            # MALE
            if idx.any():
                # sample from uninfected population using WHO prevalence
                fraction_latent_tb = latent_tb_data.loc[
                    (latent_tb_data.sex == 'M') & (latent_tb_data.age == i), 'prob_latent_tb']
                male_latent_tb = df[idx].sample(frac=fraction_latent_tb).index
                df.loc[male_latent_tb, 'tb_inf'] = 'latent_susc_primary'
                df.loc[male_latent_tb, 'tb_date_latent'] = now
                df.loc[male_latent_tb, 'tb_specific_symptoms'] = 'latent'

                # allocate some latent infections as mdr-tb
                if len(df[df.is_alive & (df.sex == 'M') & (df.tb_inf == 'latent_susc_primary')]) > 10:
                    idx_c = df[df.is_alive & (df.sex == 'M') & (df.tb_inf == 'latent_susc_primary')].sample(
                        frac=self.parameters['prop_mdr2010']).index

                    df.loc[idx_c, 'tb_inf'] = 'latent_mdr_primary'  # change to mdr-tb
                    df.loc[idx_c, 'tb_specific_symptoms'] = 'latent'


            idx = (df.age_years == i) & (df.sex == 'F') & (df.tb_inf == 'uninfected') & df.is_alive

            # FEMALE
            if idx.any():
                # sample from uninfected population using WHO latent prevalence estimates
                fraction_latent_tb = latent_tb_data.loc[
                    (latent_tb_data.sex == 'F') & (latent_tb_data.age == i), 'prob_latent_tb']
                female_latent_tb = df[idx].sample(frac=fraction_latent_tb).index
                df.loc[female_latent_tb, 'tb_inf'] = 'latent_susc_primary'
                df.loc[female_latent_tb, 'tb_date_latent'] = now
                df.loc[female_latent_tb, 'tb_specific_symptoms'] = 'latent'

            # allocate some latent infections as mdr-tb
            if len(df[df.is_alive & (df.sex == 'F') & (df.tb_inf == 'latent_susc_primary')]) > 10:
                idx_c = df[df.is_alive & (df.sex == 'F') & (df.tb_inf == 'latent_susc_primary')].sample(
                    frac=self.parameters['prop_mdr2010']).index

                df.loc[idx_c, 'tb_inf'] = 'latent_mdr_primary'  # change to mdr-tb
                df.loc[idx_c, 'tb_specific_symptoms'] = 'latent'

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ ACTIVE ~~~~~~~~~~~~~~~~~~~~~~~~~~

        # select probability of active infection
        # same for all ages and sex so pick value for M age 0
        frac_active_tb = active_tb_prob_year.loc[
            (active_tb_prob_year.Sex == 'M') & (active_tb_prob_year.ages == 0), 'incidence_per_capita']

        active = df[df.is_alive & (df.tb_inf == 'uninfected')].sample(frac=frac_active_tb).index
        print(active)
        df.loc[active, 'tb_inf'] = 'active_susc_primary'
        df.loc[active, 'tb_date_active'] = now
        df.loc[active, 'tb_specific_symptoms'] = 'active'

        # allocate some active infections as mdr-tb
        if len(active) > 10:
            idx_c = df[df.is_alive & (df.tb_inf == 'active_susc_primary')].sample(
                frac=self.parameters['prop_mdr2010']).index

            df.loc[idx_c, 'tb_inf'] = 'active_mdr_primary'
            df.loc[idx_c, 'tb_specific_symptoms'] = 'active'

    def initialise_simulation(self, sim):
        sim.schedule_event(TbEvent(self), sim.date + DateOffset(months=12))
        sim.schedule_event(TbActiveEvent(self), sim.date + DateOffset(months=12))
        sim.schedule_event(TbSelfCureEvent(self), sim.date + DateOffset(months=12))

        sim.schedule_event(TbMdrEvent(self), sim.date + DateOffset(months=12))
        sim.schedule_event(TbMdrActiveEvent(self), sim.date + DateOffset(months=12))
        sim.schedule_event(TbMdrSelfCureEvent(self), sim.date + DateOffset(months=12))

        sim.schedule_event(TbDeathEvent(self), sim.date + DateOffset(months=12))

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

        # add an event to log to screen
        sim.schedule_event(TbLoggingEvent(self), sim.date + DateOffset(days=0))

        # Register with the HealthSystem the treatment interventions that this module runs
        # and define the footprint that each intervention has on the common resources
        footprint_for_smear_test = pd.DataFrame(index=np.arange(1), data={
            'Name': tb.SPUTUM_TEST_ID,
            'Nurse_Time': 10,
            'Doctor_Time': 5,
            'Electricity': False,
            'Water': False})

        footprint_for_xpert_test = pd.DataFrame(index=np.arange(1), data={
            'Name': tb.XPERT_TEST_ID,
            'Nurse_Time': 10,
            'Doctor_Time': 5,
            'Electricity': True,
            'Water': False})

        footprint_for_treatment = pd.DataFrame(index=np.arange(1), data={
            'Name': tb.TREATMENT_ID,
            'Nurse_Time': 5,
            'Doctor_Time': 10,
            'Electricity': False,
            'Water': False})

        footprint_for_treatment_mdr = pd.DataFrame(index=np.arange(1), data={
            'Name': tb.MDR_TREATMENT_ID,
            'Nurse_Time': 5,
            'Doctor_Time': 10,
            'Electricity': False,
            'Water': False})

        footprint_for_followup = pd.DataFrame(index=np.arange(1), data={
            'Name': tb.FOLLOWUP_ID,
            'Nurse_Time': 15,
            'Doctor_Time': 10,
            'Electricity': True,
            'Water': True})

        self.sim.modules['HealthSystem'].register_interventions(footprint_for_smear_test)
        self.sim.modules['HealthSystem'].register_interventions(footprint_for_xpert_test)
        self.sim.modules['HealthSystem'].register_interventions(footprint_for_treatment)
        self.sim.modules['HealthSystem'].register_interventions(footprint_for_treatment_mdr)
        self.sim.modules['HealthSystem'].register_interventions(footprint_for_followup)

    # TODO: check all properties match list above
    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        """
        df = self.sim.population.props

        df.at[child_id, 'tb_inf'] = 'uninfected'
        df.at[child_id, 'tb_date_active'] = pd.NaT
        df.at[child_id, 'tb_date_latent'] = pd.NaT
        df.at[child_id, 'tb_date_death'] = pd.NaT
        df.at[child_id, 'tb_ever_tb'] = False
        df.at[child_id, 'tb_ever_tb_mdr'] = False
        df.at[child_id, 'tb_specific_symptoms'] = 'none'
        df.at[child_id, 'tb_unified_symptom_code'] = 0

        df.at[child_id, 'tb_ever_tested'] = False  # default: no individuals tested
        df.at[child_id, 'tb_smear_test'] = False
        df.at[child_id, 'tb_result_smear_test'] = False
        df.at[child_id, 'tb_date_smear_test'] = pd.NaT
        df.at[child_id, 'tb_xpert_test'] = False
        df.at[child_id, 'tb_result_xpert_test'] = False
        df.at[child_id, 'tb_date_xpert_test'] = pd.NaT
        df.at[child_id, 'tb_diagnosed'] = False
        df.at[child_id, 'tb_mdr_diagnosed'] = False
        df.at[child_id, 'tb_treated'] = False
        df.at[child_id, 'tb_date_treated'] = pd.NaT
        df.at[child_id, 'tb_treatment_failure'] = False
        df.at[child_id, 'tb_treatedMDR'] = False
        df.at[child_id, 'tb_date_treatedMDR'] = pd.NaT
        df.at[child_id, 'request_mdr_regimen'] = False

    def query_symptoms_now(self):
        """This is called by the health-care seeking module
        All modules refresh the symptomology of persons at this time
        And report it on the unified symptomology scale """

        logger.debug("This is tb, being asked to report unified symptomology")

        # Map the specific symptoms for this disease onto the unified coding scheme
        # this is a measure of the propensity to seek care
        df = self.sim.population.props

        df.loc[df.is_alive, 'tb_unified_symptom_code'] = df.loc[df.is_alive, 'tb_specific_symptoms'].map({
            'none': 0,
            'latent': 1,
            'active': 2
        })

        return df.loc[df.is_alive, 'tb_unified_symptom_code']

    # TODO: complete this
    def on_healthsystem_interaction(self, person_id, cue_type=None, disease_specific=None):
        logger.debug('This is tb, being alerted about a health system interaction '
                     'person %d triggered by %s : %s', person_id, cue_type, disease_specific)

        df = self.sim.population.props

        # ----------------------------------- TB TESTING -----------------------------------
        if (cue_type == 'InitialDiseaseCall' or cue_type == 'HealthCareSeekingPoll') and disease_specific == 'tb':

            # never tested & hiv- : smear sputum
            if not df.at[person_id, 'tb_ever_tested'] and not \
                df.at[person_id, 'hiv_inf'] and df.at[person_id, 'is_alive']:

                gets_test = self.sim.modules['HealthSystem'].query_access_to_service(
                    person_id, self.SPUTUM_TEST_ID)

                if gets_test:
                    event = TbTestingEvent(self, person_id)
                    self.sim.schedule_event(event, self.sim.date)

            # secondary case (relapse / reinfection) : xpert
            if ((df.at[person_id, 'tb_inf'] == 'latent_susc_secondary') or (
                df.at[person_id, 'tb_inf'] == 'latent_mdr_secondary') or (
                    df.at[person_id, 'tb_inf'] == 'active_susc_secondary') or (
                    df.at[person_id, 'tb_inf'] == 'active_mdr_secondary')) and not df.at[person_id, 'hiv_inf'] and \
                df.at[person_id, 'tb_ever_tested'] and df.at[person_id, 'is_alive']:

                gets_test = self.sim.modules['HealthSystem'].query_access_to_service(person_id, self.XPERT_TEST_ID)

                if gets_test:
                    event = TbXpertTest(self, person_id)
                    self.sim.schedule_event(event, self.sim.date)

            # smear sputum test negative, any tb status : Xpert
            if ((df.at[person_id, 'tb_inf'] == 'uninfected') or (
                df.at[person_id, 'tb_inf'] == 'latent_susc_primary') or (
                    df.at[person_id, 'tb_inf'] == 'latent_mdr_primary') or (
                    df.at[person_id, 'tb_inf'] == 'active_susc_primary') or (
                    df.at[person_id, 'tb_inf'] == 'active_mdr_primary')) and not df.at[
                person_id, 'tb_result_smear_test'] and df.at[person_id, 'is_alive']:

                gets_test = self.sim.modules['HealthSystem'].query_access_to_service(
                    person_id, self.XPERT_TEST_ID)

                if gets_test:
                    event = TbXpertTest(self, person_id)
                    self.sim.schedule_event(event, self.sim.date)

            # treatment failure : Xpert
            if df.at[person_id, 'tb_treatment_failure']:

                gets_test = self.sim.modules['HealthSystem'].query_access_to_service(
                    person_id, self.XPERT_TEST_ID)

                if gets_test:
                    event = TbXpertTest(self, person_id)
                    self.sim.schedule_event(event, self.sim.date)

            # Xpert test negative : Xpert repeat test
            if not df.at[person_id, 'tb_result_xpert_test']:

                gets_test = self.sim.modules['HealthSystem'].query_access_to_service(
                    person_id, self.XPERT_TEST_ID)

                if gets_test:
                    event = TbXpertTest(self, person_id)
                    self.sim.schedule_event(event, self.sim.date)

            # hiv+ : Xpert
            if df.at[person_id, 'hiv_inf']:

                gets_test = self.sim.modules['HealthSystem'].query_access_to_service(
                    person_id, self.XPERT_TEST_ID)

                if gets_test:
                    event = TbXpertTest(self, person_id)
                    self.sim.schedule_event(event, self.sim.date)

        # ----------------------------------- TB TREATMENT -----------------------------------
        # cue type initial disease call
        # referral from testing
        if df.at[person_id, 'tb_diagnosed']:

            # if had smear test - unknown susc/mdr
            if not df.at[person_id, 'tb_xpert_test']:
                gets_treatment = self.sim.modules['HealthSystem'].query_access_to_service(
                    person_id, self.TREATMENT_ID)

                if gets_treatment:
                    event = TbTreatmentEvent(self, person_id)
                    self.sim.schedule_event(event, self.sim.date)

            elif df.at[person_id, 'tb_xpert_test']:
                gets_treatment = self.sim.modules['HealthSystem'].query_access_to_service(
                    person_id, self.MDR_TREATMENT_ID_ID)

                if gets_treatment:
                    event = TbTreatmentMdrEvent(self, person_id)
                    self.sim.schedule_event(event, self.sim.date)


        # cue type outreach



    # TODO: complete this
    def on_followup_healthsystem_interaction(self, person_id):
        #     logger.debug('This is a follow-up appointment')

        gets_test = self.sim.modules['HealthSystem'].query_access_to_service(
            person_id, self.XPERT_TEST_ID)

        if gets_test:
            event = TbXpertTest(self, person_id)
            self.sim.schedule_event(event, self.sim.date)

    def report_qaly_values(self):
        # This must send back a dataframe that reports on the HealthStates for all individuals over
        # the past year

        logger.debug('This is tb reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe
        params = self.parameters

        health_values = df.loc[df.is_alive, 'tb_specific_symptoms'].map({
            'none': 0,
            'latent': params['qalywt_latent'],
            'active': params['qalywt_active']
        })

        return health_values.loc[df.is_alive]


# ---------------------------------------------------------------------------
#   TB infection event
# ---------------------------------------------------------------------------

class TbEvent(RegularEvent, PopulationScopeEventMixin):
    """ tb infection events
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # every 1 month

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date
        rng = self.module.rng

        df = population.props

        # ----------------------------------- FORCE OF INFECTION -----------------------------------

        # apply a force of infection to produce new latent cases
        # no age distribution for FOI but the relative risks would affect distribution of active infections
        # remember event is occurring annually so scale rates accordingly
        active_hiv_neg = len(df[((df.tb_inf == 'active_susc_primary') | (
            df.tb_inf == 'active_susc_secondary')) & ~df.hiv_inf & df.is_alive])
        active_hiv_pos = len(df[((df.tb_inf == 'active_susc_primary') | (
            df.tb_inf == 'active_susc_secondary')) & df.hiv_inf & df.is_alive])

        # population at-risk = susceptible (new infection), latent_mdr (new infection)

        uninfected_total = len(df[((df.tb_inf == 'uninfected') | (df.tb_inf == 'latent_mdr_primary ') | (
            df.tb_inf == 'latent_mdr_secondary')) & df.is_alive])
        total_population = len(df[df.is_alive])

        force_of_infection = (params['transmission_rate'] * active_hiv_neg * (active_hiv_pos * params[
            'rel_infectiousness_hiv']) * uninfected_total) / total_population
        # print('force_of_infection: ', force_of_infection)

        # ----------------------------------- NEW INFECTIONS -----------------------------------

        # pop at risk = susceptible and latent_mdr, latent_susc_primary only
        at_risk = df[((df.tb_inf == 'uninfected') | (df.tb_inf == 'latent_mdr_primary') | (
            df.tb_inf == 'latent_mdr_secondary') | (df.tb_inf == 'latent_susc_primary')) & df.is_alive].index

        #  no age/sex effect on risk of latent infection
        prob_tb_new = pd.Series(force_of_infection, index=at_risk)
        # print('prob_tb_new: ', prob_tb_new)

        is_newly_infected = prob_tb_new > rng.rand(len(prob_tb_new))
        new_case = is_newly_infected[is_newly_infected].index
        df.loc[new_case, 'tb_inf'] = 'latent_susc_primary'
        df.loc[new_case, 'tb_date_latent'] = now
        df.loc[new_case, 'tb_specific_symptoms'] = 'latent'

        # ----------------------------------- RE-INFECTIONS -----------------------------------

        # pop at risk = latent_susc_secondary
        at_risk = df[(df.tb_inf == 'latent_susc_secondary') & df.is_alive].index

        #  no age/sex effect on risk of latent infection
        prob_tb_new = pd.Series(force_of_infection, index=at_risk)
        # print('prob_tb_new: ', prob_tb_new)

        is_newly_infected = prob_tb_new > rng.rand(len(prob_tb_new))
        new_case = is_newly_infected[is_newly_infected].index
        df.loc[
            new_case, 'tb_inf'] = 'latent_susc_secondary'  # unchanged status, high risk of relapse again as if just recovered
        df.loc[new_case, 'tb_date_latent'] = now
        df.loc[new_case, 'tb_specific_symptoms'] = 'latent'


class TbActiveEvent(RegularEvent, PopulationScopeEventMixin):
    """ tb progression from latent to active infection
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # every 1 month

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date
        rng = self.module.rng

        df = population.props
        # ----------------------------------- FAST PROGRESSORS TO ACTIVE DISEASE -----------------------------------

        # if any newly infected latent cases, 14% become active directly, only for primary infection
        new_latent = df[(df.tb_inf == 'latent_susc_primary') & (df.tb_date_latent == now) & df.is_alive].sum()
        # print(new_latent)

        if new_latent.any():
            fast_progression = df[
                (df.tb_inf == 'latent_susc_primary') & (df.tb_date_latent == now) & df.is_alive].sample(
                frac=params['prop_fast_progressor']).index
            df.loc[fast_progression, 'tb_inf'] = 'active_susc_primary'
            df.loc[fast_progression, 'tb_date_active'] = now
            df.loc[fast_progression, 'tb_ever_tb'] = True
            df.loc[fast_progression, 'tb_specific_symptoms'] = 'active'

        # ----------------------------------- SLOW PROGRESSORS TO ACTIVE DISEASE -----------------------------------

        # slow progressors with latent TB become active
        # random sample with weights for RR of active disease
        eff_prob_active_tb = pd.Series(0, index=df.index)
        eff_prob_active_tb.loc[(df.tb_inf == 'latent_susc_primary')] = params['monthly_prob_progr_active']
        # print('eff_prob_active_tb: ', eff_prob_active_tb)

        hiv_stage1 = df.index[df.hiv_inf & (df.tb_inf == 'latent_susc_primary') &
                              (((now - df.hiv_date_inf).dt.days / 365.25) < 3.33)]
        # print('hiv_stage1', hiv_stage1)

        hiv_stage2 = df.index[df.hiv_inf & (df.tb_inf == 'latent_susc_primary') &
                              (((now - df.hiv_date_inf).dt.days / 365.25) >= 3.33) &
                              (((now - df.hiv_date_inf).dt.days / 365.25) < 6.67)]
        # print('hiv_stage2', hiv_stage2)

        hiv_stage3 = df.index[df.hiv_inf & (df.tb_inf == 'latent_susc_primary') &
                              (((now - df.hiv_date_inf).dt.days / 365.25) >= 6.67) &
                              (((now - df.hiv_date_inf).dt.days / 365.25) < 10)]
        # print('hiv_stage3', hiv_stage3)

        hiv_stage4 = df.index[df.hiv_inf & (df.tb_inf == 'latent_susc_primary') &
                              (((now - df.hiv_date_inf).dt.days / 365.25) >= 10)]
        # print('hiv_stage4', hiv_stage4)

        eff_prob_active_tb.loc[hiv_stage1] *= params['rr_tb_with_hiv_stages'][0]
        eff_prob_active_tb.loc[hiv_stage2] *= params['rr_tb_with_hiv_stages'][1]
        eff_prob_active_tb.loc[hiv_stage3] *= params['rr_tb_with_hiv_stages'][2]
        eff_prob_active_tb.loc[hiv_stage4] *= params['rr_tb_with_hiv_stages'][3]
        eff_prob_active_tb.loc[df.hiv_on_art == '2'] *= params['rr_tb_art']
        # eff_prob_active_tb.loc[df.is_malnourished] *= params['rr_tb_malnourished']
        # eff_prob_active_tb.loc[df.has_diabetes1] *= params['rr_tb_diabetes1']
        # eff_prob_active_tb.loc[df.high_alcohol] *= params['rr_tb_alcohol']
        # eff_prob_active_tb.loc[df.is_smoker] *= params['rr_tb_smoking']
        # eff_prob_active_tb.loc[df.high_pollution] *= params['rr_tb_pollution']

        prog_to_active = eff_prob_active_tb > rng.rand(len(eff_prob_active_tb))
        # print('prog_to_active: ', prog_to_active )
        new_active_case = prog_to_active[prog_to_active].index
        # print('new_active_case: ', new_active_case)
        df.loc[new_active_case, 'tb_inf'] = 'active_susc_primary'
        df.loc[new_active_case, 'tb_date_active'] = now
        df.loc[new_active_case, 'tb_ever_tb'] = True
        df.loc[new_active_case, 'tb_specific_symptoms'] = 'active'

        # ----------------------------------- RELAPSE -----------------------------------
        random_draw = self.sim.rng.random_sample(size=len(df))

        # relapse after treatment completion, tb_date_treated + six months
        relapse_tx_complete = df[
            (df.tb_inf == 'latent_susc_secondary') & df.is_alive & (
                self.sim.date - df.tb_date_treated > pd.to_timedelta(182.625, unit='d')) & (
                self.sim.date - df.tb_date_treated < pd.to_timedelta(732.5, unit='d')) & ~df.tb_treatment_failure & (
                random_draw < params['monthly_prob_relapse_tx_complete'])].index

        # relapse after treatment default, tb_treatment_failure=True, but make sure not tb-mdr
        relapse_tx_incomplete = df[
            (df.tb_inf == 'latent_susc_secondary') & df.is_alive & df.tb_treatment_failure & (
                self.sim.date - df.tb_date_treated > pd.to_timedelta(182.625, unit='d')) & (
                self.sim.date - df.tb_date_treated < pd.to_timedelta(732.5, unit='d')) & (
                random_draw < params['monthly_prob_relapse_tx_incomplete'])].index

        # relapse after >2 years following completion of treatment (or default)
        # use tb_date_treated + 2 years + 6 months of treatment
        relapse_tx_2yrs = df[
            (df.tb_inf == 'latent_susc_secondary') & df.is_alive & (
                self.sim.date - df.tb_date_treated >= pd.to_timedelta(732.5, unit='d')) & (
                random_draw < params['monthly_prob_relapse_2yrs'])].index

        df.loc[relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, 'tb_inf'] = 'active_susc_secondary'
        df.loc[relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, 'tb_date_active'] = now
        df.loc[relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, 'tb_ever_tb'] = True
        df.loc[relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, 'tb_specific_symptoms'] = 'active'


class TbSelfCureEvent(RegularEvent, PopulationScopeEventMixin):
    """ tb self-cure events
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # every 1 month

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date
        rng = self.module.rng

        df = population.props

        # self-cure - move from active to latent_secondary, make sure it's not the ones that just became active
        random_draw = rng.random_sample(size=len(df))

        self_cure = df[
            ((df.tb_inf == 'active_susc_primary') | (df.tb_inf == 'active_susc_secondary')) & df.is_alive & (
                df.tb_date_active < now) & (random_draw < params['monthly_prob_self_cure'])].index
        df.loc[self_cure, 'tb_inf'] = 'latent_susc_secondary'
        df.loc[self_cure, 'tb_specific_symptoms'] = 'latent'


# ---------------------------------------------------------------------------
#   TB MDR infection event
# ---------------------------------------------------------------------------

class TbMdrEvent(RegularEvent, PopulationScopeEventMixin):
    """ tb-mdr infection events
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # every 1 month

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date
        rng = self.module.rng

        df = population.props

        # ----------------------------------- FORCE OF INFECTION -----------------------------------

        # apply a force of infection to produce new latent cases
        # no age distribution for FOI but the relative risks would affect distribution of active infections
        active_hiv_neg = len(df[((df.tb_inf == 'active_mdr_primary') | (
            df.tb_inf == 'active_mdr_secondary')) & ~df.hiv_inf & df.is_alive])
        active_hiv_pos = len(df[((df.tb_inf == 'active_mdr_primary') | (
            df.tb_inf == 'active_mdr_secondary')) & df.hiv_inf & df.is_alive])

        # population at-risk = susceptible (new infection), latent_susc (new infection)

        uninfected_total = len(df[((df.tb_inf == 'uninfected') | (df.tb_inf == 'latent_susc_primary ') | (
            df.tb_inf == 'latent_susc_secondary')) & df.is_alive])
        total_population = len(df[df.is_alive])

        force_of_infection = (params['transmission_rate'] * active_hiv_neg * (active_hiv_pos * params[
            'rel_infectiousness_hiv']) * uninfected_total) / total_population
        # print('force_of_infection: ', force_of_infection)

        # ----------------------------------- NEW INFECTIONS -----------------------------------

        # pop at risk = susceptible and latent_susc, latent_mdr_primary only
        at_risk = df[((df.tb_inf == 'uninfected') | (df.tb_inf == 'latent_susc_primary') | (
            df.tb_inf == 'latent_susc_secondary') | (df.tb_inf == 'latent_mdr_primary')) & df.is_alive].index

        #  no age/sex effect on risk of latent infection
        prob_tb_new = pd.Series(force_of_infection, index=at_risk)
        # print('prob_tb_new: ', prob_tb_new)

        is_newly_infected = prob_tb_new > rng.rand(len(prob_tb_new))
        new_case = is_newly_infected[is_newly_infected].index
        df.loc[new_case, 'tb_inf'] = 'latent_mdr_primary'
        df.loc[new_case, 'tb_date_latent'] = now
        df.loc[new_case, 'tb_specific_symptoms'] = 'latent'

        # ----------------------------------- RE-INFECTIONS -----------------------------------

        # pop at risk = latent_mdr_secondary
        at_risk = df[(df.tb_inf == 'latent_mdr_secondary') & df.is_alive].index

        #  no age/sex effect on risk of latent infection
        prob_tb_new = pd.Series(force_of_infection, index=at_risk)
        # print('prob_tb_new: ', prob_tb_new)

        is_newly_infected = prob_tb_new > rng.rand(len(prob_tb_new))
        new_case = is_newly_infected[is_newly_infected].index
        df.loc[
            new_case, 'tb_inf'] = 'latent_mdr_secondary'  # unchanged status, high risk of relapse again as if just recovered
        df.loc[new_case, 'tb_date_latent'] = now
        df.loc[new_case, 'tb_specific_symptoms'] = 'latent'


class TbMdrActiveEvent(RegularEvent, PopulationScopeEventMixin):
    """ tb-mdr progression from latent to active infection
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # every 1 month

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date
        rng = self.module.rng

        df = population.props
        # ----------------------------------- FAST PROGRESSORS TO ACTIVE DISEASE -----------------------------------

        # if any newly infected latent cases, 14% become active directly, only for primary infection
        new_latent = df[(df.tb_inf == 'latent_mdr_primary') & (df.tb_date_latent == now) & df.is_alive].sum()
        # print(new_latent)

        if new_latent.any():
            fast_progression = df[
                (df.tb_inf == 'latent_mdr_primary') & (df.tb_date_latent == now) & df.is_alive].sample(
                frac=params['prop_fast_progressor']).index
            df.loc[fast_progression, 'tb_inf'] = 'active_mdr_primary'
            df.loc[fast_progression, 'tb_date_active'] = now
            df.loc[fast_progression, 'tb_ever_tb'] = True
            df.loc[fast_progression, 'tb_specific_symptoms'] = 'active'

        # ----------------------------------- SLOW PROGRESSORS TO ACTIVE DISEASE -----------------------------------

        # slow progressors with latent TB become active
        # random sample with weights for RR of active disease
        eff_prob_active_tb = pd.Series(0, index=df.index)
        eff_prob_active_tb.loc[(df.tb_inf == 'latent_mdr_primary')] = params['monthly_prob_progr_active']
        # print('eff_prob_active_tb: ', eff_prob_active_tb)

        hiv_stage1 = df.index[df.hiv_inf & (df.tb_inf == 'latent_mdr_primary') &
                              (((now - df.hiv_date_inf).dt.days / 365.25) < 3.33)]
        # print('hiv_stage1', hiv_stage1)

        hiv_stage2 = df.index[df.hiv_inf & (df.tb_inf == 'latent_mdr_primary') &
                              (((now - df.hiv_date_inf).dt.days / 365.25) >= 3.33) &
                              (((now - df.hiv_date_inf).dt.days / 365.25) < 6.67)]
        # print('hiv_stage2', hiv_stage2)

        hiv_stage3 = df.index[df.hiv_inf & (df.tb_inf == 'latent_mdr_primary') &
                              (((now - df.hiv_date_inf).dt.days / 365.25) >= 6.67) &
                              (((now - df.hiv_date_inf).dt.days / 365.25) < 10)]
        # print('hiv_stage3', hiv_stage3)

        hiv_stage4 = df.index[df.hiv_inf & (df.tb_inf == 'latent_mdr_primary') &
                              (((now - df.hiv_date_inf).dt.days / 365.25) >= 10)]
        # print('hiv_stage4', hiv_stage4)

        eff_prob_active_tb.loc[hiv_stage1] *= params['rr_tb_with_hiv_stages'][0]
        eff_prob_active_tb.loc[hiv_stage2] *= params['rr_tb_with_hiv_stages'][1]
        eff_prob_active_tb.loc[hiv_stage3] *= params['rr_tb_with_hiv_stages'][2]
        eff_prob_active_tb.loc[hiv_stage4] *= params['rr_tb_with_hiv_stages'][3]
        eff_prob_active_tb.loc[df.hiv_on_art == '2'] *= params['rr_tb_art']
        # eff_prob_active_tb.loc[df.is_malnourished] *= params['rr_tb_malnourished']
        # eff_prob_active_tb.loc[df.has_diabetes1] *= params['rr_tb_diabetes1']
        # eff_prob_active_tb.loc[df.high_alcohol] *= params['rr_tb_alcohol']
        # eff_prob_active_tb.loc[df.is_smoker] *= params['rr_tb_smoking']
        # eff_prob_active_tb.loc[df.high_pollution] *= params['rr_tb_pollution']

        prog_to_active = eff_prob_active_tb > rng.rand(len(eff_prob_active_tb))
        # print('prog_to_active: ', prog_to_active )
        new_active_case = prog_to_active[prog_to_active].index
        # print('new_active_case: ', new_active_case)
        df.loc[new_active_case, 'tb_inf'] = 'active_mdr_primary'
        df.loc[new_active_case, 'tb_date_active'] = now
        df.loc[new_active_case, 'tb_ever_tb'] = True
        df.loc[new_active_case, 'tb_specific_symptoms'] = 'active'

        # ----------------------------------- RELAPSE -----------------------------------
        random_draw = rng.random_sample(size=len(df))

        # relapse after treatment completion, tb_date_treated + six months
        relapse_tx_complete = df[
            (df.tb_inf == 'latent_mdr_secondary') & df.is_alive & (
                self.sim.date - df.tb_date_treated > pd.to_timedelta(182.625, unit='d')) & (
                self.sim.date - df.tb_date_treated < pd.to_timedelta(732.5, unit='d')) & ~df.tb_treatment_failure & (
                random_draw < params['monthly_prob_relapse_tx_complete'])].index

        # relapse after treatment default, tb_treatment_failure=True, but make sure not tb-mdr
        relapse_tx_incomplete = df[
            (df.tb_inf == 'latent_mdr_secondary') & df.is_alive & df.tb_treatment_failure & (
                self.sim.date - df.tb_date_treated > pd.to_timedelta(182.625, unit='d')) & (
                self.sim.date - df.tb_date_treated < pd.to_timedelta(732.5, unit='d')) & (
                random_draw < params['monthly_prob_relapse_tx_incomplete'])].index

        # relapse after >2 years following completion of treatment (or default)
        # use tb_date_treated + 2 years + 6 months of treatment
        relapse_tx_2yrs = df[
            (df.tb_inf == 'latent_mdr_secondary') & df.is_alive & (
                self.sim.date - df.tb_date_treated >= pd.to_timedelta(732.5, unit='d')) & (
                random_draw < params['monthly_prob_relapse_2yrs'])].index

        df.loc[relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, 'tb_inf'] = 'active_mdr_secondary'
        df.loc[relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, 'tb_date_active'] = now
        df.loc[relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, 'tb_ever_tb'] = True
        df.loc[relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, 'tb_specific_symptoms'] = 'active'


class TbMdrSelfCureEvent(RegularEvent, PopulationScopeEventMixin):
    """ tb-mdr self-cure events
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # every 1 month

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date
        rng = self.module.rng

        df = population.props

        # self-cure - move from active to latent_secondary, make sure it's not the ones that just became active
        random_draw = rng.random_sample(size=len(df))

        self_cure = df[
            ((df.tb_inf == 'active_mdr_primary') | (df.tb_inf == 'active_mdr_secondary')) & df.is_alive & (
                df.tb_date_active < now) & (random_draw < params['monthly_prob_self_cure'])].index
        df.loc[self_cure, 'tb_inf'] = 'latent_mdr_secondary'
        df.loc[self_cure, 'tb_specific_symptoms'] = 'latent'


# ---------------------------------------------------------------------------
#   Testing
# ---------------------------------------------------------------------------

class TbTestingEvent(Event, IndividualScopeEventMixin):
    """
    Testing algorithm for smear sputum tests only
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        params = self.module.parameters
        df = self.sim.population.props
        now = self.sim.date

        df.at[person_id, 'tb_ever_tested'] = True
        df.at[person_id, 'tb_smear_test'] = True
        df.at[person_id, 'tb_date_smear_test'] = now
        df.at[person_id, 'tb_result_smear_test'] = False
        df.at[person_id, 'tb_diagnosed'] = False  # default

        # ----------------------------------- OUTCOME OF TEST -----------------------------------

        # hiv+
        if (df.at[person_id, 'tb_inf'] == 'active_susc') or (df.at[person_id, 'tb_inf'] == 'active_mdr') and df.at[
            person_id, 'hiv_inf']:
            diagnosed = self.sim.rng.choice([True, False], size=1, p=[params['prop_smear_positive'],
                                                                      (1 - params['prop_smear_positive'])])
            if diagnosed:
                df.at[person_id, 'tb_result_smear_test'] = True
                df.at[person_id, 'tb_diagnosed'] = True

        # hiv-, 80% of smear tests will be negative - extrapulmonary
        elif (df.at[person_id, 'tb_inf'] == 'active_susc') or (df.at[person_id, 'tb_inf'] == 'active_mdr') and not \
            df.at[person_id, 'hiv_inf']:
            diagnosed = self.sim.rng.choice([True, False], size=1, p=[params['prop_smear_positive_hiv'],
                                                                      (1 - params['prop_smear_positive_hiv'])])

            if diagnosed:
                df.at[person_id, 'tb_result_smear_test'] = True
                df.at[person_id, 'tb_diagnosed'] = True

        # ----------------------------------- REFERRALS FOR SECONDARY TESTING -----------------------------------
        # remaining 20% of active cases and negative cases referred for xpert testing with some delay
        # schedule xpert testing at future date
        # random draw approx 2 months?

        if not df.at[person_id, 'tb_diagnosed']:
            referral_time = abs(np.random.normal(loc=(2 / 12), scale=(0.5 / 12), size=1))  # in years
            future_test_time = now + pd.to_timedelta(referral_time[0] * 365.25, unit='d')
            # schedule another initial disease call
            repeat_test = healthsystem.HealthSystemInteractionEvent(self.module, person_id,
                                                                    cue_type='InitialDiseaseCall',
                                                                    disease_specific=self.module.name)
            self.sim.schedule_event(repeat_test, future_test_time)

        # ----------------------------------- REFERRALS FOR TREATMENT -----------------------------------
        if df.at[person_id, 'tb_diagnosed']:
            # request treatment
            treatment = healthsystem.HealthSystemInteractionEvent(self.module, person_id,
                                                                  cue_type='InitialDiseaseCall',
                                                                  disease_specific=self.module.name)
            self.sim.schedule_event(treatment, now)


class TbXpertTest(Event, IndividualScopeEventMixin):
    """ Xpert test for people with negative smear result
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        now = self.sim.date

        df.at[person_id, 'tb_ever_tested'] = True
        df.at[person_id, 'tb_xpert_test'] = True
        df.at[person_id, 'tb_date_xpert_test'] = now
        df.at[person_id, 'tb_result_xpert_test'] = False
        df.at[person_id, 'tb_diagnosed_mdr'] = False  # default

        # a further 10% fail to be diagnosed with Xpert (smear-negative + sensitivity of test)
        # they will present back to the health system with some (delay 2-4 weeks)
        if (df.at[person_id, 'tb_inf'] == 'active_susc') or (df.at[person_id, 'tb_inf'] == 'active_mdr'):

            diagnosed = self.sim.rng.choice([True, False], size=1, p=[0.9, 0.1])
            if diagnosed:
                df.at[person_id, 'tb_result_xpert_test'] = True
                df.at[person_id, 'tb_diagnosed'] = True

                if df.at[person_id, 'tb_inf'] == 'active_mdr':
                    df.at[person_id, 'tb_diagnosed_mdr'] = True

            else:
                delay = abs(np.random.normal(loc=4, scale=1, size=1))  # in weeks
                future_test_time = now + pd.to_timedelta(delay[0] * 7, unit='d')
                # schedule another initial disease call
                repeat_test = healthsystem.HealthSystemInteractionEvent(self.module, person_id,
                                                                        cue_type='InitialDiseaseCall',
                                                                        disease_specific=self.module.name)
                self.sim.schedule_event(repeat_test, future_test_time)

        # ----------------------------------- REFERRALS FOR TREATMENT -----------------------------------
        if df.at[person_id, 'tb_diagnosed']:
            # request treatment
            treatment = healthsystem.HealthSystemInteractionEvent(self.module, person_id,
                                                                  cue_type='InitialDiseaseCall',
                                                                  disease_specific=self.module.name)
            self.sim.schedule_event(treatment, now)


# ---------------------------------------------------------------------------
#   Treatment
# ---------------------------------------------------------------------------

class TbTreatmentEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("We are now ready to treat this person", person_id)

        params = self.module.parameters
        now = self.sim.date
        df = self.sim.population.props

        # treatment allocated
        if df.at[person_id, 'is_alive'] and df.at[person_id, 'tb_diagnosed']:
            df.at[person_id, 'tb_treated'] = True
            df.at[person_id, 'date_tb_treated'] = now

        # schedule a 6-month event where people are cured, symptoms return to latent or not cured
        self.sim.schedule_event(TbCureEvent(self, person_id), self.sim.date + DateOffset(months=6))


#
#
# # TODO: complete this
# # TODO: make this individual event
# class TbTreatmentMdrEvent(RegularEvent, PopulationScopeEventMixin):
#
#     def __init__(self, module):
#         super().__init__(module, frequency=DateOffset(months=1))
#
#     def apply(self, population):
#         params = self.module.parameters
#         now = self.sim.date
#         df = population.props
#
#         # get a list of random numbers between 0 and 1 for the whole population
#         random_draw = self.sim.rng.random_sample(size=len(df))
#
#         # probability of mdr treatment
#         mdr_treated_idx = df.index[
#             (random_draw < params['prob_tb_mdr_treatment']) & ~df.request_mdr_regimen & df.is_alive]
#
#         df.loc[mdr_treated_idx, 'request_mdr_regimen'] = False  # switch off this flag
#         df.loc[mdr_treated_idx, 'tb_treatedMDR'] = True
#         df.loc[mdr_treated_idx, 'tb_date_treatedMDR'] = now
#
#         # if on treatment for 6 months, take off and change to cured (100%)
#         random_draw = self.sim.rng.random_sample(size=len(df))
#         cure_idx = df.index[
#             df.tb_treatedMDR & (((now - df.tb_date_treatedMDR) / np.timedelta64(1, 'M')) >= 6)]
#         df.loc[cure_idx, 'tb_treated'] = False
#         df.loc[cure_idx, 'tb_inf'] = 'latent_susc'
#
#

# ---------------------------------------------------------------------------
#   Cure
# ---------------------------------------------------------------------------

class TbCureEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("Stopping tb treatment and curing person", person_id)

        df = self.sim.population.props

        # after six months of treatment, stop
        df.at[person_id, 'tb_treated'] = False

        # if drug-susceptible then probability of successful treatment
        if df.at[person_id, 'tb_inf'] == 'active_susc':

            cured = self.sim.random_sample(size=1) < self.module.parameters['prob_treatment_success']

            if cured:
                df.at[person_id, 'tb_inf'] = 'latent_susc'
                df.at[person_id, 'tb_diagnosed'] = False

            else:
                # request a repeat / Xpert test - follow-up
                # this will include drug-susceptible treatment failures and mdr-tb cases
                followup_appt = healthsystem.HealthSystemInteractionEvent(self.module, person_id, cue_type='FollowUp',
                                                                          disease_specific=self.module.name)
                self.sim.schedule_event(followup_appt, self.sim.date)  # TODO: include a delay here


# # ---------------------------------------------------------------------------
# #   IPT
# # ---------------------------------------------------------------------------
# # TODO: complete this
# # TODO: make this individual event
# class TbIptEvent(RegularEvent, PopulationScopeEventMixin):
#     """ IPT to all paediatric contacts of a TB case - randomly select 5 children <5 yrs old
#     """
#
#     def __init__(self, module):
#         super().__init__(module, frequency=DateOffset(months=1))
#
#     def apply(self, population):
#         params = self.module.parameters
#         now = self.sim.date
#         df = population.props
#
#         #  sum number of active TB cases * 5
#         ipt_needed = len(df.index[df.tb_inf & df.is_alive & ~df.tb_treated]) * 5
#
#         # randomly sample from <5 yr olds
#         ipt_sample = df[(df.age_years <= 5) & (~df.tb_inf == 'active_susc')].sample(
#             n=ipt_needed, replace=False).index
#
#         df.loc[ipt_sample, 'on_ipt'] = True
#         df.loc[ipt_sample, 'date_ipt'] = now
#
#         # TODO: ending ipt
#
#
# # TODO: complete this
# # TODO: make this individual event
# class TbExpandedIptEvent(RegularEvent, PopulationScopeEventMixin):
#     """ IPT to all adults and adolescents with HIV
#     """
#
#     def __init__(self, module):
#         super().__init__(module, frequency=DateOffset(months=1))
#
#     def apply(self, population):
#         params = self.module.parameters
#         now = self.sim.date
#         df = population.props
#
#         # randomly sample from >=15 yrs with HIV
#         ipt_sample = df[(df.age_years >= 15) & (~df.hiv_inf)].sample(
#             frac=0.5, replace=False).index
#
#         df.loc[ipt_sample, 'on_ipt'] = True
#         df.loc[ipt_sample, 'date_ipt'] = now
#
#         # TODO: ending ipt
#
#
# ---------------------------------------------------------------------------
#   Deaths
# ---------------------------------------------------------------------------
class TbDeathEvent(RegularEvent, PopulationScopeEventMixin):
    """The regular event that kills people.
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))

    def apply(self, population):
        params = self.module.parameters
        df = population.props
        now = self.sim.date
        rng = self.module.rng

        # only active infections result in death, no deaths on treatment
        mortality_rate = pd.Series(0, index=df.index)
        mortality_rate.loc[((df.tb_inf == 'active_susc_primary') | (df.tb_inf == 'active_mdr_primary') | (
            df.tb_inf == 'active_susc_secondary') | (df.tb_inf == 'active_mdr_secondary')) & ~df.hiv_inf & (
                               ~df.tb_treated | ~df.tb_treated_mdr)] = params[
            'monthly_prob_tb_mortality']
        mortality_rate.loc[((df.tb_inf == 'active_susc_primary') | (df.tb_inf == 'active_mdr_primary') | (
            df.tb_inf == 'active_susc_secondary') | (df.tb_inf == 'active_mdr_secondary')) & df.hiv_inf & (
                               ~df.tb_treated | ~df.tb_treated_mdr)] = params[
            'monthly_prob_tb_mortality_hiv']
        # print('mort_rate: ', mortality_rate)

        # Generate a series of random numbers, one per individual
        probs = rng.rand(len(df))
        deaths = df.is_alive & (probs < mortality_rate)
        # print('deaths: ', deaths)
        will_die = (df[deaths]).index
        # print('will_die: ', will_die)

        for person in will_die:
            if df.at[person, 'is_alive']:
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id=person, cause='tb'),
                                        now)
                df.at[person, 'tb_date_death'] = now


# ---------------------------------------------------------------------------
#   Logging
# ---------------------------------------------------------------------------

class TbLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """ produce some outputs to check
        """
        # run this event every 12 months
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props
        now = self.sim.date

        active_susc = len(
            df[((df.tb_inf == 'active_susc_primary') | (df.tb_inf == 'active_susc_secondary')) & df.is_alive])

        active_mdr = len(
            df[((df.tb_inf == 'active_mdr_primary') | (df.tb_inf == 'active_mdr_secondary')) & df.is_alive])

        active_total = active_susc + active_mdr

        ad_prev = len(df[((df.tb_inf == 'active_susc_primary') | (df.tb_inf == 'active_susc_secondary') | (
            df.tb_inf == 'active_mdr_primary') | (df.tb_inf == 'active_mdr_secondary')) & df.is_alive & (
                             df.age_years >= 15)]) / len(df[df.is_alive & (df.age_years >= 15)])

        child_prev = len(df[((df.tb_inf == 'active_susc_primary') | (df.tb_inf == 'active_susc_secondary') | (
            df.tb_inf == 'active_mdr_primary') | (df.tb_inf == 'active_mdr_secondary')) & df.is_alive & (
                                df.age_years < 15)]) / len(df[df.is_alive & (df.age_years < 15)])

        ad_prev_latent = len(df[((df.tb_inf == 'latent_susc_primary') | (df.tb_inf == 'latent_susc_secondary') | (
            df.tb_inf == 'latent_mdr_primary') | (df.tb_inf == 'latent_mdr_secondary')) & df.is_alive & (
                                    df.age_years >= 15)]) / len(df[df.is_alive & (df.age_years >= 15)])

        child_prev_latent = len(df[((df.tb_inf == 'latent_susc_primary') | (df.tb_inf == 'latent_susc_secondary') | (
            df.tb_inf == 'latent_mdr_primary') | (df.tb_inf == 'latent_mdr_secondary')) & df.is_alive & (
                                       df.age_years < 15)]) / len(df[df.is_alive & (df.age_years < 15)])

        logger.info('%s|summary|%s', now,
                    {
                        'tbTotalInf': active_total,
                        'tbActiveSusc': active_susc,
                        'tbActiveMdr': active_mdr,
                        'tbLatentAdultPrev': ad_prev_latent,
                        'tbLatentChildPrev': child_prev_latent,
                        'tbAdultActivePrev': ad_prev,
                        'tbChildActivePrev': child_prev,
                    })
