"""
TB infections
"""

import logging
import os

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class tb(Module):
    """ Set up the baseline population with TB prevalence
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    PARAMETERS = {
        # baseline population
        'prop_latent_2010': Parameter(Types.REAL,
                                      'Proportion of population with latent tb in 2010'),
        'prop_active_2010': Parameter(Types.REAL,
                                      'Proportion of population with active tb in 2010'),
        'prop_mdr2010': Parameter(Types.REAL,
                                  'Proportion of active tb cases with multidrug resistance in 2010'),

        # natural history
        'transmission_rate': Parameter(Types.REAL, 'TB transmission rate, calibrated'),
        'rel_inf_hiv': Parameter(Types.REAL, 'relative infectiousness of tb in hiv+ compared with hiv-'),
        'rel_inf_smear_ng': Parameter(Types.REAL, 'relative infectiousness of tb in hiv+ compared with hiv-'),
        'rr_bcg_inf': Parameter(Types.REAL, 'relative risk of tb infection with bcg vaccination'),
        'monthly_prob_relapse_tx_complete': Parameter(Types.REAL,
                                                      'annual probability of relapse once treatment complete'),
        'monthly_prob_relapse_tx_incomplete': Parameter(Types.REAL,
                                                        'annual probability of relapse if treatment incomplete'),
        'monthly_prob_relapse_2yrs': Parameter(Types.REAL,
                                               'annual probability of relapse 2 years after treatment complete'),

        # progression
        'prop_fast_progressor': Parameter(Types.REAL,
                                          'Proportion of infections that progress directly to active stage'),
        'prog_active': Parameter(Types.REAL,
                                 'risk of progressing to active tb within two years'),
        'prog_1yr': Parameter(Types.REAL,
                              'proportion children aged <1 year progressing to active disease'),
        'prog_1_2yr': Parameter(Types.REAL,
                                'proportion children aged 1-2 year2 progressing to active disease'),
        'prog_2_5yr': Parameter(Types.REAL,
                                'proportion children aged 2-5 years progressing to active disease'),
        'prog_5_10yr': Parameter(Types.REAL,
                                 'proportion children aged 5-10 years progressing to active disease'),
        'prog_10yr': Parameter(Types.REAL,
                               'proportion children aged 10-15 years progressing to active disease'),
        'monthly_prob_self_cure': Parameter(Types.REAL, 'annual probability of self-cure'),

        # clinical features
        'pulm_tb': Parameter(Types.REAL, 'probability of pulmonary tb'),
        'prop_smear_positive': Parameter(Types.REAL, 'proportion of new active cases that will be smear-positive'),

        # mortality
        'monthly_prob_tb_mortality': Parameter(Types.REAL, 'mortality rate with active tb'),
        'monthly_prob_tb_mortality_hiv': Parameter(Types.REAL, 'mortality from tb with concurrent HIV'),
        'mortality_cotrim': Parameter(Types.REAL, 'mreduction in mortality rates due to cotrimoxazole prophylaxis'),

        # relative risks of progression to active disease
        'rr_tb_bcg': Parameter(Types.REAL,
                               'relative risk of progression to active disease for children with BCG vaccine'),
        'rr_tb_hiv': Parameter(Types.REAL, 'relative risk of progression to active disease for PLHIV'),
        'rr_tb_art_adult': Parameter(Types.REAL,
                                     'relative risk of progression to active disease for adults with HIV on ART'),
        'rr_tb_art_child': Parameter(Types.REAL,
                                     'relative risk of progression to active disease for adults with HIV on ART'),
        'rr_tb_overweight': Parameter(Types.REAL, 'relative risk of progression to active disease if overweight'),
        'rr_tb_obese': Parameter(Types.REAL, 'relative risk of progression to active disease if obese'),
        'rr_tb_diabetes1': Parameter(Types.REAL, 'relative risk of progression to active disease with type 1 diabetes'),
        'rr_tb_alcohol': Parameter(Types.REAL, 'relative risk of progression to active disease with heavy alcohol use'),
        'rr_tb_smoking': Parameter(Types.REAL, 'relative risk of progression to active disease with smoking'),

        'dur_prot_ipt': Parameter(Types.REAL, 'duration in days of protection conferred by IPT against active TB'),
        'dur_prot_ipt_infant': Parameter(Types.REAL,
                                         'duration in days of protection conferred by IPT against active TB in infants'),
        'rr_ipt_adult': Parameter(Types.REAL, 'relative risk of active TB with IPT in adults'),
        'rr_ipt_child': Parameter(Types.REAL, 'relative risk of active TB with IPT in children'),
        'rr_ipt_adult_hiv': Parameter(Types.REAL, 'relative risk of active TB with IPT in adults with hiv'),
        'rr_ipt_child_hiv': Parameter(Types.REAL, 'relative risk of active TB with IPT in children with hiv'),
        'rr_ipt_art_adult': Parameter(Types.REAL, 'relative risk of active TB with IPT and ART in adults'),
        'rr_ipt_art_child': Parameter(Types.REAL, 'relative risk of active TB with IPT and ART in children'),

        # health system interactions
        'prop_fail_xpert': Parameter(Types.REAL, 'proportion of active TB cases not detected with Xpert'),
        'prob_tx_success_new': Parameter(Types.REAL, 'Probability of treatment success for new TB cases'),
        'prob_tx_success_prev': Parameter(Types.REAL, 'Probability of treatment success for previously treated cases'),
        'prob_tx_success_hiv': Parameter(Types.REAL, 'Probability of treatment success for PLHIV'),
        'prob_tx_success_mdr': Parameter(Types.REAL, 'Probability of treatment success for MDR-TB cases'),
        'prob_tx_success_extra': Parameter(Types.REAL, 'Probability of treatment success for extrapulmonary TB cases'),
        'prob_tx_success_0_14': Parameter(Types.REAL, 'Probability of treatment success for children aged 0-14 years'),
        'prob_tx_success_5_14': Parameter(Types.REAL, 'Probability of treatment success for children aged 5-14 years'),
        'followup_times': Parameter(Types.INT, 'times(weeks) tb treatment monitoring required after tx start'),

        # daly weights, no daly weight for latent tb
        'daly_wt_susc_tb':
            Parameter(Types.REAL, 'Drug-susecptible tuberculosis, not HIV infected'),
        'daly_wt_resistant_tb':
            Parameter(Types.REAL, 'multidrug-resistant tuberculosis, not HIV infected'),
        'daly_wt_susc_tb_hiv_severe_anaemia':
            Parameter(Types.REAL, '# Drug-susecptible Tuberculosis, HIV infected and anemia, severe'),
        'daly_wt_susc_tb_hiv_moderate_anaemia':
            Parameter(Types.REAL, 'Drug-susecptible Tuberculosis, HIV infected and anemia, moderate'),
        'daly_wt_susc_tb_hiv_mild_anaemia':
            Parameter(Types.REAL, 'Drug-susecptible Tuberculosis, HIV infected and anemia, mild'),
        'daly_wt_susc_tb_hiv':
            Parameter(Types.REAL, 'Drug-susecptible Tuberculosis, HIV infected'),
        'daly_wt_resistant_tb_hiv_severe_anaemia':
            Parameter(Types.REAL, 'Multidrug resistant Tuberculosis, HIV infected and anemia, severe'),
        'daly_wt_resistant_tb_hiv':
            Parameter(Types.REAL, 'Multidrug resistant Tuberculosis, HIV infected'),
        'daly_wt_resistant_tb_hiv_moderate_anaemia':
            Parameter(Types.REAL, 'Multidrug resistant Tuberculosis, HIV infected and anemia, moderate'),
        'daly_wt_resistant_tb_hiv_mild_anaemia':
            Parameter(Types.REAL, 'Multidrug resistant Tuberculosis, HIV infected and anemia, mild'),
    }

    PROPERTIES = {
        'tb_inf': Property(Types.CATEGORICAL,
                           categories=['uninfected',
                                       'latent_susc_new', 'active_susc_new',
                                       'latent_susc_tx', 'active_susc_tx',
                                       'latent_mdr_new', 'active_mdr_new',
                                       'latent_mdr_tx', 'active_mdr_tx'],
                           description='tb status'),
        'tb_date_active': Property(Types.DATE, 'Date active tb started'),
        'tb_date_latent': Property(Types.DATE, 'Date acquired tb infection (latent stage)'),
        'tb_ever_tb': Property(Types.BOOL, 'if ever had active drug-susceptible tb'),
        'tb_ever_tb_mdr': Property(Types.BOOL, 'if ever had active multi-drug resistant tb'),
        'tb_specific_symptoms': Property(Types.CATEGORICAL, 'Level of symptoms for tb',
                                         categories=['none', 'latent', 'active_pulm', 'active_extra']),
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
        'tb_on_treatment': Property(Types.BOOL, 'on tb treatment regimen'),
        'tb_date_treated': Property(Types.DATE, 'date tb treatment started'),
        'tb_treatment_failure': Property(Types.BOOL, 'failed first line tb treatment'),
        'tb_treated_mdr': Property(Types.BOOL, 'on tb treatment MDR regimen'),
        'tb_date_treated_mdr': Property(Types.DATE, 'date tb MDR treatment started'),
        'tb_on_ipt': Property(Types.BOOL, 'if currently on ipt'),
        'tb_date_ipt': Property(Types.DATE, 'date ipt started')
    }

    def read_parameters(self, data_folder):

        workbook = pd.read_excel(os.path.join(self.resourcefilepath,
                                              'ResourceFile_TB.xlsx'), sheet_name=None)

        params = self.parameters
        params['param_list'] = workbook['parameters']
        self.param_list.set_index("parameter", inplace=True)

        # baseline
        params['prop_active_2010'], params['prop_latent_2010'] = workbook['ProportionActiveTb'], workbook[
            'Latent_TB_prob']
        params['prop_mdr2010'] = self.param_list.loc['prop_mdr2010', 'value1']

        # natural history
        params['transmission_rate'] = self.param_list.loc['transmission_rate', 'value1']
        params['rel_inf_hiv'] = self.param_list.loc['rel_inf_hiv', 'value1']
        params['rel_inf_smear_ng'] = self.param_list.loc['rel_inf_smear_ng', 'value1']
        params['rr_bcg_inf'] = self.param_list.loc['rr_bcg_inf', 'value1']
        params['monthly_prob_relapse_tx_complete'] = self.param_list.loc['monthly_prob_relapse_tx_complete', 'value1']
        params['monthly_prob_relapse_tx_incomplete'] = self.param_list.loc[
            'monthly_prob_relapse_tx_incomplete', 'value1']
        params['monthly_prob_relapse_2yrs'] = self.param_list.loc['monthly_prob_relapse_2yrs', 'value1']

        # progression
        params['prop_fast_progressor'] = self.param_list.loc['prop_fast_progressor', 'value1']
        params['prog_active'] = self.param_list.loc['prog_active', 'value1']
        params['prog_1yr'] = self.param_list.loc['progr_1yr', 'value1']
        params['prog_1_2yr'] = self.param_list.loc['progr_1_2yr', 'value1']
        params['prog_2_5yr'] = self.param_list.loc['progr_2_5yr', 'value1']
        params['prog_5_10yr'] = self.param_list.loc['progr_5_10yr', 'value1']
        params['prog_10yr'] = self.param_list.loc['progr_10yr', 'value1']
        params['monthly_prob_self_cure'] = self.param_list.loc['monthly_prob_self_cure', 'value1']

        # clinical features
        params['pulm_tb'] = workbook['pulm_tb']
        params['prop_smear_positive'] = self.param_list.loc['prop_smear_positive', 'value1']

        # mortality
        params['monthly_prob_tb_mortality'] = self.param_list.loc['monthly_prob_tb_mortality', 'value1']
        params['monthly_prob_tb_mortality_hiv'] = self.param_list.loc['monthly_prob_tb_mortality_hiv', 'value1']
        params['mortality_cotrim'] = self.param_list.loc['mortality_cotrim', 'value1']

        # relative risks of progression to active disease
        params['rr_tb_bcg'] = self.param_list.loc['rr_tb_bcg', 'value1']
        params['rr_tb_hiv'] = self.param_list.loc['rr_tb_hiv', 'value1']
        params['rr_tb_art_adult'] = self.param_list.loc['rr_tb_art_adult', 'value1']
        params['rr_tb_art_child'] = self.param_list.loc['rr_tb_art_child', 'value1']
        params['rr_tb_overweight'] = self.param_list.loc['rr_tb_overweight', 'value1']
        params['rr_tb_obese'] = self.param_list.loc['rr_tb_obese', 'value1']
        params['rr_tb_diabetes1'] = self.param_list.loc['rr_tb_diabetes1', 'value1']
        params['rr_tb_alcohol'] = self.param_list.loc['rr_tb_alcohol', 'value1']
        params['rr_tb_smoking'] = self.param_list.loc['rr_tb_smoking', 'value1']

        params['dur_prot_ipt'] = self.param_list.loc['dur_prot_ipt', 'value1']
        params['dur_prot_ipt_infant'] = self.param_list.loc['dur_prot_ipt_infant', 'value1']
        params['rr_ipt_adult'] = self.param_list.loc['rr_ipt_adult', 'value1']
        params['rr_ipt_child'] = self.param_list.loc['rr_ipt_child', 'value1']
        params['rr_ipt_adult_hiv'] = self.param_list.loc['rr_ipt_adult_hiv', 'value1']
        params['rr_ipt_child_hiv'] = self.param_list.loc['rr_ipt_child_hiv', 'value1']
        params['rr_ipt_art_adult'] = self.param_list.loc['rr_ipt_art_adult', 'value1']
        params['rr_ipt_art_child'] = self.param_list.loc['rr_ipt_art_child', 'value1']

        # health system interactions
        params['prop_fail_xpert'] = self.param_list.loc['prop_fail_xpert', 'value1']
        params['prob_tx_success_new'] = self.param_list.loc['prob_tx_success_new', 'value1']
        params['prob_tx_success_prev'] = self.param_list.loc['prob_tx_success_prev', 'value1']
        params['prob_tx_success_hiv'] = self.param_list.loc['prob_tx_success_hiv', 'value1']
        params['prob_tx_success_mdr'] = self.param_list.loc['prob_tx_success_mdr', 'value1']
        params['prob_tx_success_extra'] = self.param_list.loc['prob_tx_success_extra', 'value1']
        params['prob_tx_success_0_14'] = self.param_list.loc['prob_tx_success_0_14', 'value1']
        params['prob_tx_success_5_14'] = self.param_list.loc['prob_tx_success_5_14', 'value1']
        params['followup_times'] = workbook['followup']

        # get the DALY weight that this module will use from the weight database
        if 'HealthBurden' in self.sim.modules.keys():
            params['daly_wt_susc_tb'] = self.sim.modules['HealthBurden'].get_daly_weight(
                0)  # Drug-susecptible tuberculosis, not HIV infected
            params['daly_wt_resistant_tb'] = self.sim.modules['HealthBurden'].get_daly_weight(
                1)  # multidrug-resistant tuberculosis, not HIV infected
            params['daly_wt_susc_tb_hiv_severe_anaemia'] = self.sim.modules['HealthBurden'].get_daly_weight(
                4)  # Drug-susecptible Tuberculosis, HIV infected and anemia, severe
            params['daly_wt_susc_tb_hiv_moderate_anaemia'] = self.sim.modules['HealthBurden'].get_daly_weight(
                5)  # Drug-susecptible Tuberculosis, HIV infected and anemia, moderate
            params['daly_wt_susc_tb_hiv_mild_anaemia'] = self.sim.modules['HealthBurden'].get_daly_weight(
                6)  # Drug-susecptible Tuberculosis, HIV infected and anemia, mild
            params['daly_wt_susc_tb_hiv'] = self.sim.modules['HealthBurden'].get_daly_weight(
                7)  # Drug-susecptible Tuberculosis, HIV infected
            params['daly_wt_resistant_tb_hiv_severe_anaemia'] = self.sim.modules['HealthBurden'].get_daly_weight(
                8)  # Multidrug resistant Tuberculosis, HIV infected and anemia, severe
            params['daly_wt_resistant_tb_hiv'] = self.sim.modules['HealthBurden'].get_daly_weight(
                9)  # Multidrug resistant Tuberculosis, HIV infected
            params['daly_wt_resistant_tb_hiv_moderate_anaemia'] = self.sim.modules['HealthBurden'].get_daly_weight(
                10)  # Multidrug resistant Tuberculosis, HIV infected and anemia, moderate
            params['daly_wt_resistant_tb_hiv_mild_anaemia'] = self.sim.modules['HealthBurden'].get_daly_weight(
                11)  # Multidrug resistant Tuberculosis, HIV infected and anemia, mild

    def initialise_population(self, population):
        """Set our property values for the initial population.
        """
        df = population.props
        now = self.sim.date

        # set-up baseline population
        df['tb_inf'].values[:] = 'uninfected'
        df['tb_date_active'] = pd.NaT
        df['tb_date_latent'] = pd.NaT

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
        df['tb_on_treatment'] = False
        df['tb_date_treated'] = pd.NaT
        df['tb_treatment_failure'] = False
        df['tb_treated_mdr'] = False
        df['tb_date_treated_mdr'] = pd.NaT
        df['tb_request_mdr_regimen'] = False
        df['tb_on_ipt'] = False
        df['tb_date_ipt'] = pd.NaT

        # TB infections - active / latent
        # baseline infections not weighted by RR, randomly assigned
        active_tb_data = self.parameters['prop_active_2010']
        latent_tb_data = self.parameters['prop_latent_2010']

        active_tb_prob_year = active_tb_data.loc[
            active_tb_data.Year == now.year, ['Age', 'Sex', 'Proportion_pop_with_active_tb']]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ LATENT ~~~~~~~~~~~~~~~~~~~~~~~~~~

        # merge all susceptible individuals with their hiv probability based on sex and age
        df_tbprob = df.merge(latent_tb_data, left_on=['age_years', 'sex'],
                             right_on=['age', 'sex'],
                             how='left')

        # fill missing values with 0 (only relevant for age 80+)
        df_tbprob['prob_latent_tb'] = df_tbprob['prob_latent_tb'].fillna(0)

        assert df_tbprob.prob_latent_tb.isna().sum() == 0  # check there is a probability for every individual

        # get a list of random numbers between 0 and 1 for each infected individual
        random_draw = self.rng.random_sample(size=len(df_tbprob))

        tb_idx = df_tbprob.index[df.is_alive & (df_tbprob.prob_latent_tb > random_draw)]
        df.loc[tb_idx, 'tb_inf'] = 'latent_susc_new'
        df.loc[tb_idx, 'tb_date_latent'] = now
        df.loc[tb_idx, 'tb_specific_symptoms'] = 'latent'
        df.loc[tb_idx, 'tb_unified_symptom_code'] = 0

        # allocate some latent infections as mdr-tb
        if len(df[df.is_alive & (df.tb_inf == 'latent_susc_new')]) > 10:
            idx_c = df[df.is_alive & (df.tb_inf == 'latent_susc_new')].sample(
                frac=self.parameters['prop_mdr2010']).index

            df.loc[idx_c, 'tb_inf'] = 'latent_mdr_new'  # change to mdr-tb
            df.loc[idx_c, 'tb_specific_symptoms'] = 'latent'

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ ACTIVE ~~~~~~~~~~~~~~~~~~~~~~~~~~

        # select probability of active infection
        # same for all ages and sex so pick value for M age 0
        frac_active_tb = active_tb_prob_year.loc[
            (active_tb_prob_year.Sex == 'M') & (active_tb_prob_year.Age == 0), 'Proportion_pop_with_active_tb']

        active = df[df.is_alive & (df.tb_inf == 'uninfected')].sample(frac=frac_active_tb).index
        # print(active)
        df.loc[active, 'tb_inf'] = 'active_susc_new'
        df.loc[active, 'tb_date_active'] = now
        df.loc[active, 'tb_specific_symptoms'] = 'active_pulm'
        df.loc[tb_idx, 'tb_unified_symptom_code'] = 2

        # allocate some active infections as mdr-tb
        if len(active) > 10:
            idx_c = df[df.is_alive & (df.tb_inf == 'active_susc_new')].sample(
                frac=self.parameters['prop_mdr2010']).index

            df.loc[idx_c, 'tb_inf'] = 'active_mdr_new'
            df.loc[idx_c, 'tb_specific_symptoms'] = 'active_pulm'

    def initialise_simulation(self, sim):

        sim.schedule_event(TbEvent(self), sim.date + DateOffset(months=12))
        sim.schedule_event(TbRelapseEvent(self), sim.date + DateOffset(months=12))
        sim.schedule_event(TbSelfCureEvent(self), sim.date + DateOffset(months=12))

        sim.schedule_event(TbMdrEvent(self), sim.date + DateOffset(months=12))
        sim.schedule_event(TbMdrRelapseEvent(self), sim.date + DateOffset(months=12))
        sim.schedule_event(TbMdrSelfCureEvent(self), sim.date + DateOffset(months=12))

        sim.schedule_event(TbDeathEvent(self), sim.date + DateOffset(months=12))

        # Logging
        sim.schedule_event(TbLoggingEvent(self), sim.date + DateOffset(days=0))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ HEALTH SYSTEM ~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)


    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        """
        df = self.sim.population.props

        df.at[child_id, 'tb_inf'] = 'uninfected'
        df.at[child_id, 'tb_date_active'] = pd.NaT
        df.at[child_id, 'tb_date_latent'] = pd.NaT
        df.at[child_id, 'tb_ever_tb'] = False
        df.at[child_id, 'tb_ever_tb_mdr'] = False
        df.at[child_id, 'tb_specific_symptoms'] = 'none'
        df.at[child_id, 'tb_unified_symptom_code'] = 0

        df.at[child_id, 'tb_ever_tested'] = False  # default: not tested
        df.at[child_id, 'tb_smear_test'] = False
        df.at[child_id, 'tb_result_smear_test'] = False
        df.at[child_id, 'tb_date_smear_test'] = pd.NaT
        df.at[child_id, 'tb_xpert_test'] = False
        df.at[child_id, 'tb_result_xpert_test'] = False
        df.at[child_id, 'tb_date_xpert_test'] = pd.NaT
        df.at[child_id, 'tb_diagnosed'] = False
        df.at[child_id, 'tb_mdr_diagnosed'] = False
        df.at[child_id, 'tb_on_treatment'] = False
        df.at[child_id, 'tb_date_treated'] = pd.NaT
        df.at[child_id, 'tb_treatment_failure'] = False
        df.at[child_id, 'tb_treated_mdr'] = False
        df.at[child_id, 'tb_date_treated_mdr'] = pd.NaT
        df.at[child_id, 'tb_request_mdr_regimen'] = False
        df.at[child_id, 'tb_on_ipt'] = False
        df.at[child_id, 'tb_date_ipt'] = pd.NaT

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is TB, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

        # create list of appts which link to tb screening
        id_list = ['Hiv_Testing',
                   'Hiv_TestingInfant',
                   'Hiv_InfantTreatmentInitiation',
                   'Hiv_TreatmentInitiation',
                   'Hiv_Treatment']

        if treatment_id in id_list:
            piggy_back_dx_at_appt = HSI_Tb_Screening(self, person_id)
            piggy_back_dx_at_appt.TREATMENT_ID = 'Tb_Screening_PiggybackAppt'

            # Arbitrarily reduce the size of appt footprint to reflect that this is a piggy back appt
            for key in piggy_back_dx_at_appt.APPT_FOOTPRINT:
                piggy_back_dx_at_appt.APPT_FOOTPRINT[key] = piggy_back_dx_at_appt.APPT_FOOTPRINT[key] * 0.5

            self.sim.modules['HealthSystem'].schedule_hsi_event(piggy_back_dx_at_appt,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=None)

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.
        logger.debug('This is tb reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe
        params = self.parameters

        health_values = pd.Series(0, index=df.index)

        health_values.loc[df['tb_inf'].str.contains('active_susc') & ~df.hv_inf & df.is_alive] = params[
            'daly_wt_susc_tb']
        health_values.loc[df['tb_inf'].str.contains('active_mdr') & ~df.hv_inf & df.is_alive] = params[
            'daly_wt_resistant_tb']

        health_values.loc[df['tb_inf'].str.contains('active_susc') & df.hv_inf & df.is_alive] = params[
            'daly_wt_susc_tb_hiv']
        health_values.loc[df['tb_inf'].str.contains('active_mdr') & df.hv_inf & df.is_alive] = params[
            'daly_wt_resistant_tb_hiv']

        health_values.name = 'tb Symptoms'  # label the cause of this disability

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

        # infectious people are active_pulm
        active_hiv_neg = len(df[df['tb_inf'].str.contains('active_susc') & (
            df.tb_specific_symptoms == 'active_pulm') & ~df.hv_inf & df.is_alive])
        active_hiv_pos = len(df[df['tb_inf'].str.contains('active_susc') & (
            df.tb_specific_symptoms == 'active_pulm') & df.hv_inf & df.is_alive])

        # population at-risk of new infection = uninfected
        uninfected_total = len(df[(df.tb_inf == 'uninfected') & df.is_alive])
        total_population = len(df[df.is_alive])

        # in small pops, may get zero values which results in FOI = 0
        if active_hiv_neg == 0:
            active_hiv_neg = 1

        if active_hiv_pos == 0:
            active_hiv_pos = 1


        force_of_infection = (params['transmission_rate'] * active_hiv_neg * (active_hiv_pos * params[
            'rel_inf_hiv']) * uninfected_total) / total_population
        # print('force_of_infection: ', force_of_infection)


        # ----------------------------------- NEW INFECTIONS -----------------------------------

        # pop at risk = uninfected only
        at_risk = df[(df.tb_inf == 'uninfected') & df.is_alive].index

        #  no age/sex effect on risk of latent infection
        prob_tb_new = pd.Series(force_of_infection, index=at_risk)
        # print('prob_tb_new: ', prob_tb_new)

        is_newly_infected = prob_tb_new > rng.rand(len(prob_tb_new))
        new_case = is_newly_infected[is_newly_infected].index
        df.loc[new_case, 'tb_inf'] = 'latent_susc_new'
        df.loc[new_case, 'tb_date_latent'] = now
        df.loc[new_case, 'tb_specific_symptoms'] = 'latent'
        df.loc[new_case, 'tb_unified_symptom_code'] = 0

        # ----------------------------------- RE-INFECTIONS -----------------------------------

        # pop at risk = latent_susc_tx and latent_mdr (new & tx)
        at_risk = df[
            (df.tb_inf == 'latent_susc_tx') | df['tb_inf'].str.contains('latent_mdr') & df.is_alive].index

        #  no age/sex effect on risk of latent infection
        prob_tb_new = pd.Series(force_of_infection, index=at_risk)
        # print('prob_tb_new: ', prob_tb_new)

        repeat_infected = prob_tb_new > rng.rand(len(prob_tb_new))
        repeat_case = repeat_infected[repeat_infected].index

        # unchanged status, high risk of relapse as if just recovered
        df.loc[repeat_case, 'tb_inf'] = 'latent_susc_tx'

        df.loc[repeat_case, 'tb_date_latent'] = now
        df.loc[repeat_case, 'tb_specific_symptoms'] = 'latent'
        df.loc[repeat_case, 'tb_unified_symptom_code'] = 0

        # -----------------------------------------------------------------------------------------------------
        # PROGRESSION TO ACTIVE DISEASE


        # ----------------------------------- ADULT PROGRESSORS TO ACTIVE DISEASE -----------------------------------

        # probability of active disease
        prob_prog = pd.Series(0, index=df.index)
        prob_prog.loc[df.is_alive & df.age_years.between(15, 100)] = params['prog_active']
        prob_prog.loc[df.hv_inf] *= params['rr_tb_hiv']
        prob_prog.loc[df.hv_on_art] *= params['rr_tb_art_adult']
        # prob_prog.loc[df.xxxx] *= params['rr_tb_overweight']
        prob_prog.loc[df.li_overwt] *= params['rr_tb_obese']
        # prob_prog.loc[df.xxxx] *= params['rr_tb_diabetes1']
        prob_prog.loc[df.li_ex_alc] *= params['rr_tb_alcohol']
        prob_prog.loc[df.li_tob] *= params['rr_tb_smoking']

        dur_ipt = pd.to_timedelta(params['dur_prot_ipt'], unit='d')

        prob_prog.loc[(df.tb_date_ipt < (now - dur_ipt)) & ~df.hv_inf] *= params['rr_ipt_adult']
        prob_prog.loc[(df.tb_date_ipt < (now - dur_ipt)) & df.hv_inf & ~df.hv_on_art] *= params[
            'rr_ipt_adult_hiv']
        prob_prog.loc[(df.tb_date_ipt < (now - dur_ipt)) & df.hv_inf & df.hv_on_art] *= params[
            'rr_ipt_art_adult']
        
        # if any newly infected latent cases, 14% become active directly, only for new infections
        new_latent = df[
            (df.tb_inf == 'latent_susc_new') & (df.tb_date_latent == now) & (
                df.age_years >= 15) & df.is_alive].sum()
        # print(new_latent)

        # some proportion schedule active now, else schedule active at random date
        if new_latent.any():

            idx = df[
                (df.tb_inf == 'latent_susc_new') & (df.tb_date_latent == now) & (
                    df.age_years >= 15) & df.is_alive].index

            # for each person determine if fast progressor and schedule onset of active disease
            fast = pd.Series(data=False, index=df.loc[idx].index)

            for i in df.index[idx]:
                fast[i] = self.module.rng.rand() < params['prop_fast_progressor']
                # print('fast', fast)

                if fast.sum() > 0:
                    for person_id in fast.index[fast]:
                        logger.debug(
                            'This is TbEvent, scheduling active disease for fast progressing person %d on date %s',
                            person_id, now)
                        # schedule active disease now
                        self.sim.schedule_event(TbActiveEvent(self, person_id), now)

                for person_id in fast.index[~fast]:
                    # decide if person will develop active tb
                    active = self.module.rng.rand() < prob_prog[person_id]

                    if active:
                        # randomly select date of onset of active disease
                        # random draw of days 0-732
                        random_date = rng.randint(low=0, high=732)
                        sch_date = now + pd.to_timedelta(random_date, unit='d')

                        logger.debug(
                            'This is TbEvent, scheduling active disease for slow progressing person %d on date %s',
                            person_id, sch_date)

                        # schedule active disease
                        self.sim.schedule_event(TbActiveEvent(self, person_id), sch_date)

        # ----------------------------------- CHILD PROGRESSORS TO ACTIVE DISEASE -----------------------------------
        # probability of active disease
        prob_prog_child = pd.Series(0, index=df.index)
        prob_prog_child.loc[df.is_alive & (df.age_years < 1)] = params['prog_1yr']
        prob_prog_child.loc[df.is_alive & (df.age_years.between(1, 2))] = params['prog_1_2yr']
        prob_prog_child.loc[df.is_alive & (df.age_years.between(2, 5))] = params['prog_2_5yr']
        prob_prog_child.loc[df.is_alive & (df.age_years.between(5, 10))] = params['prog_5_10yr']
        prob_prog_child.loc[df.is_alive & (df.age_years.between(10, 15))] = params['prog_10yr']
        # prob_prog_child.loc[df.xxxx] *= params['rr_tb_bcg']        
        prob_prog_child.loc[df.hv_on_art] *= params['rr_tb_art_child']

        dur_ipt_inf = pd.to_timedelta(params['dur_prot_ipt_infant'], unit='d')

        prob_prog_child.loc[(df.tb_date_ipt < (now - dur_ipt_inf)) & ~df.hv_inf] *= params[
            'rr_ipt_child']
        prob_prog_child.loc[(df.tb_date_ipt < (now - dur_ipt_inf)) & df.hv_inf & ~df.hv_on_art] *= \
            params[
            'rr_ipt_child_hiv']
        prob_prog_child.loc[(df.tb_date_ipt < (now - dur_ipt_inf)) & df.hv_inf & df.hv_on_art] *= \
            params[
            'rr_ipt_art_child']

        # no direct progression
        # progression within 1 year
        new_latent_child = df[
            (df.tb_inf == 'latent_susc_new') & (df.tb_date_latent == now) & (
                df.age_years < 15) & df.is_alive].sum()
        # print(new_latent)

        # some proportion schedule active now, else schedule active at random date
        if new_latent_child.any():
            prog = df[
                (df.tb_inf == 'latent_susc_new') & (df.tb_date_latent == now) & (
                    df.age_years < 15) & df.is_alive].index

            for person_id in prog:
                # decide if person will develop active tb
                active = self.module.rng.rand() < prob_prog_child[person_id]

                if active:
                    # random draw of days 0-365
                    # random_date = rng.choice(list(range(0, 365)), size=1, p=[(1 / 365)] * 365)
                    random_date = rng.randint(low=0, high=365)
                    # convert days into years
                    random_days = pd.to_timedelta(random_date, unit='d')

                    sch_date = now + random_days

                    logger.debug('This is TbEvent, scheduling active disease for child %d on date %s',
                                 person_id, sch_date)

                    # schedule active disease
                    self.sim.schedule_event(TbActiveEvent(self, person_id), sch_date)


class TbActiveEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("Onset of active TB for person %d", person_id)

        df = self.sim.population.props
        params = self.sim.modules['tb'].parameters
        prob_pulm = params['pulm_tb']
        rng = self.sim.rng

        # check not on ipt now or on tb treatment

        df.at[person_id, 'tb_date_active'] = self.sim.date

        # check if new infection or re-infection
        # latent_susc_new or latent_susc_tx
        if (df.at[person_id, 'tb_inf'] == 'latent_susc_new'):
            df.at[person_id, 'tb_inf'] = 'active_susc_new'
        elif (df.at[person_id, 'tb_inf'] == 'latent_susc_tx'):
            df.at[person_id, 'tb_inf'] = 'active_susc_tx'

        # decide pulm or extra
        # depends on age and HIV status
        age = df.at[person_id, 'age_years']
        hiv = df.at[person_id, 'hv_inf']
        prob_pulm_tb = prob_pulm.loc[(prob_pulm.age == age) & (prob_pulm.hiv == hiv), 'prob_pulm'].values

        if (rng.rand() < prob_pulm_tb):
            df.at[person_id, 'tb_specific_symptoms'] = 'active_pulm'
            df.at[person_id, 'tb_unified_symptom_code'] = 2

        else:
            df.at[person_id, 'tb_specific_symptoms'] = 'active_extra'
            df.at[person_id, 'tb_unified_symptom_code'] = 3

        # ----------------------------------- ACTIVE CASES SEEKING CARE -----------------------------------

        # for each person determine whether they will seek care on symptom change
        prob_care = self.sim.modules['HealthSystem'].get_prob_seek_care(person_id, symptom_code=2)
        seeks_care = rng.rand() < prob_care

        if seeks_care:
            logger.debug('This is TbActiveEvent, scheduling HSI_Tb_Screening for person %d', person_id)

            event = HSI_Tb_Screening(self.sim.modules['tb'], person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(weeks=2)
                                                                )
        else:
            logger.debug(
                'This is TbActiveEvent, person %d is not seeking care', person_id)


class TbRelapseEvent(RegularEvent, PopulationScopeEventMixin):
    """ relapse from latent to active
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # every 1 month

    def apply(self, population):

        df = self.sim.population.props
        params = self.module.parameters
        now = self.sim.date
        rng = self.module.rng

        # ----------------------------------- RELAPSE -----------------------------------
        random_draw = rng.random_sample(size=len(df))

        # relapse after treatment completion, tb_date_treated + six months
        relapse_tx_complete = df[
            (df.tb_inf == 'latent_susc_tx') & ~df.tb_on_ipt & df.is_alive & (
                self.sim.date - df.tb_date_treated > pd.to_timedelta(182.625, unit='d')) & (
                self.sim.date - df.tb_date_treated < pd.to_timedelta(732.5, unit='d')) & ~df.tb_treatment_failure & (
                random_draw < params['monthly_prob_relapse_tx_complete'])].index

        # relapse after treatment default, tb_treatment_failure=True, but make sure not tb-mdr
        relapse_tx_incomplete = df[
            (df.tb_inf == 'latent_susc_tx') & ~df.tb_on_ipt & df.is_alive & df.tb_treatment_failure & (
                self.sim.date - df.tb_date_treated > pd.to_timedelta(182.625, unit='d')) & (
                self.sim.date - df.tb_date_treated < pd.to_timedelta(732.5, unit='d')) & (
                random_draw < params['monthly_prob_relapse_tx_incomplete'])].index

        # relapse after >2 years following completion of treatment (or default)
        # use tb_date_treated + 2 years + 6 months of treatment
        relapse_tx_2yrs = df[
            (df.tb_inf == 'latent_susc_tx') & ~df.tb_on_ipt & df.is_alive & (
                self.sim.date - df.tb_date_treated >= pd.to_timedelta(913.125, unit='d')) & (
                random_draw < params['monthly_prob_relapse_2yrs'])].index

        df.loc[relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, 'tb_inf'] = 'active_susc_tx'
        df.loc[relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, 'tb_date_active'] = now
        df.loc[relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, 'tb_ever_tb'] = True
        df.loc[relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, 'tb_specific_symptoms'] = 'active_pulm'
        df.loc[relapse_tx_complete | relapse_tx_incomplete, 'tb_unified_symptom_code'] = 2

        # ----------------------------------- RELAPSE CASES SEEKING CARE -----------------------------------

        # relapse after complete treatment course - refer for xpert testing
        seeks_care = pd.Series(data=False, index=df.loc[relapse_tx_complete].index)
        for i in df.loc[relapse_tx_complete].index:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=2)
            seeks_care[i] = self.module.rng.rand() < prob

        if seeks_care.sum() > 0:
            for person_index in seeks_care.index[seeks_care]:

                logger.debug(
                    'This is TbRelapseEvent, scheduling HSI_Tb_XpertTest for person %d',
                    person_index)
                event = HSI_Tb_XpertTest(self.module, person_id=person_index)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=2,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(weeks=2)
                                                                    )

                # add back-up check if xpert is not available, then schedule sputum smear
                self.sim.schedule_event(TbCheckXpert(self, person_index), self.sim.date + DateOffset(weeks=2))

        else:
            logger.debug(
                'This is TbRelapseEvent, there are no new relapse cases seeking care')

        # relapse after incomplete treatment course - repeat treatment course
        seeks_care = pd.Series(data=False, index=df.loc[relapse_tx_incomplete].index)
        for i in df.loc[relapse_tx_incomplete].index:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=2)
            seeks_care[i] = self.module.rng.rand() < prob

        if seeks_care.sum() > 0:
            for person_index in seeks_care.index[seeks_care]:
                if df.at[person_index, 'age_years'] < 15:

                    logger.debug(
                        'This is TbActiveEvent, scheduling HSI_Tb_StartTreatmentChild for relapsed child %d',
                        person_index)
                    event = HSI_Tb_StartTreatmentChild(self.module, person_id=person_index)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                        priority=2,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(weeks=2)
                                                                        )
                else:
                    logger.debug(
                        'This is TbActiveEvent, scheduling HSI_Tb_StartTreatmentChild for relapsed adult %d',
                        person_index)
                    event = HSI_Tb_StartTreatmentAdult(self.module, person_id=person_index)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                        priority=2,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(weeks=2)
                                                                        )

        else:
            logger.debug(
                'This is TbRelapseEvent, there are no new relapse cases seeking care')


class TbCheckXpert(Event, IndividualScopeEventMixin):
    """ if person has not received xpert test as prescribed, schedule sputum smear
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("Checking if person %d received xpert", person_id)

        df = self.sim.population.props
        now = self.sim.date

        # if no xpert test within last 14 days, schedule sputum smear
        if ~df.at(person_id, 'tb_xpert_test') | (df.at(person_id, 'tb_xpert_test') &
                                                 (now - df.at[person_id, 'tb_date_xpert_test']) > 14):
            logger.debug(
                'This is TbCheckXpert, scheduling HSI_Tb_SputumTest for person %d',
                person_id)

            event = HSI_Tb_SputumTest(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=3,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(weeks=2)
                                                                )












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
            df['tb_inf'].str.contains('active_susc') & df.is_alive & (
                df.tb_date_active < now) & (random_draw < params['monthly_prob_self_cure'])].index
        df.loc[self_cure, 'tb_inf'] = 'latent_susc_tx'
        df.loc[self_cure, 'tb_specific_symptoms'] = 'latent'
        df.loc[self_cure, 'tb_unified_symptom_code'] = 0


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

        # population at-risk = uninfected
        active_hiv_neg = len(df[df['tb_inf'].str.contains('active_mdr') & (
            df.tb_specific_symptoms == 'active_pulm') & ~df.hv_inf & df.is_alive])
        active_hiv_pos = len(df[df['tb_inf'].str.contains('active_mdr') & (
            df.tb_specific_symptoms == 'active_pulm') & df.hv_inf & df.is_alive])

        uninfected_total = len(df[(df.tb_inf == 'uninfected') & df.is_alive])
        total_population = len(df[df.is_alive])

        force_of_infection = (params['transmission_rate'] * active_hiv_neg * (active_hiv_pos * params[
            'rel_inf_hiv']) * uninfected_total) / total_population
        # print('force_of_infection: ', force_of_infection)

        # ----------------------------------- NEW INFECTIONS -----------------------------------

        # pop at risk = susceptible and latent_susc, latent_mdr_primary only
        at_risk = df[(df.tb_inf == 'uninfected') & df.is_alive].index

        #  no age/sex effect on risk of latent infection
        prob_tb_new = pd.Series(force_of_infection, index=at_risk)
        # print('prob_tb_new: ', prob_tb_new)

        is_newly_infected = prob_tb_new > rng.rand(len(prob_tb_new))
        new_case = is_newly_infected[is_newly_infected].index
        df.loc[new_case, 'tb_inf'] = 'latent_mdr_new'
        df.loc[new_case, 'tb_date_latent'] = now
        df.loc[new_case, 'tb_specific_symptoms'] = 'latent'
        df.loc[new_case, 'tb_unified_symptom_code'] = 0

        # ----------------------------------- RE-INFECTIONS -----------------------------------

        # pop at risk = latent_mdr_secondary, latent_susc (primary & secondary)
        at_risk = df[
            (df.tb_inf == 'latent_mdr_tx') | df['tb_inf'].str.contains('latent_susc') & df.is_alive].index

        #  no age/sex effect on risk of latent infection
        prob_tb_new = pd.Series(force_of_infection, index=at_risk)
        # print('prob_tb_new: ', prob_tb_new)

        repeat_infected = prob_tb_new > rng.rand(len(prob_tb_new))
        repeat_case = repeat_infected[repeat_infected].index

        # unchanged status, high risk of relapse as if just recovered
        df.loc[repeat_case, 'tb_inf'] = 'latent_mdr_tx'

        df.loc[repeat_case, 'tb_date_latent'] = now
        df.loc[repeat_case, 'tb_specific_symptoms'] = 'latent'
        df.loc[repeat_case, 'tb_unified_symptom_code'] = 0

        # -----------------------------------------------------------------------------------------------------
        # PROGRESSION TO ACTIVE MDR-TB DISEASE

        # ----------------------------------- ADULT PROGRESSORS TO ACTIVE DISEASE -----------------------------------

        # probability of active disease
        prob_prog = pd.Series(0, index=df.index)
        prob_prog.loc[df.is_alive & df.age_years.between(15, 100)] = params['prog_active']
        prob_prog.loc[df.hv_inf] *= params['rr_tb_hiv']
        prob_prog.loc[df.hv_on_art] *= params['rr_tb_art_adult']
        # prob_prog.loc[df.xxxx] *= params['rr_tb_overweight']
        prob_prog.loc[df.li_overwt] *= params['rr_tb_obese']
        # prob_prog.loc[df.xxxx] *= params['rr_tb_diabetes1']
        prob_prog.loc[df.li_ex_alc] *= params['rr_tb_alcohol']
        prob_prog.loc[df.li_tob] *= params['rr_tb_smoking']

        dur_ipt = pd.to_timedelta(params['dur_prot_ipt'], unit='d')

        prob_prog.loc[(df.tb_date_ipt < (now - dur_ipt)) & ~df.hv_inf] *= params['rr_ipt_adult']
        prob_prog.loc[(df.tb_date_ipt < (now - dur_ipt)) & df.hv_inf & ~df.hv_on_art] *= params[
            'rr_ipt_adult_hiv']
        prob_prog.loc[(df.tb_date_ipt < (now - dur_ipt)) & df.hv_inf & df.hv_on_art] *= params[
            'rr_ipt_art_adult']

        # if any newly infected latent cases, 14% become active directly, only for new infections
        new_latent = df[
            (df.tb_inf == 'latent_mdr_new') & (df.tb_date_latent == now) & (
                df.age_years >= 15) & df.is_alive].sum()
        # print(new_latent)

        # some proportion schedule active now, else schedule active at random date
        if new_latent.any():

            idx = df[
                (df.tb_inf == 'latent_mdr_new') & (df.tb_date_latent == now) & (
                    df.age_years >= 15) & df.is_alive].index

            # for each person determine if fast progressor and schedule onset of active disease
            fast = pd.Series(data=False, index=df.loc[idx].index)

            for i in df.index[idx]:
                fast[i] = self.module.rng.rand() < params['prop_fast_progressor']
                # print('fast', fast)

                if fast.sum() > 0:
                    for person_id in fast.index[fast]:
                        logger.debug(
                            'This is TbMdrEvent, scheduling active disease for fast progressing person %d on date %s',
                            person_id, now)
                        # schedule active disease now
                        self.sim.schedule_event(TbMdrActiveEvent(self, person_id), now)

                for person_id in fast.index[~fast]:
                    # decide if person will develop active tb
                    active = self.module.rng.rand() < prob_prog[person_id]

                    if active:
                        # randomly select date of onset of active disease
                        # random draw of days 0-732
                        random_date = rng.randint(low=0, high=732)
                        sch_date = now + pd.to_timedelta(random_date, unit='d')

                        logger.debug(
                            'This is TbMdrEvent, scheduling active disease for slow progressing person %d on date %s',
                            person_id, sch_date)

                        # schedule active disease
                        self.sim.schedule_event(TbMdrActiveEvent(self, person_id), sch_date)

        # ----------------------------------- CHILD PROGRESSORS TO ACTIVE DISEASE -----------------------------------
        # probability of active disease
        prob_prog_child = pd.Series(0, index=df.index)
        prob_prog_child.loc[df.is_alive & (df.age_years < 1)] = params['prog_1yr']
        prob_prog_child.loc[df.is_alive & (df.age_years.between(1, 2))] = params['prog_1_2yr']
        prob_prog_child.loc[df.is_alive & (df.age_years.between(2, 5))] = params['prog_2_5yr']
        prob_prog_child.loc[df.is_alive & (df.age_years.between(5, 10))] = params['prog_5_10yr']
        prob_prog_child.loc[df.is_alive & (df.age_years.between(10, 15))] = params['prog_10yr']
        # prob_prog_child.loc[df.xxxx] *= params['rr_tb_bcg']
        prob_prog_child.loc[df.hv_on_art] *= params['rr_tb_art_child']

        dur_ipt_inf = pd.to_timedelta(params['dur_prot_ipt_infant'], unit='d')

        prob_prog_child.loc[(df.tb_date_ipt < (now - dur_ipt_inf)) & ~df.hv_inf] *= params[
            'rr_ipt_child']
        prob_prog_child.loc[(df.tb_date_ipt < (now - dur_ipt_inf)) & df.hv_inf & ~df.hv_on_art] *= \
            params[
                'rr_ipt_child_hiv']
        prob_prog_child.loc[(df.tb_date_ipt < (now - dur_ipt_inf)) & df.hv_inf & df.hv_on_art] *= \
            params[
                'rr_ipt_art_child']

        # no direct progression
        # progression within 1 year
        new_latent_child = df[
            (df.tb_inf == 'latent_mdr_new') & (df.tb_date_latent == now) & (
                df.age_years < 15) & df.is_alive].sum()
        # print(new_latent)

        # some proportion schedule active now, else schedule active at random date
        if new_latent_child.any():
            prog = df[
                (df.tb_inf == 'latent_mdr_new') & (df.tb_date_latent == now) & (
                    df.age_years < 15) & df.is_alive].index

            for person_id in prog:
                # decide if person will develop active tb
                active = self.module.rng.rand() < prob_prog_child[person_id]

                if active:
                    # random draw of days 0-365
                    # random_date = rng.choice(list(range(0, 365)), size=1, p=[(1 / 365)] * 365)
                    random_date = rng.randint(low=0, high=365)
                    # convert days into years
                    random_days = pd.to_timedelta(random_date, unit='d')

                    sch_date = now + random_days

                    logger.debug('This is TbMdrEvent, scheduling active disease for child %d on date %s',
                                 person_id, sch_date)

                    # schedule active disease
                    self.sim.schedule_event(TbMdrActiveEvent(self, person_id), sch_date)


class TbMdrActiveEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("Onset of active MDR-TB for person %d", person_id)

        df = self.sim.population.props
        params = self.sim.modules['tb'].parameters
        prob_pulm = params['pulm_tb']
        rng = self.sim.rng

        # check not on ipt now or on tb treatment

        df.at[person_id, 'tb_date_active'] = self.sim.date

        # check if new infection or re-infection
        # latent_susc_new or latent_susc_tx
        if (df.at[person_id, 'tb_inf'] == 'latent_mdr_new'):
            df.at[person_id, 'tb_inf'] = 'active_mdr_new'
        elif (df.at[person_id, 'tb_inf'] == 'latent_mdr_tx'):
            df.at[person_id, 'tb_inf'] = 'active_mdr_tx'

        # decide pulm or extra
        # depends on age and HIV status
        age = df.at[person_id, 'age_years']
        hiv = df.at[person_id, 'hv_inf']
        prob_pulm_tb = prob_pulm.loc[(prob_pulm.age == age) & (prob_pulm.hiv == hiv), 'prob_pulm'].values

        if (rng.rand() < prob_pulm_tb):
            df.at[person_id, 'tb_specific_symptoms'] = 'active_pulm'
            df.at[person_id, 'tb_unified_symptom_code'] = 2

        else:
            df.at[person_id, 'tb_specific_symptoms'] = 'active_extra'
            df.at[person_id, 'tb_unified_symptom_code'] = 3

        # ----------------------------------- ACTIVE CASES SEEKING CARE -----------------------------------

        # for each person determine whether they will seek care on symptom change
        prob_care = self.sim.modules['HealthSystem'].get_prob_seek_care(person_id, symptom_code=2)
        seeks_care = rng.rand() < prob_care

        if seeks_care:
            logger.debug('This is TbMdrActiveEvent, scheduling HSI_Tb_Screening for person %d', person_id)

            event = HSI_Tb_Screening(self.sim.modules['tb'], person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(weeks=2)
                                                                )
        else:
            logger.debug(
                'This is TbMdrActiveEvent, person %d is not seeking care', person_id)


class TbMdrRelapseEvent(RegularEvent, PopulationScopeEventMixin):
    """ relapse from latent to active
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # every 1 month

    def apply(self, population):

        df = self.sim.population.props
        params = self.module.parameters
        now = self.sim.date
        rng = self.module.rng

        # ----------------------------------- RELAPSE -----------------------------------
        random_draw = rng.random_sample(size=len(df))

        # relapse after treatment completion, tb_date_treated + six months
        relapse_tx_complete = df[
            (df.tb_inf == 'latent_mdr_tx') & ~df.tb_on_ipt & df.is_alive & (
                self.sim.date - df.tb_date_treated > pd.to_timedelta(182.625, unit='d')) & (
                self.sim.date - df.tb_date_treated < pd.to_timedelta(732.5, unit='d')) & ~df.tb_treatment_failure & (
                random_draw < params['monthly_prob_relapse_tx_complete'])].index

        # relapse after treatment default, tb_treatment_failure=True, but make sure not tb-mdr
        relapse_tx_incomplete = df[
            (df.tb_inf == 'latent_mdr_tx') & ~df.tb_on_ipt & df.is_alive & df.tb_treatment_failure & (
                self.sim.date - df.tb_date_treated > pd.to_timedelta(182.625, unit='d')) & (
                self.sim.date - df.tb_date_treated < pd.to_timedelta(732.5, unit='d')) & (
                random_draw < params['monthly_prob_relapse_tx_incomplete'])].index

        # relapse after >2 years following completion of treatment (or default)
        # use tb_date_treated + 2 years + 6 months of treatment
        relapse_tx_2yrs = df[
            (df.tb_inf == 'latent_mdr_secondary') & ~df.tb_on_ipt & df.is_alive & (
                self.sim.date - df.tb_date_treated >= pd.to_timedelta(732.5, unit='d')) & (
                random_draw < params['monthly_prob_relapse_2yrs'])].index

        df.loc[relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, 'tb_inf'] = 'active_mdr_tx'
        df.loc[relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, 'tb_date_active'] = now
        df.loc[relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, 'tb_ever_tb'] = True
        df.loc[relapse_tx_complete | relapse_tx_incomplete | relapse_tx_2yrs, 'tb_specific_symptoms'] = 'active_pulm'
        df.loc[relapse_tx_complete | relapse_tx_incomplete, 'tb_unified_symptom_code'] = 2

        # ----------------------------------- RELAPSE CASES SEEKING CARE -----------------------------------

        # relapse after complete treatment course - refer for xpert testing
        seeks_care = pd.Series(data=False, index=df.loc[relapse_tx_complete].index)
        for i in df.loc[relapse_tx_complete].index:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=2)
            seeks_care[i] = self.module.rng.rand() < prob

        if seeks_care.sum() > 0:
            for person_index in seeks_care.index[seeks_care]:
                logger.debug(
                    'This is TbMdrRelapseEvent, scheduling HSI_Tb_XpertTest for person %d',
                    person_index)
                event = HSI_Tb_XpertTest(self.module, person_id=person_index)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=2,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(weeks=2)
                                                                    )

                # add back-up check if xpert is not available, then schedule sputum smear
                self.sim.schedule_event(TbCheckXpert(self, person_index), self.sim.date + DateOffset(weeks=2))
        else:
            logger.debug(
                'This is TbMdrRelapseEvent, there are no new relapse cases seeking care')

        # relapse after incomplete treatment course - repeat treatment course
        seeks_care = pd.Series(data=False, index=df.loc[relapse_tx_incomplete].index)
        for i in df.loc[relapse_tx_incomplete].index:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=2)
            seeks_care[i] = self.module.rng.rand() < prob

        if seeks_care.sum() > 0:
            for person_index in seeks_care.index[seeks_care]:
                if df.at[person_index, 'age_years'] < 15:

                    logger.debug(
                        'This is TbMdrActiveEvent, scheduling HSI_Tb_StartTreatmentChild for relapsed child %d',
                        person_index)
                    event = HSI_Tb_StartTreatmentChild(self.module, person_id=person_index)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                        priority=2,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(weeks=2)
                                                                        )
                else:
                    logger.debug(
                        'This is TbMdrActiveEvent, scheduling HSI_Tb_StartTreatmentChild for relapsed adult %d',
                        person_index)
                    event = HSI_Tb_StartTreatmentAdult(self.module, person_id=person_index)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                        priority=2,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(weeks=2)
                                                                        )

        else:
            logger.debug(
                'This is TbMdrRelapseEvent, there are no new relapse cases seeking care')



class TbMdrSelfCureEvent(RegularEvent, PopulationScopeEventMixin):
    """ tb-mdr self-cure events
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # every 1 month

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date

        df = population.props

        # self-cure - move from active to latent_secondary, make sure it's not the ones that just became active
        random_draw = self.sim.rng.random_sample(size=len(df))

        self_cure = df[df['tb_inf'].str.contains('active_mdr') & df.is_alive & (
            df.tb_date_active < now) & (random_draw < params['monthly_prob_self_cure'])].index
        df.loc[self_cure, 'tb_inf'] = 'latent_mdr_tx'
        df.loc[self_cure, 'tb_specific_symptoms'] = 'latent'
        df.loc[self_cure, 'tb_unified_symptom_code'] = 0


# ---------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTIONS
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#   Testing
# ---------------------------------------------------------------------------

class HSI_Tb_Screening(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It is the screening event that occurs before a sputum smear test or xpert is offered
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 0.5  # This requires a few minutes of an outpatient appt

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Tb_Testing'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ACCEPTED_FACILITY_LEVELS = ['*']  # can occur at any facility level
        self.ALERT_OTHER_DISEASES = ['hiv']

    def apply(self, person_id):
        logger.debug('This is HSI_TbScreening, a screening appointment for person %d', person_id)

        df = self.sim.population.props
        # params = self.module.parameters
        # now = self.sim.date

        # check across all disease modules if patient has: cough, fever, night sweat, weight loss
        # if any of the above conditions are present, label as presumptive tb case and request appropriate test
        if (df.at[person_id, 'tb_specific_symptoms'] == 'active_pulm') and not (df.at[person_id, 'hv_inf']):
            logger.debug("This is HSI_Tb_SputumTest scheduling xpert test for person %d", person_id)

            test = HSI_Tb_SputumTest(self.module, person_id=person_id)

            # Request the health system to give xpert test
            self.sim.modules['HealthSystem'].schedule_hsi_event(test,
                                                                priority=1,
                                                                topen=self.sim.date,
                                                                tclose=None)

        elif (df.at[person_id, 'tb_specific_symptoms'] == 'active_pulm') and (df.at[person_id, 'hv_inf']):
            logger.debug("This is HSI_Tb_SputumTest scheduling xpert test for person %d", person_id)

            test = HSI_Tb_XpertTest(self.module, person_id=person_id)

            # Request the health system to give xpert test
            self.sim.modules['HealthSystem'].schedule_hsi_event(test,
                                                                priority=1,
                                                                topen=self.sim.date,
                                                                tclose=None)

            # add back-up check if xpert is not available, then schedule sputum smear
            self.sim.schedule_event(TbCheckXpert(self, person_id), self.sim.date + DateOffset(weeks=2))



class HSI_Tb_SputumTest(Event, IndividualScopeEventMixin):
    """
    This is a sputum test for presumptive tb cases
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ConWithDCSA'] = 1  # This requires one generic outpatient appt
        the_appt_footprint['LabTBMicro'] = 1  # This requires one lab appt for microscopy

        # Get the consumables required
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = \
            pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Microscopy Test', 'Intervention_Pkg_Code'])[
                0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Tb_SputumTest'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = ['*']  # can occur at any facility level
        self.ALERT_OTHER_DISEASES = ['hiv']

    def apply(self, person_id):
        logger.debug('This is HSI_Tb_SputumTest, a first appointment for person %d', person_id)

        df = self.sim.population.props
        params = self.module.parameters
        now = self.sim.date

        df.at[person_id, 'tb_ever_tested'] = True
        df.at[person_id, 'tb_smear_test'] = True
        df.at[person_id, 'tb_date_smear_test'] = now
        df.at[person_id, 'tb_result_smear_test'] = False
        df.at[person_id, 'tb_diagnosed'] = False

        # ----------------------------------- OUTCOME OF TEST -----------------------------------

        # active tb, hiv-negative
        if (df.at[person_id, 'tb_specific_symptoms'] == 'active_pulm') and not df.at[person_id, 'hv_inf']:
            diagnosed = self.sim.rng.choice([True, False], size=1, p=[params['prop_smear_positive'],
                                                                      (1 - params['prop_smear_positive'])])
            if diagnosed:
                df.at[person_id, 'tb_result_smear_test'] = True
                df.at[person_id, 'tb_diagnosed'] = True

        # hiv+, 80% of smear tests will be negative - extrapulmonary
        elif (df.at[person_id, 'tb_specific_symptoms'] == 'active_pulm') and df.at[person_id, 'hv_inf']:
            diagnosed = self.sim.rng.choice([True, False], size=1, p=[params['prop_smear_positive_hiv'],
                                                                      (1 - params['prop_smear_positive_hiv'])])

            if diagnosed:
                df.at[person_id, 'tb_result_smear_test'] = True
                df.at[person_id, 'tb_diagnosed'] = True

        # ----------------------------------- REFERRALS FOR SECONDARY TESTING -----------------------------------

        # remaining 20% of active cases and negative cases referred for xpert testing
        # schedule xpert testing
        if not df.at[person_id, 'tb_diagnosed']:
            logger.debug("This is HSI_Tb_SputumTest scheduling xpert test for person %d", person_id)

            secondary_test = HSI_Tb_XpertTest(self.module, person_id=person_id)

            # Request the health system to give xpert test
            self.sim.modules['HealthSystem'].schedule_hsi_event(secondary_test,
                                                                priority=1,
                                                                topen=self.sim.date + DateOffset(days=1),
                                                                tclose=None)

            # add back-up check if xpert is not available, then schedule sputum smear
            self.sim.schedule_event(TbCheckXpert(self, person_id), self.sim.date + DateOffset(weeks=2))

        # ----------------------------------- REFERRALS FOR TREATMENT -----------------------------------

        if (df.at[person_id, 'tb_diagnosed'] & (
            df.at[person_id, 'tb_inf'] == 'active_susc_new') & (
            df.at[person_id, 'age_years'] < 15)):
            # request child treatment
            logger.debug("This is HSI_Tb_SputumTest scheduling HSI_Tb_StartTreatmentChild for person %d", person_id)

            treatment = HSI_Tb_StartTreatmentChild(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(treatment,
                                                                priority=1,
                                                                topen=self.sim.date + DateOffset(days=1),
                                                                tclose=None)

        if (df.at[person_id, 'tb_diagnosed'] & (
            df.at[person_id, 'tb_inf'] == 'active_susc_new') & (
            df.at[person_id, 'age_years'] >= 15)):
            # request adult treatment
            logger.debug("This is HSI_Tb_SputumTest scheduling HSI_Tb_StartTreatmentAdult for person %d", person_id)

            treatment = HSI_Tb_StartTreatmentAdult(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(treatment,
                                                                priority=1,
                                                                topen=self.sim.date + DateOffset(days=1),
                                                                tclose=None)

        if (df.at[person_id, 'tb_diagnosed'] & (
            df.at[person_id, 'tb_inf'] == 'active_susc_tx') & (
            df.at[person_id, 'age_years'] < 15)):
            # request child retreatment
            logger.debug("This is HSI_Tb_SputumTest scheduling HSI_Tb_RetreatmentChild for person %d", person_id)

            treatment = HSI_Tb_RetreatmentChild(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(treatment,
                                                                priority=1,
                                                                topen=self.sim.date + DateOffset(days=1),
                                                                tclose=None)

        if (df.at[person_id, 'tb_diagnosed'] & (
            df.at[person_id, 'tb_inf'] == 'active_susc_tx') & (
            df.at[person_id, 'age_years'] >= 15)):
            # request adult retreatment
            logger.debug("This is HSI_Tb_SputumTest scheduling HSI_Tb_RetreatmentAdult for person %d", person_id)

            treatment = HSI_Tb_RetreatmentAdult(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(treatment,
                                                                priority=1,
                                                                topen=self.sim.date + DateOffset(days=1),
                                                                tclose=None)

            # ----------------------------------- REFERRALS FOR IPT -----------------------------------

            # trigger ipt outreach event for all paediatric contacts of diagnosed case
            # randomly sample from <5 yr olds, match by district
            district = df.at[person_id, 'district_of_residence']

            if len(df[(df.age_years <= 5) & ~df.ever_tb & ~df.ever_tb_mdr &
                      df.is_alive & (df.district_of_residence == district)].index) > 5:

                ipt_sample = df[(df.age_years <= 5) &
                                ~df.ever_tb &
                                ~df.ever_tb_mdr &
                                df.is_alive &
                                (df.district_of_residence == district)].sample(n=5, replace=False).index

                for person_id in ipt_sample:
                    logger.debug("This is HSI_Tb_SputumTest scheduling HSI_Tb_Ipt for person %d", person_id)

                    ipt_event = HSI_Tb_Ipt(self.module, person_id=person_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(ipt_event,
                                                                        priority=1,
                                                                        topen=self.sim.date + DateOffset(days=1),
                                                                        tclose=None)


class HSI_Tb_XpertTest(Event, IndividualScopeEventMixin):
    """
        This is a Health System Interaction Event - tb xpert test
        """

    # TODO if xpert test not available then request sputum test or clinical diagnosis / chest x-ray

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['TBFollowUp'] = 1
        the_appt_footprint['LabSero'] = 1

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg'] == 'Xpert test',
                'Intervention_Pkg_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Tb_XpertTest'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1, 2, 3]
        self.ALERT_OTHER_DISEASES = ['hiv']

    def apply(self, person_id):
        logger.debug("This is HSI_Tb_XpertTest giving xpert test for person %d", person_id)

        df = self.sim.population.props
        now = self.sim.date

        df.at[person_id, 'tb_ever_tested'] = True
        df.at[person_id, 'tb_xpert_test'] = True
        df.at[person_id, 'tb_date_xpert_test'] = now
        df.at[person_id, 'tb_result_xpert_test'] = False
        df.at[person_id, 'tb_diagnosed_mdr'] = False  # default
        # TODO add tb_diagnosed as well, if not mdr but referred because HIV+

        # a further 10% of TB cases fail to be diagnosed with Xpert (smear-negative + sensitivity of test)
        # they will present back to the health system with some delay (2-4 weeks)
        if df.at[person_id, 'tb_inf'].startswith("active"):

            diagnosed = self.sim.rng.choice([True, False], size=1, p=[0.9, 0.1])

            if diagnosed:
                df.at[person_id, 'tb_result_xpert_test'] = True
                df.at[person_id, 'tb_diagnosed'] = True

                if df.at[person_id, 'tb_inf'].startswith('active_mdr'):
                    df.at[person_id, 'tb_diagnosed_mdr'] = True

            # if diagnosed, trigger ipt outreach event for all paediatric contacts of case
            district = df.at[person_id, 'district_of_residence']

            # check enough contacts available for sample
            if len(df[(df.age_years <= 5) &
                      ~df.tb_ever_tb &
                      ~df.tb_ever_tb_mdr &
                      df.is_alive &
                      (df.district_of_residence == district)].index) > 5:

                # randomly sample from <5 yr olds within district
                ipt_sample = df[(df.age_years <= 5) &
                                ~df.tb_ever_tb &
                                ~df.tb_ever_tb_mdr &
                                df.is_alive &
                                (df.district_of_residence == district)].sample(
                    n=5,
                    replace=False).index

                for person_id in ipt_sample:
                    logger.debug(
                        'This is HSI_Tb_XpertTest, scheduling IPT for person %d', person_id)

                    ipt_event = HSI_Tb_Ipt(self.module, person_id=person_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(ipt_event,
                                                                        priority=1,
                                                                        topen=self.sim.date,
                                                                        tclose=None)

            else:
                # Request the health system to give repeat xpert test
                logger.debug("This is HSI_Tb_XpertTest with negative result for person %d", person_id)

                # Request the health system to give another xpert test if tb still suspected
                # TODO do a symptom check here - decide if repeat testing
                # currently, random prob of still being presumptive tb case
                if self.module.rng.rand() < 0.5:
                    secondary_test = HSI_Tb_XpertTest(self.module, person_id=person_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(secondary_test,
                                                                        priority=1,
                                                                        topen=self.sim.date,
                                                                        tclose=None)

                    # add back-up check if xpert is not available, then schedule sputum smear
                    self.sim.schedule_event(TbCheckXpert(self, person_id), self.sim.date + DateOffset(weeks=2))

        # ----------------------------------- REFERRALS FOR TREATMENT -----------------------------------
        if (df.at[person_id, 'tb_diagnosed'] &
            (df.at[person_id, 'tb_inf'] == 'active_susc_new') & (
                df.at[person_id, 'age_years'] < 15)):
            # request child treatment
            logger.debug("This is HSI_Tb_XpertTest scheduling HSI_Tb_StartTreatmentChild for person %d", person_id)

            treatment = HSI_Tb_StartTreatmentChild(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(treatment,
                                                                priority=1,
                                                                topen=self.sim.date,
                                                                tclose=None)

        if (df.at[person_id, 'tb_diagnosed'] &
            (df.at[person_id, 'tb_inf'] == 'active_susc_new') & (
                df.at[person_id, 'age_years'] >= 15)):
            # request adult treatment
            logger.debug("This is HSI_Tb_XpertTest scheduling HSI_Tb_StartTreatmentAdult for person %d", person_id)

            treatment = HSI_Tb_StartTreatmentAdult(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(treatment,
                                                                priority=1,
                                                                topen=self.sim.date,
                                                                tclose=None)

        if (df.at[person_id, 'tb_diagnosed'] &
            (df.at[person_id, 'tb_inf'] == 'active_susc_tx') & (
                df.at[person_id, 'age_years'] < 15)):
            # request child retreatment
            logger.debug("This is HSI_Tb_XpertTest scheduling HSI_Tb_StartTreatmentChild for person %d", person_id)

            treatment = HSI_Tb_RetreatmentChild(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(treatment,
                                                                priority=1,
                                                                topen=self.sim.date,
                                                                tclose=None)

        if (df.at[person_id, 'tb_diagnosed'] &
            (df.at[person_id, 'tb_inf'] == 'active_susc_tx') & (
                df.at[person_id, 'age_years'] >= 15)):
            # request adult retreatment
            logger.debug("This is HSI_Tb_XpertTest scheduling HSI_Tb_StartTreatmentAdult for person %d", person_id)

            treatment = HSI_Tb_RetreatmentAdult(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(treatment,
                                                                priority=1,
                                                                topen=self.sim.date,
                                                                tclose=None)

        if df.at[person_id, 'tb_diagnosed'] & df.at[person_id, 'tb_inf'].startswith("active_mdr"):
            # request treatment
            logger.debug("This is HSI_Tb_XpertTest scheduling HSI_Tb_StartMdrTreatment for person %d", person_id)

            treatment = HSI_Tb_StartMdrTreatment(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(treatment,
                                                                priority=1,
                                                                topen=self.sim.date,
                                                                tclose=None)


# ---------------------------------------------------------------------------
#   Treatment
# ---------------------------------------------------------------------------
# the consumables at treatment initiation include the cost for the full course of treatment
# so the follow-up appts don't need to account for consumables, just appt time

class HSI_Tb_StartTreatmentAdult(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event - start tb treatment
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['TBNew'] = 1  # New tb treatment initiation appt, this include pharmacist time

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg'] == 'First line treatment for new TB cases for adults',
                'Intervention_Pkg_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Tb_TreatmentInitiationAdult'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = ['*']  # can occur at any facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug("We are now ready to treat this tb case %d", person_id)

        now = self.sim.date
        df = self.sim.population.props
        params = self.module.parameters

        # treatment allocated
        if df.at[person_id, 'is_alive'] and df.at[person_id, 'tb_diagnosed']:
            df.at[person_id, 'tb_on_treatment'] = True
            df.at[person_id, 'date_tb_treated'] = now

        # schedule a 6-month event where people are cured, symptoms return to latent or not cured
        self.sim.schedule_event(TbCureEvent(self, person_id), self.sim.date + DateOffset(months=6))

        # follow-up appts
        # TODO check intensive phase and continuation phase
        logger.debug('....This is HSI_Tb_StartTreatmentAdult: scheduling follow-up appointments for person %d',
                     person_id)

        # clinical monitoring
        followup_clin = HSI_Tb_FollowUp(self.module, person_id=person_id)

        # Request the health system to have this follow-up appointment
        weeks = params['followup_times']['treatment_clinical']
        cleanedList = [x for x in weeks if x == x]  # remove nan

        for i in range(0, len(cleanedList)):
            followup_appt_date = self.sim.date + DateOffset(months=cleanedList[i])
            self.sim.modules['HealthSystem'].schedule_hsi_event(followup_clin,
                                                                priority=1,
                                                                topen=followup_appt_date,
                                                                tclose=followup_appt_date + DateOffset(days=3)
                                                                )

        # repeat sputum smear tests
        followup_smear = HSI_Tb_FollowUp_SputumTest(self.module, person_id=person_id)

        # Request the health system to have this follow-up appointment
        weeks = params['followup_times']['treatment_sputum']
        cleanedList = [x for x in weeks if x == x]  # remove nan

        for i in range(0, len(cleanedList)):
            followup_appt_date = self.sim.date + DateOffset(months=cleanedList[i])
            self.sim.modules['HealthSystem'].schedule_hsi_event(followup_smear,
                                                                priority=1,
                                                                topen=followup_appt_date,
                                                                tclose=followup_appt_date + DateOffset(days=3)
                                                                )


class HSI_Tb_StartTreatmentChild(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event - start tb treatment
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['TBNew'] = 1  # New tb treatment initiation appt

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg'] == 'First line treatment for new TB cases for children',
                'Intervention_Pkg_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Tb_TreatmentInitiationChild'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = ['*']  # can occur at any facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug("We are now ready to treat this tb case %d", person_id)

        now = self.sim.date
        df = self.sim.population.props
        params = self.module.parameters

        # treatment allocated
        if df.at[person_id, 'is_alive'] and df.at[person_id, 'tb_diagnosed']:
            df.at[person_id, 'tb_on_treatment'] = True
            df.at[person_id, 'date_tb_treated'] = now

        # schedule a 6-month event where people are cured, symptoms return to latent or not cured
        self.sim.schedule_event(TbCureEvent(self, person_id), self.sim.date + DateOffset(months=6))

        # follow-up appts
        logger.debug('....This is HSI_Tb_StartTreatmentChild: scheduling follow-up appointments for person %d',
                     person_id)

        # clinical monitoring
        followup_clin = HSI_Tb_FollowUp(self.module, person_id=person_id)

        # Request the health system to have this follow-up appointment
        weeks = params['followup_times']['treatment_clinical']
        cleanedList = [x for x in weeks if x == x]  # remove nan

        for i in range(0, len(cleanedList)):
            followup_appt_date = self.sim.date + DateOffset(months=cleanedList[i])
            self.sim.modules['HealthSystem'].schedule_hsi_event(followup_clin,
                                                                priority=1,
                                                                topen=followup_appt_date,
                                                                tclose=followup_appt_date + DateOffset(days=3)
                                                                )

        # repeat sputum smear tests
        followup_smear = HSI_Tb_FollowUp_SputumTest(self.module, person_id=person_id)

        # Request the health system to have this follow-up appointment
        weeks = params['followup_times']['treatment_sputum']
        cleanedList = [x for x in weeks if x == x]  # remove nan

        for i in range(0, len(cleanedList)):
            followup_appt_date = self.sim.date + DateOffset(months=cleanedList[i])
            self.sim.modules['HealthSystem'].schedule_hsi_event(followup_smear,
                                                                priority=1,
                                                                topen=followup_appt_date,
                                                                tclose=followup_appt_date + DateOffset(days=3)
                                                                )


class HSI_Tb_StartMdrTreatment(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event - start tb-mdr treatment
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['TBFollowUp'] = 1

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg'] == 'Case management of MDR cases',
                'Intervention_Pkg_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Tb_MdrTreatmentInitiation'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = ['*']  # can occur at any facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug("We are now ready to treat this tb case %d", person_id)

        now = self.sim.date
        df = self.sim.population.props
        params = self.module.parameters

        # treatment allocated
        if df.at[person_id, 'is_alive'] and df.at[person_id, 'tb_diagnosed']:
            df.at[person_id, 'tb_treated_mdr'] = True
            df.at[person_id, 'date_tb_treated_mdr'] = now
            # TODO check all properties prefixed with tb_

        # schedule a 6-month event where people are cured, symptoms return to latent or not cured
        self.sim.schedule_event(TbCureEvent(self, person_id), self.sim.date + DateOffset(months=6))

        # follow-up appts
        logger.debug('....This is HSI_Tb_StartMdrTreatment: scheduling follow-up appointments for person %d',
                     person_id)

        # clinical monitoring
        followup_clin = HSI_Tb_FollowUp(self.module, person_id=person_id)

        # Request the health system to have this follow-up appointment
        weeks = params['followup_times']['mdr_clinical']
        cleanedList = [x for x in weeks if x == x]  # remove nan

        for i in range(0, len(cleanedList)):
            followup_appt_date = self.sim.date + DateOffset(months=cleanedList[i])
            self.sim.modules['HealthSystem'].schedule_hsi_event(followup_clin,
                                                                priority=1,
                                                                topen=followup_appt_date,
                                                                tclose=followup_appt_date + DateOffset(days=3)
                                                                )

        # repeat sputum smear tests
        followup_smear = HSI_Tb_FollowUp_SputumTest(self.module, person_id=person_id)

        # Request the health system to have this follow-up appointment
        weeks = params['followup_times']['mdr_sputum']
        cleanedList = [x for x in weeks if x == x]  # remove nan

        for i in range(0, len(cleanedList)):
            followup_appt_date = self.sim.date + DateOffset(months=cleanedList[i])
            self.sim.modules['HealthSystem'].schedule_hsi_event(followup_smear,
                                                                priority=1,
                                                                topen=followup_appt_date,
                                                                tclose=followup_appt_date + DateOffset(days=3)
                                                                )


class HSI_Tb_RetreatmentAdult(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event - start tb treatment
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['TBNew'] = 1  # This requires one out patient appt

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg'] == 'First line treatment for retreatment TB cases for adults',
                'Intervention_Pkg_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Tb_RetreatmentAdult'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = ['*']  # can occur at any facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug("We are now ready to treat this tb case %d", person_id)

        params = self.sim.modules['tb'].parameters
        now = self.sim.date
        df = self.sim.population.props

        # treatment allocated
        if df.at[person_id, 'is_alive'] and df.at[person_id, 'tb_diagnosed']:
            df.at[person_id, 'tb_on_treatment'] = True
            df.at[person_id, 'date_tb_treated'] = now

        # schedule a 6-month event where people are cured, symptoms return to latent or not cured
        self.sim.schedule_event(TbCureEvent(self, person_id), self.sim.date + DateOffset(months=6))

        # follow-up appts
        logger.debug('....This is HSI_Tb_RetreatmentAdult: scheduling follow-up appointments for person %d',
                     person_id)

        # clinical monitoring
        followup_clin = HSI_Tb_FollowUp(self.module, person_id=person_id)

        # Request the health system to have this follow-up appointment
        weeks = params['followup_times']['retreatment_clinical']
        cleanedList = [x for x in weeks if x == x]  # remove nan

        for i in range(0, len(cleanedList)):
            followup_appt_date = self.sim.date + DateOffset(months=cleanedList[i])
            self.sim.modules['HealthSystem'].schedule_hsi_event(followup_clin,
                                                                priority=1,
                                                                topen=followup_appt_date,
                                                                tclose=followup_appt_date + DateOffset(days=3)
                                                                )

        # repeat sputum smear tests
        followup_smear = HSI_Tb_FollowUp_SputumTest(self.module, person_id=person_id)

        # Request the health system to have this follow-up appointment
        weeks = params['followup_times']['retreatment_sputum']
        cleanedList = [x for x in weeks if x == x]  # remove nan

        for i in range(0, len(cleanedList)):
            followup_appt_date = self.sim.date + DateOffset(months=cleanedList[i])
            self.sim.modules['HealthSystem'].schedule_hsi_event(followup_smear,
                                                                priority=1,
                                                                topen=followup_appt_date,
                                                                tclose=followup_appt_date + DateOffset(days=3)
                                                                )

class HSI_Tb_RetreatmentChild(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event - start tb treatment
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['TBNew'] = 1  # New tb treatment initiation appt

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg'] == 'First line treatment for retreatment TB cases for children',
                'Intervention_Pkg_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Tb_RetreatmentChild'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = ['*']  # can occur at any facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug("We are now ready to treat this tb case %d", person_id)

        params = self.sim.modules['tb'].parameters
        now = self.sim.date
        df = self.sim.population.props

        # treatment allocated
        if df.at[person_id, 'is_alive'] and df.at[person_id, 'tb_diagnosed']:
            df.at[person_id, 'tb_on_treatment'] = True
            df.at[person_id, 'date_tb_treated'] = now

        # schedule a 6-month event where people are cured, symptoms return to latent or not cured
        self.sim.schedule_event(TbCureEvent(self, person_id), self.sim.date + DateOffset(months=6))

        # follow-up appts
        logger.debug('....This is HSI_Tb_RetreatmentChild: scheduling follow-up appointments for person %d',
                     person_id)

        # clinical monitoring
        followup_clin = HSI_Tb_FollowUp(self.module, person_id=person_id)

        # Request the health system to have this follow-up appointment
        weeks = params['followup_times']['retreatment_clinical']
        cleanedList = [x for x in weeks if x == x]  # remove nan

        for i in range(0, len(cleanedList)):
            followup_appt_date = self.sim.date + DateOffset(months=cleanedList[i])
            self.sim.modules['HealthSystem'].schedule_hsi_event(followup_clin,
                                                                priority=1,
                                                                topen=followup_appt_date,
                                                                tclose=followup_appt_date + DateOffset(days=3)
                                                                )

        # repeat sputum smear tests
        followup_smear = HSI_Tb_FollowUp_SputumTest(self.module, person_id=person_id)

        # Request the health system to have this follow-up appointment
        weeks = params['followup_times']['retreatment_sputum']
        cleanedList = [x for x in weeks if x == x]  # remove nan

        for i in range(0, len(cleanedList)):
            followup_appt_date = self.sim.date + DateOffset(months=cleanedList[i])
            self.sim.modules['HealthSystem'].schedule_hsi_event(followup_smear,
                                                                priority=1,
                                                                topen=followup_appt_date,
                                                                tclose=followup_appt_date + DateOffset(days=3)
                                                                )


# ---------------------------------------------------------------------------
#   Follow-up appts
# ---------------------------------------------------------------------------
class HSI_Tb_FollowUp(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event - start tb treatment
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['TBFollowUp'] = 1

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Tb_FollowUp'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ACCEPTED_FACILITY_LEVELS = ['*']  # can occur at any facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        # nothing needs to happen here, just log the appt
        logger.debug("Follow up appt for tb case %d", person_id)


class HSI_Tb_FollowUp_SputumTest(Event, IndividualScopeEventMixin):
    """
    This is a follow-up sputum test for confirmed tb cases
    doesn't change any properties except for date latest sputum test
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ConWithDCSA'] = 1  # This requires one generic outpatient appt
        the_appt_footprint['LabTBMicro'] = 1  # This requires one lab appt for microscopy

        # Get the consumables required
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = \
            pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Microscopy Test', 'Intervention_Pkg_Code'])[
                0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Tb_FollowUpSputumTest'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = ['*']  # can occur at any facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug('This is HSI_Tb_FollowUp_SputumTest, a follow-up sputum smear test for person %d', person_id)

        df = self.sim.population.props
        now = self.sim.date

        df.at[person_id, 'tb_ever_tested'] = True
        df.at[person_id, 'tb_smear_test'] = True
        df.at[person_id, 'tb_date_smear_test'] = now


# ---------------------------------------------------------------------------
#   Cure
# ---------------------------------------------------------------------------


class TbCureEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("Stopping tb treatment and curing person %d", person_id)

        df = self.sim.population.props
        params = self.sim.modules['tb'].parameters

        # after six months of treatment, stop
        df.at[person_id, 'tb_on_treatment'] = False

        # if drug-susceptible then probability of successful treatment for both primary and secondary
        if df.at[person_id, 'tb_inf'].startswith("active_susc"):

            cured = self.sim.rng.random_sample(size=1) < params['prob_treatment_success']

            if cured:
                df.at[person_id, 'tb_inf'] = 'latent_susc_tx'
                df.at[person_id, 'tb_diagnosed'] = False
                df.loc[person_id, 'tb_specific_symptoms'] = 'latent'
                df.loc[person_id, 'tb_unified_symptom_code'] = 1

            else:
                # request a repeat / Xpert test - follow-up
                # this will include drug-susceptible treatment failures and mdr-tb cases
                secondary_test = HSI_Tb_XpertTest(self.module, person_id=person_id)

                # Request the health system to give xpert test
                self.sim.modules['HealthSystem'].schedule_hsi_event(secondary_test,
                                                                    priority=1,
                                                                    topen=self.sim.date,
                                                                    tclose=None)

                # add back-up check if xpert is not available, then schedule sputum smear
                self.sim.schedule_event(TbCheckXpert(self, person_id), self.sim.date + DateOffset(weeks=2))




class TbCureMdrEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("Stopping tb-mdr treatment and curing person %d", person_id)

        df = self.sim.population.props

        # after six months of treatment, stop
        # assume 100% cure rate with tb-mdr treatment
        df.at[person_id, 'tb_treated_mdr'] = False
        df.at[person_id, 'tb_inf'] = 'latent_mdr_tx'
        df.at[person_id, 'tb_diagnosed'] = False
        df.loc[person_id, 'tb_specific_symptoms'] = 'latent'
        df.loc[person_id, 'tb_unified_symptom_code'] = 1


#
# ---------------------------------------------------------------------------
#   IPT
# ---------------------------------------------------------------------------
# TODO consumables should be IPT for non-HIV+
# TODO also IPT for infants of smear-positive mothers
class HSI_Tb_Ipt(Event, IndividualScopeEventMixin):
    """
        This is a Health System Interaction Event - give ipt to contacts of tb cases for 6 months
        """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient appt

        # TODO IPT from OneHealth returns intervention package code -99
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Items'] == 'Isoniazid Preventive Therapy',
                'Intervention_Pkg_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Tb_Ipt'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = ['*']  # can occur at any facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug("Starting IPT for person %d", person_id)

        df = self.sim.population.props

        df.at[person_id, 'tb_on_ipt'] = True
        df.at[person_id, 'tb_date_ipt'] = self.sim.date

        # schedule end date of ipt after six months
        self.sim.schedule_event(TbIptEndEvent(self, person_id), self.sim.date + DateOffset(months=6))


class HSI_Tb_IptHiv(Event, IndividualScopeEventMixin):
    """
        This is a Health System Interaction Event - give ipt to hiv+ persons
        called by hiv module when starting ART (adults and children)
        """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient appt

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg'] == 'Isoniazid preventative therapy for HIV+ no TB',
                'Intervention_Pkg_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Tb_IptHiv'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = ['*']  # can occur at any facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug("Starting IPT for HIV+ person %d", person_id)

        df = self.sim.population.props

        df.at[person_id, 'tb_on_ipt'] = True
        df.at[person_id, 'tb_date_ipt'] = self.sim.date

        # schedule end date of ipt after six months and repeat call to HS for another prescription
        self.sim.schedule_event(TbIptEndEvent(self, person_id), self.sim.date + DateOffset(months=6))


class TbIptEndEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("Stopping ipt for person %d", person_id)

        df = self.sim.population.props

        df.at[person_id, 'tb_on_ipt'] = False

        # if hiv+ reschedule HSI_Tb_IptHiv to continue IPT
        if df.at[person_id, 'hv_inf']:
            logger.debug(
                '....This is TbIptEndEvent: scheduling further IPT for person %d on date %s',
                person_id, self.sim.date)

            ipt_start = HSI_Tb_IptHiv(self.module, person_id=person_id)

            # Request the health system to have this follow-up appointment
            self.sim.modules['HealthSystem'].schedule_hsi_event(ipt_start,
                                                                priority=1,
                                                                topen=self.sim.date,
                                                                tclose=None
                                                                )


# ---------------------------------------------------------------------------
#   Deaths
# ---------------------------------------------------------------------------
class TbDeathEvent(RegularEvent, PopulationScopeEventMixin):
    """The regular event that kills people.
    """

    # TODO: if HIV+, cause of death should be HIV as hiv/tb deaths are counted in hiv data
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        params = self.module.parameters
        df = population.props
        now = self.sim.date
        rng = self.module.rng

        # only active infections result in death, no deaths on treatment
        mortality_rate = pd.Series(0, index=df.index)

        mortality_rate.loc[df['tb_inf'].str.contains('active') & ~df.hv_inf & (
            ~df.tb_on_treatment | ~df.tb_treated_mdr)] = params[
            'monthly_prob_tb_mortality']

        mortality_rate.loc[df['tb_inf'].str.contains('active') & df.hv_inf & (
            ~df.tb_on_treatment | ~df.tb_treated_mdr)] = params[
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
            df[df['tb_inf'].str.contains('active_susc') & df.is_alive])

        active_mdr = len(
            df[df['tb_inf'].str.contains('active_mdr') & df.is_alive])

        active_total = active_susc + active_mdr

        ad_prev = len(df[df['tb_inf'].str.contains('active') & df.is_alive & (
            df.age_years >= 15)]) / len(df[df.is_alive & (df.age_years >= 15)])

        child_prev = len(df[df['tb_inf'].str.contains('active') & df.is_alive & (
            df.age_years < 15)]) / len(df[df.is_alive & (df.age_years < 15)])

        ad_prev_latent = len(df[df['tb_inf'].str.contains('latent') & df.is_alive & (
            df.age_years >= 15)]) / len(df[df.is_alive & (df.age_years >= 15)])

        child_prev_latent = len(df[df['tb_inf'].str.contains('latent') & df.is_alive & (
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
