'''
The TB Module is under development. Here the contents of the module are commented out whilst work proceeds. It is
included because this version should replace what is currently in Master.
'''

import os

import numpy as np
import pandas as pd

from tlo import DAYS_IN_YEAR, DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, demography
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom
from tlo.util import create_age_range_lookup
from tlo.methods import hiv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Tb(Module):
    """ Set up the baseline population with TB prevalence
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)

        self.resourcefilepath = resourcefilepath
        self.daly_wts = dict()
        self.lm = dict()
        self.footprints_for_consumables_required = dict()
        self.symptom_list = {"fever", "respiratory_symptoms", "fatigue", "night_sweats"}

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    PROPERTIES = {
        # ------------------ natural history ------------------ #
        'tb_inf': Property(
            Types.CATEGORICAL,
            categories=[
                'uninfected',
                'latent',
                'active',
            ],
            description='tb status',
        ),
        'tb_strain': Property(
            Types.CATEGORICAL,
            categories=[
                'none',
                'ds',
                'mdr',
            ],
            description='tb strain: drug-susceptible (ds) or multi-drug resistant (mdr)',
        ),
        'tb_date_latent': Property(
            Types.DATE, 'Date acquired tb infection (latent stage)'
        ),
        'tb_date_active': Property(Types.DATE, 'Date active tb started'),
        'tb_smear': Property(Types.BOOL, 'smear positivity with active infection: False=negative, True=positive'),
        # 'tb_stage': Property(
        #     Types.CATEGORICAL,
        #     'Level of symptoms for tb',
        #     categories=['none', 'latent', 'active_pulm', 'active_extra'],
        # ),
        'tb_date_death': Property(Types.DATE, 'date of scheduled tb death'),

        # ------------------ testing status ------------------ #
        'tb_ever_tested': Property(Types.BOOL, 'ever had a tb test'),
        # 'tb_smear_test': Property(Types.BOOL, 'ever had a tb smear test'),
        # 'tb_result_smear_test': Property(Types.BOOL, 'result from tb smear test'),
        # 'tb_date_smear_test': Property(Types.DATE, 'date of tb smear test'),
        # 'tb_xpert_test': Property(Types.BOOL, 'ever had a tb Xpert test'),
        # 'tb_result_xpert_test': Property(Types.BOOL, 'result from tb Xpert test'),
        # 'tb_date_xpert_test': Property(Types.DATE, 'date of tb Xpert test'),
        'tb_diagnosed': Property(Types.BOOL, 'person has current diagnosis of active tb'),
        'tb_diagnosed_mdr': Property(Types.BOOL, 'person has current diagnosis of active mdr-tb'),

        # ------------------ treatment status ------------------ #
        'tb_on_treatment': Property(Types.BOOL, 'on tb treatment regimen'),
        'tb_date_treated': Property(Types.DATE, 'date tb treatment started'),
        'tb_ever_treated': Property(Types.BOOL, 'if ever treated for active tb'),
        'tb_treatment_failure': Property(Types.BOOL, 'failed first line tb treatment'),
        'tb_treated_mdr': Property(Types.BOOL, 'on tb treatment MDR regimen'),
        'tb_date_treated_mdr': Property(Types.DATE, 'date tb MDR treatment started'),

        'tb_on_ipt': Property(Types.BOOL, 'if currently on ipt'),
        'tb_date_ipt': Property(Types.DATE, 'date ipt started')
    }

    PARAMETERS = {

        # ------------------ workbooks ------------------ #
        'prop_latent_2010': Parameter(
            Types.REAL, 'Proportion of population with latent tb in 2010'
        ),
        'prop_active_2010': Parameter(
            Types.REAL, 'Proportion of population with active tb in 2010'
        ),
        'pulm_tb': Parameter(Types.REAL, 'probability of pulmonary tb'),
        'followup_times': Parameter(
            Types.INT, 'times(weeks) tb treatment monitoring required after tx start'
        ),
        'tb_high_risk_distr': Parameter(Types.LIST, 'list of ten high-risk districts'),
        'ipt_contact_cov': Parameter(
            Types.REAL,
            'coverage of IPT among contacts of TB cases in high-risk districts',
        ),
        'bcg_coverage_year': Parameter(
            Types.REAL, 'bcg coverage estimates in children <1 years by calendar year'
        ),
        'initial_bcg_coverage': Parameter(
            Types.REAL, 'bcg coverage by age in baseline population'
        ),

        # ------------------ baseline population ------------------ #
        'prop_mdr2010': Parameter(
            Types.REAL,
            'Proportion of active tb cases with multidrug resistance in 2010',
        ),

        # ------------------ natural history ------------------ #
        # 'prob_latent_tb_0_14': Parameter(Types.REAL, 'probability of latent infection in ages 0-14 years'),
        # 'prob_latent_tb_15plus': Parameter(Types.REAL, 'probability of latent infection in ages 15+'),
        'transmission_rate': Parameter(Types.REAL, 'TB transmission rate, calibrated'),
        'rel_inf_smear_ng': Parameter(
            Types.REAL, 'relative infectiousness of tb in hiv+ compared with hiv-'
        ),
        'rel_inf_poor_tx': Parameter(
            Types.REAL,
            'relative infectiousness of tb in treated people with poor adherence',
        ),
        'rr_bcg_inf': Parameter(
            Types.REAL, 'relative risk of tb infection with bcg vaccination'
        ),
        'monthly_prob_relapse_tx_complete': Parameter(
            Types.REAL, 'monthly probability of relapse once treatment complete'
        ),
        'monthly_prob_relapse_tx_incomplete': Parameter(
            Types.REAL, 'monthly probability of relapse if treatment incomplete'
        ),
        'monthly_prob_relapse_2yrs': Parameter(
            Types.REAL,
            'monthly probability of relapse 2 years after treatment complete',
        ),
        'rr_relapse_hiv': Parameter(
            Types.REAL, 'relative risk of relapse for HIV-positive people'
        ),
        'r_trans_mdr': Parameter(
            Types.REAL,
            'relative transmissibility of MDR compared with drug-susceptible TB',
        ),
        'p_mdr_new': Parameter(
            Types.REAL,
            'probability of MDR emergence per treatment course – new patients',
        ),
        'p_mdr_retreat': Parameter(
            Types.REAL,
            'multiplier for probability of MDR emergence per treatment course – re-treatment',
        ),
        'p_mdr_tx_fail': Parameter(
            Types.REAL,
            'multiplier for probability of MDR emergence per treatment course – treatment failure',
        ),

        # ------------------ progression ------------------ #
        'prop_fast_progressor': Parameter(
            Types.REAL,
            'Proportion of infections that progress directly to active stage',
        ),
        'prop_fast_progressor_hiv': Parameter(
            Types.REAL,
            'proportion of HIV+ people not on ART progressing directly to active TB disease after infection',
        ),
        'prog_active': Parameter(
            Types.REAL, 'risk of progressing to active tb within two years'
        ),
        'prog_1yr': Parameter(
            Types.REAL, 'proportion children aged <1 year progressing to active disease'
        ),
        'prog_1_2yr': Parameter(
            Types.REAL,
            'proportion children aged 1-2 year2 progressing to active disease',
        ),
        'prog_2_5yr': Parameter(
            Types.REAL,
            'proportion children aged 2-5 years progressing to active disease',
        ),
        'prog_5_10yr': Parameter(
            Types.REAL,
            'proportion children aged 5-10 years progressing to active disease',
        ),
        'prog_10yr': Parameter(
            Types.REAL,
            'proportion children aged 10-15 years progressing to active disease',
        ),
        'monthly_prob_self_cure': Parameter(
            Types.REAL, 'monthly probability of self-cure'
        ),
        'monthly_prob_self_cure_hiv': Parameter(
            Types.REAL, 'monthly probability of self-cure in plhiv'
        ),

        # ------------------ clinical features ------------------ #
        'prop_smear_positive': Parameter(
            Types.REAL, 'proportion of new active cases that will be smear-positive'
        ),
        'prop_smear_positive_hiv': Parameter(
            Types.REAL, 'proportion of hiv+ active tb cases that will be smear-positive'
        ),

        # ------------------ mortality ------------------ #
        'monthly_prob_tb_mortality': Parameter(
            Types.REAL, 'mortality rate with active tb'
        ),
        'monthly_prob_tb_mortality_hiv': Parameter(
            Types.REAL, 'mortality from tb with concurrent HIV'
        ),
        'mort_cotrim': Parameter(
            Types.REAL, 'reduction in mortality rates due to cotrimoxazole prophylaxis'
        ),
        'mort_tx': Parameter(
            Types.REAL, 'reduction in mortality rates due to effective tb treatment'
        ),

        # ------------------ progression to active disease ------------------ #
        'rr_tb_bcg': Parameter(
            Types.REAL,
            'relative risk of progression to active disease for children with BCG vaccine',
        ),
        'rr_tb_hiv': Parameter(
            Types.REAL, 'relative risk of progression to active disease for PLHIV'
        ),
        'rr_tb_aids': Parameter(
            Types.REAL,
            'relative risk of progression to active disease for PLHIV with AIDS',
        ),
        'rr_tb_art_adult': Parameter(
            Types.REAL,
            'relative risk of progression to active disease for adults with HIV on ART',
        ),
        'rr_tb_art_child': Parameter(
            Types.REAL,
            'relative risk of progression to active disease for adults with HIV on ART',
        ),
        'rr_tb_overweight': Parameter(
            Types.REAL, 'relative risk of progression to active disease if overweight'
        ),
        'rr_tb_obese': Parameter(
            Types.REAL, 'relative risk of progression to active disease if obese'
        ),
        'rr_tb_diabetes1': Parameter(
            Types.REAL,
            'relative risk of progression to active disease with type 1 diabetes',
        ),
        'rr_tb_alcohol': Parameter(
            Types.REAL,
            'relative risk of progression to active disease with heavy alcohol use',
        ),
        'rr_tb_smoking': Parameter(
            Types.REAL, 'relative risk of progression to active disease with smoking'
        ),
        'dur_prot_ipt': Parameter(
            Types.REAL,
            'duration in days of protection conferred by IPT against active TB',
        ),
        'dur_prot_ipt_infant': Parameter(
            Types.REAL,
            'duration days of protection conferred by IPT against active TB in infants',
        ),
        'rr_ipt_adult': Parameter(
            Types.REAL, 'relative risk of active TB with IPT in adults'
        ),
        'rr_ipt_child': Parameter(
            Types.REAL, 'relative risk of active TB with IPT in children'
        ),
        'rr_ipt_adult_hiv': Parameter(
            Types.REAL, 'relative risk of active TB with IPT in adults with hiv'
        ),
        'rr_ipt_child_hiv': Parameter(
            Types.REAL, 'relative risk of active TB with IPT in children with hiv'
        ),
        'rr_ipt_art_adult': Parameter(
            Types.REAL, 'relative risk of active TB with IPT and ART in adults'
        ),
        'rr_ipt_art_child': Parameter(
            Types.REAL, 'relative risk of active TB with IPT and ART in children'
        ),

        # ------------------ health system parameters ------------------ #
        'sens_xpert': Parameter(Types.REAL, 'sensitivity of Xpert test'),
        'sens_sputum_pos': Parameter(
            Types.REAL,
            'sensitivity of sputum smear microscopy in sputum positive cases',
        ),
        'sens_sputum_neg': Parameter(
            Types.REAL,
            'sensitivity of sputum smear microscopy in sputum negative cases',
        ),
        'sens_clinical': Parameter(
            Types.REAL, 'sensitivity of clinical diagnosis in detecting active TB'
        ),
        'spec_clinical': Parameter(
            Types.REAL, 'specificity of clinical diagnosis in detecting TB'
        ),
        'prob_tx_success_new': Parameter(
            Types.REAL, 'Probability of treatment success for new TB cases'
        ),
        'prob_tx_success_prev': Parameter(
            Types.REAL, 'Probability of treatment success for previously treated cases'
        ),
        'prob_tx_success_hiv': Parameter(
            Types.REAL, 'Probability of treatment success for PLHIV'
        ),
        'prob_tx_success_mdr': Parameter(
            Types.REAL, 'Probability of treatment success for MDR-TB cases'
        ),
        'prob_tx_success_extra': Parameter(
            Types.REAL, 'Probability of treatment success for extrapulmonary TB cases'
        ),
        'prob_tx_success_0_4': Parameter(
            Types.REAL, 'Probability of treatment success for children aged 0-4 years'
        ),
        'prob_tx_success_5_14': Parameter(
            Types.REAL, 'Probability of treatment success for children aged 5-14 years'
        ),
        'prop_ltfu_tx': Parameter(
            Types.REAL, 'Proportion lost to follow-up during initial treatment'
        ),
        'prop_ltfu_retx': Parameter(
            Types.REAL, 'Proportion lost to follow-up during retreatment'
        ),
        'rate_testing_tb': Parameter(
            Types.REAL,
            'rate of presenting for TB screening and testing for people with active TB',
        ),
        'rr_testing_non_tb': Parameter(
            Types.REAL,
            'rate ratio for TB testing without active TB compared with ative TB cases',
        ),
        'rate_testing_hiv': Parameter(
            Types.REAL,
            'rate of presenting for TB screening and testing for HIV-positive people with active TB',
        ),
        'presump_testing': Parameter(
            Types.REAL, 'probability of an individual without tb requesting tb test'
        ),
        # 'prob_latent_tb_0_14': Parameter(
        #     Types.REAL, 'probability of latent tb in child aged 0-14 years'
        # ),
        # 'prob_latent_tb_15plus': Parameter(
        #     Types.REAL, 'probability of latent tb in adult aged 15 plus'
        # ),
        'ds_treatment_length': Parameter(
            Types.REAL, 'length of treatment for drug-susceptible tb (first case) in days'
        ),
        'ds_retreatment_length': Parameter(
            Types.REAL, 'length of treatment for drug-susceptible tb (secondary case) in days'
        ),
        'mdr_treatment_length': Parameter(
            Types.REAL, 'length of treatment for mdr-tb in days'
        ),

        # ------------------ daly weights ------------------ #
        # no daly for latent tb
        'daly_wt_susc_tb': Parameter(
            Types.REAL, 'Drug-susecptible tuberculosis, not HIV infected'
        ),
        'daly_wt_resistant_tb': Parameter(
            Types.REAL, 'multidrug-resistant tuberculosis, not HIV infected'
        ),
        'daly_wt_susc_tb_hiv_severe_anaemia': Parameter(
            Types.REAL,
            '# Drug-susecptible Tuberculosis, HIV infected and anemia, severe',
        ),
        'daly_wt_susc_tb_hiv_moderate_anaemia': Parameter(
            Types.REAL,
            'Drug-susecptible Tuberculosis, HIV infected and anemia, moderate',
        ),
        'daly_wt_susc_tb_hiv_mild_anaemia': Parameter(
            Types.REAL, 'Drug-susecptible Tuberculosis, HIV infected and anemia, mild'
        ),
        'daly_wt_susc_tb_hiv': Parameter(
            Types.REAL, 'Drug-susecptible Tuberculosis, HIV infected'
        ),
        'daly_wt_resistant_tb_hiv_severe_anaemia': Parameter(
            Types.REAL,
            'Multidrug resistant Tuberculosis, HIV infected and anemia, severe',
        ),
        'daly_wt_resistant_tb_hiv': Parameter(
            Types.REAL, 'Multidrug resistant Tuberculosis, HIV infected'
        ),
        'daly_wt_resistant_tb_hiv_moderate_anaemia': Parameter(
            Types.REAL,
            'Multidrug resistant Tuberculosis, HIV infected and anemia, moderate',
        ),
        'daly_wt_resistant_tb_hiv_mild_anaemia': Parameter(
            Types.REAL,
            'Multidrug resistant Tuberculosis, HIV infected and anemia, mild',
        ),
    }

    # cough and fever are part of generic symptoms
    SYMPTOMS = {'fatigue', 'night_sweats'}

    def read_parameters(self, data_folder):
        """
        * 1) Reads the ResourceFiles
        * 2) Declares the DALY weights
        * 3) Declares the Symptoms
        * 4) Defines the linear models
        """

        # 1) Read the ResourceFiles
        workbook = pd.read_excel(
            os.path.join(self.resourcefilepath, 'ResourceFile_TB.xlsx'), sheet_name=None
        )
        self.load_parameters_from_dataframe(workbook['parameters'])

        p = self.parameters

        p['prop_active_2010'] = workbook['cases2010district']

        p['pulm_tb'] = workbook['pulm_tb']
        p['followup_times'] = workbook['followup']
        p['tb_high_risk_distr'] = workbook['IPTdistricts']
        p['ipt_contact_cov'] = workbook['ipt_coverage']
        p['bcg_coverage_year'] = workbook['BCG']
        p['initial_bcg_coverage'] = workbook['BCG_baseline']

        # 2) Get the DALY weights
        if 'HealthBurden' in self.sim.modules.keys():
            # HIV-negative
            # Drug-susceptible tuberculosis, not HIV infected
            self.daly_wts['daly_tb'] = self.sim.modules['HealthBurden'].get_daly_weight(0)
            # multi-drug resistant tuberculosis, not HIV infected
            self.daly_wts['daly_mdr_tb'] = self.sim.modules['HealthBurden'].get_daly_weight(1)

            # HIV-positive
            # Drug-susceptible Tuberculosis, HIV infected and anemia, moderate
            self.daly_wts['daly_tb_hiv_anaemia'] = self.sim.modules['HealthBurden'].get_daly_weight(5)
            # Multi-drug resistant Tuberculosis, HIV infected and anemia, moderate
            self.daly_wts['daly_mdr_tb_hiv_anaemia'] = self.sim.modules['HealthBurden'].get_daly_weight(10)

        # 3) Declare the Symptoms
        # no additional healthcare-seeking behaviour with these symptoms
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='fatigue')
        )
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='night_sweats')
        )

        # 4) Define the linear models
        # linear model for risk of latent tb in baseline population 2010
        # latent tb risk not affected by bcg
        # assumes intercept=1
        self.lm['latent_tb_2010'] = LinearModel.multiplicative(
            Predictor('age_years').when('<15', p["prob_latent_tb_0_14"]).otherwise(p["prob_latent_tb_15plus"]),
        )

        # linear model for relative risk of active tb infection
        # intercept= prog_active

        # adults progressing to active disease
        self.lm['active_tb'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p["prog_active"],
            Predictor('va_bcg').when(True, p['rr_tb_bcg']),
            Predictor('hv_inf').when(True, p['rr_tb_hiv']),
            Predictor('sy_aids_symptoms').when(True, p['rr_tb_aids']),
            Predictor('hv_art').when("on_VL_suppressed", p['rr_tb_art_adult']),
            Predictor('li_bmi').when('>=4', p['rr_tb_art_adult']),
            # Predictor('diabetes').when(True, p['rr_tb_diabetes1']),
            Predictor('li_ex_alc').when(True, p['rr_tb_alcohol']),
            Predictor('li_tob').when(True, p['rr_tb_smoking']),
            Predictor().when(
                '(tb_on_ipt == True) & (hv_inf == False)', p['rr_ipt_adult']),  # ipt, hiv-
            Predictor().when(
                '(tb_on_ipt == True) & (hv_inf == True) & (hv_art == "on_VL_suppressed")',
                p['rr_ipt_art_adult']),  # ipt, hiv+ on ART (suppressed)
            Predictor().when(
                '(tb_on_ipt == True) & (hv_inf == True) & (hv_art != "on_VL_suppressed")',
                p['rr_ipt_adult_hiv']),  # ipt, hiv+ not on ART (or on ART and not suppressed)

        )

        # children progressing to active disease
        self.lm['active_tb_child'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1,
            Predictor('age_years').when('<1', p['prog_1yr'])
                .when('<2', p['prog_1_2yr'])
                .when('<5', p['prog_2_5yr'])
                .when('<10', p['prog_5_10yr'])
                .when('<15', p['prog_10yr']),
            Predictor().when(
                'va_bcg & hv_inf & (age_years <10)',
                p['rr_tb_bcg']),
            Predictor('hv_art').when("on_VL_suppressed", p['rr_tb_art_child']),
            Predictor().when(
                '(tb_on_ipt) & (hv_inf == False)', p['rr_ipt_child']),  # ipt, hiv-
            Predictor().when(
                '(tb_on_ipt == True) & (hv_inf == True) & (hv_art == "on_VL_suppressed")',
                p['rr_ipt_art_child']),  # ipt, hiv+ on ART (suppressed)
            Predictor().when(
                '(tb_on_ipt == True) & (hv_inf == True) & (hv_art != "on_VL_suppressed")',
                p['rr_ipt_child_hiv']),  # ipt, hiv+ not on ART (or on ART and not suppressed)
        )

        # individual risk of relapse
        self.lm['risk_relapse'] = LinearModel(
            LinearModelType.ADDITIVE,
            0,
            Predictor().when(
                '(tb_inf == "latent") & '
                'tb_ever_treated & '
                '~tb_treatment_failure & '
                '(self.sim.date < (tb_date_treated + pd.DateOffset(days=2*365.25)))',
                p['monthly_prob_relapse_tx_complete']),  # ever treated, no tx failure and <2 years post active disease
            Predictor().when(
                '(tb_inf == "latent") & '
                'tb_ever_treated & '
                'tb_treatment_failure & '
                '(self.sim.date < (tb_date_treated + pd.DateOffset(days=2*365.25)))',
                p['monthly_prob_relapse_tx_incomplete']),  # ever treated, tx failure and <2 years post active disease
            Predictor().when(
                '(tb_inf == "latent") & '
                'tb_ever_treated & '
                '(self.sim.date >= (tb_date_treated + pd.DateOffset(days=2*365.25)))',
                p['monthly_prob_relapse_2yrs']),  # <2 years post active disease
        )

    def baseline_latent(self, population):
        """
        sample from the baseline population to assign latent infections
        using 2010 prevalence estimates
        no differences in latent tb by sex
        """

        df = population.props
        now = self.sim.date
        p = self.parameters

        # whole population susceptible to latent infection, risk determined by age
        prob_latent = self.lm['latent_tb_2010'].predict(
            df.loc[df.is_alive]
        )  # this will return pd.Series of probabilities of latent infection for each person alive

        new_latent = self.rng.random_sample(len(prob_latent)) < prob_latent
        idx_new_latent = new_latent[new_latent].index

        df.loc[idx_new_latent, 'tb_inf'] = 'latent'
        df.loc[idx_new_latent, 'tb_strain'] = 'ds'
        df.loc[idx_new_latent, 'tb_date_latent'] = now

        # allocate some latent infections as mdr-tb
        if len(df[df.is_alive & (df.tb_inf == 'latent')]) > 10:
            idx_new_latent_mdr = (
                df[df.is_alive & (df.tb_inf == 'latent')]
                    .sample(frac=p['prop_mdr2010'])
                    .index
            )

            df.loc[idx_new_latent_mdr, 'tb_strain'] = 'mdr'

    def progression_to_active(self, population):
        # from the new latent infections, select and schedule progression to active disease

        df = population.props
        p = self.parameters
        rng = self.rng
        now = self.sim.date

        # ------------------ fast progressors ------------------ #
        # adults only
        fast = df.loc[(df.tb_date_latent == now) &
                      df.is_alive &
                      (df.age_years >= 15) &
                      ~df.hv_inf &
                      (rng.rand() < p['prop_fast_progressor'])].index

        fast_hiv = df.loc[(df.tb_date_latent == now) &
                          df.is_alive &
                          (df.age_years >= 15) &
                          df.hv_inf &
                          (rng.rand() < p['prop_fast_progressor_hiv'])].index

        all_fast = fast.union(fast_hiv)  # join indices (checked)
        print(all_fast)

        for person in all_fast:
            self.sim.schedule_event(TbActiveEvent(self, person), now)

        # ------------------ slow progressors ------------------ #

        # slow progressors, based on risk factors (via a linear model)
        # select population eligible for progression to active disease
        # includes all new latent infections
        # excludes those just fast-tracked above (all_fast)

        # adults
        eligible_adults = df.loc[(df.tb_date_latent == now) &
                                 df.is_alive &
                                 (df.age_years >= 15)].index
        eligible_adults = eligible_adults[~np.isin(eligible_adults, all_fast)]

        # check no fast progressors included in the slow progressors risk
        assert not any(elem in all_fast for elem in eligible_adults)

        risk_of_progression = self.lm['active_tb'].predict(df.loc[eligible_adults])
        will_progress = self.rng.random_sample(len(risk_of_progression)) < risk_of_progression
        idx_will_progress = will_progress[will_progress].index

        # schedule for time now up to 2 years
        for person_id in idx_will_progress:
            date_progression = self.sim.date + \
                               pd.DateOffset(days=self.rng.randint(0, 732))
            self.sim.schedule_event(
                TbActiveEvent(self, person_id), date_progression
            )

        # children
        eligible_children = df.loc[(df.tb_date_latent == now) &
                                   df.is_alive &
                                   (df.age_years < 15)].index
        eligible_children = eligible_children[~np.isin(eligible_children, all_fast)]
        assert not any(elem in all_fast for elem in eligible_children)

        risk_of_progression = self.lm['active_tb_child'].predict(df.loc[eligible_children])
        will_progress = self.rng.random_sample(len(risk_of_progression)) < risk_of_progression
        idx_will_progress = will_progress[will_progress].index

        # schedule for time now up to 1 year
        for person_id in idx_will_progress:
            date_progression = self.sim.date + \
                               pd.DateOffset(days=self.rng.randint(0, 365))
            self.sim.schedule_event(
                TbActiveEvent(self, person_id), date_progression
            )

    def select_treatment(self, person_id):
        """
        helper function to select appropriate treatment and check whether
        consumables are available to start drug course
        treatment will always be for ds-tb unless mdr has been identified
        :return: drug_available [BOOL]
        """
        df = self.sim.population.props
        person = df.loc[person_id]

        drugs_available = False  # default return value

        # -------- MDR-TB -------- #

        if person['tb_diagnosed_mdr']:

            drugs_available = self.get_all_consumables(
                footprint=self.footprints_for_consumables_required['tb_mdrtx'])

        # -------- First TB infection -------- #
        # could be undiagnosed mdr or ds-tb: treat as ds-tb

        elif not person['tb_ever_treated']:

            if df.at['age_years'] >= 15:
                # treatment for ds-tb: adult
                drugs_available = self.get_all_consumables(
                    footprint=self.footprints_for_consumables_required['tb_tx_adult'])
            else:
                # treatment for ds-tb: child
                drugs_available = self.get_all_consumables(
                    footprint=self.footprints_for_consumables_required['tb_tx_child'])

        # -------- Secondary TB infection -------- #
        # person has been treated before
        # possible treatment failure or subsequent reinfection
        elif person['tb_ever_treated']:

            if df.at['age_years'] >= 15:
                # treatment for reinfection ds-tb: adult
                drugs_available = self.get_all_consumables(
                    footprint=self.footprints_for_consumables_required['tb_retx_adult'])
            else:
                # treatment for reinfection ds-tb: child
                drugs_available = self.get_all_consumables(
                    footprint=self.footprints_for_consumables_required['tb_retx_child'])

        return drugs_available

    def clinical_monitoring(self, person_id):
        """
        schedule appointments for repeat clinical monitoring events
        treatment given using DOTS strategy, this can be health worker/guardian/community member
        therefore no HSI event for DOTS

        :param person_id:
        :return:
        """
        df = self.sim.population.props
        p = self.parameters

        # default clinical monitoring schedule for first infection ds-tb
        clinical_fup = p['followup_times'].loc['ds_clinical_monitor']

        # if previously treated:
        if df.at[person_id, 'tb_ever_treated']:

            # if strain is ds and person previously treated:
            clinical_fup = p['followup_times'].loc['ds_retreatment_clinical']

        # if strain is mdr - this treatment schedule takes precedence
        elif df.at[person_id, 'tb_strain'] == 'mdr':

            # if strain is mdr:
            clinical_fup = p['followup_times'].loc['mdr_clinical']

        # todo does this use the right value for appt?
        for appt in clinical_fup:
            print(appt)
            # schedule a clinical check-up appointment
            date_appt = self.sim.date + \
                        pd.DateOffset(days=appt * 30.5)
            self.sim.schedule_event(
                HSI_Tb_FollowUp(self, person_id), date_appt
            )

    # def end_treatment(self, person_id):
    #     """
    #     end treatment (any type) for person
    #     and reset individual properties
    #     if person has died, no further action
    #
    #     """
    #     df = self.sim.population.props
    #
    #     if not df.at[person_id, "is_alive"]:
    #         return
    #
    #     if df.at[person_id, 'tb_on_treatment']:
    #         df.at[person_id, 'tb_on_treatment'] = False
    #     if df.at[person_id, 'tb_treated_mdr']:
    #         df.at[person_id, 'tb_treated_mdr'] = False
    #     if df.at[person_id, 'tb_on_ipt']:
    #         df.at[person_id, 'tb_on_ipt'] = False

    def initialise_population(self, population):

        # Set our property values for the initial population
        df = population.props

        df['tb_inf'].values[:] = 'uninfected'
        df['tb_strain'].values[:] = 'none'

        df['tb_date_latent'] = pd.NaT
        df['tb_date_active'] = pd.NaT
        df['tb_smear'] = False
        # df['tb_stage'].values[:] = 'none'
        df['tb_date_death'] = pd.NaT

        # ------------------ testing status ------------------ #
        df['tb_ever_tested'] = False
        df['tb_diagnosed'] = False
        df['tb_diagnosed_mdr'] = False

        # ------------------ treatment status ------------------ #
        df['tb_on_treatment'] = False
        df['tb_date_treated'] = pd.NaT
        df['tb_ever_treated'] = False
        df['tb_treatment_failure'] = False

        df['tb_on_ipt'] = False
        df['tb_date_ipt'] = pd.NaT

        # ------------------ infection status ------------------ #

        self.baseline_latent(population)  # allocate baseline prevalence of latent infections
        self.progression_to_active(population)  # allocate active infections from baseline prevalence

    def initialise_simulation(self, sim):
        """
        * 1) Schedule the Main TB Regular Polling Event
        * 2) Schedule the Logging Event
        * 3) Define the DxTests
        * 4) Define the treatment options
        """

        # 1) Regular events
        sim.schedule_event(TbRegularPollingEvent(self), sim.date + DateOffset(days=0))
        sim.schedule_event(TbEndTreatmentEvent(self), sim.date + DateOffset(days=30.5))
        sim.schedule_event(TbRelapseEvent(self), sim.date + DateOffset(months=1))
        sim.schedule_event(TbSelfCureEvent(self), sim.date + DateOffset(months=1))

        # 2) Logging
        sim.schedule_event(TbLoggingEvent(self), sim.date + DateOffset(days=0))

        # 3) -------- Define the DxTests and get the consumables required --------

        p = self.parameters
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # TB Sputum smear test
        # assume that if smear-positive, sputum smear test is 100% specific and sensitive
        pkg_sputum = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg'] == 'Microscopy Test',
                'Intervention_Pkg_Code',
            ]
        )[0]

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            tb_sputum_test=DxTest(
                property='tb_smear',
                sensitivity=1.0,
                specificity=1.0,
                cons_req_as_footprint={'Intervention_Package_Code': {pkg_sputum: 1}, 'Item_Code': {}}
            )
        )

        # TB GeneXpert
        pkg_xpert = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg'] == 'Xpert test', 'Intervention_Pkg_Code'
            ]
        )[0]

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            tb_xpert_test=DxTest(
                property='tb_inf',
                target_categories=['active'],
                sensitivity=p['sens_xpert'],
                specificity=1.0,
                cons_req_as_footprint={'Intervention_Package_Code': {pkg_xpert: 1}, 'Item_Code': {}}
            )
        )

        # TB Chest x-ray
        pkg_xray = pd.unique(
            consumables.loc[consumables['Item_Code'] == 175, 'Intervention_Pkg_Code']
        )[0]

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            tb_xray=DxTest(
                property='tb_inf',
                target_categories=['active'],
                sensitivity=1.0,
                specificity=1.0,
                cons_req_as_footprint={'Intervention_Package_Code': {pkg_xray: 1}, 'Item_Code': {}}
            )
        )

        # 4) -------- Define the treatment options --------

        # adult treatment - primary
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg']
                == 'First line treatment for new TB cases for adults',
                'Intervention_Pkg_Code',
            ]
        )[0]
        self.footprints_for_consumables_required['tb_tx_adult'] = {
            "Intervention_Package_Code": {pkg_code1: 1},
            "Item_Code": {}
        }

        # child treatment - primary
        pkg_code2 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg']
                == 'First line treatment for new TB cases for children',
                'Intervention_Pkg_Code',
            ]
        )[0]
        self.footprints_for_consumables_required['tb_tx_child'] = {
            "Intervention_Package_Code": {pkg_code2: 1},
            "Item_Code": {}
        }

        # adult treatment - secondary
        pkg_code3 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg']
                == 'First line treatment for retreatment TB cases for adults',
                'Intervention_Pkg_Code',
            ]
        )[0]
        self.footprints_for_consumables_required['tb_retx_adult'] = {
            "Intervention_Package_Code": {pkg_code3: 1},
            "Item_Code": {}
        }

        # child treatment - secondary
        pkg_code4 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg']
                == 'First line treatment for retreatment TB cases for children',
                'Intervention_Pkg_Code',
            ]
        )[0]
        self.footprints_for_consumables_required['tb_retx_child'] = {
            "Intervention_Package_Code": {pkg_code4: 1},
            "Item_Code": {}
        }

        # mdr treatment
        pkg_code5 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg'] == 'Case management of MDR cases',
                'Intervention_Pkg_Code',
            ]
        )[0]
        self.footprints_for_consumables_required['tb_mdrtx'] = {
            "Intervention_Package_Code": {pkg_code5: 1},
            "Item_Code": {}
        }

        # ipt
        pkg_code6 = pd.unique(
            consumables.loc[
                consumables['Items'] == 'Isoniazid Preventive Therapy',
                'Intervention_Pkg_Code',
            ]
        )[0]
        self.footprints_for_consumables_required['tb_ipt'] = {
            "Intervention_Package_Code": {pkg_code6: 1},
            "Item_Code": {}
        }

    def on_birth(self, mother_id, child_id):
        """ Initialise properties for a newborn individual
        allocate IPT for child if mother diagnosed with TB
        """

        df = self.sim.population.props
        now = self.sim.date

        df.at[child_id, 'tb_inf'] = 'uninfected'
        df.at[child_id, 'tb_strain'] = 'none'

        df.at[child_id, 'tb_date_latent'] = pd.NaT
        df.at[child_id, 'tb_date_active'] = pd.NaT
        df.at[child_id, 'tb_smear'] = False
        # df.at[child_id, 'tb_stage'] = 'none'
        df.at[child_id, 'tb_date_death'] = pd.NaT

        # ------------------ testing status ------------------ #
        df.at[child_id, 'tb_ever_tested'] = False
        # df.at[child_id, 'tb_smear_test'] = False
        # df.at[child_id, 'tb_result_smear_test'] = False
        # df.at[child_id, 'tb_date_smear_test'] = pd.NaT
        # df.at[child_id, 'tb_xpert_test'] = False
        # df.at[child_id, 'tb_result_xpert_test'] = False
        # df.at[child_id, 'tb_date_xpert_test'] = pd.NaT
        df.at[child_id, 'tb_diagnosed'] = False
        df.at[child_id, 'tb_diagnosed_mdr'] = False

        # ------------------ treatment status ------------------ #
        df.at[child_id, 'tb_on_treatment'] = False
        df.at[child_id, 'tb_date_treated'] = pd.NaT
        df.at[child_id, 'tb_treatment_failure'] = False
        df.at[child_id, 'tb_ever_treated'] = False

        df.at[child_id, 'tb_on_ipt'] = False
        df.at[child_id, 'tb_date_ipt'] = pd.NaT

        # todo
        # if mother is diagnosed with TB, give IPT to infant
        # if df.at[mother_id, 'tb_diagnosed']:
        #     event = HSI_Tb_Start_or_Continue_Ipt(self, person_id=child_id)
        #     self.sim.modules['HealthSystem'].schedule_hsi_event(
        #         event,
        #         priority=1,
        #         topen=now,
        #         tclose=now + DateOffset(days=28),
        #     )

    def on_hsi_alert(self, person_id, treatment_id):
        # This is called whenever there is an HSI event commissioned by one of the other disease modules.
        raise NotImplementedError

    def report_daly_values(self):
        """
        This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        experienced by persons in the previous month. Only rows for alive-persons must be returned.
        The names of the series of columns is taken to be the label of the cause of this disability.
        It will be recorded by the healthburden module as <ModuleName>_<Cause>.
        """
        df = self.sim.population.props  # shortcut to population properties dataframe

        health_values = pd.Series(0, index=df.index)

        # hiv-negative
        health_values.loc[
            df.is_alive & (df.tb_inf == 'active') & (df.tb_strain == 'ds') & ~df.hv_inf
            ] = self.daly_wts['daly_tb']
        health_values.loc[
            df.is_alive & (df.tb_inf == 'active') & (df.tb_strain == 'mdr') & ~df.hv_inf
            ] = self.daly_wts['daly_tb']

        # hiv-positive
        health_values.loc[
            df.is_alive & (df.tb_inf == 'active') & (df.tb_strain == 'ds') & df.hv_inf
            ] = self.daly_wts['daly_tb_hiv_anaemia']
        health_values.loc[
            df.is_alive & (df.tb_inf == 'active') & (df.tb_strain == 'mdr') & df.hv_inf
            ] = self.daly_wts['daly_mdr_tb_hiv_anaemia']

        health_values.name = 'tb'  # label the cause of this disability

        return health_values.loc[df.is_alive]


# # ---------------------------------------------------------------------------
# #   TB infection event
# # ---------------------------------------------------------------------------
# # TODO should transmission be limited within each district?
# # TODO add relative infectiousness for those on poor tx [prop tx failure]
# # TODO rel inf for smear negative - weight inf by prop smear negative
# # TODO age/sex distribution for new cases?


class TbRegularPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """ The Tb Regular Polling Events
    * Schedules persons becoming newly infected with latent tb
    * Schedules progression to active tb
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        df = population.props

        # transmission ds-tb
        # the outcome of this will be an updated df with new tb cases
        self.latent_transmission(strain='ds')

        # transmission mdr-tb
        self.latent_transmission(strain='mdr')

        # check who should progress from latent to active disease
        self.module.progression_to_active(population)

    def latent_transmission(self, strain):
        # assume while on treatment, not infectious
        # consider relative infectivity of smear positive/negative and pulmonary / extrapulmonary

        # apply a force of infection to produce new latent cases
        # no age distribution for FOI but the relative risks would affect distribution of active infections
        # this comprises both new infections and reinfections

        df = self.sim.population.props
        p = self.module.parameters
        rng = self.module.rng
        now = self.sim.date
        districts = df['district_of_residence'].unique()

        # smear-positive cases by district
        df['tmp'] = pd.Series(0, index=df.index)
        df.loc[
            df.is_alive & (df.tb_inf == 'active') &
            (df.tb_strain == strain) &
            df.tb_smear &
            ~df.tb_on_treatment, 'tmp'
        ] = 1
        smear_pos = df.groupby(['district_of_residence'])[['tmp']].sum()
        smear_pos = smear_pos.iloc[:, 0]  # convert to series

        # smear-negative cases by district
        df['tmp2'] = pd.Series(0, index=df.index)
        df.loc[
            df.is_alive & (df.tb_inf == 'active') &
            (df.tb_strain == strain) &
            ~df.tb_smear &
            ~df.tb_on_treatment,
            'tmp2',
        ] = 1
        smear_neg = df.groupby(['district_of_residence'])[['tmp2']].sum()
        smear_neg = smear_neg.iloc[:, 0]  # convert to series

        # population by district
        df['tmp3'] = pd.Series(0, index=df.index)
        df.loc[df.is_alive, 'tmp3'] = 1
        pop = df.groupby(['district_of_residence'])[['tmp3']].sum()
        pop = pop.iloc[:, 0]  # convert to series

        del df['tmp']
        del df['tmp2']
        del df['tmp3']

        # calculate foi by district
        foi = pd.Series(0, index=districts)
        foi = (
                  p['transmission_rate']
                  * (smear_pos + (smear_neg * p['rel_inf_smear_ng']))
              ) / pop

        assert foi.isna().sum() == 0  # check there is a foi for every district

        # create a dict, uses district name as keys
        foi_dict = foi.to_dict()

        # look up value for each row in df
        foi_for_individual = df['district_of_residence'].map(foi_dict)

        assert (
            foi_for_individual.isna().sum() == 0
        )  # check there is a district-level foi for every person

        # adjust individual risk by bcg status
        risk_tb = pd.Series(foi_for_individual, index=df.index)
        risk_tb.loc[~df.is_alive] *= 0
        risk_tb.loc[df.va_bcg & df.age_years < 10] *= p['rr_bcg_inf']
        del foi_for_individual

        # get a list of random numbers between 0 and 1 for each infected individual
        random_draw = rng.random_sample(size=len(df))

        # new infections can occur in:
        # uninfected
        # latent infected with this strain (reinfection)
        # latent infected with other strain - replace with latent infection with this strain
        tb_idx = df.index[
            df.is_alive & (df.tb_inf != 'active') & (random_draw < risk_tb)
            ]

        df.loc[tb_idx, 'tb_inf'] = 'latent'
        df.loc[tb_idx, 'tb_date_latent'] = now
        df.loc[tb_idx, 'tb_strain'] = strain


class TbRelapseEvent(RegularEvent, PopulationScopeEventMixin):
    """ The Tb Regular Relapse Events
    * Schedules persons who have previously been infected to relapse
    * Schedules progression to active tb
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        df = self.sim.population.props
        rng = self.module.rng
        now = self.sim.date
        p = self.module.parameters

        # need a monthly relapse for every person in df
        # should return risk=0 for everyone not eligible for relapse
        risk_of_relapse = self.module.lm['risk_relapse'].predict(df.loc[df.is_alive])

        # adjust risk of relapse to account for HIV status
        hiv_risk = pd.Series(data=0, index=df.loc[df.is_alive].index)
        hiv_risk.loc[df.hv_inf] = p['rr_relapse_hiv']
        risk_of_relapse = risk_of_relapse * hiv_risk

        will_relapse = rng.random_sample(len(risk_of_relapse)) < risk_of_relapse
        idx_will_relapse = will_relapse[will_relapse].index

        # schedule progression to active
        for person in idx_will_relapse:
            self.sim.schedule_event(TbActiveEvent(self, person), now)


class TbActiveEvent(Event, IndividualScopeEventMixin):
    """
    1. change individual properties for active disease
    2. if HIV+ schedule AIDS onset
    3. assign smear status
    4. assign symptoms
    5. schedule death
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):

        df = self.sim.population.props
        params = self.module.parameters
        rng = self.module.rng
        now = self.sim.date

        if (
            not df.at[person_id, 'tb_on_ipt']
            and not df.at[person_id, 'tb_on_treatment']
        ):

            # -------- 1) change individual properties for active disease --------

            df.at[person_id, 'tb_inf'] = 'active'
            df.at[person_id, 'tb_date_active'] = now

            # -------- 2) if HIV+ schedule AIDS onset --------
            # todo think about logging COD in HIV/TB
            if df.at[person_id, 'hv_inf']:
                self.sim.schedule_event(hiv.HivAidsOnsetEvent(self.sim.modules['Hiv'], person_id), now)

            # -------- 3) assign smear status --------

            # default value is False
            # hiv_positive
            if df.at[person_id, 'hv_inf'] & (
                rng.rand() < params['prop_smear_positive_hiv']
            ):
                df.at[person_id, 'tb_smear'] = True

            # hiv-negative
            elif rng.rand() < params['prop_smear_positive']:
                df.at[person_id, 'tb_smear'] = True

            # -------- 4) assign symptoms --------

            for symptom in self.module.symptom_list:
                self.sim.modules["SymptomManager"].change_symptom(
                    person_id=person_id,
                    symptom_string=symptom,
                    add_or_remove="+",
                    disease_module=self.module,
                    duration_in_days=None,
                )

        # todo add tb_stage == active_extra with prob based on HIV status


class TbEndTreatmentEvent(RegularEvent, PopulationScopeEventMixin):
    """
    regular monthly check for people currently on treatment
    if treatment has finished, change individual properties
    todo treatment success rates not included
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        df = population.props
        now = self.sim.date
        p = self.module.parameters
        rng = self.sim.module.rng

        # check across population on tb treatment

        # ---------------------- first case ds-tb (6 months) ---------------------- #
        end_ds_idx = df.loc[df.is_alive & df.tb_on_treatment & ~df.tb_treated_mdr &
                            (df.tb_date_treated < (now - DateOffset(days=p['ds_treatment_length']))) &
                            ~df.tb_ever_treated].index
        # sample some to have treatment failure
        # todo check whether this returns series of tx failure selected from end_ds_idx
        ds_tx_failure = rng.random_sample(size=len(end_ds_idx)) < (1 - p['prob_tx_success_new'])
        # idx_ds_tx_failure = ds_tx_failure[ds_tx_failure]
        idx_ds_tx_failure = ds_tx_failure[ds_tx_failure].index

        # ----------------------retreatment ds-tb (7 months) ---------------------- #
        # defined as a current ds-tb cases with property tb_ever_treated as true
        # has completed full tb treatment course previously
        end_ds_retx_idx = df.loc[df.is_alive & df.tb_on_treatment & ~df.tb_treated_mdr &
                                 (df.tb_date_treated < (now - DateOffset(days=p['ds_retreatment_length']))) &
                                 df.tb_ever_treated].index

        # ---------------------- mdr-tb (24 months) ---------------------- #
        end_mdr_tx_idx = df.loc[df.is_alive & df.tb_treated_mdr &
                                (df.tb_date_treated < (
                                    now - DateOffset(days=p['mdr_treatment_length'])))].index

        # todo check index appends
        end_tx_indx = end_ds_idx.union(end_ds_retx_idx, end_mdr_tx_idx)
        # end_tx_indx = end_ds_idx + end_ds_retx_idx + end_mdr_tx_idx
        # idx_tx_failure = idx_ds_tx_failure.union(idx_ds_retx_failure, idx_mdr_tx_failure)
        # idx_tx_failure = idx_ds_tx_failure + idx_ds_retx_failure + idx_mdr_tx_failure

        # change individual properties to off treatment
        df.loc[end_tx_indx, 'tb_on_treatment'] = False
        df.loc[end_tx_indx, 'tb_treated_mdr'] = False
        # this will indicate that this person has had one complete course of tb treatment
        # subsequent infections will be classified as retreatment
        df.loc[end_tx_indx, 'tb_ever_treated'] = True

        # move infection status back to latent
        df.loc[end_tx_indx, 'tb_inf'] = 'latent'
        df.loc[end_tx_indx, 'tb_strain'] = 'none'
        df.loc[end_tx_indx, 'tb_smear'] = False


class TbSelfCureEvent(RegularEvent, PopulationScopeEventMixin):
    """ regular event for TB self-cure
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date
        rng = self.module.rng

        df = population.props

        # self-cure - move from active to latent, excludes cases that just became active
        random_draw = rng.random_sample(size=len(df))

        # hiv-negative
        self_cure = df.loc[
            (df.tb_inf == 'active')
            & df.is_alive
            & ~df.hv_inf
            & (df.tb_date_active < now)
            & (random_draw < params['monthly_prob_self_cure'])
        ].index
        df.loc[self_cure, 'tb_inf'] = 'latent'
        df.loc[self_cure, 'tb_diagnosed'] = False

        # hiv-positive, not on art (or on art but not virally suppressed)
        self_cure_hiv = df.loc[
            (df.tb_inf == 'active')
            & df.is_alive
            & df.hv_inf
            & (df.hv_art != 'on_VL_suppressed')
            & (df.tb_date_active < now)
            & (random_draw < params['monthly_prob_self_cure_hiv'])
        ].index
        df.loc[self_cure_hiv, 'tb_inf'] = 'latent'
        df.loc[self_cure_hiv, 'tb_diagnosed'] = False

        # hiv-positive, on art and virally suppressed
        self_cure_art = df.loc[
            (df.tb_inf == 'active')
            & df.is_alive
            & df.hv_inf
            & (df.hv_art == 'on_VL_suppressed')
            & (df.tb_date_active < now)
            & (random_draw < params['monthly_prob_self_cure'])
        ].index
        df.loc[self_cure_art, 'tb_inf'] = 'latent'
        df.loc[self_cure_art, 'tb_diagnosed'] = False

        # check that tb symptoms are present and caused by tb before resolving
        all_self_cure = [*self_cure, *self_cure_hiv, *self_cure_art]

        for person_id in all_self_cure:
            # this will clear all tb symptoms
            self.sim.modules['SymptomManager'].clear_symptoms(
                person_id=person_id, disease_module=self.module
            )


#
# # ---------------------------------------------------------------------------
# #   HEALTH SYSTEM INTERACTIONS
# # ---------------------------------------------------------------------------
#

#                 # ----------------------------------- REFERRALS FOR IPT -----------------------------------
#                 # todo check these coverage levels relate to paed contacts not HIV cases
#                 if self.sim.date.year >= 2014:
#                     # if diagnosed, trigger ipt outreach event for all paediatric contacts of case
#                     district = df.at[person_id, 'district_of_residence']
#                     ipt_cov = params['ipt_contact_cov']
#                     ipt_cov_year = ipt_cov.loc[
#                         ipt_cov.year == self.sim.date.year
#                     ].coverage.values
#
#                     if (district in params['tb_high_risk_distr'].values) & (
#                         self.module.rng.rand() < ipt_cov_year
#                     ):
#
#                         # check enough contacts available for sample
#                         if (
#                             len(
#                                 df[
#                                     (df.age_years <= 5)
#                                     & ~df.tb_ever_tb
#                                     & ~df.tb_ever_tb_mdr
#                                     & df.is_alive
#                                     & (df.district_of_residence == district)
#                                 ].index
#                             )
#                             > 5
#                         ):
#
#                             # randomly sample from <5 yr olds within district
#                             ipt_sample = (
#                                 df[
#                                     (df.age_years <= 5)
#                                     & ~df.tb_ever_tb
#                                     & ~df.tb_ever_tb_mdr
#                                     & df.is_alive
#                                     & (df.district_of_residence == district)
#                                 ]
#                                 .sample(n=5, replace=False)
#                                 .index
#                             )
#
#                             for person_id in ipt_sample:
#                                 logger.debug(
#                                     'HSI_Tb_Sputum: scheduling IPT for person %d',
#                                     person_id,
#                                 )
#
#                                 ipt_event = HSI_Tb_Ipt(self.module, person_id=person_id)
#                                 self.sim.modules['HealthSystem'].schedule_hsi_event(
#                                     ipt_event,
#                                     priority=1,
#                                     topen=self.sim.date,
#                                     tclose=None,
#                                 )
#
#     def did_not_run(self):
#         logger.debug('HSI_Tb_SputumTest: did not run')
#         pass
#


# ---------------------------------------------------------------------------
#   Health System Interactions (HSI)
# ---------------------------------------------------------------------------

class HSI_Tb_ScreeningAndRefer(HSI_Event, IndividualScopeEventMixin):
    """
    The is the Screening-and-Refer HSI.
    A positive outcome from screening will prompt referral to tb tests (sputum/xpert/xray)
    no consumables are required for screening (4 clinical questions)

    This event is scheduled by:
        * the main event poll,
        * when someone presents for care through a Generic HSI with tb-like symptoms
        * active screening / contact tracing programmes

    Following the screening, they may or may not go on to present for uptake an HIV service: ART (if HIV-positive), VMMC (if
    HIV-negative and male) or PrEP (if HIV-negative and a female sex worker).

    If this event is called within another HSI, it may be desirable to limit the functionality of the HSI: do this
    using the arguments:
        * suppress_footprint=True : the HSI will not have any footprint
    """

    def __init__(self, module, person_id, suppress_footprint=False):
        super().__init__(module, person_id=person_id)
        assert isinstance(
            module, Tb
        )

        assert isinstance(suppress_footprint, bool)
        self.suppress_footprint = suppress_footprint

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Tb_ScreeningAndRefer"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """ Do the screening and referring to next tests """

        df = self.sim.population.props
        person = df.loc[person_id]

        if not person['is_alive']:
            return

        # If the person has previously been diagnosed do nothing do not occupy any resources
        if person['tb_diagnosed']:
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        test_result = None

        # check if patient has: cough, fever, night sweat, weight loss
        # if any of the above conditions are present request appropriate test
        persons_symptoms = self.sim.modules["SymptomManager"].has_what(person_id)
        if any(x in self.module.symptom_list for x in persons_symptoms):

            # if screening indicates presumptive tb
            # child under 5 -> chest x-ray, has to be health system level 2 or above
            if person['age_years'] < 5:
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    HSI_Tb_Xray(person_id=person_id, module=self.module),
                    topen=self.sim.date,
                    tclose=None,
                    priority=0
                )

            # never diagnosed/treated before and hiv-neg -> sputum smear
            elif not person['tb_ever_treated'] and not person['hv_inf']:
                ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1, 'LabTBMicro': 1})
                test_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                    dx_tests_to_run='tb_sputum_test',
                    hsi_event=self
                )

            # previously diagnosed/treated or hiv+ -> xpert
            elif person['tb_ever_treated'] or person['hv_inf']:
                ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1, 'LabMolec': 1})
                test_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                    dx_tests_to_run='tb_xpert_test',
                    hsi_event=self
                )
                if test_result and (person['tb_strain'] == 'mdr'):
                    person['tb_diagnosed_mdr'] = True

                # if xpert not available perform sputum test
                if test_result is None:
                    ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1, 'LabTBMicro': 1})
                    test_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                        dx_tests_to_run='tb_sputum_test',
                        hsi_event=self
                    )

        # if any of the tests are not available (particularly xpert)
        if test_result is None:
            pass

        # if a test has been performed, update person's properties
        if test_result is not None:
            person['tb_ever_tested'] = True

            # if any test returns positive result, refer for appropriate treatment
            if test_result:
                person['tb_diagnosed'] = True

            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Tb_StartTreatment(person_id=person_id, module=self.module),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )

        # Return the footprint. If it should be suppressed, return a blank footprint.
        if self.suppress_footprint:
            return self.make_appt_footprint({})
        else:
            return ACTUAL_APPT_FOOTPRINT


class HSI_Tb_Xray(HSI_Event, IndividualScopeEventMixin):
    """
    The is the x-ray HSI
    usually used for testing children unable to produce sputum
    positive result will prompt referral to start treatment

    """

    def __init__(self, module, person_id, suppress_footprint=False):
        super().__init__(module, person_id=person_id)
        assert isinstance(
            module, Tb
        )

        assert isinstance(suppress_footprint, bool)
        self.suppress_footprint = suppress_footprint

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Tb_Xray"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'DiagRadio': 1})
        self.ACCEPTED_FACILITY_LEVEL = 2
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        person = df.loc[person_id]

        if not person['is_alive']:
            return

        test_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
            dx_tests_to_run='tb_xray',
            hsi_event=self
        )

        # if test returns positive result, refer for appropriate treatment
        if test_result:
            person['tb_diagnosed'] = True

        self.sim.modules['HealthSystem'].schedule_hsi_event(
            HSI_Tb_StartTreatment(person_id=person_id, module=self.module),
            topen=self.sim.date,
            tclose=None,
            priority=0
        )


# # ---------------------------------------------------------------------------
# #   Treatment
# # ---------------------------------------------------------------------------
# # the consumables at treatment initiation include the cost for the full course of treatment
# # so the follow-up appts don't need to account for consumables, just appt time

class HSI_Tb_StartTreatment(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Tb)

        self.TREATMENT_ID = "Tb_Treatment_Initiation"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """This is a Health System Interaction Event - start TB treatment
        select appropriate treatment and request
        if available, change person's properties
        """
        df = self.sim.population.props
        person = df.loc[person_id]
        now = self.sim.date

        if not person["is_alive"]:
            return

        treatment_available = self.module.select_treatment(person_id)

        if treatment_available:
            # start person on tb treatment - update properties
            person['tb_on_treatment'] = True
            person['tb_date_treated'] = now

            # schedule clinical monitoring
            self.module.clinical_monitoring(person_id)


# # ---------------------------------------------------------------------------
# #   Follow-up appts
# # ---------------------------------------------------------------------------
class HSI_Tb_FollowUp(HSI_Event, IndividualScopeEventMixin):
    """
        This is a Health System Interaction Event
        clinical monitoring for tb patients on treatment
        will schedule sputum smear test if needed
        if positive sputum smear, schedule xpert test for drug sensitivity
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Tb)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['TBFollowUp'] = 1

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Tb_FollowUp'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        p = self.module.parameters
        df = self.sim.population.props

        # months since treatment start - to compare with monitoring schedule
        # round to lower integer value
        months_since_tx = int((self.sim.date - df.at[person_id, 'tb_date_treated']).days / 30.5)

        # default clinical monitoring schedule for first infection ds-tb
        sputum_fup = p['followup_times'].loc['ds_sputum']

        # if previously treated:
        if df.at[person_id, 'tb_ever_treated']:

            # if strain is ds and person previously treated:
            sputum_fup = p['followup_times'].loc['ds_retreatment_sputum']

        # if strain is mdr - this treatment schedule takes precedence
        elif df.at[person_id, 'tb_strain'] == 'mdr':

            # if strain is mdr:
            sputum_fup = p['followup_times'].loc['mdr_sputum']

        # check schedule for sputum test and perform if necessary
        if months_since_tx in sputum_fup:
            ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint({'TBFollowUp': 1, 'LabTBMicro': 1})
            test_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run='tb_sputum_test',
                hsi_event=self
            )

            # if sputum test was available and returned positive, schedule xpert test
            if test_result:
                ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint({'TBFollowUp': 1, 'LabTBMicro': 1, 'LabMolec': 1})
                test_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                    dx_tests_to_run='tb_xpert_test',
                    hsi_event=self
                )

                # if xpert test returns mdr-tb diagnosis
                if test_result and (df.at[person_id, 'tb_strain'] == 'mdr'):
                    df.at[person_id, 'tb_diagnosed_mdr'] = True
                    df.at[person_id, 'tb_treatment_failure'] = True

                    # restart treatment (new regimen) if newly diagnosed with mdr-tb
                    self.sim.modules['HealthSystem'].schedule_hsi_event(
                        HSI_Tb_StartTreatment(person_id=person_id, module=self.module),
                        topen=self.sim.date,
                        tclose=None,
                        priority=0
                    )

        return ACTUAL_APPT_FOOTPRINT


# ---------------------------------------------------------------------------
#   IPT
# ---------------------------------------------------------------------------
class HSI_Tb_Start_or_Continue_Ipt(HSI_Event, IndividualScopeEventMixin):
    """
        This is a Health System Interaction Event - give ipt to contacts of tb cases for 6 months
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        self.TREATMENT_ID = 'Tb_Ipt'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        logger.debug(key='message',
                     data=f'Starting IPT for person %d {person_id}')

        df = self.sim.population.props  # shortcut to the dataframe

        person = df.loc[person_id]

        # Do not run if the person is not alive or is already on IPT
        if not (person['is_alive'] & ~person['tb_on_ipt']):
            return

        # Check/log use of consumables, and give IPT if available
        # NB. If materials not available, it is assumed that no IPT is given and no further referral is offered
        if self.get_all_consumables(footprint=self.module.footprints_for_consumables_required['tb_ipt']):
            # Update properties
            df.at[person_id, 'tb_on_ipt'] = True
            df.at[person_id, 'tb_date_ipt'] = self.sim.date

            # schedule decision to continue or end IPT after 6 months
            self.sim.schedule_event(
                Tb_DecisionToContinueIPT(self.module, person_id),
                self.sim.date + DateOffset(months=6),
            )


class Tb_DecisionToContinueIPT(Event, IndividualScopeEventMixin):
    """Helper event that is used to 'decide' if someone on IPT should continue or end
    This event is scheduled by 'HSI_Tb_Start_or_Continue_Ipt' 6 months after it is run.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id]
        m = self.module

        # Check that they are on IPT currently:
        if not person['is_alive'] or not person['tb_on_ipt']:
            return

        # Determine if this appointment is actually attended by the person who has already started on IPT
        if m.rng.random_sample() < m.parameters['prob_of_being_retained_on_ipt_every_6_months']:
            # Continue on Treatment - and schedule an HSI for a continuation appointment today
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Tb_Start_or_Continue_Ipt(person_id=person_id, module=m),
                topen=self.sim.date,
                tclose=self.sim.date + pd.DateOffset(days=14),
                priority=0
            )

        else:
            # Defaults to being off IPT
            df.loc[person_id, 'tb_on_ipt'] = False


# ---------------------------------------------------------------------------
#   Deaths
# ---------------------------------------------------------------------------


class TbDeathEvent(RegularEvent, PopulationScopeEventMixin):
    """
    The regular event that kills people with active tb
    HIV-positive deaths due to TB counted have cause of death = AIDS
    assume same death rates for ds- and mdr-tb
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        p = self.module.parameters
        df = population.props
        now = self.sim.date
        rng = self.module.rng

        # ---------------------------------- TB DEATHS - HIV-NEGATIVE ------------------------------------
        # only active infections result in death
        mortality_rate = pd.Series(0, index=df.index)

        # hiv-negative, tb untreated
        mortality_rate.loc[
            (df.tb_inf == 'active')
            & ~df.hv_inf
            & ~df.tb_on_treatment
        ] = p['monthly_prob_tb_mortality']

        # hiv-negative, tb treated
        mortality_rate.loc[
            (df.tb_inf == 'active')
            & ~df.hv_inf
            & df.tb_on_treatment
        ] = p['monthly_prob_tb_mortality'] * p['mort_tx']

        # Generate a series of random numbers, one per individual
        probs = rng.rand(len(df))
        deaths = df.is_alive & (probs < mortality_rate)
        will_die = (df[deaths]).index

        for person in will_die:
            if df.at[person, 'is_alive']:

                self.sim.schedule_event(
                    demography.InstantaneousDeath(
                        self.module, individual_id=person, cause='tb'
                    ),
                    now,
                )

        # ---------------------------------- HIV-TB DEATHS ------------------------------------
        # only active infections result in death
        # if HIV+ and virally suppressed, monthly death rates same as HIV-
        # assume all on ART will also receive cotrimoxazole

        mort_hiv = pd.Series(0, index=df.index)

        # hiv-positive, on ART and virally suppressed, no TB treatment
        mort_hiv.loc[
            (df.tb_inf == 'active')
            & df.hv_inf
            & (df.hv_art == 'on_VL_suppressed')
            & ~df.tb_on_treatment
        ] = p['monthly_prob_tb_mortality'] * p['mort_cotrim']

        # hiv-positive, on ART and virally suppressed, on TB treatment
        mort_hiv.loc[
            (df.tb_inf == 'active')
            & df.hv_inf
            & (df.hv_art == 'on_VL_suppressed')
            & df.tb_on_treatment
        ] = p['monthly_prob_tb_mortality'] * p['mort_cotrim'] * p['mort_tx']

        # hiv-positive, not virally suppressed, no TB treatment
        mort_hiv.loc[
            (df.tb_inf == 'active')
            & df.hv_inf
            & (df.hv_art != 'on_VL_suppressed')
            & ~df.tb_on_treatment
        ] = p['monthly_prob_tb_mortality_hiv']

        # hiv-positive, not virally suppressed, on TB treatment
        mort_hiv.loc[
            (df.tb_inf == 'active')
            & df.hv_inf
            & (df.hv_art != 'on_VL_suppressed')
            & df.tb_on_treatment
        ] = p['monthly_prob_tb_mortality_hiv'] * p['mort_tx']

        # Generate a series of random numbers, one per individual
        probs = rng.rand(len(df))
        deaths = df.is_alive & (probs < mortality_rate)
        will_die = (df[deaths]).index

        for person in will_die:
            if df.at[person, 'is_alive']:

                self.sim.schedule_event(
                    demography.InstantaneousDeath(
                        self.module, individual_id=person, cause='AIDS'
                    ),
                    now,
                )


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

        # ------------------------------------ INCIDENCE ------------------------------------
        # total number of new active cases in last year - ds + mdr
        # may have died in the last year but still counted as active case for the year

        # number of new active cases
        new_tb_cases = len(
            df[(df.tb_date_active > (now - DateOffset(months=self.repeat)))]
        )

        # incidence per 100k
        inc100k = (new_tb_cases / len(df[df.is_alive])) * 100000

        # number of latent cases
        new_latent_cases = len(
            df[(df.tb_date_latent > (now - DateOffset(months=self.repeat)))]
        )

        # number of new active cases in HIV+
        inc_active_hiv = len(
            df[(df.tb_date_active > (now - DateOffset(months=self.repeat))) & df.hv_inf]
        )

        # proportion of active TB cases in the last year who are HIV-positive
        prop_hiv = inc_active_hiv / new_tb_cases if new_tb_cases else 0

        # incidence of TB-HIV per 100k
        inc100k_hiv = (inc_active_hiv / len(df[df.is_alive])) * 100000

        # number new mdr tb cases
        # TODO this will exclude mdr cases occurring in the last timeperiod but already cured
        new_mdr_cases = len(
            df[(df.tb_strain == 'mdr')
            & (df.tb_date_active > (now - DateOffset(months=self.repeat)))]
        )

        # incidence of mdr-tb per 100k
        inc_mdr100k = (new_mdr_cases / len(df[df.is_alive])) * 100000

        logger.info(
            '%s|tb_incidence|%s',
            now,
            {
                'tbNewActiveCases': new_tb_cases,
                'tbNewLatentCases': new_latent_cases,
                'tbIncActive100k': inc100k,
                'tb_prop_hiv_pos': prop_hiv,
                'tbNewActiveMdrCases': new_mdr_cases,
                'tbIncActive100k_hiv': inc100k_hiv,
                'tbIncActiveMdr100k': inc_mdr100k,
            },
        )
#
#         # ------------------------------------ PREVALENCE ------------------------------------
#         # prevalence should be the number of clinically active cases that occurred in the past year
#         # ACTIVE
#         new_tb_cases = len(
#             df[(df.tb_date_active > (now - DateOffset(months=self.repeat)))]
#         )
#
#         # num_active = len(df[(df.tb_inf.str.contains('active')) & df.is_alive])
#         prop_active = new_tb_cases / len(df[df.is_alive])
#
#         assert prop_active <= 1
#
#         # todo: change to adjusted prevalence counting # episodes / pop
#         # proportion of adults with active tb
#         num_active_adult = len(
#             df[(df.tb_inf.str.contains('active')) & (df.age_years >= 15) & df.is_alive]
#         )
#         prop_active_adult = num_active_adult / len(
#             df[(df.age_years >= 15) & df.is_alive]
#         )
#         assert prop_active_adult <= 1
#
#         # proportion of children with active tb
#         num_active_child = len(
#             df[(df.tb_inf.str.contains('active')) & (df.age_years < 15) & df.is_alive]
#         )
#         prop_active_child = num_active_child / len(
#             df[(df.age_years < 15) & df.is_alive]
#         )
#         assert prop_active_child <= 1
#
#         # proportion of hiv cases with active tb
#         num_active_hiv = len(
#             df[(df.tb_inf.str.contains('active')) & df.hv_inf & df.is_alive]
#         )
#
#         if (num_active_hiv > 0) & (len(df[df.hv_inf & df.is_alive]) > 0):
#             prop_hiv_tb = num_active_hiv / len(df[df.hv_inf & df.is_alive])
#         else:
#             prop_hiv_tb = 0
#
#         # proportion of population with active TB by age/sex - compare to WHO 2018
#         m_num_cases = (
#             df[df.is_alive & (df.sex == 'M') & (df.tb_inf.str.contains('active'))]
#             .groupby('age_range')
#             .size()
#         )
#         f_num_cases = (
#             df[df.is_alive & (df.sex == 'F') & (df.tb_inf.str.contains('active'))]
#             .groupby('age_range')
#             .size()
#         )
#
#         m_pop = df[df.is_alive & (df.sex == 'M')].groupby('age_range').size()
#         f_pop = df[df.is_alive & (df.sex == 'F')].groupby('age_range').size()
#
#         m_prop_active = m_num_cases / m_pop
#         m_prop_active = m_prop_active.fillna(0)
#         f_prop_active = f_num_cases / f_pop
#         f_prop_active = f_prop_active.fillna(0)
#
#         logger.info('%s|tb_propActiveTbMale|%s', self.sim.date, m_prop_active.to_dict())
#
#         logger.info(
#             '%s|tb_propActiveTbFemale|%s', self.sim.date, f_prop_active.to_dict()
#         )
#
#         # LATENT
#
#         # proportion of population with latent TB - all pop
#         num_latent = len(df[(df.tb_inf.str.contains('latent')) & df.is_alive])
#         prop_latent = num_latent / len(df[df.is_alive])
#         assert prop_latent <= 1
#
#         # proportion of population with latent TB - adults
#         num_latent_adult = len(
#             df[(df.tb_inf.str.contains('latent')) & (df.age_years >= 15) & df.is_alive]
#         )
#         prop_latent_adult = num_latent_adult / len(
#             df[(df.age_years >= 15) & df.is_alive]
#         )
#         assert prop_latent_adult <= 1
#
#         # proportion of population with latent TB - children
#         num_latent_child = len(
#             df[(df.tb_inf.str.contains('latent')) & (df.age_years < 15) & df.is_alive]
#         )
#         prop_latent_child = num_latent_child / len(
#             df[(df.age_years < 15) & df.is_alive]
#         )
#         assert prop_latent_child <= 1
#
#         logger.info(
#             '%s|tb_prevalence|%s',
#             now,
#             {
#                 'tbPropActive': prop_active,
#                 'tbPropActiveAdult': prop_active_adult,
#                 'tbPropActiveChild': prop_active_child,
#                 'tbPropActiveHiv': prop_hiv_tb,
#                 'tbPropLatent': prop_latent,
#                 'tbPropLatentAdult': prop_latent_adult,
#                 'tbPropLatentChild': prop_latent_child,
#             },
#         )
#
#         # ------------------------------------ TREATMENT ------------------------------------
#         # number of new cases in the last time period / number new treatment starts in last time period
#
#         # percentage all cases which occurred in last time period and treated in last time period
#         new_tx = len(df[(df.tb_date_treated > (now - DateOffset(months=self.repeat)))])
#
#         percent_treated = ((new_tx / new_tb_cases) * 100) if new_tb_cases else 0
#         # assert percent_treated <= 100
#
#         # percentage all adult cases which occurred in last time period and treated in last time period
#         new_tb_cases_adult = len(
#             df[
#                 (df.age_years >= 15)
#                 & (df.tb_date_active > (now - DateOffset(months=self.repeat)))
#             ]
#         )
#
#         new_tb_tx_adult = len(
#             df[
#                 (df.age_years >= 15)
#                 & (df.tb_date_treated > (now - DateOffset(months=self.repeat)))
#             ]
#         )
#
#         percent_treated_adult = (
#             ((new_tb_tx_adult / new_tb_cases_adult) * 100) if new_tb_cases_adult else 0
#         )
#         # assert percent_treated_adult <= 100
#
#         # percentage all child cases which occurred in last time period and treated in last time period
#         new_tb_cases_child = len(
#             df[
#                 (df.age_years < 15)
#                 & (df.tb_date_active > (now - DateOffset(months=self.repeat)))
#             ]
#         )
#
#         new_tb_tx_child = len(
#             df[
#                 (df.age_years < 15)
#                 & (df.tb_date_treated > (now - DateOffset(months=self.repeat)))
#             ]
#         )
#
#         percent_treated_child = (
#             ((new_tb_tx_child / new_tb_cases_child) * 100) if new_tb_cases_child else 0
#         )
#         # assert percent_treated_child <= 100
#
#         logger.info(
#             '%s|tb_treatment|%s',
#             now,
#             {
#                 'tbTreat': percent_treated,
#                 'tbTreatAdult': percent_treated_adult,
#                 'tbTreatChild': percent_treated_child,
#             },
#         )
#

