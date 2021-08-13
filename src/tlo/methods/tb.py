"""
    This module schedules TB infection and natural history
    It schedules TB treatment and follow-up appointments along with preventive therapy
    for eligible people (HIV+ and paediatric contacts of active TB cases
"""

import os

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom
from tlo.methods import hiv, demography
from tlo.methods.causes import Cause

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
        self.district_list = list()

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'TB': Cause(gbd_causes='Tuberculosis', label='non_AIDS_TB'),
        'AIDS_TB': Cause(gbd_causes='HIV/AIDS', label='AIDS'),
    }

    CAUSES_OF_DISABILITY = {
        'TB': Cause(gbd_causes='Tuberculosis', label='non_AIDS_TB'),
    }

    # Declaration of the specific symptoms that this module will use
    SYMPTOMS = {
        'fatigue',
        'night_sweats'
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

        # ------------------ testing status ------------------ #
        'tb_ever_tested': Property(Types.BOOL, 'ever had a tb test'),
        'tb_diagnosed': Property(Types.BOOL, 'person has current diagnosis of active tb'),
        'tb_date_diagnosed': Property(Types.DATE, 'date most recent tb diagnosis'),
        'tb_diagnosed_mdr': Property(Types.BOOL, 'person has current diagnosis of active mdr-tb'),

        # ------------------ treatment status ------------------ #
        'tb_on_treatment': Property(Types.BOOL, 'on tb treatment regimen'),
        'tb_date_treated': Property(Types.DATE, 'date most recent tb treatment started'),
        'tb_ever_treated': Property(Types.BOOL, 'if ever treated for active tb'),
        'tb_treatment_failure': Property(Types.BOOL, 'failed first line tb treatment'),
        'tb_treated_mdr': Property(Types.BOOL, 'on tb treatment MDR regimen'),
        'tb_date_treated_mdr': Property(Types.DATE, 'date tb MDR treatment started'),

        'tb_on_ipt': Property(Types.BOOL, 'if currently on ipt'),
        'tb_date_ipt': Property(Types.DATE, 'date ipt started'),
    }

    PARAMETERS = {

        # ------------------ workbooks ------------------ #
        'prop_active_2010': Parameter(
            Types.REAL, 'Proportion of population with active tb in 2010'
        ),
        'pulm_tb': Parameter(Types.DATA_FRAME, 'probability of pulmonary tb'),
        'followup_times': Parameter(
            Types.DATA_FRAME, 'times(weeks) tb treatment monitoring required after tx start'
        ),
        'tb_high_risk_distr': Parameter(Types.LIST, 'list of ten high-risk districts'),
        'ipt_coverage': Parameter(
            Types.DATA_FRAME,
            'national estimates of coverage of IPT in PLHIV and paediatric contacts',
        ),

        # ------------------ baseline population ------------------ #
        'prop_mdr2010': Parameter(
            Types.REAL,
            'Proportion of active tb cases with multidrug resistance in 2010',
        ),

        # ------------------ natural history ------------------ #
        'prob_latent_tb_0_14': Parameter(Types.REAL, 'probability of latent infection in ages 0-14 years'),
        'prob_latent_tb_15plus': Parameter(Types.REAL, 'probability of latent infection in ages 15+'),
        'incidence_active_tb_2010_per100k': Parameter(Types.REAL,
                                                      'incidence of active tb in 2010 per 100,000 population'),
        'transmission_rate': Parameter(Types.REAL, 'TB transmission rate, calibrated'),
        'mixing_parameter': Parameter(Types.REAL,
                                      'mixing parameter adjusts transmission rate for force of infection '
                                      'between districts, value 1=completely random mixing across districts, '
                                      'value=0 no between-district transmission'),
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
        'duration_active_disease_years': Parameter(
            Types.REAL, 'duration of active disease from onset to cure or death'
        ),

        # ------------------ clinical features ------------------ #
        'prop_smear_positive': Parameter(
            Types.REAL, 'proportion of new active cases that will be smear-positive'
        ),
        'prop_smear_positive_hiv': Parameter(
            Types.REAL, 'proportion of hiv+ active tb cases that will be smear-positive'
        ),

        # ------------------ mortality ------------------ #
        # untreated
        'death_rate_smear_pos_untreated': Parameter(
            Types.REAL, 'probability of death in smear-positive tb cases with untreated tb'
        ),
        'death_rate_smear_neg_untreated': Parameter(
            Types.REAL, 'probability of death in smear-negative tb cases with untreated tb'
        ),

        # treated
        'death_rate_child0_4_treated': Parameter(
            Types.REAL, 'probability of death in child aged 0-4 years with treated tb'
        ),
        'death_rate_child5_14_treated': Parameter(
            Types.REAL, 'probability of death in child aged 5-14 years with treated tb'
        ),
        'death_rate_adult_treated': Parameter(
            Types.REAL, 'probability of death in adult aged >=15 years with treated tb'
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
        'prob_tx_success_ds': Parameter(
            Types.REAL, 'Probability of treatment success for new and relapse TB cases'
        ),
        'prob_tx_success_mdr': Parameter(
            Types.REAL, 'Probability of treatment success for MDR-TB cases'
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
        # 'rate_testing_tb': Parameter(
        #     Types.REAL,
        #     'rate of presenting for TB screening and testing for people with active TB',
        # ),
        'rate_testing_general_pop': Parameter(
            Types.REAL,
            'rate ratio for TB testing without active TB compared with ative TB cases',
        ),
        # 'rate_testing_hiv': Parameter(
        #     Types.REAL,
        #     'rate of presenting for TB screening and testing for HIV-positive people with active TB',
        # ),
        'presump_testing': Parameter(
            Types.REAL, 'probability of an individual without tb requesting tb test'
        ),
        'ds_treatment_length': Parameter(
            Types.REAL, 'length of treatment for drug-susceptible tb (first case) in days'
        ),
        'ds_retreatment_length': Parameter(
            Types.REAL, 'length of treatment for drug-susceptible tb (secondary case) in days'
        ),
        'mdr_treatment_length': Parameter(
            Types.REAL, 'length of treatment for mdr-tb in days'
        ),
        'prob_retained_ipt_6_months': Parameter(
            Types.REAL, 'probability of being retained on IPT every 6 months if still eligible'
        ),
        'ipt_start_date': Parameter(
            Types.INT, 'year from which IPT is available for paediatric contacts of diagnosed active TB cases'
        ),
    }

    def read_parameters(self, data_folder):
        """
        * 1) Reads the ResourceFiles
        * 2) Declares the DALY weights
        * 3) Declares the Symptoms
        """

        # 1) Read the ResourceFiles
        workbook = pd.read_excel(
            os.path.join(self.resourcefilepath, 'ResourceFile_TB.xlsx'), sheet_name=None
        )
        self.load_parameters_from_dataframe(workbook['parameters'])

        p = self.parameters

        # assume cases distributed equally across districts!!
        p['prop_active_2010'] = workbook['cases2010district']

        p['pulm_tb'] = workbook['pulm_tb']
        p['followup_times'] = workbook['followup']
        p['tb_high_risk_distr'] = workbook['IPTdistricts']
        p['ipt_coverage'] = workbook['ipt_coverage']

        self.district_list = self.sim.modules['Demography'].parameters['pop_2010']['District'].unique().tolist()

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
            Symptom(name='night_sweats',
                    odds_ratio_health_seeking_in_adults=1.0,
                    odds_ratio_health_seeking_in_children=1.0)
        )

    def pre_initialise_population(self):
        """
        * Establish the Linear Models
        """
        p = self.parameters

        # linear model for risk of latent tb in baseline population 2010
        # assume latent tb risk not affected by bcg
        # intercept=1
        self.lm['latent_tb_2010'] = LinearModel.multiplicative(
            Predictor('age_years').when('<15', p["prob_latent_tb_0_14"]).otherwise(p["prob_latent_tb_15plus"]),
        )

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

        self.lm['risk_relapse_2yrs'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1,
            Predictor().when(
                '(tb_inf == "latent") & '
                'tb_ever_treated & '
                '~tb_treatment_failure',
                p['monthly_prob_relapse_tx_complete']),  # ever treated, no tx failure and <2 years post active disease
            Predictor().when(
                '(tb_inf == "latent") & '
                'tb_ever_treated & '
                'tb_treatment_failure',
                p['monthly_prob_relapse_tx_incomplete']),  # ever treated, tx failure and <2 years post active disease
            Predictor('hv_inf').when(True, p["rr_relapse_hiv"]),
        )

        # risk of relapse if >=2 years post treatment
        self.lm['risk_relapse_late'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1,
            Predictor().when(
                '(tb_inf == "latent") & '
                'tb_ever_treated',
                p['monthly_prob_relapse_2yrs']),  # ever treated,
            Predictor('hv_inf').when(True, p["rr_relapse_hiv"]),
        )

        # probability of death
        self.lm['death_rate'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1,
            Predictor().when('(tb_on_treatment == True) & (age_years <=4)', p['death_rate_child0_4_treated']),
            Predictor().when('(tb_on_treatment == True) & (age_years <=14)', p['death_rate_child5_14_treated']),
            Predictor().when('(tb_on_treatment == True) & (age_years >=15)', p['death_rate_adult_treated']),
            Predictor().when('(tb_on_treatment == False) & (tb_smear == True)', p['death_rate_smear_pos_untreated']),
            Predictor().when('(tb_on_treatment == False) & (tb_smear == False)', p['death_rate_smear_neg_untreated']),
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
        idx_new_latent_mdr = (
            df[df.is_alive & (df.tb_inf == 'latent')]
                .sample(frac=p['prop_mdr2010'])
                .index
        )

        df.loc[idx_new_latent_mdr, 'tb_strain'] = 'mdr'

    # def baseline_active(self, population):
    #     """
    #     sample from the baseline population to assign active tb infections
    #     using 2010 incidence estimates
    #     no differences in baseline active tb by age/sex
    #     """
    #
    #     df = population.props
    #     now = self.sim.date
    #     p = self.parameters
    #
    #     eligible_for_active_tb = df.loc[df.is_alive &
    #                                     (df.tb_inf == 'uninfected')].index
    #
    #     sample_active_tb = self.rng.random_sample(len(eligible_for_active_tb)) < (p['incidence_active_tb_2010_per100k'] / 100000)
    #     active_tb_idx = eligible_for_active_tb[sample_active_tb]
    #
    #     # schedule for time now up to 1 year
    #     for person_id in active_tb_idx:
    #         date_progression = now + \
    #                            pd.DateOffset(days=self.rng.randint(0, 365))
    #         self.sim.schedule_event(
    #             TbActiveEvent(self, person_id), date_progression
    #         )

    def progression_to_active(self, population):
        # from the new latent infections, select and schedule progression to active disease

        df = population.props
        p = self.parameters
        rng = self.rng
        now = self.sim.date

        # ------------------ fast progressors ------------------ #
        eligible_for_fast_progression = df.loc[(df.tb_date_latent == now) &
                                                   df.is_alive &
                                                   (df.age_years >= 15) &
                                                   ~df.hv_inf].index

        will_progress = rng.random_sample(len(eligible_for_fast_progression)) < p['prop_fast_progressor']
        fast = eligible_for_fast_progression[will_progress]

        # hiv-positive
        eligible_for_fast_progression_hiv = df.loc[(df.tb_date_latent == now) &
                                                       df.is_alive &
                                                       (df.age_years >= 15) &
                                                       df.hv_inf].index

        will_progress = rng.random_sample(len(eligible_for_fast_progression_hiv)) < p['prop_fast_progressor_hiv']
        fast_hiv = eligible_for_fast_progression_hiv[will_progress]

        fast = fast.union(fast_hiv)  # join indices (checked)

        for person in fast:
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
        eligible_adults = eligible_adults[~eligible_adults.isin(fast)]

        # check no fast progressors included in the slow progressors risk
        assert not any(elem in fast for elem in eligible_adults)

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
        eligible_children = eligible_children[~np.isin(eligible_children, fast)]
        assert not any(elem in fast for elem in eligible_children)

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

        follow_up_times = p['followup_times']

        # default clinical monitoring schedule for first infection ds-tb
        clinical_fup = follow_up_times['ds_clinical_monitor'].dropna()

        # if previously treated:
        if df.at[person_id, 'tb_ever_treated']:

            # if strain is ds and person previously treated:
            clinical_fup = follow_up_times['ds_retreatment_clinical'].dropna()

        # if strain is mdr - this treatment schedule takes precedence
        elif df.at[person_id, 'tb_strain'] == 'mdr':

            # if strain is mdr:
            clinical_fup = follow_up_times['mdr_clinical_monitor'].dropna()

        for appt in clinical_fup:
            # schedule a clinical check-up appointment
            date_appt = self.sim.date + \
                        pd.DateOffset(days=appt * 30.5)

            # this schedules all clinical monitoring appts, tests will occur in some of these
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Tb_FollowUp(person_id=person_id, module=self),
                topen=date_appt,
                tclose=None,
                priority=0
            )

    def initialise_population(self, population):

        df = population.props

        # if HIV is not registered, create a dummy property
        if 'Hiv' not in self.sim.modules:
            population.make_test_property('hv_inf', Types.BOOL)
            population.make_test_property('sy_aids_symptoms', Types.INT)
            population.make_test_property('hv_art', Types.STRING)

            df['hv_inf'] = False
            df['sy_aids_symptoms'] = 0
            df['hv_art'] = 'not'

        # Set our property values for the initial population
        df['tb_inf'].values[:] = 'uninfected'
        df['tb_strain'].values[:] = 'none'

        df['tb_date_latent'] = pd.NaT
        df['tb_date_active'] = pd.NaT
        df['tb_smear'] = False

        # ------------------ testing status ------------------ #
        df['tb_ever_tested'] = False
        df['tb_diagnosed'] = False
        df['tb_date_diagnosed'] = pd.NaT
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
        # self.baseline_active(population)  # allocate active infections from baseline population

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

        tb_sputum_test_cons_footprint = {
            'Intervention_Package_Code': {pkg_sputum: 1}, 'Item_Code': {}
        }

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            tb_sputum_test=DxTest(
                property='tb_smear',
                sensitivity=1.0,
                specificity=1.0,
                cons_req_as_footprint={'Intervention_Package_Code': {pkg_sputum: 1}, 'Item_Code': {}}
            )
        )
        self.footprints_for_consumables_required['tb_sputum_test'] = tb_sputum_test_cons_footprint

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
        # todo add consumables required: footprints_for_consumables_required


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
        # todo add consumables required: footprints_for_consumables_required

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
        item_code6 = pd.unique(
            consumables.loc[
                consumables['Items'] == 'Isoniazid Preventive Therapy',
                'Item_Code',
            ]
        )[0]
        self.footprints_for_consumables_required['tb_ipt'] = {
            "Intervention_Package_Code": {},
            "Item_Code": {item_code6: 1}
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

        # ------------------ testing status ------------------ #
        df.at[child_id, 'tb_ever_tested'] = False

        df.at[child_id, 'tb_diagnosed'] = False
        df.at[child_id, 'tb_date_diagnosed'] = pd.NaT
        df.at[child_id, 'tb_diagnosed_mdr'] = False

        # ------------------ treatment status ------------------ #
        df.at[child_id, 'tb_on_treatment'] = False
        df.at[child_id, 'tb_date_treated'] = pd.NaT
        df.at[child_id, 'tb_treatment_failure'] = False
        df.at[child_id, 'tb_ever_treated'] = False

        df.at[child_id, 'tb_on_ipt'] = False
        df.at[child_id, 'tb_date_ipt'] = pd.NaT

        if 'Hiv' not in self.sim.modules:
            df.at[child_id, 'hv_inf'] = False
            df.at[child_id, 'sy_aids_symptoms'] = 0
            df.at[child_id, 'hv_art'] = 'not'

        # if mother is diagnosed with TB, give IPT to infant
        if df.at[mother_id, 'tb_diagnosed']:
            event = HSI_Tb_Start_or_Continue_Ipt(self, person_id=child_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                event,
                priority=1,
                topen=now,
                tclose=now + DateOffset(days=28),
            )

    def report_daly_values(self):
        """
        This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        experienced by persons in the previous month. Only rows for alive-persons must be returned.
        The names of the series of columns is taken to be the label of the cause of this disability.
        It will be recorded by the healthburden module as <ModuleName>_<Cause>.
        """
        df = self.sim.population.props  # shortcut to population properties dataframe

        # health_values = pd.Series(0, index=df.index)

        # to avoid errors when hiv module not running
        df_tmp = df.loc[df.is_alive]
        health_values = pd.Series(0, index=df_tmp.index)

        # hiv-negative
        health_values.loc[
            df_tmp.is_alive & (df_tmp.tb_inf == 'active') & (df_tmp.tb_strain == 'ds') & ~df_tmp.hv_inf
            ] = self.daly_wts['daly_tb']
        health_values.loc[
            df_tmp.is_alive & (df_tmp.tb_inf == 'active') & (df_tmp.tb_strain == 'mdr') & ~df_tmp.hv_inf
            ] = self.daly_wts['daly_tb']

        # hiv-positive
        health_values.loc[
            df_tmp.is_alive & (df_tmp.tb_inf == 'active') & (df_tmp.tb_strain == 'ds') & df_tmp.hv_inf
            ] = self.daly_wts['daly_tb_hiv_anaemia']
        health_values.loc[
            df_tmp.is_alive & (df_tmp.tb_inf == 'active') & (df_tmp.tb_strain == 'mdr') & df_tmp.hv_inf
            ] = self.daly_wts['daly_mdr_tb_hiv_anaemia']

        health_values.name = 'TB'  # label the cause of this disability

        return health_values.loc[df.is_alive]

    def consider_ipt_for_those_initiating_art(self, person_id):
        """
        this is called by HIV when person is initiating ART
        checks whether person is eligible for IPT
        """
        df = self.sim.population.props

        if df.loc[person_id, 'tb_diagnosed'] or df.loc[person_id, 'tb_diagnosed_mdr']:
            pass

        high_risk_districts = self.parameters["tb_high_risk_distr"]
        district = df.at[person_id, "district_of_residence"]
        eligible = df.at[person_id, "tb_inf"] != "active"

        # select coverage rate by year:
        ipt = self.parameters["ipt_coverage"]
        ipt_year = ipt.loc[ipt.year == self.sim.date.year]
        ipt_coverage_plhiv = ipt_year.coverage_plhiv

        if (
            (district in high_risk_districts.district_name.values)
            & eligible
            & (self.rng.rand() < ipt_coverage_plhiv.values)
        ):
            # Schedule the TB treatment event:
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_Start_or_Continue_Ipt(self, person_id=person_id),
                priority=1,
                topen=self.sim.date,
                tclose=None
            )


# # ---------------------------------------------------------------------------
# #   TB infection event
# # ---------------------------------------------------------------------------

class TbRegularPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """ The Tb Regular Polling Events
    * Schedules persons becoming newly infected with latent tb
    * Schedules progression to active tb
    * schedules tb screening / testing
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

        # schedule some background rates of tb testing (non-symptom driven)
        self.send_for_screening(population)

    def latent_transmission(self, strain):
        """
        assume while on treatment, not infectious
        consider relative infectivity of smear positive/negative and pulmonary / extrapulmonary

        apply a force of infection to produce new latent cases
        no age distribution for FOI but the relative risks would affect distribution of active infections
        this comprises both new infections and reinfections
        """

        df = self.sim.population.props
        p = self.module.parameters
        rng = self.module.rng
        now = self.sim.date
        districts = self.module.district_list

        # -------------- district-level transmission -------------- #
        # get population alive by district
        pop = df.loc[
            df.is_alive
        ].groupby(['district_of_residence'])['is_alive'].sum()
        tmp = pd.DataFrame(pop, index=districts)

        # smear-positive cases by district
        smear_pos = df.loc[
            df.is_alive & (df.tb_inf == 'active') &
            (df.tb_strain == strain) &
            df.tb_smear &
            ~df.tb_on_treatment
            ].groupby(['district_of_residence'])['is_alive'].sum()

        tmp['smear_pos'] = pd.Series(smear_pos, index=districts)

        # smear-negative cases by district
        smear_neg = df.loc[
            df.is_alive & (df.tb_inf == 'active') &
            (df.tb_strain == strain) &
            ~df.tb_smear &
            ~df.tb_on_treatment
            ].groupby(['district_of_residence'])['is_alive'].sum()
        tmp['smear_neg'] = pd.Series(smear_neg, index=districts)

        tmp = tmp.fillna(0)  # fill any missing values with 0

        # calculate foi by district
        foi = pd.Series(0, index=districts)
        foi = (
                  p['transmission_rate']
                  * (tmp['smear_pos'] + (tmp['smear_neg'] * p['rel_inf_smear_ng']))
              ) / tmp['is_alive']

        foi = foi.fillna(0)  # fill any missing values with 0

        # create a dict, uses district name as keys
        foi_dict = foi.to_dict()

        # look up value for each row in df
        foi_for_individual = df['district_of_residence'].map(foi_dict)
        # foi_for_individual = foi_for_individual.fillna(0)  # newly added rows to df will have nan entries

        # -------------- national-level transmission -------------- #

        # add additional background risk of infection to occur nationally
        # accounts for population movement
        # use mixing param to adjust transmission rate
        # apply equally to all

        # total number smear-positive
        total_smear_pos = smear_pos.sum()

        # total number smear-negative
        total_smear_neg = smear_neg.sum()

        # total population
        total_pop = pop.sum()

        foi_national = (p['mixing_parameter'] *
                        p['transmission_rate']
                        * (total_smear_pos + (total_smear_neg * p['rel_inf_smear_ng']))
                        ) / total_pop

        # -------------- individual risk of acquisition -------------- #

        # adjust individual risk by bcg status
        risk_tb = pd.Series(foi_for_individual.values, dtype=float, index=df.index)  # individual risk
        risk_tb += foi_national  # add in background risk (national)
        risk_tb.loc[df.va_bcg & df.age_years < 10] *= p['rr_bcg_inf']
        risk_tb.loc[~df.is_alive] = 0

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

    def send_for_screening(self, population):
        # randomly select some individuals for screening and testing

        df = population.props
        p = self.module.parameters
        rng = self.module.rng

        # get a list of random numbers between 0 and 1 for each infected individual
        random_draw = rng.random_sample(size=len(df))
        screen_idx = df.index[
            df.is_alive & (random_draw < p['rate_testing_general_pop'])
            ]

        for person in screen_idx:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Tb_ScreeningAndRefer(person_id=person, module=self.module),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )


class TbRelapseEvent(RegularEvent, PopulationScopeEventMixin):
    """ The Tb Regular Relapse Event
    runs every month to randomly sample amongst those previously infected with active tb
    * Schedules persons who have previously been infected to relapse with a set probability
    * Schedules progression to active tb (TbActiveEvent)
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        df = self.sim.population.props
        rng = self.module.rng
        now = self.sim.date

        # need a monthly relapse for every person in df
        # should return risk=0 for everyone not eligible for relapse

        # risk of relapse if <2 years post treatment start, includes risk if HIV+
        # get df of those eligible
        relapse_risk_early = df.loc[df.tb_ever_treated &
                                    (self.sim.date < (df.tb_date_treated + pd.DateOffset(days=732.5)))].index

        risk_of_relapse_early = self.module.lm['risk_relapse_2yrs'].predict(df.loc[relapse_risk_early])

        will_relapse = rng.random_sample(len(risk_of_relapse_early)) < risk_of_relapse_early
        idx_will_relapse_early = will_relapse[will_relapse].index

        # risk of relapse if >=2 years post treatment start, includes risk if HIV+
        # get df of those eligible
        relapse_risk_later = df.loc[df.tb_ever_treated &
                                    (self.sim.date >= (df.tb_date_treated + pd.DateOffset(days=732.5)))].index

        risk_of_relapse_later = self.module.lm['risk_relapse_late'].predict(df.loc[relapse_risk_later])

        will_relapse_later = rng.random_sample(len(risk_of_relapse_later)) < risk_of_relapse_later
        idx_will_relapse_late2 = will_relapse_later[will_relapse_later].index

        # join both indices
        idx_will_relapse = idx_will_relapse_early.union(idx_will_relapse_late2)

        # schedule progression to active
        for person in idx_will_relapse:
            self.sim.schedule_event(TbActiveEvent(self.module, person), now)


class TbActiveEvent(Event, IndividualScopeEventMixin):
    """
    1. change individual properties for active disease
    2. assign smear status
    3. assign symptoms
    4. if HIV+, schedule AIDS onset and resample smear status (higher probability)
    5. schedule death
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):

        df = self.sim.population.props
        p = self.module.parameters
        rng = self.module.rng
        now = self.sim.date
        person = df.loc[person_id]

        # if on ipt or treatment - do nothing
        if (
            person['tb_on_ipt']
            or person['tb_on_treatment']
        ):
            return

        logger.debug(key='message',
                     data=f'TbActiveEvent: assigning active tb for person {person_id}')

        # -------- 1) change individual properties for active disease --------

        df.at[person_id, 'tb_inf'] = 'active'
        df.at[person_id, 'tb_date_active'] = now

        # -------- 2) assign smear status --------
        # hiv-negative assumed as default
        if rng.rand() < p['prop_smear_positive']:
            df.at[person_id, 'tb_smear'] = True

        # -------- 3) assign symptoms --------

        for symptom in self.module.symptom_list:
            self.sim.modules["SymptomManager"].change_symptom(
                person_id=person_id,
                symptom_string=symptom,
                add_or_remove="+",
                disease_module=self.module,
                duration_in_days=None,
            )

        # -------- 4) if HIV+ schedule AIDS onset --------
        if person['hv_inf']:
            # higher probability of being smear positive
            if rng.rand() < p['prop_smear_positive_hiv']:
                df.at[person_id, 'tb_smear'] = True

            if 'Hiv' in self.sim.modules:
                self.sim.schedule_event(hiv.HivAidsOnsetEvent(
                    self.sim.modules['Hiv'], person_id, cause='AIDS_TB'), now
                )

        # -------- 5) schedule TB death --------
        # only for non-HIV cases, PLHIV have deaths scheduled by hiv module
        # schedule for all TB cases, prob of death occurring determined by tbDeathEvent
        # select a random death date using duration of disease +/- 6 months
        else:
            d_active_mths = p['duration_active_disease_years'] * 12
            date_of_tb_death = self.sim.date + pd.DateOffset(
                months=int(rng.uniform(low=(d_active_mths - 6), high=(d_active_mths + 6))))
            self.sim.schedule_event(event=TbDeathEvent(person_id=person_id, module=self.module, cause='TB'),
                                    date=date_of_tb_death)


class TbEndTreatmentEvent(RegularEvent, PopulationScopeEventMixin):
    """
    * check for those eligible to finish treatment
    * sample for treatment failure and refer for follow-up screening/testing
    * if treatment has finished, change individual properties
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        df = population.props
        now = self.sim.date
        p = self.module.parameters
        rng = self.module.rng

        # check across population on tb treatment and end treatment if required

        # ---------------------- treatment end: first case ds-tb (6 months) ---------------------- #
        # end treatment for new tb (ds) cases
        # defined as a current ds-tb cases with property tb_ever_treated as false
        end_ds_tx_idx = df.loc[df.is_alive &
                               df.tb_on_treatment &
                               ~df.tb_treated_mdr &
                               (df.tb_date_treated < (now - pd.DateOffset(days=p['ds_treatment_length']))) &
                               ~df.tb_ever_treated].index

        # ---------------------- treatment end: retreatment ds-tb (7 months) ---------------------- #
        # end treatment for retreatment cases
        # defined as a current ds-tb cases with property tb_ever_treated as true
        # has completed full tb treatment course previously
        end_ds_retx_idx = df.loc[df.is_alive &
                                 df.tb_on_treatment &
                                 ~df.tb_treated_mdr &
                                 (df.tb_date_treated < (now - pd.DateOffset(days=p['ds_retreatment_length']))) &
                                 df.tb_ever_treated].index

        # ---------------------- treatment end: mdr-tb (24 months) ---------------------- #
        # end treatment for mdr-tb cases
        end_mdr_tx_idx = df.loc[df.is_alive &
                                df.tb_treated_mdr &
                                (df.tb_date_treated < (
                                    now - pd.DateOffset(days=p['mdr_treatment_length'])))].index

        # join indices
        end_tx_idx = end_ds_tx_idx.union(end_ds_retx_idx)
        end_tx_idx = end_tx_idx.union(end_mdr_tx_idx)

        # ---------------------- treatment failure ---------------------- #
        # sample some to have treatment failure
        random_var = rng.random_sample(size=len(df))

        # children aged 0-4 ds-tb
        ds_tx_failure0_4_idx = df.loc[df.is_alive &
                                   df.tb_on_treatment &
                                   ~df.tb_treated_mdr &
                                   (df.tb_date_treated < (now - pd.DateOffset(days=p['ds_treatment_length']))) &
                                   ~df.tb_ever_treated &
                                    (df.age_years < 5) &
                                   (random_var < (1 - p['prob_tx_success_0_4']))].index

        # children aged 5-14 ds-tb
        ds_tx_failure5_14_idx = df.loc[df.is_alive &
                                   df.tb_on_treatment &
                                   ~df.tb_treated_mdr &
                                   (df.tb_date_treated < (now - pd.DateOffset(days=p['ds_treatment_length']))) &
                                   ~df.tb_ever_treated &
                                    (df.age_years.between(5,14)) &
                                   (random_var < (1 - p['prob_tx_success_5_14']))].index

        # adults ds-tb
        ds_tx_failure_adult_idx = df.loc[df.is_alive &
                                   df.tb_on_treatment &
                                   ~df.tb_treated_mdr &
                                   (df.tb_date_treated < (now - pd.DateOffset(days=p['ds_treatment_length']))) &
                                   ~df.tb_ever_treated &
                                    (df.age_years.between(5,14)) &
                                   (random_var < (1 - p['prob_tx_success_ds']))].index

        # all mdr cases on ds tx will fail
        failure_in_mdr_with_ds_tx_idx = df.loc[df.is_alive &
                                        df.tb_on_treatment &
                                        ~df.tb_treated_mdr &
                                        (df.tb_date_treated < (now - pd.DateOffset(days=p['ds_treatment_length']))) &
                                        ~df.tb_ever_treated &
                                        (df.tb_strain == 'mdr')].index

        # some mdr cases on mdr treatment will fail
        failure_due_to_mdr_idx = df.loc[df.is_alive &
                                        df.tb_on_treatment &
                                        df.tb_treated_mdr &
                                        (df.tb_date_treated < (now - pd.DateOffset(days=p['mdr_treatment_length']))) &
                                        ~df.tb_ever_treated &
                                        (df.tb_strain == 'mdr')].index

        # join indices of failing cases together
        tx_failure = list(ds_tx_failure0_4_idx) + \
                        list(ds_tx_failure5_14_idx) + \
                        list(ds_tx_failure_adult_idx) + \
                        list(failure_in_mdr_with_ds_tx_idx) + \
                        list(failure_due_to_mdr_idx)

        if tx_failure:
            df.loc[tx_failure, 'tb_treatment_failure'] = True
            df.loc[tx_failure, 'tb_ever_treated'] = True  # ensure classed as retreatment case

            for person in tx_failure:
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    HSI_Tb_ScreeningAndRefer(person_id=person, module=self.module),
                    topen=self.sim.date,
                    tclose=None,
                    priority=0
                )

        # remove any treatment failure indices from the treatment end indices
        cure_idx = list(set(end_tx_idx) - set(tx_failure))

        # change individual properties for all to off treatment
        df.loc[end_tx_idx, 'tb_diagnosed'] = False
        df.loc[end_tx_idx, 'tb_on_treatment'] = False
        df.loc[end_tx_idx, 'tb_treated_mdr'] = False
        # this will indicate that this person has had one complete course of tb treatment
        # subsequent infections will be classified as retreatment
        df.loc[end_tx_idx, 'tb_ever_treated'] = True

        # if cured, move infection status back to latent
        df.loc[cure_idx, 'tb_inf'] = 'latent'
        df.loc[cure_idx, 'tb_strain'] = 'none'
        df.loc[cure_idx, 'tb_smear'] = False


class TbSelfCureEvent(RegularEvent, PopulationScopeEventMixin):
    """ annual event which allows some individuals to self-cure
    approximate time from infection to self-cure is 3 years
    HIV+ and not virally suppressed cannot self-cure
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))

    def apply(self, population):
        p = self.module.parameters
        now = self.sim.date
        rng = self.module.rng

        df = population.props

        prob_self_cure = 1/p['duration_active_disease_years']

        # self-cure - move from active to latent, excludes cases that just became active
        random_draw = rng.random_sample(size=len(df))

        # hiv-negative
        self_cure = df.loc[
            (df.tb_inf == 'active')
            & df.is_alive
            & ~df.hv_inf
            & (df.tb_date_active < now)
            & (random_draw < prob_self_cure)
            ].index

        # hiv-positive, on art and virally suppressed
        self_cure_art = df.loc[
            (df.tb_inf == 'active')
            & df.is_alive
            & df.hv_inf
            & (df.hv_art == 'on_VL_suppressed')
            & (df.tb_date_active < now)
            & (random_draw < prob_self_cure)
            ].index

        # resolve symptoms and change properties
        all_self_cure = [*self_cure, *self_cure_art]

        df.loc[all_self_cure, 'tb_inf'] = 'latent'
        df.loc[all_self_cure, 'tb_diagnosed'] = False
        df.loc[all_self_cure, 'tb_strain'] = 'none'
        df.loc[all_self_cure, 'tb_smear'] = False

        for person_id in all_self_cure:
            # this will clear all tb symptoms
            self.sim.modules['SymptomManager'].clear_symptoms(
                person_id=person_id, disease_module=self.module
            )


# ---------------------------------------------------------------------------
#   Health System Interactions (HSI)
# ---------------------------------------------------------------------------

class HSI_Tb_ScreeningAndRefer(HSI_Event, IndividualScopeEventMixin):
    """
    The is the Screening-and-Refer HSI.
    A positive outcome from symptom-based screening will prompt referral to tb tests (sputum/xpert/xray)
    no consumables are required for screening (4 clinical questions)

    This event is scheduled by:
        * the main event poll,
        * when someone presents for care through a Generic HSI with tb-like symptoms
        * active screening / contact tracing programmes

    If this event is called within another HSI, it may be desirable to limit the functionality of the HSI: do this
    using the arguments:
        * suppress_footprint=True : the HSI will not have any footprint

    This event will:
    * screen individuals for TB symptoms
    * administer appropriate TB test
    * schedule treatment if needed
    * give IPT for paediatric contacts of diagnosed case
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
        now = self.sim.date
        p = self.module.parameters
        rng = self.module.rng
        person = df.loc[person_id]

        if not person['is_alive']:
            return

        # If the person is already on treatment and not failing, do nothing do not occupy any resources
        if person['tb_on_treatment'] and not person['tb_treatment_failure']:
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        test_result = None
        ACTUAL_APPT_FOOTPRINT = self.EXPECTED_APPT_FOOTPRINT

        # ------------------------- screening ------------------------- #

        # check if patient has: cough, fever, night sweat, weight loss
        # if any of the above conditions are present request appropriate test
        persons_symptoms = self.sim.modules["SymptomManager"].has_what(person_id)
        if any(x in self.module.symptom_list for x in persons_symptoms):

            # ------------------------- testing ------------------------- #

            # if screening indicates presumptive tb
            # child under 5 -> chest x-ray, has to be health system level 2 or above
            if person['age_years'] < 5:
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    HSI_Tb_Xray(person_id=person_id, module=self.module),
                    topen=now,
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
                    df.at[person_id, 'tb_diagnosed_mdr'] = True
                    df.at[person_id, 'tb_date_diagnosed'] = now

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
            df.at[person_id, 'tb_ever_tested'] = True

            # if any test returns positive result, refer for appropriate treatment
            if test_result:
                df.at[person_id, 'tb_diagnosed'] = True
                df.at[person_id, 'tb_date_diagnosed'] = now

            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Tb_StartTreatment(person_id=person_id, module=self.module),
                topen=now,
                tclose=None,
                priority=0
            )

            # ------------------------- give IPT to contacts ------------------------- #
            # if diagnosed, trigger ipt outreach event for up to 5 paediatric contacts of case
            # only high-risk districts are eligible

            district = person['district_of_residence']
            ipt = self.module.parameters["ipt_coverage"]
            ipt_year = ipt.loc[ipt.year == self.sim.date.year]
            ipt_coverage_paed = ipt_year.coverage_paediatric

            if (district in p['tb_high_risk_distr'].district_name.values.any()) & (
                self.module.rng.rand() < ipt_coverage_paed.values
            ):

                # randomly sample from <5 yr olds within district
                ipt_eligible = df.loc[
                    (df.age_years <= 5)
                    & ~df.tb_diagnosed
                    & df.is_alive
                    & (df.district_of_residence == district)
                    ].index

                if ipt_eligible.any():
                    # sample with replacement in case eligible population n<5
                    ipt_sample = rng.choice(ipt_eligible, size=5, replace=True)
                    # retain unique indices only
                    # fine to have variability in number sampled (between 0-5)
                    ipt_sample = list(set(ipt_sample))

                    for person_id in ipt_sample:
                        logger.debug(
                            'HSI_Tb_ScreeningAndRefer: scheduling IPT for person %d',
                            person_id,
                        )

                        ipt_event = HSI_Tb_Start_or_Continue_Ipt(self.module, person_id=person_id)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(
                            ipt_event,
                            priority=1,
                            topen=now,
                            tclose=None,
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

        if not df.at[person_id, 'is_alive']:
            return

        test_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
            dx_tests_to_run='tb_xray',
            hsi_event=self
        )

        # if test returns positive result, refer for appropriate treatment
        if test_result:
            df.at[person_id, 'tb_diagnosed'] = True
            df.at[person_id, 'tb_date_diagnosed'] = self.sim.date

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
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"TBNew": 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """This is a Health System Interaction Event - start TB treatment
        select appropriate treatment and request
        if available, change person's properties
        """
        df = self.sim.population.props
        now = self.sim.date
        person = df.loc[person_id]

        if not person["is_alive"]:
            return

        treatment_available = self.select_treatment(person_id)

        if treatment_available:
            # start person on tb treatment - update properties
            df.at[person_id, 'tb_on_treatment'] = True
            df.at[person_id, 'tb_date_treated'] = now

            if person['tb_diagnosed_mdr']:
                df.at[person_id, 'tb_treated_mdr'] = True
                df.at[person_id, 'tb_date_treated_mdr'] = now

            # schedule clinical monitoring
            self.module.clinical_monitoring(person_id)

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

            drugs_available = HSI_Event.get_all_consumables(
                self, footprint=self.module.footprints_for_consumables_required[
                    'tb_mdrtx'])

        # -------- First TB infection -------- #
        # could be undiagnosed mdr or ds-tb: treat as ds-tb

        elif not person['tb_ever_treated']:

            if person['age_years'] >= 15:
                # treatment for ds-tb: adult
                drugs_available = HSI_Event.get_all_consumables(
                    self, footprint=self.module.footprints_for_consumables_required[
                        'tb_tx_adult'])
            else:
                # treatment for ds-tb: child
                drugs_available = HSI_Event.get_all_consumables(
                    self, footprint=self.module.footprints_for_consumables_required[
                        'tb_tx_child'])

        # -------- Secondary TB infection -------- #
        # person has been treated before
        # possible treatment failure or subsequent reinfection
        else:

            if person['age_years'] >= 15:
                # treatment for reinfection ds-tb: adult
                drugs_available = HSI_Event.get_all_consumables(
                    self, footprint=self.module.footprints_for_consumables_required[
                        'tb_retx_adult'])
            else:
                # treatment for reinfection ds-tb: child
                drugs_available = HSI_Event.get_all_consumables(
                    self, footprint=self.module.footprints_for_consumables_required[
                        'tb_retx_child'])

        return drugs_available


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
        the_appt_footprint = self.make_appt_footprint({'TBFollowUp': 1})

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Tb_FollowUp'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        p = self.module.parameters
        df = self.sim.population.props
        person = df.loc[person_id]

        ACTUAL_APPT_FOOTPRINT = self.EXPECTED_APPT_FOOTPRINT

        # months since treatment start - to compare with monitoring schedule
        # round to lower integer value
        months_since_tx = int((self.sim.date - df.at[person_id, 'tb_date_treated']).days / 30.5)

        # default clinical monitoring schedule for first infection ds-tb
        follow_up_times = p['followup_times']
        sputum_fup = follow_up_times['ds_sputum'].dropna()

        # if previously treated:
        if person['tb_ever_treated']:

            # if strain is ds and person previously treated:
            sputum_fup = follow_up_times['ds_retreatment_sputum'].dropna()

        # if strain is mdr - this treatment schedule takes precedence
        elif person['tb_strain'] == 'mdr':

            # if strain is mdr:
            sputum_fup = follow_up_times['mdr_sputum'].dropna()

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
                    # already diagnosed with active tb so don't update tb_date_diagnosed
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
        This is a Health System Interaction Event - give ipt to reduce risk of active TB
        It can be scheduled by:
        * HIV.HSI_Hiv_StartOrContinueTreatment for PLHIV, diagnosed and on ART
        * Tb.HSI_Tb_StartTreatment for up to 5 contacts of diagnosed active TB case

        if person referred by ART initiation (HIV+), IPT given for 36 months
        paediatric IPT is 6-9 months
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

        # Do not run if the person is not alive or already on IPT or diagnosed active infection
        if (not person['is_alive']) and person['tb_on_ipt'] and (not person['tb_diagnosed']):
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
    This event is scheduled by 'HSI_Tb_Start_or_Continue_Ipt' after 6 months

    * end IPT for all
    * schedule further IPT for HIV+ if still eligible (no active TB diagnosed, <36 months IPT)
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id]
        m = self.module

        if not (person['is_alive']):
            return

        # default update properties for all
        df.at[person_id, 'tb_on_ipt'] = False

        # decide whether PLHIV will continue
        if (not person['tb_diagnosed']) and (
            person['tb_date_ipt'] < (self.sim.date - pd.DateOffset(days=36 * 30.5))) and (
            m.rng.random_sample() < m.parameters['prob_retained_ipt_6_months']):
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Tb_Start_or_Continue_Ipt(person_id=person_id, module=m),
                topen=self.sim.date,
                tclose=self.sim.date + pd.DateOffset(days=14),
                priority=0
            )


# ---------------------------------------------------------------------------
#   Deaths
# ---------------------------------------------------------------------------

class TbDeathEvent(Event, IndividualScopeEventMixin):
    """
    The scheduled death for a tb case
    check whether this death should occur using a linear model
    will depend on treatment status, smear status and age
    """

    def __init__(self, module, person_id, cause):
        super().__init__(module, person_id=person_id)
        self.cause = cause

    def apply(self, person_id):
        df = self.sim.population.props

        if not df.at[person_id, "is_alive"]:
            return

        logger.debug(key='message',
                     data=f'TbDeathEvent: checking whether death should occur for person {person_id}')

        # use linear model to determine whether this person will die:
        rng = self.module.rng.rand()
        result = self.module.lm['death_rate'].predict(df.loc[[person_id]]).values[0]

        if result < rng:
            logger.debug(key='message',
                         data=f'TbDeathEvent: cause this death for person {person_id}')
            # self.sim.schedule_event(
            #     demography.InstantaneousDeath(
            #         self.module, person_id, cause=self.cause
            #     ),
            #     self.sim.date,
            # )
            self.sim.modules['Demography'].do_death(
                individual_id=person_id, cause=self.cause,
                originating_module=self.module)


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

        logger.info(key='tb_incidence',
                    description='Number new active and latent TB cases, total and in PLHIV',
                    data={
                        'num_new_active_tb': new_tb_cases,
                        'num_new_latent_tb': new_latent_cases,
                        'num_new_active_tb_in_hiv': inc_active_hiv,
                        'prop_active_tb_in_plhiv': prop_hiv,
                    },
                    )

        # ------------------------------------ PREVALENCE ------------------------------------
        # number of current active cases divided by population alive

        # ACTIVE
        num_active_tb_cases = len(df[(df.tb_inf == 'active') & df.is_alive])
        prev_active = num_active_tb_cases / len(df[df.is_alive])

        assert prev_active <= 1

        # prevalence of active TB in adults
        num_active_adult = len(
            df[(df.tb_inf == 'active') & (df.age_years >= 15) & df.is_alive]
        )
        prev_active_adult = num_active_adult / len(
            df[(df.age_years >= 15) & df.is_alive]
        )
        assert prev_active_adult <= 1

        # prevalence of active TB in children
        num_active_child = len(
            df[(df.tb_inf == 'active') & (df.age_years < 15) & df.is_alive]
        )
        prev_active_child = num_active_child / len(
            df[(df.age_years < 15) & df.is_alive]
        )
        assert prev_active_child <= 1

        # LATENT
        # proportion of population with latent TB - all pop
        num_latent = len(df[(df.tb_inf == 'latent') & df.is_alive])
        prev_latent = num_latent / len(df[df.is_alive])
        assert prev_latent <= 1

        # proportion of population with latent TB - adults
        num_latent_adult = len(
            df[(df.tb_inf == 'latent') & (df.age_years >= 15) & df.is_alive]
        )
        prev_latent_adult = num_latent_adult / len(
            df[(df.age_years >= 15) & df.is_alive]
        )
        assert prev_latent_adult <= 1

        # proportion of population with latent TB - children
        num_latent_child = len(
            df[(df.tb_inf == 'latent') & (df.age_years < 15) & df.is_alive]
        )
        prev_latent_child = num_latent_child / len(
            df[(df.age_years < 15) & df.is_alive]
        )
        assert prev_latent_child <= 1

        logger.info(key='tb_prevalence',
                    description='Prevalence of active and latent TB cases, total and in PLHIV',
                    data={
                        'tbPrevActive': prev_active,
                        'tbPrevActiveAdult': prev_active_adult,
                        'tbPrevActiveChild': prev_active_child,
                        'tbPrevLatent': prev_latent,
                        'tbPrevLatentAdult': prev_latent_adult,
                        'tbPrevLatentChild': prev_latent_child,
                    },
                    )

        # ------------------------------------ MDR ------------------------------------
        # number new mdr tb cases
        # TODO this will exclude mdr cases occurring in the last timeperiod but already cured
        new_mdr_cases = len(
            df[(df.tb_strain == 'mdr')
               & (df.tb_date_active > (now - DateOffset(months=self.repeat)))]
        )

        if new_mdr_cases:
            prop_mdr = new_mdr_cases / new_tb_cases
        else:
            prop_mdr = 0

        logger.info(key='tb_mdr',
                    description='Incidence of new active MDR cases and the proportion of TB cases that are MDR',
                    data={
                        'tbNewActiveMdrCases': new_mdr_cases,
                        'tbPropActiveCasesMdr': prop_mdr
                    },
                    )

        # ------------------------------------ CASE NOTIFICATIONS ------------------------------------
        # number diagnoses (new, relapse, reinfection) in last timeperiod
        new_tb_diagnosis = len(
            df[(df.tb_date_diagnosed > (now - DateOffset(months=self.repeat)))]
        )

        # ------------------------------------ TREATMENT ------------------------------------
        # number of tb cases initiated treatment in last timeperiod / new active cases
        new_tb_tx = len(
            df[(df.tb_date_treated > (now - DateOffset(months=self.repeat)))]
        )
        if new_tb_cases:
            tx_coverage = new_tb_tx / new_tb_cases
        else:
            tx_coverage = 0

        logger.info(key='tb_treatment',
                    description='TB treatment coverage',
                    data={
                        'tbNewDiagnosis': new_tb_diagnosis,
                        'tbTreatmentCoverage': tx_coverage,
                    },
                    )
