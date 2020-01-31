import logging

import pandas as pd
from pathlib import Path

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography
from tlo.methods.healthsystem import HSI_Event
from tlo.lm import LinearModel, LinearModelType, Predictor


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Labour (Module):
    """This module for labour, delivery, the immediate postpartum period and skilled birth attendance."""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # This dictionary will store additional information around delivery and birth
        self.mother_and_newborn_info = dict()

    PARAMETERS = {
        #  ===================================  NATURAL HISTORY PARAMETERS =============================================
        'prob_pregnancy': Parameter(
            Types.REAL, 'baseline probability of pregnancy - currently included as a dummy parameter'),
        'baseline_prev_cs': Parameter(
            Types.REAL, 'prevalence of women who have previously ever delivered via caesarean section'),
        'prob_prom': Parameter(
            Types.REAL, 'probability of a woman in term labour having had experience prolonged rupture of membranes'),
        'prob_pl_ol': Parameter(
            Types.REAL, 'probability of a woman entering prolonged/obstructed labour'),
        'rr_PL_OL_nuliparity': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if they are nuliparous'),
        'rr_PL_OL_para1': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if they have a parity of 1'),
        'rr_PL_OL_age_less20': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her age is less'
                        'than 20 years'),
        'prob_ptl': Parameter(
            Types.REAL, 'baseline probability of a woman entering labour at less than 37 weeks gestation'),
        'baseline_prev_labour_states': Parameter(
            Types.LIST, 'baseline prevalence of early preterm, late preterm, term and postterm gestations at delivery'),
        'prob_early_ptb': Parameter(
            Types.REAL, 'probability of a woman going into preterm labour between 28-33 weeks gestation'),
        'rr_early_ptb_age<20': Parameter(
            Types.REAL, 'relative risk of early preterm labour for women younger than 20'),
        'rr_early_ptb_prev_ptb': Parameter(
            Types.REAL, 'relative risk of early preterm labour for women who have previously delivered preterm'),
        'rr_early_ptb_anaemia': Parameter(
            Types.REAL, 'relative risk of preterm labour for suffering from anaemia'),
        'prob_late_ptb': Parameter(
            Types.REAL, 'probability of a woman going into preterm labour between 33-36 weeks gestation'),
        'rr_late_ptb_prev_ptb': Parameter(
            Types.REAL, 'relative risk of preterm labour for women younger than 20'),
        'prob_potl': Parameter(
            Types.REAL, 'probability of a woman entering labour at >42 weeks gestation'),
        'prob_ip_eclampsia': Parameter(
            Types.REAL, 'probability of an eclamptic seizure during labour'),
        'prob_aph': Parameter(
            Types.REAL, 'probability of an antepartum haemorrhage during labour'),
        'prob_ip_sepsis': Parameter(
            Types.REAL, 'probability of sepsis in labour'),
        'prob_uterine_rupture': Parameter(
            Types.REAL, 'probability of a uterine rupture during labour'),
        'rr_ur_grand_multip': Parameter(
            Types.REAL, 'relative risk of uterine rupture in women who have delivered >4 times previously'),
        'rr_ur_prev_cs': Parameter(
            Types.REAL, 'relative risk of uterine rupture in women who have previously delivered via caesarean section'),
        'rr_ur_ref_ol': Parameter(
            Types.REAL,
            'relative risk of uterine rupture in women who have been referred in obstructed labour'),
        'rr_ip_sepsis_pl_ol': Parameter(
            Types.REAL, 'relative risk of developing sepsis following obstructed labour'),
        'rr_ip_eclampsia_30_34': Parameter(
            Types.REAL, 'relative risk of eclampsia for women ages between 30 and 34'),
        'rr_ip_eclampsia_35': Parameter(
            Types.REAL, 'relative risk of eclampsia for women ages older than 35'),
        'rr_ip_eclampsia_nullip': Parameter(
            Types.REAL, 'relative risk of eclampsia for women who have not previously delivered a child'),
        'rr_ip_sepsis_anc_4': Parameter(
            Types.REAL, 'relative risk of sepsis for women who have attended greater than 4 ANC visits'),
        'rr_ip_aph_noedu': Parameter(
            Types.REAL, 'relative risk of antepartum haemorrhage for women with education of primary level or lower'),
        'rr_aph_pl_ol': Parameter(
            Types.REAL, 'relative risk of antepartum haemorrhage following obstructed labour'),
        'prob_cord_prolapse': Parameter(
            Types.REAL, 'probability of this woman experiencing a cord prolapse'),
        'cfr_obstructed_labour': Parameter(
            Types.REAL, 'case fatality rate for obstructed labour'),
        'cfr_aph': Parameter(
            Types.REAL, 'case fatality rate for antepartum haemorrhage during labour'),
        'cfr_eclampsia': Parameter(
            Types.REAL, 'case fatality rate for eclampsia during labours'),
        'cfr_sepsis': Parameter(
            Types.REAL, 'case fatality rate for sepsis during labour'),
        'cfr_uterine_rupture': Parameter(
            Types.REAL, 'case fatality rate for uterine rupture in labour'),
        'prob_still_birth_obstructed_labour': Parameter(
            Types.REAL, 'probability of a still birth following obstructed labour where the mother survives'),
        'prob_still_birth_obstructed_labour_md': Parameter(
            Types.REAL, 'probability of a still birth following obstructed labour where the mother dies'),
        'prob_still_birth_aph': Parameter(
            Types.REAL, 'probability of a still birth following antepartum haemorrhage where the mother survives'),
        'prob_still_birth_aph_md': Parameter(
            Types.REAL, 'probability of a still birth following antepartum haemorrhage where the mother dies'),
        'prob_still_birth_sepsis': Parameter(
            Types.REAL, 'probability of a still birth following sepsis in labour where the mother survives'),
        'prob_still_birth_sepsis_md': Parameter(
            Types.REAL, 'probability of a still birth following sepsis in labour where the mother dies'),
        'prob_still_birth_uterine_rupture': Parameter(
            Types.REAL, 'probability of a still birth following uterine rupture in labour where the mother survives'),
        'prob_still_birth_uterine_rupture_md': Parameter(
            Types.REAL, 'probability of a still birth following uterine rupture in labour where the mother dies'),
        'prob_still_birth_eclampsia': Parameter(
            Types.REAL, 'probability of still birth following eclampsia in labour where the mother survives'),
        'prob_still_birth_eclampsia_md': Parameter(
            Types.REAL, 'probability of still birth following eclampsia in labour where the mother dies'),
        'prob_pp_eclampsia': Parameter(
            Types.REAL, 'probability of eclampsia following delivery for women who were in spotaneous unobstructed '
                        'labour'),
        'prob_pph': Parameter(
            Types.REAL, 'probability of an postpartum haemorrhage following labour'),
        'rr_pph_pl_ol': Parameter(
            Types.REAL, 'relative risk of postpartum haemorrhage following obstructed labour'),
        'prob_pp_sepsis': Parameter(
            Types.REAL, 'probability of sepsis following delivery'),
        'cfr_pp_pph': Parameter(
            Types.REAL, 'case fatality rate for postpartum haemorrhage'),
        'cfr_pp_eclampsia': Parameter(
            Types.REAL, 'case fatality rate for eclampsia following delivery'),
        'cfr_pp_sepsis': Parameter(
            Types.REAL, 'case fatality rate for sepsis following delivery'),
        'daly_wt_haemorrhage_moderate': Parameter(
            Types.REAL, 'DALY weight for a moderate maternal haemorrhage (<1 litre)'),
        'daly_wt_haemorrhage_severe': Parameter(
            Types.REAL, 'DALY weight for a severe maternal haemorrhage (>1 litre)'),
        'daly_wt_maternal_sepsis': Parameter(
            Types.REAL, 'DALY weight for maternal sepsis'),
        'daly_wt_eclampsia': Parameter(
            Types.REAL, 'DALY weight for eclampsia'),
        'daly_wt_obstructed_labour': Parameter(
            Types.REAL, 'DALY weight for obstructed labour'),
        'prob_neonatal_sepsis': Parameter(
            Types.REAL, 'baseline probability of a child developing sepsis following birth'),
        'prob_neonatal_birth_asphyxia': Parameter(
            Types.REAL, 'baseline probability of a child developing neonatal encephalopathy following delivery'),
        'odds_homebirth': Parameter(
            Types.REAL, 'odds of a woman delivering in at home when in labour'),
        'or_homebirth_unmarried': Parameter(
            Types.REAL, 'odds ratio of an unmarried woman delivering at home when in labour'),
        'or_homebirth_wealth_4': Parameter(
            Types.REAL, 'odds ratio of a woman delivering at home whose wealth level is 4'),
        'or_homebirth_wealth_5': Parameter(
            Types.REAL, 'odds ratio of a woman delivering at home whose wealth level is 5'),
        'or_homebirth_urban': Parameter(
            Types.REAL, 'odds ratio of a woman delivering at home when she lives in a urban setting'),

        # ================================= TREATMENT PARAMETERS =====================================================
        'prob_successful_induction': Parameter(
            Types.REAL, 'probability of that induction of labour will be successful'),
        'rr_maternal_sepsis_clean_delivery': Parameter(
            Types.REAL, 'relative risk of maternal sepsis following clean birth practices employed in a facility'),
        'rr_newborn_sepsis_clean_delivery': Parameter(
            Types.REAL, 'relative risk of newborn sepsis following clean birth practices employed in a facility'),
        'rr_sepsis_post_abx_prom': Parameter(
            Types.REAL, 'relative risk of maternal sepsis following prophylactic antibiotics for PROM in a facility'),
        'rr_newborn_sepsis_proph_abx': Parameter(
            Types.REAL, 'relative risk of newborn sepsis following prophylactic antibiotics for '
                        'premature labour in a facility'),
        'rr_pph_amtsl': Parameter(
            Types.REAL, 'relative risk of severe post partum haemorrhage following active management of the third '
                        'stage of labour'),
        'prob_cure_antibiotics': Parameter(
            Types.REAL, 'Probability of sepsis resolving following the administration of antibiotics'),
        'prob_cure_mgso4': Parameter(
            Types.REAL, 'relative risk of additional seizures following of administration of magnesium sulphate'),
        'prob_prevent_mgso4': Parameter(
            Types.REAL, 'relative risk of eclampsia following administration of magnesium sulphate in women '
                        'with severe pre-eclampsia'),
        'prob_cure_diazepam': Parameter(
            Types.REAL, 'relative risk of additional seizures following of administration of diazepam'),
        'prob_cure_blood_transfusion': Parameter(
            Types.REAL, '...'),
        'prob_cure_oxytocin': Parameter(
            Types.REAL, 'probability of intravenous oxytocin arresting post-partum haemorrhage'),
        'prob_cure_misoprostol': Parameter(
            Types.REAL, 'probability of rectal misoprostol arresting post-partum haemorrhage'),
        'prob_cure_uterine_massage': Parameter(
            Types.REAL, 'probability of uterine massage arresting post-partum haemorrhage'),
        'prob_cure_uterine_tamponade': Parameter(
            Types.REAL, 'probability of uterine tamponade arresting post-partum haemorrhage'),
        'prob_cure_uterine_ligation': Parameter(
            Types.REAL, 'probability of laparotomy and uterine ligation arresting post-partum haemorrhage'),
        'prob_cure_b_lynch': Parameter(
            Types.REAL, 'probability of laparotomy and B-lynch sutures arresting post-partum haemorrhage'),
        'prob_cure_hysterectomy': Parameter(
            Types.REAL, 'probability of total hysterectomy arresting post-partum haemorrhage'),
        'prob_cure_manual_removal': Parameter(
            Types.REAL, 'probability of manual removal of retained products arresting a post partum haemorrhage'),
        'prob_cure_uterine_repair': Parameter(
            Types.REAL, 'probability repairing a ruptured uterus surgically'),
        'prob_deliver_ventouse': Parameter(
            Types.REAL, 'probability of successful delivery with ventouse'),
        'prob_deliver_forceps': Parameter(
            Types.REAL, 'probability of successful delivery with forceps'),
    }

    PROPERTIES = {
        'la_due_date_current_pregnancy': Property(Types.DATE, 'The date on which a newly pregnant woman is scheduled to'
                                                              ' go into labour'),
        'la_currently_in_labour': Property(Types.BOOL, 'whether this woman is currently in labour'),
        'la_current_labour_successful_induction': Property(Types.CATEGORICAL, 'Not Induced, Successful Induction, '
                                                                              'Failed Induction',
                                                           categories=['not_induced', 'successful_induction',
                                                                       'failed_induction']),
        'la_intrapartum_still_birth': Property(Types.BOOL, 'whether this womans most recent pregnancy has ended '
                                                           'in a stillbirth'),
        'la_parity': Property(Types.INT, 'total number of previous deliveries'),

        'la_previous_cs_delivery': Property(Types.BOOL, 'whether this woman has ever delivered via caesarean section'),
        'la_has_previously_delivered_preterm': Property(Types.BOOL, 'whether the woman has had a previous preterm '
                                                                    'delivery for any of her previous deliveries'),
        'la_obstructed_labour': Property(Types.BOOL, 'whether this womans labour has become obstructed'),
        'la_obstructed_labour_disab': Property(Types.BOOL, 'disability associated with obstructed labour'),
        'la_antepartum_haem': Property(Types.BOOL, 'whether the woman has experienced an antepartum haemorrhage in this'
                                                   'delivery'),
        'la_antepartum_haem_disab': Property(Types.BOOL, 'disability associated with antepartum haemorrhage'),
        'la_uterine_rupture': Property(Types.BOOL, 'whether the woman has experienced uterine rupture in this delivery'),
        'la_uterine_rupture_disab': Property(Types.BOOL, 'disability associated with uterine rupture'),
        'la_sepsis': Property(Types.BOOL, 'whether the woman has developed sepsis associated with in this delivery'),
        'la_sepsis_disab': Property(Types.BOOL, 'disability associated with maternal sepsis'),
        'la_eclampsia': Property(Types.BOOL, 'whether the woman has experienced an eclamptic seizure in this delivery'),
        'la_eclampsia_disab': Property(Types.BOOL, 'disability associated with maternal haemorrhage'),
        'la_postpartum_haem': Property(Types.BOOL, 'whether the woman has experienced an postpartum haemorrhage in this'
                                                   'delivery'),
        'la_postpartum_haem_disab': Property(Types.BOOL, 'disability associated with postpartum haemorrhage'),
        # TODO: above property could be categorical to reflect severity of bleed and map better with DALY weights
        'la_maternal_death': Property(Types.BOOL, ' whether the woman has died as a result of this pregnancy'),  # DUMMY
        'la_maternal_death_date': Property(Types.DATE, 'date of death for a date in pregnancy')  # DUMMY
    }

    def read_parameters(self, data_folder):

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_LabourSkilledBirthAttendance.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)
        params = self.parameters

        # Here we will include DALY weights if applicable...

        if 'HealthBurden' in self.sim.modules.keys():
            params['la_daly_wts'] = \
                {'hemorrhage_moderate': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=339),
                 'haemorrhage_severe': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=338),
                 'maternal_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=340),
                 'eclampsia': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=343),
                 'obstructed_labour': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=348)}
            # TODO: Eclampsia DALY weight is empty- this is htn disorders sequalae code
            # TODO: source DALY weight for Uterine Rupture

# ======================================= LINEAR MODEL EQUATIONS ======================================================
        # Here we define the equations that will be used throughout this module using the linear model and stored them
        # as a paramter

        params['la_labour_equations'] =\
            {'parity': LinearModel(
                LinearModelType.ADDITIVE, # TODO: first stage dummy- needs to come from data
                0.0,
                Predictor('age_years').when('.between(15,24)', self.rng.choice(range(0, 3)))
                                      .when('.between(24,40)', self.rng.choice(range(0, 5)))
                                      .when(' > 40', self.rng.choice(range(0, 7)))),

                'early_preterm_birth': LinearModel(
                LinearModelType.MULTIPLICATIVE,  # TODO: Anaemia/ Malaria / Multiple gestation
                params['prob_early_ptb'],
                Predictor('age_years').when('.between(15,20)', params['rr_early_ptb_age<20']),
                Predictor('la_has_previously_delivered_preterm').when(True, params['rr_early_ptb_prev_ptb'])),

             'late_preterm_birth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_late_ptb'],
                Predictor('la_has_previously_delivered_preterm').when(True, params['rr_late_ptb_prev_ptb'])),

             'obstructed_labour_ip': LinearModel(
                LinearModelType.MULTIPLICATIVE,  # TODO: stunting/malnutrition
                params['prob_pl_ol'],
                Predictor('la_parity').when('0', params['rr_PL_OL_nuliparity']),
                Predictor('la_parity').when('1', params['rr_PL_OL_para1']),
                Predictor('age_years').when('<20', params['rr_PL_OL_age_less20'])),

             'sepsis_ip': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_ip_sepsis'],
                Predictor('la_obstructed_labour').when(True, params['rr_ip_sepsis_pl_ol'])),

             'sepsis_pp': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_pp_sepsis'],
                Predictor('la_obstructed_labour').when(True, params['rr_ip_sepsis_pl_ol'])),
                # DUMMY, copy from above

             'eclampsia_ip': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_ip_eclampsia'],
                Predictor('age_years').when('.between(30,34)', params['rr_ip_eclampsia_30_34']),
                Predictor('age_years').when('>35', params['rr_ip_eclampsia_35']),
                Predictor('la_parity').when('0', params['rr_ip_eclampsia_nullip'])),

             'eclampsia_pp': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_ip_eclampsia'],
                Predictor('age_years').when('.between(30,34)', params['rr_ip_eclampsia_30_34']),
                Predictor('age_years').when('>35', params['rr_ip_eclampsia_35']),
                Predictor('la_parity').when('0', params['rr_ip_eclampsia_nullip'])),

             'antepartum_haem_ip': LinearModel(
                LinearModelType.MULTIPLICATIVE, # TODO: separate causal influence in praevia and abruption
                params['prob_aph'],
                Predictor('la_obstructed_labour').when(True, params['rr_aph_pl_ol'])),

             'postpartum_haem_pp': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_pph'],
                Predictor('la_obstructed_labour').when(True, params['rr_pph_pl_ol'])),

             'uterine_rupture_ip': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_uterine_rupture'],
                Predictor('la_parity').when('>4', params['rr_ur_grand_multip']),
                Predictor('la_previous_cs_delivery').when(True, params['rr_ur_prev_cs']),
                Predictor('la_obstructed_labour').when(True, params['rr_ur_ref_ol'])),

             'care_seeking': LinearModel(
                LinearModelType.LOGISTIC,  # TODO: rough cut paper, would seeking care be better than % home birth?
                params['odds_homebirth'],  # TODO: need to make these paramters
                Predictor('li_mar_stat').when('1', params['or_homebirth_unmarried']).when('3',
                                                                                    params['or_homebirth_unmarried']),
                Predictor('li_wealth').when('4', params['or_homebirth_wealth_4']).when('5',
                                                                                       params['or_homebirth_wealth_5']),
                 # wealth levels in the paper are different
                Predictor('li_urban').when(True, params['or_homebirth_urban']))
             }

        # TODO: do we need an equation for post term labour
        # TODO: we could hard code these predictors to reduce the number of actual parameters?

    def initialise_population(self, population):
        df = population.props
        params= self.parameters

        df.loc[df.sex == 'F', 'la_current_labour_successful_induction'] = 'not_induced'
        df.loc[df.sex == 'F', 'la_currently_in_labour'] = False
        df.loc[df.sex == 'F', 'la_intrapartum_still_birth'] = False
        df.loc[df.sex == 'F', 'la_parity'] = 0
        df.loc[df.sex == 'F', 'la_previous_cs_delivery'] = False
        df.loc[df.sex == 'F', 'la_has_previously_delivered_preterm'] = False
        df.loc[df.sex == 'F', 'la_due_date_current_pregnancy'] = pd.NaT
        df.loc[df.sex == 'F', 'la_obstructed_labour'] = False
        df.loc[df.sex == 'F', 'la_obstructed_labour_disab'] = False
        df.loc[df.sex == 'F', 'la_antepartum_haem'] = False
        df.loc[df.sex == 'F', 'la_antepartum_haem_disab'] = False
        df.loc[df.sex == 'F', 'la_uterine_rupture'] = False
        df.loc[df.sex == 'F', 'la_uterine_rupture_disab'] = False
        df.loc[df.sex == 'F', 'la_eclampsia'] = False
        df.loc[df.sex == 'F', 'la_eclampsia_disab'] = False
        df.loc[df.sex == 'F', 'la_postpartum_haem'] = False
        df.loc[df.sex == 'F', 'la_postpartum_haem_disab'] = False
        df.loc[df.sex == 'F', 'la_maternal_death'] = False
        df.loc[df.sex == 'F', 'la_maternal_death_date'] = pd.NaT

#  ----------------------------ASSIGNING PARITY AT BASELINE ----------------------------------------------------------

        # TODO: reformat with linear model? consider how best to to draw in real 2010 data
        # Current equation is an unweighted draw that just gives the same parity to every age group

        df.la_parity = params['la_labour_equations']['parity'].predict(df.loc[df.is_alive & (df.sex == 'F')])

    def initialise_simulation(self, sim):

        event = LabourLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def on_birth(self, mother_id, child_id):

        df = self.sim.population.props

        if df.at[child_id, 'sex'] == 'F':
            df.at[child_id, 'la_due_date_current_pregnancy'] = pd.NaT
            df.at[child_id, 'la_currently_in_labour'] = pd.NaT
            df.at[child_id, 'la_current_labour_successful_induction'] = 'not_induced'
            df.at[child_id, 'la_intrapartum_still_birth'] = False
            df.at[child_id, 'la_parity'] = 0
            df.at[child_id, 'la_previous_cs_delivery'] = False
            df.at[child_id, 'la_has_previously_delivered_preterm'] = False
            df.at[child_id, 'la_obstructed_labour'] = False
            df.at[child_id, 'la_obstructed_labour_disab'] = False
            df.at[child_id, 'la_antepartum_haem'] = False
            df.at[child_id, 'la_antepartum_haem_disab'] = False
            df.at[child_id, 'la_uterine_rupture'] = False
            df.at[child_id, 'la_uterine_rupture_disab'] = False
            df.at[child_id, 'la_sepsis'] = False
            df.at[child_id, 'la_sepsis_disab'] = False
            df.at[child_id, 'la_eclampsia'] = False
            df.at[child_id, 'la_eclampsia_disab'] = False
            df.at[child_id, 'la_postpartum_haem'] = False
            df.at[child_id, 'la_postpartum_haem_disab'] = False
            df.at[child_id, 'la_maternal_death'] = False
            df.at[child_id, 'la_maternal_death_date'] = pd.NaT

        # If a mothers labour has resulted in an intrapartum still birth her child is still generated by the simulation
        # but the death is recorded through the InstantaneousDeath function

        # Log only live births:
        if ~df.at[mother_id, 'la_intrapartum_still_birth']:
            logger.info('%s|live_births|%s',
                        self.sim.date,
                        {'mother': mother_id,
                         'child': child_id,
                         'mother_age': df.at[mother_id, 'age_years']})

        if df.at[mother_id, 'la_intrapartum_still_birth']:
            #  N.B this will only record intrapartum stillbirth
            death = demography.InstantaneousDeath(self.sim.modules['Demography'], child_id,
                                                  cause='Stillbirth')
            self.sim.schedule_event(death, self.sim.date)

            # This property is then reset in case of future pregnancies/stillbirths
            df.loc[mother_id, 'la_intrapartum_still_birth'] = False

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.info('This is Labour, being alerted about a health system interaction '
                    'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):

        # TODO: Refine disability levels (could include severity) and explore a more elegant solution involving less
        #  properties
        # TODO: Make sure that value >1.0 is being reported as current fix is temp

        logger.debug('This is Labour reporting my health values')

        df = self.sim.population.props  # shortcut to population properties data frame
        p = self.parameters

        health_values_1 = df.loc[df.is_alive, 'la_obstructed_labour_disab'].map(
            {False: 0, True: p['la_daly_wts']['obstructed_labour']})
        health_values_1.name = 'Obstructed Labour'

        health_values_2 = df.loc[df.is_alive, 'la_eclampsia_disab'].map(
            {False: 0, True: p['la_daly_wts']['eclampsia']})
        health_values_2.name = 'Eclampsia'

        health_values_3 = df.loc[df.is_alive, 'la_sepsis_disab'].map(
            {False: 0, True: p['la_daly_wts']['maternal_sepsis']})
        health_values_3.name = 'Maternal Sepsis'

        health_values_4 = df.loc[df.is_alive, 'la_antepartum_haem_disab'].map(  # TODO: consider severity
            {False: 0, True: p['la_daly_wts']['haemorrhage_severe']})
        health_values_4.name = 'Antepartum Haemorrhage'

        health_values_5 = df.loc[df.is_alive, 'la_postpartum_haem_disab'].map(  # TODO: consider severity
            {False: 0, True: p['la_daly_wts']['haemorrhage_severe']})
        health_values_5.name = 'Postpartum Haemorrhage'

        health_values_6 = df.loc[df.is_alive, 'la_uterine_rupture_disab'].map(  # TODO: consider severity
            {False: 0, True: p['la_daly_wts']['haemorrhage_severe']})
        health_values_6.name = 'Uterine Rupture'

        health_values_df = pd.concat([health_values_1.loc[df.is_alive], health_values_2.loc[df.is_alive],
                                      health_values_3.loc[df.is_alive], health_values_4.loc[df.is_alive],
                                      health_values_5.loc[df.is_alive], health_values_6.loc[df.is_alive]], axis=1)

        # Must not have one person with more than 1.00 daly weight
        # Hot fix - scale such that sum does not exceed one.
        scaling_factor = (health_values_df.sum(axis=1).clip(lower=0, upper=1) /
                          health_values_df.sum(axis=1)).fillna(1.0)
        health_values_df = health_values_df.multiply(scaling_factor, axis=0)

        return health_values_df

    # ===================================== LABOUR SCHEDULER ==========================================================

    def set_date_of_labour(self, individual_id):
        """This function, called within contraception, uses linear equations to determine a womans liklihood of preterm,
        postterm or term labour and sets their future date of labour accordingly"""

        df = self.sim.population.props
        params = self.parameters
        logger.debug('person %d is having their labour scheduled on date %s', individual_id, self.sim.date)

        # Using the linear equations defined above we calculate this womans individual risk of early and late preterm
        # labour
        eptb_prob = params['la_labour_equations']['early_preterm_birth'].predict(df.loc[[individual_id]]).values
        lptb_prob = params['la_labour_equations']['late_preterm_birth'].predict(df.loc[[individual_id]]).values

        # We then use a random draw to determine if the woman will go into preterm labour and how early she will deliver
        # We store this draw as a variable so the result can be compared against both probabilities
        random_draw = self.rng.random_sample()
        if (random_draw < lptb_prob) & (random_draw > eptb_prob):
            df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] + \
                                                                        pd.Timedelta(
                                                                            int(self.rng.random_integers(34, 36)),
                                                                            unit='W')

        elif random_draw < eptb_prob or ((random_draw < eptb_prob) & (random_draw < lptb_prob)):
             df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] + \
                                                                        pd.Timedelta(
                                                                            int(self.rng.random_integers(24, 33)),
                                                                            unit='W')

        # For women who will deliver after term we apply a risk of post term birth
        elif random_draw > lptb_prob:
            if self.rng.random_sample() < params['prob_potl']:
                df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] \
                                                                            + pd.Timedelta(
                        int(self.rng.random_integers(42, 46)), unit='W')
            else:
                df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] \
                                                                            + pd.Timedelta(
                        int(self.rng.random_integers(37, 41)), unit='W')

            # Here we check that no one can go into labour before 24 weeks gestation
        days_until_labour = df.at[individual_id, 'la_due_date_current_pregnancy'] - self.sim.date
        assert days_until_labour >= pd.Timedelta(168, unit='d')

        # and then we schedule the labour for that womans due date
        self.sim.schedule_event(LabourOnsetEvent(self, individual_id),
                                df.at[individual_id, 'la_due_date_current_pregnancy'])

    # ===================================== HELPER FUNCTIONS ==========================================================

    def eval(self, eq, person_id):
        """Compares the result of a specific linear equation with a random draw providing a boolean for the outcome
        under examination"""
        return self.rng.random_sample() < eq.predict(self.sim.population.props.loc[[person_id]]).values

    def set_home_birth_complications(self, individual_id, labour_stage, complication):
        """Uses the result of a linear equation to determine the probability of a certain complication, stores the
        probability for women delivery in facility and applies the probability for women delivering at home.
        Sets complication properties according to the result"""
        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        params = self.parameters

        if mni[individual_id]['delivery_setting'] == 'FD':
            mni[individual_id][f'risk_{labour_stage}_{complication}'] = \
                params['la_labour_equations'][f'{complication}_{labour_stage}'].predict(df.loc[[individual_id]]).values
        else:
            if self.eval(params['la_labour_equations'][f'{complication}_{labour_stage}'], individual_id):
                mni[individual_id][f'{complication}'] = True
                df.at[individual_id, f'la_{complication}'] = True
                df.at[individual_id, f'la_{complication}_disab'] = True

                logger.debug(f'person %d has developed {complication} in the community on date %s',
                             individual_id, self.sim.date)

                logger.info(f'%s|{complication}|%s', self.sim.date,
                            {'age': df.at[individual_id, 'age_years'],
                             'person_id': individual_id})

    def set_complications_during_facility_birth(self, person_id, complication, labour_stage, treatment_hsi):
        """Using each womans individual risk of a complication (which may have been modified by treatment) this function
        determines if she will experience a complication during her facility delivery. If so, additional treatment is
        scheduled"""

        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        rng = self.rng

        if rng.random_sample() < mni[person_id][f'risk_{labour_stage}_{complication}']:
            df.at[person_id, f'la_{complication}'] = True
            df.at[person_id, f'la_{complication}_disab'] = True
            mni[person_id][f'{complication}'] = True
            logger.debug(f'person %d is experiencing {complication} in a health facility', person_id)

            logger.info(f'%s|{complication}|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})

            self.sim.modules['HealthSystem'].schedule_hsi_event(treatment_hsi,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

            logger.info(f'This is HSI_Labour_PresentsForSkilledAttendanceInLabour: scheduling immediate additional '
                        f'treatment for {complication} for person %d', person_id)


class LabourOnsetEvent(Event, IndividualScopeEventMixin):
    """This is the LabourOnsetEvent. It is scheduled by the set_date_of_labour function. It represents the start of a
    womans labour. Here we assign a "type" of labour based on gestation (i.e. early preterm), we create a dictionary to
    store additional variables important to labour and HSIs, and we determine if a woman will seek care. This event
    schedules  the LabourAtHome event and the HSI_Labour_PresentsForSkilledAttendance at birth (depending on care
     seeking), the BirthEvent and the LabourDeathEvent"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # Here we populate the maternal and newborn info dictionary with baseline values before the womans labour begins
        mni = self.module.mother_and_newborn_info

        # TODO: review in context of properties- ensure what should be a property IS one, and what SHOULDN'T be isnt.
        mni[individual_id] = {'labour_state': None,  # Term Labour (TL), Early Preterm (EPTL), Late Preterm (LPTL) or
                              # Post Term (POTL)
                              'delivery_setting': None,  # Facility Delivery (FD) or Home Birth (HB)
                              'induced_labour': False,
                              'referred_for': None,  # Induction (I) or Caesarean (CS)
                              'cord_prolapse': False,
                              'PROM': False,
                              'PPROM': False,
                              'risk_ip_obstructed_labour': params['prob_pl_ol'],
                              'labour_is_currently_obstructed': False,  # True (T) or False (F)
                              'labour_has_previously_been_obstructed': False,
                              'risk_ip_sepsis': params['prob_ip_sepsis'],
                              'risk_pp_sepsis': params['prob_pp_sepsis'],
                              'sepsis': False,  # True (T) or False (F)
                              'sepsis_pp': False,  # True (T) or False (F) #do we need this
                              'source_sepsis': None,  # Obstetric (O) or Non-Obstetric (NO)
                              'risk_ip_antepartum_haem': params['prob_aph'],
                              'antepartum_haem': False,  # True (T) or False (F)
                              'source_aph': None,  # Placenta Praevia (PP) or Placental Abruption (PA) (Other?)
                              'units_transfused': 0,
                              'risk_ip_eclampsia': params['prob_ip_eclampsia'],
                              'risk_pp_eclampsia': params['prob_pp_eclampsia'],
                              'eclampsia': False,  # True (T) or False (F)
                              'eclampsia_pp': False,  # True (T) or False (F)
                              'risk_ip_uterine_rupture': params['prob_uterine_rupture'],
                              'uterine_rupture': False,  # True (T) or False (F)
                              'grade_of_UR': 'X',  # Partial (P) or Complete (C)
                              'risk_pp_postpartum_haem': params['prob_pph'],
                              'postpartum_haem': False,  # True (T) or False (F)
                              'source_pph': None,  # Uterine Atony (UA) or Retained Products/Placenta (RPP)
                              'severity_pph': None,
                              'risk_newborn_sepsis': params['prob_neonatal_sepsis'],
                              'risk_newborn_ba': params['prob_neonatal_birth_asphyxia'],
                              #  Should this just be risk of asyphixa
                              'mode_of_delivery': None,  # Vaginal Delivery (VD),Vaginal Delivery Induced (VDI),
                              # Assisted Vaginal Delivery Forceps (AVDF) Assisted Vaginal Delivery Ventouse (AVDV)
                              # Caesarean Section (CS)
                              'death_in_labour': False,  # True (T) or False (F)
                              'cause_of_death_in_labour': [],  # Appended list of cause/causes of death
                              # TODO: should we do the same thing for still birth? Would help with mapping
                              'stillbirth_in_labour': False,  # True (T) or False (F)
                              'death_postpartum': False}  # True (T) or False (F)

# ===================================== LABOUR STATE  ==================================================================
        logger.debug('person %d has just reached LabourOnsetEvent on %s', individual_id, self.sim.date)

        # Debug logging to highlight the women who have miscarried/had an abortion/stillbirth who still come to the
        # event
        if ~df.at[individual_id, 'is_pregnant']:
            logger.debug('person %d has just reached LabourOnsetEvent on %s but is no longer pregnant',
                         individual_id, self.sim.date)

        # The event is conditioned on the woman being pregnant, today being her due date and being alive
        # We don't use assert functions to ensure the due date is correct as women who lose their pregnancy will still
        # have this event scheduled
        if df.at[individual_id, 'is_pregnant'] & df.at[individual_id, 'is_alive'] & \
            (df.at[individual_id, 'la_due_date_current_pregnancy'] == self.sim.date):
            df.at[individual_id, 'la_currently_in_labour'] = True

            # If a woman has been induced/attempted induction she will already be in a facility therefore delivery_
            # setting is set to facility
            if (df.at[individual_id, 'la_current_labour_successful_induction'] == 'failed_induction') or \
                (df.at[individual_id, 'la_current_labour_successful_induction'] == 'successful_induction'):
                mni[individual_id]['delivery_setting'] = 'FD'
            # We then store her induction status if successful
            if df.at[individual_id, 'la_current_labour_successful_induction'] == 'successful_induction':
                mni[individual_id]['induced_labour'] = True

            # Now we use gestational age to categorise the 'labour_state'
            if df.at[individual_id, 'is_pregnant'] & df.at[individual_id, 'is_alive']:
                if 37 <= df.at[individual_id, 'ps_gestational_age_in_weeks'] < 42:
                    mni[individual_id]['labour_state'] = 'TL'

                elif 24 <= df.at[individual_id, 'ps_gestational_age_in_weeks'] < 34:
                    mni[individual_id]['labour_state'] = 'EPTL'
                    df.at[individual_id, 'la_has_previously_delivered_preterm'] = True

                elif 37 > df.at[individual_id, 'ps_gestational_age_in_weeks'] >= 34:
                    mni[individual_id]['labour_state'] = 'LPTL'
                    df.at[individual_id, 'la_has_previously_delivered_preterm'] = True

                elif df.at[individual_id, 'ps_gestational_age_in_weeks'] > 41:
                    mni[individual_id]['labour_state'] = 'POTL'
                    logger.info('%s|postterm_birth|%s', self.sim.date,
                                {'age': df.at[individual_id, 'age_years'],
                                 'person_id': individual_id})

                labour_state = mni[individual_id]['labour_state']
                logger.debug(f'This is LabourOnsetEvent, person %d has now gone into {labour_state} on date %s',
                             individual_id, self.sim.date)

# ======================================= PROBABILITY OF HOME DELIVERY =================================================
                facility_delivery = HSI_Labour_PresentsForSkilledAttendanceInLabour(self.module,
                                                                                    person_id=individual_id)

                # As women who have presented for induction are already in a facility we exclude them, then apply a
                # probability of home birth
                if self.module.eval(params['la_labour_equations']['care_seeking'], individual_id) & \
                   (df.at[individual_id, 'la_current_labour_successful_induction'] == 'not_induced'):
                    mni[individual_id]['delivery_setting'] = 'HB'
                    self.sim.schedule_event(LabourAtHomeEvent(self.module, individual_id), self.sim.date)
                    logger.debug('This is LabourEvent,  person %d will not seek care in labour and will deliver at home'
                                 , individual_id)
                else:
                    mni[individual_id]['delivery_setting'] = 'FD'
                    logger.info(
                        'This is LabourOnsetEvent, scheduling HSI_Labour_PresentsForSkilledAttendanceInLabour on date'
                        ' %s for person %d as they have chosen to seek care for delivery', self.sim.date, individual_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(facility_delivery,
                                                                        priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=1))

                # Here we schedule delivery care for women who have already sought care for induction, whether or not
                # that induction was successful
                if (df.at[individual_id, 'la_current_labour_successful_induction'] == 'failed_induction') or \
                   (df.at[individual_id, 'la_current_labour_successful_induction'] == 'successful_induction'):
                    self.sim.modules['HealthSystem'].schedule_hsi_event(facility_delivery,
                                                                        priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=1))

# ======================================== SCHEDULING BIRTH AND DEATH EVENTS ==========================================

            # Here we schedule the birth event for 2 days after labour- we do this prior to the death event as women who
            # die but still deliver a live child will pass through birth event
                due_date = df.at[individual_id, 'la_due_date_current_pregnancy']
                self.sim.schedule_event(BirthEvent(self.module, individual_id), due_date + DateOffset(days=3))
                logger.debug('This is LabourOnsetEvent scheduling a birth on date %s to mother %d', due_date,
                             individual_id)

            # We schedule all women to move through the death event where those who have developed a complication that
            # hasn't been treated or treatment has failed will have a case fatality rate applied
                self.sim.schedule_event(LabourDeathEvent(self.module, individual_id), self.sim.date +
                                        DateOffset(days=2))

                logger.debug('This is LabourOnsetEvent scheduling a potential death on date %s for mother %d',
                             self.sim.date, individual_id)

                # Here we set the due date of women who have been induced to pd.NaT so they dont go into labour twice
                if df.at[individual_id, 'la_current_labour_successful_induction']:
                    df.at[individual_id, 'la_due_date_current_pregnancy'] = pd.NaT


class LabourAtHomeEvent(Event, IndividualScopeEventMixin):
    """This is the LabourAtHomeEvent. It is scheduled by the LabourOnsetEvent for women who will not seek care. This
    event applies the probability that women delivering at home will experience complications, makes the appropriate
    changes to the data frame . Women who seek care, but for some reason are unable to deliver at a facility will return
    to this event"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        logger.debug('person %d has is now going to deliver at home', individual_id)

    # ============================ INDIVIDUAL RISK OF COMPLICATIONS DURING LABOUR =====================================

        # DEAL WITH VERY EARLY PRETERM BIRTHS?!?!

    # ====================================== STATUS OF MEMBRANES ======================================================

        # Here we apply a risk that this woman's labour was preceded by premature rupture of membranes, in preterm
        # women this has likely predisposed their labour
        if (mni[individual_id]['labour_state'] == 'EPTL') or (mni[individual_id]['labour_state'] == 'LPTL'):
            if self.module.rng.random_sample() < params['prob_prom']:
                mni[individual_id]['PROM'] = True
        # TODO: term labour can have prom also (also this should be an antenatal thing?)

    # ===================================  APPLICATION OF COMPLICATIONS ===============================================

        # Using the complication_application function we move through each complication in turn determining if a woman
        # will experience any of these if she has delivered at home, and storing the risk so it can be modified by
        # skilled birth attendance for women delivering in a facility (n.b 'ip' == intrapartum

        self.module.set_home_birth_complications(individual_id, labour_stage='ip', complication='obstructed_labour')
        if df.at[individual_id, 'la_obstructed_labour']:  # there's a neater way of doing this
            mni[individual_id]['labour_is_currently_obstructed'] = True
            mni[individual_id]['labour_has_previously_been_obstructed'] = True
        # TODO: As OL is a contraindication of induction, should we skip it here for induced women?
        #  late diagnosis of obstruction is possible surely

        self.module.set_home_birth_complications(individual_id, params, labour_stage='ip',
                                                 complication='antepartum_haem')
        self.module.set_home_birth_complications(individual_id, params, labour_stage='ip', complication='sepsis')
        self.module.set_home_birth_complications(individual_id, params, labour_stage='ip', complication='eclampsia')
        self.module.set_home_birth_complications(individual_id, params, labour_stage='ip',
                                                 complication='uterine_rupture')


    # TODO: here we need to use the symptom manager to determine if women who are delivering at home with comps
    #  will seek care


class BirthEvent(Event, IndividualScopeEventMixin):
    """This is the BirthEvent. It is scheduled by LabourOnsetEvent. For women who survived labour, the appropriate
    variables are reset/updated and the function do_birth is executed. This event schedules PostPartumLabourEvent for
    those women who have survived"""

    def __init__(self, module, mother_id):
        super().__init__(module, person_id=mother_id)

    def apply(self, mother_id):

        # This event tells the simulation that the woman's pregnancy is over and generates the new child in the
        # data frame
        logger.info('@@@@ A Birth is now occuring, to mother %d', mother_id)
        df = self.sim.population.props
        mni = self.module.mother_and_newborn_info

        df.at[mother_id, 'la_currently_in_labour'] = False

        # If the mother is alive and still pregnant we generate a live child and the woman is scheduled to move to the
        # postpartum event to determine if she experiences any additional complications
        if df.at[mother_id, 'is_alive'] and df.at[mother_id, 'is_pregnant']:
            #  TODO: Ensure women who have had an IP still birth still move to postpartum event
            self.sim.do_birth(mother_id)
            df.at[mother_id, 'ps_gestational_age_in_weeks'] = 0
            df.at[mother_id, 'is_pregnant'] = False
            df.at[mother_id, 'date_of_last_pregnancy'] = pd.NaT

            logger.debug('This is BirthEvent scheduling mother %d to undergo the PostPartumEvent following birth',
                         mother_id)
            self.sim.schedule_event(PostpartumLabourEvent(self.module, mother_id),
                                    self.sim.date)

        # As only live births contribute to parity we excluded women who have had an IP stillbirth
        if df.at[mother_id, 'is_alive'] and df.at[mother_id, 'is_pregnant'] & \
           ~df.at[mother_id, 'la_intrapartum_still_birth']:
            df.at[mother_id, 'la_parity'] += 1

        # If the mother has died during childbirth the child is still generated with is_alive=false to monitor
        # stillbirth rates. She will not pass through the postpartum complication events
        if ~df.at[mother_id, 'is_alive'] & df.at[mother_id, 'is_pregnant'] & mni[mother_id]['death_in_labour']:
            self.sim.do_birth(mother_id)
            df.at[mother_id, 'is_pregnant'] = False


class PostpartumLabourEvent(Event, IndividualScopeEventMixin):
    """This is PostpartumLabour event. It is scheduled by BirthEvent immediately following birth. This event applies the
    probability that women delivering at home will experience complications, makes the appropriate changes to the data
    frame. This event schedules the PostPartumDeathEvent (3 days post this event)and the DisabilityResetEvent"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self
        mni = self.module.mother_and_newborn_info

        # Here we use the complication_application function to determine if a women who has survived labour will
        # experience any further complications (risk is stored for facility deliveries)
        if df.at[individual_id, 'is_alive']:
            self.module.set_home_birth_complications(individual_id, params, labour_stage='pp',
                                                     complication='postpartum_haem')
            self.module.set_home_birth_complications(individual_id, params, labour_stage='pp', complication='sepsis')
            self.module.set_home_birth_complications(individual_id, params, labour_stage='pp', complication='eclampsia')

        # If a woman has delivered in a facility we schedule her to now receive additional care following birth
            if mni[individual_id]['delivery_setting'] == 'FD':
                logger.info('This is PostPartumEvent scheduling HSI_Labour_ReceivesCareForPostpartumPeriod for person '
                            '%d on date %s', individual_id, self.sim.date)
                
                event = HSI_Labour_ReceivesCareForPostpartumPeriod(self.module, person_id=individual_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))
                # TODO: same issue for women who seek care but cant be seek- comps wont be allocated!

            # We schedule all women to then go through the death event where those with untreated/unsuccessfully treated
            # complications may experience death
                
            self.sim.schedule_event(PostPartumDeathEvent(self.module, individual_id), self.sim.date
                                    + DateOffset(days=3))

            logger.info('This is PostPartumEvent scheduling a potential death for person %d on date %s', individual_id,
                        self.sim.date + DateOffset(days=3))  # Date offset to allow for interventions

            # Here we schedule women to an event which resets 'daly' disability associated with delivery complications
            self.sim.schedule_event(DisabilityResetEvent(self.module, individual_id),
                                    self.sim.date + DateOffset(months=1))

            # TODO: symptom manager for women who have complications at home


class LabourDeathEvent (Event, IndividualScopeEventMixin):
    """This is the LabourDeathEvent. It is scheduled by the LabourOnsetEvent. This event determines if women who have
    experienced complications in labour will die or experience an intrapartum stillbirth."""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info
        rng = self.module.rng

        # Currently we apply an untreated case fatality ratio (dummy values presently) who have experienced a
        # complication
        # Similarly we apply a risk of still birth associated with each complication

        def set_maternal_death_status_intrapartum(cause):
            logger.debug(f'{cause} has contributed to person %d death during labour', individual_id)
            if rng.random_sample() < params[f'cfr_{cause}']:
                mni[individual_id]['death_in_labour'] = True
                mni[individual_id]['cause_of_death_in_labour'].append(cause)
                df.at[individual_id, 'la_maternal_death'] = True
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date

                # then if she does die, we determine if the child will still survive
                if rng.random_sample()< params[f'prob_still_birth_{cause}_md']:
                    df.at[individual_id, 'la_intrapartum_still_birth'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True
                    df.at[individual_id, 'ps_previous_stillbirth'] = True

                # Otherwise we just determine if this complication will lead to a stillbirth
                else:
                    if rng.random_sample() < params[f'prob_still_birth_{cause}']:
                        df.at[individual_id, 'la_intrapartum_still_birth'] = True
                        mni[individual_id]['stillbirth_in_labour'] = True
                        df.at[individual_id, 'ps_previous_stillbirth'] = True

            # First we determine if the mother will die due to her complication
            if mni[individual_id]['labour_is_currently_obstructed']:
                set_maternal_death_status_intrapartum(cause='obstructed_labour')

            if mni[individual_id]['eclampsia']:
                set_maternal_death_status_intrapartum(cause='eclampsia')

            if mni[individual_id]['antepartum_haem']:
                set_maternal_death_status_intrapartum(cause='aph')

            if mni[individual_id]['sepsis']:
                set_maternal_death_status_intrapartum(cause='sepsis')

            if mni[individual_id]['uterine_rupture']:
                set_maternal_death_status_intrapartum(cause='uterine_rupture')

            # TODO: Will we apply a reduced CFR in the instance of unsuccessful interventions

            # Schedule death for women who die in labour
        if mni[individual_id]['death_in_labour']:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                  cause='labour'), self.sim.date)
            # TODO: amend cause= 'labour_' + [str(cause) + '_' for cause in list(mni[individual_id]
            #  [cause_of_death_in_labour]

            # Log the maternal death
            logger.info('This is LabourDeathEvent scheduling a death for person %d on date %s who died due to '
                        'intrapartum complications', individual_id, self.sim.date)

            logger.info('%s|labour_complications|%s', self.sim.date,
                        {'person_id': individual_id,
                         'labour_profile': mni[individual_id]})

        if df.at[individual_id, 'la_intrapartum_still_birth']:
            logger.info('@@@@ A Still Birth has occurred, to mother %s', individual_id)
            logger.info('%s|still_birth|%s', self.sim.date,
                        {'mother_id': individual_id})


class PostPartumDeathEvent (Event, IndividualScopeEventMixin):
    """This is the PostPartumDeathEvent. It is scheduled by the PostpartumLabourEvent. This event determines if women
    who have experienced complications following labour will die. This event schedules the DiseaseResetEvent for
    surviving women"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self
        mni = self.module.mother_and_newborn_info
        rng = self.module.rng

        # We apply the same structure as with the LabourDeathEvent to women who experience postpartum complications
        # if df.at[individual_id, 'la_eclampsia']:

        def set_maternal_death_status_postpartum(cause):
            if self.module.rng.random_sample() < params[f'cfr_pp_{cause}']:
                logger.debug(f'{cause} has contributed to person %d death following labour', individual_id)
                mni[individual_id]['death_postpartum'] = True
                df.at[individual_id, 'la_maternal_death'] = True
                mni[individual_id]['cause_of_death_in_labour'].append(f'{cause}_postpartum')
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date
        
        if mni[individual_id]['eclampsia_pp']:
            set_maternal_death_status_postpartum(cause='eclampsia')

        if mni[individual_id]['postpartum_haem']:
            set_maternal_death_status_postpartum(cause='pph')

        if mni[individual_id]['sepsis_pp']:
            set_maternal_death_status_postpartum(cause='sepsis')

        if mni[individual_id]['death_postpartum']:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                  cause='postpartum labour'), self.sim.date)

    # TODO: amend cause= 'labour_' + [str(cause) + '_' for cause in list(mni[individual_id][cause_of_death_in_labour]

            logger.debug('This is PostPartumDeathEvent scheduling a death for person %d on date %s who died due to '
                         'postpartum complications', individual_id,
                         self.sim.date)

            logger.debug('%s|labour_complications|%s', self.sim.date,
                         {'person_id': individual_id,
                          'labour_profile': mni[individual_id]})
        else:
            # Surviving women pass through the DiseaseResetEvent to ensure all complication variable are set to false
            # TODO: Consider how best to deal with complications that are long lasting.
            self.sim.schedule_event(DiseaseResetEvent(self.module, individual_id),
                                    self.sim.date + DateOffset(weeks=1))

            logger.debug('%s|labour_complications|%s', self.sim.date,
                         {'person_id': individual_id,
                          'labour_profile': mni[individual_id]})


class DisabilityResetEvent (Event, IndividualScopeEventMixin):
    """This is the DisabilityResetEvent. It is scheduled by the PostPartumLabourEvent. This event resets a woman's
    disability properties within the data frame """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props

        # TODO: Confirm this allows enough time for model to count DALYs before resetting

        # Here we turn off all the properties which are used to count DALYs
        if df.at[individual_id, 'is_alive']:
            logger.debug('person %d is having their disability status reset', individual_id)

            df.at[individual_id, 'la_sepsis_disab'] = False
            df.at[individual_id, 'la_obstructed_labour_disab'] = False
            df.at[individual_id, 'la_uterine_rupture_disab'] = False
            df.at[individual_id, 'la_eclampsia_disab'] = False
            df.at[individual_id, 'la_antepartum_haem_disab'] = False
            df.at[individual_id, 'la_postpartum_haem_disab'] = False


class DiseaseResetEvent (Event, IndividualScopeEventMixin):
    """This is the DisabilityResetEvent. It is scheduled by the PostPartumDeathEvent. This event resets a woman's
    disease properties within the data frame """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props

        # This event ensures that for women who have survived delivery but have suffered a complication in which
        # treatment was unsuccessful have their diseases variables reset

        if df.at[individual_id, 'is_alive']:
            logger.debug('person %d is having their maternal disease status reset', individual_id)

            df.at[individual_id, 'la_sepsis'] = False
            df.at[individual_id, 'la_obstructed_labour'] = False
            df.at[individual_id, 'la_antepartum_haem'] = False
            df.at[individual_id, 'la_uterine_rupture'] = False
            df.at[individual_id, 'la_eclampsia'] = False
            df.at[individual_id, 'la_postpartum_haem'] = False

# ======================================================================================================================
# ================================ HEALTH SYSTEM INTERACTION EVENTS ====================================================
# ======================================================================================================================
    # TODO: Discuss with the team the potential impact of deviating from guidelines- at present if a woman presents,
    #  staff time is available and resources are available she gets all needed interventions (this may not be the case,
    #  question of quality


class HSI_Labour_PresentsForInductionOfLabour(HSI_Event, IndividualScopeEventMixin):
    """ This is the HSI PresentsForInductionOfLabour. Currently it IS NOT scheduled, but will be scheduled by the
    AntenatalCare module pending its completion. It will be responsible for the intervention of induction for women who
    are postterm. As this intervention will start labour- it will schedule either the LabourOnsetEvent or the caesarean
    section event depending on success"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)
        # TODO: await antenatal care module to schedule induction
        # TODO: women with praevia, obstructed labour, should be induced
        self.TREATMENT_ID = 'Labour_PresentsForInductionOfLabour'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1
        # TODO: review appt footprint as this wont include midwife time?

        # Define the necessary information for an HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug('This is HSI_Labour_PresentsForInductionOfLabour, person %d is attending a health facility to have'
                     'their labour induced on date %s', person_id, self.sim.date)

        # TODO: discuss squeeze factors/ consumable back up with TC

        df = self.sim.population.props
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        # Initial request for consumables needed
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_induction = pd.unique(consumables.loc[consumables[
                                                       'Intervention_Pkg'] ==
                                                       'Induction of labour (beyond 41 weeks)',
                                                       'Intervention_Pkg_Code'])[0]
        # TODO: review induction guidelines to confirm appropriate 1st line/2nd line consumable use

        consumables_needed = {'Intervention_Package_Code': [{pkg_code_induction: 1}],
                              'Item_Code': [], }

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed)

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_induction]:
            logger.debug('pkg_code_induction is available, so use it.')
            # TODO: reschedule if consumables aren't available at this point in time?
        else:
            logger.debug('PkgCode1 is not available, so can' 't use it.')

        # We use a random draw to determine if this womans labour will be successfully induced
        # Indications: Post term, eclampsia, severe preeclampsia, mild preeclampsia at term, PROM > 24 hrs at term
        # or PPROM > 34 weeks EGA, and IUFD.

        if self.module.rng.random_sample() < params['prob_successful_induction']:
            logger.info('Person %d has had her labour successfully induced', person_id)
            df.at[person_id, 'la_current_labour_successful_induction'].values[:] = 'successful_induction'

            self.sim.schedule_event(LabourOnsetEvent(self.module, person_id), self.sim.date)
            # TODO: scheduling
            # For women whose induction fails they will undergo caesarean section
        else:
            logger.info('Persons %d labour has been unsuccessful induced', person_id)
            df.at[person_id, 'la_current_labour_successful_induction'].values[:] = 'failed_induction'
            # TODO: schedule CS or second attempt induction? -- will need to lead to labour event

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
    #    actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_PresentsForInductionOfLabour: did not run')
        pass

    #  TODO: 3 HSIs, for each facility level (interventions in functions)??


class HSI_Labour_PresentsForSkilledAttendanceInLabour(HSI_Event, IndividualScopeEventMixin):
    """This is the HSI PresentsForSkilledAttendanceInLabour. This event is scheduled by the LabourOnset Event. This
    event manages initial care around the time of delivery including prophylactic interventions (i.e. clean birth
    practices). This event uses a womans stored risk of complications, which may be manipulated by treatment effects to
    determines if they will experience a complication during their labour in hospital. It is responsible for scheduling
    treatment HSIs for those complications. In the event that this HSI will not run, women are scheduled back to the
    LabourAtHomeEvent. This event is not finalised an both interventions and referral are subject to change"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)
        
        self.TREATMENT_ID = 'Labour_PresentsForSkilledAttendanceInLabour'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['NormalDelivery'] = 1

        # Define the necessary information for an HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2
        self.ALERT_OTHER_DISEASES = []

    def did_not_run(self, person_id):
        mni = self.module.mother_and_newborn_info
        logger.debug('person %d sought care for a facility delivery but HSI_Labour_PresentsForSkilledAttendanceInLabour'
                     'could not run, they will now deliver at home', person_id)
        mni[person_id]['delivery_setting'] = 'HB'
        self.sim.schedule_event(LabourAtHomeEvent(self.module, person_id), self.sim.date)

    def apply(self, person_id, squeeze_factor):

        if squeeze_factor > 0.8:  # TODO: confirm
            self.module.did_not_run(person_id)

        logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: Providing initial skilled attendance '
                    'at birth for person %d on date %s', person_id, self.sim.date)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_sba_uncomp = pd.unique(consumables.loc[consumables[
                                                      'Intervention_Pkg'] ==
                                                  'Vaginal delivery - skilled attendance',
                                                  'Intervention_Pkg_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code_sba_uncomp: 1}],
            'Item_Code': []}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_sba_uncomp]:
            logger.debug('PkgCode1 is available, so use it.')
        else:
            logger.debug('PkgCode1 is not available, so can' 't use it.')
            # TODO: If delivery pack isn't available then birth will still occur but should have risk of sepsis?

        df = self.sim.population.props
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        # TODO: Discuss with Tim H the best way to capture which HCP will be attending this delivery and how that may
        #  affect outcomes?

    # ============================ CLEAN DELIVERY PRACTICES AT BIRTH ==================================================

        # First we apply the estimated impact of clean birth practices on maternal and newborn risk of sepsis

        adjusted_maternal_sepsis_risk = mni[person_id]['risk_ip_sepsis'] * \
            params['rr_maternal_sepsis_clean_delivery']
        mni[person_id]['risk_ip_sepsis'] = adjusted_maternal_sepsis_risk

        adjusted_newborn_sepsis_risk = mni[person_id]['risk_newborn_sepsis'] * \
            params['rr_newborn_sepsis_clean_delivery']
        mni[person_id]['risk_newborn_sepsis'] = adjusted_newborn_sepsis_risk

# =============================== SKILLED BIRTH ATTENDANCE EFFECT ======================================================
        # Then we apply the estimated effect of fetal surveillance methods on maternal/newborn outcomes

        # TODO: Discuss again with Tim C if it is worth trying to quantify this, as many evaluative studies look at
        #  provision of BEmOC etc. so will include the impact of the interventions

# ================================== INTERVENTIONS FOR PRE-EXSISTING CONDITIONS =====================================
        # Here we apply the effect of any 'upstream' interventions that may reduce the risk of intrapartum
        # complications/poor newborn outcomes

        # ------------------------------------- PROM ------------------------------------------------------------------
        # First we apply a risk reduction in likelihood of sepsis for women with PROM as they have received prophylactic
        # antibiotics
        if mni[person_id]['PROM']:
            treatment_effect = params['rr_sepsis_post_abx_prom']
            new_sepsis_risk = mni[person_id]['risk_ip_sepsis'] * treatment_effect
            mni[person_id]['risk_ip_sepsis'] = new_sepsis_risk

        # ------------------------------------- PREMATURITY # TODO: care bundle for women in preterm labour (pg 49)----
        # Here we apply the effect of interventions to improve outcomes of neonates born preterm

        # Antibiotics for group b strep prophylaxis: (are we going to apply this to all cause sepsis?)
        if mni[person_id]['labour_state'] == 'EPTL' or mni[person_id]['labour_state'] == 'LPTL':
            mni[person_id]['risk_newborn_sepsis'] = mni[person_id]['risk_newborn_sepsis'] * \
                                                    params['rr_newborn_sepsis_proph_abx']

        # TODO: STEROIDS - effect applied directly to newborns (need consumables here) store within maternal MNI
        # TOXOLYTICS !? - WHO advises against

# ===================================  COMPLICATIONS OF LABOUR ========================================================

    # Here, using the adjusted risks calculated following 'in-labour' interventions to determine which complications a
    # woman may experience and store those in the data frame

        self.module.set_complications_during_facility_birth(person_id, complication='obstructed_labour',
                                                            labour_stage='ip',
                                                            treatment_hsi=HSI_Labour_ReceivesCareForObstructedLabour
                                                            (self.module, person_id=person_id))
        if df.at[person_id, 'la_obstructed_labour']:
            mni[person_id]['labour_is_currently_obstructed'] = True
            mni[person_id]['labour_has_previously_been_obstructed'] = True

        self.module.set_complications_during_facility_birth(person_id, complication='eclampsia', labour_stage='ip',
                                                            treatment_hsi=
                                                            HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy
                                                            (self.module, person_id=person_id))

        self.module.set_complications_during_facility_birth(person_id, complication='antepartum_haem', labour_stage='ip',
                                                            treatment_hsi=
                                                            HSI_Labour_ReceivesCareForMaternalHaemorrhage
                                                            (self.module, person_id=person_id))

        self.module.set_complications_during_facility_birth(person_id, complication='sepsis', labour_stage='ip',
                                                            treatment_hsi=HSI_Labour_ReceivesCareForMaternalSepsis
                                                            (self.module, person_id=person_id))

        self.module.set_complications_during_facility_birth(person_id, complication='uterine_rupture', labour_stage='ip'
                                                            , treatment_hsi=HSI_Labour_ReceivesCareForObstructedLabour
                                                            (self.module, person_id=person_id))

        # TODO: issue- if we apply risk of both UR and OL here then we will negate the effect of OL treatment on
        #  reduction of incidence of UR

        # DUMMY ... (risk factors?)
        if self.module.rng.random_sample() < params['prob_cord_prolapse']:
            mni[person_id]['cord_prolapse'] = True

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT

        if mni[person_id]['uterine_rupture'] or mni[person_id]['eclampsia'] or mni[person_id]['antepartum_haem'] or\
           mni[person_id]['sepsis'] or mni[person_id]['labour_is_currently_obstructed']:
            actual_appt_footprint['NormalDelivery'] = actual_appt_footprint['CompDelivery'] * 1

        return actual_appt_footprint


class HSI_Labour_ReceivesCareForObstructedLabour(HSI_Event, IndividualScopeEventMixin):
    """This is the HSI ReceivesCareForObstructedLabour. This event is scheduled by the the HSI
    PresentsForSkilledAttendanceInLabour if a woman experiences obstructed Labour. This event contains the intervetions
    around assisted vaginal delivery. This event schedules HSI ReferredForSurgicalCareInLabour when assisted delivery
    fails. This event is not finalised an both interventions and referral are subject to change"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForObstructedLabour'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1  # TODO: to be discussed with TH/TC best appt footprint to use

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        # TODO: apply squeeze factor

        logger.info('This is HSI_Labour_ReceivesCareForObstructedLabour, management of obstructed labour for '
                    'person %d on date %s',
                    person_id, self.sim.date)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_obst_lab = pd.unique(consumables.loc[consumables[
                                                      'Intervention_Pkg'] ==
                                                      'Antibiotics for pPRoM',  # TODO: Obs labour package not working
                                                      'Intervention_Pkg_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code_obst_lab: 1}],
            'Item_Code': []}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_obst_lab]:
            logger.debug('pkg_code_obst_lab is available, so use it.')
        else:
            logger.debug('pkg_code_obst_lab is not available, so can' 't use it.')
            # TODO: This will need to be equipment by equipment- i.e. if no vacuum then forceps if none then caesarean?
            # TODO: add equipment to lines on consumable chart?

        df = self.sim.population.props  # shortcut to the dataframe
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

# =====================================  OBSTRUCTED LABOUR TREATMENT ==================================================

        # TODO: Differentiate between CPD and other?

        # For women in obstructed labour delivery is first attempted by vacuum, delivery mode is stored
        if mni[person_id]['labour_is_currently_obstructed']:
            if params['prob_deliver_ventouse'] > self.module.rng.random_sample():
                # df.at[person_id, 'la_obstructed_labour'] = False
                mni[person_id]['labour_is_currently_obstructed'] = False
                mni[person_id]['mode_of_delivery'] = 'AVDV'
                # add here effect of antibiotics?
            else:
                # If the vacuum is unsuccessful we apply the probability of successful forceps delivery
                if params['prob_deliver_forceps'] > self.module.rng.random_sample():
                    # df.at[person_id, 'la_obstructed_labour'] = False
                    mni[person_id]['labour_is_currently_obstructed'] = False
                    mni[person_id]['mode_of_delivery'] = 'AVDF'  # add here effect of antibiotitcs?

                # Finally if assisted vaginal delivery fails then CS is scheduled
                else:
                    event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                        priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=1))

                    logger.info(
                        'This is HSI_Labour_ReceivesCareForObstructedLabour referring person %d for a caesarean section'
                        ' following unsuccessful attempts at assisted vaginal delivery ', person_id)

                    # Symphysiotomy??? Does this happen in malawi

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        #    actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_PresentsForInductionOfLabour: did not run')
        pass


class HSI_Labour_ReceivesCareForMaternalSepsis(HSI_Event, IndividualScopeEventMixin):
    """This is the HSI ReceivesCareForMaternalSepsis. This event is scheduled by the either HSI
    PresentsForSkilledAttendanceInLabour or HSI ReceivesCareForPostpartumPeriod if a woman experiences maternal sepsis.
    This event deliveries antibiotics as an intervention. This event is not finalised an both interventions and referral
    are subject to change"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForMaternalSepsis'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        # TODO: Apply squeeze factor

        logger.info('This is HSI_Labour_ReceivesCareForMaternalSepsis, management of maternal sepsis for '
                    'person %d on date %s',
                    person_id, self.sim.date)

        df = self.sim.population.props
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_sepsis = pd.unique(consumables.loc[consumables[
                                                        'Intervention_Pkg'] ==
                                                    'Maternal sepsis case management',
                                                    'Intervention_Pkg_Code'])[0]

        consumables_needed = {'Intervention_Package_Code': [{pkg_code_sepsis: 1}], 'Item_Code': []}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)

        # TODO: consider 1st line/2nd line and efficacy etc etc

        # Treatment can only be delivered if appropriate antibiotics are available, if they're not the woman isn't
        # treated
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_sepsis]:
            logger.debug('pkg_code_sepsis is available, so use it.')
            if params['prob_cure_antibiotics'] > self.module.rng.random_sample():
                mni[person_id]['sepsis'] = False
        else:
            logger.debug('pkg_code_sepsis is not available, so can' 't use it.')

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        #    actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_ReceivesCareForMaternalSepsis: did not run')
        pass


class HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy(HSI_Event, IndividualScopeEventMixin):
    """This is the HSI ReceivesCareForHypertensiveDisordersOfPregnancy. This event is scheduled by the either HSI
    PresentsForSkilledAttendanceInLabour or HSI ReceivesCareForPostpartumPeriod if a woman develops or has presented in
    labour with any of the hypertensive disorders of pregnancy. This event delivers interventions including
    anti-convulsants and anti-hypertensives. Currently this event only manages interventions for Eclampsia and therefore
    is incomplete and subject to change"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForHypertensiveDisorder'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1  # TODO: to be discussed with TH/TC best appt footprint to use

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props  # shortcut to the dataframe
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        logger.info('This is HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy, management of hypertensive '
                    'disorders of pregnancy for person %d on date %s',
                    person_id, self.sim.date)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_eclampsia = pd.unique(consumables.loc[consumables[
                                                               'Intervention_Pkg'] ==
                                                       'Management of eclampsia',
                                                       'Intervention_Pkg_Code'])[0]

        item_code_nf = pd.unique(
            consumables.loc[consumables['Items'] == 'nifedipine retard 20 mg_100_IDA', 'Item_Code'])[0]
        item_code_hz = pd.unique(
            consumables.loc[consumables['Items'] == 'Hydralazine hydrochloride 20mg/ml, 1ml_each_CMST', 'Item_Code'])[0]

        # TODO: Methyldopa not being recognised?
        # item_code_md = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Methyldopa 250mg_1000_CMS','Item_Code'])[0]

        item_code_hs = pd.unique(
            consumables.loc[consumables['Items'] == "Ringer's lactate (Hartmann's solution), 500 ml_20_IDA",
                            'Item_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code_eclampsia: 1}],
            'Item_Code': [{item_code_nf: 2}, {item_code_hz:  2}, {item_code_hs: 1}]}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed )

        # TODO: Again determine how to reflect 1st/2nd line choice
#        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_obst_lab]:
#            logger.debug('pkg_code_obst_lab is available, so use it.')
#        else:
#            logger.debug('pkg_code_obst_lab is not available, so can' 't use it.')


# =======================================  HYPERTENSION TREATMENT ======================================================
        # tbc

# =======================================  SEVERE PRE-ECLAMPSIA TREATMENT ==============================================
        # tbc

# =======================================  ECLAMPSIA TREATMENT ========================================================

        # Here we apply the treatment algorithm, unsuccessful control of seizures leads to caesarean delivery
        if mni[person_id]['eclampsia'] or mni[person_id]['eclampsia_pp']:
            if params['prob_cure_mgso4'] > self.module.rng.random_sample():
                mni[person_id]['eclampsia'] = False
            else:
                if params['prob_cure_mgso4'] > self.module.rng.random_sample():
                    mni[person_id]['eclampsia'] = False
                else:
                    if params['prob_cure_diazepam'] > self.module.rng.random_sample():
                        mni[person_id]['eclampsia'] = False
                    else:
                        event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                            priority=0,
                                                                            topen=self.sim.date,
                                                                            tclose=self.sim.date + DateOffset(days=1))

                        logger.info(
                            'This is HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy referring person %d '
                            'for a caesarean section following uncontrolled eclampsia', person_id)

        # Guidelines suggest assisting with second stage of labour via AVD - consider including this?
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        #    actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_ReceivesCareForMaternalSepsis: did not run')
        pass


class HSI_Labour_ReceivesCareForMaternalHaemorrhage(HSI_Event, IndividualScopeEventMixin):
    """This is the HSI ReceivesCareForMaternalHaemorrhage. This event is scheduled by the either HSI
    PresentsForSkilledAttendanceInLabour or HSI ReceivesCareForPostpartumPeriod if a woman experiences an antepartum or
    postpartum haemorrhage.This event delivers a cascade of interventions to arrest bleeding. In the event of treatment
    failure this event schedules HSI referredForSurgicalCareInLabour. This event is not finalised an both interventions
    and referral are subject to change """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForMaternalHaemorrhage'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        # TODO: squeeze factor AND consider having APH, PPH and RPP as separate treatment events??

        df = self.sim.population.props
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        logger.info('This is HSI_Labour_ReceivesCareForMaternalHaemorrhage, management of obstructed labour for person '
                    '%d on date %s', person_id, self.sim.date)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_pph = pd.unique(consumables.loc[consumables[
                                                         'Intervention_Pkg'] ==
                                                 'Treatment of postpartum hemorrhage',
                                                 'Intervention_Pkg_Code'])[0]

        item_code_aph1 = pd.unique(consumables.loc[consumables['Items'] == 'Blood, one unit', 'Item_Code'])[0]
        item_code_aph2 = pd.unique(consumables.loc[consumables['Items'] == 'Lancet, blood, disposable', 'Item_Code'])[0]
        item_code_aph3 = pd.unique(consumables.loc[consumables['Items'] == 'Test, hemoglobin', 'Item_Code'])[0]
        item_code_aph4 = pd.unique(consumables.loc[consumables['Items'] == 'IV giving/infusion set, with needle',
                                                   'Item_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code_pph: 1}],
            'Item_Code': [{item_code_aph1: 2}, {item_code_aph2: 1}, {item_code_aph3: 1}, {item_code_aph4: 1}]}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)

        # TODO: Again determine how to use outcome of consumable request

# ===================================  ANTEPARTUM HAEMORRHAGE TREATMENT ===============================================
        # TODO: consider severity grading?

        # Here we determine the etiology of the bleed, which will determine treatment algorithm
        etiology = ['PA', 'PP']  # May need to move this to allow for risk factors?
        probabilities = [0.67, 0.33]  # DUMMY
        mni[person_id]['source_aph'] = self.module.rng.choice(etiology, size=1, p=probabilities)

        # Storing as high chance of SB in severe placental abruption
        # TODO: Needs to be dependent on blood availability and establish how we're quantifying effect
        mni[person_id]['units_transfused'] = 2

        event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
        self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                            priority=0,
                                                            topen=self.sim.date,
                                                            tclose=self.sim.date + DateOffset(days=1))

# ======================================= POSTPARTUM HAEMORRHAGE ======================================================
        # First we use a probability weighted random draw to determine the underlying etiology of this womans PPH

        if mni[person_id]['postpartum_haem']:
            etiology = ['UA', 'RPP']
            probabilities = [0.67, 0.33]  # dummy
            mni[person_id]['source_pph'] = self.module.rng.choice(etiology, size=1, p=probabilities)

            # Todo: add a level of severity of PPH -yes

            # ======================== TREATMENT CASCADE FOR ATONIC UTERUS:=============================================

            # Here we use a treatment cascade adapted from Malawian Obs/Gynae guidelines
            # Women who are bleeding due to atonic uterus first undergo medical management, oxytocin IV, misoprostol PR
            # and uterine massage in an attempt to stop bleeding

            if mni[person_id]['source_pph'] == 'UA':
                if params['prob_cure_oxytocin'] > self.module.rng.random_sample():
                    mni[person_id]['postpartum_haem'] = False
                else:
                    if params['prob_cure_misoprostol'] > self.module.rng.random_sample():
                        mni[person_id]['postpartum_haem'] = False
                    else:
                        if params['prob_cure_uterine_massage'] > self.module.rng.random_sample():
                            mni[person_id]['postpartum_haem'] = False
                        else:
                            if params['prob_cure_uterine_tamponade'] > self.module.rng.random_sample():
                                mni[person_id]['postpartum_haem'] = False

            # Todo: consider the impact of oxy + miso + massage as ONE value, Discuss with expert

            # ===================TREATMENT CASCADE FOR RETAINED PRODUCTS/PLACENTA:====================================
            if mni[person_id]['source_pph'] == 'RPP':
                if params['prob_cure_manual_removal'] > self.module.rng.random_sample():
                    mni[person_id]['postpartum_haem'] = False
                    # blood?

            # In the instance of uncontrolled bleeding a woman is referred on for surgical care
            if mni[person_id]['postpartum_haem']:
                event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        #    actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_ReceivesCareForMaternalHaemorrhage: did not run')
        pass

class HSI_Labour_ReceivesCareForPostpartumPeriod(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This event manages the Health System Interaction for women who receive post partum care following delivery
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

    # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForPostpartumPeriod'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        # TODO: Squeeze factor
        logger.info('This is HSI_Labour_ReceivesCareForPostpartumPeriod: Providing skilled attendance following birth '
                    'for person %d', person_id)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_am = pd.unique(consumables.loc[consumables[
                                                            'Intervention_Pkg'] ==
                                                'Active management of the 3rd stage of labour',
                                                'Intervention_Pkg_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code_am: 1}],'Item_Code': []}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

        df = self.sim.population.props  # shortcut to the dataframe
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

    #  =========================  ACTIVE MANAGEMENT OF THE THIRD STAGE  ===============================================

        # Here we apply a risk reduction of post partum bleeding following active management of the third stage of
        # labour (additional oxytocin, uterine massage and controlled cord traction)
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_am]:
            logger.debug('pkg_code_am is available, so use it.')
            mni[person_id]['risk_pp_postpartum_haem'] = mni[person_id]['risk_pp_postpartum_haem'] * \
                                                    params['rr_pph_amtsl']
        else:
            logger.debug('pkg_code_am is not available, so can' 't use it.')
            logger.debug('woman %d did not receive active management of the third stage of labour due to resource '
                         'constraints')
    # ===============================  POSTPARTUM COMPLICATIONS ========================================================

        # TODO: link eclampsia/sepsis diagnosis in SBA and PPC

        # As with the SkilledBirthAttendance HSI we recalculate risk of complications in light of preventative
        # interventions

        self.module.set_complications_during_facility_birth(person_id, complication='eclampsia', labour_stage='pp',
                                              treatment_hsi=HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy
                                              (self.module, person_id=person_id))

        self.module.set_complications_during_facility_birth(person_id, complication='postpartum_haem',labour_stage= 'pp',
                                              treatment_hsi=HSI_Labour_ReceivesCareForMaternalHaemorrhage(self.module,
                                                                                            person_id=person_id))

        self.module.set_complications_during_facility_birth(person_id,complication='sepsis', labour_stage='pp',
                                              treatment_hsi=HSI_Labour_ReceivesCareForMaternalSepsis(self.module,
                                                                                                person_id=person_id))

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # TODO: modify based on complications?
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_ReceivesCareForPostpartumPeriod: did not run')
        pass


class HSI_Labour_ReferredForSurgicalCareInLabour(HSI_Event, IndividualScopeEventMixin):
    """This is the HSI ReferredForSurgicalCareInLabour.Currently this event is scheduled HSIs:
    PresentsForSkilledAttendanceInLabour and ReceivesCareForMaternalHaemorrhage, but this is not finalised.
    This event delivers surgical management for haemorrhage, uterine rupture and performs caesarean section. This event
    is not finalised and may need to be restructured to better simulate referral """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReferredForSurgicalCareInLabour'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()

    #   the_appt_footprint['MajorSurg'] = 1  # this appt could be used for uterine repair/pph management
        the_appt_footprint['Csection'] = 1

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        #  TODO: squeeze factor and consider splitting HSIs for different surgeries

        df = self.sim.population.props
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        logger.info('This is HSI_Labour_ReferredForSurgicalCareInLabour,providing surgical care during labour and the'
                    ' postpartum period for person %d on date %s', person_id, self.sim.date)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_cs = pd.unique(consumables.loc[consumables[
                                                    'Intervention_Pkg'] ==
                                                'Cesearian Section with indication (with complication)',  # or without?
                                                'Intervention_Pkg_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code_cs: 1}],
            'Item_Code': []}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)
        # TODO: consumables

        # pkg_code_uterine_repair
        # pkg_code_pph_surgery
        # +/- hysterectomy?

# ====================================== EMERGENCY CAESAREAN SECTION ==================================================

        if (mni[person_id]['uterine_rupture']) or (mni[person_id]['antepartum_haem']) or \
           (mni[person_id]['eclampsia']) or (mni[person_id]['labour_is_currently_obstructed']) or \
           (df.at[person_id, 'la_current_labour_successful_induction'] == 'failed_induction'):
            # Consider all indications (elective)
            mni[person_id]['antepartum_haem'] = False
            # reset eclampsia status?
            mni[person_id]['labour_is_currently_obstructed'] = False
            mni[person_id]['mode_of_delivery'] = 'CS'
            df.at[person_id, 'la_previous_cs_delivery'] = True
            # apply risk of death from CS?

# ====================================== UTERINE REPAIR ==============================================================

        # For women with UR we determine if the uterus can be repaired surgically
        if mni[person_id]['uterine_rupture']:
            if params['prob_cure_uterine_repair'] > self.module.rng.random_sample():
                # df.at[person_id, 'la_uterine_rupture'] = False
                mni[person_id]['uterine_rupture'] = False

        # In the instance of failed surgical repair, the woman undergoes a hysterectomy
            else:
                if params['prob_cure_hysterectomy'] > self.module.rng.random_sample():
                    # df.at[person_id, 'la_uterine_rupture'] = False
                    mni[person_id]['uterine_rupture'] = False

# ================================== SURGERY FOR UNCONTROLLED POSTPARTUM HAEMORRHAGE ==================================

        # If a woman has be referred for surgery for uncontrolled post partum bleeding we use the treatment alogrith to
        # determine if her bleeding can be controlled surgically
        if mni[person_id]['postpartum_haem']:
            if params['prob_cure_uterine_ligation'] > self.module.rng.random_sample():
                mni[person_id]['postpartum_haem'] = False
            else:
                if params['prob_cure_b_lynch'] > self.module.rng.random_sample():
                    mni[person_id]['postpartum_haem'] = False
                else:
                    # Todo: similarly consider bunching surgical interventions
                    if params['prob_cure_hysterectomy'] > self.module.rng.random_sample():
                        mni[person_id]['postpartum_haem'] = False

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        #  TODO: modify based on complications?
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_ReferredForSurgicalCareInLabour: did not run')
        pass


class LabourLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Handles Labour logging"""
    def __init__(self, module):
        """schedule logging to repeat every 3 months
        """
    #    self.repeat = 12
    #    super().__init__(module, frequency=DateOffset(days=self.repeat))
        super().__init__(module, frequency=DateOffset(months=12))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props

        #  MATERNAL MORTALITY RATIO:

        # live birth total
        # one_year_prior = self.sim.date - np.timedelta64(1, 'Y')
        # live_births = df.index[(df.date_of_birth > one_year_prior) & (df.date_of_birth < self.sim.date)]
        # live_births_sum = len(live_births)
        # print(live_births_sum)

        # deaths = df.index[(df.la_maternal_death == True) & (df.la_maternal_death_date > one_year_prior) &
        #                  (df.la_maternal_death_date < self.sim.date)]

        # cumm_deaths = len(deaths)
        # print(cumm_deaths)

        # mmr = cumm_deaths/live_births_sum * 100000
        # print('The maternal mortality ratio for this year is', mmr)

        # Still Birth Rate
        # Perinatal Mortality
        # Disease Incidence
        # Intervention incidence
