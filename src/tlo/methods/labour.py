from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import demography
from tlo.methods.healthsystem import HSI_Event

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
            Types.REAL, 'relative risk of uterine rupture in women who have previously delivered via caesarean '
                        'section'),
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
        'rr_sepsis_post_abx_pprom': Parameter(
            Types.REAL, 'relative risk of maternal sepsis following prophylactic antibiotics for PPROM in a facility'),
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
        'dummy_prob_health_centre': Parameter(
            Types.REAL, 'dummy probability to determine the facility type of the Level 1 SBA HSI'),
        'squeeze_factor_threshold_delivery_attendance': Parameter(
            Types.REAL, 'Squeeze factor threshold at which there is not capacity for deliveries to be attended'),
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
        'la_parity': Property(Types.REAL, 'total number of previous deliveries'),
        # TODO: This should be an integer but could force inf value to int from LM output
        'la_previous_cs_delivery': Property(Types.BOOL, 'whether this woman has ever delivered via caesarean section'),
        'la_has_previously_delivered_preterm': Property(Types.BOOL, 'whether the woman has had a previous preterm '
                                                                    'delivery for any of her previous deliveries'),
        'la_obstructed_labour': Property(Types.BOOL, 'whether this womans labour has become obstructed'),
        'la_obstructed_labour_disab': Property(Types.BOOL, 'disability associated with obstructed labour'),
        'la_antepartum_haem': Property(Types.BOOL, 'whether the woman has experienced an antepartum haemorrhage in this'
                                                   'delivery'),
        'la_antepartum_haem_disab': Property(Types.BOOL, 'disability associated with antepartum haemorrhage'),
        'la_uterine_rupture': Property(Types.BOOL, 'whether the woman has experienced uterine rupture in this '
                                                   'delivery'),
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
        # as a parameter

        params['la_labour_equations'] =\
            {'parity': LinearModel(
                LinearModelType.ADDITIVE,
                -3,
                Predictor('age_years').apply(lambda age_years: (age_years * 0.22)),
                Predictor('li_mar_stat').when('2', 0.91).when('3', 0.16),
                Predictor('li_wealth').when('5', -0.13).when('4', -0.13).when('3', -0.26).when('2', -0.37).when('1',
                                                                                                                -0.9)),
                # TODO: first draft from rough regression of 2010 DHS data, rounded in code to ensure whole numbers

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
                LinearModelType.MULTIPLICATIVE,  # TODO: separate causal influence in praevia and abruption
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
                params['odds_homebirth'],  # TODO: need to make these parameters
                Predictor('li_mar_stat').when('1', params['or_homebirth_unmarried']).when('3',
                                                                                          params['or_homebirth'
                                                                                                 '_unmarried']),
                Predictor('li_wealth').when('4', params['or_homebirth_wealth_4']).when('5',
                                                                                       params['or_homebirth_wealth_5']),
                 # wealth levels in the paper are different
                Predictor('li_urban').when(True, params['or_homebirth_urban']))
             }

        # TODO: do we need an equation for post term labour?

    def initialise_population(self, population):
        df = population.props
        params = self.parameters

        df.loc[df.is_alive, 'la_current_labour_successful_induction'] = 'not_induced'
        df.loc[df.is_alive, 'la_currently_in_labour'] = False
        df.loc[df.is_alive, 'la_intrapartum_still_birth'] = False
        df.loc[df.is_alive, 'la_parity'] = 0
        df.loc[df.is_alive, 'la_previous_cs_delivery'] = False
        df.loc[df.is_alive, 'la_has_previously_delivered_preterm'] = False
        df.loc[df.is_alive, 'la_due_date_current_pregnancy'] = pd.NaT
        df.loc[df.is_alive, 'la_obstructed_labour'] = False
        df.loc[df.is_alive, 'la_obstructed_labour_disab'] = False
        df.loc[df.is_alive, 'la_antepartum_haem'] = False
        df.loc[df.is_alive, 'la_antepartum_haem_disab'] = False
        df.loc[df.is_alive, 'la_uterine_rupture'] = False
        df.loc[df.is_alive, 'la_uterine_rupture_disab'] = False
        df.loc[df.is_alive, 'la_eclampsia'] = False
        df.loc[df.is_alive, 'la_eclampsia_disab'] = False
        df.loc[df.is_alive, 'la_postpartum_haem'] = False
        df.loc[df.is_alive, 'la_postpartum_haem_disab'] = False
        df.loc[df.is_alive, 'la_maternal_death'] = False
        df.loc[df.is_alive, 'la_maternal_death_date'] = pd.NaT

#  ----------------------------ASSIGNING PARITY AT BASELINE ----------------------------------------------------------

        # TODO: This linear equation is from a very rough regression of DHS 2010 parity (not all predictors fully
        #  explored)

        df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14), 'la_parity'] = \
            np.around(params['la_labour_equations']['parity'].predict(df.loc[df.is_alive & (df.sex == 'F') &
                                                                             (df.age_years > 14)]))
        df.la_parity.astype(float)

    def initialise_simulation(self, sim):

        event = LabourLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def on_birth(self, mother_id, child_id):

        df = self.sim.population.props
        mother = df.loc[mother_id]

        df.at[child_id, 'la_due_date_current_pregnancy'] = pd.NaT
        df.at[child_id, 'la_currently_in_labour'] = False
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
        if ~mother.la_intrapartum_still_birth:
            df.at[mother_id, 'la_parity'] += 1  # Only live births contribute to parity
            logger.info('%s|live_births|%s',
                        self.sim.date,
                        {'mother': mother_id,
                         'child': child_id,
                         'mother_age': df.at[mother_id, 'age_years']})

        if mother.la_intrapartum_still_birth:
            #  N.B this will only record intrapartum stillbirth
            death = demography.InstantaneousDeath(self.sim.modules['Demography'], child_id,
                                                  cause='ip stillbirth')
            self.sim.schedule_event(death, self.sim.date)

            # This property is then reset in case of future pregnancies/stillbirths
            df.at[mother_id, 'la_intrapartum_still_birth'] = False

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.info('This is Labour, being alerted about a health system interaction '
                    'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):

        # TODO: Refine disability levels (could include severity) and explore a more elegant solution involving less
        #  properties (& Make sure that value >1.0 is being reported as current fix is temp)

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
        """This function, called within contraception, uses linear equations to determine a womans likelihood of
        preterm, postterm or term labour and sets their future date of labour accordingly"""

        df = self.sim.population.props
        params = self.parameters
        logger.debug('person %d is having their labour scheduled on date %s', individual_id, self.sim.date)

        # Check only alive newly pregnant women are scheduled to this function
        assert df.at[individual_id, 'is_alive'] and df.at[individual_id, 'is_pregnant']
        assert df.at[individual_id, 'date_of_last_pregnancy'] == self.sim.date

        # Using the linear equations defined above we calculate this womans individual risk of early and late preterm
        # labour
        eptb_prob = params['la_labour_equations']['early_preterm_birth'].predict(df.loc[[individual_id]])[individual_id]
        lptb_prob = params['la_labour_equations']['late_preterm_birth'].predict(df.loc[[individual_id]])[individual_id]

        # We then use a random draw to determine if the woman will go into preterm labour and how early she will deliver
        # We store this draw as a variable so the result can be compared against both probabilities
        random_draw = self.rng.random_sample()
        if random_draw < eptb_prob:
            df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] + \
                                                                        pd.Timedelta((self.rng.randint(24, 33)),
                                                                                     unit='W')

        elif random_draw < (lptb_prob + eptb_prob):
            df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] + \
                                                                        pd.Timedelta((self.rng.randint(34, 36)),
                                                                                     unit='W')

        # For women who will deliver after term we apply a risk of post term birth
        else:
            if self.rng.random_sample() < params['prob_potl']:
                df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] \
                                                                            + pd.Timedelta((self.rng.randint(42, 46)),
                                                                                           unit='W')
            else:
                df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] \
                                                                            + pd.Timedelta((self.rng.randint(37, 41)),
                                                                                           unit='W')

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
        return self.rng.random_sample() < eq.predict(self.sim.population.props.loc[[person_id]])[person_id]

    def set_home_birth_complications(self, individual_id, labour_stage, complication):
        """Uses the result of a linear equation to determine the probability of a certain complication, stores the
        probability for women delivery in facility and applies the probability for women delivering at home.
        Sets complication properties according to the result"""
        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        params = self.parameters

        if mni[individual_id]['delivery_setting'] == 'facility_delivery':
            mni[individual_id][f'risk_{labour_stage}_{complication}'] = \
                params['la_labour_equations'][f'{complication}_{labour_stage}'].predict(df.loc[[individual_id]]
                                                                                        )[individual_id]
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

        # Ensure that women delivering in facilities do not have their complication status set
        assert not mni[individual_id]['delivery_setting'] == 'facility_delivery' and df.at[individual_id,
                                                                                           f'la_{complication}'] or \
            df.at[individual_id, f'la_{complication}_disab']

    def set_complications_during_facility_birth(self, person_id, complication, labour_stage):
        """Using each womans individual risk of a complication (which may have been modified by treatment) this function
        determines if she will experience a complication during her facility delivery. If so, additional treatment is
        scheduled"""

        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        rng = self.rng

        # Ensure women delivering at home arent having complications set this way
        assert not mni[person_id]['delivery_setting'] == 'home_birth'

        if rng.random_sample() < mni[person_id][f'risk_{labour_stage}_{complication}']:
            df.at[person_id, f'la_{complication}'] = True
            df.at[person_id, f'la_{complication}_disab'] = True
            mni[person_id][f'{complication}'] = True
            logger.debug(f'person %d is experiencing {complication} in a health facility', person_id)

            logger.info(f'%s|{complication}|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})

            logger.debug(f'This is HSI_Labour_PresentsForSkilledAttendanceInLabourFacilityLevel1: person %d has '
                         f'developed {complication} during delivery at facility level 1', person_id)


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

        # Here we populate the maternal and newborn info dictionary with baseline values before the womans labour begins
        mni = self.module.mother_and_newborn_info

        logger.debug('person %d has just reached LabourOnsetEvent on %s', individual_id, self.sim.date)

        # Here we check only women who have reached their due date are going into labour
        assert df.at[individual_id, 'la_due_date_current_pregnancy'] == self.sim.date

        # TODO: review in context of properties- ensure what should be a property IS one, and what SHOULDN'T be isnt.
        mni[individual_id] = {'labour_state': None,  # Term Labour (TL), Early Preterm (EPTL), Late Preterm (LPTL) or
                              # Post Term (POTL)
                              'delivery_setting': None,  # Facility Delivery (FD) or Home Birth (HB)
                              'delivery_facility_type': None,  # health_centre, hospital, regional_hospital
                              'delivery_attended': None,  # unattended or attended
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
                              #  Should this just be risk of asphyxia
                              'mode_of_delivery': None,  # Vaginal Delivery (VD),Vaginal Delivery Induced (VDI),
                              # Assisted Vaginal Delivery Forceps (AVDF) Assisted Vaginal Delivery Ventouse (AVDV)
                              # Caesarean Section (CS)
                              'death_in_labour': False,  # True (T) or False (F)
                              'cause_of_death_in_labour': [],  # Appended list of cause/causes of death
                              # TODO: should we do the same thing for still birth? Would help with mapping
                              'stillbirth_in_labour': False,  # True (T) or False (F)
                              'death_postpartum': False}  # True (T) or False (F)

# ===================================== LABOUR STATE  ==================================================================

        # Debug logging to highlight the women who have miscarried/had an abortion/stillbirth who still come to the
        # event
        person = df.loc[individual_id]
        if ~person.is_pregnant:
            logger.debug('person %d has just reached LabourOnsetEvent on %s but is no longer pregnant',
                         individual_id, self.sim.date)
            del mni[individual_id]

        # The event is conditioned on the woman being pregnant, today being her due date and being alive
        # We don't use assert functions to ensure the due date is correct as women who lose their pregnancy will still
        # have this event scheduled
        if person.is_pregnant & person.is_alive & (person.la_due_date_current_pregnancy == self.sim.date):
            df.at[individual_id, 'la_currently_in_labour'] = True

            # If a woman has been induced/attempted induction she will already be in a facility therefore delivery_
            # setting is set to facility
            if (person.la_current_labour_successful_induction == 'failed_induction') or \
               (person.la_current_labour_successful_induction == 'successful_induction'):
                mni[individual_id]['delivery_setting'] = 'facility_delivery'
            # We then store her induction status if successful
            if person.la_current_labour_successful_induction == 'successful_induction':
                mni[individual_id]['induced_labour'] = True

            # Now we use gestational age to categorise the 'labour_state'
            if person.is_pregnant & person.is_alive:
                if 37 <= person.ps_gestational_age_in_weeks < 42:
                    mni[individual_id]['labour_state'] = 'term_labour'

                elif 24 <= person.ps_gestational_age_in_weeks < 34:
                    mni[individual_id]['labour_state'] = 'early_preterm_labour'
                    df.at[individual_id, 'la_has_previously_delivered_preterm'] = True

                elif 37 > person.ps_gestational_age_in_weeks >= 34:
                    mni[individual_id]['labour_state'] = 'late_preterm_labour'
                    df.at[individual_id, 'la_has_previously_delivered_preterm'] = True

                elif person.ps_gestational_age_in_weeks > 41:
                    mni[individual_id]['labour_state'] = 'postterm_labour'
                    logger.info('%s|postterm_birth|%s', self.sim.date,
                                {'age': df.at[individual_id, 'age_years'],
                                 'person_id': individual_id})

                assert not mni[individual_id]['labour_state'] == 'postterm_labour' and \
                    person.ps_gestational_age_in_weeks < 42
                assert not mni[individual_id]['labour_state'] == 'term_labour' and \
                    (36 > person.ps_gestational_age_in_weeks > 41)
                assert not mni[individual_id]['labour_state'] == 'early_preterm_labour' and \
                    (24 > person.ps_gestational_age_in_weeks > 33)
                assert not mni[individual_id]['labour_state'] == 'late_preterm_labour' and \
                    (34 > person.ps_gestational_age_in_weeks > 36)
                
                labour_state = mni[individual_id]['labour_state']
                logger.debug(f'This is LabourOnsetEvent, person %d has now gone into {labour_state} on date %s',
                             individual_id, self.sim.date)

# ======================================= PROBABILITY OF HOME DELIVERY =================================================

                self.sim.schedule_event(LabourAtHomeEvent(self.module, individual_id), self.sim.date)

                # TODO: choice of facility level
                facility_delivery = HSI_Labour_PresentsForSkilledAttendanceInLabourFacilityLevel1(self.module,
                                                                                                  person_id=
                                                                                                  individual_id)

                # As women who have presented for induction are already in a facility we exclude them, then apply a
                # probability of home birth
                if self.module.eval(params['la_labour_equations']['care_seeking'], individual_id) & \
                   (person.la_current_labour_successful_induction == 'not_induced'):
                    mni[individual_id]['delivery_setting'] = 'home_birth'
                    self.sim.schedule_event(LabourAtHomeEvent(self.module, individual_id), self.sim.date)
                    logger.debug('This is LabourEvent,  person %d will not seek care in labour and will deliver at '
                                 'home', individual_id)
                    logger.info('%s|home_birth|%s', self.sim.date,
                                {'person_id': individual_id})
                else:
                    mni[individual_id]['delivery_setting'] = 'facility_delivery'
                    logger.info(
                        'This is LabourOnsetEvent, scheduling HSI_Labour_PresentsForSkilledAttendanceInLabour on date'
                        ' %s for person %d as they have chosen to seek care for delivery', self.sim.date, individual_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(facility_delivery,
                                                                        priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=1))
                    logger.info('%s|facility_delivery|%s', self.sim.date,
                                {'person_id': individual_id})

                # Here we schedule delivery care for women who have already sought care for induction, whether or not
                # that induction was successful
                if (person.la_current_labour_successful_induction == 'failed_induction') or \
                   (person.la_current_labour_successful_induction == 'successful_induction'):
                    self.sim.modules['HealthSystem'].schedule_hsi_event(facility_delivery,
                                                                        priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=1))
                    logger.info('%s|facility_delivery|%s', self.sim.date,
                                {'person_id': individual_id})

# ======================================== SCHEDULING BIRTH AND DEATH EVENTS ==========================================

                # Here we schedule the birth event for 2 days after labour- we do this prior to the death event as women
                # who die but still deliver a live child will pass through birth event
                due_date = df.at[individual_id, 'la_due_date_current_pregnancy']
                self.sim.schedule_event(BirthEvent(self.module, individual_id), due_date + DateOffset(days=4))
                logger.debug('This is LabourOnsetEvent scheduling a birth on date %s to mother %d', due_date,
                             individual_id)

                # We schedule all women to move through the death event where those who have developed a complication
                # that hasn't been treated or treatment has failed will have a case fatality rate applied
                self.sim.schedule_event(LabourDeathEvent(self.module, individual_id), self.sim.date +
                                        DateOffset(days=3))

                logger.debug('This is LabourOnsetEvent scheduling a potential death on date %s for mother %d',
                             self.sim.date, individual_id)

                # Here we set the due date of women who have been induced to pd.NaT so they dont go into labour twice
                if person.la_current_labour_successful_induction != 'not_induced':
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
        if (mni[individual_id]['labour_state'] == 'early_preterm_labour') or (mni[individual_id]['labour_state'] == 'late_preterm_labour'):
            if self.module.rng.random_sample() < params['prob_prom']:
                mni[individual_id]['PROM'] = True
        # TODO: term labour can have prom also (also this should be an antenatal thing?)

    # ===================================  APPLICATION OF COMPLICATIONS ===============================================

        # Using the complication_application function we move through each complication in turn determining if a woman
        # will experience any of these if she has delivered at home, and storing the risk so it can be modified by
        # skilled birth attendance for women delivering in a facility (n.b 'ip' == intrapartum)

        self.module.set_home_birth_complications(individual_id, labour_stage='ip', complication='obstructed_labour')
        if df.at[individual_id, 'la_obstructed_labour']:  # there's a neater way of doing this
            mni[individual_id]['labour_is_currently_obstructed'] = True
            mni[individual_id]['labour_has_previously_been_obstructed'] = True
        # TODO: As OL is a contraindication of induction, should we skip it here for induced women?
        #  late diagnosis of obstruction is possible surely

        # Here labour_stage 'ip' means intrapartum
        self.module.set_home_birth_complications(individual_id, labour_stage='ip', complication='antepartum_haem')
        self.module.set_home_birth_complications(individual_id, labour_stage='ip', complication='sepsis')
        self.module.set_home_birth_complications(individual_id, labour_stage='ip', complication='eclampsia')
        self.module.set_home_birth_complications(individual_id, labour_stage='ip', complication='uterine_rupture')

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
        df = self.sim.population.props
        person = df.loc[mother_id]

        df.at[mother_id, 'la_currently_in_labour'] = False

        # If the mother is alive and still pregnant we generate a  child and the woman is scheduled to move to the
        # postpartum event to determine if she experiences any additional complications (intrapartum stillbirths till
        # trigger births for monitoring purposes_)
        if person.is_alive and person.is_pregnant:
            logger.info('@@@@ A Birth is now occuring, to mother %d', mother_id)
            self.sim.do_birth(mother_id)
            logger.debug('This is BirthEvent scheduling mother %d to undergo the PostPartumEvent following birth',
                         mother_id)
            self.sim.schedule_event(PostpartumLabourEvent(self.module, mother_id),
                                    self.sim.date)

        # If the mother has died during childbirth the child is still generated with is_alive=false to monitor
        # stillbirth rates. She will not pass through the postpartum complication events
        if ~person.is_alive and ~person.la_intrapartum_still_birth and person.la_maternal_death:
            logger.debug('@@@@ A Birth is now occuring, to mother %d who died in childbirth but her child survived',
                         mother_id)
            self.sim.do_birth(mother_id)


class PostpartumLabourEvent(Event, IndividualScopeEventMixin):
    """This is PostpartumLabour event. It is scheduled by BirthEvent immediately following birth. This event applies the
    probability that women delivering at home will experience complications, makes the appropriate changes to the data
    frame. This event schedules the PostPartumDeathEvent (4 days post this event)and the DisabilityResetEvent"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        mni = self.module.mother_and_newborn_info

        # Here we use the complication_application function to determine if a women who has survived labour will
        # experience any further complications (risk is stored for facility deliveries)
        if df.at[individual_id, 'is_alive']:
            # Here labour_stage 'pp' means postpartum
            self.module.set_home_birth_complications(individual_id, labour_stage='pp', complication='postpartum_haem')
            self.module.set_home_birth_complications(individual_id, labour_stage='pp', complication='sepsis')
            self.module.set_home_birth_complications(individual_id, labour_stage='pp', complication='eclampsia')

            # If a woman has delivered in a facility we schedule her to now receive additional care following birth
            if mni[individual_id]['delivery_setting'] == 'facility_delivery':
                logger.info('This is PostPartumEvent scheduling HSI_Labour_ReceivesCareForPostpartumPeriod for person '
                            '%d on date %s', individual_id, self.sim.date)
                event = HSI_Labour_ReceivesCareForPostpartumPeriodFacilityLevel1(self.module, person_id=individual_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))
                # TODO: same issue for women who seek care but cant be seek- comps wont be allocated!
            # We schedule all women to then go through the death event where those with untreated/unsuccessfully treated
            # complications may experience death

            self.sim.schedule_event(PostPartumDeathEvent(self.module, individual_id), self.sim.date
                                    + DateOffset(days=4))
            logger.info('This is PostPartumEvent scheduling a potential death for person %d on date %s', individual_id,
                        self.sim.date + DateOffset(days=4))  # Date offset to allow for interventions

            # Here we schedule women to an event which resets 'daly' disability associated with delivery complications
            self.sim.schedule_event(DisabilityResetEvent(self.module, individual_id),
                                    self.sim.date + DateOffset(months=1))

            # TODO: symptom manager for women who have complications at home


class LabourDeathEvent (Event, IndividualScopeEventMixin):
    """This is the LabourDeathEvent. It is scheduled by the LabourOnsetEvent for all women who go through labour. This
    event determines if women who have experienced complications in labour will die or experience an intrapartum
    stillbirth."""

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
            if rng.random_sample() < params[f'cfr_{cause}']:
                logger.debug(f'{cause} has contributed to person %d death during labour', individual_id)
                mni[individual_id]['death_in_labour'] = True
                mni[individual_id]['cause_of_death_in_labour'].append(cause)
                df.at[individual_id, 'la_maternal_death'] = True
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date

                # then if she does die, we determine if the child will still survive
                if rng.random_sample() < params[f'prob_still_birth_{cause}_md']:
                    logger.debug(f'Following person %d death due to{cause} they have experienced a still'
                                 f'birth', individual_id)
                    df.at[individual_id, 'la_intrapartum_still_birth'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True
                    df.at[individual_id, 'ps_previous_stillbirth'] = True

                # Otherwise we just determine if this complication will lead to a stillbirth
            else:
                if rng.random_sample() < params[f'prob_still_birth_{cause}']:
                    logger.debug(f'person %d has experienced a still birth following {cause} in labour')
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

            if mni[individual_id]['death_in_labour'] and df.at[individual_id, 'la_intrapartum_still_birth']:
                # We delete the mni dictionary if both mother and baby have died in labour, if the mother has died but
                # the baby has survived we delete the dictionary following the on_birth function of NewbornOutcomes
                del mni[individual_id]

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
        mni = self.module.mother_and_newborn_info

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
            del mni[individual_id]

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
    """This is the DiseaseResetEvent. It is scheduled by the PostPartumDeathEvent. This event resets a woman's
    disease properties within the data frame """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        mni = self.module.mother_and_newborn_info

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

            del mni[individual_id]

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

        # Initial request for consumables needed
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_induction = pd.unique(consumables.loc[consumables[
                                                       'Intervention_Pkg'] ==
                                                       'Induction of labour (beyond 41 weeks)',
                                                       'Intervention_Pkg_Code'])[0]
        # TODO: review induction guidelines to confirm appropriate 1st line/2nd line consumable use

        consumables_needed = {'Intervention_Package_Code': {pkg_code_induction: 1},
                              'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed)

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_induction]:
            logger.debug('pkg_code_induction is available, so use it.')
            # TODO: reschedule if consumables aren't available at this point in time?
        else:
            logger.debug('PkgCode1 is not available, so can' 't use it.')

        # We use a random draw to determine if this womans labour will be successfully induced
        # Indications: Post term, eclampsia, severe pre-eclampsia, mild pre-eclampsia at term, PROM > 24 hrs at term
        # or PPROM > 34 weeks EGA, and IUFD.

        if self.module.rng.random_sample() < params['prob_successful_induction']:
            logger.info('Person %d has had her labour successfully induced', person_id)
            df.at[person_id, 'la_current_labour_successful_induction'] = 'successful_induction'

            self.sim.schedule_event(LabourOnsetEvent(self.module, person_id), self.sim.date)
            # TODO: scheduling
        # For women whose induction fails they will undergo caesarean section
        else:
            logger.info('Persons %d labour has been unsuccessful induced', person_id)
            df.at[person_id, 'la_current_labour_successful_induction'] = 'failed_induction'
            # TODO: schedule CS or second attempt induction? -- will need to lead to labour event

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
    #    actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_PresentsForInductionOfLabour: did not run')
        pass

    #  TODO: 3 HSIs, for each facility level (interventions in functions)??


class HSI_Labour_PresentsForSkilledAttendanceInLabourFacilityLevel1(HSI_Event, IndividualScopeEventMixin):
    """This is the HSI PresentsForSkilledAttendanceInLabourFacilityLevel1. This event is scheduled by the LabourOnset
    Event. This event manages initial care around the time of delivery including prophylactic interventions (i.e. clean
    birth practices) for women presenting at Level 1 of the health system for delivery care. This event uses a womans
    stored risk of complications, which may be manipulated by treatment effects to determines if they will experience
    a complication during their labour in hospital. It is responsible for scheduling treatment HSIs for those
    complications."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_PresentsForSkilledAttendanceInLabour'
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['NormalDelivery'] = 1

        # Define the necessary information for an HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        mni = self.module.mother_and_newborn_info
        df = self.sim.population.props
        params = self.module.parameters

        # TODO: assert functions here?

        logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: Providing initial skilled attendance '
                    'at birth for person %d on date %s', person_id, self.sim.date)

    # ================================= CHECKING FACILITY TYPE AND ATTENDANT ==========================================

        # DUMMY- here we determine what facility type, within TLO facility level 1, a woman has presented too.
        if self.module.rng.random_sample() > params['dummy_prob_health_centre']:
            mni[person_id]['delivery_facility_type'] = 'health_centre'
        else:
            mni[person_id]['delivery_facility_type'] = 'hospital'

        # On presentation, we use the squeeze factor to determine if this woman will receive delivery care from a health
        # care professional, or if she will delivered unassisted in a facility
        if squeeze_factor > params['squeeze_factor_threshold_delivery_attendance']:
            mni[person_id]['delivery_attended'] = 'unattended'
            logger.debug('person %d is delivering without assistance at a level 1 health facility')
            logger.info(('%s|unattended_facility_delivery|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id}))
        else:
            mni[person_id]['delivery_attended'] = 'attended'

        assert mni[person_id]['delivery_facility_type'] is not None or mni[person_id]['delivery_attended'] is not None

    # ===================================== DEFINING CONSUMABLES (PROPHYLAXIS) =========================================

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_uncomplicated_delivery = pd.unique(consumables.loc[consumables[
                                                        'Intervention_Pkg'] == 'Vaginal delivery - skilled attendance',
                                                        'Intervention_Pkg_Code'])[0]
        # todo: do we defined the complicated delivery pkg here
        # todo: check defining here actually uses the resources within the health system

        pkg_code_clean_delivery_kit = pd.unique(consumables.loc[consumables[
                                                            'Intervention_Pkg'] == 'Clean practices and immediate'
                                                                                   ' essential newborn care '
                                                                                   '(in facility)',
                                                        'Intervention_Pkg_Code'])[0]

        item_code_abx_prom = pd.unique(
            consumables.loc[consumables['Items'] == 'Benzylpenicillin 1g (1MU), PFR_Each_CMST', 'Item_Code'])[0]

        pkg_code_pprom = pd.unique(consumables.loc[consumables['Intervention_Pkg'] == 'Antibiotics for pPRoM',
                                                   'Intervention_Pkg_Code'])[0]
        # n.b 2 additional IV abx not in guidelines

        # ===================================== PROPHYLACTIC INTERVENTIONS ============================================

        consumables_needed = {
            'Intervention_Package_Code': {pkg_code_uncomplicated_delivery: 1, pkg_code_clean_delivery_kit: 1,
                                          pkg_code_pprom: 1},
            'Item_Code': {item_code_abx_prom: 3}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)

        # If this womans delivery is attended by an SBA, we check if consumables are availble then administered
        # prophylactic interventions accordingly:
        if mni[person_id]['delivery_attended'] == 'attended':
            # Clean Delivery Kits
            if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_clean_delivery_kit]:
                logger.debug('This facility has delivery kits, so use it.')

                mni[person_id]['risk_ip_sepsis'] = mni[person_id]['risk_ip_sepsis'] * \
                    params['rr_maternal_sepsis_clean_delivery']

                mni[person_id]['risk_newborn_sepsis'] = mni[person_id]['risk_newborn_sepsis'] * \
                    params['rr_newborn_sepsis_clean_delivery']

            # If consumables are not available, the intervention is not delivered
            else:
                logger.debug('This facility has no delivery kits.')

            if mni[person_id]['PROM']:
                if outcome_of_request_for_consumables['Item_Code'][item_code_abx_prom]:
                    mni[person_id]['risk_ip_sepsis'] = mni[person_id]['risk_ip_sepsis'] *\
                                                       params['rr_sepsis_post_abx_prom']
                else:
                    logger.debug('This facility has no antibiotics for the treatment of PROM.')

            if mni[person_id]['PPROM']:
                if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_pprom]:
                    mni[person_id]['risk_ip_sepsis'] = mni[person_id]['risk_ip_sepsis'] * \
                                                        params['rr_sepsis_post_abx_pprom']
                else:
                    logger.debug('This facility has no antibiotics for the treatment of PROM.')
                    # todo: pprom treatment guidelines are written as if its happened antenatally,
                    #   not in labour? maybe we could use same consumables for both?

        # Steroids - premature
        # Group b streph prophy - premature
        # Magnesium - severe pre-eclampsia

    # ===================================== APPLYING COMPLICATION INCIDENCE ===========================================

        self.module.set_complications_during_facility_birth(person_id, complication='obstructed_labour',
                                                            labour_stage='ip')
        if df.at[person_id, 'la_obstructed_labour']:
            mni[person_id]['labour_is_currently_obstructed'] = True
            mni[person_id]['labour_has_previously_been_obstructed'] = True

        self.module.set_complications_during_facility_birth(person_id, complication='eclampsia', labour_stage='ip')

        self.module.set_complications_during_facility_birth(person_id, complication='antepartum_haem',
                                                            labour_stage='ip')

        self.module.set_complications_during_facility_birth(person_id, complication='sepsis', labour_stage='ip')

        self.module.set_complications_during_facility_birth(person_id, complication='uterine_rupture',
                                                            labour_stage='ip')

    # ======================================== COMPLICATION DIAGNOSIS =================================================
    # Should diagnosis be included in the above function
    # Now seperate between health centre and hospital

    # ================================= CONSUMABLES CHECK (BEmONC INTERVENTIONS) ======================================
    # ============================================== REFERRAL =========================================================


class HSI_Labour_ReceivesCareForPostpartumPeriodFacilityLevel1(HSI_Event, IndividualScopeEventMixin):
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
            'Intervention_Package_Code': {pkg_code_am: 1}, 'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

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

#        htn_treatment = HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy
#        self.module.\
#            set_complications_during_facility_birth(person_id, complication='eclampsia', labour_stage='pp',
#                                                    treatment_hsi=htn_treatment(self.module, person_id=person_id))

#        self.module.set_complications_during_facility_birth(person_id, complication='postpartum_haem',
#                                                            labour_stage='pp',
#                                                            treatment_hsi=HSI_Labour_ReceivesCareForMaternalHaemorrhage
#                                                            (self.module, person_id=person_id))

#        self.module.set_complications_during_facility_birth(person_id, complication='sepsis', labour_stage='pp',
#                                                            treatment_hsi=HSI_Labour_ReceivesCareForMaternalSepsis
#                                                            (self.module, person_id=person_id))

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # TODO: modify based on complications?
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_ReceivesCareForPostpartumPeriod: did not run')
        pass


class LabourLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """This is LabourLoggingEvent. Currently it calculates and produces a yearly output of maternal mortality (maternal
    deaths per 100,000 live births). It is incomplete"""
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))

    def apply(self, population):
        df = self.sim.population.props

        # n.b. this is all cause/time-point maternal death (will need to be focused intrapartum)

        # Maternal Mortality Ratio
        one_year_prior = self.sim.date - np.timedelta64(1, 'Y')
        live_births_sum = len(df.index[(df.date_of_birth > one_year_prior) & (df.date_of_birth < self.sim.date)])

        maternal_deaths = len(df.index[df.la_maternal_death & (df.la_maternal_death_date > one_year_prior) &
                                       (df.la_maternal_death_date < self.sim.date)])

    #    if maternal_deaths == 0:
    #        mmr = 0
    #    else:
    #        mmr = maternal_deaths/live_births_sum * 100000

    #    logger.info(f'The maternal mortality for this year date %s is {mmr} per 100,000 live births', self.sim.date)

        # Facility Delivery Rate

        # Still Birth Rate
        # Perinatal Mortality
        # Disease Incidence
        # Intervention incidence
