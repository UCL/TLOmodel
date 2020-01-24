import logging

import pandas as pd
import numpy as np
from pathlib import Path

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography, pregnancy_supervisor
from tlo.methods.healthsystem import HSI_Event
from tlo.lm import LinearModel, LinearModelType, Predictor


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
LOG_FILENAME = 'labour.log'
logging.basicConfig(filename=LOG_FILENAME,
                            filemode='w',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)


class Labour (Module):

    """
    This module models labour, delivery, immediate postpartum period and skilled birth attendance"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # Here we create a dictionary to store additional information around delivery and birth
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
        'la_postpartum_haem': Property(Types.BOOL, 'whether the woman has experienced an postpartum haemorrhage in this '
                                                   'delivery'),
        'la_postpartum_haem_disab': Property(Types.BOOL, 'disability associated with postpartum haemorrhage'),
        # TODO: above property could be categorical to reflect severity of bleed and map better with DALY weights
        'la_maternal_death': Property(Types.BOOL, ' whether the woman has died as a result of this pregnancy'),  # DUMMY
        'la_maternal_death_date': Property(Types.DATE, 'date of death for a date in pregnancy')  # DUMMY
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_LabourSkilledBirthAttendance.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)
        params = self.parameters

        # Here we will include DALY weights if applicable...

        if 'HealthBurden' in self.sim.modules.keys():
            params['daly_wts'] = \
                {'hemorrhage_moderate': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=339),
                 'haemorrhage_severe': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=338),
                 'maternal_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=340),
                 'eclampsia': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=343),
                 'obstructed_labour': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=348)}
            # TODO: Eclampsia DALY weight is empty- this is htn disorders sequalae code
            # TODO: source DALY weight for Uterine Rupture

# ======================================= LINEAR MODEL EQUATIONS ======================================================
        # Here we define the equations that will be used throughout this module using the linear model

        eptb_eq = LinearModel(
                LinearModelType.MULTIPLICATIVE,  # TODO: Anaemia/ Malaria / Multiple gestation
                params['prob_early_ptb'],
                Predictor('age_years').when('.between(15,20)', params['rr_early_ptb_age<20']),
                Predictor('la_has_previously_delivered_preterm').when(True, params['rr_early_ptb_prev_ptb'])
            )

        lptb_eq = LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_late_ptb'],
                Predictor('la_has_previously_delivered_preterm').when(True, params['rr_late_ptb_prev_ptb'])
            )

        # TODO: do we need an equation for post term labour

        ol_eq = LinearModel(
                LinearModelType.MULTIPLICATIVE, # TODO: stunting/malnutrition
                params['prob_pl_ol'],
                Predictor('la_parity').when('0', params['rr_PL_OL_nuliparity']),
                Predictor('la_parity').when('1', params['rr_PL_OL_para1']),
                Predictor('age_years').when('<20', params['rr_PL_OL_age_less20']),
            )

        sep_ip_eq = LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_ip_sepsis'],
                Predictor('la_obstructed_labour').when(True, params['rr_ip_sepsis_pl_ol']),
                # ISSUE: been using MNI in prediction models(may not work here)
            )

        sep_pp_eq = LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_pp_sepsis'],
                Predictor('la_obstructed_labour').when(True, params['rr_ip_sepsis_pl_ol']),
                # DUMMY, copy from above
            )

        ec_ip_eq =  LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_ip_eclampsia'],
                Predictor('age_years').when('.between(30,34)', params['rr_ip_eclampsia_30_34']),
                Predictor('age_years').when('>35', params['rr_ip_eclampsia_35']),
                Predictor('la_parity').when('0', params['rr_ip_eclampsia_nullip']),
            )

        ec_pp_eq = LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_ip_eclampsia'],
                Predictor('age_years').when('.between(30,34)', params['rr_ip_eclampsia_30_34']),
                Predictor('age_years').when('>35', params['rr_ip_eclampsia_35']),
                Predictor('la_parity').when('0', params['rr_ip_eclampsia_nullip']),
            )

        aph_eq = LinearModel(
                LinearModelType.MULTIPLICATIVE, # TODO: seperate causal influence in praevia and abruption
                params['prob_aph'],
                Predictor('la_obstructed_labour').when(True, params['rr_aph_pl_ol']),
                # ISSUE: been using MNI in prediction models(may not work here)
            )

        ur_eq = LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_uterine_rupture'],
                Predictor('la_parity').when('>4', params['rr_ur_grand_multip']),
                Predictor('la_previous_cs_delivery').when(True, params['rr_ur_prev_cs']),
                Predictor('la_obstructed_labour').when(True , params['rr_ur_ref_ol']),
            )

        pph_eq = LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_pph'],
                Predictor('la_obstructed_labour').when(True, params['rr_pph_pl_ol']),
                # ISSUE: been using MNI in prediction models(may not work here)
            )

        care_seeking = LinearModel(
                    LinearModelType.LOGISTIC,  # TODO: rough cut paper, would seeking care be better than % homebirth?
                    0.5,
                    Predictor('li_mar_stat').when('1', 1.83).when('3', 1.83),
                    Predictor('li_wealth').when('4', 0.51).when('5', 0.48), # wealth levels in the paper are different
                    Predictor('li_urban').when(True, 0.39),
            )
        # TODO: we could hard code these predictors to reduce the number of actual parameters?

        # We store the equations within a dictionary parameter to be accessed later
        params['la_labour_equations'] = {'early_preterm_birth': eptb_eq, 'late_preterm_birth': lptb_eq,
                                         'obstructed_labour_ip': ol_eq, 'sepsis_ip': sep_ip_eq,
                                         'sepsis_pp': sep_pp_eq, 'eclampsia_ip': ec_ip_eq,
                                         'eclampsia_pp': ec_pp_eq, 'antepartum_haem_ip': aph_eq,
                                         'postpartum_haem_pp': pph_eq, 'uterine_rupture_ip': ur_eq,
                                         'care_seeking': care_seeking}

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        df = population.props
        m = self
        rng = m.rng
        params = self.parameters

    # ----------------------------------------- DEFAULTS ---------------------------------------------------------------

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

# ------------------------------------ REPACKAGE PARAMETERS------------------------------------------------------------
        # todo: repackage paramters into dictionaries if worthwhile?

# -----------------------------------ASSIGN PREGNANCY AND DUE DATE AT BASELINE (DUMMY) --------------------------------
        # TODO: Tim colbourn will be applying pregnancy at baseline

        # !!!!!!!!!!!!!!!!(DUMMY CODE) THIS WILL BE REPLACED BY CONTRACEPTION CODE (TC) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Get and hold all the women who are eligible to become pregnant at baseline
        women_idx = df.index[(df.age_years >= 15) & (df.age_years <= 49) & df.is_alive & (df.sex == 'F')]

        # Apply an effective probability of pregnancy at baseline and allocate these women to be pregnant
        eff_prob_preg = pd.Series(m.prob_pregnancy, index=women_idx)

        random_draw = pd.Series(self.rng.random_sample(size=len(women_idx)),
                                index=df.index[(df.age_years >= 15) & (df.age_years <= 49) & df.is_alive
                                               & (df.sex == 'F')])

        dfx = pd.concat([eff_prob_preg, random_draw], axis=1)
        dfx.columns = ['eff_prob_pregnancy', 'random_draw']
        idx_pregnant = dfx.index[dfx.eff_prob_pregnancy > dfx.random_draw]
        df.loc[idx_pregnant, 'is_pregnant'] = True

        logger.debug(idx_pregnant)

# ---------------------------------    GESTATION AND SCHEDULING BIRTH BASELINE  ---------------------------------------

        # TODO: random_intergers produces same random integer for each person needs a fix
        # TODO: consider re-writing using a function

        # Here we apply prevalence of early preterm, late preterm, term and post term labour to all women pregnant at
        # baseline
        pregnant_idx = df.index[df.is_pregnant & df.is_alive]
        due_date_period = pd.Series(self.rng.choice(('early_preterm', 'late_preterm', 'term', 'post_term'),
                                                    p=[0.04, 0.12, 0.80, 0.04],size=len(pregnant_idx)),
                                    index=pregnant_idx)
        # TODO: use parameter values for probabilities above
        simdate = pd.Series(self.sim.date, index=pregnant_idx)
        dfx = pd.concat((simdate, due_date_period), axis=1)
        dfx.columns = ['simdate', 'due_date_period']

        # Then the following function is applied to each group, based on the catagories asigned above, to determine
        # gestation and due date
        def gestation_and_due_date_setting(df, dfx, rng, gestation, upper_limit_gest, lower_rand, higher_rand, prob):
            """This function sets a gestational age in weeks and future due date for all women pregnant at baseline"""
            # TODO: this is still applying the same random number to the index

            index = dfx.index[dfx.due_date_period == f'{gestation}']
            df.loc[index, 'ps_gestational_age_in_weeks'] = (pd.Series(rng.random_integers(0, upper_limit_gest),
                                                                      index=index))
            df.loc[index, 'date_of_last_pregnancy'] = self.sim.date - pd.to_timedelta(df['ps_gestational_age_in_weeks'],
                                                                                      unit='w')
            weeks_till_due = (pd.Series(rng.choice(list(range(lower_rand, higher_rand)), size=len(index), replace=True,
                                                   p=[(1 / prob)] * prob), index=index))
            due_on = weeks_till_due - df.loc[index, 'ps_gestational_age_in_weeks']
            df.loc[index, 'la_due_date_current_pregnancy'] = self.sim.date + pd.to_timedelta(due_on, unit='w')

        gestation_and_due_date_setting(df, dfx, self.rng, 'term', 36, 37, 42, 5)
        gestation_and_due_date_setting(df, dfx, self.rng, 'post_term', 41, 42, 47, 5)
        gestation_and_due_date_setting(df, dfx, self.rng, 'early_preterm', 23, 24, 33, 9)
        gestation_and_due_date_setting(df, dfx, self.rng, 'late_preterm', 32, 33, 37, 4)

        # Then all women are scheduled to go into labour on this due date
        for person in pregnant_idx:
            assert df.at[person, 'la_due_date_current_pregnancy'] > self.sim.date
            labour = LabourEvent(self, individual_id=person, cause='Labour')
            self.sim.schedule_event(labour, df.at[person, 'la_due_date_current_pregnancy'])

        # Todo: consider if we should apply risk factors to women at baseline (anaemia and age)

#  ----------------------------ASSIGNING PARITY AT BASELINE (DUMMY)-----------------------------------------------------

        # TODO: reformat with linear model? consider how best to to draw in real 2010 data

        # Get and hold all the women in the dataframe who between the ages of 15-24 years old
        women_parity_1524_idx = df.index[(df.age_years >= 15) & (df.age_years <= 24) & (df.is_alive == True)
                                         & (df.sex == 'F')]

        baseline_p = pd.Series(0, index=women_parity_1524_idx)

        # Probability weighted random draw is applied to each women to determine how many previous deliveries she has
        # had
        random_draw2 = pd.Series(self.rng.choice(range(0, 5), p=[0.40, 0.35, 0.15, 0.06, 0.04],
                                                 size=len(women_parity_1524_idx)),
                                 index=df.index[(df.age_years >= 15) & (df.age_years <= 24)
                                                & df.is_alive & (df.sex == 'F')])

        dfx = pd.concat([baseline_p, random_draw2], axis=1)
        dfx.columns = ['baseline_p', 'random_draw2']
        idx_parity = dfx.index[dfx.baseline_p < dfx.random_draw2]
        df.loc[idx_parity, 'la_parity'] = dfx.random_draw2

        # These steps are repeated with different weightings for the next two older age groups
        women_parity_2540_idx = df.index[(df.age_years >= 25) & (df.age_years <= 40) & (df.is_alive == True)
                                         & (df.sex == 'F')]

        baseline_p = pd.Series(0, index=women_parity_2540_idx)

        random_draw = pd.Series(self.rng.choice(range(0, 6), p=[0.05, 0.15, 0.30, 0.20, 0.2, 0.1],
                                                size=len(women_parity_2540_idx)), index=df.index[(df.age_years >= 25) &
                                                                                                 (df.age_years <= 40)
                                                                                                 & (df.is_alive == True)
                                                                                                 & (df.sex == 'F')])

        dfx = pd.concat([baseline_p, random_draw], axis=1)
        dfx.columns = ['baseline_p', 'random_draw']
        idx_parity = dfx.index[dfx.baseline_p < dfx.random_draw]
        df.loc[idx_parity, 'la_parity'] = dfx.random_draw

        women_parity_4149_idx = df.index[(df.age_years >= 41) & (df.age_years <= 49) & (df.is_alive == True)
                                         & (df.sex == 'F')]

        baseline_p = pd.Series(0, index=women_parity_4149_idx)

        random_draw = pd.Series(self.rng.choice(range(0, 7), p=[0.05, 0.10, 0.25, 0.30, 0.25, 0.03, 0.02],
                                                size=len(women_parity_4149_idx)), index=df.index[(df.age_years >= 41)
                                                                                                 & (df.age_years <= 49)
                                                                                                 & (df.is_alive == True)
                                                                                                 & (df.sex == 'F')])

        dfx = pd.concat([baseline_p, random_draw], axis=1)
        dfx.columns = ['baseline_p', 'random_draw']
        idx_parity = dfx.index[dfx.baseline_p < dfx.random_draw]
        df.loc[idx_parity, 'la_parity'] = dfx.random_draw

    # ------------------------------ ASSIGN PREVIOUS CS AT BASELINE -----------------------------------------------

        # First we get  and hold women who have delivered at least one child
        women_para1_idx = df.index[(df.la_parity >= 1)]
        random_draw = pd.Series(self.rng.random_sample(size=len(women_para1_idx)),
                                index=df.index[(df.la_parity >= 1)])
        prev_cs_baseline = pd.Series(params['baseline_prev_cs'], index=women_para1_idx)

        dfx = pd.concat([prev_cs_baseline, random_draw], axis=1)
        dfx.columns = ['prev_cs_baseline', 'random_draw']
        idx_cs = dfx.index[dfx.prev_cs_baseline < dfx.random_draw]
        df.loc[idx_cs, 'la_previous_cs_delivery'] = True

    # ------------------------------ ASSIGN PREVIOUS PTB AT BASELINE ----------------------------------------------
        # TODO: Commented out whilst consider pulling in from another data source

        # Get and hold all women who have given birth previously, excluding those with previous caesarean section
#        women_para1_nocs_idx = df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') & (df.la_parity == 1) &
#                                        ~df.la_previous_cs_delivery]

        # Get and hold all women with greater than 2 deliveries excluding those in which both deliveries were by
        # caesarean
#        women_para2_idx = df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') & (df.la_parity >= 2) &
#                                   (df.la_total_deliveries_by_cs < 2)]

#        baseline_ptb = pd.Series(m.prob_ptl, index=women_para1_nocs_idx)
#        baseline_ptb_p2 = pd.Series(m.prob_ptl, index=women_para2_idx)

#        random_draw = pd.Series(self.rng.random_sample(size=len(women_para1_nocs_idx)),
#                                index=df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') &
#                                               (df.la_parity == 1) & (df.la_total_deliveries_by_cs == 0)])
#        random_draw2 = pd.Series(self.rng.random_sample(size=len(women_para2_idx)),
#                                index=df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') &
#                                               (df.la_parity >= 2) & (df.la_total_deliveries_by_cs < 2)])

        # Use a random draw to determine if this woman's past deliveries have ever been preterm
#        dfx = pd.concat([baseline_ptb, random_draw], axis=1)
#        dfx.columns = ['baseline_ptb', 'random_draw']
#        idx_prev_ptb = dfx.index[dfx.baseline_ptb > dfx.random_draw]
#        df.loc[idx_prev_ptb, 'la_has_previously_delivered_preterm'] = True

#        dfx = pd.concat([baseline_ptb_p2, random_draw2], axis=1)
#        dfx.columns = ['baseline_ptb_p2', 'random_draw2']
#        idx_prev_ptb = dfx.index[dfx.baseline_ptb_p2 > dfx.random_draw2]
#        df.loc[idx_prev_ptb, 'la_has_previously_delivered_preterm'] = True

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
                        {
                            'mother': mother_id,
                            'child': child_id,
                            'mother_age': df.at[mother_id, 'age_years']
                        })

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

        # TODO: work out why values have to be hard coded, wont read parameters?
        # TODO: Refine disability levels (could include severity) and explore a more elegant solution involving less
        #  properties
        # TODO: Make sure that value >1.0 is being reported as current fix is temp

        logger.debug('This is Labour reporting my health values')

        df = self.sim.population.props  # shortcut to population properties data frame
        p = self.parameters

#        def get_health_values(sequence, cause): (use return)
#            sequence = df.loc[df.is_alive, f'la_{cause}_disability'].map(
#                {False: 0, True: p['daly_wts'][f'{cause}']})  # p['daly_wts']['obstructed_labour']# 0.324
#            sequence.name = cause # this needs to change with

        health_values_1 = df.loc[df.is_alive, 'la_obstructed_labour_disab'].map(
            {False: 0, True: 0.324})  # p['daly_wts']['obstructed_labour']
        health_values_1.name = 'Obstructed Labour'

        health_values_2 = df.loc[df.is_alive, 'la_eclampsia_disab'].map(
            {False: 0, True: 0.5})  # p['daly_wts']['eclampsia']
        health_values_2.name = 'Eclampsia'

        health_values_3 = df.loc[df.is_alive, 'la_sepsis_disab'].map(
            {False: 0, True: 0.133})  # p['daly_wts']['maternal_sepsis']
        health_values_3.name = 'Maternal Sepsis'

        health_values_4 = df.loc[df.is_alive, 'la_antepartum_haem_disab'].map(  # TODO: consider severity
            {False: 0, True: 0.324})  # p['daly_wts']['haemorrhage_severe']
        health_values_4.name = 'Antepartum Haemorrhage'

        health_values_5 = df.loc[df.is_alive, 'la_postpartum_haem_disab'].map(  # TODO: consider severity
            {False: 0, True: 0.324})  # p['daly_wts']['haemorrhage_severe']
        health_values_5.name = 'Postpartum Haemorrhage'

        health_values_6 = df.loc[df.is_alive, 'la_uterine_rupture_disab'].map(  # TODO: consider severity
            {False: 0, True: 0.5})  # p['daly_wts']['haemorrhage_severe']
        health_values_6.name = 'Uterine Rupture'

        health_values_df = pd.concat([health_values_1.loc[df.is_alive], health_values_2.loc[df.is_alive],
                                      health_values_3.loc[df.is_alive], health_values_4.loc[df.is_alive],
                                      health_values_5.loc[df.is_alive], health_values_6.loc[df.is_alive]], axis=1)

        # Must not have one person with more than 1.00 daly weight
        # Hot fix - scale such that sum does not exceed one.
        scaling_factor = (health_values_df.sum(axis=1).clip(lower=0, upper=1) /
                          health_values_df.sum(axis=1)).fillna(1.0)
        health_values_df = health_values_df.multiply(scaling_factor,axis=0)

        return health_values_df  # return the dataframe

# ===================================== HELPER FUNCTIONS ===============================================================

    def eval(self, eq, person_id):
        """Compares the result of a specific linear equation with a random draw providing a boolean for the outcome
        under examination"""
        return self.rng.random_sample(size=1) < eq.predict(self.sim.population.props.loc[[person_id]]).values

    def complication_application(self, df, mni, individual_id, params, labour_stage, complication):
        """Uses the result of a linear equation to determine the probability of a certain complication, stores the
        probability for women delivery in facility and applies the probability for women delivering at home.
        Sets complication properties according to the result"""

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


class LabourScheduler (Event, IndividualScopeEventMixin):
    """This event determines the gestation at which women will be scheduled to go into labour"""

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        logger.debug('person %d is having their labour scheduled on date %s', individual_id, self.sim.date)
        # Todo: could this be restructured to use 'eval' function

        # Using the linear equations defined above we calculate this womans individual risk of early and late preterm
        # labour
        eptb_prob = params['la_labour_equations']['early_preterm_birth'].predict(df.loc[[individual_id]]).values
        lptb_prob = params['la_labour_equations']['late_preterm_birth'].predict(df.loc[[individual_id]]).values

        # We then use a random draw to determine if the woman will go into preterm labour and how early she will deliver
        random_draw = self.module.rng.random_sample(size=1)
        if (random_draw < lptb_prob) & (random_draw > eptb_prob):
            random = int(self.module.rng.random_integers(34, 36))
            df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] + \
                pd.Timedelta(random, unit='W')

        elif random_draw < eptb_prob or ((random_draw < eptb_prob) & (random_draw < lptb_prob)):
            random = int(self.module.rng.random_integers(24, 33))
            df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] + \
                pd.Timedelta(random, unit='W')

        # For women who will deliver after term we apply a risk of post term birth
        elif random_draw > lptb_prob:
            random_draw_2 = self.module.rng.random_sample(size=1)
            if random_draw_2 < params['prob_potl']:
                random = int(self.module.rng.random_integers(42, 46))
                df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] \
                    + pd.Timedelta(random, unit='W')

            else:
                random = int(self.module.rng.random_integers(37, 41))
                df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] \
                    + pd.Timedelta(random, unit='W')

        # Here we check that no one can go into labour before 24 weeks gestation
        days_until_labour = df.at[individual_id, 'la_due_date_current_pregnancy'] - self.sim.date
        assert days_until_labour >= pd.Timedelta(168, unit='d')

        # and then we schedule the labour for that womans due date
        self.sim.schedule_event(LabourEvent(self.module, individual_id, cause='labour'),
                                df.at[individual_id, 'la_due_date_current_pregnancy'])


class LabourEvent(Event, IndividualScopeEventMixin):

    """This is the Labour Event. It has been scheduled to occur on the date that Labour is expected. There are a number
     of potential clinical outcomes of labour including obstructed labour, antepartum haemorrhage, sepsis, eclampsia,
     uterine rupture, postpartum haemorrhage, stillbirth and maternal death."""

    def __init__(self, module, individual_id, cause):
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
                              'risk_obstructed_labour': params['prob_pl_ol'],
                              'labour_is_currently_obstructed': False,  # True (T) or False (F)
                              'labour_has_previously_been_obstructed': False,
                              'risk_ip_sepsis': params['prob_ip_sepsis'],
                              'risk_pp_sepsis': params['prob_pp_sepsis'],
                              'sepsis': False,  # True (T) or False (F)
                              'sepsis_pp': False,  # True (T) or False (F)
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
                              'uterine_rupture': False,   # True (T) or False (F)
                              'grade_of_UR': 'X', # Partial (P) or Complete (C)
                              'risk_pp_postpartum_haem': params['prob_pph'],
                              'postpartum_haem': False,   # True (T) or False (F)
                              'source_pph': None,  # Uterine Atony (UA) or Retained Products/Placenta (RPP)
                              'severity_pph': None,
                              'risk_newborn_sepsis': params['prob_neonatal_sepsis'],
                              'risk_newborn_ba': params['prob_neonatal_birth_asphyxia'],
                              #  Should this just be risk of asyphixa
                              'mode_of_delivery': None, # Vaginal Delivery (VD),Vaginal Delivery Induced (VDI),
                              # Assisted Vaginal Delivery Forceps (AVDF) Assisted Vaginal Delivery Ventouse (AVDV)
                              # Caesarean Section (CS)
                              'death_in_labour': False,  # True (T) or False (F)
                              'cause_of_death_in_labour': [], # Appended list of cause/causes of death
                              # TODO: should we do the same thing for still birth? Would help with mapping
                              'stillbirth_in_labour': False,  # True (T) or False (F)
                              'death_postpartum': False}  # True (T) or False (F)

# ===================================== LABOUR STATE  ==================================================================
        logger.debug('person %d has just reached LabourEvent on %s', individual_id, self.sim.date)

        # Debug logging to highlight the women who have miscarried/had an abortion/stillbirth who still come to the
        # event
        if ~df.at[individual_id, 'is_pregnant']:
            logger.debug('person %d has just reached LabourEvent on %s but is no longer pregnant',
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
                logger.debug(f'This is LabourEvent, person %d has now gone into {labour_state} on date %s',
                             individual_id, self.sim.date)

# ======================================= PROBABILITY OF HOME DELIVERY =====================================
                facility_delivery = HSI_Labour_PresentsForSkilledAttendanceInLabour(self.module,
                                                                                    person_id=individual_id)

                # n.b. current regression uses outcome of home delivery not facility delivery

                # As women who have presented for induction are already in a facility we exclude them, then apply a
                # probability of home birth
                if self.module.eval(self, params['la_labour_equations']['care_seeking'], individual_id) & \
                   (df.at[individual_id, 'la_current_labour_successful_induction'] == 'not_induced'):
                    mni[individual_id]['delivery_setting'] = 'HB'
                    logger.debug('This is LabourEvent, doing nothing as person %d will not seek care in labour and will'
                                 'deliver at home', individual_id)

                else:
                    mni[individual_id]['delivery_setting'] = 'FD'
                    logger.info(
                        'This is LabourEvent, scheduling HSI_Labour_PresentsForSkilledAttendanceInLabour on date %s for'
                        ' person %d as they have chosen to seek care for delivery', self.sim.date, individual_id)

                    self.sim.modules['HealthSystem'].schedule_hsi_event(facility_delivery,
                                                                        priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=1)
                                                                        )

                # Here we schedule delivery care for women who have already sought care for induction, whether or not
                # that induction was successful
                if (df.at[individual_id, 'la_current_labour_successful_induction'] == 'failed_induction') or \
                   (df.at[individual_id, 'la_current_labour_successful_induction'] == 'successful_induction'):
                    self.sim.modules['HealthSystem'].schedule_hsi_event(facility_delivery,
                                                                        priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=1)
                                                                        )

# ============================ INDIVIDUAL RISK OF COMPLICATIONS DURING LABOUR =========================================
# DEAL WITH VERY EARLY PRETERM BIRTHS?!?!

# ====================================== STATUS OF MEMBRANES ==========================================================

        # Here we apply a risk that this woman's labour was preceded by premature rupture of membranes, in preterm
        # women this has likely predisposed their labour
                if (mni[individual_id]['labour_state'] == 'EPTL') or (mni[individual_id]['labour_state'] == 'LPTL'):
                    random = self.module.rng.random_sample(size=1)
                    if random < params['prob_prom']:
                        mni[individual_id]['PROM'] = True
                # todo: term labour can have prom also (also this should be an antenatal thing?)

# ===================================  APPLICATION OF COMPLICATIONS ====================================================

        # Using the function generated at the beginning of labour we move through each complication in turn determining
        # if a woman will experience any of these if she has delivered at home, and storing the risk so it can
        # be modified by skilled birth attendance for women delivering in a facility

                self.module.complication_application(df, mni, individual_id, params, 'ip', 'obstructed_labour')
                if df.at[individual_id, 'la_obstructed_labour']:  # there's a neater way of doing this
                    mni[individual_id]['labour_is_currently_obstructed'] = True
                    mni[individual_id]['labour_has_previously_been_obstructed'] = True
                # TODO: As OL is a contraindication of induction, should we skip it here for induced women?
                #  late diagnosis of obstruction is possible surely

                self.module.complication_application(df, mni, individual_id, params, 'ip', 'antepartum_haem')
                self.module.complication_application(df, mni, individual_id, params, 'ip', 'sepsis')
                self.module.complication_application(df, mni, individual_id, params, 'ip', 'eclampsia')
                self.module.complication_application(df, mni, individual_id, params, 'ip', 'uterine_rupture')

        # Here we schedule the birth event for 2 days after labour- we do this prior to the death event as women who
        # die but still deliver a live child will pass through birth event
                due_date = df.at[individual_id, 'la_due_date_current_pregnancy']
                self.sim.schedule_event(BirthEvent(self.module, individual_id), due_date + DateOffset(days=3))
                logger.debug('This is LabourEvent scheduling a birth on date %s to mother %d', due_date, individual_id)

        # We schedule all women to move through the death event where those who have developed a complication that
        # hasn't been treated or treatment has failed will have a case fatality rate applied
                self.sim.schedule_event(LabourDeathEvent(self.module, individual_id, cause='labour'), self.sim.date +
                                        DateOffset(days=2))
                logger.debug('This is LabourEvent scheduling a potential death on date %s for mother %d', self.sim.date,
                             individual_id)

        # Here we set the due date of women who have been induced to pd.NaT so they dont go into labour twice
                if df.at[individual_id, 'la_current_labour_successful_induction']:
                    df.at[individual_id, 'la_due_date_current_pregnancy'] = pd.NaT

        # TODO: here we need to use the symptom manager to determine if women who are delivering at home with comps
        #  will seek care


class BirthEvent(Event, IndividualScopeEventMixin):
    """A one-off event in which a pregnant mother gives birth.
    """

    def __init__(self, module, mother_id):
        """Create a new birth event."""
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
            self.sim.schedule_event(PostpartumLabourEvent(self.module, mother_id, cause='post partum'),
                                    self.sim.date)

        # As only live births contribute to parity we exlcuded women who have had an IP stillbirth
        if df.at[mother_id, 'is_alive'] and df.at[mother_id, 'is_pregnant'] & \
            ~df.at[mother_id,'la_intrapartum_still_birth']:
            df.at[mother_id, 'la_parity'] += 1

        # If the mother has died during childbirth the child is still generated with is_alive=false to monitor
        # stillbirth rates. She will not pass through the postpartum complication events
        if df.at[mother_id, 'is_alive'] == False & df.at[mother_id, 'is_pregnant'] == True & \
            (mni[mother_id]['death_in_labour'] == True):
            self.sim.do_birth(mother_id)
            df.at[mother_id, 'is_pregnant'] = False


class PostpartumLabourEvent(Event, IndividualScopeEventMixin):

    """applies probability of postpartum complications to women who have just delivered """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self
        mni = self.module.mother_and_newborn_info
  
        if df.at[individual_id, 'is_alive']:
            self.module.complication_application(df, mni, individual_id, params, 'pp', 'postpartum_haem')
            self.module.complication_application(df, mni, individual_id, params, 'pp', 'sepsis')
            self.module.complication_application(df, mni, individual_id, params, 'pp', 'eclampsia')

        # If a woman has delivered in a facility we schedule her to now receive additional care following birth
            if mni[individual_id]['delivery_setting'] == 'FD':
                logger.info('This is PostPartumEvent scheduling HSI_Labour_ReceivesCareForPostpartumPeriod for person '
                            '%d on date %s', individual_id, self.sim.date)
                
                event = HSI_Labour_ReceivesCareForPostpartumPeriod(self.module, person_id=individual_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1)
                                                                    )
                # TODO: same issue for women who seek care but cant be seek- comps wont be allocated!

            # We schedule all women to then go through the death event where those with untreated/unsuccessfully treated
            # complications may experience death
                
            self.sim.schedule_event(PostPartumDeathEvent(self.module, individual_id, cause='labour'), self.sim.date)

            logger.info('This is PostPartumEvent scheduling a potential death for person %d on date %s', individual_id,
                        self.sim.date + DateOffset(days=3))  # Date offsetted to allow for interventions

            # Here we schedule women to an event which resets 'daly' disability associated with delivery complications
            self.sim.schedule_event(DisabilityResetEvent(self.module, individual_id, cause='reset'),
                                    self.sim.date + DateOffset(months=1))


class LabourDeathEvent (Event, IndividualScopeEventMixin):

    """This is the LabourDeathEvent. This event will determine if women who are expericing a complication associate
    with labour will die because of it. Not all women who pass through this event will die. """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self
        mni = self.module.mother_and_newborn_info

        # Currently we apply an untreated case fatality ratio (dummy values presently) who have experienced a
        # complication
        # Similarly we apply a risk of still birth associated with each complication

        def dies_by_complication(rng, df, mni, cause):
            logger.debug(f'{cause} has contributed to person %d death during labour', individual_id)
            random = rng.random_sample(size=1)
            if random < params[f'cfr_{cause}']:
                mni[individual_id]['death_in_labour'] = True
                mni[individual_id]['cause_of_death_in_labour'].append(cause)
                df.at[individual_id, 'la_maternal_death'] = True
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date

                # then if she does die, we determine if the child will still survive
                random = self.module.rng.random_sample()
                if random < params[f'prob_still_birth_{cause}_md']:
                    df.at[individual_id, 'la_intrapartum_still_birth'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True
                    df.at[individual_id, 'ps_previous_stillbirth'] = True

                # Otherwise we just determine if this complication will lead to a stillbirth
                else:
                    random = rng.random_sample(size=1)
                    if random < params[f'prob_still_birth_{cause}']:
                        df.at[individual_id, 'la_intrapartum_still_birth'] = True
                        mni[individual_id]['stillbirth_in_labour'] = True
                        df.at[individual_id, 'ps_previous_stillbirth'] = True

            # First we determine if the mother will die due to her complication
            if mni[individual_id]['labour_is_currently_obstructed']:
                dies_by_complication(self.module.rng, df, mni, 'obstructed_labour')

            if mni[individual_id]['eclampsia_ip']:
                dies_by_complication(self.module.rng, df, mni, 'eclampsia')

            if mni[individual_id]['antepartum_haem']:
                dies_by_complication(self.module.rng, df, mni, 'aph')

            if mni[individual_id]['sepsis_ip']:
                dies_by_complication(self.module.rng, df, mni, 'sepsis')

            if mni[individual_id]['uterine_rupture']:
                dies_by_complication(self.module.rng, df, mni, 'uterine_rupture')

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

            complication_profile = mni[individual_id]
            logger.info('%s|labour_complications|%s', self.sim.date,
                        {'person_id': individual_id,
                         'labour_profile': complication_profile})

            logger.info('%s|maternal_death|%s', self.sim.date,
                        {'age': df.at[individual_id, 'age_years'],
                            'person_id': individual_id })

        if df.at[individual_id, 'la_intrapartum_still_birth']:
            logger.info('@@@@ A Still Birth has occurred, to mother %s', individual_id)
            logger.info('%s|still_birth|%s', self.sim.date,
                        {'mother_id': individual_id})


class PostPartumDeathEvent (Event, IndividualScopeEventMixin):

    """handles death following labour"""

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self
        mni = self.module.mother_and_newborn_info

        # We apply the same structure as with the LabourDeathEvent to women who experience postpartum complications
        # if df.at[individual_id, 'la_eclampsia']:

        def dies_by_complication_postpartum(rng, df, mni, cause):
            random = self.module.rng.random_sample(size=1)
            if random < params[f'cfr_pp_{cause}']:
                logger.debug(f'{cause} has contributed to person %d death following labour', individual_id)
                mni[individual_id]['death_postpartum'] = True
                df.at[individual_id, 'la_maternal_death'] = True
                mni[individual_id]['cause_of_death_in_labour'].append(f'{cause}_postpartum')
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date
        
        if mni[individual_id]['eclampsia_pp']:
            dies_by_complication_postpartum(self.module.rng, df, mni, 'eclampsia')

        if mni[individual_id]['postpartum_haem']:
            dies_by_complication_postpartum(self.module.rng, df, mni, 'pph')

        if mni[individual_id]['sepsis_pp']:
            dies_by_complication_postpartum(self.module.rng, df, mni, 'sepsis')

        if mni[individual_id]['death_postpartum']:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                  cause='postpartum labour'), self.sim.date)

    # TODO: amend cause= 'labour_' + [str(cause) + '_' for cause in list(mni[individual_id][cause_of_death_in_labour]

            logger.debug('This is PostPartumDeathEvent scheduling a death for person %d on date %s who died due to '
                         'postpartum complications', individual_id,
                         self.sim.date)

            logger.info('%s|maternal_death|%s', self.sim.date,
                        {'age': df.at[individual_id, 'age_years'],
                            'person_id': individual_id})

            complication_profile = mni[individual_id]
            logger.debug('%s|labour_complications|%s', self.sim.date,
                        {'person_id': individual_id,
                         'labour_profile': complication_profile})
        else:
            # Surviving women pass through the DiseaseResetEvent to ensure all complication variable are set to false
            # TODO: Consider how best to deal with complications that are long lasting.
            self.sim.schedule_event(DiseaseResetEvent(self.module, individual_id, cause='reset'),
                                    self.sim.date + DateOffset(weeks=1))

            logger.debug('%s|labour_complications|%s', self.sim.date,
                        {'person_id': individual_id,
                         'labour_profile': mni[individual_id]})


class DisabilityResetEvent (Event, IndividualScopeEventMixin):
    """resets a woman's disability properties one month after labour"""

    def __init__(self, module, individual_id, cause):
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
    """resets a woman's disability properties one month after labour"""

    def __init__(self, module, individual_id, cause):
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
    """
    This is a Health System Interaction Event.
    This interaction manages induction of labour for women with indications identified antenatally such as severe
    pre-eclampsia and being post term
    """

    # TODO: await antenatal care module to schedule induction
    # TODO: women with praevia, obstructed labour, should be induced

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)
        
        self.TREATMENT_ID = 'Labour_PresentsForInductionOfLabour'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1
        # TODO: review appt footprint as this wont include midwife time?

        # Define the necessary information for an HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props  # shortcut to the dataframe
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        logger.debug('This is HSI_Labour_PresentsForInductionOfLabour, person %d is attending a health facility to have'
                     'their labour induced on date %s', person_id, self.sim.date)

        # TODO: discuss squeeze factors/ consumable back up with TC

        df = self.sim.population.props  # shortcut to the dataframe
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
                hsi_event=self, cons_req_as_footprint=consumables_needed
            )
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_induction]:
            logger.debug('pkg_code_induction is available, so use it.')
            # TODO: reschedule if consumables aren't available at this point in time?
        else:
            logger.debug('PkgCode1 is not available, so can' 't use it.')

        # We use a random draw to determine if this womans labour will be successfully induced
        # Indications: Post term, eclampsia, severe preeclampsia, mild preeclampsia at term, PROM > 24 hrs at term
        # or PPROM > 34 weeks EGA, and IUFD.

        random = self.module.rng.random_sample(size=1)
        if random < params['prob_successful_induction']:
            logger.info('Person %d has had her labour successfully induced', person_id)
            df.at[person_id, 'la_current_labour_successful_induction'].values[:] = 'successful_induction'
            self.sim.schedule_event(LabourEvent(self.module, person_id, cause='labour'), self.sim.date)

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

#TODO: 3 HSIs, for each facility level (interventions in functions)

class HSI_Labour_PresentsForSkilledAttendanceInLabour(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This interaction manages the care a woman receives when she is admitted to a facility in spontanous labour or in
    labour following an induction
    """
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

    def apply(self, person_id, squeeze_factor):
        # TODO: Squeeze factor, consumable conditions?

        logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: Providing initial skilled attendance '
                    'at birth for person %d on date %s', person_id, self.sim.date)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_sba_uncomp = pd.unique(consumables.loc[consumables[
                                                      'Intervention_Pkg'] ==
                                                  'Vaginal delivery - skilled attendance',
                                                  'Intervention_Pkg_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code_sba_uncomp: 1}],
            'Item_Code': [],
        }

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_sba_uncomp]:
            logger.debug('PkgCode1 is available, so use it.')
        else:
            logger.debug('PkgCode1 is not available, so can' 't use it.')
            # TODO: If delivery pack isnt availble then birth will still occur but should have risk of sepsis?

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

        # Antibiotics for group b strep prophylaxsis: (are we going to apply this to all cause sepsis?)
        if mni[person_id]['labour_state'] == 'EPTL' or mni[person_id]['labour_state'] == 'LPTL':
            treatment_effect = params['rr_newborn_sepsis_proph_abx']
            new_newborn_risk = mni[person_id]['risk_newborn_sepsis'] * treatment_effect
            mni[person_id]['risk_newborn_sepsis'] = new_newborn_risk

        # TODO: STEROIDS - effect applied directly to newborns (need consumables here) store within maternal MNI
        # TOXOLYTICS !? - WHO advises against

# ===================================  COMPLICATIONS OF LABOUR ========================================================

    # Here, using the adjusted risks calculated following 'in-labour' interventions to determine which complications a
    # woman may experience and store those in the dataframe

        random = self.module.rng.random_sample(size=1)
        if random < mni[person_id]['risk_obstructed_labour']:
            df.at[person_id, 'la_obstructed_labour'] = True
            df.at[person_id, 'la_obstructed_labour_disab'] = True
            mni[person_id]['labour_is_currently_obstructed'] = True
            mni[person_id]['labour_has_previously_been_obstructed'] = True

            logger.debug('person %d is experiencing obstructed labour in a health facility',
                        person_id)

            # TODO: issue- if we apply risk of both UR and OL here then we will negate the effect of OL treatment on
            #  reduction of incidence of UR

        random = self.module.rng.random_sample(size=1)
        if random < mni[person_id]['risk_ip_eclampsia']:
            df.at[person_id, 'la_eclampsia'] = True
            df.at[person_id, 'la_eclampsia_disability'] = True
            mni[person_id]['eclampsia_ip'] = True

            logger.debug('person %d is experiencing eclampsia in a health facility',
                        person_id)
            logger.info('%s|eclampsia|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})

        if random < mni[person_id]['risk_antepartum_haem']:
            df.at[person_id, 'la_antepartum_haem'] = True
            df.at[person_id, 'la_haemorrhage_disability'] = True
            mni[person_id]['antepartum_haem'] = True

            logger.debug('person %d is experiencing an antepartum haemorrhage in a health facility',
                        person_id)
            logger.info('%s|antepartum_haemorrhage|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})

        random = self.module.rng.random_sample(size=1)
        if random < mni[person_id]['risk_ip_sepsis']:
            df.at[person_id, 'la_sepsis'] = True
            df.at[person_id, 'la_sepsis_disability'] = True
            mni[person_id]['sepsis_ip'] = True

            logger.debug('person %d has developed maternal sepsis in a health facility',
                        person_id)
            logger.info('%s|maternal_sepsis|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})
            # TODO modify newborn risk of sepsis for septic women

        random = self.module.rng.random_sample(size=1)
        if random < mni[person_id]['risk_ur']:
            mni[person_id]['uterine_rupture'] = True
            df.at[person_id, 'la_uterine_rupture'] = True
            df.at[person_id, 'la_uterine_rupture_disab'] = True

            logger.debug('person %d is experiencing a uterine rupture in a health facility',
                        person_id)

            logger.info('%s|uterine_rupture|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})

        # DUMMY ... (risk factors)
        random = self.module.rng.random_sample(size=1)
        if random < params['prob_cord_prolapse']:
            mni[person_id]['cord_prolapse'] = True

# ==================================== SCHEDULE HEALTH SYSTEM INTERACTIONS ===========================================
    # Here, if a woman has developed a complication, she is scheduled to receive any care she may need

        if mni[person_id]['labour_is_currently_obstructed']:
            logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: scheduling immediate additional '
                        'treatment for obstructed labour for person %d', person_id)

            event = HSI_Labour_ReceivesCareForObstructedLabour(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1)) # TODO: check all these offsets

        if mni[person_id]['sepsis_ip']:
            logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: scheduling immediate additional '
                        'treatment for maternal sepsis during labour for person %d', person_id)

            event = HSI_Labour_ReceivesCareForMaternalSepsis(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        if mni[person_id]['antepartum_haem']:
            logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: scheduling immediate additional '
                        'treatment for antepartum haemorrhage during labour for person %d', person_id)

            event = HSI_Labour_ReceivesCareForMaternalHaemorrhage(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        if mni[person_id]['eclampsia_ip']:
            logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: scheduling immediate additional '
                        'treatment for eclampsia during labour for person %d', person_id)

            event = HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        if mni[person_id]['uterine_rupture']:
            logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: scheduling immediate additional '
                        'treatment for uterine rupture during labour for person %d', person_id)

            event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1)
                                                                )

        if df.at[person_id, 'la_current_labour_successful_induction'] == 'failed_induction':
            logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: scheduling a caesarean section'
                        ' for person %d', person_id)

            event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT

        if mni[person_id]['uterine_rupture'] or mni[person_id]['eclampsia_ip'] or mni[person_id]['antepartum_haem'] or\
            mni[person_id]['sepsis_ip'] or mni[person_id]['labour_is_currently_obstructed']:
            actual_appt_footprint['NormalDelivery'] = actual_appt_footprint['CompDelivery'] * 1

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_PresentsForSkilledAttendanceInLabour: did not run')
        return False
        # The event wont run if a.) not allowed in policy set up b.)there isn't the available officer time
        # TODO: Here, if false is returned then if the event does not run it is taken out of the queue. In this instance
        #  we need to decide what these women would do (its unlikely they would leave)- what this essentially means is
        #  there is no SBA availble to assist in their delivery, so they will deliver unaided (may get some benifits of
        #  being in the vacinity of a facility(

        # TODO: Can we then schedule like a post-attempted facilty delivery event?

        pass


class HSI_Labour_ReceivesCareForObstructedLabour(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It manages the treatment of obstructed labour and referral in the instance of failed treatment
    """

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
            'Item_Code': [],
        }

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

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
            treatment_effect = params['prob_deliver_ventouse']
            random = self.module.rng.random_sample(size=1)
            if treatment_effect > random:
                # df.at[person_id, 'la_obstructed_labour'] = False
                mni[person_id]['labour_is_currently_obstructed'] = False
                mni[person_id]['mode_of_delivery'] = 'AVDV'
                # add here effect of antibiotics?
            else:
                # If the vacuum is unsuccessful we apply the probability of successful forceps delivery
                treatment_effect = params['prob_deliver_forceps']
                random = self.module.rng.random_sample(size=1)
                if treatment_effect > random:
                    # df.at[person_id, 'la_obstructed_labour'] = False
                    mni[person_id]['labour_is_currently_obstructed'] = False
                    mni[person_id]['mode_of_delivery'] = 'AVDF'
                    # add here effect of antibiotitcs?

                # Finally if assisted vaginal delivery fails then CS is scheduled
                else:
                    event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                        priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=14)
                                                                        )

                    logger.info(
                        'This is HSI_Labour_ReceivesCareForObstructedLabour referring person %d for a caesarean section'
                        ' following unsuccessful attempts at assisted vaginal delivery ',
                        person_id)

                    # Symphysiotomy??? Does this happen in malawi

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        #    actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_PresentsForInductionOfLabour: did not run')
        pass


class HSI_Labour_ReceivesCareForMaternalSepsis(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It manages the
    """

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

        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code_sepsis: 1}],
            'Item_Code': [],
        }

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

        # TODO: consider 1st line/2nd line and efficacy etc etc

        # Treatment can only be delivered if appropriate antibiotics are available, if they're not the woman isn't
        # treated
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_sepsis]:
            logger.debug('pkg_code_sepsis is available, so use it.')
            treatment_effect = params['prob_cure_antibiotics']
            random = self.module.rng.random_sample(size=1)
            if treatment_effect > random:
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
    """
    This is a Health System Interaction Event.
    It manages the
        """

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
            consumables.loc[consumables['Items'] == 'nifedipine retard 20 mg_100_IDA',
                            'Item_Code']
        )[0]
        item_code_hz = pd.unique(
            consumables.loc[consumables['Items'] == 'Hydralazine hydrochloride 20mg/ml, 1ml_each_CMST',
                            'Item_Code']
        )[0]

        # TODO: Methyldopa not being recognised?
        # item_code_md = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Methyldopa 250mg_1000_CMS',
        #                    'Item_Code']
        # )[0]

        item_code_hs = pd.unique(
            consumables.loc[consumables['Items'] ==  "Ringer's lactate (Hartmann's solution), 500 ml_20_IDA",
                            'Item_Code']
        )[0]

        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code_eclampsia: 1}],
            'Item_Code': [{item_code_nf: 2}, {item_code_hz:  2}, {item_code_hs: 1}],
        }

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

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
        if mni[person_id]['eclampsia_ip'] or mni[person_id]['eclampsia_pp']:
            treatment_effect = params['prob_cure_mgso4']
            random = self.module.rng.random_sample()
            if treatment_effect > random:
                mni[person_id]['eclampsia'] = False
            else:
                random = self.module.rng.random_sample()
                if treatment_effect > random:
                    mni[person_id]['eclampsia'] = False
                else:
                    treatment_effect = params['prob_cure_diazepam']
                    random = self.module.rng.random_sample()
                    if treatment_effect > random:
                        mni[person_id]['eclampsia'] = False
                    else:
                        event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                            priority=0,
                                                                            topen=self.sim.date,
                                                                            tclose=self.sim.date + DateOffset(days=14)
                                                                            )

                        logger.info(
                            'This is HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy referring person %d '
                            'for a caesarean section'
                            ' following uncontrolled eclampsia',
                            person_id)

        # Guidelines suggest assisting with second stage of labour via AVD - consider including this?
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        #    actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_ReceivesCareForMaternalSepsis: did not run')
        pass


class HSI_Labour_ReceivesCareForMaternalHaemorrhage(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It manages the
    """

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
            'Item_Code': [{item_code_aph1: 2}, {item_code_aph2: 1}, {item_code_aph3: 1}, {item_code_aph4: 1},],
        }

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

        # TODO: Again determine how to use outcome of consumable request

# ===================================  ANTEPARTUM HAEMORRHAGE TREATMENT ===============================================
        # TODO: consider severity grading?

        # Here we determine the etiology of the bleed, which will determine treatment algorithm
        etiology = ['PA', 'PP']  # May need to move this to allow for risk factors?
        probabilities = [0.67, 0.33]  # DUMMY
        random_choice = self.module.rng.choice(etiology, size=1, p=probabilities)
        # Todo: below line isnt storing correctly so effects parsing of log file
#        mni[person_id]['source_aph'] = random_choice  # Storing as high chance of SB in severe placental abruption
        # TODO: Needs to be dependent on blood availability and establish how we're quantifying effect
        mni[person_id]['units_transfused'] = 2

        event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
        self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                            priority=0,
                                                            topen=self.sim.date,
                                                            tclose=self.sim.date + DateOffset(days=1)
                                                            )

# ======================================= POSTPARTUM HAEMORRHAGE ======================================================
        # First we use a probability weighted random draw to determine the underlying etiology of this womans PPH

        if mni[person_id]['postpartum_haem']:
            etiology = ['UA', 'RPP']
            probabilities = [0.67, 0.33]  # dummy
            random_choice = self.module.rng.choice(etiology, size=1, p=probabilities)
            # Todo: below line isnt storing correctly so effects parsing of log file
            #            mni[person_id]['source_pph'] = random_choice
            # Todo: add a level of severity of PPH -yes

            # ======================== TREATMENT CASCADE FOR ATONIC UTERUS:=============================================

            # Here we use a treatment cascade adapted from Malawian Obs/Gynae guidelines
            # Women who are bleeding due to atonic uterus first undergo medical management, oxytocin IV, misoprostol PR
            # and uterine massage in an attempt to stop bleeding

            if mni[person_id]['source_pph'] == 'UA':
                random = self.module.rng.random_sample(size=1)
                if params['prob_cure_oxytocin'] > random:
                    mni[person_id]['postpartum_haem'] = False
                else:
                    random = self.module.rng.random_sample(size=1)
                    if params['prob_cure_misoprostol'] > random:
                        mni[person_id]['postpartum_haem'] = False
                    else:
                        random = self.module.rng.random_sample(size=1)
                        if params['prob_cure_uterine_massage'] > random:
                            mni[person_id]['postpartum_haem'] = False
                        else:
                            random = self.module.rng.random_sample(size=1)
                            if params['prob_cure_uterine_tamponade'] > random:
                                mni[person_id]['postpartum_haem'] = False

            # Todo: consider the impact of oxy + miso + massage as ONE value, Discuss with expert

            # ===================TREATMENT CASCADE FOR RETAINED PRODUCTS/PLACENTA:====================================
            if mni[person_id]['source_pph'] == 'RPP':
                random = self.module.rng.random_sample(size=1)
                if params['prob_cure_manual_removal'] > random:
                    mni[person_id]['postpartum_haem'] = False
                    # blood?

            # In the instance of uncontrolled bleeding a woman is referred on for surgical care
            if mni[person_id]['postpartum_haem']:
                event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1)
                                                                    )

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
        self.ACCEPTED_FACILITY_LEVEL = 2 # check this?
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
            'Intervention_Package_Code': [{pkg_code_am: 1}],
            'Item_Code': [],
        }

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
            adjusted_maternal_pph_risk = mni[person_id]['risk_pp_postpartum_haem'] * params['rr_pph_amtsl']
            mni[person_id]['risk_pp_postpartum_haem'] = adjusted_maternal_pph_risk
        else:
            logger.debug('pkg_code_am is not available, so can' 't use it.')
            logger.debug('woman %d did not receive active managment of the third stage of labour due to resource '
                         'constraints')
    # ===============================  POSTPARTUM COMPLICATIONS ========================================================

        # TODO: link eclampsia/sepsis diagnosis in SBA and PPC

        # As with the SkilledBirthAttendance HSI we recalcualte risk of complications in light of preventative
        # interventions
        random = self.module.rng.random_sample(size=1)
        if random < mni[person_id]['risk_pp_eclampsia']:
            df.at[person_id, 'la_eclampsia'] = True
            df.at[person_id, 'la_eclampsia_disability'] = True
            mni[person_id]['eclampsia_pp'] = True

            logger.debug('person %d is experiencing eclampsia in a health facility following birth',
                        person_id)
            logger.info('%s|eclampsia|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})

        if random < mni[person_id]['risk_pp_postpartum_haem']: # increased liklihood of PPH based on whats happened so far? (APH, CS)
            df.at[person_id, 'la_postpartum_haem'] = True
            df.at[person_id, 'la_haemorrhage_disability'] = True
            mni[person_id]['postpartum_haem'] = True

            logger.debug('person %d is experiencing an postpartum haemorrhage in a health facility following birth',
                        person_id)
            logger.info('%s|postpartum_haemorrhage|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})

        random = self.module.rng.random_sample(size=1)
        if random < mni[person_id]['risk_pp_sepsis']:
            df.at[person_id, 'la_sepsis'] = True
            df.at[person_id, 'la_sepsis_disability'] = True
            mni[person_id]['sepsis_pp'] = True

            logger.debug('person %d has developed maternal sepsis in a health facility following delivery',
                        person_id)
            logger.info('%s|maternal_sepsis|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})

        # =============================  SCHEDULING ADDITIONAL TREATMENT ==============================================

        if mni[person_id]['sepsis_pp']:
            logger.info('This is HSI_Labour_ReceivesCareForPostpartumPeriod: scheduling immediate additional '
                        'treatment for maternal sepsis during the postpartum period for person %d', person_id)

            event = HSI_Labour_ReceivesCareForMaternalSepsis(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                        priority=0,
                                                        topen=self.sim.date,
                                                        tclose=self.sim.date + DateOffset(days=1)
                                                        )

        if mni[person_id]['postpartum_haem']:
            logger.info('This is HSI_Labour_ReceivesCareForPostpartumPeriod: scheduling immediate additional '
                        'treatment for antepartum haemorrhage during the postpartum period for person %d', person_id)

            event = HSI_Labour_ReceivesCareForMaternalHaemorrhage(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1)
                                                                    )

        if mni[person_id]['eclampsia_pp']:
            logger.info('This is HSI_Labour_ReceivesCareForPostpartumPeriod: scheduling immediate additional '
                        'treatment for eclampsia during the postpartum period for person %d', person_id)

            event = HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1)
                                                                    )

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT #  TODO: modify based on complications?

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_ReceivesCareForPostpartumPeriod: did not run')
        pass

class HSI_Labour_ReferredForSurgicalCareInLabour(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This event manages the Health System Interaction for a woman who needs to be referred to undergo an emergency
    surgical management of complications arising in labour, in the postpartum period or for caesarean section
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module,Labour)

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
            'Item_Code': [],
        }

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )
        # TODO: consumables

        # pkg_code_uterine_repair
        # pkg_code_pph_surgery
        # +/- hysterectomy?

# ====================================== EMERGENCY CAESAREAN SECTION ==================================================

        if (mni[person_id]['uterine_rupture']) or (mni[person_id]['antepartum_haem']) or (mni[person_id]['eclampsia_ip']) or\
            (mni[person_id]['labour_is_currently_obstructed']) or \
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
            random = self.module.rng.random_sample(size=1)
            if params['prob_cure_uterine_repair'] > random:
                # df.at[person_id, 'la_uterine_rupture'] = False
                mni[person_id]['uterine_rupture'] = False

        # In the instance of failed surgical repair, the woman undergoes a hysterectomy
            else:
                random = self.module.rng.random_sample(size=1)
                if params['prob_cure_hysterectomy'] > random:
                    # df.at[person_id, 'la_uterine_rupture'] = False
                    mni[person_id]['uterine_rupture'] = False

# ================================== SURGERY FOR UNCONTROLLED POSTPARTUM HAEMORRHAGE ==================================

        # If a woman has be referred for surgery for uncontrolled post partum bleeding we use the treatment alogrith to
        # determine if her bleeding can be controlled surgically
        if mni[person_id]['postpartum_haem']:
            random = self.module.rng.random_sample(size=1)
            if params['prob_cure_uterine_ligation'] > random:
                mni[person_id]['postpartum_haem'] = False
            else:
                random = self.module.rng.random_sample(size=1)
                if params['prob_cure_b_lynch'] > random:
                    mni[person_id]['postpartum_haem'] = False
                else:
                    random = self.module.rng.random_sample(size=1)
                    # Todo: similarly consider bunching surgical interventions
                    if params['prob_cure_hysterectomy'] > random:
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
