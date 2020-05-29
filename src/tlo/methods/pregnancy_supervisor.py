from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.methods import demography
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PregnancySupervisor(Module):
    """This module is responsible for supervision of pregnancy in the population including incidence of ectopic
    pregnancy, multiple pregnancy, spontaneous abortion, induced abortion, and onset of antenatal complications. This
    module is incomplete, currently antenatal death has not been coded. Similarly antenatal care seeking will be house
    hear, for  both routine treatment and in emergencies"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        self.PregnancyDiseaseTracker = dict()

    PARAMETERS = {
        'prob_ectopic_pregnancy': Parameter(
            Types.REAL, 'probability of a womans pregnancy being ectopic and not uterine implantation'),
        'prob_multiples': Parameter(
            Types.REAL, 'probability that a woman is currently carrying more than one pregnancy'),
        'prob_spontaneous_abortion_per_month': Parameter(
            Types.REAL, 'underlying risk of spontaneous_abortion per month without the impact of risk factors'),
        'prob_induced_abortion_per_month': Parameter(
            Types.REAL, 'underlying risk of induced abortion per month without the impact of risk factors'),
        'prob_pre_eclampsia_per_month': Parameter(
            Types.REAL, 'underlying risk of pre-eclampsia per month without the impact of risk factors'),
        'prob_gest_htn_per_month': Parameter(
            Types.REAL, 'underlying risk of gestational hypertension per month without the impact of risk factors'),
        'prob_gest_diab_per_month': Parameter(
            Types.REAL, 'underlying risk of gestational diabetes per month without the impact of risk factors'),
        'prob_still_birth_per_month': Parameter(
            Types.REAL, 'underlying risk of stillbirth per month without the impact of risk factors'),
        'prob_antenatal_death_per_month': Parameter(
            Types.REAL, 'underlying risk of antenatal maternal death per month without the impact of risk factors'),
        'prob_ectopic_pregnancy_death': Parameter(
            Types.REAL, 'probability of a woman dying from a ruptured ectopic pregnancy'),
        'prob_induced_abortion_type': Parameter(
            Types.LIST, 'probabilities that the type of abortion a woman has will be 1.) Surgical or 2.) Medical'),
        'prob_haemorrhage_spontaneous_abortion': Parameter(
            Types.REAL, 'probability that a woman who has undergone a spontaneous abortion will experience haemorrhage '
                        'as a complication'),
        'prob_sepsis_spontaneous_abortion': Parameter(
            Types.REAL, 'probability that a woman who has undergone a spontaneous abortion will experience sepsis '
                        'as a complication'),
        'prob_haemorrhage_induced_abortion': Parameter(
            Types.REAL, 'probability that a woman who has undergone an induced abortion will experience haemorrhage '
                        'as a complication'),
        'prob_sepsis_induced_abortion': Parameter(
            Types.REAL, 'probability that a woman who has undergone an induced abortion will experience sepsis '
                        'as a complication'),
        'prob_injury_induced_abortion': Parameter(
            Types.REAL, 'probability that a woman who has undergone an induced abortion will experience injury '
                        'as a complication'),
        'prob_induced_abortion_death': Parameter(
            Types.REAL, 'underlying risk of death following an induced abortion'),
        'prob_spontaneous_abortion_death': Parameter(
            Types.REAL, 'underlying risk of death following an spontaneous abortion'),
    }

    PROPERTIES = {
        'ps_gestational_age_in_weeks': Property(Types.INT, 'current gestational age, in weeks, of this womans '
                                                           'pregnancy'),
        'ps_ectopic_pregnancy': Property(Types.BOOL, 'Whether this womans pregnancy is ectopic'),
        'ps_ectopic_symptoms': Property(Types.CATEGORICAL, 'Level of symptoms for ectopic pregnancy',
                                        categories=['none', 'abdominal pain', 'abdominal pain plus bleeding', 'shock']),
        'ps_ep_unified_symptom_code': Property(
            Types.CATEGORICAL,
            'Level of symptoms on the standardised scale (governing health-care seeking): '
            '0=None; 1=Mild; 2=Moderate; 3=Severe; 4=Extreme_Emergency',
            categories=[0, 1, 2, 3, 4]),
        'ps_multiple_pregnancy': Property(Types.BOOL, 'Whether this womans is pregnant with multiple fetuses'),
        'ps_induced_abortion_complication': Property(Types.LIST, 'List of any complications a woman has experience '
                                                                 'following an induced abortion'),
        'ps_spontaneous_abortion_complication': Property(Types.LIST, 'List of any complications a woman has experience '
                                                                     'following an spontaneous abortion'),
        'ps_antepartum_still_birth': Property(Types.BOOL, 'whether this woman has experienced an antepartum still birth'
                                                          'of her current pregnancy'),
        'ps_previous_stillbirth': Property(Types.BOOL, 'whether this woman has had any previous pregnancies end in '
                                                       'still birth'),  # consider if this should be an interger
        'ps_gestational_htn': Property(Types.BOOL, 'whether this woman has gestational hypertension'),
        'ps_mild_pre_eclamp': Property(Types.BOOL, 'whether this woman has mild pre-eclampsia'),
        'ps_severe_pre_eclamp': Property(Types.BOOL, 'whether this woman has severe pre-eclampsia'),
        'ps_prev_pre_eclamp': Property(Types.BOOL, 'whether this woman has experienced pre-eclampsia in a previous '
                                                   'pregnancy'),
        'ps_currently_hypertensive': Property(Types.BOOL, 'whether this woman is currently hypertensive'),
        'ps_gest_diab': Property(Types.BOOL, 'whether this woman has gestational diabetes'),
        'ps_prev_gest_diab': Property(Types.BOOL, 'whether this woman has ever suffered from gestational diabetes '
                                                  'during a previous pregnancy'),
        'ps_premature_rupture_of_membranes': Property(Types.BOOL, 'whether this woman has experience rupture of '
                                                                  'membranes before the onset of labour. If this is '
                                                                  '<37 weeks from gestation the woman has preterm '
                                                                  'premature rupture of membranes'),
    }

    def read_parameters(self, data_folder):
        params = self.parameters
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_PregnancySupervisor.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)

        if 'HealthBurden' in self.sim.modules.keys():
            params['daly_wt_abortive_outcome'] = self.sim.modules['HealthBurden'].get_daly_weight(352)

    # ==================================== LINEAR MODEL EQUATIONS =====================================================
            # All linear equations used in this module are stored within the ps_linear_equations parameter below

            params['ps_linear_equations'] = {
                'ectopic': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_ectopic_pregnancy']),

                'multiples': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_multiples']),

                'spontaneous_abortion': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_spontaneous_abortion_per_month']),

                'induced_abortion': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_induced_abortion_per_month']),

                'pre_eclampsia': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_pre_eclampsia_per_month']),

                'gest_htn': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_gest_htn_per_month']),

                'gest_diab': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_gest_diab_per_month']),

                'antenatal_stillbirth': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_still_birth_per_month']),

                'antenatal_death': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_antenatal_death_per_month']),

                'ectopic_pregnancy_death': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_ectopic_pregnancy_death']),

                'induced_abortion_death': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_induced_abortion_death']),

                'spontaneous_abortion_death': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_spontaneous_abortion_death']),

            }

    def initialise_population(self, population):

        df = population.props

        df.loc[df.is_alive, 'ps_gestational_age_in_weeks'] = 0
        df.loc[df.is_alive, 'ps_ectopic_pregnancy'] = False
        df.loc[df.is_alive, 'ps_ectopic_symptoms'].values[:] = 'none'
        df.loc[df.is_alive, 'ps_ep_unified_symptom_code'] = 0
        df.loc[df.is_alive, 'ps_multiple_pregnancy'] = False
        df.loc[df.is_alive, 'ps_induced_abortion_complication'] = 'none'
        df.loc[df.is_alive, 'ps_spontaneous_abortion_complication'] = 'none'
        df.loc[df.is_alive, 'ps_antepartum_still_birth'] = False
        df.loc[df.is_alive, 'ps_previous_stillbirth'] = False
        df.loc[df.is_alive, 'ps_gestational_htn'] = False
        df.loc[df.is_alive, 'ps_mild_pre_eclamp'] = False
        df.loc[df.is_alive, 'ps_severe_pre_eclamp'] = False
        df.loc[df.is_alive, 'ps_prev_pre_eclamp'] = False
        df.loc[df.is_alive, 'ps_currently_hypertensive'] = False
        df.loc[df.is_alive, 'ps_gest_diab'] = False
        df.loc[df.is_alive, 'ps_prev_gest_diab'] = False
        df.loc[df.is_alive, 'ps_premature_rupture_of_membranes'] = False

    def initialise_simulation(self, sim):
        """Get ready for simulation start.
        """
        event = PregnancySupervisorEvent
        sim.schedule_event(event(self),
                           sim.date + DateOffset(days=0))

        event = PregnancyDiseaseProgressionEvent
        sim.schedule_event(event(self),
                           sim.date + DateOffset(days=0))

        event = PregnancyLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=364))

        self.sim.modules['HealthSystem'].register_disease_module(self)

        self.PregnancyDiseaseTracker = {'ectopic_pregnancy': 0, 'induced_abortion': 0, 'spontaneous_abortion': 0}

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props

        df.at[child_id, 'ps_gestational_age_in_weeks'] = 0
        df.at[child_id, 'ps_ectopic_pregnancy'] = False
        df.at[child_id, 'ps_ectopic_symptoms'] = 'none'
        df.at[child_id, 'ps_ep_unified_symptom_code'] = 0
        df.at[child_id, 'ps_multiple_pregnancy'] = False
        df.at[child_id, 'ps_induced_abortion_complication'] = 'none'
        df.at[child_id, 'ps_spontaneous_abortion_complication'] = 'none'
        df.at[child_id, 'ps_antepartum_still_birth'] = False
        df.at[child_id, 'ps_previous_stillbirth'] = False
        df.at[child_id, 'ps_gestational_htn'] = False
        df.at[child_id, 'ps_mild_pre_eclamp'] = False
        df.at[child_id, 'ps_severe_pre_eclamp'] = False
        df.at[child_id, 'ps_prev_pre_eclamp'] = False
        df.at[child_id, 'ps_currently_hypertensive'] = False
        df.at[child_id, 'ps_gest_diab'] = False
        df.at[child_id, 'ps_prev_gest_diab'] = False
        df.at[child_id, 'ps_premature_rupture_of_membranes'] = False

        df.at[mother_id, 'ps_gestational_age_in_weeks'] = 0

    def on_hsi_alert(self, person_id, treatment_id):
        logger.debug('This is PregnancySupervisor, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):
        df = self.sim.population.props

        # TODO: Dummy code, waiting for new DALY set up
        logger.debug('This is PregnancySupervisor reporting my health values')

        health_values_1 = df.loc[df.is_alive, 'ps_ectopic_pregnancy'].map(
            {False: 0, True: 0.2})
        health_values_1.name = 'Ectopic Pregnancy'

        health_values_df = health_values_1
        return health_values_df

    def set_pregnancy_complications(self, index, complication):
        """This function is called from within the PregnancySupervisorEvent. It calculates risk of a number of pregnancy
        outcomes/ complications for pregnant women in the data frame using the linear model equations defined above.
        Properties are modified depending on the  complication passed to the function and the result of a random draw"""
        df = self.sim.population.props
        params = self.parameters

        # TODO: we should run some checks on the indexes being passed to this function (right women, right timing?)
        # for person in index:
        #    assert df.at[person, 'is_alive']
        #    assert df.at[person, 'is_pregnant']
        #    assert df.at[person, 'sex'] == 'F'
        #    assert df.at[person, 'age_years'] > 14
        #    assert df.at[person, 'age_years'] < 50

        # We apply the results of the linear model to the index of women in question
        result = params['ps_linear_equations'][f'{complication}'].predict(index)

        # And use the result of a random draw to determine which women will experience the complication
        random_draw = pd.Series(self.rng.random_sample(size=len(index)), index=index.index)
        temp_df = pd.concat([result, random_draw], axis=1)
        temp_df.columns = ['result', 'random_draw']

        # Then we use this index to make changes to the data frame and schedule any events required
        positive_index = temp_df.index[temp_df.random_draw < temp_df.result]

        if complication == 'ectopic':
            df.loc[positive_index, 'ps_ectopic_pregnancy'] = True
            self.PregnancyDiseaseTracker['ectopic_pregnancy'] += len(positive_index)
            for person in positive_index:
                self.sim.schedule_event(EctopicPregnancyRuptureEvent(self, person),
                                        (self.sim.date + pd.Timedelta(days=7 * 5 + self.rng.randint(0, 7 * 4))))

            if not positive_index.empty:
                logger.debug(f'The following women have experience an ectopic pregnancy,{positive_index}')

        if complication == 'multiples':
            df.loc[positive_index, 'ps_multiple_pregnancy'] = True

        if complication == 'spontaneous_abortion' or complication == 'induced_abortion':
            df.loc[positive_index, 'is_pregnant'] = False
            df.loc[positive_index, 'la_due_date_current_pregnancy'] = pd.NaT
            df.loc[positive_index, 'ps_gestational_age_in_weeks'] = 0
            for person in positive_index:
                self.sim.schedule_event(AbortionEvent(self, person, cause=f'{complication}'), self.sim.date)
            if not positive_index.empty:
                    logger.debug(f'The following women have experienced an abortion,{positive_index}')

        if complication == 'pre_eclampsia':
            df.loc[positive_index, 'ps_mild_pre_eclamp'] = True
            df.loc[positive_index, 'ps_prev_pre_eclamp'] = True
            if not positive_index.empty:
                logger.debug(f'The following women have developed pre_eclampsia,{positive_index}')

        if complication == 'gest_htn':
            df.loc[positive_index, 'ps_gestational_htn'] = True
            if not positive_index.empty:
                logger.debug(f'The following women have developed gestational hypertension,{positive_index}')

        if complication == 'gest_diab':
            df.loc[positive_index, 'ps_gest_diab'] = True
            df.loc[positive_index, 'ps_prev_gest_diab'] = True
            if not positive_index.empty:
                logger.debug(f'The following women have developed gestational diabetes,{positive_index}')

        if complication == 'antenatal_death':
            for person in positive_index:
                death = demography.InstantaneousDeath(self.sim.modules['Demography'], person,
                                                      cause='antenatal death')
                self.sim.schedule_event(death, self.sim.date)
            if not positive_index.empty:
                logger.debug(f'The following women have died during pregnancy,{positive_index}')

        if complication == 'antenatal_stillbirth':
            df.loc[positive_index, 'ps_antepartum_still_birth'] = True
            df.loc[positive_index, 'ps_previous_stillbirth'] = True
            df.loc[positive_index, 'is_pregnant'] = False
            df.loc[positive_index, 'la_due_date_current_pregnancy'] = pd.NaT
            df.loc[positive_index, 'ps_gestational_age_in_weeks'] = 0
            if not positive_index.empty:
                logger.debug(f'The following women have have experienced an antepartum stillbirth,{positive_index}')

        # TODO: consider using return function to produce negative index which could be used for the chain of functions

    def set_abortion_complications(self, individual_id, abortion_type):
        df = self.sim.population.props
        params = self.parameters

        # complications are appended into a list to allow multiple comps from each abortion
    #    if abortion_type == 'spontaneous_abortion':
    #        if params['prob_haemorrhage_spontaneous_abortion'] < self.rng.random_sample():
    #            df.at[individual_id, 'ps_spontaneous_abortion_complication'].append('haemorrhage')
    #        if params['prob_sepsis_spontaneous_abortion'] < self.rng.random_sample():
    #            df.at[individual_id, 'ps_spontaneous_abortion_complication'].append('sepsis')

        # TODO surgical vs medical induced abortions
    #    if abortion_type == 'induced_abortion':
    #        if params['prob_haemorrhage_induced_abortion'] < self.rng.random_sample():
    #            df.at[individual_id, 'ps_induced_abortion_complication'].append('haemorrhage')
    #        elif params['prob_sepsis_induced_abortion'] < self.rng.random_sample():
    #            df.at[individual_id, 'ps_induced_abortion_complication'].append('sepsis')
    #        elif params['prob_injury_induced_abortion'] < self.rng.random_sample():
    #            df.at[individual_id, 'ps_induced_abortion_complication'].append('sepsis')
    #        else:
    #            df.at[individual_id, 'ps_induced_abortion_complication'].append('none')

        # TODO: reset these complications after a certain time period


class PregnancySupervisorEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is the PregnancySupervisorEvent. It runs weekly. It updates gestational age of pregnancy in weeks.
    Presently this event has been hollowed out, additionally it will and uses set_pregnancy_complications function to
    determine if women will experience complication. This event is incomplete and will eventually apply risk of
     antenatal death and handle antenatal care seeking. """

    def __init__(self, module, ):
        super().__init__(module, frequency=DateOffset(weeks=1))

    def apply(self, population):
        df = population.props
        params = self.module.parameters

        # ===================================== UPDATING GESTATIONAL AGE IN WEEKS  ====================================
        # Here we update the gestational age in weeks of all currently pregnant women in the simulation
        alive_and_preg = df.is_alive & df.is_pregnant
        gestation_in_days = self.sim.date - df.loc[alive_and_preg, 'date_of_last_pregnancy']
        gestation_in_weeks = gestation_in_days / np.timedelta64(1, 'W')

        df.loc[alive_and_preg, 'ps_gestational_age_in_weeks'] = gestation_in_weeks.astype('int64')
        logger.debug('updating gestational ages on date %s', self.sim.date)

        # ========================PREGNANCY COMPLICATIONS - ECTOPIC PREGNANCY & MULTIPLES =============================
        # Here we use the set_pregnancy_complications function to calculate each womans risk of ectopic pregnancy,
        # conduct a draw and edit relevant properties defined above

        newly_pregnant_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 1)]
        self.module.set_pregnancy_complications(newly_pregnant_idx, 'ectopic')

        # For women who don't experience and ectopic pregnancy we use the same function to assess risk of multiple
        # pregnancy
        np_no_ectopic = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 1) &
                               ~df.ps_ectopic_pregnancy]
        self.module.set_pregnancy_complications(np_no_ectopic, 'multiples')
        # TODO: Review the necessity of including multiple pregnancies

        # ==================================== MONTH 1 PREGNANCY RISKS ================================================
        # Here we look at all the women who have reached one month gestation and apply the risk of early pregnancy loss

        month_1_idx = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks
                                                                                        == 4)]
        self.module.set_pregnancy_complications(month_1_idx, 'spontaneous_abortion')

        # TODO: I need to think if this method of creating new indexes is the best way of doing things,
        #  seems error prone

        month_1_no_spontaneous_abortion = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                                                 (df.ps_gestational_age_in_weeks == 4)]

        # TODO: also do i need to apply risk of death in the months where only pregnancy loss occurs? (tb, hiv etc?)
        # death
        self.module.set_pregnancy_complications(month_1_no_spontaneous_abortion, 'antenatal_death')

        # ========================================== MONTH 2 PREGNANCY RISKS ==========================================
        # Now we use the set_pregnancy_complications function to calculate risk and set properties for women whose
        # pregnancy is not ectopic

        # spontaneous_abortion:
        month_2_idx = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks
                                                                                        == 8)]
        self.module.set_pregnancy_complications(month_2_idx, 'spontaneous_abortion')

        # Here we use the an index of women who will not miscarry to determine who will seek an abortion
        # induced_abortion:
        month_2_no_spontaneous_abortion = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                                                 (df.ps_gestational_age_in_weeks == 8)]

        # TODO: need to know the incidence of induced abortion for GA 8 weeks to determine if its worth while
        self.module.set_pregnancy_complications(month_2_no_spontaneous_abortion, 'induced_abortion')

        month_2_no_induced_abortion = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                                             (df.ps_gestational_age_in_weeks == 8)]
        # death
        self.module.set_pregnancy_complications(month_2_no_induced_abortion, 'antenatal_death')

        # ======================================= MONTH 3 PREGNANCY RISKS ============================================
        month_3_idx = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks
                                                                                        == 13)]
        # spontaneous_abortion
        self.module.set_pregnancy_complications(month_3_idx, 'spontaneous_abortion')

        # induced_abortion:
        month_3_no_spontaneous_abortion = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                                                 (df.ps_gestational_age_in_weeks == 13)]
        self.module.set_pregnancy_complications(month_3_no_spontaneous_abortion, 'induced_abortion')

        month_3_no_spontaneous_abortion = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                                                 (df.ps_gestational_age_in_weeks == 13)]
        # death
        self.module.set_pregnancy_complications(month_3_no_spontaneous_abortion, 'antenatal_death')

        # ================================= MONTH 4 PREGNANCY RISKS ==================================================
        month_4_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 17)]
        # spontaneous_abortion
        self.module.set_pregnancy_complications(month_4_idx, 'spontaneous_abortion')

        month_4_no_spontaneous_abortion = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 17)]

        # induced_abortion:
        self.module.set_pregnancy_complications(month_4_no_spontaneous_abortion, 'induced_abortion')

        month_4_no_induced_abortion = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 17)]

        # death
        self.module.set_pregnancy_complications(month_4_no_induced_abortion, 'antenatal_death')

        # ==================================== MONTH 5 PREGNANCY RISKS ===============================================
        month_5_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22)]
        # spontaneous_abortion
        self.module.set_pregnancy_complications(month_5_idx, 'spontaneous_abortion')

        # induced_abortion:
        month_5_no_spontaneous_abortion = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22)]
        self.module.set_pregnancy_complications(month_5_no_spontaneous_abortion, 'induced_abortion')

        # Here we begin to apply the risk of developing complications which present later in pregnancy including
        # pre-eclampsia, gestational hypertension and gestational diabetes

        # PRE-ECLAMPSIA
        # Only women without pre-existing hypertensive disorders of pregnancy are can develop the disease now
        month_5_preg_continues = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22)]
        self.module.set_pregnancy_complications(month_5_preg_continues, 'pre_eclampsia')

        month_5_no_pe = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22) &
                               ~df.ps_mild_pre_eclamp & ~df.ps_severe_pre_eclamp]

        # GESTATIONAL HYPERTENSION
        # Similarly only those women who dont develop pre-eclampsia are able to develop gestational hypertension
        self.module.set_pregnancy_complications(month_5_no_pe, 'gest_htn')

        # GESTATIONAL DIABETES
        self.module.set_pregnancy_complications(month_5_preg_continues, 'gest_diab')
        # TODO: Exclude current diabetics
        # TODO: review lit in regards to onset date and potentially move this to earlier

        # death
        self.module.set_pregnancy_complications(month_5_preg_continues, 'antenatal_death')

        # =========================== MONTH 6 RISK APPLICATION =======================================================
        # TODO: should this be 28 weeks to align with still birth definition
        # From month 6 it is possible women could be in labour at the time of this event so we exclude them
        month_6_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27) &
                             ~df.la_currently_in_labour]

        # STILL BIRTH RISK
        self.module.set_pregnancy_complications(month_6_idx, 'stillbirth')

        month_6_preg_continues = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27) &
                                        ~df.la_currently_in_labour]

        # PRE ECLAMPSIA
        self.module.set_pregnancy_complications(month_6_preg_continues, 'pre_eclampsia')

        month_6_no_pe = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27) &
                               ~df.ps_mild_pre_eclamp & ~df.ps_severe_pre_eclamp]

        # GESTATIONAL HYPERTENSION
        self.module.set_pregnancy_complications(month_6_no_pe, 'gest_htn')

        # GESTATIONAL DIABETES
        self.module.set_pregnancy_complications(month_6_preg_continues, 'gest_diab')

        # death
        self.module.set_pregnancy_complications(month_6_preg_continues, 'antenatal_death')

        # =========================== MONTH 7 RISK APPLICATION =======================================================
        month_7_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) &
                             ~df.la_currently_in_labour]

        # STILL BIRTH RISK
        self.module.set_pregnancy_complications(month_7_idx, 'stillbirth')

        month_7_preg_continues = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) &
                                        ~df.la_currently_in_labour]

        # PRE ECLAMPSIA
        self.module.set_pregnancy_complications(month_7_preg_continues, 'pre_eclampsia')

        month_7_no_pe = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) &
                               ~df.ps_mild_pre_eclamp & ~df.ps_severe_pre_eclamp]

        # GESTATIONAL HYPERTENSION
        self.module.set_pregnancy_complications(month_7_no_pe, 'gest_htn')

        # GESTATIONAL DIABETES
        self.module.set_pregnancy_complications(month_7_preg_continues, 'gest_diab')

        # death
        self.module.set_pregnancy_complications(month_7_preg_continues, 'antenatal_death')

        # =========================== MONTH 8 RISK APPLICATION ========================================================
        month_8_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35) &
                             ~df.la_currently_in_labour]

        # STILL BIRTH RISK
        self.module.set_pregnancy_complications(month_8_idx, 'stillbirth')

        month_8_preg_continues = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35) &
                                        ~df.la_currently_in_labour]

        # PRE ECLAMPSIA
        self.module.set_pregnancy_complications(month_8_preg_continues, 'pre_eclampsia')

        month_8_no_pe = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35) &
                               ~df.ps_mild_pre_eclamp & ~df.ps_severe_pre_eclamp]

        # GESTATIONAL HYPERTENSION
        self.module.set_pregnancy_complications(month_8_no_pe, 'gest_htn')

        # GESTATIONAL DIABETES
        self.module.set_pregnancy_complications(month_8_preg_continues, 'gest_diab')

        # death
        self.module.set_pregnancy_complications(month_8_preg_continues, 'antenatal_death')

        # =========================== MONTH 9 RISK APPLICATION ========================================================
        month_9_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40) &
                             ~df.la_currently_in_labour]

        # STILL BIRTH RISK
        self.module.set_pregnancy_complications(month_9_idx, 'stillbirth')

        month_9_preg_continues = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40) &
                                        ~df.la_currently_in_labour]

        # PRE ECLAMPSIA
        self.module.set_pregnancy_complications(month_9_preg_continues, 'pre_eclampsia')

        month_9_no_pe = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40) &
                               ~df.ps_mild_pre_eclamp & ~df.ps_severe_pre_eclamp]

        # GESTATIONAL HYPERTENSION
        self.module.set_pregnancy_complications(month_9_no_pe, 'gest_htn')

        # GESTATIONAL DIABETES
        self.module.set_pregnancy_complications(month_9_preg_continues, 'gest_diab')

        # death
        self.module.set_pregnancy_complications(month_9_preg_continues, 'antenatal_death')

        # =========================== WEEK 41 RISK APPLICATION ========================================================
        # Risk of still birth increases significantly in women who carry pregnancies beyond term
        week_41_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 41) &
                             ~df.la_currently_in_labour]

        self.module.set_pregnancy_complications(week_41_idx, 'stillbirth')

        # =========================== WEEK 42 RISK APPLICATION ========================================================
        week_42_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 42) &
                             ~df.la_currently_in_labour]

        self.module.set_pregnancy_complications(week_42_idx, 'stillbirth')

        # =========================== WEEK 43 RISK APPLICATION ========================================================
        week_43_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 43) &
                             ~df.la_currently_in_labour]

        self.module.set_pregnancy_complications(week_43_idx, 'stillbirth')
        # =========================== WEEK 44 RISK APPLICATION ========================================================
        week_44_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 44) &
                             ~df.la_currently_in_labour]

        self.module.set_pregnancy_complications(week_44_idx, 'stillbirth')
        # =========================== WEEK 45 RISK APPLICATION ========================================================
        week_45_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 45) &
                             ~df.la_currently_in_labour]

        self.module.set_pregnancy_complications(week_45_idx, 'stillbirth')


class EctopicPregnancyRuptureEvent(Event, IndividualScopeEventMixin):
    """This is EctopicPregnancyRuptureEvent. It is scheduled by the PregnancySupervisorEvent for women who have
    experience an ectopic pregnancy. Currently this event applies a risk of death, it is unfinished."""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props

        assert df.at[individual_id, 'ps_ectopic_pregnancy']
        assert self.sim.date - df.at[individual_id, 'la_due_date_current_pregnancy'] < pd.Timedelta(43, unit='d')

        df.at[individual_id, 'is_pregnant'] = False
        df.at[individual_id, 'ps_gestational_age_in_weeks'] = 0
        df.at[individual_id, 'la_due_date_current_pregnancy'] = pd.NaT

        logger.debug('persons %d untreated ectopic pregnancy has now ruptured on date %s', individual_id,
                     self.sim.date)
        # TODO: Symptoms
        # TODO: Care seeking

        # We schedule the ectopic pregnancy death event 3 days after rupture to allow for women to see care and
        # treatment effects to reduce likelihood of death (this is not an accurate representation of time between
        # rupture and death)
        self.sim.schedule_event(EctopicPregnancyDeathEvent(self.module, individual_id), self.sim.date +
                                DateOffset(days=3))


class EctopicPregnancyDeathEvent(Event, IndividualScopeEventMixin):
    """This is EctopicPregnancyRuptureEvent. It is scheduled by the PregnancySupervisorEvent for women who have
    experience an ectopic pregnancy. Currently this event applies a risk of death, it is unfinished."""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters

        assert df.at[individual_id, 'ps_ectopic_pregnancy']

        # Here we apply a risk of death in women who have experienced a ruptured ectopic pregnancy
        random = self.module.rng.random_sample()
        risk_of_death = params['ps_linear_equations']['ectopic_pregnancy_death'].predict(df.loc[[individual_id]])[
            individual_id]

        if random < risk_of_death:
            logger.debug('person %d has died following an ruptured ectopic pregnancy on date %s', individual_id,
                         self.sim.date)
            death = demography.InstantaneousDeath(self.sim.modules['Demography'], individual_id,
                                                  cause='ectopic pregnancy')
            self.sim.schedule_event(death, self.sim.date)
        else:
            disease_reset = PregnancyDiseaseResetEvent(self.module, individual_id)
            self.sim.schedule_event(disease_reset, self.sim.date)


class AbortionEvent(Event, IndividualScopeEventMixin):
    """ This is the Abortion Event. This event is scheduled by the PregnancySupervisorEvent for women who's pregnancy
    has ended due to either spontaneous or induced abortion"""

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

        self.cause = cause

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters


        if self.cause == 'induced_abortion':
            self.module.PregnancyDiseaseTracker['induced_abortion'] += 1

            types_of_abortion = ['surgical', 'medical']
            # todo: Do we store this variable and modify probability of complications by it
            this_womans_abortion = self.module.rng.choice(types_of_abortion, p=params['prob_induced_abortion_type'])
            self.module.set_abortion_complications(individual_id, 'induced_abortion')

            # TODO: care seeking, delay death equation

            risk_of_death_post_abortion = params['ps_linear_equations']['induced_abortion_death'].predict(
                df.loc[[individual_id]])[individual_id]

            if risk_of_death_post_abortion < self.module.rng.random_sample():
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause='labour'), self.sim.date)

        if self.cause == 'spontaneous_abortion':
            # TODO: can these properties go?
            self.module.PregnancyDiseaseTracker['spontaneous_abortion'] += 1

            self.module.set_abortion_complications(individual_id, 'spontaneous_abortion')

            risk_of_death_post_spontaneous_abortion = \
                params['ps_linear_equations']['spontaneous_abortion_death'].predict(
                    df.loc[[individual_id]])[individual_id]

            # TODO: care seeking, delay death equation

            if risk_of_death_post_spontaneous_abortion < self.module.rng.random_sample():
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause='labour'), self.sim.date)

    # TODO: (currently self contained) - predictors in LM for incidence and risk fo death, Symptoms?, care seeking,
    #  abortion death event (to allow for treatment effect)


class PregnancyDiseaseProgressionEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is the PregnancyDiseaseProgressionEvent. It runs every 4 weeks and determines if women who have a disease
    of pregnancy will undergo progression to the next stage. This event will need to be recoded using the
    progression_matrix function """

    # TODO: consider renaming if only dealing with HTN diseases

    def __init__(self, module, ):
        super().__init__(module, frequency=DateOffset(weeks=4))

    def apply(self, population):
        df = self.sim.population.props
        # ============================= PROGRESSION OF PREGNANCY DISEASES ==========================================

        disease_states = ['gestational_htn', 'mild_pre_eclamp', 'severe_pre_eclamp', 'eclampsia']
        prob_matrix = pd.DataFrame(columns=disease_states, index=disease_states)

        prob_matrix['gestational_htn'] = [0.8, 0.2, 0.1, 0.0]
        prob_matrix['mild_pre_eclamp'] = [0.0, 0.8, 0.1, 0.1]
        prob_matrix['severe_pre_eclamp'] = [0.0, 0.1, 0.6, 0.3]
        prob_matrix['eclampsia'] = [0.0, 0.0, 0.1, 0.9]

        # TODO: couldnt someone just move through all these states in one go (also this doesnt work)
        # women_with_gest_htn = df.loc[df.is_alive & df.is_pregnant & df.ps_gestational_htn, 'ps_gestational_htn']
        # new_states_gest_htn = util.transition_states(women_with_gest_htn, prob_matrix, self.module.rng)
        # df.ps_gestational_htn.update(new_states_gest_htn)

        # women_with_pre_eclampsia = df.loc[df.is_alive & df.is_pregnant & df.ps_mild_pre_eclamp, 'ps_mild_pre_eclamp']
        # new_states_mild_pe = util.transition_states(women_with_pre_eclampsia, prob_matrix, self.module.rng)
        # df.ps_mild_pre_eclamp.update(new_states_mild_pe)

        # women_with_severe_pre_eclampsia = df.loc[df.is_alive & df.is_pregnant & df.ps_severe_pre_eclamp,
        #                                         'ps_severe_pre_eclamp']
        # new_states_mild_pe = util.transition_states(women_with_severe_pre_eclampsia, prob_matrix, self.module.rng)
        # df.ps_severe_pre_eclamp.update(new_states_mild_pe)


class PregnancyDiseaseResetEvent(Event, IndividualScopeEventMixin):
    """This is EctopicPregnancyRuptureEvent. It is scheduled by the PregnancySupervisorEvent for women who have
    experience an ectopic pregnancy. Currently this event applies a risk of death, it is unfinished."""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters

        df.at[individual_id, 'ps_ectopic_pregnancy'] = False


class PregnancyLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """This is LabourLoggingEvent. Currently it calculates and produces a yearly output of maternal mortality (maternal
    deaths per 100,000 live births). It is incomplete"""

    def __init__(self, module):
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        df = self.sim.population.props

        # Previous Year...
        one_year_prior = self.sim.date - np.timedelta64(1, 'Y')

        # Denominators...
        total_births_last_year = len(df.index[(df.date_of_birth > one_year_prior) & (df.date_of_birth < self.sim.date)])
        ra_lower_limit = 14
        ra_upper_limit = 50
        women_reproductive_age = df.index[(df.is_alive & (df.sex == 'F') & (df.age_years > ra_lower_limit) &
                                           (df.age_years < ra_upper_limit))]
        total_women_reproductive_age = len(women_reproductive_age)

        # Numerators
        total_ectopics = self.module.PregnancyDiseaseTracker['ectopic_pregnancy']
        total_abortions_t = self.module.PregnancyDiseaseTracker['induced_abortion']
        total_spontaneous_abortions_t = self.module.PregnancyDiseaseTracker['spontaneous_abortion']

        dict_for_output = {'repro_women': total_women_reproductive_age,
                           'total_spontaneous_abortions': total_spontaneous_abortions_t,
                           'spontaneous_abortion_rate': (total_spontaneous_abortions_t /
                                                         total_women_reproductive_age) * 1000,
                           'total_induced_abortions': total_abortions_t,
                           'induced_abortion_rate': (total_abortions_t / total_women_reproductive_age) * 1000,
                           'crude_ectopics': total_ectopics,
                           'ectopic_rate': (total_ectopics / total_women_reproductive_age) * 1000}

        logger.info('%s|summary_stats|%s', self.sim.date, dict_for_output)

        self.module.PregnancyDiseaseTracker = {'ectopic_pregnancy': 0, 'induced_abortion': 0, 'spontaneous_abortion': 0}

        # TODO: figure out why abortions keep going up

