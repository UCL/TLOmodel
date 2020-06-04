from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging, util
from tlo.methods import demography
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PregnancySupervisor(Module):
    """This module is responsible for supervision of pregnancy in the population including incidence of ectopic
    pregnancy, multiple pregnancy, spontaneous abortion, induced abortion, and onset of maternal diseases associated
    with pregnancy and their symptoms. This module also deals with the risk of antenatal death and stillbirth.
    The module is currently incomplete as it does not include care seeking for complication or use of antenatal care"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # Here we define the PregnancyDiseaseTracker dictionary used by the logger to calculate summary stats
        self.PregnancyDiseaseTracker = dict()

    PARAMETERS = {
        'prob_ectopic_pregnancy': Parameter(
            Types.REAL, 'probability of a womans pregnancy being ectopic and not uterine implantation'),
        'prob_of_unruptured_ectopic_symptoms': Parameter(
            Types.REAL, 'probability of a woman developing symptoms of an ectopic pregnancy, pre rupture'),
        'prob_multiples': Parameter(
            Types.REAL, 'probability that a woman is currently carrying more than one pregnancy'),
        'prob_spontaneous_abortion_per_month': Parameter(
            Types.REAL, 'underlying risk of spontaneous_abortion per month without the impact of risk factors'),
        'prob_induced_abortion_per_month': Parameter(
            Types.REAL, 'underlying risk of induced abortion per month without the impact of risk factors'),
        'prob_anaemia_per_month': Parameter(
            Types.REAL, 'underlying risk of induced abortion per month without the impact of risk factors'),
        'prob_pre_eclampsia_per_month': Parameter(
            Types.REAL, 'underlying risk of pre-eclampsia per month without the impact of risk factors'),
        'prob_of_symptoms_mild_pre_eclampsia': Parameter(
            Types.REAL, 'probability that a woman with mild pre-eclampsia will develop the symptoms of head ache '
                        'and/or blurred vision'),
        'prob_of_symptoms_severe_pre_eclampsia': Parameter(
            Types.REAL, 'probability that a woman with severe pre-eclampsia will develop the symptoms of head ache '
                        'and/or blurred vision and/or epigastric pain'),
        'prob_gest_htn_per_month': Parameter(
            Types.REAL, 'underlying risk of gestational hypertension per month without the impact of risk factors'),
        'prob_gest_diab_per_month': Parameter(
            Types.REAL, 'underlying risk of gestational diabetes per month without the impact of risk factors'),
        'prob_of_symptoms_gest_diab': Parameter(
            Types.REAL, 'probability that a woman with gestational diabetes will develop the symptoms of polyurea '
                        'and/or polyphagia and/or polydipsia'),
        'prob_still_birth_per_month': Parameter(
            Types.REAL, 'underlying risk of stillbirth per month without the impact of risk factors'),
        'prob_antenatal_death_per_month': Parameter(
            Types.REAL, 'underlying risk of antenatal maternal death per month without the impact of risk factors'),
        'prob_ectopic_pregnancy_death': Parameter(
            Types.REAL, 'probability of a woman dying from a ruptured ectopic pregnancy'),
        'prob_induced_abortion_type': Parameter(
            Types.LIST, 'probabilities that the type of abortion a woman has will be 1.) Surgical or 2.) Medical'),
        'prob_any_complication_induced_abortion': Parameter(
            Types.REAL, 'probability of a woman that undergoes an induced abortion experiencing any complications'),
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
        'ps_multiple_pregnancy': Property(Types.BOOL, 'Whether this womans is pregnant with multiple fetuses'),
        'ps_anaemia_in_pregnancy': Property(Types.BOOL, 'Whether this womans is anaemic during pregnancy'),
        'ps_induced_abortion_complication': Property(Types.LIST, 'List of any complications a woman has experience '
                                                                 'following an induced abortion'),
        'ps_spontaneous_abortion_complication': Property(Types.LIST, 'List of any complications a woman has experience '
                                                                     'following an spontaneous abortion'),
        'ps_antepartum_still_birth': Property(Types.BOOL, 'whether this woman has experienced an antepartum still birth'
                                                          'of her current pregnancy'),
        'ps_previous_stillbirth': Property(Types.BOOL, 'whether this woman has had any previous pregnancies end in '
                                                       'still birth'),  # consider if this should be an interger
        'ps_htn_disorders': Property(Types.CATEGORICAL, 'if this woman suffers from a hypertensive disorder of '
                                                        'pregnancy',
                                     categories=['none', 'gest_htn', 'mild_pre_eclamp', 'severe_pre_eclamp',
                                                 'eclampsia']),
        # todo: decide what to do with the HTN properties (drop booleans etc)
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

    # Symptoms that are associated with conditions of pregnancy are defined here
    SYMPTOMS = {'preg_abdo_pain', 'preg_pv_bleeding', 'em_collapse', 'preg_head_ache', 'preg_blurred_vision',
                'preg_epigastric_pain', 'preg_polydipsia', 'preg_polyurea', 'preg_polyphagia'}

    # TODO: Will we need to consider our own care seeking equation for women who are symptomatic and pregnant
    #  (outside of routine ANC)?

    def read_parameters(self, data_folder):
        params = self.parameters
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_PregnancySupervisor.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)

        # Here we define the parameters which store the probability of symptoms associated with certain maternal
        # diseases of pregnancy
        params['prob_of_unruptured_ectopic_symptoms'] = {
            'preg_abdo_pain': 0.98,
            'preg_pv_bleeding': 0.6}

        params['prob_of_symptoms_mild_pre_eclampsia'] = {
            'preg_head_ache': 0.2,
            'preg_blurred_vision': 0.1}

        params['prob_of_symptoms_severe_pre_eclampsia'] = {
            'preg_head_ache': 0.4,
            'preg_blurred_vision': 0.4,
            'preg_epigastric_pain': 0.4}

        params['prob_of_symptoms_gest_diab'] = {
            'preg_polydipsia': 0.2,
            'preg_polyurea': 0.2,
            'preg_polyphagia': 0.2}
        # TODO: confirm the neatest way to read this parameters in from the resource file

        self.sim.modules['HealthSystem'].register_disease_module(self)

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

                'maternal_anaemia': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_anaemia_per_month']),

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
                # todo: antenatal disease will act as predictors here

                'antenatal_death': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_antenatal_death_per_month']),
                # todo: antenatal disease will act as predictors here

                'ectopic_pregnancy_death': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_ectopic_pregnancy_death']),

                'induced_abortion_death': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_induced_abortion_death']),

                'spontaneous_abortion_death': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_spontaneous_abortion_death'])
        }

    def initialise_population(self, population):

        df = population.props

        df.loc[df.is_alive, 'ps_gestational_age_in_weeks'] = 0
        df.loc[df.is_alive, 'ps_ectopic_pregnancy'] = False
        df.loc[df.is_alive, 'ps_ectopic_symptoms'].values[:] = 'none'
        df.loc[df.is_alive, 'ps_multiple_pregnancy'] = False
        df.loc[df.is_alive, 'ps_anaemia_in_pregnancy'] = False
        df.loc[df.is_alive, 'ps_induced_abortion_complication'] = 'none'
        df.loc[df.is_alive, 'ps_spontaneous_abortion_complication'] = 'none'
        df.loc[df.is_alive, 'ps_antepartum_still_birth'] = False
        df.loc[df.is_alive, 'ps_previous_stillbirth'] = False
        df.loc[df.is_alive, 'ps_htn_disorders'] = 'none'
        df.loc[df.is_alive, 'ps_gestational_htn'] = False
        df.loc[df.is_alive, 'ps_mild_pre_eclamp'] = False
        df.loc[df.is_alive, 'ps_severe_pre_eclamp'] = False
        df.loc[df.is_alive, 'ps_prev_pre_eclamp'] = False
        df.loc[df.is_alive, 'ps_currently_hypertensive'] = False
        df.loc[df.is_alive, 'ps_gest_diab'] = False
        df.loc[df.is_alive, 'ps_prev_gest_diab'] = False
        df.loc[df.is_alive, 'ps_premature_rupture_of_membranes'] = False

    def initialise_simulation(self, sim):

        sim.schedule_event(PregnancySupervisorEvent(self),
                           sim.date + DateOffset(days=0))

        sim.schedule_event(PregnancyLoggingEvent(self),
                           sim.date + DateOffset(years=1))

        # Define the conditions we want to track
        self.PregnancyDiseaseTracker = {'ectopic_pregnancy': 0, 'induced_abortion': 0, 'spontaneous_abortion': 0,
                                        'ectopic_pregnancy_death': 0, 'induced_abortion_death': 0,
                                        'spontaneous_abortion_death': 0, 'maternal_anaemia': 0, 'antenatal_death': 0,
                                        'antenatal_stillbirth': 0}

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props

        df.at[child_id, 'ps_gestational_age_in_weeks'] = 0
        df.at[child_id, 'ps_ectopic_pregnancy'] = False
        df.at[child_id, 'ps_ectopic_symptoms'] = 'none'
        df.at[child_id, 'ps_multiple_pregnancy'] = False
        df.at[child_id, 'ps_anaemia_in_pregnancy'] = False
        df.at[child_id, 'ps_induced_abortion_complication'] = 'none'
        df.at[child_id, 'ps_spontaneous_abortion_complication'] = 'none'
        df.at[child_id, 'ps_antepartum_still_birth'] = False
        df.at[child_id, 'ps_previous_stillbirth'] = False
        df.at[child_id, 'ps_htn_disorders'] = 'none'
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
        Properties are modified depending on the complication passed to the function and the result of a random draw.
        Similarly symptoms are set for women who develop new complications"""
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
            # Women suffering from an ectopic pregnancy are scheduled to an event in between 4-6 weeks gestation where
            # they will become symptomatic and may seek care

            df.loc[positive_index, 'ps_ectopic_pregnancy'] = True
            self.PregnancyDiseaseTracker['ectopic_pregnancy'] += len(positive_index)
            for person in positive_index:
                self.sim.schedule_event(EctopicPregnancyEvent(self, person),
                                        (self.sim.date + pd.Timedelta(days=7 * 4 + self.rng.randint(0, 7 * 2))))
            if not positive_index.empty:
                logger.debug(f'The following women have experience an ectopic pregnancy,{positive_index}')

        if complication == 'multiples':
            df.loc[positive_index, 'ps_multiple_pregnancy'] = True
            # TODO: are we keeping multiples in the model

        if complication == 'spontaneous_abortion' or complication == 'induced_abortion':

            # Women who have an abortion have key pregnancy variables reset
            df.loc[positive_index, 'is_pregnant'] = False
            df.loc[positive_index, 'la_due_date_current_pregnancy'] = pd.NaT
            df.loc[positive_index, 'ps_gestational_age_in_weeks'] = 0

            # And are scheduled to the AbortionEvent where they may develop complications, symptoms and seek care
            for person in positive_index:
                self.sim.schedule_event(AbortionEvent(self, person, cause=f'{complication}'), self.sim.date)
            if not positive_index.empty:
                    logger.debug(f'The following women have experienced an abortion,{positive_index}')

        if complication == 'maternal_anaemia':
            df.loc[positive_index, 'ps_anaemia_in_pregnancy'] = True
            self.PregnancyDiseaseTracker['maternal_anaemia'] += len(positive_index)
            if not positive_index.empty:
                logger.debug(f'The following women have developed anaemia during their pregnancy,{positive_index}')

        if complication == 'pre_eclampsia':
            df.loc[positive_index, 'ps_mild_pre_eclamp'] = True
            df.loc[positive_index, 'ps_prev_pre_eclamp'] = True
            df.loc[positive_index, 'ps_htn_disorders'] = 'mild_pre_eclamp'

            # After the appropriate variables are set in the data frame, the set_symptoms function is called to see if
            # women with new onset pre-eclampsia will be symptomatic
            self.set_symptoms(params['prob_of_symptoms_mild_pre_eclampsia'], positive_index)
            if not positive_index.empty:
                logger.debug(f'The following women have developed pre_eclampsia,{positive_index}')

        if complication == 'gest_htn':
            # As hypertension is mostly asymptomatic no symptoms are set for these women
            df.loc[positive_index, 'ps_gestational_htn'] = True
            df.loc[positive_index, 'ps_htn_disorders'] = 'gest_htn'
            if not positive_index.empty:
                logger.debug(f'The following women have developed gestational hypertension,{positive_index}')

        if complication == 'gest_diab':
            df.loc[positive_index, 'ps_gest_diab'] = True
            df.loc[positive_index, 'ps_prev_gest_diab'] = True
            self.set_symptoms(params['prob_of_symptoms_gest_diab'], positive_index)
            if not positive_index.empty:
                logger.debug(f'The following women have developed gestational diabetes,{positive_index}')

        if complication == 'antenatal_death':
            self.PregnancyDiseaseTracker['antenatal_death'] += len(positive_index)
            for person in positive_index:
                death = demography.InstantaneousDeath(self.sim.modules['Demography'], person,
                                                      cause='antenatal death')
                self.sim.schedule_event(death, self.sim.date)
            if not positive_index.empty:
                logger.debug(f'The following women have died during pregnancy,{positive_index}')

        if complication == 'antenatal_stillbirth':
            self.PregnancyDiseaseTracker['antenatal_stillbirth'] += len(positive_index)

            df.loc[positive_index, 'ps_antepartum_still_birth'] = True
            df.loc[positive_index, 'ps_previous_stillbirth'] = True
            df.loc[positive_index, 'is_pregnant'] = False
            df.loc[positive_index, 'la_due_date_current_pregnancy'] = pd.NaT
            df.loc[positive_index, 'ps_gestational_age_in_weeks'] = 0

            # TODO: care seeking for suspected antenatal stillbirth

            if not positive_index.empty:
                logger.debug(f'The following women have have experienced an antepartum stillbirth,{positive_index}')

        # TODO: consider using return function to produce negative index which could be used for the chain of functions

    def set_symptoms(self, parameter, index):
        """This function applies symptoms to women who have developed a condition during pregnancy"""

        for symp in parameter:
            persons_id_with_symp = np.array(index)[
                self.rng.rand(len(index)) < parameter[symp]
                ]

            self.sim.modules['SymptomManager'].change_symptom(
                person_id=list(persons_id_with_symp),
                symptom_string=symp,
                add_or_remove='+',
                disease_module=self)
            #   duration_in_days=remaining_length_of_pregnancy)

        # todo: set duration of length of symptoms until end of pregnancy

    def set_abortion_complications(self, individual_id, abortion_type):
        """This function applies a probability of complication to women who have undergone an induced or
        spontaneous abortion. It is currently unfinished"""
        df = self.sim.population.props
        params = self.parameters

        # TODO: this can be handled by Asif's BitsetHandler - awaiting PR

        # TODO: for IA - we apply a risk of any complication in the abortion event. Then here we just do a random
        #  choice of the complications (or mix) and apply (as opposed to individual risk)

        if abortion_type == 'induced_abortion':
            types_of_abortion = ['surgical', 'medical']
            this_womans_abortion = self.rng.choice(types_of_abortion, p=params['prob_induced_abortion_type'])

            if this_womans_abortion == 'surgical':
                pass
                # apply probabilities of complications from a surgical induced abortion
                # apply symptoms of these complications
            else:
                pass
                # apply probabilities of complications from a medical induced abortion
                # apply symptoms of these complications
        else:
            pass
            # apply probabilities of complications from a spontaneous abortion

        # TODO: reset these complications after a certain time period

    def disease_progression(self, selected):
        """This function uses util.transition_states to apply a probability of transitioning from one state of
        hypertensive disorder to another during each month of pregnancy"""
        df = self.sim.population.props

        disease_states = ['gest_htn', 'mild_pre_eclamp', 'severe_pre_eclamp', 'eclampsia']
        prob_matrix = pd.DataFrame(columns=disease_states, index=disease_states)

        # TODO: these should be parameters
        prob_matrix['gest_htn'] = [0.8, 0.2, 0.0, 0.0]
        prob_matrix['mild_pre_eclamp'] = [0.0, 0.8, 0.1, 0.1]
        prob_matrix['severe_pre_eclamp'] = [0.0, 0.1, 0.6, 0.3]
        prob_matrix['eclampsia'] = [0.0, 0.0, 0.1, 0.9]

        current_status = df.loc[selected, "ps_htn_disorders"]
        new_status = util.transition_states(current_status, prob_matrix, self.rng)
        df.loc[selected, "ps_htn_disorders"] = new_status

        # TODO: Should symptoms and care-seeking be applied within this function if a woman progresses to a new state?
        # todo: whats the best way to count the incidence of new HDP when transitioning

    def antenatal_disease_reset(self, individual_id):
        """This function is scheduled by the DiseaseReset event in the labour module. It resets the properties for
        antenatal diseases around two weeks after the date of delivery. We do not reset these using the on_birth
        function as some of these conditions have effect in the immediate postpartum period."""

        df = self.sim.population.props

        # TODO: the on_birth function is not used to reset diseases of pregnancy as the effect of these conditions does
        #  not end immediately at birth

        # TODO: need additional checks here? there are checks in the function it comes from

        df.at[individual_id, 'ps_anaemia_in_pregnancy'] = False
        df.at[individual_id, 'ps_htn_disorders'] = 'none'
        df.at[individual_id, 'ps_gestational_htn'] = False
        df.at[individual_id, 'ps_mild_pre_eclamp'] = False
        df.at[individual_id, 'ps_severe_pre_eclamp'] = False
        df.at[individual_id, 'ps_currently_hypertensive'] = False
        df.at[individual_id, 'ps_gest_diab'] = False


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

        # =========================================== MONTH 1 =========================================================
        # Here we look at all the women who have reached one month gestation and apply the risk of early pregnancy loss
        month_1_idx = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks
                                                                                        == 4)]
        self.module.set_pregnancy_complications(month_1_idx, 'spontaneous_abortion')

        # Women whose pregnancy continues may develop anaemia associated with their pregnancy
        month_1_no_spontaneous_abortion = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                                                 (df.ps_gestational_age_in_weeks == 4)]
        self.module.set_pregnancy_complications(month_1_no_spontaneous_abortion, 'maternal_anaemia')

        # =========================================== MONTH 2 =========================================================
        # Now we use the set_pregnancy_complications function to calculate risk and set properties for women whose
        # pregnancy is not ectopic

        # spontaneous_abortion:
        month_2_idx = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                             (df.ps_gestational_age_in_weeks == 8)]
        self.module.set_pregnancy_complications(month_2_idx, 'spontaneous_abortion')

        # Here we use the an index of women who will not miscarry to determine who will seek an induced abortion
        # induced_abortion:
        month_2_no_spontaneous_abortion = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                                                 (df.ps_gestational_age_in_weeks == 8)]
        self.module.set_pregnancy_complications(month_2_no_spontaneous_abortion, 'induced_abortion')

        # anaemia
        month_2_no_induced_abortion = df.loc[
            ~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 8) &
            ~df.ps_anaemia_in_pregnancy]
        self.module.set_pregnancy_complications(month_2_no_induced_abortion, 'maternal_anaemia')

        # =========================================== MONTH 3 =========================================================
        month_3_idx = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                             (df.ps_gestational_age_in_weeks == 13)]
        # spontaneous_abortion
        self.module.set_pregnancy_complications(month_3_idx, 'spontaneous_abortion')

        # induced_abortion:
        month_3_no_spontaneous_abortion = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                                                 (df.ps_gestational_age_in_weeks == 13)]
        self.module.set_pregnancy_complications(month_3_no_spontaneous_abortion, 'induced_abortion')

        # anaemia
        month_3_no_induced_abortion = df.loc[
            ~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 13) &
            ~df.ps_anaemia_in_pregnancy]
        self.module.set_pregnancy_complications(month_3_no_induced_abortion, 'maternal_anaemia')

        # ============================================ MONTH 4 ========================================================
        month_4_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 17)]
        # spontaneous_abortion
        self.module.set_pregnancy_complications(month_4_idx, 'spontaneous_abortion')

        month_4_no_spontaneous_abortion = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 17)]
        # induced_abortion:
        self.module.set_pregnancy_complications(month_4_no_spontaneous_abortion, 'induced_abortion')

        # anaemia
        month_4_no_induced_abortion = df.loc[
            ~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 17) &
            ~df.ps_anaemia_in_pregnancy]
        self.module.set_pregnancy_complications(month_4_no_induced_abortion, 'maternal_anaemia')

        # ============================================= MONTH 5 =======================================================
        month_5_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22)]
        # spontaneous_abortion
        self.module.set_pregnancy_complications(month_5_idx, 'spontaneous_abortion')

        # induced_abortion:
        month_5_no_spontaneous_abortion = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22)]
        self.module.set_pregnancy_complications(month_5_no_spontaneous_abortion, 'induced_abortion')

        # anaemia
        month_5_no_induced_abortion = df.loc[
            ~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22) &
            ~df.ps_anaemia_in_pregnancy]
        self.module.set_pregnancy_complications(month_5_no_induced_abortion, 'maternal_anaemia')

        # Here we begin to apply the risk of developing complications which present later in pregnancy including
        # pre-eclampsia, gestational hypertension and gestational diabetes

        # pre-eclampsia
        # Only women without pre-existing hypertensive disorders of pregnancy are can develop the disease now
        month_5_preg_continues = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22) &
                                        (df.ps_htn_disorders == 'none')]
        self.module.set_pregnancy_complications(month_5_preg_continues, 'pre_eclampsia')

        month_5_no_pe = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22) &
                               ~df.ps_mild_pre_eclamp & ~df.ps_severe_pre_eclamp]

        # gestational hypertension
        # Similarly only those women who dont develop pre-eclampsia are able to develop gestational hypertension
        self.module.set_pregnancy_complications(month_5_no_pe, 'gest_htn')

        # gestational diabetes
        self.module.set_pregnancy_complications(month_5_preg_continues, 'gest_diab')

        # From month 5 we apply a monthly risk of antenatal death that considers the impact of maternal diseases
        # death
        self.module.set_pregnancy_complications(month_5_preg_continues, 'antenatal_death')

        # =========================== MONTH 6 RISK APPLICATION =======================================================
        # TODO: should this be 28 weeks to align with still birth definition
        # From month 6 it is possible women could be in labour at the time of this event so we exclude them
        month_6_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27) &
                             ~df.la_currently_in_labour]

        # still birth
        self.module.set_pregnancy_complications(month_6_idx, 'antenatal_stillbirth')

        # anaemia
        month_6_preg_continues_no_anaemia = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27)
                                                   & ~df.ps_anaemia_in_pregnancy & ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_6_preg_continues_no_anaemia, 'maternal_anaemia')

        # pre-eclampsia
        month_6_preg_continues = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27) &
                                        ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_6_preg_continues, 'pre_eclampsia')

        month_6_no_pe = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27) &
                               ~df.ps_mild_pre_eclamp & ~df.ps_severe_pre_eclamp]

        # gestational hypertension
        self.module.set_pregnancy_complications(month_6_no_pe, 'gest_htn')

        # gestational diabetes
        self.module.set_pregnancy_complications(month_6_preg_continues, 'gest_diab')

        # From month six we also determine if women suffering from any hypertensive disorders of pregnancy will progress
        # from one disease to another
        month_6_htn_disorder = df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27) & (
            df.ps_htn_disorders != 'none')
        self.module.disease_progression(month_6_htn_disorder)

        # death
        self.module.set_pregnancy_complications(month_6_preg_continues, 'antenatal_death')

        # =========================== MONTH 7 RISK APPLICATION =======================================================
        month_7_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) &
                             ~df.la_currently_in_labour]

        # still birth
        self.module.set_pregnancy_complications(month_7_idx, 'antenatal_stillbirth')

        # anaemia
        month_7_preg_continues_no_anaemia = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31)
                                                   & ~df.ps_anaemia_in_pregnancy & ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_7_preg_continues_no_anaemia, 'maternal_anaemia')

        # pre-eclampsia
        month_7_preg_continues = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) &
                                        ~df.la_currently_in_labour]

        self.module.set_pregnancy_complications(month_7_preg_continues, 'pre_eclampsia')

        month_7_no_pe = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) &
                               ~df.ps_mild_pre_eclamp & ~df.ps_severe_pre_eclamp]

        # gestational hypertension
        self.module.set_pregnancy_complications(month_7_no_pe, 'gest_htn')

        # gestational diabetes
        self.module.set_pregnancy_complications(month_7_preg_continues, 'gest_diab')

        # disease progression
        month_7_htn_disorder = df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) & (
            df.ps_htn_disorders != 'none')
        self.module.disease_progression(month_7_htn_disorder)

        # death
        self.module.set_pregnancy_complications(month_7_preg_continues, 'antenatal_death')

        # =========================== MONTH 8 RISK APPLICATION ========================================================
        month_8_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35) &
                             ~df.la_currently_in_labour]

        # still birth
        self.module.set_pregnancy_complications(month_8_idx, 'antenatal_stillbirth')

        month_8_preg_continues = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35) &
                                        ~df.la_currently_in_labour]

        # anaemia
        month_8_preg_continues_no_anaemia = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35)
                                                   & ~df.ps_anaemia_in_pregnancy & ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_8_preg_continues_no_anaemia, 'maternal_anaemia')

        # pre-eclampsia
        self.module.set_pregnancy_complications(month_8_preg_continues, 'pre_eclampsia')
        month_8_no_pe = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35) &
                               ~df.ps_mild_pre_eclamp & ~df.ps_severe_pre_eclamp]

        # gestational hypertension
        self.module.set_pregnancy_complications(month_8_no_pe, 'gest_htn')

        # gestational diabetes
        self.module.set_pregnancy_complications(month_8_preg_continues, 'gest_diab')

        # disease progression
        month_8_htn_disorder = df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35) & (
            df.ps_htn_disorders != 'none')
        self.module.disease_progression(month_8_htn_disorder)

        # death
        self.module.set_pregnancy_complications(month_8_preg_continues, 'antenatal_death')

        # =========================== MONTH 9 RISK APPLICATION ========================================================
        month_9_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40) &
                             ~df.la_currently_in_labour]

        # still birth
        self.module.set_pregnancy_complications(month_9_idx, 'antenatal_stillbirth')

        month_9_preg_continues = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40) &
                                        ~df.la_currently_in_labour]
        # anaemia
        month_9_preg_continues_no_anaemia = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40)
                                                   & ~df.ps_anaemia_in_pregnancy & ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_9_preg_continues_no_anaemia, 'maternal_anaemia')

        # pre-eclampsia
        self.module.set_pregnancy_complications(month_9_preg_continues, 'pre_eclampsia')

        month_9_no_pe = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40) &
                               ~df.ps_mild_pre_eclamp & ~df.ps_severe_pre_eclamp]

        # gestational hypertension
        self.module.set_pregnancy_complications(month_9_no_pe, 'gest_htn')

        # gestational diabetes
        self.module.set_pregnancy_complications(month_9_preg_continues, 'gest_diab')

        # disease progression
        month_9_htn_disorder = df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40) & (
            df.ps_htn_disorders != 'none')
        self.module.disease_progression(month_9_htn_disorder)

        # death
        self.module.set_pregnancy_complications(month_9_preg_continues, 'antenatal_death')

        # =========================== WEEK 41 RISK APPLICATION ========================================================
        # Risk of still birth increases significantly in women who carry pregnancies beyond term
        week_41_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 41) &
                             ~df.la_currently_in_labour]

        self.module.set_pregnancy_complications(week_41_idx, 'antenatal_stillbirth')

        # =========================== WEEK 42 RISK APPLICATION ========================================================
        week_42_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 42) &
                             ~df.la_currently_in_labour]

        self.module.set_pregnancy_complications(week_42_idx, 'antenatal_stillbirth')

        # =========================== WEEK 43 RISK APPLICATION ========================================================
        week_43_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 43) &
                             ~df.la_currently_in_labour]

        self.module.set_pregnancy_complications(week_43_idx, 'antenatal_stillbirth')
        # =========================== WEEK 44 RISK APPLICATION ========================================================
        week_44_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 44) &
                             ~df.la_currently_in_labour]

        self.module.set_pregnancy_complications(week_44_idx, 'antenatal_stillbirth')
        # =========================== WEEK 45 RISK APPLICATION ========================================================
        week_45_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 45) &
                             ~df.la_currently_in_labour]

        self.module.set_pregnancy_complications(week_45_idx, 'antenatal_stillbirth')


class EctopicPregnancyEvent(Event, IndividualScopeEventMixin):
    """This is EctopicPregnancyEvent. It is scheduled by the PregnancySupervisorEvent. This event makes changes to the
    data frame for women with ectopic pregnancies, applies a probability of symptoms and schedules the
    EctopicRuptureEvent. This event is unfinished, it will include care seeking based on symptoms"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters

        assert df.at[individual_id, 'ps_ectopic_pregnancy']

        df.at[individual_id, 'is_pregnant'] = False
        df.at[individual_id, 'ps_gestational_age_in_weeks'] = 0
        df.at[individual_id, 'la_due_date_current_pregnancy'] = pd.NaT

        # TODO: Check this is doing the right thing
        for symp in params['prob_of_unruptured_ectopic_symptoms']:
            if self.module.rng.random_sample() < params['prob_of_unruptured_ectopic_symptoms'][symp]:
                self.sim.modules['SymptomManager'].change_symptom(
                    person_id=individual_id,
                    symptom_string=symp,
                    add_or_remove='+',
                    disease_module=self.module)

        # TODO: Symptom based care seeking
        # TODO: condition rupture event on SUCCESSFUL careseeking

        self.sim.schedule_event(EctopicPregnancyRuptureEvent(self, individual_id),
                                (self.sim.date + pd.Timedelta(days=7 * 4 + self.module.rng.randint(0, 7 * 2))))

        # TODO: Currently only ruptured ectopics pass through the death event, is that ok?


class EctopicPregnancyRuptureEvent(Event, IndividualScopeEventMixin):
    """This is EctopicPregnancyRuptureEvent. It is scheduled by the EctopicPregnancyEvent for women who have
    experienced an ectopic pregnancy which has ruptured due to lack of treatment. This event manages symptoms of
    ruptured ectopic pregnancy and schedules EarlyPregnancyLossDeathEvent. The event is unfinished as it will include
    care seeking"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props

        assert df.at[individual_id, 'ps_ectopic_pregnancy']
        # assert self.sim.date - df.at[individual_id, 'la_due_date_current_pregnancy'] < pd.Timedelta(43, unit='d')

        logger.debug('persons %d untreated ectopic pregnancy has now ruptured on date %s', individual_id,
                     self.sim.date)

        self.sim.modules['SymptomManager'].change_symptom(
            person_id=individual_id,
            disease_module=self.module,
            add_or_remove='+',
            symptom_string='em_collapse'
        )

        # TODO: Care seeking

        # We schedule the ectopic pregnancy death event 3 days after rupture to allow for women to see care and
        # treatment effects to reduce likelihood of death (this is not an accurate representation of time between
        # rupture and death)
        self.sim.schedule_event(EarlyPregnancyLossDeathEvent(self.module, individual_id, cause='ectopic_pregnancy'),
                                self.sim.date + DateOffset(days=3))


class AbortionEvent(Event, IndividualScopeEventMixin):
    """ This is the Abortion Event. This event is scheduled by the PregnancySupervisorEvent for women who's pregnancy
    has ended due to either spontaneous or induced abortion"""

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

        self.cause = cause

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters

        self.module.PregnancyDiseaseTracker[f'{self.cause}'] += 1

        if params['prob_any_complication_induced_abortion'] < self.module.rng.random_sample():
            self.module.set_abortion_complications(individual_id, f'{self.cause}')

            # TODO: symptoms - in the above function?
            # TODO: care seeking

        self.sim.schedule_event(EarlyPregnancyLossDeathEvent(self.module, individual_id, cause=f'{self.cause}'),
                                self.sim.date + DateOffset(days=3))


class EarlyPregnancyLossDeathEvent(Event, IndividualScopeEventMixin):
    """This is EarlyPregnancyLossDeathEvent. It is scheduled by the EctopicPregnancyRuptureEvent & AbortionEvent for
    women who are at risk of death following a loss of their pregnancy. Currently this event applies a risk of death,
    it is unfinished."""

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

        self.cause = cause

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters

        risk_of_death = params['ps_linear_equations'][f'{self.cause}_death'].predict(df.loc[[individual_id]])[
            individual_id]

        if self.module.rng.random_sample() < risk_of_death:
            logger.debug(f'person %d has died due to {self.cause} on date %s', individual_id,
                         self.sim.date)
            self.sim.schedule_event(demography.InstantaneousDeath(
                self.module, individual_id, cause=f'{self.cause}'), self.sim.date)
            self.module.PregnancyDiseaseTracker[f'{self.cause}_death'] += 1
            self.module.PregnancyDiseaseTracker['antenatal_death'] += 1

        elif self.cause == 'ectopic_pregnancy':
            for symp in params['prob_of_unruptured_ectopic_symptoms']:
                self.sim.modules['SymptomManager'].change_symptom(
                    person_id=individual_id,
                    symptom_string=symp,
                    add_or_remove='-',
                    disease_module=self.module)

            self.sim.modules['SymptomManager'].change_symptom(
                person_id=individual_id,
                symptom_string='em_collapse',
                add_or_remove='-',
                disease_module=self.module)


class PregnancyLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """This is LabourLoggingEvent. Currently it calculates and produces a yearly output of maternal mortality (maternal
    deaths per 100,000 live births). It is incomplete"""

    def __init__(self, module):
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(years=self.repeat))

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
        antenatal_maternal_deaths = self.module.PregnancyDiseaseTracker['antenatal_death']
        antepartum_stillbirths = self.module.PregnancyDiseaseTracker['antenatal_stillbirth']
        total_ectopics = self.module.PregnancyDiseaseTracker['ectopic_pregnancy']
        total_abortions_t = self.module.PregnancyDiseaseTracker['induced_abortion']
        total_spontaneous_abortions_t = self.module.PregnancyDiseaseTracker['spontaneous_abortion']
        total_anaemia_cases = self.module.PregnancyDiseaseTracker['maternal_anaemia']
        total_ectopic_deaths = self.module.PregnancyDiseaseTracker['ectopic_pregnancy_death']
        total_ia_deaths = self.module.PregnancyDiseaseTracker['induced_abortion_death']
        total_sa_deaths = self.module.PregnancyDiseaseTracker['spontaneous_abortion_death']

        dict_for_output = {'repro_women': total_women_reproductive_age,
                           'antenatal_mmr': (antenatal_maternal_deaths/total_births_last_year) * 100000,
                           'antenatal_sbr': (antepartum_stillbirths/total_births_last_year) * 100,
                           'total_spontaneous_abortions': total_spontaneous_abortions_t,
                           'spontaneous_abortion_rate': (total_spontaneous_abortions_t /
                                                         total_women_reproductive_age) * 1000,
                           'spontaneous_abortion_death_rate': (total_sa_deaths / total_women_reproductive_age) * 1000,
                           'total_induced_abortions': total_abortions_t,
                           'induced_abortion_rate': (total_abortions_t / total_women_reproductive_age) * 1000,
                           'induced_abortion_death_rate': (total_ia_deaths / total_women_reproductive_age) * 1000,
                           'crude_ectopics': total_ectopics,
                           'ectopic_rate': (total_ectopics / total_women_reproductive_age) * 1000,
                           'ectopic_death_rate': (total_ectopic_deaths / total_women_reproductive_age) * 1000,
                           'crude_anaemia': total_anaemia_cases,
                           'anaemia_rate': (total_anaemia_cases/total_women_reproductive_age) * 1000}

        logger.info('%s|summary_stats|%s', self.sim.date, dict_for_output)

        self.module.PregnancyDiseaseTracker = \
            {'ectopic_pregnancy': 0, 'induced_abortion': 0, 'spontaneous_abortion': 0, 'ectopic_pregnancy_death': 0,
             'induced_abortion_death': 0, 'spontaneous_abortion_death': 0, 'maternal_anaemia': 0, 'antenatal_death': 0,
             'antenatal_stillbirth': 0}
