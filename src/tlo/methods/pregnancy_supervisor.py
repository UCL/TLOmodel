from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging, util
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import demography
from tlo.methods.antenatal_care import HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact
from tlo.methods.symptommanager import Symptom
from tlo.methods import Metadata

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PregnancySupervisor(Module):
    """This module is responsible for supervision of pregnancy in the population including incidence of ectopic
    pregnancy, multiple pregnancy, spontaneous abortion, induced abortion, and onset of maternal diseases associated
    with pregnancy and their symptoms (anaemia, hypertenisve disorders and gestational diabetes). This module also
    applies the risk of antenatal death and stillbirth. The module is currently unfinished."""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # Here we define the PregnancyDiseaseTracker dictionary used by the logger to calculate summary stats
        self.PregnancyDiseaseTracker = dict()

    METADATA = {Metadata.DISEASE_MODULE}  # declare that this is a disease module (leave as empty set otherwise)

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
        'rr_anaemia_iron_folic_acid': Parameter(
            Types.REAL, 'risk reduction of maternal anaemia for women taking daily iron/folic acid'),
        'prob_pre_eclampsia_per_month': Parameter(
            Types.REAL, 'underlying risk of pre-eclampsia per month without the impact of risk factors'),
        'rr_pre_eclamp_calcium': Parameter(
            Types.REAL, 'risk reduction of pre-eclampsia for women taking daily calcium supplementation'),
        'prob_of_symptoms_mild_pre_eclampsia': Parameter(
            Types.REAL, 'probability that a woman with mild pre-eclampsia will develop the symptoms of head ache '
                        'and/or blurred vision'),
        'prob_of_symptoms_severe_pre_eclampsia': Parameter(
            Types.REAL, 'probability that a woman with severe pre-eclampsia will develop the symptoms of head ache '
                        'and/or blurred vision and/or epigastric pain'),
        'prob_gest_htn_per_month': Parameter(
            Types.REAL, 'underlying risk of gestational hypertension per month without the impact of risk factors'),
        'rr_gest_htn_calcium': Parameter(
            Types.REAL, 'risk reduction of gestational hypertension for women taking daily calcium supplementation'),
        'prob_gest_diab_per_month': Parameter(
            Types.REAL, 'underlying risk of gestational diabetes per month without the impact of risk factors'),
        'prob_of_symptoms_gest_diab': Parameter(
            Types.REAL, 'probability that a woman with gestational diabetes will develop the symptoms of polyuria '
                        'and/or polyphagia and/or polydipsia'),
        'prob_antepartum_haem_per_month': Parameter(
            Types.REAL, 'monthly probability that a woman will develop antepartum bleeding during pregnancy'),
        'prob_still_birth_per_month': Parameter(
            Types.REAL, 'underlying risk of stillbirth per month without the impact of risk factors'),
        'rr_still_birth_food_supps': Parameter(
            Types.REAL, 'risk reduction of still birth for women receiving nutritional supplements'),
        'prob_antenatal_death_per_month': Parameter(
            Types.REAL, 'underlying risk of antenatal maternal death per month without the impact of risk factors'),
        'monthly_cfr_gest_htn': Parameter(
            Types.REAL, 'monthly risk of death associated with gestational hypertension'),
        'monthly_cfr_mild_pre_eclamp': Parameter(
            Types.REAL, 'monthly risk of death associated with mild pre-eclampsia'),
        'monthly_cfr_severe_pre_eclamp': Parameter(
            Types.REAL, 'monthly risk of death associated with severe pre-eclampsia'),
        'monthly_cfr_gest_diab': Parameter(
            Types.REAL, 'monthly risk of death associated with gestational diabetes'),
        'prob_ectopic_pregnancy_death': Parameter(
            Types.REAL, 'probability of a woman dying from a ruptured ectopic pregnancy'),
        'prob_induced_abortion_type': Parameter(
            Types.LIST, 'probabilities that the type of abortion a woman has will be 1.) Surgical or 2.) Medical'),
        'prob_any_complication_induced_abortion': Parameter(
            Types.REAL, 'probability of a woman that undergoes an induced abortion experiencing any complications'),
        'prob_any_complication_spontaneous_abortion': Parameter(
            Types.REAL, 'probability of a woman that experiences a late miscarriage experiencing any complications'),
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
        'prob_antepartum_haem_stillbirth': Parameter(
            Types.REAL, 'probability of stillbirth for a woman suffering acute antepartum haemorrhage'),
        'prob_antepartum_haem_death': Parameter(
            Types.REAL, 'probability of death for a woman suffering acute antepartum haemorrhage'),
        'prob_first_anc_visit_gestational_age': Parameter(
            Types.LIST, 'probability of initiation of ANC by month'),
        'prob_four_or_more_anc_visits': Parameter(
            Types.REAL, 'probability of a woman undergoing 4 or more basic ANC visits'),
        'probability_htn_persists': Parameter(
            Types.REAL, 'probability of a womans hypertension persisting post birth'),
    }

    PROPERTIES = {
        'ps_gestational_age_in_weeks': Property(Types.INT, 'current gestational age, in weeks, of this womans '
                                                           'pregnancy'),
        'ps_ectopic_pregnancy': Property(Types.BOOL, 'Whether this womans pregnancy is ectopic'),
        'ps_multiple_pregnancy': Property(Types.BOOL, 'Whether this womans is pregnant with multiple fetuses'),
        'ps_anaemia_in_pregnancy': Property(Types.BOOL, 'Whether this womans is anaemic during pregnancy'),
        'ps_will_attend_four_or_more_anc': Property(Types.BOOL, 'Whether this womans is predicted to attend 4 or more '
                                                                'antenatal care visits during her pregnancy'),
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
        'ps_prev_pre_eclamp': Property(Types.BOOL, 'whether this woman has experienced pre-eclampsia in a previous '
                                                   'pregnancy'),
        'ps_gest_diab': Property(Types.BOOL, 'whether this woman has gestational diabetes'),
        'ps_prev_gest_diab': Property(Types.BOOL, 'whether this woman has ever suffered from gestational diabetes '
                                                  'during a previous pregnancy'),
        'ps_antepartum_haemorrhage': Property(Types.BOOL, 'whether this woman has developed and antepartum haemorrhage'),
        'ps_premature_rupture_of_membranes': Property(Types.BOOL, 'whether this woman has experience rupture of '
                                                                  'membranes before the onset of labour. If this is '
                                                                  '<37 weeks from gestation the woman has preterm '
                                                                  'premature rupture of membranes'),
        'dummy_anc_counter': Property(Types.INT, 'dummy tester')
    }

    def read_parameters(self, data_folder):

        params = self.parameters
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_PregnancySupervisor.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)

        # Here we define the parameters which store the probability of symptoms associated with certain maternal
        # diseases of pregnancy

        # TODO: these symptoms are not finalised and currently i dont use them to trigger care seeking outside of a
        #  woman's ANC schedule - I plan for them to be used both diagnositcally and for additional care seeking

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
            'preg_polyuria': 0.2,
            'preg_polyphagia': 0.2}

        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='preg_abdo_pain'),
            Symptom(name='preg_pv_bleeding'),
            Symptom(name='preg_head_ache'),
            Symptom(name='preg_blurred_vision'),
            Symptom(name='preg_epigastric_pain'),
            Symptom(name='preg_polydipsia'),
            Symptom(name='preg_polyuria'),
            Symptom(name='preg_polyphagia'),
            Symptom(name='collapse',
                    emergency_in_adults=True))

        # TODO: DALYs are not finalised/captured fully yet from this model awaiting updates
        if 'HealthBurden' in self.sim.modules.keys():
            params['daly_wt_abortive_outcome'] = self.sim.modules['HealthBurden'].get_daly_weight(352)

        self.sim.modules['HealthSystem'].register_disease_module(self)

    # ==================================== LINEAR MODEL EQUATIONS =====================================================
        # All linear equations used in this module are stored within the ps_linear_equations parameter below
        # (predictors not yet included)
        # TODO: predictors not yet included/finalised

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
                    params['prob_anaemia_per_month'],
                    Predictor('ac_receiving_iron_folic_acid').when(True, params['rr_anaemia_iron_folic_acid'])),
                #   Predictor('ac_date_ifa_runs_out').when(True, params['rr_anaemia_iron_folic_acid'])),
                # TODO: i struggled to work out where i needed to write the function that would work out the date for
                #  this predictor. The treatment effect 'rr_anaemia_iron_folic_acid' should only work if self.sim.date
                #  is prior to 'ac_date_ifa_runs_out' when this equation is called

                'pre_eclampsia': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_pre_eclampsia_per_month'],
                    Predictor('ac_receiving_calcium_supplements').when(True, params['rr_pre_eclamp_calcium'])),

                'gest_htn': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_gest_htn_per_month'],
                    Predictor('ac_receiving_calcium_supplements').when(True, params['rr_gest_htn_calcium'])),

                'gest_diab': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_gest_diab_per_month']),

                'antepartum_haem': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_antepartum_haem_per_month']),

                'antenatal_stillbirth': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_still_birth_per_month'],
                    Predictor('ac_receiving_diet_supplements').when(True, params['rr_still_birth_food_supps'])),

                'antenatal_death': LinearModel(
                    LinearModelType.ADDITIVE,
                    0,
                    Predictor('ps_htn_disorders').when('gest_htn', params['monthly_cfr_gest_htn'])
                                                 .when('mild_pre_eclamp', params['monthly_cfr_mild_pre_eclamp'])
                                                 .when('severe_pre_eclamp', params['monthly_cfr_severe_pre_eclamp']),
                    Predictor('ps_gest_diab').when(True, params['monthly_cfr_gest_diab'])),

                # TODO: Will add treatment effects as appropriate
                # TODO: HIV/TB/Malaria to be added as predictors to generate HIV/TB associated maternal deaths
                #  (will need to occure postnatally to)
                # TODO: other antenatal factors - anaemia (proxy for haemorrhage?)

                'ectopic_pregnancy_death': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_ectopic_pregnancy_death']),

                'induced_abortion_death': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_induced_abortion_death']),

                'spontaneous_abortion_death': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_spontaneous_abortion_death']),

                'antepartum_haemorrhage_stillbirth': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_antepartum_haem_stillbirth']),

                'antepartum_haemorrhage_death': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_antepartum_haem_death']),

                # TODO: severity of bleed should be a predictor for both of the 2 equations

                'four_or_more_anc_visits': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_four_or_more_anc_visits'])}

    def initialise_population(self, population):

        df = population.props

        df.loc[df.is_alive, 'ps_gestational_age_in_weeks'] = 0
        df.loc[df.is_alive, 'ps_ectopic_pregnancy'] = False
        df.loc[df.is_alive, 'ps_multiple_pregnancy'] = False
        df.loc[df.is_alive, 'ps_anaemia_in_pregnancy'] = False
        df.loc[df.is_alive, 'ps_will_attend_four_or_more_anc'] = False
        df.loc[df.is_alive, 'ps_induced_abortion_complication'] = 'none'
        df.loc[df.is_alive, 'ps_spontaneous_abortion_complication'] = 'none'
        df.loc[df.is_alive, 'ps_antepartum_still_birth'] = False
        df.loc[df.is_alive, 'ps_previous_stillbirth'] = False
        df.loc[df.is_alive, 'ps_htn_disorders'] = 'none'
        df.loc[df.is_alive, 'ps_prev_pre_eclamp'] = False
        df.loc[df.is_alive, 'ps_gest_diab'] = False
        df.loc[df.is_alive, 'ps_prev_gest_diab'] = False
        df.loc[df.is_alive, 'ps_antepartum_haemorrhage'] = False
        df.loc[df.is_alive, 'ps_premature_rupture_of_membranes'] = False
        df.loc[df.is_alive, 'dummy_anc_counter'] = 0

    def initialise_simulation(self, sim):

        sim.schedule_event(PregnancySupervisorEvent(self),
                           sim.date + DateOffset(days=0))

        sim.schedule_event(PregnancyLoggingEvent(self),
                           sim.date + DateOffset(years=1))

        # Define the conditions we want to track
        self.PregnancyDiseaseTracker = {'ectopic_pregnancy': 0, 'induced_abortion': 0, 'spontaneous_abortion': 0,
                                        'ectopic_pregnancy_death': 0, 'induced_abortion_death': 0,
                                        'spontaneous_abortion_death': 0, 'maternal_anaemia': 0, 'antenatal_death': 0,
                                        'antenatal_stillbirth': 0, 'new_onset_pre_eclampsia': 0,
                                        'new_onset_gest_htn': 0, 'antepartum_haem': 0, 'antepartum_haem_death': 0}

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props
        params = self.parameters

        df.at[child_id, 'ps_gestational_age_in_weeks'] = 0
        df.at[child_id, 'ps_ectopic_pregnancy'] = False
        df.at[child_id, 'ps_multiple_pregnancy'] = False
        df.at[child_id, 'ps_anaemia_in_pregnancy'] = False
        df.at[child_id, 'ps_will_attend_four_or_more_anc'] = False
        df.at[child_id, 'ps_induced_abortion_complication'] = 'none'
        df.at[child_id, 'ps_spontaneous_abortion_complication'] = 'none'
        df.at[child_id, 'ps_antepartum_still_birth'] = False
        df.at[child_id, 'ps_previous_stillbirth'] = False
        df.at[child_id, 'ps_htn_disorders'] = 'none'
        df.at[child_id, 'ps_prev_pre_eclamp'] = False
        df.at[child_id, 'ps_gest_diab'] = False
        df.at[child_id, 'ps_prev_gest_diab'] = False
        df.at[child_id, 'ps_antepartum_haemorrhage'] = False
        df.at[child_id, 'ps_premature_rupture_of_membranes'] = False
        df.at[child_id, 'dummy_anc_counter'] = 0

        # We reset all womans gestational age when they deliver
        df.at[mother_id, 'ps_gestational_age_in_weeks'] = 0

        # And we remove all the symptoms they may have had antenatally
        # TODO: not all conditions resolve at delivery (hypertension) so this may need adapting
        if df.at[mother_id, 'is_alive']:
             self.sim.modules['SymptomManager'].clear_symptoms(
                person_id=mother_id,
                disease_module=self)

        # =========================================== RESET GDM STATUS ================================================
        # We assume that hyperglycaemia from gestational diabetes resolves following birth
        df.at[mother_id, 'ps_gest_diab'] = False
        # TODO: link with future T2DM

        # ========================== RISK OF ONGOING HTN /RESETTING STATUS AFTER BIRTH ================================
        # Here we apply a one of probability that women who have experienced a hypertensive disorder during pregnancy
        # will remain hypertensive after birth into the postnatal period
        if df.at[mother_id, 'ps_htn_disorders'] == 'gest_htn' or 'mild_pre_eclamp' or 'severe_pre_eclamp':
            if self.rng.random_sample() < params['probability_htn_persists']:
                logger.debug('mother %d will remain hypertensive despite successfully delivering')
            else:
                df.at[mother_id, 'ps_htn_disorders'] = 'none'

        # ================================= RISK OF DE NOVO HTN =======================================================
        # Finally we apply a risk of de novo hypertension in women who have been normatensive during pregnancy

        risk_of_gest_htn = params['ps_linear_equations']['gest_htn'].predict(df.loc[[mother_id]])[mother_id]
        risk_of_mild_pre_eclampsia = params['ps_linear_equations']['pre_eclampsia'].predict(df.loc[[mother_id]])[
            mother_id]

        if self.rng.random_sample() < risk_of_gest_htn:
            df.at[mother_id, 'ps_htn_disorders'] = 'gest_htn'
        else:
            if self.rng.random_sample() < risk_of_mild_pre_eclampsia:
                df.at[mother_id, 'ps_htn_disorders'] = 'mild_pre_eclamp'

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

        # Run checks on women passed to this function
        if not index.empty:
            for person in index.index:
                assert df.at[person, 'is_alive'] and df.at[person, 'is_pregnant'] and df.at[person, 'sex'] == 'F' and \
                       df.at[person, 'age_years'] > 14 and df.at[person, 'age_years'] < 51

        # We apply the results of the linear model to the index of women in question
        result = params['ps_linear_equations'][f'{complication}'].predict(index)

        # And use the result of a random draw to determine which women will experience the complication
        random_draw = pd.Series(self.rng.random_sample(size=len(index)), index=index.index)
        temp_df = pd.concat([result, random_draw], axis=1)
        temp_df.columns = ['result', 'random_draw']

        # Then we use this index to make changes to the data frame and schedule any events required
        positive_index = temp_df.index[temp_df.random_draw < temp_df.result]

        # This is done by cycling through each possible complication that can be passed to this function...
        if complication == 'ectopic':
            # Women suffering from an ectopic pregnancy are scheduled to an event at between 4-6 weeks gestation where
            # they will become symptomatic and may seek care

            df.loc[positive_index, 'ps_ectopic_pregnancy'] = True
            self.PregnancyDiseaseTracker['ectopic_pregnancy'] += len(positive_index)
            for person in positive_index:
                self.sim.schedule_event(EctopicPregnancyEvent(self, person),
                                        (self.sim.date + pd.Timedelta(days=7 * 4 + self.rng.randint(0, 7 * 2))))
            if not positive_index.empty:
                logger.debug(f'The following women have experience an ectopic pregnancy,{positive_index}')

        # TODO: this may be removed (havent yet accurately quantified how important a risk factor multiple
        #  pregnancies is)
        if complication == 'multiples':
            df.loc[positive_index, 'ps_multiple_pregnancy'] = True

        # Women who experience pregnancy loss pass through the abortion function
        if complication == 'spontaneous_abortion' or complication == 'induced_abortion':
            for person in positive_index:
                # TODO: this actually probably doesnt need to be its own function as its only called once
                self.abortion(person, complication)
            if not positive_index.empty:
                logger.debug(f'The following women have experienced an abortion,{positive_index}')

        # Women with new onset anaemia have that property set
        if complication == 'maternal_anaemia':
            df.loc[positive_index, 'ps_anaemia_in_pregnancy'] = True
            self.PregnancyDiseaseTracker['maternal_anaemia'] += len(positive_index)
            if not positive_index.empty:
                logger.debug(f'The following women have developed anaemia during their pregnancy,{positive_index}')

        if complication == 'pre_eclampsia':
            # Check only women without current hypertensive disorder can develop hypertension
            for person in positive_index:
                assert df.at[person, 'ps_htn_disorders'] == 'none'

            df.loc[positive_index, 'ps_prev_pre_eclamp'] = True
            df.loc[positive_index, 'ps_htn_disorders'] = 'mild_pre_eclamp'
            self.PregnancyDiseaseTracker['new_onset_pre_eclampsia'] += len(positive_index)

            # After the appropriate variables are set in the data frame, the set_symptoms function is called to see if
            # women with new onset pre-eclampsia will be symptomatic
            self.set_symptoms(params['prob_of_symptoms_mild_pre_eclampsia'], positive_index)
            if not positive_index.empty:
                logger.debug(f'The following women have developed pre_eclampsia,{positive_index}')

        if complication == 'gest_htn':
            for person in positive_index:
                assert df.at[person, 'ps_htn_disorders'] == 'none'

            # As hypertension is mostly asymptomatic no symptoms are set for these women
            df.loc[positive_index, 'ps_htn_disorders'] = 'gest_htn'
            self.PregnancyDiseaseTracker['new_onset_gest_htn'] += len(positive_index)
            if not positive_index.empty:
                logger.debug(f'The following women have developed gestational hypertension,{positive_index}')

        # The same process is followed for gestational diabetes
        if complication == 'gest_diab':
            for person in positive_index:
                assert ~df.at[person, 'ps_gest_diab']

            df.loc[positive_index, 'ps_gest_diab'] = True
            df.loc[positive_index, 'ps_prev_gest_diab'] = True
            self.set_symptoms(params['prob_of_symptoms_gest_diab'], positive_index)
            if not positive_index.empty:
                logger.debug(f'The following women have developed gestational diabetes,{positive_index}')

        if complication == 'antepartum_haem':
            df.loc[positive_index, 'ps_antepartum_haemorrhage'] = True
            self.PregnancyDiseaseTracker['antepartum_haem'] += len(positive_index)

            # TODO: Source (praevia/abruption), severity, care seeking

            # Because haemorrhage is an acute complication that can quickly result in death/pregnancy loss we use a
            # seperate death event to determine death risk
            for person in positive_index:
                self.sim.schedule_event(AntepartumHaemorrhageDeathEvent(self, person),
                                        (self.sim.date + pd.Timedelta(days=3)))

        # This function is also used to calculate and apply risk of death and stillbirth to women each month
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

            if not positive_index.empty:
                logger.debug(f'The following women have have experienced an antepartum stillbirth,{positive_index}')

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

        # todo: use duration in days parameter to allow symptoms of certain conditions to persist post delivery?

    def abortion(self, individual_id, cause):
        """This function makes changes to the dataframe for women who have experienced induced or spontaneous abortion.
        Additionally it determines if a woman will develop complications associated with pregnancy loss and schedules
        the relevant death events"""
        df = self.sim.population.props
        params = self.parameters

        if df.at[individual_id, 'is_alive']:
            # Women who have an abortion have key pregnancy variables reset
            df.at[individual_id, 'is_pregnant'] = False
            df.at[individual_id, 'la_due_date_current_pregnancy'] = pd.NaT
            df.at[individual_id, 'ac_total_anc_visits_current_pregnancy'] = 0

            # We store the type of abortion for analysis
            self.PregnancyDiseaseTracker[f'{cause}'] += 1

            # We apply a probability of complications to women who miscarry later than 13 weeks
            if cause == 'spontaneous_abortion' and df.at[individual_id, 'ps_gestational_age_in_weeks'] >= 13:
                if params['prob_any_complication_spontaneous_abortion'] < self.rng.random_sample():
                    # Set any complications through this function and schedule possible death

                    # TODO: set complications & symptoms

                    self.sim.schedule_event(EarlyPregnancyLossDeathEvent(self, individual_id,
                                                                         cause=f'{cause}'),
                                            self.sim.date + DateOffset(days=3))

            # We apply a probability of complications to any women who have an induced abortion
            if cause == 'induced_abortion':
                if params['prob_any_complication_induced_abortion'] < self.rng.random_sample():

                    # TODO: set complications & symptoms

                    # Again the death event is scheduled in 3 days time to allow for treatment effects
                    self.sim.schedule_event(EarlyPregnancyLossDeathEvent(self, individual_id,
                                                                         cause=f'{cause}'),
                                            self.sim.date + DateOffset(days=3))

            # We reset gestational age
            df.at[individual_id, 'ps_gestational_age_in_weeks'] = 0

            # TODO:  care seeking

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

        # todo: I think eventutally these values would need to be manipulated by treatment effects? (or will things
        #  like calcium only effect onset of pre-eclampsia not progression)
        # TODO: also should this just be linear, no progression backward or more than one step?

        current_status = df.loc[selected, "ps_htn_disorders"]
        new_status = util.transition_states(current_status, prob_matrix, self.rng)
        df.loc[selected, "ps_htn_disorders"] = new_status

        # TODO: Should symptoms and care-seeking be applied within this function if a woman progresses to a new state?
        # todo: whats the best way to count the incidence of new HDP when transitioning


class PregnancySupervisorEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is the PregnancySupervisorEvent. It runs weekly. It updates gestational age of pregnancy in weeks.
    Presently this event has been hollowed out, additionally it will and uses set_pregnancy_complications function to
    determine if women will experience complication.. """

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
                               ~df.ps_ectopic_pregnancy & (df.dummy_anc_counter == 0)]
        self.module.set_pregnancy_complications(np_no_ectopic, 'multiples')
        # TODO: Review the necessity of including multiple pregnancies

        # ====================================== Scheduling first ANC visit ===========================================
        # For women whose pregnancy continues, we determine at what stage in their pregnancy they will seek antenatal
        # care

        for person in np_no_ectopic.index:
            # We use a probability weighted random draw to determine when this woman will attend ANC1,
            # and scheduled the visit accordingly
            # assert df.loc[person, 'dummy_anc_counter'] == 0
            df.loc[person, 'dummy_anc_counter'] += 1

            # TODO: In the final equation we will assume women dont start attending until it is realistic that they are
            #  aware they are pregnant. The current probabilities are dummyes (have requested data from author of study
            #  for whom this equation is based on)

            # TODO: need to calibrate to ensure that 95% attend 1 ANC

            random_draw_gest_at_anc = self.module.rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                             p=params['prob_first_anc_visit_gestational_age'])

            first_anc_date = self.sim.date + DateOffset(months=random_draw_gest_at_anc)
            first_anc_appt = HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact(
                self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person)

            self.sim.modules['HealthSystem'].schedule_hsi_event(first_anc_appt, priority=0,
                                                                topen=first_anc_date,
                                                                tclose=first_anc_date + DateOffset(days=7))

        # Additionally, we use the linear model to predict if recently pregnant women will attended greater or fewer
        # than 4 ANC visits during pregnancy
        result = params['ps_linear_equations']['four_or_more_anc_visits'].predict(np_no_ectopic)
        random_draw = pd.Series(self.module.rng.random_sample(size=len(np_no_ectopic)), index=np_no_ectopic.index)
        temp_df = pd.concat([result, random_draw], axis=1)
        temp_df.columns = ['result', 'random_draw']

        # And store changes in the data frame accordingly
        positive_index = temp_df.index[temp_df.random_draw < temp_df.result]
        df.loc[positive_index, 'ps_will_attend_four_or_more_anc'] = True

        # =========================================== MONTH 1 =========================================================
        # TODO: its from here onwards that I'm not sure i'm taking the best approach with this repeat indexing of the
        #  dataframe and calling the same function repeatedly.

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
        # spontaneous_abortion
        month_3_idx = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                             (df.ps_gestational_age_in_weeks == 13)]
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
        # spontaneous_abortion
        month_4_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 17)]
        self.module.set_pregnancy_complications(month_4_idx, 'spontaneous_abortion')

        # induced_abortion:
        month_4_no_spontaneous_abortion = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 17)]
        self.module.set_pregnancy_complications(month_4_no_spontaneous_abortion, 'induced_abortion')

        # anaemia
        month_4_no_induced_abortion = df.loc[
            ~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 17) &
            ~df.ps_anaemia_in_pregnancy]
        self.module.set_pregnancy_complications(month_4_no_induced_abortion, 'maternal_anaemia')

        # ============================================= MONTH 5 =======================================================
        # spontaneous_abortion
        month_5_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22)]
        self.module.set_pregnancy_complications(month_5_idx, 'spontaneous_abortion')

        # induced_abortion:
        month_5_no_spontaneous_abortion = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22)]
        self.module.set_pregnancy_complications(month_5_no_spontaneous_abortion, 'induced_abortion')

        # anaemia
        month_5_no_induced_abortion = df.loc[
            df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22) & ~df.ps_anaemia_in_pregnancy]
        self.module.set_pregnancy_complications(month_5_no_induced_abortion, 'maternal_anaemia')

        # Here we begin to apply the risk of developing complications which present later in pregnancy including
        # pre-eclampsia, gestational hypertension and gestational diabetes

        # TODO: discuss with Tim H and Britta how hypertension should be handled in pregnancy, im using a
        #  very binary variable indicating hypertension.

        # pre-eclampsia
        # Only women without pre-existing hypertensive disorders of pregnancy are can develop the disease now
        month_5_no_htn = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22) &
                                (df.ps_htn_disorders == 'none')]
        self.module.set_pregnancy_complications(month_5_no_htn, 'pre_eclampsia')

        # gestational hypertension
        # This is the same for new onset hypertension
        month_5_no_pe = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22)
                               & (df.ps_htn_disorders == 'none')]
        self.module.set_pregnancy_complications(month_5_no_pe, 'gest_htn')

        # gestational diabetes
        month_5_no_diab = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22) &
                                 ~df.ps_gest_diab]
        self.module.set_pregnancy_complications(month_5_no_diab, 'gest_diab')

        # From month 5 we apply a monthly risk of antenatal death that considers the impact of maternal diseases
        # death
        month_5_all_women = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22)]
        self.module.set_pregnancy_complications(month_5_all_women, 'antenatal_death')

        # =========================== MONTH 6 RISK APPLICATION =======================================================
        # TODO: should this be 28 weeks to align with still birth definition
        # From month 6 it is possible women could be in labour at the time of this event so we exclude them

        # still birth
        month_6_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27) &
                             ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_6_idx, 'antenatal_stillbirth')

        # anaemia
        month_6_preg_continues_no_anaemia = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27)
                                                   & ~df.ps_anaemia_in_pregnancy & ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_6_preg_continues_no_anaemia, 'maternal_anaemia')

        # pre-eclampsia
        month_6_preg_continues = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27) &
                                        ~df.la_currently_in_labour & (df.ps_htn_disorders == 'none')]
        self.module.set_pregnancy_complications(month_6_preg_continues, 'pre_eclampsia')

        # gestational hypertension
        month_6_no_pe = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27) &
                               ~df.la_currently_in_labour & (df.ps_htn_disorders == 'none')]
        self.module.set_pregnancy_complications(month_6_no_pe, 'gest_htn')

        # gestational diabetes
        month_6_no_diab = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27) &
                                 ~df.ps_gest_diab & ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_6_no_diab, 'gest_diab')

        # From month six we also determine if women suffering from any hypertensive disorders of pregnancy will progress
        # from one disease to another
        month_6_htn_disorder = df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27) & (
            df.ps_htn_disorders != 'none') & ~df.la_currently_in_labour
        self.module.disease_progression(month_6_htn_disorder)

        # Antepartum haemorrhage
        # From month six we apply a risk of antenatal bleeding to all women
        month_6_preg_continues = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27) &
                                        ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_6_preg_continues, 'antepartum_haem')

        # death
        month_6_all_women = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22) &
                                   ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_6_all_women, 'antenatal_death')

        # =========================== MONTH 7 RISK APPLICATION =======================================================
        # still birth
        month_7_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) &
                             ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_7_idx, 'antenatal_stillbirth')

        # anaemia
        month_7_preg_continues_no_anaemia = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31)
                                                   & ~df.ps_anaemia_in_pregnancy & ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_7_preg_continues_no_anaemia, 'maternal_anaemia')

        # pre-eclampsia
        month_7_preg_continues = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) &
                                        ~df.la_currently_in_labour & (df.ps_htn_disorders == 'none')]
        self.module.set_pregnancy_complications(month_7_preg_continues, 'pre_eclampsia')

        # gestational hypertension
        month_7_no_pe = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) &
                               ~df.la_currently_in_labour & (df.ps_htn_disorders == 'none')]
        self.module.set_pregnancy_complications(month_7_no_pe, 'gest_htn')

        # gestational diabetes
        month_7_no_diab = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) &
                                 ~df.ps_gest_diab & ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_7_no_diab, 'gest_diab')

        # disease progression
        month_7_htn_disorder = df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) & (
            df.ps_htn_disorders != 'none') & ~df.la_currently_in_labour
        self.module.disease_progression(month_7_htn_disorder)

        # death
        month_7_all_women = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) &
                                   ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_7_all_women, 'antenatal_death')

        # =========================== MONTH 8 RISK APPLICATION ========================================================
        # still birth
        month_8_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35) &
                             ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_8_idx, 'antenatal_stillbirth')

        # anaemia
        month_8_preg_continues_no_anaemia = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35)
                                                   & ~df.ps_anaemia_in_pregnancy & ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_8_preg_continues_no_anaemia, 'maternal_anaemia')

        # pre-eclampsia
        month_8_preg_continues = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35) &
                                        ~df.la_currently_in_labour & (df.ps_htn_disorders == 'none')]
        self.module.set_pregnancy_complications(month_8_preg_continues, 'pre_eclampsia')

        # gestational hypertension
        month_8_no_pe = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35)
                               & (df.ps_htn_disorders == 'none') & ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_8_no_pe, 'gest_htn')

        # gestational diabetes
        month_8_no_diab = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35) &
                                 ~df.ps_gest_diab & ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_8_no_diab, 'gest_diab')

        # disease progression
        month_8_htn_disorder = df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35) & (
            df.ps_htn_disorders != 'none')
        self.module.disease_progression(month_8_htn_disorder)

        # death
        month_8_all_women = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35) &
                                   ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_8_all_women, 'antenatal_death')

        # =========================== MONTH 9 RISK APPLICATION ========================================================
        # still birth
        month_9_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40) &
                             ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_9_idx, 'antenatal_stillbirth')

        # anaemia
        month_9_preg_continues_no_anaemia = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40)
                                                   & ~df.ps_anaemia_in_pregnancy & ~df.la_currently_in_labour]

        self.module.set_pregnancy_complications(month_9_preg_continues_no_anaemia, 'maternal_anaemia')

        # pre-eclampsia
        month_9_preg_continues = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40) &
                                        ~df.la_currently_in_labour & (df.ps_htn_disorders == 'none')]
        self.module.set_pregnancy_complications(month_9_preg_continues, 'pre_eclampsia')

        # gestational hypertension
        month_9_no_pe = df.loc[df.is_pregnant & df.is_alive & ~df.la_currently_in_labour &
                               (df.ps_gestational_age_in_weeks == 40) & (df.ps_htn_disorders == 'none')]
        self.module.set_pregnancy_complications(month_9_no_pe, 'gest_htn')

        # gestational diabetes
        month_9_no_diab = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40) &
                                 ~df.ps_gest_diab & ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_9_no_diab, 'gest_diab')

        # disease progression
        month_9_htn_disorder = df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40) & (
            df.ps_htn_disorders != 'none') & ~df.la_currently_in_labour
        self.module.disease_progression(month_9_htn_disorder)

        # death
        month_9_all_women = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40) &
                                   ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_9_all_women, 'antenatal_death')

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

        # Check only the right women have arrived here
        assert df.at[individual_id, 'ps_ectopic_pregnancy']

        # reset pregnancy variables
        if df.at[individual_id, 'is_alive']:
            df.at[individual_id, 'is_pregnant'] = False
            df.at[individual_id, 'ps_gestational_age_in_weeks'] = 0
            df.at[individual_id, 'la_due_date_current_pregnancy'] = pd.NaT

        # Apply a probability of symptoms associated with ectopic pregnancy prior to rupture
        for symp in params['prob_of_unruptured_ectopic_symptoms']:
            if self.module.rng.random_sample() < params['prob_of_unruptured_ectopic_symptoms'][symp]:
                self.sim.modules['SymptomManager'].change_symptom(
                    person_id=individual_id,
                    symptom_string=symp,
                    add_or_remove='+',
                    disease_module=self.module)

        # TODO: Symptom based care seeking & treatment
        # TODO: The rupture event should only be scheduled with unsuccessful care seeking/ failed/incorrect treatment

        # As there is no treatment in the model currently, all women will eventually experience a rupture and may die
        self.sim.schedule_event(EctopicPregnancyRuptureEvent(self.module, individual_id),
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

        # Check the right woman has arrived at this event
        assert df.at[individual_id, 'ps_ectopic_pregnancy']
        # assert self.sim.date - df.at[individual_id, 'la_due_date_current_pregnancy'] < pd.Timedelta(43, unit='d')

        if df.at[individual_id, 'is_alive']:
            logger.debug('persons %d untreated ectopic pregnancy has now ruptured on date %s', individual_id,
                         self.sim.date)

            # We apply an emergency symptom of 'collapse' to all women who have experienced a rupture
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=individual_id,
                disease_module=self.module,
                add_or_remove='+',
                symptom_string='collapse'
            )

        # TODO:  allow for care seeking & Treatment

        # We delayed the death event by three days to allow any treatment effects to mitigate risk of death
        self.sim.schedule_event(EarlyPregnancyLossDeathEvent(self.module, individual_id, cause='ectopic_pregnancy'),
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

        if df.at[individual_id, 'is_alive']:

            # Individual risk of death is calculated through the linear model
            risk_of_death = params['ps_linear_equations'][f'{self.cause}_death'].predict(df.loc[[individual_id]])[
                individual_id]

            # If the death occurs we record it here
            if self.module.rng.random_sample() < risk_of_death:
                logger.debug(f'person %d has died due to {self.cause} on date %s', individual_id,
                             self.sim.date)
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause=f'{self.cause}'), self.sim.date)
                self.module.PregnancyDiseaseTracker[f'{self.cause}_death'] += 1
                self.module.PregnancyDiseaseTracker['antenatal_death'] += 1

            # And if the woman survives we assume she is no longer symptomatic and remove any existing symptoms
            elif self.cause == 'ectopic_pregnancy':
                self.module.sim.modules['SymptomManager'].clear_symptoms(
                    person_id=individual_id,
                    disease_module=self.module)


class AntepartumHaemorrhageDeathEvent(Event, IndividualScopeEventMixin):
    """This is AntepartumHaemorrhageDeathEvent. It is scheduled by the PregnancySupervisorEvent for
    women who are at risk of death following an acute antepartum haemorrhage. Currently this event applies a risk of
    death,it is unfinished"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters

        risk_of_death = params['ps_linear_equations']['antepartum_haemorrhage_death'].predict(df.loc[[
            individual_id]])[individual_id]

        risk_of_stillbirth = params['ps_linear_equations']['antepartum_haemorrhage_stillbirth'].predict(df.loc[[
            individual_id]])[individual_id]

        if df.at[individual_id, 'is_alive']:

            if self.module.rng.random_sample() < risk_of_death:
                logger.debug(f'person %d has died due to antepartum haemorrhage on date %s', individual_id,
                             self.sim.date)
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause='antepartum_haemorrhage'), self.sim.date)
                self.module.PregnancyDiseaseTracker['antenatal_death'] += 1
                self.module.PregnancyDiseaseTracker['antepartum_haem_death'] += 1

            elif self.module.rng.random_sample() < risk_of_stillbirth:
                logger.debug(f'person %d has experience an antepartum stillbirth on date %s', individual_id,
                             self.sim.date)
                self.module.PregnancyDiseaseTracker['antenatal_stillbirth'] += 1

                df.at[individual_id, 'ps_antepartum_still_birth'] = True
                df.at[individual_id, 'ps_previous_stillbirth'] = True
                df.at[individual_id, 'is_pregnant'] = False
                df.at[individual_id, 'la_due_date_current_pregnancy'] = pd.NaT
                df.at[individual_id, 'ps_gestational_age_in_weeks'] = 0

                df.at[individual_id, 'ps_antepartum_haemorrhage'] = False

            else:
                df.at[individual_id, 'ps_antepartum_haemorrhage'] = False


class PregnancyLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """This is PregnancyLoggingEvent. It runs yearly to produce summary statistics around pregnancy."""

    def __init__(self, module):
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(years=self.repeat))

    def apply(self, population):
        df = self.sim.population.props

        # Previous Year...
        one_year_prior = self.sim.date - np.timedelta64(1, 'Y')

        # Denominators...
        total_births_last_year = len(df.index[(df.date_of_birth > one_year_prior) & (df.date_of_birth < self.sim.date)])
        if total_births_last_year == 0:
            total_births_last_year = 1

        ra_lower_limit = 14
        ra_upper_limit = 50
        women_reproductive_age = df.index[(df.is_alive & (df.sex == 'F') & (df.age_years > ra_lower_limit) &
                                           (df.age_years < ra_upper_limit))]
        total_women_reproductive_age = len(women_reproductive_age)

        # Numerators
        antenatal_maternal_deaths = self.module.PregnancyDiseaseTracker['antenatal_death']
        antepartum_stillbirths = self.module.PregnancyDiseaseTracker['antenatal_stillbirth']

        if antenatal_maternal_deaths == 0:
            antenatal_maternal_deaths = 1

        if antepartum_stillbirths == 0:
            antepartum_stillbirths = 1

        total_ectopics = self.module.PregnancyDiseaseTracker['ectopic_pregnancy']
        total_abortions_t = self.module.PregnancyDiseaseTracker['induced_abortion']
        total_spontaneous_abortions_t = self.module.PregnancyDiseaseTracker['spontaneous_abortion']
        total_anaemia_cases = self.module.PregnancyDiseaseTracker['maternal_anaemia']
        total_ectopic_deaths = self.module.PregnancyDiseaseTracker['ectopic_pregnancy_death']
        total_ia_deaths = self.module.PregnancyDiseaseTracker['induced_abortion_death']
        total_sa_deaths = self.module.PregnancyDiseaseTracker['spontaneous_abortion_death']
        crude_new_onset_pe = self.module.PregnancyDiseaseTracker['new_onset_pre_eclampsia']
        crude_new_gh = self.module.PregnancyDiseaseTracker['new_onset_gest_htn']
        crude_aph = self.module.PregnancyDiseaseTracker['antepartum_haem']
        crude_aph_death = self.module.PregnancyDiseaseTracker['antepartum_haem_death']

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
                           'anaemia_rate': (total_anaemia_cases/total_women_reproductive_age) * 1000,
                           'crude_pe': crude_new_onset_pe,
                           'crude_gest_htn': crude_new_gh,
                           'crude_aph':crude_aph,
                           'crude_aph_death':crude_aph_death}

        logger.info('%s|summary_stats|%s', self.sim.date, dict_for_output)

        self.module.PregnancyDiseaseTracker = {'ectopic_pregnancy': 0, 'induced_abortion': 0, 'spontaneous_abortion': 0,
                                               'ectopic_pregnancy_death': 0, 'induced_abortion_death': 0,
                                               'spontaneous_abortion_death': 0, 'maternal_anaemia': 0,
                                               'antenatal_death': 0, 'antenatal_stillbirth': 0,
                                               'new_onset_pre_eclampsia': 0, 'new_onset_gest_htn': 0,
                                               'antepartum_haem': 0, 'antepartum_haem_death': 0}
