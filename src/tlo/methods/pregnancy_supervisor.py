import numpy as np
import pandas as pd
from pathlib import Path

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PregnancySupervisor(Module):
    """This module is responsible for supervision of pregnancy in the population including incidence of ectopic
    pregnancy, multiple pregnancy, miscarriage, abortion, and onset of antenatal complications. This module is
    incomplete, currently antenatal death has not been coded. Similarly antenatal care seeking will be house hear, for
    both routine treatment and in emergencies"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    PARAMETERS = {
        'prob_pregnancy_factors': Parameter(
            Types.DATA_FRAME, 'Data frame containing probabilities of key outcomes/complications associated with the '
                              'antenatal period'),
        'base_prev_pe': Parameter(
            Types.REAL, 'relative risk of miscarriage for women who have previously miscarried'),
        'base_prev_gest_htn': Parameter(
            Types.REAL, 'relative risk of miscarriage for women who have previously miscarried'),
        'base_prev_gest_diab': Parameter(
            Types.REAL, 'relative risk of miscarriage for women who have previously miscarried'),
        'rr_miscarriage_prevmiscarriage': Parameter(
            Types.REAL, 'relative risk of miscarriage for women who have previously miscarried'),
        'rr_miscarriage_35': Parameter(
            Types.REAL, 'relative risk of miscarriage for women who is over 35 years old'),
        'rr_miscarriage_3134': Parameter(
            Types.REAL, 'relative risk of miscarriage for women who is between 31 and 34 years old'),
        'rr_miscarriage_grav4': Parameter(
            Types.REAL, 'relative risk of miscarriage for women who has a gravidity of greater than 4'),
        'rr_pre_eclamp_nulip': Parameter(
            Types.REAL, 'relative risk of pre-eclampsia in nuliparous women'),
        'rr_pre_eclamp_prev_pe': Parameter(
            Types.REAL, 'relative risk of pre- eclampsia in women who have previous suffered from pre-eclampsia'),
        'rr_gest_diab_overweight': Parameter(
            Types.REAL, 'relative risk of gestational diabetes in women who are overweight at time of pregnancy'),
        'rr_gest_diab_stillbirth': Parameter(
            Types.REAL, 'relative risk of gestational diabetes in women who have previously had a still birth'),
        'rr_gest_diab_prevdiab': Parameter(
            Types.REAL, 'relative risk of gestational diabetes in women who suffered from gestational diabetes in '
                        'previous pregnancy'),
        'rr_gest_diab_chron_htn': Parameter(
            Types.REAL, 'relative risk of gestational diabetes in women who suffer from chronic hypertension'),
        'prob_ectopic_pregnancy': Parameter(
            Types.REAL, 'probability that a womans current pregnancy is ectopic'),
        'level_of_symptoms_ep': Parameter(
            Types.REAL, 'Level of symptoms that the individual will have'),
        'prob_multiples': Parameter(
            Types.REAL, 'probability that a woman is currently carrying more than one pregnancy'),
        'prob_pa_complications': Parameter(
            Types.REAL, 'probability that a woman who has had an induced abortion will experience any complications'),
        'prob_pa_complication_type': Parameter(
            Types.REAL, 'List of probabilities that determine what type of complication a woman who has had an abortion'
                        ' will experience'),
        'r_mild_pe_gest_htn': Parameter(
            Types.REAL, 'probability per month that a woman will progress from gestational hypertension to mild '
                        'pre-eclampsia'),
        'r_severe_pe_mild_pe': Parameter(
            Types.REAL, 'probability per month that a woman will progress from mild pre-eclampsia to severe '
                        'pre-eclampsia'),
        'r_eclampsia_severe_pe': Parameter(
            Types.REAL, 'probability per month that a woman will progress from severe pre-eclampsia to eclampsia'),
        'r_hellp_severe_pe': Parameter(
            Types.REAL, 'probability per month that a woman will progress from severe pre-eclampsia to HELLP syndrome'),
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
        'ps_total_miscarriages': Property(Types.INT, 'the number of miscarriages a woman has experienced'),
        'ps_total_induced_abortion': Property(Types.INT, 'the number of induced abortions a woman has experienced'),
        'ps_abortion_complication': Property(Types.CATEGORICAL, 'Type of complication following an induced abortion: '
                                                                'None; Sepsis; Haemorrhage; Sepsis and Haemorrhage',
                                             categories=['none', 'haem', 'sepsis', 'haem_sepsis']),
        'ps_antepartum_still_birth': Property(Types.BOOL, 'whether this woman has experienced an antepartum still birth'
                                                          'of her current pregnancy'),
        'ps_previous_stillbirth': Property(Types.BOOL, 'whether this woman has had any previous pregnancies end in '
                                                       'still birth'),  # consider if this should be an interger
        'ps_htn_disorder_preg': Property(Types.CATEGORICAL,  'Hypertensive disorders of pregnancy: none, '
                                                             'gestational hypertension, mild pre-eclampsia,'
                                                             'severe pre-eclampsia, eclampsia,'
                                                             ' HELLP syndrome',
                                         categories=['none', 'gest_htn', 'mild_pe', 'severe_pe', 'eclampsia', 'HELLP']),
        'ps_prev_pre_eclamp': Property(Types.BOOL, 'whether this woman has experienced pre-eclampsia in a previous '
                                                   'pregnancy'),
        'ps_gest_diab': Property(Types.BOOL, 'whether this woman has gestational diabetes'),
        'ps_prev_gest_diab': Property(Types.BOOL, 'whether this woman has ever suffered from gestational diabetes '
                                                  'during a previous pregnancy')
    }

    def read_parameters(self, data_folder):
        params = self.parameters
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_PregnancySupervisor.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)

        # These parameters presently hard coded for ease. Both may be deleted following epi review.
        params['level_of_symptoms_ep'] = pd.DataFrame(
            data={'level_of_symptoms_ep': ['none',
                                           'abdominal pain',
                                           'abdominal pain plus bleeding',
                                           'shock'], 'probability': [0.25, 0.25, 0.25, 0.25]})  # DUMMY

        params['type_pa_complication'] = pd.DataFrame(
            data={'type_pa_complication': ['haem',
                                           'sepsis',
                                           'haem_sepsis'], 'probability': [0.5, 0.3, 0.2]})

        if 'HealthBurden' in self.sim.modules.keys():
            params['daly_wt_abortive_outcome'] = self.sim.modules['HealthBurden'].get_daly_weight(352)

# ==================================== LINEAR MODEL EQUATIONS ==========================================================
        # Will live here

    def initialise_population(self, population):

        df = population.props

        df.loc[df.sex == 'F', 'ps_gestational_age_in_weeks'] = 0
        df.loc[df.sex == 'F', 'ps_ectopic_pregnancy'] = False
        df.loc[df.sex == 'F', 'ps_ectopic_symptoms'].values[:] = 'none'
        df.loc[df.sex == 'F', 'ps_ep_unified_symptom_code'] = 0
        df.loc[df.sex == 'F', 'ps_multiple_pregnancy'] = False
        df.loc[df.sex == 'F', 'ps_total_miscarriages'] = 0
        df.loc[df.sex == 'F', 'ps_total_induced_abortion'] = 0
        df.loc[df.sex == 'F', 'ps_abortion_complication'].values[:] = 'none'
        df.loc[df.sex == 'F', 'ps_antepartum_still_birth'] = False
        df.loc[df.sex == 'F', 'ps_previous_stillbirth'] = False
        df.loc[df.sex == 'F', 'ps_htn_disorder_preg'].values[:] = 'none'
        df.loc[df.sex == 'F', 'ps_prev_pre_eclamp'] = False
        df.loc[df.sex == 'F', 'ps_gest_diab'] = False
        df.loc[df.sex == 'F', 'ps_prev_gest_diab'] = False

    def initialise_simulation(self, sim):
        """Get ready for simulation start.
        """
        event = PregnancySupervisorEvent
        sim.schedule_event(event(self),
                           sim.date + DateOffset(days=0))

        event = PregnancyDiseaseProgressionEvent
        sim.schedule_event(event(self),
                           sim.date + DateOffset(days=0))

#        self.sim.modules['HealthSystem'].register_disease_module(self)

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props

        if df.at[child_id, 'sex'] == 'F':
            df.at[child_id, 'ps_gestational_age_in_weeks'] = 0
            df.at[child_id, 'ps_ectopic_pregnancy'] = False
            df.at[child_id, 'ps_ectopic_symptoms'] = 'none'
            df.at[child_id, 'ps_ep_unified_symptom_code'] = 0
            df.at[child_id, 'ps_multiple_pregnancy'] = False
            df.at[child_id, 'ps_total_miscarriages'] = 0
            df.at[child_id, 'ps_total_induced_abortion'] = 0
            df.at[child_id, 'ps_abortion_complication'] = 'none'
            df.at[child_id, 'ps_antepartum_still_birth'] = False
            df.at[child_id, 'ps_previous_stillbirth'] = False
            df.at[child_id, 'ps_htn_disorder_preg'] = 'none'
            df.at[child_id, 'ps_prev_pre_eclamp'] = False
            df.at[child_id, 'ps_gest_diab'] = False
            df.at[child_id, 'ps_prev_gest_diab'] = False

    def on_hsi_alert(self, person_id, treatment_id):

        logger.debug('This is PregnancySupervisor, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):
        logger.debug('This is PregnancySupervisor reporting my health values')
        # TODO: Antenatal DALYs


class PregnancySupervisorEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is the PregnancySupervisorEvent. It runs weekly. It updates gestational age of pregnancy in weeks.
    Presently this event has been hollowed out, additionally it will and uses set_pregnancy_complications function to
    determine if women will experience complication. This event is incomplete and will eventually apply risk of
     antenatal death and handle antenatal care seeking. """

    def __init__(self, module,):
        super().__init__(module, frequency=DateOffset(weeks=1))

    def apply(self, population):
        df = population.props

    # ===================================== UPDATING GESTATIONAL AGE IN WEEKS  ========================================
        # Here we update the gestational age in weeks of all currently pregnant women in the simulation

        gestation_in_days = self.sim.date - df.loc[df.is_pregnant, 'date_of_last_pregnancy']
        gestation_in_weeks = gestation_in_days / np.timedelta64(1, 'W')
        pregnant_idx = df.index[df.is_alive & df.is_pregnant]
        df.loc[pregnant_idx, 'ps_gestational_age_in_weeks'] = gestation_in_weeks.astype('int64')

    # ======================================= PREGNANCY COMPLICATIONS ==================================================
        # Application of pregnancy complications will occur here


class PregnancyDiseaseProgressionEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is the PregnancyDiseaseProgressionEvent. It runs every 4 weeks and determines if women who have a disease
    of pregnancy will undergo progression to the next stage. This event will need to be recoded using the
    progression_matrix function """
    # TODO: consider renaming if only dealing with HTN diseases

    def __init__(self, module,):
        super().__init__(module, frequency=DateOffset(weeks=4))

    def apply(self, population):
        """This is where progression of diseases will be handled"""

    # ============================= PROGRESSION OF PREGNANCY DISEASES ==========================================
    # Progression of pregnancy diseases will live here
