
import numpy as np

from tlo import DateOffset, Module, Property, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PregnancySupervisor(Module):
    """This module is responsible for supervision of pregnancy in the population including incidence of ectopic
    pregnancy, multiple pregnancy, miscarriage, abortion, and onset of antenatal complications. This module is
    incomplete, currently antenatal death has not been coded. Similarly antenatal care seeking will be house hear, for
    both routine treatment and in emergencies"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    PARAMETERS = {}

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
        #    dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_PregnancySupervisor.xlsx',
        #                        sheet_name='parameter_values_old')
        #    self.load_parameters_from_dataframe(dfd)

        if 'HealthBurden' in self.sim.modules.keys():
            params['daly_wt_abortive_outcome'] = self.sim.modules['HealthBurden'].get_daly_weight(352)

# ==================================== LINEAR MODEL EQUATIONS ==========================================================
        # Will live here...

    def initialise_population(self, population):

        df = population.props

        df.loc[df.is_alive, 'ps_gestational_age_in_weeks'] = 0
        df.loc[df.is_alive, 'ps_ectopic_pregnancy'] = False
        df.loc[df.is_alive, 'ps_ectopic_symptoms'].values[:] = 'none'
        df.loc[df.is_alive, 'ps_ep_unified_symptom_code'] = 0
        df.loc[df.is_alive, 'ps_multiple_pregnancy'] = False
        df.loc[df.is_alive, 'ps_total_miscarriages'] = 0
        df.loc[df.is_alive, 'ps_total_induced_abortion'] = 0
        df.loc[df.is_alive, 'ps_abortion_complication'].values[:] = 'none'
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

#        self.sim.modules['HealthSystem'].register_disease_module(self)

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props

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

        alive_and_preg = df.is_alive & df.is_pregnant
        gestation_in_days = self.sim.date - df.loc[alive_and_preg, 'date_of_last_pregnancy']
        gestation_in_weeks = gestation_in_days / np.timedelta64(1, 'W')
        df.loc[alive_and_preg, 'ps_gestational_age_in_weeks'] = gestation_in_weeks.astype('int64')
        logger.debug('updating gestational ages on date %s', self.sim.date)

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
        pass

    # ============================= PROGRESSION OF PREGNANCY DISEASES ==========================================
    # Progression of pregnancy diseases will live here
