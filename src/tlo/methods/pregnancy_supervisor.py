"""
Module responsible for supervision of pregnancy in the population including miscarriage, abortion, onset of labour and
onset of antenatal complications .
"""

import logging

import numpy as np
import pandas as pd
from pathlib import Path

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PregnancySupervisor(Module):
    """
    This module is responsible for supervision of pregnancy in the population including miscarriage, abortion, onset of labour and
    onset of antenatal complications """
    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    PARAMETERS = {
        'prob_pregnancy_factors': Parameter(
            Types.DATA_FRAME, 'Data frame containing probabilities of key outcomes/complications associate with the antenatal'
                        'period'),
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
    }

    PROPERTIES = {
        'ps_gestational_age': Property(Types.INT, 'current gestational age of this womans pregnancy in weeks'),
        'ps_ectopic_pregnancy': Property(Types.BOOL), 'Whether this womans pregnancy is ectopic'
        'ps_total_miscarriages': Property(Types.INT, 'the number of miscarriages a woman has experienced'),
        'ps_total_induced_abortion': Property(Types.INT, 'the number of induced abortions a woman has experienced'),
        'ps_still_birth_current_pregnancy': Property(Types.BOOL, 'whether this woman has experienced a still birth'),
        'ps_pre_eclampsia': Property(Types.BOOL, 'current gestational age of this womans pregnancy in weeks'),
        'ps_gest_htn': Property(Types.BOOL, 'current gestational age of this womans pregnancy in weeks'),
        'ps_gest_diab': Property(Types.BOOL, 'current gestational age of this womans pregnancy in weeks'),

    }

    TREATMENT_ID = ''

    def read_parameters(self, data_folder):
        """
        """
        params = self.parameters

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_PregnancySupervisor.xlsx',
                            sheet_name=None)

        params['prob_pregnancy_factors'] = dfd['prob_pregnancy_factors']
        dfd['prob_pregnancy_factors'].set_index('index', inplace=True)

        dfd['parameter_values'].set_index('parameter_name', inplace=True)
        params['rr_miscarriage_prevmiscarriage'] = dfd['parameter_values'].loc['rr_miscarriage_prevmiscarriage',
                                                                               'value']
        params['rr_miscarriage_35'] = dfd['parameter_values'].loc['rr_miscarriage_35', 'value']
        params['rr_miscarriage_3134'] = dfd['parameter_values'].loc['rr_miscarriage_3134', 'value']
        params['rr_miscarriage_grav4'] = dfd['parameter_values'].loc['rr_miscarriage_grav4', 'value']
        params['rr_pre_eclamp_nulip'] = dfd['parameter_values'].loc['rr_pre_eclamp_nulip', 'value']
        params['rr_pre_eclamp_prev_pe'] = dfd['parameter_values'].loc['rr_pre_eclamp_prev_pe', 'value']


    #        if 'HealthBurden' in self.sim.modules.keys():
#            params['daly_wt_haemorrhage_moderate'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=339)


    def initialise_population(self, population):

        df = population.props

        df.loc[df.sex == 'F', 'ps_gestational_age'] = 0
        df.loc[df.sex == 'F', 'ps_total_miscarriages'] = 0
        df.loc[df.sex == 'F', 'ps_total_induced_abortion'] = 0
        df.loc[df.sex == 'F', 'ps_still_birth_current_pregnancy'] = False
        df.loc[df.sex == 'F', 'ps_pre_eclampsia'] = False
        df.loc[df.sex == 'F', 'ps_gest_htn'] = False
        df.loc[df.sex == 'F', 'ps_gest_diab'] = False

    def initialise_simulation(self, sim):
        """Get ready for simulation start.
        """
        event = PregnancySupervisorEvent
        sim.schedule_event(event(self),
                           sim.date + DateOffset(days=0))

        event = PregnancySupervisorLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props

        if df.at[child_id, 'sex'] == 'F':
            df.at[child_id, 'ps_gestational_age'] = 0
            df.at[child_id, 'ps_total_miscarriages'] = 0
            df.at[child_id, 'ps_total_induced_abortion'] = 0
            df.at[child_id, 'ps_still_birth_current_pregnancy'] = False
            df.at[child_id, 'ps_pre_eclampsia'] = False
            df.at[child_id, 'ps_gest_htn'] = False
            df.at[child_id, 'ps_gest_diab'] = False


    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is PregnancySupervisor, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

#    def report_daly_values(self):

    #    logger.debug('This is mockitis reporting my health values')
    #    df = self.sim.population.props  # shortcut to population properties dataframe
    #    p = self.parameters


class PregnancySupervisorEvent(RegularEvent, PopulationScopeEventMixin):
    """ This event updates the gestational age of each pregnant woman every week and applies the risk of her developing
    any complications during her pregnancy
    """

    def __init__(self, module,):
        super().__init__(module, frequency=DateOffset(weeks=1))

    def apply(self, population):
        df = population.props
        params = self.module.parameters

    # =========================== GESTATIONAL AGE UPDATE FOR ALL PREGNANT WOMEN ========================================
        gestation_in_days = self.sim.date - df.loc[df.is_pregnant, 'date_of_last_pregnancy']
        gestation_in_weeks = gestation_in_days / np.timedelta64(1, 'W')
        df.loc[df.is_pregnant, 'ps_gestational_age'] = gestation_in_weeks.astype(int)

    # =============================================== ECTOPIC PREGNANCY ===============================================
        newly_pregnant_idx= df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 1)]

        # Apply incidence of ectopic pregnancy (turn off normal pregnancy variable)
        # at week one?- will need symptoms for care seeking etc? (unruptured vs ruptured?)

    # =========================== MULTIPLES ========================================
        # here we could apply risk of multiples? in first week?


    # ============================= DF SHORTCUTS ======================================================================
        misc_risk = params['prob_pregnancy_factors']['risk_miscarriage']
        ia_risk = params['prob_pregnancy_factors']['risk_abortion']
        sb_risk = params['prob_pregnancy_factors']['risk_still_birth']
        pe_risk = params['prob_pregnancy_factors']['risk_pre_eclamp']
        gh_risk = params['prob_pregnancy_factors']['risk_g_htn']
        gd_risk = params['prob_pregnancy_factors']['risk_g_diab']

    # =========================== MONTH 1 RISK APPLICATION ========================================
        # TODO: align moths correclty with weeks- unsure of best way
        # TODO:

        month_1_idx = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 4)]

        eff_prob_miscarriage = pd.Series(misc_risk[1], index=month_1_idx)
        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.ps_total_miscarriages >=1)] \
            *= params['rr_miscarriage_prevmiscarriage']
        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.age_years >= 35)] \
            *= params['rr_miscarriage_35']
        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.age_years >= 31 < 35)] \
            *= params['rr_miscarriage_3134']
#        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.hp_pre_eclampsia == 'none') &
        #        df.hp_prev_pre_eclamp] \
#            *= params['rr_miscarriage_grav4']

        random_draw = self.module.rng.random_sample(size=len(eff_prob_miscarriage))
        dfx = pd.concat((random_draw, eff_prob_miscarriage), axis=1)
        dfx.columns = ['random_draw', 'eff_prob_miscarriage']
        idx_mc = dfx.index[dfx.eff_prob_miscarriage > dfx.random_draw]

        df.loc[idx_mc, 'ps_total_miscarriages'] = +1  # Could this be a function
        df.loc[idx_mc, 'is_pregnant'] = False
        df.loc[idx_mc, 'ps_gestational_age'] = 0  # Complications?

    # =========================== MONTH 2 RISK APPLICATION ========================================
        month_2_idx = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 8)]

    # =========================== MONTH 3 RISK APPLICATION ========================================
        month_3_idx = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 12)]

    # =========================== MONTH 4 RISK APPLICATION ========================================
        month_4_idx = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 16)]

    # =========================== MONTH 5 RISK APPLICATION ========================================
        month_5_idx = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 20)]

    # =========================== MONTH 6 RISK APPLICATION ========================================
        month_6_idx = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 24)]

    # =========================== MONTH 7 RISK APPLICATION ========================================
        month_7_idx = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 28)]

    # =========================== MONTH 8 RISK APPLICATION ========================================
        month_8_idx = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 32)]

    # =========================== MONTH 9 RISK APPLICATION ========================================
        month_9_idx = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 36)]

    # =========================== MONTH 10 RISK APPLICATION ========================================
        month_10_idx = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 40)]


class PregnancySupervisorLoggingEvent(RegularEvent, PopulationScopeEventMixin):
        """Handles Pregnancy Supervision  logging"""

        def __init__(self, module):
            """schedule logging to repeat every 3 months
            """
            #    self.repeat = 3
            #    super().__init__(module, frequency=DateOffset(days=self.repeat))
            super().__init__(module, frequency=DateOffset(months=3))

        def apply(self, population):
            """Apply this event to the population.
            :param population: the current population
            """
            df = population.props
