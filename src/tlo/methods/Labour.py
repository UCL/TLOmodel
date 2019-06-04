"""
Documentation:
First draft of Labour module (Natural history)
"""
import logging

import pandas as pd

import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

from tlo.methods import demography, eclampsia_treatment, sepsis_treatment


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Labour (Module):

    """
    This module models labour and delivery and generates the properties for "complications" of delivery """

    # Read in data file
    file_path = '/Users/sejjj49/Dropbox/Thanzi la Onse/04 - Methods Repository/Method_Labour_Natural_History.xlsx'
    method_labour_data = pd.read_excel(file_path, sheet_name=None, header=0)
    labour_parameters = method_labour_data['parameters']

    PARAMETERS = {

        'prob_pregnancy': Parameter(
            Types.REAL, 'baseline probability of pregnancy'),  # DUMMY PARAMETER
        'prob_miscarriage': Parameter(
            Types.REAL, 'baseline probability of pregnancy loss before 28 weeks gestation'),
        'rr_miscarriage_prevmiscarriage': Parameter(
            Types.REAL, 'relative risk of pregnancy loss for women who have previously miscarried'),
        'rr_miscarriage_35': Parameter(
            Types.REAL, 'relative risk of pregnancy loss for women who is over 35 years old'),
        'rr_miscarriage_3134': Parameter(
            Types.REAL, 'relative risk of pregnancy loss for women who is between 31 and 34 years old'),
        'rr_miscarriage_grav4': Parameter(
            Types.REAL, 'relative risk of pregnancy loss for women who has a gravidity of greater than 4'),
        'prob_pl_ol': Parameter(
            Types.REAL, 'probability of a woman entering prolonged/obstructed labour'),
        'rr_PL_OL_nuliparity': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if they are nuliparous'),
        'rr_PL_OL_parity_3': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her parity is >3'),
        'rr_PL_OL_age_less20': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her age is less'
                        'than 20 years'),
        'rr_PL_OL_baby>3.5g': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her baby weighs more than 3.5'
                        'kilograms'),
        'rr_PL_OL_baby<1.5g': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her baby weighs less than 1.5'
                        'kilograms'),
        'rr_PL_OL_bmi<18': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her BMI is less than 18'),
        'rr_PL_OL_bmi>25': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her BMI is greater than 25'),
        'prob_ptl': Parameter (
            Types.REAL, 'probability of a woman entering labour at <37 weeks gestation'),
        'prob_an_eclampsia': Parameter(
            Types.REAL, 'probability of an eclamptic seizure during labour'),
        'prob_an_aph': Parameter(
            Types.REAL, 'probability of an antepartum haemorrhage during labour'),
        'prob_an_sepsis': Parameter(
            Types.REAL, 'probability of sepsis in labour'),
        'rr_an_eclampsia_30_34': Parameter(
            Types.REAL, 'relative risk of eclampsia for women ages between 30 and 34'),
        'rr_an_eclampsia_35': Parameter(
            Types.REAL, 'relative risk of eclampsia for women ages older than 35'),
        'rr_an_eclampsia_nullip': Parameter(
            Types.REAL, 'relative risk of eclampsia for women who have not previously delivered a child'),
        'rr_an_sepsis_anc_4': Parameter(
            Types.REAL, 'relative risk of sepsis for women who have attended greater than 4 ANC visits'),
        'rr_an_aph_noedu': Parameter(
            Types.REAL, 'relative risk of antepartum haemorrhage for women with education of primary level or lower'),
        'cfr_aph': Parameter(
            Types.REAL, 'case fatality rate for antepartum haemorrhage during labour'),
        'cfr_eclampsia': Parameter(
            Types.REAL, 'case fatality rate for eclampsia during labours'),
        'cfr_sepsis': Parameter(
            Types.REAL, 'case fatality rate for sepsis during labour'),
        'cfr_uterine_rupture': Parameter(
            Types.REAL, 'case fatality rate for uterine rupture in labour'),
        'prob_still_birth_aph': Parameter(
            Types.REAL, 'probability of a still birth following antepartum haemorrhage where the mother survives'),
        'prob_still_birth_aph_md': Parameter(
            Types.REAL, 'probability of a still birth following antepartum haemorrhage where the mother dies'),
        'prob_still_birth_sepsis': Parameter(
            Types.REAL, 'probability of a still birth following sepsis in labour where the mother survives'),
        'prob_still_birth_sepsis_md': Parameter(
            Types.REAL, 'probability of a still birth following sepsis in labour where the mother dies'),
        'prob_still_birth_ur': Parameter(
            Types.REAL, 'probability of a still birth following uterine rupture in labour where the mother survives'),
        'prob_still_birth_ur_md': Parameter(
            Types.REAL, 'probability of a still birth following uterine rupture in labour where the mother dies'),
        'prob_still_birth_eclampsia': Parameter(
            Types.REAL, 'probability of still birth following eclampsia in labour where the mother survives'),
        'prob_still_birth_eclampsia_md': Parameter(
            Types.REAL, 'probability of still birth following eclampsia in labour where the mother dies'),
        'prob_pn_eclampsia': Parameter(
            Types.REAL, 'probability of eclampsia following delivery for women who were in spotaneous unobstructed '
                        'labour'),
        'prob_pn_pph': Parameter(
            Types.REAL, 'probability of an postpartum haemorrhage following labour'),
        'prob_pn_sepsis': Parameter(
            Types.REAL, 'probability of sepsis following delivery'),
        'prob_sa_pph': Parameter(
            Types.REAL, 'probability of a postpartum haemorrhage following a spontaneous abortion before 28 weeks'
                        ' gestation'),
        'prob_sa_sepsis': Parameter(
            Types.REAL, 'probability of a sepsis  following a spontaneous abortion before 28 weeks gestation'),
        'cfr_pn_pph': Parameter(
            Types.REAL, 'case fatality rate for postpartum haemorrhages'),
        'cfr_pn_eclampsia': Parameter(
            Types.REAL, 'case fatality rate for eclampsia following delivery'),
        'cfr_pn_sepsis': Parameter(
            Types.REAL, 'case fatality rate for sepsis following delivery')
      }

    PROPERTIES = {

        'la_labour': Property(Types.CATEGORICAL, 'not in labour, Term labour, Preterm Labour, Post term labour',
                              categories=['not_in_labour', 'term_labour', 'preterm_labour',
                                          'post_term_labour']),
        'la_gestation_at_labour':Property(Types.INT, 'length of gestation in weeks at the point the woman enters '
                                                     'labour'),
        'la_still_birth_this_delivery': Property(Types.BOOL,'whether this womans most recent pregnancy has ended in a '
                                                            'stillbirth'),
        'la_miscarriage': Property(Types.INT, 'the number of miscarriages a woman has experienced'),
        'la_miscarriage_date': Property(Types.DATE, 'the date this woman has last experienced spontaneous miscarriage'),
        'la_parity': Property(Types.INT, 'total number of previous deliveries'),
        'la_previous_cs': Property(Types.INT, 'number of previous deliveries by caesarean section'),
        'la_previous_ptb': Property(Types.BOOL, 'whether the woman has had a previous preterm delivery for any of her'
                                                'previous deliveries'),
        'la_obstructed_labour':Property(Types.BOOL, 'whether this womans labour has become obstructed'),
        'la_aph': Property(Types.BOOL, 'whether the woman has experienced an antepartum haemorrhage in this delivery'),
        'la_uterine_rupture': Property(Types.BOOL, 'whether the woman has experienced uterine rupture in this delivery'),
        'la_sepsis': Property(Types.BOOL, 'whether the woman has developed sepsis associated with in this delivery'),
        'la_eclampsia': Property(Types.BOOL, 'whether the woman has experienced an eclamptic seizure in this delivery'),
        'la_pph': Property(Types.BOOL, 'whether the woman has experienced an postpartum haemorrhage in this delivery'),
        'la_died_in_labour': Property(Types.BOOL, 'whether the woman has died during this labour')
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters

        params['prob_pregnancy'] = 0.083  # Calculated from DHS 2010
        params['prob_miscarriage'] = 0.189
        params['rr_miscarriage_prevmiscarriage'] = 2.23
        params['rr_miscarriage_35'] = 4.02
        params['rr_miscarriage_3134'] = 2.13
        params['rr_miscarriage_grav4'] = 1.63
        params['prob_pl_ol'] = 0.06
        params['rr_PL_OL_nuliparity'] = 1.8
        params['rr_PL_OL_parity_3'] = 0.8
        params['rr_PL_OL_age_less20'] = 1.3
        params['prob_ptl'] = 0.18
        params['prob_an_eclampsia'] = 0.02
        params['prob_an_aph'] = 0.03
        params['prob_an_sepsis'] = 0.15
        params['prob_an_ur'] = 0.06
        params['rr_an_eclampsia_30_34'] = 1.4
        params['rr_an_eclampsia_35'] = 1.95
        params['rr_an_eclampsia_nullip'] = 2.04
        params['rr_an_sepsis_anc_4'] = 0.5
        params['rr_an_aph_noedu'] = 1.72
        params['cfr_aph'] = 0.05
        params['cfr_eclampsia'] = 0.03
        params['cfr_sepsis'] = 0.05
        params['cfr_uterine_rupture'] = 0.045
        params['prob_still_birth_aph'] = 0.35
        params['prob_still_birth_aph_md'] = 0.90
        params['prob_still_birth_sepsis'] = 0.25
        params['prob_still_birth_sepsis_md'] = 0.90
        params['prob_still_birth_ur'] = 0.75
        params['prob_still_birth_ur_md'] = 0.90
        params['prob_still_birth_eclampsia'] = 0.55
        params['prob_still_birth_eclampsia_md'] = 0.90
        params['prob_pn_eclampsia'] = 0.01
        params['prob_pn_pph'] = 0.01
        params['prob_pn_sepsis'] = 0.05
        params['prob_sa_pph'] = 0.12
        params['prob_sa_sepsis'] = 0.20
        params['cfr_pn_pph'] = 0.03
        params['cfr_pn_eclampsia'] = 0.05
        params['cfr_pn_sepsis'] = 0.04

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        df = population.props  # a shortcut to the data-frame storing data for individuals
        m = self
        rng = m.rng

    # ----------------------------------------- DEFAULTS----------------------------------------------------------------

        df['la_labour'] = 'not_in_labour'
        df['la_gestation_at_labour'] = 0
        df['la_miscarriage'] = 0
        df['la_miscarriage_date'] =pd.NaT
        df['la_still_birth_this_delivery'] = False
        df['la_partiy'] = 0
        df['la_previous_cs'] = 0
        df['la_previous_ptb'] = False
        df['la_due_date'] = pd.NaT
        df['la_obstructed_labour'] = False
        df['la_aph'] = False
        df['la_uterine_rupture'] = False
        df['la_sepsis'] = False
        df['la_eclampsia'] = False
        df['la_pph'] = False
        df['la_died_in_labour'] = False
    # -----------------------------------ASSIGN PREGNANCY AND DUE DATE AT BASELINE (DUMMY) ----------------------------

        # TODO: code in effect of contraception on likelihood of pregnancy at baseline (copy demography code?)

        women_idx = df.index[(df.age_years >= 15) & (df.age_years <= 49) & df.is_alive & (df.sex == 'F')]

        eff_prob_preg = pd.Series(m.prob_pregnancy, index=women_idx)

        random_draw = pd.Series(rng.random_sample(size=len(women_idx)),
                                index=df.index[(df.age_years >= 15) & (df.age_years <= 49) & df.is_alive
                                               & (df.sex == 'F')])

        dfx = pd.concat([eff_prob_preg, random_draw], axis=1)
        dfx.columns = ['eff_prob_pregnancy', 'random_draw']
        idx_pregnant = dfx.index[dfx.eff_prob_pregnancy > dfx.random_draw]
        df.loc[idx_pregnant, 'is_pregnant'] = True

# ---------------------------------    GESTATION AT BASELINE  ---------------------------------------------------------

        # Assigns a date of last pregnancy and due date for women who are pregnant at baseline

        # TODO: ensure women at baseline can go into preterm labour or be overdue

        pregnant_idx = df.index[df.is_pregnant & df.is_alive]

        random_draw = pd.Series(rng.random_sample(size=len(pregnant_idx)),
                                index=df.index[df.is_pregnant & df.is_alive])

        simdate = pd.Series(self.sim.date, index=pregnant_idx)
        dfx = pd.concat((simdate, random_draw), axis=1)
        dfx.columns = ['simdate', 'random_draw']
        dfx['gestational_age'] = (39 - 39 * dfx.random_draw)
        dfx['la_conception_date'] = dfx['simdate'] - pd.to_timedelta(dfx['gestational_age'], unit='w')
        dfx['due_date_mth'] = 39 - dfx['gestational_age']
        dfx['due_date'] = dfx['simdate'] + pd.to_timedelta(dfx['due_date_mth'], unit='w')

        df.loc[pregnant_idx, 'date_of_last_pregnancy'] = dfx.la_conception_date
        df.loc[pregnant_idx, 'due_date'] = dfx.due_date

#  ----------------------------ASSIGNING PARITY AT BASELINE (DUMMY)-----------------------------------------------------

        women_parity_1524_idx = df.index[(df.age_years >= 15) & (df.age_years <= 24) & (df.is_alive == True)
                                    & (df.sex == 'F')]

        baseline_p = pd.Series(0, index=women_parity_1524_idx)

        random_draw2 = pd.Series(self.rng.choice(range(0, 5), p=[0.40, 0.35, 0.15, 0.06, 0.04],
                                                 size=len(women_parity_1524_idx)),
                                 index=df.index[(df.age_years >= 15) & (df.age_years <= 24)
                                                & df.is_alive & (df.sex == 'F')])

        dfi = pd.concat([baseline_p, random_draw2], axis=1)
        dfi.columns = ['baseline_p', 'random_draw2']
        idx_parity = dfi.index[dfi.baseline_p < dfi.random_draw2]
        df.loc[idx_parity, 'la_parity'] = random_draw2

        women_parity_2540_idx = df.index[(df.age_years >= 25) & (df.age_years <= 40) & (df.is_alive == True)
                                         & (df.sex == 'F')]

        baseline_p = pd.Series(0, index=women_parity_2540_idx)

        random_draw = pd.Series(self.rng.choice(range(0, 6), p=[0.05, 0.15, 0.30, 0.20, 0.2, 0.1],
                                                size=len(women_parity_2540_idx)), index=df.index[(df.age_years >= 25) &
                                                                                                 (df.age_years <= 40)
                                                                                                 & (df.is_alive == True)
                                                                                                 & (df.sex == 'F')])

        dfi = pd.concat([baseline_p, random_draw], axis=1)
        dfi.columns = ['baseline_p', 'random_draw']
        idx_parity = dfi.index[dfi.baseline_p < dfi.random_draw]
        df.loc[idx_parity, 'la_parity'] = random_draw

        women_parity_4149_idx = df.index[(df.age_years >= 41) & (df.age_years <= 49) & (df.is_alive == True)
                                         & (df.sex == 'F')]

        baseline_p = pd.Series(0, index=women_parity_4149_idx)

        random_draw = pd.Series(self.rng.choice(range(0, 7), p=[0.05, 0.10, 0.25, 0.30, 0.25, 0.03, 0.02],
                                                size=len(women_parity_4149_idx)), index=df.index[(df.age_years >= 41)
                                                                                                 & (df.age_years <= 49)
                                                                                                 & (df.is_alive == True)
                                                                                                 & (df.sex == 'F')])

        dfi = pd.concat([baseline_p, random_draw], axis=1)
        dfi.columns = ['baseline_p', 'random_draw']
        idx_parity = dfi.index[dfi.baseline_p < dfi.random_draw]
        df.loc[idx_parity, 'la_parity'] = random_draw

#   ------------------------------ ASSIGN PREVIOUS CS AT BASELINE -----------------------------------------------

        women_para1_idx = df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') & (df.la_parity == 1)]
        women_para2_idx = df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') & (df.la_parity >= 2)]

        baseline_cs1 = pd.Series(df.la_previous_cs, index=women_para1_idx)
        baseline_cs2 = pd.Series(df.la_previous_cs, index=women_para2_idx)

        random_draw1 = pd.Series(self.rng.choice(range(0, 2), p=[0.91, 0.09], size=len(women_para1_idx)),
                               index=df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') &
                                              (df.la_parity == 1)])

        random_draw2 = pd.Series(self.rng.choice(range(0, 3), p=[0.90, 0.07, 0.03], size=len(women_para2_idx)),
                                 index=df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') &
                                                (df.la_parity >= 2)])

        dfx = pd.concat([baseline_cs1, random_draw1], axis=1)
        dfx.columns = ['baseline_cs', 'random_draw']
        idx_prev_cs = dfx.index[dfx.baseline_cs < dfx.random_draw]
        df.loc[idx_prev_cs, 'la_previous_cs'] = random_draw1

        dfx = pd.concat([baseline_cs2, random_draw2], axis=1)
        dfx.columns = ['baseline_cs', 'random_draw']
        idx_prev_cs = dfx.index[dfx.baseline_cs < dfx.random_draw]
        df.loc[idx_prev_cs, 'la_previous_cs'] = random_draw2

        # ------------------------------ ASSIGN PREVIOUS PTB AT BASELINE ----------------------------------------------


        # TODO: apply risk factors for preterm birth at baseline

        women_para1_nocs_idx = df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') & (df.la_parity == 1) &
                                        (df.la_previous_cs ==0)]
        women_para2_idx = df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') & (df.la_parity >= 2)]

        baseline_ptb = pd.Series(m.prob_ptl, index=women_para1_nocs_idx)
        baseline_ptb_p2 = pd.Series(m.prob_ptl, index=women_para2_idx)

        random_draw = pd.Series(rng.random_sample(size=len(women_para1_nocs_idx)),
                                index=df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') &
                                               (df.la_parity == 1) & (df.la_previous_cs == 0)])
        random_draw2 = pd.Series(rng.random_sample(size=len(women_para2_idx)),
                                index=df.index[ (df.age_years >= 15) & df.is_alive & (df.sex == 'F') &
                                                (df.la_parity >= 2)])

        dfx = pd.concat([baseline_ptb, random_draw], axis=1)
        dfx.columns = ['baseline_ptb', 'random_draw']
        idx_prev_ptb = dfx.index[dfx.baseline_ptb > dfx.random_draw]
        df.loc[idx_prev_ptb, 'la_previous_ptb'] = True

        dfx = pd.concat([baseline_ptb_p2, random_draw2], axis=1)
        dfx.columns = ['baseline_ptb_p2', 'random_draw2']
        idx_prev_ptb = dfx.index[dfx.baseline_ptb_p2 > dfx.random_draw2]
        df.loc[idx_prev_ptb, 'la_previous_ptb'] = True

    def baseline_labour_scheduler(self, population):
        """Schedules labour for women who are pregnant at baseline"""

        # The possibility of miscarriage and labour are scheduled for women who are pregnant at baseline

        df = population.props

        pregnant_baseline = df.index[(df.is_pregnant == True) & df.is_alive]
        for person in pregnant_baseline:
            gestation_date = df.at[person, 'due_date'] - (df.at[person, 'date_of_last_pregnancy'])
            gestation_weeks = gestation_date / np.timedelta64(1, 'W')
            if gestation_weeks < 28:
                self.sim.schedule_event(Labour.MiscarriageEvent(self.sim.modules['Labour'], person,
                                                                cause='miscarriage event'), self.sim.date)

        for person in pregnant_baseline:
            if df.at[person, 'is_pregnant']:
                scheduled_labour_date = df.at[person, 'due_date']
                labour = LabourEvent(self, individual_id=person, cause='labour')
                self.sim.schedule_event(labour, scheduled_labour_date)

    def initialise_simulation(self, sim):

        event = LabourLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

        self.baseline_labour_scheduler(sim.population)

    def on_birth(self, mother_id, child_id):

        df = self.sim.population.props

        df.at[child_id, 'la_labour'] = 'not_in_labour'
        df.at[child_id, 'la_gestation_at_labour'] = 0
        df.at[child_id, 'la_miscarriage'] = 0
        df.at[child_id, 'la_miscarriage_date'] = pd.NaT
        df.at[child_id, 'la_still_birth_this_delivery'] = False
        df.at[child_id, 'la_parity'] = 0
        df.at[child_id, 'la_previous_cs'] = 0
        df.at[child_id, 'la_previous_ptb'] = False
        df.at[child_id, 'la_obstructed_labour'] = False
        df.at[child_id, 'la_aph'] = False
        df.at[child_id, 'la_uterine_rupture'] = False
        df.at[child_id, 'la_sepsis'] = False
        df.at[child_id, 'la_eclampsia'] = False
        df.at[child_id, 'la_pph'] = False
        df.at[child_id, 'la_died_in_labour'] = False


class MiscarriageEvent(Event, IndividualScopeEventMixin):
    """On conception event that applies a cumulative risk of early pregnancy loss to women """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, population):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        poss_miscarriage_idx = df.index[(df.date_of_last_pregnancy == self.sim.date)]

        eff_prob_miscarriage = pd.Series(params['prob_miscarriage'], index=poss_miscarriage_idx)

        eff_prob_miscarriage.loc[(df.is_alive & (df.is_pregnant == True) & (df.la_labour == 'not_in_labour') &
                                          (df.la_miscarriage >= 1) & (df.la_parity + df.la_miscarriage <= 3) &
                                          (df.age_years <= 30))] *= params['rr_miscarriage_prevmiscarriage']

        eff_prob_miscarriage.loc[(df.is_alive & (df.is_pregnant == True) & (df.la_labour == 'not_in_labour') &
                                          (df.la_miscarriage == 0) & (df.age_years >= 31) & (df.age_years <= 34))]\
            *= params['rr_miscarriage_3134']
        eff_prob_miscarriage.loc[(df.is_alive & (df.is_pregnant == True) & (df.la_labour == 'not_in_labour') &
                                            (df.la_miscarriage == 0) & (df.age_years >= 35))]\
            *= params['rr_miscarriage_35']
        eff_prob_miscarriage.loc[(df.is_alive & (df.is_pregnant == True) & (df.la_labour == 'not_in_labour') &
                                           (df.la_miscarriage == 0) & (df.la_parity >= 4) & (df.age_years <= 30))]\
            *= params['rr_miscarriage_grav4']

        random_draw = pd.Series(self.sim.rng.random_sample(size=len(poss_miscarriage_idx)),
                                index=df.index[df.date_of_last_pregnancy == self.sim.date])

        dfx = pd.concat([eff_prob_miscarriage, random_draw], axis=1)
        dfx.columns = ['eff_prob_miscarriage', 'random_draw']
        idx_miscarriage = dfx.index[dfx.eff_prob_miscarriage > dfx.random_draw]

        # For woman who experience miscarriage, pregnancy status is reset, number of miscarriages recorded including
        # date and woman is scheduled to move to postpartum event to determine if she experiences complications

        df.loc[idx_miscarriage, 'is_pregnant'] = False
        df.loc[idx_miscarriage, 'la_miscarriage'] += 1
        df.loc[idx_miscarriage, 'due_date'] = pd.NaT
        df.loc[idx_miscarriage, 'la_miscarriage_date'] = self.sim.date

        new_miscarriage_idx = df.index[(df.la_miscarriage_date == self.sim.date)]
        for mother_id in new_miscarriage_idx:
            self.sim.schedule_event(PostpartumLabourEvent(self.module, mother_id,
                                                             cause='post miscarriage'), self.sim.date)


class LabourEvent(Event, IndividualScopeEventMixin):

    """Moves a pregnant woman into labour/spontaneous abortion based on gestation distribution """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

# ===================================== LABOUR STATE  ==================================================================

        gestation_date = df.at[individual_id, 'due_date'] - (df.at[individual_id, 'date_of_last_pregnancy'])
        gestation_weeks = gestation_date / np.timedelta64(1, 'W')

        if df.at[individual_id,'is_pregnant']:
            df.at[individual_id,'la_gestation_at_labour'] = gestation_weeks
            if df.at[individual_id, 'la_gestation_at_labour'] > 36 < 42:
                df.at[individual_id, 'la_labour'] = "term_labour"

            # TODO: apply risk factors for preterm birth

            elif df.at[individual_id, 'la_gestation_at_labour'] > 27 < 37:
                df.at[individual_id, 'la_labour'] = "preterm_labour"
                df.at[individual_id, 'la_previous_ptb'] = True

            elif df.at[individual_id, 'la_gestation_at_labour'] >= 42:
                df.at[individual_id, 'la_labour'] = "post_term_labour"

# ================================== COMPLICATIONS FOR TERM LABOUR ===============================================

        # First determine if this woman's labour will be obstructed

        in_labour = df.index[(df.due_date == self.sim.date) & (df.is_pregnant == True) & (df.la_labour == 'term_labour')]

        eff_prob_plol = pd.Series(params['prob_pl_ol'], index=in_labour)

        eff_prob_plol.loc[((df.due_date == self.sim.date) & (df.la_parity == 0) & (df.age_years >= 21))] *= \
            params['rr_PL_OL_nuliparity']
        eff_prob_plol.loc[((df.due_date == self.sim.date) & (df.la_parity >= 3) & (df.age_years >= 21))] *= \
            params['rr_PL_OL_parity_3']
        eff_prob_plol.loc[((df.due_date == self.sim.date) & (df.la_parity >= 1) & (df.la_parity < 3) &
                           (df.age_years < 20))] *= params['rr_PL_OL_age_less20']

        random_draw = pd.Series(self.sim.rng.random_sample(size=len(in_labour)),
                                index=df.index[(df.due_date == self.sim.date) & (df.is_pregnant == True)
                                               & (df.la_labour == 'term_labour')])

        dfx = pd.concat([eff_prob_plol, random_draw], axis=1)
        dfx.columns = ['eff_prob_plol', 'random_draw']
        idx_plol = dfx.index[dfx.eff_prob_plol > dfx.random_draw]
        df.loc[idx_plol, 'la_obstructed_labour'] = True

        # Following this we determine if the woman will experience any futher complications in labour

        tl_women = df.index[(df.la_labour == 'term_labour') & (df.due_date == self.sim.date)]

        eff_prob_eclamp = pd.Series(params['prob_an_eclampsia'], index=tl_women)
        eff_prob_aph = pd.Series(params['prob_an_aph'], index=tl_women)
        eff_prob_sepsis = pd.Series(params['prob_an_sepsis'], index=tl_women)
        eff_prob_ur= pd.Series(params['prob_an_ur'], index=tl_women)

        # Eclampsia risk factors

        eff_prob_eclamp.loc[((df.la_labour == 'term_labour') & (df.due_date == self.sim.date) &
                             (df.la_parity == 0) & (df.age_years <= 29))] *= \
            params['rr_an_eclampsia_nullip']
        eff_prob_eclamp.loc[((df.la_labour == 'term_labour') & (df.due_date == self.sim.date) &
                             (df.la_parity >= 1) & (df.age_years >= 35))] *= \
            params['rr_an_eclampsia_35']
        eff_prob_eclamp.loc[((df.la_labour == 'term_labour') & (df.due_date == self.sim.date) &
                             (df.la_parity >= 1) & (df.age_years >= 30) & (df.age_years <= 34))]\
            *= params['rr_an_eclampsia_30_34']

        # APH risk factors

        eff_prob_aph.loc[((df.la_labour == 'term_labour') & (df.due_date == self.sim.date) &
                          (df.li_ed_lev == 1))] *= params['rr_an_aph_noedu']

        # Todo: Uterine rupture risk factors
        # Todo: Sepsis risk factors

        random_draw = pd.Series(self.sim.rng.random_sample(size=len(tl_women)),
                                index=df.index[(df.la_labour == 'term_labour') &
                                               (df.due_date ==self.sim.date)])
        random_draw2 = pd.Series(self.sim.rng.random_sample(size=len(tl_women)),
                                index=df.index[(df.la_labour == 'term_labour') &
                                               (df.due_date == self.sim.date)])
        random_draw3 = pd.Series(self.sim.rng.random_sample(size=len(tl_women)),
                                 index=df.index[(df.la_labour == 'term_labour') &
                                                (df.due_date == self.sim.date)])
        random_draw4 = pd.Series(self.sim.rng.random_sample(size=len(tl_women)),
                                 index=df.index[(df.la_labour == 'term_labour') &
                                                (df.due_date == self.sim.date)])

        dfx = pd.concat([eff_prob_eclamp, random_draw, eff_prob_aph, random_draw2, eff_prob_sepsis, random_draw3,
                         eff_prob_ur, random_draw4],
                        axis=1)
        dfx.columns = ['eff_prob_eclamp', 'random_draw', 'eff_prob_aph', 'random_draw2', 'eff_prob_sepsis',
                       'random_draw3','eff_prob_ur','random_draw4']

        idx_eclamp = dfx.index[dfx.eff_prob_eclamp > dfx.random_draw]
        idx_aph = dfx.index[dfx.eff_prob_aph > dfx.random_draw2]
        idx_sepsis = dfx.index[dfx.eff_prob_sepsis > dfx.random_draw3]
        idx_ur = dfx.index[dfx.eff_prob_sepsis > dfx.random_draw3]
        df.loc[idx_eclamp, 'la_eclampsia'] = True
        df.loc[idx_aph, 'la_aph'] = True
        df.loc[idx_sepsis, 'la_sepsis'] = True
        df.loc[idx_ur, 'la_uterine_rupture'] = True

        # Schedule treatment for women who develop eclampsia
        # Apply probability of treatment? Or wait for healthsystem

        for individual_id in idx_eclamp:
            self.sim.schedule_event(eclampsia_treatment.EclampsiaTreatmentEvent(self.sim.modules['EclampsiaTreatment'],
                                                                                individual_id, cause='eclampsia'),
                                    self.sim.date)

        for individual_id in idx_sepsis:
            self.sim.schedule_event(sepsis_treatment.SepsisTreatmentEvent(self.sim.modules['SepsisTreatment'],
                                                                                individual_id, cause='Sepsis'),
                                    self.sim.date)

    # ============================ COMPLICATIONS FOR PRETERM LABOUR ====================================================

        # First it is determine if this woman's labour will be obstructed
        # Todo: impact of preterm labour on risk of obstructed labour

        in_labour = df.index[(df.due_date == self.sim.date) & (df.is_pregnant == True) &
                             (df.la_labour == 'preterm_labour')]
        eff_prob_plol = pd.Series(params['prob_pl_ol'], index=in_labour)

        eff_prob_plol.loc[((df.due_date == self.sim.date) & (df.la_parity == 0) & (df.age_years >= 21))] *= \
                params['rr_PL_OL_nuliparity']
        eff_prob_plol.loc[((df.due_date == self.sim.date) & (df.la_parity >= 3) & (df.age_years >= 21))] *= \
                params['rr_PL_OL_parity_3']
        eff_prob_plol.loc[((df.due_date == self.sim.date) & (df.la_parity >= 1) & (df.la_parity < 3) &
                               (df.age_years < 20))] *= params['rr_PL_OL_age_less20']

        random_draw = pd.Series(self.sim.rng.random_sample(size=len(in_labour)),
                                    index=df.index[(df.due_date == self.sim.date) & (df.is_pregnant == True)
                                                   & (df.la_labour == 'preterm_labour')])

        dfx = pd.concat([eff_prob_plol, random_draw], axis=1)
        dfx.columns = ['eff_prob_plol', 'random_draw']
        idx_plol = dfx.index[dfx.eff_prob_plol > dfx.random_draw]
        df.loc[idx_plol, 'la_obstructed_labour'] = True

        #  TODO: Apply risk factors for complications in preterm labour

        ptl_women = df.index[(df.la_labour == 'preterm_labour') & (df.due_date == self.sim.date)]
        eff_prob_eclamp = pd.Series(params['prob_an_eclampsia'], index=ptl_women)
        eff_prob_aph = pd.Series(params['prob_an_aph'], index=ptl_women)
        eff_prob_sepsis = pd.Series(params['prob_an_sepsis'], index=ptl_women)
        eff_prob_ur= pd.Series(params['prob_an_ur'], index=ptl_women)

        random_draw = pd.Series(self.sim.rng.random_sample(size=len(ptl_women)),
                                index=df.index[(df.la_labour == 'preterm_labour') &
                                               (df.due_date == self.sim.date)])
        random_draw2 = pd.Series(self.sim.rng.random_sample(size=len(ptl_women)),
                                 index=df.index[(df.la_labour == 'preterm_labour') &
                                                (df.due_date == self.sim.date)])
        random_draw3 = pd.Series(self.sim.rng.random_sample(size=len(ptl_women)),
                                 index=df.index[(df.la_labour == 'preterm_labour') &
                                                (df.due_date == self.sim.date)])
        random_draw4 = pd.Series(self.sim.rng.random_sample(size=len(ptl_women)),
                                 index=df.index[(df.la_labour == 'preterm_labour') &
                                                (df.due_date == self.sim.date)])

        dfx = pd.concat([eff_prob_eclamp, random_draw, eff_prob_aph, random_draw2, eff_prob_sepsis, random_draw3,
                         eff_prob_ur, random_draw4],
                        axis=1)
        dfx.columns = ['eff_prob_eclamp', 'random_draw', 'eff_prob_aph', 'random_draw2', 'eff_prob_sepsis',
                       'random_draw3', 'eff_prob_ur', 'random_draw4']
        idx_eclamp = dfx.index[dfx.eff_prob_eclamp > dfx.random_draw]
        idx_aph = dfx.index[dfx.eff_prob_aph > dfx.random_draw2]
        idx_sepsis = dfx.index[dfx.eff_prob_sepsis > dfx.random_draw3]
        idx_ur = dfx.index[dfx.eff_prob_ur > dfx.random_draw3]
        df.loc[idx_eclamp, 'la_eclampsia'] = True
        df.loc[idx_aph, 'la_aph'] = True
        df.loc[idx_sepsis, 'la_sepsis'] = True
        df.loc[idx_ur, 'la_uterine_rupture'] =True

    # ============================ COMPLICATIONS FOR POST TERM LABOUR ==================================================

        # First it is determine if this woman's labour will be obstructed
        # Todo: impact of preterm labour on risk of obstructed labour

        in_labour = df.index[(df.due_date == self.sim.date) & (df.is_pregnant == True) &
                             (df.la_labour == 'post_term_labour')]
        eff_prob_plol = pd.Series(params['prob_pl_ol'], index=in_labour)

        eff_prob_plol.loc[((df.due_date == self.sim.date) & (df.la_parity == 0) & (df.age_years >= 21))] *= \
            params['rr_PL_OL_nuliparity']
        eff_prob_plol.loc[((df.due_date == self.sim.date) & (df.la_parity >= 3) & (df.age_years >= 21))] *= \
            params['rr_PL_OL_parity_3']
        eff_prob_plol.loc[((df.due_date == self.sim.date) & (df.la_parity >= 1) & (df.la_parity < 3) &
                           (df.age_years < 20))] *= params['rr_PL_OL_age_less20']

        random_draw = pd.Series(self.sim.rng.random_sample(size=len(in_labour)),
                                index=df.index[(df.due_date == self.sim.date) & (df.is_pregnant == True) &
                                               (df.la_labour == 'post_term_labour')])

        dfx = pd.concat([eff_prob_plol, random_draw], axis=1)
        dfx.columns = ['eff_prob_plol', 'random_draw']
        idx_plol = dfx.index[dfx.eff_prob_plol > dfx.random_draw]
        df.loc[idx_plol, 'la_obstructed_labour'] = True

        #  TODO: Apply risk factors for complications in preterm labour

        potl_women = df.index[(df.la_labour == 'post_term_labour') & (df.due_date == self.sim.date)]
        eff_prob_eclamp = pd.Series(params['prob_an_eclampsia'], index=potl_women)
        eff_prob_aph = pd.Series(params['prob_an_aph'], index=potl_women)
        eff_prob_sepsis = pd.Series(params['prob_an_sepsis'], index=potl_women)
        eff_prob_ur = pd.Series(params['prob_an_ur'], index=potl_women)

        random_draw = pd.Series(self.sim.rng.random_sample(size=len(potl_women)),
                                index=df.index[(df.la_labour == 'post_term_labour') &
                                               (df.due_date == self.sim.date)])
        random_draw2 = pd.Series(self.sim.rng.random_sample(size=len(potl_women)),
                                 index=df.index[(df.la_labour == 'post_term_labour') &
                                                (df.due_date == self.sim.date)])
        random_draw3 = pd.Series(self.sim.rng.random_sample(size=len(potl_women)),
                                 index=df.index[(df.la_labour == 'post_term_labour') &
                                                (df.due_date == self.sim.date)])
        random_draw4 = pd.Series(self.sim.rng.random_sample(size=len(potl_women)),
                                 index=df.index[(df.la_labour == 'post_term_labour') &
                                                (df.due_date == self.sim.date)])

        dfx = pd.concat([eff_prob_eclamp, random_draw, eff_prob_aph, random_draw2, eff_prob_sepsis, random_draw3,
                         eff_prob_ur, random_draw4],
                        axis=1)
        dfx.columns = ['eff_prob_eclamp', 'random_draw', 'eff_prob_aph', 'random_draw2', 'eff_prob_sepsis',
                       'random_draw3', 'eff_prob_ur', 'random_draw4']

        idx_eclamp = dfx.index[dfx.eff_prob_eclamp > dfx.random_draw]
        idx_aph = dfx.index[dfx.eff_prob_aph > dfx.random_draw2]
        idx_sepsis = dfx.index[dfx.eff_prob_sepsis > dfx.random_draw3]
        idx_ur = dfx.index[dfx.eff_prob_ur > dfx.random_draw3]
        df.loc[idx_eclamp, 'la_eclampsia'] = True
        df.loc[idx_aph, 'la_aph'] = True
        df.loc[idx_sepsis, 'la_sepsis'] = True
        df.loc[idx_ur, 'la_uterine_rupture'] = True


class PostpartumLabourEvent(Event, IndividualScopeEventMixin):

    """applies probability of postpartum complications to women who have just delivered """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

# ============================== POSTPARTUM COMPLICATIONS FOLLOWING TERM LABOUR =======================================

        tl_women_pn = df.index[(df.la_labour == 'term_labour') & (df.is_alive == True)&
                                    (df.due_date == self.sim.date - DateOffset(days=2))]

        eff_prob_eclamp = pd.Series(params['prob_pn_eclampsia'], index=tl_women_pn)
        eff_prob_pph = pd.Series(params['prob_pn_pph'], index=tl_women_pn)
        eff_prob_sepsis = pd.Series(params['prob_pn_sepsis'], index=tl_women_pn)

        random_draw = pd.Series(self.sim.rng.random_sample(size=len(tl_women_pn)),
                                    index=df.index[(df.la_labour == 'term_labour') & (df.is_alive == True)&
                                    (df.due_date == self.sim.date - DateOffset(days=2))])
        random_draw2 = pd.Series(self.sim.rng.random_sample(size=len(tl_women_pn)),
                                     index=df.index[(df.la_labour == 'term_labour') & (df.is_alive == True)&
                                    (df.due_date == self.sim.date - DateOffset(days=2))])
        random_draw3 = pd.Series(self.sim.rng.random_sample(size=len(tl_women_pn)),
                                     index=df.index[(df.la_labour == 'term_labour') & (df.is_alive == True)&
                                    (df.due_date == self.sim.date - DateOffset(days=2))])

        dfx = pd.concat([eff_prob_eclamp, random_draw, eff_prob_pph, random_draw2, eff_prob_sepsis, random_draw3],
                            axis=1)
        dfx.columns = ['eff_prob_eclamp', 'random_draw', 'eff_prob_pph', 'random_draw2', 'eff_prob_sepsis',
                           'random_draw3']

        idx_eclamp = dfx.index[dfx.eff_prob_eclamp > dfx.random_draw]
        idx_pph = dfx.index[dfx.eff_prob_pph > dfx.random_draw2]
        idx_sepsis = dfx.index[dfx.eff_prob_sepsis > dfx.random_draw3]
        df.loc[idx_eclamp, 'la_eclampsia'] = True
        df.loc[idx_pph, 'la_pph'] = True
        df.loc[idx_sepsis, 'la_sepsis'] = True

        for individual_id in idx_eclamp:
            self.sim.schedule_event(eclampsia_treatment.EclampsiaTreatmentEventPostPartum(self.sim.modules['EclampsiaTreatment'],
                                                                                          individual_id, cause='eclampsia'),
                                    self.sim.date)

        # ============================ POSTPARTUM COMPLICATIONS FOLLOWING POST TERM  LABOUR ====================================

            # TODO: risk factors
        potl_women_pn = df.index[(df.la_labour == 'post_term_labour') & (df.is_alive == True) &
                                    (df.due_date == self.sim.date - DateOffset(days=2))]

        eff_prob_eclamp = pd.Series(params['prob_pn_eclampsia'], index=potl_women_pn)
        eff_prob_pph = pd.Series(params['prob_pn_pph'], index=potl_women_pn)
        eff_prob_sepsis = pd.Series(params['prob_pn_sepsis'], index=potl_women_pn)

        random_draw = pd.Series(self.sim.rng.random_sample(size=len(potl_women_pn)),
                                    index=df.index[
                                        (df.la_labour == 'post_term_labour') & (df.is_alive == True) &
                                        (df.due_date == self.sim.date - DateOffset(days=2))])
        random_draw2 = pd.Series(self.sim.rng.random_sample(size=len(potl_women_pn)),
                                     index=df.index[
                                         (df.la_labour == 'post_term_labour') & (df.is_alive == True) &
                                         (df.due_date == self.sim.date - DateOffset(days=2))])
        random_draw3 = pd.Series(self.sim.rng.random_sample(size=len(potl_women_pn)),
                                     index=df.index[
                                         (df.la_labour == 'post_term_labour') & (df.is_alive == True) &
                                         (df.due_date == self.sim.date - DateOffset(days=2))])

        dfx = pd.concat([eff_prob_eclamp, random_draw, eff_prob_pph, random_draw2, eff_prob_sepsis, random_draw3],
                            axis=1)
        dfx.columns = ['eff_prob_eclamp', 'random_draw', 'eff_prob_pph', 'random_draw2', 'eff_prob_sepsis',
                           'random_draw3']

        idx_eclamp = dfx.index[dfx.eff_prob_eclamp > dfx.random_draw]
        idx_pph = dfx.index[dfx.eff_prob_pph > dfx.random_draw2]
        idx_sepsis = dfx.index[dfx.eff_prob_sepsis > dfx.random_draw3]
        df.loc[idx_eclamp, 'la_eclampsia'] = True
        df.loc[idx_pph, 'la_pph'] = True
        df.loc[idx_sepsis, 'la_sepsis'] = True

# ============================ POSTPARTUM COMPLICATIONS FOLLOWING PRETERM LABOUR =======================================

        ptl_women_pn = df.index[(df.la_labour == 'preterm_labour') & (df.is_alive == True) &
                                     (df.due_date == self.sim.date - DateOffset(days=2))]

        eff_prob_eclamp = pd.Series(params['prob_pn_eclampsia'], index=ptl_women_pn)
        eff_prob_pph = pd.Series(params['prob_pn_pph'], index=ptl_women_pn)
        eff_prob_sepsis = pd.Series(params['prob_pn_sepsis'], index=ptl_women_pn)

        random_draw = pd.Series(self.sim.rng.random_sample(size=len(ptl_women_pn)),
                                    index=df.index[
                                        (df.la_labour == 'preterm_labour') & (df.is_alive == True) &
                                        (df.due_date == self.sim.date - DateOffset(days=2))])
        random_draw2 = pd.Series(self.sim.rng.random_sample(size=len(ptl_women_pn)),
                                     index=df.index[
                                         (df.la_labour == 'preterm_labour') & (df.is_alive == True) &
                                         (df.due_date == self.sim.date - DateOffset(days=2))])
        random_draw3 = pd.Series(self.sim.rng.random_sample(size=len(ptl_women_pn)),
                                     index=df.index[
                                         (df.la_labour == 'preterm_labour') & (df.is_alive == True) &
                                         (df.due_date == self.sim.date - DateOffset(days=2))])

        dfx = pd.concat([eff_prob_eclamp, random_draw, eff_prob_pph, random_draw2, eff_prob_sepsis, random_draw3],
                            axis=1)
        dfx.columns = ['eff_prob_eclamp', 'random_draw', 'eff_prob_pph', 'random_draw2', 'eff_prob_sepsis',
                           'random_draw3']

        idx_eclamp = dfx.index[dfx.eff_prob_eclamp > dfx.random_draw]
        idx_pph = dfx.index[dfx.eff_prob_pph > dfx.random_draw2]
        idx_sepsis = dfx.index[dfx.eff_prob_sepsis > dfx.random_draw3]
        df.loc[idx_eclamp, 'la_eclampsia'] = True
        df.loc[idx_pph, 'la_pph'] = True
        df.loc[idx_sepsis, 'la_sepsis'] = True


# ==============================  COMPLICATIONS FOLLOWING SPONTANEOUS MISCARRIAGE ======================================

        # TODO: consider stage of pregnancy loss and its impact on liklihood of complications i.e retained product

        post_miscarriage = df.index[ (df.is_alive == True) & (df.la_miscarriage_date == self.sim.date)]

        eff_prob_pph = pd.Series(params['prob_sa_pph'], index=post_miscarriage)
        eff_prob_sepsis = pd.Series(params['prob_sa_sepsis'], index=post_miscarriage)

        random_draw = pd.Series(self.sim.rng.random_sample(size=len(post_miscarriage)),
                                    index=df.index[((df.is_alive == True) & (df.la_miscarriage_date == self.sim.date))])

        random_draw2 = pd.Series(self.sim.rng.random_sample(size=len(post_miscarriage)),
                                    index=df.index[((df.is_alive == True) & (df.la_miscarriage_date == self.sim.date))])

        dfx = pd.concat([eff_prob_pph, random_draw, eff_prob_sepsis, random_draw2],
                            axis=1)
        dfx.columns = ['eff_prob_pph', 'random_draw', 'eff_prob_sepsis', 'random_draw2']

        idx_pph = dfx.index[dfx.eff_prob_pph > dfx.random_draw]
        idx_sepsis = dfx.index[dfx.eff_prob_sepsis > dfx.random_draw2]
        df.loc[idx_pph, 'la_pph'] = True
        df.loc[idx_sepsis, 'la_sepsis'] = True

        for individual_id in idx_pph:
            random = self.sim.rng.random_sample()
            if random < params['cfr_pn_pph']:
                    self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                          cause=' complication of miscarriage'),
                                            self.sim.date)

        for individual_id in idx_sepsis:
            random = self.sim.rng.random_sample()
            if random < params['cfr_pn_sepsis']:
                    self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                          cause='complication of miscarriage'),self.sim.date)


#  =============================================== RESET LABOUR STATUS =================================================

        # (commented out so can view effects at this point)
#            if df.at[individual_id, 'is_alive']:
#                df.at[individual_id, 'la_labour'] = "not_in_labour"


        #Todo: Health System interaction events? here?


class LabourDeathEvent (Event, IndividualScopeEventMixin):

    """handles death in labour"""

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # TODO: review and restructure as needed

        idx_eclamp = df.index[df.la_eclampsia & (df.due_date == self.sim.date)]
        idx_aph = df.index[df.la_aph & (df.due_date == self.sim.date)]
        idx_sepsis = df.index[df.la_sepsis & (df.due_date == self.sim.date)]
        idx_ur = df.index[df.la_uterine_rupture & (df.due_date == self.sim.date)]

        for individual_id in idx_eclamp:
            random = self.sim.rng.random_sample()
            if random < params['cfr_eclampsia']:
                df.at[individual_id, 'la_died_in_labour'] = True
                random = self.sim.rng.random_sample()
                if random < params['prob_still_birth_eclampsia_md']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True
            else:
                random = self.sim.rng.random_sample()
                if random < params['prob_still_birth_eclampsia']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True

        for individual_id in idx_aph:
            random = self.sim.rng.random_sample()
            if random < params['cfr_aph']:
                df.at[individual_id, 'la_died_in_labour'] = True
                random = self.sim.rng.random_sample()
                if random < params['prob_still_birth_aph_md']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True
            else:
                random = self.sim.rng.random_sample()
                if random < params['prob_still_birth_aph']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True

        for individual_id in idx_sepsis:
            random = self.sim.rng.random_sample()
            if random < params['cfr_sepsis']:
                df.at[individual_id, 'la_died_in_labour'] = True
                random = self.sim.rng.random_sample()
                if random < params['prob_still_birth_sepsis_md']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True
            else:
                random = self.sim.rng.random_sample()
                if random < params['prob_still_birth_sepsis']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True

        for individual_id in idx_ur:
            random = self.sim.rng.random_sample()
            if random < params['cfr_uterine_rupture']:
                df.at[individual_id, 'la_died_in_labour'] = True
                random = self.sim.rng.random_sample()
                if random < params['prob_still_birth_ur_md']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True
            else:
                random = self.sim.rng.random_sample()
                if random < params['prob_still_birth_ur']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True

            # Schedule death for women who die in labour

        maternal_death = df.index[(df.due_date == self.sim.date) & (df.la_labour == 'post_term_labour')
                                  & (df.la_died_in_labour == True)]

        for individual_id in maternal_death:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                  cause='labour'), self.sim.date)


class PostPartumDeathEvent (Event, IndividualScopeEventMixin):

    """handles death following labour"""

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        idx_eclamp = df.index[(df.due_date == self.sim.date - DateOffset(days=2)) & df.la_eclampsia]
        idx_pph = df.index[(df.due_date == self.sim.date - DateOffset(days=2)) & df.la_pph]
        idx_sepsis = df.index[(df.due_date == self.sim.date - DateOffset(days=2)) & df.la_sepsis]

        for individual_id in idx_eclamp:
            random = self.sim.rng.random_sample()
            if random < params['cfr_pn_eclampsia']:
                df.at[individual_id, 'la_died_in_labour'] = True

        for individual_id in idx_pph:
            random = self.sim.rng.random_sample()
            if random < params['cfr_pn_pph']:
                df.at[individual_id, 'la_died_in_labour'] = True

        for individual_id in idx_sepsis:
            random = self.sim.rng.random_sample()
            if random < params['cfr_pn_sepsis']:
                df.at[individual_id, 'la_died_in_labour'] = True

        maternal_death = df.index[(df.due_date == self.sim.date - DateOffset(days=2)) &
                                  (df.la_labour == 'preterm_labour') & (df.la_died_in_labour == True) &
                                  (df.is_alive == True)]

        for individual_id in maternal_death:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                  cause='postpartum labour'), self.sim.date)





class LabourLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Handles lifestyle logging"""
    def __init__(self, module):
        """schedule logging to repeat every 3 months
        """
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(days=self.repeat))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props

        logger.debug('%s|person_one|%s',
                          self.sim.date, df.loc[0].to_dict())
