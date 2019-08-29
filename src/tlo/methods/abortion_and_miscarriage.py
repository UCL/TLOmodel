import logging

import numpy as np
import pandas as pd

from pathlib import Path

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

from tlo.methods import demography, healthsystem, healthburden, antenatal_care, labour


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AbortionAndMiscarriage(Module):
    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    PARAMETERS = {
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
        'incidence_induced_abortion': Parameter(
            Types.REAL, 'incidence of induced abortion before 28 weeks gestation'),
        'prob_sa_pph': Parameter(
            Types.REAL, 'probability of a postpartum haemorrhage following a spontaneous abortion before 28 weeks'
                        ' gestation'),
        'prob_sa_sepsis': Parameter(
            Types.REAL, 'probability of a sepsis  following a spontaneous abortion before 28 weeks gestation'),
        'cfr_sepsis_miscarriage': Parameter(
            Types.REAL, 'case fatality rate for maternal sepsis following a miscarriage'),
        'cfr_haem_miscarriage':Parameter(
            Types.REAL, 'case fatality rate for haemorrhage following a miscarriage'),
    }

    PROPERTIES = {
        'am_total_miscarriages': Property(Types.INT, 'the number of miscarriages a woman has experienced'),
        'am_date_most_recent_miscarriage': Property(Types.DATE, 'the date this woman has last experienced spontaneous'
                                                                ' miscarriage'),
        'am_total_induced_abortion': Property(Types.INT, 'the number of induced abortions a woman has experienced'),
        'am_date_most_recent_abortion': Property(Types.DATE, 'the date this woman has last undergone an induced'
                                                             ' abortion'),
        'am_safety_of_abortion': Property(Types.CATEGORICAL, 'Null, Safe, Less Safe, Least Safe', categories=['null',
                                                                                    'safe', 'less_safe', 'least_safe']),

    }

    def read_parameters(self, data_folder):
        params = self.parameters

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_AbortionAndMiscarriage.xlsx',
                            sheet_name='parameter_values')
        dfd.set_index('parameter_name', inplace=True)

        params['prob_miscarriage'] = dfd.loc['prob_miscarriage', 'value']
        params['rr_miscarriage_prevmiscarriage'] = dfd.loc['rr_miscarriage_prevmiscarriage', 'value']
        params['rr_miscarriage_35'] = dfd.loc['rr_miscarriage_35', 'value']
        params['rr_miscarriage_3134'] = dfd.loc['rr_miscarriage_3134', 'value']
        params['rr_miscarriage_grav4'] = dfd.loc['rr_miscarriage_grav4', 'value']
        params['incidence_induced_abortion'] = dfd.loc['incidence_induced_abortion', 'value']
        params['prob_sa_pph'] = dfd.loc['prob_sa_pph', 'value']
        params['prob_sa_sepsis'] = dfd.loc['prob_sa_sepsis', 'value']
        params['cfr_sepsis_miscarriage'] = dfd.loc['cfr_sepsis_miscarriage', 'value']
        params['cfr_haem_miscarriage'] = dfd.loc['cfr_haem_miscarriage', 'value']

        # get the DALY weight that this module will use from the weight database (these codes are just random!)
    #   if 'HealthBurden' in self.sim.modules.keys():
    #        p['daly_wt_mild_sneezing'] = self.sim.modules['HealthBurden'].get_daly_weight(50)
    #        p['daly_wt_coughing'] = self.sim.modules['HealthBurden'].get_daly_weight(50)
    #        p['daly_wt_advanced'] = self.sim.modules['HealthBurden'].get_daly_weight(589)

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        df = population.props
        df.loc[df.sex == 'F', 'am_total_miscarriages'] = 0
        df.loc[df.sex == 'F', 'am_date_most_recent_miscarriage'] = pd.NaT
        df.loc[df.sex == 'F', 'am_total_induced_abortion'] = 0
        df.loc[df.sex == 'F', 'am_date_most_recent_abortion'] = pd.NaT
        df.loc[df.sex == 'F', 'am_safety_of_abortion'] = 'null'


        self.sim.modules['HealthSystem'].register_disease_module(self)

    def initialise_simulation(self, sim):

        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event
        event = InducedAbortionScheduler(self)
        sim.schedule_event(event, sim.date + DateOffset(months=1))

        # add an event to log to screen
        sim.schedule_event(AbortionAndMiscarriageLoggingEvent(self), sim.date + DateOffset(months=6))


    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """

        df = self.sim.population.props

        if df.at[child_id, 'sex'] == 'F':
            df.at[child_id, 'am_total_miscarriages'] = 0
            df.at[child_id, 'am_date_most_recent_miscarriage'] = pd.NaT
            df.at[child_id, 'am_total_induced_abortion'] = 0
            df.at[child_id, 'am_date_most_recent_abortion'] = pd.NaT
            df.at[child_id, 'am_safety_of_abortion'] = 'null'



    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is AbortionAndMiscarriage, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):
        pass

        logger.debug('This is Abortion and Miscarriage reporting my health values')

        df = self.sim.population.props
        params = self.parameters


class CheckIfNewlyPregnantWomanWillMiscarry(Event, IndividualScopeEventMixin):
    """This event checks if a woman who is newly pregnant will experience a miscarriage, and will record on what date
     this miscarriage has occured. Women who """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # todo: should we code in twins/triplets (risk factors for ptb, bleeding etc)- should be in DHS
        # First we identify if this woman has any risk factors for early pregnancy loss

        if (df.at[individual_id, 'am_total_miscarriages'] >= 1) & (df.at[individual_id, 'age_years'] <= 30) & \
            (df.at[individual_id, 'la_parity'] < 1 > 3):
            rf1 = params['rr_miscarriage_prevmiscarriage']
        else:
            rf1 = 1

        if (df.at[individual_id, 'am_total_miscarriages'] == 0) & (df.at[individual_id, 'age_years'] >= 35) & \
            (df.at[individual_id,'la_parity'] < 1 > 3):
            rf2 = params['rr_miscarriage_35']
        else:
            rf2 = 1

        if (df.at[individual_id, 'am_total_miscarriages'] == 0) & (df.at[individual_id, 'age_years'] >= 31) & \
            (df.at[individual_id, 'age_years'] <= 34) & (df.at[individual_id,'la_parity'] < 1 > 3):
            rf3 = params['rr_miscarriage_3134']
        else:
            rf3 = 1

        if (df.at[individual_id, 'am_total_miscarriages'] == 0) & (df.at[individual_id, 'age_years'] <= 30) & \
            (df.at[individual_id,'la_parity'] >= 1 <= 3):
            rf4 = params['rr_miscarriage_grav4']
        else:
            rf4 = 1

        # Next we multiply the baseline rate of miscarriage in the reference population who are absent of riskfactors
        # by the product of the relative rates for any risk factors this mother may have
        riskfactors = rf1 * rf2 * rf3 * rf4

        if riskfactors == 1:
            eff_prob_miscarriage = params['prob_miscarriage']
        else:
            eff_prob_miscarriage = riskfactors * params['prob_miscarriage']

        # Finally a random draw is used to determine if this woman will experience a miscarriage for this pregnancy
        random = self.sim.rng.random_sample(size=1)
        if random < eff_prob_miscarriage:

            # If a newly pregnant woman will miscarry her pregnancy, a random date within 24 weeks of conception is
            # generated, the date of miscarriage is scheduled for this day

            # (N.b malawi date of viability is 28 weeks, we using 24 as cut off to allow for very PTB.)

            random_draw = self.sim.rng.exponential(scale=0.5, size=1)  # todo: finalise scale (check correct)
            random_days = pd.to_timedelta(random_draw[0] * 168, unit='d')
            miscarriage_date = self.sim.date + random_days
            df.at[individual_id, 'am_date_most_recent_miscarriage'] = miscarriage_date

            self.sim.schedule_event(MiscarriageAndPostMiscarriageComplicationsEvent(self.module, individual_id,
                                                                                    cause='post miscarriage'),
                                    miscarriage_date)

            # And for women who do not have a miscarriage we move them to labour scheduler to determine at what
            # gestation they will go into labour
        else:
            assert df.at[individual_id, 'la_due_date_current_pregnancy'] != pd.NaT
            self.sim.schedule_event(labour.LabourScheduler(self.sim.modules['Labour'], individual_id, cause='pregnancy')
                                    , self.sim.date)


class MiscarriageAndPostMiscarriageComplicationsEvent(Event, IndividualScopeEventMixin):

    """applies probability of postpartum complications to women who have just experience a miscarriage """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # TODO: consider stage of pregnancy loss and its impact on likelihood of complications i.e retained product
        #  THIS SHOULD BE WEIGHTED TOWARDS LATER PREGNANCY LOSS
        # i.e. look up effects of incomplete miscarriage (apply incidence)

        df.at[individual_id, 'is_pregnant'] = False
        df.at[individual_id, 'am_total_miscarriages'] = +1
        df.at[individual_id, 'la_due_date_current_pregnancy'] = pd.NaT

        logger.info('This is MiscarriageAndPostMiscarriageComplicationsEvent, person %d has experienced a miscarriage '
                    'on date %s and is no longer pregnant',
                    individual_id, self.sim.date)

        # As with the other complication events here we determine if this woman will experience any complications
        # following her miscarriage
        riskfactors = 1  # rf1
        eff_prob_pph = riskfactors * params['prob_sa_pph']
        random = self.sim.rng.random_sample(size=1)
        if random < eff_prob_pph:
            df.at[individual_id, 'la_pph'] = True

        riskfactors = 1  # rf1
        eff_prob_pp_sepsis = riskfactors * params['prob_sa_sepsis']
        random = self.sim.rng.random_sample(size=1)
        if random < eff_prob_pp_sepsis:
                df.at[individual_id, 'la_sepsis'] = True

        # Currently if she has experienced any of these complications she is scheduled to pass through the DeathEvent
        # to determine if they are fatal
        if df.at[individual_id,'la_sepsis'] or df.at[individual_id, 'la_pph']:
            logger.info('This is MiscarriageAndPostMiscarriageComplicationsEvent scheduling a possible death for'
                     ' person %d after suffering complications of miscarriage', individual_id)
            self.sim.schedule_event(MiscarriageDeathEvent(self.module, individual_id, cause='miscarriage'),
                                    self.sim.date)


class MiscarriageDeathEvent (Event, IndividualScopeEventMixin):

    """handles death following miscarriage"""

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        if df.at[individual_id, 'la_pph']:
            random = self.sim.rng.random_sample(size=1)
            if random < params['cfr_haem_miscarriage']:
                df.at[individual_id, 'la_maternal_death'] = True
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause=' complication of miscarriage'),
                                        self.sim.date)

        if df.at[individual_id, 'la_sepsis']:
            random = self.sim.rng.random_sample(size=1)
            if random < params['cfr_sepsis_miscarriage']:
                df.at[individual_id, 'la_maternal_death'] = True
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause='complication of miscarriage'),
                                        self.sim.date)
                logger.info('This is MiscarriageDeathEvent scheduling a death for person %d on date %s who died '
                            'following complications of miscarriage', individual_id,
                             self.sim.date)

                logger.info('%s|maternal_death|%s', self.sim.date,
                        {
                            'age': df.at[individual_id, 'age_years'],
                            'person_id': individual_id
                        })


class InducedAbortionScheduler(RegularEvent, PopulationScopeEventMixin):
    """
    This event is occurring regularly at three monthly intervals and is responsible for induced abortion in pregnant
    women
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):

        df = population.props
        params= self.module.parameters

        # TODO: consider women who are scheduled to miscarry (doesnt matter if they miscarry  prior to abortion)

        # First we create an index of all the women who have become pregnant in the previous month
        pregnant_past_month = df.index[df.is_pregnant & df.is_alive & (df.ac_gestational_age <= 4) &
                                       (df.date_of_last_pregnancy > self.sim.start_date)]

        incidence_abortion = pd.Series(params['incidence_induced_abortion'], index=pregnant_past_month)

        random_draw = pd.Series(self.module.rng.random_sample(size=len(pregnant_past_month)),
                                index=df.index[df.is_pregnant & df.is_alive & (df.ac_gestational_age <= 4) &
                                               (df.date_of_last_pregnancy > self.sim.start_date)])

        dfx = pd.concat([incidence_abortion, random_draw], axis=1)
        dfx.columns = ['incidence_abortion', 'random_draw']
        idx_induced_abortion = dfx.index[dfx.incidence_abortion > dfx.random_draw]

        for person in idx_induced_abortion:
            event = AbortionAndPostAbortionComplicationsEvent(self.module, individual_id=person,
                                                              cause='induced abortion')
            random_draw = self.sim.rng.exponential(scale=0.5, size=1)  # TODO: COPIED FROM MC MAY NOT BE APPROPRIATE
            random_days = pd.to_timedelta(random_draw[0] * 168, unit='d')
            abortion_date = self.sim.date + random_days
            self.sim.schedule_event(event, abortion_date)

        # do we use a risk factor model, or maybe we just do married/unmarried/wanted/unwanted?


class AbortionAndPostAbortionComplicationsEvent(Event, IndividualScopeEventMixin):

    """applies probability of postpartum complications to women who have just experience an induced abortion """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        logger.info('%s|induced_abortion|%s', self.sim.date,
                    {
                        'age': df.at[individual_id, 'age_years'],
                        'person_id': individual_id
                    })

        # Todo: Event
        # todo: check pregnancy status

        if df.at[individual_id, 'is_pregnant']:
            df.at[individual_id, 'is_pregnant'] = False
            df.at[individual_id, ]



class HSI_AbortionAndMiscarriage_PresentsForPostAbortionCare(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This interaction manages the initial appointment for women presenting to hospital for post aboriton care for
    incomplete loss of pregnancy
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['MinorSurg'] = 1  # todo: medical management would require less

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        abortion_pkg_code = pd.unique(consumables.loc[consumables[
                                                       'Intervention_Pkg'] == 'Post-abortion case management',
                                                      'Intervention_Pkg_Code'])[0]
        the_cons_footprint = {
            'Intervention_Package_Code': [abortion_pkg_code],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'AbortionAndMiscarriage_PresentsForPostAbortionCare'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [0]
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):

        logger.debug('This is HSI_AbortionAndMiscarriage_PresentsForPostAbortionCare, a first appointment for person %d',
                     person_id)

        df = self.sim.population.props  # shortcut to the dataframe

        #  Treatment for: Incomplete abortion (medical/surgical), sepsis?, trauma, bleeding


class AbortionAndMiscarriageLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summmary of the numbers of people with respect to their 'mockitis status'
        """
        # run this event every month
        self.repeat = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

